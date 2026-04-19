"""
Phase 14 Correction Memory — structured pattern memory for learning from
user corrections.

When the user says "no, do it THIS way" or "don't use X, use Y instead",
the agent can extract a correction pattern and store it. On subsequent runs,
matching corrections are injected into the system prompt so the model avoids
repeating the same mistake.

Storage: ~/.ulcagent_corrections.json (flat JSON list, max 30 entries, FIFO).

Matching: fuzzy keyword overlap between the current goal and stored correction
contexts. No embeddings, no FAISS — corrections are short and the list is
small, so linear scan with keyword intersection is perfectly fine.

Zero servers, pure stdlib, local file only.
"""

from __future__ import annotations

import json
import logging
from datetime import datetime
from pathlib import Path
from typing import List, Optional

logger = logging.getLogger(__name__)

_DEFAULT_PATH = Path.home() / ".ulcagent_corrections.json"
_MAX_CORRECTIONS = 30


def _tokenize(text: str) -> set[str]:
    """Extract lowercase keyword tokens from text.

    Strips common noise words so matching focuses on meaningful terms.
    """
    noise = {
        "a", "an", "the", "is", "it", "in", "on", "to", "of", "for",
        "and", "or", "not", "but", "with", "this", "that", "do", "don't",
        "use", "i", "my", "me", "be", "no", "yes", "was", "are", "were",
        "been", "have", "has", "had", "will", "can", "should", "would",
        "could", "at", "by", "from", "as", "if", "when", "so", "up",
    }
    tokens = set()
    for word in text.lower().split():
        # Strip punctuation edges
        cleaned = word.strip(".,;:!?\"'`()[]{}#<>")
        if cleaned and cleaned not in noise and len(cleaned) > 1:
            tokens.add(cleaned)
    return tokens


def _relevance_score(goal_tokens: set[str], context_tokens: set[str]) -> float:
    """Compute overlap score between goal keywords and correction context.

    Returns a value between 0.0 and 1.0. Higher means more relevant.
    Uses Jaccard-like overlap: |intersection| / |context_tokens| so a short,
    specific correction can match even when the goal is long.
    """
    if not context_tokens:
        return 0.0
    overlap = goal_tokens & context_tokens
    if not overlap:
        return 0.0
    # Normalize by context size — a correction about "pytest fixtures" should
    # match any goal mentioning both words, regardless of goal length.
    return len(overlap) / len(context_tokens)


class CorrectionMemory:
    """Persistent store of user correction patterns.

    Each correction is:
        {"context": str, "wrong": str, "correct": str, "created": str}

    At most _MAX_CORRECTIONS entries are kept. When full, the oldest entry
    is evicted (FIFO).
    """

    def __init__(self, path: Optional[str | Path] = None) -> None:
        self.path = Path(path) if path is not None else _DEFAULT_PATH
        self._corrections: List[dict] = []
        self._load()

    # ── Persistence ────────────────────────────────────────────────

    def _load(self) -> None:
        """Load corrections from disk. Silently starts empty on any error."""
        if not self.path.exists():
            self._corrections = []
            return
        try:
            text = self.path.read_text(encoding="utf-8")
            data = json.loads(text)
            if isinstance(data, list):
                self._corrections = data[:_MAX_CORRECTIONS]
            else:
                logger.warning("Correction file is not a list, resetting.")
                self._corrections = []
        except (OSError, json.JSONDecodeError, ValueError) as exc:
            logger.warning("Failed to load corrections from %s: %s", self.path, exc)
            self._corrections = []

    def _save(self) -> None:
        """Write corrections to disk."""
        try:
            self.path.parent.mkdir(parents=True, exist_ok=True)
            self.path.write_text(
                json.dumps(self._corrections, indent=2, ensure_ascii=False),
                encoding="utf-8",
            )
        except OSError as exc:
            logger.warning("Failed to save corrections to %s: %s", self.path, exc)

    # ── Public API ─────────────────────────────────────────────────

    def add(self, context: str, wrong: str, correct: str) -> str:
        """Add a correction pattern. Returns a confirmation message."""
        context = (context or "").strip()
        wrong = (wrong or "").strip()
        correct = (correct or "").strip()

        if not context or not correct:
            return "Correction not saved: context and correct answer are required."

        entry = {
            "context": context,
            "wrong": wrong,
            "correct": correct,
            "created": datetime.now().isoformat(timespec="seconds"),
        }

        self._corrections.append(entry)

        # FIFO eviction when over the cap
        while len(self._corrections) > _MAX_CORRECTIONS:
            self._corrections.pop(0)

        self._save()
        return f"Correction saved ({len(self._corrections)}/{_MAX_CORRECTIONS} slots used)."

    def get_relevant(self, goal: str, max_results: int = 3) -> list[dict]:
        """Return the top matching corrections for a goal, ranked by relevance.

        Only corrections with a relevance score >= 0.4 are returned.
        """
        if not goal or not self._corrections:
            return []

        goal_tokens = _tokenize(goal)
        if not goal_tokens:
            return []

        scored: list[tuple[float, dict]] = []
        for entry in self._corrections:
            ctx_tokens = _tokenize(entry.get("context", ""))
            score = _relevance_score(goal_tokens, ctx_tokens)
            if score >= 0.4:
                scored.append((score, entry))

        # Sort by score descending, then by creation time (newest first)
        scored.sort(key=lambda x: x[0], reverse=True)
        return [entry for _, entry in scored[:max_results]]

    def format_for_prompt(self, goal: str) -> str:
        """Format matching corrections as a block for system prompt injection.

        Returns empty string if no corrections match.
        """
        relevant = self.get_relevant(goal)
        if not relevant:
            return ""

        lines = ["# User Corrections (follow these preferences)", ""]
        for i, entry in enumerate(relevant, 1):
            lines.append(f"{i}. Context: {entry['context']}")
            if entry.get("wrong"):
                lines.append(f"   WRONG: {entry['wrong']}")
            lines.append(f"   CORRECT: {entry['correct']}")
            lines.append("")

        return "\n".join(lines)

    def list_all(self) -> list[dict]:
        """Return all stored corrections."""
        return list(self._corrections)

    def delete(self, index: int) -> str:
        """Delete a correction by index (0-based). Returns a confirmation."""
        if index < 0 or index >= len(self._corrections):
            return f"Invalid index {index}. Valid range: 0-{len(self._corrections) - 1}."
        removed = self._corrections.pop(index)
        self._save()
        ctx_preview = removed.get("context", "")[:50]
        return f"Deleted correction #{index}: \"{ctx_preview}...\""

    def clear(self) -> str:
        """Delete all corrections. Returns a confirmation."""
        count = len(self._corrections)
        self._corrections = []
        self._save()
        return f"Cleared {count} correction(s)."


# ── /learn command handler ─────────────────────────────────────────

def _handle_learn(args: str, correction_memory: CorrectionMemory) -> None:
    """Handle the /learn slash command.

    Usage:
        /learn          — interactive: asks for context, wrong, and correct
        /learn list     — show all stored corrections
        /learn clear    — clear all corrections
        /learn delete N — delete correction at index N
    """
    args = (args or "").strip()

    # /learn list
    if args.lower() == "list":
        corrections = correction_memory.list_all()
        if not corrections:
            print("No corrections stored.")
            return
        print(f"\nStored corrections ({len(corrections)}/{_MAX_CORRECTIONS}):\n")
        for i, entry in enumerate(corrections):
            created = entry.get("created", "unknown")
            print(f"  [{i}] Context: {entry.get('context', '')}")
            if entry.get("wrong"):
                print(f"       Wrong:   {entry['wrong']}")
            print(f"       Correct: {entry.get('correct', '')}")
            print(f"       Created: {created}")
            print()
        return

    # /learn clear
    if args.lower() == "clear":
        result = correction_memory.clear()
        print(result)
        return

    # /learn delete N
    if args.lower().startswith("delete"):
        parts = args.split(maxsplit=1)
        if len(parts) < 2:
            print("Usage: /learn delete <index>")
            return
        try:
            idx = int(parts[1])
        except ValueError:
            print(f"Invalid index: {parts[1]!r}. Must be an integer.")
            return
        result = correction_memory.delete(idx)
        print(result)
        return

    # /learn — interactive mode
    print("Recording a correction. What was the context / situation?")
    try:
        context = input("  Context: ").strip()
        if not context:
            print("Cancelled (empty context).")
            return
        wrong = input("  What was wrong? (optional, press Enter to skip): ").strip()
        correct = input("  What's correct? ").strip()
        if not correct:
            print("Cancelled (empty correction).")
            return
    except (EOFError, KeyboardInterrupt):
        print("\nCancelled.")
        return

    result = correction_memory.add(context, wrong, correct)
    print(result)


# ── Smoke test ──────────────────────────────────────────────────

if __name__ == "__main__":
    import tempfile

    with tempfile.TemporaryDirectory() as tmp:
        path = Path(tmp) / "test_corrections.json"

        # Fresh start — empty
        cm = CorrectionMemory(path=path)
        assert cm.list_all() == []

        # Add a correction
        msg = cm.add("writing pytest fixtures", "using unittest.mock", "use pytest monkeypatch")
        assert "saved" in msg.lower()
        assert len(cm.list_all()) == 1

        # Verify persistence — reload from disk
        cm2 = CorrectionMemory(path=path)
        assert len(cm2.list_all()) == 1
        assert cm2.list_all()[0]["context"] == "writing pytest fixtures"

        # Relevance matching
        results = cm.get_relevant("help me write pytest fixtures for my app")
        assert len(results) == 1
        assert results[0]["correct"] == "use pytest monkeypatch"

        # No match for unrelated goal
        results = cm.get_relevant("deploy the app to production")
        assert len(results) == 0

        # Format for prompt
        block = cm.format_for_prompt("writing pytest fixtures")
        assert "User Corrections" in block
        assert "monkeypatch" in block

        # Empty goal returns empty string
        assert cm.format_for_prompt("") == ""

        # Add multiple corrections
        cm.add("CSS styling", "inline styles", "use Tailwind classes")
        cm.add("database queries", "raw SQL strings", "use parameterized queries")
        assert len(cm.list_all()) == 3

        # Delete by index
        msg = cm.delete(1)
        assert "Deleted" in msg
        assert len(cm.list_all()) == 2

        # Delete invalid index
        msg = cm.delete(99)
        assert "Invalid" in msg

        # Clear
        msg = cm.clear()
        assert "Cleared 2" in msg
        assert len(cm.list_all()) == 0

        # FIFO eviction at max capacity
        for i in range(_MAX_CORRECTIONS + 5):
            cm.add(f"context {i}", f"wrong {i}", f"correct {i}")
        assert len(cm.list_all()) == _MAX_CORRECTIONS
        # Oldest should have been evicted (0-4 gone, 5 is first)
        assert cm.list_all()[0]["context"] == "context 5"

        # Empty/missing fields handled
        msg = cm.add("", "", "something")
        assert "required" in msg.lower()
        msg = cm.add("context", "", "")
        assert "required" in msg.lower()

        # Tokenizer handles punctuation and noise words
        tokens = _tokenize("Don't use the old API! It's broken.")
        assert "old" in tokens
        assert "api" in tokens
        assert "broken" in tokens
        assert "the" not in tokens
        assert "don't" not in tokens

        # Relevance scoring
        assert _relevance_score({"pytest", "fixtures"}, {"pytest", "fixtures"}) == 1.0
        assert _relevance_score({"pytest", "fixtures"}, set()) == 0.0
        assert _relevance_score(set(), {"pytest"}) == 0.0

    print("OK: engine/correction_memory.py smoke test passed")
