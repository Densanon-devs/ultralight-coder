"""
Trajectory distiller — convert successful Agent runs into reusable artifacts.

The augmentor system today is hand-curated YAML pairs (query -> solution code).
That works great for codegen but doesn't capture *agentic* patterns: which tools
to call in which order, when to grep first, when to read_file before edit_file,
when to escalate to write_file vs apply_patch.

A Trajectory captures one successful agent run as a structured artifact:
    goal          — the user's task
    tool_calls    — the sequence of (name, abridged-args) the agent made
    files_touched — sorted set of files modified
    final_answer  — the agent's final summary
    intent + lang — coarse tags for retrieval

Failed runs are written to a separate review queue so a human can decide
whether the failure mode is worth a new augmentor or a harness fix.

This module is intentionally dependency-light: stdlib + pyyaml only. No
embeddings, no FAISS — keyword Jaccard is enough to validate the pipeline
end-to-end. A future pass can swap in the existing sentence-transformer.
"""
from __future__ import annotations

import hashlib
import re
import time
from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Any, Iterable

import yaml


# ── Detection heuristics ────────────────────────────────────────────

_INTENT_PATTERNS = [
    ("codegen",   r"\b(add|create|build|implement|write|generate|scaffold|new)\b"),
    ("refactor",  r"\b(refactor|rename|move|extract|inline|reorganize|reorganise|restructure)\b"),
    ("debug",     r"\b(fix|bug|error|broken|fail|crash|wrong|incorrect|issue)\b"),
    ("explain",   r"\b(explain|describe|summarize|summarise|what does|how does|why does)\b"),
    ("test",      r"\b(test|pytest|unittest|coverage|assert)\b"),
    ("review",    r"\b(review|audit|check|inspect|critique)\b"),
    ("docs",      r"\b(document|docstring|readme|comment|api docs)\b"),
]

_LANG_PATTERNS = [
    ("python",     [r"\.py\b", r"\bpython\b", r"\bpytest\b", r"\bdef \w+\(", r"\bdataclass\b", r"\bargparse\b"]),
    ("javascript", [r"\.js\b", r"\bjavascript\b", r"\bnpm\b", r"\bnode\b", r"\bconsole\.log\b"]),
    ("typescript", [r"\.ts\b", r"\.tsx\b", r"\btypescript\b", r"\binterface \w+", r"\btype \w+ =\b"]),
    ("rust",       [r"\.rs\b", r"\brust\b", r"\bcargo\b", r"\bfn \w+\(", r"\bimpl\b"]),
    ("go",         [r"\.go\b", r"\bgolang\b", r"\bgo build\b", r"\bfunc \w+\("]),
    ("html",       [r"\.html?\b", r"\bhtml\b", r"<div", r"<html"]),
    ("yaml",       [r"\.ya?ml\b", r"\byaml\b"]),
    ("json",       [r"\.json\b", r"\bjson\b"]),
    ("bash",       [r"\.sh\b", r"\bbash\b", r"#!/bin/"]),
]


def detect_intent(goal: str) -> str:
    g = goal.lower()
    for label, pat in _INTENT_PATTERNS:
        if re.search(pat, g):
            return label
    return "other"


def detect_language(goal: str, files: Iterable[str] = ()) -> str:
    blob = " ".join([goal, *files]).lower()
    for label, pats in _LANG_PATTERNS:
        for p in pats:
            if re.search(p, blob):
                return label
    return "other"


# ── Data model ──────────────────────────────────────────────────────

@dataclass
class Trajectory:
    goal: str
    success: bool
    iterations: int
    stop_reason: str
    wall_time: float
    final_answer: str
    tool_calls: list[dict[str, Any]] = field(default_factory=list)
    files_touched: list[str] = field(default_factory=list)
    intent: str = "other"
    language: str = "other"
    captured_at: float = field(default_factory=time.time)

    def signature(self) -> str:
        """Stable short identifier for filename use."""
        h = hashlib.sha256()
        h.update(self.goal.encode("utf-8"))
        h.update(str(self.captured_at).encode("utf-8"))
        return h.hexdigest()[:12]

    def to_yaml_dict(self) -> dict:
        return {
            "goal": self.goal,
            "success": self.success,
            "iterations": self.iterations,
            "stop_reason": self.stop_reason,
            "wall_time": round(self.wall_time, 2),
            "intent": self.intent,
            "language": self.language,
            "files_touched": list(self.files_touched),
            "tool_calls": list(self.tool_calls),
            "final_answer": self.final_answer,
            "captured_at": self.captured_at,
        }


# ── Distillation ────────────────────────────────────────────────────

_FILE_TOOLS = {"write_file", "edit_file", "insert_at_line", "edit_html", "apply_patch"}
_ARG_PREVIEW_LIMIT = 120


def _abridge(args: dict) -> dict:
    """Compact a tool arguments dict for storage — long content fields collapse
    to length markers so the YAML stays human-scannable."""
    out = {}
    for k, v in args.items():
        if isinstance(v, str) and len(v) > _ARG_PREVIEW_LIMIT:
            out[k] = f"<{len(v)} chars: {v[:60]!r}...>"
        elif isinstance(v, list):
            joined_len = sum(len(s) for s in v if isinstance(s, str))
            if joined_len > _ARG_PREVIEW_LIMIT:
                out[k] = f"<list[{len(v)}] {joined_len} chars total>"
            else:
                out[k] = v
        else:
            out[k] = v
    return out


def distill(result: Any, goal: str) -> Trajectory:
    """Convert an AgentResult into a Trajectory.

    Accepts duck-typed inputs so this module doesn't need to import the full
    Agent class — anything with .final_answer, .iterations, .stop_reason,
    .wall_time, .tool_calls works.
    """
    tcs = getattr(result, "tool_calls", []) or []
    summary: list[dict[str, Any]] = []
    files: list[str] = []
    for tc in tcs:
        name = getattr(tc, "name", None) or (tc.get("name") if isinstance(tc, dict) else "?")
        args = getattr(tc, "arguments", None) or (tc.get("arguments") if isinstance(tc, dict) else {})
        if not isinstance(args, dict):
            args = {}
        summary.append({"name": name, "args": _abridge(args)})
        if name in _FILE_TOOLS:
            path = args.get("path")
            if path and path not in files:
                files.append(path)

    success = (getattr(result, "stop_reason", "") == "answered")
    return Trajectory(
        goal=goal,
        success=success,
        iterations=int(getattr(result, "iterations", 0)),
        stop_reason=str(getattr(result, "stop_reason", "")),
        wall_time=float(getattr(result, "wall_time", 0.0)),
        final_answer=str(getattr(result, "final_answer", "") or ""),
        tool_calls=summary,
        files_touched=files,
        intent=detect_intent(goal),
        language=detect_language(goal, files),
    )


# ── Persistence ─────────────────────────────────────────────────────

def _trajectory_dir(root: Path) -> Path:
    return Path(root) / "data" / "trajectory_examples"


def _review_dir(root: Path) -> Path:
    return Path(root) / "data" / "trajectory_review"


def save(traj: Trajectory, repo_root: Path | str) -> Path:
    """Write the trajectory to disk. Successful runs go to
    data/trajectory_examples/, failed runs go to data/trajectory_review/.
    Returns the path written."""
    repo_root = Path(repo_root)
    target_dir = _trajectory_dir(repo_root) if traj.success else _review_dir(repo_root)
    target_dir.mkdir(parents=True, exist_ok=True)
    fname = f"{traj.intent}__{traj.language}__{traj.signature()}.yaml"
    path = target_dir / fname
    path.write_text(
        yaml.safe_dump(traj.to_yaml_dict(), sort_keys=False, allow_unicode=True),
        encoding="utf-8",
    )
    return path


def load(path: Path | str) -> Trajectory:
    data = yaml.safe_load(Path(path).read_text(encoding="utf-8"))
    return Trajectory(
        goal=data["goal"],
        success=bool(data["success"]),
        iterations=int(data["iterations"]),
        stop_reason=str(data["stop_reason"]),
        wall_time=float(data["wall_time"]),
        final_answer=str(data["final_answer"]),
        tool_calls=list(data.get("tool_calls", [])),
        files_touched=list(data.get("files_touched", [])),
        intent=str(data.get("intent", "other")),
        language=str(data.get("language", "other")),
        captured_at=float(data.get("captured_at", 0.0)),
    )


# ── Retrieval ───────────────────────────────────────────────────────

_STOPWORDS = frozenset({
    "the", "a", "an", "and", "or", "for", "to", "of", "in", "on", "with",
    "is", "are", "be", "this", "that", "it", "can", "do", "does", "did",
    "i", "you", "we", "should", "would", "could", "please", "make", "use",
})


def _tokens(text: str) -> set[str]:
    return {t for t in re.findall(r"[a-zA-Z_][a-zA-Z0-9_]+", text.lower())
            if t not in _STOPWORDS and len(t) > 2}


def _jaccard(a: set[str], b: set[str]) -> float:
    if not a or not b:
        return 0.0
    return len(a & b) / len(a | b)


def match(query: str, repo_root: Path | str, top_k: int = 3) -> list[tuple[float, Trajectory]]:
    """Keyword-match query against stored successful trajectories.

    Returns up to top_k (score, trajectory) tuples, highest score first.
    Score is Jaccard similarity over alphanumeric tokens, scaled +0.1 if
    intents match and +0.1 if languages match. Range roughly 0.0-1.2.
    """
    repo_root = Path(repo_root)
    examples_dir = _trajectory_dir(repo_root)
    if not examples_dir.exists():
        return []

    q_tokens = _tokens(query)
    q_intent = detect_intent(query)
    q_lang = detect_language(query)

    scored: list[tuple[float, Trajectory]] = []
    for f in examples_dir.glob("*.yaml"):
        try:
            traj = load(f)
        except Exception:
            continue
        score = _jaccard(q_tokens, _tokens(traj.goal))
        if traj.intent == q_intent and q_intent != "other":
            score += 0.1
        if traj.language == q_lang and q_lang != "other":
            score += 0.1
        if score > 0:
            scored.append((score, traj))

    scored.sort(key=lambda x: x[0], reverse=True)
    return scored[:top_k]


# ── Smoke test ──────────────────────────────────────────────────────

if __name__ == "__main__":
    import tempfile

    class _StubResult:
        final_answer = "Added the divide function."
        iterations = 4
        stop_reason = "answered"
        wall_time = 12.3
        tool_calls = [
            type("TC", (), {"name": "read_file", "arguments": {"path": "calc.py"}})(),
            type("TC", (), {"name": "edit_file", "arguments": {
                "path": "calc.py",
                "old_string": "def add(a, b):",
                "new_string": "def divide(a, b):\n    return a / b\n\ndef add(a, b):",
            }})(),
            type("TC", (), {"name": "run_tests", "arguments": {}})(),
        ]

    traj = distill(_StubResult(), "Add a divide function to calc.py")
    print(f"intent={traj.intent} lang={traj.language} files={traj.files_touched}")
    tmp = Path(tempfile.mkdtemp())
    p = save(traj, tmp)
    print(f"saved {p}")
    hits = match("add subtract function to calc.py", tmp)
    for score, t in hits:
        print(f"  {score:.2f}  {t.goal}")
