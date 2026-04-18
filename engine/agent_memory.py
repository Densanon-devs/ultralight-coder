"""
Phase 14 Agent Memory — cross-session notes for the agent loop.

Stores per-project notes at:
    ~/.ultralight-coder/memory/<workspace-hash>/notes.md

Each entry is one Markdown bullet line, append-only, prefixed with an ISO-ish
timestamp so ordering is preserved. Loaded into the agent's system prompt at
run start so the model has continuity between sessions ("last time we were
working on X").

Plus the agent registers a `remember(note)` tool when memory is enabled so the
model can save a note mid-loop.

Zero servers, pure stdlib, local file only. No FAISS, no embeddings — these
notes are short and cheap, the model just reads the whole list.
"""

from __future__ import annotations

import hashlib
import logging
from datetime import datetime
from pathlib import Path
from typing import Optional

logger = logging.getLogger(__name__)

_DEFAULT_ROOT = Path.home() / ".ultralight-coder" / "memory"
_DEFAULT_LOAD_ENTRIES = 30
_MAX_NOTE_LEN = 500


def _project_id(workspace: Path) -> str:
    digest = hashlib.sha256(str(workspace.resolve()).encode("utf-8")).hexdigest()
    return digest[:12]


class AgentMemory:
    """Per-project append-only Markdown note store."""

    def __init__(self, workspace: Path, root: Optional[Path] = None) -> None:
        self.workspace = Path(workspace).resolve()
        self.root = Path(root) if root is not None else _DEFAULT_ROOT
        self.dir = self.root / _project_id(self.workspace)
        self.notes_path = self.dir / "notes.md"

    def remember(self, note: str) -> str:
        """Append a single note. Returns a confirmation string for the model."""
        note = (note or "").strip()
        if not note:
            return "(empty note ignored)"
        if len(note) > _MAX_NOTE_LEN:
            note = note[:_MAX_NOTE_LEN] + " ..."

        try:
            self.dir.mkdir(parents=True, exist_ok=True)
            if not self.notes_path.exists():
                header = (
                    f"# Ultralight Coder agent notes\n"
                    f"# Workspace: {self.workspace}\n"
                    f"# Project ID: {_project_id(self.workspace)}\n\n"
                )
                self.notes_path.write_text(header, encoding="utf-8")
            ts = datetime.now().strftime("%Y-%m-%d %H:%M")
            with self.notes_path.open("a", encoding="utf-8") as f:
                f.write(f"- {ts}: {note}\n")
        except OSError as exc:
            logger.warning("Failed to write agent memory: %s", exc)
            return f"(failed to save note: {exc})"
        return f"Saved note ({len(note)} chars)"

    def load(self, max_entries: int = _DEFAULT_LOAD_ENTRIES) -> str:
        """Return the most recent N notes as a Markdown block, or empty string."""
        if not self.notes_path.exists():
            return ""
        try:
            text = self.notes_path.read_text(encoding="utf-8")
        except OSError:
            return ""
        lines = [ln for ln in text.splitlines() if ln.startswith("- ")]
        if not lines:
            return ""
        recent = lines[-max_entries:]
        return "\n".join(recent)

    def clear(self) -> None:
        """Delete the notes file (the project dir stays)."""
        if self.notes_path.exists():
            self.notes_path.unlink()


# ── Smoke test ──────────────────────────────────────────────────

if __name__ == "__main__":
    import tempfile

    with tempfile.TemporaryDirectory() as tmp_root:
        with tempfile.TemporaryDirectory() as ws:
            mem = AgentMemory(workspace=Path(ws), root=Path(tmp_root))
            assert mem.load() == ""

            mem.remember("first note")
            mem.remember("second note")
            mem.remember("third note")
            loaded = mem.load()
            assert "first note" in loaded
            assert "third note" in loaded
            assert len([ln for ln in loaded.splitlines() if ln.startswith("- ")]) == 3

            # Two different workspaces get isolated files
            with tempfile.TemporaryDirectory() as ws2:
                mem2 = AgentMemory(workspace=Path(ws2), root=Path(tmp_root))
                assert mem2.load() == ""
                mem2.remember("other project note")
                assert "other project" not in mem.load()
                assert "other project" in mem2.load()

            # Empty notes ignored
            assert "ignored" in mem.remember("").lower()
            assert "ignored" in mem.remember("   ").lower()

            # Long notes truncated
            mem.remember("x" * 1000)
            assert "..." in mem.load()

            # Max entries cap
            for i in range(50):
                mem.remember(f"note {i}")
            recent = mem.load(max_entries=10)
            assert len(recent.splitlines()) == 10
            assert "note 49" in recent
            assert "note 30" not in recent

            # Clear works
            mem.clear()
            assert mem.load() == ""

            # Different workspace -> different project_id
            from engine.agent_memory import _project_id
            assert _project_id(Path(ws)) != _project_id(Path(tmp_root))

    print("OK: engine/agent_memory.py smoke test passed")
