"""Per-call audit log for engagement workspaces.

When ulcagent runs inside an engagement directory (one with an `audit/`
subdirectory), every tool call is appended to `audit/<YYYY-MM-DD>.jsonl`.
Each line is one JSON object: timestamp, tool name, args, result preview,
error. Used for client deliverables ("here's exactly what we ran during
the engagement window") and as a defensive paper trail.

Auto-discovers the audit directory: if `<workspace>/audit/` exists, log
there. If not, no logging — preserves the original zero-server behavior
for ad-hoc uses.

Thread-safe (uses a single lock around the file write).
"""
from __future__ import annotations

import json
import logging
import threading
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Optional

logger = logging.getLogger(__name__)

AUDIT_SUBDIR = "audit"
PREVIEW_CHARS = 300  # cap each result preview to keep audit lines compact


class AuditLog:
    """Thread-safe append-only JSONL log for tool calls.

    Instantiate with the engagement workspace path. Log files are written
    under `<workspace>/audit/<YYYY-MM-DD>.jsonl` and rotate daily by date.
    """

    def __init__(self, workspace: Path | str):
        self.workspace = Path(workspace)
        self.log_dir = self.workspace / AUDIT_SUBDIR
        self._lock = threading.Lock()

    @classmethod
    def for_workspace(cls, workspace: Path | str) -> Optional["AuditLog"]:
        """Construct only if `<workspace>/audit/` already exists.

        Returns None when the workspace isn't an engagement (no audit dir).
        Use this from ulcagent so audit logging auto-engages inside an
        engagement scaffold and stays off elsewhere.
        """
        ws = Path(workspace)
        audit_dir = ws / AUDIT_SUBDIR
        if not audit_dir.exists() or not audit_dir.is_dir():
            return None
        return cls(ws)

    def _path_for_today(self) -> Path:
        date = datetime.now(timezone.utc).strftime("%Y-%m-%d")
        return self.log_dir / f"{date}.jsonl"

    def _truncate(self, value: Any) -> Any:
        """Cap string values for the on-disk preview. Non-strings pass through."""
        if isinstance(value, str) and len(value) > PREVIEW_CHARS:
            return value[:PREVIEW_CHARS] + f"...[{len(value) - PREVIEW_CHARS} more chars]"
        if isinstance(value, dict):
            return {k: self._truncate(v) for k, v in value.items()}
        if isinstance(value, list):
            return [self._truncate(v) for v in value[:50]]
        return value

    def log_call(
        self,
        tool: str,
        args: Optional[dict] = None,
        result: Optional[str] = None,
        error: Optional[str] = None,
        extra: Optional[dict] = None,
    ) -> None:
        """Append a single tool-call entry. Never raises (logging failure is
        not allowed to break the agent loop)."""
        try:
            entry = {
                "ts": datetime.now(timezone.utc).isoformat(timespec="seconds"),
                "tool": tool,
                "args": self._truncate(args or {}),
                "result_chars": len(result) if isinstance(result, str) else 0,
                "result_preview": self._truncate(result) if result else None,
                "error": error,
            }
            if extra:
                entry["extra"] = self._truncate(extra)
            with self._lock:
                self.log_dir.mkdir(parents=True, exist_ok=True)
                with self._path_for_today().open("a", encoding="utf-8") as f:
                    f.write(json.dumps(entry, default=str) + "\n")
        except Exception as exc:
            # Audit failure must never block the agent. Log via stderr-only.
            logger.warning(f"audit_log.log_call raised (suppressing): {exc!r}")
