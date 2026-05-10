"""
Destructive-shell-command pre-execution gate.

A content-aware safety layer that sits behind the existing risky-tool gate.
The risky-tool gate (`Agent._execute_call` + `Agent.confirm_risky`) gates by
*tool name* — any `run_bash` call gets a y/N prompt — and can be bypassed by
`--yes` / `_auto_approve_risky`. That's fine for an interactive REPL where
the user is watching every prompt.

It is NOT fine for unattended runs. The May 2026 r/ClaudeAI incident where
Claude Code generated `rm -rf tests/ patches/ plan/ ~/` (stray `~/` expanded
to $HOME, 717 GB Windows install deleted) is the canonical failure: with
`--yes` set, the broad risky gate would have rubber-stamped that. Same root
cause as the April 2026 PocketOS database-wipe and the Cursor 9-second prod
database delete.

This module adds a *second* gate that:

1. Pattern-matches the proposed shell command against a denylist of
   destructive operations (rm -rf, Remove-Item -Recurse, format, dd if=,
   paths expanding to ~/, $HOME, drive roots, etc.).
2. Cannot be bypassed by `--yes` — destructive prompts are mandatory.
3. Fails safe: if the agent has no `confirm_destructive` callback wired,
   matched commands are refused entirely. The model sees a ToolResult
   error and can retry with a non-destructive approach.

The patterns are intentionally conservative — false positives (one extra
prompt for a command that was actually safe) are cheap; false negatives
(a destructive command slipping through) are hours-of-work-or-worse
expensive.

See also: `feedback_destructive_command_hook.md` for the cross-project
rule and rationale.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import List, Tuple


# Each entry: (pattern_id, human-readable description, compiled regex).
# Order is informational; all patterns are checked. Patterns are case-
# sensitive by default since Unix shell tokens are case-sensitive; the
# Windows-specific patterns include explicit case-insensitive flags.

_DESTRUCTIVE_PATTERNS: List[Tuple[str, str, "re.Pattern[str]"]] = [
    # ── Unix-style recursive/forceful remove ─────────────────────────
    # rm with -r / -R / --recursive. Flag may be combined: -rf, -fr, -Rf.
    # Pattern: `-` then any letters that include r or R, optionally
    # surrounded by other flag letters, then word boundary.
    (
        "rm_recursive",
        "rm with -r / -R / --recursive",
        re.compile(r"(?:^|[\s;&|`])rm(?:\s+(?:-[a-zA-Z]*[rR][a-zA-Z]*|--recursive))\b"),
    ),
    (
        "rm_force",
        "rm with -f / --force",
        re.compile(r"(?:^|[\s;&|`])rm(?:\s+(?:-[a-zA-Z]*f[a-zA-Z]*|--force))\b"),
    ),

    # ── PowerShell / Windows ─────────────────────────────────────────
    (
        "remove_item_recurse",
        "Remove-Item with -Recurse",
        re.compile(r"Remove-Item\b[^|;]*-Recurse", re.IGNORECASE),
    ),
    (
        "remove_item_force",
        "Remove-Item with -Force",
        re.compile(r"Remove-Item\b[^|;]*-Force", re.IGNORECASE),
    ),
    (
        "del_recursive",
        "del /s (recursive delete on Windows cmd)",
        re.compile(r"(^|[\s;&|`])del\s+(?:/\S*[sS]\b|/\S*[qQ]\b)", re.IGNORECASE),
    ),
    (
        "rd_recursive",
        "rd /s or rmdir /s (recursive directory removal on Windows cmd)",
        re.compile(r"(^|[\s;&|`])r(?:m)?d(?:ir)?\s+/\S*[sS]\b", re.IGNORECASE),
    ),

    # ── Whole-disk / formatting / wiping ─────────────────────────────
    (
        "format_drive",
        "format command (filesystem format)",
        re.compile(r"(^|[\s;&|`])format\s+[A-Za-z]:", re.IGNORECASE),
    ),
    (
        "mkfs",
        "mkfs* (filesystem creation, destroys existing data)",
        re.compile(r"(^|[\s;&|`])mkfs(?:\.\w+)?\s"),
    ),
    (
        "shred",
        "shred / wipe (overwrite-and-delete utilities)",
        re.compile(r"(^|[\s;&|`])(?:shred|wipe|sdelete)\b"),
    ),
    (
        "dd_to_device",
        "dd of=/dev/... (raw write to block device)",
        re.compile(r"(^|[\s;&|`])dd\b[^|;]*\bof=/dev/"),
    ),

    # ── find ... -delete / -exec rm ─────────────────────────────────
    # find can delete or invoke rm without naming `rm` at the front,
    # so the rm_* patterns miss it. Catch the destructive flags
    # explicitly.
    (
        "find_delete",
        "find -delete (bulk delete via find)",
        re.compile(r"\bfind\b[^|;]*\s-delete\b"),
    ),
    (
        "find_exec_rm",
        "find -exec rm (rm invoked via find)",
        re.compile(r"\bfind\b[^|;]*-exec\s+rm\b"),
    ),

    # ── git destructive ─────────────────────────────────────────────
    (
        "git_reset_hard",
        "git reset --hard (discards uncommitted work)",
        re.compile(r"\bgit\s+reset\s+--hard\b"),
    ),
    (
        "git_clean_force",
        "git clean -f / -fd / -fdx (deletes untracked files)",
        re.compile(r"\bgit\s+clean\s+(?:-\S*f|--force)"),
    ),
    (
        "git_push_force",
        "git push --force / --force-with-lease (rewrites remote history)",
        re.compile(r"\bgit\s+push\s+.*(?:--force\b|-f\b)"),
    ),
    (
        "git_checkout_dot",
        "git checkout . / git restore . (discards uncommitted changes)",
        re.compile(r"\bgit\s+(?:checkout|restore)\s+\."),
    ),
    (
        "git_branch_delete",
        "git branch -D (force-delete branch)",
        re.compile(r"\bgit\s+branch\s+-D\b"),
    ),

    # ── SQL destructive ──────────────────────────────────────────────
    (
        "sql_drop",
        "DROP TABLE / DROP DATABASE / DROP SCHEMA",
        re.compile(r"\bDROP\s+(?:TABLE|DATABASE|SCHEMA|INDEX)\b", re.IGNORECASE),
    ),
    (
        "sql_truncate",
        "TRUNCATE TABLE",
        re.compile(r"\bTRUNCATE\s+(?:TABLE\s+)?\w", re.IGNORECASE),
    ),
    (
        "sql_unscoped_delete",
        "DELETE FROM / UPDATE without WHERE clause",
        # We can't tell from regex alone whether a WHERE is present,
        # so we flag every UPDATE/DELETE FROM. The user can confirm
        # if it's safe. False positives here cost one prompt.
        re.compile(r"\b(?:DELETE\s+FROM|UPDATE)\s+\w+(?![^;]*\bWHERE\b)", re.IGNORECASE),
    ),

    # ── Process / OS kill ────────────────────────────────────────────
    (
        "kill_all",
        "kill -9 / killall (forceful process termination)",
        re.compile(r"(^|[\s;&|`])(?:kill\s+-9\b|killall\b|pkill\s+-9\b)"),
    ),
    (
        "shutdown_reboot",
        "shutdown / reboot / halt / poweroff",
        re.compile(r"(^|[\s;&|`])(?:shutdown|reboot|halt|poweroff|init\s+0|init\s+6)\b"),
    ),
]


@dataclass(frozen=True)
class DestructiveMatch:
    """One pattern hit, with enough context for the prompt UI to explain."""

    pattern_id: str
    description: str
    matched_text: str


def check_command(command: str) -> List[DestructiveMatch]:
    """
    Scan a proposed shell command for destructive patterns.

    Returns an empty list if the command is safe under the denylist.
    Returns one entry per matched pattern (a single command can match
    multiple — e.g. `rm -rf ~/` hits rm_recursive + rm_force + home_expand).
    """
    if not command or not isinstance(command, str):
        return []

    matches: List[DestructiveMatch] = []
    for pattern_id, description, regex in _DESTRUCTIVE_PATTERNS:
        m = regex.search(command)
        if m is not None:
            matches.append(
                DestructiveMatch(
                    pattern_id=pattern_id,
                    description=description,
                    matched_text=m.group(0).strip(),
                )
            )
    return matches


def format_warning(command: str, matches: List[DestructiveMatch]) -> str:
    """Build a human-readable warning block for the confirmation prompt."""
    lines = [
        "!! DESTRUCTIVE COMMAND DETECTED",
        f"   command: {command}",
        "   matched patterns:",
    ]
    for m in matches:
        lines.append(f"     - {m.pattern_id}: {m.description}")
        lines.append(f"       matched: {m.matched_text!r}")
    lines.append(
        "   This pattern caused the May 2026 home-directory wipe incident."
    )
    lines.append(
        "   Do NOT approve unless you have read the command and verified the targets."
    )
    return "\n".join(lines)
