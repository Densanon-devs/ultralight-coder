"""
Recovery detector — capture before/after pairs where the agent recovered
from a failure in a later iteration.

The failure_flagger tags one-sided WRONG patterns. This module pairs each
WRONG with the matching RIGHT (the next attempt that succeeded), producing
demonstration-style augmentor entries — far more teaching-effective than
abstract rules per today's GPT-5.5 prompt-lift result (which proved the
14B responds to in-context examples but NOT to abstract behavioral rules).

Pipeline:
  1. flag(result) returns FailureRecord entries
  2. detect_recoveries(result) walks AFTER each failure looking for a
     successful retry on the same file/tool
  3. For each (failure, recovery) pair, build_recovery_yaml() produces
     a YAML augmentor with a real BEFORE-and-AFTER example.

The recovery pair is more valuable than either side alone because:
  - The model sees its OWN failure mode and how to fix it
  - The diff is concrete (real arguments, not a template)
  - Tags include "recovery" so retrieval can boost these

Recoveries that DON'T pan out (failure → another failure → bail) are not
captured — only successful recoveries become positive examples.
"""
from __future__ import annotations

import time
import hashlib
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import yaml

try:
    from engine.failure_flagger import flag, FailureRecord, _SYNTHETIC_RESULT_NAMES
except ImportError:  # pragma: no cover
    from failure_flagger import flag, FailureRecord, _SYNTHETIC_RESULT_NAMES


# ── Data model ──────────────────────────────────────────────────────


@dataclass
class RecoveryPair:
    failure: FailureRecord
    fix_call_args: dict[str, Any]
    fix_tool_name: str
    fix_iteration: int
    fix_result_excerpt: str = ""

    def signature(self) -> str:
        h = hashlib.sha256()
        h.update(self.failure.category.encode("utf-8"))
        h.update((self.failure.file_path or "").encode("utf-8"))
        h.update(str(self.failure.iteration).encode("utf-8"))
        h.update(str(self.fix_iteration).encode("utf-8"))
        h.update(str(time.time_ns()).encode("utf-8"))
        return h.hexdigest()[:10]


# ── Detection ───────────────────────────────────────────────────────


def detect_recoveries(result: Any) -> list[RecoveryPair]:
    """For each FailureRecord in result, look for a subsequent successful
    tool call that touched the SAME file (or used a different tool on the
    same problem). Returns the (failure, fix) pairs.

    Heuristic recovery match:
      - Failure has a file_path X
      - Subsequent successful call has args.path == X
      - Subsequent call's iteration > failure.iteration
      - No intermediate failure on the same file (we want CLEAN recoveries)
    """
    failures = flag(result)
    if not failures:
        return []

    calls = list(getattr(result, "tool_calls", []) or [])
    results = list(getattr(result, "tool_results", []) or [])
    if not calls:
        return []

    # Build an iteration-indexed view: walk results once, recording for
    # each iteration whether it succeeded and on which file.
    iter_log: list[dict] = []
    call_cursor = 0
    iteration = 0
    for res in results:
        rname = getattr(res, "name", "") or ""
        is_synthetic = rname in _SYNTHETIC_RESULT_NAMES
        if not is_synthetic and call_cursor < len(calls):
            call = calls[call_cursor]
            call_cursor += 1
            iteration += 1
            cargs = getattr(call, "arguments", {}) or {}
            if not isinstance(cargs, dict):
                cargs = {}
            iter_log.append({
                "iteration": iteration,
                "tool_name": getattr(call, "name", "") or "",
                "args": cargs,
                "path": cargs.get("path"),
                "success": bool(getattr(res, "success", False)),
                "result_content": str(getattr(res, "content", ""))[:300],
                "synthetic_followups_clean": True,
            })
        elif is_synthetic and iter_log:
            # If a synthetic auto_verify reports an error after the
            # most recent real call, the recovery isn't clean.
            content = (str(getattr(res, "content", "")) +
                       " " + (getattr(res, "error", "") or ""))
            if "SyntaxError" in content or "references undefined" in content \
               or "expecting" in content.lower() or not getattr(res, "success", True):
                iter_log[-1]["synthetic_followups_clean"] = False

    pairs: list[RecoveryPair] = []
    for fr in failures:
        if fr.category == "premature_bail":
            continue  # bails don't have recoveries
        # Look for the next clean iteration that operates on the same file.
        # Special case: parse_error synthetics carry no file_path or
        # tool_name on the FailureRecord (the malformed call was never
        # parsed into a ToolCall). For those, we treat the very next
        # successful iteration as the recovery — that's what's actually
        # happening in the run: model retried and got it right.
        target_path = fr.file_path
        is_parseerror_orphan = not target_path and not fr.tool_name
        for entry in iter_log:
            if entry["iteration"] <= fr.iteration:
                continue
            if not entry["success"]:
                continue
            if not entry["synthetic_followups_clean"]:
                continue
            same_file = target_path and entry["path"] == target_path
            same_tool_no_path = (
                target_path is None
                and fr.tool_name
                and entry["tool_name"] == fr.tool_name
            )
            if same_file or same_tool_no_path or is_parseerror_orphan:
                pairs.append(RecoveryPair(
                    failure=fr,
                    fix_call_args=entry["args"],
                    fix_tool_name=entry["tool_name"],
                    fix_iteration=entry["iteration"],
                    fix_result_excerpt=entry["result_content"],
                ))
                break  # one fix per failure
    return pairs


# ── YAML synthesis ──────────────────────────────────────────────────


_BEFORE_AFTER_INTRO = {
    "json_quote_leak": (
        "When the agent emits a multi-line write_file with Python single-quoted\n"
        "literals inside JSON `content`, the JSON parser rejects it. The fix\n"
        "is to use the array form for `content` so each line is its own\n"
        "string element.\n"
    ),
    "fstring_nested_quote": (
        "When a single-quoted f-string contains another single-quoted literal\n"
        "inside its `{ }` block, Python 3.10/3.11 errors with\n"
        "`f-string: expecting '}'`. The fix is to flip the OUTER f-string to\n"
        "double quotes, leaving the INNER literals single-quoted.\n"
    ),
    "missing_import": (
        "When a write_file produces a file referencing names that aren't\n"
        "imported, auto_verify catches it. The fix is to prepend the missing\n"
        "import via edit_file with empty `old_string`.\n"
    ),
    "stuck_repeat_loop": (
        "When the same tool call repeats and the agent fires `stuck_repeat`,\n"
        "the recovery is to pivot: switch tool, use a different anchor, or\n"
        "re-read the file to refresh state.\n"
    ),
}


def _abridge(args: dict, limit: int = 800) -> str:
    """Format a tool args dict for inclusion in the YAML solution body.
    Keeps it readable but caps length so multi-thousand-char content
    doesn't bloat the augmentor."""
    import json
    try:
        s = json.dumps(args, ensure_ascii=False, indent=2)
    except Exception:
        s = str(args)
    if len(s) > limit:
        s = s[:limit] + f"\n... <+{len(s) - limit} chars>"
    return s


def build_recovery_yaml(pair: RecoveryPair, goal: str) -> dict | None:
    """Convert a RecoveryPair into a YAML-shaped dict matching the
    augmentor library schema. Returns None if no template is available
    for the failure category."""
    intro = _BEFORE_AFTER_INTRO.get(pair.failure.category)
    if intro is None:
        return None
    failed_args = _abridge(pair.failure.triggering_args or {})
    fix_args = _abridge(pair.fix_call_args or {})
    solution = (
        intro
        + "\nWRONG (failure mode in this run):\n"
        + f"```\n{pair.failure.tool_name} args:\n{failed_args}\n```\n"
        + f"\nReason this failed: {pair.failure.error_excerpt[:200]}\n"
        + "\nRIGHT (the same goal, retried successfully):\n"
        + f"```\n{pair.fix_tool_name} args:\n{fix_args}\n```\n"
        + "\nUse the same shape as the RIGHT call when faced with similar\n"
        + "queries. The before-and-after pattern teaches the recovery path\n"
        + "directly."
    )
    return {
        "domain": "agentic",
        "category": f"recovery_{pair.failure.category}",
        "examples": [{
            "query": goal.strip() + "\n\n# Recovery captured from a real run",
            "solution": solution,
            "tags": ["recovery", pair.failure.category, pair.fix_tool_name],
            "difficulty": "medium",
            "source": "recovery_detector",
            "captured_at": time.time(),
            "failed_iteration": pair.failure.iteration,
            "fixed_iteration": pair.fix_iteration,
            "triggering_file": pair.failure.file_path or "",
        }],
    }


def write_recovery(pair: RecoveryPair, goal: str, repo_root: Path | str) -> Path | None:
    """Persist a RecoveryPair as a YAML in the auto_generated_review/
    queue. Same path policy as failure flagger output."""
    payload = build_recovery_yaml(pair, goal)
    if payload is None:
        return None
    repo_root = Path(repo_root)
    target_dir = (repo_root / "data" / "auto_generated_review"
                  / f"recovery_{pair.failure.category}")
    target_dir.mkdir(parents=True, exist_ok=True)
    fname = f"{pair.signature()}.yaml"
    path = target_dir / fname
    path.write_text(
        yaml.safe_dump(payload, sort_keys=False, allow_unicode=True),
        encoding="utf-8",
    )
    return path


def write_all_recoveries(result: Any, goal: str, repo_root: Path | str) -> list[Path]:
    """Detect all recoveries in `result` and write each to disk. Returns
    list of paths written (may be shorter than detected if some categories
    have no template)."""
    pairs = detect_recoveries(result)
    out: list[Path] = []
    for pair in pairs:
        p = write_recovery(pair, goal, repo_root)
        if p is not None:
            out.append(p)
    return out
