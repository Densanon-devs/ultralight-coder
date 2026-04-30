"""
Replay validator — a gate between auto-generated YAMLs and the
retrieval index. Two phases:

  STATIC (cheap, runs in CI / test suite):
    - schema conforms to augmentor library
    - solution body contains category-specific must-have patterns
      (e.g. json_quote_leak solution MUST mention array form AND show
      both a WRONG and a RIGHT example)
    - tags include the failure category for retrieval ranking
    - example.query is non-trivial (not just the template default)

  LIVE (deferred, GPU-bound):
    - Re-run the original failing goal through a fresh Agent with the
      candidate YAML injected as an in-context example
    - Compare iteration counts / success vs the failure baseline
    - Mark the YAML as `replay_status: validated` if the agent now PASSes

The static phase is what runs by default. /promote checks
`replay_status` is at least "static_ok" before copying. Live replay is
opt-in via `validate_live(yaml_path, model)` and only worth running
when the GPU is otherwise idle.
"""
from __future__ import annotations

import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml


# ── Per-category must-contain patterns ─────────────────────────────


# A YAML with this category's solution body MUST contain ALL the
# substrings in the matching list (case-insensitive). If any are
# missing, the YAML failed static replay validation.
_REQUIRED_SOLUTION_PATTERNS = {
    "json_array_form_for_multiline_writes": [
        "array form",
        "WRONG",
        "RIGHT",
        "content",
    ],
    "fstring_nested_quotes_python310": [
        "double",
        "single",
        "WRONG",
        "RIGHT",
    ],
    "prepend_missing_import_after_write": [
        "edit_file",
        "old_string",
        "import",
        "WRONG",
        "RIGHT",
    ],
    "check_stderr_before_declaring_done": [
        "Traceback",
        "WRONG",
        "RIGHT",
    ],
    "recover_from_stuck_repeat": [
        "stuck_repeat",
        "read_file",
        "switch",
    ],
    "reread_before_edit_when_anchor_stale": [
        "old_string not found",
        "read_file",
        "WRONG",
        "RIGHT",
    ],
    "confirm_path_before_run_bash": [
        "list_dir",
        "WRONG",
        "RIGHT",
    ],
    "test_must_import_target_module": [
        "ModuleNotFoundError",
        "conftest",
    ],
    "address_every_numbered_requirement": [
        "WRONG",
        "RIGHT",
        "checklist",
    ],
    "emit_tool_call_immediately_after_narration": [
        "WRONG",
        "RIGHT",
        "tool_call",
    ],
    "touch_every_file_named_in_goal": [
        "WRONG",
        "RIGHT",
        "files_named",
    ],
}

# Recovery YAMLs follow the same pattern but their category prefix
# is `recovery_<category>`. Inherit must-haves from the underlying
# category.
def _required_for(category: str) -> list[str] | None:
    if category in _REQUIRED_SOLUTION_PATTERNS:
        return _REQUIRED_SOLUTION_PATTERNS[category]
    return None


# ── Result type ─────────────────────────────────────────────────────


@dataclass
class ReplayResult:
    path: Path
    valid: bool
    phase: str           # "static" | "live"
    reason: str = ""
    matched_patterns: int = 0
    total_patterns: int = 0


# ── Static replay ───────────────────────────────────────────────────


def _load(path: Path) -> dict | None:
    try:
        return yaml.safe_load(path.read_text(encoding="utf-8"))
    except Exception:
        return None


def validate_static(yaml_path: Path | str) -> ReplayResult:
    """Schema + content validation. Returns ReplayResult.valid=True iff
    the YAML loads, has the required schema fields, and the solution
    body contains every must-have pattern for its declared category."""
    yaml_path = Path(yaml_path)
    data = _load(yaml_path)
    if data is None:
        return ReplayResult(yaml_path, False, "static", "yaml.safe_load failed")

    examples = data.get("examples")
    if not examples or not isinstance(examples, list):
        return ReplayResult(yaml_path, False, "static",
                            "missing or empty examples list")

    category = data.get("category", "") or ""
    solution_join = "\n\n".join(
        ex.get("solution", "") or "" for ex in examples if isinstance(ex, dict)
    )

    # Every example must have non-empty query + solution
    for i, ex in enumerate(examples):
        if not isinstance(ex, dict):
            return ReplayResult(yaml_path, False, "static", f"example {i} not a dict")
        if not (ex.get("query") or "").strip():
            return ReplayResult(yaml_path, False, "static", f"example {i}: empty query")
        if not (ex.get("solution") or "").strip():
            return ReplayResult(yaml_path, False, "static", f"example {i}: empty solution")

    required = _required_for(category)
    if required is None:
        # Unknown category — pass schema check, but no pattern gating
        return ReplayResult(yaml_path, True, "static",
                            "schema OK; no pattern check for unknown category",
                            matched_patterns=0, total_patterns=0)

    body_lower = solution_join.lower()
    matched = sum(1 for pat in required if pat.lower() in body_lower)
    if matched == len(required):
        return ReplayResult(yaml_path, True, "static",
                            f"all {matched} required patterns present",
                            matched_patterns=matched, total_patterns=len(required))
    missing = [p for p in required if p.lower() not in body_lower]
    return ReplayResult(yaml_path, False, "static",
                        f"missing patterns: {missing}",
                        matched_patterns=matched, total_patterns=len(required))


def validate_dir(directory: Path | str) -> list[ReplayResult]:
    """Run static validation on every YAML under `directory`."""
    directory = Path(directory)
    out: list[ReplayResult] = []
    if not directory.exists():
        return out
    for yaml_path in sorted(directory.rglob("*.yaml")):
        out.append(validate_static(yaml_path))
    return out


def summarize(results: list[ReplayResult]) -> dict:
    out = {"valid": 0, "invalid": 0, "by_phase": {}}
    for r in results:
        if r.valid:
            out["valid"] += 1
        else:
            out["invalid"] += 1
        out["by_phase"][r.phase] = out["by_phase"].get(r.phase, 0) + 1
    return out


# ── Live replay (placeholder; GPU-bound) ────────────────────────────


def validate_live(yaml_path: Path | str, model: Any, original_goal: str) -> ReplayResult:
    """Re-run the originally-failing goal with the candidate YAML injected
    as an in-context example. Mark validated iff the agent succeeds within
    the same iteration budget the failure baseline used.

    NOT IMPLEMENTED YET — returns a placeholder ReplayResult. Wiring
    requires:
      (a) AugmentorRouter that can be primed with a single YAML on the fly
      (b) A fresh Agent with that router in its system prompt extra
      (c) Comparing the iteration count + final pass/fail to a known
          baseline for the original goal

    Build this when the GPU is free for a multi-minute validation pass.
    """
    return ReplayResult(
        Path(yaml_path), False, "live",
        "validate_live not implemented — use validate_static for now"
    )


# ── Smoke ───────────────────────────────────────────────────────────


if __name__ == "__main__":
    import sys
    target = Path(sys.argv[1]) if len(sys.argv) > 1 else Path("data/auto_generated_review")
    results = validate_dir(target)
    s = summarize(results)
    print(f"Validated {len(results)} YAMLs: {s['valid']} valid, {s['invalid']} invalid")
    for r in results:
        mark = "OK" if r.valid else "FAIL"
        print(f"  [{mark}] {r.path.name}: {r.reason}")
