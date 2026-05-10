"""Unit tests for engine.self_heal — live diagnose-and-repair classifier.

Tests run against a minimal stub that mimics the ToolResult shape the
real `engine.agent.ToolResult` exposes (name, success, error, output).
No model calls, no agent loop — pure logic.
"""

from __future__ import annotations

from dataclasses import dataclass

import pytest

from engine.self_heal import (
    CWD_FAILURE,
    MISSING_IMPORT,
    NAME_NOT_DEFINED,
    PARSE_ERROR,
    STALE_ANCHOR,
    SYNTAX_ERROR,
    TOOL_REJECTED,
    TRACEBACK,
    classify_failure,
    diagnose_message,
    should_inject_diagnose,
)


@dataclass
class _R:
    name: str = ""
    success: bool = False
    error: str | None = None
    output: str | None = None


# ── classify_failure ─────────────────────────────────────────────


def test_success_returns_none():
    assert classify_failure(_R(name="read_file", success=True, output="...")) is None


def test_syntax_error_classified():
    r = _R(name="auto_verify", success=False, error="SyntaxError in foo.py at line 12")
    assert classify_failure(r) == SYNTAX_ERROR


def test_stale_anchor_classified():
    r = _R(name="edit_file", success=False, error="old_string not found in src/x.py")
    assert classify_failure(r) == STALE_ANCHOR


def test_name_not_defined_classified():
    r = _R(name="run_tests", success=False, error="NameError: name 'x' is not defined")
    assert classify_failure(r) == NAME_NOT_DEFINED


def test_missing_import_via_auto_verify_output():
    # auto_verify can report undefined names via OUTPUT not error and may set success=True
    r = _R(name="auto_verify", success=True, output="cli.py references undefined names: json")
    assert classify_failure(r) == MISSING_IMPORT


def test_cwd_failure_classified():
    r = _R(name="run_bash", success=False, error="No such file or directory: 'tests/'")
    assert classify_failure(r) == CWD_FAILURE


def test_traceback_classified():
    r = _R(
        name="run_tests",
        success=False,
        error='Traceback (most recent call last):\n  File "x.py", line 5, in <module>\n    raise ValueError',
    )
    assert classify_failure(r) == TRACEBACK


def test_parse_error_classified():
    r = _R(name="parse_error", success=False, error="tool call rejected: invalid JSON")
    assert classify_failure(r) == PARSE_ERROR


def test_unknown_failure_falls_back_to_tool_rejected():
    r = _R(name="grep", success=False, error="something went sideways")
    assert classify_failure(r) == TOOL_REJECTED


def test_no_error_text_returns_tool_rejected():
    r = _R(name="grep", success=False, error="")
    assert classify_failure(r) == TOOL_REJECTED


# ── should_inject_diagnose ───────────────────────────────────────


def test_two_consecutive_same_class_triggers():
    assert should_inject_diagnose([STALE_ANCHOR, STALE_ANCHOR]) is True


def test_three_consecutive_same_class_still_triggers():
    assert should_inject_diagnose([SYNTAX_ERROR, SYNTAX_ERROR, SYNTAX_ERROR]) is True


def test_two_different_classes_does_not_trigger():
    assert should_inject_diagnose([STALE_ANCHOR, SYNTAX_ERROR]) is False


def test_single_failure_does_not_trigger():
    assert should_inject_diagnose([STALE_ANCHOR]) is False


def test_success_in_between_resets_streak():
    assert should_inject_diagnose([STALE_ANCHOR, None, STALE_ANCHOR]) is False


def test_success_at_tail_does_not_trigger():
    assert should_inject_diagnose([STALE_ANCHOR, STALE_ANCHOR, None]) is False


def test_empty_streak_does_not_trigger():
    assert should_inject_diagnose([]) is False


def test_min_streak_three_requires_three():
    assert should_inject_diagnose([SYNTAX_ERROR, SYNTAX_ERROR], min_streak=3) is False
    assert should_inject_diagnose([SYNTAX_ERROR] * 3, min_streak=3) is True


def test_min_streak_below_two_never_fires():
    assert should_inject_diagnose([SYNTAX_ERROR] * 5, min_streak=1) is False


# ── diagnose_message ─────────────────────────────────────────────


def test_diagnose_message_mentions_failure_class():
    msg = diagnose_message(SYNTAX_ERROR, attempts=2)
    assert "syntax_error" in msg
    assert "PAUSE" in msg


def test_diagnose_message_per_class_hint_present():
    for cls in (SYNTAX_ERROR, TRACEBACK, STALE_ANCHOR, NAME_NOT_DEFINED,
                MISSING_IMPORT, CWD_FAILURE, PARSE_ERROR, TOOL_REJECTED):
        msg = diagnose_message(cls, attempts=2)
        # Each class produces a non-trivial body.
        assert len(msg) > 200
        # Each class's hint contains the word "approach", "fix", "edit",
        # "import", "list", "json", or some signal of guidance.
        assert any(kw in msg.lower() for kw in (
            "approach", "fix", "edit", "import", "list_dir", "json", "diagnose",
        ))


def test_diagnose_message_unknown_class_falls_back():
    # An unrecognized class still produces a sensible message via the
    # generic TOOL_REJECTED hint.
    msg = diagnose_message("nonexistent_class", attempts=2)
    assert "PAUSE" in msg
    assert "nonexistent_class" in msg


def test_diagnose_message_attempts_count_surfaced():
    msg = diagnose_message(SYNTAX_ERROR, attempts=3)
    assert "3 times" in msg


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-v"]))
