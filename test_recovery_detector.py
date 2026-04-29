"""
Tests for engine/recovery_detector.py — capture before/after pairs where
the agent recovered from a failure.
"""
from __future__ import annotations
import sys
import tempfile
from pathlib import Path
from types import SimpleNamespace

ROOT = Path(__file__).parent
sys.path.insert(0, str(ROOT))

_CORE_ROOT = ROOT.parent / "densanon-core"
if _CORE_ROOT.exists() and str(_CORE_ROOT) not in sys.path:
    sys.path.insert(0, str(_CORE_ROOT))

from engine.recovery_detector import (
    detect_recoveries, build_recovery_yaml, write_recovery, write_all_recoveries,
    RecoveryPair,
)


def _call(name, **args): return SimpleNamespace(name=name, arguments=args)
def _ok(content="", name="result"):  return SimpleNamespace(name=name, success=True, content=content, error="")
def _err(content="", error="", name="result"): return SimpleNamespace(name=name, success=False, content=content, error=error)


def test_no_recoveries_when_no_failures():
    res = SimpleNamespace(
        final_answer="Done.",
        tool_calls=[_call("write_file", path="x.py", content=["pass"])],
        tool_results=[_ok(content="Wrote 4 chars to x.py")],
        stop_reason="answered",
    )
    assert detect_recoveries(res) == []


def test_recovery_pair_detected_after_json_parse_fail():
    """iter1 = parse_error (json_quote_leak), iter2 = clean retry on same file."""
    res = SimpleNamespace(
        final_answer="Done.",
        tool_calls=[
            _call("write_file", path="storage.py", content=["bad"]),
            _call("write_file", path="storage.py", content=["good"]),
        ],
        tool_results=[
            SimpleNamespace(name="parse_error", success=False, content="",
                            error="bare JSON tool call failed to parse: Expecting ',' delimiter"),
            _ok(content="Wrote 100 chars to storage.py"),
        ],
        stop_reason="answered",
    )
    pairs = detect_recoveries(res)
    assert len(pairs) == 1
    pair = pairs[0]
    assert pair.failure.category == "json_quote_leak"
    assert pair.fix_iteration > pair.failure.iteration
    assert pair.fix_call_args.get("path") == "storage.py"


def test_recovery_pair_for_missing_import():
    res = SimpleNamespace(
        final_answer="Done.",
        tool_calls=[
            _call("write_file", path="cli.py", content=["..."]),
            _call("edit_file", path="cli.py", old_string="", new_string="import json\n"),
        ],
        tool_results=[
            _ok(content="Wrote 200 chars"),
            SimpleNamespace(name="auto_verify", success=False, content="",
                            error="syntax OK but cli.py references undefined names: json"),
            _ok(content="Prepended 12 chars"),
        ],
        stop_reason="answered",
    )
    pairs = detect_recoveries(res)
    assert any(p.failure.category == "missing_import" for p in pairs)


def test_no_recovery_when_followup_also_failed():
    """If the retry also failed, that's NOT a recovery — just two failures."""
    res = SimpleNamespace(
        final_answer="",
        tool_calls=[
            _call("write_file", path="x.py", content=["..."]),
            _call("write_file", path="x.py", content=["..."]),
        ],
        tool_results=[
            SimpleNamespace(name="parse_error", success=False, content="",
                            error="bare JSON tool call failed to parse: Expecting ',' delimiter"),
            SimpleNamespace(name="parse_error", success=False, content="",
                            error="bare JSON tool call failed to parse: Unterminated string"),
        ],
        stop_reason="answered",
    )
    assert detect_recoveries(res) == []


def test_premature_bail_excluded_from_recoveries():
    """premature_bail isn't a fixable iteration — it's a final-answer bug."""
    res = SimpleNamespace(
        final_answer="The task is complete. CLI works as expected.",
        tool_calls=[_call("run_bash", command="python cli.py")],
        tool_results=[_ok(content="{'stdout': '', 'stderr': 'Traceback...'}")],
        stop_reason="answered",
    )
    pairs = detect_recoveries(res)
    assert all(p.failure.category != "premature_bail" for p in pairs)


def test_build_recovery_yaml_contains_before_and_after():
    pair = RecoveryPair(
        failure=SimpleNamespace(  # type: ignore — duck-typed FailureRecord
            category="json_quote_leak",
            iteration=1,
            tool_name="write_file",
            error_excerpt="Expecting ',' delimiter",
            triggering_args={"path": "x.py", "content": "bad single quote"},
            file_path="x.py",
            suggested_fix="...",
        ),
        fix_call_args={"path": "x.py", "content": ["import json", ""]},
        fix_tool_name="write_file",
        fix_iteration=2,
        fix_result_excerpt="Wrote 30 chars",
    )
    payload = build_recovery_yaml(pair, "Build a todo CLI")
    assert payload is not None
    assert payload["domain"] == "agentic"
    assert payload["category"] == "recovery_json_quote_leak"
    sol = payload["examples"][0]["solution"]
    assert "WRONG" in sol
    assert "RIGHT" in sol


def test_build_recovery_yaml_unknown_category_returns_none():
    pair = RecoveryPair(
        failure=SimpleNamespace(  # type: ignore
            category="totally_made_up_category",
            iteration=1, tool_name="write_file", error_excerpt="x",
            triggering_args={}, file_path=None, suggested_fix="",
        ),
        fix_call_args={}, fix_tool_name="write_file",
        fix_iteration=2, fix_result_excerpt="",
    )
    assert build_recovery_yaml(pair, "any") is None


def test_write_recovery_lands_in_review_dir():
    tmp = Path(tempfile.mkdtemp())
    pair = RecoveryPair(
        failure=SimpleNamespace(  # type: ignore
            category="json_quote_leak",
            iteration=1, tool_name="write_file",
            error_excerpt="x", triggering_args={"path": "y.py"},
            file_path="y.py", suggested_fix="",
        ),
        fix_call_args={"path": "y.py"}, fix_tool_name="write_file",
        fix_iteration=2, fix_result_excerpt="",
    )
    p = write_recovery(pair, "any goal", tmp)
    assert p is not None
    assert "auto_generated_review" in str(p)
    assert "recovery_json_quote_leak" in str(p)
    assert "augmentor_examples" not in str(p), "must NOT land in retrieval scan path"


def test_write_all_recoveries_e2e():
    tmp = Path(tempfile.mkdtemp())
    res = SimpleNamespace(
        final_answer="Done.",
        tool_calls=[
            _call("write_file", path="storage.py", content=["bad"]),
            _call("write_file", path="storage.py", content=["good"]),
        ],
        tool_results=[
            SimpleNamespace(name="parse_error", success=False, content="",
                            error="bare JSON tool call failed to parse: Expecting ',' delimiter"),
            _ok(content="Wrote 100 chars"),
        ],
        stop_reason="answered",
    )
    paths = write_all_recoveries(res, "Build a todo CLI", tmp)
    assert len(paths) >= 1
    for p in paths:
        assert p.exists()


if __name__ == "__main__":
    import traceback
    tests = [v for k, v in globals().items() if k.startswith("test_") and callable(v)]
    failed = 0
    for t in tests:
        try:
            t()
            print(f"  PASS  {t.__name__}")
        except Exception:
            failed += 1
            print(f"  FAIL  {t.__name__}")
            traceback.print_exc()
    print(f"\n{len(tests) - failed}/{len(tests)} tests passed")
    sys.exit(1 if failed else 0)
