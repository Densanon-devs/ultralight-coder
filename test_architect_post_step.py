"""
Tests for the architect's mandatory post-step consolidation
(Fix #6 from the 2026-04-26 handheld walkthrough).
"""
from __future__ import annotations
import sys
import tempfile
from pathlib import Path
from types import SimpleNamespace

ROOT = Path(__file__).parent
sys.path.insert(0, str(ROOT))

_CORE = ROOT.parent / "densanon-core"
if _CORE.exists() and str(_CORE) not in sys.path:
    sys.path.insert(0, str(_CORE))

from engine.architect_agent import ArchitectAgent
from engine.agent_tools import ToolRegistry, ToolSchema


def _make_arch(workspace, registry=None):
    """Build a minimally-wired ArchitectAgent — only _final_test_pass is exercised."""
    return ArchitectAgent(
        model=SimpleNamespace(generate=lambda *a, **k: ""),
        registry=registry or ToolRegistry(),
        workspace_root=workspace,
    )


def _registry_with_run_tests(result_dict):
    """Build a registry whose run_tests tool returns `result_dict`."""
    reg = ToolRegistry()
    reg.register(ToolSchema(
        name="run_tests",
        description="run tests",
        parameters={"type": "object", "properties": {}},
        function=lambda **kw: result_dict,
        category="test",
    ))
    return reg


def test_no_verification_when_no_test_signal():
    """Goal doesn't mention tests + no test files → no verification run."""
    tmp = Path(tempfile.mkdtemp())
    arch = _make_arch(tmp)
    summary = arch._final_test_pass("Add a function to x.py")
    assert summary == ""


def test_verification_runs_when_goal_mentions_tests():
    tmp = Path(tempfile.mkdtemp())
    reg = _registry_with_run_tests({"passed": True, "exit_code": 0, "stdout": "..."})
    arch = _make_arch(tmp, registry=reg)
    summary = arch._final_test_pass("Build it. Run the tests after.")
    assert "Tests pass" in summary


def test_verification_runs_when_test_files_exist():
    """Even without the goal asking, if test_*.py is present, verify."""
    tmp = Path(tempfile.mkdtemp())
    (tmp / "test_x.py").write_text("def test_one(): assert True")
    reg = _registry_with_run_tests({"passed": True, "exit_code": 0, "stdout": "."})
    arch = _make_arch(tmp, registry=reg)
    summary = arch._final_test_pass("Build a thing.")  # no test signal in goal
    assert "Tests pass" in summary


def test_verification_surfaces_failure_in_summary():
    """Test failures must show up in the architect's final answer —
    the walkthrough's Goal 1.5 falsely claimed success despite the
    test still failing. Never again."""
    tmp = Path(tempfile.mkdtemp())
    reg = _registry_with_run_tests({"passed": False, "exit_code": 1,
                                    "stdout": "1 failed, 0 passed"})
    arch = _make_arch(tmp, registry=reg)
    summary = arch._final_test_pass("Run the tests.")
    assert "FAIL" in summary
    assert "exit=1" in summary


def test_verification_no_op_when_run_tests_tool_missing():
    """If the registry doesn't have run_tests, return empty — don't crash."""
    tmp = Path(tempfile.mkdtemp())
    arch = _make_arch(tmp, registry=ToolRegistry())  # empty registry
    summary = arch._final_test_pass("Run the tests.")
    assert summary == ""


def test_verification_swallows_exceptions():
    """Tool execution errors must not break the architect's final answer."""
    reg = ToolRegistry()
    def _raise(**kw): raise RuntimeError("boom")
    reg.register(ToolSchema(
        name="run_tests", description="run tests",
        parameters={"type": "object", "properties": {}},
        function=_raise, category="test",
    ))
    tmp = Path(tempfile.mkdtemp())
    arch = _make_arch(tmp, registry=reg)
    summary = arch._final_test_pass("Run tests.")
    assert "raised" in summary or summary == ""


def test_test_signal_keywords_match():
    """Sanity check that all the trigger phrases are caught."""
    tmp = Path(tempfile.mkdtemp())
    reg = _registry_with_run_tests({"passed": True, "exit_code": 0})
    arch = _make_arch(tmp, registry=reg)
    for phrase in ("run the tests", "run tests", "verify they pass",
                   "pytest", "unittest", "all tests pass"):
        summary = arch._final_test_pass(f"Build and {phrase}.")
        assert summary, f"phrase {phrase!r} failed to trigger verification"


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
