"""
Tests for the /proof slash command — self-diagnose-and-fix wrapper.
"""
from __future__ import annotations
import sys
from pathlib import Path

ROOT = Path(__file__).parent
sys.path.insert(0, str(ROOT))

from ulcagent import _proof_goal


def test_no_target_uses_project_wide_run_tests():
    """When called as /proof (no path), the goal should run all tests
    in the project."""
    goal = _proof_goal("")
    assert "run_tests" in goal
    assert "all tests" in goal.lower() or "all the tests" in goal.lower()


def test_with_target_mentions_path_unquoted():
    """Quoting the path made the 14B refuse with 'outside workspace'.
    The goal must reference the path unquoted so the model treats it
    as a workspace-relative argument, not a literal string."""
    goal = _proof_goal("tests/test_timer.py")
    assert "tests/test_timer.py" in goal
    # No literal-string quoting around the path
    assert "'tests/test_timer.py'" not in goal
    assert '"tests/test_timer.py"' not in goal


def test_goal_calls_out_both_failure_sources():
    """The model often assumes source is wrong when the test setup is
    actually buggy (and vice versa). Goal must explicitly say both
    are on the table."""
    goal = _proof_goal("")
    assert "source" in goal.lower()
    assert "test setup" in goal.lower()


def test_goal_mentions_mocks_fixtures_imports():
    """The 14B's most common test-setup bugs cluster around these.
    Naming them shortens the diagnostic search."""
    goal = _proof_goal("")
    assert "mock" in goal.lower()


def test_goal_steers_to_targeted_edit_file():
    """Avoid the destructive empty-old_string shortcut by explicitly
    naming targeted edit_file as the preferred tool, with write_file
    only as a 'genuine full rewrite' escape hatch."""
    goal = _proof_goal("")
    assert "edit_file" in goal
    assert "write_file" in goal


def test_goal_is_imperative_action():
    """The goal text should contain an imperative verb so the
    incomplete_deliverable detector can apply correctly."""
    goal = _proof_goal("")
    assert "fix" in goal.lower() or "find the bug" in goal.lower()


def test_goal_handles_extra_whitespace_in_target():
    goal = _proof_goal("   tests/   ")
    assert "tests/" in goal


def test_goal_says_dont_give_up():
    """Anti-bail nudge — the 14B's pre_finish_check + augmentor for
    'I cannot complete' fires post-hoc but the goal can prevent it
    upfront."""
    goal = _proof_goal("")
    assert "give up" in goal.lower() or "fixable" in goal.lower()


def test_goal_is_terse_not_multistep():
    """Earlier multi-step phrasings (STEP 1, STEP 2, ...) caused the
    model to refuse at iter 1. The terser shape that works in practice
    is closer to a human-style ad-hoc goal."""
    goal = _proof_goal("")
    # Should NOT contain the elaborate STEP-numbered structure
    assert "STEP 1." not in goal
    assert "STEP 2." not in goal


# ── Pre-flight path ──────────────────────────────────────────────


def test_preflight_passes_returns_no_op_goal(tmp_path=None):
    """If pytest already passes when /proof is invoked, the goal should
    tell the agent to confirm and stop — no edits."""
    import os, tempfile, subprocess
    from pathlib import Path
    tmp = Path(tempfile.mkdtemp())
    # Write a passing test file
    (tmp / "test_ok.py").write_text("def test_ok(): assert True\n")
    cwd_before = Path.cwd()
    try:
        os.chdir(tmp)
        goal = _proof_goal("test_ok.py", run_pytest_preflight=True)
        assert "already passed" in goal.lower() or "no edits needed" in goal.lower()
    finally:
        os.chdir(cwd_before)


def test_preflight_failure_embeds_traceback():
    """The whole point of pre-flight: convert open-ended diagnose to
    concrete fix-this-traceback. The actual failure output must be
    embedded in the goal."""
    import os, tempfile
    from pathlib import Path
    tmp = Path(tempfile.mkdtemp())
    (tmp / "test_fail.py").write_text(
        "def test_one(): assert 1 == 2  # always fails\n"
    )
    cwd_before = Path.cwd()
    try:
        os.chdir(tmp)
        goal = _proof_goal("test_fail.py", run_pytest_preflight=True)
        # Should include the failing test name and the assertion error
        assert "test_one" in goal or "FAILED" in goal or "assert 1 == 2" in goal
        # Should still contain the diagnostic guidance
        assert "fixable" in goal.lower() or "fix" in goal.lower()
    finally:
        os.chdir(cwd_before)


def test_preflight_truncates_huge_output():
    """If pytest produces multi-MB output, the goal should truncate."""
    import os, tempfile
    from pathlib import Path
    tmp = Path(tempfile.mkdtemp())
    # Generate many failing tests to bloat output
    big = "\n".join(f"def test_fail_{i}(): assert False" for i in range(200))
    (tmp / "test_big.py").write_text(big + "\n")
    cwd_before = Path.cwd()
    try:
        os.chdir(tmp)
        goal = _proof_goal("test_big.py", run_pytest_preflight=True)
        # Goal stays under a sensible cap
        assert len(goal) < 5000
    finally:
        os.chdir(cwd_before)


def test_preflight_off_skips_pytest_invocation():
    """The unit-test-friendly path (run_pytest_preflight=False) must not
    invoke pytest. If it did, this test would slow / be flaky."""
    import time as _time
    t0 = _time.monotonic()
    goal = _proof_goal("any/path", run_pytest_preflight=False)
    elapsed = _time.monotonic() - t0
    assert elapsed < 0.1  # synchronous string build only


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
