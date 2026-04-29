"""
Tests for ulcagent._tdd_goal.

Exercises:
- Empty goal returns empty string.
- Whitespace-only goal returns empty string.
- Real goal is wrapped with the 5-step structure.
- The user's goal is preserved verbatim inside the wrapper.
- max_rounds is reflected in step 4 text.
- Default max_rounds is 5.
- Output mentions all the expected tool names (write_file, run_tests, edit_file).
"""
from __future__ import annotations
import sys
from pathlib import Path

ROOT = Path(__file__).parent
sys.path.insert(0, str(ROOT))

from ulcagent import _tdd_goal


def test_empty_goal_returns_empty():
    assert _tdd_goal("") == ""


def test_whitespace_goal_returns_empty():
    assert _tdd_goal("   \n\t  ") == ""


def test_real_goal_is_wrapped():
    out = _tdd_goal("add a divide function to calculator.py")
    assert "STEP 1." in out
    assert "STEP 2." in out
    assert "STEP 3." in out
    assert "STEP 4." in out
    assert "STEP 5." in out


def test_user_goal_preserved_verbatim():
    g = "add a divide function to calculator.py"
    assert g in _tdd_goal(g)


def test_max_rounds_reflected_in_step4():
    out = _tdd_goal("write a parser", max_rounds=8)
    assert "8 times" in out
    assert "5 times" not in out


def test_default_max_rounds_is_5():
    out = _tdd_goal("any goal")
    assert "5 times" in out


def test_mentions_expected_tools():
    out = _tdd_goal("any goal")
    assert "write_file" in out
    assert "run_tests" in out
    assert "edit_file" in out


def test_mentions_red_phase():
    out = _tdd_goal("any goal")
    assert "red phase" in out.lower() or "should fail" in out.lower()


def test_strips_user_goal_whitespace():
    out = _tdd_goal("   build a thing   ")
    assert "GOAL: build a thing" in out


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
