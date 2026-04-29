"""
Tests for ulcagent._do_review_deep.

Builds a temporary git repo with known changes, runs _do_review_deep,
and checks that the produced goal string is well-formed.

Exercises:
- Returns None when there are no changes
- Returns a structured goal mentioning every changed file
- Includes the diff in a fenced ```diff block
- Diff is truncated at 400 lines for huge changes
- Workflow has the read-only invariant ("do NOT modify any files")
- Output sections: Critical / High / Medium / Low / Questions
"""
from __future__ import annotations
import os
import subprocess
import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).parent
sys.path.insert(0, str(ROOT))

from ulcagent import _do_review_deep


def _git(cwd, *args):
    return subprocess.run(["git", *args], cwd=str(cwd), capture_output=True, text=True, check=True)


def _make_repo():
    tmp = Path(tempfile.mkdtemp())
    _git(tmp, "init", "-q")
    _git(tmp, "config", "user.email", "t@example.com")
    _git(tmp, "config", "user.name", "Test")
    (tmp / "main.py").write_text("def f():\n    return 1\n", encoding="utf-8")
    (tmp / "util.py").write_text("def g():\n    return 2\n", encoding="utf-8")
    _git(tmp, "add", ".")
    _git(tmp, "commit", "-q", "-m", "init")
    return tmp


def test_no_changes_returns_none():
    tmp = _make_repo()
    out = _do_review_deep(tmp)
    assert out is None


def test_with_changes_returns_goal():
    tmp = _make_repo()
    (tmp / "main.py").write_text("def f():\n    return 999\n", encoding="utf-8")
    (tmp / "new.py").write_text("def h():\n    return 3\n", encoding="utf-8")
    _git(tmp, "add", ".")
    out = _do_review_deep(tmp)
    assert out is not None
    assert "main.py" in out
    assert "new.py" in out


def test_goal_has_diff_block():
    tmp = _make_repo()
    (tmp / "main.py").write_text("def f():\n    return 999\n", encoding="utf-8")
    _git(tmp, "add", ".")
    out = _do_review_deep(tmp)
    assert "```diff" in out
    assert "999" in out


def test_goal_is_read_only():
    tmp = _make_repo()
    (tmp / "main.py").write_text("def f():\n    return 999\n", encoding="utf-8")
    _git(tmp, "add", ".")
    out = _do_review_deep(tmp)
    assert "do NOT modify" in out or "read-only" in out.lower()


def test_severity_sections_present():
    tmp = _make_repo()
    (tmp / "main.py").write_text("def f():\n    return 999\n", encoding="utf-8")
    _git(tmp, "add", ".")
    out = _do_review_deep(tmp)
    for header in ("## Critical", "## High", "## Medium", "## Low", "## Questions"):
        assert header in out, f"missing section: {header}"


def test_large_diff_truncated():
    tmp = _make_repo()
    big = "\n".join(f"line {i}" for i in range(1000))
    (tmp / "big.py").write_text(big, encoding="utf-8")
    _git(tmp, "add", ".")
    out = _do_review_deep(tmp)
    # Truncation marker is included
    assert "more lines truncated" in out


def test_changed_file_listing():
    tmp = _make_repo()
    (tmp / "main.py").write_text("# changed\n", encoding="utf-8")
    (tmp / "util.py").write_text("# changed\n", encoding="utf-8")
    _git(tmp, "add", ".")
    out = _do_review_deep(tmp)
    # Both files should appear in the bullet list
    assert "  - main.py" in out
    assert "  - util.py" in out


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
