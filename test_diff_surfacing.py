"""
Tests for unified-diff surfacing on write_file and edit_file.

- Creating a new file: no diff (just "Created X").
- Overwriting a file with the same content: no diff.
- Overwriting with changes: return value contains a unified diff.
- edit_file with actual changes: return value contains a unified diff.
- Big diffs are truncated to max_lines with an elision marker.
"""
from __future__ import annotations
import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).parent
sys.path.insert(0, str(ROOT))

_CORE_ROOT = ROOT.parent / "densanon-core"
if _CORE_ROOT.exists() and str(_CORE_ROOT) not in sys.path:
    sys.path.insert(0, str(_CORE_ROOT))

from engine.agent_builtins import _write_file, _edit_file, _diff_preview, Workspace


def _ws(td: str) -> Workspace:
    return Workspace(root=Path(td))


def test_diff_preview_identical_is_empty():
    assert _diff_preview("abc\n", "abc\n", "x.py") == ""


def test_diff_preview_shows_changes():
    before = "def add(a, b):\n    return a + b\n"
    after = "def add(a, b):\n    \"\"\"Sum.\"\"\"\n    return a + b\n"
    d = _diff_preview(before, after, "calc.py")
    assert "a/calc.py" in d
    assert "b/calc.py" in d
    assert "+" in d
    assert "\"\"\"Sum.\"\"\"" in d


def test_diff_preview_truncates_long():
    # 60-line change → truncated to 20 lines + omitted marker
    before = "\n".join(f"line {i}" for i in range(50))
    after = "\n".join(f"changed {i}" for i in range(50))
    d = _diff_preview(before, after, "big.txt", max_lines=20)
    assert "... (" in d and "more diff lines elided)" in d
    assert d.count("\n") <= 20


def test_write_file_new_says_wrote_no_diff():
    with tempfile.TemporaryDirectory() as td:
        result = _write_file(_ws(td), "fresh.py", content="x = 1\n")
        assert "Wrote" in result
        # No diff block on fresh file creation
        assert "@@" not in result


def test_write_file_overwrite_with_changes_shows_diff():
    with tempfile.TemporaryDirectory() as td:
        ws = _ws(td)
        _write_file(ws, "x.py", content="x = 1\n")
        result = _write_file(ws, "x.py", content="x = 2\n")
        assert "Wrote" in result
        assert "@@" in result
        assert "-x = 1" in result
        assert "+x = 2" in result


def test_write_file_overwrite_same_content_no_diff_block():
    with tempfile.TemporaryDirectory() as td:
        ws = _ws(td)
        _write_file(ws, "x.py", content="x = 1\n")
        result = _write_file(ws, "x.py", content="x = 1\n")
        assert "Wrote" in result
        assert "@@" not in result  # diff block absent for unchanged content


def test_edit_file_shows_diff_on_success():
    with tempfile.TemporaryDirectory() as td:
        ws = _ws(td)
        _write_file(ws, "calc.py", content="def add(a, b):\n    return a + b\n")
        result = _edit_file(ws, "calc.py",
            old_string="def add(a, b):",
            new_string="def add(a: int, b: int) -> int:",
        )
        assert "Replaced" in result
        assert "@@" in result
        assert "-def add(a, b):" in result
        assert "+def add(a: int, b: int) -> int:" in result


def test_edit_file_diff_truncated_for_large_replace_all():
    with tempfile.TemporaryDirectory() as td:
        ws = _ws(td)
        big_body = "\n".join(f"    old_{i} = 1" for i in range(60)) + "\n"
        _write_file(ws, "big.py", content=big_body)
        result = _edit_file(ws, "big.py",
            old_string="old_", new_string="renamed_", replace_all=True,
        )
        assert "Replaced 60 occurrence(s)" in result
        assert "@@" in result
        assert "more diff lines elided" in result


if __name__ == "__main__":
    import inspect
    mod = sys.modules[__name__]
    fns = [(n, f) for n, f in inspect.getmembers(mod, inspect.isfunction) if n.startswith("test_")]
    passed = failed = 0
    for name, fn in fns:
        try:
            fn()
            print(f"  PASS  {name}")
            passed += 1
        except AssertionError as e:
            print(f"  FAIL  {name}: {e}")
            failed += 1
        except Exception as e:
            import traceback; traceback.print_exc()
            print(f"  ERR   {name}: {type(e).__name__}: {e}")
            failed += 1
    print(f"\n{passed} passed, {failed} failed")
    sys.exit(0 if failed == 0 else 1)
