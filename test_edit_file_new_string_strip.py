"""
Regression test for the 2026-04-26 fix: edit_file's fuzzy match strips
line-number prefixes from old_string but historically didn't strip them
from new_string. When the 14B copies a line from read_file output into
BOTH old_string and new_string, the old side matches but the new side
corrupts the file with `    15\t...` prefixes.

Surfaced in the counter-prime experiment iters 1-3: refusal broken,
model emitted edits, but every run produced SyntaxError on the edited
file because new_string carried the line-number prefix.
"""
from __future__ import annotations
import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).parent
sys.path.insert(0, str(ROOT))

_CORE = ROOT.parent / "densanon-core"
if _CORE.exists() and str(_CORE) not in sys.path:
    sys.path.insert(0, str(_CORE))

from engine.agent_builtins import build_default_registry


def _setup(initial: str, name: str = "x.py"):
    tmp = Path(tempfile.mkdtemp())
    p = tmp / name
    p.write_text(initial, encoding="utf-8")
    reg = build_default_registry(tmp)
    return tmp, p, reg


def _edit(reg, **kw):
    return reg.get("edit_file").function(**kw)


def test_new_string_with_line_prefix_gets_stripped():
    """The exact failure shape from counter-prime iters 1-3."""
    tmp, p, reg = _setup(
        "import pytest\n"
        "from unittest.mock import MagicMock\n"
        "\n"
        "@pytest.fixture\n"
        "def mock_clock():\n"
        "    return MagicMock(return_value=0)\n"
    )
    # Model pasted line-number prefixes into BOTH sides
    out = _edit(reg, path=p.name,
                old_string="    6\t    return MagicMock(return_value=0)",
                new_string="    6\t    return MagicMock(side_effect=[0, 10])")
    after = p.read_text(encoding="utf-8")
    # File must NOT contain `\t6\t` or any line-number prefix artifact
    assert "    6\t" not in after, after
    # Replacement DID happen
    assert "side_effect=[0, 10]" in after
    # Original target removed
    assert "return_value=0)" not in after


def test_clean_old_clean_new_unchanged():
    """When neither side has prefixes, behavior is identical to before."""
    tmp, p, reg = _setup("def foo(): return 1\n")
    _edit(reg, path=p.name,
          old_string="return 1",
          new_string="return 2")
    after = p.read_text(encoding="utf-8")
    assert "return 2" in after
    assert "return 1" not in after


def test_only_old_has_prefix_new_string_unchanged():
    """If only old_string has prefixes (e.g. model copied from read_file
    for the search target but typed the replacement fresh), new_string
    is left as-is."""
    tmp, p, reg = _setup("def foo(): return 1\n")
    _edit(reg, path=p.name,
          old_string="    1\tdef foo(): return 1",
          new_string="def foo(): return 42")
    after = p.read_text(encoding="utf-8")
    assert "def foo(): return 42" in after
    assert "    1\t" not in after


def test_replace_all_with_prefixed_new_string():
    """The replace_all path must also strip new_string prefixes."""
    tmp, p, reg = _setup("x = 1\nx = 1\n")
    _edit(reg, path=p.name,
          old_string="    1\tx = 1",
          new_string="    1\ty = 2",
          replace_all=True)
    after = p.read_text(encoding="utf-8")
    assert "y = 2" in after
    assert "    1\t" not in after


def test_multi_line_new_string_with_prefixes_all_stripped():
    """If new_string has multiple line-number-prefixed lines, all of them
    get cleaned."""
    tmp, p, reg = _setup("def foo():\n    pass\n")
    _edit(reg, path=p.name,
          old_string="    pass",
          new_string="    1\tx = 1\n    2\ty = 2\n    3\treturn x + y")
    after = p.read_text(encoding="utf-8")
    # No line-number artifacts
    import re
    assert not re.search(r"^\s*\d+\t", after, re.MULTILINE), after


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
