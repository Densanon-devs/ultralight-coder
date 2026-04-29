"""
Tests for the edit_file empty-old_string guard added 2026-04-26 after the
self-proof test catastrophically wiped a test file.

Background:
  Earlier code path: edit_file with empty old_string AND new_string
  containing both imports + def/class auto-routed to "rewrite the whole
  file." Intent was to forgive a common model mistake. But the same
  pattern fires when the model intends a targeted edit and chooses
  empty old_string as a shortcut — silently destroying the file.

  Now: rejected with a clear error directing the model to either
  write_file (for intentional rewrite) or edit_file with a real anchor
  (for targeted edit). The pure-import prepend path is preserved.
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


def _setup(initial_content: str = "x = 1\n", name: str = "x.py"):
    tmp = Path(tempfile.mkdtemp())
    p = tmp / name
    p.write_text(initial_content, encoding="utf-8")
    reg = build_default_registry(tmp)
    return tmp, p, reg


def _edit(reg, **kw):
    return reg.get("edit_file").function(**kw)


# ── Pure-import prepend still works ──────────────────────────────


def test_pure_import_prepend_works():
    tmp, p, reg = _setup("x = 1\n")
    out = _edit(reg, path=p.name, old_string="",
                new_string="from typing import List\n")
    after = p.read_text(encoding="utf-8")
    assert "from typing import List" in after
    assert "x = 1" in after, "original content must remain"


def test_multiple_imports_prepend_works():
    tmp, p, reg = _setup("def f(): pass\n")
    out = _edit(reg, path=p.name, old_string="",
                new_string="import json\nfrom dataclasses import dataclass\n")
    after = p.read_text(encoding="utf-8")
    assert "import json" in after
    assert "from dataclasses import dataclass" in after
    assert "def f(): pass" in after


def test_shebang_prepend_works():
    tmp, p, reg = _setup("print('hi')\n", name="run.py")
    out = _edit(reg, path=p.name, old_string="",
                new_string="#!/usr/bin/env python3\n")
    after = p.read_text(encoding="utf-8")
    assert after.startswith("#!/usr/bin/env python3")
    assert "print('hi')" in after


# ── Multi-construct rejection (the catastrophe-prevention path) ──


def test_imports_plus_def_rejects_empty_old_string():
    """The exact pattern from the 2026-04-26 self-proof catastrophe:
    empty old_string + (imports + function definition). Used to
    silently rewrite; now must reject with helpful error."""
    tmp, p, reg = _setup("def existing_test(): assert True\n", name="test_x.py")
    original = p.read_text(encoding="utf-8")
    try:
        _edit(reg, path=p.name, old_string="",
              new_string=(
                  "from unittest.mock import patch\n"
                  "import time\n"
                  "\n"
                  "def mock_time():\n"
                  "    return 0\n"
              ))
    except ValueError as e:
        msg = str(e)
        assert "Ambiguous" in msg or "ambiguous" in msg.lower()
        assert "write_file" in msg
        # File must NOT have been modified
        assert p.read_text(encoding="utf-8") == original
        return
    assert False, "should have rejected"


def test_imports_plus_class_rejects_empty_old_string():
    tmp, p, reg = _setup("CONST = 1\n")
    original = p.read_text(encoding="utf-8")
    try:
        _edit(reg, path=p.name, old_string="",
              new_string=(
                  "from dataclasses import dataclass\n"
                  "\n"
                  "@dataclass\n"
                  "class Foo:\n"
                  "    x: int\n"
              ))
    except ValueError as e:
        assert "write_file" in str(e)
        assert p.read_text(encoding="utf-8") == original
        return
    assert False


def test_main_guard_rejects_empty_old_string():
    """A snippet with `if __name__ == '__main__':` is a full module,
    not a prepend candidate."""
    tmp, p, reg = _setup("def helper(): pass\n", name="cli.py")
    original = p.read_text(encoding="utf-8")
    try:
        _edit(reg, path=p.name, old_string="",
              new_string=(
                  "import argparse\n"
                  "\n"
                  "if __name__ == '__main__':\n"
                  "    parser = argparse.ArgumentParser()\n"
                  "    args = parser.parse_args()\n"
              ))
    except ValueError as e:
        assert "write_file" in str(e)
        assert p.read_text(encoding="utf-8") == original
        return
    assert False


def test_pure_def_without_imports_does_not_trigger_rewrite():
    """A bare `def` snippet (no imports) was not part of the multi-
    construct pattern. The rejection should NOT fire here — it goes
    through the normal small-prepend code path."""
    tmp, p, reg = _setup("# header\n")
    out = _edit(reg, path=p.name, old_string="",
                new_string="def helper():\n    return 1\n")
    after = p.read_text(encoding="utf-8")
    assert "def helper" in after
    assert "# header" in after, "original content must be preserved"


# ── Targeted edits with real anchors — unchanged ─────────────────


def test_targeted_edit_with_real_anchor_works():
    tmp, p, reg = _setup("def add(a, b): return a + b\n")
    out = _edit(reg, path=p.name,
                old_string="def add(a, b): return a + b",
                new_string="def add(a, b, c=0): return a + b + c")
    after = p.read_text(encoding="utf-8")
    assert "def add(a, b, c=0)" in after
    assert "def add(a, b):" not in after


# ── Error message quality ────────────────────────────────────────


def test_rejection_mentions_both_write_file_and_anchor_paths():
    """The error message must guide the model to BOTH valid recoveries:
    write_file (full rewrite) and edit_file with an anchor (targeted)."""
    tmp, p, reg = _setup("def f(): pass\n")
    try:
        _edit(reg, path=p.name, old_string="",
              new_string="import x\ndef g(): pass\n")
    except ValueError as e:
        msg = str(e).lower()
        assert "write_file" in msg
        assert "old_string" in msg or "anchor" in msg
        return
    assert False


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
