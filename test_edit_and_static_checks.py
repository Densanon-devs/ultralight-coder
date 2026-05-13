"""Regression tests for two fixes landed 2026-05-12 (from the parked
iterative-scaffold experiment branch):

1. Agent._static_undefined_names no longer false-positives on lambda /
   comprehension parameters.
2. _edit_file rejects array-form new_string/old_string with a clear
   message (the array form is write_file-only; arrays in edit_file lose
   the function-body indentation and produce IndentationError).
"""
from __future__ import annotations

import tempfile
from pathlib import Path

import pytest

from engine.agent import Agent
from engine.agent_builtins import Workspace, _coerce_text, _edit_file


class TestStaticUndefinedLambdaParams:
    def test_lambda_param_not_flagged(self):
        src = "def f(xs):\n    return sorted(xs, key=lambda x: x['k'])\n"
        assert "x" not in Agent._static_undefined_names(src)

    def test_lambda_multi_params_not_flagged(self):
        src = "def f():\n    g = lambda a, b, *c, **d: a + b + sum(c) + len(d)\n    return g(1, 2)\n"
        undef = Agent._static_undefined_names(src)
        assert undef == set(), f"unexpected undefined names: {undef}"

    def test_comprehension_target_not_flagged(self):
        src = "def f(items):\n    return [y * y for y in items]\n"
        assert "y" not in Agent._static_undefined_names(src)

    def test_nested_lambda_in_comprehension(self):
        src = "def f(rows):\n    return [(lambda z: z + 1)(r) for r in rows]\n"
        undef = Agent._static_undefined_names(src)
        assert "z" not in undef and "r" not in undef

    def test_real_undefined_name_still_caught(self):
        src = "def f():\n    return some_missing_helper()\n"
        assert "some_missing_helper" in Agent._static_undefined_names(src)

    def test_real_undefined_inside_lambda_still_caught(self):
        # A genuinely-undefined name referenced inside a lambda body should
        # still be flagged (the lambda's own params are scoped, but free
        # vars resolved against module/function scope still apply).
        src = "def f(xs):\n    return sorted(xs, key=lambda x: undefined_thing(x))\n"
        assert "undefined_thing" in Agent._static_undefined_names(src)


class TestEditFileArrayRejection:
    def _ws(self, tmp: str, fname: str = "a.py", content: str = "x = 1\n") -> tuple[Workspace, Path]:
        ws = Workspace(root=Path(tmp))
        p = Path(tmp) / fname
        p.write_text(content, encoding="utf-8")
        return ws, p

    def test_array_new_string_rejected(self, tmp_path):
        ws, _ = self._ws(str(tmp_path))
        with pytest.raises(ValueError, match="array"):
            _edit_file(ws, "a.py", old_string="x = 1", new_string=["a = 2", "b = 3"])

    def test_array_old_string_rejected(self, tmp_path):
        ws, _ = self._ws(str(tmp_path))
        with pytest.raises(ValueError, match="array"):
            _edit_file(ws, "a.py", old_string=["x = 1"], new_string="y = 2")

    def test_string_new_string_works(self, tmp_path):
        ws, p = self._ws(str(tmp_path), content="def f():\n    # TODO[1]: impl\n    pass\n")
        _edit_file(ws, "a.py",
                   old_string="    # TODO[1]: impl\n    pass",
                   new_string="    x = 1\n    return x")
        result = p.read_text(encoding="utf-8")
        import ast
        ast.parse(result)  # must compile — indentation preserved
        assert "x = 1" in result and "TODO" not in result

    def test_rejection_message_directs_to_string_form(self, tmp_path):
        ws, _ = self._ws(str(tmp_path))
        with pytest.raises(ValueError) as exc:
            _edit_file(ws, "a.py", old_string="x = 1", new_string=["a"])
        msg = str(exc.value).lower()
        assert "string" in msg
        assert "write_file" in msg  # tells the model where arrays DO work


class TestCoerceTextStillWorks:
    """_coerce_text was extracted from _write_file's inner _coerce — verify
    write_file's array-form content path still works."""

    def test_list_joined_with_newlines(self):
        assert _coerce_text(["a", "b", "c"]) == "a\nb\nc\n"

    def test_none_is_empty(self):
        assert _coerce_text(None) == ""

    def test_string_passthrough(self):
        assert _coerce_text("hello\nworld") == "hello\nworld"

    def test_write_file_array_content_works(self, tmp_path):
        from engine.agent_builtins import _write_file
        ws = Workspace(root=tmp_path)
        _write_file(ws, "out.py", content=["import os", "", "def f():", "    return os.getcwd()"])
        text = (tmp_path / "out.py").read_text(encoding="utf-8")
        import ast
        ast.parse(text)
        assert "import os" in text and "return os.getcwd()" in text
