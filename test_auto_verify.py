"""
Tests for engine.agent.Agent._verify_* and the _maybe_auto_verify dispatcher.

Each verifier handles two signals: a known-good file returns success=True
with a "syntax OK" / "valid ..." content; a known-bad file returns
success=False with an error message mentioning the filename. External-tool
verifiers (node/rustc/go/bash) must also SKIP CLEANLY (return None) when
the toolchain isn't installed, so the bench never fails on a missing dep.

Run: python -m pytest test_auto_verify.py -v
     OR just: python test_auto_verify.py
"""
from __future__ import annotations

import shutil
import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).parent
sys.path.insert(0, str(ROOT))

_CORE_ROOT = ROOT.parent / "densanon-core"
if _CORE_ROOT.exists() and str(_CORE_ROOT) not in sys.path:
    sys.path.insert(0, str(_CORE_ROOT))

from engine.agent import Agent
from engine.agent_tools import ToolCall


def _write(tmp: Path, name: str, body: str) -> Path:
    p = tmp / name
    p.write_text(body, encoding="utf-8")
    return p


# ── per-language verifier tests ──

def test_verify_python_good():
    with tempfile.TemporaryDirectory() as td:
        p = _write(Path(td), "ok.py", "def f():\n    return 1\n")
        r = Agent._verify_python(p)
        assert r.success, r.error
        assert "syntax OK" in r.content
        assert "ok.py" in r.content


def test_verify_python_syntax_error():
    with tempfile.TemporaryDirectory() as td:
        p = _write(Path(td), "bad.py", "def f(\n")
        r = Agent._verify_python(p)
        assert not r.success
        assert "SyntaxError" in r.error
        assert "bad.py" in r.error


def test_verify_python_undefined_name():
    # Syntactically valid but references an undefined name — caught by
    # the static undefined-names check piggybacked on _verify_python.
    with tempfile.TemporaryDirectory() as td:
        p = _write(Path(td), "undef.py", "def g():\n    return Frobozz()\n")
        r = Agent._verify_python(p)
        assert not r.success
        assert "Frobozz" in r.error


def test_verify_json_good():
    with tempfile.TemporaryDirectory() as td:
        p = _write(Path(td), "data.json", '{"a": [1, 2, 3], "b": null}')
        r = Agent._verify_json(p)
        assert r.success, r.error
        assert "valid JSON" in r.content


def test_verify_json_bad():
    with tempfile.TemporaryDirectory() as td:
        p = _write(Path(td), "broken.json", '{"a": [1, 2, ,]}')
        r = Agent._verify_json(p)
        assert not r.success
        assert "broken.json" in r.error


def test_verify_node_good():
    if shutil.which("node") is None:
        # Skip cleanly — mirrors verifier's own None return on missing tool
        return
    with tempfile.TemporaryDirectory() as td:
        p = _write(Path(td), "ok.js", "function f() { return 1; }\nf();\n")
        r = Agent._verify_node(p)
        assert r is not None
        assert r.success, r.error


def test_verify_node_bad():
    if shutil.which("node") is None:
        return
    with tempfile.TemporaryDirectory() as td:
        p = _write(Path(td), "bad.js", "function f( {\n")
        r = Agent._verify_node(p)
        assert r is not None
        assert not r.success
        assert "bad.js" in r.error


def test_verify_node_missing_toolchain():
    # If node isn't on PATH, the verifier must return None so the agent
    # loop can skip the auto-verify without reporting a failure.
    if shutil.which("node") is not None:
        return  # can only test the None path when node is absent
    with tempfile.TemporaryDirectory() as td:
        p = _write(Path(td), "ok.js", "const x = 1;\n")
        assert Agent._verify_node(p) is None


def test_verify_bash_good():
    if shutil.which("bash") is None:
        return
    with tempfile.TemporaryDirectory() as td:
        p = _write(Path(td), "ok.sh", "#!/bin/bash\nfor i in 1 2 3; do\n  echo $i\ndone\n")
        r = Agent._verify_bash(p)
        assert r is not None
        assert r.success, r.error


def test_verify_bash_bad():
    if shutil.which("bash") is None:
        return
    with tempfile.TemporaryDirectory() as td:
        # Unterminated `if` block — bash -n catches this
        p = _write(Path(td), "bad.sh", "#!/bin/bash\nif [ 1 -eq 1 ]; then\necho missing fi\n")
        r = Agent._verify_bash(p)
        assert r is not None
        assert not r.success
        assert "bad.sh" in r.error


def test_verify_yaml_good():
    try:
        import yaml  # noqa: F401
    except ImportError:
        return
    with tempfile.TemporaryDirectory() as td:
        p = _write(Path(td), "ok.yml", "name: foo\nversion: 1\nitems:\n  - a\n  - b\n")
        r = Agent._verify_yaml(p)
        assert r is not None
        assert r.success, r.error


def test_verify_yaml_bad():
    try:
        import yaml  # noqa: F401
    except ImportError:
        return
    with tempfile.TemporaryDirectory() as td:
        # Inconsistent indent — YAML error
        p = _write(Path(td), "bad.yml", "name: foo\n  wrong: indent\nitems:\n")
        r = Agent._verify_yaml(p)
        assert r is not None
        assert not r.success
        assert "bad.yml" in r.error


# ── dispatcher tests ──


class _StubModel:
    def generate(self, *_a, **_k): return ""


def _agent(ws: Path) -> Agent:
    from engine.agent_tools import ToolRegistry
    return Agent(
        model=_StubModel(),
        registry=ToolRegistry(),
        workspace_root=ws,
        auto_verify_python=True,
    )


def _fake_result():
    from engine.agent_tools import ToolResult
    return ToolResult(name="write_file", success=True, content="ok")


def test_dispatcher_python():
    with tempfile.TemporaryDirectory() as td:
        ws = Path(td)
        _write(ws, "x.py", "x = 1\n")
        call = ToolCall(name="write_file", arguments={"path": "x.py"}, raw="")
        r = _agent(ws)._maybe_auto_verify(call, _fake_result())
        assert r is not None and r.success


def test_dispatcher_json():
    with tempfile.TemporaryDirectory() as td:
        ws = Path(td)
        _write(ws, "x.json", '{"a": 1}')
        call = ToolCall(name="write_file", arguments={"path": "x.json"}, raw="")
        r = _agent(ws)._maybe_auto_verify(call, _fake_result())
        assert r is not None and r.success


def test_dispatcher_unknown_ext_returns_none():
    with tempfile.TemporaryDirectory() as td:
        ws = Path(td)
        _write(ws, "x.md", "# Hello\n")
        call = ToolCall(name="write_file", arguments={"path": "x.md"}, raw="")
        r = _agent(ws)._maybe_auto_verify(call, _fake_result())
        assert r is None


def test_dispatcher_bash_ext_routed():
    if shutil.which("bash") is None:
        return
    with tempfile.TemporaryDirectory() as td:
        ws = Path(td)
        _write(ws, "x.sh", "#!/bin/bash\necho hi\n")
        call = ToolCall(name="write_file", arguments={"path": "x.sh"}, raw="")
        r = _agent(ws)._maybe_auto_verify(call, _fake_result())
        assert r is not None and r.success


def test_dispatcher_yaml_ext_routed():
    try:
        import yaml  # noqa: F401
    except ImportError:
        return
    with tempfile.TemporaryDirectory() as td:
        ws = Path(td)
        _write(ws, "x.yml", "a: 1\nb: 2\n")
        call = ToolCall(name="write_file", arguments={"path": "x.yml"}, raw="")
        r = _agent(ws)._maybe_auto_verify(call, _fake_result())
        assert r is not None and r.success


def test_dispatcher_non_write_call_returns_none():
    with tempfile.TemporaryDirectory() as td:
        ws = Path(td)
        _write(ws, "x.py", "x = 1\n")
        call = ToolCall(name="read_file", arguments={"path": "x.py"}, raw="")
        r = _agent(ws)._maybe_auto_verify(call, _fake_result())
        assert r is None


def test_dispatcher_disabled_flag_returns_none():
    from engine.agent_tools import ToolRegistry
    with tempfile.TemporaryDirectory() as td:
        ws = Path(td)
        _write(ws, "x.py", "x = 1\n")
        a = Agent(
            model=_StubModel(),
            registry=ToolRegistry(),
            workspace_root=ws,
            auto_verify_python=False,
        )
        call = ToolCall(name="write_file", arguments={"path": "x.py"}, raw="")
        assert a._maybe_auto_verify(call, _fake_result()) is None


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
            print(f"  ERR   {name}: {type(e).__name__}: {e}")
            failed += 1
    print(f"\n{passed} passed, {failed} failed")
    sys.exit(0 if failed == 0 else 1)
