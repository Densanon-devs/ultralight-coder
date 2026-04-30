"""
Tests for the auto-flag-on-fail hook in ulcagent and benchmark_agentic.

Exercises:
- _maybe_auto_flag is importable from ulcagent and is callable
- _maybe_auto_flag is a no-op on None / clean results / missing modules
- _maybe_auto_flag writes YAML files + prints a summary on flagged results
- benchmark_agentic CLI accepts --auto-flag flag
- benchmark_agentic.run_one_task accepts auto_flag kwarg
"""
from __future__ import annotations
import io
import sys
import tempfile
from pathlib import Path
from types import SimpleNamespace
from contextlib import redirect_stdout

ROOT = Path(__file__).parent
sys.path.insert(0, str(ROOT))

_CORE_ROOT = ROOT.parent / "densanon-core"
if _CORE_ROOT.exists() and str(_CORE_ROOT) not in sys.path:
    sys.path.insert(0, str(_CORE_ROOT))

from ulcagent import _maybe_auto_flag, _SELF


def _stub_call(name, **args): return SimpleNamespace(name=name, arguments=args)
def _stub_ok(content=""):     return SimpleNamespace(success=True, content=content, error="")
def _stub_err(content="", error=""): return SimpleNamespace(success=False, content=content, error=error)


def test_maybe_auto_flag_is_callable():
    assert callable(_maybe_auto_flag)


def test_maybe_auto_flag_handles_none_silently():
    buf = io.StringIO()
    with redirect_stdout(buf):
        _maybe_auto_flag(None, "any goal")
    assert buf.getvalue() == ""


def test_maybe_auto_flag_silent_on_clean_result():
    """Clean run (no failures) should produce no output and no YAML files."""
    res = SimpleNamespace(
        final_answer="Done.",
        tool_calls=[_stub_call("write_file", path="x.py", content=["pass"])],
        tool_results=[_stub_ok(content="Wrote 4 chars to x.py\nsyntax OK (x.py)")],
        stop_reason="answered",
    )
    buf = io.StringIO()
    with redirect_stdout(buf):
        _maybe_auto_flag(res, "Add a placeholder file")
    assert "auto-flagged" not in buf.getvalue()


def test_maybe_auto_flag_writes_yaml_on_flagged_result(tmp_path=None, monkeypatch=None):
    """When a known pattern fires, the hook should print one summary line
    AND a YAML file should land under data/augmentor_examples/_auto_generated/.
    Uses a temp _SELF override to avoid polluting the real repo."""
    import ulcagent as _ula
    real_self = _ula._SELF
    tmp = Path(tempfile.mkdtemp())
    _ula._SELF = tmp
    try:
        res = SimpleNamespace(
            final_answer="",
            tool_calls=[_stub_call("write_file", path="storage.py", content=["..."])],
            tool_results=[_stub_err(error="bare JSON tool call failed to parse: Expecting ',' delimiter")],
            stop_reason="answered",
        )
        buf = io.StringIO()
        with redirect_stdout(buf):
            _maybe_auto_flag(res, "Build a todo CLI")
        out = buf.getvalue()
        assert "auto-flagged" in out
        assert "json_quote_leak" in out
        # YAML file landed in the review dir (NOT under augmentor_examples/)
        yaml_files = list((tmp / "data" / "auto_generated_review").rglob("*.yaml"))
        assert len(yaml_files) >= 1
        assert not (tmp / "data" / "augmentor_examples").exists() \
            or not list((tmp / "data" / "augmentor_examples").rglob("*.yaml")), \
            "auto-flag must NOT write into the retrieval scan path"
    finally:
        _ula._SELF = real_self


def test_maybe_auto_flag_swallows_internal_exceptions():
    """If the failure_flagger / yaml_builder modules raise, the hook must
    not propagate — it's a side-channel feature, not a critical path."""
    bad = SimpleNamespace(
        # Missing tool_calls / tool_results / final_answer attributes
    )
    buf = io.StringIO()
    with redirect_stdout(buf):
        _maybe_auto_flag(bad, "any")
    # Should not raise; output may be empty


def test_bench_cli_accepts_auto_flag_flag():
    """benchmark_agentic --help should mention --auto-flag."""
    import subprocess
    res = subprocess.run(
        [sys.executable, "benchmark_agentic.py", "--help"],
        capture_output=True, text=True, timeout=15, cwd=str(ROOT),
    )
    assert "--auto-flag" in res.stdout, res.stdout


def test_bench_run_one_task_accepts_auto_flag_kwarg():
    import inspect
    from benchmark_agentic import run_one_task
    sig = inspect.signature(run_one_task)
    assert "auto_flag" in sig.parameters
    assert sig.parameters["auto_flag"].default is False


def test_bench_cli_accepts_auto_promote_flag():
    """benchmark_agentic --help should mention --auto-promote."""
    import subprocess
    res = subprocess.run(
        [sys.executable, "benchmark_agentic.py", "--help"],
        capture_output=True, text=True, timeout=15, cwd=str(ROOT),
    )
    assert "--auto-promote" in res.stdout, res.stdout


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
