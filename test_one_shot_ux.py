"""
Tests for the headless/one-shot UX fixes (--yes flag + spinner TTY check).

Both surfaced in the 2026-04-26 handheld walkthrough:
- run_bash silently denied because no TTY for the confirm prompt.
- Spinner ANSI codes flooded log files in piped runs.
"""
from __future__ import annotations
import sys
import subprocess
from pathlib import Path

ROOT = Path(__file__).parent
sys.path.insert(0, str(ROOT))


def test_yes_flag_in_help_source():
    """The --yes flag must appear in the _HELP_TEXT block under
    Startup flags, so /help (REPL) and ? show it to the user."""
    src = (ROOT / "ulcagent.py").read_text(encoding="utf-8")
    # Grab everything inside the _HELP_TEXT triple-quoted string
    if "_HELP_TEXT = \"\"\"" not in src:
        # alternate single-quote form
        marker = "_HELP_TEXT = '''"
    else:
        marker = "_HELP_TEXT = \"\"\""
    start = src.index(marker)
    end_marker = "\"\"\"" if marker.endswith("\"\"\"") else "'''"
    end = src.index(end_marker, start + len(marker))
    help_text = src[start:end]
    assert "--yes" in help_text, "Startup flags should document --yes"


def test_spinner_disabled_when_stdout_not_tty(monkeypatch=None):
    """The spinner must skip its print loop when stdout isn't a TTY,
    otherwise headless runs accumulate ANSI escape codes per frame."""
    # Reload the module fresh with stdout redirected to a non-TTY pipe
    # to force the isatty() check to return False at construction.
    import io
    import importlib

    # Save original stdout
    real_stdout = sys.stdout
    try:
        # Replace stdout with a non-TTY StringIO
        sys.stdout = io.StringIO()

        # Now reload the ulcagent module so _Spinner picks up the new stdout
        if "ulcagent" in sys.modules:
            ula = importlib.reload(sys.modules["ulcagent"])
        else:
            import ulcagent as ula

        s = ula._Spinner()
        assert s._enabled is False, "spinner must be disabled when stdout is not a TTY"
        # start() should be a no-op
        s.start()
        assert not s._active, "spinner thread should not start in headless mode"
        # stop() should also no-op
        s.stop()  # must not raise
    finally:
        sys.stdout = real_stdout
        # Reload again so subsequent tests see the real-stdout module
        if "ulcagent" in sys.modules:
            importlib.reload(sys.modules["ulcagent"])


def test_yes_flag_path_in_arg_parse():
    """--yes flag should be picked up by ulcagent's auto_yes detection.
    We check the source contains the flag check rather than running the
    full agent (which needs a model). This is a smoke test that prevents
    accidental rename or removal."""
    src = (ROOT / "ulcagent.py").read_text(encoding="utf-8")
    assert "--yes" in src
    assert "auto_yes" in src
    assert "auto-approved risky" in src


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
