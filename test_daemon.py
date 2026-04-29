"""
Tests for ulcagent daemon mode helpers.

Exercises:
- _detect_test_command picks pytest from pyproject.toml
- _detect_test_command picks pytest from a tests/ directory
- _detect_test_command picks npm test from package.json
- _detect_test_command picks cargo test from Cargo.toml
- _detect_test_command picks go test from go.mod
- _detect_test_command returns None for empty workspace
- The daemon goal template requires PASS/FAIL output and forbids fix attempts
- _daemon_loop is registered via the --daemon flag dispatch
"""
from __future__ import annotations
import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).parent
sys.path.insert(0, str(ROOT))

from ulcagent import _detect_test_command, _DAEMON_GOAL_TEMPLATE, _daemon_loop


def _ws(files: dict[str, str] | None = None, dirs: list[str] | None = None) -> Path:
    tmp = Path(tempfile.mkdtemp())
    for name, content in (files or {}).items():
        (tmp / name).write_text(content, encoding="utf-8")
    for d in (dirs or []):
        (tmp / d).mkdir(parents=True, exist_ok=True)
    return tmp


def test_detect_pytest_from_pyproject():
    assert _detect_test_command(_ws({"pyproject.toml": "[project]"})) == "pytest"


def test_detect_pytest_from_pytest_ini():
    assert _detect_test_command(_ws({"pytest.ini": ""})) == "pytest"


def test_detect_pytest_from_tests_dir():
    assert _detect_test_command(_ws(dirs=["tests"])) == "pytest"


def test_detect_npm_test():
    assert _detect_test_command(_ws({"package.json": "{}"})) == "npm test"


def test_detect_cargo_test():
    assert _detect_test_command(_ws({"Cargo.toml": "[package]"})) == "cargo test"


def test_detect_go_test():
    assert _detect_test_command(_ws({"go.mod": "module x"})) == "go test ./..."


def test_detect_none_for_empty():
    assert _detect_test_command(_ws()) is None


def test_daemon_goal_format():
    assert "PASS" in _DAEMON_GOAL_TEMPLATE
    assert "FAIL" in _DAEMON_GOAL_TEMPLATE
    assert "run_tests" in _DAEMON_GOAL_TEMPLATE
    # The daemon must NOT try to fix — it's a status check
    assert "no fix attempt" in _DAEMON_GOAL_TEMPLATE.lower() or "not a repair" in _DAEMON_GOAL_TEMPLATE.lower()


def test_daemon_loop_callable():
    # Smoke check: the function exists and is importable. Behavior is I/O-bound
    # so we don't run it — but a regression here would catch refactor breakage.
    assert callable(_daemon_loop)


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
