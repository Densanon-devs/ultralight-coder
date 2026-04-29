"""
Regression tests for the large-mode agentic-keyword gate (added 2026-04-26).

Phase 13 introduced `_LARGE_MODE_KEEP_KEYWORDS` to prevent the 14B from
being dragged off-canon by general-Python augmentor injection. That gate
also blocked the auto-generated agentic YAMLs (json_array_form_for_multiline_writes,
fstring_nested_quotes_python310, prepend_missing_import_after_write) from
ever reaching retrieval — defeating the self-improving pipeline.

The fix: a separate `_LARGE_MODE_AGENTIC_KEYWORDS` list. These tests
anchor the boundary so future gate edits don't accidentally re-block
agentic retrieval, AND so the decorator-style off-canon-drag protection
stays in place.
"""
from __future__ import annotations
import sys
from pathlib import Path

ROOT = Path(__file__).parent
sys.path.insert(0, str(ROOT))

_CORE = ROOT.parent / "densanon-core"
if _CORE.exists() and str(_CORE) not in sys.path:
    sys.path.insert(0, str(_CORE))

from engine.augmentors import AugmentorRouter


def _router_in_large_mode():
    r = AugmentorRouter(yaml_dir="data/augmentor_examples")
    r._large_mode = True  # don't load embeddings, just exercise the gate
    return r


# ── Agentic-task signals that SHOULD now allow augmentation ───────


def test_build_a_phrase_allows_augmentation():
    r = _router_in_large_mode()
    assert r._large_mode_should_augment("Build a minimal todo CLI from scratch")


def test_scaffold_allows_augmentation():
    r = _router_in_large_mode()
    assert r._large_mode_should_augment("Scaffold a FastAPI service with auth")


def test_tool_call_keyword_allows_augmentation():
    r = _router_in_large_mode()
    assert r._large_mode_should_augment(
        "How do I emit a write_file tool_call for a multi-line python file"
    )


def test_failure_shape_signals_allow_augmentation():
    r = _router_in_large_mode()
    assert r._large_mode_should_augment(
        "auto_verify reports references undefined names: json"
    )
    assert r._large_mode_should_augment(
        "edit_file returned old_string not found"
    )
    assert r._large_mode_should_augment(
        "f-string: expecting '}' on line 14"
    )


def test_argparse_subcommand_signal_allows_augmentation():
    r = _router_in_large_mode()
    assert r._large_mode_should_augment(
        "Add cli.py with argparse subcommands for add, list, done"
    )


# ── Phase 13 protections that MUST still gate out ─────────────────


def test_decorator_query_still_gated_out():
    """Off-canon drag protection from Phase 13 — pure Python pattern
    queries on a 14B should NOT receive augmentor injection."""
    r = _router_in_large_mode()
    assert not r._large_mode_should_augment(
        "Write a Python decorator that counts function calls"
    )


def test_lru_cache_query_still_gated_out():
    r = _router_in_large_mode()
    assert not r._large_mode_should_augment(
        "Implement an LRU cache class with get and put methods"
    )


def test_pure_python_function_query_still_gated_out():
    r = _router_in_large_mode()
    assert not r._large_mode_should_augment(
        "Write a function that validates an email address"
    )


# ── Pre-existing keep-keywords still work ─────────────────────────


def test_pytest_query_still_allowed():
    r = _router_in_large_mode()
    assert r._large_mode_should_augment("Write a pytest fixture for a tmp database")


def test_pandas_query_still_allowed():
    r = _router_in_large_mode()
    assert r._large_mode_should_augment("Read a csv with pandas and filter rows")


# ── Non-Python always allowed (Phase 13 baseline) ─────────────────


def test_javascript_query_always_allowed_in_large_mode():
    r = _router_in_large_mode()
    assert r._large_mode_should_augment("Write a TypeScript interface for a User model")


def test_rust_query_always_allowed_in_large_mode():
    r = _router_in_large_mode()
    # Use a phrase the language detector recognizes: 'write a rust ...'
    # or any of the standalone signals like 'fn main' / 'cargo'.
    assert r._large_mode_should_augment("Write a rust function that parses input")


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
