"""
Tests for engine/replay_validator.py — static + (placeholder) live
validation gate before YAMLs reach the retrieval index.
"""
from __future__ import annotations
import sys
import tempfile
from pathlib import Path

import yaml as yaml_mod

ROOT = Path(__file__).parent
sys.path.insert(0, str(ROOT))

_CORE_ROOT = ROOT.parent / "densanon-core"
if _CORE_ROOT.exists() and str(_CORE_ROOT) not in sys.path:
    sys.path.insert(0, str(_CORE_ROOT))

from engine.replay_validator import (
    validate_static, validate_dir, summarize, validate_live, _required_for,
)


def _yaml(category: str, query: str, solution: str, tmp: Path) -> Path:
    p = tmp / f"{category}.yaml"
    p.write_text(yaml_mod.safe_dump({
        "domain": "agentic",
        "category": category,
        "examples": [{
            "query": query,
            "solution": solution,
            "tags": ["test"],
        }],
    }))
    return p


# ── Schema-failure tests ───────────────────────────────────────────


def test_unloadable_yaml_fails():
    tmp = Path(tempfile.mkdtemp())
    p = tmp / "x.yaml"
    p.write_text("this is :: not :: yaml :: at all\n  -- broken")
    r = validate_static(p)
    assert not r.valid
    assert "yaml" in r.reason.lower() or "load" in r.reason.lower()


def test_missing_examples_field_fails():
    tmp = Path(tempfile.mkdtemp())
    p = tmp / "x.yaml"
    p.write_text(yaml_mod.safe_dump({"domain": "agentic", "category": "x"}))
    r = validate_static(p)
    assert not r.valid
    assert "examples" in r.reason


def test_empty_query_fails():
    tmp = Path(tempfile.mkdtemp())
    p = _yaml("json_array_form_for_multiline_writes", "",
              "WRONG ... RIGHT ... array form ... content ...", tmp)
    r = validate_static(p)
    assert not r.valid
    assert "query" in r.reason


def test_empty_solution_fails():
    tmp = Path(tempfile.mkdtemp())
    p = _yaml("any_category", "some query", "", tmp)
    r = validate_static(p)
    assert not r.valid
    assert "solution" in r.reason


# ── Pattern-match tests ────────────────────────────────────────────


def test_known_category_passes_with_all_patterns():
    tmp = Path(tempfile.mkdtemp())
    p = _yaml("json_array_form_for_multiline_writes",
              "Write multi-line file",
              "Use the array form for content. WRONG: x. RIGHT: y. content array.", tmp)
    r = validate_static(p)
    assert r.valid, r.reason
    assert r.matched_patterns == r.total_patterns


def test_known_category_fails_with_missing_patterns():
    tmp = Path(tempfile.mkdtemp())
    p = _yaml("json_array_form_for_multiline_writes",
              "Write multi-line file",
              "Just some random text without the expected demonstrations.", tmp)
    r = validate_static(p)
    assert not r.valid
    assert "missing patterns" in r.reason


def test_unknown_category_passes_schema_only():
    tmp = Path(tempfile.mkdtemp())
    p = _yaml("hand_crafted_brand_new_category", "any", "any solution body", tmp)
    r = validate_static(p)
    assert r.valid
    assert "no pattern check" in r.reason


def test_required_for_returns_none_for_unknown():
    assert _required_for("nonexistent_category") is None


def test_required_for_returns_list_for_known():
    req = _required_for("json_array_form_for_multiline_writes")
    assert isinstance(req, list)
    assert "WRONG" in req
    assert "RIGHT" in req


# ── Directory walk tests ───────────────────────────────────────────


def test_validate_dir_walks_all_yamls():
    tmp = Path(tempfile.mkdtemp())
    _yaml("json_array_form_for_multiline_writes", "q",
          "array form WRONG RIGHT content", tmp)
    sub = tmp / "sub"
    sub.mkdir()
    _yaml("recover_from_stuck_repeat", "q",
          "stuck_repeat read_file switch", sub)
    results = validate_dir(tmp)
    assert len(results) >= 2


def test_validate_dir_empty_returns_empty():
    tmp = Path(tempfile.mkdtemp())
    assert validate_dir(tmp) == []


def test_summarize_counts():
    tmp = Path(tempfile.mkdtemp())
    _yaml("json_array_form_for_multiline_writes", "q",
          "array form WRONG RIGHT content", tmp)
    _yaml("recover_from_stuck_repeat", "q", "no patterns here", tmp)
    s = summarize(validate_dir(tmp))
    assert s["valid"] == 1
    assert s["invalid"] == 1


# ── Live validator placeholder ─────────────────────────────────────


def test_validate_live_returns_unimplemented():
    """validate_live is a scaffold — should clearly indicate not implemented."""
    r = validate_live("any.yaml", model=None, original_goal="any")
    assert not r.valid
    assert "not implemented" in r.reason.lower()
    assert r.phase == "live"


# ── End-to-end with the real flagger output ────────────────────────


def test_real_flagger_yaml_passes_static_replay():
    """The YAMLs the failure_flagger writes for KNOWN categories must
    pass static replay (otherwise auto-generated YAMLs would fail to
    promote — defeating the whole pipeline)."""
    from engine.failure_flagger import FailureRecord
    from engine.yaml_augmentor_builder import write_yaml
    tmp = Path(tempfile.mkdtemp())
    rec = FailureRecord(
        category="json_quote_leak", iteration=2,
        tool_name="write_file",
        error_excerpt="Expecting ',' delimiter",
        triggering_args={"path": "x.py", "content": ["..."]},
        file_path="x.py",
        suggested_fix="Use array form...",
    )
    p = write_yaml(rec, "Build a todo CLI", tmp)
    assert p is not None
    r = validate_static(p)
    assert r.valid, f"flagger output failed static replay: {r.reason}"


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
