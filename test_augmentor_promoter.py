"""
Tests for engine/augmentor_promoter.py — validate + copy auto-generated
YAMLs from the review queue into the retrieval index.
"""
from __future__ import annotations
import sys
import tempfile
from pathlib import Path

import yaml

ROOT = Path(__file__).parent
sys.path.insert(0, str(ROOT))

_CORE_ROOT = ROOT.parent / "densanon-core"
if _CORE_ROOT.exists() and str(_CORE_ROOT) not in sys.path:
    sys.path.insert(0, str(_CORE_ROOT))

from engine.augmentor_promoter import (
    REVIEW_DIR_NAME, PROMOTED_PARENT,
    list_review_queue, promote_file, promote_category, promote_all, summarize,
)


def _make_workspace():
    tmp = Path(tempfile.mkdtemp())
    (tmp / "data" / REVIEW_DIR_NAME).mkdir(parents=True)
    return tmp


def _good_yaml(path: Path, query="any goal", solution=None):
    path.parent.mkdir(parents=True, exist_ok=True)
    if solution is None:
        # Make each helper call produce a UNIQUE solution body so the
        # dedup pass doesn't accidentally collapse distinct test fixtures.
        # Tests that specifically want duplicate bodies pass solution
        # explicitly.
        solution = f"do this for {path.name}"
    path.write_text(yaml.safe_dump({
        "domain": "agentic",
        "category": path.parent.name,
        "examples": [{
            "query": query,
            "solution": solution,
            "tags": ["test"],
        }],
    }))


def test_empty_queue_returns_empty():
    tmp = _make_workspace()
    assert list_review_queue(tmp) == {}


def test_list_review_queue_groups_by_category():
    tmp = _make_workspace()
    _good_yaml(tmp / "data" / REVIEW_DIR_NAME / "json_quote_leak" / "a.yaml")
    _good_yaml(tmp / "data" / REVIEW_DIR_NAME / "json_quote_leak" / "b.yaml")
    _good_yaml(tmp / "data" / REVIEW_DIR_NAME / "missing_import" / "c.yaml")
    queue = list_review_queue(tmp)
    assert set(queue.keys()) == {"json_quote_leak", "missing_import"}
    assert len(queue["json_quote_leak"]) == 2
    assert len(queue["missing_import"]) == 1


def test_promote_file_copies_and_keeps_original():
    tmp = _make_workspace()
    src = tmp / "data" / REVIEW_DIR_NAME / "json_quote_leak" / "x.yaml"
    _good_yaml(src)
    r = promote_file(src, tmp)
    assert r.promoted, r.reason
    assert src.exists(), "original must remain in review queue as audit trail"
    assert r.target.exists()
    assert "augmentor_examples" in str(r.target)
    assert PROMOTED_PARENT in str(r.target)


def test_promote_skips_already_promoted_unless_force():
    tmp = _make_workspace()
    src = tmp / "data" / REVIEW_DIR_NAME / "json_quote_leak" / "x.yaml"
    _good_yaml(src)
    r1 = promote_file(src, tmp)
    assert r1.promoted
    r2 = promote_file(src, tmp)
    assert not r2.promoted
    assert "already promoted" in r2.reason
    r3 = promote_file(src, tmp, force=True)
    assert r3.promoted, r3.reason


def test_promote_rejects_invalid_yaml():
    tmp = _make_workspace()
    src = tmp / "data" / REVIEW_DIR_NAME / "json_quote_leak" / "bad.yaml"
    src.parent.mkdir(parents=True, exist_ok=True)
    src.write_text(yaml.safe_dump({"domain": "agentic", "examples": [{"query": ""}]}))
    r = promote_file(src, tmp)
    assert not r.promoted
    assert "validation failed" in r.reason


def test_promote_category_processes_all_files():
    tmp = _make_workspace()
    cat = tmp / "data" / REVIEW_DIR_NAME / "json_quote_leak"
    _good_yaml(cat / "a.yaml")
    _good_yaml(cat / "b.yaml")
    _good_yaml(cat / "c.yaml")
    results = promote_category("json_quote_leak", tmp)
    assert len(results) == 3
    assert all(r.promoted for r in results)


def test_promote_all_walks_every_category():
    tmp = _make_workspace()
    _good_yaml(tmp / "data" / REVIEW_DIR_NAME / "json_quote_leak" / "a.yaml")
    _good_yaml(tmp / "data" / REVIEW_DIR_NAME / "missing_import" / "b.yaml")
    _good_yaml(tmp / "data" / REVIEW_DIR_NAME / "fstring_nested_quote" / "c.yaml")
    results = promote_all(tmp)
    assert len(results) == 3
    counts = summarize(results)
    assert counts["promoted"] == 3
    assert counts["rejected"] == 0


def test_promoted_lands_in_retrieval_scan_path():
    """Promoted YAMLs MUST land in data/augmentor_examples/<...>/ so they
    get picked up by densanon.core.example_loader.load_all_examples."""
    tmp = _make_workspace()
    src = tmp / "data" / REVIEW_DIR_NAME / "json_quote_leak" / "x.yaml"
    _good_yaml(src)
    r = promote_file(src, tmp)
    assert r.promoted
    augmentor_root = (tmp / "data" / "augmentor_examples").resolve()
    assert augmentor_root in r.target.resolve().parents


def test_summarize_counts_by_outcome():
    from engine.augmentor_promoter import PromotionResult
    results = [
        PromotionResult(Path("a"), Path("a2"), True, "ok"),
        PromotionResult(Path("b"), Path("b2"), True, "ok"),
        PromotionResult(Path("c"), None, False, "validation failed: missing query"),
        PromotionResult(Path("d"), None, False, "already promoted"),
    ]
    counts = summarize(results)
    assert counts == {"promoted": 2, "rejected": 1, "skipped": 1}


# ── Dedup tests ────────────────────────────────────────────────────


def test_dedup_skips_identical_solution_body():
    """Two YAMLs with the same solution body should not both promote —
    the second is a redundant retrieval-index entry."""
    tmp = _make_workspace()
    same_body = "WRONG x. RIGHT y."
    a = tmp / "data" / REVIEW_DIR_NAME / "json_quote_leak" / "a.yaml"
    b = tmp / "data" / REVIEW_DIR_NAME / "json_quote_leak" / "b.yaml"
    _good_yaml(a, query="goal A", solution=same_body)
    _good_yaml(b, query="goal B", solution=same_body)
    # First promotes; second should be skipped as duplicate
    from engine.augmentor_promoter import promote_file, _existing_solution_hashes
    existing = _existing_solution_hashes(tmp)
    r1 = promote_file(a, tmp, skip_replay=True, _existing_hashes=existing)
    r2 = promote_file(b, tmp, skip_replay=True, _existing_hashes=existing)
    assert r1.promoted, r1.reason
    assert not r2.promoted
    assert "duplicate" in r2.reason.lower()


def test_dedup_passes_distinct_solutions():
    """Different solution bodies must NOT be deduped against each other."""
    tmp = _make_workspace()
    a = tmp / "data" / REVIEW_DIR_NAME / "json_quote_leak" / "a.yaml"
    b = tmp / "data" / REVIEW_DIR_NAME / "json_quote_leak" / "b.yaml"
    _good_yaml(a, query="goal", solution="WRONG x. RIGHT y. solution one.")
    _good_yaml(b, query="goal", solution="WRONG x. RIGHT z. solution two.")
    from engine.augmentor_promoter import promote_file, _existing_solution_hashes
    existing = _existing_solution_hashes(tmp)
    r1 = promote_file(a, tmp, skip_replay=True, _existing_hashes=existing)
    r2 = promote_file(b, tmp, skip_replay=True, _existing_hashes=existing)
    assert r1.promoted
    assert r2.promoted, r2.reason


def test_promote_all_dedups_within_batch():
    """Three near-duplicate YAMLs in one batch should produce 1 promoted
    + 2 deduped, not 3 promoted."""
    tmp = _make_workspace()
    body = "Same WRONG. Same RIGHT."
    for n in ("a", "b", "c"):
        _good_yaml(tmp / "data" / REVIEW_DIR_NAME / "json_quote_leak" / f"{n}.yaml",
                   query=f"q{n}", solution=body)
    results = promote_all(tmp)
    counts = summarize(results)
    assert counts["promoted"] == 1, counts
    assert counts["skipped"] == 2, counts


def test_skip_dedup_flag_promotes_duplicates():
    """Opt-out path — if you really want every duplicate."""
    tmp = _make_workspace()
    body = "Same WRONG. Same RIGHT."
    _good_yaml(tmp / "data" / REVIEW_DIR_NAME / "json_quote_leak" / "a.yaml",
               query="qa", solution=body)
    _good_yaml(tmp / "data" / REVIEW_DIR_NAME / "json_quote_leak" / "b.yaml",
               query="qb", solution=body)
    results = promote_all(tmp, skip_dedup=True)
    counts = summarize(results)
    assert counts["promoted"] == 2


def test_dedup_against_existing_promoted_yamls():
    """Already-promoted YAMLs are part of the dedup baseline. A new
    candidate with the same body as a previously-promoted file is
    skipped, not re-promoted."""
    tmp = _make_workspace()
    body = "WRONG. RIGHT. canonical body."
    # Pre-existing promoted file
    existing_path = (tmp / "data" / "augmentor_examples" / "agentic"
                     / "json_quote_leak" / "old.yaml")
    _good_yaml(existing_path, query="old query", solution=body)
    # New candidate in review with the same body
    new_path = (tmp / "data" / REVIEW_DIR_NAME / "json_quote_leak" / "new.yaml")
    _good_yaml(new_path, query="new query", solution=body)

    results = promote_all(tmp)
    counts = summarize(results)
    assert counts["promoted"] == 0, counts
    assert counts["skipped"] == 1, counts


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
