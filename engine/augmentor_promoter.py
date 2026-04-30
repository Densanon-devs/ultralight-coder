"""
Promote auto-generated augmentor YAMLs from the review queue into the
retrieval index.

Pipeline:
  data/auto_generated_review/<category>/*.yaml   ← failure flagger writes here
                            |
                   (schema + replay validation)
                            |
                       (dedup pass)
                            |
  data/augmentor_examples/agentic/<category>/*.yaml   ← retrieval scans here

The original file STAYS in auto_generated_review/ as an audit trail. The
copy in augmentor_examples/agentic/ is what gets loaded by the retrieval
system on next ulcagent boot.

Schema validation runs the same `densanon.core.example_loader.load_examples_from_file`
the retrieval system uses, so any YAML that fails to load there fails to
promote here. Replay validation (engine.replay_validator) checks that the
solution body contains category-specific must-have patterns. The dedup
pass (added 2026-04-26) skips a YAML whose solution body matches an
already-promoted entry — prevents the retrieval index from filling up
with near-duplicates of the same lesson.
"""
from __future__ import annotations

import hashlib
import shutil
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import yaml


REVIEW_DIR_NAME = "auto_generated_review"
PROMOTED_PARENT = "agentic"  # under data/augmentor_examples/


@dataclass
class PromotionResult:
    source: Path
    target: Path | None
    promoted: bool
    reason: str


def _review_root(repo_root: Path) -> Path:
    return repo_root / "data" / REVIEW_DIR_NAME


def _promoted_root(repo_root: Path) -> Path:
    return repo_root / "data" / "augmentor_examples" / PROMOTED_PARENT


def list_review_queue(repo_root: Path | str) -> dict[str, list[Path]]:
    """Group all YAMLs in auto_generated_review/ by category."""
    repo_root = Path(repo_root)
    out: dict[str, list[Path]] = {}
    review = _review_root(repo_root)
    if not review.exists():
        return out
    for cat_dir in sorted(review.iterdir()):
        if not cat_dir.is_dir():
            continue
        files = sorted(cat_dir.glob("*.yaml"))
        if files:
            out[cat_dir.name] = files
    return out


def _validate_yaml(path: Path) -> tuple[bool, str]:
    """Run the candidate through the actual retrieval-side loader to
    catch schema drift before promotion. Returns (ok, reason)."""
    try:
        from densanon.core.example_loader import load_examples_from_file
    except ImportError:
        return False, "densanon.core.example_loader not importable"
    examples = load_examples_from_file(path)
    if not examples:
        return False, "loader returned 0 examples (missing required fields?)"
    for i, ex in enumerate(examples):
        if not ex.get("query"):
            return False, f"example {i}: missing query"
        if not ex.get("solution"):
            return False, f"example {i}: missing solution"
    return True, f"{len(examples)} example(s) parsed cleanly"


def _solution_hash(yaml_path: Path) -> str | None:
    """Stable hash over the example solution bodies in a YAML file.
    Two YAMLs with byte-identical solutions hash the same — that's what
    we use to detect near-duplicates from the auto-flag pipeline (which
    can write multiple YAMLs for the same goal+category)."""
    try:
        data = yaml.safe_load(yaml_path.read_text(encoding="utf-8"))
    except Exception:
        return None
    if not isinstance(data, dict) or "examples" not in data:
        return None
    bodies = [
        (ex.get("solution") or "").strip()
        for ex in data["examples"]
        if isinstance(ex, dict)
    ]
    if not bodies:
        return None
    h = hashlib.sha256()
    for body in bodies:
        h.update(body.encode("utf-8"))
        h.update(b"\x00")  # separator so concatenation isn't ambiguous
    return h.hexdigest()


def _existing_solution_hashes(repo_root: Path) -> set[str]:
    """Walk every YAML already in data/augmentor_examples/agentic/ and
    return the set of solution hashes. New candidates whose hash is in
    this set are dedup-skipped."""
    base = _promoted_root(repo_root)
    if not base.exists():
        return set()
    out: set[str] = set()
    for yaml_path in base.rglob("*.yaml"):
        h = _solution_hash(yaml_path)
        if h is not None:
            out.add(h)
    return out


def promote_file(src: Path, repo_root: Path | str, force: bool = False,
                 skip_replay: bool = False, skip_dedup: bool = False,
                 _existing_hashes: set[str] | None = None) -> PromotionResult:
    """Validate + copy one YAML from review to the live augmentor dir.
    Original stays as audit trail.

    Validation order:
      1. schema check (densanon.core.example_loader can parse it)
      2. static replay (engine.replay_validator must-have patterns)
      3. dedup (skip if a YAML with identical solution body already
         exists under data/augmentor_examples/agentic/)
      4. copy to data/augmentor_examples/agentic/<category>/

    Pass skip_replay=True to bypass step 2.
    Pass skip_dedup=True to bypass step 3 (overrides _existing_hashes).
    `_existing_hashes` is an optional pre-computed set the caller can pass
    when promoting a batch — avoids re-walking the augmentor dir per file.
    """
    repo_root = Path(repo_root)
    src = Path(src)
    if not src.exists():
        return PromotionResult(src, None, False, "source not found")

    # Category from parent dir name
    category = src.parent.name
    target_dir = _promoted_root(repo_root) / category
    target = target_dir / src.name

    if target.exists() and not force:
        return PromotionResult(src, target, False, "already promoted (pass force=True to overwrite)")

    ok, reason = _validate_yaml(src)
    if not ok:
        return PromotionResult(src, None, False, f"validation failed: {reason}")

    if not skip_replay:
        try:
            from engine.replay_validator import validate_static
            replay = validate_static(src)
            if not replay.valid:
                return PromotionResult(src, None, False,
                                       f"validation failed: replay {replay.reason}")
        except ImportError:
            pass  # replay_validator missing — fall through with schema-only

    # `force` means "force everything" — bypass dedup as well as the
    # already-promoted file-existence check. Saves the user from having
    # to pass both `force=True, skip_dedup=True` to re-promote the same
    # file (overwriting an existing copy of itself looks like a duplicate
    # to the dedup pass).
    if not skip_dedup and not force:
        if _existing_hashes is None:
            _existing_hashes = _existing_solution_hashes(repo_root)
        candidate_hash = _solution_hash(src)
        if candidate_hash is not None and candidate_hash in _existing_hashes:
            return PromotionResult(src, None, False,
                                   "duplicate of an already-promoted YAML "
                                   "(pass skip_dedup=True or force=True to override)")

    target_dir.mkdir(parents=True, exist_ok=True)
    shutil.copy2(src, target)
    # Track this hash so subsequent calls in the same batch dedup against it.
    if not skip_dedup and _existing_hashes is not None:
        new_hash = _solution_hash(src)
        if new_hash is not None:
            _existing_hashes.add(new_hash)
    return PromotionResult(src, target, True, reason)


def promote_category(category: str, repo_root: Path | str, force: bool = False,
                     skip_dedup: bool = False) -> list[PromotionResult]:
    """Promote every YAML in one category."""
    repo_root = Path(repo_root)
    cat_dir = _review_root(repo_root) / category
    if not cat_dir.exists():
        return []
    existing = None if skip_dedup else _existing_solution_hashes(repo_root)
    results: list[PromotionResult] = []
    for src in sorted(cat_dir.glob("*.yaml")):
        results.append(promote_file(src, repo_root, force=force,
                                    skip_dedup=skip_dedup,
                                    _existing_hashes=existing))
    return results


def promote_all(repo_root: Path | str, force: bool = False,
                skip_dedup: bool = False) -> list[PromotionResult]:
    """Promote every YAML across every category."""
    repo_root = Path(repo_root)
    existing = None if skip_dedup else _existing_solution_hashes(repo_root)
    results: list[PromotionResult] = []
    for cat in list_review_queue(repo_root):
        cat_dir = _review_root(repo_root) / cat
        for src in sorted(cat_dir.glob("*.yaml")):
            results.append(promote_file(src, repo_root, force=force,
                                        skip_dedup=skip_dedup,
                                        _existing_hashes=existing))
    return results


def summarize(results: Iterable[PromotionResult]) -> dict:
    """Counts by outcome."""
    out = {"promoted": 0, "skipped": 0, "rejected": 0}
    for r in results:
        if r.promoted:
            out["promoted"] += 1
        elif "validation failed" in r.reason:
            out["rejected"] += 1
        else:
            out["skipped"] += 1
    return out


# ── Smoke test ──────────────────────────────────────────────────────


if __name__ == "__main__":
    import tempfile, yaml
    tmp = Path(tempfile.mkdtemp())
    (tmp / "data" / REVIEW_DIR_NAME / "json_quote_leak").mkdir(parents=True)
    src = tmp / "data" / REVIEW_DIR_NAME / "json_quote_leak" / "test.yaml"
    src.write_text(yaml.safe_dump({
        "domain": "agentic",
        "category": "json_array_form_for_multiline_writes",
        "examples": [{
            "query": "Write a multi-line file with embedded quotes",
            "solution": "Use the array form for content...",
            "tags": ["json", "write_file"],
        }],
    }))
    results = promote_all(tmp)
    print(f"results: {summarize(results)}")
    for r in results:
        print(f"  {'OK' if r.promoted else 'SKIP'} {r.source.name} -> {r.target} ({r.reason})")
