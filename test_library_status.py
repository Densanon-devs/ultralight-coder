"""
Tests for engine/library_status.py — single-pane summary of all
learning artifacts.
"""
from __future__ import annotations
import sys
import tempfile
import time
from pathlib import Path

ROOT = Path(__file__).parent
sys.path.insert(0, str(ROOT))

_CORE_ROOT = ROOT.parent / "densanon-core"
if _CORE_ROOT.exists() and str(_CORE_ROOT) not in sys.path:
    sys.path.insert(0, str(_CORE_ROOT))

from engine.library_status import collect, render_text, _fmt_age


def _make_workspace():
    return Path(tempfile.mkdtemp())


def _touch(p: Path, text: str = "x"):
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(text, encoding="utf-8")


def test_empty_workspace():
    s = collect(_make_workspace())
    assert s.trajectories_passed == 0
    assert s.trajectories_failed == 0
    assert s.review_queue_total == 0
    assert s.promoted_total == 0
    assert s.augmentor_total == 0


def test_trajectories_counted():
    tmp = _make_workspace()
    _touch(tmp / "data" / "trajectory_examples" / "a.yaml")
    _touch(tmp / "data" / "trajectory_examples" / "b.yaml")
    _touch(tmp / "data" / "trajectory_review" / "c.yaml")
    s = collect(tmp)
    assert s.trajectories_passed == 2
    assert s.trajectories_failed == 1


def test_review_queue_grouped_by_category():
    tmp = _make_workspace()
    _touch(tmp / "data" / "auto_generated_review" / "json_quote_leak" / "a.yaml")
    _touch(tmp / "data" / "auto_generated_review" / "json_quote_leak" / "b.yaml")
    _touch(tmp / "data" / "auto_generated_review" / "missing_import" / "c.yaml")
    s = collect(tmp)
    assert s.review_queue_total == 3
    assert s.review_queue_by_category == {"json_quote_leak": 2, "missing_import": 1}


def test_promoted_counted_separately():
    tmp = _make_workspace()
    _touch(tmp / "data" / "augmentor_examples" / "agentic" / "json_quote_leak" / "a.yaml")
    _touch(tmp / "data" / "augmentor_examples" / "agentic" / "missing_import" / "b.yaml")
    s = collect(tmp)
    assert s.promoted_total == 2


def test_augmentor_total_walks_all_domains():
    tmp = _make_workspace()
    _touch(tmp / "data" / "augmentor_examples" / "pattern" / "decorator.yaml")
    _touch(tmp / "data" / "augmentor_examples" / "pattern" / "router.yaml")
    _touch(tmp / "data" / "augmentor_examples" / "python" / "basics.yaml")
    s = collect(tmp)
    assert s.augmentor_total == 3
    assert s.augmentor_by_domain == {"pattern": 2, "python": 1}


def test_last_mtime_populated_on_review_dir():
    tmp = _make_workspace()
    _touch(tmp / "data" / "auto_generated_review" / "x" / "a.yaml")
    s = collect(tmp)
    assert s.last_review_mtime is not None
    assert s.last_review_mtime <= time.time()


def test_render_text_includes_all_sections():
    tmp = _make_workspace()
    _touch(tmp / "data" / "trajectory_examples" / "t.yaml")
    _touch(tmp / "data" / "auto_generated_review" / "x" / "a.yaml")
    _touch(tmp / "data" / "augmentor_examples" / "agentic" / "x" / "b.yaml")
    rendered = render_text(collect(tmp))
    assert "Trajectories:" in rendered
    assert "Review queue:" in rendered
    assert "Promoted" in rendered
    assert "Total augmentor library:" in rendered


def test_to_dict_serializes_cleanly():
    import json
    tmp = _make_workspace()
    _touch(tmp / "data" / "auto_generated_review" / "x" / "a.yaml")
    d = collect(tmp).to_dict()
    s = json.dumps(d)  # must be JSON-serializable
    assert "review_queue" in s
    assert "augmentor_library" in s


def test_fmt_age_buckets():
    assert "s ago" in _fmt_age(30)
    assert "m ago" in _fmt_age(120)
    assert "h ago" in _fmt_age(3600 * 5)
    assert "d ago" in _fmt_age(86400 * 3)


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
