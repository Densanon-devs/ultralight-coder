"""
Library status — single-pane summary of every learning artifact.

The self-improvement pipeline now has FOUR data buckets:
  1. data/trajectory_examples/        — successful trajectories (B4)
  2. data/trajectory_review/          — failed trajectories
  3. data/auto_generated_review/      — flagged failures + recoveries (today)
  4. data/augmentor_examples/<dom>/   — promoted, retrievable

This module produces a flat status snapshot across all four. Used by the
ulcagent /library command and also written to a JSON file for CI checks.
"""
from __future__ import annotations

import time
from dataclasses import dataclass, field
from pathlib import Path


@dataclass
class LibraryStatus:
    trajectories_passed: int = 0
    trajectories_failed: int = 0
    review_queue_total: int = 0
    review_queue_by_category: dict[str, int] = field(default_factory=dict)
    promoted_total: int = 0
    promoted_by_category: dict[str, int] = field(default_factory=dict)
    augmentor_total: int = 0
    augmentor_by_domain: dict[str, int] = field(default_factory=dict)
    last_review_mtime: float | None = None
    last_promoted_mtime: float | None = None

    def to_dict(self) -> dict:
        return {
            "trajectories": {
                "passed": self.trajectories_passed,
                "failed": self.trajectories_failed,
            },
            "review_queue": {
                "total": self.review_queue_total,
                "by_category": dict(self.review_queue_by_category),
                "last_mtime": self.last_review_mtime,
            },
            "promoted": {
                "total": self.promoted_total,
                "by_category": dict(self.promoted_by_category),
                "last_mtime": self.last_promoted_mtime,
            },
            "augmentor_library": {
                "total": self.augmentor_total,
                "by_domain": dict(self.augmentor_by_domain),
            },
        }


def _count_yamls(d: Path) -> tuple[int, dict[str, int], float | None]:
    """(total_count, count_per_top_subdir, latest_mtime). Returns (0, {}, None)
    if the dir doesn't exist."""
    if not d.exists():
        return 0, {}, None
    by_sub: dict[str, int] = {}
    total = 0
    latest: float | None = None
    for sub in sorted(p for p in d.iterdir() if p.is_dir()):
        n = 0
        for f in sub.glob("*.yaml"):
            n += 1
            mt = f.stat().st_mtime
            if latest is None or mt > latest:
                latest = mt
        if n:
            by_sub[sub.name] = n
            total += n
    return total, by_sub, latest


def _count_trajectories(repo_root: Path) -> tuple[int, int]:
    """Successful + failed trajectory counts from B4."""
    success = list((repo_root / "data" / "trajectory_examples").glob("*.yaml")) \
        if (repo_root / "data" / "trajectory_examples").exists() else []
    fail = list((repo_root / "data" / "trajectory_review").glob("*.yaml")) \
        if (repo_root / "data" / "trajectory_review").exists() else []
    return len(success), len(fail)


def _count_augmentors(repo_root: Path) -> tuple[int, dict[str, int]]:
    """Total YAMLs in data/augmentor_examples/, grouped by top-level subdir."""
    base = repo_root / "data" / "augmentor_examples"
    if not base.exists():
        return 0, {}
    by_dom: dict[str, int] = {}
    total = 0
    for sub in sorted(p for p in base.iterdir() if p.is_dir()):
        n = sum(1 for _ in sub.rglob("*.yaml"))
        if n:
            by_dom[sub.name] = n
            total += n
    return total, by_dom


def collect(repo_root: Path | str) -> LibraryStatus:
    """Walk all four buckets and return a populated LibraryStatus."""
    repo_root = Path(repo_root)
    s = LibraryStatus()

    # Trajectories
    s.trajectories_passed, s.trajectories_failed = _count_trajectories(repo_root)

    # Review queue
    rt, rcat, rmt = _count_yamls(repo_root / "data" / "auto_generated_review")
    s.review_queue_total = rt
    s.review_queue_by_category = rcat
    s.last_review_mtime = rmt

    # Promoted
    pt, pcat, pmt = _count_yamls(repo_root / "data" / "augmentor_examples" / "agentic")
    s.promoted_total = pt
    s.promoted_by_category = pcat
    s.last_promoted_mtime = pmt

    # Total augmentor library
    s.augmentor_total, s.augmentor_by_domain = _count_augmentors(repo_root)

    return s


def render_text(status: LibraryStatus) -> str:
    """Human-readable multi-line summary suitable for the REPL."""
    lines: list[str] = []
    lines.append(f"Trajectories: {status.trajectories_passed} passed, {status.trajectories_failed} failed")
    lines.append(f"Review queue: {status.review_queue_total} pending")
    if status.review_queue_by_category:
        for cat, n in sorted(status.review_queue_by_category.items(), key=lambda kv: -kv[1]):
            lines.append(f"  {cat}: {n}")
    lines.append(f"Promoted (agentic): {status.promoted_total}")
    if status.promoted_by_category:
        for cat, n in sorted(status.promoted_by_category.items(), key=lambda kv: -kv[1]):
            lines.append(f"  {cat}: {n}")
    lines.append(f"Total augmentor library: {status.augmentor_total} YAMLs across {len(status.augmentor_by_domain)} domains")
    if status.last_review_mtime:
        age = time.time() - status.last_review_mtime
        lines.append(f"Last review entry: {_fmt_age(age)}")
    if status.last_promoted_mtime:
        age = time.time() - status.last_promoted_mtime
        lines.append(f"Last promotion:    {_fmt_age(age)}")
    return "\n".join(lines)


def _fmt_age(seconds: float) -> str:
    if seconds < 60:
        return f"{seconds:.0f}s ago"
    if seconds < 3600:
        return f"{seconds / 60:.0f}m ago"
    if seconds < 86400:
        return f"{seconds / 3600:.1f}h ago"
    return f"{seconds / 86400:.1f}d ago"


# ── Smoke ───────────────────────────────────────────────────────────


if __name__ == "__main__":
    import sys
    root = Path(sys.argv[1]) if len(sys.argv) > 1 else Path(__file__).resolve().parent.parent
    s = collect(root)
    print(render_text(s))
