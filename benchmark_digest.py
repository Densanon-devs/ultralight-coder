#!/usr/bin/env python3
"""
Digest Curation Benchmark — Score on-device curation against Claude ground truth.

Tests the 3-pass curation pipeline against existing digest_content.json files
(curated by Claude) to measure:
  - Selection overlap (Jaccard similarity)
  - Takeaway quality (length, structure, no meta-language)
  - Highlights quality (length, concreteness)
  - Verifier pass rates

Usage:
    python benchmark_digest.py                   # All topics
    python benchmark_digest.py --topic ai        # Single topic
    python benchmark_digest.py --verify-only     # Just check verifiers on ground truth
"""

import argparse
import json
import sys
from pathlib import Path

from engine.digest_augmentors import (
    verify_selection, verify_takeaways, verify_highlights,
)


def load_ground_truth(topics_dir: str, topic_id: str) -> tuple[list, dict]:
    """Load pending articles and Claude-curated digest for a topic."""
    base = Path(topics_dir) / topic_id

    pending_path = base / "pending_digest.json"
    content_path = base / "digest_content.json"

    if not pending_path.exists() or not content_path.exists():
        return [], {}

    with open(pending_path, encoding="utf-8") as f:
        pending = json.load(f)
    with open(content_path, encoding="utf-8") as f:
        content = json.load(f)

    return pending.get("articles", []), content


def extract_ground_truth_titles(content: dict) -> set[str]:
    """Extract all article titles from a digest_content.json."""
    titles = set()
    for section in content.get("sections", []):
        for article in section.get("articles", []):
            titles.add(article.get("title", "").lower().strip())
    return titles


def score_selection(selected_titles: set[str], gt_titles: set[str]) -> dict:
    """Score selection overlap using Jaccard similarity."""
    if not gt_titles:
        return {"jaccard": 0, "precision": 0, "recall": 0}

    intersection = selected_titles & gt_titles
    union = selected_titles | gt_titles

    jaccard = len(intersection) / len(union) if union else 0
    precision = len(intersection) / len(selected_titles) if selected_titles else 0
    recall = len(intersection) / len(gt_titles) if gt_titles else 0

    return {
        "jaccard": round(jaccard, 3),
        "precision": round(precision, 3),
        "recall": round(recall, 3),
        "matched": len(intersection),
        "selected": len(selected_titles),
        "ground_truth": len(gt_titles),
    }


def score_takeaways(content: dict) -> dict:
    """Score takeaway quality from ground truth (verifier checks)."""
    total = 0
    passed = 0
    lengths = []
    errors = []

    for section in content.get("sections", []):
        for article in section.get("articles", []):
            takeaways = article.get("takeaways", [])
            if not takeaways:
                continue

            total += 1
            response = json.dumps({"takeaways": takeaways})
            ok, err = verify_takeaways(response, "")
            if ok:
                passed += 1
            else:
                errors.append(f"  {article.get('title', '?')[:50]}: {err}")

            for t in takeaways:
                lengths.append(len(t))

    return {
        "total_articles": total,
        "verifier_passed": passed,
        "pass_rate": round(passed / total, 3) if total else 0,
        "avg_takeaway_length": round(sum(lengths) / len(lengths)) if lengths else 0,
        "min_length": min(lengths) if lengths else 0,
        "max_length": max(lengths) if lengths else 0,
        "errors": errors,
    }


def score_highlights(content: dict) -> dict:
    """Score highlights intro quality."""
    intro = content.get("highlights_intro", "")
    if not intro:
        return {"length": 0, "verifier_passed": False, "error": "No highlights_intro"}

    response = json.dumps({"highlights_intro": intro})
    ok, err = verify_highlights(response, "")

    return {
        "length": len(intro),
        "verifier_passed": ok,
        "error": err if not ok else "",
    }


def run_verify_only(topics_dir: str, topics: list[str]):
    """Run verifiers on existing ground truth to validate verifier quality."""
    print("\n" + "=" * 60)
    print("VERIFIER VALIDATION (ground truth data)")
    print("=" * 60)

    for topic_id in topics:
        articles, content = load_ground_truth(topics_dir, topic_id)
        if not content:
            print(f"\n{topic_id}: No ground truth data")
            continue

        print(f"\n--- {topic_id.upper()} ---")

        # Takeaways
        tk = score_takeaways(content)
        print(f"  Takeaways: {tk['verifier_passed']}/{tk['total_articles']} passed "
              f"({tk['pass_rate']:.0%})")
        if tk['errors']:
            for e in tk['errors']:
                print(f"    {e}")

        # Highlights
        hl = score_highlights(content)
        status = "PASS" if hl['verifier_passed'] else f"FAIL: {hl['error']}"
        print(f"  Highlights: {status} ({hl['length']} chars)")


def main():
    parser = argparse.ArgumentParser(description="Benchmark digest curation")
    parser.add_argument("--topic", help="Single topic to benchmark")
    parser.add_argument("--topics-dir", default="digest/topics",
                        help="Topics directory")
    parser.add_argument("--verify-only", action="store_true",
                        help="Only validate verifiers on ground truth")
    args = parser.parse_args()

    topics_dir = args.topics_dir

    # Discover topics
    if args.topic:
        topics = [args.topic]
    else:
        topics = []
        base = Path(topics_dir)
        if base.exists():
            for d in sorted(base.iterdir()):
                if d.is_dir() and (d / "digest_content.json").exists():
                    topics.append(d.name)

    if not topics:
        print("No topics with ground truth found")
        sys.exit(1)

    print(f"Topics: {', '.join(topics)}")

    if args.verify_only:
        run_verify_only(topics_dir, topics)
        return

    # Full benchmark — score ground truth data
    print("\n" + "=" * 60)
    print("GROUND TRUTH ANALYSIS")
    print("=" * 60)

    for topic_id in topics:
        articles, content = load_ground_truth(topics_dir, topic_id)
        if not content:
            continue

        print(f"\n{'='*40}")
        print(f"  {topic_id.upper()}")
        print(f"{'='*40}")

        # Selection analysis
        gt_titles = extract_ground_truth_titles(content)
        print(f"\n  Selection: {len(gt_titles)} articles selected by Claude")
        pending_titles = {a.get("title", "").lower().strip() for a in articles}
        available = gt_titles & pending_titles
        print(f"  Of which {len(available)} are in current pending_digest.json")

        # Takeaway quality
        tk = score_takeaways(content)
        print(f"\n  Takeaway verifier: {tk['verifier_passed']}/{tk['total_articles']} "
              f"({tk['pass_rate']:.0%})")
        print(f"  Avg length: {tk['avg_takeaway_length']} chars "
              f"(range: {tk['min_length']}-{tk['max_length']})")
        if tk['errors']:
            for e in tk['errors'][:3]:
                print(f"    {e}")

        # Highlights quality
        hl = score_highlights(content)
        print(f"\n  Highlights: {'PASS' if hl['verifier_passed'] else 'FAIL'} "
              f"({hl['length']} chars)")

    print("\n" + "=" * 60)
    print("Run with --verify-only to see detailed verifier results")
    print("Run digest_main.py to generate and compare against ground truth")


if __name__ == "__main__":
    main()
