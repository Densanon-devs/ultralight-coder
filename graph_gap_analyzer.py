#!/usr/bin/env python3
"""Graph-mode failure gap analyzer.

Reads a benchmark result JSON produced with `--augmentor-mode graph` (or
adaptive/hybrid), rebuilds the router in the same mode, and for each failing
query re-runs ONLY the retrieval step — no model needed. Reports:

  - Query text, domain, failure reason (missing keywords / min_lines / etc.)
  - Which examples the graph walk retrieved
  - The source category of each retrieved example
  - Whether the retrieved category looks topically relevant to the query domain
    (fuzzy heuristic — human review still needed)

Output is a markdown report suitable for deciding whether to author new
augmentor examples, add graph edges, or refine existing examples. Doesn't
modify anything — pure read-only analysis.

Usage:
    python graph_gap_analyzer.py phase13_qwen2.5-coder-14b-instruct-q4_k_m_aug_graph_t1024.json
    python graph_gap_analyzer.py --mode adaptive path/to/result.json
    python graph_gap_analyzer.py --all-failures result.json        # include passing-at-checks-but-failing
"""
import argparse
import json
import sys
from collections import Counter, defaultdict
from pathlib import Path
from typing import Optional

ROOT = Path(__file__).parent
sys.path.insert(0, str(ROOT))


def load_failures(result_path: Path) -> tuple[list[dict], dict]:
    """Return (failing_results, metadata_about_run)."""
    data = json.load(open(result_path))
    configs = data.get("configs", {})
    baseline = configs.get("baseline")
    if baseline is None:
        raise ValueError(f"No 'baseline' config in {result_path}")
    fails = [r for r in baseline["results"] if not r["passed"]]
    meta = {
        "target": data.get("target", "?"),
        "augmentor_mode": data.get("augmentor_mode", "?"),
        "max_tokens": data.get("max_tokens", "?"),
        "query_set": data.get("query_set", "?"),
        "total": baseline["total"],
        "passed": baseline["passed"],
        "pass_rate": baseline["pass_rate"],
    }
    return fails, meta


def build_router(mode: str):
    """Build the augmentor router in the requested mode — no model load."""
    from engine.augmentors import AugmentorRouter
    from engine.embedder import get_embedder
    router = AugmentorRouter(yaml_dir="data/augmentor_examples")
    router.init_embeddings(get_embedder())
    # Skip use_auto_augmentors — we want to force the mode for analysis
    if mode == "graph":
        router.use_graph_augmentors()
    elif mode == "adaptive":
        router.use_adaptive_augmentors()
    elif mode == "hybrid":
        router.use_hybrid_augmentors()
    elif mode == "rerank":
        router.use_rerank_augmentors()
    elif mode == "rerank1":
        router.use_rerank1_augmentors()
    else:
        raise ValueError(f"Unknown analyzer mode: {mode}")
    return router


# Fuzzy topical match: each query domain maps to a set of keywords that would
# suggest topical relevance in the retrieved example's category field. Used
# purely as a diagnostic hint — not a hard judgment.
DOMAIN_HINTS: dict[str, set[str]] = {
    "algorithm": {"algorithm", "math", "sort", "search", "dp", "graph_alg", "number"},
    "async": {"async", "concurrency", "queue", "await"},
    "cli": {"cli", "argparse", "shell", "log", "config", "env"},
    "data": {"data", "csv", "pandas", "etl", "json", "parquet"},
    "database": {"database", "sql", "sqlite", "orm", "postgres", "query"},
    "general": {"common", "basic", "string", "file", "general", "util"},
    "pattern": {"pattern", "design", "class", "decorator", "factory", "builder"},
    "testing": {"testing", "pytest", "mock", "fixture", "test"},
    "web": {"web", "http", "fastapi", "flask", "rest", "api", "route"},
}


def topically_relevant(query_domain: str, example_category: str) -> Optional[bool]:
    """Return True/False/None for topical match. None = no opinion."""
    cat = (example_category or "").lower()
    if not cat:
        return None
    hints = DOMAIN_HINTS.get(query_domain, set())
    if not hints:
        return None
    return any(h in cat for h in hints)


def analyze_one(router, q_text: str) -> dict:
    """Retrieve for a single query and return diagnostic info."""
    aug = router.select_augmentor(q_text, "code_gen")
    info = {
        "augmentor_name": None,
        "retrieval_mode": None,
        "examples": [],
        "example_categories": [],
    }
    if aug is None:
        return info
    info["augmentor_name"] = aug.name
    info["retrieval_mode"] = getattr(aug, "_retrieval_mode", "flat")
    try:
        examples = aug._retrieve_for_mode(q_text)
    except Exception as e:
        info["retrieval_error"] = f"{type(e).__name__}: {e}"
        return info
    for ex in examples or []:
        info["examples"].append({
            "query": ex.query,
            "category": ex.category,
        })
        info["example_categories"].append(ex.category or "?")
    return info


def main() -> int:
    p = argparse.ArgumentParser(description="Graph-mode failure gap analyzer (read-only)")
    p.add_argument("result", help="Path to benchmark result JSON")
    p.add_argument("--mode", default="",
                   help="Override retrieval mode (default: read from result metadata)")
    p.add_argument("--out", default="",
                   help="Write markdown report to this file (default: stdout only)")
    p.add_argument("--top-n-gaps", type=int, default=10,
                   help="How many most-common gap categories to highlight")
    args = p.parse_args()

    result_path = Path(args.result)
    fails, meta = load_failures(result_path)
    mode = args.mode or meta["augmentor_mode"]
    if mode in ("auto", "none", "?"):
        # auto/none don't have a meaningful retrieval mode to analyze
        mode = "graph"
        print(f"[note] metadata mode={meta['augmentor_mode']!r}, forcing analyzer mode=graph")

    print(f"\n=== Graph Gap Analyzer ===")
    print(f"  target:       {meta['target']}")
    print(f"  run mode:     {meta['augmentor_mode']}   (analyzer mode: {mode})")
    print(f"  query_set:    {meta['query_set']}")
    print(f"  score:        {meta['passed']}/{meta['total']}  ({meta['pass_rate']*100:.1f}%)")
    print(f"  failures:     {len(fails)}")

    router = build_router(mode)

    per_failure: list[dict] = []
    category_counter: Counter = Counter()
    domain_counter: Counter = Counter()
    missed_by_domain: defaultdict[str, list[str]] = defaultdict(list)
    non_topical_fails: list[dict] = []

    for r in fails:
        q_text = r["query"]
        diag = analyze_one(router, q_text)
        entry = {
            "query": q_text,
            "domain": r["domain"],
            "missing": r.get("missing", []),
            "lines": r.get("lines", 0),
            "completion_tokens": r.get("completion_tokens", 0),
            **diag,
        }
        per_failure.append(entry)

        domain_counter[r["domain"]] += 1
        for c in diag["example_categories"]:
            category_counter[c] += 1
        missed_by_domain[r["domain"]].append(q_text)

        if diag["example_categories"]:
            top_cat = diag["example_categories"][0]
            rel = topically_relevant(r["domain"], top_cat)
            if rel is False:
                non_topical_fails.append({
                    "query": q_text,
                    "domain": r["domain"],
                    "top_cat": top_cat,
                })

    # ── Render report ──────────────────────────────────────────────
    lines: list[str] = []
    lines.append(f"# Graph Gap Report — {meta['target']}")
    lines.append("")
    lines.append(f"- Run mode: `{meta['augmentor_mode']}` (analyzed as `{mode}`)")
    lines.append(f"- Query set: `{meta['query_set']}`")
    lines.append(f"- Score: **{meta['passed']}/{meta['total']}** ({meta['pass_rate']*100:.1f}%)")
    lines.append(f"- Failures inspected: {len(fails)}")
    lines.append("")

    lines.append("## Failures by domain")
    lines.append("")
    for dom, count in domain_counter.most_common():
        lines.append(f"- **{dom}**: {count} failure{'s' if count != 1 else ''}")
    lines.append("")

    lines.append("## Most-retrieved categories on failing queries")
    lines.append("")
    lines.append("Categories the graph walk landed on when the query ultimately failed. High counts here suggest either (a) the retrieved examples aren't teaching the right pattern shape, or (b) this category is a fallback when no better match exists.")
    lines.append("")
    for cat, count in category_counter.most_common(args.top_n_gaps):
        lines.append(f"- `{cat or '(none)'}`: {count}")
    lines.append("")

    if non_topical_fails:
        lines.append("## Topic mismatch — possible graph edge gaps")
        lines.append("")
        lines.append("Query domain is X but the top retrieved example category is Y, with no obvious topical overlap. These are prime candidates for **adding edges** to `data/pattern_graph.yaml` or **authoring new category-specific examples**.")
        lines.append("")
        for nt in non_topical_fails:
            lines.append(f"- **{nt['domain']}** query landed on `{nt['top_cat']}`")
            lines.append(f"  > {nt['query']}")
        lines.append("")

    lines.append("## Per-failure details")
    lines.append("")
    for e in per_failure:
        lines.append(f"### {e['domain']} — {e['query'][:80]}")
        lines.append(f"- failure: missing=`{e['missing']}` lines={e['lines']} tokens={e['completion_tokens']}")
        if e["augmentor_name"]:
            lines.append(f"- augmentor: `{e['augmentor_name']}` (retrieval={e['retrieval_mode']})")
        else:
            lines.append(f"- augmentor: **none selected** (router returned None — potential gate or unknown intent)")
        if e.get("retrieval_error"):
            lines.append(f"- retrieval error: `{e['retrieval_error']}`")
        if e["examples"]:
            lines.append(f"- retrieved examples:")
            for ex in e["examples"]:
                lines.append(f"  - `{ex['category']}` — *{ex['query'][:70]}*")
        lines.append("")

    report = "\n".join(lines)
    print()
    print(report)

    if args.out:
        Path(args.out).write_text(report, encoding="utf-8")
        print(f"\n[wrote] {args.out}")

    return 0


if __name__ == "__main__":
    sys.exit(main())
