#!/usr/bin/env python3
"""
Phase 13 Benchmark — 14B target models, baseline + experimental speculative.

DEFAULT: runs only the baseline config (raw 14B, no speculative decoding).
This is the mode used for Phase 13's primary hypothesis ("does the augmentor
system help or hurt 14B models?"), which does NOT require speculative
decoding to answer.

Available configs:

    1. baseline             — raw 14B, no speculative decoding (DEFAULT)
    2. prompt_lookup_n10    — EXPERIMENTAL, broken on llama-cpp-python 0.3.9
    3. prompt_lookup_n15    — EXPERIMENTAL, broken on llama-cpp-python 0.3.9

Prompt lookup status (smoke-tested 2026-04-12 on Qwen Coder 3B, 18 queries):
    - baseline:          18/18 pass, 15.8 tok/s
    - prompt_lookup_n10:   4/18 pass, 11.3 tok/s  (broken + slower)
    - prompt_lookup_n15:   3/18 pass, 10.7 tok/s  (broken + slower)

    prompt_lookup paths crash with numpy broadcast errors:
      "could not broadcast input array from shape (X,) into shape (Y,)"
    where X/Y are vocab_size multiples. Likely fixed by upgrading
    llama-cpp-python from 0.3.9 to 0.3.20+, but the upgrade carries risk
    on Windows builds and issue abetlen/llama-cpp-python#2110 reports that
    LlamaPromptLookupDecoding "gives no performance improvement" even when
    it works correctly.

    Opt in with --include-experimental only after upgrading llama-cpp-python
    OR if you want to empirically confirm the bug on your version.

Why no second-Llama draft_model path (Qwen 0.5B -> 14B):
    llama-cpp-python 0.3.9's `draft_model` kwarg only accepts LlamaDraftModel
    subclasses, and the library ships exactly one: LlamaPromptLookupDecoding.
    The 2.5x community reports on llama.cpp #10466 come from the C++ CLI tool
    `llama-speculative-simple`, not from the Python API. A subprocess-based
    benchmark would be a separate effort. See experiment_backlog.md #2.

Per config we measure:
    - total wall time
    - avg tokens/sec (completion tokens divided by generation time)
    - pass rate on the must_contain / must_not_contain / min_lines checks
    - per-domain breakdown

Usage:
    # Full matrix on Qwen 2.5 Coder 14B:
    python benchmark_phase13.py --target coder-14b

    # Full matrix on Qwen 2.5 14B (non-coder):
    python benchmark_phase13.py --target qwen-14b

    # Quick smoke test (2 queries per domain):
    python benchmark_phase13.py --target coder-14b --quick

    # Only run specific configs:
    python benchmark_phase13.py --target coder-14b --only baseline prompt_lookup_n10

    # Skip a config that's OOMing:
    python benchmark_phase13.py --target coder-14b --skip prompt_lookup_n15

OOM handling:
    Each config attempts n_ctx=4096 first, falls back to n_ctx=2048 on OOM,
    and skips the config entirely if still failing. Results are written
    incrementally after each config so a mid-run interrupt preserves work.
"""

import argparse
import gc
import json
import logging
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

PROJECT_ROOT = Path(__file__).parent
sys.path.insert(0, str(PROJECT_ROOT))

logger = logging.getLogger(__name__)


# ── Matrix definition ──────────────────────────────────────────────────────


@dataclass
class SpecConfig:
    name: str
    description: str
    mode: str  # "none" | "prompt_lookup"
    num_pred_tokens: int = 0  # only meaningful when mode == "prompt_lookup"


TARGETS = {
    "coder-14b": "models/qwen2.5-coder-14b-instruct-q4_k_m.gguf",
    # qwen-14b is a split GGUF (3 parts). llama.cpp auto-loads siblings
    # when pointed at the first shard.
    "qwen-14b": "models/qwen2.5-14b-instruct-q4_k_m-00001-of-00003.gguf",
}


def build_matrix(include_experimental: bool = False) -> list[SpecConfig]:
    matrix = [
        SpecConfig(
            name="baseline",
            description="Raw 14B, no speculative decoding",
            mode="none",
        ),
    ]
    if include_experimental:
        matrix.extend([
            SpecConfig(
                name="prompt_lookup_n10",
                description="EXPERIMENTAL: 14B + LlamaPromptLookupDecoding n=10 (broken in llama-cpp-python 0.3.9)",
                mode="prompt_lookup",
                num_pred_tokens=10,
            ),
            SpecConfig(
                name="prompt_lookup_n15",
                description="EXPERIMENTAL: 14B + LlamaPromptLookupDecoding n=15 (broken in llama-cpp-python 0.3.9)",
                mode="prompt_lookup",
                num_pred_tokens=15,
            ),
        ])
    return matrix


# ── Model loading with OOM fallback ────────────────────────────────────────


def _build_llama(target_path: Path, spec: SpecConfig, n_ctx: int, gpu_layers: int):
    from llama_cpp import Llama

    kwargs = dict(
        model_path=str(target_path),
        n_ctx=n_ctx,
        n_gpu_layers=gpu_layers,
        n_batch=512,
        verbose=False,
    )

    if spec.mode == "prompt_lookup":
        from engine.native_speculative import NativeSpeculativeConfig, build_draft_model
        cfg = NativeSpeculativeConfig(
            enabled=True,
            mode="prompt_lookup",
            num_pred_tokens=spec.num_pred_tokens,
            max_ngram_size=2,
        )
        draft = build_draft_model(cfg)
        if draft is None:
            raise RuntimeError(f"Failed to build prompt_lookup draft for {spec.name}")
        kwargs["draft_model"] = draft

    return Llama(**kwargs)


def load_with_oom_fallback(target_path: Path, spec: SpecConfig, gpu_layers: int):
    """Try n_ctx=4096, fall back to 2048, give up on third failure."""
    for attempt_ctx in (4096, 2048):
        try:
            print(f"  loading at n_ctx={attempt_ctx}...", end=" ", flush=True)
            start = time.monotonic()
            model = _build_llama(target_path, spec, attempt_ctx, gpu_layers)
            print(f"OK in {time.monotonic() - start:.1f}s")
            return model
        except Exception as e:
            err = str(e).lower()
            is_oom = any(tok in err for tok in ("out of memory", "oom", "cuda", "cublas", "alloc"))
            if is_oom:
                print(f"OOM — retrying smaller" if attempt_ctx == 4096 else "OOM — giving up")
                gc.collect()
                continue
            print(f"FAIL: {e}")
            raise
    return None


# ── Per-config benchmark run ───────────────────────────────────────────────


AUGMENTOR_MODES = ("auto", "graph", "adaptive", "hybrid", "none")


def _build_router(mode: str, model_size_mb: float):
    """Build the augmentor router for a given mode, or return None for 'none'.

    Modes map onto AugmentorRouter dispatch methods:
      auto     -> use_auto_augmentors(model_size_mb)   — size-tiered default; 14B gets large-mode gating
      graph    -> use_graph_augmentors()                — pure dependency-walked retrieval (Phase 13 followup)
      adaptive -> use_adaptive_augmentors()             — per-query flat-vs-graph auto-switch via _is_composite_query
      hybrid   -> use_hybrid_augmentors()               — graph first, flat fallback on verification failure
      none     -> returns None                          — raw baseline, no injection
    """
    if mode == "none":
        return None

    from engine.augmentors import AugmentorRouter
    from engine.embedder import get_embedder
    router = AugmentorRouter(yaml_dir="data/augmentor_examples")
    router.init_embeddings(get_embedder())

    if mode == "auto":
        router.use_auto_augmentors(model_size_mb)
    elif mode == "graph":
        router.use_graph_augmentors()
    elif mode == "adaptive":
        router.use_adaptive_augmentors()
    elif mode == "hybrid":
        router.use_hybrid_augmentors()
    else:
        raise ValueError(f"Unknown augmentor mode: {mode}. Choose from {AUGMENTOR_MODES}")
    return router


def run_config(model, queries, chat_format, model_size_mb: float,
               augmentor_mode: str = "auto", max_tokens: int = 512) -> dict:
    """Run all queries through the loaded model and return aggregate stats.

    augmentor_mode selects the retrieval strategy — see _build_router for the full list.
    'auto' is the Phase 13 default (size-tiered + large-mode gating for 14B).
    'none' is the raw baseline (no augmentor injection at all).
    'graph' / 'adaptive' / 'hybrid' are Phase 13 followup experiments to test whether
    richer structured-context retrieval beats large-mode gating at 14B scale.
    """
    from benchmark_realworld import extract_code, check_query

    router = _build_router(augmentor_mode, model_size_mb)

    stop = ["</s>", "<|im_end|>", "<|end|>", "<|eot_id|>"]
    results = []
    total_gen_tokens = 0
    total_gen_time = 0.0

    from benchmark_exec import wrap_chat
    raw_system = (
        "You are a Python coding assistant. Write clean, correct, "
        "complete Python code in ```python blocks."
    )

    for i, q in enumerate(queries):
        aug = router.select_augmentor(q.query, "code_gen") if router is not None else None
        if aug is not None:
            prompt = aug.build_prompt(q.query, chat_format)
        else:
            prompt = wrap_chat(raw_system, q.query, chat_format)

        start = time.monotonic()
        try:
            output = model(prompt, max_tokens=max_tokens, temperature=0.2, stop=stop, echo=False)
        except Exception as e:
            logger.warning(f"generation failed on query {i}: {e}")
            results.append({
                "query": q.query,
                "domain": q.domain,
                "passed": False,
                "error": str(e),
                "score": 0.0,
                "time": 0.0,
                "completion_tokens": 0,
                "tokens_per_sec": 0.0,
            })
            continue
        elapsed = time.monotonic() - start

        response = output["choices"][0]["text"].strip()
        code = extract_code(response)
        checks = check_query(code, q)
        usage = output.get("usage", {})
        completion = usage.get("completion_tokens", 0) or 0
        total_gen_tokens += completion
        total_gen_time += elapsed

        tok_s = completion / elapsed if elapsed > 0 else 0.0
        status = "PASS" if checks["passed"] else "FAIL"
        print(f"  [{i + 1}/{len(queries)}] {status} {elapsed:.1f}s ({tok_s:.1f} t/s) — {q.query[:50]}")

        results.append({
            "query": q.query,
            "domain": q.domain,
            "passed": bool(checks["passed"]),
            "score": round(checks["score"], 3),
            "time": round(elapsed, 3),
            "completion_tokens": completion,
            "tokens_per_sec": round(tok_s, 2),
            "lines": checks["line_count"],
            "missing": checks["must_contain_fail"],
            "unwanted": checks["must_not_contain_fail"],
        })

    passed = sum(1 for r in results if r["passed"])
    avg_tok_s = total_gen_tokens / total_gen_time if total_gen_time > 0 else 0.0

    domains = {}
    for r in results:
        d = r["domain"]
        if d not in domains:
            domains[d] = {"passed": 0, "total": 0}
        domains[d]["total"] += 1
        if r["passed"]:
            domains[d]["passed"] += 1

    return {
        "total": len(results),
        "passed": passed,
        "pass_rate": round(passed / len(results), 4) if results else 0,
        "total_gen_tokens": total_gen_tokens,
        "total_gen_time_s": round(total_gen_time, 2),
        "avg_tokens_per_sec": round(avg_tok_s, 2),
        "per_domain": domains,
        "results": results,
    }


# ── Orchestrator ───────────────────────────────────────────────────────────


def resolve_target(target_arg: str) -> Path:
    if target_arg in TARGETS:
        return Path(TARGETS[target_arg])
    return Path(target_arg)


QUERY_SETS = ("v1v2", "v3", "v4", "all")


def build_queries(quick: bool, query_set: str = "v1v2"):
    from benchmark_realworld import build_realworld_queries, build_realworld_queries_v2

    if query_set == "v1v2":
        queries = build_realworld_queries() + build_realworld_queries_v2()
    elif query_set == "v3":
        from benchmark_realworld_v3 import build_edge_case_queries
        queries = build_edge_case_queries()
    elif query_set == "v4":
        from benchmark_realworld_v4 import build_deep_gap_queries
        queries = build_deep_gap_queries()
    elif query_set == "all":
        from benchmark_realworld_v3 import build_edge_case_queries
        from benchmark_realworld_v4 import build_deep_gap_queries
        queries = (build_realworld_queries() + build_realworld_queries_v2()
                   + build_edge_case_queries() + build_deep_gap_queries())
    else:
        raise ValueError(f"Unknown query_set: {query_set}. Choose from {QUERY_SETS}")

    if quick:
        by_domain = {}
        for q in queries:
            by_domain.setdefault(q.domain, []).append(q)
        queries = []
        for domain_queries in by_domain.values():
            queries.extend(domain_queries[:2])
    return queries


def run_matrix(args) -> int:
    target_path = resolve_target(args.target)
    if not target_path.exists():
        print(f"Target model not found: {target_path}", file=sys.stderr)
        print(f"Download with: python download_model.py --model {args.target}", file=sys.stderr)
        return 1

    size_gb = target_path.stat().st_size / 1e9
    model_size_mb = target_path.stat().st_size / (1024 * 1024)
    print(f"Target: {target_path.name} ({size_gb:.1f} GB)")

    queries = build_queries(args.quick, args.query_set)
    mode_label = "quick" if args.quick else "full"
    print(f"Queries per config: {len(queries)} ({mode_label} mode, set={args.query_set})")

    from benchmark_exec import detect_chat_format
    chat_format = detect_chat_format(str(target_path))
    print(f"Chat format: {chat_format}")

    matrix = build_matrix(include_experimental=args.include_experimental)
    if args.only:
        matrix = [m for m in matrix if m.name in set(args.only)]
        if not matrix:
            print(f"--only filter removed all configs. Valid names: {[m.name for m in build_matrix()]}")
            return 2
    if args.skip:
        matrix = [m for m in matrix if m.name not in set(args.skip)]

    # Resolve mode: --no-augmentors is a legacy alias for --augmentor-mode none
    mode = "none" if args.no_augmentors else args.augmentor_mode
    # Output filename components
    if mode == "none":
        aug_suffix = "_raw"
    elif mode == "auto":
        aug_suffix = "_aug"
    else:
        aug_suffix = f"_aug_{mode}"  # _aug_graph / _aug_adaptive / _aug_hybrid
    tok_suffix = "" if args.max_tokens == 512 else f"_t{args.max_tokens}"
    set_suffix = "" if args.query_set == "v1v2" else f"_{args.query_set}"
    output_path = Path(args.output) if args.output else Path(f"phase13_{target_path.stem}{aug_suffix}{tok_suffix}{set_suffix}.json")
    all_results = {
        "target": target_path.name,
        "target_path": str(target_path),
        "target_size_gb": round(size_gb, 2),
        "quick_mode": args.quick,
        "queries_per_config": len(queries),
        "gpu_layers": args.gpu_layers,
        "augmentor_mode": mode,
        "augmentors_enabled": mode != "none",
        "max_tokens": args.max_tokens,
        "query_set": args.query_set,
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S"),
        "configs": {},
    }

    for spec in matrix:
        print(f"\n{'=' * 70}")
        print(f"  CONFIG: {spec.name}")
        print(f"  {spec.description}")
        print(f"{'=' * 70}")

        model = load_with_oom_fallback(target_path, spec, gpu_layers=args.gpu_layers)
        if model is None:
            all_results["configs"][spec.name] = {
                "description": spec.description,
                "status": "load_failed",
            }
            with open(output_path, "w", encoding="utf-8") as f:
                json.dump(all_results, f, indent=2)
            continue

        config_start = time.monotonic()
        try:
            stats = run_config(model, queries, chat_format, model_size_mb,
                               augmentor_mode=mode,
                               max_tokens=args.max_tokens)
            stats["description"] = spec.description
            stats["wall_time_s"] = round(time.monotonic() - config_start, 2)
            stats["status"] = "ok"
            all_results["configs"][spec.name] = stats

            print(f"\n  {spec.name}: {stats['passed']}/{stats['total']} "
                  f"({stats['pass_rate']:.1%}), "
                  f"{stats['avg_tokens_per_sec']} tok/s, "
                  f"{stats['wall_time_s']:.0f}s wall")
        except Exception as e:
            logger.exception(f"Run failed for {spec.name}")
            all_results["configs"][spec.name] = {
                "description": spec.description,
                "status": "run_failed",
                "error": str(e),
            }
        finally:
            del model
            gc.collect()

        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(all_results, f, indent=2)

    # ── Summary ────────────────────────────────────────────────────────────
    print(f"\n{'=' * 70}")
    print(f"  SUMMARY — {target_path.name}")
    print(f"{'=' * 70}")
    header = f"  {'config':22s}  {'pass':12s}  {'tok/s':>8s}  {'wall':>8s}  {'speedup':>10s}"
    print(header)
    print(f"  {'-' * 22}  {'-' * 12}  {'-' * 8}  {'-' * 8}  {'-' * 10}")
    baseline_tok = all_results["configs"].get("baseline", {}).get("avg_tokens_per_sec", 0)
    for name, stats in all_results["configs"].items():
        status = stats.get("status", "?")
        if status != "ok":
            print(f"  {name:22s}  [{status}]")
            continue
        tok = stats["avg_tokens_per_sec"]
        speedup = f"{tok / baseline_tok:.2f}x" if baseline_tok else "n/a"
        pass_str = f"{stats['passed']}/{stats['total']}"
        wall = stats["wall_time_s"]
        print(f"  {name:22s}  {pass_str:12s}  {tok:>8.1f}  {wall:>7.0f}s  {speedup:>10s}")

    print(f"\n  Results: {output_path}")
    return 0


# ── CLI ────────────────────────────────────────────────────────────────────


def main():
    p = argparse.ArgumentParser(description="Phase 13: 14B + speculative decoding matrix")
    p.add_argument("--target", required=True,
                   help=f"Target model: one of {list(TARGETS.keys())} or an explicit models/*.gguf path")
    p.add_argument("--quick", action="store_true",
                   help="Run 2 queries per domain (smoke test, ~20 queries total)")
    p.add_argument("--gpu-layers", type=int, default=99)
    p.add_argument("--only", nargs="+", default=[],
                   help="Only run these config names (e.g., baseline prompt_lookup_n10)")
    p.add_argument("--skip", nargs="+", default=[],
                   help="Skip these config names")
    p.add_argument("--include-experimental", action="store_true",
                   help="Include prompt_lookup configs (broken on llama-cpp-python 0.3.9)")
    p.add_argument("--no-augmentors", action="store_true",
                   help="Legacy alias for --augmentor-mode none. Raw baseline, no YAML example injection.")
    p.add_argument("--augmentor-mode", choices=AUGMENTOR_MODES, default="auto",
                   help=("Augmentor retrieval mode. auto (default) uses the size-tiered router "
                         "with large-mode gating for 14B. graph uses pure dependency-walked retrieval. "
                         "adaptive picks flat-vs-graph per-query. hybrid is graph-first with flat fallback. "
                         "none is the raw baseline. Phase 13 followup: test whether graph beats large-mode at 14B."))
    p.add_argument("--max-tokens", type=int, default=1024,
                   help="Generation cap per query (default 1024 — Phase 13 rerun showed 512 masked ~2-3 queries per run)")
    p.add_argument("--query-set", choices=QUERY_SETS, default="v1v2",
                   help="Benchmark set: v1v2 (200q, default), v3 (100q edge cases), v4 (100q deep gaps), all (400q)")
    p.add_argument("--output", help="Output JSON path (default: phase13_<target-stem>.json)")
    p.add_argument("-v", "--verbose", action="store_true")
    args = p.parse_args()

    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(asctime)s [%(name)s] %(levelname)s: %(message)s",
        datefmt="%H:%M:%S",
    )

    return run_matrix(args)


if __name__ == "__main__":
    sys.exit(main())
