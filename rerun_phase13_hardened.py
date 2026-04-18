#!/usr/bin/env python3
"""Re-run the residual 14B failures against the hardened V1+V2 checks.

After Phase 13 validation, coder-14b large-mode had 4 residual failures and
qwen-14b had 5 — all benchmark check brittleness (literal keyword checks that
don't accept valid alternative patterns like @contextmanager, AsyncMock,
SQL DDL, mpmath, etc.). benchmark_realworld.py's check_query now supports
tuple OR-groups for must_contain. This script replays just the failing
queries through each model (large-mode gated, max_tokens=1024) and reports
recoveries against the new checks.
"""
import gc
import json
import sys
import time
from pathlib import Path

ROOT = Path(__file__).parent
sys.path.insert(0, str(ROOT))

TARGETS = {
    "coder-14b": {
        "path": ROOT / "models" / "qwen2.5-coder-14b-instruct-q4_k_m.gguf",
        "prior_result": ROOT / "phase13_qwen2.5-coder-14b-instruct-q4_k_m_aug_t1024.json",
        "baseline_passed": 196,
    },
    "qwen-14b": {
        "path": ROOT / "models" / "qwen2.5-14b-instruct-q4_k_m-00001-of-00003.gguf",
        "prior_result": ROOT / "phase13_qwen2.5-14b-instruct-q4_k_m-00001-of-00003_aug_t1024.json",
        "baseline_passed": 195,
    },
}

MAX_TOKENS = 1024
OUT = ROOT / "phase13_hardened_rerun.json"


def main() -> int:
    from benchmark_realworld import (
        build_realworld_queries,
        build_realworld_queries_v2,
        extract_code,
        check_query,
    )
    from benchmark_exec import detect_chat_format, wrap_chat
    from llama_cpp import Llama
    from engine.augmentors import AugmentorRouter
    from engine.embedder import get_embedder

    all_queries = build_realworld_queries() + build_realworld_queries_v2()
    query_by_text = {q.query: q for q in all_queries}

    report = {"max_tokens": MAX_TOKENS, "targets": {}}

    for name, info in TARGETS.items():
        if not info["path"].exists():
            print(f"SKIP {name}: missing {info['path']}")
            continue
        prior = json.load(open(info["prior_result"]))["configs"]["baseline"]
        fails = [r for r in prior["results"] if not r["passed"]]
        print(f"\n=== {name}: {len(fails)} residual failures to re-check ===")
        for f in fails:
            print(f"  [{f['domain']:10s}] {f['query'][:60]}")

        print(f"\nLoading {info['path'].name}...")
        t0 = time.monotonic()
        model = Llama(
            model_path=str(info["path"]),
            n_ctx=4096,
            n_gpu_layers=99,
            n_batch=512,
            verbose=False,
        )
        print(f"  loaded in {time.monotonic() - t0:.1f}s")

        chat_format = detect_chat_format(str(info["path"]))
        model_size_mb = info["path"].stat().st_size / (1024 * 1024)

        router = AugmentorRouter(yaml_dir="data/augmentor_examples")
        router.init_embeddings(get_embedder())
        router.use_auto_augmentors(model_size_mb)  # activates _large_mode for 9 GB model

        raw_system = (
            "You are a Python coding assistant. Write clean, correct, "
            "complete Python code in ```python blocks."
        )
        stop = ["</s>", "<|im_end|>", "<|end|>", "<|eot_id|>"]

        rerun_results = []
        for i, f in enumerate(fails):
            q = query_by_text.get(f["query"])
            if q is None:
                print(f"  [{i+1}/{len(fails)}] SKIP — query not found in set")
                continue
            aug = router.select_augmentor(q.query, "code_gen")
            if aug is not None:
                prompt = aug.build_prompt(q.query, chat_format)
                mode = "aug"
            else:
                prompt = wrap_chat(raw_system, q.query, chat_format)
                mode = "raw"

            start = time.monotonic()
            out = model(prompt, max_tokens=MAX_TOKENS, temperature=0.2, stop=stop, echo=False)
            elapsed = time.monotonic() - start

            raw_text = out["choices"][0]["text"].strip()
            code = extract_code(raw_text)
            checks = check_query(code, q)
            status = "PASS" if checks["passed"] else "FAIL"
            tok = out.get("usage", {}).get("completion_tokens", 0) or 0
            print(f"  [{i+1}/{len(fails)}] {status} ({mode}) tok={tok} {elapsed:.1f}s  {q.query[:55]}")
            rerun_results.append({
                "query": q.query,
                "domain": q.domain,
                "passed": bool(checks["passed"]),
                "mode": mode,
                "completion_tokens": tok,
                "missing": [str(m) for m in checks["must_contain_fail"]],
                "lines": checks["line_count"],
                "generated_code": code if not checks["passed"] else None,
            })

        recovered = sum(1 for r in rerun_results if r["passed"])
        projected = info["baseline_passed"] + recovered
        print(f"\n  {name}: {recovered}/{len(rerun_results)} recovered  ->  projected {projected}/200")

        report["targets"][name] = {
            "prior_passed": info["baseline_passed"],
            "recovered": recovered,
            "projected_passed": projected,
            "rerun_results": rerun_results,
        }
        # Incremental write so a mid-run crash preserves completed targets
        with open(OUT, "w", encoding="utf-8") as f:
            json.dump(report, f, indent=2)

        del model
        gc.collect()

    print("\n" + "=" * 60)
    print("  HARDENED CHECK RE-RUN SUMMARY")
    print("=" * 60)
    for name, r in report["targets"].items():
        print(f"  {name:10s}  {r['prior_passed']} -> {r['projected_passed']}/200  (+{r['recovered']})")

    with open(OUT, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2)
    print(f"\n  Results: {OUT}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
