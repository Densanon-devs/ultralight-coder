#!/usr/bin/env python3
"""Re-run only the V3/V4 failures from the 400q runs against the hardened checks.

After the Phase 13 400q verification (coder-14b 385/400, qwen-14b 381/400), I
hardened ~20 V3/V4 queries in benchmark_realworld_v3.py and benchmark_realworld_v4.py
to accept valid alternative patterns (Counter one-liners, @total_ordering, pickle
directly, ChainMap, ThreadPoolExecutor, etc.) and loosened some over-strict
min_lines caps where the model is legitimately terse.

This script replays ONLY the V3/V4 failures from each model through the model
(large-mode gated, max_tokens=1024) and reports how many recover under the new
checks. Much cheaper than re-running the full 400q. V1+V2 results from the same
400q runs are unaffected and will be merged into the final summary separately.
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
        "prior_result": ROOT / "phase13_qwen2.5-coder-14b-instruct-q4_k_m_aug_t1024_all.json",
    },
    "qwen-14b": {
        "path": ROOT / "models" / "qwen2.5-14b-instruct-q4_k_m-00001-of-00003.gguf",
        "prior_result": ROOT / "phase13_qwen2.5-14b-instruct-q4_k_m-00001-of-00003_aug_t1024_all.json",
    },
}

MAX_TOKENS = 1024
OUT = ROOT / "phase13_v3v4_hardened_rerun.json"


def main() -> int:
    from benchmark_realworld import extract_code, check_query
    from benchmark_realworld_v3 import build_edge_case_queries
    from benchmark_realworld_v4 import build_deep_gap_queries
    from benchmark_exec import detect_chat_format, wrap_chat
    from llama_cpp import Llama
    from engine.augmentors import AugmentorRouter
    from engine.embedder import get_embedder

    v3_queries = build_edge_case_queries()
    v4_queries = build_deep_gap_queries()
    by_text = {q.query: q for q in (v3_queries + v4_queries)}
    v3_texts = {q.query for q in v3_queries}
    v4_texts = {q.query for q in v4_queries}

    report = {"max_tokens": MAX_TOKENS, "targets": {}}

    for name, info in TARGETS.items():
        if not info["path"].exists():
            print(f"SKIP {name}: missing {info['path']}")
            continue
        prior = json.load(open(info["prior_result"]))["configs"]["baseline"]
        # Only V3/V4 failures
        v3v4_fails = [r for r in prior["results"]
                      if not r["passed"] and (r["query"] in v3_texts or r["query"] in v4_texts)]
        # Count the V1+V2 passes/fails at old checks so we can compute total at new checks
        v3v4_passes = sum(1 for r in prior["results"]
                          if r["passed"] and (r["query"] in v3_texts or r["query"] in v4_texts))
        v1v2_passes = sum(1 for r in prior["results"]
                          if r["passed"] and r["query"] not in v3_texts and r["query"] not in v4_texts)
        v1v2_fails = sum(1 for r in prior["results"]
                         if not r["passed"] and r["query"] not in v3_texts and r["query"] not in v4_texts)

        print(f"\n=== {name}: {len(v3v4_fails)} V3/V4 failures to re-check ===")
        print(f"  (V1+V2 from prior: {v1v2_passes} pass, {v1v2_fails} fail — unchanged by V3/V4 hardening)")
        print(f"  (V3+V4 already passing: {v3v4_passes})")

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
        router.use_auto_augmentors(model_size_mb)  # activates _large_mode

        raw_system = (
            "You are a Python coding assistant. Write clean, correct, "
            "complete Python code in ```python blocks."
        )
        stop = ["</s>", "<|im_end|>", "<|end|>", "<|eot_id|>"]

        rerun_results = []
        for i, f in enumerate(v3v4_fails):
            q = by_text.get(f["query"])
            if q is None:
                print(f"  [{i+1}/{len(v3v4_fails)}] SKIP - not found")
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
            print(f"  [{i+1}/{len(v3v4_fails)}] {status} ({mode}) tok={tok} {elapsed:.1f}s  {q.query[:55]}")
            rerun_results.append({
                "query": q.query,
                "domain": q.domain,
                "set": "V3" if q.query in v3_texts else "V4",
                "passed": bool(checks["passed"]),
                "mode": mode,
                "completion_tokens": tok,
                "missing": [str(m) for m in checks["must_contain_fail"]],
                "lines": checks["line_count"],
                "generated_code": code if not checks["passed"] else None,
            })

        recovered = sum(1 for r in rerun_results if r["passed"])
        new_total_pass = v1v2_passes + v3v4_passes + recovered

        print(f"\n  {name}: {recovered}/{len(rerun_results)} V3/V4 recovered")
        print(f"  Projected total: {new_total_pass}/400  (was {v1v2_passes + v3v4_passes}/400)")

        report["targets"][name] = {
            "v1v2_passed_prior": v1v2_passes,
            "v3v4_passed_prior": v3v4_passes,
            "v3v4_rerun_count": len(rerun_results),
            "v3v4_recovered": recovered,
            "projected_total": new_total_pass,
            "rerun_results": rerun_results,
        }
        with open(OUT, "w", encoding="utf-8") as f:
            json.dump(report, f, indent=2)

        del model
        gc.collect()

    print("\n" + "=" * 60)
    print("  V3/V4 HARDENED CHECK RE-RUN SUMMARY")
    print("=" * 60)
    for name, r in report["targets"].items():
        prior = r["v1v2_passed_prior"] + r["v3v4_passed_prior"]
        print(f"  {name:10s}  {prior}/400 -> {r['projected_total']}/400  (+{r['v3v4_recovered']} from V3/V4 rerun)")

    print(f"\n  Results: {OUT}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
