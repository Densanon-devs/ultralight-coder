#!/usr/bin/env python3
"""Re-run Phase 13 failing queries at max_tokens=1024 to isolate truncation-masked model capability.

Loads coder-14b once, replays the failing queries from both the augmented and raw runs
in the correct mode, and reports which queries now pass. Exists to answer one question:
were the 14B failures model defects or 512-token-cap truncation?

Scope: coder-14b only (best model from the 4-run matrix).
"""
import gc
import json
import sys
import time
from pathlib import Path

ROOT = Path(__file__).parent
sys.path.insert(0, str(ROOT))

TARGET = ROOT / "models" / "qwen2.5-coder-14b-instruct-q4_k_m.gguf"
AUG_RESULTS = ROOT / "phase13_qwen2.5-coder-14b-instruct-q4_k_m_aug.json"
RAW_RESULTS = ROOT / "phase13_qwen2.5-coder-14b-instruct-q4_k_m_raw.json"
OUT = ROOT / "phase13_coder14b_rerun_1024.json"

MAX_TOKENS = 1024
GPU_LAYERS = 99
N_CTX = 4096


def load_failing_queries(results_path: Path) -> list[dict]:
    data = json.load(open(results_path))
    cfg = data["configs"]["baseline"]
    return [r for r in cfg["results"] if not r["passed"]]


def main() -> int:
    if not TARGET.exists():
        print(f"Target model missing: {TARGET}")
        return 1
    if not AUG_RESULTS.exists() or not RAW_RESULTS.exists():
        print("Need both aug and raw results JSONs before rerun.")
        return 1

    aug_fails = load_failing_queries(AUG_RESULTS)
    raw_fails = load_failing_queries(RAW_RESULTS)
    print(f"aug failures to re-run: {len(aug_fails)}")
    print(f"raw failures to re-run: {len(raw_fails)}")

    from benchmark_realworld import (
        build_realworld_queries,
        build_realworld_queries_v2,
        extract_code,
        check_query,
    )
    from benchmark_exec import detect_chat_format, wrap_chat
    from llama_cpp import Llama

    all_queries = build_realworld_queries() + build_realworld_queries_v2()
    query_by_text = {q.query: q for q in all_queries}

    chat_format = detect_chat_format(str(TARGET))
    model_size_mb = TARGET.stat().st_size / (1024 * 1024)

    print(f"Loading {TARGET.name} on GPU (n_ctx={N_CTX}, max_tokens={MAX_TOKENS})...")
    t0 = time.monotonic()
    model = Llama(
        model_path=str(TARGET),
        n_ctx=N_CTX,
        n_gpu_layers=GPU_LAYERS,
        n_batch=512,
        verbose=False,
    )
    print(f"  loaded in {time.monotonic() - t0:.1f}s")

    from engine.augmentors import AugmentorRouter
    from engine.embedder import get_embedder
    router = AugmentorRouter(yaml_dir="data/augmentor_examples")
    router.init_embeddings(get_embedder())
    router.use_auto_augmentors(model_size_mb)

    raw_system = (
        "You are a Python coding assistant. Write clean, correct, "
        "complete Python code in ```python blocks."
    )
    stop = ["</s>", "<|im_end|>", "<|end|>", "<|eot_id|>"]

    def run_one(q_text: str, use_aug: bool) -> dict:
        q = query_by_text[q_text]
        if use_aug:
            aug = router.select_augmentor(q.query, "code_gen")
            prompt = aug.build_prompt(q.query, chat_format) if aug else wrap_chat(raw_system, q.query, chat_format)
        else:
            prompt = wrap_chat(raw_system, q.query, chat_format)
        start = time.monotonic()
        out = model(prompt, max_tokens=MAX_TOKENS, temperature=0.2, stop=stop, echo=False)
        elapsed = time.monotonic() - start
        code = extract_code(out["choices"][0]["text"].strip())
        checks = check_query(code, q)
        return {
            "query": q.query,
            "domain": q.domain,
            "passed": bool(checks["passed"]),
            "completion_tokens": out.get("usage", {}).get("completion_tokens", 0) or 0,
            "time": round(elapsed, 2),
            "missing": checks["must_contain_fail"],
            "unwanted": checks["must_not_contain_fail"],
            "lines": checks["line_count"],
        }

    results = {"max_tokens": MAX_TOKENS, "aug_rerun": [], "raw_rerun": []}

    print(f"\n--- AUG re-run (max_tokens={MAX_TOKENS}) ---")
    for i, f in enumerate(aug_fails):
        r = run_one(f["query"], use_aug=True)
        status = "PASS" if r["passed"] else "FAIL"
        cap = "<<CAP>>" if r["completion_tokens"] >= MAX_TOKENS else ""
        print(f"  [{i+1}/{len(aug_fails)}] {status} tok={r['completion_tokens']:4d} {cap:7s} {r['query'][:60]}")
        results["aug_rerun"].append(r)

    print(f"\n--- RAW re-run (max_tokens={MAX_TOKENS}) ---")
    for i, f in enumerate(raw_fails):
        r = run_one(f["query"], use_aug=False)
        status = "PASS" if r["passed"] else "FAIL"
        cap = "<<CAP>>" if r["completion_tokens"] >= MAX_TOKENS else ""
        print(f"  [{i+1}/{len(raw_fails)}] {status} tok={r['completion_tokens']:4d} {cap:7s} {r['query'][:60]}")
        results["raw_rerun"].append(r)

    aug_recovered = sum(1 for r in results["aug_rerun"] if r["passed"])
    raw_recovered = sum(1 for r in results["raw_rerun"] if r["passed"])

    print("\n" + "=" * 60)
    print(f"  AUG: {aug_recovered}/{len(aug_fails)} recovered  ->  projected {189 + aug_recovered}/200")
    print(f"  RAW: {raw_recovered}/{len(raw_fails)} recovered  ->  projected {193 + raw_recovered}/200")
    print("=" * 60)

    results["aug_recovered"] = aug_recovered
    results["raw_recovered"] = raw_recovered
    results["projected_aug"] = 189 + aug_recovered
    results["projected_raw"] = 193 + raw_recovered

    with open(OUT, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)
    print(f"\n  Results: {OUT}")

    del model
    gc.collect()
    return 0


if __name__ == "__main__":
    sys.exit(main())
