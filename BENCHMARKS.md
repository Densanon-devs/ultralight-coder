# Ultralight Code Assistant — Benchmark Results

**Date:** 2026-03-29
**Models tested:** 20 total (12 structural, 8 execution)
**Benchmark tools:** `benchmark.py` (structural), `benchmark_exec.py` (execution)

---

# Phase 1: Structural Benchmark (30 tests)

Checks "does it look like code" — function definitions, return statements, keyword presence.

**Config:** max_tokens=512, temperature=0.2, gpu_layers=99, threads=8

## Structural Rankings (12 models, direct mode)

| Rank | Model | Params | Size | Score | Speed |
|:----:|-------|--------|------|:-----:|------:|
| 1 | Llama-3.2-1B Q4_K_M | 1.2B | 770 MB | **100%** | **28 tok/s** |
| 1 | Qwen2.5-Coder-1.5B Q4 | 1.5B | 1066 MB | **100%** | 21 tok/s |
| 1 | Qwen2.5-1.5B (general) Q4 | 1.5B | 1066 MB | **100%** | 8 tok/s |
| 1 | Llama-3.2-1B Q8_0 | 1.2B | 1260 MB | **100%** | 15 tok/s |
| 1 | SmolLM2-1.7B Q4_K_M | 1.7B | 1007 MB | **100%** | 10 tok/s |
| 1 | Phi-3.5 3.8B Q4_K_M | 3.8B | 2282 MB | **100%** | 7 tok/s |
| 7 | Llama-3.2-1B Q5_K_M | 1.2B | 869 MB | 96.7% | 21 tok/s |
| 7 | SmolLM2-135M Q4_K_M | 135M | 101 MB | 96.7% | 106 tok/s |
| 9 | Qwen2.5-0.5B (general) Q4 | 0.5B | 469 MB | 93.3% | 20 tok/s |
| 10 | SmolLM2-360M Q4_K_M | 360M | 258 MB | 90% | 66 tok/s |
| 11 | TinyLlama-1.1B Q4_K_M | 1.1B | 638 MB | 86.7% | 39 tok/s |
| 12 | Qwen2.5-Coder-0.5B Q4 | 0.5B | 469 MB | 80% | 32 tok/s |

## Expert System Impact (Structural)

| Model | Direct | +Expert | Delta |
|-------|:------:|:-------:|:-----:|
| Qwen2.5-Coder-0.5B | 80% | 93.3% | **+13.3%** |
| SmolLM2-360M | 90% | 96.7% | +6.7% |
| Llama-3.2-1B Q4 | 100% | 100% | 0 |
| Qwen2.5-Coder-1.5B | 100% | 90% | **-10%** |
| SmolLM2-1.7B | 100% | 90% | **-10%** |

---

# Phase 2: Execution Benchmark (35 tests)

**THE REAL TEST.** Extracts generated code, executes it, runs assertions.
Structural benchmarks were dangerously misleading — this is what matters.

## Execution Rankings — All Code Models

| # | Model | Params | Size | Execution | Perfect | Notes |
|---|-------|--------|------|:---------:|:-------:|-------|
| **1** | **Qwen2.5-Coder-3B** | **3B** | **2.0 GB** | **100%** | **35/35** | **Only model to ace every test** |
| **2** | **Qwen2.5-Coder-1.5B** | **1.5B** | **1.1 GB** | **98.1%** | **33/35** | **Best under 1.5GB** |
| **3** | **DeepSeek-Coder-1.3B** | **1.3B** | **834 MB** | **96.9%** | **33/35** | **Best under 1GB** |
| 4 | Yi-Coder-1.5B | 1.5B | 920 MB | 95.7% | 32/35 | Solid but beaten by DeepSeek |
| 5 | Llama-3.2-1B (general) | 1.2B | 770 MB | 79.6% | 24/35 | General model can't compete |
| 6 | Qwen2.5-Coder-0.5B | 0.5B | 469 MB | 77.4% | 12/20 | Floor model |
| 7 | StarCoder2-3B | 3B | 1.8 GB | 73.0% | 24/35 | Bad — 3B model worse than Llama 1B |
| 8 | Stable-Code-3B | 3B | 1.6 GB | 34.3% | 12/35 | Catastrophic — do not use |

**Removed from disk:** StarCoder2-3B, Stable-Code-3B, Yi-Coder-1.5B (outperformed by smaller DeepSeek)

## Structural vs Execution — The Reality Check

| Model | Structural | Execution | Gap |
|-------|:----------:|:---------:|:---:|
| Qwen2.5-Coder-3B | 100%* | **100%** | 0 |
| Qwen2.5-Coder-1.5B | 100% | 98.1% | -1.9% |
| DeepSeek-Coder-1.3B | — | 96.9% | — |
| Llama-3.2-1B | 100% | 79.6% | **-20.4%** |
| SmolLM2-135M | 96.7% | 50.5% | **-46.2%** |

**Lesson: structural benchmarks flatter models that write plausible-looking but broken code.**

## Quality Boosting Strategies Tested (Execution)

Every strategy we tested made Qwen-Coder-1.5B worse:

| Strategy | Score | Delta vs Direct |
|----------|:-----:|:---------------:|
| **Direct (no tricks)** | **98.1%** | **baseline** |
| Generic experts | 98.6% | +0.5% |
| Pipeline (self-repair) | 94.8% | **-3.3%** |
| Tuned experts | 95.4% | **-2.7%** |
| Multi-model (Llama debug) | 96.9% | **-1.2%** |

**The model's first attempt is its best attempt.** The PIE expert system, self-repair pipeline, and multi-model debugging all degrade execution accuracy. The orchestration layer's value for code is session management (memory, routing, caching, API), not generation quality boosting.

## Orchestration Optimization

Added **lean fusion mode** — strips XML tags, eliminates module context bloat, gives the model the cleanest possible prompt:

```
Structured mode: ~500+ chars of overhead (XML tags, module injections, format instructions)
Lean mode:       ~182 chars total (system + user, nothing else)
```

The model gets maximum attention budget for the actual code task.

## Where the Top Models Fail

### Qwen2.5-Coder-3B: 0 failures (perfect)

### Qwen2.5-Coder-1.5B failures (2/35):
- **fibonacci**: Off-by-one in loop initialization (systematic — same bug on every retry)
- Test harness edge case (try/except block parsing in benchmark)

### DeepSeek-Coder-1.3B failures (2/35):
- Minor edge cases in algorithm implementations

### Llama-3.2-1B failures (11/35):
- Kadane's algorithm, Roman numerals, Matrix multiply, Deep merge, Run-length encoding, and more

---

# Final Model Selection

## Kept on disk (4 models, 4.3GB total):

| Tier | Model | Size | Execution | Use Case |
|------|-------|------|:---------:|----------|
| **Premium** | Qwen2.5-Coder-3B Q4 | 2.0 GB | **100%** | Maximum quality |
| **Default** | Qwen2.5-Coder-1.5B Q4 | 1.1 GB | 98.1% | Best balance |
| **Efficient** | DeepSeek-Coder-1.3B Q4 | 834 MB | 96.9% | Near-premium, small |
| **Floor** | Qwen2.5-Coder-0.5B Q4 | 469 MB | 77.4% | Constrained environments |

## Removed from disk:
- Yi-Coder-1.5B (95.7% — beaten by DeepSeek-1.3B at smaller size)
- StarCoder2-3B (73.0% — 1.8GB for worse-than-Llama quality)
- Stable-Code-3B (34.3% — catastrophic)

## Config:
- **Default model:** `qwen2.5-coder-1.5b-instruct-q4_k_m.gguf`
- **Chat format:** chatml
- **Fusion mode:** lean (minimal prompt overhead)
- **Experts:** DISABLED (hurt execution accuracy)
- **Temperature:** 0.2

---

# Key Lessons Learned

1. **Structural benchmarks lie.** SmolLM2-135M scored 96.7% structural but 50.5% execution. Always run the code.

2. **Code training > model size.** DeepSeek-1.3B (834MB) beats Llama-3.2-1B (770MB) by 17 points. StarCoder2-3B (1.8GB) loses to both.

3. **Not all code training is equal.** Qwen-Coder family dominates. StarCoder2 and Stable-Code are far behind at the same size.

4. **Quality boosting hurts good models.** Experts, self-repair pipelines, and multi-model strategies all degrade execution accuracy. The model's first attempt is its best.

5. **Lean orchestration is the right approach.** Strip prompt overhead, give the model clean input, let it code. The PIE system's value for coding is session management, not generation boosting.

6. **PIE's quality boosters shine for NPCs, not code.** Few-shot experts work when "sounds right" = "is right" (dialogue). For code, "looks right" != "runs correctly."
