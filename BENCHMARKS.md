# Ultralight Code Assistant — Benchmark Results

**Date:** 2026-03-29
**Test Suite:** 30 tests across 4 categories (code_gen, debug, review, explain)
**Config:** max_tokens=512, temperature=0.2, gpu_layers=99, threads=8
**Models tested:** 12

## Full Rankings — Direct Mode (No Experts)

| Rank | Model | Params | Size | Score | Code | Debug | Review | Explain | Speed |
|:----:|-------|--------|------|:-----:|:----:|:-----:|:------:|:-------:|------:|
| 1 | **Llama-3.2-1B Q4_K_M** | 1.2B | 770 MB | **100%** | 100% | 100% | 100% | 100% | **28 tok/s** |
| 1 | Qwen2.5-Coder-1.5B Q4 | 1.5B | 1066 MB | **100%** | 100% | 100% | 100% | 100% | 21 tok/s |
| 1 | Qwen2.5-1.5B (general) Q4 | 1.5B | 1066 MB | **100%** | 100% | 100% | 100% | 100% | 8 tok/s* |
| 1 | Llama-3.2-1B Q8_0 | 1.2B | 1260 MB | **100%** | 100% | 100% | 100% | 100% | 15 tok/s |
| 1 | SmolLM2-1.7B Q4_K_M | 1.7B | 1007 MB | **100%** | 100% | 100% | 100% | 100% | 10 tok/s* |
| 1 | Phi-3.5 3.8B Q4_K_M | 3.8B | 2282 MB | **100%** | 100% | 100% | 100% | 100% | 7 tok/s* |
| 7 | Llama-3.2-1B Q5_K_M | 1.2B | 869 MB | 96.7% | 100% | 87.5% | 100% | 100% | 21 tok/s |
| 7 | **SmolLM2-135M Q4_K_M** | 135M | **101 MB** | **96.7%** | 100% | 100% | 83.3% | 100% | **106 tok/s** |
| 9 | Qwen2.5-0.5B (general) Q4 | 0.5B | 469 MB | 93.3% | 100% | 100% | 66.7% | 100% | 20 tok/s* |
| 10 | SmolLM2-360M Q4_K_M | 360M | 258 MB | 90% | 90% | 100% | 66.7% | 100% | 66 tok/s |
| 11 | TinyLlama-1.1B Q4_K_M | 1.1B | 638 MB | 86.7% | 100% | 87.5% | 50% | 100% | 39 tok/s |
| 12 | Qwen2.5-Coder-0.5B Q4 | 0.5B | 469 MB | 80% | 90% | 100% | 100% | **16.7%** | 32 tok/s |

*\*Speed reduced by GPU contention (two models benchmarked simultaneously)*

## Expert System Impact

| Model | Direct | +Expert | Delta | Verdict |
|-------|:------:|:-------:|:-----:|---------|
| Qwen2.5-Coder-0.5B | 80% | **93.3%** | **+13.3%** | Essential — fixes explanations |
| SmolLM2-360M | 90% | **96.7%** | **+6.7%** | Helpful |
| TinyLlama-1.1B | 86.7% | **93.3%** | **+6.6%** | Helpful — fixes reviews |
| Llama-3.2-1B Q5 | 96.7% | **100%** | +3.3% | Minor fix |
| Llama-3.2-1B Q4 | **100%** | **100%** | 0 | No effect |
| Phi-3.5 3.8B | **100%** | **100%** | 0 | No effect |
| Qwen2.5-1.5B (general) | **100%** | **100%** | 0 | No effect |
| SmolLM2-135M | **96.7%** | 90% | **-6.7%** | Hurts! |
| Qwen2.5-0.5B (general) | **93.3%** | 86.7% | **-6.6%** | Hurts! |
| Qwen2.5-Coder-1.5B | **100%** | 90% | **-10%** | Hurts! |
| SmolLM2-1.7B | **100%** | 90% | **-10%** | Hurts! |

## Key Findings

### 1. Llama-3.2-1B Q4_K_M is the clear winner
- **100% on all 30 tests** at **28 tok/s** in **770 MB**
- Fastest of the 100% models by 2x
- Smallest of the 100% models
- General-purpose model — not code-specialized
- No expert system needed

### 2. SmolLM2-135M is the shocking floor discovery
- **96.7% at 101 MB and 106 tok/s**
- A 135M parameter model passes 29/30 code tests
- Only miss: 1 code review test (review specificity)
- 100% on code generation, debugging, AND explanations
- The absolute minimum viable coding brain

### 3. Code specialization is unnecessary at 1B+
- Qwen2.5-Coder-1.5B = Qwen2.5-1.5B (general) = Llama-3.2-1B — all 100%
- The code-trained variant is slower, not better
- At 0.5B, code specialization helps code gen (90% vs 80% general) but kills explanations (16.7% vs 100%)

### 4. Expert system: helps weak models, hurts strong ones
- **Threshold pattern:** Models scoring <90% direct benefit from experts; >95% are hurt
- Experts constrain capable models by forcing them into few-shot patterns
- For the 0.5B coder, experts are essential (+13.3%, fixes explanations from 16.7% to 100%)
- For 1B+ models, disable experts entirely

### 5. Code review is the hardest category
- Only category where models consistently fail
- Race conditions and input validation are the toughest tests
- 0.5B models average 66-83% on review; 1B+ models get 100%

### 6. Quantization: Q4_K_M is optimal
- Q4 vs Q5 vs Q8 (same Llama-3.2-1B):
  - Q4: 100%, 28 tok/s, 770 MB
  - Q5: 96.7%, 21 tok/s, 869 MB (actually worse!)
  - Q8: 100%, 15 tok/s, 1260 MB
- Q4_K_M: highest speed, joint-highest quality, smallest size

## Speed vs Quality Chart

```
Quality
100% |  * Llama-1B-Q4(28)  * Coder-1.5B(21)  * Phi-3.5(7)  * SmolLM2-1.7B(10)
     |  * Llama-1B-Q8(15)  * Qwen-1.5B(8)
 97% |  * Llama-1B-Q5(21)  * SmolLM2-135M(106!)
 93% |  * Qwen-0.5B-gen(20)
 90% |  * SmolLM2-360M(66)
 87% |  * TinyLlama(39)
 80% |  * Coder-0.5B(32)
     +--+----+----+----+----+----+----+----+----+----+----> tok/s
        7   15   21   28   32   39   66        106
```

## Model Tiers

### Tier 1: Full-featured assistant (100%, no experts)
- **Llama-3.2-1B Q4_K_M** — 770MB, 28 tok/s (RECOMMENDED)
- Qwen2.5-1.5B general — 1066MB, 8-21 tok/s
- SmolLM2-1.7B — 1007MB, 10-20 tok/s
- Phi-3.5 3.8B — 2282MB, 5-9 tok/s (overkill)

### Tier 2: Viable with experts (90%+ with expert system)
- SmolLM2-360M + experts — 258MB, 66-74 tok/s, 96.7%
- TinyLlama-1.1B + experts — 638MB, 37-39 tok/s, 93.3%
- Qwen2.5-Coder-0.5B + experts — 469MB, 32 tok/s, 93.3%

### Tier 3: Ultralight floor
- **SmolLM2-135M — 101MB, 106 tok/s, 96.7%** (no experts!)
- The absolute minimum viable coding assistant

## Recommendations (Structural Only — see Execution results below)

| Use Case | Model | Size | Structural | Speed |
|----------|-------|------|------------|-------|
| Default (structural) | Llama-3.2-1B Q4_K_M | 770 MB | 100% | 28 tok/s |
| Embedded / ultra-constrained | SmolLM2-135M | 101 MB | 96.7% | 106 tok/s |

---

# EXECUTION BENCHMARK — Code That Actually Runs

**Date:** 2026-03-29
**Test Suite:** 35 execution tests (code extracted, run, assertions checked)
**Key insight: Structural benchmarks are misleading. Execution accuracy is what matters.**

## Execution Rankings

| Model | Size | Direct | +Generic | +Tuned | Perfect |
|-------|------|--------|----------|--------|---------|
| **Qwen2.5-Coder-1.5B** | **1066 MB** | **98.1%** | **98.6%** | 95.4% | **33-34/35** |
| Llama-3.2-1B Q4 | 770 MB | 79.6% | 64.6% | 75.0% | 19-24/35 |
| Qwen2.5-Coder-0.5B | 469 MB | 77.4% | — | — | 12/20* |
| SmolLM2-135M | 101 MB | 50.5% | — | — | 6/20* |

*\*Tested on 20-test suite only*

## Structural vs Execution — The Reality Check

| Model | Structural | Execution | Gap |
|-------|------------|-----------|-----|
| Llama-3.2-1B | 100% | 79.6% | **-20.4%** |
| Qwen2.5-Coder-1.5B | 100% | 98.1% | -1.9% |
| SmolLM2-135M | 96.7% | 50.5% | **-46.2%** |

**Structural benchmarks check "does it look like code". Execution benchmarks check "does it work".**

## Expert System Impact (Execution)

| Model | Direct | Generic | Tuned | Verdict |
|-------|--------|---------|-------|---------|
| Coder-1.5B | 98.1% | **98.6%** (+0.5) | 95.4% (-2.7) | Generic barely helps, tuned hurts |
| Llama-1B | **79.6%** | 64.6% (-15.0) | 75.0% (-4.6) | ALL experts hurt execution |

**Conclusion: Disable experts for execution accuracy.** Few-shot examples confuse models into producing subtly wrong code that passes structural checks but fails test cases.

## Where Models Fail (Execution)

### Qwen2.5-Coder-1.5B failures (2/35):
- **fibonacci**: Generates `fibonacci(2)=1` but breaks on `fibonacci(5)=5` (off-by-one in loop range)
- Test harness edge case (try/except block parsing)

### Llama-3.2-1B failures (11/35):
- **Kadane's algorithm**: Doesn't handle all-negative arrays
- **Roman numerals**: Misses subtractive notation (IV=4, IX=9)
- **Matrix multiply**: Syntax errors in generated code
- **Deep merge**: Incorrect recursive dictionary merging
- **Run-length encoding**: Wrong output format
- Several algorithm edge cases

## Final Recommendation

| Use Case | Model | Size | Execution | Config |
|----------|-------|------|-----------|--------|
| **Default / recommended** | **Qwen2.5-Coder-1.5B Q4** | **1066 MB** | **98.1%** | **chatml, experts OFF** |
| Speed-priority fallback | Llama-3.2-1B Q4 | 770 MB | 79.6% | llama3, experts OFF |
| Ultra-constrained | Qwen2.5-Coder-0.5B | 469 MB | 77.4% | chatml, experts OFF |

**The lesson: never trust structural benchmarks alone. Run the code.**
