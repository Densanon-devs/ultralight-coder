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

# Phase 3: Stress Benchmark (17 tests + 8 multi-turn steps)

**Date:** 2026-03-30
**Benchmark tool:** `benchmark_stress.py`

The execution benchmark (Phase 2) tested single functions. Real coding requires multi-method classes, mini-apps, and incremental builds. This benchmark pushes models with three tiers of increasing difficulty.

## Tier Structure

| Tier | Difficulty | max_tokens | What it tests |
|------|-----------|-----------|---------------|
| **Tier 1** | Medium (10 tests) | 1024 | Multi-method classes: LRU Cache, Trie, Graph, Event Emitter, Expression Evaluator, CSV Parser, JSON Flatten/Unflatten, Retry Decorator, Rate Limiter, State Machine |
| **Tier 2** | Heavy (5 tests) | 2048 | Mini-apps in one prompt: KV Store with TTL, Markdown-to-HTML, Task Scheduler with dependencies, Mini ORM, HTTP Router with path params |
| **Tier 3** | Multi-turn (2 tests, 4 steps each) | 1024/step | Incremental app building: Todo App (dataclass -> priority -> persistence -> search/stats), Calculator (basic -> parens -> variables -> functions) |

## Overall Rankings

| # | Model | Params | Size | Tier 1 | Tier 2 | Tier 3 | Overall | Time |
|---|-------|--------|------|:------:|:------:|:------:|:-------:|-----:|
| **1** | **Qwen2.5-Coder-3B** | **3B** | **2.0 GB** | **55%** | **18%** | **53%** | **42%** | 1081s |
| 2 | DeepSeek-Coder-1.3B | 1.3B | 834 MB | 39% | 11% | 27% | 26% | ~700s |
| 3 | Qwen2.5-Coder-1.5B | 1.5B | 1.1 GB | 42% | 26% | 24% | 31% | 698s |
| 4 | Qwen2.5-Coder-0.5B | 0.5B | 469 MB | 32% | 0% | 33% | 22% | 482s |

## Tier 1 — Medium: Detailed Results

| Test | Qwen 3B | Qwen 1.5B | DeepSeek 1.3B | Qwen 0.5B |
|------|:-------:|:---------:|:------------:|:---------:|
| LRU Cache | **100%** | **100%** | **100%** | **100%** |
| Expression Evaluator | 75% | 0% | 0% | 0% |
| Trie | 64% | 64% | **100%** | 55% |
| Event Emitter | 43% | 43% | 43% | 43% |
| CSV Parser | 75% | 75% | **100%** | 0% |
| JSON Flatten/Unflatten | **88%** | 38% | 0% | 25% |
| Retry Decorator | 0% | 0% | 0% | 0% |
| Graph BFS/DFS | 75% | 75% | 0% | 25% |
| Rate Limiter | 29% | 29% | 50% | **64%** |
| State Machine | 0% | 0% | 0% | 10% |

**Universal failures:** Retry Decorator (none could add `.attempts` attribute), State Machine (property + guard pattern too complex).

## Tier 2 — Heavy: Detailed Results

| Test | Qwen 3B | Qwen 1.5B | DeepSeek 1.3B | Qwen 0.5B |
|------|:-------:|:---------:|:------------:|:---------:|
| KV Store + TTL | 0% | **100%** | 6% | 0% |
| Markdown to HTML | 0% | 0% | **50%** | 0% |
| Task Scheduler | **90%** | 30% | 0% | 0% |
| Mini ORM | 0% | 0% | 0% | 0% |
| HTTP Router | 0% | 0% | 0% | 0% |

**Universal failures:** Mini ORM (metaclass/descriptor pattern beyond all models), HTTP Router (decorator + regex path matching too complex).

**Surprise:** Qwen 1.5B aced the KV Store (18/18) while the larger 3B scored 0%. Non-deterministic generation + prompt sensitivity.

## Tier 3 — Multi-Turn: Detailed Results

### Todo App (4 steps)

| Step | Qwen 3B | Qwen 1.5B | DeepSeek 1.3B | Qwen 0.5B |
|------|:-------:|:---------:|:------------:|:---------:|
| 1: Basic CRUD | **100%** | **100%** | 78% | **100%** |
| 2: Priority | 75% | 25% | 75% | 75% |
| 3: Persistence (JSON) | **100%** | 0% | 0% | 0% |
| 4: Search/Stats | **100%** | 20% | 60% | **90%** |
| **Overall** | **94%** | 36% | 53% | **66%** |

### Calculator (4 steps) — HARD

| Step | Qwen 3B | Qwen 1.5B | DeepSeek 1.3B | Qwen 0.5B |
|------|:-------:|:---------:|:------------:|:---------:|
| 1: Basic +,-,*,/ | 0% | 0% | 0% | 0% |
| 2: Parentheses | 0% | 0% | 0% | 0% |
| 3: Variables | 0% | 0% | 0% | 0% |
| 4: Functions | 50% | 50% | 0% | 0% |
| **Overall** | 12% | 12% | 0% | 0% |

**The Calculator is a model killer.** Building a recursive-descent parser with operator precedence is beyond every model tested — even step 1 (basic arithmetic with precedence) fails universally.

## Execution vs Stress — The Gap

| Model | Execution (Phase 2) | Stress (Phase 3) | Drop |
|-------|:-------------------:|:-----------------:|:----:|
| Qwen2.5-Coder-3B | 100% | 42% | **-58%** |
| Qwen2.5-Coder-1.5B | 98.1% | 31% | **-67%** |
| DeepSeek-Coder-1.3B | 96.9% | 26% | **-71%** |
| Qwen2.5-Coder-0.5B | 77.4% | 22% | **-55%** |

---

# Key Lessons Learned

## From Phase 2 (Execution)

1. **Structural benchmarks lie.** SmolLM2-135M scored 96.7% structural but 50.5% execution. Always run the code.

2. **Code training > model size.** DeepSeek-1.3B (834MB) beats Llama-3.2-1B (770MB) by 17 points. StarCoder2-3B (1.8GB) loses to both.

3. **Not all code training is equal.** Qwen-Coder family dominates. StarCoder2 and Stable-Code are far behind at the same size.

4. **Quality boosting hurts good models.** Experts, self-repair pipelines, and multi-model strategies all degrade execution accuracy. The model's first attempt is its best.

5. **Lean orchestration is the right approach.** Strip prompt overhead, give the model clean input, let it code. The PIE system's value for coding is session management, not generation boosting.

6. **PIE's quality boosters shine for NPCs, not code.** Few-shot experts work when "sounds right" = "is right" (dialogue). For code, "looks right" != "runs correctly."

## From Phase 3 (Stress)

7. **Simple function benchmarks massively overrate models.** 98% on single functions drops to 31% on real tasks. The gap between "write fibonacci" and "build an LRU cache class" is enormous for sub-3B models.

8. **Multi-turn is where 3B pulls ahead.** Qwen 3B scored 94% on the Todo app (4 incremental steps) while 1.5B scored 36%. Larger context and better instruction following matter for iterative development.

9. **Some patterns are universally beyond small models.** Retry decorators with function attributes, recursive-descent parsers with operator precedence, metaclass-based ORMs, and decorator-based HTTP routers all scored 0% across every model.

10. **Model behavior is non-deterministic and surprising.** Qwen 1.5B aced the KV Store (100%) while the larger 3B scored 0% on the same test. Don't assume bigger always means better on any specific task.

11. **The "medium" tier is the real differentiator.** Tier 2 (heavy) was too hard for all models. Tier 1 (medium) — multi-method classes, data structures, parsers — is where you see meaningful quality differences between models.

12. **Multi-turn persistence (JSON save/load) is a cliff.** All models except Qwen 3B failed the persistence step of the Todo app. Coordinating dataclass serialization with file I/O and state recovery is a specific weakness.
