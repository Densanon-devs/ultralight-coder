# Ultralight Code Assistant — Benchmark Results

**Date:** 2026-03-29
**Test Suite:** 30 tests across 4 categories (code_gen, debug, review, explain)
**Config:** max_tokens=512, temperature=0.2, gpu_layers=99, threads=8

## Rankings — Best Score (Direct Mode)

| Rank | Model | Size | Score | Speed | Best For |
|------|-------|------|-------|-------|----------|
| 1 | **Llama-3.2-1B Q4_K_M** | 770 MB | **100%** (30/30) | **28 tok/s** | **RECOMMENDED — perfect + fastest** |
| 1 | **Qwen2.5-Coder-1.5B Q4_K_M** | 1066 MB | **100%** (30/30) | 21 tok/s | Code specialist, slower |
| 1 | **Llama-3.2-1B Q8_0** | 1260 MB | **100%** (30/30) | 15 tok/s | Highest quant, slowest of 100% |
| 4 | Llama-3.2-1B Q5_K_M | 869 MB | 96.7% (29/30) | 21 tok/s | Higher quant, 1 debug miss |
| 5 | Qwen2.5-Coder-0.5B Q4_K_M | 469 MB | 80% (24/30) | **32 tok/s** | Fastest overall, weak at explanations |

## Category Breakdown — Direct (No Experts)

| Model | Code Gen (10) | Debug (8) | Review (6) | Explain (6) |
|-------|:---:|:---:|:---:|:---:|
| Llama-3.2-1B Q4 | **100%** | **100%** | **100%** | **100%** |
| Qwen2.5-Coder-1.5B | **100%** | **100%** | **100%** | **100%** |
| Llama-3.2-1B Q8 | **100%** | **100%** | **100%** | **100%** |
| Llama-3.2-1B Q5 | **100%** | 87.5% | **100%** | **100%** |
| Qwen2.5-Coder-0.5B | 90% | **100%** | **100%** | **16.7%** |

## Expert System Impact

| Model | Direct | +Expert | Delta | Verdict |
|-------|--------|---------|-------|---------|
| Qwen2.5-Coder-0.5B | 80% | **93.3%** | **+13.3%** | Experts essential |
| Llama-3.2-1B Q5 | 96.7% | **100%** | +3.3% | Experts help slightly |
| Llama-3.2-1B Q4 | **100%** | **100%** | 0 | No effect (already perfect) |
| Qwen2.5-Coder-1.5B | **100%** | 90% | **-10%** | Experts hurt! |

### Expert Impact by Category (Qwen2.5-Coder-0.5B)

| Category | Direct | +Expert | Delta |
|----------|--------|---------|-------|
| Code Gen | 90% | **100%** | +10% |
| Debug | **100%** | 87.5% | -12.5% |
| Review | **100%** | 83.3% | -16.7% |
| **Explain** | **16.7%** | **100%** | **+83.3%** |

## Key Findings

### 1. Llama-3.2-1B Q4_K_M is the winner
- **100% on all 30 tests without experts** at 28 tok/s
- 770 MB on disk — smaller than the code-specialized Qwen2.5-Coder-1.5B
- General-purpose model beats the code specialist on speed while matching quality
- Already in the PIE models directory — no new download needed

### 2. Code specialization has diminishing returns at 1B+
- Qwen2.5-Coder-1.5B matches Llama-3.2-1B on quality but is 25% slower
- The "coder" training helps at 0.5B (90% code gen) but isn't needed at 1.5B
- General models at 1B+ already understand code well enough

### 3. Expert system: essential for small models, harmful for capable ones
- **< 700MB models:** Experts boost +13% (critical for explanations)
- **700MB-1GB models:** Experts help slightly (+3%)
- **> 1GB models:** Experts hurt quality (-10%) by constraining the model
- **Threshold:** Disable experts for models scoring >95% direct

### 4. The 0.5B coder floor is viable but specialized
- 80% direct at 32 tok/s — fastest model tested
- Perfect at code gen/debug/review, terrible at explanations (16.7%)
- With experts: 93.3% — explanations jump to 100%
- Use case: embedded/constrained environments where speed > explanations

### 5. Quantization barely matters for code
- Q4_K_M vs Q5_K_M vs Q8_0 (same Llama-3.2-1B base):
  - Q4: 100%, 28 tok/s, 770 MB
  - Q5: 96.7%, 21 tok/s, 869 MB
  - Q8: 100%, 15 tok/s, 1260 MB
- Q4_K_M is optimal — highest speed, joint-highest quality, smallest size
- Higher quants trade speed for no quality gain

## Remaining Models (in progress)

Phi-3.5 3.8B, Qwen2.5 general 0.5B/1.5B, SmolLM2 135M/360M/1.7B, TinyLlama 1.1B
still benchmarking — GPU contention slowing parallel runs.

## Recommendations

### For the Ultralight Code Assistant:

**Primary model:** `Llama-3.2-1B-Instruct-Q4_K_M.gguf` (770MB)
- 100% quality, 28 tok/s, no experts needed
- Chat format: llama3
- Already available from PIE

**Floor model:** `qwen2.5-coder-0.5b-instruct-q4_k_m.gguf` (469MB)
- 93.3% with experts, 32 tok/s
- Chat format: chatml
- Enable expert system

**Config change needed:** Switch default model from Qwen2.5-Coder-1.5B to Llama-3.2-1B Q4_K_M.
