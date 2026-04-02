# Ultralight Code Assistant — Roadmap

## Status: Ship-Ready, Polished

**What's done:**
- 232 YAML augmentor examples across 39 categories
- Auto mode (rerank1/rerank based on model size)
- 34 failure routing categories, 251 trigger keywords
- 399/400 on real-world benchmark (4 sets, 100 queries each)
- Web UI (static/index.html), one-click launcher (launch.py)
- FastAPI server with /generate, /generate/stream, /session/reset, /status, /models/available
- Multi-turn conversation (engine memory persists across requests, New Chat resets)
- Paste-and-fix flow (code input area in UI, code field sent to /generate)
- Streaming responses in Web UI (SSE with fallback to non-streaming)
- Noisy startup logs suppressed (httpx, sentence_transformers, llama.cpp)
- Offline mode for sentence-transformers (TRANSFORMERS_OFFLINE + HF_HUB_OFFLINE)
- Model download helper (download_model.py with 8 model options)

---

## Priority 3: Expand (after initial release)

### 3.1 Multi-Language Support
Augmentor examples are Python-only. Add packs for:
- JavaScript/TypeScript (most-requested after Python)
- Go (good for small model code gen — simple syntax)
- Rust (harder but high demand)
- SQL (standalone queries)

### 3.2 Project Context
Let users point at a directory and have the system understand their codebase:
- Index files with embeddings
- Include relevant snippets in the prompt
- Understand imports and framework usage

### 3.3 Multi-Agent Pipeline (branch: multi-agent-architect)
3B architect decomposes, 0.5B workers build pieces, 3B assembles.
**Status:** Tested, decomposition works (8-9/10), assembly needs work (3-5/10).
**TODO on that branch:**
1. Pass interface specs to workers so pieces connect
2. Add execution check after assembly (run code, feed errors back)
3. Speed up assembly (concatenate + fix imports, don't rewrite)
4. Add auth/hashing keywords to failure routing

### 3.4 VS Code Extension
Wrap the FastAPI server as a VS Code extension backend. Users select code, right-click, "Generate/Review/Explain with Ultralight."

### 3.5 Benchmark Hardening
- Add execution tests to real-world benchmark (actually run the generated code)
- Track non-deterministic variance (run each query 3x, report consistency)
- Test with deliberately adversarial prompts

---

## Completed Phases

| Phase | Date | Result |
|-------|------|--------|
| 1: Structural benchmark | 2026-03-29 | Proved structural tests are useless |
| 2: Execution benchmark | 2026-03-29 | Qwen 1.5B at 98.1%, direct > augmented |
| 3: Stress benchmark | 2026-03-30 | 0.5B+experts (49%) > 3B direct (42%) |
| 4: Programmer pack | 2026-03-30 | +26-29% boost, 25 examples |
| 5: YAML modularization | 2026-03-30 | 0.5B: 94%, 1.5B: 100% |
| 6: Retrieval strategy | 2026-04-01 | Single injection best for sub-3B |
| 6b: Auto mode | 2026-04-01 | Model-size-adaptive matches hand-tuned |
| 7: Real-world 400 queries | 2026-04-01 | 399/400 (99.8%) |
| 8: Multi-agent (branch) | 2026-04-01 | Decomposition 9/10, assembly 4/10 |
| 9: Ship-ready polish | 2026-04-02 | Multi-turn, paste-fix, streaming, log suppression, offline mode |
