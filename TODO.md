# Ultralight Code Assistant — Roadmap

## Status: Core Complete, Pre-Release

**What's done:**
- 232 YAML augmentor examples across 39 categories
- Auto mode (rerank1/rerank based on model size)
- 34 failure routing categories, 251 trigger keywords
- 399/400 on real-world benchmark (4 sets, 100 queries each)
- Web UI (static/index.html), one-click launcher (launch.py)
- FastAPI server with /generate, /status, /models/available

---

## Priority 1: Ship-Ready (must-have before handing to users)

### 1.1 Multi-Turn Conversation in Web UI
**Why:** Users expect "now add error handling to that" to work. Without this, every response starts from scratch.

**What exists:** The engine has a full MemorySystem (engine/memory.py) with short-term turns (30 messages) and FAISS long-term memory. The `process()` method in main.py already stores conversation history.

**What's missing:**
- Web UI (static/index.html) sends each prompt independently — no conversation ID or history
- Server endpoint /generate doesn't accept or track conversation context
- Need: add `session_id` to /generate, send prior messages as context, FusionLayer already has `budget_conversation: 0.55` allocated for this

**Estimated scope:** ~50 lines in server.py + ~20 lines in index.html

### 1.2 Paste-and-Fix Flow (Code Review / Debug)
**Why:** Real usage is "here's my broken code, fix it" not just "write me a function."

**What exists:** code_review and debugger modules with system prompts. Router already detects review/debug keywords. AugmentorRouter has code_review and debugger augmentors.

**What's missing:**
- Web UI has no way to paste code + describe the problem
- Server doesn't route to code_review/debugger modules based on the query
- Need: textarea for code input in UI, pass module_hint based on query keywords, show diff-style output

**Estimated scope:** ~40 lines in index.html (code input area) + ~20 lines in server.py (module routing)

---

## Priority 2: Polish (nice-to-have before release)

### 2.1 Suppress Noisy Startup Logs
The HuggingFace HTTP requests, BertModel LOAD REPORT, and llama_context warnings clutter the console. Users don't need to see these.
- Set httpx and sentence_transformers loggers to WARNING
- Redirect llama.cpp verbose output

### 2.2 Model Download Helper
Users need models but there's no download script. Add a `download_model.py` or integrate into launch.py:
- Offer Qwen 0.5B (469MB, fast) and 1.5B (1.1GB, balanced) as defaults
- Download from HuggingFace with progress bar
- Verify file integrity after download

### 2.3 Offline Mode for Sentence-Transformers
The embedder tries to reach HuggingFace on every startup (HEAD requests). Cache the model locally and add `TRANSFORMERS_OFFLINE=1` support so it works fully offline.

### 2.4 Streaming Responses in Web UI
The /generate/stream SSE endpoint exists in server.py. Wire it into the web UI so code appears token-by-token instead of waiting for the full response.

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
