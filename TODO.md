# Ultralight Code Assistant — Roadmap

## Status: v0.1.0 Release

**What's done:**
- 359 YAML augmentor examples across 12 languages (Python, JS/TS, Go, Rust, SQL, C, Java, C#, Ruby, Bash, Kotlin, Swift)
- 99.2% on 130-query multi-language benchmark, 99.8% on 400-query Python benchmark
- Auto mode with language-aware routing (rerank1 for <1.5GB, rerank for >=1.5GB)
- 50+ failure routing categories with language detection boosting
- Web UI with streaming, multi-turn conversation, code paste-and-fix, project indexing
- FastAPI server with /generate, /generate/stream, /session/reset, /project/index, /project/status
- Project context: index a codebase for context-aware code generation
- One-click launcher (launch.py) with dep check, GPU detection, model selection
- Noisy startup logs suppressed, offline mode for sentence-transformers
- Model download helper (download_model.py with 8 model options)
- README, LICENSE (MIT), pyproject.toml, clean .gitignore

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
| 10: Multi-language | 2026-04-02 | 12 languages, 437 examples, 99.2% benchmark, language-aware routing |
| 11: Project context | 2026-04-02 | Codebase indexing + retrieval, FAISS-backed, UI integration |
| 12: v0.1.0 ship prep | 2026-04-02 | README, LICENSE, pyproject.toml, requirements audit, .gitignore |
