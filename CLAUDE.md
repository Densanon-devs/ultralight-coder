# Ultralight Coder

Local AI coding assistant powered by tiny GGUF models (0.5B-3B). Augmentor system injects curated YAML examples to make small models perform like models 10x their size.

**Key results:** 399/400 (99.8%) on real-world Python benchmark with 469 MB default. 129/130 (99.2%) on multi-language benchmark. Phase 13 added 14B support: both Qwen 2.5 Coder 14B and Qwen 2.5 14B hit **198/200 (99.0%) on V1+V2** and **127–128/130 (97.7–98.5%) on multi-language** via auto mode + language-scoped augmentor retrieval. **Phase 14** added agent mode: an autonomous ReAct loop with 9 builtin tools, auto-syntax-checking, risky-tool confirmation, and per-project cross-session memory — all in-process, zero servers.

## Quick Reference

| Layer | Stack | Entry Point |
|-------|-------|-------------|
| Engine | Python 3.10+, llama-cpp-python, FAISS, sentence-transformers | `engine/` |
| CLI | Interactive REPL | `main.py` |
| API | FastAPI, Uvicorn | `server.py` |
| Web UI | HTML/CSS/JS (dark theme, streaming) | `static/index.html` |
| Launcher | One-click setup + start | `launch.py` |

## Project Structure

```
ultralight-coder/
├── launch.py                    # One-click launcher (deps, GPU detect, model select, web UI)
├── server.py                    # FastAPI REST API + web UI server
├── main.py                      # CLI entry point + engine orchestrator
├── config.yaml                  # System configuration
├── download_model.py            # Model downloader (8 model options)
├── install.sh / install.ps1     # One-line installers (Linux/macOS/Windows)
├── start.bat                    # Windows zero-install launcher
├── engine/                      # Core inference + augmentor system
│   ├── augmentors.py            # YAML augmentor system (core innovation)
│   ├── architect.py             # Multi-agent task decomposition (big→plan, small→build)
│   ├── model_router.py          # Multi-model swapping (speed vs quality)
│   ├── fusion.py                # Prompt assembly (token budgets, XML tags)
│   ├── project_context.py       # Codebase indexing + FAISS semantic retrieval
│   ├── example_loader.py        # Loads YAML augmentor examples
│   ├── pattern_graph.py         # Dependency graph for pattern-based retrieval
│   ├── config.py                # Configuration dataclasses
│   ├── base_model.py            # GGUF model loading (llama-cpp-python)
│   ├── embedder.py              # Shared sentence-transformer pool
│   ├── router.py                # Query routing to modules (keyword + neural)
│   ├── memory.py                # Short-term + FAISS long-term memory
│   ├── pipeline.py              # Async thread pool for parallel I/O
│   ├── kv_cache.py              # KV cache optimization + compression
│   ├── module_manager.py        # Dynamic module loading
│   ├── micro_adapters.py        # Task-specific LoRA adapters
│   ├── tuner.py                 # Hyperparameter tuning
│   ├── tools.py                 # Legacy utility tool registry (calculate, run_python, ...)
│   ├── agent.py                 # Phase 14: autonomous ReAct loop (Agent class)
│   ├── agent_tools.py           # Phase 14: Hermes-format tool registry + parser
│   ├── agent_builtins.py        # Phase 14: 8 builtin agent tools (read/write/edit/list/glob/grep/bash/tests) + optional remember
│   ├── agent_memory.py          # Phase 14: per-project cross-session notes
│   ├── classifier.py            # Neural classifier for routing
│   └── speculative.py           # Speculative decoding
├── modules/                     # Task-specific modules (YAML manifests)
│   ├── code_gen/                # "write", "create", "implement" (priority 12)
│   ├── code_review/             # "review", "refactor", "optimize" (priority 10)
│   ├── debugger/                # "bug", "fix", "error" (priority 11)
│   └── explainer/               # "explain", "how", "why" (priority 8)
├── data/
│   ├── augmentor_examples/      # 359+ YAML examples (core innovation)
│   │   ├── pattern/             # 17 patterns (decorator, router, ORM, parser, state_machine...)
│   │   ├── algorithm/           # Tree, sorting, search
│   │   ├── async/               # Coroutine, concurrency, queue
│   │   ├── web/                 # Routes, middleware, validation
│   │   ├── database/            # CRUD, connection pooling, repository
│   │   ├── testing/             # Fixtures, mocking, patterns
│   │   ├── data_processing/     # Pandas, ETL
│   │   ├── common/              # Basics, review, debug, explain
│   │   ├── python/              # Python-specific
│   │   ├── javascript/          # JS/TS-specific
│   │   ├── go/                  # Go-specific
│   │   ├── rust/                # Rust-specific
│   │   └── [sql, c, java, csharp, ruby, bash, kotlin, swift]/
│   ├── pattern_graph.yaml       # Dependency relationships (37 nodes)
│   ├── project_index/           # Indexed projects (FAISS + metadata)
│   ├── router_model/            # Trained neural classifier
│   ├── memory/                  # Long-term FAISS indices
│   └── module_stats.json        # Module usage statistics
├── static/
│   └── index.html               # Web UI (dark Tokyonight theme, streaming, code input)
├── models/                      # GGUF model files (downloaded separately)
├── benchmark*.py                # Benchmark scripts
├── BENCHMARKS.md                # Detailed results (phases 1-12)
├── pyproject.toml               # Package metadata + dependencies
└── requirements.txt             # Python dependencies
```

## Commands

```bash
# Launcher (recommended)
python launch.py                  # Web UI (default)
python launch.py --cli            # CLI mode
python launch.py --port 9000      # Custom port

# Server
python server.py                  # API + web UI on :8000

# CLI
python main.py                    # Interactive REPL
python main.py --dry-run          # Test routing without model
python main.py --list-modules     # Show modules
python main.py --explain "prompt" # Explain routing
python main.py --agent "GOAL"     # Phase 14: run one autonomous task in cwd

# Inside the REPL
/agent GOAL                       # Phase 14: launch the agent loop on a goal

# Model download
python download_model.py --model coder-0.5b   # 469MB (default)
python download_model.py --model llama         # 770MB
python download_model.py --model coder-1.5b    # 1.1GB
python download_model.py --model coder-3b      # 2.0GB
python download_model.py --all                 # All models

# Benchmarks
python benchmark_realworld.py --both           # V1+V2 (200 queries)
python benchmark_realworld_v3.py               # V3 edge cases (100)
python benchmark_realworld_v4.py               # V4 deep gaps (100)
python benchmark_multilang.py                  # 130 queries, 12 languages
python benchmark_stress.py                     # Multi-method classes

# Install (one-liner)
curl -fsSL .../install.sh | bash              # Linux/macOS
irm .../install.ps1 | iex                     # Windows
```

## Architecture

### Augmentor System (engine/augmentors.py) — The Core Innovation

Three mechanisms that make 0.5B models competitive:

1. **Dynamic Few-Shot Retrieval** — Embeds user query, retrieves most similar YAML example from 359+ examples, injects as demonstration in prompt.

2. **Failure-Aware Routing** — 50+ keyword patterns route to specific examples for known hard cases (e.g., "expression evaluator" → pattern_parser.yaml). Bypasses similarity scoring.

3. **Auto Mode** — Detects model size. Uses `rerank1` (1 example) for <1.5GB models, `rerank` (2 examples) for >=1.5GB. Configurable via `config.yaml`.

**Key insight:** 90KB of YAML examples > 1.5GB of extra model parameters.

### Routing
- Modes: `rule_based`, `classifier`, `hybrid` (default)
- 4 modules: code_gen, code_review, debugger, explainer
- Up to 2 active modules, weighted blending with 0.7 decay

### Fusion (engine/fusion.py)
- Modes: `lean` (minimal overhead, code-optimized), `structured` (XML tags), `simple` (v1 compat)
- Token budget: 5% system, 10% modules, 10% memory, 55% conversation, 20% reserve

### Project Context (engine/project_context.py)
- Walks project directory (respects .gitignore)
- Chunks files, embeds with sentence-transformer, stores in FAISS
- Retrieves top-k relevant chunks on query

### Multi-Agent Architect (engine/architect.py)
- Architect (3B) decomposes task → 2-5 subtasks
- Workers (0.5B) build each piece with augmentors
- Assembler (3B) stitches into final module

### Multi-Model Router (engine/model_router.py)
- Modes: `single` (one model), `speed` (fastest per category), `quality` (best per category)

### Agent Mode (engine/agent.py + agent_tools.py + agent_builtins.py + agent_memory.py)

Phase 14 ReAct loop. The agent is **loader-agnostic** — it accepts any object with a `.generate(prompt, max_tokens, stop)` method, so the same `Agent` class works against the loaded `BaseModel`, against stub models in unit tests, and against any future inference backend.

**Loop:** Build ChatML prompt from running transcript → `model.generate()` → parse `<tool_call>` tags → if none, that's the final answer; if any, execute every call (with risky-confirm hook), append tool results as a `<|im_start|>tool` turn with Hermes `<tool_response>` blocks, repeat. Budgets: `max_iterations=20`, `max_wall_time=600s`, `max_tokens_per_turn=1024`.

**Tool format:** Qwen 2.5 native Hermes format: `<tool_call>{"name": "...", "arguments": {...}}</tool_call>`. Parser uses `json.JSONDecoder(strict=False)` so literal newlines/tabs inside string values (common for multi-line `write_file` content) parse cleanly. Tool schemas are emitted in the system prompt inside a `<tools>...</tools>` block, exactly the format the model was trained on.

**Builtin tools** (`engine/agent_builtins.py`, 9 total):

| Tool | Resolves paths via | Notes |
|---|---|---|
| `read_file` | Workspace | line-numbered, offset/limit for big files |
| `write_file` | Workspace | creates parent dirs |
| `edit_file` | Workspace | exact-string replace, fails on multi-match unless `replace_all` |
| `list_dir` | Workspace | depth 1–5, skips `__pycache__`/`node_modules`/`.git`/etc. |
| `glob` | Workspace | sorted by mtime, max 200 results |
| `grep` | Workspace | ripgrep if installed, stdlib `re + os.walk` fallback |
| `run_bash` | cwd or Workspace | **risky** — agent loop prompts y/N before executing |
| `run_tests` | Workspace | pytest / unittest / npm / go / cargo |
| `remember` | (memory store) | optional — only registered when `AgentMemory` is provided |

**Auto-verification:** After every successful `write_file`/`edit_file` on a `.py` file, the agent runs `compile()` on the result and threads a synthetic `auto_verify` `ToolResult` into the same observation block. If it's a `SyntaxError`, the model sees it on the very next turn and can fix it before continuing. Skipped cleanly if the file isn't resolvable (no false negatives).

**Cross-session memory** (`engine/agent_memory.py`): per-project notes at `~/.ultralight-coder/memory/<sha256(workspace)[:12]>/notes.md`. Loaded into the system prompt at run start under a `# Notes from previous sessions` block. The `remember` tool lets the model append a note mid-loop. Append-only Markdown bullets, capped at 30 entries by default, 500 chars per note.

**Risky-tool confirmation:** `run_bash` is flagged `risky=True`. The agent's `confirm_risky` callback (in `main.py._confirm_risky_tool`) shows the proposed args and prompts `Approve? [y/N]`. Default deny on Ctrl+C/EOF. Denied calls return a `ToolResult(success=False, error="User denied...")` that the model sees and can recover from.

**Privacy invariant:** The entire agent runs in-process. No HTTP, no sockets, no localhost webview. The `remember` notes file is the only thing written outside cwd, and it stays on the local disk.

**Entry points:**
- `python main.py --agent "GOAL"` — one-shot, exits after the agent finishes. Uses `UltraliteCodeAssistant.run_agent_fast()`, a static lightweight entry that loads **only** Config + BaseModel + Agent. Skips the full UCA stack (Router, Modules, Memory, Fusion, AugmentorRouter, sentence-transformer embedder, FAISS, ProjectIndex) because agent mode doesn't use any of it. On an RTX 3060 with Qwen 2.5 Coder 14B, this drops startup from ~2 minutes (UCA full init + VRAM contention with the PyTorch embedder on CUDA) to ~4 seconds, and agent loop inference runs at the expected ~20 tok/s. Opt out with `--agent-full-init` if you specifically need the full stack.
- REPL `/agent GOAL` — interactive, agent runs and control returns to the REPL. Uses the live UCA's `run_agent()` since UCA is already loaded — no fast-path benefit because the heavy init already happened.
- Both paths construct an `Agent` reusing the same `base_model` reference — one via `object.__new__(UCA)` shell (fast path), one via the live UCA instance.

### ulcagent — Interactive CLI Launcher (ulcagent.py)

The daily-driver entry point for using ultralight-coder as a local Claude Code replacement. Provides an interactive REPL with session memory, auto-profile switching, and slash commands.

**Installation:** `ulcagent` is available as a PowerShell function (added to `$PROFILE` during setup).

**Usage:**
```bash
cd D:\LLCWork\my-project
ulcagent                    # interactive REPL
ulcagent "fix the bug"      # one-shot
ulcagent --warm             # keep model loaded between goals (~10GB VRAM, instant response)
ulcagent --extended         # enable 21 advanced tools (rename, git, checkpoint, etc.)
```

**Profiles (auto-detected from goal keywords):**
- `code` — Qwen 2.5 Coder 14B (precise code edits, tests, refactoring)
- `general` — Qwen 2.5 14B Instruct (exploration, system tasks, Q&A)

Override with `/code <goal>` or `/general <goal>`.

**Slash commands:**
| Command | Action |
|---------|--------|
| `?` | Show help with tips and examples |
| `/code <goal>` | Force Coder model for this goal |
| `/general <goal>` | Force General model for this goal |
| `/context <files>` | Load files into working memory for cross-file reasoning |
| `/context clear` | Clear loaded context |
| `/context list` | Show currently loaded context files |
| `/diff` | Show uncommitted git changes |
| `/commit` | Stage + commit with auto-generated message |
| `/undo` | Revert all changes from last goal |
| `/clear` | Reset session memory (project notes persist) |
| `/models` | List available GGUF models |
| `/model code <name>` | Set the code-profile model |
| `/model general <name>` | Set the general-profile model |
| `/default <name>` | Set the default model for both profiles |
| `/modelpath` | Show current model search directories |
| `/modelpath add <dir>` | Add a directory to the model search path |
| `/modelpath remove <dir>` | Remove a directory from the model search path |
| `/review` | Review recent changes (code review mode) |
| `/export` | Export session transcript |
| `/paste` | Paste clipboard contents as context |
| `/copy` | Copy last response to clipboard |
| `/snippet list` | List saved snippets |
| `/snippet save <name>` | Save last response as a named snippet |
| `/snippet <name>` | Insert a saved snippet as context |
| `/snippet delete <name>` | Delete a saved snippet |
| `/stats` | Show session statistics (tokens, turns, timing) |
| `/test` | Run project tests (auto-detects framework) |
| `/lint` | Run project linter |
| `/format` | Run project formatter |
| `cd <path>` | Switch workspace |
| `exit` | Quit ulcagent |

**Session memory:** Goals build on each other within a session. "Read server.py" then "fix the endpoint you just read" works. The context meter `[ctx: 45%]` shows usage; `/clear` resets when full.

**Auto-unload:** Model unloads between goals by default (GPU freed at `>>>` prompt). Use `--warm` to keep loaded for instant responses.

**Features:** Color output, thinking spinner, input history (up-arrow), startup greeting with project context (git status + cross-session notes), post-task suggestions ("Run the tests?"), project file index auto-injected on first goal.

**Project rules (`.ulcagent` file):** Drop a `.ulcagent` file in any project root to give ulcagent project-specific instructions and custom slash commands. The file is plain text with two sections:

1. **Instructions** — Free-form text at the top. Injected into the system prompt so the agent follows project conventions (e.g., "Always use tabs", "Run `npm test` after edits", "This is a Django project using pytest").

2. **`[aliases]` section** — Custom slash commands. Each line is `/<name> = <expansion>`. The expansion is sent as a goal, so it can be any natural-language instruction or shell command.

Example `.ulcagent`:
```
This is a FastAPI project. Use pydantic v2 model_validator, not root_validator.
Tests live in tests/ and use pytest. Always run tests after code changes.

[aliases]
/serve = Run uvicorn main:app --reload
/migrate = Run alembic upgrade head
/check = Run pytest and then run ruff check .
```

When ulcagent starts in a directory containing `.ulcagent`, it reads the file automatically. Instructions appear in the agent's system prompt; aliases appear alongside the built-in slash commands.

### Known Ceilings and Workarounds

**Large file creation (100+ lines):** The 14B cannot write big HTML/config/multi-function files in a single tool call — the JSON escaping of multi-line content breaks at ~60-80 lines. **Workaround:** Scaffold the file yourself (or copy a template), then use ulcagent for targeted edits.

**Scaffolding workflow (recommended for new projects):**
1. Create the file structure yourself (empty files, templates, boilerplate)
2. Use ulcagent to fill in logic, fix bugs, add features, write tests
3. Use `/diff` and `/commit` to review and save changes

The model excels at **targeted edits** to existing files — reading 20-line functions, modifying 5 lines, verifying syntax. It struggles at **whole-file generation** beyond ~60 lines.

**File size limit:** Files over ~200 lines should be read in sections (`read_file` with `offset`/`limit`), or use `read_function` (extended tools) to extract specific functions by name.

**Tool count:** The lean 10-tool registry scores 100% on benchmarks. The extended 21-tool set drops to ~86% because the extra schemas consume context and confuse the 14B. Use `--extended` only when you need git/rename/checkpoint tools and are guiding the agent step by step.

**Non-Qwen models:** Only Qwen 2.5 family models support the Hermes `<tool_call>` format natively. Gemma 4 was tested and scored 30% (rejected). Future models need Hermes-format fine-tuning or a per-model adapter.

### Benchmark Results — Phase 14 Agent Mode

**Definitive: 28/28 (100%) at --repeat 2 with lean 10-tool registry.**

13 tasks spanning Python + JS + TS + YAML + JSON + bash + real HTML fixture:

| Task | Diff | Pass rate | Notes |
|------|------|-----------|-------|
| add_docstring | 1 | 100% | |
| add_json_field | 1 | 100% | JSON guard on empty old_string |
| extend_calculator | 2 | 100% | |
| fix_js_reducer | 2 | 100% | JS off-by-one |
| fix_yaml_indent | 2 | 100% | YAML tab auto-fix |
| write_bash_lister | 2 | 100% | Bash script creation |
| add_cli_flag | 3 | 100% | |
| add_ts_interface | 3 | 100% | TypeScript |
| fix_paginate | 3 | 100% | |
| rename_function | 3 | 100% | Multi-file rename |
| extend_real_gallery | 4 | 100% | Real 192-line HTML + auto-retry |
| fix_import_cycle | 4 | 100% | Circular import + strategy hint |
| refactor_dataclass | 4 | 100% | |
| build_todo_cli | 5 | 100% | 4-file project + parser recovery |

Key harness features that enable 100%: truncated-array JSON recovery, per-task strategy hints, pre_finish_check auto-retry, YAML tab auto-fix, insert_at_line tool, unified diff in tool results, context auto-compaction.

## API Endpoints

| Method | Path | Purpose |
|--------|------|---------|
| POST | `/generate` | Generate response |
| POST | `/generate/stream` | SSE streaming |
| POST | `/session/reset` | Clear conversation |
| POST | `/project/index` | Index a codebase directory |
| GET | `/project/status` | Index status |
| GET | `/models/available` | List downloaded models |
| GET | `/status` | System status |
| GET | `/health` | Health check |

## Configuration (config.yaml)

Key sections:
- `base_model`: path, context_length (4096), gpu_layers (99), temperature (0.2), max_tokens (512)
- `router`: mode (hybrid), max_active_modules (2), classifier confidence_threshold (0.4)
- `augmentors`: enabled (true), mode (auto), examples_dir, failure_routing (true). Auto mode has three tiers: <1.5 GB → rerank1 (1 example), 1.5–3 GB → rerank (2 examples), >=3 GB → rerank + **large-mode** (augmentors gated on to testing/data intents only, off for async/cli/web/general where 14B's canonical patterns win)
- `fusion`: mode (lean), token budgets
- `memory`: short_term max_turns (30), long_term FAISS backend
- `project_context`: enabled, top_k (5), similarity_threshold (0.35)

## Augmentor Examples Format

```yaml
domain: pattern
category: pattern_decorator
examples:
  - query: |
      Write a decorator that counts function calls...
    solution: |
      ```python
      import functools
      def count_calls(func): ...
      ```
    tags: [decorator, functools]
    difficulty: medium
```

12 languages supported: Python, JavaScript/TypeScript, Go, Rust, SQL, C, Java, C#, Ruby, Bash, Kotlin, Swift.

## Key Files by Task

### Adding augmentor examples
1. Create `data/augmentor_examples/{domain}/{category}.yaml`
2. Add 3-5 query/solution pairs with tags
3. Optionally add to `FAILURE_PATTERNS` in `engine/augmentors.py` for keyword routing

### Adding a module
1. Create `modules/{name}/manifest.yaml` with keywords, priority, system_prompt_injection
2. Auto-discovered on restart

### Modifying routing
- Keywords: `engine/router.py`
- Classifier: `engine/classifier.py`
- Config: `config.yaml` → `router.*`

### Modifying prompt assembly
- Fusion modes: `engine/fusion.py`
- Token budgets: `config.yaml` → `fusion.budget_*`

### Project indexing
- Index config: `config.yaml` → `project_context.*`
- Implementation: `engine/project_context.py`

## Benchmark Results

### Default 0.5B–3B stack (phases 1–12)

| Benchmark | Score |
|-----------|-------|
| V1 (fundamentals) | 100/100 |
| V2 (project-oriented) | 100/100 |
| V3 (edge cases + architecture) | 99/100 |
| V4 (deep gaps + niche) | 100/100 |
| **Total real-world Python** | **399/400 (99.8%)** |
| Multi-language (12 langs) | 129/130 (99.2%) |

### 14B models (Phase 13 + scoping followup, 2026-04-14)

**V1+V2 (200 queries):**

| Model | Raw | Auto + scoping (default) |
|---|:---:|:---:|
| Qwen 2.5 Coder 14B | 193/200 | **198/200 (99.0%)** |
| Qwen 2.5 14B (non-coder) | 189/200 | **198/200 (99.0%)** |

**Multi-language (130 queries, 12 languages):**

| Model | Old (no scoping) | Scoping + multi-lang hardening |
|---|:---:|:---:|
| Qwen 2.5 Coder 14B | 112/130 (86.2%) | **128/130 (98.5%)** |
| Qwen 2.5 14B (non-coder) | 110/130 (84.6%) | **127/130 (97.7%)** |

Auto mode activates automatically for models ≥3 GB via `use_auto_augmentors(model_size_mb)`. Three mechanisms layered together:

1. **Large-mode gating** — `_large_mode_should_augment(query)` keeps augmentors on for Python testing/data queries + all non-Python queries, skips for Python non-testing
2. **Language scoping** — `_detect_query_language(query)` + `_filter_examples_for_language` filter retrieval candidates by matching-language categories, preventing cross-language example bleeding (Phase 10's multi-language library introduced this failure mode)
3. **max_tokens auto-bump** — `run_benchmark` and `benchmark_phase13.py` use 1024 tokens for models ≥3 GB (Phase 13 found 512 truncated mid-function on several queries)

See `BENCHMARKS.md` Phase 13 + Phase 13 Followup sections for the full story.

## Environment Variables

```bash
TRANSFORMERS_OFFLINE=1      # Skip HuggingFace checks (offline mode)
HF_HUB_OFFLINE=1            # Skip hub checks
LLAMA_LOG_LEVEL=ERROR       # Suppress llama.cpp verbosity
HF_HOME=/custom/cache       # Custom model cache location
```

## densanon-core Dependency

Shared LLM/AI modules were extracted into [densanon-core](https://github.com/densanon-devs/densanon-core). ultralight-coder imports from `densanon.core.*` for: model_loader, config, embeddings, pipeline, cache, tools, chat_format, example_loader, pattern_graph, tuner, project_context.

## GitHub

- Org: densanon-devs
- Repo: densanon-devs/ultralight-coder
