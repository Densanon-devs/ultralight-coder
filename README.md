# Ultralite Code Assistant

A local AI coding assistant powered by tiny GGUF models (0.5B-3B parameters). Uses a curated augmentor system to make small models perform like large ones -- 437 examples across 12 languages push a 469MB model to 99.2% accuracy on real-world coding queries.

Everything runs on your machine. No API keys, no cloud, no data leaves your computer.

## Features

- **12 languages** -- Python, JavaScript, TypeScript, Go, Rust, SQL, C, Java, C#, Ruby, Bash, Kotlin, Swift
- **99.2% accuracy** on 130-query multi-language benchmark (469MB model)
- **Web UI** with streaming responses, multi-turn conversation, code paste-and-fix
- **Project context** -- index your codebase so the model references your actual code
- **One-click launcher** -- `python launch.py` handles deps, GPU detection, model selection
- **Augmentor system** -- 437 YAML examples injected via semantic similarity to guide small models
- **Offline** -- works fully offline after first setup (sentence-transformers cached locally)

## Quick Start

**One-line install (Linux/macOS):**
```bash
curl -fsSL https://raw.githubusercontent.com/densanon-devs/ultralite-coder/master/install.sh | bash
```

**One-line install (Windows PowerShell):**
```powershell
irm https://raw.githubusercontent.com/densanon-devs/ultralite-coder/master/install.ps1 | iex
```

**With GPU support** -- add `--gpu` (Linux/macOS) or `-GPU` (Windows):
```bash
curl -fsSL https://raw.githubusercontent.com/densanon-devs/ultralite-coder/master/install.sh | bash -s -- --gpu
```

**Manual install:**
```bash
git clone https://github.com/densanon-devs/ultralite-coder.git
cd ultralite-coder
pip install -r requirements.txt
python download_model.py
python launch.py
```

Or run step-by-step:

```bash
python download_model.py --model coder-0.5b   # 469MB, fast
python download_model.py --model coder-1.5b   # 1.1GB, balanced
python download_model.py --model coder-3b     # 2.0GB, best quality

python server.py                # API + web UI on :8000
python main.py                  # Terminal REPL
python main.py --dry-run        # Test routing without a model
```

## Web UI

Open `http://localhost:8000` after starting the server.

- **New Chat** -- clears conversation history
- **{ }** button -- attach code for review/debug
- **Streaming** -- tokens appear live as the model generates
- **Model selector** -- switch between downloaded models
- **Project bar** -- index a directory for project-aware generation

## Project Context

Index your codebase so the model sees relevant code when answering:

```bash
# Via the web UI: enter a path in the Project bar and click Index
# Via API:
curl -X POST http://localhost:8000/project/index \
  -H "Content-Type: application/json" \
  -d '{"path": "/path/to/your/project"}'
```

The model will then reference your actual code when generating -- matching your patterns, imports, and conventions.

## API

All endpoints are documented at `http://localhost:8000/docs` (Swagger UI).

| Endpoint | Method | Description |
|---|---|---|
| `/generate` | POST | Generate a response (prompt + optional code) |
| `/generate/stream` | POST | Streaming generation via SSE |
| `/session/reset` | POST | Clear conversation history |
| `/project/index` | POST | Index a project directory |
| `/project/status` | GET | Check project index status |
| `/status` | GET | System status |
| `/models/available` | GET | List downloaded models |
| `/health` | GET | Health check |

## Models

| Model | Size | Benchmark | Speed | Notes |
|---|---|---|---|---|
| Qwen2.5-Coder-0.5B Q4 | 469 MB | 99.2% | ~30 tok/s | Default. Best size/quality ratio |
| Qwen2.5-Coder-1.5B Q4 | 1.1 GB | 100% | ~20 tok/s | Balanced |
| Qwen2.5-Coder-3B Q4 | 2.0 GB | 100% | ~12 tok/s | Best quality |
| Llama-3.2-1B Q4 | 770 MB | 100% | ~28 tok/s | Good general-purpose |

Download with `python download_model.py --model <name>`. Run `python download_model.py --help` for all options.

## How It Works

The augmentor system is the key innovation. Instead of relying on the model's raw ability, we:

1. **Embed** the user's query with a sentence-transformer
2. **Retrieve** the most similar solved example from 359 curated YAML examples
3. **Inject** that example into the prompt as a few-shot demonstration
4. **Route** queries to language-matched examples using failure pattern detection + language boosting
5. **Generate** with the small model, which now has a concrete pattern to follow

This lets a 469MB model match the output quality of models 10x its size on structured coding tasks.

## Benchmarks

### Multi-Language (130 queries, 10 per language)

```
Python       10/10  100%   ##########
JavaScript   10/10  100%   ##########
TypeScript   10/10  100%   ##########
C            10/10  100%   ##########
Ruby         10/10  100%   ##########
Java         10/10  100%   ##########
C#           10/10  100%   ##########
Go            9/10   90%   #########.
Rust          9/10   90%   #########.
SQL           9/10   90%   #########.
Bash          9/10   90%   #########.
Kotlin        9/10   90%   #########.
Swift         9/10   90%   #########.
TOTAL       129/130  99.2%
```

### Real-World Python (400 queries, 4 benchmark sets)

```
V1 (fundamentals):        100/100
V2 (project-oriented):    100/100
V3 (edge cases):           99/100
V4 (deep gaps + niche):   100/100
TOTAL:                    399/400  99.8%
```

## Configuration

Edit `config.yaml` to customize. Key settings:

```yaml
base_model:
  path: models/qwen2.5-coder-0.5b-instruct-q4_k_m.gguf
  context_length: 4096    # Token context window
  gpu_layers: 99          # GPU offload (99 = all layers)
  temperature: 0.2        # Lower = more deterministic
  max_tokens: 512         # Max generation length

fusion:
  mode: lean              # lean (fast) or structured (more context)
  max_prompt_tokens: 2048 # Total prompt budget

augmentors:
  mode: auto              # auto selects rerank1 (<1.5GB) or rerank (>=1.5GB)
```

## Project Structure

```
ultralite-coder/
  launch.py              # One-click launcher
  server.py              # FastAPI server + web UI
  main.py                # CLI entry point + engine orchestrator
  config.yaml            # Configuration
  download_model.py      # Model downloader
  engine/                # Core engine components
    augmentors.py        # Augmentor system (example retrieval + injection)
    base_model.py        # GGUF model loading via llama-cpp-python
    config.py            # Configuration dataclasses
    embedder.py          # Shared sentence-transformer pool
    fusion.py            # Prompt assembly layer
    memory.py            # Short-term + FAISS long-term memory
    project_context.py   # Codebase indexing and retrieval
    router.py            # Query routing to modules
    ...
  data/
    augmentor_examples/  # 359 YAML examples across 12 languages
  modules/               # Code gen, review, debug, explain modules
  models/                # GGUF model files (downloaded separately)
  static/                # Web UI
  benchmark_*.py         # Benchmark scripts
```

## Requirements

- Python 3.10+
- 2GB+ RAM (4GB recommended)
- GPU optional (CUDA supported via llama-cpp-python)

## License

MIT
