#!/usr/bin/env python3
"""
Ultralite Code Assistant — One-Click Launcher

Run this file. It handles everything:
  1. Checks Python version
  2. Installs missing dependencies
  3. Detects GPU availability
  4. Finds models on disk
  5. Starts the web UI

Usage:
    python launch.py              # Start with web UI (default)
    python launch.py --cli        # Start in terminal mode
    python launch.py --port 9000  # Custom port
"""

import importlib
import logging
import os
import platform
import shutil
import subprocess
import sys
from pathlib import Path

# Suppress noisy third-party logs before anything imports them
for _name in ("httpx", "httpcore", "sentence_transformers", "transformers",
              "huggingface_hub", "filelock"):
    logging.getLogger(_name).setLevel(logging.WARNING)

# Suppress llama.cpp verbose output
os.environ.setdefault("LLAMA_LOG_LEVEL", "ERROR")

# Offline mode for sentence-transformers — skip HuggingFace HEAD requests
# on every startup. Only set if the model cache likely exists already.
_hf_cache = Path(os.environ.get("HF_HOME", Path.home() / ".cache" / "huggingface"))
if _hf_cache.exists():
    os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")
    os.environ.setdefault("HF_HUB_OFFLINE", "1")

PROJECT_ROOT = Path(__file__).parent
MODELS_DIR = PROJECT_ROOT / "models"
CONFIG_PATH = PROJECT_ROOT / "config.yaml"

# ── Colors (works on Windows 10+ and all Unix) ──

def _supports_color():
    if os.environ.get("NO_COLOR"):
        return False
    if platform.system() == "Windows":
        os.system("")  # enable ANSI on Windows
    return hasattr(sys.stdout, "isatty") and sys.stdout.isatty()

USE_COLOR = _supports_color()

def dim(s):    return f"\033[90m{s}\033[0m" if USE_COLOR else s
def green(s):  return f"\033[92m{s}\033[0m" if USE_COLOR else s
def yellow(s): return f"\033[93m{s}\033[0m" if USE_COLOR else s
def red(s):    return f"\033[91m{s}\033[0m" if USE_COLOR else s
def bold(s):   return f"\033[1m{s}\033[0m" if USE_COLOR else s


def banner():
    print()
    print(bold("  Ultralite Code Assistant"))
    print(dim("  Local AI coding assistant powered by tiny models"))
    print()


def check_python():
    v = sys.version_info
    if v.major < 3 or (v.major == 3 and v.minor < 10):
        print(red(f"  Python 3.10+ required (you have {v.major}.{v.minor})"))
        sys.exit(1)
    if v.major == 3 and v.minor >= 13:
        print(f"  Python {v.major}.{v.minor}.{v.micro} {yellow('(3.13+ — pre-built wheels may not be available)')}")
    else:
        print(f"  Python {v.major}.{v.minor}.{v.micro} {green('OK')}")


def check_dependency(name, pip_name=None, required=True):
    """Check if a package is importable. Returns True if available."""
    pip_name = pip_name or name
    try:
        importlib.import_module(name)
        return True
    except ImportError:
        if required:
            return False
        return False


def detect_gpu():
    """Detect CUDA GPU availability."""
    try:
        result = subprocess.run(
            ["nvidia-smi", "--query-gpu=name,memory.total", "--format=csv,noheader"],
            capture_output=True, text=True, timeout=5,
        )
        if result.returncode == 0 and result.stdout.strip():
            gpu = result.stdout.strip().split("\n")[0]
            print(f"  GPU: {green(gpu)}")
            return True
    except (FileNotFoundError, subprocess.TimeoutExpired):
        pass
    print(f"  GPU: {dim('not detected (CPU mode)')}")
    return False


def install_dependencies(has_gpu):
    """Install missing packages."""
    deps = {
        "llama_cpp": "llama-cpp-python",
        "sentence_transformers": "sentence-transformers",
        "yaml": "pyyaml",
        "numpy": "numpy",
        "faiss": "faiss-cpu",
        "fastapi": "fastapi",
        "uvicorn": "uvicorn",
    }

    missing = []
    for module, pip_name in deps.items():
        if not check_dependency(module, required=False):
            missing.append(pip_name)

    if not missing:
        print(f"  Dependencies: {green('all installed')}")
        return

    print(f"  Missing packages: {', '.join(missing)}")
    print()
    resp = input(f"  Install them now? [Y/n] ").strip().lower()
    if resp and resp != "y":
        print(red("  Cannot run without dependencies. Exiting."))
        sys.exit(1)

    print()
    for pkg in missing:
        # Use GPU-accelerated llama-cpp if GPU detected
        if pkg == "llama-cpp-python" and has_gpu:
            print(f"  Installing {pkg} (GPU)...")
            subprocess.run([
                sys.executable, "-m", "pip", "install", pkg,
                "--extra-index-url", "https://abetlen.github.io/llama-cpp-python/whl/cu121",
            ], check=False)
        else:
            print(f"  Installing {pkg}...")
            subprocess.run([sys.executable, "-m", "pip", "install", pkg], check=False)

    # Verify
    still_missing = [m for m, p in deps.items() if not check_dependency(m, required=False)]
    if still_missing:
        print(red(f"\n  Failed to install: {still_missing}"))
        print(red("  Try installing manually with pip."))
        sys.exit(1)

    print(f"\n  Dependencies: {green('installed')}")


def find_models():
    """Find .gguf model files."""
    if not MODELS_DIR.exists():
        return []
    models = sorted(MODELS_DIR.glob("*.gguf"), key=lambda p: p.stat().st_size)
    return models


def select_model(models):
    """Let user pick a model if multiple are available."""
    if not models:
        print(red("\n  No models found!"))
        print(f"  Place .gguf model files in: {MODELS_DIR}")
        print(dim("  Recommended: qwen2.5-coder-1.5b-instruct-q4_k_m.gguf"))
        sys.exit(1)

    # Check which model config.yaml currently points to
    current = None
    if CONFIG_PATH.exists():
        try:
            import yaml
            with open(CONFIG_PATH) as f:
                cfg = yaml.safe_load(f)
            current = cfg.get("base_model", {}).get("path", "")
        except Exception:
            pass

    if len(models) == 1:
        m = models[0]
        size_mb = m.stat().st_size / (1024 * 1024)
        print(f"  Model: {green(m.stem)} ({size_mb:.0f} MB)")
        return m

    print(f"\n  Available models:")
    for i, m in enumerate(models):
        size_mb = m.stat().st_size / (1024 * 1024)
        is_current = str(m).replace("\\", "/").endswith(current.replace("\\", "/")) if current else False
        marker = " (current)" if is_current else ""
        # Tier label
        if size_mb < 600:
            tier = dim("tiny")
        elif size_mb < 1200:
            tier = "balanced"
        elif size_mb < 2500:
            tier = bold("quality")
        else:
            tier = yellow("large")
        print(f"    {i + 1}. {m.stem}  {dim(f'({size_mb:.0f} MB, {tier})')}{green(marker)}")

    print()
    choice = input(f"  Select model [1-{len(models)}] (Enter for current): ").strip()
    if not choice:
        # Find current model in list, or default to first
        for m in models:
            if current and str(m).replace("\\", "/").endswith(current.replace("\\", "/")):
                return m
        return models[0]

    try:
        idx = int(choice) - 1
        if 0 <= idx < len(models):
            return models[idx]
    except ValueError:
        pass

    print(dim("  Invalid choice, using first model"))
    return models[0]


def update_config_model(model_path):
    """Update config.yaml to point to the selected model."""
    if not CONFIG_PATH.exists():
        return

    try:
        import yaml
    except ImportError:
        return

    with open(CONFIG_PATH) as f:
        cfg = yaml.safe_load(f)

    rel_path = str(model_path.relative_to(PROJECT_ROOT)).replace("\\", "/")
    current = cfg.get("base_model", {}).get("path", "")

    if current != rel_path:
        cfg["base_model"]["path"] = rel_path
        with open(CONFIG_PATH, "w") as f:
            yaml.dump(cfg, f, default_flow_style=False, sort_keys=False)
        print(f"  Config updated: {dim(rel_path)}")


def start_desktop(port):
    """Start the desktop app (native window). Falls back to browser if pywebview not installed."""
    os.chdir(str(PROJECT_ROOT))
    sys.path.insert(0, str(PROJECT_ROOT))

    try:
        importlib.import_module("webview")
        has_webview = True
    except ImportError:
        has_webview = False

    if has_webview:
        print()
        print(bold("  Starting desktop app..."))
        print(dim("  Close the window to stop"))
        print()
        from desktop import start_server, wait_for_server
        import webview
        import threading

        server_thread = threading.Thread(target=start_server, args=(port,), daemon=True)
        server_thread.start()
        if not wait_for_server(port):
            print(red("  Server failed to start."))
            return
        window = webview.create_window(
            "Ultralite Code Assistant",
            f"http://127.0.0.1:{port}",
            width=1100, height=750, min_size=(800, 500),
        )
        webview.start()
    else:
        print(dim("  pywebview not installed — opening in browser"))
        print(dim("  Install for desktop mode: pip install pywebview"))
        start_web(port, "127.0.0.1")


def start_web(port, host):
    """Start the web UI server (browser mode)."""
    print()
    print(bold("  Starting web UI..."))
    print(f"  Open in browser: {green(f'http://{host}:{port}')}")
    print(dim("  Press Ctrl+C to stop"))
    print()

    os.chdir(str(PROJECT_ROOT))
    sys.path.insert(0, str(PROJECT_ROOT))

    import uvicorn
    uvicorn.run(
        "server:create_app",
        host=host,
        port=port,
        factory=True,
        log_level="warning",
    )


def start_cli():
    """Start interactive CLI mode."""
    print()
    print(bold("  Starting interactive mode..."))
    print(dim("  Type 'quit' to exit"))
    print()

    os.chdir(str(PROJECT_ROOT))
    sys.path.insert(0, str(PROJECT_ROOT))

    from main import UltraliteCodeAssistant
    engine = UltraliteCodeAssistant()
    engine.initialize()
    engine.interactive()


def main():
    import argparse
    parser = argparse.ArgumentParser(description="Ultralite Code Assistant Launcher")
    parser.add_argument("--cli", action="store_true", help="Start in terminal mode")
    parser.add_argument("--browser", action="store_true", help="Force browser mode (skip desktop window)")
    parser.add_argument("--port", type=int, default=8000, help="Web UI port (default: 8000)")
    parser.add_argument("--host", default="127.0.0.1", help="Host to bind to")
    args = parser.parse_args()

    banner()

    # Step 1: Check Python
    check_python()

    # Step 2: Detect GPU
    has_gpu = detect_gpu()

    # Step 3: Install dependencies
    install_dependencies(has_gpu)

    # Step 4: Find and select model
    models = find_models()
    selected = select_model(models)
    update_config_model(selected)

    size_mb = selected.stat().st_size / (1024 * 1024)
    mode = "rerank (2 examples)" if size_mb >= 1500 else "rerank1 (1 example)"
    print(f"  Augmentor mode: {green(mode)}")
    print(f"  Examples: {green('437 across 12 languages')}")

    # Step 5: Start
    if args.cli:
        start_cli()
    elif args.browser:
        start_web(args.port, args.host)
    else:
        # Default: try desktop window, fall back to browser
        start_desktop(args.port)


if __name__ == "__main__":
    main()
