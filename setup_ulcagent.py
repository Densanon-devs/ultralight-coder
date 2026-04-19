#!/usr/bin/env python
"""
ulcagent first-run setup.

Checks: Python version, llama-cpp-python, CUDA/GPU, models.
Downloads a model if none found. Sets up PowerShell profile.

Usage: python setup_ulcagent.py
"""
import os
import sys
import shutil
import subprocess
from pathlib import Path

_SELF = Path(__file__).resolve().parent
_MODELS = _SELF / "models"

def _green(t): return f"\033[32m{t}\033[0m"
def _red(t):   return f"\033[31m{t}\033[0m"
def _yellow(t): return f"\033[33m{t}\033[0m"
def _bold(t):  return f"\033[1m{t}\033[0m"
def _dim(t):   return f"\033[2m{t}\033[0m"


def check_python():
    v = sys.version_info
    ok = v >= (3, 8)
    print(f"  Python {v.major}.{v.minor}.{v.micro}: {_green('OK') if ok else _red('FAIL (need 3.8+)')}")
    return ok


def check_llama_cpp():
    try:
        import llama_cpp
        ver = getattr(llama_cpp, "__version__", "unknown")
        print(f"  llama-cpp-python {ver}: {_green('OK')}")
        return True
    except ImportError:
        print(f"  llama-cpp-python: {_red('NOT INSTALLED')}")
        print(f"    {_dim('Install: pip install llama-cpp-python')}")
        print(f"    {_dim('For CUDA: pip install llama-cpp-python --extra-index-url https://abetlen.github.io/llama-cpp-python/whl/cu121')}")
        return False


def check_gpu():
    try:
        import llama_cpp
        # CUDA info shows up in stderr during import
        result = subprocess.run(
            [sys.executable, "-c", "import llama_cpp"],
            capture_output=True, text=True, timeout=10,
        )
        if "CUDA" in result.stderr:
            for line in result.stderr.splitlines():
                if "CUDA devices" in line or "Device 0" in line:
                    print(f"  GPU: {_green(line.strip())}")
            return True
        print(f"  GPU: {_yellow('No CUDA detected (CPU-only mode — slower)')}")
        return True
    except Exception:
        print(f"  GPU: {_dim('Could not detect')}")
        return True


def check_models():
    _MODELS.mkdir(exist_ok=True)
    ggufs = list(_MODELS.glob("*.gguf"))
    if ggufs:
        print(f"  Models: {_green(f'{len(ggufs)} GGUF files found')}")
        for g in ggufs[:5]:
            size_gb = g.stat().st_size / (1024**3)
            print(f"    {g.name} ({size_gb:.1f} GB)")
        if len(ggufs) > 5:
            print(f"    ... and {len(ggufs) - 5} more")
        return True
    print(f"  Models: {_yellow('No GGUF files found in models/')}")
    return False


def download_model():
    print(f"\n  {_bold('Download a model?')}")
    print(f"  Recommended: Qwen 2.5 Coder 14B (8.4 GB, best for coding)")
    print(f"  Alternative:  Qwen 2.5 Coder 3B  (2.0 GB, faster, less capable)")
    print()
    choice = input("  Download? [14b/3b/skip] ").strip().lower()

    if choice in ("14b", "14"):
        model = "Qwen/Qwen2.5-Coder-14B-Instruct-GGUF"
        filename = "qwen2.5-coder-14b-instruct-q4_k_m.gguf"
    elif choice in ("3b", "3"):
        model = "Qwen/Qwen2.5-Coder-3B-Instruct-GGUF"
        filename = "qwen2.5-coder-3b-instruct-q4_k_m.gguf"
    else:
        print(f"  {_dim('Skipped. Add a GGUF file to models/ manually.')}")
        return False

    print(f"\n  Downloading {filename}...")
    try:
        from huggingface_hub import hf_hub_download
        path = hf_hub_download(model, filename=filename, local_dir=str(_MODELS))
        print(f"  {_green('Downloaded:')} {path}")
        return True
    except ImportError:
        print(f"  {_red('huggingface_hub not installed.')}")
        print(f"  {_dim('pip install huggingface-hub')}")
        cmd = f'python -c "from huggingface_hub import hf_hub_download; hf_hub_download(\'{model}\', filename=\'{filename}\', local_dir=\'{_MODELS}\')"'
        print(f"  {_dim('Then: ' + cmd)}")
        return False
    except Exception as e:
        print(f"  {_red(f'Download failed: {e}')}")
        return False


def generate_config():
    """Generate config YAML if missing, pointing at the first available model."""
    config_path = _SELF / "config_agent14b.yaml"
    if config_path.exists():
        return

    ggufs = sorted(_MODELS.glob("*.gguf"))
    if not ggufs:
        return

    model_path = f"models/{ggufs[0].name}"
    config_path.write_text(f"""system:
  name: Ultralite Code Assistant
  version: 0.1.0
  log_level: INFO
base_model:
  path: {model_path}
  context_length: 16384
  gpu_layers: 99
  threads: 8
  temperature: 0.1
  max_tokens: 1024
  batch_size: 512
""", encoding="utf-8")
    print(f"  {_green('Generated:')} {config_path.name} -> {model_path}")


def setup_powershell():
    if os.name != "nt":
        print(f"  PowerShell: {_dim('Not Windows, skipping')}")
        return

    profile_dir = Path.home() / "Documents" / "WindowsPowerShell"
    profile_path = profile_dir / "Microsoft.PowerShell_profile.ps1"

    if profile_path.exists():
        content = profile_path.read_text(encoding="utf-8", errors="replace")
        if "ulcagent" in content:
            print(f"  PowerShell: {_green('ulcagent already in profile')}")
            return

    print(f"\n  {_bold('Add ulcagent to PowerShell?')}")
    print(f"  This lets you type 'ulcagent' from any directory.")
    choice = input("  Add? [y/N] ").strip().lower()
    if choice not in ("y", "yes"):
        print(f"  {_dim('Skipped. Run manually: python ' + str(_SELF / 'ulcagent.py'))}")
        return

    profile_dir.mkdir(parents=True, exist_ok=True)
    ulcagent_path = str(_SELF / "ulcagent.py").replace("/", "\\")
    line = f'\nfunction ulcagent {{ python "{ulcagent_path}" @args }}\n'
    with open(profile_path, "a", encoding="utf-8") as f:
        f.write(line)
    print(f"  {_green('Added to')} {profile_path}")
    print(f"  {_dim('Restart PowerShell to use.')}")


def main():
    print(f"\n{_bold('ulcagent setup')}\n")

    print(_bold("Checking requirements:"))
    py_ok = check_python()
    llama_ok = check_llama_cpp()
    check_gpu()
    models_ok = check_models()

    if not py_ok:
        print(f"\n{_red('Python 3.8+ required.')}")
        return 1

    if not llama_ok:
        print(f"\n{_yellow('Install llama-cpp-python before using ulcagent.')}")

    if not models_ok:
        download_model()
        models_ok = check_models()

    generate_config()
    setup_powershell()

    print(f"\n{_bold('Setup complete.')}")
    if models_ok and llama_ok:
        print(f"  {_green('Ready to use:')} ulcagent")
        print(f"  {_dim('Or: python ' + str(_SELF / 'ulcagent.py'))}")
    else:
        missing = []
        if not llama_ok:
            missing.append("llama-cpp-python")
        if not models_ok:
            missing.append("a GGUF model in models/")
        print(f"  {_yellow('Still needed:')} {', '.join(missing)}")

    return 0


if __name__ == "__main__":
    sys.exit(main())
