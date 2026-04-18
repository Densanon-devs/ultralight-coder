#!/usr/bin/env python3
"""
Download the smallest viable coding model.

Primary:  Qwen2.5-Coder-1.5B-Instruct (Q4_K_M) — 1.0GB, 43.9% HumanEval
Fallback: Qwen2.5-Coder-0.5B-Instruct (Q4_K_M) — ~400MB, 28% HumanEval

Usage:
    python download_model.py                # Download primary (1.5B)
    python download_model.py --small        # Download fallback (0.5B)
    python download_model.py --both         # Download both
"""

import argparse
import sys
from pathlib import Path

MODELS = {
    "llama": {
        "repo": "bartowski/Llama-3.2-1B-Instruct-GGUF",
        "filename": "Llama-3.2-1B-Instruct-Q4_K_M.gguf",
        "description": "Llama-3.2-1B Q4_K_M (770MB) — RECOMMENDED: 100% benchmark, 28 tok/s",
    },
    "coder-1.5b": {
        "repo": "Qwen/Qwen2.5-Coder-1.5B-Instruct-GGUF",
        "filename": "qwen2.5-coder-1.5b-instruct-q4_k_m.gguf",
        "description": "Qwen2.5-Coder-1.5B Q4_K_M (~1.0GB) — Code specialist, 100% benchmark",
    },
    "coder-0.5b": {
        "repo": "Qwen/Qwen2.5-Coder-0.5B-Instruct-GGUF",
        "filename": "qwen2.5-coder-0.5b-instruct-q4_k_m.gguf",
        "description": "Qwen2.5-Coder-0.5B Q4_K_M (~400MB) — Floor model, 93% w/ augmentors",
    },
    "coder-3b": {
        "repo": "Qwen/Qwen2.5-Coder-3B-Instruct-GGUF",
        "filename": "qwen2.5-coder-3b-instruct-q4_k_m.gguf",
        "description": "Qwen2.5-Coder-3B Q4_K_M (~1.9GB) — Premium model, 100% benchmark",
    },
    "deepseek-1.3b": {
        "repo": "TheBloke/deepseek-coder-1.3b-instruct-GGUF",
        "filename": "deepseek-coder-1.3b-instruct.Q4_K_M.gguf",
        "description": "DeepSeek-Coder-1.3B Q4_K_M (~834MB) — Efficient, 96.9% execution",
    },
    "phi-3.5-mini": {
        "repo": "bartowski/Phi-3.5-mini-instruct-GGUF",
        "filename": "Phi-3.5-mini-instruct-Q4_K_M.gguf",
        "description": "Phi-3.5-mini 3.8B Q4_K_M (~2.3GB) — Microsoft, strong code perf",
    },
    "stable-code-3b": {
        "repo": "bartowski/stable-code-instruct-3b-GGUF",
        "filename": "stable-code-instruct-3b-Q4_K_M.gguf",
        "description": "Stable Code Instruct 3B Q4_K_M (~1.8GB) — StabilityAI, chatml format",
    },
    "yi-coder-1.5b": {
        "repo": "bartowski/Yi-Coder-1.5B-Chat-GGUF",
        "filename": "Yi-Coder-1.5B-Chat-Q4_K_M.gguf",
        "description": "Yi-Coder-1.5B-Chat Q4_K_M (~1.0GB) — 01-ai, chatml, Apache 2.0",
    },
    "coder-14b": {
        "repo": "Qwen/Qwen2.5-Coder-14B-Instruct-GGUF",
        "filename": "qwen2.5-coder-14b-instruct-q4_k_m.gguf",
        "description": "Qwen2.5-Coder-14B Q4_K_M (~9.0GB) — Phase 13 target, code specialist",
    },
    "qwen-14b": {
        "repo": "Qwen/Qwen2.5-14B-Instruct-GGUF",
        "filenames": [
            "qwen2.5-14b-instruct-q4_k_m-00001-of-00003.gguf",
            "qwen2.5-14b-instruct-q4_k_m-00002-of-00003.gguf",
            "qwen2.5-14b-instruct-q4_k_m-00003-of-00003.gguf",
        ],
        "filename": "qwen2.5-14b-instruct-q4_k_m-00001-of-00003.gguf",
        "description": "Qwen2.5-14B Q4_K_M split (3 files, ~9.0GB total) — Phase 13 target, general instruct. llama.cpp auto-loads siblings from the first file.",
    },
}

MODELS_DIR = Path(__file__).parent / "models"


def download_model(key: str):
    try:
        from huggingface_hub import hf_hub_download
    except ImportError:
        print("huggingface-hub not installed. Install with: pip install huggingface-hub")
        sys.exit(1)

    info = MODELS[key]
    # Support both single-file and split-file entries. `filenames` (plural)
    # is the authoritative list when present; `filename` is the first-part
    # path that llama.cpp should be pointed at to auto-discover siblings.
    filenames = info.get("filenames") or [info["filename"]]

    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    print(f"Downloading: {info['description']}")
    print(f"  From: {info['repo']}")
    print(f"  Files: {len(filenames)}")
    print()

    for fn in filenames:
        dest = MODELS_DIR / fn
        if dest.exists():
            size_mb = dest.stat().st_size / (1024 * 1024)
            print(f"  [exists] {fn} ({size_mb:.0f}MB)")
            continue
        print(f"  [fetch]  {fn}")
        path = hf_hub_download(
            repo_id=info["repo"],
            filename=fn,
            local_dir=str(MODELS_DIR),
        )
        print(f"           -> {path}")
    print(f"\nPrimary file: {MODELS_DIR / info['filename']}")


def main():
    parser = argparse.ArgumentParser(description="Download coding models")
    parser.add_argument("--model", choices=list(MODELS.keys()), default="llama",
                        help="Model to download (default: llama)")
    parser.add_argument("--all", action="store_true", help="Download all models")
    args = parser.parse_args()

    if args.all:
        for key in MODELS:
            download_model(key)
            print()
    else:
        download_model(args.model)

    print("\nDone. Update config.yaml 'base_model.path' if needed.")


if __name__ == "__main__":
    main()
