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
        "description": "Qwen2.5-Coder-0.5B Q4_K_M (~400MB) — Floor model, 93% w/ experts",
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
    dest = MODELS_DIR / info["filename"]

    if dest.exists():
        size_mb = dest.stat().st_size / (1024 * 1024)
        print(f"Already exists: {dest} ({size_mb:.0f}MB)")
        return

    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    print(f"Downloading: {info['description']}")
    print(f"  From: {info['repo']}")
    print(f"  File: {info['filename']}")
    print()

    path = hf_hub_download(
        repo_id=info["repo"],
        filename=info["filename"],
        local_dir=str(MODELS_DIR),
        local_dir_use_symlinks=False,
    )
    print(f"\nSaved to: {path}")


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
