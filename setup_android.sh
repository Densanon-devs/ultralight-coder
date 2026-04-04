#!/data/data/com.termux/files/usr/bin/bash
# ──────────────────────────────────────────────────────────
# Android Setup Script — Digest Curation Engine on Termux
# Run this after installing Termux from F-Droid
# ──────────────────────────────────────────────────────────
set -e

echo "═══════════════════════════════════════════════"
echo "  Digest Engine — Android Setup"
echo "═══════════════════════════════════════════════"

# ── Step 1: System packages ──────────────────────────────
echo ""
echo "[1/6] Installing system packages..."
pkg update -y
pkg upgrade -y
pkg install -y python git cmake make clang libxml2 libxslt openssl

# ── Step 2: Storage access ───────────────────────────────
echo ""
echo "[2/6] Setting up storage access..."
if [ ! -d ~/storage ]; then
    termux-setup-storage
    echo "  Grant storage permission in the popup, then re-run this script."
    exit 0
fi

# ── Step 3: Python deps ─────────────────────────────────
echo ""
echo "[3/6] Installing Python packages..."
pip install pyyaml feedparser beautifulsoup4 requests numpy

echo ""
echo "[3b/6] Compiling llama-cpp-python (this takes ~10 min)..."
CMAKE_ARGS="" pip install llama-cpp-python --no-cache-dir

# ── Step 4: Clone repo ──────────────────────────────────
echo ""
echo "[4/6] Cloning repository..."
WORK_DIR="$HOME/digest-engine"
if [ -d "$WORK_DIR" ]; then
    echo "  $WORK_DIR exists, pulling latest..."
    cd "$WORK_DIR"
    git pull
else
    git clone https://github.com/Densanon-devs/ultralite-coder.git "$WORK_DIR"
    cd "$WORK_DIR"
fi
git checkout mobile-digest-engine

# ── Step 5: Download model ──────────────────────────────
echo ""
echo "[5/6] Setting up model..."
mkdir -p models

MODEL_PATH="models/qwen2.5-coder-0.5b-instruct-q4_k_m.gguf"
if [ -f "$MODEL_PATH" ]; then
    echo "  Model already exists: $MODEL_PATH"
else
    echo "  Downloading Qwen2.5 0.5B (469 MB)..."
    pip install huggingface-hub 2>/dev/null
    python -c "
from huggingface_hub import hf_hub_download
hf_hub_download(
    repo_id='Qwen/Qwen2.5-Coder-0.5B-Instruct-GGUF',
    filename='qwen2.5-coder-0.5b-instruct-q4_k_m.gguf',
    local_dir='models',
)
print('Download complete.')
"
fi

# ── Step 6: Verify ──────────────────────────────────────
echo ""
echo "[6/6] Verifying installation..."
python -c "
import yaml, feedparser, requests, numpy
from engine.digest_augmentors import DigestAugmentorRouter
router = DigestAugmentorRouter()
total = len(router.selection.examples) + len(router.takeaway.examples) + len(router.highlights.examples)
print(f'  Augmentors loaded: {total} examples')
print(f'  Selection grammar: {\"OK\" if router.selection_grammar else \"MISSING\"}')
print(f'  Takeaways grammar: {\"OK\" if router.takeaways_grammar else \"MISSING\"}')
print(f'  Highlights grammar: {\"OK\" if router.highlights_grammar else \"MISSING\"}')
"

echo ""
echo "═══════════════════════════════════════════════"
echo "  Setup complete!"
echo ""
echo "  Quick commands:"
echo "    cd $WORK_DIR"
echo "    python digest_main.py --fetch ai          # fetch articles"
echo "    python digest_main.py --curate ai         # curate with LLM"
echo "    python digest_main.py ai                  # fetch + curate"
echo "    python benchmark_digest.py --verify-only  # run verifiers"
echo "═══════════════════════════════════════════════"
