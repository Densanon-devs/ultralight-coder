#!/bin/bash
# Ultralight Code Assistant — One-Line Installer
#
# Usage:
#   curl -fsSL https://raw.githubusercontent.com/densanon-devs/ultralight-coder/master/install.sh | bash
#
# Or manually:
#   bash install.sh
#   bash install.sh --no-model    # Skip model download
#   bash install.sh --gpu         # Install GPU-accelerated llama-cpp-python

set -e

GREEN='\033[92m'
DIM='\033[90m'
BOLD='\033[1m'
RESET='\033[0m'

echo ""
echo -e "${BOLD}  Ultralight Code Assistant — Installer${RESET}"
echo -e "${DIM}  Local AI coding assistant, 12 languages, 469MB model${RESET}"
echo ""

# Parse args
SKIP_MODEL=false
GPU=false
for arg in "$@"; do
    case "$arg" in
        --no-model) SKIP_MODEL=true ;;
        --gpu) GPU=true ;;
    esac
done

# Check Python
if ! command -v python3 &>/dev/null && ! command -v python &>/dev/null; then
    echo "  Error: Python 3.10+ is required but not found."
    echo "  Install from https://python.org"
    exit 1
fi

PY=$(command -v python3 || command -v python)
PY_VERSION=$($PY -c "import sys; print(f'{sys.version_info.major}.{sys.version_info.minor}')")
PY_MAJOR=$($PY -c "import sys; print(sys.version_info.major)")
PY_MINOR=$($PY -c "import sys; print(sys.version_info.minor)")

if [ "$PY_MAJOR" -lt 3 ] || ([ "$PY_MAJOR" -eq 3 ] && [ "$PY_MINOR" -lt 10 ]); then
    echo "  Error: Python 3.10+ required (you have $PY_VERSION)"
    exit 1
fi
echo -e "  Python $PY_VERSION ${GREEN}OK${RESET}"

# Clone
INSTALL_DIR="ultralight-coder"
if [ -d "$INSTALL_DIR" ]; then
    echo "  Directory $INSTALL_DIR already exists, pulling latest..."
    cd "$INSTALL_DIR"
    git pull --ff-only
else
    echo "  Cloning repository..."
    git clone https://github.com/densanon-devs/ultralight-coder.git
    cd "$INSTALL_DIR"
fi

# Install deps (use pre-built wheels to avoid 20min C++ compile)
echo ""
echo "  Installing dependencies..."
if [ "$GPU" = true ]; then
    echo -e "  ${DIM}(GPU mode — using pre-built CUDA wheels)${RESET}"
    $PY -m pip install llama-cpp-python --extra-index-url https://abetlen.github.io/llama-cpp-python/whl/cu121 --quiet --force-reinstall
else
    echo -e "  ${DIM}(Using pre-built CPU wheels)${RESET}"
    $PY -m pip install llama-cpp-python --extra-index-url https://abetlen.github.io/llama-cpp-python/whl/cpu --quiet
fi
$PY -m pip install -r requirements.txt --quiet
echo -e "  Dependencies ${GREEN}installed${RESET}"

# Download model
if [ "$SKIP_MODEL" = false ]; then
    echo ""
    echo "  Downloading default model (Qwen 0.5B, 469MB)..."
    $PY download_model.py --model coder-0.5b
    echo -e "  Model ${GREEN}ready${RESET}"
else
    echo ""
    echo -e "  ${DIM}Skipping model download (--no-model)${RESET}"
    echo "  Run later: python download_model.py"
fi

# Done
echo ""
echo -e "${BOLD}  Setup complete!${RESET}"
echo ""
echo "  Start the web UI:"
echo -e "    cd $INSTALL_DIR"
echo -e "    $PY launch.py"
echo ""
echo -e "  Then open ${GREEN}http://localhost:8000${RESET} in your browser."
echo ""
