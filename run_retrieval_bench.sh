#!/usr/bin/env bash
# ─────────────────────────────────────────────────────────────
# Run programmer-pack benchmarks across all retrieval modes
# for the 4 code-specialized models. Each mode runs sequentially.
#
# Usage:
#   bash run_retrieval_bench.sh          # full run (16 tests per model)
#   bash run_retrieval_bench.sh --quick  # 4 tests per model (fast check)
# ─────────────────────────────────────────────────────────────

set -e
cd "$(dirname "$0")"

QUICK_FLAG=""
if [[ "$1" == "--quick" ]]; then
    QUICK_FLAG="--quick"
    echo "=== QUICK MODE (4 tests per model) ==="
fi

TIMESTAMP=$(date +%Y%m%d_%H%M%S)
OUTDIR="bench_retrieval_${TIMESTAMP}"
mkdir -p "$OUTDIR"

# Only the 4 code-specialized models we care about
MODELS=(
    "models/qwen2.5-coder-0.5b-instruct-q4_k_m.gguf"
    "models/qwen2.5-coder-1.5b-instruct-q4_k_m.gguf"
    "models/qwen2.5-coder-3b-instruct-q4_k_m.gguf"
    "models/deepseek-coder-1.3b-instruct.Q4_K_M.gguf"
)

echo ""
echo "=============================================="
echo "  Retrieval Mode Benchmark Suite"
echo "  Output: $OUTDIR/"
echo "  Modes: direct, flat, graph, rerank, plan"
echo "  Models: ${#MODELS[@]} code-specialized"
echo "=============================================="
echo ""

# Run each mode, iterating models within
for MODE in direct flat graph rerank plan; do
    echo ""
    echo "══════════════════════════════════════════"
    echo "  MODE: ${MODE^^}"
    echo "══════════════════════════════════════════"

    FLAG=""
    case "$MODE" in
        flat)     FLAG="--yaml" ;;
        graph)    FLAG="--graph" ;;
        rerank)   FLAG="--rerank" ;;
        plan)     FLAG="--plan" ;;
        direct)   FLAG="" ;;
    esac

    for MODEL in "${MODELS[@]}"; do
        MODEL_NAME=$(basename "$MODEL" .gguf)
        echo "  → $MODEL_NAME [$MODE]"
        python benchmark_programmer.py \
            --model "$MODEL" \
            $QUICK_FLAG \
            $FLAG \
            --output "$OUTDIR/bench_${MODE}_${MODEL_NAME}.json"
    done

    echo "  ✓ ${MODE^^} complete"
done

echo ""
echo "=============================================="
echo "  All modes complete!"
echo "  Results in: $OUTDIR/"
echo "=============================================="
echo ""
ls -la "$OUTDIR/"
