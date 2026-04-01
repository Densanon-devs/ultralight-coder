#!/usr/bin/env bash
# ─────────────────────────────────────────────────────────────
# Run programmer-pack benchmarks across all retrieval modes
# for all models on disk. Each mode runs sequentially.
#
# Usage:
#   bash run_retrieval_bench.sh          # full run, all models
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

echo ""
echo "=============================================="
echo "  Retrieval Mode Benchmark Suite"
echo "  Output: $OUTDIR/"
echo "  Modes: direct, flat, graph, rerank, plan"
echo "=============================================="
echo ""

# Run each mode across all models
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

    python benchmark_programmer.py \
        --all \
        $QUICK_FLAG \
        $FLAG \
        --output "$OUTDIR/bench_${MODE}.json"

    echo "  ✓ ${MODE^^} complete → $OUTDIR/bench_${MODE}.json"
done

echo ""
echo "=============================================="
echo "  All modes complete!"
echo "  Results in: $OUTDIR/"
echo "=============================================="
echo ""
echo "  Files:"
ls -la "$OUTDIR/"
