#!/usr/bin/env bash
#
# run-cuda.sh — Build and run the Feynman-Kac CUDA benchmark on the remote GPU machine
#
# Usage (run from the repo root on the CUDA machine):
#   ./scripts/feynman-kac/run-cuda.sh <model>
#   ./scripts/feynman-kac/run-cuda.sh codex
#   ./scripts/feynman-kac/run-cuda.sh opus
#   ./scripts/feynman-kac/run-cuda.sh gemini
#
# Prerequisites:
#   - Running on a machine with nvcc + GPU
#

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"

MODEL="${1:-}"

# --- Validation ---

if [[ -z "$MODEL" ]]; then
    echo "Error: model argument required."
    echo "Usage: $0 <model>"
    echo "  model: codex | opus | gemini"
    exit 1
fi

# --- Resolve source file and flags ---

EXTRA_LIBS=""
case "$MODEL" in
    codex)
        SRC_FILE="$REPO_ROOT/problems/feynman-kac/results/codex/code/cuda/feynman-kac-gpu.cu"
        ;;
    opus)
        SRC_FILE="$REPO_ROOT/problems/feynman-kac/results/opus/code/cuda/feynman-kac-cuda.cu"
        EXTRA_LIBS="-lcurand"
        ;;
    gemini)
        SRC_FILE="$REPO_ROOT/problems/feynman-kac/results/gemini/code/cuda/feynman-kac-cuda.cu"
        EXTRA_LIBS="-lcurand"
        ;;
    *)
        echo "Error: unknown model '$MODEL'. Use: codex | opus | gemini"
        exit 1
        ;;
esac

if [[ ! -f "$SRC_FILE" ]]; then
    echo "Error: source file not found: $SRC_FILE"
    exit 1
fi

echo "============================================"
echo " Feynman-Kac CUDA Benchmark Runner"
echo "============================================"
echo " Model:        $MODEL"
echo " Source:       $SRC_FILE"
echo "============================================"

# --- Compile ---

WORK_DIR=$(mktemp -d)
cp "$SRC_FILE" "$WORK_DIR/feynman-kac-cuda.cu"
cd "$WORK_DIR"

echo ""
echo "[1/2] Compiling..."
nvcc -O2 -arch=sm_86 -o feynman-kac feynman-kac-cuda.cu -lm $EXTRA_LIBS
echo "  -> Compiled successfully"

# --- Run ---

echo ""
echo "============================================"
echo " Running Feynman-Kac CUDA"
echo "============================================"
echo ""

time ./feynman-kac

echo ""
echo "============================================"
echo " Benchmark complete!"
echo "============================================"

# --- Cleanup ---
cd /
rm -rf "$WORK_DIR"
