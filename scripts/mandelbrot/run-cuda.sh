#!/usr/bin/env bash
#
# run-cuda.sh — Build and run the Mandelbrot CUDA benchmark on the remote GPU machine
#
# Usage (run from the repo root on the CUDA machine):
#   ./scripts/mandelbrot/run-cuda.sh <model>
#   ./scripts/mandelbrot/run-cuda.sh codex
#   ./scripts/mandelbrot/run-cuda.sh opus
#   ./scripts/mandelbrot/run-cuda.sh gemini
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

# --- Resolve source file ---

case "$MODEL" in
    codex)
        SRC_FILE="$REPO_ROOT/problems/mandelbrot/results/codex/code/cuda/mandelbrot_cuda.cu"
        ;;
    opus)
        SRC_FILE="$REPO_ROOT/problems/mandelbrot/results/opus/code/cuda/mandelbrot_cuda.cu"
        ;;
    gemini)
        SRC_FILE="$REPO_ROOT/problems/mandelbrot/results/gemini/code/cuda/mandelbrot_cuda.cu"
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
echo " Mandelbrot CUDA Benchmark Runner"
echo "============================================"
echo " Model:        $MODEL"
echo " Source:       $SRC_FILE"
echo "============================================"

# --- Compile ---

WORK_DIR=$(mktemp -d)
cp "$SRC_FILE" "$WORK_DIR/mandelbrot_cuda.cu"
cd "$WORK_DIR"

echo ""
echo "[1/2] Compiling..."
nvcc -O2 -arch=sm_86 -o mandelbrot mandelbrot_cuda.cu
echo "  -> Compiled successfully"

# --- Run ---

echo ""
echo "============================================"
echo " Running Mandelbrot CUDA"
echo "============================================"
echo ""

time ./mandelbrot

echo ""
echo "============================================"
echo " Benchmark complete!"
echo "============================================"

# --- Cleanup ---
cd /
rm -rf "$WORK_DIR"
