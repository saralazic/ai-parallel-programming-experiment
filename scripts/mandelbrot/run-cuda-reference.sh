#!/usr/bin/env bash
#
# run-cuda-reference.sh — Build and run Mandelbrot CUDA reference on remote GPU machine
#
# Usage (run from the repo root on the CUDA machine):
#   ./scripts/mandelbrot/run-cuda-reference.sh
#
# Prerequisites:
#   - Running on a machine with nvcc + GPU
#

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"

SRC_FILE="$REPO_ROOT/problems/mandelbrot/reference/cuda/mandelbrot.cu"

if [[ ! -f "$SRC_FILE" ]]; then
    echo "Error: reference CUDA source not found: $SRC_FILE"
    exit 1
fi

echo "============================================"
echo " Mandelbrot CUDA Reference Runner"
echo "============================================"
echo " Source:       $SRC_FILE"
echo "============================================"

if ! command -v nvcc &> /dev/null; then
    echo "Error: nvcc not found. Please ensure CUDA toolkit is installed."
    exit 1
fi
echo " NVCC:         $(nvcc --version | grep release | awk '{print $5}' | tr -d ',')"
echo "============================================"

WORK_DIR=$(mktemp -d)
cp "$SRC_FILE" "$WORK_DIR/mandelbrot.cu"
cd "$WORK_DIR"

echo ""
echo "[1/2] Compiling..."
nvcc -O2 -arch=sm_86 -o mandelbrot mandelbrot.cu -lm
echo "  -> Compiled successfully"

echo ""
echo "============================================"
echo " Running Mandelbrot CUDA reference"
echo "============================================"
echo ""

time ./mandelbrot

echo ""
echo "============================================"
echo " Benchmark complete!"
echo "============================================"

cd /
rm -rf "$WORK_DIR"
