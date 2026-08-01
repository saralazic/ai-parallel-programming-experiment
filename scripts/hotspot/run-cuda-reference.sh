#!/usr/bin/env bash
#
# run-cuda-reference.sh — Build and run Rodinia Hotspot CUDA reference on remote GPU machine
#
# Usage (run from the repo root on the CUDA machine):
#   ./scripts/hotspot/run-cuda-reference.sh [grid_size] [iterations]
#   ./scripts/hotspot/run-cuda-reference.sh 1024 10000
#
# Prerequisites:
#   - Running on a machine with nvcc + GPU
#   - Data files in problems/hotspot/data/
#

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"

GRID_SIZE="${1:-1024}"
ITERATIONS="${2:-10000}"
PYRAMID_HEIGHT=1

SRC_FILE="$REPO_ROOT/problems/hotspot/reference/cuda/hotspot.cu"
DATA_DIR="$REPO_ROOT/problems/hotspot/data"

if [[ ! -f "$SRC_FILE" ]]; then
    echo "Error: reference CUDA source not found: $SRC_FILE"
    exit 1
fi

TEMP_FILE="$DATA_DIR/temp_${GRID_SIZE}"
POWER_FILE="$DATA_DIR/power_${GRID_SIZE}"

if [[ ! -f "$TEMP_FILE" ]]; then
    echo "Error: data file not found: $TEMP_FILE"
    exit 1
fi

echo "============================================"
echo " Hotspot CUDA Reference (Rodinia)"
echo "============================================"
echo " Grid:         ${GRID_SIZE}x${GRID_SIZE}"
echo " Iterations:   $ITERATIONS"
echo " Pyramid H:    $PYRAMID_HEIGHT"
echo "============================================"

if ! command -v nvcc &> /dev/null; then
    echo "Error: nvcc not found. Please ensure CUDA toolkit is installed."
    exit 1
fi
echo " NVCC:         $(nvcc --version | grep release | awk '{print $5}' | tr -d ',')"
echo "============================================"

WORK_DIR=$(mktemp -d)
cp "$SRC_FILE" "$WORK_DIR/hotspot.cu"
cp "$TEMP_FILE" "$WORK_DIR/"
cp "$POWER_FILE" "$WORK_DIR/"
cd "$WORK_DIR"

echo ""
echo "[1/2] Compiling..."
nvcc -O2 -arch=sm_86 -o hotspot hotspot.cu
echo "  -> Compiled successfully"

echo ""
echo "============================================"
echo " Running Hotspot CUDA reference: grid=${GRID_SIZE}x${GRID_SIZE} iter=$ITERATIONS"
echo "============================================"
echo ""

time ./hotspot $GRID_SIZE $PYRAMID_HEIGHT $ITERATIONS temp_${GRID_SIZE} power_${GRID_SIZE}

echo ""
echo "============================================"
echo " Benchmark complete!"
echo "============================================"

cd /
rm -rf "$WORK_DIR"
