#!/usr/bin/env bash
#
# run-cuda.sh — Build and run the Hotspot CUDA benchmark on the remote GPU machine
#
# Usage (run from the repo root on the CUDA machine):
#   ./scripts/hotspot/run-cuda.sh <model> [grid_size] [iterations]
#   ./scripts/hotspot/run-cuda.sh codex   512  10000
#   ./scripts/hotspot/run-cuda.sh opus    1024 10000
#   ./scripts/hotspot/run-cuda.sh gemini  512  10000
#
# Prerequisites:
#   - Running on a machine with nvcc + GPU
#   - Rodinia data files at ~/rodinia-master/data/hotspot/ (or set RODINIA_DATA)
#

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"

MODEL="${1:-}"
GRID_SIZE="${2:-512}"
ITERATIONS="${3:-10000}"

# --- Validation ---

if [[ -z "$MODEL" ]]; then
    echo "Error: model argument required."
    echo "Usage: $0 <model> [grid_size] [iterations]"
    echo "  model:      codex | opus | gemini"
    echo "  grid_size:  512 or 1024 (default: 512)"
    echo "  iterations: simulation iterations (default: 10000)"
    exit 1
fi

# --- Resolve source file ---

case "$MODEL" in
    codex)
        SRC_FILE="$REPO_ROOT/problems/hotspot/results/codex/code/cuda/hotspot-cuda.cu"
        ;;
    opus)
        SRC_FILE="$REPO_ROOT/problems/hotspot/results/opus/code/cuda/hotspot-cuda.cu"
        ;;
    gemini)
        SRC_FILE="$REPO_ROOT/problems/hotspot/results/gemini/code/cuda/hotspot-cuda.cu"
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

# --- Locate data files ---

DATA_DIR="$REPO_ROOT/problems/hotspot/data"
TEMP_FILE="$DATA_DIR/temp_${GRID_SIZE}"
POWER_FILE="$DATA_DIR/power_${GRID_SIZE}"

if [[ ! -f "$TEMP_FILE" ]]; then
    echo "Error: data file not found: $TEMP_FILE"
    echo "Make sure hotspot data files are in problems/hotspot/data/"
    exit 1
fi
if [[ ! -f "$POWER_FILE" ]]; then
    echo "Error: data file not found: $POWER_FILE"
    exit 1
fi

echo "============================================"
echo " Hotspot CUDA Benchmark Runner"
echo "============================================"
echo " Model:        $MODEL"
echo " Grid:         ${GRID_SIZE}x${GRID_SIZE}"
echo " Iterations:   $ITERATIONS"
echo " Source:       $SRC_FILE"
echo "============================================"

# --- Compile ---

WORK_DIR=$(mktemp -d)
cp "$SRC_FILE" "$WORK_DIR/hotspot-cuda.cu"
cd "$WORK_DIR"

echo ""
echo "[1/2] Compiling..."
nvcc -O2 -arch=sm_86 -o hotspot hotspot-cuda.cu
echo "  -> Compiled successfully"

# --- Run ---

echo ""
echo "============================================"
echo " Running Hotspot CUDA: grid=${GRID_SIZE}x${GRID_SIZE} iter=$ITERATIONS"
echo "============================================"
echo ""

time ./hotspot $GRID_SIZE $GRID_SIZE $ITERATIONS 1 "$TEMP_FILE" "$POWER_FILE" output.txt

echo ""
echo "============================================"
echo " Benchmark complete!"
echo "============================================"

# --- Cleanup ---
cd /
rm -rf "$WORK_DIR"
