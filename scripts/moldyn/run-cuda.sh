#!/usr/bin/env bash
#
# run-cuda.sh — Build and run the Moldyn CUDA benchmark on the remote GPU machine
#
# Usage (run from the repo root on the CUDA machine):
#   ./scripts/moldyn/run-cuda.sh <model> [particles]
#   ./scripts/moldyn/run-cuda.sh codex   20
#   ./scripts/moldyn/run-cuda.sh opus    20
#   ./scripts/moldyn/run-cuda.sh gemini  20
#
# Prerequisites:
#   - Running on a machine with nvcc + GPU
#

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"

MODEL="${1:-}"
PARTICLES="${2:-20}"

# --- Validation ---

if [[ -z "$MODEL" ]]; then
    echo "Error: model argument required."
    echo "Usage: $0 <model> [particles]"
    echo "  model:     codex | opus | gemini"
    echo "  particles: number of particles (default: 20)"
    exit 1
fi

# --- Resolve source file ---

case "$MODEL" in
    codex|opus|gemini)
        SRC_FILE="$REPO_ROOT/problems/moldyn/results/$MODEL/code/cuda/moldyn.cu"
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
echo " Moldyn CUDA Benchmark Runner"
echo "============================================"
echo " Model:        $MODEL"
echo " Particles:    $PARTICLES"
echo " Source:       $SRC_FILE"
echo "============================================"

# --- Compile ---

WORK_DIR=$(mktemp -d)
cp "$SRC_FILE" "$WORK_DIR/moldyn.cu"
cd "$WORK_DIR"

echo ""
echo "[1/2] Compiling..."
nvcc -O2 -arch=sm_86 -o moldyn moldyn.cu -lm
echo "  -> Compiled successfully"

# --- Run ---

echo ""
echo "============================================"
echo " Running Moldyn CUDA: particles=$PARTICLES"
echo "============================================"
echo ""

time ./moldyn $PARTICLES

echo ""
echo "============================================"
echo " Benchmark complete!"
echo "============================================"

# --- Cleanup ---
cd /
rm -rf "$WORK_DIR"
