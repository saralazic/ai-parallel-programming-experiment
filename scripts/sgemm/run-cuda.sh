#!/usr/bin/env bash
#
# run-cuda.sh — Build and run the SGEMM CUDA benchmark on the remote GPU machine
#
# Usage (run from the repo root on the CUDA machine):
#   ./scripts/sgemm/run-cuda.sh <model>
#   ./scripts/sgemm/run-cuda.sh codex
#   ./scripts/sgemm/run-cuda.sh opus
#   ./scripts/sgemm/run-cuda.sh gemini
#
# Prerequisites:
#   - Running on rticuda.etf.bg.ac.rs (or any machine with nvcc + GPU)
#   - Repo cloned/copied to the machine
#   - Input data files present in medium/input/ within the code directory
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

# --- Resolve source directory ---

case "$MODEL" in
    codex)
        SRC_DIR="$REPO_ROOT/problems/sgemm/results/codex/code/cuda"
        ;;
    opus)
        SRC_DIR="$REPO_ROOT/problems/sgemm/results/opus/code/cuda"
        ;;
    gemini)
        SRC_DIR="$REPO_ROOT/problems/sgemm/results/gemini/code/cuda"
        ;;
    *)
        echo "Error: unknown model '$MODEL'. Use: codex | opus | gemini"
        exit 1
        ;;
esac

if [[ ! -d "$SRC_DIR" ]]; then
    echo "Error: source directory not found: $SRC_DIR"
    exit 1
fi

echo "============================================"
echo " SGEMM CUDA Benchmark Runner"
echo "============================================"
echo " Model:        $MODEL"
echo " Source:       $SRC_DIR"
echo "============================================"

# --- Check for nvcc ---

if ! command -v nvcc &> /dev/null; then
    echo "Error: nvcc not found. Please ensure CUDA toolkit is installed."
    exit 1
fi
echo " NVCC:         $(nvcc --version | grep release | awk '{print $5}' | tr -d ',')"
echo "============================================"

cd "$SRC_DIR"

# --- Clean previous build ---

echo ""
echo "[1/3] Cleaning previous build..."
make clean 2>/dev/null || true

# --- Compile ---

echo ""
echo "[2/3] Compiling..."
if grep -q "^cuda:" Makefile 2>/dev/null; then
    make cuda NVCCFLAGS="-O2 -arch=sm_86"
else
    make NVCCFLAGS="-O2 -arch=sm_86"
fi

# --- Run ---

echo ""
echo "============================================"
echo " Running SGEMM CUDA benchmark"
echo "============================================"
echo ""

if grep -q "^run_cuda:" Makefile 2>/dev/null; then
    time make run_cuda
else
    time make run
fi

# --- Verify output ---

echo ""
echo "============================================"
OUTFILE="output.txt"
EXPECTED="medium/output/matrix3.txt"

if [[ -f "$OUTFILE" && -f "$EXPECTED" ]]; then
    echo "[Verify] Comparing output with expected..."
    if diff -q "$OUTFILE" "$EXPECTED" > /dev/null 2>&1; then
        echo "  -> PASS: output matches expected"
    else
        echo "  -> DIFF: output differs from expected"
        diff -y "$OUTFILE" "$EXPECTED" | head -10 || true
    fi
else
    if [[ ! -f "$EXPECTED" ]]; then
        echo "[Verify] No reference output at $EXPECTED — skipping verification"
    fi
fi

echo ""
echo "============================================"
echo " Benchmark complete!"
echo "============================================"
