#!/usr/bin/env bash
#
# run-cuda-reference.sh — Build and run Parboil SGEMM CUDA reference on remote GPU machine
#
# Usage (run from the repo root on the CUDA machine):
#   ./scripts/sgemm/run-cuda-reference.sh <variant>
#   ./scripts/sgemm/run-cuda-reference.sh base
#   ./scripts/sgemm/run-cuda-reference.sh optimized
#
# Prerequisites:
#   - Running on a machine with nvcc + GPU
#   - Repo cloned/copied to the machine
#

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"

VARIANT="${1:-}"

# --- Validation ---

if [[ -z "$VARIANT" ]]; then
    echo "Error: variant argument required."
    echo "Usage: $0 <variant>"
    echo "  variant: base | optimized"
    exit 1
fi

case "$VARIANT" in
    base)
        SRC_DIR="$REPO_ROOT/problems/sgemm/reference/cuda"
        ;;
    optimized)
        SRC_DIR="$REPO_ROOT/problems/sgemm/reference/cuda_optimized"
        ;;
    *)
        echo "Error: unknown variant '$VARIANT'. Use: base | optimized"
        exit 1
        ;;
esac

if [[ ! -d "$SRC_DIR" ]]; then
    echo "Error: source directory not found: $SRC_DIR"
    exit 1
fi

echo "============================================"
echo " SGEMM CUDA Reference Benchmark Runner"
echo "============================================"
echo " Variant:      $VARIANT"
echo " Source:       $SRC_DIR"
echo "============================================"

# --- Check for nvcc ---

if ! command -v nvcc &> /dev/null; then
    echo "Error: nvcc not found. Please ensure CUDA toolkit is installed."
    exit 1
fi
echo " NVCC:         $(nvcc --version | grep release | awk '{print $5}' | tr -d ',')"
echo "============================================"

# --- Ensure data files are present ---

DATA_SRC="$REPO_ROOT/problems/sgemm/original/base-without-parboil/medium"
if [[ ! -d "$SRC_DIR/medium" ]]; then
    echo ""
    echo "[0/3] Copying input data..."
    cp -r "$DATA_SRC" "$SRC_DIR/medium"
    echo "  -> Data copied to $SRC_DIR/medium"
fi

cd "$SRC_DIR"

# --- Clean previous build ---

echo ""
echo "[1/3] Cleaning previous build..."
make clean 2>/dev/null || true

# --- Compile ---

echo ""
echo "[2/3] Compiling..."
make NVCCFLAGS="-O2 -arch=sm_86"

# --- Run ---

echo ""
echo "============================================"
echo " Running SGEMM CUDA reference ($VARIANT)"
echo "============================================"
echo ""

time make run

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
        diff "$OUTFILE" "$EXPECTED" | head -10 || true
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
