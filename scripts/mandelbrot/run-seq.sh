#!/usr/bin/env bash
#
# run-seq.sh — Build and run the Mandelbrot sequential (baseline) benchmark inside Docker
#
# Usage:
#   ./scripts/mandelbrot/run-seq.sh
#
# Requirements:
#   - Docker installed and running
#

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"

SRC_FILE="$REPO_ROOT/problems/mandelbrot/original/mandelbrot_seq.c"

if [[ ! -f "$SRC_FILE" ]]; then
    echo "Error: source file not found: $SRC_FILE"
    exit 1
fi

echo "============================================"
echo " Mandelbrot Sequential (Baseline) Runner"
echo "============================================"
echo " Source:  $SRC_FILE"
echo "============================================"

# --- Prepare workspace ---

WORK_DIR=$(mktemp -d)
cp "$SRC_FILE" "$WORK_DIR/mandelbrot.c"

echo ""
echo "[1/3] Source copied to workspace"

# --- Docker script ---

DOCKER_SCRIPT=$(cat << 'INNEREOF'
#!/bin/bash
set -e

echo ""
echo "[2/3] Installing build dependencies..."
apt-get update -qq > /dev/null 2>&1
apt-get install -y -qq build-essential > /dev/null 2>&1
echo "  -> gcc $(gcc -dumpversion)"

echo ""
echo "[3/3] Compiling and running..."
gcc -O2 -o mandelbrot mandelbrot.c -lm
echo "  -> Compiled successfully (no OpenMP)"

echo ""
echo "============================================"
echo " Running Mandelbrot sequential baseline"
echo "============================================"
echo ""

time ./mandelbrot

echo ""
echo "============================================"
echo " Benchmark complete!"
echo " Output: mandelbrot_openmp.ppm"
echo "============================================"
INNEREOF
)

# --- Run ---

echo ""
echo "Starting Docker container (ubuntu:22.04)..."
echo ""

docker run -it --rm \
    -v "$WORK_DIR:/work" \
    -w /work \
    ubuntu:22.04 \
    bash -c "$DOCKER_SCRIPT"

# --- Cleanup ---
rm -rf "$WORK_DIR"
