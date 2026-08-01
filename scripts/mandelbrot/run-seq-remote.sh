#!/usr/bin/env bash
#
# run-seq-remote.sh — Build and run Mandelbrot sequential baseline (no Docker)
#
# Usage (from repo root on remote machine):
#   ./scripts/mandelbrot/run-seq-remote.sh
#

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"

SRC_FILE="$REPO_ROOT/problems/mandelbrot/original/mandelbrot_seq.c"

echo "============================================"
echo " Mandelbrot — Sequential Baseline"
echo "============================================"

WORK_DIR=$(mktemp -d)
cp "$SRC_FILE" "$WORK_DIR/mandelbrot.c"
cd "$WORK_DIR"

gcc -O2 -o mandelbrot mandelbrot.c -lm
echo "  -> Compiled"
echo ""

time ./mandelbrot

echo ""
echo "============================================"
echo " Done!"
echo "============================================"

cd /
rm -rf "$WORK_DIR"
