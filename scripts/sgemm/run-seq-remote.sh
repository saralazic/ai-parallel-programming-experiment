#!/usr/bin/env bash
#
# run-seq-remote.sh — Build and run SGEMM sequential baseline (no Docker)
#
# Usage (from repo root on remote machine):
#   ./scripts/sgemm/run-seq-remote.sh
#

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"

SRC_DIR="$REPO_ROOT/problems/sgemm/original/base-without-parboil"

echo "============================================"
echo " SGEMM — Sequential Baseline"
echo "============================================"

WORK_DIR=$(mktemp -d)
cp "$SRC_DIR"/*.cc "$WORK_DIR/"
cp -r "$SRC_DIR"/medium "$WORK_DIR/"
cd "$WORK_DIR"

g++ -O2 -o sgemm main.cc io.cc -lm
echo "  -> Compiled"
echo ""

time ./sgemm medium/input/matrix1.txt medium/input/matrix2t.txt medium/input/matrix2t.txt output.txt

echo ""
echo "============================================"
echo " Done!"
echo "============================================"

cd /
rm -rf "$WORK_DIR"
