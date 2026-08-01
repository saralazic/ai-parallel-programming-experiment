#!/usr/bin/env bash
#
# run-seq-remote.sh — Build and run Hotspot sequential baseline (no Docker)
#
# Usage (from repo root on remote machine):
#   ./scripts/hotspot/run-seq-remote.sh [grid_size] [iterations]
#   ./scripts/hotspot/run-seq-remote.sh 512 10000
#   ./scripts/hotspot/run-seq-remote.sh 1024 10000
#

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"

GRID_SIZE="${1:-512}"
ITERATIONS="${2:-10000}"

SRC_FILE="$REPO_ROOT/problems/hotspot/original/hotspot-seq.cpp"
DATA_DIR="$REPO_ROOT/problems/hotspot/data"

if [[ ! -f "$DATA_DIR/temp_${GRID_SIZE}" ]]; then
    echo "Error: data file not found: $DATA_DIR/temp_${GRID_SIZE}"
    exit 1
fi

echo "============================================"
echo " Hotspot — Sequential Baseline (${GRID_SIZE}x${GRID_SIZE}, $ITERATIONS iter)"
echo "============================================"

WORK_DIR=$(mktemp -d)
cp "$SRC_FILE" "$WORK_DIR/hotspot.cpp"
cd "$WORK_DIR"

g++ -O2 -o hotspot hotspot.cpp
echo "  -> Compiled"
echo ""

time ./hotspot $GRID_SIZE $GRID_SIZE $ITERATIONS 1 "$DATA_DIR/temp_${GRID_SIZE}" "$DATA_DIR/power_${GRID_SIZE}" output.txt

echo ""
echo "============================================"
echo " Done!"
echo "============================================"

cd /
rm -rf "$WORK_DIR"
