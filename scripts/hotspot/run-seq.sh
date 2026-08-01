#!/usr/bin/env bash
#
# run-seq.sh — Build and run the Hotspot sequential (baseline) benchmark inside Docker
#
# Usage:
#   ./scripts/hotspot/run-seq.sh [grid_size] [iterations]
#   ./scripts/hotspot/run-seq.sh 512 10000
#   ./scripts/hotspot/run-seq.sh 1024 10000
#
# Requirements:
#   - Docker installed and running
#   - Data files in problems/hotspot/data/
#

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"

GRID_SIZE="${1:-512}"
ITERATIONS="${2:-10000}"

SRC_FILE="$REPO_ROOT/problems/hotspot/original/hotspot-seq.cpp"
DATA_DIR="$REPO_ROOT/problems/hotspot/data"

if [[ ! -f "$SRC_FILE" ]]; then
    echo "Error: source file not found: $SRC_FILE"
    exit 1
fi

if [[ "$GRID_SIZE" != "512" && "$GRID_SIZE" != "1024" ]]; then
    echo "Error: grid_size must be 512 or 1024"
    exit 1
fi

if [[ ! -f "$DATA_DIR/temp_${GRID_SIZE}" ]]; then
    echo "Error: data file not found: $DATA_DIR/temp_${GRID_SIZE}"
    echo "Make sure hotspot data files are in problems/hotspot/data/"
    exit 1
fi

echo "============================================"
echo " Hotspot Sequential (Baseline) Runner"
echo "============================================"
echo " Grid:         ${GRID_SIZE}x${GRID_SIZE}"
echo " Iterations:   $ITERATIONS"
echo "============================================"

# --- Prepare workspace ---

WORK_DIR=$(mktemp -d)
cp "$SRC_FILE" "$WORK_DIR/hotspot.cpp"
cp "$DATA_DIR/temp_${GRID_SIZE}" "$WORK_DIR/"
cp "$DATA_DIR/power_${GRID_SIZE}" "$WORK_DIR/"

echo ""
echo "[1/3] Source and data copied to workspace"

# --- Docker script ---

DOCKER_SCRIPT=$(cat << 'INNEREOF'
#!/bin/bash
set -e

echo ""
echo "[2/3] Installing build dependencies..."
apt-get update -qq > /dev/null 2>&1
apt-get install -y -qq g++ > /dev/null 2>&1
echo "  -> g++ $(g++ -dumpversion)"

echo ""
echo "[3/3] Compiling and running..."
g++ -O2 -o hotspot hotspot.cpp
echo "  -> Compiled successfully (no OpenMP)"
INNEREOF
)

DOCKER_SCRIPT+="
echo \"\"
echo \"============================================\"
echo \" Running Hotspot sequential: grid=${GRID_SIZE}x${GRID_SIZE} iter=$ITERATIONS\"
echo \"============================================\"
echo \"\"

time ./hotspot $GRID_SIZE $GRID_SIZE $ITERATIONS 1 temp_${GRID_SIZE} power_${GRID_SIZE} output.txt

echo \"\"
echo \"============================================\"
echo \" Benchmark complete!\"
echo \"============================================\"
"

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
