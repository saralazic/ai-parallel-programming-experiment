#!/usr/bin/env bash
#
# run-omp-reference.sh — Build and run Rodinia Hotspot OpenMP reference inside Docker
#
# Usage:
#   ./scripts/hotspot/run-omp-reference.sh [threads] [grid_size] [iterations]
#   ./scripts/hotspot/run-omp-reference.sh 10
#   ./scripts/hotspot/run-omp-reference.sh 8 1024 10000
#
# Requirements:
#   - Docker installed and running
#   - Data files in problems/hotspot/data/
#

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"

NUM_THREADS="${1:-10}"
GRID_SIZE="${2:-1024}"
ITERATIONS="${3:-10000}"

SRC_FILE="$REPO_ROOT/problems/hotspot/reference/omp/hotspot.cpp"
DATA_DIR="$REPO_ROOT/problems/hotspot/data"

if [[ ! -f "$SRC_FILE" ]]; then
    echo "Error: reference OMP source not found: $SRC_FILE"
    exit 1
fi

TEMP_FILE="$DATA_DIR/temp_${GRID_SIZE}"
POWER_FILE="$DATA_DIR/power_${GRID_SIZE}"

if [[ ! -f "$TEMP_FILE" ]]; then
    echo "Error: data file not found: $TEMP_FILE"
    exit 1
fi

echo "============================================"
echo " Hotspot OpenMP Reference (Rodinia)"
echo "============================================"
echo " Threads:      $NUM_THREADS"
echo " Grid:         ${GRID_SIZE}x${GRID_SIZE}"
echo " Iterations:   $ITERATIONS"
echo "============================================"

WORK_DIR=$(mktemp -d)
cp "$SRC_FILE" "$WORK_DIR/hotspot.cpp"
cp "$TEMP_FILE" "$WORK_DIR/"
cp "$POWER_FILE" "$WORK_DIR/"

DOCKER_SCRIPT=$(cat << 'INNEREOF'
#!/bin/bash
set -e

echo ""
echo "[1/2] Installing build dependencies..."
apt-get update -qq > /dev/null 2>&1
apt-get install -y -qq g++ libomp-dev > /dev/null 2>&1
echo "  -> g++ $(g++ -dumpversion)"

echo ""
echo "[2/2] Compiling..."
g++ -O2 -fopenmp -o hotspot hotspot.cpp
echo "  -> Compiled successfully"
INNEREOF
)

DOCKER_SCRIPT+="
echo \"\"
echo \"============================================\"
echo \" Running: threads=$NUM_THREADS grid=${GRID_SIZE}x${GRID_SIZE} iter=$ITERATIONS\"
echo \"============================================\"
echo \"\"

export OMP_NUM_THREADS=$NUM_THREADS
time ./hotspot $GRID_SIZE $GRID_SIZE $ITERATIONS $NUM_THREADS temp_${GRID_SIZE} power_${GRID_SIZE} output.txt

echo \"\"
echo \"============================================\"
echo \" Benchmark complete!\"
echo \"============================================\"
"

echo ""
echo "Starting Docker container (ubuntu:22.04)..."
echo ""

docker run -it --rm \
    -v "$WORK_DIR:/work" \
    -w /work \
    ubuntu:22.04 \
    bash -c "$DOCKER_SCRIPT"

rm -rf "$WORK_DIR"
