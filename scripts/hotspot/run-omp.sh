#!/usr/bin/env bash
#
# run-omp.sh — Build and run the Hotspot OpenMP benchmark inside Docker
#
# Usage:
#   ./scripts/hotspot/run-omp.sh <model> [threads] [grid] [iterations]
#   ./scripts/hotspot/run-omp.sh codex    4   512   10000
#   ./scripts/hotspot/run-omp.sh opus     10  1024  10000
#   ./scripts/hotspot/run-omp.sh gemini   4   512   10000
#
# Requirements:
#   - Docker installed and running
#   - Data files in problems/hotspot/data/
#

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"

MODEL="${1:-}"
NUM_THREADS="${2:-4}"
GRID_SIZE="${3:-512}"
ITERATIONS="${4:-10000}"

# --- Validation ---

if [[ -z "$MODEL" ]]; then
    echo "Error: model argument required."
    echo "Usage: $0 <model> [threads] [grid_size] [iterations]"
    echo "  model:      codex | opus | gemini"
    echo "  threads:    number of OpenMP threads (default: 4)"
    echo "  grid_size:  512 or 1024 (default: 512)"
    echo "  iterations: simulation iterations (default: 10000)"
    exit 1
fi

if [[ "$GRID_SIZE" != "512" && "$GRID_SIZE" != "1024" ]]; then
    echo "Error: grid_size must be 512 or 1024"
    exit 1
fi

# --- Resolve source file ---

case "$MODEL" in
    codex)
        SRC_FILE="$REPO_ROOT/problems/hotspot/results/codex/code/openmp/hotspot-codex.cpp"
        ;;
    opus)
        SRC_FILE="$REPO_ROOT/problems/hotspot/results/opus/code/openmp/hotspot-opus.cpp"
        ;;
    gemini)
        SRC_FILE="$REPO_ROOT/problems/hotspot/results/gemini/code/openmp/hotspot-gemini.cpp"
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

DATA_DIR="$REPO_ROOT/problems/hotspot/data"
if [[ ! -f "$DATA_DIR/temp_${GRID_SIZE}" ]]; then
    echo "Error: data file not found: $DATA_DIR/temp_${GRID_SIZE}"
    echo "Make sure hotspot data files are in problems/hotspot/data/"
    exit 1
fi

echo "============================================"
echo " Hotspot OpenMP Benchmark Runner"
echo "============================================"
echo " Model:        $MODEL"
echo " Threads:      $NUM_THREADS"
echo " Grid:         ${GRID_SIZE}x${GRID_SIZE}"
echo " Iterations:   $ITERATIONS"
echo " Source:       $SRC_FILE"
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
