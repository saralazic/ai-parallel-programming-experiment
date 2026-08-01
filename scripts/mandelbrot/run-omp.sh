#!/usr/bin/env bash
#
# run-omp.sh — Build and run the Mandelbrot OpenMP benchmark inside Docker
#
# Usage:
#   ./scripts/mandelbrot/run-omp.sh <model> [threads]
#   ./scripts/mandelbrot/run-omp.sh codex    4
#   ./scripts/mandelbrot/run-omp.sh opus     8
#   ./scripts/mandelbrot/run-omp.sh gemini   4
#
# Requirements:
#   - Docker installed and running
#

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"

MODEL="${1:-}"
NUM_THREADS="${2:-4}"

# --- Validation ---

if [[ -z "$MODEL" ]]; then
    echo "Error: model argument required."
    echo "Usage: $0 <model> [threads]"
    echo "  model:   codex | opus | gemini | original"
    echo "  threads: number of OpenMP threads (default: 4)"
    exit 1
fi

# --- Resolve source file ---

case "$MODEL" in
    codex)
        SRC_FILE="$REPO_ROOT/problems/mandelbrot/results/codex/code/openmp/mandelbrot-codex.c"
        ;;
    opus)
        SRC_FILE="$REPO_ROOT/problems/mandelbrot/results/opus/code/openmp/mandelbrot-opus.c"
        ;;
    gemini)
        SRC_FILE="$REPO_ROOT/problems/mandelbrot/results/gemini/code/openmp/mandelbrot-gemini.c"
        ;;
    original)
        SRC_FILE="$REPO_ROOT/problems/mandelbrot/original/mandelbrot_seq.c"
        ;;
    *)
        echo "Error: unknown model '$MODEL'. Use: codex | opus | gemini | original"
        exit 1
        ;;
esac

if [[ ! -f "$SRC_FILE" ]]; then
    echo "Error: source file not found: $SRC_FILE"
    exit 1
fi

echo "============================================"
echo " Mandelbrot OpenMP Benchmark Runner"
echo "============================================"
echo " Model:        $MODEL"
echo " Threads:      $NUM_THREADS"
echo " Source:       $SRC_FILE"
echo "============================================"

# --- Prepare workspace with source file ---

WORK_DIR=$(mktemp -d)
cp "$SRC_FILE" "$WORK_DIR/mandelbrot.c"

echo ""
echo "[1/3] Source copied to temp workspace"

# --- Build the Docker run script ---

DOCKER_SCRIPT=$(cat << 'INNEREOF'
#!/bin/bash
set -e

echo ""
echo "[2/3] Installing build dependencies..."
apt-get update -qq > /dev/null 2>&1
apt-get install -y -qq build-essential > /dev/null 2>&1
echo "  -> gcc $(gcc -dumpversion)"

echo ""
echo "[3/3] Compiling and running benchmark..."
gcc -O2 -fopenmp mandelbrot.c -lm -o mandelbrot
echo "  -> Compiled successfully"

echo ""
echo "============================================"
INNEREOF
)

# Append the dynamic part
DOCKER_SCRIPT+="
echo \" Running: OMP_NUM_THREADS=$NUM_THREADS\"
echo \"============================================\"
echo \"\"

export OMP_NUM_THREADS=$NUM_THREADS
time ./mandelbrot

echo \"\"
echo \"============================================\"
echo \" Benchmark complete!\"
echo \" Output: mandelbrot_openmp.ppm\"
echo \"============================================\"
"

# --- Run Docker container ---

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
