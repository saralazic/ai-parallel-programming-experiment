#!/usr/bin/env bash
#
# run-omp-reference.sh — Build and run Mandelbrot OpenMP reference inside Docker
#
# Usage:
#   ./scripts/mandelbrot/run-omp-reference.sh [threads]
#   ./scripts/mandelbrot/run-omp-reference.sh 10
#
# Requirements:
#   - Docker installed and running
#

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"

NUM_THREADS="${1:-10}"

SRC_FILE="$REPO_ROOT/problems/mandelbrot/reference/omp/mandelbrot.c"

if [[ ! -f "$SRC_FILE" ]]; then
    echo "Error: reference OMP source not found: $SRC_FILE"
    exit 1
fi

echo "============================================"
echo " Mandelbrot OpenMP Reference"
echo "============================================"
echo " Threads:      $NUM_THREADS"
echo " Source:       $SRC_FILE"
echo "============================================"

WORK_DIR=$(mktemp -d)
cp "$SRC_FILE" "$WORK_DIR/mandelbrot.c"

DOCKER_SCRIPT=$(cat << 'INNEREOF'
#!/bin/bash
set -e

echo ""
echo "[1/2] Installing build dependencies..."
apt-get update -qq > /dev/null 2>&1
apt-get install -y -qq build-essential libomp-dev > /dev/null 2>&1
echo "  -> gcc $(gcc -dumpversion)"

echo ""
echo "[2/2] Compiling..."
gcc -O2 -fopenmp mandelbrot.c -lm -o mandelbrot
echo "  -> Compiled successfully"

echo ""
echo "============================================"
INNEREOF
)

DOCKER_SCRIPT+="
echo \" Running: OMP_NUM_THREADS=$NUM_THREADS\"
echo \"============================================\"
echo \"\"

export OMP_NUM_THREADS=$NUM_THREADS
time ./mandelbrot

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
