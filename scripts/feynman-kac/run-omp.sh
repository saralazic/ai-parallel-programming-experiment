#!/usr/bin/env bash
#
# run-omp.sh — Build and run the Feynman-Kac OpenMP benchmark inside Docker
#
# Usage:
#   ./scripts/feynman-kac/run-omp.sh <model> [threads]
#   ./scripts/feynman-kac/run-omp.sh codex    4
#   ./scripts/feynman-kac/run-omp.sh opus     8
#   ./scripts/feynman-kac/run-omp.sh gemini   4
#   ./scripts/feynman-kac/run-omp.sh original 4
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
        SRC_FILE="$REPO_ROOT/problems/feynman-kac/results/codex/code/openmp/feynman-kac-codex.c"
        ;;
    opus)
        SRC_FILE="$REPO_ROOT/problems/feynman-kac/results/opus/code/openmp/feynman-kac-opus.c"
        ;;
    gemini)
        SRC_FILE="$REPO_ROOT/problems/feynman-kac/results/gemini/code/openmp/feynman-kac-gemini.c"
        ;;
    original)
        SRC_FILE="$REPO_ROOT/problems/feynman-kac/original/feynman-kac-seq.c"
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
echo " Feynman-Kac OpenMP Benchmark Runner"
echo "============================================"
echo " Model:        $MODEL"
echo " Threads:      $NUM_THREADS"
echo " Source:       $SRC_FILE"
echo "============================================"

# --- Prepare workspace with source file ---

WORK_DIR=$(mktemp -d)
cp "$SRC_FILE" "$WORK_DIR/feynman-kac.c"

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
gcc -O2 -fopenmp -o feynman-kac feynman-kac.c -lm
echo "  -> Compiled successfully"

echo ""
echo "[3/3] Running benchmark..."
echo ""
echo "============================================"
INNEREOF
)

DOCKER_SCRIPT+="
echo \" Running: OMP_NUM_THREADS=$NUM_THREADS\"
echo \"============================================\"
echo \"\"

export OMP_NUM_THREADS=$NUM_THREADS
time ./feynman-kac

echo \"\"
echo \"============================================\"
echo \" Benchmark complete!\"
echo \"============================================\"
"

# --- Run Docker container ---

echo ""
echo "Starting Docker container (ubuntu:22.04)..."
echo ""

docker run -it --rm \
    -v "$WORK_DIR:/app" \
    -w /app \
    ubuntu:22.04 \
    bash -c "$DOCKER_SCRIPT"

# --- Cleanup ---
rm -rf "$WORK_DIR"
