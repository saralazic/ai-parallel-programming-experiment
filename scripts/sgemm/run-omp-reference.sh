#!/usr/bin/env bash
#
# run-omp-reference.sh — Build and run the Parboil SGEMM OpenMP reference inside Docker
#
# Usage:
#   ./scripts/sgemm/run-omp-reference.sh [threads]
#   ./scripts/sgemm/run-omp-reference.sh 10
#
# Requirements:
#   - Docker installed and running
#   - ~/Desktop/parboil directory with Parboil benchmark suite
#

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"
PARBOIL_HOST="$HOME/Desktop/parboil"
PARBOIL_CONTAINER="/parboil"
BENCHMARK="sgemm"
VARIANT="omp_base"

OMP_THREADS="${1:-10}"

# --- Validation ---

if [[ ! -d "$PARBOIL_HOST" ]]; then
    echo "Error: Parboil directory not found at $PARBOIL_HOST"
    exit 1
fi

SRC_DIR="$REPO_ROOT/problems/sgemm/reference/omp"

if [[ ! -d "$SRC_DIR" ]]; then
    echo "Error: reference OMP source not found: $SRC_DIR"
    exit 1
fi

echo "============================================"
echo " SGEMM OpenMP Reference (Parboil omp_base)"
echo "============================================"
echo " Threads:      $OMP_THREADS"
echo " Source:       $SRC_DIR"
echo "============================================"

# --- Copy reference source files into Parboil benchmark directory ---

DEST_DIR="$PARBOIL_HOST/benchmarks/$BENCHMARK/src/$VARIANT"
mkdir -p "$DEST_DIR"

echo ""
echo "[1/4] Copying reference source files to Parboil..."
cp "$SRC_DIR"/main.cc "$DEST_DIR/"
cp "$SRC_DIR"/io.cc "$DEST_DIR/"
cp "$SRC_DIR"/sgemm_kernel.cc "$DEST_DIR/"
cp "$SRC_DIR"/Makefile "$DEST_DIR/"
echo "  -> Files copied to $DEST_DIR"

# --- Create Makefile.conf if missing ---

MAKEFILE_CONF="$PARBOIL_HOST/common/Makefile.conf"
if [[ ! -f "$MAKEFILE_CONF" ]]; then
    echo ""
    echo "[*] Creating Makefile.conf..."
    cat > "$MAKEFILE_CONF" << 'EOF'
PARBOIL_COMMON = $(PARBOIL_ROOT)/common

CC = gcc
CXX = g++

CFLAGS += -fopenmp -I$(PARBOIL_COMMON)/include
CXXFLAGS += -fopenmp -I$(PARBOIL_COMMON)/include
LDFLAGS += -fopenmp -lomp -lpthread -lm

NVCC =
CUDA_PATH =
OPENCL_PATH =
OPENCL_LIB_PATH =
EOF
    echo "  -> Created $MAKEFILE_CONF"
fi

# --- Build the Docker run script ---

DOCKER_SCRIPT=$(cat << INNEREOF
#!/bin/bash
set -e

echo ""
echo "[2/4] Installing build dependencies..."
apt-get update -qq > /dev/null 2>&1
apt-get install -y -qq build-essential wget curl libreadline-dev libncurses5-dev \
    libssl-dev zlib1g-dev libbz2-dev libsqlite3-dev libffi-dev libomp-dev \
    > /dev/null 2>&1
echo "  -> Dependencies installed"

echo ""
echo "[3/4] Installing Python 2.7.18..."
if [[ ! -f /usr/local/python2/bin/python2 ]]; then
    cd /tmp
    wget -q https://www.python.org/ftp/python/2.7.18/Python-2.7.18.tgz
    tar xf Python-2.7.18.tgz
    cd Python-2.7.18
    ./configure --prefix=/usr/local/python2 --enable-shared > /dev/null 2>&1
    make -j\$(nproc) > /dev/null 2>&1
    make install > /dev/null 2>&1
fi

export PATH=/usr/local/python2/bin:\$PATH
export LD_LIBRARY_PATH=/usr/local/python2/lib:\${LD_LIBRARY_PATH:-}
echo "  -> Python version: \$(python2 --version 2>&1)"

export PARBOIL_ROOT=$PARBOIL_CONTAINER
cd \$PARBOIL_ROOT

echo ""
echo "[4/4] Compiling and running benchmark..."
echo "  -> Cleaning previous build..."
python2 ./parboil clean $BENCHMARK $VARIANT 2>/dev/null || true

echo "  -> Compiling $BENCHMARK ($VARIANT)..."
python2 ./parboil compile $BENCHMARK $VARIANT

echo ""
echo "============================================"
echo " Running: OMP_NUM_THREADS=$OMP_THREADS"
echo "============================================"
echo ""

export OMP_NUM_THREADS=$OMP_THREADS
time python2 ./parboil run $BENCHMARK $VARIANT medium

echo ""
echo "============================================"
echo " Benchmark complete!"
echo "============================================"
INNEREOF
)

# --- Run Docker container ---

echo ""
echo "Starting Docker container (ubuntu:22.04)..."
echo ""

docker run -it --rm \
    -v "$PARBOIL_HOST:$PARBOIL_CONTAINER" \
    ubuntu:22.04 \
    bash -c "$DOCKER_SCRIPT"
