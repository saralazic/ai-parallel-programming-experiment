#!/usr/bin/env bash
#
# run-seq-remote.sh — Build and run Moldyn sequential baseline (no Docker)
#
# Usage (from repo root on remote machine):
#   ./scripts/moldyn/run-seq-remote.sh [particles]
#   ./scripts/moldyn/run-seq-remote.sh 20
#

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"

PARTICLES="${1:-20}"

SRC_FILE="$REPO_ROOT/problems/moldyn/original/moldyn-seq.c"

echo "============================================"
echo " Moldyn — Sequential Baseline (particles=$PARTICLES)"
echo "============================================"

WORK_DIR=$(mktemp -d)
cp "$SRC_FILE" "$WORK_DIR/moldyn.c"
cd "$WORK_DIR"

gcc -O2 -o moldyn moldyn.c -lm
echo "  -> Compiled"
echo ""

time ./moldyn $PARTICLES

echo ""
echo "============================================"
echo " Done!"
echo "============================================"

cd /
rm -rf "$WORK_DIR"
