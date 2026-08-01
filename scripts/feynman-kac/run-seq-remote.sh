#!/usr/bin/env bash
#
# run-seq-remote.sh — Build and run Feynman-Kac sequential baseline (no Docker)
#
# Usage (from repo root on remote machine):
#   ./scripts/feynman-kac/run-seq-remote.sh
#

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"

SRC_FILE="$REPO_ROOT/problems/feynman-kac/original/feynman-kac-seq.c"

echo "============================================"
echo " Feynman-Kac — Sequential Baseline"
echo "============================================"

WORK_DIR=$(mktemp -d)
cp "$SRC_FILE" "$WORK_DIR/feynman-kac.c"
cd "$WORK_DIR"

gcc -O2 -o feynman-kac feynman-kac.c -lm
echo "  -> Compiled"
echo ""

time ./feynman-kac

echo ""
echo "============================================"
echo " Done!"
echo "============================================"

cd /
rm -rf "$WORK_DIR"
