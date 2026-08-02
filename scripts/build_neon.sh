#!/data/data/com.termux/files/usr/bin/bash
# Build the fp16 NEON matmul kernel.
# Requires: clang (pkg install clang). Produces libandroidllm_neon.so
# next to the androidllm package or in ~/.androidllm.
set -e

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
ROOT="$(dirname "$SCRIPT_DIR")"
SRC="$ROOT/src/androidllm_neon.c"
OUT_DIR="${1:-$HOME/.androidllm}"

mkdir -p "$OUT_DIR"
echo ">> compiling $SRC -> $OUT_DIR/libandroidllm_neon.so"
clang -O3 -ffast-math -fPIC -shared -pthread -o "$OUT_DIR/libandroidllm_neon.so" "$SRC"
echo ">> done: $OUT_DIR/libandroidllm_neon.so"
