#!/data/data/com.termux/files/usr/bin/bash
# androidllm setup for plain Termux (no proot, no distro).
# Installs the package, builds the NEON kernel, downloads a model from HF,
# and shards it into androidllm layer files.
#
# Usage:
#   bash setup_termux.sh [MODEL_ID] [OUT_NAME]
#   bash setup_termux.sh Qwen/Qwen2.5-1.5B-Instruct qwen15
#
# Works whether you cloned the repo to ~/androidllm or copied just this
# script; the full repo is copied into ~/androidllm if needed.
set -euo pipefail

MODEL_ID="${1:-Qwen/Qwen2.5-1.5B-Instruct}"
OUT_NAME="${2:-qwen15}"
APP_DIR="$HOME/androidllm"
MODEL_DIR="$APP_DIR/models/src-${OUT_NAME}"
OUT_DIR="$APP_DIR/models/${OUT_NAME}"
BASE="https://huggingface.co/${MODEL_ID}/resolve/main"

echo "== androidllm setup =="
pkg update -y
pkg install -y python clang make git wget openblas || true

python -m pip install --upgrade pip
python -m pip install numpy

# Make sure the repo (or at least the package) lives in ~/androidllm.
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
ROOT="$(dirname "$SCRIPT_DIR")"
if [ ! -e "$APP_DIR/pyproject.toml" ]; then
  echo ">> copying androidllm repo into $APP_DIR ..."
  mkdir -p "$APP_DIR"
  cp -r "$ROOT/androidllm" "$ROOT/src" "$ROOT/scripts" \
        "$ROOT/pyproject.toml" "$ROOT/README.md" "$APP_DIR/"
fi
cd "$APP_DIR"
python -m pip install -e "$APP_DIR" || python -m pip install "$APP_DIR"

echo ">> building NEON kernel ..."
bash "$APP_DIR/scripts/build_neon.sh" || echo "(neon build skipped; numpy fallback will be used)"

# Download config + tokenizer + a single-shard weight file with resume.
echo ">> downloading ${MODEL_ID} ..."
mkdir -p "$MODEL_DIR"
for f in config.json tokenizer.json tokenizer_config.json generation_config.json model.safetensors; do
  if [ ! -s "$MODEL_DIR/$f" ]; then
    wget -q --show-progress --continue -O "$MODEL_DIR/$f" "$BASE/$f" \
      || echo "(failed to fetch $f; skipping)"
  fi
done

echo ">> sharding ${MODEL_ID} -> $OUT_DIR ..."
python -m androidllm.shard --source "$MODEL_DIR" --out "$OUT_DIR"

echo
echo "== done. Serve with:"
echo "   androidllm-serve --model $OUT_DIR --port 8080"
