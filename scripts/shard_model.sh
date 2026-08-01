#!/data/data/com.termux/files/usr/bin/bash
# Download + shard one HF model into ~/androidllm/models/<id>.
# No dependency install (that's setup_termux.sh's job). Resumable via wget.
# Usage: bash shard_model.sh [HF_REPO] [ID]
#   e.g. bash shard_model.sh Qwen/Qwen2.5-1.5B-Instruct qwen15
set -euo pipefail

REPO="${1:?usage: shard_model.sh <HF_REPO> <id>}"
ID="${2:?usage: shard_model.sh <HF_REPO> <id>}"
APP_DIR="${ANDROIDLLM_DIR:-$HOME/androidllm}"
MODEL_DIR="$APP_DIR/models/src-${ID}"
OUT_DIR="$APP_DIR/models/${ID}"
BASE="https://huggingface.co/${REPO}/resolve/main"

mkdir -p "$MODEL_DIR"
for f in config.json tokenizer.json tokenizer_config.json generation_config.json model.safetensors; do
  if [ ! -s "$MODEL_DIR/$f" ]; then
    wget -q --show-progress --continue -O "$MODEL_DIR/$f" "$BASE/$f" \
      || echo "(failed to fetch $f; skipping)"
  fi
done

python -m androidllm.shard --source "$MODEL_DIR" --out "$OUT_DIR"
echo "sharded ${REPO} -> $OUT_DIR"
