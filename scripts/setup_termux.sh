#!/data/data/com.termux/files/usr/bin/bash
# androidllm setup for plain Termux (no proot, no distro).
# One-time env setup: installs deps, copies the repo, builds the NEON kernel,
# installs the package, then shards the requested model.
#
# Usage:
#   bash setup_termux.sh [MODEL_REPO] [ID]
#   bash setup_termux.sh Qwen/Qwen2.5-1.5B-Instruct qwen15
#
# Works whether you cloned the repo to ~/androidllm or copied just this
# script; the full repo is copied into ~/androidllm if needed.
set -euo pipefail

MODEL_REPO="${1:-Qwen/Qwen2.5-1.5B-Instruct}"
ID="${2:-qwen15}"
APP_DIR="$HOME/androidllm"

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
        "$ROOT/androidllm_rs" \
        "$ROOT/pyproject.toml" "$ROOT/README.md" "$APP_DIR/"
fi
cd "$APP_DIR"
python -m pip install -e "$APP_DIR" || python -m pip install "$APP_DIR"

echo ">> building NEON kernel ..."
bash "$APP_DIR/scripts/build_neon.sh" || echo "(neon build skipped; numpy fallback will be used)"

echo ">> building Rust accelerator (androidllm_rs) ..."
pkg install -y rust || true
python -m pip install maturin || true
bash "$APP_DIR/scripts/build_rust.sh" || echo "(rust build skipped; numpy fallback will be used)"

echo ">> sharding ${MODEL_REPO} -> models/${ID} ..."
bash "$APP_DIR/scripts/shard_model.sh" "$MODEL_REPO" "$ID"

echo
echo "== done. Serve with:"
echo "   androidllm-serve --model $APP_DIR/models/$ID --port 8080"
echo "Switch models anytime:  bash $APP_DIR/scripts/switch_model.sh <id>"
