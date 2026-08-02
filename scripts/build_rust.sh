#!/data/data/com.termux/files/usr/bin/bash
# Build the PyO3 Rust accelerator (androidllm_rs).
# Requires: rust + maturin. On Termux:
#   pkg install rust
#   python -m pip install maturin
# Produces and installs the androidllm_rs module; numpy fallback otherwise.
set -e

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
ROOT="$(dirname "$SCRIPT_DIR")"
CRATE="$ROOT/androidllm_rs"

if [ ! -d "$CRATE" ]; then
  echo ">> androidllm_rs crate not found at $CRATE; skipping Rust build"
  echo "   (numpy fallback will be used)"
  exit 0
fi

cd "$CRATE"

if ! command -v cargo >/dev/null 2>&1; then
  echo ">> cargo not found (pkg install rust); skipping Rust build"
  exit 0
fi

if ! python -c "import maturin" >/dev/null 2>&1; then
  echo ">> maturin not found (python -m pip install maturin); skipping Rust build"
  exit 0
fi

echo ">> building androidllm_rs with maturin ..."
python -m maturin build --release --interpreter python
WHEEL="$(ls -t target/wheels/androidllm_rs-*.whl | head -1)"
echo ">> installing $WHEEL"
python -m pip install --force-reinstall "$WHEEL"
echo ">> done: androidllm_rs installed"
