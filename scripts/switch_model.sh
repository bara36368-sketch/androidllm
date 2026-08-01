#!/data/data/com.termux/files/usr/bin/bash
# Switch which sharded model androidllm serves. Only one model runs at a time:
# the previous serve is killed and runner.py restarts androidllm-serve on the
# new model within a poll cycle.
# Usage: bash switch_model.sh <id>    e.g. qwen15 | smollm2 | qwen3
set -euo pipefail

ID="${1:?usage: switch_model.sh <id>}"
APP_DIR="${ANDROIDLLM_DIR:-$HOME/androidllm}"
MODEL_DIR="$APP_DIR/models/$ID"
STATE="$APP_DIR/current_model.json"

if [ ! -f "$MODEL_DIR/manifest.json" ]; then
  echo "model '$ID' is not sharded yet: $MODEL_DIR"
  echo "shard it first:  bash $APP_DIR/scripts/shard_model.sh <HF_REPO> $ID"
  exit 1
fi

printf '{"id": "%s", "path": "%s"}\n' "$ID" "$MODEL_DIR" > "$STATE"
echo "switched to $ID (state written), restarting androidllm-serve..."
pkill -f "androidllm-serve" 2>/dev/null || true
echo "done. runner.py restarts the serve on the new model within ~15s."
