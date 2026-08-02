#!/data/data/com.termux/files/usr/bin/bash
# androidllm on-device speech-to-text (whisper.cpp, no cloud).
#
# Usage:
#   bash asr.sh setup                 - one-time: install deps + build whisper.cpp
#   bash asr.sh transcribe FILE       - ffmpeg -> 16k mono wav -> whisper txt
#   bash asr.sh transcribe FILE hi    - same, language hint (en/hi/...)
#   bash asr.sh mic 60                - record 60s from mic, transcribe, print text
#   bash asr.sh dictate [max_secs]    - dictation: record until silence, then print
#   bash asr.sh translate FILE [lang] - transcribe + translate via local androidllm
#
# Telegram voice notes arrive as .ogg (opus); ffmpeg handles them directly.
# Output: stdout = transcript text (also written to FILE.txt).

set -e

DIR="$(cd "$(dirname "$0")/.." && pwd)"
WHISPER_DIR="$HOME/whisper.cpp"
MODEL_DIR="$DIR/models/whisper"
MODEL="$MODEL_DIR/ggml-base.en.bin"
NPROC="$(nproc 2>/dev/null || echo 4)"

setup() {
    pkg install -y git cmake clang make ffmpeg wget
    if [ ! -d "$WHISPER_DIR" ]; then
        git clone --depth 1 https://github.com/ggerganov/whisper.cpp "$WHISPER_DIR"
    fi
    cd "$WHISPER_DIR"
    cmake -S . -B build -DGGML_NO_OPENMP=ON -DGGML_NATIVE=OFF
    cmake --build build -j"$NPROC"
    mkdir -p "$MODEL_DIR"
    bash "$WHISPER_DIR/models/download-ggml-model.sh" base.en "$MODEL_DIR"
    echo ">> asr ready. model: $MODEL"
    echo ">> try: bash asr.sh mic 10"
}

transcribe() {
    local src="$1"
    local lang="${2:-auto}"
    [ -f "$src" ] || { echo "file not found: $src" >&2; exit 1; }
    local wav="/tmp/asr_$$.wav"
    ffmpeg -y -i "$src" -ar 16000 -ac 1 -c:a pcm_s16le "$wav" >/dev/null 2>&1
    local out="/tmp/asr_$$"
    if [ "$lang" = "auto" ]; then
        "$WHISPER_DIR/build/bin/whisper-cli" -m "$MODEL" -f "$wav" \
            -otxt -of "$out" >/dev/null 2>&1
    else
        "$WHISPER_DIR/build/bin/whisper-cli" -m "$MODEL" -f "$wav" -l "$lang" \
            -otxt -of "$out" >/dev/null 2>&1
    fi
    rm -f "$wav"
    if [ -f "$out.txt" ]; then
        cat "$out.txt"
        cp "$out.txt" "$src.txt"
        rm -f "$out.txt"
    else
        echo "(asr failed)" >&2
        exit 1
    fi
}

mic() {
    local secs="${1:-30}"
    command -v termux-microphone-record >/dev/null || pkg install -y termux-api
    local raw="/tmp/asr_mic_$$.wav"
    echo ">> recording ${secs}s... (say something)"
    termux-microphone-record -f "$raw" -l "$secs" start
    sleep "$secs"
    termux-microphone-record stop 2>/dev/null || true
    transcribe "$raw"
    rm -f "$raw"
}

dictate() {
    # Record in 3s chunks; stop after N consecutive quiet chunks (or max_secs).
    local max_secs="${1:-120}"
    local quiet_rounds="${2:-2}"
    local chunk="${3:-3}"
    command -v termux-microphone-record >/dev/null || pkg install -y termux-api
    local raw="/tmp/asr_dict_$$.wav"
    local pcm="/tmp/asr_dict_$$.pcm"
    local chunk_file="/tmp/asr_chunk_$$.wav"
    local started quiet lvl now
    started="$(date +%s)"
    quiet=0
    rm -f "$raw" "$pcm"
    echo ">> dictation: keep talking... (stops after ${quiet_rounds}x${chunk}s silence, max ${max_secs}s)"
    while true; do
        now="$(date +%s)"
        [ $((now - started)) -lt "$max_secs" ] || break
        termux-microphone-record stop >/dev/null 2>&1 || true
        termux-microphone-record -f "$chunk_file" -l "$chunk" start
        sleep "$chunk"
        termux-microphone-record stop >/dev/null 2>&1 || true
        [ -f "$chunk_file" ] || continue
        lvl="$(ffmpeg -i "$chunk_file" -af volumedetect -f null - 2>&1 \
               | awk -F: '/mean_volume/ {gsub(/ /, "", $2); print $2}')" || lvl=""
        if [ -n "$lvl" ] && awk -v v="$lvl" 'BEGIN { exit !(v < -45.0) }'; then
            quiet=$((quiet + 1))
        else
            quiet=0
        fi
        ffmpeg -y -i "$chunk_file" -ar 16000 -ac 1 -f s16le >> "$pcm" 2>/dev/null || true
        rm -f "$chunk_file"
        [ "$quiet" -lt "$quiet_rounds" ] || { echo ">> silence detected, stopping." >&2; break; }
    done
    if [ -s "$pcm" ]; then
        ffmpeg -y -f s16le -ar 16000 -ac 1 -i "$pcm" -c:a pcm_s16le "$raw" >/dev/null 2>&1
    fi
    rm -f "$pcm"
    [ -s "$raw" ] || { echo "(nothing recorded)" >&2; exit 1; }
    transcribe "$raw"
    rm -f "$raw"
}

translate() {
    # Translation booth: transcribe locally, then translate with the local
    # androidllm model (needs androidllm-serve running; zero cloud usage).
    local src="$1"
    local target="${2:-English}"
    local txt
    txt="$(transcribe "$src")"
    echo ">> transcript: $txt" >&2
    printf '%s' "$txt" | TARGET="$target" python3 - <<'PY'
import json, os, sys, urllib.request
text = sys.stdin.read().strip()
target = os.environ.get("TARGET", "English")
if not text:
    sys.exit(1)
prompt = ("Translate the following text to " + target + ". "
          "Output only the translation, nothing else.\n\n" + text)
body = json.dumps({"model": "auto",
                   "messages": [{"role": "user", "content": prompt}],
                   "max_tokens": 1024}).encode("utf-8")
req = urllib.request.Request(
    "http://127.0.0.1:8080/v1/chat/completions", data=body,
    headers={"Content-Type": "application/json", "Authorization": "Bearer skip-auth"})
try:
    with urllib.request.urlopen(req, timeout=120) as r:
        data = json.loads(r.read())
        print(data["choices"][0]["message"]["content"])
except Exception as e:
    print(f"(translate failed: {e})", file=sys.stderr)
    sys.exit(1)
PY
}

case "${1:-help}" in
    setup) setup ;;
    transcribe) transcribe "$2" "${3:-auto}" ;;
    mic) mic "$2" ;;
    dictate) dictate "$2" "$3" "$4" ;;
    translate) translate "$2" "$3" ;;
    *) echo "usage: asr.sh setup | transcribe FILE [lang] | mic [secs] | dictate [max_secs] | translate FILE [target]" ;;
esac
