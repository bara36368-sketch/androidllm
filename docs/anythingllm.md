# Wiring AnythingLLM to androidllm

AnythingLLM (https://anythingllm.com) is a desktop/document RAG app with a
local-first "bring your own LLM" model. androidllm exposes an OpenAI-compatible
API, so AnythingLLM can treat the phone's local model as its default chat
engine — full offline mode, $0 per token, no cloud dependency.

## What androidllm exposes

| Endpoint | Purpose |
|---|---|
| `GET /health` | Liveness (no auth). |
| `GET /v1/models` | Loaded model + installed (via `/v1/pull`) models. |
| `POST /v1/chat/completions` | Chat completions, streaming (SSE) or one-shot. |
| `POST /v1/completions` | Text completions. |
| `POST /v1/token-count` | Token count of a message list. |
| `GET /v1/keys` | Returns `base_url` + `api_key` for client config. |
| `POST /v1/pull` / `GET /v1/pulls/{id}` | Remote model install (lemonade flow). |
| `GET/POST /v1/preset` | Runtime preset switching (performance/battery/quality). |

Auth: `Authorization: Bearer <ANDROIDLLM_API_KEY>` (enabled once the env var
is set; otherwise open).

## 1. Serve androidllm so the desktop can reach it

By default androidllm binds 127.0.0.1 and is noisy on battery. For AnythingLLM
running on a laptop, expose it on the LAN and (recommended) enable auth:

```bash
ANDROIDLLM_API_KEY=change-me \
ANDROIDLLM_BATCH_MAX=4 \
python -m androidllm.serve --model <shard-dir> \
  --host 0.0.0.0 --port 8080
```

(`--host`/`--port` are CLI flags; `--api-key` also works. Model id served at
`/v1/models` is the shard's manifest `name`, e.g. `smollm2` — read it live
from `GET /v1/models` rather than assuming.)

Find the phone's LAN address (e.g. `192.168.1.5`). Everything below uses
`http://<phone-ip>:8080`.

## 2. Configure AnythingLLM

In AnythingLLM: **Settings → LLM Preference**:

1. **LLM Provider**: `OpenAI` (the generic connector — do NOT pick Azure or
   a vendor-specific one).
2. **Model**: `<manifest name>` (e.g. `smollm2`; read the live id from
   `curl http://<phone-ip>:8080/v1/models`).
3. **API Key**: the `ANDROIDLLM_API_KEY` value (any non-empty string works).
4. **Base URL / Endpoint** (varies by AnythingLLM version):
   - Recent versions: `http://<phone-ip>:8080/v1`
   - Older versions ask for the full chat path: `http://<phone-ip>:8080/v1/chat/completions`
5. Save and run a test chat. AnythingLLM will list androidllm's model in the
   model picker once `/v1/models` responds.

### Embedder (document RAG)

androidllm has no embeddings endpoint, so leave the embedder at the built-in
**AnythingLLM Local** embedder (offline, works out of the box) or any cloud
embedder you already use. Only the chat/completion path goes to the phone.

## 3. What works / what to avoid

**Works:**

- Streaming chat (`stream: true`) — SSE chunks with `data: [DONE]`.
- Document chat with history: messages are packed through the model's chat
  template with KV prefix reuse, so multi-turn is fast and memory-bounded.
- Token counting via `/v1/token-count` (AnythingLLM page-token accounting).
- Offline workspace chat: air-gapped, zero cost.

**Does not work / caveats:**

- **Tool calling / function calling is not implemented** — AnythingLLM
  features that depend on native tool-calling will not work. Use the model
  purely as a chat backend.
- Only **one** model is loaded at a time (the phone holds the model in RAM).
  Switching models means restarting serve with a different `--model`
  (or using `/v1/pull` to install more) — not dynamic per-request routing.
- Responses can be interrupted by phone power management: keep
  `ANDROIDLLM_IDLE_PAUSE` low or 0 while chatting, and don't let the screen
  sleep mid-conversation.
- Long context degrades on the phone: keep `max_tokens` per request modest.

## 4. Quick test without AnythingLLM

```bash
curl -s http://<phone-ip>:8080/v1/models \
  -H "Authorization: Bearer change-me"

curl -s http://<phone-ip>:8080/v1/chat/completions \
  -H "Authorization: Bearer change-me" \
  -H "Content-Type: application/json" \
  -d '{"model":"smollm2","messages":[{"role":"user","content":"hello from anythingllm"}]}' \
  --no-buffer
```

## 5. Tuning

- **`ANDROIDLLM_BATCH_MAX`** — number of concurrent chat slots (default 4).
  AnythingLLM is single-user; 2 is plenty and cooler on the battery.
- **`ANDROIDLLM_PRESET`** (or `--preset`) — `performance` | `balanced` |
  `battery`; the phone advertises battery state on `/stats`, and windowed
  batching keeps generation smooth for chat.
- Set `ANDROIDLLM_API_KEY` even for LAN use — the phone may be on shared Wi-Fi.

Also see: `scripts/setup_termux.sh` in this repo for phone-side setup, and the
`/v1/pull` flow for installing more models over Wi-Fi without ADB.