import argparse
import json
import os
import re
import secrets
import threading
import time

from .engine import LayerStreamingEngine
from .json_grammar import JsonGrammar
from .batching import BatchScheduler, SessionPool
from . import neon


def _api_key_path():
    """Where the auto-generated API key lives (~/.androidllm/api_key)."""
    base = os.environ.get("ANDROIDLLM_DIR", os.path.expanduser("~/androidllm"))
    return os.path.join(base, "api_key")


def _load_api_key():
    """Resolve the server API key:
    1. --api-key flag (handled by caller via ANDROIDLLM_API_KEY env)
    2. ANDROIDLLM_API_KEY env
    3. persisted key file (survives restarts)
    4. generate a fresh random key, persist it, and return it.
    Returns (key, generated) where generated tells the caller to log it."""
    key = os.environ.get("ANDROIDLLM_API_KEY", "").strip()
    if key:
        return key, False
    p = _api_key_path()
    try:
        with open(p, encoding="utf-8") as f:
            key = f.read().strip()
        if len(key) >= 16:
            return key, False
    except OSError:
        pass
    key = "sk-androidllm-" + secrets.token_hex(24)
    try:
        os.makedirs(os.path.dirname(p), exist_ok=True)
        with open(p, "w", encoding="utf-8") as f:
            f.write(key + "\n")
    except OSError:
        pass
    return key, True


def _check_auth(handler, key):
    """Validate the Authorization header against the server key.
    Returns True if authorized; otherwise writes a 401 and returns False."""
    if not key:
        return True
    header = handler.headers.get("Authorization", "")
    if header.startswith("Bearer ") and secrets.compare_digest(
            header[len("Bearer "):].strip(), key):
        return True
    handler._reply(401, {"error": "invalid api key",
                         "message": "provide a valid api key via "
                                    "'Authorization: Bearer <key>'"})
    return False


def _parse_model_name(engine, name):
    return name or engine.manifest.get("name", "androidllm")


def _draft_dir():
    """ANDROIDLLM_DRAFT = model dir or a model id under <ANDROIDLLM_DIR>/models."""
    raw = os.environ.get("ANDROIDLLM_DRAFT", "").strip()
    if not raw:
        return None
    if os.path.isdir(raw):
        return raw
    base = os.environ.get("ANDROIDLLM_DIR", os.path.expanduser("~/androidllm"))
    cand = os.path.join(base, "models", raw)
    return cand if os.path.isdir(cand) else None


def build_engine(model_dir):
    keep = int(os.environ.get("ANDROIDLLM_KEEP_LAYERS", "0"))
    draft = _draft_dir()
    spec_k = int(os.environ.get("ANDROIDLLM_SPEC_K", "0"))
    engine = LayerStreamingEngine(model_dir, keep_layers=keep,
                                  draft_dir=draft, spec_k=spec_k)
    if engine.draft is None and spec_k > 0 and draft:
        print("warning: draft model dir missing (%s) - running without "
              "speculative decoding" % draft)
    if keep > 0 or engine._lru > 0:
        def warm():
            try:
                for i in range(min(keep, engine.n_layers)):
                    engine.load_layer(i)
            except Exception:
                pass
        threading.Thread(target=warm, daemon=True).start()
    return engine


# -- battery-aware speed policy -------------------------------------------

_BATTERY_DIRS = (
    "/sys/class/power_supply/battery",
    "/sys/class/power_supply/BAT0",
)


def _sysfs(name):
    for d in _BATTERY_DIRS:
        try:
            with open(os.path.join(d, name), encoding="utf-8") as f:
                return f.read().strip()
        except OSError:
            continue
    return None


def battery_info():
    status = _sysfs("status")
    cap = _sysfs("capacity")
    temp = _sysfs("temp")
    return {
        "charging": bool(status and status.lower().startswith(("charging", "full"))),
        "capacity": int(cap) if cap and cap.isdigit() else None,
        "temp_c": (int(temp) / 10.0) if temp and temp.isdigit() else None,
    }


def pick_threads(info, base):
    """Charging -> base threads; battery under 50% -> half; low -> 1.
    Hot battery caps too."""
    n = base
    if info["capacity"] is not None:
        if info["capacity"] <= 15:
            n = 1
        elif info["capacity"] <= 50 and not info["charging"]:
            n = max(1, base // 2)
    if info["temp_c"] is not None and info["temp_c"] >= 45:
        n = min(n, 2)
    return n


# Battery percent at which serving pauses entirely (0 disables).
# Requests get a 503 while paused; charging or crossing the threshold
# resumes automatically. Throttle kicks in well before the pause.
_PAUSE_PCT = int(os.environ.get("ANDROIDLLM_BATTERY_PAUSE", "15"))


def start_battery_policy(engine, interval=30):
    def loop():
        while True:
            time.sleep(interval)
            try:
                info = battery_info()
                charging = info["charging"]
                cap = info["capacity"]
                engine.paused = (_PAUSE_PCT > 0 and cap is not None
                                 and cap <= _PAUSE_PCT and not charging)
                if charging:
                    engine.throttle_ms = 0
                else:
                    engine.throttle_ms = 20 if (cap is not None and cap <= 15) else 0
                neon.set_threads(pick_threads(info, _BASE_THREADS))
            except Exception:
                pass
    threading.Thread(target=loop, daemon=True).start()


_BASE_THREADS = int(os.environ.get("ANDROIDLLM_THREADS", "4"))


# -- reasoning-tag stripping ----------------------------------------------
# Qwen3-style models emit <think>...</think> blocks. The chat API consumer
# (the deck bot) wants the answer only; strip them unless disabled.

_STRIP_THINK = os.environ.get("ANDROIDLLM_STRIP_THINK", "1") not in ("0", "false", "")


def _think_strip(text):
    if not _STRIP_THINK:
        return text
    out = re.sub(r"<think>.*?</think>", "", text, flags=re.S)
    out = re.sub(r"\n{3,}", "\n\n", out)
    return out.strip()


class _ThinkStripper:
    """Incremental <think>...</think> removal for SSE deltas.
    Holds back the last 8 chars in case a tag is split across tokens."""

    def __init__(self):
        self.inside = False
        self.buf = ""

    def feed(self, text):
        if not _STRIP_THINK:
            return text
        self.buf += text
        if self.inside:
            j = self.buf.find("</think>")
            if j == -1:
                k = min(len(self.buf), len("</think>"))
                self.buf = self.buf[-k:] if k else ""
                return ""
            self.buf = self.buf[j + len("</think>"):]
            self.inside = False
            return self.feed("")
        j = self.buf.find("<think>")
        if j == -1:
            k = min(len(self.buf), len("<think>"))
            head, self.buf = self.buf[:-k], self.buf[-k:] if k else ""
            return head
        head, rest = self.buf[:j], self.buf[j + len("<think>"):]
        self.buf = rest
        self.inside = True
        return head + self.feed("")

    def flush(self):
        """End of stream: emit any held-back tail (drops an unclosed tag)."""
        if self.inside:
            self.buf = ""
            self.inside = False
            return ""
        out, self.buf = self.buf, ""
        return out


# -- request handling -----------------------------------------------------

def _new_id(prefix):
    return "%s-%d" % (prefix, time.time_ns())


def _defaults(body):
    return (body.get("max_tokens", body.get("max_new_tokens", 64)),
            body.get("temperature", 0.8),
            body.get("top_p", 0.9),
            float(body.get("min_p", os.environ.get("ANDROIDLLM_MIN_P", "0.0"))))


def _usage(prompt_ids, out_ids):
    return {"prompt_tokens": len(prompt_ids), "completion_tokens": len(out_ids),
            "total_tokens": len(prompt_ids) + len(out_ids)}


def _chat_prompt_ids(engine, messages):
    prompt = engine.tokenizer.apply_template(messages, add_generation_prompt=True)
    return engine.tokenizer.encode(prompt)


def _models_list(engine):
    return {
        "object": "list",
        "data": [{"id": _parse_model_name(engine, None), "object": "model",
                  "created": int(time.time()), "owned_by": "androidllm"}],
    }


def _grammar(body):
    g = body.get("grammar") or body.get("json_schema")
    if not g:
        return None
    return JsonGrammar(g)


def _stream_base(engine, kind, body):
    created = int(time.time())
    name = _parse_model_name(engine, body.get("model"))
    if kind == "chat":
        return {"id": _new_id("chatcmpl"), "object": "chat.completion.chunk",
                "created": created, "model": name}
    return {"id": _new_id("cmpl"), "object": "text_completion",
            "created": created, "model": name}


_ROUTES = [
    (re.compile(r"^/v1/completions$"), "completions"),
    (re.compile(r"^/v1/chat/completions$"), "chat"),
    (re.compile(r"^/completions$"), "completions"),
    (re.compile(r"^/chat/completions$"), "chat"),
]


def run_server(engine, host="127.0.0.1", port=8080):
    from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

    api_key, key_generated = _load_api_key()
    if api_key and key_generated:
        print(">> generated new API key: %s" % api_key)
        print(">> saved to %s (reuse it as ANDROIDLLM_API_KEY)" % _api_key_path())
    elif api_key:
        print(">> API key auth enabled (ANDROIDLLM_API_KEY)")

    scheduler = BatchScheduler(engine,
                               max_slots=int(os.environ.get("ANDROIDLLM_BATCH_MAX", "4")))
    pool = SessionPool(engine, max_pooled=4)
    idle_since = time.time()
    idle_pause = int(os.environ.get("ANDROIDLLM_IDLE_PAUSE", "0"))

    def _finish(sess):
        pool.put(sess)

    class Handler(BaseHTTPRequestHandler):
        protocol_version = "HTTP/1.1"

        def log_message(self, fmt, *args):
            pass

        def _reply(self, code, obj):
            data = json.dumps(obj, ensure_ascii=False).encode("utf-8")
            self.send_response(code)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(data)))
            self.end_headers()
            self.wfile.write(data)

        def _begin_sse(self):
            self.send_response(200)
            self.send_header("Content-Type", "text/event-stream")
            self.send_header("Cache-Control", "no-cache")
            self.send_header("Connection", "close")
            self.end_headers()

        def _end_sse(self):
            self.wfile.write(b"data: [DONE]\n\n")
            self.wfile.flush()

        def do_GET(self):
            nonlocal idle_since
            if self.path == "/health":
                idle_since = time.time()
                self._reply(200, {"status": "ok"})
            elif self.path in ("/v1/models", "/stats", "/v1/keys") or self.path.startswith("/v1/"):
                if not _check_auth(self, api_key):
                    return
                idle_since = time.time()
                if self.path == "/v1/models":
                    self._reply(200, _models_list(engine))
                elif self.path == "/v1/keys":
                    self._reply(200, {"api_key": api_key,
                                      "base_url": "http://%s:%d/v1" % (host, port)})
                else:
                    info = battery_info()
                    snap = dict(engine.snapshot(), battery=info)
                    snap["strip_think"] = _STRIP_THINK
                    snap["pause_pct"] = _PAUSE_PCT
                    snap["batch"] = scheduler.snapshot()
                    snap["pooled"] = len(pool)
                    self._reply(200, snap)
            else:
                self._reply(404, {"error": "not found"})

        def _submit(self, body, kind, stream):
            """Acquire a session (with KV prefix reuse), submit to the
            scheduler, stream or return on completion."""
            nonlocal idle_since
            idle_since = time.time()
            if kind == "chat":
                ids = _chat_prompt_ids(engine, body.get("messages", []))
            else:
                ids = engine.tokenizer.encode(body.get("prompt", ""))
            sess, pfx = pool.acquire(ids)
            if pfx > 1:
                engine.stats["prefix_calls"] += 1
                engine.stats["prefix_tokens"] += pfx
            max_tokens, temperature, top_p, min_p = _defaults(body)
            sess.max_new_tokens = max_tokens
            sess.temperature = temperature
            sess.top_p = top_p
            sess.min_p = min_p
            sess.stop_ids = engine.stop_ids
            sess.grammar = _grammar(body)
            done = threading.Event()
            outcome = {}

            def on_done(s):
                outcome["error"] = s.error
                done.set()

            if not scheduler.submit(sess, on_done):
                pool.put(sess)
                return None, "busy"
            if not stream:
                done.wait()
                pool.put(sess)
                return sess, None

            # SSE: frames written from the scheduler thread; the handler
            # thread just waits, then finalizes.
            write_lock = threading.Lock()
            base = _stream_base(engine, kind, body)
            generated = []
            state = {"prev": "", "role": False}
            strip = _ThinkStripper()
            final = {}

            def emit(delta):
                if not delta:
                    return
                if kind == "chat":
                    d = {"content": delta}
                    if not state["role"]:
                        d["role"] = "assistant"
                        state["role"] = True
                    write({"index": 0, "delta": d, "finish_reason": None})
                else:
                    write({"index": 0, "text": delta, "finish_reason": None})

            def write(choice):
                payload = dict(base)
                payload["choices"] = [choice]
                frame = b"data: " + json.dumps(payload, ensure_ascii=False).encode("utf-8") + b"\n\n"
                with write_lock:
                    self.wfile.write(frame)
                    self.wfile.flush()

            def cb(tok):
                generated.append(tok)
                text = engine.tokenizer.decode(generated)
                raw_delta = text[len(state["prev"]):]
                state["prev"] = text
                emit(strip.feed(raw_delta))

            sess_step = sess.step
            def step_wrapper():
                finished, toks = sess_step()
                for t in toks:
                    cb(t)
                return finished, toks
            sess.step = step_wrapper
            done.wait()
            emit(strip.flush())
            if kind == "chat":
                write({"index": 0, "delta": {}, "finish_reason": "stop"})
            else:
                write({"index": 0, "text": "", "finish_reason": "stop"})
            pool.put(sess)
            return sess, None

        def do_POST(self):
            nonlocal idle_since
            if not _check_auth(self, api_key):
                return
            length = int(self.headers.get("Content-Length", 0))
            body = json.loads(self.rfile.read(length).decode("utf-8"))
            for pat, kind in _ROUTES:
                if pat.match(self.path):
                    if getattr(engine, "paused", False):
                        info = battery_info()
                        cap = info.get("capacity")
                        self._reply(503, {"error": "battery low - serving paused",
                                          "battery": info})
                        return
                    if engine.tokenizer is None:
                        self._reply(500, {"error": "model has no tokenizer; "
                                                   "convert it with androidllm-shard"})
                        return
                    stream = bool(body.get("stream"))
                    try:
                        if stream:
                            self._begin_sse()
                        sess, err = self._submit(body, kind, stream)
                        if sess is None:
                            if stream:
                                self.wfile.write(
                                    b"data: " + json.dumps(
                                        {"error": "scheduler busy",
                                         "batch": scheduler.snapshot()},
                                        ensure_ascii=False).encode("utf-8") + b"\n\n")
                                self.wfile.flush()
                                self._end_sse()
                            else:
                                self._reply(503, {"error": "scheduler busy",
                                                  "batch": scheduler.snapshot()})
                            return
                        if stream:
                            self._end_sse()
                            return
                        if sess.error:
                            self._reply(500, {"error": sess.error})
                            return
                        text = _think_strip(engine.tokenizer.decode(sess.generated))
                        out_ids = sess.generated
                        if kind == "chat":
                            self._reply(200, {
                                "id": _new_id("chatcmpl"),
                                "object": "chat.completion",
                                "created": int(time.time()),
                                "model": _parse_model_name(engine, body.get("model")),
                                "choices": [{"index": 0,
                                             "message": {"role": "assistant",
                                                         "content": text},
                                             "finish_reason": "stop"}],
                                "usage": _usage(
                                    sess.prompt_ids[:sess.pos + 1], out_ids),
                            })
                        else:
                            self._reply(200, {
                                "id": _new_id("cmpl"),
                                "object": "text_completion",
                                "created": int(time.time()),
                                "model": _parse_model_name(engine, body.get("model")),
                                "choices": [{"index": 0, "text": text,
                                             "finish_reason": "stop"}],
                                "usage": _usage(
                                    sess.prompt_ids[:sess.pos + 1], out_ids),
                            })
                    except Exception as exc:
                        if stream:
                            self.wfile.write(
                                b"data: " + json.dumps({"error": str(exc)},
                                                       ensure_ascii=False).encode("utf-8")
                                + b"\n\n")
                            self.wfile.flush()
                            self._end_sse()
                        else:
                            self._reply(500, {"error": str(exc)})
                    return
            self._reply(404, {"error": "not found"})

    httpd = ThreadingHTTPServer((host, port), Handler)
    print("androidllm serving on http://%s:%d" % (host, port))

    def idle_watch():
        while True:
            time.sleep(30)
            if idle_pause > 0 and time.time() - idle_since > idle_pause:
                print("idle for %ds - pausing serve (runner will restart on demand)"
                      % idle_pause)
                os._exit(0)

    threading.Thread(target=idle_watch, daemon=True).start()
    start_battery_policy(engine)

    try:
        httpd.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        httpd.server_close()
        scheduler.close()


def main():
    ap = argparse.ArgumentParser(description="Serve an androidllm shard")
    ap.add_argument("--model", required=True, help="path to sharded model dir")
    ap.add_argument("--host", default="127.0.0.1")
    ap.add_argument("--port", type=int, default=8080)
    ap.add_argument("--api-key", default=os.environ.get("ANDROIDLLM_API_KEY", ""),
                    help="require this key via Authorization: Bearer; "
                         "defaults to a random key generated on first run")
    args = ap.parse_args()
    if args.api_key:
        os.environ["ANDROIDLLM_API_KEY"] = args.api_key
    engine = build_engine(args.model)
    run_server(engine, args.host, args.port)


if __name__ == "__main__":
    main()
