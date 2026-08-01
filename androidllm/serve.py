import argparse
import json
import re
import threading
import time

from .engine import LayerStreamingEngine


def _parse_model_name(engine, name):
    return name or engine.manifest.get("name", "androidllm")


def build_engine(model_dir):
    return LayerStreamingEngine(model_dir)


def _new_id(prefix):
    return "%s-%d" % (prefix, time.time_ns())


def _defaults(body):
    return (body.get("max_tokens", body.get("max_new_tokens", 64)),
            body.get("temperature", 0.8),
            body.get("top_p", 0.9))


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


def _non_stream_response(engine, kind, body):
    max_tokens, temperature, top_p = _defaults(body)
    if kind == "chat":
        ids = _chat_prompt_ids(engine, body.get("messages", []))
        out_ids = engine.generate(ids, max_tokens, temperature, top_p,
                                  stop_ids=engine.stop_ids)
        text = engine.tokenizer.decode(out_ids)
        return {
            "id": _new_id("chatcmpl"),
            "object": "chat.completion",
            "created": int(time.time()),
            "model": _parse_model_name(engine, body.get("model")),
            "choices": [{"index": 0, "message": {"role": "assistant", "content": text},
                         "finish_reason": "stop"}],
            "usage": _usage(ids, out_ids),
        }
    ids = engine.tokenizer.encode(body.get("prompt", ""))
    out_ids = engine.generate(ids, max_tokens, temperature, top_p,
                              stop_ids=engine.stop_ids)
    text = engine.tokenizer.decode(out_ids)
    return {
        "id": _new_id("cmpl"),
        "object": "text_completion",
        "created": int(time.time()),
        "model": _parse_model_name(engine, body.get("model")),
        "choices": [{"index": 0, "text": text, "finish_reason": "stop"}],
        "usage": _usage(ids, out_ids),
    }


def _stream_base(engine, kind, body):
    created = int(time.time())
    name = _parse_model_name(engine, body.get("model"))
    if kind == "chat":
        return {"id": _new_id("chatcmpl"), "object": "chat.completion.chunk",
                "created": created, "model": name}
    return {"id": _new_id("cmpl"), "object": "text_completion",
            "created": created, "model": name}


def _stream(handler, engine, kind, body):
    """Write one SSE frame per generated token (delta-encoded) and a [DONE]."""
    base = _stream_base(engine, kind, body)
    generated = []
    state = {"prev": "", "role": False}

    def write(choice):
        payload = dict(base)
        payload["choices"] = [choice]
        frame = b"data: " + json.dumps(payload, ensure_ascii=False).encode("utf-8") + b"\n\n"
        handler.wfile.write(frame)
        handler.wfile.flush()

    def cb(tok):
        generated.append(tok)
        text = engine.tokenizer.decode(generated)
        delta = text[len(state["prev"]):]
        state["prev"] = text
        if kind == "chat":
            d = {"content": delta}
            if not state["role"]:
                d["role"] = "assistant"
                state["role"] = True
            write({"index": 0, "delta": d, "finish_reason": None})
        else:
            write({"index": 0, "text": delta, "finish_reason": None})

    if kind == "chat":
        ids = _chat_prompt_ids(engine, body.get("messages", []))
    else:
        ids = engine.tokenizer.encode(body.get("prompt", ""))
    engine.generate(ids, *_defaults(body), stop_ids=engine.stop_ids, callback=cb)

    if kind == "chat":
        write({"index": 0, "delta": {}, "finish_reason": "stop"})
    else:
        write({"index": 0, "text": "", "finish_reason": "stop"})


_ROUTES = [
    (re.compile(r"^/v1/completions$"), "completions"),
    (re.compile(r"^/v1/chat/completions$"), "chat"),
    (re.compile(r"^/completions$"), "completions"),
    (re.compile(r"^/chat/completions$"), "chat"),
]


def run_server(engine, host="127.0.0.1", port=8080):
    from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

    q_lock = threading.Lock()

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
            if self.path == "/health":
                self._reply(200, {"status": "ok"})
            elif self.path == "/v1/models":
                self._reply(200, _models_list(engine))
            else:
                self._reply(404, {"error": "not found"})

        def do_POST(self):
            length = int(self.headers.get("Content-Length", 0))
            body = json.loads(self.rfile.read(length).decode("utf-8"))
            for pat, kind in _ROUTES:
                if pat.match(self.path):
                    if engine.tokenizer is None:
                        self._reply(500, {"error": "model has no tokenizer; "
                                                   "convert it with androidllm-shard"})
                        return
                    try:
                        with q_lock:
                            if body.get("stream"):
                                self._begin_sse()
                                try:
                                    _stream(self, engine, kind, body)
                                except Exception as exc:
                                    self.wfile.write(
                                        b"data: " + json.dumps({"error": str(exc)},
                                                               ensure_ascii=False).encode("utf-8")
                                        + b"\n\n")
                                    self.wfile.flush()
                                finally:
                                    self._end_sse()
                            else:
                                self._reply(200, _non_stream_response(engine, kind, body))
                    except Exception as exc:
                        self._reply(500, {"error": str(exc)})
                    return
            self._reply(404, {"error": "not found"})

    httpd = ThreadingHTTPServer((host, port), Handler)
    print("androidllm serving on http://%s:%d" % (host, port))
    try:
        httpd.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        httpd.server_close()


def main():
    ap = argparse.ArgumentParser(description="Serve an androidllm shard")
    ap.add_argument("--model", required=True, help="path to sharded model dir")
    ap.add_argument("--host", default="127.0.0.1")
    ap.add_argument("--port", type=int, default=8080)
    args = ap.parse_args()
    engine = build_engine(args.model)
    run_server(engine, args.host, args.port)


if __name__ == "__main__":
    main()
