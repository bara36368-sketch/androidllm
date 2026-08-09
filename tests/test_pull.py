"""Model installer tests (lemonade /v1/pull + LocalAI gallery port).

pull.py: repo_files listing (sizes + LFS sha256), download + checksum
verification, resolve_source (catalog id vs HF repo), pull() end-to-end
with a fake HF server and a monkeypatched shard worker, registry
installed/remove. serve.py: POST /v1/pull + GET /v1/pulls/{id} + the
installed-model entries in /v1/models.
"""
import json
import os
import threading
import time

import pytest

from androidllm import pull
from androidllm import modelpicker as mp


# ------------------------------------------------------------------ fake HF repo

class FakeHF:
    """A tiny HF-compatible server: one repo with config.json + a small
    safetensors + tokenizer files, serving the tree API and raw files."""

    def __init__(self, tmp_path, corrupt_sha=False):
        self.dir = tmp_path / "repo"
        self.dir.mkdir()
        (self.dir / "config.json").write_text(
            json.dumps({"model_type": "qwen2", "hidden_size": 64,
                        "num_attention_heads": 4, "num_key_value_heads": 4,
                        "num_hidden_layers": 1, "vocab_size": 100,
                        "max_position_embeddings": 512,
                        "tie_word_embeddings": False}))
        (self.dir / "model.safetensors").write_bytes(b"\x00" * 4096)
        (self.dir / "tokenizer.json").write_text("{}")
        self.files = sorted(p.name for p in self.dir.iterdir())
        self.sha = {n: pull._sha256(str(self.dir / n))
                    for n in self.files}
        if corrupt_sha:
            self.sha["model.safetensors"] = "0" * 64
        import http.server
        repo_dir = self.dir
        sha_map = self.sha

        class H(http.server.BaseHTTPRequestHandler):
            def log_message(self, *a):
                pass

            def do_GET(self):
                if self.path.startswith("/api/models/"):
                    if not self.path.endswith("/tree/main?recursive=true"):
                        self.send_response(404)
                        self.end_headers()
                        return
                    rows = []
                    for n in repo_dir.iterdir():
                        rows.append({"path": n.name,
                                     "size": n.stat().st_size,
                                     "lfs": {"sha256": sha_map[n.name]}})
                    self._json(rows)
                elif self.path.startswith("/resolve/main/"):
                    name = self.path.rsplit("/", 1)[-1]
                    p = repo_dir / name
                    if not p.exists():
                        self.send_response(404)
                        self.end_headers()
                        return
                    data = p.read_bytes()
                    self.send_response(200)
                    self.send_header("Content-Length", str(len(data)))
                    self.end_headers()
                    self.wfile.write(data)
                else:
                    self.send_response(404)
                    self.end_headers()

            def _json(self, obj):
                data = json.dumps(obj).encode("utf-8")
                self.send_response(200)
                self.send_header("Content-Type", "application/json")
                self.send_header("Content-Length", str(len(data)))
                self.end_headers()
                self.wfile.write(data)

        self.httpd = http.server.ThreadingHTTPServer(("127.0.0.1", 0), H)
        self.port = self.httpd.server_address[1]
        self.thread = threading.Thread(target=self.httpd.serve_forever,
                                       daemon=True)
        self.thread.start()

    def url(self, path):
        return f"http://127.0.0.1:{self.port}{path}"

    def close(self):
        self.httpd.shutdown()


@pytest.fixture
def fake_hf(tmp_path):
    srv = FakeHF(tmp_path)
    yield srv
    srv.close()


# --------------------------------------------------------------- resolve + list

def test_resolve_source_catalog_id():
    mid, repo, entry = pull.resolve_source("qwen15")
    assert mid == "qwen15"
    assert repo == "Qwen/Qwen2.5-1.5B-Instruct"
    assert entry["params_b"] > 0


def test_resolve_source_repo():
    mid, repo, entry = pull.resolve_source("Owner/Model-7B")
    assert mid == "model-7b" and repo == "Owner/Model-7B" and entry is None


def test_resolve_source_unknown():
    assert pull.resolve_source("nope-xyz") is None


def test_repo_files_lists_needed_only(fake_hf):
    with pytest.MonkeyPatch.context() as m:
        m.setattr(pull, "_HF_TREE", fake_hf.url("/api/models/{repo}/tree/main?recursive=true"))
        m.setattr(pull, "_HF_RAW", fake_hf.url("/resolve/main/{path}"))
        files = pull.repo_files("Fake/Repo")
        assert {f["path"] for f in files} == set(fake_hf.files)
        for f in files:
            assert f["size"] > 0 and f["sha256"]


def test_repo_files_uses_lfs_oid_when_sha256_missing(monkeypatch):
    rows = [{"path": "model.safetensors", "size": 5,
             "lfs": {"oid": "a" * 64}}]
    import urllib.request
    monkeypatch.setattr(pull, "_hf_json", lambda url: rows)
    files = pull.repo_files("Fake/Repo")
    assert files[0]["sha256"] == "a" * 64


# --------------------------------------------------------------- download + sha

def test_download_verifies_sha(tmp_path, fake_hf):
    with pytest.MonkeyPatch.context() as m:
        m.setattr(pull, "_HF_RAW", fake_hf.url("/resolve/main/{path}"))
        dest = str(tmp_path / "model.safetensors")
        pull._download(fake_hf.url("/resolve/main/model.safetensors"),
                       dest, fake_hf.sha["model.safetensors"])
        assert os.path.getsize(dest) == 4096


def test_download_rejects_bad_sha(tmp_path, fake_hf):
    with pytest.MonkeyPatch.context() as m:
        m.setattr(pull, "_HF_RAW", fake_hf.url("/resolve/main/{path}"))
        dest = str(tmp_path / "model.safetensors")
        with pytest.raises(ValueError, match="sha256 mismatch"):
            pull._download(fake_hf.url("/resolve/main/model.safetensors"),
                           dest, "0" * 64)
        assert not os.path.exists(dest)  # poisoned file removed


def test_download_skips_verified_existing(tmp_path, fake_hf):
    with pytest.MonkeyPatch.context() as m:
        m.setattr(pull, "_HF_RAW", fake_hf.url("/resolve/main/{path}"))
        dest = str(tmp_path / "model.safetensors")
        with open(dest, "wb") as f:
            f.write(b"\x00" * 4096)
        assert pull._download(fake_hf.url("/resolve/main/model.safetensors"),
                              dest, fake_hf.sha["model.safetensors"]) is True
        assert os.path.getsize(dest) == 4096  # unchanged, no re-download


# --------------------------------------------------------------- pull end-to-end

def _fake_shard(staging, out, attn_bits=4, mlp_bits=8, embed_bits=8):
    os.makedirs(out)
    with open(os.path.join(out, "manifest.json"), "w") as f:
        f.write("{}")
    with open(os.path.join(out, "layer_0.safetensors"), "wb") as f:
        f.write(b"\x00" * 8)
    with open(os.path.join(out, "embeddings.safetensors"), "wb") as f:
        f.write(b"\x00" * 8)


def test_pull_installs_and_registers(tmp_path, fake_hf, monkeypatch):
    monkeypatch.setattr(pull, "_HF_TREE",
                        fake_hf.url("/api/models/{repo}/tree/main?recursive=true"))
    monkeypatch.setattr(pull, "_HF_RAW", fake_hf.url("/resolve/main/{path}"))
    monkeypatch.setattr(pull, "MODELS_ROOT", str(tmp_path / "models"))
    monkeypatch.setattr(pull, "INDEX_PATH", str(tmp_path / "models" / "installed.json"))
    monkeypatch.setattr(pull, "_models_root", lambda: pull.MODELS_ROOT)

    conf = pull.pull("qwen15", shard=_fake_shard)
    assert conf["id"] == "qwen15"
    assert conf["repo"] == "Qwen/Qwen2.5-1.5B-Instruct"
    assert conf["ctx"] == 512
    d = tmp_path / "models" / "qwen15"
    assert d.is_dir() and (d / "model.conf.json").exists()
    idx = json.loads((tmp_path / "models" / "installed.json").read_text())
    assert "qwen15" in idx and idx["qwen15"]["repo"] == conf["repo"]
    # staging dir cleaned up
    assert not list((tmp_path / "models").glob(".tmp-*"))


def test_pull_rejects_already_installed(tmp_path, fake_hf, monkeypatch):
    monkeypatch.setattr(pull, "MODELS_ROOT", str(tmp_path / "models"))
    monkeypatch.setattr(pull, "INDEX_PATH", str(tmp_path / "models" / "installed.json"))
    monkeypatch.setattr(pull, "_models_root", lambda: pull.MODELS_ROOT)
    monkeypatch.setattr(pull, "_HF_TREE",
                        fake_hf.url("/api/models/{repo}/tree/main?recursive=true"))
    monkeypatch.setattr(pull, "_HF_RAW", fake_hf.url("/resolve/main/{path}"))
    pull.pull("qwen15", shard=_fake_shard)
    with pytest.raises(ValueError, match="already installed"):
        pull.pull("qwen15", shard=_fake_shard)


def test_pull_bad_sha_fails_and_cleans(tmp_path, monkeypatch):
    srv = FakeHF(tmp_path, corrupt_sha=True)
    try:
        monkeypatch.setattr(pull, "_HF_TREE",
                            srv.url("/api/models/{repo}/tree/main?recursive=true"))
        monkeypatch.setattr(pull, "_HF_RAW", srv.url("/resolve/main/{path}"))
        monkeypatch.setattr(pull, "MODELS_ROOT", str(tmp_path / "models"))
        monkeypatch.setattr(pull, "INDEX_PATH", str(tmp_path / "models" / "installed.json"))
        monkeypatch.setattr(pull, "_models_root", lambda: pull.MODELS_ROOT)
        with pytest.raises(ValueError, match="sha256 mismatch"):
            pull.pull("qwen15")
        assert not (tmp_path / "models" / "qwen15").exists()
        assert not list((tmp_path / "models").glob(".tmp-*"))
    finally:
        srv.close()


def test_remove_unregisters(tmp_path, monkeypatch):
    monkeypatch.setattr(pull, "MODELS_ROOT", str(tmp_path / "models"))
    monkeypatch.setattr(pull, "INDEX_PATH", str(tmp_path / "models" / "installed.json"))
    (tmp_path / "models").mkdir()
    (tmp_path / "models" / "qwen15").mkdir()
    pull._save_index({"qwen15": {"repo": "x"}})
    pull.remove("qwen15")
    assert pull.installed_models() == {}
    with pytest.raises(ValueError, match="not installed"):
        pull.remove("qwen15")


# --------------------------------------------------------------- serve endpoints

def test_pull_http_endpoints(tmp_path, monkeypatch):
    import urllib.request
    import urllib.error
    from androidllm import serve

    class FakeEngine:
        tokenizer = None
        paused = False
        ctx_len = 2048
        canon = {"layers": 4, "kv_heads": 4, "head_dim": 32}
        draft = None
        spec_k = 0
        manifest = {"name": "qwen15"}

        def snapshot(self):
            return {"uptime_s": 0, "tokens": 0}

    calls = {}

    def fake_pull(ident, attn_bits=4, mlp_bits=8, embed_bits=8, progress=None):
        calls["ident"] = ident
        if progress:
            progress(0, 0, "model.safetensors")
        time.sleep(0.1)
        return {"id": "qwen15", "repo": "Qwen/Qwen2.5-1.5B-Instruct",
                "size_gb": 1.1}

    def fake_installed_list():
        return [{"id": "qwen15", "repo": "Qwen/Qwen2.5-1.5B-Instruct",
                 "installed_ts": 1234, "dir": "/m", "size_gb": 1.1}]

    monkeypatch.setattr(serve, "_load_api_key", lambda: ("", False))
    monkeypatch.setattr(pull, "pull", fake_pull)
    monkeypatch.setattr(pull, "installed_list", fake_installed_list)
    port = 8272
    t = threading.Thread(target=serve.run_server,
                         args=(FakeEngine(), "127.0.0.1", port), daemon=True)
    t.start()
    time.sleep(1.5)

    def http(method, path, body=None):
        req = urllib.request.Request(f"http://127.0.0.1:{port}{path}",
                                     method=method)
        if body is not None:
            req.add_header("Content-Type", "application/json")
            req.data = json.dumps(body).encode("utf-8")
        try:
            with urllib.request.urlopen(req, timeout=5) as r:
                return r.status, json.loads(r.read().decode("utf-8"))
        except urllib.error.HTTPError as e:
            return e.code, json.loads(e.read().decode("utf-8"))

    status, started = http("POST", "/v1/pull", {"model": "qwen15"})
    assert status == 202
    assert started["status"] == "started"
    jid = started["id"]

    status, job = None, None
    for _ in range(20):
        status, job = http("GET", f"/v1/pulls/{jid}")
        if job["status"] != "started":
            break
        time.sleep(0.1)
    assert status == 200 and job["status"] == "done"
    assert calls.get("ident") == "qwen15"
    assert job["model"] == "qwen15"

    status, missing = http("GET", "/v1/pulls/nope")
    assert status == 404

    status, empty = http("POST", "/v1/pull", {"model": "  "})
    assert status == 400

    status, models = http("GET", "/v1/models")
    assert status == 200
    ids = [m["id"] for m in models["data"]]
    assert ids[0] == "qwen15" and models["data"][0]["status"] == "loaded"
    installed = next(m for m in models["data"] if m["status"] == "installed")
    assert installed["id"] == "qwen15" and installed["size_gb"] == 1.1
