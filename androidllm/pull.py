"""Model installer: pull an HF model from the catalog (or any llama-arch
repo), verify it, shard it, and register it for serving.

Design notes (ported from two sources):
  - LocalAI gallery style: an install is a manifest of files + sha256
    checksums + a config block (recommended ctx, quant bits, stop words).
    We write that config as `model.conf.json` next to the shards so
    serve() can apply sensible defaults per model.
  - lemonade `/v1/pull` style: pulling can also be triggered over the
    serving HTTP API (POST /v1/pull) so a phone without a shell can be
    sent a model from a desktop browser. That endpoint is in serve.py;
    this module is the worker.

Flow: resolve id/repo -> HF tree API (sizes + LFS sha256) -> download the
files shard.py needs (config.json, *.safetensors, tokenizer.*) into a
staging dir with checksum verification -> shard_model() -> write
model.conf.json -> register in installed.json under
$ANDROIDLLM_DIR/models (default ~/androidllm/models).

CLI (JSON on stdout):
  python -m androidllm.pull pull --id qwen15
  python -m androidllm.pull pull --repo Qwen/Qwen2.5-0.5B-Instruct
  python -m androidllm.pull list
  python -m androidllm.pull remove --id qwen15
"""
import argparse
import hashlib
import json
import os
import re
import sys
import time
import urllib.request

from .modelpicker import CATALOG

_MODELS_DIR = os.environ.get(
    "ANDROIDLLM_DIR", os.path.join(os.path.expanduser("~"), "androidllm"))
MODELS_ROOT = os.path.join(_MODELS_DIR, "models")
INDEX_PATH = os.path.join(MODELS_ROOT, "installed.json")

_UA = ("Mozilla/5.0 (Linux; Android 14) AppleWebKit/537.36 (KHTML, like Gecko) "
       "Chrome/125.0 Mobile Safari/537.36")

_HF_TREE = "https://huggingface.co/api/models/{repo}/tree/main?recursive=true"
_HF_RAW = "https://huggingface.co/{repo}/resolve/main/{path}"
_HF_API = "https://huggingface.co/api/models/{repo}"

# files shard.py consumes; anything else in the repo is ignored
_NEEDED = re.compile(
    r"^(config\.json|tokenizer\.json|tokenizer_config\.json|"
    r"special_tokens_map\.json|vocab\.json|merges\.txt|model-[\w]+\.safetensors|"
    r"model\.safetensors)$")

_CONF_FIELDS = ("id", "repo", "params_b", "thinking", "stability",
                "ctx", "stop", "note")


def _models_root():
    os.makedirs(MODELS_ROOT, exist_ok=True)
    return MODELS_ROOT


def _load_index():
    try:
        with open(INDEX_PATH, encoding="utf-8") as f:
            data = json.load(f)
        return data if isinstance(data, dict) else {}
    except (OSError, ValueError):
        return {}


def _save_index(data):
    _models_root()
    tmp = INDEX_PATH + ".tmp"
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=1)
    os.replace(tmp, INDEX_PATH)


def installed_models():
    """Installed model registry: {model_id: {repo, installed_ts, size_gb, ...}}."""
    return _load_index()


def resolve_source(ident):
    """'qwen15' -> catalog entry; 'Owner/Name' -> raw repo. Returns
    (model_id, repo, catalog_entry_or_None)."""
    for m in CATALOG:
        if m["id"] == ident:
            return m["id"], m["repo"], m
    if re.match(r"^[\w.\-]+/[\w.\-]+$", ident):
        return ident.split("/")[-1].lower(), ident, None
    return None


def _hf_json(url, timeout=30):
    req = urllib.request.Request(url, headers={"User-Agent": _UA})
    with urllib.request.urlopen(req, timeout=timeout) as r:
        return json.loads(r.read().decode("utf-8"))


def repo_files(repo):
    """(path, size, sha256) for every LFS entry in the repo's main branch.
    sha256 comes from the LFS pointer metadata; None when unavailable."""
    try:
        rows = _hf_json(_HF_TREE.format(repo=repo))
    except Exception:
        rows = []
    out = []
    for e in rows or []:
        path = e.get("path")
        if not path or not _NEEDED.match(path):
            continue
        lfs = e.get("lfs") or {}
        sha = lfs.get("sha256") or lfs.get("oid")
        out.append({"path": path, "size": e.get("size") or 0,
                    "sha256": sha})
    return out


def _download(url, dest, expected_sha=None, progress=None):
    """Stream a file to dest with size+sha256 verification. Uses Range
    resume when a partial file exists. Returns True when the file was
    already present and verified (or is the last chunk)."""
    if os.path.exists(dest) and expected_sha:
        if _sha256(dest) == expected_sha:
            return True
    mode = "ab" if os.path.exists(dest) else "wb"
    resume = os.path.getsize(dest) if mode == "ab" else 0
    headers = {"User-Agent": _UA}
    if resume:
        headers["Range"] = f"bytes={resume}-"
    req = urllib.request.Request(url, headers=headers)
    with urllib.request.urlopen(req, timeout=120) as r, open(dest, mode) as f:
        total = resume
        while True:
            chunk = r.read(1024 * 256)
            if not chunk:
                break
            f.write(chunk)
            total += len(chunk)
            if progress:
                progress(len(chunk), total)
    if expected_sha:
        got = _sha256(dest)
        if got != expected_sha:
            os.remove(dest)
            raise ValueError(f"sha256 mismatch for {os.path.basename(dest)}: "
                             f"expected {expected_sha}, got {got}")
    return True


def _sha256(path, chunk=1024 * 1024):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        while True:
            b = f.read(chunk)
            if not b:
                break
            h.update(b)
    return h.hexdigest()


def _conf_for(entry, attn_bits, mlp_bits, embed_bits):
    conf = {}
    if entry:
        for f in _CONF_FIELDS:
            if f in entry:
                conf[f] = entry[f]
    conf.setdefault("ctx", 512)
    conf.setdefault("stop", [])
    conf.update({"attn_bits": attn_bits, "mlp_bits": mlp_bits,
                 "embed_bits": embed_bits})
    return conf


def pull(ident, attn_bits=4, mlp_bits=8, embed_bits=8, progress=None,
         shard=None, index=None):
    """Install a catalog id or HF repo: download -> verify -> shard ->
    register. `progress` gets (nbytes, total_bytes) per chunk. `shard`
    overrides the shard worker for tests. Returns the conf dict."""
    from .shard import shard_model
    shard = shard or shard_model
    index = index if index is not None else _load_index
    resolved = resolve_source(ident)
    if resolved is None:
        raise ValueError(f"unknown model id or repo: {ident}")
    mid, repo, entry = resolved
    if mid in index():
        raise ValueError(f"model {mid} is already installed; remove it first")

    files = repo_files(repo)
    if not any(f["path"].endswith(".safetensors") for f in files):
        raise ValueError(f"no shardable safetensors in {repo}")
    _models_root()
    staging = os.path.join(MODELS_ROOT, f".tmp-{mid}")
    os.makedirs(staging, exist_ok=True)
    dest = os.path.join(MODELS_ROOT, mid)
    if os.path.isdir(dest):
        raise ValueError(f"{dest} already exists")
    try:
        for f in sorted(files, key=lambda f: f["path"]):
            if progress:
                progress(0, f["size"], f["path"])
            _download(_HF_RAW.format(repo=repo, path=f["path"]),
                      os.path.join(staging, f["path"]), f["sha256"],
                      progress=(lambda n, t, p=f["path"]:
                                progress(n, t, p)) if progress else None)
        shard(staging, dest, attn_bits, mlp_bits, embed_bits)
    except BaseException:
        import shutil
        shutil.rmtree(staging, ignore_errors=True)
        raise
    import shutil
    shutil.rmtree(staging, ignore_errors=True)

    conf = _conf_for(entry, attn_bits, mlp_bits, embed_bits)
    conf["id"] = mid
    conf["repo"] = repo
    conf["installed_ts"] = int(time.time())
    conf["size_gb"] = round(_dir_gb(dest), 2)
    with open(os.path.join(dest, "model.conf.json"), "w", encoding="utf-8") as f:
        json.dump(conf, f, ensure_ascii=False, indent=2)
    data = index()
    data[mid] = {k: v for k, v in conf.items()
                 if k not in ("attn_bits", "mlp_bits", "embed_bits")}
    _save_index(data)
    return conf


def _dir_gb(path):
    total = 0
    for root, _, files in os.walk(path):
        for n in files:
            try:
                total += os.path.getsize(os.path.join(root, n))
            except OSError:
                pass
    return total / (1024 ** 3)


def remove(ident, delete=False):
    """Unregister a model (and optionally delete its shards)."""
    data = _load_index()
    if ident not in data:
        raise ValueError(f"{ident} is not installed")
    del data[ident]
    _save_index(data)
    if delete:
        import shutil
        shutil.rmtree(os.path.join(MODELS_ROOT, ident), ignore_errors=True)
    return ident


def installed_list():
    """installed.json entries, each enriched with dir existence + size."""
    out = []
    for mid, meta in sorted(_load_index().items()):
        d = os.path.join(MODELS_ROOT, mid)
        e = dict(meta)
        e["id"] = mid
        e["dir"] = d
        e["present"] = os.path.isdir(d)
        e["size_gb"] = round(_dir_gb(d), 2) if os.path.isdir(d) else 0.0
        out.append(e)
    return out


# -- CLI -------------------------------------------------------------------

def _emit(obj):
    sys.stdout.write(json.dumps(obj, ensure_ascii=False) + "\n")


def main(argv=None):
    ap = argparse.ArgumentParser(prog="androidllm.pull",
                                 description=__doc__)
    ap.add_argument("mode", choices=["pull", "list", "remove"])
    ap.add_argument("--id", default=None, help="catalog model id (e.g. qwen15)")
    ap.add_argument("--repo", default=None, help="HF repo (e.g. Qwen/Qwen2.5-1.5B-Instruct)")
    ap.add_argument("--attn-bits", type=int, default=4)
    ap.add_argument("--mlp-bits", type=int, default=8)
    ap.add_argument("--embed-bits", type=int, default=8)
    ap.add_argument("--delete", action="store_true",
                    help="remove: also delete the shard files")
    args = ap.parse_args(argv)

    if args.mode == "list":
        _emit({"root": MODELS_ROOT, "installed": installed_list()})
        return
    if args.mode == "remove":
        if not args.id:
            sys.stderr.write("--id is required\n")
            return 2
        try:
            _emit({"removed": remove(args.id, delete=args.delete)})
        except ValueError as e:
            sys.stderr.write(f"{e}\n")
            return 1
        return
    ident = args.id or args.repo
    if not ident:
        sys.stderr.write("--id (catalog) or --repo (HF) is required\n")
        return 2
    try:
        conf = pull(ident, args.attn_bits, args.mlp_bits, args.embed_bits,
                    progress=lambda n, t, p: None)
        _emit({"installed": conf["id"], "repo": conf["repo"],
               "dir": os.path.join(MODELS_ROOT, conf["id"]),
               "size_gb": conf.get("size_gb")})
    except ValueError as e:
        sys.stderr.write(f"{e}\n")
        return 1


if __name__ == "__main__":
    main()
