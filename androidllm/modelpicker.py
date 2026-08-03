"""Model picker: pick the most stable + smart + fast llama-arch model for a
given device, or search HuggingFace for candidates.

Design:
  - CATALOG: curated llama-family chat-tuned models known to shard and run
    on androidllm (Llama/Qwen/SmolLM arches only; Gemma/Phi need unsupported
    arch changes).
  - score(): hard gates first (RAM resident budget, disk for download), then
    a weighted score: speed (est. tok/s on G85-class eMMC) + smart (params
    vs 1.7B reference) + stability (curated/known-good + community usage).
  - search(): HuggingFace API (stdlib urllib) filtered to llama-arch
    instruct models, scored the same way.

CLI (JSON on stdout):
  python -m androidllm.modelpicker pick [--specs "8gb ram 32gb storage"]
  python -m androidllm.modelpicker search --q "qwen3 instruct"
  python -m androidllm.modelpicker specs
"""
import argparse
import json
import sys
import time

from .devicespec import describe, device_specs, specs_from_text

# curated llama-arch chat-tuned models
# dl_gb = bf16 safetensors download, shard_gb = Q4 layer shards,
# params_b = billions, thinking = emits <think> (strip handled by serve),
# stability = known-good on androidllm (1.0 = battle-tested).
CATALOG = [
    {"id": "qwen15", "repo": "Qwen/Qwen2.5-1.5B-Instruct",
     "params_b": 1.54, "dl_gb": 3.2, "shard_gb": 1.1, "thinking": False,
     "stability": 1.0, "note": "best overall, great tool use",
     "draft": "qwen05", "spec_k": 4},
    {"id": "qwen3", "repo": "Qwen/Qwen3-1.7B-Instruct",
     "params_b": 1.72, "dl_gb": 3.4, "shard_gb": 1.28, "thinking": True,
     "stability": 0.85, "note": "thinking mode (min-p sampling)",
     "draft": "qwen3-06", "spec_k": 4},
    {"id": "smollm2", "repo": "HuggingFaceTB/SmolLM2-1.7B-Instruct",
     "params_b": 1.72, "dl_gb": 3.4, "shard_gb": 1.06, "thinking": False,
     "stability": 0.92, "note": "English-only, clean templates",
     "draft": "smollm2-135m", "spec_k": 5},
    {"id": "qwen3-06", "repo": "Qwen/Qwen3-0.6B-Instruct",
     "params_b": 0.62, "dl_gb": 1.3, "shard_gb": 0.5, "thinking": True,
     "stability": 0.85, "note": "tiny thinking model"},
    {"id": "qwen05", "repo": "Qwen/Qwen2.5-0.5B-Instruct",
     "params_b": 0.5, "dl_gb": 1.0, "shard_gb": 0.45, "thinking": False,
     "stability": 0.9, "note": "fastest useful size"},
    {"id": "smollm2-360m", "repo": "HuggingFaceTB/SmolLM2-360M-Instruct",
     "params_b": 0.36, "dl_gb": 0.72, "shard_gb": 0.32, "thinking": False,
     "stability": 0.95, "note": "very fast, weak reasoning"},
    {"id": "smollm2-135m", "repo": "HuggingFaceTB/SmolLM2-135M-Instruct",
     "params_b": 0.14, "dl_gb": 0.27, "shard_gb": 0.14, "thinking": False,
     "stability": 0.9, "note": "completion speed, minimal smarts"},
    # -- larger tier models (fits 5GB+ RAM: resident gate params*0.35+0.35
    #    against ram-1.2; scores rank them below the small fast ones unless
    #    the device actually has the RAM + disk) --
    {"id": "qwen25-3b", "repo": "Qwen/Qwen2.5-3B-Instruct",
     "params_b": 3.1, "dl_gb": 6.4, "shard_gb": 2.2, "thinking": False,
     "stability": 0.9, "note": "3B — fits 5GB RAM, big step up from 1.5B",
     "draft": "qwen05", "spec_k": 4},
    {"id": "qwen3-4b", "repo": "Qwen/Qwen3-4B-Instruct",
     "params_b": 4.0, "dl_gb": 8.2, "shard_gb": 2.8, "thinking": True,
     "stability": 0.85, "note": "4B thinking — fits 5GB RAM",
     "draft": "qwen3-06", "spec_k": 4},
    {"id": "qwen25-7b", "repo": "Qwen/Qwen2.5-7B-Instruct",
     "params_b": 7.6, "dl_gb": 15.6, "shard_gb": 5.3, "thinking": False,
     "stability": 0.85, "note": "7B generalist — fits 5GB+ RAM, needs ~17GB free"},
    {"id": "mistral-7b", "repo": "mistralai/Mistral-7B-Instruct-v0.3",
     "params_b": 7.3, "dl_gb": 14.9, "shard_gb": 5.1, "thinking": False,
     "stability": 0.8, "note": "7B Mistral v0.3 — fits 5GB+ RAM"},
    {"id": "qwen3-8b", "repo": "Qwen/Qwen3-8B-Instruct",
     "params_b": 8.0, "dl_gb": 16.4, "shard_gb": 5.6, "thinking": True,
     "stability": 0.8, "note": "8B thinking — fits 6GB+ RAM, needs ~18GB free"},
    {"id": "qwen3-14b", "repo": "Qwen/Qwen3-14B-Instruct",
     "params_b": 14.8, "dl_gb": 30.3, "shard_gb": 10.4, "thinking": True,
     "stability": 0.72, "note": "14B thinking — fits 7GB+ RAM, needs ~33GB free"},
    {"id": "qwen25-14b", "repo": "Qwen/Qwen2.5-14B-Instruct",
     "params_b": 14.8, "dl_gb": 30.3, "shard_gb": 10.4, "thinking": False,
     "stability": 0.75, "note": "14B generalist — fits 7GB+ RAM"},
    {"id": "mistral-24b", "repo": "mistralai/Mistral-Small-24B-Instruct-2501",
     "params_b": 24.1, "dl_gb": 49.4, "shard_gb": 16.9, "thinking": False,
     "stability": 0.65, "note": "24B — fits 10GB+ RAM, needs ~54GB free"},
    {"id": "qwen25-32b", "repo": "Qwen/Qwen2.5-32B-Instruct",
     "params_b": 32.8, "dl_gb": 67.3, "shard_gb": 23.0, "thinking": False,
     "stability": 0.65, "note": "32B generalist — fits 14GB+ RAM, needs ~74GB free"},
    {"id": "qwen3-32b", "repo": "Qwen/Qwen3-32B-Instruct",
     "params_b": 32.8, "dl_gb": 67.3, "shard_gb": 23.0, "thinking": True,
     "stability": 0.6, "note": "32B thinking — fits 14GB+ RAM, needs ~74GB free"},
]

# llama-arch search filter on the HF API
_ARCH_FILTERS = {"llama", "qwen", "smollm", "mistral"}


def _est_speed(params_b):
    """tok/s estimate on G85-class (eMMC layer streaming, numpy base)."""
    return max(0.15, 1.05 / params_b ** 0.85)


def _smart(params_b, thinking=False):
    s = min(1.0, params_b / 1.7)
    if thinking:
        s = min(1.0, s + 0.08)
    return round(s, 3)


def score(model, specs, est_speed=None):
    """Return (score, breakdown) or (None, reason) when it doesn't fit."""
    ram = specs.get("ram_gb") or 0.0
    disk = specs.get("disk_free_gb") or 0.0
    params_b = model.get("params_b", 1.0)
    resident = params_b * 0.35 + 0.35  # fp16 embed + KV + one layer
    need_dl = model.get("dl_gb", model.get("shard_gb", 1.0) * 3.2)
    shard = model.get("shard_gb", need_dl / 3.2)
    available = ram - 1.2  # OS + bot processes keep ~1.2 GB
    if available <= 0.1:
        return None, "RAM unknown or too low"
    if resident > available:
        return None, f"{resident:.2f} GB resident > {available:.1f} GB available RAM"
    if need_dl > max(disk * 0.9, 0.5):
        return None, f"{need_dl:.1f} GB download needs {need_dl:.1f} GB free (have {disk:.1f})"
    speed = est_speed(params_b) if est_speed else _est_speed(params_b)
    sp = min(1.0, speed / 1.2)
    sm = _smart(params_b, model.get("thinking", False))
    st = float(model.get("stability", 1.0))
    total = round(0.45 * sp + 0.35 * sm + 0.20 * st, 3)
    return total, {
        "speed": sp, "smart": sm, "stable": st,
        "est_tps": round(speed, 2), "resident_gb": round(resident, 2),
        "download_gb": round(need_dl, 1), "shard_gb": round(shard, 2),
    }


def pick(specs, catalog=None):
    """Rank the catalog for the given specs. Returns list of
    (model, score, breakdown) fitting the device, best first."""
    out = []
    for m in catalog or CATALOG:
        s, b = score(m, specs)
        if s is not None:
            out.append((m, s, b))
    out.sort(key=lambda x: x[1], reverse=True)
    return out


# -- HuggingFace search ----------------------------------------------------

_HF_API = "https://huggingface.co/api/models"


def _hf_json(url, timeout=20):
    import urllib.request
    req = urllib.request.Request(url, headers={"User-Agent": "androidllm-picker"})
    with urllib.request.urlopen(req, timeout=timeout) as r:
        return json.loads(r.read().decode("utf-8"))


def _tag_param(tags):
    for t in tags or []:
        m = __import__("re").match(r"^(\d+(?:\.\d+)?)[MB]$", t)
        if m:
            v = float(m.group(1))
            return v if t.endswith("B") else v / 1000
    return None


# llama-family architectures androidllm can shard + run
_SUPPORTED_ARCH = ("llama", "qwen2", "qwen3", "smollm", "mistral")


def _id_params(rid):
    """Param count from the model id ('Qwen3-4B-Instruct-2507' -> 4.0)."""
    m = __import__("re").search(r"(\d+(?:\.\d+)?)\s*([BM])", rid)
    if not m:
        return None
    return float(m.group(1)) if m.group(2) == "B" else float(m.group(1)) / 1000


def search_hf(query, limit=12, min_downloads=1000, specs=None):
    """HF search -> llama-family instruct candidates, scored for the current
    device via fit heuristics (bf16 download ~ params*2.05 GB)."""
    import re
    specs = specs or device_specs()
    url = (f"{_HF_API}?search={urllib_quote(query)}&sort=downloads&direction=-1&limit={limit}"
           )
    try:
        rows = _hf_json(url)
    except Exception as e:
        return {"error": str(e)}
    out = []
    for r in rows:
        rid = r.get("id", "")
        if not re.search(r"instruct|chat", rid, re.IGNORECASE):
            continue
        if re.search(r"-vl[-.]|vision", rid, re.IGNORECASE):
            continue
        tags = " ".join(r.get("tags") or []).lower()
        if not any(a in tags for a in _SUPPORTED_ARCH):
            continue
        params = _id_params(rid)
        if not params or params > 5.0:
            continue
        dl = r.get("downloads", 0)
        if dl < min_downloads:
            continue
        cand = {
            "id": rid.split("/")[-1].lower().replace(" ", "-")[:20],
            "repo": rid,
            "params_b": params,
            "dl_gb": round(params * 2.05, 1),
            "shard_gb": round(params * 0.7, 2),
            "thinking": "qwen3" in rid.lower(),
            "stability": min(1.0, dl / 500000),
            "downloads": dl,
        }
        s, b = score(cand, specs)
        if s is not None:
            cand["score"] = s
            cand["breakdown"] = b
            out.append(cand)
    out.sort(key=lambda c: c["score"], reverse=True)
    return {"specs": specs, "results": out[:8]}


def urllib_quote(s):
    import urllib.parse
    return urllib.parse.quote(s, safe="")


# -- CLI -------------------------------------------------------------------

def _emit(obj):
    sys.stdout.write(json.dumps(obj, ensure_ascii=False) + "\n")


def _tier_bounds(tier):
    """Parse a RAM-tier filter like '4-8' (fits 4-8 GB) or '8' (8+ GB).
    Returns (lo, hi) in GB with hi=None meaning unbounded; None if invalid."""
    if tier is None:
        return None
    t = tier.strip().lower().replace("gb", "")
    if "-" in t:
        lo, _, hi = t.partition("-")
        try:
            return float(lo), float(hi)
        except ValueError:
            return None
    try:
        return float(t), None
    except ValueError:
        return None


def _model_tier_label(resident_gb):
    """Coarse RAM tier for a model's resident footprint (Group 45 #1)."""
    if resident_gb <= 1.5:
        return "1-2GB"
    if resident_gb <= 4:
        return "2-4GB"
    if resident_gb <= 8:
        return "4-8GB"
    if resident_gb <= 16:
        return "8-16GB"
    return "16GB+"


def catalog_list(tier=None, specs=None):
    """All catalog entries with tier + fit info (Group 45 #4). Each entry:
    {id, repo, params_b, tier, resident_gb, download_gb, thinking, stability,
    note, fits: {ram, disk} | null} — fits evaluated against `specs` when given."""
    lo_hi = _tier_bounds(tier)
    entries = []
    for m in CATALOG:
        resident = round(m.get("params_b", 1.0) * 0.35 + 0.35, 2)
        if lo_hi is not None:
            lo, hi = lo_hi
            if resident < lo or (hi is not None and resident > hi):
                continue
        e = {
            "id": m["id"], "repo": m["repo"], "params_b": m["params_b"],
            "tier": _model_tier_label(resident), "resident_gb": resident,
            "download_gb": m.get("dl_gb", m.get("shard_gb", 1.0) * 3.2),
            "thinking": m.get("thinking", False),
            "stability": m.get("stability", 1.0), "note": m.get("note", ""),
            "fits": None,
        }
        if specs:
            s, b = score(m, specs)
            if s is not None:
                e["fits"] = {"ram": True, "disk": True, "score": s,
                             "resident_gb": b["resident_gb"],
                             "download_gb": b["download_gb"]}
            else:
                e["fits"] = {"ram": "RAM" not in b and "download" not in b,
                             "disk": "download" not in b,
                             "reason": b}
        entries.append(e)
    return {"tier_filter": tier, "count": len(entries), "entries": entries}


def main(argv=None):
    ap = argparse.ArgumentParser(prog="androidllm.modelpicker",
                                 description=__doc__)
    ap.add_argument("mode", choices=["pick", "search", "specs", "list"])
    ap.add_argument("--specs", help='manual specs, e.g. "8gb ram 32gb storage"')
    ap.add_argument("--q", default="qwen instruct", help="HF search query")
    ap.add_argument("--tier", default=None,
                    help="RAM tier filter, e.g. '4-8' (fits 4-8 GB) or '8' (=8+ GB)")
    args = ap.parse_args(argv)

    if args.mode == "specs":
        _emit(specs_from_text(args.specs))
        return

    if args.mode == "list":
        _emit(catalog_list(tier=args.tier, specs=specs_from_text(args.specs)))
        return

    specs = specs_from_text(args.specs)

    if args.mode == "pick":
        ranked = pick(specs)
        out = {
            "specs": specs,
            "specs_text": describe(specs),
            "time": int(time.time()),
            "picks": [{"model": m, "score": s, "breakdown": b}
                      for m, s, b in ranked],
        }
        _emit(out)
        return

    if args.mode == "search":
        _emit(search_hf(args.q, specs=specs))
        return


if __name__ == "__main__":
    main()
