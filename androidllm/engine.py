import gc
import json
import os
import threading
import time
from collections import OrderedDict
from concurrent.futures import ThreadPoolExecutor

import numpy as np

from .quant import dequantize_packed
from .safetensors import read_header, read_tensor
from .models.llama import LlamaModel

# Optional Rust acceleration for sampling
try:
    import androidllm_rs as _rs

    _HAS_RS_SAMPLE = True
except Exception:
    _rs = None
    _HAS_RS_SAMPLE = False


def _sample(logits, temperature, top_p, min_p, rng):
    logits = np.asarray(logits, dtype=np.float32).reshape(-1)
    if temperature <= 0:
        return int(np.argmax(logits))
    probs = np.exp((logits - np.max(logits)) / max(temperature, 1e-9))
    probs = probs / probs.sum()
    if min_p and min_p > 0 and min_p < 1.0:
        keep = probs >= min_p * probs.max()
        if keep.any():
            probs = np.where(keep, probs, 0.0)
            probs = probs / probs.sum()
    if top_p < 1.0:
        order = np.argsort(-probs)
        cum = np.cumsum(probs[order])
        keep = order[cum <= top_p]
        if len(keep) == 0:
            keep = order[:1]
        sub = probs[keep] / probs[keep].sum()
        return int(keep[rng.choice(len(keep), p=sub)])
    return int(rng.choice(len(probs), p=probs))


def _sample_rs(logits, temperature, top_p, min_p, rng):
    """Optional Rust-accelerated sampling with fallback to numpy."""
    if _HAS_RS_SAMPLE:
        # Use a seed from the RNG for deterministic per-call sampling
        # rng.integers returns Python int, convert to uint64
        seed = int(rng.integers(0, 2**63 - 1, dtype=np.uint64))
        return int(_rs.sample(
            np.asarray(logits, dtype=np.float32).reshape(-1),
            float(temperature),
            float(top_p),
            float(min_p),
            seed,
        ))
    return _sample(logits, temperature, top_p, min_p, rng)


def _rss_bytes():
    try:
        with open("/proc/self/status", encoding="utf-8") as f:
            for line in f:
                if line.startswith("VmRSS:"):
                    return int(line.split()[1]) * 1024
    except OSError:
        pass
    return 0


class LayerStreamingEngine:
    """Streams quantized layer files from disk one at a time, like AirLLM.
    Extras on top of plain AirLLM:
      - keep-first-N pinned layers + LRU tail (ANDROIDLLM_KEEP_LAYERS /
        ANDROIDLLM_LRU_LAYERS)
      - per-layer KV cache with cross-turn prefix reuse
        (ANDROIDLLM_PREFIX_KV)
      - layer skipping (ANDROIDLLM_SKIP_EVERY)
      - chunked prefill with gc relief (ANDROIDLLM_PREFILL_CHUNK)
      - optional per-token throttle pause (ANDROIDLLM_THROTTLE_MS)
      - speculative decoding via a small draft model (draft_dir/spec_k):
        the draft guesses K next tokens cheaply; the target verifies them
        in one pass and accepts the matching prefix, so each accepted token
        skips a full target forward pass (ANDROIDLLM_SPEC_K)."""

    def __init__(self, model_dir, keep_layers=0, draft_dir=None, spec_k=0):
        self.model_dir = model_dir
        with open(os.path.join(model_dir, "manifest.json"), encoding="utf-8") as f:
            self.manifest = json.load(f)
        canon = self.manifest["config"]
        self.canon = canon
        self.n_layers = canon["layers"]
        self.layer_meta = self.manifest.get("quant", {}).get("layers", {})
        embed_path = os.path.join(model_dir, "embeddings.safetensors")
        eq = self.manifest.get("embed_quant") or {}
        if eq:
            packed = read_tensor(embed_path, "embed.q")
            scale = read_tensor(embed_path, "embed.scale")
            embed = dequantize_packed(packed, scale, eq)
        else:
            embed = read_tensor(embed_path, "embed")
        final_norm = read_tensor(os.path.join(model_dir, "norms.safetensors"), "final_norm")
        lm_head = None
        if self.manifest.get("has_lm_head"):
            lq = self.manifest.get("lm_head_quant") or {}
            if lq:
                packed = read_tensor(os.path.join(model_dir, "lm_head.safetensors"), "lm_head.q")
                scale = read_tensor(os.path.join(model_dir, "lm_head.safetensors"), "lm_head.scale")
                lm_head = dequantize_packed(packed, scale, lq)
            else:
                lm_head = read_tensor(os.path.join(model_dir, "lm_head.safetensors"), "lm_head")
        self.model = LlamaModel(canon, embed, final_norm, lm_head)
        self._pool = ThreadPoolExecutor(max_workers=1)
        self._rng = np.random.default_rng()
        self.ctx_len = min(int(canon["max_len"]),
                           int(os.environ.get("ANDROIDLLM_MAX_CTX", "4096")))
        self._keep = max(0, int(keep_layers))
        self._lru = max(0, int(os.environ.get("ANDROIDLLM_LRU_LAYERS", "0")))
        self._skip_every = max(0, int(os.environ.get("ANDROIDLLM_SKIP_EVERY", "0")))
        self._prefix_kv = os.environ.get("ANDROIDLLM_PREFIX_KV", "1") not in ("0", "false", "")
        self._prefill_chunk = max(1, int(os.environ.get("ANDROIDLLM_PREFILL_CHUNK", "128")))
        self.throttle_ms = max(0, int(os.environ.get("ANDROIDLLM_THROTTLE_MS", "0")))
        self._cache = OrderedDict()
        self._lock = threading.Lock()
        self._kv = None
        self._kv_prompt_ids = []
        self._kv_len = 0
        self.stats = {"cache_hits": 0, "cache_misses": 0, "load_ms": 0.0,
                      "load_calls": 0, "compute_ms": 0.0, "compute_tokens": 0,
                      "tokens_served": 0, "prefix_calls": 0, "prefix_tokens": 0,
                      "started": time.time()}
        # -- speculative decoding -----------------------------------------
        self.draft = None
        self.spec_k = max(0, int(spec_k))
        if draft_dir:
            self.draft = LayerStreamingEngine(draft_dir, keep_layers=0)
            self.draft.spec_k = 0
        self.tokenizer = None
        if os.path.exists(os.path.join(model_dir, "vocab.txt")):
            from .tokenizer import ByteLevelBPE
            self.tokenizer = ByteLevelBPE(model_dir)
        self.stop_ids = self._default_stops()

    def _default_stops(self):
        stops = []
        for name in ("<|im_end|>", "<|eot_id|>", "</s>", "<|endoftext|>"):
            tid = self.tokenizer.specials.get(name) if self.tokenizer else None
            if tid is not None:
                stops.append(tid)
        if not stops:
            tid = self.tokenizer.token_to_id.get("<|im_end|>") if self.tokenizer else None
            if tid is not None:
                stops.append(tid)
        return tuple(stops)

    def _load_layer_raw(self, i):
        path = os.path.join(self.model_dir, "layer_%d.safetensors" % i)
        header, _, _ = read_header(path)
        layer = {}
        for base, qm in self.layer_meta.get(str(i), {}).items():
            packed = read_tensor(path, base + ".q", header)
            scale = read_tensor(path, base + ".scale", header)
            deq = dequantize_packed(packed, scale, qm)
            in_real = qm.get("in_real", qm["in"])
            if in_real != qm["in"]:
                deq = deq[:, :in_real]
            layer[base] = deq
        layer["n_in"] = read_tensor(path, "input_layernorm.weight", header)
        layer["n_post"] = read_tensor(path, "post_attention_layernorm.weight", header)
        return layer

    def _cache_get(self, i):
        with self._lock:
            layer = self._cache.get(i)
            if layer is not None and i >= self._keep:
                self._cache.move_to_end(i)
            return layer

    def _cache_put(self, i, layer):
        with self._lock:
            self._cache[i] = layer
            cap = self._keep + self._lru
            while len(self._cache) > cap and self._lru > 0:
                self._cache.popitem(last=False)

    def _skip(self, i):
        return (self._skip_every > 1 and i > 0 and i < self.n_layers - 1
                and i % self._skip_every == 0)

    def _layers(self):
        """Yield (i, layer-or-None) with one-deep prefetch; skipped layers
        yield None without touching disk. Fresh iterator per forward pass."""
        pending = None
        for i in range(self.n_layers):
            skip = self._skip(i)
            nxt_skip = i + 1 >= self.n_layers or self._skip(i + 1)
            nxt = (self._pool.submit(self.load_layer, i + 1)
                   if i + 1 < self.n_layers and not nxt_skip else None)
            layer = None
            if pending is not None:
                layer = pending.result()
            elif not skip:
                layer = self.load_layer(i)
            pending = nxt
            yield i, layer

    def load_layer(self, i):
        """Layer cache: layers < keep stay pinned; others are LRU-capped at
        lru. Cached layers never touch disk again."""
        if i < self._keep or self._lru > 0:
            cached = self._cache_get(i)
            if cached is not None:
                self.stats["cache_hits"] += 1
                return cached
            t0 = time.time()
            layer = self._load_layer_raw(i)
            self.stats["load_ms"] += (time.time() - t0) * 1000
            self.stats["load_calls"] += 1
            self.stats["cache_misses"] += 1
            self._cache_put(i, layer)
            return layer
        t0 = time.time()
        layer = self._load_layer_raw(i)
        self.stats["load_ms"] += (time.time() - t0) * 1000
        self.stats["load_calls"] += 1
        return layer

    def clean_memory(self):
        """Free transient buffers (KV cache is recreated per generate)."""
        if self._pool:
            pass

    def close(self):
        """Stop prefetch threads and release the draft engine."""
        try:
            self._pool.shutdown(wait=False)
        except Exception:
            pass
        if self.draft is not None:
            self.draft.close()
            self.draft = None

    def _match_prefix(self, ids):
        if not self._prefix_kv or self._kv is None or not self._kv_prompt_ids:
            return 0
        old = self._kv_prompt_ids
        n = min(len(old), len(ids), self.ctx_len)
        p = 0
        while p < n and old[p] == ids[p]:
            p += 1
        return p

    def _forward(self, x, kv, pos):
        """One full layer pass at position pos, writing into kv[pos].
        A fresh layer iterator is created per pass (layers are finite)."""
        for l, layer in self._layers():
            if layer is not None:
                t0 = time.time()
                x = self.model.layer_forward(x, layer, kv[l], pos)
                self.stats["compute_ms"] += (time.time() - t0) * 1000
                self.stats["compute_tokens"] += 1
        return x

    def _score(self, x):
        return self.model.logits(x)

    def _step(self, kv, pos, token, temperature, top_p, min_p, grammar, buf_parts):
        """Embed token, run a full forward pass at pos, sample the next token."""
        x = self._forward(self.model.embed[token].reshape(1, self.model.hidden), kv, pos)
        logits = self._score(x)
        if grammar is not None:
            mask = grammar.allowed_mask("".join(buf_parts), self.tokenizer)
            mask = mask[:logits.shape[-1]]
            logits = np.where(mask, logits, -np.inf)
        return _sample_rs(logits, temperature, top_p, min_p, self._rng)

    def _spec_step(self, kv, dkv, pos, token, temperature, top_p, min_p):
        """Speculative step: draft guesses K tokens, target verifies.
        Returns (emitted, next_token, next_pos, n_draft_accepted) where
        emitted are the generated tokens (draft-matched ones, then a bonus
        or a correction from the target). Invariants in/out: kv filled
        0..pos-1, token = id at pos."""
        d = self.draft
        K = self.spec_k
        cands = []
        dtok = token
        dpos = pos
        for _ in range(K):
            dx = d._forward(d.model.embed[dtok].reshape(1, d.model.hidden), dkv, dpos)
            cands.append(_sample_rs(d._score(dx), temperature, top_p, min_p, self._rng))
            dtok = cands[-1]
            dpos += 1
        self.stats["draft_tokens"] += K

        emitted = []
        tpos = pos
        tok = token
        for c in cands:
            g = self._step(kv, tpos, tok, temperature, top_p, min_p, None, None)
            if g == c:
                emitted.append(c)
                tok = c
                tpos += 1
            else:
                emitted.append(g)
                return emitted, g, tpos + 1, len(emitted) - 1
        g = self._step(kv, tpos, tok, temperature, top_p, min_p, None, None)
        emitted.append(g)
        return emitted, g, tpos + 1, len(emitted) - 1

    def generate(self, prompt_ids, max_new_tokens=64, temperature=0.8,
                 top_p=0.9, min_p=0.0, stop_ids=(), callback=None,
                 grammar=None):
        prompt_ids = list(prompt_ids)[: self.ctx_len]
        pfx = self._match_prefix(prompt_ids)
        if pfx > 1:
            kv = self._kv
            pos = pfx
            self.stats["prefix_calls"] += 1
            self.stats["prefix_tokens"] += pfx
        else:
            kv = self.model.prepare_kv(self.ctx_len)
            self._kv = kv
            pos = 0
        self._kv_prompt_ids = prompt_ids
        spec = (self.draft is not None and self.spec_k > 0
                and grammar is None and self.tokenizer is not None)
        if spec:
            self.draft_kv = self.draft.model.prepare_kv(self.ctx_len)
            self.stats.setdefault("draft_tokens", 0)
            self.stats.setdefault("spec_accepted", 0)
            self.stats.setdefault("spec_bonus", 0)
            self.stats.setdefault("spec_steps", 0)

        # prefill: kv[0..n-1] filled, token = id at position n-1
        for i in range(pos, len(prompt_ids) - 1):
            self._forward(self.model.embed[prompt_ids[i]].reshape(1, self.model.hidden),
                          kv, i)
            if (i % self._prefill_chunk) == 0:
                gc.collect()
        token = prompt_ids[-1]
        pos = len(prompt_ids) - 1
        if spec:
            # draft kv mirrors the full prompt once (positions 0..len-1)
            for i in range(len(prompt_ids)):
                self.draft._forward(
                    self.draft.model.embed[prompt_ids[i]].reshape(1, self.draft.model.hidden),
                    self.draft_kv, i)

        generated = []
        buf_parts = []
        while len(generated) < max_new_tokens:
            if token in stop_ids:
                break
            if spec:
                emitted, token, pos, n_draft = self._spec_step(
                    kv, self.draft_kv, pos, token, temperature, top_p, min_p)
                self.stats["spec_steps"] += 1
                for i, t in enumerate(emitted):
                    if t in stop_ids:
                        token = t
                        break
                    if len(generated) >= max_new_tokens:
                        break
                    generated.append(t)
                    self.stats["tokens_served"] += 1
                    if i < n_draft:
                        self.stats["spec_accepted"] += 1
                    else:
                        self.stats["spec_bonus"] += 1
                    if callback:
                        callback(t)
                if len(generated) >= max_new_tokens:
                    break
            else:
                token = self._step(kv, pos, token, temperature, top_p, min_p,
                                   grammar, buf_parts)
                pos += 1
                if token in stop_ids:
                    break
                generated.append(token)
                if grammar is not None:
                    buf_parts.append(self.tokenizer.decode([token]))
                self.stats["tokens_served"] += 1
                if callback:
                    callback(token)
            if self.throttle_ms > 0:
                time.sleep(self.throttle_ms / 1000.0)
        self._kv_len = pos
        if spec:
            del self.draft_kv
        return generated

    def chat(self, messages, max_new_tokens=64, temperature=0.8, top_p=0.9,
             min_p=0.0):
        if self.tokenizer is None:
            raise ValueError("model has no tokenizer; convert it with androidllm-shard")
        prompt = self.tokenizer.apply_template(messages, add_generation_prompt=True)
        ids = self.tokenizer.encode(prompt)
        out = self.generate(ids, max_new_tokens, temperature, top_p, min_p,
                            stop_ids=self.stop_ids)
        return self.tokenizer.decode(out)

    def snapshot(self):
        up = time.time() - self.stats["started"]
        snap = {
            "model": self.manifest.get("name", "androidllm"),
            "uptime_s": int(up),
            "tokens_served": self.stats["tokens_served"],
            "cache_hits": self.stats["cache_hits"],
            "cache_misses": self.stats["cache_misses"],
            "cache_size": len(self._cache),
            "keep_layers": self._keep,
            "lru_layers": self._lru,
            "skip_every": self._skip_every,
            "ctx_len": self.ctx_len,
            "prefix_kv": self._prefix_kv,
            "prefix_calls": self.stats["prefix_calls"],
            "prefix_tokens": self.stats["prefix_tokens"],
            "context_used": self._kv_len,
            "context_pct": round(self._kv_len / self.ctx_len * 100, 1)
            if self.ctx_len else 0.0,
            "paused": bool(getattr(self, "paused", False)),
            "throttle_ms": self.throttle_ms,
            "load_ms_avg": (self.stats["load_ms"] / self.stats["load_calls"]
                            if self.stats["load_calls"] else 0.0),
            "compute_ms_per_token": (self.stats["compute_ms"]
                                     / self.stats["compute_tokens"]
                                     if self.stats["compute_tokens"] else 0.0),
            "rss_mb": round(_rss_bytes() / 1048576, 1),
        }
        if self.draft is not None:
            dt = self.stats.get("draft_tokens", 0)
            acc = self.stats.get("spec_accepted", 0)
            bon = self.stats.get("spec_bonus", 0)
            snap["spec"] = {
                "enabled": self.spec_k > 0,
                "k": self.spec_k,
                "draft": self.draft.manifest.get("name", "draft"),
                "steps": self.stats.get("spec_steps", 0),
                "draft_tokens": dt,
                "accepted": acc,
                "bonus": bon,
                "accept_rate": round(acc / (acc + bon), 3) if (acc + bon) else 0.0,
            }
        return snap
