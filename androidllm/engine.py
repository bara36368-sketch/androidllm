import json
import os
from concurrent.futures import ThreadPoolExecutor

import numpy as np

from .quant import dequantize_packed
from .safetensors import read_header, read_tensor
from .models.llama import LlamaModel


def _sample(logits, temperature, top_p, rng):
    logits = np.asarray(logits, dtype=np.float32).reshape(-1)
    if temperature <= 0:
        return int(np.argmax(logits))
    probs = np.exp((logits - np.max(logits)) / max(temperature, 1e-9))
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


class LayerStreamingEngine:
    """Streams quantized layer files from disk one at a time, like AirLLM:
    load layer -> compute -> clean_memory, with the next layer prefetched
    while the current one computes. KV cache stays resident."""

    def __init__(self, model_dir):
        self.model_dir = model_dir
        with open(os.path.join(model_dir, "manifest.json"), encoding="utf-8") as f:
            self.manifest = json.load(f)
        canon = self.manifest["config"]
        self.canon = canon
        self.n_layers = canon["layers"]
        self.layer_meta = self.manifest.get("quant", {}).get("layers", {})
        embed = read_tensor(os.path.join(model_dir, "embeddings.safetensors"), "embed")
        final_norm = read_tensor(os.path.join(model_dir, "norms.safetensors"), "final_norm")
        lm_head = None
        if self.manifest.get("has_lm_head"):
            lm_head = read_tensor(os.path.join(model_dir, "lm_head.safetensors"), "lm_head")
        self.model = LlamaModel(canon, embed, final_norm, lm_head)
        self._pool = ThreadPoolExecutor(max_workers=1)
        self._rng = np.random.default_rng()
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

    def load_layer(self, i):
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

    def clean_memory(self):
        """Free transient buffers (KV cache is recreated per generate)."""
        if self._pool:
            pass

    def generate(self, prompt_ids, max_new_tokens=64, temperature=0.8,
                 top_p=0.9, stop_ids=(), callback=None):
        kv = self.model.prepare_kv(self.canon["max_len"])
        token = prompt_ids[0]
        generated = []
        for pos in range(len(prompt_ids) - 1):
            pending = self._pool.submit(self.load_layer, 0)
            for l in range(self.n_layers):
                layer = pending.result()
                pending = (self._pool.submit(self.load_layer, l + 1)
                           if l + 1 < self.n_layers else None)
                self.model.forward_one(token, layer, kv, pos)
                layer = None
            token = prompt_ids[pos + 1]
        pos = len(prompt_ids) - 1
        while len(generated) < max_new_tokens:
            pending = self._pool.submit(self.load_layer, 0)
            x = None
            for l in range(self.n_layers):
                layer = pending.result()
                pending = (self._pool.submit(self.load_layer, l + 1)
                           if l + 1 < self.n_layers else None)
                x = self.model.forward_one(token, layer, kv, pos)
                layer = None
            logits = self.model.logits(x)
            token = _sample(logits, temperature, top_p, self._rng)
            if token in stop_ids:
                break
            generated.append(token)
            pos += 1
            if callback:
                callback(token)
        return generated

    def chat(self, messages, max_new_tokens=64, temperature=0.8, top_p=0.9):
        if self.tokenizer is None:
            raise ValueError("model has no tokenizer; convert it with androidllm-shard")
        prompt = self.tokenizer.apply_template(messages, add_generation_prompt=True)
        ids = self.tokenizer.encode(prompt)
        out = self.generate(ids, max_new_tokens, temperature, top_p, stop_ids=self.stop_ids)
        return self.tokenizer.decode(out)
