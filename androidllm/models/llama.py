import math

import numpy as np

from ..neon import matmul_f16


def rms_norm(x, w, eps=1e-6):
    x = x.astype(np.float32)
    mean = np.mean(x * x, axis=-1, keepdims=True)
    return (x / np.sqrt(mean + eps)) * w.astype(np.float32)


def precompute_rope(head_dim, max_len, theta=10000.0):
    inv = 1.0 / (theta ** (np.arange(0, head_dim, 2) / head_dim))
    t = np.arange(max_len)
    freqs = np.outer(t, inv)
    return np.concatenate([np.cos(freqs), np.sin(freqs)], axis=1).astype(np.float16)


def apply_rope(x, pos, rope):
    head_dim = x.shape[-1]
    half = head_dim // 2
    c = rope[pos, :half].astype(np.float32)
    s = rope[pos, half:].astype(np.float32)
    x = x.astype(np.float32)
    a = x[..., :half]
    b = x[..., half:]
    r = np.empty_like(x)
    r[..., :half] = a * c - b * s
    r[..., half:] = a * s + b * c
    return r


def swiglu(gate, up):
    gate = gate.astype(np.float32)
    return gate * (1.0 / (1.0 + np.exp(-gate))) * up.astype(np.float32)


def quantized_mm(x, w):
    return matmul_f16(x, w)


class LlamaModel:
    def __init__(self, canon, embed, final_norm, lm_head=None):
        self.canon = canon
        self.hidden = canon["hidden"]
        self.heads = canon["heads"]
        self.kv_heads = canon["kv_heads"]
        self.head_dim = canon["head_dim"]
        self.rope_theta = canon["rope_theta"]
        self.rms_eps = canon["rms_eps"]
        self.embed = embed
        self.final_norm = final_norm
        self.lm_head = lm_head
        self.rope = precompute_rope(self.head_dim, canon["max_len"], self.rope_theta)

    def layer_forward(self, x, layer, kv, pos):
        q = quantized_mm(x, layer["q"]).reshape(1, self.heads, self.head_dim)
        k = quantized_mm(x, layer["k"]).reshape(1, self.kv_heads, self.head_dim)
        v = quantized_mm(x, layer["v"]).reshape(1, self.kv_heads, self.head_dim)
        q = apply_rope(q, pos, self.rope)
        k = apply_rope(k, pos, self.rope)
        kv_k, kv_v = kv
        kv_k[pos, :, :] = k[0].astype(np.float16)
        kv_v[pos, :, :] = v[0].astype(np.float16)
        g = self.heads // self.kv_heads
        k_rep = np.repeat(kv_k[: pos + 1], g, axis=1).transpose(1, 2, 0)
        v_rep = np.repeat(kv_v[: pos + 1], g, axis=1).transpose(1, 0, 2)
        scores = (q[0][:, None, :].astype(np.float32) @ k_rep.astype(np.float32)) / math.sqrt(self.head_dim)
        probs = np.exp(scores - np.max(scores, axis=-1, keepdims=True))
        probs = probs / np.sum(probs, axis=-1, keepdims=True)
        ctx = (probs.astype(np.float16) @ v_rep.astype(np.float16)).reshape(self.heads, self.head_dim)
        o = quantized_mm(ctx.reshape(1, self.hidden), layer["o"])
        x = x + o.reshape(1, self.hidden).astype(np.float16)
        x = rms_norm(x, layer["n_in"], self.rms_eps).astype(np.float16)
        gate = quantized_mm(x, layer["gate"])
        up = quantized_mm(x, layer["up"])
        mlp = swiglu(gate, up)
        down = quantized_mm(mlp.reshape(1, -1), layer["down"])
        x = x + down.reshape(1, self.hidden).astype(np.float16)
        x = rms_norm(x, layer["n_post"], self.rms_eps).astype(np.float16)
        return x

    def forward_one(self, token, layer, kv, pos):
        x = self.embed[token].reshape(1, self.hidden)
        x = self.layer_forward(x, layer, kv, pos)
        return x

    def logits(self, x):
        h = rms_norm(x, self.final_norm, self.rms_eps)
        w = self.lm_head if self.lm_head is not None else self.embed
        return (h.astype(np.float16) @ w.astype(np.float16).T).astype(np.float32)

    def prepare_kv(self, max_len):
        shape = (max_len, self.kv_heads, self.head_dim)
        return (np.zeros(shape, dtype=np.float16), np.zeros(shape, dtype=np.float16))
