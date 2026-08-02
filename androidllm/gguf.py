"""GGUF -> androidllm shard converter.

Reads llama.cpp GGUF files (v3), dequantizes supported tensor types back to
f32, maps tensor names to HF-style names, then re-quantizes through the same
shard pipeline as the safetensors importer (so all existing manifest/layer
format logic is reused).

Supported GGML types: F32, F16, BF16, Q8_0, Q8_1, Q4_0, Q4_1, Q5_0, Q5_1.
K-quants and IQ variants are rejected with a clear error (re-download the
model as fp16 or Q8_0).
"""

import argparse
import json
import os
import struct

import numpy as np

from .shard import bf16_to_f32, shard_from_tensors

GGUF_MAGIC = b"GGUF"
GGUF_ALIGNMENT = 32

GGML_T = {
    0: "F32", 1: "F16", 2: "Q4_0", 3: "Q4_1", 4: "Q4_2", 5: "Q4_3",
    6: "Q5_0", 7: "Q5_1", 8: "Q8_0", 9: "Q8_1", 10: "Q2_K", 11: "Q3_K",
    12: "Q4_K", 13: "Q5_K", 14: "Q6_K", 15: "Q8_K", 16: "IQ2_XXS",
    17: "IQ2_XS", 18: "IQ3_XXS", 19: "IQ1_S", 20: "IQ4_NL", 21: "IQ3_S",
    22: "IQ2_S", 23: "IQ4_XS", 24: "I8", 25: "I16", 26: "I32", 27: "I64",
    28: "F64", 29: "IQ1_M", 30: "BF16",
}

_KINDS = ("F32", "F16", "BF16", "Q8_0", "Q8_1", "Q4_0", "Q4_1", "Q5_0", "Q5_1")
_VAL_TAGS = {
    0: "u8", 1: "i8", 2: "u16", 3: "i16", 4: "u32", 5: "i32", 6: "f32",
    7: "bool", 8: "string", 9: "array", 10: "u64", 11: "i64", 12: "f64",
}
_VAL_FMT = {
    "u8": ("<B", 1), "i8": ("<b", 1), "u16": ("<H", 2), "i16": ("<h", 2),
    "u32": ("<I", 4), "i32": ("<i", 4), "f32": ("<f", 4), "bool": ("<?", 1),
    "u64": ("<Q", 8), "i64": ("<q", 8), "f64": ("<d", 8),
}


def _pad_to(n, align=GGUF_ALIGNMENT):
    return (align - n % align) % align


class GGUFReader:
    def __init__(self, path):
        self.path = path
        # file handle kept open for the reader's lifetime: tensors are read
        # on demand via seek() in _raw(); use `with GGUFReader(...)` to close.
        self.f = open(path, "rb")  # noqa: SIM115
        magic = self.f.read(4)
        if magic != GGUF_MAGIC:
            raise ValueError(f"not a GGUF file: {path}")
        self.version = struct.unpack("<I", self.f.read(4))[0]
        if self.version != 3:
            raise ValueError(f"unsupported GGUF version {self.version} (want 3)")
        n_tensors = struct.unpack("<Q", self.f.read(8))[0]
        n_kv = struct.unpack("<Q", self.f.read(8))[0]
        self.meta = {}
        for _ in range(n_kv):
            key, value = self._read_kv()
            self.meta[key] = value
        self.tensors = {}
        for _ in range(n_tensors):
            name = self._read_string()
            n_dim = struct.unpack("<I", self.f.read(4))[0]
            dims = struct.unpack("<" + "I" * n_dim, self.f.read(4 * n_dim))
            gtype = struct.unpack("<I", self.f.read(4))[0]
            off = struct.unpack("<Q", self.f.read(8))[0]
            self.tensors[name] = {"dims": tuple(int(d) for d in dims),
                                  "type": int(gtype), "offset": int(off)}
        self.data_start = (self.f.tell() + GGUF_ALIGNMENT - 1) & ~(GGUF_ALIGNMENT - 1)

    def close(self):
        self.f.close()

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc, tb):
        self.close()

    def _read_string(self):
        n = struct.unpack("<Q", self.f.read(8))[0]
        return self.f.read(n).decode("utf-8", errors="replace")

    def _read_scalar(self, tag):
        fmt, size = _VAL_FMT[tag]
        return struct.unpack(fmt, self.f.read(size))[0]

    def _read_value(self, tag=None):
        if tag is None:
            tag = _VAL_TAGS[struct.unpack("<I", self.f.read(4))[0]]
        if tag == "string":
            return self._read_string()
        if tag == "array":
            etag = _VAL_TAGS[struct.unpack("<I", self.f.read(4))[0]]
            count = struct.unpack("<Q", self.f.read(8))[0]
            if etag == "string":
                return [self._read_string() for _ in range(count)]
            if etag in _VAL_FMT:
                fmt, size = _VAL_FMT[etag]
                return [struct.unpack(fmt, self.f.read(size))[0]
                        for _ in range(count)]
            raise ValueError(f"unsupported array element type {etag}")
        return self._read_scalar(tag)

    def _read_kv(self):
        key = self._read_string()
        return key, self._read_value()

    # -- tensor decoding ------------------------------------------------

    def _raw(self, name):
        info = self.tensors[name]
        int(np.prod(info["dims"])) * 4  # upper bound (f32)
        self.f.seek(self.data_start + info["offset"])
        t = info["type"]
        kind = GGML_T.get(t)
        if kind not in _KINDS:
            raise ValueError(
                f"tensor {name}: unsupported GGML type {kind} (supported: "
                f"{', '.join(_KINDS)}); re-export the model as fp16 or Q8_0")
        return info, kind

    def read_tensor(self, name):
        """Dequantize one tensor to float32."""
        info, kind = self._raw(name)
        shape = tuple(reversed(info["dims"]))
        f = self.f
        f.seek(self.data_start + info["offset"])
        if kind == "F32":
            return np.frombuffer(f.read(4 * int(np.prod(shape))),
                                 dtype=np.float32).reshape(shape)
        if kind == "F16":
            return np.frombuffer(f.read(2 * int(np.prod(shape))),
                                 dtype=np.float16).reshape(shape).astype(np.float32)
        if kind == "BF16":
            return bf16_to_f32(np.frombuffer(f.read(2 * int(np.prod(shape))),
                                             dtype=np.uint16).reshape(shape))
        if kind in ("Q8_0", "Q8_1"):
            return self._q8(f, shape, kind == "Q8_1")
        if kind in ("Q4_0", "Q4_1"):
            return self._q4(f, shape, kind == "Q4_1")
        if kind in ("Q5_0", "Q5_1"):
            return self._q5(f, shape, kind == "Q5_1")
        raise ValueError(f"tensor {name}: unsupported type {kind}")

    @staticmethod
    def _q8(f, shape, has_sum):
        out, inp = shape
        n_blocks = (out * inp) // 32
        buf = np.frombuffer(f.read(n_blocks * (36 if has_sum else 34)),
                            dtype=np.uint8).reshape(n_blocks, -1)
        d = buf[:, 0:2].view(np.float16).astype(np.float32).reshape(n_blocks, 1)
        q = buf[:, (4 if has_sum else 2):(36 if has_sum else 34)].view(np.int8).astype(np.float32)
        deq = q * d
        if has_sum:
            deq = deq + buf[:, 2:4].view(np.float16).astype(np.float32)
        return deq.reshape(out, inp)

    @staticmethod
    def _q4(f, shape, has_sum):
        out, inp = shape
        n_blocks = (out * inp) // 32
        buf = np.frombuffer(f.read(n_blocks * (20 if has_sum else 18)),
                            dtype=np.uint8).reshape(n_blocks, -1)
        d = buf[:, 0:2].view(np.float16).astype(np.float32).reshape(n_blocks, 1)
        nib = buf[:, 4:20 if has_sum else 18] if has_sum else buf[:, 2:18]
        lo = (nib & 0x0F).astype(np.int32)
        hi = ((nib >> 4) & 0x0F).astype(np.int32)
        q = np.empty(nib.size * 2, dtype=np.int32)
        q[0::2] = lo
        q[1::2] = hi
        deq = (q - 8).astype(np.float32).reshape(n_blocks, 32) * d
        if has_sum:
            deq = deq + buf[:, 2:4].view(np.float16).astype(np.float32)
        return deq.reshape(out, inp)

    @staticmethod
    def _q5(f, shape, has_sum):
        out, inp = shape
        n_blocks = (out * inp) // 32
        buf = np.frombuffer(f.read(n_blocks * (24 if has_sum else 22)),
                            dtype=np.uint8).reshape(n_blocks, -1)
        d = buf[:, 0:2].view(np.float16).astype(np.float32).reshape(n_blocks, 1)
        off_hi = 6 if has_sum else 2
        qh = buf[:, off_hi:off_hi + 4].copy().view(np.uint32)
        ql = buf[:, off_hi + 4:off_hi + 20]
        lo = (ql & 0x0F).astype(np.int32)
        hi = ((ql >> 4) & 0x0F).astype(np.int32)
        q = np.empty(ql.size * 2, dtype=np.int32)
        q[0::2] = lo
        q[1::2] = hi
        high = np.repeat(qh, 32)
        bits = np.tile(np.arange(32, dtype=np.uint32), n_blocks)
        q = q + ((high >> bits) & 1).astype(np.int32) * 16
        deq = (q - 16).astype(np.float32).reshape(n_blocks, 32) * d
        if has_sum:
            deq = deq + buf[:, 2:4].view(np.float16).astype(np.float32)
        return deq.reshape(out, inp)


# -- tensor name mapping ----------------------------------------------------

_TENSOR_MAP = {
    "token_embd.weight": "model.embed_tokens.weight",
    "output_norm.weight": "model.norm.weight",
    "output.weight": "model.lm_head.weight",
    "blk.{i}.attn_norm.weight": "model.layers.{i}.input_layernorm.weight",
    "blk.{i}.attn_q.weight": "model.layers.{i}.self_attn.q_proj.weight",
    "blk.{i}.attn_k.weight": "model.layers.{i}.self_attn.k_proj.weight",
    "blk.{i}.attn_v.weight": "model.layers.{i}.self_attn.v_proj.weight",
    "blk.{i}.attn_output.weight": "model.layers.{i}.self_attn.o_proj.weight",
    "blk.{i}.ffn_norm.weight": "model.layers.{i}.post_attention_layernorm.weight",
    "blk.{i}.ffn_gate.weight": "model.layers.{i}.mlp.gate_proj.weight",
    "blk.{i}.ffn_up.weight": "model.layers.{i}.mlp.up_proj.weight",
    "blk.{i}.ffn_down.weight": "model.layers.{i}.mlp.down_proj.weight",
}

_SUPPORTED_ARCH = {"llama", "mistral", "qwen2", "qwen3", "phi3", "starcoder2",
                   "minicpm", "deepseek2"}


def hf_name(gguf_name):
    if gguf_name in _TENSOR_MAP:
        return _TENSOR_MAP[gguf_name]
    m = gguf_name.startswith("blk.")
    if m:
        i = int(gguf_name.split(".")[1])
        for pat, repl in _TENSOR_MAP.items():
            if pat.startswith("blk."):
                key = pat.format(i=i)
                if gguf_name == key:
                    return repl.format(i=i)
    return None


def canon_from_meta(meta):
    arch = meta.get("general.architecture")
    if not arch or arch not in _SUPPORTED_ARCH:
        raise ValueError(f"unsupported GGUF architecture: {arch!r} "
                         f"(supported: {', '.join(sorted(_SUPPORTED_ARCH))})")
    n = arch if arch in ("qwen2", "qwen3") else "llama"
    def get(key, default=0):
        return meta.get(f"{arch}.{key}", default)
    hidden = int(get("embedding_length"))
    heads = int(get("attention.head_count"))
    kv_heads = int(get("attention.head_count_kv", heads) or heads)
    if kv_heads == 0:
        kv_heads = heads
    head_dim = hidden // heads
    max_len = int(get("attention.max_position_embeddings", 2048) or 2048)
    return {
        "model_type": n,
        "hidden": hidden,
        "layers": int(get("block_count")),
        "heads": heads,
        "kv_heads": kv_heads,
        "head_dim": head_dim,
        "intermediate": int(get("feed_forward_length")),
        "vocab": int(get("vocab_size") or 0),
        "rope_theta": float(get("rope.freq_base", 10000.0) or 10000.0),
        "rms_eps": float(get("attention.layer_norm_rms_epsilon", 1e-5) or 1e-5),
        "max_len": max_len,
        "tied": False,
        "name": meta.get("general.name") or os.path.basename(arch),
    }


# -- tokenizer ----------------------------------------------------------------

def _fallback_template(arch):
    if arch in ("qwen2", "qwen3"):
        return (
            "{% for message in messages %}{% if message['role'] == 'system' %}"
            "{{ message['content'] }}{% elif message['role'] == 'user' %}"
            "<|im_start|>user\n{{ message['content'] }}<|im_end|>\n"
            "{% elif message['role'] == 'assistant' %}<|im_start|>assistant\n"
            "{{ message['content'] }}<|im_end|>\n{% endif %}{% endfor %}"
            "{% if add_generation_prompt %}<|im_start|>assistant\n{% endif %}"
        )
    return (
        "{% for message in messages %}{{ '<s>' if loop.index == 1 }}"
        "{% if message['role'] == 'user' %}[INST] {{ message['content'] }} [/INST]"
        "{% elif message['role'] == 'assistant' %}{{ message['content'] }}{% endif %}"
        "{% if not loop.last %}{{ ' ' }}{% endif %}{% endfor %}"
    )


def write_tokenizer(out_dir, meta, arch):
    tokens = meta.get("tokenizer.ggml.tokens") or []
    merges = meta.get("tokenizer.ggml.merges") or []
    specials = meta.get("tokenizer.ggml.special_tokens") or []
    if not tokens:
        raise ValueError("GGUF has no tokenizer.ggml.tokens (need a tokenizer-equipped GGUF)")
    with open(os.path.join(out_dir, "vocab.txt"), "w", encoding="utf-8") as f:
        for t in tokens:
            f.write(t.replace("\n", "\\n") + "\n")
    with open(os.path.join(out_dir, "merges.txt"), "w", encoding="utf-8") as f:
        for m in merges:
            f.write(m + "\n")
    spec = {}
    token_ids = {t: i for i, t in enumerate(tokens)}
    for s in specials:
        if s in token_ids:
            spec[s] = token_ids[s]
    with open(os.path.join(out_dir, "special_tokens.json"), "w", encoding="utf-8") as f:
        json.dump(spec, f)
    template = meta.get("tokenizer.chat_template") or _fallback_template(arch)
    with open(os.path.join(out_dir, "template.txt"), "w", encoding="utf-8") as f:
        f.write(template)


# -- CLI --------------------------------------------------------------------

def convert_gguf(source, out, attn_bits=4, mlp_bits=8, block=64, embed_bits=8):
    r = GGUFReader(source)
    try:
        arch = r.meta.get("general.architecture")
        canon = canon_from_meta(r.meta)
        if "output.weight" in r.tensors:
            canon["tied"] = False
        else:
            canon["tied"] = True
        os.makedirs(out, exist_ok=True)
        write_tokenizer(out, r.meta, arch)
        tensors = {}
        for gname in r.tensors:
            hf = hf_name(gname)
            if hf is None:
                continue
            tensors[hf] = (lambda n=gname: r.read_tensor(n))
        if not tensors:
            raise ValueError("no supported tensors found in GGUF (check architecture)")
        manifest = shard_from_tensors(tensors, canon, out, attn_bits, mlp_bits,
                                      block, embed_bits)
    finally:
        r.close()
    return manifest


def main():
    ap = argparse.ArgumentParser(description="Convert a GGUF model to androidllm shards")
    ap.add_argument("--source", required=True, help="path to .gguf file")
    ap.add_argument("--out", required=True)
    ap.add_argument("--attn-bits", type=int, default=4)
    ap.add_argument("--mlp-bits", type=int, default=8)
    ap.add_argument("--block", type=int, default=64)
    ap.add_argument("--embed-bits", type=int, default=8,
                    help="quantize the embedding table; 0 = keep f16")
    args = ap.parse_args()
    convert_gguf(args.source, args.out, args.attn_bits, args.mlp_bits,
                 args.block, args.embed_bits)
    print(f"converted GGUF -> {args.out}")


if __name__ == "__main__":
    main()
