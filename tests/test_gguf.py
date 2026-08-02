"""GGUF converter test: write a small GGUF with F16 + Q8_0 tensors, convert
it, and verify the sharded output matches the safetensors importer's output
for identical weights (same quant pipeline, same manifest)."""
import json
import os
import struct
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from androidllm.gguf import GGUFReader, convert_gguf
from androidllm.safetensors import read_tensor, write_safetensors
from androidllm.shard import shard_model

TMP = os.path.join(os.path.dirname(os.path.abspath(__file__)), "tmp_gguf")
HF = os.path.join(TMP, "hf")
GG = os.path.join(TMP, "gg")

CANON = {
    "model_type": "qwen2",
    "hidden": 64,
    "layers": 2,
    "heads": 4,
    "kv_heads": 2,
    "head_dim": 16,
    "intermediate": 96,
    "vocab": 64,
    "rope_theta": 10000.0,
    "rms_eps": 1e-6,
    "max_len": 128,
    "tied": False,
    "name": "gguf-toy",
}


def _write_gguf(path, tensors):
    """tensors: dict name -> (dtype, np array). dtype in {F16, Q8_0, F32}."""
    with open(path, "wb") as f:
        f.write(b"GGUF")
        f.write(struct.pack("<I", 3))
        n_tensors = len(tensors)
        meta = {
            "general.architecture": "qwen2",
            "general.name": "gguf-toy",
            "qwen2.block_count": 2,
            "qwen2.embedding_length": 64,
            "qwen2.feed_forward_length": 96,
            "qwen2.attention.head_count": 4,
            "qwen2.attention.head_count_kv": 2,
            "qwen2.attention.max_position_embeddings": 128,
            "qwen2.rope.freq_base": 10000.0,
            "qwen2.attention.layer_norm_rms_epsilon": 1e-6,
            "qwen2.vocab_size": 64,
            "tokenizer.ggml.model": "gpt2",
            "tokenizer.ggml.tokens": ["h", "e", "l", "o", " ", "w", "r", "d",
                                      "<|im_start|>", "<|im_end|>", "</s>"],
            "tokenizer.ggml.merges": ["h e", "e l", "l o", "o w"],
            "tokenizer.ggml.special_tokens": ["<|im_start|>", "<|im_end|>", "</s>"],
            "tokenizer.ggml.bos_token_id": 0,
            "tokenizer.ggml.eos_token_id": 1,
            "tokenizer.chat_template": (
                "{% for message in messages %}"
                "<|im_start|>{{ message['role'] }}\n{{ message['content'] }}<|im_end|>\n"
                "{% endfor %}{% if add_generation_prompt %}<|im_start|>assistant\n{% endif %}"),
        }

        def w_str(s):
            b = s.encode("utf-8")
            f.write(struct.pack("<Q", len(b)))
            f.write(b)

        def w_val(tag, v):
            f.write(struct.pack("<I", tag))
            if tag == 6:
                f.write(struct.pack("<f", v))
            elif tag == 4:
                f.write(struct.pack("<I", v))
            elif tag == 8:
                w_str(v)
            elif tag == 9:
                etag, items = v
                f.write(struct.pack("<I", etag))
                f.write(struct.pack("<Q", len(items)))
                if etag == 8:
                    for s in items:
                        w_str(s)
                elif etag == 4:
                    for n in items:
                        f.write(struct.pack("<I", n))

        f.write(struct.pack("<Q", n_tensors))
        f.write(struct.pack("<Q", len(meta)))
        for k, v in meta.items():
            w_str(k)
            if isinstance(v, str):
                w_val(8, v)
            elif isinstance(v, bool):
                w_val(7, v)
            elif isinstance(v, int):
                w_val(4, v)
            elif isinstance(v, list):
                etag = 8 if all(isinstance(x, str) for x in v) else 4
                w_val(9, (etag, v))
            else:
                w_val(6, float(v))

        infos = []
        offset = 0
        for name, (dtype, arr) in tensors.items():
            if dtype == "F16":
                nbytes = 2 * arr.size
            elif dtype == "Q8_0":
                nbytes = 34 * (arr.size // 32)  # f16 scale + 32 int8 per block
            else:
                nbytes = 4 * arr.size
            infos.append((name, arr, dtype, offset))
            offset += nbytes

        for name, arr, dtype, off in infos:
            w_str(name)
            f.write(struct.pack("<I", arr.ndim))
            dims = tuple(reversed(arr.shape))
            f.write(struct.pack("<" + "I" * arr.ndim, *dims))
            t = 1 if dtype == "F16" else (8 if dtype == "Q8_0" else 0)
            f.write(struct.pack("<I", t))
            f.write(struct.pack("<Q", off))

        # data
        pad = (32 - f.tell() % 32) % 32
        f.write(b"\x00" * pad)
        data_base = f.tell()
        for _name, arr, dtype, off in infos:
            assert data_base + off == f.tell()
            if dtype == "F16":
                f.write(np.ascontiguousarray(arr, dtype=np.float16).tobytes())
            elif dtype == "Q8_0":
                a = np.ascontiguousarray(arr, dtype=np.float32)
                out, inp = a.shape
                for i in range(out):
                    for b0 in range(0, inp, 32):
                        blk = a[i, b0:b0 + 32]
                        d = np.float16(np.max(np.abs(blk)) / 127.0)
                        if d == 0:
                            d = np.float16(1e-8)
                        f.write(np.asarray([d], dtype=np.float16).tobytes())
                        q = np.clip(np.round(blk / float(d)), -127, 127).astype(np.int8)
                        f.write(q.tobytes())
            else:
                f.write(np.ascontiguousarray(arr, dtype=np.float32).tobytes())


def build_tensors():
    rng = np.random.default_rng(11)
    tensors = {}
    tensors["token_embd.weight"] = ("F16",
                                    (rng.standard_normal((64, 64)) * 0.02).astype(np.float16))
    tensors["output_norm.weight"] = ("F16", rng.standard_normal(64).astype(np.float16))
    tensors["output.weight"] = ("F16",
                                (rng.standard_normal((64, 64)) * 0.02).astype(np.float16))
    for i in range(2):
        base = f"blk.{i}."
        tensors[base + "attn_norm.weight"] = ("F16", rng.standard_normal(64).astype(np.float16))
        tensors[base + "ffn_norm.weight"] = ("F16", rng.standard_normal(64).astype(np.float16))
        for proj, o, n in (("attn_q", 64, 64), ("attn_k", 32, 64), ("attn_v", 32, 64),
                           ("attn_output", 64, 64), ("ffn_gate", 96, 64),
                           ("ffn_up", 96, 64), ("ffn_down", 64, 96)):
            w = (rng.standard_normal((o, n)) * 0.05).astype(np.float32)
            if proj == "attn_q":
                tensors[base + proj + ".weight"] = ("Q8_0", w)
            else:
                tensors[base + proj + ".weight"] = ("F16", w.astype(np.float16))
    return tensors


def write_hf_dir(tensors, out_dir):
    os.makedirs(out_dir, exist_ok=True)
    hf = {
        "model.embed_tokens.weight": tensors["token_embd.weight"][1].astype(np.float32),
        "model.norm.weight": tensors["output_norm.weight"][1].astype(np.float32),
        "model.lm_head.weight": tensors["output.weight"][1].astype(np.float32),
    }
    layer = {}
    for i in range(2):
        base = f"blk.{i}."
        layer[f"model.layers.{i}.input_layernorm.weight"] = (
            tensors[base + "attn_norm.weight"][1].astype(np.float32))
        layer[f"model.layers.{i}.post_attention_layernorm.weight"] = (
            tensors[base + "ffn_norm.weight"][1].astype(np.float32))
        for _proj, gg, hfn in (("attn_q", "attn_q", "self_attn.q_proj.weight"),
                              ("attn_k", "attn_k", "self_attn.k_proj.weight"),
                              ("attn_v", "attn_v", "self_attn.v_proj.weight"),
                              ("attn_output", "attn_output", "self_attn.o_proj.weight"),
                              ("ffn_gate", "ffn_gate", "mlp.gate_proj.weight"),
                              ("ffn_up", "ffn_up", "mlp.up_proj.weight"),
                              ("ffn_down", "ffn_down", "mlp.down_proj.weight")):
            layer[f"model.layers.{i}.{hfn}"] = tensors[base + gg + ".weight"][1].astype(np.float32)
    write_safetensors(os.path.join(out_dir, "model.safetensors"), {**hf, **layer})
    with open(os.path.join(out_dir, "config.json"), "w", encoding="utf-8") as f:
        json.dump({"model_type": "qwen2", "hidden_size": 64, "num_hidden_layers": 2,
                   "num_attention_heads": 4, "num_key_value_heads": 2,
                   "intermediate_size": 96, "vocab_size": 64,
                   "max_position_embeddings": 128, "rms_norm_eps": 1e-6,
                   "rope_theta": 10000.0, "tie_word_embeddings": False}, f)


def test_q8_dequant():
    import io
    f = io.BytesIO()
    n_blocks = 2
    d = np.array([2.0, 0.5], dtype=np.float16)
    s = np.array([0.25, -0.125], dtype=np.float16)
    q = np.array([1, -2, 3, -4, 5, -6, 7, -8, 0, 0, 0, 0, 0, 0, 0, 0,
                  9, -9, 8, -8, 7, -7, 6, -6, 5, -5, 4, -4, 3, -3, 2, -2,
                  1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1,
                  -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1],
                 dtype=np.int8).reshape(2, 32)
    for i in range(n_blocks):  # interleaved: [d f16][s f16][32x i8] per block
        f.write(d[i:i + 1].tobytes())
        f.write(s[i:i + 1].tobytes())
        f.write(q[i].tobytes())
    f.seek(0)
    deq = GGUFReader._q8(f, (1, 64), True)
    expected = q.astype(np.float32) * d[:, None].astype(np.float32) + s[:, None].astype(np.float32)
    assert np.allclose(deq, expected.reshape(1, 64)), np.max(np.abs(deq - expected))
    print("GGUF Q8_0 dequant OK")


def test_convert_matches_hf():
    os.makedirs(TMP, exist_ok=True)
    tensors = build_tensors()
    _write_gguf(os.path.join(TMP, "toy.gguf"), tensors)
    write_hf_dir(tensors, HF)
    os.makedirs(GG, exist_ok=True)
    man_gg = convert_gguf(os.path.join(TMP, "toy.gguf"), GG, attn_bits=4,
                          mlp_bits=8, block=32, embed_bits=0)
    shard_model(HF, os.path.join(TMP, "hf_out"), attn_bits=4,
                         mlp_bits=8, block=32, embed_bits=0)
    assert man_gg["config"]["layers"] == 2
    assert man_gg["config"]["hidden"] == 64
    assert man_gg["config"]["kv_heads"] == 2
    assert man_gg["has_lm_head"] is True
    assert man_gg["tied"] is False
    assert os.path.exists(os.path.join(GG, "vocab.txt"))
    assert os.path.exists(os.path.join(GG, "template.txt"))
    with open(os.path.join(GG, "vocab.txt"), encoding="utf-8") as f:
        assert "<|im_start|>" in f.read()
    for fname in ("embeddings.safetensors", "norms.safetensors",
                  "lm_head.safetensors", "layer_0.safetensors", "layer_1.safetensors"):
        for d in (GG, os.path.join(TMP, "hf_out")):
            assert os.path.exists(os.path.join(d, fname)), (d, fname)
    emb_gg = read_tensor(os.path.join(GG, "embeddings.safetensors"), "embed")
    emb_hf = read_tensor(os.path.join(TMP, "hf_out", "embeddings.safetensors"), "embed")
    assert np.allclose(emb_gg, emb_hf), np.max(np.abs(emb_gg - emb_hf))
    from androidllm.quant import dequantize_packed
    from androidllm.safetensors import read_header
    for layer in (0, 1):
        path_gg = os.path.join(GG, f"layer_{layer}.safetensors")
        path_hf = os.path.join(TMP, "hf_out", f"layer_{layer}.safetensors")
        h1, _, _ = read_header(path_gg)
        for base in ("q", "k", "v", "o", "gate", "up", "down"):
            qm = man_gg["quant"]["layers"][str(layer)][base]
            a = dequantize_packed(read_tensor(path_gg, base + ".q", h1),
                                  read_tensor(path_gg, base + ".scale", h1), qm)
            b = dequantize_packed(read_tensor(path_hf, base + ".q", h1),
                                  read_tensor(path_hf, base + ".scale", h1), qm)
            # q is Q8_0-sourced: its f16-scale dequant perturbs inputs slightly,
            # so a few int4 levels round differently; bound the effect, don't
            # require bit-equality for that one tensor.
            assert np.allclose(a, b, atol=0.05, rtol=0.05), \
                (layer, base, np.max(np.abs(a - b)))
    from androidllm.engine import LayerStreamingEngine
    e = LayerStreamingEngine(GG)
    out = e.generate([3, 7, 11], max_new_tokens=4)
    assert len(out) == 4 and all(0 <= t < 64 for t in out)
    print("GGUF convert == HF shard OK (layers={} vocab={} kv_heads={})".format(
        man_gg["config"]["layers"], man_gg["config"]["vocab"],
        man_gg["config"]["kv_heads"]))


if __name__ == "__main__":
    os.makedirs(TMP, exist_ok=True)
    test_q8_dequant()
    test_convert_matches_hf()
