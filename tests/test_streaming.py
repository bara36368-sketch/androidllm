"""End-to-end: shard a tiny synthetic model, then verify the streaming engine
produces identical logits to loading all layers at once (AirLLM-style)."""
import json
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from androidllm.engine import LayerStreamingEngine
from androidllm.models.llama import LlamaModel
from androidllm.quant import dequantize_packed, pack_int4, quantize_matrix
from androidllm.safetensors import read_header, read_tensor, write_safetensors

TMP = os.path.join(os.path.dirname(os.path.abspath(__file__)), "tmp_model")

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
    "tied": True,
    "name": "toy",
}


def build():
    os.makedirs(TMP, exist_ok=True)
    rng = np.random.default_rng(7)
    canon = CANON
    embed = (rng.standard_normal((canon["vocab"], canon["hidden"])) * 0.02).astype(np.float16)
    final_norm = rng.standard_normal(canon["hidden"]).astype(np.float16)
    write_safetensors(os.path.join(TMP, "embeddings.safetensors"), {"embed": embed})
    write_safetensors(os.path.join(TMP, "norms.safetensors"), {"final_norm": final_norm})
    quant_meta = {}
    for i in range(canon["layers"]):
        tensors = {}
        meta_layer = {}
        for base, outdim, indim in (("q", canon["hidden"], canon["hidden"]),
                                    ("k", canon["kv_heads"] * canon["head_dim"], canon["hidden"]),
                                    ("v", canon["kv_heads"] * canon["head_dim"], canon["hidden"]),
                                    ("o", canon["hidden"], canon["hidden"]),
                                    ("gate", canon["intermediate"], canon["hidden"]),
                                    ("up", canon["intermediate"], canon["hidden"]),
                                    ("down", canon["hidden"], canon["intermediate"])):
            w = (rng.standard_normal((outdim, indim)) * 0.05).astype(np.float32)
            q, scale = quantize_matrix(w, bits=4, block=32)
            tensors[base + ".q"] = pack_int4(q)
            tensors[base + ".scale"] = scale.astype(np.float16)
            meta_layer[base] = {"bits": 4, "block": 32, "out": outdim,
                                "in": indim, "packed": "U8"}
        tensors["input_layernorm.weight"] = rng.standard_normal(canon["hidden"]).astype(np.float16)
        tensors["post_attention_layernorm.weight"] = rng.standard_normal(canon["hidden"]).astype(np.float16)
        write_safetensors(os.path.join(TMP, "layer_%d.safetensors" % i), tensors)
        quant_meta[str(i)] = meta_layer
    manifest = {
        "format": "androidllm/v1",
        "name": "toy",
        "config": canon,
        "quant": {"attn_bits": 4, "mlp_bits": 8, "block": 32, "layers": quant_meta},
        "tied": True,
        "has_lm_head": False,
    }
    with open(os.path.join(TMP, "manifest.json"), "w", encoding="utf-8") as f:
        json.dump(manifest, f)


def load_all():
    canon = CANON
    embed = read_tensor(os.path.join(TMP, "embeddings.safetensors"), "embed")
    final_norm = read_tensor(os.path.join(TMP, "norms.safetensors"), "final_norm")
    model = LlamaModel(canon, embed, final_norm)
    with open(os.path.join(TMP, "manifest.json"), encoding="utf-8") as f:
        meta = json.load(f)["quant"]["layers"]
    layers = []
    for i in range(canon["layers"]):
        path = os.path.join(TMP, "layer_%d.safetensors" % i)
        header, _, _ = read_header(path)
        layer = {}
        for base, qm in meta[str(i)].items():
            packed = read_tensor(path, base + ".q", header)
            scale = read_tensor(path, base + ".scale", header)
            layer[base] = dequantize_packed(packed, scale, qm)
        layer["n_in"] = read_tensor(path, "input_layernorm.weight", header)
        layer["n_post"] = read_tensor(path, "post_attention_layernorm.weight", header)
        layers.append(layer)
    return model, layers


def run_reference(prompt):
    model, layers = load_all()
    kv = model.prepare_kv(CANON["max_len"])
    x = model.embed[prompt[0]].reshape(1, -1)
    for pos in range(len(prompt)):
        for l in range(CANON["layers"]):
            x = model.layer_forward(x, layers[l], kv[l], pos)
    return model.logits(x)


def run_streaming(prompt):
    engine = LayerStreamingEngine(TMP)
    kv = engine.model.prepare_kv(CANON["max_len"])
    x = engine.model.embed[prompt[0]].reshape(1, -1)
    for pos in range(len(prompt)):
        pending = engine._pool.submit(engine.load_layer, 0)
        for l in range(engine.n_layers):
            layer = pending.result()
            pending = (engine._pool.submit(engine.load_layer, l + 1)
                       if l + 1 < engine.n_layers else None)
            x = engine.model.layer_forward(x, layer, kv[l], pos)
    return engine.model.logits(x)


def test_streaming_equals_loaded():
    build()
    prompt = [3, 7, 11, 5]
    ref = run_reference(prompt)
    got = run_streaming(prompt)
    assert ref.shape == got.shape
    err = np.max(np.abs(ref - got)) / max(1e-6, np.max(np.abs(ref)))
    assert err < 1e-3, "max diff %g" % err
    print("streaming == all-layers-loaded OK")


def test_generate_runs():
    engine = LayerStreamingEngine(TMP)
    out = engine.generate([3, 7, 11], max_new_tokens=5, temperature=0.9, top_p=0.9)
    assert len(out) == 5 and all(0 <= t < CANON["vocab"] for t in out)
    print("generate OK:", out)


if __name__ == "__main__":
    test_streaming_equals_loaded()
    test_generate_runs()
