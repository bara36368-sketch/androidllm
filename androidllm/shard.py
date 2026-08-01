import argparse
import glob
import json
import os
import sys

import numpy as np

from .config import load_config
from .quant import pack_int4, quantize_matrix
from .safetensors import read_header

_BF16_LAYER_WEIGHTS = {
    "embed_tokens.weight": "embed",
    "norm.weight": "final_norm",
    "lm_head.weight": "lm_head",
}
_ATTN_PROJ = ("self_attn.q_proj.weight", "self_attn.k_proj.weight",
              "self_attn.v_proj.weight", "self_attn.o_proj.weight")
_MLP_PROJ = ("mlp.gate_proj.weight", "mlp.up_proj.weight", "mlp.down_proj.weight")
_NORMS = ("input_layernorm.weight", "post_attention_layernorm.weight")


def bf16_to_f32(u16):
    return (u16.astype(np.uint32) << 16).view(np.float32)


def to_f32(arr, dtype_name):
    if dtype_name == "BF16":
        return bf16_to_f32(arr)
    if dtype_name == "F16":
        return arr.astype(np.float32)
    return arr.astype(np.float32)


def _strip_prefix(name):
    if name.startswith("model."):
        return name[len("model."):]
    return name


def classify(name):
    n = _strip_prefix(name)
    if n in _BF16_LAYER_WEIGHTS:
        return "global", _BF16_LAYER_WEIGHTS[n]
    if n.startswith("layers."):
        rest = n[len("layers."):]
        dot = rest.index(".")
        layer = int(rest[:dot])
        sub = rest[dot + 1:]
        if sub in _ATTN_PROJ:
            return "attn", (layer, sub[len("self_attn."):])
        if sub in _MLP_PROJ:
            return "mlp", (layer, sub[len("mlp."):])
        if sub in _NORMS:
            return "norm", (layer, sub)
    return "other", None


def _quant_name(base):
    return base + ".q"


def _scale_name(base):
    return base + ".scale"


def shard_model(source, out, attn_bits=4, mlp_bits=8, block=64):
    config_path = os.path.join(source, "config.json")
    if not os.path.exists(config_path):
        raise FileNotFoundError(f"no config.json in {source}")
    canon = load_config(config_path)
    files = sorted(glob.glob(os.path.join(source, "*.safetensors")))
    if not files:
        raise FileNotFoundError(f"no safetensors files in {source}")

    layout = {}
    file_meta = {}
    for f in files:
        header, data_start, _ = read_header(f)
        file_meta[f] = data_start
        for name, meta in header.items():
            kind, key = classify(name)
            layout[name] = (f, meta["dtype"], tuple(meta["shape"]),
                            meta["data_offsets"], kind, key)

    os.makedirs(out, exist_ok=True)
    tok_json = os.path.join(source, "tokenizer.json")
    if os.path.exists(tok_json):
        from .tokenizer import convert_hf_tokenizer
        convert_hf_tokenizer(tok_json, out,
                             tokenizer_config_path=os.path.join(source, "tokenizer_config.json"))
    quant_meta = {}
    global_tensors = {}
    layer_tensors = {}

    def read_matrix(name):
        f, dtype_name, shape, (a, b), _, _ = layout[name]
        nbytes = b - a
        with open(f, "rb") as fh:
            fh.seek(file_meta[f] + a)
            blob = fh.read(nbytes)
        if dtype_name == "BF16":
            return bf16_to_f32(np.frombuffer(blob, dtype=np.uint16).reshape(shape))
        np_dtype = np.float16 if dtype_name == "F16" else np.float32
        return np.frombuffer(blob, dtype=np_dtype).reshape(shape).astype(np.float32)

    for name, (f, dtype_name, shape, offs, kind, key) in layout.items():
        if kind == "other":
            continue
        if kind == "global":
            w = read_matrix(name)
            global_tensors[key] = w.astype(np.float16)
            continue
        layer = key[0]
        bucket = layer_tensors.setdefault(layer, {})
        if kind == "norm":
            w = read_matrix(name)
            bucket.setdefault("norms", {})[key[1]] = w.astype(np.float16)
            continue
        w = read_matrix(name)
        bits = attn_bits if kind == "attn" else mlp_bits
        q, scale = quantize_matrix(w, bits=bits, block=block)
        base = key[1].split("_")[0]
        bucket.setdefault("weights", {})[_quant_name(base)] = (pack_int4(q) if bits == 4 else q.astype(np.int8))
        bucket["weights"][_scale_name(base)] = scale.astype(np.float16)
        quant_meta.setdefault(str(layer), {})
        quant_meta[str(layer)][base] = {
            "bits": bits, "block": block, "out": int(q.shape[0]),
            "in": int(q.shape[1]), "in_real": int(w.shape[1]),
            "packed": "U8" if bits == 4 else "I8",
        }
        del q, scale, w

    for layer_id in sorted(layer_tensors):
        tensors = {}
        tensors.update(layer_tensors[layer_id].get("weights", {}))
        tensors.update(layer_tensors[layer_id].get("norms", {}))
        from .safetensors import write_safetensors
        write_safetensors(os.path.join(out, f"layer_{layer_id}.safetensors"), tensors)
        del layer_tensors[layer_id]

    write_safetensors(os.path.join(out, "embeddings.safetensors"),
                      {"embed": global_tensors["embed"]})
    write_safetensors(os.path.join(out, "norms.safetensors"),
                      {"final_norm": global_tensors["final_norm"]})
    if "lm_head" in global_tensors:
        write_safetensors(os.path.join(out, "lm_head.safetensors"),
                          {"lm_head": global_tensors["lm_head"]})

    manifest = {
        "format": "androidllm/v1",
        "name": canon["name"],
        "config": canon,
        "quant": {"attn_bits": attn_bits, "mlp_bits": mlp_bits, "block": block,
                  "layers": quant_meta},
        "tied": canon["tied"],
        "has_lm_head": "lm_head" in global_tensors,
    }
    with open(os.path.join(out, "manifest.json"), "w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2)
    return manifest


def main():
    ap = argparse.ArgumentParser(description="Shard an HF model into androidllm layer files")
    ap.add_argument("--source", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--attn-bits", type=int, default=4)
    ap.add_argument("--mlp-bits", type=int, default=8)
    ap.add_argument("--block", type=int, default=64)
    args = ap.parse_args()
    shard_model(args.source, args.out, args.attn_bits, args.mlp_bits, args.block)
    print(f"sharded to {args.out}")


if __name__ == "__main__":
    main()
