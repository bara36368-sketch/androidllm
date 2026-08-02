import argparse
import glob
import json
import os

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


def shard_from_tensors(source_tensors, canon, out, attn_bits=4, mlp_bits=8,
                       block=64, embed_bits=8):
    """Quantize + write shards from a dict of HF-style tensor names. Shared
    by the safetensors and GGUF import paths."""
    layout = {}
    for name, arr in source_tensors.items():
        kind, key = classify(name)
        if kind == "other":
            continue
        layout[name] = (kind, key, arr)

    os.makedirs(out, exist_ok=True)
    quant_meta = {}
    embed_quant = {}
    lm_head_quant = {}
    global_tensors = {}
    layer_tensors = {}

    for _name, (kind, key, arr) in layout.items():
        w = arr() if callable(arr) else arr
        w = np.ascontiguousarray(w, dtype=np.float32)
        if kind == "global":
            if key == "embed" and embed_bits:
                q, scale = quantize_matrix(w, bits=embed_bits, block=w.shape[1])
                global_tensors["embed.q"] = pack_int4(q) if embed_bits == 4 else q.astype(np.int8)
                global_tensors["embed.scale"] = scale.astype(np.float16)
                embed_quant = {"bits": embed_bits, "block": int(w.shape[1]),
                               "out": int(w.shape[0]), "in": int(w.shape[1])}
            elif key == "lm_head" and embed_bits:
                q, scale = quantize_matrix(w, bits=embed_bits, block=w.shape[1])
                global_tensors["lm_head.q"] = pack_int4(q) if embed_bits == 4 else q.astype(np.int8)
                global_tensors["lm_head.scale"] = scale.astype(np.float16)
                lm_head_quant = {"bits": embed_bits, "block": int(w.shape[1]),
                                 "out": int(w.shape[0]), "in": int(w.shape[1])}
            else:
                global_tensors[key] = w.astype(np.float16)
            continue
        layer = key[0]
        bucket = layer_tensors.setdefault(layer, {})
        if kind == "norm":
            bucket.setdefault("norms", {})[key[1]] = w.astype(np.float16)
            continue
        bits = attn_bits if kind == "attn" else mlp_bits
        q_raw, scale = quantize_matrix(w, bits=bits, block=block)
        base = key[1].split("_")[0]
        q = pack_int4(q_raw) if bits == 4 else q_raw.astype(np.int8)
        bucket.setdefault("weights", {})[_quant_name(base)] = q
        bucket["weights"][_scale_name(base)] = scale.astype(np.float16)
        quant_meta.setdefault(str(layer), {})
        quant_meta[str(layer)][base] = {
            "bits": bits, "block": block, "out": int(q_raw.shape[0]),
            "in": int(q_raw.shape[1]), "in_real": int(w.shape[1]),
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
                      {k: v for k, v in global_tensors.items() if k.startswith("embed")})
    write_safetensors(os.path.join(out, "norms.safetensors"),
                      {"final_norm": global_tensors["final_norm"]})
    if "lm_head" in global_tensors or lm_head_quant:
        write_safetensors(os.path.join(out, "lm_head.safetensors"),
                          {k: v for k, v in global_tensors.items() if k.startswith("lm_head")})

    manifest = {
        "format": "androidllm/v1",
        "name": canon["name"],
        "config": canon,
        "quant": {"attn_bits": attn_bits, "mlp_bits": mlp_bits, "block": block,
                  "layers": quant_meta},
        "embed_quant": embed_quant,
        "lm_head_quant": lm_head_quant,
        "tied": canon["tied"],
        "has_lm_head": bool(lm_head_quant) or "lm_head" in global_tensors,
    }
    with open(os.path.join(out, "manifest.json"), "w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2)
    return manifest


def shard_model(source, out, attn_bits=4, mlp_bits=8, block=64, embed_bits=8):
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

    tok_json = os.path.join(source, "tokenizer.json")
    if os.path.exists(tok_json):
        from .tokenizer import convert_hf_tokenizer
        convert_hf_tokenizer(tok_json, out,
                             tokenizer_config_path=os.path.join(source, "tokenizer_config.json"))

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

    tensors = {name: (lambda n=name: read_matrix(n))
               for name in layout if classify(name)[0] != "other"}
    return shard_from_tensors(tensors, canon, out, attn_bits, mlp_bits,
                              block, embed_bits)


def main():
    ap = argparse.ArgumentParser(description="Shard an HF model into androidllm layer files")
    ap.add_argument("--source", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--attn-bits", type=int, default=4)
    ap.add_argument("--mlp-bits", type=int, default=8)
    ap.add_argument("--block", type=int, default=64)
    ap.add_argument("--embed-bits", type=int, default=8,
                    help="quantize the embedding table (and lm_head if untied); 0 = keep f16")
    args = ap.parse_args()
    shard_model(args.source, args.out, args.attn_bits, args.mlp_bits,
                args.block, args.embed_bits)
    print(f"sharded to {args.out}")


if __name__ == "__main__":
    main()
