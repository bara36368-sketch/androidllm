import json
import os

_FAMILY_ALIASES = {
    "llama": "llama3",
    "llama3": "llama3",
    "llama3d": "llama3",
    "qwen2": "qwen",
    "qwen3": "qwen",
    "smollm2": "smollm",
    "mistral": "llama3",
    "phi3": "llama3",
    "phi-3": "llama3",
    "gemma": None,
    "gemma2": None,
    "gemma3": None,
    "qwen2_moe": None,
    "qwen3_moe": None,
    "qwen3d": None,
    "deepseek_v2": None,
}


def load_config(config_path):
    with open(config_path, encoding="utf-8") as f:
        cfg = json.load(f)
    if cfg.get("model_type") not in _FAMILY_ALIASES:
        raise ValueError(
            "unsupported model_type {} (supported: llama, qwen2, qwen3, "
            "smollm2, mistral, phi3)".format(cfg.get("model_type"))
        )
    hidden = cfg.get("hidden_size") or cfg.get("d_model")
    heads = cfg.get("num_attention_heads") or cfg.get("n_head")
    kv_heads = cfg.get("num_key_value_heads") or cfg.get("n_kv_heads") or heads
    head_dim = cfg.get("head_dim") or (hidden // heads)
    intermediate = cfg.get("intermediate_size") or cfg.get("ffn_dim") or hidden * 4
    canon = {
        "model_type": cfg.get("model_type"),
        "hidden": hidden,
        "layers": cfg.get("num_hidden_layers") or cfg.get("n_layer"),
        "heads": heads,
        "kv_heads": kv_heads,
        "head_dim": head_dim,
        "intermediate": intermediate,
        "vocab": cfg.get("vocab_size") or cfg.get("n_vocab"),
        "rope_theta": float(cfg.get("rope_theta", 10000.0)),
        "rms_eps": float(cfg.get("rms_norm_eps", 1e-6)),
        "max_len": int(cfg.get("max_position_embeddings", 4096)),
        "tied": bool(cfg.get("tie_word_embeddings", True)),
        "name": cfg.get("_name_or_path", os.path.basename(os.path.dirname(config_path))),
    }
    return canon


def family(canon):
    return _FAMILY_ALIASES.get(canon["model_type"])


def rope_base(family_name):
    return family_name
