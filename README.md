# androidllm

Layer-streaming LLM inference for Android/Termux, modeled on
[AirLLM](https://github.com/lyogavin/airllm) but with **zero PyTorch**:
pure Python + numpy, with an optional ARM NEON fp16 matmul kernel.

Runs on a 4 GB RAM / Helio G85 phone in plain Termux (no proot, no distro):
the same trick AirLLM uses on consumer GPUs - keep only ONE transformer
layer in RAM at a time, stream it from disk, compute, then free it.

## Why no torch

There are no aarch64 `torch` manylinux wheels for Termux's Bionic libc, so
AirLLM forces proot-distro Ubuntu. androidllm removes torch entirely, which
removes the proot requirement too. numpy ships on Termux as a proper package.

## How it works (AirLLM parity)

| AirLLM concept | androidllm |
|---|---|
| layer sharding to safetensors per layer | `androidllm/shard.py` writes `layer_N.safetensors` + `manifest.json` |
| block-wise weight-only 4/8-bit quant, dequant to fp16 in RAM | `androidllm/quant.py` (sym, per-block scale, block 64) |
| stream load -> compute -> `clean_memory()` | `LayerStreamingEngine` in `androidllm/engine.py` |
| ThreadPoolExecutor prefetch of next layer | `engine._pool`, next layer loaded while current computes |
| resident KV cache | `LlamaModel.prepare_kv` (fp16, ~2 KB/token for 1.5B) |
| model forward | `androidllm/models/llama.py` (Llama/Qwen family: GQA, RoPE, SwiGLU, RMSNorm) |
| HF tokenizer | `androidllm/tokenizer.py`: byte-level BPE + chat templates (Jinja subset) |
| own safetensors reader (no `safetensors` pkg needed) | `androidllm/safetensors.py` |

## Pipeline

1. **Shard** an HF model once (on desktop or phone):

```
python -m androidllm.shard --source Qwen/Qwen2.5-1.5B-Instruct --out models/qwen15
```

Produces: `layer_N.safetensors` (quantized weights + scales + norms),
`embeddings.safetensors`, `norms.safetensors`, optional `lm_head.safetensors`,
`manifest.json`, and the converted tokenizer files
(`vocab.txt`, `merges.txt`, `special_tokens.json`, `template.txt`).

`tokenizer.json`/`tokenizer_config.json` in the source dir are converted
automatically. Source tensors may be fp16/fp32/bf16 and may span
`model-0000X-of-0000Y.safetensors` files.

2. **Serve** (OpenAI-compatible):

```
androidllm-serve --model models/qwen15 --port 8080
```

Endpoints: `/v1/completions`, `/v1/chat/completions`, `/v1/models`, `/health`.
Both completion endpoints accept `"stream": true` and return OpenAI-style SSE
(`data:` frames per token, `data: [DONE]` at the end). Chat uses the model's
own chat template (Qwen/llama3/smollm styles).

3. **Optional NEON kernel** (much faster matmul, ~2x):

```
bash scripts/build_neon.sh    # needs `clang` from Termux
```

Loads `libandroidllm_neon.so` from the repo root or `~/.androidllm`;
falls back to numpy automatically. See `src/androidllm_neon.c`.

4. **Optional Rust accelerator** (`androidllm_rs`, PyO3):

```
pkg install rust            # Termux
python -m pip install maturin
bash scripts/build_rust.sh  # builds the wheel and pip-installs it
```

Fuses the per-layer forward (7 matmuls + rope + GQA attention + swiglu +
rms_norm), the head matmul, and sampling into native calls. Semantics match
the numpy path (same f16 weights, f32 accumulation); outputs are bit-equal
at the matmul level and differ only in f16 rounding at the very end.
On the toy model this gives ~2x on layer_forward; larger models (hidden
1024+) benefit more since matmuls dominate. Falls back to numpy when the
module is missing. See `androidllm_rs/`.

## Model fit for a 4 GB phone

| model | params | Q4 disk | notes |
|---|---|---|---|
| Qwen2.5-1.5B-Instruct | 1.5B | ~1.1 GB | best overall, great tool use |
| SmolLM2-1.7B | 1.7B | ~1.06 GB | English-only |
| Qwen3-1.7B | 1.7B | ~1.28 GB | thinking mode |

3B+ models work but leave little headroom next to the bot processes.
Embedding table stays in RAM in fp16 (Qwen2.5-1.5B: ~0.5 GB).

Performance on G85-class silicon (A75 @ 2.0 GHz, eMMC):
roughly 1.5-4 s/token with the numpy fallback, faster with NEON.
Like AirLLM, every generated token re-streams all layer files, so disk
speed matters.

## Notes / limitations

- Streaming is single-request at a time (engine has one prefetch worker).
- `run_streaming` (a fresh `generate` call) restarts the KV cache; long
  contexts mean re-reading the prompt each call.
- The Jinja chat-template subset covers the standard Llama/Qwen templates
  (`if/elif/else`, `for`, `set`, `loop.last`, `|trim`, string `+`).
- Tokenizer is a from-scratch byte-level BPE; it matches HF output for
  standard vocabularies (vocab entries must include all merge results).

## Tests

```
python tests/test_quant.py      # quant + int4 pack roundtrip
python tests/test_tokenizer.py  # BPE, specials, templates, HF convert
python tests/test_streaming.py  # engine == all-layers-loaded, generate
```

The key invariant: `test_streaming.py` proves streaming through layer files
produces identical logits to loading the whole model into RAM.

## Project layout

```
androidllm/          package (numpy only)
  config.py          HF config.json -> canonical config
  engine.py          LayerStreamingEngine (AirLLM-style loop + prefetch)
  models/llama.py    Llama/Qwen family forward pass (+ optional rust path)
  neon.py            fp16 matmul: NEON lib or numpy fallback
  quant.py           block-wise 4/8-bit quant, pack, dequant
  safetensors.py     minimal safetensors reader/writer
  serve.py           OpenAI-compatible HTTP server
  shard.py           HF model dir -> layer shards + manifest
  tokenizer.py       byte-level BPE + chat templates + HF convert
src/androidllm_neon.c   ARM NEON fp16 kernel
androidllm_rs/          PyO3 Rust accelerator (layer_forward/head_logits/sample)
scripts/                setup_termux.sh, shard_model.sh, switch_model.sh,
                        build_neon.sh, build_rust.sh
tests/                  roundtrip + equality tests
```

## Switching models (one at a time)

Only one model is served at a time. A `current_model.json` state file in
`~/androidllm` records the active shard; `runner.py` (which supervises
`androidllm-serve`) restarts the server on the model it points to.

```
bash scripts/switch_model.sh qwen15   # switch to an already-sharded model
bash scripts/shard_model.sh Qwen/Qwen3-1.7B-Instruct qwen3   # shard a new one first
```

The Telegram bot's `/model` command lists the recommended models (qwen15,
smollm2, qwen3) with shard status, switches between them, and auto-downloads
+ shards any you pick that isn't sharded yet.
