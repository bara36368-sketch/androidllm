# Changelog

All notable changes to this project are documented here.
The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### Added
- `modelpicker list` CLI: full catalog with RAM-tier labels, `--tier` filter
  (e.g. `4-8` for 4-8 GB RAM) and per-model fit flags vs `--specs`
- `/v1/token-count` endpoint (OpenAI-style token counting for prompt budgets)
- RotorQuant/PlanarQuant-style compressed KV cache (`androidllm/kv_cache.py`):
  block-diagonal Givens (planar) / quaternion (iso) rotations + Lloyd-Max
  codebooks with deferred quantization (fp16 prefill staging, quantized on
  decode insert), matching the llama.cpp planar3/iso3 layout (~5.1x smaller
  KV); enabled with `ANDROIDLLM_KV_BITS=3|4` and `ANDROIDLLM_KV_ROT=planar|iso`

## [0.2.1] - 2026-08-02

### Added
- Model catalog now covers 5-16 GB RAM devices: Qwen2.5-3B, Qwen3-4B,
  Qwen2.5-7B, Mistral-7B-v0.3, Qwen3-8B, Qwen3-14B, Qwen2.5-14B,
  Mistral-Small-24B, Qwen2.5-32B, Qwen3-32B (auto-gated per device RAM/disk).

## [0.2.0] - 2026-08-02

### Added
- Bearer API key auth on all `/v1/*` routes and `/stats` (`--api-key` flag or
  `ANDROIDLLM_API_KEY`); a random key is generated and persisted to
  `~/.androidllm/api_key` on first run.
- `GET /v1/keys` endpoint returning the active API key and base URL.
- `py.typed` marker for type checkers.
- GitHub Actions CI: ruff lint, pytest on Python 3.9/3.11/3.13, wheel build.
- `CONTRIBUTING.md` with project design rules.

### Changed
- README gains an API key auth section and quickstart curl example.
- Project metadata modernized in `pyproject.toml` (SPDX license, classifiers,
  URLs, ruff/pytest config, `dev` extra).

## [0.1.0] - 2026-07-19

### Added
- Layer-streaming Llama/Qwen inference engine (AirLLM-style), zero PyTorch.
- Block-wise 4/8-bit weight quantization (`androidllm/quant.py`).
- Safetensors reader/writer, HF tokenizer conversion (byte-level BPE + Jinja
  subset chat templates).
- OpenAI-compatible HTTP server (`androidllm-serve`) with SSE streaming.
- Model sharding CLI (`androidllm-shard`) with optional `huggingface_hub`.
- ARM NEON fp16 matmul kernel (`src/androidllm_neon.c`) with numpy fallback.
- PyO3 Rust accelerator (`androidllm_rs`): fused layer forward, head logits,
  and sampling; bit-equal to the numpy path.
- GGUF reader + conversion to shards.
- Speculative decoding, batching scheduler, JSON grammar constrained decoding.
- Termux setup and model-management scripts (`scripts/`).
