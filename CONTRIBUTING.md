# Contributing to androidllm

Thanks for your interest! This project is small on purpose: a numpy-only
layer-streaming LLM runtime that fits on a 4 GB Android phone. Keep that
constraint in mind with every change.

## Development setup

```bash
python -m venv .venv
source .venv/bin/activate        # Windows: .venv\Scripts\activate
pip install -e ".[dev]"
```

The `dev` extra installs `pytest` and `ruff`. The runtime only requires
`numpy`; `huggingface_hub` is optional (`.[hub]`) and only used by
`androidllm.shard` for downloads.

## Running the checks

```bash
ruff check .           # lint (E, F, I, UP, B, SIM)
pytest                 # full test suite
```

Both run in CI on every push/PR — make them pass locally first. The tests
are fully offline (they build tiny toy models on the fly), so no network
or model downloads are needed.

## What belongs where

| Concern | File |
|---|---|
| HF config → canonical config | `androidllm/config.py` |
| Layer streaming engine | `androidllm/engine.py` |
| Llama/Qwen forward pass | `androidllm/models/llama.py` |
| FP16 matmul (NEON or numpy) | `androidllm/neon.py` |
| Block-wise quantization | `androidllm/quant.py` |
| Safetensors reader/writer | `androidllm/safetensors.py` |
| OpenAI-compatible HTTP server | `androidllm/serve.py` |
| HF → shards + manifest | `androidllm/shard.py` |
| Byte-level BPE + templates | `androidllm/tokenizer.py` |
| GGUF conversion | `androidllm/gguf.py` |

## Design rules

1. **No PyTorch, ever.** The whole point is removing the `torch` dependency
   (there are no aarch64 wheels for Termux's Bionic libc). NumPy only.
2. **The Rust accelerator must be optional.** Anything in
   `androidllm_rs/` is a strict optimization: same f16 weights, f32
   accumulation, bit-equal matmuls. The numpy path is the source of truth
   and must keep working if the module is missing.
3. **Mobile-first.** New code should not assume fast CPUs, big RAM, or a
   filesystem that can hold a full model. Streaming layers is the core
   trick — don't load whole models into RAM.
4. **Keep it offline-testable.** Tests build toy models; they must never
   download from the network.

## Commit conventions

- Imperative, lowercase-ish subject: `fix: handle empty prompt in serve`,
  `feat: add gguf conversion`.
- Keep changes focused; a PR should do one thing.
- Run `ruff check .` and `pytest` before pushing.

## Releasing

Version lives in `pyproject.toml`. Bump it, update `CHANGELOG.md`, tag the
release. The CI build job produces the sdist + wheel artifact.
