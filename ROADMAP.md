# androidllm roadmap

Status: [x] done / [ ] open. Effort S/M/L, impact (what it buys you) ranked.
Research sources: whisper.cpp Termux guides, Arm NEON matmul literature,
AirLLM design, NeonFlux kernel notes.

## Current state (v0.1 + switcher)

Layer-streaming inference (AirLLM scheme, zero torch), block-wise 4/8-bit
quant, BPE tokenizer + chat templates, OpenAI-compatible server with SSE,
NEON fp16 kernel (single-thread), Termux scripts, one-at-a-time model
switcher (/model), bot integration as localhost provider.

## A. Performance (biggest wins first)

1. [x] NEON kernel — single-thread vectorized fp16 dot product.
2. [ ] **Multithreaded NEON kernel** (pthreads over the n dimension).
   G85 = 2x A75 + 6x A55. Compute is the bottleneck at m=1 gemv, and
   output rows are independent -> trivial tiling. Expect 3-5x on the
   A75 cores (NeonFlux got ~5x multithread; OpenMP not needed, pthreads
   are lighter and always in Termux).
   Effort S. Impact: biggest single speedup available.
3. [ ] **Keep-N layer cache** (AirLLM partial-offload trick). Hold 2-4
   layers resident (fp16, ~110 MB/layer at 1.5B) so generation does not
   re-read them from eMMC every token. With N=3 you skip ~10% of disk
   traffic; tune against RAM. Config: ANDROIDLLM_KEEP_LAYERS.
   Effort S. Impact: faster prompt pass and warmer turns.
4. [ ] Register-blocked microkernel (8x4 or 8x8) with L1 cache blocking
   and B-packing (NeonFlux/Arm blog: ~10x over scalar, 2-3x over naive
   vector loop). The A75 data engine wants sustained fmla lanes; our
   current per-output dot loop leaves FMA issue slots on the table.
   Effort M. Impact: 1.5-2x over current single-thread kernel.
5. [ ] fp16 -> int8 KV cache (per-channel scale). Shrinks resident KV
   ~2x and frees headroom for longer contexts; Qwen3 thinking mode
   burns context fast.
   Effort M. Impact: longer conversations on 4 GB.
6. [ ] mmap-backed layer reads instead of read() (safetensors already
   supports memmap; page cache then overlaps disk and compute without
   the prefetch thread's copies).
   Effort S. Impact: less RAM churn during streaming.
7. [ ] Benchmark harness: t/s, ms/token split (load vs compute), model
   catalog table refresh. Grounds every optimization decision.
   Effort S.

## B. Voice (ASR + TTS) — researched path

8. [ ] **On-device ASR via whisper.cpp.** The known-good Termux recipe:
   - `pkg install git cmake clang make ffmpeg`
   - clone whisper.cpp, build `-DGGML_NO_OPENMP=ON` (stable on Termux)
   - `ggml-base.en.bin` (~75 MB) fits the 4 GB phone; small.en (~466 MB)
     optional
   - Telegram voice messages arrive as .ogg(opus) -> ffmpeg to 16 kHz
     mono wav -> `whisper-cli -m ... -f in.wav -otxt -of out` -> text
   - Phone mic: `termux-microphone-record`, speech out: `termux-tts-speak`
   Deliverables: `scripts/asr.sh` (wrapper) + wire the bot's voice
   handler (cyberdeck_bot.py:3761 currently replies "not available") to
   transcribe -> feed the transcript to androidllm -> reply; optionally
   TTS the reply back.
   Effort M. Impact: the phone becomes a spoken chatbot end to end.
9. [ ] Language switch for whisper (base.en -> multilingual small for
   non-English; `-l auto`).
   Effort S.

## C. Correctness / quality

10. [ ] Asymmetric block quant (per-block min/max with fp16 offset) —
    AirLLM-quality at same bits, notably better on activations-heavy
    layers. Falls out of the existing quant.py shape.
    Effort M.
11. [ ] Sampling upgrades: repeat penalty, frequency penalty, min-p
    (Qwen3 thinking wants min-p). Server params passthrough already
    exists; add to engine._sample.
    Effort S-M.
12. [x] EOS echo / stream cleanup: thinking-tag stripping (<think>) for
    Qwen3-style models — server-side, stream-safe (ANDROIDLLM_STRIP_THINK).
13. [x] Session KV persistence (vLLM-style prefix cache): keep KV across
    turns so repeated prompts don't re-stream all layers every call.
    (ANDROIDLLM_PREFIX_KV; prefix_tokens/context_used now in /stats.)

## D. Bot / product

14. [ ] `/voice` command: record 60s via termux-api, transcribe, answer
    with voice (termux-tts). Full offline voice loop (B8).
    Note: scripts/asr.sh (whisper.cpp wrapper) exists; bot voice-handler
    wiring is the remaining piece.
15. [ ] `/bench` command: on-device tokens/sec for the active model.
16. [ ] Model download progress notifications (already backgrounded;
    add /model % status via current_model.json fields).
17. [ ] Auto-switch: pin androidllm provider to the current model in
    /provider output (model field already auto).
18. [ ] Nudge: show androidllm speed on /stats so cloud vs local cost
    comparison is visible.
    Note: /status now shows context bar, battery/paused state, speed,
    prefix reuse — wired from the serve /stats endpoint.

## E. Portability / packaging

19. [ ] pip package (`pip install .` already works; add pyproject entry
    points for asr.sh, benchmark).
20. [ ] Desktop CI: run the 3 test suites on aarch64-emulated runner?
    Low value vs on-phone testing; skip unless publishing.
21. [ ] Dockerfile for desktop dev (same shard->serve flow, no Termux).

## Dependency order (suggested)

1. A2 multithread kernel (S, no deps) — biggest speed win.
2. A3 keep-N cache (S, independent).
3. A4 microkernel (M, builds on A2's threading model).
4. B8 ASR (M, independent; biggest capability win).
5. C10 asymmetric quant (M, independent).
6. C11 sampling upgrades (S).
7. C13 session KV (M, builds on engine internals).
8. A5 int8 KV (M, builds on C10 scales).

## Ideas to research further (uncertain payoff)

- Speculative decoding with a tiny draft model on the same phone —
  draft adds a second model's disk traffic; likely net-negative on
  eMMC, positive only with keep-N cache resident. Test, don't assume.
- Vulkan/Turnip GPU offload: Adreno driver path exists but is
  Snapdragon-focused; Mali-G52 on G85 has no working Mesa path today.
  Revisit only if a Turnip-style driver ships for Mali.
- Int8 (not fp16) NEON dotprod kernel: A75 lacks dotprod instructions
  (ARMv8.2 optional, A76+ have them), so no win on G85 hardware.
- Streaming longer than 4K ctx: KV stays resident but layer re-reads
  scale with prompt length; consider prompt batching or KV compression.
