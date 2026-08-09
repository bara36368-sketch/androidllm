import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from androidllm.kv_cache import (
    QuantizedKVCache,
    _pack_bits,
    _unpack_bits,
    build_rotation,
    givens_rotation,
    quaternion_rotation,
)


def _attn_focus(kq, kcache, bits, rotation, n=64):
    rng = np.random.default_rng(1)
    kv = QuantizedKVCache(n + 4, 8, 32, bits=bits, rotation=rotation)
    ks = rng.standard_normal((n, 8, 32)).astype(np.float16)
    for p in range(n):
        kv.stage_write(p, ks[p], rng.standard_normal((8, 32)).astype(np.float16))
    kv.finalize(n)
    kq2, _ = kv.frames(n)
    s = np.sqrt(32.0)
    corr = []
    for h in range(8):
        lf = (ks[:, h].astype(np.float32) @ kq[h].astype(np.float32)) / s
        lq = (kq2[:, h].astype(np.float32) @ kq[h].astype(np.float32)) / s
        corr.append(np.corrcoef(lf, lq)[0, 1])
    return float(np.mean(corr))


def test_rotations_orthogonal():
    for dim in (32, 64, 128):
        for R in (givens_rotation(dim), quaternion_rotation(dim)):
            ident = R @ R.T
            assert np.abs(ident - np.eye(dim)).max() < 1e-5


def test_pack_roundtrip():
    rng = np.random.default_rng(0)
    for bits in (2, 3, 4, 5):
        c = rng.integers(0, 1 << bits, size=(2, 3, 64))
        p = _pack_bits(c, bits)
        u = _unpack_bits(p, bits, 64)
        assert np.array_equal(c, u), f"bits={bits}"


def test_write_frames_roundtrip():
    rng = np.random.default_rng(0)
    kv = QuantizedKVCache(16, 4, 32, bits=3)
    x = rng.standard_normal((4, 32)).astype(np.float16)
    kv.write(0, x[0], x[1])
    kv.write(5, x[2], x[3])
    k, v = kv.frames(6)
    assert k.shape == (6, 4, 32) and v.shape == (6, 4, 32)
    # dequant is lossy but must stay near fp16 magnitude
    assert np.abs(k[0].astype(np.float32) - x[0].astype(np.float32)).mean() < 0.25
    assert np.abs(v[5].astype(np.float32) - x[3].astype(np.float32)).mean() < 0.25


def test_deferred_quantization_flow():
    """Prefill staging + finalize must produce the quantized store, and
    frames() must return compressed data after finalize."""
    rng = np.random.default_rng(3)
    kv = QuantizedKVCache(64, 8, 32, bits=3)
    x = rng.standard_normal((64, 8, 32)).astype(np.float16)
    for p in range(64):
        kv.stage_write(p, x[p], x[p])
    assert not kv._finalized
    assert kv.frames(64)[0].shape == (64, 8, 32)
    kv.finalize(64)
    assert kv._finalized
    k, v = kv.frames(64)
    assert k.shape == (64, 8, 32)
    # quantized store must be smaller than a full fp16 cache
    fp16 = 64 * 8 * 32 * 2
    assert kv.nbytes < fp16, f"{kv.nbytes} vs {fp16}"
    # re-finalize must be a no-op
    kv.finalize(64)
    kv.reset()
    assert kv._n == 0 and not kv._finalized


def test_attention_survives_compression():
    rng = np.random.default_rng(1)
    kq = rng.standard_normal((8, 32)).astype(np.float16)
    # planar3 and iso3 both must keep attention highly correlated with fp16
    for rotation in ("planar", "iso"):
        corr = _attn_focus(kq, None, 3, rotation)
        assert corr > 0.95, f"{rotation} corr={corr:.4f}"


def test_compression_ratio():
    # per (token, kv_head): bits*dim/8 codes + 2-byte norm, x2 for K+V
    # head_dim=128 -> 50 bytes vs 256 fp16 = 5.12x (matches llama.cpp planar3)
    kv = QuantizedKVCache(4096, 8, 128, bits=3)
    fp16 = 4096 * 8 * 128 * 2 * 2  # K and V buffers
    assert kv.nbytes < fp16 / 5, f"{kv.nbytes} vs {fp16}"
    assert kv.nbytes > fp16 / 6  # norm overhead keeps it just under 5.33x
    kv4 = QuantizedKVCache(4096, 8, 128, bits=4)
    assert kv4.nbytes < fp16 / 3.8


def test_env_knob_prepare_kv():
    from androidllm.models.llama import LlamaModel

    canon = {"hidden": 32, "heads": 8, "kv_heads": 8, "head_dim": 32,
             "rope_theta": 10000.0, "rms_eps": 1e-6, "max_len": 128,
             "layers": 2}
    model = LlamaModel(canon, None, None, None)
    kv_plain = model.prepare_kv(32)
    assert not isinstance(kv_plain[0], QuantizedKVCache)
    os.environ["ANDROIDLLM_KV_BITS"] = "3"
    try:
        kv_q = model.prepare_kv(32)
        assert len(kv_q) == 2
        assert isinstance(kv_q[0], QuantizedKVCache)
        assert kv_q[0].bits == 3
    finally:
        os.environ.pop("ANDROIDLLM_KV_BITS", None)


def test_engine_generate_with_compressed_kv():
    """Full engine prefill -> finalize -> decode with ANDROIDLLM_KV_BITS=3.
    Must run without touching the Rust path and finish in reasonable time."""
    from androidllm.engine import LayerStreamingEngine

    from test_streaming import TMP, build
    build()
    os.environ["ANDROIDLLM_KV_BITS"] = "3"
    os.environ["ANDROIDLLM_THREADS"] = "2"
    try:
        e = LayerStreamingEngine(TMP)
        assert isinstance(e.model.prepare_kv(16)[0], QuantizedKVCache)
        # prefill of the toy model with a 12-token prompt then generate
        prompt = list(range(12))
        out = e.generate(prompt, max_new_tokens=8, temperature=0.0)
        assert len(out) > 0, "compressed-KV generation produced no tokens"
        assert all(isinstance(t, int) for t in out)
    finally:
        os.environ.pop("ANDROIDLLM_KV_BITS", None)
        os.environ.pop("ANDROIDLLM_THREADS", None)