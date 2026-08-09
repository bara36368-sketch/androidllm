"""Compressed KV cache for androidllm.

Reverse-engineered from the RotorQuant/PlanarQuant/IsoQuant KV cache
compression (llama.cpp cache types planar3/iso3, "5.3x faster prefill,
drop-in llama.cpp integration"):

  quantize:  rotate head-dim vector by fixed block-diagonal rotation
             (2D Givens pairs for planar, 4D quaternion blocks for iso)
             -> normalize by the vector's max-abs (stored as fp16 norm)
             -> nearest Lloyd-Max centroid per coordinate (fixed codebook)
  dequant:   centroid lookup * norm -> inverse rotation (R is orthogonal,
             so inverse = R.T) -> fp16

Key facts from the llama.cpp implementation that this mirrors:
- one fp16 norm per 128-element block (50 bytes/128 elems for 3-bit:
  2 bytes norm + 12 bytes 3-bit codes + padding); 16->~3 bits = ~5.3x.
- the codebook is shared (Lloyd-Max on the normalized distribution), and
  the V path MUST apply the *inverse* rotation on dequant (commit 6e5a4aa
  fixed PPL 15369 -> 7.05).
- rotation is applied at quantize time only; attention dequant inlines
  centroid lookup -> inverse rotation -> scale by norm.

Pure numpy, no torch. Drop-in:

    kv = QuantizedKVCache(max_len, kv_heads, head_dim, bits=3,
                          rotation="iso")
    kv.stage_write(pos, k, v)   # prefill fast path (fp16 staging)
    kv.finalize(n)              # after prefill -> convert to quantized
    kv.write(pos, k, v)         # decode tokens: quantized on insert
    k, v = kv.frames(n)         # fp16 (n, kv_heads, dim)
    kv.reset()

Enable with ANDROIDLLM_KV_BITS=3|4 (default 0 = classic fp16 cache).
"""
import numpy as np


def givens_rotation(dim, seed=7):
    """Block-diagonal 2D Givens rotations over (0,1),(2,3),... pairs.

    Orthogonal -> inner products preserved. Angles are seeded but fixed at
    import: the transform is a deterministic decorrelator, exactly like the
    planar3 codebook rotation."""
    rng = np.random.default_rng(seed)
    R = np.eye(dim, dtype=np.float32)
    for j in range(0, dim - 1, 2):
        ang = rng.uniform(0.4, 1.3)
        c, s = np.cos(ang), np.sin(ang)
        R[j, j] = c
        R[j, j + 1] = -s
        R[j + 1, j] = s
        R[j + 1, j + 1] = c
    return R


def quaternion_rotation(dim, seed=11):
    """Block-diagonal 4D rotations (product of 3 Givens per 4-block)."""
    rng = np.random.default_rng(seed)
    R = np.eye(dim, dtype=np.float32)
    for j in range(0, dim - 1, 4):
        n = min(4, dim - j)
        a = rng.uniform(0.3, 1.4, size=3)
        M = np.eye(n, dtype=np.float32)
        for i, ang in enumerate(a):
            c, s = np.cos(ang), np.sin(ang)
            G = np.eye(n, dtype=np.float32)
            p, q = i % n, (i + 1) % n
            G[p, p] = c
            G[p, q] = -s
            G[q, p] = s
            G[q, q] = c
            M = M @ G
        R[j:j + n, j:j + n] = M
    return R


def lloyd_max_centroids(bits, seed=42, n_samples=65536, iters=40):
    """Lloyd-Max centroids for the max-abs-normalized normal distribution.

    u = x / max|x| for x ~ N(0, 1)^d concentrates the mass of each
    coordinate; the centroid set below is what the max-abs normalized
    rotated components see, so the codebook is near-optimal for every
    block and only the per-vector norm rescales it."""
    rng = np.random.default_rng(seed)
    x = rng.standard_normal((n_samples // 64, 64)).astype(np.float32)
    u = x / np.max(np.abs(x), axis=-1, keepdims=True)
    u = u.reshape(-1)
    k = 1 << bits
    lo, hi = np.quantile(u, [0.5 / k, 1 - 0.5 / k])
    c = np.linspace(lo, hi, k, dtype=np.float32)
    idx = np.arange(u.size)
    rng.shuffle(idx)
    for _ in range(iters):
        d = np.abs(u[:, None] - c[None, :])
        a = d.argmin(axis=1)
        for i in range(k):
            m = a == i
            if m.any():
                c[i] = u[m].mean()
    return c


def build_rotation(kind, head_dim):
    if kind in ("iso", "quat", "iso3", "iso4", "quaternion"):
        return quaternion_rotation(head_dim)
    return givens_rotation(head_dim)


def _pack_bits(codes, bits):
    """(..., dim) int codes in [0, 2^bits) -> (..., nbytes) uint8."""
    codes = codes.astype(np.uint16, copy=False)
    dim = codes.shape[-1]
    total = dim * bits
    nbytes = (total + 7) // 8
    out = np.zeros(codes.shape[:-1] + (nbytes,), dtype=np.uint8)
    for i in range(dim):
        byte_idx = (i * bits) // 8
        bit_off = (i * bits) % 8
        codepart = codes[..., i]
        for b in range(bits):
            bit = (codepart >> b) & 1
            bit_abs = bit_off + b
            if bit_abs < 8:
                out[..., byte_idx] |= np.uint8(bit << bit_abs)
            else:
                out[..., byte_idx + 1] |= np.uint8(bit << (bit_abs - 8))
    return out


def _unpack_bits(packed, bits, dim):
    """Inverse of _pack_bits: (..., nbytes) -> (..., dim) int arrays."""
    codes = np.zeros(packed.shape[:-1] + (dim,), dtype=np.uint16)
    for i in range(dim):
        byte_idx = (i * bits) // 8
        bit_off = (i * bits) % 8
        if bit_off + bits <= 8:
            codes[..., i] = (packed[..., byte_idx] >> bit_off) & ((1 << bits) - 1)
        else:
            lo = packed[..., byte_idx] >> bit_off
            hi = packed[..., byte_idx + 1] & ((1 << (bits + bit_off - 8)) - 1)
            codes[..., i] = np.uint16((hi.astype(np.uint16) << (8 - bit_off)) | lo)
    return codes


class QuantizedKVCache:
    """One layer's compressed (K, V) store.

    Prefill: stage_write() keeps fp16 (deferred quantization — zero error
    compounding during prompt processing, and no rotation cost at prefill).
    finalize() converts the staged window into the quantized store.
    Decode: write() quantizes directly on insert.
    Read: frames() -> centroid lookup * norm @ R (inverse rotation).
    """

    def __init__(self, max_len, kv_heads, head_dim, bits=3, rotation="planar"):
        self.max_len = int(max_len)
        self.kv_heads = int(kv_heads)
        self.head_dim = int(head_dim)
        self.bits = int(bits)
        assert 2 <= self.bits <= 5, "bits 2..5 (0 = fp16 via caller)"
        self.rotation = rotation
        self.R = build_rotation(rotation, self.head_dim)
        self.cent = lloyd_max_centroids(self.bits).astype(np.float32)
        self.codes_per_vec = (self.head_dim * self.bits + 7) // 8
        self._stage_k = None
        self._stage_v = None
        self._qk = np.zeros((max_len, kv_heads, self.codes_per_vec), dtype=np.uint8)
        self._qv = np.zeros((max_len, kv_heads, self.codes_per_vec), dtype=np.uint8)
        self._sk = np.zeros((max_len, kv_heads), dtype=np.float16)
        self._sv = np.zeros((max_len, kv_heads), dtype=np.float16)
        self._stage_upto = 0
        self._n = 0
        self._finalized = False

    # quantize (kv_heads, dim) fp16 -> codes + per-(token, head) norm
    def _quant(self, x):
        x32 = x.astype(np.float32)
        r = x32 @ self.R.T
        scale = np.max(np.abs(r), axis=-1, keepdims=True)
        scale = np.where(scale < 1e-9, 1.0, scale)
        u = r / scale
        codes = np.abs(u[..., None] - self.cent[None, :]).argmin(-1)
        return codes, scale[..., 0]

    # dequant codes + norms -> fp16 (apply norm then INVERSE rotation)
    def _dequant(self, packed, scale):
        codes = _unpack_bits(packed, self.bits, self.head_dim).astype(np.int32)
        q = self.cent[codes] * scale[..., None]
        return (q @ self.R).astype(np.float16)

    def write(self, pos, k, v):
        k = np.asarray(k, dtype=np.float16)
        v = np.asarray(v, dtype=np.float16)
        ck, sk = self._quant(k)
        cv, sv = self._quant(v)
        self._qk[pos] = _pack_bits(ck, self.bits)
        self._qv[pos] = _pack_bits(cv, self.bits)
        self._sk[pos] = sk
        self._sv[pos] = sv
        self._n = max(self._n, pos + 1)

    def stage_write(self, pos, k, v):
        if self._stage_k is None:
            self._stage_k = np.zeros((self.max_len, self.kv_heads, self.head_dim),
                                     dtype=np.float16)
            self._stage_v = np.zeros((self.max_len, self.kv_heads, self.head_dim),
                                     dtype=np.float16)
        self._stage_k[pos] = np.asarray(k, dtype=np.float16)
        self._stage_v[pos] = np.asarray(v, dtype=np.float16)
        self._stage_upto = max(self._stage_upto, pos + 1)
        self._n = max(self._n, pos + 1)

    def finalize(self, n):
        """Convert the fp16 staging window [0, n) into the quantized store.

        After this, frames() reads the compressed representation."""
        if self._stage_k is None or self._finalized:
            return
        n = int(min(n, self._stage_k.shape[0]))
        if n <= 0:
            return
        ck, sk = self._quant(self._stage_k[:n])
        cv, sv = self._quant(self._stage_v[:n])
        for p in range(n):
            self._qk[p] = _pack_bits(ck[p], self.bits)
            self._qv[p] = _pack_bits(cv[p], self.bits)
            self._sk[p] = sk[p]
            self._sv[p] = sv[p]
        self._finalized = True

    def frames(self, n=None):
        n = int(self._n) if n is None else int(n)
        if not self._finalized and self._stage_k is not None:
            return self._stage_k[:n], self._stage_v[:n]
        k = self._dequant(self._qk[:n], self._sk[:n])
        v = self._dequant(self._qv[:n], self._sv[:n])
        return k, v

    def reset(self):
        self._qk.fill(0)
        self._qv.fill(0)
        self._sk.fill(0)
        self._sv.fill(0)
        self._stage_upto = 0
        self._n = 0
        self._finalized = False

    @property
    def nbytes(self):
        return int(self._qk.nbytes + self._qv.nbytes +
                   self._sk.nbytes + self._sv.nbytes)