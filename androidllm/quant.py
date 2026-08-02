import numpy as np


def quantize_matrix(w, bits=4, block=64):
    if w.ndim != 2:
        raise ValueError("quantize_matrix expects a 2D matrix")
    out, in_dim = w.shape
    w = np.asarray(w, dtype=np.float32)
    if in_dim % block:
        pad = block - in_dim % block
        w = np.pad(w, ((0, 0), (0, pad)))
        in_dim += pad
    n_blocks = in_dim // block
    w2 = w.reshape(out, n_blocks, block)
    amax = np.max(np.abs(w2), axis=2, keepdims=True)
    max_val = 2 ** (bits - 1) - 1
    scale = (amax / max_val).astype(np.float16)
    scale = np.where(scale == 0, np.float16(1e-8), scale)
    q = np.round(w2 / scale.astype(np.float32)).astype(np.int32)
    q = np.clip(q, -(2 ** (bits - 1)), 2 ** (bits - 1) - 1)
    scale = scale.reshape(out, n_blocks)
    return q.reshape(out, in_dim), scale


def pack_int4(q):
    q = np.asarray(q, dtype=np.int32)
    if q.size % 2:
        raise ValueError("int4 pack needs even element count")
    out_dim = q.shape[0]
    q = q.reshape(-1, 2)
    lo = np.asarray(q[:, 0] & 0x0F, dtype=np.uint8)
    hi = np.asarray(q[:, 1] & 0x0F, dtype=np.uint8)
    return (lo | (hi << 4)).reshape(out_dim, -1)


def unpack_int4(packed, out_dim, in_dim):
    packed = np.asarray(packed, dtype=np.uint8).reshape(-1)
    lo = (packed & 0x0F).astype(np.int8)
    hi = ((packed >> 4) & 0x0F).astype(np.int8)
    vals = np.empty(packed.size * 2, dtype=np.int8)
    vals[0::2] = lo
    vals[1::2] = hi
    vals = vals.astype(np.int32)
    vals = np.where(vals >= 8, vals - 16, vals)
    return vals.reshape(out_dim, in_dim)


def dequantize_matrix(q, scale, bits=4):
    q = np.asarray(q)
    scale = np.asarray(scale, dtype=np.float32)
    if q.ndim != 2 or scale.ndim != 2:
        raise ValueError("dequantize_matrix expects 2D arrays")
    block = q.shape[1] // scale.shape[1]
    s = np.repeat(scale, block, axis=1)
    return (q.astype(np.float32) * s).astype(np.float16)


def dequantize_packed(packed, scale, meta):
    bits = meta["bits"]
    block = meta["block"]
    out = meta["out"]
    inp = meta["in"]
    q = unpack_int4(packed, out, inp) if bits == 4 else packed.astype(np.float32)
    s = np.repeat(np.asarray(scale, dtype=np.float32), block, axis=1)
    return (q * s).astype(np.float16)
