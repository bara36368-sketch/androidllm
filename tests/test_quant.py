import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from androidllm.quant import pack_int4, unpack_int4, quantize_matrix, dequantize_packed, dequantize_matrix


def test_quant_roundtrip():
    rng = np.random.default_rng(42)
    for bits in (4, 8):
        for block in (32, 64):
            for shape in ((64, 128), (13, 200)):
                w = rng.standard_normal(shape).astype(np.float32)
                q, scale = quantize_matrix(w, bits=bits, block=block)
                if bits == 4:
                    packed = pack_int4(q)
                    assert packed.shape == (shape[0], q.shape[1] // 2)
                    deq = dequantize_packed(packed, scale, {
                        "bits": bits, "block": block, "out": shape[0], "in": q.shape[1]})
                else:
                    deq = dequantize_matrix(q.astype(np.int8), scale, bits=bits)
                pad_amt = block - shape[1] % block
                padded = w if pad_amt == block else np.pad(w, ((0, 0), (0, pad_amt)))
                amax = np.max(np.abs(padded))
                err = np.max(np.abs(padded.astype(np.float16) - deq))
                assert err < 0.15 * amax + 1e-6, f"bits={bits} block={block} err={err} amax={amax}"
    print("quant roundtrip OK")


def test_pack_int4_roundtrip():
    q = np.array([[0, 1, 2, 7, -8, -1, -7, -8, 3, 4],
                  [5, 6, -2, 0, -8, -5, 3, 1, 2, 7]], dtype=np.int32)
    packed = pack_int4(q)
    assert packed.shape == (2, 5)
    out = unpack_int4(packed, 2, 10)
    assert np.array_equal(q, out), (q, out)
    print("pack roundtrip OK")


if __name__ == "__main__":
    test_quant_roundtrip()
    test_pack_int4_roundtrip()
