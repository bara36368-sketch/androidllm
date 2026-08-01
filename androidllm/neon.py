import ctypes
import os

import numpy as np

_FALLBACK = True
_lib = None

_SO_NAME = "libandroidllm_neon.so"


def _load():
    global _lib, _FALLBACK
    for base in (os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                 os.path.join(os.path.expanduser("~"), ".androidllm")):
        so = os.path.join(base, _SO_NAME)
        if os.path.exists(so):
            try:
                lib = ctypes.CDLL(so)
                lib.matmul_f16_f16.argtypes = [
                    ctypes.POINTER(ctypes.c_uint16), ctypes.POINTER(ctypes.c_uint16),
                    ctypes.POINTER(ctypes.c_uint16), ctypes.c_int, ctypes.c_int, ctypes.c_int,
                ]
                _lib = lib
                _FALLBACK = False
                return
            except OSError:
                pass
    _FALLBACK = True


def matmul_f16(a, b, _=None):
    a = np.ascontiguousarray(a, dtype=np.float16)
    b = np.ascontiguousarray(b, dtype=np.float16)
    if _lib is not None and a.shape[1] >= 64:
        m, k = a.shape
        n = b.shape[0]
        out = np.zeros((m, n), dtype=np.float16)
        _lib.matmul_f16_f16(a.ctypes.data_as(ctypes.POINTER(ctypes.c_uint16)),
                            b.ctypes.data_as(ctypes.POINTER(ctypes.c_uint16)),
                            out.ctypes.data_as(ctypes.POINTER(ctypes.c_uint16)),
                            m, k, n)
        return out
    return a @ b.T


_load()
