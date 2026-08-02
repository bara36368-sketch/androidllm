import json
import struct

import numpy as np

_DTYPE_NAMES = {
    "F64": np.dtype(np.float64), "F32": np.dtype(np.float32), "F16": np.dtype(np.float16),
    "BF16": np.dtype(np.uint16),
    "I64": np.dtype(np.int64), "I32": np.dtype(np.int32), "I16": np.dtype(np.int16),
    "I8": np.dtype(np.int8),
    "U64": np.dtype(np.uint64), "U32": np.dtype(np.uint32), "U16": np.dtype(np.uint16),
    "U8": np.dtype(np.uint8), "BOOL": np.dtype(np.bool_),
}
_DTYPE_TO_NAME = {np.dtype(d): name for name, d in _DTYPE_NAMES.items()}

_ALIGN = 8


def _pad(n):
    return (_ALIGN - n % _ALIGN) % _ALIGN


def write_safetensors(path, tensors):
    header = {}
    offset = 0
    for name, arr in tensors.items():
        arr = np.ascontiguousarray(arr)
        nbytes = arr.nbytes
        header[name] = {
            "dtype": _DTYPE_TO_NAME[arr.dtype],
            "shape": list(arr.shape),
            "data_offsets": [offset, offset + nbytes],
        }
        offset += nbytes + _pad(nbytes)
    raw = json.dumps(header, separators=(",", ":")).encode("utf-8")
    with open(path, "wb") as f:
        f.write(struct.pack("<Q", len(raw)))
        f.write(raw)
        for _name, arr in tensors.items():
            arr = np.ascontiguousarray(arr)
            f.write(arr.tobytes())
            f.write(b"\x00" * _pad(arr.nbytes))


def read_header(path):
    with open(path, "rb") as f:
        raw = f.read(8)
        if len(raw) < 8:
            raise ValueError(f"not a safetensors file: {path}")
        header_len = struct.unpack("<Q", raw)[0]
        header = json.loads(f.read(header_len))
        data_start = 8 + header_len
        f.seek(0, 2)
        file_size = f.tell()
    return header, data_start, file_size


def _to_numpy(dtype_name):
    return _DTYPE_NAMES[dtype_name]


def read_tensor(path, name, header=None, memmap=True):
    h, data_start, _ = read_header(path)
    if header is None:
        header = h
    meta = header.get(name)
    if meta is None:
        raise KeyError(f"{name} not in {path}")
    dtype = _to_numpy(meta["dtype"])
    shape = tuple(meta["shape"])
    a, b = meta["data_offsets"]
    if memmap:
        arr = np.memmap(path, dtype=dtype, mode="r", offset=data_start + a, shape=shape)
    else:
        with open(path, "rb") as f:
            f.seek(data_start + a)
            nbytes = b - a
            arr = np.frombuffer(f.read(nbytes), dtype=dtype).reshape(shape).copy()
    return arr


def tensor_names(path):
    header, _, _ = read_header(path)
    return list(header.keys())
