"""Reconstruct one real EXL3 weight with the numpy specification and print the same checksum the C++
kernel does, so the two can be compared on the actual model.

    python3 tools/exl3/check_real.py <shard.safetensors> <base_name>
"""
from __future__ import annotations

import pathlib
import sys

import numpy as np

HERE = pathlib.Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(HERE.parent))
import exl3                                                             # noqa: E402
from exl3.codebook import CB_3INST, CB_MCG, CB_MUL1                    # noqa: E402


def fnv1a(b: bytes) -> int:
    h = 1469598103934665603
    for x in b:
        h ^= x
        h = (h * 1099511628211) & 0xFFFFFFFFFFFFFFFF
    return h


def main() -> int:
    from safetensors import safe_open
    shard, base = sys.argv[1], sys.argv[2]
    with safe_open(shard, framework="numpy") as f:
        keys = set(f.keys())
        tr = f.get_tensor(base + ".trellis").astype(np.int16)
        suh = f.get_tensor(base + ".suh").astype(np.float16)
        svh = f.get_tensor(base + ".svh").astype(np.float16)
    cb = CB_MUL1 if (base + ".mul1") in keys else (CB_MCG if (base + ".mcg") in keys else CB_3INST)
    ki, nj, words = tr.shape
    bits = words * 16 // 256
    print("trellis I16 [%d,%d,%d] K=%d cb=%d" % (ki, nj, words, bits, cb))
    w = exl3.reconstruct_weight(tr, suh, svh, cb)
    h = fnv1a(np.ascontiguousarray(w, dtype="<f2").tobytes())
    print("reconstructed %dx%d fp16 | fnv1a=%016x" % (w.shape[0], w.shape[1], h))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
