"""Emit the EXL3 CPU-kernel parity fixture from the verified numpy reference.

    python3 tools/exl3/emit_fixture.py --out <dir>

Writes <dir>/exl3_fixture.bin, read by src/kernels/exl3_parity.cpp.  The oracle is
tools/exl3 (the format spec); a C++ mismatch means the port is wrong, never the fixture.
"""
from __future__ import annotations

import argparse
import pathlib
import struct
import sys

import numpy as np

HERE = pathlib.Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(HERE.parent))
import exl3                                                        # noqa: E402
from exl3.codebook import CB_MUL1, CB_3INST, CB_MCG               # noqa: E402
from exl3 import ngram as ng                                       # noqa: E402

MAGIC = b"EXL3"


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("--cb", type=int, default=CB_MUL1)
    ap.add_argument("--ki", type=int, default=8)
    ap.add_argument("--nj", type=int, default=8)
    ap.add_argument("--bits", type=int, default=3)
    ap.add_argument("--tokens", type=int, default=2)
    ap.add_argument("--seed", type=int, default=1234)
    args = ap.parse_args()

    rng = np.random.default_rng(args.seed)
    ki, nj, bits, tokens = args.ki, args.nj, args.bits, args.tokens
    words = 256 * bits // 16
    k, n = ki * 16, nj * 16

    trellis = rng.integers(-(1 << 15), 1 << 15, size=(ki, nj, words), dtype=np.int16)
    suh = (rng.standard_normal(k) * 0.5).astype(np.float16)
    svh = (rng.standard_normal(n) * 0.5).astype(np.float16)
    x = (rng.standard_normal((tokens, k)) * 0.3).astype(np.float16)

    lut = exl3.codebook_lut(args.cb)                              # (65536,) fp16
    w_hat = exl3.decode_weight_hat(trellis, args.cb)              # (k,n) fp16
    w = exl3.reconstruct_weight(trellis, suh, svh, args.cb)       # (k,n) fp16
    y = exl3.folded_forward(x, trellis, suh, svh, args.cb)        # (tokens,n) fp32

    out = pathlib.Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    path = out / "exl3_fixture.bin"
    with open(path, "wb") as f:
        f.write(MAGIC)
        f.write(struct.pack("<iiiiii", args.cb, ki, nj, bits, tokens, words))
        f.write(trellis.astype("<i2").tobytes())
        f.write(suh.astype("<f2").tobytes())                       # raw half bit patterns
        f.write(svh.astype("<f2").tobytes())
        f.write(x.astype("<f2").tobytes())
        f.write(lut.astype("<f2").tobytes())
        f.write(w_hat.astype("<f2").tobytes())
        f.write(w.astype("<f2").tobytes())
        f.write(y.astype("<f4").tobytes())
        f.write(w.astype(np.float32).tobytes())                   # reference W in fp32, for the tol check
        # ---- ngram ring block: K, dim, has_bias, ring words, bias, expected ----
        K, dim = 5, 160
        sym = rng.integers(0, 1 << K, size=dim, dtype=np.int64)
        st = np.zeros(dim, dtype=np.int64)
        for j in range((15 + K) // K):
            st |= np.roll(sym, j) << (j * K)
        ring = ng.pack_row((st & 0xFFFF).astype(np.uint16), np.float16(rng.uniform(0.01, 0.1)), K)
        bias = (rng.standard_normal(dim) * 0.01).astype(np.float32)
        exp = ng.dequant_row(ring, K, ng.mul1_codebook(), bias)
        f.write(struct.pack("<iii", K, dim, 1))
        f.write(ring.astype("<i2").tobytes())
        f.write(bias.astype("<f4").tobytes())
        f.write(exp.astype("<f4").tobytes())
    print("wrote %s (%d bytes): cb=%d ki=%d nj=%d bits=%d tokens=%d" %
          (path, path.stat().st_size, args.cb, ki, nj, bits, tokens))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
