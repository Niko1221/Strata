#!/usr/bin/env python3
"""The reference tree for IQ4_NL, with the accumulation spelled the way nvcc compiles it.

`row_dot_40_sub8` accumulates `s[c] += S::apply(...)` inside its loop, and with `--use_fast_math` nvcc
contracts that into one `fma` per call - so lane t's two calls (k = t, k = t + 32) chain into a single
register, each with one rounding, instead of two independently rounded products added together.  The
`__fadd_rn` calls around the s16 and s8+s24 steps are intrinsics and stay plain adds.  Comparing the two
spellings against `iq_mmvq` says which one the engine ran.
"""

import struct
import sys

KV = [-127, -104, -83, -65, -49, -35, -22, -10, 1, 13, 25, 38, 53, 69, 89, 113]


def f32(x):
    return struct.unpack('<f', struct.pack('<f', float(x)))[0]


def fma32(a, b, c):
    """fma.rn.f32: the double product is exact, so a single round is the whole story."""
    return f32(float(a) * float(b) + float(c))


def u32(b, o):
    return b[o] | (b[o + 1] << 8) | (b[o + 2] << 16) | (b[o + 3] << 24)


def s8(v):
    return v - 256 if v >= 128 else v


def f16(u):
    s, e, m = (u >> 15) & 1, (u >> 10) & 0x1F, u & 0x3FF
    if e == 31:
        return float("nan")
    if e == 0:
        v = m * 5.9604644775390625e-08
        return -v if s else v
    return (-1.0 if s else 1.0) * (1.0 + m / 1024.0) * (2.0 ** (e - 15))


def dp4a(g, w):
    return sum(s8((g >> (8 * i)) & 0xFF) * s8((w >> (8 * i)) & 0xFF) for i in range(4))


def gword(qs, dword, hi):
    sh = 4 if hi else 0
    return sum((KV[(qs[dword * 4 + b] >> sh) & 0xF] & 0xFF) << (8 * b) for b in range(4))


def main():
    raw = open("iq4nl_dump.bin", "rb").read()
    _, _, DO, DR = struct.unpack('<4I', raw[:16])
    o = 16
    blob = raw[o:o + 1971200]; o += 1971200
    hq = raw[o:o + 720]; o += 720
    o += 10240                                   # `got`, the peer engine's own rows
    y = struct.unpack('<2560f', raw[o:o + 10240]); o += 10240
    ref = struct.unpack('<2560f', raw[o:o + 10240])
    rep = struct.unpack('<2560f', open("repro_row.bin", "rb").read())

    def call(row, k):
        """`S::apply`, split: the float scale the engine multiplies in, and the integer dot it scales."""
        kbx, m = k // 2, k % 2
        base = DO + row * DR + kbx * 18
        dw = f16(blob[base] | (blob[base + 1] << 8))
        qs = blob[base + 2: base + 18]
        ab = kbx * 36
        dx = f16(hq[ab] | (hq[ab + 1] << 8))
        words = [u32(hq, ab + 4 + 4 * j) for j in range(8)]
        si = 0
        for l in range(2):
            si += dp4a(gword(qs, 2 * m + l, False), words[2 * m + l])
            si += dp4a(gword(qs, 2 * m + l, True), words[2 * m + l + 4])
        return f32(dw * dx), float(si)

    def term(row, k, acc):
        scale, si = call(row, k)
        return fma32(scale, si, acc)

    def tree(row, chained):
        out = []
        for t in range(8):
            if chained:
                s = term(row, t, f32(0.0))
                s = term(row, t + 32, s)
            else:
                s = f32(term(row, t, f32(0.0)) + term(row, t + 32, f32(0.0)))
            s = f32(s + term(row, t + 16, f32(0.0)))               # __fadd_rn, no contraction
            s8 = term(row, t + 8, f32(0.0))
            s24 = term(row, t + 24, f32(0.0))
            out.append(f32(s + f32(s8 + s24)))
        for o_ in (4, 2, 1):
            out = [f32(out[l] + out[l ^ o_]) for l in range(8)]
        return out[0]

    n = 200
    for chained in (True, False):
        mm = sum(1 for r in range(n) if tree(r, chained) != y[r])
        rr = sum(1 for r in range(n) if tree(r, chained) != ref[r])
        lbl = "fma-chained (s += apply)" if chained else "plain adds of rounded products"
        print(f"{lbl:34s}  vs mmvq {mm:3d}/{n}   vs ref {rr:3d}/{n}")
    print(f"{'peer kernel (repro_row.bin)':34s}  vs mmvq "
          f"{sum(1 for r in range(n) if rep[r] != y[r]):3d}/{n}")
    print()
    for r in [0, 4, 9, 15, 18, 33]:
        print(f"row {r:4d}  chained {tree(r, True):14.9g}  plain {tree(r, False):14.9g}  "
              f"mmvq {y[r]:14.9g}  peer {rep[r]:14.9g}")
    print(f"\nmmvq == ref on all {n} rows: {all(y[r] == ref[r] for r in range(n))}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
