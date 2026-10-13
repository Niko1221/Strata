#!/usr/bin/env python3
"""Find which row of the blob (if any) reproduces the peer's row 0.

The peer's `hq` is byte-identical to the reference and its IQ4_NL decode text is right, yet its row 0
is 447.458 where the reference says 322.  So it is reading *different weights*.  Scanning every row of
the layer blob with the verified model turns "it is wrong" into "it is reading row N instead of row 0",
which names the offset bug directly.
"""

import struct
import sys

H, FF = 2560, 640
KV = [-127, -104, -83, -65, -49, -35, -22, -10, 1, 13, 25, 38, 53, 69, 89, 113]


def f32(x):
    return struct.unpack('<f', struct.pack('<f', float(x)))[0]


def fma32(a, b, c):
    return f32(float(a) * float(b) + float(c))


def u32(b, o):
    return b[o] | (b[o + 1] << 8) | (b[o + 2] << 16) | (b[o + 3] << 24)


def s8(v):
    return v - 256 if v >= 128 else v


def f16(u):
    s = (u >> 15) & 1
    e = (u >> 10) & 0x1F
    m = u & 0x3FF
    if e == 31:
        return float("nan")
    if e == 0:
        v = m * 5.9604644775390625e-08
        return -v if s else v
    return (-1.0 if s else 1.0) * (1.0 + m / 1024.0) * (2.0 ** (e - 15))


def dp4a(g, w):
    r = 0
    for i in range(4):
        r += s8((g >> (8 * i)) & 0xFF) * s8((w >> (8 * i)) & 0xFF)
    return r


def group_word(qs, dword, high):
    sh = 4 if high else 0
    g = 0
    for b in range(4):
        g |= (KV[(qs[dword * 4 + b] >> sh) & 0xF] & 0xFF) << (8 * b)
    return g


def build_hq(hqbuf):
    """word 0 = {fp16 d, fp16 sum}, words 1..8 = 32 int8 -> (dx, 8 q8 words)."""
    out = []
    for i in range(0, len(hqbuf), 36):
        b = hqbuf[i:i + 36]
        out.append((f16(b[0] | (b[1] << 8)), [u32(b, 4 + 4 * j) for j in range(8)]))
    return out


def row0(blob, hq, row_off, act):
    """One row: the 40 calls at Fmt<20> ipb=2 step=2, then the 8-lane tree + xor butterfly."""
    xs = {}
    for k in range(40):
        kbx, m = k // 2, k % 2
        base = row_off + kbx * 18
        dw = f16(blob[base] | (blob[base + 1] << 8))
        qs = blob[base + 2: base + 18]
        dx, words = act[kbx]
        si = 0
        # l = 0 -> q4 dword 0 of this pair, l = 1 -> dword 1; x -> q8 word l, y -> q8 word l+4
        for l in range(2):
            si += dp4a(group_word(qs, 2 * m + l, False), words[2 * m + l])
            si += dp4a(group_word(qs, 2 * m + l, True), words[2 * m + l + 4])
        xs[k] = fma32(dw * dx, float(si), f32(0))
    lane = [f32(xs[t] + xs[t + 32]) for t in range(8)]
    out = [f32(f32(lane[t] + xs[t + 16]) + f32(xs[t + 8] + xs[t + 24])) for t in range(8)]
    for o in (4, 2, 1):
        out = [f32(out[l] + out[l ^ o]) for l in range(8)]
    return out[0]


def main():
    raw = open("iq4nl_dump.bin", "rb").read()
    h, ff, down_off, d_row = struct.unpack("<4I", raw[:16])
    print(f"H={h} FF={ff} down_off={down_off} d_row={d_row}")
    off = 16
    blob = raw[off: off + 1971200]; off += 1971200
    hqbuf = raw[off: off + 720]; off += 720
    got = struct.unpack('<%df' % H, raw[off: off + H * 4]); off += H * 4
    y = struct.unpack('<%df' % H, raw[off: off + H * 4]); off += H * 4

    act = build_hq(hqbuf)
    v0 = row0(blob, act, down_off, act)
    print(f"model row 0 at down_off      -> {v0:.6g}   mmvq {y[0]:.6g}   got {got[0]:.6g}")
    print(f"   ({'model reproduces the reference' if abs(v0 - y[0]) < 1e-3 else 'MODEL WRONG'})\n")

    target = got[0]
    hits = []
    for r in range(H):
        v = row0(blob, act, down_off + r * d_row, act)
        if abs(v - target) < 1e-3:
            hits.append(r)
        if r < 4:
            print(f"  row {r}: {v:.6g}")
    print()
    if hits:
        print(f"peer row 0 == model row(s) {hits[:8]} of the blob (down_off + r*d_row)")
    else:
        print("no row of the blob reproduces the peer's row 0 -> the weights are not the problem; "
              "the peer's own arithmetic differs from the reference tree")

    # and how the peer's whole vector compares: is it a shifted copy of mmvq?
    best = None
    for sh in range(-8, 9):
        n = 0
        for r in range(200, H - 200):
            if abs(got[r] - y[r + sh]) < 1e-3:
                n += 1
        if best is None or n > best[1]:
            best = (sh, n)
    print(f"\nshift scan: got[r] vs mmvq[r+sh], best sh={best[0]} with {best[1]} of {H-400} rows matching")
    return 0


if __name__ == "__main__":
    sys.exit(main())
