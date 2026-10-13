#!/usr/bin/env python3
"""Which IQ4_NL mapping produces the peer's row 0?

`ocl_peer_parity` dumps the layer-0 expert-0 blob, the q8_1 of the SwiGLU output, and three rows of
the down projection: the peer's, `iq_mmvq`'s and `native_expert_grouped`'s (the last two agree, and are
the correct answer).  This recomputes row 0 under every plausible reading of `Split<20>::load` /
`apply` and reports which one lands on the peer's number, so the difference can be named rather than
guessed.  It also has to reproduce the correct answer first - otherwise the model of the kernel is
wrong and the search means nothing.
"""

import itertools
import struct
import sys

H, FF = 2560, 640
KV = [-127, -104, -83, -65, -49, -35, -22, -10, 1, 13, 25, 38, 53, 69, 89, 113]

def f32(x):
    return struct.unpack('<f', struct.pack('<f', float(x)))[0]


def fma32(a, b, c):
    """fma.rn.f32 for float32 inputs, via an exact (48-bit product) double step."""
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
    scale = 2.0 ** (e - 15)
    v = (1.0 + m / 1024.0) * scale
    return f32(-v if s else v)


def dp4a(g, w, acc):
    r = acc
    for i in range(4):
        r += s8((g >> (8 * i)) & 0xFF) * s8((w >> (8 * i)) & 0xFF)
    return r


def group_word(qs, dword, nibble):
    """The four table entries of one (dword, nibble half) group, packed little-endian."""
    sh = 0 if nibble == 0 else 4
    g = 0
    for b in range(4):
        g |= (KV[(qs[dword * 4 + b] >> sh) & 0xF] & 0xFF) << (8 * b)
    return g


def one_call(blob, hq, down_off, k, act_block, groups):
    """`Split<20>::apply`: dw*dx rounds on its own, then one fma into the accumulator."""
    kbx, m = k // 2, k % 2
    wb = down_off + kbx * 18
    qs = blob[wb + 2: wb + 18]
    dw = f16(blob[wb] | (blob[wb + 1] << 8))
    ab = act_block(kbx, m)
    dx = f16(hq[ab * 36] | (hq[ab * 36 + 1] << 8))
    words = [u32(hq, ab * 36 + 4 + 4 * j) for j in range(8)]
    si = 0
    for dword, nibble, widx in groups(m):
        si = dp4a(group_word(qs, dword, nibble), words[widx], si)
    return f32(dw * dx), f32(si)


def row0(blob, hq, down_off, d_row, act_block, groups):
    """One row of the down projection: the 40 calls, the 8-lane tree, then the xor butterfly.

    `row_dot_40_sub8` gives lane t the calls k = t and k = t + 32 accumulated into one register, and
    k = t + 16, k = t + 8, k = t + 24 in three more, combined as ((a + b) + c) + (d + e) with a plain
    rounded add each time.  The final reduction is __shfl_xor over 4, 2, 1.
    """
    scale, si = {}, {}
    for k in range(40):
        scale[k], si[k] = one_call(blob, hq, down_off, k, act_block, groups)
    lane = []
    for t in range(8):
        v = fma32(scale[t], si[t], f32(0))
        v = fma32(scale[t + 32], si[t + 32], v)
        lane.append(v)
    out = []
    for t in range(8):
        v = f32(lane[t] + fma32(scale[t + 16], si[t + 16], f32(0)))
        w = f32(fma32(scale[t + 8], si[t + 8], f32(0)) + fma32(scale[t + 24], si[t + 24], f32(0)))
        out.append(f32(v + w))
    for o in (4, 2, 1):
        out = [f32(out[l] + out[l ^ o]) for l in range(8)]
    return float(out[0])


CORRECT_GROUPS = lambda m: [(2 * m + 0, 0, 2 * m + 0), (2 * m + 0, 1, 2 * m + 4),
                            (2 * m + 1, 0, 2 * m + 1), (2 * m + 1, 1, 2 * m + 5)]


def main():
    raw = open("iq4nl_dump.bin", "rb").read()
    h, ff, down_off, d_row = struct.unpack("<4I", raw[:16])
    assert (h, ff) == (H, FF), (h, ff)
    off = 16
    blob = raw[off: off + 1971200]
    off += 1971200
    hq = raw[off: off + 720]
    off += 720
    got = struct.unpack('<%df' % H, raw[off: off + H * 4]); off += H * 4
    y = struct.unpack('<%df' % H, raw[off: off + H * 4]); off += H * 4
    ref = struct.unpack('<%df' % H, raw[off: off + H * 4])
    print(f"h={h} ff={ff} down_off={down_off} d_row={d_row}")
    print(f"got[0] = {got[0]:.6g}   mmvq[0] = {y[0]:.6g}   ref[0] = {ref[0]:.6g}\n")

    # the model has to be able to produce the correct answer first
    good = row0(blob, hq, down_off, d_row, lambda k, m: k, CORRECT_GROUPS)
    print(f"correct mapping  -> {good:.6g}   (mmvq {y[0]:.6g})   "
          f"{'MODEL OK' if abs(good - y[0]) < 1e-3 else 'MODEL WRONG'}")

    # then: which act-block / word pairing gives the peer's number?
    hits = []
    for name, ab in {"kbx": lambda k, m: k, "2kbx+m": lambda k, m: 2 * k + m,
                     "2kbx": lambda k, m: 2 * k, "kbx+m": lambda k, m: k + m}.items():
        for perm in itertools.permutations((0, 1, 4, 5)):
            g = lambda m, p=perm: [(2 * m + 0, 0, 2 * m + p[0]), (2 * m + 0, 1, 2 * m + p[1]),
                                   (2 * m + 1, 0, 2 * m + p[2]), (2 * m + 1, 1, 2 * m + p[3])]
            try:
                v = row0(blob, hq, down_off, d_row, ab, g)
            except IndexError:
                continue
            if abs(v - got[0]) < 1e-3:
                hits.append((name, perm, v))
            if name == "kbx" and perm == (0, 4, 1, 5):
                print(f"  peer's own rules -> {v:.6g}   (got {got[0]:.6g})")
    print()
    if hits:
        for name, perm, v in hits:
            print(f"MATCH: act block = {name}, words {perm} -> {v:.6g}")
    else:
        print("no (act block, word pairing) combination reproduces the peer's row 0")
    return 0


if __name__ == "__main__":
    sys.exit(main())
