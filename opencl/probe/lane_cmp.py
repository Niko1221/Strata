#!/usr/bin/env python3
"""Per-call comparison: the peer's IQ4_NL loop body against the reference decode.

`ocl_lane_IQ4_NL.cl` is the generated row kernel with its loop body byte-for-byte and its reduction
replaced by a dump of each lane's own number, so the two sides can be compared call by call instead of
row by row.  Group 0 only, so the dump is a single row's four rows' worth of contributions.
"""

import re
import struct
import sys

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
    s, e, m = (u >> 15) & 1, (u >> 10) & 0x1F, u & 0x3FF
    if e == 31:
        return float("nan")
    if e == 0:
        v = m * 5.9604644775390625e-08
        return -v if s else v
    return (-1.0 if s else 1.0) * (1.0 + m / 1024.0) * (2.0 ** (e - 15))


def dp4a(g, w):
    return sum(s8((g >> (8 * i)) & 0xFF) * s8((w >> (8 * i)) & 0xFF) for i in range(4))


def gword(qs, dw_, hi):
    sh = 4 if hi else 0
    return sum((KV[(qs[dw_ * 4 + b] >> sh) & 0xF] & 0xFF) << (8 * b) for b in range(4))


def main():
    bf = open("peer_blob_l0.bin", "rb").read()
    hdr = struct.unpack('<8Q', bf[:64])
    DO, DR = hdr[1], hdr[2]
    blob = bf[64:]
    hq = open("peer_hq_l0.bin", "rb").read()

    def call(k, row):
        """The reference's own value for call k of `row`, split so each part can be checked."""
        kbx, m = k // 2, k % 2
        base = DO + row * DR + kbx * 18
        dw = f16(blob[base] | (blob[base + 1] << 8))
        qs = blob[base + 2: base + 18]
        ab = kbx * 36
        dx = f16(hq[ab] | (hq[ab + 1] << 8))
        w = [u32(hq, ab + 4 + 4 * j) for j in range(8)]
        si = 0
        for l in range(2):
            si += dp4a(gword(qs, 2 * m + l, False), w[2 * m + l])
            si += dp4a(gword(qs, 2 * m + l, True), w[2 * m + l + 4])
        return fma32(dw * dx, float(si), f32(0)), dw, dx, si

    txt = open("lane_out.txt").read()
    vals = {}
    for r in range(4):
        blk = txt.split(' r%d:' % r)[1].split('\n r')[0]
        nums = re.findall(r'\d+\s+(-?[\d.eE+-]+)', blk)
        for l, v in enumerate(nums[:40]):
            vals[(r, l)] = float(v)

    print("row lane  kernel          reference       verdict")
    bad = 0
    for r in range(4):
        n = 0
        for k in range(40):
            got = vals[(r, k)]
            want = call(k, r)[0]
            if got != want:
                if n < 4:
                    print("%3d %4d  %-15.7g %-15.7g DIFF" % (r, k, got, want))
                n += 1
        bad += n
        print("  row %d: %d of 40 calls differ" % (r, n))

    if bad == 0:
        print("\nevery call agrees - the loop body is right and the reduction is what differs")
    return 0


if __name__ == "__main__":
    sys.exit(main())
