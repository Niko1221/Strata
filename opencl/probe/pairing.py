#!/usr/bin/env python3
"""Which activation words does the peer's kernel really pair with which nibble groups?

The per-call values are close to the reference but not equal - the signature of the same four dp4a partial
products handed to the wrong q8_1 words, not a wrong codebook.  Recovering the integer `si` each side
computed and then searching every assignment of the block's eight words to the four groups names the
exact pairing the kernel used.
"""

import itertools
import re
import struct
import sys

KV = [-127, -104, -83, -65, -49, -35, -22, -10, 1, 13, 25, 38, 53, 69, 89, 113]


def f32(x):
    return struct.unpack('<f', struct.pack('<f', float(x)))[0]


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


def solve_si(v, dw, dx):
    """Invert the kernel's store for the integer si.

    `acc[r] = fma(dw * dx, (float) si, acc[r])` has three plausible lowerings: `f32(dw*dx)` rounds
    first; or the driver contracts despite FP_CONTRACT OFF and rounds `dw*dx*si` once.  dw and dx are
    fp16 so the exact triple product fits a double, which makes both testable exactly.
    """
    if dw == 0.0 or dx == 0.0:
        return 0 if v == 0.0 else None
    base = round(v / (dw * dx))
    for s in range(base - 3, base + 4):
        if f32(f32(dw * dx) * float(s)) == v:
            return s
        if f32(dw * dx * float(s)) == v:
            return s
    return None


def main():
    bf = open("peer_blob_l0.bin", "rb").read()
    hdr = struct.unpack('<8Q', bf[:64])
    DO, DR = hdr[1], hdr[2]
    blob = bf[64:]
    hq = open("peer_hq_l0.bin", "rb").read()

    got = {}
    txt = open("lane_out.txt").read()
    for r in range(4):
        blk = txt.split(' r%d:' % r)[1].split('\n r')[0]
        for l, v in enumerate(re.findall(r'\d+\s+(-?[\d.eE+-]+)', blk)[:40]):
            got[(r, l)] = float(v)

    tally = {}
    solved = 0
    si_ok = 0
    for r in range(4):
        for k in range(40):
            kbx, m = k // 2, k % 2
            base = DO + r * DR + kbx * 18
            dw = f16(blob[base] | (blob[base + 1] << 8))
            qs = blob[base + 2: base + 18]
            ab = kbx * 36
            dx = f16(hq[ab] | (hq[ab + 1] << 8))
            words = [u32(hq, ab + 4 + 4 * j) for j in range(8)]
            groups = {(0, False): gword(qs, 2 * m + 0, False), (0, True): gword(qs, 2 * m + 0, True),
                      (1, False): gword(qs, 2 * m + 1, False), (1, True): gword(qs, 2 * m + 1, True)}
            ref = [(0, False), (0, True), (1, False), (1, True)]
            si_ref = sum(dp4a(groups[g], words[2 * m + (0 if g[0] == 0 else 1) + (4 if g[1] else 0)])
                         for g in ref)

            si_k = solve_si(got[(r, k)], dw, dx)
            if si_k is None:
                continue
            solved += 1
            if si_k == si_ref:
                si_ok += 1
                continue
            # which assignment of the block's words to the four groups reproduces si_k?
            found = None
            for wperm in itertools.permutations(range(8), 4):
                if sum(dp4a(groups[g], words[j]) for g, j in zip(ref, wperm)) == si_k:
                    found = wperm
                    break
            cnt = tally.setdefault(found, [0, (k, si_k, si_ref, 2 * m)])
            cnt[0] += 1

    print(f"calls with a recoverable si: {solved}/160")
    print(f"of those, si == the reference's: {si_ok}")
    print("\nword assignment [g0,g1,g2,g3] -> block word index, for the calls that differ:")
    for key, (n, ex) in sorted(tally.items(), key=lambda kv: -kv[1][0]):
        k, si_k, si_ref, xm = ex
        refwords = [xm + (0 if i // 2 == 0 else 1) + (4 if i % 2 else 0) for i in range(4)]
        print(f"  {str(key):24s} {n:4d} calls   e.g. k={k} si_kernel={si_k} si_ref={si_ref} "
              f"reference words {refwords}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
