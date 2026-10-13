#!/usr/bin/env python3
"""The peer kernel's integer dot product, call by call, against the reference's.

Now that the debug table is race-free it carries the exact `si` each call computed, so this is an
integer comparison with no rounding in the way.  For every call that differs it searches which of the
q8_1 block's eight words, and which nibble half, would have produced that number.
"""

import itertools
import struct
import sys

KV = [-127, -104, -83, -65, -49, -35, -22, -10, 1, 13, 25, 38, 53, 69, 89, 113]


def f16(u):
    s, e, m = (u >> 15) & 1, (u >> 10) & 0x1F, u & 0x3FF
    if e == 31:
        return float("nan")
    if e == 0:
        v = m * 5.9604644775390625e-08
        return -v if s else v
    return (-1.0 if s else 1.0) * (1.0 + m / 1024.0) * (2.0 ** (e - 15))


def u32(b, o):
    return b[o] | (b[o + 1] << 8) | (b[o + 2] << 16) | (b[o + 3] << 24)


def s8(v):
    return v - 256 if v >= 128 else v


def dp4a(g, w):
    return sum(s8((g >> (8 * i)) & 0xFF) * s8((w >> (8 * i)) & 0xFF) for i in range(4))


def gword(qs, dword, hi):
    sh = 4 if hi else 0
    return sum((KV[(qs[dword * 4 + b] >> sh) & 0xF] & 0xFF) << (8 * b) for b in range(4))


def main():
    bf = open("peer_blob_l0.bin", "rb").read()
    hdr = struct.unpack('<8Q', bf[:64])
    DO, DR = hdr[1], hdr[2]
    blob = bf[64:]
    hq = open("peer_hq_l0.bin", "rb").read()

    k_si, k_dw, k_dx = {}, {}, {}
    for line in sys.stdin.read().splitlines():
        t = line.split()
        if len(t) == 10 and t[0].isdigit() and t[1].isdigit():
            k = int(t[1])
            k_si[k] = int(t[4])
            k_dw[k] = float(t[5])
            k_dx[k] = float(t[6])

    print(" k  si_kernel  si_reference   match   kbx  m")
    ok = bad = 0
    variants = {}
    for k in range(40):
        kbx, m = k // 2, k % 2
        base = DO + kbx * 18
        dw = f16(blob[base] | (blob[base + 1] << 8))
        qs = blob[base + 2: base + 18]
        ab = kbx * 36
        dx = f16(hq[ab] | (hq[ab + 1] << 8))
        words = [u32(hq, ab + 4 + 4 * j) for j in range(8)]
        groups = [(0, False), (0, True), (1, False), (1, True)]
        ref = sum(dp4a(gword(qs, 2 * m + g[0], g[1]), words[2 * m + (0 if g[0] == 0 else 1) + (4 if g[1] else 0)])
                  for g in groups)
        got = k_si.get(k)
        tag = ""
        if got == ref:
            ok += 1
            tag = "ok"
        else:
            bad += 1
            # which words/halves would give that number?
            found = []
            for wperm in itertools.permutations(range(8), 4):
                for halves in itertools.product([False, True], repeat=4):
                    if sum(dp4a(gword(qs, 2 * m + g[0], h), words[j])
                           for g, j, h in zip(groups, wperm, halves)) == got:
                        found.append((wperm, halves))
            tag = f"DIFF -> {found[:3]}" if found else "DIFF -> nothing in this block"
            variants.setdefault(str(found[:2]) if found else "none", []).append(k)
        dwok = "" if k_dw[k] == dw else " DWMISMATCH"
        dxok = "" if k_dx[k] == dx else " DXMISMATCH"
        print(f"{k:3d}  {got:9d}  {ref:13d}   {tag}{dwok}{dxok}")
    print(f"\nsi agrees on {ok}/40 calls, differs on {bad}/40")
    return 0


if __name__ == "__main__":
    sys.exit(main())
