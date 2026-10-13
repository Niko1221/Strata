#!/usr/bin/env python3
"""Compare the kernel's own per-lane numbers against the reference decode, call by call.

The debug build reports, for every lane, the exact `si`, `dw`, `dx`, `qa`, `qb` it used and the value it
accumulated.  Putting the reference's own `si` next to them says whether the two sides differ in which
weights they read, which activation words they pair them with, or the tree above them.
"""

import struct
import sys

FF = 640
KV = [-127, -104, -83, -65, -49, -35, -22, -10, 1, 13, 25, 38, 53, 69, 89, 113]


def f32(x):
    return struct.unpack('<f', struct.pack('<f', float(x)))[0]


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


def gword(qs, dword, high):
    sh = 4 if high else 0
    g = 0
    for b in range(4):
        g |= (KV[(qs[dword * 4 + b] >> sh) & 0xF] & 0xFF) << (8 * b)
    return g


def main():
    blobfile = open("peer_blob_l0.bin", "rb").read()
    hdr = struct.unpack('<8Q', blobfile[:64])
    down_off, d_row = hdr[1], hdr[2]
    blob = blobfile[64:]
    hq = open("peer_hq_l0.bin", "rb").read()

    # the reference's own numbers for every call of row 0
    ref_si, ref_dw, ref_dx = {}, {}, {}
    for k in range(40):
        kbx, m = k // 2, k % 2
        base = down_off + kbx * 18
        ref_dw[k] = f16(blob[base] | (blob[base + 1] << 8))
        qs = blob[base + 2: base + 18]
        ab = kbx * 36
        ref_dx[k] = f16(hq[ab] | (hq[ab + 1] << 8))
        w = [u32(hq, ab + 4 + 4 * j) for j in range(8)]
        si = 0
        for l in range(2):
            si += dp4a(gword(qs, 2 * m + l, False), w[2 * m + l])
            si += dp4a(gword(qs, 2 * m + l, True), w[2 * m + l + 4])
        ref_si[k] = si

    # what the kernel reported
    lines = [l.split() for l in sys.stdin.read().splitlines() if l.strip()]
    rows = []
    for t in lines:
        if len(t) == 10 and t[0].isdigit() and t[1].isdigit():
            rows.append(t)
    print(f"kernel lanes parsed: {len(rows)}")
    print("\n lane  k  si_kernel   si_ref     dw ok  dx ok   qa          qb")
    bad_si = bad_dw = bad_dx = 0
    for t in rows:
        lane, k = int(t[0]), int(t[1])
        si_k, dw_k, dx_k = int(float(t[4])), float(t[5]), float(t[6])
        qa, qb = t[7], t[8]
        okd = "   " if dw_k == ref_dw[k] else " DW"
        okx = "   " if dx_k == ref_dx[k] else " DX"
        oks = "" if si_k == ref_si[k] else "SI"
        if si_k != ref_si[k]:
            bad_si += 1
        if dw_k != ref_dw[k]:
            bad_dw += 1
        if dx_k != ref_dx[k]:
            bad_dx += 1
        if oks or okd.strip() or okx.strip() or k < 4:
            print(f"{lane:5d} {k:3d}  {si_k:9d}  {ref_si[k]:9d}  {okd} {okx}  {qa}  {qb}  {oks}")
    print(f"\nsi mismatches {bad_si}/40   dw {bad_dw}/40   dx {bad_dx}/40")

    # is the kernel's si a permutation of the reference's?  (same multiset => wrong activation pairing)
    if bad_si:
        a = sorted(int(float(t[4])) for t in rows)
        b = sorted(ref_si.values())
        print(f"\nkernel si multiset == reference si multiset: {a == b}")
        if a != b:
            sa, sb = set(a), set(b)
            print(f"  in kernel only: {sorted(sa - sb)[:10]}")
            print(f"  in ref only:    {sorted(sb - sa)[:10]}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
