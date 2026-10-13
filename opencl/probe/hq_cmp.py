#!/usr/bin/env python3
"""Is the peer's layer-0 `hq` the same buffer the reference produced?

`iq4nl_dump.bin` embeds the reference `hq`; `peer_hq_l0.bin` is what the peer's own down pass actually
read.  If these differ, the IQ4_NL decoder is innocent and the fault is upstream (gate/up proj or the
SwiGLU + q8_1 quantiser).  Comparing them settles which side of the pivot to debug.
"""

import struct
import sys

H, FF = 2560, 640


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
    return (-1.0 if s else 1.0) * (1.0 + m / 1024.0) * scale


def blocks(buf):
    """36 bytes per block: fp16 d, fp16 sum, then 32 int8."""
    out = []
    for i in range(0, len(buf), 36):
        b = buf[i:i + 36]
        d = f16(b[0] | (b[1] << 8))
        s = f16(b[2] | (b[3] << 8))
        qs = struct.unpack('<32b', b[4:36])
        out.append((d, s, list(qs)))
    return out


def main():
    raw = open("iq4nl_dump.bin", "rb").read()
    off = 16 + 1971200
    ref = raw[off: off + 720]
    peer = open("peer_hq_l0.bin", "rb").read()

    print(f"ref  {len(ref)} bytes   peer {len(peer)} bytes")
    if ref == peer:
        print("\nthe two hq buffers are IDENTICAL -> the IQ4_NL down decode is the defect")
        return 0

    diff = [i for i in range(min(len(ref), len(peer))) if ref[i] != peer[i]]
    nb = len(ref) // 36
    bad = sorted({i // 36 for i in diff})
    print(f"\nDIFFER: {len(diff)} of {len(ref)} bytes, {len(bad)} of {nb} blocks")
    print(f"blocks affected: {bad[:6]}{' ...' if len(bad) > 6 else ''}")

    rb, pb = blocks(ref), blocks(peer)
    for blk in bad[:4]:
        rd, rs, rq = rb[blk]
        pd, ps, pq = pb[blk]
        print(f"\n block {blk}  (d {rd:.6g} vs {pd:.6g}   sum {rs:.6g} vs {ps:.6g})")
        print(f"   ref  {list(rq)}")
        print(f"   peer {list(pq)}")

    # how different in magnitude, not just bytes
    def nz(q):
        return [(q[i], i) for i in range(32) if q[i]]
    if bad:
        blk = bad[0]
        print(f"\n block {bad[0]}: ref nz {nz(rb[blk][2])[:8]}")
        print(f" block {bad[0]}: peer nz {nz(pb[blk][2])[:8]}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
