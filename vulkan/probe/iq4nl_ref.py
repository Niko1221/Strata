#!/usr/bin/env python3
"""Reference data for the Vulkan IQ4_NL down-projection kernel.

The model's down projection is [n_embd=2560 rows] x [n_ff=640 cols], which in the pack's own terms is
`d_row = iq_row_bytes(20, n_ff) = n_ff/32 * 18 = 360` bytes a row and 921600 bytes for the matrix -
native_experts.txt reports an IQ3_XXS/IQ4_NL layer's expert blob as 980*640 + 980*640 + 360*2560,
so this is that third.

block_iq4_nl is 18 bytes for 32 values: fp16 d, then 16 bytes of nibbles.  Element j is
table[qs[j] & 0xF] and element j+16 is table[qs[j] >> 4] (dequant.hpp, dequantize_row_iq4_nl).

Fmt<20> gives qk = 32, ipb = 2, step = 2, so a row is nb*ipb = 40 dot calls: call k has
kbx = k / 2 and iqs = 2 * (k % 2), and reads q8 words iqs+l and iqs+l+4 for l in {0,1}.

Writes iq4nl.bin: d[] and qs[] split (2 B and 16 B per block), x_ds/x_qs, the packed codebook, refs.
"""

import math
import struct
import sys

from iq3_ref import f32, f16_at, quantize_q8_1

N_EMBD = 2560       # rows of the down matrix (it maps n_ff back to n_embd)
N_FF = 640          # columns
NB = N_FF // 32     # 20 blocks per row
BLOCK = 18
ROW_BYTES = NB * BLOCK
REF_ROWS = 32
OUT = r"G:/01_Projects/Strata/vulkan/probe/iq4nl.bin"

T = [-127, -104, -83, -65, -49, -35, -22, -10, 1, 13, 25, 38, 53, 69, 89, 113]


def s8(v):
    return v - 256 if v >= 128 else v


def dp4a(a, b):
    return sum(s8((a >> (8 * i)) & 0xFF) * s8((b >> (8 * i)) & 0xFF) for i in range(4))


def call_ref(row, xq_words, xq_d, kbx, iqs):
    """One of the row's 40 dot calls -> the integer sum, exactly as the kernel chains its dp4a."""
    base = kbx * BLOCK + 2                       # past the fp16 d
    s = 0
    for l in range(2):
        q4 = int.from_bytes(row[base + 4 * (iqs + l):base + 4 * (iqs + l) + 4], "little")
        vx = 0
        vy = 0
        for i in range(4):
            vx |= (T[(q4 >> (8 * i)) & 0xF] & 0xFF) << (8 * i)
            vy |= (T[(q4 >> (8 * i + 4)) & 0xF] & 0xFF) << (8 * i)
        s += dp4a(vx, xq_words[kbx * 8 + iqs + l])
        s += dp4a(vy, xq_words[kbx * 8 + iqs + l + 4])
    return s


def row_ref(row, xq_words, xq_d):
    """The kernel's order: lane l owns calls l, l+32, then a 32-wide tree (WG=32 variant)."""
    lanes = []
    for lane in range(32):
        acc = 0.0
        for k in range(lane, NB * 2, 32):
            kbx, iqs = k // 2, 2 * (k % 2)
            dw = f16_at(row, kbx * BLOCK)
            acc = f32(acc + f32(f32(dw * xq_d[kbx]) * float(call_ref(row, xq_words, xq_d, kbx, iqs))))
        lanes.append(acc)
    for off in (16, 8, 4, 2, 1):
        for i in range(off):
            lanes[i] = f32(lanes[i] + lanes[i + off])
    return lanes[0]


def row_ref_float(row, xq_words, xq_d):
    """Naive dequantization, never touching the nibble tricks: the two agreeing says the block is read
    correctly.  Accumulated in fp64 on purpose - it is an independent check, not the kernel's order."""
    total = 0.0
    for kbx in range(NB):
        base = kbx * BLOCK
        d = f16_at(row, base)
        for j in range(16):
            lo = T[row[base + 2 + j] & 0xF]
            hi = T[row[base + 2 + j] >> 4]
            xi_lo = s8((xq_words[kbx * 8 + (j // 4)] >> (8 * (j % 4))) & 0xFF)
            xi_hi = s8((xq_words[kbx * 8 + 4 + (j // 4)] >> (8 * (j % 4))) & 0xFF)
            total += d * xq_d[kbx] * (lo * xi_lo + hi * xi_hi)
    return total


def main():
    import random
    rng = random.Random(0x1A40)

    rows = []
    for _ in range(N_EMBD):
        row = bytearray()
        for _ in range(NB):
            row += struct.pack("<e", rng.uniform(0.001, 0.02))
            row += bytes(rng.randrange(256) for _ in range(16))
        assert len(row) == ROW_BYTES
        rows.append(bytes(row))

    xq = quantize_q8_1([rng.uniform(-2.0, 2.0) for _ in range(N_FF)])
    assert len(xq) == NB
    xq_words = []
    xq_ds = []
    for d, qsb in xq:
        xq_ds.append(d)
        for w in range(8):
            xq_words.append(int.from_bytes(qsb[4 * w:4 * w + 4], "little"))

    refs = [row_ref(rows[r], xq_words, xq_ds) for r in range(REF_ROWS)]
    refs_f = [row_ref_float(rows[r], xq_words, xq_ds) for r in range(REF_ROWS)]
    worst = max(abs(a - b) for a, b in zip(refs, refs_f))
    scale = max(abs(v) for v in refs) or 1.0
    print(f"rows={N_EMBD} cols={N_FF} blocks/row={NB} row_bytes={ROW_BYTES} matrix={N_EMBD * ROW_BYTES / 1e6:.3f} MB")
    print(f"self-check  int vs naive dequant: max abs diff = {worst:.6f}  (max |ref| = {scale:.3f})")
    print(f"refs[0..3] = {[round(v, 6) for v in refs[:4]]}")

    d_arr = bytearray()
    qs_arr = bytearray()
    for row in rows:
        for kbx in range(NB):
            base = kbx * BLOCK
            d_arr += row[base:base + 2]
            qs_arr += row[base + 2:base + BLOCK]
    assert len(d_arr) == N_EMBD * NB * 2
    assert len(qs_arr) == N_EMBD * NB * 16

    with open(OUT, "wb") as f:
        f.write(struct.pack("<IIIII", N_EMBD, N_FF, NB, REF_ROWS, len(T)))
        f.write(d_arr)
        f.write(qs_arr)
        f.write(struct.pack(f"<{NB}e", *xq_ds))
        f.write(struct.pack(f"<{NB * 8}I", *xq_words))
        f.write(struct.pack("<4I", *[int.from_bytes(struct.pack("<4b", *T[4 * i:4 * i + 4]), "little") for i in range(4)]))
        f.write(struct.pack(f"<{REF_ROWS}f", *refs))
    print(f"wrote {OUT}")


if __name__ == "__main__":
    main()
