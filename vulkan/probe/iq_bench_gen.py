#!/usr/bin/env python3
"""Benchmark data for the Vulkan IQ3_XXS kernel, at the shape the engine actually uses.

The model's experts (GGUF metadata, qwen4exp): embedding_length 2560, expert_count 512,
expert_used_count 10, expert_feed_forward_length 640, block_count 48.  One expert's gate+up is
1280 rows of n_embd = 2560 values, which at IQ3_XXS (98 bytes per 256 values) is 1254400 bytes -
exactly the blob_bytes the pack's native_experts.txt reports for an IQ3_XXS/IQ4_NL layer.

Two weight layouts are written, both exactly 98 bytes per 256 values:

  packed.bin   the GGUF layout: fp16 d followed by 96 bytes of qs, one after another.  Byte 2 of
               every block means the 8-byte chunk a lane needs is never 4-byte aligned.
  split.bin    the same bytes separated into d[] (2 B/block), aux[] (32 B/block) and qs[]
               (64 B/block).  Every access the kernel makes lands on an aligned vector load and
               the byte count is identical, so the only thing that changes is how many load
               instructions each chunk costs.

Everything the kernel needs is in one file so the C++ side stays dumb:
  u32 n_rows, n_embd, nb, ref_rows  then d[], aux[], qs[] (split layout), x_ds, x_qs,
  256 u32 of codebook, refs.
"""

import struct
import sys

from iq3_ref import (
    BLOCK_BYTES, CHUNKS_PER_BLOCK, CB, cdiv, chunk_decode, f16_at, f32, quantize_q8_1, s8,
)

N_EMBD = 2560
NB = N_EMBD // 256              # 10 blocks per gate/up row
N_ROWS = 1280                   # gate (640) + up (640) of one expert
REF_ROWS = 32                   # rows the Python reference checks (the rest is timing only)
OUT = r"G:/01_Projects/Strata/vulkan/probe/iq_bench.bin"


def term(row, xq, k):
    """One 32-value chunk's contribution, in the engine's integer path."""
    kbx, c = k // CHUNKS_PER_BLOCK, k % CHUNKS_PER_BLOCK
    base = kbx * BLOCK_BYTES
    dw = f16_at(row, base)
    ls, g = chunk_decode(row[base + 2:base + BLOCK_BYTES], c)
    dx, xqbytes = xq[kbx * CHUNKS_PER_BLOCK + c]
    s = 0
    for j in range(8):
        w = (xqbytes[4 * j] | (xqbytes[4 * j + 1] << 8)
             | (xqbytes[4 * j + 2] << 16) | (xqbytes[4 * j + 3] << 24))
        s += sum(s8((g[j] >> (8 * i)) & 0xFF) * s8((w >> (8 * i)) & 0xFF) for i in range(4))
    sumi = float(cdiv(ls * s + cdiv(s, 2), 2))
    return f32(f32(dw * dx) * sumi)


def row_ref_int(row, xq):
    """The kernel's own summation order: lane l owns chunks l, l+32, l+64, then a 32-wide tree."""
    lanes = []
    for lane in range(32):
        acc = 0.0
        for k in range(lane, NB * CHUNKS_PER_BLOCK, 32):
            acc = f32(acc + term(row, xq, k))
        lanes.append(acc)
    for off in (16, 8, 4, 2, 1):
        for i in range(off):
            lanes[i] = f32(lanes[i] + lanes[i + off])
    return lanes[0]


def main():
    import random
    rng = random.Random(0x5EED)

    rows = []
    for _ in range(N_ROWS):
        row = bytearray()
        for _ in range(NB):
            row += struct.pack("<e", rng.uniform(0.001, 0.01))
            row += bytes(rng.randrange(256) for _ in range(96))
        assert len(row) == NB * BLOCK_BYTES
        rows.append(bytes(row))

    xq = quantize_q8_1([rng.uniform(-2.0, 2.0) for _ in range(N_EMBD)])
    assert len(xq) == NB * CHUNKS_PER_BLOCK

    refs = [row_ref_int(rows[r], xq) for r in range(REF_ROWS)]

    # ---- split layout ----
    d_arr = bytearray()
    aux_arr = bytearray()
    qs_arr = bytearray()
    for row in rows:
        for kbx in range(NB):
            base = kbx * BLOCK_BYTES
            d_arr += row[base:base + 2]
            aux_arr += row[base + 66:base + 98]      # the 8 aux32 words
            qs_arr += row[base + 2:base + 66]        # the 64 codebook indices
    assert len(d_arr) == N_ROWS * NB * 2
    assert len(aux_arr) == N_ROWS * NB * 8 * 4
    assert len(qs_arr) == N_ROWS * NB * 64

    x_ds = bytearray()
    x_qs = bytearray()
    for d, qsb in xq:
        x_ds += struct.pack("<e", d)
        x_qs += qsb

    with open(OUT, "wb") as f:
        f.write(struct.pack("<IIII", N_ROWS, N_EMBD, NB, REF_ROWS))
        f.write(d_arr)
        f.write(aux_arr)
        f.write(qs_arr)
        f.write(x_ds)
        f.write(x_qs)
        f.write(struct.pack("<256I", *CB))
        f.write(struct.pack(f"<{REF_ROWS}f", *refs))

    weight_bytes = N_ROWS * NB * BLOCK_BYTES
    x_bytes = len(x_ds) + len(x_qs)
    print(f"rows={N_ROWS} n_embd={N_EMBD} nb={NB}  weight={weight_bytes/1e6:.3f} MB  x={x_bytes} B")
    print(f"refs[0..3] = {[round(v, 5) for v in refs[:4]]}")
    print(f"wrote {OUT}  ({weight_bytes/1e6:.1f} MB + x + 1 KB codebook + {REF_ROWS * 4} B of refs)")


if __name__ == "__main__":
    main()
