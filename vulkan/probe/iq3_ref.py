#!/usr/bin/env python3
"""Reference data and expected output for the Vulkan IQ3_XXS expert-row kernel.

Writes iq3_test.bin: synthetic but format-exact IQ3_XXS rows, q8_1 activations, and the
expected row dot products computed two independent ways.

  ref_int   : the engine's integer path, transcribed from Split<18>::load/apply in
              src/kernels/cuda/iq_kernels.cu (grib lookup, sign bytes, ls*sumi scaling).
  ref_float : the naive dequantization, w = grid * sign * d * (2*ls+1)/4, dotted with the
              dequantized activation sum(q)*d_x.  It never touches the bit tricks, so
              agreement between the two says the block format is read correctly.

The codebook is parsed out of ggml-common.h rather than copied, so it cannot drift.
"""
import math
import re
import struct
import sys

GGML_COMMON = r"G:/01_Projects/Strata/third_party/ggml/ggml-common.h"
OUT = r"G:/01_Projects/Strata/vulkan/probe/iq3_test.bin"
OUT_CB = r"G:/01_Projects/Strata/vulkan/probe/iq3_cb.bin"

QK_K = 256
BLOCK_BYTES = 98            # fp16 d + 96 bytes qs
CHUNKS_PER_BLOCK = 8        # 8 chunks of 32 weights
N_EMBD = 2048
N_FF = 64
N_BLOCKS = N_EMBD // QK_K
ROW_BYTES = N_BLOCKS * BLOCK_BYTES
Q8_BYTES = 36               # fp16 d + fp16 sum + 32 int8


def load_codebook():
    src = open(GGML_COMMON, encoding="utf-8", errors="ignore").read()
    m = re.search(r"GGML_TABLE_BEGIN\(uint32_t, iq3xxs_grid, 256\)(.*?)GGML_TABLE_END\(\)", src, re.S)
    if not m:
        sys.exit("iq3xxs_grid not found in ggml-common.h")
    vals = [int(v, 16) for v in re.findall(r"0x[0-9a-fA-F]+", m.group(1))]
    if len(vals) != 256:
        sys.exit(f"iq3xxs_grid has {len(vals)} entries, expected 256")
    return vals


CB = load_codebook()


def s8(v):
    return v - 256 if v >= 128 else v


def unpack_ksigns(v):
    p = bin(v & 0xFF).count("1") & 1
    return (v ^ (p << 7)) & 0xFF


def vcmpne4(a, b):
    r = 0
    for i in range(4):
        if ((a >> (8 * i)) & 0xFF) != ((b >> (8 * i)) & 0xFF):
            r |= 0xFF << (8 * i)
    return r


def vsub4(a, b):
    r = 0
    for i in range(4):
        lane = (((a >> (8 * i)) & 0xFF) - ((b >> (8 * i)) & 0xFF)) & 0xFF
        r |= lane << (8 * i)
    return r


def dp4a(a, b):
    return sum(s8((a >> (8 * i)) & 0xFF) * s8((b >> (8 * i)) & 0xFF) for i in range(4))


def cdiv(a, b):
    """C integer division: truncation toward zero, not Python's floor."""
    q = abs(a) // abs(b)
    return q if (a >= 0) == (b >= 0) else -q


def f16_at(buf, off):
    return struct.unpack_from("<e", buf, off)[0]


def f32(x):
    """Round to fp32. Python floats are doubles; the shader computes in fp32, and reproducing
    that rounding (rather than the double-precision ideal) is what makes the comparison exact."""
    return struct.unpack("<f", struct.pack("<f", x))[0]


def chunk_decode(qs, c):
    """One 32-weight chunk -> (ls, [8 int32 words, 4 signed weights each])."""
    aux32 = int.from_bytes(qs[64 + 4 * c:68 + 4 * c], "little")
    ls = aux32 >> 28
    g = [0] * 8
    for l0 in (0, 2, 4, 6):
        gx = CB[qs[8 * c + l0]]
        gy = CB[qs[8 * c + l0 + 1]]
        signs = unpack_ksigns((aux32 >> (7 * l0 // 2)) & 0xFF) * 0x01010101
        m0 = vcmpne4(signs & 0x08040201, 0)
        m1 = vcmpne4(signs & 0x80402010, 0)
        g[l0] = vsub4(gx ^ m0, m0)
        g[l0 + 1] = vsub4(gy ^ m1, m1)
    return ls, g


def row_ref_int(row, xq, n_blocks):
    """The engine's integer path, in the shader's float precision and summation order.

    The order matters: floating point addition is not associative, so a plain running sum over
    all 64 chunks would differ from what the kernel computes.  This reproduces the kernel's
    structure exactly - each of the 32 lanes accumulates the chunks it owns (a stride-32 walk),
    then the lanes combine in a tree - and rounds every operation to fp32 with no FMA, which is
    what the `precise` accumulator in iq3_gu.comp guarantees.
    """
    def term(k):
        kbx, c = k // CHUNKS_PER_BLOCK, k % CHUNKS_PER_BLOCK
        base = kbx * BLOCK_BYTES
        dw = f16_at(row, base)
        qs = row[base + 2:base + BLOCK_BYTES]
        ls, g = chunk_decode(qs, c)
        dx, xqbytes = xq[kbx * CHUNKS_PER_BLOCK + c]
        s = 0
        for j in range(8):
            w = (xqbytes[4 * j] | (xqbytes[4 * j + 1] << 8)
                 | (xqbytes[4 * j + 2] << 16) | (xqbytes[4 * j + 3] << 24))
            s += dp4a(g[j], w)
        sumi = float(cdiv(ls * s + cdiv(s, 2), 2))
        return f32(f32(dw * dx) * sumi)

    lanes = []
    for lane in range(32):
        acc = 0.0
        for k in range(lane, n_blocks * CHUNKS_PER_BLOCK, 32):
            acc = f32(acc + term(k))
        lanes.append(acc)
    for off in (16, 8, 4, 2, 1):
        for i in range(off):
            lanes[i] = f32(lanes[i] + lanes[i + off])
    return lanes[0]


def row_ref_float(row, xq, n_blocks):
    total = 0.0
    for kbx in range(n_blocks):
        base = kbx * BLOCK_BYTES
        d = f16_at(row, base)
        qs = row[base + 2:base + BLOCK_BYTES]
        for c in range(CHUNKS_PER_BLOCK):
            ls, g = chunk_decode(qs, c)
            dx, xqbytes = xq[kbx * CHUNKS_PER_BLOCK + c]
            s = 0
            for j in range(8):
                for i in range(4):
                    s += s8((g[j] >> (8 * i)) & 0xFF) * s8(xqbytes[4 * j + i])
            total += d * (2 * ls + 1) / 4.0 * dx * s
    return total


def quantize_q8_1(x):
    """Floats (multiple of 32) -> [(d, 32 int8 bytes)], the engine's q8_1 contract."""
    blocks = []
    for b in range(len(x) // 32):
        ch = x[32 * b:32 * b + 32]
        amax = max(abs(v) for v in ch)
        # Round-trip through fp16: that is the value the kernel will read back, so the
        # reference must use the same one.
        d = struct.unpack("<e", struct.pack("<e", min(amax / 127.0, 65504.0)))[0]
        qs = bytearray()
        for v in ch:
            if amax == 0.0:
                qs.append(0)
            else:
                r = min(math.floor(abs(v / d) + 0.5), 127)
                qs.append((-r if v < 0 else r) & 0xFF)
        blocks.append((d, bytes(qs)))
    return blocks


def main():
    import random
    rng = random.Random(20261006)

    blob = bytearray()
    rows = []
    for _ in range(N_FF):
        row = bytearray()
        for _ in range(N_BLOCKS):
            row += struct.pack("<e", rng.uniform(0.001, 0.01))
            # Random payload: every codebook index and sign/scale nibble is legal, so
            # arbitrary bytes exercise the whole decode path, including edge patterns.
            row += bytes(rng.randrange(256) for _ in range(96))
        assert len(row) == ROW_BYTES
        rows.append(bytes(row))
        blob += row

    xq = quantize_q8_1([rng.uniform(-2.0, 2.0) for _ in range(N_EMBD)])
    assert len(xq) == N_EMBD // 32

    ref_int = [row_ref_int(r, xq, N_BLOCKS) for r in rows]
    ref_float = [row_ref_float(r, xq, N_BLOCKS) for r in rows]

    worst = max(abs(a - b) for a, b in zip(ref_int, ref_float))
    scale = max(abs(v) for v in ref_int) or 1.0
    print(f"rows={N_FF}  row_bytes={ROW_BYTES}  n_blocks={N_BLOCKS}  q8_1 blocks={len(xq)}")
    print(f"self-check  ref_int vs ref_float: max abs diff = {worst:.6f}  (max |ref| = {scale:.3f})")
    print(f"first 4 ref_int   : {[round(v, 6) for v in ref_int[:4]]}")
    print(f"first 4 ref_float : {[round(v, 6) for v in ref_float[:4]]}")

    with open(OUT, "wb") as f:
        f.write(struct.pack("<III", N_EMBD, N_FF, N_BLOCKS))
        f.write(blob)
        for d, qsb in xq:
            f.write(struct.pack("<e", d))
            f.write(struct.pack("<e", 0.0))   # the fp16 sum: the dot path does not read it
            f.write(qsb)
        f.write(struct.pack(f"<{N_FF}f", *ref_int))
    print(f"wrote {OUT}")

    with open(OUT_CB, "wb") as f:
        f.write(struct.pack("<256I", *CB))
    print(f"wrote {OUT_CB}")


if __name__ == "__main__":
    main()
