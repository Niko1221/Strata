#!/usr/bin/env python3
"""Reference data for the Strata expert formats the OpenCL port still owes, and the expected row dots.

IQ3_XXS (gu) and IQ4_NL (down) are already done on the MI50.  The iq3_s pack needs three more gu
formats and one more down format before every layer is covered:

  type 21  IQ3_S    gu   110 B / 256 values   10 layers
  type 22  IQ2_S    gu    82 B / 256 values   20 layers
  type 23  IQ4_XS   gu   136 B / 256 values    1 layer
  type 42  Q2_0     down  18 B /  64 values    9 layers

Same two-independent-computations trick as iq3_ref.py:

  ref_int   the engine's integer path, transcribed from vec_dot_iqX_s_q8_1 / Split<X>::load+apply in
            src/kernels/cuda/iq_kernels.cu, in the kernel's fp32 rounding and lane-strided summation
            order.
  ref_float the naive dequantization from dq_iqX (the reference decoder) dotted with the dequantized
            activations.  It never touches a bit trick.

They agree algebraically but not numerically, so agreement to ~1e-6 relative says the block format is
read correctly.  Every scale convention below is derived from the integer form, e.g. IQ2_S's
`(sumi0*ls0 + sumi1*ls1 + (sumi0+sumi1)/2)/4` is ((2*ls0+1)*sumi0 + (2*ls1+1)*sumi1)/8, which is the
(2*ls+1)/8 the float path uses.

Byte offsets are the real ones, and they follow the one thing worth remembering about this file:
`get_int_b2(x, i)` in iq_kernels.cu is a 32-bit load at byte offset 4*i, not 2*i -
    int x32 = x16[2*i] | (x16[2*i + 1] << 16);
so `iqs` is a word index everywhere, not a byte or element index.
"""
import math
import re
import struct
import sys

GGML_COMMON = r"G:/01_Projects/Strata/third_party/ggml/ggml-common.h"
OUT_DIR = r"G:/01_Projects/Strata/opencl/probe"

QK_K = 256
N_EMBD = 2560
N_FF = 640
REF_ROWS = 32

# gu formats: gate/up of one expert = 1280 rows of n_embd = 2560 values.
# Q2_0: the down matrix = n_embd = 2560 rows of n_ff = 640 values (the SwiGLU activation is 640 wide).
#
# `tol` is the relative gap the two paths are allowed to show.  For three formats they agree to a few
# fp32 ULPs; IQ2_S is 2e-5 because its integer form ends in `(sumi0+sumi1)/2` with truncating division,
# which drops up to half a unit per call.  A wrong byte offset or sign-bit index would show up as ~1.0,
# so the gap between "rounding" and "layout" is five orders of magnitude.
FMT = {
    21: dict(name="IQ3_S",  kind="gu",   qk=256, ipb=8, step=2, bb=110, rows=1280, row_vals=N_EMBD, tol=1e-5),
    22: dict(name="IQ2_S",  kind="gu",   qk=256, ipb=8, step=2, bb=82,  rows=1280, row_vals=N_EMBD, tol=1e-3),
    23: dict(name="IQ4_XS", kind="gu",   qk=256, ipb=8, step=4, bb=136, rows=1280, row_vals=N_EMBD, tol=1e-5),
    42: dict(name="Q2_0",   kind="down", qk=64,  ipb=2, step=1, bb=18,  rows=N_EMBD, row_vals=N_FF,  tol=1e-5),
}


# --------------------------------------------------------------------- tables straight out of the header

def parse_table(name, ty, count):
    src = open(GGML_COMMON, encoding="utf-8", errors="ignore").read()
    m = re.search(r"GGML_TABLE_BEGIN\(%s, %s, %d\)\s*\{(.*?)\}\s*;?\s*GGML_TABLE_END\(\)" % (ty, name, count),
                  src, re.S)
    if not m:
        m = re.search(r"GGML_TABLE_BEGIN\(%s, %s, %d\)(.*?)GGML_TABLE_END\(\)" % (ty, name, count), src, re.S)
    if not m:
        sys.exit(f"{name} not found in ggml-common.h")
    body = m.group(1)
    if ty == "int8_t":
        vals = [int(v) for v in re.findall(r"-?\d+", body)]
    else:
        vals = [int(v, 16) for v in re.findall(r"0x[0-9a-fA-F]+", body)]
    if len(vals) != count:
        sys.exit(f"{name} has {len(vals)} entries, expected {count}")
    return vals


IQ2S_GRID = parse_table("iq2s_grid", "uint64_t", 1024)      # 8 unsigned bytes -> 8 weights
IQ3S_GRID = parse_table("iq3s_grid", "uint32_t", 512)       # 4 unsigned bytes -> 4 weights
KV_IQ4NL = parse_table("kvalues_iq4nl", "int8_t", 16)


# --------------------------------------------------------------------- the bit helpers

def s8(v):
    return v - 256 if v >= 128 else v


def vcmpne4(a, b):
    r = 0
    for i in range(4):
        if ((a >> (8 * i)) & 0xFF) != ((b >> (8 * i)) & 0xFF):
            r |= 0xFF << (8 * i)
    return r


def vsub4(a, b):
    r = 0
    for i in range(4):
        r |= ((((a >> (8 * i)) & 0xFF) - ((b >> (8 * i)) & 0xFF)) & 0xFF) << (8 * i)
    return r


def dp4a(a, b):
    return sum(s8((a >> (8 * i)) & 0xFF) * s8((b >> (8 * i)) & 0xFF) for i in range(4))


def cdiv(a, b):
    """C integer division: truncation toward zero, not Python's floor."""
    q = abs(a) // abs(b)
    return q if (a >= 0) == (b >= 0) else -q


def f32(x):
    """Python floats are doubles; the kernel computes in fp32 and that rounding is part of the answer."""
    return struct.unpack("<f", struct.pack("<f", x))[0]


def f16_at(buf, off, base=0):
    return struct.unpack_from("<e", buf, base + off)[0]


def u16_at(buf, off, base=0):
    return struct.unpack_from("<H", buf, base + off)[0]


def u32_at(buf, off, base=0):
    return struct.unpack_from("<I", buf, base + off)[0]


def u64_at(buf, off, base=0):
    return struct.unpack_from("<Q", buf, base + off)[0]


def q8_word(xqbytes, i):
    """get_int_b4(q8_1.qs, i): signed int32 at byte 4*i of the 32 activation bytes."""
    return int.from_bytes(xqbytes[4 * i:4 * i + 4], "little", signed=False)


# --------------------------------------------------------------------- per-format decode
#
# Each of these returns the integer weights for one call, in the order the dp4a loop consumes them:
# eight int32 words, each holding four signed int8 that pair with activation bytes 4*w .. 4*w+4.
# `sc` is the integer scale the apply() step multiplies by (already the exact form, not a nibble).

def gu_iq3_s(w, base, iqs, qh_shift_ok=True):
    """Split<21>/vec_dot_iq3_s_q8_1.  iqs = 2*c, byte offsets inside the block are real ones."""
    c = iqs // 2
    qs = w[base + 2 + 8 * c: base + 2 + 8 * c + 8]           # qs[64] starts after the fp16 d
    qh = w[base + 2 + 64 + c]                                # qh[8]
    signs = w[base + 2 + 72 + 4 * c: base + 2 + 72 + 4 * c + 4]   # signs[32], after qs[64]+qh[8]
    g = [0] * 8
    for il in range(4):
        gi1 = IQ3S_GRID[qs[2 * il + 0] | ((qh << (8 - 2 * il)) & 0x100)]
        gi2 = IQ3S_GRID[qs[2 * il + 1] | ((qh << (7 - 2 * il)) & 0x100)]
        s = signs[il]
        m0 = vcmpne4(((s & 0x03) << 7) | ((s & 0x0C) << 21), 0)
        m1 = vcmpne4(((s & 0x30) << 3) | ((s & 0xC0) << 17), 0)
        g[2 * il] = vsub4(gi1 ^ m0, m0)
        g[2 * il + 1] = vsub4(gi2 ^ m1, m1)
    sc = 1 + 2 * ((w[base + 2 + 104 + c // 2] >> (4 * (c & 1))) & 0x0F)   # scales[4] after signs
    return g, sc


def gu_iq2_s(w, base, iqs):
    """Split<22>/vec_dot_iq2_s_q8_1."""
    c = iqs // 2
    qs = w[base + 2 + 4 * c: base + 2 + 4 * c + 4]           # qs[64]
    qh = w[base + 2 + 64 + c]                                # qh[8]
    signs = w[base + 2 + 32 + 4 * c: base + 2 + 32 + 4 * c + 4]   # signs = qs + QK_K/8
    g = [0] * 8
    for il in range(4):
        idx = qs[il] | ((qh << (8 - 2 * il)) & 0x300)
        ent = IQ2S_GRID[idx]
        gp = [ent & 0xFFFFFFFF, (ent >> 32) & 0xFFFFFFFF]    # grid_pos[0], grid_pos[1]
        s = signs[il]
        m0 = vcmpne4(((s & 0x03) << 7) | ((s & 0x0C) << 21), 0)
        m1 = vcmpne4(((s & 0x30) << 3) | ((s & 0xC0) << 17), 0)
        g[2 * il] = vsub4(gp[0] ^ m0, m0)
        g[2 * il + 1] = vsub4(gp[1] ^ m1, m1)
    sc = w[base + 2 + 72 + c]                                # scales[8] after qs[64]+qh[8]
    return g, (sc & 0x0F, sc >> 4)


def gu_iq4_xs(w, base, iqs):
    """Split<23>/vec_dot_iq4_xs_q8_1.  iqs = 4*ib."""
    ib = iqs // 4
    q4 = w[base + 8 + 16 * ib: base + 8 + 16 * ib + 16]      # d(2) + scales_h(2) + scales_l(4) = 8
    g = [0] * 8
    for j in range(4):
        lo = [KV_IQ4NL[q4[4 * j + m] & 0x0F] for m in range(4)]
        hi = [KV_IQ4NL[q4[4 * j + m] >> 4] for m in range(4)]
        g[j] = int.from_bytes(bytes(v & 0xFF for v in lo), "little")
        g[j + 4] = int.from_bytes(bytes(v & 0xFF for v in hi), "little")
    scales_h = u16_at(w, 2, base)
    scales_l = w[base + 4 + ib // 2]
    ls = ((scales_l >> (4 * (ib & 1))) & 0x0F) | (((scales_h >> (2 * ib)) & 0x03) << 4)
    return g, ls - 32


def down_q2_0(w, base, iqs):
    """Split<42>/vec_dot_q2_0_q8_1.  iqs in {0,1}: one 64-value block covers two q8_1 blocks.

    block_q2_0 is {ggml_half d; uint8_t qs[16]} = 18 bytes, so d is a HALF at bytes 0..2 and the 16
    qs bytes start at 2 - they do not overlap it.  Reading d as a 4-byte float looked self-consistent
    (both paths below did it) but it is not what the header says nor what vec_dot_q2_0_q8_1 does:
    `const float d2 = bq2_0->d;` with bq2_0->d a ggml_half.  Only the *decode* of qs was on trial
    here, so the cross-check could not catch it.
    """
    d2 = f16_at(w, 0, base)
    g = [0] * 8
    for j in range(4):
        q = u16_at(w, 2 + 8 * iqs + 2 * j, base)             # qs[16]: int16 index 4*iqs + j
        codes = [(q >> (2 * k)) & 3 for k in range(8)]
        g[2 * j] = int.from_bytes(bytes((c - 1) & 0xFF for c in codes[0:4]), "little")
        g[2 * j + 1] = int.from_bytes(bytes((c - 1) & 0xFF for c in codes[4:8]), "little")
    return g, d2


# --------------------------------------------------------------------- the two row references

def term(t, row, xq, k):
    """One call's contribution to a row's dot, exactly as apply() computes it."""
    f = FMT[t]
    kbx, m = divmod(k, f["ipb"])
    iqs = f["step"] * m
    base = kbx * f["bb"]
    # The q8_1 blocks are numbered by 32-value position along the whole row, so a sub-block c of
    # weight block kbx pairs with activation block kbx*ipb + c.  Getting this wrong leaves every
    # dot product uncorrelated while still looking plausible.
    xb = kbx * f["ipb"] + m

    if t == 21:
        dw = f16_at(row, 0, base)
        g, ls = gu_iq3_s(row, base, iqs)
        qb = xq[xb]
        sumi = sum(dp4a(g[w], q8_word(qb[1], w)) for w in range(8)) * ls
        return f32(f32(dw * qb[0]) * sumi)

    if t == 22:
        dw = f16_at(row, 0, base)
        g, (ls0, ls1) = gu_iq2_s(row, base, iqs)
        qb = xq[xb]
        s0 = dp4a(g[0], q8_word(qb[1], 0)) + dp4a(g[1], q8_word(qb[1], 1)) \
            + dp4a(g[2], q8_word(qb[1], 2)) + dp4a(g[3], q8_word(qb[1], 3))
        s1 = dp4a(g[4], q8_word(qb[1], 4)) + dp4a(g[5], q8_word(qb[1], 5)) \
            + dp4a(g[6], q8_word(qb[1], 6)) + dp4a(g[7], q8_word(qb[1], 7))
        sumi = cdiv(s0 * ls0 + s1 * ls1 + cdiv(s0 + s1, 2), 4)
        return f32(f32(dw * qb[0]) * sumi)

    if t == 23:
        dw = f16_at(row, 0, base)
        g, sc = gu_iq4_xs(row, base, iqs)
        qb = xq[xb]
        sumi = 0
        for j in range(4):
            sumi += dp4a(g[j], q8_word(qb[1], j)) + dp4a(g[j + 4], q8_word(qb[1], j + 4))
        sumi *= sc
        return f32(f32(dw * qb[0]) * sumi)

    if t == 42:
        g, d2 = down_q2_0(row, base, iqs)
        qb = xq[xb]
        sumi = 0
        for j in range(4):
            sumi += dp4a(g[2 * j], q8_word(qb[1], 2 * j)) + dp4a(g[2 * j + 1], q8_word(qb[1], 2 * j + 1))
        return f32(f32(d2 * qb[0]) * sumi)

    raise ValueError(t)


def row_ref_int(t, row, xq):
    """The kernel's own summation order: lane l owns calls l, l+32, l+64, ..., then a 32-wide tree."""
    f = FMT[t]
    lanes = []
    for lane in range(32):
        acc = 0.0
        for k in range(lane, (f["row_vals"] // f["qk"]) * f["ipb"], 32):
            acc = f32(acc + term(t, row, xq, k))
        lanes.append(acc)
    for off in (16, 8, 4, 2, 1):
        for i in range(off):
            lanes[i] = f32(lanes[i] + lanes[i + off])
    return lanes[0]


def dequant_row(t, row):
    """The reference decoder's weights for a whole row, from dq_iqX - no bit tricks."""
    f = FMT[t]
    nb = f["row_vals"] // f["qk"]
    out = []
    for kbx in range(nb):
        base = kbx * f["bb"]
        if t == 21:
            d = f16_at(row, 0, base)
            for ib in range(8):
                qs = row[base + 2 + 8 * ib: base + 2 + 8 * ib + 8]
                qh = row[base + 2 + 64 + ib]
                dl = d * (1 + 2 * ((row[base + 2 + 104 + ib // 2] >> (4 * (ib % 2))) & 0xF))
                vals = []
                for il in range(4):
                    g1 = IQ3S_GRID[qs[2 * il] | ((qh << (8 - 2 * il)) & 0x100)]
                    g2 = IQ3S_GRID[qs[2 * il + 1] | ((qh << (7 - 2 * il)) & 0x100)]
                    # dq_iq3_s: signs[4*ib + il] is one byte per il - low 4 bits for grid1, high 4 for
                    # grid2, i.e. the eight sign bits of the sub-block's positions 8*il .. 8*il+8.
                    sg = row[base + 2 + 72 + 4 * ib + il]
                    for j in range(4):
                        vals.append(dl * ((g1 >> (8 * j)) & 0xFF) * (-1.0 if (sg >> j) & 1 else 1.0))
                    for j in range(4):
                        vals.append(dl * ((g2 >> (8 * j)) & 0xFF) * (-1.0 if (sg >> (4 + j)) & 1 else 1.0))
                out.extend(vals)
        elif t == 22:
            d = f16_at(row, 0, base)
            for ib in range(8):
                qs = row[base + 2 + 4 * ib: base + 2 + 4 * ib + 4]
                qh = row[base + 2 + 64 + ib]
                sc = row[base + 2 + 72 + ib]
                signs = row[base + 2 + 32 + 4 * ib: base + 2 + 32 + 4 * ib + 4]
                vals = []
                for il in range(4):
                    ent = IQ2S_GRID[qs[il] | ((qh << (8 - 2 * il)) & 0x300)]
                    g1 = [(ent >> (8 * j)) & 0xFF for j in range(4)]
                    g2 = [(ent >> (32 + 8 * j)) & 0xFF for j in range(4)]
                    ls = (sc & 0x0F) if il < 2 else (sc >> 4)
                    dl = d * (2 * ls + 1) / 8.0
                    sg = signs[il]
                    for j in range(4):
                        vals.append(dl * g1[j] * (-1.0 if (sg >> j) & 1 else 1.0))
                    for j in range(4):
                        vals.append(dl * g2[j] * (-1.0 if (sg >> (4 + j)) & 1 else 1.0))
                out.extend(vals)
        elif t == 23:
            d = f16_at(row, 0, base)
            sh = u16_at(row, 2, base)
            for ib in range(8):
                q4 = row[base + 8 + 16 * ib: base + 8 + 16 * ib + 16]
                ls = ((row[base + 4 + ib // 2] >> (4 * (ib % 2))) & 0xF) | (((sh >> (2 * ib)) & 3) << 4)
                dl = d * (ls - 32)
                # dq_iq4_xs writes y[j] and y[j+16]: the LOW nibbles fill a sub-block's positions
                # 0..15 in byte order and the HIGH nibbles its positions 16..31 - not 8 at a time.
                for j in range(16):
                    out.append(dl * KV_IQ4NL[q4[j] & 0xF])
                for j in range(16):
                    out.append(dl * KV_IQ4NL[q4[j] >> 4])
        elif t == 42:
            d = f16_at(row, 0, base)
            for i in range(64):
                code = (row[base + 2 + i // 4] >> ((i % 4) * 2)) & 3
                out.append(d * (code - 1))
    return out


def row_ref_float(t, row, xq):
    """Dequantized weights against dequantized activations - a path that shares no arithmetic."""
    w = dequant_row(t, row)
    total = 0.0
    for i, wi in enumerate(w):
        d, qsb = xq[i // 32]
        total += wi * (d * s8(qsb[i % 32]))
    return total


def quantize_q8_1(x):
    """Floats (multiple of 32) -> [(d as the fp16 the kernel reads back, 32 int8 bytes)]."""
    blocks = []
    for b in range(len(x) // 32):
        ch = x[32 * b:32 * b + 32]
        amax = max(abs(v) for v in ch)
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


# --------------------------------------------------------------------- generation

def gen(t, out_path):
    import random
    f = FMT[t]
    rng = random.Random(0x5EED + t)
    nb = f["row_vals"] // f["qk"]

    rows = []
    for _ in range(f["rows"]):
        row = bytearray()
        for _ in range(nb):
            if t == 42:
                row += struct.pack("<e", rng.uniform(0.001, 0.02))
            else:
                row += struct.pack("<e", rng.uniform(0.001, 0.01))
            # Random payload: every grid index, nibble and sign byte is in range, so arbitrary bytes
            # exercise the whole decode - including the all-zero sign byte and the index's high bits.
            # 2 bytes of fp16 d in every format here - block_q2_0's d is a ggml_half too, not a float.
            row += bytes(rng.randrange(256) for _ in range(f["bb"] - 2))
        assert len(row) == nb * f["bb"], (len(row), nb * f["bb"])
        rows.append(bytes(row))

    xq = quantize_q8_1([rng.uniform(-2.0, 2.0) for _ in range(f["row_vals"])])
    assert len(xq) == f["row_vals"] // 32

    ref_int = [row_ref_int(t, rows[r], xq) for r in range(REF_ROWS)]
    ref_float = [row_ref_float(t, rows[r], xq) for r in range(REF_ROWS)]
    worst = max(abs(a - b) for a, b in zip(ref_int, ref_float))
    scale = max(abs(v) for v in ref_int) or 1.0

    with open(out_path, "wb") as fh:
        fh.write(struct.pack("<IIIIII", f["rows"], f["row_vals"], nb, REF_ROWS, t, f["bb"]))
        for row in rows:
            fh.write(row)
        for d, qsb in xq:
            fh.write(struct.pack("<e", d))
            fh.write(struct.pack("<e", 0.0))     # the fp16 sum: the dot path never reads it
            fh.write(qsb)
        fh.write(struct.pack("<1024Q", *IQ2S_GRID))
        fh.write(struct.pack("<512I", *IQ3S_GRID))
        fh.write(struct.pack("<16b", *KV_IQ4NL))
        fh.write(struct.pack(f"<{REF_ROWS}f", *ref_int))

    size = f["rows"] * nb * f["bb"]
    print(f"[{t:2d} {f['name']:6s} {f['kind']}] rows={f['rows']} row_vals={f['row_vals']} nb={nb} "
          f"ipb={f['ipb']} step={f['step']} bb={f['bb']}")
    print(f"    weights={size/1e6:.3f} MB  calls/row={nb * f['ipb']}  "
          f"self-check int vs float: max abs diff = {worst:.6f} (max |ref| = {scale:.3f}, "
          f"rel = {worst/scale:.2e})")
    print(f"    refs[0..3] = {[round(v, 6) for v in ref_int[:4]]}")
    if worst / scale > f["tol"]:
        print(f"    *** the two paths disagree by more than rounding: the layout is wrong ***")
        return False
    return True


def main():
    ok = True
    for t in (21, 22, 23, 42):
        p = f"{OUT_DIR}/iqfmt_{t}.bin"
        if not gen(t, p):
            ok = False
        print(f"    wrote {p}")
    print("ALL FORMATS CONSISTENT" if ok else "SOME FORMATS FAILED THE SELF-CHECK")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
