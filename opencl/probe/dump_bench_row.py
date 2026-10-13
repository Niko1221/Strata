#!/usr/bin/env python3
"""Dump row 0 of iq_bench.bin chunk by chunk, exactly as the OpenCL kernel reads it.

The OpenCL IQ3_XXS kernel came out wrong on the MI50 (max_err 75.6 against the Python reference, and
identical for every work-group size, so the fault is in the per-chunk decode rather than in the
reduction).  This prints the same five quantities the kernel's debug kernel writes out - qlo, qhi, a32,
s, sumi, and the three scales - so the two can be lined up line by line.

Usage: python dump_bench_row.py [row] [n_chunks]
"""

import struct
import sys

HERE = r"G:/01_Projects/Strata/vulkan/probe/iq_bench.bin"


def main():
    row_want = int(sys.argv[1]) if len(sys.argv) > 1 else 0
    show = int(sys.argv[2]) if len(sys.argv) > 2 else 4

    raw = open(HERE, "rb").read()
    n_rows, n_embd, nb, ref_rows = struct.unpack_from("<IIII", raw, 0)
    n_chunks = n_embd // 32
    off = 16

    d_bytes = n_rows * nb * 2
    aux_bytes = n_rows * nb * 8 * 4
    qs_bytes = n_rows * nb * 64
    d = raw[off:off + d_bytes]; off += d_bytes
    aux = raw[off:off + aux_bytes]; off += aux_bytes
    qs = raw[off:off + qs_bytes]; off += qs_bytes
    xd = raw[off:off + ((n_chunks + 1) // 2) * 4]; off += ((n_chunks + 1) // 2) * 4
    xq = raw[off:off + n_chunks * 8 * 4]; off += n_chunks * 8 * 4
    cb_raw = raw[off:off + 256 * 4]; off += 256 * 4
    refs = struct.unpack_from(f"<{ref_rows}f", raw, off)

    print(f"n_rows={n_rows} n_embd={n_embd} nb={nb} n_chunks={n_chunks} refs[{row_want}]={refs[row_want]}")

    # the u32 views the kernel uses
    d_u32 = struct.unpack(f"<{d_bytes // 4}I", d)
    aux_u32 = struct.unpack(f"<{aux_bytes // 4}I", aux)
    qs_u32 = struct.unpack(f"<{qs_bytes // 4}I", qs)
    xd_u32 = struct.unpack(f"<{len(xd) // 4}I", xd)
    xq_u32 = struct.unpack(f"<{len(xq) // 4}I", xq)
    cb = struct.unpack("<256I", cb_raw)

    def f16_at(buf, byteoff):
        v = (buf[byteoff >> 2] >> ((byteoff & 2) * 8)) & 0xFFFF
        return struct.unpack("<e", struct.pack("<H", v))[0]

    def s8(v):
        return v - 256 if v >= 128 else v

    def popcount(v):
        return bin(v & 0xFF).count("1")

    def unpack_ksigns(v):
        p = popcount(v & 0xFF) & 1
        return (v & 0xFF) ^ (p << 7)

    def vcmpne4(a, b):
        kHigh = 0x80808080
        t = (a ^ b) & 0xFFFFFFFF
        nz = ((((t & (~kHigh & 0xFFFFFFFF)) + (~kHigh & 0xFFFFFFFF)) & 0xFFFFFFFF) | t) & kHigh
        return (((nz >> 7) * 0xFF) & 0xFFFFFFFF)

    def vsub4(a, b):
        kHigh = 0x80808080
        ua, ub = a & 0xFFFFFFFF, b & 0xFFFFFFFF
        return (((ua | kHigh) - (ub & (~kHigh & 0xFFFFFFFF))) ^ ((ua ^ (~ub & 0xFFFFFFFF)) & kHigh)) & 0xFFFFFFFF

    def as_s32(v):
        return v - (1 << 32) if v >= (1 << 31) else v

    def dp4x8(a, b, c):
        # what four mad24s with a sign-extended byte compute
        s = c
        for i in range(4):
            x = ((a << (24 - 8 * i)) & 0xFFFFFFFF) >> 24
            y = ((b << (24 - 8 * i)) & 0xFFFFFFFF) >> 24
            # arithmetic shift right by 24
            x = x - 256 if x >= 128 else x
            y = y - 256 if y >= 128 else y
            s = (x * y + s) & 0xFFFFFFFF
        return as_s32(s)

    total = 0.0
    print(f"{'k':>3} {'qlo':>10} {'qhi':>10} {'a32':>10} {'s':>10} {'sumi':>10} {'dw':>12} {'dx':>12} {'term':>14} {'g[0..7]'}")
    for k in range(n_chunks):
        kbx, c = k // 8, k % 8
        qs_row = row_want * n_chunks
        a32 = aux_u32[row_want * n_chunks + k]
        qlo = qs_u32[(qs_row + k) * 2]
        qhi = qs_u32[(qs_row + k) * 2 + 1]

        g = [0] * 8
        for li in range(4):
            l0 = li * 2
            w = qlo if l0 < 4 else qhi
            sh0 = 8 * l0 if l0 < 4 else 8 * l0 - 32
            gx = cb[(w >> sh0) & 0xFF]
            gy = cb[(w >> (sh0 + 8)) & 0xFF]
            signs = (unpack_ksigns(a32 >> (7 * l0 // 2)) * 0x01010101) & 0xFFFFFFFF
            m0 = vcmpne4((signs & 0x08040201) & 0xFFFFFFFF, 0)
            m1 = vcmpne4((signs & 0x80402010) & 0xFFFFFFFF, 0)
            g[l0] = as_s32(vsub4(gx ^ m0, m0))
            g[l0 + 1] = as_s32(vsub4(gy ^ m1, m1))

        si = 0
        for j in range(8):
            si = dp4x8(g[j], as_s32(xq_u32[k * 8 + j]), si)

        sumi = (as_s32((a32 >> 28)) * si + (si // 2 if si >= 0 else -((-si) // 2))) // 2
        # python's // floors; C truncates.  Redo with truncation to match the kernel.
        num = as_s32(a32 >> 28) * si + int(si / 2)
        sumi = int(num / 2)

        dw = f16_at(d_u32, row_want * nb * 2 + kbx * 2)
        dx = f16_at(xd_u32, k * 2)   # fp16 element k lives at byte offset 2k
        term = dw * dx * sumi
        total += term
        if k < show:
            print(f"{k:>3} {qlo:>10x} {qhi:>10x} {a32:>10x} {si:>10} {sumi:>10} {dw:>12.6g} {dx:>12.6g} {term:>14.6f}")
            print("        g =", " ".join(f"{v:>11d}" for v in g))
            print("        xw=", " ".join(f"{v:>11d}" for v in [as_s32(xq_u32[k*8+j]) for j in range(8)]))
    print(f"sum over all chunks (fp64)      = {total:.6f}")
    print(f"reference from the Python model = {refs[row_want]:.6f}")


if __name__ == "__main__":
    main()
