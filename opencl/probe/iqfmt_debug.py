#!/usr/bin/env python3
"""Per-call diff of the integer path (Split<>::load+apply) against the reference decoder (dq_iqX).

When iqfmt_ref.py's self-check fails, the two paths disagree but not *where*.  This prints, for one
random block, the 32 weights each call produces under both readings, so the mismatch shows up as a
byte offset or a sign-bit index rather than as a number.
"""
import random
import struct
import sys

sys.path.insert(0, r"G:/01_Projects/Strata/opencl/probe")
import iqfmt_ref as R


def int_weights(t, row, base, iqs, xq_dummy):
    """The integer path's 32 signed weights for one call, as plain ints (no activation, no scale)."""
    if t == 21:
        g, ls = R.gu_iq3_s(row, base, iqs)
        return [R.s8((g[w] >> (8 * b)) & 0xFF) * ls for w in range(8) for b in range(4)]
    if t == 22:
        g, (ls0, ls1) = R.gu_iq2_s(row, base, iqs)
        # the two halves have different scales; report the raw words so the layout is visible
        return [R.s8((g[w] >> (8 * b)) & 0xFF) for w in range(8) for b in range(4)]
    if t == 23:
        g, sc = R.gu_iq4_xs(row, base, iqs)
        return [R.s8((g[w] >> (8 * b)) & 0xFF) for w in range(8) for b in range(4)]
    if t == 42:
        g, d2 = R.down_q2_0(row, base, iqs)
        return [R.s8((g[w] >> (8 * b)) & 0xFF) for w in range(8) for b in range(4)]
    raise ValueError(t)


def float_weights(t, row, base):
    """The reference decoder's weights for the block at `base` (dequant_row walks a whole row)."""
    return R.dequant_row(t, row)[base:]


def main(t):
    f = R.FMT[t]
    nb = f["row_vals"] // f["qk"]
    rng = random.Random(1234 + t)
    row = bytearray()
    for _ in range(nb):
        row += struct.pack("<e", rng.uniform(0.001, 0.01)) if t != 42 else struct.pack("<f", 0.01)
        row += bytes(rng.randrange(256) for _ in range(f["bb"] - (4 if t == 42 else 2)))

    print(f"=== {f['name']} (type {t})  bb={f['bb']} qk={f['qk']} ipb={f['ipb']} step={f['step']} ===")
    fi = float_weights(t, bytes(row), 0)
    for m in range(f["ipb"]):
        iqs = f["step"] * m
        ii = int_weights(t, bytes(row), 0, iqs, None)
        ff = fi[m * 32:(m + 1) * 32]
        # compare magnitudes only: the integer path returns unnormalised codes for most formats
        mag_i = [abs(v) for v in ii]
        mag_f = [abs(v) for v in ff]
        same = sum(1 for a, b in zip(mag_i, mag_f) if abs(a - b) < 1e-9)
        print(f"  call m={m} iqs={iqs}: |int| vs |float| matching in 32 positions: {same}/32")
        if same < 32:
            print(f"    int  : {ii}")
            print(f"    float: {[round(v, 6) for v in ff]}")
            print(f"    |int| : {mag_i}")
            print(f"    |flo| : {[round(v, 4) for v in mag_f]}")
            # where does the float block's multiset of magnitudes first appear?
            for off in range(0, 64):
                if mag_f[off:off + 8] == mag_i[off:off + 8]:
                    print(f"    >>> |int| matches |float| starting at index {off}")


if __name__ == "__main__":
    for t in ([int(a) for a in sys.argv[1:]] or [21, 23]):
        main(t)
