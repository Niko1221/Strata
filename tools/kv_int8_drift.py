#!/usr/bin/env python
"""Does Strata's int8 KV quantization itself shift the mean?

The question (2026-11-11): we measure our arms against the int8 KV cache.  Does that int8 codec
itself introduce a mean drift or bias against fp16?  The answer decides what our reference is
worth: a reference that carries a bias of its own is not the same yardstick as a clean one.

The int8 KV codec (include/strata/kernels/kv_q8.hpp, verbatim):

    scale = fp16(max|x| / 127),  code = clamp(rint(x / scale), -127, 127),  x' = fp16(code * scale)

per group of 64 values, per cell, per KV head, on the **unrotated** rows (the int8 path does not
rotate; only the q4 modes do).  So the codec is symmetric with no zero point: round-to-nearest is
unbiased per value, and the only systematic part is the **fp16 rounding of the scale**, which
multiplies the group's values by (1 + eps) and therefore scales the group's mean by the same
relative amount.

The measurement: the cell dump holds the *rotated* fp16 rows; the round trip H(H(y)) is exact, so
un-rotating gives the fp16 rows the engine quantizes.  On those rows the script applies the int8
codec and the 4-bit codecs and measures the mean drift per channel, in units of the channel's own
standard deviation, plus the multiplicative drift of the fp16 scale.

Usage:
  python tools/kv_int8_drift.py --dump <dump.bin> [--max-cells N]
"""
import argparse
import importlib.util
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
spec = importlib.util.spec_from_file_location("kv_codec_mse", HERE / "kv_codec_mse.py")
kq = importlib.util.module_from_spec(spec)
spec.loader.exec_module(kq)

GROUP = 64


def to_fp16(x):
    """The engine's fp16 rounding (the scale and the dequantized value are fp16)."""
    return np.asarray(x, dtype=np.float32).astype(np.float16).astype(np.float64)


def int8_codec(x):
    """scale = fp16(max|x|/127) per group of 64, code = clamp(rint(x/scale)), x' = fp16(code*scale).

    x: (..., n) with the last axis the group axis (the head dimension, 256 = 4 groups).
    Returns the reconstruction, the fp16 scale and the diagnostics the clamp question needs:
    how many values the clamp actually catches, and how far the extreme value is shrunk.
    """
    sh = x.shape
    g = x.reshape(-1, sh[-1] // GROUP, GROUP)
    amax = np.max(np.abs(g), axis=2, keepdims=True)
    scale = to_fp16(amax / 127.0)
    scale = np.where(scale > 0, scale, 1.0)
    raw = g / scale
    code = np.clip(np.rint(raw), -127, 127)
    clamped = int(np.sum(np.abs(raw) > 127.5))
    out = to_fp16(code * scale)
    # the extreme value's reconstruction against the original: the systematic part of the scale's
    # fp16 rounding (the scale rounds down -> the max is reconstructed below the original)
    top = np.argmax(np.abs(g), axis=2, keepdims=True)
    x_top = np.take_along_axis(g, top, axis=2)
    r_top = np.take_along_axis(out.reshape(-1, sh[-1] // GROUP, GROUP), top, axis=2)
    return out.reshape(sh), scale, clamped, float(np.mean(r_top - x_top)), float(np.mean(np.abs(r_top - x_top)))


def uniform_max(x, levels=16):
    """The 4-bit codec's scale rule: d = amax / lmax over the block of 32, then round to the grid."""
    lmax = {16: 87.0, 8: 69.0}[levels]
    sh = x.shape
    b = x.reshape(-1, sh[-1] // 32, 32)
    d = np.max(np.abs(b), axis=2, keepdims=True) / lmax
    code = np.clip(np.rint(b / np.where(d > 0, d, 1.0)), -lmax, lmax)
    return (code * d).reshape(sh)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dump", required=True)
    ap.add_argument("--max-cells", type=int, default=None)
    args = ap.parse_args()

    recs = kq.read_dump(args.dump)
    if not recs:
        raise SystemExit("no records in the dump")
    print(f"dump: {len(recs)} records, layers {sorted(set(r[0] for r in recs))}")

    # per (side, layer, head): the sums we need
    acc = {}

    def add(key, n, sum_ref, sum_rec, sum_ref2):
        a, b, c, d = acc.get(key, (0.0, 0.0, 0.0, 0))
        acc[key] = (a + sum_ref, b + sum_rec, c + sum_ref2, d + n)

    eps_int8 = []
    n_records = 0
    clamp_hits = [0]
    top_sum = [0.0]
    top_abs_sum = [0.0]
    groups_seen = [0]
    for layer, p0, T, nkv, hd, karr, varr in recs:
        if args.max_cells:
            karr, varr = karr[:args.max_cells], varr[:args.max_cells]
            T = args.max_cells
        n_records += 1
        for side, rot in (("K", karr), ("V", varr)):
            # the engine's fp16 rows in the channel domain: un-rotate (the round trip is exact)
            x = kq.fwht(rot)
            for h in range(nkv):
                row = x[:, h, :]
                # --- int8: the codec the engine actually applies to the unrotated rows
                q8, scale, clamped, top_shift, top_abs = int8_codec(row)
                amax_g = np.max(np.abs(row.reshape(-1, row.shape[-1] // GROUP, GROUP)), axis=2).reshape(-1)
                eps_int8.append(scale.reshape(-1) / (amax_g / 127.0) - 1.0)
                clamp_hits[0] += clamped
                top_sum[0] += top_shift
                top_abs_sum[0] += top_abs
                groups_seen[0] += scale.size
                # --- 4-bit codecs on the rotated rows (what the engine does)
                rot_h = rot[:, h, :]
                q4 = uniform_max(rot_h)
                for tag, rec in (("int8", q8), ("q4lm16", q4)):
                    key = (side, layer, h, tag)
                    add(key, T, float(np.sum(row)), float(np.sum(rec)), float(np.sum(row ** 2)))
                    # the per-channel drift of the *reconstructed* rows: for int8 the reconstruction
                    # lives in the channel space, for the 4-bit codec in the rotated space - so each
                    # is compared against its own reference there.
                    if tag == "int8":
                        d_ref, d_rec = row, rec
                    else:
                        d_ref, d_rec = rot_h, rec
                    m_ref = np.mean(d_ref, axis=0)
                    m_rec = np.mean(d_rec, axis=0)
                    add(key + ("drift",), 1, float(np.sum(m_rec - m_ref)),
                        0.0, float(np.sum((m_rec - m_ref) ** 2)))

    print()
    print(f"{'side':>4} {'layer':>5} {'codec':>8} {'mean shift per channel (signed avg)':>38} "
          f"{'RMS of the per-channel drift':>32}")
    print("-" * 96)
    summary = {}
    for (side, layer, h, tag) in sorted(k for k in acc if len(k) == 4):
        da, _, dc, _ = acc[(side, layer, h, tag, "drift")]
        # da: the sum over the records of the sum over the channels of (m_rec - m_ref)
        shift = da / max(n_records * 256, 1)
        var = dc / max(n_records * 256, 1)
        summary.setdefault((side, tag), []).append((shift, var))
    for (side, tag), vals in sorted(summary.items()):
        shifts = [v[0] for v in vals]
        vars_ = [v[1] for v in vals]
        print(f"{side:>4} {'all':>5} {tag:>8} {np.mean(shifts):38.3e} "
              f"{np.mean(np.sqrt(np.maximum(vars_, 0))):32.3e}")
    print()
    print("The drift column is in the units of the row's own standard deviation (the rows are")
    print("N(0,1)-ish), so it can be read against the data's own channel bias of 0.2673.")
    if eps_int8:
        e = np.concatenate([np.asarray(x).reshape(-1) for x in eps_int8])
        print()
        print(f"the fp16 scale's relative rounding: median |eps| = {np.median(np.abs(e)):.3e}, "
              f"99.9 % quantile = {np.quantile(np.abs(e), 0.999):.3e}")
        print()
        print("the clamp question:")
        print(f"  values the clamp actually caught: {clamp_hits[0]} of "
              f"{groups_seen[0] * GROUP} (the clamp can only bite if the fp16 scale rounds down by")
        print(f"  more than 0.5/127 = {0.5/127:.2e}; the fp16 rounding is at most {np.max(np.abs(e)):.2e})")
        print(f"  the extreme value's reconstruction minus the original: mean {top_sum[0]/max(groups_seen[0],1):+.3e}, "
              f"mean |.| {top_abs_sum[0]/max(groups_seen[0],1):.3e}  (in the row's own sd)")
        print("  -> the clamp is inert; what remains is the scale's own fp16 rounding, which shrinks")
        print("     the extreme by |eps| * amax whenever the scale rounds down.  That part is")
        print("     data-dependent (it scales with the crest factor) and one-sided per group.")


if __name__ == "__main__":
    main()
