#!/usr/bin/env python
"""Where does the per-channel bias come from: the data, or the transform?

The question (2026-11-11): the Hadamard is orthonormal and invertible, so a fixed per-channel
offset m maps to a fixed vector H*m in the transform space, and the inverse maps it back.  Could
the bias we measure in the transform space be an *artifact* of the transform's arithmetic instead
- a bias that the back-transformation cancels completely?

The data answer it directly, because the two halves of the claim are measured separately:

  * the S2 probe measures the K channel means BEFORE the rotation (prefill.cpp accumulates m.Kc
    into the probe before fwht256_kv_inplace_cuda runs), so it is the channel-space mean m;
  * the cell dump holds the rotated rows, so its per-coordinate mean r is the transform-space
    mean, and H*r is the channel-space mean *as the engine sees it*.

If H*r == m, the transform is faithful, the bias is a property of the data, and the offset table
the engine builds from the dump is right.  If H*r is (near) zero while m is not, the bias
vanishes at the back-transformation, the table is (near) zero, and the calibration is a no-op.

The script also measures the round trip H(H(y)) - y on the dumped cells: its mean component is
the systematic part of the transform's arithmetic, its RMS the noise.

Usage:
  python tools/kv_bias_origin.py --dump <dump.bin> --s2 <s2.txt> [--max-cells N]
"""
import argparse
import importlib.util
import math
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
spec = importlib.util.spec_from_file_location("kv_codec_mse", HERE / "kv_codec_mse.py")
kq = importlib.util.module_from_spec(spec)
spec.loader.exec_module(kq)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dump", required=True)
    ap.add_argument("--s2", required=True)
    ap.add_argument("--max-cells", type=int, default=None)
    args = ap.parse_args()

    s2_means = kq.read_s2_means(args.s2)          # channel space (before the rotation), head-merged
    recs = kq.read_dump(args.dump)
    if not recs:
        raise SystemExit("no records in the dump")
    print(f"dump: {len(recs)} records, layers {sorted(set(r[0] for r in recs))}")
    print(f"S2 probe: {len(s2_means)} layers with channel means")

    # the rotated mean per (layer, head, coordinate) and the round-trip error
    rot_sum, cnt = {}, {}
    rt_sum, rt_sq, rt_n = {}, {}, {}
    for layer, p0, T, nkv, hd, karr, varr in recs:
        if args.max_cells:
            karr, varr = karr[:args.max_cells], varr[:args.max_cells]
            T = args.max_cells
        for side, arr in (("K", karr), ("V", varr)):
            for h in range(nkv):
                key = (side, layer, h)
                s = rot_sum.setdefault(key, np.zeros(hd))
                s += arr[:, h, :].sum(axis=0)
                cnt[key] = cnt.get(key, 0) + T
                # the round trip on a slice of the cells: H(H(y)) - y
                take = arr[: min(T, 512), h, :]
                back = kq.fwht(kq.fwht(take))
                err = back - take
                a = rt_sum.setdefault(key, np.zeros(hd))
                a += err.sum(axis=0)
                b = rt_sq.setdefault(key, np.zeros(hd))
                b += (err ** 2).sum(axis=0)
                rt_n[key] = rt_n.get(key, 0) + err.shape[0]

    print()
    print(f"{'layer':>5} {'side':>4} {'|H*r|/sd (dump, un-rotated)':>30} {'|m|/sd (S2, channel space)':>30} "
          f"{'ratio':>7} {'round-trip bias/sd':>20} {'round-trip rms/sd':>19}")
    print("-" * 122)

    rows = []
    for (side, layer, h), s in sorted(rot_sum.items()):
        r = s / cnt[(side, layer, h)]          # the transform-space mean
        m_dump = kq.fwht(r)                    # the channel-space mean as the engine sees it
        if side != "K" or layer not in s2_means:
            continue
        m_s2 = s2_means[layer]                # head-merged, channel space
        # the dump's head-merged channel-space mean, to compare like with like
        other = rot_sum.get((side, layer, 1 - h))
        if other is None:
            continue
        m_merged = kq.fwht((s + other) / 2.0 / cnt[(side, layer, h)])
        sd = float(np.std(m_dump)) or 1.0
        a = float(np.median(np.abs(m_merged))) / sd
        b = float(np.median(np.abs(m_s2))) / sd
        bias = rt_sum[(side, layer, h)] / rt_n[(side, layer, h)]
        rms = np.sqrt(rt_sq[(side, layer, h)] / rt_n[(side, layer, h)])
        rows.append((layer, side, a, b, a / b if b else float("nan"),
                     float(np.median(np.abs(bias))) / sd, float(np.median(rms)) / sd))
        print(f"{layer:5d} {side:>4} {a:30.4f} {b:30.4f} {rows[-1][4]:7.3f} "
              f"{rows[-1][5]:20.2e} {rows[-1][6]:19.2e}")

    if rows:
        print()
        print(f"median ratio |H*r| / |m| over the layers: {np.median([r[4] for r in rows]):.3f}"
              f"   (1.0 = the transform reproduces the probe's bias exactly)")
        print(f"median round-trip bias / sd: {np.median([r[5] for r in rows]):.2e}"
              f"   (the systematic part of the transform's arithmetic)")
        print(f"median round-trip rms / sd: {np.median([r[6] for r in rows]):.2e}"
              f"   (the noise part; the engine's values live in fp32)")
        print()
        print("Reading: the ratio near 1.0 means the bias in the transform space is the image of a")
        print("real bias in the channel space, and the offset table the engine builds from the dump")
        print("is that bias, not an artifact of the transform.")


if __name__ == "__main__":
    main()
