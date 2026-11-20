#!/usr/bin/env python
"""Where does the centering's block gain sit: which layer, which side?

C4 measured, end to end, that the centering does not improve the 16/16 arm
(q4lm32w: KL 0.01689 -> 0.01768, the paired difference -0.00079 nats, not significant),
while the block harness (C1) had reported that the same centering is worth 1.91 dB of
block MSE on the same cells.  Both cannot be right for the same codec.

The harness averages K and V, and end to end the value side enters the attention output
directly while the key side enters only through the softmax weights (second order).  So this
script splits the block measurement by layer and by side, with the same quantizer code as
tools/kv_codec_mse.py (imported, not copied).

Usage:
  python tools/kv_center_attrib.py --dump <dump.bin> --s2 <s2.txt> [--max-cells N] [--layers N]
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


def block_mse(arr, kind, levels, fit, nblk=8):
    """The mean squared error over the blocks of one side, in the same way as the harness."""
    tot, n = 0.0, 0
    bl = kq.blocks(arr, nblk)
    for i in range(bl.shape[0]):
        x = bl[i]
        if kind == "uniform":
            recon, _d = kq.quantize_uniform(x, fit)
        else:
            recon, _d = kq.quantize_lloyd(x, levels, fit)
        tot += float(np.mean((recon - x) ** 2))
        n += 1
    return tot / max(n, 1)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dump", required=True)
    ap.add_argument("--s2", required=True)
    ap.add_argument("--layers", type=int, default=None)
    ap.add_argument("--max-cells", type=int, default=None)
    args = ap.parse_args()

    recs = kq.read_dump(args.dump, want_layers=set(range(args.layers)) if args.layers else None)
    if not recs:
        raise SystemExit("no records in the dump")
    print(f"dump: {len(recs)} records, layers {sorted(set(r[0] for r in recs))}")

    # the offsets from the dump itself: per layer, per KV head, per channel, per side
    offs, cnts = {}, {}
    for layer, p0, T, nkv, hd, karr, varr in recs:
        if args.max_cells:
            karr, varr = karr[:args.max_cells], varr[:args.max_cells]
            T = args.max_cells
        for tag, arr in (("K", karr), ("V", varr)):
            for h in range(arr.shape[1]):
                s = offs.setdefault((tag, layer, h), np.zeros(hd))
                s += arr[:, h, :].sum(axis=0)
                cnts[(tag, layer, h)] = cnts.get((tag, layer, h), 0) + T
    for key in offs:
        offs[key] = offs[key] / max(cnts[key], 1)

    # the codecs: the shipped 16/16 (lloyd16 on both sides) and the q4lm8v probe (16 K, 8 V),
    # both with the max scale (the shipped one)
    codecs = {"lloyd16 (16/16)": (kq.L16, kq.L16), "lloyd16K+lloyd8V (q4lm8v)": (kq.L16, kq.L8)}

    by_layer = {}
    for layer, p0, T, nkv, hd, karr, varr in recs:
        if args.max_cells:
            karr, varr = karr[:args.max_cells], varr[:args.max_cells]
            T = args.max_cells
        for side, arr in (("K", karr), ("V", varr)):
            for name, (lk, lv) in codecs.items():
                lev = lk if side == "K" else lv
                off = np.stack([offs[(side, layer, h)] for h in range(nkv)])  # rotated domain
                plain = block_mse(arr, "lloyd", lev, False)
                cent = block_mse(arr - off[None, :, :], "lloyd", lev, False)
                key = (layer, side, name)
                a, b, n = by_layer.get(key, (0.0, 0.0, 0))
                by_layer[key] = (a + plain * T, b + cent * T, n + T)

    print()
    print(f"{'layer':>5} {'side':>4} {'codec':>26} {'MSE':>12} {'MSE centered':>14} {'gain':>8}")
    print("-" * 74)
    for (layer, side, name), (a, b, n) in sorted(by_layer.items()):
        plain, cent = a / n, b / n
        db = 10.0 * math.log10(cent / plain) if plain > 0 else float("nan")
        print(f"{layer:5d} {side:>4} {name:>26} {plain:12.4e} {cent:14.4e} {db:+7.2f} dB")

    print()
    print("summarized per side (all layers):")
    for name in codecs:
        for side in ("K", "V"):
            a = sum(v[0] for k, v in by_layer.items() if k[1] == side and k[2] == name)
            b = sum(v[1] for k, v in by_layer.items() if k[1] == side and k[2] == name)
            n = sum(v[2] for k, v in by_layer.items() if k[1] == side and k[2] == name)
            plain, cent = a / n, b / n
            db = 10.0 * math.log10(cent / plain) if plain > 0 else float("nan")
            print(f"  {name:>26} {side}: {plain:.4e} -> {cent:.4e}  {db:+6.2f} dB")
    print()
    print("the engine's table is the same kind of offset (per layer, per KV head, per channel, K and V")
    print("separate); the harness's --center mode above measures the same thing on the same cells.")


if __name__ == "__main__":
    main()
