#!/usr/bin/env python3
"""Bisection: does one half of the QSA layers carry the 16/16 tail?

P0b/P1 closed the codec side: the KV error at the tail positions is normal, the
tail is the model's sensitivity to the 16/16 error level.  The one lever left
that is not "more bits everywhere" is per-layer allocation (Dave): keep q4 on
some layers, store the rest as fp16.  The offline KV split cannot answer it (the
error share per layer is nearly uniform), so this tool scores two GPU runs:

  A  q4 on the late half (27,31,35,39,43,47), fp16 on the early half
  B  q4 on the early half (3,7,11,15,19,23), fp16 on the late half

Both against the same fp16 reference, int8 and P1's pure 16/16 arm, on the same
6,144 positions.  The gate was fixed before the runs: at P1's top-18 positions
of the paired gap (pure - int8),

    max(gap_A, gap_B) / (gap_A + gap_B) >= 0.65

means one half dominates and gets refined; otherwise B1 (5.0 bit) stays closed.

Measured 2026-11-08: A +0.086958, the early half +0.082001, ratio 0.515 -> B1
stays closed.  The first early-half run (B-early.tsv) was corrupted (mean
1.048770 against pure's 0.001817); the mask was re-run twice, bit-identically,
and those runs are the early arm.  --b names the file, so the excluded run
stays on disk without being scored.

    python tools/kv_bisect_layers.py --run DIR [--a FILE] [--b FILE] [--json OUT.json]

No GPU, no model: it reads the run TSVs.
"""
from __future__ import annotations

import argparse
import json
import pathlib
import sys

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
import quant_quality as qq  # noqa: E402

TOPK = 256
GATE = 0.65
P0A_TOP18 = [130972, 127887, 130973, 130952, 130942, 130968, 126191, 125074, 128793,
             130954, 128883, 124891, 127727, 128709, 128516, 125064, 128583, 130951]
P1_TOP18 = [130954, 130972, 130951, 124891, 130943, 127887, 128883, 130952, 130942, 126924,
            130973, 128583, 128547, 130945, 128793, 128516, 128533, 126867]


def one(path: str) -> dict:
    return next(iter(qq.read_run(path).values()))


def shape(kl: dict[int, float]) -> dict:
    v = sorted(kl.values())
    n = len(v)
    cut = max(1, int(round(0.01 * n)))
    return {"n": n, "mean": sum(v) / n, "median": v[n // 2], "p99": v[int(0.99 * (n - 1))],
            "max": max(v), "mean_no_top1": sum(v[:-cut]) / (n - cut)}


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--run", default=r"<strata-data>\bench\2026-11-08-bisect")
    ap.add_argument("--fp16", default=r"<strata-data>\bench\2026-11-03-tag-arm\128k\fp16.tsv")
    ap.add_argument("--int8", default=r"<strata-data>\bench\2026-11-03-tag-arm\128k\int8.tsv")
    ap.add_argument("--pure", default=r"<strata-data>\bench\2026-11-07-p1-pure16\p1-pure16.tsv")
    ap.add_argument("--a", help="default: RUN\\A-late.tsv")
    ap.add_argument("--b", help="default: RUN\\B-early.tsv")
    ap.add_argument("--json")
    a = ap.parse_args()

    run = pathlib.Path(a.run)
    arms = {"fp16": one(a.fp16), "int8": one(a.int8), "pure": one(a.pure),
            "A late": one(a.a or str(run / "A-late.tsv")),
            "B early": one(a.b or str(run / "B-early.tsv"))}
    pos = sorted(set.intersection(*(set(v) for v in arms.values())))
    print(f"positions: {len(pos)}")
    if len(pos) != 6144:
        print(f"warning: expected 6,144 positions, got {len(pos)}")

    ref = arms["fp16"]
    kl = {n: {p: qq.kl_between(ref[p][4], v[p][4], TOPK) for p in pos}
          for n, v in arms.items() if n != "fp16"}
    print(f"\nKL(fp16 || arm), fp16's top-{TOPK}:")
    print(f"  {'arm':7s} {'mean':>10s} {'median':>10s} {'p99':>9s} {'max':>8s} {'mean no top 1%':>15s}")
    for n, k in kl.items():
        s = shape(k)
        print(f"  {n:7s} {s['mean']:10.6f} {s['median']:10.6f} {s['p99']:9.6f} {s['max']:8.4f} "
              f"{s['mean_no_top1']:15.6f}")

    # The gap is the q4 cost against int8, at the positions where pure pays it.
    print("\npaired vs int8 (arm - int8), and how much of pure's gap the arm removes:")
    print(f"  {'arm':7s} {'mean':>11s} {'no top 1%':>11s} {'at P1 18':>10s} {'at P0a 18':>10s} {'removed':>8s}")
    gaps = {}
    for n, k in kl.items():
        d = {p: k[p] - kl["int8"][p] for p in pos}
        v = sorted(d.values())
        cut = max(1, int(round(0.01 * len(v))))
        g1 = sum(d[p] for p in P1_TOP18) / len(P1_TOP18)
        g0 = sum(d[p] for p in P0A_TOP18) / len(P0A_TOP18)
        removed = 1.0 - g1 / (sum(kl["pure"][p] - kl["int8"][p] for p in P1_TOP18) / len(P1_TOP18))
        gaps[n] = {"mean": sum(v) / len(v), "mean_no_top1": sum(v[:-cut]) / (len(v) - cut),
                   "at_p1_18": g1, "at_p0a_18": g0, "removed_at_p1_18": removed}
        print(f"  {n:7s} {gaps[n]['mean']:+11.6f} {gaps[n]['mean_no_top1']:+11.6f} "
              f"{g1:+10.6f} {g0:+10.6f} {removed:7.1%}")

    ga, gb = gaps["A late"]["at_p1_18"], gaps["B early"]["at_p1_18"]
    tot = ga + gb
    ratio = max(ga, gb) / tot if tot > 0 else float("nan")
    print(f"\ngate: A {ga:+.6f} + B {gb:+.6f} = {tot:+.6f}; "
          f"max/(A+B) = {ratio:.3f} (threshold {GATE})")
    if tot <= 0:
        verdict = "both halves are at or below int8 at these positions: no q4 cost to allocate, B1 closes"
    elif ratio >= GATE:
        verdict = f"{'A (late)' if ga > gb else 'B (early)'} carries {ratio:.0%} of the two halves' gap: refine it"
    else:
        verdict = f"neither half dominates ({ratio:.0%} < {GATE:.0%}): B1 closes"
    print(f"verdict: {verdict}")

    if a.json:
        pathlib.Path(a.json).write_text(json.dumps(
            {"positions": len(pos), "arms": {n: shape(k) for n, k in kl.items()},
             "gap_vs_int8": gaps,
             "gate": {"A": ga, "B": gb, "sum": tot, "ratio": ratio, "threshold": GATE,
                      "verdict": verdict}}, indent=1), encoding="utf-8")
        print(f"\nwrote {a.json}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
