#!/usr/bin/env python3
"""Every table of this report from data/positions.tsv.gz (make_positions.py; standard library only):

    python analyze.py [--data data/positions.tsv.gz] [--boot 2000] [--block 32]

KL is tools/quant_quality.py compare's: KL(Q8_0 || run) over Q8_0's 64 most likely tokens.  Paired differences: per
position, run B's KL minus run A's (both against Q8_0), with two 95% ranges from bootstrap resamples (seed 1):
of single positions, and of blocks of --block consecutive positions within each text (neighbouring tokens are not
independent, so the block range is the one to read).
"""
from __future__ import annotations

import argparse
import gzip
import math
import pathlib
import random

RUNS = {
    "iq3s_cpu": "IQ3_S, experts on the CPU (the published yardstick method)",
    "iq3s_cpu_iq4nl": "IQ3_S, experts on the CPU, the GGUF's IQ4_NL n-gram table instead of FP8",
    "skip0": "IQ3_S, GPU experts (auto), tail skip off, FP16 K/V",
    "skip7": "IQ3_S, GPU experts (auto), tail skip on (=7), FP16 K/V",
    "skip0_int8": "IQ3_S, GPU experts (auto), tail skip off, int8 K/V (the live settings)",
}
PAIRS = [("skip0", "skip7", "tail skip on vs off"),
         ("iq3s_cpu", "iq3s_cpu_iq4nl", "IQ4_NL vs FP8 n-gram table"),
         ("skip0", "skip0_int8", "int8 vs FP16 K/V"),
         ("iq3s_cpu", "skip0", "GPU vs CPU experts (placement)")]
LIVE = "skip0_int8"


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default=str(pathlib.Path(__file__).resolve().parent / "data" / "positions.tsv.gz"))
    ap.add_argument("--boot", type=int, default=2000)
    ap.add_argument("--block", type=int, default=32)
    a = ap.parse_args()
    lines = gzip.open(a.data, "rt").read().splitlines()
    head = lines[0].split("\t")
    rows = [dict(zip(head, l.split("\t"))) for l in lines[1:]]
    segs = list(dict.fromkeys(r["text"] for r in rows))
    for r in rows:
        r["target"], r["q8_top"], r["q8_logp"] = int(r["target"]), int(r["q8_top"]), float(r["q8_logp"])
        for k in RUNS:
            r[f"{k}_top"], r[f"{k}_logp"], r[f"{k}_kl"] = int(r[f"{k}_top"]), float(r[f"{k}_logp"]), float(r[f"{k}_kl"])

    def stats(k, sel):
        n = len(sel)
        v = sorted(r[f"{k}_kl"] for r in sel)
        return {"top1": sum(r["q8_top"] == r[f"{k}_top"] for r in sel) / n, "kl": sum(v) / n,
                "p99": v[int(0.99 * (n - 1))], "kl_gt1": sum(x > 1 for x in v),
                "ppl": math.exp(-sum(r[f"{k}_logp"] for r in sel) / n), "ppl_ref": math.exp(-sum(r["q8_logp"] for r in sel) / n),
                "hit": sum(r[f"{k}_top"] == r["target"] for r in sel) / n, "hit_ref": sum(r["q8_top"] == r["target"] for r in sel) / n}

    def pick(seg):
        return rows if seg == "all" else [r for r in rows if r["text"] == seg]

    print(f"positions: {len(rows)} ({', '.join(segs)})\n")
    print("== Every run against Q8_0, all texts")
    print(f"{'run':16s} {'same top':>8s} {'KL':>7s} {'p99':>5s} {'KL>1':>5s} {'PPL':>6s} {'Q8 PPL':>6s}")
    for k, what in RUNS.items():
        t = stats(k, rows)
        print(f"{k:16s} {100 * t['top1']:7.2f}% {t['kl']:7.4f} {t['p99']:5.2f} {t['kl_gt1']:5d} {t['ppl']:6.3f} "
              f"{t['ppl_ref']:6.3f}   {what}")

    print(f"\n== The live settings ({LIVE}) against Q8_0, per text")
    print(f"{'text':13s} {'same top':>8s} {'KL':>6s} {'KL>1':>5s} {'real next token: run / Q8':>27s} {'PPL: run / Q8':>18s}")
    for seg in segs + ["all"]:
        t = stats(LIVE, pick(seg))
        print(f"{seg:13s} {100 * t['top1']:7.1f}% {t['kl']:6.3f} {t['kl_gt1']:5d} {100 * t['hit']:16.1f}% / "
              f"{100 * t['hit_ref']:.1f}% {t['ppl']:9.3f} / {t['ppl_ref']:.3f} ({100 * (t['ppl'] / t['ppl_ref'] - 1):+.1f}%)")

    print(f"\n== Where {LIVE} and Q8_0 pick different top tokens: which one is the text's real next token")
    print(f"{'text':13s} {'differ':>6s} {'run right':>9s} {'Q8 right':>8s} {'neither':>7s}")
    for seg in segs + ["all"]:
        rr = qr = nn = 0
        for r in pick(seg):
            t, q8, run = r["target"], r["q8_top"], r[f"{LIVE}_top"]
            if q8 == run:
                continue
            rr += run == t
            qr += q8 == t
            nn += run != t and q8 != t
        print(f"{seg:13s} {rr + qr + nn:6d} {rr:9d} {qr:8d} {nn:7d}")

    print(f"\n== Paired: run B's KL minus run A's, per position (both against Q8_0); 95% ranges of single positions "
          f"and of blocks of {a.block}")
    random.seed(1)
    lo_i, hi_i = int(0.025 * a.boot), int(0.975 * a.boot) - 1
    for x, y, what in PAIRS:
        print(f"{what} ({y} - {x})")
        for seg in segs + ["all"]:
            blocks = []
            for s in (segs if seg == "all" else [seg]):
                v = [r[f"{y}_kl"] - r[f"{x}_kl"] for r in rows if r["text"] == s]
                blocks += [v[i:i + a.block] for i in range(0, len(v), a.block)]
            dd = [v for b in blocks for v in b]
            n = len(dd)
            single = sorted(sum(dd[random.randrange(n)] for _ in range(n)) / n for _ in range(a.boot))
            blk = []
            for _ in range(a.boot):
                ch = [blocks[random.randrange(len(blocks))] for _ in blocks]
                blk.append(sum(sum(b) for b in ch) / sum(len(b) for b in ch))
            blk.sort()
            worse, better = sum(v > 0.01 for v in dd), sum(v < -0.01 for v in dd)
            print(f"  {seg:13s} {sum(dd) / n:+.4f}  positions [{single[lo_i]:+.4f}, {single[hi_i]:+.4f}]  "
                  f"blocks [{blk[lo_i]:+.4f}, {blk[hi_i]:+.4f}]  worse/better by >0.01: {worse}/{better}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
