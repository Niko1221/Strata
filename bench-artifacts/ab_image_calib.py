#!/usr/bin/env python3
"""Paired A/B of two image-suite item logs on the identical TEST items.

Answers: does the SHIPPED calibrator (calib-D, live) change POPE / CV-Bench /
CountBench decisions or confidence vs serving identity?

Everything is computed from the *served* `pr` vector in each item log, so a
calibrator installed on the box is measured directly rather than re-applied
offline. This is deliberately NOT the suite's own raw/cal columns: those apply a
dev-fit on top of already-calibrated probabilities and compound the transform.

  ab_image_calib.py BASE_LOG NEW_LOG [--old identity] [--new calib-D]
                    [--data /opt/data/llm-swap/strata-data/eval/real/data]
"""
import argparse
import json
import math
from pathlib import Path

KINDS = {"pope": "noul-pope", "cvb": "choice-cvb", "cb": "score-countbench"}


def load_log(path):
    rows = {}
    for line in open(path):
        line = line.strip()
        if not line:
            continue
        r = json.loads(line)
        rows[r["qid"]] = r
    return rows


def test_qids(datadir, kind):
    out = set()
    for f in sorted(Path(datadir, KINDS[kind], "test").glob("*.json")):
        j = json.loads(f.read_text())
        out.add(next(iter(j["questions"].items()))[0])
    return out


def stats(rows, gold):
    """rows: list of (pr, gold) -> acc, brier, nll, ece15 on the top-label protocol."""
    acc = 0
    brier = nll = 0.0
    ps, gs = [], []
    for pr, g in rows:
        amax = max(range(len(pr)), key=lambda i: pr[i])
        acc += int(amax == g)
        brier += sum((p - (1.0 if i == g else 0.0)) ** 2 for i, p in enumerate(pr))
        nll -= math.log(max(pr[g], 1e-12))
        ps.append(max(pr[amax], 1e-9))
        gs.append(1 if amax == g else 0)
    n = len(rows)
    bins = [[] for _ in range(15)]
    for p, g in zip(ps, gs):
        bins[min(14, int(p * 15))].append((p, g))
    ece = sum(len(b) / n * abs(sum(p for p, _ in b) / len(b) - sum(g for _, g in b) / len(b))
              for b in bins if b)
    return {"n": n, "acc": acc / n, "brier": brier / n, "nll": nll / n, "ece15": ece}


def mcnemar(b, n):
    """exact two-sided binomial on the discordant pairs"""
    if b + n == 0:
        return 1.0
    k, m = min(b, n), b + n
    tail = sum(math.comb(m, i) for i in range(0, k + 1)) / 2 ** m
    return min(1.0, 2 * tail)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("base")
    ap.add_argument("new")
    ap.add_argument("--old", default="identity")
    ap.add_argument("--new-label", default="calib-D")
    ap.add_argument("--data", default="/opt/data/llm-swap/strata-data/eval/real/data")
    args = ap.parse_args()

    A, B = load_log(args.base), load_log(args.new)
    print(f"{args.old}: {len(A)} rows   {args.new_label}: {len(B)} rows\n")
    served = sorted({r.get("calib") for r in B.values()})
    print(f"served calib in the {args.new_label} run: {served}\n")

    for kind in KINDS:
        tq = test_qids(args.data, kind)
        common = sorted(tq & set(A) & set(B))
        if not common:
            print(f"[{kind}] no common test items\n")
            continue
        ra = [(A[q]["pr"], A[q]["gold"]) for q in common]
        rb = [(B[q]["pr"], B[q]["gold"]) for q in common]
        sa, sb = stats(ra, None), stats(rb, None)
        flips = 0
        only_a = only_b = 0
        for (pra, g), (prb, _) in zip(ra, rb):
            ma = max(range(len(pra)), key=lambda i: pra[i])
            mb = max(range(len(prb)), key=lambda i: prb[i])
            if ma != mb:
                flips += 1
                if ma == g:
                    only_a += 1
                if mb == g:
                    only_b += 1
        print(f"[{kind}] test items matched: {len(common)}")
        for tag, s in ((args.old, sa), (args.new_label, sb)):
            print(f"  {tag:<10} acc={s['acc']:.3f} brier={s['brier']:.4f} "
                  f"nll={s['nll']:.4f} ece15={s['ece15']:.4f}")
        print(f"  argmax flips: {flips}/{len(common)}  "
              f"({args.old}-only-right {only_a} / {args.new_label}-only-right {only_b}  "
              f"p={mcnemar(only_a, only_b):.4f})")
        print(f"  delta: acc {sb['acc'] - sa['acc']:+.3f}  brier {sb['brier'] - sa['brier']:+.4f}  "
              f"ece15 {sb['ece15'] - sa['ece15']:+.4f}\n")


if __name__ == "__main__":
    main()
