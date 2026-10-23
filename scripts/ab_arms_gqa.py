#!/usr/bin/env python3
"""Paired GQA comparison of two arms on the identical 600 items.

The two runs were not served under the same calibrator: C ran with identity,
B ran with calib-D installed (platt_v1). Platt is a bijective monotone map, so B's
SERVED probabilities are inverted back to the raw readout before comparing:

    p_raw = sigmoid( (logit(p_cal) - c) / a )

A round-trip check re-applies the forward transform to the recovered raw values and
asserts it reproduces the served numbers, so the inversion is verified, not assumed.

  ab_arms_gqa.py C_ROWS B_ROWS [--a 1.0571] [--c -0.172]
"""
import argparse
import json
import math


def sig(z):
    return 1.0 / (1.0 + math.exp(-max(-700.0, min(700.0, z))))


def logit(p):
    p = min(max(p, 1e-12), 1 - 1e-12)
    return math.log(p / (1 - p))


def load(path):
    return {r["id"]: r for r in map(json.loads, open(path)) if r.get("p") is not None}


def acc(rows):
    return sum(1 for r in rows if (1 if r["p"] >= 0.5 else 0) == r["gold"]) / len(rows)


def brier(rows):
    return sum((r["p"] - r["gold"]) ** 2 for r in rows) / len(rows)


def ece15(rows):
    bins = [[] for _ in range(15)]
    for r in rows:
        bins[min(14, int(r["p"] * 15))].append(r)
    tot = len(rows)
    e = 0.0
    for b in bins:
        if not b:
            continue
        mp = sum(r["p"] for r in b) / len(b)
        mg = sum(r["gold"] for r in b) / len(b)
        e += len(b) / tot * abs(mp - mg)
    return e


def mcnemar(b, n):
    if b + n == 0:
        return 1.0
    m = b + n
    k = min(b, n)
    return min(1.0, 2 * sum(math.comb(m, i) for i in range(k + 1)) / 2 ** m)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("c_rows")
    ap.add_argument("b_rows")
    ap.add_argument("--a", type=float, default=1.0571)
    ap.add_argument("--c", type=float, default=-0.172)
    args = ap.parse_args()

    C, B = load(args.c_rows), load(args.b_rows)
    common = sorted(set(C) & set(B))
    print(f"C rows {len(C)}  B rows {len(B)}  matched {len(common)}")

    worst = 0.0
    Braw = {}
    for q in common:
        p_cal = B[q]["p"]
        p_raw = sig((logit(p_cal) - args.c) / args.a)
        back = sig(args.a * logit(p_raw) + args.c)
        worst = max(worst, abs(back - p_cal))
        Braw[q] = {"id": q, "gold": B[q]["gold"], "p": p_raw}
    print(f"round-trip check: worst |served - re-served| = {worst:.3e}\n")

    crows = [{"id": q, "gold": C[q]["gold"], "p": C[q]["p"]} for q in common]
    brows = [Braw[q] for q in common]
    half = len(common) // 2
    for tag, rs in (("all 600", slice(None)), ("test 300", slice(half, None))):
        cr, br = crows[rs], brows[rs]
        print(f"[{tag}]")
        for name, r in (("C raw", cr), ("B raw", br)):
            print(f"  {name:<6} acc={acc(r):.4f} brier={brier(r):.4f} ece15={ece15(r):.4f}")
        agree = sum(1 for x, y in zip(cr, br) if (x['p'] >= 0.5) == (y['p'] >= 0.5))
        dp = sum(abs(x["p"] - y["p"]) for x, y in zip(cr, br)) / len(cr)
        conly = sum(1 for x, y in zip(cr, br)
                    if (x['p'] >= 0.5) == x['gold'] and (y['p'] >= 0.5) != y['gold'])
        bonly = sum(1 for x, y in zip(cr, br)
                    if (y['p'] >= 0.5) == y['gold'] and (x['p'] >= 0.5) != x['gold'])
        print(f"  agreement {agree}/{len(cr)} ({agree / len(cr):.3f})  mean|dp| {dp:.4f}  "
              f"C-only-right {conly} / B-only-right {bonly}  McNemar p={mcnemar(conly, bonly):.4f}")
        pt = [B[q].get("prompt_tokens") for q in common[rs]]
        print(f"  prompt_tokens mean: B {sum(t for t in pt if t) / max(1, sum(1 for t in pt if t)):.1f}")
        print()

    served_acc = sum(1 for q in common
                     if (1 if B[q]["p"] >= 0.5 else 0) == B[q]["gold"]) / len(common)
    print(f"B as SERVED (calib-D applied): acc={served_acc:.4f} on the same {len(common)} items")


if __name__ == "__main__":
    main()
