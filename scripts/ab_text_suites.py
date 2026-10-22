#!/usr/bin/env python3
"""ab_text_suites.py - compare two runs of jevport-suite.py text (arm A/B) on the same items.

  ab_text_suites.py suite-B-20261022 suite-C-20261022 [--fit-n 60 --test-n 100]

Reads each outdir's text-suite.json for the headline numbers and text-item-log.jsonl for a
PAIRED comparison: the item logs are emitted in a fixed order (per qtype: fit_n dev rows then
test_n test rows), so test rows are sliced positionally and joined between runs.  Reports the
discordant counts and exact McNemar per qtype, so a one-item difference cannot read as a win.
"""
import argparse
import json
import math
from pathlib import Path

QTYPES = ("noul", "choice", "score")


def mcnemar(a_ok, b_ok):
    n10 = sum(1 for x, y in zip(a_ok, b_ok) if x and not y)
    n01 = sum(1 for x, y in zip(a_ok, b_ok) if y and not x)
    m = n10 + n01
    if m == 0:
        return n10, n01, 1.0
    k = min(n10, n01)
    return n10, n01, min(1.0, 2 * sum(math.comb(m, i) for i in range(k + 1)) / (2 ** m))


def blocks(log_path, fit_n, test_n):
    rows = [json.loads(l) for l in open(log_path) if l.strip()]
    per = fit_n + test_n
    out = {}
    for i, qt in enumerate(QTYPES):
        chunk = rows[i * per:(i + 1) * per]
        if len(chunk) < per:
            continue
        out[qt] = chunk[fit_n:]
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("a")
    ap.add_argument("b")
    ap.add_argument("--fit-n", type=int, default=60)
    ap.add_argument("--test-n", type=int, default=100)
    ap.add_argument("--labels", default="A,B")
    args = ap.parse_args()
    la, lb = args.labels.split(",")

    ra = json.load(open(Path(args.a) / "text-suite.json"))
    rb = json.load(open(Path(args.b) / "text-suite.json"))

    print(f"{'qtype':<8}{'dataset':<10}{la + ' raw':>9}{la + ' cal':>9}"
          f"{lb + ' raw':>10}{lb + ' cal':>9}   {'ECE ' + la:>9}{'ECE ' + lb:>8}")
    print("-" * 82)
    for qt in QTYPES:
        a, b = ra.get(qt), rb.get(qt)
        if not a or not b:
            continue
        print(f"{qt:<8}{a['dataset']:<10}{a['raw']['acc']:>9.3f}{a['cal']['acc']:>9.3f}"
              f"{b['raw']['acc']:>10.3f}{b['cal']['acc']:>9.3f}   "
              f"{a['cal']['ece15']:>9.4f}{b['cal']['ece15']:>8.4f}")

    ba = blocks(Path(args.a) / "text-item-log.jsonl", args.fit_n, args.test_n)
    bb = blocks(Path(args.b) / "text-item-log.jsonl", args.fit_n, args.test_n)
    print("\npaired on identical test items (McNemar, raw argmax):")
    for qt in QTYPES:
        if qt not in ba or qt not in bb:
            continue
        x, y = ba[qt], bb[qt]
        n = min(len(x), len(y))
        a_ok = [r["argmax"] == r["gold"] for r in x[:n]]
        b_ok = [r["argmax"] == r["gold"] for r in y[:n]]
        n10, n01, p = mcnemar(a_ok, b_ok)
        same_pr = sum(1 for i in range(n) if x[i]["pr"] == y[i]["pr"])
        print(f"  {qt:<8} n={n} acc {la}={sum(a_ok)/n:.3f} {lb}={sum(b_ok)/n:.3f}  "
              f"{la} only {n10}, {lb} only {n01}, p={p:.3f}  identical pr rows {same_pr}/{n}")


if __name__ == "__main__":
    main()
