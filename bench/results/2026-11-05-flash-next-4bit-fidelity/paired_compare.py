#!/usr/bin/env python3
"""Paired comparison of two yardstick runs: McNemar on top-1 agreement + paired perplexity delta.

Both runs scored the same 4,995 positions, so the right question is not 'what are the two rates' but
'at the positions where they disagree, is one of them wrong more often than the other'.
"""
import json
import math
import sys

Q = "/mnt/cache/strata/quality-run"
TS = "20261105-2014"


def load(path):
    rows = {}
    with open(path) as f:
        idx = 0
        for line in f:
            p = line.rstrip("\n").split("\t")
            if len(p) < 6:
                continue
            # key by ROW ORDER, not by pos: every segment numbers its own positions 0..998, so pos alone
            # collapses 4,995 rows onto 999 keys. All runs score the same positions in the same order.
            agree = p[5].strip() in ("1", "true", "True")
            rows[idx] = (agree, float(p[2]))          # p[2] = logp of the true token
            idx += 1
    return rows


def mcnemar(a, b):
    """Exact two-sided binomial on the discordant pairs."""
    only_a = sum(1 for k in a if a[k][0] and not b[k][0])   # A right, B wrong
    only_b = sum(1 for k in a if b[k][0] and not a[k][0])   # B right, A wrong
    n = only_a + only_b
    if n == 0:
        return only_a, only_b, 1.0
    k = min(only_a, only_b)
    p = 2 * sum(math.comb(n, i) * 0.5 ** n for i in range(0, k + 1))
    return only_a, only_b, min(1.0, p)


def seg_map():
    """Line ranges per segment from the .seg.json the tool writes."""
    m = {}
    for arm in ("ref", "iq4xs", "q4kxl"):
        try:
            d = json.load(open(f"{Q}/{arm}-{TS}.tsv.seg.json"))
        except FileNotFoundError:
            continue
        segs = d.get("segments") or d
        out = {}
        if isinstance(segs, dict):
            for name, v in segs.items():
                if isinstance(v, dict) and "pos" in v:
                    out[name] = set(v["pos"]) if isinstance(v["pos"], list) else None
                elif isinstance(v, (list, tuple)) and len(v) == 2:
                    out[name] = set(range(int(v[0]), int(v[1]) + 1))
        m[arm] = out
    return m


def main():
    arms = {a: load(f"{Q}/{a}-{TS}.tsv") for a in ("ref", "iq4xs", "q4kxl")}
    for a, r in arms.items():
        print(f"{a:>6}: {len(r)} positions, {sum(1 for v in r.values() if v[0])} correct "
              f"({100*sum(1 for v in r.values() if v[0])/len(r):.2f}%)")

    SEGS = ["code", "docs", "english", "multilingual", "agent"]
    label_of = {i: SEGS[min(i // 999, len(SEGS) - 1)] for i in range(4995)}

    for x, y in (("iq4xs", "q4kxl"), ("iq4xs", "ref"), ("q4kxl", "ref")):
        a, b = arms[x], arms[y]
        common = set(a) & set(b)
        aa = {k: a[k] for k in common}
        bb = {k: b[k] for k in common}
        only_x, only_y, p = mcnemar(aa, bb)
        lp_a = sum(v[1] for v in aa.values()) / len(aa)
        lp_b = sum(v[1] for v in bb.values()) / len(bb)
        print(f"\n{x} vs {y}: {only_x} positions only {x} gets right, {only_y} only {y} "
              f"-> McNemar exact p = {p:.4g}")
        print(f"  mean logp(true token): {x} {lp_a:.4f} vs {y} {lp_b:.4f}  "
              f"(ppl {math.exp(-lp_a):.3f} vs {math.exp(-lp_b):.3f}, "
              f"{(math.exp(-lp_a)/math.exp(-lp_b)-1)*100:+.2f}%)")
        if label_of:
            agg = {}
            for k in common:
                s = label_of.get(k, "?")
                d = agg.setdefault(s, [0, 0, 0.0, 0.0])
                if aa[k][0] and not bb[k][0]:
                    d[0] += 1
                if bb[k][0] and not aa[k][0]:
                    d[1] += 1
                d[2] += aa[k][1]
                d[3] += bb[k][1]
                d.append(0)
                d[4] += 1
            print(f"  per segment: {'seg':<13}{x+'-only':>10}{y+'-only':>10}{'p':>8}   logp {x} / {y}")
            for s, d in agg.items():
                cnt = d[4]
                n = d[0] + d[1]
                if n:
                    kk = min(d[0], d[1])
                    pp = min(1.0, 2 * sum(math.comb(n, i) * 0.5 ** n for i in range(kk + 1)))
                else:
                    pp = 1.0
                print(f"    {s:<13}{d[0]:>10}{d[1]:>10}{pp:>8.3f}   {d[2]/cnt:.3f} / {d[3]/cnt:.3f}")


if __name__ == "__main__":
    sys.exit(main())
