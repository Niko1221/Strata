#!/usr/bin/env python3
"""Summary of ab/perf-<label>.jsonl: per size, prefill tok/s and ms per forward, rep 0 (the warm-up) dropped.

usage: python3 ab/win_summary.py [--drop 1] LABEL [LABEL ...]
Prints one line per (label, size) with the median/min/max of the kept reps, then a side-by-side ratio table
against the first label given.
"""
import json
import statistics
import sys


def rows(path):
    out = []
    for line in open(path):
        line = line.strip()
        if not line:
            continue
        r = json.loads(line)
        if "error" in r or not r.get("prompt_tps"):
            continue
        out.append(r)
    return out


def main() -> int:
    a = sys.argv[1:]
    drop = 1
    if a and a[0] == "--drop":
        drop = int(a[1]); a = a[2:]
    table = {}
    for label in a:
        for r in rows("ab/perf-%s.jsonl" % label):
            if r["rep"] < drop:
                continue
            table.setdefault((label, r["target"]), []).append(r)
    print("%-10s %8s %5s %10s %10s %8s %8s" % ("label", "target", "n", "prefill", "ms/fwd", "tok/s", "acc%"))
    for (label, target), rs in sorted(table.items()):
        pp = statistics.median([x["prompt_tps"] for x in rs])
        mf = statistics.median([x["ms_per_fwd"] for x in rs if x.get("ms_per_fwd")])
        ts = statistics.median([x["tok_s"] for x in rs if x.get("tok_s")])
        acc = statistics.median([x["acc_pct"] for x in rs if x.get("acc_pct")])
        print("%-10s %8d %5d %10.0f %10.2f %8.1f %8.1f" % (label, target, len(rs), pp, mf, ts, acc or 0))
    if len(a) >= 2:
        ref = a[0]
        print("\nratios vs %s (prefill / ms-per-forward; >1.03 better, <0.97 worse):" % ref)
        for label in a[1:]:
            for target in sorted({t for (l, t) in table if l == ref}):
                if (label, target) not in table:
                    continue
                rp = statistics.median([x["prompt_tps"] for x in table[(ref, target)]])
                rn = statistics.median([x["prompt_tps"] for x in table[(label, target)]])
                mp = statistics.median([x["ms_per_fwd"] for x in table[(ref, target)] if x.get("ms_per_fwd")])
                mn = statistics.median([x["ms_per_fwd"] for x in table[(label, target)] if x.get("ms_per_fwd")])
                print("  %-8s %7d  prefill %.3f  decode %.3f" % (label, target, rn / rp, mp / mn))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
