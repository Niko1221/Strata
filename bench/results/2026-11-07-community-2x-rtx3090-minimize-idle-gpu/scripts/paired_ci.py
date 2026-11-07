#!/usr/bin/env python3
"""Paired medians with a bootstrapped confidence interval - the error bar a median alone does not carry.

Run: python3 paired_ci.py <file.jsonl> [<more.jsonl> ...] --base 'pw=0' [--arm PATTERN] [--kind code]
                                                                   [--boot 4000 --seed 7]

The geometry of the measurement (since 05.11.): in one round every arm gets **the same document**, and the
order rotates. A comparison of two arms is therefore a paired sample comparison over the rounds, and the
unit of uncertainty is NOT the scatter of tok/s but the scatter of the **ratio** of the same document under
two arms. That is also why a single sweep of three rounds says little: the acceptance luck of 500 greedy
tokens moves one arm by ±10 %.

`--base` is the reference arm (usually `pw=0`); `--arm` selects the arms to compare against it (regex,
default: everything else). Several files are pooled, but **pairing stays inside one file** (otherwise
arms of different servers get compared). Output per arm and kind: n, median of the ratio in percent, the
2.5/97.5 percentiles of the bootstrapped median, and the share of rounds the arm led. `raw/*-paare.txt` in
the folder above are this tool's output, and every paired figure in the README was read off them.
"""
import argparse
import json
import random
import re
import statistics
from collections import defaultdict


def load(paths):
    rows = []
    for i, p in enumerate(paths):
        for line in open(p):
            if not line.strip():
                continue
            try:
                r = json.loads(line)
            except json.JSONDecodeError:
                continue
            r["_file"] = i                       # pairing stays inside one run
            rows.append(r)
    return [r for r in rows if r.get("gen") and not r.get("contaminated")]


def boot_ci(xs, n=4000, seed=7):
    rnd = random.Random(seed)
    meds = []
    for _ in range(n):
        sample = [rnd.choice(xs) for _ in xs]
        meds.append(statistics.median(sample))
    meds.sort()
    return meds[int(0.025 * n)], meds[int(0.975 * n)]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("files", nargs="+")
    ap.add_argument("--base", default="pw=0")
    ap.add_argument("--arm", default=".", help="regex of the arms to compare (default: all but the base)")
    ap.add_argument("--kind", default="")
    ap.add_argument("--boot", type=int, default=4000)
    ap.add_argument("--seed", type=int, default=7)
    a = ap.parse_args()

    rows = load(a.files)
    base = defaultdict(dict)
    for r in rows:
        if r["arm"] == a.base:
            # the group (one server start) BELONGS in the pairing key: without it a sweep that spans
            # groups would compare an arm against the reference arm of a different server
            base[(r.get("kind"), r.get("_file"), r.get("group", ""), r.get("run"), r.get("doc"))] = r
    rel = defaultdict(list)
    for r in rows:
        if r["arm"] == a.base or not re.search(a.arm, r["arm"]):
            continue
        b = base.get((r.get("kind"), r.get("_file"), r.get("group", ""), r.get("run"), r.get("doc")))
        if b:
            rel[(r["arm"], r.get("kind"))].append(100.0 * (r["tok_s"] / b["tok_s"] - 1))
    print(f"reference arm {a.base!r}; paired on document+round+run; bootstrap {a.boot} x median")
    print(f"{'arm':<36} {'kind':<6} {'n':>3} {'median':>7} {'95 % CI':>18} {'ahead':>7}")
    for (arm, kind), xs in sorted(rel.items(), key=lambda kv: (kv[0][1], -statistics.median(kv[1]))):
        if a.kind and kind != a.kind:
            continue
        lo, hi = boot_ci(xs, a.boot, a.seed)
        print(f"{arm:<36} {kind:<6} {len(xs):3d} {statistics.median(xs):+6.1f} % "
              f"[{lo:+6.1f}, {hi:+6.1f}] {sum(1 for x in xs if x > 0):2d}/{len(xs)}")


if __name__ == "__main__":
    main()
