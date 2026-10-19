#!/usr/bin/env python3
"""data/positions.tsv.gz from the full runs (quant_quality.py's STRATA_LOGPOS rows, <run>.tsv or <run>.tsv.gz, each
with its <run>.tsv.seg.json):

    python make_positions.py --runs DIR [--out data/positions.tsv.gz]

One row per scored position: text, pos, target id, then for Q8_0 logp(target) and its top token, then for every IQ3_S
run logp(target), its top token and KL(Q8_0 || run) - the KL of tools/quant_quality.py compare: over Q8_0's 64 most
likely tokens, both renormalized to that set, a token the run does not list getting the run's 64th log-probability.
Only q8_0.tsv.gz is kept in full in data/ (no Q8_0 reference ships with Strata); the IQ3_S runs are this table.
"""
from __future__ import annotations

import argparse
import gzip
import json
import math
import pathlib

TOPK = 64
IQ3 = ["iq3s_cpu", "iq3s_cpu_iq4nl", "skip0", "skip7", "skip0_int8"]


def read_run(d: pathlib.Path, name: str) -> dict:
    p = d / f"{name}.tsv"
    rows = (open(p) if p.exists() else gzip.open(d / f"{name}.tsv.gz", "rt")).read().splitlines()
    out = {}
    for s in json.load(open(d / f"{name}.tsv.seg.json"))["segments"]:
        seg = {}
        for line in rows[s["first_line"]:s["end_line"]]:
            f = line.split("\t")
            if len(f) < 8 + TOPK:
                continue
            seg[int(f[0])] = (int(f[1]), float(f[2]), int(f[3]),
                              [(int(x), float(y)) for x, y in (z.split(":") for z in f[8:8 + TOPK])])
        out[s["name"]] = seg
    return out


def kl_at(r, a) -> float:
    kr, ka = r[3], a[3]
    da, floor = dict(ka), min(b for _, b in ka)
    pr = [math.exp(b) for _, b in kr]
    pa = [math.exp(da.get(i, floor)) for i, _ in kr]
    sr, sa = sum(pr), sum(pa)
    return sum((x / sr) * math.log((x / sr) / (y / sa)) for x, y in zip(pr, pa) if x > 0)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--runs", required=True, help="the folder with the full runs")
    ap.add_argument("--out", default=str(pathlib.Path(__file__).resolve().parent / "data" / "positions.tsv.gz"))
    a = ap.parse_args()
    d = pathlib.Path(a.runs)
    ref = read_run(d, "q8_0")
    runs = {k: read_run(d, k) for k in IQ3}
    head = ["text", "pos", "target", "q8_logp", "q8_top"]
    for k in IQ3:
        head += [f"{k}_logp", f"{k}_top", f"{k}_kl"]
    lines = ["\t".join(head)]
    for s in ref:
        for q in sorted(ref[s]):
            if not all(q in r[s] for r in runs.values()):
                continue
            t, lp, top, _ = ref[s][q]
            row = [s, str(q), str(t), f"{lp:.6f}", str(top)]
            for k in IQ3:
                x = runs[k][s][q]
                if x[0] != t:
                    raise SystemExit(f"{k} {s} {q}: target {x[0]} != {t} (not the same text)")
                row += [f"{x[1]:.6f}", str(x[2]), f"{kl_at(ref[s][q], x):.6f}"]
            lines.append("\t".join(row))
    with gzip.open(a.out, "wt", compresslevel=9) as f:
        f.write("\n".join(lines) + "\n")
    print(f"{len(lines) - 1} positions -> {a.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
