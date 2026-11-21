#!/usr/bin/env python3
"""G3: how many of the ten gold words each arm's generation names.

    python tools/kv_g3_score.py --gold data/quant_quality/cwe-4k/gold.json \\
        --anchor fp16=runs\\fp16.json --base int8=runs\\int8.json \\
        --arm q4lm32=runs\\q4lm32.json ... --out score.json

`recall` counts the gold words that appear anywhere in the generated text (a `[a-z]+` run, lowercased);
`recall_first10` counts them among the first ten distinct six-letter words, which is the "did it name the
right ten" reading.  Differences are paired over segments (cluster = segment, n = 8, t with df = 7), with
the same 1.895 * sd / sqrt(8) bound as G1 and G2.  The saturation and floor rule of the pre-registration
is applied to the anchor and printed, not hidden.
"""
from __future__ import annotations

import argparse
import json
import math
import pathlib
import re

T7_95 = 1.895          # one-sided 95 % for df = 7, as in G1 and G2
SATURATED = 9.5        # of 10: the anchor is at the ceiling
FLOOR = 2.5            # of 10: the anchor cannot do the task

WORD = re.compile(r"[a-z]+")


def words(text: str) -> list[str]:
    return WORD.findall(text.lower())


def score_segment(text: str, gold: list[str]) -> dict:
    ws = words(text)
    six = [w for w in ws if len(w) == 6]
    distinct = list(dict.fromkeys(six))
    hit = [w for w in gold if w in set(ws)]
    hit10 = [w for w in gold if w in set(distinct[:10])]
    return {"words": len(ws), "six": len(six), "distinct6": len(distinct),
            "hits": len(hit), "recall": len(hit) / len(gold),
            "hits10": len(hit10), "recall_first10": len(hit10) / len(gold),
            "missing": [w for w in gold if w not in set(ws)]}


def paired(a: dict, b: dict, key: str) -> dict | None:
    segs = [s for s in a if s in b]
    if len(segs) < 2:
        return None
    d = [a[s][key] - b[s][key] for s in segs]
    n = len(d)
    m = math.fsum(d) / n
    sd = math.sqrt(math.fsum((x - m) ** 2 for x in d) / (n - 1))
    sem = sd / math.sqrt(n)
    return {"n": n, "mean": m, "sd": sd, "sem": sem, "bound": T7_95 * sem,
            "a_worse": sum(1 for x in d if x < 0), "a_better": sum(1 for x in d if x > 0)}


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--gold", default="data/quant_quality/cwe-4k/gold.json")
    ap.add_argument("--anchor", default="fp16", help="name of the anchor run (default fp16)")
    ap.add_argument("--base", default="int8", help="name of the default arm (default int8)")
    ap.add_argument("--run", action="append", required=True, metavar="NAME=PATH")
    ap.add_argument("--out", default="")
    a = ap.parse_args()

    gold = {g["name"]: g["answer"] for g in json.load(open(a.gold, encoding="utf-8"))}
    runs, meta = {}, {}
    for spec in a.run:
        name, _, path = spec.partition("=")
        d = json.load(open(path, encoding="utf-8"))
        meta[name] = {"path": path, "kv": d.get("kv"), "env": d.get("env"), "new": d.get("new"),
                      "short_read": d.get("short_read"), "drop_spec": d.get("drop_spec")}
        runs[name] = {s["name"]: {**score_segment(s["text"], gold[s["name"]]), "generated": s["generated"],
                                  "stopped": s["stopped"], "text": s["text"]}
                      for s in d["segments"]}

    names = list(runs)
    res = {"runs": {}, "paired": {}, "verdict": {}}
    print(f"{'run':10s} {'recall':>7s} {'first10':>8s} {'words':>6s} {'distinct6':>9s} "
          f"{'eos':>4s}  per segment (hits of 10)")
    for n in names:
        r = runs[n]
        segs = sorted(r)
        m = math.fsum(r[s]["recall"] for s in segs) / len(segs)
        m10 = math.fsum(r[s]["recall_first10"] for s in segs) / len(segs)
        res["runs"][n] = {"meta": meta[n], "mean": m, "mean_first10": m10,
                          "per_segment": {s: {k: v for k, v in r[s].items() if k != "text"} for s in segs}}
        eos = sum(1 for s in segs if r[s]["stopped"] == "eos")
        print(f"{n:10s} {m:7.3f} {m10:8.3f} "
              f"{math.fsum(r[s]['words'] for s in segs) / len(segs):6.1f} "
              f"{math.fsum(r[s]['distinct6'] for s in segs) / len(segs):9.1f} {eos:4d}  "
              + " ".join(str(r[s]["hits"]) for s in segs))

    for ref in (a.base, a.anchor):
        if ref not in runs:
            continue
        res["paired"][ref] = {}
        print(f"\npaired against {ref} (mean, bound = 1.895 sd/sqrt(8), segments worse/better):")
        for n in names:
            if n == ref:
                continue
            for key in ("recall", "recall_first10"):
                p = paired(runs[n], runs[ref], key)
                if p is None:
                    continue
                res["paired"][ref].setdefault(n, {})[key] = p
                flag = ">" if p["mean"] > p["bound"] else "<" if p["mean"] < -p["bound"] else "~"
                print(f"  {n:10s} {key:14s} {p['mean']:+.3f} {flag}  bound {p['bound']:.3f}  "
                      f"{p['a_worse']}/{p['a_better']}")

    if a.anchor in runs:
        # `recall` is a fraction (0..1) and SATURATED/FLOOR are on the 0..10 scale: compare the hits, not the share.
        # Before 2026-10-31 this compared 0.950 against 9.5 and called a ceiling a floor.
        m = res["runs"][a.anchor]["mean"]
        hits = 10.0 * m
        if hits >= SATURATED:
            v = "saturated"
        elif hits <= FLOOR:
            v = "floor"
        else:
            v = "resolution"
        res["verdict"] = {"anchor": a.anchor, "anchor_mean": m, "anchor_hits_of_10": hits, "verdict": v,
                          "saturated_at": SATURATED, "floor_at": FLOOR}
        print(f"\nanchor {a.anchor}: mean recall {m:.3f} = {hits:.2f} of 10 -> {v} "
              f"(ceiling {SATURATED}, floor {FLOOR})")

    if a.out:
        pathlib.Path(a.out).write_text(json.dumps(res, indent=1))
        print(f"wrote {a.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
