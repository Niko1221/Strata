"""Compare two scored runs (score.py output) on the ids they share.

    python compare.py REF_scored.jsonl CAND_scored.jsonl [--margin 2.0] [--boot 10000] [--seed 0]

REF is the reference arm (the unchanged engine and model), CAND the candidate (the lossy change).  Prints, per set and pooled over all shared items:
accuracy of both arms, the paired difference (CAND - REF, in percentage points) with a bootstrap 95% CI (items resampled in pairs),
and the exact two-sided McNemar p-value.  Verdict on the pooled difference, for a non-inferiority margin M (percentage points):
  NON-INFERIOR  the CI's lower bound is above -M;  INFERIOR  its upper bound is below -M;  INCONCLUSIVE  the CI contains -M.
(A two-sided 95% CI's lower bound is the one-sided 97.5% bound.)  Mean output tokens per arm and the share of replies cut off at max_tokens
are shown too: a change can also change the answer length.  Exit code 0 NON-INFERIOR, 1 INCONCLUSIVE, 2 INFERIOR.
"""
from __future__ import annotations

import argparse
import json
import random
import sys
from collections import defaultdict
from math import comb
from pathlib import Path


def mcnemar_exact(b: int, c: int) -> float:
    """Two-sided exact p for b = REF right/CAND wrong and c = REF wrong/CAND right (binomial, p 0.5)."""
    n = b + c
    if n == 0:
        return 1.0
    k = min(b, c)
    return min(1.0, 2 * sum(comb(n, i) for i in range(k + 1)) / 2 ** n)


def bootstrap_ci(diffs: list[int], boot: int, rng: random.Random) -> tuple[float, float]:
    n = len(diffs)
    means = sorted(sum(rng.choices(diffs, k=n)) / n for _ in range(boot))
    return means[int(0.025 * boot)], means[min(boot - 1, int(0.975 * boot))]


def load(path: str) -> dict[str, dict]:
    return {r["id"]: r for r in map(json.loads, Path(path).read_text(encoding="utf-8").splitlines()) if r}


def stats(a: list[dict], b: list[dict], boot: int, seed: int) -> dict:
    diffs = [int(y["correct"]) - int(x["correct"]) for x, y in zip(a, b)]
    n = len(diffs)
    lo, hi = bootstrap_ci(diffs, boot, random.Random(seed))
    tok = lambda rows: [r["completion_tokens"] for r in rows if r.get("completion_tokens") is not None]
    cut = lambda rows: sum(r.get("finish_reason") == "length" for r in rows) / len(rows)
    ta, tb = tok(a), tok(b)
    return {"n": n, "acc_a": sum(x["correct"] for x in a) / n, "acc_b": sum(y["correct"] for y in b) / n,
            "diff": sum(diffs) / n, "lo": lo, "hi": hi,
            "p": mcnemar_exact(sum(d < 0 for d in diffs), sum(d > 0 for d in diffs)),
            "only_a": sum(d < 0 for d in diffs), "only_b": sum(d > 0 for d in diffs),
            "tok_a": sum(ta) / len(ta) if ta else float("nan"), "tok_b": sum(tb) / len(tb) if tb else float("nan"),
            "cut_a": cut(a), "cut_b": cut(b)}


def verdict(pooled: dict, margin_pp: float) -> str:
    m = margin_pp / 100
    if pooled["lo"] > -m:
        return "NON-INFERIOR"
    if pooled["hi"] < -m:
        return "INFERIOR"
    return "INCONCLUSIVE"


def compare(ra: dict, rb: dict, margin: float = 2.0, boot: int = 10000, seed: int = 0) -> dict:
    ids = sorted(set(ra) & set(rb))
    by = defaultdict(list)
    for i in ids:
        by[ra[i]["set"]].append(i)
    per = {s: stats([ra[i] for i in v], [rb[i] for i in v], boot, seed) for s, v in sorted(by.items())}
    pooled = stats([ra[i] for i in ids], [rb[i] for i in ids], boot, seed)
    return {"ids": len(ids), "only_a": len(set(ra) - set(rb)), "only_b": len(set(rb) - set(ra)), "per_set": per, "pooled": pooled,
            "margin_pp": margin, "verdict": verdict(pooled, margin)}


def render(res: dict) -> str:
    rows = [f"{'set':16s}{'n':>5s}{'ref %':>7s}{'cand %':>7s}{'c-r pp':>8s}{'95% CI pp':>17s}{'ref only':>9s}{'cand only':>10s}{'McNemar p':>10s}"
            f"{'tok ref':>8s}{'tok cnd':>8s}{'cut ref':>8s}{'cut cnd':>8s}"]
    for name, s in [*res["per_set"].items(), ("POOLED", res["pooled"])]:
        rows.append(f"{name:16s}{s['n']:5d}{100 * s['acc_a']:7.1f}{100 * s['acc_b']:7.1f}{100 * s['diff']:+8.2f}"
                    f"  [{100 * s['lo']:+6.2f},{100 * s['hi']:+6.2f}]{s['only_a']:9d}{s['only_b']:10d}{s['p']:10.4f}"
                    f"{s['tok_a']:8.0f}{s['tok_b']:8.0f}{100 * s['cut_a']:7.1f}%{100 * s['cut_b']:7.1f}%")
    rows.append(f"\n{res['ids']} shared ids ({res['only_a']} only in REF, {res['only_b']} only in CAND ignored)")
    rows.append(f"non-inferiority margin {res['margin_pp']} pp on the pooled accuracy: {res['verdict']}")
    return "\n".join(rows)


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("a")
    ap.add_argument("b")
    ap.add_argument("--margin", type=float, default=2.0, help="non-inferiority margin, percentage points (default 2)")
    ap.add_argument("--boot", type=int, default=10000)
    ap.add_argument("--seed", type=int, default=0)
    ns = ap.parse_args(argv)
    res = compare(load(ns.a), load(ns.b), ns.margin, ns.boot, ns.seed)
    print(render(res))
    return {"NON-INFERIOR": 0, "INCONCLUSIVE": 1, "INFERIOR": 2}[res["verdict"]]


if __name__ == "__main__":
    sys.exit(main())
