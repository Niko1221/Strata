#!/usr/bin/env python3
"""Cross-domain calibration transfer test.

Question this answers: is a Platt fit on one task transferable to another task,
or is the calibrator genuinely per-task (the claim in the PR thread)?

Usage:
  cross_domain_calib.py --sets bank=/tmp/mf-C.json,pope=...,gqa=/tmp/gqa-C.json.rows.jsonl \
                        --fit bank --eval bank,pope,gqa
  cross_domain_calib.py --sweep        # every fit x every eval

Every set is reduced to rows of (id, gold 0/1, p_yes) so all three tasks land on
one common footing: binary yes/no, probability of class 1.
"""
import argparse
import json
import math
import os

# ---------------------------------------------------------------- loaders

def load_bank(path):
    d = json.load(open(path))
    rows = d["rows"] if isinstance(d, dict) and "rows" in d else d
    return [(r["id"], int(r["gold"]["is_card"]), float(r["p_yes"])) for r in rows]


def load_suite_log(path, prefix):
    out = []
    for line in open(path):
        line = line.strip()
        if not line:
            continue
        r = json.loads(line)
        if not str(r.get("qid", "")).startswith(prefix):
            continue
        pr = r.get("pr")
        if not pr or len(pr) != 2:
            continue
        out.append((r["qid"], int(r["gold"]), float(pr[1])))
    return out


def load_rows_jsonl(path):
    return [(r["id"], int(r["gold"]), float(r["p"])) for r in map(json.loads, open(path)) if r.get("p") is not None]


LOADERS = {
    "bank": load_bank,
    "gqa": load_rows_jsonl,
}


def load_set(name, path):
    # route by file shape, not by name: .json is the banking77 result shape, a .jsonl
    # is either a jevport-suite item log (has qid) or a runner row dump (has p).
    if path.endswith(".json"):
        return load_bank(path)
    first = next((json.loads(l) for l in open(path) if l.strip()), {})
    if "qid" in first:
        return load_suite_log(path, name.split("_")[0])
    if "p" in first:
        return load_rows_jsonl(path)
    raise SystemExit(f"{name}: unrecognised row shape in {path}: keys={sorted(first)}")


# ---------------------------------------------------------------- maths

def sig(z):
    return 1.0 / (1.0 + math.exp(-max(-700.0, min(700.0, z))))


def platt_p(p, a, c):
    q = min(max(p, 1e-9), 1 - 1e-9)
    return sig(a * math.log(q / (1 - q)) + c)


def fit_platt(rows, ridge=1e-6):
    """MLE of P(yes) = sigmoid(a*logit(p_raw) + c).

    Copies the proven formulation from fit_calib_mf.py: standardise x (raw logits
    reach +-20 and make the plain normal equations ill-conditioned) and step-halve
    so Newton cannot diverge. Returns (a, c) in the *unstandardised* parameterisation
    that the calibrators and /admin/calib expect.
    """
    xs = [math.log(min(max(p, 1e-9), 1 - 1e-9) / (1 - min(max(p, 1e-9), 1 - 1e-9))) for _, _, p in rows]
    ys = [float(g) for _, g, _ in rows]
    n = len(xs)
    mx = sum(xs) / n
    sdx = math.sqrt(sum((x - mx) ** 2 for x in xs) / n) or 1.0
    zs = [(x - mx) / sdx for x in xs]
    az, cz = 1.0, 0.0

    def nll(az_, cz_):
        return -sum(math.log(max(sig(az_ * z + cz_) if y else 1 - sig(az_ * z + cz_), 1e-12))
                    for z, y in zip(zs, ys))

    cur = nll(az, cz)
    for _ in range(200):
        g0 = g1 = h00 = h01 = h11 = 0.0
        for z, y in zip(zs, ys):
            pr = sig(az * z + cz)
            w = pr * (1 - pr)
            g0 += (y - pr) * z
            g1 += (y - pr)
            h00 += w * z * z
            h01 += w * z
            h11 += w
        g0 -= ridge * az
        g1 -= ridge * cz
        h00 += ridge
        h11 += ridge
        det = h00 * h11 - h01 * h01
        if abs(det) < 1e-12:
            break
        da = (h11 * g0 - h01 * g1) / det
        dc = (h00 * g1 - h01 * g0) / det
        step = 1.0
        for _ in range(30):
            cand = nll(az + step * da, cz + step * dc)
            if cand <= cur:
                break
            step *= 0.5
        else:
            break
        az, cz = az + step * da, cz + step * dc
        cur = cand
        if abs(step * da) < 1e-10 and abs(step * dc) < 1e-10:
            break
    a, c = az / sdx, cz - az * mx / sdx
    # Divergence guard: Newton on a separable/saturated split can run away instead of settling
    # (seen 2026-10-23: an early version returned a=-199109, c=-128141 and the caller printed it as
    # a fit).  A Platt slope far from the 0.05..20 range carries no information - say so loudly
    # rather than let it reach a calibrator file.
    if not (math.isfinite(a) and math.isfinite(c)) or abs(a) > 20.0 or abs(c) > 50.0:
        print(f"  !! WARNING: Platt fit did not converge (a={a:.6g} c={c:.6g}); "
              f"the split is probably separable - do NOT install this fit", flush=True)
    return a, c


def brier(rows):
    return sum((p - g) ** 2 for _, g, p in rows) / len(rows)


def acc(rows):
    return sum(1 for _, g, p in rows if (p >= 0.5) == (g == 1)) / len(rows)


def ece15(rows):
    bins = [[] for _ in range(15)]
    for _, g, p in rows:
        b = min(14, int(p * 15))
        bins[b].append((p, g))
    tot = len(rows)
    e = 0.0
    for b in bins:
        if not b:
            continue
        mp = sum(p for p, _ in b) / len(b)
        mg = sum(g for _, g in b) / len(b)
        e += len(b) / tot * abs(mp - mg)
    return e


def nll(rows):
    return -sum(math.log(max(g and p or (1 - p), 1e-12)) for _, g, p in rows) / len(rows)


def report(tag, rows):
    return (f"{tag:<28} n={len(rows):<4} acc={acc(rows):.3f} brier={brier(rows):.4f} "
            f"nll={nll(rows):.4f} ece={ece15(rows):.4f}")


def apply(rows, a, c):
    return [(i, g, platt_p(p, a, c)) for i, g, p in rows]


# ---------------------------------------------------------------- main

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--sets", required=True,
                    help="name=path pairs, comma separated. Suite logs use the qid prefix as the name.")
    ap.add_argument("--fit", required=True, help="name of the set to fit Platt on")
    ap.add_argument("--eval", required=True, help="comma separated set names to score")
    ap.add_argument("--split", type=float, default=0.0,
                    help="if >0, fit on the first (1-split) fraction of the fit set only")
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    specs = dict(kv.split("=", 1) for kv in args.sets.split(","))
    sets = {}
    for name, path in specs.items():
        if not os.path.exists(path):
            raise SystemExit(f"missing {name}: {path}")
        rows = load_set(name, path)
        if not rows:
            raise SystemExit(f"{name}: parsed 0 rows from {path}")
        sets[name] = rows

    fit_rows = sets[args.fit]
    if args.split > 0:
        cut = int(len(fit_rows) * (1 - args.split))
        fit_rows = fit_rows[:cut]
    a, c = fit_platt(fit_rows)
    print(f"fit on {args.fit} (n={len(fit_rows)}): a={a:.6f} c={c:.6f}")

    out = {"fit_set": args.fit, "fit_n": len(fit_rows), "a": a, "c": c, "evals": {}}
    for name in args.eval.split(","):
        rows = sets[name]
        before, after = apply(rows, 1.0, 0.0), apply(rows, a, c)
        flips = sum(1 for b, f in zip(before, after) if (b[2] >= 0.5) != (f[2] >= 0.5))
        print(report(f"{name} raw", rows))
        print(report(f"{name} <- {args.fit} platt", after))
        print(f"{'':<28} flips={flips}/{len(rows)} "
              f"({100.0 * flips / len(rows):.1f}%)")
        out["evals"][name] = {
            "n": len(rows),
            "raw": {"acc": acc(rows), "brier": brier(rows), "nll": nll(rows), "ece": ece15(rows)},
            "cal": {"acc": acc(after), "brier": brier(after), "nll": nll(after), "ece": ece15(after)},
            "flips": flips,
        }
        print()

    if args.out:
        json.dump(out, open(args.out, "w"), indent=2)
        print(f"wrote {args.out}")


if __name__ == "__main__":
    main()
