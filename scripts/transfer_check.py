#!/usr/bin/env python3
"""transfer_check.py - do calibrators fitted elsewhere still help on THIS data?

Reads a `jevport-suite.py ... real|text` item log (which is only meaningful when the run was taken
with the calibrators REMOVED, i.e. identity-served) and re-scores the test split under three maps:

  identity                 - no calibration at all
  shipped                  - the calibrator currently installed in the deployment (a_noul/c_noul/T_choice
                             from a calib-*.json), i.e. "what production was already doing"
  refit                    - the fit produced on this suite's own dev split

Why: the suite's own table only compares identity vs refit.  The question that decides whether a
calibrator ships is whether a fit made for one task still helps another - the banking77 result
(BoolQ/ARC/SST fit cost 13 accuracy points on banking77) says it often does not.

  transfer_check.py <real-item-log.jsonl> --shipped /path/calib-D.json [--qtype noul]
"""
import argparse
import json
import math

BLOCKS = [("noul", 120), ("choice", 40), ("score", 40)]      # real suite: pope 60+60, cvb 20+20, cb 20+20


def _sig(z):
    z = max(-700.0, min(700.0, z))
    return 1.0 / (1.0 + math.exp(-z)) if z >= 0 else math.exp(z) / (1.0 + math.exp(z))


def apply_platt(pr, a, c):
    p = min(max(pr[1] if len(pr) > 1 else 0.0, 1e-9), 1 - 1e-9)
    q = _sig(a * math.log(p / (1 - p)) + c)
    return [1 - q, q]


def apply_temp(pr, t):
    if t <= 0.01 or abs(t - 1.0) < 1e-9:
        return list(pr)
    lm = math.log(max(max(pr), 1e-300))
    lse = math.log(sum(math.exp((math.log(max(x, 1e-300)) - lm) / t) for x in pr))
    out = [math.exp((math.log(max(x, 1e-300)) - lm) / t - lse) for x in pr]
    s = sum(out)
    return [x / s for x in out]


def ece(pairs, bins=15):
    if not pairs:
        return 0.0
    tot, s = len(pairs), 0.0
    for b in range(bins):
        lo, hi = b / bins, (b + 1) / bins
        sel = [(p, y) for p, y in pairs if (lo <= p < hi) or (b == bins - 1 and p == 1.0)]
        if sel:
            s += len(sel) / tot * abs(sum(p for p, _ in sel) / len(sel) - sum(y for _, y in sel) / len(sel))
    return s


def score(rows, mapfn):
    pairs, acc, nll, brier = [], 0, 0.0, 0.0
    for pr, g in rows:
        q = mapfn(list(pr))
        am = max(range(len(q)), key=lambda i: q[i])
        ok = am == g
        acc += ok
        pairs.append((max(q[am], 1e-9), ok))
        nll -= math.log(max(q[g] if g < len(q) else 1e-9, 1e-9))
        brier += sum((q[i] - (1.0 if i == g else 0.0)) ** 2 for i in range(len(q)))
    n = len(rows)
    return {"acc": acc / n, "ece": ece(pairs), "nll": nll / n, "brier": brier / n, "n": n}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("log")
    ap.add_argument("--shipped", default="/opt/data/llm-swap/strata/Jev-port-cuda/calib/calib-D.json")
    args = ap.parse_args()

    rows = [json.loads(l) for l in open(args.log) if l.strip()]
    ship = json.load(open(args.shipped))
    sa, sc = float(ship.get("a_noul", 1.0)), float(ship.get("c_noul", 0.0))
    sT_choice, sT_score = float(ship.get("T_choice", 1.0)), float(ship.get("T_score", 1.0))
    print(f"shipped {args.shipped.split('/')[-1]}: a_noul={sa} c_noul={sc} T_choice={sT_choice} T_score={sT_score} "
          f"({ship.get('provenance', '?')})\n")

    off = 0
    for qtype, n in BLOCKS:
        block = rows[off:off + n]
        off += n
        if len(block) < n:
            print(f"{qtype}: block short ({len(block)}/{n}), skipped")
            continue
        dev, test = block[:n // 2], block[n // 2:]
        dev = [(r["pr"], r["gold"]) for r in dev]
        test = [(r["pr"], r["gold"]) for r in test]

        if qtype == "noul":
            ds = [math.log(max(p[1], 1e-9) / max(1 - p[1], 1e-9)) for p, _ in dev]
            cs = [g for _, g in dev]
            # simple IRLS-free Platt by gradient descent (same as the suite, kept local to stay standalone)
            a, c = 1.0, 0.0
            for _ in range(500):
                ga = gc = 0.0
                for d, y in zip(ds, cs):
                    p = _sig(a * d + c)
                    ga += (p - y) * d
                    gc += (p - y)
                a -= 0.1 * ga / len(ds)
                c -= 0.1 * gc / len(ds)
            refit = lambda pr, a=a, c=c: apply_platt(pr, a, c)
            label = f"refit a={a:.3f} c={c:.3f}"
            shipped = lambda pr: apply_platt(pr, sa, sc)
        else:
            best, bt = 1e9, 1.0
            for k in range(0, 60):
                t = 2 ** (-3 + 6 * k / 59)
                l = 0.0
                for pr, g in dev:
                    l -= math.log(max(apply_temp(pr, t)[g], 1e-12))
                if l < best:
                    best, bt = l, t
            refit = lambda pr, t=bt: apply_temp(pr, t)
            label = f"refit T={bt:.3f}"
            # score and choice use DIFFERENT temperatures in the deployed file - using T_choice for
            # score rows silently measures the wrong map (this caught that exact mistake).
            shipped = lambda pr, t=(sT_score if qtype == "score" else sT_choice): apply_temp(pr, t)

        print(f"{qtype} (n_test={len(test)})")
        for name, fn in (("identity", lambda pr: list(pr)), ("shipped", shipped), (label, refit)):
            s = score(test, fn)
            print(f"  {name:<20} acc={s['acc']:.3f}  ece={s['ece']:.4f}  nll={s['nll']:.4f}  brier={s['brier']:.4f}")
        print()


if __name__ == "__main__":
    main()
