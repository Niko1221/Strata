#!/usr/bin/env python3
"""fit_calib_mf.py - fit a per-readout calibrator (Platt for noul, temperature for choice)
on LABELLED multi-field rows, and report ECE/Brier before vs after on held-out rows.

Why the inversion step: the rows we collect come from a server that ALREADY applies
calib-L.json, so `p_yes` / `group_probs` in an arm file are f_old(raw_pr).  The inference
path applies the calibrator to RAW probabilities, so a fit on the collected values would
be g(f_old(raw)) at fit time but g(raw) live - a silent mismatch.  We invert f_old exactly
(Platt is affine in logit space, temperature is a power law) and fit on raw.

  fit_calib_mf.py --train mf-B-train.json --test mf-B.json --old-calib calib-L.json \
                  --out calib-L-new.json [--folds 5]

Reported, per field type, before (old calibrator) and after (new fit) on the TEST file:
accuracy, mean confidence, Brier, NLL, and 15-bin ECE.  Train ECE is also cross-validated so an
overfit (n=600, 2 parameters) cannot hide.

WARNING, measured 2026-10-23: the temperature maps are argmax-preserving, but Platt for `noul` is
NOT accuracy-neutral - a non-zero offset c is a decision threshold, and a fitted (a, c) moved
held-out banking77 from .670 to .800.  So "accuracy must be identical" is right for choice/score
and WRONG for noul; report the accuracy delta per type and treat a |delta| that comes mostly from
c as a threshold/prior decision that has to be justified on representative data, never as
calibration quality.  A fit whose accuracy moves on a balanced set will not survive a skewed one.
"""
import argparse
import json
import math


# ---------- the two calibration maps (must mirror serve/jev_calib.py exactly) ----------
def apply_noul(p_yes, a, c):
    q = min(max(p_yes, 1e-9), 1 - 1e-9)
    d = math.log(q / (1 - q))
    return 1.0 / (1.0 + math.exp(-(a * d + c)))


def apply_temp(pr, T):
    if T <= 0.01 or abs(T - 1.0) < 1e-9:
        return list(pr)
    lm = math.log(max(max(pr), 1e-300))
    lse = math.log(sum(math.exp((math.log(max(x, 1e-300)) - lm) / T) for x in pr))
    out = [math.exp((math.log(max(x, 1e-300)) - lm) / T - lse) for x in pr]
    s = sum(out)
    return [x / s for x in out]


def invert_noul(q_cal, a, c):
    """q_cal = sigmoid(a*d_raw + c) -> p_raw."""
    q = min(max(q_cal, 1e-9), 1 - 1e-9)
    d = (math.log(q / (1 - q)) - c) / a
    return 1.0 / (1.0 + math.exp(-d))


def invert_temp(pr_cal, T):
    """apply_temp raised each label to 1/T, so the inverse is the power T."""
    if T <= 0.01 or abs(T - 1.0) < 1e-9:
        return list(pr_cal)
    p = [max(x, 1e-12) ** T for x in pr_cal]
    s = sum(p)
    return [x / s for x in p]


# ---------- data ----------
def load_rows(path, a_old, c_old, T_old):
    """-> list of {gold_card, gold_group, p_raw_yes, pr_raw, n_labels} (raw = pre-calibrator)."""
    d = json.load(open(path))
    out = []
    for r in d["rows"]:
        if r.get("p_yes") is None or not r.get("group_probs"):
            continue
        out.append({
            "gold_card": r["gold"]["is_card"],
            "gold_group": r["gold"]["group"],
            "p_raw_yes": invert_noul(r["p_yes"], a_old, c_old),
            "pr_raw": invert_temp([float(x) for x in r["group_probs"]], T_old),
            "labels": None,          # filled from the set file when needed
            "id": r["id"],
        })
    return out


def labels_for(path, ids: list[str]):
    """group label order for each id, from the .jsonl set file (criteria order)."""
    want = set(ids)
    got = {}
    for line in open(path):
        line = line.strip()
        if not line:
            continue
        it = json.loads(line)
        if it["id"] in want:
            got[it["id"]] = list(it["questions"]["group"]["criteria"])
    return got


# ---------- fit ----------
def _sig(z):
    """overflow-safe logistic."""
    z = max(-700.0, min(700.0, z))
    if z >= 0:
        return 1.0 / (1.0 + math.exp(-z))
    e = math.exp(z)
    return e / (1.0 + e)


def fit_platt(rows, ridge=1e-6):
    """MLE of P(yes) = sigmoid(a*logit(p_raw) + c); Newton-Raphson on standardised x
    (raw logits can be +-20, which makes the plain normal equations ill-conditioned)."""
    xs = [math.log(r["p_raw_yes"] / (1 - r["p_raw_yes"])) for r in rows]
    ys = [float(r["gold_card"]) for r in rows]
    n = len(xs)
    mx = sum(xs) / n
    sdx = math.sqrt(sum((x - mx) ** 2 for x in xs) / n) or 1.0
    zs = [(x - mx) / sdx for x in xs]
    az, cz = 1.0, 0.0

    def nll(az_, cz_):
        s = 0.0
        for z, y in zip(zs, ys):
            p = _sig(az_ * z + cz_)
            s -= math.log(max(p if y else 1 - p, 1e-12))
        return s

    cur = nll(az, cz)
    for _ in range(200):
        g0 = g1 = h00 = h01 = h11 = 0.0
        for z, y in zip(zs, ys):
            p = _sig(az * z + cz)
            w = p * (1 - p)
            g0 += (y - p) * z
            g1 += (y - p)
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
        for _ in range(30):                       # step-halving keeps Newton monotone
            cand = nll(az + step * da, cz + step * dc)
            if cand <= cur:
                break
            step *= 0.5
        else:
            break
        az, cz = az + step * da, cz + step * dc
        done = abs(step * da) < 1e-10 and abs(step * dc) < 1e-10
        cur = cand
        if done:
            break
    return az / sdx, cz - az * mx / sdx


def fit_temp(rows, lo=0.05, hi=8.0):
    """T minimising NLL of the gold label under p_raw**(1/T)."""
    def nll(T):
        s = 0.0
        for r in rows:
            p = apply_temp(r["pr_raw"], T)
            idx = r["gold_idx"]
            s -= math.log(max(p[idx], 1e-12))
        return s / len(rows)

    # coarse scan then golden-section (NLL is unimodal in T for a power family)
    best, best_v = 1.0, nll(1.0)
    T = lo
    while T <= hi:
        v = nll(T)
        if v < best_v:
            best, best_v = T, v
        T *= 1.15
    a, b = best / 1.15, min(best * 1.15, hi)
    gr = (math.sqrt(5) - 1) / 2
    c = b - gr * (b - a)
    d = a + gr * (b - a)
    fc, fd = nll(c), nll(d)
    for _ in range(60):
        if fc < fd:
            b, d, fd = d, c, fc
            c = b - gr * (b - a)
            fc = nll(c)
        else:
            a, c, fc = c, d, fd
            d = a + gr * (b - a)
            fd = nll(d)
        if b - a < 1e-6:
            break
    T = (a + b) / 2
    return T


# ---------- metrics ----------
def ece(pairs, bins=15):
    """pairs: [(conf, correct)] -> expected calibration error."""
    if not pairs:
        return 0.0
    tot = len(pairs)
    acc = 0.0
    for b in range(bins):
        lo, hi = b / bins, (b + 1) / bins
        sel = [(p, y) for p, y in pairs if (lo <= p < hi) or (b == bins - 1 and p == 1.0)]
        if not sel:
            continue
        conf = sum(p for p, _ in sel) / len(sel)
        hit = sum(y for _, y in sel) / len(sel)
        acc += len(sel) / tot * abs(conf - hit)
    return acc


def evaluate(rows, a, c, T, labels_by_id):
    """-> dict of metrics for both fields under the given calibrator."""
    noul_pairs, ch_pairs = [], []
    brier = nll = 0.0
    acc_c = acc_g = 0
    conf_c_sum = conf_g_sum = 0.0
    for r in rows:
        q = apply_noul(r["p_raw_yes"], a, c)
        pred_c = 1 if q >= 0.5 else 0
        ok = pred_c == r["gold_card"]
        acc_c += ok
        conf_c = q if pred_c == 1 else 1 - q
        conf_c_sum += conf_c
        noul_pairs.append((conf_c, ok))
        brier += (q - r["gold_card"]) ** 2
        nll -= math.log(max(q if r["gold_card"] == 1 else 1 - q, 1e-12))
        pr = apply_temp(r["pr_raw"], T)
        j = max(range(len(pr)), key=lambda i: pr[i])
        names = r["labels"] or labels_by_id.get(r["id"]) or []
        pred_g = names[j] if j < len(names) else None
        ok_g = pred_g == r["gold_group"]
        acc_g += ok_g
        conf_g_sum += pr[j]
        ch_pairs.append((pr[j], ok_g))
        idx = names.index(r["gold_group"]) if r["gold_group"] in names else 0
        nll -= math.log(max(pr[idx], 1e-12))
    n = len(rows)
    return {
        "n": n,
        "is_card": {"acc": acc_c / n, "mean_conf": conf_c_sum / n, "brier": brier / n,
                    "ece": ece(noul_pairs), "nll": nll / (2 * n)},
        "group": {"acc": acc_g / n, "mean_conf": conf_g_sum / n,
                  "ece": ece(ch_pairs), "nll": nll / (2 * n)},
        "gap_card": conf_c_sum / n - acc_c / n,
        "gap_group": conf_g_sum / n - acc_g / n,
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--train", required=True)
    ap.add_argument("--test", required=True)
    ap.add_argument("--set-train", default="/opt/data/eval/banking/mf-600-train.jsonl")
    ap.add_argument("--set-test", default="/opt/data/eval/banking/mf-200.jsonl")
    ap.add_argument("--old-calib", default="/opt/data/llm-swap/strata/Jev-port-cuda/calib/calib-L.json")
    ap.add_argument("--out", default="")
    ap.add_argument("--folds", type=int, default=5)
    args = ap.parse_args()

    old = json.load(open(args.old_calib))
    a_old, c_old, T_old = float(old["a_noul"]), float(old["c_noul"]), float(old["T_choice"])
    print(f"old calibrator: a_noul={a_old} c_noul={c_old} T_choice={T_old}  ({old.get('provenance','?')})")

    tr_raw = load_rows(args.train, a_old, c_old, T_old)
    te_raw = load_rows(args.test, a_old, c_old, T_old)
    lab_tr = labels_for(args.set_train, [r["id"] for r in tr_raw])
    lab_te = labels_for(args.set_test, [r["id"] for r in te_raw])
    for r in tr_raw:
        r["labels"] = lab_tr.get(r["id"], [])
        r["gold_idx"] = r["labels"].index(r["gold_group"]) if r["gold_group"] in r["labels"] else 0
    for r in te_raw:
        r["labels"] = lab_te.get(r["id"], [])
        r["gold_idx"] = r["labels"].index(r["gold_group"]) if r["gold_group"] in r["labels"] else 0
    print(f"train n={len(tr_raw)}  test n={len(te_raw)}")

    a, c = fit_platt(tr_raw)
    T = fit_temp(tr_raw)
    print(f"NEW calibrator: a_noul={a:.4f} c_noul={c:.4f} T_choice={T:.4f}")

    before = evaluate(te_raw, a_old, c_old, T_old, lab_te)
    after = evaluate(te_raw, a, c, T, lab_te)

    # cross-validate the fit on the train file so a lucky fit cannot pass as real
    cv = []
    k = max(2, args.folds)
    for f in range(k):
        fit_rows = [r for i, r in enumerate(tr_raw) if i % k != f]
        ev_rows = [r for i, r in enumerate(tr_raw) if i % k == f]
        fa, fc = fit_platt(fit_rows)
        fT = fit_temp(fit_rows)
        m0 = evaluate(ev_rows, a_old, c_old, T_old, lab_tr)
        m1 = evaluate(ev_rows, fa, fc, fT, lab_tr)
        cv.append((m0["is_card"]["ece"], m1["is_card"]["ece"],
                   m0["group"]["ece"], m1["group"]["ece"]))
    cv_before_c = sum(x[0] for x in cv) / k
    cv_after_c = sum(x[1] for x in cv) / k
    cv_before_g = sum(x[2] for x in cv) / k
    cv_after_g = sum(x[3] for x in cv) / k

    def line(tag, m):
        print(f"  {tag:<8} is_card acc={m['is_card']['acc']:.3f} conf={m['is_card']['mean_conf']:.3f} "
              f"brier={m['is_card']['brier']:.4f} ece={m['is_card']['ece']:.4f} | "
              f"group acc={m['group']['acc']:.3f} conf={m['group']['mean_conf']:.3f} "
              f"ece={m['group']['ece']:.4f}")

    print("\nHELD-OUT (never used in the fit):")
    line("before", before)
    line("after", after)
    print(f"\n{k}-fold CV on train: is_card ECE {cv_before_c:.4f} -> {cv_after_c:.4f} | "
          f"group ECE {cv_before_g:.4f} -> {cv_after_g:.4f}")

    out = {
        "tuple": "qfn-iq3_xxs|jev-port|none|v1|L|mf-banking77",
        "a_noul": round(a, 6),
        "c_noul": round(c, 6),
        "T_choice": round(T, 6),
        "T_score": float(old.get("T_score", 1.0)),
        "provenance": (f"fit {args.train} (banking77 two-field is_card+group, seed 11, n={len(tr_raw)}); "
                       f"held-out {args.test} n={len(te_raw)}: is_card ECE "
                       f"{before['is_card']['ece']:.4f}->{after['is_card']['ece']:.4f}, group ECE "
                       f"{before['group']['ece']:.4f}->{after['group']['ece']:.4f}"),
    }
    if args.out:
        json.dump(out, open(args.out, "w"), indent=1)
        print(f"\nwrote {args.out}")
    else:
        print("\n" + json.dumps(out, indent=1))


if __name__ == "__main__":
    main()
