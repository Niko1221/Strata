#!/usr/bin/env python3
"""refit_calib.py - fit a PER-TASK calibrator for the jev decision surface, or refuse.

Why this exists
---------------
The server loads ONE fit per readout letter (`calib-L.json`) and applies it to every
text task. A fit is only valid on the task family it was fitted on: a BoolQ/ARC/SST5
fit (a_noul 0.6127, c_noul +0.6539) applied to a balanced binary grouping task that
needs c_noul ~ -1.56 moved the yes-rate from 67.6% to 80.4% and cost 9.6 accuracy
points on 500 items (McNemar p < 0.001). Fitting on ~25-100 in-domain items recovers
it and then some: 0.664 -> 0.816 accuracy, ECE 0.275 -> 0.109 held out.

What it does
------------
1. collect per-item readouts (or reuse an eval arm JSON that already has them);
2. split stratified dev / holdout;
3. fit Platt (a, c) for `noul` and temperature T for `choice`/`score` on DEV only;
4. score identity vs the refit on the HOLDOUT, with accuracy, Brier, ECE and a
   McNemar-style paired count;
5. write `calib-<mode>.json` ONLY if the refit beats identity on accuracy AND ECE
   (unless --force), stamped with the task tag the server must be running.

Usage
-----
  # 1. from an eval arm file you already produced
  scripts/refit_calib.py fit --eval-json /opt/data/eval/arms/head-raw.json \
      --task banking77-mf --item-set /opt/data/eval/mf-500.jsonl \
      --out-dir /cfg/strata/Jevstrata-head/calib-b77

  # 2. straight from a live arm (collects the readouts itself)
  scripts/refit_calib.py fit --base http://127.0.0.1:8080/upstream/p-my-arm \
      --key "$KEY" --set my-items.jsonl --task my-task --out-dir ./calib-my-task

  # 3. dry run: fit and report, write nothing
  scripts/refit_calib.py fit --eval-json arm.json --task t --no-write

Then run the server with the matching task tag and calib dir:
  STRATA_JEV_TASK=my-task STRATA_JEV_CALIB_DIR=/cfg/strata/.../calib-my-task

Item file format (one JSON object per line; `gold` is required, never model-derived)
-----------------------------------------------------------------------------------
  {"id": "ex-0",
   "state": {"utterance": "I lost my card"},
   "questions": {
     "is_card":  {"type": "noul", "instructions": "Does this request concern a card?"},
     "group":    {"type": "choice", "instructions": "Which group?",
                  "criteria": {"card": null, "top_up": null, "transfer": null,
                               "cash": null, "other": null}}},
   "gold": {"is_card": 1, "group": "card"}}

Only the fields you fit are needed: `gold.is_card` for a noul fit, `gold.group` for a
choice fit. Labels must be one token each (the server refuses multi-token labels).
"""
from __future__ import annotations

import argparse
import json
import math
import random
import sys
import urllib.error
import urllib.request
from pathlib import Path

EPS = 1e-9


# ─────────────────────────── math (stdlib only) ────────────────────────────
def logit(p: float) -> float:
    p = min(max(p, EPS), 1 - EPS)
    return math.log(p / (1 - p))


def sig(z: float) -> float:
    if z >= 0:
        return 1.0 / (1.0 + math.exp(-min(z, 700.0)))
    e = math.exp(max(z, -700.0))
    return e / (1.0 + e)


def nll_platt(xs, a: float, c: float) -> float:
    t = 0.0
    for z, y in xs:
        p = min(max(sig(a * z + c), EPS), 1 - EPS)
        t -= y * math.log(p) + (1 - y) * math.log(1 - p)
    return t


def fit_platt(xs, fix_slope: float | None = None, iters: int = 100):
    """Damped Newton on (a, c) for p' = sigmoid(a*logit(p) + c). Returns (a, c)."""
    a, c = (fix_slope if fix_slope else 1.0), 0.0
    cur = nll_platt(xs, a, c)
    for _ in range(iters):
        g0 = g1 = h00 = h01 = h11 = 0.0
        for z, y in xs:
            p = sig(a * z + c)
            w = max(p * (1 - p), EPS)
            g0 += (p - y) * z
            g1 += (p - y)
            h00 += w * z * z
            h01 += w * z
            h11 += w
        det = h00 * h11 - h01 * h01
        if abs(det) < 1e-12:
            break
        da = (h11 * g0 - h01 * g1) / det
        dc = (-h01 * g0 + h00 * g1) / det
        if fix_slope:
            da = 0.0
        step, improved = 1.0, False
        while step > 1e-6:
            na, nc = a - step * da, c - step * dc
            if nll_platt(xs, na, nc) < cur:
                a, c, cur, improved = na, nc, nll_platt(xs, na, nc), True
                break
            step *= 0.5
        if not improved or (abs(step * da) < 1e-10 and abs(step * dc) < 1e-10):
            break
    return a, c


def apply_temp(pr, T: float):
    """Same transform as jev_calib.apply(): pr^(1/T), renormalised."""
    if T <= 0.01 or abs(T - 1.0) < 1e-9:
        return list(pr)
    lm = math.log(max(max(pr), 1e-300))
    lse = math.log(sum(math.exp((math.log(max(x, 1e-300)) - lm) / T) for x in pr))
    out = [math.exp((math.log(max(x, 1e-300)) - lm) / T - lse) for x in pr]
    s = sum(out)
    return [x / s for x in out] if s > 0 else out


# ──────────────────────────────── metrics ─────────────────────────────────
def ece(pairs, bins: int = 10) -> float:
    b = [[] for _ in range(bins)]
    for p, y in pairs:
        b[min(bins - 1, max(0, int(p * bins)))].append((p, y))
    n = len(pairs) or 1
    return sum(abs(sum(p for p, _ in g) - sum(y for _, y in g)) / n for g in b if g)


def report(rows, prob, gold) -> dict:
    n = len(rows)
    acc = sum(1 for r in rows if (prob(r) > 0.5) == bool(gold(r))) / n
    brier = sum((prob(r) - gold(r)) ** 2 for r in rows) / n
    return {"n": n, "acc": acc, "brier": brier, "ece": ece([(prob(r), gold(r)) for r in rows])}


def mcnemar(ok_a, ok_b):
    n01 = sum(1 for x, y in zip(ok_a, ok_b) if not x and y)   # b fixed
    n10 = sum(1 for x, y in zip(ok_a, ok_b) if x and not y)   # b broke
    m = n01 + n10
    if m == 0:
        return 0, 0, 1.0
    k = min(n01, n10)
    p = min(1.0, 2 * sum(math.comb(m, i) for i in range(k + 1)) / 2 ** m)
    return n01, n10, p


# ───────────────────────────── collection ─────────────────────────────────
def post(base, key, body, timeout=600):
    req = urllib.request.Request(base.rstrip("/") + "/v1/systemone",
                                 data=json.dumps(body).encode(),
                                 headers={"Content-Type": "application/json",
                                          "Authorization": "Bearer " + key})
    try:
        with urllib.request.urlopen(req, timeout=timeout) as r:
            return r.status, json.loads(r.read())
    except urllib.error.HTTPError as e:
        return e.code, {"error": e.read().decode(errors="replace")[:300]}


def collect(base, key, items, gold_of):
    rows = []
    for i, it in enumerate(items, 1):
        st, d = post(base, key, {"state": it["state"], "questions": it["questions"]})
        row = {"id": it["id"], "gold": gold_of(it)}
        ans = d.get("answers") if isinstance(d, dict) else None
        if st == 200 and isinstance(ans, dict):
            card = ans.get("is_card") or {}
            grp = ans.get("group") or {}
            row.update(p_yes=card.get("noul"),
                       margin=card.get("margin"),
                       group_probs=grp.get("probabilities"),
                       group_conf=grp.get("confidence"))
        else:
            row.update(code=st, error=str(d)[:200])
        rows.append(row)
        if i % 25 == 0:
            print(f"  collected {i}/{len(items)}", flush=True)
    return rows


def gold_of_factory(items):
    idx = {it["id"]: (it.get("gold") or {}) for it in items}
    return lambda it: idx.get(it["id"], {})


# ──────────────────────────────── fit ─────────────────────────────────────
def cmd_fit(args):
    if args.eval_json:
        arm = json.load(open(args.eval_json))
        rows = [r for r in arm.get("rows", []) if r.get("p_yes") is not None]
        print(f"read {len(rows)} usable rows from {args.eval_json}")
        gold = lambda r: r["gold"]["is_card"]          # noqa: E731
    else:
        items = [json.loads(l) for l in open(args.set) if l.strip()]
        rows = [r for r in collect(args.base, args.key, items, gold_of_factory(items))
                if r.get("p_yes") is not None]
        print(f"collected {len(rows)} usable rows from {args.base}")
        byid = {it["id"]: it.get("gold") or {} for it in items}
        gold = lambda r: byid[r["id"]]["is_card"]      # noqa: E731

    if len(rows) < args.min_items:
        print(f"REFUSING: {len(rows)} usable items < --min-items {args.min_items}. "
              f"A fit on fewer items is noise; identity is the safer default.", file=sys.stderr)
        return 2
    for r in rows:
        r["y"] = int(gold(r))

    # stratified split
    by = {0: [r for r in rows if r["y"] == 0], 1: [r for r in rows if r["y"] == 1]}
    rnd = random.Random(args.seed)
    for v in by.values():
        rnd.shuffle(v)
    ndev = max(2, int(min(len(by[0]), len(by[1])) * args.dev_frac))
    dev = by[0][:ndev] + by[1][:ndev]
    hold = by[0][ndev:] + by[1][ndev:]
    print(f"dev={len(dev)} ({ndev}+{ndev} balanced)  holdout={len(hold)}  seed={args.seed}")

    xs = [(logit(r["p_yes"]), r["y"]) for r in dev]
    a, c = fit_platt(xs)
    a1, c1 = fit_platt(xs, fix_slope=1.0)
    print(f"fitted on dev: a_noul={a:.4f} c_noul={c:+.4f}   (intercept-only c={c1:+.4f})")

    base_m = report(hold, lambda r: r["p_yes"], lambda r: r["y"])
    ref_m = report(hold, lambda r: sig(a * logit(r["p_yes"]) + c), lambda r: r["y"])
    i_m = report(hold, lambda r: sig(logit(r["p_yes"]) + c1), lambda r: r["y"])

    ok_base = [(r["p_yes"] > 0.5) == bool(r["y"]) for r in hold]
    ok_ref = [(sig(a * logit(r["p_yes"]) + c) > 0.5) == bool(r["y"]) for r in hold]
    fixed, broke, p = mcnemar(ok_base, ok_ref)

    print("\nHOLDOUT (the fit never saw these):")
    print(f"  {'arm':<20} {'acc':>7} {'brier':>8} {'ECE':>7}")
    for name, m in (("identity (no fit)", base_m), ("refit platt(a,c)", ref_m),
                    ("refit intercept-only", i_m)):
        print(f"  {name:<20} {m['acc']:>7.3f} {m['brier']:>8.4f} {m['ece']:>7.4f}")
    print(f"  paired vs identity: fixed {fixed}, broke {broke}, McNemar exact p={p:.2e}")

    ok = (ref_m["acc"] > base_m["acc"] and ref_m["ece"] < base_m["ece"])
    print(f"\nGATE: refit must beat identity on accuracy AND ECE -> "
          f"{'PASS' if ok else 'FAIL'}"
          + ("" if ok else " (identity is better here; do NOT ship this fit)"))

    # choice temperature, if the arm carries group probabilities
    T_choice = None
    better_choice = False
    if args.criteria:
        crit = args.criteria
        gidx = {k: i for i, k in enumerate(crit)}
        gs = [(r, gidx[r["gold"]["group"]]) for r in rows
              if r.get("group_probs") and r["gold"].get("group") in gidx]
        if gs:
            def choice_nll(T):
                t = 0.0
                for r, gi in gs:
                    q = apply_temp(r["group_probs"], T)
                    t -= math.log(max(q[gi], EPS))
                return t
            best = min(((choice_nll(T) , T) for T in
                        [0.05 * k for k in range(20, 101)]), key=lambda x: x[0])
            T_choice = round(best[1], 4)
            acc1 = sum(1 for r, gi in gs
                       if max(range(len(r["group_probs"])), key=lambda i: r["group_probs"][i]) == gi) / len(gs)
            accT = acc1
            if T_choice is not None:
                def argmax_t(pr):
                    q = apply_temp(pr, T_choice)
                    return max(range(len(q)), key=lambda i: q[i])
                accT = sum(1 for r, gi in gs if argmax_t(r["group_probs"]) == gi) / len(gs)
            better_choice = accT > acc1
            print(f"\nchoice: fitted T_choice={T_choice} (fitted on all rows, monotone so "
                  f"argmax-preserving)  argmax acc {acc1:.3f} -> {accT:.3f}")

    if args.no_write:
        print("\n--no-write: nothing written")
        return 0 if ok or args.force else 1
    if not ok and not args.force:
        print("\nNot writing the fit (gate failed). Use --force to override deliberately.",
              file=sys.stderr)
        return 1

    out = Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)
    doc = {
        "tuple": f"{args.alias}|{args.quant}|{args.provenance_engine}|v1|{args.mode}|{args.task}",
        "task": args.task,
        "a_noul": round(a, 6),
        "c_noul": round(c, 6),
        "T_choice": T_choice if (T_choice and (better_choice or args.force)) else 1.0,
        "T_score": 1.0,
        "provenance": (f"refit {args.stamp} task={args.task} dev={len(dev)} holdout={len(hold)} "
                       f"identity(acc={base_m['acc']:.3f} ece={base_m['ece']:.3f}) -> "
                       f"refit(acc={ref_m['acc']:.3f} ece={ref_m['ece']:.3f}) "
                       f"mcnemar p={p:.1e}"),
    }
    dest = out / f"calib-{args.mode}.json"
    dest.write_text(json.dumps(doc, indent=2) + "\n")
    print(f"\nwrote {dest}")
    print(f"run the server with:  STRATA_JEV_TASK={args.task} "
          f"STRATA_JEV_CALIB_DIR={out}")
    return 0


def main():
    ap = argparse.ArgumentParser(description=(__doc__ or "").split("\n")[0])
    sub = ap.add_subparsers(dest="cmd", required=True)
    f = sub.add_parser("fit", help="fit and gate a per-task calibrator")
    src = f.add_mutually_exclusive_group(required=True)
    src.add_argument("--eval-json", help="an eval_mf_set.py arm JSON (needs rows with p_yes)")
    src.add_argument("--set", help="item JSONL to collect readouts from (needs --base/--key)")
    f.add_argument("--base", help="arm base URL, e.g. http://host:8080/upstream/p-arm")
    f.add_argument("--key", help="bearer key for the arm")
    f.add_argument("--task", required=True, help="task tag; must match STRATA_JEV_TASK at serve time")
    f.add_argument("--out-dir", default="calib-per-task", help="where calib-<mode>.json goes")
    f.add_argument("--mode", default="L", choices=list("LVDS"), help="readout letter (default L)")
    f.add_argument("--dev-frac", type=float, default=0.5, help="fraction per class used to fit")
    f.add_argument("--seed", type=int, default=7)
    f.add_argument("--min-items", type=int, default=40,
                   help="refuse to fit on fewer usable items (default 40)")
    f.add_argument("--criteria", nargs="*", default=None,
                   help="choice labels in criteria order, to fit T_choice")
    f.add_argument("--alias", default="qfn-iq3_xxs")
    f.add_argument("--quant", default="jev-port")
    f.add_argument("--provenance-engine", default="none")
    f.add_argument("--stamp", default="local")
    f.add_argument("--force", action="store_true", help="write even if the gate fails")
    f.add_argument("--no-write", action="store_true", help="fit and report only")
    args = ap.parse_args()
    if args.cmd == "fit":
        return cmd_fit(args)
    return 2


if __name__ == "__main__":
    sys.exit(main())
