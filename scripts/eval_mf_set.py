#!/usr/bin/env python3
"""eval_mf_set.py - run and compare LABELLED multi-field accuracy for decision arms.

  eval_mf_set.py run --base URL --key KEY --set mf-200.jsonl --out arm.json --label C-pin
  eval_mf_set.py compare arm-C.json arm-B.json ...

Per item, per field, the gold answer comes from the set (build_mf_set.py), never from a model, so
"accuracy" here is real accuracy - not agreement with another arm.

Reported per arm: accuracy and Brier for is_card (noul), accuracy for group (choice), the fraction
where BOTH fields are right, and the mean calibrated confidence.  Between arms: the paired accuracy
difference with a McNemar exact-ish test (discordant pairs only) and the Brier difference.
"""
import argparse
import json
import math
import sys
import urllib.error
import urllib.request


def post(base, key, body, timeout=300):
    req = urllib.request.Request(base + "/v1/systemone", data=json.dumps(body).encode(),
                                 headers={"Content-Type": "application/json",
                                          "Authorization": "Bearer " + key})
    try:
        with urllib.request.urlopen(req, timeout=timeout) as r:
            return r.status, json.loads(r.read())
    except urllib.error.HTTPError as e:
        return e.code, {"error": e.read().decode(errors="replace")[:300]}


def run(args):
    items = [json.loads(l) for l in open(args.set)]
    if args.limit:
        items = items[: args.limit]
    rows = []
    for n, it in enumerate(items, 1):
        body = {"state": it["state"], "questions": it["questions"]}
        code, body_out = post(args.base, args.key, body)
        if code != 200:
            print(f"  ! {it['id']} http={code}", file=sys.stderr)
            rows.append({"id": it["id"], "code": code, "gold": it["gold"]})
            continue
        ans = body_out.get("answers", {})
        card = ans.get("is_card", {})
        grp = ans.get("group", {})
        p_yes = card.get("noul")
        labels = list(it["questions"]["group"]["criteria"])
        pred_group = None
        if isinstance(grp.get("choice"), int) and 0 <= grp["choice"] < len(labels):
            pred_group = labels[grp["choice"]]
        rows.append({
            "id": it["id"],
            "gold": it["gold"],
            "p_yes": p_yes,
            "pred_card": None if p_yes is None else (1 if p_yes >= 0.5 else 0),
            "pred_group": pred_group,
            "group_probs": grp.get("probabilities"),
            "group_conf": grp.get("confidence"),
            "card_margin": card.get("margin"),
            "usage": body_out.get("usage"),
        })
        if n % 25 == 0:
            print(f"  {n}/{len(items)}", flush=True)

    ok = [r for r in rows if r.get("pred_card") is not None]
    n = len(ok)
    if not n:
        print("no usable rows", file=sys.stderr)
        return 1
    acc_card = sum(1 for r in ok if r["pred_card"] == r["gold"]["is_card"]) / n
    brier = sum((r["p_yes"] - r["gold"]["is_card"]) ** 2 for r in ok) / n
    acc_grp = sum(1 for r in ok if r["pred_group"] == r["gold"]["group"]) / n
    both = sum(1 for r in ok if r["pred_card"] == r["gold"]["is_card"]
               and r["pred_group"] == r["gold"]["group"]) / n
    confs = [r["group_conf"] for r in ok if isinstance(r["group_conf"], (int, float))]
    out = {"label": args.label, "set": args.set, "n": n,
           "acc_is_card": acc_card, "brier_is_card": brier, "acc_group": acc_grp, "both_right": both,
           "mean_group_conf": (sum(confs) / len(confs)) if confs else None,
           "rows": rows}
    json.dump(out, open(args.out, "w"), indent=1)
    print(f"[{args.label}] n={n} is_card acc={acc_card:.3f} brier={brier:.4f} "
          f"group acc={acc_grp:.3f} both={both:.3f} mean_conf={out['mean_group_conf']:.3f}"
          if out["mean_group_conf"] is not None else f"[{args.label}] n={n} acc_card={acc_card:.3f}")
    return 0


def mcnemar(b, c):
    """Discordant pairs -> two-sided exact binomial p (no scipy)."""
    n01 = sum(1 for x, y in zip(b, c) if not x and y)
    n10 = sum(1 for x, y in zip(b, c) if x and not y)
    m = n01 + n10
    if m == 0:
        return 0, 0, 1.0
    k = min(n01, n10)
    tail = sum(math.comb(m, i) for i in range(0, k + 1)) / (2 ** m)
    return n01, n10, min(1.0, 2 * tail)


def compare(args):
    arms = [json.load(open(p)) for p in args.files]
    ref = arms[0]
    byid = {r["id"]: r for r in ref["rows"]}
    print(f"reference: {ref['label']}  (n={ref['n']})")
    print(f"{'arm':<12} {'is_card':>8} {'group':>7} {'both':>6} {'brier':>8} {'conf':>6}  {'discord':>10} {'p':>7}")
    print(f"{ref['label']:<12} {ref['acc_is_card']:>8.3f} {ref['acc_group']:>7.3f} {ref['both_right']:>6.3f} "
          f"{ref['brier_is_card']:>8.4f} {(ref['mean_group_conf'] or 0):>6.3f}")
    for arm in arms[1:]:
        rows = {r["id"]: r for r in arm["rows"]}
        ids = [i for i in byid if i in rows and byid[i].get("pred_card") is not None
               and rows[i].get("pred_card") is not None]
        a_ok = [byid[i]["pred_card"] == byid[i]["gold"]["is_card"] for i in ids]
        b_ok = [rows[i]["pred_card"] == rows[i]["gold"]["is_card"] for i in ids]
        n01, n10, p = mcnemar(a_ok, b_ok)
        print(f"{arm['label']:<12} {arm['acc_is_card']:>8.3f} {arm['acc_group']:>7.3f} {arm['both_right']:>6.3f} "
              f"{arm['brier_is_card']:>8.4f} {(arm['mean_group_conf'] or 0):>6.3f}  "
              f"{'+%d/-%d' % (n01, n10):>10} {p:>7.3f}")
    print("\ndiscord: pairs where the arm fixed (+) / broke (-) an is_card decision vs the reference; p = McNemar exact")


def main():
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    r = sub.add_parser("run")
    r.add_argument("--base", required=True)
    r.add_argument("--key", required=True)
    r.add_argument("--set", required=True)
    r.add_argument("--out", required=True)
    r.add_argument("--label", required=True)
    r.add_argument("--limit", type=int, default=0)
    c = sub.add_parser("compare")
    c.add_argument("files", nargs="+")
    a = ap.parse_args()
    if a.cmd == "run":
        return run(a)
    return compare(a)


if __name__ == "__main__":
    sys.exit(main())
