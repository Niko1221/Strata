#!/usr/bin/env python3
"""ab_arms_mf.py - paired A/B of two decision arms over the SAME labelled items.

  ab_arms_mf.py --a /tmp/mf-C-cal-live.json --b /tmp/mf-B-cal-live.json \
                --set /opt/data/eval/banking/mf-200.jsonl [--labels A,B]

Both arms must have been run over the same set (matched by item id).  Reports, per arm:
is_card accuracy / Brier / ECE (top-label, 15 bins), group accuracy / ECE, mean confidence,
both-fields-right, and mean latency when the response carried timing.  Between arms: the
paired discordant counts and exact McNemar p for is_card, for group, and for both-right.

Why paired: two arms on the same items share item difficulty, so an unpaired accuracy
comparison throws away the power the pairing buys (and hides which items flipped).
"""
import argparse
import json
import math


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


def mcnemar(pairs):
    """pairs: [(ok_a, ok_b)] -> (n_a_only, n_b_only, two-sided exact p)."""
    n10 = sum(1 for x, y in pairs if x and not y)
    n01 = sum(1 for x, y in pairs if y and not x)
    m = n10 + n01
    if m == 0:
        return n10, n01, 1.0
    k = min(n10, n01)
    tail = sum(math.comb(m, i) for i in range(k + 1)) / (2 ** m)
    return n10, n01, min(1.0, 2 * tail)


def metrics(rows, labels):
    card, grp = [], []
    acc_c = acc_g = both = 0
    brier = 0.0
    conf_c = conf_g = 0.0
    lat = []
    for r in rows:
        q = float(r["p_yes"])
        pred_c = 1 if q >= 0.5 else 0
        ok_c = pred_c == r["gold"]["is_card"]
        acc_c += ok_c
        conf_c += q if pred_c else 1 - q
        brier += (q - r["gold"]["is_card"]) ** 2
        card.append((q if pred_c else 1 - q, ok_c))
        gp = [float(x) for x in r["group_probs"]]
        j = max(range(len(gp)), key=lambda i: gp[i])
        pred_g = labels[j] if j < len(labels) else None
        ok_g = pred_g == r["gold"]["group"]
        acc_g += ok_g
        both += ok_c and ok_g
        conf_g += gp[j]
        grp.append((gp[j], ok_g))
        u = r.get("usage") or {}
        t = (u.get("timing_ms") or {}).get("total") if isinstance(u.get("timing_ms"), dict) else u.get("total_ms")
        if isinstance(t, (int, float)):
            lat.append(float(t))
    n = len(rows)
    out = {
        "n": n,
        "acc_card": acc_c / n, "brier": brier / n, "ece_card": ece(card),
        "conf_card": conf_c / n,
        "acc_group": acc_g / n, "ece_group": ece(grp), "conf_group": conf_g / n,
        "both": both / n,
    }
    if lat:
        lat.sort()
        out["lat_mean"] = sum(lat) / len(lat)
        out["lat_p50"] = lat[len(lat) // 2]
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--a", required=True)
    ap.add_argument("--b", required=True)
    ap.add_argument("--set", required=True)
    ap.add_argument("--labels", default="A,B")
    args = ap.parse_args()

    names = args.labels.split(",")
    ra = {r["id"]: r for r in json.load(open(args.a))["rows"]}
    rb = {r["id"]: r for r in json.load(open(args.b))["rows"]}
    lab = {}
    for line in open(args.set):
        line = line.strip()
        if line:
            it = json.loads(line)
            lab[it["id"]] = list(it["questions"]["group"]["criteria"])
    ids = [i for i in ra if i in rb and ra[i].get("p_yes") is not None and rb[i].get("p_yes") is not None]
    rows_a = [ra[i] for i in ids]
    rows_b = [rb[i] for i in ids]
    labels = lab[ids[0]]

    ma, mb = metrics(rows_a, labels), metrics(rows_b, labels)

    print(f"paired on {len(ids)} items, same set, each arm with its OWN fitted calibrator\n")
    hdr = f"{'metric':<16}{names[0]:>10}{names[1]:>10}"
    print(hdr)
    print("-" * len(hdr))
    for k, fmt in (("is_card acc", "acc_card"), ("is_card Brier", "brier"), ("is_card ECE", "ece_card"),
                   ("is_card conf", "conf_card"), ("group acc", "acc_group"), ("group ECE", "ece_group"),
                   ("group conf", "conf_group"), ("both right", "both")):
        print(f"{k:<16}{ma[fmt]:>10.4f}{mb[fmt]:>10.4f}")
    if "lat_mean" in ma and "lat_mean" in mb:
        print(f"{'latency mean':<16}{ma['lat_mean']:>10.0f}{mb['lat_mean']:>10.0f}  ms")
        print(f"{'latency p50':<16}{ma['lat_p50']:>10.0f}{mb['lat_p50']:>10.0f}  ms")

    def ok_c(r, _=None):
        q = float(r["p_yes"])
        return (1 if q >= 0.5 else 0) == r["gold"]["is_card"]

    def ok_g(r, _=None):
        gp = [float(x) for x in r["group_probs"]]
        j = max(range(len(gp)), key=lambda i: gp[i])
        return (labels[j] if j < len(labels) else None) == r["gold"]["group"]

    def ok_both(r, _=None):
        return ok_c(r) and ok_g(r)

    for name, fn in (("is_card", ok_c), ("group", ok_g), ("both", ok_both)):
        pairs = [(fn(x), fn(y)) for x, y in zip(rows_a, rows_b)]
        n10, n01, p = mcnemar(pairs)
        print(f"\nMcNemar {name}: {names[0]} only {n10}, {names[1]} only {n01}, p={p:.3f}")


if __name__ == "__main__":
    main()
