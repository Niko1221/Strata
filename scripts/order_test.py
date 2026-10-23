#!/usr/bin/env python3
"""order_test.py - does the FIELD ORDER change the answers? (leonardomurillo's ask on #3643)

He found that answers can change when the same fields are asked in a different order, and that his
endpoint now fixes the order internally.  For a path that reads several fields from ONE prompt (our B
arm, `jevpos` row capture) this is the sharpest risk there is: the fields share a context, so moving
one changes what every later field conditions on.  This test asks the same items twice - fields in
file order, then reversed - and reports, per field:

  agreement      same label under both orders (the number that matters for callers)
  flips          items that changed label, split by direction
  McNemar        paired test on correctness-vs-gold between the two orders
  mean |dp|      mean absolute probability change (noul) - how much the readout moved
  prompt_tokens  per request, from usage (his harness ask)

  order_test.py --base URL --key KEY --set mf-200.jsonl --out order-C.json --label C

Runs with whatever calibrator the server has loaded; use identity (`jevport-arm.sh <arm> --calib
identity`) so the comparison isolates order, not threshold.
"""
import argparse
import json
import math
import time
import urllib.error
import urllib.request


def post(base, key, body, timeout=300):
    req = urllib.request.Request(base + "/v1/systemone", data=json.dumps(body).encode(),
                                 headers={"Content-Type": "application/json",
                                          "Authorization": "Bearer " + key})
    t0 = time.time()
    try:
        with urllib.request.urlopen(req, timeout=timeout) as r:
            return r.status, json.loads(r.read()), (time.time() - t0) * 1000.0
    except urllib.error.HTTPError as e:
        return e.code, {"error": e.read().decode(errors="replace")[:300]}, (time.time() - t0) * 1000.0


def field_answer(ans, qid, labels):
    a = ans.get("answers", {}).get(qid, {})
    if "noul" in a:
        p = float(a["noul"])
        return {"kind": "noul", "p": p, "label": int(p >= 0.5), "conf": p if p >= 0.5 else 1 - p,
                "calib": a.get("calib")}
    if "choice" in a:
        idx = a["choice"] if isinstance(a["choice"], int) else None
        pr = a.get("probabilities") or []
        name = labels[idx] if (idx is not None and idx < len(labels)) else None
        return {"kind": "choice", "label": name, "conf": a.get("confidence"),
                "prob": pr[idx] if (idx is not None and idx < len(pr)) else None, "calib": a.get("calib")}
    return {"kind": "?", "label": None, "conf": None, "calib": a.get("calib")}


def mcnemar(a_ok, b_ok):
    n10 = sum(1 for x, y in zip(a_ok, b_ok) if x and not y)
    n01 = sum(1 for x, y in zip(a_ok, b_ok) if y and not x)
    m = n10 + n01
    if m == 0:
        return n10, n01, 1.0
    k = min(n10, n01)
    return n10, n01, min(1.0, 2 * sum(math.comb(m, i) for i in range(k + 1)) / (2 ** m))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--base", required=True)
    ap.add_argument("--key", required=True)
    ap.add_argument("--set", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--label", required=True)
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--alternate", type=int, default=1,
                    help="1 (default): half the items run fwd-first and half rev-first, so a cold first "
                         "prefill does not get blamed on the order; 0: always fwd then rev")
    args = ap.parse_args()

    items = [json.loads(l) for l in open(args.set) if l.strip()]
    if args.limit:
        items = items[: args.limit]
    rows = []
    for n, it in enumerate(items, 1):
        qs = it["questions"]
        order = list(qs)                             # file order
        rev = list(reversed(order))
        labels = list(qs["group"]["criteria"]) if "group" in qs else []

        out = {}
        # The FIRST request per item pays the cold prefill; the second reuses the state.  Alternating
        # which order goes first balances that, and `pos` is recorded so the effect stays visible
        # instead of being read as "the reverse order is 3x faster".
        seqs = [("fwd", order), ("rev", rev)]
        if args.alternate and n % 2 == 0:
            seqs = [("rev", rev), ("fwd", order)]
        for pos, (tag, seq) in enumerate(seqs, 1):
            body = {"state": it["state"], "questions": {k: qs[k] for k in seq}}
            code, resp, ms = post(args.base, args.key, body)
            if code != 200:
                print(f"  ! {it['id']} {tag} http={code}")
                out[tag] = None
                continue
            usage = resp.get("usage") or {}
            out[tag] = {"ms": round(ms), "pos": pos, "prompt_tokens": usage.get("prompt_tokens"),
                        "fields": {q: field_answer(resp, q, labels) for q in order}}
        rows.append({"id": it["id"], "gold": it["gold"], **out})
        if n % 25 == 0:
            print(f"  {n}/{len(items)}", flush=True)

    ok = [r for r in rows if r.get("fwd") and r.get("rev")]
    report = {"label": args.label, "n": len(ok), "per_field": {}}
    for q in ("is_card", "group"):
        agree = flips_fr = flips_rf = 0
        dsum = dn = 0
        corr = {"fwd": [], "rev": []}
        for r in ok:
            a = r["fwd"]["fields"][q]
            b = r["rev"]["fields"][q]
            if a["label"] == b["label"]:
                agree += 1
            elif q == "is_card":
                if a["label"] == 1:
                    flips_fr += 1
                else:
                    flips_rf += 1
            if q == "is_card":
                dsum += abs(a["p"] - b["p"])
                dn += 1
            gold_ok = {"is_card": r["gold"]["is_card"], "group": r["gold"]["group"]}[q]
            corr["fwd"].append((int(a["label"] >= 0.5) if q == "is_card" else a["label"]) == gold_ok)
            corr["rev"].append((int(b["label"] >= 0.5) if q == "is_card" else b["label"]) == gold_ok)
        n10, n01, p = mcnemar(corr["fwd"], corr["rev"])
        report["per_field"][q] = {
            "agree": agree, "agree_rate": round(agree / len(ok), 4),
            "flips": {"fwd_yes_rev_no": flips_fr, "fwd_no_rev_yes": flips_rf} if q == "is_card" else "n/a",
            "mean_abs_dp": round(dsum / dn, 4) if dn else None,
            "acc_fwd": round(sum(corr["fwd"]) / len(ok), 4),
            "acc_rev": round(sum(corr["rev"]) / len(ok), 4),
            "mcnemar_fwd_only": n10, "mcnemar_rev_only": n01, "p": round(p, 4),
        }
    for tag in ("fwd", "rev"):
        ms = [r[tag]["ms"] for r in ok]
        pt = [r[tag]["prompt_tokens"] for r in ok if r[tag].get("prompt_tokens")]
        report[f"{tag}_latency_ms_mean"] = round(sum(ms) / len(ms), 1)
        report[f"{tag}_prompt_tokens_mean"] = round(sum(pt) / len(pt), 1) if pt else None
    # latency by REQUEST POSITION: 1 = cold prefill for this state, 2 = state already read
    for pos in (1, 2):
        ms = [r[t]["ms"] for r in ok for t in ("fwd", "rev") if r[t]["pos"] == pos]
        report[f"pos{pos}_latency_ms_mean"] = round(sum(ms) / len(ms), 1) if ms else None
        report[f"pos{pos}_n"] = len(ms)

    json.dump({"report": report, "rows": rows}, open(args.out, "w"), indent=1)
    pf = report["per_field"]
    print(f"\n[{args.label}] n={report['n']}")
    for q, m in pf.items():
        print(f"  {q:<8} agree {m['agree_rate']:.3f} ({m['agree']}/{report['n']})  "
              f"acc fwd {m['acc_fwd']:.3f} rev {m['acc_rev']:.3f}  "
              f"{'dp %.3f  ' % m['mean_abs_dp'] if m['mean_abs_dp'] is not None else ''}"
              f"flips {m['flips']}  McNemar fwd-only {m['mcnemar_fwd_only']} rev-only "
              f"{m['mcnemar_rev_only']} p={m['p']}")
    print(f"  latency fwd {report['fwd_latency_ms_mean']} ms  rev {report['rev_latency_ms_mean']} ms  "
          f"prompt_tokens fwd {report['fwd_prompt_tokens_mean']} rev {report['rev_prompt_tokens_mean']}")
    print(f"  latency by position: 1st request {report['pos1_latency_ms_mean']} ms (n={report['pos1_n']}), "
          f"2nd {report['pos2_latency_ms_mean']} ms (n={report['pos2_n']}) - the gap is the cold prefill, "
          f"not the field order")
    print(f"  -> {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
