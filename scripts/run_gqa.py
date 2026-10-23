#!/usr/bin/env python3
"""run_gqa.py - run the GQA binary set through /v1/systemone (image direct, noul) and fit a calibrator.

  run_gqa.py --base URL --key KEY --set /opt/data/eval/gqa/gqa-600.jsonl --out gqa-C.json --label C

Splits the set in half: dev to fit the Platt (a_noul, c_noul), test to score.  Reports accuracy,
Brier, NLL and 15-bin ECE for identity vs the fit, plus how many items the fit flips (a Platt with a
non-zero offset MOVES the yes/no threshold, so accuracy can change - that is the whole point of
testing a noul field).

Save rows so the cross-domain question ("does a fit made on GQA still help POPE, and vice versa?")
can be answered offline without another GPU run.
"""
import argparse
import importlib.util
import json
import math
import urllib.error
import urllib.request
from pathlib import Path

spec = importlib.util.spec_from_file_location("suite", "/opt/data/eval/jevport-suite.py")
S = importlib.util.module_from_spec(spec)
spec.loader.exec_module(S)


def post(base, key, body, timeout=300):
    req = urllib.request.Request(base + "/v1/systemone", data=json.dumps(body).encode(),
                                 headers={"Content-Type": "application/json",
                                          "Authorization": "Bearer " + key})
    with urllib.request.urlopen(req, timeout=timeout) as r:
        return r.status, json.loads(r.read())


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--base", required=True)
    ap.add_argument("--key", required=True)
    ap.add_argument("--set", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--label", required=True)
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--image-mode", default="direct")
    ap.add_argument("--score-only", action="store_true", help="re-score the rows already on disk")
    args = ap.parse_args()

    items = [json.loads(l) for l in open(args.set) if l.strip()]
    if args.limit:
        items = items[: args.limit]

    rows, done = [], set()
    part = args.out + ".rows.jsonl"
    if Path(part).exists():                                  # resume: skip items already answered
        rows = [json.loads(l) for l in open(part) if l.strip()]
        done = {r["id"] for r in rows}
        print(f"resuming: {len(rows)} rows already on disk")
    if args.score_only:
        items = []
    for n, it in enumerate(items, 1):
        if it["id"] in done:
            continue
        body = {"image_b64": it["image_b64"], "image_mode": args.image_mode,
                "questions": {it["qid"]: it["q"]}}
        try:
            code, resp = post(args.base, args.key, body)
        except urllib.error.HTTPError as e:
            print(f"  ! {it['id']} http={e.code} {e.read().decode(errors='replace')[:120]}")
            continue
        a = resp["answers"][it["qid"]]
        p = float(a["noul"])
        usage = resp.get("usage") or {}
        row = {"id": it["id"], "gold": it["gold"], "p": p,
                     "pred": 1 if p >= 0.5 else 0, "calib": a.get("calib"),
                     "image_sha": it.get("image_sha"),
               "prompt_tokens": usage.get("prompt_tokens")}
        rows.append(row)
        with open(args.out + ".rows.jsonl", "a") as fh:       # incremental: a crash must not cost the run
            fh.write(json.dumps(row) + "\n")
        if n % 25 == 0:
            print(f"  {n}/{len(items)}", flush=True)

    if len(rows) < 20:
        print(f"only {len(rows)} usable rows - aborting")
        return 1

    half = len(rows) // 2
    dev, test = rows[:half], rows[half:]
    dl = [math.log(max(r["p"], 1e-9) / max(1 - r["p"], 1e-9)) for r in dev]
    cs = [r["gold"] for r in dev]
    fi = S.fit_platt_ex(dl, cs)

    def platt(p, a, c):
        q = min(max(p, 1e-9), 1 - 1e-9)
        z = a * math.log(q / (1 - q)) + c
        return 1.0 / (1.0 + math.exp(-max(-700.0, min(700.0, z))))

    test_rows = [([1 - r["p"], r["p"]], r["gold"]) for r in test]
    before = S.metrics(test_rows, None)                       # the suite's own acc/NLL/ECE15/flips
    after = S.metrics(test_rows, ("platt", fi["a"], fi["c"]))
    before["brier"] = sum((r["p"] - r["gold"]) ** 2 for r in test) / len(test)
    after["brier"] = sum((platt(r["p"], fi["a"], fi["c"]) - r["gold"]) ** 2 for r in test) / len(test)
    flips = sum(1 for r in test
                if (1 if r["p"] >= 0.5 else 0) != (1 if platt(r["p"], fi["a"], fi["c"]) >= 0.5 else 0))
    out = {"label": args.label, "set": args.set, "image_mode": args.image_mode,
           "n_dev": len(dev), "n_test": len(test), "fit": fi,
           "identity": before, "calibrated": after,
           "flips_on_test": flips,
           "served_calib": sorted({r["calib"] for r in rows}),
           "prompt_tokens_mean": round(sum(r["prompt_tokens"] for r in rows if r["prompt_tokens"])
                                       / max(1, sum(1 for r in rows if r["prompt_tokens"])), 1),
           "rows": rows}
    json.dump(out, open(args.out, "w"), indent=1)
    print(f"\n[{args.label}] n_dev={len(dev)} n_test={len(test)} served_calib={out['served_calib']}")
    print(f"  fit  a={fi['a']:.4f} c={fi['c']:.4f} flat={fi['flat']} boundary={fi['boundary']} "
          f"improvement={fi['improvement']:.3f}")
    for name, m in (("identity", before), ("calibrated", after)):
        print(f"  {name:<11} acc={m['acc']:.4f}  brier={m['brier']:.4f}  ece15={m['ece15']:.4f}  nll={m['nll']:.4f}")
    print(f"  items the fit flips on test: {flips}/{len(test)}  "
          f"prompt_tokens_mean={out['prompt_tokens_mean']}")
    print(f"  -> {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
