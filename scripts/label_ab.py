#!/usr/bin/env python3
"""Does the new label alphabet (0-9/a-z) change the readout, and does the
choice calibrator fitted on A-Z still fit?

Run the same choice items under two STRATA_JEV_LABELS settings, then compare.

  label_ab.py run --tag upper --out /opt/data/eval/label-upper.jsonl
  label_ab.py run --tag digit-lower --out /opt/data/eval/label-digit.jsonl
  label_ab.py compare /opt/data/eval/label-upper.jsonl /opt/data/eval/label-digit.jsonl

The first 60 items are the dev split and the last 100 the test split, matching
jevport-suite.py's text protocol so the fitted T is directly comparable to the
shipped T_choice.
"""
import argparse
import importlib.util
import json
import math
import urllib.request

SUITE = "/opt/data/eval/jevport-suite.py"
BASE = "http://10.89.0.57:8080/upstream/p-jevport-vision-3090"
DATA = "/opt/data/llm-swap/strata-data/eval/data"
DEV_N, TEST_N = 60, 100
SHIPPED_T_CHOICE = 1.0557          # calib-L.json T_choice, fitted on A-Z labels


def suite():
    spec = importlib.util.spec_from_file_location("suite", SUITE)
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    return m


def key():
    for line in open("/opt/data/llm-swap/.container-env"):
        if line.startswith("LLM_SWAP_API_KEY="):
            return line.split("=", 1)[1].strip()
    raise SystemExit("no api key")


def apply_temp(pr, T):
    lm = math.log(max(max(pr), 1e-300))
    lse = math.log(sum(math.exp((math.log(max(x, 1e-300)) - lm) / T) for x in pr))
    out = [math.exp((math.log(max(x, 1e-300)) - lm) / T - lse) for x in pr]
    s = sum(out)
    return [x / s for x in out] if s > 0 else out


def run(args):
    S = suite()
    items = S.text_items("choice", DATA, DEV_N + TEST_N)
    rows = []
    for i, it in enumerate(items, 1):
        body = {"state": it["state"], "questions": {it["qid"]: it["q"]}}
        req = urllib.request.Request(
            BASE + "/v1/systemone", data=json.dumps(body).encode(),
            headers={"Content-Type": "application/json", "Authorization": "Bearer " + key()})
        try:
            r = json.load(urllib.request.urlopen(req, timeout=240))
        except Exception as e:
            print(f"  ! {it['qid']} {e}")
            continue
        a = r["answers"][it["qid"]]
        rows.append({"qid": it["qid"], "gold": it["gold"], "tag": args.tag,
                     "choice": a.get("choice"), "probabilities": a.get("probabilities"),
                     "calib": a.get("calib"), "confidence": a.get("confidence"),
                     "prompt_tokens": (r.get("usage") or {}).get("prompt_tokens")})
        if i % 20 == 0:
            print(f"  {i}/{len(items)}", flush=True)
    with open(args.out, "w") as fh:
        for r in rows:
            fh.write(json.dumps(r) + "\n")
    print(f"wrote {args.out} ({len(rows)} rows, calib={sorted({r['calib'] for r in rows})})")


def metrics(rows):
    """rows: [{'pr': [...], 'gold': int}] -> acc, nll, brier, ece15 on the top-label protocol"""
    acc = 0
    nll = brier = 0.0
    ps, gs = [], []
    for r in rows:
        pr, g = r["pr"], r["gold"]
        amax = max(range(len(pr)), key=lambda i: pr[i])
        acc += int(amax == g)
        nll -= math.log(max(pr[g], 1e-12))
        brier += sum((p - (1.0 if i == g else 0.0)) ** 2 for i, p in enumerate(pr))
        ps.append(max(pr[amax], 1e-9))
        gs.append(1 if amax == g else 0)
    n = len(rows)
    bins = [[] for _ in range(15)]
    for p, g in zip(ps, gs):
        bins[min(14, int(p * 15))].append((p, g))
    ece = sum(len(b) / n * abs(sum(p for p, _ in b) / len(b) - sum(g for _, g in b) / len(b))
              for b in bins if b)
    return {"acc": acc / n, "nll": nll / n, "brier": brier / n, "ece15": ece}


def best_T(rows, lo=0.30, hi=4.00, step=0.01):
    """Bounded grid search for the NLL-optimal temperature.

    Bounded on purpose: on a saturated split (acc 1.000, NLL ~0.004) the unbounded fit
    runs to its floor (0.05) and the number is meaningless - T is unidentifiable there.
    """
    best, T = None, lo
    t = lo
    while t <= hi + 1e-9:
        v = metrics([{"pr": apply_temp(r["pr"], t), "gold": r["gold"]} for r in rows])["nll"]
        if best is None or v < best:
            best, T = v, t
        t += step
    return T, best


def compare(a_path, b_path):
    A = {r["qid"]: r for r in map(json.loads, open(a_path))}
    B = {r["qid"]: r for r in map(json.loads, open(b_path))}
    common = sorted(set(A) & set(B))
    print(f"{len(A)} vs {len(B)} rows, {len(common)} matched")
    print(f"served calib: {sorted({r['calib'] for r in A.values()})} / "
          f"{sorted({r['calib'] for r in B.values()})}\n")

    # T is served on both sides, so invert it to compare the raw readout
    T = SHIPPED_T_CHOICE
    for tag, D in (("upper", A), ("digit-lower", B)):
        for q in common:
            D[q]["raw"] = apply_temp(D[q]["probabilities"], 1.0 / T) if D[q]["calib"] == "T_choice_v1" \
                else D[q]["probabilities"]

    agree = sum(1 for q in common if A[q]["choice"] == B[q]["choice"])
    dp = sum(abs(x - y) for q in common for x, y in zip(A[q]["raw"], B[q]["raw"])) / (len(common) * 4)
    same_vec = sum(1 for q in common
                   if max(abs(x - y) for x, y in zip(A[q]["raw"], B[q]["raw"])) < 1e-9)
    print(f"argmax agreement: {agree}/{len(common)} ({agree / len(common):.3f})")
    print(f"mean |dp| per option: {dp:.4f}   identical raw vectors: {same_vec}/{len(common)}")
    print(f"mean prompt_tokens: {sum(r['prompt_tokens'] for r in A.values()) / len(A):.1f} (upper) / "
          f"{sum(r['prompt_tokens'] for r in B.values()) / len(B):.1f} (digit-lower)\n")

    for split, sl in (("dev 60", slice(0, DEV_N)), ("test 100", slice(DEV_N, None))):
        print(f"[{split}]")
        opt = {}
        for tag, D in (("upper", A), ("digit-lower", B)):
            rows = [{"pr": D[q]["raw"], "gold": D[q]["gold"]} for q in common[sl]]
            raw = metrics(rows)
            srv = metrics([{"pr": apply_temp(r["pr"], T), "gold": r["gold"]} for r in rows])
            t_opt, nll_opt = best_T(rows)
            opt[tag] = t_opt
            print(f"  {tag:<12} raw acc={raw['acc']:.3f} nll={raw['nll']:.4f} brier={raw['brier']:.4f} "
                  f"ece={raw['ece15']:.4f}")
            print(f"  {'':<12} served(T={T}) acc={srv['acc']:.3f} nll={srv['nll']:.4f} "
                  f"ece={srv['ece15']:.4f} | best T={t_opt:.2f} (nll {nll_opt:.4f})")
        print()

    gap = abs(opt["digit-lower"] - opt["upper"]) / opt["upper"]
    print(f"best T per alphabet (test): upper {opt['upper']:.2f} vs digit-lower {opt['digit-lower']:.2f} "
          f"-> {gap * 100:.1f}% apart; shipped {SHIPPED_T_CHOICE}")
    print(f"verdict: {'REFIT' if gap > 0.20 else 'no refit needed'} "
          f"(threshold 20% between the two alphabets' own optima)")


def main():
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    r = sub.add_parser("run")
    r.add_argument("--tag", required=True)
    r.add_argument("--out", required=True)
    c = sub.add_parser("compare")
    c.add_argument("a")
    c.add_argument("b")
    args = ap.parse_args()
    if args.cmd == "run":
        run(args)
    else:
        compare(args.a, args.b)


if __name__ == "__main__":
    main()
