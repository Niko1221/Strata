#!/usr/bin/env python3
"""rowreadout_compare.py - compare two rowreadout_probe.py runs (per-question vs row-readout).

usage: rowreadout_compare.py A.json B.json

What it reports, per field count:
  - latency median side by side and the speed-up
  - usage.prompt_tokens side by side (the state must be prefilled once, not N times)
  - per-question answer delta: for field 1 a NON-ZERO delta is a FAILURE (same prompt, same
    row - the consolidated path must reproduce M3.0 exactly); for fields 2..N the delta is the
    context shift the accuracy A/B has to justify, so it is reported, not asserted.

Exit code 1 only when field 1 moved (that is a correctness regression, not a measurement).
"""
import json
import sys


def flat(ans):
    """A comparable view of one answer: the decision plus the probability vector."""
    if not isinstance(ans, dict):
        return None
    out = {}
    for k in ("type", "noul", "choice", "score", "confidence", "calib"):
        if k in ans:
            out[k] = ans[k]
    pr = ans.get("probabilities")
    if isinstance(pr, dict):
        out["probabilities"] = {k: round(float(v), 6) for k, v in pr.items()}
    elif isinstance(pr, list):
        out["probabilities"] = [round(float(v), 6) for v in pr]
    return out


def delta(a, b):
    """Max abs difference over numeric fields, plus the keys that differ at all."""
    fa, fb = flat(a) or {}, flat(b) or {}
    worst, keys = 0.0, []
    for k in sorted(set(fa) | set(fb)):
        va, vb = fa.get(k), fb.get(k)
        if isinstance(va, dict) and isinstance(vb, dict) and set(va) == set(vb):
            d = max((abs(va[kk] - vb[kk]) for kk in va), default=0.0)
        elif isinstance(va, list) and isinstance(vb, list) and len(va) == len(vb):
            d = max((abs(x - y) for x, y in zip(va, vb)), default=0.0)
        elif isinstance(va, (int, float)) and isinstance(vb, (int, float)) \
                and not isinstance(va, bool) and not isinstance(vb, bool):
            d = abs(float(va) - float(vb))
        elif va == vb:
            continue
        else:
            keys.append(k)
            continue
        if d > 0:
            keys.append(k)
        worst = max(worst, d)
    return worst, keys


def main():
    pa, pb = sys.argv[1], sys.argv[2]
    a = json.load(open(pa))
    b = json.load(open(pb))
    print(f"A = {a['label']}  ({pa})")
    print(f"B = {b['label']}  ({pb})")
    print()
    print(f"{'fields':>6} {'A ms':>8} {'B ms':>8} {'speed-up':>9} {'A tok':>7} {'B tok':>7}")
    regress = []
    for ca, cb in zip(a["cases"], b["cases"]):
        ma, mb = ca["ms_median"], cb["ms_median"]
        sp = (ma / mb) if (ma and mb) else 0.0
        ta = (ca["usage"] or {}).get("prompt_tokens")
        tb = (cb["usage"] or {}).get("prompt_tokens")
        print(f"{ca['fields']:>6} {ma:>8} {mb:>8} {sp:>8.2f}x {ta:>7} {tb:>7}")
        ans_a, ans_b = ca.get("answers") or {}, cb.get("answers") or {}
        for i, qid in enumerate(ca["qids"]):
            w, keys = delta(ans_a.get(qid), ans_b.get(qid))
            tag = "FIELD-1" if i == 0 else f"field-{i + 1}"
            print(f"    {tag:>8} {qid:<10} max|delta|={w:.6f}  differing={keys or '-'}")
            if i == 0 and w > 0:
                regress.append((ca["fields"], qid, w, keys))
    print()
    if regress:
        print("FAIL: field 1 moved between the two paths - the consolidated row is not M3.0's row")
        for f, q, w, k in regress:
            print(f"  fields={f} {q} max|delta|={w} {k}")
        return 1
    print("OK: field 1 is identical on both paths (same prompt, same row)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
