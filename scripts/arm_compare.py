#!/usr/bin/env python3
"""arm_compare.py - compare decision arms on the SAME states: latency, agreement, probability and
confidence moves, per field.

usage: arm_compare.py REFERENCE.json ARM.json [ARM2.json ...]

"Accuracy" here means agreement with the reference arm - there is no gold label in these probes, so it
measures how much a path MOVES the decision, not whether it is right.  That is the number that decides
whether an alternative readout can reuse the calibration fitted on the reference.

Reported per field:
  agree   the argmax is the same as the reference (accuracy proxy: 1 - flip rate)
  max|dp| the largest absolute move in any reported probability
  d_conf  the move in the calibrated confidence (the number callers threshold on)
"""
import json
import sys


def probs(ans):
    if not isinstance(ans, dict):
        return []
    p = ans.get("probabilities")
    if isinstance(p, list):
        return [float(x) for x in p]
    return []


def decision(ans):
    """The arm's answer as a hashable decision: the chosen label (any type)."""
    if not isinstance(ans, dict):
        return None
    for k in ("choice", "noul", "score"):
        if k in ans:
            v = ans[k]
            return round(float(v), 3) if isinstance(v, (int, float)) else v
    return None


def main():
    ref = json.load(open(sys.argv[1]))
    arms = [json.load(open(p)) for p in sys.argv[2:]]
    ref_cases = {c["fields"]: c for c in ref["cases"]}
    print(f"reference: {ref['label']}")
    for arm in arms:
        print(f"\n=== {arm['label']} ===")
        print(f"{'fields':>6} {'ref ms':>8} {'arm ms':>8} {'x':>6}  {'field':>6} {'agree':>5} {'max|dp|':>8} {'d_conf':>7}")
        for c in arm["cases"]:
            r = ref_cases.get(c["fields"])
            if not r:
                continue
            rm, am = r.get("ms_median"), c.get("ms_median")
            speed = (rm / am) if (rm and am) else 0.0
            ra, aa = r.get("answers") or {}, c.get("answers") or {}
            for i, qid in enumerate(c["qids"]):
                a, b = ra.get(qid), aa.get(qid)
                dp = 0.0
                pa, pb = probs(a), probs(b)
                if pa and pb and len(pa) == len(pb):
                    dp = max(abs(x - y) for x, y in zip(pa, pb))
                elif isinstance(a, dict) and isinstance(b, dict) and "noul" in a and "noul" in b:
                    dp = abs(float(a["noul"]) - float(b["noul"]))
                dc = 0.0
                if isinstance(a, dict) and isinstance(b, dict) and a.get("confidence") is not None \
                        and b.get("confidence") is not None:
                    dc = abs(float(a["confidence"]) - float(b["confidence"]))
                agree = "yes" if decision(a) == decision(b) else "NO"
                if i == 0:
                    print(f"{c['fields']:>6} {rm:>8} {am:>8} {speed:>5.2f}x  {i + 1:>6} {agree:>5} {dp:>8.4f} {dc:>7.4f}")
                else:
                    print(f"{'':>6} {'':>8} {'':>8} {'':>6}  {i + 1:>6} {agree:>5} {dp:>8.4f} {dc:>7.4f}")


if __name__ == "__main__":
    main()
