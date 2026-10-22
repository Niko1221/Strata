#!/usr/bin/env python3
"""rowreadout_probe.py - M3.1 live probe for the multi-field readout (one prompt, N rows).

Runs the SAME state with 1, 2 and 3 decision fields through /v1/systemone and records, per
repetition: wall ms, usage.prompt_tokens, usage.readout_tokens and the full answers payload.

Two runs of this script (one with the per-question path, one with STRATA_JEV_ROWREADOUT=1)
are compared by rowreadout_compare.py: field 1 is expected to be identical (same row, same
prompt), later fields may move (they see the earlier turns) and that move is the number the
accuracy/Brier A/B has to justify.

usage:
  rowreadout_probe.py --base URL --key KEY --out FILE [--reps 3] [--label per-question]

The base URL is llama-swap's upstream path for the preset, e.g.
  http://10.89.0.57:8080/upstream/p-jevport-vision-3090
"""
import argparse
import json
import time
import urllib.error
import urllib.request

STATE = {"engine": "hot", "load": 0.9, "queue": 12}

FIELDS = {
    "throttle": {"type": "noul", "instructions": "Throttle the engine?"},
    "route": {"type": "choice", "instructions": "Which route?",
              "criteria": {"bridge": None, "tunnel": None, "ferry": None}},
    "urgency": {"type": "score", "instructions": "How urgent is this?",
                "criteria": ["low", "normal", "high", "critical"]},
}
ORDER = ["throttle", "route", "urgency"]


def post(base, key, body, timeout=900):
    req = urllib.request.Request(base + "/v1/systemone", data=json.dumps(body).encode(),
                                 headers={"Content-Type": "application/json",
                                          "Authorization": "Bearer " + key})
    t0 = time.monotonic()
    try:
        with urllib.request.urlopen(req, timeout=timeout) as r:
            payload = json.loads(r.read())
            code = r.status
    except urllib.error.HTTPError as e:
        payload = {"error_body": e.read().decode(errors="replace")[:400]}
        code = e.code
    return code, payload, (time.monotonic() - t0) * 1000.0


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--base", required=True)
    ap.add_argument("--key", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--label", default="run")
    ap.add_argument("--reps", type=int, default=3)
    ap.add_argument("--unique", action="store_true",
                    help="change the state on every repetition, so no two requests share a prompt "
                         "(without it, reps 2..N are cache hits of rep 1 and every number is optimistic)")
    args = ap.parse_args()

    out = {"label": args.label, "base": args.base, "cases": []}
    for n in (1, 2, 3):
        qids = ORDER[:n]
        runs = []
        for i in range(args.reps + 1):                       # first one is the load/warmup
            state = dict(STATE)
            if args.unique:
                state["queue"] = 12 + i                       # a different prompt every repetition
            body = {"state": state, "questions": {q: FIELDS[q] for q in qids}}
            code, payload, ms = post(args.base, args.key, body)
            runs.append({"code": code, "ms": round(ms, 1), "payload": payload})
            print(f"[{args.label}] fields={n} rep={i} code={code} ms={ms:.0f} "
                  f"usage={payload.get('usage') if isinstance(payload, dict) else None}")
        warm = runs[1:]                                      # drop the warmup
        ok = [r for r in warm if r["code"] == 200]
        ms = sorted(r["ms"] for r in ok)
        out["cases"].append({
            "fields": n,
            "qids": qids,
            "code": warm[-1]["code"],
            "answers": ok[-1]["payload"].get("answers") if ok else None,
            "usage": ok[-1]["payload"].get("usage") if ok else None,
            "ms": ms,
            "ms_median": ms[len(ms) // 2] if ms else None,
            "runs": [{"code": r["code"], "ms": r["ms"]} for r in warm],
        })
        print(f"[{args.label}] fields={n} median={out['cases'][-1]['ms_median']} "
              f"prompt_tokens={(out['cases'][-1]['usage'] or {}).get('prompt_tokens')}")

    with open(args.out, "w") as f:
        json.dump(out, f, indent=1)
    print("wrote", args.out)


if __name__ == "__main__":
    main()
