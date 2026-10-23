#!/usr/bin/env python3
"""fit_tuple_from_log.py - fit one calibration tuple from a jevport-suite.py item log.

The suite already fits and prints, but its numbers live in a markdown table.  This turns the same
log into an INSTALLABLE calib-<LETTER>.json for one tuple, with the dev/test split read from the
log's block layout (per qtype: fit_n dev rows, then the rest is test).

  fit_tuple_from_log.py --log suite-real-summary/real-item-log.jsonl \
      --blocks "noul:120,choice:40,score:40" --letter S --mode image-summary \
      --out calib-S.json [--quant iq3_xxs --alias qfn-iq3_xxs]

Only meaningful when the log was taken with the calibrators REMOVED (identity-served): otherwise
the fit compounds the transform that produced the probabilities (see the pitfall in the skill).

Refuses to write a file that `scripts/check_calib.py` would reject (unknown keys / degenerate T).
"""
import argparse
import importlib.util
import json
import math
import subprocess
import sys
from pathlib import Path

SUITE = "/opt/data/eval/jevport-suite.py"
spec = importlib.util.spec_from_file_location("suite", SUITE)
S = importlib.util.module_from_spec(spec)
spec.loader.exec_module(S)


def evaluate(rows, cal):
    """rows: [(pr, gold)] -> acc / ece15 / nll (top-label protocol, same as the suite)."""
    return S.metrics(rows, cal)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--log", required=True)
    ap.add_argument("--blocks", required=True, help='"noul:120,choice:40,score:40" (dev+test per qtype)')
    ap.add_argument("--letter", required=True, choices=list("LVD S".replace(" ", "")))
    ap.add_argument("--mode", required=True, help="tuple field 6, e.g. image-summary / image-direct / text")
    ap.add_argument("--out", required=True)
    ap.add_argument("--alias", default="qfn-iq3_xxs")
    ap.add_argument("--quant", default="jev-port")
    args = ap.parse_args()

    rows = [json.loads(l) for l in open(args.log) if l.strip()]
    spec_blocks = []
    for part in args.blocks.split(","):
        k, n = part.split(":")
        spec_blocks.append((k.strip(), int(n)))

    doc = {"tuple": f"{args.alias}|{args.quant}|none|v1|{args.letter}|{args.mode}"}
    fits, report = {}, []
    off = 0
    for qtype, n in spec_blocks:
        block = rows[off:off + n]
        off += n
        if len(block) < n:
            sys.exit(f"{qtype}: block short ({len(block)}/{n}) - wrong --blocks?")
        half = n // 2
        dev = [(r["pr"], r["gold"]) for r in block[:half]]
        test = [(r["pr"], r["gold"]) for r in block[half:]]
        served = sorted({r.get("calib") for r in block})
        if any(s not in (None, "identity") for s in served):
            sys.exit(f"{qtype}: log was served by calibrator {served} - the fit would compound it. "
                     f"Rerun with the calibrators removed (jevport-arm.sh <arm> --calib identity).")

        if qtype == "noul":
            dl = [math.log(max(p[1], 1e-9) / max(1 - p[1], 1e-9)) for p, _ in dev]
            cs = [g for _, g in dev]
            fi = S.fit_platt_ex(dl, cs)
            cal = ("platt", fi["a"], fi["c"])
            doc["a_noul"], doc["c_noul"] = fi["a"], fi["c"]
        else:
            fi = S.fit_temp_ex(dev)
            cal = ("T", fi["T"])
            doc["T_score" if qtype == "score" else "T_choice"] = fi["T"]

        before, after = evaluate(test, None), evaluate(test, cal)
        report.append((qtype, len(test), before, after, fi))

    doc.setdefault("a_noul", 1.0)
    doc.setdefault("c_noul", 0.0)
    doc.setdefault("T_choice", 1.0)
    doc.setdefault("T_score", 1.0)
    flat = [f"{r[0]}:{'flat' if r[4]['flat'] else 'fit'}" for r in report]
    doc["provenance"] = (f"fit {Path(args.log).name} ({args.mode}, identity-served, dev = first half of "
                         f"each block), {', '.join(flat)}")

    print(f"tuple {doc['tuple']}")
    for qtype, n, before, after, fi in report:
        print(f"  {qtype:<7} n_test={n:<4} acc {before['acc']:.3f}->{after['acc']:.3f}  "
              f"ECE {before['ece15']:.4f}->{after['ece15']:.4f}  NLL {before['nll']:.4f}->{after['nll']:.4f}"
              f"  {'flat(identity)' if fi['flat'] else ('BOUNDARY' if fi['boundary'] else '')}")
    print("  " + json.dumps({k: doc[k] for k in ("a_noul", "c_noul", "T_choice", "T_score")}))

    Path(args.out).write_text(json.dumps(doc, indent=1) + "\n")
    r = subprocess.run([sys.executable, "/opt/data/builds/jev-port/scripts/check_calib.py",
                        str(Path(args.out).parent)], capture_output=True, text=True)
    print(f"\nwrote {args.out}; check_calib says:\n{r.stdout.strip() or r.stderr.strip()}")
    return 0 if r.returncode == 0 else 1


if __name__ == "__main__":
    sys.exit(main())
