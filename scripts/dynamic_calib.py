#!/usr/bin/env python3
"""dynamic_calib.py - recalibrate the jev decision boundary from LIVE traffic, without labels.

The idea
--------
A static fit tells you the boundary that was right for last month's data. When the mix of
requests shifts, the boundary is wrong again -- and waiting for a fresh labelled set means
being wrong for however long the labelling takes. But you do not need labels on the traffic
to fix the boundary. You need:

  1. a labelled DEV set (once), which gives the confusion matrix  C[pred][y];
  2. the LIVE marginal of predictions, which the server already logs to
     `calib/data/log-<mode>.jsonl` (one row per answered question, with `raw_p`);
  3. the assumption the base rate is stable enough to balance against.

Then BBSE (black-box shift estimation) solves  C . pi = q  for the traffic class prior `pi`,
and you place the threshold so the predicted positive rate matches it. No traffic labels.

Measured on 500 gold-labelled banking77 items (2026-11-01, one GPU):
    shipped static fit (wrong task)   acc 0.664
    identity (no calibration)         acc 0.764
    BBSE prior-matched, NO labels     acc 0.816   <- matches the oracle prior
    labelled offline refit            acc 0.816   <- same number, but needs 250 labels
Streaming 50-item batches with shrinkage + EMA: dynamic 0.828 vs static 0.764.
Per-batch BBSE is noisy (mean 0.488 vs true 0.500, sd 0.108 on n=50) -- shrink and smooth,
or the boundary jitters.

Usage
-----
  # what would the boundary be, from the log only?
  scripts/dynamic_calib.py estimate --log /cfg/.../calib/data/log-L.jsonl \
      --dev-eval /opt/data/eval/arms/head-raw.json

  # write an updated calib file (bounded step, hot-reloadable via POST /admin/calib)
  scripts/dynamic_calib.py update --log ... --dev-eval ... \
      --calib /cfg/.../calib-L.json --out /cfg/.../calib-L.json --stamp 2026-11-01T22

Deliberately NOT an in-process control loop: this is a scheduled job that writes a file the
existing reload path picks up. A serving process that silently re-tunes its own boundary on
live traffic with no human-visible record is a runaway waiting to happen.
"""
from __future__ import annotations

import argparse
import json
import math
import sys
import time
from pathlib import Path

EPS = 1e-9


def logit(p: float) -> float:
    p = min(max(p, EPS), 1 - EPS)
    return math.log(p / (1 - p))


def sig(z: float) -> float:
    if z >= 0:
        return 1.0 / (1.0 + math.exp(-min(z, 700.0)))
    e = math.exp(max(z, -700.0))
    return e / (1.0 + e)


def read_rows(log: Path, qtype: str, window: int | None) -> list[float]:
    """-> the yes-probability of every answered question of `qtype`, oldest first."""
    keep: list[float] = []
    for line in log.read_text().splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            r = json.loads(line)
        except ValueError:
            continue
        if r.get("type") != qtype:
            continue
        pr = r.get("raw_p") or []
        if len(pr) < 2:
            continue
        keep.append(float(pr[1]))
    return keep[-window:] if window else keep


def confusion(dev_eval: Path) -> tuple[list[list[float]], int]:
    """C[pred][y] = P(pred | y) on a labelled eval arm file (rows with p_yes + gold)."""
    arm = json.load(open(dev_eval))
    rows = [r for r in arm.get("rows", []) if r.get("p_yes") is not None]
    C = [[0.0, 0.0], [0.0, 0.0]]
    n = 0
    for y in (0, 1):
        idx = [r for r in rows if r["gold"]["is_card"] == y]
        if not idx:
            raise SystemExit(f"dev-eval has no gold.is_card=={y} rows; a confusion matrix needs both")
        for r in idx:
            C[1 if r["p_yes"] > 0.5 else 0][y] += 1.0 / len(idx)
        n += len(idx)
    return C, n


def bbse(C, probs: list[float]) -> tuple[float | None, list[float]]:
    """Solve C.pi = q for the positive rate. -> (pi_yes, q)"""
    q = [0.0, 0.0]
    for p in probs:
        q[1 if p > 0.5 else 0] += 1.0 / len(probs)
    det = C[0][0] * C[1][1] - C[0][1] * C[1][0]
    if abs(det) < 1e-9:
        return None, q
    pi1 = (-C[1][0] * q[0] + C[0][0] * q[1]) / det
    return pi1, q


def threshold_for_rate(probs: list[float], rate: float) -> float:
    v = sorted(probs)
    k = int(round((1.0 - rate) * (len(v) - 1)))
    return v[max(0, min(len(v) - 1, k))]


def cmd_estimate(a) -> int:
    probs = read_rows(Path(a.log), a.type, a.window)
    if not probs:
        print(f"REFUSING: no {a.type} rows in {a.log}", file=sys.stderr)
        return 2
    print(f"log: {a.log}  {a.type} rows used: {len(probs)}")
    if len(probs) < a.min_n:
        print(f"  WARNING: n={len(probs)} < --min-n {a.min_n}; estimate will be noisy", file=sys.stderr)

    if a.dev_eval:
        C, ndev = confusion(Path(a.dev_eval))
        pi_raw, q = bbse(C, probs)
        print(f"  dev confusion from {a.dev_eval} (n={ndev})")
        print(f"  live prediction marginal: yes={q[1]:.3f}")
        if pi_raw is None:
            print("  REFUSING: the confusion matrix is singular (the readout never "
                  "separates the classes at 0.5); BBSE cannot invert it", file=sys.stderr)
            return 2
        est = pi_raw
        src = "BBSE from live marginal"
    else:
        est = a.prior_target
        src = "--prior-target (no dev set)"
        print(f"  no --dev-eval: assuming the base rate is {est:.3f}")

    lam = len(probs) / (len(probs) + a.n0)
    est = lam * est + (1 - lam) * a.prior_target
    if a.ema is not None:
        est = (1 - a.ema) * est + a.ema * a.prior_target
    est = min(max(est, a.clip_lo), a.clip_hi)
    print(f"  estimate ({src}): shrunk to {est:.3f}  [lambda={lam:.2f}, n0={a.n0}]")

    if abs(est - a.prior_target) > a.alarm:
        print(f"  ALARM: estimate is {abs(est-a.prior_target):.3f} away from the target prior "
              f"(> {a.alarm}); this looks like a distribution shift or broken logging, not drift. "
              f"Refusing to move the boundary.", file=sys.stderr)
        return 3

    t = threshold_for_rate(probs, est)
    c_new = -a.slope * logit(t)
    print(f"  threshold for that rate: raw p > {t:.4f}")
    print(f"  a_noul={a.slope} -> c_noul={c_new:+.4f}")
    if a.calib:
        cur = json.loads(Path(a.calib).read_text())
        c_old = float(cur.get("c_noul", 0.0))
        print(f"  current file c_noul={c_old:+.4f}  step={c_new-c_old:+.4f}"
              f"  (max allowed {a.max_step})")
    return 0


def cmd_update(a) -> int:
    cur = json.loads(Path(a.calib).read_text())
    a.slope = float(cur.get("a_noul", 1.0))
    rc = cmd_estimate(a)
    if rc != 0:
        return rc
    probs = read_rows(Path(a.log), a.type, a.window)
    if a.dev_eval:
        C, _ = confusion(Path(a.dev_eval))
        pi_raw, _ = bbse(C, probs)
    else:
        pi_raw = a.prior_target
    lam = len(probs) / (len(probs) + a.n0)
    est = min(max(lam * pi_raw + (1 - lam) * a.prior_target, a.clip_lo), a.clip_hi)
    t = threshold_for_rate(probs, est)
    c_new = -a.slope * logit(t)

    c_old = float(cur.get("c_noul", 0.0))
    step = c_new - c_old
    if abs(step) > a.max_step:
        c_new = c_old + math.copysign(a.max_step, step)
        print(f"  BOUNDED: step {step:+.4f} -> {c_new-c_old:+.4f} (max_step {a.max_step})")

    cur["c_noul"] = round(c_new, 6)
    cur["provenance"] = (
        f"{cur.get('provenance','')} | dynamic {a.stamp}: n={len(probs)} est_prior={est:.3f} "
        f"thr={t:.4f} c {c_old:+.4f}->{c_new:+.4f}").strip(" |")
    cur.setdefault("dynamic", {})[a.stamp] = {
        "n": len(probs), "est_prior_yes": round(est, 4), "threshold": round(t, 5),
        "c_prev": round(c_old, 6), "c_new": round(c_new, 6), "source": a.dev_eval or "prior-target"}
    out = Path(a.out or a.calib)
    out.write_text(json.dumps(cur, indent=2) + "\n")
    print(f"\nwrote {out}")
    print(f"hot-reload: curl -X POST -H 'Authorization: Bearer $STRATA_JEV_ADMIN_KEY' .../admin/calib")
    return 0


def main() -> int:
    ap = argparse.ArgumentParser(description=(__doc__ or "").split("\n")[0])
    sub = ap.add_subparsers(dest="cmd", required=True)
    for name in ("estimate", "update"):
        s = sub.add_parser(name)
        s.add_argument("--log", required=True, help="calib/data/log-<mode>.jsonl")
        s.add_argument("--type", default="noul", help="question type to calibrate (default noul)")
        s.add_argument("--window", type=int, default=2000, help="use the last N rows (default 2000)")
        s.add_argument("--dev-eval", help="labelled eval arm JSON (rows with p_yes + gold.is_card)")
        s.add_argument("--prior-target", type=float, default=0.5,
                       help="the base rate you expect; also the shrink target (default 0.5)")
        s.add_argument("--n0", type=float, default=200.0, help="shrinkage strength (default 200)")
        s.add_argument("--ema", type=float, default=None,
                       help="blend toward --prior-target with this weight (0-1); use for a "
                            "slow-moving estimate across successive updates")
        s.add_argument("--min-n", type=int, default=200, help="warn below this many rows")
        s.add_argument("--clip-lo", type=float, default=0.02)
        s.add_argument("--clip-hi", type=float, default=0.98)
        s.add_argument("--alarm", type=float, default=0.45,
                       help="refuse to move if the estimate is this far from the target prior")
        s.add_argument("--slope", type=float, default=1.0, help="a_noul (read from the file on update)")
        s.add_argument("--max-step", type=float, default=0.5, help="max |delta c_noul| per update")
        s.add_argument("--calib", help="existing calib file (for the current c and the slope)")
        s.add_argument("--out", help="write here (default: overwrite --calib)")
        s.add_argument("--stamp", default=time.strftime("%Y-%m-%dT%H:%M"))
    args = ap.parse_args()
    if args.cmd == "update" and not args.calib:
        ap.error("update needs --calib")
    return cmd_estimate(args) if args.cmd == "estimate" else cmd_update(args)


if __name__ == "__main__":
    sys.exit(main())
