#!/usr/bin/env python3
"""Check a documented greedy near tie, with full target-logit evidence.

This is a fixture-specific investigation, not a general exactness exemption.
Run with the integration fixture, --index 72, on the hardware in the review report.
Exact output passes. A difference passes only at the final tested token, with the
same two winning IDs, a target margin <= 0.05, RMS logit drift <= 0.1 and maximum
drift <= 1.0. All earlier tokens must match. Other mismatches fail and retain logs.
"""
from __future__ import annotations

import argparse
import array
import heapq
import json
import math
import os
from pathlib import Path

from dflash_sampling import Engine, replace, validate_residency


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--config", type=Path, required=True)
    ap.add_argument("--engine", type=Path, required=True)
    ap.add_argument("--fixtures", type=Path, required=True)
    ap.add_argument("--case", default="integration")
    ap.add_argument("--index", type=int, default=72, help="zero-based output index to probe")
    ap.add_argument("--output", type=Path, required=True)
    ap.add_argument("--expert-cache", type=int, default=3500, help="fixed expert slots; must fit both arms")
    a = ap.parse_args()
    cfg = json.loads(a.config.read_text())
    args = list(cfg["args"])
    if "--dflash" not in args or "--mtp" in args or a.index < 0:
        ap.error("DFlash configuration without MTP and a nonnegative index required")
    fixture = next(f for f in json.loads(a.fixtures.read_text()) if f["name"] == a.case)
    ids = fixture["tokens"]
    if not ids or len(ids) + a.index + 9 > 4096:
        ap.error("prompt and generated tokens must fit the test context")
    for key, value in (("--expert-cache", str(a.expert_cache)), ("--adapt-every", "0"),
                       ("--adapt-swaps", "0"), ("--pcie-frac", "0.20"), ("--prefill", "256"),
                       ("--short-read", "0"), ("--max-context", "4096"),
                       ("--kv-resident", "4096"), ("--dflash-window", "4096"),
                       ("--spec", "8"), ("--dflash-block", "5"), ("--spec-min-p", "0"),
                       ("--suffix-draft", "0"), ("--lookup-chain", "0"), ("--prompt-cache", "0"),
                       ("--rope-scaling", None), ("--rope-scale", "1")):
        replace(args, key, value)
    out = a.output.resolve()
    out.mkdir(parents=True, exist_ok=True)
    cwd = Path(cfg.get("cwd", Path(__file__).resolve().parents[1]))
    records, logits, infos = {}, {}, {}
    position = len(ids) - 1 + a.index
    for arm in ("target", "dflash"):
        arm_args = args.copy()
        if arm == "target":
            for key in ("--dflash", "--dflash-block", "--dflash-window", "--dflash-vocab",
                        "--dflash-head", "--dflash-mask-token"):
                replace(arm_args, key, None)
        path = out / f"{arm}.f32"
        path.unlink(missing_ok=True)
        env = dict(os.environ, STRATA_IQ_MT_MIN="1", STRATA_PREFILL_CPU_SHARE="0",
                   STRATA_SPEC_PROB="0", STRATA_SPEC_COUPLED="0", STRATA_SPEC_GUMBEL="0",
                   STRATA_DF_TARGET_PROBE=str(position), STRATA_DF_TARGET_PROBE_FILE=str(path))
        engine = Engine(a.engine.resolve(), arm_args, cwd, env, out / f"{arm}-engine.log")
        try:
            infos[arm] = engine.info
            (out / "engine-info.json").write_text(json.dumps(infos, indent=2))
            validate_residency(engine.info, a.expert_cache, infos.get("target"))
            records[arm] = engine.request(ids, "temperature=0 seed=7", a.index + 1)
        finally:
            engine.close()
        if "error" in records[arm] or not path.exists():
            raise RuntimeError(records[arm])
        logits[arm] = array.array("f")
        logits[arm].frombytes(path.read_bytes())
    x, y = logits["target"], logits["dflash"]
    assert len(x) == len(y) and len(x) >= 2 and all(math.isfinite(v) for v in (*x, *y))
    top = {k: heapq.nlargest(2, range(len(v)), key=v.__getitem__) for k, v in logits.items()}
    first = next((i for i, (u, v) in enumerate(zip(records["target"]["tokens"], records["dflash"]["tokens"])) if u != v), None)
    delta = [u - v for u, v in zip(x, y)]
    margin = x[top["target"][0]] - x[top["target"][1]]
    rms = math.sqrt(sum(d * d for d in delta) / len(delta))
    maximum = max(map(abs, delta))
    result = dict(position=position, first_mismatch=first, target_margin=margin,
                  max_abs=maximum, rms=rms, records=records,
                  top2={k: [(i, float(logits[k][i])) for i in v] for k, v in top.items()})
    (out / "results.json").write_text(json.dumps(result, indent=2))
    assert all(len(r["tokens"]) == a.index + 1 for r in records.values())
    assert first is None or (first == a.index and set(top["target"]) == set(top["dflash"]) and
                            margin <= 0.05 and rms <= 0.1 and maximum <= 1.0 and
                            all(records[k]["tokens"][-1] == top[k][0] for k in records)), result
    print("PASS:", "exact tokens" if first is None else "documented fixture near tie", flush=True)


if __name__ == "__main__":
    main()
