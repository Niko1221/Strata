#!/usr/bin/env python3
"""Alternate target-only, MTP and DFlash math runs on the same target and residency.

Fixtures are JSON objects with name/text/tokens, such as math-fixtures.json in
bench/results/2026-10-17-dflash-review. No model downloads or configuration writes.
Outputs include token IDs, resolved arguments and engine logs; startup/prefill
are excluded from decode throughput. Stop other GPU work before running.
"""
from __future__ import annotations

import argparse
import json
import math
import os
from pathlib import Path

from dflash_sampling import Engine, replace, validate_residency


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--config", type=Path, required=True)
    ap.add_argument("--engine", type=Path, required=True)
    ap.add_argument("--mtp", type=Path, required=True)
    ap.add_argument("--fixtures", type=Path, required=True)
    ap.add_argument("--output", type=Path, required=True)
    ap.add_argument("--rounds", type=int, default=5)
    ap.add_argument("--count", type=int, default=300)
    ap.add_argument("--block", type=int, default=5)
    ap.add_argument("--expert-cache", "--expert-cache-mib", dest="expert_cache", type=int, default=2000,
                    help="fixed expert slots (legacy --expert-cache-mib alias also counts slots)")
    ap.add_argument("--auto-expert-cache", action="store_true", help="include the drafter's effect on expert capacity")
    ap.add_argument("--spec-min-p", type=float, default=0.0)
    a = ap.parse_args()
    if (a.rounds < 1 or a.count < 1 or a.expert_cache < 1 or not 1 <= a.block <= 7 or
            not math.isfinite(a.spec_min_p) or not 0 <= a.spec_min_p <= 1):
        ap.error("positive rounds/count, block in 1..7 and spec-min-p in 0..1 required")
    cfg = json.loads(a.config.read_text())
    args = list(cfg["args"])
    if "--dflash" not in args or "--mtp" in args:
        ap.error("configuration must select DFlash without MTP")
    fixtures = json.loads(a.fixtures.read_text())
    if not fixtures or any(not f["tokens"] or len(f["tokens"]) + a.count + 8 > 4096 for f in fixtures):
        ap.error("fixtures must fit the 4096-token test context")
    for key, value in (("--expert-cache", "auto" if a.auto_expert_cache else str(a.expert_cache)), ("--adapt-every", "0"),
                       ("--adapt-swaps", "0"), ("--pcie-frac", "0.20"), ("--prefill", "256"),
                       ("--short-read", "0"), ("--max-context", "4096"),
                       ("--kv-resident", "4096"), ("--dflash-window", "4096"),
                       ("--spec", "8"), ("--dflash-block", str(a.block)), ("--spec-min-p", str(a.spec_min_p)),
                       ("--suffix-draft", "0"), ("--lookup-chain", "0"), ("--prompt-cache", "0"),
                       ("--rope-scaling", None), ("--rope-scale", "1")):
        replace(args, key, value)
    env = dict(os.environ, STRATA_IQ_MT_MIN="1", STRATA_PREFILL_CPU_SHARE="0",
               STRATA_SPEC_PROB="0", STRATA_SPEC_COUPLED="0", STRATA_SPEC_GUMBEL="0")
    out = a.output.resolve()
    out.mkdir(parents=True, exist_ok=True)
    cwd = Path(cfg.get("cwd", Path(__file__).resolve().parents[1]))
    results = []
    for rnd in range(a.rounds):
        for arm in (["target", "mtp", "dflash"] if rnd % 2 == 0 else ["dflash", "mtp", "target"]):
            arm_args = args.copy()
            if arm != "dflash":
                for key in ("--dflash", "--dflash-block", "--dflash-window", "--dflash-vocab", "--dflash-head", "--dflash-mask-token"):
                    replace(arm_args, key, None)
            if arm == "mtp":
                replace(arm_args, "--spec", "4")
                replace(arm_args, "--mtp", str(a.mtp.resolve()))
            engine = Engine(a.engine.resolve(), arm_args, cwd, env, out / f"math-{rnd}-{arm}-engine.log")
            try:
                if not a.auto_expert_cache:
                    validate_residency(engine.info, a.expert_cache)
                engine.request(fixtures[0]["tokens"], "temperature=0 seed=7", 16)
                for f in fixtures:
                    result = engine.request(f["tokens"], "temperature=0 seed=7", a.count)
                    if "error" in result:
                        raise RuntimeError(result["error"])
                    results.append(dict(round=rnd, arm=arm, prompt=f["name"], args=arm_args,
                                        info=engine.info, **result))
                    (out / "math-results.json").write_text(json.dumps(results, indent=2))
                    print(rnd, arm, f["name"], result["done"], flush=True)
            finally:
                engine.close()


if __name__ == "__main__":
    main()
