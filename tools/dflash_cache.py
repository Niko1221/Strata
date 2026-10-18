#!/usr/bin/env python3
"""GPU regression for DFlash live continuation, checkpoint rewind and RAM parking.

No downloads. Uses an existing DFlash config and one-candidate split windows to
keep target arithmetic consistent between fresh and cached reads. Writes tokens, timings and engine
logs to --output, and checks cached answers against fresh requests on the same
binary with fixed expert residency. Disk sessions remain unsupported.
"""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

from dflash_sampling import Engine, replace


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--config", type=Path, required=True)
    ap.add_argument("--engine", type=Path, required=True)
    ap.add_argument("--output", type=Path, required=True)
    ap.add_argument("--tokens-file", type=Path)
    a = ap.parse_args()
    out = a.output.resolve()
    out.mkdir(parents=True, exist_ok=True)
    cfg = json.loads(a.config.read_text())
    args = list(cfg["args"])
    if "--dflash" not in args or "--mtp" in args:
        ap.error("configuration must select DFlash without MTP")
    ids = ([int(t) for t in a.tokens_file.read_text().replace(",", " ").split()]
           if a.tokens_file else [50, 3815, 2923, 539, 2923, 13, 6558, 281, 2007, 8,
                                  283, 220, 18, 87, 61, 17, 471, 220, 22, 87, 478, 220, 17, 13])
    for key, value in (("--expert-cache", "3500"), ("--adapt-every", "0"),
                       ("--adapt-swaps", "0"), ("--pcie-frac", "0"), ("--prefill", "256"),
                       ("--short-read", "4096"), ("--max-context", "4096"),
                       ("--kv-resident", "4096"), ("--dflash-window", "4096"),
                       ("--spec", "2"), ("--dflash-block", "1"), ("--spec-min-p", "0"),
                       ("--suffix-draft", "0"), ("--lookup-chain", "0"),
                       ("--prompt-cache", "16"), ("--prompt-cache-every", "16"),
                       ("--conversation-cache-mib", "256"), ("--conversation-cache-slots", "4"),
                       ("--conversation-cache-min-tokens", "1"), ("--conversation-cache-park-gain", "0")):
        replace(args, key, value)
    args.append("--spec-split")
    env = dict(os.environ, STRATA_IQ_MT_MIN="1", STRATA_PREFILL_CPU_SHARE="0",
               STRATA_SPEC_PROB="0", STRATA_SPEC_COUPLED="0", STRATA_SPEC_GUMBEL="0")
    cwd = Path(cfg.get("cwd", Path(__file__).resolve().parents[1]))
    engine = Engine(a.engine.resolve(), args, cwd, env, out / "cached-engine.txt")
    records = {}
    prompts = {}
    try:
        def request(label, tokens, sampling="temperature=0 seed=7 pin=16"):
            prompts[label] = (tokens, sampling)
            r = engine.request(tokens, sampling, 32)
            assert "error" not in r, r
            records[label] = r
            print(label, r["done"], flush=True)
            (out / "results.json").write_text(json.dumps(records, indent=2))
            return r["tokens"]
        first = request("initial", ids)
        continuation = ids + first + [13]
        second = request("live_continuation", continuation)
        request("other_conversation", [50, 3815, 220, 17, 10, 17, 28])
        request("parked_continuation", continuation + second + [13])
        request("parked_prompt", ids)
        request("rewind", ids + first[:16] + [13])
        sampled_prompt = ids + first[:16] + [13]
        sampled = request("sampled", sampled_prompt, "temperature=0.7 seed=7 pin=16")
        request("sampled_to_greedy", sampled_prompt + sampled + [13])
    finally:
        engine.close()
    replace(args, "--prompt-cache", "0")
    engine = Engine(a.engine.resolve(), args, cwd, env, out / "fresh-engine.txt")
    try:
        for label, (tokens, sampling) in prompts.items():
            r = engine.request(tokens, sampling, 32)
            records[label]["fresh"] = r
            (out / "results.json").write_text(json.dumps(records, indent=2))
            assert "error" not in r and r["tokens"] == records[label]["tokens"], label
    finally:
        engine.close()
    log = (out / "cached-engine.txt").read_text()
    assert "conversation cache: parked" in log and "conversation cache: restored" in log
    fields = records["live_continuation"]["done"].split()
    assert int(fields[8]) == len(ids) + len(first) - 1, "live state extends beyond the emitted prefix"
    assert int(fields[14]) < int(fields[2]), "live prefix was reread"
    assert int(records["parked_continuation"]["done"].split()[8]) == len(continuation) + len(second) - 1
    assert int(records["parked_prompt"]["done"].split()[8]) == 16, "checkpoint was not restored"
    assert records["sampled"]["offered"] == 0 and records["sampled_to_greedy"]["offered"] > 0
    print("DFlash live continuation, checkpoint rewind and conversation parking: PASS")


if __name__ == "__main__":
    main()
