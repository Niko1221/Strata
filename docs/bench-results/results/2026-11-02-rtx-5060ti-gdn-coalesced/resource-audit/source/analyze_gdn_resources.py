#!/usr/bin/env python3
"""Summarize compiler resources and paired *synthetic* GDN timings; no E2E verdict."""
import argparse
import csv
import json
import math
from pathlib import Path
import re
import statistics

ARMS = ("verify_kernels", "verify_kernels_ungrouped_q")
UNITS = ("fused_gdn", "verify_kernels", "verify_kernels_two_blocks", "verify_kernels_ungrouped_q")


def resources(root, name):
    log = (root / (name + ".ptxas.log")).read_text(encoding="utf-8-sig")
    pattern = r"Function properties for (\S+)\s*\n\s*(\d+) bytes stack frame, (\d+) bytes spill stores, (\d+) bytes spill loads"
    props = {m[0]: tuple(map(int, m[1:])) for m in re.findall(pattern, log)}
    with (root / (name + ".resources.csv")).open(encoding="utf-8-sig", newline="") as handle:
        rows = list(csv.DictReader(handle))
    selected = []
    for row in rows:
        if not re.search(r"gdn_step_norm_(?:multi_)?kernel<", row["kernel"]):
            continue
        if row["sm"] != "sm_120" or row["mangled"] not in props:
            raise ValueError("Missing sm_120 ptxas record for " + row["kernel"])
        stack, stores, loads = props[row["mangled"]]
        if stack != int(row["stack"]):
            raise ValueError("cuobjdump and ptxas stack sizes disagree")
        selected.append({"kernel": row["kernel"], "registers": int(row["reg"]),
                         "stack_bytes": stack, "local_bytes": int(row["local"]),
                         "spill_store_bytes": stores, "spill_load_bytes": loads})
    if len(selected) != (2 if name == "fused_gdn" else 4):
        raise ValueError("Unexpected number of GDN variants in " + name)
    return selected


def timings(path):
    rows = [json.loads(line) for line in path.read_text(encoding="utf-8-sig").splitlines()]
    if rows[-1] != {"kind": "status", "completed": True}:
        raise ValueError("Incomplete benchmark: " + str(path))
    data = [r for r in rows if r.get("kind") == "timing"]
    if len(data) != 120:
        raise ValueError("Expected 8 configurations x 15 timings")
    result = {}
    for buffers in (1, 48):
        for tokens in (1, 2, 4, 8):
            subset = [r for r in data if (r["buffers"], r["tokens"]) == (buffers, tokens)]
            if sorted(r["round"] for r in subset) != list(range(15)):
                raise ValueError("Missing or duplicated timing round")
            for r in subset:
                if r["calls"] != (64 if buffers == 1 else 192) or not math.isfinite(r["us_per_call"]) or r["us_per_call"] <= 0:
                    raise ValueError("Invalid timing")
            result[buffers, tokens] = statistics.median(r["us_per_call"] for r in subset)
    return result


def summarize(root):
    measured = {arm: [timings(root / f"{arm}-r{i}.jsonl") for i in range(4)] for arm in ARMS}
    for arm in ARMS:
        parity = (root / (arm + ".parity.log")).read_text(encoding="utf-8-sig")
        if "gdn: 0 failures" not in parity or "gdn_parity OK" not in parity:
            raise ValueError("Parity did not pass: " + arm)
    results = []
    for buffers in (1, 48):
        for tokens in (1, 2, 4, 8):
            baseline = [r[buffers, tokens] for r in measured[ARMS[0]]]
            candidate = [r[buffers, tokens] for r in measured[ARMS[1]]]
            results.append({"buffers": buffers, "tokens": tokens,
                            "baseline_run_median_us": baseline, "candidate_run_median_us": candidate,
                            "baseline_median_of_runs_us": statistics.median(baseline),
                            "candidate_median_of_runs_us": statistics.median(candidate),
                            "paired_candidate_over_baseline": [c / b for b, c in zip(baseline, candidate)]})
    return {"schema": 1, "performance_qualified": False,
            "limits": "Shared GPU; unfixed clocks; synthetic recurrent inputs; event timing excludes allocation/host/model work; no DRAM counters or E2E run. No speedup adoption verdict.",
            "resources": {unit: resources(root, unit) for unit in UNITS}, "timings": results}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("directory", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    result = summarize(args.directory)
    args.output.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print("Resource records and all 960 timing samples validated; performance remains unqualified.")


if __name__ == "__main__":
    main()