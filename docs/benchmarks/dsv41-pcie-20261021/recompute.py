#!/usr/bin/env python3
"""Recompute historical rates; uses only the Python standard library, no GPU."""
import csv
import hashlib
import json
from pathlib import Path
from statistics import mean, median

root = Path(__file__).resolve().parent
meta = json.loads((root / "provenance.json").read_text())
for name, expected in meta["published_files_sha256"].items():
    assert hashlib.sha256((root / name).read_bytes()).hexdigest() == expected, name
with (root / "measurements.csv").open(newline="") as f:
    rows = list(csv.DictReader(f))
assert len(rows) == 24
rates = {}
for run in meta["run_order"]:
    group = [r for r in rows if r["run"] == run]
    assert [int(r["pass"]) for r in group] == [1, 2, 3]
    for r in group:
        assert r["warmup"] == "False" and int(r["bytes"]) == meta["payload_bytes_per_pass"]
        assert int(r["count"]) == meta["cold_experts"]
        assert abs(int(r["bytes"]) / float(r["wall_ms"]) / 1e6 - float(r["wall_GBps"])) < 1e-7
        assert abs(int(r["bytes"]) / float(r["event_ms"]) / 1e6 - float(r["event_GBps"])) < 1e-7
    rates[run] = median(float(r["wall_GBps"]) for r in group)
arm_rates = {}
for arm in "ABCD":
    arm_rates[arm] = mean([rates[arm + "1"], rates[arm + "2"]])
    print(f"{arm}: {arm_rates[arm]:.6f} GB/s; process medians {rates[arm+'1']:.6f}, {rates[arm+'2']:.6f}")
print(f"D/A: {(arm_rates['D'] / arm_rates['A'] - 1) * 100:.6f}%")
print("24 recorded timed passes verified; payload integrity and this PR's GPU behavior remain unverified.")
