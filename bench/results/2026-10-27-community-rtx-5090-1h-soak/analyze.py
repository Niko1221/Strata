#!/usr/bin/env python3
"""Turn the soak's raw JSONL into the compact files a report folder carries.

Keeps every request (one CSV row each) and every GPU/health sample, drops nothing, and writes the aggregates a
reader can check against the CSV.
"""
from __future__ import annotations

import csv
import json
import statistics
import sys
from pathlib import Path

SRC = Path(sys.argv[1] if len(sys.argv) > 1 else "/workspace/Strata/results-soak-1h")
DST = Path(sys.argv[2] if len(sys.argv) > 2 else "/workspace/Strata/soak-report")


def load(name):
    p = SRC / name
    return [json.loads(ln) for ln in p.read_text(encoding="utf-8").splitlines() if ln.strip()] if p.exists() else []


def pct(xs, q):
    xs = sorted(xs)
    return xs[min(int(q * (len(xs) - 1)), len(xs) - 1)] if xs else None


DST.mkdir(parents=True, exist_ok=True)
reqs, gpus, health = load("requests.jsonl"), load("gpu.jsonl"), load("health.jsonl")
ok = [r for r in reqs if r["status"] == "ok"]
span = reqs[-1]["elapsed_s"] - reqs[0]["elapsed_s"] if len(reqs) > 1 else 0.0

with (DST / "requests.csv").open("w", newline="", encoding="utf-8") as f:
    w = csv.writer(f)
    w.writerow(["index", "at", "elapsed_s", "status", "ttft_s", "total_s", "prompt_tokens", "completion_tokens",
                "visible_chars", "end_to_end_tok_s"])
    for r in reqs:
        w.writerow([r["index"], r["at"], r["elapsed_s"], r["status"],
                    "" if r["ttft_s"] is None else round(r["ttft_s"], 4),
                    round(r["total_s"], 4), r.get("prompt_tokens") or 0, r["completion_tokens"],
                    r.get("chars") or 0,
                    "" if not r["total_s"] else round(r["completion_tokens"] / r["total_s"], 2)])

for name, rows, cols in (("gpu.csv", gpus, ["at", "index", "utilization", "memory_used_mib", "memory_total_mib",
                                            "temperature_c", "power_w", "clock_sm_mhz"]),
                         ("health.csv", health, ["at", "status", "max_context", "loaded", "images", "error"])):
    with (DST / name).open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=cols, extrasaction="ignore")
        w.writeheader()
        for row in rows:
            w.writerow({k: row.get(k, "") for k in cols})

e2e = [r["completion_tokens"] / r["total_s"] for r in ok if r["total_s"]]
ttft = [r["ttft_s"] for r in ok if r["ttft_s"]]
total = [r["total_s"] for r in ok]
tok = sum(r["completion_tokens"] for r in ok)
first = [r for r in ok if r["elapsed_s"] <= 600]
last = [r for r in ok if r["elapsed_s"] >= span - 600]
summary = {
    "what": "one hour of back-to-back chat requests, concurrency 1, to the running Strata server",
    "requests": len(reqs), "ok": len(ok), "failed": len(reqs) - len(ok),
    "wall_clock_s": round(span, 1), "rate_requests_per_min": round(60 * len(reqs) / span, 2),
    "generated_tokens": tok, "generated_tokens_per_min": round(60 * tok / span, 1),
    "end_to_end_tok_s": {"median": round(statistics.median(e2e), 1), "p10": round(pct(e2e, 0.1), 1),
                         "p90": round(pct(e2e, 0.9), 1)},
    "latency_s": {"median": round(statistics.median(total), 3), "p95": round(pct(total, 0.95), 3), "max": round(max(total), 3)},
    "ttft_content_s": {"median": round(statistics.median(ttft), 3), "p95": round(pct(ttft, 0.95), 3),
                       "requests_without_visible_content": len(ok) - len(ttft)},
    "max_tokens": 192, "capped_requests": sum(1 for r in ok if r["completion_tokens"] >= 192),
    "drift": {
        "first_10min": {"n": len(first), "end_to_end_median": round(statistics.median(
            [r["completion_tokens"] / r["total_s"] for r in first if r["total_s"]]), 1)},
        "last_10min": {"n": len(last), "end_to_end_median": round(statistics.median(
            [r["completion_tokens"] / r["total_s"] for r in last if r["total_s"]]), 1)},
    },
    "gpu": {"samples": len(gpus),
            "utilization_median": statistics.median([g["utilization"] for g in gpus]) if gpus else None,
            "temperature_c_max": max((g["temperature_c"] for g in gpus), default=None),
            "power_w_max": max((g["power_w"] for g in gpus), default=None),
            "clock_sm_mhz_max": max((g["clock_sm_mhz"] for g in gpus), default=None),
            "memory_used_mib": sorted({int(g["memory_used_mib"]) for g in gpus}) if gpus else []},
    "health_samples": len(health), "health_errors": sum(1 for h in health if h.get("error")),
}
(DST / "summary.json").write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
print(json.dumps(summary, indent=2))
for f in sorted(DST.iterdir()):
    print("%-16s %8d bytes" % (f.name, f.stat().st_size))
