"""Recalculate saved measurements; no server requests."""
import json
import statistics
from pathlib import Path

rows = json.loads((Path(__file__).parent / "results.json").read_text())
groups = {}
for row in rows:
    key = "/".join((row["date"], row["arm"], row["group"]))
    groups.setdefault(key, []).append(row)

summary = {}
for key, group in groups.items():
    metrics = [row["metrics"] for row in group]
    rates = [m["decode_tok_s"] for m in metrics]
    summary[key] = {
        "n": len(group),
        "weighted_decode_tps": sum(m["output_tokens"] for m in metrics)
        / (sum(m["decode_ms"] for m in metrics) / 1000),
        "median_decode_tps": statistics.median(rates),
        "range_decode_tps": [min(rates), max(rates)],
        "prompt_tokens": sorted({m["prompt_tokens"] for m in metrics}),
        "output_range": [min(m["output_tokens"] for m in metrics),
                         max(m["output_tokens"] for m in metrics)],
        "prompt_seconds": [m["prompt_ms"] / 1000 for m in metrics],
        "json_checks": [sum(row["correct"] for row in group), len(group)],
    }
print(json.dumps(summary, ensure_ascii=False, indent=2))
