import hashlib
import json
import statistics
from collections import defaultdict
from pathlib import Path

p = Path(__file__).resolve().parent
rows = [json.loads(f.read_text()) for f in p.glob("r*.json")]
assert len(rows) == 24, len(rows)
assert all(r["correct"] and r["finish"] == "stop" for r in rows)
inputs = defaultdict(set)
groups = defaultdict(list)
for row in rows:
    repeat, phase = row["label"].split("-", 1)
    inputs[repeat, row["case"]].add(hashlib.sha256(json.dumps(row["request"], sort_keys=True).encode()).hexdigest())
    groups[repeat, phase].append(row)
assert all(len(hashes) == 1 for hashes in inputs.values())


def tps(items):
    return sum(r["metrics"]["output_tokens"] for r in items) * 1000 / sum(r["metrics"]["decode_ms"] for r in items)


summary = {"requests": len(rows), "correct": len(rows), "matched_inputs": True, "phases": {}, "paired_ratios": []}
for phase in ["full", "gpu0-4g", "gpu1-4g"]:
    selected = [r for r in rows if ("-full-" in r["label"] if phase == "full" else r["label"].endswith(phase))]
    trials = [tps(rs) for (repeat, name), rs in groups.items() if (name.startswith("full-") if phase == "full" else name == phase)]
    summary["phases"][phase] = {"requests": len(selected), "weighted_tok_s": tps(selected),
        "median_trial_tok_s": statistics.median(trials), "trial_range_tok_s": [min(trials), max(trials)],
        "by_case": {name: tps([r for r in selected if r["case"] == name]) for name in sorted({r["case"] for r in selected})}}
for repeat in range(3):
    full = groups[f"r{repeat}", "full-before"] + groups[f"r{repeat}", "full-after"]
    summary["paired_ratios"].append({"repeat": repeat, "full_tok_s": tps(full),
        **{phase: tps(groups[f"r{repeat}", phase]) / tps(full) for phase in ["gpu0-4g", "gpu1-4g"]}})
events = json.loads((p / "events.json").read_text())
summary["released_mib"] = defaultdict(list)
for e in events:
    if e["gpu"] is not None:
        used = lambda s: {int(line.split(",")[0]): int(line.split(",")[1]) for line in s.splitlines()}
        summary["released_mib"][e["phase"]].append(used(e["before"])[e["gpu"]] - used(e["after"])[e["gpu"]])
(p / "summary.json").write_text(json.dumps(summary, indent=2))
print(json.dumps(summary, indent=2))
