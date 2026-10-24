"""Recompute the cache correctness report from the recorded requests."""
import json
from pathlib import Path

root = Path(__file__).resolve().parent
data = json.loads((root / "results.json").read_text(encoding="utf-8"))
assert data["completed"] and data["recovery"]["restored"]
base, fixed = data["arms"]
assert len(base["requests"]) == len(fixed["requests"]) == 15
mismatches = []
for a, b in zip(base["requests"], fixed["requests"]):
    assert a["name"] == b["name"]
    if a["output_ids"] != b["output_ids"]:
        mismatches.append(a["name"])
summary = {"output_mismatches": mismatches, "arms": []}
for arm in data["arms"]:
    logs = "\n".join(line for r in arm["requests"] for line in r["cache_log"])
    revisit = [r["prompt_read"] for r in arm["requests"] if r["name"].startswith("revisit-")]
    summary["arms"].append({"label": arm["label"], "parking_refused": logs.count("skip parking"),
                            "restores": logs.count("restored"), "revisit_prompt_read": revisit,
                            "revisit_prompt_read_total": sum(revisit)})
assert not mismatches
assert summary["arms"][0]["parking_refused"] > 0
assert summary["arms"][1]["parking_refused"] == 0
print(json.dumps(summary, indent=2))
