#!/usr/bin/env python3
# analyze.py: pools every request of all passes per arm (raw/<arm>-p<pass>.json), prints medians and the change against the model's base arm.
import glob, json, re, statistics as st, sys
from collections import defaultdict

D = sys.argv[1] if len(sys.argv) > 1 else "."
pooled = defaultdict(lambda: defaultdict(list))   # arm -> metric -> values
passes = defaultdict(lambda: defaultdict(list))   # arm -> metric -> per-pass medians
slots = {}

for f in sorted(glob.glob(f"{D}/raw/*-p*.json")):
    arm, p = re.search(r"raw/(.+)-p(\d+)\.json$", f).groups()
    doc = json.load(open(f)); r = doc["results"]
    m = {
        "chat tok/s": [q["decode_tps"] for q in r["chat"]["requests"] if q["ok"]],
        "agent tok/s": [q["decode_tps"] for run in r["agent"]["sessions"] for q in run if q["ok"] and q["generated_tokens"] >= 8],
        "agent TTFT s": [q["ttft_s"] for run in r["agent"]["sessions"] for q in run if q["ok"]],
        "4K prompt tok/s": [q["prompt_tps"] for q in r["prompts"]["4K"]["requests"] if q["ok"]],
        "16K prompt tok/s": [q["prompt_tps"] for q in r["prompts"]["16K"]["requests"] if q["ok"]],
    }
    for k, v in m.items():
        pooled[arm][k] += v
        if v: passes[arm][k].append(st.median(v))
    s = (doc["machine"].get("engine_info") or {}).get("expert_slots")
    if s: slots[arm] = s

metrics = ["chat tok/s", "agent tok/s", "agent TTFT s", "4K prompt tok/s", "16K prompt tok/s"]
print(f"{'arm':<18}{'n':>3}{'slots':>7}  " + "  ".join(f"{k:>22}" for k in metrics))
for arm in sorted(pooled, key=lambda a: (a.split("-")[0], a.split("-")[1] != "base", a)):
    model = arm.split("-")[0]
    base = pooled.get(model + "-base") or pooled[min(a for a in pooled if a.split("-")[0] == model)]
    cells = []
    for k in metrics:
        v = pooled[arm][k]
        if not v: cells.append(f"{'-':>22}"); continue
        med = st.median(v)
        d = "" if arm.endswith("-base") else f" ({(med / st.median(base[k]) - 1) * 100:+.1f}%)"
        cells.append(f"{med:>12.1f}{d:>10}")
    print(f"{arm:<18}{len(passes[arm]['chat tok/s']):>3}{slots.get(arm, '-'):>7}  " + "  ".join(cells))

print("\nper-pass medians (spread check):")
for arm in sorted(passes):
    print(f"  {arm:<18} " + "  ".join(f"{k.split()[0]}={'/'.join(f'{x:.0f}' for x in passes[arm][k])}" for k in metrics))
