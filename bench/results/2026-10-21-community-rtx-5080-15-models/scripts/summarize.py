#!/usr/bin/env python
"""Print the community report tables from the checked-in runs/ directory."""
import json
import statistics
from pathlib import Path

BASE = Path(__file__).resolve().parents[1]


def load(p):
    return json.loads(Path(p).read_text(encoding="utf-8-sig"))


def rng(d):
    return f"{d['median']:.1f} [{d['min']:.1f}-{d['max']:.1f}]"


print("| Model | Chat decode | TTFT | 4K prompt | 16K prompt | 32K prompt | Agent decode |")
print("| --- | --- | --- | --- | --- | --- | --- |")
for run in sorted((BASE / "runs").iterdir()):
    p = run / "results.json"
    if not p.is_file():
        continue
    j = load(p)
    r = j["results"]
    print(f"| {run.name} | {rng(r['chat']['decode_tps'])} | {rng(r['chat']['ttft_s'])} | "
          f"{rng(r['prompts']['4K']['prompt_tps'])} | {rng(r['prompts']['16K']['prompt_tps'])} | "
          f"{rng(r['prompts']['32K']['prompt_tps'])} | {rng(r['agent']['turn_decode_tps'])} |")

print()
print("| Model | 32K | 60K | 64K | 128K | 200K |")
print("| --- | ---: | ---: | ---: | ---: | ---: |")
for run in sorted((BASE / "runs").iterdir()):
    q = run / "quality"
    if not q.is_dir():
        continue
    rows = load(q / "needles.json") if (q / "needles.json").is_file() else []
    rows += load(q / "needles-60k.json") if (q / "needles-60k.json").is_file() else []
    agg = {}
    for e in rows:
        if e.get("length"):
            a = agg.setdefault(e["length"], [0, 0])
            a[0] += int(bool(e.get("found")))
            a[1] += 1
    cells = []
    for length in ("32k", "60k", "64k", "128k", "200k"):
        a = agg.get(length)
        cells.append(f"{a[0]}/{a[1]}" if a else "-")
    print(f"| {run.name} | " + " | ".join(cells) + " |")
