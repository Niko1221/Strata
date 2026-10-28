"""prefill-table.py - median read time per size and arm from a round's prefill-curve.py outputs.
Usage: python3 -I prefill-table.py RESULTS_DIR"""
import collections, glob, json, os, re, statistics as st, sys

d = sys.argv[1]
res = collections.defaultdict(lambda: collections.defaultdict(list))
for f in sorted(glob.glob(d + "/speed-*.txt")):
    arm = re.match(r"speed-(.+)-\d+\.txt", os.path.basename(f)).group(1)
    for s, runs in re.findall(r"prefill (\d+): median read\+1 [\d.]+s runs ([\d.,]+)", open(f).read()):
        res[arm][int(s)] += [float(x) for x in runs.split(",")]
order = [json.loads(l)["arm"] for l in open(d + "/sweep.jsonl")]
sizes = sorted(next(iter(res.values())))
print("| Arm | " + " | ".join(str(s) for s in sizes) + " | Sum |\n|---|" + "---|" * (len(sizes) + 1))
for a in order:
    if a in res:
        m = [st.median(res[a][s]) for s in sizes]
        print(f"| {a} | " + " | ".join(f"{x:.2f}" for x in m) + f" | {sum(m):.1f} |")
