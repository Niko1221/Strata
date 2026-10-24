import json, statistics, sys
path = sys.argv[1]
rows = [json.loads(l) for l in open(path)]
arms = sorted({r["arm"] for r in rows}, key=lambda a: (a != "default", a))
by = {}
for r in rows:
    by[(r["round"], r["req"], r["arm"])] = r
rounds = sorted({r["round"] for r in rows}); reqs = sorted({r["req"] for r in rows})
a0, a1 = arms[0], arms[1]
def pooled(arm, which):
    v = [r["decode"] for r in rows if r["arm"] == arm and r["decode"] and (which == "all" or r["req"].startswith("chat"))]
    return statistics.median(v), len(v), min(v), max(v)
for which in ("all", "chat"):
    for a in arms:
        m, n, lo, hi = pooled(a, which)
        print(f"{which:4s} {a:8s} median {m:5.1f} tok/s  (n={n}, {lo:.1f}-{hi:.1f})")
pairs = [(by.get((rd, q, a0)), by.get((rd, q, a1))) for rd in rounds for q in reqs]
pairs = [(x, y) for x, y in pairs if x and y and x["decode"] and y["decode"]]
w = sum(1 for x, y in pairs if x["decode"] > y["decode"])
print(f"{a0} faster than {a1} in {w} of {len(pairs)} pairs (round x request)")
print("answers shorter than the cap:", sorted({(r['req'], r['n']) for r in rows if r['n'] is not None and r['n'] < int(sys.argv[2])})[:6] if len(sys.argv) > 2 else "")
