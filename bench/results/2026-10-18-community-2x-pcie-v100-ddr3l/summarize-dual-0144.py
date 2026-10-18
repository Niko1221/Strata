# -*- coding: utf-8 -*-
"""Summarize dual-GPU 0.1.44.1 arms vs single-GPU baselines (paired table)."""
import json, os

D = __import__("os").path.dirname(__import__("os").path.abspath(__file__))
def load(name):
    p = os.path.join(D, name)
    return {r["tag"]: r for r in json.load(open(p, encoding="utf-8"))} if os.path.exists(p) else None

def avg(arms, tag, key):
    vals = [a[tag][key] for a in arms if a and tag in a and a[tag].get(key) is not None]
    return sum(vals) / len(vals) if vals else None

ARMS = {
    "XXS sg  (n=2)": [load("results-flash-0144-v100sa.json"), load("results-flash-0144-v100sb.json")],
    "XXS dual(n=2)": [load("results-flash-0144-dual1.json"), load("results-flash-0144-dual2.json")],
    "XXS dual cal ": [load("results-flash-0144-dual-cal.json")],
    "XXS dual pw2 ": [load("results-flash-0144-dual-pw2.json")],
    "UD  sg  (n=2)": [load("results-flash-0144-ud-sg1.json"), load("results-flash-0144-ud-sg2.json")],
    "UD  dual(n=2)": [load("results-flash-0144-ud1.json"), load("results-flash-0144-ud2.json")],
    "UD  dual cal ": [load("results-flash-0144-ud-cal.json")],
}
TAGS = ["short", "long", "L1-4k-doc", "H1-cap4000"]
print(f"{'arm':<15} {'short':>6} {'long':>6} {'L1dec':>6} {'H1dec':>6} {'L1pre':>7}  H1ok")
for name, arms in ARMS.items():
    vals = [avg(arms, t, "decode_tps") for t in TAGS]
    pre = avg(arms, "L1-4k-doc", "prompt_tps")
    h1 = [a.get("H1-cap4000", {}).get("h1_has_8090") for a in arms if a]
    print(f"{name:<15} " + " ".join(f"{v:6.1f}" for v in vals) + f" {pre:7.0f}  {sum(1 for x in h1 if x)}/{len(h1)}")

print("\nPaired speedups (dual / single, same engine 0.1.44.1, same suite):")
def sp(sg, du, tag, key="decode_tps"):
    a, b = avg(sg, tag, key), avg(du, tag, key)
    return f"{b/a:.2f}x" if a and b else "-"
xxs_sg, xxs_du, xxs_pw = ARMS["XXS sg  (n=2)"], ARMS["XXS dual(n=2)"], ARMS["XXS dual pw2 "]
ud_sg, ud_du = ARMS["UD  sg  (n=2)"], ARMS["UD  dual(n=2)"]
for t in TAGS:
    print(f"  {t:<12} XXS {sp(xxs_sg,xxs_du,t):>6}  (+pw2 {sp(xxs_sg,xxs_pw,t):>6})   UD {sp(ud_sg,ud_du,t):>6}")
print(f"  {'L1 prefill':<12} XXS {sp(xxs_sg,xxs_du,'L1-4k-doc','prompt_tps'):>6}  (+pw2 {sp(xxs_sg,xxs_pw,'L1-4k-doc','prompt_tps'):>6})   UD {sp(ud_sg,ud_du,'L1-4k-doc','prompt_tps'):>6}")
