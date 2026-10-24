#!/usr/bin/env python3
"""Compare STRATA_LOGPOS files: logpos_compare.py REF RUN [RUN ...]  (top-1 agreement, KL(REF || RUN) over REF's top 20
renormalized, perplexity, bit for bit; one row per case, then all)"""
import sys, math
NAMES = ["docs 34k", "docs 24k", "code 16k", "docs 8k", "code 3k"]
def load(path):
    reqs, cur, last = [], [], None
    for ln in open(path):
        f = ln.split()
        if len(f) < 8: continue
        pos = int(f[0])
        if last is not None and pos < last: reqs.append(cur); cur = []
        last = pos
        top = {int(a.split(":")[0]): float(a.split(":")[1]) for a in f[8:] if ":" in a}
        cur.append(dict(pos=pos, tgt=int(f[1]), lp=float(f[2]), am=int(f[3]), top=top))
    if cur: reqs.append(cur)
    return [r for r in reqs if len(r) > 100]
def kl(ref, run):
    ids = list(ref["top"]); pr = [math.exp(ref["top"][i]) for i in ids]
    floor = min(run["top"].values()) if run["top"] else -30.0
    qr = [math.exp(run["top"].get(i, floor)) for i in ids]
    sp, sq = sum(pr), sum(qr)
    return sum((p/sp) * math.log((p/sp) / (q/sq)) for p, q in zip(pr, qr) if p > 0)
def stats(a, b):
    n = min(len(a), len(b)); a, b = a[:n], b[:n]
    k = [kl(x, y) for x, y in zip(a, b)]; k.sort()
    agree = sum(x["am"] == y["am"] for x, y in zip(a, b)) / n
    ppl = lambda r: math.exp(-sum(x["lp"] for x in r) / len(r))
    bit = all(x["lp"] == y["lp"] and x["top"] == y["top"] for x, y in zip(a, b))
    return n, agree, sum(k)/n, k[int(0.99*(n-1))], ppl(a), ppl(b), bit
ref = load(sys.argv[1])
print(f"REF {sys.argv[1].split('/')[-1]}: {len(ref)} cases, rows {[len(r) for r in ref]}")
for p in sys.argv[2:]:
    run = load(p); print(f"\n== {p.split('/')[-1]} against REF")
    print(f"{'case':10} {'n':>5} {'argmax':>7} {'KL mean':>9} {'KL p99':>8} {'ppl ref/run':>14} bit for bit")
    for i, (a, b) in enumerate(zip(ref, run)):
        n, ag, km, kp, pa, pb, bit = stats(a, b)
        print(f"{NAMES[i] if i < len(NAMES) else i:10} {n:5d} {ag*100:6.1f}% {km:9.4f} {kp:8.4f} {pa:7.2f}/{pb:6.2f}  {'yes' if bit else 'no'}")
    A = [x for r in ref for x in r]; B = [x for r in run for x in r]
    n, ag, km, kp, pa, pb, bit = stats(A, B)
    print(f"{'all':10} {n:5d} {ag*100:6.1f}% {km:9.4f} {kp:8.4f} {pa:7.2f}/{pb:6.2f}  {'yes' if bit else 'no'}")
