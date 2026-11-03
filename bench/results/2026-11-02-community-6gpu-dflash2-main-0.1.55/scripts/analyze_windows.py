import glob, json, re, statistics as st

root = "/tmp/strata-matrix2/out/"
LINE = re.compile(r"prompt (\d+) tokens = \d+ reused .*?, (\d+) generated in (\d+) ms \([\d.]+ tok/s\), drafts accepted (\d+) of (\d+)")
CL = ["code", "prose", "json", "edit", "chat", "tool", "thinking"]
RUNS = [("window 4", ["r1-swift-mtp", "r2-swift-mtp"], ["r1-swift-dflash2", "r2-swift-dflash2"]),
        ("window 6", ["s6-swift-mtp"], ["s6-swift-dflash2"]),
        ("window 8", ["s8-swift-mtp"], ["s8-swift-dflash2"])]


def stats(runs):
    tps = {c: [] for c in CL}
    W = R = T = MS = A = O = 0
    pts = []
    err = 0
    for r in runs:
        rows = [json.loads(l) for l in open(root + r + ".jsonl")]
        err += sum(1 for x in rows if x.get("error"))
        eng = [tuple(map(int, m)) for m in LINE.findall(open(root + r + ".engine.log", errors="replace").read())]
        for x in rows:
            if x.get("section") != "decode" or not x.get("timings"):
                continue
            tps[x["tag"].split("-")[0]].append(x["timings"]["predicted_per_second"])
            for i, e in enumerate(eng):
                if e[0] == x["prompt_tokens"] and e[1] == x["completion_tokens"]:
                    g, ms, a, o = e[1:]
                    W += g - a; R += g - a + o; T += g; MS += ms; A += a; O += o
                    pts.append((g - a, g - a + o, ms))
                    eng.pop(i)
                    break
    allv = [v for c in CL for v in tps[c]]
    sww = sum(p[0] ** 2 for p in pts); swr = sum(p[0] * p[1] for p in pts); srr = sum(p[1] ** 2 for p in pts)
    swm = sum(p[0] * p[2] for p in pts); srm = sum(p[1] * p[2] for p in pts)
    det = sww * srr - swr ** 2
    return dict(all=st.median(allv), cls={c: st.median(tps[c]) for c in CL}, wall=1000 * T / MS, msw=MS / W, rows=R / W, tok=T / W, off=O / W, acc=100 * A / O, tokens=T, fixed=(swm * srr - srm * swr) / det, per=(sww * srm - swr * swm) / det, err=err, n=len(allv))


for name, m, d in RUNS:
    print("==", name)
    for lab, runs in (("mtp", m), ("dflash2", d)):
        s = stats(runs)
        print(f"{lab:8} median {s['all']:.1f} tok/s; time-weighted {s['wall']:.1f}; rows/window {s['rows']:.2f}; tokens/window {s['tok']:.2f}; drafts offered/window {s['off']:.2f}; accepted {s['acc']:.1f}%; ms/window {s['msw']:.1f}; fit fixed {s['fixed']:.1f} + {s['per']:.1f}/row; tokens {s['tokens']}; n {s['n']}; errors {s['err']}")
        print("         " + ", ".join(f"{c} {s['cls'][c]:.1f}" for c in CL))
