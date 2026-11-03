"""Reads the matrix's raw files and prints every table for the report. Pairs engine log lines with API calls by prompt size and generated count."""
import glob, json, os, re, statistics as st, sys

root = sys.argv[1] if len(sys.argv) > 1 else "/tmp/strata-matrix2/out"
RUNS = ["r1-swift-mtp", "r1-swift-dflash2", "r1-orig-mtp", "r1-orig-dflash2", "r2-orig-dflash2", "r2-orig-mtp", "r2-swift-dflash2", "r2-swift-mtp"]
CFG = [("swift", "mtp"), ("swift", "dflash2"), ("orig", "mtp"), ("orig", "dflash2")]
LINE = re.compile(r"prompt (\d+) tokens = (\d+) reused \+ (?:(\d+) of \d+|(\d+)) read in (\d+) ms.*?, (\d+) generated in (\d+) ms \(([\d.]+) tok/s\), drafts accepted (\d+) of (\d+)")


def load(run):
    path = f"{root}/{run}.jsonl"
    if not os.path.exists(path):
        return None
    rows = [json.loads(l) for l in open(path)]
    log = open(f"{root}/{run}.engine.log", errors="replace").read()
    eng = [tuple(int(x or 0) for x in (m[1], m[2], m[3] or m[4], m[5], m[6], m[7], m[9], m[10])) for m in LINE.finditer(log)]
    # (prompt, reused, read, read_ms, gen, gen_ms, accepted, offered)
    return rows, eng, log


def med(v):
    v = [x for x in v if x is not None]
    return (st.median(v), min(v), max(v), len(v)) if v else (None, None, None, 0)


def fmt(m, d=1):
    return "n/a" if m[0] is None else f"{m[0]:.{d}f} ({m[1]:.{d}f}-{m[2]:.{d}f})"


data = {r: load(r) for r in RUNS}
data = {r: v for r, v in data.items() if v}
def runs(model, drafter):
    return [r for r in RUNS if r in data and f"-{model}-{drafter}" in r]


print("# accounting")
for r, (rows, eng, log) in data.items():
    errs = [x["tag"] for x in rows if x.get("error")]
    start = rows[0]
    print(f"{r}: calls {sum(1 for x in rows if 'tag' in x and not x['tag'].endswith('wall'))} output {sum(x.get('completion_tokens') or 0 for x in rows)} engine_gen {sum(e[4] for e in eng)} prompt_read {sum(e[2] for e in eng)} errors {errs} drafting {start.get('drafting')} note {start.get('note')!r} slots {start.get('expert_slots')} cache_mib {start.get('expert_cache_mib')} nonfinite {len(re.findall('non-?finite', log))} cuda_err {log.count('CUDA error')}")

print("\n# prefill tok/s (engine timings; fresh prompt; 3 reps in each of 2 runs) median (min-max) n")
print("config | 4k | 32k | 128k | ttft 128k s")
for model, drafter in CFG:
    cells = []
    for size in ("4k", "32k", "128k"):
        v = []
        for r in runs(model, drafter):
            for x in data[r][0]:
                if x.get("section") == "prefill" and x["tag"].startswith(f"prefill-{size}-") and x.get("timings") and not x.get("error"):
                    t = x["timings"]
                    v.append(t["prompt_n"] / t["prompt_ms"] * 1000 if t.get("prompt_ms") else None)
        cells.append(fmt(med(v), 0))
    ttft = [x.get("ttft_s") for r in runs(model, drafter) for x in data[r][0] if x.get("section") == "prefill" and "128k" in x["tag"]]
    print(f"{model}/{drafter} | " + " | ".join(cells) + f" | {fmt(med(ttft))}")
print("\nprefill per run, order effect (median of 3):")
for r in data:
    row = []
    for size in ("4k", "32k", "128k"):
        v = [x["timings"]["prompt_n"] / x["timings"]["prompt_ms"] * 1000 for x in data[r][0] if x.get("section") == "prefill" and x["tag"].startswith(f"prefill-{size}-") and x.get("timings")]
        row.append(f"{st.median(v):.0f}" if v else "n/a")
    print(r, row)

print("\n# decode: engine tok/s per prompt class; 3 reps in each of 2 runs; median (min-max) n, tokens")
classes = ["code", "prose", "json", "edit", "chat", "tool", "thinking"]
print("class | " + " | ".join(f"{m}/{d}" for m, d in CFG))
agg = {}
for c in classes:
    cells = []
    for model, drafter in CFG:
        v, tok = [], 0
        for r in runs(model, drafter):
            for x in data[r][0]:
                if x.get("section") == "decode" and x["tag"].startswith(c + "-") and x.get("timings"):
                    v.append(x["timings"].get("predicted_per_second"))
                    tok += x.get("completion_tokens") or 0
        agg.setdefault((model, drafter), []).extend(v)
        cells.append(f"{fmt(med(v))} [{tok}]")
    print(c + " | " + " | ".join(cells))
print("all | " + " | ".join(fmt(med(agg[k])) for k in CFG))

print("\n# paired: drafter effect within model and run round, per class (DFlash2 vs MTP, median of 3 reps each)")
for model in ("swift", "orig"):
    for rnd in ("r1", "r2"):
        mr, dr = f"{rnd}-{model}-mtp", f"{rnd}-{model}-dflash2"
        if mr not in data or dr not in data:
            continue
        out = []
        for c in classes:
            a = [x["timings"]["predicted_per_second"] for x in data[mr][0] if x.get("section") == "decode" and x["tag"].startswith(c + "-") and x.get("timings")]
            b = [x["timings"]["predicted_per_second"] for x in data[dr][0] if x.get("section") == "decode" and x["tag"].startswith(c + "-") and x.get("timings")]
            out.append(f"{c} {100 * (st.median(b) / st.median(a) - 1):+.1f}%")
        print(model, rnd, ", ".join(out))
print("\n# paired: weights effect within drafter and round (original vs Swift)")
for drafter in ("mtp", "dflash2"):
    for rnd in ("r1", "r2"):
        sr, or_ = f"{rnd}-swift-{drafter}", f"{rnd}-orig-{drafter}"
        if sr not in data or or_ not in data:
            continue
        out = []
        for c in classes:
            a = [x["timings"]["predicted_per_second"] for x in data[sr][0] if x.get("section") == "decode" and x["tag"].startswith(c + "-") and x.get("timings")]
            b = [x["timings"]["predicted_per_second"] for x in data[or_][0] if x.get("section") == "decode" and x["tag"].startswith(c + "-") and x.get("timings")]
            out.append(f"{c} {100 * (st.median(b) / st.median(a) - 1):+.1f}%")
        print(drafter, rnd, ", ".join(out))


def accept_for(run, tag_rows):
    """accepted/offered from the engine lines that match these calls' (prompt tokens, generated)."""
    eng = list(data[run][1])
    acc = off = gen = ms = 0
    for x in tag_rows:
        for i, e in enumerate(eng):
            if e[0] == x.get("prompt_tokens") and e[4] == x.get("completion_tokens"):
                acc += e[6]; off += e[7]; gen += e[4]; ms += e[5]
                eng.pop(i)
                break
    return acc, off, gen, ms


print("\n# drafts accepted, decode section, by class: accepted/offered = %")
print("class | " + " | ".join(f"{m}/{d}" for m, d in CFG))
for c in classes + ["ALL"]:
    cells = []
    for model, drafter in CFG:
        a = o = 0
        for r in runs(model, drafter):
            rows = [x for x in data[r][0] if x.get("section") == "decode" and (c == "ALL" or x["tag"].startswith(c + "-"))]
            ac, of, _, _ = accept_for(r, rows)
            a += ac; o += of
        cells.append(f"{a}/{o} = {100 * a / o:.1f}%" if o else "n/a")
    print(c + " | " + " | ".join(cells))

ALLRUNS = RUNS + sorted(os.path.basename(p)[:-6] for p in glob.glob(f"{root}/c*-noring.jsonl"))
for r in ALLRUNS:
    if r not in data:
        d = load(r)
        if d:
            data[r] = d

print("\n# session: reused tokens and seconds, one request at a time")
tags = ["turn0", "turn1", "turn2", "turn3", "turn4", "turn5", "side-request", "turn-after-side", "other-fresh", "first-again", "other-again", "after-parallel0", "after-parallel1", "cancel", "after-cancel"]
for tag in tags:
    cells = []
    for model, drafter in CFG:
        c = []
        for r in runs(model, drafter):
            x = next((y for y in data[r][0] if y.get("tag") == tag), None)
            c.append(f"{x.get('cached_tokens')}/{x.get('prompt_tokens')} {x.get('ttft_s')}s" if x else "-")
        cells.append("; ".join(c))
    print(tag + " | " + " | ".join(cells))
print("\nreuse totals (session section, follow-ups): reused of prompt; median ttft of follow-ups")
for model, drafter in CFG:
    cached = prompt = 0
    tt = []
    for r in runs(model, drafter):
        for x in data[r][0]:
            if x.get("section") == "session" and x.get("tag") in ("turn1", "turn2", "turn3", "turn4", "turn5", "turn-after-side", "first-again", "other-again", "after-parallel0", "after-parallel1"):
                cached += x.get("cached_tokens") or 0
                prompt += x.get("prompt_tokens") or 0
                tt.append(x.get("ttft_s"))
    print(f"{model}/{drafter}: {cached}/{prompt} = {100 * cached / max(prompt, 1):.2f}%, ttft {fmt(med(tt), 2)}")

print("\n# parallel: wall seconds per round; output tokens; rate")
for model, drafter in CFG:
    long_w, short_w, long_t, short_t = [], [], [], []
    for r in runs(model, drafter):
        rows = data[r][0]
        for x in rows:
            if x.get("section") != "parallel":
                continue
            if x["tag"].endswith("wall"):
                (long_w if x["tag"].startswith("long") else short_w).append(x["total_s"])
            elif x["tag"].startswith("long"):
                long_t.append(x.get("completion_tokens") or 0)
            else:
                short_t.append(x.get("completion_tokens") or 0)
    print(f"{model}/{drafter}: 2 long wall {fmt(med(long_w), 2)} tokens {sum(long_t)} = {sum(long_t) / max(sum(long_w), 1e-9):.1f} tok/s aggregate; 3 short (2 slots) wall {fmt(med(short_w), 2)} tokens {sum(short_t)} = {sum(short_t) / max(sum(short_w), 1e-9):.1f} tok/s aggregate")

print("\n# output identity: same tag across runs (sha), temperature 0 classes; distinct hashes among the 2 x 3 reps cannot match because tags differ; between runs of one config:")
for model, drafter in CFG:
    rs = runs(model, drafter)
    if len(rs) < 2:
        continue
    same = tot = 0
    a = {x["tag"]: x["sha"] for x in data[rs[0]][0] if x.get("section") in ("decode", "prefill", "ring") and "sha" in x}
    b = {x["tag"]: x["sha"] for x in data[rs[1]][0] if x.get("section") in ("decode", "prefill", "ring") and "sha" in x}
    for k in a:
        if k in b and not k.startswith("thinking"):
            tot += 1; same += a[k] == b[k]
    print(f"{model}/{drafter}: {same}/{tot} identical between the two runs")
print("between drafters, same model, same round:")
for model in ("swift", "orig"):
    for rnd in ("r1", "r2"):
        ra, rb = f"{rnd}-{model}-mtp", f"{rnd}-{model}-dflash2"
        if ra in data and rb in data:
            a = {x["tag"]: x["sha"] for x in data[ra][0] if x.get("section") in ("decode", "prefill") and "sha" in x}
            b = {x["tag"]: x["sha"] for x in data[rb][0] if x.get("section") in ("decode", "prefill") and "sha" in x}
            ks = [k for k in a if k in b and not k.startswith("thinking")]
            print(f"{model} {rnd}: {sum(a[k] == b[k] for k in ks)}/{len(ks)} identical")

print("\n# VRAM after load MiB per card")
for r in ALLRUNS:
    p = f"{root}/{r}.vram.csv"
    if os.path.exists(p):
        print(r, [l.split(",")[2].strip() for l in open(p)])
print("\n# GPU throttle and temperature (max temp per card; throttle reasons other than idle/none)")
for r in ALLRUNS:
    p = f"{root}/{r}.gpu.csv"
    if not os.path.exists(p):
        continue
    temps, reasons = {}, {}
    for l in open(p):
        f = [x.strip() for x in l.split(",")]
        if len(f) < 7:
            continue
        temps[f[1]] = max(temps.get(f[1], 0), int(f[2]))
        if f[6] not in ("0x0000000000000000", "0x0000000000000001", "0x0000000000000004"):
            reasons[(f[1], f[6])] = reasons.get((f[1], f[6]), 0) + 1
    print(r, temps, reasons)

print("\n# control lines from the engine (no saved rings vs saved rings)")
for r in ALLRUNS:
    if r in data:
        lines = re.findall(r"DFlash2 drafter: (\d+ of the \d+ context rows below token \d+ kept \([^)]*\))", data[r][2])
        kinds = {}
        for l in lines:
            k = re.search(r"\(([^)]*)\)", l).group(1)
            kinds[k] = kinds.get(k, 0) + 1
        if lines:
            print(r, kinds)
