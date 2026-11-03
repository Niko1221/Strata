"""The ring and parallel tables, reading engine lines in order (one request at a time: the n-th line with this prompt size is the n-th call)."""
import glob, json, os, re, statistics as st, sys

root = sys.argv[1] if len(sys.argv) > 1 else "/tmp/strata-matrix2/out"
LINE = re.compile(r"prompt (\d+) tokens = (\d+) reused \+ (?:(\d+) of \d+|(\d+)) read in (\d+) ms.*?, (\d+) generated in (\d+) ms \(([\d.]+) tok/s\), drafts accepted (\d+) of (\d+)")
RUNS = ["r1-swift-mtp", "r1-swift-dflash2", "r1-orig-mtp", "r1-orig-dflash2", "r2-orig-dflash2", "r2-orig-mtp", "r2-swift-dflash2", "r2-swift-mtp", "c1-swift-dflash2-noring", "c1-orig-dflash2-noring"]


def load(r):
    rows = [json.loads(l) for l in open(f"{root}/{r}.jsonl")]
    log = open(f"{root}/{r}.engine.log", errors="replace").read()
    eng = [dict(prompt=int(m[1]), reused=int(m[2]), read=int(m[3] or m[4]), read_ms=int(m[5]), gen=int(m[6]), gen_ms=int(m[7]), tps=float(m[8]), acc=int(m[9]), off=int(m[10])) for m in LINE.finditer(log)]
    return rows, eng, log


print("# ring: accepted/offered by state, in order")
RING = ("ring-fresh", "ring-in-place", "ring-saved-copy", "ring-saved-copy-2")
tot = {}
for r in RUNS:
    rows, eng, log = load(r)
    calls = [x for x in rows if x.get("tag") in RING]
    p = calls[0]["prompt_tokens"]
    lines = [e for e in eng if e["prompt"] == p and e["gen"] == 300]
    if len(lines) != 4:
        print(r, "lines", len(lines))
    out = []
    for x, e in zip(calls, lines):
        out.append(f"{x['tag'][5:]}: {e['acc']}/{e['off']} {e['tps']} tok/s read {e['read']} sha {x['sha'][:6]}")
        kind = "noring" if "noring" in r else "mtp" if "mtp" in r else "saved"
        model = "swift" if "swift" in r else "orig"
        k = (model, kind, x["tag"])
        a = tot.setdefault(k, [0, 0, []])
        a[0] += e["acc"]; a[1] += e["off"]; a[2].append(e["tps"])
    print(r, "|", " | ".join(out))
    print("   kept:", re.findall(r"DFlash2 drafter: (\d+ of the \d+ context rows below token 17601 kept \([^)]*\))", log)[:6])
print("\ntotals")
for k in sorted(tot):
    a = tot[k]
    print(k, f"{a[0]}/{a[1]} = {100 * a[0] / a[1]:.1f}% tok/s {a[2]}")

print("\n# session follow-ups: accepted/offered, tok/s; saved rings against none, DFlash2 only, plus MTP for scale (replies >= 20 tokens, one request at a time)")
TAGS = {"turn1", "turn2", "turn3", "turn4", "turn5", "turn-after-side", "first-again", "other-again", "after-parallel0", "after-parallel1", "after-cancel"}
RETURN = {"turn-after-side", "first-again", "other-again", "after-parallel0", "after-parallel1"}
for r in RUNS:
    rows, eng, log = load(r)
    idx = 0
    res = {}
    for x in rows:
        if x.get("section") != "session" or x.get("tag") not in TAGS:
            continue
        for e in eng:
            if e["prompt"] == x["prompt_tokens"] and e["gen"] == x["completion_tokens"] and e["reused"] == x["cached_tokens"]:
                res[x["tag"]] = e
                break
    allt = [res[t] for t in res]
    ret = [res[t] for t in res if t in RETURN]
    def s(v):
        a, o, g, m = sum(e["acc"] for e in v), sum(e["off"] for e in v), sum(e["gen"] for e in v), sum(e["gen_ms"] for e in v)
        return f"{a}/{o} = {100 * a / max(o, 1):.1f}% {1000 * g / max(m, 1):.1f} tok/s over {g} tok"
    print(f"{r}: all {s(allt)} | after a switch {s(ret)}")

print("\n# parallel, per round")
for r in RUNS:
    rows, eng, log = load(r)
    walls = {x["tag"]: x["total_s"] for x in rows if x.get("section") == "parallel" and x["tag"].endswith("wall")}
    out = []
    for n in range(3):
        ca = [(x["cached_tokens"], x["prompt_tokens"], x["completion_tokens"], x["ttft_s"]) for x in rows if x.get("tag") in (f"long0-round{n}", f"long1-round{n}")]
        out.append(f"round{n} wall {walls.get(f'long-round{n}-wall')}: " + ", ".join(f"reused {a}/{b} gen {c} ttft {d}" for a, b, c, d in ca))
    print(r, "|", " | ".join(out))
    for m in re.findall(r"strata batch \(pipelined[^\n]*", log)[:2]:
        pass
print("\nprefill delta MTP vs DFlash2 in seconds (median of the read time)")
for model in ("swift", "orig"):
    for size, tag in (("4k", "prefill-4k"), ("32k", "prefill-32k"), ("128k", "prefill-128k"), ("18.7k", None)):
        vals = {}
        for dr in ("mtp", "dflash2"):
            v = []
            for r in RUNS:
                if f"-{model}-{dr}" in r and "noring" not in r:
                    rows, _, _ = load(r)
                    for x in rows:
                        if tag and x.get("tag", "").startswith(tag) and x.get("timings"):
                            v.append(x["timings"]["prompt_ms"] / 1000)
                        if not tag and x.get("tag") == "turn0":
                            v.append(x["timings"]["prompt_ms"] / 1000)
            vals[dr] = st.median(v)
        print(model, size, f"mtp {vals['mtp']:.2f}s dflash2 {vals['dflash2']:.2f}s delta {vals['dflash2'] - vals['mtp']:+.2f}s ({100 * (vals['dflash2'] / vals['mtp'] - 1):+.1f}%)")

print("\nexact pairs for the answer identity: of all temperature-0 calls of one (model, tag), how many distinct outputs per drafter, and how many hashes both drafters produced")
for model in ("swift", "orig"):
    per = {"mtp": {}, "dflash2": {}}
    for r in RUNS:
        if "noring" in r or model not in r:
            continue
        dr = "dflash2" if "dflash2" in r else "mtp"
        rows, _, _ = load(r)
        for x in rows:
            if x.get("section") in ("decode", "ring", "prefill") and "sha" in x and not x["tag"].startswith(("thinking", "other")):
                key = "ring" if x["tag"].startswith("ring") else x["tag"]
                per[dr].setdefault(key, []).append(x["sha"])
    keys = [k for k in per["mtp"] if k in per["dflash2"]]
    shared = sum(bool(set(per["mtp"][k]) & set(per["dflash2"][k])) for k in keys)
    print(model, f"prompts {len(keys)}, a same answer reached by both drafters on {shared}; ring prompt outputs: mtp {sorted(set(per['mtp']['ring']))} dflash2 {sorted(set(per['dflash2']['ring']))}")
