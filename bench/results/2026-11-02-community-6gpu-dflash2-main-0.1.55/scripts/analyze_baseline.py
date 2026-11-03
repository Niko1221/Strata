"""MTP on upstream main against MTP on the branch: same weights, same flags; main ran before and after the branch's runs."""
import json, re, statistics as st

root = "/tmp/strata-matrix2/out/"
GROUPS = {"main b1": "b1-swift-mtp-main", "main b2": "b2-swift-mtp-main", "branch r1": "r1-swift-mtp", "branch r2": "r2-swift-mtp"}
LINE = re.compile(r"prompt (\d+) tokens = (\d+) reused .*?, (\d+) generated in (\d+) ms \([\d.]+ tok/s\), drafts accepted (\d+) of (\d+)")
CL = ["code", "prose", "json", "edit", "chat", "tool", "thinking"]
SIZES = ["4k", "32k", "128k"]
res = {}
for name, r in GROUPS.items():
    rows = [json.loads(l) for l in open(root + r + ".jsonl")]
    d = {}
    for c in CL:
        d[c] = st.median(x["timings"]["predicted_per_second"] for x in rows if x.get("section") == "decode" and x["tag"].startswith(c + "-"))
    d["all"] = st.median(x["timings"]["predicted_per_second"] for x in rows if x.get("section") == "decode")
    for s in SIZES:
        d["read " + s] = st.median(x["timings"]["prompt_n"] / x["timings"]["prompt_ms"] * 1000 for x in rows if x.get("tag", "").startswith(f"prefill-{s}-"))
    fu = [x for x in rows if x.get("section") == "session" and x.get("tag") in ("turn1", "turn2", "turn3", "turn4", "turn5", "turn-after-side", "first-again", "other-again", "after-parallel0", "after-parallel1")]
    d["ttft follow-up"] = st.median(x["ttft_s"] for x in fu)
    d["reused %"] = 100 * sum(x["cached_tokens"] for x in fu) / sum(x["prompt_tokens"] for x in fu)
    d["errors"] = sum(1 for x in rows if x.get("error"))
    d["output"] = sum(x.get("completion_tokens") or 0 for x in rows)
    eng = LINE.findall(open(root + r + ".engine.log", errors="replace").read())
    a = o = 0
    for x in rows:
        if x.get("section") == "decode":
            for i, e in enumerate(eng):
                if int(e[0]) == x["prompt_tokens"] and int(e[2]) == x["completion_tokens"]:
                    a += int(e[4]); o += int(e[5]); eng.pop(i); break
    d["accepted %"] = 100 * a / o
    res[name] = d
keys = ["all"] + CL + ["read " + s for s in SIZES] + ["ttft follow-up", "reused %", "accepted %", "output", "errors"]
print(f"{'':16}" + "".join(f"{n:>11}" for n in GROUPS))
for k in keys:
    print(f"{k:16}" + "".join(f"{res[n][k]:11.2f}" for n in GROUPS))
print("\nidentical decode and prefill outputs between runs, same tags:")
sh = {n: {x["tag"]: x["sha"] for x in map(json.loads, open(root + r + ".jsonl")) if x.get("section") in ("decode", "prefill", "ring") and "sha" in x and not x["tag"].startswith("thinking")} for n, r in GROUPS.items()}
names = list(GROUPS)
for i in range(4):
    for j in range(i + 1, 4):
        a, b = sh[names[i]], sh[names[j]]
        print(f"{names[i]} / {names[j]}: {sum(a[k] == b[k] for k in a if k in b)} of {len(a)}")
print("\nstarted with:")
for n, r in GROUPS.items():
    t = open(root + r + ".engine.log", errors="replace").read()
    m = re.search(r"engine started: .*", t)
    args = m.group(0).split("/strata ", 1)[1]
    print(n, hash(args) & 0xffff, "vram:", [l.split(",")[2].strip() for l in open(root + r + ".vram.csv")])
