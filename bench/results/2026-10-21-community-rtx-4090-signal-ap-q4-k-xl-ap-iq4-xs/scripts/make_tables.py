"""Rebuild the tables in README.md from runs/*/results.json, runs/*/pre-run-state.txt and ab/*/ab_engine.jsonl.

    python3 -I scripts/make_tables.py summary|perrun|start|ab        (run from this folder or anywhere: paths are relative to the script)

Only requests with ok == true count. "Agent turns" are turns 2-6 of each 6-turn session (the first turn has no earlier turn to reuse).
The statistic is the median over requests, with the lowest and the highest request as the range, unless a table says "median of the runs"
(the median of the runs' own medians, which is the figure docs/SIGNAL_AP_Q4_K_XL.md quotes)."""
import json, os, re, statistics as st, sys

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE) if os.path.basename(HERE) == "scripts" else HERE
RUNS = os.path.join(ROOT, "runs")

CONFIGS = [   # (name, labels)
    ("A  AP-Q4_K_XL, config setup writes", ["pass1-A-setup-ap-q4_k_xl", "pass2-A-setup-ap-q4_k_xl", "pass3-A-setup-ap-q4_k_xl"]),
    ("B  AP-IQ4_XS, config setup writes", ["pass1-B-setup-ap-iq4_xs", "pass2-B-setup-ap-iq4_xs", "pass3-B-setup-ap-iq4_xs"]),
    ("C  AP-Q4_K_XL, the document's hand-built config", ["pass1-C-docs-handbuilt-ap-q4_k_xl", "pass2-C-docs-handbuilt-ap-q4_k_xl",
                                                         "pass3-C-docs-handbuilt-ap-q4_k_xl", "pass3-C2-docs-handbuilt-ap-q4_k_xl"]),
    ("C with --kv int8 (the same config as C, two more runs)", ["kv-1-int8", "kv-2-int8"]),
    ("C with --kv q4_0", ["kv-1-q4_0", "kv-2-q4_0"]),
]

def load(label):
    return json.load(open(os.path.join(RUNS, label, "results.json"), encoding="utf-8"))["results"]

def ok(reqs): return [q for q in reqs if q.get("ok")]
def agent_turns(r): return [t for s in r["agent"]["sessions"] for t in s if t.get("ok") and t.get("turn", 0) >= 1]

def series(label):
    r = load(label)
    chat, p4, p16, ag = ok(r["chat"]["requests"]), ok(r["prompts"]["4K"]["requests"]), ok(r["prompts"]["16K"]["requests"]), agent_turns(r)
    g = lambda L, k: [q[k] for q in L if q.get(k) is not None]
    return {"chat": chat, "4K": p4, "16K": p16, "agent": ag, "failures": len(r["failures"]), "hit": r["agent"]["hit_pct"]["median"]}

def mr(v, f):   # "median (min-max)"
    return f"{f.format(st.median(v))} ({f.format(min(v))}-{f.format(max(v))})"

def summary():
    print("| Configuration | Test | Requests | Actual prompt tokens | Reused tokens | Generated tokens | Prompt tok/s median (range) | Decode tok/s median (range) | First token s median (range) |")
    print("| --- | --- | ---: | --- | --- | --- | --- | --- | --- |")
    for name, labels in CONFIGS:
        S = [series(l) for l in labels]
        for key, test in (("chat", "short chat"), ("4K", "4K prompt, fresh"), ("16K", "16K prompt, fresh"), ("agent", "agent session, turns 2-6")):
            L = [q for s in S for q in s[key]]
            col = lambda k: [q[k] for q in L if q.get(k) is not None]
            prompt = mr(col("prompt_tps"), "{:.0f}") if key in ("4K", "16K") else "not measured"
            dec = mr(col("decode_tps"), "{:.1f}") if key in ("chat", "agent") else "not measured"
            print(f"| {name} | {test} | {len(L)} | {mr(col('prompt_tokens'), '{:.0f}')} | {mr(col('reused_tokens'), '{:.0f}')} | {mr(col('generated_tokens'), '{:.0f}')} | {prompt} | {dec} | {mr(col('ttft_s'), '{:.2f}')} |")

def perrun():
    for name, labels in CONFIGS:
        S = [series(l) for l in labels]
        print(f"\n**{name}** ({len(labels)} runs, failed requests: {sum(s['failures'] for s in S)})\n")
        print("| test | median of the runs | each run's median | slowest to fastest request |\n| --- | --- | --- | --- |")
        rows = [("short chat, decode (tokens/s)", "chat", "decode_tps", "{:.1f}"), ("short chat, first token (s)", "chat", "ttft_s", "{:.2f}"),
                ("4K prompt (tokens/s)", "4K", "prompt_tps", "{:.0f}"), ("4K prompt, first token (s)", "4K", "ttft_s", "{:.2f}"),
                ("16K prompt (tokens/s)", "16K", "prompt_tps", "{:.0f}"), ("16K prompt, first token (s)", "16K", "ttft_s", "{:.2f}"),
                ("agent turns, decode (tokens/s)", "agent", "decode_tps", "{:.1f}"), ("agent turns, first token (s)", "agent", "ttft_s", "{:.2f}")]
        for title, key, k, f in rows:
            meds = [st.median([q[k] for q in s[key] if q.get(k) is not None]) for s in S]
            allv = [q[k] for s in S for q in s[key] if q.get(k) is not None]
            print(f"| {title} | {f.format(st.median(meds))} | {', '.join(f.format(m) for m in meds)} | {f.format(min(allv))}-{f.format(max(allv))} (n={len(allv)}) |")
        print(f"| agent turns, prompt reused from the cache (%) | {st.median([s['hit'] for s in S]):.1f} | {', '.join(f'{s['hit']:.1f}' for s in S)} | |")

def start():
    print("| run | started (PDT, 2026-10-21) | 1-minute load average at start | available memory at start (MiB) | GPU memory in use at start | seconds measured |")
    print("| --- | --- | ---: | ---: | --- | ---: |")
    for _, labels in CONFIGS:
        for l in labels:
            t = open(os.path.join(RUNS, l, "pre-run-state.txt"), encoding="utf-8").read()
            started = re.search(r"^start: \S+T(\d\d:\d\d:\d\d)", t, re.M).group(1)
            load1 = re.search(r"loadavg: ([\d.]+)", t).group(1)
            avail = re.search(r"^Mem:\s+(?:\d+\s+){5}(\d+)", t, re.M).group(1)
            gpu = re.search(r"GeForce RTX 4090, [\d.]+, (\d+) MiB", t).group(1)
            secs = json.load(open(os.path.join(RUNS, l, "results.json"), encoding="utf-8"))["elapsed_s"]
            print(f"| {l} | {started} | {load1} | {avail} | {gpu} MiB | {secs:.0f} |")

def ab():
    for d in sorted(os.listdir(os.path.join(ROOT, "ab"))):
        rows = [json.loads(l) for l in open(os.path.join(ROOT, "ab", d, "ab_engine.jsonl"), encoding="utf-8")]
        arms = sorted({r["arm"] for r in rows}, key=lambda a: (a != "default", a))
        by = {(r["round"], r["req"], r["arm"]): r for r in rows}
        pairs = [(by.get((rd, q, arms[0])), by.get((rd, q, arms[1]))) for rd in sorted({r["round"] for r in rows}) for q in sorted({r["req"] for r in rows})]
        pairs = [(x, y) for x, y in pairs if x and y and x["decode"] and y["decode"]]
        wins = sum(1 for x, y in pairs if x["decode"] > y["decode"])
        med = {a: st.median([r["decode"] for r in rows if r["arm"] == a and r["decode"]]) for a in arms}
        n = {a: len([r for r in rows if r["arm"] == a and r["decode"]]) for a in arms}
        print(f"| {d} | {arms[0]} {med[arms[0]]:.1f} (n={n[arms[0]]}) | {arms[1]} {med[arms[1]]:.1f} (n={n[arms[1]]}) | {arms[0]} faster in {wins} of {len(pairs)} pairs |")

if __name__ == "__main__":
    {"summary": summary, "perrun": perrun, "start": start, "ab": ab}[sys.argv[1]]()
