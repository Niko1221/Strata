#!/usr/bin/env python3
"""Fixed-prompt benchmark for DeepSeek-V4.1-Flash (prompt set v1, byte-identical across boots).

Per request: streaming with stream_options.include_usage. TTFT = arrival of the first token delta;
completion tokens = the server's usage block (never counted from chunks: speculative decoding packs
several tokens into one chunk). Every request carries a deterministic tag at the FRONT, unique within a
run (no prefix-cache hits) and identical across boots. Temperature 0, thinking off.

usage: v41bench.py --base http://127.0.0.1:8000/v1 --model deepseek-v4.1-flash --label boot4 --out DIR
       [--levels 1,2,3,4,5,6] [--prefill 2000,8000,32000,64000] [--notes "..."]
       v41bench.py --dump-prompts prompts-v1.json
"""
import argparse, json, random, statistics as st, threading, time, urllib.request
from concurrent.futures import ThreadPoolExecutor

PROMPT_SET_VERSION = "v1"
SUMMARY_PASSAGE = (
    "Grid-scale batteries have moved from pilot projects to a routine part of how electricity systems run. "
    "Their main job is to shift energy in time: they charge when solar and wind output is high and prices are low, "
    "then discharge in the evening peak when demand climbs and those sources fade. Operators also use them for "
    "services that last seconds rather than hours, such as holding grid frequency steady when a large power plant "
    "trips offline, because a battery can respond far faster than a spinning turbine. Most installations today use "
    "lithium-ion cells, which fell sharply in price over the last decade, but they are best suited to durations of "
    "two to four hours. Longer gaps, such as a windless winter week, need other approaches: flow batteries, "
    "compressed air, pumped hydro, or simply more transmission lines that move power between regions with different "
    "weather. Critics point out that batteries do not generate energy and lose a portion of it on every cycle, and "
    "that mining the metals they contain carries environmental costs. Supporters answer that the alternative, "
    "keeping fossil plants idling as backup, is more expensive and dirtier, and that recycling programs are "
    "starting to recover lithium, nickel and cobalt at scale. The practical consensus among grid planners is that "
    "batteries are one tool among several, most valuable where they are placed close to congestion points or large "
    "solar farms, and least valuable when asked to cover long, rare shortages on their own."
)
CATEGORIES = [
    ("coding", "Write a Python function merge_intervals(intervals) that merges overlapping intervals and returns them sorted. Include a one-line docstring and two example calls.", 200),
    ("json", "Return only a JSON object describing a fictional bookstore with keys: name (string), city (string), founded (integer year), genres (array of 5 strings), staff (array of 3 objects, each with name and role). No prose.", 200),
    ("narrative", "Write a 120-word short story about a lighthouse keeper who finds a message in a bottle. Use vivid sensory detail.", 200),
    ("prose", "Explain in about 120 words how a refrigerator keeps food cold, for a curious 12-year-old.", 200),
    ("math", "A train leaves at 9:40. It travels 212 km at 80 km/h, stops for 12 minutes, then travels 95 km at 60 km/h. At what time does it arrive? Show the steps briefly, then give the final time.", 200),
    ("reasoning", "Ana, Ben and Cal each own exactly one pet: a cat, a dog, or a fish. Ana does not own the dog. Ben owns neither the cat nor the fish. Who owns which pet? Explain the deduction step by step.", 200),
    ("summary", "Summarize the following passage in exactly three bullet points.\n\n" + SUMMARY_PASSAGE, 150),
    ("format", "Convert this list into a Markdown table with columns Item, Qty, Price, Total (Qty x Price), and add a final row with the grand total: apples 3 @ 0.50; bread 1 @ 2.25; milk 2 @ 1.10; eggs 12 @ 0.25.", 200),
]
CEILING = ("ceiling_count", "Count from 1 to 80, comma separated, nothing else.", 256)
FILLER_WORDS = ("amber basin cedar delta ember fjord garnet harbor iris juniper kestrel lumen meadow nimbus orchid "
                "pylon quartz raven sierra tundra umber vessel willow xenon yarrow zephyr").split()

def stream_chat(base, model, prompt, max_tokens, timeout=900):
    body = {"model": model, "messages": [{"role": "user", "content": prompt}], "max_tokens": max_tokens,
            "temperature": 0, "stream": True, "stream_options": {"include_usage": True},
            "chat_template_kwargs": {"thinking": False, "enable_thinking": False}}
    req = urllib.request.Request(base + "/chat/completions", data=json.dumps(body).encode(),
                                 headers={"Content-Type": "application/json"})
    t0 = time.time(); ttft = None; usage = None; chars = 0
    with urllib.request.urlopen(req, timeout=timeout) as r:
        for raw in r:
            line = raw.decode("utf-8", "ignore").strip()
            if not line.startswith("data:"):
                continue
            data = line[5:].strip()
            if data == "[DONE]":
                break
            ev = json.loads(data)
            if ev.get("usage"):
                usage = ev["usage"]
            for ch in ev.get("choices") or []:
                d = ch.get("delta") or {}
                piece = (d.get("content") or "") + (d.get("reasoning") or d.get("reasoning_content") or "")
                if piece:
                    if ttft is None:
                        ttft = time.time() - t0
                    chars += len(piece)
    total = time.time() - t0
    ct = (usage or {}).get("completion_tokens", 0); pt = (usage or {}).get("prompt_tokens", 0)
    dec = (ct - 1) / (total - ttft) if (ttft is not None and ct > 1 and total > ttft) else None
    return {"ttft_s": ttft, "total_s": total, "completion_tokens": ct, "prompt_tokens": pt, "decode_tok_s": dec, "chars": chars}

def run_batch(args, c, cat, prompt, max_tokens, tag):
    barrier = threading.Barrier(c)
    def one(i):
        p = f"[bench {PROMPT_SET_VERSION} {tag} {cat} c{c} s{i}] " + prompt
        barrier.wait()
        return stream_chat(args.base, args.model, p, max_tokens)
    t0 = time.time()
    with ThreadPoolExecutor(max_workers=c) as ex:
        res = list(ex.map(one, range(c)))
    wall = time.time() - t0
    toks = sum(r["completion_tokens"] for r in res)
    decs = [r["decode_tok_s"] for r in res if r["decode_tok_s"]]
    ttfts = [r["ttft_s"] for r in res if r["ttft_s"] is not None]
    return {"c": c, "category": cat, "wall_s": round(wall, 3), "tokens": toks,
            "agg_tok_s": round(toks / wall, 2) if wall else 0,
            "per_stream_tok_s": round(st.mean(decs), 2) if decs else None,
            "ttft_mean_s": round(st.mean(ttfts), 3) if ttfts else None, "requests": res}

def filler(n_words, seed):
    rnd = random.Random(seed)
    return " ".join(f"{rnd.choice(FILLER_WORDS)}{rnd.randint(0, 999)}" for _ in range(n_words))

def md_table(headers, rows):
    out = ["| " + " | ".join(headers) + " |", "|" + "|".join("---" for _ in headers) + "|"]
    out += ["| " + " | ".join("" if v is None else str(v) for v in r) + " |" for r in rows]
    return "\n".join(out)

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--base", default="http://127.0.0.1:8000/v1"); ap.add_argument("--model", default="deepseek-v4.1-flash")
    ap.add_argument("--label", default="run"); ap.add_argument("--out", default=".")
    ap.add_argument("--levels", default="1,2,3,4,5,6"); ap.add_argument("--prefill", default="2000,8000,32000,64000")
    ap.add_argument("--notes", default=""); ap.add_argument("--dump-prompts")
    args = ap.parse_args()
    if args.dump_prompts:
        json.dump({"version": PROMPT_SET_VERSION, "temperature": 0, "thinking": False,
                   "categories": [{"name": n, "prompt": p, "max_tokens": m} for n, p, m in CATEGORIES],
                   "ceiling": {"name": CEILING[0], "prompt": CEILING[1], "max_tokens": CEILING[2]},
                   "prefill_filler": {"words": FILLER_WORDS, "seed_rule": "seed = target tokens", "words_per_target_token": 0.55}},
                  open(args.dump_prompts, "w"), indent=1); print("wrote", args.dump_prompts); return
    levels = [int(x) for x in args.levels.split(",")]; pre = [int(x) for x in args.prefill.split(",") if x]
    res = {"label": args.label, "prompt_set": PROMPT_SET_VERSION, "model": args.model, "notes": args.notes,
           "started": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()), "batches": [], "prefill": []}
    print(f"[{args.label}] warmup", flush=True)
    for n, p, m in CATEGORIES[:3]:
        run_batch(args, 1, n, p, 64, "warm")
    for c in levels:
        for n, p, m in CATEGORIES + [CEILING]:
            b = run_batch(args, c, n, p, m, "run"); res["batches"].append(b)
            print(f"  C{c} {n:13} agg {b['agg_tok_s']:7.2f} tok/s  per-stream {b['per_stream_tok_s']}  ttft {b['ttft_mean_s']}s", flush=True)
    for tgt in pre:
        r = stream_chat(args.base, args.model, f"[bench {PROMPT_SET_VERSION} prefill {tgt}] " + filler(int(tgt * 0.55), tgt)
                        + "\n\nReply with the single word OK.", 1, timeout=1800)
        row = {"target": tgt, "prompt_tokens": r["prompt_tokens"], "ttft_s": round(r["total_s"], 3),
               "prefill_tok_s": round(r["prompt_tokens"] / r["total_s"], 1) if r["total_s"] else None}
        res["prefill"].append(row); print(f"  prefill {tgt:>6}: {row}", flush=True)
    res["finished"] = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
    head = []
    for c in levels:
        bs = [b for b in res["batches"] if b["c"] == c and b["category"] != CEILING[0]]
        head.append([f"C{c}", round(st.mean(b["agg_tok_s"] for b in bs), 2),
                     round(st.mean(b["per_stream_tok_s"] for b in bs if b["per_stream_tok_s"]), 2),
                     round(st.mean(b["ttft_mean_s"] for b in bs if b["ttft_mean_s"] is not None), 3)])
    res["headline"] = [dict(zip(["level", "agg_tok_s", "per_stream_tok_s", "ttft_mean_s"], h)) for h in head]
    cat_rows = [[n] + [next((b["per_stream_tok_s"] for b in res["batches"] if b["c"] == c and b["category"] == n), None) for c in levels]
                for n, _, _ in CATEGORIES + [CEILING]]
    md = [f"## {args.label} ({res['started']})", "", args.notes, "",
          f"Prompt set `{PROMPT_SET_VERSION}` (identical across boots), temperature 0, thinking off. "
          "Tokens from the server's usage block; TTFT = first token delta.", "",
          "### Throughput by concurrency (8 categories; the counting ceiling is excluded)", "",
          md_table(["C", "aggregate tok/s", "per-stream tok/s", "mean TTFT (s)"], head), "",
          "### Per-stream tok/s by category", "", md_table(["category"] + [f"C{c}" for c in levels], cat_rows), "",
          "### Cold prefill (unique prefix)", "",
          md_table(["target", "prompt tokens", "TTFT (s)", "prefill tok/s"], [[p["target"], p["prompt_tokens"], p["ttft_s"], p["prefill_tok_s"]] for p in res["prefill"]])]
    json.dump(res, open(f"{args.out}/bench-{args.label}.json", "w"), indent=1)
    open(f"{args.out}/bench-{args.label}.md", "w").write("\n".join(md) + "\n")
    print("\n".join(md[4:13])); print(f"wrote {args.out}/bench-{args.label}.json and .md", flush=True)

if __name__ == "__main__":
    main()
