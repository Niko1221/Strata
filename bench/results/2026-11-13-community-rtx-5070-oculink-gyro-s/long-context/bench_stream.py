import json, random, sys, time, urllib.request
BASE = sys.argv[1]; MODEL = sys.argv[2]
TARGETS = [int(x) for x in sys.argv[3].split(",")] if len(sys.argv) > 3 else [30000, 60000]

def call(messages, max_tokens=400):
    body = json.dumps({"model": MODEL, "messages": messages, "max_tokens": max_tokens, "temperature": 0.6,
                       "stream": True, "stream_options": {"include_usage": True}}).encode()
    req = urllib.request.Request(BASE + "/v1/chat/completions", body, {"Content-Type": "application/json"})
    t0 = time.time(); first = None; text = ""; reason = ""; usage = {}; extra = {}
    with urllib.request.urlopen(req, timeout=3600) as r:
        for line in r:
            line = line.decode("utf-8", "replace").strip()
            if not line.startswith("data:") or line.endswith("[DONE]"): continue
            ev = json.loads(line[5:])
            if ev.get("usage"): usage = ev["usage"]
            if ev.get("timings"): extra = ev["timings"]
            for c in ev.get("choices", []):
                d = c.get("delta", {})
                piece = (d.get("content") or ""); rp = (d.get("reasoning_content") or d.get("reasoning") or "")
                if (piece or rp) and first is None: first = time.time() - t0
                text += piece; reason += rp
    total = time.time() - t0
    out = usage.get("completion_tokens") or 0
    tps = out / (total - first) if first is not None and total > first and out else None
    return {"prompt": usage.get("prompt_tokens"), "cached": (usage.get("prompt_tokens_details") or {}).get("cached_tokens"),
            "out": out, "ttft": round(first, 2) if first is not None else None, "tps": round(tps, 1) if tps else None,
            "wall": round(total, 1), "text": text, "reason_chars": len(reason), "extra": extra}

random.seed(7)
W = "amber basin cedar delta ember fjord garnet harbor iris juniper kelp lantern meadow nickel orchard pebble quartz river saffron timber umber valley willow yarrow zinc copper marble thistle canyon glacier".split()
def filler(lines, facts):
    out = []
    for i in range(lines):
        if i in facts: out.append(facts[i])
        out.append(f"Record {i}: the {random.choice(W)} {random.choice(W)} moved {random.randint(10,999)} units toward the {random.choice(W)} {random.choice(W)}.")
    return "\n".join(out)
def show(name, r):
    print(f"{name}: prompt={r['prompt']} cached={r['cached']} out={r['out']} ttft={r['ttft']}s tps={r['tps']} wall={r['wall']}s reasoning_chars={r['reason_chars']}", flush=True)

for i in range(3):
    show(f"fresh{i+1}", call([{"role": "user", "content": f"Write a detailed story about a lighthouse keeper, part {i+1}. At least 500 words."}], 600))
r = call([{"role": "user", "content": filler(500, {}) + "\n\nReply with the single word OK."}], 300)
per_line = r["prompt"] / 500
show("calib", r); print("tokens/line", round(per_line, 2), "read tok/s", round(r["prompt"] / r["ttft"]), flush=True)
for target in TARGETS:
    n = int(target / per_line)
    facts = {5: "IMPORTANT: the vault code is 48213.", n // 2: "IMPORTANT: the courier's name is Odalys Brennan."}
    msgs = [{"role": "user", "content": filler(n, facts) + "\n\nQuestion: what is the vault code and what is the courier's name? Then write a 300-word summary of what kind of data this is."}]
    r = call(msgs, 3000)
    show(f"long{target//1000}K", r)
    print("  facts found (start, middle):", ("48213" in r["text"], "Odalys" in r["text"]), "| read tok/s:", round(r["prompt"] / r["ttft"]) if r["ttft"] else None, flush=True)
    msgs += [{"role": "assistant", "content": r["text"]}, {"role": "user", "content": "Now write a 200-word limerick collection about the courier."}]
    show(f"  followup{target//1000}K", call(msgs, 600))
