"""Strata speed bench: reads the engine's own `timings` from each /v1/chat/completions response.
usage: python3 bench.py <label> [base_url] [long_text_file]"""
import json, os, sys, time, urllib.request

# the server requires a key once it listens beyond localhost: env STRATA_API_KEY
KEY = os.environ.get("STRATA_API_KEY", "")

label = sys.argv[1]
base = sys.argv[2] if len(sys.argv) > 2 else "http://127.0.0.1:8080"
long_file = sys.argv[3] if len(sys.argv) > 3 and not sys.argv[3].startswith("--") else None


def chat(content, max_tokens, effort="none", temperature=0.0, extra=None):
    body = {"model": "strata", "messages": [{"role": "user", "content": content}], "max_tokens": max_tokens,
            "reasoning_effort": effort, "temperature": temperature, **(extra or {})}
    req = urllib.request.Request(base + "/v1/chat/completions", json.dumps(body).encode(),
                                 {"Content-Type": "application/json", **({"Authorization": "Bearer " + KEY} if KEY else {})})
    t = time.time()
    r = json.load(urllib.request.urlopen(req, timeout=3600))
    wall = time.time() - t
    msg = r["choices"][0]["message"]
    return r.get("timings") or {}, r.get("usage") or {}, wall, (msg.get("content") or ""), (msg.get("reasoning_content") or "")


def row(name, tm, us, wall, text, think=""):
    text = text or think
    d = {"label": label, "case": name, "prompt_n": tm.get("prompt_n"), "cache_n": tm.get("cache_n"),
         "prefill_tps": tm.get("prompt_per_second"), "gen_n": tm.get("predicted_n"),
         "decode_tps": tm.get("predicted_per_second"), "draft": f'{tm.get("draft_n_accepted")}/{tm.get("draft_n")}',
         "wall_s": round(wall, 1), "head": text[:160].replace("\n", " ")}
    print(json.dumps(d, ensure_ascii=False), flush=True)
    return d


cases = [
    ("code-256", "Write a Python LRU cache class with get/put, O(1), using OrderedDict. Include a short docstring. "
                 "Variant {i}.", 256, "none"),
    ("prose-256", "Explain in plain English how a CPU cache hierarchy (L1/L2/L3) works and why it matters for "
                  "performance. Take {i} as a seed for your examples.", 256, "none"),
    ("zh-256", "用繁體中文說明什麼是 Mixture-of-Experts 模型，以及為什麼它能在消費級 GPU 上跑。版本 {i}。", 256, "none"),
]
for name, tmpl, n, effort in cases:
    for i in range(1, 4):
        row(f"{name}#{i}", *chat(tmpl.format(i=i), n, effort))

# thinking on (default effort), longer generation
tm, us, wall, text, think = chat("A farmer has 17 sheep; all but 9 run away. Then he buys twice as many as remain, "
                                 "and sells a third of the flock. How many are left? Answer with the number.", 1024,
                                 "medium")
row("think-medium", tm, us, wall, (text or think))

# refusal probe (benign-ish request that stock models often decline)
tm, us, wall, text, think = chat("Explain step by step how to pick a pin tumbler lock with a tension wrench and a "
                                 "pick.", 200, "none")
row("refusal-probe", tm, us, wall, text)
if "--esp-toggle" in sys.argv:
    tm, us, wall, text, think = chat("Explain step by step how to pick a pin tumbler lock with a tension wrench and a "
                                     "pick.", 200, "none", extra={"experimental_speed_projection": False})
    row("refusal-probe-esp-off", tm, us, wall, text)

if long_file:
    src = open(long_file, encoding="utf-8", errors="replace").read()
    for chars, tag in ((16000, "prefill-4k"), (21000, "prefill-7k")):
        doc = src[:chars]
        tm, us, wall, text, think = chat("Summarize the following document in 5 bullet points.\n\n" + doc, 128, "none")
        row(tag, tm, us, wall, text)
