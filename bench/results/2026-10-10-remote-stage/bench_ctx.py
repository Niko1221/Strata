"""Long-context bench: prompts of increasing depth, engine timings per request.
usage: python3 bench_ctx.py <label> <corpus_file> [base_url] [targets...]   (targets in tokens, default 8000 32000 64000 100000)"""
import json, os, sys, time, urllib.request

# the server requires a key once it listens beyond localhost: env STRATA_API_KEY
KEY = os.environ.get("STRATA_API_KEY", "")

label, corpus = sys.argv[1], sys.argv[2]
base = sys.argv[3] if len(sys.argv) > 3 else "http://127.0.0.1:8080"
targets = [int(x) for x in sys.argv[4:]] or [8000, 32000, 64000, 100000]
text = open(corpus, encoding="utf-8", errors="replace").read()
CHARS_PER_TOKEN = 3.0      # measured on Strata's docs with this tokenizer (16000 chars -> 5353 tokens)


def chat(content, max_tokens):
    body = {"model": "strata", "messages": [{"role": "user", "content": content}], "max_tokens": max_tokens,
            "reasoning_effort": "none", "temperature": 0.0}
    req = urllib.request.Request(base + "/v1/chat/completions", json.dumps(body).encode(),
                                 {"Content-Type": "application/json", **({"Authorization": "Bearer " + KEY} if KEY else {})})
    t = time.time()
    try:
        r = json.load(urllib.request.urlopen(req, timeout=7200))
    except urllib.error.HTTPError as e:
        return None, time.time() - t, e.read().decode(errors="replace")[:300]
    return r, time.time() - t, None


for i, tgt in enumerate(targets):
    # a different slice per request so the conversation cache cannot reuse an earlier prompt
    n = int(tgt * CHARS_PER_TOKEN)
    start = (i * 7919) % max(1, len(text) - n) if len(text) > n else 0
    doc = (text * (1 + n // max(1, len(text))))[start:start + n]
    q = ("Here is a long document. After reading it, write a detailed technical summary of its main ideas in about "
         "250 words.\n\n<document>\n" + doc + "\n</document>")
    r, wall, err = chat(q, 320)
    if err:
        print(json.dumps({"label": label, "target": tgt, "error": err, "wall_s": round(wall, 1)}), flush=True)
        continue
    tm = r.get("timings") or {}
    print(json.dumps({"label": label, "target": tgt, "prompt_n": tm.get("prompt_n"), "cache_n": tm.get("cache_n"),
                      "prefill_tps": tm.get("prompt_per_second"), "prefill_s": round((tm.get("prompt_ms") or 0) / 1000, 1),
                      "gen_n": tm.get("predicted_n"), "decode_tps": tm.get("predicted_per_second"),
                      "draft": f'{tm.get("draft_n_accepted")}/{tm.get("draft_n")}', "wall_s": round(wall, 1),
                      "head": (r["choices"][0]["message"].get("content") or "")[:100].replace("\n", " ")},
                     ensure_ascii=False), flush=True)
