# Strata community benchmark - conforme a docs/COMMUNITY_BENCHMARKS.md
# Misure separate: prefill fresco (nonce anti-cache), decode (timings server, NON
# token/secondi totali), TTFT in streaming (primo token = reasoning, dichiarato),
# richieste parallele, needle di correttezza. Nessuna credenziale nello script:
# la chiave arriva da env STRATA_API_KEY (o --api-key). Uso:
#   python bench.py --url http://127.0.0.1:8080 --runs 3 --config iq2_xs --outdir runs
import json, os, sys, time, argparse, statistics, uuid
import concurrent.futures
import requests

def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--url", default="http://127.0.0.1:8080")
    p.add_argument("--api-key", default=os.environ.get("STRATA_API_KEY", ""))
    p.add_argument("--runs", type=int, default=3)
    p.add_argument("--config", default="run", help="etichetta configurazione (es. iq2_xs-single)")
    p.add_argument("--outdir", default="runs")
    p.add_argument("--skip-needle", action="store_true")
    return p.parse_args()

A = parse_args()
S = requests.Session(); S.trust_env = False
if A.api_key:
    S.headers["Authorization"] = "Bearer " + A.api_key
os.makedirs(A.outdir, exist_ok=True)

PARA = ("Systems engineers measure latency and throughput when they design inference "
        "pipelines for large language models running on consumer hardware with memory "
        "bandwidth limits. The quick brown fox jumps over the lazy dog near the river. ")

def filler(target_tokens, nonce):
    txt = nonce + "\n" + PARA * (target_tokens * 4 // len(PARA) + 1)
    return txt[:target_tokens * 4]

def chat(messages, max_tokens, timeout=1800):
    t0 = time.time()
    r = S.post(A.url + "/v1/chat/completions",
               json={"model": "x", "messages": messages, "max_tokens": max_tokens, "temperature": 0},
               timeout=timeout)
    wall = time.time() - t0
    r.raise_for_status()
    d = r.json()
    u, t = d.get("usage", {}), d.get("timings", {})
    return {"wall_s": round(wall, 2),
            "prompt_tokens": u.get("prompt_tokens"),
            "reused_tokens": (u.get("prompt_tokens_details") or {}).get("cached_tokens", 0),
            "generated_tokens": u.get("completion_tokens"),
            "fresh_prompt_tokens": u.get("prompt_tokens", 0) - (u.get("prompt_tokens_details") or {}).get("cached_tokens", 0),
            "prompt_tok_s": t.get("prompt_per_second"),
            "decode_tok_s": t.get("predicted_per_second"),
            "ttft_s": None, "finish": d["choices"][0]["finish_reason"]}

def ttft_test():
    # streaming: primo token (e' reasoning, il modello e' tipo ibrido thinking)
    q = uuid.uuid4().hex
    t0 = time.time(); first = None; deltas = 0; utoks = 0
    with S.post(A.url + "/v1/chat/completions",
                json={"model": "x", "max_tokens": 512, "temperature": 0, "stream": True,
                      "stream_options": {"include_usage": True},
                      "messages": [{"role": "user", "content":
                          "Explain in a few sentences why memory bandwidth matters for LLM decode. " + q}]},
                stream=True, timeout=1800) as r:
        for line in r.iter_lines():
            if not line or not line.startswith(b"data: "): continue
            payload = line[6:].strip()
            if payload == b"[DONE]": continue
            try: j = json.loads(payload)
            except json.JSONDecodeError: continue
            if j.get("usage") and j["usage"].get("completion_tokens"): utoks = j["usage"]["completion_tokens"]
            for ch in j.get("choices", []):
                dd = ch.get("delta", {})
                if dd.get("content") or dd.get("reasoning_content"):
                    deltas += 1
                    if first is None: first = time.time() - t0
    return {"ttft_s": round(first, 3) if first else None,
            "first_token_is_reasoning": True, "generated_tokens": utoks or deltas,
            "decode_tok_s_stream": round((utoks or deltas) / (time.time() - t0 - (first or 0)), 1) if first else None,
            "note": "TTFT = primo token reasoning (modello ibrido), streaming, keep-alive esclusi"}

def parallel_test():
    tasks = ["Explain in 3 sentences what a MOSFET does.",
             "Write a Python function that checks if a string is a palindrome.",
             "List 4 tips for writing clean commit messages."]
    def one(p):
        t0 = time.time()
        d = chat([{"role": "user", "content": p + " " + uuid.uuid4().hex}], 256)
        d["wall_s"] = round(time.time() - t0, 2)
        return d
    t0 = time.time()
    with concurrent.futures.ThreadPoolExecutor(3) as ex:
        res = list(ex.map(one, tasks))
    return {"clients": 3, "total_wall_s": round(time.time() - t0, 2),
            "per_request": res,
            "note": "server single-slot: le richieste vengono accodate; wall include attesa in coda"}

def needle_test():
    nonce = uuid.uuid4().hex
    code = "ZEBRA-" + nonce[:4].upper()
    body = ("IMPORTANT NOTE: the secret access code for today is " + code + ".\n"
            + filler(40000, nonce))
    r = S.post(A.url + "/v1/chat/completions",
               json={"model": "x", "temperature": 0, "max_tokens": 512,
                     "messages": [{"role": "user", "content": body + "\n\nWhat is the secret access code "
                                  "mentioned at the very beginning? Reply with just the code."}]}, timeout=1800)
    d = r.json(); u = d.get("usage", {}); t = d.get("timings", {})
    txt = json.dumps(d["choices"][0]["message"])
    return {"needle": code, "found": code in txt,
            "prompt_tokens": u.get("prompt_tokens"),
            "reused_tokens": (u.get("prompt_tokens_details") or {}).get("cached_tokens", 0),
            "prompt_tok_s": t.get("prompt_per_second"), "decode_tok_s": t.get("predicted_per_second"),
            "note": "codice segreto in testa a ~42K token; found=True se il codice compare nella risposta"}

def one_run(run_idx):
    res = {"run": run_idx, "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S")}
    print(f"[{A.config}] run {run_idx}: TTFT...", flush=True)
    res["ttft"] = ttft_test()
    for n in (2600, 10000, 42000):
        print(f"[{A.config}] run {run_idx}: prefill ~{n} tok...", flush=True)
        nonce = uuid.uuid4().hex
        d = chat([{"role": "user", "content": filler(n, nonce) + "\n\nReply with exactly: OK"}], 128)
        res[f"prefill_{n}"] = d
    print(f"[{A.config}] run {run_idx}: 3 client paralleli...", flush=True)
    res["parallel"] = parallel_test()
    if not A.skip_needle:
        print(f"[{A.config}] run {run_idx}: needle 42K...", flush=True)
        res["needle"] = needle_test()
    return res

def aggregate(all_runs):
    agg = {}
    for k in ("ttft",):
        vals = [r[k]["ttft_s"] for r in all_runs if r.get(k, {}).get("ttft_s") is not None]
        if vals: agg["ttft_s"] = {"median": statistics.median(vals), "min": min(vals), "max": max(vals)}
    for n in (2600, 10000, 42000):
        key = f"prefill_{n}"
        for metric in ("prompt_tok_s", "fresh_prompt_tokens", "decode_tok_s"):
            vals = [r[key][metric] for r in all_runs if r.get(key, {}).get(metric) is not None]
            if vals: agg.setdefault(key, {})[metric] = {"median": round(statistics.median(vals), 1),
                                                        "min": round(min(vals), 1), "max": round(max(vals), 1)}
    return agg

runs = []
for i in range(1, A.runs + 1):
    runs.append(one_run(i))
    json.dump(runs, open(os.path.join(A.outdir, f"{A.config}_runs.json"), "w"), indent=1)
summary = {"config": A.config, "runs": A.runs, "aggregate": aggregate(runs),
           "failed_runs": [r["run"] for r in runs if r.get("error")]}
json.dump(summary, open(os.path.join(A.outdir, f"{A.config}_summary.json"), "w"), indent=1)
print(json.dumps(summary["aggregate"], indent=1))
print("Salvato in", A.outdir)
