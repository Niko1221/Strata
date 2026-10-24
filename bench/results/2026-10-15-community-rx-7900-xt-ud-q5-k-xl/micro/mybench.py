"""Deterministic Strata micro-benchmark for A/B tuning arms (stdlib only).

decode   : fixed coding prompts, temperature 0, exactly MAXTOK tokens (ignore EOS
           where the server allows), engine timings -> tok/s + draft acceptance
prefill  : fresh ~4K and ~16K prompts (unique nonce prefix: zero cache reuse)
followup : the 16K context again plus one short new turn -> time to first token
Prints one JSON line of medians, plus per-sample values.
"""
import json, os, statistics, sys, time, urllib.error, urllib.request, uuid

URL = "http://127.0.0.1:8080/v1/chat/completions"
KEY = os.environ["STRATA_KEY"]
MAXTOK = 256
REPS = int(os.environ.get("REPS", "2"))

DECODE_PROMPTS = [
    "Write a Python function that parses an ISO-8601 duration string like 'P3DT4H5M' into seconds. Include type hints and a docstring.",
    "Implement an LRU cache class in TypeScript with get/put in O(1), using a Map. Show the full class.",
    "Write a bash script that finds the 10 largest files under a directory given as $1, handling spaces in names.",
    "Refactor this Go code to use context cancellation and return errors instead of panicking: func fetch(u string) []byte { r, _ := http.Get(u); b, _ := io.ReadAll(r.Body); return b }",
]
FILLER = ("def handler_{i}(event, context):\n    \"\"\"Process record {i} and emit metrics.\"\"\"\n"
          "    total = sum(r['value'] for r in event.get('records', []) if r.get('ok'))\n"
          "    return {{'id': {i}, 'total': total, 'count': len(event.get('records', []))}}\n\n")


def call(messages, max_tokens, stream_first_token=False):
    body = {"model": "x", "messages": messages, "max_tokens": max_tokens, "temperature": 0,
            "top_k": 1, "seed": 42, "chat_template_kwargs": {"enable_thinking": False}}
    if stream_first_token:
        body["stream"] = True
        body["stream_options"] = {"include_usage": True}
    req = urllib.request.Request(URL, data=json.dumps(body).encode(), headers={
        "Authorization": "Bearer " + KEY, "Content-Type": "application/json"})
    t0 = time.time()
    with urllib.request.urlopen(req, timeout=1800) as r:
        if not stream_first_token:
            d = json.load(r)
            return d, time.time() - t0, None
        first = None; last = None
        for line in r:
            line = line.decode().strip()
            if not line.startswith("data:") or line == "data: [DONE]":
                continue
            ev = json.loads(line[5:])
            ch = ev.get("choices") or []
            if first is None and ch and (ch[0].get("delta") or {}).get("content"):
                first = time.time() - t0
            last = ev
        return last, time.time() - t0, first


def code_context(n_tokens_approx, nonce):
    # ~95 tokens per filler block
    blocks = max(1, n_tokens_approx // 95)
    body = "".join(FILLER.format(i=i) for i in range(blocks))
    return f"# session {nonce}\n" + body


res = {"decode_tps": [], "accept": [], "prefill4k_tps": [], "prefill16k_tps": [], "followup_ttft_s": [],
       "tool2k_ttft_s": []}

for _ in range(120):  # warm-up; the server answers 503 until its expert cache is filled
    try:
        call([{"role": "user", "content": "Say ok."}], 8)
        break
    except urllib.error.HTTPError as e:
        if e.code != 503:
            raise
        time.sleep(5)

for rep in range(REPS):
    for p in DECODE_PROMPTS:
        d, wall, _ = call([{"role": "user", "content": p}], MAXTOK)
        t = d.get("timings", {})
        res["decode_tps"].append(round(t.get("predicted_per_second", 0), 2))
        if t.get("draft_n"):
            res["accept"].append(round(t.get("draft_n_accepted", 0) / t["draft_n"], 3))
    for size, key in ((4096, "prefill4k_tps"), (16384, "prefill16k_tps")):
        nonce = uuid.uuid4().hex
        ctx = code_context(size, nonce)
        msgs = [{"role": "system", "content": "You are a coding agent."},
                {"role": "user", "content": ctx + "\nSummarise handler_3 in one line."}]
        d, wall, _ = call(msgs, 16)
        res[key].append(round(d.get("timings", {}).get("prompt_per_second", 0), 1))
        if size == 16384:
            # follow-up turn on the same (now cached) context
            msgs2 = msgs + [{"role": "assistant", "content": d["choices"][0]["message"]["content"]},
                            {"role": "user", "content": "Now write a pytest test for handler_7."}]
            _, _, first = call(msgs2, 32, stream_first_token=True)
            res["followup_ttft_s"].append(round(first or -1, 2))
            # agent-like turn: a ~2K-token tool result appended to the cached context
            tool = code_context(2000, uuid.uuid4().hex)
            msgs3 = msgs + [{"role": "assistant", "content": d["choices"][0]["message"]["content"]},
                            {"role": "user", "content": "Tool output (cat utils.py):\n" + tool + "\nWhich handler has the highest id?"}]
            _, _, first3 = call(msgs3, 32, stream_first_token=True)
            res["tool2k_ttft_s"].append(round(first3 or -1, 2))

med = {k: (statistics.median(v) if v else None) for k, v in res.items()}
print(json.dumps({"label": os.environ.get("LABEL", "?"), "median": med, "samples": res}))
