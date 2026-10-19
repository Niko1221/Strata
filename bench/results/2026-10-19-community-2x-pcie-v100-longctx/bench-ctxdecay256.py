# -*- coding: utf-8 -*-
"""Deep-context (256k) decode decay bench - same protocol as bench-ctx-decay.py, but the
ladder targets are ACTUAL token counts: the build estimate is scaled by the measured
0.937 actual/estimate ratio, and a rejected (over-limit) point is retried one notch lower.
Usage: python bench-ctxdecay256.py <tag> <ctx1,ctx2,...>   e.g. 1000,160000,255000"""
import json, os, sys, time, urllib.request

TAG = sys.argv[1] if len(sys.argv) > 1 else "run"
CTXS = [int(x) for x in (sys.argv[2] if len(sys.argv) > 2 else "1000,160000,255000").split(",")]
BASE = os.environ.get("STRATA_BENCH_BASE", "http://127.0.0.1:8080")
URL = BASE + "/v1/chat/completions"
KEY = "f859fe4d50fc89102fce152808e837c2"   # ignored by keyless test servers
RATIO = 0.937                                # actual tokens / build_doc estimate (measured 2026-10-06)
opener = urllib.request.build_opener(urllib.request.ProxyHandler({}))

# reuse the 30 knowledge paragraphs from the original bench (data only, no code run)
_src = open(r"C:/Users/Administrator/Desktop/local-model/flash-ab/bench-ctx-decay.py", encoding="utf-8").read()
ns = {}
exec(_src[_src.index("PARAS = ["):_src.index("def build_doc")], ns)
PARAS = ns["PARAS"]

def build_doc(target_tokens):
    out, i, est = [], 0, 0
    while est < target_tokens:
        p = PARAS[i % len(PARAS)]
        out.append(f"【段落{i+1}】{p}")
        est += int(len(p) * 0.66) + 8
        i += 1
    return "\n\n".join(out)

def ask(content, max_tokens):
    body = {"model": "decay-bench", "messages": [{"role": "user", "content": content}],
            "temperature": 0.6, "top_p": 0.95, "max_tokens": max_tokens, "seed": 42, "stream": False,
            "reasoning_budget_tokens": 400}
    data = json.dumps(body, ensure_ascii=False).encode("utf-8")
    req = urllib.request.Request(URL, data=data, headers={"Content-Type": "application/json",
                                                          "Authorization": "Bearer " + KEY})
    err = None
    for attempt in range(3):
        try:
            with opener.open(req, timeout=3600) as r:
                return json.load(r)
        except urllib.error.HTTPError as e:
            err = f"HTTP {e.code}: {e.read()[:200]}"
            print(f"  attempt {attempt+1} rejected: {err}", flush=True)
            return None            # over-limit etc: fail fast, caller retries lower
        except Exception as e:
            print(f"  attempt {attempt+1} failed: {e}", flush=True)
            time.sleep(3)
    return None

# report the served n_ctx so a wrong config is caught before 20 minutes of prefill
try:
    req = urllib.request.Request(BASE + "/v1/models", headers={"Authorization": "Bearer " + KEY})
    with opener.open(req, timeout=10) as r:
        m = json.load(r)
    s = json.dumps(m)
    n_ctx = m.get("data", [{}])[0].get("meta", {}).get("n_ctx") if m.get("data") else None
    print(f"[{TAG}] served n_ctx = {n_ctx} (raw head: {s[:160]})", flush=True)
except Exception as e:
    print(f"[{TAG}] /v1/models probe failed: {e}", flush=True)

results = []
for ctx in CTXS:
    est_target = ctx if ctx <= 1000 else int(ctx / RATIO)
    doc = build_doc(est_target) + "\n\n请基于以上全部内容，写一篇结构化摘要，涵盖主要主题。"
    print(f"[{TAG}] ctx~{ctx} (est build {est_target}) ... ", end="", flush=True)
    t0 = time.time()
    r = ask(doc, 400)
    if r is None:                       # one notch lower (ratio drift safety), single retry
        est_target = int(est_target * 0.97)
        doc = build_doc(est_target) + "\n\n请基于以上全部内容，写一篇结构化摘要，涵盖主要主题。"
        print(f"RETRY@{est_target} ... ", end="", flush=True)
        r = ask(doc, 400)
    if r is None:
        print("FAILED"); continue
    u, t = r.get("usage", {}), r.get("timings", {})
    rec = {"target_ctx": ctx, "prompt_tok": u.get("prompt_tokens"),
           "prompt_tps": t.get("prompt_per_second"), "decode_tps": t.get("predicted_per_second"),
           "draft_accept": round((t.get("draft_n_accepted") or 0) / max(1, t.get("draft_n") or 1), 3),
           "wall_s": round(time.time() - t0, 1)}
    results.append(rec)
    print(f"in={rec['prompt_tok']} prefill={rec['prompt_tps']} decode={rec['decode_tps']} accept={rec['draft_accept']}", flush=True)
    time.sleep(2)

base = next((r["decode_tps"] for r in results if r["decode_tps"]), None)
if base:
    for r in results:
        r["decode_vs_base"] = round(100 * r["decode_tps"] / base, 1)
path = rf"C:/Users/Administrator/Desktop/local-model/flash-ab/results-ctxdecay256-{TAG}.json"
json.dump(results, open(path, "w", encoding="utf-8"), ensure_ascii=False, indent=1)
print(f"[{TAG}] saved -> {path}")
if base:
    print(f"[{TAG}] decay curve: " + " ".join(f"{r['prompt_tok']//1000}k:{r['decode_vs_base']}%" for r in results))
