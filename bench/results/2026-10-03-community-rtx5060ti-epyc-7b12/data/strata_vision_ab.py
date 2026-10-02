#!/usr/bin/env python3
"""编码耗时纯 A/B：max_tokens=1，首呼(含编码) vs 二呼(嵌入命中)，差值=编码耗时。用法: <port> <img> <tag> <out>"""
import json, sys, time, urllib.request
PORT, IMG, TAG, OUT = int(sys.argv[1]), sys.argv[2], sys.argv[3], sys.argv[4]
URL = f"http://127.0.0.1:{PORT}/v1/chat/completions"
def call(content, mt=1, tag=""):
    body = {"messages": [{"role": "user", "content": content}], "max_tokens": mt,
            "temperature": 0, "stream": False, "reasoning_effort": "none"}
    req = urllib.request.Request(URL, data=json.dumps(body).encode(), headers={"Content-Type": "application/json"})
    t0 = time.time()
    with urllib.request.urlopen(req, timeout=1800) as r: d = json.load(r)
    w = time.time() - t0; t = d.get("timings", {}) or {}
    return {"wall": round(w, 3), "prompt_n": t.get("prompt_n"), "cache_n": t.get("cache_n"),
            "prefill": round(t.get("prompt_per_second") or 0, 1), "ct": (d.get("usage") or {}).get("completion_tokens")}
r = {"tag": TAG}
r["text_1tok"] = call("用一句话说明你是什么模型。", 1, "text")
img = [{"type": "text", "text": "这张图里画的是什么？"}, {"type": "image_url", "image_url": {"url": IMG}}]
runs = [call(img, 1, f"img{i}") for i in range(3)]
r["image_runs"] = runs
r["encode_s_est"] = round(runs[0]["wall"] - runs[1]["wall"], 3) if len(runs) > 1 else None
try:
    with urllib.request.urlopen(f"http://127.0.0.1:{PORT}/metrics", timeout=20) as f: d = json.load(f)
    r["engine"] = {k: d["engine"].get(k) for k in ("expert_slots", "vram_free_mib", "context", "kv_resident")}
except Exception as e: r["metrics_error"] = str(e)
json.dump(r, open(OUT, "w"), ensure_ascii=False, indent=1)
print(json.dumps(r, ensure_ascii=False))
