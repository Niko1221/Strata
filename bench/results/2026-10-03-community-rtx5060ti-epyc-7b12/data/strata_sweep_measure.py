#!/usr/bin/env python3
"""扫描用测量器：一根配置的 decode/prefill + 引擎侧层级计数器（命中率、文件回读、KV 命中）。

用法: strata_sweep_measure.py <port> <engine.log> <out.json>
判据全部来自引擎自己的日志与 /metrics，不靠推算。
"""
import json, re, statistics, sys, time, urllib.request

PORT, LOG, OUT = int(sys.argv[1]), sys.argv[2], sys.argv[3]
URL = f"http://127.0.0.1:{PORT}/v1/chat/completions"
DOC = open("/home/ai-agent/DSHW/strata-study/docs/DETAILS.md", encoding="utf-8").read()


def call(prompt, mt=256, effort="none"):
    body = {"messages": [{"role": "user", "content": prompt}], "max_tokens": mt,
            "temperature": 0, "stream": False, "reasoning_effort": effort}
    req = urllib.request.Request(URL, data=json.dumps(body).encode(),
                                 headers={"Content-Type": "application/json"})
    t0 = time.time()
    with urllib.request.urlopen(req, timeout=3600) as r:
        d = json.load(r)
    dt = time.time() - t0
    t = d.get("timings", {}) or {}
    return dict(wall=round(dt, 1), pn=t.get("prompt_n"),
                pp=round(t.get("prompt_per_second") or 0, 1),
                dec=round(t.get("predicted_per_second") or 0, 2),
                acc=(round(100 * (t.get("draft_n_accepted") or 0) / t["draft_n"], 1) if t.get("draft_n") else None))


out = {}
rows = [call(DOC[:3000] + "\n用一句话总结。\n<!--s%d-->" % i, 256) for i in range(2)]
out["probe"] = rows
out["decode_med"] = statistics.median([r["dec"] for r in rows])
out["prefill_med"] = statistics.median([r["pp"] for r in rows])
accs = [r["acc"] for r in rows if r["acc"] is not None]
out["accept_med"] = statistics.median(accs) if accs else None

try:
    txt = open(LOG, encoding="utf-8", errors="replace").read()
    m = re.findall(r"expert tiers: GPU (\d+) hits this request; since the start RAM (\d+) blobs, files (\d+) blobs "
                   r"([\d.]+) MB read", txt)
    k = re.findall(r"KV streaming: ([\d.]+)% of (\d+) block reads hit VRAM, ([\d.]+) MiB read from RAM", txt)
    r = re.findall(r"resident RAM: ([\d.]+) GiB of experts in RAM, (\d+) exchanged", txt)
    if m:
        out["expert_tiers"] = {"gpu_hits_this_req": int(m[-1][0]), "ram_blobs": int(m[-1][1]),
                               "file_blobs": int(m[-1][2]), "file_mb_read": float(m[-1][3])}
    if k:
        out["kv_stream"] = {"vram_hit_pct": float(k[-1][0]), "block_reads": int(k[-1][1]),
                            "ram_mib_read": float(k[-1][2])}
    if r:
        out["resident_gib"] = float(r[-1][0])
except Exception as e:
    out["log_parse_error"] = str(e)

try:
    with urllib.request.urlopen(f"http://127.0.0.1:{PORT}/metrics", timeout=20) as f:
        d = json.load(f)
    e, h = d["engine"], d["hardware"]
    out["engine"] = {k: e.get(k) for k in ("kv", "kv_resident", "expert_slots", "vram_free_mib",
                                           "arena_mib", "pool_workers", "pcie_frac", "context")}
    out["engine"]["gpu_mem_gib"] = round(h["gpu_mem_used"] / 2**30, 2)
    out["engine"]["ram_used_gib"] = round(h["ram_used"] / 2**30, 1)
except Exception as e:
    out["metrics_error"] = str(e)

json.dump(out, open(OUT, "w"), ensure_ascii=False, indent=1)
print(json.dumps({k: v for k, v in out.items() if k != "probe"}, ensure_ascii=False))
