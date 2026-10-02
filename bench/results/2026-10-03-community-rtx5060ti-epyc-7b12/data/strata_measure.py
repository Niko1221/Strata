#!/usr/bin/env python3
"""一根测量臂：短prompt decode×3 / 长prompt prefill / 深上下文(~120k) / 视觉：速度 + 识图精度。

用法: strata_measure.py <port> <arm名> <vision:0|1> <图片路径或-> <输出json>
判分 ground truth（不入 prompt）：已知 ∠CAB=40°、∠CBA=40°、∠DBC=20°、∠DCA=20°，求 ∠DAC = 10°。
"""
import json, re, statistics, sys, time, urllib.request, urllib.error

PORT = int(sys.argv[1])
ARM = sys.argv[2]
HASV = int(sys.argv[3])
IMG = None if len(sys.argv) < 5 or sys.argv[4] in ("-", "") else sys.argv[4]
OUT = sys.argv[5] if len(sys.argv) > 5 else "/dev/stdout"
URL = f"http://127.0.0.1:{PORT}/v1/chat/completions"
DOC = open("/home/ai-agent/DSHW/strata-study/docs/DETAILS.md", encoding="utf-8").read()

SOLVE_ASK = ("附件是一道几何题的图片。请：(1) 逐字读出图中给出的已知条件与所求；"
             "(2) 给出完整的推导或证明过程；(3) 最后单独一行以「∠DAC = ??°」的形式写出最终答案。")


def call(content, max_tokens=256, effort="none", tag="", budget=None, full=False):
    body = {"messages": [{"role": "user", "content": content}], "max_tokens": max_tokens,
            "temperature": 0, "stream": False, "reasoning_effort": effort}
    if budget is not None:
        body["reasoning_budget_tokens"] = budget
    req = urllib.request.Request(URL, data=json.dumps(body).encode(),
                                 headers={"Content-Type": "application/json"})
    t0 = time.time()
    try:
        with urllib.request.urlopen(req, timeout=7200) as r:
            d = json.load(r)
    except urllib.error.HTTPError as e:                 # 未知字段/超限：去掉 budget 重试一次
        if budget is None:
            raise
        body.pop("reasoning_budget_tokens", None)
        req = urllib.request.Request(URL, data=json.dumps(body).encode(),
                                     headers={"Content-Type": "application/json"})
        t0 = time.time()
        with urllib.request.urlopen(req, timeout=7200) as r:
            d = json.load(r)
    dt = time.time() - t0
    t = d.get("timings", {}) or {}
    m = d["choices"][0]["message"]
    rec = dict(tag=tag, wall=round(dt, 1), prompt_n=t.get("prompt_n"), cache_n=t.get("cache_n"),
               prefill=round(t.get("prompt_per_second") or 0, 1),
               decode=round(t.get("predicted_per_second") or 0, 2),
               dn=t.get("draft_n"), da=t.get("draft_n_accepted"),
               accept=(round(100 * (t.get("draft_n_accepted") or 0) / t["draft_n"], 1) if t.get("draft_n") else None),
               ct=(d.get("usage") or {}).get("completion_tokens"),
               finish=d["choices"][0].get("finish_reason"),
               text=(m.get("content") or "")[:200])
    if full:
        rec["full"] = (m.get("content") or "") + "\n<<<REASONING>>>\n" + (m.get("reasoning_content") or "")
    return rec


def grade(solve):
    """自动判分：读条件(40/40/20/20) + 答案(10°)。全部在 content+reasoning 上判。"""
    txt = solve.get("full", "")
    cond = {k: bool(re.search(p, txt)) for k, p in {
        "CAB=40": r"CAB[^0-9]{0,12}40", "CBA=40": r"CBA[^0-9]{0,12}40",
        "DBC=20": r"DBC[^0-9]{0,12}20", "DCA=20": r"DCA[^0-9]{0,12}20"}.items()}
    ans = re.findall(r"[∠∠]?\s*DAC\s*=\s*(\d{1,3})\s*°", txt) or \
          re.findall(r"∠DAC\s*为\s*(\d{1,3})\s*°", txt)
    final = ans[-1] if ans else None
    return {"conditions": cond, "conditions_ok": sum(cond.values()), "final_answer": final,
            "answer_correct": final == "10"}


def metrics():
    try:
        with urllib.request.urlopen(f"http://127.0.0.1:{PORT}/metrics", timeout=20) as r:
            d = json.load(r)
        e, h = d["engine"], d["hardware"]
        return {k: e.get(k) for k in ("kv", "kv_resident", "expert_slots", "expert_cache_mib",
                                      "vram_free_mib", "arena_mib", "spec", "mtp_max", "lookup",
                                      "pool_workers", "pcie_frac", "context")} | \
               {"gpu_mem_used_gib": round(h["gpu_mem_used"] / 2**30, 2),
                "ram_used_gib": round(h["ram_used"] / 2**30, 1)}
    except Exception as ex:
        return {"error": str(ex)}


res = {"arm": ARM, "port": PORT, "vision": HASV, "engine": metrics()}

rows = [call("用一句话说明你是什么模型。\n<!--run %d-->" % i, 256, "none", "short%d" % i) for i in range(3)]
res["short"] = rows
res["short_decode_med"] = statistics.median([r["decode"] for r in rows if r["decode"]])

rows = [call(DOC[:32000] + "\n用一句话总结。\n<!--long %d-->" % i, 64, "none", "long%d" % i) for i in range(2)]
res["long"] = rows
res["long_prefill_med"] = statistics.median([r["prefill"] for r in rows if r["prefill"]])
res["long_prompt_n"] = rows[0]["prompt_n"]

deep = DOC * max(1, 500_000 // len(DOC))
rows = [call(deep + "\n用一句话总结最后一段。", 64, "none", "deep")]
res["deep"] = rows
res["deep_prompt_n"] = rows[0]["prompt_n"]

if IMG:
    # 识图+解题（可判分）：第一次含编码，第二次命中图像缓存（差值≈编码耗时）
    s1 = call([{"type": "text", "text": SOLVE_ASK}, {"type": "image_url", "image_url": {"url": IMG}}],
              2048, "high", "img_solve", budget=1500, full=True)
    s2 = call([{"type": "text", "text": SOLVE_ASK}, {"type": "image_url", "image_url": {"url": IMG}}],
              32, "none", "img_cached", full=True)
    res["solve"] = {k: s1[k] for k in ("wall", "prompt_n", "prefill", "decode", "finish", "ct", "da", "dn")}
    res["solve_grade"] = grade(s1)
    res["image_cached_call"] = {k: s2[k] for k in ("wall", "cache_n", "prompt_n")}
    res["image_tokens_prompt_n"] = s1["prompt_n"]
    if s2.get("cache_n"):
        res["encode_s_est"] = round(s1["wall"] - s2["wall"], 1)
    open(OUT.replace(".json", "") + "-solve.txt", "w").write(s1.get("full", ""))

json.dump(res, open(OUT, "w"), ensure_ascii=False, indent=1)
flat = {k: v for k, v in res.items() if not isinstance(v, list)}
print(json.dumps(flat, ensure_ascii=False))
