#!/usr/bin/env python3
"""补齐长/深档到 >=3 次运行：与 strata_measure.py 同 prompt、同 max_tokens，便于同口径合并。
用法: strata_topup.py <port> <arm> <out.json>   —— 2 次 long(10,381 tok, cap 64) + 2 次 deep(144,610 tok, cap 64)
"""
import json, sys, time, urllib.request

PORT, ARM, OUT = int(sys.argv[1]), sys.argv[2], sys.argv[3]
URL = f"http://127.0.0.1:{PORT}/v1/chat/completions"
DOC = open("/home/ai-agent/DSHW/strata-study/docs/DETAILS.md", encoding="utf-8").read()


def call(content, max_tokens=64, tag=""):
    body = {"messages": [{"role": "user", "content": content}], "max_tokens": max_tokens,
            "temperature": 0, "stream": False, "reasoning_effort": "none"}
    req = urllib.request.Request(URL, data=json.dumps(body).encode(),
                                 headers={"Content-Type": "application/json"})
    t0 = time.time()
    with urllib.request.urlopen(req, timeout=7200) as r:
        d = json.load(r)
    dt = time.time() - t0
    t = d.get("timings", {}) or {}
    return dict(tag=tag, wall=round(dt, 1), prompt_n=t.get("prompt_n"), cache_n=t.get("cache_n"),
                prefill=round(t.get("prompt_per_second") or 0, 1),
                decode=round(t.get("predicted_per_second") or 0, 2),
                dn=t.get("draft_n"), da=t.get("draft_n_accepted"),
                accept=(round(100 * (t.get("draft_n_accepted") or 0) / t["draft_n"], 1) if t.get("draft_n") else None),
                ct=(d.get("usage") or {}).get("completion_tokens"),
                finish=d["choices"][0].get("finish_reason"))


deep = DOC * max(1, 500_000 // len(DOC))
res = {"arm": ARM, "port": PORT, "kind": "topup",
       "long": [call(DOC[:32000] + "\n用一句话总结。\n<!--long-topup %d-->" % i, 64, "long-topup%d" % i) for i in range(2)],
       "deep": [call(deep + "\n用一句话总结最后一段。\n<!--deep-topup %d-->" % i, 64, "deep-topup%d" % i) for i in range(2)]}
json.dump(res, open(OUT, "w"), ensure_ascii=False, indent=1)
print(json.dumps(res, ensure_ascii=False))
