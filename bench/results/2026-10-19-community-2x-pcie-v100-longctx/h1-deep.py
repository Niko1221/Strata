# -*- coding: utf-8 -*-
"""H1 quality guardrail at arbitrary depth (UD-IQ4_XS dual).

Usage: python tmp-h1-ud-deep.py <doc_tokens> <max_tokens> <tag>
Writes results-h1-<tag>.json. Keeps prompt+max_tokens inside the served n_ctx.
"""
import json
import sys
import time
import urllib.request

DOC_TOK = int(sys.argv[1]) if len(sys.argv) > 1 else 245000
MAXTOK = int(sys.argv[2]) if len(sys.argv) > 2 else 6000
TAG = sys.argv[3] if len(sys.argv) > 3 else "ud-deep"

KEY = "f859fe4d50fc89102fce152808e837c2"
op = urllib.request.build_opener(urllib.request.ProxyHandler({}))
RATIO = 0.937

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


H1 = "求最小的正整数 n，使得 n! 的十进制表示末尾恰好有 2019 个连续的零。给出 n 和简要证明。"
doc = build_doc(int(DOC_TOK / RATIO)) + "\n\n先忽略上面的全部材料。" + H1

body = {"model": "h1-" + TAG, "messages": [{"role": "user", "content": doc}],
        "temperature": 0.6, "top_p": 0.95, "max_tokens": MAXTOK, "seed": 42, "stream": False,
        "reasoning_budget_tokens": 4000}
data = json.dumps(body, ensure_ascii=False).encode("utf-8")
req = urllib.request.Request("http://127.0.0.1:8080/v1/chat/completions", data=data,
                             headers={"Content-Type": "application/json", "Authorization": "Bearer " + KEY})
t0 = time.time()
r = json.load(op.open(req, timeout=7200))
u, t = r.get("usage", {}), r.get("timings", {})
ans = r.get("choices", [{}])[0].get("message", {}).get("content", "")
rec = {"tag": TAG, "doc_target": DOC_TOK, "prompt_tok": u.get("prompt_tokens"),
       "out_tok": u.get("completion_tokens"),
       "prompt_tps": t.get("prompt_per_second"), "decode_tps": t.get("predicted_per_second"),
       "draft_accept": round((t.get("draft_n_accepted") or 0) / max(1, t.get("draft_n") or 1), 3),
       "h1_has_8090": "8090" in ans, "wall_s": round(time.time() - t0, 1),
       "answer_head": ans[:300]}
json.dump(rec, open(rf"C:/Users/Administrator/Desktop/local-model/flash-ab/results-h1-{TAG}.json", "w", encoding="utf-8"),
          ensure_ascii=False, indent=1)
print(json.dumps({k: v for k, v in rec.items() if k != "answer_head"}, ensure_ascii=False))
print("ANSWER HEAD:", rec["answer_head"][:200])
