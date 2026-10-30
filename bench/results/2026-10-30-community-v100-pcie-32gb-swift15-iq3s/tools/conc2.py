#!/usr/bin/env python3
"""RES-20261029-01 (TASK-20261029-07): two long requests at once, the shape of two agents with long transcripts on a
2-slot server. Both are decode-bench prompts (code and prose at the same context, so no shared prefix past the first
tokens), prefixed "[conc2]" so that no earlier request's cached prompt is reused, sent together; per request TTFT, decode tok/s and output sha256, plus the pair's wall time and the summed
decode rate while both decode. Standard library only.
usage: conc2.py --base URL --model ID --ctx 131072 --arm NAME --rep N --out DIR"""
import argparse, hashlib, importlib.util, json, os, threading, time

spec = importlib.util.spec_from_file_location("db", os.path.join(os.path.dirname(os.path.abspath(__file__)), "decode-bench.py"))
db = importlib.util.module_from_spec(spec); spec.loader.exec_module(db)

ap = argparse.ArgumentParser()
ap.add_argument("--base", required=True); ap.add_argument("--model", required=True); ap.add_argument("--ctx", type=int, default=131072)
ap.add_argument("--arm", required=True); ap.add_argument("--rep", type=int, default=1); ap.add_argument("--out", required=True)
a = ap.parse_args()
os.makedirs(a.out, exist_ok=True)
res = {}
def one(task):
    t0 = time.time()
    text, m = db.chat(a.base, a.model, "[conc2] " + db.prompt_for(a.ctx, task), 512)
    res[task] = {**m, "t_start": t0, "t_end": time.time(), "sha256": hashlib.sha256(text.encode()).hexdigest()}
    open(os.path.join(a.out, f"out-conc2-{a.arm}-r{a.rep}-{task}.txt"), "w", encoding="utf-8").write(text)
t0 = time.time()
th = [threading.Thread(target=one, args=(t,)) for t in ("code", "prose")]
[t.start() for t in th]; [t.join() for t in th]
wall = time.time() - t0
row = {"arm": a.arm, "rep": a.rep, "ctx": a.ctx, "wall_s": round(wall, 1),
       "sum_decode_tok_s": round(sum(r["decode_tok_s"] or 0 for r in res.values()), 2),
       **{f"{k}_{f}": r[f] for k, r in res.items() for f in ("ttft_s", "decode_tok_s", "completion_tokens", "sha256")}}
print(json.dumps(row), flush=True)
with open(os.path.join(a.out, "conc2.jsonl"), "a") as f:
    f.write(json.dumps(row) + "\n")
