#!/usr/bin/env python3
"""Fresh prompts of about 4K / 32K / 128K tokens, 256 new tokens each, 3 runs per size, through a running Strata server.

    python decode-sweep.py --url http://127.0.0.1:8080 --out sweep.json [--sizes 4096,32768,131072] [--runs 3]

Uses tools/strata_bench.py's streamed client (`chat`, `digest`): prompt and decode tok/s are the engine's own timings,
first-token time and total are measured by this client.  Every prompt starts with a unique tag (no cache reuse).
The key is read from $STRATA_KEY.  Greedy, thinking off.
"""
import argparse, json, os, statistics, sys, time
from pathlib import Path

ap = argparse.ArgumentParser()
ap.add_argument("--url", default="http://127.0.0.1:8080")
ap.add_argument("--strata", default=str(Path(__file__).resolve().parents[3]))   # the Strata folder this report is in
ap.add_argument("--sizes", default="4096,32768,131072")
ap.add_argument("--runs", type=int, default=3)
ap.add_argument("--max-new", type=int, default=256)
ap.add_argument("--tokens-per-word", type=float, default=1.08)   # strata_bench's calibration on this model
ap.add_argument("--out", required=True)
a = ap.parse_args()
sys.path.insert(0, str(Path(a.strata) / "tools"))
from strata_bench import chat, digest          # noqa: E402
from ab_engine import long_prompt              # noqa: E402

key = os.environ.get("STRATA_KEY", "")
ASK = ("\n\nNow write a long Python module that sorts these words in ten different ways, one function per way, "
       "with docstrings and comments. Start with the code at once.")
nonce = str(int(time.time()))
rows = []
digest(chat(a.url, key, [{"role": "user", "content": f"[warm {nonce}] Say hi."}], 8))          # warm-up, not kept
for size in [int(x) for x in a.sizes.split(",")]:
    words = int((size - 60) / a.tokens_per_word)
    for run in range(a.runs):
        msg = f"[sweep {size}-{run}-{nonce}]\n" + long_prompt(words) + ASK
        d = digest(chat(a.url, key, [{"role": "user", "content": msg}], a.max_new))
        d.update(size=size, run=run, words=words)
        rows.append(d)
        print(f"{size:>7} run {run}: {d['prompt_tokens']} tokens ({d['reused_tokens']} reused), prompt {d['prompt_tps']} tok/s, "
              f"decode {d['decode_tps']} tok/s, {d['generated_tokens']} generated, first token {d['ttft_s']} s, "
              f"total {d['total_s']} s, drafts {d['drafts_accepted']}/{d['drafts_offered']}"
              + (f"  ERROR {d['err']}" if d['err'] else ""), flush=True)
Path(a.out).write_text(json.dumps(rows, indent=1))
for size in sorted({r["size"] for r in rows}):
    rs = [r for r in rows if r["size"] == size and r["ok"]]
    for k in ("prompt_tps", "decode_tps", "ttft_s", "total_s"):
        v = [r[k] for r in rs if r[k] is not None]
        if v:
            print(f"{size:>7} {k:11s} median {statistics.median(v):.2f} ({min(v):.2f}-{max(v):.2f}) n={len(v)}")
