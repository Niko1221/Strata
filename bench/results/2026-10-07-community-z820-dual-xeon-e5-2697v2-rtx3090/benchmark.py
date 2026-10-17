#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Strata community benchmark script (the one behind the results table).

Builds a deterministic synthetic filler with a unique nonce (so no prefix
cache reuse), appends an instruction that forces a long output, streams the
response, and records TTFT plus the engine's own timing values.

Example:
  python benchmark.py --base http://127.0.0.1:18092 --model <model-id> \
    --lengths 4096,32768,131072,500000 --runs 3 --max-tokens 320
"""
from __future__ import annotations

import argparse
import json
import time
import urllib.request
import uuid
from pathlib import Path

OUTPUT_TAIL = ("\n\nIGNORE the filler above. Output the integers from 1 to 130, "
               "one per line, with no other text.")


def filler(n_chars: int, nonce: str) -> str:
    head = (f"# bench nonce {nonce}\n"
            "# The following lines are synthetic filler for prompt-size benchmarking.\n")
    parts = [head]
    size = len(head)
    n = 0
    while size < n_chars:
        line = f"def synth_{n:06d}(x):  # filler line {n}\n    return x + {n}\n\n"
        parts.append(line)
        size += len(line)
        n += 1
    return "".join(parts)


def run_one(base: str, model: str, prompt: str, max_tokens: int, out_dir: Path) -> dict:
    url = base.rstrip("/") + "/v1/chat/completions"
    body = {
        "model": model,
        "messages": [{"role": "user", "content": prompt}],
        "max_tokens": max_tokens,
        "temperature": 0,
        "stream": True,
        "stream_options": {"include_usage": True},
        "chat_template_kwargs": {"enable_thinking": False},
    }
    t0 = time.time()
    ttft = None
    usage = timings = None
    events = []
    req = urllib.request.Request(url, data=json.dumps(body).encode("utf-8"),
                                 headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=5400) as r:
        for raw in r:
            line = raw.decode("utf-8", "replace").strip()
            if not line.startswith("data:"):
                continue
            payload = line[5:].strip()
            if payload == "[DONE]":
                break
            try:
                ev = json.loads(payload)
            except json.JSONDecodeError:
                continue
            events.append(ev)
            delta = ((ev.get("choices") or [{}])[0].get("delta") or {})
            if (delta.get("content") or "") and ttft is None:
                ttft = time.time() - t0
            usage = ev.get("usage") or usage
            timings = ev.get("timings") or timings
    wall = time.time() - t0
    rec = {"prompt_chars": len(prompt), "wall_s": round(wall, 3),
           "ttft_s": None if ttft is None else round(ttft, 3),
           "usage": usage, "timings": timings}
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "result.json").write_text(json.dumps(rec, ensure_ascii=False, indent=1), encoding="utf-8")
    (out_dir / "events.json").write_text(json.dumps(events, ensure_ascii=False), encoding="utf-8")
    return rec


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--base", default="http://127.0.0.1:18092")
    ap.add_argument("--model", default="strata")
    ap.add_argument("--lengths", default="4096,32768,131072")
    ap.add_argument("--runs", type=int, default=3)
    ap.add_argument("--max-tokens", type=int, default=320)
    ap.add_argument("--ratio", type=float, default=2.03,
                    help="characters per prompt token (calibrate on your model/tokenizer)")
    ap.add_argument("--out", default="results")
    a = ap.parse_args()
    out = Path(a.out)
    for length in [int(x) for x in a.lengths.split(",")]:
        for i in range(a.runs):
            prompt = filler(int(length * a.ratio), uuid.uuid4().hex[:12]) + OUTPUT_TAIL
            rec = run_one(a.base, a.model, prompt, a.max_tokens, out / f"{length}/run{i+1}")
            tm = rec.get("timings") or {}
            print(f"{length}/run{i+1}: prompt_n={tm.get('prompt_n')} "
                  f"prompt_tps={tm.get('prompt_per_second')} pred_n={tm.get('predicted_n')} "
                  f"decode_tps={tm.get('predicted_per_second')} ttft={rec['ttft_s']}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
