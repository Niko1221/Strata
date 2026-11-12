#!/usr/bin/env python3
"""strata-prefill.py OUT — холодное чтение промпта боевой Strata (:8070): куски стога 8k/32k/100k токенов
с уникальной меткой в начале (кэш бесед не срабатывает), по 2 повтора, плюс фиксированный промпт для сверки ответа.
Ключ из $LLM_API_KEY. Строки — OUT/prefill.jsonl, ответы сверки — OUT/check.txt. Замер gpu-unlock, 2026-10-26."""
import json
import os
import sys
import time
import urllib.request
import uuid
from pathlib import Path

KEY = os.environ["LLM_API_KEY"]
HAY = Path("<root>/bench/prompts/hay-680k.txt").read_text(encoding="utf-8")
SIZES = {"8k": 26_000, "32k": 105_000, "100k": 330_000}   # знаков, ≈3.3 знака на токен
REPEATS = 2
OUT = Path(sys.argv[1])
OUT.mkdir(parents=True, exist_ok=True)


def post(messages, max_tokens):
    body = {"messages": messages, "max_tokens": max_tokens, "temperature": 0, "seed": 1234, "stream": False}
    req = urllib.request.Request("http://127.0.0.1:8070/v1/chat/completions", json.dumps(body).encode(),
                                 headers={"Content-Type": "application/json", "Authorization": f"Bearer {KEY}"})
    with urllib.request.urlopen(req, timeout=1800) as r:
        return json.load(r)


for label, nchars in SIZES.items():
    for rep in range(REPEATS):
        text = f"[{uuid.uuid4()}]\n" + HAY[:nchars] + "\n\nНазови одним словом язык этого кода."
        t0 = time.time()
        t = post([{"role": "user", "content": text}], 8).get("timings", {})
        row = {"size": label, "rep": rep, "prompt_n": t.get("prompt_n"), "cache_n": t.get("cache_n"),
               "prompt": t.get("prompt_per_second"), "wall": round(time.time() - t0, 1)}
        with open(OUT / "prefill.jsonl", "a") as f:
            f.write(json.dumps(row) + "\n")
        print(f"[{time.strftime('%T')}] {label}#{rep}: {row['prompt']} tok/s, промпт {row['prompt_n']}, {row['wall']} с",
              flush=True)

q = "Объясни в пяти пунктах, чем отличается TCP от UDP."
r = post([{"role": "user", "content": q}], 300)
(OUT / "check.txt").write_text(r["choices"][0]["message"]["content"], encoding="utf-8")
print("сверка:", r.get("timings", {}).get("predicted_per_second"), "tok/s")
