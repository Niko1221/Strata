"""Decode speed by prompt length, measured from the engine's own log line.

Sends the same answer-shaped request at several prompt sizes with thinking off, so the only
variable is how many tokens the model must attend to while it decodes.
"""
import json
import re
import sys
import time
import urllib.error
import urllib.request
from pathlib import Path

BASE = "http://127.0.0.1:8080"
KEY = "sk-ouzhiyingyyds666!"
LOG = Path(r"F:\Strata-dev\strata-gyro-gyro-m.log")
MODEL = "Qwen3.8-Flash-Next"

FILLER = ("The quick brown fox jumps over the lazy dog. "
          "A steady rain fell on the empty street all afternoon. ")


def post(body):
    req = urllib.request.Request(
        BASE + "/v1/chat/completions",
        data=json.dumps(body).encode("utf-8"),
        headers={"Content-Type": "application/json", "Authorization": "Bearer " + KEY},
        method="POST",
    )
    try:
        with urllib.request.urlopen(req, timeout=1800) as resp:
            return resp.status, json.loads(resp.read().decode("utf-8"))
    except urllib.error.HTTPError as exc:
        try:
            return exc.code, json.loads(exc.read().decode("utf-8"))
        except Exception:
            return exc.code, {"error": {"message": "unparsed"}}


def new_lines(mark: int):
    with LOG.open("r", encoding="utf-8", errors="replace") as f:
        f.seek(mark)
        return f.read()


def engine_line(text: str):
    """The engine's own accounting: prompt N ... P generated in T ms (S tok/s), drafts accepted A of B."""
    found = re.findall(
        r"prompt (\d+) tokens = \d+ reused \+ \d+ read in \d+ ms \([\d.]+ tok/s\), "
        r"(\d+) generated in (\d+) ms \(([\d.]+) tok/s\), drafts accepted (\d+) of (\d+)", text)
    return found[-1] if found else None


def run(label: str, filler_reps: int):
    mark = LOG.stat().st_size
    prompt = (FILLER * filler_reps + "\n\nIn exactly one sentence, say what happened.").strip()
    t0 = time.time()
    status, body = post({
        "model": MODEL,
        "messages": [{"role": "user", "content": prompt}],
        "max_tokens": 400,
        "temperature": 0,
        "chat_template_kwargs": {"enable_thinking": False},
    })
    wall = time.time() - t0
    if status >= 400:
        print(f"{label:22s} ERROR {status} {json.dumps(body.get('error'))[:160]}")
        return None
    time.sleep(1.0)                      # the log line lands just after the reply
    row = engine_line(new_lines(mark))
    if row is None:
        print(f"{label:22s} no engine line (wall {wall:.1f}s)")
        return None
    ptok, gen, ms, speed, acc, tried = row
    print(f"{label:22s} prompt={ptok:>6s}  gen={gen:>4s}  {speed:>7s} tok/s  "
          f"draft {acc}/{tried} ({100*int(acc)//max(1,int(tried))}%)")
    return float(speed)


print("decode speed vs prompt length (temperature 0, thinking off, max_tokens 400)\n")
print("the request shape matters: the same engine decodes faster over a short context\n")
for label, reps in [("tiny prompt", 1), ("small prompt", 30),
                    ("medium prompt", 150), ("large prompt", 500)]:
    run(label, reps)
