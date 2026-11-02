"""Decode speed for LONG generations, by prompt length — the shape a real chat has.

A 20-token answer fits one draft window and reads fast whatever the engine costs, so it says
nothing about the speed a user sees.  This forces an answer of several hundred tokens and reads
the engine's own log line.
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

FILLER = ("A steady rain fell on the empty street all afternoon while the traffic lights "
          "changed for nobody at all. ")
ASK = ("\n\nWrite a detailed technical explanation in at least 600 words. Use plain prose, "
       "several paragraphs, no lists, no code. Topic: how a mixture-of-experts transformer "
       "routes each token and what that costs in memory bandwidth.")


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


def engine_line(text):
    found = re.findall(
        r"prompt (\d+) tokens = \d+ reused \+ \d+ read in \d+ ms \([\d.]+ tok/s\), "
        r"(\d+) generated in (\d+) ms \(([\d.]+) tok/s\), drafts accepted (\d+) of (\d+)", text)
    return found[-1] if found else None


def run(label, filler_reps):
    mark = LOG.stat().st_size
    prompt = (FILLER * filler_reps + ASK).strip()
    status, body = post({"model": MODEL,
                         "messages": [{"role": "user", "content": prompt}],
                         "max_tokens": 900, "temperature": 0,
                         "chat_template_kwargs": {"enable_thinking": False}})
    if status >= 400:
        print(f"{label:20s} ERROR {status} {json.dumps(body.get('error'))[:150]}")
        return
    time.sleep(1.0)
    with LOG.open("r", encoding="utf-8", errors="replace") as f:
        f.seek(mark)
        row = engine_line(f.read())
    if row is None:
        print(f"{label:20s} no engine line")
        return
    ptok, gen, ms, speed, acc, tried = row
    print(f"{label:20s} prompt={ptok:>6s}  gen={gen:>4s}  {speed:>7s} tok/s  "
          f"draft {acc}/{tried} ({100*int(acc)//max(1,int(tried))}%)")


print("LONG generations (max_tokens 900), temperature 0, thinking off\n")
for label, reps in [("short prompt", 2), ("medium prompt", 200), ("long prompt", 900)]:
    run(label, reps)
print("\nsame engine, same start: only the context it must attend to changes")
