"""Does a multi-turn conversation body decode slower than a single user turn?

Cherry Studio sends a system prompt plus the whole conversation, with assistant turns in it.
Every request so far that reproduced ~105 tok/s had 2-3 "checkpoints" in the engine's line and a
prompt over 1800 tokens; every request at ~240 tok/s had one turn and a short prompt.  Context
length alone was ruled out (a 7959-token system prompt still ran at 228 tok/s), so this test
isolates the conversation SHAPE: the same question, the same total context, once as one user
turn and once as history + question.
"""
import json
import re
import statistics
import time
import urllib.error
import urllib.request
from pathlib import Path

BASE = "http://127.0.0.1:8080"
KEY = "sk-ouzhiyingyyds666!"
LOG = Path(r"F:\Strata-dev\strata-gyro-gyro-m.log")
MODEL = "Qwen3.8-Flash-Next"
PROMPT = "从0开始列出60位斐波那契数列，直接说结果，用“，”分割。"

LINE = re.compile(r"prompt (\d+) tokens = \d+ reused \+ \d+ read in \d+ ms \([\d.]+ tok/s\), "
                  r"(\d+) generated in (\d+) ms \(([\d.]+) tok/s\), drafts accepted (\d+) of (\d+)"
                  r"(?:, (\d+) checkpoints)?")

SYSTEM = ("You are a helpful assistant running inside a desktop chat client. Answer the user's "
          "question directly. Follow formatting instructions exactly. Be concise.")

TURNS = [
    ("What is a Fibonacci sequence?", "A Fibonacci sequence starts with 0 and 1, and each later "
     "number is the sum of the two before it: 0, 1, 1, 2, 3, 5, 8, 13, and so on."),
    ("How many numbers are in the first 60 terms?", "Sixty terms, counting from 0 as the first."),
    ("Do you separate them with commas?", "Yes, commas with no spaces are the usual way."),
]


def post(body):
    req = urllib.request.Request(
        BASE + "/v1/chat/completions",
        data=json.dumps(body).encode("utf-8"),
        headers={"Content-Type": "application/json", "Authorization": "Bearer " + KEY},
        method="POST")
    try:
        with urllib.request.urlopen(req, timeout=2400) as r:
            r.read()
        return None
    except urllib.error.HTTPError as exc:
        return f"HTTP {exc.code} {exc.read().decode('utf-8', 'replace')[:110]}"


def ask(label, messages, repeats=3, **extra):
    rows = []
    for i in range(repeats):
        mark = LOG.stat().st_size
        body = {"model": MODEL, "messages": messages, "max_tokens": 900,
                "temperature": 0, "chat_template_kwargs": {"enable_thinking": False}}
        body.update(extra)
        err = post(body)
        if err:
            print(f"{label:18s} turn{i}  {err}")
            continue
        time.sleep(1.0)
        with LOG.open("r", encoding="utf-8", errors="replace") as f:
            f.seek(mark)
            got = LINE.findall(f.read())
        if not got:
            print(f"{label:18s} turn{i}  no engine line")
            continue
        p, g, ms, speed, acc, tried, ck = got[-1]
        rows.append({"prompt": int(p), "speed": float(speed), "gen": int(g),
                     "acc": int(acc), "tried": int(tried), "ck": ck or "?"})
        print(f"{label:18s} turn{i}  prompt={p:>5s} gen={g:>4s} {speed:>7s} tok/s  "
              f"accept {100*int(acc)//max(1,int(tried))}%  checkpoints={ck}")
    if rows:
        med = statistics.median(r["speed"] for r in rows)
        acc = 100 * sum(r["acc"] for r in rows) / max(1, sum(r["tried"] for r in rows))
        print(f"{label:18s} ==> median {med:7.1f} tok/s, acceptance {acc:5.1f}%\n")
    return rows


# A: exactly what the web page sends
single = [{"role": "user", "content": PROMPT}]

# B: Cherry Studio's shape - system prompt, history with assistant turns, then the question
history = [{"role": "system", "content": SYSTEM}]
for q, a in TURNS:
    history.append({"role": "user", "content": q})
    history.append({"role": "assistant", "content": a})
history.append({"role": "user", "content": PROMPT})

# C: the same body as B but with the history flattened into one user message:
#    same question, similar prompt length, no assistant turns in it
flat = "".join(f"Q: {q}\nA: {a}\n\n" for q, a in TURNS)
flat = [{"role": "system", "content": SYSTEM},
        {"role": "user", "content": flat + "Q: " + PROMPT}]

print("one live engine: the only variable is the conversation shape\n")
ask("single turn", single)
ask("chat history", history)
ask("history flattened", flat)
