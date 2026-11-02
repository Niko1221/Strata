"""Decompose the web-page (200 tok/s) vs Cherry Studio (<100 tok/s) gap on ONE live server.

Same prompt, same model, same engine process: only one factor changes per row, so the rows add up
to the difference instead of guessing at it.

  base        the fibonacci prompt alone (what the web page sends: prompt=75)
  +context    the same prompt behind ~4000 tokens of system text (what a client with a big system
              prompt and a long tool catalogue sends: prompt~4900)
  +tools      the prompt alone plus a tool list (the call grammar instead of free text)
  +thinking   the prompt alone with thinking left on
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
                  r"(\d+) generated in (\d+) ms \(([\d.]+) tok/s\), drafts accepted (\d+) of (\d+)")

# a plausible client system prompt: long, structured, the sort Cherry Studio / Cline send
SYSTEM = ("You are a helpful assistant. Follow the user's formatting instructions exactly. "
          "Answer in the language of the question. Do not add commentary when the user asks for "
          "a bare result.\n\n") * 220

TOOLS = [{
    "type": "function",
    "function": {
        "name": "report_sequence",
        "description": "Report a numeric sequence to the caller.",
        "parameters": {
            "type": "object",
            "properties": {
                "name": {"type": "string"},
                "values": {"type": "array", "items": {"type": "integer", "minimum": 0}},
            },
            "required": ["name", "values"],
            "additionalProperties": False,
        },
    },
}]


def ask(label, body, repeats=3):
    rows = []
    for i in range(repeats):
        mark = LOG.stat().st_size
        req = urllib.request.Request(
            BASE + "/v1/chat/completions",
            data=json.dumps(body).encode("utf-8"),
            headers={"Content-Type": "application/json", "Authorization": "Bearer " + KEY},
            method="POST")
        try:
            with urllib.request.urlopen(req, timeout=2400) as r:
                r.read()
        except urllib.error.HTTPError as exc:
            body_txt = exc.read().decode("utf-8", "replace")
            print(f"{label:11s} turn{i}  HTTP {exc.code} {body_txt[:110]}")
            continue
        time.sleep(1.0)
        with LOG.open("r", encoding="utf-8", errors="replace") as f:
            f.seek(mark)
            got = LINE.findall(f.read())
        if not got:
            print(f"{label:11s} turn{i}  no engine line")
            continue
        p, g, ms, speed, acc, tried = got[-1]
        rows.append({"prompt": int(p), "speed": float(speed), "gen": int(g),
                     "acc": int(acc), "tried": int(tried)})
        print(f"{label:11s} turn{i}  prompt={p:>5s} gen={g:>4s} {speed:>7s} tok/s  "
              f"accept {100*int(acc)//max(1,int(tried))}%")
    if rows:
        med = statistics.median(r["speed"] for r in rows)
        acc = 100 * sum(r["acc"] for r in rows) / max(1, sum(r["tried"] for r in rows))
        print(f"{label:11s} ==> median {med:7.1f} tok/s, acceptance {acc:5.1f}%\n")
    return rows


def body(messages, **extra):
    b = {"model": MODEL, "messages": messages, "max_tokens": 900, "temperature": 0,
         "chat_template_kwargs": {"enable_thinking": False}}
    b.update(extra)
    return b


print("same engine process, one factor per row\n")
base = ask("base", body([{"role": "user", "content": PROMPT}]))
ask("+context", body([{"role": "system", "content": SYSTEM},
                      {"role": "user", "content": PROMPT}]))
ask("+tools", body([{"role": "user", "content": PROMPT}], tools=TOOLS, tool_choice="auto"))
think = body([{"role": "user", "content": PROMPT}])
think["chat_template_kwargs"] = {"enable_thinking": True}
think["max_tokens"] = 3000
ask("+thinking", think, repeats=2)

print("the web page sends the `base` shape; Cherry Studio typically sends `+context` and `+tools` together")
