"""2x2: does the grammar cost grow with the prompt?

Row 1 of the reproduction showed tools (the call grammar) taking 250 -> 183 tok/s at prompt 686.
Cherry Studio's own requests sat at 91-110 with prompts of 1886-3950 and the same tools.  Either
the grammar's cost grows with the context, or a second factor stacks on top.  Two factors, two
levels each, same question every time:

    tools off / small prompt     tools on / small prompt
    tools off / large prompt     tools on / large prompt
"""
import json
import re
import statistics
import sys
import time
import urllib.error
import urllib.request
from pathlib import Path

BASE = "http://127.0.0.1:8080"
KEY = "sk-ouzhiyingyyds666!"
LOG = Path(r"F:\Strata-dev\strata-gyro-gyro-m.log")
MODEL = "Qwen3.8-Flash-Next"
ASK = "从0开始列出60位斐波那契数列，直接说结果，用“，”分割。"

BIG = (
    "<citations>\nWhen a statement in your answer is based on a tool result, append a citation "
    "marker immediately after the statement, in the form [1], [2], numbered in the order the "
    "sources first appear. Do not cite a source you did not read.\n</citations>\n\n"
    "You are a careful assistant working inside a desktop chat client. Answer the user's question "
    "directly. Follow formatting instructions exactly. When the user asks for a bare result, give "
    "the result and nothing else. Prefer plain prose. Do not wrap the answer in a code fence unless "
    "the user asks for code.\n\n"
) * 12

TOOLS = [
    {"type": "function", "function": {
        "name": name, "description": desc,
        "parameters": {"type": "object", "properties": {
            prop: {"type": "string"}}, "required": [prop], "additionalProperties": False}}}
    for name, desc, prop in [
        ("web_search", "Search the web for pages matching a query.", "query"),
        ("web_fetch", "Fetch one page and return its readable text.", "url"),
        ("fs_read", "Read a file from the workspace.", "path"),
        ("automation_repl", "Run a script in the automation sandbox.", "code"),
    ]
]

LINE = re.compile(r"prompt (\d+) tokens = \d+ reused \+ \d+ read in \d+ ms \([\d.]+ tok/s\), "
                  r"(\d+) generated in (\d+) ms \(([\d.]+) tok/s\), drafts accepted (\d+) of (\d+)")


def body(big, tools):
    msgs = []
    if big:
        msgs.append({"role": "system", "content": BIG})
    msgs.append({"role": "user", "content": ASK})
    b = {"model": MODEL, "messages": msgs, "max_tokens": 2000,
         "temperature": 0.7, "top_p": 0.95, "reasoning_effort": "none"}
    if tools:
        b["tools"] = TOOLS
    return b


def ask(label, b, repeats=3):
    rows = []
    for i in range(repeats):
        mark = LOG.stat().st_size
        req = urllib.request.Request(
            BASE + "/v1/chat/completions", data=json.dumps(b).encode("utf-8"),
            headers={"Content-Type": "application/json", "Authorization": "Bearer " + KEY},
            method="POST")
        try:
            with urllib.request.urlopen(req, timeout=3600) as r:
                r.read()
        except urllib.error.HTTPError as exc:
            print(f"{label:22s} turn{i}  HTTP {exc.code} "
                  f"{exc.read().decode('utf-8', 'replace')[:110]}")
            continue
        time.sleep(1.0)
        with LOG.open("r", encoding="utf-8", errors="replace") as f:
            f.seek(mark)
            got = LINE.findall(f.read())
        if not got:
            print(f"{label:22s} turn{i}  no engine line")
            continue
        p, g, ms, speed, acc, tried = got[-1]
        rows.append({"speed": float(speed), "acc": int(acc), "tried": int(tried),
                     "gen": int(g), "prompt": int(p)})
        print(f"{label:22s} turn{i}  prompt={p:>5s} gen={g:>4s} {speed:>7s} tok/s  "
              f"accept {100*int(acc)//max(1,int(tried))}%")
    if rows:
        med = statistics.median(r["speed"] for r in rows)
        acc = 100 * sum(r["acc"] for r in rows) / max(1, sum(r["tried"] for r in rows))
        pr = statistics.median(r["prompt"] for r in rows)
        print(f"{label:22s} ==> median {med:7.1f} tok/s  prompt~{pr:.0f}  acceptance {acc:5.1f}%\n")
    return rows


print("2x2: tools (the call grammar) x prompt size\n")
a = ask("tools off, small", body(False, False))
b = ask("tools on,  small", body(False, True))
c = ask("tools off, large", body(True, False))
d = ask("tools on,  large", body(True, True))

def med(rows):
    return statistics.median(r["speed"] for r in rows) if rows else float("nan")

print("summary")
print(f"  the grammar's cost at a small prompt: {med(a):.1f} -> {med(b):.1f} "
      f"({100*(med(b)-med(a))/med(a):+.1f}%)")
print(f"  the grammar's cost at a large prompt: {med(c):.1f} -> {med(d):.1f} "
      f"({100*(med(d)-med(c))/med(c):+.1f}%)")
print(f"  the prompt's cost without tools:      {med(a):.1f} -> {med(c):.1f} "
      f"({100*(med(c)-med(a))/med(a):+.1f}%)")
print(f"  the prompt's cost with tools:         {med(b):.1f} -> {med(d):.1f} "
      f"({100*(med(d)-med(b))/med(b):+.1f}%)")
Path(".zcode-obj/grammar2x2.json").write_text(
    json.dumps({"off_small": a, "on_small": b, "off_large": c, "on_large": d}, indent=1),
    encoding="utf-8")
