"""Single-variable check: a tool request with a LONG answer, with and without the call grammar.

The earlier tools rows generated 77-220 tokens (the model called a tool instead of answering), which
is too short to time reliably.  Here the user demands a bare number list, so the answer is ~500
tokens either way; `strata_grammar: false` removes only the constraint, leaving the tools, the
messages and the sampling identical.  That is the comparison the numbers need.
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
ASK = "从0开始列出60位斐波那契数列，直接说结果，用“，”分割。"

TOOLS = [
    {"type": "function", "function": {
        "name": name, "description": desc,
        "parameters": {"type": "object", "properties": {prop: {"type": "string"}},
                       "required": [prop], "additionalProperties": False}}}
    for name, desc, prop in [
        ("web_search", "Search the web for pages matching a query.", "query"),
        ("web_fetch", "Fetch one page and return its readable text.", "url"),
        ("fs_read", "Read a file from the workspace.", "path"),
        ("automation_repl", "Run a script in the automation sandbox.", "code"),
        ("automation_repl_reset", "Reset the automation sandbox.", "reason"),
    ]
]

LINE = re.compile(r"prompt (\d+) tokens = \d+ reused \+ \d+ read in \d+ ms \([\d.]+ tok/s\), "
                  r"(\d+) generated in (\d+) ms \(([\d.]+) tok/s\), drafts accepted (\d+) of (\d+)")


def ask(label, grammar_on, thinking, repeats=3):
    rows = []
    for i in range(repeats):
        mark = LOG.stat().st_size
        b = {"model": MODEL, "messages": [{"role": "user", "content": ASK}],
             "max_tokens": 900, "temperature": 0,
             "chat_template_kwargs": {"enable_thinking": thinking},
             "tools": TOOLS, "tool_choice": "auto"}
        if not grammar_on:
            b["strata_grammar"] = False
        req = urllib.request.Request(
            BASE + "/v1/chat/completions", data=json.dumps(b).encode("utf-8"),
            headers={"Content-Type": "application/json", "Authorization": "Bearer " + KEY},
            method="POST")
        status, resp = None, None
        try:
            with urllib.request.urlopen(req, timeout=3600) as r:
                status, resp = r.status, json.loads(r.read().decode("utf-8"))
        except urllib.error.HTTPError as exc:
            print(f"{label:24s} turn{i}  HTTP {exc.code} "
                  f"{exc.read().decode('utf-8','replace')[:110]}")
            continue
        time.sleep(1.0)
        with LOG.open("r", encoding="utf-8", errors="replace") as f:
            f.seek(mark)
            got = LINE.findall(f.read())
        if not got:
            print(f"{label:24s} turn{i}  no engine line")
            continue
        p, g, ms, speed, acc, tried = got[-1]
        rows.append({"speed": float(speed), "gen": int(g), "acc": int(acc),
                     "tried": int(tried), "prompt": int(p)})
        print(f"{label:24s} turn{i}  prompt={p:>5s} gen={g:>4s} {speed:>7s} tok/s  "
              f"accept {100*int(acc)//max(1,int(tried))}%")
    if rows:
        med = statistics.median(r["speed"] for r in rows)
        gen = statistics.median(r["gen"] for r in rows)
        acc = 100 * sum(r["acc"] for r in rows) / max(1, sum(r["tried"] for r in rows))
        print(f"{label:24s} ==> median {med:7.1f} tok/s  gen~{gen:.0f}  acceptance {acc:5.1f}%\n")
    return rows


def med(rows):
    return statistics.median(r["speed"] for r in rows) if rows else float("nan")


print("tools present in every row; only the constraint and thinking change\n")
a = ask("grammar on,  no think", True, False)
b = ask("grammar OFF, no think", False, False)
c = ask("grammar on,  think", True, True)
d = ask("grammar OFF, think", False, True)

print("summary (median tok/s)")
print(f"  grammar on  vs off, thinking off: {med(a):7.1f} -> {med(b):7.1f} "
      f"({100*(med(b)-med(a))/max(1e-9,med(a)):+.1f}% when the grammar is removed)")
print(f"  grammar on  vs off, thinking on : {med(c):7.1f} -> {med(d):7.1f} "
      f"({100*(med(d)-med(c))/max(1e-9,med(c)):+.1f}% when the grammar is removed)")
Path(".zcode-obj/tools_grammar_off.json").write_text(
    json.dumps({"on_off_think": a, "off_off_think": b, "on_on_think": c, "off_on_think": d},
               indent=1), encoding="utf-8")
