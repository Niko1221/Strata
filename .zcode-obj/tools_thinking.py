"""Tools plus thinking: is the reasoning region's exclusion automaton the cost?

Evidence so far:
  bare GBNF, any prompt size     ~-6% or none   (the mask itself is cheap)
  tools, thinking OFF            ~-6%           (serve/grammar_tools.py's call grammar)
  tools, thinking ON (xhigh)     ~-27%          (the Cherry Studio shape)

With thinking on, the tool grammar's reasoning region is free text that must never spell
`</think>`, a JSON fence or `<tool_call>` - i.e. every reasoning token walks an exclusion
automaton.  Cherry Studio sends `reasoning_effort: "xhigh"` with 5 tools, which is exactly this
combination.  Three cells, one prompt, long answers:

  tools off, thinking off
  tools on,  thinking off
  tools on,  thinking on
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
LOG = Path(r"F:\Strata-dev\strata-gyro-gyo-m.log")   # replaced below from the real config
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


def ask(label, tools, thinking, repeats=3):
    rows = []
    for i in range(repeats):
        mark = LOG.stat().st_size
        b = {"model": MODEL, "messages": [{"role": "user", "content": ASK}],
             "max_tokens": 4000, "temperature": 0.7, "top_p": 0.95}
        b["reasoning_effort"] = "xhigh" if thinking else "none"
        if tools:
            b["tools"] = TOOLS
        req = urllib.request.Request(
            BASE + "/v1/chat/completions", data=json.dumps(b).encode("utf-8"),
            headers={"Content-Type": "application/json", "Authorization": "Bearer " + KEY},
            method="POST")
        try:
            with urllib.request.urlopen(req, timeout=3600) as r:
                r.read()
        except urllib.error.HTTPError as exc:
            print(f"{label:26s} turn{i}  HTTP {exc.code} "
                  f"{exc.read().decode('utf-8', 'replace')[:110]}")
            continue
        time.sleep(1.0)
        with LOG.open("r", encoding="utf-8", errors="replace") as f:
            f.seek(mark)
            got = LINE.findall(f.read())
        if not got:
            print(f"{label:26s} turn{i}  no engine line")
            continue
        p, g, ms, speed, acc, tried = got[-1]
        rows.append({"speed": float(speed), "gen": int(g), "acc": int(acc),
                     "tried": int(tried), "prompt": int(p)})
        print(f"{label:26s} turn{i}  prompt={p:>5s} gen={g:>5s} {speed:>7s} tok/s  "
              f"accept {100*int(acc)//max(1,int(tried))}%")
    if rows:
        med = statistics.median(r["speed"] for r in rows)
        gen = statistics.median(r["gen"] for r in rows)
        acc = 100 * sum(r["acc"] for r in rows) / max(1, sum(r["tried"] for r in rows))
        print(f"{label:26s} ==> median {med:7.1f} tok/s  gen~{gen:.0f}  acceptance {acc:5.1f}%\n")
    return rows


def med(rows):
    return statistics.median(r["speed"] for r in rows) if rows else float("nan")


print("tools x thinking, one prompt, answers long enough to time\n")
a = ask("tools off, thinking off", False, False)
b = ask("tools on,  thinking off", True, False)
c = ask("tools on,  thinking on", True, True)
d = ask("tools off, thinking on", False, True)

print("summary (median tok/s)")
print(f"  tools off, thinking off  {med(a):7.1f}")
print(f"  tools on,  thinking off  {med(b):7.1f}   tools cost {100*(med(b)-med(a))/med(a):+.1f}%")
print(f"  tools on,  thinking on   {med(c):7.1f}   tools+thinking vs tools {100*(med(c)-med(b))/med(b):+.1f}%")
print(f"  tools off, thinking on   {med(d):7.1f}   thinking alone {100*(med(d)-med(a))/med(a):+.1f}%")
print(f"\n  the Cherry Studio shape is `tools on, thinking on` = {med(c):.1f} tok/s")
Path(".zcode-obj/tools_thinking.json").write_text(
    json.dumps({"off_off": a, "on_off": b, "on_on": c, "off_on": d}, indent=1), encoding="utf-8")
