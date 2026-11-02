"""Reproduce Cherry Studio's request exactly, taken from the capture proxy record, one factor at a time.

The captured request (it sent `POST /v/v1/chat/completions`, so the path was doubled, but the body
is the real thing):

    model=Qwen3.8-Flash-Next  stream=True  max_tokens=32768
    temperature=1  top_p=0.95
    tools=<5: automation_repl automation_repl_reset fs_read web_fetch web_search>
    reasoning_effort="xhigh"
    messages: system 899 chars (citations policy), user 29 chars (the fibonacci question)

Each row removes exactly one of those, so whichever row jumps to ~240 says which one costs the speed.
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

SYSTEM = (
    "<citations>\n"
    "When a statement in your answer is based on a web_search, web_fetch, kb_search, or kb_read "
    "result, append a citation marker immediately after the statement, in the form [1], [2], and so "
    "on, numbered in the order the sources first appear. Do not cite a source you did not read. If "
    "the answer needs no source, write no marker at all. Keep the markers inside the sentence they "
    "support, before the full stop, and never group several markers at the end of a paragraph.\n"
    "</citations>\n\n"
    "You are a careful assistant. Answer the user's question directly. Follow formatting "
    "instructions exactly. When the user asks for a bare result, give the result and nothing else."
)

TOOLS = [
    {"type": "function", "function": {
        "name": "web_search", "description": "Search the web for pages matching a query.",
        "parameters": {"type": "object", "properties": {
            "query": {"type": "string"}, "count": {"type": "integer"}},
            "required": ["query"], "additionalProperties": False}}},
    {"type": "function", "function": {
        "name": "web_fetch", "description": "Fetch one page and return its readable text.",
        "parameters": {"type": "object", "properties": {
            "url": {"type": "string"}},
            "required": ["url"], "additionalProperties": False}}},
    {"type": "function", "function": {
        "name": "fs_read", "description": "Read a file from the workspace.",
        "parameters": {"type": "object", "properties": {
            "path": {"type": "string"}},
            "required": ["path"], "additionalProperties": False}}},
    {"type": "function", "function": {
        "name": "automation_repl", "description": "Run a script in the automation sandbox.",
        "parameters": {"type": "object", "properties": {
            "code": {"type": "string"}},
            "required": ["code"], "additionalProperties": False}}},
    {"type": "function", "function": {
        "name": "automation_repl_reset", "description": "Reset the automation sandbox.",
        "parameters": {"type": "object", "properties": {}, "additionalProperties": False}}},
]

LINE = re.compile(r"prompt (\d+) tokens = \d+ reused \+ \d+ read in \d+ ms \([\d.]+ tok/s\), "
                  r"(\d+) generated in (\d+) ms \(([\d.]+) tok/s\), drafts accepted (\d+) of (\d+)")


def post(body):
    req = urllib.request.Request(
        BASE + "/v1/chat/completions",
        data=json.dumps(body).encode("utf-8"),
        headers={"Content-Type": "application/json", "Authorization": "Bearer " + KEY},
        method="POST")
    try:
        with urllib.request.urlopen(req, timeout=3600) as r:
            r.read()
        return None
    except urllib.error.HTTPError as exc:
        return f"HTTP {exc.code} {exc.read().decode('utf-8', 'replace')[:130]}"


def ask(label, body, repeats=2):
    rows = []
    for i in range(repeats):
        mark = LOG.stat().st_size
        err = post(body)
        if err:
            print(f"{label:28s} turn{i}  {err}")
            continue
        time.sleep(1.0)
        with LOG.open("r", encoding="utf-8", errors="replace") as f:
            f.seek(mark)
            got = LINE.findall(f.read())
        if not got:
            print(f"{label:28s} turn{i}  no engine line")
            continue
        p, g, ms, speed, acc, tried = got[-1]
        rows.append({"speed": float(speed), "acc": int(acc), "tried": int(tried), "gen": int(g)})
        print(f"{label:28s} turn{i}  prompt={p:>5s} gen={g:>5s} {speed:>7s} tok/s  "
              f"accept {100*int(acc)//max(1,int(tried))}%")
    if rows:
        med = statistics.median(r["speed"] for r in rows)
        acc = 100 * sum(r["acc"] for r in rows) / max(1, sum(r["tried"] for r in rows))
        gen = statistics.median(r["gen"] for r in rows)
        print(f"{label:28s} ==> median {med:7.1f} tok/s  gen~{gen:.0f}  acceptance {acc:5.1f}%\n")
    return rows


def cherry(system=True, tools=True, effort="xhigh", temperature=1, top_p=0.95,
           max_tokens=2000):
    msgs = []
    if system:
        msgs.append({"role": "system", "content": SYSTEM})
    msgs.append({"role": "user", "content": ASK})
    body = {"model": MODEL, "messages": msgs, "stream": False,
            "max_tokens": max_tokens, "temperature": temperature, "top_p": top_p}
    if tools:
        body["tools"] = TOOLS
    if effort is not None:
        body["reasoning_effort"] = effort
    return body


print("Cherry Studio's captured body, then one factor removed per row\n")
ask("exact Cherry body", cherry())
ask("no reasoning_effort", cherry(effort=None))
ask("no tools", cherry(tools=False))
ask("no system prompt", cherry(system=False))
ask("effort none", cherry(effort="none"))
ask("temperature 0", cherry(temperature=0))
ask("max_tokens 900", cherry(max_tokens=900))
ask("bare (mine earlier)", {"model": MODEL, "messages": [{"role": "user", "content": ASK}],
                            "max_tokens": 900, "temperature": 0,
                            "chat_template_kwargs": {"enable_thinking": False}})
