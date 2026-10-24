#!/usr/bin/env python3
"""tools/reuse_replay.py - how much of each prompt the engine reuses over a long agent conversation.

Replays a recorded agent session (a JSONL file in the pi coding agent's session format: its "message" rows, with the
roles system, user, assistant and toolResult) against a running Strata server: the recorded task,
then one request per recorded assistant turn.  The model's own replies go back into the conversation (as a client
sends them: reasoning_content, content, tool_calls with JSON-string arguments), each followed by the recorded tool
results of that turn (paired with the reply's calls in order; nothing is executed).  So every prompt is the previous
prompt, the reply the engine generated and a tool result: a request the engine can continue from the state it holds.
Per request it prints the prompt size and timings.cache_n (the tokens reused); at the end the requests that reused
nothing, those that reused only part of the held tokens (not counting a request after a reply without a call: it gets a
"Continue." user turn, after which the template drops the earlier thinking), and the prompt tokens read in all.

    python tools/reuse_replay.py SESSION.jsonl --url http://127.0.0.1:8095 --model MODEL --turns 40 --cap 8192 --out run.jsonl
"""
from __future__ import annotations

import argparse
import json
import sys
import time
import urllib.request

PI_TOOLS = [
    {"type": "function", "function": {"name": "read", "description": "Read file contents", "parameters": {
        "type": "object", "properties": {"path": {"type": "string"}, "offset": {"type": "number"},
                                         "limit": {"type": "number"}}, "required": ["path"]}}},
    {"type": "function", "function": {"name": "bash", "description": "Execute bash commands", "parameters": {
        "type": "object", "properties": {"command": {"type": "string"}, "timeout": {"type": "number"}},
        "required": ["command"]}}},
    {"type": "function", "function": {"name": "edit", "description": "Edit a single file using exact text replacement",
                                      "parameters": {"type": "object", "properties": {
                                          "path": {"type": "string"},
                                          "edits": {"type": "array", "items": {"type": "object", "properties": {
                                              "oldText": {"type": "string"}, "newText": {"type": "string"}}}}},
                                          "required": ["path", "edits"]}}},
    {"type": "function", "function": {"name": "write", "description": "Create or overwrite files", "parameters": {
        "type": "object", "properties": {"path": {"type": "string"}, "content": {"type": "string"}},
        "required": ["path", "content"]}}},
]


def load(path):
    rows = [json.loads(x) for x in open(path, encoding="utf-8")]
    msgs = [r["message"] for r in rows if r.get("type") == "message"]
    system = next((m for m in msgs if m["role"] == "system"), None)
    sys_text = "\n\n".join(v for v in (system or {}).get("sections", {}).values() if isinstance(v, str)) or \
        "You are a coding agent."
    task = next(m for m in msgs if m["role"] == "user")
    task_text = task["content"] if isinstance(task["content"], str) else \
        "".join(c.get("text", "") for c in task["content"] if c.get("type") == "text")
    turns = []                                  # per recorded assistant turn: the tool results that followed it
    for i, m in enumerate(msgs):
        if m["role"] != "assistant":
            continue
        results = []
        for n in msgs[i + 1:]:
            if n["role"] != "toolResult":
                break
            results.append("\n".join(c["text"] for c in n["content"] if c.get("type") == "text") or "(no tool output)")
        turns.append(results)
    return sys_text, task_text, turns


def post(url, body, timeout=3600):
    req = urllib.request.Request(url.rstrip("/") + "/v1/chat/completions", data=json.dumps(body).encode(),
                                 headers={"content-type": "application/json"})
    with urllib.request.urlopen(req, timeout=timeout) as r:
        return json.loads(r.read())


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("session")
    ap.add_argument("--url", default="http://127.0.0.1:8095")
    ap.add_argument("--model", required=True, help="the served model's name")
    ap.add_argument("--turns", type=int, default=40)
    ap.add_argument("--cap", type=int, default=8192, help="max_tokens of each request")
    ap.add_argument("--effort", default="high")
    ap.add_argument("--budget", type=int, default=8000, help="reasoning_budget_tokens (0: the server's)")
    ap.add_argument("--out")
    a = ap.parse_args()
    sys_text, task_text, turns = load(a.session)
    messages = [{"role": "system", "content": sys_text}, {"role": "user", "content": task_text}]
    out = open(a.out, "w", encoding="utf-8") if a.out else None
    held = 0                    # the tokens the engine should hold after the previous request (prompt + reply - 1)
    misses = partial = read = 0
    user_turn = False           # the previous reply had no call: a "Continue." user turn (the template drops earlier thinking)
    for k, results in enumerate(turns[:a.turns]):
        t0 = time.time()
        r = post(a.url, {"model": a.model, "messages": messages, "tools": PI_TOOLS, "max_tokens": a.cap,
                         "reasoning_effort": a.effort, "stream": False,
                         **({"reasoning_budget_tokens": a.budget} if a.budget else {})})
        msg = r["choices"][0]["message"]
        usage, timings = r.get("usage") or {}, r.get("timings") or {}
        prompt, gen = usage.get("prompt_tokens", 0), usage.get("completion_tokens", 0)
        cache_n, prompt_n = timings.get("cache_n", 0), timings.get("prompt_n", prompt)
        row = {"turn": k, "prompt": prompt, "cache_n": cache_n, "prompt_n": prompt_n,
               "prompt_ms": timings.get("prompt_ms"), "generated": gen,
               "finish": r["choices"][0].get("finish_reason"), "held_before": held, "after_user_turn": user_turn,
               "wall_s": round(time.time() - t0, 1)}
        if k > 0 and not user_turn:
            misses += cache_n == 0
            partial += 0 < cache_n < held
        read += prompt_n
        print(json.dumps(row), flush=True)
        if out:
            out.write(json.dumps(row) + "\n")
        held = prompt + gen - 1
        reply = {"role": "assistant", "content": msg.get("content") or ""}
        if msg.get("reasoning_content"):
            reply["reasoning_content"] = msg["reasoning_content"]
        calls = msg.get("tool_calls") or []
        if calls:
            reply["tool_calls"] = [{"id": c.get("id") or f"call_{k}_{j}", "type": "function",
                                    "function": {"name": c["function"]["name"],
                                                 "arguments": c["function"]["arguments"]}} for j, c in enumerate(calls)]
        messages.append(reply)
        user_turn = not calls
        if calls:
            for j, c in enumerate(reply["tool_calls"]):
                text = results[min(j, len(results) - 1)] if results else "(no tool output)"
                messages.append({"role": "tool", "content": text, "tool_call_id": c["id"]})
        else:
            messages.append({"role": "user", "content": "Continue."})
    summary = {"requests": min(len(turns), a.turns), "reused_nothing": misses, "reused_part": partial,
               "prompt_tokens_read": read}
    print(json.dumps(summary))
    if out:
        out.write(json.dumps(summary) + "\n")
    return 0


if __name__ == "__main__":
    sys.exit(main())
