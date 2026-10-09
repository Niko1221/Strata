#!/usr/bin/env python3
"""default_path_check.py URL OUT.json [API_KEY_FILE] — greedy answers to fixed prompts from a running Strata server,
for comparing two builds' default path (no remote stage): the same config served by each build, then diff the JSONs.

Prompts: short code / prose / Chinese, a reasoning one, and an ~8K-token prompt built from corpus.txt (next to this
script, or env CORPUS).  temperature 0, top_k 1, max_tokens 200, reasoning off.  Records each answer's text, its
token counts and the finish reason.
"""
import json
import os
import sys
import urllib.request
from pathlib import Path

url, out = sys.argv[1].rstrip("/"), Path(sys.argv[2])
key = Path(sys.argv[3]).read_text().strip() if len(sys.argv) > 3 else os.environ.get("STRATA_API_KEY", "")
corpus = Path(os.environ.get("CORPUS", Path(__file__).with_name("corpus.txt"))).read_text(encoding="utf-8")
prompts = {
    "code": "Write a Python function that merges two sorted lists into one sorted list, with a short docstring.",
    "prose": "Explain in one paragraph why the sky looks blue during the day and red at sunset.",
    "zh": "用繁體中文簡單說明什麼是快取（cache），以及它為什麼能讓程式變快。",
    "math": "A train leaves at 9:40 and arrives at 13:05. How long is the trip in minutes? Show the steps.",
    "long8k": "Summarize the main ideas of this document in five bullet points.\n\n<document>\n"
              + corpus[:30000] + "\n</document>",
}
res = {}
for name, text in prompts.items():
    body = {"model": "strata", "messages": [{"role": "user", "content": text}], "max_tokens": 200,
            "temperature": 0, "top_k": 1, "top_p": 1, "stream": False,
            "chat_template_kwargs": {"enable_thinking": False}}
    req = urllib.request.Request(url + "/v1/chat/completions", data=json.dumps(body).encode(),
                                 headers={"Content-Type": "application/json", "Authorization": f"Bearer {key}"})
    with urllib.request.urlopen(req, timeout=900) as r:
        j = json.load(r)
    c = j["choices"][0]
    res[name] = {"text": c["message"].get("content"), "finish": c.get("finish_reason"),
                 "prompt_tokens": j.get("usage", {}).get("prompt_tokens"),
                 "completion_tokens": j.get("usage", {}).get("completion_tokens")}
    print(name, res[name]["prompt_tokens"], res[name]["completion_tokens"], repr((res[name]["text"] or "")[:60]))
out.write_text(json.dumps(res, ensure_ascii=False, indent=1), encoding="utf-8")
