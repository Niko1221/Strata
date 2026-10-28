"""agent-turns.py - decode speed on coding-agent-shaped turns, the T630's real workload.

speed-checks.py measures fresh short prompts, where Strata's suffix drafting (drafts copied from an earlier repeat
in the context) never fires. The coder's turns are the opposite: a ~65K-token context that the prompt cache mostly
holds, and answers that copy long stretches of it (an edited file, a function rewritten). This sends one shared
~55K-token context of real repo files, then four turns over it:

  edit-file   return box/strata/bench/t630-sweep.py whole with a small change    (copy-heavy: suffix drafts)
  edit-lock   return tools/coder/lock.sh whole with a small change             (copy-heavy, shell)
  new-test    write a unittest for one function                                (mostly new text)
  plan        a refactor plan with thinking on                                 (thinking, prose)

Each turn streams; decode tok/s = (completion tokens - 1) / (last token - first token), so prompt reading is
out of it. Server-default sampling (production: temperature 0.6), the way the agents send. The first turn after a
start also reads the whole context; its TTFT is reported separately.
Usage: python3 -I agent-turns.py [base_url]   (STRATA_KEY as in speed-checks.py)
"""
import json
import os
import sys
import time
import urllib.request

BASE = sys.argv[1] if len(sys.argv) > 1 else "http://127.0.0.1:8080"
KEY = os.environ.get("STRATA_KEY", "")
AUTH = {**({"Authorization": f"Bearer {KEY}", "x-api-key": KEY} if KEY else {})}
REPO = os.path.expanduser("~/home_config")
CONTEXT_FILES = ["tools/coder/workspace.py", "tools/coder/run.py", "tools/coder/bash_guard.py",
                 "tools/coder/lock.sh", "box/strata/bench/t630-sweep.py"]


def read(rel):
    with open(os.path.join(REPO, rel), encoding="utf-8") as f:
        return f.read()


SYSTEM = ("You are a coding agent working in the user's repository. The repository files you need are below. "
          "Answer with exactly what is asked: when asked for a whole file, output the complete file in one fenced "
          "block and nothing else.\n\n" +
          "\n\n".join(f"### {p}\n```\n{read(p)}\n```" for p in CONTEXT_FILES))

TURNS = [
    ("edit-file", False, 4096,
     "Return box/strata/bench/t630-sweep.py in full, unchanged except: add a module-level constant "
     "COOLDOWN_S = 30 after the existing constants and sleep for COOLDOWN_S seconds at the end of run_arm()."),
    ("edit-lock", False, 3072,
     "Return tools/coder/lock.sh in full, unchanged except: add a line that echoes the date in UTC at the very "
     "start of the script, right after the shebang and the comment block."),
    ("new-test", False, 1536,
     "Write a Python unittest TestCase for _make_writable in tools/coder/workspace.py covering a read-only "
     "directory tree and a symlink to a file outside the tree. Output only the test module."),
    ("plan", True, 2048,
     "Plan how to split tools/coder/bash_guard.py into one module per ecosystem (pip, npm, go) without changing "
     "behaviour. List the steps and the risks."),
]


# Follow-up turns: ONE growing conversation, the shape of most coder requests (median 288 tokens generated, a few
# hundred to a few thousand new prompt tokens each). Each turn appends the model's real reply and a new tool result
# of ~300 / ~1,000 / ~2,500 tokens, so every request extends the conversation already on the cards, as the coder's
# do (a branch would also time a restore from the conversation cache, which real turns rarely pay: 10 of 529 on
# 2026-10-28). Their TTFT is the incremental prompt read. SPEED_NONCE (set per run by t630-sweep.py) opens the
# conversation's first user message, so the chain is new every run while the system context stays cached.
TOOL_SRC = read("tools/boss/server.py")
NONCE = os.environ.get("SPEED_NONCE", str(time.time()))
READS = [("read-300", 0, 1200), ("read-1k", 1200, 5200), ("read-2.5k", 5200, 15200)]   # (name, char span)


def followups():
    msgs = [{"role": "system", "content": SYSTEM},
            {"role": "user", "content": f"[run {NONCE}] Read tools/boss/server.py piece by piece and say what each "
                                        "piece does in one sentence."},
            {"role": "assistant", "content": "I'll read the file first."}]
    for name, a, b in READS:
        msgs.append({"role": "user", "content": "Tool result (next part of tools/boss/server.py):\n" +
                     TOOL_SRC[a:b] + "\n\nIn one sentence: what does this part do?"})
        _, text = turn(name, False, 64, None, msgs)
        msgs.append({"role": "assistant", "content": text})


def turn(name, thinking, max_tokens, prompt, messages=None):
    body = {"model": "x", "stream": True, "max_tokens": max_tokens, "stream_options": {"include_usage": True},
            "messages": messages or [{"role": "system", "content": SYSTEM}, {"role": "user", "content": prompt}],
            "chat_template_kwargs": {"enable_thinking": thinking}}
    req = urllib.request.Request(BASE + "/v1/chat/completions", data=json.dumps(body).encode(),
                                 headers={"Content-Type": "application/json", **AUTH})
    t0 = time.time()
    first = last = None
    chunks = 0
    usage = None
    text = []
    with urllib.request.urlopen(req, timeout=1200) as r:
        for raw in r:
            line = raw.decode("utf-8", "replace").strip()
            if not line.startswith("data:"):
                continue
            data = line[5:].strip()
            if data == "[DONE]":
                break
            j = json.loads(data)
            if j.get("usage"):
                usage = j["usage"]
            for c in j.get("choices") or []:
                d = c.get("delta") or {}
                if d.get("content"):
                    text.append(d["content"])
                if d.get("content") or d.get("reasoning_content"):
                    now = time.time()
                    first = first or now
                    last = now
                    chunks += 1
    n = (usage or {}).get("completion_tokens") or chunks
    ttft = (first or time.time()) - t0
    rate = (n - 1) / (last - first) if first and last and last > first and n > 1 else 0.0
    print(f"  agent {name}: {n} tokens, decode {rate:.1f} tok/s, ttft {ttft:.2f}s", flush=True)
    return rate, "".join(text)


if __name__ == "__main__":
    print(f"== agent turns: context {len(SYSTEM)} chars (~{len(SYSTEM) // 4} tokens), {len(TURNS)} turns")
    # The thinking-off turns, the follow-up chain, then the thinking-on plan last: switching thinking re-reads the
    # whole context once (a separate cache entry per mode), which would otherwise land on the first follow-up.
    for t in TURNS:
        if not t[1]:
            turn(*t)
    followups()
    for t in TURNS:
        if t[1]:
            turn(*t)
