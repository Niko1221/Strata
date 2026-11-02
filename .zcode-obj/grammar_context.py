"""Does the grammar's cost grow with the prompt?  A bare GBNF, two prompt sizes, a long answer both times.

An earlier 2x2 suggested the call grammar costs 7% at a 35-token prompt but 46% at ~1900 - but the
large-prompt rows generated only 6-13 tokens (the model called a tool instead of answering), which is
too short to time.  This test removes that confound: one bare GBNF that forces digits-and-commas,
which the model satisfies with ~500 tokens whatever the tools are, at a small and a large prompt.
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
GBNF = 'root ::= nums\nnums ::= [0-9]+ ("," [0-9]+)*'

FILLER = (
    "The report describes a long-running experiment. Each section states what was measured, on "
    "which card, and what the numbers were, so a later reader can repeat it without guessing. "
) * 55

LINE = re.compile(r"prompt (\d+) tokens = \d+ reused \+ \d+ read in \d+ ms \([\d.]+ tok/s\), "
                  r"(\d+) generated in (\d+) ms \(([\d.]+) tok/s\), drafts accepted (\d+) of (\d+)")


def ask(label, messages, grammar, repeats=3):
    rows = []
    for i in range(repeats):
        mark = LOG.stat().st_size
        b = {"model": MODEL, "messages": messages, "max_tokens": 1200,
             "temperature": 0, "chat_template_kwargs": {"enable_thinking": False}}
        if grammar:
            b["grammar"] = GBNF
        req = urllib.request.Request(
            BASE + "/v1/chat/completions", data=json.dumps(b).encode("utf-8"),
            headers={"Content-Type": "application/json", "Authorization": "Bearer " + KEY},
            method="POST")
        try:
            with urllib.request.urlopen(req, timeout=3600) as r:
                r.read()
        except urllib.error.HTTPError as exc:
            print(f"{label:24s} turn{i}  HTTP {exc.code} "
                  f"{exc.read().decode('utf-8', 'replace')[:110]}")
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
        pr = statistics.median(r["prompt"] for r in rows)
        print(f"{label:24s} ==> median {med:7.1f} tok/s  prompt~{pr:.0f}  gen~{gen:.0f}\n")
    return rows


def med(rows):
    return statistics.median(r["speed"] for r in rows) if rows else float("nan")


small = [{"role": "user", "content": ASK}]
large = [{"role": "user", "content": FILLER + "\n\n" + ASK}]

print("one bare GBNF, two prompt sizes, ~500-token answers in both\n")
a = ask("plain,   small prompt", small, False)
b = ask("grammar, small prompt", small, True)
c = ask("plain,   large prompt", large, False)
d = ask("grammar, large prompt", large, True)

print("summary (median tok/s)")
print(f"  plain small {med(a):7.1f}   grammar small {med(b):7.1f}   "
      f"grammar cost {100*(med(b)-med(a))/med(a):+.1f}%")
print(f"  plain large {med(c):7.1f}   grammar large {med(d):7.1f}   "
      f"grammar cost {100*(med(d)-med(c))/med(c):+.1f}%")
print(f"  prompt cost without grammar {100*(med(c)-med(a))/med(a):+.1f}%   "
      f"with grammar {100*(med(d)-med(b))/med(b):+.1f}%")
Path(".zcode-obj/grammar_context.json").write_text(
    json.dumps({"plain_small": a, "grammar_small": b, "plain_large": c, "grammar_large": d},
               indent=1), encoding="utf-8")
