"""A/B one config: send the same requests, read the engine's own log line, print the medians.

Usage: python ab_run.py <config.json> <label>

Both arms run the same prompts at the same temperature so the only difference is the engine's
draft-acceptance rule.  Speed and acceptance come from the engine, never from client-side maths
(thinking tokens stream outside `content`, which makes a client-side rate meaningless).
"""
import json
import re
import statistics
import sys
import time
import urllib.error
import urllib.request
from pathlib import Path

CONFIG = Path(sys.argv[1])
LABEL = sys.argv[2]
CFG = json.loads(CONFIG.read_text(encoding="utf-8"))
LOG = Path(CFG["log"])
KEY = CFG["api_key"]
MODEL = CFG["model_name"]
BASE = f"http://127.0.0.1:{CFG['port']}"

FILLER = ("A steady rain fell on the empty street all afternoon while the traffic lights "
          "changed for nobody at all. ")
PROMPTS = [
    ("technical", FILLER * 2 + "\n\nWrite a detailed technical explanation in at least 600 words. "
                 "Plain prose, several paragraphs, no lists. Topic: how a mixture-of-experts "
                 "transformer routes each token and what that costs in memory bandwidth."),
    ("narrative", FILLER + "\n\nWrite a 600-word story about a lighthouse keeper who finds a "
                  "message in a bottle. Plain prose, several paragraphs."),
    ("predictable", "List the integers from 1 to 600 in order, separated by commas. Nothing else."),
]

LINE = re.compile(r"prompt (\d+) tokens = \d+ reused \+ \d+ read in \d+ ms \([\d.]+ tok/s\), "
                  r"(\d+) generated in (\d+) ms \(([\d.]+) tok/s\), drafts accepted (\d+) of (\d+)")


def alive(timeout=1800):
    deadline = time.time() + timeout
    while time.time() < deadline:
        try:
            with urllib.request.urlopen(BASE + "/health", timeout=10) as r:
                if r.status == 200:
                    return True
        except Exception:
            pass
        time.sleep(3)
    return False


def post(body):
    req = urllib.request.Request(
        BASE + "/v1/chat/completions",
        data=json.dumps(body).encode("utf-8"),
        headers={"Content-Type": "application/json", "Authorization": "Bearer " + KEY},
        method="POST")
    with urllib.request.urlopen(req, timeout=2400) as r:
        return r.read()


def ask(name, prompt, temperature):
    mark = LOG.stat().st_size
    post({"model": MODEL, "messages": [{"role": "user", "content": prompt}],
          "max_tokens": 1200, "temperature": temperature,
          "chat_template_kwargs": {"enable_thinking": False}})
    time.sleep(1.0)
    with LOG.open("r", encoding="utf-8", errors="replace") as f:
        f.seek(mark)
        rows = LINE.findall(f.read())
    if not rows:
        return None
    p, g, ms, speed, acc, tried = rows[-1]
    return {"name": name, "speed": float(speed), "gen": int(g),
            "acc": int(acc), "tried": int(tried)}


want = alive()
print(f"[{LABEL}] server ready: {want}")
if not want:
    sys.exit(1)

results = {}
for temperature in (0.6,):
    for name, prompt in PROMPTS:
        for turn in range(3):                    # repeats per prompt: the spread is real
            got = ask(name, prompt, temperature)
            if got is None:
                print(f"[{LABEL}] {name} turn {turn}: no engine line")
                continue
            results.setdefault(name, []).append(got)
            print(f"[{LABEL}] {name:12s} t={temperature} turn{turn}  gen={got['gen']:>4d}  "
                  f"{got['speed']:>6.1f} tok/s  accept {100*got['acc']//max(1,got['tried'])}%")

print()
for name, rows in results.items():
    speed = statistics.median(r["speed"] for r in rows)
    acc = 100 * sum(r["acc"] for r in rows) / max(1, sum(r["tried"] for r in rows))
    print(f"[{LABEL}] {name:12s} median {speed:6.1f} tok/s   acceptance {acc:5.1f}%")
summary = statistics.median(r["speed"] for rows in results.values() for r in rows)
acc = 100 * sum(r["acc"] for rows in results.values() for r in rows) / \
    max(1, sum(r["tried"] for rows in results.values() for r in rows))
print(f"[{LABEL}] OVERALL median {summary:.1f} tok/s, acceptance {acc:.1f}%")
Path(f".zcode-obj/ab-result-{LABEL}.json").write_text(json.dumps(results, indent=1), encoding="utf-8")
