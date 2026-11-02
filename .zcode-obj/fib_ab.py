"""Fibonacci prompt under four request shapes, on one engine at a time.

The prompt asks for a comma-separated number list.  That shape is where a per-token constraint
overhead shows up worst: the drafter accepts nearly every token, so the speed ceiling is the
engine's own per-token cost rather than the acceptance rate.  Four shapes:

  plain          no grammar at all (the baseline people remember)
  strata_off     the same request with strata_grammar:false (the switch this branch added)
  grammar        a bare GBNF: digits, commas, nothing else
  structured     a json_schema response_format through the strict tool-value converter

Usage: python fib_ab.py <config.json> <label>
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

PROMPT = "从0开始列出60位斐波那契数列，直接说结果，用“，”分割。"

# digits and commas only: exactly the shape the answer has, so the constraint is realistic
DIGITS_GBNF = 'root ::= nums\nnums ::= [0-9]+ ("," [0-9]+)*'

SCHEMA = {
    "type": "object",
    "properties": {
        "sequence": {
            "type": "array",
            "items": {"type": "integer", "minimum": 0},
            "minItems": 60,
            "maxItems": 60,
        }
    },
    "required": ["sequence"],
    "additionalProperties": False,
}

LINE = re.compile(r"prompt (\d+) tokens = \d+ reused \+ \d+ read in \d+ ms \([\d.]+ tok/s\), "
                  r"(\d+) generated in (\d+) ms \(([\d.]+) tok/s\), drafts accepted (\d+) of (\d+)")


def post(body):
    req = urllib.request.Request(
        BASE + "/v1/chat/completions",
        data=json.dumps(body).encode("utf-8"),
        headers={"Content-Type": "application/json", "Authorization": "Bearer " + KEY},
        method="POST")
    try:
        with urllib.request.urlopen(req, timeout=2400) as r:
            return r.status, json.loads(r.read().decode("utf-8"))
    except urllib.error.HTTPError as exc:
        try:
            return exc.code, json.loads(exc.read().decode("utf-8"))
        except Exception:
            return exc.code, {"error": {"message": "unparsed"}}


def one(shape, repeats):
    body = {"model": MODEL,
            "messages": [{"role": "user", "content": PROMPT}],
            "max_tokens": 900,
            "temperature": 0,
            "chat_template_kwargs": {"enable_thinking": False}}
    if shape == "strata_off":
        body["strata_grammar"] = False
    elif shape == "grammar":
        body["grammar"] = DIGITS_GBNF
    elif shape == "structured":
        body["response_format"] = {"type": "json_schema",
                                   "json_schema": {"name": "fib", "strict": True, "schema": SCHEMA}}
    rows = []
    for i in range(repeats):
        mark = LOG.stat().st_size
        status, resp = post(body)
        if status >= 400:
            err = (resp.get("error") or {}).get("message", "")
            print(f"[{LABEL}] {shape:11s} turn{i}  HTTP {status} {err[:120]}")
            continue
        time.sleep(1.0)
        with LOG.open("r", encoding="utf-8", errors="replace") as f:
            f.seek(mark)
            got = LINE.findall(f.read())
        if not got:
            print(f"[{LABEL}] {shape:11s} turn{i}  no engine line")
            continue
        p, g, ms, speed, acc, tried = got[-1]
        rows.append({"speed": float(speed), "gen": int(g), "acc": int(acc), "tried": int(tried)})
        print(f"[{LABEL}] {shape:11s} turn{i}  gen={g:>4s}  {speed:>7s} tok/s  "
              f"accept {100*int(acc)//max(1,int(tried))}%")
    return rows


print(f"[{LABEL}] engine: {CFG['exe']}")
print(f"[{LABEL}] prompt: {PROMPT}")
print()
out = {}
for shape in ("plain", "strata_off", "grammar", "structured"):
    out[shape] = one(shape, 3)

print()
for shape, rows in out.items():
    if not rows:
        print(f"[{LABEL}] {shape:11s} no samples")
        continue
    speed = statistics.median(r["speed"] for r in rows)
    acc = 100 * sum(r["acc"] for r in rows) / max(1, sum(r["tried"] for r in rows))
    gen = statistics.median(r["gen"] for r in rows)
    print(f"[{LABEL}] {shape:11s} median {speed:7.1f} tok/s  gen~{gen:.0f}  acceptance {acc:5.1f}%")
Path(f".zcode-obj/fib-result-{LABEL}.json").write_text(json.dumps(out, indent=1), encoding="utf-8")
