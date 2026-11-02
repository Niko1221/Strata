"""Is the remaining grammar cost the mechanism, or the grammar being stricter than the model?

Two request shapes with identical prompts differ only in whether a constraint is present, but that
comparison mixes two effects: the mask machinery, and the fact that a constraint bans tokens the
model wanted (each ban turns a draft into a rejection, which is what `drafts accepted` counts).

This separates them with a grammar that cannot reject anything the model wanted: it takes the plain
answer's own text and spells it as one literal.  Every character is forced, so no token is ever
banned that the model proposed - and the acceptance rate must land back at the plain rate.  If it
does, the machinery costs nothing and what the earlier -6% measured was strictness (the digits
grammar forbids the spaces the model writes after each comma), which is the constraint doing its
job, not an implementation tax.

  plain    no constraint
  literal  root ::= "<the plain answer>"
  loose    digits, commas, and the spaces the model actually writes
  strict   digits and commas only, no spaces (the grammar the earlier run used)

Usage: python grammar_cost_split.py <config.json>
"""
import json
import re
import statistics
import sys
import time
import urllib.error
import urllib.request
from pathlib import Path

CFG = json.loads(Path(sys.argv[1]).read_text(encoding="utf-8"))
LOG = Path(CFG["log"])
KEY = CFG["api_key"]
MODEL = CFG["model_name"]
BASE = f"http://127.0.0.1:{CFG['port']}"
PROMPT = "从0开始列出60位斐波那契数列，直接说结果，用“，”分割。"

STRICT = 'root ::= [0-9]+ ("," [0-9]+)*'
# one optional space either side of the separator: what the model writes when nothing stops it
LOOSE = 'root ::= [0-9]+ (sp "," sp [0-9]+)*\nsp ::= " "?'

LINE = re.compile(r"prompt (\d+) tokens = \d+ reused \+ \d+ read in \d+ ms \([\d.]+ tok/s\), "
                  r"(\d+) generated in (\d+) ms \(([\d.]+) tok/s\), drafts accepted (\d+) of (\d+)")


def post(body):
    req = urllib.request.Request(
        BASE + "/v1/chat/completions", data=json.dumps(body).encode("utf-8"),
        headers={"Content-Type": "application/json", "Authorization": "Bearer " + KEY},
        method="POST")
    try:
        with urllib.request.urlopen(req, timeout=2400) as r:
            return r.status, json.loads(r.read().decode("utf-8"))
    except urllib.error.HTTPError as exc:
        try:
            return exc.code, json.loads(exc.read().decode("utf-8"))
        except Exception:
            return exc.code, {"error": {"message": exc.read().decode("utf-8", "replace")[:200]}}


def one(label, grammar, repeats=3):
    rows = []
    for i in range(repeats):
        body = {"model": MODEL, "messages": [{"role": "user", "content": PROMPT}],
                "max_tokens": 900, "temperature": 0,
                "chat_template_kwargs": {"enable_thinking": False}}
        if grammar is not None:
            body["grammar"] = grammar
        mark = LOG.stat().st_size
        status, resp = post(body)
        if status != 200:
            print(f"{label:8s} turn{i}  HTTP {status} {str(resp)[:120]}")
            return rows, None
        text = (resp.get("choices") or [{}])[0].get("message", {}).get("content") or ""
        time.sleep(1.0)
        with LOG.open("r", encoding="utf-8", errors="replace") as f:
            f.seek(mark)
            got = LINE.findall(f.read())
        if not got:
            print(f"{label:8s} turn{i}  no engine line")
            continue
        p, g, ms, speed, acc, tried = got[-1]
        rows.append({"speed": float(speed), "gen": int(g), "acc": int(acc), "tried": int(tried)})
        print(f"{label:8s} turn{i}  gen={g:>4s} {speed:>7s} tok/s  "
              f"accept {100*int(acc)//max(1,int(tried)):>3d}%  {len(text)} chars")
    return rows, text


def summarise(label, rows):
    if not rows:
        print(f"{label:8s} ==> no rows")
        return float("nan"), 0.0
    med = statistics.median(r["speed"] for r in rows)
    acc = 100 * sum(r["acc"] for r in rows) / max(1, sum(r["tried"] for r in rows))
    print(f"{label:8s} ==> median {med:7.1f} tok/s  acceptance {acc:5.1f}%\n")
    return med, acc


print("fibonacci prompt; only the constraint changes\n")
plain_rows, answer = one("plain", None)
plain, plain_acc = summarise("plain", plain_rows)

if not answer:
    raise SystemExit("the plain answer was empty; nothing to spell")

literal = json.dumps(answer, ensure_ascii=False)  # a quoted GBNF literal of the answer itself
lit_rows, _ = one("literal", "root ::= " + literal)
lit, lit_acc = summarise("literal", lit_rows)

loose_rows, _ = one("loose", LOOSE)
loose, loose_acc = summarise("loose", loose_rows)

strict_rows, _ = one("strict", STRICT)
strict, strict_acc = summarise("strict", strict_rows)

print("summary")
print(f"  plain    {plain:7.1f} tok/s   acceptance {plain_acc:5.1f}%")
print(f"  literal  {lit:7.1f} tok/s   acceptance {lit_acc:5.1f}%   "
      f"({100*(lit-plain)/plain:+.1f}%: the machinery alone, no token banned)")
print(f"  loose    {loose:7.1f} tok/s   acceptance {loose_acc:5.1f}%   "
      f"({100*(loose-plain)/plain:+.1f}%: a constraint the model can satisfy as it writes)")
print(f"  strict   {strict:7.1f} tok/s   acceptance {strict_acc:5.1f}%   "
      f"({100*(strict-plain)/plain:+.1f}%: a constraint that forbids what the model wants)")
print("\n  a large gap between literal and strict is strictness, not overhead: the drafts keep")
print("  proposing tokens the grammar bans, and each of those is a rejected draft.")
