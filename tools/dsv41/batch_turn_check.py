#!/usr/bin/env python3
"""A two-turn conversation through the batched path: turn 2 must continue the first (same answer as production, and the
engine's own line says how much of its prompt it reused).

    python3 batch_turn_check.py baseline     # against production, 8080: saves both answers
    python3 batch_turn_check.py check        # against the test server, 8099: compares them

Exit 0 when both turns match.
"""
from __future__ import annotations

import json
import sys
import urllib.request

KEY = open("/data/works/Strata/api-key.txt").read().strip()
MODEL = "deepseek-v4.1-flash"
REF = "/data/works/turn_baseline.json"
T1 = "What is the capital of France? Answer in one short sentence."
T2 = "And of Italy?"


def ask(port, messages, max_tokens=48):
    body = {"model": MODEL, "temperature": 0, "max_tokens": max_tokens, "messages": messages}
    req = urllib.request.Request("http://127.0.0.1:%d/v1/chat/completions" % port, data=json.dumps(body).encode(),
                                 headers={"Content-Type": "application/json", "Authorization": "Bearer " + KEY})
    with urllib.request.urlopen(req, timeout=1800) as r:
        d = json.loads(r.read().decode())
    m = d["choices"][0]["message"]
    return {"content": m.get("content") or "", "usage": d.get("usage"),
            "reasoning": (m.get("reasoning_content") or "")[:200]}


def turns(port):
    a1 = ask(port, [{"role": "user", "content": T1}])
    # another conversation between the turns: on the batch path its slot gets the model, so turn 1's state is parked
    # before turn 2 asks for it back - the case the conversation cache exists for
    side = ask(port, [{"role": "user", "content": "Name three colours, one per line."}], max_tokens=40)
    a2 = ask(port, [{"role": "user", "content": T1}, {"role": "assistant", "content": a1["content"]},
                    {"role": "user", "content": T2}])
    return {"t1": a1, "t2": a2, "side": side}


def main() -> int:
    mode = sys.argv[1] if len(sys.argv) > 1 else "check"
    if mode == "baseline":
        got = turns(8080)
        print("baseline turn 1:", repr(got["t1"]["content"][:80]), "|", json.dumps(got["t1"]["usage"]))
        print("baseline turn 2:", repr(got["t2"]["content"][:80]), "|", json.dumps(got["t2"]["usage"]))
        json.dump(got, open(REF, "w"))
        return 0
    want = json.load(open(REF))
    got = turns(8099)
    ok1 = got["t1"]["content"] == want["t1"]["content"]
    ok2 = got["t2"]["content"] == want["t2"]["content"]
    print("turn 1: %s | prompt %s tokens" % ("same as production" if ok1 else "DIFFERENT",
                                             (got["t1"]["usage"] or {}).get("prompt_tokens")))
    print("turn 2: %s | prompt %s tokens (production %s)" % (
        "same as production" if ok2 else "DIFFERENT", (got["t2"]["usage"] or {}).get("prompt_tokens"),
        (want["t2"]["usage"] or {}).get("prompt_tokens")))
    if not ok1:
        print("  want %r" % want["t1"]["content"][:100])
        print("  got  %r" % got["t1"]["content"][:100])
    if not ok2:
        print("  want %r" % want["t2"]["content"][:100])
        print("  got  %r" % got["t2"]["content"][:100])
    print("batch_turn_check:", "ok" if (ok1 and ok2) else "FAILED")
    return 0 if (ok1 and ok2) else 1


if __name__ == "__main__":
    raise SystemExit(main())