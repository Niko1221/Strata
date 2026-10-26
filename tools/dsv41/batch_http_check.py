#!/usr/bin/env python3
"""The batched serve path over HTTP: the same answer as production, a reference answer through a slot, three clients whose
tokens interleave, a one-token request, a client that leaves.  Run on the box (the server is on 127.0.0.1).

    python3 batch_http_check.py baseline            # against production, 8080: saves the reference answer
    python3 batch_http_check.py check               # against the test server, 8099: compares and runs the rest

Exit 0 when every check passes.
"""
from __future__ import annotations

import json
import sys
import threading
import time
import urllib.request

KEY = open("/data/works/Strata/api-key.txt").read().strip()
MODEL = "deepseek-v4.1-flash"
PROMPT = "What is the capital of France? Answer in one short sentence."
REF = "/data/works/batch_baseline.json"
fails = []
events = []                     # (client, monotonic time) of every token, to see the clients interleave
lock = threading.Lock()


def ask(prompt, max_tokens, temperature=0.0, stream=False, port=8080, timeout=1800, name=None):
    body = {"model": MODEL, "stream": stream, "temperature": temperature, "max_tokens": max_tokens,
            "messages": [{"role": "user", "content": prompt}]}
    req = urllib.request.Request("http://127.0.0.1:%d/v1/chat/completions" % port,
                                 data=json.dumps(body).encode(),
                                 headers={"Content-Type": "application/json", "Authorization": "Bearer " + KEY})
    t0 = time.time()
    with urllib.request.urlopen(req, timeout=timeout) as r:
        if not stream:
            d = json.loads(r.read().decode())
            return {"text": d["choices"][0]["message"].get("content") or "",
                    "usage": d.get("usage"), "wall": round(time.time() - t0, 2)}
        first, text = None, []
        for raw in r:
            line = raw.decode("utf-8", "replace").strip()
            if not line.startswith("data:"):
                continue
            payload = line[5:].strip()
            if payload == "[DONE]":
                break
            d = json.loads(payload)
            delta = d.get("choices", [{}])[0].get("delta", {})
            piece = delta.get("content") or ""      # (the baseline compares the answer, not the thinking)
            if piece:
                if first is None:
                    first = time.time() - t0
                if name is not None:
                    with lock:
                        events.append((name, time.monotonic()))
                text.append(piece)
        return {"text": "".join(text), "first_token_s": round(first or 0, 2), "wall": round(time.time() - t0, 2)}


def check(name, ok, detail=""):
    print(("ok   " if ok else "FAIL ") + name + (("  (" + detail + ")") if detail else ""), flush=True)
    if not ok:
        fails.append(name)


def in_thread(out, key, prompt, max_tokens):
    def run():
        try:
            out[key] = ask(prompt, max_tokens, port=8099, stream=True, name=key)
        except Exception as e:            # noqa: BLE001 - printed by the check below
            out[key] = {"error": repr(e)}
    return threading.Thread(target=run)


def baseline():
    got = ask(PROMPT, 40)
    print("baseline:", json.dumps(got["usage"]), "| text:", repr(got["text"][:120]))
    json.dump(got, open(REF, "w"))
    return 0


def timing():
    """The three clients only: wall time, first tokens and how the two slots' tokens interleave (an A/B against --batch 1)."""
    out = {}
    ta = in_thread(out, "a", "Write a long list of colours, one per line.", 120)
    tb = in_thread(out, "b", "Explain in two sentences why batching helps throughput.", 60)
    tc = in_thread(out, "c", PROMPT, 40)
    t0 = time.time()
    ta.start()
    time.sleep(0.6)
    tb.start()
    time.sleep(0.6)
    tc.start()
    [t.join() for t in (ta, tb, tc)]
    wall = round(time.time() - t0, 2)
    with lock:
        order = [e[0] for e in events]
    switches = sum(1 for x, y in zip(order, order[1:]) if x != y)
    ok = all("text" in out[k] and len(out[k]["text"]) > 10 for k in "abc")
    print("timing: wall %.2f s | first tokens %s | handovers %d over %d tokens | all answered %s" % (
        wall, {k: out[k].get("first_token_s") for k in "abc"}, switches, len(order), ok), flush=True)
    return 0 if ok else 1


def check_all():
    want = json.load(open(REF))
    got = ask(PROMPT, 40, port=8099)
    check("alone: the same answer as production", got["text"] == want["text"],
          "want %r got %r" % (want["text"][:70], got["text"][:70]))
    print("    wall %.2f s (production %.2f s)" % (got["wall"], want["wall"]))

    # Three clients.  A long one and a medium one should take the engine's two slots and decode side by side (their
    # tokens interleave in time); the reference prompt arrives behind them and is admitted into a slot with BGEN once
    # one frees - and must give the byte-identical production answer.
    out = {}
    ta = in_thread(out, "a", "Write a long list of colours, one per line.", 120)
    tb = in_thread(out, "b", "Explain in two sentences why batching helps throughput.", 60)
    tc = in_thread(out, "c", PROMPT, 40)
    t0 = time.time()
    ta.start()
    time.sleep(0.6)
    tb.start()
    time.sleep(0.6)
    tc.start()
    [t.join() for t in (ta, tb, tc)]
    wall = round(time.time() - t0, 2)
    check("three clients all answer",
          all("text" in out[k] and len(out[k]["text"]) > 10 for k in "abc"),
          " ; ".join("%s: %s" % (k, (out[k].get("text") or out[k].get("error", ""))[:40]) for k in "abc"))
    check("the batched path gives the reference answer", out["c"].get("text") == want["text"],
          "want %r got %r" % (want["text"][:70], (out["c"].get("text") or "")[:70]))
    with lock:
        order = [e[0] for e in events]
    switches = sum(1 for x, y in zip(order, order[1:]) if x != y)
    check("the clients' tokens interleave (two slots at once)", switches >= 4,
          "%d handovers over %d tokens: %s" % (switches, len(order), "".join(order[:60])))
    print("    wall %.2f s; first tokens %s" % (wall, {k: out[k].get("first_token_s") for k in "abc"}))

    one = ask("Say yes.", 1, port=8099)
    check("a max_tokens 1 request completes", "text" in one, repr(one.get("text"))[:60])

    # a client that leaves in mid-stream: the server BSTOPs the slot, and the next request still works
    body = {"model": MODEL, "stream": True, "temperature": 0, "max_tokens": 400,
            "messages": [{"role": "user", "content": "Write a long list of animals, one per line."}]}
    req = urllib.request.Request("http://127.0.0.1:8099/v1/chat/completions", data=json.dumps(body).encode(),
                                 headers={"Content-Type": "application/json", "Authorization": "Bearer " + KEY})
    pieces = 0
    r = urllib.request.urlopen(req, timeout=1800)
    for raw in r:
        if raw.decode("utf-8", "replace").strip().startswith("data:"):
            pieces += 1
            if pieces >= 6:
                break
    r.close()                                  # the client goes away mid-answer
    time.sleep(2)
    after = ask("Say good morning in three words.", 64, port=8099)
    check("a request after a client left mid-stream still answers", len(after["text"]) > 0,
          repr(after["text"][:60]) + " " + json.dumps(after.get("usage")))

    if fails:
        print("\n%d check(s) failed: %s" % (len(fails), ", ".join(fails)))
        return 1
    print("\nbatch_http_check: all checks passed")
    return 0


if __name__ == "__main__":
    mode = sys.argv[1] if len(sys.argv) > 1 else "check"
    sys.exit(baseline() if mode == "baseline" else timing() if mode == "timing" else check_all())