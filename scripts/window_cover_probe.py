#!/usr/bin/env python3
"""Probe the one-prompt/N-row path on the live box with a short and a long state.

  short state -> one consolidated prompt, both cues captured in the single verify window
  long state  -> the prompt is read in SEVERAL windows; the engine must capture each named cue from
                 whichever window covers it (`capture_window_rows`), so the request still answers

Run it with the preset on arm B (STRATA_JEV_ROWS=1).  Verify the capture from the engine side with
STRATA_JEVROWS_TRACE=1: the trace line reports `jevpos=N` and the window cuts, and `first`/`last`
are the absolute positions the request named.  A 422 mentioning `BROWAT` means a named position was
NOT captured - i.e. the request's rows never ran, which is the failure this probe exists to catch.
"""
import json
import sys
import urllib.request

BASE = "http://10.89.0.57:8080/upstream/p-jevport-vision-3090"
KEY = [l.split("=", 1)[1].strip() for l in open("/opt/data/llm-swap/.container-env")
       if l.startswith("LLM_SWAP_API_KEY=")][0]

Q = {"q1": {"type": "noul", "instructions": "Is the loop a current threat?"},
     "q2": {"type": "choice", "instructions": "Which action first?",
            "criteria": {"inspect": None, "throttle": None, "shutdown": None}}}


def call(state, tag):
    body = {"state": state, "questions": Q}
    req = urllib.request.Request(BASE + "/v1/systemone", data=json.dumps(body).encode(),
                                 headers={"Content-Type": "application/json",
                                          "Authorization": "Bearer " + KEY})
    try:
        with urllib.request.urlopen(req, timeout=600) as r:
            j = json.load(r)
        print(f"[{tag}] http=200 prompt_tokens={j['usage']['prompt_tokens']} "
              f"readouts={j['usage']['readout_tokens']} "
              f"answers={ {k: (v.get('noul') or v.get('choice')) for k, v in j['answers'].items()} }")
        return j
    except urllib.error.HTTPError as e:
        print(f"[{tag}] http={e.code} {e.read().decode(errors='replace')[:300]}")
        return None


short = "Reactor at 84%. Coolant loop A nominal, loop B pressure rising slowly."
long = " ".join([short] * 200)          # long enough to exceed one prefill window
print(f"short state: {len(short)} chars | long state: {len(long)} chars")

a = call(short, "short")
b = call(long, "long")
if a and b:
    ok = bool(a["answers"] and b["answers"])
    print("\nVERDICT:", "both answered" if ok else "a request failed")
    print("Both must be 200: the long state proves the capture works across several verify windows.")
    print("Note this does NOT prove the rows are the right ones - pair it with a labelled run")
    print("(eval_mf_set.py / ab_arms_mf.py) against arm C on the same items.")
