#!/usr/bin/env python3
"""Is the v0.1.42 thinking-loop behaviour the release's `STRATA_ROUTE_TAIL_SKIP=7` default?

0.1.42 turned the router tail skip ON by default on CUDA serve (docs/DETAILS.md): a *missed* expert that every
token of a verify window ranks at rank >= 7 is neither copied nor computed, so its contribution is zero.  The docs
measure a small but real answer change vs 0.1.41 (teacher-forced KL ~0.006, top-1 agreement 97.7%) and
`STRATA_ROUTE_TAIL_SKIP=0` restores 0.1.41's answers byte for byte.  A logit perturbation that small is exactly the
kind of thing that tips a marginal reasoning chain into repetition - so this runs the SAME release binary over the
SAME battery twice, once with the default and once with the skip off, and counts answers and loops.

    python tools/kv_loop_ab.py            # release engine, q4_0 KV, 6 questions x 3 seeds per arm
"""
from __future__ import annotations

import json
import os
import re
import subprocess
import sys
import time
import urllib.request
from collections import Counter
from pathlib import Path

ROOT = Path("C:/Users/blsor/Strata")
DATA = Path("C:/Users/blsor/Strata-data")
PY = ROOT / ".venv/Scripts/python.exe"
KEY = (DATA / "strata_api_key.txt").read_text(encoding="utf-8").strip()
PORT = 8088
RELEASE = "C:\\Users\\blsor\\Strata\\engine\\strata.exe"
BASE = json.loads((ROOT / "strata-swift-iq3_xxs.json").read_text(encoding="utf-8"))

BATTERY = [
    ("What is 17 * 23? Reply with just the number.", ["391"]),
    ("What is 144 / 12? Reply with just the number.", ["12"]),
    ("What is the capital of Australia? Reply with just the city name.", ["canberra"]),
    ("How many days are in a leap year? Reply with just the number.", ["366"]),
    ("Which planet is closest to the Sun? Reply with just its name.", ["mercury"]),
    ("List the first three prime numbers, comma separated.", ["2, 3, 5", "2,3,5"]),
]
SEEDS = [11, 12, 13]

ARMS = [
    ("0.1.42 default (tail skip 7)", {}),
    ("0.1.42 STRATA_ROUTE_TAIL_SKIP=0", {"STRATA_ROUTE_TAIL_SKIP": "0"}),
]


def cfg_for(name: str) -> Path:
    cfg = json.loads(json.dumps(BASE))
    cfg["exe"] = RELEASE
    cfg["log"] = f"C:\\Users\\blsor\\Strata\\loop-ab-{re.sub(r'[^a-z0-9]+', '-', name)}.log"
    args, i, src = [], 0, cfg["args"]
    while i < len(src):
        if src[i] == "--max-context":
            args += ["--max-context", "32768"]; i += 2; continue
        if src[i] == "--kv-resident":
            i += 2; continue
        if src[i] == "--conversation-cache-mib":
            args += ["--conversation-cache-mib", "512"]; i += 2; continue
        args.append(src[i]); i += 1
    cfg["args"] = args
    cfg["port"] = PORT
    p = ROOT / f"loop-ab-{re.sub(r'[^a-z0-9]+', '-', name)}.json"
    p.write_text(json.dumps(cfg, indent=1), encoding="utf-8")
    return p


def wait_ready(timeout_s: int = 420) -> bool:
    end = time.time() + timeout_s
    while time.time() < end:
        try:
            req = urllib.request.Request(f"http://127.0.0.1:{PORT}/v1/models",
                                         headers={"Authorization": f"Bearer {KEY}"})
            with urllib.request.urlopen(req, timeout=5) as r:
                if r.status == 200:
                    return True
        except Exception:
            pass
        time.sleep(3)
    return False


def ask(prompt: str, seed: int, budget: int = 300, max_tokens: int = 600) -> dict:
    body = json.dumps({"model": "swift-1.5-iq3_xxs", "messages": [{"role": "user", "content": prompt}],
                       "temperature": 0, "max_tokens": max_tokens, "reasoning_budget_tokens": budget,
                       "seed": seed}).encode()
    req = urllib.request.Request(f"http://127.0.0.1:{PORT}/v1/chat/completions", data=body,
                                 headers={"Authorization": f"Bearer {KEY}", "Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=600) as r:
        return json.loads(r.read())


def looped(reply: dict) -> bool:
    """More than half of the reply's sentences are repeats - the repetition mode seen in the failing runs."""
    msg = reply["choices"][0]["message"]
    text = ((msg.get("reasoning_content") or "") + "\n" + (msg.get("content") or "")).strip()
    sentences = [s.strip().lower() for s in re.split(r"(?<=[.!?])\s+|\n+", text) if len(s.strip()) > 8]
    if len(sentences) < 6:
        return False
    counts = Counter(sentences)
    dup = sum(c - 1 for c in counts.values() if c > 1)
    return dup / max(1, len(sentences)) > 0.5


def score(reply: dict, accepted: list[str]) -> bool:
    msg = reply["choices"][0]["message"]
    text = ((msg.get("content") or "") + "\n" + (msg.get("reasoning_content") or "")).lower()
    return any(a in text for a in accepted)


def main() -> int:
    out = {}
    for name, env in ARMS:
        cfg = cfg_for(name)
        e = dict(os.environ)
        e.update(env)
        log = ROOT / f"loop-ab-run-{re.sub(r'[^a-z0-9]+', '-', name)}.log"
        with open(log, "w", encoding="utf-8") as fh:
            proc = subprocess.Popen([str(PY), str(ROOT / "serve/server.py"), "--engine", "strata",
                                     "--config", str(cfg), "--port", str(PORT)],
                                    stdout=fh, stderr=subprocess.STDOUT, env=e, cwd=str(ROOT))
        replies = []
        try:
            if not wait_ready():
                print(f"{name}: engine never came up", flush=True)
                continue
            for seed in SEEDS:
                for q, accepted in BATTERY:
                    r = ask(q, seed)
                    replies.append({"q": q, "seed": seed, "accepted": accepted,
                                    "content": r["choices"][0]["message"].get("content"),
                                    "thinking": r["choices"][0]["message"].get("reasoning_content"),
                                    "finish": r["choices"][0].get("finish_reason"),
                                    "timings": r.get("timings"), "usage": r.get("usage"),
                                    "ok": score(r, accepted), "looped": looped(r)})
            ok = sum(x["ok"] for x in replies)
            lp = sum(x["looped"] for x in replies)
            sp = [x["timings"]["predicted_per_second"] for x in replies if x["timings"]]
            print(f"{name:34s} answers {ok}/{len(replies)}  loops {lp}/{len(replies)}  "
                  f"decode {round(sum(sp) / len(sp), 1) if sp else '-'} tok/s", flush=True)
            out[name] = replies
        finally:
            subprocess.run(["taskkill", "/F", "/IM", "strata.exe"], capture_output=True)
            proc.terminate()
            try:
                proc.wait(timeout=20)
            except Exception:
                proc.kill()
            time.sleep(4)

    (ROOT / "loop-ab-replies.json").write_text(json.dumps(out, indent=1), encoding="utf-8")
    print("\n==== tail skip A/B (release 0.1.42, q4_0) ====")
    for name, replies in out.items():
        ok = sum(x["ok"] for x in replies)
        lp = sum(x["looped"] for x in replies)
        print(f"{name:34s} answers {ok}/{len(replies)}  loops {lp}/{len(replies)}")
    print("raw replies: loop-ab-replies.json")
    return 0


if __name__ == "__main__":
    sys.exit(main())