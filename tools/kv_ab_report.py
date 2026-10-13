#!/usr/bin/env python3
"""Speed + accuracy A/B for the Strata KV cache, including the FP16 sliding window (kv_window.hpp).

Runs a small battery of questions with verifiable answers plus one long filler prompt for prefill timing, against
several arms of the engine (release binary and the locally built one, different --kv settings), and prints one
table of easy-to-read numbers:

    arm             answers   prompt tok/s   decode tok/s
    release q4_0      5/6        1234.5          121.3
    ...

Usage:  python tools/kv_ab_report.py            (writes each arm's raw replies next to the config)
"""
from __future__ import annotations

import json
import os
import re
import socket
import subprocess
import sys
import time
import urllib.error
import urllib.request
from pathlib import Path

ROOT = Path("C:/Users/blsor/Strata")
DATA = Path("C:/Users/blsor/Strata-data")
PY = ROOT / ".venv/Scripts/python.exe"
KEY = (DATA / "strata_api_key.txt").read_text(encoding="utf-8").strip()
PORT = 8087
BASE_CFG = json.loads((ROOT / "strata-swift-iq3_xxs.json").read_text(encoding="utf-8"))
RELEASE = "C:\\Users\\blsor\\Strata\\engine\\strata.exe"
DEV = "C:\\Users\\blsor\\Strata\\build\\strata.exe"

# (question, [accepted answer spellings]) - every answer is checkable without a judge
BATTERY = [
    ("What is 17 * 23? Reply with just the number.", ["391"]),
    ("What is 144 / 12? Reply with just the number.", ["12"]),
    ("What is the capital of Australia? Reply with just the city name.", ["canberra"]),
    ("What is the capital of Canada? Reply with just the city name.", ["ottawa"]),
    ("How many days are in a leap year? Reply with just the number.", ["366"]),
    ("Which planet is closest to the Sun? Reply with just its name.", ["mercury"]),
]

ARMS = [
    # name, engine, kv mode, env for the engine
    ("release q4_0", RELEASE, "q4_0", {}),
    ("dev q4_0", DEV, "q4_0", {}),
    ("dev q4_0+window", DEV, "q4_0", {"STRATA_KV_WINDOW": "128", "STRATA_KV_SINK": "8"}),
    ("dev fp16", DEV, "fp16", {}),
]


def arm_config(name: str, engine: str, kv: str) -> Path:
    cfg = json.loads(json.dumps(BASE_CFG))
    cfg["exe"] = engine
    cfg["log"] = f"C:\\Users\\blsor\\Strata\\kv-ab-{kv}-{abs(hash(name)) % 10000}.log"
    args, i, src = [], 0, cfg["args"]
    while i < len(src):
        if src[i] == "--max-context":
            args += ["--max-context", "32768"]; i += 2; continue
        if src[i] == "--kv-resident":
            i += 2; continue                      # fully resident: no 10 GB pinned host copy for a benchmark
        if src[i] == "--kv":
            args += ["--kv", kv]; i += 2; continue
        if src[i] == "--conversation-cache-mib":
            args += ["--conversation-cache-mib", "512"]; i += 2; continue
        args.append(src[i]); i += 1
    cfg["args"] = args
    cfg["port"] = PORT
    path = ROOT / f"kv-ab-{abs(hash(name)) % 10000}.json"
    path.write_text(json.dumps(cfg, indent=1), encoding="utf-8")
    return path


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


def kill_arm(proc: subprocess.Popen) -> None:
    subprocess.run(["taskkill", "/F", "/IM", "strata.exe"], capture_output=True)
    proc.terminate()
    try:
        proc.wait(timeout=20)
    except Exception:
        proc.kill()
    time.sleep(4)


def ask(prompt: str, max_tokens: int = 500, budget: int = 300) -> dict:
    body = json.dumps({
        "model": "swift-1.5-iq3_xxs",
        "messages": [{"role": "user", "content": prompt}],
        "temperature": 0, "max_tokens": max_tokens,
        "reasoning_budget_tokens": budget, "seed": 11,
    }).encode()
    req = urllib.request.Request(f"http://127.0.0.1:{PORT}/v1/chat/completions", data=body,
                                 headers={"Authorization": f"Bearer {KEY}", "Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=600) as r:
        return json.loads(r.read())


def score(reply: dict, accepted: list[str]) -> bool:
    msg = reply["choices"][0]["message"]
    text = ((msg.get("content") or "") + "\n" + (msg.get("reasoning_content") or "")).lower()
    return any(a in text for a in accepted)


def filler_prompt(nonce: int) -> str:
    # a long, unique prompt: no conversation-cache hit, so prompt_per_second is the real prefill rate
    para = ("The quick brown fox jumps over the lazy dog while the engineer measures throughput. "
            "Latency and memory bandwidth decide how many tokens a second a transformer can produce. ")
    return f"Benchmark nonce {nonce}. " + para * 90 + "\nSummarize the paragraph above in one short sentence."


def main() -> int:
    rows = []
    for name, engine, kv, env in ARMS:
        cfg = arm_config(name, engine, kv)
        e = dict(os.environ)
        e.update(env)
        log = ROOT / f"kv-ab-run-{re.sub(r'[^a-z0-9]+', '-', name)}.log"
        with open(log, "w", encoding="utf-8") as fh:
            proc = subprocess.Popen([str(PY), str(ROOT / "serve/server.py"), "--engine", "strata",
                                     "--config", str(cfg), "--port", str(PORT)],
                                    stdout=fh, stderr=subprocess.STDOUT, env=e, cwd=str(ROOT))
        try:
            if not wait_ready():
                rows.append((name, "engine never came up", None, None))
                continue
            good, prefill, decode = 0, None, []
            for q, accepted in BATTERY:
                reply = ask(q)
                good += score(reply, accepted)
                t = reply.get("timings") or {}
                if t.get("predicted_per_second"):
                    decode.append(t["predicted_per_second"])
            # prefill timing on a fresh, long prompt (unique nonce -> no cached prefix)
            r = ask(filler_prompt(int(time.time()) % 100000), max_tokens=32, budget=0)
            t = r.get("timings") or {}
            prefill = t.get("prompt_per_second")
            rows.append((name, f"{good}/{len(BATTERY)}", prefill,
                         (sum(decode) / len(decode)) if decode else None))
            print(f"{name:18s} answers {good}/{len(BATTERY)}  prompt {prefill} tok/s  "
                  f"decode {rows[-1][3] and round(rows[-1][3], 1)} tok/s", flush=True)
        finally:
            kill_arm(proc)

    print("\n==== KV A/B ====")
    print(f"{'arm':18s} {'answers':>8s} {'prompt tok/s':>13s} {'decode tok/s':>13s}")
    for name, answers, prefill, decode in rows:
        p = f"{prefill:.1f}" if isinstance(prefill, (int, float)) else "-"
        d = f"{decode:.1f}" if isinstance(decode, (int, float)) else "-"
        print(f"{name:18s} {answers:>8s} {p:>13s} {d:>13s}")
    return 0


if __name__ == "__main__":
    sys.exit(main())