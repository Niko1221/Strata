#!/usr/bin/env python3
"""2x2 A/B on the local Strata install: STRATA_ROUTE_TAIL_SKIP x STRATA_ADAPT_OVERLAP.

The config's "env" overrides the process environment for the engine (serve/server.py:2797),
so every arm gets its own config copy.  Prompts and the request shape come from
tools/ab_engine.py, so these numbers line up with the project's own A/B runs.
Bound to 127.0.0.1 with no API key: this must never be reachable from the LAN.
"""
import json
import os
import statistics
import subprocess
import sys
import time
import urllib.request
from pathlib import Path

ROOT = Path("/home/user/Strata")
OUT = Path("/tmp/strata_ab")
OUT.mkdir(exist_ok=True)
sys.path.insert(0, str(ROOT / "tools"))
from ab_engine import PROMPTS, long_prompt, post, wait_ready  # noqa: E402

PORT = 8091
ROUNDS = int(os.environ.get("AB_ROUNDS", "2"))
MAXNEW = 256
BASE = json.loads((ROOT / "strata-unsloth-ud-iq4_xs.json").read_text())

# A = what runs today, B = tail skip back to its default, C = overlap only, D = both
ARMS = [
    ("A_cur",  {"STRATA_HC_Q8": "1", "STRATA_ROUTE_TAIL_SKIP": "0"}),
    ("B_tail", {"STRATA_HC_Q8": "1"}),
    ("C_ovl",  {"STRATA_HC_Q8": "1", "STRATA_ROUTE_TAIL_SKIP": "0", "STRATA_ADAPT_OVERLAP": "1"}),
    ("D_both", {"STRATA_HC_Q8": "1", "STRATA_ADAPT_OVERLAP": "1"}),
]

REQS = [(f"chat{i}", p) for i, p in enumerate(PROMPTS)] + [("long", long_prompt(6000))]
NEEDLES = (("tail", "ROUTE_TAIL_SKIP"), ("ovl", "adaptive overlap"), ("vram", "VRAM free with everything loaded"))


def vram_used():
    out = subprocess.run(["nvidia-smi", "--query-gpu=memory.used", "--format=csv,noheader,nounits"],
                         capture_output=True, text=True).stdout.strip().splitlines()
    return int(out[0]) if out else 0


def wait_vram_free(mib=1500, timeout=300):
    t0 = time.time()
    while time.time() - t0 < timeout:
        if vram_used() < mib:
            return True
        time.sleep(3)
    return False


def run_arm(name, env, rnd):
    cfg = dict(BASE)
    cfg["port"] = PORT
    cfg["host"] = "127.0.0.1"
    cfg["env"] = dict(env)
    log = OUT / f"{name}_r{rnd}.log"
    cfg["log"] = str(log)
    cpath = OUT / f"{name}.json"
    cpath.write_text(json.dumps(cfg, indent=1))
    penv = dict(os.environ)
    penv.pop("STRATA_API_KEY", None)
    proc = subprocess.Popen([sys.executable, str(ROOT / "serve" / "server.py"), "--engine", "strata",
                             "--config", str(cpath), "--port", str(PORT)],
                            cwd=str(ROOT), env=penv, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    rows = []
    try:
        t0 = time.time()
        wait_ready(PORT, proc)
        ready = time.time() - t0
        for tag, prompt in REQS:
            r = post(PORT, prompt, MAXNEW)
            t = r.get("timings", {})
            rows.append({"round": rnd, "arm": name, "req": tag, "decode": t.get("predicted_per_second"),
                         "prompt": t.get("prompt_per_second"), "prompt_n": t.get("prompt_n"),
                         "n": (r.get("usage") or {}).get("completion_tokens")})
    finally:
        proc.terminate()
        try:
            proc.wait(120)
        except subprocess.TimeoutExpired:
            proc.kill()
    wait_vram_free()
    engaged = {}
    try:
        lines = log.read_text(errors="replace").splitlines()
        for key, needle in NEEDLES:
            hit = [ln for ln in lines if needle in ln]
            engaged[key] = hit[-1][:160] if hit else "(none)"
    except FileNotFoundError:
        for key, _ in NEEDLES:
            engaged[key] = "(no log)"
    return ready, rows, engaged


def main():
    all_rows, engage = [], {}
    for rnd in range(ROUNDS):
        seq = ARMS if rnd % 2 == 0 else ARMS[::-1]
        for name, env in seq:
            ready, rows, eng = run_arm(name, env, rnd)
            engage.setdefault(name, eng)
            all_rows.extend(rows)
            med = statistics.median([r["decode"] for r in rows if r["decode"]])
            print(f"round {rnd} {name:7s} ready {ready:5.0f}s  decode median {med:6.1f} tok/s  "
                  f"({len(rows)}/{len(REQS)} reqs)", flush=True)
    (OUT / "rows.jsonl").write_text("\n".join(json.dumps(r) for r in all_rows) + "\n")

    print("\n=== did the arm engage? (startup lines) ===")
    for name, _ in ARMS:
        for key, _ in NEEDLES:
            print(f"{name:7s} {key:5s} {engage[name].get(key)}")

    print("\n=== decode tok/s per request (median over rounds) ===")
    for tag, _ in REQS:
        cells = []
        for name, _ in ARMS:
            v = [r["decode"] for r in all_rows if r["arm"] == name and r["req"] == tag and r["decode"]]
            cells.append(f"{name}={statistics.median(v):6.1f}" if v else f"{name}=  n/a")
        print(f"{tag:6s} " + "  ".join(cells))

    print("\n=== pooled decode (all 4 requests) ===")
    base = statistics.median([r["decode"] for r in all_rows if r["arm"] == "A_cur" and r["decode"]])
    for name, _ in ARMS:
        v = [r["decode"] for r in all_rows if r["arm"] == name and r["decode"]]
        local = [r["decode"] for r in all_rows if r["arm"] == name and r["req"] != "long" and r["decode"]]
        print(f"{name:7s} pooled {statistics.median(v):6.1f}  (chat-only {statistics.median(local):6.1f})  "
              f"x{statistics.median(v)/base:.3f}")

    print("\n=== prompt tok/s on the long request ===")
    for name, _ in ARMS:
        v = [r["prompt"] for r in all_rows if r["arm"] == name and r["req"] == "long" and r["prompt"]]
        print(f"{name:7s} {statistics.median(v):7.1f}")


main()
