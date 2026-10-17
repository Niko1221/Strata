#!/usr/bin/env python3
"""Concurrency A/B: what --slots 2 buys and what it costs, on the winning config.

2x2: slots {1,2} x concurrent requests {1,2}.  Requests are the project's own chat prompts
with a unique tag so no conversation cache helps.  Each arm gets its own config copy
(127.0.0.1, no API key) and the config env wins over the process env.
"""
import concurrent.futures as cf
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
sys.path.insert(0, str(ROOT / "tools"))
from ab_engine import PROMPTS, post, wait_ready  # noqa: E402

PORT = 8092
ROUNDS = int(os.environ.get("AB_ROUNDS", "2"))
MAXNEW = 256
BASE = json.loads((ROOT / "strata-unsloth-ud-iq4_xs.json").read_text())
ENV = {"STRATA_HC_Q8": "1", "STRATA_ADAPT_OVERLAP": "1"}

ARMS = [
    ("s1c1", ["--slots", "1"], 1),
    ("s1c2", ["--slots", "1"], 2),
    ("s2c2", ["--slots", "2"], 2),
    ("s2c1", ["--slots", "2"], 1),
]


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


def burst(tag: str, n: int) -> dict:
    prompts = [f"[{tag}-{i}] {PROMPTS[i % len(PROMPTS)]}" for i in range(n)]
    t0 = time.time()
    with cf.ThreadPoolExecutor(max_workers=n) as ex:
        res = list(ex.map(lambda p: post(PORT, p, MAXNEW), prompts))
    wall = time.time() - t0
    dec = [r["timings"]["predicted_per_second"] for r in res]
    ntok = [(r.get("usage") or {}).get("completion_tokens") or 0 for r in res]
    return {"wall": wall, "total_tok": sum(ntok), "agg": sum(ntok) / wall if wall else 0,
            "per_req": statistics.mean(dec), "min_req": min(dec), "n": n}


def run_arm(name, extra, conc, rnd):
    cfg = dict(BASE)
    cfg["port"] = PORT
    cfg["host"] = "127.0.0.1"
    cfg["env"] = dict(ENV)
    cfg["args"] = list(BASE["args"]) + list(extra)
    log = OUT / f"r3_{name}_r{rnd}.log"
    cfg["log"] = str(log)
    (OUT / f"r3_{name}.json").write_text(json.dumps(cfg, indent=1))
    penv = dict(os.environ)
    penv.pop("STRATA_API_KEY", None)
    proc = subprocess.Popen([sys.executable, str(ROOT / "serve" / "server.py"), "--engine", "strata",
                             "--config", str(OUT / f"r3_{name}.json"), "--port", str(PORT)],
                            cwd=str(ROOT), env=penv, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    try:
        wait_ready(PORT, proc)
        time.sleep(2)
        r = burst(f"{name}r{rnd}", conc)
    finally:
        proc.terminate()
        try:
            proc.wait(120)
        except subprocess.TimeoutExpired:
            proc.kill()
    wait_vram_free()
    slots = None
    try:
        import re
        m = re.findall(r"expert cache (\d+) slots", log.read_text(errors="replace"))
        slots = m[-1] if m else None
    except FileNotFoundError:
        pass
    r["slots"] = slots
    return r


def main():
    rows = {name: [] for name, _, _ in ARMS}
    for rnd in range(ROUNDS):
        seq = ARMS if rnd % 2 == 0 else ARMS[::-1]
        for name, extra, conc in seq:
            r = run_arm(name, extra, conc, rnd)
            rows[name].append(r)
            print(f"round {rnd} {name}  conc={conc}  wall {r['wall']:5.2f}s  aggregate {r['agg']:6.1f} tok/s  "
                  f"per-request mean {r['per_req']:6.1f}  worst {r['min_req']:6.1f}  slots={r['slots']}", flush=True)

    print("\n=== medians over rounds ===")
    solo = statistics.median([r["per_req"] for r in rows["s1c1"]])
    print(f"{'arm':6s} {'conc':>4s} {'slots':>5s} {'wall s':>7s} {'agg tok/s':>10s} {'per-req':>8s} {'worst':>7s}  {'vs solo':>8s}")
    for name, _, conc in ARMS:
        v = rows[name]
        per = statistics.median([r["per_req"] for r in v])
        print(f"{name:6s} {conc:4d} {str(v[0]['slots']):>5s} {statistics.median([r['wall'] for r in v]):7.2f} "
              f"{statistics.median([r['agg'] for r in v]):10.1f} {per:8.1f} "
              f"{statistics.median([r['min_req'] for r in v]):7.1f}  x{per/solo:.3f}")


main()
