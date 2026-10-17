#!/usr/bin/env python3
"""Round 2: --aux-cpus / --vram-reserve-mib / --kv-grow on top of the round-1 winner
(STRATA_ROUTE_TAIL_SKIP at its default + STRATA_ADAPT_OVERLAP=1).

Same instrument as ab.py: one config copy per arm (the config's env overrides the process
env), the project's own fixed prompts, interleaved rounds, 127.0.0.1 and no API key.
Also parses each arm's startup/request log for what actually changed (cache slots, hit rate,
free VRAM) -- the flags are only worth anything if they engage.
"""
import json
import os
import re
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
ENV = {"STRATA_HC_Q8": "1", "STRATA_ADAPT_OVERLAP": "1"}   # the round-1 winner

ARMS = [
    ("base", []),
    ("aux",  ["--aux-cpus", "auto"]),
    ("res",  ["--vram-reserve-mib", "896"]),
    ("kv",   ["--kv-grow"]),
]

REQS = [(f"chat{i}", p) for i, p in enumerate(PROMPTS)] + [("long", long_prompt(6000))]
PATTERNS = {
    "slots":    r"expert cache (\d+) slots",
    "vramfree": r"(\d+) MiB of VRAM free with everything loaded",
    "hitrate":  r"decode expert cache hit rate: ([\d.]+)%",
    "aux":      r"strata aux cpus: threads that are neither pool workers nor the host go to CPUs (\S+?) ",
}


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


def facts(log: Path) -> dict:
    try:
        txt = log.read_text(errors="replace")
    except FileNotFoundError:
        return {}
    out = {}
    for key, pat in PATTERNS.items():
        hits = re.findall(pat, txt)
        out[key] = hits[-1] if key == "slots" else (hits[0] if key == "vramfree" else (hits[-1] if hits else None))
    kv = [ln for ln in txt.splitlines() if "grow" in ln.lower() and "kv" in ln.lower()]
    out["kv_line"] = kv[-1][:120] if kv else None
    return out


def run_arm(name, extra, rnd):
    cfg = dict(BASE)
    cfg["port"] = PORT
    cfg["host"] = "127.0.0.1"
    cfg["env"] = dict(ENV)
    cfg["args"] = list(BASE["args"]) + list(extra)
    log = OUT / f"r2_{name}_r{rnd}.log"
    cfg["log"] = str(log)
    cpath = OUT / f"r2_{name}.json"
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
                         "prompt": t.get("prompt_per_second"), "prompt_n": t.get("prompt_n")})
    finally:
        proc.terminate()
        try:
            proc.wait(120)
        except subprocess.TimeoutExpired:
            proc.kill()
    wait_vram_free()
    return ready, rows, facts(log)


def main():
    all_rows, fact = [], {}
    for rnd in range(ROUNDS):
        seq = ARMS if rnd % 2 == 0 else ARMS[::-1]
        for name, extra in seq:
            ready, rows, f = run_arm(name, extra, rnd)
            fact.setdefault(name, f)
            all_rows.extend(rows)
            med = statistics.median([r["decode"] for r in rows if r["decode"]])
            print(f"round {rnd} {name:5s} ready {ready:5.0f}s  decode median {med:6.1f} tok/s  "
                  f"slots={f.get('slots')} vramfree={f.get('vramfree')} hit={f.get('hitrate')}", flush=True)
    (OUT / "rows_r2.jsonl").write_text("\n".join(json.dumps(r) for r in all_rows) + "\n")

    print("\n=== what each arm actually got ===")
    for name, _ in ARMS:
        f = fact[name]
        print(f"{name:5s} slots={f.get('slots')}  vram_free={f.get('vramfree')} MiB  "
              f"decode_hit={f.get('hitrate')}%  aux={f.get('aux')}")
        if f.get("kv_line"):
            print(f"      kv: {f['kv_line']}")

    print("\n=== decode tok/s per request (median over rounds) ===")
    for tag, _ in REQS:
        cells = []
        for name, _ in ARMS:
            v = [r["decode"] for r in all_rows if r["arm"] == name and r["req"] == tag and r["decode"]]
            cells.append(f"{name}={statistics.median(v):6.1f}" if v else f"{name}=  n/a")
        print(f"{tag:6s} " + "  ".join(cells))

    print("\n=== pooled decode, ratio to base ===")
    base = statistics.median([r["decode"] for r in all_rows if r["arm"] == "base" and r["decode"]])
    for name, _ in ARMS:
        v = [r["decode"] for r in all_rows if r["arm"] == name and r["decode"]]
        print(f"{name:5s} {statistics.median(v):6.1f} tok/s   x{statistics.median(v)/base:.3f}")

    print("\n=== prompt tok/s on the long request ===")
    for name, _ in ARMS:
        v = [r["prompt"] for r in all_rows if r["arm"] == name and r["req"] == "long" and r["prompt"]]
        print(f"{name:5s} {statistics.median(v):7.1f}")


main()
