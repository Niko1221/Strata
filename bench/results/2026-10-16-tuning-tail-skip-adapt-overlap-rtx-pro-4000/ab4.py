#!/usr/bin/env python3
"""Round 4: STRATA_HC_Q8 (VRAM vs verify-read speed) and the prefill chunk, on the winner.

The agent workload here sends 43K-77K token prompts, so the long request is ~32K tokens
(four chunks at the default 8192) to make the chunk size visible.
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
sys.path.insert(0, str(ROOT / "tools"))
from ab_engine import PROMPTS, long_prompt, post, wait_ready  # noqa: E402

PORT = 8093
ROUNDS = int(os.environ.get("AB_ROUNDS", "2"))
MAXNEW = 200
BASE = json.loads((ROOT / "strata-unsloth-ud-iq4_xs.json").read_text())
ENV = {"STRATA_HC_Q8": "1", "STRATA_ADAPT_OVERLAP": "1"}

ARMS = [
    ("base", {}, []),
    ("hc0",  {"STRATA_HC_Q8": "0"}, []),
    ("pf16", {}, ["--prefill", "auto:16384"]),
    ("pf32", {}, ["--prefill", "auto:32768"]),
]

REQS = [("chat0", PROMPTS[0]), ("chat1", PROMPTS[1]), ("long32k", long_prompt(24000))]


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


def run_arm(name, env_over, extra, rnd):
    cfg = dict(BASE)
    cfg["port"] = PORT
    cfg["host"] = "127.0.0.1"
    cfg["env"] = dict(ENV, **env_over)
    cfg["args"] = list(BASE["args"]) + list(extra)
    log = OUT / f"r4_{name}_r{rnd}.log"
    cfg["log"] = str(log)
    (OUT / f"r4_{name}.json").write_text(json.dumps(cfg, indent=1))
    penv = dict(os.environ)
    penv.pop("STRATA_API_KEY", None)
    proc = subprocess.Popen([sys.executable, str(ROOT / "serve" / "server.py"), "--engine", "strata",
                             "--config", str(OUT / f"r4_{name}.json"), "--port", str(PORT)],
                            cwd=str(ROOT), env=penv, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    rows = []
    try:
        wait_ready(PORT, proc)
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
    txt = log.read_text(errors="replace") if log.exists() else ""
    m = re.findall(r"expert cache (\d+) slots", txt)
    chunk = re.findall(r"prompt chunk auto: (\d+) tokens", txt)
    return rows, (m[-1] if m else None), (chunk[-1] if chunk else None)


def main():
    all_rows, meta = [], {}
    for rnd in range(ROUNDS):
        seq = ARMS if rnd % 2 == 0 else ARMS[::-1]
        for name, env_over, extra in seq:
            rows, slots, chunk = run_arm(name, env_over, extra, rnd)
            meta.setdefault(name, (slots, chunk))
            all_rows.extend(rows)
            dec = [r["decode"] for r in rows if r["decode"]]
            p32 = [r["prompt"] for r in rows if r["req"] == "long32k" and r["prompt"]]
            print(f"round {rnd} {name:5s} slots={slots} chunk={chunk}  decode median {statistics.median(dec):6.1f}  "
                  f"32K prompt {statistics.median(p32):7.1f} tok/s", flush=True)
    (OUT / "rows_r4.jsonl").write_text("\n".join(json.dumps(r) for r in all_rows) + "\n")

    print("\n=== medians ===")
    base_p = None
    for name, _, _ in ARMS:
        dec = statistics.median([r["decode"] for r in all_rows if r["arm"] == name and r["decode"]])
        chat = statistics.median([r["decode"] for r in all_rows if r["arm"] == name and r["req"] != "long32k" and r["decode"]])
        p32 = statistics.median([r["prompt"] for r in all_rows if r["arm"] == name and r["req"] == "long32k" and r["prompt"]])
        base_p = p32 if name == "base" else base_p
        slots, chunk = meta[name]
        print(f"{name:5s} slots={slots} chunk={chunk}  decode {dec:6.1f} (chat {chat:6.1f})  "
              f"32K prompt {p32:7.1f} tok/s  x{p32/base_p:.3f} vs base")


main()
