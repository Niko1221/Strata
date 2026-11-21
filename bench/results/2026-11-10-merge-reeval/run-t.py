#!/usr/bin/env python
"""The small runtime comparison (campaign 2026-11-10-merge-reeval, deviation 7).

The timing numbers taken during the campaign were taken with foreign GPU load running and are
withdrawn.  This driver re-measures the prompt read time per needle on an idle machine, same
build, same haystack, same day, one arm at a time.

Five arms, so that the two questions separate:
  int8                  the fast path (int8 K/V, the tensor-core prompt attention)
  q4_0                  the fast path with a 4-bit codec (uniform grid, mode 4 accepted)
  q4lm8v                the level-table mode: the prompt fast path declines it, the decode
                        kernel runs instead (the suspected cause of the withdrawn 2.36x)
  q4lm8v+center         the same, with the centering (the offset subtraction and the V
                        add-back must not cost anything)
  q4lm8v+window         the same, with the exact-cell window (the overlay's own price)

The measure is the prompt read time per needle (the needle tool records prompt_tokens and the
wall time of the request); the generation is 40 tokens and negligible.
"""
import hashlib
import json
import os
import signal
import subprocess
import sys
import time
import urllib.error
import urllib.request
from pathlib import Path

REPO = Path(__file__).resolve().parents[3]   # bench/results/<ordner>/run-t.py -> the repo root
PY = sys.executable
URL = "http://127.0.0.1:7603"      # the needle tool appends /v1/chat/completions itself
OUT = Path(r"C:\Users\krone\git\Strata-data\bench\2026-11-10-merge-reeval\t")
BASE_CFG = REPO / "bench" / "configs" / "needle-swift-iq3_xxs-merged-int8.json"
TABLE = Path(r"C:\Users\krone\git\Strata-data\bench\2026-11-10-centering\center-table.bin")

# The window arm's ring must hold one prompt chunk: the engine's default chunk is 6,144 cells
# (the needle config does not set --prefill), so the ring needs STRATA_KV_WINDOW_BATCH 6,144 --
# the geometry setup writes at 128K.  With 512 (the quality arms' geometry, which use
# --prefill 512) the engine refuses the run loudly: "6144 cells in one batch do not fit the
# 768-row recent ring".  Overridable so that both configurations stay reproducible.
WINDOW_BATCH = os.environ.get("RUN_T_WINDOW_BATCH", "512")
ONLY = [a for a in os.environ.get("RUN_T_ONLY", "").split(",") if a]

# tag, --kv, extra env, expects the ring line
ARMS = [
    ("int8", "int8", {}, False),
    ("q4_0", "q4_0", {}, False),
    ("q4lm8v", "q4lm8v", {"STRATA_VERIFY_ASYNC_LAUNCH": "1"}, False),
    ("q4lm8v-center", "q4lm8v", {"STRATA_VERIFY_ASYNC_LAUNCH": "1",
                                  "STRATA_KV_CENTER": str(TABLE)}, False),
    ("q4lm8v-win", "q4lm8v", {"STRATA_VERIFY_ASYNC_LAUNCH": "1",
                               "STRATA_KV_WINDOW": "1", "STRATA_KV_WINDOW_S": "256",
                               "STRATA_KV_WINDOW_R": "64", "STRATA_KV_WINDOW_BATCH": WINDOW_BATCH}, True),
]


def wait_up(timeout=900):
    t0 = time.time()
    while time.time() - t0 < timeout:
        try:
            with urllib.request.urlopen(URL + "/health", timeout=5) as r:
                if r.status == 200:
                    return True
        except Exception:
            pass
        time.sleep(3)
    return False


def stop(server):
    if server.poll() is None:
        server.send_signal(signal.CTRL_BREAK_EVENT)
        try:
            server.wait(timeout=120)
        except subprocess.TimeoutExpired:
            server.kill()


def strata_running():
    raw = subprocess.run(["tasklist", "/FI", "IMAGENAME eq strata.exe"], capture_output=True).stdout
    out = raw.decode("utf-8", errors="replace")
    return any(line.lower().lstrip().startswith("strata.exe") for line in out.splitlines())


def wait_engine_gone(seconds=180):
    waited = 0
    while strata_running() and waited < seconds:
        time.sleep(5)
        waited += 5
    return not strata_running()


def write_cfg(tag, kv, env):
    cfg = json.loads(BASE_CFG.read_text(encoding="utf-8-sig"))
    args = cfg["args"]
    if "--kv" in args:
        args[args.index("--kv") + 1] = kv
    else:
        args += ["--kv", kv]
    cfg["env"] = dict(cfg.get("env", {}))
    cfg["env"].update(env)
    cfg["log"] = str(OUT / f"needle-{tag}.engine.log")
    path = OUT / f"cfg-{tag}.json"
    path.write_text(json.dumps(cfg, indent=1), encoding="utf-8")
    return path


def arm(tag, kv, env, window):
    print(f"=== arm {tag}: kv={kv} env={sorted(env)} ===", flush=True)
    cfg = write_cfg(tag, kv, env)
    h = hashlib.sha256()
    with open(json.loads(cfg.read_text(encoding="utf-8"))["exe"], "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    print(f"arm {tag}: engine sha256 {h.hexdigest()}", flush=True)
    if strata_running():
        print(f"arm {tag}: a strata.exe is already running; refusing to start (one GPU job at a time)", flush=True)
        return 1
    server = subprocess.Popen([PY, str(REPO / "serve" / "server.py"), "--engine", "strata",
                               "--config", str(cfg), "--port", "7603", "--no-api-key"],
                              cwd=str(REPO), creationflags=subprocess.CREATE_NEW_PROCESS_GROUP)
    if not wait_up():
        print(f"arm {tag}: server never came up", flush=True)
        stop(server)
        return 1
    print(f"arm {tag}: server up, needle_bench (32k x depths 10,50,90)", flush=True)
    done = subprocess.run([PY, str(REPO / "tools" / "needle_bench.py"), "--url", URL,
                           "--lengths", "32k", "--depths", "10,50,90",
                           "--out", str(OUT / f"needles-{tag}.json")], cwd=str(REPO))
    stop(server)
    if not wait_engine_gone():
        print(f"arm {tag}: the engine did not exit after the server stopped", flush=True)
        return 1
    text = (OUT / f"needle-{tag}.engine.log")
    log = text.read_text(encoding="utf-8", errors="replace") if text.exists() else ""
    ring = "ring sized" in log
    print(f"arm {tag}: ring line {'present' if ring else 'absent'} (expected "
          f"{'present' if window else 'absent'})", flush=True)
    rc = done.returncode
    if ring != window:
        rc = rc or 1
    time.sleep(30)   # the campaign rule: 30 s between engine starts
    return rc


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    rc = 0
    arms = [a for a in ARMS if not ONLY or a[0] in ONLY]
    for tag, kv, env, window in arms:
        rc = arm(tag, kv, env, window) or rc
    print("=== SUMMARY: prompt read time per needle (32k) ===", flush=True)
    for tag, _kv, _env, _w in arms:
        p = OUT / f"needles-{tag}.json"
        if not p.exists():
            print(f"{tag:18s} (no result)")
            continue
        rows = json.loads(p.read_text(encoding="utf-8"))
        times = [r.get("seconds", 0.0) for r in rows]
        toks = [r.get("prompt_tokens", 0) for r in rows]
        rate = sum(toks) / sum(times) if sum(times) else 0.0
        found = sum(1 for r in rows if r.get("found"))
        print(f"{tag:18s} {len(times)} needles, {found} found, "
              f"{' '.join(f'{t:.1f}' for t in times)} s, mean {rate:.1f} tok/s")
    return rc


if __name__ == "__main__":
    sys.exit(main())
