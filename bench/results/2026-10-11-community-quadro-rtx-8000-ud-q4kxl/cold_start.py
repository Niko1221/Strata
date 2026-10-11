#!/usr/bin/env python3
"""Cold-engine arm: restart the engine, time the load, then measure first-touch runs.

`POST /unload` ends the engine process (and the image encoder); `POST /load` starts a fresh
one and returns only once it is ready. The Python server keeps running, so the API endpoint
does not change and nothing else has to be restarted.

What this measures is a **cold engine with a warm OS file cache**: the model files stay in the
page cache, so the file reads are not cold. A cold *boot* needs a reboot and is a different
number; do not label this one as that. The report says which of the two it has.

Each length gets ONE run, because the point is first touch: the expert cache has not seen this
prompt's routing before, and neither has the conversation cache. Runs land in <out>/runs.jsonl
so they cannot be mixed into the warm table.

Usage: python3 cold_start.py --out cold
"""
import argparse
import json
import os
import re
import subprocess
import sys
import time
import urllib.error
import urllib.request

HERE = os.path.dirname(os.path.abspath(__file__))


def http(url, path, method="GET", body=None, timeout=1800):
    data = json.dumps(body).encode() if body is not None else None
    req = urllib.request.Request(url + path, data=data, method=method)
    if data is not None:
        req.add_header("Content-Type", "application/json")
    try:
        with urllib.request.urlopen(req, timeout=timeout) as r:
            raw = r.read().decode("utf-8", "replace")
            return r.status, (json.loads(raw) if raw.strip().startswith(("{", "[")) else raw.strip())
    except urllib.error.HTTPError as e:
        raw = e.read().decode("utf-8", "replace")
        try:
            raw = json.loads(raw)
        except ValueError:
            pass
        return e.code, raw


def engine_pid():
    """The engine child process, to prove the restart really happened."""
    out = subprocess.run(["pgrep", "-f", "engine/strata --serve"], capture_output=True, text=True)
    return out.stdout.split()


def gpu_used():
    out = subprocess.run(["nvidia-smi", "--query-gpu=memory.used", "--format=csv,noheader,nounits"],
                         capture_output=True, text=True)
    try:
        return int(out.stdout.strip().splitlines()[0])
    except (ValueError, IndexError):
        return None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--url", default="http://127.0.0.1:8080")
    ap.add_argument("--out", default=os.path.join(HERE, "cold"))
    ap.add_argument("--log", default="", help="engine log (default: the 'log' key of run-config.json)")
    ap.add_argument("--lengths", default="4k,32k,128000")
    ap.add_argument("--max-tokens", type=int, default=256)
    ap.add_argument("--load-timeout", type=int, default=1200, help="seconds to wait for the engine to come back")
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)

    log_path = a.log
    if not log_path:
        cfg = os.path.join(HERE, "run-config.json")
        if os.path.exists(cfg):
            log_path = json.load(open(cfg)).get("log", "")

    before_pid, before_gpu = engine_pid(), gpu_used()
    st, health = http(a.url, "/health")
    print(f"before: /health {st} loaded={health.get('loaded')} engine_pid={before_pid} gpu_used={before_gpu} MiB",
          flush=True)
    offset = os.path.getsize(log_path) if log_path and os.path.exists(log_path) else 0

    # 1. unload. 409 means a request is in flight; back off and try a few times.
    t_unload = time.time()
    for attempt in range(10):
        st, body = http(a.url, "/unload", "POST", {})
        print(f"  POST /unload -> {st} {body}", flush=True)
        if st == 200:
            break
        time.sleep(3)
    unload_s = round(time.time() - t_unload, 2)
    time.sleep(3)
    _, health_off = http(a.url, "/health")
    after_unload_pid, free_gpu = engine_pid(), gpu_used()
    print(f"  unloaded: loaded={health_off.get('loaded')} engine_pid={after_unload_pid} gpu_used={free_gpu} MiB",
          flush=True)

    # 2. load. The endpoint blocks until the engine is ready, so this wall time is the load.
    t_load = time.time()
    for attempt in range(3):
        st, body = http(a.url, "/load", "POST", {}, timeout=a.load_timeout)
        load_s = round(time.time() - t_load, 1)
        print(f"  POST /load (try {attempt + 1}) -> {st} {body} after {load_s} s", flush=True)
        if st == 200:
            break
        time.sleep(10)
    if st != 200:
        print(f"REFUSING to continue: the engine did not come back ({st} {body}). "
              f"Restart it by hand before measuring anything.", flush=True)
        return 1

    deadline = time.time() + 120
    loaded = False
    while time.time() < deadline:
        _, h = http(a.url, "/health")
        if h.get("loaded"):
            loaded = True
            break
        time.sleep(2)
    after_pid, after_gpu = engine_pid(), gpu_used()
    print(f"  ready: loaded={loaded} engine_pid {before_pid} -> {after_pid}, gpu_used={after_gpu} MiB", flush=True)
    if not loaded or after_pid == before_pid:
        print("REFUSING to continue: engine did not report loaded, or the process did not change.", flush=True)
        return 1

    # 3. the new session's own startup lines, from the byte offset taken before the unload.
    startup = {}
    if offset:
        with open(log_path, "rb") as f:
            f.seek(offset)
            new = f.read().decode("utf-8", "replace")
        session = os.path.join(a.out, "engine-cold-session.log")
        open(session, "w").write(new)
        want = ("engine started", "expert cache", "pinned cache complement", "weights loaded",
                "VRAM free with everything loaded", "resident experts", "draft head", "PCIe", "pool")
        startup["log_lines"] = [l.strip() for l in new.splitlines() if any(w in l for w in want)]
        for key, pat in (("resident_gib", r"resident ([\d.]+) GiB, pinned ([\d.]+) GiB in (\d+) s"),
                         ("weights_loaded_s", r"weights loaded \(([\d.]+) GiB[^)]*\) in ([\d.]+) s"),
                         ("vram_free_mib", r"(\d+) MiB of VRAM free with everything loaded")):
            m = re.search(pat, new)
            if m:
                startup[key] = m.group(1) if pat.count("(") == 1 else m.groups()
        print(f"  new session log: {session} ({len(new)} bytes)", flush=True)

    result = {
        "measured_at": time.strftime("%Y-%m-%d %H:%M:%S%z"),
        "kind": "cold engine, warm OS file cache (NOT a cold boot)",
        "unload_s": unload_s,
        "load_s": load_s,
        "engine_pid_before": before_pid,
        "engine_pid_after": after_pid,
        "gpu_used_mib_before": before_gpu,
        "gpu_used_mib_after_unload": free_gpu,
        "gpu_used_mib_after_load": after_gpu,
        "engine_startup": startup,
        "note": ("load_s is the wall time of POST /load, which blocks until the engine is ready. "
                 "The model files stay in the OS page cache across an unload, so the file reads "
                 "are not cold; a cold boot is a different, larger number."),
    }
    json.dump(result, open(os.path.join(a.out, "cold-start.json"), "w"), indent=1)
    print(json.dumps(result, indent=1)[:1500], flush=True)

    # 4. first-touch runs, one per length, into <out>/runs.jsonl.
    for text in a.lengths.split(","):
        cmd = [sys.executable, os.path.join(HERE, "benchmark.py"), "--url", a.url, "--out", a.out,
               "--lengths", text, "--runs", "1", "--max-tokens", str(a.max_tokens), "--settle", "3"]
        print(f"=== cold run {text} ===", flush=True)
        subprocess.run(cmd, check=False)
    return 0


if __name__ == "__main__":
    sys.exit(main())
