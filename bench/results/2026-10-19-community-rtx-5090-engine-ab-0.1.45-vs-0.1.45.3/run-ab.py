"""One A/B side: start the server with a config, wait for the model, run the bench suite, stop.

The bench tool's own "start the server" path is broken (tools/strata_bench.py:990
references an undefined `port`, so it dies with a NameError), so the server is started
here and the bench is pointed at it with --url.  The bench still reads the machine and
engine facts from the running server, so the compare table is complete.

usage: ab_run.py <config.json> <port> <out-dir>
"""
import json
import os
import subprocess
import sys
import time
import urllib.request

ROOT = r"D:\Strata"
cfg_path, port, out = sys.argv[1], int(sys.argv[2]), sys.argv[3]
KEY = "sk-ab-test"
BASE = f"http://127.0.0.1:{port}"

env = dict(os.environ)
env.update({
    "STRATA_HC_Q8": "1",
    "STRATA_EXCHANGE_ROTATE": "1",
    "STRATA_ATTN_DAILY": "1",
    "STRATA_EXPDEC": "1",
    "STRATA_HC_UPMIX": "1",
    "STRATA_QSA_SPLIT_ATTN": "1",
})

os.makedirs(out, exist_ok=True)
server_log = open(f"{out}.server.log", "w", encoding="utf-8")
print(f"starting the server: {cfg_path} on {BASE}", flush=True)
proc = subprocess.Popen(
    [sys.executable, f"{ROOT}\\serve\\server.py", "--engine", "strata", "--config", cfg_path,
     "--host", "127.0.0.1", "--port", str(port), "--api-key", KEY],
    cwd=ROOT, env=env, stdout=server_log, stderr=subprocess.STDOUT)


def health(timeout=5):
    req = urllib.request.Request(f"{BASE}/health", headers={"Authorization": f"Bearer {KEY}"})
    with urllib.request.urlopen(req, timeout=timeout) as r:
        return json.loads(r.read().decode("utf-8", "replace"))


t0 = time.time()
try:
    while True:
        if proc.poll() is not None:
            print(f"the server exited with {proc.returncode}; see {out}.server.log", flush=True)
            raise SystemExit(1)
        try:
            h = health()
            if h.get("loaded"):
                break
        except Exception:
            pass
        if time.time() - t0 > 1800:
            raise SystemExit("the model did not load within 30 minutes")
        time.sleep(5)
    print(f"model loaded in {time.time() - t0:.0f}s", flush=True)

    bench = subprocess.run(
        [sys.executable, f"{ROOT}\\tools\\strata_bench.py", "--url", BASE, "--api-key", KEY,
         "--out", out, "--runs", "3", "--long"],
        cwd=ROOT, env=env)
    print(f"bench exit code: {bench.returncode}", flush=True)
finally:
    proc.terminate()
    try:
        proc.wait(timeout=60)
    except subprocess.TimeoutExpired:
        proc.kill()
    server_log.close()
    print("server stopped", flush=True)
