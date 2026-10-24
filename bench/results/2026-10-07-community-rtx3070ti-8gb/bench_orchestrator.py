"""Run the community harness (bench/results/2026-09-30-community-rtx-5090/benchmark.py, unchanged) against several
engine configurations on this PC, one server start per block, and record each start's engine lines and memory.

Every block: stop the server, write a config variant, start serve/server.py on it (no browser), wait until loaded,
snapshot memory, run the harness, keep its results. The last block also runs tools/needle_bench.py. At the end the
user's normal start script (run-iq2_xs.bat, the final config) is started again.
"""
import json
import os
import pathlib
import subprocess
import sys
import time

import psutil
import requests

STRATA = pathlib.Path(r"<repo>")
PY = str(STRATA / ".venv" / "Scripts" / "python.exe")
HARNESS = str(STRATA / "bench" / "results" / "2026-09-30-community-rtx-5090" / "benchmark.py")
PACK = r"<data-dir>\packs\iq2_xs"
HERE = pathlib.Path(__file__).resolve().parent
OUT = HERE / "bench"
URL = "http://127.0.0.1:8080"
BASE = json.loads((STRATA / "strata-iq2_xs.json").read_text(encoding="utf-8"))
ENV = dict(os.environ, PYTHONUTF8="1")

VARIANTS = {
    "A-32k-int8": dict(ctx=32768, kv="int8", resident=None),          # setup's default on this PC
    "B-128k-k8v4-res32k": dict(ctx=131072, kv="k8v4", resident=32768),  # setup --context 131072 --kv k8v4
    "C-128k-k8v4-res20k": dict(ctx=131072, kv="k8v4", resident=20480),  # B with --kv-resident 20480
    "D-128k-int8-res32k": dict(ctx=131072, kv="int8", resident=32768),  # setup --context 131072
    # isolation (after review): C's memory settings with B's 768-token prompt chunk, expert streaming off (E1) and
    # on (E2: the streaming threshold lowered below the chunk)
    "E1-C-prefill768": dict(ctx=131072, kv="k8v4", resident=20480, prefill="768"),
    "E2-C-prefill768-streammin512": dict(ctx=131072, kv="k8v4", resident=20480, prefill="768",
                                         env={"STRATA_PREFILL_STREAM_MIN": "512"}),
}
# (block label, variant, [(targets, runs)], needles)
BLOCKS = [
    ("1-A", "A-32k-int8", [("4096,16384", 5)], False),
    ("2-B", "B-128k-k8v4-res32k", [("4096,16384", 5)], False),
    ("3-C", "C-128k-k8v4-res20k", [("4096,16384", 5)], False),
    ("4-D", "D-128k-int8-res32k", [("4096,16384", 5)], False),
    ("5-B", "B-128k-k8v4-res32k", [("4096,16384", 5)], False),
    ("6-C", "C-128k-k8v4-res20k", [("4096,16384,32768", 5), ("128000", 3)], True),
    # second session, reversed order relative to B -> C
    ("7-E1", "E1-C-prefill768", [("4096,16384", 5)], False),
    ("8-E2", "E2-C-prefill768-streammin512", [("4096,16384", 5)], False),
    ("9-C", "C-128k-k8v4-res20k", [("4096,16384", 5)], False),
    ("10-B", "B-128k-k8v4-res32k", [("4096,16384", 5)], False),
]


def log(*a):
    print(time.strftime("%H:%M:%S"), *a, flush=True)


def config_args(v):
    a = list(BASE["args"])
    for flag in ("--max-context", "--kv", "--kv-resident", "--conversation-cache-mib", "--conversation-cache-slots"):
        while flag in a:
            i = a.index(flag)
            del a[i:i + 2]
    a += ["--max-context", str(v["ctx"]), "--kv", v["kv"]]
    if v["resident"]:
        a += ["--kv-resident", str(v["resident"])]
    a += ["--conversation-cache-mib", "4096", "--conversation-cache-slots", "4"]
    if v.get("prefill"):
        a[a.index("--prefill") + 1] = v["prefill"]
    return a


def strata_procs():
    found = []
    for p in psutil.process_iter(["name", "cmdline"]):
        try:
            cmd = " ".join(p.info["cmdline"] or [])
            name = (p.info["name"] or "").lower()
            if name == "strata.exe" or ("server.py" in cmd and "--config" in cmd) or \
                    (name == "cmd.exe" and "run-iq2_xs.bat" in cmd):
                found.append(p)
        except (psutil.NoSuchProcess, psutil.AccessDenied):
            pass
    return found


def stop_server():
    try:
        requests.post(URL + "/unload", json={}, timeout=120)
    except Exception:
        pass
    time.sleep(2)
    procs = strata_procs()
    for p in procs:
        try:
            for c in p.children(recursive=True):
                c.kill()
            p.kill()
        except psutil.NoSuchProcess:
            pass
    t = time.time() + 60
    while strata_procs() and time.time() < t:
        time.sleep(1)
    time.sleep(3)


def snapshot():
    snap = {"ram_available_gib": round(psutil.virtual_memory().available / 2**30, 2)}
    for p in strata_procs():
        if (p.info["name"] or "").lower() == "strata.exe":
            try:
                snap["strata_rss_gib"] = round(p.memory_info().rss / 2**30, 2)
            except psutil.NoSuchProcess:
                pass
    q = subprocess.run(["nvidia-smi", "--query-gpu=memory.used,memory.total,temperature.gpu,driver_version",
                        "--format=csv,noheader,nounits"], capture_output=True, text=True)
    snap["nvidia_smi"] = q.stdout.strip()
    return snap


def start_server(block, name):
    d = OUT / block
    d.mkdir(parents=True, exist_ok=True)
    cfg = dict(BASE, args=config_args(VARIANTS[name]), log=str(d / "engine.log"))
    if VARIANTS[name].get("env"):
        cfg["env"] = dict(VARIANTS[name]["env"])
    cfg_path = d / "config.json"
    cfg_path.write_text(json.dumps(cfg, indent=1), encoding="utf-8")
    server_out = open(d / "server.out", "w", encoding="utf-8")
    subprocess.Popen([PY, str(STRATA / "serve" / "server.py"), "--engine", "strata", "--config", str(cfg_path),
                      "--port", "8080"], cwd=str(STRATA), stdout=server_out, stderr=subprocess.STDOUT, env=ENV,
                     creationflags=subprocess.CREATE_NO_WINDOW)
    t0 = time.time()
    while time.time() - t0 < 600:
        try:
            h = requests.get(URL + "/health", timeout=5).json()
            if h.get("loaded"):
                log(f"  loaded in {time.time() - t0:.0f}s, max_context {h.get('max_context')}")
                return d
        except Exception:
            pass
        time.sleep(3)
    raise RuntimeError("server did not become ready")


def engine_start_lines(d):
    keys = ("KV streaming:", "expert cache auto", "expert cache ", "prompt chunk", "-slot ring", "borrows",
            "VRAM free with", "session is up", "pool workers", "loaded ", "STRATA_PREFILL")
    lines = (d / "engine.log").read_text(encoding="utf-8", errors="replace").splitlines()
    return [l for l in lines if any(k in l for k in keys) and "hit rate" not in l][:20]


def main():
    OUT.mkdir(exist_ok=True)
    blocks = [b for b in BLOCKS if not sys.argv[1:] or b[0] in sys.argv[1:]]
    prev = OUT / "summary-all.json"
    summary = json.loads(prev.read_text(encoding="utf-8")) if prev.exists() else {}
    for label, name, sweeps, needles in blocks:
        log(f"block {label}: {name}")
        stop_server()
        d = start_server(label, name)
        time.sleep(5)
        info = {"variant": name, "settings": VARIANTS[name], "after_load": snapshot(),
                "engine_start": engine_start_lines(d), "sweeps": {}}
        for targets, runs in sweeps:
            sub = d / f"targets-{targets.replace(',', '-')}"
            log(f"  harness targets {targets} x{runs}")
            r = subprocess.run([PY, HARNESS, "--root", str(STRATA), "--pack", PACK, "--url", URL, "--out", str(sub),
                                "--targets", targets, "--runs", str(runs)], cwd=str(STRATA), env=ENV,
                               capture_output=True, text=True, encoding="utf-8", errors="replace")
            (sub / "harness.out").write_text(r.stdout + "\n--- stderr ---\n" + r.stderr, encoding="utf-8")
            if r.returncode != 0:
                log(f"  harness FAILED ({r.returncode}): {r.stderr[-400:]}")
                info["sweeps"][targets] = {"error": r.stderr[-2000:]}
                continue
            info["sweeps"][targets] = json.loads((sub / "summary.json").read_text(encoding="utf-8"))
            for t, s in info["sweeps"][targets].items():
                log(f"    {t}: prefill {s['prefill_tok_s']['median']:.0f} ({s['prefill_tok_s']['min']:.0f}-"
                    f"{s['prefill_tok_s']['max']:.0f}) decode {s['decode_tok_s']['median']:.1f} "
                    f"({s['decode_tok_s']['min']:.1f}-{s['decode_tok_s']['max']:.1f}) ttft {s['client_ttft_s']['median']:.1f}s")
        info["after_runs"] = snapshot()
        if needles:
            log("  needle_bench 32k,128k x depths 10,50,90")
            r = subprocess.run([PY, str(STRATA / "tools" / "needle_bench.py"), "--url", URL, "--lengths", "32k,128k",
                                "--depths", "10,50,90", "--timeout", "3600", "--out", str(d / "needles.json")],
                               cwd=str(STRATA), env=ENV, capture_output=True, text=True, encoding="utf-8",
                               errors="replace")
            (d / "needles.out").write_text(r.stdout + "\n--- stderr ---\n" + r.stderr, encoding="utf-8")
            log("  needles:", r.stdout.strip().splitlines()[-3:] if r.stdout.strip() else r.stderr[-300:])
        summary[label] = info
        (OUT / "summary-all.json").write_text(json.dumps(summary, indent=1), encoding="utf-8")
    log("restoring the normal start (run-iq2_xs.bat)")
    stop_server()
    subprocess.Popen(["cmd", "/c", "start", "", str(STRATA / "run-iq2_xs.bat")], cwd=str(STRATA))
    log("done")


if __name__ == "__main__":
    main()
