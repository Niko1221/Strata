"""Re-run the 0.1.40.3 report's method on 0.2.0-beta (tag 3d1fb48, release engine 0.1.45.3): the arms are the configs
0.2.0-beta's setup wrote (+ the hand-added conversation cache), one server start per block, the unchanged community
harness, tools/needle_bench.py and the task check where the plan asks for them, and a monitor that samples the GPU and
the other processes' CPU every 2 s so a disturbed request can be found.

    beta_orchestrator.py smoke | session1 | session2 | e | taskcheck
"""
import csv
import json
import os
import pathlib
import shutil
import subprocess
import sys
import threading
import time

import psutil
import requests

STRATA = pathlib.Path(r"<repo>")
PY = str(STRATA / ".venv" / "Scripts" / "python.exe")
HARNESS = str(STRATA / "bench" / "results" / "2026-09-30-community-rtx-5090" / "benchmark.py")
NEEDLE = str(STRATA / "tools" / "needle_bench.py")
PACK = r"<data-dir>\packs\iq2_xs"
W = pathlib.Path(__file__).resolve().parent
ARMS = W / "configs-arms"
OUT = W / "runs"
URL = "http://127.0.0.1:8080"
ENV = dict(os.environ, PYTHONUTF8="1")
OURS = {"strata.exe", "python.exe", "system idle process", "idle"}

S = "4096,16384"
PLANS = {   # (block, arm, [(targets, runs)], needles, task check)
    "smoke": [("00-smoke-A", "A", [("1024", 1)], False, False)],
    "session1": [("01-A", "A", [(S, 5)], False, False), ("02-B", "B", [(S, 5)], False, False),
                 ("03-C", "C", [(S, 5)], False, False), ("04-D", "D", [(S, 5)], False, False),
                 ("05-B", "B", [(S, 5)], False, False),
                 ("06-C", "C", [("4096,16384,32768", 5), ("128000", 3)], True, False)],
    "session2": [("07-C", "C", [(S, 5)], False, False), ("08-B", "B", [(S, 5)], False, False),
                 ("09-B-share-off", "B-share-off", [(S, 5)], False, False)],
    "e": [("10-E1", "E1", [(S, 5)], False, False), ("11-E2", "E2", [(S, 5)], False, False)],
    "taskcheck": [("12-C-taskcheck", "C", [], False, True)],
}


def log(*a):
    print(time.strftime("%H:%M:%S"), *a, flush=True)


class Monitor(threading.Thread):
    def __init__(self, path):
        super().__init__(daemon=True)
        self.path, self.stop_flag = path, threading.Event()

    def run(self):
        prev = {}
        psutil.cpu_percent(None)
        with open(self.path, "w", newline="", encoding="utf-8") as f:
            w = csv.writer(f)
            w.writerow(["t", "gpu_util", "sm_mhz", "power_w", "temp_c", "throttle", "cpu_total_pct", "other_cpu_s",
                        "other_top"])
            while not self.stop_flag.wait(2.0):
                q = subprocess.run(["nvidia-smi", "--query-gpu=utilization.gpu,clocks.sm,power.draw,temperature.gpu,"
                                    "clocks_throttle_reasons.active", "--format=csv,noheader,nounits"],
                                   capture_output=True, text=True).stdout.strip().split(", ")
                cur, by_name = {}, {}
                for p in psutil.process_iter(["name", "cpu_times"]):
                    try:
                        ct = p.info["cpu_times"]
                        cur[p.pid] = (p.info["name"] or "", ct.user + ct.system)
                    except (psutil.NoSuchProcess, psutil.AccessDenied, TypeError):
                        pass
                for pid, (name, s) in cur.items():
                    if name.lower() in OURS or pid not in prev or prev[pid][0] != name:
                        continue
                    by_name[name] = by_name.get(name, 0.0) + max(0.0, s - prev[pid][1])
                prev = cur
                top = sorted(by_name.items(), key=lambda kv: -kv[1])[:3]
                w.writerow([round(time.time(), 1), *(q + [""] * (5 - len(q)))[:5], psutil.cpu_percent(None),
                            round(sum(by_name.values()), 2), ";".join(f"{n}:{s:.2f}" for n, s in top)])
                f.flush()


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
    for p in strata_procs():
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


def arm_config(arm):
    if arm in ("E1", "E2"):          # C with the chunk fixed at 768 (E2: streaming allowed at 768), as in 0.1.40.3
        c = json.loads((ARMS / "C.json").read_text(encoding="utf-8"))
        a = c["args"]
        a[a.index("--prefill") + 1] = "768"
        if arm == "E2":
            c["env"] = {**c.get("env", {}), "STRATA_PREFILL_STREAM_MIN": "512"}
        return c
    return json.loads((ARMS / f"{arm}.json").read_text(encoding="utf-8"))


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


def start_server(d, arm):
    cfg = arm_config(arm)
    cfg["log"] = str(d / "engine.log")
    (d / "config.json").write_text(json.dumps(cfg, indent=1), encoding="utf-8")
    out = open(d / "server.out", "w", encoding="utf-8")
    subprocess.Popen([PY, str(STRATA / "serve" / "server.py"), "--engine", "strata", "--config", str(d / "config.json"),
                      "--port", "8080"], cwd=str(STRATA), stdout=out, stderr=subprocess.STDOUT, env=ENV,
                     creationflags=subprocess.CREATE_NO_WINDOW)
    t0 = time.time()
    while time.time() - t0 < 900:
        try:
            if requests.get(URL + "/health", timeout=5).json().get("loaded"):
                log(f"  loaded in {time.time() - t0:.0f}s")
                return True
        except Exception:
            pass
        time.sleep(3)
    log("  server did not become ready")
    return False


def plan_lines(d):
    keys = ("KV streaming:", "expert cache auto", "expert cache ", "prompt chunk", "-slot ring", "borrows",
            "VRAM free with", "session is up", "pool workers", "loaded ", "CPU share", "STRATA_PREFILL",
            "large pages")
    lines = (d / "engine.log").read_text(encoding="utf-8", errors="replace").splitlines()
    return [l for l in lines if any(k in l for k in keys) and "hit rate" not in l and "block reads" not in l][:24]


def main():
    plan = sys.argv[1]
    OUT.mkdir(exist_ok=True)
    summ_path = OUT / "summary-all.json"
    summary = json.loads(summ_path.read_text(encoding="utf-8")) if summ_path.exists() else {}
    for label, arm, sweeps, needles, task in PLANS[plan]:
        d = OUT / label
        d.mkdir(exist_ok=True)
        log(f"block {label}: arm {arm}")
        stop_server()
        info = {"arm": arm, "sweeps": {}, "started": time.strftime("%Y-%m-%d %H:%M:%S")}
        if not start_server(d, arm):
            info["error"] = "server did not become ready"
            summary[label] = info
            summ_path.write_text(json.dumps(summary, indent=1), encoding="utf-8")
            continue
        time.sleep(5)
        info["after_load"] = snapshot()
        info["plan"] = plan_lines(d)
        mon = Monitor(d / "monitor.csv")
        mon.start()
        for targets, runs in sweeps:
            sub = d / f"targets-{targets.replace(',', '-')}"
            log(f"  harness targets {targets} x{runs}")
            r = subprocess.run([PY, HARNESS, "--root", str(STRATA), "--pack", PACK, "--url", URL, "--out", str(sub),
                                "--targets", targets, "--runs", str(runs)], cwd=str(STRATA), env=ENV,
                               capture_output=True, text=True, encoding="utf-8", errors="replace")
            sub.mkdir(exist_ok=True)
            (sub / "harness.out").write_text(r.stdout + "\n--- stderr ---\n" + r.stderr, encoding="utf-8")
            if r.returncode != 0:
                log(f"  harness FAILED ({r.returncode}): {r.stderr[-400:]}")
                info["sweeps"][targets] = {"error": r.stderr[-3000:]}
                continue
            s = json.loads((sub / "summary.json").read_text(encoding="utf-8"))
            info["sweeps"][targets] = s
            for t, v in s.items():
                log(f"    {t}: prefill {v['prefill_tok_s']['median']:.1f} ({v['prefill_tok_s']['min']:.1f}-"
                    f"{v['prefill_tok_s']['max']:.1f}) decode {v['decode_tok_s']['median']:.1f} "
                    f"({v['decode_tok_s']['min']:.1f}-{v['decode_tok_s']['max']:.1f}) ttft {v['client_ttft_s']['median']:.1f}s")
        if needles:
            log("  needle_bench 32k,128k x depths 10,50,90")
            r = subprocess.run([PY, NEEDLE, "--url", URL, "--lengths", "32k,128k", "--depths", "10,50,90",
                                "--timeout", "3600", "--out", str(d / "needles.json")], cwd=str(STRATA), env=ENV,
                               capture_output=True, text=True, encoding="utf-8", errors="replace")
            (d / "needles.out").write_text(r.stdout + "\n--- stderr ---\n" + r.stderr, encoding="utf-8")
            log("  needles:", (r.stdout.strip().splitlines() or [r.stderr[-300:]])[-1])
        if task:
            log("  task check (13 prompts)")
            r = subprocess.run([PY, "-X", "utf8", str(W / "task_check.py")], cwd=str(W),
                               env={**ENV, "TASK_OUT": str(d / "task-check-C.json")},
                               capture_output=True, text=True, encoding="utf-8", errors="replace")
            (d / "task-check.out").write_text(r.stdout + "\n--- stderr ---\n" + r.stderr, encoding="utf-8")
            sc = subprocess.run([PY, "-I", "-X", "utf8", str(W / "check_results.py"), str(d / "task-check-C.json")],
                                capture_output=True, text=True, encoding="utf-8", errors="replace")
            (d / "task-check-C.scored.txt").write_text(sc.stdout + ("\n--- stderr ---\n" + sc.stderr if sc.stderr else ""),
                                                      encoding="utf-8")
            log("  scored:", " | ".join(l for l in sc.stdout.splitlines() if l.startswith(("code_expr", "json_strict", "poem"))))
        mon.stop_flag.set()
        mon.join(5)
        info["after_runs"] = snapshot()
        summary[label] = info
        summ_path.write_text(json.dumps(summary, indent=1), encoding="utf-8")
    stop_server()
    log("done")


if __name__ == "__main__":
    main()
