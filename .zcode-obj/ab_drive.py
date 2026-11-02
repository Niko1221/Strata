"""Drive the A/B: stop the running server, run arm A, run arm B, put the normal server back.

Only one engine may be on the GPU at a time - two engines make both slow, which is exactly the
artefact this test must not measure.  Each arm therefore gets the card to itself, and the user's
own 8080 server is restored at the end whatever happens.

Usage: python ab_drive.py
"""
import json
import os
import signal
import subprocess
import sys
import time
import urllib.request
from pathlib import Path

ROOT = Path(r"F:\Strata-dev")
PY = str(ROOT / ".venv" / "Scripts" / "python.exe")
SERVER = str(ROOT / "serve" / "server.py")
Z = ROOT / ".zcode-obj"
LOG = Z / "ab-drive.txt"


def say(msg):
    line = f"[{time.strftime('%H:%M:%S')}] {msg}"
    print(line, flush=True)
    with LOG.open("a", encoding="utf-8") as f:
        f.write(line + "\n")


def powershell(script):
    return subprocess.run(["powershell.exe", "-NoProfile", "-Command", script],
                          capture_output=True, text=True, timeout=300)


def strata_pids():
    out = powershell("Get-CimInstance Win32_Process -Filter \"Name='strata.exe'\" | "
                     "ForEach-Object { $_.ProcessId }").stdout
    return [int(x) for x in out.split() if x.strip().isdigit()]


def server_pids(pattern="serve"):
    out = powershell(
        "Get-CimInstance Win32_Process -Filter \"Name='python.exe'\" | Where-Object { "
        f"$_.CommandLine -like '*{pattern}*' }} | ForEach-Object {{ $_.ProcessId }}").stdout
    return [int(x) for x in out.split() if x.strip().isdigit()]


def kill(pids):
    for pid in pids:
        powershell(f"Stop-Process -Id {pid} -Force -ErrorAction SilentlyContinue")
    if pids:
        time.sleep(3)


def gpu_free_mib():
    out = subprocess.run(["nvidia-smi", "--query-gpu=memory.free", "--format=csv,noheader,nounits"],
                         capture_output=True, text=True, timeout=60).stdout
    try:
        return int(out.split()[0])
    except (IndexError, ValueError):
        return -1


def wait_ready(port, timeout=600):
    # /health answers without the API key; /v1/models is 401 until a key is given, so it cannot probe readiness
    deadline = time.time() + timeout
    while time.time() < deadline:
        try:
            with urllib.request.urlopen(f"http://127.0.0.1:{port}/health", timeout=15) as r:
                if r.status == 200:
                    return True
        except Exception:
            pass
        time.sleep(5)
    return False


def free_all():
    kill(server_pids())
    kill(strata_pids())
    time.sleep(5)
    # the vision encoder holds a little VRAM too
    out = powershell("Get-CimInstance Win32_Process -Filter \"Name='strata-vision.exe'\" | "
                     "ForEach-Object { $_.ProcessId }").stdout
    kill([int(x) for x in out.split() if x.strip().isdigit()])
    time.sleep(3)
    say(f"GPU free now: {gpu_free_mib()} MiB")


def run_arm(label, config, port):
    say(f"=== arm {label}: {config} ===")
    log = Z / f"ab-{label}-server.txt"
    with log.open("w", encoding="utf-8") as fh:
        proc = subprocess.Popen([PY, SERVER, "--engine", "strata", "--config", str(config),
                                 "--port", str(port)], cwd=str(ROOT), stdout=fh,
                                stderr=subprocess.STDOUT, creationflags=0)
    try:
        if not wait_ready(port):
            say(f"arm {label}: server never became ready (see {log})")
            return None
        say(f"arm {label}: ready, GPU free {gpu_free_mib()} MiB")
        out = subprocess.run([PY, str(Z / "fib_ab.py"), str(config), label], cwd=str(ROOT),
                             capture_output=True, text=True, timeout=3600)
        for line in out.stdout.splitlines():
            say("  " + line)
        if out.returncode != 0:
            say(f"arm {label}: runner exit {out.returncode}: {out.stderr.strip()[:300]}")
        return out.returncode
    finally:
        say(f"arm {label}: stopping")
        kill(server_pids())
        kill(strata_pids())
        time.sleep(5)


def main():
    LOG.write_text("", encoding="utf-8")
    arms = [(a.split("=")[0], Path(a.split("=", 1)[1])) for a in sys.argv[1:]]
    if not arms:
        arms = [("A", Z / "ab-A.json"), ("B", Z / "ab-B.json")]
    say(f"arms: {[a for a, _ in arms]}")
    say("arms run one at a time so only one engine holds the GPU")
    restored = False
    try:
        for label, config in arms:
            free_all()
            run_arm(label, config, 8084)
    finally:
        say("=== restoring the normal 8080 server ===")
        free_all()
        log = Z / "ab-restore-server.txt"
        with log.open("w", encoding="utf-8") as fh:
            subprocess.Popen([PY, SERVER, "--engine", "strata", "--config",
                              str(ROOT / "strata-gyro-gyro-m.json"), "--port", "8080"],
                             cwd=str(ROOT), stdout=fh, stderr=subprocess.STDOUT)
        if wait_ready(8080, timeout=900):
            say(f"8080 is back, GPU free {gpu_free_mib()} MiB")
            restored = True
        else:
            say(f"8080 did not come back; see {log}")
    say(f"done (8080 restored: {restored})")
    return 0


if __name__ == "__main__":
    sys.exit(main())
