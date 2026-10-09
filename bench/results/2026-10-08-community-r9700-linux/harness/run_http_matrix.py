#!/usr/bin/env python3
"""Own and test private loopback servers sequentially; retain raw requests and logs."""
import argparse
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import signal
import socket
import subprocess
import sys
import time
import urllib.error

from bench_engine import save
from http_quality import Client
from single_gpu import ROOT


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--configs", type=Path, required=True)
    parser.add_argument("--quant", choices=["iq3_s", "iq2_xs"], required=True)
    parser.add_argument("--arms", nargs="+", default=["baseline", "wmma"])
    parser.add_argument("--fixtures", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--port", type=int, default=8097)
    args = parser.parse_args()
    args.out.mkdir(parents=True, exist_ok=False)
    matrix = {"scope": "private HTTP quality regression; cache=6 for multi-turn; not performance pairing",
              "runner_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
              "fixtures_sha256": hashlib.sha256(args.fixtures.read_bytes()).hexdigest(), "sessions": []}
    for arm in args.arms:
        cfg = json.loads((args.configs / f"{args.quant}-{arm}.json").read_text())
        cfg["args"][cfg["args"].index("--prompt-cache") + 1] = "6"
        cfg["model_name"] = f"r9700-quality-{args.quant}-{arm}"
        cfg["log"] = str((args.out / f"{arm}-engine.log").resolve())
        cfg_path = args.out / f"{arm}-config.json"
        save(cfg_path, cfg)
        # Refuse to reuse an existing endpoint. Every request must reach our own model name.
        with socket.socket() as check:
            check.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
            check.bind(("127.0.0.1", args.port))
        cmd = [sys.executable, str(ROOT / "tools/hip/r9700/single_gpu.py"),
               "--device-binary", str(Path(cfg["exe"]).with_name("strata-device")),
               "--config", str(cfg_path.resolve()), "--record", str((args.out / f"{arm}-launch.json").resolve()),
               "--cpu-nodes", "0,1", "--preferred-node", "1", "--port", str(args.port), "serve"]
        entry = {"arm": arm, "command": cmd, "status": "starting",
                 "binary_sha256": hashlib.sha256(Path(cfg["exe"]).read_bytes()).hexdigest(),
                 "config_sha256": hashlib.sha256(cfg_path.read_bytes()).hexdigest(),
                 "start_utc": datetime.now(timezone.utc).isoformat()}
        matrix["sessions"].append(entry)
        save(args.out / "matrix.json", matrix)
        print("start", arm, flush=True)
        with (args.out / f"{arm}-server.log").open("x") as log:
            process = subprocess.Popen(cmd, cwd=ROOT, stdout=log, stderr=subprocess.STDOUT, start_new_session=True)
            entry["server_pid"] = process.pid
            try:
                client = Client(f"http://127.0.0.1:{args.port}")
                deadline = time.monotonic() + 600
                while time.monotonic() < deadline:
                    if process.poll() is not None:
                        raise RuntimeError(f"server exited {process.returncode}")
                    try:
                        metrics = client.json("/metrics")
                        if metrics["engine"]["model"] != cfg["model_name"]:
                            raise RuntimeError("unexpected server model identity")
                        if metrics["live"]["state"] == "idle":
                            break
                    except (OSError, urllib.error.URLError):
                        pass
                    time.sleep(.5)
                else:
                    raise RuntimeError("server startup timeout")
                entry["status"] = "testing"
                save(args.out / "matrix.json", matrix)
                test = [sys.executable, str(ROOT / "tools/hip/r9700/http_quality.py"),
                        "--url", client.url, "--fixtures", str(args.fixtures.resolve()),
                        "--out", str((args.out / arm).resolve())]
                with (args.out / f"{arm}-test.log").open("x") as test_log:
                    result = subprocess.run(test, cwd=ROOT, stdout=test_log, stderr=subprocess.STDOUT)
                entry.update(status="passed" if result.returncode == 0 else "failed", returncode=result.returncode,
                             test_command=test)
            finally:
                # Only the process group created above belongs to this test.
                if process.poll() is None:
                    os.killpg(process.pid, signal.SIGINT)
                    try:
                        process.wait(timeout=15)
                    except subprocess.TimeoutExpired:
                        os.killpg(process.pid, signal.SIGKILL)
                        process.wait()
                entry["server_exit"] = process.returncode
                entry["end_utc"] = datetime.now(timezone.utc).isoformat()
                save(args.out / "matrix.json", matrix)
        print(arm, entry["status"], flush=True)
    matrix["complete"] = True
    save(args.out / "matrix.json", matrix)


if __name__ == "__main__":
    main()
