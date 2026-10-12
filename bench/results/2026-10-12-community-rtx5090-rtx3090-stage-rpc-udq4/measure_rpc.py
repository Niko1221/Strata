#!/usr/bin/env python3
"""Run the upstream suite against the existing server and sample both RPC nodes.

No server lifecycle or configuration operations. Artifacts are local/private;
upstream's sanitized report is copied separately into remote-results/.
"""
import argparse
import os
import datetime
import json
import shlex
import subprocess
import threading
import time
import urllib.request
from pathlib import Path

URL = os.environ.get("STRATA_API_URL", "http://127.0.0.1:8090")
REMOTE = os.environ.get("STRATA_REMOTE", "")
REPO = os.environ.get("STRATA_REPO", "")


def fetch(path):
    with urllib.request.urlopen(URL + path, timeout=10) as response:
        return json.load(response)


def run(command):
    result = subprocess.run(command, text=True, capture_output=True, timeout=30)
    return {"exit_code": result.returncode, "stdout": result.stdout, "stderr": result.stderr}


def facts(remote=False):
    service = "strata-rpc-coordinator-5090" if remote else "strata-rpc-worker-3090"
    config = REPO + "/strata-rpc.json" if remote else os.environ.get("STRATA_WORKER_CONFIG", str(Path.home() / ".local/state/strata-stage-rpc/worker.json"))
    commands = {
        "service": ["systemctl", "--user", "show", service + ".service", "--property=MainPID,InvocationID,ActiveState,MemoryCurrent,MemoryPeak"],
        "config_sha256": ["sha256sum", config],
        "gpu": ["nvidia-smi", "--query-gpu=name,memory.total,memory.used,memory.free,pci.bus_id,pcie.link.gen.current,pcie.link.width.current,power.limit,driver_version", "--format=csv,noheader,nounits"],
        "source": ["git", "-C", REPO, "rev-parse", "HEAD"],
        "dirty": ["git", "-C", REPO, "status", "--short"],
        "binary_sha256": ["sha256sum", REPO + "/build-rpc-kquant/strata"],
    }
    return {key: run(["ssh", "-o", "BatchMode=yes", "-o", "ConnectTimeout=8", REMOTE, shlex.join(command)] if remote else command)
            for key, command in commands.items()}


def write_json(path, value):
    path.write_text(json.dumps(value, indent=2) + "\n")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    if not REMOTE or not REPO:
        parser.error("Set STRATA_REMOTE and STRATA_REPO before measuring")
    out = args.out.resolve()
    out.mkdir(parents=True, exist_ok=True)
    health, initial = fetch("/health"), fetch("/metrics")
    if not health.get("loaded") or initial["live"]["state"] != "idle" or initial["live"].get("queued"):
        raise SystemExit("Refusing to benchmark a server that is not loaded and idle")
    before = {"health": health, "local": facts(), "remote": facts(True), "totals": initial.get("totals")}
    write_json(out / "before.json", before)
    remote_out = str(out / "upstream")
    benchmark = ["python3", "tools/strata_bench.py", "--config", "strata-rpc.json", "--url", os.environ.get("STRATA_REMOTE_API_URL", "http://127.0.0.1:8090"),
                 "--runs", "3", "--long", "--out", remote_out]
    remote_command = "cd " + shlex.quote(REPO) + " && " + shlex.join(benchmark)
    metadata = {"started_utc": datetime.datetime.now(datetime.timezone.utc).isoformat(),
                "managed_run": os.environ.get("STRATA_MANAGED_RUN"), "profile": os.environ.get("STRATA_PROFILE"),
                "benchmark_command": remote_command, "sample_interval_s": 2,
                "reasoning": "off for benchmark requests only", "temperature": 0,
                "limitations": ["Warm existing engine; no fresh-process or cold expert-cache measurement",
                                "Direct native chat API, not Codex/Responses or the custom proxy",
                                "Agent fixture measures prefix reuse and does not execute tools",
                                "Desktop remains active; GPU totals include desktop allocations",
                                "Telemetry and nvidia-smi introduce small measurement overhead",
                                "Existing configured context is 192K; suite prompts stop at about 32K"]}
    write_json(out / "run.json", metadata)
    process = subprocess.Popen(["ssh", "-o", "BatchMode=yes", "-o", "ConnectTimeout=8", REMOTE, remote_command],
                               stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, bufsize=1)
    print(f"BENCHMARK_STARTED remote_ssh_pid={process.pid} artifacts={out}", flush=True)

    def copy_output():
        with (out / "benchmark.log").open("w") as log:
            for line in process.stdout:
                log.write(line)
                log.flush()
                print(line, end="", flush=True)

    reader = threading.Thread(target=copy_output)
    reader.start()
    samples, errors = [], []
    with (out / "telemetry.jsonl").open("w") as telemetry:
        while True:
            try:
                metrics = fetch("/metrics")
                gpu = run(["nvidia-smi", "--query-compute-apps=pid,used_gpu_memory", "--format=csv,noheader,nounits"])
                sample = {"time": time.time(), "live": metrics.get("live"), "totals": metrics.get("totals"),
                          "worker_process_vram": gpu,
                          "nodes": [{k: node.get(k) for k in ("id", "role", "layers", "hardware_static", "hardware", "freshness", "engine_cache")}
                                    for node in metrics.get("nodes", [])]}
                samples.append(sample)
                telemetry.write(json.dumps(sample) + "\n")
                telemetry.flush()
            except Exception as error:
                errors.append({"time": time.time(), "error": str(error)})
            if process.poll() is not None:
                break
            time.sleep(2)
    reader.join()
    after = {"health": fetch("/health"), "local": facts(), "remote": facts(True), "totals": fetch("/metrics").get("totals")}
    write_json(out / "after.json", after)
    peaks = {}
    for sample in samples:
        for node in sample["nodes"]:
            reading = node.get("hardware") or {}
            peak = peaks.setdefault(node["id"], {})
            for key in ("gpu_mem_used", "gpu_util", "gpu_power", "ram_used", "net_rx_mib_s", "net_tx_mib_s"):
                value = reading.get(key)
                if isinstance(value, (int, float)):
                    peak[key] = max(value, peak.get(key, value))
    unchanged = {}
    for node in ("local", "remote"):
        unchanged[node + "_config"] = before[node]["config_sha256"] == after[node]["config_sha256"]
        for property_name in ("MainPID", "InvocationID", "ActiveState"):
            def prop(fact):
                return next((line for line in fact[node]["service"]["stdout"].splitlines() if line.startswith(property_name + "=")), None)
            unchanged[node + "_" + property_name] = prop(before) == prop(after) and prop(before) is not None
    destination = out / "remote-results"
    destination.mkdir(exist_ok=True)
    copied = run(["rsync", "-a", REMOTE + ":" + remote_out + "/", str(destination) + "/"])
    summary = {"benchmark_exit_code": process.returncode, "samples": len(samples), "telemetry_errors": errors,
               "observed_peaks": peaks, "unchanged": unchanged, "report_copy": copied,
               "completed_utc": datetime.datetime.now(datetime.timezone.utc).isoformat()}
    write_json(out / "telemetry-summary.json", summary)
    print("BENCHMARK_FINISHED " + json.dumps(summary), flush=True)
    return process.returncode or (1 if copied["exit_code"] or not all(unchanged.values()) else 0)


if __name__ == "__main__":
    raise SystemExit(main())
