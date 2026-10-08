#!/usr/bin/env python3
"""Sample RAM, engine PSS, GPU state, and Strata /v1/status once per second."""
import json
import subprocess
import sys
import time
import urllib.request
from pathlib import Path

MEM_KEYS = ("MemTotal", "MemFree", "MemAvailable", "Cached", "Shmem", "SwapTotal", "SwapFree")
SMAP_KEYS = ("Rss", "Pss", "Pss_Anon", "Pss_File", "Anonymous", "Swap")


def engine_pid():
    try:
        out = subprocess.check_output(["pgrep", "-f", "engine/strata --serve"], text=True)
    except subprocess.CalledProcessError:
        return None
    pids = [int(x) for x in out.split()]
    return pids[0] if pids else None


def smaps_rollup(pid):
    try:
        text = Path(f"/proc/{pid}/smaps_rollup").read_text()
    except OSError:
        return None
    vals = {}
    for line in text.splitlines():
        key, _, value = line.partition(":")
        if key in SMAP_KEYS:
            vals[key + "_KiB"] = int(value.strip().split()[0])
    return vals or None


def get_json(url):
    try:
        with urllib.request.urlopen(url, timeout=5) as response:
            return json.load(response)
    except Exception as exc:
        return {"error": str(exc)}


def gpu():
    try:
        out = subprocess.check_output(
            ["nvidia-smi", "--query-gpu=memory.used,memory.total,utilization.gpu,power.draw,temperature.gpu",
             "--format=csv,noheader,nounits"],
            text=True,
        ).strip()
        used, total, util, power, temp = [x.strip() for x in out.split(",")]
        return {
            "used_mib": float(used),
            "total_mib": float(total),
            "util_pct": float(util),
            "power_w": float(power) if power != "N/A" else None,
            "temp_c": float(temp),
        }
    except Exception as exc:
        return {"error": str(exc)}


def main():
    out_path = Path(sys.argv[1] if len(sys.argv) > 1 else "telemetry.jsonl")
    url = (sys.argv[2] if len(sys.argv) > 2 else "http://127.0.0.1:8080").rstrip("/")
    with out_path.open("a") as output:
        while True:
            memory = {}
            for line in Path("/proc/meminfo").read_text().splitlines():
                key, value = line.split(":", 1)
                if key in MEM_KEYS:
                    memory[key + "_KiB"] = int(value.strip().split()[0])
            pid = engine_pid()
            status = get_json(url + "/v1/status")
            row = {
                "epoch_s": time.time(),
                "memory": memory,
                "engine": {"pid": pid, "smaps": smaps_rollup(pid)} if pid else None,
                "gpu": gpu(),
                "status": {
                    "loaded": status.get("loaded"),
                    "activity": status.get("activity"),
                    "vram": status.get("vram"),
                    "machine": status.get("machine"),
                },
            }
            output.write(json.dumps(row) + "\n")
            output.flush()
            time.sleep(1)


if __name__ == "__main__":
    main()
