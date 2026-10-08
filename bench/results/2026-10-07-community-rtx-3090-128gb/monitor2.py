"""Sample system RAM, the engine process's own memory (PSS), and GPU memory once per second.

monitor.py only took MemTotal/MemAvailable, which counts the engine's anonymous resident
set but ignores reclaimable page cache, so it understates the model's footprint. This adds
MemFree/Cached/Shmem and the engine process's smaps_rollup (Pss, Pss_Anon, Pss_File) so the
true resident footprint can be reported. The engine is the `engine/strata --serve` process,
not the Python server or the vision helper.
"""
import json
import subprocess
import sys
import time
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


with Path(sys.argv[1] if len(sys.argv) > 1 else "telemetry.jsonl").open("a") as output:
    while True:
        memory = {}
        for line in Path("/proc/meminfo").read_text().splitlines():
            key, value = line.split(":", 1)
            if key in MEM_KEYS:
                memory[key + "_KiB"] = int(value.strip().split()[0])
        pid = engine_pid()
        engine = {"pid": pid, "smaps": smaps_rollup(pid)} if pid else None
        gpu = subprocess.check_output(
            ["nvidia-smi", "--query-gpu=memory.used,memory.total,utilization.gpu,power.draw,temperature.gpu",
             "--format=csv,noheader,nounits"], text=True).strip()
        output.write(json.dumps({"epoch_s": time.time(), "memory": memory, "engine": engine, "gpu": gpu}) + "\n")
        output.flush()
        time.sleep(1)
