"""Sample GPU, the llama.cpp server's /metrics and /props, and the container's RAM once per second.

The model lives in VRAM (llama.cpp -ngl 99), so unlike the Strata monitor the RAM
figure is the container's, not an engine PSS. GPU comes from nvidia-smi; throughput
counters from the server's Prometheus /metrics; context/slots from /props.
"""
import json
import subprocess
import sys
import time
import urllib.request
from pathlib import Path

CONTAINER = sys.argv[2] if len(sys.argv) > 2 else "llama-cpp-qwen38-27b-single"
URL = (sys.argv[3] if len(sys.argv) > 3 else "http://127.0.0.1:8090").rstrip("/")
METRIC_KEYS = ("llamacpp:tokens_evaluated_total", "llamacpp:predictions_total",
               "llamacpp:requests_processing", "llamacpp:requests_deferred")


def metrics():
    try:
        text = urllib.request.urlopen(URL + "/metrics", timeout=5).read().decode()
    except Exception:
        return None
    out = {}
    for line in text.splitlines():
        if not line or line.startswith("#"):
            continue
        name, _, val = line.partition(" ")
        if name.split("{")[0] in METRIC_KEYS:
            try:
                out[name.split("{")[0]] = float(val)
            except ValueError:
                pass
    return out


def props():
    try:
        p = json.loads(urllib.request.urlopen(URL + "/props", timeout=5).read())
        return {"n_ctx": (p.get("default_generation_settings") or {}).get("n_ctx"),
                "total_slots": p.get("total_slots")}
    except Exception:
        return None


def docker_mem():
    try:
        out = subprocess.check_output(["docker", "stats", "--no-stream", "--format",
                                       "{{.MemUsage}} {{.MemPerc}}", CONTAINER], text=True).strip()
    except Exception:
        return None
    return out or None


with Path(sys.argv[1] if len(sys.argv) > 1 else "telemetry.jsonl").open("a") as output:
    while True:
        gpu = subprocess.check_output(
            ["nvidia-smi", "--query-gpu=memory.used,memory.total,utilization.gpu,power.draw,temperature.gpu",
             "--format=csv,noheader,nounits"], text=True).strip()
        rec = {"epoch_s": time.time(), "gpu": gpu, "metrics": metrics(), "props": props(), "container_mem": docker_mem()}
        output.write(json.dumps(rec) + "\n")
        output.flush()
        time.sleep(1)
