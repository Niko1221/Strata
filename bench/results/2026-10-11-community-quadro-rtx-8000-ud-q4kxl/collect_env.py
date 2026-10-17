#!/usr/bin/env python3
"""Collect hardware, driver, build and run-configuration provenance into environment.json.

Reads only: nvidia-smi, /proc, /etc/os-release, the Strata checkout and the server's
/metrics and /props. It does not start, stop or reconfigure the engine.

    python3 collect_env.py            # writes environment.json next to this script
"""
import json
import os
import platform
import re
import socket
import subprocess
import sys
import urllib.request

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.abspath(os.path.join(HERE, "..", "..", ".."))
URL = os.environ.get("STRATA_URL", "http://127.0.0.1:8080")


def sh(cmd):
    try:
        return subprocess.run(cmd, shell=True, capture_output=True, text=True, timeout=20).stdout.strip()
    except Exception as e:
        return f"unavailable ({e})"


def get(path):
    try:
        with urllib.request.urlopen(URL + path, timeout=10) as f:
            return json.loads(f.read())
    except Exception as e:
        return {"_error": str(e)}


def gpu():
    q = "name,index,memory.total,memory.used,driver_version,power.limit,power.draw," \
        "pcie.link.gen.current,pcie.link.gen.max,pcie.link.width.current,pcie.link.width.max," \
        "temperature.gpu,clocks.sm,clocks.max.sm,vbios_version"
    out = sh(f"nvidia-smi --query-gpu={q} --format=csv,noheader")
    cards = []
    for line in out.splitlines():
        f = [x.strip() for x in line.split(",")]
        if len(f) < 15:
            continue
        cards.append({
            "name": f[0], "index": f[1], "memory_total_mib": f[2], "memory_used_mib": f[3],
            "driver_version": f[4], "power_limit_w": f[5], "power_draw_w": f[6],
            "pcie_gen_current": f[7], "pcie_gen_max": f[8],
            "pcie_width_current": f[9], "pcie_width_max": f[10],
            "temperature_c": f[11], "clock_sm_mhz": f[12], "clock_sm_max_mhz": f[13],
            "vbios": f[14],
        })
    return cards


def cpuinfo():
    txt = open("/proc/cpuinfo").read()
    flags = set()
    for line in txt.splitlines():
        if line.startswith("flags"):
            flags = set(line.split(":", 1)[1].split())
            break
    return {
        "model": sh("grep -m1 'model name' /proc/cpuinfo | cut -d: -f2- | sed 's/^ //'"),
        "logical_processors": os.cpu_count(),
        "physical_cores": sh("lscpu --parse=CORE 2>/dev/null | grep -v '^#' | sort -u | wc -l") or "not measured",
        "avx2": "avx2" in flags,
        "avx512f": "avx512f" in flags,
        "avx512bw": "avx512bw" in flags,
        "amx_tile": "amx_tile" in flags,
        "governor": sh("cat /sys/devices/system/cpu/cpu0/cpufreq/scaling_governor 2>/dev/null") or "not measured",
        "march": sh("gcc -march=native -Q --help=target 2>/dev/null | grep -m1 march") or "not measured",
    }


def memory():
    mi = dict(l.split(":", 1) for l in open("/proc/meminfo").read().splitlines() if ":" in l)
    gib = lambda k: round(int(mi[k].split()[0]) / 1048576, 2)
    speeds = sh("dmidecode -t memory 2>/dev/null | grep -m2 'Speed:'")
    return {
        "mem_total_gib": gib("MemTotal"),
        "mem_available_gib": gib("MemAvailable"),
        "swap_total_gib": gib("SwapTotal"),
        "hugepages_total": mi.get("HugePages_Total", "0").split()[0],
        "rated_ram_speed": speeds if speeds else "not measured (needs root)",
    }


def storage(path):
    dev = sh(f"df {path} --output=source | tail -1").replace("/dev/", "")
    rot = sh(f"cat /sys/block/{dev.split('/')[0]}/queue/rotational 2>/dev/null")
    model = sh(f"cat /sys/block/{dev.split('/')[0]}/device/model 2>/dev/null")
    return {"device": dev, "rotational": rot or "not measured", "model": model or "not measured"}


def git(*args):
    return sh(f"git -C {REPO} " + " ".join(args))


def main():
    metrics = get("/metrics")
    props = get("/props")
    cfg_path = os.environ.get("STRATA_CONFIG", "")
    if not cfg_path:
        for f in os.listdir(REPO):
            if f.startswith("strata-") and f.endswith(".json") and "expert" not in f:
                cfg_path = os.path.join(REPO, f)
        cfg_path = cfg_path or ""
    cfg = json.load(open(cfg_path)) if cfg_path and os.path.exists(cfg_path) else {}
    build = {}
    for candidate in ("engine/BUILD.json",):
        p = os.path.join(REPO, candidate)
        if os.path.exists(p):
            build = json.load(open(p))

    env = {
        "collected_at": time_now(),
        "hostname": socket.gethostname(),
        "os": dict(l.split("=", 1) for l in open("/etc/os-release") if "=" in l)
               .get("PRETTY_NAME", "").strip().strip('"'),
        "kernel": platform.release(),
        "nvcc_version": sh("nvcc --version 2>/dev/null | grep -o 'release [0-9.]*'") or "not measured",
        "cuda_build_dir": ",".join(json.load(open(os.path.join(REPO, "engine/BUILD.json")))
                                   .get("cuda_dirs", [])) if os.path.exists(
            os.path.join(REPO, "engine/BUILD.json")) else "not measured",
        "cuda_driver_reported": sh("nvidia-smi --query-gpu=driver_version --format=csv,noheader | head -1"),
        "rocm": "n/a (CUDA build)",
        "gpus": gpu(),
        "cpu": cpuinfo(),
        "memory": memory(),
        "storage_model_dir": storage(os.path.dirname(cfg.get("native", "/opt/models")) or "/"),
        "storage_repo_dir": storage(REPO),
        "power_profile": sh("powerprofilesctl get 2>/dev/null") or "not measured",
        "background_load": {
            "loadavg_1_5_15": sh("cat /proc/loadavg | cut -d' ' -f1-3"),
            "other_gpu_processes": sh(
                "nvidia-smi --query-compute-apps=pid,process_name,used_memory --format=csv,noheader")
            or "none",
        },
        "strata": {
            "repo": REPO,
            "commit": git("rev-parse HEAD"),
            "commit_short": git("rev-parse --short HEAD"),
            "describe": git("describe --tags"),
            "dirty": bool(git("status --porcelain")),
            "engine_version": (metrics.get("engine") or {}).get("version", "not measured"),
            "build_provenance": build,
        },
        "engine_config": {
            "config_file": os.path.basename(cfg_path),
            "args": cfg.get("args"),
            "model_name": cfg.get("model_name"),
            "port": cfg.get("port"),
            "gpu": cfg.get("gpu"),
            "vision": bool(cfg.get("vision")),
        },
        "engine_reported": metrics.get("engine"),
        "hardware_static": metrics.get("hardware_static"),
        "props": {k: props.get(k) for k in ("n_ctx", "total_slots", "engine_version", "model_path") if k in props},
        "not_measured": [
            "PCIe bandwidth under load (the engine's own startup probe is quoted instead, from the log)",
            "RAM rated speed (dmidecode needs root)",
        ],
    }

    out = os.path.join(HERE, "environment.json")
    json.dump(env, open(out, "w"), indent=1)
    print(json.dumps({k: env[k] for k in ("os", "kernel", "nvcc_version", "cuda_build_dir", "gpus",
                                          "cpu", "memory")}, indent=1))
    print(f"wrote {out}")
    return 0


def time_now():
    import time
    return time.strftime("%Y-%m-%dT%H:%M:%S%z")


if __name__ == "__main__":
    sys.exit(main())
