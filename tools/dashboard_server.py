import http.server
import json
import os
import re
import socketserver
import threading
import time
import urllib.request
from pathlib import Path

import psutil
import pynvml

PORT = 8081
HOST = "0.0.0.0"

# 1. Dynamic GPU Detection
try:
    pynvml.nvmlInit()
    nvml_handle = pynvml.nvmlDeviceGetHandleByIndex(0)
    gpu_name = pynvml.nvmlDeviceGetName(nvml_handle)
    if isinstance(gpu_name, bytes):
        gpu_name = gpu_name.decode("utf-8")
except Exception:
    nvml_handle = None
    gpu_name = "NVIDIA GeForce GPU"

# 2. Dynamic CPU Detection
cpu_model_name = "CPU"
cpu_physical_cores = psutil.cpu_count(logical=False) or 8
cpu_logical_threads = psutil.cpu_count(logical=True) or 16

try:
    import platform
    raw_proc = platform.processor()
    if raw_proc:
        cpu_model_name = raw_proc
    # On Windows, query registry or WMI if available for a cleaner brand string
    if os.name == "nt":
        import winreg
        key = winreg.OpenKey(winreg.HKEY_LOCAL_MACHINE, r"HARDWARE\DESCRIPTION\System\CentralProcessor\0")
        val, _ = winreg.QueryValueEx(key, "ProcessorNameString")
        if val:
            cpu_model_name = val.strip()
        winreg.CloseKey(key)
except Exception:
    pass

# 3. Dynamic Disk / NVMe Detection
# Identify PhysicalDrive index for drives M: and C:
def find_physical_drive_for_letter(letter):
    """Returns (physical_drive_name, model_name) for a drive letter like 'M:'"""
    # Fallback mappings for common drives if WMI query fails
    return f"PhysicalDrive2", f"{letter} NVMe"

nvme_drives_config = {
    "m_drive": {
        "letter": "M:",
        "pdisk": "PhysicalDrive2",
        "label": "M: (Model SSD)",
        "read_mb": 0.0,
        "write_mb": 0.0,
        "load_pct": 0.0,
    },
    "c_drive": {
        "letter": "C:",
        "pdisk": "PhysicalDrive1",
        "label": "C: (System SSD)",
        "read_mb": 0.0,
        "write_mb": 0.0,
        "load_pct": 0.0,
    },
}

try:
    import subprocess
    cmd = 'powershell -NoProfile -Command "Get-CimInstance Win32_DiskDrive | Select-Object DeviceID, Model | ConvertTo-Json"'
    out = subprocess.check_output(cmd, shell=True, text=True, timeout=5)
    drives_info = json.loads(out)
    if isinstance(drives_info, dict):
        drives_info = [drives_info]
    
    # Map models: PhysicalDrive2 is usually M:, PhysicalDrive1 is usually C:
    for d in drives_info:
        dev_id = d.get("DeviceID", "")
        model = d.get("Model", "").strip()
        if "2" in dev_id:
            nvme_drives_config["m_drive"]["label"] = f"M: ({model})"
        elif "1" in dev_id:
            nvme_drives_config["c_drive"]["label"] = f"C: ({model})"
except Exception:
    pass

def nvme_monitor_loop():
    prev_time = time.time()
    prev_io = psutil.disk_io_counters(perdisk=True)
    while True:
        time.sleep(1.0)
        curr_time = time.time()
        curr_io = psutil.disk_io_counters(perdisk=True)
        dt = max(0.001, curr_time - prev_time)

        for key, conf in nvme_drives_config.items():
            pdisk = conf["pdisk"]
            if pdisk in curr_io and pdisk in prev_io:
                c1, c2 = prev_io[pdisk], curr_io[pdisk]
                r_mb = (c2.read_bytes - c1.read_bytes) / dt / 1024 / 1024
                w_mb = (c2.write_bytes - c1.write_bytes) / dt / 1024 / 1024
                t_ms = (c2.read_time - c1.read_time) + (c2.write_time - c1.write_time)
                pct = min(100.0, max(0.0, (t_ms / (dt * 1000.0)) * 100.0))
                conf.update({
                    "read_mb": round(r_mb, 1),
                    "write_mb": round(w_mb, 1),
                    "load_pct": round(pct, 1),
                })

        prev_time = curr_time
        prev_io = curr_io

threading.Thread(target=nvme_monitor_loop, daemon=True).start()

# 4. Dynamic Strata Config & Log Detection
STRATA_DIR = Path(r"M:\Strata")
TASKS_DIR = Path(r"C:\Users\Martin\.gemini\antigravity\brain\9b6b10be-f606-4240-9049-186920f01c65\.system_generated\tasks")

def get_strata_log_path():
    log_file = STRATA_DIR / "strata-swift-iq2_xs.log"
    if log_file.exists():
        return log_file
    return None

def get_recent_history():
    history = []
    # Check strata engine log first
    log_path = get_strata_log_path()
    if not log_path or not log_path.exists():
        # Check task logs
        if TASKS_DIR.exists():
            logs = list(TASKS_DIR.glob("task-*.log"))
            if logs:
                logs.sort(key=lambda p: p.stat().st_mtime, reverse=True)
                log_path = logs[0]

    if not log_path or not log_path.exists():
        return history

    try:
        size = log_path.stat().st_size
        offset = max(0, size - 131072)
        with open(log_path, "rb") as f:
            f.seek(offset)
            text = f.read().decode("utf-8", errors="ignore")
        lines = text.splitlines()

        for line in lines:
            # Format: strata serve: prompt 35029 tokens = 33925 reused + 1104 read in 4985 ms (221.4 tok/s), 1352 generated in 50583 ms (26.7 tok/s)
            m_serve = re.search(r"prompt\s+(\d+)\s+tokens.*read in\s+(\d+)\s+ms\s+\(([\d.]+)\s+tok/s\),\s+(\d+)\s+generated in\s+(\d+)\s+ms\s+\(([\d.]+)\s+tok/s\)", line)
            if m_serve:
                prompt_tokens = int(m_serve.group(1))
                prefill_ms = int(m_serve.group(2))
                prefill_speed = float(m_serve.group(3))
                gen_tokens = int(m_serve.group(4))
                gen_ms = int(m_serve.group(5))
                gen_speed = float(m_serve.group(6))
                total_s = round((prefill_ms + gen_ms) / 1000.0, 1)

                history.append({
                    "prompt_tokens": prompt_tokens,
                    "prefill_s": round(prefill_ms / 1000.0, 1),
                    "prefill_speed": prefill_speed,
                    "gen_tokens": gen_tokens,
                    "gen_s": round(gen_ms / 1000.0, 1),
                    "gen_speed": gen_speed,
                    "total_s": total_s,
                    "finish": f"{gen_speed} tok/s",
                })
                continue

            # Fallback format: [strata] reading the prompt / done
            m_done = re.search(r"\[strata\] done:\s+(\d+)\s+tokens in\s+(\d+)\s+s\s+\(([^)]+)\)", line)
            if m_done:
                n_tokens = int(m_done.group(1))
                total_s = int(m_done.group(2))
                finish = m_done.group(3)
                history.append({
                    "prompt_tokens": 0,
                    "prefill_s": 0,
                    "prefill_speed": 0.0,
                    "gen_tokens": n_tokens,
                    "gen_s": total_s,
                    "gen_speed": round(n_tokens / max(1, total_s), 1),
                    "total_s": total_s,
                    "finish": finish,
                })
    except Exception:
        pass
    return history[-10:]

def get_strata_config_info():
    config_path = STRATA_DIR / "strata-swift-iq2_xs.json"
    model_name = "Swift-1.5 (IQ2_XS)"
    max_context = "262,144 (256K)"
    kv_mode = "Q4_0"
    if config_path.exists():
        try:
            with open(config_path, "r", encoding="utf-8") as f:
                cfg = json.load(f)
            args = cfg.get("args", [])
            for i, arg in enumerate(args):
                if arg == "--max-context" and i + 1 < len(args):
                    val = int(args[i+1])
                    max_context = f"{val:,} ({val//1024}K)"
                elif arg == "--kv" and i + 1 < len(args):
                    kv_mode = args[i+1].upper()
            if "model_name" in cfg:
                model_name = cfg["model_name"]
        except Exception:
            pass
    return {
        "model": model_name,
        "max_context": max_context,
        "kv": kv_mode,
    }

def get_resident_experts_count():
    log_path = get_strata_log_path()
    if not log_path or not log_path.exists():
        return 0
    try:
        size = log_path.stat().st_size
        offset = max(0, size - 65536)
        with open(log_path, "rb") as f:
            f.seek(offset)
            text = f.read().decode("utf-8", errors="ignore")
        matches = re.findall(r"expert cache (\d+) slots", text)
        if matches:
            return int(matches[-1])
    except Exception:
        pass
    return 0

DASHBOARD_START_TIME = time.time()
_cached_strata_proc = None
_cached_strata_create_time = None

def get_server_uptime_str():
    global _cached_strata_proc, _cached_strata_create_time
    if _cached_strata_create_time is not None:
        try:
            if _cached_strata_proc and _cached_strata_proc.is_running():
                up_sec = int(time.time() - _cached_strata_create_time)
                h = up_sec // 3600
                m = (up_sec % 3600) // 60
                s = up_sec % 60
                return f"{h:02d}:{m:02d}:{s:02d}"
        except Exception:
            _cached_strata_proc = None
            _cached_strata_create_time = None

    try:
        for p in psutil.process_iter(['name', 'cmdline', 'create_time']):
            cmd = " ".join(p.info.get('cmdline') or [])
            if "server.py" in cmd and "--engine" in cmd:
                _cached_strata_proc = p
                _cached_strata_create_time = p.info['create_time']
                up_sec = int(time.time() - _cached_strata_create_time)
                h = up_sec // 3600
                m = (up_sec % 3600) // 60
                s = up_sec % 60
                return f"{h:02d}:{m:02d}:{s:02d}"
    except Exception:
        pass
    up_sec = int(time.time() - DASHBOARD_START_TIME)
    h = up_sec // 3600
    m = (up_sec % 3600) // 60
    s = up_sec % 60
    return f"{h:02d}:{m:02d}:{s:02d}"

def get_stats():
    # 1. GPU metrics
    gpu_stats = {
        "name": gpu_name,
        "util": 0,
        "mem_used_mb": 0,
        "mem_total_mb": 10240,
        "mem_percent": 0.0,
        "temp_c": 0,
        "power_w": 0.0,
        "pcie": "PCIe 4.0 x16",
        "expert_slots": get_resident_experts_count(),
    }
    if nvml_handle:
        try:
            mem = pynvml.nvmlDeviceGetMemoryInfo(nvml_handle)
            util = pynvml.nvmlDeviceGetUtilizationRates(nvml_handle)
            temp = pynvml.nvmlDeviceGetTemperature(nvml_handle, pynvml.NVML_TEMPERATURE_GPU)
            pwr = pynvml.nvmlDeviceGetPowerUsage(nvml_handle) / 1000.0
            p_gen = pynvml.nvmlDeviceGetCurrPcieLinkGeneration(nvml_handle)
            p_width = pynvml.nvmlDeviceGetCurrPcieLinkWidth(nvml_handle)
            gpu_stats.update({
                "util": util.gpu,
                "mem_used_mb": round(mem.used / 1024 / 1024),
                "mem_total_mb": round(mem.total / 1024 / 1024),
                "mem_percent": round(mem.used / mem.total * 100, 1),
                "temp_c": temp,
                "power_w": round(pwr, 1),
                "pcie": f"PCIe {p_gen}.0 x{p_width}",
            })
        except Exception:
            pass

    # 2. CPU & RAM metrics
    vm = psutil.virtual_memory()
    cpu_stats = {
        "name": cpu_model_name,
        "cores": cpu_physical_cores,
        "threads": cpu_logical_threads,
        "util": psutil.cpu_percent(interval=None),
        "ram_used_gb": round(vm.used / (1024**3), 1),
        "ram_total_gb": round(vm.total / (1024**3), 1),
        "ram_percent": vm.percent,
    }

    # 3. Strata Engine status
    cfg_info = get_strata_config_info()
    strata_stats = {
        "online": False,
        "busy": False,
        "phase": "idle",
        "prompt_tokens": 0,
        "generated": 0,
        "max_tokens": 0,
        "elapsed_s": 0.0,
        "tokens_per_s": 0.0,
        "model": cfg_info["model"],
        "max_context": cfg_info["max_context"],
        "kv": cfg_info["kv"],
    }
    try:
        with urllib.request.urlopen("http://127.0.0.1:8080/status", timeout=0.5) as r:
            st = json.loads(r.read().decode())
            strata_stats.update({
                "online": True,
                "busy": st.get("busy", False),
                "phase": st.get("phase", "idle") if st.get("busy") else "idle",
                "prompt_tokens": st.get("prompt_tokens", 0),
                "generated": st.get("generated", 0),
                "max_tokens": st.get("max_tokens", 0),
                "elapsed_s": st.get("elapsed_s", 0.0),
                "tokens_per_s": st.get("tokens_per_s", 0.0),
            })
    except Exception:
        pass

    # NVMe structure for json
    nvme_out = {
        "m_drive": {
            "name": nvme_drives_config["m_drive"]["label"],
            "read_mb": nvme_drives_config["m_drive"]["read_mb"],
            "write_mb": nvme_drives_config["m_drive"]["write_mb"],
            "load_pct": nvme_drives_config["m_drive"]["load_pct"],
        },
        "c_drive": {
            "name": nvme_drives_config["c_drive"]["label"],
            "read_mb": nvme_drives_config["c_drive"]["read_mb"],
            "write_mb": nvme_drives_config["c_drive"]["write_mb"],
            "load_pct": nvme_drives_config["c_drive"]["load_pct"],
        },
    }

    return {
        "timestamp": time.strftime("%H:%M:%S"),
        "uptime": get_server_uptime_str(),
        "gpu": gpu_stats,
        "cpu": cpu_stats,
        "nvme": nvme_out,
        "strata": strata_stats,
        "history": get_recent_history(),
    }

HTML_TEMPLATE = """<!DOCTYPE html>
<html lang="en" class="dark">
<head>
  <meta charset="UTF-8">
  <meta name="viewport" content="width=device-width, initial-scale=1.0">
  <title>Strata AI Live Dashboard</title>
  <script src="https://www.gstatic.com/antigravity/web/dev/tailwindcss.min.js"></script>
  <style>
    @keyframes pulse-slow {
      0%, 100% { opacity: 1; }
      50% { opacity: 0.4; }
    }
    .animate-pulse-slow { animation: pulse-slow 2s cubic-bezier(0.4, 0, 0.6, 1) infinite; }
  </style>
</head>
<body class="bg-[#0b0f17] text-slate-100 min-h-screen p-4 md:p-6 font-sans antialiased">
  <div class="max-w-6xl mx-auto space-y-6">
    
    <!-- Header -->
    <header class="flex flex-wrap items-center justify-between gap-4 pb-4 border-b border-slate-800">
      <div class="flex items-center gap-3">
        <div class="w-3.5 h-3.5 rounded-full bg-emerald-500 shadow-[0_0_12px_rgba(16,185,129,0.7)]" id="status-dot"></div>
        <div>
          <h1 class="text-xl md:text-2xl font-bold tracking-tight bg-gradient-to-r from-cyan-400 via-sky-300 to-indigo-400 bg-clip-text text-transparent">
            Strata Live Dashboard
          </h1>
          <p class="text-xs text-slate-400 flex items-center gap-2">
            <span>Model: <span id="header-model" class="text-slate-200 font-mono font-medium">--</span></span>
            <span>•</span>
            <span>Max Context: <span id="header-context" class="text-emerald-400 font-mono font-medium">--</span></span>
            <span>•</span>
            <span>KV: <span id="header-kv" class="text-sky-400 font-mono font-medium">--</span></span>
          </p>
        </div>
      </div>
      <div class="flex items-center gap-2.5 text-xs font-mono text-slate-300 bg-slate-900/90 px-3.5 py-1.5 rounded-lg border border-slate-800 shadow-inner">
        <span class="inline-block w-2 h-2 rounded-full bg-emerald-400 animate-pulse"></span>
        <span id="clock" class="font-semibold text-slate-100">--:--:--</span>
        <span class="text-slate-700">|</span>
        <span class="px-1.5 py-0.5 rounded bg-emerald-500/20 text-emerald-300 border border-emerald-500/40 text-[10px] font-bold tracking-wider">LIVE</span>
        <span class="text-slate-700">|</span>
        <span class="text-slate-400 text-xs flex items-center gap-1.5">
          <span class="text-slate-500">Up:</span>
          <span id="uptime" class="text-slate-300 font-medium">--:--:--</span>
        </span>
      </div>
    </header>

    <!-- Main Live Activity Card -->
    <section class="bg-gradient-to-br from-slate-900/90 to-slate-950 border border-slate-800 rounded-2xl p-5 md:p-6 shadow-xl relative overflow-hidden">
      <div class="flex flex-wrap items-center justify-between gap-4 mb-4">
        <div class="flex items-center gap-3">
          <span class="text-xs font-semibold uppercase tracking-wider text-slate-400">Current Task</span>
          <span id="phase-badge" class="px-2.5 py-1 text-xs font-semibold rounded-full bg-slate-800 text-slate-300 border border-slate-700">
            IDLE
          </span>
        </div>
        <div class="text-right">
          <div class="text-xs text-slate-400">Generation Speed</div>
          <div class="text-2xl font-bold font-mono text-cyan-400" id="live-speed">-- tok/s</div>
        </div>
      </div>

      <!-- Progress / Stats Row -->
      <div class="grid grid-cols-2 sm:grid-cols-4 gap-4 pt-2 pb-4 border-t border-slate-800/80">
        <div>
          <div class="text-xs text-slate-400">Prompt Tokens</div>
          <div class="text-lg font-bold font-mono text-slate-200" id="prompt-tokens">0</div>
        </div>
        <div>
          <div class="text-xs text-slate-400">Generated Tokens</div>
          <div class="text-lg font-bold font-mono text-emerald-400" id="gen-tokens">0</div>
        </div>
        <div>
          <div class="text-xs text-slate-400">Elapsed Time</div>
          <div class="text-lg font-bold font-mono text-sky-400" id="elapsed-time">0.0 s</div>
        </div>
        <div>
          <div class="text-xs text-slate-400">Server State</div>
          <div class="text-lg font-bold font-mono text-slate-300" id="server-state">Ready</div>
        </div>
      </div>

      <!-- Animated Progress Bar -->
      <div class="w-full bg-slate-950 rounded-full h-2.5 overflow-hidden border border-slate-800">
        <div id="progress-bar" class="h-full bg-gradient-to-r from-cyan-500 via-sky-400 to-emerald-400 rounded-full transition-all duration-300 w-0"></div>
      </div>
    </section>

    <!-- Hardware Metrics Grid (GPU, CPU, NVMe) -->
    <div class="grid grid-cols-1 md:grid-cols-3 gap-5">
      
      <!-- GPU Card -->
      <div class="bg-slate-900/80 border border-slate-800 rounded-2xl p-4 shadow-lg space-y-3.5">
        <div class="flex items-center justify-between">
          <div class="flex items-center gap-2">
            <svg class="w-4 h-4 text-emerald-400" fill="none" stroke="currentColor" viewBox="0 0 24 24"><path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M9 3v2m6-2v2M9 19v2m6-2v2M5 9H3m2 6H3m18-6h-2m2 6h-2M7 19h10a2 2 0 002-2V7a2 2 0 00-2-2H7a2 2 0 00-2 2v10a2 2 0 002 2zM9 9h6v6H9V9z"></path></svg>
            <h2 class="font-bold text-slate-200 text-sm truncate max-w-[190px]" id="gpu-name">GPU</h2>
          </div>
          <span class="text-[11px] font-mono font-semibold px-2 py-0.5 rounded bg-emerald-950/80 text-emerald-400 border border-emerald-800/60" id="gpu-util">0% LOAD</span>
        </div>

        <div class="space-y-1">
          <div class="flex justify-between text-xs">
            <span class="text-slate-400">VRAM Usage</span>
            <span class="font-mono text-slate-300 text-[11px]" id="vram-text">0 / 0 MB</span>
          </div>
          <div class="w-full bg-slate-950 rounded-full h-2.5 overflow-hidden border border-slate-800">
            <div id="vram-bar" class="h-full bg-gradient-to-r from-emerald-500 via-yellow-500 to-rose-500 rounded-full transition-all duration-500 w-0"></div>
          </div>
        </div>

        <div class="grid grid-cols-2 gap-2 pt-1 border-t border-slate-800 text-center">
          <div class="bg-slate-950/60 p-2 rounded-xl border border-slate-800/60">
            <div class="text-[10px] text-slate-400">Temp / Power</div>
            <div class="text-xs font-bold font-mono text-slate-200" id="gpu-temp-pwr">0 °C / 0 W</div>
          </div>
          <div class="bg-slate-950/60 p-2 rounded-xl border border-slate-800/60">
            <div class="text-[10px] text-slate-400">Expert Slots</div>
            <div class="text-xs font-bold font-mono text-cyan-400" id="expert-slots">-- slots</div>
          </div>
        </div>
      </div>

      <!-- CPU & RAM Card -->
      <div class="bg-slate-900/80 border border-slate-800 rounded-2xl p-4 shadow-lg space-y-3.5">
        <div class="flex items-center justify-between">
          <div class="flex items-center gap-2">
            <svg class="w-4 h-4 text-indigo-400" fill="none" stroke="currentColor" viewBox="0 0 24 24"><path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M13 10V3L4 14h7v7l9-11h-7z"></path></svg>
            <h2 class="font-bold text-slate-200 text-sm truncate max-w-[190px]" id="cpu-name">CPU</h2>
          </div>
          <span class="text-[11px] font-mono font-semibold px-2 py-0.5 rounded bg-indigo-950/80 text-indigo-400 border border-indigo-800/60" id="cpu-util">0% CPU</span>
        </div>

        <div class="space-y-1">
          <div class="flex justify-between text-xs">
            <span class="text-slate-400">RAM (Weights Arena)</span>
            <span class="font-mono text-slate-300 text-[11px]" id="ram-text">0 / 0 GB</span>
          </div>
          <div class="w-full bg-slate-950 rounded-full h-2.5 overflow-hidden border border-slate-800">
            <div id="ram-bar" class="h-full bg-gradient-to-r from-indigo-500 to-purple-500 rounded-full transition-all duration-500 w-0"></div>
          </div>
        </div>

        <div class="grid grid-cols-2 gap-2 pt-1 border-t border-slate-800 text-center">
          <div class="bg-slate-950/60 p-2 rounded-xl border border-slate-800/60">
            <div class="text-[10px] text-slate-400">Threads</div>
            <div class="text-xs font-bold font-mono text-slate-200" id="cpu-threads">--</div>
          </div>
          <div class="bg-slate-950/60 p-2 rounded-xl border border-slate-800/60">
            <div class="text-[10px] text-slate-400">PCIe Mode</div>
            <div class="text-xs font-bold font-mono text-sky-400" id="pcie-mode">PCIe --</div>
          </div>
        </div>
      </div>

      <!-- NVMe Storage Card -->
      <div class="bg-slate-900/80 border border-slate-800 rounded-2xl p-4 shadow-lg space-y-3.5">
        <div class="flex items-center justify-between">
          <div class="flex items-center gap-2">
            <svg class="w-4 h-4 text-amber-400" fill="none" stroke="currentColor" viewBox="0 0 24 24"><path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M4 7v10c0 2 1 3 3 3h10c2 0 3-1 3-3V7c0-2-1-3-3-3H7C5 4 4 5 4 7zM9 9h6M9 13h6M9 17h2"></path></svg>
            <h2 class="font-bold text-slate-200 text-sm">NVMe Storage (SSD)</h2>
          </div>
          <span class="text-[11px] font-mono font-semibold px-2 py-0.5 rounded bg-amber-950/80 text-amber-400 border border-amber-800/60" id="nvme-m-load">M: 0% IO</span>
        </div>

        <!-- M: Disk -->
        <div class="space-y-1">
          <div class="flex justify-between text-xs">
            <span class="text-slate-400 truncate max-w-[200px]" id="nvme-m-name">M: Model SSD</span>
            <span class="font-mono text-amber-300 text-[11px]" id="nvme-m-speed">R: 0 MB/s | W: 0 MB/s</span>
          </div>
          <div class="w-full bg-slate-950 rounded-full h-2 overflow-hidden border border-slate-800">
            <div id="nvme-m-bar" class="h-full bg-gradient-to-r from-amber-500 to-orange-500 rounded-full transition-all duration-300 w-0"></div>
          </div>
        </div>

        <!-- C: Disk -->
        <div class="space-y-1">
          <div class="flex justify-between text-xs">
            <span class="text-slate-400 truncate max-w-[200px]" id="nvme-c-name">C: System SSD</span>
            <span class="font-mono text-slate-300 text-[11px]" id="nvme-c-speed">R: 0 MB/s | W: 0 MB/s</span>
          </div>
          <div class="w-full bg-slate-950 rounded-full h-2 overflow-hidden border border-slate-800">
            <div id="nvme-c-bar" class="h-full bg-gradient-to-r from-sky-500 to-indigo-500 rounded-full transition-all duration-300 w-0"></div>
          </div>
        </div>
      </div>

    </div>

    <!-- Recent History Table -->
    <div class="bg-slate-900/80 border border-slate-800 rounded-2xl p-5 shadow-lg space-y-3">
      <h3 class="font-semibold text-slate-200 text-sm flex items-center gap-2">
        <svg class="w-4 h-4 text-sky-400" fill="none" stroke="currentColor" viewBox="0 0 24 24"><path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M12 8v4l3 3m6-3a9 9 0 11-18 0 9 9 0 0118 0z"></path></svg>
        Recent Completed Requests (Prefill vs Generation breakdown)
      </h3>
      <div class="overflow-x-auto">
        <table class="w-full text-left text-xs text-slate-300 font-mono">
          <thead class="bg-slate-950/80 text-slate-400 uppercase text-[10px]">
            <tr>
              <th class="py-2.5 px-3 rounded-l-lg">Prompt Size</th>
              <th class="py-2.5 px-3">Prefill Time</th>
              <th class="py-2.5 px-3">Prefill Speed</th>
              <th class="py-2.5 px-3">Gen Tokens</th>
              <th class="py-2.5 px-3">Gen Time</th>
              <th class="py-2.5 px-3">Gen Speed</th>
              <th class="py-2.5 px-3">Total Time</th>
              <th class="py-2.5 px-3 rounded-r-lg">Finish</th>
            </tr>
          </thead>
          <tbody id="history-table" class="divide-y divide-slate-800/50">
            <tr><td colspan="8" class="text-center py-4 text-slate-500">Loading history...</td></tr>
          </tbody>
        </table>
      </div>
    </div>

  </div>

  <script>
    async function updateDashboard() {
      try {
        const res = await fetch('/api/stats?t=' + Date.now());
        const data = await res.json();
        
        document.getElementById('clock').innerText = data.timestamp;
        if (data.uptime && document.getElementById('uptime')) {
          document.getElementById('uptime').innerText = data.uptime;
        }

        // Strata Status
        const st = data.strata;
        if (st.model) document.getElementById('header-model').innerText = st.model;
        if (st.max_context) document.getElementById('header-context').innerText = st.max_context;
        if (st.kv) document.getElementById('header-kv').innerText = st.kv;

        const phaseBadge = document.getElementById('phase-badge');
        const liveSpeed = document.getElementById('live-speed');
        const promptTokens = document.getElementById('prompt-tokens');
        const genTokens = document.getElementById('gen-tokens');
        const elapsedTime = document.getElementById('elapsed-time');
        const serverState = document.getElementById('server-state');
        const progressBar = document.getElementById('progress-bar');
        const statusDot = document.getElementById('status-dot');

        if (st.busy) {
          statusDot.className = 'w-3.5 h-3.5 rounded-full bg-cyan-400 shadow-[0_0_12px_rgba(6,182,212,0.8)] animate-pulse';
          serverState.innerText = 'Processing...';
          serverState.className = 'text-lg font-bold font-mono text-cyan-400';

          if (st.phase === 'reading the prompt') {
            phaseBadge.innerText = 'PREFILL (Reading prompt)';
            phaseBadge.className = 'px-2.5 py-1 text-xs font-semibold rounded-full bg-amber-500/20 text-amber-300 border border-amber-500/40 animate-pulse';
            progressBar.className = 'h-full bg-gradient-to-r from-amber-500 to-yellow-400 rounded-full transition-all duration-300 w-1/3 animate-pulse-slow';
          } else {
            phaseBadge.innerText = st.phase ? st.phase.toUpperCase() : 'GENERATING';
            phaseBadge.className = 'px-2.5 py-1 text-xs font-semibold rounded-full bg-emerald-500/20 text-emerald-300 border border-emerald-500/40 animate-pulse';
            progressBar.className = 'h-full bg-gradient-to-r from-cyan-500 to-emerald-400 rounded-full transition-all duration-300 w-2/3';
          }

          liveSpeed.innerText = st.tokens_per_s > 0 ? (st.tokens_per_s + ' tok/s') : '-- tok/s';
          promptTokens.innerText = st.prompt_tokens.toLocaleString();
          genTokens.innerText = st.generated.toLocaleString();
          elapsedTime.innerText = st.elapsed_s + ' s';
        } else {
          statusDot.className = 'w-3.5 h-3.5 rounded-full bg-emerald-500 shadow-[0_0_12px_rgba(16,185,129,0.7)]';
          phaseBadge.innerText = 'IDLE (Ready)';
          phaseBadge.className = 'px-2.5 py-1 text-xs font-semibold rounded-full bg-slate-800 text-slate-400 border border-slate-700';
          serverState.innerText = 'Ready / Listening';
          serverState.className = 'text-lg font-bold font-mono text-slate-300';
          liveSpeed.innerText = '-- tok/s';
          elapsedTime.innerText = '0.0 s';
          progressBar.style.width = '0%';
        }

        // GPU
        const gpu = data.gpu;
        if (gpu.name) document.getElementById('gpu-name').innerText = gpu.name;
        document.getElementById('gpu-util').innerText = gpu.util + '% LOAD';
        document.getElementById('vram-text').innerText = `${gpu.mem_used_mb.toLocaleString()} / ${gpu.mem_total_mb.toLocaleString()} MB (${gpu.mem_percent}%)`;
        document.getElementById('vram-bar').style.width = gpu.mem_percent + '%';
        document.getElementById('gpu-temp-pwr').innerText = `${gpu.temp_c} °C / ${gpu.power_w} W`;
        if (gpu.pcie) document.getElementById('pcie-mode').innerText = gpu.pcie;
        if (gpu.expert_slots) {
          document.getElementById('expert-slots').innerText = `${gpu.expert_slots.toLocaleString()} slots`;
        } else {
          document.getElementById('expert-slots').innerText = '--';
        }

        // CPU & RAM
        const cpu = data.cpu;
        if (cpu.name) document.getElementById('cpu-name').innerText = cpu.name;
        if (cpu.cores && cpu.threads) {
          document.getElementById('cpu-threads').innerText = `${cpu.cores}C / ${cpu.threads}T`;
        }
        document.getElementById('cpu-util').innerText = Math.round(cpu.util) + '% CPU';
        document.getElementById('ram-text').innerText = `${cpu.ram_used_gb} / ${cpu.ram_total_gb} GB (${cpu.ram_percent}%)`;
        document.getElementById('ram-bar').style.width = cpu.ram_percent + '%';

        // NVMe Disks
        if (data.nvme) {
          const m = data.nvme.m_drive;
          if (m.name) document.getElementById('nvme-m-name').innerText = m.name;
          document.getElementById('nvme-m-load').innerText = `M: ${m.load_pct}% IO`;
          document.getElementById('nvme-m-speed').innerText = `R: ${m.read_mb} MB/s | W: ${m.write_mb} MB/s`;
          document.getElementById('nvme-m-bar').style.width = Math.min(100, Math.max(m.load_pct, (m.read_mb / 2000) * 100)) + '%';

          const c = data.nvme.c_drive;
          if (c.name) document.getElementById('nvme-c-name').innerText = c.name;
          document.getElementById('nvme-c-speed').innerText = `R: ${c.read_mb} MB/s | W: ${c.write_mb} MB/s`;
          document.getElementById('nvme-c-bar').style.width = Math.min(100, Math.max(c.load_pct, (c.read_mb / 2000) * 100)) + '%';
        }

        // History
        const histTable = document.getElementById('history-table');
        if (data.history && data.history.length > 0) {
          histTable.innerHTML = data.history.slice().reverse().map(h => `
            <tr class="hover:bg-slate-800/40 transition-colors">
              <td class="py-2.5 px-3 font-semibold text-slate-200">${h.prompt_tokens.toLocaleString()} tok</td>
              <td class="py-2.5 px-3 text-amber-300/90">${h.prefill_s} s</td>
              <td class="py-2.5 px-3 text-amber-400 font-bold">${h.prefill_speed} tok/s</td>
              <td class="py-2.5 px-3 text-emerald-300">${h.gen_tokens} tok</td>
              <td class="py-2.5 px-3 text-sky-300">${h.gen_s} s</td>
              <td class="py-2.5 px-3 text-cyan-400 font-bold">${h.gen_speed} tok/s</td>
              <td class="py-2.5 px-3 text-slate-400">${h.total_s} s</td>
              <td class="py-2.5 px-3"><span class="px-1.5 py-0.5 rounded bg-slate-800 text-[10px] text-slate-300">${h.finish}</span></td>
            </tr>
          `).join('');
        }
      } catch (err) {
        console.error('Fetch error:', err);
      }
    }

    updateDashboard();
    setInterval(updateDashboard, 1000);
  </script>
</body>
</html>
"""

class Handler(http.server.SimpleHTTPRequestHandler):
    def do_HEAD(self):
        if self.path in ("/", "/index.html"):
            body = HTML_TEMPLATE.encode("utf-8")
            self.send_response(200)
            self.send_header("Content-Type", "text/html; charset=utf-8")
            self.send_header("Content-Length", str(len(body)))
            self.send_header("Access-Control-Allow-Origin", "*")
            self.send_header("Cache-Control", "no-cache, no-store, must-revalidate")
            self.send_header("Pragma", "no-cache")
            self.send_header("Expires", "0")
            self.end_headers()
        elif self.path.startswith("/api/stats"):
            data = json.dumps(get_stats()).encode("utf-8")
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(data)))
            self.send_header("Access-Control-Allow-Origin", "*")
            self.send_header("Cache-Control", "no-cache, no-store, must-revalidate")
            self.send_header("Pragma", "no-cache")
            self.send_header("Expires", "0")
            self.end_headers()
        else:
            self.send_response(404)
            self.end_headers()

    def do_GET(self):
        if self.path in ("/", "/index.html"):
            body = HTML_TEMPLATE.encode("utf-8")
            self.send_response(200)
            self.send_header("Content-Type", "text/html; charset=utf-8")
            self.send_header("Content-Length", str(len(body)))
            self.send_header("Access-Control-Allow-Origin", "*")
            self.send_header("Cache-Control", "no-cache, no-store, must-revalidate")
            self.send_header("Pragma", "no-cache")
            self.send_header("Expires", "0")
            self.end_headers()
            self.wfile.write(body)
        elif self.path.startswith("/api/stats"):
            data = json.dumps(get_stats()).encode("utf-8")
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(data)))
            self.send_header("Access-Control-Allow-Origin", "*")
            self.send_header("Cache-Control", "no-cache, no-store, must-revalidate")
            self.send_header("Pragma", "no-cache")
            self.send_header("Expires", "0")
            self.end_headers()
            self.wfile.write(data)
        else:
            self.send_response(404)
            self.end_headers()

    def log_message(self, format, *args):
        pass

if __name__ == "__main__":
    http.server.ThreadingHTTPServer.allow_reuse_address = True
    with http.server.ThreadingHTTPServer((HOST, PORT), Handler) as httpd:
        print(f"Dashboard running on http://{HOST}:{PORT}")
        httpd.serve_forever()
