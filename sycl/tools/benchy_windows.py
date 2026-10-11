#!/usr/bin/env python3
"""benchy_windows.py: benchy v1 on Windows - sycl/tools/perf_matrix.py's bench, run natively against engine\\strata.exe
(sycl\\tools\\benchy_windows.bat runs it from the .venv).

    .venv\\Scripts\\python sycl\\tools\\benchy_windows.py [--configs strata-a.json,...] [--sizes 20,2185,...] [--runs N]
                                                         [--out DIR] [--force] [--warm]

The bench is benchy v1, unchanged, so a Windows B70 reads against the PR's Linux tables (docs/INTEL_PERFORMANCE.md,
"Benchy v1 on our B70" and "Submitting numbers"):

- **Prompts:** sycl/bench/v1. `short.ids` is the 20-token Fibonacci prompt, `long.ids` 2,185 tokens of code and prose;
  the longer sizes are long.ids' first 2,000 tokens repeated to length (perf_matrix.py's prompt_file, copied). The
  files are comma-separated ids, which is what the engine's --tokens-file reads (generate.cpp: "commas or whitespace").
- **Sizes:** 20, 2,185, 8,000, 40,000, 128,000 and 256,000 tokens. A size that does not fit the config's `--max-context`
  with the 256 new tokens is skipped and listed (the engine refuses it: generate.cpp's "must fit the prompt and
  generation").
- **Each run:** a fresh engine process: the config's `"exe"` (engine\\strata.exe) with the config's args verbatim plus
  `--tokens-file <ids> --max-new 256 --greedy --stats`, cwd the checkout (the config's "cwd"), and the environment the
  server gives the engine (sycl/serve/server_intel.py's sycl_env in docs/INTEL_WINDOWS.md: ONEAPI_DEVICE_SELECTOR
  for the config's card, the Level Zero v2 adapter, STRATA_VERIFY_*, the stager threads, the heartbeat, engine\\ first on
  the PATH). Not through the server: generate mode reads nothing from stdin and prints its stats on stdout, which goes
  with stderr into one log per run, as perf_matrix.py keeps them.
- **Configs:** every `strata-*.json` with `"backend": "sycl"` next to the checkout (what sycl\\setup_intel.py writes),
  except steering variants (`--control-vector*`).

What differs from Linux, and the table says so:

- **Warm.** There is no page-cache drop on Windows, so every run starts warm (`--warm` is accepted and ignored).
- **The sampler.** VRAM in use and power come from Level Zero Sysman through sycl/serve/xe_telemetry.py's Windows reader,
  sampled every second on a thread; "n/a" when the reader does not answer on this driver (unverified on the B70 as of
  2026-10-05). RAM is the drop in the host's available memory (psutil, as perf_matrix.py reads /proc/meminfo), with the
  engine's peak working set beside it in matrix.jsonl. SSD reads are the physical disk's counters (psutil), so with a
  warm cache they are small.
- **The engine is exclusive on the card:** benchy refuses to start while a strata.exe is running or Sysman reports more
  than 2 GiB of VRAM in use (--force overrides). A timeout (a hang, not a slow run) kills the engine and ends the
  matrix: on Windows a wedged queue gives no TDR, and a forced kill made the next run worse (docs/INTEL_WINDOWS.md,
  "TDR and liveness").
- `--runs N` repeats each size N times: the table shows the median and, after PP, TTFT and TG, the min-max spread;
  the "Completed requests" column reads k/N; matrix.jsonl keeps every run.

Out (--out, default sycl\\benchy-results\\v1-<date>-windows, a new folder per invocation): `matrix.md` (the PR's two
tables, the machine and the caveats - post it), `config-<name>.txt` (each config's args - post them too), `matrix.jsonl`
(raw numbers, one line per run), `machine.json`, and one log per run. matrix.md is rewritten after every run, so an
interrupted matrix leaves what it has.
"""
from __future__ import annotations

import argparse
import json
import os
import platform
import re
import shutil
import statistics
import subprocess
import sys
import threading
import time
from pathlib import Path

VERSION = "v1"
HERE = Path(__file__).resolve().parent
REPO = HERE.parents[1]
PROMPTS = REPO / "sycl" / "bench" / VERSION
SIZES = [20, 2185, 8000, 40000, 128000, 256000]
NEW = 256
HELD_BYTES = 2 * 2**30           # VRAM in use that means a served model holds the card (perf_matrix.py's 2048 MiB)
SAMPLE_S = 1.0                   # the sampler's period
sys.path.insert(0, str(REPO))
sys.path.insert(0, str(REPO / "sycl" / "serve"))

if os.name != "nt":
    sys.exit("benchy: this is the Windows benchy (engine\\strata.exe); on Linux run sycl/benchy.sh")
try:
    import psutil                      # setup installs it into the .venv (requirements.txt)
    import server_intel as SI          # noqa: E402  sycl_env, physical_cores; it imports serve/server.py
    from xe_telemetry import xe_gpu    # noqa: E402  the Windows Level Zero Sysman reader
except ImportError as e:               # jinja2 (serve/frontend.py's) and psutil live in the .venv, not the system Python
    sys.exit(f"benchy: {e}\n  run it from the .venv: sycl\\tools\\benchy_windows.bat, or first "
             ".venv\\Scripts\\python -m pip install -r requirements.txt")


# ---------------------------------------------------------------- the machine
def ps(cmd: str) -> str:
    """One PowerShell command's output (setup.py asks WMI this way: Get-CimInstance, no wmic); "" when it does not
    answer.  Windows PowerShell writes a pipe in the console's code page unless told otherwise, so the command is
    prefixed with the UTF-8 switch; anything that still does not decode is replaced."""
    utf8 = "[Console]::OutputEncoding = [System.Text.Encoding]::UTF8; "
    try:
        r = subprocess.run(["powershell", "-NoProfile", "-NonInteractive", "-Command", utf8 + cmd], capture_output=True,
                           text=True, encoding="utf-8", errors="replace", timeout=90)
        return r.stdout
    except (OSError, subprocess.TimeoutExpired):
        return ""


def reg(path: str, name: str):
    """A value under HKEY_LOCAL_MACHINE, or None."""
    try:
        import winreg
        with winreg.OpenKey(winreg.HKEY_LOCAL_MACHINE, path) as k:
            return winreg.QueryValueEx(k, name)[0]
    except (ImportError, OSError):
        return None


def windows_version() -> str:
    """'Windows 11 Pro 25H2 (build 26200.1234)' from the registry; platform.platform() when it does not answer.  The
    registry's ProductName still says "Windows 10" on Windows 11 (builds 22000 and up), so the name is fixed from the
    build."""
    nt = r"SOFTWARE\Microsoft\Windows NT\CurrentVersion"
    build, ubr = str(reg(nt, "CurrentBuildNumber") or ""), reg(nt, "UBR")
    disp, prod = str(reg(nt, "DisplayVersion") or ""), str(reg(nt, "ProductName") or "Windows")
    if not build.isdigit():
        return platform.platform()
    if int(build) >= 22000:
        prod = prod.replace("Windows 10", "Windows 11")
    return " ".join(x for x in (prod, disp, f"(build {build}{'.' + str(ubr) if ubr is not None else ''})") if x)


def cpu_name() -> str:
    n = reg(r"HARDWARE\DESCRIPTION\System\CentralProcessor\0", "ProcessorNameString")
    return str(n).strip() if n else (platform.processor() or "?")


def display_adapters() -> list[dict]:
    """Every display adapter Windows lists, with its driver: [{name, pnp, version, date}] from Win32_VideoController
    (setup.py's _win_display_adapters, plus DriverVersion and DriverDate).  The Intel graphics driver version was the
    one figure missing from the 2026-10-04 record (nothing the engine prints carries it)."""
    cmd = ("Get-CimInstance Win32_VideoController | ForEach-Object { $_.Name + '|' + $_.PNPDeviceID + '|' + "
           "$_.DriverVersion + '|' + $(if ($_.DriverDate) { $_.DriverDate.ToString('yyyy-MM-dd') } else { '' }) }")
    found = []
    for line in ps(cmd).splitlines():
        parts = [p.strip() for p in line.split("|")]
        if len(parts) >= 3 and parts[1]:
            found.append({"name": parts[0], "pnp": parts[1], "version": parts[2], "date": parts[3] if len(parts) > 3 else ""})
    return found


def intel_adapter(adapters: list[dict]) -> dict | None:
    """The BMG-G31 card (PCI VEN_8086, DEV_E220-E223: the Arc Pro B70 is E223, the B65 E222; sycl/setup_intel.py's
    table), else the first Intel adapter, else None."""
    for pat in (r"VEN_8086&DEV_E22[0-3]", r"VEN_8086&DEV_"):
        for a in adapters:
            if re.search(pat, a.get("pnp", ""), re.I):
                return a
    return None


def pcie_link(pnp: str) -> str:
    """'Gen5 x16 (card max Gen5 x16)', the link the card trained at, from the PCI properties Windows keeps for the
    device (devpkey.h's DEVPKEY_PciDevice_CurrentLinkSpeed / Width / MaxLinkSpeed / MaxLinkWidth; the speed value is
    the PCIe generation, 1-6); "?" where they are not answered.  The PR's table prints the same line from sysfs.
    unverified on the B70: that this driver fills these properties."""
    if not pnp or "'" in pnp:
        return "?"
    keys = ["DEVPKEY_PciDevice_CurrentLinkSpeed", "DEVPKEY_PciDevice_CurrentLinkWidth",
            "DEVPKEY_PciDevice_MaxLinkSpeed", "DEVPKEY_PciDevice_MaxLinkWidth"]
    cmd = (f"Get-PnpDeviceProperty -InstanceId '{pnp}' -KeyName {','.join(keys)} -ErrorAction SilentlyContinue | "
           "ForEach-Object { $_.KeyName + '=' + $_.Data }")
    got = {}
    for line in ps(cmd).splitlines():
        k, _, v = line.strip().partition("=")
        got[k] = v.strip()
    g = lambda k: got.get(k, "") if got.get(k, "").isdigit() else "?"
    return f"Gen{g(keys[0])} x{g(keys[1])} (card max Gen{g(keys[2])} x{g(keys[3])})"


def disk_of(path) -> tuple[str, str]:
    """The physical disk a path lives on, as psutil names it ("PhysicalDrive1"), and its model and bus from Get-Disk;
    ("", "every disk") when Windows does not say (no drive letter, a network path) or psutil has no counters for it.
    perf_matrix.py's disk_of does this with findmnt."""
    letter = os.path.splitdrive(str(path))[0].rstrip(":")
    if len(letter) != 1 or not letter.isalpha():
        return "", "every disk"
    cmd = (f"$p = Get-Partition -DriveLetter {letter} -ErrorAction SilentlyContinue | Select-Object -First 1; "
           "if ($p) { $d = Get-Disk -Number $p.DiskNumber; '' + $p.DiskNumber + '|' + $d.FriendlyName + '|' + $d.BusType }")
    parts = [x.strip() for x in ps(cmd).strip().split("|")]
    if len(parts) < 3 or not parts[0].isdigit():
        return "", "every disk"
    dev = f"PhysicalDrive{parts[0]}"
    try:
        known = dev in psutil.disk_io_counters(perdisk=True)
    except (RuntimeError, OSError):
        known = False
    desc = f"{dev} ({parts[1]}, {parts[2]})"
    return (dev, desc) if known else ("", desc + "; reads counted over every disk")


def disk_read_bytes(dev: str) -> int:
    """Bytes read from that physical disk since boot (every disk when dev is ""); 0 when Windows has no counters
    (psutil: older systems need `diskperf -y` once)."""
    try:
        if dev:
            return int(psutil.disk_io_counters(perdisk=True)[dev].read_bytes)
        c = psutil.disk_io_counters()
        return int(c.read_bytes) if c else 0
    except (KeyError, RuntimeError, OSError, AttributeError):
        return 0


def build_json() -> dict:
    """engine\\BUILD.json: version, aot, oneapi, src, built."""
    try:
        return json.loads((REPO / "engine" / "BUILD.json").read_text(encoding="utf-8-sig"))
    except (OSError, ValueError):
        return {}


def git_commit() -> str:
    """As perf_matrix.py records it; "?" without git (this tree is also run from a plain copy)."""
    if not shutil.which("git"):
        return "?"
    try:
        git = lambda *a: subprocess.run(["git", "-C", str(REPO), *a], capture_output=True, text=True,
                                        encoding="utf-8", errors="replace", timeout=30).stdout.strip()
        head = git("rev-parse", "--short=12", "HEAD")
        return (head or "?") + (" + local changes" if head and git("status", "--porcelain", "--untracked-files=no") else "")
    except (OSError, subprocess.TimeoutExpired):
        return "?"


def env_summary(env: dict) -> str:
    """The C4 variables the engine ran with, in the table's order, then any other STRATA_/SYCL_/UR_/ONEAPI_/ZES_ one
    the environment or the config added ("Submitting numbers" asks for them)."""
    keys = ("ONEAPI_DEVICE_SELECTOR", "SYCL_UR_USE_LEVEL_ZERO_V2", "SYCL_CACHE_PERSISTENT",
            "UR_L0_ENABLE_RELAXED_ALLOCATION_LIMITS", "ZES_ENABLE_SYSMAN", "STRATA_VERIFY_DEVICE_PLAN",
            "STRATA_VERIFY_NO_HOST", "STRATA_STAGER_THREADS", "STRATA_SYCL_HEARTBEAT_S")
    extra = sorted(k for k in env if k not in keys and k.startswith(("STRATA_", "SYCL_", "UR_", "ONEAPI_", "ZES_")))
    return " ".join(f"{k}={env[k]}" for k in keys + tuple(extra) if k in env)


def machine(cfg: dict, env: dict, gpu, dev_desc: str) -> dict:
    adapters = display_adapters()
    intel = intel_adapter(adapters) or {}
    try:
        reading = gpu.read() if gpu is not None else {}
    except Exception:   # noqa: BLE001 - a ctypes reader must never stop the bench
        reading = {}
    total = reading.get("mem_total")
    # the card's name: Sysman's when it gave one, else Windows' adapter name, else the reader's generic one
    name = getattr(gpu, "name_", None) or intel.get("name") or (gpu.name() if gpu is not None else None) or "?"
    exe = Path(cfg["exe"])
    try:
        binary = str(exe.relative_to(REPO))
    except ValueError:
        binary = str(exe)
    return {
        "bench": f"benchy {VERSION}", "date": time.strftime("%Y-%m-%d"), "platform": "Windows, native",
        "card": {"name": name, "vram_gb": round(total / 2**30) if total else None,
                 "link": pcie_link(intel.get("pnp", "")), "pnp": intel.get("pnp", "")},
        "driver": intel.get("version") or "?", "driver_date": intel.get("date") or "",
        "other_adapters": [f"{a['name']} (driver {a['version']})" for a in adapters if a is not intel],
        "cpu": cpu_name(), "threads": os.cpu_count(), "cores": SI.physical_cores(),
        "ram_gib": round(psutil.virtual_memory().total / 2**30, 1), "ssd": dev_desc, "os": windows_version(),
        "python": platform.python_version(), "commit": git_commit(), "build": build_json(), "binary": binary,
        "engine_env": env_summary(env), "cold_cache": False,
        "vram_source": "sysman" if gpu is not None else None,
    }


# ---------------------------------------------------------------- one run
def prompt_file(n: int, outdir: Path) -> Path:
    """the v1 prompt of n tokens (perf_matrix.py's prompt_file: short.ids at 20, long.ids at 2,185, else long.ids'
    first 2,000 ids repeated), written under the out dir as the engine's --tokens-file reads it: ids with commas"""
    short = (PROMPTS / "short.ids").read_text(encoding="utf-8").strip().split(",")
    long_ = (PROMPTS / "long.ids").read_text(encoding="utf-8").strip().split(",")
    if n == len(short):
        ids = short
    elif n == len(long_):
        ids = long_
    else:
        ids = (long_[:2000] * (n // 2000 + 1))[:n]
    ids = [x.strip() for x in ids]
    if not ids or not all(x.isdigit() for x in ids):
        sys.exit(f"benchy: {PROMPTS}: the id files must be comma-separated integers")
    p = outdir / "prompts" / f"{n}.ids"
    p.parent.mkdir(parents=True, exist_ok=True)
    if not p.exists():
        p.write_text(",".join(ids), encoding="utf-8")
    return p


def parse(log: str) -> dict:
    """perf_matrix.py's parse(), copied unchanged so matrix.jsonl carries the PR's field names: the --stats lines on
    stdout (generate.cpp: `decode N tokens in T ms  ->  R tok/s`, `prefill N tokens in T ms  ->  R tok/s  (time to first
    token T ms)`, `speculation ... drafts accepted A of B (f)`; src/kernels/ngram.cpp's `ple io: ... SSD reads (M MB)`),
    the start-up lines on stderr (`session is up (engine V)`, `expert cache N slots, G GiB of VRAM`, `N of M experts
    missing from VRAM mirrored in pinned host memory (G GiB`, `prompt path borrows N cache slots`) and the exit line
    this script appends."""
    g = lambda pat, t=float: (lambda m: t(m.group(1)) if m else None)(re.search(pat, log, re.M))
    return {
        "engine": g(r"session is up \(engine ([^)]+)\)", str),
        "pp": g(r"^prefill\s+\d+ tokens in [\d.]+ ms\s+->\s+([\d.]+) tok/s"),
        "ttft_s": (g(r"time to first token ([\d.]+) ms") or 0) / 1000 or None,
        "decode": g(r"^decode\s+\d+ tokens in [\d.]+ ms\s+->\s+([\d.]+) tok/s"),
        "decoded": g(r"^decode\s+(\d+) tokens in", int),
        "accept": g(r"drafts accepted \d+ of \d+ \(([\d.]+)\)"),
        "cache_slots": g(r"expert cache (\d+) slots", int),
        "cache_gib": g(r"expert cache \d+ slots, ([\d.]+) GiB"),
        "mirror_experts": g(r"(\d+) of \d+ experts missing from VRAM mirrored", int) or 0,
        "mirror_gib": g(r"mirrored in pinned host memory \(([\d.]+) GiB") or 0.0,
        "lent_slots": g(r"prompt path borrows (\d+) cache slots", int) or 0,
        "ple_ssd_mb": g(r"SSD reads \(([\d.]+) MB\)"),
        "exit": g(r"ENGINE EXIT (-?\d+)", int),
    }


def parse_scythe(log: str) -> dict:
    """the Windows port's own start-up lines, kept beside the PR's fields because the first model run's log was never pasted
    (2026-10-05): the VRAM arena (C7, device.dp.cpp), the expert arena's registration (pinned.dp.cpp), the WDDM budget
    (C6), the SYCL runtime and Level Zero loader DLLs the process loaded (generate.cpp), the heartbeat line (C9, serve
    mode only, so usually absent here), the prefill line on stderr, and whether the device was lost."""
    g = lambda pat: (lambda m: m.group(1).strip() if m else None)(re.search(pat, log, re.M))
    return {
        "arena": g(r"^strata: VRAM arena (.*)$"),
        "registered": g(r"(prepare_for_device_copy ok \([^)]*\))"),
        "wddm_budget": g(r"^strata: Windows budgets (\d+ of this card's \d+ MiB)"),
        "sycl_dll": g(r"^strata generate: sycl\d\.dll is (.*)$"),
        "ze_loader": g(r"^strata generate: ze_loader\.dll is (.*)$"),
        "heartbeat": g(r"^strata: VRAM heartbeat every (\d+ s, touching \d+ MiB)"),
        "prefill_stderr": g(r"^strata generate: prefill (\d+ tokens in \d+ chunks?, [\d.]+ ms \([\d.]+ tok/s\))"),
        "device_lost": bool(re.search(r"DEVICE_LOST", log)),
    }


class Sampler(threading.Thread):
    """Every second while a run lasts, on its own thread (a slow Sysman call must not delay the timeout): the card's
    VRAM in use and power from xe_telemetry's Sysman reader (None without it), the engine's peak working set and bytes
    read (psutil), and the host's available memory.  Daemon: a stuck reader cannot hold the run."""

    def __init__(self, gpu, proc):
        super().__init__(daemon=True)
        self.gpu, self.proc = gpu, proc
        self.stop = threading.Event()
        self.lock = threading.Lock()
        self.vram: list[tuple[float, int]] = []      # (time, bytes in use on the card)
        self.power: list[tuple[float, float]] = []   # (time, W between two energy samples)
        self.min_avail = psutil.virtual_memory().available
        self.peak_wset = 0
        self.proc_read = 0
        self.samples = 0

    def run(self):
        while not self.stop.wait(SAMPLE_S):
            self.sample()
        self.sample()

    def sample(self):
        t = time.time()
        try:
            r = self.gpu.read() if self.gpu is not None else {}
        except Exception:   # noqa: BLE001 - a ctypes reader must never take the run down
            r = {}
        avail = psutil.virtual_memory().available
        wset = rd = None
        try:
            p = psutil.Process(self.proc.pid)
            with p.oneshot():
                wset = getattr(p.memory_info(), "peak_wset", None)   # Windows: the process's peak working set
                rd = p.io_counters().read_bytes
        except (psutil.Error, OSError):
            pass
        with self.lock:
            if r.get("mem_used") is not None:
                self.vram.append((t, int(r["mem_used"])))
            if r.get("power") is not None:
                self.power.append((t, float(r["power"])))
            self.min_avail = min(self.min_avail, avail)
            if wset:
                self.peak_wset = max(self.peak_wset, int(wset))
            if rd:
                self.proc_read = max(self.proc_read, int(rd))
            self.samples += 1


def read_log(p: Path) -> str:
    try:
        return p.read_text(encoding="utf-8", errors="replace")
    except OSError:
        return ""


def append_log(p: Path, line: str) -> None:
    """A line of our own at the end of a run's log, after the engine has closed it (OneDrive may hold a file it is
    syncing for a moment, hence the retries)."""
    for _ in range(5):
        try:
            with open(p, "a", encoding="utf-8", errors="replace") as f:
                f.write(line)
            return
        except OSError:
            time.sleep(1)


def exe_of(cfg: dict) -> Path:
    """The config's "exe", made absolute against its "cwd" as serve/server.py does (CreateProcess resolves nothing)."""
    exe = str(cfg["exe"])
    return Path(exe) if os.path.isabs(exe) else Path(cfg.get("cwd") or REPO) / exe


def vram_used(gpu):
    """Bytes in use on the card now, or None without the reader."""
    if gpu is None:
        return None
    try:
        return gpu.read().get("mem_used")
    except Exception:   # noqa: BLE001
        return None


def run_one(name: str, cfg: dict, n: int, k: int, outdir: Path, env: dict, gpu, dev: str, timeout: float) -> dict:
    args0 = list(cfg["args"])
    ctx = int(args0[args0.index("--max-context") + 1]) if "--max-context" in args0 else 0
    if ctx and n + NEW > ctx:
        return {"config": name, "prompt": n, "ctx": ctx, "skipped": f"needs --max-context {n + NEW}"}
    exe, cwd = exe_of(cfg), Path(cfg.get("cwd") or REPO)
    args = args0 + ["--tokens-file", str(prompt_file(n, outdir)), "--max-new", str(NEW), "--greedy", "--stats"]
    cmd = [str(exe), *args]
    log_path = outdir / (f"{name}-{n}.log" if k == 0 else f"{name}-{n}-run{k + 1}.log")
    base_avail, base_disk, base_vram = psutil.virtual_memory().available, disk_read_bytes(dev), vram_used(gpu) or 0
    t0 = time.time()
    load_s = load_disk = t_up = t_end = None
    timed_out = False
    with open(log_path, "wb") as lf:
        # the command and the C4 variables first, then everything the engine prints (stdout: the --stats lines and the
        # ids; stderr: its start-up notes), as perf_matrix.py keeps them in one file
        lf.write((f"benchy {VERSION} windows: {subprocess.list2cmdline(cmd)}\nbenchy env: {env_summary(env)}\n")
                 .encode("utf-8", "replace"))
        lf.flush()
        p = subprocess.Popen(cmd, cwd=str(cwd), env=env, stdin=subprocess.DEVNULL, stdout=lf, stderr=subprocess.STDOUT)
        s = Sampler(gpu, p)
        s.start()
        try:
            while p.poll() is None:
                time.sleep(SAMPLE_S)
                text = read_log(log_path)
                if load_s is None and "session is up" in text:   # perf_matrix.py's load time: launch to this line
                    load_s, load_disk, t_up = time.time() - t0, disk_read_bytes(dev) - base_disk, time.time()
                if t_end is None and re.search(r"^decode\s+\d+ tokens", text, re.M):
                    t_end = time.time()
                if time.time() - t0 > timeout:   # a hang, not a slow run: the timeout is several times the expected time
                    timed_out = True
                    p.kill()
                    break
        except KeyboardInterrupt:
            if p.poll() is None:
                p.kill()
            p.wait()
            s.stop.set()
            append_log(log_path, "\nBENCHY INTERRUPTED\n")
            raise
        p.wait()
        s.stop.set()
        s.join(timeout=10)
    if timed_out:
        append_log(log_path, "\nBENCHY TIMEOUT\n")
    append_log(log_path, f"\nENGINE EXIT {p.returncode}\n")
    total_disk = disk_read_bytes(dev) - base_disk
    log = read_log(log_path)
    if t_end is None:
        t_end = time.time()
    with s.lock:
        vram, power = list(s.vram), list(s.power)
        min_avail, peak_wset, proc_read, samples = s.min_avail, s.peak_wset, s.proc_read, s.samples
    peak = max((v for _, v in vram), default=None)
    watts = [w for t, w in power if t_up is not None and t_up <= t <= t_end]
    return {"config": name, "prompt": n, "run": k + 1, "new": NEW, "ctx": ctx, **parse(log), **parse_scythe(log),
            "peak_vram_gb": (peak - base_vram) / 2**30 if peak is not None else None,
            "peak_vram_abs_gb": peak / 2**30 if peak is not None else None,
            "ram_gb": (base_avail - min_avail) / 2**30, "peak_wset_gb": peak_wset / 2**30 if peak_wset else None,
            "avg_power_w": sum(watts) / len(watts) if watts else None, "peak_power_w": max(watts) if watts else None,
            "load_s": load_s, "ssd_load_gb": (load_disk or 0) / 1e9,
            "ssd_request_gb": (total_disk - (load_disk or 0)) / 1e9,
            "proc_read_gb": proc_read / 1e9 if proc_read else None, "wall_s": time.time() - t0,
            "timeout": timed_out, "samples": samples, "log": log_path.name}


# ---------------------------------------------------------------- the table
NUM_MEDIAN = ("pp", "ttft_s", "decode", "accept", "cache_gib", "mirror_gib", "ple_ssd_mb", "peak_vram_gb",
              "peak_vram_abs_gb", "ram_gb", "peak_wset_gb", "avg_power_w", "peak_power_w", "load_s", "ssd_load_gb",
              "ssd_request_gb", "proc_read_gb", "wall_s")
NUM_LOW = ("decoded", "cache_slots", "mirror_experts", "lent_slots")   # counts: an observed value, not a half


def completed(r: dict) -> bool:
    return r.get("exit") == 0 and not r.get("timeout") and r.get("decode") is not None


def median(xs, low=False):
    xs = [x for x in xs if x is not None]
    if not xs:
        return None
    return statistics.median_low(xs) if low else statistics.median(xs)


def spread(xs):
    xs = [x for x in xs if x is not None]
    return (min(xs), max(xs)) if len(xs) > 1 else None


def aggregate(rows: list[dict]) -> list[dict]:
    """One table row per (config, prompt): the median over the completed runs (over every run when none completed),
    the min-max spread of PP, TTFT and TG, and how many of the runs completed.  With --runs 1 the row is the run."""
    groups: dict = {}
    for r in rows:
        groups.setdefault((r["config"], r["prompt"]), []).append(r)
    out = []
    for rs in groups.values():
        if "skipped" in rs[0]:
            out.append(dict(rs[0]))
            continue
        done = [r for r in rs if completed(r)] or rs
        agg = dict(done[0])
        for key in NUM_MEDIAN:
            agg[key] = median([r.get(key) for r in done])
        for key in NUM_LOW:
            agg[key] = median([r.get(key) for r in done], low=True)
        agg["runs"], agg["completed"] = len(rs), sum(completed(r) for r in rs)
        agg["spread"] = {key: spread([r.get(key) for r in done]) for key in ("pp", "ttft_s", "decode")}
        out.append(agg)
    return out


def fmt(v, spec):
    return "-" if v is None else format(v, spec)


def cell(v, spec, sp=None):
    """the median, and the min-max spread after it when the runs differed"""
    s = fmt(v, spec)
    if sp and sp[0] != sp[1]:
        s += f" ({format(sp[0], spec)}-{format(sp[1], spec)})"
    return s


def table(rows: list[dict], m: dict, runs: int) -> str:
    """perf_matrix.py's table(): the same two tables (the speed table in the layout of gresstant's B580 report on the
    upstream PR, then resources), the system and build, and the caveats, with the Windows lines added"""
    agg = aggregate(rows)
    done = [r for r in agg if "skipped" not in r]
    out = [f"**benchy {VERSION}** ({m['date']}, Windows)", "",
           "| Model / configuration | Input tokens | PP (tok/s) | TTFT (s) | TG (tok/s) | Drafts accepted | Actual output tokens | Completed requests |",
           "|---|---:|---:|---:|---:|---:|---:|---:|"]
    for r in done:
        sp = r.get("spread") or {}
        out.append(f"| {r['config']}, {r['ctx'] // 1024}K context | {r['prompt']:,} | {cell(r['pp'], ',.2f', sp.get('pp'))} "
                   f"| {cell(r['ttft_s'], '.1f', sp.get('ttft_s'))} | {cell(r['decode'], '.2f', sp.get('decode'))} "
                   f"| {fmt(r['accept'], '.0%')} | {fmt(r.get('decoded'), ',')} | {r['completed']}/{r['runs']} |")
    out += ["", "| Model / configuration | Input tokens | Experts in VRAM | Offloaded to the RAM mirror | Slots lent to the prompt "
                "| Peak VRAM (GB) | RAM (GB) | SSD read at load (GB) | SSD read, request (GB) | PLE rows from SSD (MB) | Load (s) | Avg power (W) |",
            "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|"]
    for r in done:
        out.append(f"| {r['config']} | {r['prompt']:,} | {fmt(r['cache_slots'], ',')} ({fmt(r['cache_gib'], '.1f')} GiB) "
                   f"| {r['mirror_experts']:,} ({r['mirror_gib']:.1f} GiB) | {r['lent_slots']:,} | {fmt(r['peak_vram_gb'], '.1f')} "
                   f"| {r['ram_gb']:.1f} | {r['ssd_load_gb']:.1f} | {r['ssd_request_gb']:.2f} | {fmt(r['ple_ssd_mb'], ',.0f')} "
                   f"| {fmt(r['load_s'], '.0f')} | {fmt(r['avg_power_w'], '.0f')} |")
    failed = [f"{r['config']} {r['prompt']:,}" + (f" run {r['run']}" if runs > 1 else "")
              + f" (exit {r.get('exit')}{', timeout' if r.get('timeout') else ''}{', device lost' if r.get('device_lost') else ''})"
              for r in rows if "skipped" not in r and not completed(r)]
    skipped = [f"{r['config']} {r['prompt']:,} ({r['skipped']})" for r in agg if "skipped" in r]
    c, b = m["card"], m["build"]
    build = ", ".join(x for x in (f"version {b.get('version')}" if b.get("version") else "",
                                  f"aot {b.get('aot') or 'none'}" if b else "", b.get("oneapi", ""),
                                  f"built {b.get('built')}" if b.get("built") else "", f"src {b.get('src')}" if b.get("src") else "") if x)
    others = ("; beside it " + ", ".join(m["other_adapters"])) if m.get("other_adapters") else ""
    out += ["", "System and build:",
            f"- {c['name']}, {c['vram_gb'] if c['vram_gb'] else '?'} GB, PCIe {c['link']}; Windows graphics driver {m['driver']}"
            + (f" ({m['driver_date']})" if m.get("driver_date") else "") + f"{others}.",
            f"- {m['cpu']} ({m['threads']} threads, {m['cores']} cores), {m['ram_gib']} GiB RAM; models on {m['ssd']}.",
            f"- {m['os']}; Python {m['python']}.",
            f"- Engine {next((r['engine'] for r in rows if r.get('engine')), '?')}, commit `{m['commit']}`, "
            f"`{m['binary']}` (BUILD.json: {build or '?'}). Each configuration's engine args: `config-<name>.txt`.",
            f"- The engine's environment (sycl/serve/server_intel.py): `{m['engine_env']}`.",
            "", "Caveats:",
            "- Windows, native: engine\\strata.exe run directly (no Docker, no server), a WARM page cache - Windows has no "
            "page-cache drop, so every run starts warm; " + (f"{runs} runs per row, the median, with the min-max spread in brackets after PP, TTFT and TG; "
                                                            if runs > 1 else "one run per row, ")
            + f"greedy, {NEW} new tokens, a fresh engine each time.",
            "- TG includes speculative decoding (the MTP draft layer): it depends on the text, and on how often drafts are accepted.",
            "- RAM is the drop in the host's available memory (psutil; pinned memory included); the engine's peak working set is in "
            "matrix.jsonl (peak_wset_gb). VRAM and power: "
            + ("the card's figures from Level Zero Sysman (sycl/serve/xe_telemetry.py), sampled every second - VRAM is the peak in "
               "use less what was in use before the run (the absolute peak is matrix.jsonl's peak_vram_abs_gb), power the mean "
               "between the engine's `session is up` line and the end of decode."
               if m["vram_source"] else "n/a, the Sysman reader did not answer on this driver."),
            "- SSD reads are the physical disk's counters (psutil), the bytes that actually came from the disk: with a warm cache "
            "they are smaller than what the engine asked for (matrix.jsonl's proc_read_gb is the engine's own read count).",
            "- Peak VRAM and RAM are in GiB (2^30 bytes) and the SSD columns in GB (10^9), as perf_matrix.py computes them."]
    if failed:
        out.append("- Did not complete: " + ", ".join(failed) + ".")
    if skipped:
        out.append("- Skipped (context too small): " + ", ".join(skipped) + ".")
    return "\n".join(out) + "\n"


def write_md(outdir: Path, rows: list[dict], m: dict, runs: int) -> str:
    md = table(rows, m, runs)
    (outdir / "matrix.md").write_text(md, encoding="utf-8")
    return md


def line_of(r: dict) -> str:
    """one line on the console per run (the whole row goes to matrix.jsonl)"""
    if "skipped" in r:
        return f"  {r['config']} {r['prompt']:,}: skipped ({r['skipped']})"
    return (f"  {r['config']} {r['prompt']:,} run {r['run']}: PP {fmt(r['pp'], ',.1f')} tok/s, TTFT {fmt(r['ttft_s'], '.1f')} s, "
            f"TG {fmt(r['decode'], '.2f')} tok/s, drafts {fmt(r['accept'], '.0%')}, load {fmt(r['load_s'], '.0f')} s, "
            f"peak VRAM {fmt(r['peak_vram_gb'], '.1f')} GB, RAM {r['ram_gb']:.1f} GB, {fmt(r['avg_power_w'], '.0f')} W, "
            f"exit {r.get('exit')}{', TIMEOUT' if r.get('timeout') else ''}{', DEVICE LOST' if r.get('device_lost') else ''}, "
            f"{r['wall_s']:.0f} s ({r['log']})")


def default_configs() -> list[Path]:
    out = []
    for p in sorted(REPO.glob("strata-*.json")):
        try:
            c = json.loads(p.read_text(encoding="utf-8-sig"))
        except (OSError, ValueError):
            continue
        if c.get("backend") == "sycl" and not any(str(a).startswith("--control-vector") for a in c.get("args", [])):
            out.append(p)
    return out


def reader_for(cfg: dict):
    """xe_telemetry's Sysman reader for the config's card (its "gpu" is the SYCL index; Level Zero enumerates in the same
    order), or None when the loader does not answer."""
    gl = SI.S.gpu_list(cfg)
    try:
        g = xe_gpu(gl[0] if gl else 0)
        return g if g.ok() else None
    except Exception:   # noqa: BLE001 - a ctypes reader must never stop the bench
        return None


def main() -> int:
    ap = argparse.ArgumentParser(description=f"benchy {VERSION} on Windows: the Intel SYCL performance matrix, run natively")
    ap.add_argument("--configs", help="comma-separated serve configs (default: every sycl strata-*.json next to the checkout)")
    ap.add_argument("--sizes", default=",".join(map(str, SIZES)), help="prompt sizes (default: v1's)")
    ap.add_argument("--runs", type=int, default=1, help="runs per size: the table shows the median and the spread (default 1)")
    ap.add_argument("--warm", action="store_true", help="accepted for sycl/benchy.sh's command line; every run is warm on Windows anyway")
    ap.add_argument("--force", action="store_true", help="run even if a strata.exe is running or VRAM is in use")
    ap.add_argument("--out", help="output directory (default sycl\\benchy-results\\v1-<date>-windows)")
    a = ap.parse_args()
    if a.runs < 1:
        ap.error("--runs must be 1 or more")
    try:
        sizes = [int(s) for s in a.sizes.split(",") if s.strip()]
    except ValueError:
        ap.error("--sizes: comma-separated token counts")
    if not sizes or any(n < 1 for n in sizes):
        ap.error("--sizes: comma-separated token counts")
    for f in ("short.ids", "long.ids"):
        if not (PROMPTS / f).exists():
            sys.exit(f"benchy: {PROMPTS / f} is missing (the v1 prompts)")
    cfgs = [Path(c) if os.path.isabs(c) else REPO / c for c in a.configs.split(",") if c.strip()] if a.configs else default_configs()
    if not cfgs:
        sys.exit("benchy: no serve config found - run sycl\\START-HERE.bat --backend sycl for a model first")
    configs = []
    for cp in cfgs:
        try:
            cfg = json.loads(cp.read_text(encoding="utf-8-sig"))
        except (OSError, ValueError) as e:
            sys.exit(f"benchy: {cp}: {e}")
        if cfg.get("backend") != "sycl" or not cfg.get("exe") or not isinstance(cfg.get("args"), list):
            sys.exit(f"benchy: {cp.name} is not a SYCL serve config (sycl\\setup_intel.py writes those)")
        if not exe_of(cfg).exists():
            sys.exit(f"benchy: {exe_of(cfg)} is not built (sycl\\tools\\build_windows.bat)")
        configs.append((cp, cfg))
    # the card must be idle: the engine is exclusive on it
    running = []
    for p in psutil.process_iter(["name", "pid"]):
        if (p.info.get("name") or "").lower() == "strata.exe":
            running.append(p.info["pid"])
    first_cfg = configs[0][1]
    gpu = reader_for(first_cfg)
    held = vram_used(gpu)
    if running and not a.force:
        sys.exit(f"benchy: strata.exe is running (pid {running[0]}) - stop the served model first (close its window), or --force")
    if held is not None and held > HELD_BYTES and not a.force:
        sys.exit(f"benchy: {held / 2**30:.1f} GiB of VRAM is in use - stop the served model first (or --force)")
    if gpu is None:
        print("benchy: the Level Zero Sysman reader does not answer (sycl/serve/xe_telemetry.py): VRAM and power will be n/a",
              flush=True)
    if a.out:
        outdir = Path(a.out)
    else:
        base = REPO / "sycl" / "benchy-results" / f"{VERSION}-{time.strftime('%Y-%m-%d')}-windows"
        outdir, i = base, 1
        while outdir.exists():   # a new folder per invocation: a rerun the same day does not mix with the first
            i += 1
            outdir = base.with_name(f"{base.name}-{i}")
    outdir.mkdir(parents=True, exist_ok=True)
    first = first_cfg["args"]
    dev, dev_desc = disk_of(first[first.index("--native") + 1] if "--native" in first[:-1] else REPO)
    env0 = SI.sycl_env(first_cfg)
    m = machine(first_cfg, env0, gpu, dev_desc)
    (outdir / "machine.json").write_text(json.dumps(m, indent=1), encoding="utf-8")
    print(f"benchy {VERSION} on Windows: {len(configs)} config(s), sizes {', '.join(f'{n:,}' for n in sizes)}, "
          f"{a.runs} run(s) each, warm -> {outdir}", flush=True)
    print(f"  {m['card']['name']}, driver {m['driver']}; {m['cpu']}; {m['os']}; VRAM/power sampler: "
          f"{'Level Zero Sysman' if m['vram_source'] else 'n/a'}", flush=True)
    rows: list[dict] = []
    stop = None
    try:
        with open(outdir / "matrix.jsonl", "a", encoding="utf-8") as jl:
            for cp, cfg in configs:
                name = cp.stem[len("strata-"):] if cp.stem.startswith("strata-") else cp.stem
                (outdir / f"config-{name}.txt").write_text(subprocess.list2cmdline(cfg["args"]) + "\n", encoding="utf-8")
                env, gpu = SI.sycl_env(cfg), reader_for(cfg)
                for n in sizes:
                    for k in range(a.runs):
                        r = run_one(name, cfg, n, k, outdir, env, gpu, dev, timeout=300 + n * 0.02)
                        rows.append(r)
                        jl.write(json.dumps(r) + "\n")
                        jl.flush()
                        print(line_of(r), flush=True)
                        if "skipped" in r:
                            break   # the same size is skipped on every run
                        write_md(outdir, rows, m, a.runs)
                        if r.get("timeout") or r.get("device_lost"):
                            stop = ("the engine hung and was killed" if r.get("timeout") else "the device was lost") + \
                                   f" ({r['log']}); the card may be wedged: Disable-PnpDevice / Enable-PnpDevice it or " \
                                   "restart before the next run (docs/INTEL_WINDOWS.md, TDR and liveness)"
                            break
                    if stop:
                        break
                if stop:
                    break
    except KeyboardInterrupt:
        stop = "interrupted (Ctrl+C); the engine was ended"
    md = write_md(outdir, rows, m, a.runs)
    print("\n" + md + f"\nwritten to {outdir}: post matrix.md and the config-*.txt files")
    if stop:
        print(f"benchy: stopped early: {stop}", flush=True)
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
