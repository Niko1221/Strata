#!/usr/bin/env python3
"""report_windows.py: one paste that tells us everything after a model run on the Windows B70.

    .venv\\Scripts\\python sycl\\tools\\report_windows.py [--log PATH] [--config PATH] [--tail N] [--since LINE] [--out PATH]

It prints, and writes to sycl\\benchy-results\\report-<timestamp>.txt (the folder sycl/.gitignore keeps out of the
tree, as benchy's results are):

  (a) the machine: Windows version and build, CPU, RAM, every display adapter with its driver version and date - the
      B70/B65 is the PCI\\VEN_8086&DEV_E22x one (BMG-G31; docs/INTEL_WINDOWS.md wants driver 32.0.101.9034 or newer, or
      a bundled ze_loader.dll 1.32+), the NVIDIA card that drives the desktop, the processor's graphics - and the Level
      Zero loader in System32.  From PowerShell's CIM classes in one call (Win32_OperatingSystem, Win32_Processor,
      Win32_ComputerSystem, Win32_VideoController), so no package is needed;
  (b) engine\\BUILD.json and what sits beside it;
  (c) the newest strata-*.json (exe, args, gpu, lib_dirs, env, log) and its run-*.bat;
  (d) the engine log's key lines with their line numbers (the arena, the registration, the heartbeat, the runtime
      DLLs, the budget, the expert cache, every warning or error, the request lines), at most 200 of them;
  (e) the log's last lines (--tail, default 60);
  (f) the card's Level Zero Sysman readings now, through sycl/serve/xe_telemetry.py's Windows reader;
  (g) what docs/INTEL_WINDOWS.md still wants measured.

The log is the config's "log" (setup writes strata-<model>.log next to the checkout; serve/server.py appends the
engine's stderr to it, every start to the same file), else the newest strata-*.log there, else --log.  Standard
library only; psutil (setup installs it into the .venv) adds the RAM figures when it is there.  Nothing here measures:
it collects what is already on the PC, with or without the server running.
"""
from __future__ import annotations

import argparse
import json
import os
import platform
import re
import subprocess
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
ENGINE = ROOT / "engine"
SERVE = ROOT / "sycl" / "serve"
OUT_DIR = ROOT / "sycl" / "benchy-results"
WIN = os.name == "nt"

# (d) what to pull out of the engine log, case-insensitively: the port's own start lines (docs/INTEL_WINDOWS.md: the arena
# rule C7, the registration, the heartbeat C9, the runtime DLLs, the budget C6, the device check C3), upstream's
# expert-cache and request lines, and anything that reads like trouble
KEY_PATTERNS = ["strata: VRAM arena", "prepare_for_device_copy", "cudaHostRegister", "VRAM heartbeat", "SYCL runtime",
                "sycl9.dll", "ze_loader", "Windows budgets", "cannot run", "expert cache", "slots", "resident", "mirror",
                "stream", "warning", "error", "FAILED", "exception", "DEVICE_LOST", "timed out", "spin", "doorbell",
                "READY", "tok/s", "prompt", "accept"]
KEY_RE = re.compile("|".join(re.escape(p) for p in KEY_PATTERNS), re.IGNORECASE)
KEY_CAP = 200
START_MARK = "session is up"           # generate.cpp's line once the model is loaded: one per engine start
LINE_CLIP = 400                        # a log line longer than this is cut (paths, the policy dump)
MIN_DRIVER = (32, 0, 101, 9034)        # docs/INTEL_WINDOWS.md "The machine": the loader-1.32 driver, or newer
BMG_G31 = re.compile(r"VEN_8086&DEV_E22[0-9A-F]", re.IGNORECASE)   # E220-E223: the Pro B70/B65 die

# (g) read from nothing: what docs/INTEL_WINDOWS.md ("Known limits and open questions", "Acceptance") and the 2026-10-05
# record still want, as of this file's date
STILL_WANTED = [
    "  1. The idle test (the heartbeat; 'The heartbeat's effect: open'): start the model, send one prompt,",
    "     leave it 15 minutes without a request, send the same prompt again; paste the server window's lines for both",
    "     answers (tok/s, the seconds to the first token) and this report (the log's 'VRAM heartbeat' lines); then the",
    "     A/B with \"env\": {\"STRATA_SYCL_HEARTBEAT_S\": \"0\"} in strata-coder-iq1_m.json.",
    "  2. A clean exit: close the server window (Ctrl+C), then this report again: the log's last lines must show no",
    "     abort at teardown (the heartbeat thread is joined before the runtime goes), and the next start must find the",
    "     card without a Disable-PnpDevice / Enable-PnpDevice.",
    "  3. ctest: from a oneAPI command window, `ctest --test-dir build-sycl-win --output-on-failure` (the parity suite;",
    "     the PR on Linux: 22 pass, one mode refused, three need fixtures, s2_expert_grouped_parity fails).",
    "  4. benchy on Windows (sycl/benchy.sh is Linux-only: docker, sudo, sysfs): the v1 prompts in sycl/bench/v1 with",
    "     --tokens-file, --max-new 256 --greedy --stats under the C4 environment, sizes 20 and 2,185 first; the engine's",
    "     --stats output is what the speed table (docs/INTEL_WINDOWS.md) and 'Submitting numbers' (INTEL_PERFORMANCE.md)",
    "     take.  A benchy_windows script is not written yet.",
    "  5. The greedy tokens of sycl/bench/v1/short.ids against PR #423's Linux build (the same prompt, --greedy).",
    "  6. The Windows graphics driver version and date (section (a) above) into the speed table's driver column.",
]

# (a) one PowerShell call; its last line is JSON.  CIM's DriverDate is a DateTime, formatted here (ConvertTo-Json in
# PowerShell 5.1 would write it as \\/Date(...)\\/); the console encoding is set so the bytes are UTF-8.
PS_MACHINE = r"""
$ErrorActionPreference = 'SilentlyContinue'
[Console]::OutputEncoding = [System.Text.Encoding]::UTF8
$o = [ordered]@{}
$os = Get-CimInstance Win32_OperatingSystem
$o.os = [ordered]@{caption=$os.Caption; version=$os.Version; build=$os.BuildNumber; arch=$os.OSArchitecture;
                   ram_kib=$os.TotalVisibleMemorySize; free_kib=$os.FreePhysicalMemory}
$cv = Get-ItemProperty 'HKLM:\SOFTWARE\Microsoft\Windows NT\CurrentVersion'
$o.release = [ordered]@{display=$cv.DisplayVersion; ubr=$cv.UBR}
$o.cpu = @(Get-CimInstance Win32_Processor | ForEach-Object {
    [ordered]@{name=$_.Name; cores=$_.NumberOfCores; threads=$_.NumberOfLogicalProcessors; mhz=$_.MaxClockSpeed} })
$o.ram_bytes = (Get-CimInstance Win32_ComputerSystem).TotalPhysicalMemory
$o.gpus = @(Get-CimInstance Win32_VideoController | ForEach-Object {
    $d = $null
    if ($_.DriverDate) { $d = $_.DriverDate.ToString('yyyy-MM-dd') }
    [ordered]@{name=$_.Name; pnp=$_.PNPDeviceID; driver=$_.DriverVersion; date=$d; status=$_.Status;
               ram=$_.AdapterRAM; width=$_.CurrentHorizontalResolution} })
$z = Join-Path $env:SystemRoot 'System32\ze_loader.dll'
if (Test-Path $z) {
    $v = (Get-Item $z).VersionInfo
    $o.ze_loader = [ordered]@{path=$z; file=$v.FileVersion; product=$v.ProductVersion}
}
$o | ConvertTo-Json -Depth 5 -Compress
"""


# ---------------------------------------------------------------- helpers
def section(title: str) -> str:
    return f"\n=== {title} ==="


def when(ts: float) -> str:
    return time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(ts))


def gib(n) -> str:
    try:
        return f"{int(n) / 2**30:.1f} GiB"
    except (TypeError, ValueError):
        return "?"


def as_list(x) -> list:
    """ConvertTo-Json writes a one-element array as an array when it is nested, but a caller cannot rely on it."""
    return x if isinstance(x, list) else ([] if x is None else [x])


def clip(line: str) -> str:
    line = line.rstrip()
    return line if len(line) <= LINE_CLIP else line[:LINE_CLIP] + " ..."


def newest(pattern: str):
    """The newest file matching `pattern` next to the checkout, by modification time; None without one."""
    files = [p for p in ROOT.glob(pattern) if p.is_file()]
    return max(files, key=lambda p: p.stat().st_mtime) if files else None


def powershell_json(script: str, timeout: float = 90.0):
    """One PowerShell call whose output ends in a JSON object -> the parsed object; None when PowerShell is missing,
    times out, or the output does not parse (the section then says so and the rest still prints)."""
    try:
        r = subprocess.run(["powershell", "-NoProfile", "-NonInteractive", "-Command", script],
                           capture_output=True, timeout=timeout)
    except (OSError, subprocess.TimeoutExpired):
        return None
    out = r.stdout.decode("utf-8", "replace").strip()
    if "{" not in out:
        return None
    try:
        return json.loads(out[out.index("{"):])
    except ValueError:
        return None


def driver_note(ver) -> str:
    """'' when the Intel driver is at least MIN_DRIVER, else the docs' advice; '' too when the string is not 4 numbers."""
    m = re.match(r"^(\d+)\.(\d+)\.(\d+)\.(\d+)$", str(ver or ""))
    if not m:
        return ""
    if tuple(int(x) for x in m.groups()) >= MIN_DRIVER:
        return ""
    return (" - OLDER than %s: docs/INTEL_WINDOWS.md wants that or newer, or ze_loader.dll 1.32+ beside the exes"
            % ".".join(str(x) for x in MIN_DRIVER))


def psutil_ram():
    """(total, available) bytes from psutil, or None without it."""
    try:
        import psutil
        vm = psutil.virtual_memory()
        return int(vm.total), int(vm.available)
    except (ImportError, OSError, RuntimeError, AttributeError):
        return None


# ---------------------------------------------------------------- (a) the machine
def machine_lines() -> list[str]:
    L = []
    m = powershell_json(PS_MACHINE) if WIN else None
    if m is None:
        L.append("  (PowerShell's CIM query did not answer; Python's view only)")
        L.append(f"  OS: {platform.platform()}")
        L.append(f"  CPU: {platform.processor() or '?'}, {os.cpu_count()} logical processors")
        ram = psutil_ram()
        L.append("  RAM: " + (f"{gib(ram[0])} physical, {gib(ram[1])} available now" if ram else "? (no psutil)"))
        L.append("  display adapters: unknown (run `Get-CimInstance Win32_VideoController` by hand)")
    else:
        o, rel = m.get("os") or {}, m.get("release") or {}
        L.append(f"  Windows: {o.get('caption') or '?'} {rel.get('display') or ''} build {o.get('build') or '?'}"
                 f".{rel.get('ubr') if rel.get('ubr') is not None else '?'} ({o.get('version') or '?'}, "
                 f"{o.get('arch') or '?'})")
        for c in as_list(m.get("cpu")):
            L.append(f"  CPU: {c.get('name') or '?'} - {c.get('cores')} cores, {c.get('threads')} threads, "
                     f"{c.get('mhz')} MHz max")
        free = o.get("free_kib")
        L.append(f"  RAM: {gib(m.get('ram_bytes'))} physical"
                 + (f", {gib(int(free) * 1024)} free now" if free not in (None, "") else ""))
        gpus = as_list(m.get("gpus"))
        bmg = [g for g in gpus if BMG_G31.search(str(g.get("pnp") or ""))]
        nvidia = [g for g in gpus if "VEN_10DE" in str(g.get("pnp") or "").upper()]
        intel_other = [g for g in gpus if "VEN_8086" in str(g.get("pnp") or "").upper() and g not in bmg]
        if bmg:
            for g in bmg:
                dev = re.search(r"DEV_([0-9A-F]{4})", str(g.get("pnp")), re.IGNORECASE)
                L.append(f"  Intel Arc Pro B70/B65 (BMG-G31, DEV_{dev.group(1).upper() if dev else '?'}): "
                         f"{g.get('name') or '?'}; driver {g.get('driver') or '?'} dated {g.get('date') or '?'}; "
                         f"status {g.get('status') or '?'}{driver_note(g.get('driver'))}")
        else:
            L.append("  Intel Arc Pro B70/B65: NO adapter with PCI\\VEN_8086&DEV_E22x among the display adapters")
        L.append("  NVIDIA adapter: " + (", ".join(f"{g.get('name')} (driver {g.get('driver')})" for g in nvidia)
                                         if nvidia else "none"))
        for g in intel_other:
            L.append(f"  other Intel adapter: {g.get('name')} ({g.get('pnp')}); driver {g.get('driver')} dated "
                     f"{g.get('date') or '?'}")
        L.append("  all display adapters:")
        for g in gpus:
            L.append(f"    {g.get('name') or '?'} | {g.get('pnp') or '?'} | driver {g.get('driver') or '?'} "
                     f"({g.get('date') or '?'}) | status {g.get('status') or '?'}"
                     + (f" | {g.get('width')} px wide (drives a display)" if g.get("width") else ""))
        ze = m.get("ze_loader")
        if ze:
            L.append(f"  Level Zero loader: {ze.get('path')} file {ze.get('file')}, product {ze.get('product')}")
        else:
            L.append("  Level Zero loader: no System32\\ze_loader.dll (then the engine folder's copy, if any: (b))")
        ram = psutil_ram()
        if ram:
            L.append(f"  psutil: {gib(ram[0])} total, {gib(ram[1])} available now")
    L.append(f"  Python: {sys.version.split()[0]} at {sys.executable}")
    L.append(f"  ONEAPI_ROOT: {os.environ.get('ONEAPI_ROOT') or '(unset: the engine runs on the DLLs beside it)'}")
    return L


# ---------------------------------------------------------------- (b) the build
def build_lines() -> list[str]:
    L = []
    bj = ENGINE / "BUILD.json"
    if bj.is_file():
        L.append(f"  {bj}:")
        try:
            L += ["    " + x for x in bj.read_text(encoding="utf-8-sig").strip().splitlines()]
        except OSError as e:
            L.append(f"    cannot read: {e}")
    else:
        L.append(f"  {bj}: missing (sycl\\tools\\build_windows.bat writes it)")
    if not ENGINE.is_dir():
        L.append(f"  {ENGINE}: missing")
        return L
    files = sorted(p for p in ENGINE.iterdir() if p.is_file())
    exes = [p for p in files if p.suffix.lower() == ".exe"]
    dlls = [p for p in files if p.suffix.lower() == ".dll"]
    bundled = any(p.name.lower() == "ze_loader.dll" for p in dlls)
    L.append(f"  engine\\: {len(exes)} exe ({', '.join(p.name for p in exes)}), {len(dlls)} DLLs"
             + ("; ze_loader.dll bundled" if bundled else "; no bundled ze_loader.dll (the driver's in System32 is used)"))
    strata = ENGINE / "strata.exe"
    if strata.is_file():
        st = strata.stat()
        L.append(f"  strata.exe: {st.st_size:,} bytes, modified {when(st.st_mtime)}")
    L.append("  DLLs: " + (", ".join(p.name for p in dlls) or "none"))
    return L


# ---------------------------------------------------------------- (c) the config and the run script
def path_note(v) -> str:
    """For an argument that looks like a path: its size, [folder], or [MISSING]; '' for anything else."""
    if not isinstance(v, str) or not ("\\" in v or "/" in v):
        return ""
    p = Path(v)
    if not p.exists() and ":" in v[2:]:                 # --control-vector-scaled <path>:<scale>
        p = Path(v[:v.rfind(":")])
    if p.is_file():
        n = p.stat().st_size
        return f"   [{n / 1e9:.1f} GB]" if n >= 1e8 else f"   [{n:,} B]"
    if p.is_dir():
        return "   [folder]"
    return "   [MISSING]"


def config_lines(cfg_path) -> tuple[list[str], dict]:
    L = []
    if cfg_path is None:
        L.append("  no strata-*.json next to the checkout (sycl\\START-HERE.bat --backend sycl writes one)")
        return L, {}
    try:
        cfg = json.loads(Path(cfg_path).read_text(encoding="utf-8-sig"))
    except (OSError, ValueError) as e:
        L.append(f"  {cfg_path}: cannot read ({e})")
        return L, {}
    cfg_path = Path(cfg_path)
    L.append(f"  {cfg_path} (modified {when(cfg_path.stat().st_mtime)}):")
    for k in ("backend", "exe", "gpu", "lib_dirs", "env", "cwd", "log", "port", "model_name", "tokenizer", "sampling",
              "layer_split", "engine_silence_s", "model_switcher"):
        if k in cfg:
            v = cfg[k]
            L.append(f"    {k}: {v if isinstance(v, str) else json.dumps(v)}"
                     + (path_note(v) if k in ("exe", "tokenizer") else ""))
    L.append("    args:")
    args = [str(x) for x in (cfg.get("args") or [])]
    i = 0
    while i < len(args):
        a = args[i]
        if a.startswith("--") and i + 1 < len(args) and not args[i + 1].startswith("--"):
            L.append(f"      {a} {args[i + 1]}{path_note(args[i + 1])}")
            i += 2
        else:
            L.append(f"      {a}")
            i += 1
    script = ROOT / f"run-{cfg_path.stem[len('strata-'):]}.bat"
    if script.is_file():
        L.append(f"  {script.name}:")
        L += ["    " + x for x in script.read_text(encoding="utf-8-sig").splitlines() if x.strip()]
    else:
        L.append(f"  {script.name}: missing")
    return L, cfg


# ---------------------------------------------------------------- (d) and (e) the log
def find_log(arg_log, cfg: dict):
    if arg_log:
        return Path(arg_log)
    if cfg.get("log") and Path(str(cfg["log"])).is_file():
        return Path(str(cfg["log"]))
    return newest("strata-*.log")


def log_lines(path, tail_n: int, since: int) -> tuple[list[str], list[str], list[str]]:
    """(the log's facts and the key-line summary, the key lines, the tail): every line matching KEY_RE from line
    `since` on, with its 1-based line number, at most KEY_CAP of them (the first and the last half, with the count
    dropped between: the start's lines and the latest requests both survive)."""
    if path is None:
        return ["  no engine log (strata-*.log next to the checkout, or --log PATH)"], [], ["  (no log)"]
    try:
        data = path.read_bytes()
    except OSError as e:
        return [f"  {path}: cannot read ({e})"], [], ["  (no log)"]
    lines = data.decode("utf-8", "replace").splitlines()
    st = path.stat()
    head = [f"  {path} - {st.st_size:,} bytes, {len(lines):,} lines, modified {when(st.st_mtime)}"]
    starts = [i + 1 for i, x in enumerate(lines) if START_MARK in x]
    head.append(f"  engine starts seen ('{START_MARK}' lines): {len(starts)}"
                + (f", at line{'s' if len(starts) > 1 else ''} {', '.join(str(n) for n in starts)}" if starts else "")
                + "; every start appends to the same file (--since LINE narrows the key lines)")
    first = max(1, since)
    matches = [(n, x) for n, x in enumerate(lines, start=1) if n >= first and KEY_RE.search(x)]
    if len(matches) > KEY_CAP:
        half = KEY_CAP // 2
        shown = matches[:half] + [None] + matches[-half:]
        head.append(f"  {len(matches):,} key lines from line {first}: the first {half} and the last {half}; "
                    f"{len(matches) - KEY_CAP:,} dropped between them")
    else:
        shown = matches
        head.append(f"  {len(matches)} key lines from line {first}")
    key = ["  ..." if m is None else f"  {m[0]:>7}: {clip(m[1])}" for m in shown]
    tail_lines = lines[-tail_n:] if tail_n > 0 else []
    start_n = len(lines) - len(tail_lines) + 1
    tail = [f"  {n:>7}: {clip(x)}" for n, x in enumerate(tail_lines, start=start_n)] or ["  (empty log)"]
    return head, key, tail


# ---------------------------------------------------------------- (f) the card now
def sysman_lines(index: int) -> list[str]:
    """VRAM used/free, power, temperature and load from Level Zero Sysman (sycl/serve/xe_telemetry.py's Windows
    reader, the Monitor tab's source), two samples a second apart for the rates; 'not available' with the reason
    when the loader or the reader does not answer.  Whatever is on the card now counts, the engine's or not."""
    if not WIN:
        return ["  not available (not Windows)"]
    os.environ.setdefault("ZES_ENABLE_SYSMAN", "1")    # the server's value, for a loader that needs it before init
    sys.path.insert(0, str(SERVE))
    try:
        import xe_telemetry as XE
    except Exception as e:  # noqa: BLE001 - a broken import must not take the report down
        return [f"  not available (sycl/serve/xe_telemetry.py did not import: {e})"]
    try:
        g = XE.xe_gpu(index)
        if not g.ok():
            return [f"  not available (no Level Zero loader answered, or no GPU at Sysman index {index}: "
                    "System32\\ze_loader.dll, else one on the PATH)"]
        g.read()
        time.sleep(1.0)
        r = g.read()
    except Exception as e:  # noqa: BLE001
        return [f"  not available (the reader failed: {e})"]
    L = [f"  device {index}: {g.name()}"]
    used, total = r.get("mem_used"), r.get("mem_total")
    if used is not None and total:
        L.append(f"  VRAM: {gib(used)} used, {gib(int(total) - int(used))} free, {gib(total)} total")
    else:
        L.append("  VRAM: not reported")
    L.append("  power: " + (f"{r['power']:.1f} W" if r.get("power") is not None else "not reported")
             + (f" (limit {r['power_limit']:.0f} W)" if r.get("power_limit") else ""))
    L.append("  temperature: " + (f"{r['temp']:.0f} C" if r.get("temp") is not None else "not reported")
             + (f", memory {r['vram_temp']:.0f} C" if r.get("vram_temp") is not None else ""))
    L.append("  load: " + (f"{r['util']:.0f}%" if r.get("util") is not None else "not reported"))
    return L


# ---------------------------------------------------------------- main
def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description="one paste after a model run on the Windows B70: the machine, the build, "
                                             "the config, the engine log's key lines and tail, the card now")
    ap.add_argument("--log", help="the engine log (default: the config's \"log\", else the newest strata-*.log)")
    ap.add_argument("--config", help="the run config (default: the newest strata-*.json next to the checkout)")
    ap.add_argument("--tail", type=int, default=60, help="how many of the log's last lines (default 60)")
    ap.add_argument("--since", type=int, default=1, help="scan the log's key lines from this line number (default 1)")
    ap.add_argument("--out", help="where to write the report (default sycl\\benchy-results\\report-<timestamp>.txt)")
    a = ap.parse_args(argv)
    try:
        sys.stdout.reconfigure(errors="replace")         # a log line the console's code page cannot show
    except (AttributeError, ValueError):
        pass
    out = Path(a.out) if a.out else OUT_DIR / f"report-{time.strftime('%Y%m%d-%H%M%S')}.txt"

    R = [f"Strata report_windows.py (sycl/tools) - {time.strftime('%Y-%m-%d %H:%M:%S')} - {ROOT}"]
    R.append(section("(a) the machine"))
    R += machine_lines()
    R.append(section("(b) the engine build (engine\\BUILD.json)"))
    R += build_lines()
    R.append(section("(c) the run config and its script"))
    cfg_lines, cfg = config_lines(Path(a.config) if a.config else newest("strata-*.json"))
    R += cfg_lines
    head, key, tail = log_lines(find_log(a.log, cfg), a.tail, a.since)
    R.append(section("(d) the engine log's key lines"))
    R += head + key
    R.append(section(f"(e) the engine log's last {a.tail} lines"))
    R += tail
    gpu = cfg.get("gpu")
    if isinstance(gpu, list):                             # a layer split names several; the first one
        gpu = gpu[0] if gpu else 0
    R.append(section("(f) the card now (Level Zero Sysman)"))
    R += sysman_lines(int(gpu) if isinstance(gpu, int) and not isinstance(gpu, bool) else 0)
    R.append(section("(g) still to measure (docs/INTEL_WINDOWS.md)"))
    R += STILL_WANTED
    text = "\n".join(R) + "\n"
    print(text)
    try:
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(text, encoding="utf-8")
        print(f"written to {out}")
    except OSError as e:
        print(f"not written to {out}: {e}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
