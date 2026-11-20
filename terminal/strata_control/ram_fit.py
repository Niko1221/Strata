"""strata_control/ram_fit.py - the RAM side of a start on a small-RAM PC (about 32 GB), the Sync-Strata way.

The engine keeps a page-locked copy of the most-used experts in RAM (--resident-budget-gib) and clamps that copy to
what it finds free at its start - the less of the physical RAM and the Windows commit limit (RAM plus the page file),
not just the total (src/core/expert_source.cpp, clamp_resident_budget).  What the Terminal can still add on top:

- a warning before the copy when the commit limit will cut the budget, with what gives the room back (the page file
  set to "system managed"; the programs that hold the most commit now) - a start is never refused;
- one memory squeeze before the engine copies: the other programs of this session are asked to move their pages to
  the page file (EmptyWorkingSet, then a short lock of the free RAM so Windows writes them out).  No program is
  closed and every page comes back when its program uses it; the freed RAM is what the expert copy takes.

Both helpers are advisory and best effort: every entry point catches its own errors and a start never depends on
them running.  Standard library only.  The squeeze stages follow sync-strata's launcher (zapabob/sync-strata,
app/launcher/ramsqueeze.py, MIT) with this fork's process names.
"""
from __future__ import annotations

import ctypes
import json
import os
import subprocess
import sys
import time

IS_WIN = os.name == "nt"
GiB = 1024 ** 3

#: What the engine commits beside the expert copy before it starts copying (dense parts, CUDA context): the
#: Sync-Strata measurement on a 32 GB PC (their launcher's ENGINE_COMMIT_GIB).
ENGINE_COMMIT_GIB = 2.0
#: The engine's clamp leaves this margin under the room it grants (src/core/expert_source.cpp: 256 MiB).
ENGINE_MARGIN_GIB = 0.25
#: The engine's default copy headroom (src/program/generate.cpp: 4 GiB; STRATA_RESIDENT_HEADROOM_GIB, when set in the
#: environment the engine gets, decides instead).
DEFAULT_HEADROOM_GIB = 4.0
#: How much RAM the other programs may keep during the squeeze (the lock stops at this much left for them).
SQUEEZE_TARGET_OTHERS_GIB = 3.0

# Stage 1 leaves these alone (lower case, no .exe): the processes Windows needs, and this fork's own ones
# (the engine and the app shell are in the own tree anyway; the names are a second net).
_EXCLUDE_NAMES = {
    "system", "registry", "smss", "csrss", "wininit", "winlogon", "services", "lsass", "lsm", "svchost", "fontdrvhost",
    "dwm", "sihost", "ctfmon", "audiodg", "explorer", "searchhost", "startmenuexperiencehost", "shellexperiencehost",
    "textinputhost", "securityhealthsystray", "memory compression", "vmmem", "vmmemwsl", "nvcontainer", "nvidia overlay",
    "strata", "strata-vision", "python", "py",
}


def _own_markers() -> list[str]:
    """Path fragments that mean "this fork's own install or checkout": its processes are never trimmed."""
    markers = []
    here = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))    # strata_control's parent (ROOT)
    if here:
        markers.append(os.path.abspath(here).lower())
    markers.append("strata")
    return [m for m in markers if m]


def mem_gib():
    """(total, available, commit_free) in GiB, or None when unreadable.  Windows: GlobalMemoryStatusEx - the same
    view the engine and model_memory.py use, where "available" counts the standby cache.  Linux: /proc/meminfo's
    MemAvailable (the file cache is in there, so the commit idea does not apply)."""
    if IS_WIN:
        class _MS(ctypes.Structure):
            _fields_ = [("dwLength", ctypes.c_ulong), ("dwMemoryLoad", ctypes.c_ulong),
                        ("ullTotalPhys", ctypes.c_ulonglong), ("ullAvailPhys", ctypes.c_ulonglong),
                        ("ullTotalPageFile", ctypes.c_ulonglong), ("ullAvailPageFile", ctypes.c_ulonglong),
                        ("ullTotalVirtual", ctypes.c_ulonglong), ("ullAvailVirtual", ctypes.c_ulonglong),
                        ("ullAvailExtendedVirtual", ctypes.c_ulonglong)]
        m = _MS()
        m.dwLength = ctypes.sizeof(_MS)
        if not ctypes.windll.kernel32.GlobalMemoryStatusEx(ctypes.byref(m)):
            return None
        return {"total_gib": m.ullTotalPhys / GiB, "avail_gib": m.ullAvailPhys / GiB,
                "commit_free_gib": m.ullAvailPageFile / GiB}
    if sys.platform == "linux":
        try:
            kv = {}
            with open("/proc/meminfo", encoding="ascii", errors="replace") as f:
                for line in f:
                    k, _, v = line.partition(":")
                    kv[k] = int(v.split()[0]) * 1024
            return {"total_gib": kv.get("MemTotal", 0) / GiB,
                    "avail_gib": kv.get("MemAvailable", kv.get("MemFree", 0)) / GiB, "commit_free_gib": None}
        except (OSError, ValueError):
            return None
    return None


def _headroom_gib() -> float:
    """The copy headroom the engine will use (its env decides; the server hands the engine its own env)."""
    raw = os.environ.get("STRATA_RESIDENT_HEADROOM_GIB")
    if raw is None:
        return DEFAULT_HEADROOM_GIB
    try:
        return max(0.0, float(raw))
    except ValueError:
        return DEFAULT_HEADROOM_GIB


def commit_room_gib(commit_free_gib=None) -> float | None:
    """What the engine's clamp can grant from the commit limit, in GiB (None: the commit limit is unknown, so the
    Terminal does not estimate).  The engine grants min(physical, commit) less the headroom, less a 0.25 GiB margin;
    here the engine's own pre-copy commit is taken off the commit side first (the physical side stays the caller's
    business: the free-RAM warning covers it).  Rounded down to 0.25 GiB, like the launcher of sync-strata does."""
    if commit_free_gib is None:
        return None
    room = max(0.0, float(commit_free_gib) - ENGINE_COMMIT_GIB - _headroom_gib() - ENGINE_MARGIN_GIB)
    return int(room * 4) / 4.0


def top_commit_apps(n=3, min_gib=0.3):
    """The programs using the most commit now (Windows: PrivateMemorySize64 grouped by name), so the advice names
    what closing would give back.  [] when they cannot be read."""
    if not IS_WIN:
        return []
    try:
        out = subprocess.run(
            ["powershell", "-NoProfile", "-NonInteractive", "-Command",
             "$ErrorActionPreference='SilentlyContinue';"
             "Get-Process | Group-Object Name | ForEach-Object { @{ name=$_.Name; gib=[math]::Round(($_.Group | "
             "Measure-Object PrivateMemorySize64 -Sum).Sum / 1GB, 2) } } | ConvertTo-Json -Compress"],
            capture_output=True, timeout=30, creationflags=0x08000000)
        rows = json.loads(out.stdout.decode("utf-8", "replace") or "[]")
    except (OSError, ValueError, subprocess.SubprocessError):
        return []
    if isinstance(rows, dict):
        rows = [rows]
    apps = [(str(r.get("name") or ""), float(r.get("gib") or 0.0)) for r in rows if isinstance(r, dict)]
    apps = [(nm, g) for nm, g in apps if nm.lower() not in _EXCLUDE_NAMES and g >= min_gib]
    return sorted(apps, key=lambda x: -x[1])[:n]


def commit_cut_notes(budget_gib: float, mem: dict | None = None) -> list[str]:
    """The advisory lines for a RAM budget the Windows commit limit will cut ([] when it does not, or when the
    commit limit is unknown).  The engine clamps the same way at its start; these lines say so before it does."""
    if mem is None:
        mem = mem_gib()
    if not mem:
        return []
    cfree = mem.get("commit_free_gib")
    if cfree is None:
        return []
    room = commit_room_gib(cfree)
    if room is None or float(budget_gib) <= room:
        return []
    lines = [f"the Windows commit limit (RAM plus the page file) has about {cfree:.0f} GiB free: the engine will keep "
             f"only about {room:g} GiB of the {budget_gib:g} GiB RAM budget in RAM and read the rest from the model "
             "files"]
    apps = top_commit_apps()
    lines.append("setting the page file to \"system managed\" gives that room back"
                 + (", as does closing " + ", ".join(f"{nm} {g:.1f} GiB" for nm, g in apps) if apps else ""))
    return lines


def budget_from_args(args) -> float | None:
    """The RAM budget a start asks for (--resident-budget-gib N), or None: the squeeze's "auto" only runs for one."""
    try:
        flat = list(args)
        return float(flat[flat.index("--resident-budget-gib") + 1])
    except (ValueError, IndexError, TypeError):
        return None


# ---------------------------------------------------------------- the squeeze (Windows)


def _procs_ps():
    """The processes of this log-in session (name, path, command line, parent, working set), one PowerShell call."""
    ps = ("$ErrorActionPreference='SilentlyContinue';"
          "$me=(Get-Process -Id $PID).SessionId;"
          "Get-CimInstance Win32_Process | Where-Object { $_.SessionId -eq $me } | "
          "ForEach-Object { @{ pid=$_.ProcessId; ppid=$_.ParentProcessId; name=$_.Name; exe=$_.ExecutablePath; "
          "cmd=$_.CommandLine; ws=$_.WorkingSetSize } } | ConvertTo-Json -Compress")
    out = subprocess.run(["powershell", "-NoProfile", "-NonInteractive", "-Command", ps],
                         capture_output=True, timeout=30, creationflags=0x08000000).stdout.decode("utf-8", "replace").strip()
    d = json.loads(out) if out else []
    return d if isinstance(d, list) else [d]


def _ancestors(procs, pid):
    by = {p["pid"]: p for p in procs}
    seen, cur = set(), pid
    while cur in by and cur not in seen:
        seen.add(cur)
        cur = by[cur].get("ppid")
    return seen


def _excluded(p, own_pids, markers) -> bool:
    name = (p.get("name") or "").lower()
    name = name[:-4] if name.endswith(".exe") else name
    if p["pid"] in own_pids or name in _EXCLUDE_NAMES:
        return True
    hay = ((p.get("exe") or "") + " " + (p.get("cmd") or "")).lower()
    return any(m in hay for m in markers)


def trim_working_sets():
    """Stage 1: EmptyWorkingSet on the session's other programs.  Returns (trimmed count, working set trimmed, GiB)."""
    k32 = ctypes.WinDLL("kernel32", use_last_error=True)
    psapi = ctypes.WinDLL("psapi", use_last_error=True)
    k32.OpenProcess.restype = ctypes.c_void_p
    k32.OpenProcess.argtypes = [ctypes.c_ulong, ctypes.c_int, ctypes.c_ulong]
    k32.CloseHandle.argtypes = [ctypes.c_void_p]
    psapi.EmptyWorkingSet.argtypes = [ctypes.c_void_p]
    procs = _procs_ps()
    me = os.getpid()
    own = _ancestors(procs, me) | {me}
    markers = _own_markers()
    n, ws = 0, 0
    for p in procs:
        if not isinstance(p, dict) or _excluded(p, own, markers):
            continue
        h = k32.OpenProcess(0x0100 | 0x1000, 0, int(p["pid"]))   # PROCESS_SET_QUOTA | PROCESS_QUERY_LIMITED_INFORMATION
        if not h:
            continue
        try:
            if psapi.EmptyWorkingSet(h):
                n += 1
                ws += int(p.get("ws") or 0)
        finally:
            k32.CloseHandle(h)
    return n, ws / GiB


class _MS(ctypes.Structure):
    _fields_ = [("dwLength", ctypes.c_ulong), ("dwMemoryLoad", ctypes.c_ulong),
                ("ullTotalPhys", ctypes.c_ulonglong), ("ullAvailPhys", ctypes.c_ulonglong),
                ("ullTotalPageFile", ctypes.c_ulonglong), ("ullAvailPageFile", ctypes.c_ulonglong),
                ("ullTotalVirtual", ctypes.c_ulonglong), ("ullAvailVirtual", ctypes.c_ulonglong),
                ("ullAvailExtendedVirtual", ctypes.c_ulonglong)]


def _ms():
    m = _MS()
    m.dwLength = ctypes.sizeof(_MS)
    ctypes.windll.kernel32.GlobalMemoryStatusEx(ctypes.byref(m))
    return m


def lock_squeeze(target_others_gib=SQUEEZE_TARGET_OTHERS_GIB, time_limit=30.0, slow_s=4.0, chunk_mib=256, hold_s=1.0):
    """Stage 2: lock the free RAM in 256 MiB blocks for a moment, so Windows writes the other programs' pages to the
    page file, then let go all at once.  Returns (locked GiB, why it stopped).  The commit side keeps 4 GiB free: the
    lock consumes commit too.  The process's working-set limits are read first and always restored (a minimum left
    high makes Windows reserve it, and a later engine pin is refused)."""
    k32 = ctypes.WinDLL("kernel32", use_last_error=True)
    k32.VirtualAlloc.restype = ctypes.c_void_p
    k32.VirtualAlloc.argtypes = [ctypes.c_void_p, ctypes.c_size_t, ctypes.c_ulong, ctypes.c_ulong]
    k32.VirtualLock.argtypes = [ctypes.c_void_p, ctypes.c_size_t]
    k32.VirtualUnlock.argtypes = [ctypes.c_void_p, ctypes.c_size_t]
    k32.VirtualFree.argtypes = [ctypes.c_void_p, ctypes.c_size_t, ctypes.c_ulong]
    k32.GetCurrentProcess.restype = ctypes.c_void_p
    k32.SetProcessWorkingSetSizeEx.argtypes = [ctypes.c_void_p, ctypes.c_size_t, ctypes.c_size_t, ctypes.c_ulong]
    k32.GetProcessWorkingSetSizeEx.argtypes = [ctypes.c_void_p, ctypes.POINTER(ctypes.c_size_t),
                                               ctypes.POINTER(ctypes.c_size_t), ctypes.POINTER(ctypes.c_ulong)]
    proc = k32.GetCurrentProcess()
    o_mn, o_mx, o_fl = ctypes.c_size_t(0), ctypes.c_size_t(0), ctypes.c_ulong(0)
    if not k32.GetProcessWorkingSetSizeEx(proc, ctypes.byref(o_mn), ctypes.byref(o_mx), ctypes.byref(o_fl)):
        o_mn, o_mx = ctypes.c_size_t(1 * 1024 ** 2), ctypes.c_size_t(64 * 1024 ** 2)
    chunk = chunk_mib * 1024 ** 2
    m0 = _ms()
    total = m0.ullTotalPhys
    commit_cap = max(0, m0.ullAvailPageFile - 4 * GiB)
    want = min(commit_cap, max(0, total - int(target_others_gib * GiB)))
    blocks, locked, why, t0 = [], 0, "RAM was enough", time.time()
    try:
        while locked + chunk <= want:
            if time.time() - t0 > time_limit:
                why = "the time limit"
                break
            if total - locked - _ms().ullAvailPhys <= target_others_gib * GiB:
                why = "the target"
                break
            tc = time.time()
            if not k32.SetProcessWorkingSetSizeEx(proc, locked + chunk + 256 * 1024 ** 2,
                                                  locked + chunk + 512 * 1024 ** 2, 0):
                why = f"the working-set minimum was refused ({ctypes.get_last_error()})"
                break
            p = k32.VirtualAlloc(None, chunk, 0x3000, 0x04)
            if not p:
                why = f"VirtualAlloc failed ({ctypes.get_last_error()})"
                break
            blocks.append(p)
            if not k32.VirtualLock(p, chunk):
                why = f"VirtualLock failed ({ctypes.get_last_error()})"
                break
            locked += chunk
            if time.time() - tc > slow_s:
                why = f"one block took {time.time() - tc:.1f} s"
                break
        time.sleep(hold_s)
    finally:
        for p in blocks:
            k32.VirtualUnlock(p, chunk)
            k32.VirtualFree(p, 0, 0x8000)
        k32.SetProcessWorkingSetSizeEx(proc, ctypes.c_size_t(-1).value, ctypes.c_size_t(-1).value, 0)
        if not k32.SetProcessWorkingSetSizeEx(proc, o_mn.value, o_mx.value, 0):   # the read-back limits, always restored
            k32.SetProcessWorkingSetSizeEx(proc, 1 * 1024 ** 2, 64 * 1024 ** 2, 0)
    return locked / GiB, why


# ---------------------------------------------------------------- the entry point


def squeeze(mode=None, budget_gib=None):
    """One squeeze before the engine copies its experts.  STRATA_RAM_SQUEEZE decides: "off" never, "on" always,
    "auto" (the default) only for a start that asks for a RAM budget (the freed RAM is what the copy needs).
    Returns {"ran", "mode", "before_gib", "after_gib", "freed_gib", "detail"} and never raises: a squeeze that
    cannot run or frees nothing stays quiet ("" detail) so starts stay readable."""
    mode = (mode if mode is not None else os.environ.get("STRATA_RAM_SQUEEZE") or "auto").strip().lower()
    out = {"ran": False, "mode": mode or "auto", "before_gib": None, "after_gib": None, "freed_gib": 0.0, "detail": ""}
    if mode == "off" or (mode not in ("on", "auto")):
        return out
    if mode == "auto" and budget_gib is None:
        return out
    fake = os.environ.get("STRATA_FAKE_SQUEEZE")
    t0 = time.time()
    if fake:
        try:
            before, after = (float(x) for x in fake.split(":"))
        except ValueError:
            return out
        out.update(ran=True, before_gib=before, after_gib=after, freed_gib=round(max(0.0, after - before), 2),
                   detail="test fake (STRATA_FAKE_SQUEEZE)")
    elif IS_WIN:
        try:
            m0 = mem_gib()
            if not m0:
                return out
            before = m0["avail_gib"]
            n, _ws = trim_working_sets()
            _locked, why = lock_squeeze()
            m1 = mem_gib() or m0
            after = m1["avail_gib"]
            out.update(ran=True, before_gib=round(before, 2), after_gib=round(after, 2),
                       freed_gib=round(max(0.0, after - before), 2),
                       detail=f"trimmed {n} programs' working sets and locked {_locked:.1f} GiB to push pages out "
                              f"(stopped at {why}; nothing was closed)")
        except Exception as e:                                   # noqa: BLE001 - the squeeze is an aid, never a gate
            out["detail"] = f"the RAM squeeze could not run ({type(e).__name__}: {e}); starting anyway"
            return out
    else:
        try:                                                     # Linux: only root may drop the file cache
            path = "/proc/sys/vm/drop_caches"
            if not os.access(path, os.W_OK):
                return out
            m0 = mem_gib()
            os.sync()
            with open(path, "w", encoding="ascii") as f:
                f.write("1\n")
            m1 = mem_gib() or m0
            if m0:
                out.update(ran=True, before_gib=round(m0["avail_gib"], 2), after_gib=round(m1["avail_gib"], 2),
                           freed_gib=round(max(0.0, m1["avail_gib"] - m0["avail_gib"]), 2),
                           detail="dropped the file cache (root)")
        except OSError as e:
            out["detail"] = f"the RAM squeeze could not run ({type(e).__name__}: {e}); starting anyway"
            return out
    out["seconds"] = round(time.time() - t0, 1)
    if out["freed_gib"] < 0.1:
        out["detail"] = ""          # nothing came out: stay quiet instead of printing a line on every start
    return out
