"""Intel Arc (xe driver) readings for the server's Monitor tab, in the shape of serve/telemetry.py's readers.

Kept out of serve/telemetry.py so the SYCL port touches no shared file: sycl/serve/server_intel.py installs it as
telemetry.gpu_reader when there is no NVIDIA card.  Load and VRAM need root (xe's only VRAM accounting is per-client
fdinfo), so they come from a root sampler's /run/gpustat.json (sycl/tools/gpustat.py); temperature, power and the
PCIe link are read from sysfs, so they show without it.
"""
from __future__ import annotations

import ctypes
import os
import re
import subprocess
import sys
import time

# -------------------------------------------------------------------------------------- Windows: the OS's own GPU counters
# xe exposes nothing here: no sysfs, no user-mode Level Zero adapter, so the Monitor tab had a name and a VRAM size
# and nothing else.  Windows does keep counters for every WDDM adapter, and two of them answer the tiles that were
# empty: pdh.dll's "GPU Engine" (per process, per engine, summed per card as Task Manager does) and "GPU Adapter
# Memory" (dedicated bytes in use).  Measured on an Arc Pro B70: 84-100% while the engine decoded a 1,200-token answer
# and 30.01 of 32.00 GiB of dedicated memory in use - the 23.4 GiB of experts, the KV and the driver's own buffers.
# Temperature, power and the PCIe link have no Windows counter at all (only the OS knows), so those tiles stay empty.
#
# PDH's wildcard API answers PDH_INVALID_ARGUMENT on this machine for every path, core counters included
# (PdhExpandWildCardPathW and PdhGetFormattedCounterArrayW), so the instance list comes from `Get-Counter -ListSet`
# and every instance gets its own counter.  Sampling is then pure PDH: 3.8 ms to collect the 392 instances of the two
# sets on this card and 0.4 ms to read them, against a PowerShell call per sample.  An instance belongs to the process
# that made it, so the list is rebuilt when the card reads idle - a process that started since the last look has none
# yet - and at most every BIND_MIN_S, which is the cost of one `Get-Counter -ListSet` (~100 ms warm).
PDH_FMT_DOUBLE, PDH_FMT_LARGE, PDH_FMT_NOCAP100 = 0x200, 0x400, 0x8000
PDH_CSTATUS_NO_OBJECT, PDH_INVALID_DATA = 0xC0000BB8, 0xC0000BC6
BIND_MIN_S = 30.0


class _PdhValue(ctypes.Union):             # PDH_FMT_COUNTERVALUE_FMT_VALUE
    _fields_ = [("longValue", ctypes.c_long), ("doubleValue", ctypes.c_double), ("largeValue", ctypes.c_longlong)]


class _PdhSingle(ctypes.Structure):        # PDH_FMT_COUNTERVALUE
    _fields_ = [("status", ctypes.c_ulong), ("value", _PdhValue)]


def _counter_instances(counter_set):
    """The instance paths of a Windows counter set, from `Get-Counter -ListSet`."""
    try:
        r = subprocess.run(["powershell", "-NoProfile", "-NonInteractive", "-Command",
                            "(Get-Counter -ListSet %r).PathsWithInstances" % counter_set],
                           capture_output=True, timeout=60)
    except (OSError, subprocess.SubprocessError):
        return []
    return [line.strip() for line in r.stdout.decode(errors="replace").splitlines() if line.strip()]


class _PdhSet:
    """One counter set read through PDH, instance by instance.  `large` asks for the 64-bit integer form (byte counts)
    rather than a double (percentages): PDH fills the union to match the format, so a double's bits read as an integer
    are nonsense.  Instances come and go with the processes using the GPU, and a counter whose instance has gone is
    dropped after three samples."""

    def __init__(self, lib, query, counter_set, suffix, large=False, luid=None, cap=1024):
        self.lib, self.query, self.counter_set = lib, query, counter_set
        self.suffix, self.large, self.luid, self.cap = suffix, large, luid, cap
        self.fmt = (PDH_FMT_LARGE if large else PDH_FMT_DOUBLE) | PDH_FMT_NOCAP100
        self.counters = {}                  # path -> [handle, samples its instance has been gone]

    def refresh(self):
        added = 0
        for path in _counter_instances(self.counter_set):
            if self.suffix not in path or (self.luid and self.luid not in path):
                continue
            if path in self.counters or len(self.counters) >= self.cap:
                continue
            counter = ctypes.c_void_p()
            if self.lib.PdhAddCounterW(self.query, ctypes.c_wchar_p(path), 0, ctypes.byref(counter)) == 0:
                self.counters[path] = [counter, 0]
                added += 1
        return added

    def values(self):
        if not self.counters:
            return ()
        self.lib.PdhCollectQueryData(self.query)
        out = []
        for path, entry in list(self.counters.items()):
            v = _PdhSingle()
            rc = self.lib.PdhGetFormattedCounterValue(entry[0], self.fmt, None, ctypes.byref(v))
            gone = rc == PDH_CSTATUS_NO_OBJECT or v.status == PDH_CSTATUS_NO_OBJECT
            if rc == 0 and (v.status == 0 or v.status == PDH_INVALID_DATA):
                out.append(float(v.value.largeValue if self.large else v.value.doubleValue))
                entry[1] = 0
            elif gone:
                entry[1] += 1
                if entry[1] >= 3:
                    del self.counters[path]
        return out


class _WinCounters:
    """The one card's load and dedicated memory in use.  The card is the adapter with the most dedicated memory in
    use - the discrete one a model runs on - and every instance is filtered to its LUID, so an iGPU beside it does not
    add to the numbers.  Nothing here is allowed to raise: a Monitor tile that cannot be read is empty, not a crash."""

    def __init__(self):
        self.lib = self.engine = self.memory = None
        self.luid = None
        self.last_bind = 0.0

    def ok(self):
        return self.lib is not None and self.engine is not None and self.luid is not None

    def open(self):
        try:
            self.lib = ctypes.windll.LoadLibrary("pdh.dll")
            self.query = ctypes.c_void_p()
            if self.lib.PdhOpenQueryW(None, 0, ctypes.byref(self.query)) != 0:
                raise OSError("PdhOpenQueryW")
            self.engine = _PdhSet(self.lib, self.query, "GPU Engine", "Utilization Percentage")
            self.memory = _PdhSet(self.lib, self.query, "GPU Adapter Memory", "Dedicated Usage", large=True)
            self._pick_card()
        except (OSError, AttributeError):
            self.lib = None
        return self.ok()

    def _pick_card(self):
        """The LUID of the adapter holding the most dedicated memory, from the instance names (luid_0x... in the
        path) and a reading of each."""
        best = -1.0
        for path in _counter_instances("GPU Adapter Memory"):
            if "Dedicated Usage" not in path:
                continue
            m = re.search(r"\((luid_[^)]+)\)", path)
            if not m:
                continue
            counter = ctypes.c_void_p()
            if self.lib.PdhAddCounterW(self.query, ctypes.c_wchar_p(path), 0, ctypes.byref(counter)) != 0:
                continue
            for _ in range(2):
                self.lib.PdhCollectQueryData(self.query)
            v = _PdhSingle()
            self.lib.PdhGetFormattedCounterValue(counter, self.memory.fmt, None, ctypes.byref(v))
            if v.value.largeValue > best:
                best, self.luid = float(v.value.largeValue), m.group(1)
        self.engine.luid = self.memory.luid = self.luid

    def refresh(self):
        if not self.ok():
            return 0
        self.last_bind = time.time()
        return self.engine.refresh() + self.memory.refresh()

    def read(self):
        """{"util": percent, "mem_used": bytes}: the card's engine utilisation summed as Task Manager sums it, and
        its dedicated memory in use.  A card that reads idle gets its instance list rebuilt (a process that has just
        started has none yet), at most every BIND_MIN_S."""
        if not self.ok():
            return {}
        try:
            util = min(100.0, sum(self.engine.values()))
            mem = self.memory.values()
            if not util and time.time() - self.last_bind > BIND_MIN_S:
                self.refresh()
            return {"util": util, "mem_used": int(max(mem, default=0))}
        except (OSError, AttributeError, ValueError):
            return {}


class _XeGpu:
    """The Intel Arc readings in NVML's shape. Load and VRAM come from /run/gpustat.json (a root sampler: the only
    VRAM accounting xe exposes is per-client fdinfo, readable by root only); temperature, power and the PCIe link are
    read from sysfs too, so they show without the sampler."""
    STAT = "/run/gpustat.json"

    def __init__(self, index=0):
        import glob
        self.dev = None
        for d in sorted(glob.glob("/sys/class/drm/card[0-9]*/device")):
            try:
                if open(f"{d}/vendor").read().strip() == "0x8086" and os.path.isdir(f"{d}/tile0"):
                    self.dev = os.path.realpath(d)
                    break
            except OSError:
                continue
        self.hwmon = None
        for n in glob.glob(f"{self.dev}/hwmon/hwmon*/name") if self.dev else []:
            self.hwmon = os.path.dirname(n)
        self._e = None                          # (t, card energy uJ) for power without the sampler

    def ok(self):
        if sys.platform.startswith("linux"):
            return self.dev is not None
        # Windows: sysfs has no card; the name/VRAM come from setup's display-adapter
        # detection (setup.intel_gpus_windows).  Load and VRAM in use come from the OS's own
        # GPU counters (the class above); temperature/power/PCIe have no Windows counter.
        try:
            from pathlib import Path as _P
            import sys as _s
            _s.path.insert(0, str(_P(__file__).resolve().parents[2]))
            import setup as _S
            intel = _S.intel_gpus_windows() if _S.WIN else []
            self._win = [g for g in intel if _S.intel_problem(g) is None]
            if not self._win:
                return False
            # once: Telemetry.sample() asks ok() every second, and a fresh reader would throw the bound instances away
            if getattr(self, "_counters", None) is None:
                self._counters = _WinCounters()
                self._counters.open()
            return True
        except (OSError, ValueError, ImportError):
            return False

    def name(self):
        st = self._stat()
        if (st or {}).get("name"):
            return st["name"]
        try:
            return (self.__dict__.get("_win") or [{}])[0].get("name") or "Intel Arc GPU"
        except (AttributeError, IndexError):
            return "Intel Arc GPU"

    def _stat(self):
        try:
            import json
            with open(self.STAT) as f:
                st = json.load(f)
            return st if time.time() - float(st.get("ts", 0)) < 15 else None
        except (OSError, ValueError):
            return None

    def _rd(self, p):
        try:
            return open(p).read().strip()
        except OSError:
            return None

    def _link(self):
        """The link the card trained at: its own functions sit behind an internal x1 switch, so the first port up
        the path wider than x1; the max is what card AND slot allow (the root port caps it)."""
        gen = {"2.5": 1, "5.0": 2, "8.0": 3, "16.0": 4, "32.0": 5, "64.0": 6}
        d, chain = self.dev, []
        while d and d.startswith("/sys/devices/pci") and os.path.exists(f"{d}/current_link_speed"):
            chain.append(d)
            d = os.path.dirname(d)
        for p in chain:
            w = int(self._rd(f"{p}/current_link_width") or 0)
            if w > 1:
                g = lambda q, k: gen.get((self._rd(f"{q}/{k}_link_speed") or "").split(" ")[0])
                card, slot = g(p, "max"), g(chain[-1], "max")
                return g(p, "current"), min(x for x in (card, slot) if x) if (card or slot) else None, w
        return None, None, None

    def read(self):
        out = {}
        if not sys.platform.startswith("linux"):
            # Windows: the OS's GPU counters for load and VRAM in use (the reader binds its instances on open()).
            out.update(getattr(self, "_counters", None).read() if getattr(self, "_counters", None) else {})
        st = self._stat()
        if st:
            out["util"] = st.get("busy_pct")
            if st.get("vram_used_mb") is not None:
                out["mem_used"] = st["vram_used_mb"] * 2**20
                out["mem_total"] = st.get("vram_total_mb", 0) * 2**20 or None
            out["temp"] = st.get("temp_pkg")
            out["power"] = st.get("power_w")
            out["power_limit"] = st.get("power_cap_w")
            out["vram_temp"] = st.get("temp_vram_max") or st.get("temp_vram")
        if self.hwmon:
            if out.get("temp") is None:
                for lab in os.listdir(self.hwmon):
                    if lab.endswith("_label") and self._rd(f"{self.hwmon}/{lab}") == "pkg":
                        v = self._rd(f"{self.hwmon}/{lab[:-6]}_input")
                        out["temp"] = int(v) / 1000 if v else None
            if out.get("power") is None:
                e, t = self._rd(f"{self.hwmon}/energy1_input"), time.time()
                if e and e.isdigit():
                    if self._e and t > self._e[0] and int(e) >= self._e[1]:
                        out["power"] = (int(e) - self._e[1]) / 1e6 / (t - self._e[0])
                    self._e = (t, int(e))
            if out.get("power_limit") is None:
                cap = self._rd(f"{self.hwmon}/power1_cap")
                out["power_limit"] = int(cap) / 1e6 if cap and cap.isdigit() and int(cap) > 0 else None
        out["pcie_gen"], out["pcie_gen_max"], out["pcie_width"] = self._link()
        out["pcie_rx_mb"] = out["pcie_tx_mb"] = None    # xe exposes no PCIe traffic counters
        if not sys.platform.startswith("linux"):
            # Windows: report the card's VRAM size so the Monitor tab sizes correctly.
            try:
                win = (self.__dict__.get("_win") or [{}])[0]
                if out.get("mem_total") is None and win.get("vram_gb"):
                    out["mem_total"] = win["vram_gb"] * 2**30
            except (AttributeError, IndexError, TypeError):
                pass
        return out
