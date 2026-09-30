"""serve/telemetry.py - hardware readings for the web app's Monitor tab (idea from PR #22 by code-martin).

A background thread samples once a second and keeps the last 60 readings of each series for the sparklines:
- GPU: NVIDIA's own NVML library (nvml.dll / libnvidia-ml.so.1, installed with every driver) through ctypes, so no
  pip package is needed: load, VRAM, temperature, power, PCIe link and throughput.
- GPU on ROCm: the same fields from amdgpu's sysfs (/sys/class/drm/card*/device - see _AmdSysfs), because NVML does
  not exist on an AMD box.  PCIe throughput has no sysfs counter there: the link itself (Gen N x wide) is reported
  and the throughput series stays absent rather than invented.
- CPU, RAM, disk: `psutil` when it is installed (setup installs it); without it the CPU, RAM and disk readings fall
  back to the OS (Windows GlobalMemoryStatusEx / GetSystemTimes, Linux /proc).
Anything that cannot be read is None; nothing here can stop the server.
"""
from __future__ import annotations

import collections
import ctypes
import os
import platform
import re
import sys
import threading
import time
from pathlib import Path

HISTORY = 60

# whole disks only: never their partitions, and never loop/zram/dm (a device-mapper layer would count the same
# bytes twice).  /proc/diskstats counts 512-byte sectors - the same source psutil reads.
_WHOLE_DISK = re.compile(r"^(sd[a-z]+|nvme\d+n\d+|vd[a-z]+|xvd[a-z]+|mmcblk\d+|hd[a-z]+)$")

# amdgpu exposes no marketing name, only the PCI ids.  This is the kernel's own description of the one card
# this port is built for (gfx1100, the RX 7900 XT/XTX/GRE family); other cards show their ids.
_AMD_PCI_NAMES = {"1002:744C": "Navi 31 [Radeon RX 7900 XT/XTX/GRE] (gfx1100)"}


def _pcie_gen(speed):
    """'16.0 GT/s PCIe' -> 4 (the generation the link runs at).  NVML reports this as a number already."""
    if not speed:
        return None
    m = re.search(r"([0-9.]+)\s*GT/s", str(speed))
    return {2.5: 1, 5.0: 2, 8.0: 3, 16.0: 4, 32.0: 5, 64.0: 6}.get(float(m.group(1))) if m else None


def _proc_lines(path):
    """A /proc file's lines, closed properly; None when it cannot be read (also on Windows)."""
    try:
        with open(path, encoding="utf-8", errors="replace") as f:
            return f.read().splitlines()
    except OSError:
        return None


def _diskstats():
    """(read_bytes, write_bytes) over the physical disks, from /proc/diskstats.  None when it cannot be read."""
    lines = _proc_lines("/proc/diskstats")
    if lines is None:
        return None
    read = write = 0
    for line in lines:
        f = line.split()
        if len(f) < 10 or not _WHOLE_DISK.match(f[2]):
            continue
        read += int(f[5]) * 512
        write += int(f[9]) * 512
    return read, write


def _physical_cores():
    """How many physical cores /proc/cpuinfo shows (what psutil's cpu_count(logical=False) returns)."""
    lines = _proc_lines("/proc/cpuinfo")
    if lines is None:
        return None
    seen, phys, core = set(), None, None
    for line in lines + [""]:
        if line.startswith("physical id"):
            phys = line.split(":", 1)[1].strip()
        elif line.startswith("core id"):
            core = line.split(":", 1)[1].strip()
        elif not line.strip():
            if phys is not None and core is not None:
                seen.add((phys, core))
            phys = core = None
    return len(seen) or None


# ------------------------------------------------------------------------------------------------ NVML
class _Nvml:
    class Util(ctypes.Structure):
        _fields_ = [("gpu", ctypes.c_uint), ("memory", ctypes.c_uint)]

    class Mem(ctypes.Structure):
        _fields_ = [("total", ctypes.c_ulonglong), ("free", ctypes.c_ulonglong), ("used", ctypes.c_ulonglong)]

    def __init__(self, index=0):
        self.lib = self.dev = None
        names = ["nvml.dll", os.path.join(os.environ.get("ProgramFiles", r"C:\Program Files"),
                                          "NVIDIA Corporation", "NVSMI", "nvml.dll")] if os.name == "nt" \
            else ["libnvidia-ml.so.1", "libnvidia-ml.so"]
        for n in names:
            try:
                self.lib = ctypes.CDLL(n)
                break
            except OSError:
                continue
        if self.lib is None:
            return
        try:
            init = getattr(self.lib, "nvmlInit_v2", None) or self.lib.nvmlInit
            if init() != 0:
                self.lib = None
                return
            h = ctypes.c_void_p()
            get = getattr(self.lib, "nvmlDeviceGetHandleByIndex_v2", None) or self.lib.nvmlDeviceGetHandleByIndex
            if get(ctypes.c_uint(index), ctypes.byref(h)) != 0:
                self.lib = None
                return
            self.dev = h
        except (AttributeError, OSError):
            self.lib = None

    def ok(self):
        return self.lib is not None and self.dev is not None

    def _uint(self, fn, *args):
        v = ctypes.c_uint()
        try:
            return v.value if getattr(self.lib, fn)(self.dev, *args, ctypes.byref(v)) == 0 else None
        except (AttributeError, OSError):
            return None

    def name(self):
        buf = ctypes.create_string_buffer(96)
        try:
            if self.lib.nvmlDeviceGetName(self.dev, buf, ctypes.c_uint(96)) == 0:
                return buf.value.decode(errors="replace")
        except (AttributeError, OSError):
            pass
        return None

    def read(self):
        out = {}
        u = self.Util()
        try:
            if self.lib.nvmlDeviceGetUtilizationRates(self.dev, ctypes.byref(u)) == 0:
                out["util"] = u.gpu
        except (AttributeError, OSError):
            pass
        m = self.Mem()
        try:
            if self.lib.nvmlDeviceGetMemoryInfo(self.dev, ctypes.byref(m)) == 0:
                out["mem_used"], out["mem_total"] = m.used, m.total
        except (AttributeError, OSError):
            pass
        out["temp"] = self._uint("nvmlDeviceGetTemperature", ctypes.c_uint(0))          # NVML_TEMPERATURE_GPU
        mw = self._uint("nvmlDeviceGetPowerUsage")
        out["power"] = mw / 1000.0 if mw is not None else None
        lim = self._uint("nvmlDeviceGetEnforcedPowerLimit")
        out["power_limit"] = lim / 1000.0 if lim is not None else None
        out["pcie_gen"] = self._uint("nvmlDeviceGetCurrPcieLinkGeneration")        # drops at idle (power saving)
        out["pcie_gen_max"] = self._uint("nvmlDeviceGetMaxPcieLinkGeneration")
        out["pcie_width"] = self._uint("nvmlDeviceGetCurrPcieLinkWidth")
        rx = self._uint("nvmlDeviceGetPcieThroughput", ctypes.c_uint(1))                 # NVML_PCIE_UTIL_RX_BYTES, KB/s
        tx = self._uint("nvmlDeviceGetPcieThroughput", ctypes.c_uint(0))
        out["pcie_rx_mb"] = rx / 1024.0 if rx is not None else None
        out["pcie_tx_mb"] = tx / 1024.0 if tx is not None else None
        return out


def _read_text(path):
    try:
        return path.read_text().strip()
    except OSError:
        return None


# ------------------------------------------------------------------------------------------------ amdgpu (ROCm)
_SYS_DRM = Path("/sys/class/drm")


def _card_slot(dev):
    """A card device's PCI slot name ('0000:03:00.0') from its uevent."""
    try:
        for line in (dev / "uevent").read_text().splitlines():
            if line.startswith("PCI_SLOT_NAME="):
                return line.split("=", 1)[1].strip()
    except OSError:
        pass
    return None


def _links_hip_runtime(exe):
    """True when the engine binary links AMD's HIP runtime - the ground truth for 'the model runs on the AMD
    card'.  Only a string-table entry is read; the binary is never executed."""
    try:
        return b"libamdhip64" in Path(exe).read_bytes()
    except OSError:
        return False


def _hip_pci_map():
    """{HIP device ordinal: PCI slot name}, straight from AMD's HIP runtime.  {} whenever the runtime or the
    query is unavailable - callers then fall back to card order."""
    try:
        lib = ctypes.CDLL("libamdhip64.so.1") if os.name != "nt" else ctypes.WinDLL("amdhip64.dll")
        if lib.hipInit(0) != 0:
            return {}
        lib.hipDeviceGetCount.restype = ctypes.c_int
        n = ctypes.c_int()
        if lib.hipDeviceGetCount(ctypes.byref(n)) != 0:
            return {}
        lib.hipDeviceGetPciBusId.argtypes = [ctypes.c_char_p, ctypes.c_int, ctypes.c_int]
        out = {}
        for i in range(n.value):
            buf = ctypes.create_string_buffer(64)
            if lib.hipDeviceGetPciBusId(buf, 64, i) == 0:
                out[i] = buf.value.decode()
        return out
    except Exception:
        return {}


def engine_runs_on_amd(cfg):
    """Whose numbers the Monitor shows follows where the model runs: 0.1.26's setup lets a PC with both kinds
    of card choose AMD (setup.py asks), and NVML still loads there (it ships with the NVIDIA driver)."""
    backend = (cfg or {}).get("backend")
    if backend == "hip":
        return True
    if backend:
        return False
    return _links_hip_runtime((cfg or {}).get("exe") or "")


class _AmdSysfs:
    """The NVML fields, from amdgpu's sysfs - what a ROCm box has instead of libnvidia-ml.

    The card is found under /sys/class/drm/card*/device; the number is not fixed (a box with another DRM device
    can have the AMD card at card1), so the backend scans for DRIVER=amdgpu.  With a HIP map the card is the one
    at the model device's PCI slot - never "the index-th amdgpu", which on a Ryzen box can be the iGPU.  Every
    reading is a small sysfs file read: a sample cannot block the server, and anything the kernel does not
    expose stays None so the Monitor shows "-" instead of a made-up number.  PCIe throughput has no sysfs
    counter - the link generation and width are reported, the MB/s series stays empty."""

    def __init__(self, index=0, hip_map=None, root=None):
        self.dir = None
        self._temp = None
        cards = self._cards(root)
        slot = (hip_map or {}).get(index)
        if slot:
            for d in cards:
                if _card_slot(d) == slot:
                    self.dir = d
                    break
        if self.dir is None:                      # no HIP mapping: the index-th amdgpu card, as before
            for d in cards:
                if index <= 0:
                    self.dir = d
                    break
                index -= 1
        if self.dir is not None:
            self._temp = self._find_temp()

    @staticmethod
    def _cards(root=None):
        """Every <root>/cardN/device whose driver is amdgpu, in card order."""
        out = []
        paths = sorted((root or _SYS_DRM).glob("card[0-9]*/device"),
                       key=lambda p: int(re.sub(r"\D", "", p.parent.name) or 0))
        for p in paths:
            try:
                if "DRIVER=amdgpu" in (p / "uevent").read_text():
                    out.append(p)
            except OSError:
                continue
        return out

    def _find_temp(self):
        """The hwmon sensor labelled 'edge' (what NVML calls the GPU temperature), else the first temp input."""
        for h in sorted((self.dir / "hwmon").glob("hwmon*")):
            for label, value in (("temp1_label", "temp1_input"), ("temp2_label", "temp2_input"),
                                 ("temp3_label", "temp3_input")):
                try:
                    if (h / label).read_text().strip() == "edge":
                        return h / value
                except OSError:
                    continue
            if (h / "temp1_input").exists():
                return h / "temp1_input"
        return None

    def ok(self):
        return self.dir is not None

    @staticmethod
    def _read(path, scale=1.0, digits=None):
        try:
            raw = int(path.read_text().strip())
        except (OSError, ValueError):
            return None
        v = raw / scale if scale != 1.0 else raw
        return round(v, digits) if digits is not None else v

    def name(self):
        if self.dir is None:
            return "AMD GPU"
        pci = None
        try:
            for line in (self.dir / "uevent").read_text().splitlines():
                if line.startswith("PCI_ID="):
                    pci = line.split("=", 1)[1]
        except OSError:
            pass
        if not pci:
            return "AMD GPU"
        return _AMD_PCI_NAMES.get(pci.upper(), f"AMD GPU ({pci})")

    def read(self):
        if self.dir is None:                     # no amdgpu card: report nothing, the way NVML does when it cannot load
            return {}
        d = self.dir
        out = {"util": self._read(d / "gpu_busy_percent"),
               "mem_used": self._read(d / "mem_info_vram_used"),
               "mem_total": self._read(d / "mem_info_vram_total")}
        if self._temp is not None:
            out["temp"] = self._read(self._temp, 1000.0)                 # millidegrees -> °C
        for h in sorted((d / "hwmon").glob("hwmon*")):
            if out.get("power") is None:
                out["power"] = self._read(h / "power1_average", 1e6, 1)  # microwatts -> W
            if out.get("power_limit") is None:
                out["power_limit"] = self._read(h / "power1_cap", 1e6, 1)
        out["pcie_gen"] = _pcie_gen(_read_text(d / "current_link_speed"))            # drops at idle (power saving)
        out["pcie_gen_max"] = _pcie_gen(_read_text(d / "max_link_speed")) or out["pcie_gen"]
        out["pcie_width"] = self._read(d / "current_link_width")
        return {k: v for k, v in out.items() if v is not None}


# ------------------------------------------------------------------------------------------------ CPU / RAM
def _cpu_name():
    if os.name == "nt":
        try:
            import winreg
            k = winreg.OpenKey(winreg.HKEY_LOCAL_MACHINE, r"HARDWARE\DESCRIPTION\System\CentralProcessor\0")
            return winreg.QueryValueEx(k, "ProcessorNameString")[0].strip()
        except OSError:
            pass
    elif os.path.exists("/proc/cpuinfo"):
        for line in open("/proc/cpuinfo", encoding="utf-8", errors="replace"):
            if line.startswith("model name"):
                return line.split(":", 1)[1].strip()
    return platform.processor() or None


class _CpuRamFallback:
    """CPU load and RAM without psutil."""

    def __init__(self):
        self.prev = self._times()

    def _times(self):
        if os.name == "nt":
            idle, kern, user = (ctypes.c_ulonglong() for _ in range(3))
            if ctypes.windll.kernel32.GetSystemTimes(ctypes.byref(idle), ctypes.byref(kern), ctypes.byref(user)):
                return idle.value, kern.value + user.value           # kernel time includes idle
            return None
        try:
            f = [int(x) for x in open("/proc/stat").readline().split()[1:]]
            return f[3] + f[4], sum(f)
        except (OSError, ValueError):
            return None

    def cpu(self):
        cur = self._times()
        prev, self.prev = self.prev, cur
        if not cur or not prev or cur[1] == prev[1]:
            return None
        return max(0.0, min(100.0, 100.0 * (1 - (cur[0] - prev[0]) / (cur[1] - prev[1]))))

    @staticmethod
    def ram():
        if os.name == "nt":
            class MS(ctypes.Structure):
                _fields_ = [("dwLength", ctypes.c_ulong), ("dwMemoryLoad", ctypes.c_ulong),
                            ("ullTotalPhys", ctypes.c_ulonglong), ("ullAvailPhys", ctypes.c_ulonglong),
                            ("ullTotalPageFile", ctypes.c_ulonglong), ("ullAvailPageFile", ctypes.c_ulonglong),
                            ("ullTotalVirtual", ctypes.c_ulonglong), ("ullAvailVirtual", ctypes.c_ulonglong),
                            ("ullAvailExtendedVirtual", ctypes.c_ulonglong)]
            m = MS()
            m.dwLength = ctypes.sizeof(MS)
            if ctypes.windll.kernel32.GlobalMemoryStatusEx(ctypes.byref(m)):
                return m.ullTotalPhys - m.ullAvailPhys, m.ullTotalPhys
            return None, None
        try:
            info = dict(line.split(":", 1) for line in open("/proc/meminfo"))
            total = int(info["MemTotal"].split()[0]) * 1024
            avail = int(info["MemAvailable"].split()[0]) * 1024
            return total - avail, total
        except (OSError, KeyError, ValueError):
            return None, None


# ------------------------------------------------------------------------------------------------ the sampler
class Telemetry:
    def __init__(self, extra=None, gpu_index=0, gpu_indices=None, hip=None):
        """`extra()` -> dict of more series to record each second (the server's tok/s).  `gpu_index`: the card the
        engine runs on, numbered as nvidia-smi and NVML number them (by PCI bus); `gpu_indices`: all of them when
        the model is split across several (issue #112) - the gpu_* readings are then their total (memory, power,
        PCIe traffic), mean (load) or hottest (temperature), and "gpus" has each card's own.  `hip`: True when
        the model runs on the AMD card - the readings then come from amdgpu's sysfs even where NVML loads, and
        the cards are the ones at the model's HIP devices (a Ryzen iGPU is amdgpu too)."""
        self.extra = extra
        self.lock = threading.Lock()
        self.now: dict = {}
        self.hist = collections.defaultdict(lambda: collections.deque(maxlen=HISTORY))
        idx = list(gpu_indices) if gpu_indices and len(gpu_indices) > 1 else [gpu_index]
        if hip:
            hip_map = _hip_pci_map()              # the model's cards by PCI slot, not "the index-th amdgpu"
            self.gpus = [(i, g) for i, g in ((i, _AmdSysfs(i, hip_map)) for i in idx) if g.ok()] \
                or [(idx[0], _AmdSysfs(idx[0], hip_map))]
        else:
            self.gpus = [(i, _Nvml(i)) for i in idx]
            ok = [(i, g) for i, g in self.gpus if g.ok()]
            if not ok:                                    # ROCm box (no libnvidia-ml): the same fields from amdgpu's sysfs
                ok = [(i, g) for i, g in ((i, _AmdSysfs(i)) for i in idx) if g.ok()]
            self.gpus = ok or self.gpus[:1]
        self.gpu = self.gpus[0][1]
        try:
            import psutil  # noqa: F401
            self.ps = sys.modules["psutil"]
        except ImportError:
            self.ps = None
        self.fallback = _CpuRamFallback()
        self.static = {
            "gpu_name": " + ".join(g.name() or "?" for _, g in self.gpus) if self.gpu.ok() else None,
            "gpu_count": len(self.gpus),
            "cpu_name": _cpu_name(),
            "cores": (self.ps.cpu_count(logical=False) if self.ps else None) or _physical_cores(),
            "threads": os.cpu_count(),
            "psutil": self.ps is not None,
        }
        self._disk_prev = None
        threading.Thread(target=self._loop, daemon=True).start()

    def _disk(self):
        if not self.ps:
            cur = _diskstats()                   # no psutil: /proc/diskstats, the source psutil itself reads
            if cur is None:
                return None, None
            t = time.time()
            prev, self._disk_prev = self._disk_prev, (t, cur[0], cur[1])
            if prev is None or t <= prev[0]:
                return None, None
            dt = t - prev[0]
            return (cur[0] - prev[1]) / dt / 2**20, (cur[1] - prev[2]) / dt / 2**20
        try:
            c = self.ps.disk_io_counters()
        except (OSError, RuntimeError):
            return None, None
        t = time.time()
        prev, self._disk_prev = self._disk_prev, (t, c.read_bytes, c.write_bytes)
        if prev is None or t <= prev[0]:
            return None, None
        dt = t - prev[0]
        return (c.read_bytes - prev[1]) / dt / 2**20, (c.write_bytes - prev[2]) / dt / 2**20

    def sample(self):
        s = {}
        if self.gpu.ok():
            reads = [(i, g.read()) for i, g in self.gpus]
            g = dict(reads[0][1])
            if len(reads) > 1:
                def vals(k):
                    return [r[k] for _, r in reads if r.get(k) is not None]
                for k in ("mem_used", "mem_total", "power", "power_limit", "pcie_rx_mb", "pcie_tx_mb"):
                    v = vals(k)
                    g[k] = sum(v) if v else None
                u = vals("util")
                g["util"] = sum(u) / len(u) if u else None
                t = vals("temp")
                g["temp"] = max(t) if t else None
                s["gpus"] = [{"index": i, "util": r.get("util"), "mem_used": r.get("mem_used"),
                              "mem_total": r.get("mem_total"), "temp": r.get("temp"), "power": r.get("power")}
                             for i, r in reads]
            s.update({f"gpu_{k}": v for k, v in g.items()})
        if self.ps:
            try:
                s["cpu"] = self.ps.cpu_percent(interval=None)
                vm = self.ps.virtual_memory()
                s["ram_used"], s["ram_total"] = vm.total - vm.available, vm.total
            except (OSError, RuntimeError):
                pass
        else:
            s["cpu"] = self.fallback.cpu()
            s["ram_used"], s["ram_total"] = self.fallback.ram()
        s["disk_read_mb"], s["disk_write_mb"] = self._disk()
        if self.extra:
            try:
                s.update(self.extra())
            except Exception:  # noqa: BLE001 - telemetry must never take the server down
                pass
        return s

    def _loop(self):
        while True:
            s = self.sample()
            with self.lock:
                self.now = s
                for k in ("gpu_util", "gpu_mem_used", "gpu_temp", "gpu_power", "gpu_pcie_rx_mb", "cpu", "ram_used",
                          "disk_read_mb", "tok_s", "prefill_tok_s_mean"):
                    v = s.get(k)
                    self.hist[k].append(round(v, 2) if isinstance(v, float) else v)
            time.sleep(1.0)

    def snapshot(self):
        with self.lock:
            return {"now": dict(self.now), "history": {k: list(v) for k, v in self.hist.items()},
                    "static": dict(self.static)}
