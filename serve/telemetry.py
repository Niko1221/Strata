"""serve/telemetry.py - hardware readings for the web app's Monitor tab (idea from PR #22 by code-martin).

A background thread samples once a second and keeps the last 60 readings of each series for the sparklines:
- GPU: NVIDIA's own NVML library (nvml.dll / libnvidia-ml.so.1, installed with every driver) through ctypes, so no
  pip package is needed: load, VRAM, temperature, power, PCIe link and throughput. Multi-card engine readings and
  histories are also kept per card; NVIDIA cards outside the engine are reported separately and included in
  all-GPU power. With the AMD backend (#301): the amdgpu driver's Linux sysfs files - load, VRAM, temperature and
  power.
- CPU, RAM, disk: `psutil` when it is installed (setup installs it); without it the CPU and RAM readings fall back to
  the OS (Windows GlobalMemoryStatusEx / GetSystemTimes, Linux /proc) and the disk rate is absent.
- Optional CPU package sensors: LibreHardwareMonitor's local JSON web server at `127.0.0.1:8085`, configurable
  with `STRATA_LHM_URL`; failures back off for 30 s. On Linux the CPU temperature also comes from the kernel's hwmon
  files (k10temp / zenpower on AMD, coretemp on Intel) when LHM gives none.
Anything that cannot be read is None; nothing here can stop the server.
"""
from __future__ import annotations

import collections
import ctypes
import json
import os
import platform
import re
import sys
import threading
import time
import urllib.request

HISTORY = 60
CARD_FIELDS = ("util", "mem_used", "mem_total", "temp", "power", "power_limit", "pcie_rx_mb", "pcie_tx_mb",
               "pcie_gen", "pcie_gen_max", "pcie_width")


def _card(index, reading):
    """One card's entry in `gpus` / `other_gpus`."""
    return {"index": index, **{k: reading.get(k) for k in CARD_FIELDS}}


# ------------------------------------------------------------------------------------------------ NVML
def _load_nvml():
    """Load and initialize NVML, or return None when it is unavailable."""
    names = ["nvml.dll", os.path.join(os.environ.get("ProgramFiles", r"C:\Program Files"),
                                      "NVIDIA Corporation", "NVSMI", "nvml.dll")] if os.name == "nt" \
        else ["libnvidia-ml.so.1", "libnvidia-ml.so"]
    lib = None
    for name in names:
        try:
            lib = ctypes.CDLL(name)
            break
        except OSError:
            continue
    if lib is None:
        return None
    try:
        init = getattr(lib, "nvmlInit_v2", None) or lib.nvmlInit
        return lib if init() == 0 else None
    except (AttributeError, OSError):
        return None


class _Nvml:
    class Util(ctypes.Structure):
        _fields_ = [("gpu", ctypes.c_uint), ("memory", ctypes.c_uint)]

    class Mem(ctypes.Structure):
        _fields_ = [("total", ctypes.c_ulonglong), ("free", ctypes.c_ulonglong), ("used", ctypes.c_ulonglong)]

    def __init__(self, index=0):
        self.lib, self.dev = _load_nvml(), None
        if self.lib is None:
            return
        try:
            h = ctypes.c_void_p()
            get = getattr(self.lib, "nvmlDeviceGetHandleByIndex_v2", None) or self.lib.nvmlDeviceGetHandleByIndex
            if get(ctypes.c_uint(index), ctypes.byref(h)) != 0:
                self.lib = None
                return
            self.dev = h
        except (AttributeError, OSError):
            self.lib = None

    @staticmethod
    def count():
        """Number of NVML devices, or zero if NVML cannot report it."""
        try:
            lib = _load_nvml()
            if lib is None:
                return 0
            get = getattr(lib, "nvmlDeviceGetCount_v2", None) or lib.nvmlDeviceGetCount
            count = ctypes.c_uint()
            return count.value if get(ctypes.byref(count)) == 0 else 0
        except Exception:  # noqa: BLE001 - an unavailable NVML count is zero
            return 0

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


# ------------------------------------------------------------------------------------------------ AMD (Linux sysfs)
SYSFS = "/sys"


def amd_device_dir(index, sysfs=None):
    """The amdgpu sysfs folder (/sys/class/drm/renderD<N>/device) of the AMD GPU that HIP numbers `index`: the KFD
    topology's GPU nodes in order, the CPU nodes skipped, linked to their render node by drm_render_minor - the
    numbering setup's amd_gpus() and HIP_VISIBLE_DEVICES use.  None when there is no such card (or no amdgpu)."""
    base = os.path.join(sysfs or SYSFS, "class", "kfd", "kfd", "topology", "nodes")
    try:
        nodes = sorted((n for n in os.listdir(base) if n.isdigit()), key=int)
    except OSError:
        return None
    gpus = []
    for n in nodes:
        try:
            with open(os.path.join(base, n, "properties"), encoding="utf-8") as f:
                props = dict(line.strip().partition(" ")[::2] for line in f if line.strip())
            if int(props.get("gfx_target_version") or 0) == 0 or int(props.get("simd_count") or 0) == 0:
                continue
            gpus.append(props)
        except (OSError, ValueError):
            continue
    if not 0 <= index < len(gpus) or not gpus[index].get("drm_render_minor"):
        return None
    dev = os.path.join(sysfs or SYSFS, "class", "drm", "renderD" + gpus[index]["drm_render_minor"].strip(), "device")
    return dev if os.path.isdir(dev) else None


class _Amd:
    """#301: an AMD card's readings from the amdgpu driver's sysfs files (Linux; no ROCm library needed), with _Nvml's
    interface: load (gpu_busy_percent), VRAM (mem_info_vram_used / _total), and from its hwmon folder the temperature
    (temp1_input, the edge sensor, m°C), power (power1_average or power1_input, µW) and its cap (power1_cap)."""

    def __init__(self, index=0, sysfs=None):
        self.dev = amd_device_dir(index, sysfs)
        self.hwmon = None
        if self.dev:
            try:
                hw = sorted(os.listdir(os.path.join(self.dev, "hwmon")))
                self.hwmon = os.path.join(self.dev, "hwmon", hw[0]) if hw else None
            except OSError:
                pass

    def ok(self):
        return self.dev is not None

    @staticmethod
    def _int(path):
        try:
            with open(path, encoding="utf-8") as f:
                return int(f.read().strip())
        except (OSError, ValueError, TypeError):
            return None

    def name(self):
        try:
            with open(os.path.join(self.dev, "product_name"), encoding="utf-8") as f:
                return f.read().strip() or "AMD Radeon"
        except (OSError, TypeError):
            return "AMD Radeon"

    def read(self):
        out = {"util": self._int(os.path.join(self.dev, "gpu_busy_percent")),
               "mem_used": self._int(os.path.join(self.dev, "mem_info_vram_used")),
               "mem_total": self._int(os.path.join(self.dev, "mem_info_vram_total"))}
        if self.hwmon:
            t = self._int(os.path.join(self.hwmon, "temp1_input"))
            out["temp"] = t / 1000.0 if t is not None else None
            p = self._int(os.path.join(self.hwmon, "power1_average"))
            if p is None:
                p = self._int(os.path.join(self.hwmon, "power1_input"))
            out["power"] = p / 1e6 if p is not None else None
            cap = self._int(os.path.join(self.hwmon, "power1_cap"))
            out["power_limit"] = cap / 1e6 if cap is not None else None
        return out


def gpu_reader(index=0, amd=False):
    """The card's readings: NVML (NVIDIA), or the amdgpu sysfs files with the AMD backend (#301)."""
    return _Amd(index) if amd else _Nvml(index)


def free_vram_mib(index=0, amd=False):
    """Free VRAM of a card in MiB, or None when it cannot be read."""
    g = gpu_reader(index, amd)
    if not g.ok():
        return None
    r = g.read()
    if r.get("mem_total") is None or r.get("mem_used") is None:
        return None
    return int((r["mem_total"] - r["mem_used"]) >> 20)


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
def _fetch_lhm(url, timeout):
    with urllib.request.urlopen(url, timeout=timeout) as response:
        return json.loads(response.read())


class _LhmCpu:
    """Optional CPU package sensors from LibreHardwareMonitor's local JSON web server."""
    DEFAULT_URL = "http://127.0.0.1:8085/data.json"
    # The first CPU's sensors, picked by name: LHM numbers them per CPU model (the package temperature is
    # /intelcpu/0/temperature/10 on an i9-9900KF, another index elsewhere) but names them the same way.
    CPU_SENSOR = re.compile(r"^/(?:intel|amd)cpu/0/(temperature|power)/\d+$")
    NAMES = {"temperature": ("CPU Package", "Package", "Core (Tctl/Tdie)", "Core (Tctl)", "Core (Tdie)", "Core Max"),
             "power": ("CPU Package", "Package")}
    FIELDS = {"temperature": "cpu_temp", "power": "cpu_power"}

    def __init__(self, url=None, fetch=None, clock=None):
        self.url = os.environ.get("STRATA_LHM_URL", self.DEFAULT_URL) if url is None else url
        self.url = self.url.strip()
        self.enabled = bool(self.url)
        self.fetch = fetch or _fetch_lhm
        self.clock = clock or time.monotonic
        self.next_try = 0.0
        self.last_success = None
        self.values = {"cpu_temp": None, "cpu_power": None}

    @staticmethod
    def _number(value):
        if not isinstance(value, str):
            return None
        match = re.match(r"^\s*([+-]?(?:\d+(?:[.,]\d*)?|[.,]\d+))", value)
        if not match:
            return None
        try:
            return float(match.group(1).replace(",", "."))
        except ValueError:
            return None

    @classmethod
    def _sensors(cls, tree):
        named = {kind: {} for kind in cls.NAMES}
        stack = [tree]
        while stack:
            node = stack.pop()
            if isinstance(node, dict):
                match = cls.CPU_SENSOR.match(str(node.get("SensorId") or ""))
                if match and isinstance(node.get("Text"), str):
                    named[match.group(1)][node["Text"].strip()] = cls._number(node.get("Value"))
                children = node.get("Children")
                if isinstance(children, list):
                    stack.extend(children)
            elif isinstance(node, list):
                stack.extend(node)
        return {cls.FIELDS[kind]: next((named[kind][name] for name in names if named[kind].get(name) is not None), None)
                for kind, names in cls.NAMES.items()}

    def read(self):
        if not self.enabled:
            return {}
        now = self.clock()
        if now < self.next_try:
            if self.last_success is None or now - self.last_success > 10:
                return {"cpu_temp": None, "cpu_power": None}
            return dict(self.values)
        try:
            self.values = self._sensors(self.fetch(self.url, 0.5))
            self.last_success = now
            self.next_try = now + 2
        except Exception:  # noqa: BLE001 - optional sensors must never stop the sampler
            self.values = {"cpu_temp": None, "cpu_power": None}
            self.last_success = None
            self.next_try = now + 30
        return dict(self.values)


class _HwmonCpuTemp:
    """Linux CPU package temperature from the kernel's hwmon files: no root and no lm-sensors needed. k10temp and
    zenpower (AMD) label it Tctl / Tdie, coretemp (Intel) "Package id 0"; an older k10temp has only an unlabelled
    temp1. The first CPU's file is found once; each read is one small sysfs read."""
    ROOT = "/sys/class/hwmon"
    LABELS = {"k10temp": ("Tctl", "Tdie"), "zenpower": ("Tdie", "Tctl"), "coretemp": ("Package id 0",)}

    def __init__(self, root=None):
        self.root = self.ROOT if root is None else root
        self.path = self._find()
        self.enabled = self.path is not None

    @staticmethod
    def _text(path):
        try:
            with open(path, encoding="utf-8") as f:
                return f.read().strip()
        except OSError:
            return None

    def _find(self):
        try:
            entries = sorted(os.listdir(self.root))
        except OSError:
            return None
        for entry in entries:
            folder = os.path.join(self.root, entry)
            labels = self.LABELS.get(self._text(os.path.join(folder, "name")))
            if not labels:
                continue
            try:
                files = os.listdir(folder)
            except OSError:
                continue
            found = {self._text(os.path.join(folder, f)): os.path.join(folder, f[:-len("_label")] + "_input")
                     for f in files if f.startswith("temp") and f.endswith("_label")}
            for label in labels:
                if label in found and os.path.exists(found[label]):
                    return found[label]
            if not found and labels[0] != "Package id 0" and "temp1_input" in files:
                return os.path.join(folder, "temp1_input")
        return None

    def read(self):
        if not self.enabled:
            return {}
        try:
            return {"cpu_temp": int(self._text(self.path)) / 1000}
        except (TypeError, ValueError):
            return {"cpu_temp": None}


class Telemetry:
    def __init__(self, extra=None, gpu_index=0, gpu_indices=None, amd=False):
        """`extra()` -> dict of more series to record each second (the server's tok/s).  `gpu_index`: the card the
        engine runs on, numbered as nvidia-smi and NVML number them (by PCI bus); `gpu_indices`: all of them when
        the model is split across several (issue #112) - the gpu_* readings are then their total (memory, power,
        PCIe traffic), mean (load) or hottest (temperature), and `gpus` and `history.gpus` have each card's own
        readings. Multi-card entries also include per-card power limit and PCIe link/throughput fields. NVIDIA cards
        outside the engine appear in `other_gpus`; when any power is readable, `all_gpu_power` sums every NVML card.
        `amd`: the AMD backend's cards, numbered as HIP numbers them, read from sysfs (#301); NVIDIA-only fields are
        not emitted for AMD. When `STRATA_LHM_URL` is non-empty (default localhost:8085), CPU package temperature
        and power are read from LibreHardwareMonitor in this sampling loop, with a 0.5 s timeout and 30 s error
        back-off. `measured_power` combines readable GPU power with CPU package power."""
        self.extra = extra
        self.lock = threading.Lock()
        self.now: dict = {}
        self.hist = collections.defaultdict(lambda: collections.deque(maxlen=HISTORY))
        self.gpu_hist = collections.defaultdict(lambda: collections.defaultdict(
            lambda: collections.deque(maxlen=HISTORY)))
        idx = list(gpu_indices) if gpu_indices and len(gpu_indices) > 1 else [gpu_index]
        engine_indices = set(idx)
        self.gpus = [(i, gpu_reader(i, amd)) for i in idx]
        self.gpus = [(i, g) for i, g in self.gpus if g.ok()] or self.gpus[:1]
        self.gpu = self.gpus[0][1]
        self.other_gpus = []
        if not amd:
            try:
                for i in range(_Nvml.count()):
                    if i in engine_indices:
                        continue
                    try:
                        reader = gpu_reader(i)
                        if reader.ok():
                            self.other_gpus.append((i, reader))
                    except Exception:  # noqa: BLE001 - an extra card must not stop telemetry
                        continue
            except Exception:  # noqa: BLE001 - NVML enumeration is optional
                pass
        try:
            import psutil  # noqa: F401
            self.ps = sys.modules["psutil"]
        except ImportError:
            self.ps = None
        self.fallback = _CpuRamFallback()
        self.lhm = _LhmCpu()
        self.hwmon_cpu = _HwmonCpuTemp() if platform.system() == "Linux" else None
        self.static = {
            "os": platform.system().lower(),
            "gpu_name": " + ".join(g.name() or "?" for _, g in self.gpus) if self.gpu.ok() else None,
            "gpu_count": len(self.gpus),
            # #1380: the AMD readings are the amdgpu driver's Linux sysfs files; a Windows AMD card has none yet, and the
            # dashboard said "not readable (NVML)" or showed empty tiles with no word why
            "gpu_note": ("no GPU load or VRAM readings for AMD cards on Windows yet (Linux reads them from the amdgpu "
                         "driver); the engine's own VRAM figures are in its log" if amd and not self.gpu.ok() else None),
            "cpu_name": _cpu_name(),
            "cores": (self.ps.cpu_count(logical=False) if self.ps else None) or None,
            "threads": os.cpu_count(),
            "psutil": self.ps is not None,
        }
        if self.other_gpus:
            self.static["other_gpu_names"] = [g.name() for _, g in self.other_gpus]
        self._disk_prev = None
        self._stop = threading.Event()
        threading.Thread(target=self._loop, daemon=True).start()

    def _disk(self):
        if not self.ps:
            return None, None
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
            s.update({f"gpu_{k}": v for k, v in g.items()})
            if len(reads) > 1 or self.other_gpus:
                s["gpus"] = [_card(i, r) for i, r in reads]
        if self.other_gpus:
            others = []
            for i, reader in self.other_gpus:
                try:
                    r = reader.read()
                except Exception:  # noqa: BLE001 - a bad extra card is just unreadable
                    r = {}
                others.append(_card(i, r))
            s["other_gpus"] = others
            powers = [s["gpu_power"]] if s.get("gpu_power") is not None else []
            powers.extend(g["power"] for g in others if g.get("power") is not None)
            if powers:
                s["all_gpu_power"] = sum(powers)
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
        lhm = self.lhm.read()
        s.update(lhm)
        if self.hwmon_cpu and s.get("cpu_temp") is None:
            s.update(self.hwmon_cpu.read())
        if lhm and lhm.get("cpu_power") is not None:
            gpu_power = s.get("all_gpu_power")
            if gpu_power is None:
                gpu_power = s.get("gpu_power")
            if gpu_power is not None:
                s["measured_power"] = gpu_power + lhm["cpu_power"]
        return s

    def record(self, sample):
        """Store one sample and its rolling top-level and multi-card history."""
        with self.lock:
            self.now = sample
            for k in ("gpu_util", "gpu_mem_used", "gpu_temp", "gpu_power", "gpu_pcie_rx_mb", "gpu_pcie_tx_mb",
                      "cpu", "ram_used", "disk_read_mb", "tok_s", "prefill_tok_s_mean"):
                v = sample.get(k)
                self.hist[k].append(round(v, 2) if isinstance(v, float) else v)
            if self.other_gpus:
                value = sample.get("all_gpu_power")
                self.hist["all_gpu_power"].append(round(value, 2) if isinstance(value, float) else value)
            if self.lhm.enabled:
                cpu_keys = ("cpu_temp", "cpu_power", "measured_power")
            else:
                cpu_keys = ("cpu_temp",) if self.hwmon_cpu and self.hwmon_cpu.enabled else ()
            if cpu_keys:
                for key in cpu_keys:
                    value = sample.get(key)
                    self.hist[key].append(round(value, 2) if isinstance(value, float) else value)
            cards = list(sample.get("gpus") or [])
            if self.other_gpus:
                if len(self.gpus) == 1 and not cards:
                    index = self.gpus[0][0]
                    cards.append({"index": index, "util": sample.get("gpu_util"),
                                  "mem_used": sample.get("gpu_mem_used"), "temp": sample.get("gpu_temp"),
                                  "power": sample.get("gpu_power"), "pcie_rx_mb": sample.get("gpu_pcie_rx_mb")})
                cards.extend(sample.get("other_gpus") or [])
            for card in cards:
                h = self.gpu_hist[str(card["index"])]
                for key in ("util", "mem_used", "temp", "power", "pcie_rx_mb"):
                    value = card.get(key)
                    h[key].append(round(value, 2) if isinstance(value, float) else value)

    def close(self):
        """Ends the sampler thread (a server that stops, a test's service): it used to run for the life of the process."""
        self._stop.set()

    def _loop(self):
        while not self._stop.is_set():
            self.record(self.sample())
            self._stop.wait(1.0)

    def snapshot(self):
        with self.lock:
            history = {k: list(v) for k, v in self.hist.items()}
            if self.gpu_hist:
                history["gpus"] = {i: {k: list(v) for k, v in series.items()}
                                   for i, series in self.gpu_hist.items()}
            return {"now": dict(self.now), "history": history,
                    "static": dict(self.static)}
