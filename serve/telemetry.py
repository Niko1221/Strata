"""serve/telemetry.py - hardware readings for the web app's Monitor tab (idea from PR #22 by code-martin).

A background thread samples once a second and keeps the last 60 readings of each series for the sparklines:
- GPU: NVIDIA's own NVML library (nvml.dll / libnvidia-ml.so.1, installed with every driver) through ctypes, so no
  pip package is needed: load, VRAM, temperature, power, PCIe link and throughput.  With the AMD backend (#301): the
  amdgpu driver's Linux sysfs files - load, VRAM, temperature and power; on Windows (#1380) DXGI (the card and its
  VRAM size), the performance counters (load, VRAM in use), AMD's driver library ADL (temperature, power, load) and
  the PCI link properties (PCIe generation and width).
- CPU, RAM, disk: `psutil` when it is installed (setup installs it); without it the CPU and RAM readings fall back to
  the OS (Windows GlobalMemoryStatusEx / GetSystemTimes, Linux /proc) and the disk rate is absent.
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

HISTORY = 60
IS_WIN = os.name == "nt"


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

    @staticmethod
    def _gen(text):
        """The PCIe generation of a link-speed file's contents ("16.0 GT/s PCIe" -> 4), or None."""
        try:
            return {2.5: 1, 5.0: 2, 8.0: 3, 16.0: 4, 32.0: 5, 64.0: 6}[float(str(text).split()[0])]
        except (ValueError, TypeError, IndexError, KeyError):
            return None

    @staticmethod
    def _bdf(s):
        """True for a sysfs pci device name ("0000:03:00.0"); nothing is imported for it."""
        return (len(s) == 12 and s[4] == ":" and s[7] == ":" and s[10] == "." and s[11] in "01234567"
                and all(c in "0123456789abcdef" for c in s[0:4] + s[5:7] + s[8:10]))

    @staticmethod
    def _read(d, name):
        try:
            with open(os.path.join(d, name), encoding="utf-8") as f:
                return f.read().strip()
        except (OSError, TypeError):
            return None

    def _hops(self):
        """The PCIe devices between the root port and this card, the card last: the sysfs path names every
        hop (a card behind a bridge chain has more than one; a directly attached card has one)."""
        out, prefix = [], []
        for part in os.path.realpath(self.dev or "").split("/"):
            prefix.append(part)
            if self._bdf(part):
                out.append("/".join(prefix))
        return out

    def link(self):
        """The PCIe link the card actually gets: the **narrowest/slowest hop** between the root port and the card,
        from each hop's `max_link_speed` / `max_link_width` (the capability, so a power-saving downgrade or a Gen3
        slot cannot make it read low).  A card that is Gen4 on its own hop but sits behind a Gen3 root port really
        runs at Gen3, and that is the number a PCIe bandwidth budget needs.
        `pcie_own_gen` keeps the card's own hop aside: the gap between the two is what a user has to see.

        The bottleneck is only claimed when **every** hop of the path could be read; `pcie_path` reports how many
        of them were ("4/4").  A kernel or a container that hides part of /sys/devices would otherwise drop the
        unreadable hops in silence and report the card's own Gen4 hop as the whole path - i.e. be optimistic
        exactly where it matters.  When the walk is incomplete the reading falls back to the card's own negotiated
        link and pcie_path says so."""
        hops = self._hops()
        gens, widths, read = [], [], 0
        for d in hops:
            g = self._gen(_Amd._read(d, "max_link_speed"))
            w = self._int(os.path.join(d, "max_link_width"))
            read += 1 if (g is not None or w is not None) else 0
            if g:
                gens.append(g)
            if w:
                widths.append(w)
        own = self.dev or ""
        own_gen = self._gen(_Amd._read(own, "current_link_speed"))
        whole = bool(hops) and read == len(hops)
        gen = min(gens) if (whole and gens) else own_gen
        width = min(widths) if (whole and widths) else self._int(os.path.join(own, "current_link_width"))
        return {"pcie_gen": gen, "pcie_gen_max": gen, "pcie_own_gen": own_gen, "pcie_width": width,
                "pcie_path": "%d/%d" % (read, len(hops))}


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
        out.update(self.link())
        return out


# ------------------------------------------------------------------------------------------------ AMD (Windows)
# #1380: Windows has no amdgpu sysfs, so the same readings come from four places, each optional (what one cannot give
# stays None and the others still show):
#   DXGI                 which card (name, PCI device id, LUID) and how much VRAM it has
#   performance counters "GPU Adapter Memory" (VRAM in use, all processes) and "GPU Engine" (load), by the card's LUID
#   ADL (atiadlxx.dll)   AMD's own driver library, installed with Adrenalin: PMLog sensors - temperature, power, load
#   PCI link properties  SetupAPI: the current and maximum PCIe generation and width (the throughput is not readable)
# Everything is ctypes against DLLs Windows or the AMD driver already has: no pip package, no SDK.
AMD_VENDOR = 0x1002
ADL_SENSORS = {"activity_gfx": 19, "temp_edge": 8, "temp_gfx": 28, "temp_hotspot": 27, "asic_power": 23,
               "board_power": 73, "gfx_power": 30}
ADL_AMD_VENDORS = (1002, AMD_VENDOR)      # ADL reports AMD's vendor id as the decimal number 1002 (AMD's samples), not 0x1002
LUID_RE = re.compile(r"luid_0x([0-9a-fA-F]+)_0x([0-9a-fA-F]+)")


class _Guid(ctypes.Structure):
    _fields_ = [("d1", ctypes.c_uint32), ("d2", ctypes.c_uint16), ("d3", ctypes.c_uint16), ("d4", ctypes.c_ubyte * 8)]


def _guid(text):
    import uuid
    u = uuid.UUID(text)
    return _Guid(u.fields[0], u.fields[1], u.fields[2], (ctypes.c_ubyte * 8)(*u.bytes[8:]))


def _com_call(obj, slot, argtypes, *args):
    """Calls method number `slot` of a COM object (its vtable entry) through ctypes; returns the HRESULT."""
    vtbl = ctypes.cast(obj, ctypes.POINTER(ctypes.POINTER(ctypes.c_void_p)))[0]
    return ctypes.WINFUNCTYPE(ctypes.c_long, ctypes.c_void_p, *argtypes)(vtbl[slot])(obj, *args)


class _DxgiDesc1(ctypes.Structure):
    _fields_ = [("Description", ctypes.c_wchar * 128), ("VendorId", ctypes.c_uint), ("DeviceId", ctypes.c_uint),
                ("SubSysId", ctypes.c_uint), ("Revision", ctypes.c_uint), ("DedicatedVideoMemory", ctypes.c_size_t),
                ("DedicatedSystemMemory", ctypes.c_size_t), ("SharedSystemMemory", ctypes.c_size_t),
                ("LuidLow", ctypes.c_uint), ("LuidHigh", ctypes.c_int), ("Flags", ctypes.c_uint)]


def amd_order(adapters):
    """The AMD adapters in the order HIP numbers them (#325): an integrated Radeon (little dedicated memory) first,
    then the discrete cards, each group in DXGI's order."""
    return sorted(adapters, key=lambda a: a["vram"] >= 3 << 30)


def dxgi_amd_adapters():
    """Windows: the AMD GPUs DXGI lists (software adapters left out), as dicts with name, device id, LUID (low, high),
    dedicated VRAM in bytes - in HIP's order (amd_order).  [] when there is none or DXGI is not usable."""
    if not IS_WIN:
        return []
    out = []
    try:
        dxgi = ctypes.WinDLL("dxgi.dll")
        factory = ctypes.c_void_p()
        if dxgi.CreateDXGIFactory1(ctypes.byref(_guid("770aae78-f26f-4dba-a829-253c83d1b387")),
                                   ctypes.byref(factory)) != 0:
            return []
        try:
            for i in range(64):
                ad = ctypes.c_void_p()
                if _com_call(factory, 12, (ctypes.c_uint, ctypes.POINTER(ctypes.c_void_p)), i, ctypes.byref(ad)) != 0:
                    break                                       # IDXGIFactory1::EnumAdapters1
                try:
                    d = _DxgiDesc1()
                    if (_com_call(ad, 10, (ctypes.POINTER(_DxgiDesc1),), ctypes.byref(d)) == 0     # GetDesc1
                            and d.VendorId == AMD_VENDOR and not d.Flags & 2):                      # 2: software
                        out.append({"name": d.Description.strip(), "device": d.DeviceId, "vram": int(d.DedicatedVideoMemory),
                                    "luid": (d.LuidHigh & 0xFFFFFFFF, d.LuidLow & 0xFFFFFFFF)})
                finally:
                    _com_call(ad, 2, ())                                                            # Release
        finally:
            _com_call(factory, 2, ())
    except (OSError, AttributeError, ValueError):
        return []
    return amd_order(out)


# --- performance counters
class _PdhValue(ctypes.Structure):
    _fields_ = [("status", ctypes.c_ulong), ("value", ctypes.c_double)]


class _PdhItem(ctypes.Structure):
    _fields_ = [("name", ctypes.c_wchar_p), ("val", _PdhValue)]


def luid_of(instance):
    """(high, low) of the adapter a performance-counter instance name belongs to, e.g.
    'pid_4_luid_0x00000000_0x0000E3F7_phys_0_eng_0_engtype_3D' -> (0, 0xE3F7); None without one."""
    m = LUID_RE.search(instance or "")
    return (int(m.group(1), 16), int(m.group(2), 16)) if m else None


def pdh_load(items, luid):
    """The card's load in percent from "GPU Engine\\Utilization Percentage" instances [(name, value)]: each engine's
    use summed over the processes on it, the busiest engine counts (what Task Manager shows).  None without any."""
    engines: dict = {}
    for name, v in items:
        if luid_of(name) == luid and "_phys_" in name:
            key = name.split("_phys_", 1)[1]
            engines[key] = engines.get(key, 0.0) + v
    return min(100.0, max(engines.values())) if engines else None


def pdh_vram_used(items, luid):
    """VRAM in use (bytes) from "GPU Adapter Memory\\Dedicated Usage" instances [(name, value)], this card's."""
    vals = [v for name, v in items if luid_of(name) == luid]
    return sum(vals) if vals else None


class _Pdh:
    """A Performance Data Helper query over wildcard counters, read once a second; instances (a process using the GPU)
    come and go, so the query is opened anew every REOPEN reads."""
    REOPEN = 15
    MORE_DATA = 0x800007D2
    FMT_DOUBLE = 0x200

    def __init__(self, paths):
        self.lib = ctypes.WinDLL("pdh.dll")
        for fn in ("PdhOpenQueryW", "PdhAddEnglishCounterW", "PdhCollectQueryData", "PdhCloseQuery",
                   "PdhGetFormattedCounterArrayW"):
            getattr(self.lib, fn).restype = ctypes.c_ulong
        self.lib.PdhOpenQueryW.argtypes = [ctypes.c_wchar_p, ctypes.c_size_t, ctypes.POINTER(ctypes.c_void_p)]
        self.lib.PdhAddEnglishCounterW.argtypes = [ctypes.c_void_p, ctypes.c_wchar_p, ctypes.c_size_t,
                                                   ctypes.POINTER(ctypes.c_void_p)]
        self.lib.PdhCollectQueryData.argtypes = [ctypes.c_void_p]
        self.lib.PdhCloseQuery.argtypes = [ctypes.c_void_p]
        self.lib.PdhGetFormattedCounterArrayW.argtypes = [ctypes.c_void_p, ctypes.c_ulong, ctypes.POINTER(ctypes.c_ulong),
                                                          ctypes.POINTER(ctypes.c_ulong), ctypes.c_void_p]
        self.paths, self.query, self.counters, self.reads = list(paths), None, [], 0
        self.open()

    def open(self):
        self.close()
        q = ctypes.c_void_p()
        if self.lib.PdhOpenQueryW(None, 0, ctypes.byref(q)) != 0:
            raise OSError("PdhOpenQuery")
        self.query = q
        for p in self.paths:
            c = ctypes.c_void_p()
            if self.lib.PdhAddEnglishCounterW(q, p, 0, ctypes.byref(c)) != 0:
                raise OSError("PdhAddCounter " + p)
            self.counters.append(c)
        self.lib.PdhCollectQueryData(q)
        self.reads = 0

    def close(self):
        if self.query is not None:
            self.lib.PdhCloseQuery(self.query)
        self.query, self.counters = None, []

    def read(self):
        """[[(instance, value) ...] per counter]; the first read after (re)opening has no rates yet: all empty."""
        self.reads += 1
        if self.reads >= self.REOPEN:
            self.open()
            return [[] for _ in self.counters]
        self.lib.PdhCollectQueryData(self.query)
        out = []
        for c in self.counters:
            size, count = ctypes.c_ulong(0), ctypes.c_ulong(0)
            if self.lib.PdhGetFormattedCounterArrayW(c, self.FMT_DOUBLE, ctypes.byref(size), ctypes.byref(count),
                                                     None) != self.MORE_DATA:
                out.append([])
                continue
            buf = ctypes.create_string_buffer(size.value)
            if self.lib.PdhGetFormattedCounterArrayW(c, self.FMT_DOUBLE, ctypes.byref(size), ctypes.byref(count),
                                                     buf) != 0:
                out.append([])
                continue
            items = ctypes.cast(buf, ctypes.POINTER(_PdhItem))
            out.append([(items[k].name, items[k].val.value) for k in range(count.value) if items[k].val.status == 0])
        return out


# --- ADL
class _AdlInfo(ctypes.Structure):
    _fields_ = [("iSize", ctypes.c_int), ("iAdapterIndex", ctypes.c_int), ("strUDID", ctypes.c_char * 256),
                ("iBusNumber", ctypes.c_int), ("iDeviceNumber", ctypes.c_int), ("iFunctionNumber", ctypes.c_int),
                ("iVendorID", ctypes.c_int), ("strAdapterName", ctypes.c_char * 256),
                ("strDisplayName", ctypes.c_char * 256), ("iPresent", ctypes.c_int), ("iExist", ctypes.c_int),
                ("strDriverPath", ctypes.c_char * 256), ("strDriverPathExt", ctypes.c_char * 256),
                ("strPNPString", ctypes.c_char * 256), ("iOSDisplayIndex", ctypes.c_int)]


class _AdlSensor(ctypes.Structure):
    _fields_ = [("supported", ctypes.c_int), ("value", ctypes.c_int)]


class _AdlPmLog(ctypes.Structure):
    _fields_ = [("size", ctypes.c_int), ("sensors", _AdlSensor * 256)]            # ADL_PMLOG_MAX_SENSORS


def adl_pick(sensors):
    """temperature (C), power (W) and load (%) out of ADL's PMLog sensors {id: value} (only the supported ones)."""
    def first(*names):
        for n in names:
            v = sensors.get(ADL_SENSORS[n])
            if v is not None:
                return v
        return None
    temp = first("temp_edge", "temp_gfx", "temp_hotspot")
    power = first("asic_power", "board_power", "gfx_power")
    load = first("activity_gfx")
    return {"temp": float(temp) if temp is not None and 0 < temp < 150 else None,
            "power": float(power) if power is not None and power >= 0 else None,
            "util": float(load) if load is not None and 0 <= load <= 100 else None}


class _Adl:
    """AMD's driver library (atiadlxx.dll, installed with the Adrenalin driver): the PMLog sensors of one card."""

    def __init__(self, device_id):
        self.lib = ctypes.WinDLL("atiadlxx.dll")
        malloc = ctypes.cdll.msvcrt.malloc
        malloc.restype, malloc.argtypes = ctypes.c_void_p, [ctypes.c_size_t]
        self._cb = ctypes.WINFUNCTYPE(ctypes.c_void_p, ctypes.c_int)(lambda n: malloc(n))       # kept alive
        self.ctx, self.rc = ctypes.c_void_p(), None
        if self.lib.ADL2_Main_Control_Create(self._cb, 1, ctypes.byref(self.ctx)) < 0:
            raise OSError("ADL2_Main_Control_Create")
        n = ctypes.c_int(0)
        if self.lib.ADL2_Adapter_NumberOfAdapters_Get(self.ctx, ctypes.byref(n)) < 0 or n.value <= 0:
            raise OSError("no ADL adapters")
        infos = (_AdlInfo * n.value)()
        if self.lib.ADL2_Adapter_AdapterInfo_Get(self.ctx, infos, ctypes.sizeof(infos)) < 0:
            raise OSError("ADL2_Adapter_AdapterInfo_Get")
        want = f"DEV_{device_id:04X}"
        self.index, self.seen = None, []
        amd = [a for a in infos if a.iVendorID in ADL_AMD_VENDORS]
        for a in infos:
            self.seen.append((a.iAdapterIndex, a.iVendorID, a.strAdapterName.decode(errors="replace"),
                              a.strPNPString.decode(errors="replace")))
        # the card's ADL index handles (one per display): by its PCI device id first, then any AMD one (a single AMD card)
        for a in sorted(amd, key=lambda a: want not in a.strPNPString.decode(errors="replace").upper()):
            if self.raw(a.iAdapterIndex):
                self.index = a.iAdapterIndex
                break
        if self.index is None:
            raise OSError(f"no PMLog sensors from ADL (last result {self.rc}); adapters: {self.seen}")

    def raw(self, index=None):
        """{sensor id: value} of the sensors the driver says it supports (ids: ADL_PMLOG_SENSORS)."""
        d = _AdlPmLog()
        d.size = ctypes.sizeof(d)
        self.rc = self.lib.ADL2_New_QueryPMLogData_Get(self.ctx, self.index if index is None else index, ctypes.byref(d))
        if self.rc != 0:
            return {}
        return {i: d.sensors[i].value for i in range(256) if d.sensors[i].supported}

    def read(self):
        return adl_pick(self.raw())


# --- PCI link
class _DevPropKey(ctypes.Structure):
    _fields_ = [("fmtid", _Guid), ("pid", ctypes.c_ulong)]


class _SpDevInfo(ctypes.Structure):
    _fields_ = [("cbSize", ctypes.c_ulong), ("ClassGuid", _Guid), ("DevInst", ctypes.c_ulong), ("Reserved", ctypes.c_size_t)]


class _PciLink:
    """The card's PCIe link from the PCI device's Windows properties (devpkey: DEVPKEY_PciDevice_CurrentLinkSpeed 9,
    CurrentLinkWidth 10, MaxLinkSpeed 11, MaxLinkWidth 12).  The speed is the generation: 1 = 2.5, 2 = 5, 3 = 8,
    4 = 16, 5 = 32 GT/s.  The current one drops at idle."""
    KEY = "3ab22e31-8264-4b4e-9af5-a8d2d8e33e62"
    PIDS = {"pcie_gen": 9, "pcie_width": 10, "pcie_gen_max": 11, "pcie_width_max": 12}

    def __init__(self, device_id):
        lib = self.lib = ctypes.WinDLL("setupapi.dll")
        lib.SetupDiGetClassDevsW.restype = ctypes.c_void_p
        lib.SetupDiGetClassDevsW.argtypes = [ctypes.POINTER(_Guid), ctypes.c_wchar_p, ctypes.c_void_p, ctypes.c_ulong]
        lib.SetupDiEnumDeviceInfo.argtypes = [ctypes.c_void_p, ctypes.c_ulong, ctypes.POINTER(_SpDevInfo)]
        lib.SetupDiGetDeviceInstanceIdW.argtypes = [ctypes.c_void_p, ctypes.POINTER(_SpDevInfo), ctypes.c_wchar_p,
                                                    ctypes.c_ulong, ctypes.c_void_p]
        lib.SetupDiGetDevicePropertyW.argtypes = [ctypes.c_void_p, ctypes.POINTER(_SpDevInfo), ctypes.POINTER(_DevPropKey),
                                                  ctypes.POINTER(ctypes.c_ulong), ctypes.c_void_p, ctypes.c_ulong,
                                                  ctypes.POINTER(ctypes.c_ulong), ctypes.c_ulong]
        self.key = _guid(self.KEY)
        self.h = lib.SetupDiGetClassDevsW(ctypes.byref(_guid("4d36e968-e325-11ce-bfc1-08002be10318")), None, None, 2)
        if self.h in (None, ctypes.c_void_p(-1).value):          # DIGCF_PRESENT, the display class
            raise OSError("SetupDiGetClassDevs")
        self.dev = None
        want = f"VEN_{AMD_VENDOR:04X}&DEV_{device_id:04X}"
        for i in range(32):
            d = _SpDevInfo()
            d.cbSize = ctypes.sizeof(d)
            if not lib.SetupDiEnumDeviceInfo(self.h, i, ctypes.byref(d)):
                break
            buf = ctypes.create_unicode_buffer(512)
            if lib.SetupDiGetDeviceInstanceIdW(self.h, ctypes.byref(d), buf, 512, None) and want in buf.value.upper():
                self.dev = d
                break
        if self.dev is None:
            raise OSError("no PCI device for the card")

    def _prop(self, pid):
        t, need, v = ctypes.c_ulong(0), ctypes.c_ulong(0), ctypes.c_uint(0)
        ok = self.lib.SetupDiGetDevicePropertyW(self.h, ctypes.byref(self.dev), ctypes.byref(_DevPropKey(self.key, pid)),
                                                ctypes.byref(t), ctypes.byref(v), ctypes.sizeof(v), ctypes.byref(need), 0)
        return v.value if ok and t.value == 7 else None          # DEVPROP_TYPE_UINT32

    def read(self):
        r = {k: self._prop(pid) for k, pid in self.PIDS.items()}
        for k in ("pcie_gen", "pcie_gen_max"):
            if r[k] is not None and not 1 <= r[k] <= 6:
                r[k] = None
        return r


class _AmdWin:
    """#1380: an AMD card's readings on Windows, with _Nvml's interface (see the section comment).  `index` is the
    card's HIP number.  Any source that fails to start or to read leaves its readings None."""

    def __init__(self, index=0):
        self.adapter = self.pdh = self.adl = self.pci = None
        self.cache, self.stale, self.errors = {}, 0, {}
        adapters = dxgi_amd_adapters()
        if not adapters:
            return
        self.adapter = adapters[index] if 0 <= index < len(adapters) else adapters[0]
        for attr, make in (("pdh", lambda: _Pdh([r"\GPU Engine(*)\Utilization Percentage",
                                                  r"\GPU Adapter Memory(*)\Dedicated Usage"])),
                           ("adl", lambda: _Adl(self.adapter["device"])),
                           ("pci", lambda: _PciLink(self.adapter["device"]))):
            try:
                setattr(self, attr, make())
            except Exception as e:  # noqa: BLE001 - optional source: the dashboard shows what is left
                self.errors[attr] = f"{type(e).__name__}: {e}"

    def ok(self):
        return self.adapter is not None

    def name(self):
        return (self.adapter or {}).get("name") or "AMD Radeon"

    def _counters(self):
        """(load %, VRAM used bytes) from the performance counters; the last good pair for up to 3 reads in a row
        that have none (the query is reopened now and then, a counter can skip a second)."""
        if self.pdh is None:
            return None, None
        try:
            load_items, mem_items = self.pdh.read()
            r = (pdh_load(load_items, self.adapter["luid"]), pdh_vram_used(mem_items, self.adapter["luid"]))
        except Exception:  # noqa: BLE001
            r = (None, None)
        if r == (None, None) and self.stale < 3 and self.cache.get("counters"):
            self.stale += 1
            return self.cache["counters"]
        self.stale = 0 if r != (None, None) else self.stale
        if r != (None, None):
            self.cache["counters"] = r
        return r

    def read(self):
        load, used = self._counters()
        out = {"util": load, "mem_used": used, "mem_total": (self.adapter or {}).get("vram") or None}
        if self.adl is not None:
            try:
                a = self.adl.read()
                out["temp"], out["power"] = a["temp"], a["power"]
                if a["util"] is not None:                       # the driver's own busy figure, as Linux's gpu_busy_percent
                    out["util"] = a["util"]
            except Exception:  # noqa: BLE001
                pass
        if self.pci is not None:
            try:
                p = self.pci.read()
                out["pcie_gen"], out["pcie_gen_max"] = p["pcie_gen"], p["pcie_gen_max"]
                out["pcie_width"] = p["pcie_width"]
            except Exception:  # noqa: BLE001
                pass
        return out


def gpu_reader(index=0, amd=False):
    """The card's readings: NVML (NVIDIA), or with the AMD backend the amdgpu sysfs files (#301; Windows: _AmdWin)."""
    if amd:
        return _AmdWin(index) if IS_WIN else _Amd(index)
    return _Nvml(index)


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
class Telemetry:
    def __init__(self, extra=None, gpu_index=0, gpu_indices=None, amd=False):
        """`extra()` -> dict of more series to record each second (the server's tok/s).  `gpu_index`: the card the
        engine runs on, numbered as nvidia-smi and NVML number them (by PCI bus); `gpu_indices`: all of them when
        the model is split across several (issue #112) - the gpu_* readings are then their total (memory, power,
        PCIe traffic), mean (load) or hottest (temperature), and "gpus" has each card's own.  `amd`: the AMD backend's
        cards, numbered as HIP numbers them, read from sysfs (#301)."""
        self.extra = extra
        self.lock = threading.Lock()
        self.now: dict = {}
        self.hist = collections.defaultdict(lambda: collections.deque(maxlen=HISTORY))
        idx = list(gpu_indices) if gpu_indices and len(gpu_indices) > 1 else [gpu_index]
        self.gpus = [(i, gpu_reader(i, amd)) for i in idx]
        self.gpus = [(i, g) for i, g in self.gpus if g.ok()] or self.gpus[:1]
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
            # #1380: the AMD readings are the amdgpu driver's sysfs files (Linux) or DXGI / performance counters / ADL
            # (Windows); this is said when none of them found the card
            "gpu_note": ("no readings for this AMD card (Linux: the amdgpu driver's sysfs files; Windows: DXGI, the "
                         "performance counters and AMD's ADL library); the engine's own VRAM figures are in its log"
                         if amd and not self.gpu.ok() else None),
            "cpu_name": _cpu_name(),
            "cores": (self.ps.cpu_count(logical=False) if self.ps else None) or None,
            "threads": os.cpu_count(),
            "psutil": self.ps is not None,
        }
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
        if c is None:   # psutil found no disk (a gVisor container, Windows with its disk counters off)
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

    def close(self):
        """Ends the sampler thread (a server that stops, a test's service): it used to run for the life of the process."""
        self._stop.set()

    def _loop(self):
        while not self._stop.is_set():
            s = self.sample()
            with self.lock:
                self.now = s
                for k in ("gpu_util", "gpu_mem_used", "gpu_temp", "gpu_power", "gpu_pcie_rx_mb", "cpu", "ram_used",
                          "disk_read_mb", "tok_s", "prefill_tok_s_mean"):
                    v = s.get(k)
                    self.hist[k].append(round(v, 2) if isinstance(v, float) else v)
            self._stop.wait(1.0)

    def snapshot(self):
        with self.lock:
            return {"now": dict(self.now), "history": {k: list(v) for k, v in self.hist.items()},
                    "static": dict(self.static)}


if __name__ == "__main__":      # python -m serve.telemetry [GPU number] [amd]: what the dashboard would show, 8 readings
    _i = int(sys.argv[1]) if len(sys.argv) > 1 and sys.argv[1].isdigit() else 0
    _amd = "amd" in sys.argv[1:]
    _g = gpu_reader(_i, _amd)
    print(f"GPU {_i}: {_g.name() if _g.ok() else 'not found'}")
    if _amd and IS_WIN and _g.ok():
        print("sources:", {k: getattr(_g, k) is not None for k in ("pdh", "adl", "pci")})
        for _k, _e in _g.errors.items():
            print(f"  {_k} did not start - {_e}")
        if _g.adl is not None:
            print("ADL supported sensors (id: value):", _g.adl.raw())
    for _ in range(8):
        time.sleep(1)
        print({k: (round(v, 1) if isinstance(v, float) else v) for k, v in _g.read().items()} if _g.ok() else {})
