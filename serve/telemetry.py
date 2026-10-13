"""serve/telemetry.py - hardware readings for the web app's Monitor tab (idea from PR #22 by code-martin).

A background thread samples once a second and keeps the last 60 readings of each series for the sparklines:
- GPU: NVIDIA's own NVML library (nvml.dll / libnvidia-ml.so.1, installed with every driver) through ctypes, so no
  pip package is needed: load, VRAM, temperature, power, PCIe link and throughput.  With the AMD backend (#301): the
  amdgpu driver's Linux sysfs files - load, VRAM, temperature and power; on Windows, where there is no sysfs, the
  card's VRAM and load from DXGI (#1380) and its temperature and power from AMD's own ADLX library.
- CPU, RAM, disk: `psutil` when it is installed (setup installs it); without it the CPU and RAM readings fall back to
  the OS (Windows GlobalMemoryStatusEx / GetSystemTimes, Linux /proc) and the disk rate is absent.
Anything that cannot be read is None; nothing here can stop the server.
"""
from __future__ import annotations

import collections
import ctypes
import os
import platform
import sys
import threading
import time

HISTORY = 60


# ------------------------------------------------------------------------------------------------ NVML
class _Nvml:
    # Whether this reader sees the PCIe traffic itself.  NVML does (nvmlDeviceGetPcieThroughput, in read() below);
    # the AMD paths do not, and the server then fills gpu_pcie_rx_mb from the bytes the engine moved over the link.
    has_pcie_throughput = True

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

    has_pcie_throughput = False          # sysfs has no PCIe byte counter: the link files are link state, not traffic

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


# ------------------------------------------------------------------------------------------------ AMD (Windows, DXGI)
class _DxgiLuid(ctypes.Structure):                                             # LUID, inside DXGI_ADAPTER_DESC1
    _fields_ = [("LowPart", ctypes.c_uint), ("HighPart", ctypes.c_int)]


class _DxgiDVal(ctypes.Structure):                                    # PDH_FMT_COUNTERVALUE, PDH_FMT_DOUBLE
    _fields_ = [("Status", ctypes.c_ulong), ("Value", ctypes.c_double)]


class _DxgiItem(ctypes.Structure):                                    # PDH_FMT_COUNTERVALUE_ITEM_W (pdh.h)
    _fields_ = [("name", ctypes.c_wchar_p), ("value", _DxgiDVal)]


class _Adlx:
    """An AMD card's temperature and power on Windows, from AMD's own ADLX library (amdadlx64.dll, installed with the
    Adrenalin driver).  A driver newer than the library knows is gated out of its GPU list (measured: ADLX 1.5.0.124,
    driver 32.0.23033.1002: plain ADLXInitialize answers GetGPUs with an empty list); ADLXInitializeWithIncompatible
    Driver2 takes it, and the GPU then comes from the ADL mapping - GetADLXGPUFromAdlAdapterIndex - not from the list,
    whose At() answers UNKNOWN_INTERFACE there.  Measured on an RX 7900 XT on that driver while the engine held 19 GiB:
    GPUTemperature 48 C and GPUTotalBoardPower 370 W while the GPU ran at 100 percent, 82-101 W near idle; GPUPower
    and GPUIntakeTemperature answer NOT_SUPPORTED, so the power falls back to the board power.  No device, no context,
    no VRAM taken.  The vtable slots are from the SDK headers at the library's version, and a wrong slot is silent:
    slot 8 is the hotspot temperature (18 C above the GPU one), and slot 11, the fan speed, is an int that answers a
    denormal double when read as one.  The values are sanity-checked anyway.  Stays silent when the library, the
    export or the card is not there."""

    SYS_PERFMON = 9                                       # IADLXSystem::GetPerformanceMonitoringServices
    MAP_GPU = 1                                           # IADLMapping::GetADLXGPUFromAdlAdapterIndex
    IFACE_RELEASE = 1                                     # IADLXInterface::Release
    GPU_NAME = 7                                          # IADLXGPU::Name
    PM_START, PM_CURRENT = 11, 18                         # PerformanceMonitoringServices::Start, ::GetCurrentGPUMetrics
    M_TEMP, M_POWER, M_BOARD = 7, 9, 10                   # GPUMetrics::GPUTemperature, ::GPUPower, ::GPUTotalBoardPower
    _shared = None                                        # (system, mapping, pm): ADLX initializes once per process,
    #                                                     # and its GetPerformanceMonitoringServices hangs on the
    #                                                     # second call (measured) - so the service is kept too

    @staticmethod
    def _call(obj, slot, argtypes, *args, restype=ctypes.c_int):
        """Call the slot-th method of an ADLX object's vtable, this pointer as the first argument; the ADLX_RESULT
        back.  ADLX interfaces are plain C++ vtables, not COM: no QueryInterface, no IID."""
        vt = ctypes.cast(obj, ctypes.POINTER(ctypes.c_void_p)).contents.value
        at = vt + slot * ctypes.sizeof(ctypes.c_void_p)
        fn = ctypes.cast(ctypes.cast(at, ctypes.POINTER(ctypes.c_void_p)).contents.value,
                         ctypes.CFUNCTYPE(restype, *argtypes))
        return fn(*args)

    def _open(self):
        """The ADLX system, mapping and metrics service, once per process: a second ADLXInitialize answers
        ALREADY_INITIALIZED and a second GetPerformanceMonitoringServices hangs (both measured), so every later
        reader (a second card, free_vram_mib's) reuses the first one's objects.  The tracking is started here,
        once; GetCurrentGPUMetrics on the shared service is safe to call any number of times."""
        if type(self)._shared is not None:
            return type(self)._shared
        dll = ctypes.WinDLL("amdadlx64.dll")
        init = getattr(dll, "ADLXInitializeWithIncompatibleDriver2", None)
        if init is None:                                   # ADLX before 1.2: no way past the driver gate
            return None
        ver = ctypes.c_ulonglong(0)
        dll.ADLXQueryFullVersion.argtypes = [ctypes.POINTER(ctypes.c_ulonglong)]
        if dll.ADLXQueryFullVersion(ctypes.byref(ver)) != 0:
            return None
        system, mapping = ctypes.c_void_p(), ctypes.c_void_p()
        init.argtypes = [ctypes.c_ulonglong, ctypes.POINTER(ctypes.c_void_p), ctypes.POINTER(ctypes.c_void_p)]
        if init(ver.value, ctypes.byref(system), ctypes.byref(mapping)) != 0 or not mapping.value:
            return None
        pm = ctypes.c_void_p()
        if self._call(system, self.SYS_PERFMON, [ctypes.c_void_p, ctypes.POINTER(ctypes.c_void_p)],
                      system, ctypes.byref(pm)) != 0 or not pm.value:
            return None
        if self._call(pm, self.PM_START, [ctypes.c_void_p], pm) != 0:   # StartPerformanceMetricsTracking
            return None
        type(self)._shared = (system, mapping, pm)
        return type(self)._shared

    def __init__(self, want=None):
        self._pm = self._gpu = None
        if os.name != "nt":
            return
        try:
            opened = self._open()
            if opened is None:
                return
            _, mapping, pm = opened
            gpu = self._find(mapping, want)
            if gpu is None:
                return
            self._pm, self._gpu = pm, gpu
        except OSError:
            self._pm = self._gpu = None

    def _find(self, mapping, want):
        """The GPU object for this card: the ADL adapter indexes in order (the mapping is the only path that reaches
        it on a gated driver).  The name must be the one DXGI reported for the card; with exactly one card found,
        that one."""
        found = []
        for i in range(16):
            g = ctypes.c_void_p()
            if self._call(mapping, self.MAP_GPU,
                          [ctypes.c_void_p, ctypes.c_int, ctypes.POINTER(ctypes.c_void_p)],
                          mapping, i, ctypes.byref(g)) != 0 or not g.value:
                continue
            s = ctypes.c_char_p()
            if self._call(g, self.GPU_NAME, [ctypes.c_void_p, ctypes.POINTER(ctypes.c_char_p)],
                          g, ctypes.byref(s)) == 0:
                if want and (s.value or b"").decode(errors="replace") == want:
                    return g
                found.append(g)
        return found[0] if len(found) == 1 else None

    def _double(self, obj, slot):
        v = ctypes.c_double()
        return v.value if self._call(obj, slot, [ctypes.c_void_p, ctypes.POINTER(ctypes.c_double)],
                                     obj, ctypes.byref(v)) == 0 else None

    def read(self):
        """{"temp": C, "power": W} for the values that answer and are plausible; {} for the rest.  The metrics object
        is a fresh reference each read, so it is released again."""
        out = {}
        if self._pm is None:
            return out
        try:
            m = ctypes.c_void_p()
            if self._call(self._pm, self.PM_CURRENT,
                          [ctypes.c_void_p, ctypes.c_void_p, ctypes.POINTER(ctypes.c_void_p)],
                          self._pm, self._gpu, ctypes.byref(m)) != 0 or not m.value:
                return out
            try:
                t = self._double(m, self.M_TEMP)
                if t is not None and 0.0 < t < 150.0:
                    out["temp"] = t
                w = self._double(m, self.M_POWER)
                if w is None or not 0.0 < w < 1200.0:
                    w = self._double(m, self.M_BOARD)
                if w is not None and 0.0 < w < 1200.0:
                    out["power"] = w
            finally:
                self._call(m, self.IFACE_RELEASE, [ctypes.c_void_p], m)
        except OSError:
            pass
        return out


class _Dxgi:
    """#1380: an AMD card's readings on Windows, where there is no NVML and no amdgpu sysfs to read.  DXGI is the
    display API every Windows GPU driver implements: it gives the card's name and its dedicated memory (total).
    For the VRAM in use, DXGI's own QueryVideoMemoryInfo is per-process - it reported 0 on the AMD driver while the
    engine held 19 GiB - so the used figure comes from the same performance counter Task Manager reads, selected by
    the adapter's LUID.  So does the busy percent, from the GPU Engine counter beside it: the busiest engine, as
    Task Manager shows it.  Enumeration and counters only - no device, no context: it takes no VRAM from the engine
    and cannot stall it.  Temperature and power come
    from the ADLX library below (_Adlx), the only one that answered them on this machine: ADL enumerates the card but
    rejects every sensor call (-1/-5), and no performance counter or WMI class reports them."""

    AMD_VENDOR = 0x1002          # the PCI vendor ID of ATI/AMD: the card HIP runs on, not an NVIDIA display card
    # No PCIe traffic here to read: the performance counters have no PCIe counter set (only the five GPU sets and
    # RDMA Activity), amdadlx64.dll and atiadlxx.dll export no bus metric, and ADLX 1.5's headers name no link
    # counter - measured on an RX 7900 XT.  The server fills that figure from the bytes the engine moved instead.
    has_pcie_throughput = False

    class Desc1(ctypes.Structure):                                             # DXGI_ADAPTER_DESC1
        _fields_ = [("Description", ctypes.c_wchar * 128), ("VendorId", ctypes.c_uint),
                    ("DeviceId", ctypes.c_uint), ("SubSysId", ctypes.c_uint), ("Revision", ctypes.c_uint),
                    ("DedicatedVideoMemory", ctypes.c_size_t), ("DedicatedSystemMemory", ctypes.c_size_t),
                    ("SharedSystemMemory", ctypes.c_size_t), ("AdapterLuid", _DxgiLuid), ("Flags", ctypes.c_uint)]

    class Val(ctypes.Structure):                                               # PDH_FMT_COUNTERVALUE, PDH_FMT_LARGE
        _fields_ = [("Status", ctypes.c_long), ("Value", ctypes.c_longlong)]

    @staticmethod
    def _iid(s):
        """A GUID string as the 16 bytes REFIID wants: the first three fields little-endian, the last two as written."""
        b = bytes.fromhex(s.replace("-", ""))
        return (ctypes.c_ubyte * 16).from_buffer_copy(b[3::-1] + b[5:3:-1] + b[7:5:-1] + b[8:])

    @staticmethod
    def _com(obj, slot, argtypes, *args):
        """Call the slot-th method of a COM object's vtable, this pointer as the first argument; the HRESULT back.
        The vtable holds function pointers, so read the slot before calling it."""
        vt = ctypes.cast(obj, ctypes.POINTER(ctypes.c_void_p)).contents.value
        at = vt + slot * ctypes.sizeof(ctypes.c_void_p)
        fn = ctypes.cast(ctypes.cast(at, ctypes.POINTER(ctypes.c_void_p)).contents.value,
                         ctypes.CFUNCTYPE(ctypes.c_long, *argtypes))
        return fn(*args)

    def __init__(self, index=0):
        self.ad = self.total = self._name = self.luid = None
        self._pdh = self._q = self._c = None                                  # the busy-percent query, kept open
        self._adlx = None
        if os.name != "nt":
            return
        try:
            fac = ctypes.c_void_p()
            create = ctypes.CDLL("dxgi.dll").CreateDXGIFactory1
            create.argtypes = [ctypes.c_void_p, ctypes.POINTER(ctypes.c_void_p)]
            create.restype = ctypes.c_long
            # IID_IDXGIFactory1 {770AAE78-F26F-4DBA-A829-253C83D1B387} (dxgi.h)
            if create(self._iid("770aae78-f26f-4dba-a829-253c83d1b387"), ctypes.byref(fac)) != 0:
                return
            seen = -1
            for i in range(16):
                ad = ctypes.c_void_p()
                if self._com(fac, 12, [ctypes.c_void_p, ctypes.c_uint, ctypes.POINTER(ctypes.c_void_p)],
                             fac, i, ctypes.byref(ad)) != 0:                  # IDXGIFactory1::EnumAdapters1
                    break
                d = self.Desc1()
                if self._com(ad, 10, [ctypes.c_void_p, ctypes.POINTER(self.Desc1)], ad,
                             ctypes.byref(d)) == 0 and d.VendorId == self.AMD_VENDOR:   # IDXGIAdapter1::GetDesc1
                    seen += 1
                    if seen == index:
                        self.ad, self.total, self._name = ad, d.DedicatedVideoMemory or None, d.Description or None
                        self.luid = (d.AdapterLuid.HighPart & 0xFFFFFFFF, d.AdapterLuid.LowPart)
                        break
                self._com(ad, 2, [ctypes.c_void_p], ad)                       # Release the cards we do not keep
        except OSError:
            self.ad = None
        if self.ad is not None:
            self._adlx = _Adlx(self._name)                                    # temperature and power, above

    def ok(self):
        return self.ad is not None

    def name(self):
        return self._name

    def _dedicated_usage(self):
        """The VRAM this card holds, from the performance counter Task Manager reads; the counter instance is the
        adapter's LUID.  None when the counter is not there."""
        pdh = ctypes.WinDLL("pdh.dll")
        q = ctypes.c_void_p()
        pdh.PdhOpenQueryW.argtypes = [ctypes.c_void_p, ctypes.c_void_p, ctypes.POINTER(ctypes.c_void_p)]
        if pdh.PdhOpenQueryW(None, None, ctypes.byref(q)) != 0:
            return None
        try:
            c = ctypes.c_void_p()
            path = "\\GPU Adapter Memory(luid_0x%08x_0x%08x_phys_0)\\Dedicated Usage" % self.luid
            pdh.PdhAddEnglishCounterW.argtypes = [ctypes.c_void_p, ctypes.c_wchar_p, ctypes.c_void_p,
                                                  ctypes.POINTER(ctypes.c_void_p)]
            if pdh.PdhAddEnglishCounterW(q, path, None, ctypes.byref(c)) != 0:
                return None
            if pdh.PdhCollectQueryData(q) != 0:
                return None
            v = self.Val()
            pdh.PdhGetFormattedCounterValue.argtypes = [ctypes.c_void_p, ctypes.c_ulong, ctypes.c_void_p,
                                                        ctypes.POINTER(self.Val)]
            if pdh.PdhGetFormattedCounterValue(c, 0x400, None, ctypes.byref(v)) != 0:   # PDH_FMT_LARGE
                return None
            return v.Value if v.Status == 0 else None
        finally:
            pdh.PdhCloseQuery(q)

    def _util(self):
        """How busy the card is, from the GPU Engine counter Task Manager's GPU column is built from, its instances
        picked by the adapter's LUID: the busiest engine, which is what Task Manager shows.  The counter is a rate
        over the time since its last sample, so the query and its counter stay open between reads - a handle with
        no previous sample answers PDH_NO_DATA - and the first read after startup has no value.  Measured on an RX
        7900 XT decoding at 60 tok/s: 93% on the compute engine, where the Running Time counter beside it computed
        94% over the same window.  None when the counter set is not there."""
        if self.luid is None:
            return None
        if self._q is None:
            pdh = ctypes.WinDLL("pdh.dll")
            q = ctypes.c_void_p()
            pdh.PdhOpenQueryW.argtypes = [ctypes.c_void_p, ctypes.c_void_p, ctypes.POINTER(ctypes.c_void_p)]
            if pdh.PdhOpenQueryW(None, None, ctypes.byref(q)) != 0:
                return None
            c = ctypes.c_void_p()
            path = "\\GPU Engine(*luid_0x%08x_0x%08x_phys_0*)\\Utilization Percentage" % self.luid
            pdh.PdhAddEnglishCounterW.argtypes = [ctypes.c_void_p, ctypes.c_wchar_p, ctypes.c_void_p,
                                                  ctypes.POINTER(ctypes.c_void_p)]
            if pdh.PdhAddEnglishCounterW(q, path, None, ctypes.byref(c)) != 0:
                pdh.PdhCloseQuery(q)
                return None
            self._pdh, self._q, self._c = pdh, q, c
        pdh = self._pdh
        if pdh.PdhCollectQueryData(self._q) != 0:
            return None
        n, k = ctypes.c_ulong(0), ctypes.c_ulong(0)
        pdh.PdhGetFormattedCounterArrayW.argtypes = [ctypes.c_void_p, ctypes.c_ulong, ctypes.POINTER(ctypes.c_ulong),
                                                     ctypes.POINTER(ctypes.c_ulong), ctypes.c_void_p]
        # 0x2 = PDH_FMT_DOUBLE.  The first call only fills the size and answers PDH_MORE_DATA, so its return is not
        # checked; a counter with no second sample yet reports no size at all.
        pdh.PdhGetFormattedCounterArrayW(self._c, 0x2, ctypes.byref(n), ctypes.byref(k), None)
        if not n.value:
            return None
        buf = ctypes.create_string_buffer(n.value)
        if pdh.PdhGetFormattedCounterArrayW(self._c, 0x2, ctypes.byref(n), ctypes.byref(k), buf) != 0:
            return None
        top = None
        for i in range(k.value):
            it = _DxgiItem.from_address(ctypes.addressof(buf) + i * ctypes.sizeof(_DxgiItem))
            if it.value.Status == 0 and (top is None or it.value.Value > top):
                top = it.value.Value
        return None if top is None else max(0, min(100, int(round(top))))

    def read(self):
        out = {}
        if self.ad is None:
            return out
        try:
            used = self._dedicated_usage()
            if used is not None:
                out["mem_used"], out["mem_total"] = used, self.total
            u = self._util()
            if u is not None:
                out["util"] = u
        except OSError:
            pass
        if self._adlx is not None:
            out.update(self._adlx.read())
        return out


def gpu_reader(index=0, amd=False):
    """The card's readings: NVML (NVIDIA), or the amdgpu sysfs files with the AMD backend (#301) - and on Windows,
    where there is no /sys at all, its VRAM from DXGI (#1380).  A sysfs root that exists but holds no amdgpu (a
    Linux without the driver, a test tree) stays on the sysfs path: nothing, as before."""
    if not amd:
        return _Nvml(index)
    g = _Amd(index)
    return g if g.ok() or os.path.isdir(SYSFS) else _Dxgi(index)


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
            # #1380: the AMD readings are the amdgpu driver's Linux sysfs files, on Windows DXGI; a card none of them
            # reaches gets a word why, instead of "not readable (NVML)" or empty tiles in silence
            "gpu_note": ("no GPU readings for this card: an AMD card on Windows is read through DXGI, and it gave "
                         "nothing; the engine's own VRAM figures are in its log" if amd and not self.gpu.ok() else None),
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
