"""Intel Arc readings for the server's Monitor tab, in the shape of serve/telemetry.py's readers.

Kept out of serve/telemetry.py so the SYCL port touches no shared file: sycl/serve/server_intel.py installs it as
telemetry.gpu_reader.  Two readers, one per platform (xe_gpu picks):

- Linux (the xe driver): load and VRAM need root (xe's only VRAM accounting is per-client fdinfo), so they come
  from a root sampler's /run/gpustat.json (sycl/tools/gpustat.py); temperature, power and the PCIe link are read
  from sysfs, so they show without it.
- Windows: Level Zero Sysman through ctypes on the driver's ze_loader.dll - VRAM used/total from the device memory
  modules, watts from two energy-counter samples, temperature from the sensors (the B70 reports none on Linux, so
  the field may stay empty), load from the engine groups' active time.  Best effort: any failure leaves the field
  None, and a loader that does not answer leaves the Monitor without a GPU, as upstream's Windows HIP state.  The
  card is the config's SYCL device index: Level Zero enumerates in the same order as SYCL's level_zero backend.
"""
from __future__ import annotations

import ctypes
import os
import sys
import time

WIN = os.name == "nt"


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
        return self.dev is not None and sys.platform.startswith("linux")

    def name(self):
        st = self._stat()
        return (st or {}).get("name") or "Intel Arc GPU"

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
        return out


# ------------------------------------------------------------------------------------------------ Windows: Level Zero Sysman
# The constants and struct layouts are the Level Zero spec's (oneapi-src/level-zero-spec, scripts/core/device.yml and
# scripts/sysman/{common,memory,power,temperature,engine}.yml, checked 2026-10-04): every Sysman properties/state
# struct starts with stype + pNext, the ze_device_properties_t core struct sits inside zes_device_properties_t.
# unverified on the B70: the whole reader - it has run against no Windows driver yet (no Intel GPU where it was written).
ZE_RESULT_SUCCESS = 0
ZE_INIT_FLAG_GPU_ONLY = 1
ZE_DEVICE_TYPE_GPU = 1
ZE_STRUCTURE_TYPE_DEVICE_PROPERTIES = 0x3
ZES_STRUCTURE_TYPE_DEVICE_PROPERTIES = 0x1
ZES_STRUCTURE_TYPE_ENGINE_PROPERTIES = 0x5
ZES_STRUCTURE_TYPE_MEM_PROPERTIES = 0xb
ZES_STRUCTURE_TYPE_POWER_PROPERTIES = 0xd
ZES_STRUCTURE_TYPE_TEMP_PROPERTIES = 0x14
ZES_STRUCTURE_TYPE_MEM_STATE = 0x1e
ZES_MEM_LOC_DEVICE = 1                      # zes_mem_loc_t: SYSTEM = 0, DEVICE = 1
ZES_TEMP_SENSORS_GLOBAL, ZES_TEMP_SENSORS_GPU, ZES_TEMP_SENSORS_MEMORY = 0, 1, 2
ZES_ENGINE_GROUP_ALL, ZES_ENGINE_GROUP_COMPUTE_ALL = 0, 1

_u8, _i32, _u32, _u64 = ctypes.c_uint8, ctypes.c_int32, ctypes.c_uint32, ctypes.c_uint64


class _ZeDeviceProperties(ctypes.Structure):          # ze_device_properties_t (ze_api.h)
    _fields_ = [("stype", ctypes.c_int), ("pNext", ctypes.c_void_p), ("type", ctypes.c_int), ("vendorId", _u32),
                ("deviceId", _u32), ("flags", _u32), ("subdeviceId", _u32), ("coreClockRate", _u32),
                ("maxMemAllocSize", _u64), ("maxHardwareContexts", _u32), ("maxCommandQueuePriority", _u32),
                ("numThreadsPerEU", _u32), ("physicalEUSimdWidth", _u32), ("numEUsPerSubslice", _u32),
                ("numSubslicesPerSlice", _u32), ("numSlices", _u32), ("timerResolution", _u64),
                ("timestampValidBits", _u32), ("kernelTimestampValidBits", _u32), ("uuid", _u8 * 16),
                ("name", ctypes.c_char * 256)]


class _ZesDeviceProperties(ctypes.Structure):         # zes_device_properties_t
    _fields_ = [("stype", ctypes.c_int), ("pNext", ctypes.c_void_p), ("core", _ZeDeviceProperties),
                ("numSubdevices", _u32), ("serialNumber", ctypes.c_char * 64), ("boardNumber", ctypes.c_char * 64),
                ("brandName", ctypes.c_char * 64), ("modelName", ctypes.c_char * 64),
                ("vendorName", ctypes.c_char * 64), ("driverVersion", ctypes.c_char * 64)]


class _ZesMemProperties(ctypes.Structure):            # zes_mem_properties_t
    _fields_ = [("stype", ctypes.c_int), ("pNext", ctypes.c_void_p), ("type", ctypes.c_int), ("onSubdevice", _u8),
                ("subdeviceId", _u32), ("location", ctypes.c_int), ("physicalSize", _u64), ("busWidth", _i32),
                ("numChannels", _i32)]


class _ZesMemState(ctypes.Structure):                 # zes_mem_state_t
    _fields_ = [("stype", ctypes.c_int), ("pNext", ctypes.c_void_p), ("health", ctypes.c_int), ("free", _u64),
                ("size", _u64)]


class _ZesPowerProperties(ctypes.Structure):          # zes_power_properties_t
    _fields_ = [("stype", ctypes.c_int), ("pNext", ctypes.c_void_p), ("onSubdevice", _u8), ("subdeviceId", _u32),
                ("canControl", _u8), ("isEnergyThresholdSupported", _u8), ("defaultLimit", _i32),
                ("minLimit", _i32), ("maxLimit", _i32)]


class _ZesPowerEnergyCounter(ctypes.Structure):       # zes_power_energy_counter_t: microjoules, microseconds
    _fields_ = [("energy", _u64), ("timestamp", _u64)]


class _ZesTempProperties(ctypes.Structure):           # zes_temp_properties_t
    _fields_ = [("stype", ctypes.c_int), ("pNext", ctypes.c_void_p), ("type", ctypes.c_int), ("onSubdevice", _u8),
                ("subdeviceId", _u32), ("maxTemperature", ctypes.c_double), ("isCriticalTempSupported", _u8),
                ("isThreshold1Supported", _u8), ("isThreshold2Supported", _u8)]


class _ZesEngineProperties(ctypes.Structure):         # zes_engine_properties_t
    _fields_ = [("stype", ctypes.c_int), ("pNext", ctypes.c_void_p), ("type", ctypes.c_int), ("onSubdevice", _u8),
                ("subdeviceId", _u32)]


class _ZesEngineStats(ctypes.Structure):              # zes_engine_stats_t: active microseconds, timestamp microseconds
    _fields_ = [("activeTime", _u64), ("timestamp", _u64)]


def _call(lib, name, *args) -> bool:
    """One Level Zero call: True on ZE_RESULT_SUCCESS.  Handles are passed as c_void_p (an int would be truncated to
    32 bits without argtypes), structs by reference; a missing entry point or a crash in the call is a failure."""
    try:
        fn = getattr(lib, name)
    except AttributeError:
        return False
    fn.restype = ctypes.c_uint32
    try:
        return fn(*args) == ZE_RESULT_SUCCESS
    except (OSError, ctypes.ArgumentError, ValueError):
        return False


def _load_loader():
    """The Level Zero loader the Intel driver installs (System32\\ze_loader.dll), else one on the PATH (the engine
    folder's bundled copy when the server runs with it); None without one.  Its entry points are __cdecl (ZE_APICALL)."""
    sysdir = os.path.join(os.environ.get("SystemRoot", r"C:\Windows"), "System32", "ze_loader.dll")
    for n in (sysdir, "ze_loader.dll"):
        try:
            return ctypes.CDLL(n)
        except OSError:
            continue
    return None


_L0 = None                                            # (lib, [device handles]) once initialised; False: it failed


def _level_zero():
    """The loader with Sysman initialised and its devices in enumeration order: zesInit + zesDriverGet + zesDeviceGet
    (loader 1.5+; the recommended way); on an older loader ZES_ENABLE_SYSMAN=1 before zeInit, and Sysman on the core
    device handles.  None when there is no loader or it does not answer; tried once per process (the Monitor samples
    every second, and a driver that is not there does not come back without a restart)."""
    global _L0
    if _L0 is not None:
        return _L0 or None
    _L0 = False
    lib = _load_loader()
    if lib is None:
        return None
    try:
        sysman = hasattr(lib, "zesInit")
        if sysman:
            init, drv_get, dev_get = lib.zesInit, lib.zesDriverGet, lib.zesDeviceGet
        else:
            os.environ["ZES_ENABLE_SYSMAN"] = "1"
            init, drv_get, dev_get = lib.zeInit, lib.zeDriverGet, lib.zeDeviceGet
        init.argtypes, init.restype = [ctypes.c_uint32], ctypes.c_uint32
        drv_get.argtypes, drv_get.restype = [ctypes.POINTER(ctypes.c_uint32), ctypes.c_void_p], ctypes.c_uint32
        dev_get.argtypes = [ctypes.c_void_p, ctypes.POINTER(ctypes.c_uint32), ctypes.c_void_p]
        dev_get.restype = ctypes.c_uint32
        if init(0 if sysman else ZE_INIT_FLAG_GPU_ONLY) != ZE_RESULT_SUCCESS:
            return None
        n = ctypes.c_uint32(0)
        if drv_get(ctypes.byref(n), None) != ZE_RESULT_SUCCESS or n.value == 0:
            return None
        drivers = (ctypes.c_void_p * n.value)()
        if drv_get(ctypes.byref(n), drivers) != ZE_RESULT_SUCCESS:
            return None
        devices = []
        for d in drivers:
            if not d:
                continue
            m = ctypes.c_uint32(0)
            if dev_get(ctypes.c_void_p(d), ctypes.byref(m), None) != ZE_RESULT_SUCCESS or m.value == 0:
                continue
            devs = (ctypes.c_void_p * m.value)()
            if dev_get(ctypes.c_void_p(d), ctypes.byref(m), devs) != ZE_RESULT_SUCCESS:
                continue
            devices += [ctypes.c_void_p(h) for h in devs if h]
    except (OSError, AttributeError, ctypes.ArgumentError, ValueError):
        return None
    if not devices:
        return None
    _L0 = (lib, devices)
    return _L0


class _XeSysman:
    """Windows: the Intel Arc's readings from Level Zero Sysman, in NVML's shape (see the module docstring).  index:
    the config's SYCL device index.  Devices that report another type than GPU (an NPU's Level Zero driver, if the
    loader lists one) are skipped, as SYCL's level_zero backend skips them.  unverified on the B70: that order."""

    def __init__(self, index=0):
        self.lib = self.dev = None
        self.name_ = None
        self.mem, self.temps = [], []
        self.pwr = self.eng = None
        self._energy = self._busy = None               # the previous (value, timestamp) sample
        l0 = _level_zero()
        if l0 is None:
            return
        lib, devices = l0
        gpus = []
        for h in devices:
            p = _ZesDeviceProperties()
            p.stype, p.core.stype = ZES_STRUCTURE_TYPE_DEVICE_PROPERTIES, ZE_STRUCTURE_TYPE_DEVICE_PROPERTIES
            got = _call(lib, "zesDeviceGetProperties", h, ctypes.byref(p))
            if got and p.core.type not in (0, ZE_DEVICE_TYPE_GPU):
                continue
            gpus.append((h, p if got else None))
        if not 0 <= index < len(gpus):
            return
        self.lib = lib
        self.dev, p = gpus[index]
        if p is not None:
            name = p.core.name.decode(errors="replace").strip() or p.modelName.decode(errors="replace").strip()
            self.name_ = name or None
        # the device memory modules (not system memory), the card-level power domain (one not on a sub-device, else
        # the first), the sensors with their kind, the engine group that covers everything (else compute, else the first)
        for h in self._enum("zesDeviceEnumMemoryModules"):
            mp = _ZesMemProperties()
            mp.stype = ZES_STRUCTURE_TYPE_MEM_PROPERTIES
            if _call(lib, "zesMemoryGetProperties", h, ctypes.byref(mp)) and mp.location != ZES_MEM_LOC_DEVICE:
                continue
            self.mem.append(h)
        domains = self._enum("zesDeviceEnumPowerDomains")
        for h in domains:
            pp = _ZesPowerProperties()
            pp.stype = ZES_STRUCTURE_TYPE_POWER_PROPERTIES
            if _call(lib, "zesPowerGetProperties", h, ctypes.byref(pp)) and not pp.onSubdevice:
                self.pwr = h
                break
        if self.pwr is None and domains:
            self.pwr = domains[0]
        for h in self._enum("zesDeviceEnumTemperatureSensors"):
            tp = _ZesTempProperties()
            tp.stype = ZES_STRUCTURE_TYPE_TEMP_PROPERTIES
            kind = tp.type if _call(lib, "zesTemperatureGetProperties", h, ctypes.byref(tp)) else ZES_TEMP_SENSORS_GLOBAL
            self.temps.append((h, kind))
        engines = self._enum("zesDeviceEnumEngineGroups")
        kinds = {}
        for h in engines:
            ep = _ZesEngineProperties()
            ep.stype = ZES_STRUCTURE_TYPE_ENGINE_PROPERTIES
            if _call(lib, "zesEngineGetProperties", h, ctypes.byref(ep)):
                kinds.setdefault(ep.type, h)
        for kind in (ZES_ENGINE_GROUP_ALL, ZES_ENGINE_GROUP_COMPUTE_ALL):
            if kind in kinds:
                self.eng = kinds[kind]
                break
        if self.eng is None and engines:
            self.eng = engines[0]

    def _enum(self, name):
        """The handles a zesDeviceEnum* call lists (count first, then the array); [] when none or it fails."""
        n = ctypes.c_uint32(0)
        if not _call(self.lib, name, self.dev, ctypes.byref(n), None) or n.value == 0:
            return []
        arr = (ctypes.c_void_p * n.value)()
        if not _call(self.lib, name, self.dev, ctypes.byref(n), arr):
            return []
        return [ctypes.c_void_p(h) for h in list(arr)[:n.value] if h]

    def ok(self):
        return self.lib is not None and self.dev is not None

    def name(self):
        return self.name_ or "Intel Arc GPU"

    def read(self):
        out = {"util": None, "mem_used": None, "mem_total": None, "temp": None, "power": None, "power_limit": None,
               "vram_temp": None, "pcie_gen": None, "pcie_gen_max": None, "pcie_width": None, "pcie_rx_mb": None,
               "pcie_tx_mb": None}
        if not self.ok():
            return out
        used = total = 0
        got = False
        for h in self.mem:
            st = _ZesMemState()
            st.stype = ZES_STRUCTURE_TYPE_MEM_STATE
            if _call(self.lib, "zesMemoryGetState", h, ctypes.byref(st)) and st.size:
                used += st.size - min(st.free, st.size)
                total += st.size
                got = True
        if got:
            out["mem_used"], out["mem_total"] = used, total
        if self.pwr is not None:                        # microjoules over microseconds = watts, between two samples
            c = _ZesPowerEnergyCounter()
            if _call(self.lib, "zesPowerGetEnergyCounter", self.pwr, ctypes.byref(c)):
                if self._energy and c.timestamp > self._energy[1] and c.energy >= self._energy[0]:
                    out["power"] = (c.energy - self._energy[0]) / (c.timestamp - self._energy[1])
                self._energy = (c.energy, c.timestamp)
        for h, kind in self.temps:                      # the GPU sensor, else the global one; memory on its own
            t = ctypes.c_double()
            if not _call(self.lib, "zesTemperatureGetState", h, ctypes.byref(t)):
                continue
            if kind == ZES_TEMP_SENSORS_MEMORY:
                out["vram_temp"] = t.value
            elif kind == ZES_TEMP_SENSORS_GPU or (kind == ZES_TEMP_SENSORS_GLOBAL and out["temp"] is None):
                out["temp"] = t.value
        if self.eng is not None:                        # busy share of the time between two samples
            s = _ZesEngineStats()
            if _call(self.lib, "zesEngineGetActivity", self.eng, ctypes.byref(s)):
                if self._busy and s.timestamp > self._busy[1]:
                    out["util"] = max(0.0, min(100.0, 100.0 * (s.activeTime - self._busy[0])
                                               / (s.timestamp - self._busy[1])))
                self._busy = (s.activeTime, s.timestamp)
        return out


def xe_gpu(index=0):
    """The platform's Intel reader for the card SYCL numbers `index`."""
    return _XeSysman(index) if WIN else _XeGpu(index)
