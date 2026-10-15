"""serve/telemetry.py - hardware readings for the web app's Monitor tab (idea from PR #22 by code-martin).

A background thread samples once a second and keeps the last 60 readings of each series for the sparklines:
- GPU: NVIDIA's own NVML library (nvml.dll / libnvidia-ml.so.1, installed with every driver) through ctypes, so no
  pip package is needed: load, VRAM, temperature, power, PCIe link and throughput.  With the AMD backend (#301): the
  amdgpu driver's Linux sysfs files - load, VRAM, temperature and power.
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
        out["power_limit"] = float(lim) if lim else None
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


# ------------------------------------------------------------------------------------------------ AMD (Windows ADL2)
# The AMD card's readings on Windows come from the driver's own AMD Display Library (atiadlxx.dll), the same
# interface HWInfo and LibreHardwareMonitor use, so no ROCm or pip package is needed.  ADL2's PMLog (Overdrive 8)
# gives load, temperature and power; DedicatedVRAMUsage + MemoryInfoX4 give VRAM.  ctypes only; anything that
# cannot be read is None and nothing here can stop the server.  The ADL2 call sequence and struct layout follow
# LibreHardwareMonitor's AtiAdlxx.cs (the ABI varies by Adrenalin version, so every call is guarded).

class SingleSensor(ctypes.Structure):
    """One PMLog sensor reading: whether the card reports it, and its raw value."""
    _fields_ = [("supported", ctypes.c_int), ("value", ctypes.c_int)]


class _Adl:
    """The ADL2 ctypes bindings, kept module-level so the structs are built once.  None when atiadlxx.dll is
    absent (never on a non-Windows box, since gpu_reader only builds _AmdWin there)."""

    class AdapterInfo(ctypes.Structure):
        _fields_ = [("Size", ctypes.c_int), ("AdapterIndex", ctypes.c_int),
                    ("UDID", ctypes.c_char * 256), ("BusNumber", ctypes.c_int),
                    ("DeviceNumber", ctypes.c_int), ("FunctionNumber", ctypes.c_int),
                    ("VendorID", ctypes.c_int), ("AdapterName", ctypes.c_char * 256),
                    ("DisplayName", ctypes.c_char * 256), ("Present", ctypes.c_int),
                    ("Exist", ctypes.c_int), ("DriverPath", ctypes.c_char * 256),
                    ("DriverPathExt", ctypes.c_char * 256), ("PNPString", ctypes.c_char * 256),
                    ("OSDisplayIndex", ctypes.c_int)]

    class GcnInfo(ctypes.Structure):
        _fields_ = [("CuCount", ctypes.c_int), ("TexCount", ctypes.c_int),
                    ("RopCount", ctypes.c_int), ("ASICFamilyId", ctypes.c_int),
                    ("ASICRevisionId", ctypes.c_int)]

    class MemInfoX4(ctypes.Structure):
        _fields_ = [("iMemorySize", ctypes.c_longlong), ("strMemoryType", ctypes.c_char * 256),
                    ("iMemoryBandwidth", ctypes.c_longlong), ("iHyperMemorySize", ctypes.c_longlong),
                    ("iInvisibleMemorySize", ctypes.c_longlong), ("iVisibleMemorySize", ctypes.c_longlong),
                    ("iVramVendorRevId", ctypes.c_longlong), ("iMemoryBandwidthX2", ctypes.c_longlong),
                    ("iMemoryBitRateX2", ctypes.c_longlong)]

    class PmLogSupport(ctypes.Structure):
        _fields_ = [("usSensors", ctypes.c_ushort * 256), ("iReserved", ctypes.c_int * 16)]

    class PmLogStartInput(ctypes.Structure):
        _fields_ = [("usSensors", ctypes.c_ushort * 256), ("ulSampleRate", ctypes.c_uint),
                    ("iReserved", ctypes.c_int * 15)]

    class PmLogStartOutput(ctypes.Structure):
        _fields_ = [("pLoggingAddress", ctypes.c_void_p)]

    class PmLogDataOutput(ctypes.Structure):
        _fields_ = [("size", ctypes.c_int), ("sensors", SingleSensor * 256)]

    # PMLog sensor ids (LibreHardwareMonitor's ADLPMLogSensors enum).
    S_EDGE, S_MEM, S_ASIC_POWER, S_ACTIVITY_GFX, S_ACTIVITY_MEM = 8, 9, 23, 19, 20
    S_GFX_POWER, S_BUS_SPEED, S_BUS_LANES, S_TOTAL_POWERLIMIT = 30, 40, 41, 47
    S_BOARD_POWER = 73

    def _alloc(self, size):
        # return a raw heap pointer ADL can free(); ctypes keeps the int alive as a void_p return
        p = self.crt.malloc(size) if size else None
        return ctypes.cast(p, ctypes.c_void_p).value if p else None

    def __init__(self):
        self.lib = None
        if os.name != "nt":
            return
        try:
            self.lib = ctypes.CDLL("atiadlxx.dll")
            # ADL2_Main_Control_Create calls this callback to allocate, and frees the memory on Destroy.  atiadlxx
            # is a modern MSVC DLL that links ucrtbase.dll, so malloc from ucrtbase shares the exact heap ADL
            # frees with (LibreHardwareMonitor's AllocHGlobal reaches the same UCRT heap).  If ucrtbase is not
            # reachable, fall back to the process's own C runtime; if neither works, leave ADL off (lib = None).
            try:
                self.crt = ctypes.CDLL("ucrtbase.dll")
            except OSError:
                self.crt = ctypes.CDLL(None)
            self._alloc_keep = self._alloc
            self.ALLOC = ctypes.CFUNCTYPE(ctypes.c_void_p, ctypes.c_int)(self._alloc_keep)
        except (OSError, TypeError, ValueError):
            self.lib = None

    def ok(self):
        return self.lib is not None


class _AmdWin:
    """#... : an AMD card's readings on Windows through ADL2's PMLog, with _Nvml's interface: util, mem_used,
    mem_total, temp, power, power_limit, and best-effort pcie_gen_max / pcie_width from Device Manager."""

    def __init__(self, index=0):
        self.adl = _Adl()
        self.lib = self.adl.lib
        self.ctx = None
        self.adapter = None
        self.card_count = 0
        self.dev_code = None
        self.device = ctypes.c_uint(0)
        self.data = None
        if self.lib is None:
            return
        self._setup(index)

    def _call(self, fn, *args):
        try:
            return fn(*args)
        except (AttributeError, OSError):
            return None

    def _setup(self, index):
        ctx = ctypes.c_void_p()
        if self._call(self.lib.ADL2_Main_Control_Create, self.adl.ALLOC, 1, ctypes.byref(ctx)) != 0:
            return
        self.ctx = ctx
        n = ctypes.c_int(0)
        if self._call(self.lib.ADL2_Adapter_NumberOfAdapters_Get, ctx, ctypes.byref(n)) != 0:
            return
        count = max(1, min(n.value, 40))
        buf = (self.adl.AdapterInfo * count)()
        if self._call(self.lib.ADL2_Adapter_AdapterInfo_Get, ctx, ctypes.cast(buf, ctypes.c_void_p),
                      ctypes.sizeof(buf)) != 0:
            return
        # every present discrete card, deduped by its DEV_xxxx code (ADL lists each card once per display).
        # Active_Get only marks the display-driving card, so it would hide a second idle card - the Monitor
        # wants a tile per card, so enumerate present discrete cards instead (skip the integrated Radeon).
        cards, seen = [], set()
        for i in range(count):
            nm = buf[i].AdapterName
            if not nm or b"Graphics" in nm:                       # integrated Radeon or empty
                continue
            if not (buf[i].Present or buf[i].Exist):              # not plugged in
                continue
            udid = buf[i].UDID or b""
            key = udid.split(b"DEV_", 1)[1].split(b"&", 1)[0] if b"DEV_" in udid else nm
            if key in seen:
                continue
            seen.add(key)
            cards.append(i)
        if not 0 <= index < len(cards):
            return
        self.adapter = cards[index]
        self.card_count = len(cards)
        # the card's DEV_xxxx code, to match its Device Manager Plug-and-Play record (the real PCI link)
        udid = buf[self.adapter].UDID or b""
        self.dev_code = udid.split(b"DEV_", 1)[1].split(b"&", 1)[0].decode(errors="replace") if b"DEV_" in udid else None
        # AdapterName is the friendly card name ("AMD Radeon AI PRO R9700"); DisplayName is just "\.\DISPLAYn".
        self.display_name = (buf[self.adapter].AdapterName or buf[self.adapter].DisplayName or b"").decode(
            errors="replace").strip() or None

    def ok(self):
        return self.lib is not None and self.adapter is not None

    def name(self):
        if not self.ok():
            return "AMD Radeon"
        return self.display_name or "AMD Radeon"

    def _read_pmlog(self):
        if not self.ok():
            return None
        if self.data is None:
            self.data = self.adl.PmLogDataOutput()
        if self._call(self.lib.ADL2_New_QueryPMLogData_Get, self.ctx, ctypes.c_int(self.adapter),
                      ctypes.byref(self.data)) != 0:
            return None
        return self.data

    def _start_pmlog(self):
        if not self.ok() or self.device.value == 0:
            return
        if self._call(self.lib.ADL2_Device_PMLog_Device_Create, self.ctx, ctypes.c_int(self.adapter),
                      ctypes.byref(self.device)) != 0:
            return
        sup = self.adl.PmLogSupport()
        if self._call(self.lib.ADL2_Adapter_PMLog_Support_Get, self.ctx, ctypes.c_int(self.adapter),
                      ctypes.byref(sup)) != 0:
            self.device = ctypes.c_uint(0)
            return
        inp = self.adl.PmLogStartInput()
        for k in range(256):
            inp.usSensors[k] = sup.usSensors[k]
        inp.ulSampleRate = 1000
        out = self.adl.PmLogStartOutput()
        if self._call(self.lib.ADL2_Adapter_PMLog_Start, self.ctx, ctypes.c_int(self.adapter),
                      ctypes.byref(inp), ctypes.byref(out), self.device) != 0:
            self.device = ctypes.c_uint(0)

    def close(self):
        """Give back what the driver lent this reader: the PMLog device handle and the ADL context.  A one-off read
        (free_vram_mib, the startup probes, amd_card_count) builds a reader per call, and an ADL context that is
        never destroyed is a handle that stays: #1434 measured +2 handles per free_vram_mib() call on a 7900 XTX
        before its close().  Safe to call twice and on a reader whose _setup failed part way (the context is
        destroyed only when it exists); a closed reader simply reads nothing."""
        if self.ctx is not None:
            if self.device.value and self.adapter is not None:
                self._call(self.lib.ADL2_Device_PMLog_Device_Destroy, self.ctx, ctypes.c_int(self.adapter),
                           self.device)
            self.device = ctypes.c_uint(0)
            self._call(self.lib.ADL2_Main_Control_Destroy, self.ctx)
            self.ctx = None
        self.adapter = None
        self.card_count = 0

    def __del__(self):
        try:
            self.close()      # the sampler's readers go away with the server; a finalizer must never raise
        except Exception:  # noqa: BLE001 - interpreter shutdown can leave the driver DLL unloaded already
            pass

    def read(self):
        out = {}
        if not self.ok():
            return out
        d = self._read_pmlog()
        if d is None:
            self._start_pmlog()
            d = self._read_pmlog()
        lanes = None
        if d is not None:
            def val(sid):
                s = d.sensors[sid]
                return s.value if s.supported else None
            gfx, mem = val(self.adl.S_ACTIVITY_GFX), val(self.adl.S_ACTIVITY_MEM)
            u = [x for x in (gfx, mem) if x is not None]
            out["util"] = max(u) if u else None
            t = val(self.adl.S_EDGE)
            if t is None:
                t = val(self.adl.S_MEM)
            out["temp"] = float(t) if t is not None else None
            # RDNA3+ report power as BOARD_POWER (id 73); older cards ASIC_POWER (23) or GFX_POWER (30).
            # LibreHardwareMonitor reads these with factor 1.0, so the raw value is already watts.
            p = val(self.adl.S_BOARD_POWER)
            if p is None:
                p = val(self.adl.S_ASIC_POWER)
            if p is None:
                p = val(self.adl.S_GFX_POWER)
            out["power"] = float(p) if p else None
            lim = val(self.adl.S_TOTAL_POWERLIMIT)
            out["power_limit"] = float(lim) if lim else None
            lanes = val(self.adl.S_BUS_LANES)
        # VRAM: used from DedicatedVRAMUsage (MB), total from MemoryInfoX4 (bytes)
        used = ctypes.c_int(0)
        if self._call(self.lib.ADL2_Adapter_DedicatedVRAMUsage_Get, self.ctx, ctypes.c_int(self.adapter),
                      ctypes.byref(used)) == 0 and used.value > 0:
            out["mem_used"] = used.value * 1024 * 1024
        mi = self.adl.MemInfoX4()
        if self._call(self.lib.ADL2_Adapter_MemoryInfoX4_Get, self.ctx, ctypes.c_int(self.adapter),
                      ctypes.byref(mi)) == 0 and mi.iMemorySize > 0:
            out["mem_total"] = int(mi.iMemorySize)
        # PCIe link: the slot's bus speed (generation) from Device Manager's Plug-and-Play record
        # (DEVPKEY_PciDevice_MaxLinkSpeed), matched by the card's DEV_xxxx code.  ADL PMLog's BUS_SPEED is
        # NOT the slot link - it is the GPU's own internal bus, which Adrenalin power-saves down to Gen4
        # while the card is idle, so it reads 4 even on a Gen5 slot.  The negotiated width comes from PMLog
        # BUS_LANES (the lanes the card actually runs at - what a motherboard's PCI bifurcation limits);
        # Device Manager's CurrentLinkWidth reports the slot's x16 capability, not the negotiated width.
        # Measured on a B650E: RX 9070 XT Gen5 x8, AI PRO R9700 Gen5 x4 (HWInfo "PCIe v5.0 x16 @ x4").
        dm = _dm_link_map()
        if self.dev_code and self.dev_code in dm:
            cur_speed, cur_width, max_speed, max_width = dm[self.dev_code]
            if max_speed:
                out["pcie_gen_max"] = max_speed
            if cur_speed and max_speed and cur_speed < max_speed:
                out["pcie_gen"] = cur_speed
        if lanes:
            out["pcie_width"] = lanes
        elif self.dev_code and self.dev_code in dm and dm[self.dev_code][1]:
            out["pcie_width"] = dm[self.dev_code][1]
        return out


_DM_LINK = None  # cached {DEV_xxxx code: (current_speed, current_width, max_speed, max_width)} from Device Manager


def _dm_link_map():
    """Every present PCI device's link speed and width from Windows Device Manager's own Plug-and-Play records
    (DEVPKEY_PciDevice_CurrentLinkSpeed / CurrentLinkWidth and MaxLinkSpeed / MaxLinkWidth), keyed by the card's
    DEV_xxxx code so each GPU tile shows its own slot's link.  This is the real PCI link (what a motherboard's
    PCI bifurcation limits), unlike ADL PMLog's BUS_SPEED / BUS_LANES which report the GPU's internal bus.  Queried
    once and cached, so the one-time subprocess cost is not on the per-second sample path; {} when unavailable."""
    global _DM_LINK
    if _DM_LINK is not None:
        return _DM_LINK
    if os.name != "nt":
        _DM_LINK = {}
        return _DM_LINK
    import subprocess
    ps = '''
Get-PnpDevice -Class Display -ErrorAction SilentlyContinue |
  Where-Object { $_.Status -ne 'Not Connected' } |
  ForEach-Object {
    $id = $_.InstanceId
    if ($id -match 'DEV_([0-9A-Fa-f]{4})') { $dev = $matches[1] } else { return }
    $cs = (Get-PnpDeviceProperty -InstanceId $id -KeyName 'DEVPKEY_PciDevice_CurrentLinkSpeed' -ErrorAction SilentlyContinue).Data
    $cw = (Get-PnpDeviceProperty -InstanceId $id -KeyName 'DEVPKEY_PciDevice_CurrentLinkWidth' -ErrorAction SilentlyContinue).Data
    $ms = (Get-PnpDeviceProperty -InstanceId $id -KeyName 'DEVPKEY_PciDevice_MaxLinkSpeed' -ErrorAction SilentlyContinue).Data
    $mw = (Get-PnpDeviceProperty -InstanceId $id -KeyName 'DEVPKEY_PciDevice_MaxLinkWidth' -ErrorAction SilentlyContinue).Data
    Write-Output "DEV_$dev $cs $cw $ms $mw"
  }
'''
    try:
        out = subprocess.run(["powershell", "-NoLogo", "-NonInteractive", "-Command", ps],
                             capture_output=True, text=True, timeout=10)
    except (OSError, subprocess.SubprocessError):
        return {}  # a failed query is not cached - the next sample tries again (a one-off PowerShell
        # failure must not blank every card's PCIe link for the life of the server)
    m = {}
    for line in out.stdout.splitlines():
        parts = line.split()
        if len(parts) == 5 and parts[0].startswith("DEV_"):
            try:
                cs, cw, ms, mw = (int(parts[1]), int(parts[2]), int(parts[3]), int(parts[4]))
            except ValueError:
                continue
            m[parts[0][4:]] = (cs, cw, ms, mw)
    if not m and out.returncode != 0:
        return {}  # PowerShell errored (or printed nothing usable): retry next sample, don't cache
    _DM_LINK = m
    return m


def amd_card_count():
    """How many present discrete AMD cards ADL lists (so the Monitor can show a tile per card on Windows).
    0 when there is no AMD driver here.  Reads card 0, whose _setup counts every present discrete card."""
    if os.name != "nt":
        return 0
    reader = _AmdWin(0)
    n = getattr(reader, "card_count", 0)
    reader.close()          # a throwaway reader: its ADL context goes back now, not at process exit
    return n


def gpu_reader(index=0, amd=False):
    """The card's readings: NVML (NVIDIA), the amdgpu sysfs files with the AMD backend on Linux (#301), or the
    driver's ADL2 PMLog with the AMD backend on Windows."""
    if amd:
        return _AmdWin(index) if os.name == "nt" else _Amd(index)
    return _Nvml(index)


_DISPLAY_CLASS = r"SYSTEM\CurrentControlSet\Control\Class\{4d36e968-e325-11ce-bfc1-08002be10318}"


def windows_display_adapters(vendor=None) -> list:
    """#1587: the display adapters Windows has drivers for, from the registry's display class (no subprocess, no vendor
    library): [{"name", "provider", "driver", "vram_bytes"}] in the class's order.  `vendor`: "amd", "nvidia" or
    "intel" keeps that vendor's cards.  [] off Windows or when nothing can be read."""
    if os.name != "nt":
        return []
    try:
        import winreg
        root = winreg.OpenKey(winreg.HKEY_LOCAL_MACHINE, _DISPLAY_CLASS)
    except OSError:
        return []
    out = []
    i = 0
    while True:
        try:
            sub = winreg.EnumKey(root, i)
        except OSError:
            break
        i += 1
        if not sub.isdigit():
            continue
        try:
            k = winreg.OpenKey(root, sub)
        except OSError:
            continue

        def val(name, k=k):
            try:
                return winreg.QueryValueEx(k, name)[0]
            except OSError:
                return None
        name = val("DriverDesc")
        if not isinstance(name, str) or not name.strip():
            continue
        mem = val("HardwareInformation.qwMemorySize")
        if mem is None:
            mem = val("HardwareInformation.MemorySize")
        if isinstance(mem, (bytes, bytearray)):
            mem = int.from_bytes(bytes(mem[:8]), "little") if mem else None
        out.append({"name": name.strip(), "provider": str(val("ProviderName") or ""), "driver": val("DriverVersion"),
                    "vram_bytes": mem if isinstance(mem, int) and mem > 0 else None})
    if vendor:
        keys = {"amd": ("advanced micro devices", "amd", "radeon"), "nvidia": ("nvidia",), "intel": ("intel",)}[vendor]
        out = [a for a in out if any(s in (a["provider"] + " " + a["name"]).lower() for s in keys)]
    return out


def windows_amd_cards(adapters=None) -> list:
    """The AMD cards HIP runs on, as windows_display_adapters lists them.  A Ryzen's integrated Radeon is listed too
    (512 MiB on a Ryzen 5 7600): beside a card with 2 GiB or more it is left out (an APU alone keeps it)."""
    cards = windows_display_adapters("amd") if adapters is None else adapters
    return [c for c in cards if (c["vram_bytes"] or 0) >= 2 << 30] or cards


def free_vram_mib(index=0, amd=False):
    """Free VRAM of a card in MiB, or None when it cannot be read.  The reader is thrown away, so its driver
    handles (ADL context, PMLog device) are released here rather than left for the process to end with: the
    server calls this in a 0.5 s retry loop while it waits for VRAM."""
    g = gpu_reader(index, amd)
    try:
        if not g.ok():
            return None
        r = g.read()
        if r.get("mem_total") is None or r.get("mem_used") is None:
            return None
        return int((r["mem_total"] - r["mem_used"]) >> 20)
    finally:
        close = getattr(g, "close", None)      # NVML has no cleanup; the AMD readers do
        if close is not None:
            close()


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
        # Windows AMD: the Monitor shows a tile per card, so read every present discrete card, not just the
        # one the engine runs on (a second idle card still has load/VRAM/temp/power/PCIe worth showing).
        # HIP numbers all adapters (the integrated one included), ADL enumerates only the discrete ones,
        # so the engine's HIP indices do not match _AmdWin's - enumerate by card count instead.
        if amd and os.name == "nt":
            n = amd_card_count()
            if n:
                idx = list(range(n))
        self.gpus = [(i, gpu_reader(i, amd)) for i in idx]
        self.gpus = [(i, g) for i, g in self.gpus if g.ok()] or self.gpus[:1]
        self.gpu = self.gpus[0][1]
        try:
            import psutil  # noqa: F401
            self.ps = sys.modules["psutil"]
        except ImportError:
            self.ps = None
        self.fallback = _CpuRamFallback()
        gpu_name = " + ".join(g.name() or "?" for _, g in self.gpus) if self.gpu.ok() else None
        win_amd = None
        if gpu_name is None and amd and os.name == "nt":
            # #1587: no readings for an AMD card on Windows yet, but its name (and VRAM size) are in the registry, so the
            # dashboard and strata_bench do not call the run "CPU only".  The class order is taken as HIP's order.
            cards = windows_amd_cards()
            win_amd = [cards[i] for i in idx if 0 <= i < len(cards)] or cards[:1]
            gpu_name = " + ".join(c["name"] for c in win_amd) if win_amd else None
        self.static = {
            "gpu_name": gpu_name,
            "gpu_vram_total_mib": (sum(c["vram_bytes"] for c in win_amd) >> 20
                                   if win_amd and all(c["vram_bytes"] for c in win_amd) else None),
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
            reads = [(i, g, g.read()) for i, g in self.gpus]
            g = dict(reads[0][2])
            if len(reads) > 1:
                def vals(k):
                    return [r[k] for _, _, r in reads if r.get(k) is not None]
                for k in ("mem_used", "mem_total", "power", "power_limit", "pcie_rx_mb", "pcie_tx_mb"):
                    v = vals(k)
                    g[k] = sum(v) if v else None
                u = vals("util")
                g["util"] = sum(u) / len(u) if u else None
                t = vals("temp")
                g["temp"] = max(t) if t else None
            # every card's own readings, so the Monitor can show a tile per GPU (name, load, VRAM, temp, power,
            # and each card's own PCIe link - a motherboard's PCI bifurcation gives the slots different widths)
            s["gpus"] = [{"index": i, "name": rd.name(), "util": r.get("util"), "mem_used": r.get("mem_used"),
                          "mem_total": r.get("mem_total"), "temp": r.get("temp"), "power": r.get("power"),
                          "pcie_gen": r.get("pcie_gen_max"), "pcie_width": r.get("pcie_width")}
                         for i, rd, r in reads]
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
