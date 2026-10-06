"""Read Windows Radeon sensors through the driver's ADL library; no extra packages.

The ctypes layouts and sensor IDs follow AMD's display-library/include headers:
https://github.com/GPUOpen-LibrariesAndSDKs/display-library
Only query APIs are used; no clock, power or fan settings are changed.
"""
import ctypes as c
import os


class Adapter(c.Structure):
    _fields_ = [("size", c.c_int), ("index", c.c_int), ("udid", c.c_char * 256),
                ("bus", c.c_int), ("device", c.c_int), ("function", c.c_int), ("vendor", c.c_int),
                ("name", c.c_char * 256), ("display", c.c_char * 256), ("present", c.c_int),
                ("exist", c.c_int), ("path", c.c_char * 256), ("path_ext", c.c_char * 256),
                ("pnp", c.c_char * 256), ("os_index", c.c_int)]


class Memory(c.Structure):
    _fields_ = [("size", c.c_longlong), ("kind", c.c_char * 256), ("bandwidth", c.c_longlong),
                ("hyper", c.c_longlong), ("invisible", c.c_longlong), ("visible", c.c_longlong)]


class Sensor(c.Structure):
    _fields_ = [("supported", c.c_int), ("value", c.c_int)]


class Metrics(c.Structure):
    _fields_ = [("size", c.c_int), ("sensors", Sensor * 256)]


class Chipset(c.Structure):
    _fields_ = [("bus_type", c.c_int), ("speed_type", c.c_int), ("max_width", c.c_int),
                ("width", c.c_int), ("agp_speeds", c.c_int), ("agp_speed", c.c_int)]


def _hip_pci(index):
    """Match HIP's device number to ADL by PCI address, including PCs with an integrated Radeon."""
    system = os.path.join(os.environ.get("SystemRoot", r"C:\Windows"), "System32")
    for dll in ("amdhip64_7.dll", "amdhip64_6.dll"):
        try:
            hip = c.CDLL(os.path.join(system, dll))
            get = hip.hipDeviceGetPCIBusId
            get.argtypes = [c.c_char_p, c.c_int, c.c_int]
            get.restype = c.c_int
            buf = c.create_string_buffer(32)
            if get(buf, len(buf), index) == 0:
                _, bus, slot = buf.value.decode().split(":")
                device, function = slot.split(".")
                return int(bus, 16), int(device, 16), int(function, 16)
        except (AttributeError, OSError, ValueError):
            continue
    return None


class AmdWindows:
    def __init__(self, index=0):
        self.lib, self.dev = None, None
        self.context = c.c_void_p()
        self._buffers = []
        self._name = None
        try:
            system = os.path.join(os.environ.get("SystemRoot", r"C:\Windows"), "System32")
            self.lib = c.CDLL(os.path.join(system, "atiadlxx.dll" if c.sizeof(c.c_void_p) == 8 else "atiadlxy.dll"))
            # ADL retains the allocator: keep both the callback and its buffers alive until context destruction.
            buffers = self._buffers
            def allocate(size):
                buf = c.create_string_buffer(size)
                buffers.append(buf)
                return c.addressof(buf)
            self._allocate = c.CFUNCTYPE(c.c_void_p, c.c_int)(allocate)
            create = self.lib.ADL2_Main_Control_Create
            create.argtypes = [c.CFUNCTYPE(c.c_void_p, c.c_int), c.c_int, c.POINTER(c.c_void_p)]
            create.restype = c.c_int
            if create(self._allocate, 1, c.byref(self.context)) != 0:
                self.close()
                return
            count = c.c_int()
            if self._call("ADL2_Adapter_NumberOfAdapters_Get", c.byref(count)) != 0 or not 0 < count.value < 256:
                self.close()
                return
            adapters = (Adapter * count.value)()
            for a in adapters:
                a.size = c.sizeof(Adapter)
            if self._call("ADL2_Adapter_AdapterInfo_Get", adapters, c.c_int(c.sizeof(adapters))) != 0:
                self.close()
                return
            physical = {}
            for a in adapters:
                if a.vendor in (1002, 0x1002) and a.present and a.exist:
                    physical.setdefault((a.bus, a.device, a.function), a)
            pci = _hip_pci(index)
            # Without HIP, only a single physical card has an unambiguous device number.
            a = physical.get(pci) if pci is not None else None
            if pci is None and index == 0 and len(physical) == 1:
                a = next(iter(physical.values()))
            if a is None:
                self.close()
                return
            self.dev = a.index
            self._name = a.name.decode(errors="replace")
        except (AttributeError, OSError):
            self.close()

    def _call(self, name, *args):
        try:
            fn = getattr(self.lib, name)
            fn.restype = c.c_int
            return fn(self.context, *args)
        except (AttributeError, OSError):
            return -1

    def close(self):
        if self.context.value:
            self._call("ADL2_Main_Control_Destroy")
            self.context = c.c_void_p()
        self.dev = None
        self._buffers.clear()

    def __del__(self):
        self.close()

    def ok(self):
        return self.dev is not None

    def name(self):
        return self._name

    def read(self):
        out = {}
        if not self.ok():
            return out
        dev = c.c_int(self.dev)
        mem = Memory()
        if self._call("ADL2_Adapter_MemoryInfo2_Get", dev, c.byref(mem)) == 0:
            out["mem_total"] = mem.size
        used = c.c_int()
        # WDDM dedicated usage, rather than ADL's allocation accounting (which can exceed physical VRAM).
        if self._call("ADL2_Adapter_DedicatedVRAMUsage_Get", dev, c.byref(used)) == 0 and used.value >= 0:
            value = used.value * 2**20
            if out.get("mem_total") is None or value <= out["mem_total"]:
                out["mem_used"] = value
        metrics = Metrics()
        metrics.size = c.sizeof(metrics)
        if self._call("ADL2_New_QueryPMLogData_Get", dev, c.byref(metrics)) == 0:
            def sensor(*ids):
                return next((metrics.sensors[i].value for i in ids if metrics.sensors[i].supported), None)
            out.update(util=sensor(19), temp=sensor(8), power=sensor(73, 23))
            gen, width = sensor(40), sensor(41)
            out["pcie_gen"] = gen if gen is not None and 1 <= gen <= 6 else None
            out["pcie_width"] = width if width is not None and width > 0 else None
        chip = Chipset()
        if self._call("ADL2_Adapter_ChipSetInfo_Get", dev, c.byref(chip)) == 0:
            # ADL_BUSTYPE_PCIE .. PCIE_GEN5 are 2..6.
            if 2 <= chip.speed_type <= 6:
                out["pcie_gen_max"] = chip.speed_type - 1
            if chip.width > 0 and not out.get("pcie_width"):
                out["pcie_width"] = chip.width
        return out
