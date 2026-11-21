"""Read-only Vulkan facts for ``strata doctor`` (the Arc B390 research path).

The native Vulkan engine selects the first ``VK_PHYSICAL_DEVICE_TYPE_INTEGRATED_GPU``
from vendor 0x8086, asks for Vulkan 1.3, and needs a host-visible host-*cached*
memory type for its mapped expert slots and logits. On the B390 that type is 0xf
(DEVICE_LOCAL | HOST_VISIBLE | HOST_COHERENT | HOST_CACHED); using the plain 0x7
type instead costs about 2.4x in decode speed. This module answers those facts
without starting an engine, loading a model or allocating GPU memory: it opens
the Vulkan loader, enumerates physical devices, and reads properties, extensions
and memory heaps.

Everything here is deliberately bounded and read-only. A malformed driver answer
must not make the doctor crash, and no Vulkan command is issued that submits
work or changes driver state. The ctypes declarations mirror vulkan_core.h; the
constants are the current header values (1000059001, 50, 1000225000), not the
older small numbers some online examples use.
"""

from __future__ import annotations

import ctypes as C
import ctypes.util
import os
import platform
import sys
from pathlib import Path

# Vulkan header values (VK_HEADER_VERSION 363).
VK_STRUCTURE_TYPE_APPLICATION_INFO = 0
VK_STRUCTURE_TYPE_INSTANCE_CREATE_INFO = 1
VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_PROPERTIES_2 = 1000059001
VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_VULKAN_1_1_PROPERTIES = 50
VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_SUBGROUP_SIZE_CONTROL_PROPERTIES_EXT = 1000225000

# Property bits (VkMemoryPropertyFlagBits / VkMemoryHeapFlagBits).
MEMORY_DEVICE_LOCAL = 0x00000001
MEMORY_HOST_VISIBLE = 0x00000002
MEMORY_HOST_COHERENT = 0x00000004
MEMORY_HOST_CACHED = 0x00000008

DEVICE_TYPES = {0: "other", 1: "integrated_gpu", 2: "discrete_gpu", 3: "virtual_gpu", 4: "cpu"}
QUEUE_COMPUTE = 0x00000002

# Extensions the engine or its shaders actually care about. Everything else is
# counted but not listed, so a doctor report stays small on a 200-extension ICD.
INTERESTING_EXTENSIONS = (
    "VK_EXT_subgroup_size_control",
    "VK_KHR_cooperative_matrix",
    "VK_KHR_shader_float16_int8",
    "VK_KHR_portability_subset",
    "VK_EXT_shader_subgroup_ballot",
    "VK_EXT_shader_subgroup_vote",
)

# The engine's default 2048 slots at the model's largest expert tensor
# (860,160 B down expert, aligned) are about 1.6 GiB. Keep a little headroom in
# the estimate so a plan never over-commits shared memory.
SLOT_BYTES = 860160
DEFAULT_SLOTS = 2048
MIN_SLOTS = 64
# Leave this much of MemAvailable for the OS, the page cache and other programs.
MEMORY_HEADROOM_MIB = 4096


class _VkApplicationInfo(C.Structure):
    _fields_ = [("sType", C.c_int32), ("pNext", C.c_void_p),
                ("pApplicationName", C.c_char_p), ("applicationVersion", C.c_uint32),
                ("pEngineName", C.c_char_p), ("engineVersion", C.c_uint32),
                ("apiVersion", C.c_uint32)]


class _VkInstanceCreateInfo(C.Structure):
    _fields_ = [("sType", C.c_int32), ("pNext", C.c_void_p), ("flags", C.c_uint32),
                ("pApplicationInfo", C.POINTER(_VkApplicationInfo)),
                ("enabledLayerCount", C.c_uint32), ("ppEnabledLayerNames", C.POINTER(C.c_char_p)),
                ("enabledExtensionCount", C.c_uint32), ("ppEnabledExtensionNames", C.POINTER(C.c_char_p))]


class _VkPropertiesHead(C.Structure):
    """The leading fields of VkPhysicalDeviceProperties, read from a byte buffer.

    VkPhysicalDeviceProperties embeds a large, version-specific ``limits`` struct
    we do not need. The base fields have had the same offsets since Vulkan 1.0,
    so we read only those and let the driver write the rest into a spare buffer.
    """

    _fields_ = [("sType", C.c_int32), ("pNext", C.c_void_p),
                ("apiVersion", C.c_uint32), ("driverVersion", C.c_uint32),
                ("vendorID", C.c_uint32), ("deviceID", C.c_uint32),
                ("deviceType", C.c_int32), ("deviceName", C.c_char * 256)]


class _VkProperties2Head(C.Structure):
    _fields_ = [("sType", C.c_int32), ("pNext", C.c_void_p)]


class _VkVulkan11Properties(C.Structure):
    # VK_MAX_UUID_SIZE 16, VK_MAX_LUID_SIZE 8. The field order is exactly the
    # header's; an extra or missing field here shifts subgroupSize and silently
    # returns 0.
    _fields_ = [("sType", C.c_int32), ("pNext", C.c_void_p),
                ("deviceUUID", C.c_uint8 * 16), ("driverUUID", C.c_uint8 * 16),
                ("deviceLUID", C.c_uint8 * 8), ("deviceNodeMask", C.c_uint32),
                ("deviceLUIDValid", C.c_uint32), ("subgroupSize", C.c_uint32),
                ("subgroupSupportedStages", C.c_uint32), ("subgroupSupportedOperations", C.c_uint32),
                ("subgroupQuadOperationsInAllStages", C.c_uint32),
                ("pointClippingBehavior", C.c_int32), ("maxMultiviewViewCount", C.c_uint32),
                ("maxMultiviewInstanceIndex", C.c_uint32), ("protectedNoFault", C.c_uint32),
                ("maxPerSetDescriptors", C.c_uint32), ("maxMemoryAllocationSize", C.c_uint64)]


class _VkSubgroupSizeControlProperties(C.Structure):
    _fields_ = [("sType", C.c_int32), ("pNext", C.c_void_p),
                ("minSubgroupSize", C.c_uint32), ("maxSubgroupSize", C.c_uint32),
                ("maxComputeWorkgroupSubgroups", C.c_uint32), ("requiredSubgroupSizeStages", C.c_uint32)]


class _VkMemoryType(C.Structure):
    _fields_ = [("propertyFlags", C.c_uint32), ("heapIndex", C.c_uint32)]


class _VkMemoryHeap(C.Structure):
    _fields_ = [("size", C.c_uint64), ("flags", C.c_uint32)]


class _VkMemoryProperties(C.Structure):
    _fields_ = [("memoryTypeCount", C.c_uint32), ("memoryTypes", _VkMemoryType * 32),
                ("memoryHeapCount", C.c_uint32), ("memoryHeaps", _VkMemoryHeap * 16)]


class _VkExtensionProperties(C.Structure):
    _fields_ = [("extensionName", C.c_char * 256), ("specVersion", C.c_uint32)]


class _VkQueueFamilyProperties(C.Structure):
    _fields_ = [("queueFlags", C.c_uint32), ("queueCount", C.c_uint32),
                ("timestampValidBits", C.c_uint32),
                ("minImageTransferGranularity", C.c_uint32 * 3)]


def _version(value: int) -> str:
    """An Vulkan apiVersion integer to ``major.minor.patch``."""
    return f"{value >> 22}.{(value >> 12) & 0x3ff}.{value & 0xfff}"


def _flags(value: int, names: dict[int, str]) -> list[str]:
    return [label for bit, label in names.items() if value & bit]


MEMORY_FLAG_NAMES = {
    MEMORY_DEVICE_LOCAL: "device_local", MEMORY_HOST_VISIBLE: "host_visible",
    MEMORY_HOST_COHERENT: "host_coherent", MEMORY_HOST_CACHED: "host_cached",
    0x10: "lazily_allocated", 0x20: "protected",
}


def _load_vulkan() -> tuple[object | None, str | None]:
    """Return (CDLL, None) or (None, reason). Never raises."""
    if sys.platform.startswith("win"):
        name = "vulkan-1.dll"
    elif sys.platform == "darwin":
        name = ctypes.util.find_library("vulkan") or "libvulkan.dylib"
    else:
        name = ctypes.util.find_library("vulkan") or "libvulkan.so.1"
    try:
        lib = C.CDLL(name)
    except OSError:
        return None, f"the Vulkan loader ({name}) is not installed"
    except Exception as exc:                                  # noqa: BLE001 - never crash the doctor
        return None, f"cannot open the Vulkan loader ({name}): {type(exc).__name__}"
    return lib, None


def _bind(lib: object) -> None:
    """Declare every symbol we call. Wrong restypes on a 64-bit host corrupt pointers."""
    lib.vkEnumerateInstanceVersion.argtypes = [C.POINTER(C.c_uint32)]
    lib.vkEnumerateInstanceVersion.restype = C.c_int
    lib.vkCreateInstance.argtypes = [C.POINTER(_VkInstanceCreateInfo), C.c_void_p, C.POINTER(C.c_void_p)]
    lib.vkCreateInstance.restype = C.c_int
    lib.vkDestroyInstance.argtypes = [C.c_void_p, C.c_void_p]
    lib.vkDestroyInstance.restype = None
    lib.vkEnumeratePhysicalDevices.argtypes = [C.c_void_p, C.POINTER(C.c_uint32), C.POINTER(C.c_void_p)]
    lib.vkEnumeratePhysicalDevices.restype = C.c_int
    lib.vkGetPhysicalDeviceProperties2.argtypes = [C.c_void_p, C.c_void_p]
    lib.vkGetPhysicalDeviceProperties2.restype = None
    lib.vkEnumerateDeviceExtensionProperties.argtypes = [C.c_void_p, C.c_char_p, C.POINTER(C.c_uint32),
                                                          C.POINTER(_VkExtensionProperties)]
    lib.vkEnumerateDeviceExtensionProperties.restype = C.c_int
    lib.vkGetPhysicalDeviceMemoryProperties.argtypes = [C.c_void_p, C.POINTER(_VkMemoryProperties)]
    lib.vkGetPhysicalDeviceMemoryProperties.restype = None
    lib.vkGetPhysicalDeviceQueueFamilyProperties.argtypes = [C.c_void_p, C.POINTER(C.c_uint32),
                                                             C.POINTER(_VkQueueFamilyProperties)]
    lib.vkGetPhysicalDeviceQueueFamilyProperties.restype = None


def _describe_device(lib: object, dev: C.c_void_p) -> dict:
    """Read one physical device. All counts are bounded before use."""
    result: dict = {
        "name": "unknown", "vendor_id": None, "device_id": None, "device_type": "unknown",
        "api_version": None, "driver_version": None, "intel": False,
        "queue_families": [], "extensions": {}, "extension_count": 0,
        "subgroup": {"size": None, "min": None, "max": None, "size_control": False,
                     "required_subgroup_size_stages": None},
        "memory": {"heaps": [], "types": [], "device_local_mib": 0,
                   "host_visible_mib": 0, "cached_host_visible_type": None},
    }

    # Properties2 + a pNext chain for Vulkan 1.1 properties and subgroup size
    # control. The base struct is written into a spare buffer; only the head is
    # interpreted. pNext points at our own structs, never at driver memory.
    buf = C.create_string_buffer(8192)
    head = _VkProperties2Head.from_buffer(buf)
    head.sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_PROPERTIES_2
    v11 = _VkVulkan11Properties(sType=VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_VULKAN_1_1_PROPERTIES)
    sg = _VkSubgroupSizeControlProperties(
        sType=VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_SUBGROUP_SIZE_CONTROL_PROPERTIES_EXT)
    head.pNext = C.cast(C.byref(v11), C.c_void_p)
    v11.pNext = C.cast(C.byref(sg), C.c_void_p)
    lib.vkGetPhysicalDeviceProperties2(dev, C.byref(head))
    props = _VkPropertiesHead.from_buffer(buf)
    result["name"] = props.deviceName.decode("utf-8", "replace").split("\0", 1)[0] or "unknown"
    result["vendor_id"] = hex(props.vendorID)
    result["device_id"] = hex(props.deviceID)
    result["device_type"] = DEVICE_TYPES.get(props.deviceType, "unknown")
    result["intel"] = props.vendorID == 0x8086
    result["api_version"] = _version(props.apiVersion) if props.apiVersion else None
    result["driver_version"] = hex(props.driverVersion)
    result["subgroup"]["size"] = v11.subgroupSize or None
    result["subgroup"]["min"] = sg.minSubgroupSize or None
    result["subgroup"]["max"] = sg.maxSubgroupSize or None
    result["subgroup"]["required_subgroup_size_stages"] = hex(sg.requiredSubgroupSizeStages)

    count = C.c_uint32()
    lib.vkEnumerateDeviceExtensionProperties(dev, None, C.byref(count), None)
    n = count.value
    if 0 < n <= 512:
        exts = (_VkExtensionProperties * n)()
        lib.vkEnumerateDeviceExtensionProperties(dev, None, C.byref(count), exts)
        names = {exts[i].extensionName.decode("ascii", "replace").split("\0", 1)[0]
                 for i in range(min(n, count.value))}
        result["extension_count"] = len(names)
        result["extensions"] = {key: (key in names) for key in INTERESTING_EXTENSIONS}
        result["subgroup"]["size_control"] = "VK_EXT_subgroup_size_control" in names
    elif n > 512:
        result["extension_count"] = n  # reported but not enumerated: implausibly large

    mp = _VkMemoryProperties()
    lib.vkGetPhysicalDeviceMemoryProperties(dev, C.byref(mp))
    heap_count = min(mp.memoryHeapCount, 16)
    heap_local = [bool(mp.memoryHeaps[i].flags & MEMORY_DEVICE_LOCAL) for i in range(heap_count)]
    for i in range(heap_count):
        heap = mp.memoryHeaps[i]
        result["memory"]["heaps"].append({"index": i, "size_mib": heap.size >> 20,
                                           "device_local": bool(heap.flags & MEMORY_DEVICE_LOCAL)})
    result["memory"]["device_local_mib"] = sum(
        (mp.memoryHeaps[i].size >> 20) for i in range(heap_count) if heap_local[i])
    type_count = min(mp.memoryTypeCount, 32)
    for i in range(type_count):
        mt = mp.memoryTypes[i]
        flags = mt.propertyFlags
        result["memory"]["types"].append({
            "index": i, "flags": _flags(flags, MEMORY_FLAG_NAMES), "heap_index": mt.heapIndex,
            "device_local": bool(flags & MEMORY_DEVICE_LOCAL),
            "host_visible": bool(flags & MEMORY_HOST_VISIBLE),
            "host_coherent": bool(flags & MEMORY_HOST_COHERENT),
            "host_cached": bool(flags & MEMORY_HOST_CACHED),
        })
    # The engine prefers a cached, host-visible, coherent type. Report the first
    # one, not just any host-visible type: a plain 0x7 type is a 2.4x slowdown.
    for i in range(type_count):
        fl = mp.memoryTypes[i].propertyFlags
        if fl & MEMORY_HOST_VISIBLE and fl & MEMORY_HOST_COHERENT and fl & MEMORY_HOST_CACHED:
            result["memory"]["cached_host_visible_type"] = i
            break
    result["memory"]["host_visible_mib"] = sum(
        (mp.memoryHeaps[mp.memoryTypes[i].heapIndex].size >> 20)
        for i in range(type_count) if mp.memoryTypes[i].propertyFlags & MEMORY_HOST_VISIBLE
        and mp.memoryTypes[i].heapIndex < heap_count)

    qcount = C.c_uint32()
    lib.vkGetPhysicalDeviceQueueFamilyProperties(dev, C.byref(qcount), None)
    qn = qcount.value
    if 0 < qn <= 64:
        families = (_VkQueueFamilyProperties * qn)()
        lib.vkGetPhysicalDeviceQueueFamilyProperties(dev, C.byref(qcount), families)
        result["queue_families"] = [
            {"index": i, "flags": hex(families[i].queueFlags), "count": families[i].queueCount,
             "compute": bool(families[i].queueFlags & QUEUE_COMPUTE)}
            for i in range(min(qn, qcount.value))]
    return result


def probe() -> dict:
    """Read-only Vulkan inventory. Returns a JSON-serialisable dict; never raises.

    ``status`` is ``ok`` with ``devices`` populated, ``unavailable`` when there
    is no loader/ICD/device, or ``error`` for an unexpected failure. A device
    count above 64 is rejected rather than trusted.
    """
    out: dict = {"status": "unavailable", "reason": None, "loader": None,
                 "instance_version": None, "devices": []}
    if not sys.platform.startswith("linux"):
        out["reason"] = "the Vulkan expert-streaming path is supported on Linux only"
        return out
    lib, reason = _load_vulkan()
    if lib is None:
        out["reason"] = reason
        return out
    out["loader"] = "libvulkan"
    inst = C.c_void_p()
    try:
        _bind(lib)
        version = C.c_uint32()
        if lib.vkEnumerateInstanceVersion(C.byref(version)) == 0:
            out["instance_version"] = _version(version.value)
        app = _VkApplicationInfo(sType=VK_STRUCTURE_TYPE_APPLICATION_INFO,
                                 apiVersion=(1 << 22) | (3 << 12))   # request 1.3
        ici = _VkInstanceCreateInfo(sType=VK_STRUCTURE_TYPE_INSTANCE_CREATE_INFO,
                                    pApplicationInfo=C.pointer(app))
        if lib.vkCreateInstance(C.byref(ici), None, C.byref(inst)) != 0 or not inst.value:
            out["reason"] = "the Vulkan loader could not create an instance (no ICD installed?)"
            return out
        count = C.c_uint32()
        lib.vkEnumeratePhysicalDevices(inst, C.byref(count), None)
        n = count.value
        if n == 0:
            out["reason"] = "no Vulkan physical device is visible to this user"
            return out
        if n > 64:
            out["status"] = "error"
            out["reason"] = f"implausible Vulkan device count ({n})"
            return out
        devices = (C.c_void_p * n)()
        lib.vkEnumeratePhysicalDevices(inst, C.byref(count), devices)
        out["devices"] = [_describe_device(lib, devices[i]) for i in range(min(n, count.value))]
        out["status"] = "ok"
        return out
    except Exception as exc:                                      # noqa: BLE001 - doctor must not crash
        out["status"] = "error"
        out["reason"] = f"{type(exc).__name__}: {exc}"
        return out
    finally:
        if inst.value:
            try:
                lib.vkDestroyInstance(inst, None)
            except Exception:                                     # noqa: BLE001 - teardown is best effort
                pass


def _meminfo(path: str = "/proc/meminfo") -> dict:
    """MemTotal and MemAvailable in MiB; zeros where they cannot be read."""
    values = {"MemTotal": 0, "MemAvailable": 0}
    if not sys.platform.startswith("linux"):
        return values
    try:
        for line in Path(path).read_text(encoding="utf-8", errors="replace").splitlines():
            key, sep, rest = line.partition(":")
            if sep and key in values:
                parts = rest.split()
                if parts and parts[0].isdigit():
                    values[key] = int(parts[0]) // 1024
    except OSError:
        pass
    return values


def _render_nodes(dev_root: str = "/dev") -> list[str]:
    if not sys.platform.startswith("linux"):
        return []
    return sorted(str(p) for p in Path(dev_root, "dri").glob("renderD[0-9]*"))


def _cgroup_available(cgroup_root: str = "/sys/fs/cgroup") -> tuple[int | None, str]:
    """Room under the closest finite cgroup memory limit, in MiB; None when unbounded.

    Mirrors the engine's memory guard: MemAvailable alone overstates what a
    container can use. Both cgroup v2 (memory.max / memory.current) and v1
    (memory/memory.limit_in_bytes) are read; ``max`` means unbounded.
    """
    root = Path(cgroup_root)
    if not root.is_dir():
        return None, "none"
    limit = None
    limit_source = "cgroup"
    for relative, parser in (
        ("memory.max", lambda text: None if text.strip() == "max" else int(text)),
        ("memory/memory.limit_in_bytes", int),
    ):
        try:
            text = (root / relative).read_text(encoding="ascii", errors="replace").strip()
            value = parser(text)
            if value is not None and 0 < value < (1 << 60):
                limit, limit_source = value, "memory"
                break
        except (OSError, ValueError):
            continue
    if limit is None:
        return None, "unbounded"
    used = 0
    for relative in ("memory.current", "memory/memory.usage_in_bytes"):
        try:
            used = int((root / relative).read_text(encoding="ascii", errors="replace").strip())
            break
        except (OSError, ValueError):
            continue
    return max(0, (limit - used) // (1024 * 1024)), limit_source


def plan(cfg: dict | None = None, *, meminfo_path: str = "/proc/meminfo",
         dev_root: str = "/dev", cgroup_root: str = "/sys/fs/cgroup") -> dict:
    """Shared-memory-aware sizing advice for a Vulkan run config.

    On an iGPU the Vulkan DEVICE_LOCAL heap *is* system RAM. A runtime that
    reports ~0.2 GB of "VRAM" (issue #515) is reading the PCI BAR, not the heap,
    and the native engine's ``vram_free_mib`` is a MemAvailable estimate, not a
    dedicated-VRAM measurement. Size the expert cache from MemAvailable instead.
    Pure except for reading /proc.
    """
    cfg = cfg or {}
    info = _meminfo(meminfo_path)
    args = cfg.get("args") or []
    slots = None
    if "--slots" in args:
        try:
            slots = int(args[args.index("--slots") + 1])
        except (ValueError, IndexError):
            slots = None
    slots = DEFAULT_SLOTS if slots is None else slots
    cache_mib = slots * SLOT_BYTES // (1024 * 1024)
    available = info["MemAvailable"]
    cgroup_room, cgroup_source = _cgroup_available(cgroup_root)
    memory_source = "MemAvailable"
    if cgroup_room is not None and cgroup_room < available:
        available, memory_source = cgroup_room, f"cgroup ({cgroup_source})"
    budget = max(0, available - MEMORY_HEADROOM_MIB)
    recommended = max(MIN_SLOTS, min(DEFAULT_SLOTS, budget * 1024 * 1024 // SLOT_BYTES)) if available else None
    nodes = _render_nodes(dev_root)
    return {
        "mem_total_mib": info["MemTotal"],
        "mem_available_mib": available,
        "memory_source": memory_source,
        "headroom_mib": MEMORY_HEADROOM_MIB,
        "configured_slots": slots,
        "cache_bytes": slots * SLOT_BYTES,
        "cache_mib": cache_mib,
        "recommended_slots": recommended,
        "recommended_cache_mib": (recommended * SLOT_BYTES // (1024 * 1024)) if recommended else None,
        "ple_io_ram": ("--ple-io" in args and args[args.index("--ple-io") + 1:args.index("--ple-io") + 2] == ["ram"]),
        "render_nodes": nodes,
        "render_node_accessible": (not nodes) or any(
            os.access(node, os.R_OK | os.W_OK) for node in nodes),
    }


def issues(probe_result: dict, plan_result: dict | None = None, *, required: bool = False) -> list[dict]:
    """Turn a probe and a plan into doctor issues. ``required`` makes missing Vulkan an error."""
    result: list[dict] = []
    sev = "error" if required else "warning"
    status = probe_result.get("status")
    if status != "ok":
        result.append({"code": "VULKAN_UNAVAILABLE", "severity": sev,
                       "message": "Vulkan is not ready here: " + (probe_result.get("reason") or "no usable device")})
        return result
    devices = probe_result.get("devices") or []
    intel = [d for d in devices if d.get("intel") and any(q.get("compute") for q in d.get("queue_families", []))]
    if not intel:
        result.append({"code": "VULKAN_NO_INTEL_DEVICE", "severity": sev,
                       "message": "no Intel Vulkan device with a compute queue was found; the native engine "
                                  "currently selects vendor 0x8086 only"})
        return result
    for d in intel:
        cached = d.get("memory", {}).get("cached_host_visible_type")
        if cached is None:
            result.append({"code": "VULKAN_NO_CACHED_HOST_MEMORY", "severity": sev,
                           "message": f"{d['name']} has no HOST_VISIBLE + HOST_COHERENT + HOST_CACHED memory "
                                      "type; the engine falls back to uncached memory, which measured ~2.4x slower "
                                      "on the B390"})
        api = d.get("api_version") or "0.0"
        try:
            major, minor = (int(x) for x in api.split(".")[:2])
        except ValueError:
            major, minor = 0, 0
        if (major, minor) < (1, 3):
            result.append({"code": "VULKAN_API_TOO_OLD", "severity": sev,
                           "message": f"{d['name']} reports Vulkan {api}; the native engine needs 1.3"})
        groups = d.get("subgroup", {}).get("size")
        if not groups:
            result.append({"code": "VULKAN_SUBGROUP_UNKNOWN", "severity": "warning",
                           "message": f"{d['name']} did not report a subgroup size; the GEMV shaders use subgroup "
                                      "arithmetic and rely on the driver default"})
    if plan_result:
        p = plan_result
        if p.get("ple_io_ram"):
            result.append({"code": "VULKAN_PLE_IO_RAM", "severity": "warning",
                           "message": "--ple-io ram locks the n-gram table in RAM (28.8 GB on this model); on a "
                                      "~30 GB shared-memory machine that can starve the expert cache. The native "
                                      "Vulkan path needs no PLE table"})
        if p.get("recommended_slots") and p.get("configured_slots", 0) > p["recommended_slots"]:
            result.append({"code": "VULKAN_CACHE_TOO_LARGE", "severity": "warning",
                           "message": f"the expert cache is {p['configured_slots']} slots (~{p['cache_mib']} MiB) "
                                      f"but only ~{p['mem_available_mib']} MiB is available; about "
                                      f"{p['recommended_slots']} slots fit with headroom"})
        if p.get("render_nodes") and not p.get("render_node_accessible"):
            result.append({"code": "VULKAN_DEVICE_ACCESS", "severity": "warning",
                           "message": f"this user cannot open {', '.join(p['render_nodes'])}; add the user to the "
                                      "`render` group and log in again, or the engine will not see the card"})
        if any(d.get("device_type") == "integrated_gpu" for d in intel):
            result.append({"code": "VULKAN_SHARED_MEMORY", "severity": "info",
                           "message": "this is an integrated GPU: the Vulkan DEVICE_LOCAL heap is system RAM, not "
                                      "dedicated VRAM. Size the expert cache from MemAvailable; the engine's "
                                      "vram_free_mib is a shared-RAM estimate, and a runtime that reports ~0.2 GB "
                                      "of VRAM is reading the PCI BAR (issue #515)"})
    return result


def inspect(*, cfg: dict | None = None, meminfo_path: str = "/proc/meminfo",
            dev_root: str = "/dev") -> dict:
    """Everything the doctor needs in one call: probe + shared-memory plan + issues."""
    p = probe()
    plan_result = plan(cfg, meminfo_path=meminfo_path, dev_root=dev_root)
    return {"probe": p, "plan": plan_result,
            "issues": issues(p, plan_result, required=bool(cfg and cfg.get("backend") == "vulkan"))}
