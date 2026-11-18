// src/platform/numa_os.cpp - the Windows arm of the OS-side NUMA answers (numa_os.hpp).
//
// Every function answers "-1/false/1/nullptr" on every failure so callers keep their legacy path.
// Non-Windows builds return those "unknown" answers for everything; the port points are marked
// inline (Linux: sysfs numa_node, mbind, sched_setaffinity, get_mempolicy).

#include "strata/platform/numa_os.hpp"

#if defined(_WIN32)
#  ifndef NOMINMAX
#    define NOMINMAX
#  endif
#  include <windows.h>
#  include <setupapi.h>
#  include <devpropdef.h>
#  ifndef INITGUID
#    define INITGUID   // devpkey.h/devguid.h emit their DEVPKEY_*/GUID_DEVCLASS_* definitions only under it
#  endif
#  include <devpkey.h>
#  include <devguid.h>
#endif

#include <cuda_runtime.h>

#include <cstddef>
#include <vector>

namespace strata::platform {

int numa_node_count_os() {
#if defined(_WIN32)
    ULONG highest = 0;
    return GetNumaHighestNodeNumber(&highest) ? (int) highest + 1 : 1;
#else
    return 1;   // port point: Linux counts sysfs nodes whose cpulist holds CPUs (numa_pool.hpp's rule)
#endif
}

int device_numa_node(int dev) {
#if defined(_WIN32)
    // SetupAPI over GUID_DEVCLASS_DISPLAY: match cudaDeviceProp's PCI bus (and slot when reported)
    // against DEVPKEY_Device_BusNumber/_Address, read DEVPKEY_Device_Numa_Node.  Soft-loaded like the
    // ring's other system calls so nothing links setupapi.  Bus ids could repeat across PCI segments;
    // the slot check keeps a repeat from answering another card's node, and a missing address
    // property does not disqualify a match.
    cudaDeviceProp prop{};
    if (cudaGetDeviceProperties(&prop, dev) != cudaSuccess) { (void) cudaGetLastError(); return -1; }
    static const auto sa = [] {
        struct Api {
            HDEVINFO(WINAPI* class_devs)(const GUID*, PCWSTR, HWND, DWORD);
            BOOL(WINAPI* enum_info)(HDEVINFO, DWORD, PSP_DEVINFO_DATA);
            BOOL(WINAPI* get_prop)(HDEVINFO, PSP_DEVINFO_DATA, const DEVPROPKEY*, DEVPROPTYPE*, PBYTE, DWORD, PDWORD, DWORD);
            BOOL(WINAPI* destroy)(HDEVINFO);
        } api{};
        if (HMODULE m = LoadLibraryExW(L"setupapi.dll", nullptr, LOAD_LIBRARY_SEARCH_SYSTEM32)) {
            api.class_devs = (decltype(api.class_devs)) GetProcAddress(m, "SetupDiGetClassDevsW");
            api.enum_info = (decltype(api.enum_info)) GetProcAddress(m, "SetupDiEnumDeviceInfo");
            api.get_prop = (decltype(api.get_prop)) GetProcAddress(m, "SetupDiGetDevicePropertyW");
            api.destroy = (decltype(api.destroy)) GetProcAddress(m, "SetupDiDestroyDeviceInfoList");
        }
        return api;
    }();
    if (!sa.class_devs || !sa.enum_info || !sa.get_prop || !sa.destroy) return -1;
    int node = -1;
    HDEVINFO set = sa.class_devs(&GUID_DEVCLASS_DISPLAY, nullptr, nullptr, DIGCF_PRESENT);
    if (set == INVALID_HANDLE_VALUE) return -1;
    for (DWORD i = 0;; ++i) {
        SP_DEVINFO_DATA di{};
        di.cbSize = sizeof di;
        if (!sa.enum_info(set, i, &di)) break;
        DEVPROPTYPE ty = 0;
        DWORD bus = ~0u;
        if (!sa.get_prop(set, &di, &DEVPKEY_Device_BusNumber, &ty, (PBYTE) &bus, sizeof bus, nullptr, 0) ||
            bus != (DWORD) prop.pciBusID) continue;
        DWORD addr = ~0u;
        if (sa.get_prop(set, &di, &DEVPKEY_Device_Address, &ty, (PBYTE) &addr, sizeof addr, nullptr, 0) &&
            addr != ~0u && (int) (addr >> 16) != prop.pciDeviceID) continue;
        DWORD nn = ~0u;
        if (sa.get_prop(set, &di, &DEVPKEY_Device_Numa_Node, &ty, (PBYTE) &nn, sizeof nn, nullptr, 0) && nn != ~0u)
            node = (int) nn;
        break;
    }
    sa.destroy(set);
    return node;
#else
    (void) dev;
    return -1;   // port point: Linux reads the PCI function's numa_node sysfs attribute
#endif
}

bool bind_thread_to_node(int node) {
#if defined(_WIN32)
    // CPU Sets, like the pool's own pinning (pool_affinity_win.hpp): correct across processor groups
    // on >64-CPU machines, where a GROUP_AFFINITY can only name one group and a node's CPUs can span
    // several.
    ULONG bytes = 0;
    if (!GetSystemCpuSetInformation(nullptr, 0, &bytes, GetCurrentProcess(), 0) &&
        GetLastError() != ERROR_INSUFFICIENT_BUFFER) return false;
    std::vector<unsigned char> buf(bytes);
    if (bytes && !GetSystemCpuSetInformation(reinterpret_cast<PSYSTEM_CPU_SET_INFORMATION>(buf.data()),
                                            bytes, &bytes, GetCurrentProcess(), 0)) return false;
    constexpr size_t header_size = offsetof(SYSTEM_CPU_SET_INFORMATION, CpuSet);
    // NumaNodeIndex is group-relative: on a multi-group machine the same local index names
    // different global nodes, so a local-index match can pick CPU Sets on the wrong socket.
    // Resolve each set's global node from (Group, LogicalProcessorIndex) instead; keep the
    // local index only as the fallback when resolution fails for every set.
    std::vector<ULONG> ids, legacy;
    unsigned char seen[256] = {};
    int distinct = 0;
    bool resolved = false;
    for (size_t off = 0; off + header_size <= bytes;) {
        const auto* info = reinterpret_cast<const SYSTEM_CPU_SET_INFORMATION*>(buf.data() + off);
        if (info->Size < header_size || info->Size > bytes - off) break;
        if (info->Type == CpuSetInformation && info->Size >= sizeof(SYSTEM_CPU_SET_INFORMATION)) {
            const BYTE nn = info->CpuSet.NumaNodeIndex;
            if (!seen[nn]) { seen[nn] = 1; ++distinct; }
            if (!info->CpuSet.Allocated || info->CpuSet.AllocatedToTargetProcess) {
                PROCESSOR_NUMBER pn{ info->CpuSet.Group, info->CpuSet.LogicalProcessorIndex, 0 };
                USHORT gn = 0xFFFF;
                if (GetNumaProcessorNodeEx(&pn, &gn)) {
                    resolved = true;
                    if ((int) gn == node) ids.push_back(info->CpuSet.Id);
                } else if ((int) nn == node)
                    legacy.push_back(info->CpuSet.Id);
            }
        }
        off += info->Size;
    }
    if (!ids.empty())
        return SetThreadSelectedCpuSets(GetCurrentThread(), ids.data(), (ULONG) ids.size()) != FALSE;
    // CPU Sets carry no NUMA mapping on some firmware (every entry reports NumaNodeIndex 0 even on a
    // real multi-node box): a node-0 match there selects EVERY socket and reports success.  The raw
    // index may only be trusted when it can actually tell the machine's nodes apart.
    if (!resolved && !legacy.empty() && (distinct >= 2 || numa_node_count_os() <= 1))
        return SetThreadSelectedCpuSets(GetCurrentThread(), legacy.data(), (ULONG) legacy.size()) != FALSE;
    // Fall back to the node's processor-group mask.  A node whose CPUs span several groups
    // (>64 threads) gets only the first group here - still better than an unbound thread.
    GROUP_AFFINITY ga;
    return GetNumaNodeProcessorMaskEx((UCHAR) node, &ga) && ga.Mask &&
           SetThreadGroupAffinity(GetCurrentThread(), &ga, nullptr) != FALSE;
#else
    (void) node;
    return false;   // port point: Linux sched_setaffinity on the node's cpulist
#endif
}

#if defined(_WIN32)
namespace {
/// VirtualAllocExNuma, soft-loaded once so nothing new links.
LPVOID(WINAPI* numa_va_fn())(HANDLE, LPVOID, SIZE_T, DWORD, DWORD, DWORD) {
    static const auto f = [] {
        HMODULE k32 = GetModuleHandleW(L"kernel32.dll");
        return k32 ? (LPVOID(WINAPI*)(HANDLE, LPVOID, SIZE_T, DWORD, DWORD, DWORD))
                     GetProcAddress(k32, "VirtualAllocExNuma") : nullptr;
    }();
    return f;
}
}  // namespace
#endif

bool numa_va_supported() {
#if defined(_WIN32)
    return numa_va_fn() != nullptr;
#else
    return false;
#endif
}

void* numa_virtual_alloc(void* addr, uint64_t bytes, bool large, int node) {
#if defined(_WIN32)
    const auto f = numa_va_fn();
    if (f == nullptr) return nullptr;
    const DWORD flags = MEM_RESERVE | MEM_COMMIT | (large ? MEM_LARGE_PAGES : 0);
    return f(GetCurrentProcess(), addr, (SIZE_T) bytes, flags, PAGE_READWRITE, (DWORD) node);
#else
    (void) addr; (void) bytes; (void) large; (void) node;
    return nullptr;   // port point: Linux mmap + mbind MPOL_BIND, first-touched
#endif
}

}  // namespace strata::platform
