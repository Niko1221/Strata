#include "strata/core/vmm.hpp"
#include <string>

#if !defined(STRATA_USE_HIP) && !defined(STRATA_HIP_GFX906) && !defined(STRATA_GPU_METAL)
#include <cuda.h>
#include <cuda_runtime.h>

#include <mutex>

namespace strata::core {
namespace {

struct Api {
    bool ok = false;
    uint64_t gran = 0;
    int dev = -1;
    decltype(&cuMemAddressReserve) reserve = nullptr;
    decltype(&cuMemAddressFree) free_va = nullptr;
    decltype(&cuMemCreate) create = nullptr;
    decltype(&cuMemRelease) release = nullptr;
    decltype(&cuMemMap) map = nullptr;
    decltype(&cuMemUnmap) unmap = nullptr;
    decltype(&cuMemSetAccess) access = nullptr;
};

template <class F> bool resolve(const char* name, F& f) {
    cudaDriverEntryPointQueryResult q{};
    void* p = nullptr;
#if CUDART_VERSION >= 12050
    const cudaError_t e = cudaGetDriverEntryPointByVersion(name, &p, 12000, cudaEnableDefault, &q);
#else   // the versioned query arrived in CUDA 12.5
    const cudaError_t e = cudaGetDriverEntryPoint(name, &p, cudaEnableDefault, &q);
#endif
    if (e != cudaSuccess || q != cudaDriverEntryPointSuccess || p == nullptr) {
        (void) cudaGetLastError();
        return false;
    }
    f = (F) p;
    return true;
}

// never destroyed: the K/V pools and the cache release their chunks from static destructors at exit
const Api& api() {
    static Api& a = *new Api;
    static std::once_flag once;
    std::call_once(once, [] {
        decltype(&cuDeviceGetAttribute) attr = nullptr;
        decltype(&cuMemGetAllocationGranularity) granularity = nullptr;
        if (cudaGetDevice(&a.dev) != cudaSuccess) return;
        if (!resolve("cuDeviceGetAttribute", attr) || !resolve("cuMemGetAllocationGranularity", granularity) ||
            !resolve("cuMemAddressReserve", a.reserve) || !resolve("cuMemAddressFree", a.free_va) ||
            !resolve("cuMemCreate", a.create) || !resolve("cuMemRelease", a.release) || !resolve("cuMemMap", a.map) ||
            !resolve("cuMemUnmap", a.unmap) || !resolve("cuMemSetAccess", a.access))
            return;
        int vmm = 0;
        if (attr(&vmm, CU_DEVICE_ATTRIBUTE_VIRTUAL_MEMORY_MANAGEMENT_SUPPORTED, (CUdevice) a.dev) != CUDA_SUCCESS || !vmm)
            return;
        CUmemAllocationProp prop{};
        prop.type = CU_MEM_ALLOCATION_TYPE_PINNED;
        prop.location.type = CU_MEM_LOCATION_TYPE_DEVICE;
        prop.location.id = a.dev;
        size_t g = 0;
        if (granularity(&g, &prop, CU_MEM_ALLOC_GRANULARITY_MINIMUM) != CUDA_SUCCESS || g == 0) return;
        a.gran = g;
        a.ok = true;
    });
    return a;
}

}  // namespace

bool vmm_available() { return api().ok; }
std::string vmm_off_reason() { return api().ok ? "" : "CUDA virtual memory is unavailable"; }
uint64_t vmm_granularity() { return api().ok ? api().gran : 0; }

VmmChunk vmm_chunk_new() {
    const Api& a = api();
    if (!a.ok) return 0;
    CUmemAllocationProp prop{};
    prop.type = CU_MEM_ALLOCATION_TYPE_PINNED;
    prop.location.type = CU_MEM_LOCATION_TYPE_DEVICE;
    prop.location.id = a.dev;
    CUmemGenericAllocationHandle h = 0;
    if (a.create(&h, (size_t) a.gran, &prop, 0) != CUDA_SUCCESS) return 0;
    return (VmmChunk) h;
}

void vmm_chunk_free(VmmChunk h) {
    if (h != 0 && api().ok) api().release((CUmemGenericAllocationHandle) h);
}

bool VmmRange::reserve(uint64_t bytes) {
    release();
    const Api& a = api();
    if (!a.ok || bytes == 0) return false;
    const uint64_t n = (bytes + a.gran - 1) / a.gran;
    CUdeviceptr p = 0;
    if (a.reserve(&p, (size_t) (n * a.gran), 0, 0, 0) != CUDA_SUCCESS) return false;
    base_ = (unsigned long long) p;
    h_.assign((size_t) n, 0);
    return true;
}

void VmmRange::release() {
    if (base_ == 0) return;
    const Api& a = api();
    for (int64_t i = 0; i < chunks(); ++i) vmm_chunk_free(unmap(i));
    a.free_va((CUdeviceptr) base_, (size_t) ((uint64_t) h_.size() * a.gran));
    base_ = 0;
    h_.clear();
}

int64_t VmmRange::mapped_count() const {
    int64_t n = 0;
    for (const VmmChunk h : h_) n += h != 0;
    return n;
}

bool VmmRange::map_one(int64_t i, VmmChunk h) {
    const Api& a = api();
    if (a.map((CUdeviceptr) (base_ + (uint64_t) i * a.gran), (size_t) a.gran, 0, (CUmemGenericAllocationHandle) h, 0) !=
        CUDA_SUCCESS)
        return false;
    h_[(size_t) i] = h;
    return true;
}

bool VmmRange::set_access(int64_t lo, int64_t hi) {
    if (hi <= lo) return true;
    const Api& a = api();
    CUmemAccessDesc d{};
    d.location.type = CU_MEM_LOCATION_TYPE_DEVICE;
    d.location.id = a.dev;
    d.flags = CU_MEM_ACCESS_FLAGS_PROT_READWRITE;
    return a.access((CUdeviceptr) (base_ + (uint64_t) lo * a.gran), (size_t) ((uint64_t) (hi - lo) * a.gran), &d, 1) ==
           CUDA_SUCCESS;
}

bool VmmRange::commit_run(int64_t lo, int64_t hi) {
    if (set_access(lo, hi)) return true;
    for (int64_t c = lo; c < hi; ++c) vmm_chunk_free(unmap(c));
    return false;
}

VmmChunk VmmRange::unmap(int64_t i) {
    if (!mapped(i)) return 0;
    const Api& a = api();
    const VmmChunk h = h_[(size_t) i];
    if (a.unmap((CUdeviceptr) (base_ + (uint64_t) i * a.gran), (size_t) a.gran) != CUDA_SUCCESS) return 0;
    h_[(size_t) i] = 0;
    return h;
}


bool vmm_device_supported() { return vmm_available(); }
uint64_t vmm_segment_granularity() {
    const Api& a = api();
    if (!a.ok) return 0;
    decltype(&cuMemGetAllocationGranularity) f = nullptr;
    if (!resolve("cuMemGetAllocationGranularity", f)) return 0;
    CUmemAllocationProp prop{};
    prop.type = CU_MEM_ALLOCATION_TYPE_PINNED;
    prop.location.type = CU_MEM_LOCATION_TYPE_DEVICE;
    prop.location.id = a.dev;
    size_t g = 0;
    return f(&g, &prop, CU_MEM_ALLOC_GRANULARITY_RECOMMENDED) == CUDA_SUCCESS ? g : 0;
}
uintptr_t vmm_reserve(uint64_t bytes) {
    CUdeviceptr va = 0;
    return api().ok && api().reserve(&va, bytes, 0, 0, 0) == CUDA_SUCCESS ? (uintptr_t) va : 0;
}
void vmm_address_free(uintptr_t va, uint64_t bytes) { api().free_va((CUdeviceptr) va, bytes); }
bool vmm_segment_map(uintptr_t va, uint64_t bytes, VmmChunk& handle) {
    const Api& a = api();
    if (!a.ok) return false;
    CUmemAllocationProp prop{};
    prop.type = CU_MEM_ALLOCATION_TYPE_PINNED;
    prop.location.type = CU_MEM_LOCATION_TYPE_DEVICE;
    prop.location.id = a.dev;
    CUmemGenericAllocationHandle h = 0;
    if (a.create(&h, bytes, &prop, 0) != CUDA_SUCCESS) return false;
    if (a.map((CUdeviceptr) va, bytes, 0, h, 0) != CUDA_SUCCESS) { a.release(h); return false; }
    CUmemAccessDesc d{};
    d.location = prop.location;
    d.flags = CU_MEM_ACCESS_FLAGS_PROT_READWRITE;
    if (a.access((CUdeviceptr) va, bytes, &d, 1) != CUDA_SUCCESS) {
        a.unmap((CUdeviceptr) va, bytes); a.release(h); return false;
    }
    handle = (VmmChunk) h;
    return true;
}
bool vmm_segment_unmap(uintptr_t va, uint64_t bytes, VmmChunk h) {
    return api().unmap((CUdeviceptr) va, bytes) == CUDA_SUCCESS &&
           api().release((CUmemGenericAllocationHandle) h) == CUDA_SUCCESS;
}
}  // namespace strata::core

#else  // not CUDA
// HIP (ROCm 6+): the same elastic range through hipMemAddressReserve / hipMemCreate / hipMemMap / hipMemSetAccess.  Available
// only where the device reports virtual memory management support (STRATA_HIP_VMM=0 turns it off): --kv-grow stays off
// otherwise, as before.  Everything else (Metal, gfx906, ROCm older than 6): no virtual memory, the elastic K/V stays off.
#if defined(STRATA_USE_HIP) && !defined(STRATA_HIP_GFX906)
#include <hip/hip_runtime.h>
#include <hip/hip_version.h>
#include <map>
#include <algorithm>
#include <chrono>
#include <thread>
#if defined(__linux__)
#include <sys/resource.h>
#endif

#include <cstdio>
#include <cstdlib>
#include <mutex>
#endif

#if defined(STRATA_USE_HIP) && !defined(STRATA_HIP_GFX906) && defined(HIP_VERSION_MAJOR) && HIP_VERSION_MAJOR >= 6
namespace strata::core {
namespace {
struct Api {
    bool ok = false;
    uint64_t gran = 0;
    int dev = -1;
    std::string why = "HIP virtual memory is unavailable";
};
bool failed() {
    (void) hipGetLastError();
    return false;
}

hipMemAllocationProp device_prop(int dev) {
    hipMemAllocationProp prop{};
    prop.type = hipMemAllocationTypePinned;
    prop.location.type = hipMemLocationTypeDevice;
    prop.location.id = dev;
    return prop;
}

hipMemGenericAllocationHandle_t handle_of(VmmChunk h) { return (hipMemGenericAllocationHandle_t) (uintptr_t) h; }

bool p_reserve(uint64_t bytes, uintptr_t& va) {
    void* p = nullptr;
    if (hipMemAddressReserve(&p, (size_t) bytes, 0, nullptr, 0) != hipSuccess) return failed();
    va = (uintptr_t) p;
    return true;
}

void p_address_free(uintptr_t va, uint64_t bytes) {
    if (hipMemAddressFree((void*) va, (size_t) bytes) != hipSuccess) failed();
}

bool p_create(int dev, uint64_t bytes, VmmChunk& h) {
    const hipMemAllocationProp prop = device_prop(dev);
    hipMemGenericAllocationHandle_t x{};
    if (hipMemCreate(&x, (size_t) bytes, &prop, 0) != hipSuccess) return failed();
    h = (VmmChunk) (uintptr_t) x;
    return true;
}

bool p_release(VmmChunk h) { return hipMemRelease(handle_of(h)) == hipSuccess || failed(); }

bool p_map(uintptr_t va, uint64_t bytes, VmmChunk h) {
    return hipMemMap((void*) va, (size_t) bytes, 0, handle_of(h), 0) == hipSuccess || failed();
}

bool p_unmap(uintptr_t va, uint64_t bytes) { return hipMemUnmap((void*) va, (size_t) bytes) == hipSuccess || failed(); }

bool p_access(int dev, uintptr_t va, uint64_t bytes) {
    hipMemAccessDesc d{};
    d.location.type = hipMemLocationTypeDevice;
    d.location.id = dev;
    d.flags = hipMemAccessFlagsProtReadWrite;
    return hipMemSetAccess((void*) va, (size_t) bytes, &d, 1) == hipSuccess || failed();
}

uint64_t free_bytes() {
    size_t f = 0, t = 0;
    if (hipMemGetInfo(&f, &t) != hipSuccess) failed();
    return (uint64_t) f;
}

bool fill(uintptr_t va, uint64_t bytes) {
    return (hipMemset((void*) va, 0x5a, (size_t) bytes) == hipSuccess && hipDeviceSynchronize() == hipSuccess) ||
           failed();
}

// On Linux ROCm keeps an open file (a dmabuf) for every physical allocation, so each chunk is one: a 24 GiB cache
// holds ~12,000, far past the usual soft limit of 1,024 open files, and the maps past it fail as "out of memory".  So
// the soft limit goes up to what every chunk the card can hold needs, as far as the hard limit allows (the engine
// itself uses no select(), the one thing a higher limit could break).  False when the hard limit is lower.
bool open_files_for(uint64_t g, std::string& why) {
#if defined(__linux__)
    size_t f = 0, t = 0;
    if (hipMemGetInfo(&f, &t) != hipSuccess) { why = "hipMemGetInfo failed"; return failed(); }
    const rlim_t need = (rlim_t) (t / g) + 1024;   // every chunk of the card, and the files the engine opens itself
    rlimit rl{};
    if (getrlimit(RLIMIT_NOFILE, &rl) != 0) { why = "cannot read the open-file limit"; return false; }
    if (rl.rlim_cur == RLIM_INFINITY || rl.rlim_cur >= need) return true;
    rlimit up = rl;
    up.rlim_cur = rl.rlim_max == RLIM_INFINITY ? need : std::min<rlim_t>(rl.rlim_max, need);
    if (setrlimit(RLIMIT_NOFILE, &up) == 0) {
        std::fprintf(stderr,
                     "strata: open-file limit %llu -> %llu: ROCm keeps one open per chunk of GPU virtual memory\n",
                     (unsigned long long) rl.rlim_cur, (unsigned long long) up.rlim_cur);
        rl = up;
    }
    if (rl.rlim_cur >= need) return true;
    why = "ROCm keeps an open file per " + std::to_string(g >> 20) + " MiB chunk: " +
          std::to_string((unsigned long long) need) +
          " needed, and the open-file limit is " + std::to_string((unsigned long long) rl.rlim_cur) +
          " (hard " + std::to_string((unsigned long long) rl.rlim_max) + "; ulimit -n, systemd LimitNOFILE=)";
    return false;
#else
    (void) g;
    (void) why;
    return true;
#endif
}

// Not every ROCm runtime does what was measured above: setup's ROCm 7.10 wheels report a recommended granularity
// below 2 MiB and give no VRAM back on release.  So before the first use on a device the runtime has to show it, with
// chunks of the size the users take: a granularity of 2 MiB or more, the free-memory counter (the one VRAM's checks
// and the cache's sizing read) falling by about what is mapped, and rising again once it is unmapped and released
// while the addresses stay reserved.  Anything else: false, and `why` says what was seen.
bool runtime_check(int dev, uint64_t g, std::string& why) {
    auto mib = [](uint64_t b) { return std::to_string((b + (1u << 19)) >> 20) + " MiB"; };
    if (g < (2u << 20)) {
        why = "the recommended allocation granularity is " + std::to_string(g >> 10) + " KiB, not 2 MiB or more";
        return false;
    }
    if (!open_files_for(g, why)) return false;
    const int n = (int) std::max<uint64_t>(4, (32ull << 20) / g);
    const uint64_t bytes = (uint64_t) n * g;
    // another program on the card can move the counter at the same moment: one try more before giving up
    for (int attempt = 0; attempt < 2; ++attempt) {
        uintptr_t va = 0;
        if (!p_reserve(bytes, va)) {
            why = "hipMemAddressReserve failed";
            return false;
        }
        std::vector<VmmChunk> h;
        const uint64_t f0 = free_bytes();
        bool ok = true;
        for (int i = 0; i < n && ok; ++i) {
            VmmChunk c = 0;
            ok = p_create(dev, g, c) && (h.push_back(c), p_map(va + (uint64_t) i * g, g, c));
        }
        ok = ok && p_access(dev, va, bytes) && fill(va, bytes);
        const uint64_t f1 = free_bytes();
        for (size_t i = 0; i < h.size(); ++i) p_unmap(va + (uint64_t) i * g, g);
        for (const VmmChunk c : h) p_release(c);
        uint64_t f2 = free_bytes();
        for (int i = 0; ok && i < 20 && f2 < f1 + bytes / 2; ++i) {   // the counter may follow a moment later
            std::this_thread::sleep_for(std::chrono::milliseconds(10));
            f2 = free_bytes();
        }
        p_address_free(va, bytes);
        if (!ok) {
            why = "mapping " + mib(bytes) + " in " + std::to_string(n) + " chunks failed";
            return false;
        }
        const uint64_t took = f0 > f1 ? f0 - f1 : 0, back = f2 > f1 ? f2 - f1 : 0;
        if (took >= bytes / 2 && took <= 2 * bytes + (64ull << 20) && back >= bytes / 2) return true;
        why = "the free VRAM fell by " + mib(took) + " for " + mib(bytes) + " mapped and rose by " + mib(back) +
              " after its release";
    }
    return false;
}

std::string runtime_name() {
    int v = 0;
    if (hipRuntimeGetVersion(&v) != hipSuccess) failed();
    return "HIP runtime " + std::to_string(v / 10000000) + "." + std::to_string(v / 100000 % 100) + "." +
           std::to_string(v % 100000);
}

void fail(const char* what, hipError_t e) {
    static int n = 0;
    if (n++ < 8) std::fprintf(stderr, "strata: HIP virtual memory: %s failed: %s\n", what, hipGetErrorString(e));
    (void) hipGetLastError();
}
hipMemAllocationProp alloc_prop(int dev) {
    hipMemAllocationProp p{};
    p.type = hipMemAllocationTypePinned;
    p.requestedHandleType = hipMemHandleTypeNone;
    p.location.type = hipMemLocationTypeDevice;
    p.location.id = dev;
    return p;
}
const Api& api() {
    // Stable per-device entries outlive ranges released by static destructors.
    static auto& cache = *new std::map<int, Api>;
    static std::mutex mu;
    std::lock_guard<std::mutex> lock(mu);
    int dev = -1;
    if (hipGetDevice(&dev) != hipSuccess) {
        (void) hipGetLastError();
        static const Api unavailable;
        return unavailable;
    }
    auto [it, fresh] = cache.try_emplace(dev);
    Api& a = it->second;
    if (fresh) [&] {
        const char* off = std::getenv("STRATA_HIP_VMM");
        if (off != nullptr && off[0] == '0') { a.why = "STRATA_HIP_VMM=0"; return; }
        a.dev = dev;
        int vmm = 0;
        if (hipDeviceGetAttribute(&vmm, hipDeviceAttributeVirtualMemoryManagementSupported, a.dev) != hipSuccess || !vmm) {
            (void) hipGetLastError();
            return;
        }
        const hipMemAllocationProp prop = alloc_prop(a.dev);
        size_t g = 0;
        if (hipMemGetAllocationGranularity(&g, &prop, hipMemAllocationGranularityMinimum) != hipSuccess || g == 0) {
            (void) hipGetLastError();
            return;
        }
        size_t recommended = 0;
        if (hipMemGetAllocationGranularity(&recommended, &prop, hipMemAllocationGranularityRecommended) != hipSuccess) {
            (void) hipGetLastError();
            a.why = "cannot read the recommended allocation granularity";
            return;
        }
        a.gran = recommended;
        if (recommended == 0 || recommended % g != 0 || !runtime_check(a.dev, a.gran, a.why)) {
            a.why += " (" + runtime_name() + ")";
            return;
        }
        a.why.clear();
        a.ok = true;
    }();
    return a;
}

}  // namespace

bool vmm_available() { return api().ok; }
std::string vmm_off_reason() { return api().ok ? "" : api().why; }
uint64_t vmm_granularity() { return api().ok ? api().gran : 0; }

VmmChunk vmm_chunk_new() {
    const Api& a = api();
    if (!a.ok) return 0;
    const hipMemAllocationProp prop = alloc_prop(a.dev);
    hipMemGenericAllocationHandle_t h = nullptr;
    if (const hipError_t e = hipMemCreate(&h, (size_t) a.gran, &prop, 0); e != hipSuccess) { fail("hipMemCreate", e); return 0; }
    return (VmmChunk) (uintptr_t) h;
}

void vmm_chunk_free(VmmChunk h) {
    if (h != 0 && api().ok) (void) hipMemRelease((hipMemGenericAllocationHandle_t) (uintptr_t) h);
}

bool VmmRange::reserve(uint64_t bytes) {
    release();
    const Api& a = api();
    if (!a.ok || bytes == 0) return false;
    const uint64_t n = (bytes + a.gran - 1) / a.gran;
    void* p = nullptr;
    if (const hipError_t e = hipMemAddressReserve(&p, (size_t) (n * a.gran), 0, nullptr, 0); e != hipSuccess) { fail("hipMemAddressReserve", e); return false; }
    base_ = (unsigned long long) (uintptr_t) p;
    h_.assign((size_t) n, 0);
    return true;
}

void VmmRange::release() {
    if (base_ == 0) return;
    const Api& a = api();
    for (int64_t i = 0; i < chunks(); ++i) vmm_chunk_free(unmap(i));
    (void) hipMemAddressFree((void*) (uintptr_t) base_, (size_t) ((uint64_t) h_.size() * a.gran));
    base_ = 0;
    h_.clear();
}

int64_t VmmRange::mapped_count() const {
    int64_t n = 0;
    for (const VmmChunk h : h_) n += h != 0;
    return n;
}

bool VmmRange::map_one(int64_t i, VmmChunk h) {
    const Api& a = api();
    if (const hipError_t e = hipMemMap((void*) (uintptr_t) (base_ + (uint64_t) i * a.gran), (size_t) a.gran, 0,
                                      (hipMemGenericAllocationHandle_t) (uintptr_t) h, 0);
        e != hipSuccess) {
        fail("hipMemMap", e);
        return false;
    }
    h_[(size_t) i] = h;
    return true;
}

bool VmmRange::set_access(int64_t lo, int64_t hi) {
    if (hi <= lo) return true;
    const Api& a = api();
    hipMemAccessDesc d{};
    d.location.type = hipMemLocationTypeDevice;
    d.location.id = a.dev;
    d.flags = hipMemAccessFlagsProtReadWrite;
    const hipError_t e = hipMemSetAccess((void*) (uintptr_t) (base_ + (uint64_t) lo * a.gran), (size_t) ((uint64_t) (hi - lo) * a.gran),
                                         &d, 1);
    const bool ok = e == hipSuccess;
    if (!ok) fail("hipMemSetAccess", e);
    return ok;
}

bool VmmRange::commit_run(int64_t lo, int64_t hi) {
    if (set_access(lo, hi)) return true;
    for (int64_t c = lo; c < hi; ++c) vmm_chunk_free(unmap(c));
    return false;
}

VmmChunk VmmRange::unmap(int64_t i) {
    if (!mapped(i)) return 0;
    const Api& a = api();
    const VmmChunk h = h_[(size_t) i];
    if (hipMemUnmap((void*) (uintptr_t) (base_ + (uint64_t) i * a.gran), (size_t) a.gran) != hipSuccess) { (void) hipGetLastError(); return 0; }
    h_[(size_t) i] = 0;
    return h;
}

bool vmm_device_supported() { return vmm_available(); }
uint64_t vmm_segment_granularity() { return vmm_granularity(); }
uintptr_t vmm_reserve(uint64_t bytes) {
    uintptr_t va = 0;
    return api().ok && p_reserve(bytes, va) ? va : 0;
}
void vmm_address_free(uintptr_t va, uint64_t bytes) { p_address_free(va, bytes); }
bool vmm_segment_map(uintptr_t va, uint64_t bytes, VmmChunk& handle) {
    VmmChunk h = 0;
    if (!api().ok || !p_create(api().dev, bytes, h)) return false;
    if (!p_map(va, bytes, h)) { p_release(h); return false; }
    if (!p_access(api().dev, va, bytes)) { p_unmap(va, bytes); p_release(h); return false; }
    handle = h;
    return true;
}
bool vmm_segment_unmap(uintptr_t va, uint64_t bytes, VmmChunk h) {
    return p_unmap(va, bytes) && p_release(h);
}
}  // namespace strata::core
#else


namespace strata::core {
bool vmm_available() { return false; }
std::string vmm_off_reason() { return "this build has no GPU virtual memory"; }
uint64_t vmm_granularity() { return 0; }
VmmChunk vmm_chunk_new() { return 0; }
void vmm_chunk_free(VmmChunk) {}
bool VmmRange::reserve(uint64_t) { return false; }
void VmmRange::release() {}
int64_t VmmRange::mapped_count() const { return 0; }
bool VmmRange::map_one(int64_t, VmmChunk) { return false; }
bool VmmRange::set_access(int64_t, int64_t) { return false; }
bool VmmRange::commit_run(int64_t, int64_t) { return false; }
VmmChunk VmmRange::unmap(int64_t) { return 0; }

bool vmm_device_supported() { return false; }
uint64_t vmm_segment_granularity() { return 0; }
uintptr_t vmm_reserve(uint64_t) { return 0; }
void vmm_address_free(uintptr_t, uint64_t) {}
bool vmm_segment_map(uintptr_t, uint64_t, VmmChunk&) { return false; }
bool vmm_segment_unmap(uintptr_t, uint64_t, VmmChunk) { return false; }
}  // namespace strata::core

#endif
#endif
