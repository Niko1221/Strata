#include "strata/core/vmm.hpp"

// One set of driver operations (reserve, create, map, set access, unmap, release, free the addresses) per backend,
// and everything else - the chunks, VmmRange, #533's segments - written once on top of it.  The gfx906 compat build
// compiles this file as HIP without STRATA_USE_HIP and was never measured with virtual memory, and Metal has none: no
// backend there.
#if defined(STRATA_HIP_GFX906) || defined(STRATA_GPU_METAL)
#define STRATA_VMM_NONE 1
#elif !defined(STRATA_USE_HIP)
#include <cuda.h>
#include <cuda_runtime.h>
#endif

#include <mutex>

namespace strata::core {

#if defined(STRATA_VMM_NONE)

bool vmm_available() { return false; }
uint64_t vmm_granularity() { return 0; }
VmmChunk vmm_chunk_new() { return 0; }
void vmm_chunk_free(VmmChunk) {}
bool vmm_device_supported() { return false; }
uint64_t vmm_segment_granularity() { return 0; }
uintptr_t vmm_reserve(uint64_t) { return 0; }
void vmm_address_free(uintptr_t, uint64_t) {}
bool vmm_segment_map(uintptr_t, uint64_t, VmmChunk&) { return false; }
bool vmm_segment_unmap(uintptr_t, uint64_t, VmmChunk) { return false; }
bool VmmRange::reserve(uint64_t) { return false; }
void VmmRange::release() {}
int64_t VmmRange::mapped_count() const { return 0; }
bool VmmRange::map_one(int64_t, VmmChunk) { return false; }
bool VmmRange::set_access(int64_t, int64_t) { return false; }
bool VmmRange::commit_run(int64_t, int64_t) { return false; }
VmmChunk VmmRange::unmap(int64_t) { return 0; }

#else

namespace {

#if defined(STRATA_USE_HIP)
// ---- HIP (the RDNA backend): the runtime's own functions.  A failed call also sets the runtime's last error, which
// later cudaGetLastError checks would read as theirs: every failure clears it.  Measured on gfx1201 / ROCm 7.14.1
// (Linux) before this was written: the VRAM of a released handle is back for other programs within 10 ms while its
// addresses stay reserved, a range cannot be freed in part, and one hipMemSetAccess or hipMemcpy cannot span two
// reservations - so, as on CUDA, one reservation per range.

// the chunk is the RECOMMENDED granularity: the minimum one is 4 KiB here, and hipMemGetInfo counts each such chunk as
// 2 MiB (the recommended size, which is CUDA's minimum)
constexpr bool kChunkRecommended = true;

bool fns_ok() { return true; }

int current_device() {
    int d = -1;
    return hipGetDevice(&d) == hipSuccess ? d : -1;
}

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

bool dev_supported(int dev) {
    int v = 0;
    if (hipDeviceGetAttribute(&v, hipDeviceAttributeVirtualMemoryManagementSupported, dev) != hipSuccess)
        return failed();
    return v != 0;
}

uint64_t dev_granularity(int dev, bool recommended) {
    const hipMemAllocationProp prop = device_prop(dev);
    size_t g = 0;
    if (hipMemGetAllocationGranularity(&g, &prop, recommended ? hipMemAllocationGranularityRecommended
                                                              : hipMemAllocationGranularityMinimum) != hipSuccess) {
        failed();
        return 0;
    }
    return (uint64_t) g;
}

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

#else
// ---- CUDA: the driver API, reached through the runtime's entry-point query (nothing links the driver library)

constexpr bool kChunkRecommended = false;   // CUDA's minimum granularity is the 2 MiB chunk

struct Fns {
    bool ok = false;
    decltype(&cuDeviceGet) device_get = nullptr;
    decltype(&cuDeviceGetAttribute) attribute = nullptr;
    decltype(&cuMemGetAllocationGranularity) granularity = nullptr;
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
const Fns& fns() {
    static Fns& f = *new Fns;
    static std::once_flag once;
    std::call_once(once, [] {
        f.ok = resolve("cuDeviceGet", f.device_get) && resolve("cuDeviceGetAttribute", f.attribute) &&
               resolve("cuMemGetAllocationGranularity", f.granularity) &&
               resolve("cuMemAddressReserve", f.reserve) && resolve("cuMemAddressFree", f.free_va) &&
               resolve("cuMemCreate", f.create) && resolve("cuMemRelease", f.release) &&
               resolve("cuMemMap", f.map) && resolve("cuMemUnmap", f.unmap) && resolve("cuMemSetAccess", f.access);
    });
    return f;
}

bool fns_ok() { return fns().ok; }

int current_device() {
    int d = -1;
    return cudaGetDevice(&d) == cudaSuccess ? d : -1;
}

CUmemAllocationProp device_prop(int dev) {
    CUmemAllocationProp prop{};
    prop.type = CU_MEM_ALLOCATION_TYPE_PINNED;
    prop.location.type = CU_MEM_LOCATION_TYPE_DEVICE;
    prop.location.id = dev;
    return prop;
}

bool dev_supported(int dev) {
    CUdevice cu = 0;
    int v = 0;
    return fns().device_get(&cu, dev) == CUDA_SUCCESS &&
           fns().attribute(&v, CU_DEVICE_ATTRIBUTE_VIRTUAL_MEMORY_MANAGEMENT_SUPPORTED, cu) == CUDA_SUCCESS && v != 0;
}

uint64_t dev_granularity(int dev, bool recommended) {
    const CUmemAllocationProp prop = device_prop(dev);
    size_t g = 0;
    if (fns().granularity(&g, &prop, recommended ? CU_MEM_ALLOC_GRANULARITY_RECOMMENDED
                                                 : CU_MEM_ALLOC_GRANULARITY_MINIMUM) != CUDA_SUCCESS)
        return 0;
    return (uint64_t) g;
}

bool p_reserve(uint64_t bytes, uintptr_t& va) {
    CUdeviceptr p = 0;
    if (fns().reserve(&p, (size_t) bytes, 0, 0, 0) != CUDA_SUCCESS) return false;
    va = (uintptr_t) p;
    return true;
}

void p_address_free(uintptr_t va, uint64_t bytes) { fns().free_va((CUdeviceptr) va, (size_t) bytes); }

bool p_create(int dev, uint64_t bytes, VmmChunk& h) {
    const CUmemAllocationProp prop = device_prop(dev);
    CUmemGenericAllocationHandle x = 0;
    if (fns().create(&x, (size_t) bytes, &prop, 0) != CUDA_SUCCESS) return false;
    h = (VmmChunk) x;
    return true;
}

bool p_release(VmmChunk h) { return fns().release((CUmemGenericAllocationHandle) h) == CUDA_SUCCESS; }

bool p_map(uintptr_t va, uint64_t bytes, VmmChunk h) {
    return fns().map((CUdeviceptr) va, (size_t) bytes, 0, (CUmemGenericAllocationHandle) h, 0) == CUDA_SUCCESS;
}

bool p_unmap(uintptr_t va, uint64_t bytes) { return fns().unmap((CUdeviceptr) va, (size_t) bytes) == CUDA_SUCCESS; }

bool p_access(int dev, uintptr_t va, uint64_t bytes) {
    CUmemAccessDesc d{};
    d.location.type = CU_MEM_LOCATION_TYPE_DEVICE;
    d.location.id = dev;
    d.flags = CU_MEM_ACCESS_FLAGS_PROT_READWRITE;
    return fns().access((CUdeviceptr) va, (size_t) bytes, &d, 1) == CUDA_SUCCESS;
}
#endif

// ---- both backends from here on

struct Api {
    bool ok = false;
    uint64_t gran = 0;   // the chunk
    int dev = -1;        // the elastic K/V's device: the one current at the first call
};

// never destroyed: the K/V pools and the cache release their chunks from static destructors at exit
const Api& api() {
    static Api& a = *new Api;
    static std::once_flag once;
    std::call_once(once, [] {
        const int dev = current_device();
        if (dev < 0 || !fns_ok() || !dev_supported(dev)) return;
        const uint64_t g = dev_granularity(dev, kChunkRecommended);
        if (g == 0) return;
        a.dev = dev;
        a.gran = g;
        a.ok = true;
    });
    return a;
}

}  // namespace

bool vmm_available() { return api().ok; }
uint64_t vmm_granularity() { return api().ok ? api().gran : 0; }

VmmChunk vmm_chunk_new() {
    const Api& a = api();
    VmmChunk h = 0;
    if (!a.ok || !p_create(a.dev, a.gran, h)) return 0;
    return h;
}

void vmm_chunk_free(VmmChunk h) {
    if (h != 0 && api().ok) p_release(h);
}

bool vmm_device_supported() {
    const int dev = current_device();
    return dev >= 0 && fns_ok() && dev_supported(dev);
}

uint64_t vmm_segment_granularity() {
    const int dev = current_device();
    return dev >= 0 && fns_ok() ? dev_granularity(dev, true) : 0;
}

uintptr_t vmm_reserve(uint64_t bytes) {
    uintptr_t va = 0;
    return fns_ok() && bytes > 0 && p_reserve(bytes, va) ? va : 0;
}

void vmm_address_free(uintptr_t va, uint64_t bytes) {
    if (va != 0 && fns_ok()) p_address_free(va, bytes);
}

bool vmm_segment_map(uintptr_t va, uint64_t bytes, VmmChunk& handle) {
    const int dev = current_device();
    VmmChunk h = 0;
    if (dev < 0 || !fns_ok() || !p_create(dev, bytes, h)) return false;
    if (!p_map(va, bytes, h)) {
        p_release(h);
        return false;
    }
    if (!p_access(dev, va, bytes)) {
        p_unmap(va, bytes);
        p_release(h);
        return false;
    }
    handle = h;
    return true;
}

bool vmm_segment_unmap(uintptr_t va, uint64_t bytes, VmmChunk handle) {
    return fns_ok() && p_unmap(va, bytes) && p_release(handle);
}

bool VmmRange::reserve(uint64_t bytes) {
    release();
    const Api& a = api();
    if (!a.ok || bytes == 0) return false;
    const uint64_t n = (bytes + a.gran - 1) / a.gran;
    uintptr_t p = 0;
    if (!p_reserve(n * a.gran, p)) return false;
    base_ = (unsigned long long) p;
    h_.assign((size_t) n, 0);
    return true;
}

void VmmRange::release() {
    if (base_ == 0) return;
    const Api& a = api();
    for (int64_t i = 0; i < chunks(); ++i) vmm_chunk_free(unmap(i));
    p_address_free((uintptr_t) base_, (uint64_t) h_.size() * a.gran);
    base_ = 0;
    h_.clear();
}

int64_t VmmRange::mapped_count() const {
    int64_t n = 0;
    for (const VmmChunk h : h_) n += h != 0;
    return n;
}

bool VmmRange::map_one(int64_t i, VmmChunk h) {
    if (!p_map((uintptr_t) (base_ + (uint64_t) i * api().gran), api().gran, h)) return false;
    h_[(size_t) i] = h;
    return true;
}

bool VmmRange::set_access(int64_t lo, int64_t hi) {
    if (hi <= lo) return true;
    const Api& a = api();
    return p_access(a.dev, (uintptr_t) (base_ + (uint64_t) lo * a.gran), (uint64_t) (hi - lo) * a.gran);
}

bool VmmRange::commit_run(int64_t lo, int64_t hi) {
    if (set_access(lo, hi)) return true;
    for (int64_t c = lo; c < hi; ++c) vmm_chunk_free(unmap(c));
    return false;
}

VmmChunk VmmRange::unmap(int64_t i) {
    if (!mapped(i)) return 0;
    const VmmChunk h = h_[(size_t) i];
    if (!p_unmap((uintptr_t) (base_ + (uint64_t) i * api().gran), api().gran)) return 0;
    h_[(size_t) i] = 0;
    return h;
}

#endif

}  // namespace strata::core
