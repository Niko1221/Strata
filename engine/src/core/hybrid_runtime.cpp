// CUDA host path. Ordinals past the CUDA devices are HIP. hybrid_hip.cpp owns
// those calls. A hybrid ordinal must not be passed to hipSetDevice.
#include <cuda_runtime.h>

#include <cublas_v2.h>

#include "hip_api.h"

#include <dlfcn.h>

#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <mutex>
#include <string>
#include <unordered_set>
#include <vector>

// CUDA 13 dropped the versioned cudart symbols and carries the _v2/_v3
// signatures under the plain names. The shim pairs and the --wrap list in
// cmake/hybrid.cmake flip on the same switch; the CUDA-side callers follow
// the same idiom already used in src/core/verify.cpp.
#if CUDART_VERSION >= 13000
#define strata_real_getDeviceProperties __real_cudaGetDeviceProperties
#define strata_wrap_getDeviceProperties __wrap_cudaGetDeviceProperties
#define strata_real_graphGetEdges __real_cudaGraphGetEdges
#define strata_wrap_graphGetEdges __wrap_cudaGraphGetEdges
#define strata_real_streamGetCaptureInfo __real_cudaStreamGetCaptureInfo
#define strata_wrap_streamGetCaptureInfo __wrap_cudaStreamGetCaptureInfo
#else
#define strata_real_getDeviceProperties __real_cudaGetDeviceProperties_v2
#define strata_wrap_getDeviceProperties __wrap_cudaGetDeviceProperties_v2
#define strata_real_graphGetEdges __real_cudaGraphGetEdges_v2
#define strata_wrap_graphGetEdges __wrap_cudaGraphGetEdges_v2
#define strata_real_streamGetCaptureInfo __real_cudaStreamGetCaptureInfo_v3
#define strata_wrap_streamGetCaptureInfo __wrap_cudaStreamGetCaptureInfo_v3
#endif

extern "C" __attribute__((tls_model("initial-exec"))) thread_local int strata_hybrid_on_hip;

namespace {

struct Interval {
    uintptr_t base;
    uintptr_t end;
    int hip_index;
};

int g_cuda_count = 0;
int g_hip_count = 0;
int g_hip_overflow = 0;
std::string g_hip_arch[8];
std::string g_hip_name[8];
std::once_flag g_once;
thread_local int g_device = 0;
thread_local dim3 g_grid{};
thread_local dim3 g_block{};
thread_local size_t g_shared = 0;
thread_local cudaStream_t g_stream = nullptr;
thread_local bool g_have_cfg = false;

std::mutex g_mu;
std::vector<Interval> g_hip_mem;
std::unordered_set<void*> g_hip_host;
std::unordered_set<void*> g_hip_stream;
std::unordered_set<void*> g_hip_event;
std::unordered_set<void*> g_hip_graph;
std::unordered_set<void*> g_hip_exec;
std::unordered_set<void*> g_hip_blas;

cudaError_t Ce(int e) { return static_cast<cudaError_t>(e); }

}  // namespace

extern "C" cudaError_t __real_cudaGetDeviceCount(int*);
extern "C" cudaError_t __real_cudaGetLastError(void);
extern "C" cudaError_t __real_cudaSetDevice(int);

namespace {

void sync_flag() {
    const bool hip = g_device >= g_cuda_count && g_device < g_cuda_count + g_hip_count;
    strata_hybrid_on_hip = hip ? 1 : 0;
}

void ensure() {
    std::call_once(g_once, [] {
        int n = 0;
        if (__real_cudaGetDeviceCount(&n) != cudaSuccess || n < 0) n = 0;
        (void) __real_cudaGetLastError();
        g_cuda_count = n;
        const int raw = hh_device_count();
        if (raw > 8) {
            std::fprintf(stderr, "strata hybrid: %d HIP devices; this build supports 8\n", raw);
            g_hip_overflow = raw;
            g_hip_count = 0;
        } else {
            HhDevice devs[8]{};
            int h = hh_devices(devs, 8);
            if (h < 0) h = 0;
            if (h > 8) h = 8;
            g_hip_count = h;
            for (int i = 0; i < h; ++i) {
                g_hip_name[i] = devs[i].name;
                g_hip_arch[i] = devs[i].arch;
            }
        }
        std::fprintf(stderr, "strata hybrid: %d CUDA device(s), %d HIP device(s)\n", g_cuda_count, g_hip_count);
        for (int i = 0; i < g_hip_count; ++i)
            std::fprintf(stderr, "strata hybrid: device %d: %s %s\n", g_cuda_count + i, g_hip_name[i].c_str(),
                         g_hip_arch[i].c_str());
    });
    sync_flag();
}

bool hip_ordinal(int d) {
    ensure();
    return d >= g_cuda_count && d < g_cuda_count + g_hip_count;
}

bool on_hip() {
    ensure();
    return hip_ordinal(g_device);
}

void bind_hip_index(int hip_i) {
    if (hip_i < 0 || hip_i >= g_hip_count) return;
    (void) hh_set_device(hip_i);
}

void bind_hip() {
    ensure();
    if (!on_hip()) return;
    bind_hip_index(g_device - g_cuda_count);
}

int mem_hip_index(const void* p);

void bind_tracked(const void* a, const void* b = nullptr) {
    int hip_i = mem_hip_index(a);
    if (hip_i < 0 && b != nullptr) hip_i = mem_hip_index(b);
    if (hip_i >= 0) bind_hip_index(hip_i);
    else bind_hip();
}

template <typename Set>
void track(Set& set, void* p) {
    if (p == nullptr) return;
    std::lock_guard<std::mutex> lock(g_mu);
    set.insert(p);
}
template <typename Set>
void untrack(Set& set, void* p) {
    if (p == nullptr) return;
    std::lock_guard<std::mutex> lock(g_mu);
    set.erase(p);
}
template <typename Set>
bool has(const Set& set, void* p) {
    if (p == nullptr) return false;
    std::lock_guard<std::mutex> lock(g_mu);
    return set.count(p) != 0;
}

void track_mem(void* p, size_t n, int hip_index) {
    if (p == nullptr || hip_index < 0) return;
    std::lock_guard<std::mutex> lock(g_mu);
    g_hip_mem.push_back(
        Interval{reinterpret_cast<uintptr_t>(p), reinterpret_cast<uintptr_t>(p) + (n == 0 ? 1 : n), hip_index});
}
int mem_hip_index(const void* p) {
    if (p == nullptr) return -1;
    auto a = reinterpret_cast<uintptr_t>(p);
    std::lock_guard<std::mutex> lock(g_mu);
    for (const auto& iv : g_hip_mem)
        if (a >= iv.base && a < iv.end) return iv.hip_index;
    return -1;
}
void untrack_mem(void* p) {
    if (p == nullptr) return;
    auto b = reinterpret_cast<uintptr_t>(p);
    std::lock_guard<std::mutex> lock(g_mu);
    std::vector<Interval> kept;
    for (const auto& iv : g_hip_mem)
        if (iv.base != b) kept.push_back(iv);
    g_hip_mem.swap(kept);
}
bool mem_is_hip(const void* p) {
    if (p == nullptr) return false;
    auto a = reinterpret_cast<uintptr_t>(p);
    std::lock_guard<std::mutex> lock(g_mu);
    for (const auto& iv : g_hip_mem)
        if (a >= iv.base && a < iv.end) return true;
    return false;
}
bool hip_copy(const void* a, const void* b) { return on_hip() || mem_is_hip(a) || mem_is_hip(b); }
bool hip_stream(cudaStream_t s) { return s != nullptr && has(g_hip_stream, s); }

void* hip_kernel(const void* fn) {
    Dl_info info{};
    if (fn == nullptr || dladdr(fn, &info) == 0 || info.dli_sname == nullptr) return nullptr;
    std::string name = info.dli_sname;
    if (name.rfind("hipdual_", 0) == 0) return const_cast<void*>(fn);
    name = "hipdual_" + name;
    void* hip = dlsym(RTLD_DEFAULT, name.c_str());
    if (hip == nullptr) {
        static int said = 0;
        if (said++ < 8) std::fprintf(stderr, "strata hybrid: no HIP kernel for %s\n", info.dli_sname);
    }
    return hip;
}

void fill_prop(cudaDeviceProp* out, int hip_i) {
    std::memset(out, 0, sizeof(*out));
    HhDevice devs[8]{};
    int n = hh_devices(devs, 8);
    if (hip_i < 0 || hip_i >= n) return;
    std::snprintf(out->name, sizeof out->name, "%s", devs[hip_i].name);
    out->totalGlobalMem = devs[hip_i].total_mem;
    out->sharedMemPerBlock = devs[hip_i].shared_mem;
    out->warpSize = devs[hip_i].warp;
    out->multiProcessorCount = devs[hip_i].sms;
    out->maxThreadsPerBlock = devs[hip_i].max_threads;
#if CUDART_VERSION < 13000
    // CUDA 13 removed cudaDeviceProp::clockRate. fill_prop memsets the
    // struct, and no CUDA-side consumer in this tree reads a clock field.
    out->clockRate = devs[hip_i].clock_khz;
#endif
    out->l2CacheSize = devs[hip_i].l2;
    out->integrated = devs[hip_i].integrated;
    out->pciBusID = devs[hip_i].pci_bus;
    out->pciDeviceID = devs[hip_i].pci_dev;
    // major/minor stay 0. ROCm's compute capability is the ISA version, so
    // gfx1201 would otherwise look like sm_120.
}

}  // namespace

// The host trampoline (cmake/hip-host-tramp.py) reads this with a %fs
// initial-exec load and then tail-jumps. It has to stay a signed 32-bit TLS
// offset in the executable, and it has to change only on this thread.
extern "C" __attribute__((tls_model("initial-exec"))) thread_local int strata_hybrid_on_hip = 0;

extern "C" int strata_hybrid_hip_overflow() {
    ensure();
    return g_hip_overflow;
}
extern "C" void strata_hybrid_note_device(int ordinal) {
    ensure();
    if (ordinal >= 0 && ordinal < g_cuda_count) {
        g_device = ordinal;
        strata_hybrid_on_hip = 0;
        return;
    }
    if (!hip_ordinal(ordinal)) return;
    g_device = ordinal;
    strata_hybrid_on_hip = 1;
}
extern "C" int strata_hybrid_current_ordinal() {
    ensure();
    return g_device;
}
extern "C" int strata_hybrid_device_count() {
    ensure();
    // Negative when more than 8 HIP devices were dropped. Callers that only
    // look at the CUDA count would keep going with those GPUs missing.
    if (g_hip_overflow > 8) return -g_hip_overflow;
    return g_cuda_count + g_hip_count;
}
extern "C" int strata_hybrid_is_hip_device(int ordinal) {
    ensure();
    return hip_ordinal(ordinal) ? 1 : 0;
}
extern "C" int strata_hybrid_hip_index(int ordinal) {
    ensure();
    if (!hip_ordinal(ordinal)) return -1;
    return ordinal - g_cuda_count;
}
extern "C" const char* strata_hybrid_arch(int ordinal) {
    ensure();
    if (!hip_ordinal(ordinal)) return "";
    return g_hip_arch[ordinal - g_cuda_count].c_str();
}

// Map a CUDA-registered pin so each HIP device can read it in place. One
// registration, then one device pointer per HIP device. This does not change
// the CUDA device or strata_hybrid_on_hip, and the pointer stays out of
// g_hip_host so the primary's own alias is unchanged. A device whose first
// 64 bytes do not match is left null. If none match, the registration is
// dropped and the caller keeps those rows on the CPU.
extern "C" int strata_hip_map_host(void* host, size_t bytes, void** devs, int cap) {
    if (host == nullptr || devs == nullptr || cap < 1 || bytes < 64) return 1;
    for (int i = 0; i < cap; ++i) devs[i] = nullptr;
    ensure();
    if (g_hip_count < 1) return 2;
    const int n = g_hip_count < cap ? g_hip_count : cap;
    int prev = 0;
    const bool restore = hh_get_device(&prev) == 0;
    // hipHostRegisterPortable | hipHostRegisterMapped, the same bits CUDA used.
    const int reg = hh_host_register(host, bytes, 1u | 2u);
    if (reg != 0) {
        if (restore) (void) hh_set_device(prev);
        return reg;
    }
    int ok = 0;
    for (int i = 0; i < n; ++i) {
        if (hh_set_device(i) != 0) continue;
        void* p = nullptr;
        if (hh_host_device_pointer(&p, host, 0) != 0 || p == nullptr) continue;
        unsigned char probe[64];
        // hipMemcpyDeviceToHost. The mapped bytes have to match the pin.
        if (hh_memcpy(probe, p, 64, 2) != 0 || std::memcmp(probe, host, 64) != 0) continue;
        devs[i] = p;
        ++ok;
    }
    if (restore) (void) hh_set_device(prev);
    if (ok == 0) {
        (void) hh_host_unregister(host);
        for (int i = 0; i < cap; ++i) devs[i] = nullptr;
        return 5;
    }
    return 0;
}
extern "C" int strata_hip_unmap_host(void* host) {
    if (host == nullptr) return 0;
    ensure();
    if (g_hip_count < 1) return 0;
    int prev = 0;
    const bool restore = hh_get_device(&prev) == 0;
    const int rc = hh_host_unregister(host);
    if (restore) (void) hh_set_device(prev);
    return rc;
}

extern "C" cudaError_t __wrap_cudaGetDeviceCount(int* n) {
    if (n == nullptr) return cudaErrorInvalidValue;
    ensure();
    // MMQ walks this count and aborts on a HIP ordinal. The full count is
    // strata_hybrid_device_count().
    *n = g_cuda_count;
    return cudaSuccess;
}
extern "C" cudaError_t __wrap_cudaSetDevice(int d) {
    ensure();
    if (d >= 0 && d < g_cuda_count) {
        g_device = d;
        strata_hybrid_on_hip = 0;
        return __real_cudaSetDevice(d);
    }
    if (!hip_ordinal(d)) return cudaErrorInvalidDevice;
    int e = hh_set_device(d - g_cuda_count);
    if (e != 0) return Ce(e);
    g_device = d;
    strata_hybrid_on_hip = 1;
    return cudaSuccess;
}
extern "C" cudaError_t __real_cudaGetDevice(int*);
extern "C" cudaError_t __wrap_cudaGetDevice(int* d) {
    if (d == nullptr) return cudaErrorInvalidValue;
    ensure();
    *d = g_device;
    return cudaSuccess;
}
extern "C" cudaError_t strata_real_getDeviceProperties(cudaDeviceProp*, int);
extern "C" cudaError_t strata_wrap_getDeviceProperties(cudaDeviceProp* p, int d) {
    ensure();
    if (p == nullptr) return cudaErrorInvalidValue;
    if (!hip_ordinal(d)) return strata_real_getDeviceProperties(p, d);
    fill_prop(p, d - g_cuda_count);
    return cudaSuccess;
}
extern "C" cudaError_t __wrap_cudaGetLastError(void) {
    if (!on_hip()) return __real_cudaGetLastError();
    return Ce(hh_last_error());
}
extern "C" cudaError_t __real_cudaPeekAtLastError(void);
extern "C" cudaError_t __wrap_cudaPeekAtLastError(void) {
    if (!on_hip()) return __real_cudaPeekAtLastError();
    return Ce(hh_peek_error());
}
extern "C" const char* __real_cudaGetErrorString(cudaError_t);
extern "C" const char* __wrap_cudaGetErrorString(cudaError_t e) {
    if (!on_hip()) return __real_cudaGetErrorString(e);
    return hh_error_string((int) e);
}
extern "C" cudaError_t __real_cudaDeviceSynchronize(void);
extern "C" cudaError_t __wrap_cudaDeviceSynchronize(void) {
    if (!on_hip()) return __real_cudaDeviceSynchronize();
    bind_hip();
    return Ce(hh_sync());
}
extern "C" cudaError_t __real_cudaMalloc(void**, size_t);
extern "C" cudaError_t __wrap_cudaMalloc(void** p, size_t n) {
    if (!on_hip()) return __real_cudaMalloc(p, n);
    const int hip_i = g_device - g_cuda_count;
    if (hip_i < 0 || hip_i >= g_hip_count) return cudaErrorInvalidDevice;
    bind_hip_index(hip_i);
    int e = hh_malloc(p, n);
    if (e == 0 && p != nullptr) track_mem(*p, n, hip_i);
    return Ce(e);
}
extern "C" int strata_hybrid_hip_malloc(void** p, size_t n) { return (int) __wrap_cudaMalloc(p, n); }
extern "C" cudaError_t __real_cudaFree(void*);
extern "C" cudaError_t __wrap_cudaFree(void* p) {
    const int hip_i = mem_hip_index(p);
    if (hip_i >= 0) {
        bind_hip_index(hip_i);
        int e = hh_free(p);
        untrack_mem(p);
        return Ce(e);
    }
    return __real_cudaFree(p);
}
extern "C" int strata_hybrid_hip_free(void* p) { return (int) __wrap_cudaFree(p); }
extern "C" cudaError_t __real_cudaMallocHost(void**, size_t);
extern "C" cudaError_t __wrap_cudaMallocHost(void** p, size_t n) {
    if (!on_hip()) return __real_cudaMallocHost(p, n);
    bind_hip();
    int e = hh_host_malloc(p, n, 0);
    if (e == 0 && p != nullptr) track(g_hip_host, *p);
    return Ce(e);
}
extern "C" cudaError_t __real_cudaFreeHost(void*);
extern "C" cudaError_t __wrap_cudaFreeHost(void* p) {
    if (!has(g_hip_host, p)) return __real_cudaFreeHost(p);
    untrack(g_hip_host, p);
    return Ce(hh_host_free(p));
}
extern "C" cudaError_t __real_cudaHostAlloc(void**, size_t, unsigned int);
extern "C" cudaError_t __wrap_cudaHostAlloc(void** p, size_t n, unsigned int flags) {
    if (!on_hip()) return __real_cudaHostAlloc(p, n, flags);
    bind_hip();
    int e = hh_host_malloc(p, n, flags);
    if (e == 0 && p != nullptr) track(g_hip_host, *p);
    return Ce(e);
}
extern "C" cudaError_t __real_cudaHostRegister(void*, size_t, unsigned int);
extern "C" cudaError_t __wrap_cudaHostRegister(void* p, size_t n, unsigned int flags) {
    if (!on_hip()) return __real_cudaHostRegister(p, n, flags);
    bind_hip();
    int e = hh_host_register(p, n, flags);
    if (e == 0) track(g_hip_host, p);
    return Ce(e);
}
extern "C" cudaError_t __real_cudaHostUnregister(void*);
extern "C" cudaError_t __wrap_cudaHostUnregister(void* p) {
    if (!has(g_hip_host, p)) return __real_cudaHostUnregister(p);
    untrack(g_hip_host, p);
    return Ce(hh_host_unregister(p));
}
extern "C" cudaError_t __real_cudaHostGetDevicePointer(void**, void*, unsigned int);
extern "C" cudaError_t __wrap_cudaHostGetDevicePointer(void** dev, void* host, unsigned int flags) {
    if (!has(g_hip_host, host) && !on_hip()) return __real_cudaHostGetDevicePointer(dev, host, flags);
    bind_hip();
    int e = hh_host_device_pointer(dev, host, flags);
    if (e == 0 && dev != nullptr) track_mem(*dev, 1, on_hip() ? g_device - g_cuda_count : 0);
    return Ce(e);
}
extern "C" cudaError_t __real_cudaMemcpy(void*, const void*, size_t, cudaMemcpyKind);
extern "C" cudaError_t __wrap_cudaMemcpy(void* dst, const void* src, size_t n, cudaMemcpyKind kind) {
    if (!hip_copy(dst, src)) return __real_cudaMemcpy(dst, src, n, kind);
    bind_tracked(dst, src);
    return Ce(hh_memcpy(dst, src, n, (int) kind));
}
extern "C" cudaError_t __real_cudaMemcpyAsync(void*, const void*, size_t, cudaMemcpyKind, cudaStream_t);
extern "C" cudaError_t __wrap_cudaMemcpyAsync(void* dst, const void* src, size_t n, cudaMemcpyKind kind, cudaStream_t s) {
    if (!hip_copy(dst, src) && !hip_stream(s)) return __real_cudaMemcpyAsync(dst, src, n, kind, s);
    bind_tracked(dst, src);
    return Ce(hh_memcpy_async(dst, src, n, (int) kind, s));
}
extern "C" cudaError_t __real_cudaMemcpy2DAsync(void*, size_t, const void*, size_t, size_t, size_t, cudaMemcpyKind, cudaStream_t);
extern "C" cudaError_t __wrap_cudaMemcpy2DAsync(void* dst, size_t dp, const void* src, size_t sp, size_t w, size_t h,
                                               cudaMemcpyKind kind, cudaStream_t s) {
    if (!hip_copy(dst, src) && !hip_stream(s)) return __real_cudaMemcpy2DAsync(dst, dp, src, sp, w, h, kind, s);
    bind_tracked(dst, src);
    return Ce(hh_memcpy_2d_async(dst, dp, src, sp, w, h, (int) kind, s));
}
extern "C" cudaError_t __real_cudaMemset(void*, int, size_t);
extern "C" cudaError_t __wrap_cudaMemset(void* p, int v, size_t n) {
    if (!mem_is_hip(p) && !on_hip()) return __real_cudaMemset(p, v, n);
    bind_tracked(p);
    return Ce(hh_memset(p, v, n));
}
extern "C" cudaError_t __real_cudaMemsetAsync(void*, int, size_t, cudaStream_t);
extern "C" cudaError_t __wrap_cudaMemsetAsync(void* p, int v, size_t n, cudaStream_t s) {
    if (!mem_is_hip(p) && !hip_stream(s) && !on_hip()) return __real_cudaMemsetAsync(p, v, n, s);
    bind_tracked(p);
    return Ce(hh_memset_async(p, v, n, s));
}
extern "C" cudaError_t __real_cudaMemGetInfo(size_t*, size_t*);
extern "C" cudaError_t __wrap_cudaMemGetInfo(size_t* free_b, size_t* total_b) {
    if (!on_hip()) return __real_cudaMemGetInfo(free_b, total_b);
    bind_hip();
    return Ce(hh_mem_info(free_b, total_b));
}

extern "C" cudaError_t __wrap_cudaStreamSynchronize(cudaStream_t);
extern "C" cudaError_t __real_cudaMemcpyPeer(void*, int, const void*, int, size_t);
cudaError_t peer_bounce(void* dst, int dst_dev, const void* src, int src_dev, size_t n) {
    constexpr size_t kChunk = 1u << 20;
    void* buf = std::malloc(n < kChunk ? (n == 0 ? 1 : n) : kChunk);
    if (buf == nullptr) return cudaErrorMemoryAllocation;
    const int saved = g_device;
    cudaError_t e = cudaSuccess;
    for (size_t off = 0; off < n && e == cudaSuccess;) {
        const size_t m = n - off < kChunk ? n - off : kChunk;
        e = __wrap_cudaSetDevice(src_dev);
        if (e == cudaSuccess)
            e = __wrap_cudaMemcpy(static_cast<char*>(buf), static_cast<const char*>(src) + off, m, cudaMemcpyDeviceToHost);
        if (e == cudaSuccess) e = __wrap_cudaSetDevice(dst_dev);
        if (e == cudaSuccess)
            e = __wrap_cudaMemcpy(static_cast<char*>(dst) + off, buf, m, cudaMemcpyHostToDevice);
        off += m;
    }
    (void) __wrap_cudaSetDevice(saved);
    std::free(buf);
    return e;
}
extern "C" cudaError_t __wrap_cudaMemcpyPeer(void* dst, int dst_dev, const void* src, int src_dev, size_t n) {
    if (!hip_ordinal(dst_dev) && !hip_ordinal(src_dev)) return __real_cudaMemcpyPeer(dst, dst_dev, src, src_dev, n);
    return peer_bounce(dst, dst_dev, src, src_dev, n);
}
extern "C" cudaError_t __real_cudaMemcpyPeerAsync(void*, int, const void*, int, size_t, cudaStream_t);
extern "C" cudaError_t __wrap_cudaMemcpyPeerAsync(void* dst, int dst_dev, const void* src, int src_dev, size_t n, cudaStream_t s) {
    if (!hip_ordinal(dst_dev) && !hip_ordinal(src_dev)) return __real_cudaMemcpyPeerAsync(dst, dst_dev, src, src_dev, n, s);
    // The bounce finishes before return, so work already queued on s has to
    // finish first. A captured stream cannot take that wait.
    if (s != nullptr) {
        cudaStreamCaptureStatus cap = cudaStreamCaptureStatusNone;
        const cudaError_t query = cudaStreamIsCapturing(s, &cap);
        if (query == cudaSuccess && cap != cudaStreamCaptureStatusNone) return cudaErrorStreamCaptureUnsupported;
        if (query != cudaSuccess) (void) cudaGetLastError();
    }
    if (const cudaError_t waited = cudaStreamSynchronize(s); waited != cudaSuccess) return waited;
    return peer_bounce(dst, dst_dev, src, src_dev, n);
}

extern "C" unsigned __real___cudaPushCallConfiguration(dim3, dim3, size_t, void*);
extern "C" unsigned __wrap___cudaPushCallConfiguration(dim3 grid, dim3 block, size_t shared, void* stream) {
    if (!on_hip() && !hip_stream(static_cast<cudaStream_t>(stream))) {
        g_have_cfg = false;
        return __real___cudaPushCallConfiguration(grid, block, shared, stream);
    }
    g_grid = grid;
    g_block = block;
    g_shared = shared;
    g_stream = static_cast<cudaStream_t>(stream);
    g_have_cfg = true;
    // nvcc launches the stub only when this returns 0.
    return 0;
}
extern "C" cudaError_t __real___cudaPopCallConfiguration(dim3*, dim3*, size_t*, void*);
extern "C" cudaError_t __wrap___cudaPopCallConfiguration(dim3* grid, dim3* block, size_t* shared, void* stream) {
    if (!g_have_cfg) return __real___cudaPopCallConfiguration(grid, block, shared, stream);
    if (grid) *grid = g_grid;
    if (block) *block = g_block;
    if (shared) *shared = g_shared;
    if (stream) *static_cast<cudaStream_t*>(stream) = g_stream;
    g_have_cfg = false;
    return cudaSuccess;
}
extern "C" cudaError_t __real_cudaLaunchKernel(const void*, dim3, dim3, void**, size_t, cudaStream_t);
extern "C" cudaError_t __wrap_cudaLaunchKernel(const void* fn, dim3 grid, dim3 block, void** args, size_t shared, cudaStream_t stream) {
    if (!on_hip() && !hip_stream(stream)) return __real_cudaLaunchKernel(fn, grid, block, args, shared, stream);
    bind_hip();
    void* hip = hip_kernel(fn);
    if (hip == nullptr) return cudaErrorInvalidDeviceFunction;
    return Ce(hh_launch(hip, grid.x, grid.y, grid.z, block.x, block.y, block.z, args, shared, stream));
}
extern "C" cudaError_t __real_cudaLaunchKernelExC(const cudaLaunchConfig_t*, const void*, void**);
extern "C" cudaError_t __wrap_cudaLaunchKernelExC(const cudaLaunchConfig_t* cfg, const void* fn, void** args) {
    if (cfg == nullptr) return cudaErrorInvalidValue;
    if (!on_hip() && !hip_stream(cfg->stream)) return __real_cudaLaunchKernelExC(cfg, fn, args);
    bind_hip();
    void* hip = hip_kernel(fn);
    if (hip == nullptr) return cudaErrorInvalidDeviceFunction;
    return Ce(hh_launch(hip, cfg->gridDim.x, cfg->gridDim.y, cfg->gridDim.z, cfg->blockDim.x, cfg->blockDim.y, cfg->blockDim.z,
                        args, cfg->dynamicSmemBytes, cfg->stream));
}
extern "C" cudaError_t __real_cudaLaunchHostFunc(cudaStream_t, cudaHostFn_t, void*);
extern "C" cudaError_t __wrap_cudaLaunchHostFunc(cudaStream_t s, cudaHostFn_t fn, void* data) {
    if (!on_hip() && !hip_stream(s)) return __real_cudaLaunchHostFunc(s, fn, data);
    bind_hip();
    return Ce(hh_launch_host(s, reinterpret_cast<void (*)(void*)>(fn), data));
}

extern "C" cudaError_t __real_cudaStreamCreate(cudaStream_t*);
extern "C" cudaError_t __wrap_cudaStreamCreate(cudaStream_t* s) {
    if (!on_hip()) return __real_cudaStreamCreate(s);
    bind_hip();
    void* stream = nullptr;
    int e = hh_stream_create(&stream, 0, 0, 0);
    if (e == 0 && s != nullptr) {
        *s = static_cast<cudaStream_t>(stream);
        track(g_hip_stream, *s);
    }
    return Ce(e);
}
extern "C" cudaError_t __real_cudaStreamCreateWithFlags(cudaStream_t*, unsigned int);
extern "C" cudaError_t __wrap_cudaStreamCreateWithFlags(cudaStream_t* s, unsigned int flags) {
    if (!on_hip()) return __real_cudaStreamCreateWithFlags(s, flags);
    bind_hip();
    void* stream = nullptr;
    int e = hh_stream_create(&stream, flags, 0, 0);
    if (e == 0 && s != nullptr) {
        *s = static_cast<cudaStream_t>(stream);
        track(g_hip_stream, *s);
    }
    return Ce(e);
}
extern "C" cudaError_t __real_cudaStreamCreateWithPriority(cudaStream_t*, unsigned int, int);
extern "C" cudaError_t __wrap_cudaStreamCreateWithPriority(cudaStream_t* s, unsigned int flags, int priority) {
    if (!on_hip()) return __real_cudaStreamCreateWithPriority(s, flags, priority);
    bind_hip();
    void* stream = nullptr;
    int e = hh_stream_create(&stream, flags, priority, 1);
    if (e == 0 && s != nullptr) {
        *s = static_cast<cudaStream_t>(stream);
        track(g_hip_stream, *s);
    }
    return Ce(e);
}
extern "C" cudaError_t __real_cudaStreamDestroy(cudaStream_t);
extern "C" cudaError_t __wrap_cudaStreamDestroy(cudaStream_t s) {
    if (!hip_stream(s)) return __real_cudaStreamDestroy(s);
    untrack(g_hip_stream, s);
    return Ce(hh_stream_destroy(s));
}
extern "C" cudaError_t __real_cudaStreamSynchronize(cudaStream_t);
extern "C" cudaError_t __wrap_cudaStreamSynchronize(cudaStream_t s) {
    if (!hip_stream(s) && !on_hip()) return __real_cudaStreamSynchronize(s);
    bind_hip();
    return Ce(hh_stream_sync(s));
}
extern "C" cudaError_t __real_cudaStreamQuery(cudaStream_t);
extern "C" cudaError_t __wrap_cudaStreamQuery(cudaStream_t s) {
    if (!hip_stream(s) && !on_hip()) return __real_cudaStreamQuery(s);
    return Ce(hh_stream_query(s));
}
extern "C" cudaError_t __real_cudaStreamBeginCapture(cudaStream_t, cudaStreamCaptureMode);
extern "C" cudaError_t __wrap_cudaStreamBeginCapture(cudaStream_t s, cudaStreamCaptureMode mode) {
    if (!hip_stream(s) && !on_hip()) return __real_cudaStreamBeginCapture(s, mode);
    bind_hip();
    return Ce(hh_stream_begin_capture(s, (int) mode));
}
extern "C" cudaError_t __real_cudaStreamEndCapture(cudaStream_t, cudaGraph_t*);
extern "C" cudaError_t __wrap_cudaStreamEndCapture(cudaStream_t s, cudaGraph_t* g) {
    if (!hip_stream(s) && !on_hip()) return __real_cudaStreamEndCapture(s, g);
    bind_hip();
    void* graph = nullptr;
    int e = hh_stream_end_capture(s, &graph);
    if (e == 0 && g != nullptr) {
        *g = static_cast<cudaGraph_t>(graph);
        track(g_hip_graph, *g);
    }
    return Ce(e);
}
extern "C" cudaError_t __real_cudaStreamIsCapturing(cudaStream_t, cudaStreamCaptureStatus*);
extern "C" cudaError_t __wrap_cudaStreamIsCapturing(cudaStream_t s, cudaStreamCaptureStatus* st) {
    if (!hip_stream(s) && !on_hip()) return __real_cudaStreamIsCapturing(s, st);
    return Ce(hh_stream_is_capturing(s, reinterpret_cast<int*>(st)));
}
extern "C" cudaError_t strata_real_streamGetCaptureInfo(cudaStream_t, cudaStreamCaptureStatus*, unsigned long long*,
                                                         cudaGraph_t*, const cudaGraphNode_t**, const cudaGraphEdgeData**, size_t*);
extern "C" cudaError_t strata_wrap_streamGetCaptureInfo(cudaStream_t s, cudaStreamCaptureStatus* st, unsigned long long* id,
                                                         cudaGraph_t* graph, const cudaGraphNode_t** deps,
                                                         const cudaGraphEdgeData** edges, size_t* ndeps) {
    if (!hip_stream(s) && !on_hip()) return strata_real_streamGetCaptureInfo(s, st, id, graph, deps, edges, ndeps);
    if (edges) *edges = nullptr;
    void* g = nullptr;
    const void** d = nullptr;
    int status = 0;
    int e = hh_stream_capture_info(s, &status, id, &g, &d, ndeps);
    if (st) *st = static_cast<cudaStreamCaptureStatus>(status);
    if (graph) *graph = static_cast<cudaGraph_t>(g);
    if (deps) *deps = (const cudaGraphNode_t*) d;
    return Ce(e);
}
extern "C" cudaError_t __real_cudaStreamWaitEvent(cudaStream_t, cudaEvent_t, unsigned int);
extern "C" cudaError_t __wrap_cudaStreamWaitEvent(cudaStream_t s, cudaEvent_t ev, unsigned int flags) {
    if (!hip_stream(s) && !has(g_hip_event, ev)) return __real_cudaStreamWaitEvent(s, ev, flags);
    return Ce(hh_stream_wait_event(s, ev, flags));
}

extern "C" cudaError_t __real_cudaEventCreate(cudaEvent_t*);
extern "C" cudaError_t __wrap_cudaEventCreate(cudaEvent_t* ev) {
    if (!on_hip()) return __real_cudaEventCreate(ev);
    bind_hip();
    void* e = nullptr;
    int err = hh_event_create(&e, 0, 0);
    if (err == 0 && ev) {
        *ev = static_cast<cudaEvent_t>(e);
        track(g_hip_event, *ev);
    }
    return Ce(err);
}
extern "C" cudaError_t __real_cudaEventCreateWithFlags(cudaEvent_t*, unsigned int);
extern "C" cudaError_t __wrap_cudaEventCreateWithFlags(cudaEvent_t* ev, unsigned int flags) {
    if (!on_hip()) return __real_cudaEventCreateWithFlags(ev, flags);
    bind_hip();
    void* e = nullptr;
    int err = hh_event_create(&e, flags, 1);
    if (err == 0 && ev) {
        *ev = static_cast<cudaEvent_t>(e);
        track(g_hip_event, *ev);
    }
    return Ce(err);
}
extern "C" cudaError_t __real_cudaEventDestroy(cudaEvent_t);
extern "C" cudaError_t __wrap_cudaEventDestroy(cudaEvent_t ev) {
    if (!has(g_hip_event, ev)) return __real_cudaEventDestroy(ev);
    untrack(g_hip_event, ev);
    return Ce(hh_event_destroy(ev));
}
extern "C" cudaError_t __real_cudaEventRecord(cudaEvent_t, cudaStream_t);
extern "C" cudaError_t __wrap_cudaEventRecord(cudaEvent_t ev, cudaStream_t s) {
    if (!has(g_hip_event, ev) && !hip_stream(s)) return __real_cudaEventRecord(ev, s);
    return Ce(hh_event_record(ev, s));
}
extern "C" cudaError_t __real_cudaEventSynchronize(cudaEvent_t);
extern "C" cudaError_t __wrap_cudaEventSynchronize(cudaEvent_t ev) {
    if (!has(g_hip_event, ev)) return __real_cudaEventSynchronize(ev);
    return Ce(hh_event_sync(ev));
}
extern "C" cudaError_t __real_cudaEventQuery(cudaEvent_t);
extern "C" cudaError_t __wrap_cudaEventQuery(cudaEvent_t ev) {
    if (!has(g_hip_event, ev)) return __real_cudaEventQuery(ev);
    return Ce(hh_event_query(ev));
}
extern "C" cudaError_t __real_cudaEventElapsedTime(float*, cudaEvent_t, cudaEvent_t);
extern "C" cudaError_t __wrap_cudaEventElapsedTime(float* ms, cudaEvent_t a, cudaEvent_t b) {
    if (!has(g_hip_event, a) && !has(g_hip_event, b)) return __real_cudaEventElapsedTime(ms, a, b);
    return Ce(hh_event_elapsed(ms, a, b));
}

extern "C" cudaError_t __real_cudaGraphDestroy(cudaGraph_t);
extern "C" cudaError_t __wrap_cudaGraphDestroy(cudaGraph_t g) {
    if (!has(g_hip_graph, g)) return __real_cudaGraphDestroy(g);
    untrack(g_hip_graph, g);
    return Ce(hh_graph_destroy(g));
}
extern "C" cudaError_t __real_cudaGraphExecDestroy(cudaGraphExec_t);
extern "C" cudaError_t __wrap_cudaGraphExecDestroy(cudaGraphExec_t g) {
    if (!has(g_hip_exec, g)) return __real_cudaGraphExecDestroy(g);
    untrack(g_hip_exec, g);
    return Ce(hh_graph_exec_destroy(g));
}
extern "C" cudaError_t __real_cudaGraphInstantiate(cudaGraphExec_t*, cudaGraph_t, unsigned long long);
extern "C" cudaError_t __wrap_cudaGraphInstantiate(cudaGraphExec_t* exec, cudaGraph_t g, unsigned long long flags) {
    if (!has(g_hip_graph, g)) return __real_cudaGraphInstantiate(exec, g, flags);
    bind_hip();
    void* out = nullptr;
    int e = hh_graph_instantiate(&out, g, flags);
    if (e == 0 && exec) {
        *exec = static_cast<cudaGraphExec_t>(out);
        track(g_hip_exec, *exec);
    }
    return Ce(e);
}
extern "C" cudaError_t __real_cudaGraphLaunch(cudaGraphExec_t, cudaStream_t);
extern "C" cudaError_t __wrap_cudaGraphLaunch(cudaGraphExec_t exec, cudaStream_t s) {
    if (!has(g_hip_exec, exec)) return __real_cudaGraphLaunch(exec, s);
    bind_hip();
    return Ce(hh_graph_launch(exec, s));
}
extern "C" cudaError_t __real_cudaGraphUpload(cudaGraphExec_t, cudaStream_t);
extern "C" cudaError_t __wrap_cudaGraphUpload(cudaGraphExec_t exec, cudaStream_t s) {
    if (!has(g_hip_exec, exec)) return __real_cudaGraphUpload(exec, s);
    return Ce(hh_graph_upload(exec, s));
}
extern "C" cudaError_t __real_cudaGraphGetNodes(cudaGraph_t, cudaGraphNode_t*, size_t*);
extern "C" cudaError_t __wrap_cudaGraphGetNodes(cudaGraph_t g, cudaGraphNode_t* nodes, size_t* n) {
    if (!has(g_hip_graph, g)) return __real_cudaGraphGetNodes(g, nodes, n);
    return Ce(hh_graph_nodes(g, reinterpret_cast<void**>(nodes), n));
}
extern "C" cudaError_t strata_real_graphGetEdges(cudaGraph_t, cudaGraphNode_t*, cudaGraphNode_t*, cudaGraphEdgeData*, size_t*);
extern "C" cudaError_t strata_wrap_graphGetEdges(cudaGraph_t g, cudaGraphNode_t* from, cudaGraphNode_t* to,
                                                  cudaGraphEdgeData* edges, size_t* n) {
    if (!has(g_hip_graph, g)) return strata_real_graphGetEdges(g, from, to, edges, n);
    int e = hh_graph_edges(g, reinterpret_cast<void**>(from), reinterpret_cast<void**>(to), n);
    if (e == 0 && edges && n) std::memset(edges, 0, sizeof(cudaGraphEdgeData) * (*n));
    return Ce(e);
}
extern "C" cudaError_t __real_cudaGraphNodeGetType(cudaGraphNode_t, cudaGraphNodeType*);
extern "C" cudaError_t __wrap_cudaGraphNodeGetType(cudaGraphNode_t node, cudaGraphNodeType* ty) {
    if (on_hip()) {
        int kind = 0;
        if (hh_graph_node_type(node, &kind) == 0) {
            if (ty) *ty = static_cast<cudaGraphNodeType>(kind);
            return cudaSuccess;
        }
    }
    return __real_cudaGraphNodeGetType(node, ty);
}
extern "C" cudaError_t __real_cudaGraphKernelNodeGetParams(cudaGraphNode_t, cudaKernelNodeParams*);
extern "C" cudaError_t __wrap_cudaGraphKernelNodeGetParams(cudaGraphNode_t node, cudaKernelNodeParams* p) {
    if (on_hip() && p) {
        void* fn = nullptr;
        unsigned gx = 1, gy = 1, gz = 1, bx = 1, by = 1, bz = 1, shared = 0;
        void** kp = nullptr;
        void** extra = nullptr;
        if (hh_graph_kernel_params(node, &fn, &gx, &gy, &gz, &bx, &by, &bz, &shared, &kp, &extra) == 0) {
            std::memset(p, 0, sizeof(*p));
            p->func = fn;
            p->gridDim = dim3(gx, gy, gz);
            p->blockDim = dim3(bx, by, bz);
            p->sharedMemBytes = shared;
            p->kernelParams = kp;
            p->extra = extra;
            return cudaSuccess;
        }
    }
    return __real_cudaGraphKernelNodeGetParams(node, p);
}

extern "C" cudaError_t __real_cudaFuncGetAttributes(cudaFuncAttributes*, const void*);
extern "C" cudaError_t __wrap_cudaFuncGetAttributes(cudaFuncAttributes* attr, const void* fn) {
    void* hip = on_hip() ? hip_kernel(fn) : nullptr;
    if (hip == nullptr) return __real_cudaFuncGetAttributes(attr, fn);
    if (attr == nullptr) return cudaErrorInvalidValue;
    std::memset(attr, 0, sizeof(*attr));
    return Ce(hh_func_attrs(hip, &attr->sharedSizeBytes, &attr->constSizeBytes, &attr->localSizeBytes, &attr->maxThreadsPerBlock,
                            &attr->numRegs, &attr->ptxVersion, &attr->binaryVersion, &attr->cacheModeCA,
                            &attr->maxDynamicSharedSizeBytes, &attr->preferredShmemCarveout));
}
extern "C" cudaError_t __real_cudaFuncSetAttribute(const void*, cudaFuncAttribute, int);
extern "C" cudaError_t __wrap_cudaFuncSetAttribute(const void* fn, cudaFuncAttribute attr, int value) {
    void* hip = on_hip() ? hip_kernel(fn) : nullptr;
    if (hip == nullptr) return __real_cudaFuncSetAttribute(fn, attr, value);
    return Ce(hh_func_set_attr(hip, (int) attr, value));
}
extern "C" cudaError_t __real_cudaFuncGetName(const char**, const void*);
extern "C" cudaError_t __wrap_cudaFuncGetName(const char** name, const void* fn) {
    if (!on_hip()) return __real_cudaFuncGetName(name, fn);
    Dl_info info{};
    if (name == nullptr || dladdr(fn, &info) == 0 || info.dli_sname == nullptr) return cudaErrorInvalidDeviceFunction;
    *name = info.dli_sname;
    return cudaSuccess;
}
extern "C" cudaError_t __real_cudaGetSymbolAddress(void**, const void*);
extern "C" cudaError_t __wrap_cudaGetSymbolAddress(void** dev, const void* symbol) {
    if (!on_hip()) return __real_cudaGetSymbolAddress(dev, symbol);
    bind_hip();
    void* hip = hip_kernel(symbol);
    return Ce(hh_symbol_address(dev, hip != nullptr ? hip : symbol));
}
extern "C" cudaError_t __real_cudaDeviceGetAttribute(int*, cudaDeviceAttr, int);
extern "C" cudaError_t __wrap_cudaDeviceGetAttribute(int* v, cudaDeviceAttr attr, int d) {
    if (!hip_ordinal(d)) return __real_cudaDeviceGetAttribute(v, attr, d);
    return Ce(hh_device_attr(v, (int) attr, d - g_cuda_count));
}
extern "C" cudaError_t __real_cudaDeviceGetStreamPriorityRange(int*, int*);
extern "C" cudaError_t __wrap_cudaDeviceGetStreamPriorityRange(int* least, int* greatest) {
    if (!on_hip()) return __real_cudaDeviceGetStreamPriorityRange(least, greatest);
    bind_hip();
    return Ce(hh_stream_priority_range(least, greatest));
}
extern "C" cudaError_t __real_cudaDeviceCanAccessPeer(int*, int, int);
extern "C" cudaError_t __wrap_cudaDeviceCanAccessPeer(int* can, int dev, int peer) {
    if (can == nullptr) return cudaErrorInvalidValue;
    if (hip_ordinal(dev) || hip_ordinal(peer)) {
        *can = 0;
        return cudaSuccess;
    }
    return __real_cudaDeviceCanAccessPeer(can, dev, peer);
}
extern "C" cudaError_t __real_cudaDeviceEnablePeerAccess(int, unsigned int);
extern "C" cudaError_t __wrap_cudaDeviceEnablePeerAccess(int peer, unsigned int flags) {
    if (hip_ordinal(peer) || on_hip()) return cudaSuccess;
    return __real_cudaDeviceEnablePeerAccess(peer, flags);
}
extern "C" cudaError_t __real_cudaDeviceDisablePeerAccess(int);
extern "C" cudaError_t __wrap_cudaDeviceDisablePeerAccess(int peer) {
    if (hip_ordinal(peer) || on_hip()) return cudaSuccess;
    return __real_cudaDeviceDisablePeerAccess(peer);
}
extern "C" cudaError_t __real_cudaDriverGetVersion(int*);
extern "C" cudaError_t __wrap_cudaDriverGetVersion(int* v) {
    if (!on_hip()) return __real_cudaDriverGetVersion(v);
    return Ce(hh_driver_version(v));
}
extern "C" cudaError_t __real_cudaRuntimeGetVersion(int*);
extern "C" cudaError_t __wrap_cudaRuntimeGetVersion(int* v) {
    if (!on_hip()) return __real_cudaRuntimeGetVersion(v);
    return Ce(hh_runtime_version(v));
}
extern "C" cudaError_t __real_cudaInitDevice(int, unsigned int, unsigned int);
extern "C" cudaError_t __wrap_cudaInitDevice(int d, unsigned int device_flags, unsigned int mem) {
    if (!hip_ordinal(d)) return __real_cudaInitDevice(d, device_flags, mem);
    (void) mem;
    // Must not leave the helper current: DeviceScope reads the current device after this returns.
    const int saved = g_device;
    const cudaError_t e = __wrap_cudaSetDevice(d);
    if (e == cudaSuccess) (void) hh_set_device_flags(device_flags);
    const cudaError_t back = __wrap_cudaSetDevice(saved);
    return e != cudaSuccess ? e : back;
}
extern "C" cudaError_t __real_cudaGetDriverEntryPointByVersion(const char*, void**, unsigned int, unsigned long long,
                                                              cudaDriverEntryPointQueryResult*);
extern "C" cudaError_t __wrap_cudaGetDriverEntryPointByVersion(const char* symbol, void** fn, unsigned int version,
                                                              unsigned long long flags, cudaDriverEntryPointQueryResult* status) {
    if (!on_hip()) return __real_cudaGetDriverEntryPointByVersion(symbol, fn, version, flags, status);
    if (status) *status = cudaDriverEntryPointSymbolNotFound;
    if (fn) *fn = nullptr;
    return cudaErrorNotSupported;
}
extern "C" cudaError_t __real_cudaOccupancyMaxActiveBlocksPerMultiprocessorWithFlags(int*, const void*, int, size_t, unsigned int);
extern "C" cudaError_t __wrap_cudaOccupancyMaxActiveBlocksPerMultiprocessorWithFlags(int* blocks, const void* fn, int block,
                                                                                    size_t smem, unsigned int flags) {
    void* hip = on_hip() ? hip_kernel(fn) : nullptr;
    if (hip == nullptr) return __real_cudaOccupancyMaxActiveBlocksPerMultiprocessorWithFlags(blocks, fn, block, smem, flags);
    bind_hip();
    return Ce(hh_occupancy(blocks, hip, block, smem, flags));
}
extern "C" cudaError_t __real_cudaOccupancyMaxActiveClusters(int*, const void*, const cudaLaunchConfig_t*);
extern "C" cudaError_t __wrap_cudaOccupancyMaxActiveClusters(int* clusters, const void* fn, const cudaLaunchConfig_t* cfg) {
    if (!on_hip()) return __real_cudaOccupancyMaxActiveClusters(clusters, fn, cfg);
    if (clusters) *clusters = 0;
    return cudaErrorNotSupported;
}

#ifndef STRATA_HYBRID_NO_BLAS
namespace {
bool blas_hip(cublasHandle_t h) { return has(g_hip_blas, h) || on_hip(); }
cublasStatus_t Bs(int s) { return static_cast<cublasStatus_t>(s); }
}  // namespace
extern "C" cublasStatus_t __real_cublasCreate_v2(cublasHandle_t*);
extern "C" cublasStatus_t __wrap_cublasCreate_v2(cublasHandle_t* h) {
    if (!on_hip()) return __real_cublasCreate_v2(h);
    bind_hip();
    void* handle = nullptr;
    int s = hh_blas_create(&handle);
    if (s == 0 && h) {
        *h = static_cast<cublasHandle_t>(handle);
        track(g_hip_blas, *h);
    }
    return Bs(s);
}
extern "C" cublasStatus_t __real_cublasDestroy_v2(cublasHandle_t);
extern "C" cublasStatus_t __wrap_cublasDestroy_v2(cublasHandle_t h) {
    if (!has(g_hip_blas, h)) return __real_cublasDestroy_v2(h);
    untrack(g_hip_blas, h);
    return Bs(hh_blas_destroy(h));
}
extern "C" cublasStatus_t __real_cublasSetStream_v2(cublasHandle_t, cudaStream_t);
extern "C" cublasStatus_t __wrap_cublasSetStream_v2(cublasHandle_t h, cudaStream_t s) {
    if (!blas_hip(h)) return __real_cublasSetStream_v2(h, s);
    return Bs(hh_blas_set_stream(h, s));
}
extern "C" cublasStatus_t __real_cublasSetMathMode(cublasHandle_t, cublasMath_t);
extern "C" cublasStatus_t __wrap_cublasSetMathMode(cublasHandle_t h, cublasMath_t mode) {
    if (!blas_hip(h)) return __real_cublasSetMathMode(h, mode);
    int s = hh_blas_set_math(h, (int) mode);
    return s == 0 ? CUBLAS_STATUS_SUCCESS : Bs(s);
}
extern "C" cublasStatus_t __real_cublasSetWorkspace_v2(cublasHandle_t, void*, size_t);
extern "C" cublasStatus_t __wrap_cublasSetWorkspace_v2(cublasHandle_t h, void* workspace, size_t bytes) {
    if (!blas_hip(h)) return __real_cublasSetWorkspace_v2(h, workspace, bytes);
    return Bs(hh_blas_set_workspace(h, workspace, bytes));
}
extern "C" cublasStatus_t __real_cublasSgemm_v2(cublasHandle_t, cublasOperation_t, cublasOperation_t, int, int, int, const float*,
                                               const float*, int, const float*, int, const float*, float*, int);
extern "C" cublasStatus_t __wrap_cublasSgemm_v2(cublasHandle_t h, cublasOperation_t ta, cublasOperation_t tb, int m, int n, int k,
                                               const float* alpha, const float* A, int lda, const float* B, int ldb,
                                               const float* beta, float* C, int ldc) {
    if (!blas_hip(h)) return __real_cublasSgemm_v2(h, ta, tb, m, n, k, alpha, A, lda, B, ldb, beta, C, ldc);
    return Bs(hh_blas_sgemm(h, (int) ta, (int) tb, m, n, k, alpha, A, lda, B, ldb, beta, C, ldc));
}
extern "C" cublasStatus_t __real_cublasGemmEx(cublasHandle_t, cublasOperation_t, cublasOperation_t, int, int, int, const void*,
                                             const void*, cudaDataType, int, const void*, cudaDataType, int, const void*, void*,
                                             cudaDataType, int, cublasComputeType_t, cublasGemmAlgo_t);
extern "C" cublasStatus_t __wrap_cublasGemmEx(cublasHandle_t h, cublasOperation_t ta, cublasOperation_t tb, int m, int n, int k,
                                             const void* alpha, const void* A, cudaDataType At, int lda, const void* B,
                                             cudaDataType Bt, int ldb, const void* beta, void* C, cudaDataType Ct, int ldc,
                                             cublasComputeType_t compute, cublasGemmAlgo_t algo) {
    if (!blas_hip(h))
        return __real_cublasGemmEx(h, ta, tb, m, n, k, alpha, A, At, lda, B, Bt, ldb, beta, C, Ct, ldc, compute, algo);
    return Bs(hh_blas_gemm_ex(h, (int) ta, (int) tb, m, n, k, alpha, A, (int) At, lda, B, (int) Bt, ldb, beta, C, (int) Ct, ldc,
                              (int) compute, (int) algo));
}
extern "C" cublasStatus_t __real_cublasGemmStridedBatchedEx(cublasHandle_t, cublasOperation_t, cublasOperation_t, int, int, int,
                                                           const void*, const void*, cudaDataType, int, long long int, const void*,
                                                           cudaDataType, int, long long int, const void*, void*, cudaDataType, int,
                                                           long long int, int, cublasComputeType_t, cublasGemmAlgo_t);
extern "C" cublasStatus_t __wrap_cublasGemmStridedBatchedEx(
    cublasHandle_t h, cublasOperation_t ta, cublasOperation_t tb, int m, int n, int k, const void* alpha, const void* A,
    cudaDataType At, int lda, long long strideA, const void* B, cudaDataType Bt, int ldb, long long strideB, const void* beta,
    void* C, cudaDataType Ct, int ldc, long long strideC, int batch, cublasComputeType_t compute, cublasGemmAlgo_t algo) {
    if (!blas_hip(h))
        return __real_cublasGemmStridedBatchedEx(h, ta, tb, m, n, k, alpha, A, At, lda, strideA, B, Bt, ldb, strideB, beta, C, Ct,
                                                 ldc, strideC, batch, compute, algo);
    return Bs(hh_blas_gemm_strided(h, (int) ta, (int) tb, m, n, k, alpha, A, (int) At, lda, strideA, B, (int) Bt, ldb, strideB, beta,
                                   C, (int) Ct, ldc, strideC, batch, (int) compute, (int) algo));
}
#endif
