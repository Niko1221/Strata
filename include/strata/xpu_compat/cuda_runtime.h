#pragma once
// Intel XPU compatibility header. Included in place of NVIDIA's <cuda_runtime.h>
// when STRATA_ENABLE_XPU=ON. Kernel source is rewritten (tools/xpu/rewrite_cuda.py)
// so `foo<<<grid, block, smem, stream>>>(args)` becomes strata::xpu::launch(...).
//
// Graphs: capture records the launches; instantiate additionally replays them into a native
// ext_oneapi_graph so launch is ONE submission (the eager per-op loop remains as the fallback,
// STRATA_XPU_EAGER_GRAPHS=1). The GPU runs the same kernels either way.
#ifndef __CUDA_RUNTIME_H__
#define __CUDA_RUNTIME_H__

#include <sycl/sycl.hpp>
#include <sycl/ext/oneapi/kernel_properties/properties.hpp>
#include <sycl/ext/oneapi/experimental/graph.hpp>

#include <algorithm>
#include <atomic>
#include <cstdint>
#include <cstring>
#include <functional>
#include <map>
#include <memory>
#include <mutex>
#include <optional>
#include <stdexcept>
#include <string>
#include <unordered_map>
#include <unordered_set>
#include <utility>
#include <vector>

// Portable (non-PTX) kernel paths already written for the HIP backend.
#ifndef __HIPCC__
#define __HIPCC__ 1
#endif
#define STRATA_USE_XPU 1
#define CUDART_VERSION 12080
// Portable kernel paths are gated on __HIPCC__. A few of those paths also call HIP runtime
// helpers; map the ones that show up to the CUDA-shaped names this header implements.
#define hipSuccess cudaSuccess
#define hipError_t cudaError_t
#define hipGetLastError cudaGetLastError
#define hipGetDevice cudaGetDevice
#define hipGetErrorString cudaGetErrorString
#define hipGetDeviceProperties cudaGetDeviceProperties

#define __global__
#define __device__
#define __host__
#define __forceinline__ inline __attribute__((always_inline))
#define __noinline__ __attribute__((noinline))
#define __restrict__ __restrict
#define __launch_bounds__(...)
#define __align__(n) alignas(n)
#define __grid_constant__

using cudaError_t = int;
constexpr cudaError_t cudaSuccess = 0;
constexpr cudaError_t cudaErrorInvalidValue = 1;
constexpr cudaError_t cudaErrorMemoryAllocation = 2;
constexpr cudaError_t cudaErrorNotReady = 3;
constexpr cudaError_t cudaErrorUnknown = 4;
constexpr cudaError_t cudaErrorPeerAccessAlreadyEnabled = 5;
constexpr cudaError_t cudaErrorStreamCaptureUnsupported = 6;

enum cudaMemcpyKind {
    cudaMemcpyHostToHost = 0,
    cudaMemcpyHostToDevice = 1,
    cudaMemcpyDeviceToHost = 2,
    cudaMemcpyDeviceToDevice = 3,
    cudaMemcpyDefault = 4
};

enum cudaStreamCaptureMode { cudaStreamCaptureModeGlobal = 0, cudaStreamCaptureModeThreadLocal = 1, cudaStreamCaptureModeRelaxed = 2 };
enum cudaStreamCaptureStatus { cudaStreamCaptureStatusNone = 0, cudaStreamCaptureStatusActive = 1, cudaStreamCaptureStatusInvalidated = 2 };
constexpr unsigned cudaStreamNonBlocking = 1;
constexpr unsigned cudaEventDisableTiming = 2;
constexpr unsigned cudaHostAllocDefault = 0;
constexpr unsigned cudaHostAllocPortable = 1;
constexpr unsigned cudaHostAllocMapped = 2;
constexpr unsigned cudaHostAllocWriteCombined = 4;
constexpr unsigned cudaHostRegisterDefault = 0;
constexpr unsigned cudaHostRegisterPortable = 1;
constexpr unsigned cudaHostRegisterMapped = 2;
constexpr int cudaDevAttrWarpSize = 10;
constexpr int cudaDevAttrMaxSharedMemoryPerBlock = 8;
constexpr int cudaDevAttrMaxSharedMemoryPerBlockOptin = 97;
constexpr int cudaDevAttrMaxSharedMemoryPerMultiprocessor = 81;
constexpr int cudaDevAttrMultiProcessorCount = 16;
constexpr int cudaDevAttrClockRate = 13;
constexpr int cudaDevAttrComputeCapabilityMajor = 75;
constexpr int cudaDevAttrComputeCapabilityMinor = 76;
constexpr int cudaDevAttrCooperativeLaunch = 95;
constexpr int cudaDevAttrIntegrated = 18;
constexpr int cudaDevAttrReservedSharedMemoryPerBlock = 111;
constexpr int cudaFuncAttributeMaxDynamicSharedMemorySize = 8;
constexpr int cudaFuncAttributePreferredSharedMemoryCarveout = 9;
constexpr int cudaSharedmemCarveoutMaxShared = 100;
constexpr int cudaDeviceMapHost = 8;
constexpr int cudaDeviceScheduleSpin = 4;

struct dim3 {
    unsigned x, y, z;
    dim3(unsigned x_ = 1, unsigned y_ = 1, unsigned z_ = 1) : x(x_), y(y_), z(z_) {}
};
inline dim3 operator*(dim3 a, dim3 b) { return dim3(a.x * b.x, a.y * b.y, a.z * b.z); }

struct float2 { float x, y; };
struct float4 { float x, y, z, w; };
struct int2 { int x, y; };
struct int4 { int x, y, z, w; };
struct uint2 { unsigned x, y; };
struct uint4 { unsigned x, y, z, w; };
struct char2 { signed char x, y; };
struct char4 { signed char x, y, z, w; };
struct uchar2 { unsigned char x, y; };
struct uchar4 { unsigned char x, y, z, w; };
struct short2 { short x, y; };
struct short4 { short x, y, z, w; };
struct ushort2 { unsigned short x, y; };
struct ushort4 { unsigned short x, y, z, w; };
using ushort = unsigned short;
inline float2 make_float2(float x, float y) { return {x, y}; }
inline float4 make_float4(float x, float y, float z, float w) { return {x, y, z, w}; }
inline int2 make_int2(int x, int y) { return {x, y}; }
inline int4 make_int4(int x, int y, int z, int w) { return {x, y, z, w}; }
inline uint2 make_uint2(unsigned x, unsigned y) { return {x, y}; }
inline uint4 make_uint4(unsigned x, unsigned y, unsigned z, unsigned w) { return {x, y, z, w}; }

struct cudaDeviceProp {
    char name[256] = {};
    char gcnArchName[256] = {};
    int major = 8;
    int minor = 0;
    int warpSize = 32;
    int multiProcessorCount = 1;
    int maxThreadsPerBlock = 1024;
    int maxThreadsPerMultiProcessor = 2048;
    size_t totalGlobalMem = 0;
    size_t sharedMemPerBlock = 64 * 1024;
    size_t sharedMemPerBlockOptin = 128 * 1024;
    size_t sharedMemPerMultiprocessor = 128 * 1024;
    int clockRate = 1000000;
    int integrated = 0;
    int canMapHostMemory = 1;
    int cooperativeLaunch = 0;
    unsigned char luid[8] = {};
};

using hipDeviceProp_t = cudaDeviceProp;

struct cudaFuncAttributes {
    int maxThreadsPerBlock = 1024;
    size_t sharedSizeBytes = 0;
    int numRegs = 32;
    int ptxVersion = 80;
    int binaryVersion = 80;
};

enum cudaGraphNodeType {
    cudaGraphNodeTypeKernel = 0,
    cudaGraphNodeTypeMemcpy = 1,
    cudaGraphNodeTypeMemset = 2,
    cudaGraphNodeTypeHost = 3,
    cudaGraphNodeTypeGraph = 4,
    cudaGraphNodeTypeEmpty = 5,
    cudaGraphNodeTypeWaitEvent = 6,
    cudaGraphNodeTypeEventRecord = 7
};

namespace strata::xpu {

inline sycl::queue& default_queue();

inline bool trace_enabled() {
    static const bool on = std::getenv("STRATA_XPU_TRACE") != nullptr;
    return on;
}

struct Graph {
    std::vector<std::function<void(sycl::queue&)>> ops;
    std::vector<std::string> names;
    // Native replay: the op list recorded into an ext_oneapi_graph at instantiate time.
    // Replaying it is ~1.2 us/op amortised vs ~31 us/op for the eager loop (measured on b60-dgpu:
    // a 2000-kernel graph replays in 2.3 ms vs 62 ms). Empty when recording fails or is disabled
    // (STRATA_XPU_EAGER_GRAPHS=1); launch then falls back to the eager loop.
    std::optional<sycl::ext::oneapi::experimental::command_graph<
        sycl::ext::oneapi::experimental::graph_state::executable>>
        native;
    sycl::queue* record_q = nullptr;   // the capture stream's queue: recording stays off shared queues
    void push(std::function<void(sycl::queue&)>&& op, const char* name) {
        ops.push_back(std::move(op));
        names.push_back(name ? name : "?");
    }
};

struct Stream {
    sycl::queue q;
    Graph* capture = nullptr;
    explicit Stream(sycl::queue queue) : q(std::move(queue)) {}
};

struct Event {
    sycl::event ev;
    bool recorded = false;
};

inline thread_local cudaError_t last_error = cudaSuccess;
inline thread_local std::string last_error_text;
inline void set_error(cudaError_t e) { if (e != cudaSuccess) last_error = e; }

inline sycl::device gpu_device(int ordinal = 0) {
    auto devs = sycl::device::get_devices(sycl::info::device_type::gpu);
    if (devs.empty()) throw std::runtime_error("no Intel GPU visible to SYCL");
    if (ordinal < 0 || ordinal >= static_cast<int>(devs.size())) ordinal = 0;
    return devs[static_cast<size_t>(ordinal)];
}

inline sycl::queue make_queue(int ordinal = 0) {
    return sycl::queue(gpu_device(ordinal), sycl::property_list{
        sycl::property::queue::in_order{},
        sycl::property::queue::enable_profiling{}});
}

inline sycl::queue& default_queue() {
    static sycl::queue q = make_queue(0);
    return q;
}

inline Stream* default_stream() {
    static Stream s(make_queue(0));
    return &s;
}

inline Stream* as_stream(void* stream) {
    return stream ? static_cast<Stream*>(stream) : default_stream();
}

inline sycl::queue& queue_of(void* stream) { return as_stream(stream)->q; }

struct Dim {
    unsigned x, y, z;
};
inline Dim to_dim(dim3 d) { return {d.x, d.y, d.z}; }
inline Dim to_dim(unsigned v) { return {v, 1, 1}; }
inline Dim to_dim(int v) { return {static_cast<unsigned>(v), 1, 1}; }
inline Dim to_dim(unsigned long v) { return {static_cast<unsigned>(v), 1, 1}; }
inline Dim to_dim(unsigned long long v) { return {static_cast<unsigned>(v), 1, 1}; }
inline Dim to_dim(long v) { return {static_cast<unsigned>(v), 1, 1}; }
inline Dim to_dim(long long v) { return {static_cast<unsigned>(v), 1, 1}; }

inline sycl::nd_item<3> item() { return sycl::ext::oneapi::this_work_item::get_nd_item<3>(); }

struct Idx { unsigned x, y, z; };
inline Idx thread_idx() {
    auto it = item();
    return {static_cast<unsigned>(it.get_local_id(2)), static_cast<unsigned>(it.get_local_id(1)),
            static_cast<unsigned>(it.get_local_id(0))};
}
inline Idx block_idx() {
    auto it = item();
    return {static_cast<unsigned>(it.get_group(2)), static_cast<unsigned>(it.get_group(1)),
            static_cast<unsigned>(it.get_group(0))};
}
inline Idx block_dim() {
    auto it = item();
    return {static_cast<unsigned>(it.get_local_range(2)), static_cast<unsigned>(it.get_local_range(1)),
            static_cast<unsigned>(it.get_local_range(0))};
}
inline Idx grid_dim() {
    auto it = item();
    return {static_cast<unsigned>(it.get_group_range(2)), static_cast<unsigned>(it.get_group_range(1)),
            static_cast<unsigned>(it.get_group_range(0))};
}

inline constexpr size_t kDynamicSmem = 96 * 1024;

template <typename T>
inline T* dynamic_smem() {
    auto& buf = *sycl::ext::oneapi::group_local_memory_for_overwrite<unsigned char[kDynamicSmem]>(
        sycl::ext::oneapi::this_work_item::get_work_group<3>());
    return reinterpret_cast<T*>(buf);
}

inline void* shared_symbol(size_t bytes) {
    return sycl::malloc_shared(bytes, default_queue());
}

template <class Body>
void submit(sycl::queue& q, Dim grid, Dim block, Body body) {
    if (grid.x == 0 || grid.y == 0 || grid.z == 0 || block.x == 0 || block.y == 0 || block.z == 0) return;
    sycl::range<3> local(block.z, block.y, block.x);
    sycl::range<3> global(grid.z * block.z, grid.y * block.y, grid.x * block.x);
    const size_t threads = static_cast<size_t>(block.x) * block.y * block.z;
    if (threads % 32 == 0) {
        q.parallel_for(sycl::nd_range<3>(global, local),
                       sycl::ext::oneapi::experimental::properties{
                           sycl::ext::oneapi::experimental::sub_group_size<32>},
                       [=](sycl::nd_item<3>) { body(); });
    } else {
        q.parallel_for(sycl::nd_range<3>(global, local), [=](sycl::nd_item<3>) { body(); });
    }
}

template <class Grid, class Block, class Body>
cudaError_t launch(Grid grid_v, Block block_v, size_t /*smem*/, void* stream, const char* name, Body body) {
    try {
        Stream* s = as_stream(stream);
        Dim grid = to_dim(grid_v);
        Dim block = to_dim(block_v);
        auto op = [=](sycl::queue& q) {
            if (trace_enabled()) {
                std::fprintf(stderr, "xpu trace: launch %s grid %dx%dx%d block %dx%dx%d\n", name,
                             grid.x, grid.y, grid.z, block.x, block.y, block.z);
                std::fflush(stderr);
            }
            submit(q, grid, block, body);
        };
        if (s->capture) {
            s->capture->push(std::move(op), name);
            return cudaSuccess;
        }
        op(s->q);
        return cudaSuccess;
    } catch (const sycl::exception& e) {
        (void) e;
        return cudaErrorInvalidValue;
    }
}

}  // namespace strata::xpu



using cudaStream_t = strata::xpu::Stream*;
using cudaEvent_t = strata::xpu::Event*;
using cudaGraph_t = strata::xpu::Graph*;
using cudaGraphExec_t = strata::xpu::Graph*;
using cudaGraphNode_t = void*;

struct cudaKernelNodeParams {
    void* func = nullptr;
    dim3 gridDim{};
    dim3 blockDim{};
    unsigned int sharedMemBytes = 0;
    void** kernelParams = nullptr;
    void** extra = nullptr;
};

inline const char* cudaGetErrorString(cudaError_t e) {
    if (e != cudaSuccess && !strata::xpu::last_error_text.empty()) return strata::xpu::last_error_text.c_str();
    switch (e) {
        case cudaSuccess: return "no error";
        case cudaErrorInvalidValue: return "invalid argument";
        case cudaErrorMemoryAllocation: return "out of memory";
        case cudaErrorNotReady: return "not ready";
        default: return "unknown error";
    }
}
inline cudaError_t cudaGetLastError() {
    cudaError_t e = strata::xpu::last_error;
    strata::xpu::last_error = cudaSuccess;
    return e;
}
inline cudaError_t cudaPeekAtLastError() { return strata::xpu::last_error; }

inline cudaError_t cudaGetDeviceCount(int* count) {
    try {
        *count = static_cast<int>(sycl::device::get_devices(sycl::info::device_type::gpu).size());
        return cudaSuccess;
    } catch (...) {
        return cudaErrorUnknown;
    }
}
inline cudaError_t cudaSetDevice(int ordinal) {
    (void) ordinal;
    return cudaSuccess;
}
inline cudaError_t cudaInitDevice(int device, unsigned flags, unsigned) {
    (void) flags;
    return cudaSetDevice(device);
}
inline cudaError_t cudaGetDevice(int* ordinal) {
    *ordinal = 0;
    return cudaSuccess;
}
inline cudaError_t cudaGetDeviceProperties(cudaDeviceProp* p, int ordinal) {
    try {
        auto d = strata::xpu::gpu_device(ordinal);
        std::memset(p, 0, sizeof(*p));
        auto name = d.get_info<sycl::info::device::name>();
        std::snprintf(p->name, sizeof(p->name), "%s", name.c_str());
        std::snprintf(p->gcnArchName, sizeof(p->gcnArchName), "intel_gpu_bmg");
        p->major = 8;
        p->minor = 0;
        p->warpSize = 32;
        p->multiProcessorCount = static_cast<int>(d.get_info<sycl::info::device::max_compute_units>());
        p->maxThreadsPerBlock = static_cast<int>(d.get_info<sycl::info::device::max_work_group_size>());
        p->totalGlobalMem = d.get_info<sycl::info::device::global_mem_size>();
        p->sharedMemPerBlock = d.get_info<sycl::info::device::local_mem_size>();
        p->sharedMemPerBlockOptin = p->sharedMemPerBlock;
        p->sharedMemPerMultiprocessor = p->sharedMemPerBlock;
        p->canMapHostMemory = 1;
        return cudaSuccess;
    } catch (...) {
        return cudaErrorInvalidValue;
    }
}
inline cudaError_t cudaDeviceGetAttribute(int* value, int attr, int ordinal) {
    cudaDeviceProp p{};
    if (cudaGetDeviceProperties(&p, ordinal) != cudaSuccess) return cudaErrorInvalidValue;
    switch (attr) {
        case cudaDevAttrWarpSize: *value = p.warpSize; break;
        case cudaDevAttrMultiProcessorCount: *value = p.multiProcessorCount; break;
        case cudaDevAttrMaxSharedMemoryPerBlock:
        case cudaDevAttrMaxSharedMemoryPerBlockOptin:
        case cudaDevAttrMaxSharedMemoryPerMultiprocessor:
            *value = static_cast<int>(p.sharedMemPerBlock); break;
        case cudaDevAttrClockRate: *value = 1500000; break;
        case cudaDevAttrComputeCapabilityMajor: *value = p.major; break;
        case cudaDevAttrComputeCapabilityMinor: *value = p.minor; break;
        case cudaDevAttrCooperativeLaunch: *value = 0; break;
        case cudaDevAttrIntegrated: *value = 0; break;
        case cudaDevAttrReservedSharedMemoryPerBlock: *value = 0; break;
        default: *value = 0; break;
    }
    return cudaSuccess;
}
inline std::atomic<uint64_t>& device_used_bytes() {
    static std::atomic<uint64_t> used{0};
    return used;
}
inline std::mutex& device_alloc_mu() {
    static std::mutex mu;
    return mu;
}
inline std::unordered_map<void*, size_t>& device_allocs() {
    static std::unordered_map<void*, size_t> m;
    return m;
}
inline cudaError_t cudaMemGetInfo(size_t* free_bytes, size_t* total_bytes) {
    try {
        auto d = strata::xpu::gpu_device(0);
        *total_bytes = d.get_info<sycl::info::device::global_mem_size>();
        const uint64_t used = device_used_bytes().load();
        *free_bytes = *total_bytes > used ? *total_bytes - used : 0;
        return cudaSuccess;
    } catch (...) {
        return cudaErrorUnknown;
    }
}
inline cudaError_t cudaDriverGetVersion(int* v) { *v = 12080; return cudaSuccess; }
inline cudaError_t cudaRuntimeGetVersion(int* v) { *v = 12080; return cudaSuccess; }
inline cudaError_t cudaDeviceSynchronize() {
    try { strata::xpu::default_queue().wait(); return cudaSuccess; }
    catch (...) { return cudaErrorUnknown; }
}
inline cudaError_t cudaDeviceCanAccessPeer(int* can, int, int) { *can = 0; return cudaSuccess; }
inline cudaError_t cudaDeviceEnablePeerAccess(int, unsigned) { return cudaErrorInvalidValue; }

inline cudaError_t cudaStreamCreate(cudaStream_t* stream) {
    try {
        *stream = new strata::xpu::Stream(strata::xpu::make_queue());
        return cudaSuccess;
    } catch (...) { return cudaErrorMemoryAllocation; }
}
inline cudaError_t cudaStreamCreateWithFlags(cudaStream_t* stream, unsigned) { return cudaStreamCreate(stream); }
inline cudaError_t cudaStreamDestroy(cudaStream_t stream) {
    if (stream && stream != strata::xpu::default_stream()) delete stream;
    return cudaSuccess;
}
inline cudaError_t cudaStreamSynchronize(cudaStream_t stream) {
    try { strata::xpu::as_stream(stream)->q.wait(); return cudaSuccess; }
    catch (...) { return cudaErrorUnknown; }
}
inline cudaError_t cudaStreamQuery(cudaStream_t stream) {
    // Honest non-blocking idle check: the verify window polls this while spinning on a doorbell,
    // so a lie here reads as "graph finished" while the GPU is still running (or JIT-ing).
    // The queues are in_order, so the last event completing implies everything before it did.
    try {
        auto& q = strata::xpu::as_stream(stream)->q;
        auto ev = q.ext_oneapi_get_last_event();
        if (!ev.has_value()) return cudaSuccess;   // nothing ever submitted: idle
        const auto st = ev->get_info<sycl::info::event::command_execution_status>();
        return st == sycl::info::event_command_status::complete ? cudaSuccess : cudaErrorNotReady;
    } catch (...) {
        return cudaSuccess;   // nothing ever submitted: idle
    }
}
inline cudaError_t cudaStreamWaitEvent(cudaStream_t stream, cudaEvent_t event, unsigned) {
    if (!event || !event->recorded) return cudaSuccess;
    try {
        strata::xpu::as_stream(stream)->q.ext_oneapi_submit_barrier({event->ev});
        return cudaSuccess;
    } catch (...) { return cudaErrorInvalidValue; }
}

inline cudaError_t cudaEventCreate(cudaEvent_t* event) {
    *event = new strata::xpu::Event();
    return cudaSuccess;
}
inline cudaError_t cudaEventCreateWithFlags(cudaEvent_t* event, unsigned) { return cudaEventCreate(event); }
inline cudaError_t cudaEventDestroy(cudaEvent_t event) { delete event; return cudaSuccess; }
inline cudaError_t cudaEventRecord(cudaEvent_t event, cudaStream_t stream = nullptr) {
    if (!event) return cudaErrorInvalidValue;
    try {
        auto& q = strata::xpu::as_stream(stream)->q;
        event->ev = q.ext_oneapi_submit_barrier();
        event->recorded = true;
        return cudaSuccess;
    } catch (...) { return cudaErrorUnknown; }
}
inline cudaError_t cudaEventSynchronize(cudaEvent_t event) {
    if (!event || !event->recorded) return cudaSuccess;
    try { event->ev.wait(); return cudaSuccess; }
    catch (...) { return cudaErrorUnknown; }
}
inline cudaError_t cudaEventQuery(cudaEvent_t event) {
    if (!event || !event->recorded) return cudaSuccess;
    try {
        auto st = event->ev.get_info<sycl::info::event::command_execution_status>();
        return st == sycl::info::event_command_status::complete ? cudaSuccess : cudaErrorNotReady;
    } catch (...) { return cudaErrorNotReady; }
}
inline cudaError_t cudaEventElapsedTime(float* ms, cudaEvent_t start, cudaEvent_t end) {
    try {
        auto a = start->ev.get_profiling_info<sycl::info::event_profiling::command_end>();
        auto b = end->ev.get_profiling_info<sycl::info::event_profiling::command_end>();
        *ms = static_cast<float>(static_cast<double>(b - a) / 1.0e6);
        return cudaSuccess;
    } catch (...) {
        *ms = 0.f;
        return cudaSuccess;
    }
}

template <typename T>
inline cudaError_t cudaMalloc(T** ptr, size_t bytes) {
    try {
        void* p = bytes ? sycl::malloc_device(bytes, strata::xpu::default_queue()) : nullptr;
        if (bytes && !p) {
            strata::xpu::last_error = cudaErrorMemoryAllocation;
            strata::xpu::last_error_text = "out of memory";
            return cudaErrorMemoryAllocation;
        }
        if (p && bytes) {
            std::lock_guard<std::mutex> lock(device_alloc_mu());
            device_allocs()[p] = bytes;
            device_used_bytes() += bytes;
        }
        *ptr = static_cast<T*>(p);
        return cudaSuccess;
    } catch (...) {
        strata::xpu::last_error = cudaErrorMemoryAllocation;
        strata::xpu::last_error_text = "out of memory";
        return cudaErrorMemoryAllocation;
    }
}
inline cudaError_t cudaFree(void* ptr) {
    if (!ptr) return cudaSuccess;
    size_t bytes = 0;
    {
        std::lock_guard<std::mutex> lock(device_alloc_mu());
        auto it = device_allocs().find(ptr);
        if (it != device_allocs().end()) {
            bytes = it->second;
            device_allocs().erase(it);
        }
    }
    if (bytes) device_used_bytes() -= bytes;
    sycl::free(ptr, strata::xpu::default_queue());
    return cudaSuccess;
}
// Shared USM migrates to the device after the first copy, and the next fread into it
// returns EFAULT. Staging buffers (no Mapped flag) stay ordinary host memory. Mapped
// buffers use malloc_host, which the device can read and which stays mapped.
inline std::unordered_set<void*>& sycl_host_ptrs() {
    static std::unordered_set<void*> ptrs;
    return ptrs;
}
inline std::mutex& sycl_host_mu() {
    static std::mutex mu;
    return mu;
}
template <typename T>
inline cudaError_t cudaHostAlloc(T** ptr, size_t bytes, unsigned flags) {
    try {
        void* p = nullptr;
        if (bytes == 0) {
            *ptr = nullptr;
            return cudaSuccess;
        }
        if (flags & cudaHostAllocMapped) {
            p = sycl::malloc_host(bytes, strata::xpu::default_queue());
            if (!p) return cudaErrorMemoryAllocation;
            std::lock_guard<std::mutex> lock(sycl_host_mu());
            sycl_host_ptrs().insert(p);
        } else {
            p = std::aligned_alloc(64, (bytes + 63u) & ~size_t{63});
            if (!p) return cudaErrorMemoryAllocation;
        }
        *ptr = static_cast<T*>(p);
        return cudaSuccess;
    } catch (...) { return cudaErrorMemoryAllocation; }
}
template <typename T>
inline cudaError_t cudaMallocHost(T** ptr, size_t bytes) {
    return cudaHostAlloc(ptr, bytes, cudaHostAllocMapped);
}
inline cudaError_t cudaFreeHost(void* ptr) {
    if (!ptr) return cudaSuccess;
    bool sycl = false;
    {
        std::lock_guard<std::mutex> lock(sycl_host_mu());
        sycl = sycl_host_ptrs().erase(ptr) != 0;
    }
    if (sycl) sycl::free(ptr, strata::xpu::default_queue());
    else std::free(ptr);
    return cudaSuccess;
}
template <typename T>
inline cudaError_t cudaHostGetDevicePointer(T** dev, void* host, unsigned) {
    *dev = static_cast<T*>(host);
    return host ? cudaSuccess : cudaErrorInvalidValue;
}
inline cudaError_t cudaHostRegister(void*, size_t, unsigned) { return cudaErrorInvalidValue; }
inline cudaError_t cudaHostUnregister(void*) { return cudaSuccess; }

inline bool usm_known(const void* p, sycl::queue& q) {
    if (!p) return false;
    return sycl::get_pointer_type(p, q.get_context()) != sycl::usm::alloc::unknown;
}
inline cudaError_t cudaMemcpy(void* dst, const void* src, size_t bytes, cudaMemcpyKind) {
    if (!bytes) return cudaSuccess;
    auto& q = strata::xpu::default_queue();
    try {
        // A failed memcpy of an ordinary host pointer (GGUF mapping, malloc) can leak
        // a device buffer. Stage those through malloc_host before the runtime sees them.
        if (!usm_known(src, q) || !usm_known(dst, q)) {
            void* tmp_src = nullptr;
            void* tmp_dst = nullptr;
            const void* copy_src = src;
            void* copy_dst = dst;
            if (!usm_known(src, q)) {
                tmp_src = sycl::malloc_host(bytes, q);
                if (!tmp_src) return cudaErrorMemoryAllocation;
                std::memcpy(tmp_src, src, bytes);
                copy_src = tmp_src;
            }
            if (!usm_known(dst, q)) {
                tmp_dst = sycl::malloc_host(bytes, q);
                if (!tmp_dst) {
                    if (tmp_src) sycl::free(tmp_src, q);
                    return cudaErrorMemoryAllocation;
                }
                copy_dst = tmp_dst;
            }
            q.memcpy(copy_dst, copy_src, bytes).wait();
            if (tmp_dst) {
                std::memcpy(dst, tmp_dst, bytes);
                sycl::free(tmp_dst, q);
            }
            if (tmp_src) sycl::free(tmp_src, q);
            return cudaSuccess;
        }
        q.memcpy(dst, src, bytes).wait();
        return cudaSuccess;
    } catch (const sycl::exception& e) {
        strata::xpu::last_error_text = e.what();
        return cudaErrorInvalidValue;
    }
}
inline void*& host_bounce() { static void* p = nullptr; return p; }
inline size_t& host_bounce_cap() { static size_t s = 0; return s; }
inline std::mutex& host_bounce_mu() { static std::mutex m; return m; }
// One shared malloc_host bounce, filled by a host_task in queue order: on an in-order queue the
// device copy of op N runs after op N's fill and before op N+1's fill, so one buffer suffices.
inline void* host_bounce_get(size_t bytes, sycl::queue& q) {
    std::lock_guard<std::mutex> lock(host_bounce_mu());
    if (host_bounce_cap() < bytes) {
        if (host_bounce()) sycl::free(host_bounce(), q);
        host_bounce_cap() = std::max(bytes, (size_t) 32 << 20);
        host_bounce() = sycl::malloc_host(host_bounce_cap(), q);
    }
    return host_bounce();
}
inline cudaError_t cudaMemcpyAsync(void* dst, const void* src, size_t bytes, cudaMemcpyKind kind, cudaStream_t stream = nullptr) {
    if (!bytes) return cudaSuccess;
    try {
        auto* s = strata::xpu::as_stream(stream);
        auto& q = s->q;
        const bool src_usm = usm_known(src, q);
        const bool dst_usm = usm_known(dst, q);
        if ((src_usm && dst_usm) || kind == cudaMemcpyDeviceToDevice) {
            auto op = [dst, src, bytes](sycl::queue& q2) { q2.memcpy(dst, src, bytes); };
            if (s->capture) s->capture->push(std::move(op), "memcpyAsync");
            else op(q);
            return cudaSuccess;
        }
        // Level Zero copies fault on ordinary host pages (mmap'd weight files, malloc'd vectors), and a
        // faulted async copy strands every later command on the stream - including host functions that
        // raise handshake flags. Stage through the malloc_host bounce: a host_task fills it in queue
        // order, then the device copy reads it.
        void* b = host_bounce_get(bytes, q);
        if (strata::xpu::trace_enabled()) {
            std::fprintf(stderr, "xpu trace: memcpyAsync staged %zu B (%s -> %s)\n", bytes,
                         src_usm ? "usm" : "host", dst_usm ? "usm" : "host");
            std::fflush(stderr);
        }
        if (src_usm && !dst_usm) {           // D2H into an ordinary host buffer
            auto op = [dst, src, bytes, b](sycl::queue& q2) {
                q2.memcpy(b, src, bytes);
                q2.submit([&](sycl::handler& h) { h.host_task([=] { std::memcpy(dst, b, bytes); }); });
            };
            if (s->capture) s->capture->push(std::move(op), "memcpyAsync");
            else op(q);
            return cudaSuccess;
        }
        {                                     // H2D (or any) from an ordinary host buffer
            auto op = [dst, src, bytes, b](sycl::queue& q2) {
                q2.submit([&](sycl::handler& h) { h.host_task([=] { std::memcpy(b, src, bytes); }); });
                q2.memcpy(dst, b, bytes);
            };
            if (s->capture) s->capture->push(std::move(op), "memcpyAsync");
            else op(q);
        }
        return cudaSuccess;
    } catch (...) { return cudaErrorInvalidValue; }
}
inline cudaError_t cudaMemcpy2DAsync(void* dst, size_t dpitch, const void* src, size_t spitch, size_t width, size_t height, cudaMemcpyKind kind, cudaStream_t stream) {
    for (size_t row = 0; row < height; ++row) {
        auto e = cudaMemcpyAsync(static_cast<char*>(dst) + row * dpitch,
                                 static_cast<const char*>(src) + row * spitch, width, kind, stream);
        if (e != cudaSuccess) return e;
    }
    return cudaSuccess;
}
inline cudaError_t cudaMemcpyPeerAsync(void* dst, int, const void* src, int, size_t bytes, cudaStream_t stream) {
    return cudaMemcpyAsync(dst, src, bytes, cudaMemcpyDeviceToDevice, stream);
}
template <typename T>
inline cudaError_t cudaMemcpyToSymbol(T& symbol, const void* src, size_t bytes, size_t offset = 0, cudaMemcpyKind = cudaMemcpyDefault) {
    std::memcpy(reinterpret_cast<char*>(&symbol) + offset, src, bytes);
    return cudaSuccess;
}
inline cudaError_t cudaMemset(void* ptr, int value, size_t bytes) {
    auto& q = strata::xpu::default_queue();
    constexpr size_t chunk = 256ull << 20;
    try {
        for (size_t off = 0; off < bytes; off += chunk) {
            const size_t n = std::min(chunk, bytes - off);
            q.memset(static_cast<char*>(ptr) + off, value, n).wait();
        }
        return cudaSuccess;
    } catch (const sycl::exception& e) {
        strata::xpu::last_error_text = e.what();
        strata::xpu::last_error = cudaErrorInvalidValue;
        return cudaErrorInvalidValue;
    }
}
inline cudaError_t cudaMemsetAsync(void* ptr, int value, size_t bytes, cudaStream_t stream) {
    try {
        auto* s = strata::xpu::as_stream(stream);
        auto op = [ptr, value, bytes](sycl::queue& q) { q.memset(ptr, value, bytes); };
        if (s->capture) s->capture->push(std::move(op), "memsetAsync");
        else op(s->q);
        return cudaSuccess;
    } catch (...) { return cudaErrorInvalidValue; }
}

inline cudaError_t cudaStreamBeginCapture(cudaStream_t stream, cudaStreamCaptureMode) {
    auto* s = strata::xpu::as_stream(stream);
    if (s->capture) return cudaErrorStreamCaptureUnsupported;
    s->capture = new strata::xpu::Graph();
    s->capture->record_q = &s->q;
    return cudaSuccess;
}
inline cudaError_t cudaStreamEndCapture(cudaStream_t stream, cudaGraph_t* graph) {
    auto* s = strata::xpu::as_stream(stream);
    if (!s->capture) return cudaErrorInvalidValue;
    *graph = s->capture;
    s->capture = nullptr;
    if ((*graph)->ops.empty()) return cudaErrorInvalidValue;
    return cudaSuccess;
}
inline cudaError_t cudaStreamIsCapturing(cudaStream_t stream, cudaStreamCaptureStatus* status) {
    *status = strata::xpu::as_stream(stream)->capture ? cudaStreamCaptureStatusActive : cudaStreamCaptureStatusNone;
    return cudaSuccess;
}
inline cudaError_t cudaGraphDestroy(cudaGraph_t graph) { delete graph; return cudaSuccess; }
inline bool graph_profile_mode();
inline void profile_print();
inline cudaError_t cudaGraphExecDestroy(cudaGraphExec_t graph) {
    if (graph && !graph->native.has_value() && graph_profile_mode()) {
        static bool printed = false;   // one summary per process, at the first teardown
        if (!printed) { printed = true; profile_print(); }
    }
    delete graph; return cudaSuccess;
}
inline bool native_graphs_disabled() {
    static const bool off = std::getenv("STRATA_XPU_EAGER_GRAPHS") != nullptr;
    return off;
}
inline bool graph_profile_mode() {
    static const bool on = std::getenv("STRATA_XPU_PROFILE") != nullptr;
    return on;
}
// STRATA_XPU_PROFILE: replay eagerly and bracket each op with barrier events (non-blocking;
// per-op waits would deadlock on the doorbell spin kernels). The deltas are read at teardown,
// when every event has long completed, and the top offenders print.
inline std::mutex& profile_mu() { static std::mutex m; return m; }
inline std::map<std::string, std::pair<double, int>>& profile_acc() {
    static std::map<std::string, std::pair<double, int>> m;
    return m;
}
struct ProfileSpan {
    std::string name;
    sycl::event before, after;
};
inline std::vector<ProfileSpan>& profile_spans() {
    static std::vector<ProfileSpan> v;
    return v;
}
inline void profile_collect() {
    // Called at the START of a launch: by then the previous window's replay has fully completed
    // (the engine serves every layer before the next launch), so every event is queryable while
    // its queue is alive. Never called from teardown - events may outlive their queue there.
    std::lock_guard<std::mutex> lock(profile_mu());
    for (auto& s : profile_spans()) {
        try {
            const uint64_t t0 = s.before.get_profiling_info<sycl::info::event_profiling::command_end>();
            const uint64_t t1 = s.after.get_profiling_info<sycl::info::event_profiling::command_end>();
            auto& acc = profile_acc()[s.name];
            acc.first += double(t1 - t0) / 1.0e6;   // ns -> ms
            acc.second += 1;
        } catch (...) { /* an incomplete span: skip it */ }
    }
    profile_spans().clear();
}
inline void profile_print() {
    std::vector<std::pair<double, std::pair<std::string, int>>> rows;
    double total = 0;
    for (auto& [name, v] : profile_acc()) {
        rows.push_back({v.first, {name, v.second}});
        total += v.first;
    }
    std::sort(rows.begin(), rows.end(), [](auto& a, auto& b) { return a.first > b.first; });
    std::fprintf(stderr, "xpu profile: %zu kernels, %.1f ms total; top 25 by GPU time\n", rows.size(), total);
    for (size_t i = 0; i < rows.size() && i < 25; ++i)
        std::fprintf(stderr, "  %8.2f ms  %5d x  %s\n", rows[i].first, rows[i].second.second,
                     rows[i].second.first.c_str());
}
inline cudaError_t cudaGraphInstantiate(cudaGraphExec_t* exec, cudaGraph_t graph, unsigned long long) {
    if (!graph || !exec) return cudaErrorInvalidValue;
    // CUDA semantics: the exec is independent of the graph. The engine destroys the graph right
    // after instantiating (verify.cpp, session.cpp, mtp.cpp, graph.cpp all do), so aliasing the
    // two pointers would make every later cudaGraphLaunch a use-after-free.
    auto* e = new strata::xpu::Graph();
    e->ops = graph->ops;   // std::function copies; captured buffer pointers are shared, as in CUDA
    e->names = graph->names;
    e->record_q = graph->record_q;
    // Record the op list into a native ext_oneapi_graph: one replay submission instead of one
    // host call per op. Recorded on the capture stream's own queue so nothing else is captured.
    if (!native_graphs_disabled() && !graph_profile_mode() && graph->record_q != nullptr) {
        namespace exp = sycl::ext::oneapi::experimental;
        try {
            sycl::queue& rq = *graph->record_q;
            exp::command_graph g{rq.get_context(), rq.get_device()};
            g.begin_recording(rq);
            try {
                for (auto& op : e->ops) op(rq);
            } catch (...) {
                g.end_recording();   // leave the queue usable whatever happened mid-recording
                throw;
            }
            g.end_recording();
            e->native = g.finalize();
        } catch (...) {
            e->native.reset();   // fall back to the eager loop
        }
    }
    *exec = e;
    return cudaSuccess;
}
inline cudaError_t cudaGraphInstantiate(cudaGraphExec_t* exec, cudaGraph_t graph, void*, char*, size_t) {
    return cudaGraphInstantiate(exec, graph, 0);
}
inline cudaError_t cudaGraphLaunch(cudaGraphExec_t exec, cudaStream_t stream) {
    if (!exec) return cudaErrorInvalidValue;
    try {
        auto& q = strata::xpu::as_stream(stream)->q;
        if (graph_profile_mode()) profile_collect();   // drain the previous replay's spans first
        if (exec->native.has_value()) {
            q.ext_oneapi_graph(*exec->native);   // one submission for the whole graph
            return cudaSuccess;
        }
        const bool prof = graph_profile_mode();
        for (size_t i = 0; i < exec->ops.size(); ++i) {
            if (strata::xpu::trace_enabled()) {
                std::fprintf(stderr, "xpu trace: replay[%zu/%zu] %s\n", i, exec->ops.size(), exec->names[i].c_str());
                std::fflush(stderr);
            }
            if (prof) {
                ProfileSpan span;
                span.name = exec->names[i];
                span.before = q.ext_oneapi_submit_barrier();
                exec->ops[i](q);
                span.after = q.ext_oneapi_submit_barrier();
                std::lock_guard<std::mutex> lock(profile_mu());
                if (profile_spans().size() < 40000) profile_spans().push_back(std::move(span));
                continue;
            }
            exec->ops[i](q);
            if (strata::xpu::trace_enabled()) {
                const std::string nm = exec->names[i];
                q.submit([&](sycl::handler& h) {
                    h.host_task([i, nm] {
                        std::fprintf(stderr, "xpu trace: exec[%zu] %s done\n", i, nm.c_str());
                        std::fflush(stderr);
                    });
                });
            }
        }
        return cudaSuccess;
    } catch (...) { return cudaErrorUnknown; }
}
inline cudaError_t cudaGraphUpload(cudaGraphExec_t, cudaStream_t) { return cudaSuccess; }
inline cudaError_t cudaGraphGetNodes(cudaGraph_t graph, cudaGraphNode_t* nodes, size_t* num) {
    size_t n = graph ? graph->ops.size() : 0;
    if (!nodes) { *num = n; return cudaSuccess; }
    size_t m = std::min(*num, n);
    for (size_t i = 0; i < m; ++i) nodes[i] = nullptr;
    *num = m;
    return cudaSuccess;
}
inline cudaError_t cudaGraphNodeGetType(cudaGraphNode_t, cudaGraphNodeType* ty) {
    *ty = cudaGraphNodeTypeKernel;
    return cudaSuccess;
}
inline cudaError_t cudaGraphKernelNodeGetParams(cudaGraphNode_t, cudaKernelNodeParams*) { return cudaErrorInvalidValue; }

inline cudaError_t cudaLaunchHostFunc(cudaStream_t stream, void (*fn)(void*), void* user) {
    auto* s = strata::xpu::as_stream(stream);
    auto op = [fn, user](sycl::queue& q) { q.submit([&](sycl::handler& h) { h.host_task([=] { fn(user); }); }); };
    if (s->capture) s->capture->push(std::move(op), "hostFunc");
    else op(s->q);
    return cudaSuccess;
}

template <typename Kernel>
inline cudaError_t cudaFuncSetAttribute(Kernel, int, int) { return cudaSuccess; }
inline cudaError_t cudaFuncGetName(const char** name, const void*) {
    if (name) *name = "xpu_kernel";
    return cudaSuccess;
}
template <typename Kernel>
inline cudaError_t cudaFuncGetAttributes(cudaFuncAttributes* a, Kernel) {
    *a = {};
    a->maxThreadsPerBlock = 1024;
    a->ptxVersion = 80;
    a->binaryVersion = 80;
    return cudaSuccess;
}
template <typename Kernel>
inline cudaError_t cudaOccupancyMaxActiveBlocksPerMultiprocessor(int* num, Kernel, int block, size_t smem) {
    int sms = 20;
    cudaDeviceGetAttribute(&sms, cudaDevAttrMultiProcessorCount, 0);
    int by_smem = 4;
    if (smem > 0) {
        size_t local = 64 * 1024;
        cudaDeviceProp p{};
        if (cudaGetDeviceProperties(&p, 0) == cudaSuccess && p.sharedMemPerBlock)
            local = p.sharedMemPerBlock;
        by_smem = static_cast<int>(std::max<size_t>(1, local / smem));
    }
    int by_threads = block > 0 ? std::max(1, 2048 / block) : 1;
    *num = std::max(1, std::min(sms, std::min(by_smem, by_threads)));
    return cudaSuccess;
}
inline cudaError_t cudaOccupancyMaxActiveClusters(int* clusters, const void*, const void*) {
    *clusters = 0;
    return cudaErrorInvalidValue;
}

struct cudaLaunchAttribute { int id; int val; };
struct cudaLaunchConfig_t {
    dim3 gridDim{};
    dim3 blockDim{};
    size_t dynamicSmemBytes = 0;
    cudaStream_t stream = nullptr;
    cudaLaunchAttribute* attrs = nullptr;
    int numAttrs = 0;
};
constexpr int cudaLaunchAttributeClusterDimension = 1;
template <typename Kernel, typename... Args>
inline cudaError_t cudaLaunchKernelEx(const cudaLaunchConfig_t*, Kernel, Args...) {
    return cudaErrorInvalidValue;
}

inline int __int_as_float_host_unused() { return 0; }
inline float __int_as_float(int v) { float f; std::memcpy(&f, &v, 4); return f; }
inline float __uint_as_float(unsigned v) { float f; std::memcpy(&f, &v, 4); return f; }
inline int __float2int_rn(float x) { return static_cast<int>(sycl::rint(x)); }
inline int __float2int_rz(float x) { return static_cast<int>(x); }
inline int __float2int_rd(float x) { return static_cast<int>(sycl::floor(x)); }
inline int __float2int_ru(float x) { return static_cast<int>(sycl::ceil(x)); }
inline int __float_as_int(float v) { int i; std::memcpy(&i, &v, 4); return i; }
inline unsigned __float_as_uint(float v) { unsigned i; std::memcpy(&i, &v, 4); return i; }

#include "cuda_fp16.h"
inline float2 __half22float2(__half2 h) { return {__half2float(h.x), __half2float(h.y)}; }
inline float2 __half22float2(sycl::half2 h) { return {static_cast<float>(h[0]), static_cast<float>(h[1])}; }

#include "intrinsics.hpp"

#endif  // __CUDA_RUNTIME_H__
