// Test-only runtime declarations. This is NOT a CUDA implementation or a production backend.
// GPU allocations fail; host-registration results can be injected to test the real arena's fallback.
#pragma once
#ifndef STRATA_MLOCK_HOST_TEST
#error "The mlock CUDA stub must never be used in a production target"
#endif
#include <cstddef>
#include <cstdlib>
#include <cstring>
using cudaError_t = int;
using cudaStream_t = void*;
using cudaEvent_t = void*;
using cudaGraph_t = void*;
using cudaGraphExec_t = void*;
using cudaGraphNode_t = void*;
constexpr int cudaSuccess = 0, cudaErrorMemoryAllocation = 2, cudaErrorNotReady = 34, cudaErrorNotSupported = 801;
constexpr unsigned cudaHostAllocDefault = 0, cudaHostAllocPortable = 1, cudaHostAllocMapped = 2;
constexpr unsigned cudaHostRegisterPortable = 1, cudaHostRegisterMapped = 2;
constexpr unsigned cudaStreamNonBlocking = 1, cudaEventDisableTiming = 2, cudaEventBlockingSync = 1;
enum cudaMemcpyKind { cudaMemcpyHostToHost, cudaMemcpyHostToDevice, cudaMemcpyDeviceToHost, cudaMemcpyDeviceToDevice, cudaMemcpyDefault };
constexpr int cudaMemoryTypeDevice = 2;
struct cudaPointerAttributes { int type = 0, device = 0; void* hostPointer = nullptr; void* devicePointer = nullptr; };
struct cudaDeviceProp { int integrated = 0, major = 8, minor = 0, multiProcessorCount = 1, warpSize = 32; char name[256] = {}; char luid[8] = {}; };
inline bool strata_test_register_success = false;
inline cudaError_t cudaHostRegister(void*, size_t, unsigned) { return strata_test_register_success ? cudaSuccess : cudaErrorMemoryAllocation; }
inline cudaError_t cudaHostUnregister(void*) { return cudaSuccess; }
inline cudaError_t cudaHostAlloc(void**, size_t, unsigned) { return cudaErrorMemoryAllocation; }
inline cudaError_t cudaMallocHost(void** p, size_t n) { return cudaHostAlloc(p, n, 0); }
inline cudaError_t cudaFreeHost(void* p) { std::free(p); return cudaSuccess; }
inline cudaError_t cudaMalloc(void**, size_t) { return cudaErrorMemoryAllocation; }
template<class T> inline cudaError_t cudaMalloc(T** p, size_t n) { return cudaMalloc(reinterpret_cast<void**>(p), n); }
inline cudaError_t cudaFree(void*) { return cudaSuccess; }
inline cudaError_t cudaHostGetDevicePointer(void** d, void* h, unsigned) { *d = h; return cudaSuccess; }
inline cudaError_t cudaGetDevice(int* d) { *d = 0; return cudaSuccess; }
inline cudaError_t cudaSetDevice(int) { return cudaSuccess; }
inline cudaError_t cudaGetDeviceProperties(cudaDeviceProp* p, int) { *p = {}; return cudaSuccess; }
inline cudaError_t cudaMemGetInfo(size_t* f, size_t* t) { *f = *t = 0; return cudaSuccess; }
inline cudaError_t cudaGetLastError() { return cudaSuccess; }
inline cudaError_t cudaPeekAtLastError() { return cudaSuccess; }
inline const char* cudaGetErrorString(cudaError_t) { return "test runtime: GPU unavailable / injected registration refusal"; }
inline cudaError_t cudaDeviceSynchronize() { return cudaSuccess; }
inline cudaError_t cudaStreamCreate(cudaStream_t* s) { *s = reinterpret_cast<void*>(1); return cudaSuccess; }
inline cudaError_t cudaStreamCreateWithFlags(cudaStream_t* s, unsigned) { return cudaStreamCreate(s); }
inline cudaError_t cudaStreamDestroy(cudaStream_t) { return cudaSuccess; }
inline cudaError_t cudaStreamSynchronize(cudaStream_t) { return cudaSuccess; }
inline cudaError_t cudaEventCreate(cudaEvent_t* e) { *e = reinterpret_cast<void*>(1); return cudaSuccess; }
inline cudaError_t cudaEventCreateWithFlags(cudaEvent_t* e, unsigned) { return cudaEventCreate(e); }
inline cudaError_t cudaEventDestroy(cudaEvent_t) { return cudaSuccess; }
inline cudaError_t cudaEventQuery(cudaEvent_t) { return cudaSuccess; }
inline cudaError_t cudaEventRecord(cudaEvent_t, cudaStream_t = nullptr) { return cudaSuccess; }
inline cudaError_t cudaEventSynchronize(cudaEvent_t) { return cudaSuccess; }
inline cudaError_t cudaEventElapsedTime(float* t, cudaEvent_t, cudaEvent_t) { *t = 0; return cudaSuccess; }
inline cudaError_t cudaStreamWaitEvent(cudaStream_t, cudaEvent_t, unsigned) { return cudaSuccess; }
inline cudaError_t cudaPointerGetAttributes(cudaPointerAttributes* p, const void*) { *p = {}; return cudaSuccess; }
inline cudaError_t strata_test_copy_error = cudaSuccess;
inline cudaError_t cudaMemcpy(void* d, const void* s, size_t n, cudaMemcpyKind) {
    if (strata_test_copy_error != cudaSuccess) return strata_test_copy_error;
    std::memcpy(d, s, n); return cudaSuccess;
}
inline cudaError_t cudaMemcpyAsync(void* d, const void* s, size_t n, cudaMemcpyKind k, cudaStream_t = nullptr) { return cudaMemcpy(d, s, n, k); }
inline cudaError_t cudaMemset(void* p, int v, size_t n) { std::memset(p, v, n); return cudaSuccess; }
inline cudaError_t cudaMemsetAsync(void* p, int v, size_t n, cudaStream_t = nullptr) { return cudaMemset(p, v, n); }
inline cudaError_t cudaMemcpy2DAsync(void* d, size_t dp, const void* s, size_t sp, size_t w, size_t h, cudaMemcpyKind, cudaStream_t = nullptr) {
    for (size_t r = 0; r < h; ++r) std::memcpy(static_cast<char*>(d) + r * dp, static_cast<const char*>(s) + r * sp, w);
    return cudaSuccess;
}
