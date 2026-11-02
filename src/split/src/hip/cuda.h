// HIP stand-in for the CUDA driver API calls src/split makes: stream memory operations (the prompt path's copy-engine ring and
// attention flags). The sources include <cuda.h>; the HIP build puts this directory first on the include path.
#pragma once
#include <hip/hip_runtime.h>
#include <cstdint>

typedef hipError_t CUresult;
typedef hipStream_t CUstream;
typedef unsigned long long CUdeviceptr;
#define CUDA_SUCCESS hipSuccess
#define CU_STREAM_WAIT_VALUE_GEQ hipStreamWaitValueGte

inline CUresult cuGetErrorString(CUresult e, const char** s) {
    *s = hipGetErrorString(e);
    return hipSuccess;
}
// a 32-bit write / wait on the stream (host memory the card maps). Two implementations, chosen once by FNX_HIP_MEMOP:
//   kernel (default): a one-thread kernel writes the flag (after a system fence) or spins on it, in stream order. Every wait is a tiny
//                     kernel on the stream; it needs the card to run it while the other card's stream catches up, nothing else.
//   native (FNX_HIP_MEMOP=native): hipStreamWriteValue32 / hipStreamWaitValue32 (the runtime's own stream memory operations).
// A wait that does not see its value within ~4 s sets bit 1 of nothing and returns (the engine's ring error check then catches the sum).
#include <cstdlib>
#include <cstring>
static __global__ void fnx_memop_write(volatile unsigned* p, unsigned v) {
    __threadfence_system();   // everything this stream wrote before (the copy engine's / kernels' stores into host memory) first
    *p = v;
    __threadfence_system();
}
static __global__ void fnx_memop_wait(volatile unsigned* p, unsigned v) {
    const long long t0 = clock64();
    while ((int) (*p - v) < 0) {   // wrapping >=
        __builtin_amdgcn_s_sleep(1);
        if (clock64() - t0 > 8000000000LL) break;
    }
    __threadfence_system();
}
inline bool fnx_native_memops() {
    static const bool native = [] { const char* e = std::getenv("FNX_HIP_MEMOP"); return e && !std::strcmp(e, "native"); }();
    return native;
}
inline CUresult cuStreamWriteValue32(CUstream s, CUdeviceptr p, uint32_t v, unsigned flags) {
    if (fnx_native_memops()) return hipStreamWriteValue32(s, reinterpret_cast<void*>(p), static_cast<int32_t>(v), flags);
    fnx_memop_write<<<1, 1, 0, s>>>(reinterpret_cast<volatile unsigned*>(p), v);
    return hipGetLastError();
}
inline CUresult cuStreamWaitValue32(CUstream s, CUdeviceptr p, uint32_t v, unsigned flags) {
    if (fnx_native_memops()) return hipStreamWaitValue32(s, reinterpret_cast<void*>(p), static_cast<int32_t>(v), flags, 0xffffffffu);
    fnx_memop_wait<<<1, 1, 0, s>>>(reinterpret_cast<volatile unsigned*>(p), v);
    return hipGetLastError();
}
inline CUresult cuInit(unsigned) { return hipInit(0); }
