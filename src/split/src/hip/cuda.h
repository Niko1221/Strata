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
// a 32-bit write / wait on the stream (host memory the card maps): hipStreamWriteValue32 / hipStreamWaitValue32
inline CUresult cuStreamWriteValue32(CUstream s, CUdeviceptr p, uint32_t v, unsigned flags) {
    return hipStreamWriteValue32(s, reinterpret_cast<void*>(p), static_cast<int32_t>(v), flags);
}
inline CUresult cuStreamWaitValue32(CUstream s, CUdeviceptr p, uint32_t v, unsigned flags) {
    return hipStreamWaitValue32(s, reinterpret_cast<void*>(p), static_cast<int32_t>(v), flags, 0xffffffffu);
}
inline CUresult cuInit(unsigned) { return hipInit(0); }
