// Force-included (after the tree's hip_compat/cuda_runtime.h) into every src/split source of the HIP build: the CUDA runtime and
// cuBLAS names src/split uses that the shared shim does not rename.
#pragma once
#include <hip/hip_runtime.h>

#define cudaDeviceGetPCIBusId hipDeviceGetPCIBusId
#define cudaPointerAttributes hipPointerAttribute_t
#define cudaPointerGetAttributes hipPointerGetAttributes
#define cudaGetErrorName hipGetErrorName
#define cudaStreamCaptureModeRelaxed hipStreamCaptureModeRelaxed
#define cudaMemcpyHostToHost hipMemcpyHostToHost
#define cudaGraphExecUpdateResultInfo fnx_graph_update_info
struct fnx_graph_update_info {
    hipGraphNode_t errorNode = nullptr;
    hipGraphExecUpdateResult result = hipGraphExecUpdateSuccess;
};
inline hipError_t cudaGraphExecUpdate(hipGraphExec_t exec, hipGraph_t g, fnx_graph_update_info* info) {
    return hipGraphExecUpdate(exec, g, &info->errorNode, &info->result);
}

// cuBLAS: the f16 products with tensor-op math are hipBLAS's default on a gfx12 card
#define CUBLAS_COMPUTE_16F HIPBLAS_COMPUTE_16F
#define CUBLAS_GEMM_DEFAULT_TENSOR_OP HIPBLAS_GEMM_DEFAULT
#define CUBLAS_TENSOR_OP_MATH HIPBLAS_DEFAULT_MATH

// __ldcg: a load that skips the non-coherent caches (data another block of the same launch wrote, behind the hc read's grid barrier):
// a relaxed agent-scope atomic load of 4 / 8 bytes
template <typename T> __device__ __forceinline__ T __ldcg(const T* p) {
    static_assert(sizeof(T) == 4 || sizeof(T) == 8, "__ldcg: 4 or 8 bytes");
    if constexpr (sizeof(T) == 4) {
        const unsigned u = __hip_atomic_load(reinterpret_cast<const unsigned*>(p), __ATOMIC_RELAXED, __HIP_MEMORY_SCOPE_AGENT);
        T r;
        __builtin_memcpy(&r, &u, 4);
        return r;
    } else {
        const unsigned long long u = __hip_atomic_load(reinterpret_cast<const unsigned long long*>(p), __ATOMIC_RELAXED, __HIP_MEMORY_SCOPE_AGENT);
        T r;
        __builtin_memcpy(&r, &u, 8);
        return r;
    }
}
