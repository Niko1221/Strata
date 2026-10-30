#ifndef __HIP_PLATFORM_AMD__
#define __HIP_PLATFORM_AMD__ 1
#endif
#include "hip_api.h"

#include <hip/hip_runtime_api.h>
#ifndef STRATA_HYBRID_NO_BLAS
#include <hipblas/hipblas.h>
#endif

#include <cstdio>
#include <cstring>

static int E(hipError_t e) { return (int) e; }

int hh_device_count(void) {
    int n = 0;
    if (hipGetDeviceCount(&n) != hipSuccess || n < 0) {
        (void) hipGetLastError();
        return 0;
    }
    return n;
}

int hh_devices(HhDevice* out, int cap) {
    int n = hh_device_count();
    if (n > cap) n = cap;
    for (int i = 0; i < n; ++i) {
        hipDeviceProp_t p{};
        std::memset(&out[i], 0, sizeof out[i]);
        if (hipGetDeviceProperties(&p, i) != hipSuccess) continue;
        std::snprintf(out[i].name, sizeof out[i].name, "%s", p.name);
        std::snprintf(out[i].arch, sizeof out[i].arch, "%s", p.gcnArchName);
        char* colon = std::strchr(out[i].arch, ':');
        if (colon != nullptr) *colon = 0;
        out[i].total_mem = p.totalGlobalMem;
        out[i].shared_mem = p.sharedMemPerBlock;
        out[i].warp = p.warpSize;
        out[i].sms = p.multiProcessorCount;
        out[i].max_threads = p.maxThreadsPerBlock;
        out[i].clock_khz = p.clockRate;
        out[i].l2 = (int) p.l2CacheSize;
        out[i].integrated = p.integrated;
        out[i].pci_bus = p.pciBusID;
        out[i].pci_dev = p.pciDeviceID;
    }
    return n;
}

int hh_get_device(int* index) {
    if (index == nullptr) return 1;
    return E(hipGetDevice(index));
}

int hh_set_device(int index) {
    if (index < 0) return 1;
    return E(hipSetDevice(index));
}
int hh_set_device_flags(unsigned flags) {
    // A context that already exists returns hipErrorSetOnActiveProcess. The
    // flags are a hint; MapHost is already the ROCm default.
    if (hipSetDeviceFlags(flags) != hipSuccess) (void) hipGetLastError();
    return 0;
}
int hh_sync(void) { return E(hipDeviceSynchronize()); }
int hh_malloc(void** p, size_t n) { return E(hipMalloc(p, n)); }
int hh_free(void* p) { return E(hipFree(p)); }
int hh_host_malloc(void** p, size_t n, unsigned flags) { return E(hipHostMalloc(p, n, flags)); }
int hh_host_free(void* p) { return E(hipFreeHost(p)); }
int hh_host_register(void* p, size_t n, unsigned flags) { return E(hipHostRegister(p, n, flags)); }
int hh_host_unregister(void* p) { return E(hipHostUnregister(p)); }
int hh_host_device_pointer(void** dev, void* host, unsigned flags) { return E(hipHostGetDevicePointer(dev, host, flags)); }
int hh_memcpy(void* dst, const void* src, size_t n, int kind) {
    return E(hipMemcpy(dst, src, n, (hipMemcpyKind) kind));
}
int hh_memcpy_async(void* dst, const void* src, size_t n, int kind, void* stream) {
    return E(hipMemcpyAsync(dst, src, n, (hipMemcpyKind) kind, (hipStream_t) stream));
}
int hh_memcpy_2d_async(void* dst, size_t dp, const void* src, size_t sp, size_t w, size_t h, int kind, void* stream) {
    return E(hipMemcpy2DAsync(dst, dp, src, sp, w, h, (hipMemcpyKind) kind, (hipStream_t) stream));
}
int hh_memset(void* p, int v, size_t n) { return E(hipMemset(p, v, n)); }
int hh_memset_async(void* p, int v, size_t n, void* stream) { return E(hipMemsetAsync(p, v, n, (hipStream_t) stream)); }
int hh_mem_info(size_t* free_b, size_t* total_b) { return E(hipMemGetInfo(free_b, total_b)); }

int hh_launch(void* fn, unsigned gx, unsigned gy, unsigned gz, unsigned bx, unsigned by, unsigned bz, void** args,
              size_t shared, void* stream) {
    hipError_t e = hipLaunchKernel(fn, dim3(gx, gy, gz), dim3(bx, by, bz), args, shared, (hipStream_t) stream);
    if (e != hipSuccess)
        std::fprintf(stderr, "strata hybrid: hipLaunchKernel %p -> %s\n", fn, hipGetErrorString(e));
    return E(e);
}
int hh_launch_host(void* stream, void (*fn)(void*), void* data) {
    return E(hipLaunchHostFunc((hipStream_t) stream, (hipHostFn_t) fn, data));
}
int hh_stream_create(void** s, unsigned flags, int priority, int with_priority) {
    hipStream_t stream = nullptr;
    hipError_t e = with_priority ? hipStreamCreateWithPriority(&stream, flags, priority)
                                 : (flags ? hipStreamCreateWithFlags(&stream, flags) : hipStreamCreate(&stream));
    if (e == hipSuccess && s != nullptr) *s = stream;
    return E(e);
}
int hh_stream_destroy(void* s) { return E(hipStreamDestroy((hipStream_t) s)); }
int hh_stream_sync(void* s) { return E(hipStreamSynchronize((hipStream_t) s)); }
int hh_stream_query(void* s) { return E(hipStreamQuery((hipStream_t) s)); }
int hh_stream_begin_capture(void* s, int mode) {
    return E(hipStreamBeginCapture((hipStream_t) s, (hipStreamCaptureMode) mode));
}
int hh_stream_end_capture(void* s, void** graph) {
    return E(hipStreamEndCapture((hipStream_t) s, (hipGraph_t*) graph));
}
int hh_stream_is_capturing(void* s, int* status) {
    return E(hipStreamIsCapturing((hipStream_t) s, (hipStreamCaptureStatus*) status));
}
int hh_stream_capture_info(void* s, int* status, unsigned long long* id, void** graph, const void*** deps, size_t* ndeps) {
    return E(hipStreamGetCaptureInfo_v2((hipStream_t) s, (hipStreamCaptureStatus*) status, id, (hipGraph_t*) graph,
                                        (const hipGraphNode_t**) deps, ndeps));
}
int hh_stream_wait_event(void* s, void* event, unsigned flags) {
    return E(hipStreamWaitEvent((hipStream_t) s, (hipEvent_t) event, flags));
}
int hh_event_create(void** ev, unsigned flags, int with_flags) {
    hipEvent_t e = nullptr;
    hipError_t err = with_flags ? hipEventCreateWithFlags(&e, flags) : hipEventCreate(&e);
    if (err == hipSuccess && ev != nullptr) *ev = e;
    return E(err);
}
int hh_event_destroy(void* ev) { return E(hipEventDestroy((hipEvent_t) ev)); }
int hh_event_record(void* ev, void* s) { return E(hipEventRecord((hipEvent_t) ev, (hipStream_t) s)); }
int hh_event_sync(void* ev) { return E(hipEventSynchronize((hipEvent_t) ev)); }
int hh_event_query(void* ev) { return E(hipEventQuery((hipEvent_t) ev)); }
int hh_event_elapsed(float* ms, void* a, void* b) { return E(hipEventElapsedTime(ms, (hipEvent_t) a, (hipEvent_t) b)); }
int hh_graph_destroy(void* g) { return E(hipGraphDestroy((hipGraph_t) g)); }
int hh_graph_exec_destroy(void* g) { return E(hipGraphExecDestroy((hipGraphExec_t) g)); }
int hh_graph_instantiate(void** exec, void* graph, unsigned long long flags) {
    return E(hipGraphInstantiateWithFlags((hipGraphExec_t*) exec, (hipGraph_t) graph, flags));
}
int hh_graph_launch(void* exec, void* s) { return E(hipGraphLaunch((hipGraphExec_t) exec, (hipStream_t) s)); }
int hh_graph_upload(void* exec, void* s) { return E(hipGraphUpload((hipGraphExec_t) exec, (hipStream_t) s)); }
int hh_graph_nodes(void* g, void** nodes, size_t* n) {
    return E(hipGraphGetNodes((hipGraph_t) g, (hipGraphNode_t*) nodes, n));
}
int hh_graph_edges(void* g, void** from, void** to, size_t* n) {
    return E(hipGraphGetEdges((hipGraph_t) g, (hipGraphNode_t*) from, (hipGraphNode_t*) to, n));
}
int hh_graph_node_type(void* node, int* type) {
    return E(hipGraphNodeGetType((hipGraphNode_t) node, (hipGraphNodeType*) type));
}
int hh_graph_kernel_params(void* node, void** fn, unsigned* gx, unsigned* gy, unsigned* gz, unsigned* bx, unsigned* by,
                           unsigned* bz, unsigned* shared, void*** kparams, void*** extra) {
    hipKernelNodeParams p{};
    hipError_t e = hipGraphKernelNodeGetParams((hipGraphNode_t) node, &p);
    if (e != hipSuccess) return E(e);
    if (fn) *fn = p.func;
    if (gx) *gx = p.gridDim.x;
    if (gy) *gy = p.gridDim.y;
    if (gz) *gz = p.gridDim.z;
    if (bx) *bx = p.blockDim.x;
    if (by) *by = p.blockDim.y;
    if (bz) *bz = p.blockDim.z;
    if (shared) *shared = p.sharedMemBytes;
    if (kparams) *kparams = p.kernelParams;
    if (extra) *extra = p.extra;
    return 0;
}
int hh_func_attrs(void* fn, size_t* shared, size_t* cnst, size_t* local, int* max_threads, int* regs, int* ptx, int* binary,
                  int* cache_ca, int* max_dynamic, int* carveout) {
    hipFuncAttributes a{};
    hipError_t e = hipFuncGetAttributes(&a, fn);
    if (e != hipSuccess) return E(e);
    if (shared) *shared = a.sharedSizeBytes;
    if (cnst) *cnst = a.constSizeBytes;
    if (local) *local = a.localSizeBytes;
    if (max_threads) *max_threads = a.maxThreadsPerBlock;
    if (regs) *regs = a.numRegs;
    if (ptx) *ptx = a.ptxVersion;
    if (binary) *binary = a.binaryVersion;
    if (cache_ca) *cache_ca = a.cacheModeCA;
    if (max_dynamic) *max_dynamic = a.maxDynamicSharedSizeBytes;
    if (carveout) *carveout = a.preferredShmemCarveout;
    return 0;
}
int hh_func_set_attr(void* fn, int attr, int value) {
    return E(hipFuncSetAttribute(fn, (hipFuncAttribute) attr, value));
}
int hh_symbol_address(void** dev, const void* symbol) { return E(hipGetSymbolAddress(dev, symbol)); }
// CUDA cudaDeviceAttr and HIP hipDeviceAttribute_t use different numbers.
// The live host is nvcc, so it passes the CUDA numbers. Casting one onto
// the other makes hipDeviceGetAttribute fail (CUDA 97) or answer a
// different question (CUDA 16 is HIP's Integrated, not the SM count).
static bool cuda_attr_to_hip(int cuda_attr, hipDeviceAttribute_t* out) {
    switch (cuda_attr) {
    case 1: *out = hipDeviceAttributeMaxThreadsPerBlock; return true;             // MaxThreadsPerBlock
    case 2: *out = hipDeviceAttributeMaxBlockDimX; return true;                   // MaxBlockDimX
    case 3: *out = hipDeviceAttributeMaxBlockDimY; return true;                   // MaxBlockDimY
    case 4: *out = hipDeviceAttributeMaxBlockDimZ; return true;                   // MaxBlockDimZ
    case 5: *out = hipDeviceAttributeMaxGridDimX; return true;                    // MaxGridDimX
    case 6: *out = hipDeviceAttributeMaxGridDimY; return true;                    // MaxGridDimY
    case 7: *out = hipDeviceAttributeMaxGridDimZ; return true;                    // MaxGridDimZ
    case 8: *out = hipDeviceAttributeMaxSharedMemoryPerBlock; return true;        // MaxSharedMemoryPerBlock
    case 9: *out = hipDeviceAttributeTotalConstantMemory; return true;            // TotalConstantMemory
    case 10: *out = hipDeviceAttributeWarpSize; return true;                      // WarpSize
    case 12: *out = hipDeviceAttributeMaxRegistersPerBlock; return true;          // MaxRegistersPerBlock
    case 13: *out = hipDeviceAttributeClockRate; return true;                     // ClockRate
    case 16: *out = hipDeviceAttributeMultiprocessorCount; return true;           // MultiProcessorCount
    case 17: *out = hipDeviceAttributeKernelExecTimeout; return true;             // KernelExecTimeout
    case 18: *out = hipDeviceAttributeIntegrated; return true;                    // Integrated
    case 19: *out = hipDeviceAttributeCanMapHostMemory; return true;              // CanMapHostMemory
    case 20: *out = hipDeviceAttributeComputeMode; return true;                   // ComputeMode
    case 36: *out = hipDeviceAttributeMemoryClockRate; return true;               // MemoryClockRate
    case 37: *out = hipDeviceAttributeMemoryBusWidth; return true;                // GlobalMemoryBusWidth
    case 38: *out = hipDeviceAttributeL2CacheSize; return true;                   // L2CacheSize
    case 39: *out = hipDeviceAttributeMaxThreadsPerMultiProcessor; return true;   // MaxThreadsPerMultiProcessor
    case 40: *out = hipDeviceAttributeAsyncEngineCount; return true;              // AsyncEngineCount
    case 81: *out = hipDeviceAttributeSharedMemPerMultiprocessor; return true;    // MaxSharedMemoryPerMultiprocessor
    case 82: *out = hipDeviceAttributeMaxRegistersPerMultiprocessor; return true; // MaxRegistersPerMultiprocessor
    case 95: *out = hipDeviceAttributeCooperativeLaunch; return true;             // CooperativeLaunch
    case 97: *out = hipDeviceAttributeSharedMemPerBlockOptin; return true;        // MaxSharedMemoryPerBlockOptin
    default: return false;
    }
}
int hh_device_attr(int* value, int attr, int index) {
    if (value == nullptr) return 1;
    // CUDA 75/76 are compute capability. ROCm answers with the ISA version.
    if (attr == 75 || attr == 76) {
        *value = 0;
        return 0;
    }
    hipDeviceAttribute_t hip_attr{};
    if (!cuda_attr_to_hip(attr, &hip_attr)) return 1;
    return E(hipDeviceGetAttribute(value, hip_attr, index));
}
int hh_stream_priority_range(int* least, int* greatest) { return E(hipDeviceGetStreamPriorityRange(least, greatest)); }
int hh_driver_version(int* v) { return E(hipDriverGetVersion(v)); }
int hh_runtime_version(int* v) { return E(hipRuntimeGetVersion(v)); }
int hh_last_error(void) { return E(hipGetLastError()); }
int hh_peek_error(void) { return E(hipPeekAtLastError()); }
const char* hh_error_string(int e) { return hipGetErrorString((hipError_t) e); }
int hh_occupancy(int* blocks, void* fn, int block, size_t smem, unsigned flags) {
    return E(hipOccupancyMaxActiveBlocksPerMultiprocessorWithFlags(blocks, fn, block, smem, flags));
}

#ifndef STRATA_HYBRID_NO_BLAS
// The ints returned below are cublasStatus_t. hipBLAS numbers the same names differently.
// CUDA ops are 0/1/2. Anything else, including conjugate (3), is not a hipBLAS op.
static bool hip_op(int op, hipblasOperation_t* out) {
    switch (op) {
    case 0: *out = HIPBLAS_OP_N; return true;
    case 1: *out = HIPBLAS_OP_T; return true;
    case 2: *out = HIPBLAS_OP_C; return true;
    default: return false;
    }
}
// cuBLAS 4 is BF16x9 emulation and 8 is FP64 fixed-point emulation. hipBLAS has
// neither. 16 is a flag or-ed onto a mode; hipBLAS can say disallow-reduction
// only as its own mode, so a combined value fails instead of becoming default.
static bool hip_math(int mode, hipblasMath_t* out) {
    switch (mode) {
    case 0: *out = HIPBLAS_DEFAULT_MATH; return true;
    case 1: *out = HIPBLAS_TENSOR_OP_MATH; return true;
    case 2: *out = HIPBLAS_PEDANTIC_MATH; return true;
    case 3: *out = HIPBLAS_TF32_TENSOR_OP_MATH; return true;
    case 16: *out = HIPBLAS_MATH_DISALLOW_REDUCED_PRECISION_REDUCTION; return true;
    default: return false;
    }
}
// cuBLAS compute types start at 64 (cublas_api.h). hipBLAS numbers the same names from 0
// (hipblas-common.h, 7.2.4: HIPBLAS_COMPUTE_32F is 2, CUBLAS_COMPUTE_32F is 68).
static bool hip_compute(int compute, hipblasComputeType_t* out) {
    switch (compute) {
    case 64: *out = HIPBLAS_COMPUTE_16F; return true;
    case 65: *out = HIPBLAS_COMPUTE_16F_PEDANTIC; return true;
    case 68: *out = HIPBLAS_COMPUTE_32F; return true;
    case 69: *out = HIPBLAS_COMPUTE_32F_PEDANTIC; return true;
    case 70: *out = HIPBLAS_COMPUTE_64F; return true;
    case 71: *out = HIPBLAS_COMPUTE_64F_PEDANTIC; return true;
    case 72: *out = HIPBLAS_COMPUTE_32I; return true;
    case 73: *out = HIPBLAS_COMPUTE_32I_PEDANTIC; return true;
    case 74: *out = HIPBLAS_COMPUTE_32F_FAST_16F; return true;
    case 75: *out = HIPBLAS_COMPUTE_32F_FAST_16BF; return true;
    case 77: *out = HIPBLAS_COMPUTE_32F_FAST_TF32; return true;
    default: return false;
    }
}
static int cublas_status(hipblasStatus_t s) {
    switch (s) {
    case HIPBLAS_STATUS_SUCCESS: return 0;
    case HIPBLAS_STATUS_NOT_INITIALIZED: return 1;
    case HIPBLAS_STATUS_ALLOC_FAILED: return 3;
    case HIPBLAS_STATUS_INVALID_VALUE: return 7;
    case HIPBLAS_STATUS_ARCH_MISMATCH: return 8;
    case HIPBLAS_STATUS_MAPPING_ERROR: return 11;
    case HIPBLAS_STATUS_EXECUTION_FAILED: return 13;
    case HIPBLAS_STATUS_INTERNAL_ERROR: return 14;
    case HIPBLAS_STATUS_NOT_SUPPORTED: return 15;
    default: return 7;
    }
}
int hh_blas_create(void** h) { return cublas_status(hipblasCreate((hipblasHandle_t*) h)); }
int hh_blas_destroy(void* h) { return cublas_status(hipblasDestroy((hipblasHandle_t) h)); }
int hh_blas_set_stream(void* h, void* s) { return cublas_status(hipblasSetStream((hipblasHandle_t) h, (hipStream_t) s)); }
int hh_blas_set_math(void* h, int mode) {
    hipblasMath_t math{};
    if (!hip_math(mode, &math)) return 7;
    return cublas_status(hipblasSetMathMode((hipblasHandle_t) h, math));
}
int hh_blas_set_workspace(void* h, void* workspace, size_t bytes) {
    return cublas_status(hipblasSetWorkspace((hipblasHandle_t) h, workspace, bytes));
}
int hh_blas_sgemm(void* h, int ta, int tb, int m, int n, int k, const float* alpha, const float* A, int lda, const float* B,
                  int ldb, const float* beta, float* C, int ldc) {
    hipblasOperation_t oa, ob;
    if (!hip_op(ta, &oa) || !hip_op(tb, &ob)) return 7;
    return cublas_status(hipblasSgemm((hipblasHandle_t) h, oa, ob, m, n, k, alpha, A, lda, B, ldb, beta, C, ldc));
}
int hh_blas_gemm_ex(void* h, int ta, int tb, int m, int n, int k, const void* alpha, const void* A, int at, int lda,
                    const void* B, int bt, int ldb, const void* beta, void* C, int ct, int ldc, int compute, int algo) {
    (void) algo;
    hipblasOperation_t oa, ob;
    hipblasComputeType_t ctype{};
    if (!hip_op(ta, &oa) || !hip_op(tb, &ob) || !hip_compute(compute, &ctype)) return 7;
    return cublas_status(hipblasGemmEx((hipblasHandle_t) h, oa, ob, m, n, k, alpha, A, (hipDataType) at, lda, B,
                                       (hipDataType) bt, ldb, beta, C, (hipDataType) ct, ldc, ctype, HIPBLAS_GEMM_DEFAULT));
}
int hh_blas_gemm_strided(void* h, int ta, int tb, int m, int n, int k, const void* alpha, const void* A, int at, int lda,
                         long long stride_a, const void* B, int bt, int ldb, long long stride_b, const void* beta, void* C,
                         int ct, int ldc, long long stride_c, int batch, int compute, int algo) {
    (void) algo;
    hipblasOperation_t oa, ob;
    hipblasComputeType_t ctype{};
    if (!hip_op(ta, &oa) || !hip_op(tb, &ob) || !hip_compute(compute, &ctype)) return 7;
    return cublas_status(hipblasGemmStridedBatchedEx(
        (hipblasHandle_t) h, oa, ob, m, n, k, alpha, A, (hipDataType) at, lda, stride_a, B, (hipDataType) bt, ldb,
        stride_b, beta, C, (hipDataType) ct, ldc, stride_c, batch, ctype, HIPBLAS_GEMM_DEFAULT));
}
#endif
