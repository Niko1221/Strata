#pragma once
// cuBLAS subset used by src/prefill/gemm.cu, on oneMKL SYCL BLAS.
#include "cuda_runtime.h"
#include <oneapi/mkl/blas.hpp>

enum cublasStatus_t { CUBLAS_STATUS_SUCCESS = 0, CUBLAS_STATUS_NOT_INITIALIZED = 1, CUBLAS_STATUS_INVALID_VALUE = 7, CUBLAS_STATUS_EXECUTION_FAILED = 13 };
enum cublasOperation_t { CUBLAS_OP_N = 0, CUBLAS_OP_T = 1, CUBLAS_OP_C = 2 };
enum cudaDataType_t { CUDA_R_16F = 2, CUDA_R_32F = 0, CUDA_R_16BF = 14 };
enum cublasComputeType_t { CUBLAS_COMPUTE_32F = 68 };
enum cublasGemmAlgo_t { CUBLAS_GEMM_DEFAULT = -1 };
constexpr int CUBLAS_DEFAULT_MATH = 0;

struct cublasContext {
    sycl::queue* q = nullptr;
};
using cublasHandle_t = cublasContext*;

inline cublasStatus_t cublasCreate(cublasHandle_t* handle) {
    auto* h = new cublasContext();
    h->q = &strata::xpu::default_queue();
    *handle = h;
    return CUBLAS_STATUS_SUCCESS;
}
inline cublasStatus_t cublasDestroy(cublasHandle_t handle) { delete handle; return CUBLAS_STATUS_SUCCESS; }
inline cublasStatus_t cublasSetStream(cublasHandle_t handle, cudaStream_t stream) {
    if (!handle) return CUBLAS_STATUS_NOT_INITIALIZED;
    handle->q = &strata::xpu::as_stream(stream)->q;
    return CUBLAS_STATUS_SUCCESS;
}
inline cublasStatus_t cublasSetWorkspace(cublasHandle_t, void*, size_t) { return CUBLAS_STATUS_SUCCESS; }
inline cublasStatus_t cublasSetMathMode(cublasHandle_t, int) { return CUBLAS_STATUS_SUCCESS; }

inline oneapi::mkl::transpose to_trans(cublasOperation_t op) {
    return op == CUBLAS_OP_N ? oneapi::mkl::transpose::nontrans : oneapi::mkl::transpose::trans;
}

inline cublasStatus_t cublasGemmEx(cublasHandle_t handle, cublasOperation_t transa, cublasOperation_t transb,
                                   int m, int n, int k, const void* alpha, const void* A, cudaDataType_t Atype, int lda,
                                   const void* B, cudaDataType_t Btype, int ldb, const void* beta, void* C,
                                   cudaDataType_t Ctype, int ldc, cublasComputeType_t, cublasGemmAlgo_t) {
    if (!handle || !handle->q) return CUBLAS_STATUS_NOT_INITIALIZED;
    if (Atype != Btype || Ctype != CUDA_R_32F) return CUBLAS_STATUS_INVALID_VALUE;
    try {
        auto& q = *handle->q;
        const float a = *static_cast<const float*>(alpha);
        const float b = *static_cast<const float*>(beta);
        auto ta = to_trans(transa);
        auto tb = to_trans(transb);
        if (Atype == CUDA_R_16BF) {
            oneapi::mkl::blas::column_major::gemm(
                q, ta, tb, m, n, k, a,
                static_cast<const sycl::ext::oneapi::bfloat16*>(A), lda,
                static_cast<const sycl::ext::oneapi::bfloat16*>(B), ldb,
                b, static_cast<float*>(C), ldc);
        } else if (Atype == CUDA_R_16F) {
            oneapi::mkl::blas::column_major::gemm(
                q, ta, tb, m, n, k, a,
                static_cast<const sycl::half*>(A), lda,
                static_cast<const sycl::half*>(B), ldb,
                b, static_cast<float*>(C), ldc);
        } else if (Atype == CUDA_R_32F) {
            oneapi::mkl::blas::column_major::gemm(
                q, ta, tb, m, n, k, a,
                static_cast<const float*>(A), lda,
                static_cast<const float*>(B), ldb,
                b, static_cast<float*>(C), ldc);
        } else {
            return CUBLAS_STATUS_INVALID_VALUE;
        }
        return CUBLAS_STATUS_SUCCESS;
    } catch (...) {
        return CUBLAS_STATUS_EXECUTION_FAILED;
    }
}
