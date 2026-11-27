// cublas_v2.h for a STRATA_ENABLE_HIP build: the cuBLAS names src/prefill/gemm.cu uses, on hipBLAS.
#pragma once
#include "strata_hip.h"
#ifndef HIPBLAS_V2
#define HIPBLAS_V2
#endif
#include <hipblas/hipblas.h>

#define cublasHandle_t hipblasHandle_t
#define cublasStatus_t hipblasStatus_t
#define CUBLAS_STATUS_SUCCESS HIPBLAS_STATUS_SUCCESS
#define cublasCreate hipblasCreate
#define cublasDestroy hipblasDestroy
#define cublasSetStream hipblasSetStream
#define cublasGemmEx hipblasGemmEx
#define CUBLAS_OP_N HIPBLAS_OP_N
#define CUBLAS_OP_T HIPBLAS_OP_T
#define CUBLAS_GEMM_DEFAULT HIPBLAS_GEMM_DEFAULT
#if hipblasVersionMajor < 2
// strata-mi50 (2026-10-23): hipBLAS 1.x (HIP SDK 5.7, Windows gfx906 build) has no hipblasComputeType_t and its hipblasGemmEx
// takes hipblasDatatype_t for both the data and the compute type. Newer hipBLAS keeps the mapping below, unchanged.
#define CUBLAS_COMPUTE_32F HIPBLAS_R_32F
#define CUDA_R_32F HIPBLAS_R_32F
#define CUDA_R_16F HIPBLAS_R_16F
#define CUDA_R_16BF HIPBLAS_R_16B
#else
#define CUBLAS_COMPUTE_32F HIPBLAS_COMPUTE_32F
#define CUDA_R_32F HIP_R_32F
#define CUDA_R_16F HIP_R_16F
#define CUDA_R_16BF HIP_R_16BF
#endif
// rocBLAS picks its own math and workspace; the NVIDIA-only knobs are accepted and ignored
#define CUBLAS_DEFAULT_MATH 0
#define cublasSetMathMode(handle, mode) HIPBLAS_STATUS_SUCCESS
#define cublasSetWorkspace(handle, ptr, bytes) HIPBLAS_STATUS_SUCCESS
