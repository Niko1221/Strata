#define DPCT_COMPAT_RT_VERSION 12090
#pragma once

#define DPCT_PROFILING_ENABLED
#include <sycl/sycl.hpp>
#include <dpct/dpct.hpp>
#include <dpct/blas_utils.hpp>

#ifdef GGML_USE_NCCL
#include <nccl.h>
#endif // GGML_USE_NCCL

#if DPCT_COMPAT_RT_VERSION >= 11080
// Native CUDA FP8 types are not used by this SYCL port.
#endif // CUDART_VERSION >= 11080

#if DPCT_COMPAT_RT_VERSION >= 12080
#endif // CUDART_VERSION >= 12080

#if DPCT_COMPAT_RT_VERSION < 11020
#define CU_DEVICE_ATTRIBUTE_VIRTUAL_MEMORY_MANAGEMENT_SUPPORTED CU_DEVICE_ATTRIBUTE_VIRTUAL_ADDRESS_MANAGEMENT_SUPPORTED
#define CUBLAS_TF32_TENSOR_OP_MATH CUBLAS_TENSOR_OP_MATH
#define CUBLAS_COMPUTE_16F CUDA_R_16F
#define CUBLAS_COMPUTE_32F CUDA_R_32F
#define cublasComputeType_t cudaDataType_t
#endif // CUDART_VERSION < 11020
