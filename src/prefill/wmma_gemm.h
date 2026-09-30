// src/prefill/wmma_gemm.h - RDNA3 WMMA FP16 & BF16 GEMM for Strata prefill.
#pragma once

#include <cstdint>

/// Compute Y[t, n] = beta * Y[t, n] + sum_k W[n, k] * X[t, k] using RDNA3 WMMA instructions.
/// X: T x K row-major (leading dim K), fp16 (uint16_t)
/// W: N x K row-major (leading dim K), fp16 (uint16_t)
/// Y: T x N row-major with leading dimension ldy >= N, fp32 (float)
/// Returns true if executed on WMMA, false if unsupported shape/parameters (fall back to hipblas).
bool strata_wmma_gemm_f16(const uint16_t* X, const uint16_t* W, float* Y,
                          int64_t T, int64_t N, int64_t K, int64_t ldy, float beta,
                          void* stream = nullptr);

/// Compute Y[t, n] = beta * Y[t, n] + sum_k W[n, k] * X[t, k] using RDNA3 WMMA instructions.
/// X: T x K row-major (leading dim K), bf16 (uint16_t)
/// W: N x K row-major (leading dim K), bf16 (uint16_t)
/// Y: T x N row-major with leading dimension ldy >= N, fp32 (float)
/// Returns true if executed on WMMA, false if unsupported shape/parameters (fall back to hipblas).
bool strata_wmma_gemm_bf16(const uint16_t* X, const uint16_t* W, float* Y,
                           int64_t T, int64_t N, int64_t K, int64_t ldy, float beta,
                           void* stream = nullptr);
