#pragma once
#include "strata/qwen38/quant_tensor.hpp"
#include <cuda_fp16.h>
#include <cuda_runtime.h>
#include <cstddef>

namespace strata::qwen38 {

/// Bounded double-buffered workspace for polymorphic dequantization + cuBLAS Tensor Core GEMM.
class PrefillGemmWorkspace {
public:
    /// Default 2 x 32 MiB = 64 MiB dequant weight ping-pong buffer + activation FP16 staging.
    static constexpr size_t kDefaultTileBufBytes = 32ull * 1024 * 1024;

    PrefillGemmWorkspace() = default;
    ~PrefillGemmWorkspace();
    PrefillGemmWorkspace(const PrefillGemmWorkspace&) = delete;
    PrefillGemmWorkspace& operator=(const PrefillGemmWorkspace&) = delete;

    /// Allocate ping-pong weight buffers and FP16 activation staging (in bytes).
    void init(int max_tokens, int max_k, size_t tile_buf_bytes = kDefaultTileBufBytes);
    void release();

    /// Total GPU bytes allocated by this workspace.
    size_t allocated_bytes() const { return total_bytes_; }

    /// Polymorphic batched matrix multiply: Y[n_tokens, rows] = X[n_tokens, n] * W[rows, n]^T.
    /// Automatically dispatches to cuBLAS Tensor Core GEMM for n_tokens >= 8 (and matvec for smaller).
    void matmul_batch(const StrataQuantTensor& weight, const float* x_fp32, float* y_fp32,
                      int n_tokens, cudaStream_t stream);

private:
    struct Impl;
    Impl* impl_ = nullptr;
    size_t total_bytes_ = 0;
};

/// Expose tiled FP16 dequantization for direct unit testing.
void dequant_rows_fp16(const StrataQuantTensor& weight, int row_start, int tile_rows,
                       half* dst_fp16, cudaStream_t stream);

} // namespace strata::qwen38
