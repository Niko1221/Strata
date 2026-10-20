// src/kernels/cuda/qns_gather.cu - see include/strata/kernels/qns_gather.hpp.
#include "strata/kernels/qns_gather.hpp"

namespace strata::kernels {
namespace {

__global__ void qns_gather_rows_kernel(const uint32_t* __restrict__ words, const float* __restrict__ scales, int W,
                                       const int32_t* __restrict__ rows, int n, uint32_t* __restrict__ dst,
                                       float* __restrict__ dst_scales) {
    const int i = blockIdx.x, r = i < n ? rows[i] : -1;
    for (int k = threadIdx.x; k < W; k += blockDim.x)
        dst[((size_t) (i / 32) * W + k) * 32 + i % 32] = r < 0 ? 0u : words[((size_t) (r / 32) * W + k) * 32 + r % 32];
    if (threadIdx.x == 0) dst_scales[i] = r < 0 ? 0.0f : scales[r];
}

}  // namespace

void qns_gather_rows(const uint32_t* words, const float* scales, int W, const int32_t* rows, int n, int n_pad,
                     uint32_t* dst_words, float* dst_scales, cudaStream_t s) {
    qns_gather_rows_kernel<<<n_pad, 128, 0, s>>>(words, scales, W, rows, n, dst_words, dst_scales);
}

}  // namespace strata::kernels
