// Compile the fused hyper-connection kernel with the arithmetic flags of its reference kernels.
#include "strata/hy4flash_gguf/ihc.hpp"

#include "strata/kernels/elementwise.hpp"
#include "strata/kernels/native_mmvq.hpp"
#include "strata/kernels/pdl.hpp"
#include "strata/kernels/q8_1_finite.hpp"

#include <cuda_fp16.h>
#include <cstdlib>
#include <stdexcept>
#include <string>

namespace strata::hy4_flash {
namespace {

using strata::kernels::launch_pdl;
using strata::kernels::pdl_trigger;
using strata::kernels::pdl_wait;
using strata::kernels::q8_1_ds;
using strata::kernels::q8_1_finite;
using strata::kernels::q8_1_quant;

void launched(const char* what) {
    const auto e = cudaGetLastError();
    if (e != cudaSuccess) throw std::runtime_error(std::string(what) + ": " + cudaGetErrorString(e));
}

/// The quantiser's 32-lane reductions (must stay identical to native_mmvq.cu's; the fused kernel is
/// bitwise tested against the split path, so any drift shows up there).
__device__ __forceinline__ float hc_warp_sum(float x) {
#pragma unroll
    for (int offset = 32 / 2; offset > 0; offset >>= 1) {
        x += __shfl_xor_sync(0xffffffff, x, offset, 32);
    }
    return x;
}

__device__ __forceinline__ float hc_warp_max(float x) {
#pragma unroll
    for (int offset = 32 / 2; offset > 0; offset >>= 1) {
        x = fmaxf(x, __shfl_xor_sync(0xffffffff, x, offset, 32));
    }
    return x;
}

/// Compute the flat RMS and Q8_1 activation using the unfused path's reduction order.
template <int kThreads>
__device__ __forceinline__ float hc_rms_c50_block_sum(float v, float* shared) {
#pragma unroll
    for (int o = 16; o > 0; o >>= 1) v += __shfl_xor_sync(0xffffffffu, v, o, 32);
    if constexpr (kThreads > 32) {
        const int warp_id = (int) threadIdx.x / 32, lane_id = (int) threadIdx.x % 32;
        if (lane_id == 0) shared[warp_id] = v;
        __syncthreads();
        v = 0.0f;
        if (lane_id < kThreads / 32) v = shared[lane_id];
#pragma unroll
        for (int o = 16; o > 0; o >>= 1) v += __shfl_xor_sync(0xffffffffu, v, o, 32);
    }
    return v;
}

struct HcQ81Block { half2 ds; int8_t qs[32]; };
static_assert(sizeof(HcQ81Block) == 36 && alignof(HcQ81Block) == 4 && offsetof(HcQ81Block, qs) == 4,
              "the Q8_1 block layout the quantiser writes");

/// Each block normalizes and quantizes a slice of 32 Q8_1 blocks.
template <int kVpt>
__global__ void hc_rms_quant_fused_kernel(const float* x_, HcQ81Block* q8_, int hc_width, int parts,
                                          float eps) {
    pdl_trigger();
    pdl_wait();
    const int tid = (int) threadIdx.x;
    const int row = (int) (blockIdx.x / parts);
    const int part = (int) (blockIdx.x % parts);
    const float4* x4 = (const float4*) (x_ + (long long) row * hc_width);
    const int ncols4 = hc_width / 4;
    __shared__ float s_sum[32];
    float4 xs[kVpt];
    float tmp = 0.0f;
#pragma unroll
    for (int k = 0; k < kVpt; ++k) {
        const int i = tid + k * 1024;
        if (i < ncols4) {
            xs[k] = x4[i];
            tmp += xs[k].x * xs[k].x + xs[k].y * xs[k].y + xs[k].z * xs[k].z + xs[k].w * xs[k].w;
        }
    }
    tmp = hc_rms_c50_block_sum<1024>(tmp, s_sum);
    const float scale = rsqrtf(__fdividef(tmp, (float) hc_width) + eps);
    __syncthreads();
    // This block's 32 Q8_1 blocks, one a warp, from xi = x * scale.
    const int warp = tid >> 5, lane = tid & 31;
    const int b = part * 32 + warp;
    const float xi = __fmul_rn(x_[row * hc_width + b * 32 + lane], scale);   // the norm's exact value
    const float amax = hc_warp_max(fabsf(xi));
    const float sum = hc_warp_sum(xi);
    const float d = q8_1_finite(amax / 127.0f);
    const int8_t q = q8_1_quant(xi, d, amax);
    HcQ81Block* qr = q8_ + (long long) row * (hc_width / 32);
    qr[b].qs[lane] = q;
    if (lane == 0) qr[b].ds = q8_1_ds(d, sum);
}

}  // namespace

bool hc_fuse_on() {
    static const bool on = [] {
        const char* v = std::getenv("STRATA_HY4_FLASH_HC_FUSE");
        return v != nullptr && std::atoi(v) != 0;
    }();
    return on;
}

void hc_pre_rms_quant_fused(const float* x, void* q8, int hc_width, int n_tok, float eps, void* stream) {
    // Only where the split path would take rms_norm_block's 1024-thread vector form - the butterfly in
    // the fused kernel is that kernel's, and every other width would be a different reduction - and only
    // whole 1024-element slices, one a block: a remainder would be left unquantised.
    const int n4 = hc_width / 4;
    if (n4 <= 1024 || n4 > 4096 || hc_width % 1024 != 0)
        throw std::invalid_argument("hc_pre_rms_quant_fused: 4096 < hc_width <= 16384 and hc_width % 1024 == 0");
    auto* q8b = static_cast<HcQ81Block*>(q8);
    const auto s = static_cast<cudaStream_t>(stream);
    const int parts = hc_width / 32 / 32;   // 32 Q8_1 blocks (one a warp) a 1024-thread block
    const unsigned grid = (unsigned) (n_tok * parts);
    if (n4 <= 2048) launch_pdl(hc_rms_quant_fused_kernel<2>, grid, 1024, 0, s, x, q8b, hc_width, parts, eps);
    else if (n4 <= 3072) launch_pdl(hc_rms_quant_fused_kernel<3>, grid, 1024, 0, s, x, q8b, hc_width, parts, eps);
    else launch_pdl(hc_rms_quant_fused_kernel<4>, grid, 1024, 0, s, x, q8b, hc_width, parts, eps);
    launched("hc_pre fused rms+quantise");
}

}  // namespace strata::hy4_flash
