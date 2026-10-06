// src/prefill/kernels.cu - see include/strata/prefill/kernels.hpp.
#include "strata/prefill/kernels.hpp"
#include "strata/kernels/mrope.hpp"
#include "strata/kernels/gfx_arch.hpp"
#include "strata/kernels/router_top10.hpp"

#include <cuda_fp16.h>
#include <cuda_runtime.h>
#include <algorithm>

#include <algorithm>
#include <cfloat>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>

namespace strata::prefill {
namespace {

constexpr int N = 2560, HC = 4, D = N * HC, LR = 320;
constexpr int S = 128, HK = 16, HV = 48, C = 10240;

__device__ __forceinline__ float warp_sum(float v) {
#pragma unroll
    for (int o = 16; o > 0; o >>= 1) v += __shfl_xor_sync(0xffffffffu, v, o);
    return v;
}
__device__ __forceinline__ float warp_max(float v) {
#pragma unroll
    for (int o = 16; o > 0; o >>= 1) v = fmaxf(v, __shfl_xor_sync(0xffffffffu, v, o));
    return v;
}
__device__ __forceinline__ uint16_t bf(float f) {
    uint32_t u = __float_as_uint(f);
    u += 0x7fffu + ((u >> 16) & 1u);
    return (uint16_t) (u >> 16);
}
// The BF16 GEMMs' second operand (STRATA_PREFILL_BF16X2): what the BF16 image `hi` left out of f, itself in BF16.
// W.hi + W.lo carries ~16 mantissa bits of the activation - the decode path's FP32 x to within ~1e-5.
__device__ __forceinline__ uint16_t bf_lo(float f, uint16_t hi) { return bf(f - __uint_as_float((uint32_t) hi << 16)); }
__device__ __forceinline__ float sigm(float x) { return 1.0f / (1.0f + __expf(-x)); }
__device__ __forceinline__ uint16_t hf(float f) { return __half_as_ushort(__float2half_rn(f)); }
// A SwiGLU product for an FP16 GEMM: saturated, so a token with a massive activation cannot turn into inf and then
// NaN in the down projection (decode's q8_1 has room to ~8e6; FP16 ends at 65504).  A NaN stays NaN (fminf/fmaxf
// would make it -65504 and hide where it came from); finite values below 65504 round exactly as before.
__device__ __forceinline__ uint16_t hf_sat(float f) { return hf(isnan(f) ? f : fminf(fmaxf(f, -65504.0f), 65504.0f)); }
// The prompt path's 16-bit activation image for the BF16-weight GEMMs: BF16, or FP16 where the GEMM library is fast
// only in FP16 (prompt_f16() in gemm.cu: rocBLAS on gfx103x).  Set once per device before the first prompt.
#if defined(__HIPCC__)   // HIP only (#835): the CUDA kernels stay exactly as they were, they never read the flag
__device__ int g_act_f16 = 0;
__device__ __forceinline__ uint16_t act16(float f) { return g_act_f16 ? hf_sat(f) : bf(f); }
#else
__device__ __forceinline__ uint16_t act16(float f) { return bf(f); }
#endif
// block-wide sum for blockDim.x <= 1024, result broadcast
__device__ float block_sum(float v, float* sh) {
    const int lane = threadIdx.x & 31, w = threadIdx.x >> 5;
    v = warp_sum(v);
    __syncthreads();
    if (lane == 0) sh[w] = v;
    __syncthreads();
    const int nw = (blockDim.x + 31) >> 5;
    float t = (threadIdx.x < nw) ? sh[threadIdx.x] : 0.0f;
    if (w == 0) t = warp_sum(t);
    if (threadIdx.x == 0) sh[0] = t;
    __syncthreads();
    return sh[0];
}
void check(const char* what) {
    const cudaError_t e = cudaGetLastError();
    if (e != cudaSuccess) { std::fprintf(stderr, "prefill %s: %s\n", what, cudaGetErrorString(e)); std::exit(1); }
}
unsigned blocks_for(int64_t n, int t = 256) { return (unsigned) ((n + t - 1) / t); }

// ---------------------------------------------------------------- hyper-connection
__global__ void gr_norm_kernel(const float* __restrict__ R, const float* __restrict__ w, float eps,
                               float* __restrict__ xn, uint16_t* __restrict__ xn16, uint16_t* __restrict__ xn16_lo) {
    __shared__ float sh[32];
    const int64_t row = blockIdx.x;                 // t * 4 + c
    const int c = (int) (row % HC);
    const float* r = R + row * N;
    float ss = 0.0f;
    for (int d = threadIdx.x; d < N; d += blockDim.x) ss += r[d] * r[d];
    const float rs = rsqrtf(block_sum(ss, sh) / (float) N + eps);
    for (int d = threadIdx.x; d < N; d += blockDim.x) {
        const float v = r[d] * rs * w[c * N + d];
        xn[row * N + d] = v;
        const uint16_t h = act16(v);
        xn16[row * N + d] = h;
        if (xn16_lo) xn16_lo[row * N + d] = bf_lo(v, h);
    }
}
// F-1: the row scale only (and the BF16 image); gr_mix_r_kernel recomputes r * rs * w itself, in the same order,
// so the FP32 copy of the normalized rows (T x 10240 floats) is neither written nor read
__global__ void gr_norm_rs_kernel(const float* __restrict__ R, const float* __restrict__ w, float eps,
                                  float* __restrict__ rs_out, uint16_t* __restrict__ xn16,
                                  uint16_t* __restrict__ xn16_lo, int64_t ldx) {
    __shared__ float sh[32];
    const int64_t row = blockIdx.x;                 // t * 4 + c
    const int c = (int) (row % HC);
    const float* r = R + row * N;
    const int64_t xo = (row / HC) * ldx + (int64_t) c * N;   // the BF16 image's row (token stride ldx)
    float ss = 0.0f;
    for (int d = threadIdx.x; d < N; d += blockDim.x) ss += r[d] * r[d];
    const float rs = rsqrtf(block_sum(ss, sh) / (float) N + eps);
    if (threadIdx.x == 0) rs_out[row] = rs;
    for (int d = threadIdx.x; d < N; d += blockDim.x) {
        const float v = r[d] * rs * w[c * N + d];
        const uint16_t h = act16(v);
        xn16[xo + d] = h;
        if (xn16_lo) xn16_lo[xo + d] = bf_lo(v, h);
    }
}
__global__ void gr_mix_r_kernel(const float* __restrict__ R, const float* __restrict__ rs, const float* __restrict__ w,
                                const float* __restrict__ g, float* __restrict__ mixed, uint16_t* __restrict__ mixed16,
                                int64_t T, uint16_t* __restrict__ mixed_h, uint16_t* __restrict__ mixed16_lo) {
    const int64_t i = (int64_t) blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= T * N) return;
    const int64_t t = i / N, d = i % N;
    float s = 0.0f;
#pragma unroll
    for (int c = 0; c < HC; ++c) {
        const int64_t j = t * D + c * N + d;
        const float x = R[j] * rs[t * HC + c] * w[c * N + d];   // gr_norm_kernel's value, bit for bit
        s = fmaf(x, sigm(g[j]), s);
    }
    s /= (float) HC;
    mixed[i] = s;
    if (mixed16) {
        const uint16_t h = act16(s);
        mixed16[i] = h;
        if (mixed16_lo) mixed16_lo[i] = bf_lo(s, h);
    }
    if (mixed_h) mixed_h[i] = hf(s);
}
// F-2: gr_write_kernel for one row (t, c), then gr_norm_rs_kernel's reduction over it with the next half's norm
// weights - the same thread-to-element mapping (256 threads, stride 256) and block_sum, so rs and the BF16 image are
// the same bits, and R is not read back
constexpr int GRW_PER = (N + 255) / 256;
__global__ void __launch_bounds__(256) gr_write_norm_rs_kernel(float* __restrict__ R, const float* __restrict__ bo,
                                                               const float* __restrict__ inj, int64_t inj_ld,
                                                               const float* __restrict__ w, float eps,
                                                               float* __restrict__ rs_out, uint16_t* __restrict__ xn16,
                                                               uint16_t* __restrict__ xn16_lo, int64_t ldx) {
    __shared__ float sh[32];
    const int64_t row = blockIdx.x;                 // t * 4 + c
    const int64_t t = row / HC;
    const int c = (int) (row % HC);
    float* r = R + row * N;
    const int64_t xo = t * ldx + (int64_t) c * N;
    const float sc = 2.0f * sigm(inj[t * inj_ld + c] / (float) HC);
    float v[GRW_PER];
    float ss = 0.0f;
    int k = 0;
#pragma unroll
    for (int d = threadIdx.x; d < N; d += 256, ++k) {
        const float x = fmaf(bo[t * N + d], sc, r[d]);
        r[d] = x;
        v[k] = x;
        ss += x * x;
    }
    const float rs = rsqrtf(block_sum(ss, sh) / (float) N + eps);
    if (threadIdx.x == 0) rs_out[row] = rs;
    k = 0;
#pragma unroll
    for (int d = threadIdx.x; d < N; d += 256, ++k) {
        const float x = v[k] * rs * w[c * N + d];
        const uint16_t h = act16(x);
        xn16[xo + d] = h;
        if (xn16_lo) xn16_lo[xo + d] = bf_lo(x, h);
    }
}
// S23 (opt-in STRATA_CVEC_FUSE=1): a steered layer's FFN write, its control vector and the next half's norm in one
// pass over R (was gr_write + cvec_apply + gr_norm_rs: three).  The write is gr_write_kernel's, the vector
// cvec_kernel's (the same per-thread dot order over d = tid + 256 k, the same two-level xor reduction), the norm
// gr_norm_rs_kernel's (the same ss order and block_sum): the same bits.
__global__ void __launch_bounds__(256) gr_write_cvec_norm_rs_kernel(
    float* __restrict__ R, const float* __restrict__ bo, const float* __restrict__ inj, int64_t inj_ld,
    const float* __restrict__ v_l, const float* __restrict__ s_l, const int* __restrict__ on, int mode,
    const float* __restrict__ w, float eps, float* __restrict__ rs_out, uint16_t* __restrict__ xn16,
    uint16_t* __restrict__ xn16_lo, int64_t ldx) {
    __shared__ float sh[32];
    __shared__ float part[8];
    const int64_t row = blockIdx.x;                 // t * 4 + c
    const int64_t t = row / HC;
    const int c = (int) (row % HC);
    float* r = R + row * N;
    const float sc = 2.0f * sigm(inj[t * inj_ld + c] / (float) HC);
    const float s = *s_l;
    const bool steer = *on != 0 && s != 0.0f;
    float x[GRW_PER];
    float dot = 0.0f;
    int k = 0;
#pragma unroll
    for (int d = threadIdx.x; d < N; d += 256, ++k) {
        const float xv = fmaf(bo[t * N + d], sc, r[d]);
        x[k] = xv;
        if (steer && mode == 0) dot = fmaf(xv, v_l[d], dot);
    }
    if (steer && mode == 0) {
#pragma unroll
        for (int o = 16; o > 0; o >>= 1) dot += __shfl_xor_sync(0xffffffffu, dot, o);
        if ((threadIdx.x & 31) == 0) part[threadIdx.x >> 5] = dot;
        __syncthreads();
        if (threadIdx.x < 32) {
            float p = threadIdx.x < 8 ? part[threadIdx.x] : 0.0f;
#pragma unroll
            for (int o = 16; o > 0; o >>= 1) p += __shfl_xor_sync(0xffffffffu, p, o);
            if (threadIdx.x == 0) part[0] = p;
        }
        __syncthreads();
        dot = part[0] * s;
    }
    float ss = 0.0f;
    k = 0;
#pragma unroll
    for (int d = threadIdx.x; d < N; d += 256, ++k) {
        float xv = x[k];
        if (steer) xv = mode == 0 ? fmaf(-dot, v_l[d], xv) : xv + v_l[d];
        r[d] = xv;
        x[k] = xv;
        ss += xv * xv;
    }
    const float rs = rsqrtf(block_sum(ss, sh) / (float) N + eps);
    if (threadIdx.x == 0) rs_out[row] = rs;
    k = 0;
#pragma unroll
    for (int d = threadIdx.x; d < N; d += 256, ++k) {
        const float y = x[k] * rs * w[c * N + d];
        const uint16_t h = act16(y);
        xn16[t * ldx + (int64_t) c * N + d] = h;
        if (xn16_lo) xn16_lo[t * ldx + (int64_t) c * N + d] = bf_lo(y, h);
    }
}
__global__ void gr_silu_kernel(const float* __restrict__ lo, uint16_t* __restrict__ lo16, uint16_t* __restrict__ lo16_lo,
                               int64_t n) {
    const int64_t i = (int64_t) blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;
    const float x = lo[i] / (float) HC;
    const float v = x / (1.0f + __expf(-x));
    const uint16_t h = act16(v);
    lo16[i] = h;
    if (lo16_lo) lo16_lo[i] = bf_lo(v, h);
}
__global__ void gr_mix_kernel(const float* __restrict__ xn, const float* __restrict__ g, float* __restrict__ mixed,
                              uint16_t* __restrict__ mixed16, int64_t T, uint16_t* __restrict__ mixed_h,
                              uint16_t* __restrict__ mixed16_lo) {
    const int64_t i = (int64_t) blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= T * N) return;
    const int64_t t = i / N, d = i % N;
    float s = 0.0f;
#pragma unroll
    for (int c = 0; c < HC; ++c) {
        const int64_t j = t * D + c * N + d;
        s = fmaf(xn[j], sigm(g[j]), s);
    }
    s /= (float) HC;
    mixed[i] = s;
    if (mixed16) {
        const uint16_t h = act16(s);
        mixed16[i] = h;
        if (mixed16_lo) mixed16_lo[i] = bf_lo(s, h);
    }
    if (mixed_h) mixed_h[i] = hf(s);
}
__global__ void gr_write_kernel(float* __restrict__ R, const float* __restrict__ bo, const float* __restrict__ inj,
                                int64_t inj_ld, int64_t T) {
    const int64_t i = (int64_t) blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= T * D) return;
    const int64_t t = i / D, c = (i % D) / N, d = i % N;
    R[i] = fmaf(bo[t * N + d], 2.0f * sigm(inj[t * inj_ld + c] / (float) HC), R[i]);
}
__global__ void gr_broadcast_kernel(const float* __restrict__ e, float* __restrict__ R, int64_t T) {
    const int64_t i = (int64_t) blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= T * D) return;
    const int64_t t = i / D, d = i % N;
    R[i] = e[t * N + d];
}

// ---------------------------------------------------------------- GDN
__global__ void gdn_gates_kernel(const float* __restrict__ ab, const float* __restrict__ dt,
                                 const float* __restrict__ ssm_a, float* __restrict__ gate, float* __restrict__ beta,
                                 int64_t T) {
    const int64_t i = (int64_t) blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= T * HV) return;
    const int64_t t = i / HV, h = i % HV;
    const float v = ab[t * 2 * HV + h] + dt[h];
    gate[i] = (v > 20.0f ? v : log1pf(__expf(v))) * ssm_a[h];
    beta[i] = sigm(ab[t * 2 * HV + HV + h]);
}
// one thread per channel, walks the chunk; then a second kernel normalises
__global__ void gdn_conv_kernel(float* __restrict__ hist, const float* __restrict__ qkv, const float* __restrict__ w,
                                float* __restrict__ h, int64_t T) {
    const int c = blockIdx.x * blockDim.x + threadIdx.x;
    if (c >= C) return;
    float v0 = hist[c * 3], v1 = hist[c * 3 + 1], v2 = hist[c * 3 + 2];
    const float w0 = w[c * 4], w1 = w[c * 4 + 1], w2 = w[c * 4 + 2], w3 = w[c * 4 + 3];
    for (int64_t t = 0; t < T; ++t) {
        const float x = qkv[t * C + c];
        const float s = v0 * w0 + v1 * w1 + v2 * w2 + x * w3;
        h[t * C + c] = s / (1.0f + __expf(-s));
        v0 = v1; v1 = v2; v2 = x;
    }
    hist[c * 3] = v0; hist[c * 3 + 1] = v1; hist[c * 3 + 2] = v2;
}
// C-3: the same 4-tap causal conv, tiled over tokens: thread (c, tile) reads its tile's 3 predecessors from the
// chunk (or the history before it) instead of carrying them - the conv reads inputs, not its own outputs, so the
// tiles are independent. The same expression per element (so the same bits); the history is written afterwards.
constexpr int CONV_TILE = 64;
__global__ void gdn_conv_tiled_kernel(const float* __restrict__ hist, const float* __restrict__ qkv,
                                      const float* __restrict__ w, float* __restrict__ h, int64_t T) {
    const int c = blockIdx.x * blockDim.x + threadIdx.x;
    if (c >= C) return;
    const int64_t t0 = (int64_t) blockIdx.y * CONV_TILE;
    if (t0 >= T) return;
    const int64_t t1 = t0 + CONV_TILE < T ? t0 + CONV_TILE : T;
    auto input = [&](int64_t t) -> float { return t >= 0 ? qkv[t * C + c] : hist[c * 3 + (int) (t + 3)]; };
    float v0 = input(t0 - 3), v1 = input(t0 - 2), v2 = input(t0 - 1);
    const float w0 = w[c * 4], w1 = w[c * 4 + 1], w2 = w[c * 4 + 2], w3 = w[c * 4 + 3];
    for (int64_t t = t0; t < t1; ++t) {
        const float x = qkv[t * C + c];
        const float s = v0 * w0 + v1 * w1 + v2 * w2 + x * w3;
        h[t * C + c] = s / (1.0f + __expf(-s));
        v0 = v1; v1 = v2; v2 = x;
    }
}
// Aurora (S23, opt-in STRATA_GDN_CONVL2=1): gdn_conv_tiled_kernel with gdn_l2_kernel folded in.  A block's 128 threads
// are the 128 channels of one head, so for the q / k heads (block < 2 HK) the norm's sum of squares is the block's:
// the same warp_sum, the same four partials added in the same order, the same v * rsqrtf(ss + eps) on the same
// stored value - the same bits, without writing h and reading it back (16 tokens per round).
__global__ void gdn_conv_l2_kernel(const float* __restrict__ hist, const float* __restrict__ qkv,
                                   const float* __restrict__ w, float* __restrict__ h, int64_t T, float eps) {
    const int c = blockIdx.x * blockDim.x + threadIdx.x;
    const int64_t t0 = (int64_t) blockIdx.y * CONV_TILE;
    if (t0 >= T) return;
    const int64_t t1 = t0 + CONV_TILE < T ? t0 + CONV_TILE : T;
    auto input = [&](int64_t t) -> float { return t >= 0 ? qkv[t * C + c] : hist[c * 3 + (int) (t + 3)]; };
    float v0 = input(t0 - 3), v1 = input(t0 - 2), v2 = input(t0 - 1);
    const float w0 = w[c * 4], w1 = w[c * 4 + 1], w2 = w[c * 4 + 2], w3 = w[c * 4 + 3];
    for (int64_t t = t0; t < t1; ++t) {   // gdn_conv_tiled_kernel's loop, verbatim
        const float x = qkv[t * C + c];
        const float s = v0 * w0 + v1 * w1 + v2 * w2 + x * w3;
        h[t * C + c] = s / (1.0f + __expf(-s));
        v0 = v1; v1 = v2; v2 = x;
    }
    if (blockIdx.x >= 2 * HK) return;   // v channels: no norm
    __shared__ float part[16][4];
    for (int64_t tb = t0; tb < t1; tb += 16) {
        const int n = (int) (t1 - tb < 16 ? t1 - tb : 16);
        for (int j = 0; j < n; ++j) {   // (each thread reads back its own stored value)
            const float v = h[(tb + j) * C + c];
            // gdn_l2_kernel as the compiler built it: the first xor step adds the partner's rounded square to an
            // unrounded own one (an fma), the other steps are plain adds - spelled out so no contraction can differ
            float sq = __fmaf_rn(v, v, __shfl_xor_sync(0xffffffffu, __fmul_rn(v, v), 16));
#pragma unroll
            for (int o = 8; o > 0; o >>= 1) sq = __fadd_rn(sq, __shfl_xor_sync(0xffffffffu, sq, o));
            if ((threadIdx.x & 31) == 0) part[j][threadIdx.x >> 5] = sq;
        }
        __syncthreads();
        for (int j = 0; j < n; ++j) {
            const float ss = part[j][0] + part[j][1] + part[j][2] + part[j][3];
            h[(tb + j) * C + c] = h[(tb + j) * C + c] * rsqrtf(ss + eps);
        }
        __syncthreads();
    }
}
// the history after the chunk: its last three inputs (the older history where the chunk is shorter than 3)
__global__ void gdn_conv_hist_kernel(float* __restrict__ hist, const float* __restrict__ qkv, int64_t T) {
    const int c = blockIdx.x * blockDim.x + threadIdx.x;
    if (c >= C) return;
    float v[3];
    for (int k = 0; k < 3; ++k) {
        const int64_t t = T - 3 + k;
        v[k] = t >= 0 ? qkv[t * C + c] : hist[c * 3 + (int) (t + 3)];
    }
    hist[c * 3] = v[0]; hist[c * 3 + 1] = v[1]; hist[c * 3 + 2] = v[2];
}
__global__ void gdn_l2_kernel(float* __restrict__ h, float eps) {
    // block (t, head) over the 32 q/k heads, 128 threads
    const int64_t t = blockIdx.y;
    const int head = blockIdx.x;
    float* x = h + t * C + head * S;
    const float v = x[threadIdx.x];
    float sq = warp_sum(v * v);
    __shared__ float part[4];
    if ((threadIdx.x & 31) == 0) part[threadIdx.x >> 5] = sq;
    __syncthreads();
    const float ss = part[0] + part[1] + part[2] + part[3];
    x[threadIdx.x] = v * rsqrtf(ss + eps);
}
constexpr int RG = 4, RPG = S / RG;
__global__ void __launch_bounds__(S * RG) gdn_rec_kernel(float* __restrict__ state, const float* __restrict__ h,
                                                         const float* __restrict__ gate,
                                                         const float* __restrict__ beta, const float* __restrict__ z,
                                                         const float* __restrict__ gamma, float eps,
                                                         float* __restrict__ y, uint16_t* __restrict__ y16, int64_t T,
                                                         int64_t ld16) {
    __shared__ float sk[S], sq[S], red[RG][S], wsum[16];
    const int head = blockIdx.x, col = threadIdx.x, rg = threadIdx.y, tid = rg * S + col;
    const int qh = head % HK;
    float s[RPG];
    float* base = state + ((size_t) (rg * RPG) * HV + head) * S + col;
    const size_t rs = (size_t) HV * S;
#pragma unroll
    for (int r = 0; r < RPG; ++r) s[r] = base[r * rs];
    const float g_col = gamma[col];
    for (int64_t t = 0; t < T; ++t) {
        const float* ht = h + t * C;
        __syncthreads();
        if (tid < S) { sq[tid] = ht[qh * S + tid]; sk[tid] = ht[HK * S + qh * S + tid]; }
        __syncthreads();
        const float g = __expf(gate[t * HV + head]);
        float kv = 0.0f;
#pragma unroll
        for (int r = 0; r < RPG; ++r) kv = fmaf(s[r], sk[rg * RPG + r], kv);
        red[rg][col] = kv;
        __syncthreads();
        const float kv_col = red[0][col] + red[1][col] + red[2][col] + red[3][col];
        const float delta = (ht[2 * HK * S + head * S + col] - g * kv_col) * beta[t * HV + head];
        float o = 0.0f;
#pragma unroll
        for (int r = 0; r < RPG; ++r) {
            s[r] = fmaf(g, s[r], sk[rg * RPG + r] * delta);
            o = fmaf(s[r], sq[rg * RPG + r], o);
        }
        __syncthreads();
        red[rg][col] = o;
        __syncthreads();
        float oc = 0.0f, sp = 0.0f;
        if (rg == 0) {
            oc = (red[0][col] + red[1][col] + red[2][col] + red[3][col]) * rsqrtf((float) S);
            sp = oc * oc;
        }
        sp = warp_sum(sp);
        if ((tid & 31) == 0) wsum[tid >> 5] = sp;
        __syncthreads();
        if (rg == 0) {
            const float ss = wsum[0] + wsum[1] + wsum[2] + wsum[3];
            const float v = oc * rsqrtf(ss / (float) S + eps) * g_col * sigm(z[t * HV * S + head * S + col]);
            y[t * HV * S + head * S + col] = v;
            y16[t * ld16 + head * S + col] = hf(v);
        }
    }
#pragma unroll
    for (int r = 0; r < RPG; ++r) base[r * rs] = s[r];
}

// D-2: the recurrence with the value columns split over 4 blocks per head (4x the blocks of the kernel above, a
// quarter of its threads per __syncthreads), and the output norm - the only step that couples the head's columns -
// in its own kernel. Per column the same arithmetic in the same order (the 4 row-group partial sums added as
// red[0] + red[1] + red[2] + red[3]; the norm's warp sums over the same 32-column warps): the same bits.
constexpr int CB = 32, NCB = S / CB;
__global__ void __launch_bounds__(CB * RG) gdn_rec_cols_kernel(float* __restrict__ state, const float* __restrict__ h,
                                                                const float* __restrict__ gate,
                                                                const float* __restrict__ beta,
                                                                float* __restrict__ oc_out, int64_t T) {
    __shared__ float sk[S], sq[S], red[RG][CB];
    const int head = blockIdx.x / NCB, cb = blockIdx.x % NCB;
    const int c = threadIdx.x, rg = threadIdx.y, tid = rg * CB + c, col = cb * CB + c;
    const int qh = head % HK;
    float s[RPG];
    float* base = state + ((size_t) (rg * RPG) * HV + head) * S + col;
    const size_t rs = (size_t) HV * S;
#pragma unroll
    for (int r = 0; r < RPG; ++r) s[r] = base[r * rs];
    for (int64_t t = 0; t < T; ++t) {
        const float* ht = h + t * C;
        __syncthreads();
        if (tid < S) { sq[tid] = ht[qh * S + tid]; sk[tid] = ht[HK * S + qh * S + tid]; }
        __syncthreads();
        const float g = __expf(gate[t * HV + head]);
        float kv = 0.0f;
#pragma unroll
        for (int r = 0; r < RPG; ++r) kv = fmaf(s[r], sk[rg * RPG + r], kv);
        red[rg][c] = kv;
        __syncthreads();
        const float kv_col = red[0][c] + red[1][c] + red[2][c] + red[3][c];
        const float delta = (ht[2 * HK * S + head * S + col] - g * kv_col) * beta[t * HV + head];
        float o = 0.0f;
#pragma unroll
        for (int r = 0; r < RPG; ++r) {
            s[r] = fmaf(g, s[r], sk[rg * RPG + r] * delta);
            o = fmaf(s[r], sq[rg * RPG + r], o);
        }
        __syncthreads();
        red[rg][c] = o;
        __syncthreads();
        if (rg == 0) oc_out[t * HV * S + head * S + col] = (red[0][c] + red[1][c] + red[2][c] + red[3][c]) * rsqrtf((float) S);
    }
#pragma unroll
    for (int r = 0; r < RPG; ++r) base[r * rs] = s[r];
}
// gdn_rec_cols_kernel with the next token's inputs (q/k rows, v, gate, beta) loaded into registers while this token
// computes (software pipelining).  The same arithmetic in the same order: the same bits, and the same CB-column split.
// STRATA_GDN_PIPELINE=0: gdn_rec_cols_kernel.
__global__ void __launch_bounds__(CB * RG) gdn_rec_cols_pipe_kernel(float* __restrict__ state, const float* __restrict__ h,
                                                                      const float* __restrict__ gate,
                                                                      const float* __restrict__ beta,
                                                                      float* __restrict__ oc_out, int64_t T) {
    constexpr int NT = CB * RG, LPT = S / NT;   // threads, q/k rows loaded per thread
    __shared__ float sk[S], sq[S], red[RG][CB];
    const int head = blockIdx.x / NCB, cb = blockIdx.x % NCB;
    const int c = threadIdx.x, rg = threadIdx.y, tid = rg * CB + c, col = cb * CB + c;
    const int qh = head % HK;
    float s[RPG];
    float* base = state + ((size_t) (rg * RPG) * HV + head) * S + col;
    const size_t rs = (size_t) HV * S;
#pragma unroll
    for (int r = 0; r < RPG; ++r) s[r] = base[r * rs];
    float nq[LPT], nk[LPT], nv = 0.0f, ng = 0.0f, nb = 0.0f;
    auto fetch = [&](int64_t t) {
        const float* ht = h + t * C;
#pragma unroll
        for (int u = 0; u < LPT; ++u) { nq[u] = ht[qh * S + tid + u * NT]; nk[u] = ht[HK * S + qh * S + tid + u * NT]; }
        nv = ht[2 * HK * S + head * S + col];
        ng = gate[t * HV + head];
        nb = beta[t * HV + head];
    };
    if (T > 0) fetch(0);
    for (int64_t t = 0; t < T; ++t) {
        float cq[LPT], ck[LPT];
#pragma unroll
        for (int u = 0; u < LPT; ++u) { cq[u] = nq[u]; ck[u] = nk[u]; }
        const float cv = nv, cg = ng, cbt = nb;
        __syncthreads();
#pragma unroll
        for (int u = 0; u < LPT; ++u) { sq[tid + u * NT] = cq[u]; sk[tid + u * NT] = ck[u]; }
        __syncthreads();
        if (t + 1 < T) fetch(t + 1);
        const float g = __expf(cg);
        float kv = 0.0f;
#pragma unroll
        for (int r = 0; r < RPG; ++r) kv = fmaf(s[r], sk[rg * RPG + r], kv);
        red[rg][c] = kv;
        __syncthreads();
        const float kv_col = red[0][c] + red[1][c] + red[2][c] + red[3][c];
        const float delta = (cv - g * kv_col) * cbt;
        float o = 0.0f;
#pragma unroll
        for (int r = 0; r < RPG; ++r) {
            s[r] = fmaf(g, s[r], sk[rg * RPG + r] * delta);
            o = fmaf(s[r], sq[rg * RPG + r], o);
        }
        __syncthreads();
        red[rg][c] = o;
        __syncthreads();
        if (rg == 0) oc_out[t * HV * S + head * S + col] = (red[0][c] + red[1][c] + red[2][c] + red[3][c]) * rsqrtf((float) S);
    }
#pragma unroll
    for (int r = 0; r < RPG; ++r) base[r * rs] = s[r];
}
#if !defined(__HIPCC__)
// The recurrence with one thread for the three value heads that share a key head (head % HK): column c of heads
// qh, qh + 16 and qh + 32, row group rg.  gdn_rec_cols_pipe_kernel spends its time in shared memory, not in
// arithmetic: every thread of a warp needs the same 32 q and k values per token (the k twice), and a warp receives one
// such broadcast value per clock however wide the load.  Here every q/k value a thread loads feeds three heads, and a
// token's k row goes into registers once for both of its uses.  The inputs come in blocks of GDN_TB tokens, copied to
// shared memory by cp.async while the block before computes (one token ahead is shorter than a load from L2 takes),
// and the two cross-row-group sums have their own arrays, so a token needs 2 __syncthreads instead of 5: the second
// one of a token orders every read of rkv before the next token's writes, the next token's first one every read of ro
// before the writes after it.  64 blocks instead of 192.  Per value head and column the same arithmetic in the same
// order: the same bits (src/prefill/gdn_rec_parity.cu checks them and times the variants: 1.41x on a 4080 Super).
// sm_80+ cards that hold its 64 blocks at once (gdn_keyhead_ok); STRATA_GDN_KEYHEAD=0: gdn_rec_cols_pipe_kernel.
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ < 800
#define STRATA_GDN_CP_ASYNC 0   // Turing builds: plain copies (never launched there, see gdn_keyhead_ok)
#else
#define STRATA_GDN_CP_ASYNC 1
#endif
__device__ __forceinline__ void gdn_cp4(float* smem, const float* gmem) {
#if STRATA_GDN_CP_ASYNC
    asm volatile("cp.async.ca.shared.global [%0], [%1], 4;\n" ::"r"((unsigned) __cvta_generic_to_shared(smem)), "l"(gmem));
#else
    *smem = *gmem;
#endif
}
__device__ __forceinline__ void gdn_cp16(float* smem, const float* gmem) {
#if STRATA_GDN_CP_ASYNC
    asm volatile("cp.async.cg.shared.global [%0], [%1], 16;\n" ::"r"((unsigned) __cvta_generic_to_shared(smem)), "l"(gmem));
#else
    *reinterpret_cast<float4*>(smem) = *reinterpret_cast<const float4*>(gmem);
#endif
}
__device__ __forceinline__ void gdn_cp_commit() {
#if STRATA_GDN_CP_ASYNC
    asm volatile("cp.async.commit_group;\n" ::);
#endif
}
__device__ __forceinline__ void gdn_cp_wait_prev() {   // every group but the newest has landed
#if STRATA_GDN_CP_ASYNC
    asm volatile("cp.async.wait_group 1;\n" ::);
#endif
}
constexpr int GDN_TB = 8, VPK = HV / HK;   // tokens per staged block, value heads per key head
__global__ void __launch_bounds__(CB * RG) gdn_rec_kh_kernel(float* __restrict__ state, const float* __restrict__ h,
                                                               const float* __restrict__ gate,
                                                               const float* __restrict__ beta,
                                                               float* __restrict__ oc_out, int64_t T) {
    constexpr int TB = GDN_TB, NT = CB * RG, QKP = S / 4, VP = CB / 4;   // threads, 16-byte pieces of a q/k row, of v
    __shared__ __align__(16) float sq[2][TB][S];
    __shared__ __align__(16) float sk[2][TB][S];
    __shared__ __align__(16) float sv[2][TB][VPK][CB];
    __shared__ float sg[2][TB][VPK], sb[2][TB][VPK], rkv[VPK][RG][CB], ro[VPK][RG][CB];
    const int qh = blockIdx.x / NCB, cb = blockIdx.x % NCB;
    const int c = threadIdx.x, rg = threadIdx.y, tid = rg * CB + c, col = cb * CB + c;
    float s[VPK][RPG];
    const size_t rs = (size_t) HV * S;
#pragma unroll
    for (int j = 0; j < VPK; ++j) {
        const float* base = state + ((size_t) (rg * RPG) * HV + qh + j * HK) * S + col;
#pragma unroll
        for (int r = 0; r < RPG; ++r) s[j][r] = base[r * rs];
    }
    const int64_t nblk = (T + TB - 1) / TB;
    auto stage = [&](int64_t k) {   // tokens [k * TB, k * TB + TB) into buffer k & 1
        const int bb = (int) (k & 1);
        const int64_t t0 = k * TB;
        for (int p = tid; p < TB * 2 * QKP; p += NT) {
            const int i = p / (2 * QKP), w = p % (2 * QKP), isk = w / QKP, jj = (w % QKP) * 4;
            if (t0 + i < T)
                gdn_cp16(isk ? &sk[bb][i][jj] : &sq[bb][i][jj], h + (t0 + i) * C + (isk ? HK * S : 0) + qh * S + jj);
        }
        for (int p = tid; p < TB * VPK * VP; p += NT) {
            const int i = p / (VPK * VP), w = p % (VPK * VP), j = w / VP, jj = (w % VP) * 4;
            if (t0 + i < T)
                gdn_cp16(&sv[bb][i][j][jj], h + (t0 + i) * C + 2 * HK * S + (qh + j * HK) * S + cb * CB + jj);
        }
        for (int p = tid; p < 2 * TB * VPK; p += NT) {
            const int isb = p / (TB * VPK), w = p % (TB * VPK), i = w / VPK, j = w % VPK;
            if (t0 + i < T)
                gdn_cp4(isb ? &sb[bb][i][j] : &sg[bb][i][j], (isb ? beta : gate) + (t0 + i) * HV + qh + j * HK);
        }
    };
    if (nblk > 0) stage(0);
    gdn_cp_commit();
    for (int64_t k = 0; k < nblk; ++k) {
        // buffer (k + 1) & 1 was read by block k - 1, whose last token's second __syncthreads every thread has passed
        if (k + 1 < nblk) stage(k + 1);
        gdn_cp_commit();
        gdn_cp_wait_prev();
        __syncthreads();
        const int bb = (int) (k & 1);
        const int n = (int) ((T - k * TB) < TB ? (T - k * TB) : TB);
        for (int i = 0; i < n; ++i) {
            const int64_t t = k * TB + i;
            float kc[RPG];
#pragma unroll
            for (int r = 0; r < RPG; ++r) kc[r] = sk[bb][i][rg * RPG + r];
            float g[VPK], kv[VPK], delta[VPK], o[VPK];
#pragma unroll
            for (int j = 0; j < VPK; ++j) { g[j] = __expf(sg[bb][i][j]); kv[j] = 0.0f; o[j] = 0.0f; }
#pragma unroll
            for (int r = 0; r < RPG; ++r)
#pragma unroll
                for (int j = 0; j < VPK; ++j) kv[j] = fmaf(s[j][r], kc[r], kv[j]);
#pragma unroll
            for (int j = 0; j < VPK; ++j) rkv[j][rg][c] = kv[j];
            __syncthreads();
#pragma unroll
            for (int j = 0; j < VPK; ++j) {
                const float kv_col = rkv[j][0][c] + rkv[j][1][c] + rkv[j][2][c] + rkv[j][3][c];
                delta[j] = (sv[bb][i][j][c] - g[j] * kv_col) * sb[bb][i][j];
            }
#pragma unroll
            for (int r = 0; r < RPG; ++r) {
                const float qr = sq[bb][i][rg * RPG + r];
#pragma unroll
                for (int j = 0; j < VPK; ++j) {
                    s[j][r] = fmaf(g[j], s[j][r], kc[r] * delta[j]);
                    o[j] = fmaf(s[j][r], qr, o[j]);
                }
            }
#pragma unroll
            for (int j = 0; j < VPK; ++j) ro[j][rg][c] = o[j];
            __syncthreads();
            if (rg < VPK)   // row group j writes head j's output
                oc_out[t * HV * S + (qh + rg * HK) * S + col] =
                    (ro[rg][0][c] + ro[rg][1][c] + ro[rg][2][c] + ro[rg][3][c]) * rsqrtf((float) S);
        }
    }
#pragma unroll
    for (int j = 0; j < VPK; ++j) {
        float* base = state + ((size_t) (rg * RPG) * HV + qh + j * HK) * S + col;
#pragma unroll
        for (int r = 0; r < RPG; ++r) base[r * rs] = s[j][r];
    }
}
// gdn_rec_kh_kernel where it pays: a CUDA card with cp.async (sm_80+) that holds all 64 of its blocks at once (each
// walks the whole chunk, so blocks left for a second wave would double the time).  The busiest SM sets the pace: from
// 64 SMs up this kernel has one block per SM, below that two on some SMs, while the kernel before has ceil(192 / SMs).
// With the engine's grids on fewer SMs (gdn_rec_parity --bench), on a 4080 SUPER / a 3090: 1.40-1.42x / 1.28-1.29x at
// 64 SMs and more, 1.02-1.04x / 1.03-1.06x at 48 to 63 (two blocks against four: a draw, slower on neither),
// 1.28-1.31x / 1.28-1.30x at 39 to 47, 1.53-1.57x / 1.54-1.55x at 32 to 38.  Per call, from the current device (a
// layer split can mix cards).
bool gdn_keyhead_ok() {
    static const bool off = [] { const char* v = std::getenv("STRATA_GDN_KEYHEAD"); return v != nullptr && std::atoi(v) == 0; }();
    if (off) return false;
    static int known[64] = {};   // per device: 0 not asked yet, 1 yes, 2 no
    int dev = 0;
    if (cudaGetDevice(&dev) != cudaSuccess || dev < 0 || dev >= 64) { cudaGetLastError(); return false; }
    if (known[dev] == 0) {
        int major = 0, sms = 0, per_sm = 0;
        const bool yes = cudaDeviceGetAttribute(&major, cudaDevAttrComputeCapabilityMajor, dev) == cudaSuccess &&
                         cudaDeviceGetAttribute(&sms, cudaDevAttrMultiProcessorCount, dev) == cudaSuccess && major >= 8 &&
                         cudaOccupancyMaxActiveBlocksPerMultiprocessor(&per_sm, gdn_rec_kh_kernel, CB * RG, 0) ==
                             cudaSuccess &&
                         (int64_t) per_sm * sms >= (int64_t) HK * NCB;
        if (!yes) cudaGetLastError();
        known[dev] = yes ? 1 : 2;
    }
    return known[dev] == 1;
}
#endif
#if defined(__HIPCC__)   // the Aurora (gfx11) recurrence kernels: AMD builds only
// Aurora (S23): the recurrence with the column-split grid of gdn_rec_cols_pipe_kernel (four 32-column blocks per value
// head) but FOUR LANES per column, lane g holding the column's state rows 32g .. 32g + 31 in registers - exactly the
// row group the old kernel's thread (col, rg = g) held - so a token needs no __syncthreads: the four kv / o partials
// meet through wave shuffles and are added in the old order red[0] + red[1] + red[2] + red[3] (the same bits;
// tests/hip/gdn_rec_head.cpp).  The q/k rows, v, gate and beta are staged in LDS HT tokens at a time, the next chunk
// loaded into registers while this one computes, and a token's k and q rows are read into registers before its
// products (one LDS latency per token, not one per load).  The old kernels took ~4 barriers per token and ~3.5 us
// per token and layer on the Radeon 8060S; the output norm runs after it in gdn_out_norm_warp_kernel.
// STRATA_GDN_HEAD=0: the old kernels.
constexpr int HT = 4;                           // tokens per staged chunk (LDS ~10 KB: all 192 blocks resident)
constexpr int QCB = 32;                         // columns per block
constexpr int QTH = QCB * 4;                    // threads per block: 4 lanes per column
constexpr int QRS = 36;                         // staged row-group stride (32 rows + 4 pad: the 4 groups' reads
                                                // land in different LDS banks)
constexpr int QPF = HT * 2 * S / 4 / QTH;       // float4 of q/k each thread prefetches per chunk (2)
__global__ void __launch_bounds__(QTH) gdn_rec_quad_kernel(float* __restrict__ state, const float* __restrict__ h,
                                                           const float* __restrict__ gate, const float* __restrict__ beta,
                                                           float* __restrict__ oc_out, int64_t T) {
    __shared__ __align__(16) float sqk[2][HT][2][4 * QRS];   // [buffer][token][q | k][group][row (padded)]
    __shared__ float sgb[2][2][HT];                         // [buffer][gate | beta][token]
    __shared__ float sv[2][HT][QCB];                        // [buffer][token][column] v
    const int head = blockIdx.x / (S / QCB), cb = blockIdx.x % (S / QCB);
    const int tid = threadIdx.x, lane = tid & 31, wave = tid >> 5;
    const int g = lane >> 3, cl = wave * 8 + (lane & 7), col = cb * QCB + cl;   // row group, column in block / head
    const int qh = head % HK;
    float s[32];
    {
        const float* base = state + ((size_t) (g * 32) * HV + head) * S + col;
#pragma unroll
        for (int r = 0; r < 32; ++r) s[r] = base[(size_t) r * HV * S];
    }
    float4 pf[QPF];
    float pv[HT], pgb = 0.0f;
    // chunk c's inputs into registers: q/k (float4 i of the chunk's [token][q|k][S/4] block), the column's v (lanes of
    // group 0) and gate (threads 0..HT-1) / beta (HT..2HT-1) of token `tid % HT`
    auto fetch = [&](int64_t c) {
        const int64_t t0 = c * HT;
#pragma unroll
        for (int u = 0; u < QPF; ++u) {
            const int i = tid + u * QTH;
            const int tt = i / (2 * S / 4), rem = i % (2 * S / 4), part = rem / (S / 4), r4 = rem % (S / 4);
            const int64_t t = t0 + tt;
            pf[u] = t < T ? *reinterpret_cast<const float4*>(h + t * C + part * HK * S + qh * S + r4 * 4)
                          : make_float4(0.f, 0.f, 0.f, 0.f);
        }
        if (g == 0) {
#pragma unroll
            for (int tt = 0; tt < HT; ++tt) {
                const int64_t t = t0 + tt;
                pv[tt] = t < T ? h[t * C + 2 * HK * S + head * S + col] : 0.0f;
            }
        }
        if (tid < 2 * HT) {
            const int64_t t = t0 + (tid % HT);
            pgb = t < T ? (tid < HT ? gate : beta)[t * HV + head] : 0.0f;
        }
    };
    const float rs_s = rsqrtf((float) S);
    const int src0 = lane & 7;                       // lane of group 0 for this column; group j is src0 + 8 j
    const int64_t nch = (T + HT - 1) / HT;
    if (nch > 0) fetch(0);
    for (int64_t c = 0; c < nch; ++c) {
        const int b = (int) (c & 1);
        if (g == 0) {
#pragma unroll
            for (int tt = 0; tt < HT; ++tt) sv[b][tt][cl] = pv[tt];
        }
#pragma unroll
        for (int u = 0; u < QPF; ++u) {
            const int i = tid + u * QTH;
            const int tt = i / (2 * S / 4), rem = i % (2 * S / 4), part = rem / (S / 4), r4 = rem % (S / 4);
            const int row = r4 * 4;
            *reinterpret_cast<float4*>(&sqk[b][tt][part][(row >> 5) * QRS + (row & 31)]) = pf[u];
        }
        if (tid < 2 * HT) sgb[b][tid / HT][tid % HT] = pgb;
        __syncthreads();                               // chunk c staged (buffer b was last read two chunks ago)
        if (c + 1 < nch) fetch(c + 1);
        const int64_t t0 = c * HT;
        const int nt = (int) (T - t0 < HT ? T - t0 : HT);
        for (int tt = 0; tt < nt; ++tt) {
            float k[32], q[32];
            {
                const float4* k4 = reinterpret_cast<const float4*>(&sqk[b][tt][1][g * QRS]);
                const float4* q4 = reinterpret_cast<const float4*>(&sqk[b][tt][0][g * QRS]);
#pragma unroll
                for (int j = 0; j < 8; ++j) {
                    const float4 kk = k4[j], qq = q4[j];
                    k[4 * j] = kk.x; k[4 * j + 1] = kk.y; k[4 * j + 2] = kk.z; k[4 * j + 3] = kk.w;
                    q[4 * j] = qq.x; q[4 * j + 1] = qq.y; q[4 * j + 2] = qq.z; q[4 * j + 3] = qq.w;
                }
            }
            const float g_exp = __expf(sgb[b][0][tt]);
            float kv = 0.0f;
#pragma unroll
            for (int r = 0; r < 32; ++r) kv = fmaf(s[r], k[r], kv);
            const float p0 = __shfl_sync(0xffffffffu, kv, src0), p1 = __shfl_sync(0xffffffffu, kv, src0 + 8),
                        p2 = __shfl_sync(0xffffffffu, kv, src0 + 16), p3 = __shfl_sync(0xffffffffu, kv, src0 + 24);
            const float kv_col = p0 + p1 + p2 + p3;
            const float cvt = sv[b][tt][cl], cbt = sgb[b][1][tt];
            const float delta = (cvt - g_exp * kv_col) * cbt;
            float o = 0.0f;
#pragma unroll
            for (int r = 0; r < 32; ++r) {
                s[r] = fmaf(g_exp, s[r], k[r] * delta);
                o = fmaf(s[r], q[r], o);
            }
            const float o0 = __shfl_sync(0xffffffffu, o, src0), o1 = __shfl_sync(0xffffffffu, o, src0 + 8),
                        o2 = __shfl_sync(0xffffffffu, o, src0 + 16), o3 = __shfl_sync(0xffffffffu, o, src0 + 24);
            if (g == 0) oc_out[(t0 + tt) * HV * S + head * S + col] = (o0 + o1 + o2 + o3) * rs_s;
        }
    }
    {
        float* base = state + ((size_t) (g * 32) * HV + head) * S + col;
#pragma unroll
        for (int r = 0; r < 32; ++r) base[(size_t) r * HV * S] = s[r];
    }
}
// Aurora (S23, opt-in STRATA_GDN_PP=1): gdn_rec_quad_kernel with the same arithmetic in the same order, two changes in
// how it is compiled.  (1) The VGPR count: the 32 state-row addresses of the first loads were kept live across the
// whole walk to be reused by the final stores (64 VGPRs, 196 in all = 7 waves a SIMD for the 192 blocks of 4 waves
// that need 9.6): the stores' lane offset goes through an asm so the addresses are made again, 133 VGPRs, all blocks
// resident.  (2) A full chunk's four tokens run with the state alternating between two register sets (the update
// writes the other set), so the compiler emits no 32 register copies after the update.
__global__ void __launch_bounds__(QTH) gdn_rec_quad_pp_kernel(float* __restrict__ state, const float* __restrict__ h,
                                                           const float* __restrict__ gate, const float* __restrict__ beta,
                                                           float* __restrict__ oc_out, int64_t T) {
    __shared__ __align__(16) float sqk[2][HT][2][4 * QRS];   // [buffer][token][q | k][group][row (padded)]
    __shared__ float sgb[2][2][HT];                         // [buffer][gate | beta][token]
    __shared__ float sv[2][HT][QCB];                        // [buffer][token][column] v
    const int head = blockIdx.x / (S / QCB), cb = blockIdx.x % (S / QCB);
    const int tid = threadIdx.x, lane = tid & 31, wave = tid >> 5;
    const int g = lane >> 3, cl = wave * 8 + (lane & 7), col = cb * QCB + cl;   // row group, column in block / head
    const int qh = head % HK;
    float s[32];
    {
        const float* sb = state + (size_t) head * S;
        const uint32_t voff = (uint32_t) ((g * 32) * HV * S + col);
#pragma unroll
        for (int r = 0; r < 32; ++r) s[r] = sb[(size_t) r * HV * S + voff];
    }
    float4 pf[QPF];
    float pv[HT], pgb = 0.0f;
    // chunk c's inputs into registers: q/k (float4 i of the chunk's [token][q|k][S/4] block), the column's v (lanes of
    // group 0) and gate (threads 0..HT-1) / beta (HT..2HT-1) of token `tid % HT`
    auto fetch = [&](int64_t c) {
        const int64_t t0 = c * HT;
#pragma unroll
        for (int u = 0; u < QPF; ++u) {
            const int i = tid + u * QTH;
            const int tt = i / (2 * S / 4), rem = i % (2 * S / 4), part = rem / (S / 4), r4 = rem % (S / 4);
            const int64_t t = t0 + tt;
            pf[u] = t < T ? *reinterpret_cast<const float4*>(h + t * C + part * HK * S + qh * S + r4 * 4)
                          : make_float4(0.f, 0.f, 0.f, 0.f);
        }
        if (g == 0) {
#pragma unroll
            for (int tt = 0; tt < HT; ++tt) {
                const int64_t t = t0 + tt;
                pv[tt] = t < T ? h[t * C + 2 * HK * S + head * S + col] : 0.0f;
            }
        }
        if (tid < 2 * HT) {
            const int64_t t = t0 + (tid % HT);
            pgb = t < T ? (tid < HT ? gate : beta)[t * HV + head] : 0.0f;
        }
    };
    const float rs_s = rsqrtf((float) S);
    const int src0 = lane & 7;                       // lane of group 0 for this column; group j is src0 + 8 j
    const int64_t nch = (T + HT - 1) / HT;
    if (nch > 0) fetch(0);
    for (int64_t c = 0; c < nch; ++c) {
        const int b = (int) (c & 1);
        if (g == 0) {
#pragma unroll
            for (int tt = 0; tt < HT; ++tt) sv[b][tt][cl] = pv[tt];
        }
#pragma unroll
        for (int u = 0; u < QPF; ++u) {
            const int i = tid + u * QTH;
            const int tt = i / (2 * S / 4), rem = i % (2 * S / 4), part = rem / (S / 4), r4 = rem % (S / 4);
            const int row = r4 * 4;
            *reinterpret_cast<float4*>(&sqk[b][tt][part][(row >> 5) * QRS + (row & 31)]) = pf[u];
        }
        if (tid < 2 * HT) sgb[b][tid / HT][tid % HT] = pgb;
        __syncthreads();                               // chunk c staged (buffer b was last read two chunks ago)
        if (c + 1 < nch) fetch(c + 1);
        const int64_t t0 = c * HT;
        const int nt = (int) (T - t0 < HT ? T - t0 : HT);
        auto token = [&](int tt, float (&sin)[32], float (&sout)[32]) {
            float k[32], q[32];
            {
                const float4* k4 = reinterpret_cast<const float4*>(&sqk[b][tt][1][g * QRS]);
                const float4* q4 = reinterpret_cast<const float4*>(&sqk[b][tt][0][g * QRS]);
#pragma unroll
                for (int j = 0; j < 8; ++j) {
                    const float4 kk = k4[j], qq = q4[j];
                    k[4 * j] = kk.x; k[4 * j + 1] = kk.y; k[4 * j + 2] = kk.z; k[4 * j + 3] = kk.w;
                    q[4 * j] = qq.x; q[4 * j + 1] = qq.y; q[4 * j + 2] = qq.z; q[4 * j + 3] = qq.w;
                }
            }
            const float g_exp = __expf(sgb[b][0][tt]);
            float kv = 0.0f;
#pragma unroll
            for (int r = 0; r < 32; ++r) kv = fmaf(sin[r], k[r], kv);
            const float p0 = __shfl_sync(0xffffffffu, kv, src0), p1 = __shfl_sync(0xffffffffu, kv, src0 + 8),
                        p2 = __shfl_sync(0xffffffffu, kv, src0 + 16), p3 = __shfl_sync(0xffffffffu, kv, src0 + 24);
            const float kv_col = p0 + p1 + p2 + p3;
            const float cvt = sv[b][tt][cl], cbt = sgb[b][1][tt];
            const float delta = (cvt - g_exp * kv_col) * cbt;
            float o = 0.0f;
#pragma unroll
            for (int r = 0; r < 32; ++r) {
                sout[r] = fmaf(g_exp, sin[r], k[r] * delta);
                o = fmaf(sout[r], q[r], o);
            }
            const float o0 = __shfl_sync(0xffffffffu, o, src0), o1 = __shfl_sync(0xffffffffu, o, src0 + 8),
                        o2 = __shfl_sync(0xffffffffu, o, src0 + 16), o3 = __shfl_sync(0xffffffffu, o, src0 + 24);
            if (g == 0) oc_out[(t0 + tt) * HV * S + head * S + col] = (o0 + o1 + o2 + o3) * rs_s;
        };
        if (nt == HT) {   // the state alternates between two register sets: no copy after the update
            float s2[32];
            token(0, s, s2);
            token(1, s2, s);
            token(2, s, s2);
            token(3, s2, s);
        } else {
            for (int tt = 0; tt < nt; ++tt) {
                float s2[32];
                token(tt, s, s2);
#pragma unroll
                for (int r = 0; r < 32; ++r) s[r] = s2[r];
            }
        }
    }
    {
        float* sb = state + (size_t) head * S;
        uint32_t voff = (uint32_t) ((g * 32) * HV * S + col);
        asm volatile("" : "+v"(voff));   // (so the loads' 32 addresses are not kept alive across the whole loop)
#pragma unroll
        for (int r = 0; r < 32; ++r) sb[(size_t) r * HV * S + voff] = s[r];
    }
}

// Aurora (S23, opt-in STRATA_GDN_PP=2): gdn_rec_quad_kernel with two columns per lane (a wave: 4 row groups x 8 lanes x
// 2 columns = 16 columns; a block of 4 waves 64 columns, 96 blocks).  A lane's two columns share the k / q rows it
// reads from shared memory (half the LDS reads and shuffles per column, the kernel's limit) and give each wave two
// independent dependency chains; the 384 waves fit in one round at 230 VGPRs.  Per column the same arithmetic in
// the same order (the same bits as gdn_rec_quad_kernel).  The store of the state goes through an asm for the reason
// given at gdn_rec_quad_pp_kernel.

// ---- C2: two columns per lane (a wave: 4 row groups x 8 lanes x 2 columns = 16 columns; a block of 4 waves 64
// columns).  A lane's two columns share the k / q rows it reads from LDS (half the LDS reads and shuffles per column
// and two independent dependency chains in each wave).  Per column the same arithmetic in the same order.
constexpr int C2CB = 64;                          // columns per block
constexpr int C2PF = HT * 2 * S / 4 / QTH;        // float4 of q/k each thread prefetches per chunk (2)
__global__ void __launch_bounds__(QTH) gdn_rec_quad2c_kernel(float* __restrict__ state, const float* __restrict__ h,
                                                             const float* __restrict__ gate, const float* __restrict__ beta,
                                                             float* __restrict__ oc_out, int64_t T) {
    __shared__ __align__(16) float sqk[2][HT][2][4 * QRS];
    __shared__ float sgb[2][2][HT];
    __shared__ float sv[2][HT][C2CB];
    const int head = blockIdx.x / (S / C2CB), cb = blockIdx.x % (S / C2CB);
    const int tid = threadIdx.x, lane = tid & 31, wave = tid >> 5;
    const int g = lane >> 3, cl0 = wave * 16 + (lane & 7), cl1 = cl0 + 8;   // the lane's two columns in the block
    const int col0 = cb * C2CB + cl0, col1 = cb * C2CB + cl1;
    const int qh = head % HK;
    float s0[32], s1[32];
    {
        const float* sb = state + (size_t) head * S;
        const uint32_t voff = (uint32_t) ((g * 32) * HV * S);
#pragma unroll
        for (int r = 0; r < 32; ++r) { s0[r] = sb[(size_t) r * HV * S + voff + col0]; s1[r] = sb[(size_t) r * HV * S + voff + col1]; }
    }
    float4 pf[C2PF];
    float pv[2][HT], pgb = 0.0f;
    auto fetch = [&](int64_t c) {
        const int64_t t0 = c * HT;
#pragma unroll
        for (int u = 0; u < C2PF; ++u) {
            const int i = tid + u * QTH;
            const int tt = i / (2 * S / 4), rem = i % (2 * S / 4), part = rem / (S / 4), r4 = rem % (S / 4);
            const int64_t t = t0 + tt;
            pf[u] = t < T ? *reinterpret_cast<const float4*>(h + t * C + part * HK * S + qh * S + r4 * 4)
                          : make_float4(0.f, 0.f, 0.f, 0.f);
        }
        if (g == 0) {
#pragma unroll
            for (int tt = 0; tt < HT; ++tt) {
                const int64_t t = t0 + tt;
                pv[0][tt] = t < T ? h[t * C + 2 * HK * S + head * S + col0] : 0.0f;
                pv[1][tt] = t < T ? h[t * C + 2 * HK * S + head * S + col1] : 0.0f;
            }
        }
        if (tid < 2 * HT) {
            const int64_t t = t0 + (tid % HT);
            pgb = t < T ? (tid < HT ? gate : beta)[t * HV + head] : 0.0f;
        }
    };
    const float rs_s = rsqrtf((float) S);
    const int src0 = lane & 7;
    const int64_t nch = (T + HT - 1) / HT;
    if (nch > 0) fetch(0);
    for (int64_t c = 0; c < nch; ++c) {
        const int b = (int) (c & 1);
        if (g == 0) {
#pragma unroll
            for (int tt = 0; tt < HT; ++tt) { sv[b][tt][cl0] = pv[0][tt]; sv[b][tt][cl1] = pv[1][tt]; }
        }
#pragma unroll
        for (int u = 0; u < C2PF; ++u) {
            const int i = tid + u * QTH;
            const int tt = i / (2 * S / 4), rem = i % (2 * S / 4), part = rem / (S / 4), r4 = rem % (S / 4);
            const int row = r4 * 4;
            *reinterpret_cast<float4*>(&sqk[b][tt][part][(row >> 5) * QRS + (row & 31)]) = pf[u];
        }
        if (tid < 2 * HT) sgb[b][tid / HT][tid % HT] = pgb;
        __syncthreads();
        if (c + 1 < nch) fetch(c + 1);
        const int64_t t0 = c * HT;
        const int nt = (int) (T - t0 < HT ? T - t0 : HT);
        for (int tt = 0; tt < nt; ++tt) {
            float k[32], q[32];
            {
                const float4* k4 = reinterpret_cast<const float4*>(&sqk[b][tt][1][g * QRS]);
                const float4* q4 = reinterpret_cast<const float4*>(&sqk[b][tt][0][g * QRS]);
#pragma unroll
                for (int j = 0; j < 8; ++j) {
                    const float4 kk = k4[j], qq = q4[j];
                    k[4 * j] = kk.x; k[4 * j + 1] = kk.y; k[4 * j + 2] = kk.z; k[4 * j + 3] = kk.w;
                    q[4 * j] = qq.x; q[4 * j + 1] = qq.y; q[4 * j + 2] = qq.z; q[4 * j + 3] = qq.w;
                }
            }
            const float g_exp = __expf(sgb[b][0][tt]);
            float kv0 = 0.0f, kv1 = 0.0f;
#pragma unroll
            for (int r = 0; r < 32; ++r) { kv0 = fmaf(s0[r], k[r], kv0); kv1 = fmaf(s1[r], k[r], kv1); }
            const float a0 = __shfl_sync(0xffffffffu, kv0, src0), a1 = __shfl_sync(0xffffffffu, kv0, src0 + 8),
                        a2 = __shfl_sync(0xffffffffu, kv0, src0 + 16), a3 = __shfl_sync(0xffffffffu, kv0, src0 + 24);
            const float b0 = __shfl_sync(0xffffffffu, kv1, src0), b1 = __shfl_sync(0xffffffffu, kv1, src0 + 8),
                        b2 = __shfl_sync(0xffffffffu, kv1, src0 + 16), b3 = __shfl_sync(0xffffffffu, kv1, src0 + 24);
            const float kvc0 = a0 + a1 + a2 + a3, kvc1 = b0 + b1 + b2 + b3;
            const float cbt = sgb[b][1][tt];
            const float d0 = (sv[b][tt][cl0] - g_exp * kvc0) * cbt, d1 = (sv[b][tt][cl1] - g_exp * kvc1) * cbt;
            float o0 = 0.0f, o1 = 0.0f;
#pragma unroll
            for (int r = 0; r < 32; ++r) {
                s0[r] = fmaf(g_exp, s0[r], k[r] * d0);
                s1[r] = fmaf(g_exp, s1[r], k[r] * d1);
                o0 = fmaf(s0[r], q[r], o0);
                o1 = fmaf(s1[r], q[r], o1);
            }
            const float x0 = __shfl_sync(0xffffffffu, o0, src0), x1 = __shfl_sync(0xffffffffu, o0, src0 + 8),
                        x2 = __shfl_sync(0xffffffffu, o0, src0 + 16), x3 = __shfl_sync(0xffffffffu, o0, src0 + 24);
            const float y0 = __shfl_sync(0xffffffffu, o1, src0), y1 = __shfl_sync(0xffffffffu, o1, src0 + 8),
                        y2 = __shfl_sync(0xffffffffu, o1, src0 + 16), y3 = __shfl_sync(0xffffffffu, o1, src0 + 24);
            if (g == 0) {
                oc_out[(t0 + tt) * HV * S + head * S + col0] = (x0 + x1 + x2 + x3) * rs_s;
                oc_out[(t0 + tt) * HV * S + head * S + col1] = (y0 + y1 + y2 + y3) * rs_s;
            }
        }
    }
    {
        float* sb = state + (size_t) head * S;
        uint32_t voff = (uint32_t) ((g * 32) * HV * S);
        asm volatile("" : "+v"(voff));
#pragma unroll
        for (int r = 0; r < 32; ++r) { sb[(size_t) r * HV * S + voff + col0] = s0[r]; sb[(size_t) r * HV * S + voff + col1] = s1[r]; }
    }
}

// gdn_out_norm_kernel's body over a grid-stride walk of the (token, head) rows: the old kernel launched a 128-thread
// block per row (T * 48 blocks; ~45 GB/s on the 8060S).  The same code per row (the same compiled arithmetic: a wave
// per row with four values per lane rounded differently in its first xor step).
template <bool WY>
__global__ void __launch_bounds__(S) gdn_out_norm_loop_kernel(const float* __restrict__ z, const float* __restrict__ gamma,
                                                              float eps, float* __restrict__ y, uint16_t* __restrict__ y16,
                                                              int64_t rows, int64_t ld16) {
    __shared__ float wsum[4];
    const int col = threadIdx.x;
    for (int64_t row = blockIdx.x; row < rows; row += gridDim.x) {
        const size_t at = (size_t) row * S + col;
        const float oc = y[at];
        // (warp_sum as the compiler built it: the first xor step adds the partner's rounded square to an unrounded own
        // one, an fma; spelled out so neither instance's contraction can differ)
        float sp = __fmaf_rn(oc, oc, __shfl_xor_sync(0xffffffffu, __fmul_rn(oc, oc), 16));
#pragma unroll
        for (int o = 8; o > 0; o >>= 1) sp = __fadd_rn(sp, __shfl_xor_sync(0xffffffffu, sp, o));
        __syncthreads();                               // the previous row's sums are read
        if ((col & 31) == 0) wsum[col >> 5] = sp;
        __syncthreads();
        const float ss = wsum[0] + wsum[1] + wsum[2] + wsum[3];
        const float bp = oc * rsqrtf(ss / (float) S + eps) * gamma[col], sg = sigm(z[at]);
        const size_t o16 = (size_t) (row / HV) * ld16 + (size_t) (row % HV) * S + col;
        if constexpr (WY) {
            const float v = bp * sg;
            y[at] = v;
            y16[o16] = hf(v);
        } else {
            // STRATA_GDN_NOY=1: nothing reads the FP32 result (403 MB per 16K chunk).  The FP16 value is what the
            // compiler makes of hf(v) next to the store: v_fma_mixlo_f16, the product rounded to FP16 once
            unsigned r;
            asm("v_fma_mixlo_f16 %0, %1, %2, 0" : "=v"(r) : "v"(sg), "v"(bp));
            y16[o16] = (uint16_t) r;
        }
    }
}
#endif   // __HIPCC__
__global__ void __launch_bounds__(S) gdn_out_norm_kernel(const float* __restrict__ z, const float* __restrict__ gamma,
                                                         float eps, float* __restrict__ y, uint16_t* __restrict__ y16,
                                                         int64_t ld16) {
    __shared__ float wsum[4];
    const int64_t t = blockIdx.x;
    const int head = blockIdx.y, col = threadIdx.x;
    const size_t at = (size_t) t * HV * S + (size_t) head * S + col;
    const float oc = y[at];
    float sp = warp_sum(oc * oc);
    if ((col & 31) == 0) wsum[col >> 5] = sp;
    __syncthreads();
    const float ss = wsum[0] + wsum[1] + wsum[2] + wsum[3];
    const float v = oc * rsqrtf(ss / (float) S + eps) * gamma[col] * sigm(z[t * HV * S + head * S + col]);
    y16[(size_t) t * ld16 + (size_t) head * S + col] = hf(v);   // (the FP32 normalized value is not stored: nothing reads it)
}

// ---------------------------------------------------------------- MoE
template <int REG>
__global__ void route_kernel(const float* __restrict__ logits, int32_t* __restrict__ ids, float* __restrict__ wout,
                             int64_t T) {
    const int64_t t = (int64_t) blockIdx.x * (blockDim.x >> 5) + (threadIdx.x >> 5);
    if (t >= T) return;
    const int lane = threadIdx.x & 31;
    const float* lg = logits + t * (REG * 32);
    float v[REG];
#pragma unroll
    for (int i = 0; i < REG; ++i) v[i] = lg[lane + i * 32];
    float mx = -INFINITY;
#pragma unroll
    for (int i = 0; i < REG; ++i) mx = fmaxf(mx, v[i]);
    mx = warp_max(mx);
    float sum = 0.0f;
#pragma unroll
    for (int i = 0; i < REG; ++i) { v[i] = expf(v[i] - mx); sum += v[i]; }
    const float rcp = 1.0f / warp_sum(sum);
#pragma unroll
    for (int i = 0; i < REG; ++i) { v[i] *= rcp; if (isnan(v[i])) v[i] = -FLT_MAX; }
    float selected = 0.0f, selected_sum = 0.0f;
    for (int rank = 0; rank < 10; ++rank) {
        float best = v[0];
        int ex = lane;
#pragma unroll
        for (int i = 1; i < REG; ++i) if (v[i] > best) { best = v[i]; ex = lane + i * 32; }
#pragma unroll
        for (int m = 16; m; m >>= 1) {
            const float ob = __shfl_xor_sync(0xffffffffu, best, m);
            const int oi = __shfl_xor_sync(0xffffffffu, ex, m);
            if (ob > best || (ob == best && oi < ex)) { best = ob; ex = oi; }
        }
        if ((ex & 31) == lane) { v[ex / 32] = -INFINITY; selected_sum += best; }
        if (lane == 0) ids[t * 10 + rank] = ex;
        if (rank == lane) selected = best;
    }
    selected_sum = fmaxf(warp_sum(selected_sum), 6.103515625e-5f);
    if (lane < 10) wout[t * 10 + lane] = selected / selected_sum;
}
// Strata blob: gate/up codes [1280][640 B], down codes [2560][160 B], gate/up scales [1280][40] f16, down scales [2560][10] f16
template <bool HALF>
__global__ void blob_dequant_kernel(const uint8_t* __restrict__ blob, uint16_t* __restrict__ gu16,
                                    uint16_t* __restrict__ d16) {
    constexpr size_t O_D_CODES = (size_t) 1280 * 640, O_GU_SC = O_D_CODES + (size_t) 2560 * 160,
                     O_D_SC = O_GU_SC + (size_t) 1280 * 40 * 2;
    const int64_t i = (int64_t) blockIdx.x * blockDim.x + threadIdx.x;   // one thread per 4 weights (one code byte)
    const int64_t n_gu = 1280LL * 640, n_d = 2560LL * 160;
    if (i < n_gu) {
        const int64_t row = i / 640, byte = i % 640;
        const uint8_t c = blob[row * 640 + byte];
        const uint8_t* sp = blob + O_GU_SC + (size_t) (row * 40 + (byte * 4) / 64) * 2;
        const float d = __half2float(__ushort_as_half((uint16_t) (sp[0] | (sp[1] << 8))));
        uint16_t* o = gu16 + row * 2560 + byte * 4;
#pragma unroll
        for (int k = 0; k < 4; ++k) { const float v = (float) (((c >> (2 * k)) & 3) - 1) * d; o[k] = HALF ? hf(v) : bf(v); }
    } else if (i < n_gu + n_d) {
        const int64_t j = i - n_gu, row = j / 160, byte = j % 160;
        const uint8_t c = blob[O_D_CODES + row * 160 + byte];
        const uint8_t* sp = blob + O_D_SC + (size_t) (row * 10 + (byte * 4) / 64) * 2;
        const float d = __half2float(__ushort_as_half((uint16_t) (sp[0] | (sp[1] << 8))));
        uint16_t* o = d16 + row * 640 + byte * 4;
#pragma unroll
        for (int k = 0; k < 4; ++k) { const float v = (float) (((c >> (2 * k)) & 3) - 1) * d; o[k] = HALF ? hf(v) : bf(v); }
    }
}
__global__ void swiglu_il_kernel(const float* __restrict__ gu, uint16_t* __restrict__ h16, int64_t n) {
    const int64_t i = (int64_t) blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n * 640) return;
    const int64_t r = i / 640, k = i % 640;
    const float g = gu[r * 1280 + 2 * k], u = gu[r * 1280 + 2 * k + 1];
    h16[i] = hf_sat(g / (1.0f + __expf(-g)) * u);
}
__global__ void swiglu_pair_kernel(const float* __restrict__ g, const float* __restrict__ u, uint16_t* __restrict__ h16,
                                   int64_t n) {
    const int64_t i = (int64_t) blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n * 640) return;
    const float a = g[i];
    h16[i] = hf_sat(a / (1.0f + __expf(-a)) * u[i]);
}
__global__ void gather_rows16_kernel(const uint16_t* __restrict__ x, const int32_t* __restrict__ src,
                                     uint16_t* __restrict__ dst, int64_t n, int64_t width) {
    const int64_t i = (int64_t) blockIdx.x * blockDim.x + threadIdx.x;   // one uint4 (8 bf16)
    const int64_t per = width / 8;
    if (i >= n * per) return;
    const int64_t r = i / per, j = i % per;
    reinterpret_cast<uint4*>(dst)[r * per + j] = reinterpret_cast<const uint4*>(x)[(int64_t) src[r] * per + j];
}
__global__ void moe_combine_kernel(const float* __restrict__ Dm, const int32_t* __restrict__ slot,
                                   const float* __restrict__ w, const float* __restrict__ shared,
                                   const float* __restrict__ sg, float* __restrict__ bo, int64_t T) {
    const int64_t i = (int64_t) blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= T * N) return;
    const int64_t t = i / N, d = i % N;
    float s = 0.0f;
#pragma unroll
    for (int k = 0; k < 10; ++k) s = fmaf(w[t * 10 + k], Dm[(int64_t) slot[t * 10 + k] * N + d], s);
    bo[i] = s + shared[i] * sigm(sg[t]);
}

// W (X5): four columns per thread, one float4 load per slot; the per-element expression and the k order are unchanged
// (bitwise equal to moe_combine_kernel). Needs 16-byte aligned Dm / shared / bo (N * 4 bytes per row is a multiple of 16).
__global__ void moe_combine4_kernel(const float* __restrict__ Dm, const int32_t* __restrict__ slot,
                                    const float* __restrict__ w, const float4* __restrict__ shared,
                                    const float* __restrict__ sg, float4* __restrict__ bo, int64_t T) {
    const int64_t i = (int64_t) blockIdx.x * blockDim.x + threadIdx.x;   // float4 index
    if (i >= T * (N / 4)) return;
    const int64_t t = i / (N / 4), d = (i % (N / 4)) * 4;
    float4 s = make_float4(0.0f, 0.0f, 0.0f, 0.0f);
    float4 v[10];
    float ww[10];
#pragma unroll
    for (int k = 0; k < 10; ++k) {
        ww[k] = w[t * 10 + k];
        v[k] = *reinterpret_cast<const float4*>(Dm + (int64_t) slot[t * 10 + k] * N + d);
    }
#pragma unroll
    for (int k = 0; k < 10; ++k) {
        s.x = fmaf(ww[k], v[k].x, s.x); s.y = fmaf(ww[k], v[k].y, s.y);
        s.z = fmaf(ww[k], v[k].z, s.z); s.w = fmaf(ww[k], v[k].w, s.w);
    }
    const float g = sigm(sg[t]);
    const float4 sh = shared[i];
    bo[i] = make_float4(s.x + sh.x * g, s.y + sh.y * g, s.z + sh.z * g, s.w + sh.w * g);
}

// ---------------------------------------------------------------- QSA helpers
__global__ void rms_rows_kernel(float* __restrict__ x, const float* __restrict__ w, int64_t cols, int64_t ld, float eps) {
    __shared__ float sh[32];
    float* r = x + (int64_t) blockIdx.x * ld;
    float ss = 0.0f;
    for (int64_t c = threadIdx.x; c < cols; c += blockDim.x) ss += r[c] * r[c];
    const float s = rsqrtf(block_sum(ss, sh) / (float) cols + eps);
    __syncthreads();
    for (int64_t c = threadIdx.x; c < cols; c += blockDim.x) r[c] = s * r[c] * w[c];
}
// TAB (#280, STRATA_ROPE_TABLE=1): the angles from the session's float64 table.  The host launches <false> whenever
// no table applies - the default - so the default kernel is 0.1.31's code exactly (the table read is not in it;
// with it merely skipped at run time, the compiled default path changed its results).
template <bool TAB>
__global__ void rope_kernel(float* __restrict__ x, int64_t heads, int64_t dim, int64_t ld, int64_t pos0,
                            float theta_scale, float freq_scale, float corr_low, float corr_high,
                            float ext_factor, float mscale, const int32_t* __restrict__ mtab,
                            strata::kernels::RopeTab rt) {
    const int64_t row = blockIdx.x;             // t * heads + h
    const int pair = threadIdx.x;               // 0..31
    const int64_t t = row / heads, h = row % heads;
    float* p = x + t * ld + h * dim;
    float c, s;
    if (!(TAB && strata::kernels::rope_tab_cs(rt, strata::kernels::mrope_pos(mtab, (int) (pos0 + t), pair), pair, c,
                                              s))) {
        const float theta_extrap =
            (float) strata::kernels::mrope_pos(mtab, (int) (pos0 + t), pair) * powf(theta_scale, (float) pair);
        strata::kernels::rope_scaled_angle(theta_extrap, freq_scale, corr_low, corr_high, ext_factor, mscale,
                                           pair, c, s);
    }
    const float a = p[pair], b = p[pair + 32];
    p[pair] = a * c - b * s;
    p[pair + 32] = a * s + b * c;
}
__global__ void split_q_kernel(const float* __restrict__ qf, float* __restrict__ q, int64_t T) {
    const int64_t i = (int64_t) blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= T * 24 * 256) return;
    const int64_t t = i / (24 * 256), h = (i / 256) % 24, d = i % 256;
    q[i] = qf[t * 24 * 512 + h * 512 + d];
}
__global__ void gate_attn_kernel(const float* __restrict__ a, const float* __restrict__ qf, uint16_t* __restrict__ o16,
                                 int64_t T, int64_t ld16) {
    const int64_t i = (int64_t) blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= T * 24 * 256) return;
    const int64_t t = i / (24 * 256), h = (i / 256) % 24, d = i % 256;
    o16[t * ld16 + h * 256 + d] = hf(a[i] * (1.0f / (1.0f + expf(-qf[t * 24 * 512 + h * 512 + 256 + d]))));
}

// one block per (token, kv head, 64-value group); 64 threads. KV streaming: the pool page only if the block is
// resident (table >= 0), and the host copy and the prompt path's staging pool (both identity layout) when given.
__global__ void kv_append_kernel(const float* __restrict__ K, const float* __restrict__ V, int64_t pos0,
                                 const int32_t* __restrict__ table, int64_t page_size, uint16_t* k_pool,
                                 uint16_t* v_pool, int8_t* k_q, int8_t* v_q, uint16_t* k_scale, uint16_t* v_scale,
                                 strata::kernels::KvHostPools host, strata::kernels::KvHostPools stage) {
    const int64_t t = blockIdx.x;
    const int kvh = blockIdx.y, g = blockIdx.z >> 1;
    const bool is_v = (blockIdx.z & 1) != 0;
    const int d = g * 64 + threadIdx.x;
    const float x = (is_v ? V : K)[t * 512 + kvh * 256 + d];
    const int64_t pos = pos0 + t;
    const int64_t page = table[pos / page_size];
    const int64_t row = (page * 2 + kvh) * page_size + pos % page_size;
    const int64_t row_id = ((pos / page_size) * 2 + kvh) * page_size + pos % page_size;
    if (k_pool != nullptr) {
        const uint16_t h = hf(x);
        if (page >= 0) (is_v ? v_pool : k_pool)[row * 256 + d] = h;
        if (host.k_pool != nullptr) (is_v ? host.v_pool : host.k_pool)[row_id * 256 + d] = h;
        if (stage.k_pool != nullptr) (is_v ? stage.v_pool : stage.k_pool)[row_id * 256 + d] = h;
        return;
    }
    float a = fabsf(x);
    for (int o = 16; o > 0; o >>= 1) a = fmaxf(a, __shfl_xor_sync(0xffffffffu, a, o));
    __shared__ float wm[2];
    if ((threadIdx.x & 31) == 0) wm[threadIdx.x >> 5] = a;
    __syncthreads();
    const float amax = fmaxf(wm[0], wm[1]);
    const uint16_t sb = hf(amax / 127.0f);
    const float sf = __half2float(__ushort_as_half(sb));
    int q = 0;
    if (sf > 0.0f) { q = __float2int_rn(x / sf); q = q < -127 ? -127 : (q > 127 ? 127 : q); }
    if (page >= 0) {
        (is_v ? v_q : k_q)[row * 256 + d] = (int8_t) q;
        if (threadIdx.x == 0) (is_v ? v_scale : k_scale)[row * 4 + g] = sb;
    }
    if (host.k_q != nullptr) {
        (is_v ? host.v_q : host.k_q)[row_id * 256 + d] = (int8_t) q;
        if (threadIdx.x == 0) (is_v ? host.v_scale : host.k_scale)[row_id * 4 + g] = sb;
    }
    if (stage.k_q != nullptr) {
        (is_v ? stage.v_q : stage.k_q)[row_id * 256 + d] = (int8_t) q;
        if (threadIdx.x == 0) (is_v ? stage.v_scale : stage.k_scale)[row_id * 4 + g] = sb;
    }
}
__global__ void to_f16_kernel(const float* __restrict__ x, uint16_t* __restrict__ y, int64_t n) {
    for (int64_t i = (int64_t) blockIdx.x * blockDim.x + threadIdx.x; i < n; i += (int64_t) gridDim.x * blockDim.x)
        y[i] = hf(x[i]);
}
__global__ void round_f16_kernel(const float* __restrict__ x, float* __restrict__ y, int64_t n) {
    for (int64_t i = (int64_t) blockIdx.x * blockDim.x + threadIdx.x; i < n; i += (int64_t) gridDim.x * blockDim.x)
        y[i] = __half2float(__float2half_rn(x[i]));
}
__global__ void to_bf16_kernel(const float* __restrict__ x, uint16_t* __restrict__ y, uint16_t* __restrict__ ylo,
                               int64_t n) {
    for (int64_t i = (int64_t) blockIdx.x * blockDim.x + threadIdx.x; i < n; i += (int64_t) gridDim.x * blockDim.x) {
        const uint16_t h = act16(x[i]);
        y[i] = h;
        if (ylo) ylo[i] = bf_lo(x[i], h);
    }
}

// S23 (opt-in STRATA_HC_UPMIX=1, gfx11): the hyper-connection read's up projection with gr_mix_r as its epilogue.  The
// up GEMM's FP32 output `gated` (T x 10240: 671 MB per 16K chunk) is never written or read back: a block computes a
// tile of 64 tokens x 64 columns for all 4 streams on the matrix cores (BF16 in, FP32 accumulate over K = 320; the
// lo16 and w_up tiles staged in LDS), and each lane already holds the 4 streams' values of its (token, column) - the
// mix (x_c = R * rs * w, sum of x_c * sigmoid(gated_c), / 4) runs in registers, in gr_mix_r_kernel's order.  The
// GEMM sums K in another order than hipBLASLt: rounding-level, hence opt-in and quality-gated.
#if defined(__HIPCC__)
typedef __bf16 um_b16 __attribute__((ext_vector_type(16)));
typedef float um_f8 __attribute__((ext_vector_type(8)));
__device__ __forceinline__ um_f8 um_wmma(um_b16 a, um_b16 b, um_f8 c) {
#if defined(__gfx1100__) || defined(__gfx1101__) || defined(__gfx1102__) || defined(__gfx1150__) || defined(__gfx1151__)
    return __builtin_amdgcn_wmma_f32_16x16x16_bf16_w32(a, b, c);
#else
    (void) a; (void) b; return c;   // not a gfx11 device pass: never launched (runtime gate)
#endif
}
constexpr int UM_T = 128, UM_D = 16, UM_LD = LR + 8;   // a block: 128 tokens x 16 columns x the 4 streams
__device__ __forceinline__ um_b16 um_frag(const uint16_t* p) {   // 16 BF16 (32 bytes, 16-byte aligned)
    const uint4 a = *reinterpret_cast<const uint4*>(p), b = *reinterpret_cast<const uint4*>(p + 8);
    const uint32_t w[8] = {a.x, a.y, a.z, a.w, b.x, b.y, b.z, b.w};
    return __builtin_bit_cast(um_b16, w);
}
// The block's w_up rows (16 columns x 4 streams x all 320 k) are staged in LDS once; each wave streams its own 16
// tokens' lo16 rows from global memory (every A fragment feeds the 4 streams' WMMAs) - one barrier per block.
__global__ void __launch_bounds__(256) gr_upmix_kernel(const uint16_t* __restrict__ lo16, const uint16_t* __restrict__ wu,
                                                       const float* __restrict__ R, const float* __restrict__ rs,
                                                       const float* __restrict__ w, float* __restrict__ mixed,
                                                       uint16_t* __restrict__ mixed16, uint16_t* __restrict__ mixed_h,
                                                       int64_t T) {
    __shared__ __align__(16) uint16_t sB[HC * UM_D][UM_LD];
    const int tid = threadIdx.x, lane = tid & 31, wave = tid >> 5, l16 = lane & 15, hi = lane >> 4;
    // blocks ordered column-tile fastest: the blocks in flight share a token tile, so the epilogue's R reads run
    // along rows (token-tile fastest scattered them over 512 rows x 40 KB per block: 28 ms instead of 3.7 at 8K)
    const int64_t t0 = (int64_t) blockIdx.y * UM_T + 16 * wave;
    const int d0 = blockIdx.x * UM_D;
    for (int i = tid; i < HC * UM_D * (LR / 8); i += 256) {
        const int rr = i / (LR / 8), q = i % (LR / 8);
        const int c = rr / UM_D, dd = rr % UM_D;
        *reinterpret_cast<uint4*>(&sB[rr][8 * q]) = *reinterpret_cast<const uint4*>(wu + (size_t) (c * N + d0 + dd) * LR + 8 * q);
    }
    __syncthreads();
    if (t0 >= T) return;
    const int64_t ta = t0 + l16 < T ? t0 + l16 : T - 1;
    const uint16_t* arow = lo16 + ta * LR;
    um_f8 acc[HC];
#pragma unroll
    for (int c = 0; c < HC; ++c) acc[c] = um_f8{0.f, 0.f, 0.f, 0.f, 0.f, 0.f, 0.f, 0.f};
#pragma unroll
    for (int ks = 0; ks < LR; ks += 16) {
        const um_b16 a = um_frag(arow + ks);
#pragma unroll
        for (int c = 0; c < HC; ++c) acc[c] = um_wmma(a, um_frag(&sB[c * UM_D + l16][ks]), acc[c]);
    }
    const int d = d0 + l16;
#pragma unroll
    for (int i = 0; i < 8; ++i) {
        const int64_t t = t0 + 2 * i + hi;
        if (t >= T) continue;
        float s = 0.0f;
#pragma unroll
        for (int c = 0; c < HC; ++c) {
            const float x = R[t * D + c * N + d] * rs[t * HC + c] * w[c * N + d];   // gr_mix_r_kernel's order
            s = fmaf(x, sigm(acc[c][i]), s);
        }
        s /= (float) HC;
        mixed[t * N + d] = s;
        if (mixed16) mixed16[t * N + d] = act16(s);
        if (mixed_h) mixed_h[t * N + d] = hf(s);
    }
}
#endif
}  // namespace

void kv_append(const float* K, const float* V, int64_t T, int64_t pos0, const int32_t* page_table, int64_t page_size,
               uint16_t* k_pool, uint16_t* v_pool, int8_t* k_q, int8_t* v_q, uint16_t* k_scale, uint16_t* v_scale,
               void* stream, const strata::kernels::KvHostPools* host, const strata::kernels::KvHostPools* stage) {
    if (T <= 0) return;
    kv_append_kernel<<<dim3((unsigned) T, 2, 8), 64, 0, (cudaStream_t) stream>>>(
        K, V, pos0, page_table, page_size, k_pool, v_pool, k_q, v_q, k_scale, v_scale,
        host ? *host : strata::kernels::KvHostPools{}, stage ? *stage : strata::kernels::KvHostPools{});
    check("kv_append");
}
void set_act_f16(bool on) {
#if !defined(__HIPCC__)
    (void) on;   // CUDA: the activation image is always BF16
    return;
#else
    const int v = on ? 1 : 0;
    if (cudaMemcpyToSymbol(g_act_f16, &v, sizeof v) != cudaSuccess) {
        std::fprintf(stderr, "prefill: setting the FP16 activation image failed: %s\n", cudaGetErrorString(cudaGetLastError()));
        std::exit(1);
    }
#endif
}
void to_f16(const float* x, uint16_t* y, int64_t n, void* stream) {
    if (n <= 0) return;
    to_f16_kernel<<<(unsigned) ((n + 255) / 256 < 4096 ? (n + 255) / 256 : 4096), 256, 0, (cudaStream_t) stream>>>(x, y, n);
    check("to_f16");
}
void round_f16(const float* x, float* y, int64_t n, void* stream) {
    if (n <= 0) return;
    round_f16_kernel<<<(unsigned) ((n + 255) / 256 < 4096 ? (n + 255) / 256 : 4096), 256, 0, (cudaStream_t) stream>>>(x, y, n);
    check("round_f16");
}
void to_bf16(const float* x, uint16_t* y, int64_t n, void* stream, uint16_t* ylo) {
    if (n <= 0) return;
    to_bf16_kernel<<<(unsigned) ((n + 255) / 256 < 4096 ? (n + 255) / 256 : 4096), 256, 0, (cudaStream_t) stream>>>(x, y, ylo, n);
    check("to_bf16");
}

void gr_norm(const float* R, const float* w_norm, float eps, float* xn, uint16_t* xn16, int64_t T, void* stream,
             uint16_t* xn16_lo) {
    gr_norm_kernel<<<(unsigned) (T * HC), 256, 0, (cudaStream_t) stream>>>(R, w_norm, eps, xn, xn16, xn16_lo);
    check("gr_norm");
}
void gr_norm_rs(const float* R, const float* w_norm, float eps, float* rs, uint16_t* xn16, int64_t T, void* stream,
                uint16_t* xn16_lo, int64_t ldx) {
    gr_norm_rs_kernel<<<(unsigned) (T * HC), 256, 0, (cudaStream_t) stream>>>(R, w_norm, eps, rs, xn16, xn16_lo,
                                                                                ldx > 0 ? ldx : D);
    check("gr_norm_rs");
}
void gr_mix_r(const float* R, const float* rs, const float* w_norm, const float* gated, float* mixed, uint16_t* mixed16,
              int64_t T, void* stream, uint16_t* mixed_h, uint16_t* mixed16_lo) {
    gr_mix_r_kernel<<<blocks_for(T * N), 256, 0, (cudaStream_t) stream>>>(R, rs, w_norm, gated, mixed, mixed16, T,
                                                                          mixed_h, mixed16_lo);
    check("gr_mix_r");
}
bool gr_upmix(const uint16_t* lo16, const uint16_t* w_up, const float* R, const float* rs, const float* w_norm,
              float* mixed, uint16_t* mixed16, uint16_t* mixed_h, int64_t T, void* stream) {
#if defined(__HIPCC__)
    static const bool gfx11 = [] {
        int dev = 0;
        cudaDeviceProp p;
        if (cudaGetDevice(&dev) != cudaSuccess || cudaGetDeviceProperties(&p, dev) != cudaSuccess) return false;
        return strata::kernels::gfx_arch_is_gfx11_wmma(p.gcnArchName);
    }();
    if (!gfx11 || T <= 0) return false;
    gr_upmix_kernel<<<dim3(N / UM_D, (unsigned) ((T + UM_T - 1) / UM_T)), 256, 0, (cudaStream_t) stream>>>(
        lo16, w_up, R, rs, w_norm, mixed, mixed16, mixed_h, T);
    check("gr_upmix");
    return true;
#else
    (void) lo16; (void) w_up; (void) R; (void) rs; (void) w_norm; (void) mixed; (void) mixed16; (void) mixed_h;
    (void) T; (void) stream;
    return false;
#endif
}
void gr_write_norm_rs(float* R, const float* bo, const float* inj, int64_t inj_ld, const float* w_norm_next, float eps,
                      float* rs, uint16_t* xn16, int64_t T, void* stream, uint16_t* xn16_lo, int64_t ldx) {
    gr_write_norm_rs_kernel<<<(unsigned) (T * HC), 256, 0, (cudaStream_t) stream>>>(R, bo, inj, inj_ld, w_norm_next, eps,
                                                                                      rs, xn16, xn16_lo,
                                                                                      ldx > 0 ? ldx : D);
    check("gr_write_norm_rs");
}
void gr_write_cvec_norm_rs(float* R, const float* bo, const float* inj, int64_t inj_ld, const float* v_l,
                           const float* s_l, const int* on, int mode, const float* w_norm_next, float eps, float* rs,
                           uint16_t* xn16, int64_t T, void* stream, uint16_t* xn16_lo, int64_t ldx) {
    gr_write_cvec_norm_rs_kernel<<<(unsigned) (T * HC), 256, 0, (cudaStream_t) stream>>>(
        R, bo, inj, inj_ld, v_l, s_l, on, mode, w_norm_next, eps, rs, xn16, xn16_lo, ldx > 0 ? ldx : D);
    check("gr_write_cvec_norm_rs");
}
void gr_silu(const float* lo, uint16_t* lo16, int64_t T, void* stream, uint16_t* lo16_lo) {
    gr_silu_kernel<<<blocks_for(T * LR), 256, 0, (cudaStream_t) stream>>>(lo, lo16, lo16_lo, T * LR);
    check("gr_silu");
}
void gr_mix(const float* xn, const float* gated, float* mixed, uint16_t* mixed16, int64_t T, void* stream,
            uint16_t* mixed_h, uint16_t* mixed16_lo) {
    gr_mix_kernel<<<blocks_for(T * N), 256, 0, (cudaStream_t) stream>>>(xn, gated, mixed, mixed16, T, mixed_h, mixed16_lo);
    check("gr_mix");
}
void gr_write(float* R, const float* bo, const float* inj, int64_t inj_ld, int64_t T, void* stream) {
    gr_write_kernel<<<blocks_for(T * D), 256, 0, (cudaStream_t) stream>>>(R, bo, inj, inj_ld, T);
    check("gr_write");
}
void gr_broadcast(const float* e, float* R, int64_t T, void* stream) {
    gr_broadcast_kernel<<<blocks_for(T * D), 256, 0, (cudaStream_t) stream>>>(e, R, T);
    check("gr_broadcast");
}
void gdn_gates(const float* ab, const float* dt, const float* ssm_a, float* gate, float* beta, int64_t T, void* stream) {
    gdn_gates_kernel<<<blocks_for(T * HV), 256, 0, (cudaStream_t) stream>>>(ab, dt, ssm_a, gate, beta, T);
    check("gdn_gates");
}
void gdn_conv(float* history, const float* qkv, const float* conv_w, float* h, int64_t T, float eps, void* stream) {
    static const bool serial = std::getenv("STRATA_GDN_CONV_SERIAL") != nullptr;   // the old walk (A/B)
    if (serial || T <= CONV_TILE) {
        gdn_conv_kernel<<<C / 128, 128, 0, (cudaStream_t) stream>>>(history, qkv, conv_w, h, T);
    } else {
        static const bool fused = [] { const char* v = std::getenv("STRATA_GDN_CONVL2"); return v && std::atoi(v) != 0; }();
        if (fused) {
            gdn_conv_l2_kernel<<<dim3(C / 128, (unsigned) ((T + CONV_TILE - 1) / CONV_TILE)), 128, 0,
                                 (cudaStream_t) stream>>>(history, qkv, conv_w, h, T, eps);
            gdn_conv_hist_kernel<<<C / 128, 128, 0, (cudaStream_t) stream>>>(history, qkv, T);
            check("gdn_conv");
            return;
        }
        gdn_conv_tiled_kernel<<<dim3(C / 128, (unsigned) ((T + CONV_TILE - 1) / CONV_TILE)), 128, 0,
                                (cudaStream_t) stream>>>(history, qkv, conv_w, h, T);
        gdn_conv_hist_kernel<<<C / 128, 128, 0, (cudaStream_t) stream>>>(history, qkv, T);
    }
    gdn_l2_kernel<<<dim3(2 * HK, (unsigned) T), S, 0, (cudaStream_t) stream>>>(h, eps);
    check("gdn_conv");
}
void gdn_recurrence_variant(int variant, float* state, const float* h, const float* gate, const float* beta,
                            const float* z, const float* gamma, float eps, float* y, uint16_t* y16, int64_t T,
                            void* stream, int64_t ld16) {
    if (ld16 <= 0) ld16 = (int64_t) HV * S;
#if defined(__HIPCC__)
    if (variant == 3) {   // diagnostics: the quad recurrence + the old norm kernel
        if (T > 0) {
            gdn_rec_quad_kernel<<<HV * (S / QCB), QTH, 0, (cudaStream_t) stream>>>(state, h, gate, beta, y, T);
            gdn_out_norm_kernel<<<dim3((unsigned) T, HV), S, 0, (cudaStream_t) stream>>>(z, gamma, eps, y, y16, ld16);
        }
        return;
    }
    if (variant == 4 || variant == 5) {   // diagnostics: the quad recurrence's PP / two-column kernels + the norm
        if (T > 0) {
            if (variant == 5)
                gdn_rec_quad2c_kernel<<<HV * (S / C2CB), QTH, 0, (cudaStream_t) stream>>>(state, h, gate, beta, y, T);
            else
                gdn_rec_quad_pp_kernel<<<HV * (S / QCB), QTH, 0, (cudaStream_t) stream>>>(state, h, gate, beta, y, T);
            gdn_out_norm_loop_kernel<true><<<(unsigned) std::min<int64_t>(T * HV, 4096), S, 0, (cudaStream_t) stream>>>(
                z, gamma, eps, y, y16, T * HV, ld16);
        }
        check("gdn_recurrence (quad pp)");
        return;
    }
    if (variant == 1) {
        if (T > 0) {
            static const int pp = [] { const char* v = std::getenv("STRATA_GDN_PP"); return v ? std::atoi(v) : 0; }();
            if (pp == 2)
                gdn_rec_quad2c_kernel<<<HV * (S / C2CB), QTH, 0, (cudaStream_t) stream>>>(state, h, gate, beta, y, T);
            else if (pp)
                gdn_rec_quad_pp_kernel<<<HV * (S / QCB), QTH, 0, (cudaStream_t) stream>>>(state, h, gate, beta, y, T);
            else
                gdn_rec_quad_kernel<<<HV * (S / QCB), QTH, 0, (cudaStream_t) stream>>>(state, h, gate, beta, y, T);
            static const bool noy = [] { const char* v = std::getenv("STRATA_GDN_NOY"); return v && std::atoi(v) != 0; }();
            if (noy)
                gdn_out_norm_loop_kernel<false><<<(unsigned) std::min<int64_t>(T * HV, 4096), S, 0, (cudaStream_t) stream>>>(
                    z, gamma, eps, y, y16, T * HV, ld16);
            else
                gdn_out_norm_loop_kernel<true><<<(unsigned) std::min<int64_t>(T * HV, 4096), S, 0, (cudaStream_t) stream>>>(
                    z, gamma, eps, y, y16, T * HV, ld16);
        }
        check("gdn_recurrence (quad)");
        return;
    }
#else
    (void) variant;   // the quad recurrence kernels are AMD-only
#endif
    static const bool serial = std::getenv("STRATA_GDN_REC_HEADS") != nullptr;   // the one-block-per-head kernel (A/B)
    if (serial || T <= 0 || variant == 2) {
        gdn_rec_kernel<<<HV, dim3(S, RG), 0, (cudaStream_t) stream>>>(state, h, gate, beta, z, gamma, eps, y, y16, T, ld16);
    } else {
        static const bool pipe = [] { const char* v = std::getenv("STRATA_GDN_PIPELINE"); return v == nullptr || std::atoi(v) != 0; }();
#if !defined(__HIPCC__)
        if (pipe && gdn_keyhead_ok())   // the value heads of a key head in one thread (same bits)
            gdn_rec_kh_kernel<<<HK * NCB, dim3(CB, RG), 0, (cudaStream_t) stream>>>(state, h, gate, beta, y, T);
        else
#endif
        if (pipe)   // the software-pipelined loads (same bits)
            gdn_rec_cols_pipe_kernel<<<HV * NCB, dim3(CB, RG), 0, (cudaStream_t) stream>>>(state, h, gate, beta, y, T);
        else
            gdn_rec_cols_kernel<<<HV * NCB, dim3(CB, RG), 0, (cudaStream_t) stream>>>(state, h, gate, beta, y, T);
        gdn_out_norm_kernel<<<dim3((unsigned) T, HV), S, 0, (cudaStream_t) stream>>>(z, gamma, eps, y, y16, ld16);
    }
    check("gdn_recurrence");
}
void gdn_recurrence(float* state, const float* h, const float* gate, const float* beta, const float* z,
                    const float* gamma, float eps, float* y, uint16_t* y16, int64_t T, void* stream, int64_t ld16) {
    // Aurora (S23): four lanes per column + the grid-stride norm, the same bits (tests/hip/gdn_rec_head.cpp);
    // STRATA_GDN_HEAD=1 (on by default on gfx1151; off with STRATA_GDN_HEAD=0 or STRATA_GDN_REC_HEADS) takes them
    // (AMD builds only; the arch defaults set STRATA_GDN_HEAD=1 on gfx1151 and leave other cards on the earlier kernels)
    static const bool head = [] { const char* v = std::getenv("STRATA_GDN_HEAD"); return v != nullptr && std::atoi(v) != 0; }();
    static const bool serial = std::getenv("STRATA_GDN_REC_HEADS") != nullptr;
    gdn_recurrence_variant(head && !serial ? 1 : 0, state, h, gate, beta, z, gamma, eps, y, y16, T, stream, ld16);
}
void route(const float* logits, int32_t* ids, float* weights, int64_t T, int64_t n_expert, void* stream) {
    if (n_expert == 512)
        route_kernel<16><<<(unsigned) ((T + 7) / 8), 256, 0, (cudaStream_t) stream>>>(logits, ids, weights, T);
    else if (n_expert == 256)
        route_kernel<8><<<(unsigned) ((T + 7) / 8), 256, 0, (cudaStream_t) stream>>>(logits, ids, weights, T);
    else
        strata::kernels::router_top10(logits, (int) T, (int) n_expert, 10, ids, weights, stream);
    check("route");
}
void blob_dequant(const uint8_t* blob, uint16_t* gu16, uint16_t* down16, void* stream) {
    blob_dequant_kernel<false><<<blocks_for(1280LL * 640 + 2560LL * 160), 256, 0, (cudaStream_t) stream>>>(blob, gu16, down16);
    check("blob_dequant");
}
void blob_dequant_f16(const uint8_t* blob, uint16_t* gu16, uint16_t* down16, void* stream) {
    blob_dequant_kernel<true><<<blocks_for(1280LL * 640 + 2560LL * 160), 256, 0, (cudaStream_t) stream>>>(blob, gu16, down16);
    check("blob_dequant_f16");
}
void swiglu_interleaved(const float* gu, uint16_t* h16, int64_t n, void* stream) {
    if (n <= 0) return;
    swiglu_il_kernel<<<blocks_for(n * 640), 256, 0, (cudaStream_t) stream>>>(gu, h16, n);
    check("swiglu_interleaved");
}
void swiglu_pair(const float* g, const float* u, uint16_t* h16, int64_t n, void* stream) {
    swiglu_pair_kernel<<<blocks_for(n * 640), 256, 0, (cudaStream_t) stream>>>(g, u, h16, n);
    check("swiglu_pair");
}
namespace {
__global__ void copy_i32_kernel(int32_t* __restrict__ dst, const int32_t* __restrict__ src, int64_t n) {
    for (int64_t i = (int64_t) blockIdx.x * blockDim.x + threadIdx.x; i < n; i += (int64_t) gridDim.x * blockDim.x)
        dst[i] = src[i];
}
}  // namespace
namespace {
__global__ void copy_f4_kernel(float4* __restrict__ dst, const float4* __restrict__ src, int64_t n4) {
    for (int64_t i = (int64_t) blockIdx.x * blockDim.x + threadIdx.x; i < n4; i += (int64_t) gridDim.x * blockDim.x)
        dst[i] = src[i];
}
}  // namespace
void copy_f32_wide(float* dst, const float* src, int64_t n, void* stream) {
    if (n <= 0) return;
    const int64_t n4 = n / 4, b = std::min<int64_t>((n4 + 255) / 256, 4096);
    if (n4 > 0) copy_f4_kernel<<<(unsigned) b, 256, 0, (cudaStream_t) stream>>>((float4*) dst, (const float4*) src, n4);
    if (n % 4) copy_i32((int32_t*) dst + n4 * 4, (const int32_t*) src + n4 * 4, n % 4, stream);
    check("copy_f32_wide");
}
void copy_i32(int32_t* dst, const int32_t* src, int64_t n, void* stream) {
    if (n <= 0) return;
    const int64_t b = (n + 255) / 256;
    copy_i32_kernel<<<(unsigned) (b < 256 ? b : 256), 256, 0, (cudaStream_t) stream>>>(dst, src, n);
    check("copy_i32");
}
void gather_rows16(const uint16_t* x16, const int32_t* src, uint16_t* dst16, int64_t n, int64_t width, void* stream) {
    if (n <= 0) return;
    gather_rows16_kernel<<<blocks_for(n * (width / 8)), 256, 0, (cudaStream_t) stream>>>(x16, src, dst16, n, width);
    check("gather_rows16");
}
namespace {
__global__ void moe_combine_peer_kernel(const float* __restrict__ Dm, const int32_t* __restrict__ slot,
                                        const float* __restrict__ w, const float* __restrict__ shared,
                                        const float* __restrict__ sg, const float* __restrict__ peer, int64_t rows_local,
                                        float* __restrict__ bo, int64_t T) {
    const int64_t i = (int64_t) blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= T * N) return;
    const int64_t t = i / N, d = i % N;
    float s = 0.0f;
#pragma unroll
    for (int k = 0; k < 10; ++k) {
        const int32_t r = slot[t * 10 + k];
        if (r < rows_local) s = fmaf(w[t * 10 + k], Dm[(int64_t) r * N + d], s);
    }
    bo[i] = (s + peer[i]) + shared[i] * sigm(sg[t]);
}
__global__ void peer_scatter_add_kernel(float* __restrict__ sum, const float* __restrict__ rows,
                                        const float* __restrict__ wk, const int32_t* __restrict__ pair, int64_t n) {
    const int64_t i = (int64_t) blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n * N) return;
    const int64_t r = i / N, d = i % N;
    const int32_t p = pair[r];
    float* o = sum + (int64_t) (p / 10) * N + d;
    *o = fmaf(wk[p], rows[i], *o);
}
__global__ void moe_combine_peer16_kernel(const float* __restrict__ Dm, const int32_t* __restrict__ slot,
                                          const float* __restrict__ w, const float* __restrict__ shared,
                                          const float* __restrict__ sg, const uint16_t* __restrict__ peer16,
                                          int64_t rows_local, float* __restrict__ bo, int64_t T) {
    const int64_t i = (int64_t) blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= T * N) return;
    const int64_t t = i / N, d = i % N;
    float s = 0.0f;
#pragma unroll
    for (int k = 0; k < 10; ++k) {
        const int32_t r = slot[t * 10 + k];
        if (r < rows_local) s = fmaf(w[t * 10 + k], Dm[(int64_t) r * N + d], s);
    }
    bo[i] = (s + __half2float(__ushort_as_half(peer16[i]))) + shared[i] * sigm(sg[t]);
}
__global__ void sums_to_f16_kernel(const float* __restrict__ x, uint16_t* __restrict__ y, int64_t n) {
    const int64_t i = (int64_t) blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) y[i] = hf_sat(x[i]);
}
__global__ void f16_to_f32_wide_kernel(float4* __restrict__ dst, const uint4* __restrict__ src, int64_t n8) {
    const int64_t i = (int64_t) blockIdx.x * blockDim.x + threadIdx.x;   // eight values
    if (i >= n8) return;
    const uint4 v = src[i];
    const __half2* h = reinterpret_cast<const __half2*>(&v);
    const float2 a = __half22float2(h[0]), b = __half22float2(h[1]), c = __half22float2(h[2]), e = __half22float2(h[3]);
    dst[2 * i] = make_float4(a.x, a.y, b.x, b.y);
    dst[2 * i + 1] = make_float4(c.x, c.y, e.x, e.y);
}
}  // namespace
void moe_combine_peer16(const float* Dm, const int32_t* slot, const float* w, const float* shared, const float* sg,
                        const uint16_t* peer16, int64_t rows_local, float* bo, int64_t T, void* stream) {
    moe_combine_peer16_kernel<<<blocks_for(T * N), 256, 0, (cudaStream_t) stream>>>(Dm, slot, w, shared, sg, peer16,
                                                                                      rows_local, bo, T);
    check("moe_combine_peer16");
}
void sums_to_f16(const float* x, uint16_t* y, int64_t n, void* stream) {
    if (n <= 0) return;
    sums_to_f16_kernel<<<blocks_for(n), 256, 0, (cudaStream_t) stream>>>(x, y, n);
    check("sums_to_f16");
}
void f16_to_f32_wide(float* dst, const uint16_t* src, int64_t n, void* stream) {
    if (n <= 0) return;
    f16_to_f32_wide_kernel<<<blocks_for(n / 8), 256, 0, (cudaStream_t) stream>>>(reinterpret_cast<float4*>(dst),
                                                                                    reinterpret_cast<const uint4*>(src), n / 8);
    check("f16_to_f32_wide");
}
void moe_combine_peer(const float* Dm, const int32_t* slot, const float* w, const float* shared, const float* sg,
                      const float* peer, int64_t rows_local, float* bo, int64_t T, void* stream) {
    moe_combine_peer_kernel<<<blocks_for(T * N), 256, 0, (cudaStream_t) stream>>>(Dm, slot, w, shared, sg, peer, rows_local,
                                                                                    bo, T);
    check("moe_combine_peer");
}
void peer_scatter_add(float* sum, const float* rows, const float* wk, const int32_t* pair, int64_t n, void* stream) {
    if (n <= 0) return;
    peer_scatter_add_kernel<<<blocks_for(n * N), 256, 0, (cudaStream_t) stream>>>(sum, rows, wk, pair, n);
    check("peer_scatter_add");
}
void moe_combine(const float* Dm, const int32_t* slot, const float* w, const float* shared, const float* sg, float* bo,
                 int64_t T, void* stream) {
#ifndef STRATA_W_NO_COMB
    if (((reinterpret_cast<uintptr_t>(Dm) | reinterpret_cast<uintptr_t>(shared) | reinterpret_cast<uintptr_t>(bo)) & 15) == 0) {
        moe_combine4_kernel<<<blocks_for(T * (N / 4)), 256, 0, (cudaStream_t) stream>>>(
            Dm, slot, w, reinterpret_cast<const float4*>(shared), sg, reinterpret_cast<float4*>(bo), T);
        check("moe_combine");
        return;
    }
#endif
    moe_combine_kernel<<<blocks_for(T * N), 256, 0, (cudaStream_t) stream>>>(Dm, slot, w, shared, sg, bo, T);
    check("moe_combine");
}
void rms_rows(float* x, const float* w, int64_t rows, int64_t cols, int64_t ld, float eps, void* stream) {
    if (rows <= 0) return;
    rms_rows_kernel<<<(unsigned) rows, 256, 0, (cudaStream_t) stream>>>(x, w, cols, ld, eps);
    check("rms_rows");
}
void rope(float* x, int64_t T, int64_t heads, int64_t dim, int64_t ld, int64_t pos0,
          const strata::kernels::RopeScaling& scaling, void* stream) {
    // The engine validates the resolved config at startup with the same rule (generate.cpp), so this only
    // fires for a caller that bypassed it; the prompt path has no error return here, so it stops the process.
    if (const char* why = strata::kernels::rope_scaling_invalid(scaling)) {
        std::fprintf(stderr, "prefill rope: invalid rope scaling: %s\n", why);
        std::exit(1);
    }
    const float theta_scale = powf((float) scaling.freq_base, -2.0f / 64.0f);
    const strata::kernels::RopeKernelArgs k = scaling.kernel_args(64);   // none: the identity constants
    const strata::kernels::RopeTab rt = strata::kernels::rope_table_for(scaling);
    if (rt.cos != nullptr)
        rope_kernel<true><<<(unsigned) (T * heads), 32, 0, (cudaStream_t) stream>>>(
            x, heads, dim, ld, pos0, theta_scale, k.freq_scale, k.corr_low, k.corr_high, k.ext_factor, k.attn_factor,
            strata::kernels::mrope_table(), rt);
    else
        rope_kernel<false><<<(unsigned) (T * heads), 32, 0, (cudaStream_t) stream>>>(
            x, heads, dim, ld, pos0, theta_scale, k.freq_scale, k.corr_low, k.corr_high, k.ext_factor, k.attn_factor,
            strata::kernels::mrope_table(), rt);
    check("rope");
}
void split_q(const float* q_full, float* q, int64_t T, void* stream) {
    split_q_kernel<<<blocks_for(T * 24 * 256), 256, 0, (cudaStream_t) stream>>>(q_full, q, T);
    check("split_q");
}
void gate_attn(const float* attn, const float* q_full, uint16_t* out16, int64_t T, void* stream, int64_t ld16) {
    gate_attn_kernel<<<blocks_for(T * 24 * 256), 256, 0, (cudaStream_t) stream>>>(attn, q_full, out16, T,
                                                                                ld16 > 0 ? ld16 : 24 * 256);
    check("gate_attn");
}

}  // namespace strata::prefill
