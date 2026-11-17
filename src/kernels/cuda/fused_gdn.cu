// src/kernels/cuda/fused_gdn.cu - see include/strata/kernels/fused_gdn.hpp.
#include "strata/kernels/fused_gdn.hpp"
#include "strata/core/geometry.hpp"

#include <cuda_runtime.h>

#include <cstdio>
#include <cstdlib>

namespace strata::kernels {
namespace {

constexpr int S = 128;          // state size (rows = cols = 128)
static_assert(S == strata::core::kCompiledGeometry.ssm_state_size, "fused_gdn is compiled for core/geometry.hpp");
constexpr int RG = 4;           // row groups
constexpr int RPG = S / RG;     // 32 rows per thread

// SILU: the output gate is SiLU(z) (Qwen3.6, qwen35moe) instead of sigmoid(z) (Flash-Next); a compile-time flag, so
// the sigmoid instantiation is the kernel it was.
template <bool SILU>
__global__ void __launch_bounds__(S * RG) gdn_step_norm_kernel(float* __restrict__ state, const float* __restrict__ q,
                                                               const float* __restrict__ k, const float* __restrict__ v,
                                                               const float* __restrict__ gate,
                                                               const float* __restrict__ beta,
                                                               const float* __restrict__ z,
                                                               const float* __restrict__ gamma, float eps,
                                                               float* __restrict__ y, int h_k, int h_v) {
    __shared__ float sk[S], sq[S];
    __shared__ float red_kv[RG][S];
    __shared__ float red_o[RG][S];
    __shared__ float wsum[S * RG / 32];
    const int head = blockIdx.x;
    const int col = threadIdx.x;          // 0..127
    const int rg = threadIdx.y;           // 0..3
    const int tid = rg * S + col;
    const int qh = head % h_k;
    if (tid < S) { sk[tid] = k[qh * S + tid]; sq[tid] = q[qh * S + tid]; }
    float s[RPG];
    float* base = state + ((size_t) (rg * RPG) * h_v + head) * S + col;
    const size_t row_stride = (size_t) h_v * S;
#pragma unroll
    for (int r = 0; r < RPG; ++r) s[r] = base[r * row_stride];
    __syncthreads();
    const float g = __expf(gate[head]);
    float kv = 0.0f;
#pragma unroll
    for (int r = 0; r < RPG; ++r) kv = fmaf(s[r], sk[rg * RPG + r], kv);
    red_kv[rg][col] = kv;
    __syncthreads();
    const float kv_col = red_kv[0][col] + red_kv[1][col] + red_kv[2][col] + red_kv[3][col];
    const float delta = (v[head * S + col] - g * kv_col) * beta[head];
    float o = 0.0f;
#pragma unroll
    for (int r = 0; r < RPG; ++r) {
        s[r] = fmaf(g, s[r], sk[rg * RPG + r] * delta);
        o = fmaf(s[r], sq[rg * RPG + r], o);
        base[r * row_stride] = s[r];
    }
    red_o[rg][col] = o;
    __syncthreads();
    float oc = 0.0f, sq_part = 0.0f;
    if (rg == 0) {
        oc = (red_o[0][col] + red_o[1][col] + red_o[2][col] + red_o[3][col]) * rsqrtf((float) S);
        sq_part = oc * oc;
    }
    // RMS over the head's 128 outputs: warps of row group 0 are threads 0..127.
    for (int o2 = 16; o2 > 0; o2 >>= 1) sq_part += __shfl_xor_sync(0xffffffffu, sq_part, o2);
    if ((tid & 31) == 0) wsum[tid >> 5] = sq_part;
    __syncthreads();
    if (rg == 0) {
        const float ss = wsum[0] + wsum[1] + wsum[2] + wsum[3];
        const float scale = rsqrtf(ss / (float) S + eps);
        const float zz = z[head * S + col];
        if constexpr (SILU) y[head * S + col] = oc * scale * gamma[col] * (zz / (1.0f + __expf(-zz)));
        else y[head * S + col] = oc * scale * gamma[col] * (1.0f / (1.0f + __expf(-zz)));
    }
}

__global__ void __launch_bounds__(S) gdn_conv_l2_kernel(float* __restrict__ hist, const float* __restrict__ qkv,
                                                        const float* __restrict__ w, float* __restrict__ h,
                                                        int qk_heads, float eps) {
    __shared__ float part[S / 32];
    const int c = blockIdx.x * S + threadIdx.x;
    const float v0 = hist[c * 3], v1 = hist[c * 3 + 1], v2 = hist[c * 3 + 2], x = qkv[c];
    float sum = v0 * w[c * 4] + v1 * w[c * 4 + 1] + v2 * w[c * 4 + 2] + x * w[c * 4 + 3];
    hist[c * 3] = v1;
    hist[c * 3 + 1] = v2;
    hist[c * 3 + 2] = x;
    float y = sum / (1.0f + __expf(-sum));
    if ((int) blockIdx.x < qk_heads) {
        float sq = y * y;
        for (int o = 16; o > 0; o >>= 1) sq += __shfl_xor_sync(0xffffffffu, sq, o);
        if ((threadIdx.x & 31) == 0) part[threadIdx.x >> 5] = sq;
        __syncthreads();
        const float ss = part[0] + part[1] + part[2] + part[3];
        y *= rsqrtf(ss + eps);
    }
    h[c] = y;
}

template <bool SILU>
__global__ void __launch_bounds__(S * RG) gdn_step_norm_batch_kernel(
    float* __restrict__ state, const float* __restrict__ qkv_batch, int qkv_stride,
    const float* __restrict__ gate, const float* __restrict__ beta, const float* __restrict__ z,
    const float* __restrict__ gamma, float eps, float* __restrict__ y, int h_k, int h_v, int n_tokens) {
    __shared__ float sk[S], sq[S];
    __shared__ float red_kv[RG][S];
    __shared__ float red_o[RG][S];
    __shared__ float wsum[S * RG / 32];
    const int head = blockIdx.x;
    const int col = threadIdx.x;
    const int rg = threadIdx.y;
    const int tid = rg * S + col;
    const int qh = head % h_k;
    float s[RPG];
    float* base = state + ((size_t)(rg * RPG) * h_v + head) * S + col;
    const size_t row_stride = (size_t)h_v * S;
#pragma unroll
    for (int r = 0; r < RPG; ++r) s[r] = base[r * row_stride];
    const float gam = (rg == 0) ? gamma[col] : 0.0f;

    for (int t = 0; t < n_tokens; ++t) {
        const float* qt = qkv_batch + (size_t)t * qkv_stride;
        const float* kt = qt + (size_t)h_k * S;
        const float* vt = kt + (size_t)h_k * S;
        if (tid < S) {
            sq[tid] = qt[qh * S + tid];
            sk[tid] = kt[qh * S + tid];
        }
        __syncthreads();
        const float g = __expf(gate[(size_t)t * h_v + head]);
        float kv = 0.0f;
#pragma unroll
        for (int r = 0; r < RPG; ++r) kv = fmaf(s[r], sk[rg * RPG + r], kv);
        red_kv[rg][col] = kv;
        __syncthreads();
        const float kv_col = red_kv[0][col] + red_kv[1][col] + red_kv[2][col] + red_kv[3][col];
        const float delta = (vt[head * S + col] - g * kv_col) * beta[(size_t)t * h_v + head];
        float o = 0.0f;
#pragma unroll
        for (int r = 0; r < RPG; ++r) {
            s[r] = fmaf(g, s[r], sk[rg * RPG + r] * delta);
            o = fmaf(s[r], sq[rg * RPG + r], o);
        }
        red_o[rg][col] = o;
        __syncthreads();
        float oc = 0.0f, sq_part = 0.0f;
        if (rg == 0) {
            oc = (red_o[0][col] + red_o[1][col] + red_o[2][col] + red_o[3][col]) * rsqrtf((float)S);
            sq_part = oc * oc;
        }
        for (int o2 = 16; o2 > 0; o2 >>= 1) sq_part += __shfl_xor_sync(0xffffffffu, sq_part, o2);
        if ((tid & 31) == 0) wsum[tid >> 5] = sq_part;
        __syncthreads();
        if (rg == 0) {
            const float ss = wsum[0] + wsum[1] + wsum[2] + wsum[3];
            const float scale = rsqrtf(ss / (float)S + eps);
            const size_t yoff = ((size_t)t * h_v + head) * S + col;
            const float zz = z[yoff];
            if constexpr (SILU) y[yoff] = oc * scale * gam * (zz / (1.0f + __expf(-zz)));
            else y[yoff] = oc * scale * gam * (1.0f / (1.0f + __expf(-zz)));
        }
        __syncthreads();
    }
#pragma unroll
    for (int r = 0; r < RPG; ++r) base[r * row_stride] = s[r];
}

__global__ void __launch_bounds__(S) gdn_conv_l2_batch_kernel(
    float* __restrict__ hist, const float* __restrict__ qkv_batch, const float* __restrict__ w,
    float* __restrict__ h_batch, int channels, int qk_heads, float eps, int n_tokens) {
    __shared__ float part[S / 32];
    const int c = blockIdx.x * S + threadIdx.x;
    float v0 = hist[c * 3], v1 = hist[c * 3 + 1], v2 = hist[c * 3 + 2];
    const float w0 = w[c * 4], w1 = w[c * 4 + 1], w2 = w[c * 4 + 2], w3 = w[c * 4 + 3];
    for (int t = 0; t < n_tokens; ++t) {
        const size_t off = (size_t)t * channels + c;
        const float x = qkv_batch[off];
        const float sum = v0 * w0 + v1 * w1 + v2 * w2 + x * w3;
        v0 = v1; v1 = v2; v2 = x;
        float y = sum / (1.0f + __expf(-sum));
        if ((int)blockIdx.x < qk_heads) {
            float sq = y * y;
            for (int o = 16; o > 0; o >>= 1) sq += __shfl_xor_sync(0xffffffffu, sq, o);
            if ((threadIdx.x & 31) == 0) part[threadIdx.x >> 5] = sq;
            __syncthreads();
            const float ss = part[0] + part[1] + part[2] + part[3];
            y *= rsqrtf(ss + eps);
        }
        h_batch[off] = y;
    }
    hist[c * 3] = v0;
    hist[c * 3 + 1] = v1;
    hist[c * 3 + 2] = v2;
}

__global__ void __launch_bounds__(256) gdn_ab_kernel(const float* __restrict__ x, const uint16_t* __restrict__ wa,
                                                     const uint16_t* __restrict__ wb, const float* __restrict__ dt,
                                                     const float* __restrict__ ssm_a, float* __restrict__ gate,
                                                     float* __restrict__ beta, int n, int h_v) {
    const int row = blockIdx.x * 8 + (threadIdx.x >> 5), lane = threadIdx.x & 31;
    if (row >= 2 * h_v) return;
    const bool is_beta = row >= h_v;
    const int r = is_beta ? row - h_v : row;
    const uint4* w4 = reinterpret_cast<const uint4*>((is_beta ? wb : wa) + (size_t) r * n);
    float acc = 0.0f;
    for (int j = lane; j < n / 8; j += 32) {
        const uint4 wv = __ldg(w4 + j);
        const float4 xa = *reinterpret_cast<const float4*>(x + j * 8);
        const float4 xb = *reinterpret_cast<const float4*>(x + j * 8 + 4);
        acc = fmaf(__uint_as_float(wv.x << 16), xa.x, acc); acc = fmaf(__uint_as_float(wv.x & 0xffff0000u), xa.y, acc);
        acc = fmaf(__uint_as_float(wv.y << 16), xa.z, acc); acc = fmaf(__uint_as_float(wv.y & 0xffff0000u), xa.w, acc);
        acc = fmaf(__uint_as_float(wv.z << 16), xb.x, acc); acc = fmaf(__uint_as_float(wv.z & 0xffff0000u), xb.y, acc);
        acc = fmaf(__uint_as_float(wv.w << 16), xb.z, acc); acc = fmaf(__uint_as_float(wv.w & 0xffff0000u), xb.w, acc);
    }
    for (int o = 16; o > 0; o >>= 1) acc += __shfl_xor_sync(0xffffffffu, acc, o);
    if (lane != 0) return;
    if (is_beta) {
        beta[r] = 1.0f / (1.0f + __expf(-acc));
    } else {
        const float v = acc + dt[r];
        const float sp = v > 20.0f ? v : log1pf(__expf(v));
        gate[r] = sp * ssm_a[r];
    }
}

}  // namespace

void fused_gdn_conv_l2(float* history, const float* qkv, const float* conv_w, float* h, int channels, int qk_heads,
                       float eps, void* stream) {
    if (!history || !qkv || !conv_w || !h || channels % S != 0 || qk_heads < 0 || qk_heads > channels / S) {
        std::fprintf(stderr, "fused_gdn_conv_l2: invalid arguments\n");
        std::exit(1);
    }
    gdn_conv_l2_kernel<<<(unsigned) (channels / S), S, 0, (cudaStream_t) stream>>>(history, qkv, conv_w, h, qk_heads, eps);
    const cudaError_t e = cudaGetLastError();
    if (e != cudaSuccess) { std::fprintf(stderr, "fused_gdn_conv_l2: %s\n", cudaGetErrorString(e)); std::exit(1); }
}

void fused_gdn_ab(const float* x, const uint16_t* w_alpha, const uint16_t* w_beta, const float* dt, const float* ssm_a,
                  float* gate, float* beta, int n_embd, int h_v, void* stream) {
    if (!x || !w_alpha || !w_beta || !dt || !ssm_a || !gate || !beta || n_embd % 8 != 0 || h_v <= 0) {
        std::fprintf(stderr, "fused_gdn_ab: invalid arguments\n");
        std::exit(1);
    }
    gdn_ab_kernel<<<(unsigned) ((2 * h_v + 7) / 8), 256, 0, (cudaStream_t) stream>>>(x, w_alpha, w_beta, dt, ssm_a, gate,
                                                                                     beta, n_embd, h_v);
    const cudaError_t e = cudaGetLastError();
    if (e != cudaSuccess) { std::fprintf(stderr, "fused_gdn_ab: %s\n", cudaGetErrorString(e)); std::exit(1); }
}

namespace {
template <bool SILU>
void step_norm(float* state, const float* q, const float* k, const float* v, const float* gate, const float* beta,
               const float* z, const float* gamma, float eps, float* y, int h_k, int h_v, void* stream) {
    if (!state || !q || !k || !v || !gate || !beta || !z || !gamma || !y || h_k <= 0 || h_v <= 0 || h_v % h_k) {
        std::fprintf(stderr, "fused_gdn_step_norm: invalid arguments\n");
        std::exit(1);
    }
    gdn_step_norm_kernel<SILU><<<(unsigned) h_v, dim3(S, RG), 0, (cudaStream_t) stream>>>(state, q, k, v, gate, beta,
                                                                                         z, gamma, eps, y, h_k, h_v);
    const cudaError_t e = cudaGetLastError();
    if (e != cudaSuccess) {
        std::fprintf(stderr, "fused_gdn_step_norm: %s\n", cudaGetErrorString(e));
        std::exit(1);
    }
}
}  // namespace

void fused_gdn_step_norm(float* state, const float* q, const float* k, const float* v, const float* gate,
                         const float* beta, const float* z, const float* gamma, float eps, float* y, int h_k, int h_v,
                         void* stream) {
    step_norm<false>(state, q, k, v, gate, beta, z, gamma, eps, y, h_k, h_v, stream);
}

void fused_gdn_step_norm_silu(float* state, const float* q, const float* k, const float* v, const float* gate,
                              const float* beta, const float* z, const float* gamma, float eps, float* y, int h_k,
                              int h_v, void* stream) {
    step_norm<true>(state, q, k, v, gate, beta, z, gamma, eps, y, h_k, h_v, stream);
}

void fused_gdn_conv_l2_batch(float* history, const float* qkv_batch, const float* conv_w, float* h_batch,
                             int channels, int qk_heads, float eps, int n_tokens, void* stream) {
    if (n_tokens <= 0) return;
    if (!history || !qkv_batch || !conv_w || !h_batch || channels % S != 0 || qk_heads < 0 || qk_heads > channels / S) {
        std::fprintf(stderr, "fused_gdn_conv_l2_batch: invalid arguments\n");
        std::exit(1);
    }
    gdn_conv_l2_batch_kernel<<<(unsigned)(channels / S), S, 0, (cudaStream_t)stream>>>(
        history, qkv_batch, conv_w, h_batch, channels, qk_heads, eps, n_tokens);
    const cudaError_t e = cudaGetLastError();
    if (e != cudaSuccess) { std::fprintf(stderr, "fused_gdn_conv_l2_batch: %s\n", cudaGetErrorString(e)); std::exit(1); }
}

void fused_gdn_step_norm_silu_batch(float* state, const float* qkv_batch, int qkv_stride, const float* gate,
                                    const float* beta, const float* z, const float* gamma, float eps, float* y,
                                    int h_k, int h_v, int n_tokens, void* stream) {
    if (n_tokens <= 0) return;
    if (!state || !qkv_batch || !gate || !beta || !z || !gamma || !y || h_k <= 0 || h_v <= 0 || h_v % h_k) {
        std::fprintf(stderr, "fused_gdn_step_norm_silu_batch: invalid arguments\n");
        std::exit(1);
    }
    gdn_step_norm_batch_kernel<true><<<(unsigned)h_v, dim3(S, RG), 0, (cudaStream_t)stream>>>(
        state, qkv_batch, qkv_stride, gate, beta, z, gamma, eps, y, h_k, h_v, n_tokens);
    const cudaError_t e = cudaGetLastError();
    if (e != cudaSuccess) { std::fprintf(stderr, "fused_gdn_step_norm_silu_batch: %s\n", cudaGetErrorString(e)); std::exit(1); }
}

}  // namespace strata::kernels
