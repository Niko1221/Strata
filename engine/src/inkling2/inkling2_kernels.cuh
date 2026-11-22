// Ported from strata-inkling2 by kuraneko1 (_ryu15_)
// https://github.com/kuraneko1/Strata (MIT License)
// See docs/third-party/inkling2-kuraneko/CREDITS.md
// src/inkling2/inkling2_kernels.cuh - CUDA kernels for Inkling-2-Mini (inkling2).
#pragma once

#include <cuda_fp16.h>
#include <cuda_runtime.h>
#if !defined(__CUDA_ARCH__) || __CUDA_ARCH__ >= 700
#include <mma.h>
#endif
#include <cstdint>
#include <cmath>

namespace strata::inkling2 {

inline constexpr int kThreads = 256;

__device__ __forceinline__ float sigmf(float x) { return __fdividef(1.0f, 1.0f + __expf(-x)); }
__device__ __forceinline__ float fast_tanh(float x) {
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 750
    float y;
    asm("tanh.approx.f32 %0, %1;" : "=f"(y) : "f"(x));
    return y;
#else
    const float xc = fmaxf(-9.0f, fminf(9.0f, x));
    return __fdividef(2.0f, 1.0f + __expf(-2.0f * xc)) - 1.0f;
#endif
}

__device__ __forceinline__ float block_sum(float v, float* sh) {
    const int lane = threadIdx.x & 31, warp = threadIdx.x >> 5;
#pragma unroll
    for (int o = 16; o > 0; o >>= 1) v += __shfl_xor_sync(0xffffffffu, v, o);
    if (lane == 0) sh[warp] = v;
    __syncthreads();
    if (warp == 0) {
        v = lane < (blockDim.x >> 5) ? sh[lane] : 0.0f;
#pragma unroll
        for (int o = 16; o > 0; o >>= 1) v += __shfl_xor_sync(0xffffffffu, v, o);
        if (lane == 0) sh[0] = v;
    }
    __syncthreads();
    return sh[0];
}

__global__ void k_f32_to_bf16(const float* __restrict__ x, uint16_t* __restrict__ y, int64_t n) {
    const int64_t i = (int64_t) blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;
    uint32_t u;
    memcpy(&u, x + i, sizeof u);
    const uint32_t bias = 0x7FFFu + ((u >> 16) & 1u);
    y[i] = (uint16_t) ((u + bias) >> 16);
}

__global__ void k_f32_to_f16(const float* __restrict__ x, uint16_t* __restrict__ y, int64_t n) {
    if ((n & 3) == 0) {
        const int64_t i = (int64_t) blockIdx.x * blockDim.x + threadIdx.x;
        const int64_t n4 = n >> 2;
        if (i >= n4) return;
        const float4 v = reinterpret_cast<const float4*>(x)[i];
        __half2 p0 = __halves2half2(__float2half_rn(v.x), __float2half_rn(v.y));
        __half2 p1 = __halves2half2(__float2half_rn(v.z), __float2half_rn(v.w));
        uint2 out;
        memcpy(&out.x, &p0, 4);
        memcpy(&out.y, &p1, 4);
        reinterpret_cast<uint2*>(y)[i] = out;
        return;
    }
    const int64_t i = (int64_t) blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;
    __half h = __float2half_rn(x[i]);
    uint16_t u;
    memcpy(&u, &h, sizeof u);
    y[i] = u;
}

__global__ void k_bf16_to_f32(const uint16_t* __restrict__ x, float* __restrict__ y, int64_t n) {
    const int64_t i = (int64_t) blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;
    uint32_t u = (uint32_t) x[i] << 16;
    float f;
    memcpy(&f, &u, sizeof f);
    y[i] = f;
}

__global__ void k_add(float* __restrict__ x, const float* __restrict__ y, int64_t n) {
    const int64_t i = (int64_t) blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) x[i] += y[i];
}

__global__ void k_rms_norm(const float* __restrict__ x, const float* __restrict__ w, float* __restrict__ y,
                           int n, float eps) {
    __shared__ float sh[32];
    const int t = blockIdx.x;
    const float* xr = x + (size_t) t * n;
    float* yr = y + (size_t) t * n;
    float s = 0.0f;
    if ((n & 3) == 0) {
        const int n4 = n >> 2;
        const float4* x4 = (const float4*) xr;
        float4* y4 = (float4*) yr;
        for (int i = threadIdx.x; i < n4; i += blockDim.x) {
            const float4 v = x4[i];
            s += v.x * v.x + v.y * v.y + v.z * v.z + v.w * v.w;
        }
        s = block_sum(s, sh);
        const float r = rsqrtf(s / (float) n + eps);
        const float4* w4 = w ? (const float4*) w : nullptr;
        for (int i = threadIdx.x; i < n4; i += blockDim.x) {
            const float4 v = x4[i];
            if (w4) {
                const float4 ww = w4[i];
                y4[i] = make_float4(v.x * r * ww.x, v.y * r * ww.y, v.z * r * ww.z, v.w * r * ww.w);
            } else {
                y4[i] = make_float4(v.x * r, v.y * r, v.z * r, v.w * r);
            }
        }
        return;
    }
    for (int i = threadIdx.x; i < n; i += blockDim.x) {
        const float v = xr[i];
        s += v * v;
    }
    s = block_sum(s, sh);
    const float r = rsqrtf(s / (float) n + eps);
    for (int i = threadIdx.x; i < n; i += blockDim.x) {
        yr[i] = xr[i] * r * (w ? w[i] : 1.0f);
    }
}

/// Normalize 3 independent vectors (q, k, v) with shared weight w in a single grid(3, T) launch.
__global__ void k_rms_norm_3(float* __restrict__ q, float* __restrict__ k, float* __restrict__ v,
                             const float* __restrict__ w, int n, float eps) {
    __shared__ float sh[32];
    const int which = blockIdx.x, t = blockIdx.y;
    float* ptr = (which == 0 ? q : (which == 1 ? k : v)) + (size_t) t * n;
    float s = 0.0f;
    if ((n & 3) == 0) {
        const int n4 = n >> 2;
        float4* p4 = (float4*) ptr;
        const float4* w4 = w ? (const float4*) w : nullptr;
        for (int i = threadIdx.x; i < n4; i += blockDim.x) {
            const float4 x = p4[i];
            s += x.x * x.x + x.y * x.y + x.z * x.z + x.w * x.w;
        }
        s = block_sum(s, sh);
        const float r = rsqrtf(s / (float) n + eps);
        for (int i = threadIdx.x; i < n4; i += blockDim.x) {
            const float4 x = p4[i];
            if (w4) {
                const float4 ww = w4[i];
                p4[i] = make_float4(x.x * r * ww.x, x.y * r * ww.y, x.z * r * ww.z, x.w * r * ww.w);
            } else {
                p4[i] = make_float4(x.x * r, x.y * r, x.z * r, x.w * r);
            }
        }
        return;
    }
    for (int i = threadIdx.x; i < n; i += blockDim.x) {
        const float x = ptr[i];
        s += x * x;
    }
    s = block_sum(s, sh);
    const float r = rsqrtf(s / (float) n + eps);
    for (int i = threadIdx.x; i < n; i += blockDim.x) {
        ptr[i] = ptr[i] * r * (w ? w[i] : 1.0f);
    }
}

struct BlkPtrs {
    const float* p[8];
};

struct MixOuts {
    float* p[3];
};

/// Normalize D source vectors (blocks[0..D-2], then cur) for T tokens into contiguous stack_norm[T][D][N].
__global__ void k_stack_rms_norm(BlkPtrs ptrs, const float* __restrict__ cur,
                                 float* __restrict__ out, int D, int N, int T, float eps) {
    __shared__ float sh[32];
    const int d = blockIdx.x, t = blockIdx.y;
    const float* src = (d == D - 1) ? (cur + (size_t) t * N) : (ptrs.p[d] + (size_t) t * N);
    float* dst = out + ((size_t) t * D + d) * N;
    float s = 0.0f;
    for (int i = threadIdx.x; i < N; i += blockDim.x) {
        const float v = src[i];
        s += v * v;
    }
    s = block_sum(s, sh);
    const float r = rsqrtf(s / (float) N + eps);
    for (int i = threadIdx.x; i < N; i += blockDim.x) dst[i] = src[i] * r;
}

/// mix() across depth stack: out[q][t][N] = sum_d softmax(dot(stack_norm[t][d], query[q]) / sqrt(N))[d] * stack_norm[t][d]
/// grid (n_q, T), block 256. Max depth D <= 8.
__global__ void k_mix(const float* __restrict__ stack_norm, const float* __restrict__ query, MixOuts out,
                      int D, int N, int n_q, float inv_sqrt_n, int swa_broadcast) {
    __shared__ float sh[32];
    __shared__ float w[16];
    const int q = blockIdx.x, t = blockIdx.y;
    const float* qv = query + (size_t) q * N;
    float mx = -1e30f;
    for (int d = 0; d < D; ++d) {
        const float* sv = stack_norm + ((size_t) t * D + d) * N;
        float dot = 0.0f;
        for (int i = threadIdx.x; i < N; i += blockDim.x) dot += sv[i] * qv[i];
        dot = block_sum(dot, sh) * inv_sqrt_n;
        if (threadIdx.x == 0) {
            w[d] = dot;
            if (dot > mx) mx = dot;
        }
        __syncthreads();
    }
    if (threadIdx.x == 0) {
        float m = -1e30f;
        for (int d = 0; d < D; ++d) if (w[d] > m) m = w[d];
        float sum = 0.0f;
        for (int d = 0; d < D; ++d) {
            w[d] = __expf(w[d] - m);
            sum += w[d];
        }
        const float inv = __fdividef(1.0f, sum);
        for (int d = 0; d < D; ++d) w[d] *= inv;
    }
    __syncthreads();
    if (swa_broadcast) {
        for (int i = threadIdx.x; i < N; i += blockDim.x) {
            float acc = 0.0f;
            for (int d = 0; d < D; ++d) acc += w[d] * stack_norm[((size_t) t * D + d) * N + i];
            out.p[0][(size_t) t * N + i] = acc;
            out.p[1][(size_t) t * N + i] = acc;
            out.p[2][(size_t) t * N + i] = acc;
        }
    } else {
        float* dst = out.p[q] + (size_t) t * N;
        for (int i = threadIdx.x; i < N; i += blockDim.x) {
            float acc = 0.0f;
            for (int d = 0; d < D; ++d) acc += w[d] * stack_norm[((size_t) t * D + d) * N + i];
            dst[i] = acc;
        }
    }
}

/// Mix across D already-RMSNormalized pointers (norm_ptrs[0..D-1]): out[q][t][N] = sum_d softmax(dot(v[d], query[q]) / sqrt(N))[d] * v[d].
/// Eliminates redundant stack_norm materialization when block residuals are cached normalized. grid (n_q, T), block 256.
struct NormMixPtrs {
    const float* p[8];
};

__global__ void k_mix_normed(NormMixPtrs ptrs, int D, const float* __restrict__ query, MixOuts out,
                             int n_q, int N, float inv_sqrt_n, int swa_broadcast) {
    __shared__ float sh[32];
    __shared__ float w[16];
    const int q = blockIdx.x, t = blockIdx.y;
    const float* qv = query + (size_t) q * N;
    const int n4 = N >> 2;
    const float4* q4 = (const float4*) qv;
    for (int d = 0; d < D; ++d) {
        const float* sv = ptrs.p[d] + (size_t) t * N;
        float dot = 0.0f;
        if ((N & 3) == 0) {
            const float4* s4 = (const float4*) sv;
            for (int i = threadIdx.x; i < n4; i += blockDim.x) {
                const float4 a = s4[i], b = q4[i];
                dot += a.x * b.x + a.y * b.y + a.z * b.z + a.w * b.w;
            }
        } else {
            for (int i = threadIdx.x; i < N; i += blockDim.x) dot += sv[i] * qv[i];
        }
        dot = block_sum(dot, sh) * inv_sqrt_n;
        if (threadIdx.x == 0) w[d] = dot;
        __syncthreads();
    }
    if (threadIdx.x == 0) {
        float m = -1e30f;
        for (int d = 0; d < D; ++d) if (w[d] > m) m = w[d];
        float sum = 0.0f;
        for (int d = 0; d < D; ++d) {
            w[d] = __expf(w[d] - m);
            sum += w[d];
        }
        const float inv = __fdividef(1.0f, sum);
        for (int d = 0; d < D; ++d) w[d] *= inv;
    }
    __syncthreads();
    if ((N & 3) == 0) {
        if (swa_broadcast) {
            float4* o0 = (float4*) (out.p[0] + (size_t) t * N);
            float4* o1 = (float4*) (out.p[1] + (size_t) t * N);
            float4* o2 = (float4*) (out.p[2] + (size_t) t * N);
            for (int i = threadIdx.x; i < n4; i += blockDim.x) {
                float ax = 0.0f, ay = 0.0f, az = 0.0f, aw = 0.0f;
                for (int d = 0; d < D; ++d) {
                    const float4 v = ((const float4*) (ptrs.p[d] + (size_t) t * N))[i];
                    const float wd = w[d];
                    ax += wd * v.x; ay += wd * v.y; az += wd * v.z; aw += wd * v.w;
                }
                const float4 r = make_float4(ax, ay, az, aw);
                o0[i] = r; o1[i] = r; o2[i] = r;
            }
        } else {
            float4* dst = (float4*) (out.p[q] + (size_t) t * N);
            for (int i = threadIdx.x; i < n4; i += blockDim.x) {
                float ax = 0.0f, ay = 0.0f, az = 0.0f, aw = 0.0f;
                for (int d = 0; d < D; ++d) {
                    const float4 v = ((const float4*) (ptrs.p[d] + (size_t) t * N))[i];
                    const float wd = w[d];
                    ax += wd * v.x; ay += wd * v.y; az += wd * v.z; aw += wd * v.w;
                }
                dst[i] = make_float4(ax, ay, az, aw);
            }
        }
        return;
    }
    if (swa_broadcast) {
        for (int i = threadIdx.x; i < N; i += blockDim.x) {
            float acc = 0.0f;
            for (int d = 0; d < D; ++d) acc += w[d] * ptrs.p[d][(size_t) t * N + i];
            out.p[0][(size_t) t * N + i] = acc;
            out.p[1][(size_t) t * N + i] = acc;
            out.p[2][(size_t) t * N + i] = acc;
        }
    } else {
        float* dst = out.p[q] + (size_t) t * N;
        for (int i = threadIdx.x; i < N; i += blockDim.x) {
            float acc = 0.0f;
            for (int d = 0; d < D; ++d) acc += w[d] * ptrs.p[d][(size_t) t * N + i];
            dst[i] = acc;
        }
    }
}

/// Fused 4-query mix for loop bank + bank_last (`q_attn` + `q_kv` in 1 grid(4, T) launch).
struct MixOuts4 {
    float* p[4];
};

__global__ void k_mix_normed_4(NormMixPtrs ptrs, int D, const float* __restrict__ q0, const float* __restrict__ q1,
                               MixOuts4 out, int N, float inv_sqrt_n) {
    __shared__ float sh[32];
    __shared__ float w[16];
    const int q = blockIdx.x, t = blockIdx.y;
    const float* qv = (q < 2 ? (q0 + (size_t) q * N) : (q1 + (size_t) (q - 2) * N));
    const int n4 = N >> 2;
    const float4* q4 = (const float4*) qv;
    for (int d = 0; d < D; ++d) {
        const float* sv = ptrs.p[d] + (size_t) t * N;
        float dot = 0.0f;
        if ((N & 3) == 0) {
            const float4* s4 = (const float4*) sv;
            for (int i = threadIdx.x; i < n4; i += blockDim.x) {
                const float4 a = s4[i], b = q4[i];
                dot += a.x * b.x + a.y * b.y + a.z * b.z + a.w * b.w;
            }
        } else {
            for (int i = threadIdx.x; i < N; i += blockDim.x) dot += sv[i] * qv[i];
        }
        dot = block_sum(dot, sh) * inv_sqrt_n;
        if (threadIdx.x == 0) w[d] = dot;
        __syncthreads();
    }
    if (threadIdx.x == 0) {
        float m = -1e30f;
        for (int d = 0; d < D; ++d) if (w[d] > m) m = w[d];
        float sum = 0.0f;
        for (int d = 0; d < D; ++d) {
            w[d] = __expf(w[d] - m);
            sum += w[d];
        }
        const float inv = __fdividef(1.0f, sum);
        for (int d = 0; d < D; ++d) w[d] *= inv;
    }
    __syncthreads();
    if ((N & 3) == 0) {
        float4* dst = (float4*) (out.p[q] + (size_t) t * N);
        for (int i = threadIdx.x; i < n4; i += blockDim.x) {
            float ax = 0.0f, ay = 0.0f, az = 0.0f, aw = 0.0f;
            for (int d = 0; d < D; ++d) {
                const float4 v = ((const float4*) (ptrs.p[d] + (size_t) t * N))[i];
                const float wd = w[d];
                ax += wd * v.x; ay += wd * v.y; az += wd * v.z; aw += wd * v.w;
            }
            dst[i] = make_float4(ax, ay, az, aw);
        }
        return;
    }
    float* dst = out.p[q] + (size_t) t * N;
    for (int i = threadIdx.x; i < N; i += blockDim.x) {
        float acc = 0.0f;
        for (int d = 0; d < D; ++d) acc += w[d] * ptrs.p[d][(size_t) t * N + i];
        dst[i] = acc;
    }
}

/// Fused single-pass multi-query mix (prefill, (N&3)==0, D<=8, n_q in {1,3,4}):
/// 1 CTA per token reads each ptrs.p[d] ONCE across all n_q queries (3x-4x less HBM traffic, 3 barriers total vs 3*D).
template <int NQ>
__global__ void k_mix_normed_fused(NormMixPtrs ptrs, int D, const float* __restrict__ q0, const float* __restrict__ q1,
                                   MixOuts4 out, int N, float inv_sqrt_n, int swa_broadcast) {
    __shared__ float s_dot[NQ][8][8];  // [q][d][warp]
    __shared__ float s_w[NQ][8];
    const int t = blockIdx.x;
    const int tid = threadIdx.x, warp = tid >> 5, lane = tid & 31;
    const int n4 = N >> 2;
    const float4* qp[NQ];
#pragma unroll
    for (int q = 0; q < NQ; ++q) {
        if constexpr (NQ == 4) {
            qp[q] = (const float4*) (q < 2 ? (q0 + (size_t) q * N) : (q1 + (size_t) (q - 2) * N));
        } else {
            qp[q] = (const float4*) (q0 + (size_t) q * N);
        }
    }
    for (int d = 0; d < D; ++d) {
        const float4* s4 = (const float4*) (ptrs.p[d] + (size_t) t * N);
        float acc[NQ] = {};
        for (int i = tid; i < n4; i += 256) {
            const float4 v = s4[i];
#pragma unroll
            for (int q = 0; q < NQ; ++q) {
                const float4 b = qp[q][i];
                acc[q] += v.x * b.x + v.y * b.y + v.z * b.z + v.w * b.w;
            }
        }
#pragma unroll
        for (int q = 0; q < NQ; ++q) {
            float v = acc[q];
#pragma unroll
            for (int o = 16; o > 0; o >>= 1) v += __shfl_xor_sync(0xffffffffu, v, o);
            if (lane == 0) s_dot[q][d][warp] = v;
        }
    }
    __syncthreads();
    if (tid < NQ * D) {
        const int q = tid / D, d = tid % D;
        float s = 0.0f;
#pragma unroll
        for (int w = 0; w < 8; ++w) s += s_dot[q][d][w];
        s_w[q][d] = s * inv_sqrt_n;
    }
    __syncthreads();
    if (tid < NQ) {
        const int q = tid;
        float m = -1e30f;
        for (int d = 0; d < D; ++d) m = fmaxf(m, s_w[q][d]);
        float sum = 0.0f;
        for (int d = 0; d < D; ++d) {
            s_w[q][d] = __expf(s_w[q][d] - m);
            sum += s_w[q][d];
        }
        const float inv = __fdividef(1.0f, sum);
        for (int d = 0; d < D; ++d) s_w[q][d] *= inv;
    }
    __syncthreads();
    if constexpr (NQ == 1) {
        if (swa_broadcast) {
            float4* o0 = (float4*) (out.p[0] + (size_t) t * N);
            float4* o1 = (float4*) (out.p[1] + (size_t) t * N);
            float4* o2 = (float4*) (out.p[2] + (size_t) t * N);
            for (int i = tid; i < n4; i += 256) {
                float ax = 0.0f, ay = 0.0f, az = 0.0f, aw = 0.0f;
                for (int d = 0; d < D; ++d) {
                    const float4 v = ((const float4*) (ptrs.p[d] + (size_t) t * N))[i];
                    const float wd = s_w[0][d];
                    ax += wd * v.x; ay += wd * v.y; az += wd * v.z; aw += wd * v.w;
                }
                const float4 r = make_float4(ax, ay, az, aw);
                o0[i] = r; o1[i] = r; o2[i] = r;
            }
            return;
        }
    }
    float4* dst[NQ];
#pragma unroll
    for (int q = 0; q < NQ; ++q) dst[q] = (float4*) (out.p[q] + (size_t) t * N);
    for (int i = tid; i < n4; i += 256) {
        float ax[NQ] = {}, ay[NQ] = {}, az[NQ] = {}, aw[NQ] = {};
        for (int d = 0; d < D; ++d) {
            const float4 v = ((const float4*) (ptrs.p[d] + (size_t) t * N))[i];
#pragma unroll
            for (int q = 0; q < NQ; ++q) {
                const float wd = s_w[q][d];
                ax[q] += wd * v.x; ay[q] += wd * v.y; az[q] += wd * v.z; aw[q] += wd * v.w;
            }
        }
#pragma unroll
        for (int q = 0; q < NQ; ++q) dst[q][i] = make_float4(ax[q], ay[q], az[q], aw[q]);
    }
}

/// control prep: out2N[t][0..N-1] = rms_norm(cur[t]), out2N[t][N..2N-1] = rms_norm(msg[t]). grid(2, T), block 256.
__global__ void k_control_prep(const float* __restrict__ cur, const float* __restrict__ msg,
                               float* __restrict__ out2N, int N, float eps) {
    __shared__ float sh[32];
    const int k = blockIdx.x, t = blockIdx.y;
    const float* src = (k == 0 ? cur : msg) + (size_t) t * N;
    float* dst = out2N + (size_t) t * (2 * N) + (k * N);
    float s = 0.0f;
    if ((N & 3) == 0) {
        const int n4 = N >> 2;
        const float4* s4 = (const float4*) src;
        float4* d4 = (float4*) dst;
        for (int i = threadIdx.x; i < n4; i += blockDim.x) {
            const float4 v = s4[i];
            s += v.x * v.x + v.y * v.y + v.z * v.z + v.w * v.w;
        }
        s = block_sum(s, sh);
        const float r = rsqrtf(s / (float) N + eps);
        for (int i = threadIdx.x; i < n4; i += blockDim.x) {
            const float4 v = s4[i];
            d4[i] = make_float4(v.x * r, v.y * r, v.z * r, v.w * r);
        }
        return;
    }
    for (int i = threadIdx.x; i < N; i += blockDim.x) {
        const float v = src[i];
        s += v * v;
    }
    s = block_sum(s, sh);
    const float r = rsqrtf(s / (float) N + eps);
    for (int i = threadIdx.x; i < N; i += blockDim.x) dst[i] = src[i] * r;
}

/// Fused control prep directly into FP16 [T][2*N] for Tensor Core loop_down GEMM (prefill, (N&3)==0).
__global__ void k_control_prep_f16(const float* __restrict__ cur, const float* __restrict__ msg,
                                   uint16_t* __restrict__ out2N_f16, int N, float eps) {
    __shared__ float sh[32];
    const int k = blockIdx.x, t = blockIdx.y;
    const float* src = (k == 0 ? cur : msg) + (size_t) t * N;
    __half2* dst2 = reinterpret_cast<__half2*>(out2N_f16 + (size_t) t * (2 * N) + (k * N));
    const int n4 = N >> 2;
    const float4* s4 = (const float4*) src;
    float s = 0.0f;
    for (int i = threadIdx.x; i < n4; i += blockDim.x) {
        const float4 v = s4[i];
        s += v.x * v.x + v.y * v.y + v.z * v.z + v.w * v.w;
    }
    s = block_sum(s, sh);
    const float r = rsqrtf(s / (float) N + eps);
    for (int i = threadIdx.x; i < n4; i += blockDim.x) {
        const float4 v = s4[i];
        dst2[i * 2 + 0] = __halves2half2(__float2half_rn(v.x * r), __float2half_rn(v.y * r));
        dst2[i * 2 + 1] = __halves2half2(__float2half_rn(v.z * r), __float2half_rn(v.w * r));
    }
}

__global__ void k_control_combine(const float* __restrict__ cur, const float* __restrict__ msg,
                                  const float* __restrict__ gain, const float* __restrict__ forget,
                                  float* __restrict__ out, int64_t n) {
    if ((n & 3) == 0) {
        const int64_t i = (int64_t) blockIdx.x * blockDim.x + threadIdx.x;
        const int64_t n4 = n >> 2;
        if (i >= n4) return;
        const float4 c = reinterpret_cast<const float4*>(cur)[i];
        const float4 m = reinterpret_cast<const float4*>(msg)[i];
        const float4 g = reinterpret_cast<const float4*>(gain)[i];
        const float4 f = reinterpret_cast<const float4*>(forget)[i];
        reinterpret_cast<float4*>(out)[i] = make_float4(
            (1.0f - fast_tanh(f.x)) * c.x + fast_tanh(g.x) * m.x,
            (1.0f - fast_tanh(f.y)) * c.y + fast_tanh(g.y) * m.y,
            (1.0f - fast_tanh(f.z)) * c.z + fast_tanh(g.z) * m.z,
            (1.0f - fast_tanh(f.w)) * c.w + fast_tanh(g.w) * m.w);
        return;
    }
    const int64_t i = (int64_t) blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;
    const float g = fast_tanh(gain[i]);
    const float f = fast_tanh(forget[i]);
    out[i] = (1.0f - f) * cur[i] + g * msg[i];
}

/// Fused T=1 loop_gain + loop_forget ([R=64 -> N] F32) + k_control_combine: 8 warps/block, 4 rows/warp -> 4x fewer CTAs & barriers.
__global__ void k_control_up_combine_t1(const float* __restrict__ cur, const float* __restrict__ msg,
                                        const float* __restrict__ w_gain, const float* __restrict__ w_forget,
                                        const float* __restrict__ ctrl64, float* __restrict__ out,
                                        int R, int N) {
    __shared__ float sx[64];
    if (threadIdx.x < R) sx[threadIdx.x] = ctrl64[threadIdx.x];
    __syncthreads();
    const int warp = threadIdx.x >> 5, lane = threadIdx.x & 31;
    const int base_o = (blockIdx.x * 8 + warp) * 4;
    const int half = lane >> 4;
    const int i = lane & 15;
    const float4 x = reinterpret_cast<const float4*>(sx)[i];
    const float* wg = half ? w_forget : w_gain;
#pragma unroll
    for (int k = 0; k < 4; ++k) {
        const int o = base_o + k;
        if (o >= N) return;
        const float4 w = reinterpret_cast<const float4*>(wg + (size_t) o * R)[i];
        float acc = w.x * x.x + w.y * x.y + w.z * x.z + w.w * x.w;
#pragma unroll
        for (int m = 8; m > 0; m >>= 1) acc += __shfl_xor_sync(0xffffffffu, acc, m);
        const float f_val = __shfl_sync(0xffffffffu, acc, 16);
        if (lane == 0) {
            out[o] = (1.0f - fast_tanh(f_val)) * cur[o] + fast_tanh(acc) * msg[o];
        }
    }
}

/// High-occupancy F32 GEMV for small n_out (1 CTA of 128 threads per output row).
__global__ void k_gemv_f32_cta(const float* __restrict__ W, const float* __restrict__ x, float* __restrict__ y,
                               int n_in, int n_out) {
    __shared__ float sh[32];
    const int o = blockIdx.x;
    if (o >= n_out) return;
    const float4* wr = (const float4*) (W + (size_t) o * n_in);
    const float4* xr = (const float4*) x;
    float acc = 0.0f;
    const int n4 = n_in / 4;
    for (int i = threadIdx.x; i < n4; i += blockDim.x) {
        const float4 a = wr[i], b = xr[i];
        acc += a.x * b.x + a.y * b.y + a.z * b.z + a.w * b.w;
    }
    acc = block_sum(acc, sh);
    if (threadIdx.x == 0) y[o] = acc;
}

/// Causal 4-tap shortconv (+ optional dynamic per-head taps for full-attn K/V).
/// hist has [hist_stride, 3] floats? Wait: hist is [n_channels_total][3] for coalesced / per-channel state:
/// specifically hist[channel_offset + c][0..2] holds the 3 oldest->newest prior inputs.
/// Dynamic weight `dyn` is [T][n_heads][4] where lag 0 = current token, lag 3 = t-3.
/// Static weight `w` is [n_ch][4] where w[0] = t-3, w[3] = current token.
__global__ void k_shortconv(const float* __restrict__ cur, const float* __restrict__ w,
                            const float* __restrict__ dyn, float* __restrict__ hist,
                            float* __restrict__ out, int n_ch, int hist_offset, int n_heads, int T,
                            const float* __restrict__ res_add = nullptr,
                            float* __restrict__ hist_trace = nullptr, size_t hist_trace_stride = 0) {
    const int c = blockIdx.x * blockDim.x + threadIdx.x;
    if (c >= n_ch) return;
    float* hc = hist + (size_t) (hist_offset + c) * 3;
    float h0 = hc[0], h1 = hc[1], h2 = hc[2];
    const float4 w4 = *reinterpret_cast<const float4*>(w + (size_t) c * 4);
    const float w0 = w4.x, w1 = w4.y, w2 = w4.z, w3 = w4.w;
    const int head_dim = n_heads > 0 ? (n_ch / n_heads) : 0;
    const int h_idx = n_heads > 0 ? (c / head_dim) : 0;
    const float scale = rsqrtf(dyn ? 9.0f : 5.0f);
    for (int t = 0; t < T; ++t) {
        const size_t idx = (size_t) t * n_ch + c;
        const float x = cur[idx];
        float s = h0 * w0 + h1 * w1 + h2 * w2 + x * w3;
        if (dyn) {
            const float* dt = dyn + ((size_t) t * n_heads + h_idx) * 4;
            s += x * dt[0] + h2 * dt[1] + h1 * dt[2] + h0 * dt[3];
        }
        float y = (x + s) * scale;
        if (res_add) y += res_add[idx];
        out[idx] = y;
        h0 = h1;
        h1 = h2;
        h2 = x;
        if (hist_trace) {
            float* ht = hist_trace + (size_t) t * hist_trace_stride + (size_t) (hist_offset + c) * 3;
            ht[0] = h0;
            ht[1] = h1;
            ht[2] = h2;
        }
    }
    hc[0] = h0;
    hc[1] = h1;
    hc[2] = h2;
}

/// Fused shortconv for K [0..kv_dim-1] and V [kv_dim..2*kv_dim-1]: single launch of 2*kv_dim threads.
__global__ void k_shortconv_kv(const float* __restrict__ K, const float* __restrict__ V,
                               const float* __restrict__ wk, const float* __restrict__ wv,
                               const float* __restrict__ dyn_k, const float* __restrict__ dyn_v,
                               float* __restrict__ hist, float* __restrict__ K_out, float* __restrict__ V_out,
                               int kv_dim, int n_heads, int T,
                               float* __restrict__ hist_trace = nullptr, size_t hist_trace_stride = 0) {
    const int c_all = blockIdx.x * blockDim.x + threadIdx.x;
    if (c_all >= 2 * kv_dim) return;
    const bool is_v = (c_all >= kv_dim);
    const int c = is_v ? (c_all - kv_dim) : c_all;
    const float* cur = is_v ? V : K;
    const float* dyn = is_v ? dyn_v : dyn_k;
    float* out = is_v ? V_out : K_out;
    float* hc = hist + (size_t) c_all * 3;
    float h0 = hc[0], h1 = hc[1], h2 = hc[2];
    const float4 w4 = *reinterpret_cast<const float4*>((is_v ? wv : wk) + (size_t) c * 4);
    const float w0 = w4.x, w1 = w4.y, w2 = w4.z, w3 = w4.w;
    const int head_dim = n_heads > 0 ? (kv_dim / n_heads) : 0;
    const int h_idx = n_heads > 0 ? (c / head_dim) : 0;
    const float scale = rsqrtf(dyn ? 9.0f : 5.0f);
    for (int t = 0; t < T; ++t) {
        const size_t idx = (size_t) t * kv_dim + c;
        const float x = cur[idx];
        float s = h0 * w0 + h1 * w1 + h2 * w2 + x * w3;
        if (dyn) {
            const float* dt = dyn + ((size_t) t * n_heads + h_idx) * 4;
            s += x * dt[0] + h2 * dt[1] + h1 * dt[2] + h0 * dt[3];
        }
        out[idx] = (x + s) * scale;
        h0 = h1;
        h1 = h2;
        h2 = x;
        if (hist_trace) {
            float* ht = hist_trace + (size_t) t * hist_trace_stride + (size_t) c_all * 3;
            ht[0] = h0;
            ht[1] = h1;
            ht[2] = h2;
        }
    }
    hc[0] = h0;
    hc[1] = h1;
    hc[2] = h2;
}

/// Parallel tiled shortconv (prefill, T >= 32, read-only on `hist` from blockIdx.y==0):
/// Grid: ((n_ch+255)/256, (T+15)/16), Block: 256. Followed by k_shortconv_hist_push.
__global__ void k_shortconv_pp(const float* __restrict__ cur, const float* __restrict__ w,
                               const float* __restrict__ hist, float* __restrict__ out,
                               int n_ch, int hist_offset, int T, const float* __restrict__ res_add) {
    const int c = blockIdx.x * blockDim.x + threadIdx.x;
    if (c >= n_ch) return;
    const int t0 = blockIdx.y * 16;
    const int t1 = min(T, t0 + 16);
    float h0, h1, h2;
    if (t0 == 0) {
        const float* hc = hist + (size_t) (hist_offset + c) * 3;
        h0 = hc[0]; h1 = hc[1]; h2 = hc[2];
    } else {
        h0 = cur[(size_t) (t0 - 3) * n_ch + c];
        h1 = cur[(size_t) (t0 - 2) * n_ch + c];
        h2 = cur[(size_t) (t0 - 1) * n_ch + c];
    }
    const float4 w4 = *reinterpret_cast<const float4*>(w + (size_t) c * 4);
    const float scale = rsqrtf(5.0f);
    for (int t = t0; t < t1; ++t) {
        const size_t idx = (size_t) t * n_ch + c;
        const float x = cur[idx];
        const float s = h0 * w4.x + h1 * w4.y + h2 * w4.z + x * w4.w;
        float y = (x + s) * scale;
        if (res_add) y += res_add[idx];
        out[idx] = y;
        h0 = h1; h1 = h2; h2 = x;
    }
}

__global__ void k_shortconv_kv_pp(const float* __restrict__ K, const float* __restrict__ V,
                                  const float* __restrict__ wk, const float* __restrict__ wv,
                                  const float* __restrict__ dyn_k, const float* __restrict__ dyn_v,
                                  const float* __restrict__ hist, float* __restrict__ K_out, float* __restrict__ V_out,
                                  int kv_dim, int n_heads, int T) {
    const int c_all = blockIdx.x * blockDim.x + threadIdx.x;
    if (c_all >= 2 * kv_dim) return;
    const bool is_v = (c_all >= kv_dim);
    const int c = is_v ? (c_all - kv_dim) : c_all;
    const float* cur = is_v ? V : K;
    const float* dyn = is_v ? dyn_v : dyn_k;
    float* out = is_v ? V_out : K_out;
    const int t0 = blockIdx.y * 16;
    const int t1 = min(T, t0 + 16);
    float h0, h1, h2;
    if (t0 == 0) {
        const float* hc = hist + (size_t) c_all * 3;
        h0 = hc[0]; h1 = hc[1]; h2 = hc[2];
    } else {
        h0 = cur[(size_t) (t0 - 3) * kv_dim + c];
        h1 = cur[(size_t) (t0 - 2) * kv_dim + c];
        h2 = cur[(size_t) (t0 - 1) * kv_dim + c];
    }
    const float4 w4 = *reinterpret_cast<const float4*>((is_v ? wv : wk) + (size_t) c * 4);
    const int head_dim = n_heads > 0 ? (kv_dim / n_heads) : 0;
    const int h_idx = n_heads > 0 ? (c / head_dim) : 0;
    const float scale = rsqrtf(dyn ? 9.0f : 5.0f);
    for (int t = t0; t < t1; ++t) {
        const size_t idx = (size_t) t * kv_dim + c;
        const float x = cur[idx];
        float s = h0 * w4.x + h1 * w4.y + h2 * w4.z + x * w4.w;
        if (dyn) {
            const float* dt = dyn + ((size_t) t * n_heads + h_idx) * 4;
            s += x * dt[0] + h2 * dt[1] + h1 * dt[2] + h0 * dt[3];
        }
        out[idx] = (x + s) * scale;
        h0 = h1; h1 = h2; h2 = x;
    }
}

/// Update `hist` after parallel k_shortconv_pp / k_shortconv_kv_pp (for T >= 3).
__global__ void k_shortconv_hist_push(const float* __restrict__ cur, float* __restrict__ hist,
                                      int n_ch, int hist_offset, int T) {
    const int c = blockIdx.x * blockDim.x + threadIdx.x;
    if (c >= n_ch) return;
    float* hc = hist + (size_t) (hist_offset + c) * 3;
    hc[0] = cur[(size_t) (T - 3) * n_ch + c];
    hc[1] = cur[(size_t) (T - 2) * n_ch + c];
    hc[2] = cur[(size_t) (T - 1) * n_ch + c];
}

__global__ void k_shortconv_kv_hist_push(const float* __restrict__ K, const float* __restrict__ V,
                                         float* __restrict__ hist, int kv_dim, int T) {
    const int c_all = blockIdx.x * blockDim.x + threadIdx.x;
    if (c_all >= 2 * kv_dim) return;
    const bool is_v = (c_all >= kv_dim);
    const int c = is_v ? (c_all - kv_dim) : c_all;
    const float* cur = is_v ? V : K;
    float* hc = hist + (size_t) c_all * 3;
    hc[0] = cur[(size_t) (T - 3) * kv_dim + c];
    hc[1] = cur[(size_t) (T - 2) * kv_dim + c];
    hc[2] = cur[(size_t) (T - 1) * kv_dim + c];
}

/// clip(x, scale, thr) per [T][n_heads][D]. Also optionally multiplies by per-token tau if alpha > 0.
/// grid (n_heads, T), block min(128, D) or 32 (one warp when D <= 128).
__global__ void k_clip_and_tau(float* __restrict__ x, const float* __restrict__ scale, float thr, int D, int n_heads,
                               int apply_tau, const int32_t* __restrict__ pos0, int floor_n, float alpha) {
    const int h = blockIdx.x, t = blockIdx.y;
    const int lane = threadIdx.x & 31;
    float* v = x + ((size_t) t * n_heads + h) * D;
    float sq = 0.0f;
    float vals[4];
    const int iters = (D + 31) / 32;   // D = 128 -> 4, D = 16 -> 1
    for (int k = 0; k < iters; ++k) {
        const int i = k * 32 + lane;
        float val = 0.0f;
        if (i < D) {
            val = v[i];
            if (scale) val *= scale[(size_t) h * D + i];
        }
        vals[k] = val;
        sq += val * val;
    }
#pragma unroll
    for (int o = 16; o > 0; o >>= 1) sq += __shfl_xor_sync(0xffffffffu, sq, o);
    const float norm = sqrtf(sq) + 1e-6f;
    float factor = fminf(1.0f, thr / norm);
    if (apply_tau) {
        const int pos = pos0[0] + t;
        const float ratio = fmaxf(1.0f, (float) (pos + 1) / (float) floor_n);
        factor *= (1.0f + alpha * logf(ratio));
    }
    for (int k = 0; k < iters; ++k) {
        const int i = k * 32 + lane;
        if (i < D) v[i] = vals[k] * factor;
    }
}

/// Per-head RMSNorm on Vcur [T][n_heads_kv][head_dim] (no weight). grid (n_heads_kv, T), block 32.
__global__ void k_v_rms_norm(float* __restrict__ v, int D, int n_heads, float eps) {
    const int h = blockIdx.x, t = blockIdx.y, lane = threadIdx.x & 31;
    float* vr = v + ((size_t) t * n_heads + h) * D;
    float sq = 0.0f;
    float vals[4];
    for (int k = 0; k < 4; ++k) {
        const int i = k * 32 + lane;
        const float val = i < D ? vr[i] : 0.0f;
        vals[k] = val;
        sq += val * val;
    }
#pragma unroll
    for (int o = 16; o > 0; o >>= 1) sq += __shfl_xor_sync(0xffffffffu, sq, o);
    const float r = rsqrtf(sq / (float) D + eps);
    for (int k = 0; k < 4; ++k) {
        const int i = k * 32 + lane;
        if (i < D) vr[i] = vals[k] * r;
    }
}

/// Store K, V [T][kv_dim] into FP16 or Q8_0 cache [cap][kv_dim].
/// When q8_kv is enabled: kc_q8 stores [cap][n_kv_heads][128] int8_t, kc_s stores [cap][n_kv_heads] __half scale (1 scale per 128-dim head).
__global__ void k_kv_store(const float* __restrict__ k, const float* __restrict__ v,
                           __half* __restrict__ kc, __half* __restrict__ vc,
                           int kv_dim, const int32_t* __restrict__ pos0, int cap, int is_swa) {
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    const int t = blockIdx.y;
    if (i >= kv_dim) return;
    const int pos = pos0[0] + t;
    const int slot = is_swa ? (((cap & (cap - 1)) == 0) ? (pos & (cap - 1)) : (pos % cap)) : pos;
    kc[(size_t) slot * kv_dim + i] = __float2half(k[(size_t) t * kv_dim + i]);
    vc[(size_t) slot * kv_dim + i] = __float2half(v[(size_t) t * kv_dim + i]);
}

/// Q8 per-head (128-dim) KV store: grid (n_head_kv, T), block 32 (1 warp per head).
__global__ void k_kv_store_q8(const float* __restrict__ k, const float* __restrict__ v,
                              int8_t* __restrict__ kq, __half* __restrict__ ks,
                              int8_t* __restrict__ vq, __half* __restrict__ vs,
                              int n_head_kv, const int32_t* __restrict__ pos0, int cap, int is_swa) {
    const int hkv = blockIdx.x, t = blockIdx.y, lane = threadIdx.x & 31;
    const int kv_dim = n_head_kv * 128;
    const float* kp = k + (size_t) t * kv_dim + hkv * 128;
    const float* vp = v + (size_t) t * kv_dim + hkv * 128;
    float kv[4], vv[4], ak = 0.0f, av = 0.0f;
#pragma unroll
    for (int i = 0; i < 4; ++i) {
        kv[i] = kp[i * 32 + lane];
        vv[i] = vp[i * 32 + lane];
        ak = fmaxf(ak, fabsf(kv[i]));
        av = fmaxf(av, fabsf(vv[i]));
    }
#pragma unroll
    for (int o = 16; o > 0; o >>= 1) {
        ak = fmaxf(ak, __shfl_xor_sync(0xffffffffu, ak, o));
        av = fmaxf(av, __shfl_xor_sync(0xffffffffu, av, o));
    }
    const float dk = ak / 127.0f, dv = av / 127.0f;
    const float ik = dk > 0.0f ? (127.0f / ak) : 0.0f;
    const float iv = dv > 0.0f ? (127.0f / av) : 0.0f;
    const int pos = pos0[0] + t;
    const int slot = is_swa ? (((cap & (cap - 1)) == 0) ? (pos & (cap - 1)) : (pos % cap)) : pos;
    int8_t* kdst = kq + (size_t) slot * kv_dim + hkv * 128;
    int8_t* vdst = vq + (size_t) slot * kv_dim + hkv * 128;
#pragma unroll
    for (int i = 0; i < 4; ++i) {
        kdst[i * 32 + lane] = (int8_t) __float2int_rn(kv[i] * ik);
        vdst[i * 32 + lane] = (int8_t) __float2int_rn(vv[i] * iv);
    }
    if (lane == 0) {
        ks[(size_t) slot * n_head_kv + hkv] = __float2half(dk);
        vs[(size_t) slot * n_head_kv + hkv] = __float2half(dv);
    }
}

/// Fused clip(K) + rms_norm(V) + Q8 KV store (head_dim == 128): eliminates 2 kernel launches + K_conv/V_conv DRAM roundtrips.
__global__ void k_clip_norm_kv_store_q8(const float* __restrict__ k, const float* __restrict__ v,
                                         int8_t* __restrict__ kq, __half* __restrict__ ks,
                                         int8_t* __restrict__ vq, __half* __restrict__ vs,
                                         int n_head_kv, float qk_clip, float rms_eps,
                                         const int32_t* __restrict__ pos0, int cap, int is_swa) {
    const int hkv = blockIdx.x, t = blockIdx.y, lane = threadIdx.x & 31;
    const int kv_dim = n_head_kv * 128;
    const float* kp = k + (size_t) t * kv_dim + hkv * 128;
    const float* vp = v + (size_t) t * kv_dim + hkv * 128;
    float kv[4], vv[4], sk = 0.0f, sv = 0.0f;
#pragma unroll
    for (int i = 0; i < 4; ++i) {
        kv[i] = kp[i * 32 + lane];
        vv[i] = vp[i * 32 + lane];
        sk += kv[i] * kv[i];
        sv += vv[i] * vv[i];
    }
#pragma unroll
    for (int o = 16; o > 0; o >>= 1) {
        sk += __shfl_xor_sync(0xffffffffu, sk, o);
        sv += __shfl_xor_sync(0xffffffffu, sv, o);
    }
    const float fk = fminf(1.0f, qk_clip / (sqrtf(sk) + 1e-6f));
    const float fv = rsqrtf(sv / 128.0f + rms_eps);
    float ak = 0.0f, av = 0.0f;
#pragma unroll
    for (int i = 0; i < 4; ++i) {
        kv[i] *= fk;
        vv[i] *= fv;
        ak = fmaxf(ak, fabsf(kv[i]));
        av = fmaxf(av, fabsf(vv[i]));
    }
#pragma unroll
    for (int o = 16; o > 0; o >>= 1) {
        ak = fmaxf(ak, __shfl_xor_sync(0xffffffffu, ak, o));
        av = fmaxf(av, __shfl_xor_sync(0xffffffffu, av, o));
    }
    const float dk = ak / 127.0f, dv = av / 127.0f;
    const float ik = dk > 0.0f ? (127.0f / ak) : 0.0f;
    const float iv = dv > 0.0f ? (127.0f / av) : 0.0f;
    const int pos = pos0[0] + t;
    const int slot = is_swa ? (((cap & (cap - 1)) == 0) ? (pos & (cap - 1)) : (pos % cap)) : pos;
    int8_t* kdst = kq + (size_t) slot * kv_dim + hkv * 128;
    int8_t* vdst = vq + (size_t) slot * kv_dim + hkv * 128;
#pragma unroll
    for (int i = 0; i < 4; ++i) {
        kdst[i * 32 + lane] = (int8_t) __float2int_rn(kv[i] * ik);
        vdst[i * 32 + lane] = (int8_t) __float2int_rn(vv[i] * iv);
    }
    if (lane == 0) {
        ks[(size_t) slot * n_head_kv + hkv] = __float2half(dk);
        vs[(size_t) slot * n_head_kv + hkv] = __float2half(dv);
    }
}

/// Compute relative bias distance table minus sink: table[t][h][d] = dot(R[t,h], rel_bias[d]) - sink[t,h]
/// R is [T][n_head][n_rel=16], rel_bias is [n_ext=1024][16] (F32 contiguous along n_rel=16), sink is [T][n_head].
/// Reads *pos0 on device so grid is CUDA-Graph invariant. grid (n_head, (T + 7) / 8), block 256.
__global__ void k_rel_table(const float* __restrict__ R, const float* __restrict__ rel_bias,
                            const float* __restrict__ sink, float* __restrict__ table,
                            int n_head, int n_rel, int n_ext, int max_rel, const int32_t* __restrict__ pos0, int T) {
    __shared__ float4 sr[8][4];
    __shared__ float s_sk[8];
    const int h = blockIdx.x, t0 = blockIdx.y * 8;
    const int p0 = pos0[0];
    const int n_rows = min(8, T - t0);
    if (n_rows <= 0) return;
    if (threadIdx.x < n_rows * 4) {
        const int r = threadIdx.x >> 2, c = threadIdx.x & 3;
        const float* rr = R + ((size_t) (t0 + r) * n_head + h) * n_rel;
        sr[r][c] = reinterpret_cast<const float4*>(rr)[c];
        if (c == 0) s_sk[r] = sink[(size_t) (t0 + r) * n_head + h];
    }
    __syncthreads();
    const int max_d = min(min(n_ext, max_rel), p0 + t0 + n_rows);
    for (int d = threadIdx.x; d < max_d; d += blockDim.x) {
        const float4* rb4 = reinterpret_cast<const float4*>(rel_bias + (size_t) d * n_rel);
        const float4 b0 = rb4[0], b1 = rb4[1], b2 = rb4[2], b3 = rb4[3];
#pragma unroll
        for (int r = 0; r < 8; ++r) {
            if (r < n_rows && d <= p0 + t0 + r) {
                const float4 q0 = sr[r][0], q1 = sr[r][1], q2 = sr[r][2], q3 = sr[r][3];
                const float dot = (q0.x * b0.x + q0.y * b0.y + q0.z * b0.z + q0.w * b0.w)
                                + (q1.x * b1.x + q1.y * b1.y + q1.z * b1.z + q1.w * b1.w)
                                + (q2.x * b2.x + q2.y * b2.y + q2.z * b2.z + q2.w * b2.w)
                                + (q3.x * b3.x + q3.y * b3.y + q3.z * b3.z + q3.w * b3.w);
                table[(((size_t) (t0 + r) * n_head + h) * n_ext) + d] = dot - s_sk[r];
            }
        }
    }
}

/// 32-token-tile FP32 k_rel_table for prefill (T >= 32, n_rel == 16): 4x fewer rel_bias reads & CTAs, bit-identical FP32.
/// grid (n_head, (T + 31) / 32), block 256.
__global__ void k_rel_table_t32(const float* __restrict__ R, const float* __restrict__ rel_bias,
                                const float* __restrict__ sink, float* __restrict__ table,
                                int n_head, int n_rel, int n_ext, int max_rel, const int32_t* __restrict__ pos0, int T) {
    __shared__ float4 sr[32][4];
    __shared__ float s_sk[32];
    const int h = blockIdx.x, t0 = blockIdx.y * 32;
    const int p0 = pos0[0];
    const int n_rows = min(32, T - t0);
    if (n_rows <= 0) return;
    if (threadIdx.x < n_rows * 4) {
        const int r = threadIdx.x >> 2, c = threadIdx.x & 3;
        const float* rr = R + ((size_t) (t0 + r) * n_head + h) * n_rel;
        sr[r][c] = reinterpret_cast<const float4*>(rr)[c];
        if (c == 0) s_sk[r] = sink[(size_t) (t0 + r) * n_head + h];
    }
    __syncthreads();
    const int max_d = min(min(n_ext, max_rel), p0 + t0 + n_rows);
    const int full_d = min(max_d, p0 + t0 + 1);
    for (int d = threadIdx.x; d < max_d; d += blockDim.x) {
        const float4* rb4 = reinterpret_cast<const float4*>(rel_bias + (size_t) d * n_rel);
        const float4 b0 = rb4[0], b1 = rb4[1], b2 = rb4[2], b3 = rb4[3];
        if (n_rows == 32 && d < full_d) {
#pragma unroll 8
            for (int r = 0; r < 32; ++r) {
                const float4 q0 = sr[r][0], q1 = sr[r][1], q2 = sr[r][2], q3 = sr[r][3];
                const float dot = (q0.x * b0.x + q0.y * b0.y + q0.z * b0.z + q0.w * b0.w)
                                + (q1.x * b1.x + q1.y * b1.y + q1.z * b1.z + q1.w * b1.w)
                                + (q2.x * b2.x + q2.y * b2.y + q2.z * b2.z + q2.w * b2.w)
                                + (q3.x * b3.x + q3.y * b3.y + q3.z * b3.z + q3.w * b3.w);
                table[(((size_t) (t0 + r) * n_head + h) * n_ext) + d] = dot - s_sk[r];
            }
        } else {
            for (int r = 0; r < n_rows; ++r) {
                if (d <= p0 + t0 + r) {
                    const float4 q0 = sr[r][0], q1 = sr[r][1], q2 = sr[r][2], q3 = sr[r][3];
                    const float dot = (q0.x * b0.x + q0.y * b0.y + q0.z * b0.z + q0.w * b0.w)
                                    + (q1.x * b1.x + q1.y * b1.y + q1.z * b1.z + q1.w * b1.w)
                                    + (q2.x * b2.x + q2.y * b2.y + q2.z * b2.z + q2.w * b2.w)
                                    + (q3.x * b3.x + q3.y * b3.y + q3.z * b3.z + q3.w * b3.w);
                    table[(((size_t) (t0 + r) * n_head + h) * n_ext) + d] = dot - s_sk[r];
                }
            }
        }
    }
}

__global__ void k_attn_q8(const float* __restrict__ Q,
                          const int8_t* __restrict__ kq, const __half* __restrict__ ks,
                          const int8_t* __restrict__ vq, const __half* __restrict__ vs,
                          const float* __restrict__ rel_tab, const float* __restrict__ sink,
                          const float* __restrict__ gate, float* __restrict__ out,
                          int n_head, int n_head_kv, int n_ext, int cap, int is_swa, int n_swa,
                          const int32_t* __restrict__ pos0, float kq_scale) {
    __shared__ float sq[128];
    __shared__ float s_m[4];
    __shared__ float s_l[4];
    __shared__ float s_acc[4][128];
    const int h = blockIdx.x, t = blockIdx.y;
    const int warp = threadIdx.x >> 5, lane = threadIdx.x & 31;
    const int hkv = h / (n_head / n_head_kv);
    const int kv_dim = n_head_kv * 128;
    const float* qv = Q + ((size_t) t * n_head + h) * 128;
    if (threadIdx.x < 128) sq[threadIdx.x] = qv[threadIdx.x];
    __syncthreads();

    const int base4 = lane * 4;
    const float q0 = sq[base4], q1 = sq[base4 + 1], q2 = sq[base4 + 2], q3 = sq[base4 + 3];

    const int pos_q = pos0[0] + t;
    const int j0 = is_swa ? max(0, pos_q - n_swa + 1) : 0;
    const int j1 = pos_q + 1;
    const float* rt = rel_tab + ((size_t) t * n_head + h) * n_ext;
    const float neg_sink = -sink[(size_t) t * n_head + h];

    float m = -1e30f;
    float l = 0.0f;
    float acc0 = 0.0f, acc1 = 0.0f, acc2 = 0.0f, acc3 = 0.0f;
    if (warp == 0) {
        m = 0.0f;
        l = 1.0f;
    }
    for (int j = j0 + warp; j < j1; j += 4) {
        const int slot = is_swa ? (j % cap) : j;
        const size_t idx = (size_t) slot * n_head_kv + hkv;
        const float d_k = __half2float(ks[idx]);
        const float d_v = __half2float(vs[idx]);
        const int8_t* kj = kq + (size_t) slot * kv_dim + hkv * 128;
        const int8_t* vj = vq + (size_t) slot * kv_dim + hkv * 128;
        const char4 kc4 = *reinterpret_cast<const char4*>(kj + base4);
        float dot = q0 * (float) kc4.x + q1 * (float) kc4.y + q2 * (float) kc4.z + q3 * (float) kc4.w;
#pragma unroll
        for (int o = 16; o > 0; o >>= 1) dot += __shfl_xor_sync(0xffffffffu, dot, o);
        const int dist = pos_q - j;
        const float bias = dist < n_ext ? rt[dist] : neg_sink;
        const float sc = dot * d_k * kq_scale + bias;
        const char4 vc4 = *reinterpret_cast<const char4*>(vj + base4);
        // FlashAttention-4 (2026) warp-uniform conditional softmax rescaling: skip expf+rescale when sc <= m (>95% of steps).
        if (sc > m) {
            const float alpha = l > 0.0f ? expf(m - sc) : 0.0f;
            l = l * alpha + 1.0f;
            acc0 = fmaf(acc0, alpha, d_v * (float) vc4.x);
            acc1 = fmaf(acc1, alpha, d_v * (float) vc4.y);
            acc2 = fmaf(acc2, alpha, d_v * (float) vc4.z);
            acc3 = fmaf(acc3, alpha, d_v * (float) vc4.w);
            m = sc;
        } else {
            const float p = expf(sc - m);
            const float pv = p * d_v;
            l += p;
            acc0 = fmaf(pv, (float) vc4.x, acc0);
            acc1 = fmaf(pv, (float) vc4.y, acc1);
            acc2 = fmaf(pv, (float) vc4.z, acc2);
            acc3 = fmaf(pv, (float) vc4.w, acc3);
        }
    }
    if (lane == 0) {
        s_m[warp] = m;
        s_l[warp] = l;
    }
    s_acc[warp][base4] = acc0;
    s_acc[warp][base4 + 1] = acc1;
    s_acc[warp][base4 + 2] = acc2;
    s_acc[warp][base4 + 3] = acc3;
    __syncthreads();
    if (threadIdx.x < 128) {
        float mg = s_m[0];
        for (int w = 1; w < 4; ++w) if (s_l[w] > 0.0f && s_m[w] > mg) mg = s_m[w];
        float lg = 0.0f;
        float og = 0.0f;
        for (int w = 0; w < 4; ++w) {
            if (s_l[w] > 0.0f) {
                const float c = expf(s_m[w] - mg);
                lg += s_l[w] * c;
                og += s_acc[w][threadIdx.x] * c;
            }
        }
        const size_t oi = ((size_t) t * n_head + h) * 128 + threadIdx.x;
        float val = og / lg;
        if (gate) val *= sigmf(gate[oi]);
        out[oi] = val;
    }
}

inline constexpr int kAttnSplitMax = 240;

/// GQA-3 prefill Q8 KV attention: 1 CTA of 128 threads processes all 3 Q-heads per KV head, 3x less KV HBM traffic.
/// grid (n_head_kv, T), block 128. Requires n_head == 3 * n_head_kv and head_dim == 128.
__global__ void k_attn_q8_gqa3_pp(const float* __restrict__ Q,
                               const int8_t* __restrict__ kq, const __half* __restrict__ ks,
                               const int8_t* __restrict__ vq, const __half* __restrict__ vs,
                               const float* __restrict__ rel_tab, const float* __restrict__ sink,
                               const float* __restrict__ gate, float* __restrict__ out,
                               int n_head, int n_head_kv, int n_ext, int cap, int is_swa, int n_swa,
                               const int32_t* __restrict__ pos0, float kq_scale) {
    __shared__ float sq[3][128];
    __shared__ float s_m[3][4];
    __shared__ float s_l[3][4];
    __shared__ float s_acc[3][4][128];
    const int hkv = blockIdx.x, t = blockIdx.y;
    const int warp = threadIdx.x >> 5, lane = threadIdx.x & 31;
    const int kv_dim = n_head_kv * 128;
    const int h0 = 3 * hkv;
    if (threadIdx.x < 128) {
#pragma unroll
        for (int g = 0; g < 3; ++g) {
            sq[g][threadIdx.x] = Q[((size_t) t * n_head + (h0 + g)) * 128 + threadIdx.x];
        }
    }
    __syncthreads();

    const int base4 = lane * 4;
    float q[3][4];
#pragma unroll
    for (int g = 0; g < 3; ++g) {
        q[g][0] = sq[g][base4];
        q[g][1] = sq[g][base4 + 1];
        q[g][2] = sq[g][base4 + 2];
        q[g][3] = sq[g][base4 + 3];
    }

    const int pos_q = pos0[0] + t;
    const int j0 = is_swa ? max(0, pos_q - n_swa + 1) : 0;
    const int j1 = pos_q + 1;
    const bool pow2_cap = (cap & (cap - 1)) == 0;
    const int cap_mask = cap - 1;

    float neg_sink[3];
    const float* rt[3];
#pragma unroll
    for (int g = 0; g < 3; ++g) {
        const int h = h0 + g;
        rt[g] = rel_tab + ((size_t) t * n_head + h) * n_ext;
        neg_sink[g] = -sink[(size_t) t * n_head + h];
    }

    float m[3], l[3], acc[3][4];
#pragma unroll
    for (int g = 0; g < 3; ++g) {
        m[g] = (warp == 0) ? 0.0f : -1e30f;
        l[g] = (warp == 0) ? 1.0f : 0.0f;
#pragma unroll
        for (int i = 0; i < 4; ++i) acc[g][i] = 0.0f;
    }

    for (int j = j0 + warp; j < j1; j += 4) {
        const int slot = is_swa ? (pow2_cap ? (j & cap_mask) : (j % cap)) : j;
        const size_t idx = (size_t) slot * n_head_kv + hkv;
        const float d_k = __half2float(ks[idx]);
        const float d_v = __half2float(vs[idx]);
        const int8_t* kj = kq + (size_t) slot * kv_dim + hkv * 128;
        const int8_t* vj = vq + (size_t) slot * kv_dim + hkv * 128;
        const char4 kc4 = *reinterpret_cast<const char4*>(kj + base4);
        const float k0 = (float) kc4.x, k1 = (float) kc4.y, k2 = (float) kc4.z, k3 = (float) kc4.w;
        const char4 vc4 = *reinterpret_cast<const char4*>(vj + base4);
        const float v0 = (float) vc4.x, v1 = (float) vc4.y, v2 = (float) vc4.z, v3 = (float) vc4.w;
        const int dist = pos_q - j;
        const float dkkq = d_k * kq_scale;

        float dot0 = q[0][0] * k0 + q[0][1] * k1 + q[0][2] * k2 + q[0][3] * k3;
        float dot1 = q[1][0] * k0 + q[1][1] * k1 + q[1][2] * k2 + q[1][3] * k3;
        float dot2 = q[2][0] * k0 + q[2][1] * k1 + q[2][2] * k2 + q[2][3] * k3;
#pragma unroll
        for (int o = 16; o > 0; o >>= 1) {
            dot0 += __shfl_xor_sync(0xffffffffu, dot0, o);
            dot1 += __shfl_xor_sync(0xffffffffu, dot1, o);
            dot2 += __shfl_xor_sync(0xffffffffu, dot2, o);
        }
        const float dots[3] = {dot0, dot1, dot2};
#pragma unroll
        for (int g = 0; g < 3; ++g) {
            const float bias = dist < n_ext ? rt[g][dist] : neg_sink[g];
            const float sc = dots[g] * dkkq + bias;
            // FlashAttention-4 (2026) warp-uniform conditional softmax rescaling.
            if (sc > m[g]) {
                const float alpha = l[g] > 0.0f ? __expf(m[g] - sc) : 0.0f;
                l[g] = l[g] * alpha + 1.0f;
                acc[g][0] = fmaf(acc[g][0], alpha, d_v * v0);
                acc[g][1] = fmaf(acc[g][1], alpha, d_v * v1);
                acc[g][2] = fmaf(acc[g][2], alpha, d_v * v2);
                acc[g][3] = fmaf(acc[g][3], alpha, d_v * v3);
                m[g] = sc;
            } else {
                const float p = __expf(sc - m[g]);
                const float pv = p * d_v;
                l[g] += p;
                acc[g][0] = fmaf(pv, v0, acc[g][0]);
                acc[g][1] = fmaf(pv, v1, acc[g][1]);
                acc[g][2] = fmaf(pv, v2, acc[g][2]);
                acc[g][3] = fmaf(pv, v3, acc[g][3]);
            }
        }
    }

#pragma unroll
    for (int g = 0; g < 3; ++g) {
        if (lane == 0) {
            s_m[g][warp] = m[g];
            s_l[g][warp] = l[g];
        }
        s_acc[g][warp][base4] = acc[g][0];
        s_acc[g][warp][base4 + 1] = acc[g][1];
        s_acc[g][warp][base4 + 2] = acc[g][2];
        s_acc[g][warp][base4 + 3] = acc[g][3];
    }
    __syncthreads();
    if (threadIdx.x < 128) {
#pragma unroll
        for (int g = 0; g < 3; ++g) {
            float mg = s_m[g][0];
            for (int w = 1; w < 4; ++w) if (s_l[g][w] > 0.0f && s_m[g][w] > mg) mg = s_m[g][w];
            float lg = 0.0f, og = 0.0f;
            for (int w = 0; w < 4; ++w) {
                if (s_l[g][w] > 0.0f) {
                    const float c = __expf(s_m[g][w] - mg);
                    lg += s_l[g][w] * c;
                    og += s_acc[g][w][threadIdx.x] * c;
                }
            }
            const size_t oi = ((size_t) t * n_head + (h0 + g)) * 128 + threadIdx.x;
            float val = __fdividef(og, lg);
            if (gate) val *= sigmf(gate[oi]);
            out[oi] = val;
        }
    }
}

#if !defined(__CUDA_ARCH__) || __CUDA_ARCH__ >= 530
/// Vectorized 4x int8 -> 2x half2 in-register decode via byte_perm + __hsub2 (bit-exact for [-128, 127]).
__device__ __forceinline__ void i8x4_to_h2x2(uint32_t packed_i8x4, __half2& out0, __half2& out1) {
    const uint32_t u = packed_i8x4 ^ 0x80808080u;
    const uint32_t lo = __byte_perm(u, 0x6400u, 0x5150u);
    const uint32_t hi = __byte_perm(u, 0x6400u, 0x5352u);
    const __half2 bias = __half2half2(__float2half(1152.0f));
    out0 = __hsub2(*reinterpret_cast<const __half2*>(&lo), bias);
    out1 = __hsub2(*reinterpret_cast<const __half2*>(&hi), bias);
}
#endif

/// FlashAttention-style FP16 WMMA prefill attention for GQA Q8 KV (head_dim=128, sm_70+).
/// One CTA (128 threads = 4 warps) computes 16 queries x 128 dims using 64-key tiles:
/// cuts KV HBM traffic 16x and runs Q*K^T & P*V on Tensor Cores while keeping int8 KV exact in FP16.
__global__ void __launch_bounds__(128, 3) k_attn_q8_gqa3_pp_wmma(
        const float* __restrict__ Q,
        const int8_t* __restrict__ kq, const __half* __restrict__ ks,
        const int8_t* __restrict__ vq, const __half* __restrict__ vs,
        const float* __restrict__ rel_tab, const float* __restrict__ sink,
        const float* __restrict__ gate, float* __restrict__ out,
        int n_head, int n_head_kv, int n_ext, int cap, int is_swa, int n_swa,
        const int32_t* __restrict__ pos0, float kq_scale, int T) {
#if !defined(__CUDA_ARCH__) || __CUDA_ARCH__ >= 700
    using namespace nvcuda;
    extern __shared__ char smem_raw[];
    // Align each region to 16 bytes:
    // sQ: 16*136 __half (4352 B) | sKV: 64*136 __half (17408 B, aliases s_tile 16*136 float = 8704 B)
    // sS: 16*64 float (4096 B)   | sP: 16*72 __half (2304 B) | scalars (< 1 KB)
    __half* sQ = reinterpret_cast<__half*>(smem_raw);
    __half* sKV = reinterpret_cast<__half*>(smem_raw + 4352);
    float* s_tile = reinterpret_cast<float*>(sKV);
    float* sS = reinterpret_cast<float*>(smem_raw + 4352 + 17408);
    __half* sP = reinterpret_cast<__half*>(smem_raw + 4352 + 17408 + 4096);
    float* s_qscale = reinterpret_cast<float*>(smem_raw + 4352 + 17408 + 4096 + 2304);
    float* s_dk = s_qscale + 16;
    float* s_dv = s_dk + 64;
    float* s_m = s_dv + 64;
    float* s_l = s_m + 16;
    float* s_alpha = s_l + 16;
    float* s_pvmax = s_alpha + 16;

    const int h = blockIdx.x;
    const int gqa = n_head / n_head_kv;
    const int hkv = h / gqa;
    const int t0 = blockIdx.y * 16;
    const int tid = threadIdx.x;
    const int warp = tid >> 5;
    const int lane = tid & 31;
    const int kv_dim = n_head_kv * 128;

    // Cooperatively load Q[t0..t0+15][h][0..127] (8 threads/row x 16 floats) and normalize per row for FP16 safety.
    {
        const int qr = tid >> 3;
        const int qc0 = (tid & 7) * 16;
        const int t = t0 + qr;
        float amax = 0.0f;
        const float* qp = (t < T) ? (Q + ((size_t) t * n_head + h) * 128 + qc0) : nullptr;
        if (qp) {
#pragma unroll
            for (int i = 0; i < 4; ++i) {
                const float4 v = *reinterpret_cast<const float4*>(qp + i * 4);
                amax = fmaxf(amax, fmaxf(fmaxf(fabsf(v.x), fabsf(v.y)), fmaxf(fabsf(v.z), fabsf(v.w))));
            }
        }
        amax = fmaxf(amax, __shfl_xor_sync(0xffffffffu, amax, 4));
        amax = fmaxf(amax, __shfl_xor_sync(0xffffffffu, amax, 2));
        amax = fmaxf(amax, __shfl_xor_sync(0xffffffffu, amax, 1));
        const float qscale = (amax > 0.0f) ? amax : 1.0f;
        const float inv_qs = __fdividef(1.0f, qscale);
        if ((tid & 7) == 0) {
            s_qscale[qr] = qscale;
            s_m[qr] = 0.0f;
            s_l[qr] = 1.0f;
        }
        __half2* dst2 = reinterpret_cast<__half2*>(sQ + qr * 136 + qc0);
        if (qp) {
#pragma unroll
            for (int i = 0; i < 4; ++i) {
                const float4 v = *reinterpret_cast<const float4*>(qp + i * 4);
                dst2[i * 2 + 0] = __halves2half2(__float2half(v.x * inv_qs), __float2half(v.y * inv_qs));
                dst2[i * 2 + 1] = __halves2half2(__float2half(v.z * inv_qs), __float2half(v.w * inv_qs));
            }
        } else {
            const __half2 z2 = __halves2half2(__float2half(0.0f), __float2half(0.0f));
#pragma unroll
            for (int i = 0; i < 8; ++i) dst2[i] = z2;
        }
    }
    __syncthreads();

    const int p0 = pos0[0];
    const int t_last = min(t0 + 15, T - 1);
    const int j_start = is_swa ? max(0, (p0 + t0) - n_swa + 1) : 0;
    const int j_end = p0 + t_last + 1;
    const int bulk_end = is_swa ? j_start : max(j_start, min(j_end, p0 + t0 - n_ext + 1));
    const bool pow2_cap = (cap & (cap - 1)) == 0;
    const int cap_mask = cap - 1;

    float acc[16];
#pragma unroll
    for (int r = 0; r < 16; ++r) acc[r] = 0.0f;

    const int kv_r = tid >> 1;
    const int kv_c0 = (tid & 1) * 64;

    auto step_tile = [&](int jb, bool is_bulk) {
        // 1. Load K[jb..jb+63] into sKV[64][136] + s_dk, s_dv
        {
            const int j = jb + kv_r;
            const bool valid = is_bulk ? true : (j < j_end);
            const int slot = is_bulk ? j : (valid ? (is_swa ? (pow2_cap ? (j & cap_mask) : (j % cap)) : j) : 0);
            const size_t idx = (size_t) slot * n_head_kv + hkv;
            if ((tid & 1) == 0) {
                s_dk[kv_r] = valid ? __half2float(__ldg(&ks[idx])) : 0.0f;
                s_dv[kv_r] = valid ? __half2float(__ldg(&vs[idx])) : 0.0f;
            }
            __half* dst = sKV + kv_r * 136 + kv_c0;
            if (valid) {
                const int8_t* kp = kq + (size_t) slot * kv_dim + hkv * 128 + kv_c0;
#pragma unroll
                for (int i = 0; i < 4; ++i) {
                    const uint4 raw = *reinterpret_cast<const uint4*>(kp + i * 16);
                    __half2 h2[8];
                    i8x4_to_h2x2(raw.x, h2[0], h2[1]);
                    i8x4_to_h2x2(raw.y, h2[2], h2[3]);
                    i8x4_to_h2x2(raw.z, h2[4], h2[5]);
                    i8x4_to_h2x2(raw.w, h2[6], h2[7]);
                    *reinterpret_cast<uint4*>(dst + i * 16) = *reinterpret_cast<const uint4*>(h2);
                    *reinterpret_cast<uint4*>(dst + i * 16 + 8) = *reinterpret_cast<const uint4*>(h2 + 4);
                }
            } else {
                const uint4 z = make_uint4(0, 0, 0, 0);
#pragma unroll
                for (int i = 0; i < 8; ++i) *reinterpret_cast<uint4*>(dst + i * 8) = z;
            }
        }
        __syncthreads();

        // 2. S = Q * K^T ([16, 128] x [64, 128]^T -> [16, 64], 1 16x16 tile per warp)
        {
            wmma::fragment<wmma::matrix_a, 16, 16, 16, __half, wmma::row_major> fa;
            wmma::fragment<wmma::matrix_b, 16, 16, 16, __half, wmma::col_major> fb;
            wmma::fragment<wmma::accumulator, 16, 16, 16, float> fc;
            wmma::fill_fragment(fc, 0.0f);
#pragma unroll
            for (int k = 0; k < 128; k += 16) {
                wmma::load_matrix_sync(fa, sQ + k, 136);
                wmma::load_matrix_sync(fb, sKV + (warp * 16) * 136 + k, 136);
                wmma::mma_sync(fc, fa, fb, fc);
            }
            wmma::store_matrix_sync(sS + warp * 16, fc, 64, wmma::mem_row_major);
        }
        __syncthreads();

        // 3. Load V[jb..jb+63] into sKV[64][136] AND run online softmax (4 rows per warp) -> sP[16][72]
        {
            const int j = jb + kv_r;
            const bool valid = is_bulk ? true : (j < j_end);
            __half* dst = sKV + kv_r * 136 + kv_c0;
            if (valid) {
                const int slot = is_bulk ? j : (is_swa ? (pow2_cap ? (j & cap_mask) : (j % cap)) : j);
                const int8_t* vp = vq + (size_t) slot * kv_dim + hkv * 128 + kv_c0;
#pragma unroll
                for (int i = 0; i < 4; ++i) {
                    const uint4 raw = *reinterpret_cast<const uint4*>(vp + i * 16);
                    __half2 h2[8];
                    i8x4_to_h2x2(raw.x, h2[0], h2[1]);
                    i8x4_to_h2x2(raw.y, h2[2], h2[3]);
                    i8x4_to_h2x2(raw.z, h2[4], h2[5]);
                    i8x4_to_h2x2(raw.w, h2[6], h2[7]);
                    *reinterpret_cast<uint4*>(dst + i * 16) = *reinterpret_cast<const uint4*>(h2);
                    *reinterpret_cast<uint4*>(dst + i * 16 + 8) = *reinterpret_cast<const uint4*>(h2 + 4);
                }
            } else {
                const uint4 z = make_uint4(0, 0, 0, 0);
#pragma unroll
                for (int i = 0; i < 8; ++i) *reinterpret_cast<uint4*>(dst + i * 8) = z;
            }

            const int col0 = lane;
            const int col1 = lane + 32;
            const int j_0 = jb + col0;
            const int j_1 = jb + col1;
            const float dk0 = s_dk[col0], dk1 = s_dk[col1];
            const float dv0 = s_dv[col0], dv1 = s_dv[col1];

#pragma unroll
            for (int rr = 0; rr < 4; ++rr) {
                const int r = warp * 4 + rr;
                const int t = t0 + r;
                const bool t_ok = t < T;
                const float nsink = t_ok ? -sink[(size_t) t * n_head + h] : 0.0f;
                const float qks = s_qscale[r] * kq_scale;

                float sc0, sc1;
                bool v0, v1;
                if (is_bulk) {
                    v0 = t_ok;
                    v1 = t_ok;
                    sc0 = t_ok ? (sS[r * 64 + col0] * qks * dk0 + nsink) : -1e30f;
                    sc1 = t_ok ? (sS[r * 64 + col1] * qks * dk1 + nsink) : -1e30f;
                } else {
                    const int pos_q = p0 + t;
                    const float* rt = t_ok ? (rel_tab + ((size_t) t * n_head + h) * n_ext) : nullptr;
                    const int d0 = pos_q - j_0;
                    const int d1 = pos_q - j_1;
                    v0 = t_ok && (j_0 <= pos_q) && (!is_swa || d0 < n_swa);
                    v1 = t_ok && (j_1 <= pos_q) && (!is_swa || d1 < n_swa);
                    sc0 = v0 ? (sS[r * 64 + col0] * qks * dk0 + (d0 < n_ext ? rt[d0] : nsink)) : -1e30f;
                    sc1 = v1 ? (sS[r * 64 + col1] * qks * dk1 + (d1 < n_ext ? rt[d1] : nsink)) : -1e30f;
                }

                float rmax = fmaxf(sc0, sc1);
#pragma unroll
                for (int o = 16; o > 0; o >>= 1) rmax = fmaxf(rmax, __shfl_xor_sync(0xffffffffu, rmax, o));

                const float m_old = s_m[r];
                const float l_old = s_l[r];
                const float m_new = fmaxf(m_old, rmax);
                const float alpha = l_old > 0.0f ? __expf(m_old - m_new) : 0.0f;
                const float p_0 = v0 ? __expf(sc0 - m_new) : 0.0f;
                const float p_1 = v1 ? __expf(sc1 - m_new) : 0.0f;

                float psum = p_0 + p_1;
#pragma unroll
                for (int o = 16; o > 0; o >>= 1) psum += __shfl_xor_sync(0xffffffffu, psum, o);

                const float pv_0 = p_0 * dv0;
                const float pv_1 = p_1 * dv1;
                float pvmax = fmaxf(pv_0, pv_1);
#pragma unroll
                for (int o = 16; o > 0; o >>= 1) pvmax = fmaxf(pvmax, __shfl_xor_sync(0xffffffffu, pvmax, o));
                const float inv_pv = pvmax > 0.0f ? __fdividef(1.0f, pvmax) : 0.0f;

                sP[r * 72 + col0] = __float2half(pv_0 * inv_pv);
                sP[r * 72 + col1] = __float2half(pv_1 * inv_pv);
                if (lane == 0) {
                    s_m[r] = m_new;
                    s_l[r] = l_old * alpha + psum;
                    s_alpha[r] = alpha;
                    s_pvmax[r] = pvmax;
                }
            }
        }
        __syncthreads();

        // 4. O_tile = P * V ([16, 64] x [64, 128] -> [16, 128], 2 16x16 tiles = 32 cols per warp)
        {
            wmma::fragment<wmma::matrix_a, 16, 16, 16, __half, wmma::row_major> fa;
            wmma::fragment<wmma::matrix_b, 16, 16, 16, __half, wmma::row_major> fb0, fb1;
            wmma::fragment<wmma::accumulator, 16, 16, 16, float> fc0, fc1;
            wmma::fill_fragment(fc0, 0.0f);
            wmma::fill_fragment(fc1, 0.0f);
            const int cbase = warp * 32;
#pragma unroll
            for (int k = 0; k < 64; k += 16) {
                wmma::load_matrix_sync(fa, sP + k, 72);
                wmma::load_matrix_sync(fb0, sKV + k * 136 + cbase, 136);
                wmma::mma_sync(fc0, fa, fb0, fc0);
                wmma::load_matrix_sync(fb1, sKV + k * 136 + cbase + 16, 136);
                wmma::mma_sync(fc1, fa, fb1, fc1);
            }
            __syncthreads();
            wmma::store_matrix_sync(s_tile + cbase, fc0, 136, wmma::mem_row_major);
            wmma::store_matrix_sync(s_tile + cbase + 16, fc1, 136, wmma::mem_row_major);
        }
        __syncthreads();

        // 5. Rescale + accumulate into thread's own column across all 16 rows
#pragma unroll
        for (int r = 0; r < 16; ++r) {
            acc[r] = acc[r] * s_alpha[r] + s_pvmax[r] * s_tile[r * 136 + tid];
        }
        __syncthreads();
    };

    int jb = j_start;
    for (; jb + 64 <= bulk_end; jb += 64) step_tile(jb, true);
    for (; jb < j_end; jb += 64) step_tile(jb, false);

#pragma unroll
    for (int r = 0; r < 16; ++r) {
        const int t = t0 + r;
        if (t < T) {
            const size_t oi = ((size_t) t * n_head + h) * 128 + tid;
            float val = __fdividef(acc[r], s_l[r]);
            if (gate) val *= sigmf(gate[oi]);
            out[oi] = val;
        }
    }
#endif
}

/// 32-query-tile (256 threads = 8 warps) FP16 WMMA prefill attention for GQA Q8 KV (head_dim=128, T >= 128):
/// Cuts KV HBM2 reads & syncs 2x relative to the 16-row tile with balanced 8-warp WMMA utilization.
__global__ void __launch_bounds__(256, 2) k_attn_q8_gqa3_pp_wmma_t32(
        const float* __restrict__ Q,
        const int8_t* __restrict__ kq, const __half* __restrict__ ks,
        const int8_t* __restrict__ vq, const __half* __restrict__ vs,
        const float* __restrict__ rel_tab, const float* __restrict__ sink,
        const float* __restrict__ gate, float* __restrict__ out,
        int n_head, int n_head_kv, int n_ext, int cap, int is_swa, int n_swa,
        const int32_t* __restrict__ pos0, float kq_scale, int T) {
#if !defined(__CUDA_ARCH__) || __CUDA_ARCH__ >= 700
    using namespace nvcuda;
    extern __shared__ __align__(16) char smem_raw[];
    // 48,768 B total (fits 2 blk/SM in 96/100 KB): sQ (8704) | sKV (17408, separate!) | sP (4608) |
    // sS (8192) overlaid by s_tile (32*132 = 16896) | scalars (1152). Cuts barriers from 6 -> 4 per tile.
    __half* sQ = reinterpret_cast<__half*>(smem_raw);
    __half* sKV = sQ + 32 * 136;
    __half* sP = sKV + 64 * 136;
    float* sS = reinterpret_cast<float*>(sP + 32 * 72);
    float* s_tile = sS;
    float* s_qscale = s_tile + 32 * 132;
    float* s_dk = s_qscale + 32;
    float* s_dv = s_dk + 64;
    float* s_m = s_dv + 64;
    float* s_l = s_m + 32;
    float* s_alpha = s_l + 32;
    float* s_pvmax = s_alpha + 32;

    const int h = blockIdx.x;
    const int hkv = h / (n_head / n_head_kv);
    const int t0 = blockIdx.y * 32;
    const int tid = threadIdx.x;
    const int warp = tid >> 5;
    const int lane = tid & 31;
    const int kv_dim = n_head_kv * 128;
    constexpr float kLog2e = 1.4426950408889634f;

    {
        const int qr = tid >> 3;
        const int qc0 = (tid & 7) * 16;
        const int t = t0 + qr;
        float amax = 0.0f;
        const float* qp = (t < T) ? (Q + ((size_t) t * n_head + h) * 128 + qc0) : nullptr;
        if (qp) {
#pragma unroll
            for (int i = 0; i < 4; ++i) {
                const float4 v = *reinterpret_cast<const float4*>(qp + i * 4);
                amax = fmaxf(amax, fmaxf(fmaxf(fabsf(v.x), fabsf(v.y)), fmaxf(fabsf(v.z), fabsf(v.w))));
            }
        }
        amax = fmaxf(amax, __shfl_xor_sync(0xffffffffu, amax, 4));
        amax = fmaxf(amax, __shfl_xor_sync(0xffffffffu, amax, 2));
        amax = fmaxf(amax, __shfl_xor_sync(0xffffffffu, amax, 1));
        const float qscale = (amax > 0.0f) ? amax : 1.0f;
        const float inv_qs = __fdividef(1.0f, qscale);
        if ((tid & 7) == 0) {
            s_qscale[qr] = qscale * kq_scale * kLog2e;
            s_m[qr] = 0.0f;
            s_l[qr] = 1.0f;
        }
        __half2* dst2 = reinterpret_cast<__half2*>(sQ + qr * 136 + qc0);
        if (qp) {
#pragma unroll
            for (int i = 0; i < 4; ++i) {
                const float4 v = *reinterpret_cast<const float4*>(qp + i * 4);
                dst2[i * 2 + 0] = __halves2half2(__float2half(v.x * inv_qs), __float2half(v.y * inv_qs));
                dst2[i * 2 + 1] = __halves2half2(__float2half(v.z * inv_qs), __float2half(v.w * inv_qs));
            }
        } else {
            const __half2 z2 = __halves2half2(__float2half(0.0f), __float2half(0.0f));
#pragma unroll
            for (int i = 0; i < 8; ++i) dst2[i] = z2;
        }
    }
    __syncthreads();

    const int p0 = pos0[0];
    const int t_last = min(t0 + 31, T - 1);
    const int j_start = is_swa ? max(0, (p0 + t0) - n_swa + 1) : 0;
    const int j_end = p0 + t_last + 1;
    const int bulk_end = is_swa ? j_start : max(j_start, min(j_end, p0 + t0 - n_ext + 1));
    const bool pow2_cap = (cap & (cap - 1)) == 0;
    const int cap_mask = cap - 1;

    float acc[16] = {};
    const int kv_r = tid >> 2;
    const int kv_c0 = (tid & 3) * 32;
    const int my_col = tid & 127;
    const int my_half = tid >> 7;

    auto step_tile = [&](int jb, bool is_bulk) {
        {
            const int j = jb + kv_r;
            const bool valid = is_bulk ? true : (j < j_end);
            const int slot = is_bulk ? j : (valid ? (is_swa ? (pow2_cap ? (j & cap_mask) : (j % cap)) : j) : 0);
            if ((tid & 3) == 0) {
                const size_t idx = (size_t) slot * n_head_kv + hkv;
                s_dk[kv_r] = valid ? __half2float(__ldg(&ks[idx])) : 0.0f;
                s_dv[kv_r] = valid ? __half2float(__ldg(&vs[idx])) : 0.0f;
            }
            __half* dst = sKV + kv_r * 136 + kv_c0;
            if (valid) {
                const int8_t* kp = kq + (size_t) slot * kv_dim + hkv * 128 + kv_c0;
#pragma unroll
                for (int i = 0; i < 2; ++i) {
                    const uint4 raw = *reinterpret_cast<const uint4*>(kp + i * 16);
                    __half2 h2[8];
                    i8x4_to_h2x2(raw.x, h2[0], h2[1]);
                    i8x4_to_h2x2(raw.y, h2[2], h2[3]);
                    i8x4_to_h2x2(raw.z, h2[4], h2[5]);
                    i8x4_to_h2x2(raw.w, h2[6], h2[7]);
                    *reinterpret_cast<uint4*>(dst + i * 16) = *reinterpret_cast<const uint4*>(h2);
                    *reinterpret_cast<uint4*>(dst + i * 16 + 8) = *reinterpret_cast<const uint4*>(h2 + 4);
                }
            } else {
                const uint4 z = make_uint4(0, 0, 0, 0);
#pragma unroll
                for (int i = 0; i < 4; ++i) *reinterpret_cast<uint4*>(dst + i * 8) = z;
            }
        }
        __syncthreads();

        {
            const int rt = warp >> 2;
            const int ct = warp & 3;
            wmma::fragment<wmma::matrix_a, 16, 16, 16, __half, wmma::row_major> fa;
            wmma::fragment<wmma::matrix_b, 16, 16, 16, __half, wmma::col_major> fb;
            wmma::fragment<wmma::accumulator, 16, 16, 16, float> fc;
            wmma::fill_fragment(fc, 0.0f);
#pragma unroll
            for (int k = 0; k < 128; k += 16) {
                wmma::load_matrix_sync(fa, sQ + (rt * 16) * 136 + k, 136);
                wmma::load_matrix_sync(fb, sKV + (ct * 16) * 136 + k, 136);
                wmma::mma_sync(fc, fa, fb, fc);
            }
            wmma::store_matrix_sync(sS + (rt * 16) * 64 + ct * 16, fc, 64, wmma::mem_row_major);
        }
        __syncthreads();

        {
            const int j = jb + kv_r;
            const bool valid = is_bulk ? true : (j < j_end);
            __half* dst = sKV + kv_r * 136 + kv_c0;
            if (valid) {
                const int slot = is_bulk ? j : (is_swa ? (pow2_cap ? (j & cap_mask) : (j % cap)) : j);
                const int8_t* vp = vq + (size_t) slot * kv_dim + hkv * 128 + kv_c0;
#pragma unroll
                for (int i = 0; i < 2; ++i) {
                    const uint4 raw = *reinterpret_cast<const uint4*>(vp + i * 16);
                    __half2 h2[8];
                    i8x4_to_h2x2(raw.x, h2[0], h2[1]);
                    i8x4_to_h2x2(raw.y, h2[2], h2[3]);
                    i8x4_to_h2x2(raw.z, h2[4], h2[5]);
                    i8x4_to_h2x2(raw.w, h2[6], h2[7]);
                    *reinterpret_cast<uint4*>(dst + i * 16) = *reinterpret_cast<const uint4*>(h2);
                    *reinterpret_cast<uint4*>(dst + i * 16 + 8) = *reinterpret_cast<const uint4*>(h2 + 4);
                }
            } else {
                const uint4 z = make_uint4(0, 0, 0, 0);
#pragma unroll
                for (int i = 0; i < 4; ++i) *reinterpret_cast<uint4*>(dst + i * 8) = z;
            }

            const int col0 = lane;
            const int col1 = lane + 32;
            const int j_0 = jb + col0;
            const int j_1 = jb + col1;
            const float dk0 = s_dk[col0], dk1 = s_dk[col1];
            const float dv0 = s_dv[col0], dv1 = s_dv[col1];

#pragma unroll
            for (int rr = 0; rr < 4; ++rr) {
                const int r = warp * 4 + rr;
                const int t = t0 + r;
                const bool t_ok = t < T;
                const float nsink = t_ok ? (-sink[(size_t) t * n_head + h] * kLog2e) : 0.0f;
                const float qks2 = s_qscale[r];

                float sc0, sc1;
                bool v0, v1;
                if (is_bulk) {
                    v0 = t_ok;
                    v1 = t_ok;
                    sc0 = t_ok ? (sS[r * 64 + col0] * qks2 * dk0 + nsink) : -1e30f;
                    sc1 = t_ok ? (sS[r * 64 + col1] * qks2 * dk1 + nsink) : -1e30f;
                } else {
                    const int pos_q = p0 + t;
                    const float* rt = t_ok ? (rel_tab + ((size_t) t * n_head + h) * n_ext) : nullptr;
                    const int d0 = pos_q - j_0;
                    const int d1 = pos_q - j_1;
                    v0 = t_ok && (j_0 <= pos_q) && (!is_swa || d0 < n_swa);
                    v1 = t_ok && (j_1 <= pos_q) && (!is_swa || d1 < n_swa);
                    sc0 = v0 ? (sS[r * 64 + col0] * qks2 * dk0 + (d0 < n_ext ? (rt[d0] * kLog2e) : nsink)) : -1e30f;
                    sc1 = v1 ? (sS[r * 64 + col1] * qks2 * dk1 + (d1 < n_ext ? (rt[d1] * kLog2e) : nsink)) : -1e30f;
                }

                float rmax = fmaxf(sc0, sc1);
#pragma unroll
                for (int o = 16; o > 0; o >>= 1) rmax = fmaxf(rmax, __shfl_xor_sync(0xffffffffu, rmax, o));
                const float m_old = s_m[r], l_old = s_l[r];
                const float m_new = fmaxf(m_old, rmax);
                const float alpha = l_old > 0.0f ? exp2f(m_old - m_new) : 0.0f;
                const float p_0 = v0 ? exp2f(sc0 - m_new) : 0.0f;
                const float p_1 = v1 ? exp2f(sc1 - m_new) : 0.0f;
                float psum = p_0 + p_1;
#pragma unroll
                for (int o = 16; o > 0; o >>= 1) psum += __shfl_xor_sync(0xffffffffu, psum, o);

                const float pv_0 = p_0 * dv0, pv_1 = p_1 * dv1;
                float pvmax = fmaxf(pv_0, pv_1);
#pragma unroll
                for (int o = 16; o > 0; o >>= 1) pvmax = fmaxf(pvmax, __shfl_xor_sync(0xffffffffu, pvmax, o));
                const float inv_pv = pvmax > 0.0f ? __fdividef(1.0f, pvmax) : 0.0f;

                sP[r * 72 + col0] = __float2half(pv_0 * inv_pv);
                sP[r * 72 + col1] = __float2half(pv_1 * inv_pv);
                if (lane == 0) {
                    s_m[r] = m_new;
                    s_l[r] = l_old * alpha + psum;
                    s_alpha[r] = alpha;
                    s_pvmax[r] = pvmax;
                }
            }
        }
        __syncthreads();

        {
            const int cbase = warp * 16;
            wmma::fragment<wmma::accumulator, 16, 16, 16, float> fc0, fc1;
            wmma::fill_fragment(fc0, 0.0f);
            wmma::fill_fragment(fc1, 0.0f);
#pragma unroll
            for (int k = 0; k < 64; k += 16) {
                wmma::fragment<wmma::matrix_a, 16, 16, 16, __half, wmma::row_major> fa0, fa1;
                wmma::fragment<wmma::matrix_b, 16, 16, 16, __half, wmma::row_major> fb;
                wmma::load_matrix_sync(fa0, sP + k, 72);
                wmma::load_matrix_sync(fa1, sP + 16 * 72 + k, 72);
                wmma::load_matrix_sync(fb, sKV + k * 136 + cbase, 136);
                wmma::mma_sync(fc0, fa0, fb, fc0);
                wmma::mma_sync(fc1, fa1, fb, fc1);
            }
            wmma::store_matrix_sync(s_tile + cbase, fc0, 132, wmma::mem_row_major);
            wmma::store_matrix_sync(s_tile + 16 * 132 + cbase, fc1, 132, wmma::mem_row_major);
        }
        __syncthreads();

        const int row0 = my_half * 16;
#pragma unroll
        for (int r = 0; r < 16; ++r) {
            acc[r] = acc[r] * s_alpha[row0 + r] + s_pvmax[row0 + r] * s_tile[(row0 + r) * 132 + my_col];
        }
    };

    int jb = j_start;
    for (; jb + 64 <= bulk_end; jb += 64) step_tile(jb, true);
    for (; jb < j_end; jb += 64) step_tile(jb, false);

    const int row0 = my_half * 16;
#pragma unroll
    for (int r = 0; r < 16; ++r) {
        const int t = t0 + row0 + r;
        if (t < T) {
            const size_t oi = ((size_t) t * n_head + h) * 128 + my_col;
            float val = __fdividef(acc[r], s_l[row0 + r]);
            if (gate) val *= sigmf(gate[oi]);
            out[oi] = val;
        }
    }
#endif
}

/// GQA-3 Shared-sKV 48-row (16 tokens x 3 GQA heads, 256 threads = 8 warps) FP16 WMMA prefill attention (head_dim=128, sm_70+).
/// Loads and i8x4_to_h2x2-decodes K/V ONCE per BK=64 tile and reuses across all 3 GQA heads in bank-conflict-free 44,224 B SMEM (2 blk/SM).
__global__ void __launch_bounds__(256, 2) k_attn_q8_gqa3_pp_wmma_gqa3(
        const float* __restrict__ Q,
        const int8_t* __restrict__ kq, const __half* __restrict__ ks,
        const int8_t* __restrict__ vq, const __half* __restrict__ vs,
        const float* __restrict__ rel_tab, const float* __restrict__ sink,
        const float* __restrict__ gate, float* __restrict__ out,
        int n_head, int n_head_kv, int n_ext, int cap, int is_swa, int n_swa,
        const int32_t* __restrict__ pos0, float kq_scale, int T) {
#if !defined(__CUDA_ARCH__) || __CUDA_ARCH__ >= 700
    using namespace nvcuda;
    extern __shared__ __align__(16) char smem_raw[];
    // 44,224 B <= 49,152 B (2 blk/SM on 96/100 KB SMEM):
    // sQ (48*136*2 = 13056) | sKV (64*136*2 = 17408) | sS (48*64*4 = 12288, reused as sP 48*72*2 = 6912) | scalars (1472).
    // s_tile (48*132*4 = 25344) overlays dead sKV+sS (29696) for single-shot 3-head P*V store.
    __half* sQ = reinterpret_cast<__half*>(smem_raw);
    __half* sKV = sQ + 48 * 136;
    float* sS = reinterpret_cast<float*>(sKV + 64 * 136);
    __half* sP = reinterpret_cast<__half*>(sS);
    float* s_tile = reinterpret_cast<float*>(sKV);
    float* s_qscale = sS + 48 * 64;
    float* s_dk = s_qscale + 48;
    float* s_dv = s_dk + 64;
    float* s_m = s_dv + 64;
    float* s_l = s_m + 48;
    float* s_alpha = s_l + 48;
    float* s_pvmax = s_alpha + 48;

    const int hkv = blockIdx.x;
    const int h0 = 3 * hkv;
    const int t0 = blockIdx.y * 16;
    const int tid = threadIdx.x;
    const int warp = tid >> 5;
    const int lane = tid & 31;
    const int kv_dim = n_head_kv * 128;
    constexpr float kLog2e = 1.4426950408889634f;

    for (int base_r = 0; base_r < 48; base_r += 32) {
        const int qr = base_r + (tid >> 3);
        const int qc0 = (tid & 7) * 16;
        if (qr < 48) {
            const int g = qr >> 4;
            const int r = qr & 15;
            const int t = t0 + r;
            const int h = h0 + g;
            float amax = 0.0f;
            const float* qp = (t < T) ? (Q + ((size_t) t * n_head + h) * 128 + qc0) : nullptr;
            if (qp) {
#pragma unroll
                for (int i = 0; i < 4; ++i) {
                    const float4 v = *reinterpret_cast<const float4*>(qp + i * 4);
                    amax = fmaxf(amax, fmaxf(fmaxf(fabsf(v.x), fabsf(v.y)), fmaxf(fabsf(v.z), fabsf(v.w))));
                }
            }
            amax = fmaxf(amax, __shfl_xor_sync(0xffffffffu, amax, 4));
            amax = fmaxf(amax, __shfl_xor_sync(0xffffffffu, amax, 2));
            amax = fmaxf(amax, __shfl_xor_sync(0xffffffffu, amax, 1));
            const float qscale = (amax > 0.0f) ? amax : 1.0f;
            const float inv_qs = __fdividef(1.0f, qscale);
            if ((tid & 7) == 0) {
                s_qscale[qr] = qscale * kq_scale * kLog2e;
                s_m[qr] = 0.0f;
                s_l[qr] = 1.0f;
            }
            __half2* dst2 = reinterpret_cast<__half2*>(sQ + qr * 136 + qc0);
            if (qp) {
#pragma unroll
                for (int i = 0; i < 4; ++i) {
                    const float4 v = *reinterpret_cast<const float4*>(qp + i * 4);
                    dst2[i * 2 + 0] = __halves2half2(__float2half(v.x * inv_qs), __float2half(v.y * inv_qs));
                    dst2[i * 2 + 1] = __halves2half2(__float2half(v.z * inv_qs), __float2half(v.w * inv_qs));
                }
            } else {
                const __half2 z2 = __halves2half2(__float2half(0.0f), __float2half(0.0f));
#pragma unroll
                for (int i = 0; i < 8; ++i) dst2[i] = z2;
            }
        }
    }
    __syncthreads();

    const int p0 = pos0[0];
    const int t_last = min(t0 + 15, T - 1);
    const int j_start = is_swa ? max(0, (p0 + t0) - n_swa + 1) : 0;
    const int j_end = p0 + t_last + 1;
    const int bulk_end = is_swa ? j_start : max(j_start, min(j_end, p0 + t0 - n_ext + 1));
    const bool pow2_cap = (cap & (cap - 1)) == 0;
    const int cap_mask = cap - 1;

    float acc[3][8] = {};
    const int kv_r = tid >> 2;
    const int kv_c0 = (tid & 3) * 32;
    const int my_col = tid & 127;
    const int my_half = tid >> 7;

    auto step_tile = [&](int jb, bool is_bulk) {
        {
            const int j = jb + kv_r;
            const bool valid = is_bulk ? true : (j < j_end);
            const int slot = is_bulk ? j : (valid ? (is_swa ? (pow2_cap ? (j & cap_mask) : (j % cap)) : j) : 0);
            if ((tid & 3) == 0) {
                const size_t idx = (size_t) slot * n_head_kv + hkv;
                s_dk[kv_r] = valid ? __half2float(__ldg(&ks[idx])) : 0.0f;
                s_dv[kv_r] = valid ? __half2float(__ldg(&vs[idx])) : 0.0f;
            }
            __half* dst = sKV + kv_r * 136 + kv_c0;
            if (valid) {
                const int8_t* kp = kq + (size_t) slot * kv_dim + hkv * 128 + kv_c0;
#pragma unroll
                for (int i = 0; i < 2; ++i) {
                    const uint4 raw = *reinterpret_cast<const uint4*>(kp + i * 16);
                    __half2 h2[8];
                    i8x4_to_h2x2(raw.x, h2[0], h2[1]);
                    i8x4_to_h2x2(raw.y, h2[2], h2[3]);
                    i8x4_to_h2x2(raw.z, h2[4], h2[5]);
                    i8x4_to_h2x2(raw.w, h2[6], h2[7]);
                    *reinterpret_cast<uint4*>(dst + i * 16) = *reinterpret_cast<const uint4*>(h2);
                    *reinterpret_cast<uint4*>(dst + i * 16 + 8) = *reinterpret_cast<const uint4*>(h2 + 4);
                }
            } else {
                const uint4 z = make_uint4(0, 0, 0, 0);
#pragma unroll
                for (int i = 0; i < 4; ++i) *reinterpret_cast<uint4*>(dst + i * 8) = z;
            }
        }
        __syncthreads();

        {
            const int ct = warp & 3;
            const int wh = warp >> 2;
            wmma::fragment<wmma::matrix_a, 16, 16, 16, __half, wmma::row_major> fa0, fa1;
            wmma::fragment<wmma::matrix_b, 16, 16, 16, __half, wmma::col_major> fb;
            if (wh == 0) {
                wmma::fragment<wmma::accumulator, 16, 16, 16, float> fc0, fc2;
                wmma::fill_fragment(fc0, 0.0f);
                wmma::fill_fragment(fc2, 0.0f);
#pragma unroll
                for (int k = 0; k < 128; k += 16) {
                    wmma::load_matrix_sync(fb, sKV + (ct * 16) * 136 + k, 136);
                    wmma::load_matrix_sync(fa0, sQ + k, 136);
                    wmma::load_matrix_sync(fa1, sQ + 32 * 136 + k, 136);
                    wmma::mma_sync(fc0, fa0, fb, fc0);
                    wmma::mma_sync(fc2, fa1, fb, fc2);
                }
                wmma::store_matrix_sync(sS + ct * 16, fc0, 64, wmma::mem_row_major);
                wmma::store_matrix_sync(sS + 32 * 64 + ct * 16, fc2, 64, wmma::mem_row_major);
            } else {
                wmma::fragment<wmma::accumulator, 16, 16, 16, float> fc1;
                wmma::fill_fragment(fc1, 0.0f);
#pragma unroll
                for (int k = 0; k < 128; k += 16) {
                    wmma::load_matrix_sync(fb, sKV + (ct * 16) * 136 + k, 136);
                    wmma::load_matrix_sync(fa0, sQ + 16 * 136 + k, 136);
                    wmma::mma_sync(fc1, fa0, fb, fc1);
                }
                wmma::store_matrix_sync(sS + 16 * 64 + ct * 16, fc1, 64, wmma::mem_row_major);
            }
        }
        __syncthreads();

        __half p_reg0[6], p_reg1[6];
        {
            const int j = jb + kv_r;
            const bool valid = is_bulk ? true : (j < j_end);
            __half* dst = sKV + kv_r * 136 + kv_c0;
            if (valid) {
                const int slot = is_bulk ? j : (is_swa ? (pow2_cap ? (j & cap_mask) : (j % cap)) : j);
                const int8_t* vp = vq + (size_t) slot * kv_dim + hkv * 128 + kv_c0;
#pragma unroll
                for (int i = 0; i < 2; ++i) {
                    const uint4 raw = *reinterpret_cast<const uint4*>(vp + i * 16);
                    __half2 h2[8];
                    i8x4_to_h2x2(raw.x, h2[0], h2[1]);
                    i8x4_to_h2x2(raw.y, h2[2], h2[3]);
                    i8x4_to_h2x2(raw.z, h2[4], h2[5]);
                    i8x4_to_h2x2(raw.w, h2[6], h2[7]);
                    *reinterpret_cast<uint4*>(dst + i * 16) = *reinterpret_cast<const uint4*>(h2);
                    *reinterpret_cast<uint4*>(dst + i * 16 + 8) = *reinterpret_cast<const uint4*>(h2 + 4);
                }
            } else {
                const uint4 z = make_uint4(0, 0, 0, 0);
#pragma unroll
                for (int i = 0; i < 4; ++i) *reinterpret_cast<uint4*>(dst + i * 8) = z;
            }

            const int col0 = lane;
            const int col1 = lane + 32;
            const int j_0 = jb + col0;
            const int j_1 = jb + col1;
            const float dk0 = s_dk[col0], dk1 = s_dk[col1];
            const float dv0 = s_dv[col0], dv1 = s_dv[col1];

#pragma unroll
            for (int rr = 0; rr < 6; ++rr) {
                const int qr = warp * 6 + rr;
                const int g = qr >> 4;
                const int r = qr & 15;
                const int t = t0 + r;
                const int h = h0 + g;
                const bool t_ok = t < T;
                const float nsink = t_ok ? (-sink[(size_t) t * n_head + h] * kLog2e) : 0.0f;
                const float qks2 = s_qscale[qr];

                float sc0, sc1;
                bool v0, v1;
                if (is_bulk) {
                    v0 = t_ok;
                    v1 = t_ok;
                    sc0 = t_ok ? (sS[qr * 64 + col0] * qks2 * dk0 + nsink) : -1e30f;
                    sc1 = t_ok ? (sS[qr * 64 + col1] * qks2 * dk1 + nsink) : -1e30f;
                } else {
                    const int pos_q = p0 + t;
                    const float* rt = t_ok ? (rel_tab + ((size_t) t * n_head + h) * n_ext) : nullptr;
                    const int d0 = pos_q - j_0;
                    const int d1 = pos_q - j_1;
                    v0 = t_ok && (j_0 <= pos_q) && (!is_swa || d0 < n_swa);
                    v1 = t_ok && (j_1 <= pos_q) && (!is_swa || d1 < n_swa);
                    sc0 = v0 ? (sS[qr * 64 + col0] * qks2 * dk0 + (d0 < n_ext ? (rt[d0] * kLog2e) : nsink)) : -1e30f;
                    sc1 = v1 ? (sS[qr * 64 + col1] * qks2 * dk1 + (d1 < n_ext ? (rt[d1] * kLog2e) : nsink)) : -1e30f;
                }

                float rmax = fmaxf(sc0, sc1);
#pragma unroll
                for (int o = 16; o > 0; o >>= 1) rmax = fmaxf(rmax, __shfl_xor_sync(0xffffffffu, rmax, o));
                const float m_old = s_m[qr], l_old = s_l[qr];
                const float m_new = fmaxf(m_old, rmax);
                const float alpha = l_old > 0.0f ? exp2f(m_old - m_new) : 0.0f;
                const float p_0 = v0 ? exp2f(sc0 - m_new) : 0.0f;
                const float p_1 = v1 ? exp2f(sc1 - m_new) : 0.0f;
                float psum = p_0 + p_1;
#pragma unroll
                for (int o = 16; o > 0; o >>= 1) psum += __shfl_xor_sync(0xffffffffu, psum, o);

                const float pv_0 = p_0 * dv0, pv_1 = p_1 * dv1;
                float pvmax = fmaxf(pv_0, pv_1);
#pragma unroll
                for (int o = 16; o > 0; o >>= 1) pvmax = fmaxf(pvmax, __shfl_xor_sync(0xffffffffu, pvmax, o));
                const float inv_pv = pvmax > 0.0f ? __fdividef(1.0f, pvmax) : 0.0f;

                p_reg0[rr] = __float2half(pv_0 * inv_pv);
                p_reg1[rr] = __float2half(pv_1 * inv_pv);
                if (lane == 0) {
                    s_m[qr] = m_new;
                    s_l[qr] = l_old * alpha + psum;
                    s_alpha[qr] = alpha;
                    s_pvmax[qr] = pvmax;
                }
            }
        }
        __syncthreads();

        {
            const int col0 = lane;
            const int col1 = lane + 32;
#pragma unroll
            for (int rr = 0; rr < 6; ++rr) {
                const int qr = warp * 6 + rr;
                sP[qr * 72 + col0] = p_reg0[rr];
                sP[qr * 72 + col1] = p_reg1[rr];
            }
        }
        __syncthreads();

        {
            const int cbase = warp * 16;
            wmma::fragment<wmma::accumulator, 16, 16, 16, float> fc0, fc1, fc2;
            wmma::fill_fragment(fc0, 0.0f);
            wmma::fill_fragment(fc1, 0.0f);
            wmma::fill_fragment(fc2, 0.0f);
#pragma unroll
            for (int k = 0; k < 64; k += 16) {
                wmma::fragment<wmma::matrix_a, 16, 16, 16, __half, wmma::row_major> fa0, fa1, fa2;
                wmma::fragment<wmma::matrix_b, 16, 16, 16, __half, wmma::row_major> fb;
                wmma::load_matrix_sync(fb, sKV + k * 136 + cbase, 136);
                wmma::load_matrix_sync(fa0, sP + k, 72);
                wmma::load_matrix_sync(fa1, sP + 16 * 72 + k, 72);
                wmma::load_matrix_sync(fa2, sP + 32 * 72 + k, 72);
                wmma::mma_sync(fc0, fa0, fb, fc0);
                wmma::mma_sync(fc1, fa1, fb, fc1);
                wmma::mma_sync(fc2, fa2, fb, fc2);
            }
            __syncthreads();
            wmma::store_matrix_sync(s_tile + cbase, fc0, 132, wmma::mem_row_major);
            wmma::store_matrix_sync(s_tile + 16 * 132 + cbase, fc1, 132, wmma::mem_row_major);
            wmma::store_matrix_sync(s_tile + 32 * 132 + cbase, fc2, 132, wmma::mem_row_major);
        }
        __syncthreads();

        const int row0 = my_half * 8;
#pragma unroll
        for (int g = 0; g < 3; ++g) {
#pragma unroll
            for (int r = 0; r < 8; ++r) {
                const int qr = g * 16 + row0 + r;
                acc[g][r] = acc[g][r] * s_alpha[qr] + s_pvmax[qr] * s_tile[qr * 132 + my_col];
            }
        }
        __syncthreads();
    };

    int jb = j_start;
    for (; jb + 64 <= bulk_end; jb += 64) step_tile(jb, true);
    for (; jb < j_end; jb += 64) step_tile(jb, false);

    const int row0 = my_half * 8;
#pragma unroll
    for (int g = 0; g < 3; ++g) {
        const int h = h0 + g;
#pragma unroll
        for (int r = 0; r < 8; ++r) {
            const int t = t0 + row0 + r;
            if (t < T) {
                const size_t oi = ((size_t) t * n_head + h) * 128 + my_col;
                float val = __fdividef(acc[g][r], s_l[g * 16 + row0 + r]);
                if (gate) val *= sigmf(gate[oi]);
                out[oi] = val;
            }
        }
    }
#endif
}

#if !defined(__CUDA_ARCH__) || __CUDA_ARCH__ >= 700
__device__ __forceinline__ float fast_exp2f_hw(float x) {
    float y;
    asm("ex2.approx.ftz.f32 %0, %1;" : "=f"(y) : "f"(x));
    return y;
}

__device__ __forceinline__ void i8x4_to_h2x2_fma(uint32_t packed_i8x4, __half2 sc2, __half2 neg_bias_sc2,
                                                 __half2& out0, __half2& out1) {
    const uint32_t u = packed_i8x4 ^ 0x80808080u;
    const uint32_t lo = __byte_perm(u, 0x6400u, 0x5150u);
    const uint32_t hi = __byte_perm(u, 0x6400u, 0x5352u);
    out0 = __hfma2(*reinterpret_cast<const __half2*>(&lo), sc2, neg_bias_sc2);
    out1 = __hfma2(*reinterpret_cast<const __half2*>(&hi), sc2, neg_bias_sc2);
}

__device__ __forceinline__ void i8x4_to_h2x2_scaled(uint32_t packed_i8x4, __half2 scale2, __half2& out0, __half2& out1) {
    const __half2 neg1152 = __half2half2(__float2half(-1152.0f));
    i8x4_to_h2x2_fma(packed_i8x4, scale2, __hmul2(neg1152, scale2), out0, out1);
}

__device__ __forceinline__ void wmma_acc_layout_16x16(int lane, int& r0, int& r1, int& c0, int& c1) {
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ < 750
    r0 = ((lane & 16) >> 2) + ((lane & 4) << 1) + (lane & 1);
    r1 = r0 + 2;
    c0 = (lane & 8) + (lane & 2);
    c1 = c0 + 4;
#else
    r0 = lane >> 2;
    r1 = r0 + 8;
    c0 = (lane & 3) * 2;
    c1 = c0 + 8;
#endif
}

__device__ __forceinline__ float wmma_row_max_16(float v) {
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ < 750
    v = fmaxf(v, __shfl_xor_sync(0xffffffffu, v, 2));
    v = fmaxf(v, __shfl_xor_sync(0xffffffffu, v, 8));
#else
    v = fmaxf(v, __shfl_xor_sync(0xffffffffu, v, 1));
    v = fmaxf(v, __shfl_xor_sync(0xffffffffu, v, 2));
#endif
    return v;
}

__device__ __forceinline__ float wmma_row_sum_16(float v) {
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ < 750
    v += __shfl_xor_sync(0xffffffffu, v, 2);
    v += __shfl_xor_sync(0xffffffffu, v, 8);
#else
    v += __shfl_xor_sync(0xffffffffu, v, 1);
    v += __shfl_xor_sync(0xffffffffu, v, 2);
#endif
    return v;
}

__device__ __forceinline__ bool wmma_row_leader_16(int lane) {
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ < 750
    return (lane & 10) == 0;
#else
    return (lane & 3) == 0;
#endif
}

__device__ __forceinline__ void wmma_rescale_acc(nvcuda::wmma::fragment<nvcuda::wmma::accumulator, 16, 16, 16, float>& fc,
                                                 float a0, float a1) {
    fc.x[0] *= a0; fc.x[1] *= a0; fc.x[4] *= a0; fc.x[5] *= a0;
    fc.x[2] *= a1; fc.x[3] *= a1; fc.x[6] *= a1; fc.x[7] *= a1;
}

__device__ __forceinline__ uint32_t f2_to_h2u(float a, float b) {
    const __half2 h = __float22half2_rn(make_float2(a, b));
    uint32_t u;
    memcpy(&u, &h, 4);
    return u;
}

/// Exact in-register converter: 16x16 FP32 WMMA accumulator (after ex2.approx) -> 16x16 FP16 row_major matrix_a!
/// Zero sP SMEM: 6 warp shuffles on Volta sm_70 (verified 0.00000000 max_err), 0 shuffles on Turing/Ampere/Ada sm_75+.
template <bool MASKED>
__device__ __forceinline__ void wmma_acc_to_matA_16x16(
        const nvcuda::wmma::fragment<nvcuda::wmma::accumulator, 16, 16, 16, float>& fc,
        float m_new0, float m_new1, float& ps0, float& ps1,
        nvcuda::wmma::fragment<nvcuda::wmma::matrix_a, 16, 16, 16, __half, nvcuda::wmma::row_major>& fa, int lane) {
    const float p00 = (!MASKED || fc.x[0] > -1e20f) ? fast_exp2f_hw(fc.x[0] - m_new0) : 0.0f;
    const float p01 = (!MASKED || fc.x[1] > -1e20f) ? fast_exp2f_hw(fc.x[1] - m_new0) : 0.0f;
    const float p04 = (!MASKED || fc.x[4] > -1e20f) ? fast_exp2f_hw(fc.x[4] - m_new0) : 0.0f;
    const float p05 = (!MASKED || fc.x[5] > -1e20f) ? fast_exp2f_hw(fc.x[5] - m_new0) : 0.0f;
    ps0 += (p00 + p01) + (p04 + p05);
    const float p12 = (!MASKED || fc.x[2] > -1e20f) ? fast_exp2f_hw(fc.x[2] - m_new1) : 0.0f;
    const float p13 = (!MASKED || fc.x[3] > -1e20f) ? fast_exp2f_hw(fc.x[3] - m_new1) : 0.0f;
    const float p16 = (!MASKED || fc.x[6] > -1e20f) ? fast_exp2f_hw(fc.x[6] - m_new1) : 0.0f;
    const float p17 = (!MASKED || fc.x[7] > -1e20f) ? fast_exp2f_hw(fc.x[7] - m_new1) : 0.0f;
    ps1 += (p12 + p13) + (p16 + p17);

    const uint32_t a0 = f2_to_h2u(p00, p01), a1 = f2_to_h2u(p04, p05);
    const uint32_t b0 = f2_to_h2u(p12, p13), b1 = f2_to_h2u(p16, p17);
    uint32_t* fa_u = reinterpret_cast<uint32_t*>(fa.x);
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ < 750
    const bool want_r1 = (lane & 2) != 0;
    const uint32_t recv0 = __shfl_xor_sync(0xffffffffu, want_r1 ? a0 : b0, 2);
    const uint32_t recv1 = __shfl_xor_sync(0xffffffffu, want_r1 ? a1 : b1, 2);
    const uint32_t h0 = want_r1 ? recv0 : a0, h1 = want_r1 ? b0 : recv0;
    const uint32_t h2 = want_r1 ? recv1 : a1, h3 = want_r1 ? b1 : recv1;
    const uint32_t o0 = __shfl_xor_sync(0xffffffffu, h0, 8), o1 = __shfl_xor_sync(0xffffffffu, h1, 8);
    const uint32_t o2 = __shfl_xor_sync(0xffffffffu, h2, 8), o3 = __shfl_xor_sync(0xffffffffu, h3, 8);
    const bool hi8 = (lane & 8) != 0;
    fa_u[0] = hi8 ? o0 : h0; fa_u[1] = hi8 ? o1 : h1;
    fa_u[2] = hi8 ? o2 : h2; fa_u[3] = hi8 ? o3 : h3;
    fa_u[4] = hi8 ? h0 : o0; fa_u[5] = hi8 ? h1 : o1;
    fa_u[6] = hi8 ? h2 : o2; fa_u[7] = hi8 ? h3 : o3;
#else
    (void) lane;
    fa_u[0] = fa_u[4] = a0;
    fa_u[1] = fa_u[5] = b0;
    fa_u[2] = fa_u[6] = a1;
    fa_u[3] = fa_u[7] = b1;
#endif
}
#endif

/// FlashAttention-2 192-row (64 tokens x 3 GQA heads, 256 threads = 8 warps, 1 blk/SM, 97,280 B SMEM) FP16 WMMA prefill:
/// 8-warp (4 row-groups x 2 col-groups, exact 2 warps/SMSP, 0 register spills): 3x2 Q*K^T -> bar.sync 64-thr half-row max
/// merge -> sP -> 3x4 register-resident P*V (0.583 SMEM loads/WMMA), HFMA2 KV dequant, shifted-nsink bulk loop, lazy acc rescale.
template <bool SPLIT_KV, bool IS_SWA>
__global__ void __launch_bounds__(256, 1) k_attn_q8_gqa3_pp_fa2_192(
        const float* __restrict__ Q,
        const int8_t* __restrict__ kq, const __half* __restrict__ ks,
        const int8_t* __restrict__ vq, const __half* __restrict__ vs,
        const float* __restrict__ rel_tab, const float* __restrict__ sink,
        const float* __restrict__ gate, float* __restrict__ out,
        float* __restrict__ split_out,
        int n_head, int n_head_kv, int n_ext, int cap, int n_swa,
        const int32_t* __restrict__ pos0, float kq_scale, int T, int split_n) {
#if !defined(__CUDA_ARCH__) || __CUDA_ARCH__ >= 700
    using namespace nvcuda;
    extern __shared__ __align__(16) char smem_raw[];
    // 97,280 B <= 98,304 B (96 KB on sm_70, 100 KB on sm_89):
    // sQ: 192*136*2 = 52,224 B | sKV: 64*136*2 = 17,408 B | sP: 192*72*2 = 27,648 B
    // Scalar metadata packed into unused 16 B tail of each sP row (sP[qr*72 + 64..71]):
    //   +64: half-max cg=0 (and final rm in epilogue), +66: half-max cg=1, +68: final rl0, +70: final rl1.
    __half* sQ = reinterpret_cast<__half*>(smem_raw);
    __half* sKV = sQ + 192 * 136;
    __half* sP = sKV + 64 * 136;
    auto s_hmax_ptr = [&](int qr, int c) -> float* { return reinterpret_cast<float*>(sP + qr * 72 + 64 + (c << 1)); };
    auto s_m_ptr    = [&](int qr) -> float* { return reinterpret_cast<float*>(sP + qr * 72 + 64); };
    auto s_l0_ptr   = [&](int qr) -> float* { return reinterpret_cast<float*>(sP + qr * 72 + 68); };
    auto s_l1_ptr   = [&](int qr) -> float* { return reinterpret_cast<float*>(sP + qr * 72 + 70); };

    const int hkv = blockIdx.x, h0 = 3 * hkv, t0 = blockIdx.y * 64;
    const int sp = SPLIT_KV ? (int) blockIdx.z : 0;
    const int tid = threadIdx.x, warp = tid >> 5, lane = tid & 31;
    const int rg = warp >> 1;      // 0..3 (48 rows each)
    const int cg = warp & 1;       // 0..1 (QK^T: cols cg*32..+31; P*V: output cols cg*64..+63)
    const int kv_dim = n_head_kv * 128;
    constexpr float kLog2e = 1.4426950408889634f;
    const float qscale_all = kq_scale * kLog2e;

    const int p0 = pos0[0];
    const int t_last = min(t0 + 63, T - 1);
    const int j_start_all = IS_SWA ? max(0, (p0 + t0) - n_swa + 1) : 0;
    const int j_end_all = p0 + t_last + 1;

    int j_start = j_start_all;
    int j_end = j_end_all;
    if constexpr (SPLIT_KV) {
        const int n_tiles = (j_end_all - j_start_all + 63) / 64;
        const int t_per_sp = (n_tiles + split_n - 1) / split_n;
        j_start = j_start_all + sp * t_per_sp * 64;
        j_end = min(j_end_all, j_start + t_per_sp * 64);
        if (j_start >= j_end_all) {
            if (tid < 192) {
                const int g = tid >> 6, r = tid & 63, t = t0 + r;
                if (t < T) {
                    float* dst = split_out + (((size_t) t * split_n + sp) * n_head + (h0 + g)) * 130;
                    dst[128] = -1e30f;
                    dst[129] = 0.0f;
                }
            }
            return;
        }
    }

    // 256 threads load 192x128 Q -> sQ in 6 iterations (32 rows/iter, 8 threads/row, 16 elems/thread).
    for (int base_r = 0; base_r < 192; base_r += 32) {
        const int qr = base_r + (tid >> 3);
        const int qc0 = (tid & 7) * 16;
        const int g = qr >> 6, r = qr & 63, t = t0 + r, h = h0 + g;
        __half2* dst2 = reinterpret_cast<__half2*>(sQ + qr * 136 + qc0);
        if (t < T) {
            const float* qp = Q + ((size_t) t * n_head + h) * 128 + qc0;
#pragma unroll
            for (int i = 0; i < 4; ++i) {
                const float4 v = *reinterpret_cast<const float4*>(qp + i * 4);
                dst2[i * 2 + 0] = __halves2half2(__float2half(v.x * qscale_all), __float2half(v.y * qscale_all));
                dst2[i * 2 + 1] = __halves2half2(__float2half(v.z * qscale_all), __float2half(v.w * qscale_all));
            }
        } else {
            const __half2 z2 = __halves2half2(__float2half(0.0f), __float2half(0.0f));
#pragma unroll
            for (int i = 0; i < 8; ++i) dst2[i] = z2;
        }
    }
    __syncthreads();

    const int bulk_end = IS_SWA ? j_start : max(j_start, min(j_end, p0 + t0 - n_ext + 1));
    const bool pow2_cap = IS_SWA ? ((cap & (cap - 1)) == 0) : false;
    const int cap_mask = cap - 1;

    int acc_r0, acc_r1, acc_c0, acc_c1;
    wmma_acc_layout_16x16(lane, acc_r0, acc_r1, acc_c0, acc_c1);
    const bool is_leader = wmma_row_leader_16(lane);

    const int qr_base = rg * 48;
    const int kv_cbase = cg * 32;
    const int pv_pc0 = cg * 64;
    const bool has_sink = (!SPLIT_KV || sp == 0);
    const float init_m = has_sink ? 0.0f : -1e30f;
    const float init_l = (has_sink && cg == 0) ? 1.0f : 0.0f;

    float rm0[3], rm1[3], rl0[3], rl1[3];
#pragma unroll
    for (int m = 0; m < 3; ++m) {
        rm0[m] = init_m; rl0[m] = init_l;
        rm1[m] = init_m; rl1[m] = init_l;
    }

    auto load_nsink = [&](int qr) -> float {
        const int tr = t0 + (qr & 63), hr = h0 + (qr >> 6);
        return (tr < T) ? (-__ldg(&sink[(size_t) tr * n_head + hr]) * kLog2e) : 0.0f;
    };

    wmma::fragment<wmma::accumulator, 16, 16, 16, float> acc[3][4];
#pragma unroll
    for (int m = 0; m < 3; ++m) {
#pragma unroll
        for (int n = 0; n < 4; ++n) wmma::fill_fragment(acc[m][n], 0.0f);
    }

    // All 256 threads participate branchlessly in 64x128 KV loading (4 threads/row x 32 elems/thread).
    const int kv_r = tid >> 2, kv_c0 = (tid & 3) * 32;
    const __half2 neg1152 = __half2half2(__float2half(-1152.0f));

    // Fast specialized bulk loop (!IS_SWA, dist >= n_ext): shifted-nsink (0 bulk adds), HFMA2 KV prefetch, lazy acc rescale.
    int jb = j_start;
    if constexpr (!IS_SWA) {
    const int bulk_tiles_end = bulk_end - 63;
    if (jb < bulk_tiles_end) {
#pragma unroll
        for (int m = 0; m < 3; ++m) {
            if (rm0[m] > -1e20f) rm0[m] -= load_nsink(qr_base + m * 16 + acc_r0);
            if (rm1[m] > -1e20f) rm1[m] -= load_nsink(qr_base + m * 16 + acc_r1);
        }
        uint4 kraw0, kraw1;
        __half2 dk2;
        {
            const int j = jb + kv_r;
            dk2 = __half2half2(__ldg(&ks[(size_t) j * n_head_kv + hkv]));
            const int8_t* kp = kq + (size_t) j * kv_dim + hkv * 128 + kv_c0;
            kraw0 = *reinterpret_cast<const uint4*>(kp);
            kraw1 = *reinterpret_cast<const uint4*>(kp + 16);
        }
        for (; jb < bulk_tiles_end; jb += 64) {
            {
                __half* dst = sKV + kv_r * 136 + kv_c0;
                const __half2 kbias = __hmul2(neg1152, dk2);
                __half2 h2[8];
                i8x4_to_h2x2_fma(kraw0.x, dk2, kbias, h2[0], h2[1]); i8x4_to_h2x2_fma(kraw0.y, dk2, kbias, h2[2], h2[3]);
                i8x4_to_h2x2_fma(kraw0.z, dk2, kbias, h2[4], h2[5]); i8x4_to_h2x2_fma(kraw0.w, dk2, kbias, h2[6], h2[7]);
                *reinterpret_cast<uint4*>(dst) = *reinterpret_cast<const uint4*>(h2);
                *reinterpret_cast<uint4*>(dst + 8) = *reinterpret_cast<const uint4*>(h2 + 4);
                i8x4_to_h2x2_fma(kraw1.x, dk2, kbias, h2[0], h2[1]); i8x4_to_h2x2_fma(kraw1.y, dk2, kbias, h2[2], h2[3]);
                i8x4_to_h2x2_fma(kraw1.z, dk2, kbias, h2[4], h2[5]); i8x4_to_h2x2_fma(kraw1.w, dk2, kbias, h2[6], h2[7]);
                *reinterpret_cast<uint4*>(dst + 16) = *reinterpret_cast<const uint4*>(h2);
                *reinterpret_cast<uint4*>(dst + 24) = *reinterpret_cast<const uint4*>(h2 + 4);
            }
            __syncthreads();

            uint4 vraw0, vraw1;
            __half2 dv2;
            {
                const int j = jb + kv_r;
                dv2 = __half2half2(__ldg(&vs[(size_t) j * n_head_kv + hkv]));
                const int8_t* vp = vq + (size_t) j * kv_dim + hkv * 128 + kv_c0;
                vraw0 = *reinterpret_cast<const uint4*>(vp);
                vraw1 = *reinterpret_cast<const uint4*>(vp + 16);
            }

            wmma::fragment<wmma::accumulator, 16, 16, 16, float> fc[3][2];
#pragma unroll
            for (int m = 0; m < 3; ++m) {
                wmma::fill_fragment(fc[m][0], 0.0f);
                wmma::fill_fragment(fc[m][1], 0.0f);
            }
#pragma unroll
            for (int k = 0; k < 128; k += 16) {
                wmma::fragment<wmma::matrix_b, 16, 16, 16, __half, wmma::col_major> fb0, fb1;
                wmma::load_matrix_sync(fb0, sKV + (kv_cbase + 0) * 136 + k, 136);
                wmma::load_matrix_sync(fb1, sKV + (kv_cbase + 16) * 136 + k, 136);
#pragma unroll
                for (int m = 0; m < 3; ++m) {
                    wmma::fragment<wmma::matrix_a, 16, 16, 16, __half, wmma::row_major> fa;
                    wmma::load_matrix_sync(fa, sQ + (qr_base + m * 16) * 136 + k, 136);
                    wmma::mma_sync(fc[m][0], fa, fb0, fc[m][0]);
                    wmma::mma_sync(fc[m][1], fa, fb1, fc[m][1]);
                }
            }

#pragma unroll
            for (int m = 0; m < 3; ++m) {
                const float mx0 = wmma_row_max_16(
                    fmaxf(fmaxf(fmaxf(fc[m][0].x[0], fc[m][0].x[1]), fmaxf(fc[m][0].x[4], fc[m][0].x[5])),
                          fmaxf(fmaxf(fc[m][1].x[0], fc[m][1].x[1]), fmaxf(fc[m][1].x[4], fc[m][1].x[5]))));
                const float mx1 = wmma_row_max_16(
                    fmaxf(fmaxf(fmaxf(fc[m][0].x[2], fc[m][0].x[3]), fmaxf(fc[m][0].x[6], fc[m][0].x[7])),
                          fmaxf(fmaxf(fc[m][1].x[2], fc[m][1].x[3]), fmaxf(fc[m][1].x[6], fc[m][1].x[7]))));
                if (is_leader) {
                    *s_hmax_ptr(qr_base + m * 16 + acc_r0, cg) = mx0;
                    *s_hmax_ptr(qr_base + m * 16 + acc_r1, cg) = mx1;
                }
            }
            asm volatile("bar.sync %0, 64;" :: "r"(rg + 1) : "memory");

#pragma unroll
            for (int m = 0; m < 3; ++m) {
                const int qr0 = qr_base + m * 16 + acc_r0;
                const int qr1 = qr_base + m * 16 + acc_r1;
                const float mx0 = fmaxf(*s_hmax_ptr(qr0, 0), *s_hmax_ptr(qr0, 1));
                const float mx1 = fmaxf(*s_hmax_ptr(qr1, 0), *s_hmax_ptr(qr1, 1));
                const float m_new0 = fmaxf(rm0[m], mx0);
                const float m_new1 = fmaxf(rm1[m], mx1);
                const float a0 = (rm0[m] > -1e20f) ? ((m_new0 == rm0[m]) ? 1.0f : fast_exp2f_hw(rm0[m] - m_new0)) : 0.0f;
                const float a1 = (rm1[m] > -1e20f) ? ((m_new1 == rm1[m]) ? 1.0f : fast_exp2f_hw(rm1[m] - m_new1)) : 0.0f;
                if (__any_sync(0xffffffffu, a0 != 1.0f || a1 != 1.0f)) {
#pragma unroll
                    for (int n = 0; n < 4; ++n) wmma_rescale_acc(acc[m][n], a0, a1);
                }
                float psum0 = 0.0f, psum1 = 0.0f;
#pragma unroll
                for (int ct = 0; ct < 2; ++ct) {
                    const float p00 = fast_exp2f_hw(fc[m][ct].x[0] - m_new0), p01 = fast_exp2f_hw(fc[m][ct].x[1] - m_new0);
                    const float p04 = fast_exp2f_hw(fc[m][ct].x[4] - m_new0), p05 = fast_exp2f_hw(fc[m][ct].x[5] - m_new0);
                    psum0 += (p00 + p01) + (p04 + p05);
                    *reinterpret_cast<__half2*>(sP + qr0 * 72 + kv_cbase + ct * 16 + acc_c0) = __halves2half2(__float2half(p00), __float2half(p01));
                    *reinterpret_cast<__half2*>(sP + qr0 * 72 + kv_cbase + ct * 16 + acc_c1) = __halves2half2(__float2half(p04), __float2half(p05));

                    const float p12 = fast_exp2f_hw(fc[m][ct].x[2] - m_new1), p13 = fast_exp2f_hw(fc[m][ct].x[3] - m_new1);
                    const float p16 = fast_exp2f_hw(fc[m][ct].x[6] - m_new1), p17 = fast_exp2f_hw(fc[m][ct].x[7] - m_new1);
                    psum1 += (p12 + p13) + (p16 + p17);
                    *reinterpret_cast<__half2*>(sP + qr1 * 72 + kv_cbase + ct * 16 + acc_c0) = __halves2half2(__float2half(p12), __float2half(p13));
                    *reinterpret_cast<__half2*>(sP + qr1 * 72 + kv_cbase + ct * 16 + acc_c1) = __halves2half2(__float2half(p16), __float2half(p17));
                }
                psum0 = wmma_row_sum_16(psum0);
                psum1 = wmma_row_sum_16(psum1);
                rm0[m] = m_new0; rl0[m] = rl0[m] * a0 + psum0;
                rm1[m] = m_new1; rl1[m] = rl1[m] * a1 + psum1;
            }
            __syncthreads();

            {
                __half* dst = sKV + kv_r * 136 + kv_c0;
                const __half2 vbias = __hmul2(neg1152, dv2);
                __half2 h2[8];
                i8x4_to_h2x2_fma(vraw0.x, dv2, vbias, h2[0], h2[1]); i8x4_to_h2x2_fma(vraw0.y, dv2, vbias, h2[2], h2[3]);
                i8x4_to_h2x2_fma(vraw0.z, dv2, vbias, h2[4], h2[5]); i8x4_to_h2x2_fma(vraw0.w, dv2, vbias, h2[6], h2[7]);
                *reinterpret_cast<uint4*>(dst) = *reinterpret_cast<const uint4*>(h2);
                *reinterpret_cast<uint4*>(dst + 8) = *reinterpret_cast<const uint4*>(h2 + 4);
                i8x4_to_h2x2_fma(vraw1.x, dv2, vbias, h2[0], h2[1]); i8x4_to_h2x2_fma(vraw1.y, dv2, vbias, h2[2], h2[3]);
                i8x4_to_h2x2_fma(vraw1.z, dv2, vbias, h2[4], h2[5]); i8x4_to_h2x2_fma(vraw1.w, dv2, vbias, h2[6], h2[7]);
                *reinterpret_cast<uint4*>(dst + 16) = *reinterpret_cast<const uint4*>(h2);
                *reinterpret_cast<uint4*>(dst + 24) = *reinterpret_cast<const uint4*>(h2 + 4);
            }
            __syncthreads();

            if (jb + 64 < bulk_tiles_end) {
                const int j = jb + 64 + kv_r;
                dk2 = __half2half2(__ldg(&ks[(size_t) j * n_head_kv + hkv]));
                const int8_t* kp = kq + (size_t) j * kv_dim + hkv * 128 + kv_c0;
                kraw0 = *reinterpret_cast<const uint4*>(kp);
                kraw1 = *reinterpret_cast<const uint4*>(kp + 16);
            }
#pragma unroll
            for (int k = 0; k < 64; k += 16) {
                wmma::fragment<wmma::matrix_a, 16, 16, 16, __half, wmma::row_major> fa[3];
#pragma unroll
                for (int m = 0; m < 3; ++m) {
                    wmma::load_matrix_sync(fa[m], sP + (qr_base + m * 16) * 72 + k, 72);
                }
#pragma unroll
                for (int n = 0; n < 4; ++n) {
                    wmma::fragment<wmma::matrix_b, 16, 16, 16, __half, wmma::row_major> fb;
                    wmma::load_matrix_sync(fb, sKV + k * 136 + pv_pc0 + n * 16, 136);
#pragma unroll
                    for (int m = 0; m < 3; ++m) {
                        wmma::mma_sync(acc[m][n], fa[m], fb, acc[m][n]);
                    }
                }
            }
            __syncthreads();
        }
#pragma unroll
        for (int m = 0; m < 3; ++m) {
            if (rm0[m] > -1e20f) rm0[m] += load_nsink(qr_base + m * 16 + acc_r0);
            if (rm1[m] > -1e20f) rm1[m] += load_nsink(qr_base + m * 16 + acc_r1);
        }
    }
    }

    // Tail / SWA loop (causal masking + relative distance bias; #pragma unroll 1 keeps bulk loop at 0 register spills).
    if (jb < j_end) {
        auto mask_bias = [&](float s, int qr, int j) -> float {
            const int tr = t0 + (qr & 63);
            const int pos_q = p0 + tr;
            const int d = pos_q - j;
            const bool v = (tr < T) && (j <= pos_q) && (!IS_SWA || d < n_swa);
            if (!v) return -1e30f;
            const int hr = h0 + (qr >> 6);
            const float b = (d < n_ext)
                ? (__ldg(&rel_tab[((size_t) tr * n_head + hr) * n_ext + d]) * kLog2e)
                : (-__ldg(&sink[(size_t) tr * n_head + hr]) * kLog2e);
            return s + b;
        };
        for (; jb < j_end; jb += 64) {
            {
                const int j = jb + kv_r;
                const bool valid = j < j_end;
                __half* dst = sKV + kv_r * 136 + kv_c0;
                if (valid) {
                    const int slot = IS_SWA ? (pow2_cap ? (j & cap_mask) : (j % cap)) : j;
                    const __half2 dk2_t = __half2half2(__ldg(&ks[(size_t) slot * n_head_kv + hkv]));
                    const __half2 kbias = __hmul2(neg1152, dk2_t);
                    const int8_t* kp = kq + (size_t) slot * kv_dim + hkv * 128 + kv_c0;
#pragma unroll 1
                    for (int i = 0; i < 2; ++i) {
                        const uint4 raw = *reinterpret_cast<const uint4*>(kp + i * 16);
                        __half2 h2[8];
                        i8x4_to_h2x2_fma(raw.x, dk2_t, kbias, h2[0], h2[1]); i8x4_to_h2x2_fma(raw.y, dk2_t, kbias, h2[2], h2[3]);
                        i8x4_to_h2x2_fma(raw.z, dk2_t, kbias, h2[4], h2[5]); i8x4_to_h2x2_fma(raw.w, dk2_t, kbias, h2[6], h2[7]);
                        *reinterpret_cast<uint4*>(dst + i * 16) = *reinterpret_cast<const uint4*>(h2);
                        *reinterpret_cast<uint4*>(dst + i * 16 + 8) = *reinterpret_cast<const uint4*>(h2 + 4);
                    }
                } else {
                    const uint4 z = make_uint4(0, 0, 0, 0);
#pragma unroll
                    for (int i = 0; i < 4; ++i) *reinterpret_cast<uint4*>(dst + i * 8) = z;
                }
            }
            __syncthreads();

            wmma::fragment<wmma::accumulator, 16, 16, 16, float> fc[3][2];
#pragma unroll
            for (int m = 0; m < 3; ++m) {
                wmma::fill_fragment(fc[m][0], 0.0f);
                wmma::fill_fragment(fc[m][1], 0.0f);
            }
#pragma unroll 1
            for (int k = 0; k < 128; k += 16) {
                wmma::fragment<wmma::matrix_b, 16, 16, 16, __half, wmma::col_major> fb0, fb1;
                wmma::load_matrix_sync(fb0, sKV + (kv_cbase + 0) * 136 + k, 136);
                wmma::load_matrix_sync(fb1, sKV + (kv_cbase + 16) * 136 + k, 136);
#pragma unroll
                for (int m = 0; m < 3; ++m) {
                    wmma::fragment<wmma::matrix_a, 16, 16, 16, __half, wmma::row_major> fa;
                    wmma::load_matrix_sync(fa, sQ + (qr_base + m * 16) * 136 + k, 136);
                    wmma::mma_sync(fc[m][0], fa, fb0, fc[m][0]);
                    wmma::mma_sync(fc[m][1], fa, fb1, fc[m][1]);
                }
            }

#pragma unroll
            for (int m = 0; m < 3; ++m) {
                const int qr0 = qr_base + m * 16 + acc_r0;
                const int qr1 = qr_base + m * 16 + acc_r1;
                float mx0 = -1e30f, mx1 = -1e30f;
#pragma unroll
                for (int ct = 0; ct < 2; ++ct) {
                    const int j0 = jb + kv_cbase + ct * 16 + acc_c0;
                    const int j1 = jb + kv_cbase + ct * 16 + acc_c1;
                    fc[m][ct].x[0] = mask_bias(fc[m][ct].x[0], qr0, j0);
                    fc[m][ct].x[1] = mask_bias(fc[m][ct].x[1], qr0, j0 + 1);
                    fc[m][ct].x[4] = mask_bias(fc[m][ct].x[4], qr0, j1);
                    fc[m][ct].x[5] = mask_bias(fc[m][ct].x[5], qr0, j1 + 1);
                    mx0 = fmaxf(mx0, fmaxf(fmaxf(fc[m][ct].x[0], fc[m][ct].x[1]), fmaxf(fc[m][ct].x[4], fc[m][ct].x[5])));

                    fc[m][ct].x[2] = mask_bias(fc[m][ct].x[2], qr1, j0);
                    fc[m][ct].x[3] = mask_bias(fc[m][ct].x[3], qr1, j0 + 1);
                    fc[m][ct].x[6] = mask_bias(fc[m][ct].x[6], qr1, j1);
                    fc[m][ct].x[7] = mask_bias(fc[m][ct].x[7], qr1, j1 + 1);
                    mx1 = fmaxf(mx1, fmaxf(fmaxf(fc[m][ct].x[2], fc[m][ct].x[3]), fmaxf(fc[m][ct].x[6], fc[m][ct].x[7])));
                }
                mx0 = wmma_row_max_16(mx0);
                mx1 = wmma_row_max_16(mx1);
                if (is_leader) {
                    *s_hmax_ptr(qr0, cg) = mx0;
                    *s_hmax_ptr(qr1, cg) = mx1;
                }
            }
            asm volatile("bar.sync %0, 64;" :: "r"(rg + 1) : "memory");

#pragma unroll
            for (int m = 0; m < 3; ++m) {
                const int qr0 = qr_base + m * 16 + acc_r0;
                const int qr1 = qr_base + m * 16 + acc_r1;
                const float mx0 = fmaxf(*s_hmax_ptr(qr0, 0), *s_hmax_ptr(qr0, 1));
                const float mx1 = fmaxf(*s_hmax_ptr(qr1, 0), *s_hmax_ptr(qr1, 1));
                const float m_new0 = fmaxf(rm0[m], mx0);
                const float m_new1 = fmaxf(rm1[m], mx1);
                const float a0 = (rm0[m] > -1e20f) ? ((m_new0 == rm0[m]) ? 1.0f : fast_exp2f_hw(rm0[m] - m_new0)) : 0.0f;
                const float a1 = (rm1[m] > -1e20f) ? ((m_new1 == rm1[m]) ? 1.0f : fast_exp2f_hw(rm1[m] - m_new1)) : 0.0f;
                if (__any_sync(0xffffffffu, a0 != 1.0f || a1 != 1.0f)) {
#pragma unroll
                    for (int n = 0; n < 4; ++n) wmma_rescale_acc(acc[m][n], a0, a1);
                }
                float psum0 = 0.0f, psum1 = 0.0f;
#pragma unroll
                for (int ct = 0; ct < 2; ++ct) {
                    const float p00 = (fc[m][ct].x[0] > -1e20f) ? fast_exp2f_hw(fc[m][ct].x[0] - m_new0) : 0.0f;
                    const float p01 = (fc[m][ct].x[1] > -1e20f) ? fast_exp2f_hw(fc[m][ct].x[1] - m_new0) : 0.0f;
                    const float p04 = (fc[m][ct].x[4] > -1e20f) ? fast_exp2f_hw(fc[m][ct].x[4] - m_new0) : 0.0f;
                    const float p05 = (fc[m][ct].x[5] > -1e20f) ? fast_exp2f_hw(fc[m][ct].x[5] - m_new0) : 0.0f;
                    psum0 += (p00 + p01) + (p04 + p05);
                    *reinterpret_cast<__half2*>(sP + qr0 * 72 + kv_cbase + ct * 16 + acc_c0) = __halves2half2(__float2half(p00), __float2half(p01));
                    *reinterpret_cast<__half2*>(sP + qr0 * 72 + kv_cbase + ct * 16 + acc_c1) = __halves2half2(__float2half(p04), __float2half(p05));

                    const float p12 = (fc[m][ct].x[2] > -1e20f) ? fast_exp2f_hw(fc[m][ct].x[2] - m_new1) : 0.0f;
                    const float p13 = (fc[m][ct].x[3] > -1e20f) ? fast_exp2f_hw(fc[m][ct].x[3] - m_new1) : 0.0f;
                    const float p16 = (fc[m][ct].x[6] > -1e20f) ? fast_exp2f_hw(fc[m][ct].x[6] - m_new1) : 0.0f;
                    const float p17 = (fc[m][ct].x[7] > -1e20f) ? fast_exp2f_hw(fc[m][ct].x[7] - m_new1) : 0.0f;
                    psum1 += (p12 + p13) + (p16 + p17);
                    *reinterpret_cast<__half2*>(sP + qr1 * 72 + kv_cbase + ct * 16 + acc_c0) = __halves2half2(__float2half(p12), __float2half(p13));
                    *reinterpret_cast<__half2*>(sP + qr1 * 72 + kv_cbase + ct * 16 + acc_c1) = __halves2half2(__float2half(p16), __float2half(p17));
                }
                psum0 = wmma_row_sum_16(psum0);
                psum1 = wmma_row_sum_16(psum1);
                rm0[m] = m_new0; rl0[m] = rl0[m] * a0 + psum0;
                rm1[m] = m_new1; rl1[m] = rl1[m] * a1 + psum1;
            }
            __syncthreads();

            {
                const int j = jb + kv_r;
                const bool valid = j < j_end;
                __half* dst = sKV + kv_r * 136 + kv_c0;
                if (valid) {
                    const int slot = IS_SWA ? (pow2_cap ? (j & cap_mask) : (j % cap)) : j;
                    const __half2 dv2_t = __half2half2(__ldg(&vs[(size_t) slot * n_head_kv + hkv]));
                    const __half2 vbias = __hmul2(neg1152, dv2_t);
                    const int8_t* vp = vq + (size_t) slot * kv_dim + hkv * 128 + kv_c0;
#pragma unroll 1
                    for (int i = 0; i < 2; ++i) {
                        const uint4 raw = *reinterpret_cast<const uint4*>(vp + i * 16);
                        __half2 h2[8];
                        i8x4_to_h2x2_fma(raw.x, dv2_t, vbias, h2[0], h2[1]); i8x4_to_h2x2_fma(raw.y, dv2_t, vbias, h2[2], h2[3]);
                        i8x4_to_h2x2_fma(raw.z, dv2_t, vbias, h2[4], h2[5]); i8x4_to_h2x2_fma(raw.w, dv2_t, vbias, h2[6], h2[7]);
                        *reinterpret_cast<uint4*>(dst + i * 16) = *reinterpret_cast<const uint4*>(h2);
                        *reinterpret_cast<uint4*>(dst + i * 16 + 8) = *reinterpret_cast<const uint4*>(h2 + 4);
                    }
                } else {
                    const uint4 z = make_uint4(0, 0, 0, 0);
#pragma unroll
                    for (int i = 0; i < 4; ++i) *reinterpret_cast<uint4*>(dst + i * 8) = z;
                }
            }
            __syncthreads();

#pragma unroll 1
            for (int k = 0; k < 64; k += 16) {
#pragma unroll
                for (int n = 0; n < 4; ++n) {
                    wmma::fragment<wmma::matrix_b, 16, 16, 16, __half, wmma::row_major> fb;
                    wmma::load_matrix_sync(fb, sKV + k * 136 + pv_pc0 + n * 16, 136);
#pragma unroll
                    for (int m = 0; m < 3; ++m) {
                        wmma::fragment<wmma::matrix_a, 16, 16, 16, __half, wmma::row_major> fa;
                        wmma::load_matrix_sync(fa, sP + (qr_base + m * 16) * 72 + k, 72);
                        wmma::mma_sync(acc[m][n], fa, fb, acc[m][n]);
                    }
                }
            }
            __syncthreads();
        }
    }

    if (is_leader) {
#pragma unroll
        for (int m = 0; m < 3; ++m) {
            const int qr0 = qr_base + m * 16 + acc_r0;
            const int qr1 = qr_base + m * 16 + acc_r1;
            if (cg == 0) {
                *s_l0_ptr(qr0) = rl0[m]; *s_l0_ptr(qr1) = rl1[m];
                if constexpr (SPLIT_KV) { *s_m_ptr(qr0) = rm0[m]; *s_m_ptr(qr1) = rm1[m]; }
            } else {
                *s_l1_ptr(qr0) = rl0[m]; *s_l1_ptr(qr1) = rl1[m];
            }
        }
    }
    __syncthreads();

    // Epilogue: 2 passes x 96 rows (96*128*4 = 49,152 B <= 52,224 B sQ), vectorized float4 reads/writes.
    float* s_out = reinterpret_cast<float*>(sQ);
    for (int g_rg = 0; g_rg < 4; g_rg += 2) {
        if (rg == g_rg || rg == g_rg + 1) {
            const int local_r0 = (rg - g_rg) * 48;
#pragma unroll
            for (int m = 0; m < 3; ++m) {
#pragma unroll
                for (int n = 0; n < 4; ++n) {
                    wmma::store_matrix_sync(s_out + (local_r0 + m * 16) * 128 + pv_pc0 + n * 16,
                                            acc[m][n], 128, wmma::mem_row_major);
                }
            }
        }
        __syncthreads();
        for (int idx4 = tid; idx4 < 96 * 32; idx4 += 256) {
            const int r = idx4 >> 5, col4 = idx4 & 31, col = col4 << 2;
            const int qr = g_rg * 48 + r;
            const int head_g = qr >> 6, tok_r = qr & 63, t = t0 + tok_r;
            if (t < T) {
                const float4 v = reinterpret_cast<const float4*>(s_out)[idx4];
                if constexpr (SPLIT_KV) {
                    float* dst = split_out + (((size_t) t * split_n + sp) * n_head + (h0 + head_g)) * 130;
                    *reinterpret_cast<float2*>(dst + col)     = make_float2(v.x, v.y);
                    *reinterpret_cast<float2*>(dst + col + 2) = make_float2(v.z, v.w);
                    if (col == 0) {
                        dst[128] = *s_m_ptr(qr);
                        dst[129] = *s_l0_ptr(qr) + *s_l1_ptr(qr);
                    }
                } else {
                    const float inv_l = __fdividef(1.0f, *s_l0_ptr(qr) + *s_l1_ptr(qr));
                    const size_t oi = ((size_t) t * n_head + (h0 + head_g)) * 128 + col;
                    float4 o = make_float4(v.x * inv_l, v.y * inv_l, v.z * inv_l, v.w * inv_l);
                    if (gate) {
                        const float4 g4 = *reinterpret_cast<const float4*>(gate + oi);
                        o.x *= sigmf(g4.x); o.y *= sigmf(g4.y); o.z *= sigmf(g4.z); o.w *= sigmf(g4.w);
                    }
                    *reinterpret_cast<float4*>(out + oi) = o;
                }
            }
        }
        __syncthreads();
    }
#endif
}

/// Zero-sP, Zero-Spill FlashAttention-2 256-row BULK Split-KV kernel for deep global prefill (sm_70+):
/// Covers [0, bulk_aligned) across sp = 0..bulk_splits-1 (writes into split_out stride total_splits = bulk_splits + 1).
/// On Volta sm_70: uses native PTX HMMA.884 (mma.sync.aligned.m8n8k4) with 128-bit LDS.128 QK^T, 8-way batched
/// LDS.64 P*V, conflict-free stride-144 SMEM (92,160 B <= 96 KB), and compact packed4 softmax (74.9 TFLOPS = 89% cuBLAS).
/// On sm_75+: keeps WMMA 2x4 Q*K^T -> 0-shuffle wmma_acc_to_matA_16x16 -> 2x8 P*V (87,040 B SMEM).
__global__ void __launch_bounds__(256, 1) k_attn_q8_gqa3_pp_fa2_256_bulk(
        const float* __restrict__ Q,
        const int8_t* __restrict__ kq, const __half* __restrict__ ks,
        const int8_t* __restrict__ vq, const __half* __restrict__ vs,
        const float* __restrict__ sink, float* __restrict__ split_out,
        int n_head, int n_head_kv, int bulk_aligned,
        float kq_scale, int T, int bulk_splits, int total_splits) {
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 700 && __CUDA_ARCH__ < 750
    // Volta sm_70 native HMMA.884 path (stride 144 -> 92,160 B SMEM, bit-exact with WMMA path, +19% faster!).
    constexpr int STRIDE = 144;
    extern __shared__ __align__(16) char smem_raw[];
    __half* sQ = reinterpret_cast<__half*>(smem_raw);
    __half* sKV = sQ + 256 * STRIDE;

    const int hkv = blockIdx.y;
    const int g = blockIdx.x % 3;
    const int qt = blockIdx.x / 3;
    const int h = 3 * hkv + g;
    const int t0 = qt * 256;
    const int sp = blockIdx.z;
    const int tid = threadIdx.x, warp = tid >> 5, lane = tid & 31;
    const int kv_dim = n_head_kv * 128;
    constexpr float kLog2e = 1.4426950408889634f;
    const float qscale_all = kq_scale * kLog2e;

    const int n_tiles = bulk_aligned >> 6;
    const int t_per_sp = (n_tiles + bulk_splits - 1) / bulk_splits;
    const int j_start = sp * t_per_sp * 64;
    const int j_end = min(bulk_aligned, j_start + t_per_sp * 64);
    if (j_start >= bulk_aligned) {
        const int t = t0 + tid;
        if (t < T) {
            float* dst = split_out + (((size_t) t * total_splits + sp) * n_head + h) * 130;
            dst[128] = -1e30f;
            dst[129] = 0.0f;
        }
        return;
    }

    for (int base_r = 0; base_r < 256; base_r += 32) {
        const int qr = base_r + (tid >> 3), qc0 = (tid & 7) * 16;
        const int t = t0 + qr;
        __half2* dst2 = reinterpret_cast<__half2*>(sQ + qr * STRIDE + qc0);
        if (t < T) {
            const float* qp = Q + ((size_t) t * n_head + h) * 128 + qc0;
#pragma unroll
            for (int i = 0; i < 4; ++i) {
                const float4 v = *reinterpret_cast<const float4*>(qp + i * 4);
                dst2[i * 2 + 0] = __halves2half2(__float2half(v.x * qscale_all), __float2half(v.y * qscale_all));
                dst2[i * 2 + 1] = __halves2half2(__float2half(v.z * qscale_all), __float2half(v.w * qscale_all));
            }
        } else {
            const __half2 z2 = __halves2half2(__float2half(0.0f), __float2half(0.0f));
#pragma unroll
            for (int i = 0; i < 8; ++i) dst2[i] = z2;
        }
    }
    __syncthreads();

    auto mma_rc = [](float d[8], uint32_t a0, uint32_t a1, uint32_t b0, uint32_t b1) {
        asm volatile(
            "mma.sync.aligned.m8n8k4.row.col.f32.f16.f16.f32 "
            "{%0, %1, %2, %3, %4, %5, %6, %7}, {%8, %9}, {%10, %11}, {%0, %1, %2, %3, %4, %5, %6, %7};\n"
            : "+f"(d[0]), "+f"(d[1]), "+f"(d[2]), "+f"(d[3]), "+f"(d[4]), "+f"(d[5]), "+f"(d[6]), "+f"(d[7])
            : "r"(a0), "r"(a1), "r"(b0), "r"(b1));
    };
    auto mma_rr = [](float d[8], uint32_t a0, uint32_t a1, uint32_t b0, uint32_t b1) {
        asm volatile(
            "mma.sync.aligned.m8n8k4.row.row.f32.f16.f16.f32 "
            "{%0, %1, %2, %3, %4, %5, %6, %7}, {%8, %9}, {%10, %11}, {%0, %1, %2, %3, %4, %5, %6, %7};\n"
            : "+f"(d[0]), "+f"(d[1]), "+f"(d[2]), "+f"(d[3]), "+f"(d[4]), "+f"(d[5]), "+f"(d[6]), "+f"(d[7])
            : "r"(a0), "r"(a1), "r"(b0), "r"(b1));
    };

    const int qr_base = warp * 32;
    float rm0[2] = {-1e30f, -1e30f}, rm1[2] = {-1e30f, -1e30f};
    float rl0[2] = {0.0f, 0.0f}, rl1[2] = {0.0f, 0.0f};
    float acc[2][8][8];
#pragma unroll
    for (int m = 0; m < 2; ++m)
#pragma unroll
        for (int n = 0; n < 8; ++n)
#pragma unroll
            for (int i = 0; i < 8; ++i) acc[m][n][i] = 0.0f;

    const int kv_r = tid >> 2, kv_c0 = (tid & 3) * 32;
    const __half2 neg1152 = __half2half2(__float2half(-1152.0f));
    const int8_t* kq_base = kq + (size_t) hkv * 128 + kv_c0;
    const int8_t* vq_base = vq + (size_t) hkv * 128 + kv_c0;

    const int qa_row = ((lane & 16) >> 2) + ((lane & 4) << 1) + (lane & 3);
    const int kb_row = ((lane & 16) >> 2) + (lane & 8) + (lane & 3);
    const int vb_row = lane & 3;
    const int vb_col = (lane & 8) + ((lane & 16) >> 2);
    const bool hi8 = (lane & 8) != 0;
    const bool want_r1 = (lane & 2) != 0;

    const __half* sq_r0 = sQ + (qr_base + 0  + qa_row) * STRIDE;
    const __half* sq_r1 = sQ + (qr_base + 16 + qa_row) * STRIDE;
    const __half* sk_r  = sKV + kb_row * STRIDE;
    const __half* sv_r  = sKV + vb_row * STRIDE + vb_col;

    uint4 kraw0, kraw1;
    __half2 dk2;
    {
        const int j = j_start + kv_r;
        dk2 = __half2half2(__ldg(&ks[(size_t) j * n_head_kv + hkv]));
        const int8_t* kp = kq_base + (size_t) j * kv_dim;
        kraw0 = *reinterpret_cast<const uint4*>(kp);
        kraw1 = *reinterpret_cast<const uint4*>(kp + 16);
    }

    for (int jb = j_start; jb < j_end; jb += 64) {
        {
            __half* dst = sKV + kv_r * STRIDE + kv_c0;
            const __half2 kb = __hmul2(neg1152, dk2);
            __half2 h2[8];
            i8x4_to_h2x2_fma(kraw0.x, dk2, kb, h2[0], h2[1]); i8x4_to_h2x2_fma(kraw0.y, dk2, kb, h2[2], h2[3]);
            i8x4_to_h2x2_fma(kraw0.z, dk2, kb, h2[4], h2[5]); i8x4_to_h2x2_fma(kraw0.w, dk2, kb, h2[6], h2[7]);
            *reinterpret_cast<uint4*>(dst) = *reinterpret_cast<const uint4*>(h2);
            *reinterpret_cast<uint4*>(dst + 8) = *reinterpret_cast<const uint4*>(h2 + 4);
            i8x4_to_h2x2_fma(kraw1.x, dk2, kb, h2[0], h2[1]); i8x4_to_h2x2_fma(kraw1.y, dk2, kb, h2[2], h2[3]);
            i8x4_to_h2x2_fma(kraw1.z, dk2, kb, h2[4], h2[5]); i8x4_to_h2x2_fma(kraw1.w, dk2, kb, h2[6], h2[7]);
            *reinterpret_cast<uint4*>(dst + 16) = *reinterpret_cast<const uint4*>(h2);
            *reinterpret_cast<uint4*>(dst + 24) = *reinterpret_cast<const uint4*>(h2 + 4);
        }
        __syncthreads();

        float fc[2][4][8];
#pragma unroll
        for (int m = 0; m < 2; ++m)
#pragma unroll
            for (int ct = 0; ct < 4; ++ct)
#pragma unroll
                for (int i = 0; i < 8; ++i) fc[m][ct][i] = 0.0f;

#pragma unroll
        for (int k = 0; k < 128; k += 8) {
            const uint4 q0 = *reinterpret_cast<const uint4*>(sq_r0 + k);
            const uint4 q1 = *reinterpret_cast<const uint4*>(sq_r1 + k);
            const uint4 kb0 = *reinterpret_cast<const uint4*>(sk_r + 0  * STRIDE + k);
            const uint4 kb1 = *reinterpret_cast<const uint4*>(sk_r + 16 * STRIDE + k);
            const uint4 kb2 = *reinterpret_cast<const uint4*>(sk_r + 32 * STRIDE + k);
            const uint4 kb3 = *reinterpret_cast<const uint4*>(sk_r + 48 * STRIDE + k);
            mma_rc(fc[0][0], q0.x, q0.y, kb0.x, kb0.y); mma_rc(fc[1][0], q1.x, q1.y, kb0.x, kb0.y);
            mma_rc(fc[0][1], q0.x, q0.y, kb1.x, kb1.y); mma_rc(fc[1][1], q1.x, q1.y, kb1.x, kb1.y);
            mma_rc(fc[0][2], q0.x, q0.y, kb2.x, kb2.y); mma_rc(fc[1][2], q1.x, q1.y, kb2.x, kb2.y);
            mma_rc(fc[0][3], q0.x, q0.y, kb3.x, kb3.y); mma_rc(fc[1][3], q1.x, q1.y, kb3.x, kb3.y);
            mma_rc(fc[0][0], q0.z, q0.w, kb0.z, kb0.w); mma_rc(fc[1][0], q1.z, q1.w, kb0.z, kb0.w);
            mma_rc(fc[0][1], q0.z, q0.w, kb1.z, kb1.w); mma_rc(fc[1][1], q1.z, q1.w, kb1.z, kb1.w);
            mma_rc(fc[0][2], q0.z, q0.w, kb2.z, kb2.w); mma_rc(fc[1][2], q1.z, q1.w, kb2.z, kb2.w);
            mma_rc(fc[0][3], q0.z, q0.w, kb3.z, kb3.w); mma_rc(fc[1][3], q1.z, q1.w, kb3.z, kb3.w);
        }

        const int jv = jb + kv_r;
        const __half2 dv2 = __half2half2(__ldg(&vs[(size_t) jv * n_head_kv + hkv]));
        const int8_t* vp = vq_base + (size_t) jv * kv_dim;
        const uint4 vraw0 = *reinterpret_cast<const uint4*>(vp);
        const uint4 vraw1 = *reinterpret_cast<const uint4*>(vp + 16);

        uint32_t ph[2][4][4];
#pragma unroll
        for (int m = 0; m < 2; ++m) {
            float mx0 = -1e30f, mx1 = -1e30f;
#pragma unroll
            for (int ct = 0; ct < 4; ++ct) {
                mx0 = fmaxf(mx0, fmaxf(fmaxf(fc[m][ct][0], fc[m][ct][1]), fmaxf(fc[m][ct][4], fc[m][ct][5])));
                mx1 = fmaxf(mx1, fmaxf(fmaxf(fc[m][ct][2], fc[m][ct][3]), fmaxf(fc[m][ct][6], fc[m][ct][7])));
            }
            const float m_new0 = fmaxf(rm0[m], wmma_row_max_16(mx0));
            const float m_new1 = fmaxf(rm1[m], wmma_row_max_16(mx1));
            const float a0 = (rm0[m] > -1e20f) ? ((m_new0 == rm0[m]) ? 1.0f : fast_exp2f_hw(rm0[m] - m_new0)) : 0.0f;
            const float a1 = (rm1[m] > -1e20f) ? ((m_new1 == rm1[m]) ? 1.0f : fast_exp2f_hw(rm1[m] - m_new1)) : 0.0f;
            if (__any_sync(0xffffffffu, a0 != 1.0f || a1 != 1.0f)) {
#pragma unroll
                for (int n = 0; n < 8; ++n) {
                    acc[m][n][0] *= a0; acc[m][n][1] *= a0; acc[m][n][4] *= a0; acc[m][n][5] *= a0;
                    acc[m][n][2] *= a1; acc[m][n][3] *= a1; acc[m][n][6] *= a1; acc[m][n][7] *= a1;
                }
            }
            float ps0 = 0.0f, ps1 = 0.0f;
#pragma unroll
            for (int ct = 0; ct < 4; ++ct) {
                const float p00 = fast_exp2f_hw(fc[m][ct][0] - m_new0);
                const float p01 = fast_exp2f_hw(fc[m][ct][1] - m_new0);
                const float p04 = fast_exp2f_hw(fc[m][ct][4] - m_new0);
                const float p05 = fast_exp2f_hw(fc[m][ct][5] - m_new0);
                ps0 += (p00 + p01) + (p04 + p05);
                const float p12 = fast_exp2f_hw(fc[m][ct][2] - m_new1);
                const float p13 = fast_exp2f_hw(fc[m][ct][3] - m_new1);
                const float p16 = fast_exp2f_hw(fc[m][ct][6] - m_new1);
                const float p17 = fast_exp2f_hw(fc[m][ct][7] - m_new1);
                ps1 += (p12 + p13) + (p16 + p17);
                const uint32_t u0 = f2_to_h2u(p00, p01), u1 = f2_to_h2u(p04, p05);
                const uint32_t v0 = f2_to_h2u(p12, p13), v1 = f2_to_h2u(p16, p17);
                const uint32_t r0 = __shfl_xor_sync(0xffffffffu, want_r1 ? u0 : v0, 2);
                const uint32_t r1 = __shfl_xor_sync(0xffffffffu, want_r1 ? u1 : v1, 2);
                ph[m][ct][0] = want_r1 ? r0 : u0; ph[m][ct][1] = want_r1 ? v0 : r0;
                ph[m][ct][2] = want_r1 ? r1 : u1; ph[m][ct][3] = want_r1 ? v1 : r1;
            }
            rm0[m] = m_new0; rl0[m] = rl0[m] * a0 + wmma_row_sum_16(ps0);
            rm1[m] = m_new1; rl1[m] = rl1[m] * a1 + wmma_row_sum_16(ps1);
        }
        __syncthreads();

        {
            __half* dst = sKV + kv_r * STRIDE + kv_c0;
            const __half2 vb = __hmul2(neg1152, dv2);
            __half2 h2[8];
            i8x4_to_h2x2_fma(vraw0.x, dv2, vb, h2[0], h2[1]); i8x4_to_h2x2_fma(vraw0.y, dv2, vb, h2[2], h2[3]);
            i8x4_to_h2x2_fma(vraw0.z, dv2, vb, h2[4], h2[5]); i8x4_to_h2x2_fma(vraw0.w, dv2, vb, h2[6], h2[7]);
            *reinterpret_cast<uint4*>(dst) = *reinterpret_cast<const uint4*>(h2);
            *reinterpret_cast<uint4*>(dst + 8) = *reinterpret_cast<const uint4*>(h2 + 4);
            i8x4_to_h2x2_fma(vraw1.x, dv2, vb, h2[0], h2[1]); i8x4_to_h2x2_fma(vraw1.y, dv2, vb, h2[2], h2[3]);
            i8x4_to_h2x2_fma(vraw1.z, dv2, vb, h2[4], h2[5]); i8x4_to_h2x2_fma(vraw1.w, dv2, vb, h2[6], h2[7]);
            *reinterpret_cast<uint4*>(dst + 16) = *reinterpret_cast<const uint4*>(h2);
            *reinterpret_cast<uint4*>(dst + 24) = *reinterpret_cast<const uint4*>(h2 + 4);
        }
        __syncthreads();

        if (jb + 64 < j_end) {
            const int j = jb + 64 + kv_r;
            dk2 = __half2half2(__ldg(&ks[(size_t) j * n_head_kv + hkv]));
            const int8_t* kp = kq_base + (size_t) j * kv_dim;
            kraw0 = *reinterpret_cast<const uint4*>(kp);
            kraw1 = *reinterpret_cast<const uint4*>(kp + 16);
        }

#pragma unroll
        for (int ct = 0; ct < 4; ++ct) {
            uint32_t a0_0, a0_1, a0_2, a0_3, a0_4, a0_5, a0_6, a0_7;
            uint32_t a1_0, a1_1, a1_2, a1_3, a1_4, a1_5, a1_6, a1_7;
            {
                const uint32_t o0 = __shfl_xor_sync(0xffffffffu, ph[0][ct][0], 8);
                const uint32_t o1 = __shfl_xor_sync(0xffffffffu, ph[0][ct][1], 8);
                const uint32_t o2 = __shfl_xor_sync(0xffffffffu, ph[0][ct][2], 8);
                const uint32_t o3 = __shfl_xor_sync(0xffffffffu, ph[0][ct][3], 8);
                a0_0 = hi8 ? o0 : ph[0][ct][0]; a0_1 = hi8 ? o1 : ph[0][ct][1];
                a0_2 = hi8 ? o2 : ph[0][ct][2]; a0_3 = hi8 ? o3 : ph[0][ct][3];
                a0_4 = hi8 ? ph[0][ct][0] : o0; a0_5 = hi8 ? ph[0][ct][1] : o1;
                a0_6 = hi8 ? ph[0][ct][2] : o2; a0_7 = hi8 ? ph[0][ct][3] : o3;
            }
            {
                const uint32_t o0 = __shfl_xor_sync(0xffffffffu, ph[1][ct][0], 8);
                const uint32_t o1 = __shfl_xor_sync(0xffffffffu, ph[1][ct][1], 8);
                const uint32_t o2 = __shfl_xor_sync(0xffffffffu, ph[1][ct][2], 8);
                const uint32_t o3 = __shfl_xor_sync(0xffffffffu, ph[1][ct][3], 8);
                a1_0 = hi8 ? o0 : ph[1][ct][0]; a1_1 = hi8 ? o1 : ph[1][ct][1];
                a1_2 = hi8 ? o2 : ph[1][ct][2]; a1_3 = hi8 ? o3 : ph[1][ct][3];
                a1_4 = hi8 ? ph[1][ct][0] : o0; a1_5 = hi8 ? ph[1][ct][1] : o1;
                a1_6 = hi8 ? ph[1][ct][2] : o2; a1_7 = hi8 ? ph[1][ct][3] : o3;
            }
            const __half* sv_base = sv_r + (ct * 16) * STRIDE;
            uint2 b[8];
#pragma unroll
            for (int n = 0; n < 8; ++n) b[n] = *reinterpret_cast<const uint2*>(sv_base + 0 * STRIDE + n * 16);
#pragma unroll
            for (int n = 0; n < 8; ++n) { mma_rr(acc[0][n], a0_0, a0_1, b[n].x, b[n].y); mma_rr(acc[1][n], a1_0, a1_1, b[n].x, b[n].y); }
#pragma unroll
            for (int n = 0; n < 8; ++n) b[n] = *reinterpret_cast<const uint2*>(sv_base + 4 * STRIDE + n * 16);
#pragma unroll
            for (int n = 0; n < 8; ++n) { mma_rr(acc[0][n], a0_2, a0_3, b[n].x, b[n].y); mma_rr(acc[1][n], a1_2, a1_3, b[n].x, b[n].y); }
#pragma unroll
            for (int n = 0; n < 8; ++n) b[n] = *reinterpret_cast<const uint2*>(sv_base + 8 * STRIDE + n * 16);
#pragma unroll
            for (int n = 0; n < 8; ++n) { mma_rr(acc[0][n], a0_4, a0_5, b[n].x, b[n].y); mma_rr(acc[1][n], a1_4, a1_5, b[n].x, b[n].y); }
#pragma unroll
            for (int n = 0; n < 8; ++n) b[n] = *reinterpret_cast<const uint2*>(sv_base + 12 * STRIDE + n * 16);
#pragma unroll
            for (int n = 0; n < 8; ++n) { mma_rr(acc[0][n], a0_6, a0_7, b[n].x, b[n].y); mma_rr(acc[1][n], a1_6, a1_7, b[n].x, b[n].y); }
        }
        __syncthreads();
    }

    int acc_r0, acc_r1, acc_c0, acc_c1;
    wmma_acc_layout_16x16(lane, acc_r0, acc_r1, acc_c0, acc_c1);
    const bool is_leader = wmma_row_leader_16(lane);
    float* s_out = reinterpret_cast<float*>(smem_raw);
    float* s_rm = reinterpret_cast<float*>(smem_raw + (256 + 64) * STRIDE * 2 - 2048);
    float* s_rl = s_rm + 256;
    if (is_leader) {
#pragma unroll
        for (int m = 0; m < 2; ++m) {
            const int qr0 = qr_base + m * 16 + acc_r0, qr1 = qr_base + m * 16 + acc_r1;
            const int tr0 = t0 + qr0, tr1 = t0 + qr1;
            const float ns0 = (tr0 < T) ? (-__ldg(&sink[(size_t) tr0 * n_head + h]) * kLog2e) : 0.0f;
            const float ns1 = (tr1 < T) ? (-__ldg(&sink[(size_t) tr1 * n_head + h]) * kLog2e) : 0.0f;
            s_rm[qr0] = rm0[m] + ns0; s_rl[qr0] = rl0[m];
            s_rm[qr1] = rm1[m] + ns1; s_rl[qr1] = rl1[m];
        }
    }
    __syncthreads();

    for (int w_base = 0; w_base < 8; w_base += 4) {
        if (warp >= w_base && warp < w_base + 4) {
            const int lr0 = (warp - w_base) * 32;
#pragma unroll
            for (int m = 0; m < 2; ++m) {
#pragma unroll
                for (int n = 0; n < 8; ++n) {
                    float* dst0 = s_out + (lr0 + m * 16 + acc_r0) * 128 + n * 16;
                    float* dst1 = s_out + (lr0 + m * 16 + acc_r1) * 128 + n * 16;
                    *reinterpret_cast<float2*>(dst0 + acc_c0) = make_float2(acc[m][n][0], acc[m][n][1]);
                    *reinterpret_cast<float2*>(dst0 + acc_c1) = make_float2(acc[m][n][4], acc[m][n][5]);
                    *reinterpret_cast<float2*>(dst1 + acc_c0) = make_float2(acc[m][n][2], acc[m][n][3]);
                    *reinterpret_cast<float2*>(dst1 + acc_c1) = make_float2(acc[m][n][6], acc[m][n][7]);
                }
            }
        }
        __syncthreads();
        for (int idx4 = tid; idx4 < 128 * 32; idx4 += 256) {
            const int r = idx4 >> 5, col = (idx4 & 31) << 2;
            const int qr = w_base * 32 + r, t = t0 + qr;
            if (t < T) {
                const float4 v = reinterpret_cast<const float4*>(s_out)[idx4];
                float* dst = split_out + (((size_t) t * total_splits + sp) * n_head + h) * 130;
                *reinterpret_cast<float2*>(dst + col)     = make_float2(v.x, v.y);
                *reinterpret_cast<float2*>(dst + col + 2) = make_float2(v.z, v.w);
                if (col == 0) { dst[128] = s_rm[qr]; dst[129] = s_rl[qr]; }
            }
        }
        __syncthreads();
    }
#elif !defined(__CUDA_ARCH__) || __CUDA_ARCH__ >= 750
    using namespace nvcuda;
    extern __shared__ __align__(16) char smem_raw[];
    // 87,040 B <= 98,304 B: sQ (256*136*2 = 69,632 B) | sKV (64*136*2 = 17,408 B). ZERO sP!
    __half* sQ = reinterpret_cast<__half*>(smem_raw);
    __half* sKV = sQ + 256 * 136;

    const int hkv = blockIdx.y;
    const int g = blockIdx.x % 3;
    const int qt = blockIdx.x / 3;
    const int h = 3 * hkv + g;
    const int t0 = qt * 256;
    const int sp = blockIdx.z;
    const int tid = threadIdx.x, warp = tid >> 5, lane = tid & 31;
    const int kv_dim = n_head_kv * 128;
    constexpr float kLog2e = 1.4426950408889634f;
    const float qscale_all = kq_scale * kLog2e;

    const int n_tiles = bulk_aligned >> 6;
    const int t_per_sp = (n_tiles + bulk_splits - 1) / bulk_splits;
    const int j_start = sp * t_per_sp * 64;
    const int j_end = min(bulk_aligned, j_start + t_per_sp * 64);
    if (j_start >= bulk_aligned) {
        const int t = t0 + tid;
        if (t < T) {
            float* dst = split_out + (((size_t) t * total_splits + sp) * n_head + h) * 130;
            dst[128] = -1e30f;
            dst[129] = 0.0f;
        }
        return;
    }

    for (int base_r = 0; base_r < 256; base_r += 32) {
        const int qr = base_r + (tid >> 3), qc0 = (tid & 7) * 16;
        const int t = t0 + qr;
        __half2* dst2 = reinterpret_cast<__half2*>(sQ + qr * 136 + qc0);
        if (t < T) {
            const float* qp = Q + ((size_t) t * n_head + h) * 128 + qc0;
#pragma unroll
            for (int i = 0; i < 4; ++i) {
                const float4 v = *reinterpret_cast<const float4*>(qp + i * 4);
                dst2[i * 2 + 0] = __halves2half2(__float2half(v.x * qscale_all), __float2half(v.y * qscale_all));
                dst2[i * 2 + 1] = __halves2half2(__float2half(v.z * qscale_all), __float2half(v.w * qscale_all));
            }
        } else {
            const __half2 z2 = __halves2half2(__float2half(0.0f), __float2half(0.0f));
#pragma unroll
            for (int i = 0; i < 8; ++i) dst2[i] = z2;
        }
    }
    __syncthreads();

    const int qr_base = warp * 32;
    float rm0[2] = {-1e30f, -1e30f}, rm1[2] = {-1e30f, -1e30f};
    float rl0[2] = {0.0f, 0.0f}, rl1[2] = {0.0f, 0.0f};
    wmma::fragment<wmma::accumulator, 16, 16, 16, float> acc[2][8];
#pragma unroll
    for (int m = 0; m < 2; ++m) {
#pragma unroll
        for (int n = 0; n < 8; ++n) wmma::fill_fragment(acc[m][n], 0.0f);
    }

    const int kv_r = tid >> 2, kv_c0 = (tid & 3) * 32;
    const __half2 neg1152 = __half2half2(__float2half(-1152.0f));

    uint4 kraw0, kraw1;
    __half2 dk2;
    {
        const int j = j_start + kv_r;
        dk2 = __half2half2(__ldg(&ks[(size_t) j * n_head_kv + hkv]));
        const int8_t* kp = kq + (size_t) j * kv_dim + hkv * 128 + kv_c0;
        kraw0 = *reinterpret_cast<const uint4*>(kp);
        kraw1 = *reinterpret_cast<const uint4*>(kp + 16);
    }

    for (int jb = j_start; jb < j_end; jb += 64) {
        {
            __half* dst = sKV + kv_r * 136 + kv_c0;
            const __half2 kb = __hmul2(neg1152, dk2);
            __half2 h2[8];
            i8x4_to_h2x2_fma(kraw0.x, dk2, kb, h2[0], h2[1]); i8x4_to_h2x2_fma(kraw0.y, dk2, kb, h2[2], h2[3]);
            i8x4_to_h2x2_fma(kraw0.z, dk2, kb, h2[4], h2[5]); i8x4_to_h2x2_fma(kraw0.w, dk2, kb, h2[6], h2[7]);
            *reinterpret_cast<uint4*>(dst) = *reinterpret_cast<const uint4*>(h2);
            *reinterpret_cast<uint4*>(dst + 8) = *reinterpret_cast<const uint4*>(h2 + 4);
            i8x4_to_h2x2_fma(kraw1.x, dk2, kb, h2[0], h2[1]); i8x4_to_h2x2_fma(kraw1.y, dk2, kb, h2[2], h2[3]);
            i8x4_to_h2x2_fma(kraw1.z, dk2, kb, h2[4], h2[5]); i8x4_to_h2x2_fma(kraw1.w, dk2, kb, h2[6], h2[7]);
            *reinterpret_cast<uint4*>(dst + 16) = *reinterpret_cast<const uint4*>(h2);
            *reinterpret_cast<uint4*>(dst + 24) = *reinterpret_cast<const uint4*>(h2 + 4);
        }
        __syncthreads();

        uint4 vraw0, vraw1;
        __half2 dv2;
        {
            const int j = jb + kv_r;
            dv2 = __half2half2(__ldg(&vs[(size_t) j * n_head_kv + hkv]));
            const int8_t* vp = vq + (size_t) j * kv_dim + hkv * 128 + kv_c0;
            vraw0 = *reinterpret_cast<const uint4*>(vp);
            vraw1 = *reinterpret_cast<const uint4*>(vp + 16);
        }

        wmma::fragment<wmma::accumulator, 16, 16, 16, float> fc[2][4];
#pragma unroll
        for (int m = 0; m < 2; ++m) {
#pragma unroll
            for (int ct = 0; ct < 4; ++ct) wmma::fill_fragment(fc[m][ct], 0.0f);
        }
#pragma unroll
        for (int k = 0; k < 128; k += 16) {
            wmma::fragment<wmma::matrix_a, 16, 16, 16, __half, wmma::row_major> fa0, fa1;
            wmma::load_matrix_sync(fa0, sQ + (qr_base + 0) * 136 + k, 136);
            wmma::load_matrix_sync(fa1, sQ + (qr_base + 16) * 136 + k, 136);
#pragma unroll
            for (int ct = 0; ct < 4; ++ct) {
                wmma::fragment<wmma::matrix_b, 16, 16, 16, __half, wmma::col_major> fb;
                wmma::load_matrix_sync(fb, sKV + (ct * 16) * 136 + k, 136);
                wmma::mma_sync(fc[0][ct], fa0, fb, fc[0][ct]);
                wmma::mma_sync(fc[1][ct], fa1, fb, fc[1][ct]);
            }
        }
        __syncthreads();

        {
            __half* dst = sKV + kv_r * 136 + kv_c0;
            const __half2 vb = __hmul2(neg1152, dv2);
            __half2 h2[8];
            i8x4_to_h2x2_fma(vraw0.x, dv2, vb, h2[0], h2[1]); i8x4_to_h2x2_fma(vraw0.y, dv2, vb, h2[2], h2[3]);
            i8x4_to_h2x2_fma(vraw0.z, dv2, vb, h2[4], h2[5]); i8x4_to_h2x2_fma(vraw0.w, dv2, vb, h2[6], h2[7]);
            *reinterpret_cast<uint4*>(dst) = *reinterpret_cast<const uint4*>(h2);
            *reinterpret_cast<uint4*>(dst + 8) = *reinterpret_cast<const uint4*>(h2 + 4);
            i8x4_to_h2x2_fma(vraw1.x, dv2, vb, h2[0], h2[1]); i8x4_to_h2x2_fma(vraw1.y, dv2, vb, h2[2], h2[3]);
            i8x4_to_h2x2_fma(vraw1.z, dv2, vb, h2[4], h2[5]); i8x4_to_h2x2_fma(vraw1.w, dv2, vb, h2[6], h2[7]);
            *reinterpret_cast<uint4*>(dst + 16) = *reinterpret_cast<const uint4*>(h2);
            *reinterpret_cast<uint4*>(dst + 24) = *reinterpret_cast<const uint4*>(h2 + 4);
        }

        wmma::fragment<wmma::matrix_a, 16, 16, 16, __half, wmma::row_major> fa_p[2][4];
#pragma unroll
        for (int m = 0; m < 2; ++m) {
            float mx0 = -1e30f, mx1 = -1e30f;
#pragma unroll
            for (int ct = 0; ct < 4; ++ct) {
                mx0 = fmaxf(mx0, fmaxf(fmaxf(fc[m][ct].x[0], fc[m][ct].x[1]), fmaxf(fc[m][ct].x[4], fc[m][ct].x[5])));
                mx1 = fmaxf(mx1, fmaxf(fmaxf(fc[m][ct].x[2], fc[m][ct].x[3]), fmaxf(fc[m][ct].x[6], fc[m][ct].x[7])));
            }
            const float m_new0 = fmaxf(rm0[m], wmma_row_max_16(mx0));
            const float m_new1 = fmaxf(rm1[m], wmma_row_max_16(mx1));
            const float a0 = (rm0[m] > -1e20f) ? ((m_new0 == rm0[m]) ? 1.0f : fast_exp2f_hw(rm0[m] - m_new0)) : 0.0f;
            const float a1 = (rm1[m] > -1e20f) ? ((m_new1 == rm1[m]) ? 1.0f : fast_exp2f_hw(rm1[m] - m_new1)) : 0.0f;
            if (__any_sync(0xffffffffu, a0 != 1.0f || a1 != 1.0f)) {
#pragma unroll
                for (int n = 0; n < 8; ++n) wmma_rescale_acc(acc[m][n], a0, a1);
            }
            float ps0 = 0.0f, ps1 = 0.0f;
#pragma unroll
            for (int ct = 0; ct < 4; ++ct) wmma_acc_to_matA_16x16<false>(fc[m][ct], m_new0, m_new1, ps0, ps1, fa_p[m][ct], lane);
            rm0[m] = m_new0; rl0[m] = rl0[m] * a0 + wmma_row_sum_16(ps0);
            rm1[m] = m_new1; rl1[m] = rl1[m] * a1 + wmma_row_sum_16(ps1);
        }
        __syncthreads();

        if (jb + 64 < j_end) {
            const int j = jb + 64 + kv_r;
            dk2 = __half2half2(__ldg(&ks[(size_t) j * n_head_kv + hkv]));
            const int8_t* kp = kq + (size_t) j * kv_dim + hkv * 128 + kv_c0;
            kraw0 = *reinterpret_cast<const uint4*>(kp);
            kraw1 = *reinterpret_cast<const uint4*>(kp + 16);
        }
#pragma unroll
        for (int ct = 0; ct < 4; ++ct) {
#pragma unroll
            for (int n = 0; n < 8; ++n) {
                wmma::fragment<wmma::matrix_b, 16, 16, 16, __half, wmma::row_major> fb;
                wmma::load_matrix_sync(fb, sKV + (ct * 16) * 136 + n * 16, 136);
                wmma::mma_sync(acc[0][n], fa_p[0][ct], fb, acc[0][n]);
                wmma::mma_sync(acc[1][n], fa_p[1][ct], fb, acc[1][n]);
            }
        }
        __syncthreads();
    }

    int acc_r0, acc_r1, acc_c0, acc_c1;
    wmma_acc_layout_16x16(lane, acc_r0, acc_r1, acc_c0, acc_c1);
    const bool is_leader = wmma_row_leader_16(lane);
    float* s_out = reinterpret_cast<float*>(smem_raw);
    float* s_rm = reinterpret_cast<float*>(smem_raw + 87040 - 2048);
    float* s_rl = s_rm + 256;
    if (is_leader) {
#pragma unroll
        for (int m = 0; m < 2; ++m) {
            const int qr0 = qr_base + m * 16 + acc_r0, qr1 = qr_base + m * 16 + acc_r1;
            const int tr0 = t0 + qr0, tr1 = t0 + qr1;
            const float ns0 = (tr0 < T) ? (-__ldg(&sink[(size_t) tr0 * n_head + h]) * kLog2e) : 0.0f;
            const float ns1 = (tr1 < T) ? (-__ldg(&sink[(size_t) tr1 * n_head + h]) * kLog2e) : 0.0f;
            s_rm[qr0] = rm0[m] + ns0; s_rl[qr0] = rl0[m];
            s_rm[qr1] = rm1[m] + ns1; s_rl[qr1] = rl1[m];
        }
    }
    __syncthreads();

    for (int w_base = 0; w_base < 8; w_base += 4) {
        if (warp >= w_base && warp < w_base + 4) {
            const int lr0 = (warp - w_base) * 32;
#pragma unroll
            for (int m = 0; m < 2; ++m) {
#pragma unroll
                for (int n = 0; n < 8; ++n) {
                    wmma::store_matrix_sync(s_out + (lr0 + m * 16) * 128 + n * 16, acc[m][n], 128, wmma::mem_row_major);
                }
            }
        }
        __syncthreads();
        for (int idx4 = tid; idx4 < 128 * 32; idx4 += 256) {
            const int r = idx4 >> 5, col = (idx4 & 31) << 2;
            const int qr = w_base * 32 + r, t = t0 + qr;
            if (t < T) {
                const float4 v = reinterpret_cast<const float4*>(s_out)[idx4];
                float* dst = split_out + (((size_t) t * total_splits + sp) * n_head + h) * 130;
                *reinterpret_cast<float2*>(dst + col)     = make_float2(v.x, v.y);
                *reinterpret_cast<float2*>(dst + col + 2) = make_float2(v.z, v.w);
                if (col == 0) { dst[128] = s_rm[qr]; dst[129] = s_rl[qr]; }
            }
        }
        __syncthreads();
    }
#endif
}

/// Companion tail split kernel for k_attn_q8_gqa3_pp_fa2_256_bulk:
/// Covers [bulk_aligned, pos0[0] + T) (causal + rel_tab + sink 1.0) and writes to split index tail_sp = total_splits - 1.
__global__ void __launch_bounds__(256, 1) k_attn_q8_gqa3_pp_fa2_256_tail(
        const float* __restrict__ Q,
        const int8_t* __restrict__ kq, const __half* __restrict__ ks,
        const int8_t* __restrict__ vq, const __half* __restrict__ vs,
        const float* __restrict__ rel_tab, const float* __restrict__ sink,
        float* __restrict__ split_out,
        int n_head, int n_head_kv, int n_ext, int bulk_aligned,
        const int32_t* __restrict__ pos0, float kq_scale, int T, int tail_sp, int total_splits) {
#if !defined(__CUDA_ARCH__) || __CUDA_ARCH__ >= 700
    using namespace nvcuda;
    extern __shared__ __align__(16) char smem_raw[];
    __half* sQ = reinterpret_cast<__half*>(smem_raw);
    __half* sKV = sQ + 256 * 136;

    const int hkv = blockIdx.y;
    const int g = blockIdx.x % 3;
    const int qt = blockIdx.x / 3;
    const int h = 3 * hkv + g;
    const int t0 = qt * 256;
    const int tid = threadIdx.x, warp = tid >> 5, lane = tid & 31;
    const int kv_dim = n_head_kv * 128;
    constexpr float kLog2e = 1.4426950408889634f;
    const float qscale_all = kq_scale * kLog2e;

    const int p0 = pos0[0];
    const int t_last = min(t0 + 255, T - 1);
    const int j_end = p0 + t_last + 1;

    for (int base_r = 0; base_r < 256; base_r += 32) {
        const int qr = base_r + (tid >> 3), qc0 = (tid & 7) * 16;
        const int t = t0 + qr;
        __half2* dst2 = reinterpret_cast<__half2*>(sQ + qr * 136 + qc0);
        if (t < T) {
            const float* qp = Q + ((size_t) t * n_head + h) * 128 + qc0;
#pragma unroll
            for (int i = 0; i < 4; ++i) {
                const float4 v = *reinterpret_cast<const float4*>(qp + i * 4);
                dst2[i * 2 + 0] = __halves2half2(__float2half(v.x * qscale_all), __float2half(v.y * qscale_all));
                dst2[i * 2 + 1] = __halves2half2(__float2half(v.z * qscale_all), __float2half(v.w * qscale_all));
            }
        } else {
            const __half2 z2 = __halves2half2(__float2half(0.0f), __float2half(0.0f));
#pragma unroll
            for (int i = 0; i < 8; ++i) dst2[i] = z2;
        }
    }
    __syncthreads();

    int acc_r0, acc_r1, acc_c0, acc_c1;
    wmma_acc_layout_16x16(lane, acc_r0, acc_r1, acc_c0, acc_c1);
    const bool is_leader = wmma_row_leader_16(lane);
    const int qr_base = warp * 32;
    // Tail split carries the sink weight (m = 0, l = 1).
    float rm0[2] = {0.0f, 0.0f}, rm1[2] = {0.0f, 0.0f};
    float rl0[2] = {1.0f, 1.0f}, rl1[2] = {1.0f, 1.0f};
    wmma::fragment<wmma::accumulator, 16, 16, 16, float> acc[2][8];
#pragma unroll
    for (int m = 0; m < 2; ++m) for (int n = 0; n < 8; ++n) wmma::fill_fragment(acc[m][n], 0.0f);

    const int kv_r = tid >> 2, kv_c0 = (tid & 3) * 32;
    const __half2 neg1152 = __half2half2(__float2half(-1152.0f));
    auto mask_bias = [&](float s, int qr, int j) -> float {
        const int tr = t0 + qr;
        const int pos_q = p0 + tr;
        if (tr >= T || j > pos_q || j >= j_end) return -1e30f;
        const int d = pos_q - j;
        const float b = (d < n_ext)
            ? (__ldg(&rel_tab[((size_t) tr * n_head + h) * n_ext + d]) * kLog2e)
            : (-__ldg(&sink[(size_t) tr * n_head + h]) * kLog2e);
        return s + b;
    };

    for (int jb = bulk_aligned; jb < j_end; jb += 64) {
        {
            const int j = jb + kv_r;
            __half* dst = sKV + kv_r * 136 + kv_c0;
            if (j < j_end) {
                const __half2 dk2 = __half2half2(__ldg(&ks[(size_t) j * n_head_kv + hkv]));
                const __half2 kb = __hmul2(neg1152, dk2);
                const int8_t* kp = kq + (size_t) j * kv_dim + hkv * 128 + kv_c0;
#pragma unroll
                for (int i = 0; i < 2; ++i) {
                    const uint4 raw = *reinterpret_cast<const uint4*>(kp + i * 16);
                    __half2 h2[8];
                    i8x4_to_h2x2_fma(raw.x, dk2, kb, h2[0], h2[1]); i8x4_to_h2x2_fma(raw.y, dk2, kb, h2[2], h2[3]);
                    i8x4_to_h2x2_fma(raw.z, dk2, kb, h2[4], h2[5]); i8x4_to_h2x2_fma(raw.w, dk2, kb, h2[6], h2[7]);
                    *reinterpret_cast<uint4*>(dst + i * 16) = *reinterpret_cast<const uint4*>(h2);
                    *reinterpret_cast<uint4*>(dst + i * 16 + 8) = *reinterpret_cast<const uint4*>(h2 + 4);
                }
            } else {
                const uint4 z = make_uint4(0, 0, 0, 0);
#pragma unroll
                for (int i = 0; i < 4; ++i) *reinterpret_cast<uint4*>(dst + i * 8) = z;
            }
        }
        __syncthreads();

        wmma::fragment<wmma::accumulator, 16, 16, 16, float> fc[2][4];
#pragma unroll
        for (int m = 0; m < 2; ++m) for (int ct = 0; ct < 4; ++ct) wmma::fill_fragment(fc[m][ct], 0.0f);
#pragma unroll 1
        for (int k = 0; k < 128; k += 16) {
            wmma::fragment<wmma::matrix_a, 16, 16, 16, __half, wmma::row_major> fa0, fa1;
            wmma::load_matrix_sync(fa0, sQ + (qr_base + 0) * 136 + k, 136);
            wmma::load_matrix_sync(fa1, sQ + (qr_base + 16) * 136 + k, 136);
#pragma unroll
            for (int ct = 0; ct < 4; ++ct) {
                wmma::fragment<wmma::matrix_b, 16, 16, 16, __half, wmma::col_major> fb;
                wmma::load_matrix_sync(fb, sKV + (ct * 16) * 136 + k, 136);
                wmma::mma_sync(fc[0][ct], fa0, fb, fc[0][ct]);
                wmma::mma_sync(fc[1][ct], fa1, fb, fc[1][ct]);
            }
        }
        __syncthreads();

        {
            const int j = jb + kv_r;
            __half* dst = sKV + kv_r * 136 + kv_c0;
            if (j < j_end) {
                const __half2 dv2 = __half2half2(__ldg(&vs[(size_t) j * n_head_kv + hkv]));
                const __half2 vb = __hmul2(neg1152, dv2);
                const int8_t* vp = vq + (size_t) j * kv_dim + hkv * 128 + kv_c0;
#pragma unroll
                for (int i = 0; i < 2; ++i) {
                    const uint4 raw = *reinterpret_cast<const uint4*>(vp + i * 16);
                    __half2 h2[8];
                    i8x4_to_h2x2_fma(raw.x, dv2, vb, h2[0], h2[1]); i8x4_to_h2x2_fma(raw.y, dv2, vb, h2[2], h2[3]);
                    i8x4_to_h2x2_fma(raw.z, dv2, vb, h2[4], h2[5]); i8x4_to_h2x2_fma(raw.w, dv2, vb, h2[6], h2[7]);
                    *reinterpret_cast<uint4*>(dst + i * 16) = *reinterpret_cast<const uint4*>(h2);
                    *reinterpret_cast<uint4*>(dst + i * 16 + 8) = *reinterpret_cast<const uint4*>(h2 + 4);
                }
            } else {
                const uint4 z = make_uint4(0, 0, 0, 0);
#pragma unroll
                for (int i = 0; i < 4; ++i) *reinterpret_cast<uint4*>(dst + i * 8) = z;
            }
        }

        wmma::fragment<wmma::matrix_a, 16, 16, 16, __half, wmma::row_major> fa_p[2][4];
#pragma unroll
        for (int m = 0; m < 2; ++m) {
            const int qr0 = qr_base + m * 16 + acc_r0, qr1 = qr_base + m * 16 + acc_r1;
            float mx0 = -1e30f, mx1 = -1e30f;
#pragma unroll
            for (int ct = 0; ct < 4; ++ct) {
                const int j0 = jb + ct * 16 + acc_c0, j1 = jb + ct * 16 + acc_c1;
                fc[m][ct].x[0] = mask_bias(fc[m][ct].x[0], qr0, j0);
                fc[m][ct].x[1] = mask_bias(fc[m][ct].x[1], qr0, j0 + 1);
                fc[m][ct].x[4] = mask_bias(fc[m][ct].x[4], qr0, j1);
                fc[m][ct].x[5] = mask_bias(fc[m][ct].x[5], qr0, j1 + 1);
                mx0 = fmaxf(mx0, fmaxf(fmaxf(fc[m][ct].x[0], fc[m][ct].x[1]), fmaxf(fc[m][ct].x[4], fc[m][ct].x[5])));
                fc[m][ct].x[2] = mask_bias(fc[m][ct].x[2], qr1, j0);
                fc[m][ct].x[3] = mask_bias(fc[m][ct].x[3], qr1, j0 + 1);
                fc[m][ct].x[6] = mask_bias(fc[m][ct].x[6], qr1, j1);
                fc[m][ct].x[7] = mask_bias(fc[m][ct].x[7], qr1, j1 + 1);
                mx1 = fmaxf(mx1, fmaxf(fmaxf(fc[m][ct].x[2], fc[m][ct].x[3]), fmaxf(fc[m][ct].x[6], fc[m][ct].x[7])));
            }
            const float m_new0 = fmaxf(rm0[m], wmma_row_max_16(mx0));
            const float m_new1 = fmaxf(rm1[m], wmma_row_max_16(mx1));
            const float a0 = (m_new0 == rm0[m]) ? 1.0f : fast_exp2f_hw(rm0[m] - m_new0);
            const float a1 = (m_new1 == rm1[m]) ? 1.0f : fast_exp2f_hw(rm1[m] - m_new1);
            if (__any_sync(0xffffffffu, a0 != 1.0f || a1 != 1.0f)) {
#pragma unroll
                for (int n = 0; n < 8; ++n) wmma_rescale_acc(acc[m][n], a0, a1);
            }
            float ps0 = 0.0f, ps1 = 0.0f;
#pragma unroll
            for (int ct = 0; ct < 4; ++ct) wmma_acc_to_matA_16x16<true>(fc[m][ct], m_new0, m_new1, ps0, ps1, fa_p[m][ct], lane);
            rm0[m] = m_new0; rl0[m] = rl0[m] * a0 + wmma_row_sum_16(ps0);
            rm1[m] = m_new1; rl1[m] = rl1[m] * a1 + wmma_row_sum_16(ps1);
        }
        __syncthreads();

#pragma unroll
        for (int ct = 0; ct < 4; ++ct) {
#pragma unroll
            for (int n = 0; n < 8; ++n) {
                wmma::fragment<wmma::matrix_b, 16, 16, 16, __half, wmma::row_major> fb;
                wmma::load_matrix_sync(fb, sKV + (ct * 16) * 136 + n * 16, 136);
                wmma::mma_sync(acc[0][n], fa_p[0][ct], fb, acc[0][n]);
                wmma::mma_sync(acc[1][n], fa_p[1][ct], fb, acc[1][n]);
            }
        }
        __syncthreads();
    }

    float* s_out = reinterpret_cast<float*>(smem_raw);
    float* s_rm = reinterpret_cast<float*>(smem_raw + 87040 - 2048);
    float* s_rl = s_rm + 256;
    if (is_leader) {
#pragma unroll
        for (int m = 0; m < 2; ++m) {
            const int qr0 = qr_base + m * 16 + acc_r0, qr1 = qr_base + m * 16 + acc_r1;
            s_rm[qr0] = rm0[m]; s_rl[qr0] = rl0[m];
            s_rm[qr1] = rm1[m]; s_rl[qr1] = rl1[m];
        }
    }
    __syncthreads();

    for (int w_base = 0; w_base < 8; w_base += 4) {
        if (warp >= w_base && warp < w_base + 4) {
            const int lr0 = (warp - w_base) * 32;
#pragma unroll
            for (int m = 0; m < 2; ++m) {
#pragma unroll
                for (int n = 0; n < 8; ++n) {
                    wmma::store_matrix_sync(s_out + (lr0 + m * 16) * 128 + n * 16, acc[m][n], 128, wmma::mem_row_major);
                }
            }
        }
        __syncthreads();
        for (int idx4 = tid; idx4 < 128 * 32; idx4 += 256) {
            const int r = idx4 >> 5, col = (idx4 & 31) << 2;
            const int qr = w_base * 32 + r, t = t0 + qr;
            if (t < T) {
                const float4 v = reinterpret_cast<const float4*>(s_out)[idx4];
                float* dst = split_out + (((size_t) t * total_splits + tail_sp) * n_head + h) * 130;
                *reinterpret_cast<float2*>(dst + col)     = make_float2(v.x, v.y);
                *reinterpret_cast<float2*>(dst + col + 2) = make_float2(v.z, v.w);
                if (col == 0) { dst[128] = s_rm[qr]; dst[129] = s_rl[qr]; }
            }
        }
        __syncthreads();
    }
#endif
}

/// Combine Split-KV prefill FA2 outputs (log2-base m/l) and apply optional sigmoid gate. grid (n_head, T), block 128.
__global__ void k_attn_pp_split_combine(const float* __restrict__ split_out, const float* __restrict__ gate,
                                        float* __restrict__ out, int n_head, int split_n) {
    const int h = blockIdx.x, t = blockIdx.y, col = threadIdx.x;
    float mg = -1e30f;
    for (int sp = 0; sp < split_n; ++sp) {
        const float* src = split_out + (((size_t) t * split_n + sp) * n_head + h) * 130;
        if (src[129] > 0.0f) mg = fmaxf(mg, src[128]);
    }
    float lg = 0.0f, og = 0.0f;
    for (int sp = 0; sp < split_n; ++sp) {
        const float* src = split_out + (((size_t) t * split_n + sp) * n_head + h) * 130;
        const float l = src[129];
        if (l > 0.0f) {
            const float c = exp2f(src[128] - mg);
            lg += l * c;
            og += src[col] * c;
        }
    }
    const size_t oi = ((size_t) t * n_head + h) * 128 + col;
    float val = __fdividef(og, lg);
    if (gate) val *= sigmf(gate[oi]);
    out[oi] = val;
}

/// GQA-3 Split-KV Q8 decode / verify attention (1 <= T < 16):
/// grid (n_head_kv, split_n, T), block 128 (8 half-warps of 16 threads).
/// Each half-warp handles 1 key token via uint2 KV loads, SIMD i8x4_to_h2x2 decode, __half2 V acc, and log2 exp2f.
__global__ void __launch_bounds__(128, 8) k_attn_q8_gqa3_split(
        const float* __restrict__ Q,
        const int8_t* __restrict__ kq, const __half* __restrict__ ks,
        const int8_t* __restrict__ vq, const __half* __restrict__ vs,
        const float* __restrict__ rel_tab, const float* __restrict__ sink,
        float* __restrict__ split_out,
        int n_head, int n_head_kv, int n_ext, int cap, int is_swa, int n_swa,
        const int32_t* __restrict__ pos0, float kq_scale, int split_n) {
    __shared__ __half2 sq2[3][64];
    __shared__ float s_m[3][8];
    __shared__ float s_l[3][8];
    __shared__ float s_acc[8][128];
    const int hkv = blockIdx.x, sp = blockIdx.y, t = blockIdx.z;
    const int tid = threadIdx.x;
    const int hw = tid >> 4;
    const int hlane = tid & 15;
    const int kv_dim = n_head_kv * 128;
    const int h0 = 3 * hkv;
    constexpr float kLog2e = 1.4426950408889634f;
    const float kq_l2 = kq_scale * kLog2e;

    const int pos_q = pos0[0] + t;
    const int j0 = is_swa ? max(0, pos_q - n_swa + 1) : 0;
    const int j1 = pos_q + 1;
    const int n_tok = j1 - j0;
    const int chunk_sz = max(64, (n_tok + split_n - 1) / split_n);
    const int c_beg = j0 + sp * chunk_sz;
    if (c_beg >= j1) {
        if (tid < 3) {
            float* dst = split_out + (((size_t) t * split_n + sp) * n_head + (h0 + tid)) * 130;
            dst[128] = -1e30f;
            dst[129] = 0.0f;
        }
        return;
    }
    const int c_end = min(j1, c_beg + chunk_sz);

    if (tid < 64) {
#pragma unroll
        for (int g = 0; g < 3; ++g) {
            const float2 qf = *reinterpret_cast<const float2*>(Q + ((size_t) t * n_head + (h0 + g)) * 128 + tid * 2);
            sq2[g][tid] = __float22half2_rn(qf);
        }
    }
    __syncthreads();

    const int base8 = hlane * 8;
    __half2 q2[3][4];
#pragma unroll
    for (int g = 0; g < 3; ++g) {
        q2[g][0] = sq2[g][hlane * 4 + 0];
        q2[g][1] = sq2[g][hlane * 4 + 1];
        q2[g][2] = sq2[g][hlane * 4 + 2];
        q2[g][3] = sq2[g][hlane * 4 + 3];
    }

    float neg_sink2[3];
    const float* rt[3];
#pragma unroll
    for (int g = 0; g < 3; ++g) {
        const int h = h0 + g;
        rt[g] = rel_tab + ((size_t) t * n_head + h) * n_ext;
        neg_sink2[g] = -sink[(size_t) t * n_head + h] * kLog2e;
    }

    const bool has_sink = (sp == 0 && hw == 0);
    const __half2 z2 = __half2half2(__float2half(0.0f));
    float m[3], l[3];
    __half2 acc2[3][4];
#pragma unroll
    for (int g = 0; g < 3; ++g) {
        m[g] = has_sink ? 0.0f : -1e30f;
        l[g] = has_sink ? 1.0f : 0.0f;
#pragma unroll
        for (int i = 0; i < 4; ++i) acc2[g][i] = z2;
    }

    // Bulk loop for !is_swa tokens with dist >= n_ext: linear slot == j, bias == -sink[h].
    // Uses shifted-nsink (0 bulk adds), fast_exp2f_hw, lazy rescale (only when sc > m[g]), and 0xffffffffu warp shuffles.
    const int bulk_end = is_swa ? c_beg : min(c_end, max(c_beg, pos_q - n_ext + 1));
    if (c_beg < bulk_end) {
#pragma unroll
        for (int g = 0; g < 3; ++g) {
            if (m[g] > -1e20f) m[g] -= neg_sink2[g];
        }
        const int bulk_vec_end = c_beg + ((bulk_end - c_beg) & ~7);
        const unsigned hw_mask = (hw & 1) ? 0xffff0000u : 0x0000ffffu;
        int j = c_beg + hw;
        for (; j < bulk_vec_end; j += 8) {
            const size_t idx = (size_t) j * n_head_kv + hkv;
            const float dkkq2 = __half2float(__ldg(&ks[idx])) * kq_l2;
            const uint2 k8 = *reinterpret_cast<const uint2*>(kq + (size_t) j * kv_dim + hkv * 128 + base8);
            __half2 kh[4];
            i8x4_to_h2x2(k8.x, kh[0], kh[1]);
            i8x4_to_h2x2(k8.y, kh[2], kh[3]);
            float sc[3];
#pragma unroll
            for (int g = 0; g < 3; ++g) {
                __half2 d2 = __hmul2(q2[g][0], kh[0]);
                d2 = __hfma2(q2[g][1], kh[1], d2);
                d2 = __hfma2(q2[g][2], kh[2], d2);
                d2 = __hfma2(q2[g][3], kh[3], d2);
                float dt = __half2float(__low2half(d2)) + __half2float(__high2half(d2));
#pragma unroll
                for (int o = 8; o > 0; o >>= 1) dt += __shfl_xor_sync(hw_mask, dt, o, 16);
                sc[g] = dt * dkkq2;
            }
            // Exact FP16-zero V-skip: 2^-28 = 3.7e-9 -> __float2half(p * d_v) == 0.0 bit-identically!
            if (sc[0] - m[0] >= -28.0f || sc[1] - m[1] >= -28.0f || sc[2] - m[2] >= -28.0f) {
                const float d_v = __half2float(__ldg(&vs[idx]));
                const uint2 v8 = *reinterpret_cast<const uint2*>(vq + (size_t) j * kv_dim + hkv * 128 + base8);
                __half2 vh[4];
                i8x4_to_h2x2(v8.x, vh[0], vh[1]);
                i8x4_to_h2x2(v8.y, vh[2], vh[3]);
#pragma unroll
                for (int g = 0; g < 3; ++g) {
                    if (sc[g] - m[g] >= -28.0f) {
                        float p;
                        if (sc[g] > m[g]) {
                            const float alpha = l[g] > 0.0f ? fast_exp2f_hw(m[g] - sc[g]) : 0.0f;
                            const __half2 a2 = __half2half2(__float2half(alpha));
                            l[g] = l[g] * alpha + 1.0f;
#pragma unroll
                            for (int i = 0; i < 4; ++i) acc2[g][i] = __hmul2(acc2[g][i], a2);
                            m[g] = sc[g];
                            p = 1.0f;
                        } else {
                            p = fast_exp2f_hw(sc[g] - m[g]);
                            l[g] += p;
                        }
                        const __half2 pv2 = __half2half2(__float2half(p * d_v));
#pragma unroll
                        for (int i = 0; i < 4; ++i) acc2[g][i] = __hfma2(pv2, vh[i], acc2[g][i]);
                    }
                }
            }
        }
        if (j < bulk_end) {
            const size_t idx = (size_t) j * n_head_kv + hkv;
            const float dkkq2 = __half2float(__ldg(&ks[idx])) * kq_l2;
            const float d_v = __half2float(__ldg(&vs[idx]));
            const uint2 k8 = *reinterpret_cast<const uint2*>(kq + (size_t) j * kv_dim + hkv * 128 + base8);
            const uint2 v8 = *reinterpret_cast<const uint2*>(vq + (size_t) j * kv_dim + hkv * 128 + base8);
            __half2 kh[4], vh[4];
            i8x4_to_h2x2(k8.x, kh[0], kh[1]);
            i8x4_to_h2x2(k8.y, kh[2], kh[3]);
            i8x4_to_h2x2(v8.x, vh[0], vh[1]);
            i8x4_to_h2x2(v8.y, vh[2], vh[3]);
            const unsigned am = __activemask();
#pragma unroll
            for (int g = 0; g < 3; ++g) {
                __half2 d2 = __hmul2(q2[g][0], kh[0]);
                d2 = __hfma2(q2[g][1], kh[1], d2);
                d2 = __hfma2(q2[g][2], kh[2], d2);
                d2 = __hfma2(q2[g][3], kh[3], d2);
                float dt = __half2float(__low2half(d2)) + __half2float(__high2half(d2));
#pragma unroll
                for (int o = 8; o > 0; o >>= 1) dt += __shfl_xor_sync(am, dt, o, 16);
                const float sc = dt * dkkq2;
                float p;
                if (sc > m[g]) {
                    const float alpha = l[g] > 0.0f ? fast_exp2f_hw(m[g] - sc) : 0.0f;
                    const __half2 a2 = __half2half2(__float2half(alpha));
                    l[g] = l[g] * alpha + 1.0f;
#pragma unroll
                    for (int i = 0; i < 4; ++i) acc2[g][i] = __hmul2(acc2[g][i], a2);
                    m[g] = sc;
                    p = 1.0f;
                } else {
                    p = fast_exp2f_hw(sc - m[g]);
                    l[g] += p;
                }
                const __half2 pv2 = __half2half2(__float2half(p * d_v));
#pragma unroll
                for (int i = 0; i < 4; ++i) acc2[g][i] = __hfma2(pv2, vh[i], acc2[g][i]);
            }
        }
#pragma unroll
        for (int g = 0; g < 3; ++g) {
            if (m[g] > -1e20f) m[g] += neg_sink2[g];
        }
    }

    const bool pow2_cap = (cap & (cap - 1)) == 0;
    const int cap_mask = cap - 1;
    for (int jb = max(c_beg, bulk_end) + (hw & ~1); jb < c_end; jb += 8) {
        const int j = jb + (hw & 1);
        if (j < c_end) {
            const int slot = is_swa ? (pow2_cap ? (j & cap_mask) : (j % cap)) : j;
            const size_t idx = (size_t) slot * n_head_kv + hkv;
            const float dkkq2 = __half2float(__ldg(&ks[idx])) * kq_l2;
            const float d_v = __half2float(__ldg(&vs[idx]));
            const uint2 k8 = *reinterpret_cast<const uint2*>(kq + (size_t) slot * kv_dim + hkv * 128 + base8);
            const uint2 v8 = *reinterpret_cast<const uint2*>(vq + (size_t) slot * kv_dim + hkv * 128 + base8);
            __half2 kh[4], vh[4];
            i8x4_to_h2x2(k8.x, kh[0], kh[1]);
            i8x4_to_h2x2(k8.y, kh[2], kh[3]);
            i8x4_to_h2x2(v8.x, vh[0], vh[1]);
            i8x4_to_h2x2(v8.y, vh[2], vh[3]);
            const int dist = pos_q - j;
            const unsigned am = __activemask();
#pragma unroll
            for (int g = 0; g < 3; ++g) {
                __half2 d2 = __hmul2(q2[g][0], kh[0]);
                d2 = __hfma2(q2[g][1], kh[1], d2);
                d2 = __hfma2(q2[g][2], kh[2], d2);
                d2 = __hfma2(q2[g][3], kh[3], d2);
                float dt = __half2float(__low2half(d2)) + __half2float(__high2half(d2));
#pragma unroll
                for (int o = 8; o > 0; o >>= 1) dt += __shfl_xor_sync(am, dt, o, 16);
                const float bias = dist < n_ext ? (rt[g][dist] * kLog2e) : neg_sink2[g];
                const float sc = dt * dkkq2 + bias;
                float p;
                if (sc > m[g]) {
                    const float alpha = l[g] > 0.0f ? fast_exp2f_hw(m[g] - sc) : 0.0f;
                    const __half2 a2 = __half2half2(__float2half(alpha));
                    l[g] = l[g] * alpha + 1.0f;
#pragma unroll
                    for (int i = 0; i < 4; ++i) acc2[g][i] = __hmul2(acc2[g][i], a2);
                    m[g] = sc;
                    p = 1.0f;
                } else {
                    p = fast_exp2f_hw(sc - m[g]);
                    l[g] += p;
                }
                const __half2 pv2 = __half2half2(__float2half(p * d_v));
#pragma unroll
                for (int i = 0; i < 4; ++i) acc2[g][i] = __hfma2(pv2, vh[i], acc2[g][i]);
            }
        }
    }

#pragma unroll
    for (int g = 0; g < 3; ++g) {
        if (hlane == 0) {
            s_m[g][hw] = m[g];
            s_l[g][hw] = l[g];
        }
    }
#pragma unroll
    for (int g = 0; g < 3; ++g) {
#pragma unroll
        for (int i = 0; i < 4; ++i) {
            const float2 f2 = __half22float2(acc2[g][i]);
            s_acc[hw][base8 + 2 * i + 0] = f2.x;
            s_acc[hw][base8 + 2 * i + 1] = f2.y;
        }
        __syncthreads();
        const int h = h0 + g;
        float mg = -1e30f;
        for (int w = 0; w < 8; ++w) if (s_l[g][w] > 0.0f && s_m[g][w] > mg) mg = s_m[g][w];
        float lg = 0.0f, og = 0.0f;
        for (int w = 0; w < 8; ++w) {
            if (s_l[g][w] > 0.0f) {
                const float c = exp2f(s_m[g][w] - mg);
                lg += s_l[g][w] * c;
                og += s_acc[w][tid] * c;
            }
        }
        float* dst = split_out + (((size_t) t * split_n + sp) * n_head + h) * 130;
        dst[tid] = og;
        if (tid == 0) {
            dst[128] = mg;
            dst[129] = lg;
        }
        __syncthreads();
    }
}

/// Shared-KV Split-KV Q8 verify attention for T == 4 (non-SWA, GQA-3, head_dim=128):
/// Loads K/V once per BK=16 tile into shared memory instead of 4x from HBM;
/// pairs 2 half-warps per query token (parity 0/1) for full 8-half-warp occupancy and CUDA-graph dynamic *pos0.
__global__ void __launch_bounds__(128, 6) k_attn_q8_gqa3_verify4_split(
        const float* __restrict__ Q,
        const int8_t* __restrict__ kq, const __half* __restrict__ ks,
        const int8_t* __restrict__ vq, const __half* __restrict__ vs,
        const float* __restrict__ rel_tab, const float* __restrict__ sink,
        float* __restrict__ split_out,
        int n_head, int n_head_kv, int n_ext,
        const int32_t* __restrict__ pos0, float kq_scale, int split_n) {
    constexpr int BK = 16;
    __shared__ uint2 sK[BK][16];
    __shared__ uint2 sV[BK][16];
    __shared__ float sDK[BK];
    __shared__ float sDV[BK];
    __shared__ float s_m[3][8];
    __shared__ float s_l[3][8];
    __shared__ float s_acc[4][128];

    const int hkv = blockIdx.x, sp = blockIdx.y;
    const int tid = threadIdx.x, hw = tid >> 4, hlane = tid & 15;
    const int t = hw & 3;
    const int parity = hw >> 2;
    const int h0 = 3 * hkv, kv_dim = n_head_kv * 128;
    const int base8 = hlane * 8;
    constexpr float kLog2e = 1.4426950408889634f;
    const float kq_l2 = kq_scale * kLog2e;

    const int p0 = pos0[0];
    const int total = p0 + 4;
    const int chunk_sz = max(64, (total + split_n - 1) / split_n);
    const int c_beg = sp * chunk_sz;
    if (c_beg >= total) {
        if (parity == 0 && hlane == 0) {
            for (int g = 0; g < 3; ++g) {
                float* dst = split_out + (((size_t) t * split_n + sp) * n_head + (h0 + g)) * 130;
                dst[128] = -1e30f;
                dst[129] = 0.0f;
            }
        }
        return;
    }
    const int c_end = min(total, c_beg + chunk_sz);

    __half2 q2[3][4];
#pragma unroll
    for (int g = 0; g < 3; ++g) {
        const float* qp = Q + ((size_t) t * n_head + (h0 + g)) * 128 + base8;
        q2[g][0] = __float22half2_rn(*reinterpret_cast<const float2*>(qp + 0));
        q2[g][1] = __float22half2_rn(*reinterpret_cast<const float2*>(qp + 2));
        q2[g][2] = __float22half2_rn(*reinterpret_cast<const float2*>(qp + 4));
        q2[g][3] = __float22half2_rn(*reinterpret_cast<const float2*>(qp + 6));
    }

    const int pos_q = p0 + t;
    float neg_sink2[3];
    const float* rt[3];
#pragma unroll
    for (int g = 0; g < 3; ++g) {
        const int h = h0 + g;
        rt[g] = rel_tab + ((size_t) t * n_head + h) * n_ext;
        neg_sink2[g] = -sink[(size_t) t * n_head + h] * kLog2e;
    }

    const bool has_sink = (sp == 0 && parity == 0);
    const __half2 z2 = __half2half2(__float2half(0.0f));
    float m[3], l[3];
    __half2 acc2[3][4];
#pragma unroll
    for (int g = 0; g < 3; ++g) {
        m[g] = has_sink ? 0.0f : -1e30f;
        l[g] = has_sink ? 1.0f : 0.0f;
#pragma unroll
        for (int i = 0; i < 4; ++i) acc2[g][i] = z2;
    }

    const int bulk_limit = p0 - n_ext + 1;
    const int bulk_end = min(c_end, max(c_beg, bulk_limit));
    int jb = c_beg;
    for (; jb + BK <= bulk_end; jb += BK) {
        for (int r = hw; r < BK; r += 8) {
            const int j = jb + r;
            sK[r][hlane] = *reinterpret_cast<const uint2*>(kq + (size_t) j * kv_dim + hkv * 128 + base8);
            sV[r][hlane] = *reinterpret_cast<const uint2*>(vq + (size_t) j * kv_dim + hkv * 128 + base8);
            if (hlane == 0) {
                const size_t idx = (size_t) j * n_head_kv + hkv;
                sDK[r] = __half2float(__ldg(&ks[idx])) * kq_l2;
                sDV[r] = __half2float(__ldg(&vs[idx]));
            }
        }
        __syncthreads();

#pragma unroll
        for (int k = parity; k < BK; k += 2) {
            const uint2 k8 = sK[k][hlane];
            const uint2 v8 = sV[k][hlane];
            const float dkkq2 = sDK[k], d_v = sDV[k];
            __half2 kh[4], vh[4];
            i8x4_to_h2x2(k8.x, kh[0], kh[1]);
            i8x4_to_h2x2(k8.y, kh[2], kh[3]);
            i8x4_to_h2x2(v8.x, vh[0], vh[1]);
            i8x4_to_h2x2(v8.y, vh[2], vh[3]);
            float dots[3];
#pragma unroll
            for (int g = 0; g < 3; ++g) {
                __half2 d2 = __hmul2(q2[g][0], kh[0]);
                d2 = __hfma2(q2[g][1], kh[1], d2);
                d2 = __hfma2(q2[g][2], kh[2], d2);
                d2 = __hfma2(q2[g][3], kh[3], d2);
                float dt = __half2float(__low2half(d2)) + __half2float(__high2half(d2));
#pragma unroll
                for (int o = 8; o > 0; o >>= 1) dt += __shfl_xor_sync(__activemask(), dt, o, 16);
                dots[g] = dt;
            }
#pragma unroll
            for (int g = 0; g < 3; ++g) {
                const float sc = dots[g] * dkkq2 + neg_sink2[g];
                const float m_new = fmaxf(m[g], sc);
                const float alpha = l[g] > 0.0f ? exp2f(m[g] - m_new) : 0.0f;
                const float p = exp2f(sc - m_new);
                const __half2 a2 = __half2half2(__float2half(alpha));
                const __half2 pv2 = __half2half2(__float2half(p * d_v));
                l[g] = l[g] * alpha + p;
#pragma unroll
                for (int i = 0; i < 4; ++i) acc2[g][i] = __hfma2(acc2[g][i], a2, __hmul2(pv2, vh[i]));
                m[g] = m_new;
            }
        }
        __syncthreads();
    }

    for (; jb < c_end; jb += BK) {
        const int nk = min(BK, c_end - jb);
        for (int r = hw; r < nk; r += 8) {
            const int j = jb + r;
            sK[r][hlane] = *reinterpret_cast<const uint2*>(kq + (size_t) j * kv_dim + hkv * 128 + base8);
            sV[r][hlane] = *reinterpret_cast<const uint2*>(vq + (size_t) j * kv_dim + hkv * 128 + base8);
            if (hlane == 0) {
                const size_t idx = (size_t) j * n_head_kv + hkv;
                sDK[r] = __half2float(__ldg(&ks[idx])) * kq_l2;
                sDV[r] = __half2float(__ldg(&vs[idx]));
            }
        }
        __syncthreads();

        for (int k = parity; k < nk; k += 2) {
            const int j = jb + k;
            if (j <= pos_q) {
                const uint2 k8 = sK[k][hlane];
                const uint2 v8 = sV[k][hlane];
                const float dkkq2 = sDK[k], d_v = sDV[k];
                __half2 kh[4], vh[4];
                i8x4_to_h2x2(k8.x, kh[0], kh[1]);
                i8x4_to_h2x2(k8.y, kh[2], kh[3]);
                i8x4_to_h2x2(v8.x, vh[0], vh[1]);
                i8x4_to_h2x2(v8.y, vh[2], vh[3]);
                const int dist = pos_q - j;
                float dots[3];
#pragma unroll
                for (int g = 0; g < 3; ++g) {
                    __half2 d2 = __hmul2(q2[g][0], kh[0]);
                    d2 = __hfma2(q2[g][1], kh[1], d2);
                    d2 = __hfma2(q2[g][2], kh[2], d2);
                    d2 = __hfma2(q2[g][3], kh[3], d2);
                    float dt = __half2float(__low2half(d2)) + __half2float(__high2half(d2));
#pragma unroll
                    for (int o = 8; o > 0; o >>= 1) dt += __shfl_xor_sync(__activemask(), dt, o, 16);
                    dots[g] = dt;
                }
#pragma unroll
                for (int g = 0; g < 3; ++g) {
                    const float bias = dist < n_ext ? (rt[g][dist] * kLog2e) : neg_sink2[g];
                    const float sc = dots[g] * dkkq2 + bias;
                    const float m_new = fmaxf(m[g], sc);
                    const float alpha = l[g] > 0.0f ? exp2f(m[g] - m_new) : 0.0f;
                    const float p = exp2f(sc - m_new);
                    const __half2 a2 = __half2half2(__float2half(alpha));
                    const __half2 pv2 = __half2half2(__float2half(p * d_v));
                    l[g] = l[g] * alpha + p;
#pragma unroll
                    for (int i = 0; i < 4; ++i) acc2[g][i] = __hfma2(acc2[g][i], a2, __hmul2(pv2, vh[i]));
                    m[g] = m_new;
                }
            }
        }
        __syncthreads();
    }

#pragma unroll
    for (int g = 0; g < 3; ++g) {
        if (hlane == 0) {
            s_m[g][hw] = m[g];
            s_l[g][hw] = l[g];
        }
    }
#pragma unroll
    for (int g = 0; g < 3; ++g) {
        if (parity == 1) {
#pragma unroll
            for (int i = 0; i < 4; ++i) {
                const float2 f2 = __half22float2(acc2[g][i]);
                s_acc[t][base8 + 2 * i + 0] = f2.x;
                s_acc[t][base8 + 2 * i + 1] = f2.y;
            }
        }
        __syncthreads();
        if (parity == 0) {
            const float m0 = m[g], l0 = l[g];
            const float m1 = s_m[g][t + 4], l1 = s_l[g][t + 4];
            float mg = -1e30f;
            if (l0 > 0.0f && m0 > mg) mg = m0;
            if (l1 > 0.0f && m1 > mg) mg = m1;
            const float c0 = (l0 > 0.0f) ? exp2f(m0 - mg) : 0.0f;
            const float c1 = (l1 > 0.0f) ? exp2f(m1 - mg) : 0.0f;
            const float lg = l0 * c0 + l1 * c1;
            float* dst = split_out + (((size_t) t * split_n + sp) * n_head + (h0 + g)) * 130;
#pragma unroll
            for (int i = 0; i < 4; ++i) {
                const float2 f0 = __half22float2(acc2[g][i]);
                dst[base8 + 2 * i + 0] = f0.x * c0 + s_acc[t][base8 + 2 * i + 0] * c1;
                dst[base8 + 2 * i + 1] = f0.y * c0 + s_acc[t][base8 + 2 * i + 1] * c1;
            }
            if (hlane == 0) {
                dst[128] = mg;
                dst[129] = lg;
            }
        }
        __syncthreads();
    }
}

/// Zero-Sync Warp-Independent Tensor-Core (WMMA 16x16x16) Split-KV Q8 verify attention for T == 4 (non-SWA, GQA-3, D=128):
/// 4 warps (128 thr, 4 blk/SM, 21,760 B SMEM, 128 regs, 0 spills): each warp owns a private 16x136 sKV slice and steps
/// independently with ZERO __syncthreads() inside the entire 800K KV loop, ZERO sS/sP SMEM via wmma_acc_to_matA_16x16,
/// V prefetch overlapping QK^T, shifted-nsink bulk loop, lazy acc rescale, and single 4-warp epilogue combine.
__global__ void __launch_bounds__(128, 4) k_attn_q8_gqa3_verify4_wmma_split(
        const float* __restrict__ Q,
        const int8_t* __restrict__ kq, const __half* __restrict__ ks,
        const int8_t* __restrict__ vq, const __half* __restrict__ vs,
        const float* __restrict__ rel_tab, const float* __restrict__ sink,
        float* __restrict__ split_out,
        int n_head, int n_head_kv, int n_ext,
        const int32_t* __restrict__ pos0, float kq_scale, int split_n) {
#if !defined(__CUDA_ARCH__) || __CUDA_ARCH__ >= 700
    using namespace nvcuda;
    extern __shared__ __align__(16) char smem_raw[];
    // 21,760 B <= 22,016 B (4 blk/SM on V100/Ada): sQ (16*136*2 = 4,352 B) | sKV_all (4*16*136*2 = 17,408 B).
    __half* sQ = reinterpret_cast<__half*>(smem_raw);
    __half* sKV_all = sQ + 16 * 136;

    const int hkv = blockIdx.x, sp = blockIdx.y;
    const int tid = threadIdx.x, warp = tid >> 5, lane = tid & 31;
    const int h0 = 3 * hkv, kv_dim = n_head_kv * 128;
    constexpr float kLog2e = 1.4426950408889634f;
    const float qscale_all = kq_scale * kLog2e;

    const int p0 = pos0[0];
    const int total = p0 + 4;
    const int chunk_sz = max(64, (total + split_n - 1) / split_n);
    const int c_beg = sp * chunk_sz;
    if (c_beg >= total) {
        if (tid < 12) {
            float* dst = split_out + (((size_t) (tid / 3) * split_n + sp) * n_head + (h0 + (tid % 3))) * 130;
            dst[128] = -1e30f;
            dst[129] = 0.0f;
        }
        return;
    }
    const int c_end = min(total, c_beg + chunk_sz);

    {
        const int qr = tid >> 3, qc0 = (tid & 7) * 16;
        const float* qp = (qr < 12) ? (Q + ((size_t) (qr / 3) * n_head + (h0 + (qr % 3))) * 128 + qc0) : nullptr;
        __half2* dst2 = reinterpret_cast<__half2*>(sQ + qr * 136 + qc0);
        if (qp) {
#pragma unroll
            for (int i = 0; i < 4; ++i) {
                const float4 v = *reinterpret_cast<const float4*>(qp + i * 4);
                dst2[i * 2 + 0] = __halves2half2(__float2half(v.x * qscale_all), __float2half(v.y * qscale_all));
                dst2[i * 2 + 1] = __halves2half2(__float2half(v.z * qscale_all), __float2half(v.w * qscale_all));
            }
        } else {
            const __half2 z2 = __halves2half2(__float2half(0.0f), __float2half(0.0f));
#pragma unroll
            for (int i = 0; i < 8; ++i) dst2[i] = z2;
        }
    }
    __syncthreads();

    int acc_r0, acc_r1, acc_c0, acc_c1;
    wmma_acc_layout_16x16(lane, acc_r0, acc_r1, acc_c0, acc_c1);
    const bool is_leader = wmma_row_leader_16(lane);

    const bool has_sink = (sp == 0 && warp == 0);
    float rm0 = has_sink ? 0.0f : -1e30f;
    float rm1 = rm0;
    float rl0 = has_sink ? 1.0f : 0.0f;
    float rl1 = rl0;

    auto load_nsink = [&](int r) -> float {
        return (r < 12) ? (-__ldg(&sink[(size_t) (r / 3) * n_head + (h0 + (r % 3))]) * kLog2e) : 0.0f;
    };

    wmma::fragment<wmma::accumulator, 16, 16, 16, float> acc[8];
#pragma unroll
    for (int n = 0; n < 8; ++n) wmma::fill_fragment(acc[n], 0.0f);

    __half* sKV = sKV_all + warp * 16 * 136;
    const int wr = lane >> 1;          // 0..15 (2 lanes/row, 64 bytes/lane = 4 uint4s)
    const int wc0 = (lane & 1) * 64;
    const __half2 neg1152 = __half2half2(__float2half(-1152.0f));
    const int bulk_end = min(c_end, max(c_beg, p0 - n_ext + 1));

    int jb = c_beg + warp * 16;
    // Bulk loop: ZERO __syncthreads(), shifted-nsink (0 bulk adds), V prefetch overlapping QK^T!
    if (jb <= bulk_end - 16) {
        const float ns0 = load_nsink(acc_r0), ns1 = load_nsink(acc_r1);
        if (rm0 > -1e20f) rm0 -= ns0;
        if (rm1 > -1e20f) rm1 -= ns1;
        for (; jb <= bulk_end - 16; jb += 64) {
            const int j = jb + wr;
            {
                const __half2 dk2 = __half2half2(__ldg(&ks[(size_t) j * n_head_kv + hkv]));
                const __half2 kb = __hmul2(neg1152, dk2);
                const int8_t* kp = kq + (size_t) j * kv_dim + hkv * 128 + wc0;
                __half* dst = sKV + wr * 136 + wc0;
#pragma unroll
                for (int i = 0; i < 4; ++i) {
                    const uint4 raw = *reinterpret_cast<const uint4*>(kp + i * 16);
                    __half2 h2[8];
                    i8x4_to_h2x2_fma(raw.x, dk2, kb, h2[0], h2[1]); i8x4_to_h2x2_fma(raw.y, dk2, kb, h2[2], h2[3]);
                    i8x4_to_h2x2_fma(raw.z, dk2, kb, h2[4], h2[5]); i8x4_to_h2x2_fma(raw.w, dk2, kb, h2[6], h2[7]);
                    *reinterpret_cast<uint4*>(dst + i * 16)     = *reinterpret_cast<const uint4*>(h2);
                    *reinterpret_cast<uint4*>(dst + i * 16 + 8) = *reinterpret_cast<const uint4*>(h2 + 4);
                }
            }
            __syncwarp();

            const __half2 dv2 = __half2half2(__ldg(&vs[(size_t) j * n_head_kv + hkv]));
            const int8_t* vp = vq + (size_t) j * kv_dim + hkv * 128 + wc0;
            const uint4 vr0 = *reinterpret_cast<const uint4*>(vp);
            const uint4 vr1 = *reinterpret_cast<const uint4*>(vp + 16);
            const uint4 vr2 = *reinterpret_cast<const uint4*>(vp + 32);
            const uint4 vr3 = *reinterpret_cast<const uint4*>(vp + 48);

            wmma::fragment<wmma::accumulator, 16, 16, 16, float> fc;
            wmma::fill_fragment(fc, 0.0f);
#pragma unroll
            for (int k = 0; k < 128; k += 16) {
                wmma::fragment<wmma::matrix_a, 16, 16, 16, __half, wmma::row_major> fa;
                wmma::fragment<wmma::matrix_b, 16, 16, 16, __half, wmma::col_major> fb;
                wmma::load_matrix_sync(fa, sQ + k, 136);
                wmma::load_matrix_sync(fb, sKV + k, 136);
                wmma::mma_sync(fc, fa, fb, fc);
            }
            __syncwarp();

            {
                const __half2 vb = __hmul2(neg1152, dv2);
                __half* dst = sKV + wr * 136 + wc0;
                __half2 h2[8];
                i8x4_to_h2x2_fma(vr0.x, dv2, vb, h2[0], h2[1]); i8x4_to_h2x2_fma(vr0.y, dv2, vb, h2[2], h2[3]);
                i8x4_to_h2x2_fma(vr0.z, dv2, vb, h2[4], h2[5]); i8x4_to_h2x2_fma(vr0.w, dv2, vb, h2[6], h2[7]);
                *reinterpret_cast<uint4*>(dst) = *reinterpret_cast<const uint4*>(h2); *reinterpret_cast<uint4*>(dst + 8) = *reinterpret_cast<const uint4*>(h2 + 4);
                i8x4_to_h2x2_fma(vr1.x, dv2, vb, h2[0], h2[1]); i8x4_to_h2x2_fma(vr1.y, dv2, vb, h2[2], h2[3]);
                i8x4_to_h2x2_fma(vr1.z, dv2, vb, h2[4], h2[5]); i8x4_to_h2x2_fma(vr1.w, dv2, vb, h2[6], h2[7]);
                *reinterpret_cast<uint4*>(dst + 16) = *reinterpret_cast<const uint4*>(h2); *reinterpret_cast<uint4*>(dst + 24) = *reinterpret_cast<const uint4*>(h2 + 4);
                i8x4_to_h2x2_fma(vr2.x, dv2, vb, h2[0], h2[1]); i8x4_to_h2x2_fma(vr2.y, dv2, vb, h2[2], h2[3]);
                i8x4_to_h2x2_fma(vr2.z, dv2, vb, h2[4], h2[5]); i8x4_to_h2x2_fma(vr2.w, dv2, vb, h2[6], h2[7]);
                *reinterpret_cast<uint4*>(dst + 32) = *reinterpret_cast<const uint4*>(h2); *reinterpret_cast<uint4*>(dst + 40) = *reinterpret_cast<const uint4*>(h2 + 4);
                i8x4_to_h2x2_fma(vr3.x, dv2, vb, h2[0], h2[1]); i8x4_to_h2x2_fma(vr3.y, dv2, vb, h2[2], h2[3]);
                i8x4_to_h2x2_fma(vr3.z, dv2, vb, h2[4], h2[5]); i8x4_to_h2x2_fma(vr3.w, dv2, vb, h2[6], h2[7]);
                *reinterpret_cast<uint4*>(dst + 48) = *reinterpret_cast<const uint4*>(h2); *reinterpret_cast<uint4*>(dst + 56) = *reinterpret_cast<const uint4*>(h2 + 4);
            }

            const float mx0 = wmma_row_max_16(fmaxf(fmaxf(fc.x[0], fc.x[1]), fmaxf(fc.x[4], fc.x[5])));
            const float mx1 = wmma_row_max_16(fmaxf(fmaxf(fc.x[2], fc.x[3]), fmaxf(fc.x[6], fc.x[7])));
            const float m_new0 = fmaxf(rm0, mx0), m_new1 = fmaxf(rm1, mx1);
            const float a0 = (rm0 > -1e20f) ? ((m_new0 == rm0) ? 1.0f : fast_exp2f_hw(rm0 - m_new0)) : 0.0f;
            const float a1 = (rm1 > -1e20f) ? ((m_new1 == rm1) ? 1.0f : fast_exp2f_hw(rm1 - m_new1)) : 0.0f;
            if (__any_sync(0xffffffffu, a0 != 1.0f || a1 != 1.0f)) {
#pragma unroll
                for (int n = 0; n < 8; ++n) wmma_rescale_acc(acc[n], a0, a1);
            }
            wmma::fragment<wmma::matrix_a, 16, 16, 16, __half, wmma::row_major> fa_p;
            float ps0 = 0.0f, ps1 = 0.0f;
            wmma_acc_to_matA_16x16<false>(fc, m_new0, m_new1, ps0, ps1, fa_p, lane);
            rm0 = m_new0; rl0 = rl0 * a0 + wmma_row_sum_16(ps0);
            rm1 = m_new1; rl1 = rl1 * a1 + wmma_row_sum_16(ps1);
            __syncwarp();

#pragma unroll
            for (int n = 0; n < 8; ++n) {
                wmma::fragment<wmma::matrix_b, 16, 16, 16, __half, wmma::row_major> fb;
                wmma::load_matrix_sync(fb, sKV + n * 16, 136);
                wmma::mma_sync(acc[n], fa_p, fb, acc[n]);
            }
            __syncwarp();
        }
        if (rm0 > -1e20f) rm0 += ns0;
        if (rm1 > -1e20f) rm1 += ns1;
    }

    // Tail loop (causal + rel_tab; #pragma unroll 1 keeps bulk loop at 128 regs / 0 spills).
    if (jb < c_end) {
        auto mask_bias = [&](float s, int r, int j) -> float {
            if (r >= 12 || j >= c_end) return -1e30f;
            const int rt = r / 3, h = h0 + (r % 3), pos_q = p0 + rt;
            if (j > pos_q) return -1e30f;
            const int d = pos_q - j;
            const float b = (d < n_ext)
                ? (__ldg(&rel_tab[((size_t) rt * n_head + h) * n_ext + d]) * kLog2e)
                : (-__ldg(&sink[(size_t) rt * n_head + h]) * kLog2e);
            return s + b;
        };
        for (; jb < c_end; jb += 64) {
            const int j = jb + wr;
            {
                __half* dst = sKV + wr * 136 + wc0;
                if (j < c_end) {
                    const __half2 dk2 = __half2half2(__ldg(&ks[(size_t) j * n_head_kv + hkv]));
                    const __half2 kb = __hmul2(neg1152, dk2);
                    const int8_t* kp = kq + (size_t) j * kv_dim + hkv * 128 + wc0;
#pragma unroll 1
                    for (int i = 0; i < 4; ++i) {
                        const uint4 raw = *reinterpret_cast<const uint4*>(kp + i * 16);
                        __half2 h2[8];
                        i8x4_to_h2x2_fma(raw.x, dk2, kb, h2[0], h2[1]); i8x4_to_h2x2_fma(raw.y, dk2, kb, h2[2], h2[3]);
                        i8x4_to_h2x2_fma(raw.z, dk2, kb, h2[4], h2[5]); i8x4_to_h2x2_fma(raw.w, dk2, kb, h2[6], h2[7]);
                        *reinterpret_cast<uint4*>(dst + i * 16)     = *reinterpret_cast<const uint4*>(h2);
                        *reinterpret_cast<uint4*>(dst + i * 16 + 8) = *reinterpret_cast<const uint4*>(h2 + 4);
                    }
                } else {
                    const uint4 z = make_uint4(0, 0, 0, 0);
#pragma unroll 1
                    for (int i = 0; i < 8; ++i) *reinterpret_cast<uint4*>(dst + i * 8) = z;
                }
            }
            __syncwarp();
            wmma::fragment<wmma::accumulator, 16, 16, 16, float> fc;
            wmma::fill_fragment(fc, 0.0f);
#pragma unroll 1
            for (int k = 0; k < 128; k += 16) {
                wmma::fragment<wmma::matrix_a, 16, 16, 16, __half, wmma::row_major> fa;
                wmma::fragment<wmma::matrix_b, 16, 16, 16, __half, wmma::col_major> fb;
                wmma::load_matrix_sync(fa, sQ + k, 136);
                wmma::load_matrix_sync(fb, sKV + k, 136);
                wmma::mma_sync(fc, fa, fb, fc);
            }
            __syncwarp();
            {
                __half* dst = sKV + wr * 136 + wc0;
                if (j < c_end) {
                    const __half2 dv2 = __half2half2(__ldg(&vs[(size_t) j * n_head_kv + hkv]));
                    const __half2 vb = __hmul2(neg1152, dv2);
                    const int8_t* vp = vq + (size_t) j * kv_dim + hkv * 128 + wc0;
#pragma unroll 1
                    for (int i = 0; i < 4; ++i) {
                        const uint4 raw = *reinterpret_cast<const uint4*>(vp + i * 16);
                        __half2 h2[8];
                        i8x4_to_h2x2_fma(raw.x, dv2, vb, h2[0], h2[1]); i8x4_to_h2x2_fma(raw.y, dv2, vb, h2[2], h2[3]);
                        i8x4_to_h2x2_fma(raw.z, dv2, vb, h2[4], h2[5]); i8x4_to_h2x2_fma(raw.w, dv2, vb, h2[6], h2[7]);
                        *reinterpret_cast<uint4*>(dst + i * 16)     = *reinterpret_cast<const uint4*>(h2);
                        *reinterpret_cast<uint4*>(dst + i * 16 + 8) = *reinterpret_cast<const uint4*>(h2 + 4);
                    }
                } else {
                    const uint4 z = make_uint4(0, 0, 0, 0);
#pragma unroll 1
                    for (int i = 0; i < 8; ++i) *reinterpret_cast<uint4*>(dst + i * 8) = z;
                }
            }
            const int j0 = jb + acc_c0, j1 = jb + acc_c1;
            fc.x[0] = mask_bias(fc.x[0], acc_r0, j0);     fc.x[1] = mask_bias(fc.x[1], acc_r0, j0 + 1);
            fc.x[4] = mask_bias(fc.x[4], acc_r0, j1);     fc.x[5] = mask_bias(fc.x[5], acc_r0, j1 + 1);
            fc.x[2] = mask_bias(fc.x[2], acc_r1, j0);     fc.x[3] = mask_bias(fc.x[3], acc_r1, j0 + 1);
            fc.x[6] = mask_bias(fc.x[6], acc_r1, j1);     fc.x[7] = mask_bias(fc.x[7], acc_r1, j1 + 1);
            const float mx0 = wmma_row_max_16(fmaxf(fmaxf(fc.x[0], fc.x[1]), fmaxf(fc.x[4], fc.x[5])));
            const float mx1 = wmma_row_max_16(fmaxf(fmaxf(fc.x[2], fc.x[3]), fmaxf(fc.x[6], fc.x[7])));
            const float m_new0 = fmaxf(rm0, mx0), m_new1 = fmaxf(rm1, mx1);
            const float a0 = (rm0 > -1e20f) ? ((m_new0 == rm0) ? 1.0f : fast_exp2f_hw(rm0 - m_new0)) : 0.0f;
            const float a1 = (rm1 > -1e20f) ? ((m_new1 == rm1) ? 1.0f : fast_exp2f_hw(rm1 - m_new1)) : 0.0f;
            if (__any_sync(0xffffffffu, a0 != 1.0f || a1 != 1.0f)) {
#pragma unroll
                for (int n = 0; n < 8; ++n) wmma_rescale_acc(acc[n], a0, a1);
            }
            wmma::fragment<wmma::matrix_a, 16, 16, 16, __half, wmma::row_major> fa_p;
            float ps0 = 0.0f, ps1 = 0.0f;
            wmma_acc_to_matA_16x16<true>(fc, m_new0, m_new1, ps0, ps1, fa_p, lane);
            rm0 = m_new0; rl0 = rl0 * a0 + wmma_row_sum_16(ps0);
            rm1 = m_new1; rl1 = rl1 * a1 + wmma_row_sum_16(ps1);
            __syncwarp();

#pragma unroll
            for (int n = 0; n < 8; ++n) {
                wmma::fragment<wmma::matrix_b, 16, 16, 16, __half, wmma::row_major> fb;
                wmma::load_matrix_sync(fb, sKV + n * 16, 136);
                wmma::mma_sync(acc[n], fa_p, fb, acc[n]);
            }
            __syncwarp();
        }
    }

    // Single 4-warp epilogue combine (reuses sQ for s_wm/s_wl and sKV_all for s_out + s_tmp = 16,384 B <= 17,408 B).
    __syncthreads();
    float* s_wm = reinterpret_cast<float*>(sQ);          // [4][16] = 256 B
    float* s_wl = s_wm + 64;                             // [4][16] = 256 B
    float* s_mg = s_wl + 64;                             // [16]    = 64 B
    float* s_lg = s_mg + 16;                             // [16]    = 64 B
    if (is_leader) {
        s_wm[warp * 16 + acc_r0] = rm0; s_wl[warp * 16 + acc_r0] = rl0;
        s_wm[warp * 16 + acc_r1] = rm1; s_wl[warp * 16 + acc_r1] = rl1;
    }
    __syncthreads();
    if (tid < 12) {
        float mg = -1e30f;
#pragma unroll
        for (int w = 0; w < 4; ++w) if (s_wl[w * 16 + tid] > 0.0f) mg = fmaxf(mg, s_wm[w * 16 + tid]);
        float lg = 0.0f;
#pragma unroll
        for (int w = 0; w < 4; ++w) if (s_wl[w * 16 + tid] > 0.0f) lg += s_wl[w * 16 + tid] * fast_exp2f_hw(s_wm[w * 16 + tid] - mg);
        s_mg[tid] = mg;
        s_lg[tid] = lg;
    }
    __syncthreads();
    {
        const float cw0 = (acc_r0 < 12 && rl0 > 0.0f) ? fast_exp2f_hw(rm0 - s_mg[acc_r0]) : 0.0f;
        const float cw1 = (acc_r1 < 12 && rl1 > 0.0f) ? fast_exp2f_hw(rm1 - s_mg[acc_r1]) : 0.0f;
#pragma unroll
        for (int n = 0; n < 8; ++n) wmma_rescale_acc(acc[n], cw0, cw1);
    }
    float* s_out = reinterpret_cast<float*>(sKV_all);
    float* s_tmp = s_out + 16 * 128;
    if (warp == 0) {
#pragma unroll
        for (int n = 0; n < 8; ++n) wmma::store_matrix_sync(s_out + n * 16, acc[n], 128, wmma::mem_row_major);
    } else if (warp == 1) {
#pragma unroll
        for (int n = 0; n < 8; ++n) wmma::store_matrix_sync(s_tmp + n * 16, acc[n], 128, wmma::mem_row_major);
    }
    __syncthreads();
#pragma unroll
    for (int r = 0; r < 12; ++r) s_out[r * 128 + tid] += s_tmp[r * 128 + tid];
    __syncthreads();
    if (warp == 2) {
#pragma unroll
        for (int n = 0; n < 8; ++n) wmma::store_matrix_sync(s_tmp + n * 16, acc[n], 128, wmma::mem_row_major);
    }
    __syncthreads();
#pragma unroll
    for (int r = 0; r < 12; ++r) s_out[r * 128 + tid] += s_tmp[r * 128 + tid];
    __syncthreads();
    if (warp == 3) {
#pragma unroll
        for (int n = 0; n < 8; ++n) wmma::store_matrix_sync(s_tmp + n * 16, acc[n], 128, wmma::mem_row_major);
    }
    __syncthreads();
#pragma unroll
    for (int r = 0; r < 12; ++r) {
        float* dst = split_out + (((size_t) (r / 3) * split_n + sp) * n_head + (h0 + (r % 3))) * 130;
        dst[tid] = s_out[r * 128 + tid] + s_tmp[r * 128 + tid];
        if (tid == 0) {
            dst[128] = s_mg[r];
            dst[129] = s_lg[r];
        }
    }
#endif
}

/// Combine Split-KV partials across active splits and apply fused sigmoid gate: grid (n_head, T), block 128.
__global__ void k_attn_split_combine(const float* __restrict__ split_out, const float* __restrict__ gate,
                                     float* __restrict__ out, int n_head, int split_n,
                                     const int32_t* __restrict__ pos0, int is_swa, int n_swa) {
    const int h = blockIdx.x, t = blockIdx.y, d = threadIdx.x;
    const int pos_q = pos0[0] + t;
    const int n_tok = is_swa ? min(pos_q + 1, n_swa) : (pos_q + 1);
    const int chunk_sz = max(64, (n_tok + split_n - 1) / split_n);
    const int active_splits = min(split_n, (n_tok + chunk_sz - 1) / chunk_sz);

    float mg = -1e30f;
    for (int s = 0; s < active_splits; ++s) {
        const float* src = split_out + (((size_t) t * split_n + s) * n_head + h) * 130;
        const float ls = src[129];
        const float ms = src[128];
        if (ls > 0.0f && ms > mg) mg = ms;
    }
    float lg = 0.0f, og = 0.0f;
    for (int s = 0; s < active_splits; ++s) {
        const float* src = split_out + (((size_t) t * split_n + s) * n_head + h) * 130;
        const float ls = src[129];
        if (ls > 0.0f) {
            const float c = exp2f(src[128] - mg);
            lg += ls * c;
            og += src[d] * c;
        }
    }
    const size_t oi = ((size_t) t * n_head + h) * 128 + d;
    float val = __fdividef(og, lg);
    if (gate) val *= sigmf(gate[oi]);
    out[oi] = val;
}

/// Decode / single-token-group attention with relative-position bias and zero sink (1 warp per head, single pass for short ctx, or general loop over valid KV).
/// Supports arbitrary T tokens at pos = *pos0 + t, head_dim = 128, n_head = 24, n_head_kv = 8.
/// grid (n_head, T), block 128 (4 warps split KV keys, then reduce across warps in shared mem!).
__global__ void k_attn(const float* __restrict__ Q, const __half* __restrict__ kc, const __half* __restrict__ vc,
                       const float* __restrict__ rel_tab, const float* __restrict__ sink, float* __restrict__ out,
                       int n_head, int n_head_kv, int n_ext, int cap, int is_swa, int n_swa,
                       const int32_t* __restrict__ pos0, float kq_scale) {
    __shared__ float sq[128];
    __shared__ float s_m[4];
    __shared__ float s_l[4];
    __shared__ float s_acc[4][128];
    const int h = blockIdx.x, t = blockIdx.y;
    const int warp = threadIdx.x >> 5, lane = threadIdx.x & 31;
    const int hkv = h / (n_head / n_head_kv);
    const int kv_dim = n_head_kv * 128;
    const float* qv = Q + ((size_t) t * n_head + h) * 128;
    if (threadIdx.x < 128) sq[threadIdx.x] = qv[threadIdx.x];
    __syncthreads();

    const int pos_q = pos0[0] + t;
    const int j0 = is_swa ? max(0, pos_q - n_swa + 1) : 0;
    const int j1 = pos_q + 1;   // inclusive pos_q
    const float* rt = rel_tab + ((size_t) t * n_head + h) * n_ext;
    const float neg_sink = -sink[(size_t) t * n_head + h];

    float m = -1e30f;
    float l = 0.0f;
    float acc0 = 0.0f, acc1 = 0.0f, acc2 = 0.0f, acc3 = 0.0f;
    if (warp == 0) {
        m = 0.0f;
        l = 1.0f;   // zero sink = exp(0)
    }
    for (int j = j0 + warp; j < j1; j += 4) {
        const int slot = is_swa ? (j % cap) : j;
        const __half* kj = kc + (size_t) slot * kv_dim + hkv * 128;
        const __half* vj = vc + (size_t) slot * kv_dim + hkv * 128;
        float dot = 0.0f;
        dot += sq[lane] * __half2float(kj[lane]);
        dot += sq[32 + lane] * __half2float(kj[32 + lane]);
        dot += sq[64 + lane] * __half2float(kj[64 + lane]);
        dot += sq[96 + lane] * __half2float(kj[96 + lane]);
#pragma unroll
        for (int o = 16; o > 0; o >>= 1) dot += __shfl_xor_sync(0xffffffffu, dot, o);
        const int dist = pos_q - j;
        const float bias = dist < n_ext ? rt[dist] : neg_sink;
        const float sc = dot * kq_scale + bias;
        const float m_new = fmaxf(m, sc);
        const float alpha = l > 0.0f ? expf(m - m_new) : 0.0f;
        const float p = expf(sc - m_new);
        l = l * alpha + p;
        acc0 = acc0 * alpha + p * __half2float(vj[lane]);
        acc1 = acc1 * alpha + p * __half2float(vj[32 + lane]);
        acc2 = acc2 * alpha + p * __half2float(vj[64 + lane]);
        acc3 = acc3 * alpha + p * __half2float(vj[96 + lane]);
        m = m_new;
    }
    if (lane == 0) {
        s_m[warp] = m;
        s_l[warp] = l;
    }
    s_acc[warp][lane] = acc0;
    s_acc[warp][32 + lane] = acc1;
    s_acc[warp][64 + lane] = acc2;
    s_acc[warp][96 + lane] = acc3;
    __syncthreads();
    if (threadIdx.x < 128) {
        float mg = s_m[0];
        for (int w = 1; w < 4; ++w) if (s_l[w] > 0.0f && s_m[w] > mg) mg = s_m[w];
        float lg = 0.0f;
        float og = 0.0f;
        for (int w = 0; w < 4; ++w) {
            if (s_l[w] > 0.0f) {
                const float c = expf(s_m[w] - mg);
                lg += s_l[w] * c;
                og += s_acc[w][threadIdx.x] * c;
            }
        }
        out[((size_t) t * n_head + h) * 128 + threadIdx.x] = og / lg;
    }
}

__global__ void k_mul_sigmoid(float* __restrict__ att, const float* __restrict__ gate, int64_t n) {
    const int64_t i = (int64_t) blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) att[i] *= sigmf(gate[i]);
}

/// Order-preserving IEEE-754 float -> uint32 mapping for branchless uint64_t top-k reduction.
__device__ __forceinline__ uint32_t ink_float_to_ordered_u32(float f) {
    const uint32_t u = __float_as_uint(f);
    return (u & 0x80000000u) ? ~u : (u | 0x80000000u);
}

/// MoE router for Inkling2: sigmoid(logits) + exp_probs_b -> top-k -> normalize weights -> * scale.
/// Evolved branchless uint64_t key-pack (score, 127 - e) for zero-divergence warp reduction.
/// n_exp <= 128, top_k <= 16. 1 warp (32 threads) per token (grid T).
__global__ void k_route_inkling2(const float* __restrict__ logits, const float* __restrict__ bias,
                                 int32_t* __restrict__ ids, float* __restrict__ ewts,
                                 int n_exp, int top_k, float scale,
                                 uint32_t* __restrict__ d_hits = nullptr) {
    const int t = blockIdx.x, lane = threadIdx.x;
    uint64_t key[4];
    float p[4];
#pragma unroll
    for (int j = 0; j < 4; ++j) {
        const int e = j * 32 + lane;
        if (e < n_exp) {
            const float pr = sigmf(logits[(size_t) t * n_exp + e]);
            const float sc = pr + (bias ? bias[e] : 0.0f);
            p[j] = pr;
            key[j] = (uint64_t(ink_float_to_ordered_u32(sc)) << 8) | uint64_t(127 - e);
        } else {
            p[j] = 0.0f;
            key[j] = 0ull;
        }
    }
    float wlocal[16];
    float wsum = 0.0f;
    for (int k = 0; k < top_k; ++k) {
        uint64_t bk = key[0];
        float bp = p[0];
        if (key[1] > bk) { bk = key[1]; bp = p[1]; }
        if (key[2] > bk) { bk = key[2]; bp = p[2]; }
        if (key[3] > bk) { bk = key[3]; bp = p[3]; }
#pragma unroll
        for (int o = 16; o > 0; o >>= 1) {
            const uint32_t lo = __shfl_xor_sync(0xffffffffu, uint32_t(bk), o);
            const uint32_t hi = __shfl_xor_sync(0xffffffffu, uint32_t(bk >> 32), o);
            const uint64_t ok = (uint64_t(hi) << 32) | lo;
            const float op = __shfl_xor_sync(0xffffffffu, bp, o);
            if (ok > bk) {
                bk = ok;
                bp = op;
            }
        }
        const int be = 127 - int(bk & 0xFFu);
        if (lane == 0) {
            ids[(size_t) t * top_k + k] = be;
            if (d_hits) atomicAdd(&d_hits[be], 1u);
            wlocal[k] = bp;
            wsum += bp;
        }
        if ((be & 31) == lane) key[be >> 5] = 0ull;
    }
    if (lane == 0) {
        const float denom = fmaxf(wsum, 6.103515625e-5f);
        const float fac = scale / denom;
        for (int k = 0; k < top_k; ++k) ewts[(size_t) t * top_k + k] = wlocal[k] * fac;
    }
}

__global__ void k_combine_experts(const float* __restrict__ eparts, const float* __restrict__ ewts,
                                  float* __restrict__ out, int N, int top_k) {
    const int t = blockIdx.y;
    const size_t base = (size_t) t * top_k;
    if ((N & 3) == 0) {
        const int n4 = N >> 2;
        const int i4 = blockIdx.x * blockDim.x + threadIdx.x;
        if (i4 >= n4) return;
        float sx = 0.0f, sy = 0.0f, sz = 0.0f, sw = 0.0f;
        if (top_k == 8) {
#pragma unroll
            for (int k = 0; k < 8; ++k) {
                const float w = __ldg(ewts + base + k);
                const float4 v = __ldg(reinterpret_cast<const float4*>(eparts + (base + k) * N) + i4);
                sx = fmaf(w, v.x, sx); sy = fmaf(w, v.y, sy); sz = fmaf(w, v.z, sz); sw = fmaf(w, v.w, sw);
            }
        } else {
            for (int k = 0; k < top_k; ++k) {
                const float w = __ldg(ewts + base + k);
                const float4 v = __ldg(reinterpret_cast<const float4*>(eparts + (base + k) * N) + i4);
                sx = fmaf(w, v.x, sx); sy = fmaf(w, v.y, sy); sz = fmaf(w, v.z, sz); sw = fmaf(w, v.w, sw);
            }
        }
        reinterpret_cast<float4*>(out + (size_t) t * N)[i4] = make_float4(sx, sy, sz, sw);
        return;
    }
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= N) return;
    float s = 0.0f;
    for (int k = 0; k < top_k; ++k) s += ewts[base + k] * eparts[(base + k) * N + i];
    out[(size_t) t * N + i] = s;
}

__global__ void k_combine_experts_hybrid(const float* __restrict__ eparts, const float* __restrict__ ewts,
                                         const int32_t* __restrict__ ids, const int32_t* __restrict__ slot_of,
                                         const float* __restrict__ cpu_sum, float* __restrict__ out,
                                         int N, int top_k) {
    if ((N & 3) == 0) {
        const int n4 = N >> 2;
        const int i4 = blockIdx.x * blockDim.x + threadIdx.x;
        if (i4 >= n4) return;
        const float4 c = reinterpret_cast<const float4*>(cpu_sum)[i4];
        float sx = c.x, sy = c.y, sz = c.z, sw = c.w;
        for (int k = 0; k < top_k; ++k) {
            if (slot_of[ids[k]] >= 0) {
                const float w = ewts[k];
                const float4 v = reinterpret_cast<const float4*>(eparts + (size_t) k * N)[i4];
                sx += w * v.x; sy += w * v.y; sz += w * v.z; sw += w * v.w;
            }
        }
        reinterpret_cast<float4*>(out)[i4] = make_float4(sx, sy, sz, sw);
        return;
    }
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= N) return;
    float s = cpu_sum[i];
    for (int k = 0; k < top_k; ++k) {
        if (slot_of[ids[k]] >= 0) s += ewts[k] * eparts[(size_t) k * N + i];
    }
    out[i] = s;
}

__global__ void k_merge_cpu(float* __restrict__ eparts, const float* __restrict__ ecpu,
                            const int32_t* __restrict__ miss, int N) {
    const int e = blockIdx.x;
    if (!miss[e]) return;
    for (int i = threadIdx.x; i < N; i += blockDim.x) eparts[(size_t) e * N + i] = ecpu[(size_t) e * N + i];
}

__global__ void k_group_experts(const int32_t* __restrict__ ids, const int32_t* __restrict__ slot_of,
                                const uint8_t* __restrict__ blobs, size_t stride,
                                int n_ent, int n_exp, int n_slots,
                                unsigned long long* __restrict__ grp_ptr, int32_t* __restrict__ grp_start,
                                int32_t* __restrict__ n_groups, int32_t* __restrict__ ent_dst,
                                int32_t* __restrict__ ent_tok, int top_k,
                                uint32_t* __restrict__ d_hits = nullptr) {
    __shared__ int32_t cnt[128];
    __shared__ int32_t slot_id[128];
    __shared__ int32_t cur_off[128];
    __shared__ int32_t warp_ng[4];
    __shared__ int32_t warp_off[4];
    const int tid = threadIdx.x;
    const int warp = tid >> 5, lane = tid & 31;
    if (tid < n_exp) {
        cnt[tid] = 0;
        slot_id[tid] = n_slots > 0 ? slot_of[tid] : -1;
    }
    __syncthreads();
    for (int e = tid; e < n_ent; e += blockDim.x) {
        const int x = ids[e];
        if (slot_id[x] >= 0) atomicAdd(&cnt[x], 1);
    }
    __syncthreads();
    const int c = (tid < n_exp) ? cnt[tid] : 0;
    if (d_hits && c > 0) {
        d_hits[tid] += (uint32_t) c;
    }
    const int active = (c > 0) ? 1 : 0;
    const uint32_t mask = __ballot_sync(0xffffffffu, active);
    const int w_ng = __popc(mask & ((1u << lane) - 1u));
    int w_c = c;
#pragma unroll
    for (int d = 1; d < 32; d <<= 1) {
        const int prev = __shfl_up_sync(0xffffffffu, w_c, d);
        if (lane >= d) w_c += prev;
    }
    if (lane == 31) {
        warp_ng[warp] = __popc(mask);
        warp_off[warp] = w_c;
    }
    __syncthreads();
    if (warp == 0 && lane == 0) {
        const int ng0 = warp_ng[0], ng1 = warp_ng[1], ng2 = warp_ng[2], ng3 = warp_ng[3];
        warp_ng[0] = 0; warp_ng[1] = ng0; warp_ng[2] = ng0 + ng1; warp_ng[3] = ng0 + ng1 + ng2;
        const int o0 = warp_off[0], o1 = warp_off[1], o2 = warp_off[2], o3 = warp_off[3];
        warp_off[0] = 0; warp_off[1] = o0; warp_off[2] = o0 + o1; warp_off[3] = o0 + o1 + o2;
        const int tot_ng = ng0 + ng1 + ng2 + ng3;
        grp_start[tot_ng] = o0 + o1 + o2 + o3;
        *n_groups = tot_ng;
    }
    __syncthreads();
    if (active) {
        const int ng = warp_ng[warp] + w_ng;
        const int off = warp_off[warp] + (w_c - c);
        grp_ptr[ng] = (unsigned long long) (blobs + (size_t) slot_id[tid] * stride);
        grp_start[ng] = off;
        cur_off[tid] = off;
    }
    __syncthreads();
    for (int e = tid; e < n_ent; e += blockDim.x) {
        const int x = ids[e];
        if (slot_id[x] >= 0) {
            const int pos = atomicAdd(&cur_off[x], 1);
            ent_dst[pos] = e;
            ent_tok[pos] = e / top_k;
        }
    }
}

/// Group entries for PCIe-staged misses in expert ID range [e0, e1): matches slot_of[x] < 0 && x >= e0 && x < e1,
/// resolving each weight blob at `stage_blobs + (x - e0) * stride`.
__global__ void k_group_experts_miss_range(const int32_t* __restrict__ ids, const int32_t* __restrict__ slot_of,
                                           const uint8_t* __restrict__ stage_blobs, size_t stride,
                                           int n_ent, int n_exp, int n_slots, int e0, int e1,
                                           unsigned long long* __restrict__ grp_ptr, int32_t* __restrict__ grp_start,
                                           int32_t* __restrict__ n_groups, int32_t* __restrict__ ent_dst,
                                           int32_t* __restrict__ ent_tok, int top_k) {
    __shared__ int32_t cnt[128];
    __shared__ int8_t active[128];
    __shared__ int32_t cur_off[128];
    if (threadIdx.x < n_exp) {
        cnt[threadIdx.x] = 0;
        const int x = (int) threadIdx.x;
        const bool ok = (x >= e0 && x < e1) && (n_slots <= 0 || slot_of[x] < 0);
        active[x] = ok ? 1 : 0;
    }
    __syncthreads();
    for (int e = threadIdx.x; e < n_ent; e += blockDim.x) {
        const int x = ids[e];
        if (active[x]) atomicAdd(&cnt[x], 1);
    }
    __syncthreads();
    if (threadIdx.x == 0) {
        int ng = 0;
        int off = 0;
        for (int x = e0; x < e1; ++x) {
            if (cnt[x] > 0) {
                grp_ptr[ng] = (unsigned long long) (stage_blobs + (size_t) (x - e0) * stride);
                grp_start[ng] = off;
                cur_off[x] = off;
                off += cnt[x];
                ++ng;
            }
        }
        grp_start[ng] = off;
        *n_groups = ng;
    }
    __syncthreads();
    for (int e = threadIdx.x; e < n_ent; e += blockDim.x) {
        const int x = ids[e];
        if (active[x]) {
            const int pos = atomicAdd(&cur_off[x], 1);
            ent_dst[pos] = e;
            ent_tok[pos] = e / top_k;
        }
    }
}

/// Compact slot mapping inside `pp_stage[b]` for missing experts in one staging batch (-1 = not in batch).
struct PpStageMap {
    int16_t pos[128];
};

/// Group entries for compact PCIe-staged misses where `map.pos[x] >= 0` gives the compact slot inside `stage_blobs`.
__global__ void k_group_experts_miss_map(const int32_t* __restrict__ ids,
                                         const uint8_t* __restrict__ stage_blobs, size_t stride,
                                         int n_ent, int n_exp, PpStageMap map,
                                         unsigned long long* __restrict__ grp_ptr, int32_t* __restrict__ grp_start,
                                         int32_t* __restrict__ n_groups, int32_t* __restrict__ ent_dst,
                                         int32_t* __restrict__ ent_tok, int top_k) {
    __shared__ int32_t cnt[128];
    __shared__ int16_t spos[128];
    __shared__ int32_t cur_off[128];
    if (threadIdx.x < n_exp) {
        cnt[threadIdx.x] = 0;
        spos[threadIdx.x] = map.pos[threadIdx.x];
    }
    __syncthreads();
    for (int e = threadIdx.x; e < n_ent; e += blockDim.x) {
        const int x = ids[e];
        if (spos[x] >= 0) atomicAdd(&cnt[x], 1);
    }
    __syncthreads();
    if (threadIdx.x == 0) {
        int ng = 0;
        int off = 0;
        for (int x = 0; x < n_exp; ++x) {
            if (cnt[x] > 0) {
                grp_ptr[ng] = (unsigned long long) (stage_blobs + (size_t) spos[x] * stride);
                grp_start[ng] = off;
                cur_off[x] = off;
                off += cnt[x];
                ++ng;
            }
        }
        grp_start[ng] = off;
        *n_groups = ng;
    }
    __syncthreads();
    for (int e = threadIdx.x; e < n_ent; e += blockDim.x) {
        const int x = ids[e];
        if (spos[x] >= 0) {
            const int pos = atomicAdd(&cur_off[x], 1);
            ent_dst[pos] = e;
            ent_tok[pos] = e / top_k;
        }
    }
}

__global__ void k_softcap(float* __restrict__ logits, int64_t n, float cap) {
    const int64_t i = (int64_t) blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n && cap > 0.0f) logits[i] = cap * tanhf(logits[i] / cap);
}

__global__ void k_gemv_f32(const float* __restrict__ W, const float* __restrict__ x, float* __restrict__ y,
                           int n_in, int n_out) {
    const int warp = threadIdx.x >> 5, lane = threadIdx.x & 31;
    const int o = blockIdx.x * 8 + warp;
    if (o >= n_out) return;
    const float4* wr = (const float4*) (W + (size_t) o * n_in);
    const float4* xr = (const float4*) x;
    float acc = 0.0f;
    for (int i = lane; i < n_in / 4; i += 32) {
        const float4 a = wr[i], b = xr[i];
        acc += a.x * b.x + a.y * b.y + a.z * b.z + a.w * b.w;
    }
#pragma unroll
    for (int m = 16; m > 0; m >>= 1) acc += __shfl_xor_sync(0xffffffffu, acc, m);
    if (lane == 0) y[o] = acc;
}

__global__ void k_argmax(const float* __restrict__ logits, int32_t* __restrict__ pick, int V) {
    __shared__ float sv[256];
    __shared__ int32_t si[256];
    const float* row = logits + (size_t) blockIdx.x * V;
    float mv = -1e30f;
    int32_t mi = 0;
    for (int i = threadIdx.x; i < V; i += blockDim.x) {
        const float v = row[i];
        if (v > mv) { mv = v; mi = i; }
    }
    sv[threadIdx.x] = mv;
    si[threadIdx.x] = mi;
    __syncthreads();
    for (int s = 128; s > 0; s >>= 1) {
        if (threadIdx.x < s) {
            if (sv[threadIdx.x + s] > sv[threadIdx.x] ||
                (sv[threadIdx.x + s] == sv[threadIdx.x] && si[threadIdx.x + s] < si[threadIdx.x])) {
                sv[threadIdx.x] = sv[threadIdx.x + s];
                si[threadIdx.x] = si[threadIdx.x + s];
            }
        }
        __syncthreads();
    }
    if (threadIdx.x == 0) pick[blockIdx.x] = si[0];
}

struct SwaLayerPtrs {
    int n_swa_layers = 0;
    int8_t* kq[48] = {};
    __half* ks[48] = {};
    int8_t* vq[48] = {};
    __half* vs[48] = {};
    __half* kc[48] = {};
    __half* vc[48] = {};
};

/// Save (`restore == 0`) or restore (`restore == 1`) `n_slots` SWA KV ring entries (`pos_base + si`) across all SWA layers.
/// Grid: (n_slots, n_swa_layers), Block: 256. Each slot backup holds `[kq(kv_dim), vq(kv_dim), ks(nhkv), vs(nhkv)]` or `[kc, vc]`.
__global__ void k_swa_q8_bak(SwaLayerPtrs swa, uint8_t* __restrict__ bak, int kv_dim, int n_head_kv, int cap,
                             int pos_base, int first_si, int n_slots, int restore) {
    const int si = first_si + blockIdx.x;
    const int s = blockIdx.y;
    if (s >= swa.n_swa_layers) return;
    const int p = pos_base + si;
    if (restore && p < cap) return;   // slot was empty prior to overwrite
    const int slot = p % cap;
    const size_t slot_bytes = (size_t) kv_dim * 2 + (size_t) n_head_kv * 2 * sizeof(__half);
    uint8_t* b = bak + ((size_t) si * swa.n_swa_layers + s) * slot_bytes;
    int8_t* bkq = (int8_t*) b;
    int8_t* bvq = bkq + kv_dim;
    __half* bks = (__half*) (bvq + kv_dim);
    __half* bvs = bks + n_head_kv;
    int8_t* dkq = swa.kq[s] + (size_t) slot * kv_dim;
    int8_t* dvq = swa.vq[s] + (size_t) slot * kv_dim;
    __half* dks = swa.ks[s] + (size_t) slot * n_head_kv;
    __half* dvs = swa.vs[s] + (size_t) slot * n_head_kv;
    for (int i = threadIdx.x; i < kv_dim; i += blockDim.x) {
        if (restore) { dkq[i] = bkq[i]; dvq[i] = bvq[i]; }
        else { bkq[i] = dkq[i]; bvq[i] = dvq[i]; }
    }
    if (threadIdx.x < n_head_kv) {
        if (restore) { dks[threadIdx.x] = bks[threadIdx.x]; dvs[threadIdx.x] = bvs[threadIdx.x]; }
        else { bks[threadIdx.x] = dks[threadIdx.x]; bvs[threadIdx.x] = dvs[threadIdx.x]; }
    }
}

__global__ void k_swa_f16_bak(SwaLayerPtrs swa, uint8_t* __restrict__ bak, int kv_dim, int cap,
                              int pos_base, int first_si, int n_slots, int restore) {
    const int si = first_si + blockIdx.x;
    const int s = blockIdx.y;
    if (s >= swa.n_swa_layers) return;
    const int p = pos_base + si;
    if (restore && p < cap) return;
    const int slot = p % cap;
    const size_t slot_bytes = (size_t) kv_dim * 2 * sizeof(__half);
    __half* b = (__half*) (bak + ((size_t) si * swa.n_swa_layers + s) * slot_bytes);
    __half* bkc = b;
    __half* bvc = b + kv_dim;
    __half* dkc = swa.kc[s] + (size_t) slot * kv_dim;
    __half* dvc = swa.vc[s] + (size_t) slot * kv_dim;
    for (int i = threadIdx.x; i < kv_dim; i += blockDim.x) {
        if (restore) { dkc[i] = bkc[i]; dvc[i] = bvc[i]; }
        else { bkc[i] = dkc[i]; bvc[i] = dvc[i]; }
    }
}

/// Elastic Looped Transformer (ELT, arXiv:2604.09168) loop-exit convergence metric:
/// computes delta = ||cur - prev||_2 / (||cur||_2 + 1e-6) in a single 256-thread CTA using float4 + warp shuffles,
/// and updates prev <- cur in-place so zero extra memcpy is needed between passes.
__global__ void __launch_bounds__(256) k_elt_rel_delta(const float* __restrict__ cur,
                                                       float* __restrict__ prev,
                                                       float* __restrict__ out_delta,
                                                       int N) {
    __shared__ float s_diff[8];
    __shared__ float s_norm[8];
    const int tid = threadIdx.x;
    const int n4 = N >> 2;
    const float4* c4 = reinterpret_cast<const float4*>(cur);
    float4* p4 = reinterpret_cast<float4*>(prev);
    float d_acc = 0.0f;
    float n_acc = 0.0f;
    for (int i = tid; i < n4; i += 256) {
        const float4 cv = __ldg(c4 + i);
        const float4 pv = p4[i];
        p4[i] = cv;
        const float dx = cv.x - pv.x, dy = cv.y - pv.y, dz = cv.z - pv.z, dw = cv.w - pv.w;
        d_acc = fmaf(dx, dx, fmaf(dy, dy, fmaf(dz, dz, fmaf(dw, dw, d_acc))));
        n_acc = fmaf(cv.x, cv.x, fmaf(cv.y, cv.y, fmaf(cv.z, cv.z, fmaf(cv.w, cv.w, n_acc))));
    }
#pragma unroll
    for (int off = 16; off > 0; off >>= 1) {
        d_acc += __shfl_xor_sync(0xffffffffu, d_acc, off);
        n_acc += __shfl_xor_sync(0xffffffffu, n_acc, off);
    }
    if ((tid & 31) == 0) {
        s_diff[tid >> 5] = d_acc;
        s_norm[tid >> 5] = n_acc;
    }
    __syncthreads();
    if (tid == 0) {
        float dt = 0.0f, nt = 0.0f;
#pragma unroll
        for (int w = 0; w < 8; ++w) { dt += s_diff[w]; nt += s_norm[w]; }
        out_delta[0] = sqrtf(dt) / (sqrtf(nt) + 1e-6f);
    }
}

}  // namespace strata::inkling2
