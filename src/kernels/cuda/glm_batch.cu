// src/kernels/cuda/glm_batch.cu - see include/strata/kernels/glm_batch.hpp.
//
// Every kernel here is the batched twin of a one-token kernel in glm_fast.cu with the same per-token arithmetic
// (same operation order where the order is observable), so a prompt read in chunks leaves the state the token path
// would have left up to float rounding - the projections around them (FP16 tensor-core GEMMs instead of q8_1 dot
// products) are where the two paths really differ, as llama.cpp's batched and one-token paths do.
#include "strata/kernels/glm_batch.hpp"
#include "dsa_topk.cuh"

#include <cuda_fp16.h>
#include <cuda_runtime.h>

#include <atomic>
#include <cmath>
#include <cstdio>

namespace strata::kernels::glmb {
namespace {

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
// block-wide sum; every thread gets the total.  sh: >= 32 floats.  blockDim % 32 == 0.
__device__ __forceinline__ float block_sum(float v, float* sh) {
    const int lane = threadIdx.x & 31, w = threadIdx.x >> 5, nw = blockDim.x >> 5;
    v = warp_sum(v);
    __syncthreads();
    if (lane == 0) sh[w] = v;
    __syncthreads();
    float t = 0.0f;
    for (int i = 0; i < nw; ++i) t += sh[i];
    return t;
}
__device__ __forceinline__ float dsigmoid(float x) { return 1.0f / (1.0f + __expf(-x)); }

std::atomic<int> g_errors{0};
void check(const char* what) {
    const cudaError_t e = cudaGetLastError();
    if (e != cudaSuccess) {
        std::fprintf(stderr, "glm_batch %s: %s\n", what, cudaGetErrorString(e));
        g_errors.fetch_add(1, std::memory_order_relaxed);
    }
}
unsigned nblk(int64_t n, int b = 256) { return (unsigned) ((n + b - 1) / b); }

// ---------------------------------------------------------------- conversions
__global__ void bf16_to_f32_kernel(const uint16_t* __restrict__ src, float* __restrict__ dst, int64_t n) {
    const int64_t i = (int64_t) blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) dst[i] = __uint_as_float((uint32_t) src[i] << 16);
}
__global__ void f32_to_f16_kernel(const float* __restrict__ src, __half* __restrict__ dst, int64_t n) {
    const int64_t i = (int64_t) blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) dst[i] = __float2half(src[i]);
}
__global__ void embed_rows_kernel(const float* __restrict__ emb, float* __restrict__ R, int T, int E) {
    const int64_t i = (int64_t) blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= (int64_t) T * 4 * E) return;
    const int64_t t = i / (4 * (int64_t) E);
    const int e = (int) (i % E);
    R[i] = emb[t * E + e];
}

// ---------------------------------------------------------------- mHC
__global__ void __launch_bounds__(256) hc_update_kernel(const float* __restrict__ block_out, float* __restrict__ R,
                                                        const float* __restrict__ post, const float* __restrict__ comb,
                                                        float* __restrict__ ss, int E) {
    const int t = blockIdx.x;
    float* Rt = R + (size_t) t * 4 * E;
    __shared__ float sred[32];
    __shared__ float sp[4], sc[16];
    if (block_out != nullptr) {
        if (threadIdx.x < 4) sp[threadIdx.x] = post[t * 4 + threadIdx.x];
        if (threadIdx.x < 16) sc[threadIdx.x] = comb[t * 16 + threadIdx.x];
    }
    __syncthreads();
    float acc = 0.0f;
    for (int e = threadIdx.x; e < E; e += blockDim.x) {
        float r[4] = {Rt[e], Rt[E + e], Rt[2 * E + e], Rt[3 * E + e]};
        if (block_out != nullptr) {
            const float b = block_out[(size_t) t * E + e];
            float n[4];
#pragma unroll
            for (int d = 0; d < 4; ++d) {
                float v = b * sp[d];
#pragma unroll
                for (int s = 0; s < 4; ++s) v += sc[d * 4 + s] * r[s];
                n[d] = v;
            }
#pragma unroll
            for (int d = 0; d < 4; ++d) {
                r[d] = n[d];
                Rt[d * E + e] = n[d];
            }
        }
        acc += r[0] * r[0] + r[1] * r[1] + r[2] * r[2] + r[3] * r[3];
    }
    if (ss == nullptr) return;
    acc = block_sum(acc, sred);
    if (threadIdx.x == 0) ss[t] = acc;
}

__global__ void __launch_bounds__(256) hc_finish_kernel(const HcFinishArgs a) {
    const int hc = 4;
    const int t = blockIdx.x, tid = threadIdx.x, lane = tid & 31, warp = tid >> 5;
    const int E = a.n_embd, hc_dim = hc * E;
    __shared__ float smix[25], s_pre[4], sred[32];
    if (tid < 24) smix[tid] = a.mix[(size_t) t * 24 + tid];
    if (tid == 24) smix[24] = a.ss[t];
    __syncthreads();
    const float inv = rsqrtf(smix[24] / (float) hc_dim + a.norm_eps);
    if (warp == 0) {
        if (lane < hc) {
            const float pre_v = dsigmoid(smix[lane] * inv * a.w_scale[0] + a.w_base[lane]) + a.hc_eps;
            s_pre[lane] = pre_v;
            a.pre[t * 4 + lane] = pre_v;
            a.post[t * 4 + lane] = 2.0f * dsigmoid(smix[hc + lane] * inv * a.w_scale[1] + a.w_base[hc + lane]);
        }
        const int l = lane & 15, d = l >> 2, s = l & 3;
        float c = smix[2 * hc + d + hc * s] * inv * a.w_scale[2] + a.w_base[2 * hc + d + hc * s];
        float mx = fmaxf(c, __shfl_xor_sync(0xffffffffu, c, 4));
        mx = fmaxf(mx, __shfl_xor_sync(0xffffffffu, mx, 8));
        const float ex = __expf(c - mx);
        float den = ex + __shfl_xor_sync(0xffffffffu, ex, 4);
        den += __shfl_xor_sync(0xffffffffu, den, 8);
        c = ex / den + a.hc_eps;
        {
            float sm = c + __shfl_xor_sync(0xffffffffu, c, 1);
            sm += __shfl_xor_sync(0xffffffffu, sm, 2);
            c = c / (a.hc_eps + sm);
        }
        for (int it = 1; it < a.iters; ++it) {
            float sm = c + __shfl_xor_sync(0xffffffffu, c, 4);
            sm += __shfl_xor_sync(0xffffffffu, sm, 8);
            c = c / (a.hc_eps + sm);
            sm = c + __shfl_xor_sync(0xffffffffu, c, 1);
            sm += __shfl_xor_sync(0xffffffffu, sm, 2);
            c = c / (a.hc_eps + sm);
        }
        if (lane < 16) a.comb[t * 16 + d * hc + s] = c;
    }
    __syncthreads();
    const float* Rt = a.R + (size_t) t * hc * E;
    float mixed[16];   // E <= 16 * blockDim.x
    float ss2 = 0.0f;
    const int per = E / (int) blockDim.x;
#pragma unroll
    for (int i = 0; i < 16; ++i) {
        if (i >= per) break;
        const int e = tid + i * (int) blockDim.x;
        float acc = 0.0f;
#pragma unroll
        for (int s = 0; s < hc; ++s) acc += s_pre[s] * Rt[s * E + e];
        mixed[i] = acc;
        ss2 += acc * acc;
    }
    ss2 = block_sum(ss2, sred);
    const float inv2 = rsqrtf(ss2 / (float) E + a.norm_eps);
    __half* x16 = (__half*) a.x16;
#pragma unroll
    for (int i = 0; i < 16; ++i) {
        if (i >= per) break;
        const int e = tid + i * (int) blockDim.x;
        const float xv = mixed[i] * inv2 * a.norm_w[e];
        a.x[(size_t) t * E + e] = xv;
        x16[(size_t) t * E + e] = __float2half(xv);
    }
}

// ---------------------------------------------------------------- KDA
struct Ptr3 {
    const float* p[3];
};
struct WPtr3 {
    float* p[3];
};

__global__ void kda_conv_kernel(Ptr3 proj, Ptr3 w, const float* __restrict__ conv_state, WPtr3 out, int T, int DI,
                                int dconv) {
    const int64_t i = (int64_t) blockIdx.x * blockDim.x + threadIdx.x;
    const int64_t per = (int64_t) T * DI;
    if (i >= 3 * per) return;
    const int role = (int) (i / per);
    const int64_t rem = i - role * per;
    const int t = (int) (rem / DI), c = (int) (rem - (int64_t) t * DI);
    const int hist = dconv - 1;
    const float* in = proj.p[role];
    const float* st = conv_state + (size_t) role * DI * hist + (size_t) hist * c;
    const float* wc = w.p[role] + (size_t) dconv * c;
    float acc = 0.0f;
    for (int tap = 0; tap < hist; ++tap) {
        const int tt = t - hist + tap;
        acc += wc[tap] * (tt >= 0 ? in[(size_t) tt * DI + c] : st[hist + tt]);
    }
    acc += wc[hist] * in[(size_t) t * DI + c];
    out.p[role][(size_t) t * DI + c] = acc / (1.0f + __expf(-acc));
}

__global__ void kda_conv_state_kernel(Ptr3 proj, float* __restrict__ conv_state, int T, int DI, int dconv) {
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= 3 * DI) return;
    const int role = i / DI, c = i - role * DI;
    const int hist = dconv - 1;
    float* st = conv_state + (size_t) role * DI * hist + (size_t) hist * c;
    float old[8];
    for (int j = 0; j < hist && j < 8; ++j) old[j] = st[j];
    for (int j = 0; j < hist && j < 8; ++j) {
        const int tt = T - hist + j;
        st[j] = tt >= 0 ? proj.p[role][(size_t) tt * DI + c] : old[hist + tt];
    }
}

__global__ void __launch_bounds__(256) kda_rec_kernel(const float* __restrict__ q, const float* __restrict__ k,
                                                      const float* __restrict__ v, const float* __restrict__ g1_raw,
                                                      const float* __restrict__ dt_bias, const float* __restrict__ ssm_a,
                                                      float lb, const float* __restrict__ beta_raw,
                                                      float* __restrict__ state, const float* __restrict__ g2_raw,
                                                      const float* __restrict__ norm_w, float eps, int n_head, int T,
                                                      __half* __restrict__ out) {
    constexpr int HD = 128;
    const int h = blockIdx.x, t = threadIdx.x, j = t & (HD - 1), half = t >> 7;
    const int DI = n_head * HD;
    __shared__ float sq[HD], sk[HD], sdec[HD], sdel[HD];
    __shared__ float sacc[2][HD];
    __shared__ float sred[32];
    float* Sh = state + (size_t) h * HD * HD;
    float reg[64];
#pragma unroll
    for (int ii = 0; ii < 64; ++ii) reg[ii] = Sh[(half * 64 + ii) * HD + j];
    const float a_h = ssm_a[h];
    const float nw = norm_w[j];
    const float dtb = t < HD ? dt_bias[h * HD + t] : 0.0f;
    for (int tok = 0; tok < T; ++tok) {
        const size_t base = (size_t) tok * DI + (size_t) h * HD;
        float qv = 0.0f, kv = 0.0f, gr = 0.0f;
        if (t < HD) {
            qv = q[base + t];
            kv = k[base + t];
            gr = g1_raw[base + t];
        }
        const float b = 1.0f / (1.0f + expf(-beta_raw[(size_t) tok * n_head + h]));
        const float vj = half == 0 ? v[base + j] : 0.0f;
        const float g2 = half == 0 ? g2_raw[base + j] : 0.0f;
        const float ssq = block_sum(qv * qv, sred);
        const float ssk = block_sum(kv * kv, sred);
        if (t < HD) {
            sq[t] = qv * rsqrtf(ssq + 1e-6f);
            sk[t] = kv * rsqrtf(ssk + 1e-6f);
            sdec[t] = __expf(lb * dsigmoid(-(gr + dtb) * a_h));
        }
        __syncthreads();
        float A = 0.0f;
#pragma unroll
        for (int ii = 0; ii < 64; ++ii) {
            const int i = half * 64 + ii;
            const float s = reg[ii] * sdec[i];
            reg[ii] = s;
            A += s * sk[i];
        }
        sacc[half][j] = A;
        __syncthreads();
        if (half == 0) sdel[j] = b * (vj - (sacc[0][j] + sacc[1][j]));
        __syncthreads();
        const float dl = sdel[j];
        float O = 0.0f;
#pragma unroll
        for (int ii = 0; ii < 64; ++ii) {
            const int i = half * 64 + ii;
            const float s2 = reg[ii] + sk[i] * dl;
            reg[ii] = s2;
            O += s2 * sq[i];
        }
        __syncthreads();
        sacc[half][j] = O;
        __syncthreads();
        const float o = (sacc[0][j] + sacc[1][j]) * rsqrtf((float) HD);
        float ss = half == 0 ? o * o : 0.0f;
        ss = block_sum(ss, sred);
        const float inv = rsqrtf(ss / (float) HD + eps);
        if (half == 0) out[base + j] = __float2half(o * inv * nw * dsigmoid(g2));
    }
#pragma unroll
    for (int ii = 0; ii < 64; ++ii) Sh[(half * 64 + ii) * HD + j] = reg[ii];
}

// ---------------------------------------------------------------- DSA
__global__ void __launch_bounds__(512) dsa_prep_kernel(const DsaPrepArgs a) {
    __shared__ float sred[32];
    const int t = blockIdx.x, role = blockIdx.y, tid = threadIdx.x;
    const int p = a.p0 + t;
    if (role == 0) {
        if (a.qr_raw == nullptr) return;   // the NextN block's cache fill needs no queries
        float vals[4];
        float ss = 0.0f;
        int n = 0;
        for (int e = tid; e < a.q_lora && n < 4; e += 512, ++n) {
            vals[n] = a.qr_raw[(size_t) t * a.q_lora + e];
            ss += vals[n] * vals[n];
        }
        ss = block_sum(ss, sred);
        const float inv = rsqrtf(ss / (float) a.q_lora + a.eps);
        n = 0;
        __half* q16 = (__half*) a.qr16;
        for (int e = tid; e < a.q_lora && n < 4; e += 512, ++n) {
            const float y = vals[n] * inv * a.q_a_norm[e];
            a.qr[(size_t) t * a.q_lora + e] = y;
            q16[(size_t) t * a.q_lora + e] = __float2half(y);
        }
        return;
    }
    if (role == 1) {
        const float v = tid < a.kv_lora ? a.kv_raw[(size_t) t * a.kv_lora + tid] : 0.0f;
        const float ss = block_sum(v * v, sred);
        const float inv = rsqrtf(ss / (float) a.kv_lora + a.eps);
        if (tid < a.kv_lora) a.lat[(size_t) a.kv_lora * p + tid] = v * inv * a.kv_norm[tid];
        return;
    }
    const int K = a.idx_key;
    const float v = tid < K ? a.ik_raw[(size_t) t * K + tid] : 0.0f;
    const float s1 = block_sum(v, sred);
    const float s2 = block_sum(v * v, sred);
    const float mu = s1 / (float) K;
    const float inv = rsqrtf(s2 / (float) K - mu * mu + a.eps);
    if (tid < K) {
        a.ik_cache[(size_t) K * p + tid] = (v - mu) * inv * a.k_norm_w[tid] + a.k_norm_b[tid];
        a.ig_cache[(size_t) K * p + tid] = a.ig_raw[(size_t) t * K + tid];
    }
}

__global__ void dsa_pool_kernel(const float* __restrict__ ik, const float* __restrict__ ig, const float* __restrict__ ape,
                                float* __restrict__ pooled, int K, int kpool, int pool0) {
    const int pi = pool0 + blockIdx.x, tid = threadIdx.x;
    if (tid >= K) return;
    float lg[16];
    float mx = -INFINITY;
    for (int m = 0; m < kpool && m < 16; ++m) {
        lg[m] = ig[(size_t) tid + (size_t) K * (pi * kpool + m)] + ape[tid + K * m];
        mx = fmaxf(mx, lg[m]);
    }
    float den = 0.0f;
    for (int m = 0; m < kpool && m < 16; ++m) {
        lg[m] = expf(lg[m] - mx);
        den += lg[m];
    }
    float acc = 0.0f;
    for (int m = 0; m < kpool && m < 16; ++m) acc += (lg[m] / den) * ik[(size_t) tid + (size_t) K * (pi * kpool + m)];
    pooled[(size_t) tid + (size_t) K * pi] = acc;
}

constexpr int SCORE_POOLS = 64;   // pools per block (8 warps x 8)
__global__ void __launch_bounds__(256) dsa_score_kernel(const float* __restrict__ iq, const float* __restrict__ pooled,
                                                        const float* __restrict__ iw, int key_dim, int idx_heads,
                                                        int p0, int kpool, float* __restrict__ score, int score_ld) {
    extern __shared__ float s_iq[];   // idx_heads * key_dim, then idx_heads weights
    const int t = blockIdx.y;
    const int n_vis = (p0 + t + 1) / kpool;
    const int pbeg = blockIdx.x * SCORE_POOLS;
    if (pbeg >= n_vis) return;
    float* s_iw = s_iq + idx_heads * key_dim;
    for (int i = threadIdx.x; i < idx_heads * key_dim; i += blockDim.x) s_iq[i] = iq[(size_t) t * idx_heads * key_dim + i];
    for (int i = threadIdx.x; i < idx_heads; i += blockDim.x) s_iw[i] = iw[(size_t) t * idx_heads + i];
    __syncthreads();
    const int lane = threadIdx.x & 31, warp = threadIdx.x >> 5;
    const int pend = min(n_vis, pbeg + SCORE_POOLS);
    for (int p = pbeg + warp; p < pend; p += 8) {
        const float* pk = pooled + (size_t) key_dim * p;
        float acc = 0.0f;
        for (int h = 0; h < idx_heads; ++h) {
            float dot = 0.0f;
            for (int e = lane; e < key_dim; e += 32) dot += s_iq[h * key_dim + e] * pk[e];
            dot = warp_sum(dot);
            acc += fmaxf(dot, 0.0f) * s_iw[h];
        }
        if (lane == 0) score[(size_t) t * score_ld + p] = acc;
    }
}

__global__ void __launch_bounds__(1024) dsa_select_kernel(const float* __restrict__ score, int score_ld, int p0,
                                                          int kpool, int top_pools_max, int tail, int n_sel_max,
                                                          int* __restrict__ cells_all, int* __restrict__ n_sel_out) {
    const int t = blockIdx.x;
    const int pos = p0 + t;
    const int n_vis = (pos + 1) / kpool;
    const int top = min(top_pools_max, n_vis);
    const int n_sel = kpool * top + (tail ? kpool - 1 : 0);
    int* cells = cells_all + (size_t) t * n_sel_max;
    for (int i = threadIdx.x; i < n_sel; i += blockDim.x) cells[i] = -1;
    if (threadIdx.x == 0) n_sel_out[t] = n_sel;
    __syncthreads();
    if (n_vis <= top_pools_max) {
        for (int i = threadIdx.x; i < n_vis * kpool; i += blockDim.x) cells[i] = i;
    } else {
        // the top pools in rank order, O(n_vis) - src/kernels/cuda/dsa_topk.cuh
        strata::kernels::dsa::select_top_pools(score + (size_t) t * score_ld, n_vis, top, kpool, cells);
    }
    __syncthreads();
    if (threadIdx.x == 0 && tail) {
        for (int m = 0; m < kpool - 1; ++m) {
            const int cell = n_vis * kpool + m;
            if (cell <= pos) cells[top * kpool + m] = cell;
        }
    }
}

// MLA attention for a chunk: block = (token, 16 heads).  The token's cells are walked in chunks of 32 latent rows
// (loaded into shared memory once for the 16 heads), with an online softmax per head; thread c owns the context
// columns c and c + 256 of all 16 heads.
constexpr int MB_HG = 16, MB_CH = 32;
__global__ void __launch_bounds__(256) mla_attn_kernel(const float* __restrict__ q_abs, const float* __restrict__ lat,
                                                       const int* __restrict__ cells_all, const int* __restrict__ n_sel_arr,
                                                       int n_sel_max, int n_head, float scale, float* __restrict__ ctx) {
    constexpr int KV = 512;
    extern __shared__ float sm[];
    float* sL = sm;                  // MB_CH x KV
    float* sP = sL + MB_CH * KV;     // MB_HG x MB_CH
    __shared__ float s_m[MB_HG], s_l[MB_HG], s_scale[MB_HG];
    __shared__ int s_cell[MB_CH];
    const int t = blockIdx.x, h0 = blockIdx.y * MB_HG, tid = threadIdx.x, lane = tid & 31, warp = tid >> 5;
    const int ns = n_sel_arr[t];
    const int* cl = cells_all + (size_t) t * n_sel_max;
    float q0[16], q1[16];
    {
        const float* qa = q_abs + ((size_t) t * n_head + h0 + 2 * warp) * KV;
#pragma unroll
        for (int j = 0; j < 16; ++j) {
            q0[j] = qa[lane + 32 * j];
            q1[j] = qa[KV + lane + 32 * j];
        }
    }
    float acc0[MB_HG], acc1[MB_HG];
#pragma unroll
    for (int hh = 0; hh < MB_HG; ++hh) acc0[hh] = acc1[hh] = 0.0f;
    if (tid < MB_HG) {
        s_m[tid] = -INFINITY;
        s_l[tid] = 0.0f;
    }
    for (int s0 = 0; s0 < ns; s0 += MB_CH) {
        __syncthreads();
        if (tid < MB_CH) s_cell[tid] = s0 + tid < ns ? cl[s0 + tid] : -1;
        __syncthreads();
        for (int i = tid; i < MB_CH * KV / 4; i += blockDim.x) {
            const int s = i / (KV / 4), c4 = i - s * (KV / 4);
            const int cell = s_cell[s];
            float4 v = make_float4(0.f, 0.f, 0.f, 0.f);
            if (cell >= 0) v = ((const float4*) (lat + (size_t) KV * cell))[c4];
            ((float4*) sL)[i] = v;
        }
        __syncthreads();
        for (int s = 0; s < MB_CH; ++s) {
            float a0 = 0.0f, a1 = 0.0f;
#pragma unroll
            for (int j = 0; j < 16; ++j) {
                const float l = sL[s * KV + lane + 32 * j];
                a0 += q0[j] * l;
                a1 += q1[j] * l;
            }
            a0 = warp_sum(a0);
            a1 = warp_sum(a1);
            if (lane == 0) {
                const bool ok = s_cell[s] >= 0;
                sP[(2 * warp) * MB_CH + s] = ok ? a0 * scale : -INFINITY;
                sP[(2 * warp + 1) * MB_CH + s] = ok ? a1 * scale : -INFINITY;
            }
        }
        __syncthreads();
#pragma unroll
        for (int hh = 0; hh < 2; ++hh) {
            const int h = 2 * warp + hh;
            const float v = sP[h * MB_CH + lane];
            const float m_old = s_m[h];
            const float m_new = fmaxf(m_old, warp_max(v));
            const float e = (v == -INFINITY) ? 0.0f : expf(v - m_new);
            const float l = warp_sum(e);
            sP[h * MB_CH + lane] = e;
            __syncwarp();
            if (lane == 0) {
                const float sc = (m_old == -INFINITY) ? 0.0f : expf(m_old - m_new);
                s_scale[h] = sc;
                s_l[h] = s_l[h] * sc + l;
                s_m[h] = m_new;
            }
        }
        __syncthreads();
#pragma unroll
        for (int hh = 0; hh < MB_HG; ++hh) {
            const float sc = s_scale[hh];
            acc0[hh] *= sc;
            acc1[hh] *= sc;
        }
        for (int s = 0; s < MB_CH; ++s) {
            const float l0 = sL[s * KV + tid], l1 = sL[s * KV + tid + 256];
#pragma unroll
            for (int hh = 0; hh < MB_HG; ++hh) {
                const float p = sP[hh * MB_CH + s];
                acc0[hh] += p * l0;
                acc1[hh] += p * l1;
            }
        }
    }
    __syncthreads();
#pragma unroll
    for (int hh = 0; hh < MB_HG; ++hh) {
        const float L = s_l[hh];
        const float inv = L > 0.0f ? 1.0f / L : 0.0f;
        float* o = ctx + ((size_t) t * n_head + h0 + hh) * KV;
        o[tid] = acc0[hh] * inv;
        o[tid + 256] = acc1[hh] * inv;
    }
}

// ---------------------------------------------------------------- the NextN block's caches over a prompt
// h[t] = rms(mean_s R[t][s]) * w (the final hidden state the draft block reads; head_prep's arithmetic)
__global__ void __launch_bounds__(256) head_rows_kernel(const float* __restrict__ R, const float* __restrict__ w,
                                                        float eps, int E, float* __restrict__ h) {
    __shared__ float sred[32];
    const int t = blockIdx.x, tid = threadIdx.x;
    const float* Rt = R + (size_t) t * 4 * E;
    float v[16];
    float ss = 0.0f;
    const int per = E / (int) blockDim.x;
#pragma unroll
    for (int i = 0; i < 16; ++i) {
        if (i >= per) break;
        const int e = tid + i * (int) blockDim.x;
        float acc = 0.0f;
        for (int s = 0; s < 4; ++s) acc += Rt[e + (size_t) E * s];
        acc = acc / 4.0f;
        v[i] = acc;
        ss += acc * acc;
    }
    ss = block_sum(ss, sred);
    const float inv = rsqrtf(ss / (float) E + eps);
#pragma unroll
    for (int i = 0; i < 16; ++i) {
        if (i >= per) break;
        const int e = tid + i * (int) blockDim.x;
        h[(size_t) t * E + e] = v[i] * inv * w[e];
    }
}
// out16[t] = [rms(emb[t]) * enorm, rms(h[t]) * hnorm] (FP16, 2E per row)
__global__ void __launch_bounds__(256) mtp_in_rows_kernel(const float* __restrict__ emb, const float* __restrict__ h,
                                                          const float* __restrict__ enorm,
                                                          const float* __restrict__ hnorm, float eps, int E,
                                                          __half* __restrict__ out) {
    __shared__ float sred[32];
    const int t = blockIdx.x, tid = threadIdx.x;
    float a[16], b[16];
    float sa = 0.0f, sb = 0.0f;
    const int per = E / (int) blockDim.x;
#pragma unroll
    for (int i = 0; i < 16; ++i) {
        if (i >= per) break;
        const int e = tid + i * (int) blockDim.x;
        a[i] = emb[(size_t) t * E + e];
        b[i] = h[(size_t) t * E + e];
        sa += a[i] * a[i];
        sb += b[i] * b[i];
    }
    sa = block_sum(sa, sred);
    sb = block_sum(sb, sred);
    const float ia = rsqrtf(sa / (float) E + eps), ib = rsqrtf(sb / (float) E + eps);
#pragma unroll
    for (int i = 0; i < 16; ++i) {
        if (i >= per) break;
        const int e = tid + i * (int) blockDim.x;
        out[(size_t) t * 2 * E + e] = __float2half(a[i] * ia * enorm[e]);
        out[(size_t) t * 2 * E + E + e] = __float2half(b[i] * ib * hnorm[e]);
    }
}
// x[t] = rms(in[t]) * w, F32 and FP16
__global__ void __launch_bounds__(256) rms_rows_kernel(const float* __restrict__ in, const float* __restrict__ w, float eps,
                                                       int E, float* __restrict__ x, __half* __restrict__ x16) {
    __shared__ float sred[32];
    const int t = blockIdx.x, tid = threadIdx.x;
    float v[16];
    float ss = 0.0f;
    const int per = E / (int) blockDim.x;
#pragma unroll
    for (int i = 0; i < 16; ++i) {
        if (i >= per) break;
        const int e = tid + i * (int) blockDim.x;
        v[i] = in[(size_t) t * E + e];
        ss += v[i] * v[i];
    }
    ss = block_sum(ss, sred);
    const float inv = rsqrtf(ss / (float) E + eps);
#pragma unroll
    for (int i = 0; i < 16; ++i) {
        if (i >= per) break;
        const int e = tid + i * (int) blockDim.x;
        const float xv = v[i] * inv * w[e];
        x[(size_t) t * E + e] = xv;
        x16[(size_t) t * E + e] = __float2half(xv);
    }
}

// ---------------------------------------------------------------- the routed experts
__device__ void topk_argmax(float* sel, int E, int k, int* out, float* s_bv, int* s_bi) {
    const int tid = threadIdx.x, lane = tid & 31, warp = tid >> 5, nw = blockDim.x >> 5;
    for (int r = 0; r < k; ++r) {
        float bv = -INFINITY;
        int bi = 0x7fffffff;
        for (int e = tid; e < E; e += blockDim.x) {
            const float v = sel[e];
            if (v > bv || (v == bv && e < bi)) {
                bv = v;
                bi = e;
            }
        }
#pragma unroll
        for (int o = 16; o > 0; o >>= 1) {
            const float ov = __shfl_xor_sync(0xffffffffu, bv, o);
            const int oi = __shfl_xor_sync(0xffffffffu, bi, o);
            if (ov > bv || (ov == bv && oi < bi)) {
                bv = ov;
                bi = oi;
            }
        }
        if (lane == 0) {
            s_bv[warp] = bv;
            s_bi[warp] = bi;
        }
        __syncthreads();
        if (tid == 0) {
            float v = s_bv[0];
            int i = s_bi[0];
            for (int w = 1; w < nw; ++w)
                if (s_bv[w] > v || (s_bv[w] == v && s_bi[w] < i)) {
                    v = s_bv[w];
                    i = s_bi[w];
                }
            out[r] = i;
            if (i >= 0 && i < E) sel[i] = -INFINITY;
        }
        __syncthreads();
    }
}

__global__ void __launch_bounds__(256) route_kernel(const float* __restrict__ logits, const float* __restrict__ bias,
                                                    int E, int k, float w_scale, int norm_w, int* __restrict__ ids,
                                                    float* __restrict__ w) {
    const int t = blockIdx.x, tid = threadIdx.x;
    __shared__ float s_p[512], s_sel[512];
    __shared__ int s_ids[8];
    __shared__ float s_bv[32];
    __shared__ int s_bi[32];
    for (int e = tid; e < E; e += blockDim.x) {
        const float p = 1.0f / (1.0f + expf(-logits[(size_t) t * E + e]));
        s_p[e] = p;
        s_sel[e] = bias ? p + bias[e] : p;
    }
    __syncthreads();
    topk_argmax(s_sel, E, k, s_ids, s_bv, s_bi);
    if (tid == 0) {
        double sum = 0.0;
        for (int i = 0; i < k; ++i) sum += (double) s_p[s_ids[i]];
        const float inv = (float) (norm_w ? 1.0 / fmax(sum, 6.103515625e-5) : 1.0);
        for (int i = 0; i < k; ++i) {
            ids[(size_t) t * k + i] = s_ids[i];
            w[(size_t) t * k + i] = (float) ((double) s_p[s_ids[i]] * inv * w_scale);
        }
    }
}

__global__ void __launch_bounds__(256) expert_count_kernel(const int* __restrict__ ids, int n, int* __restrict__ counts,
                                                           int* __restrict__ rank) {
    const int e = blockIdx.x, tid = threadIdx.x, lane = tid & 31, warp = tid >> 5;
    __shared__ int s_wc[8];
    int base = 0;
    for (int c0 = 0; c0 < n; c0 += 256) {
        const int i = c0 + tid;
        const bool m = i < n && ids[i] == e;
        const unsigned ballot = __ballot_sync(0xffffffffu, m);
        if (lane == 0) s_wc[warp] = __popc(ballot);
        __syncthreads();
        int off = base;
        for (int w2 = 0; w2 < warp; ++w2) off += s_wc[w2];
        if (m) rank[i] = off + __popc(ballot & ((1u << lane) - 1u));
        int tot = 0;
        for (int w2 = 0; w2 < 8; ++w2) tot += s_wc[w2];
        base += tot;
        __syncthreads();
    }
    if (tid == 0) counts[e] = base;
}

__global__ void expert_scatter_kernel(const int* __restrict__ ids, const int* __restrict__ rank,
                                      const int* __restrict__ base, int n, int k, int* __restrict__ row_tok,
                                      int* __restrict__ pos) {
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;
    const int row = base[ids[i]] + rank[i];
    row_tok[row] = i / k;
    pos[i] = row;
}

__global__ void swiglu_rows_kernel(const float* __restrict__ gu, float* __restrict__ h, int rows, int n_ff, float limit) {
    const int64_t i = (int64_t) blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= (int64_t) rows * n_ff) return;
    const int64_t r = i / n_ff;
    const int c = (int) (i - r * n_ff);
    const float* row = gu + r * 2 * n_ff;
    const float g = fminf(row[c], limit);
    const float u = fminf(fmaxf(row[n_ff + c], -limit), limit);
    h[i] = (g / (1.0f + expf(-g))) * u;
}

__global__ void swiglu_f16_kernel(const float* __restrict__ gate, const float* __restrict__ up, __half* __restrict__ h,
                                  int64_t n, float limit) {
    const int64_t i = (int64_t) blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;
    const float g = fminf(gate[i], limit);
    const float u = fminf(fmaxf(up[i], -limit), limit);
    h[i] = __float2half((g / (1.0f + expf(-g))) * u);
}

__global__ void moe_combine_kernel(const float* __restrict__ out, const int* __restrict__ pos, const float* __restrict__ w,
                                   const float* __restrict__ sh, int k, int E, float* __restrict__ ffn) {
    const int t = blockIdx.x;
    __shared__ int s_pos[8];
    __shared__ float s_w[8];
    if (threadIdx.x < k) {
        s_pos[threadIdx.x] = pos[t * k + threadIdx.x];
        s_w[threadIdx.x] = w[t * k + threadIdx.x];
    }
    __syncthreads();
    for (int e = threadIdx.x; e < E; e += blockDim.x) {
        float m = 0.0f;
        for (int j = 0; j < k; ++j) m += s_w[j] * out[(size_t) s_pos[j] * E + e];
        ffn[(size_t) t * E + e] = m + (sh ? sh[(size_t) t * E + e] : 0.0f);
    }
}

}  // namespace

int launch_errors() { return g_errors.load(std::memory_order_relaxed); }

void bf16_to_f32(const uint16_t* src, float* dst, int64_t n, cudaStream_t s) {
    if (n <= 0) return;
    bf16_to_f32_kernel<<<nblk(n), 256, 0, s>>>(src, dst, n);
    check("bf16_to_f32");
}

void f32_to_f16(const float* src, void* dst, int64_t n, cudaStream_t s) {
    if (n <= 0) return;
    f32_to_f16_kernel<<<nblk(n), 256, 0, s>>>(src, (__half*) dst, n);
    check("f32_to_f16");
}

void embed_rows(const float* emb, float* R, int T, int n_embd, cudaStream_t s) {
    if (T <= 0) return;
    embed_rows_kernel<<<nblk((int64_t) T * 4 * n_embd), 256, 0, s>>>(emb, R, T, n_embd);
    check("embed_rows");
}

void hc_update(const float* block_out, float* R, const float* post, const float* comb, float* ss, int T, int n_embd,
               cudaStream_t s) {
    if (T <= 0) return;
    hc_update_kernel<<<T, 256, 0, s>>>(block_out, R, post, comb, ss, n_embd);
    check("hc_update");
}

void hc_finish(const HcFinishArgs& a, cudaStream_t s) {
    if (a.T <= 0) return;
    if (a.n_embd > 16 * 256 || a.n_embd % 256 != 0) {
        std::fprintf(stderr, "glm_batch hc_finish: n_embd %d unsupported\n", a.n_embd);
        return;
    }
    hc_finish_kernel<<<a.T, 256, 0, s>>>(a);
    check("hc_finish");
}

void kda_conv(const float* const proj[3], const float* const conv_w[3], const float* conv_state, float* const out[3],
              int T, int d_inner, int d_conv, cudaStream_t s) {
    if (T <= 0) return;
    Ptr3 p{{proj[0], proj[1], proj[2]}}, w{{conv_w[0], conv_w[1], conv_w[2]}};
    WPtr3 o{{out[0], out[1], out[2]}};
    kda_conv_kernel<<<nblk(3 * (int64_t) T * d_inner), 256, 0, s>>>(p, w, conv_state, o, T, d_inner, d_conv);
    check("kda_conv");
}

void kda_conv_state(const float* const proj[3], float* conv_state, int T, int d_inner, int d_conv, cudaStream_t s) {
    if (T <= 0) return;
    Ptr3 p{{proj[0], proj[1], proj[2]}};
    kda_conv_state_kernel<<<nblk(3 * (int64_t) d_inner), 256, 0, s>>>(p, conv_state, T, d_inner, d_conv);
    check("kda_conv_state");
}

void kda_rec(const float* q, const float* k, const float* v, const float* g1_raw, const float* dt_bias,
             const float* ssm_a, float lower_bound, const float* beta_raw, float* state, const float* g2_raw,
             const float* norm_w, float eps, int n_head, int T, void* out16, cudaStream_t s) {
    if (T <= 0) return;
    kda_rec_kernel<<<n_head, 256, 0, s>>>(q, k, v, g1_raw, dt_bias, ssm_a, lower_bound, beta_raw, state, g2_raw,
                                          norm_w, eps, n_head, T, (__half*) out16);
    check("kda_rec");
}

void dsa_prep(const DsaPrepArgs& a, cudaStream_t s) {
    if (a.T <= 0) return;
    dsa_prep_kernel<<<dim3((unsigned) a.T, 3), 512, 0, s>>>(a);
    check("dsa_prep");
}

void dsa_pool(const float* ik_cache, const float* ig_cache, const float* ape, float* pooled, int idx_key, int kpool,
              int pool0, int n, cudaStream_t s) {
    if (n <= 0) return;
    dsa_pool_kernel<<<n, ((idx_key + 31) / 32) * 32, 0, s>>>(ik_cache, ig_cache, ape, pooled, idx_key, kpool, pool0);
    check("dsa_pool");
}

void dsa_score(const float* iq, const float* pooled, const float* iw, int key_dim, int idx_heads, int p0, int kpool,
               int T, int max_vis, float* score, int score_ld, cudaStream_t s) {
    if (T <= 0 || max_vis <= 0) return;
    const size_t smem = ((size_t) idx_heads * key_dim + idx_heads) * sizeof(float);
    dsa_score_kernel<<<dim3((unsigned) ((max_vis + SCORE_POOLS - 1) / SCORE_POOLS), (unsigned) T), 256, smem, s>>>(
        iq, pooled, iw, key_dim, idx_heads, p0, kpool, score, score_ld);
    check("dsa_score");
}

void dsa_select(const float* score, int score_ld, int p0, int kpool, int top_pools_max, int tail, int T,
                int n_sel_max, int* cells, int* n_sel, cudaStream_t s) {
    if (T <= 0) return;
    dsa_select_kernel<<<T, 1024, 0, s>>>(score, score_ld, p0, kpool, top_pools_max, tail, n_sel_max, cells, n_sel);
    check("dsa_select");
}

void mla_attn(const float* q_abs, const float* lat, const int* cells, const int* n_sel, int n_sel_max, int n_head,
              int kv_lora, float scale, int T, float* ctx, cudaStream_t s) {
    if (T <= 0) return;
    if (kv_lora != 512 || n_head % MB_HG != 0) {
        std::fprintf(stderr, "glm_batch mla_attn: kv_lora %d / n_head %d unsupported\n", kv_lora, n_head);
        return;
    }
    const size_t smem = ((size_t) MB_CH * 512 + (size_t) MB_HG * MB_CH) * sizeof(float);
    static bool attr[16] = {};
    int dev = 0;
    cudaGetDevice(&dev);
    if (dev >= 0 && dev < 16 && !attr[dev]) {
        cudaFuncSetAttribute(mla_attn_kernel, cudaFuncAttributeMaxDynamicSharedMemorySize, (int) smem);
        attr[dev] = true;
    }
    mla_attn_kernel<<<dim3((unsigned) T, (unsigned) (n_head / MB_HG)), 256, smem, s>>>(q_abs, lat, cells, n_sel,
                                                                                    n_sel_max, n_head, scale, ctx);
    check("mla_attn");
}

void head_rows(const float* R, const float* w, float eps, int T, int n_embd, float* h, cudaStream_t s) {
    if (T <= 0) return;
    head_rows_kernel<<<T, 256, 0, s>>>(R, w, eps, n_embd, h);
    check("head_rows");
}

void mtp_in_rows(const float* emb, const float* h, const float* enorm, const float* hnorm, float eps, int T, int n_embd,
                 void* out16, cudaStream_t s) {
    if (T <= 0) return;
    mtp_in_rows_kernel<<<T, 256, 0, s>>>(emb, h, enorm, hnorm, eps, n_embd, (__half*) out16);
    check("mtp_in_rows");
}

void rms_rows(const float* in, const float* w, float eps, int T, int n_embd, float* x, void* x16, cudaStream_t s) {
    if (T <= 0) return;
    rms_rows_kernel<<<T, 256, 0, s>>>(in, w, eps, n_embd, x, (__half*) x16);
    check("rms_rows");
}

void route(const float* logits, const float* bias, int n_expert, int k, float w_scale, bool norm_w, int T, int* ids,
           float* w, cudaStream_t s) {
    if (T <= 0) return;
    route_kernel<<<T, 256, 0, s>>>(logits, bias, n_expert, k, w_scale, norm_w ? 1 : 0, ids, w);
    check("route");
}

void expert_count(const int* ids, int n, int n_expert, int* counts, int* rank, cudaStream_t s) {
    expert_count_kernel<<<n_expert, 256, 0, s>>>(ids, n, counts, rank);
    check("expert_count");
}

void expert_scatter(const int* ids, const int* rank, const int* base, int n, int k, int* row_tok, int* pos,
                    cudaStream_t s) {
    if (n <= 0) return;
    expert_scatter_kernel<<<nblk(n), 256, 0, s>>>(ids, rank, base, n, k, row_tok, pos);
    check("expert_scatter");
}

void swiglu_rows(const float* gu, float* h, int rows, int n_ff, float limit, cudaStream_t s) {
    if (rows <= 0) return;
    swiglu_rows_kernel<<<nblk((int64_t) rows * n_ff), 256, 0, s>>>(gu, h, rows, n_ff, limit);
    check("swiglu_rows");
}

void swiglu_f16(const float* gate, const float* up, void* h16, int64_t n, float limit, cudaStream_t s) {
    if (n <= 0) return;
    swiglu_f16_kernel<<<nblk(n), 256, 0, s>>>(gate, up, (__half*) h16, n, limit);
    check("swiglu_f16");
}

void moe_combine(const float* out, const int* pos, const float* w, const float* sh, int T, int k, int n_embd,
                 float* ffn, cudaStream_t s) {
    if (T <= 0) return;
    moe_combine_kernel<<<T, 256, 0, s>>>(out, pos, w, sh, k, n_embd, ffn);
    check("moe_combine");
}

}  // namespace strata::kernels::glmb
