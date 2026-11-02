// src/flint/flint_kernels.cuh - CUDA kernels for Flint 40B Preview.
#pragma once

#include "strata/kernels/dp4a.hpp"
#include "strata/kernels/q8_1_finite.hpp"

#include <cuda_fp16.h>
#include <cuda_runtime.h>

#include <cfloat>
#include <cmath>
#include <cstdint>

namespace flintk {

struct QB {   // a q8_1 activation block
    half2 ds;
    int8_t qs[32];
};
static_assert(sizeof(QB) == 36, "q8_1 block");

#pragma pack(push, 1)
struct BlockQ80 {   // standard ggml block_q8_0 KV cache block
    half d;
    int8_t qs[32];
};
#pragma pack(pop)
static_assert(sizeof(BlockQ80) == 34, "q8_0 block size");

__device__ __forceinline__ float wsum(float x) {
#pragma unroll
    for (int o = 16; o > 0; o >>= 1) x += __shfl_xor_sync(0xffffffffu, x, o, 32);
    return x;
}

__device__ __forceinline__ float wmax(float x) {
#pragma unroll
    for (int o = 16; o > 0; o >>= 1) x = fmaxf(x, __shfl_xor_sync(0xffffffffu, x, o, 32));
    return x;
}

__device__ __forceinline__ float sigm(float v) { return 1.0f / (1.0f + expf(-v)); }
__device__ __forceinline__ float silu(float v) { return v / (1.0f + expf(-v)); }

#if defined(__CUDACC__) && !defined(__HIPCC__)
__device__ __forceinline__ float fdiv_fm(float a, float b) {
    float r;
    asm("div.approx.ftz.f32 %0, %1, %2;" : "=f"(r) : "f"(a), "f"(b));
    return r;
}
__device__ __forceinline__ float frcp_fm(float a) {
    float r;
    asm("rcp.approx.ftz.f32 %0, %1;" : "=f"(r) : "f"(a));
    return r;
}
__device__ __forceinline__ float fexp_fm(float a) { return __expf(a); }
#else
__device__ __forceinline__ float fdiv_fm(float a, float b) { return a / b; }
__device__ __forceinline__ float frcp_fm(float a) { return 1.0f / a; }
__device__ __forceinline__ float fexp_fm(float a) { return expf(a); }
#endif

__device__ __forceinline__ void q8_store(QB* __restrict__ y, int i, float v, int lane) {
    const float amax = wmax(fabsf(v));
    const float sum = wsum(v);
    const float d = strata::kernels::q8_1_finite(fdiv_fm(amax, 127.0f));
    int8_t q = 0;
    if (amax != 0.0f) {
        const float qf = roundf(fdiv_fm(v, d));
        q = (int8_t) (qf > 127.0f ? 127.0f : qf < -127.0f ? -127.0f : qf);
    }
    y[i >> 5].qs[lane] = q;
    if (lane == 0) y[i >> 5].ds = strata::kernels::q8_1_ds(d, sum);
}

__device__ inline float block_sum(float v, float* sh) {
    sh[threadIdx.x] = v;
    __syncthreads();
    for (int s = blockDim.x / 2; s > 0; s >>= 1) {
        if ((int) threadIdx.x < s) sh[threadIdx.x] += sh[threadIdx.x + s];
        __syncthreads();
    }
    const float r = sh[0];
    __syncthreads();
    return r;
}

__device__ inline float block_max(float v, float* sh) {
    sh[threadIdx.x] = v;
    __syncthreads();
    for (int s = blockDim.x / 2; s > 0; s >>= 1) {
        if ((int) threadIdx.x < s) sh[threadIdx.x] = fmaxf(sh[threadIdx.x], sh[threadIdx.x + s]);
        __syncthreads();
    }
    const float r = sh[0];
    __syncthreads();
    return r;
}

__global__ void k_f32_to_f16(const float* __restrict__ src, half* __restrict__ dst, int64_t n) {
    const int64_t idx = (int64_t) blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        dst[idx] = __float2half(src[idx]);
    }
}

// ---------------------------------------------------------------- Flint RoPE & Attention Prep
// H = 32, HKV = 4, HD = 128
__global__ void k_flint_attn_prep(const float* __restrict__ q_in, const float* __restrict__ kx, const float* __restrict__ vx,
                                  const float* __restrict__ qw, const float* __restrict__ kw,
                                  float* __restrict__ q_out, BlockQ80* __restrict__ kc, BlockQ80* __restrict__ vc,
                                  int H, int HKV, int HD, int n_rot,
                                  const int32_t* __restrict__ pos0, int ctx, float base, float eps,
                                  int kv_resident = 0) {
    __shared__ float sh[256];
    __shared__ float buf[128];
    const int head = blockIdx.x, t = blockIdx.y;
    const int pos = *pos0 + t;
    const bool isq = head < H;
    const float* src = isq ? q_in + ((size_t) t * H + head) * HD : kx + ((size_t) t * HKV + head - H) * HD;
    const int i = threadIdx.x;
    const float v0 = i < HD ? src[i] : 0.f;
    const float s = block_sum(v0 * v0, sh);
    const float r = rsqrtf(s / (float) HD + eps);
    if (i < HD) buf[i] = v0 * r * (isq ? (qw != nullptr ? qw[i] : 1.0f) : (kw != nullptr ? kw[i] : 1.0f));
    __syncthreads();
    if (i >= HD) return;
    float v = buf[i];
    const int half = n_rot / 2;
    if (i < n_rot) {
        const int p = i < half ? i : i - half;
        const float theta = (float) pos * powf(base, -2.0f * (float) p / (float) n_rot);
        float sn, cs;
        sincosf(theta, &sn, &cs);
        v = i < half ? buf[i] * cs - buf[i + half] * sn : buf[i - half] * sn + buf[i] * cs;
    }
    if (isq) {
        q_out[((size_t) t * H + head) * HD + i] = v;
    } else {
        const int kh = head - H;
        const float val_v = vx[((size_t) t * HKV + kh) * HD + i];
        const int b = i >> 5;
        const int lane = i & 31;

        const float amax_k = wmax(fabsf(v));
        const float id_k = amax_k > 0.0f ? 127.0f / amax_k : 0.0f;
        const float qf_k = rintf(v * id_k);
        const int8_t q_k = (int8_t) fmaxf(-128.0f, fminf(127.0f, qf_k));

        const float amax_v = wmax(fabsf(val_v));
        const float id_v = amax_v > 0.0f ? 127.0f / amax_v : 0.0f;
        const float qf_v = rintf(val_v * id_v);
        const int8_t q_v = (int8_t) fmaxf(-128.0f, fminf(127.0f, qf_v));

        const int vpos = (kv_resident > 0 && pos >= 4) ? (4 + ((pos - 4) % (kv_resident - 4))) : pos;
        const size_t b_idx = ((size_t) kh * (kv_resident > 0 ? kv_resident : ctx) + vpos) * 4 + b;
        kc[b_idx].qs[lane] = q_k;
        vc[b_idx].qs[lane] = q_v;
        if (lane == 0) {
            kc[b_idx].d = __float2half(amax_k / 127.0f);
            vc[b_idx].d = __float2half(amax_v / 127.0f);
        }
    }
}

// ---------------------------------------------------------------- Batched Tensor Core Attention Kernels
__global__ void k_flint_kv_unpack_f16(const BlockQ80* __restrict__ kc, const BlockQ80* __restrict__ vc,
                                      half* __restrict__ k16, half* __restrict__ v16,
                                      int ctx, int P, int HD, int HKV, int kv_resident) {
    const int64_t i = (int64_t) blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= (int64_t) HKV * P * HD) return;
    const int64_t per = (int64_t) P * HD;
    const int64_t kvh = i / per;
    const int64_t rest = i % per;
    const int64_t p = rest / HD;
    const int64_t d = rest % HD;
    const int b = (int) (d >> 5);
    const int lane = (int) (d & 31);

    const int vpos = (kv_resident > 0 && p >= 4) ? (4 + (int) ((p - 4) % (kv_resident - 4))) : (int) p;
    const size_t b_idx = ((size_t) kvh * (kv_resident > 0 ? kv_resident : ctx) + vpos) * 4 + b;
    const float dk = __half2float(kc[b_idx].d);
    const float dv = __half2float(vc[b_idx].d);
    k16[i] = __float2half((float) kc[b_idx].qs[lane] * dk);
    v16[i] = __float2half((float) vc[b_idx].qs[lane] * dv);
}

__global__ void k_flint_attn_softmax16(const float* __restrict__ S, half* __restrict__ S16,
                                       int ld, int nt, int P, int pos0, float scale,
                                       int sliding_window, const float* __restrict__ sinks) {
    __shared__ float sh[256];
    const int t = blockIdx.x, h = blockIdx.y;
    const int pos = pos0 + t;
    const float* row = S + ((size_t) h * nt + t) * ld;
    half* out = S16 + ((size_t) h * nt + t) * ld;

    float mx = -1e30f;
    for (int p = threadIdx.x; p < P; p += blockDim.x) {
        if (p <= pos && (sliding_window <= 0 || p < 4 || (pos - p) < sliding_window)) {
            float val = row[p] * scale;
            if (p == 0 && sinks != nullptr) val += sinks[h];
            mx = fmaxf(mx, val);
        }
    }
    mx = block_max(mx, sh);

    float sum = 0.f;
    for (int p = threadIdx.x; p < P; p += blockDim.x) {
        if (p <= pos && (sliding_window <= 0 || p < 4 || (pos - p) < sliding_window)) {
            float val = row[p] * scale;
            if (p == 0 && sinks != nullptr) val += sinks[h];
            sum += expf(val - mx);
        }
    }
    sum = block_sum(sum, sh);
    const float inv = sum > 0.f ? 1.0f / sum : 0.f;

    for (int p = threadIdx.x; p < ld; p += blockDim.x) {
        if (p < P && p <= pos && (sliding_window <= 0 || p < 4 || (pos - p) < sliding_window)) {
            float val = row[p] * scale;
            if (p == 0 && sinks != nullptr) val += sinks[h];
            out[p] = __float2half(expf(val - mx) * inv);
        } else {
            out[p] = __float2half(0.f);
        }
    }
}

// ---------------------------------------------------------------- Split-K FlashAttention for Flint
constexpr int kChunkCells = 256;

__global__ void k_flint_attn_part(const float* __restrict__ q, const BlockQ80* __restrict__ kc, const BlockQ80* __restrict__ vc,
                                  const int32_t* __restrict__ pos0, int t0, int H, int HKV, int ctx,
                                  float scale, int n_chunk, float* __restrict__ pacc,
                                  float* __restrict__ pm, float* __restrict__ pl,
                                  const float* __restrict__ sinks, int sliding_window,
                                  int kv_resident = 0) {
    constexpr int HD = 128;
    __shared__ float qs[HD], sc[kChunkCells], sh[256];
    const int c = blockIdx.x, row = blockIdx.y;
    const int t = t0 + row / H, h = row % H, kvh = h / (H / HKV);
    const int npos = *pos0 + t + 1;
    const int p0 = c * kChunkCells;
    if (p0 >= npos) return;
    if (sliding_window > 0 && c > 0 && p0 + kChunkCells <= npos - sliding_window) return;
    const int j = threadIdx.x;
    if (j < HD) qs[j] = q[((size_t) t * H + h) * HD + j];
    __syncthreads();
    const int p = p0 + j;
    float s = -1e30f;
    if (p < npos) {
        // sliding window check if applicable
        if (sliding_window <= 0 || (npos - 1 - p) < sliding_window) {
            const int vp_idx = (kv_resident > 0 && p >= 4) ? (4 + ((p - 4) % (kv_resident - 4))) : p;
            const BlockQ80* kp = kc + ((size_t) kvh * (kv_resident > 0 ? kv_resident : ctx) + vp_idx) * 4;
            float a = 0.f;
            #pragma unroll
            for (int b = 0; b < 4; ++b) {
                const float d = __half2float(kp[b].d);
                float sum_b = 0.f;
                #pragma unroll
                for (int i = 0; i < 32; ++i) {
                    sum_b += qs[b * 32 + i] * (float) kp[b].qs[i];
                }
                a += sum_b * d;
            }
            s = a * scale;
            if (p == 0 && sinks != nullptr) s += sinks[h];
        }
    }
    const float m = block_max(s, sh);
    const float e = (p < npos && s > -1e9f) ? expf(s - m) : 0.f;
    sc[j] = e;
    const float l = block_sum(e, sh);
    if (j < HD) {
        const int n = min(kChunkCells, npos - p0);
        const int b = j >> 5;
        const int idx = j & 31;
        float acc = 0.f;
        for (int k = 0; k < n; ++k) {
            const int cur_p = p0 + k;
            const int cur_vpos = (kv_resident > 0 && cur_p >= 4) ? (4 + ((cur_p - 4) % (kv_resident - 4))) : cur_p;
            const BlockQ80* vp = vc + ((size_t) kvh * (kv_resident > 0 ? kv_resident : ctx) + cur_vpos) * 4 + b;
            const float v_val = (float) vp->qs[idx] * __half2float(vp->d);
            acc += sc[k] * v_val;
        }
        const size_t o = (size_t) row * n_chunk + c;
        pacc[o * HD + j] = acc;
        if (j == 0) {
            pm[o] = m;
            pl[o] = l;
        }
    }
}

__global__ void k_flint_attn_merge(const float* __restrict__ pacc, const float* __restrict__ pm,
                                   const float* __restrict__ pl, const int32_t* __restrict__ pos0,
                                   int t0, int H, int n_chunk, float* __restrict__ out) {
    constexpr int HD = 128;
    const int row = blockIdx.x, t = t0 + row / H, d = threadIdx.x;
    const int npos = *pos0 + t + 1;
    const int nc = (npos + kChunkCells - 1) / kChunkCells;
    float M = -1e30f;
    for (int c = 0; c < nc; ++c) {
        const size_t o = (size_t) row * n_chunk + c;
        M = fmaxf(M, pm[o]);
    }
    float D = 0.f;
    for (int c = 0; c < nc; ++c) {
        const size_t o = (size_t) row * n_chunk + c;
        D += pl[o] * expf(pm[o] - M);
    }
    float acc = 0.f;
    for (int c = 0; c < nc; ++c) {
        const size_t o = (size_t) row * n_chunk + c;
        acc += pacc[o * HD + d] * expf(pm[o] - M);
    }
    out[(size_t) row * HD + d] = acc / (D + 1e-9f);
}

// ---------------------------------------------------------------- Router GEMV (FP32)
__global__ void k_gemv_f32(const float* __restrict__ W, const float* __restrict__ x, float* __restrict__ y, int n_in, int n_out) {
    const int warp = threadIdx.x / 32, lane = threadIdx.x % 32;
    const int o = blockIdx.x * 8 + warp;
    if (o >= n_out) return;
    const float4* wr = (const float4*) (W + (size_t) o * n_in);
    const float4* xr = (const float4*) (x + (size_t) blockIdx.y * n_in);
    float acc = 0.f;
    for (int i = lane; i < n_in / 4; i += 32) {
        const float4 a = wr[i], b = xr[i];
        acc += a.x * b.x + a.y * b.y + a.z * b.z + a.w * b.w;
    }
    for (int m = 16; m > 0; m >>= 1) acc += __shfl_xor_sync(0xffffffffu, acc, m);
    if (lane == 0) y[(size_t) blockIdx.y * n_out + o] = acc;
}

// ---------------------------------------------------------------- MoE Router Top-K (256 experts, top 6)
template <int NE, int K>
__launch_bounds__(256, 1)
__global__ void route_topk(const float* __restrict__ logits, int32_t* __restrict__ ids, float* __restrict__ weights,
                           int n_tok) {
    constexpr int V = NE / 32;
    const int tk = (int) blockIdx.x * 8 + (int) threadIdx.y;
    if (tk >= n_tok) return;
    const float* lg = logits + (size_t) tk * NE;
    int32_t* id = ids + (size_t) tk * K;
    float* wt = weights + (size_t) tk * K;
    const int lane = threadIdx.x;
    float values[V];
#pragma unroll
    for (int i = 0; i < V; ++i) values[i] = lg[lane + i * 32];
    float maximum = -1e30f;
#pragma unroll
    for (int i = 0; i < V; ++i) maximum = max(maximum, values[i]);
    for (int mask = 16; mask; mask >>= 1) maximum = max(maximum, __shfl_xor_sync(0xffffffffu, maximum, mask, 32));
    float sum = 0.0f;
#pragma unroll
    for (int i = 0; i < V; ++i) {
        values[i] = expf(values[i] - maximum);
        sum += values[i];
    }
    for (int mask = 16; mask; mask >>= 1) sum += __shfl_xor_sync(0xffffffffu, sum, mask, 32);
    const float reciprocal = 1.0f / sum;
#pragma unroll
    for (int i = 0; i < V; ++i) {
        values[i] *= reciprocal;
        if (__isnanf(values[i])) values[i] = -FLT_MAX;
    }
    float selected = 0.0f, selected_sum = 0.0f;
    for (int rank = 0; rank < K; ++rank) {
        float best = values[0];
        int expert = lane;
#pragma unroll
        for (int i = 1; i < V; ++i) {
            if (values[i] > best) { best = values[i]; expert = lane + i * 32; }
        }
#pragma unroll
        for (int mask = 16; mask; mask >>= 1) {
            const float other = __shfl_xor_sync(0xffffffffu, best, mask, 32);
            const int other_id = __shfl_xor_sync(0xffffffffu, expert, mask, 32);
            if (other > best || (other == best && other_id < expert)) { best = other; expert = other_id; }
        }
        if ((expert & 31) == lane) {
            values[expert / 32] = -1e30f;
            id[rank] = expert;
            selected_sum += best;
        }
        if (rank == lane) selected = best;
    }
    for (int mask = 16; mask; mask >>= 1) selected_sum += __shfl_xor_sync(0xffffffffu, selected_sum, mask, 32);
    selected_sum = max(selected_sum, 6.103515625e-5f);
    const float inverse_selected_sum = 1.0f / selected_sum;
    if (lane < K) wt[lane] = selected * inverse_selected_sum;
}

// ---------------------------------------------------------------- MoE Residency-Biased Routing (Zero PCIe Latency)
template <int NE, int K>
__global__ void route_resident(const float* __restrict__ logits, int32_t* __restrict__ ids, float* __restrict__ weights,
                               const int32_t* __restrict__ res, int n_tok, float margin) {
    const int t = blockIdx.x;
    if (t >= n_tok) return;
    const int lane = (int) threadIdx.x;
    const float* l = logits + (size_t) t * NE;
    int32_t* id = ids + (size_t) t * K;
    int32_t my[K];
    for (int r = 0; r < K; ++r) my[r] = id[r];
    int swaps = 0;
    for (int r = 0; r < K; ++r) {
        const int e = my[r];
        if (res[e] >= 0) continue;
        float bv = -1e30f;
        int bi = -1;
        for (int f = lane; f < NE; f += 32) {
            if (res[f] < 0) continue;
            bool used = false;
            for (int q = 0; q < K; ++q) used |= my[q] == f;
            if (used) continue;
            if (l[f] > bv) { bv = l[f]; bi = f; }
        }
#pragma unroll
        for (int off = 16; off > 0; off >>= 1) {
            const float ov = __shfl_down_sync(0xffffffffu, bv, off);
            const int oi = __shfl_down_sync(0xffffffffu, bi, off);
            if (ov > bv || (ov == bv && oi >= 0 && (bi < 0 || oi < bi))) { bv = ov; bi = oi; }
        }
        if (lane == 0 && bi >= 0 && l[e] - bv <= margin) { my[r] = bi; ++swaps; }
        my[r] = __shfl_sync(0xffffffffu, my[r], 0);
    }
    const int sw = __shfl_sync(0xffffffffu, swaps, 0);
    if (sw) {
        float m = -1e30f;
        for (int r = 0; r < K; ++r) m = fmaxf(m, l[my[r]]);
        float ex[K], sum = 0.0f;
        for (int r = 0; r < K; ++r) { ex[r] = expf(l[my[r]] - m); sum += ex[r]; }
        for (int r = 0; r < K; ++r) weights[(size_t) t * K + r] = ex[r] / sum;
    }
    if (lane == 0) {
        for (int r = 0; r < K; ++r) id[r] = my[r];
    }
}


// ---------------------------------------------------------------- MoE Grouping (Direct VRAM / Staged Blobs)
__global__ void k_group(const int32_t* ids, int n_ent, int K_, int n_expert,
                        const uint8_t* blobs, unsigned long long blob, unsigned long long* grp_ptr, int32_t* grp_start,
                        int32_t* n_groups, int32_t* ent_dst, int32_t* ent_tok) {
    __shared__ int cnt[512], off[512];
    for (int x = threadIdx.x; x < n_expert; x += blockDim.x) cnt[x] = 0;
    __syncthreads();
    for (int e = threadIdx.x; e < n_ent; e += blockDim.x)
        atomicAdd(&cnt[ids[e]], 1);
    __syncthreads();
    if (threadIdx.x == 0) {
        int g = 0, pos = 0;
        for (int x = 0; x < n_expert; ++x) {
            off[x] = pos;
            if (cnt[x] == 0) continue;
            grp_ptr[g] = (unsigned long long) (blobs + (size_t) x * blob);
            grp_start[g] = pos;
            pos += cnt[x];
            ++g;
        }
        grp_start[g] = pos;
        *n_groups = g;
        for (int e = 0; e < n_ent; ++e) {
            const int p = off[ids[e]]++;
            ent_dst[p] = e;
            ent_tok[p] = e / K_;
        }
    }
}

// ---------------------------------------------------------------- MoE Grouping with Slot-Mapping
__global__ void k_group_hy(const int32_t* ids, int n_ent, int K_, int n_expert, const int32_t* slot_of,
                           const uint8_t* blobs, unsigned long long blob, unsigned long long* grp_ptr, int32_t* grp_start,
                           int32_t* n_groups, int32_t* ent_dst, int32_t* ent_tok) {
    __shared__ int cnt[512], off[512];
    for (int x = threadIdx.x; x < n_expert; x += blockDim.x) cnt[x] = 0;
    __syncthreads();
    for (int e = threadIdx.x; e < n_ent; e += blockDim.x)
        if (slot_of[ids[e]] >= 0) atomicAdd(&cnt[ids[e]], 1);
    __syncthreads();
    if (threadIdx.x == 0) {
        int g = 0, pos = 0;
        for (int x = 0; x < n_expert; ++x) {
            off[x] = pos;
            if (cnt[x] == 0) continue;
            grp_ptr[g] = (unsigned long long) (blobs + (size_t) slot_of[x] * blob);
            grp_start[g] = pos;
            pos += cnt[x];
            ++g;
        }
        grp_start[g] = pos;
        *n_groups = g;
        for (int e = 0; e < n_ent; ++e) {
            if (slot_of[ids[e]] < 0) continue;
            const int p = off[ids[e]]++;
            ent_dst[p] = e;
            ent_tok[p] = e / K_;
        }
    }
}

// ---------------------------------------------------------------- Expert Routing Usage Counter
__global__ void k_count_usage(const int32_t* ids, uint32_t* counts, int n_ent) {
    const int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n_ent) {
        atomicAdd(&counts[ids[idx]], 1);
    }
}

// ---------------------------------------------------------------- Clear Miss Rows in Eparts
__global__ void k_clear_miss(float* eparts, const int32_t* slot_of, const int32_t* ids, int n_ent, int n) {
    const int e = blockIdx.x;
    if (e >= n_ent) return;
    if (slot_of[ids[e]] >= 0) return; // hit: computed on GPU
    float* row = eparts + (size_t) e * n;
    for (int i = threadIdx.x; i < n; i += blockDim.x) {
        row[i] = 0.0f;
    }
}

// ---------------------------------------------------------------- CPU Fallback Merge
__global__ void k_merge_cpu(float* parts, const float* cpu, const int32_t* miss, int n) {
    const int e = blockIdx.x;
    if (!miss[e]) return;
    for (int i = threadIdx.x; i < n; i += blockDim.x) parts[(size_t) e * n + i] = cpu[(size_t) e * n + i];
}

// ---------------------------------------------------------------- MoE Combine & Shared Expert Add
__global__ void __launch_bounds__(1024) k_flint_moe_out(float* x, const float* parts, const int32_t* prow,
                                                       const float* wts, int K_, const float* shared,
                                                       const float* wn, float* h, int n, float eps, QB* hq = nullptr) {
    __shared__ float vv[2048];
    __shared__ float sh[256];
    __shared__ const float* pk[16];
    const int t = blockIdx.x;
    float* xr = x + (size_t) t * n;
    if (threadIdx.x < K_) pk[threadIdx.x] = parts + (size_t) (prow ? prow[t * K_ + threadIdx.x] : t * K_ + threadIdx.x) * n;
    __syncthreads();
    const float* wr = wts + (size_t) t * K_;
    for (int i = threadIdx.x; i < n; i += 1024) {
        float acc = wr[0] * pk[0][i];
        for (int k = 1; k < K_; ++k) acc = fmaf(wr[k], pk[k][i], acc);
        if (shared) acc += shared[(size_t) t * n + i];
        const float v = xr[i] + acc;
        xr[i] = v;
        vv[i] = v;
    }
    if (wn == nullptr) return;
    __syncthreads();
    if (threadIdx.x < 256) {
        float s = 0.f;
        for (int i = threadIdx.x; i < n; i += 256) s += vv[i] * vv[i];
        sh[threadIdx.x] = s;
    }
    __syncthreads();
    for (int s = 128; s > 0; s >>= 1) {
        if ((int) threadIdx.x < s) sh[threadIdx.x] += sh[threadIdx.x + s];
        __syncthreads();
    }
    const float r = rsqrtf(sh[0] / (float) n + eps);
    const int lane = threadIdx.x % 32;
    for (int i = threadIdx.x; i < n; i += 1024) {
        const float v = vv[i] * r * wn[i];
        if (h) h[(size_t) t * n + i] = v;
        if (hq) q8_store(hq + (size_t) t * (n >> 5), i, v, lane);
    }
}

// ---------------------------------------------------------------- SwiGLU & Residual
__global__ void k_swiglu(const float* __restrict__ gate, const float* __restrict__ up,
                         float* __restrict__ out, int count) {
    const int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < count) {
        out[idx] = silu(gate[idx]) * up[idx];
    }
}

__global__ void k_add_rms(float* x, const float* y, const float* wn, float* h, int n, float eps) {
    __shared__ float sh[256];
    const int t = blockIdx.x;
    float* xr = x + (size_t) t * n;
    const float* yr = y + (size_t) t * n;
    for (int i = threadIdx.x; i < n; i += 256) xr[i] += yr[i];
    __syncthreads();
    float s = 0.f;
    for (int i = threadIdx.x; i < n; i += 256) s += xr[i] * xr[i];
    const float total = block_sum(s, sh);
    const float r = rsqrtf(total / (float) n + eps);
    for (int i = threadIdx.x; i < n; i += 256) h[(size_t) t * n + i] = xr[i] * r * wn[i];
}

} // namespace flintk
