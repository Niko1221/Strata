// src/kernels/cuda/qsa_decode_attn.cu - see include/strata/kernels/qsa_decode_attn.hpp.
#include "strata/kernels/qsa_decode_attn.hpp"
#include "strata/kernels/kv_q8.hpp"

#include <cuda_fp16.h>
#include <cuda_runtime.h>

#include <cfloat>
#include <cstdio>
#include <cstdlib>

namespace strata::kernels {
namespace {

constexpr int HD = 256;          // head_dim
constexpr int G = 12;            // query heads per KV head (24 / 2)
constexpr int CHUNK = 64;        // cells per block
constexpr int THREADS = 256;
constexpr int WARPS = THREADS / 32;

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

// 8 consecutive values of one cell's key or value row for KV head `kvh`, dimensions [d0, d0+8).
template <bool INT8>
__device__ __forceinline__ void load8(const QsaAttnPools& p, bool value, long long row, int d0, float* out) {
    if constexpr (!INT8) {
        const uint16_t* base = (value ? p.v_pool : p.k_pool) + row * HD + d0;
        const uint4 raw = *reinterpret_cast<const uint4*>(base);
        const __half2* h2 = reinterpret_cast<const __half2*>(&raw);
#pragma unroll
        for (int j = 0; j < 4; ++j) {
            const float2 f = __half22float2(h2[j]);
            out[2 * j] = f.x;
            out[2 * j + 1] = f.y;
        }
    } else {
        const int8_t* codes = (value ? p.v_q : p.k_q) + row * HD + d0;
        const uint16_t sbits = (value ? p.v_scale : p.k_scale)[row * (HD / KV_Q8_GROUP) + d0 / KV_Q8_GROUP];
        const float sc = __half2float(__ushort_as_half(sbits));
        const uint2 raw = *reinterpret_cast<const uint2*>(codes);
        const int8_t* c = reinterpret_cast<const int8_t*>(&raw);
#pragma unroll
        for (int j = 0; j < 8; ++j) out[j] = (float) c[j] * sc;
    }
}

template <bool INT8>
__global__ void __launch_bounds__(THREADS) attn_chunk_kernel(const float* __restrict__ q, QsaAttnPools p,
                                                             const int32_t* __restrict__ ids,
                                                             const int32_t* __restrict__ step, int n_kv_heads,
                                                             int page_size, float scale, float* __restrict__ part_acc,
                                                             float* __restrict__ part_m, float* __restrict__ part_l,
                                                             int n_chunks, int cap = 0, long long scratch_stride = 0) {
    // batched form: query blockIdx.z, with its own q row, selection, step and scratch
    q += (size_t) blockIdx.z * (size_t) (n_kv_heads * G) * HD;
    ids += (size_t) blockIdx.z * (size_t) cap;
    step += (size_t) blockIdx.z * kStepCount;
    part_acc += (size_t) blockIdx.z * (size_t) scratch_stride;
    part_m += (size_t) blockIdx.z * (size_t) scratch_stride;
    part_l += (size_t) blockIdx.z * (size_t) scratch_stride;
    __shared__ __align__(16) float sq[G][HD];     // 12 KB: this KV head's query heads
    __shared__ float sp[G][CHUNK];                // scores, then probabilities
    __shared__ long long srow[CHUNK];             // pool row of each cell (page, kv head, slot)
    const int n_ids = __ldg(step + kStepWidth);
    const int chunk = blockIdx.x, kvh = blockIdx.y;
    const int t = threadIdx.x, lane = t & 31, warp = t >> 5;
    const int c0 = chunk * CHUNK;
    const int n_here = min(CHUNK, n_ids - c0);
    const int slot = kvh * n_chunks + chunk;
    if (n_here <= 0) {
        if (t < G) { part_m[slot * G + t] = -FLT_MAX; part_l[slot * G + t] = 0.0f; }
        return;
    }
    for (int i = t; i < G * HD; i += THREADS) sq[i / HD][i % HD] = q[(size_t) (kvh * G) * HD + i];
    if (t < CHUNK) {
        long long r = -1;
        if (t < n_here) {
            const int cell = ids[c0 + t];
            const long long page = (long long) p.page_table[cell / page_size];
            r = (page * n_kv_heads + kvh) * page_size + (cell % page_size);
        }
        srow[t] = r;
    }
    __syncthreads();
    // scores: each warp takes cells warp, warp+8, ...; each lane holds 8 of the 256 dimensions.
    for (int c = warp; c < CHUNK; c += WARPS) {
        if (c >= n_here) {
            if (lane < G) sp[lane][c] = -FLT_MAX;
            continue;
        }
        float k8[8];
        load8<INT8>(p, false, srow[c], lane * 8, k8);
#pragma unroll
        for (int h = 0; h < G; ++h) {
            const float4 qa = *reinterpret_cast<const float4*>(&sq[h][lane * 8]);
            const float4 qb = *reinterpret_cast<const float4*>(&sq[h][lane * 8 + 4]);
            float s = k8[0] * qa.x + k8[1] * qa.y + k8[2] * qa.z + k8[3] * qa.w +
                      k8[4] * qb.x + k8[5] * qb.y + k8[6] * qb.z + k8[7] * qb.w;
            s = warp_sum(s);
            if (lane == 0) sp[h][c] = s * scale;
        }
    }
    __syncthreads();
    // per-head chunk max and exp-sum: warp w handles heads w and w+8.
    for (int h = warp; h < G; h += WARPS) {
        const float a = sp[h][lane], b = sp[h][lane + 32];
        const float m = warp_max(fmaxf(a, b));
        const float ea = (lane < n_here) ? __expf(a - m) : 0.0f;
        const float eb = (lane + 32 < n_here) ? __expf(b - m) : 0.0f;
        sp[h][lane] = ea;
        sp[h][lane + 32] = eb;
        const float l = warp_sum(ea + eb);
        if (lane == 0) { part_m[slot * G + h] = m; part_l[slot * G + h] = l; }
    }
    __syncthreads();
    // values: thread t owns dimension t for all 12 heads.
    float acc[G];
#pragma unroll
    for (int h = 0; h < G; ++h) acc[h] = 0.0f;
    for (int c = 0; c < n_here; ++c) {
        float v;
        if constexpr (!INT8) {
            v = __half2float(__ushort_as_half(p.v_pool[srow[c] * HD + t]));
        } else {
            const float sc = __half2float(__ushort_as_half(p.v_scale[srow[c] * (HD / KV_Q8_GROUP) + t / KV_Q8_GROUP]));
            v = (float) p.v_q[srow[c] * HD + t] * sc;
        }
#pragma unroll
        for (int h = 0; h < G; ++h) acc[h] = fmaf(sp[h][c], v, acc[h]);
    }
#pragma unroll
    for (int h = 0; h < G; ++h) part_acc[((size_t) slot * G + h) * HD + t] = acc[h];
}

__global__ void __launch_bounds__(HD) attn_merge_kernel(const float* __restrict__ part_acc,
                                                        const float* __restrict__ part_m,
                                                        const float* __restrict__ part_l, int n_chunks,
                                                        float* __restrict__ attn, long long scratch_stride = 0) {
    part_acc += (size_t) blockIdx.y * (size_t) scratch_stride;
    part_m += (size_t) blockIdx.y * (size_t) scratch_stride;
    part_l += (size_t) blockIdx.y * (size_t) scratch_stride;
    attn += (size_t) blockIdx.y * (size_t) gridDim.x * HD;
    const int h = blockIdx.x;                 // global query head
    const int kvh = h / G, hl = h % G;
    const int d = threadIdx.x;
    float M = -FLT_MAX;
    for (int c = 0; c < n_chunks; ++c) M = fmaxf(M, part_m[(kvh * n_chunks + c) * G + hl]);
    float L = 0.0f, acc = 0.0f;
    for (int c = 0; c < n_chunks; ++c) {
        const int slot = kvh * n_chunks + c;
        const float m = part_m[slot * G + hl];
        if (m == -FLT_MAX) continue;
        const float w = __expf(m - M);
        L = fmaf(part_l[slot * G + hl], w, L);
        acc = fmaf(part_acc[((size_t) slot * G + hl) * HD + d], w, acc);
    }
    attn[(size_t) h * HD + d] = L > 0.0f ? acc / L : 0.0f;
}

// ---- the prompt path: a block per (KV head, query) walks all of the query's cells in tiles, online softmax
constexpr int PT = 64;   // cells per tile
static_assert(KV_Q8_GROUP == 64, "a score thread's 64 dimensions are one int8 scale group");

// Four int8 codes as floats without I2F (a quarter-rate instruction on sm_86): flip the sign bits, put each byte
// in the mantissa of 2^23 and subtract 2^23 + 128.  Exact.
__device__ __forceinline__ void i8x4(uint32_t w, float* f) {
    const uint32_t u = w ^ 0x80808080u;
#pragma unroll
    for (int j = 0; j < 4; ++j) f[j] = __int_as_float((int) __byte_perm(u, 0x4B000000u, 0x7650u | j)) - 8388736.0f;
}

// Scores: thread (cell, quarter) dots its cell's 64 dimensions of that quarter with the 12 query heads, the quarters
// summed through shared memory.  Values: thread (dimension pair, cell parity) accumulates its two dimensions of the
// 12 heads over every other cell; the parities are added at the end.
template <bool INT8>
__global__ void __launch_bounds__(THREADS) attn_prefill_kernel(const float* __restrict__ q, QsaAttnPools p,
                                                               const int32_t* __restrict__ ids,
                                                               const int32_t* __restrict__ steps, int n_kv_heads,
                                                               int page_size, float scale, float* __restrict__ attn,
                                                               int cap) {
    const int kvh = blockIdx.x;
    const size_t qi = blockIdx.y;
    const size_t head0 = qi * (size_t) (n_kv_heads * G) + (size_t) kvh * G;
    q += head0 * HD;
    attn += head0 * HD;
    ids += qi * (size_t) cap;
    const int n_ids = __ldg(steps + qi * kStepCount + kStepWidth);
    __shared__ __align__(16) float sq[G][HD];      // 12 KB: this KV head's query heads
    __shared__ __align__(16) float spart[4][G][PT];  // 12 KB: the quarters' partial scores; the parity sums at the end
    __shared__ __align__(16) float sp[PT][G];      // probabilities, cell-major
    __shared__ long long srow[PT];
    __shared__ float s_alpha[G], s_m[G], s_l[G];
    const int t = threadIdx.x, lane = t & 31, warp = t >> 5;
    for (int i = t; i < G * HD; i += THREADS) sq[i / HD][i % HD] = q[i];
    if (t < G) { s_m[t] = -FLT_MAX; s_l[t] = 0.0f; }
    const int quarter = warp >> 1, cell = ((warp & 1) << 5) | lane;   // scores
    const int parity = t >> 7, d0 = 2 * (t & 127);                     // values
    float acc0[G], acc1[G];
#pragma unroll
    for (int h = 0; h < G; ++h) acc0[h] = acc1[h] = 0.0f;
    for (int c0 = 0; c0 < n_ids; c0 += PT) {
        const int n_here = min(PT, n_ids - c0);
        if (t < PT) {
            long long r = -1;
            if (t < n_here) {
                const int id = ids[c0 + t];
                r = ((long long) p.page_table[id / page_size] * n_kv_heads + kvh) * page_size + id % page_size;
            }
            srow[t] = r;
        }
        __syncthreads();
        float s[G];
#pragma unroll
        for (int h = 0; h < G; ++h) s[h] = 0.0f;
        if (cell < n_here) {
            const long long row = srow[cell];
#pragma unroll
            for (int j = 0; j < 64; j += 16) {
                const int d = quarter * 64 + j;
                float kf[16];
                if constexpr (INT8) {
                    const uint4 raw = *reinterpret_cast<const uint4*>(p.k_q + row * HD + d);
                    i8x4(raw.x, kf); i8x4(raw.y, kf + 4); i8x4(raw.z, kf + 8); i8x4(raw.w, kf + 12);
                } else {
                    const uint4* src = reinterpret_cast<const uint4*>(p.k_pool + row * HD + d);
#pragma unroll
                    for (int u = 0; u < 2; ++u) {
                        const uint4 raw = src[u];
                        const __half2* h2 = reinterpret_cast<const __half2*>(&raw);
#pragma unroll
                        for (int v = 0; v < 4; ++v) {
                            const float2 f = __half22float2(h2[v]);
                            kf[8 * u + 2 * v] = f.x;
                            kf[8 * u + 2 * v + 1] = f.y;
                        }
                    }
                }
#pragma unroll
                for (int h = 0; h < G; ++h) {
                    const float4* qh = reinterpret_cast<const float4*>(&sq[h][d]);
                    float a = s[h];
#pragma unroll
                    for (int v = 0; v < 4; ++v) {
                        const float4 q4 = qh[v];
                        a = fmaf(kf[4 * v], q4.x, a);
                        a = fmaf(kf[4 * v + 1], q4.y, a);
                        a = fmaf(kf[4 * v + 2], q4.z, a);
                        a = fmaf(kf[4 * v + 3], q4.w, a);
                    }
                    s[h] = a;
                }
            }
            if constexpr (INT8) {
                const float ks = __half2float(__ushort_as_half(p.k_scale[row * (HD / KV_Q8_GROUP) + quarter]));
#pragma unroll
                for (int h = 0; h < G; ++h) s[h] *= ks;
            }
        }
#pragma unroll
        for (int h = 0; h < G; ++h) spart[quarter][h][cell] = s[h];
        __syncthreads();
        // online softmax: warp w takes heads w and w + 8, a lane cells lane and lane + 32
        for (int h = warp; h < G; h += WARPS) {
            float a = -FLT_MAX, b = -FLT_MAX;
            if (lane < n_here) a = (spart[0][h][lane] + spart[1][h][lane] + spart[2][h][lane] + spart[3][h][lane]) * scale;
            if (lane + 32 < n_here)
                b = (spart[0][h][lane + 32] + spart[1][h][lane + 32] + spart[2][h][lane + 32] + spart[3][h][lane + 32]) * scale;
            const float m_old = s_m[h];
            const float m_new = fmaxf(m_old, warp_max(fmaxf(a, b)));
            const float ea = lane < n_here ? __expf(a - m_new) : 0.0f;
            const float eb = lane + 32 < n_here ? __expf(b - m_new) : 0.0f;
            sp[lane][h] = ea;
            sp[lane + 32][h] = eb;
            const float l = warp_sum(ea + eb);
            if (lane == 0) {
                const float alpha = __expf(m_old - m_new);
                s_alpha[h] = alpha;
                s_l[h] = fmaf(s_l[h], alpha, l);
                s_m[h] = m_new;
            }
        }
        __syncthreads();
#pragma unroll
        for (int h = 0; h < G; ++h) {
            acc0[h] *= s_alpha[h];
            acc1[h] *= s_alpha[h];
        }
        for (int c = parity; c < n_here; c += 2) {
            const long long r = srow[c];
            float v0, v1;
            if constexpr (INT8) {
                const uint32_t w = *reinterpret_cast<const uint16_t*>(p.v_q + r * HD + d0) ^ 0x8080u;
                const float sc = __half2float(__ushort_as_half(p.v_scale[r * (HD / KV_Q8_GROUP) + d0 / KV_Q8_GROUP]));
                v0 = (__int_as_float((int) __byte_perm(w, 0x4B000000u, 0x7650u)) - 8388736.0f) * sc;
                v1 = (__int_as_float((int) __byte_perm(w, 0x4B000000u, 0x7651u)) - 8388736.0f) * sc;
            } else {
                const float2 f = __half22float2(*reinterpret_cast<const __half2*>(p.v_pool + r * HD + d0));
                v0 = f.x;
                v1 = f.y;
            }
            const float4* pc = reinterpret_cast<const float4*>(sp[c]);
            const float4 pa = pc[0], pb = pc[1], pd = pc[2];
            const float pr[G] = {pa.x, pa.y, pa.z, pa.w, pb.x, pb.y, pb.z, pb.w, pd.x, pd.y, pd.z, pd.w};
#pragma unroll
            for (int h = 0; h < G; ++h) {
                acc0[h] = fmaf(pr[h], v0, acc0[h]);
                acc1[h] = fmaf(pr[h], v1, acc1[h]);
            }
        }
        __syncthreads();   // srow, sp and s_alpha are rewritten by the next tile
    }
    // the odd cells' sums into shared memory, added to the even ones'
    float* odd = &spart[0][0][0];   // G x HD floats
    if (parity == 1) {
#pragma unroll
        for (int h = 0; h < G; ++h) {
            odd[h * HD + d0] = acc0[h];
            odd[h * HD + d0 + 1] = acc1[h];
        }
    }
    __syncthreads();
    if (parity == 0) {
#pragma unroll
        for (int h = 0; h < G; ++h) {
            const float l = s_l[h];
            attn[(size_t) h * HD + d0] = l > 0.0f ? (acc0[h] + odd[h * HD + d0]) / l : 0.0f;
            attn[(size_t) h * HD + d0 + 1] = l > 0.0f ? (acc1[h] + odd[h * HD + d0 + 1]) / l : 0.0f;
        }
    }
}

}  // namespace

void qsa_prefill_attn(const float* q, const QsaAttnPools& pools, const int32_t* ids, const int32_t* steps, int64_t cap,
                      const QsaShapes& s, float* attn, int64_t n_q, void* stream) {
    if (n_q <= 0) return;
    const bool int8 = pools.k_q != nullptr;
    if (s.head_dim != HD || s.n_head != (int64_t) G * s.n_head_kv || cap <= 0 || !ids || !steps || !pools.page_table ||
        (int8 ? (!pools.v_q || !pools.k_scale || !pools.v_scale) : (!pools.k_pool || !pools.v_pool))) {
        std::fprintf(stderr, "qsa_prefill_attn: unsupported geometry or missing buffers\n");
        std::exit(1);
    }
    const float scale = 1.0f / sqrtf((float) HD);
    cudaStream_t st = (cudaStream_t) stream;
    for (int64_t q0 = 0; q0 < n_q; q0 += 65535) {
        const int64_t nb = n_q - q0 < 65535 ? n_q - q0 : 65535;
        const dim3 grid((unsigned) s.n_head_kv, (unsigned) nb);
        const size_t qo = (size_t) q0 * (size_t) s.n_head * HD;
        if (int8)
            attn_prefill_kernel<true><<<grid, THREADS, 0, st>>>(q + qo, pools, ids + (size_t) q0 * (size_t) cap,
                                                                steps + (size_t) q0 * kStepCount, (int) s.n_head_kv,
                                                                (int) s.page_size, scale, attn + qo, (int) cap);
        else
            attn_prefill_kernel<false><<<grid, THREADS, 0, st>>>(q + qo, pools, ids + (size_t) q0 * (size_t) cap,
                                                                 steps + (size_t) q0 * kStepCount, (int) s.n_head_kv,
                                                                 (int) s.page_size, scale, attn + qo, (int) cap);
    }
    const cudaError_t e = cudaGetLastError();
    if (e != cudaSuccess) {
        std::fprintf(stderr, "qsa_prefill_attn: %s\n", cudaGetErrorString(e));
        std::exit(1);
    }
}

void qsa_decode_attn_batch(const float* q, const QsaAttnPools& pools, const int32_t* ids, const int32_t* steps,
                           int64_t cap, const QsaShapes& s, float* scratch, float* attn, int64_t n_q, void* stream) {
    if (n_q <= 0) return;
    if (s.head_dim != HD || s.n_head != (int64_t) G * s.n_head_kv || cap <= 0 || !scratch || !ids || !steps ||
        !pools.page_table || n_q > 65535) {
        std::fprintf(stderr, "qsa_decode_attn_batch: unsupported geometry or missing buffers\n");
        std::exit(1);
    }
    const bool int8 = pools.k_q != nullptr;
    const int n_chunks = (int) ((cap + CHUNK - 1) / CHUNK);
    // per query: [acc: n_chunks*n_head*HD][m: n_chunks*n_head][l: n_chunks*n_head], all offsets from one stride
    const long long stride = (long long) qsa_decode_attn_scratch_floats(cap, s);
    float* part_acc = scratch;
    float* part_m = scratch + (size_t) n_chunks * s.n_head * HD;
    float* part_l = part_m + (size_t) n_chunks * s.n_head;
    const float scale = 1.0f / sqrtf((float) HD);
    const dim3 grid((unsigned) n_chunks, (unsigned) s.n_head_kv, (unsigned) n_q);
    cudaStream_t st = (cudaStream_t) stream;
    if (int8)
        attn_chunk_kernel<true><<<grid, THREADS, 0, st>>>(q, pools, ids, steps, (int) s.n_head_kv, (int) s.page_size,
                                                          scale, part_acc, part_m, part_l, n_chunks, (int) cap, stride);
    else
        attn_chunk_kernel<false><<<grid, THREADS, 0, st>>>(q, pools, ids, steps, (int) s.n_head_kv, (int) s.page_size,
                                                           scale, part_acc, part_m, part_l, n_chunks, (int) cap, stride);
    attn_merge_kernel<<<dim3((unsigned) s.n_head, (unsigned) n_q), HD, 0, st>>>(part_acc, part_m, part_l, n_chunks,
                                                                                  attn, stride);
    const cudaError_t e = cudaGetLastError();
    if (e != cudaSuccess) {
        std::fprintf(stderr, "qsa_decode_attn_batch: %s\n", cudaGetErrorString(e));
        std::exit(1);
    }
}

uint64_t qsa_decode_attn_scratch_floats(int64_t cap, const QsaShapes& s) {
    const int64_t chunks = (cap + CHUNK - 1) / CHUNK;
    return (uint64_t) chunks * (uint64_t) s.n_head * (HD + 2) + 64;
}

void qsa_decode_attn_step(const float* q, const QsaAttnPools& pools, const int32_t* ids, const int32_t* step,
                          int64_t cap, const QsaShapes& s, float* scratch, float* attn, void* stream) {
    if (s.head_dim != HD || s.n_head != (int64_t) G * s.n_head_kv || cap <= 0 || !scratch || !ids || !step ||
        !pools.page_table) {
        std::fprintf(stderr, "qsa_decode_attn: unsupported geometry or missing buffers\n");
        std::exit(1);
    }
    const bool int8 = pools.k_q != nullptr;
    if (int8 ? (!pools.v_q || !pools.k_scale || !pools.v_scale) : (!pools.k_pool || !pools.v_pool)) {
        std::fprintf(stderr, "qsa_decode_attn: incomplete KV pools\n");
        std::exit(1);
    }
    const int n_chunks = (int) ((cap + CHUNK - 1) / CHUNK);
    float* part_acc = scratch;
    float* part_m = scratch + (size_t) n_chunks * s.n_head * HD;
    float* part_l = part_m + (size_t) n_chunks * s.n_head;
    const float scale = 1.0f / sqrtf((float) HD);
    const dim3 grid((unsigned) n_chunks, (unsigned) s.n_head_kv);
    cudaStream_t st = (cudaStream_t) stream;
    if (int8)
        attn_chunk_kernel<true><<<grid, THREADS, 0, st>>>(q, pools, ids, step, (int) s.n_head_kv, (int) s.page_size,
                                                          scale, part_acc, part_m, part_l, n_chunks);
    else
        attn_chunk_kernel<false><<<grid, THREADS, 0, st>>>(q, pools, ids, step, (int) s.n_head_kv, (int) s.page_size,
                                                           scale, part_acc, part_m, part_l, n_chunks);
    attn_merge_kernel<<<(unsigned) s.n_head, HD, 0, st>>>(part_acc, part_m, part_l, n_chunks, attn);
    const cudaError_t e = cudaGetLastError();
    if (e != cudaSuccess) {
        std::fprintf(stderr, "qsa_decode_attn: %s\n", cudaGetErrorString(e));
        std::exit(1);
    }
}

}  // namespace strata::kernels
