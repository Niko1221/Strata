// src/kernels/cuda/qsa_pattn.cu - see include/strata/kernels/qsa_pattn.hpp.
//
// Every value below is computed by the operation the production path uses, on the same operands, in the same order:
//   - score of (cell, head): each lane l's partial k8[0] * q[8l] + ... + k8[7] * q[8l + 7] with k8[e] = (float) code * sc
//     (load8_q8), the 32 partials summed by reduce12 (warp_sum's pairs), times `scale`;
//   - chunk softmax: m = warp_max over fmaxf(s[lane], s[lane + 32]); e = __expf(s - m) (0 for a masked or absent cell);
//     l = warp_sum(e[lane] + e[lane + 32]);
//   - chunk value: acc[h] = fmaf(e[c][h], v, acc[h]) over the chunk's cells in ascending order, masked cells skipped,
//     v = (float) code * sc (load_v1);
//   - merge (attn_merge_kernel / attn_merge_v2_kernel): M = fmaxf over the chunks' m; for each chunk in order with
//     m != -FLT_MAX: w = __expf(m - M), L = fmaf(l, w, L), acc = fmaf(acc_c, w, acc); out = L > 0 ? acc / L : 0.
// pattn_bench memcmps the outputs against qsa_decode_attn_batch's on recorded selections.
#include "strata/kernels/qsa_pattn.hpp"
#include "strata/kernels/kv_q8.hpp"
#include "strata/core/geometry.hpp"

#include <cuda_fp16.h>
#include <cuda_runtime.h>

#include <cfloat>
#include <cstdio>
#include <cstdlib>
#include <cstring>

namespace strata::kernels {

int pattn_mode() {
    static const int m = [] {
        const char* v = std::getenv("STRATA_PATTN_SM60");
        const int x = v != nullptr && *v != '\0' ? std::atoi(v) : 0;
        if (x != 0)
            std::fprintf(stderr, "strata pattn: STRATA_PATTN_SM60=%d: the prompt path's QSA attention on the sm_6x score / "
                                 "value kernels (bit for bit qsa_decode_attn_batch; int8 K/V)\n", x);
        return x != 0 ? 1 : 0;
    }();
    return m;
}

#if !defined(__HIPCC__) && defined(STRATA_EXPERIMENTAL_SM60)
namespace {

constexpr int HD = 256;
constexpr int G = 12;
static_assert(HD == strata::core::kCompiledGeometry.head_dim &&
              G == strata::core::kCompiledGeometry.n_head / strata::core::kCompiledGeometry.n_head_kv,
              "qsa_pattn is compiled for core/geometry.hpp");
constexpr int CHUNK = 64;
constexpr int THREADS = 256;
constexpr int WARPS = THREADS / 32;
constexpr int NSC = HD / KV_Q8_GROUP;   // scales per row (4)
constexpr int NCMAX = 40;                // chunks a fused CTA keeps m / l for (cap <= 2560)

__device__ __forceinline__ float warp_sum_x(float v) {
    v += __shfl_xor_sync(0xffffffffu, v, 16); v += __shfl_xor_sync(0xffffffffu, v, 8);
    v += __shfl_xor_sync(0xffffffffu, v, 4); v += __shfl_xor_sync(0xffffffffu, v, 2);
    v += __shfl_xor_sync(0xffffffffu, v, 1);
    return v;
}
__device__ __forceinline__ float warp_max_x(float v) {
    v = fmaxf(v, __shfl_xor_sync(0xffffffffu, v, 16)); v = fmaxf(v, __shfl_xor_sync(0xffffffffu, v, 8));
    v = fmaxf(v, __shfl_xor_sync(0xffffffffu, v, 4)); v = fmaxf(v, __shfl_xor_sync(0xffffffffu, v, 2));
    v = fmaxf(v, __shfl_xor_sync(0xffffffffu, v, 1));
    return v;
}
// qsa_decode_attn.cu's reduce12<false> / head_of_lane, verbatim
__device__ __forceinline__ float reduce12(const float (&part)[16], int lane) {
    float r8[8];
#pragma unroll
    for (int i = 0; i < 8; ++i) {
        const bool hi = (lane & 16) != 0;
        r8[i] = (hi ? part[i + 8] : part[i]) + __shfl_xor_sync(0xffffffffu, hi ? part[i] : part[i + 8], 16);
    }
    float r4[4];
#pragma unroll
    for (int i = 0; i < 4; ++i) {
        const bool hi = (lane & 8) != 0;
        r4[i] = (hi ? r8[i + 4] : r8[i]) + __shfl_xor_sync(0xffffffffu, hi ? r8[i] : r8[i + 4], 8);
    }
    float r2[2];
#pragma unroll
    for (int i = 0; i < 2; ++i) {
        const bool hi = (lane & 4) != 0;
        r2[i] = (hi ? r4[i + 2] : r4[i]) + __shfl_xor_sync(0xffffffffu, hi ? r4[i] : r4[i + 2], 4);
    }
    const bool hi2 = (lane & 2) != 0;
    const float r1 = (hi2 ? r2[1] : r2[0]) + __shfl_xor_sync(0xffffffffu, hi2 ? r2[0] : r2[1], 2);
    return r1 + __shfl_xor_sync(0xffffffffu, r1, 1);
}
__device__ __forceinline__ int head_of_lane(int lane) {
    return ((lane >> 4) & 1) * 8 + ((lane >> 3) & 1) * 4 + ((lane >> 2) & 1) * 2 + ((lane >> 1) & 1);
}
__device__ __forceinline__ float load_v1_q8(const QsaAttnPools& p, long long row, int d) {
    const float sc = __half2float(__ushort_as_half(p.v_scale[row * NSC + d / KV_Q8_GROUP]));
    return (float) p.v_q[row * HD + d] * sc;
}

// pool row of cell `c0 + t` of query selection `ids` (attn_chunk_kernel_pre75's srow), -1 masked / absent
__device__ __forceinline__ long long cell_row(const QsaAttnPools& p, const int32_t* __restrict__ ids, int c0, int t,
                                              int n_here, int n_kv_heads, int kvh, int page_size) {
    long long r = -1;
    if (t < n_here) {
        const int cell = __ldg(ids + c0 + t);
        const long long page = (long long) __ldg(p.page_table + cell / page_size);
        if (page >= 0) r = (page * n_kv_heads + kvh) * page_size + (cell % page_size);
    }
    return r;
}

// the q slice of lane `lane` into registers: dims [8 lane, 8 lane + 8) of the KV head's 12 query heads
__device__ __forceinline__ void load_q(const float* __restrict__ qh, int lane, float4 (&qa)[G], float4 (&qb)[G]) {
#pragma unroll
    for (int h = 0; h < G; ++h) {
        const float4* p4 = reinterpret_cast<const float4*>(qh + (size_t) h * HD + lane * 8);
        qa[h] = __ldg(p4);
        qb[h] = __ldg(p4 + 1);
    }
}

// the chunk's K codes (16-byte loads) and scales (as float) into shared memory; masked / absent cells read row 0
__device__ __forceinline__ void stage_rows(const int8_t* __restrict__ codes, const uint16_t* __restrict__ scales,
                                           const long long* srow, int n_here, int t, int8_t (*sk)[HD],
                                           float (*sks)[NSC]) {
    uint4 kv[4];
#pragma unroll
    for (int j = 0; j < 4; ++j) {
        const int idx = t + THREADS * j, c = idx >> 4, part = idx & 15;
        const long long row = srow[c];
        const long long rr = (c < n_here && row >= 0) ? row : 0;
        kv[j] = __ldg(reinterpret_cast<const uint4*>(codes + rr * HD) + part);
    }
    float sc;
    {
        const int c = t >> 2, g = t & 3;
        const long long row = srow[c];
        const long long rr = (c < n_here && row >= 0) ? row : 0;
        sc = __half2float(__ushort_as_half(__ldg(scales + rr * NSC + g)));
    }
#pragma unroll
    for (int j = 0; j < 4; ++j) reinterpret_cast<uint4*>(&sk[0][0])[t + THREADS * j] = kv[j];
    sks[t >> 2][t & 3] = sc;
}

// one cell's 12 lane partials (part[12..15] = 0): attn_chunk_kernel_pre75's expression
template <bool KMAGIC>
__device__ __forceinline__ void cell_partials(const float4 (&qa)[G], const float4 (&qb)[G], const int8_t* krow,
                                              float sc, int lane, float (&part)[16]) {
    const uint2 raw = *reinterpret_cast<const uint2*>(krow + lane * 8);
    float k8[8];
    if constexpr (KMAGIC) {
        const uint32_t u0 = raw.x ^ 0x80808080u, u1 = raw.y ^ 0x80808080u;
#pragma unroll
        for (int e = 0; e < 4; ++e) {
            k8[e] = (__uint_as_float(__byte_perm(u0, 0x4B00u, 0x5440u | (unsigned) e)) - 8388736.0f) * sc;
            k8[e + 4] = (__uint_as_float(__byte_perm(u1, 0x4B00u, 0x5440u | (unsigned) e)) - 8388736.0f) * sc;
        }
    } else {
        const int8_t* cb = reinterpret_cast<const int8_t*>(&raw);
#pragma unroll
        for (int e = 0; e < 8; ++e) k8[e] = (float) cb[e] * sc;
    }
#pragma unroll
    for (int h = 0; h < G; ++h)
        part[h] = k8[0] * qa[h].x + k8[1] * qa[h].y + k8[2] * qa[h].z + k8[3] * qa[h].w +
                  k8[4] * qb[h].x + k8[5] * qb[h].y + k8[6] * qb[h].z + k8[7] * qb[h].w;
#pragma unroll
    for (int h = G; h < 16; ++h) part[h] = 0.0f;
}

// Two cells' reduce12 in one reduce-scatter: every one of the 24 sums is still warp_sum's tree (at each level o = 16,
// 8, 4, 2, 1 the value at lane L is its partial at L plus its partial at L ^ o), for 25 shuffles instead of 32.  Lane
// bit 4 picks the cell (0: A, 1: B); the heads: o0 is head 6 b3 + 3 b2 + 2 b1, o1 head 6 b3 + 3 b2 + 1 (b1 = 0 only).
__device__ __forceinline__ void reduce24(const float (&A)[16], const float (&B)[16], int lane, float& o0, float& o1) {
    const bool h16 = (lane & 16) != 0;
    float r[12];
#pragma unroll
    for (int i = 0; i < 12; ++i) r[i] = (h16 ? B[i] : A[i]) + __shfl_xor_sync(0xffffffffu, h16 ? A[i] : B[i], 16);
    const bool h8 = (lane & 8) != 0;
    float r6[6];
#pragma unroll
    for (int i = 0; i < 6; ++i) r6[i] = (h8 ? r[i + 6] : r[i]) + __shfl_xor_sync(0xffffffffu, h8 ? r[i] : r[i + 6], 8);
    const bool h4 = (lane & 4) != 0;
    float r3[3];
#pragma unroll
    for (int i = 0; i < 3; ++i) r3[i] = (h4 ? r6[i + 3] : r6[i]) + __shfl_xor_sync(0xffffffffu, h4 ? r6[i] : r6[i + 3], 4);
    const bool h2 = (lane & 2) != 0;
    const float x0 = (h2 ? r3[2] : r3[0]) + __shfl_xor_sync(0xffffffffu, h2 ? r3[0] : r3[2], 2);
    const float x1 = r3[1] + __shfl_xor_sync(0xffffffffu, r3[1], 2);   // meaningful where b1 = 0
    o0 = x0 + __shfl_xor_sync(0xffffffffu, x0, 1);
    o1 = x1 + __shfl_xor_sync(0xffffffffu, x1, 1);
}

// scores of the chunk's cells warp, warp + 8, ... (attn_chunk_kernel_pre75's score phase), K from the staged tile, q
// from registers.  KMAGIC: each code converted as (bits 0x4B0000xx with xx = code + 128) - 8388736.0f instead of I2F -
// the same float, exactly (both are the integer `code`), without the quarter-rate conversion pipe.  RED24: cells in
// pairs through reduce24 (a pair with a masked or absent cell goes one cell at a time).
template <bool KMAGIC, bool RED24>
__device__ __forceinline__ void score_chunk(const float4 (&qa)[G], const float4 (&qb)[G], const int8_t (*sk)[HD],
                                            const float (*sks)[NSC], const long long* srow, float (*sp)[CHUNK],
                                            int n_here, float scale, int warp, int lane) {
    const int hd = head_of_lane(lane);
    const auto one = [&](int c) {
        if (c >= n_here || srow[c] < 0) {
            if (lane < G) sp[lane][c] = -FLT_MAX;
            return;
        }
        float part[16];
        cell_partials<KMAGIC>(qa, qb, sk[c], sks[c][lane >> 3], lane, part);
        const float sc = reduce12(part, lane);
        if ((lane & 1) == 0 && hd < G) sp[hd][c] = sc * scale;
    };
    if constexpr (RED24) {
        const int hb = 6 * ((lane >> 3) & 1) + 3 * ((lane >> 2) & 1);
        const int h0 = hb + ((lane & 2) ? 2 : 0), h1 = hb + 1;
#pragma unroll 1
        for (int i = 0; i < CHUNK / WARPS; i += 2) {
            const int ca = warp + WARPS * i, cb = warp + WARPS * (i + 1);
            if (ca < n_here && cb < n_here && srow[ca] >= 0 && srow[cb] >= 0) {
                float pa[16], pb[16];
                cell_partials<KMAGIC>(qa, qb, sk[ca], sks[ca][lane >> 3], lane, pa);
                cell_partials<KMAGIC>(qa, qb, sk[cb], sks[cb][lane >> 3], lane, pb);
                float o0, o1;
                reduce24(pa, pb, lane, o0, o1);
                const int c = (lane & 16) ? cb : ca;
                if ((lane & 1) == 0) {
                    sp[h0][c] = o0 * scale;
                    if ((lane & 2) == 0) sp[h1][c] = o1 * scale;
                }
            } else {
                one(ca);
                one(cb);
            }
        }
    } else {
#pragma unroll 2
        for (int i = 0; i < CHUNK / WARPS; ++i) one(warp + WARPS * i);
    }
}

// one cell's 12 weights into the 12 accumulators (attn_chunk_kernel_pre75's fold)
__device__ __forceinline__ void fold(float (&acc)[G], const float* w, float v) {
    const float4* w4 = reinterpret_cast<const float4*>(w);
    const float4 w0 = w4[0], w1 = w4[1], w2 = w4[2];
    acc[0] = fmaf(w0.x, v, acc[0]);  acc[1] = fmaf(w0.y, v, acc[1]);
    acc[2] = fmaf(w0.z, v, acc[2]);  acc[3] = fmaf(w0.w, v, acc[3]);
    acc[4] = fmaf(w1.x, v, acc[4]);  acc[5] = fmaf(w1.y, v, acc[5]);
    acc[6] = fmaf(w1.z, v, acc[6]);  acc[7] = fmaf(w1.w, v, acc[7]);
    acc[8] = fmaf(w2.x, v, acc[8]);  acc[9] = fmaf(w2.y, v, acc[9]);
    acc[10] = fmaf(w2.z, v, acc[10]); acc[11] = fmaf(w2.w, v, acc[11]);
}

// the value fold of one chunk, V read from global memory four cells at a time (attn_chunk_kernel_pre75's loop)
__device__ __forceinline__ void value_chunk_global(const QsaAttnPools& p, const float (*spt)[G], const long long* srow,
                                                   int n_here, int t, float (&acc)[G]) {
#pragma unroll
    for (int h = 0; h < G; ++h) acc[h] = 0.0f;
    int c = 0;
    for (; c + 4 <= n_here; c += 4) {
        const long long r0 = srow[c], r1 = srow[c + 1], r2 = srow[c + 2], r3 = srow[c + 3];
        if ((r0 | r1 | r2 | r3) < 0) break;
        const float v0 = load_v1_q8(p, r0, t), v1 = load_v1_q8(p, r1, t);
        const float v2 = load_v1_q8(p, r2, t), v3 = load_v1_q8(p, r3, t);
        fold(acc, spt[c], v0);
        fold(acc, spt[c + 1], v1);
        fold(acc, spt[c + 2], v2);
        fold(acc, spt[c + 3], v3);
    }
    for (; c < n_here; ++c) {
        if (srow[c] < 0) continue;
        fold(acc, spt[c], load_v1_q8(p, srow[c], t));
    }
}

// the same fold with the chunk's V codes / scales staged in shared memory
__device__ __forceinline__ void value_chunk_staged(const int8_t (*sv)[HD], const float (*svs)[NSC], const float (*spt)[G],
                                                   const long long* srow, int n_here, int t, float (&acc)[G]) {
#pragma unroll
    for (int h = 0; h < G; ++h) acc[h] = 0.0f;
    const int g = t / KV_Q8_GROUP;
#pragma unroll 4
    for (int c = 0; c < n_here; ++c) {
        if (srow[c] < 0) continue;
        fold(acc, spt[c], (float) sv[c][t] * svs[c][g]);
    }
}

// the fold with two dimensions a thread (2u, 2u + 1: one 16-bit load of codes, one scale - both dims are in one
// 64-value scale group), each cell's 12 weights read once for both
__device__ __forceinline__ void value_chunk_dim2(const QsaAttnPools& p, const float (*spt)[G], const long long* srow,
                                                 int n_here, int u, float (&a0)[G], float (&a1)[G]) {
#pragma unroll
    for (int h = 0; h < G; ++h) { a0[h] = 0.0f; a1[h] = 0.0f; }
    const auto vload = [&](long long r, float& v0, float& v1) {
        const float sc = __half2float(__ushort_as_half(p.v_scale[r * NSC + (2 * u) / KV_Q8_GROUP]));
        const uint16_t two = *reinterpret_cast<const uint16_t*>(p.v_q + r * HD + 2 * u);
        v0 = (float) (int8_t) (two & 0xFF) * sc;
        v1 = (float) (int8_t) (two >> 8) * sc;
    };
    const auto fold2 = [&](const float* w, float v0, float v1) {
        const float4* w4 = reinterpret_cast<const float4*>(w);
        const float4 w0 = w4[0], w1 = w4[1], w2 = w4[2];
        const float ww[G] = {w0.x, w0.y, w0.z, w0.w, w1.x, w1.y, w1.z, w1.w, w2.x, w2.y, w2.z, w2.w};
#pragma unroll
        for (int h = 0; h < G; ++h) { a0[h] = fmaf(ww[h], v0, a0[h]); a1[h] = fmaf(ww[h], v1, a1[h]); }
    };
    int c = 0;
    for (; c + 4 <= n_here; c += 4) {
        const long long r0 = srow[c], r1 = srow[c + 1], r2 = srow[c + 2], r3 = srow[c + 3];
        if ((r0 | r1 | r2 | r3) < 0) break;
        float x0, y0, x1, y1, x2, y2, x3, y3;
        vload(r0, x0, y0); vload(r1, x1, y1); vload(r2, x2, y2); vload(r3, x3, y3);
        fold2(spt[c], x0, y0);
        fold2(spt[c + 1], x1, y1);
        fold2(spt[c + 2], x2, y2);
        fold2(spt[c + 3], x3, y3);
    }
    for (; c < n_here; ++c) {
        if (srow[c] < 0) continue;
        float x, y;
        vload(srow[c], x, y);
        fold2(spt[c], x, y);
    }
}

// ---------------------------------------------------------------------------------------------------------------------
// variant 1: one CTA per (chunk, KV head, query) as attn_chunk_kernel_pre75, q in registers, K staged; partials to the
// scratch for attn_merge_v2_kernel
template <int MINB>
__global__ void __launch_bounds__(THREADS, MINB) pa_chunk_kernel(const float* __restrict__ q, QsaAttnPools p,
                                                                 const int32_t* __restrict__ ids,
                                                                 const int32_t* __restrict__ step, int n_kv_heads,
                                                                 int page_size, float scale, float* __restrict__ part_acc,
                                                                 float* __restrict__ part_m, float* __restrict__ part_l,
                                                                 int n_chunks, int cap, long long scratch_stride) {
    q += (size_t) blockIdx.z * (size_t) (n_kv_heads * G) * HD;
    ids += (size_t) blockIdx.z * (size_t) cap;
    step += (size_t) blockIdx.z * kStepCount;
    part_acc += (size_t) blockIdx.z * (size_t) scratch_stride;
    part_m += (size_t) blockIdx.z * (size_t) scratch_stride;
    part_l += (size_t) blockIdx.z * (size_t) scratch_stride;
    __shared__ __align__(16) int8_t sk[CHUNK][HD];   // 16 KB: K codes, then the probabilities (cell-major)
    __shared__ __align__(16) float sks[CHUNK][NSC];
    __shared__ float sp[G][CHUNK];
    __shared__ long long srow[CHUNK];
    float (*const spt)[G] = reinterpret_cast<float (*)[G]>(&sk[0][0]);
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
    if (t < CHUNK) srow[t] = cell_row(p, ids, c0, t, n_here, n_kv_heads, kvh, page_size);
    float4 qa[G], qb[G];
    load_q(q + (size_t) (kvh * G) * HD, lane, qa, qb);
    __syncthreads();
    stage_rows(p.k_q, p.k_scale, srow, n_here, t, sk, sks);
    __syncthreads();
    score_chunk<false, false>(qa, qb, sk, sks, srow, sp, n_here, scale, warp, lane);
    __syncthreads();
    for (int h = warp; h < G; h += WARPS) {
        const float a = sp[h][lane], b = sp[h][lane + 32];
        const float m = warp_max_x(fmaxf(a, b));
        const float ea = (lane < n_here && srow[lane] >= 0) ? __expf(a - m) : 0.0f;
        const float eb = (lane + 32 < n_here && srow[lane + 32] >= 0) ? __expf(b - m) : 0.0f;
        spt[lane][h] = ea;
        spt[lane + 32][h] = eb;
        const float l = warp_sum_x(ea + eb);
        if (lane == 0) { part_m[slot * G + h] = m; part_l[slot * G + h] = l; }
    }
    __syncthreads();
    float acc[G];
    value_chunk_global(p, spt, srow, n_here, t, acc);
#pragma unroll
    for (int h = 0; h < G; ++h) part_acc[((size_t) slot * G + h) * HD + t] = acc[h];
}

// attn_merge_v2_kernel<16> (qsa_decode_attn.cu), verbatim
constexpr int kMergeU = 16;
template <int U, bool TAIL>
__device__ __forceinline__ void merge_max_block(const float* __restrict__ pm, int c0, int n_act, float& M) {
    float mm[U];
#pragma unroll
    for (int u = 0; u < U; ++u) mm[u] = (!TAIL || c0 + u < n_act) ? pm[(size_t) (c0 + u) * G] : -FLT_MAX;
#pragma unroll
    for (int u = 0; u < U; ++u)
        if (!TAIL || c0 + u < n_act) M = fmaxf(M, mm[u]);
}
template <int U, bool TAIL>
__device__ __forceinline__ void merge_sum_block(const float* __restrict__ pm, const float* __restrict__ pl,
                                                const float* __restrict__ pa, int c0, int n_act, float M, float& L,
                                                float& acc) {
    float mm[U], ll[U], aa[U];
#pragma unroll
    for (int u = 0; u < U; ++u) {
        const bool in = !TAIL || c0 + u < n_act;
        mm[u] = in ? pm[(size_t) (c0 + u) * G] : -FLT_MAX;
        ll[u] = in ? pl[(size_t) (c0 + u) * G] : 0.0f;
        aa[u] = in ? pa[(size_t) (c0 + u) * G * HD] : 0.0f;
    }
#pragma unroll
    for (int u = 0; u < U; ++u) {
        const float w = __expf(mm[u] - M);
        const float L_next = fmaf(ll[u], w, L);
        const float acc_next = fmaf(aa[u], w, acc);
        const bool live = !(mm[u] == -FLT_MAX);
        L = live ? L_next : L;
        acc = live ? acc_next : acc;
    }
}
template <int U>
__global__ void __launch_bounds__(HD) pa_merge_v2_kernel(const float* __restrict__ part_acc,
                                                         const float* __restrict__ part_m,
                                                         const float* __restrict__ part_l, int n_chunks,
                                                         const int32_t* __restrict__ steps, float* __restrict__ attn,
                                                         long long scratch_stride) {
    part_acc += (size_t) blockIdx.y * (size_t) scratch_stride;
    part_m += (size_t) blockIdx.y * (size_t) scratch_stride;
    part_l += (size_t) blockIdx.y * (size_t) scratch_stride;
    attn += (size_t) blockIdx.y * (size_t) gridDim.x * HD;
    const int width = __ldg(steps + (size_t) blockIdx.y * kStepCount + kStepWidth);
    const int n_act = width > 0 ? min(n_chunks, (width - 1) / CHUNK + 1) : 0;
    const int h = blockIdx.x;
    const int kvh = h / G, hl = h % G;
    const int d = threadIdx.x;
    const size_t base = (size_t) kvh * (size_t) n_chunks * G + (size_t) hl;
    const float* pm = part_m + base;
    const float* pl = part_l + base;
    const float* pa = part_acc + base * HD + (size_t) d;
    const int n_full = n_act - n_act % U;
    float M = -FLT_MAX;
    for (int c0 = 0; c0 < n_full; c0 += U) merge_max_block<U, false>(pm, c0, n_act, M);
    if (n_full < n_act) merge_max_block<U, true>(pm, n_full, n_act, M);
    float L = 0.0f, acc = 0.0f;
    for (int c0 = 0; c0 < n_full; c0 += U) merge_sum_block<U, false>(pm, pl, pa, c0, n_act, M, L, acc);
    if (n_full < n_act) merge_sum_block<U, true>(pm, pl, pa, n_full, n_act, M, L, acc);
    attn[(size_t) h * HD + d] = L > 0.0f ? acc / L : 0.0f;
}

// ---------------------------------------------------------------------------------------------------------------------
// variants 2 / 3: fused.  Persistent CTAs; CTA b takes work items (query, KV head) b, b + gridDim.x, ... and owns
// scratch slot b: the probabilities of all chunks (head-major per chunk: [chunk][head][64]) and the chunk rows.
// Pass 1, chunk by chunk: rows, K staged, scores, softmax -> slot (e), shared (m, l).  Then M per head.  Pass 2, chunk
// by chunk in ascending order: rows + probabilities back (cell-major in shared memory), the value fold, and the merge
// step of attn_merge_kernel for that chunk.  VMODE 0: V read from global memory (one dim a thread, the production
// loop); 1: pass 2 stages the V codes / scales like pass 1 stages K; 2: two chunks at a time, two dims a thread (threads
// 0-127 chunk c, 128-255 chunk c + 1, whose partials go through shared memory to their merge in chunk order).
template <int MINB, int VMODE, bool KMAGIC, bool RED24>
__global__ void __launch_bounds__(THREADS, MINB) pf_fused_kernel(const float* __restrict__ q, QsaAttnPools p,
                                                                 const int32_t* __restrict__ ids,
                                                                 const int32_t* __restrict__ steps, int n_kv_heads,
                                                                 int page_size, float scale, float* slots,
                                                                 long long slot_floats, int n_chunks, int cap, int n_q,
                                                                 float* __restrict__ attn) {
    __shared__ __align__(16) int8_t sk[CHUNK][HD];   // pass 1: K codes; pass 2: V codes (VSTAGE)
    __shared__ __align__(16) float sks[CHUNK][NSC];  // their scales
    __shared__ __align__(16) float sp[2 * G * CHUNK];   // pass 1: scores [G][CHUNK]; pass 2: probabilities [.][CHUNK][G]
    __shared__ long long srow[2 * CHUNK];
    __shared__ float sm_m[NCMAX][G], sm_l[NCMAX][G];
    float (*const sps)[CHUNK] = reinterpret_cast<float (*)[CHUNK]>(sp);
    float (*const spt)[G] = reinterpret_cast<float (*)[G]>(sp);
    const int t = threadIdx.x, lane = t & 31, warp = t >> 5;
    float* const pb = slots + (size_t) blockIdx.x * (size_t) slot_floats;          // [n_chunks][G][CHUNK]
    long long* const prow = reinterpret_cast<long long*>(pb + (size_t) n_chunks * G * CHUNK);   // [n_chunks][CHUNK]
    const int n_items = n_q * n_kv_heads;
    for (int item = blockIdx.x; item < n_items; item += gridDim.x) {
        const int z = item / n_kv_heads, kvh = item % n_kv_heads;
        const int32_t* idz = ids + (size_t) z * cap;
        const int width = __ldg(steps + (size_t) z * kStepCount + kStepWidth);
        const int n_act = width > 0 ? min(n_chunks, (width - 1) / CHUNK + 1) : 0;
        float4 qa[G], qb[G];
        load_q(q + ((size_t) z * n_kv_heads + kvh) * G * HD, lane, qa, qb);
        // ---- pass 1
        for (int c = 0; c < n_act; ++c) {
            const int c0 = c * CHUNK, n_here = min(CHUNK, width - c0);
            __syncthreads();   // the previous chunk's (or item's) readers of srow / sk / sp are done
            if (t < CHUNK) {
                const long long r = cell_row(p, idz, c0, t, n_here, n_kv_heads, kvh, page_size);
                srow[t] = r;
                prow[(size_t) c * CHUNK + t] = r;
            }
            __syncthreads();
            stage_rows(p.k_q, p.k_scale, srow, n_here, t, sk, sks);
            __syncthreads();
            score_chunk<KMAGIC, RED24>(qa, qb, sk, sks, srow, sps, n_here, scale, warp, lane);
            __syncthreads();
            for (int h = warp; h < G; h += WARPS) {
                const float a = sps[h][lane], b = sps[h][lane + 32];
                const float m = warp_max_x(fmaxf(a, b));
                const float ea = (lane < n_here && srow[lane] >= 0) ? __expf(a - m) : 0.0f;
                const float eb = (lane + 32 < n_here && srow[lane + 32] >= 0) ? __expf(b - m) : 0.0f;
                float* pe = pb + ((size_t) c * G + h) * CHUNK;
                pe[lane] = ea;
                pe[lane + 32] = eb;
                const float l = warp_sum_x(ea + eb);
                if (lane == 0) { sm_m[c][h] = m; sm_l[c][h] = l; }
            }
        }
        __syncthreads();   // m / l complete; the slot's probabilities and rows visible to the whole CTA
        float M[G];
#pragma unroll
        for (int h = 0; h < G; ++h) M[h] = -FLT_MAX;
        for (int c = 0; c < n_act; ++c)
#pragma unroll
            for (int h = 0; h < G; ++h) M[h] = fmaxf(M[h], sm_m[c][h]);
        float acc[G], L[G];
#pragma unroll
        for (int h = 0; h < G; ++h) { acc[h] = 0.0f; L[h] = 0.0f; }
        // ---- pass 2
        if constexpr (VMODE != 2) {
            for (int c = 0; c < n_act; ++c) {
                const int c0 = c * CHUNK, n_here = min(CHUNK, width - c0);
                __syncthreads();   // the previous chunk's readers of srow / spt / sk are done
                if (t < CHUNK) srow[t] = prow[(size_t) c * CHUNK + t];
#pragma unroll
                for (int j = 0; j < 3; ++j) {
                    const int idx = t + THREADS * j, h = idx / CHUNK, cell = idx % CHUNK;
                    spt[cell][h] = pb[(size_t) c * G * CHUNK + idx];
                }
                __syncthreads();
                float a[G];
                if constexpr (VMODE == 1) {
                    stage_rows(p.v_q, p.v_scale, srow, n_here, t, sk, sks);
                    __syncthreads();
                    value_chunk_staged(sk, sks, spt, srow, n_here, t, a);
                } else {
                    value_chunk_global(p, spt, srow, n_here, t, a);
                }
#pragma unroll
                for (int h = 0; h < G; ++h) {
                    const float m = sm_m[c][h];
                    const float w = __expf(m - M[h]);
                    const float L_next = fmaf(sm_l[c][h], w, L[h]);
                    const float acc_next = fmaf(a[h], w, acc[h]);
                    const bool live = !(m == -FLT_MAX);
                    L[h] = live ? L_next : L[h];
                    acc[h] = live ? acc_next : acc[h];
                }
            }
            float* out = attn + ((size_t) z * n_kv_heads + kvh) * G * HD;
#pragma unroll
            for (int h = 0; h < G; ++h) out[(size_t) h * HD + t] = L[h] > 0.0f ? acc[h] / L[h] : 0.0f;
        } else {
            const int grp = t >> 7, u = t & 127;
            float (*const xch)[2 * G] = reinterpret_cast<float (*)[2 * G]>(&sk[0][0]);   // 128 x 24 floats
            float (*const spt2)[CHUNK][G] = reinterpret_cast<float (*)[CHUNK][G]>(sp);
            float acc1[G];   // dim 2u + 1 (acc: dim 2u); group 0 only
#pragma unroll
            for (int h = 0; h < G; ++h) acc1[h] = 0.0f;
            for (int c = 0; c < n_act; c += 2) {
                const bool two = c + 1 < n_act;
                __syncthreads();   // the previous pair's readers of srow / spt / xch are done
                if (t < 2 * CHUNK && (t < CHUNK || two)) srow[t] = prow[(size_t) c * CHUNK + t];
#pragma unroll
                for (int j = 0; j < 6; ++j) {
                    const int idx = t + THREADS * j, k = idx / (G * CHUNK), r = idx % (G * CHUNK);
                    if (k == 0 || two) spt2[k][r % CHUNK][r / CHUNK] = pb[(size_t) (c + k) * G * CHUNK + r];
                }
                __syncthreads();
                const int cc = c + grp;
                float a0[G], a1[G];
                if (cc < n_act) {
                    const int n_here = min(CHUNK, width - cc * CHUNK);
                    value_chunk_dim2(p, spt2[grp], srow + grp * CHUNK, n_here, u, a0, a1);
                    if (grp == 1) {
#pragma unroll
                        for (int h = 0; h < G; ++h) { xch[u][h] = a0[h]; xch[u][G + h] = a1[h]; }
                    }
                }
                __syncthreads();
                if (grp == 0) {
#pragma unroll 1
                    for (int k = 0; k < 2; ++k) {
                        if (k == 1) {
                            if (!two) break;
#pragma unroll
                            for (int h = 0; h < G; ++h) { a0[h] = xch[u][h]; a1[h] = xch[u][G + h]; }
                        }
#pragma unroll
                        for (int h = 0; h < G; ++h) {
                            const float m = sm_m[c + k][h];
                            const float w = __expf(m - M[h]);
                            const float L_next = fmaf(sm_l[c + k][h], w, L[h]);
                            const float acc_next = fmaf(a0[h], w, acc[h]);
                            const float acc1_next = fmaf(a1[h], w, acc1[h]);
                            const bool live = !(m == -FLT_MAX);
                            L[h] = live ? L_next : L[h];
                            acc[h] = live ? acc_next : acc[h];
                            acc1[h] = live ? acc1_next : acc1[h];
                        }
                    }
                }
            }
            if (grp == 0) {
                float* out = attn + ((size_t) z * n_kv_heads + kvh) * G * HD;
#pragma unroll
                for (int h = 0; h < G; ++h)
                    *reinterpret_cast<float2*>(out + (size_t) h * HD + 2 * u) =
                        make_float2(L[h] > 0.0f ? acc[h] / L[h] : 0.0f, L[h] > 0.0f ? acc1[h] / L[h] : 0.0f);
            }
        }
    }
}

// ---------------------------------------------------------------------------------------------------------------------
// variants 10-13: split.  ps_score_kernel - one CTA per (chunk, KV head, query) as attn_chunk_kernel_pre75, q in
// registers, K staged - writes the chunk's probabilities (head-major), rows, m and l to a block of the scratch;
// ps_value_kernel - one CTA per (KV head, query) - folds V chunk by chunk in ascending order and merges each chunk into
// the output as attn_merge_kernel does (M first from all the chunks' m).  No partial accumulators, no merge kernel; the
// two kernels keep their own register budgets (the fused kernel spills).
constexpr int PS_BLK = G * CHUNK + 2 * CHUNK + 2 * G;   // floats per (query, KV head, chunk): e, rows (int64), m, l

template <bool KMAGIC, bool RED24>
__global__ void __launch_bounds__(THREADS, 2) ps_score_kernel(const float* __restrict__ q, QsaAttnPools p,
                                                              const int32_t* __restrict__ ids,
                                                              const int32_t* __restrict__ step, int n_kv_heads,
                                                              int page_size, float scale, float* __restrict__ blocks,
                                                              int n_chunks, int cap) {
    q += (size_t) blockIdx.z * (size_t) (n_kv_heads * G) * HD;
    ids += (size_t) blockIdx.z * (size_t) cap;
    step += (size_t) blockIdx.z * kStepCount;
    const int chunk = blockIdx.x, kvh = blockIdx.y;
    float* const blk = blocks + (((size_t) blockIdx.z * n_kv_heads + kvh) * n_chunks + chunk) * PS_BLK;
    __shared__ __align__(16) int8_t sk[CHUNK][HD];
    __shared__ __align__(16) float sks[CHUNK][NSC];
    __shared__ float sp[G][CHUNK];
    __shared__ long long srow[CHUNK];
    const int n_ids = __ldg(step + kStepWidth);
    const int t = threadIdx.x, lane = t & 31, warp = t >> 5;
    const int c0 = chunk * CHUNK;
    const int n_here = min(CHUNK, n_ids - c0);
    if (n_here <= 0) {
        if (t < G) { blk[G * CHUNK + 2 * CHUNK + t] = -FLT_MAX; blk[G * CHUNK + 2 * CHUNK + G + t] = 0.0f; }
        return;
    }
    if (t < CHUNK) {
        const long long r = cell_row(p, ids, c0, t, n_here, n_kv_heads, kvh, page_size);
        srow[t] = r;
        reinterpret_cast<long long*>(blk + G * CHUNK)[t] = r;
    }
    float4 qa[G], qb[G];
    load_q(q + (size_t) (kvh * G) * HD, lane, qa, qb);
    __syncthreads();
    stage_rows(p.k_q, p.k_scale, srow, n_here, t, sk, sks);
    __syncthreads();
    score_chunk<KMAGIC, RED24>(qa, qb, sk, sks, srow, sp, n_here, scale, warp, lane);
    __syncthreads();
    for (int h = warp; h < G; h += WARPS) {
        const float a = sp[h][lane], b = sp[h][lane + 32];
        const float m = warp_max_x(fmaxf(a, b));
        const float ea = (lane < n_here && srow[lane] >= 0) ? __expf(a - m) : 0.0f;
        const float eb = (lane + 32 < n_here && srow[lane + 32] >= 0) ? __expf(b - m) : 0.0f;
        blk[h * CHUNK + lane] = ea;
        blk[h * CHUNK + lane + 32] = eb;
        const float l = warp_sum_x(ea + eb);
        if (lane == 0) { blk[G * CHUNK + 2 * CHUNK + h] = m; blk[G * CHUNK + 2 * CHUNK + G + h] = l; }
    }
}

template <int DIMS>
__global__ void __launch_bounds__(HD / DIMS) ps_value_kernel(QsaAttnPools p, const int32_t* __restrict__ steps,
                                                             const float* __restrict__ blocks, int n_kv_heads,
                                                             int n_chunks, float* __restrict__ attn) {
    constexpr int NT = HD / DIMS;
    const int kvh = blockIdx.x, z = blockIdx.y;
    const int t = threadIdx.x;
    const float* const blk0 = blocks + ((size_t) z * n_kv_heads + kvh) * n_chunks * PS_BLK;
    __shared__ float sm_m[NCMAX][G], sm_l[NCMAX][G];
    __shared__ __align__(16) float spt[CHUNK][G];
    __shared__ long long srow[CHUNK];
    const int width = __ldg(steps + (size_t) z * kStepCount + kStepWidth);
    const int n_act = width > 0 ? min(n_chunks, (width - 1) / CHUNK + 1) : 0;
    for (int i = t; i < n_act * G; i += NT) {
        const int c = i / G, h = i % G;
        sm_m[c][h] = __ldg(blk0 + (size_t) c * PS_BLK + G * CHUNK + 2 * CHUNK + h);
        sm_l[c][h] = __ldg(blk0 + (size_t) c * PS_BLK + G * CHUNK + 2 * CHUNK + G + h);
    }
    __syncthreads();
    float M[G];
#pragma unroll
    for (int h = 0; h < G; ++h) M[h] = -FLT_MAX;
    for (int c = 0; c < n_act; ++c)
#pragma unroll
        for (int h = 0; h < G; ++h) M[h] = fmaxf(M[h], sm_m[c][h]);
    float acc[DIMS][G], L[G];
#pragma unroll
    for (int h = 0; h < G; ++h) {
        L[h] = 0.0f;
#pragma unroll
        for (int d = 0; d < DIMS; ++d) acc[d][h] = 0.0f;
    }
    for (int c = 0; c < n_act; ++c) {
        const float* blk = blk0 + (size_t) c * PS_BLK;
        const int n_here = min(CHUNK, width - c * CHUNK);
        __syncthreads();   // the previous chunk's readers of srow / spt are done
        if (t < CHUNK) srow[t] = __ldg(reinterpret_cast<const long long*>(blk + G * CHUNK) + t);
#pragma unroll
        for (int j = 0; j < G * CHUNK / NT; ++j) {
            const int idx = t + NT * j, h = idx / CHUNK, cell = idx % CHUNK;
            spt[cell][h] = __ldg(blk + idx);
        }
        __syncthreads();
        float a[DIMS][G];
        if constexpr (DIMS == 1) value_chunk_global(p, spt, srow, n_here, t, a[0]);
        else value_chunk_dim2(p, spt, srow, n_here, t, a[0], a[1]);
#pragma unroll
        for (int h = 0; h < G; ++h) {
            const float m = sm_m[c][h];
            const float w = __expf(m - M[h]);
            const bool live = !(m == -FLT_MAX);
            const float L_next = fmaf(sm_l[c][h], w, L[h]);
            L[h] = live ? L_next : L[h];
#pragma unroll
            for (int d = 0; d < DIMS; ++d) {
                const float acc_next = fmaf(a[d][h], w, acc[d][h]);
                acc[d][h] = live ? acc_next : acc[d][h];
            }
        }
    }
    float* out = attn + ((size_t) z * n_kv_heads + kvh) * G * HD;
#pragma unroll
    for (int h = 0; h < G; ++h) {
        if constexpr (DIMS == 1) {
            out[(size_t) h * HD + t] = L[h] > 0.0f ? acc[0][h] / L[h] : 0.0f;
        } else {
            *reinterpret_cast<float2*>(out + (size_t) h * HD + 2 * t) =
                make_float2(L[h] > 0.0f ? acc[0][h] / L[h] : 0.0f, L[h] > 0.0f ? acc[1][h] / L[h] : 0.0f);
        }
    }
}

// ps_score_s_kernel: ps_score_kernel with attn_chunk_kernel_pre75<1, 2>'s own score phase (q in shared memory, two
// cells per step, K read from global memory per cell) - its 64 registers keep 4 CTAs per SM
__device__ __forceinline__ void load8_k(const QsaAttnPools& p, long long row, int d0, float* out) {
    const int8_t* codes = p.k_q + row * HD + d0;
    const uint16_t sbits = p.k_scale[row * NSC + d0 / KV_Q8_GROUP];
    const float sc = __half2float(__ushort_as_half(sbits));
    const uint2 raw = *reinterpret_cast<const uint2*>(codes);
    const int8_t* c = reinterpret_cast<const int8_t*>(&raw);
#pragma unroll
    for (int j = 0; j < 8; ++j) out[j] = (float) c[j] * sc;
}

__global__ void __launch_bounds__(THREADS) ps_score_s_kernel(const float* __restrict__ q, QsaAttnPools p,
                                                             const int32_t* __restrict__ ids,
                                                             const int32_t* __restrict__ step, int n_kv_heads,
                                                             int page_size, float scale, float* __restrict__ blocks,
                                                             int n_chunks, int cap) {
    q += (size_t) blockIdx.z * (size_t) (n_kv_heads * G) * HD;
    ids += (size_t) blockIdx.z * (size_t) cap;
    step += (size_t) blockIdx.z * kStepCount;
    const int chunk = blockIdx.x, kvh = blockIdx.y;
    float* const blk = blocks + (((size_t) blockIdx.z * n_kv_heads + kvh) * n_chunks + chunk) * PS_BLK;
    __shared__ __align__(16) float sq[G][HD];
    __shared__ float sp[G][CHUNK];
    __shared__ long long srow[CHUNK];
    const int n_ids = __ldg(step + kStepWidth);
    const int t = threadIdx.x, lane = t & 31, warp = t >> 5;
    const int c0 = chunk * CHUNK;
    const int n_here = min(CHUNK, n_ids - c0);
    if (n_here <= 0) {
        if (t < G) { blk[G * CHUNK + 2 * CHUNK + t] = -FLT_MAX; blk[G * CHUNK + 2 * CHUNK + G + t] = 0.0f; }
        return;
    }
    for (int i = t; i < G * HD; i += THREADS) sq[i / HD][i % HD] = q[(size_t) (kvh * G) * HD + i];
    if (t < CHUNK) {
        const long long r = cell_row(p, ids, c0, t, n_here, n_kv_heads, kvh, page_size);
        srow[t] = r;
        reinterpret_cast<long long*>(blk + G * CHUNK)[t] = r;
    }
    __syncthreads();
    constexpr int NC = 2;
    const int hd = head_of_lane(lane);
    for (int i = 0; i < CHUNK / WARPS; i += NC) {
        int cc[NC];
        bool all_ok = true;
#pragma unroll
        for (int j = 0; j < NC; ++j) {
            cc[j] = warp + WARPS * (i + j);
            all_ok = all_ok && cc[j] < n_here && srow[cc[j]] >= 0;
        }
        if (all_ok) {
            float kk[NC][8];
#pragma unroll
            for (int j = 0; j < NC; ++j) load8_k(p, srow[cc[j]], lane * 8, kk[j]);
            float pp[NC][16];
#pragma unroll
            for (int h = 0; h < G; ++h) {
                const float4 qa = *reinterpret_cast<const float4*>(&sq[h][lane * 8]);
                const float4 qb = *reinterpret_cast<const float4*>(&sq[h][lane * 8 + 4]);
#pragma unroll
                for (int j = 0; j < NC; ++j)
                    pp[j][h] = kk[j][0] * qa.x + kk[j][1] * qa.y + kk[j][2] * qa.z + kk[j][3] * qa.w +
                               kk[j][4] * qb.x + kk[j][5] * qb.y + kk[j][6] * qb.z + kk[j][7] * qb.w;
            }
#pragma unroll
            for (int j = 0; j < NC; ++j) {
#pragma unroll
                for (int h = G; h < 16; ++h) pp[j][h] = 0.0f;
                const float sj = reduce12(pp[j], lane);
                if ((lane & 1) == 0 && hd < G) sp[hd][cc[j]] = sj * scale;
            }
        } else {
            for (int k = 0; k < NC; ++k) {
                const int c = cc[k];
                if (c >= n_here || srow[c] < 0) {
                    if (lane < G) sp[lane][c] = -FLT_MAX;
                    continue;
                }
                float k8[8];
                load8_k(p, srow[c], lane * 8, k8);
                float part[16];
#pragma unroll
                for (int h = 0; h < G; ++h) {
                    const float4 qa = *reinterpret_cast<const float4*>(&sq[h][lane * 8]);
                    const float4 qb = *reinterpret_cast<const float4*>(&sq[h][lane * 8 + 4]);
                    part[h] = k8[0] * qa.x + k8[1] * qa.y + k8[2] * qa.z + k8[3] * qa.w +
                              k8[4] * qb.x + k8[5] * qb.y + k8[6] * qb.z + k8[7] * qb.w;
                }
#pragma unroll
                for (int h = G; h < 16; ++h) part[h] = 0.0f;
                const float sc = reduce12(part, lane);
                if ((lane & 1) == 0 && hd < G) sp[hd][c] = sc * scale;
            }
        }
    }
    __syncthreads();
    for (int h = warp; h < G; h += WARPS) {
        const float a = sp[h][lane], b = sp[h][lane + 32];
        const float m = warp_max_x(fmaxf(a, b));
        const float ea = (lane < n_here && srow[lane] >= 0) ? __expf(a - m) : 0.0f;
        const float eb = (lane + 32 < n_here && srow[lane + 32] >= 0) ? __expf(b - m) : 0.0f;
        blk[h * CHUNK + lane] = ea;
        blk[h * CHUNK + lane + 32] = eb;
        const float l = warp_sum_x(ea + eb);
        if (lane == 0) { blk[G * CHUNK + 2 * CHUNK + h] = m; blk[G * CHUNK + 2 * CHUNK + G + h] = l; }
    }
}

// ps_value_kernel with the V rows staged: VC cells at a time (16-byte loads into shared memory), then the chunk's
// ascending fold from there; DIMS 1 (256 threads) or 2 (128 threads, two dims each).  M first; L by 12 threads at the
// end (the merge's fmaf chain over the chunks, from shared memory); each chunk's weight w = __expf(m - M) once in
// shared memory.
constexpr int NCM = 33;   // chunks the split value kernel keeps (cap <= 2112)
template <int DIMS, int VC>
__global__ void __launch_bounds__(HD / DIMS) ps_value_st_kernel(QsaAttnPools p, const int32_t* __restrict__ steps,
                                                                const float* __restrict__ blocks, int n_kv_heads,
                                                                int n_chunks, float* __restrict__ attn) {
    constexpr int NT = HD / DIMS;
    const int kvh = blockIdx.x, z = blockIdx.y;
    const int t = threadIdx.x;
    const float* const blk0 = blocks + ((size_t) z * n_kv_heads + kvh) * n_chunks * PS_BLK;
    __shared__ float sm_m[NCM][G], sm_w[NCM][G];   // m; then the weight __expf(m - M)
    __shared__ float sm_l[NCM][G];
    __shared__ float sM[G], sL[G];
    __shared__ __align__(16) float spt[CHUNK][G];
    __shared__ long long srow[CHUNK];
    __shared__ __align__(16) int8_t sv[VC][HD];
    __shared__ float svs[VC][NSC];
    const int width = __ldg(steps + (size_t) z * kStepCount + kStepWidth);
    const int n_act = width > 0 ? min(n_chunks, (width - 1) / CHUNK + 1) : 0;
    for (int i = t; i < n_act * G; i += NT) {
        const int c = i / G, h = i % G;
        sm_m[c][h] = __ldg(blk0 + (size_t) c * PS_BLK + G * CHUNK + 2 * CHUNK + h);
        sm_l[c][h] = __ldg(blk0 + (size_t) c * PS_BLK + G * CHUNK + 2 * CHUNK + G + h);
    }
    __syncthreads();
    if (t < G) {
        float M = -FLT_MAX;
        for (int c = 0; c < n_act; ++c) M = fmaxf(M, sm_m[c][t]);
        float L = 0.0f;
        for (int c = 0; c < n_act; ++c) {
            const float m = sm_m[c][t];
            const float w = __expf(m - M);
            sm_w[c][t] = w;
            const float L_next = fmaf(sm_l[c][t], w, L);
            L = (m == -FLT_MAX) ? L : L_next;
        }
        sM[t] = M;
        sL[t] = L;
    }
    float acc[DIMS][G];
#pragma unroll
    for (int h = 0; h < G; ++h)
#pragma unroll
        for (int d = 0; d < DIMS; ++d) acc[d][h] = 0.0f;
    for (int c = 0; c < n_act; ++c) {
        const float* blk = blk0 + (size_t) c * PS_BLK;
        const int n_here = min(CHUNK, width - c * CHUNK);
        __syncthreads();   // the previous chunk's readers of srow / spt / sv are done (and sm_w written)
        if (t < CHUNK) srow[t] = __ldg(reinterpret_cast<const long long*>(blk + G * CHUNK) + t);
#pragma unroll
        for (int j = 0; j < G * CHUNK / NT; ++j) {
            const int idx = t + NT * j, h = idx / CHUNK, cell = idx % CHUNK;
            spt[cell][h] = __ldg(blk + idx);
        }
        float a[DIMS][G];
#pragma unroll
        for (int h = 0; h < G; ++h)
#pragma unroll
            for (int d = 0; d < DIMS; ++d) a[d][h] = 0.0f;
        for (int cb = 0; cb < n_here; cb += VC) {
            __syncthreads();   // srow / spt loaded; the previous round's readers of sv are done
#pragma unroll
            for (int j = 0; j < VC * 16 / NT; ++j) {
                const int idx = t + NT * j, cl = idx >> 4, part = idx & 15;
                const long long row = srow[cb + cl];
                const long long rr = (cb + cl < n_here && row >= 0) ? row : 0;
                reinterpret_cast<uint4*>(&sv[0][0])[idx] = __ldg(reinterpret_cast<const uint4*>(p.v_q + rr * HD) + part);
            }
            for (int i = t; i < VC * NSC; i += NT) {
                const int cl = i / NSC, g = i % NSC;
                const long long row = srow[cb + cl];
                const long long rr = (cb + cl < n_here && row >= 0) ? row : 0;
                svs[cl][g] = __half2float(__ushort_as_half(__ldg(p.v_scale + rr * NSC + g)));
            }
            __syncthreads();
            const int ce = min(VC, n_here - cb);
            if constexpr (DIMS == 1) {
                const int g = t / KV_Q8_GROUP;
#pragma unroll 4
                for (int cl = 0; cl < ce; ++cl) {
                    if (srow[cb + cl] < 0) continue;
                    fold(a[0], spt[cb + cl], (float) sv[cl][t] * svs[cl][g]);
                }
            } else {
                const int g = (2 * t) / KV_Q8_GROUP;
#pragma unroll 4
                for (int cl = 0; cl < ce; ++cl) {
                    if (srow[cb + cl] < 0) continue;
                    const uint16_t two = *reinterpret_cast<const uint16_t*>(&sv[cl][2 * t]);
                    const float sc = svs[cl][g];
                    const float v0 = (float) (int8_t) (two & 0xFF) * sc, v1 = (float) (int8_t) (two >> 8) * sc;
                    const float4* w4 = reinterpret_cast<const float4*>(spt[cb + cl]);
                    const float4 w0 = w4[0], w1 = w4[1], w2 = w4[2];
                    const float ww[G] = {w0.x, w0.y, w0.z, w0.w, w1.x, w1.y, w1.z, w1.w, w2.x, w2.y, w2.z, w2.w};
#pragma unroll
                    for (int h = 0; h < G; ++h) { a[0][h] = fmaf(ww[h], v0, a[0][h]); a[1][h] = fmaf(ww[h], v1, a[1][h]); }
                }
            }
        }
#pragma unroll
        for (int h = 0; h < G; ++h) {
            const bool live = !(sm_m[c][h] == -FLT_MAX);
            const float w = sm_w[c][h];
#pragma unroll
            for (int d = 0; d < DIMS; ++d) {
                const float acc_next = fmaf(a[d][h], w, acc[d][h]);
                acc[d][h] = live ? acc_next : acc[d][h];
            }
        }
    }
    __syncthreads();
    float* out = attn + ((size_t) z * n_kv_heads + kvh) * G * HD;
#pragma unroll
    for (int h = 0; h < G; ++h) {
        const float L = sL[h];
        if constexpr (DIMS == 1) {
            out[(size_t) h * HD + t] = L > 0.0f ? acc[0][h] / L : 0.0f;
        } else {
            *reinterpret_cast<float2*>(out + (size_t) h * HD + 2 * t) =
                make_float2(L > 0.0f ? acc[0][h] / L : 0.0f, L > 0.0f ? acc[1][h] / L : 0.0f);
        }
    }
}

// ---------------------------------------------------------------------------------------------------------------------
// ps_score_x_kernel: the score phase two cells at a time with a select-free reduce-scatter.  reduce12's selects exist
// because the two lanes of an exchange send different slots; here every lane keeps the same slots and sends the same
// slots, and the lane-dependent part is moved into which query head each register slot holds (q is loaded once, so the
// permutation costs nothing per cell) and which of the two cells is "first":
//   slot i (0..11) of lane L holds head P_L(i) = 6 ((i / 6) ^ b3) + ((i % 6 + 3 b2) % 6)       (b_k: bit k of L)
//   first = cell B if b4 else cell A, second = the other one; part1[i] / part2[i] = lane partials of (first / second,
//   head P_L(i)) - the same expression as attn_chunk_kernel_pre75.
//   level 16: r[i]  = part1[i] + shfl(part2[i], 16)   (partner: b4 flipped, so its second is my first, same P)
//   level 8 : r6[i] = r[i] + shfl(r[i + 6], 8)         (P_{L^8}(i) = P_L(i + 6))
//   level 4 : r3[i] = r6[i] + shfl(r6[i + 3], 4)       (P_{L^4}(i) = P_L(i + 3), i < 3)
//   level 2 / 1: reduce24's (heads 6 b3 + 3 b2 + {0, 1, 2}; o0 head +2 b1, o1 head +1 where b1 = 0)
// At every level the value at lane L is its partial at L plus its partial at L ^ o, the same operands warp_sum adds,
// so every score is reduce12's bit for bit.  A pair with an absent / masked cell runs the same code on row 0's staged
// codes and that cell's outputs are not stored (it gets -FLT_MAX).  25 shuffles, 25 adds and 4 selects per pair
// instead of 2 x (15 + 15 + ~30).
__device__ __forceinline__ int px_head(int i, int lane) {
    const int b3 = (lane >> 3) & 1, b2 = (lane >> 2) & 1;
    return 6 * ((i / 6) ^ b3) + ((i % 6 + 3 * b2) % 6);
}
__device__ __forceinline__ void load_q_perm(const float* __restrict__ qh, int lane, float4 (&qa)[G], float4 (&qb)[G]) {
#pragma unroll
    for (int i = 0; i < G; ++i) {
        const float4* p4 = reinterpret_cast<const float4*>(qh + (size_t) px_head(i, lane) * HD + lane * 8);
        qa[i] = __ldg(p4);
        qb[i] = __ldg(p4 + 1);
    }
}
__device__ __forceinline__ void dequant8(const int8_t* krow, float sc, int lane, float (&k8)[8]) {
    const uint2 raw = *reinterpret_cast<const uint2*>(krow + lane * 8);
    const int8_t* cb = reinterpret_cast<const int8_t*>(&raw);
#pragma unroll
    for (int e = 0; e < 8; ++e) k8[e] = (float) cb[e] * sc;
}
__device__ __forceinline__ float dot8(const float (&k8)[8], const float4& qa, const float4& qb) {
    return k8[0] * qa.x + k8[1] * qa.y + k8[2] * qa.z + k8[3] * qa.w + k8[4] * qb.x + k8[5] * qb.y + k8[6] * qb.z +
           k8[7] * qb.w;
}
template <int UNR = 1>
__device__ __forceinline__ void score_chunk_x(const float4 (&qa)[G], const float4 (&qb)[G], const int8_t (*sk)[HD],
                                              const float (*sks)[NSC], const long long* srow, float (*sp)[CHUNK],
                                              int n_here, float scale, int warp, int lane) {
    const bool h16 = (lane & 16) != 0, h2 = (lane & 2) != 0, even = (lane & 1) == 0;
    const int hb = 6 * ((lane >> 3) & 1) + 3 * ((lane >> 2) & 1);
    const int h0 = hb + (h2 ? 2 : 0), h1 = hb + 1;
#pragma unroll UNR
    for (int i = 0; i < CHUNK / WARPS; i += 2) {
        const int ca = warp + WARPS * i, cb = warp + WARPS * (i + 1);
        const bool va = ca < n_here && srow[ca] >= 0, vb = cb < n_here && srow[cb] >= 0;
        if (!va && !vb) {
            if (lane < G) { sp[lane][ca] = -FLT_MAX; sp[lane][cb] = -FLT_MAX; }
            continue;
        }
        const int c1 = h16 ? cb : ca, c2 = h16 ? ca : cb;   // first / second (the staged rows of absent cells are row 0's)
        float r[G];
        {
            float k8[8];
            dequant8(sk[c2], sks[c2][lane >> 3], lane, k8);
            float p2[G];
#pragma unroll
            for (int s = 0; s < G; ++s) p2[s] = dot8(k8, qa[s], qb[s]);
            dequant8(sk[c1], sks[c1][lane >> 3], lane, k8);
#pragma unroll
            for (int s = 0; s < G; ++s) r[s] = dot8(k8, qa[s], qb[s]) + __shfl_xor_sync(0xffffffffu, p2[s], 16);
        }
        float r6[6];
#pragma unroll
        for (int s = 0; s < 6; ++s) r6[s] = r[s] + __shfl_xor_sync(0xffffffffu, r[s + 6], 8);
        float r3[3];
#pragma unroll
        for (int s = 0; s < 3; ++s) r3[s] = r6[s] + __shfl_xor_sync(0xffffffffu, r6[s + 3], 4);
        const float x0 = (h2 ? r3[2] : r3[0]) + __shfl_xor_sync(0xffffffffu, h2 ? r3[0] : r3[2], 2);
        const float x1 = r3[1] + __shfl_xor_sync(0xffffffffu, r3[1], 2);
        const float o0 = x0 + __shfl_xor_sync(0xffffffffu, x0, 1);
        const float o1 = x1 + __shfl_xor_sync(0xffffffffu, x1, 1);
        const bool vmine = h16 ? vb : va;
        if (even && vmine) {
            sp[h0][c1] = o0 * scale;
            if (!h2) sp[h1][c1] = o1 * scale;
        }
        if (lane < G) {
            if (!va) sp[lane][ca] = -FLT_MAX;
            if (!vb) sp[lane][cb] = -FLT_MAX;
        }
    }
}

// ps_score_kernel with score_chunk_x (q loaded in the permuted slot order)
template <int UNR>
__global__ void __launch_bounds__(THREADS, 2) ps_score_x_kernel(const float* __restrict__ q, QsaAttnPools p,
                                                                const int32_t* __restrict__ ids,
                                                                const int32_t* __restrict__ step, int n_kv_heads,
                                                                int page_size, float scale, float* __restrict__ blocks,
                                                                int n_chunks, int cap) {
    q += (size_t) blockIdx.z * (size_t) (n_kv_heads * G) * HD;
    ids += (size_t) blockIdx.z * (size_t) cap;
    step += (size_t) blockIdx.z * kStepCount;
    const int chunk = blockIdx.x, kvh = blockIdx.y;
    float* const blk = blocks + (((size_t) blockIdx.z * n_kv_heads + kvh) * n_chunks + chunk) * PS_BLK;
    __shared__ __align__(16) int8_t sk[CHUNK][HD];
    __shared__ __align__(16) float sks[CHUNK][NSC];
    __shared__ float sp[G][CHUNK];
    __shared__ long long srow[CHUNK];
    const int n_ids = __ldg(step + kStepWidth);
    const int t = threadIdx.x, lane = t & 31, warp = t >> 5;
    const int c0 = chunk * CHUNK;
    const int n_here = min(CHUNK, n_ids - c0);
    if (n_here <= 0) {
        if (t < G) { blk[G * CHUNK + 2 * CHUNK + t] = -FLT_MAX; blk[G * CHUNK + 2 * CHUNK + G + t] = 0.0f; }
        return;
    }
    if (t < CHUNK) {
        const long long r = cell_row(p, ids, c0, t, n_here, n_kv_heads, kvh, page_size);
        srow[t] = r;
        reinterpret_cast<long long*>(blk + G * CHUNK)[t] = r;
    }
    float4 qa[G], qb[G];
    load_q_perm(q + (size_t) (kvh * G) * HD, lane, qa, qb);
    __syncthreads();
    stage_rows(p.k_q, p.k_scale, srow, n_here, t, sk, sks);
    __syncthreads();
    score_chunk_x<UNR>(qa, qb, sk, sks, srow, sp, n_here, scale, warp, lane);
    __syncthreads();
    for (int h = warp; h < G; h += WARPS) {
        const float a = sp[h][lane], b = sp[h][lane + 32];
        const float m = warp_max_x(fmaxf(a, b));
        const float ea = (lane < n_here && srow[lane] >= 0) ? __expf(a - m) : 0.0f;
        const float eb = (lane + 32 < n_here && srow[lane + 32] >= 0) ? __expf(b - m) : 0.0f;
        blk[h * CHUNK + lane] = ea;
        blk[h * CHUNK + lane + 32] = eb;
        const float l = warp_sum_x(ea + eb);
        if (lane == 0) { blk[G * CHUNK + 2 * CHUNK + h] = m; blk[G * CHUNK + 2 * CHUNK + G + h] = l; }
    }
}

// ps_value_d4_kernel: the value kernel with four dims a thread (64 threads, one CTA per (KV head, query)): a cell's four
// codes are one 32-bit load (a warp reads 128 contiguous bytes), its 12 weights three broadcast float4 reads for 48
// fmaf.  V loads run four cells ahead.  Per (dim, head) the fold and the merge are ps_value_kernel's.
__global__ void __launch_bounds__(64) ps_value_d4_kernel(QsaAttnPools p, const int32_t* __restrict__ steps,
                                                         const float* __restrict__ blocks, int n_kv_heads, int n_chunks,
                                                         float* __restrict__ attn) {
    constexpr int NT = 64;
    const int kvh = blockIdx.x, z = blockIdx.y;
    const int t = threadIdx.x;
    const float* const blk0 = blocks + ((size_t) z * n_kv_heads + kvh) * n_chunks * PS_BLK;
    __shared__ float sm_m[NCM][G], sm_w[NCM][G], sm_l[NCM][G];
    __shared__ float sL[G];
    __shared__ __align__(16) float spt[CHUNK][G];
    __shared__ long long srow[CHUNK];
    __shared__ int s_masked;
    const int width = __ldg(steps + (size_t) z * kStepCount + kStepWidth);
    const int n_act = width > 0 ? min(n_chunks, (width - 1) / CHUNK + 1) : 0;
    for (int i = t; i < n_act * G; i += NT) {
        const int c = i / G, h = i % G;
        sm_m[c][h] = __ldg(blk0 + (size_t) c * PS_BLK + G * CHUNK + 2 * CHUNK + h);
        sm_l[c][h] = __ldg(blk0 + (size_t) c * PS_BLK + G * CHUNK + 2 * CHUNK + G + h);
    }
    __syncthreads();
    if (t < G) {
        float M = -FLT_MAX;
        for (int c = 0; c < n_act; ++c) M = fmaxf(M, sm_m[c][t]);
        float L = 0.0f;
        for (int c = 0; c < n_act; ++c) {
            const float m = sm_m[c][t];
            const float w = __expf(m - M);
            sm_w[c][t] = w;
            const float L_next = fmaf(sm_l[c][t], w, L);
            L = (m == -FLT_MAX) ? L : L_next;
        }
        sL[t] = L;
    }
    const int d0 = 4 * t, g = d0 / KV_Q8_GROUP;
    float acc[4][G];
#pragma unroll
    for (int h = 0; h < G; ++h)
#pragma unroll
        for (int d = 0; d < 4; ++d) acc[d][h] = 0.0f;
    for (int c = 0; c < n_act; ++c) {
        const float* blk = blk0 + (size_t) c * PS_BLK;
        const int n_here = min(CHUNK, width - c * CHUNK);
        __syncthreads();
        if (t == 0) s_masked = 0;
        __syncthreads();
        {
            const long long r = __ldg(reinterpret_cast<const long long*>(blk + G * CHUNK) + t);
            srow[t] = r;
            if (t < n_here && r < 0) s_masked = 1;
        }
#pragma unroll
        for (int j = 0; j < G * CHUNK / NT; ++j) {
            const int idx = t + NT * j, h = idx / CHUNK, cell = idx % CHUNK;
            spt[cell][h] = __ldg(blk + idx);
        }
        __syncthreads();
        float a[4][G];
#pragma unroll
        for (int h = 0; h < G; ++h)
#pragma unroll
            for (int d = 0; d < 4; ++d) a[d][h] = 0.0f;
        const auto fold4 = [&](int cl, uint32_t codes, float sc) {
            const float v0 = (float) (int8_t) (codes & 0xFF) * sc, v1 = (float) (int8_t) ((codes >> 8) & 0xFF) * sc;
            const float v2 = (float) (int8_t) ((codes >> 16) & 0xFF) * sc, v3 = (float) (int8_t) (codes >> 24) * sc;
            const float4* w4 = reinterpret_cast<const float4*>(spt[cl]);
            const float4 w0 = w4[0], w1 = w4[1], w2 = w4[2];
            const float ww[G] = {w0.x, w0.y, w0.z, w0.w, w1.x, w1.y, w1.z, w1.w, w2.x, w2.y, w2.z, w2.w};
#pragma unroll
            for (int h = 0; h < G; ++h) {
                a[0][h] = fmaf(ww[h], v0, a[0][h]);
                a[1][h] = fmaf(ww[h], v1, a[1][h]);
                a[2][h] = fmaf(ww[h], v2, a[2][h]);
                a[3][h] = fmaf(ww[h], v3, a[3][h]);
            }
        };
        const auto vld = [&](int cl, uint32_t& codes, float& sc) {
            const long long r = srow[cl];
            const long long rr = r >= 0 ? r : 0;
            codes = __ldg(reinterpret_cast<const uint32_t*>(p.v_q + rr * HD + d0));
            sc = __half2float(__ushort_as_half(__ldg(p.v_scale + rr * NSC + g)));
        };
        if (!s_masked) {
            int cl = 0;
            for (; cl + 4 <= n_here; cl += 4) {
                uint32_t k0, k1, k2, k3;
                float s0, s1, s2, s3;
                vld(cl, k0, s0); vld(cl + 1, k1, s1); vld(cl + 2, k2, s2); vld(cl + 3, k3, s3);
                fold4(cl, k0, s0);
                fold4(cl + 1, k1, s1);
                fold4(cl + 2, k2, s2);
                fold4(cl + 3, k3, s3);
            }
            for (; cl < n_here; ++cl) {
                uint32_t k0;
                float s0;
                vld(cl, k0, s0);
                fold4(cl, k0, s0);
            }
        } else {
            for (int cl = 0; cl < n_here; ++cl) {
                if (srow[cl] < 0) continue;
                uint32_t k0;
                float s0;
                vld(cl, k0, s0);
                fold4(cl, k0, s0);
            }
        }
#pragma unroll
        for (int h = 0; h < G; ++h) {
            const bool live = !(sm_m[c][h] == -FLT_MAX);
            const float w = sm_w[c][h];
#pragma unroll
            for (int d = 0; d < 4; ++d) {
                const float acc_next = fmaf(a[d][h], w, acc[d][h]);
                acc[d][h] = live ? acc_next : acc[d][h];
            }
        }
    }
    __syncthreads();
    float* out = attn + ((size_t) z * n_kv_heads + kvh) * G * HD;
#pragma unroll
    for (int h = 0; h < G; ++h) {
        const float L = sL[h];
        *reinterpret_cast<float4*>(out + (size_t) h * HD + d0) =
            make_float4(L > 0.0f ? acc[0][h] / L : 0.0f, L > 0.0f ? acc[1][h] / L : 0.0f,
                        L > 0.0f ? acc[2][h] / L : 0.0f, L > 0.0f ? acc[3][h] / L : 0.0f);
    }
}

// ps_value_d4h_kernel<HG>: ps_value_d4_kernel with the 12 heads split over HG thread groups of 64 (thread = 4 dims x
// 12 / HG heads): HG times the warps per (KV head, query) for the same arithmetic per (dim, head).
template <int HG>
__global__ void __launch_bounds__(64 * HG) ps_value_d4h_kernel(QsaAttnPools p, const int32_t* __restrict__ steps,
                                                               const float* __restrict__ blocks, int n_kv_heads,
                                                               int n_chunks, float* __restrict__ attn) {
    constexpr int NT = 64 * HG, GH = G / HG;
    static_assert(GH % 2 == 0, "heads per group: pairs of float2");
    const int kvh = blockIdx.x, z = blockIdx.y;
    const int t = threadIdx.x, u = t % 64, hg = t / 64, h0 = hg * GH;
    const float* const blk0 = blocks + ((size_t) z * n_kv_heads + kvh) * n_chunks * PS_BLK;
    __shared__ float sm_m[NCM][G], sm_w[NCM][G], sm_l[NCM][G];
    __shared__ float sL[G];
    __shared__ __align__(16) float spt[CHUNK][G];
    __shared__ long long srow[CHUNK];
    __shared__ int s_masked;
    const int width = __ldg(steps + (size_t) z * kStepCount + kStepWidth);
    const int n_act = width > 0 ? min(n_chunks, (width - 1) / CHUNK + 1) : 0;
    for (int i = t; i < n_act * G; i += NT) {
        const int c = i / G, h = i % G;
        sm_m[c][h] = __ldg(blk0 + (size_t) c * PS_BLK + G * CHUNK + 2 * CHUNK + h);
        sm_l[c][h] = __ldg(blk0 + (size_t) c * PS_BLK + G * CHUNK + 2 * CHUNK + G + h);
    }
    __syncthreads();
    if (t < G) {
        float M = -FLT_MAX;
        for (int c = 0; c < n_act; ++c) M = fmaxf(M, sm_m[c][t]);
        float L = 0.0f;
        for (int c = 0; c < n_act; ++c) {
            const float m = sm_m[c][t];
            const float w = __expf(m - M);
            sm_w[c][t] = w;
            const float L_next = fmaf(sm_l[c][t], w, L);
            L = (m == -FLT_MAX) ? L : L_next;
        }
        sL[t] = L;
    }
    const int d0 = 4 * u, g = d0 / KV_Q8_GROUP;
    float acc[4][GH];
#pragma unroll
    for (int h = 0; h < GH; ++h)
#pragma unroll
        for (int d = 0; d < 4; ++d) acc[d][h] = 0.0f;
    for (int c = 0; c < n_act; ++c) {
        const float* blk = blk0 + (size_t) c * PS_BLK;
        const int n_here = min(CHUNK, width - c * CHUNK);
        __syncthreads();
        if (t == 0) s_masked = 0;
        __syncthreads();
        if (t < CHUNK) {
            const long long r = __ldg(reinterpret_cast<const long long*>(blk + G * CHUNK) + t);
            srow[t] = r;
            if (t < n_here && r < 0) s_masked = 1;
        }
#pragma unroll
        for (int j = 0; j < G * CHUNK / NT; ++j) {
            const int idx = t + NT * j, h = idx / CHUNK, cell = idx % CHUNK;
            spt[cell][h] = __ldg(blk + idx);
        }
        __syncthreads();
        float a[4][GH];
#pragma unroll
        for (int h = 0; h < GH; ++h)
#pragma unroll
            for (int d = 0; d < 4; ++d) a[d][h] = 0.0f;
        const auto fold4 = [&](int cl, uint32_t codes, float sc) {
            const float v0 = (float) (int8_t) (codes & 0xFF) * sc, v1 = (float) (int8_t) ((codes >> 8) & 0xFF) * sc;
            const float v2 = (float) (int8_t) ((codes >> 16) & 0xFF) * sc, v3 = (float) (int8_t) (codes >> 24) * sc;
            float ww[GH];
            const float2* w2 = reinterpret_cast<const float2*>(&spt[cl][h0]);
#pragma unroll
            for (int k = 0; k < GH / 2; ++k) { const float2 x = w2[k]; ww[2 * k] = x.x; ww[2 * k + 1] = x.y; }
#pragma unroll
            for (int h = 0; h < GH; ++h) {
                a[0][h] = fmaf(ww[h], v0, a[0][h]);
                a[1][h] = fmaf(ww[h], v1, a[1][h]);
                a[2][h] = fmaf(ww[h], v2, a[2][h]);
                a[3][h] = fmaf(ww[h], v3, a[3][h]);
            }
        };
        const auto vld = [&](int cl, uint32_t& codes, float& sc) {
            const long long r = srow[cl];
            const long long rr = r >= 0 ? r : 0;
            codes = __ldg(reinterpret_cast<const uint32_t*>(p.v_q + rr * HD + d0));
            sc = __half2float(__ushort_as_half(__ldg(p.v_scale + rr * NSC + g)));
        };
        if (!s_masked) {
            int cl = 0;
            for (; cl + 4 <= n_here; cl += 4) {
                uint32_t k0, k1, k2, k3;
                float s0, s1, s2, s3;
                vld(cl, k0, s0); vld(cl + 1, k1, s1); vld(cl + 2, k2, s2); vld(cl + 3, k3, s3);
                fold4(cl, k0, s0);
                fold4(cl + 1, k1, s1);
                fold4(cl + 2, k2, s2);
                fold4(cl + 3, k3, s3);
            }
            for (; cl < n_here; ++cl) {
                uint32_t k0;
                float s0;
                vld(cl, k0, s0);
                fold4(cl, k0, s0);
            }
        } else {
            for (int cl = 0; cl < n_here; ++cl) {
                if (srow[cl] < 0) continue;
                uint32_t k0;
                float s0;
                vld(cl, k0, s0);
                fold4(cl, k0, s0);
            }
        }
#pragma unroll
        for (int h = 0; h < GH; ++h) {
            const bool live = !(sm_m[c][h0 + h] == -FLT_MAX);
            const float w = sm_w[c][h0 + h];
#pragma unroll
            for (int d = 0; d < 4; ++d) {
                const float acc_next = fmaf(a[d][h], w, acc[d][h]);
                acc[d][h] = live ? acc_next : acc[d][h];
            }
        }
    }
    __syncthreads();
    float* out = attn + ((size_t) z * n_kv_heads + kvh) * G * HD;
#pragma unroll
    for (int h = 0; h < GH; ++h) {
        const float L = sL[h0 + h];
        *reinterpret_cast<float4*>(out + (size_t) (h0 + h) * HD + d0) =
            make_float4(L > 0.0f ? acc[0][h] / L : 0.0f, L > 0.0f ? acc[1][h] / L : 0.0f,
                        L > 0.0f ? acc[2][h] / L : 0.0f, L > 0.0f ? acc[3][h] / L : 0.0f);
    }
}

// ps_score_x_kernel over CPC consecutive chunks of one (KV head, query) per CTA: q loaded once (96 KB of L1/L2 reads a
// CTA), the chunks one after the other with ps_score_x_kernel's body
template <int CPC>
__global__ void __launch_bounds__(THREADS, 2) ps_score_xc_kernel(const float* __restrict__ q, QsaAttnPools p,
                                                                 const int32_t* __restrict__ ids,
                                                                 const int32_t* __restrict__ step, int n_kv_heads,
                                                                 int page_size, float scale, float* __restrict__ blocks,
                                                                 int n_chunks, int cap) {
    q += (size_t) blockIdx.z * (size_t) (n_kv_heads * G) * HD;
    ids += (size_t) blockIdx.z * (size_t) cap;
    step += (size_t) blockIdx.z * kStepCount;
    const int kvh = blockIdx.y;
    __shared__ __align__(16) int8_t sk[CHUNK][HD];
    __shared__ __align__(16) float sks[CHUNK][NSC];
    __shared__ float sp[G][CHUNK];
    __shared__ long long srow[CHUNK];
    const int n_ids = __ldg(step + kStepWidth);
    const int t = threadIdx.x, lane = t & 31, warp = t >> 5;
    float4 qa[G], qb[G];
    bool q_loaded = false;
    for (int k = 0; k < CPC; ++k) {
        const int chunk = blockIdx.x * CPC + k;
        if (chunk >= n_chunks) break;
        float* const blk = blocks + (((size_t) blockIdx.z * n_kv_heads + kvh) * n_chunks + chunk) * PS_BLK;
        const int c0 = chunk * CHUNK;
        const int n_here = min(CHUNK, n_ids - c0);
        if (n_here <= 0) {   // CTA-uniform; every later chunk is absent too
            if (t < G) { blk[G * CHUNK + 2 * CHUNK + t] = -FLT_MAX; blk[G * CHUNK + 2 * CHUNK + G + t] = 0.0f; }
            continue;
        }
        if (k > 0) __syncthreads();   // the previous chunk's softmax read srow / sp
        if (t < CHUNK) {
            const long long r = cell_row(p, ids, c0, t, n_here, n_kv_heads, kvh, page_size);
            srow[t] = r;
            reinterpret_cast<long long*>(blk + G * CHUNK)[t] = r;
        }
        if (!q_loaded) { load_q_perm(q + (size_t) (kvh * G) * HD, lane, qa, qb); q_loaded = true; }
        __syncthreads();
        stage_rows(p.k_q, p.k_scale, srow, n_here, t, sk, sks);
        __syncthreads();
        score_chunk_x(qa, qb, sk, sks, srow, sp, n_here, scale, warp, lane);
        __syncthreads();
        for (int h = warp; h < G; h += WARPS) {
            const float a = sp[h][lane], b = sp[h][lane + 32];
            const float m = warp_max_x(fmaxf(a, b));
            const float ea = (lane < n_here && srow[lane] >= 0) ? __expf(a - m) : 0.0f;
            const float eb = (lane + 32 < n_here && srow[lane + 32] >= 0) ? __expf(b - m) : 0.0f;
            blk[h * CHUNK + lane] = ea;
            blk[h * CHUNK + lane + 32] = eb;
            const float l = warp_sum_x(ea + eb);
            if (lane == 0) { blk[G * CHUNK + 2 * CHUNK + h] = m; blk[G * CHUNK + 2 * CHUNK + G + h] = l; }
        }
    }
}

constexpr int FUSED_MINB = 2;

int sm_count() {
    static int sms[64] = {};
    int dev = 0;
    if (cudaGetDevice(&dev) != cudaSuccess || dev < 0 || dev >= 64) { cudaGetLastError(); return 0; }
    if (sms[dev] == 0 && cudaDeviceGetAttribute(&sms[dev], cudaDevAttrMultiProcessorCount, dev) != cudaSuccess) {
        cudaGetLastError();
        sms[dev] = 0;
    }
    return sms[dev];
}

bool sm6x_device() {
    static int cc[64] = {};
    int dev = 0;
    if (cudaGetDevice(&dev) != cudaSuccess || dev < 0 || dev >= 64) { cudaGetLastError(); return false; }
    if (cc[dev] == 0) {
        int major = 0;
        if (cudaDeviceGetAttribute(&major, cudaDevAttrComputeCapabilityMajor, dev) != cudaSuccess) {
            cudaGetLastError();
            return false;
        }
        cc[dev] = major;
    }
    return cc[dev] == 6;
}

}  // namespace

bool qsa_pattn_batch(const float* q, const QsaAttnPools& pools, const int32_t* ids, const int32_t* steps, int64_t cap,
                     const QsaShapes& s, float* scratch, uint64_t scratch_floats, float* attn, int64_t n_q,
                     void* stream, int variant) {
    if (n_q <= 0) return true;
    if (s.head_dim != HD || s.n_head != (int64_t) G * s.n_head_kv || cap <= 0 || !scratch || !ids || !steps ||
        !pools.page_table || pools.k_q == nullptr || pools.v_q == nullptr || pools.k_q4 != nullptr ||
        pools.v_q4 != nullptr || !pools.k_scale || !pools.v_scale || !sm6x_device())
        return false;
    const cudaStream_t st = (cudaStream_t) stream;
    const int n_chunks = (int) ((cap + CHUNK - 1) / CHUNK);
    const float scale = 1.0f / sqrtf((float) HD);
    if (variant == 0) variant = 1463;   // the default: ps_score_x_kernel + ps_value_d4_kernel
    if (variant == 1) {
        const long long stride = (long long) qsa_decode_attn_scratch_floats(cap, s);
        const int64_t batch = (int64_t) (scratch_floats / (uint64_t) stride);
        if (batch < 1) return false;
        const int64_t nb_max = batch < 32 ? batch : 32;   // prefill.cpp's attn_batch
        for (int64_t t0 = 0; t0 < n_q; t0 += nb_max) {
            const int64_t nb = n_q - t0 < nb_max ? n_q - t0 : nb_max;
            float* part_acc = scratch;
            float* part_m = scratch + (size_t) n_chunks * s.n_head * HD;
            float* part_l = part_m + (size_t) n_chunks * s.n_head;
            const dim3 grid((unsigned) n_chunks, (unsigned) s.n_head_kv, (unsigned) nb);
            pa_chunk_kernel<2><<<grid, THREADS, 0, st>>>(q + (size_t) t0 * s.n_head * HD, pools, ids + (size_t) t0 * cap,
                                                         steps + (size_t) t0 * kStepCount, (int) s.n_head_kv,
                                                         (int) s.page_size, scale, part_acc, part_m, part_l, n_chunks,
                                                         (int) cap, stride);
            pa_merge_v2_kernel<kMergeU><<<dim3((unsigned) s.n_head, (unsigned) nb), HD, 0, st>>>(
                part_acc, part_m, part_l, n_chunks, steps + (size_t) t0 * kStepCount, attn + (size_t) t0 * s.n_head * HD,
                stride);
        }
    } else if (variant >= 2 && variant <= 9) {
        if (n_chunks > NCMAX) return false;
        // a slot: probabilities [n_chunks][G][CHUNK] floats + rows [n_chunks][CHUNK] int64, rounded to 16 bytes
        const long long slot_floats = ((long long) n_chunks * G * CHUNK + 2LL * n_chunks * CHUNK + 3) / 4 * 4;
        const int64_t n_slots = (int64_t) (scratch_floats / (uint64_t) slot_floats);
        const int64_t items = n_q * s.n_head_kv;
        int per_sm = 0;
        const void* fn = variant == 2   ? (const void*) pf_fused_kernel<FUSED_MINB, 0, false, false>
                         : variant == 3 ? (const void*) pf_fused_kernel<FUSED_MINB, 1, false, false>
                         : variant == 4 ? (const void*) pf_fused_kernel<FUSED_MINB, 2, false, false>
                         : variant == 5 ? (const void*) pf_fused_kernel<FUSED_MINB, 0, true, false>
                         : variant == 6 ? (const void*) pf_fused_kernel<FUSED_MINB, 2, true, false>
                         : variant == 7 ? (const void*) pf_fused_kernel<FUSED_MINB, 0, false, true>
                         : variant == 8 ? (const void*) pf_fused_kernel<FUSED_MINB, 2, true, true>
                                        : (const void*) pf_fused_kernel<FUSED_MINB, 2, false, true>;
        if (cudaOccupancyMaxActiveBlocksPerMultiprocessor(&per_sm, fn, THREADS, 0) != cudaSuccess) {
            cudaGetLastError();
            return false;
        }
        int64_t grid = (int64_t) per_sm * sm_count();
        if (grid > n_slots) grid = n_slots;
        if (grid > items) grid = items;
        if (grid < 1) return false;
#define PF_LAUNCH(V, K, R) pf_fused_kernel<FUSED_MINB, V, K, R><<<(unsigned) grid, THREADS, 0, st>>>(q, pools, ids, steps, \
            (int) s.n_head_kv, (int) s.page_size, scale, scratch, slot_floats, n_chunks, (int) cap, (int) n_q, attn)
        switch (variant) {
            case 2: PF_LAUNCH(0, false, false); break;
            case 3: PF_LAUNCH(1, false, false); break;
            case 4: PF_LAUNCH(2, false, false); break;
            case 5: PF_LAUNCH(0, true, false); break;
            case 6: PF_LAUNCH(2, true, false); break;
            case 7: PF_LAUNCH(0, false, true); break;
            case 8: PF_LAUNCH(2, true, true); break;
            default: PF_LAUNCH(2, false, true); break;
        }
#undef PF_LAUNCH
    } else if ((variant >= 10 && variant <= 13) || (variant >= 1000 && variant < 2000)) {
        // split: 1000 + 100 S + 10 V + mask.  S: 0 ps_score (q in registers), 1 + reduce24, 2 + magic + reduce24,
        // 3 ps_score_s (q in shared memory: the production score phase).  V: 0 ps_value<1> (V from global memory),
        // 1 ps_value<2>, 2 st<1, 32>, 3 st<1, 64>, 4 st<2, 32>, 5 st<2, 64>.  mask: 3 both kernels, 1 the score
        // kernel only, 2 the value kernel only (timing).
        int code = variant;
        if (variant == 10) code = 1003;
        else if (variant == 11) code = 1013;
        else if (variant == 12) code = 1113;
        else if (variant == 13) code = 1213;
        const int S = (code / 100) % 10, V = (code / 10) % 10, mask = code % 10;
        if (n_chunks > NCMAX || n_chunks > NCM || S > 8 || V > 8 || mask < 1 || mask > 3) return false;
        const uint64_t per_q = (uint64_t) s.n_head_kv * (uint64_t) n_chunks * PS_BLK;
        const int64_t max_batch = (int64_t) (scratch_floats / per_q);
        if (max_batch < 1) return false;
        const int64_t n_batches = (n_q + max_batch - 1) / max_batch;
        const int64_t batch = (n_q + n_batches - 1) / n_batches;   // even batches
        for (int64_t t0 = 0; t0 < n_q; t0 += batch) {
            const int64_t nb = n_q - t0 < batch ? n_q - t0 : batch;
            const dim3 g1((unsigned) n_chunks, (unsigned) s.n_head_kv, (unsigned) nb);
            const float* qb = q + (size_t) t0 * s.n_head * HD;
            const int32_t* ib = ids + (size_t) t0 * cap;
            const int32_t* sb = steps + (size_t) t0 * kStepCount;
            if (mask & 1) {
#define PS_SCORE(K, R) ps_score_kernel<K, R><<<g1, THREADS, 0, st>>>(qb, pools, ib, sb, (int) s.n_head_kv, \
            (int) s.page_size, scale, scratch, n_chunks, (int) cap)
                switch (S) {
                    case 0: PS_SCORE(false, false); break;
                    case 1: PS_SCORE(false, true); break;
                    case 2: PS_SCORE(true, true); break;
                    case 3:
                        ps_score_s_kernel<<<g1, THREADS, 0, st>>>(qb, pools, ib, sb, (int) s.n_head_kv,
                                                                  (int) s.page_size, scale, scratch, n_chunks, (int) cap);
                        break;
                    case 4:
                        ps_score_x_kernel<1><<<g1, THREADS, 0, st>>>(qb, pools, ib, sb, (int) s.n_head_kv,
                                                                  (int) s.page_size, scale, scratch, n_chunks, (int) cap);
                        break;
                    case 8:
                        ps_score_x_kernel<2><<<g1, THREADS, 0, st>>>(qb, pools, ib, sb, (int) s.n_head_kv,
                                                                     (int) s.page_size, scale, scratch, n_chunks, (int) cap);
                        break;
                    default: {
                        const int cpc = S - 3;   // 5: 2 chunks a CTA, 6: 3, 7: 4
                        const dim3 gc((unsigned) ((n_chunks + cpc - 1) / cpc), (unsigned) s.n_head_kv, (unsigned) nb);
#define PS_XC(N) ps_score_xc_kernel<N><<<gc, THREADS, 0, st>>>(qb, pools, ib, sb, (int) s.n_head_kv, \
                                                                  (int) s.page_size, scale, scratch, n_chunks, (int) cap)
                        if (cpc == 2) PS_XC(2);
                        else if (cpc == 3) PS_XC(3);
                        else PS_XC(4);
#undef PS_XC
                        break;
                    }
                }
#undef PS_SCORE
            }
            if (mask & 2) {
                const dim3 g2((unsigned) s.n_head_kv, (unsigned) nb);
                float* ob = attn + (size_t) t0 * s.n_head * HD;
                switch (V) {
                    case 0: ps_value_kernel<1><<<g2, HD, 0, st>>>(pools, sb, scratch, (int) s.n_head_kv, n_chunks, ob); break;
                    case 1: ps_value_kernel<2><<<g2, HD / 2, 0, st>>>(pools, sb, scratch, (int) s.n_head_kv, n_chunks, ob); break;
                    case 2: ps_value_st_kernel<1, 32><<<g2, HD, 0, st>>>(pools, sb, scratch, (int) s.n_head_kv, n_chunks, ob); break;
                    case 3: ps_value_st_kernel<1, 64><<<g2, HD, 0, st>>>(pools, sb, scratch, (int) s.n_head_kv, n_chunks, ob); break;
                    case 4: ps_value_st_kernel<2, 32><<<g2, HD / 2, 0, st>>>(pools, sb, scratch, (int) s.n_head_kv, n_chunks, ob); break;
                    case 5: ps_value_st_kernel<2, 64><<<g2, HD / 2, 0, st>>>(pools, sb, scratch, (int) s.n_head_kv, n_chunks, ob); break;
                    case 6: ps_value_d4_kernel<<<g2, 64, 0, st>>>(pools, sb, scratch, (int) s.n_head_kv, n_chunks, ob); break;
                    case 7: ps_value_d4h_kernel<2><<<g2, 128, 0, st>>>(pools, sb, scratch, (int) s.n_head_kv, n_chunks, ob); break;
                    default: ps_value_d4h_kernel<3><<<g2, 192, 0, st>>>(pools, sb, scratch, (int) s.n_head_kv, n_chunks, ob); break;
                }
            }
        }
    } else {
        return false;
    }
    const cudaError_t e = cudaGetLastError();
    if (e != cudaSuccess) { std::fprintf(stderr, "qsa_pattn_batch: %s\n", cudaGetErrorString(e)); std::exit(1); }
    return true;
}

void qsa_pattn_info(char* buf, int len) {
    int n = 0;
    auto add = [&](const char* name, const void* f) {
        cudaFuncAttributes at{};
        int per_sm = 0;
        if (cudaFuncGetAttributes(&at, f) != cudaSuccess ||
            cudaOccupancyMaxActiveBlocksPerMultiprocessor(&per_sm, f, THREADS, 0) != cudaSuccess) {
            cudaGetLastError();
            return;
        }
        if (n < len)
            n += std::snprintf(buf + n, (size_t) (len - n), "  %-24s regs %3d  smem %6zu  local %4zu  CTAs/SM %d\n", name,
                               at.numRegs, at.sharedSizeBytes, at.localSizeBytes, per_sm);
    };
    add("pa_chunk<2>", (const void*) pa_chunk_kernel<2>);
    add("ps_score <I2F>", (const void*) ps_score_kernel<false, false>);
    add("ps_score <I2F,r24>", (const void*) ps_score_kernel<false, true>);
    add("ps_score <magic,r24>", (const void*) ps_score_kernel<true, true>);
    add("ps_score_s", (const void*) ps_score_s_kernel);
    add("ps_score_x<1>", (const void*) ps_score_x_kernel<1>);
    add("ps_score_x<2>", (const void*) ps_score_x_kernel<2>);
    {
        cudaFuncAttributes at{};
        int a4 = 0;
        if (cudaFuncGetAttributes(&at, (const void*) ps_value_d4_kernel) == cudaSuccess &&
            cudaOccupancyMaxActiveBlocksPerMultiprocessor(&a4, (const void*) ps_value_d4_kernel, 64, 0) == cudaSuccess &&
            n < len)
            n += std::snprintf(buf + n, (size_t) (len - n), "  %-24s regs %3d  smem %6zu  local %4zu  CTAs/SM %d (64 thr)\n",
                               "ps_value_d4", at.numRegs, at.sharedSizeBytes, at.localSizeBytes, a4);
        if (cudaFuncGetAttributes(&at, (const void*) ps_value_d4h_kernel<2>) == cudaSuccess &&
            cudaOccupancyMaxActiveBlocksPerMultiprocessor(&a4, (const void*) ps_value_d4h_kernel<2>, 128, 0) == cudaSuccess &&
            n < len)
            n += std::snprintf(buf + n, (size_t) (len - n), "  %-24s regs %3d  smem %6zu  local %4zu  CTAs/SM %d (128 thr)\n",
                               "ps_value_d4h<2>", at.numRegs, at.sharedSizeBytes, at.localSizeBytes, a4);
        if (cudaFuncGetAttributes(&at, (const void*) ps_value_d4h_kernel<3>) == cudaSuccess &&
            cudaOccupancyMaxActiveBlocksPerMultiprocessor(&a4, (const void*) ps_value_d4h_kernel<3>, 192, 0) == cudaSuccess &&
            n < len)
            n += std::snprintf(buf + n, (size_t) (len - n), "  %-24s regs %3d  smem %6zu  local %4zu  CTAs/SM %d (192 thr)\n",
                               "ps_value_d4h<3>", at.numRegs, at.sharedSizeBytes, at.localSizeBytes, a4);
        cudaGetLastError();
    }
    add("ps_value_st<1,32>", (const void*) ps_value_st_kernel<1, 32>);
    add("ps_value_st<1,64>", (const void*) ps_value_st_kernel<1, 64>);
    {
        cudaFuncAttributes at{};
        int a1 = 0, a2 = 0;
        if (cudaFuncGetAttributes(&at, (const void*) ps_value_kernel<1>) == cudaSuccess &&
            cudaOccupancyMaxActiveBlocksPerMultiprocessor(&a1, (const void*) ps_value_kernel<1>, HD, 0) == cudaSuccess &&
            n < len)
            n += std::snprintf(buf + n, (size_t) (len - n), "  %-24s regs %3d  smem %6zu  local %4zu  CTAs/SM %d (256 thr)\n",
                               "ps_value<1>", at.numRegs, at.sharedSizeBytes, at.localSizeBytes, a1);
        if (cudaFuncGetAttributes(&at, (const void*) ps_value_kernel<2>) == cudaSuccess &&
            cudaOccupancyMaxActiveBlocksPerMultiprocessor(&a2, (const void*) ps_value_kernel<2>, HD / 2, 0) == cudaSuccess &&
            n < len)
            n += std::snprintf(buf + n, (size_t) (len - n), "  %-24s regs %3d  smem %6zu  local %4zu  CTAs/SM %d (128 thr)\n",
                               "ps_value<2>", at.numRegs, at.sharedSizeBytes, at.localSizeBytes, a2);
        cudaGetLastError();
    }
    add("pf_fused v2 <V0,I2F>", (const void*) pf_fused_kernel<FUSED_MINB, 0, false, false>);
    add("pf_fused v3 <V1,I2F>", (const void*) pf_fused_kernel<FUSED_MINB, 1, false, false>);
    add("pf_fused v4 <V2,I2F>", (const void*) pf_fused_kernel<FUSED_MINB, 2, false, false>);
    add("pf_fused v5 <V0,magic>", (const void*) pf_fused_kernel<FUSED_MINB, 0, true, false>);
    add("pf_fused v6 <V2,magic>", (const void*) pf_fused_kernel<FUSED_MINB, 2, true, false>);
    add("pf_fused v7 <V0,I2F,r24>", (const void*) pf_fused_kernel<FUSED_MINB, 0, false, true>);
    add("pf_fused v8 <V2,magic,r24>", (const void*) pf_fused_kernel<FUSED_MINB, 2, true, true>);
    add("pf_fused v9 <V2,I2F,r24>", (const void*) pf_fused_kernel<FUSED_MINB, 2, false, true>);
}

#else  // not the experimental sm_60 CUDA build
bool qsa_pattn_batch(const float*, const QsaAttnPools&, const int32_t*, const int32_t*, int64_t, const QsaShapes&,
                     float*, uint64_t, float*, int64_t, void*, int) { return false; }
void qsa_pattn_info(char* buf, int len) { if (len > 0) buf[0] = 0; }
#endif

}  // namespace strata::kernels
