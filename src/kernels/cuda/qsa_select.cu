// src/kernels/cuda/qsa_select.cu - see include/strata/kernels/qsa_select.hpp.
#include "strata/core/emulate.hpp"
#include "strata/core/geometry.hpp"
#include <cstdlib>
#include <cstring>
#include "strata/kernels/qsa_select.hpp"
#include "strata/kernels/gfx_arch.hpp"

#include <cuda_runtime.h>

#include <cfloat>
#include <cstdio>
#include <cstdlib>
#include <cstring>

namespace strata::kernels {
namespace {

constexpr int IDX_DIM = 128, IDX_HEADS = 4, R = 4;
static_assert(IDX_DIM == strata::core::kCompiledGeometry.idx_key_dim && IDX_HEADS == strata::core::kCompiledGeometry.idx_q_heads && R == strata::core::kCompiledGeometry.idx_block,
              "qsa_select is compiled for core/geometry.hpp");
constexpr int SCORE_WARPS = 8;
constexpr int TOPK_T = 256;

__device__ __forceinline__ uint32_t order_key(float s) {
    const float v = s + 0.0f;
    if (!(v == v)) return 0u;
    const uint32_t b = __float_as_uint(v);
    return (b & 0x80000000u) ? ~b : (b | 0x80000000u);
}

__global__ void __launch_bounds__(SCORE_WARPS * 32) block_scores_kernel(const float* __restrict__ pooled,
                                                                        const float* __restrict__ dead,
                                                                        const float* __restrict__ q_idx,
                                                                        const int32_t* __restrict__ steps,
                                                                        int64_t max_blocks, float* __restrict__ out) {
    const int64_t qi = blockIdx.y;
    const int32_t* st = steps + qi * kStepCount;
    const int64_t n_kv = st[kStepNKv], n_bid = st[kStepNBid];
    const int64_t b = (int64_t) blockIdx.x * SCORE_WARPS + (threadIdx.x >> 5);
    if (b > n_bid || b >= max_blocks) return;
    const int lane = threadIdx.x & 31;
    const float* key = (b == n_bid) ? dead : pooled + b * IDX_DIM;
    const float4 k4 = *reinterpret_cast<const float4*>(key + lane * 4);
    const float* q = q_idx + qi * IDX_HEADS * IDX_DIM + lane * 4;
    float score = 0.0f;
#pragma unroll
    for (int h = 0; h < IDX_HEADS; ++h) {
        const float4 q4 = *reinterpret_cast<const float4*>(q + h * IDX_DIM);
        float d = k4.x * q4.x + k4.y * q4.y + k4.z * q4.z + k4.w * q4.w;
#pragma unroll
        for (int o = 16; o > 0; o >>= 1) d += __shfl_xor_sync(0xffffffffu, d, o);
        score += d > 0.0f ? d : 0.0f;
    }
    if (lane == 0) {
        if (b == n_bid && n_kv % R != 0) score += 1e9f;
        out[qi * max_blocks + b] = score;
    }
}

__global__ void __launch_bounds__(TOPK_T) block_topk_kernel(const float* __restrict__ scores,
                                                            const int32_t* __restrict__ steps, int64_t max_blocks,
                                                            int64_t cap, int32_t* __restrict__ ids) {
    __shared__ int hist[256];
    __shared__ int s_a[TOPK_T], s_b[TOPK_T];
    __shared__ int s_digit, s_above;
    const int64_t qi = blockIdx.x;
    const int32_t* st = steps + qi * kStepCount;
    const int64_t n_kv = st[kStepNKv], n_bid = st[kStepNBid], width = st[kStepWidth];
    int32_t* out = ids + qi * cap;
    const int t = threadIdx.x;
    if (n_kv <= width) {                               // everything is selected: the identity, ascending
        for (int64_t j = t; j < n_kv; j += TOPK_T) out[j] = (int32_t) j;
        return;
    }
    const float* sc = scores + qi * max_blocks;
    const int64_t nb = n_bid + 1;                      // blocks 0..n_bid, the last possibly empty
    const int64_t per = (nb + TOPK_T - 1) / TOPK_T;
    const int64_t b0 = (int64_t) t * per, b1 = (b0 + per < nb) ? b0 + per : nb;
    auto weight = [&](int64_t b) -> int { return b < n_bid ? R : (int) (n_kv - n_bid * R); };
    // ---- radix select: the largest key thr with (cells with key >= thr) >= width, 8 bits at a time
    uint32_t prefix = 0;
    int above = 0;                                     // cells strictly above the digits fixed so far
    for (int shift = 24; shift >= 0; shift -= 8) {
        for (int i = t; i < 256; i += TOPK_T) hist[i] = 0;
        __syncthreads();
        const uint32_t hi_mask = shift == 24 ? 0u : (0xffffffffu << (shift + 8));
        for (int64_t b = b0; b < b1; ++b) {
            const int w = weight(b);
            if (w == 0) continue;
            const uint32_t k = order_key(sc[b]);
            if ((k & hi_mask) == (prefix & hi_mask)) atomicAdd(&hist[(k >> shift) & 255], w);
        }
        __syncthreads();
        if (t == 0) {
            int cum = above, d = 255;
            for (; d > 0; --d) {
                if (cum + hist[d] >= width) break;
                cum += hist[d];
            }
            s_digit = d;
            s_above = cum;
        }
        __syncthreads();
        prefix |= (uint32_t) s_digit << shift;
        above = s_above;
        __syncthreads();
    }
    const uint32_t thr = prefix;
    const int64_t eq_budget = width - above;          // cells equal to thr that fit, lowest index first
    // ---- per-thread counts of cells above and at the threshold, then their exclusive prefixes
    int gt = 0, eq = 0;
    for (int64_t b = b0; b < b1; ++b) {
        const int w = weight(b);
        if (w == 0) continue;
        const uint32_t k = order_key(sc[b]);
        if (k > thr) gt += w;
        else if (k == thr) eq += w;
    }
    s_a[t] = gt;
    s_b[t] = eq;
    __syncthreads();
    if (t == 0) {
        int ag = 0, ae = 0;
        for (int i = 0; i < TOPK_T; ++i) {
            const int g = s_a[i], e = s_b[i];
            s_a[i] = ag; s_b[i] = ae;
            ag += g; ae += e;
        }
    }
    __syncthreads();
    const int64_t eq_before = s_b[t];
    int64_t my_eq = eq_budget - eq_before;
    if (my_eq < 0) my_eq = 0;
    if (my_eq > eq) my_eq = eq;
    const int sel = gt + (int) my_eq;
    __syncthreads();
    s_a[t] = sel;
    __syncthreads();
    if (t == 0) {
        int a = 0;
        for (int i = 0; i < TOPK_T; ++i) { const int c = s_a[i]; s_a[i] = a; a += c; }
    }
    __syncthreads();
    int64_t wpos = s_a[t];
    int64_t eq_left = my_eq;
    for (int64_t b = b0; b < b1; ++b) {
        const int w = weight(b);
        if (w == 0) continue;
        const uint32_t k = order_key(sc[b]);
        if (k > thr) {
            for (int c = 0; c < w; ++c) out[wpos++] = (int32_t) (b * R + c);
        } else if (k == thr) {
            for (int c = 0; c < w && eq_left > 0; ++c, --eq_left) out[wpos++] = (int32_t) (b * R + c);
        }
    }
}


// ---- QSA select on tensor cores (perf-review, after D-1): the block scores of many queries are one GEMM,
// rows (query, indexer head) x columns (blocks), K = 128, with relu per head summed. 3xTF32 (each operand split
// into a TF32 hi and lo part, hi*hi + hi*lo + lo*hi) keeps FP32-level accuracy; the summation order differs from
// the warp kernel, so a score can move in its last bits and a near-tie can select differently (not bitwise).
// Blocks < n_bid only; the tail block n_bid (the `dead` key, +1e9) is scored by the warp kernel's own code.
constexpr int TC_QT = 16;                 // queries per CTA (one m16 tile per indexer head)
constexpr int TC_NB = 32;                 // blocks per tile (4 warps x n8)
constexpr int TC_ITER = 4;                // tiles per CTA (the query tile is loaded once)
constexpr int TC_QS = IDX_HEADS * IDX_DIM + 4;   // query row stride in floats
constexpr int TC_KS = IDX_DIM + 4;               // key row stride

// TF32 conversion and MMA need sm_80: below it they compile to a trap and qsa_block_scores_tc refuses the device
#if defined(__HIPCC__)          // AMD: no mma.sync / cp.async; the host keeps the warp kernel (below)
#define STRATA_SEL_SM80 0
#elif !defined(__CUDA_ARCH__) || __CUDA_ARCH__ >= 800
#define STRATA_SEL_SM80 1
#else
#define STRATA_SEL_SM80 0
#endif
__device__ __forceinline__ uint32_t tf32_hi(float x) {
#if STRATA_SEL_SM80
    uint32_t r;
    asm("cvt.rna.tf32.f32 %0, %1;" : "=r"(r) : "f"(x));
    return r;
#else
    return __float_as_uint(x);
#endif
}
__device__ __forceinline__ void mma_tf32(float* c, const uint32_t* a, const uint32_t* b) {
#if !STRATA_SEL_SM80
    __trap();
#else
    asm volatile("mma.sync.aligned.m16n8k8.row.col.f32.tf32.tf32.f32 {%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, "
                 "{%0,%1,%2,%3};\n"
                 : "+f"(c[0]), "+f"(c[1]), "+f"(c[2]), "+f"(c[3])
                 : "r"(a[0]), "r"(a[1]), "r"(a[2]), "r"(a[3]), "r"(b[0]), "r"(b[1]));
#endif
}

__global__ void __launch_bounds__(128) block_scores_tc_kernel(const float* __restrict__ pooled,
                                                              const float* __restrict__ q_idx,
                                                              const int32_t* __restrict__ steps, int64_t nq,
                                                              int64_t max_blocks, int64_t reach,
                                                              float* __restrict__ out) {
    extern __shared__ __align__(16) float sm[];
    float* sQ = sm;                                   // [TC_QT][TC_QS]
    float* sK = sm + TC_QT * TC_QS;                   // [TC_NB][TC_KS]
    __shared__ int s_nbid[TC_QT];
    const int t = threadIdx.x, lane = t & 31, warp = t >> 5, gid = lane >> 2, tig = lane & 3;
    const int64_t q0 = (int64_t) blockIdx.y * TC_QT;
    for (int i = t; i < TC_QT * IDX_HEADS * IDX_DIM / 4; i += 128) {
        const int r = i / (IDX_HEADS * IDX_DIM / 4), c = i % (IDX_HEADS * IDX_DIM / 4);
        float4 v = make_float4(0.f, 0.f, 0.f, 0.f);
        if (q0 + r < nq) v = reinterpret_cast<const float4*>(q_idx + (q0 + r) * IDX_HEADS * IDX_DIM)[c];
        *reinterpret_cast<float4*>(sQ + r * TC_QS + c * 4) = v;
    }
    if (t < TC_QT) s_nbid[t] = q0 + t < nq ? steps[(q0 + t) * kStepCount + kStepNBid] : 0;
    int lo_nbid = 0x7fffffff, hi_nbid = 0;
    __syncthreads();
    for (int i = 0; i < TC_QT; ++i) {
        if (q0 + i >= nq) break;
        lo_nbid = min(lo_nbid, s_nbid[i]);
        hi_nbid = max(hi_nbid, s_nbid[i]);
    }
    for (int it = 0; it < TC_ITER; ++it) {
        const int64_t b0 = ((int64_t) blockIdx.x * TC_ITER + it) * TC_NB;
        if (b0 >= reach || b0 >= hi_nbid) break;      // blocks >= every query's n_bid: nothing to score
        __syncthreads();                              // the previous tile's reads are done
        for (int i = t; i < TC_NB * IDX_DIM / 4; i += 128) {
            const int r = i / (IDX_DIM / 4), c = i % (IDX_DIM / 4);
            float4 v = make_float4(0.f, 0.f, 0.f, 0.f);
            if (b0 + r < hi_nbid) v = reinterpret_cast<const float4*>(pooled + (b0 + r) * IDX_DIM)[c];
            *reinterpret_cast<float4*>(sK + r * TC_KS + c * 4) = v;
        }
        __syncthreads();
        float acc[IDX_HEADS][4];
#pragma unroll
        for (int h = 0; h < IDX_HEADS; ++h) acc[h][0] = acc[h][1] = acc[h][2] = acc[h][3] = 0.f;
        const float* kr = sK + (warp * 8 + gid) * TC_KS;
#pragma unroll 4
        for (int k0 = 0; k0 < IDX_DIM; k0 += 8) {
            const float kx0 = kr[k0 + tig], kx1 = kr[k0 + tig + 4];
            uint32_t bh[2], bl[2];
            bh[0] = tf32_hi(kx0);
            bh[1] = tf32_hi(kx1);
            bl[0] = tf32_hi(kx0 - __uint_as_float(bh[0]));
            bl[1] = tf32_hi(kx1 - __uint_as_float(bh[1]));
#pragma unroll
            for (int h = 0; h < IDX_HEADS; ++h) {
                const float* qa = sQ + h * IDX_DIM + k0 + tig;
                const float x0 = qa[gid * TC_QS], x1 = qa[(gid + 8) * TC_QS];
                const float x2 = qa[gid * TC_QS + 4], x3 = qa[(gid + 8) * TC_QS + 4];
                uint32_t ah[4], al[4];
                ah[0] = tf32_hi(x0); ah[1] = tf32_hi(x1); ah[2] = tf32_hi(x2); ah[3] = tf32_hi(x3);
                al[0] = tf32_hi(x0 - __uint_as_float(ah[0]));
                al[1] = tf32_hi(x1 - __uint_as_float(ah[1]));
                al[2] = tf32_hi(x2 - __uint_as_float(ah[2]));
                al[3] = tf32_hi(x3 - __uint_as_float(ah[3]));
                mma_tf32(acc[h], al, bh);
                mma_tf32(acc[h], ah, bl);
                mma_tf32(acc[h], ah, bh);
            }
        }
        // relu per head, heads added in order (as the warp kernel), written where the block completed for the query
        const int64_t bc = b0 + warp * 8 + 2 * tig;
#pragma unroll
        for (int half = 0; half < 2; ++half) {
            const int qr = gid + half * 8;
            const int64_t qi = q0 + qr;
            if (qi >= nq) continue;
            const int nb = s_nbid[qr];
#pragma unroll
            for (int j = 0; j < 2; ++j) {
                const int64_t b = bc + j;
                if (b >= nb || b >= max_blocks) continue;
                float score = 0.0f;
#pragma unroll
                for (int h = 0; h < IDX_HEADS; ++h) {
                    const float d = acc[h][half * 2 + j];
                    score += d > 0.0f ? d : 0.0f;
                }
                out[qi * max_blocks + b] = score;
            }
        }
    }
}

#if defined(__HIPCC__)
// ---- gfx12 (RDNA4: gfx1200 / gfx1201) scorer on v_wmma_f32_16x16x16_bf16 (wave32).  Same GEMM as above: for a tile of 16
// queries, every block's score is sum over the 4 indexer heads of relu(q_h . k), K = 128.  FP32-level accuracy from a
// three-way bf16 split of both operands (x = hi + mid + lo EXACTLY - the top 8, next 8 and last 8 bits of the fp32
// mantissa, so bf16's fp32-sized exponent range needs no scaling) and the six products of order <= 2, smallest first;
// like the TF32 kernel it sums in another order than the warp kernel (not bitwise).  The tail block n_bid is the warp
// kernel's arithmetic (block_scores_tail_kernel).
//   D = A x B with A = 16 key blocks (row l%16 of a lane, k = 8*(l/16)+i), B = 16 queries (column l%16, same k):
//   a lane ends up with 8 consecutive blocks of ONE query, written as one run.  No LDS for keys: a lane reads its
//   32-byte slice of the key row from global (the 16 query tiles of a launch re-read them from L2); the queries are
//   split once per CTA into LDS.
// gfx11 (RDNA3 / RDNA3.5) has the same instruction with another fragment layout: a lane holds all 16 k of its row
// (A) or column (B) - lanes 16..31 repeat lanes 0..15 - and D[2i + l/16][l%16]; SEL_KL is the k per lane, SEL_KOFF(g)
// the first, SEL_ROW(i, g) the D row of element i.  Same products, same order per element as on gfx12.
#if defined(__gfx1200__) || defined(__gfx1201__)
#define STRATA_SEL_GFX12 1
#define STRATA_SEL_GFX11 0
#elif defined(__gfx1100__) || defined(__gfx1101__) || defined(__gfx1102__) || defined(__gfx1103__) || defined(__gfx1150__) || \
    defined(__gfx1151__)
#define STRATA_SEL_GFX12 0
#define STRATA_SEL_GFX11 1
#else
#define STRATA_SEL_GFX12 0
#define STRATA_SEL_GFX11 0
#endif
#if STRATA_SEL_GFX11
constexpr int SEL_KL = 16;
#define SEL_KOFF(g) 0
#define SEL_ROW(i, g) (2 * (i) + (g))
#else
constexpr int SEL_KL = 8;
#define SEL_KOFF(g) ((g) * 8)
#define SEL_ROW(i, g) ((g) * 8 + (i))
#endif
typedef short sel_s8 __attribute__((ext_vector_type(SEL_KL)));
typedef float sel_f8 __attribute__((ext_vector_type(8)));
typedef uint32_t sel_u4 __attribute__((ext_vector_type(SEL_KL / 2)));
constexpr int WQT = 16;                    // queries per CTA (the N of one WMMA)
constexpr int WITER = 4;                   // key tiles per warp (the CTA covers 4 warps * WITER * 16 blocks)
constexpr int WQS = IDX_DIM + 8;           // bf16 elements per LDS row: 272 bytes, conflict-free 16-byte reads

__device__ __forceinline__ sel_f8 wmma_bf16(const sel_s8& a, const sel_s8& b, const sel_f8& c) {
#if STRATA_SEL_GFX12
    return __builtin_amdgcn_wmma_f32_16x16x16_bf16_w32_gfx12(a, b, c);
#elif STRATA_SEL_GFX11
    return __builtin_amdgcn_wmma_f32_16x16x16_bf16_w32(a, b, c);
#else
    __trap();
    return c;
#endif
}
// the bf16 pieces of x as fp32 bit patterns whose upper 16 bits are the bf16 (lower 16 are zero)
__device__ __forceinline__ void split3(float x, uint32_t& hi, uint32_t& mid, uint32_t& lo) {
    hi = __float_as_uint(x) & 0xffff0000u;
    const float r1 = x - __uint_as_float(hi);                   // exact
    mid = __float_as_uint(r1) & 0xffff0000u;
    lo = __float_as_uint(r1 - __uint_as_float(mid)) & 0xffff0000u;   // exact: at most 8 significant bits left
}
__device__ __forceinline__ uint32_t pack_bf16x2(uint32_t a, uint32_t b) {   // low half = a's bf16, high half = b's
#if STRATA_SEL_GFX12 || STRATA_SEL_GFX11
    return __builtin_amdgcn_perm(b, a, 0x07060302u);
#else
    return (a >> 16) | (b & 0xffff0000u);
#endif
}

__global__ void __launch_bounds__(128) block_scores_wmma_kernel(const float* __restrict__ pooled,
                                                                const float* __restrict__ q_idx,
                                                                const int32_t* __restrict__ steps, int64_t nq,
                                                                int64_t max_blocks, int64_t reach,
                                                                float* __restrict__ out) {
    __shared__ __align__(16) uint16_t sq[3][IDX_HEADS][WQT][WQS];   // hi / mid / lo, [head][query][dim]
    __shared__ int s_nbid[WQT];
    const int t = threadIdx.x, lane = t & 31, warp = t >> 5, l16 = lane & 15, g = lane >> 4;
    const int64_t q0 = (int64_t) blockIdx.y * WQT;
    for (int i = t; i < WQT * IDX_HEADS * IDX_DIM / 4; i += 128) {
        const int row = i / (IDX_DIM / 4), c = i % (IDX_DIM / 4);
        const int qr = row / IDX_HEADS, h = row % IDX_HEADS;
        float4 v = make_float4(0.f, 0.f, 0.f, 0.f);
        if (q0 + qr < nq) v = reinterpret_cast<const float4*>(q_idx + (q0 + qr) * IDX_HEADS * IDX_DIM)[h * (IDX_DIM / 4) + c];
        const float x[4] = {v.x, v.y, v.z, v.w};
        uint32_t hb[4], mb[4], lb[4];
#pragma unroll
        for (int j = 0; j < 4; ++j) split3(x[j], hb[j], mb[j], lb[j]);
        *reinterpret_cast<uint2*>(&sq[0][h][qr][c * 4]) = make_uint2(pack_bf16x2(hb[0], hb[1]), pack_bf16x2(hb[2], hb[3]));
        *reinterpret_cast<uint2*>(&sq[1][h][qr][c * 4]) = make_uint2(pack_bf16x2(mb[0], mb[1]), pack_bf16x2(mb[2], mb[3]));
        *reinterpret_cast<uint2*>(&sq[2][h][qr][c * 4]) = make_uint2(pack_bf16x2(lb[0], lb[1]), pack_bf16x2(lb[2], lb[3]));
    }
    if (t < WQT) s_nbid[t] = q0 + t < nq ? steps[(q0 + t) * kStepCount + kStepNBid] : 0;
    __syncthreads();
    int hi_nbid = 0;
#pragma unroll
    for (int i = 0; i < WQT; ++i) hi_nbid = max(hi_nbid, s_nbid[i]);
    const int64_t qi = q0 + l16;
    const int nb_q = s_nbid[l16];
    for (int it = 0; it < WITER; ++it) {
        const int64_t b0 = (((int64_t) blockIdx.x * WITER + it) * 4 + warp) * 16;
        if (b0 >= reach || b0 >= hi_nbid) break;          // warp-uniform; later tiles start higher
        const int64_t row = b0 + l16;
        const bool rv = row < hi_nbid && row < max_blocks;
        const float* kp = pooled + (rv ? row : 0) * IDX_DIM + SEL_KOFF(g);
        // acc: the hi*hi products; cor: the five smaller ones (<= 2^-8 of it). Kept apart and added once at the end: a
        // correction summed into the big accumulator is rounded to ITS ulp at every one of the 48 steps (measured: 8e-7
        // of the score scale; apart, near the warp kernel's)
        sel_f8 acc[IDX_HEADS], cor[IDX_HEADS];
#pragma unroll
        for (int h = 0; h < IDX_HEADS; ++h) acc[h] = cor[h] = sel_f8{0.f, 0.f, 0.f, 0.f, 0.f, 0.f, 0.f, 0.f};
#pragma unroll 2
        for (int kk = 0; kk < IDX_DIM / 16; ++kk) {
            float x[SEL_KL];
#pragma unroll
            for (int j = 0; j < SEL_KL / 4; ++j) {
                float4 kv = make_float4(0.f, 0.f, 0.f, 0.f);
                if (rv) kv = *reinterpret_cast<const float4*>(kp + kk * 16 + 4 * j);
                x[4 * j] = kv.x; x[4 * j + 1] = kv.y; x[4 * j + 2] = kv.z; x[4 * j + 3] = kv.w;
            }
            uint32_t hb[SEL_KL], mb[SEL_KL], lb[SEL_KL];
#pragma unroll
            for (int j = 0; j < SEL_KL; ++j) split3(x[j], hb[j], mb[j], lb[j]);
            sel_u4 ah, am, al;
#pragma unroll
            for (int j = 0; j < SEL_KL / 2; ++j) {
                ah[j] = pack_bf16x2(hb[2 * j], hb[2 * j + 1]);
                am[j] = pack_bf16x2(mb[2 * j], mb[2 * j + 1]);
                al[j] = pack_bf16x2(lb[2 * j], lb[2 * j + 1]);
            }
            const sel_s8 Ah = __builtin_bit_cast(sel_s8, ah), Am = __builtin_bit_cast(sel_s8, am),
                         Al = __builtin_bit_cast(sel_s8, al);
#pragma unroll
            for (int h = 0; h < IDX_HEADS; ++h) {
                const sel_s8 Bh = *reinterpret_cast<const sel_s8*>(&sq[0][h][l16][kk * 16 + SEL_KOFF(g)]);
                const sel_s8 Bm = *reinterpret_cast<const sel_s8*>(&sq[1][h][l16][kk * 16 + SEL_KOFF(g)]);
                const sel_s8 Bl = *reinterpret_cast<const sel_s8*>(&sq[2][h][l16][kk * 16 + SEL_KOFF(g)]);
                cor[h] = wmma_bf16(Al, Bh, cor[h]);
                cor[h] = wmma_bf16(Ah, Bl, cor[h]);
                cor[h] = wmma_bf16(Am, Bm, cor[h]);
                cor[h] = wmma_bf16(Ah, Bm, cor[h]);
                cor[h] = wmma_bf16(Am, Bh, cor[h]);
                acc[h] = wmma_bf16(Ah, Bh, acc[h]);
            }
        }
        // relu per head, heads added in order (as the warp kernel); this lane: query qi, blocks b0 + SEL_ROW(i, g)
        if (qi < nq) {
#pragma unroll
            for (int i = 0; i < 8; ++i) {
                const int64_t b = b0 + SEL_ROW(i, g);
                if (b >= nb_q || b >= max_blocks) continue;
                float score = 0.0f;
#pragma unroll
                for (int h = 0; h < IDX_HEADS; ++h) {
                    const float d = acc[h][i] + cor[h][i];
                    score += d > 0.0f ? d : 0.0f;
                }
                out[qi * max_blocks + b] = score;
            }
        }
    }
}

// the WMMA kernel needs gfx12 (gfx1200/gfx1201) or gfx11 (gfx110x/gfx115x: the other fragment layout) code objects
bool sel_gfx12_device() {
    static int ok[64] = {};   // per device: 0 unknown, 1 yes, 2 no
    int dev = 0;
    if (cudaGetDevice(&dev) != cudaSuccess || dev < 0 || dev >= 64) { cudaGetLastError(); return false; }
    if (ok[dev] == 0) {
        cudaDeviceProp prop;
        ok[dev] = (cudaGetDeviceProperties(&prop, dev) == cudaSuccess &&
                   (std::strncmp(prop.gcnArchName, "gfx1200", 7) == 0 || std::strncmp(prop.gcnArchName, "gfx1201", 7) == 0 ||
                    strata::kernels::gfx_arch_is_gfx11_wmma(prop.gcnArchName)))
                      ? 1 : 2;
        cudaGetLastError();
    }
    return ok[dev] == 1;
}
#endif  // __HIPCC__

// the tail block n_bid of each query: exactly block_scores_kernel's arithmetic for that block
__global__ void __launch_bounds__(32) block_scores_tail_kernel(const float* __restrict__ dead,
                                                               const float* __restrict__ q_idx,
                                                               const int32_t* __restrict__ steps,
                                                               int64_t max_blocks, float* __restrict__ out) {
    const int64_t qi = blockIdx.x;
    const int32_t* st = steps + qi * kStepCount;
    const int64_t n_kv = st[kStepNKv], n_bid = st[kStepNBid];
    if (n_bid >= max_blocks) return;
    const int lane = threadIdx.x & 31;
    const float4 k4 = *reinterpret_cast<const float4*>(dead + lane * 4);
    const float* q = q_idx + qi * IDX_HEADS * IDX_DIM + lane * 4;
    float score = 0.0f;
#pragma unroll
    for (int h = 0; h < IDX_HEADS; ++h) {
        const float4 q4 = *reinterpret_cast<const float4*>(q + h * IDX_DIM);
        float d = k4.x * q4.x + k4.y * q4.y + k4.z * q4.z + k4.w * q4.w;
#pragma unroll
        for (int o = 16; o > 0; o >>= 1) d += __shfl_xor_sync(0xffffffffu, d, o);
        score += d > 0.0f ? d : 0.0f;
    }
    if (lane == 0) {
        if (n_kv % R != 0) score += 1e9f;
        out[qi * max_blocks + n_bid] = score;
    }
}

// ---- the block scores as an FP32 tiled GEMM for CUDA cards without TF32 tensor cores (below sm_80): rows (query,
// indexer head) x columns (blocks), K = 128 through shared memory, 2 queries x 4 heads x 4 blocks per thread, relu per
// head summed in registers.  The warp kernel spends 5 shuffles per 4 products and reached ~1 TFLOPS on an RTX 2080
// Ti; this one ~6.9 (256 queries: 4.5x at 4K, 6.8x at 120K-250K).  FP32 like the warp kernel in another summation
// order, so not bitwise (as the tensor-core scorer).  Blocks < n_bid only; the tail block is block_scores_tail_kernel's.
constexpr int SM_QT = 32, SM_NB = 64, SM_KC = 32, SM_QS = SM_QT * IDX_HEADS + 4, SM_KS = SM_NB + 4;
__global__ void __launch_bounds__(256) block_scores_simt_kernel(const float* __restrict__ pooled,
                                                                const float* __restrict__ q_idx,
                                                                const int32_t* __restrict__ steps, int64_t nq,
                                                                int64_t max_blocks, int64_t reach,
                                                                float* __restrict__ out) {
    __shared__ __align__(16) float Qs[SM_KC][SM_QS];
    __shared__ __align__(16) float Ks[SM_KC][SM_KS];
    const int64_t q0 = (int64_t) blockIdx.y * SM_QT, b0 = (int64_t) blockIdx.x * SM_NB;
    const int64_t qlast = (q0 + SM_QT < nq ? q0 + SM_QT : nq) - 1;
    if (b0 >= steps[qlast * kStepCount + kStepNBid]) return;   // n_bid rises with the position
    const int tid = threadIdx.x, tx = tid & 15, ty = tid >> 4;
    float acc[8][4] = {};
    for (int kc = 0; kc < IDX_DIM; kc += SM_KC) {
#pragma unroll
        for (int i = 0; i < 4; ++i) {   // the query rows: 128 x 8 float4
            const int idx = tid + i * 256, r = idx >> 3, k4 = idx & 7;
            float4 v = make_float4(0.f, 0.f, 0.f, 0.f);
            if (q0 + (r >> 2) < nq)
                v = *reinterpret_cast<const float4*>(q_idx + (q0 * IDX_HEADS + r) * IDX_DIM + kc + k4 * 4);
            Qs[k4 * 4 + 0][r] = v.x; Qs[k4 * 4 + 1][r] = v.y; Qs[k4 * 4 + 2][r] = v.z; Qs[k4 * 4 + 3][r] = v.w;
        }
#pragma unroll
        for (int i = 0; i < 2; ++i) {   // the pooled keys: 64 x 8 float4
            const int idx = tid + i * 256, c = idx >> 3, k4 = idx & 7;
            float4 v = make_float4(0.f, 0.f, 0.f, 0.f);
            if (b0 + c < reach) v = *reinterpret_cast<const float4*>(pooled + (b0 + c) * IDX_DIM + kc + k4 * 4);
            Ks[k4 * 4 + 0][c] = v.x; Ks[k4 * 4 + 1][c] = v.y; Ks[k4 * 4 + 2][c] = v.z; Ks[k4 * 4 + 3][c] = v.w;
        }
        __syncthreads();
#pragma unroll 8
        for (int k = 0; k < SM_KC; ++k) {
            const float4 qa = *reinterpret_cast<const float4*>(&Qs[k][ty * 8]);
            const float4 qb = *reinterpret_cast<const float4*>(&Qs[k][ty * 8 + 4]);
            const float4 kv = *reinterpret_cast<const float4*>(&Ks[k][tx * 4]);
            const float qr[8] = {qa.x, qa.y, qa.z, qa.w, qb.x, qb.y, qb.z, qb.w}, kr[4] = {kv.x, kv.y, kv.z, kv.w};
#pragma unroll
            for (int i = 0; i < 8; ++i)
#pragma unroll
                for (int j = 0; j < 4; ++j) acc[i][j] = fmaf(qr[i], kr[j], acc[i][j]);
        }
        __syncthreads();
    }
#pragma unroll
    for (int i = 0; i < 2; ++i) {
        const int64_t q = q0 + ty * 2 + i;
        if (q >= nq) continue;
        const int64_t n_bid = steps[q * kStepCount + kStepNBid];
#pragma unroll
        for (int j = 0; j < 4; ++j) {
            const int64_t b = b0 + tx * 4 + j;
            if (b >= n_bid || b >= max_blocks) continue;
            float sc = 0.0f;
#pragma unroll
            for (int h = 0; h < IDX_HEADS; ++h) {
                const float d = acc[i * 4 + h][j];
                sc += d > 0.0f ? d : 0.0f;
            }
            out[q * max_blocks + b] = sc;
        }
    }
}

// ---- the same top-k with each query's keys read once: 1,024 threads hold up to TK_PER consecutive blocks' keys in
// registers (contexts up to 4 * 1024 * TK_PER cells), per-warp histograms, block-wide scans. The selection rule is
// block_topk_kernel's (radix threshold, ties to the lowest index, cells ascending): identical ids.
constexpr int TK_T = 1024;
constexpr int TK_PER = 33;
#if defined(__HIPCC__)
// AMD (RDNA, wave32): a 1,024-thread block may use 192 VGPRs per lane, so the keys of 66 blocks per thread fit - contexts
// up to 4 * 1024 * 66 = 270,336 cells (the 262,144 --max-context) keep the register kernel. NVIDIA's 64 registers per
// thread at 1,024 threads allow only TK_PER.
constexpr int TK_PER_MAX = 66;
#else
constexpr int TK_PER_MAX = TK_PER;
#endif

__device__ __forceinline__ int block_excl_scan(int v, int* s_warp, int& total) {
    const int lane = threadIdx.x & 31, warp = threadIdx.x >> 5;
    int x = v;
#pragma unroll
    for (int o = 1; o < 32; o <<= 1) {
        const int y = __shfl_up_sync(0xffffffffu, x, o);
        if (lane >= o) x += y;
    }
    if (lane == 31) s_warp[warp] = x;
    __syncthreads();
    if (warp == 0) {
        int w = s_warp[lane];
        int z = w;
#pragma unroll
        for (int o = 1; o < 32; o <<= 1) {
            const int y = __shfl_up_sync(0xffffffffu, z, o);
            if (lane >= o) z += y;
        }
        s_warp[lane] = z - w;               // exclusive per warp
        if (lane == 31) s_warp[32] = z;     // total
    }
    __syncthreads();
    const int r = s_warp[warp] + x - v;
    total = s_warp[32];
    __syncthreads();
    return r;
}

template <int PER>
__global__ void __launch_bounds__(TK_T) block_topk_reg_kernel(const float* __restrict__ scores,
                                                              const int32_t* __restrict__ steps, int64_t max_blocks,
                                                              int64_t cap, int32_t* __restrict__ ids) {
    __shared__ int hist[TK_T / 32][256];
    __shared__ int s_warp[33];
    __shared__ int s_digit, s_above;
    const int64_t qi = blockIdx.x;
    const int32_t* st = steps + qi * kStepCount;
    const int64_t n_kv = st[kStepNKv], n_bid = st[kStepNBid], width = st[kStepWidth];
    int32_t* out = ids + qi * cap;
    const int t = threadIdx.x, lane = t & 31, warp = t >> 5;
    if (n_kv <= width) {
        for (int64_t j = t; j < n_kv; j += TK_T) out[j] = (int32_t) j;
        return;
    }
    const float* sc = scores + qi * max_blocks;
    const int64_t nb = n_bid + 1;
    const int64_t per = (nb + TK_T - 1) / TK_T;       // <= PER (the caller checks)
    const int64_t b0 = (int64_t) t * per, b1 = (b0 + per < nb) ? b0 + per : nb;
    uint32_t key[PER];
#pragma unroll
    for (int j = 0; j < PER; ++j) key[j] = (b0 + j < b1) ? order_key(sc[b0 + j]) : 0u;
    auto weight = [&](int64_t b) -> int { return b < n_bid ? R : (int) (n_kv - n_bid * R); };
    uint32_t prefix = 0;
    int above = 0;
    for (int shift = 24; shift >= 0; shift -= 8) {
        for (int i = lane; i < 256; i += 32) hist[warp][i] = 0;
        __syncwarp();
        const uint32_t hi_mask = shift == 24 ? 0u : (0xffffffffu << (shift + 8));
#pragma unroll
        for (int j = 0; j < PER; ++j) {
            const int64_t b = b0 + j;
            if (b >= b1) break;
            const int w = weight(b);
            if (w == 0) continue;
            if ((key[j] & hi_mask) == (prefix & hi_mask)) atomicAdd(&hist[warp][(key[j] >> shift) & 255], w);
        }
        __syncthreads();
        if (t < 256) {                                // fold the warps' histograms into warp 0's
            int s = 0;
            for (int w2 = 0; w2 < TK_T / 32; ++w2) s += hist[w2][t];
            hist[0][t] = s;
        }
        __syncthreads();
        if (t == 0) {
            int cum = above, d = 255;
            for (; d > 0; --d) {
                if (cum + hist[0][d] >= width) break;
                cum += hist[0][d];
            }
            s_digit = d;
            s_above = cum;
        }
        __syncthreads();
        prefix |= (uint32_t) s_digit << shift;
        above = s_above;
        __syncthreads();
    }
    const uint32_t thr = prefix;
    const int64_t eq_budget = width - above;
    int gt = 0, eq = 0;
#pragma unroll
    for (int j = 0; j < PER; ++j) {
        const int64_t b = b0 + j;
        if (b >= b1) break;
        const int w = weight(b);
        if (w == 0) continue;
        if (key[j] > thr) gt += w;
        else if (key[j] == thr) eq += w;
    }
    int tot;
    const int eq_before = block_excl_scan(eq, s_warp, tot);
    int64_t my_eq = eq_budget - eq_before;
    if (my_eq < 0) my_eq = 0;
    if (my_eq > eq) my_eq = eq;
    const int sel = gt + (int) my_eq;
    int64_t wpos = block_excl_scan(sel, s_warp, tot);
    int64_t eq_left = my_eq;
#pragma unroll
    for (int j = 0; j < PER; ++j) {
        const int64_t b = b0 + j;
        if (b >= b1) break;
        const int w = weight(b);
        if (w == 0) continue;
        if (key[j] > thr) {
            for (int c = 0; c < w; ++c) out[wpos++] = (int32_t) (b * R + c);
        } else if (key[j] == thr) {
            for (int c = 0; c < w && eq_left > 0; ++c, --eq_left) out[wpos++] = (int32_t) (b * R + c);
        }
    }
}

#if !defined(__HIPCC__)
// ---- a query with more blocks than the register kernel holds (contexts past 4 * 1024 * TK_PER cells): the register
// kernel's threads, per-warp histograms and warp scans, with each key read from memory again on every pass.  A
// histogram has no order, so on the four radix passes thread t reads blocks t, t + 1024, ...: a warp reads 32
// neighbours at a time.  The cells are emitted ascending in the order (warp, row, lane): warp w holds the `per` rows
// of 32 consecutive blocks from block w * 32 * per.  block_topk_kernel's selection rule (radix threshold, ties to the
// lowest index): identical ids.
constexpr int TS_MAXQ = 4096;
__device__ int g_tk_fail[TS_MAXQ];   // block_topk_sample_kernel: 1 = the query is block_topk_wide_kernel<true>'s
template <bool ONLY_FAILED = false>
__global__ void __launch_bounds__(TK_T) block_topk_wide_kernel(const float* __restrict__ scores,
                                                               const int32_t* __restrict__ steps, int64_t max_blocks,
                                                               int64_t cap, int32_t* __restrict__ ids) {
    if (ONLY_FAILED && !g_tk_fail[blockIdx.x]) return;
    __shared__ int hist[TK_T / 32][256];
    __shared__ int s_gt[TK_T / 32], s_eq[TK_T / 32];
    __shared__ int s_digit, s_above;
    const int64_t qi = blockIdx.x;
    const int32_t* st = steps + qi * kStepCount;
    const int64_t n_kv = st[kStepNKv], n_bid = st[kStepNBid], width = st[kStepWidth];
    int32_t* out = ids + qi * cap;
    const int t = threadIdx.x, lane = t & 31, warp = t >> 5;
    if (n_kv <= width) {
        for (int64_t j = t; j < n_kv; j += TK_T) out[j] = (int32_t) j;
        return;
    }
    const float* sc = scores + qi * max_blocks;
    const int64_t nb = n_bid + 1;
    const int64_t per = (nb + TK_T - 1) / TK_T;       // rows of 32 blocks per warp
    auto weight = [&](int64_t b) -> int { return b < n_bid ? R : (int) (n_kv - n_bid * R); };
    uint32_t prefix = 0;
    int above = 0;
    for (int shift = 24; shift >= 0; shift -= 8) {
        for (int i = lane; i < 256; i += 32) hist[warp][i] = 0;
        __syncwarp();
        const uint32_t hi_mask = shift == 24 ? 0u : (0xffffffffu << (shift + 8));
        for (int64_t b = t; b < nb; b += TK_T) {
            const int w = weight(b);
            if (w == 0) continue;
            const uint32_t k = order_key(sc[b]);
            if ((k & hi_mask) == (prefix & hi_mask)) atomicAdd(&hist[warp][(k >> shift) & 255], w);
        }
        __syncthreads();
        if (t < 256) {                                // fold the warps' histograms into warp 0's
            int sum = 0;
            for (int w2 = 0; w2 < TK_T / 32; ++w2) sum += hist[w2][t];
            hist[0][t] = sum;
        }
        __syncthreads();
        if (t == 0) {
            int cum = above, d = 255;
            for (; d > 0; --d) {
                if (cum + hist[0][d] >= width) break;
                cum += hist[0][d];
            }
            s_digit = d;
            s_above = cum;
        }
        __syncthreads();
        prefix |= (uint32_t) s_digit << shift;
        above = s_above;
        __syncthreads();
    }
    const uint32_t thr = prefix;
    const int eq_budget = (int) (width - above);      // cells equal to thr that fit, lowest index first
    const int64_t w0 = (int64_t) warp * 32 * per;
    int gt = 0, eq = 0;
    for (int64_t b = w0 + lane; b < nb && b < w0 + 32 * per; b += 32) {
        const int w = weight(b);
        if (w == 0) continue;
        const uint32_t k = order_key(sc[b]);
        if (k > thr) gt += w;
        else if (k == thr) eq += w;
    }
#pragma unroll
    for (int o = 16; o > 0; o >>= 1) {
        gt += __shfl_xor_sync(0xffffffffu, gt, o);
        eq += __shfl_xor_sync(0xffffffffu, eq, o);
    }
    if (lane == 0) { s_gt[warp] = gt; s_eq[warp] = eq; }
    __syncthreads();
    if (t == 0) {                                     // per warp: the selected cells and the tied cells before it
        int eb = 0, sb = 0;
        for (int w2 = 0; w2 < TK_T / 32; ++w2) {
            const int g2 = s_gt[w2], e2 = s_eq[w2];
            const int take = eq_budget - eb < 0 ? 0 : (eq_budget - eb > e2 ? e2 : eq_budget - eb);
            s_gt[w2] = sb;
            s_eq[w2] = eb;
            sb += g2 + take;
            eb += e2;
        }
    }
    __syncthreads();
    int run_sel = s_gt[warp], run_eq = s_eq[warp];
    for (int64_t r0 = w0; r0 < nb && r0 < w0 + 32 * per; r0 += 32) {
        const int64_t b = r0 + lane;
        const int w = b < nb ? weight(b) : 0;
        const uint32_t k = w ? order_key(sc[b]) : 0u;
        const int my_gt = (w && k > thr) ? w : 0, my_eq = (w && k == thr) ? w : 0;
        if (__ballot_sync(0xffffffffu, (my_gt | my_eq) != 0) == 0u) continue;   // a row without a selected block
        int pe = my_eq;                               // tied cells up to and with this lane
#pragma unroll
        for (int o = 1; o < 32; o <<= 1) {
            const int y = __shfl_up_sync(0xffffffffu, pe, o);
            if (lane >= o) pe += y;
        }
        const int left = eq_budget - (run_eq + pe - my_eq);
        const int take = left < 0 ? 0 : (left > my_eq ? my_eq : left);
        const int my_sel = my_gt + take;
        int ps = my_sel;
#pragma unroll
        for (int o = 1; o < 32; o <<= 1) {
            const int y = __shfl_up_sync(0xffffffffu, ps, o);
            if (lane >= o) ps += y;
        }
        int32_t* dst = out + run_sel + ps - my_sel;
        for (int c = 0; c < my_sel; ++c) dst[c] = (int32_t) (b * R + c);
        run_eq += __shfl_sync(0xffffffffu, pe, 31);
        run_sel += __shfl_sync(0xffffffffu, ps, 31);
    }
}
#endif


// Block scores with every key block read ONCE for all of a call's queries (block_scores_kernel's grid is
// (max_blocks / 8) x nq: ~24,600 mostly-idle blocks per layer at a decode window, each key re-read per query).  A fixed
// grid strides over the blocks; per (block, query) the same arithmetic in the same order as block_scores_kernel.
// qsa_block_scores takes it for every call without an active-block count and at most MQ queries: the captured decode
// window, the uncaptured decode, and prefill's pooled16 call.
constexpr int MQ = 8;
__global__ void __launch_bounds__(SCORE_WARPS * 32) block_scores_multi_kernel(const float* __restrict__ pooled,
                                                                              const float* __restrict__ dead,
                                                                              const float* __restrict__ q_idx,
                                                                              const int32_t* __restrict__ steps, int nq,
                                                                              int64_t max_blocks, float* __restrict__ out,
                                                                              bool early_exit) {
    __shared__ __align__(16) float qs[MQ * IDX_HEADS * IDX_DIM];
    __shared__ int64_t s_nkv[MQ], s_nbid[MQ];
    if (!early_exit)
        for (int i = threadIdx.x; i < nq * IDX_HEADS * IDX_DIM; i += blockDim.x) qs[i] = q_idx[i];
    if (threadIdx.x < nq) {
        s_nkv[threadIdx.x] = steps[threadIdx.x * kStepCount + kStepNKv];
        s_nbid[threadIdx.x] = steps[threadIdx.x * kStepCount + kStepNBid];
    }
    __syncthreads();
    int64_t top = 0;
    for (int q = 0; q < nq; ++q) top = s_nbid[q] > top ? s_nbid[q] : top;
    if (early_exit) {
        // (#783, stuchapin909) a CTA whose first block is past the batch's largest n_bid has no work: the grid is a
        // fixed 256 CTAs and a decode window reaches only a few of them, so those return before staging the queries.
        // Every thread of the CTA takes the same branch (top and blockIdx are CTA-uniform), so the barrier is safe.
        if ((int64_t) blockIdx.x * SCORE_WARPS > top) return;
        for (int i = threadIdx.x; i < nq * IDX_HEADS * IDX_DIM; i += blockDim.x) qs[i] = q_idx[i];
        __syncthreads();
    }
    const int lane = threadIdx.x & 31;
    const int64_t wstride = (int64_t) gridDim.x * SCORE_WARPS;
    for (int64_t b = (int64_t) blockIdx.x * SCORE_WARPS + (threadIdx.x >> 5); b <= top && b < max_blocks; b += wstride) {
        const float4 kp = *reinterpret_cast<const float4*>(pooled + b * IDX_DIM + lane * 4);
        const float4 kd = *reinterpret_cast<const float4*>(dead + lane * 4);
        for (int qi = 0; qi < nq; ++qi) {
            const int64_t n_bid = s_nbid[qi];
            if (b > n_bid) continue;
            const float4 k4 = (b == n_bid) ? kd : kp;
            const float* q = qs + qi * IDX_HEADS * IDX_DIM + lane * 4;
            float score = 0.0f;
#pragma unroll
            for (int h = 0; h < IDX_HEADS; ++h) {
                const float4 q4 = *reinterpret_cast<const float4*>(q + h * IDX_DIM);
                float d = k4.x * q4.x + k4.y * q4.y + k4.z * q4.z + k4.w * q4.w;
#pragma unroll
                for (int o = 16; o > 0; o >>= 1) d += __shfl_xor_sync(0xffffffffu, d, o);
                score += d > 0.0f ? d : 0.0f;
            }
            if (lane == 0) {
                if (b == n_bid && s_nkv[qi] % R != 0) score += 1e9f;
                out[qi * max_blocks + b] = score;
            }
        }
    }
}

#if !defined(__HIPCC__)
// ---- the decode top-k on a thread-block cluster (sm_90+; S19).  One CTA per query (block_topk_reg_kernel, or
// block_topk_kernel above 33,792 blocks: a --max-context over ~135K) makes four radix passes and two scans over up to
// 65,538 blocks on ONE SM while the rest of the GPU idles - a decode window has 1-5 queries.  Here a cluster of CL_N
// CTAs shares a query: CTA r holds the keys of blocks [r * per, (r + 1) * per) in shared memory, builds the digit
// histogram of its keys, and PUSHES it into slot r of every CTA's `hin` (distributed shared memory); after one cluster
// barrier each CTA sums the CL_N slots itself, so all of them take the same digit.  The emit pass needs each CTA's
// cells above / at the threshold: pushed the same way, then each CTA offsets its own by the ranks before it.
// RTX 5070, per call at capacity = context (decode_cluster_parity --bench): 21.9 -> 15.6 us at 32K, 58 -> 18 at 128K,
// 200 -> 22 at 262K (the one-CTA kernels' digit search was serial, too: one thread over 256 bins per pass).
//
// IDENTICAL IDS, by construction rather than by luck: the threshold thr (the width-th largest key, cells counted with
// their weights) and `above` (cells with a larger key) are pure functions of the multiset of (key, weight) - integer
// histograms, summed in any order - and the digit rule is block_topk_kernel's, written as a scan (the largest digit d
// whose cells at or above it reach `need`; 0 when none). The cells emitted are then fixed: every cell of a block with
// key > thr, and the first eq_budget = width - above cells at thr in ascending order (a block at thr may be cut),
// written in ascending cell order. Position of a thread's first cell = (cells above thr before it) + min(cells at thr
// before it, eq_budget): the telescoped sum of the reference's per-thread clamp. NaN keys are 0 (order_key) as there.
//
// Barriers: barrier.cluster arrive / wait (each CTA's threads all take part; it also orders the CTA's own shared
// memory, so it doubles as __syncthreads). Phase 0 is a relaxed arrive at entry, waited before the first remote store
// (a CTA's shared memory exists once it runs). Pushes go out before the arrive that releases them, and nothing reads
// another CTA's memory after the last wait - so a CTA may exit without a closing barrier (a wait counts the threads
// that have not exited). `hin` alternates by pass parity: CTA x reads slot buffer p&1 in pass p before arriving at
// pass p+1's barrier, and nobody writes buffer p&1 again (pass p+2) before waiting on that barrier.
constexpr int CL_N = 8;        // CTAs per query (the portable cluster size)
constexpr int CL_T = 1024;     // threads per CTA
constexpr int CL_MAXQ = 16;    // larger calls (prefill sub-batches) fill the GPU with one CTA per query already
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 900
#define STRATA_SEL_CLUSTER 1
#else
#define STRATA_SEL_CLUSTER 0   // an older target's code is a trap; qsa_block_topk_cluster never launches it there
#endif

#if STRATA_SEL_CLUSTER
__device__ __forceinline__ void cl_arrive_relaxed() {
    asm volatile("barrier.cluster.arrive.relaxed.aligned;\n" ::: "memory");
}
__device__ __forceinline__ void cl_arrive() { asm volatile("barrier.cluster.arrive.release.aligned;\n" ::: "memory"); }
__device__ __forceinline__ void cl_wait() { asm volatile("barrier.cluster.wait.acquire.aligned;\n" ::: "memory"); }
__device__ __forceinline__ unsigned cl_rank() {
    unsigned r;
    asm volatile("mov.u32 %0, %%cluster_ctarank;\n" : "=r"(r));
    return r;
}
template <typename T> __device__ __forceinline__ T* cl_map(T* p, unsigned rank) {   // p in CTA `rank`'s shared memory
    uint64_t o;
    asm volatile("mapa.u64 %0, %1, %2;\n" : "=l"(o) : "l"((uint64_t) p), "r"(rank));
    return reinterpret_cast<T*>(o);
}
// exclusive block scan of a 64-bit value over CL_T threads; `total` = the CTA's sum
__device__ __forceinline__ unsigned long long cl_excl_scan64(unsigned long long v, unsigned long long* s_warp,
                                                             unsigned long long& total) {
    const int lane = threadIdx.x & 31, warp = threadIdx.x >> 5;
    unsigned long long x = v;
#pragma unroll
    for (int o = 1; o < 32; o <<= 1) {
        const unsigned long long y = __shfl_up_sync(0xffffffffu, x, o);
        if (lane >= o) x += y;
    }
    if (lane == 31) s_warp[warp] = x;
    __syncthreads();
    if (warp == 0) {
        const unsigned long long w = s_warp[lane];
        unsigned long long z = w;
#pragma unroll
        for (int o = 1; o < 32; o <<= 1) {
            const unsigned long long y = __shfl_up_sync(0xffffffffu, z, o);
            if (lane >= o) z += y;
        }
        s_warp[lane] = z - w;
        if (lane == 31) s_warp[32] = z;
    }
    __syncthreads();
    total = s_warp[32];
    return s_warp[warp] + x - v;
}
#endif

// grid (CL_N, nq), cluster (CL_N, 1, 1), CL_T threads, dynamic shared memory: ceil(max_blocks / CL_N) keys
__global__ void __launch_bounds__(CL_T) block_topk_cluster_kernel(const float* __restrict__ scores,
                                                                  const int32_t* __restrict__ steps, int64_t max_blocks,
                                                                  int64_t cap, int32_t* __restrict__ ids) {
#if STRATA_SEL_CLUSTER
    extern __shared__ uint32_t keys[];                  // this CTA's blocks' keys
    __shared__ __align__(16) int hin[2][CL_N][256];     // the cluster's histograms, slot = the pushing rank
    __shared__ __align__(16) int hloc[2][256];          // this CTA's histogram
    __shared__ unsigned long long cnt[CL_N];            // per rank: its cells above thr (low 32 bits) and at it (high)
    __shared__ unsigned long long s_warp[33];
    __shared__ uint32_t s_prefix;
    __shared__ int s_above;
    const unsigned rank = cl_rank();
    const int t = threadIdx.x, lane = t & 31, warp = t >> 5;
    cl_arrive_relaxed();                                // phase 0: this CTA runs
    const int64_t qi = blockIdx.y;
    const int32_t* st = steps + qi * kStepCount;
    const int64_t n_kv = st[kStepNKv], n_bid = st[kStepNBid], width = st[kStepWidth];
    int32_t* out = ids + qi * cap;
    if (n_kv <= width) {                                // the identity, split over the cluster; no remote access
        for (int64_t j = (int64_t) rank * CL_T + t; j < n_kv; j += (int64_t) CL_N * CL_T) out[j] = (int32_t) j;
        return;
    }
    // the tail block n_bid weighs its cells (0..3); with none it is not a candidate (the reference skips w == 0)
    const int wtail = (int) (n_kv - n_bid * R);
    const int64_t nbe = n_bid + (wtail > 0 ? 1 : 0);
    const int64_t per = (nbe + CL_N - 1) / CL_N;       // <= ceil(max_blocks / CL_N), the keys the host sized
    const int64_t lo = (int64_t) rank * per < nbe ? (int64_t) rank * per : nbe;
    const int64_t hi = lo + per < nbe ? lo + per : nbe;
    const int n = (int) (hi - lo);
    const int tail_i = (wtail > 0 && n_bid >= lo && n_bid < hi) ? (int) (n_bid - lo) : -1;
    const float* sc = scores + qi * max_blocks + lo;
    for (int i = t; i < n; i += CL_T) keys[i] = order_key(sc[i]);
    for (int i = t; i < 2 * 256; i += CL_T) (&hloc[0][0])[i] = 0;
    __syncthreads();
    uint32_t prefix = 0;
    int above = 0;                                      // cells strictly above the digits fixed so far
    for (int pass = 0; pass < 4; ++pass) {
        const int shift = 24 - 8 * pass;
        const uint32_t hmask = pass == 0 ? 0u : (0xffffffffu << (shift + 8));
        int* hl = hloc[pass & 1];
        // warp-aggregated: the scores share their top bits, so plain atomics would queue on a few bins
        for (int i0 = 0; i0 < n; i0 += CL_T) {
            const int i = i0 + t;
            int bin = -1;
            if (i < n) {
                const uint32_t k = keys[i];
                if ((k & hmask) == prefix) bin = (int) ((k >> shift) & 255);
            }
            const unsigned same = __match_any_sync(0xffffffffu, bin);
            if (bin >= 0 && lane == __ffs(same) - 1) atomicAdd(&hl[bin], __popc(same) * R);
            if (bin >= 0 && i == tail_i) atomicAdd(&hl[bin], wtail - R);
        }
        __syncthreads();
        if (pass == 0) cl_wait();                       // phase 0 done: every CTA's shared memory is there
        for (int i = t; i < CL_N * 64; i += CL_T) {     // the histogram into slot `rank` of every CTA, as int4
            const int dst = i >> 6, c = i & 63;
            *cl_map(reinterpret_cast<int4*>(&hin[pass & 1][rank][0]) + c, (unsigned) dst) =
                reinterpret_cast<const int4*>(hl)[c];
        }
        for (int i = t; i < 256; i += CL_T) hloc[(pass + 1) & 1][i] = 0;
        cl_arrive();
        cl_wait();
        if (warp == 0) {
            // lane L owns digits 255 - 8L down to 248 - 8L; the digit holding the need-th cell from the top
            const int need = (int) width - above;
            int tot[8], part = 0;
#pragma unroll
            for (int k = 0; k < 8; ++k) {
                const int d = 255 - 8 * lane - k;
                int v = 0;
#pragma unroll
                for (int r = 0; r < CL_N; ++r) v += hin[pass & 1][r][d];
                tot[k] = v;
                part += v;
            }
            int incl = part;
#pragma unroll
            for (int o = 1; o < 32; o <<= 1) {
                const int y = __shfl_up_sync(0xffffffffu, incl, o);
                if (lane >= o) incl += y;
            }
            const int excl = incl - part;
            const unsigned hit = __ballot_sync(0xffffffffu, excl < need && incl >= need);
            // no hit cannot happen (the cells total n_kv > width); the reference would then take digit 0
            const int who = hit ? __ffs(hit) - 1 : 31;
            if (lane == who) {
                int acc = excl, k = 0;
                for (; k < 7; ++k) {
                    if (acc + tot[k] >= need) break;
                    acc += tot[k];
                }
                s_prefix = prefix | ((uint32_t) (255 - 8 * lane - k) << shift);
                s_above = above + acc;
            }
        }
        __syncthreads();
        prefix = s_prefix;
        above = s_above;
    }
    const uint32_t thr = prefix;
    const int64_t eq_budget = width - above;           // cells at thr that fit, lowest index first
    // each thread a contiguous run of the CTA's keys; an odd run length keeps the shared reads conflict-free
    int seg = (n + CL_T - 1) / CL_T;
    if ((seg & 1) == 0 && seg > 0) ++seg;
    const int s0 = t * seg < n ? t * seg : n, s1 = s0 + seg < n ? s0 + seg : n;
    uint32_t gt = 0, eq = 0;
    for (int i = s0; i < s1; ++i) {
        const uint32_t k = keys[i], w = i == tail_i ? (uint32_t) wtail : (uint32_t) R;
        if (k > thr) gt += w;
        else if (k == thr) eq += w;
    }
    unsigned long long cta;
    const unsigned long long before =
        cl_excl_scan64((unsigned long long) gt | ((unsigned long long) eq << 32), s_warp, cta);
    if (t < CL_N) *cl_map(&cnt[rank], (unsigned) t) = cta;
    cl_arrive();
    cl_wait();
    int64_t gt_off = 0, eq_off = 0;
    for (unsigned r = 0; r < rank; ++r) {
        gt_off += (int64_t) (cnt[r] & 0xffffffffu);
        eq_off += (int64_t) (cnt[r] >> 32);
    }
    const int64_t eq_before = eq_off + (int64_t) (before >> 32);
    int64_t wpos = gt_off + (int64_t) (before & 0xffffffffu) + (eq_before < eq_budget ? eq_before : eq_budget);
    int64_t eq_left = eq_budget - eq_before;
    if (eq_left < 0) eq_left = 0;
    for (int i = s0; i < s1; ++i) {
        const uint32_t k = keys[i];
        const int w = i == tail_i ? wtail : R;
        const int64_t b = lo + i;
        if (k > thr) {
            for (int c = 0; c < w; ++c) out[wpos++] = (int32_t) (b * R + c);
        } else if (k == thr) {
            for (int c = 0; c < w && eq_left > 0; ++c, --eq_left) out[wpos++] = (int32_t) (b * R + c);
        }
    }
#else
    (void) scores; (void) steps; (void) max_blocks; (void) cap; (void) ids;
    __trap();
#endif
}
#endif  // !__HIPCC__
}  // namespace

// block_scores_multi_kernel with the warp sums as one recursive-halving reduction.  A (query, head) dot is a lane's
// 4-term product summed over the 32 lanes by an xor butterfly (offsets 16, 8, 4, 2, 1): 5 shuffles per value, 20 per
// (block, query) - on Turing the decode scorer's time grew with the window's queries (420K cells: 117 / 265 / 509 us
// for 1 / 3 / 6 queries), the shuffles, not the keys.  Here the V = 4 x queries values (padded to a power of 2) are
// reduced together: at offset o each lane keeps half of its values and adds its partner's copy of them - the pair
// (lane l, lane l ^ o) the butterfly adds for that value at that offset, so every value's additions are the
// butterfly's, operand for operand (float addition commutes): the same bits, with V - 1 shuffles plus the plain
// butterfly's remaining stages instead of 5 V.  Lane l ends with value (l >> (5 - log2 V)); the query's 4 heads are
// then gathered by its head-0 lane and summed in block_scores_multi_kernel's order.  STRATA_SCORES_HALVE=0: the old kernel.
template <int V>
__global__ void __launch_bounds__(SCORE_WARPS * 32) block_scores_halve_kernel(const float* __restrict__ pooled,
                                                                              const float* __restrict__ dead,
                                                                              const float* __restrict__ q_idx,
                                                                              const int32_t* __restrict__ steps, int nq,
                                                                              int64_t max_blocks, float* __restrict__ out) {
    constexpr int NQ = V / IDX_HEADS;              // query slots (nq <= NQ)
    constexpr int LV = V == 4 ? 2 : V == 8 ? 3 : V == 16 ? 4 : 5, SH = 5 - LV;
    static_assert(V == 4 || V == 8 || V == 16 || V == 32, "V");
    __shared__ __align__(16) float qs[NQ * IDX_HEADS * IDX_DIM];
    __shared__ int64_t s_nkv[NQ], s_nbid[NQ];
    if (threadIdx.x < NQ) {
        s_nkv[threadIdx.x] = threadIdx.x < nq ? steps[threadIdx.x * kStepCount + kStepNKv] : 0;
        s_nbid[threadIdx.x] = threadIdx.x < nq ? steps[threadIdx.x * kStepCount + kStepNBid] : -1;
    }
    __syncthreads();
    int64_t top = 0;
    for (int q = 0; q < nq; ++q) top = s_nbid[q] > top ? s_nbid[q] : top;
    if ((int64_t) blockIdx.x * SCORE_WARPS > top) return;   // CTA-uniform (as block_scores_multi_kernel's early exit)
    for (int i = threadIdx.x; i < NQ * IDX_HEADS * IDX_DIM; i += blockDim.x)
        qs[i] = i < nq * IDX_HEADS * IDX_DIM ? q_idx[i] : 0.0f;
    __syncthreads();
    const int lane = threadIdx.x & 31;
    const int64_t wstride = (int64_t) gridDim.x * SCORE_WARPS;
    const float4 kd = *reinterpret_cast<const float4*>(dead + lane * 4);
    // up to 4 queries: the lane's query values in registers (read once, not once per block from shared memory: the
    // shared-memory reads were the kernel's limit, ncu mio_throttle)
    constexpr bool QREG = V <= 16;
    float4 qr[QREG ? V : 1];
    if constexpr (QREG) {
#pragma unroll
        for (int i = 0; i < V; ++i) qr[i] = *reinterpret_cast<const float4*>(qs + (i >> 2) * IDX_HEADS * IDX_DIM + (i & 3) * IDX_DIM + lane * 4);
    }
    int s_nbid_r[NQ];                               // n_bid < 2^31
#pragma unroll
    for (int qi = 0; qi < NQ; ++qi) s_nbid_r[qi] = (int) s_nbid[qi];
    for (int64_t b = (int64_t) blockIdx.x * SCORE_WARPS + (threadIdx.x >> 5); b <= top && b < max_blocks; b += wstride) {
        const float4 kp = *reinterpret_cast<const float4*>(pooled + b * IDX_DIM + lane * 4);
        float v[V];
#pragma unroll
        for (int qi = 0; qi < NQ; ++qi) {
            const float4 k4 = (b == s_nbid_r[qi]) ? kd : kp;
            const float* q = qs + qi * IDX_HEADS * IDX_DIM + lane * 4;
#pragma unroll
            for (int h = 0; h < IDX_HEADS; ++h) {
                float4 q4;
                if constexpr (QREG) q4 = qr[qi * IDX_HEADS + h];
                else q4 = *reinterpret_cast<const float4*>(q + h * IDX_DIM);
                v[qi * IDX_HEADS + h] = k4.x * q4.x + k4.y * q4.y + k4.z * q4.z + k4.w * q4.w;
            }
        }
        // recursive halving over offsets 16, 8, ... while more than one value is left, then the plain butterfly
        int cnt = V;
#pragma unroll
        for (int o = 16; o > 0; o >>= 1) {
            if (cnt > 1) {
                const int half = cnt >> 1;
                const bool hi = (lane & o) != 0;
#pragma unroll
                for (int i = 0; i < V / 2; ++i) {
                    if (i < half) {
                        const float send = hi ? v[i] : v[i + half];
                        const float keep = hi ? v[i + half] : v[i];
                        const float recv = __shfl_xor_sync(0xffffffffu, send, o);
                        v[i] = keep + recv;
                    }
                }
                cnt = half;
            } else {
                v[0] += __shfl_xor_sync(0xffffffffu, v[0], o);
            }
        }
        // lane l holds value j = l >> SH = qi * 4 + h; the head-0 lane of each query gathers heads 1-3
        const int j = lane >> SH, qi = j >> 2;
        const float d1 = __shfl_sync(0xffffffffu, v[0], (lane + (1 << SH)) & 31);
        const float d2 = __shfl_sync(0xffffffffu, v[0], (lane + (2 << SH)) & 31);
        const float d3 = __shfl_sync(0xffffffffu, v[0], (lane + (3 << SH)) & 31);
        if ((j & 3) == 0 && (lane & ((1 << SH) - 1)) == 0 && qi < nq && b <= s_nbid_r[qi & (NQ - 1)]) {
            const float d[4] = {v[0], d1, d2, d3};
            float score = 0.0f;
#pragma unroll
            for (int h = 0; h < IDX_HEADS; ++h) score += d[h] > 0.0f ? d[h] : 0.0f;
            if (b == s_nbid_r[qi & (NQ - 1)] && s_nkv[qi] % R != 0) score += 1e9f;
            out[qi * max_blocks + b] = score;
        }
    }
}

bool scores_halve_on() {
    static const int env = [] { const char* v = std::getenv("STRATA_SCORES_HALVE"); return v ? (std::atoi(v) != 0 ? 1 : 0) : -1; }();
    if (env >= 0) return env != 0;
    static int ok[64] = {};   // 0 unknown, 1 yes, 2 no
    int dev = 0;
    if (cudaGetDevice(&dev) != cudaSuccess || dev < 0 || dev >= 64) { cudaGetLastError(); return false; }
    if (ok[dev] == 0) {
        int major = 0, minor = 0;
        cudaDeviceGetAttribute(&major, cudaDevAttrComputeCapabilityMajor, dev);
        cudaDeviceGetAttribute(&minor, cudaDevAttrComputeCapabilityMinor, dev);
        ok[dev] = major == 7 && minor == 5 ? 1 : 2;
    }
    return ok[dev] == 1;
}

void qsa_block_scores(const float* pooled, const float* dead, const float* q_idx, const int32_t* steps, int64_t nq,
                      int64_t max_blocks, const QsaShapes& s, float* scores, void* stream, int64_t active_blocks) {
    if (nq <= 0) return;
    if (s.idx_dim != IDX_DIM || s.idx_n_head != IDX_HEADS || s.idx_block != R || nq > 65535) {
        std::fprintf(stderr, "qsa_block_scores: unsupported indexer geometry\n");
        std::exit(1);
    }
    // a block past a query's n_bid returns at once: the grid need only reach the batch's largest n_bid (C-1)
    static const bool multi = [] { const char* v = std::getenv("STRATA_SCORES_MULTI"); return v == nullptr || std::atoi(v) != 0; }();
    if (multi && nq <= MQ && active_blocks <= 0) {   // no active count: decode (captured or not) and prefill's pooled16
        if (scores_halve_on()) {   // Turing: block_scores_halve_kernel, the same bits
            const cudaStream_t st = (cudaStream_t) stream;
            if (nq <= 1) block_scores_halve_kernel<4><<<256, SCORE_WARPS * 32, 0, st>>>(pooled, dead, q_idx, steps, (int) nq, max_blocks, scores);
            else if (nq <= 2) block_scores_halve_kernel<8><<<256, SCORE_WARPS * 32, 0, st>>>(pooled, dead, q_idx, steps, (int) nq, max_blocks, scores);
            else if (nq <= 4) block_scores_halve_kernel<16><<<256, SCORE_WARPS * 32, 0, st>>>(pooled, dead, q_idx, steps, (int) nq, max_blocks, scores);
            else {   // 5-8 queries: two calls of up to 4 (the register form; each query's scores are its own)
                block_scores_halve_kernel<16><<<256, SCORE_WARPS * 32, 0, st>>>(pooled, dead, q_idx, steps, 4, max_blocks, scores);
                const int rest = (int) nq - 4;
                const float* q2 = q_idx + 4 * IDX_HEADS * IDX_DIM;
                const int32_t* st2 = steps + 4 * kStepCount;
                float* sc2 = scores + 4 * max_blocks;
                if (rest <= 1) block_scores_halve_kernel<4><<<256, SCORE_WARPS * 32, 0, st>>>(pooled, dead, q2, st2, rest, max_blocks, sc2);
                else if (rest <= 2) block_scores_halve_kernel<8><<<256, SCORE_WARPS * 32, 0, st>>>(pooled, dead, q2, st2, rest, max_blocks, sc2);
                else block_scores_halve_kernel<16><<<256, SCORE_WARPS * 32, 0, st>>>(pooled, dead, q2, st2, rest, max_blocks, sc2);
            }
            const cudaError_t e = cudaGetLastError();
            if (e != cudaSuccess) { std::fprintf(stderr, "qsa_block_scores halve: %s\n", cudaGetErrorString(e)); std::exit(1); }
            return;
        }
        // STRATA_QSA_EARLY_EXIT=0 keeps the old staging (every CTA loads the queries first); the scores are the same bits
        static const bool early = [] { const char* v = std::getenv("STRATA_QSA_EARLY_EXIT"); return v == nullptr || std::atoi(v) != 0; }();
        block_scores_multi_kernel<<<256, SCORE_WARPS * 32, 0, (cudaStream_t) stream>>>(pooled, dead, q_idx, steps, (int) nq,
                                                                                     max_blocks, scores, early);
        const cudaError_t e = cudaGetLastError();
        if (e != cudaSuccess) { std::fprintf(stderr, "qsa_block_scores multi: %s\n", cudaGetErrorString(e)); std::exit(1); }
        return;
    }
    const int64_t reach = active_blocks > 0 && active_blocks < max_blocks ? active_blocks : max_blocks;
    const dim3 grid((unsigned) ((reach + SCORE_WARPS - 1) / SCORE_WARPS), (unsigned) nq);
    block_scores_kernel<<<grid, SCORE_WARPS * 32, 0, (cudaStream_t) stream>>>(pooled, dead, q_idx, steps, max_blocks,
                                                                              scores);
    const cudaError_t e = cudaGetLastError();
    if (e != cudaSuccess) { std::fprintf(stderr, "qsa_block_scores: %s\n", cudaGetErrorString(e)); std::exit(1); }
}

bool qsa_block_scores_tc(const float* pooled, const float* dead, const float* q_idx, const int32_t* steps, int64_t nq,
                         int64_t max_blocks, const QsaShapes& s, float* scores, void* stream, int64_t active_blocks) {
    if (nq <= 0) return true;
    if (s.idx_dim != IDX_DIM || s.idx_n_head != IDX_HEADS || s.idx_block != R || nq > 65535 * TC_QT) return false;
#if defined(__HIPCC__)
    // AMD: the gfx12 (RDNA4) WMMA scorer, opt-in (STRATA_SELECT_WMMA=1): it selects slightly differently from the warp
    // kernel (254/256 queries the same), so the default keeps the warp kernel; every other target keeps it too (false)
    static const bool wmma_on = [] {
        const char* v = std::getenv("STRATA_SELECT_WMMA");
        return v != nullptr && v[0] != '\0' && v[0] != '0';
    }();
    if (!wmma_on || !sel_gfx12_device()) return false;
    {
        const int64_t reach = active_blocks > 0 && active_blocks < max_blocks ? active_blocks : max_blocks;
        const int64_t per = (int64_t) 4 * WITER * 16;
        const dim3 grid((unsigned) ((reach + per - 1) / per), (unsigned) ((nq + WQT - 1) / WQT));
        block_scores_wmma_kernel<<<grid, 128, 0, (cudaStream_t) stream>>>(pooled, q_idx, steps, nq, max_blocks, reach, scores);
        block_scores_tail_kernel<<<(unsigned) nq, 32, 0, (cudaStream_t) stream>>>(dead, q_idx, steps, max_blocks, scores);
        const cudaError_t e = cudaGetLastError();
        if (e != cudaSuccess) { std::fprintf(stderr, "qsa_block_scores_tc: %s\n", cudaGetErrorString(e)); std::exit(1); }
        return true;
    }
#else
    {   // sm_80 or newer (TF32 MMA); an older card keeps the warp kernel
        static int cc_major[64] = {};
        int dev = 0;
        if (cudaGetDevice(&dev) != cudaSuccess || dev < 0 || dev >= 64) { cudaGetLastError(); return false; }
        if (cc_major[dev] == 0) {
            int major = 0;
            if (cudaDeviceGetAttribute(&major, cudaDevAttrComputeCapabilityMajor, dev) != cudaSuccess) {
                cudaGetLastError();
                return false;
            }
            // STRATA_QSA_WARP=1|select (an A/B arm): the pre-sm_80 kernels on any card, as RTX 20 runs them
            const char* w = std::getenv("STRATA_QSA_WARP");
            cc_major[dev] = w && (!std::strcmp(w, "1") || !std::strcmp(w, "select")) ? 7 : strata::cc_major_of(major);
        }
        if (cc_major[dev] < 8) {
            // no TF32 tensor cores: the FP32 tiled kernel, opt-in (STRATA_SELECT_SIMT=1): its sum order differs from the
            // warp kernel's, so the default stays the caller's warp kernel and the output bytes stay the same
            static const bool simt = [] {
                const char* v = std::getenv("STRATA_SELECT_SIMT");
                return v != nullptr && v[0] == '1';
            }();
            if (!simt) return false;
            const int64_t reach = active_blocks > 0 && active_blocks < max_blocks ? active_blocks : max_blocks;
            const dim3 grid((unsigned) ((reach + SM_NB - 1) / SM_NB), (unsigned) ((nq + SM_QT - 1) / SM_QT));
            if (grid.y > 65535) return false;
            block_scores_simt_kernel<<<grid, 256, 0, (cudaStream_t) stream>>>(pooled, q_idx, steps, nq, max_blocks,
                                                                              reach, scores);
            block_scores_tail_kernel<<<(unsigned) nq, 32, 0, (cudaStream_t) stream>>>(dead, q_idx, steps, max_blocks,
                                                                                    scores);
            const cudaError_t e = cudaGetLastError();
            if (e != cudaSuccess) { std::fprintf(stderr, "qsa_block_scores (simt): %s\n", cudaGetErrorString(e)); std::exit(1); }
            return true;
        }
    }
    static bool attr[64] = {};   // the shared-memory opt-in is per device (a layer split runs it on several)
    int adev = 0;
    cudaGetDevice(&adev);
    const int bytes = (TC_QT * TC_QS + TC_NB * TC_KS) * (int) sizeof(float);
    if (!attr[adev]) {
        if (cudaFuncSetAttribute(block_scores_tc_kernel, cudaFuncAttributeMaxDynamicSharedMemorySize, bytes) !=
            cudaSuccess) {
            cudaGetLastError();
            return false;
        }
        attr[adev] = true;
    }
    const int64_t reach = active_blocks > 0 && active_blocks < max_blocks ? active_blocks : max_blocks;
    const int64_t per = (int64_t) TC_NB * TC_ITER;
    const dim3 grid((unsigned) ((reach + per - 1) / per), (unsigned) ((nq + TC_QT - 1) / TC_QT));
    block_scores_tc_kernel<<<grid, 128, bytes, (cudaStream_t) stream>>>(pooled, q_idx, steps, nq, max_blocks, reach,
                                                                         scores);
    block_scores_tail_kernel<<<(unsigned) nq, 32, 0, (cudaStream_t) stream>>>(dead, q_idx, steps, max_blocks, scores);
    const cudaError_t e = cudaGetLastError();
    if (e != cudaSuccess) { std::fprintf(stderr, "qsa_block_scores_tc: %s\n", cudaGetErrorString(e)); std::exit(1); }
    return true;
#endif
}

void qsa_block_topk_ref(const float* scores, const int32_t* steps, int64_t nq, int64_t max_blocks, int64_t cap,
                        const QsaShapes& s, int32_t* ids, void* stream) {
    if (nq <= 0) return;
    if (s.idx_block != R || cap < qsa_selection_width(kTopkMaxCells, s)) {
        std::fprintf(stderr, "qsa_block_topk: unsupported geometry or cap\n");
        std::exit(1);
    }
    block_topk_kernel<<<(unsigned) nq, TOPK_T, 0, (cudaStream_t) stream>>>(scores, steps, max_blocks, cap, ids);
    const cudaError_t e = cudaGetLastError();
    if (e != cudaSuccess) { std::fprintf(stderr, "qsa_block_topk: %s\n", cudaGetErrorString(e)); std::exit(1); }
}

#if !defined(__HIPCC__)
// Only Turing has a retained model measurement for this CUDA dispatch. Other CUDA devices keep the capacity rule
// (the RTX 5070 regression below). Cache the properties per calling thread; layer-split device switches are checked.
static bool topk_active_turing_device() {
    int dev = 0;
    if (cudaGetDevice(&dev) != cudaSuccess) return false;
    static thread_local int cached_device = -1;
    static thread_local bool turing = false;
    if (dev != cached_device) {
        cudaDeviceProp prop{};
        if (cudaGetDeviceProperties(&prop, dev) != cudaSuccess) return false;
        turing = prop.major == 7 && prop.minor == 5;
        cached_device = dev;
    }
    return turing;
}
#endif

bool qsa_block_topk_cluster(const float* scores, const int32_t* steps, int64_t nq, int64_t max_blocks, int64_t cap,
                            const QsaShapes& s, int32_t* ids, void* stream) {
#if defined(__HIPCC__)
    (void) scores; (void) steps; (void) nq; (void) max_blocks; (void) cap; (void) s; (void) ids; (void) stream;
    return false;
#else
    if (nq <= 0) return true;
    if (s.idx_block != R || cap < qsa_selection_width(kTopkMaxCells, s) || nq > 65535 || max_blocks <= 0) return false;
    const size_t smem = (size_t) ((max_blocks + CL_N - 1) / CL_N) * sizeof(uint32_t);
    // Per device (a layer split runs on several): 1 the cluster kernel runs here, 2 it does not. It needs sm_90+ (the
    // card's, or STRATA_EMULATE_CC's) AND code built for it: a build with only older code JIT-compiles their PTX, whose
    // copy of this kernel is a trap - the function's PTX version says which. `opt`: the dynamic shared memory opted in.
    static int ok[64] = {};
    static size_t opt[64] = {};
    int dev = 0;
    if (cudaGetDevice(&dev) != cudaSuccess || dev < 0 || dev >= 64) { cudaGetLastError(); return false; }
    if (ok[dev] == 0) {
        int major = 0;
        cudaFuncAttributes fa{};
        const bool code = cudaDeviceGetAttribute(&major, cudaDevAttrComputeCapabilityMajor, dev) == cudaSuccess &&
                          strata::cc_major_of(major) >= 9 &&
                          cudaFuncGetAttributes(&fa, block_topk_cluster_kernel) == cudaSuccess && fa.ptxVersion >= 90 &&
                          fa.binaryVersion >= 90;
        cudaGetLastError();
        ok[dev] = code ? 1 : 2;
    }
    if (ok[dev] != 1) return false;
    if (smem > opt[dev]) {   // a larger capacity: opt in, and check that a cluster of CL_N such CTAs can be resident
        int clusters = 0;
        cudaLaunchConfig_t q{};
        cudaLaunchAttribute qa[1];
        qa[0].id = cudaLaunchAttributeClusterDimension;
        qa[0].val.clusterDim.x = CL_N;
        qa[0].val.clusterDim.y = 1;
        qa[0].val.clusterDim.z = 1;
        q.gridDim = dim3(CL_N, 1, 1);
        q.blockDim = dim3(CL_T, 1, 1);
        q.dynamicSmemBytes = smem;
        q.attrs = qa;
        q.numAttrs = 1;
        if (cudaFuncSetAttribute(block_topk_cluster_kernel, cudaFuncAttributeMaxDynamicSharedMemorySize, (int) smem) !=
                cudaSuccess ||
            cudaOccupancyMaxActiveClusters(&clusters, block_topk_cluster_kernel, &q) != cudaSuccess || clusters < 1) {
            cudaGetLastError();
            return false;    // this capacity takes the one-CTA kernels; a smaller one may still fit
        }
        opt[dev] = smem;
    }
    cudaLaunchConfig_t cfg{};
    cudaLaunchAttribute at[1];
    at[0].id = cudaLaunchAttributeClusterDimension;
    at[0].val.clusterDim.x = CL_N;
    at[0].val.clusterDim.y = 1;
    at[0].val.clusterDim.z = 1;
    cfg.gridDim = dim3(CL_N, (unsigned) nq, 1);
    cfg.blockDim = dim3(CL_T, 1, 1);
    cfg.dynamicSmemBytes = smem;
    cfg.stream = (cudaStream_t) stream;
    cfg.attrs = at;
    cfg.numAttrs = 1;
    const cudaError_t e = cudaLaunchKernelEx(&cfg, block_topk_cluster_kernel, scores, steps, max_blocks, cap, ids);
    if (e != cudaSuccess) { std::fprintf(stderr, "qsa_block_topk cluster: %s\n", cudaGetErrorString(e)); std::exit(1); }
    return true;
#endif
}

#if !defined(__HIPCC__)
// ---- the decode top-k split over many CTAs, for cards without thread-block clusters (Turing: a decode window's 1-6
// queries ran block_topk_wide_kernel on 1-6 SMs, ~230 us a call at 420K cells, RTX 2080 Ti).  A query's blocks are
// cut into chunks of SP_CH; six launches: four radix passes (each chunk adds its 8-bit histogram to the query's global
// one), a count of each chunk's cells above / at the threshold, and the emit.  Each pass's CTAs take the previous
// digit from the complete histogram with block_topk_kernel's rule (the largest digit whose cells at or above it reach
// the width; 0 when none) - pure functions of integer counts, so every CTA finds the same digit, and the threshold,
// `above` and the emitted cells are block_topk_kernel's: identical ids.  STRATA_TOPK_SPLIT=0 keeps the wide kernel.
constexpr int SP_T = 256, SP_W = SP_T / 32, SP_CH = 2048, SP_MAXQ = 16, SP_MAXCH = 512;
__device__ int g_sp_hist[SP_MAXQ][4][256];
__device__ int g_sp_state[5][SP_MAXQ][2];          // after p digits: prefix, above
__device__ int g_sp_cnt[SP_MAXQ][SP_MAXCH][2];     // per chunk: cells above the threshold, cells at it

// the digit of pass `p` from its complete histogram: block_topk_kernel's scan as a parallel suffix sum
__device__ void sp_digit(const int* __restrict__ h, int width, int above, int& digit, int& above_out, int* s_buf) {
    const int t = threadIdx.x;
    // s_buf[d] = sum of h[d'] for d' > d (exclusive suffix), d in 0..255
    const int v = h[t];
    int x = v;                                     // inclusive suffix scan over t = 255..0: reverse index r = 255 - t
    const int lane = t & 31, w = t >> 5;
    __shared__ int s_w[SP_W];
    // scan in reverse order: thread t holds element r = 255 - t
    int e = h[255 - t];
    for (int o = 1; o < 32; o <<= 1) { const int y = __shfl_up_sync(0xffffffffu, e, o); if (lane >= o) e += y; }
    if (lane == 31) s_w[w] = e;
    __syncthreads();
    int add = 0;
    for (int i = 0; i < w; ++i) add += s_w[i];
    e += add;                                      // inclusive suffix sum of h from 255 down to r = 255 - t
    (void) x; (void) v;
    s_buf[255 - t] = e;                            // s_buf[r] = sum_{d >= r} h[d]
    __syncthreads();
    // the scan picks the largest d in 1..255 with above + sum_{d' >= d} h >= width, else 0
    __shared__ int s_best;
    if (t == 0) s_best = 0;
    __syncthreads();
    if (t >= 1 && above + s_buf[t] >= width) atomicMax(&s_best, t);
    __syncthreads();
    const int d = s_best;
    digit = d;
    above_out = above + (d == 255 ? 0 : s_buf[d + 1]);   // cells strictly above digit d (the scan's cum)
    if (d == 0) above_out = above + s_buf[1];
    __syncthreads();
}

template <int P>
__global__ void __launch_bounds__(SP_T) topk_split_hist_kernel(const float* __restrict__ scores,
                                                               const int32_t* __restrict__ steps, int64_t max_blocks) {
    __shared__ int hist[SP_W][256];
    __shared__ int s_buf[256];
    const int q = blockIdx.y, t = threadIdx.x, lane = t & 31, warp = t >> 5;
    const int32_t* st = steps + (int64_t) q * kStepCount;
    const int64_t n_kv = st[kStepNKv], n_bid = st[kStepNBid], width = st[kStepWidth];
    if (n_kv <= width) return;
    const int64_t nb = n_bid + 1, b0 = (int64_t) blockIdx.x * SP_CH;
    uint32_t prefix = 0;
    int above = 0;
    if (P > 0) {
        const int pa = g_sp_state[P - 1][q][0], ab = g_sp_state[P - 1][q][1];
        int d, ab2;
        sp_digit(g_sp_hist[q][P - 1], (int) width, ab, d, ab2, s_buf);
        prefix = (uint32_t) pa | ((uint32_t) d << (24 - 8 * (P - 1)));
        above = ab2;
        if (blockIdx.x == 0 && t == 0) { g_sp_state[P][q][0] = (int) prefix; g_sp_state[P][q][1] = above; }
    } else if (blockIdx.x == 0 && t == 0) {
        g_sp_state[0][q][0] = 0; g_sp_state[0][q][1] = 0;
    }
    if (P == 4 || b0 >= nb) return;
    constexpr int shift = 24 - 8 * (P < 4 ? P : 0);
    const uint32_t hi_mask = P == 0 ? 0u : (0xffffffffu << (shift + 8));
    for (int i = lane; i < 256; i += 32) hist[warp][i] = 0;
    __syncwarp();
    const float* sc = scores + (int64_t) q * max_blocks;
    const int64_t b1 = b0 + SP_CH < nb ? b0 + SP_CH : nb;
    for (int64_t b = b0 + t; b < b1; b += SP_T) {
        const int w = b < n_bid ? R : (int) (n_kv - n_bid * R);
        if (w == 0) continue;
        const uint32_t k = order_key(sc[b]);
        if ((k & hi_mask) == (prefix & hi_mask)) atomicAdd(&hist[warp][(k >> shift) & 255], w);
    }
    __syncthreads();
    int s = 0;
    for (int w2 = 0; w2 < SP_W; ++w2) s += hist[w2][t];
    if (s) atomicAdd(&g_sp_hist[q][P][t], s);
}

// the threshold from pass 3's histogram, then each chunk's cells above / at it
__global__ void __launch_bounds__(SP_T) topk_split_count_kernel(const float* __restrict__ scores,
                                                                const int32_t* __restrict__ steps, int64_t max_blocks) {
    __shared__ int s_buf[256];
    __shared__ int s_r[2][SP_W];
    const int q = blockIdx.y, t = threadIdx.x, lane = t & 31, warp = t >> 5;
    const int32_t* st = steps + (int64_t) q * kStepCount;
    const int64_t n_kv = st[kStepNKv], n_bid = st[kStepNBid], width = st[kStepWidth];
    if (n_kv <= width) return;
    const int pa = g_sp_state[3][q][0], ab = g_sp_state[3][q][1];
    int d, ab2;
    sp_digit(g_sp_hist[q][3], (int) width, ab, d, ab2, s_buf);
    const uint32_t thr = (uint32_t) pa | (uint32_t) d;
    if (blockIdx.x == 0 && t == 0) { g_sp_state[4][q][0] = (int) thr; g_sp_state[4][q][1] = ab2; }
    const int64_t nb = n_bid + 1, b0 = (int64_t) blockIdx.x * SP_CH;
    const int64_t b1 = b0 + SP_CH < nb ? b0 + SP_CH : nb;
    const float* sc = scores + (int64_t) q * max_blocks;
    int gt = 0, eq = 0;
    for (int64_t b = b0 + t; b < b1; b += SP_T) {
        const int w = b < n_bid ? R : (int) (n_kv - n_bid * R);
        if (w == 0) continue;
        const uint32_t k = order_key(sc[b]);
        if (k > thr) gt += w;
        else if (k == thr) eq += w;
    }
    for (int o = 16; o > 0; o >>= 1) {
        gt += __shfl_xor_sync(0xffffffffu, gt, o);
        eq += __shfl_xor_sync(0xffffffffu, eq, o);
    }
    if (lane == 0) { s_r[0][warp] = gt; s_r[1][warp] = eq; }
    __syncthreads();
    if (t == 0) {
        int g2 = 0, e2 = 0;
        for (int w2 = 0; w2 < SP_W; ++w2) { g2 += s_r[0][w2]; e2 += s_r[1][w2]; }
        g_sp_cnt[q][blockIdx.x][0] = g2;
        g_sp_cnt[q][blockIdx.x][1] = e2;
    }
}

// the emit: block_topk_wide_kernel's within a chunk, offset by the chunks before it
__global__ void __launch_bounds__(SP_T) topk_split_emit_kernel(const float* __restrict__ scores,
                                                               const int32_t* __restrict__ steps, int64_t max_blocks,
                                                               int64_t cap, int32_t* __restrict__ ids) {
    __shared__ int s_gt[SP_W], s_eq[SP_W];
    __shared__ int s_sel0, s_eqb0;
    const int q = blockIdx.y, t = threadIdx.x, lane = t & 31, warp = t >> 5;
    const int32_t* st = steps + (int64_t) q * kStepCount;
    const int64_t n_kv = st[kStepNKv], n_bid = st[kStepNBid], width = st[kStepWidth];
    int32_t* out = ids + (int64_t) q * cap;
    const int64_t nb = n_bid + 1, b0 = (int64_t) blockIdx.x * SP_CH;
    if (n_kv <= width) {                               // everything is selected: the identity, ascending
        const int64_t c0 = b0 * R, c1 = (b0 + SP_CH) * R < n_kv ? (b0 + SP_CH) * R : n_kv;
        for (int64_t j = c0 + t; j < c1; j += SP_T) out[j] = (int32_t) j;
        return;
    }
    if (b0 >= nb) return;
    const uint32_t thr = (uint32_t) g_sp_state[4][q][0];
    const int eq_budget = (int) (width - g_sp_state[4][q][1]);
    if (t == 0) {                                      // the cells the chunks before this one select / hold at thr
        int sel = 0, eqb = 0;
        for (int c = 0; c < (int) blockIdx.x; ++c) {
            const int g2 = g_sp_cnt[q][c][0], e2 = g_sp_cnt[q][c][1];
            const int take = eq_budget - eqb < 0 ? 0 : (eq_budget - eqb > e2 ? e2 : eq_budget - eqb);
            sel += g2 + take;
            eqb += e2;
        }
        s_sel0 = sel;
        s_eqb0 = eqb;
    }
    const int64_t b1 = b0 + SP_CH < nb ? b0 + SP_CH : nb;
    const int64_t per = (SP_CH / 32) / SP_W;           // rows of 32 blocks per warp
    const int64_t w0 = b0 + (int64_t) warp * 32 * per, w1 = w0 + 32 * per < b1 ? w0 + 32 * per : b1;
    const float* sc = scores + (int64_t) q * max_blocks;
    auto weight = [&](int64_t b) -> int { return b < n_bid ? R : (int) (n_kv - n_bid * R); };
    int gt = 0, eq = 0;
    for (int64_t b = w0 + lane; b < w1; b += 32) {
        const int w = weight(b);
        if (w == 0) continue;
        const uint32_t k = order_key(sc[b]);
        if (k > thr) gt += w;
        else if (k == thr) eq += w;
    }
#pragma unroll
    for (int o = 16; o > 0; o >>= 1) {
        gt += __shfl_xor_sync(0xffffffffu, gt, o);
        eq += __shfl_xor_sync(0xffffffffu, eq, o);
    }
    if (lane == 0) { s_gt[warp] = gt; s_eq[warp] = eq; }
    __syncthreads();
    if (t == 0) {
        int eb = s_eqb0, sb = s_sel0;
        for (int w2 = 0; w2 < SP_W; ++w2) {
            const int g2 = s_gt[w2], e2 = s_eq[w2];
            const int take = eq_budget - eb < 0 ? 0 : (eq_budget - eb > e2 ? e2 : eq_budget - eb);
            s_gt[w2] = sb;
            s_eq[w2] = eb;
            sb += g2 + take;
            eb += e2;
        }
    }
    __syncthreads();
    int run_sel = s_gt[warp], run_eq = s_eq[warp];
    for (int64_t r0 = w0; r0 < w1; r0 += 32) {
        const int64_t b = r0 + lane;
        const int w = b < w1 ? weight(b) : 0;
        const uint32_t k = w ? order_key(sc[b]) : 0u;
        const int my_gt = (w && k > thr) ? w : 0, my_eq = (w && k == thr) ? w : 0;
        if (__ballot_sync(0xffffffffu, (my_gt | my_eq) != 0) == 0u) continue;
        int pe = my_eq;
#pragma unroll
        for (int o = 1; o < 32; o <<= 1) {
            const int y = __shfl_up_sync(0xffffffffu, pe, o);
            if (lane >= o) pe += y;
        }
        const int left = eq_budget - (run_eq + pe - my_eq);
        const int take = left < 0 ? 0 : (left > my_eq ? my_eq : left);
        const int my_sel = my_gt + take;
        int ps = my_sel;
#pragma unroll
        for (int o = 1; o < 32; o <<= 1) {
            const int y = __shfl_up_sync(0xffffffffu, ps, o);
            if (lane >= o) ps += y;
        }
        int32_t* dst = out + run_sel + ps - my_sel;
        for (int c = 0; c < my_sel; ++c) dst[c] = (int32_t) (b * R + c);
        run_eq += __shfl_sync(0xffffffffu, pe, 31);
        run_sel += __shfl_sync(0xffffffffu, ps, 31);
    }
}

bool topk_split_device() {
    static int ok[64] = {};   // 0 unknown, 1 yes, 2 no
    static const int env = [] { const char* v = std::getenv("STRATA_TOPK_SPLIT"); return v ? (std::atoi(v) != 0 ? 1 : 0) : -1; }();
    if (env == 0) return false;
    int dev = 0;
    if (cudaGetDevice(&dev) != cudaSuccess || dev < 0 || dev >= 64) { cudaGetLastError(); return false; }
    if (ok[dev] == 0) {
        int major = 0, minor = 0;
        cudaDeviceGetAttribute(&major, cudaDevAttrComputeCapabilityMajor, dev);
        cudaDeviceGetAttribute(&minor, cudaDevAttrComputeCapabilityMinor, dev);
        ok[dev] = (env == 1 || (major == 7 && minor == 5)) ? 1 : 2;
    }
    return ok[dev] == 1;
}

bool qsa_block_topk_split(const float* scores, const int32_t* steps, int64_t nq, int64_t max_blocks, int64_t cap,
                          int32_t* ids, cudaStream_t s) {
    const int64_t nch = (max_blocks + SP_CH - 1) / SP_CH;
    if (nq < 1 || nq > SP_MAXQ || nch > SP_MAXCH) return false;
    static void* hist_ptr[64] = {};
    int dev = 0;
    if (cudaGetDevice(&dev) != cudaSuccess || dev < 0 || dev >= 64) { cudaGetLastError(); return false; }
    if (hist_ptr[dev] == nullptr && cudaGetSymbolAddress(&hist_ptr[dev], g_sp_hist) != cudaSuccess) {
        cudaGetLastError();
        hist_ptr[dev] = nullptr;
        return false;
    }
    if (cudaMemsetAsync(hist_ptr[dev], 0, (size_t) nq * 4 * 256 * sizeof(int), s) != cudaSuccess) return false;
    const dim3 grid((unsigned) nch, (unsigned) nq);
    topk_split_hist_kernel<0><<<grid, SP_T, 0, s>>>(scores, steps, max_blocks);
    topk_split_hist_kernel<1><<<grid, SP_T, 0, s>>>(scores, steps, max_blocks);
    topk_split_hist_kernel<2><<<grid, SP_T, 0, s>>>(scores, steps, max_blocks);
    topk_split_hist_kernel<3><<<grid, SP_T, 0, s>>>(scores, steps, max_blocks);
    topk_split_count_kernel<<<grid, SP_T, 0, s>>>(scores, steps, max_blocks);
    topk_split_emit_kernel<<<grid, SP_T, 0, s>>>(scores, steps, max_blocks, cap, ids);
    return true;
}
#endif

#if !defined(__HIPCC__)
// ---- the prompt path's top-k with one full read of the scores (Turing).  block_topk_wide_kernel reads a query's scores
// six times (four radix passes, a count, the emit); past ~90K cells that is DRAM traffic (256 queries at 400K: 0.6 GB a
// layer, ~1.5 ms on an RTX 2080 Ti).  Here a CTA first reads every 32nd block and takes the KS-th largest of those keys
// as a bound LB, then reads all blocks once and keeps the ones with key >= LB in shared memory.  When those hold at
// least `width` cells, the width-th largest cell key (block_topk_kernel's threshold) is >= LB, so every block at or
// above the threshold is a candidate: the threshold, `above` and the tie budget computed over the candidates with
// block_topk_kernel's rule are the reference's, and the candidates sorted by block emit the reference's cells in its
// order.  Otherwise (too few cells above LB, or more than SM_CAP candidates) the query is flagged and
// block_topk_wide_kernel<true> runs it.  STRATA_TOPK_SAMPLE=0 keeps the wide kernel alone.
constexpr int TS_STRIDE = 32, TS_CAP = 4096;

__device__ __forceinline__ int ts_block_scan(int v, int* s_w) {   // exclusive scan over the CTA's TK_T threads
    const int t = threadIdx.x, lane = t & 31, w = t >> 5;
    int x = v;
#pragma unroll
    for (int o = 1; o < 32; o <<= 1) { const int y = __shfl_up_sync(0xffffffffu, x, o); if (lane >= o) x += y; }
    if (lane == 31) s_w[w] = x;
    __syncthreads();
    if (w == 0) {
        int y = lane < TK_T / 32 ? s_w[lane] : 0;
#pragma unroll
        for (int o = 1; o < 32; o <<= 1) { const int z = __shfl_up_sync(0xffffffffu, y, o); if (lane >= o) y += z; }
        if (lane < TK_T / 32) s_w[lane] = y;   // inclusive warp totals
    }
    __syncthreads();
    const int base = w > 0 ? s_w[w - 1] : 0;
    const int r = base + x - v;
    __syncthreads();
    return r;
}

__global__ void __launch_bounds__(TK_T) block_topk_sample_kernel(const float* __restrict__ scores,
                                                                 const int32_t* __restrict__ steps, int64_t max_blocks,
                                                                 int64_t cap, int32_t* __restrict__ ids) {
    __shared__ uint32_t s_key[TS_CAP];
    __shared__ int32_t s_idx[TS_CAP];
    __shared__ int hist[256];
    __shared__ int s_w[TK_T / 32];
    __shared__ int s_cnt, s_cells, s_digit, s_above;
    const int64_t qi = blockIdx.x;
    const int32_t* st = steps + qi * kStepCount;
    const int64_t n_kv = st[kStepNKv], n_bid = st[kStepNBid], width = st[kStepWidth];
    int32_t* out = ids + qi * cap;
    const int t = threadIdx.x;
    if (n_kv <= width) {
        if (t == 0) g_tk_fail[qi] = 0;
        for (int64_t j = t; j < n_kv; j += TK_T) out[j] = (int32_t) j;
        return;
    }
    const float* sc = scores + qi * max_blocks;
    const int64_t nb = n_bid + 1;
    auto weight = [&](int64_t b) -> int { return b < n_bid ? R : (int) (n_kv - n_bid * R); };
    const int ns = (int) ((nb + TS_STRIDE - 1) / TS_STRIDE);
    if (ns > TS_CAP) { if (t == 0) g_tk_fail[qi] = 1; return; }
    // 1. the bound: the KS-th largest of every 32nd block's key (a block without cells counts as the smallest key)
    for (int i = t; i < ns; i += TK_T) {
        const int64_t b = (int64_t) i * TS_STRIDE;
        s_key[i] = weight(b) ? order_key(sc[b]) : 0u;
    }
    const int ks_want = (int) (2 * ((width + R * TS_STRIDE - 1) / (R * TS_STRIDE)) + 16);
    int need = ks_want < ns ? ks_want : ns;
    uint32_t prefix = 0;
    for (int shift = 24; shift >= 0; shift -= 8) {
        for (int i = t; i < 256; i += TK_T) hist[i] = 0;
        __syncthreads();
        const uint32_t hi_mask = shift == 24 ? 0u : (0xffffffffu << (shift + 8));
        for (int i = t; i < ns; i += TK_T) {
            const uint32_t k = s_key[i];
            if ((k & hi_mask) == (prefix & hi_mask)) atomicAdd(&hist[(k >> shift) & 255], 1);
        }
        __syncthreads();
        if (t == 0) {
            int cum = 0, d = 255;
            for (; d > 0; --d) {
                if (cum + hist[d] >= need) break;
                cum += hist[d];
            }
            s_digit = d;
            s_above = need - cum;
        }
        __syncthreads();
        prefix |= (uint32_t) s_digit << shift;
        need = s_above;
        __syncthreads();
    }
    const uint32_t lb = prefix;
    // 2. one read of every block: the candidates (key >= LB) and their cells
    if (t == 0) { s_cnt = 0; s_cells = 0; }
    __syncthreads();
    for (int64_t b = t; b < nb; b += TK_T) {
        const int w = weight(b);
        if (w == 0) continue;
        const uint32_t k = order_key(sc[b]);
        if (k >= lb) {
            const int pos = atomicAdd(&s_cnt, 1);
            if (pos < TS_CAP) { s_key[pos] = k; s_idx[pos] = (int32_t) b; }
            atomicAdd(&s_cells, w);
        }
    }
    __syncthreads();
    const int n = s_cnt;
    if (n > TS_CAP || s_cells < width) {   // the bound was too low or too high: the wide kernel takes this query
        if (t == 0) g_tk_fail[qi] = 1;
        return;
    }
    if (t == 0) g_tk_fail[qi] = 0;
    // 3. block_topk_kernel's threshold over the candidates (weighted)
    prefix = 0;
    int above = 0;
    for (int shift = 24; shift >= 0; shift -= 8) {
        for (int i = t; i < 256; i += TK_T) hist[i] = 0;
        __syncthreads();
        const uint32_t hi_mask = shift == 24 ? 0u : (0xffffffffu << (shift + 8));
        for (int i = t; i < n; i += TK_T) {
            const uint32_t k = s_key[i];
            if ((k & hi_mask) == (prefix & hi_mask)) atomicAdd(&hist[(k >> shift) & 255], weight(s_idx[i]));
        }
        __syncthreads();
        if (t == 0) {
            int cum = above, d = 255;
            for (; d > 0; --d) {
                if (cum + hist[d] >= width) break;
                cum += hist[d];
            }
            s_digit = d;
            s_above = cum;
        }
        __syncthreads();
        prefix |= (uint32_t) s_digit << shift;
        above = s_above;
        __syncthreads();
    }
    const uint32_t thr = prefix;
    const int eq_budget = (int) (width - above);
    // 4. the candidates in block order (bitonic sort; padding sorts last)
    int n2 = 1;
    while (n2 < n) n2 <<= 1;
    for (int i = n + t; i < n2; i += TK_T) { s_idx[i] = 0x7fffffff; s_key[i] = 0u; }
    __syncthreads();
    for (int kk = 2; kk <= n2; kk <<= 1) {
        for (int j = kk >> 1; j > 0; j >>= 1) {
            for (int i = t; i < n2; i += TK_T) {
                const int p = i ^ j;
                if (p > i) {
                    const bool up = (i & kk) == 0;
                    const int a = s_idx[i], b = s_idx[p];
                    if ((a > b) == up) {
                        s_idx[i] = b; s_idx[p] = a;
                        const uint32_t ka = s_key[i]; s_key[i] = s_key[p]; s_key[p] = ka;
                    }
                }
            }
            __syncthreads();
        }
    }
    // 5. the emit: cells above thr, and the first eq_budget cells at thr, in block order
    constexpr int PER = TS_CAP / TK_T;              // candidates per thread, consecutive
    int eqc[PER], gtc[PER];
    int my_eq = 0;
#pragma unroll
    for (int r = 0; r < PER; ++r) {
        const int i = t * PER + r;
        const int w = i < n ? weight(s_idx[i]) : 0;
        const uint32_t k = s_key[i < n ? i : 0];
        gtc[r] = (i < n && k > thr) ? w : 0;
        eqc[r] = (i < n && k == thr) ? w : 0;
        my_eq += eqc[r];
    }
    int eq_before = ts_block_scan(my_eq, s_w);
    int sel[PER], my_sel = 0;
#pragma unroll
    for (int r = 0; r < PER; ++r) {
        const int left = eq_budget - eq_before;
        const int take = left < 0 ? 0 : (left > eqc[r] ? eqc[r] : left);
        sel[r] = gtc[r] + take;
        my_sel += sel[r];
        eq_before += eqc[r];
    }
    int pos = ts_block_scan(my_sel, s_w);
#pragma unroll
    for (int r = 0; r < PER; ++r) {
        const int i = t * PER + r;
        if (sel[r] > 0) {
            const int64_t b = s_idx[i];
            for (int c = 0; c < sel[r]; ++c) out[pos + c] = (int32_t) (b * R + c);
        }
        pos += sel[r];
    }
}

bool topk_sample_on() {
    static const int env = [] { const char* v = std::getenv("STRATA_TOPK_SAMPLE"); return v ? (std::atoi(v) != 0 ? 1 : 0) : -1; }();
    return env != 0;
}
#endif

void qsa_block_topk(const float* scores, const int32_t* steps, int64_t nq, int64_t max_blocks, int64_t cap,
                    const QsaShapes& s, int32_t* ids, void* stream, int64_t active_blocks) {
    // keys in registers when every query's blocks fit (contexts up to ~135K cells), else (CUDA) the same threads
    // reading them from memory; the same ids. STRATA_TOPK_OLD=1: the original kernel
    static const bool old = std::getenv("STRATA_TOPK_OLD") != nullptr;
    if (nq <= 0) return;
#if !defined(__HIPCC__)
    // sm_90+: a cluster of CL_N CTAs per query for the calls of a few queries (decode windows); the same ids.
    // STRATA_QSA_CLUSTER=0: the one-CTA kernels below
    static const bool cluster = [] {
        const char* v = std::getenv("STRATA_QSA_CLUSTER");
        return !v || std::atoi(v) != 0;
    }();
    if (cluster && !old && nq <= CL_MAXQ && qsa_block_topk_cluster(scores, steps, nq, max_blocks, cap, s, ids, stream))
        return;
#endif
#if !defined(__HIPCC__)
    // Turing decode (a window's few queries, no block count): the split kernels above; the same ids
    if (!old && active_blocks <= 0 && max_blocks >= 16384 && s.idx_block == R &&
        cap >= qsa_selection_width(kTopkMaxCells, s) && topk_split_device() &&
        qsa_block_topk_split(scores, steps, nq, max_blocks, cap, ids, (cudaStream_t) stream)) {
        const cudaError_t e = cudaGetLastError();
        if (e != cudaSuccess) { std::fprintf(stderr, "qsa_block_topk (split): %s\n", cudaGetErrorString(e)); std::exit(1); }
        return;
    }
#endif
    // the blocks a query can have: the call's active count when the caller knows it (the prompt path), else the capacity.
    // Decode (no count) keeps the capacity rule and the original register width: nothing changes there.
#if defined(__HIPCC__)
    const bool counted = active_blocks > 0;
#else
    // Turing: --max-context 262144 makes the stride 65538, even while a 131K prompt's active blocks fit in
    // TK_T * TK_PER registers. Use the prefill bound on sm_75, keeping max_blocks as the score-row stride.
    // Other CUDA devices keep 0.1.32's capacity rule: #337 was measured on RDNA4, and RTX 5070 64K prompts were
    // 1-3% slower. Decode/captured graphs omit the bound and never query the device here.
    static const bool capacity_guard = std::getenv("STRATA_TOPK_CAPACITY_GUARD") != nullptr;
    // STRATA_TOPK_ACTIVE_ANY=1 (tests): the Turing dispatch on any CUDA card, so qsa_topk_active_parity checks it
    // on whatever card runs the tests (the kernels are the same on every architecture)
    static const bool any_card = [] { const char* v = std::getenv("STRATA_TOPK_ACTIVE_ANY"); return v && v[0] == '1'; }();
    const bool counted = !capacity_guard && active_blocks > 0 && active_blocks <= max_blocks &&
                         (any_card || topk_active_turing_device());
#endif
    const int64_t reach = counted && active_blocks < max_blocks ? active_blocks : max_blocks;
    const int64_t fit = (int64_t) TK_T * (counted ? TK_PER_MAX : TK_PER);
#if defined(__HIPCC__)
    constexpr int64_t kRegMinBlocks = 7168;   // gfx1201: below ~28K cells the 1,024-thread kernel's fixed cost loses to the ref
    const bool too_small = counted && reach < kRegMinBlocks;
    if (old || too_small || reach > fit) {
#else
    constexpr int64_t kWideMinBlocks = 4608;   // RTX 3060: a prompt batch below ~18K cells is faster on the ref
    const bool too_small = reach > fit && active_blocks > 0 && active_blocks < kWideMinBlocks;
    if (old || too_small) {
#endif
        qsa_block_topk_ref(scores, steps, nq, max_blocks, cap, s, ids, stream);
        return;
    }
    if (s.idx_block != R || cap < qsa_selection_width(kTopkMaxCells, s)) {
        std::fprintf(stderr, "qsa_block_topk: unsupported geometry or cap\n");
        std::exit(1);
    }
#if !defined(__HIPCC__)
    // Turing prefill (the `counted` bound) above ~90K cells (22,528 blocks): the wide kernel, which reads the keys
    // coalesced, instead of the register kernel's uncoalesced per-thread runs (PR #743: 131K, 1.03 -> 0.75 ms); the
    // same ids.  STRATA_TOPK_STREAM=0 restores the register kernel there.
    static const bool turing_wide = [] {
        const char* v = std::getenv("STRATA_TOPK_STREAM");
        return v == nullptr || v[0] != '0';
    }();
    if (reach > fit || (turing_wide && counted && reach > 22528)) {   // past the register kernel's reach, or Turing's band
        if (counted && nq <= TS_MAXQ && reach > 28672 && topk_sample_on()) {   // one read of the scores (below ~115K cells the wide kernel is as fast)
            block_topk_sample_kernel<<<(unsigned) nq, TK_T, 0, (cudaStream_t) stream>>>(scores, steps, max_blocks, cap, ids);
            block_topk_wide_kernel<true><<<(unsigned) nq, TK_T, 0, (cudaStream_t) stream>>>(scores, steps, max_blocks, cap, ids);
        } else {
            block_topk_wide_kernel<false><<<(unsigned) nq, TK_T, 0, (cudaStream_t) stream>>>(scores, steps, max_blocks, cap, ids);
        }
    } else
#endif
    if (reach <= (int64_t) TK_T * TK_PER)
        block_topk_reg_kernel<TK_PER><<<(unsigned) nq, TK_T, 0, (cudaStream_t) stream>>>(scores, steps, max_blocks, cap, ids);
    else
        block_topk_reg_kernel<TK_PER_MAX><<<(unsigned) nq, TK_T, 0, (cudaStream_t) stream>>>(scores, steps, max_blocks, cap, ids);
    const cudaError_t e = cudaGetLastError();
    if (e != cudaSuccess) { std::fprintf(stderr, "qsa_block_topk: %s\n", cudaGetErrorString(e)); std::exit(1); }
}

}  // namespace strata::kernels
