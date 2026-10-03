// src/kernels/cuda/qsa_grouped_attn.cu - see include/strata/kernels/qsa_grouped_attn.hpp.
//
// The Volta WMMA wrapper below is adapted from llama.cpp-v100 ggml/src/ggml-cuda/fattn-sm70-grouped.cuh
// (Apache-2.0), itself adapted from 1Cat-vLLM's include/fused_mma.h: raw PTX wmma.m16n16k16 with the sm_70
// fragment register layout.  The split-K partial + combine shape is Strata's own qsa_decode_attn contract.
//
// WHY THE ROWS GROUP BY LAUNCH AND NOT BY SHARED TILES.  The window's M = 2..8 rows each carry their own top-k
// selection (qsa_select.hpp), so a shared cell stream (the union of the selections) makes the MMA tiles compute
// row/cell pairs nobody selected: measured at M = 8, 32K context, the 8 x 2,051 selections union to 5,500 cells
// - 2.68x of extra QK/PV compute, which ate the KV-read savings (the first design of this file, with the union
// and a per-cell membership mask, lost to the FP32 kernel).  What this kernel does instead: one CTA per
// (split, KV head, token) walks that token's OWN selection in 64-cell tiles, every selected KV row is read once
// and shared by the token's 12 query heads, and the window's rows run as ONE launch so the concurrent token
// CTAs' overlapping reads meet in L2.  On a 2x V100 (CUDA 12.8) this is 145 us per window against 181 us for
// `qsa_decode_attn_batch`'s FP32 kernel and 552 us for its per-row replay, at M = 8, 32K context, 2,051 cells.
//
// NUMBERS.  FP32-level, not bitwise, the `qsa_prompt_attn.hpp` precedent.  Queries and probabilities enter the
// MMAs as exact hi+lo FP16 pairs (~22 bits, kept out of the subnormal range by power-of-two prescaling), int8 K
// codes enter as exact FP16 and their per-64 scales fold into the score in FP32, FP16 pools enter as is.  What
// is left: the FP16 dequant of int8/q4 V (and of q4 K), and a different summation order.  `qsa_grouped_attn_
// parity` bounds it (1e-3 of the output scale; measured 2.2e-5 on fp16 KV, 3.2e-4 on int8 KV).  `STRATA_GA_
// HILO=0` builds trade the hi+lo pairs for one FP16 cast each (~1.4x faster, ~6e-4 of scale).  Deterministic:
// the tiles and the combine run in a fixed order, so a repeat run computes the same float sequence.
#include "strata/kernels/qsa_grouped_attn.hpp"
#include "strata/core/emulate.hpp"
#include "strata/kernels/kv_q8.hpp"
#include "strata/kernels/kv_q4.hpp"

#include <cuda_fp16.h>
#include <cuda_runtime.h>

#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>

// A/B: 1 = q and p enter the MMAs as hi+lo FP16 pairs (~22 bits, qsa_prompt_attn's contract), 0 = one FP16 cast
// each (llama.cpp's contract).  Measured to decide the shipped default.
#ifndef STRATA_GA_HILO
#define STRATA_GA_HILO 1
#endif

namespace strata::kernels {
namespace {

constexpr int HD = 256;             // head_dim
constexpr int G = 12;               // query heads per KV head (24 / 2)
constexpr int NH = 24;              // query heads total
constexpr int HEADS = 6;            // query heads per CTA (2 head groups per KV head -> 48-row CTAs)
constexpr int BN = 64;              // cells per KV tile
constexpr int THREADS = 512;
constexpr int WARPS = THREADS / 32;
constexpr int QS = HD + 8;          // panel row stride in halves (16-byte rows, spread banks)
constexpr int SS = BN + 4;          // score/accumulator row stride in floats
constexpr int PS = BN + 8;          // p-panel row stride in halves (every row start 16B aligned for wmma.load)
constexpr int kCellCap = 32768;     // kTopkMaxCells: the cell id space the union bitmap covers
constexpr float kNegInf = -1.0e30f; // not -INFINITY: all-masked rows must not produce NaN in exp2
constexpr float kVup = 16384.0f;    // p enters the MMAs as hi+lo parts of p * 2^14 (exact, normal)

// every panel row start must be 16-byte aligned: wmma.load/store touch each row at base + row * ldm
static_assert(QS * 2 % 16 == 0 && SS * 4 % 16 == 0 && PS * 2 % 16 == 0, "panel strides must keep rows 16B aligned");

// ---------------- Volta WMMA (raw PTX; sm_70 fragment layouts) ----------------
namespace wmma70 {

struct row_major {};
struct col_major {};
struct matrix_a {};
struct matrix_b {};
struct accumulator {};

enum layout_t { mem_row_major, mem_col_major };

template <typename Use, int M, int N, int K, typename T, typename Layout = void>
struct fragment;

template <>
struct fragment<matrix_a, 16, 16, 16, half, row_major> {
    uint32_t x[8];
    static constexpr int num_elements = 16;
};
template <>
struct fragment<matrix_b, 16, 16, 16, half, col_major> {
    uint32_t x[8];
    static constexpr int num_elements = 16;
};
template <>
struct fragment<matrix_b, 16, 16, 16, half, row_major> {
    uint32_t x[8];
    static constexpr int num_elements = 16;
};
template <>
struct fragment<accumulator, 16, 16, 16, float> {
    float x[8];
    static constexpr int num_elements = 8;
};

__device__ __forceinline__ void fill_fragment(fragment<accumulator, 16, 16, 16, float>& frag, float value) {
#pragma unroll
    for (int i = 0; i < 8; ++i) frag.x[i] = value;
}

__device__ __forceinline__ void load_matrix_sync(fragment<matrix_a, 16, 16, 16, half, row_major>& frag,
                                                const half* smem_ptr, unsigned ldm) {
    asm volatile("wmma.load.a.sync.aligned.row.m16n16k16.f16 {%0,%1,%2,%3,%4,%5,%6,%7}, [%8], %9;"
                 : "=r"(frag.x[0]), "=r"(frag.x[1]), "=r"(frag.x[2]), "=r"(frag.x[3]), "=r"(frag.x[4]),
                   "=r"(frag.x[5]), "=r"(frag.x[6]), "=r"(frag.x[7])
                 : "l"(smem_ptr), "r"(ldm)
                 : "memory");
}

__device__ __forceinline__ void load_matrix_sync(fragment<matrix_b, 16, 16, 16, half, row_major>& frag,
                                                const half* smem_ptr, unsigned ldm) {
    asm volatile("wmma.load.b.sync.aligned.row.m16n16k16.f16 {%0,%1,%2,%3,%4,%5,%6,%7}, [%8], %9;"
                 : "=r"(frag.x[0]), "=r"(frag.x[1]), "=r"(frag.x[2]), "=r"(frag.x[3]), "=r"(frag.x[4]),
                   "=r"(frag.x[5]), "=r"(frag.x[6]), "=r"(frag.x[7])
                 : "l"(smem_ptr), "r"(ldm)
                 : "memory");
}

__device__ __forceinline__ void load_matrix_sync(fragment<matrix_b, 16, 16, 16, half, col_major>& frag,
                                                const half* smem_ptr, unsigned ldm) {
    asm volatile("wmma.load.b.sync.aligned.col.m16n16k16.f16 {%0,%1,%2,%3,%4,%5,%6,%7}, [%8], %9;"
                 : "=r"(frag.x[0]), "=r"(frag.x[1]), "=r"(frag.x[2]), "=r"(frag.x[3]), "=r"(frag.x[4]),
                   "=r"(frag.x[5]), "=r"(frag.x[6]), "=r"(frag.x[7])
                 : "l"(smem_ptr), "r"(ldm)
                 : "memory");
}

__device__ __forceinline__ void store_matrix_sync(float* smem_ptr,
                                                  const fragment<accumulator, 16, 16, 16, float>& frag,
                                                  unsigned ldm, layout_t layout) {
    if (layout == mem_row_major) {
        asm volatile("wmma.store.d.sync.aligned.row.m16n16k16.f32 [%0], {%1,%2,%3,%4,%5,%6,%7,%8}, %9;"
                     :
                     : "l"(smem_ptr), "f"(frag.x[0]), "f"(frag.x[1]), "f"(frag.x[2]), "f"(frag.x[3]), "f"(frag.x[4]),
                       "f"(frag.x[5]), "f"(frag.x[6]), "f"(frag.x[7]), "r"(ldm)
                     : "memory");
    } else {
        asm volatile("wmma.store.d.sync.aligned.col.m16n16k16.f32 [%0], {%1,%2,%3,%4,%5,%6,%7,%8}, %9;"
                     :
                     : "l"(smem_ptr), "f"(frag.x[0]), "f"(frag.x[1]), "f"(frag.x[2]), "f"(frag.x[3]), "f"(frag.x[4]),
                       "f"(frag.x[5]), "f"(frag.x[6]), "f"(frag.x[7]), "r"(ldm)
                     : "memory");
    }
}

__device__ __forceinline__ void mma_sync(fragment<accumulator, 16, 16, 16, float>& d,
                                         const fragment<matrix_a, 16, 16, 16, half, row_major>& a,
                                         const fragment<matrix_b, 16, 16, 16, half, col_major>& b,
                                         const fragment<accumulator, 16, 16, 16, float>& c) {
    asm volatile("wmma.mma.sync.aligned.row.col.m16n16k16.f32.f32 "
                 "{%0,%1,%2,%3,%4,%5,%6,%7}, {%8,%9,%10,%11,%12,%13,%14,%15}, "
                 "{%16,%17,%18,%19,%20,%21,%22,%23}, {%24,%25,%26,%27,%28,%29,%30,%31};"
                 : "=f"(d.x[0]), "=f"(d.x[1]), "=f"(d.x[2]), "=f"(d.x[3]), "=f"(d.x[4]), "=f"(d.x[5]), "=f"(d.x[6]),
                   "=f"(d.x[7])
                 : "r"(a.x[0]), "r"(a.x[1]), "r"(a.x[2]), "r"(a.x[3]), "r"(a.x[4]), "r"(a.x[5]), "r"(a.x[6]),
                   "r"(a.x[7]), "r"(b.x[0]), "r"(b.x[1]), "r"(b.x[2]), "r"(b.x[3]), "r"(b.x[4]), "r"(b.x[5]),
                   "r"(b.x[6]), "r"(b.x[7]), "f"(c.x[0]), "f"(c.x[1]), "f"(c.x[2]), "f"(c.x[3]), "f"(c.x[4]),
                   "f"(c.x[5]), "f"(c.x[6]), "f"(c.x[7]));
}

__device__ __forceinline__ void mma_sync(fragment<accumulator, 16, 16, 16, float>& d,
                                         const fragment<matrix_a, 16, 16, 16, half, row_major>& a,
                                         const fragment<matrix_b, 16, 16, 16, half, row_major>& b,
                                         const fragment<accumulator, 16, 16, 16, float>& c) {
    asm volatile("wmma.mma.sync.aligned.row.row.m16n16k16.f32.f32 "
                 "{%0,%1,%2,%3,%4,%5,%6,%7}, {%8,%9,%10,%11,%12,%13,%14,%15}, "
                 "{%16,%17,%18,%19,%20,%21,%22,%23}, {%24,%25,%26,%27,%28,%29,%30,%31};"
                 : "=f"(d.x[0]), "=f"(d.x[1]), "=f"(d.x[2]), "=f"(d.x[3]), "=f"(d.x[4]), "=f"(d.x[5]), "=f"(d.x[6]),
                   "=f"(d.x[7])
                 : "r"(a.x[0]), "r"(a.x[1]), "r"(a.x[2]), "r"(a.x[3]), "r"(a.x[4]), "r"(a.x[5]), "r"(a.x[6]),
                   "r"(a.x[7]), "r"(b.x[0]), "r"(b.x[1]), "r"(b.x[2]), "r"(b.x[3]), "r"(b.x[4]), "r"(b.x[5]),
                   "r"(b.x[6]), "r"(b.x[7]), "f"(c.x[0]), "f"(c.x[1]), "f"(c.x[2]), "f"(c.x[3]), "f"(c.x[4]),
                   "f"(c.x[5]), "f"(c.x[6]), "f"(c.x[7]));
}

}  // namespace wmma70

// The sm_70 accumulator fragment's row mapping (fattn-sm70-grouped.cuh's scale_output_fragment): a lane's
// x[0],x[1],x[4],x[5] are fragment rows `row`, x[2],x[3],x[6],x[7] rows `row + 2`.
__device__ __forceinline__ void scale_output_fragment(wmma70::fragment<wmma70::accumulator, 16, 16, 16, float>& f,
                                                      const float* row_scale, int tile_row_start) {
    const int lane = threadIdx.x & 31;
    const int row = (lane & 1) + ((lane >> 2) & 1) * 8 + ((lane >> 4) & 1) * 4;
    const float a = row_scale[tile_row_start + row], b = row_scale[tile_row_start + row + 2];
    f.x[0] *= a;
    f.x[1] *= a;
    f.x[2] *= b;
    f.x[3] *= b;
    f.x[4] *= a;
    f.x[5] *= a;
    f.x[6] *= b;
    f.x[7] *= b;
}

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

// The panel fills below load 16-byte pieces (the per-element loop of the first version measured 88 us of the
// 264 us window for the K panel alone).  int8 K codes convert EXACTLY to FP16 (their per-64 scales fold into
// the score in FP32); int8 V and q4_0 rows dequantize here (the one rounding the parity bound covers).
__device__ void fill_q4_piece(const uint8_t* q4, long long row, int b, __half* out) {
    constexpr int bytes_per_head = (HD / QK4_0) * (int) sizeof(block_q4_0);
    const block_q4_0* blk = reinterpret_cast<const block_q4_0*>(q4 + row * bytes_per_head) + b;
    const float dscale = row >= 0 ? __half2float(__ushort_as_half(blk->d)) : 0.0f;
    const uint8_t* qs = row >= 0 ? blk->qs : nullptr;
    for (int j = 0; j < QK4_0 / 2; ++j) {
        const int lo = qs ? (int) (qs[j] & 0x0F) - 8 : 0;
        const int hi = qs ? (int) (qs[j] >> 4) - 8 : 0;
        out[j] = __float2half_rn((float) lo * dscale);
        out[j + QK4_0 / 2] = __float2half_rn((float) hi * dscale);
    }
}

template <int KV_MODE>
__device__ void fill_panel_rows(const QsaAttnPools& p, bool value, int c0, int n_here, const int32_t* ids,
                                int page_size, int n_kv_heads, int kvh, __half* panel) {
    auto row_of = [&](int c) -> long long {
        const int cell = ids[c];
        const long long page = (long long) p.page_table[cell / page_size];
        return page >= 0 ? (page * n_kv_heads + kvh) * page_size + (cell % page_size) : -1;
    };
    if constexpr (KV_MODE == 0) {
        const uint16_t* pool = value ? p.v_pool : p.k_pool;
        for (int i = threadIdx.x; i < BN * (HD / 8); i += THREADS) {
            const int c = i / (HD / 8), pc = i % (HD / 8);
            const long long row = c < n_here ? row_of(c0 + c) : -1;
            uint4 v = make_uint4(0, 0, 0, 0);
            if (row >= 0) v = *reinterpret_cast<const uint4*>(pool + row * HD + pc * 8);
            *reinterpret_cast<uint4*>(panel + (size_t) c * QS + pc * 8) = v;
        }
    } else if constexpr (KV_MODE == 2) {
        for (int i = threadIdx.x; i < BN * (HD / QK4_0); i += THREADS) {
            const int c = i / (HD / QK4_0), b = i % (HD / QK4_0);
            fill_q4_piece(value ? p.v_q4 : p.k_q4, c < n_here ? row_of(c0 + c) : -1, b,
                          panel + (size_t) c * QS + b * QK4_0);
        }
    } else {   // 1: int8 K (codes exact) + int8 V (dequant); 3: int8 K + q4_0 V
        const bool q4_v = KV_MODE == 3 && value;
        if (q4_v) {
            for (int i = threadIdx.x; i < BN * (HD / QK4_0); i += THREADS) {
                const int c = i / (HD / QK4_0), b = i % (HD / QK4_0);
                fill_q4_piece(p.v_q4, c < n_here ? row_of(c0 + c) : -1, b, panel + (size_t) c * QS + b * QK4_0);
            }
        } else {
            const int8_t* codes = value ? p.v_q : p.k_q;
            const uint16_t* sc = value ? p.v_scale : nullptr;
            for (int i = threadIdx.x; i < BN * (HD / 16); i += THREADS) {
                const int c = i / (HD / 16), pc = i % (HD / 16);
                const long long row = c < n_here ? row_of(c0 + c) : -1;
                uint4 raw = make_uint4(0, 0, 0, 0);
                float d = 1.0f;
                if (row >= 0) {
                    raw = *reinterpret_cast<const uint4*>(codes + row * HD + pc * 16);
                    if (value) d = __half2float(__ushort_as_half(sc[row * (HD / KV_Q8_GROUP) + pc / 4]));
                }
                const int8_t* cb = reinterpret_cast<const int8_t*>(&raw);
                __half* out = panel + (size_t) c * QS + pc * 16;
                for (int j = 0; j < 16; ++j) out[j] = __float2half_rn((float) cb[j] * d);
            }
        }
    }
}

// One CTA: one token's 12 query rows for one KV head over its own selection.  grid = (n_splits, n_head_kv, n_q)
// - the window's M = 2..8 rows run as ONE kernel launch.  Each KV row of a selected cell is loaded once per CTA
// and shared by all 12 heads; the concurrent token CTAs' overlapping reads meet in L2 (the rows' top-k selections
// differ, and the measured union-to-row ratio at M = 8 is 2.68x of extra QK/PV compute - the reason the rows are
// grouped by launch and by shared reads rather than by shared tiles).
struct Smem {
    union {
        struct {
            __half qh[16][QS];
#if STRATA_GA_HILO
            __half ql[16][QS];
#endif
            __half kv[BN][QS];        // the K panel during QK, the V panel during p.v
            float s[16][SS];          // wmma store scratch / p (unnormalized exp2)
            float a[16][SS];          // int8 K scale-group fold accumulator
            __half ph[16][PS];        // p (scaled by 2^14) hi and lo parts
#if STRATA_GA_HILO
            __half pl[16][PS];
#endif
            float ks[BN][4];          // int8 K scale of each cell's four 64-dim groups
            unsigned char cok[BN];    // the cell's pool row resolved (page resident)
        } c;
        float out[16][HD];
    } u;
    float rmax[16], rsum[16], rsc[16];
    float qmx[WARPS];
};

template <int KV_MODE>
__global__ void __launch_bounds__(THREADS) grouped_attn_kernel(
    const float* __restrict__ q, QsaAttnPools p, const int32_t* __restrict__ ids, const int32_t* __restrict__ steps,
    int64_t cap, int n_q, int n_kv_heads, int page_size, float scale_log2, float* __restrict__ part,
    int n_splits) {
    extern __shared__ __align__(256) unsigned char raw[];
    Smem& S = *reinterpret_cast<Smem*>(raw);
    const int token = blockIdx.z, kvh = blockIdx.y;
    const int t = threadIdx.x, lane = t & 31, warp = t >> 5;
    ids += (size_t) token * (size_t) cap;
    const int width = __ldg(steps + (size_t) token * kStepCount + kStepWidth);
    // every filler resolves its own pool row (the srow array's barrier measured ~1 us; this is 2 loads)
    auto row_of = [&](int c) -> long long {
        const int cell = ids[c];
        const long long page = (long long) p.page_table[cell / page_size];
        return page >= 0 ? (page * n_kv_heads + kvh) * page_size + (cell % page_size) : -1;
    };

    // ---- q: this token's 12 head rows (4 zero pad rows complete the 16-row WMMA tile)
    float qm = 0.0f;
    for (int i = t; i < G * HD; i += THREADS) qm = fmaxf(qm, fabsf(q[((size_t) token * NH + kvh * G) * HD + i]));
    qm = warp_max(qm);
    if (lane == 0) S.qmx[warp] = qm;
    __syncthreads();
    {
        float m = 0.0f;
#pragma unroll
        for (int w = 0; w < WARPS; ++w) m = fmaxf(m, S.qmx[w]);
        qm = m;
    }
    int qe = 0;
    if (qm > 0.0f) frexpf(qm, &qe);
    const float qup = ldexpf(1.0f, 14 - qe), qdown = ldexpf(scale_log2, qe - 14);
    for (int i = t; i < 16 * HD; i += THREADS) {
        const int h = i / HD, d = i % HD;
        float x = h < G ? q[((size_t) token * NH + kvh * G + h) * HD + d] * qup : 0.0f;
        S.u.c.qh[h][d] = __float2half_rn(x);
#if STRATA_GA_HILO
        const __half hi = S.u.c.qh[h][d];
        S.u.c.ql[h][d] = __float2half_rn(x - __half2float(hi));
#endif
    }
    for (int r = t; r < 16; r += THREADS) {
        S.rmax[r] = kNegInf;
        S.rsum[r] = 0.0f;
        S.rsc[r] = 1.0f;
    }
    __syncthreads();

    wmma70::fragment<wmma70::accumulator, 16, 16, 16, float> out_frag;
    wmma70::fill_fragment(out_frag, 0.0f);

    const int n_tiles = (width + BN - 1) / BN;
    const int ts = (int) ((long long) blockIdx.x * n_tiles / n_splits);
    const int te = (int) ((long long) (blockIdx.x + 1) * n_tiles / n_splits);

    // Four barriers per tile (was six): p.v done -> K panel -> QK -> softmax (p into ph) + V panel -> p.v.
    for (int tile = ts; tile < te; ++tile) {
        __syncthreads();
        const int c0 = tile * BN;
        const int n_here = min(BN, width - c0);
        // ---- K panel (and the int8 K scales) into panel 0
        fill_panel_rows<KV_MODE>(p, false, c0, n_here, ids, page_size, n_kv_heads, kvh, &S.u.c.kv[0][0]);
        if (t < BN) S.u.c.cok[t] = (t < n_here && row_of(c0 + t) >= 0) ? 1 : 0;
        if (KV_MODE == 1 || KV_MODE == 3) {
            for (int i = t; i < BN * 4; i += THREADS) {
                const int c = i / 4, g = i % 4;
                const long long row = c < n_here ? row_of(c0 + c) : -1;
                S.u.c.ks[c][g] =
                    row >= 0 ? __half2float(__ushort_as_half(p.k_scale[row * (HD / KV_Q8_GROUP) + g])) : 0.0f;
            }
        }
        __syncthreads();   // A: the K panel is complete

        // ---- q.k: one warp per 16-cell tile, hi + lo parts, int8 scales folded per 64-dim group
        if (warp < BN / 16) {
            const int nt = warp;
            using frag_t = wmma70::fragment<wmma70::accumulator, 16, 16, 16, float>;
            if (KV_MODE == 1 || KV_MODE == 3) {
                for (int i = lane; i < 16 * 16; i += 32) S.u.c.a[i / 16][nt * 16 + i % 16] = 0.0f;
                frag_t fg[4];
#pragma unroll
                for (int g = 0; g < 4; ++g) wmma70::fill_fragment(fg[g], 0.0f);
#pragma unroll
                for (int g = 0; g < 4; ++g) {
#pragma unroll
                    for (int kk = 0; kk < 4; ++kk) {
                        const int k0 = (g * 4 + kk) * 16;
                        wmma70::fragment<wmma70::matrix_a, 16, 16, 16, half, wmma70::row_major> ah;
                        wmma70::fragment<wmma70::matrix_b, 16, 16, 16, half, wmma70::col_major> b;
                        wmma70::load_matrix_sync(ah, &S.u.c.qh[0][k0], QS);
                        wmma70::load_matrix_sync(b, &S.u.c.kv[nt * 16][k0], QS);
                        wmma70::mma_sync(fg[g], ah, b, fg[g]);
#if STRATA_GA_HILO
                        wmma70::fragment<wmma70::matrix_a, 16, 16, 16, half, wmma70::row_major> al;
                        wmma70::load_matrix_sync(al, &S.u.c.ql[0][k0], QS);
                        wmma70::mma_sync(fg[g], al, b, fg[g]);
#endif
                    }
                }
                for (int g = 0; g < 4; ++g) {
                    wmma70::store_matrix_sync(&S.u.c.s[0][nt * 16], fg[g], SS, wmma70::mem_row_major);
                    __syncwarp();
                    for (int i = lane; i < 16 * 16; i += 32) {
                        const int rr = i / 16, cc = nt * 16 + i % 16;
                        S.u.c.a[rr][cc] += S.u.c.ks[cc][g] * S.u.c.s[rr][cc];
                    }
                }
                __syncwarp();
                for (int i = lane; i < 16 * 16; i += 32) {
                    const int rr = i / 16, cc = nt * 16 + i % 16;
                    S.u.c.s[rr][cc] = (rr < G && S.u.c.cok[cc]) ? S.u.c.a[rr][cc] * qdown : kNegInf;
                }
            } else {
                frag_t fq;
                wmma70::fill_fragment(fq, 0.0f);
#pragma unroll
                for (int kk = 0; kk < 16; ++kk) {
                    const int k0 = kk * 16;
                    wmma70::fragment<wmma70::matrix_a, 16, 16, 16, half, wmma70::row_major> ah;
                    wmma70::fragment<wmma70::matrix_b, 16, 16, 16, half, wmma70::col_major> b;
                    wmma70::load_matrix_sync(ah, &S.u.c.qh[0][k0], QS);
                    wmma70::load_matrix_sync(b, &S.u.c.kv[nt * 16][k0], QS);
                    wmma70::mma_sync(fq, ah, b, fq);
#if STRATA_GA_HILO
                    wmma70::fragment<wmma70::matrix_a, 16, 16, 16, half, wmma70::row_major> al;
                    wmma70::load_matrix_sync(al, &S.u.c.ql[0][k0], QS);
                    wmma70::mma_sync(fq, al, b, fq);
#endif
                }
                wmma70::store_matrix_sync(&S.u.c.s[0][nt * 16], fq, SS, wmma70::mem_row_major);
                __syncwarp();
                for (int i = lane; i < 16 * 16; i += 32) {
                    const int rr = i / 16, cc = nt * 16 + i % 16;
                    S.u.c.s[rr][cc] = (rr < G && S.u.c.cok[cc]) ? S.u.c.s[rr][cc] * qdown : kNegInf;
                }
            }
        }
        __syncthreads();   // B: final scores are in s

        // ---- online softmax (exp2 domain) writes p AND the p * 2^14 panel, so no later pass touches s;
        // the V fill into panel 1 runs beside it (different buffers)
        fill_panel_rows<KV_MODE>(p, true, c0, n_here, ids, page_size, n_kv_heads, kvh, &S.u.c.kv[0][0]);
        {
            const int r = warp;
            const float x0 = S.u.c.s[r][lane], x1 = S.u.c.s[r][lane + 32];   // two cells per lane
            const float xm = warp_max(fmaxf(x0, x1));
            float m_new = fmaxf(S.rmax[r], xm);
            if (m_new < kNegInf / 2) m_new = kNegInf;
            const float alpha = S.rmax[r] < kNegInf / 2 ? 1.0f : exp2f(S.rmax[r] - m_new);
            const float p0 = x0 < kNegInf / 2 ? 0.0f : exp2f(x0 - m_new);
            const float p1 = x1 < kNegInf / 2 ? 0.0f : exp2f(x1 - m_new);
            const float sum = warp_sum(p0 + p1);
            __syncwarp();
            const float s0 = p0 * kVup, s1 = p1 * kVup;
            const __half h0 = __float2half_rn(s0), h1 = __float2half_rn(s1);
            S.u.c.ph[r][lane] = h0;
            S.u.c.ph[r][lane + 32] = h1;
#if STRATA_GA_HILO
            S.u.c.pl[r][lane] = __float2half_rn(s0 - __half2float(h0));
            S.u.c.pl[r][lane + 32] = __float2half_rn(s1 - __half2float(h1));
#endif
            if (lane == 0) {
                S.rsc[r] = alpha;
                S.rsum[r] = S.rsum[r] * alpha + sum;
                S.rmax[r] = m_new;
            }
        }
        __syncthreads();   // C: p panels and the V panel are ready
        // ---- p.v: one warp per 16-dim output tile; rescale the accumulator first (online softmax)
        scale_output_fragment(out_frag, S.rsc, 0);
#pragma unroll
        for (int kk = 0; kk < BN / 16; ++kk) {
            wmma70::fragment<wmma70::matrix_b, 16, 16, 16, half, wmma70::row_major> b;
            wmma70::load_matrix_sync(b, &S.u.c.kv[kk * 16][warp * 16], QS);
            wmma70::fragment<wmma70::matrix_a, 16, 16, 16, half, wmma70::row_major> ah;
            wmma70::load_matrix_sync(ah, &S.u.c.ph[0][kk * 16], PS);
            wmma70::mma_sync(out_frag, ah, b, out_frag);
#if STRATA_GA_HILO
            wmma70::fragment<wmma70::matrix_a, 16, 16, 16, half, wmma70::row_major> al;
            wmma70::load_matrix_sync(al, &S.u.c.pl[0][kk * 16], PS);
            wmma70::mma_sync(out_frag, al, b, out_frag);
#endif
        }
    }

    // ---- unnormalized numerators + (max, sum) per split, the flash combine's contract
    __syncthreads();
    wmma70::store_matrix_sync(&S.u.out[0][warp * 16], out_frag, HD, wmma70::mem_row_major);
    __syncthreads();
    for (int i = t; i < G * HD; i += THREADS) {
        const int h = i / HD, d = i % HD;
        float* row = part + (((size_t) blockIdx.x * n_q + token) * NH + kvh * G + h) * (HD + 2);
        row[d] = S.u.out[h][d] * (1.0f / kVup);
        if (d == 0) {
            row[HD] = S.rmax[h];
            row[HD + 1] = S.rsum[h];
        }
    }
}

__global__ void __launch_bounds__(HD) group_combine_kernel(const float* __restrict__ part, float* __restrict__ attn,
                                                           int n_q, int n_splits) {
    const int token = blockIdx.x, head = blockIdx.y, d = threadIdx.x;
    __shared__ float sM, sL;
    if (d == 0) {
        float M = kNegInf;
        for (int s = 0; s < n_splits; ++s) {
            const float* row = part + (((size_t) s * n_q + token) * NH + head) * (HD + 2);
            M = fmaxf(M, row[HD]);
        }
        float L = 0.0f;
        for (int s = 0; s < n_splits; ++s) {
            const float* row = part + (((size_t) s * n_q + token) * NH + head) * (HD + 2);
            if (row[HD + 1] > 0.0f && row[HD] > kNegInf / 2) L += row[HD + 1] * exp2f(row[HD] - M);
        }
        sM = M;
        sL = L;
    }
    __syncthreads();
    float acc = 0.0f;
    for (int s = 0; s < n_splits; ++s) {
        const float* row = part + (((size_t) s * n_q + token) * NH + head) * (HD + 2);
        if (row[HD + 1] > 0.0f && row[HD] > kNegInf / 2) acc += row[d] * exp2f(row[HD] - sM);
    }
    attn[((size_t) token * NH + head) * HD + d] = sL > 0.0f ? acc / sL : 0.0f;
}

// ---- host side -------------------------------------------------------------

inline int grouped_dev() {
    int dev = 0;
    if (cudaGetDevice(&dev) != cudaSuccess || dev < 0 || dev >= 64) {
        cudaGetLastError();   // do not leave a probe error for a later error check to pick up
        return 0;
    }
    return dev;
}

// The shared-memory opt-in AND the capability probe are context properties of a DEVICE, not of the
// process: a --layer-split runs this kernel on two cards and each card's context must be opted in
// before its 63 KiB dynamic-smem launch is captured into the verify window's graph.  A process-global
// `static bool` let device 1 reuse device 0's success, skip its own opt-in, and the resulting failed
// launch invalidated the capture ("operation failed due to a previous error during capture").  Keep
// one record per device.  Every probe/opt-in failure clears the runtime error with cudaGetLastError():
// a refused opt-in sends the caller down the FP32 fallback, and the leaked error would otherwise
// surface later in that fallback's own cudaGetLastError() as a spurious abort.
struct GroupedDev {
    int cc = 0;              // 10*major + minor, 0 = not probed
    int smem_cap = 0;        // cudaDevAttrMaxSharedMemoryPerBlockOptin (emulation-aware), 0 = not probed
    int optin[4] = {};       // per KV_MODE: 0 = not tried, 1 = opted in, -1 = refused
};
inline GroupedDev& grouped_state() {
    static GroupedDev by_dev[64];
    return by_dev[grouped_dev()];
}

int grouped_cc() {
    int dev = 0;
    if (cudaGetDevice(&dev) != cudaSuccess || dev < 0 || dev >= 64) {
        cudaGetLastError();
        return -1;
    }
    GroupedDev& d = grouped_state();
    if (d.cc == 0) {
        int major = 0, minor = 0;
        if (cudaDeviceGetAttribute(&major, cudaDevAttrComputeCapabilityMajor, dev) != cudaSuccess ||
            cudaDeviceGetAttribute(&minor, cudaDevAttrComputeCapabilityMinor, dev) != cudaSuccess) {
            cudaGetLastError();
            return -1;
        }
        d.cc = 10 * strata::cc_major_of(major) + strata::cc_minor_of(minor);
    }
    return d.cc;
}

int grouped_env() {   // -1 default (cc gate), 0 off, 1 on; read per call so a test can A/B in one process
    const char* e = std::getenv("STRATA_GROUPED_ATTN");
    if (e == nullptr) return -1;
    return std::atoi(e) == 0 ? 0 : 1;
}

template <int KV_MODE>
bool set_attr() {
    GroupedDev& d = grouped_state();
    if (d.optin[KV_MODE] == 0) {
        const bool ok = cudaFuncSetAttribute(grouped_attn_kernel<KV_MODE>,
                                             cudaFuncAttributeMaxDynamicSharedMemorySize, (int) sizeof(Smem)) ==
                        cudaSuccess;
        d.optin[KV_MODE] = ok ? 1 : -1;
        cudaGetLastError();   // a refused opt-in must not poison the capture the fallback continues in
    }
    return d.optin[KV_MODE] == 1;
}

inline bool smem_ok() {
    GroupedDev& d = grouped_state();
    if (d.smem_cap == 0) {
        int optin = 0;
        if (cudaDeviceGetAttribute(&optin, cudaDevAttrMaxSharedMemoryPerBlockOptin, grouped_dev()) != cudaSuccess) {
            cudaGetLastError();
            return false;
        }
        d.smem_cap = strata::smem_optin_of(optin);
    }
    return (int) sizeof(Smem) <= d.smem_cap;
}

template <int KV_MODE>
void launch_grouped(const float* q, const QsaAttnPools& p, const int32_t* ids, const int32_t* steps, int64_t cap,
                    int n_q, int n_kv_heads, int page_size, float scale_log2, float* part, int n_splits,
                    cudaStream_t st) {   // n_q doubles as the token grid size
    grouped_attn_kernel<KV_MODE>
        <<<dim3((unsigned) n_splits, (unsigned) n_kv_heads, (unsigned) n_q), THREADS, sizeof(Smem), st>>>(
            q, p, ids, steps, cap, n_q, n_kv_heads, page_size, scale_log2, part, n_splits);
}

}  // namespace

bool qsa_grouped_attn_batch(const float* q, const QsaAttnPools& pools, const int32_t* ids, const int32_t* steps,
                            int64_t cap, const QsaShapes& s, float* scratch, float* attn, int64_t n_q,
                            void* stream) {
    if (n_q < 2 || n_q > 8) return false;
    if (s.head_dim != HD || s.n_head != NH || s.n_head_kv != 2 || cap <= 0 || cap > (int64_t) kCellCap) return false;
    if (q == nullptr || ids == nullptr || steps == nullptr || scratch == nullptr || attn == nullptr ||
        pools.page_table == nullptr)
        return false;
    const int env = grouped_env();
    if (env == 0) return false;
    if (env < 0) {
        const int cc = grouped_cc();
        if (cc < 70 || cc >= 80) return false;   // the Volta path; sm_80+ keeps the default kernel
    }
    const int kv_mode = pools.k_q4 != nullptr ? 2
                        : (pools.k_q != nullptr && pools.v_q4 != nullptr ? 3
                                        : (pools.k_q != nullptr ? 1 : 0));
    if (kv_mode == 3 ? (pools.k_scale == nullptr || pools.v_q4 == nullptr)
                     : (kv_mode == 2 ? (pools.k_q4 == nullptr || pools.v_q4 == nullptr)
                                     : (kv_mode == 1 ? (pools.k_q == nullptr || pools.v_q == nullptr ||
                                                        pools.k_scale == nullptr || pools.v_scale == nullptr)
                                                     : (pools.k_pool == nullptr || pools.v_pool == nullptr))))
        return false;
    if (!smem_ok()) return false;
    const bool attr = kv_mode == 0 ? set_attr<0>() : kv_mode == 1 ? set_attr<1>() : kv_mode == 2 ? set_attr<2>()
                                                                                                : set_attr<3>();
    if (!attr) return false;

    // The callers size scratch as n_q times `qsa_decode_attn_scratch_floats`.  The split-K partial rows are all
    // this path needs; as many splits as fit is pure parallelism.
    const int64_t budget = n_q * (int64_t) qsa_decode_attn_scratch_floats(cap, s);
    const int64_t per_split = (int64_t) n_q * NH * (HD + 2);
    if (budget < 4 * per_split) return false;
    const int n_splits = min(16, (int) (budget / per_split));   // the combine reads one partial row per split
    float* part = scratch;

    cudaStream_t st = (cudaStream_t) stream;
    const float scale_log2 = (1.0f / sqrtf((float) HD)) * 1.4426950408889634f;
    if (kv_mode == 0) launch_grouped<0>(q, pools, ids, steps, cap, (int) n_q, (int) s.n_head_kv, (int) s.page_size,
                                        scale_log2, part, n_splits, st);
    else if (kv_mode == 1) launch_grouped<1>(q, pools, ids, steps, cap, (int) n_q, (int) s.n_head_kv,
                                             (int) s.page_size, scale_log2, part, n_splits, st);
    else if (kv_mode == 2) launch_grouped<2>(q, pools, ids, steps, cap, (int) n_q, (int) s.n_head_kv,
                                             (int) s.page_size, scale_log2, part, n_splits, st);
    else launch_grouped<3>(q, pools, ids, steps, cap, (int) n_q, (int) s.n_head_kv, (int) s.page_size, scale_log2,
                           part, n_splits, st);
    group_combine_kernel<<<dim3((unsigned) n_q, (unsigned) NH), HD, 0, st>>>(part, attn, (int) n_q, n_splits);
    const cudaError_t e = cudaGetLastError();
    if (e != cudaSuccess) {
        std::fprintf(stderr, "qsa_grouped_attn_batch: %s\n", cudaGetErrorString(e));
        std::exit(1);
    }
    return true;
}

}  // namespace strata::kernels
