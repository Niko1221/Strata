// engine/src/inkling2/cuda/ink_attn_tc.cuh - tensor-core split-K decode attention for the 8 global visits at long context (sm_80+; tuned for sm_120).
//
// MIT License
// Copyright (c) 2026 Niko1221 and the Strata contributors
//
// FAST MODE ONLY. This kernel does not reproduce the rounding of k_ink_flash_gqa_splitk: q is rounded to FP16, the scores and the PV product run on
// FP16 tensor cores with FP32 accumulation, the probabilities that feed PV are rounded to FP16 (x 2^15, with the V scale folded in), the
// denominator keeps the FP32 probabilities. It is therefore never part of an exact set.
//
// Why it exists: the old kernel walks every key with a 5-step warp reduction per query head (3 per key), so at 128K it is ALU bound (about 20 ms per
// verify step for the 8 global visits on a 5090, 6-8x above the memory time). Here a block owns one KV head, one slice of the keys and a group of rows
// (3 query heads per row, up to 16 rows = 48 query rows = 3 slabs of 16); every K/V tile is read once for the whole group, converted to FP16 in shared
// memory and consumed by mma.m16n8k16; one warp owns one 16-row slab and keeps its scores, probabilities (as A fragments) and output in registers.
//
// Tensor-core split-K: 32-key tiles, FP16 probabilities scaled by 2^15 for PV, FP32 denominator, per-row slices that depend only on the row's own key count.
//
// Semantics are those of the old kernel: bias = rel_table[row, head, min(qpos - key, 1024)], per-query sink in the denominator (folded into split 0 for a
// plain step, into the tail slot of a verify step), 24:8 GQA, head_dim 128, FP16 and FP8 (E4M3 x per-(cell, head) FP32 scale) KV, host-tail cells
// (kv.dev_cap / kv.host_cap) read like ink_kv_load4 does. K8V8 / K8V4 are not handled (the caller keeps the old kernel).
//
// Partial layout (consumed by k_ink_flash_gqa_merge / _merge_h2 with n_splits = kInkTcSlots): slot s < kInkTcSplits is the s-th slice of the committed
// keys [0, nc), slot kInkTcSplits is the verify tail (the row's own ancestors, FP32, exact) with the sink; inactive slots hold m = -1e30, l = 0.
#pragma once

#include "ink_attn.cuh"
#include <cuda_fp16.h>
#include <cuda_fp8.h>
#include <cstdint>
#include <cstdlib>

namespace strata::ink {
namespace tcattn {

constexpr int kTile = 32;                 // keys per tile
constexpr int kLd = 136;                  // padded FP16 row stride (272 B): ldmatrix rows land on different banks
constexpr float kPScale = 32768.0f;       // 2^15, probabilities x kPScale go into the FP16 PV operand
constexpr int kMinSlice = 128;            // smallest key slice of a block (keys)
// flags (drafter block attention): no relative bias, no sink in the softmax, every row sees every tail key (a block, not a chain)
constexpr int kFlagNoBias = 1, kFlagNoSink = 2, kFlagBlockTail = 4;
constexpr int kMaxSplits = 96;        // most slices of the committed keys per KV head (the buffers are sized for this)
constexpr int kSlots = kMaxSplits + 1;   // + the verify tail
#define kNegInf __int_as_float(0xff800000)   // -inf (the INFINITY macro warns on MSVC)

__device__ __forceinline__ uint32_t smem_u32(const void* p) { return (uint32_t)__cvta_generic_to_shared(p); }

// ldmatrix and mma.m16n8k16 need sm_80 or newer (Turing / Volta device passes of a multi-arch build must still assemble: the bodies are a trap there, the host never
// selects the tensor-core attention below sm_120: it is a fast-set part of the RTX 50 rows)
#if defined(__CUDA_ARCH__) && (__CUDA_ARCH__ < 800)
#define INK_TC_ATTN_NO_MMA 1
#endif
__device__ __forceinline__ void ldsm_x4(uint32_t (&r)[4], const void* p) {
#ifdef INK_TC_ATTN_NO_MMA
    r[0] = r[1] = r[2] = r[3] = 0u; (void)p; __trap();
#else
    asm volatile("ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%0,%1,%2,%3}, [%4];\n"
                 : "=r"(r[0]), "=r"(r[1]), "=r"(r[2]), "=r"(r[3]) : "r"(smem_u32(p)));
#endif
}
__device__ __forceinline__ void ldsm_x4_t(uint32_t (&r)[4], const void* p) {
#ifdef INK_TC_ATTN_NO_MMA
    r[0] = r[1] = r[2] = r[3] = 0u; (void)p; __trap();
#else
    asm volatile("ldmatrix.sync.aligned.m8n8.x4.trans.shared.b16 {%0,%1,%2,%3}, [%4];\n"
                 : "=r"(r[0]), "=r"(r[1]), "=r"(r[2]), "=r"(r[3]) : "r"(smem_u32(p)));
#endif
}
__device__ __forceinline__ void mma16816(float (&c)[4], const uint32_t (&a)[4], uint32_t b0, uint32_t b1) {
#ifdef INK_TC_ATTN_NO_MMA
    (void)c; (void)a; (void)b0; (void)b1; __trap();
#else
    asm volatile("mma.sync.aligned.m16n8k16.row.col.f32.f16.f16.f32 {%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, {%0,%1,%2,%3};\n"
                 : "+f"(c[0]), "+f"(c[1]), "+f"(c[2]), "+f"(c[3])
                 : "r"(a[0]), "r"(a[1]), "r"(a[2]), "r"(a[3]), "r"(b0), "r"(b1));
#endif
}
__device__ __forceinline__ uint32_t pack_h2(float lo, float hi) {
    const __half2 h = __floats2half2_rn(lo, hi);
    return *reinterpret_cast<const uint32_t*>(&h);
}
__device__ __forceinline__ uint32_t pack_h2_sat(float lo, float hi) {   // probabilities: stay finite in FP16
    return pack_h2(fminf(lo, 65504.0f), fminf(hi, 65504.0f));
}
// 2 E4M3 bytes -> 2 halves (exact)
__device__ __forceinline__ uint32_t e4m3x2_to_h2(uint32_t two) {
    const __half2_raw r = __nv_cvt_fp8x2_to_halfraw2((__nv_fp8x2_storage_t)(two & 0xFFFFu), __NV_E4M3);
    return (uint32_t)r.x | ((uint32_t)r.y << 16);
}

// One cell of the global visit: the block's KV head of key `key` (the same addressing as ink_kv_load4).
struct CellPtr {
    const uint8_t* k;   // row of this head: 128 bytes (FP8) or 256 bytes (FP16)
    const uint8_t* v;
    const float* ks;    // FP8 only
    const float* vs;
};
__device__ __forceinline__ CellPtr cell_of(const InkKvVisitDesc& kv, int sl, int key, int kh) {
    const bool use_host = (kv.host_cap > 0 && key >= kv.dev_cap);
    const int stride = use_host ? kv.host_cap : (kv.host_cap > 0 ? kv.dev_cap : kv.cap_pos);
    const int pos_l = use_host ? (key - kv.dev_cap) : key;
    const size_t hbase = ((size_t)sl * stride + pos_l) * kInkKVHeads + kh;
    CellPtr c;
    if (kv.kv_fmt == 0) {
        c.k = reinterpret_cast<const uint8_t*>(use_host ? kv.h_k_f16 : kv.k_f16) + hbase * 256;
        c.v = reinterpret_cast<const uint8_t*>(use_host ? kv.h_v_f16 : kv.v_f16) + hbase * 256;
        c.ks = nullptr; c.vs = nullptr;
    } else {
        c.k = (use_host ? kv.h_k_f8 : kv.k_f8) + hbase * 128;
        c.v = (use_host ? kv.h_v_f8 : kv.v_f8) + hbase * 128;
        c.ks = (use_host ? kv.h_k_sc : kv.k_sc) + hbase;
        c.vs = (use_host ? kv.h_v_sc : kv.v_sc) + hbase;
    }
    return c;
}

// Slice of the committed keys of a row group: the same on every row of the group (and in the merge step), a function of nc alone.
__host__ __device__ __forceinline__ int slice_len(int nc, int nsp) {
    int per = (nc + nsp - 1) / nsp;
    per = (per + kTile - 1) / kTile * kTile;
    return per < kMinSlice ? kMinSlice : per;
}

template <int FMT>
struct Fmt {
    static constexpr int kPieces = (FMT == 0) ? 16 : 8;            // 16-byte pieces of a K (or V) row
    static constexpr int kKeysPerPass = 128 / kPieces;             // keys covered by the 128 threads in one pass
    static constexpr int kPasses = kTile / kKeysPerPass;           // 4 (FP16) / 2 (FP8)
};

template <int FMT>
struct RawTile {
    uint4 k[Fmt<FMT>::kPasses];
    uint4 v[Fmt<FMT>::kPasses];
    float ks[Fmt<FMT>::kPasses];
    float vs[Fmt<FMT>::kPasses];
};

template <int FMT>
__device__ __forceinline__ void load_tile(RawTile<FMT>& t, const InkKvVisitDesc& kv, int sl, int kh, int tile_lo, int hi, int tid) {
    constexpr int P = Fmt<FMT>::kPieces, KPP = Fmt<FMT>::kKeysPerPass, NP = Fmt<FMT>::kPasses;
    const int piece = tid % P;
#pragma unroll
    for (int i = 0; i < NP; ++i) {
        const int j = tid / P + KPP * i;
        const int key = tile_lo + j;
        t.k[i] = make_uint4(0, 0, 0, 0);
        t.v[i] = make_uint4(0, 0, 0, 0);
        t.ks[i] = 0.0f;
        t.vs[i] = 0.0f;
        if (key < hi) {
            const CellPtr c = cell_of(kv, sl, key, kh);
            t.k[i] = __ldcs(reinterpret_cast<const uint4*>(c.k) + piece);
            t.v[i] = __ldcs(reinterpret_cast<const uint4*>(c.v) + piece);
            if constexpr (FMT == 1) {
                t.ks[i] = __ldg(c.ks);
                t.vs[i] = __ldg(c.vs);
            }
        }
    }
}

template <int FMT>
__device__ __forceinline__ void store_tile(const RawTile<FMT>& t, __half (*sk)[kLd], __half (*sv)[kLd], float* sks, float* svs, int tid) {
    constexpr int P = Fmt<FMT>::kPieces, KPP = Fmt<FMT>::kKeysPerPass, NP = Fmt<FMT>::kPasses;
    const int piece = tid % P;
#pragma unroll
    for (int i = 0; i < NP; ++i) {
        const int j = tid / P + KPP * i;
        if constexpr (FMT == 0) {
            *reinterpret_cast<uint4*>(&sk[j][piece * 8]) = t.k[i];
            *reinterpret_cast<uint4*>(&sv[j][piece * 8]) = t.v[i];
        } else {
            const uint32_t kw[4] = {t.k[i].x, t.k[i].y, t.k[i].z, t.k[i].w};
            const uint32_t vw[4] = {t.v[i].x, t.v[i].y, t.v[i].z, t.v[i].w};
            uint32_t ko[8], vo[8];
#pragma unroll
            for (int w = 0; w < 4; ++w) {
                ko[2 * w] = e4m3x2_to_h2(kw[w]);
                ko[2 * w + 1] = e4m3x2_to_h2(kw[w] >> 16);
                vo[2 * w] = e4m3x2_to_h2(vw[w]);
                vo[2 * w + 1] = e4m3x2_to_h2(vw[w] >> 16);
            }
            *reinterpret_cast<uint4*>(&sk[j][piece * 16]) = make_uint4(ko[0], ko[1], ko[2], ko[3]);
            *reinterpret_cast<uint4*>(&sk[j][piece * 16 + 8]) = make_uint4(ko[4], ko[5], ko[6], ko[7]);
            *reinterpret_cast<uint4*>(&sv[j][piece * 16]) = make_uint4(vo[0], vo[1], vo[2], vo[3]);
            *reinterpret_cast<uint4*>(&sv[j][piece * 16 + 8]) = make_uint4(vo[4], vo[5], vo[6], vo[7]);
            if (piece == 0) {
                sks[j] = t.ks[i];
                svs[j] = t.vs[i];
            }
        }
    }
}

template <int NBUF>
struct alignas(16) TcSmem {
    __half k[NBUF][kTile][kLd];
    __half v[NBUF][kTile][kLd];
    float ks[NBUF][kTile];
    float vs[NBUF][kTile];
};

// Tail of a verify row: the row's ancestors (FP32 K/V of the uncommitted tree nodes), exact, with the sink. One warp per query head, a lane per ancestor.
__device__ __forceinline__ void tail_rows(const float* q, const float* rel_table, const float* sinks, const int32_t* q_pos, const float* tree_k, const float* tree_v,
                                          const uint8_t* tree_anc, const uint8_t* tree_dep, int anc_stride, int chain_k, int flags, int kh, int r0, int nrows, float* part_m, float* part_l,
                                          float* part_o, int slots_per_head, int tail_slot) {
    const int warp = threadIdx.x >> 5, lane = threadIdx.x & 31;
    if (warp >= 3) return;
    const int tp0 = chain_k > 0 ? q_pos[r0] : q_pos[0];   // chain (batched drafting): the group is one slot and its K chain rows; node j of the chain is row r0 + j
    const float scale = 0.08838834764831845f;
    for (int rr = 0; rr < nrows; ++rr) {
        const int m = r0 + rr;
        const int h = 3 * kh + warp;
        const int qp = (flags & kFlagNoBias) ? 0 : q_pos[m];
        const int dep = (flags & kFlagBlockTail) ? (nrows - 1) : (chain_k > 0 ? rr : (int)tree_dep[m]);
        const float* qh = q + ((size_t)m * kInkHeads + h) * kInkHeadDim;
        const float* rt = rel_table + ((size_t)m * kInkHeads + h) * (kInkRelExtent + 1);
        int node = 0;
        float s = kNegInf;
        const bool valid = lane <= dep;
        if (valid) {
            node = chain_k > 0 ? (r0 + lane) : (int)tree_anc[(size_t)m * anc_stride + lane];
            const float4* kp = reinterpret_cast<const float4*>(tree_k + ((size_t)node * kInkKVHeads + kh) * kInkHeadDim);
            const float4* q4 = reinterpret_cast<const float4*>(qh);
            float acc = 0.0f;
#pragma unroll 8
            for (int d4 = 0; d4 < kInkHeadDim / 4; ++d4) {
                const float4 a = __ldg(q4 + d4), b = kp[d4];
                acc = fmaf(a.x, b.x, acc); acc = fmaf(a.y, b.y, acc); acc = fmaf(a.z, b.z, acc); acc = fmaf(a.w, b.w, acc);
            }
            const int dist = min(max(qp - (tp0 + lane), 0), kInkRelExtent);   // (qp = 0 and no bias with kFlagNoBias)
            s = acc * scale + ((flags & kFlagNoBias) ? 0.0f : rt[dist]);
        }
        const float sink = (flags & kFlagNoSink) ? kNegInf : sinks[(size_t)m * kInkHeads + h];
        float mx = fmaxf(s, sink);
#pragma unroll
        for (int o = 16; o; o >>= 1) mx = fmaxf(mx, __shfl_xor_sync(0xffffffffu, mx, o));
        const float p = valid ? __expf(s - mx) : 0.0f;
        float l = p;
#pragma unroll
        for (int o = 16; o; o >>= 1) l += __shfl_xor_sync(0xffffffffu, l, o);
        if (!(flags & kFlagNoSink)) l += __expf(sink - mx);
        float4 acc4 = make_float4(0.f, 0.f, 0.f, 0.f);
        for (int k = 0; k <= dep; ++k) {
            const float pk = __shfl_sync(0xffffffffu, p, k);
            const int nk = __shfl_sync(0xffffffffu, node, k);
            const float4 v4 = *reinterpret_cast<const float4*>(tree_v + ((size_t)nk * kInkKVHeads + kh) * kInkHeadDim + lane * 4);
            acc4.x = fmaf(pk, v4.x, acc4.x); acc4.y = fmaf(pk, v4.y, acc4.y); acc4.z = fmaf(pk, v4.z, acc4.z); acc4.w = fmaf(pk, v4.w, acc4.w);
        }
        const size_t idx = ((size_t)m * kInkHeads + h) * slots_per_head + tail_slot;
        if (lane == 0) { part_m[idx] = mx; part_l[idx] = l; }
        *reinterpret_cast<float4*>(part_o + idx * kInkHeadDim + lane * 4) = acc4;
    }
}

// FMT 0 = FP16 KV, 1 = FP8 E4M3 KV. NBUF = number of K/V tile buffers in shared memory. D = tiles of raw K/V kept in flight in registers (1..3).
// nslab = number of 16-row slabs that compute (warps 0..nslab-1; rows of a group = 16 * nslab / 3); the other warps only load and convert.
// Grid: (8 KV heads, nsp + 1, groups). y == nsp is the verify tail (tree mode only).
template <int FMT, int NBUF, int D>
__global__ void __launch_bounds__(128, 2) k_ink_attn_tc(
    const float* __restrict__ q,          // [M, 24, 128]
    const float* __restrict__ rel_table,  // [M, 24, 1025]
    const float* __restrict__ sinks,      // [M, 24]
    InkKvVisitDesc kv,
    const int32_t* __restrict__ slots,    // mode 1: [M] slot of each row (else null)
    int slot0,
    const int32_t* __restrict__ q_pos,    // [M]
    const float* __restrict__ tree_k,     // verify: [M, 8, 128] FP32 K of the uncommitted tree nodes (null otherwise)
    const float* __restrict__ tree_v,
    const uint8_t* __restrict__ tree_anc, // [M, anc_stride]
    const uint8_t* __restrict__ tree_dep, // [M]
    int anc_stride,
    int chain_k,                          // > 0: batched drafting, groups of chain_k consecutive rows = one slot (anc / dep are null)
    int flags,
    float* __restrict__ part_m,           // [M, 24, nsp + 1]
    float* __restrict__ part_l,
    float* __restrict__ part_o,           // [M, 24, nsp + 1, 128]
    int M, int rows_per_cta, int nslab, int nsp) {
    extern __shared__ __align__(16) unsigned char smem_raw[];
    auto& sm = *reinterpret_cast<TcSmem<NBUF>*>(smem_raw);
    const bool tree = (tree_k != nullptr);
    const int kh = blockIdx.x, sp = blockIdx.y, grp = blockIdx.z;
    const int r0 = grp * rows_per_cta;
    const int nrows = min(rows_per_cta, M - r0);
    if (nrows <= 0) return;
    const int slots_per_head = nsp + 1;
    if (sp == nsp) {   // verify tail
        if (tree) tail_rows(q, rel_table, sinks, q_pos, tree_k, tree_v, tree_anc, tree_dep, anc_stride, chain_k, flags, kh, r0, nrows, part_m, part_l, part_o, slots_per_head, nsp);
        return;
    }
    const int tid = threadIdx.x, warp = tid >> 5, lane = tid & 31;
    const int sl = ((!tree || chain_k > 0) && slots) ? slots[r0] : slot0;
    int nc = ((flags & kFlagBlockTail) || chain_k <= 0) ? (tree ? q_pos[0] : (q_pos[r0] + 1)) : q_pos[r0];
    nc = min(nc, kv.cap_pos);
    const int per = slice_len(nc, nsp);
    const int lo = sp * per;
    const int hi = min(nc, lo + per);

    // the two query rows of this thread (rows lane/4 and lane/4 + 8 of its slab)
    const int qrA = 16 * warp + (lane >> 2), qrB = qrA + 8;
    const bool compute = warp < nslab;
    const bool vA = compute && qrA < 3 * nrows, vB = compute && qrB < 3 * nrows;
    const int mA = r0 + qrA / 3, hA = 3 * kh + qrA % 3;
    const int mB = r0 + qrB / 3, hB = 3 * kh + qrB % 3;
    const size_t idxA = ((size_t)mA * kInkHeads + hA) * slots_per_head + sp;
    const size_t idxB = ((size_t)mB * kInkHeads + hB) * slots_per_head + sp;

    if (lo >= hi) {   // inactive slice: neutral partial
        if (compute && (lane & 3) == 0) {
            if (vA) { part_m[idxA] = -1e30f; part_l[idxA] = 0.0f; }
            if (vB) { part_m[idxB] = -1e30f; part_l[idxB] = 0.0f; }
        }
        return;
    }

    const float scale = 0.08838834764831845f;
    const int c = lane & 3;
    uint32_t qf[8][4] = {};
    int qpA = 0, qpB = 0;
    const float *rtA = nullptr, *rtB = nullptr;
    float mrA = kNegInf, mrB = kNegInf, lrA = 0.0f, lrB = 0.0f;
    if (compute) {
        if (vA) {
            const float* qa = q + ((size_t)mA * kInkHeads + hA) * kInkHeadDim;
#pragma unroll
            for (int d = 0; d < 8; ++d) {
                const float2 f0 = *reinterpret_cast<const float2*>(qa + d * 16 + 2 * c);
                const float2 f1 = *reinterpret_cast<const float2*>(qa + d * 16 + 8 + 2 * c);
                qf[d][0] = pack_h2(f0.x, f0.y);
                qf[d][2] = pack_h2(f1.x, f1.y);
            }
            qpA = (flags & kFlagNoBias) ? 0 : q_pos[mA];
            rtA = rel_table ? rel_table + ((size_t)mA * kInkHeads + hA) * (kInkRelExtent + 1) : nullptr;
            if (!tree && sp == 0 && !(flags & kFlagNoSink)) { mrA = sinks[(size_t)mA * kInkHeads + hA]; lrA = (c == 0) ? 1.0f : 0.0f; }
        }
        if (vB) {
            const float* qb = q + ((size_t)mB * kInkHeads + hB) * kInkHeadDim;
#pragma unroll
            for (int d = 0; d < 8; ++d) {
                const float2 f0 = *reinterpret_cast<const float2*>(qb + d * 16 + 2 * c);
                const float2 f1 = *reinterpret_cast<const float2*>(qb + d * 16 + 8 + 2 * c);
                qf[d][1] = pack_h2(f0.x, f0.y);
                qf[d][3] = pack_h2(f1.x, f1.y);
            }
            qpB = (flags & kFlagNoBias) ? 0 : q_pos[mB];
            rtB = rel_table ? rel_table + ((size_t)mB * kInkHeads + hB) * (kInkRelExtent + 1) : nullptr;
            if (!tree && sp == 0 && !(flags & kFlagNoSink)) { mrB = sinks[(size_t)mB * kInkHeads + hB]; lrB = (c == 0) ? 1.0f : 0.0f; }
        }
    }
    float acc[16][4];
#pragma unroll
    for (int n = 0; n < 16; ++n) { acc[n][0] = acc[n][1] = acc[n][2] = acc[n][3] = 0.0f; }

    int buf = 0;
    // one tile: convert the raw registers into shared memory, barrier, scores / softmax / PV on the slab's warp, then refill the registers D tiles ahead
    auto step = [&](RawTile<FMT>& raw, const int tile) {
        store_tile<FMT>(raw, sm.k[buf], sm.v[buf], sm.ks[buf], sm.vs[buf], tid);
        load_tile<FMT>(raw, kv, sl, kh, tile + D * kTile, hi, tid);
        __syncthreads();
        if (compute) {
            // scores: S[16 rows, 32 keys] = Q K^T (FP32 accumulate)
            float sc[4][4];
#pragma unroll
            for (int n = 0; n < 4; ++n) { sc[n][0] = sc[n][1] = sc[n][2] = sc[n][3] = 0.0f; }
#pragma unroll
            for (int d = 0; d < 8; ++d) {
#pragma unroll
                for (int np = 0; np < 2; ++np) {
                    uint32_t b[4];
                    ldsm_x4(b, &sm.k[buf][8 * (2 * np + (lane >> 4)) + (lane & 7)][d * 16 + ((lane >> 3) & 1) * 8]);
                    mma16816(sc[2 * np], qf[d], b[0], b[1]);
                    mma16816(sc[2 * np + 1], qf[d], b[2], b[3]);
                }
            }
            // scale, K scale, relative bias, mask
            float sA[4][2], sB[4][2];
            const bool no_bias = (flags & kFlagNoBias) != 0;
            const bool near_rows = !no_bias && ((vA && qpA - tile < kInkRelExtent + kTile) || (vB && qpB - tile < kInkRelExtent + kTile));
            const float tailA = (vA && !near_rows && !no_bias) ? rtA[kInkRelExtent] : 0.0f;   // every key of the tile is 1024+ positions away: one bias value
            const float tailB = (vB && !near_rows && !no_bias) ? rtB[kInkRelExtent] : 0.0f;
#pragma unroll
            for (int n = 0; n < 4; ++n) {
#pragma unroll
                for (int e = 0; e < 2; ++e) {
                    const int j = 8 * n + 2 * c + e;
                    const int key = tile + j;
                    float ksc = 1.0f;
                    if constexpr (FMT == 1) ksc = sm.ks[buf][j];
                    float a = sc[n][e] * (ksc * scale), b = sc[n][2 + e] * (ksc * scale);
                    if (key < hi) {
                        if (near_rows) {
                            if (vA) a += rtA[min(max(qpA - key, 0), kInkRelExtent)];
                            if (vB) b += rtB[min(max(qpB - key, 0), kInkRelExtent)];
                        } else { a += tailA; b += tailB; }
                    } else { a = kNegInf; b = kNegInf; }
                    sA[n][e] = vA ? a : kNegInf;
                    sB[n][e] = vB ? b : kNegInf;
                }
            }
            // online softmax: the two rows of the thread, a quad (4 lanes) shares a row
            float mxA = kNegInf, mxB = kNegInf;
#pragma unroll
            for (int n = 0; n < 4; ++n) {
                mxA = fmaxf(mxA, fmaxf(sA[n][0], sA[n][1]));
                mxB = fmaxf(mxB, fmaxf(sB[n][0], sB[n][1]));
            }
            mxA = fmaxf(mxA, __shfl_xor_sync(0xffffffffu, mxA, 1)); mxA = fmaxf(mxA, __shfl_xor_sync(0xffffffffu, mxA, 2));
            mxB = fmaxf(mxB, __shfl_xor_sync(0xffffffffu, mxB, 1)); mxB = fmaxf(mxB, __shfl_xor_sync(0xffffffffu, mxB, 2));
            const float mnA = fmaxf(mrA, mxA), mnB = fmaxf(mrB, mxB);
            const float corA = (mnA == kNegInf) ? 1.0f : __expf(mrA - mnA);
            const float corB = (mnB == kNegInf) ? 1.0f : __expf(mrB - mnB);
            mrA = mnA; mrB = mnB;
            float pA[4][2], pB[4][2], suA = 0.0f, suB = 0.0f;
#pragma unroll
            for (int n = 0; n < 4; ++n) {
#pragma unroll
                for (int e = 0; e < 2; ++e) {
                    pA[n][e] = (sA[n][e] == kNegInf) ? 0.0f : __expf(sA[n][e] - mnA);
                    pB[n][e] = (sB[n][e] == kNegInf) ? 0.0f : __expf(sB[n][e] - mnB);
                    suA += pA[n][e]; suB += pB[n][e];
                }
            }
            lrA = lrA * corA + suA;
            lrB = lrB * corB + suB;
#pragma unroll
            for (int n = 0; n < 16; ++n) {
                acc[n][0] *= corA; acc[n][1] *= corA;
                acc[n][2] *= corB; acc[n][3] *= corB;
            }
            // PV: P (A fragments straight from the score registers, V scale folded in) x V (ldmatrix.trans)
#pragma unroll
            for (int kk = 0; kk < 2; ++kk) {
                float v0 = kPScale, v1 = kPScale, v2 = kPScale, v3 = kPScale;
                if constexpr (FMT == 1) {
                    const float* svs = sm.vs[buf];
                    v0 = svs[16 * kk + 2 * c] * kPScale;     v1 = svs[16 * kk + 2 * c + 1] * kPScale;
                    v2 = svs[16 * kk + 8 + 2 * c] * kPScale; v3 = svs[16 * kk + 8 + 2 * c + 1] * kPScale;
                }
                uint32_t a[4];
                a[0] = pack_h2_sat(pA[2 * kk][0] * v0, pA[2 * kk][1] * v1);
                a[1] = pack_h2_sat(pB[2 * kk][0] * v0, pB[2 * kk][1] * v1);
                a[2] = pack_h2_sat(pA[2 * kk + 1][0] * v2, pA[2 * kk + 1][1] * v3);
                a[3] = pack_h2_sat(pB[2 * kk + 1][0] * v2, pB[2 * kk + 1][1] * v3);
#pragma unroll
                for (int np = 0; np < 8; ++np) {
                    uint32_t b[4];
                    ldsm_x4_t(b, &sm.v[buf][16 * kk + (lane & 15)][np * 16 + (lane >> 4) * 8]);
                    mma16816(acc[2 * np], a, b[0], b[1]);
                    mma16816(acc[2 * np + 1], a, b[2], b[3]);
                }
            }
        }
        if constexpr (NBUF == 1) __syncthreads();
        if constexpr (NBUF == 2) buf ^= 1;
    };

    RawTile<FMT> r0t, r1t, r2t;
    load_tile<FMT>(r0t, kv, sl, kh, lo, hi, tid);
    if constexpr (D >= 2) load_tile<FMT>(r1t, kv, sl, kh, lo + kTile, hi, tid);
    if constexpr (D >= 3) load_tile<FMT>(r2t, kv, sl, kh, lo + 2 * kTile, hi, tid);
    for (int tile = lo; tile < hi; tile += D * kTile) {
        step(r0t, tile);
        if constexpr (D >= 2) { if (tile + kTile < hi) step(r1t, tile + kTile); }
        if constexpr (D >= 3) { if (tile + 2 * kTile < hi) step(r2t, tile + 2 * kTile); }
    }

    if (compute) {
        // the denominator: a row's partial sums live in the 4 lanes of its quad
        lrA += __shfl_xor_sync(0xffffffffu, lrA, 1); lrA += __shfl_xor_sync(0xffffffffu, lrA, 2);
        lrB += __shfl_xor_sync(0xffffffffu, lrB, 1); lrB += __shfl_xor_sync(0xffffffffu, lrB, 2);
        const float inv_ps = 1.0f / kPScale;
        if (vA) {
            if (c == 0) { part_m[idxA] = mrA; part_l[idxA] = lrA; }
            float* o = part_o + idxA * kInkHeadDim;
#pragma unroll
            for (int n = 0; n < 16; ++n) *reinterpret_cast<float2*>(o + 8 * n + 2 * c) = make_float2(acc[n][0] * inv_ps, acc[n][1] * inv_ps);
        }
        if (vB) {
            if (c == 0) { part_m[idxB] = mrB; part_l[idxB] = lrB; }
            float* o = part_o + idxB * kInkHeadDim;
#pragma unroll
            for (int n = 0; n < 16; ++n) *reinterpret_cast<float2*>(o + 8 * n + 2 * c) = make_float2(acc[n][2] * inv_ps, acc[n][3] * inv_ps);
        }
    }
}

// Merge of the slices a block actually wrote (the count follows from the row's key count, so a short context reads 1-2 slots, not 65) + sigmoid(gate),
// optionally also as __half2 for the output projection (like k_ink_flash_gqa_merge_h2). Grid (24 heads, M), 128 threads (a thread per channel).
template <bool H2>
__global__ void __launch_bounds__(128) k_ink_tc_merge(
    const float* __restrict__ part_m, const float* __restrict__ part_l, const float* __restrict__ part_o, const float* __restrict__ gate_attn,
    float* __restrict__ out, __half2* __restrict__ out_h2, const int32_t* __restrict__ q_pos, bool tree, int chain_k, int flags, int cap_pos, int M, int nsp) {
    const int h = blockIdx.x, m = blockIdx.y, d = threadIdx.x;
    if (h >= kInkHeads || m >= M) return;
    int nc = ((flags & kFlagBlockTail) || chain_k <= 0) ? (tree ? q_pos[0] : (q_pos[m] + 1)) : q_pos[m - m % chain_k];
    nc = min(nc, cap_pos);
    const int per = slice_len(nc, nsp);
    const int n_act = (nc + per - 1) / per;
    const int count = n_act + (tree ? 1 : 0);
    const int slots = nsp + 1;
    const size_t base = ((size_t)m * kInkHeads + h) * slots;
    __shared__ float s_w[kMaxSplits + 1], s_l[kMaxSplits + 1];
    __shared__ int s_slot[kMaxSplits + 1];
    if (d < count) {
        const int slot = (d < n_act) ? d : nsp;
        s_slot[d] = slot;
        s_w[d] = part_m[base + slot];
        s_l[d] = part_l[base + slot];
    }
    __syncthreads();
    float mg = -1e30f;
    for (int t = 0; t < count; ++t) mg = fmaxf(mg, s_l[t] > 0.0f ? s_w[t] : -1e30f);
    float ls = 0.0f, acc = 0.0f;
    int t = 0;
    for (; t + 4 <= count; t += 4) {
        float w[4], o[4];
#pragma unroll
        for (int u = 0; u < 4; ++u) {
            w[u] = s_l[t + u] > 0.0f ? __expf(s_w[t + u] - mg) : 0.0f;
            o[u] = part_o[(base + s_slot[t + u]) * kInkHeadDim + d];
        }
#pragma unroll
        for (int u = 0; u < 4; ++u) { ls += s_l[t + u] * w[u]; acc = fmaf(o[u], w[u], acc); }
    }
    for (; t < count; ++t) {
        const float w = s_l[t] > 0.0f ? __expf(s_w[t] - mg) : 0.0f;
        ls += s_l[t] * w;
        acc = fmaf(part_o[(base + s_slot[t]) * kInkHeadDim + d], w, acc);
    }
    float res = (ls > 0.0f) ? (acc / ls) : 0.0f;
    const size_t ob = ((size_t)m * kInkHeads + h) * kInkHeadDim + d;
    if (gate_attn) res *= 1.0f / (1.0f + __expf(-gate_attn[ob]));
    out[ob] = res;
    if constexpr (H2) {
        const float odd = __shfl_xor_sync(0xFFFFFFFFu, res, 1);
        if ((d & 1) == 0) out_h2[ob >> 1] = __floats2half2_rn(fminf(fmaxf(res, -65504.0f), 65504.0f), fminf(fmaxf(odd, -65504.0f), 65504.0f));
    }
}


// Runtime override (the KL harness of f-perflong-kl sets it): -1 = the part switch decides, 0 = old kernels, 1 = this kernel (when its buffers exist).
inline int& force_ref() { static int v = -1; return v; }

// the kernels of this file run only on sm_80 and newer (mma.m16n8k16)
inline bool tc_attn_dev_ok() {
    static int ok[16] = {};   // 0 = unknown, 1 = yes, 2 = no
    int dev = 0;
    cudaGetDevice(&dev);
    if (dev < 0 || dev >= 16) return false;
    if (ok[dev] == 0) {
        int major = 0;
        ok[dev] = (cudaDeviceGetAttribute(&major, cudaDevAttrComputeCapabilityMajor, dev) == cudaSuccess && major >= 8) ? 1 : 2;
    }
    return ok[dev] == 1;
}

// Slices per KV head for the card in use: 8 heads x nsp blocks = two full waves of the resident blocks (2 per SM), so no wave is left part empty
// (5090, 170 SMs: 85 slices = 680 blocks; 5070, 48 SMs: 24). STRATA_INK_TC_ATTN_SPLITS=<n> (16..96) overrides it for the A/B runs.
inline int splits_dev(int dev) {
    static int per_dev[16] = {};
    if (dev < 0 || dev >= 16) dev = 0;
    if (per_dev[dev] == 0) {
        int sms = 0;
        cudaDeviceGetAttribute(&sms, cudaDevAttrMultiProcessorCount, dev);
        int n = (sms * 2 * 2) / kInkKVHeads;
        if (const char* e = std::getenv("STRATA_INK_TC_ATTN_SPLITS")) n = std::atoi(e);
        per_dev[dev] = n < 16 ? 16 : (n > kMaxSplits ? kMaxSplits : n);
    }
    return per_dev[dev];
}
inline int splits_cur() {
    int dev = 0;
    cudaGetDevice(&dev);
    return splits_dev(dev);
}

// K/V tile buffers in shared memory: 2 = one barrier per tile (35 KB per block, 2 blocks per SM), 1 = two barriers (18 KB). D = tiles of raw K/V in flight per
// thread (registers). STRATA_INK_TC_ATTN_NBUF=1|2 and STRATA_INK_TC_ATTN_DEPTH=1|2|3 pick them for the A/B runs.
inline int nbuf_cfg() {
    static const int v = [] {
        const char* e = std::getenv("STRATA_INK_TC_ATTN_NBUF");
        return (e && e[0] == '1') ? 1 : 2;
    }();
    return v;
}
inline int depth_cfg() {
    static const int v = [] {
        const char* e = std::getenv("STRATA_INK_TC_ATTN_DEPTH");
        return (e && e[0] >= '1' && e[0] <= '3') ? (e[0] - '0') : 2;
    }();
    return v;
}

template <int FMT, int NBUF, int D>
inline void launch_one(cudaStream_t st, const float* q, const float* rel, const float* sinks, const InkKvVisitDesc& kv, const int32_t* slots, int slot0, const int32_t* q_pos,
                       const float* tk, const float* tv, const uint8_t* anc, const uint8_t* dep, int anc_stride, int chain_k, int flags, float* pm, float* pl, float* po, int M,
                       int rows_per_cta, int nslab) {
    constexpr size_t kSmem = sizeof(TcSmem<NBUF>);
    static bool attr = false;
    if (!attr) {
        cudaFuncSetAttribute(k_ink_attn_tc<FMT, NBUF, D>, cudaFuncAttributeMaxDynamicSharedMemorySize, (int)kSmem);
        attr = true;
    }
    const int nsp = splits_cur();
    const dim3 grid(kInkKVHeads, nsp + 1, (M + rows_per_cta - 1) / rows_per_cta);
    k_ink_attn_tc<FMT, NBUF, D><<<grid, 128, kSmem, st>>>(q, rel, sinks, kv, slots, slot0, q_pos, tk, tv, anc, dep, anc_stride, chain_k, flags, pm, pl, po, M, rows_per_cta, nslab, nsp);
}

// Returns false when the kernel does not apply (the caller keeps the old one). tree = verify step (tk / tv / anc / dep set); else a plain step / batched rows.
inline bool launch(cudaStream_t st, const float* q, const float* rel, const float* sinks, const InkKvVisitDesc& kv, const int32_t* slots, int slot0, const int32_t* q_pos,
                   const float* tk, const float* tv, const uint8_t* anc, const uint8_t* dep, int anc_stride, float* pm, float* pl, float* po, int M, int chain_k = 0, int flags = 0,
                   int nbuf = nbuf_cfg(), int depth = depth_cfg()) {
    if (kv.is_swa || (kv.kv_fmt != 0 && kv.kv_fmt != 1) || M < 1) return false;
    if (!tc_attn_dev_ok()) return false;   // sm_80+ only (mma.m16n8k16): an older card keeps the old kernels even if the part is asked for by name
    const bool tree = (tk != nullptr);
    int nslab = 1, rows = 1;
    if (chain_k > 0) {   // batched drafting: M = slots x chain_k rows, one group per slot
        if (!tree || chain_k > 16 || M % chain_k != 0) return false;
        nslab = (3 * chain_k + 15) / 16;
        rows = chain_k;
    } else if (tree) {
        if (M <= 5) { nslab = 1; rows = 5; }
        else if (M <= 10) { nslab = 2; rows = 10; }
        else { nslab = 3; rows = 16; }
    }
#define INK_TC_CASE(F, NB, DD)                                                                                                          \
    if (kv.kv_fmt == F && nbuf == NB && depth == DD) {                                                                                  \
        launch_one<F, NB, DD>(st, q, rel, sinks, kv, slots, slot0, q_pos, tk, tv, anc, dep, anc_stride, chain_k, flags, pm, pl, po, M, rows, nslab);    \
        return true;                                                                                                                    \
    }
    INK_TC_CASE(1, 2, 1) INK_TC_CASE(1, 2, 2) INK_TC_CASE(1, 2, 3) INK_TC_CASE(1, 1, 1) INK_TC_CASE(1, 1, 2) INK_TC_CASE(1, 1, 3)
    INK_TC_CASE(0, 2, 1) INK_TC_CASE(0, 2, 2) INK_TC_CASE(0, 2, 3) INK_TC_CASE(0, 1, 1) INK_TC_CASE(0, 1, 2) INK_TC_CASE(0, 1, 3)
#undef INK_TC_CASE
    return false;
}

// The merge that goes with launch(): tree = verify step (the same flag as tk != nullptr), cap_pos = kv.cap_pos.
inline void merge(cudaStream_t st, const float* pm, const float* pl, const float* po, const float* gate, float* out, __half2* out_h2, const int32_t* q_pos, bool tree,
                  int cap_pos, int M, int chain_k = 0, int flags = 0) {
    const dim3 grid(kInkHeads, M);
    if (out_h2) k_ink_tc_merge<true><<<grid, 128, 0, st>>>(pm, pl, po, gate, out, out_h2, q_pos, tree, chain_k, flags, cap_pos, M, splits_cur());
    else        k_ink_tc_merge<false><<<grid, 128, 0, st>>>(pm, pl, po, gate, out, nullptr, q_pos, tree, chain_k, flags, cap_pos, M, splits_cur());
}

}  // namespace tcattn
}  // namespace strata::ink
