// Vendored from mx-llama.cpp (github.com/mxxm-t/mx-llama.cpp, MIT) ggml/src/ggml-cuda/q8_repack/repack-common.cuh at c4d12e2c;
// Changes: the include paths (Strata compiles against ggml-cuda of llama.cpp 3cf0325), and
// repack_quantize_src1_q8_1 (graph glue) compiled out.
// Shared layout helpers and device-side data types for the Q8_0 repacked-weight path.
#pragma once

#include "common.cuh"
#include "mmq.cuh"
#include "quantize.cuh"
#include "vecdotq.cuh"   // get_int_from_table_16, for the MXFP4 nibble expand

#include <cstddef>

#define MMQ_RP_Q8_BK 4
#define MMQ_RP_Q8_TN 2
#define MMQ_RP_Q8_BM 64
#define MMQ_RP_Q8_NROW_LANES 4
// Widths at or below this take one multi-column mat-vec pass instead of the
// tiled GEMM. Measured on Qwen3.8-27B pp512, 4x MI50 -sm tensor: the mat-vec
// leads from 2 tokens (77.2 against the tile's 14.3) through 8 (159.6 against
// 98.9), and the tile takes back over by 12, so the crossover is 8 to 12.
#define MMQ_RP_Q8_MMV_MAX_TOKENS 8
// MoE narrow batch: at or below this many tokens, run one mat-vec per
// assignment instead of the tiled GEMM. Expert token counts are per expert, so
// at these widths nearly every active expert holds a single assignment.
#define MMQ_RP_Q8_MOE_MMV_MAX_TOKENS 8

// qs plane row stride in BYTES: ne0 (one byte per quant), bumped by 16 whenever
// it lands on a multiple of 128 so row starts do not alias HBM channels. 16
// keeps uint4 alignment and costs half of the padding sub-block it replaces,
// which also only fired when n_sub was a power of two and so missed shapes like
// ne0=5120. Scales keep their own plane: the narrow mat-vec reads several rows
// per lane and wants them adjacent.
template <typename T>
static __host__ __device__ inline T repack_qs_stride(const T ne0) {
    T rs = ne0;
    if (rs % 128 == 0) {
        rs += 16;
    }
    return rs;
}

// Per-type qs row stride and scale row bytes: Q8_0 rows carry ne0 payload
// bytes and 2 B f16 scales per sub-block, MXFP4 rows carry packed nibbles
// (ne0/2 B) and 1 B e8m0 scales per sub-block. Both share the de-alias bump.
template <typename T>
static __host__ __device__ inline T repack_qs_row_stride(const ggml_type type, const T ne0) {
    return repack_qs_stride((type == GGML_TYPE_MXFP4 || type == GGML_TYPE_IQ4_NL || type == GGML_TYPE_Q6_K || type == GGML_TYPE_Q4_K || type == GGML_TYPE_Q5_K || type == GGML_TYPE_Q5_1) ? ne0 / 2 : ne0);
}
template <typename T>
static __host__ __device__ inline T repack_scale_row_bytes(const ggml_type type, const T ne0) {
    return (ne0 / 32) * (type == GGML_TYPE_MXFP4 ? 1 : 2);
}

static inline size_t repack_gcn_nbytes(const ggml_type type, const int64_t ne0, const int64_t ne1) {
    GGML_ASSERT(ne0 % 32 == 0);
    switch (type) {
        case GGML_TYPE_Q8_0:
            return (size_t) ne1 * ((size_t) repack_qs_stride(ne0) + (size_t)(ne0 / 32) * 2);
        // nibble plane row [ne0/2 B, de-aliased like the Q8_0 qs rows] + 1-byte
        // e8m0 plane [ne0/32]/row. 17 B/block = canonical size, so the repack
        // stays VRAM-neutral (a 2-byte scale slot cost +5.9% and OOMed
        // gpt-oss-120b at 2-GPU full offload). The GEMM's LDS scale array stays
        // uint16_t; only the GLOBAL load narrows.
        case GGML_TYPE_MXFP4:
            return (size_t) ne1 * ((size_t) repack_qs_stride(ne0 / 2) + (size_t)(ne0 / 32));
        // IQ4_NL: the MXFP4 nibble plane with a 2-byte f16 scale plane, 18 B/block = canonical size, VRAM-neutral like the other two.
        case GGML_TYPE_IQ4_NL:
            return (size_t) ne1 * ((size_t) repack_qs_stride(ne0 / 2) + (size_t)(ne0 / 32) * 2);
        // Q6_K: de-aliased low-nibble rows, then a highs plane [8 B/32], an int8 scale-pair plane [2 B/32] and an f16 d plane [2 B/256]: canonical 6.5625 bpw.
        case GGML_TYPE_Q6_K:
            return (size_t) ne1 * ((size_t) repack_qs_stride(ne0 / 2) + (size_t)(ne0 / 32) * 10 + (size_t)((ne0 / 32 + 7) / 8) * 2);
        // Q4_K: de-aliased nibble rows, then the canonical 12-byte scale/min record per super-block and a half2 {d, dmin} plane: canonical 4.5 bpw.
        case GGML_TYPE_Q4_K:
            return (size_t) ne1 * ((size_t) repack_qs_stride(ne0 / 2) + (size_t)((ne0 / 32 + 7) / 8) * 16);
        // Q5_K: Q4_K's planes plus a 4-byte fifth-bit plane per sub-block: 5.5 bpw.
        case GGML_TYPE_Q5_K:
            return (size_t) ne1 * ((size_t) repack_qs_stride(ne0 / 2) + (size_t)(ne0 / 32) * 4 + (size_t)((ne0 / 32 + 7) / 8) * 16);
        // Q5_1: de-aliased nibble rows, a 4-byte fifth-bit plane and a half2 {d, m} plane per 32-value block: canonical 6 bpw.
        case GGML_TYPE_Q5_1:
            return (size_t) ne1 * ((size_t) repack_qs_stride(ne0 / 2) + (size_t)(ne0 / 32) * 8);
        default:             GGML_ABORT("unsupported repack type");
    }
}

// Bytes per sub-block in the qs plane, and uint4 loads needed to fetch one.
static __host__ __device__ inline int repack_qs_bytes(const ggml_type type) {
    return (type == GGML_TYPE_MXFP4 || type == GGML_TYPE_IQ4_NL || type == GGML_TYPE_Q6_K || type == GGML_TYPE_Q4_K || type == GGML_TYPE_Q5_K || type == GGML_TYPE_Q5_1) ? 16 : 32;
}

#if defined(GGML_USE_HIP) && defined(__gfx906__)
// MXFP4 sub-block: 16 payload bytes coding 32 values as nibbles.
// get_int_from_table_16 returns v.x for the four LOW nibbles and v.y for the four
// HIGH nibbles, and stock load_tiles_mxfp4 stores them at k0 and k0 + QI_MXFP4 -
// i.e. the high nibbles of int32 slot j are values 16+4j..19+4j, NOT 4j+4...
// So all four v.x concatenate into values 0..15 and all four v.y into 16..31,
// which is exactly the lo/hi split the Q8_0 path already feeds to dp4a.
// Swapping these compiles, runs, and produces plausible-looking wrong output.
static __device__ __forceinline__ void rp_mxfp4_expand(
        const uint4 nib, uint4 & lo, uint4 & hi) {
    const int2 v0 = get_int_from_table_16((int) nib.x, kvalues_mxfp4);
    const int2 v1 = get_int_from_table_16((int) nib.y, kvalues_mxfp4);
    const int2 v2 = get_int_from_table_16((int) nib.z, kvalues_mxfp4);
    const int2 v3 = get_int_from_table_16((int) nib.w, kvalues_mxfp4);
    lo = make_uint4((uint32_t) v0.x, (uint32_t) v1.x, (uint32_t) v2.x, (uint32_t) v3.x);
    hi = make_uint4((uint32_t) v0.y, (uint32_t) v1.y, (uint32_t) v2.y, (uint32_t) v3.y);
}

// IQ4_NL sub-block: the same nibble layout and the same lo/hi split, looked up in kvalues_iq4nl instead of kvalues_mxfp4.
static __device__ __forceinline__ void rp_iq4nl_expand(
        const uint4 nib, uint4 & lo, uint4 & hi) {
    const int2 v0 = get_int_from_table_16((int) nib.x, kvalues_iq4nl);
    const int2 v1 = get_int_from_table_16((int) nib.y, kvalues_iq4nl);
    const int2 v2 = get_int_from_table_16((int) nib.z, kvalues_iq4nl);
    const int2 v3 = get_int_from_table_16((int) nib.w, kvalues_iq4nl);
    lo = make_uint4((uint32_t) v0.x, (uint32_t) v1.x, (uint32_t) v2.x, (uint32_t) v3.x);
    hi = make_uint4((uint32_t) v0.y, (uint32_t) v1.y, (uint32_t) v2.y, (uint32_t) v3.y);
}

// Spread one byte's four 2-bit fields into the low 2 bits of each byte lane.
static __device__ __forceinline__ uint32_t rp_spread2(const uint32_t h) {
    return (h | (h << 6) | (h << 12) | (h << 18)) & 0x03030303u;
}
// Spread one nibble's four bits into bit 0 of each byte lane.
static __device__ __forceinline__ uint32_t rp_spread1(const uint32_t b) {
    return (b | (b << 7) | (b << 14) | (b << 21)) & 0x01010101u;
}

// Stock MMQ folds a 0.5f into the MXFP4 tile scale
// (x_df = ggml_cuda_e8m0_to_fp32(e)*0.5f). Reproduce it exactly or the result is
// off by a factor of two per block rather than merely imprecise.
template <ggml_type WT>
static __device__ __forceinline__ float rp_scale_from_slot(const uint16_t s) {
    if constexpr (WT == GGML_TYPE_MXFP4) {
        return ggml_cuda_e8m0_to_fp32((uint8_t) s) * 0.5f;
    } else {
        return __half2float(*reinterpret_cast<const __half *>(&s));
    }
}
#endif

template <int CW>
static __device__ __forceinline__ int sX_swizzle(int lr) {
    if constexpr (CW == 64) {
        const int n  = lr >> 6;
        int       tx = lr & 63;
        tx ^= (tx >> 5) << 4;
        return (n << 6) | tx;
    } else {
        const int n  = lr >> 5;
        int       tx = lr & 31;
        tx ^= (tx >> 4) << 3;
        return (n << 5) | tx;
    }
}

struct rp_x_sub {
    uint4 q0, q1;
    float d;
};

struct block_q8_1_mmq_h {
    float  d4[4];
    int8_t qs[QK8_1_MMQ];
};

static_assert(sizeof(block_q8_1_mmq_h) == sizeof(block_q8_1_mmq),
              "Unexpected block_q8_1_mmq_h size");
static_assert(offsetof(block_q8_1_mmq_h, d4) == offsetof(block_q8_1_mmq, d4),
              "block_q8_1_mmq_h d4 offset mismatch");
static_assert(offsetof(block_q8_1_mmq_h, qs) == offsetof(block_q8_1_mmq, qs),
              "block_q8_1_mmq_h qs offset mismatch");

// DS4 view of the same 16-byte header: affine quants need the activation SUM per 32-block for their -dmin*m term, and quantize_mmq_q8_1_cuda emits half2 {d, s} slots for them (MMQ_Q8_1_DS_LAYOUT_DS4, keyed off the src0 type it is passed).
struct block_q8_1_mmq_ds_h {
    half2  ds4[4];
    int8_t qs[QK8_1_MMQ];
};

static_assert(sizeof(block_q8_1_mmq_ds_h) == sizeof(block_q8_1_mmq),
              "Unexpected block_q8_1_mmq_ds_h size");
static_assert(offsetof(block_q8_1_mmq_ds_h, qs) == offsetof(block_q8_1_mmq, qs),
              "block_q8_1_mmq_ds_h qs offset mismatch");

struct sXq_row_q8 {
    uint4 q[MMQ_RP_Q8_BK][2];
    uint4 pad;
};

static_assert(sizeof(sXq_row_q8) == (2 * MMQ_RP_Q8_BK + 1) * 16,
              "unexpected sXq row size");

static __device__ __forceinline__ uint4 rp_ldcs_u4(const uint4 * __restrict__ p) {
    return *p;
}

#if defined(GGML_USE_HIP) && defined(__gfx906__)

#define RP_DPP_ADD(name, nop, dpp_ctrl)                                      \
    static __device__ __forceinline__ float name(const float x) {            \
        float r;                                                             \
        asm volatile(                                                        \
            nop                                                              \
            "v_add_f32_dpp %0, %1, %1 " dpp_ctrl " row_mask:0xf bank_mask:0xf" \
            : "=v"(r) : "v"(x) : "memory");                                  \
        return r;                                                            \
    }

RP_DPP_ADD(rp_dpp_add_xor1, "s_nop 4\n", "quad_perm:[1,0,3,2]")
RP_DPP_ADD(rp_dpp_add_xor2, "s_nop 1\n", "quad_perm:[2,3,0,1]")
RP_DPP_ADD(rp_dpp_add_xor8, "s_nop 1\n", "row_ror:8")

#undef RP_DPP_ADD

static __device__ __forceinline__ float rp_dpp_xfer_xor4(const float x) {
    int d;
    asm volatile("v_mov_b32 %0, %1\n"
                 "s_nop 1\n"
                 "v_mov_b32_dpp %0, %1 row_shl:4 row_mask:0xf bank_mask:0x5\n"
                 "v_mov_b32_dpp %0, %1 row_shr:4 row_mask:0xf bank_mask:0xa\n"
                 : "=v"(d) : "v"(__float_as_int(x)) : "memory");
    return __int_as_float(d);
}

static __device__ __forceinline__ float rp_dpp_xfer_xor16(const float x) {
    int d;
    asm volatile("ds_swizzle_b32 %0, %1 offset:swizzle(SWAP,16)\n"
                 "s_waitcnt lgkmcnt(0)\n"
                 : "=v"(d) : "v"(__float_as_int(x)) : "memory");
    return __int_as_float(d);
}

template <int width>
static __device__ __forceinline__ float rp_warp_reduce_sum(const float x) {
    static_assert(width >= 1 && (width & (width - 1)) == 0);
    float r = x;
    if constexpr (width >=  2) { r = rp_dpp_add_xor1(r); }
    if constexpr (width >=  4) { r = rp_dpp_add_xor2(r); }
    if constexpr (width >=  8) { r += rp_dpp_xfer_xor4(r); }
    if constexpr (width >= 16) { r = rp_dpp_add_xor8(r); }
    if constexpr (width >= 32) { r += rp_dpp_xfer_xor16(r); }
    if constexpr (width >= 64) { r += __shfl_xor_sync(0xffffffff, r, 32, 64); }
    return r;
}

#else

template <int width>
static __device__ __forceinline__ float rp_warp_reduce_sum(const float x) {
    return warp_reduce_sum<width>(x);
}

#endif

__device__ __forceinline__ rp_x_sub rp_x_sub_from_mmq_group(
        const block_q8_1_mmq_h * __restrict__ group, const uint32_t col, const uint32_t lk) {
    const block_q8_1_mmq_h & m = group[col];
    const uint4           * mq = reinterpret_cast<const uint4 *>(m.qs + lk * QK8_1);
    rp_x_sub out;
    out.q0 = mq[0];
    out.q1 = mq[1];
    out.d  = m.d4[lk];
    return out;
}

// DS4 variant: same qs fetch, scale AND sum decoded from the half2 slot.
__device__ __forceinline__ rp_x_sub rp_x_sub_from_mmq_group_ds(
        const block_q8_1_mmq_h * __restrict__ group, const uint32_t col, const uint32_t lk,
        float & s) {
    const block_q8_1_mmq_ds_h & m = *reinterpret_cast<const block_q8_1_mmq_ds_h *>(group + col);
    const uint4               * mq = reinterpret_cast<const uint4 *>(m.qs + lk * QK8_1);
    rp_x_sub out;
    out.q0 = mq[0];
    out.q1 = mq[1];
    const float2 ds = __half22float2(m.ds4[lk]);
    out.d = ds.x;
    s     = ds.y;
    return out;
}

// Zero a raw scale slot for a dead row (the nc kernels clamp the row index and kill the term through the slot).
// Integral slots and the uint2 affine slot.
template <typename T>
__device__ __forceinline__ T rp_slot_keep(const T v, const bool keep) {
    return keep ? v : (T) 0;
}
template <>
__device__ __forceinline__ uint2 rp_slot_keep<uint2>(const uint2 v, const bool keep) {
    return keep ? v : make_uint2(0u, 0u);
}

// ---- per-type traits ---------------------------------------------------------
// Everything the shared kernels need to know about a repacked quant type.
// Adding a type = one specialization here + one line in RP_FOREACH_TYPE below
// (plus host/device extract functions until those join the traits too).
//   geom    - per-tensor constants, built once per kernel launch
//   load_w  - fetch sub-block (wrow, sb): 32 int8 values as a lo/hi uint4 pair
//             plus the raw scale slot. THE hot path.
//   scale   - decode a raw scale slot to float.
template <ggml_type T> struct rp_traits;

template <> struct rp_traits<GGML_TYPE_Q8_0> {
    using slot_t = uint16_t;   // f16 scale
    static constexpr bool split_scales = false;
    static constexpr bool affine       = false;
    static constexpr bool raw_lds = false;
    // de-aliased qs rows [ne1 x rs], then an f16 scale plane [ne1 x n_sub].
    struct geom {
        uint32_t rs, rs_u4, n_sub;
        size_t   dplane_off;
        __host__ __device__ geom(uint32_t ne0, uint32_t ne1)
            : rs(repack_qs_row_stride(GGML_TYPE_Q8_0, ne0)), rs_u4(rs >> 4),
              n_sub(ne0 >> 5), dplane_off((size_t) ne1 * rs) {}
    };
#if defined(GGML_USE_HIP) && defined(__gfx906__)
    static __device__ __forceinline__ void load_w(const uint8_t * __restrict__ wbase,
            const geom & g, uint32_t wrow, uint32_t sb, uint4 & lo, uint4 & hi, uint16_t & d) {
        const uint4 * qsp = reinterpret_cast<const uint4 *>(wbase);
        const size_t  q4  = (size_t) wrow * g.rs_u4 + 2 * sb;
        lo = rp_ldcs_u4(qsp + q4);
        hi = rp_ldcs_u4(qsp + q4 + 1);
        d  = reinterpret_cast<const uint16_t *>(wbase + g.dplane_off)[(size_t) wrow * g.n_sub + sb];
    }
    static __device__ __forceinline__ float scale(const uint16_t s) {
        return __half2float(*reinterpret_cast<const __half *>(&s));
    }
#endif
};

template <> struct rp_traits<GGML_TYPE_MXFP4> {
    using slot_t = uint16_t;   // e8m0 byte, widened on load
    static constexpr bool split_scales = false;
    static constexpr bool affine       = false;
    // de-aliased nibble rows [ne1 x rs], then a 1-byte e8m0 plane [ne1 x n_sub].
    struct geom {
        uint32_t rs, rs_u4, n_sub;
        size_t   dplane_off;
        __host__ __device__ geom(uint32_t ne0, uint32_t ne1)
            : rs(repack_qs_row_stride(GGML_TYPE_MXFP4, ne0)), rs_u4(rs >> 4),
              n_sub(ne0 >> 5), dplane_off((size_t) ne1 * rs) {}
    };
    // The GEMM stages RAW nibbles (one uint4 per sub-block) and expands at
    // consume: half the LDS and half the prefetch registers of the expanded
    // form, which measured 16-22 spilled VGPRs per lane when staged expanded.
    static constexpr bool raw_lds = true;
#if defined(GGML_USE_HIP) && defined(__gfx906__)
    static __device__ __forceinline__ void load_w(const uint8_t * __restrict__ wbase,
            const geom & g, uint32_t wrow, uint32_t sb, uint4 & lo, uint4 & hi, uint16_t & d) {
        const uint4 * qsp = reinterpret_cast<const uint4 *>(wbase);
        rp_mxfp4_expand(rp_ldcs_u4(qsp + (size_t) wrow * g.rs_u4 + sb), lo, hi);
        d = wbase[g.dplane_off + (size_t) wrow * g.n_sub + sb];
    }
    static __device__ __forceinline__ void load_w_raw(const uint8_t * __restrict__ wbase,
            const geom & g, uint32_t wrow, uint32_t sb, uint4 & raw, uint16_t & d) {
        const uint4 * qsp = reinterpret_cast<const uint4 *>(wbase);
        raw = rp_ldcs_u4(qsp + (size_t) wrow * g.rs_u4 + sb);
        d   = wbase[g.dplane_off + (size_t) wrow * g.n_sub + sb];
    }
    static __device__ __forceinline__ float scale(const uint16_t s) {
        return ggml_cuda_e8m0_to_fp32((uint8_t) s) * 0.5f;
    }
    static __device__ __forceinline__ void expand_raw(const uint4 raw, uint4 & lo, uint4 & hi) {
        rp_mxfp4_expand(raw, lo, hi);
    }
#endif
};

template <> struct rp_traits<GGML_TYPE_IQ4_NL> {
    using slot_t = uint16_t;   // f16 scale
    static constexpr bool split_scales = false;
    static constexpr bool affine       = false;
    // MXFP4's twin: de-aliased nibble rows [ne1 x rs], then an f16 scale plane [ne1 x n_sub], values from kvalues_iq4nl, per-32 scaling with no 0.5 factor, so every kernel takes the Q8_0 epilogue.
    // Raw nibbles are staged like MXFP4.
    struct geom {
        uint32_t rs, rs_u4, n_sub;
        size_t   dplane_off;
        __host__ __device__ geom(uint32_t ne0, uint32_t ne1)
            : rs(repack_qs_row_stride(GGML_TYPE_IQ4_NL, ne0)), rs_u4(rs >> 4),
              n_sub(ne0 >> 5), dplane_off((size_t) ne1 * rs) {}
    };
    static constexpr bool raw_lds = true;
#if defined(GGML_USE_HIP) && defined(__gfx906__)
    static __device__ __forceinline__ void load_w(const uint8_t * __restrict__ wbase,
            const geom & g, uint32_t wrow, uint32_t sb, uint4 & lo, uint4 & hi, uint16_t & d) {
        const uint4 * qsp = reinterpret_cast<const uint4 *>(wbase);
        rp_iq4nl_expand(rp_ldcs_u4(qsp + (size_t) wrow * g.rs_u4 + sb), lo, hi);
        d = reinterpret_cast<const uint16_t *>(wbase + g.dplane_off)[(size_t) wrow * g.n_sub + sb];
    }
    static __device__ __forceinline__ void load_w_raw(const uint8_t * __restrict__ wbase,
            const geom & g, uint32_t wrow, uint32_t sb, uint4 & raw, uint16_t & d) {
        const uint4 * qsp = reinterpret_cast<const uint4 *>(wbase);
        raw = rp_ldcs_u4(qsp + (size_t) wrow * g.rs_u4 + sb);
        d   = reinterpret_cast<const uint16_t *>(wbase + g.dplane_off)[(size_t) wrow * g.n_sub + sb];
    }
    static __device__ __forceinline__ float scale(const uint16_t s) {
        return __half2float(*reinterpret_cast<const __half *>(&s));
    }
    static __device__ __forceinline__ void expand_raw(const uint4 raw, uint4 & lo, uint4 & hi) {
        rp_iq4nl_expand(raw, lo, hi);
    }
#endif
};

template <> struct rp_traits<GGML_TYPE_Q6_K> {
    // Four planes: de-aliased packed lows [16 B/32], packed 2-bit highs [8 B/32], int8 scale pairs [2 B/32] (Q6_K scales are per 16 values) and f16 d per 256.
    // Values are pre-shifted, ((q6 - 32) << 2) as int8 (a shift and an XOR 0x80, gfx906 has no byte subtract), and the exact 0.25 is folded into the scale.
    // The per-16 scales ride the lo/hi dp4a split: split_scales.
    using slot_t = uint32_t;   // sc_lo int8 | sc_hi int8 << 8 | d f16 << 16
    static constexpr bool split_scales = true;
    static constexpr bool affine       = false;
    static constexpr bool raw_lds      = false;
    struct geom {
        uint32_t rs, rs_u4, n_sub, nds;
        size_t   hoff, soff, doff;
        __host__ __device__ geom(uint32_t ne0, uint32_t ne1)
            : rs(repack_qs_row_stride(GGML_TYPE_Q6_K, ne0)), rs_u4(rs >> 4),
              n_sub(ne0 >> 5), nds((n_sub + 7) >> 3),
              hoff((size_t) ne1 * rs),
              soff(hoff + (size_t) ne1 * n_sub * 8),
              doff(soff + (size_t) ne1 * n_sub * 2) {}
    };
#if defined(GGML_USE_HIP) && defined(__gfx906__)
    static __device__ __forceinline__ void load_w(const uint8_t * __restrict__ wbase,
            const geom & g, uint32_t wrow, uint32_t sb, uint4 & lo, uint4 & hi, uint32_t & d) {
        const uint4 * qsp = reinterpret_cast<const uint4 *>(wbase);
        const uint4  pk = rp_ldcs_u4(qsp + (size_t) wrow * g.rs_u4 + sb);
        const size_t wi = (size_t) wrow * g.n_sub + sb;
        const uint2  hq = *reinterpret_cast<const uint2 *>(wbase + g.hoff + wi * 8);
        const uint32_t * pq  = reinterpret_cast<const uint32_t *>(&pk);
        uint32_t       * plo = reinterpret_cast<uint32_t *>(&lo);
        uint32_t       * phi = reinterpret_cast<uint32_t *>(&hi);
#pragma unroll
        for (int j = 0; j < 4; j++) {
            const uint32_t hl = (hq.x >> (8 * j)) & 0xFFu;
            const uint32_t hh = (hq.y >> (8 * j)) & 0xFFu;
            const uint32_t vl = (pq[j] & 0x0F0F0F0Fu)        | (rp_spread2(hl) << 4);
            const uint32_t vh = ((pq[j] >> 4) & 0x0F0F0F0Fu) | (rp_spread2(hh) << 4);
            plo[j] = ((vl << 2) & 0xFCFCFCFCu) ^ 0x80808080u;
            phi[j] = ((vh << 2) & 0xFCFCFCFCu) ^ 0x80808080u;
        }
        const uint32_t sc = *reinterpret_cast<const uint16_t *>(wbase + g.soff + wi * 2);
        const uint32_t dv = *reinterpret_cast<const uint16_t *>(
            wbase + g.doff + ((size_t) wrow * g.nds + (sb >> 3)) * 2);
        d = sc | (dv << 16);
    }
    // {d*sc_lo, d*sc_hi} in f32, the 0.25 undoes the << 2 of the stored values.
    static __device__ __forceinline__ float2 scale2(const uint32_t slot) {
        const uint16_t db = (uint16_t)(slot >> 16);
        const float d = __half2float(*reinterpret_cast<const __half *>(&db)) * 0.25f;
        return make_float2(d * (float)(int8_t)(slot & 0xFFu), d * (float)(int8_t)((slot >> 8) & 0xFFu));
    }
#endif
};

template <> struct rp_traits<GGML_TYPE_Q4_K> {
    // Affine: x = (d*sc)*q - (dmin*m), q in [0,15], a 6-bit scale/min pair per 32 values.
    // The dot therefore needs the activation SUM next to the int dot, and both stock quantizers already provide it (block_q8_1.ds.y on the row path, the DS4 MMQ layout on the GEMM path).
    // Planes: packed nibbles on the de-aliased row stride [16 B/32, byte i = value i | value (16+i) << 4], the canonical 12-byte scale/min record VERBATIM per super-block (decoded at prefetch, not in the inner loop), half2 {d, dmin} per super-block. d*sc and dmin*m are formed in f32 at use, no extra rounding against stock.
    using slot_t = uint2;   // .x = sc | (m << 8), .y = half2 {d, dmin} bits
    static constexpr bool split_scales = false;
    static constexpr bool affine       = true;
    static constexpr bool raw_lds      = false;
    struct geom {
        uint32_t rs, rs_u4, n_sub, nsb;
        size_t   soff, doff;
        __host__ __device__ geom(uint32_t ne0, uint32_t ne1)
            : rs(repack_qs_row_stride(GGML_TYPE_Q4_K, ne0)), rs_u4(rs >> 4),
              n_sub(ne0 >> 5), nsb((n_sub + 7) >> 3),
              soff((size_t) ne1 * rs),
              doff(soff + (size_t) ne1 * nsb * 12) {}
    };
#if defined(GGML_USE_HIP) && defined(__gfx906__)
    static __device__ __forceinline__ void load_w(const uint8_t * __restrict__ wbase,
            const geom & g, uint32_t wrow, uint32_t sb, uint4 & lo, uint4 & hi, uint2 & d) {
        const uint4 * qsp = reinterpret_cast<const uint4 *>(wbase);
        const uint4  pk = rp_ldcs_u4(qsp + (size_t) wrow * g.rs_u4 + sb);
        lo = make_uint4(pk.x & 0x0F0F0F0Fu, pk.y & 0x0F0F0F0Fu,
                        pk.z & 0x0F0F0F0Fu, pk.w & 0x0F0F0F0Fu);
        hi = make_uint4((pk.x >> 4) & 0x0F0F0F0Fu, (pk.y >> 4) & 0x0F0F0F0Fu,
                        (pk.z >> 4) & 0x0F0F0F0Fu, (pk.w >> 4) & 0x0F0F0F0Fu);
        // canonical get_scale_min_k4 on the verbatim 12-byte record
        const size_t     sbk = (size_t) wrow * g.nsb + (sb >> 3);
        const uint32_t * s3  = reinterpret_cast<const uint32_t *>(wbase + g.soff + sbk * 12);
        const uint32_t s0 = s3[0], s1 = s3[1], s2 = s3[2];
        const uint32_t j  = sb & 7;
        uint32_t sc, m;
        if (j < 4) {
            sc = (s0 >> (8 * j)) & 63u;
            m  = (s1 >> (8 * j)) & 63u;
        } else {
            const uint32_t jj = 8 * (j - 4);
            sc = ((s2 >> jj) & 0xFu)       | (((s0 >> (jj + 6)) & 3u) << 4);
            m  = ((s2 >> (jj + 4)) & 0xFu) | (((s1 >> (jj + 6)) & 3u) << 4);
        }
        d.x = sc | (m << 8);
        d.y = *reinterpret_cast<const uint32_t *>(wbase + g.doff + sbk * 4);
    }
    // {d*sc, dmin*m} in f32: the affine epilogue does dlo*dx*idot - dhi*sum.
    static __device__ __forceinline__ float2 scale2(const uint2 slot) {
        const float2 dm = __half22float2(*reinterpret_cast<const half2 *>(&slot.y));
        return make_float2(dm.x * (float)(slot.x & 0xFFu),
                           dm.y * (float)((slot.x >> 8) & 0xFFu));
    }
#endif
};

template <> struct rp_traits<GGML_TYPE_Q5_K> {
    // Q4_K's affine twin: q in [0,31], the fifth bit of every value comes from a 4-byte plane per 32-value sub-block (bit i = value i) and is spread into the byte lanes at load.
    // Scale record, {d, dmin} and the epilogue are Q4_K's.
    using slot_t = uint2;   // .x = sc | (m << 8), .y = half2 {d, dmin} bits
    static constexpr bool split_scales = false;
    static constexpr bool affine       = true;
    static constexpr bool raw_lds      = false;
    struct geom {
        uint32_t rs, rs_u4, n_sub, nsb;
        size_t   hoff, soff, doff;
        __host__ __device__ geom(uint32_t ne0, uint32_t ne1)
            : rs(repack_qs_row_stride(GGML_TYPE_Q5_K, ne0)), rs_u4(rs >> 4),
              n_sub(ne0 >> 5), nsb((n_sub + 7) >> 3),
              hoff((size_t) ne1 * rs),
              soff(hoff + (size_t) ne1 * n_sub * 4),
              doff(soff + (size_t) ne1 * nsb * 12) {}
    };
#if defined(GGML_USE_HIP) && defined(__gfx906__)
    static __device__ __forceinline__ void load_w(const uint8_t * __restrict__ wbase,
            const geom & g, uint32_t wrow, uint32_t sb, uint4 & lo, uint4 & hi, uint2 & d) {
        const uint4 * qsp = reinterpret_cast<const uint4 *>(wbase);
        const uint4  pk = rp_ldcs_u4(qsp + (size_t) wrow * g.rs_u4 + sb);
        const size_t wi = (size_t) wrow * g.n_sub + sb;
        const uint32_t h = *reinterpret_cast<const uint32_t *>(wbase + g.hoff + wi * 4);
        const uint32_t * pq  = reinterpret_cast<const uint32_t *>(&pk);
        uint32_t       * plo = reinterpret_cast<uint32_t *>(&lo);
        uint32_t       * phi = reinterpret_cast<uint32_t *>(&hi);
#pragma unroll
        for (int j = 0; j < 4; j++) {
            plo[j] = (pq[j] & 0x0F0F0F0Fu)        | (rp_spread1((h >> (4 * j))      & 0xFu) << 4);
            phi[j] = ((pq[j] >> 4) & 0x0F0F0F0Fu) | (rp_spread1((h >> (16 + 4 * j)) & 0xFu) << 4);
        }
        // canonical get_scale_min_k4 on the verbatim 12-byte record
        const size_t     sbk = (size_t) wrow * g.nsb + (sb >> 3);
        const uint32_t * s3  = reinterpret_cast<const uint32_t *>(wbase + g.soff + sbk * 12);
        const uint32_t s0 = s3[0], s1 = s3[1], s2 = s3[2];
        const uint32_t j  = sb & 7;
        uint32_t sc, m;
        if (j < 4) {
            sc = (s0 >> (8 * j)) & 63u;
            m  = (s1 >> (8 * j)) & 63u;
        } else {
            const uint32_t jj = 8 * (j - 4);
            sc = ((s2 >> jj) & 0xFu)       | (((s0 >> (jj + 6)) & 3u) << 4);
            m  = ((s2 >> (jj + 4)) & 0xFu) | (((s1 >> (jj + 6)) & 3u) << 4);
        }
        d.x = sc | (m << 8);
        d.y = *reinterpret_cast<const uint32_t *>(wbase + g.doff + sbk * 4);
    }
    static __device__ __forceinline__ float2 scale2(const uint2 slot) {
        const float2 dm = __half22float2(*reinterpret_cast<const half2 *>(&slot.y));
        return make_float2(dm.x * (float)(slot.x & 0xFFu),
                           dm.y * (float)((slot.x >> 8) & 0xFFu));
    }
#endif
};

template <> struct rp_traits<GGML_TYPE_Q5_1> {
    // The 32-value affine block: x = d*q + m, q in [0,31], one half2 {d, m} per block, no super-block.
    // Quantizers fall back to it whenever a row is not a multiple of 256, so it is the type the K-quants cannot cover (the 640-column expert downs of Qwen3.8).
    // Canonical block_q5_1 already keeps the nibbles as byte i = value i | value (16+i) << 4 and the fifth bits as one 32-bit word, bit i = value i, so the planes are copies: nibble rows on the de-aliased stride, a 4-byte fifth-bit plane, a half2 {d, m} plane.
    // The affine epilogue does dlo*dx*idot - dhi*sum, so scale2 returns {d, -m}.
    using slot_t = uint32_t;   // half2 {d, m} bits
    static constexpr bool split_scales = false;
    static constexpr bool affine       = true;
    static constexpr bool raw_lds      = false;
    struct geom {
        uint32_t rs, rs_u4, n_sub;
        size_t   hoff, doff;
        __host__ __device__ geom(uint32_t ne0, uint32_t ne1)
            : rs(repack_qs_row_stride(GGML_TYPE_Q5_1, ne0)), rs_u4(rs >> 4),
              n_sub(ne0 >> 5),
              hoff((size_t) ne1 * rs),
              doff(hoff + (size_t) ne1 * n_sub * 4) {}
    };
#if defined(GGML_USE_HIP) && defined(__gfx906__)
    static __device__ __forceinline__ void load_w(const uint8_t * __restrict__ wbase,
            const geom & g, uint32_t wrow, uint32_t sb, uint4 & lo, uint4 & hi, uint32_t & d) {
        const uint4 * qsp = reinterpret_cast<const uint4 *>(wbase);
        const uint4  pk = rp_ldcs_u4(qsp + (size_t) wrow * g.rs_u4 + sb);
        const size_t wi = (size_t) wrow * g.n_sub + sb;
        const uint32_t h = *reinterpret_cast<const uint32_t *>(wbase + g.hoff + wi * 4);
        const uint32_t * pq  = reinterpret_cast<const uint32_t *>(&pk);
        uint32_t       * plo = reinterpret_cast<uint32_t *>(&lo);
        uint32_t       * phi = reinterpret_cast<uint32_t *>(&hi);
#pragma unroll
        for (int j = 0; j < 4; j++) {
            plo[j] = (pq[j] & 0x0F0F0F0Fu)        | (rp_spread1((h >> (4 * j))      & 0xFu) << 4);
            phi[j] = ((pq[j] >> 4) & 0x0F0F0F0Fu) | (rp_spread1((h >> (16 + 4 * j)) & 0xFu) << 4);
        }
        d = *reinterpret_cast<const uint32_t *>(wbase + g.doff + wi * 4);
    }
    static __device__ __forceinline__ float2 scale2(const uint32_t slot) {
        const float2 dm = __half22float2(*reinterpret_cast<const half2 *>(&slot));
        return make_float2(dm.x, -dm.y);
    }
#endif
};

// One entry per supported repack type - generates the dispatch switches.
#define RP_FOREACH_TYPE(X)     X(GGML_TYPE_Q8_0)          X(GGML_TYPE_MXFP4)          X(GGML_TYPE_IQ4_NL)          X(GGML_TYPE_Q6_K)          X(GGML_TYPE_Q4_K)          X(GGML_TYPE_Q5_K)          X(GGML_TYPE_Q5_1)
void repack_q8_0_host(const block_q8_0 * blocks, uint8_t * dst, const int64_t ne0, const int64_t ne1);
void repack_mxfp4_host(const block_mxfp4 * blocks, uint8_t * dst, const int64_t ne0, const int64_t ne1);
void repack_iq4nl_host(const block_iq4_nl * blocks, uint8_t * dst, const int64_t ne0, const int64_t ne1);
void repack_q6k_host(const block_q6_K * blocks, uint8_t * dst, const int64_t ne0, const int64_t ne1);
void repack_q4k_host(const block_q4_K * blocks, uint8_t * dst, const int64_t ne0, const int64_t ne1);
void repack_q5k_host(const block_q5_K * blocks, uint8_t * dst, const int64_t ne0, const int64_t ne1);
void repack_q51_host(const block_q5_1 * blocks, uint8_t * dst, const int64_t ne0, const int64_t ne1);
void repack_host(ggml_type type, const void * blocks, uint8_t * dst, const int64_t ne0, const int64_t ne1);

const uint8_t * repack_view_get_cached(
        const ggml_tensor * view, const ggml_tensor * base, cudaStream_t stream);

// Drop cached re-packed views that live inside [base, base+size) - called when the
// owning buffer is freed so the allocator cannot hand the same addresses to a
// different model and hit a stale entry.
void repack_view_cache_purge(int device, const void * base, size_t size);

// Quantize src1 into Q8_1 rows, reusing the graph-wide activation cache when the
// same activation was already quantized to this exact layout. One activation
// feeds several matmuls back to back - q/k/v off one attention norm, router and
// routed gate/up off one ffn norm - and each re-quantized its own copy, so the
// launch count tracked the matmul count exactly. quantize_row_q8_1_cuda ignores
// type_src0 and always writes the plain row layout, so variant 0 is the same
// layout the canonical mat-vec caches under and the two share entries. The MoE
// sites pass the rows flattened as (n_cols, 1, 1) where the dense sites pass
// (ne11, ne12, ne13). That is still the same bytes: the kernel writes
// i_cont = ((i3*ne2 + i2)*ne1 + i1)*ne0 + i0, which is contiguous either way,
// and it can only collide in the cache when the strides and the size already
// match, which is the contiguous case where both walk src1 identically.
// Fusing the quantize into the mat-vec instead is the wrong shape: it is a
// shared producer, and making every block re-quantize the row cost 42 percent.
#if 0   // Strata: ggml graph glue (needs mx's q8_1 activation cache, not in llama.cpp 3cf0325); unused here
static inline block_q8_1 * repack_quantize_src1_q8_1(
        ggml_backend_cuda_context & ctx, const ggml_tensor * src0, const ggml_tensor * src1,
        int64_t ne10, int64_t ne10_padded, int64_t s11, int64_t s12, int64_t s13,
        int64_t nrows, int64_t ne2, int64_t ne3, size_t nblocks,
        ggml_cuda_pool_alloc<block_q8_1> & fallback, cudaStream_t stream) {
    bool hit = false;
    char * cached = ggml_cuda_q8_1_cache_acquire(ctx, src1, /*variant =*/ 0, ne10_padded,
                                                 s11, s12, s13, nblocks * sizeof(block_q8_1), hit);
    block_q8_1 * dstq;
    if (cached) {
        dstq = (block_q8_1 *) cached;
    } else {
        fallback.alloc(ctx.pool(), nblocks);
        dstq = fallback.get();
    }
    if (!hit) {
        quantize_row_q8_1_cuda((const float *) src1->data, nullptr, dstq,
            src0->type, ne10, s11, s12, s13, ne10_padded, nrows, ne2, ne3, stream);
    }
    return dstq;
}
#endif
