// src/prefill/rp/repack-iq4xs.cuh - Strata's IQ4_XS extension of mx-llama.cpp's repacked GEMM (repack-common.cuh's
// rp_traits; mx has no IQ4_XS).  IQ4_XS is IQ4_NL's codebook in 256-value super-blocks: each 32-value sub-block has
// IQ4_NL's nibble layout and the scale d * (ls - 32) (f16 d per super-block, 6-bit ls per sub-block), which llama.cpp's
// MMQ computes in FP32 per sub-block (mmq-load-tiles.cuh, load_tiles_iq4_xs) and multiplies like IQ4_NL's.  The
// repacked layout: IQ4_NL's de-aliased nibble rows [ne1 x rs], then an FP32 plane of those per-sub-block scales
// [ne1 x n_sub], precomputed, so the GEMM takes IQ4_NL's path with a float slot (5 bpw against the GGUF 4.25: a
// per-call scratch, never resident).
#pragma once

#include "repack-common.cuh"

static inline size_t rp_iq4xs_nbytes(const int64_t ne0, const int64_t ne1) {
    return (size_t) ne1 * ((size_t) repack_qs_stride(ne0 / 2) + (size_t) (ne0 / 32) * 4);
}

template <> struct rp_traits<GGML_TYPE_IQ4_XS> {
    using slot_t = float;   // d * (ls - 32), exact in FP32
    static constexpr bool split_scales = false;
    static constexpr bool affine       = false;
    struct geom {
        uint32_t rs, rs_u4, n_sub;
        size_t   dplane_off;
        __host__ __device__ geom(uint32_t ne0, uint32_t ne1)
            : rs(repack_qs_stride(ne0 / 2)), rs_u4(rs >> 4), n_sub(ne0 >> 5), dplane_off((size_t) ne1 * rs) {}
    };
    static constexpr bool raw_lds = true;
#if defined(GGML_USE_HIP) && defined(__gfx906__)
    static __device__ __forceinline__ void load_w(const uint8_t * __restrict__ wbase,
            const geom & g, uint32_t wrow, uint32_t sb, uint4 & lo, uint4 & hi, float & d) {
        const uint4 * qsp = reinterpret_cast<const uint4 *>(wbase);
        rp_iq4nl_expand(rp_ldcs_u4(qsp + (size_t) wrow * g.rs_u4 + sb), lo, hi);
        d = reinterpret_cast<const float *>(wbase + g.dplane_off)[(size_t) wrow * g.n_sub + sb];
    }
    static __device__ __forceinline__ void load_w_raw(const uint8_t * __restrict__ wbase,
            const geom & g, uint32_t wrow, uint32_t sb, uint4 & raw, float & d) {
        const uint4 * qsp = reinterpret_cast<const uint4 *>(wbase);
        raw = rp_ldcs_u4(qsp + (size_t) wrow * g.rs_u4 + sb);
        d   = reinterpret_cast<const float *>(wbase + g.dplane_off)[(size_t) wrow * g.n_sub + sb];
    }
    static __device__ __forceinline__ float scale(const float s) { return s; }
    static __device__ __forceinline__ void expand_raw(const uint4 raw, uint4 & lo, uint4 & hi) {
        rp_iq4nl_expand(raw, lo, hi);
    }
#endif
};

// GGUF block_iq4_xs (136 B: f16 d, u16 scales_h, u8 scales_l[4], u8 qs[128]) to the planes above: one thread per
// 32-value sub-block, the same scale formula as llama.cpp's MMQ loader.
static __global__ void repack_iq4xs_kernel(
        const uint8_t * __restrict__ src, uint8_t * __restrict__ dst,
        const int64_t ne1, const int64_t n_sub, const int64_t qs_str,
        const int64_t qs_len, const int64_t src_stride, const int64_t dst_stride,
        const int64_t total) {
    for (int64_t i = (int64_t) blockIdx.x * blockDim.x + threadIdx.x;
         i < total; i += (int64_t) gridDim.x * blockDim.x) {
        const int64_t sb  = i % n_sub;
        const int64_t row = (i / n_sub) % ne1;
        const int64_t e   = i / (n_sub * ne1);
        const block_iq4_xs * b = reinterpret_cast<const block_iq4_xs *>(src + e * src_stride) + row * (n_sub / 8) + sb / 8;
        const int ib = (int) (sb % 8);
        const float d = __half2float(b->d);
        const int ls = ((b->scales_l[ib / 2] >> (4 * (ib % 2))) & 0x0F) | (((b->scales_h >> (2 * ib)) & 0x03) << 4);
        const uint32_t * q = reinterpret_cast<const uint32_t *>(b->qs + ib * 16);   // 8-byte aligned (offset 8 of 136)
        *reinterpret_cast<uint4 *>(dst + e * dst_stride + row * qs_str + sb * 16) = make_uint4(q[0], q[1], q[2], q[3]);
        reinterpret_cast<float *>(dst + e * dst_stride + qs_len)[row * n_sub + sb] = d * (ls - 32);
    }
}
