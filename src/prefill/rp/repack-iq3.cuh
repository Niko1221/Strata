// src/prefill/rp/repack-iq3.cuh - Strata's codebook extension of mx-llama.cpp's repacked GEMM (repack-common.cuh's
// rp_traits; mx has none of these): IQ3_XXS, IQ3_S and IQ2_S, the gate/up experts of the GSQ-RCO IQ3_S file.
//
// llama.cpp's MMQ decodes these per tile: grid lookups and sign masks into int8 values, and an FP32 scale per 32 values
// (IQ3_XXS, IQ3_S; the Q8_0 dot) or per 16 (IQ2_S; the q8_0_16 dot) - mmq-load-tiles.cuh, load_tiles_iq3_xxs / _iq3_s /
// _iq2_s.  The repacked layout is that decoded form, made once per gather: de-aliased int8 rows [ne1 x rs] (Q8_0's
// qs plane) and a 4-byte scale plane [ne1 x n_sub] (the FP32 scale; IQ2_S: its f16 d and 4-bit scale pair, the two
// per-16 scales formed in the GEMM by MMQ's expression - an 8-byte slot of two floats spilled 152 bytes per lane), so
// the GEMM takes the Q8_0 path with a float slot (IQ2_S: the split-scale path, summed in MMQ's order).  9 bpw against
// the GGUF 2.5-3.4: a per-chunk scratch (the group buffers), never resident.
#pragma once

#include "repack-common.cuh"

#include <type_traits>

static inline size_t rp_iq3_nbytes(const ggml_type t, const int64_t ne0, const int64_t ne1) {
    GGML_UNUSED(t);
    return (size_t) ne1 * ((size_t) repack_qs_stride(ne0) + (size_t) (ne0 / 32) * 4);
}

// The int8 rows and FP32 scales, per 32-value sub-block.
struct rp_i8f32_traits {
    using slot_t = float;
    static constexpr bool split_scales = false;
    static constexpr bool affine       = false;
    static constexpr bool raw_lds      = false;
    struct geom {
        uint32_t rs, rs_u4, n_sub;
        size_t   dplane_off;
        __host__ __device__ geom(uint32_t ne0, uint32_t ne1)
            : rs(repack_qs_stride(ne0)), rs_u4(rs >> 4), n_sub(ne0 >> 5), dplane_off((size_t) ne1 * rs) {}
    };
#if defined(GGML_USE_HIP) && defined(__gfx906__)
    static __device__ __forceinline__ void load_w(const uint8_t * __restrict__ wbase,
            const geom & g, uint32_t wrow, uint32_t sb, uint4 & lo, uint4 & hi, float & d) {
        const uint4 * qsp = reinterpret_cast<const uint4 *>(wbase);
        const size_t  q4  = (size_t) wrow * g.rs_u4 + 2 * sb;
        lo = rp_ldcs_u4(qsp + q4);
        hi = rp_ldcs_u4(qsp + q4 + 1);
        d  = reinterpret_cast<const float *>(wbase + g.dplane_off)[(size_t) wrow * g.n_sub + sb];
    }
    static __device__ __forceinline__ float scale(const float s) { return s; }
#endif
};

template <> struct rp_traits<GGML_TYPE_IQ3_XXS> : rp_i8f32_traits {};
template <> struct rp_traits<GGML_TYPE_IQ3_S> : rp_i8f32_traits {};

// IQ2_S: the same rows, two FP32 scales per sub-block (values 0-15, 16-31), combined as MMQ's q8_0_16 dot does
// (dx * (d_lo * lo + d_hi * hi), not Q6_K's d_lo * dx * lo + d_hi * dx * hi): split_mmq16.
template <> struct rp_traits<GGML_TYPE_IQ2_S> {
    using slot_t = uint32_t;   // f16 d | the scale byte (low nibble: values 0-15, high: 16-31) << 16
    static constexpr bool split_scales = true;
    static constexpr bool split_mmq16  = true;
    static constexpr bool affine       = false;
    static constexpr bool raw_lds      = false;
    using geom = rp_i8f32_traits::geom;
#if defined(GGML_USE_HIP) && defined(__gfx906__)
    static __device__ __forceinline__ void load_w(const uint8_t * __restrict__ wbase,
            const geom & g, uint32_t wrow, uint32_t sb, uint4 & lo, uint4 & hi, uint32_t & d) {
        const uint4 * qsp = reinterpret_cast<const uint4 *>(wbase);
        const size_t  q4  = (size_t) wrow * g.rs_u4 + 2 * sb;
        lo = rp_ldcs_u4(qsp + q4);
        hi = rp_ldcs_u4(qsp + q4 + 1);
        d  = reinterpret_cast<const uint32_t *>(wbase + g.dplane_off)[(size_t) wrow * g.n_sub + sb];
    }
    // load_tiles_iq2_s's x_df
    static __device__ __forceinline__ float2 scale2(const uint32_t s) {
        const uint16_t db = (uint16_t) (s & 0xFFFFu);
        const float d = __half2float(*reinterpret_cast<const __half *>(&db));
        const int ls = (int) (s >> 16);
        return make_float2(((ls &  0x0F)*d + d/2)/4, ((ls >>    4)*d + d/2)/4);
    }
#endif
};

// Sub-block k (0-7) of one GGUF super-block decoded as MMQ's tile loaders do: 8 ints (32 int8 values in order) and the
// scale(s).  The same expressions as mmq-load-tiles.cuh, so the same values.
static __device__ __forceinline__ void rp_decode_iq3_xxs(const block_iq3_xxs * bxi, const int k, int v[8], float s[2],
                                                         const uint32_t * __restrict__ grid) {
    const int2 q3_packed = make_int2(get_int_b2(bxi->qs, 2*k+0), get_int_b2(bxi->qs, 2*k+1));
    const uint8_t * q3 = (const uint8_t *) &q3_packed;
    const uint32_t aux32 = get_int_b2(bxi->qs, QK_K/16 + k);
#pragma unroll
    for (int l = 0; l < QR3_XXS; ++l) {
        const int2 grid_pos = make_int2(grid[q3[2*l+0]], grid[q3[2*l+1]]);
        const uint32_t signs = unpack_ksigns(aux32 >> (7*l));
        const int signs0 = __vcmpne4(signs & 0x08040201, 0);
        const int signs1 = __vcmpne4(signs & 0x80402010, 0);
        v[2*l+0] = __vsub4(grid_pos.x ^ signs0, signs0);
        v[2*l+1] = __vsub4(grid_pos.y ^ signs1, signs1);
    }
    const int ls = aux32 >> 28;
    const float d = bxi->d;
    s[0] = (ls*d + d/2)/2;
}

static __device__ __forceinline__ void rp_decode_iq3_s(const block_iq3_s * bxi, const int k, int v[8], float s[2],
                                                       const uint32_t * __restrict__ grid) {
    const int2 qs_packed = make_int2(get_int_b2(bxi->qs, 2*k+0), get_int_b2(bxi->qs, 2*k+1));
    const uint8_t * qs = (const uint8_t *) &qs_packed;
    const int qh = bxi->qh[k];
    const int signs_packed_32 = get_int_b2(bxi->signs, k);
    const uint8_t * signs_packed_8 = (const uint8_t *) &signs_packed_32;
#pragma unroll
    for (int l = 0; l < QR3_S; ++l) {
        const int2 grid_pos = make_int2(
            grid[qs[2*l+0] | ((qh << (8 - 2*l)) & 0x100)],
            grid[qs[2*l+1] | ((qh << (7 - 2*l)) & 0x100)]);
        const int signs0 = __vcmpne4(((signs_packed_8[l] & 0x03) << 7) | ((signs_packed_8[l] & 0x0C) << 21), 0x00000000);
        const int signs1 = __vcmpne4(((signs_packed_8[l] & 0x30) << 3) | ((signs_packed_8[l] & 0xC0) << 17), 0x00000000);
        v[2*l+0] = __vsub4(grid_pos.x ^ signs0, signs0);
        v[2*l+1] = __vsub4(grid_pos.y ^ signs1, signs1);
    }
    const int ls = 1 + 2*((bxi->scales[k/2] >> (((2*k) << 1) & 0x04)) & 0x0F);
    const float d = bxi->d;
    s[0] = ls*d;
}

static __device__ __forceinline__ void rp_decode_iq2_s(const block_iq2_s * bxi, const int k, int v[8], float s[2],
                                                       const uint64_t * __restrict__ grid) {
    const int qs_packed = get_int_b2(bxi->qs, k);
    const uint8_t * qs = (const uint8_t *) &qs_packed;
    const int qh = bxi->qh[k];
    const int signs_packed_32 = get_int_b2(bxi->qs, QK_K/32 + k);
    const uint8_t * signs_packed_8 = (const uint8_t *) &signs_packed_32;
#pragma unroll
    for (int l = 0; l < QR2_S; ++l) {
        const int * grid_pos = (const int *)(grid + (qs[l] | ((qh << (8-2*l)) & 0x300)));
        const int signs0 = __vcmpne4(((signs_packed_8[l] & 0x03) << 7) | ((signs_packed_8[l] & 0x0C) << 21), 0x00000000);
        const int signs1 = __vcmpne4(((signs_packed_8[l] & 0x30) << 3) | ((signs_packed_8[l] & 0xC0) << 17), 0x00000000);
        v[2*l+0] = __vsub4(grid_pos[0] ^ signs0, signs0);
        v[2*l+1] = __vsub4(grid_pos[1] ^ signs1, signs1);
    }
    const int ls = bxi->scales[k];
    const float d = bxi->d;
    s[0] = ((ls &  0x0F)*d + d/2)/4;
    s[1] = ((ls >>    4)*d + d/2)/4;
}

// The type's codebook copied into LDS by the whole block (random per-lane lookups into the global table halved the
// gather's speed); every thread of the block calls it before its first decode.
template <ggml_type T> struct rp_iq3_grid {
    using type = std::conditional_t<T == GGML_TYPE_IQ2_S, uint64_t, uint32_t>;
    static constexpr int n = T == GGML_TYPE_IQ3_XXS ? 256 : T == GGML_TYPE_IQ3_S ? 512 : 1024;
};
template <ggml_type T>
static __device__ __forceinline__ const typename rp_iq3_grid<T>::type * rp_iq3_grid_lds() {
    __shared__ typename rp_iq3_grid<T>::type g[rp_iq3_grid<T>::n];
    for (int i = threadIdx.x; i < rp_iq3_grid<T>::n; i += blockDim.x) {
        if constexpr (T == GGML_TYPE_IQ3_XXS) g[i] = iq3xxs_grid[i];
        else if constexpr (T == GGML_TYPE_IQ3_S) g[i] = iq3s_grid[i];
        else g[i] = iq2s_grid[i];
    }
    __syncthreads();
    return g;
}

// One sub-block from GGUF super-block `blk` (bytes) to row `row`, sub-block `sb` of a repacked matrix at `m`.
template <ggml_type T>
static __device__ __forceinline__ void rp_iq3_put(const uint8_t * blk, const int k, uint8_t * m, const int64_t rows,
                                                  const int64_t qs_str, const int64_t n_sub, const int64_t row,
                                                  const int64_t sb, const typename rp_iq3_grid<T>::type * grid) {
    int v[8];
    float s[2] = {0.0f, 0.0f};
    if constexpr (T == GGML_TYPE_IQ3_XXS) rp_decode_iq3_xxs(reinterpret_cast<const block_iq3_xxs *>(blk), k, v, s, grid);
    else if constexpr (T == GGML_TYPE_IQ3_S) rp_decode_iq3_s(reinterpret_cast<const block_iq3_s *>(blk), k, v, s, grid);
    else rp_decode_iq2_s(reinterpret_cast<const block_iq2_s *>(blk), k, v, s, grid);
    uint4 * q = reinterpret_cast<uint4 *>(m + (size_t) row * qs_str + (size_t) sb * 32);
    q[0] = make_uint4((uint32_t) v[0], (uint32_t) v[1], (uint32_t) v[2], (uint32_t) v[3]);
    q[1] = make_uint4((uint32_t) v[4], (uint32_t) v[5], (uint32_t) v[6], (uint32_t) v[7]);
    if constexpr (T == GGML_TYPE_IQ2_S) {
        const block_iq2_s * b = reinterpret_cast<const block_iq2_s *>(blk);
        reinterpret_cast<uint32_t *>(m + (size_t) rows * qs_str)[row * n_sub + sb] =
            (uint32_t) *reinterpret_cast<const uint16_t *>(&b->d) | ((uint32_t) b->scales[k] << 16);
    } else
        reinterpret_cast<float *>(m + (size_t) rows * qs_str)[row * n_sub + sb] = s[0];
}

template <ggml_type T> static constexpr int64_t rp_iq3_block_bytes() {
    return T == GGML_TYPE_IQ3_XXS ? (int64_t) sizeof(block_iq3_xxs)
         : T == GGML_TYPE_IQ3_S   ? (int64_t) sizeof(block_iq3_s) : (int64_t) sizeof(block_iq2_s);
}

// The repack of whole GGUF matrices: blockIdx.y = matrix, one thread per sub-block of its ne1 x n_sub (32-bit index
// math: the 64-bit divisions of a flat index halved the kernel's speed).
template <ggml_type T>
static __global__ void repack_iq3_kernel(const uint8_t * __restrict__ src, uint8_t * __restrict__ dst,
                                         const uint32_t ne1, const uint32_t n_sub, const uint32_t qs_str,
                                         const size_t src_stride, const size_t dst_stride) {
    const uint8_t * s = src + blockIdx.y * src_stride;
    uint8_t * m = dst + blockIdx.y * dst_stride;
    const uint32_t total = ne1 * n_sub;
    const auto * grid = rp_iq3_grid_lds<T>();
    for (uint32_t i = blockIdx.x * blockDim.x + threadIdx.x; i < total; i += gridDim.x * blockDim.x) {
        const uint32_t row = i / n_sub, sb = i - row * n_sub;
        const uint8_t * blk = s + ((size_t) row * (n_sub / 8) + sb / 8) * rp_iq3_block_bytes<T>();
        rp_iq3_put<T>(blk, (int) (sb % 8), m, ne1, qs_str, n_sub, row, sb, grid);
    }
}
