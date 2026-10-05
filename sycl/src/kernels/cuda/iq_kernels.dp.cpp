// src/kernels/cuda/iq_kernels.cu - see include/strata/kernels/iq_kernels.hpp.
//
// The dot products (vec_dot_*_q8_1), the dequantizers and the q8_1 quantizer are transcribed from llama.cpp
// (ggml/src/ggml-cuda/vecdotq.cuh, dequantize.cuh, quantize.cu at the commit in third_party/ggml/VERSION.txt;
// MIT license, third_party/ggml/LICENSE).  The block structs and codebook grids come from its ggml-common.h,
// included unchanged.
#define DPCT_PROFILING_ENABLED
#include <sycl/sycl.hpp>
#include <dpct/dpct.hpp>
#include "strata/sycl_math.hpp"
#include "strata/sycl_queue.hpp"
#include "strata/kernels/iq_kernels.hpp"
#include "strata/kernels/dp4a.hpp"
#include "strata/kernels/q8_1_finite.hpp"

#define GGML_COMMON_DECL_SYCL
#define GGML_COMMON_IMPL_SYCL
#include "ggml-common.h"

#include <sycl/ext/oneapi/matrix/matrix.hpp>
#include <algorithm>
#include <type_traits>
#include <cstdio>
#include <cstdlib>

namespace strata::kernels {
namespace {

void check(const char* what) {
    /*
    DPCT1010: SYCL uses exceptions to report errors and does not use the
    error codes. The cudaGetLastError function call was replaced with 0. You
    need to rewrite this code.
    */
    const dpct::err0 e = 0;
    /*
    DPCT1009: SYCL reports errors using exceptions and does not use error
    codes. Please replace the "get_error_string_dummy(...)" with a real
    error-handling function.
    */
}

// ---------------------------------------------------------------- llama.cpp helpers (vecdotq.cuh)
__dpct_inline__ int get_int_b2(const void *x, const int &i32) {
    const uint16_t* x16 = (const uint16_t*) x;
    int x32 = x16[2 * i32 + 0] << 0;
    x32 |= x16[2 * i32 + 1] << 16;
    return x32;
}
__dpct_inline__ int get_int_b4(const void *x, const int &i32) {
    return ((const int *)x)[i32];
}
__dpct_inline__ uint32_t unpack_ksigns(const uint8_t v) {
    const uint32_t p = sycl::popcount(v) & 1;
    const uint32_t s = v ^ p << 7;
    return s * 0x01010101;
}
__dpct_inline__ sycl::int2 get_int_from_table_16(const int &q4,
                                                 const int8_t *table) {
    const uint32_t* table32 = (const uint32_t*) table;
    uint32_t tmp[2];
    const uint32_t low_high_selection_indices = (0x32103210 | ((q4 & 0x88888888) >> 1));
#pragma unroll
    for (uint32_t i = 0; i < 2; ++i) {
        const uint32_t shift = 16 * i;
        const uint32_t low =
            dpct::byte_level_permute(table32[0], table32[1], q4 >> shift);
        const uint32_t high =
            dpct::byte_level_permute(table32[2], table32[3], q4 >> shift);
        tmp[i] = dpct::byte_level_permute(low, high,
                                          low_high_selection_indices >> shift);
    }
    return sycl::int2(dpct::byte_level_permute(tmp[0], tmp[1], 0x6420),
                      dpct::byte_level_permute(tmp[0], tmp[1], 0x7531));
}
#define ggml_cuda_dp4a(a, b, c) strata::dp4a((a), (b), (c))   // SYCL port: the one dp4a (strata/sycl_math.hpp)

// ---------------------------------------------------------------- the dot products (vecdotq.cuh)
__dpct_inline__ float vec_dot_q2_0_q8_1(const void *__restrict__ vbq,
                                        const block_q8_1 *__restrict__ bq8_1,
                                        const int &kbx, const int &iqs) {
    const block_q2_0* bq2_0 = (const block_q2_0*) vbq + kbx;
    const float d2 = bq2_0->d;
    const int16_t* qs = (const int16_t*) bq2_0->qs + iqs * 4;
    const block_q8_1* bq8_1_chunk = bq8_1 + iqs;
    int sumi = 0;
#pragma unroll
    for (int j = 0; j < 4; ++j) {
        const int q = qs[j];
        const int u = get_int_b4(bq8_1_chunk->qs, j * 2 + 0);
        const int v = get_int_b4(bq8_1_chunk->qs, j * 2 + 1);
        const int qe = dpct::byte_level_permute(0x020100FF, 0x020100FF, q >> 0);
        const int qo = dpct::byte_level_permute(0x020100FF, 0x020100FF, q >> 2);
        const int qx = dpct::byte_level_permute(qe, qo, 0x5140);
        const int qy = dpct::byte_level_permute(qe, qo, 0x7362);
        sumi = ggml_cuda_dp4a(u, qx, sumi);
        sumi = ggml_cuda_dp4a(v, qy, sumi);
    }
    const float d8 = bq8_1_chunk->ds[0];
    return d2 * d8 * sumi;
}

// SYCL port: ggml's __vcmpne4 / __vsub4 came out of dpct as per-byte loops (extract, compare, insert, x4); on Xe the
// expert dot kernels were ALU-bound with them (77% XVE active, metrics 2026-09-30). SWAR forms of the two uses:
//   swar_ne4(x): 0xFF in every byte of x that is non-zero (the sign masks: at most one bit per byte)
//   swar_sub4(a, s) with s from swar_ne4: per byte a - s, i.e. a + 1 where the sign byte is set (two's complement
//   of the grid value after the xor: the same bits as __vsub4(a, s) for s in {0x00, 0xFF})
#ifndef STRATA_SWAR
#define STRATA_SWAR 1   // 1: the SWAR forms; 0: dpct's per-byte loops (A/B 2026-09-30: both correct)
#endif
#if STRATA_SWAR
__dpct_inline__ int swar_ne4(unsigned x) {
    x = (x | (x >> 4)) & 0x0F0F0F0Fu;
    x = (x | (x >> 2)) & 0x03030303u;
    x = (x | (x >> 1)) & 0x01010101u;
    return (int) (x * 0xFFu);
}
__dpct_inline__ int swar_sub4(unsigned a, unsigned s) {
    return (int) (((a & 0x7F7F7F7Fu) + (s & 0x01010101u)) ^ (a & 0x80808080u));
}
#else
__dpct_inline__ int swar_ne4(unsigned x) { return dpct::vectorized_binary<sycl::uchar4>(x, 0, std::not_equal_to<>()); }
__dpct_inline__ int swar_sub4(unsigned a, unsigned s) { return dpct::vectorized_binary<sycl::uchar4>(a, s, std::minus<>()); }
#endif

__dpct_inline__ float vec_dot_iq2_xxs_q8_1(const void *__restrict__ vbq,
                                           const block_q8_1 *__restrict__ bq8_1,
                                           const int &kbx, const int &iqs) {
    const block_iq2_xxs* bq2 = (const block_iq2_xxs*) vbq + kbx;
    const int q2 = get_int_b2(bq2->qs, iqs);
    const uint8_t* aux8 = (const uint8_t*) &q2;
    const uint32_t aux32 = get_int_b2(bq2->qs, iqs + 1);
    int sumi = 0;
#pragma unroll
    for (int k0 = 0; k0 < 8; k0 += 2) {
        const sycl::uint2 grid_pos =
            ((const sycl::uint2 *)iq2xxs_grid)[aux8[k0 / 2]];
        const uint32_t signs = unpack_ksigns(aux32 >> (7 * k0 / 2));
        const int signs0 = swar_ne4(signs & 0x08040201);
        const int grid0 = swar_sub4(grid_pos.x() ^ signs0, signs0);
        const int u0 = get_int_b4(bq8_1[iqs / 2].qs, k0 + 0);
        sumi = ggml_cuda_dp4a(grid0, u0, sumi);
        const int signs1 = swar_ne4(signs & 0x80402010);
        const int grid1 = swar_sub4(grid_pos.y() ^ signs1, signs1);
        const int u1 = get_int_b4(bq8_1[iqs / 2].qs, k0 + 1);
        sumi = ggml_cuda_dp4a(grid1, u1, sumi);
    }
    const int ls = aux32 >> 27 | 1;
    sumi = sumi * ls / 8;
    const float d = sycl::vec<sycl::half, 1>(bq2->d)
                        .convert<float, sycl::rounding_mode::automatic>()[0] *
                    bq8_1[iqs / 2].ds[0];
    return d * sumi;
}

__dpct_inline__ float vec_dot_iq2_xs_q8_1(const void *__restrict__ vbq,
                                          const block_q8_1 *__restrict__ bq8_1,
                                          const int &kbx, const int &iqs) {
    const block_iq2_xs* bq2 = (const block_iq2_xs*) vbq + kbx;
    // SYCL port: the four 16-bit codes by shifts. ggml reads them through a uint16_t* to an int2, which is type
    // punning the SYCL device compiler does not honour (non-char aliasing): IQ2_XS products came out garbage
    // (iq_multi_parity, Swift 1.5 IQ2_XS) while the dequantizer, reading the codes directly, was exact.
    const uint32_t q2_lo = (uint32_t) get_int_b2(bq2->qs, iqs + 0), q2_hi = (uint32_t) get_int_b2(bq2->qs, iqs + 1);
    const uint16_t q2[4] = {(uint16_t) q2_lo, (uint16_t) (q2_lo >> 16), (uint16_t) q2_hi, (uint16_t) (q2_hi >> 16)};
    const int ls0 = bq2->scales[iqs / 2] & 0x0F;
    const int ls1 = bq2->scales[iqs / 2] >> 4;
    int sumi0 = 0, sumi1 = 0;
#pragma unroll
    for (int l0 = 0; l0 < 8; l0 += 2) {
        const sycl::uint2 grid_pos =
            ((const sycl::uint2 *)iq2xs_grid)[q2[l0 / 2] & 0x1FF];
        const uint32_t signs = unpack_ksigns(q2[l0 / 2] >> 9);
        const int signs0 = swar_ne4(signs & 0x08040201);
        const int grid_l = swar_sub4(grid_pos.x() ^ signs0, signs0);
        const int u0 = get_int_b4(bq8_1[iqs / 2].qs, l0 + 0);
        const int signs1 = swar_ne4(signs & 0x80402010);
        const int grid_h = swar_sub4(grid_pos.y() ^ signs1, signs1);
        const int u1 = get_int_b4(bq8_1[iqs / 2].qs, l0 + 1);
        if (l0 < 4) {
            sumi0 = ggml_cuda_dp4a(grid_l, u0, sumi0);
            sumi0 = ggml_cuda_dp4a(grid_h, u1, sumi0);
        } else {
            sumi1 = ggml_cuda_dp4a(grid_l, u0, sumi1);
            sumi1 = ggml_cuda_dp4a(grid_h, u1, sumi1);
        }
    }
    const int sumi = (sumi0 * ls0 + sumi1 * ls1 + (sumi0 + sumi1) / 2) / 4;
    const float d = sycl::vec<sycl::half, 1>(bq2->d)
                        .convert<float, sycl::rounding_mode::automatic>()[0] *
                    bq8_1[iqs / 2].ds[0];
    return d * sumi;
}

__dpct_inline__ float vec_dot_iq2_s_q8_1(const void *__restrict__ vbq,
                                         const block_q8_1 *__restrict__ bq8_1,
                                         const int &kbx, const int &iqs) {
    const block_iq2_s* bq2 = (const block_iq2_s*) vbq + kbx;
    const int qs_packed = get_int_b2(bq2->qs, iqs / 2);
    const uint8_t* qs = (const uint8_t*) &qs_packed;
    const int qh = bq2->qh[iqs / 2];
    const int signs_packed_32 = get_int_b2(bq2->qs, QK_K / 32 + iqs / 2);
    const uint8_t* signs_packed_8 = (const uint8_t*) &signs_packed_32;
    const int ls0 = bq2->scales[iqs / 2] & 0x0F;
    const int ls1 = bq2->scales[iqs / 2] >> 4;
    int sumi0 = 0, sumi1 = 0;
#pragma unroll
    for (int l0 = 0; l0 < 8; l0 += 2) {
        const int* grid_pos = (const int*) (iq2s_grid + (qs[l0 / 2] | ((qh << (8 - l0)) & 0x300)));
        const int signs0 = swar_ne4(((signs_packed_8[l0 / 2] & 0x03) << 7) |
                ((signs_packed_8[l0 / 2] & 0x0C) << 21));
        const int signs1 = swar_ne4(((signs_packed_8[l0 / 2] & 0x30) << 3) |
                ((signs_packed_8[l0 / 2] & 0xC0) << 17));
        const int grid_l = swar_sub4(grid_pos[0] ^ signs0, signs0);
        const int grid_h = swar_sub4(grid_pos[1] ^ signs1, signs1);
        const int u0 = get_int_b4(bq8_1[iqs / 2].qs, l0 + 0);
        const int u1 = get_int_b4(bq8_1[iqs / 2].qs, l0 + 1);
        if (l0 < 4) {
            sumi0 = ggml_cuda_dp4a(grid_l, u0, sumi0);
            sumi0 = ggml_cuda_dp4a(grid_h, u1, sumi0);
        } else {
            sumi1 = ggml_cuda_dp4a(grid_l, u0, sumi1);
            sumi1 = ggml_cuda_dp4a(grid_h, u1, sumi1);
        }
    }
    const int sumi = (sumi0 * ls0 + sumi1 * ls1 + (sumi0 + sumi1) / 2) / 4;
    const float d = sycl::vec<sycl::half, 1>(bq2->d)
                        .convert<float, sycl::rounding_mode::automatic>()[0] *
                    bq8_1[iqs / 2].ds[0];
    return d * sumi;
}

__dpct_inline__ float vec_dot_iq3_xxs_q8_1(const void *__restrict__ vbq,
                                           const block_q8_1 *__restrict__ bq8_1,
                                           const int &kbx, const int &iqs) {
    const block_iq3_xxs* bq3 = (const block_iq3_xxs*) vbq + kbx;
    const sycl::int2 q3_packed =
        sycl::int2(get_int_b2(bq3->qs, iqs), get_int_b2(bq3->qs, iqs + 1));
    const uint8_t* q3 = (const uint8_t*) &q3_packed;
    const uint32_t aux32 = get_int_b2(bq3->qs, QK_K / 16 + iqs / 2);
    int sumi = 0;
#pragma unroll
    for (int l0 = 0; l0 < 8; l0 += 2) {
        const sycl::int2 grid_pos =
            sycl::int2(iq3xxs_grid[q3[l0 + 0]], iq3xxs_grid[q3[l0 + 1]]);
        const uint32_t signs = unpack_ksigns(aux32 >> (7 * l0 / 2));
        const int signs0 = swar_ne4(signs & 0x08040201);
        const int grid_l = swar_sub4(grid_pos.x() ^ signs0, signs0);
        const int u0 = get_int_b4(bq8_1[iqs / 2].qs, l0 + 0);
        const int signs1 = swar_ne4(signs & 0x80402010);
        const int grid_h = swar_sub4(grid_pos.y() ^ signs1, signs1);
        const int u1 = get_int_b4(bq8_1[iqs / 2].qs, l0 + 1);
        sumi = ggml_cuda_dp4a(grid_l, u0, sumi);
        sumi = ggml_cuda_dp4a(grid_h, u1, sumi);
    }
    const int ls = aux32 >> 28;
    sumi = (ls * sumi + sumi / 2) / 2;
    const float d = sycl::vec<sycl::half, 1>(bq3->d)
                        .convert<float, sycl::rounding_mode::automatic>()[0] *
                    bq8_1[iqs / 2].ds[0];
    return d * sumi;
}

__dpct_inline__ float vec_dot_iq3_s_q8_1(const void *__restrict__ vbq,
                                         const block_q8_1 *__restrict__ bq8_1,
                                         const int &kbx, const int &iqs) {
    const block_iq3_s* bq3 = (const block_iq3_s*) vbq + kbx;
    const sycl::int2 qs_packed =
        sycl::int2(get_int_b2(bq3->qs, iqs + 0), get_int_b2(bq3->qs, iqs + 1));
    const uint8_t* qs = (const uint8_t*) &qs_packed;
    const int qh = bq3->qh[iqs / 2];
    const int signs_packed_32 = get_int_b2(bq3->signs, iqs / 2);
    const uint8_t* signs_packed_8 = (const uint8_t*) &signs_packed_32;
    int sumi = 0;
#pragma unroll
    for (int l0 = 0; l0 < 8; l0 += 2) {
        const sycl::int2 grid_pos =
            sycl::int2(iq3s_grid[qs[l0 + 0] | ((qh << (8 - l0)) & 0x100)],
                       iq3s_grid[qs[l0 + 1] | ((qh << (7 - l0)) & 0x100)]);
        const int signs0 = swar_ne4(((signs_packed_8[l0 / 2] & 0x03) << 7) |
                ((signs_packed_8[l0 / 2] & 0x0C) << 21));
        const int signs1 = swar_ne4(((signs_packed_8[l0 / 2] & 0x30) << 3) |
                ((signs_packed_8[l0 / 2] & 0xC0) << 17));
        const int grid_l = swar_sub4(grid_pos.x() ^ signs0, signs0);
        const int grid_h = swar_sub4(grid_pos.y() ^ signs1, signs1);
        const int u0 = get_int_b4(bq8_1[iqs / 2].qs, l0 + 0);
        const int u1 = get_int_b4(bq8_1[iqs / 2].qs, l0 + 1);
        sumi = ggml_cuda_dp4a(grid_l, u0, sumi);
        sumi = ggml_cuda_dp4a(grid_h, u1, sumi);
    }
    sumi *= 1 + 2 * ((bq3->scales[iqs / 4] >> ((iqs << 1) & 0x04)) & 0x0F);
    const float d = sycl::vec<sycl::half, 1>(bq3->d)
                        .convert<float, sycl::rounding_mode::automatic>()[0] *
                    bq8_1[iqs / 2].ds[0];
    return d * sumi;
}

__dpct_inline__ float vec_dot_iq1_m_q8_1(const void *__restrict__ vbq,
                                         const block_q8_1 *__restrict__ bq8_1,
                                         const int &kbx, const int &iqs) {
    const block_iq1_m* bq1 = (const block_iq1_m*) vbq + kbx;
    const int qs_packed = get_int_b4(bq1->qs, iqs);
    const uint8_t* qs = (const uint8_t*) &qs_packed;
    int sumi[2] = {0, 0};
    float sumf[2] = {0.0f, 0.0f};
#pragma unroll
    for (int l0 = 0; l0 < 8; l0 += 2) {
        const int qhl = bq1->qh[2 * iqs + l0 / 4] >> (4 * ((l0 / 2) % 2));
        const int grid = iq1s_grid_gpu[qs[l0 / 2] | ((qhl & 0x07) << 8)];
        const int grid0 = (grid >> 0) & 0x0F0F0F0F;
        const int grid1 = (grid >> 4) & 0x0F0F0F0F;
        const int u0 = get_int_b4(bq8_1[iqs].qs, l0 + 0);
        const int u1 = get_int_b4(bq8_1[iqs].qs, l0 + 1);
        sumi[l0 / 4] = ggml_cuda_dp4a(grid0, u0, sumi[l0 / 4]);
        sumi[l0 / 4] = ggml_cuda_dp4a(grid1, u1, sumi[l0 / 4]);
        const float delta = -1.0f + IQ1M_DELTA - (qhl & 0x08) * (2.0f * IQ1M_DELTA / 0x08);
        int sumy = 0;
        sumy = ggml_cuda_dp4a(u0, 0x01010101, sumy);
        sumy = ggml_cuda_dp4a(u1, 0x01010101, sumy);
        sumf[l0 / 4] += delta * sumy;
    }
    const uint16_t* sc = (const uint16_t*) bq1->scales;
    iq1m_scale_t scale;
    scale.u16 = (sc[0] >> 12) | ((sc[1] >> 8) & 0x00F0) | ((sc[2] >> 4) & 0x0F00) | (sc[3] & 0xF000);
    const float d = sycl::vec<sycl::half, 1>(scale.f16)
                        .convert<float, sycl::rounding_mode::automatic>()[0] *
                    bq8_1[iqs].ds[0];
    const int tmp = sc[iqs / 2] >> (6 * (iqs % 2));
    const int sc0 = 2 * ((tmp >> 0) & 0x07) + 1;
    const int sc1 = 2 * ((tmp >> 3) & 0x07) + 1;
    return d * ((sumi[0] + sumf[0]) * sc0 + (sumi[1] + sumf[1]) * sc1);
}

// SYCL port: the IQ4_NL codebook in registers and its 2-byte-aligned weights as aligned loads. ggml reads 4 uint32 of
// the codebook from global memory per lookup and each weight int as two 16-bit loads (block_iq4_nl is 18 bytes, its
// qs 2-byte aligned); the expert down projection (IQ4_NL in 39 of 48 layers) was the largest expert kernel. Same ints
// out, so the dots are bitwise unchanged.
__dpct_inline__ uint32_t iq4nl_lut4(uint32_t q4) {   // four nibbles -> four kvalues_iq4nl bytes
    constexpr uint32_t t0 = 0xBFAD9881u, t1 = 0xF6EADDCFu, t2 = 0x26190D01u, t3 = 0x71594535u;
    uint32_t r = 0;
#pragma unroll
    for (int i = 0; i < 4; ++i) {
        const uint32_t n = (q4 >> (8 * i)) & 0xF;
        const uint32_t t = n < 4 ? t0 : n < 8 ? t1 : n < 12 ? t2 : t3;
        r |= ((t >> (8 * (n & 3))) & 0xFF) << (8 * i);
    }
    return r;
}
__dpct_inline__ sycl::int2 iq4nl_pair(int aux) {   // = get_int_from_table_16(aux)
    return sycl::int2((int) iq4nl_lut4((uint32_t) aux & 0x0f0f0f0fu), (int) iq4nl_lut4(((uint32_t) aux >> 4) & 0x0f0f0f0fu));
}
__dpct_inline__ sycl::int2 load8_a2(const uint8_t* p) {   // 8 bytes at a 2-byte-aligned address, as two ints
    const uintptr_t a = reinterpret_cast<uintptr_t>(p);
    const uint32_t* q = reinterpret_cast<const uint32_t*>(a & ~uintptr_t(3));
    const uint32_t w0 = q[0], w1 = q[1];
    if (!(a & 2)) return sycl::int2((int) w0, (int) w1);
    const uint32_t w2 = q[2];   // only when unaligned: it then holds needed bytes, so it cannot cross a page
    return sycl::int2((int) ((w0 >> 16) | (w1 << 16)), (int) ((w1 >> 16) | (w2 << 16)));
}
__dpct_inline__ int load4_a2(const uint8_t* p) {   // 4 bytes at a 2-byte-aligned address (= get_int_b2)
    const uintptr_t a = reinterpret_cast<uintptr_t>(p);
    const uint32_t* q = reinterpret_cast<const uint32_t*>(a & ~uintptr_t(3));
    const uint32_t w0 = q[0];
    if (!(a & 2)) return (int) w0;
    return (int) ((w0 >> 16) | (q[1] << 16));   // the second word holds needed bytes: page-safe
}
// SYCL port: the gate/up kernels can stage their format's codebook grid in local memory (STRATA_GRID_SLM=1, compile
// time). Measured slower and off: copying the grid into each of ~1,200 work-groups per call costs more than the
// lookups save (IQ2_S 52.5 -> 62.7 us, IQ3_S 60.8 -> 70.0, IQ3_XXS 60.3 -> 62.1 per call; unitrace 2026-10-01).
#ifndef STRATA_GRID_SLM
#define STRATA_GRID_SLM 0
#endif
#ifndef STRATA_IQ4NL_FAST
#define STRATA_IQ4NL_FAST 1   // compile-time: -DSTRATA_IQ4NL_FAST=0 for ggml's table/16-bit-load path
#endif

__dpct_inline__ float vec_dot_iq4_nl_q8_1(const void *__restrict__ vbq,
                                          const block_q8_1 *__restrict__ bq8_1,
                                          const int &kbx, const int &iqs) {
    const block_iq4_nl* bq4 = (const block_iq4_nl*) vbq + kbx;
    const int* q8 = (const int*) bq8_1->qs + iqs;
    int sumi = 0;
#if STRATA_IQ4NL_FAST
    const sycl::int2 qq = load8_a2(bq4->qs + 4 * iqs);
#endif
#pragma unroll
    for (int l = 0; l < 2; ++l) {
#if STRATA_IQ4NL_FAST
        const sycl::int2 v = iq4nl_pair(l ? qq.y() : qq.x());
#else
        const int aux_q4 = get_int_b2(bq4->qs, iqs + l);
        const sycl::int2 v = get_int_from_table_16(aux_q4, kvalues_iq4nl);
#endif
        sumi = ggml_cuda_dp4a(v.x(), q8[l + 0], sumi);
        sumi = ggml_cuda_dp4a(v.y(), q8[l + 4], sumi);
    }
    const float d = sycl::vec<sycl::half, 1>(bq4->d)
                        .convert<float, sycl::rounding_mode::automatic>()[0] *
                    bq8_1->ds[0];
    return d * sumi;
}

// IQ4_XS: 256 values as 8 sub-blocks of 32 (6-bit scale each); one call covers one sub-block (iqs = 4 * sub-block),
// and `bq8_1` is the super-block's first q8_1 block, so the call's activation is bq8_1[iqs / 4].  The GSQ-RCO IQ3_S
// file keeps one layer's routed gate/up experts in this format.
__dpct_inline__ float vec_dot_iq4_xs_q8_1(const void *__restrict__ vbq,
                                          const block_q8_1 *__restrict__ bq8_1,
                                          const int &kbx, const int &iqs) {
    const block_iq4_xs* bq4 = (const block_iq4_xs*) vbq + kbx;
    int sumi = 0;
#pragma unroll
    for (int j = 0; j < 4; ++j) {
        const int aux_q4 = get_int_b4(bq4->qs, iqs + j);
        const sycl::int2 v = get_int_from_table_16(aux_q4, kvalues_iq4nl);
        const int u0 = get_int_b4(bq8_1[iqs / 4].qs, j + 0);
        const int u1 = get_int_b4(bq8_1[iqs / 4].qs, j + 4);
        sumi = ggml_cuda_dp4a(v.x(), u0, sumi);
        sumi = ggml_cuda_dp4a(v.y(), u1, sumi);
    }
    const int ls = ((bq4->scales_l[iqs / 8] >> (iqs & 0x04)) & 0x0F) | (((bq4->scales_h >> (iqs / 2)) & 0x03) << 4);
    sumi *= ls - 32;
    const float d = sycl::vec<sycl::half, 1>(bq4->d)
                        .convert<float, sycl::rounding_mode::automatic>()[0] *
                    bq8_1[iqs / 4].ds[0];
    return d * sumi;
}

// ---------------------------------------------------------------- Unsloth's UD-Q4_K_XL experts
// Q4_K / Q5_K gate/up and Q5_1 / Q8_0 down: llama.cpp's vec_dot_*_q8_1 (vecdotq.cuh, VDR 2 each), transcribed; the
// Q5_1 min term is the one departure (below).  Via eddoursul/Strata 8029fa9 (iq_dot.cuh) and #255 (Q8_0,
// gopinath87607), which agree with llama.cpp and with each other.
constexpr int VDR_Q4_K = 2, VDR_Q5_K = 2, VDR_Q5_1 = 2, VDR_Q5_0 = 2, VDR_Q8_0 = 2;

__dpct_inline__ float vec_dot_q4_K_q8_1_impl_vmmq(
    const int *__restrict__ v, const int *__restrict__ u,
    const uint8_t *__restrict__ sc, const uint8_t *__restrict__ m,
    const sycl::half2 &dm4, const float *__restrict__ d8) {
    float sumf_d = 0.0f;
    float sumf_m = 0.0f;
#pragma unroll
    for (int i = 0; i < QR4_K; ++i) {
        const int v0i = (v[0] >> (4 * i)) & 0x0F0F0F0F;
        const int v1i = (v[1] >> (4 * i)) & 0x0F0F0F0F;
        const int dot1 = ggml_cuda_dp4a(v1i, u[2 * i + 1], ggml_cuda_dp4a(v0i, u[2 * i + 0], 0));
        const int dot2 = ggml_cuda_dp4a(0x01010101, u[2 * i + 1], ggml_cuda_dp4a(0x01010101, u[2 * i + 0], 0));
        sumf_d += d8[i] * (dot1 * sc[i]);
        sumf_m += d8[i] * (dot2 * m[i]);   // the min times the sum of the QUANTIZED activations
    }
    const sycl::float2 dm4f =
        dm4.template convert<float, sycl::rounding_mode::automatic>();
    return dm4f.x() * sumf_d - dm4f.y() * sumf_m;
}
__dpct_inline__ float vec_dot_q5_K_q8_1_impl_vmmq(
    const int *__restrict__ vl, const int *__restrict__ vh,
    const int *__restrict__ u, const uint8_t *__restrict__ sc,
    const uint8_t *__restrict__ m, const sycl::half2 &dm5,
    const float *__restrict__ d8) {
    float sumf_d = 0.0f;
    float sumf_m = 0.0f;
#pragma unroll
    for (int i = 0; i < QR5_K; ++i) {
        const int vl0i = (vl[0] >> (4 * i)) & 0x0F0F0F0F;
        const int vl1i = (vl[1] >> (4 * i)) & 0x0F0F0F0F;
        const int vh0i = ((vh[0] >> i) << 4) & 0x10101010;
        const int vh1i = ((vh[1] >> i) << 4) & 0x10101010;
        const int v0i = vl0i | vh0i;
        const int v1i = vl1i | vh1i;
        const int dot1 = ggml_cuda_dp4a(v0i, u[2 * i + 0], ggml_cuda_dp4a(v1i, u[2 * i + 1], 0));
        const int dot2 = ggml_cuda_dp4a(0x01010101, u[2 * i + 0], ggml_cuda_dp4a(0x01010101, u[2 * i + 1], 0));
        sumf_d += d8[i] * (dot1 * sc[i]);
        sumf_m += d8[i] * (dot2 * m[i]);
    }
    const sycl::float2 dm5f =
        dm5.template convert<float, sycl::rounding_mode::automatic>();
    return dm5f.x() * sumf_d - dm5f.y() * sumf_m;
}
// the 6-bit scales and mins of the 32-value group pair bq8_offset / 2, branchless (llama.cpp; shared by Q4_K, Q5_K)
__dpct_inline__ void k_scale_min(const uint8_t *scales8, int bq8_offset,
                                 uint16_t aux[2]) {
    const uint16_t* scales = (const uint16_t*) scales8;
    const int j = bq8_offset / 2;
    const int jm = j & 1;
    const uint32_t s0 = scales[jm + 0];
    const uint32_t s2 = scales[jm + 2];
    const uint32_t s4 = scales[jm + 4];
    const uint32_t hi = (uint32_t) -(int32_t) (j >= 2);
    aux[0] = (uint16_t) (((s0 & 0x3f3f) & ~hi) | ((((s4 >> 0) & 0x0f0f) | ((s0 & 0xc0c0) >> 2)) & hi));
    aux[1] = (uint16_t) (((s2 & 0x3f3f) & ~hi) | ((((s4 >> 4) & 0x0f0f) | ((s2 & 0xc0c0) >> 2)) & hi));
}
__dpct_inline__ float vec_dot_q4_K_q8_1(const void *__restrict__ vbq,
                                        const block_q8_1 *__restrict__ bq8_1,
                                        const int &kbx, const int &iqs) {
    const block_q4_K* bq4_K = (const block_q4_K*) vbq + kbx;
    int v[2];
    int u[2 * QR4_K];
    float d8[QR4_K];
    const int bq8_offset = QR4_K * ((iqs / 2) / (QI8_1 / 2));
    const int* q4 = (const int*) (bq4_K->qs + 16 * bq8_offset + 4 * ((iqs / 2) % 4));
    v[0] = q4[0];
    v[1] = q4[4];
    uint16_t aux[2];
    k_scale_min(bq4_K->scales, bq8_offset, aux);
    const uint8_t* sc = (const uint8_t*) aux;
    const uint8_t* m = sc + 2;
#pragma unroll
    for (int i = 0; i < QR4_K; ++i) {
        const block_q8_1* bq8i = bq8_1 + bq8_offset + i;
        d8[i] = bq8i->ds[0];
        const int* q8 = (const int*) bq8i->qs + ((iqs / 2) % 4);
        u[2 * i + 0] = q8[0];
        u[2 * i + 1] = q8[4];
    }
    return vec_dot_q4_K_q8_1_impl_vmmq(v, u, sc, m, bq4_K->dm, d8);
}
__dpct_inline__ float vec_dot_q5_K_q8_1(const void *__restrict__ vbq,
                                        const block_q8_1 *__restrict__ bq8_1,
                                        const int &kbx, const int &iqs) {
    const block_q5_K* bq5_K = (const block_q5_K*) vbq + kbx;
    int vl[2];
    int vh[2];
    int u[2 * QR5_K];
    float d8[QR5_K];
    const int bq8_offset = QR5_K * ((iqs / 2) / (QI8_1 / 2));
    const int* ql = (const int*) (bq5_K->qs + 16 * bq8_offset + 4 * ((iqs / 2) % 4));
    const int* qh = (const int*) (bq5_K->qh + 4 * ((iqs / 2) % 4));
    vl[0] = ql[0];
    vl[1] = ql[4];
    vh[0] = qh[0] >> bq8_offset;
    vh[1] = qh[4] >> bq8_offset;
    uint16_t aux[2];
    k_scale_min(bq5_K->scales, bq8_offset, aux);
    const uint8_t* sc = (const uint8_t*) aux;
    const uint8_t* m = sc + 2;
#pragma unroll
    for (int i = 0; i < QR5_K; ++i) {
        const block_q8_1* bq8i = bq8_1 + bq8_offset + i;
        d8[i] = bq8i->ds[0];
        const int* q8 = (const int*) bq8i->qs + ((iqs / 2) % 4);
        u[2 * i + 0] = q8[0];
        u[2 * i + 1] = q8[4];
    }
    return vec_dot_q5_K_q8_1_impl_vmmq(vl, vh, u, sc, m, bq5_K->dm, d8);
}
// Q5_1: llama.cpp's integer chain, but the min term multiplies the sum of the QUANTIZED activations (dp4a with
// 0x01010101, times d8) instead of the q8_1 block's `ds.y`, which our quantizer (like llama.cpp's) fills with the sum
// of the ORIGINAL activations.  That is ggml-cpu's convention (its q8_1 `s` is d * sum(q)) and the one the K-quant
// mins above use; the scaled and the min term then see the same activation (eddoursul/Strata measured 1.1-1.2%
// against 1.9% relative error per expert).  Result: sumi * (d5 * d8) + sumu * (m5 * d8).
// Q5_0: llama.cpp's integer chain verbatim (vecdotq.cuh vec_dot_q5_0_q8_1_impl) -- symmetric (one delta, no
// learned min), so unlike Q5_1 there is no min-term rounding choice to make: the constant -16 per-weight offset
// is accounted for via the q8_1 block's `ds.y` (sum of the ORIGINAL activations), exactly as upstream.
__dpct_inline__ float vec_dot_q5_0_q8_1(const void *__restrict__ vbq,
                                        const block_q8_1 *__restrict__ bq8_1,
                                        const int &kbx, const int &iqs) {
    const block_q5_0* bq5_0 = (const block_q5_0*) vbq + kbx;
    int sumi = 0;
#pragma unroll
    for (int i = 0; i < VDR_Q5_0; ++i) {
        const int vl = get_int_b2(bq5_0->qs, iqs + i);
        const int vh = get_int_b2(bq5_0->qh, 0) >> (4 * (iqs + i));
        const int u0 = get_int_b4(bq8_1->qs, iqs + i), u1 = get_int_b4(bq8_1->qs, iqs + i + QI5_0);
        int vi0 = (vl >> 0) & 0x0F0F0F0F;
        vi0 |= (vh << 4) & 0x00000010;
        vi0 |= (vh << 11) & 0x00001000;
        vi0 |= (vh << 18) & 0x00100000;
        vi0 |= (vh << 25) & 0x10000000;
        sumi = ggml_cuda_dp4a(vi0, u0, sumi);
        int vi1 = (vl >> 4) & 0x0F0F0F0F;
        vi1 |= (vh >> 12) & 0x00000010;
        vi1 |= (vh >> 5) & 0x00001000;
        vi1 |= (vh << 2) & 0x00100000;
        vi1 |= (vh << 9) & 0x10000000;
        sumi = ggml_cuda_dp4a(vi1, u1, sumi);
    }
    const float d5 = sycl::vec<sycl::half, 1>(bq5_0->d)
                         .convert<float, sycl::rounding_mode::automatic>()[0];
    const float d8 = bq8_1->ds[0];
    const float s8 = bq8_1->ds[1];
    return d5 * (sumi * d8 - (16.0f * VDR_Q5_0 / QI5_0) * s8);
}
__dpct_inline__ float vec_dot_q5_1_q8_1(const void *__restrict__ vbq,
                                        const block_q8_1 *__restrict__ bq8_1,
                                        const int &kbx, const int &iqs) {
    const block_q5_1* bq5_1 = (const block_q5_1*) vbq + kbx;
    int sumi = 0, sumu = 0;
#pragma unroll
    for (int i = 0; i < VDR_Q5_1; ++i) {
        const int vl = get_int_b4(bq5_1->qs, iqs + i);
        const int vh = get_int_b4(bq5_1->qh, 0) >> (4 * (iqs + i));
        const int u0 = get_int_b4(bq8_1->qs, iqs + i), u1 = get_int_b4(bq8_1->qs, iqs + i + QI5_1);
        int vi0 = (vl >> 0) & 0x0F0F0F0F;
        vi0 |= (vh << 4) & 0x00000010;
        vi0 |= (vh << 11) & 0x00001000;
        vi0 |= (vh << 18) & 0x00100000;
        vi0 |= (vh << 25) & 0x10000000;
        sumi = ggml_cuda_dp4a(vi0, u0, sumi);
        int vi1 = (vl >> 4) & 0x0F0F0F0F;
        vi1 |= (vh >> 12) & 0x00000010;
        vi1 |= (vh >> 5) & 0x00001000;
        vi1 |= (vh << 2) & 0x00100000;
        vi1 |= (vh << 9) & 0x10000000;
        sumi = ggml_cuda_dp4a(vi1, u1, sumi);
        sumu = ggml_cuda_dp4a(0x01010101, u1, ggml_cuda_dp4a(0x01010101, u0, sumu));
    }
    const sycl::float2 dm5 =
        (bq5_1->dm).template convert<float, sycl::rounding_mode::automatic>();
    const float d8 = bq8_1->ds[0];
    return sumi * (dm5.x() * d8) + sumu * (dm5.y() * d8);
}
__dpct_inline__ float vec_dot_q8_0_q8_1(const void *__restrict__ vbq,
                                        const block_q8_1 *__restrict__ bq8_1,
                                        const int &kbx, const int &iqs) {
    const block_q8_0* bq8_0 = (const block_q8_0*) vbq + kbx;
    int sumi = 0;
#pragma unroll
    for (int i = 0; i < VDR_Q8_0; ++i)
        sumi = ggml_cuda_dp4a(get_int_b2(bq8_0->qs, iqs + i), get_int_b4(bq8_1->qs, iqs + i), sumi);
    const float d8_0 = sycl::vec<sycl::half, 1>(bq8_0->d)
                           .convert<float, sycl::rounding_mode::automatic>()[0],
                d8_1 = bq8_1->ds[0];
    return d8_0 * d8_1 * ((float) sumi);
}

// ---------------------------------------------------------------- the formats
// qk = values per block, ipb = dot calls per block (qi / vdr), step = the iqs stride between calls.
template<int TY> 
struct Fmt;
template<> struct Fmt<16> { static constexpr int qk = 256, ipb = 8, step = 2;
    static float dot(const void* v, const block_q8_1* y, int kbx, int iqs) { return vec_dot_iq2_xxs_q8_1(v, y, kbx, iqs); } };
template<> struct Fmt<17> { static constexpr int qk = 256, ipb = 8, step = 2;
    static float dot(const void* v, const block_q8_1* y, int kbx, int iqs) { return vec_dot_iq2_xs_q8_1(v, y, kbx, iqs); } };
template<> struct Fmt<18> { static constexpr int qk = 256, ipb = 8, step = 2;
    static float dot(const void* v, const block_q8_1* y, int kbx, int iqs) { return vec_dot_iq3_xxs_q8_1(v, y, kbx, iqs); } };
template<> struct Fmt<20> { static constexpr int qk = 32, ipb = 2, step = 2;
    static float dot(const void* v, const block_q8_1* y, int kbx, int iqs) { return vec_dot_iq4_nl_q8_1(v, y, kbx, iqs); } };
template<> struct Fmt<21> { static constexpr int qk = 256, ipb = 8, step = 2;
    static float dot(const void* v, const block_q8_1* y, int kbx, int iqs) { return vec_dot_iq3_s_q8_1(v, y, kbx, iqs); } };
template<> struct Fmt<23> { static constexpr int qk = 256, ipb = 8, step = 4;
    static float dot(const void* v, const block_q8_1* y, int kbx, int iqs) { return vec_dot_iq4_xs_q8_1(v, y, kbx, iqs); } };
template<> struct Fmt<22> { static constexpr int qk = 256, ipb = 8, step = 2;
    static float dot(const void* v, const block_q8_1* y, int kbx, int iqs) { return vec_dot_iq2_s_q8_1(v, y, kbx, iqs); } };
template<> struct Fmt<29> { static constexpr int qk = 256, ipb = 8, step = 1;
    static float dot(const void* v, const block_q8_1* y, int kbx, int iqs) { return vec_dot_iq1_m_q8_1(v, y, kbx, iqs); } };
template<> struct Fmt<42> { static constexpr int qk = 64, ipb = 2, step = 1;
    static float dot(const void* v, const block_q8_1* y, int kbx, int iqs) { return vec_dot_q2_0_q8_1(v, y, kbx, iqs); } };
template<> struct Fmt<12> { static constexpr int qk = 256, ipb = QI4_K / VDR_Q4_K, step = VDR_Q4_K;
    static float dot(const void* v, const block_q8_1* y, int kbx, int iqs) { return vec_dot_q4_K_q8_1(v, y, kbx, iqs); } };
template<> struct Fmt<13> { static constexpr int qk = 256, ipb = QI5_K / VDR_Q5_K, step = VDR_Q5_K;
    static float dot(const void* v, const block_q8_1* y, int kbx, int iqs) { return vec_dot_q5_K_q8_1(v, y, kbx, iqs); } };
template<> struct Fmt<7> { static constexpr int qk = 32, ipb = QI5_1 / VDR_Q5_1, step = VDR_Q5_1;
    static float dot(const void* v, const block_q8_1* y, int kbx, int iqs) { return vec_dot_q5_1_q8_1(v, y, kbx, iqs); } };
template<> struct Fmt<6> { static constexpr int qk = 32, ipb = QI5_0 / VDR_Q5_0, step = VDR_Q5_0;
    static float dot(const void* v, const block_q8_1* y, int kbx, int iqs) { return vec_dot_q5_0_q8_1(v, y, kbx, iqs); } };
template<> struct Fmt<8> { static constexpr int qk = 32, ipb = QI8_0 / VDR_Q8_0, step = VDR_Q8_0;
    static float dot(const void* v, const block_q8_1* y, int kbx, int iqs) { return vec_dot_q8_0_q8_1(v, y, kbx, iqs); } };

// The formats of each role, one list each so a type cannot be in one switch and missing from another.  Every
// entry is a kernel template for each CUDA architecture of the build, hence two lists rather than one.
#define STRATA_GU_FMTS(X) X(16) X(17) X(18) X(21) X(22) X(23) X(29) X(42) X(12) X(13) X(6) X(8)
#define STRATA_D_FMTS(X) X(20) X(23) X(42) X(7) X(6) X(8)
#define STRATA_MMVQ_FMTS(X) X(16) X(17) X(18) X(20) X(21) X(22) X(23) X(29) X(42) X(12) X(13) X(7) X(6) X(8)

__dpct_inline__ float warp_sum(float v) {
#pragma unroll
    /*
    DPCT1108: '__shfl_xor_sync' was migrated with the experimental feature
    masked sub_group function which may not be supported by all compilers or
    runtimes. You may need to adjust the code.
    */
    for (int o = 16; o > 0; o >>= 1) v +=
        dpct::experimental::permute_sub_group_by_xor(
            0xffffffffu, sycl::ext::oneapi::this_work_item::get_sub_group(), v,
            o);
    return v;
}

// SYCL port: lanes per row for the grouped expert kernels. A 2560-wide row is 80 (block, part) calls; over 32
// lanes that is 2.5 calls each and a 5-step reduction - the work per sub-group is too small for the fixed cost.
// STRATA_EXPERT_LANES (compile time) lanes share a row; 256 / that many rows per work-group.
#ifndef STRATA_EXPERT_LANES
#define STRATA_EXPERT_LANES 8
#endif
constexpr int kExpertLanes = STRATA_EXPERT_LANES;
constexpr int kExpertRows = 256 / kExpertLanes;
template <int LANES>
__dpct_inline__ float lanes_sum(float v) {
#pragma unroll
    for (int o = LANES / 2; o > 0; o >>= 1)
        v += dpct::experimental::permute_sub_group_by_xor(0xffffffffu, sycl::ext::oneapi::this_work_item::get_sub_group(), v, o);
    return v;
}
template <int TY, int LANES>
__dpct_inline__ float row_dot_lanes(const uint8_t *row, const block_q8_1 *x, int nb, int sub) {
    using F = Fmt<TY>;
    float s = 0.0f;
    for (int k = sub; k < nb * F::ipb; k += LANES) {
        const int kbx = k / F::ipb, iqs = F::step * (k % F::ipb);
        s += F::dot(row, x + kbx * (F::qk / 32), kbx, iqs);
    }
    return lanes_sum<LANES>(s);
}

// One row against one q8_1 activation, the whole warp: call k = (block, part) is lane-strided.
template <int TY>
__dpct_inline__ float row_dot(const uint8_t *row, const block_q8_1 *x, int nb,
                              int lane) {
    using F = Fmt<TY>;
    float s = 0.0f;
    for (int k = lane; k < nb * F::ipb; k += 32) {
        const int kbx = k / F::ipb, iqs = F::step * (k % F::ipb);
        /*
        DPCT1084: The function call "Fmt::dot" has multiple migration
        results in different template instantiations that could not be unified.
        You may need to adjust the code.
        */
        s += F::dot(row, x + kbx * (F::qk / 32), kbx, iqs);
    }
    return warp_sum(s);
}

template <int TY>
__dpct_inline__ void
mmvq_kernel(const uint8_t *__restrict__ w, size_t row_bytes,
            const block_q8_1 *__restrict__ x, float *__restrict__ y, int n_in,
            int n_out, int ncols) {
    auto item_ct1 = sycl::ext::oneapi::this_work_item::get_nd_item<3>();
    const int row = item_ct1.get_group(2) * 4 + item_ct1.get_local_id(1);
    if (row >= n_out) return;
    const int lane = item_ct1.get_local_id(2);
    const int nb = n_in / Fmt<TY>::qk;
    const uint8_t* wr = w + (size_t) row * row_bytes;
    for (int c = 0; c < ncols; ++c) {
        const float s =
            row_dot<TY>(wr, x + (size_t)c * (n_in / 32), nb, lane);
        if (lane == 0) y[(size_t) c * n_out + row] = s;
    }
}

// ---------------------------------------------------------------- decode once, apply to every column
// The kernels above call Fmt<TY>::dot once per (call, column): each column re-reads the weight words and redoes
// the grid lookups and sign unpacking.  Here each dot is split, as native_mmvq.cu's multi-column traits are, into
// `load` (everything that depends only on the weight: the signed grid words, the integer scales, the fp16 block
// scale as a float) and `apply` (the activation loads, the dp4a chain in the same order, the same integer scale
// step and the same float expression).  `apply(load(...))` does the dot's integer and float operations in the same
// order on the same values, so a column of the kernels below is BITWISE equal to the same column of mmvq_kernel /
// native_gu_kernel / native_down_kernel (iq_multi_parity checks it; STRATA_OLD_IQ_MMVQ=1 keeps the old kernels).
template<int TY> struct Split;
// Formats with a Split below take the decode-once kernels; the others (Q4_K, Q5_K, Q5_1, Q8_0: UD-Q4_K_XL) the
// per-entry ones, which call Fmt<TY>::dot per column exactly as before #242 (the launchers test kSplit at compile
// time, so the multi kernels are never instantiated for a type without a Split).
template<int TY> inline constexpr bool kSplit = false;
template<> inline constexpr bool kSplit<16> = true;
template<> struct Split<16> {   // IQ2_XXS
    struct W { int g[8]; int ls; float dw; };
    static W load(const void* __restrict__ vbq, int kbx, int iqs) {
        const block_iq2_xxs* bq2 = (const block_iq2_xxs*) vbq + kbx;
        const int q2 = get_int_b2(bq2->qs, iqs);
        const uint8_t* aux8 = (const uint8_t*) &q2;
        const uint32_t aux32 = get_int_b2(bq2->qs, iqs + 1);
        W r;
#pragma unroll
        for (int k0 = 0; k0 < 8; k0 += 2) {
            const sycl::uint2 grid_pos =
                ((const sycl::uint2 *)iq2xxs_grid)[aux8[k0 / 2]];
            const uint32_t signs = unpack_ksigns(aux32 >> (7 * k0 / 2));
            const int signs0 = dpct::vectorized_binary<sycl::uchar4>(
                signs & 0x08040201, 0, std::not_equal_to<>());
            r.g[k0 + 0] = dpct::vectorized_binary<sycl::uchar4>(
                grid_pos.x() ^ signs0, signs0, std::minus<>());
            const int signs1 = dpct::vectorized_binary<sycl::uchar4>(
                signs & 0x80402010, 0, std::not_equal_to<>());
            r.g[k0 + 1] = dpct::vectorized_binary<sycl::uchar4>(
                grid_pos.y() ^ signs1, signs1, std::minus<>());
        }
        r.ls = aux32 >> 27 | 1;
        r.dw = sycl::vec<sycl::half, 1>(bq2->d)
                   .convert<float, sycl::rounding_mode::automatic>()[0];
        return r;
    }
    static float apply(const W& r, const block_q8_1* __restrict__ bq8_1, int iqs) {
        int sumi = 0;
#pragma unroll
        for (int j = 0; j < 8; ++j) sumi = ggml_cuda_dp4a(r.g[j], get_int_b4(bq8_1[iqs / 2].qs, j), sumi);
        sumi = sumi * r.ls / 8;
        const float d = r.dw * bq8_1[iqs / 2].ds[0];
        return d * sumi;
    }
};
// IQ2_XS and IQ2_S share the apply: two half sums, two 4-bit scales
struct SplitLs2 {
    struct W { int g[8]; int ls0, ls1; float dw; };
    static float apply(const W& r, const block_q8_1* __restrict__ bq8_1, int iqs) {
        int sumi0 = 0, sumi1 = 0;
#pragma unroll
        for (int j = 0; j < 4; ++j) sumi0 = ggml_cuda_dp4a(r.g[j], get_int_b4(bq8_1[iqs / 2].qs, j), sumi0);
#pragma unroll
        for (int j = 4; j < 8; ++j) sumi1 = ggml_cuda_dp4a(r.g[j], get_int_b4(bq8_1[iqs / 2].qs, j), sumi1);
        const int sumi = (sumi0 * r.ls0 + sumi1 * r.ls1 + (sumi0 + sumi1) / 2) / 4;
        const float d = r.dw * bq8_1[iqs / 2].ds[0];
        return d * sumi;
    }
};
template<> inline constexpr bool kSplit<17> = true;
template<> struct Split<17> : SplitLs2 {   // IQ2_XS
    static W load(const void* __restrict__ vbq, int kbx, int iqs) {
        const block_iq2_xs* bq2 = (const block_iq2_xs*) vbq + kbx;
        // SYCL port: the codes by shifts, not through a uint16_t* to an int2 (see vec_dot_iq2_xs_q8_1)
        const uint32_t q2_lo = (uint32_t) get_int_b2(bq2->qs, iqs + 0), q2_hi = (uint32_t) get_int_b2(bq2->qs, iqs + 1);
        const uint16_t q2[4] = {(uint16_t) q2_lo, (uint16_t) (q2_lo >> 16), (uint16_t) q2_hi, (uint16_t) (q2_hi >> 16)};
        W r;
        r.ls0 = bq2->scales[iqs / 2] & 0x0F;
        r.ls1 = bq2->scales[iqs / 2] >> 4;
#pragma unroll
        for (int l0 = 0; l0 < 8; l0 += 2) {
            const sycl::uint2 grid_pos =
                ((const sycl::uint2 *)iq2xs_grid)[q2[l0 / 2] & 0x1FF];
            const uint32_t signs = unpack_ksigns(q2[l0 / 2] >> 9);
            const int signs0 = dpct::vectorized_binary<sycl::uchar4>(
                signs & 0x08040201, 0, std::not_equal_to<>());
            r.g[l0 + 0] = dpct::vectorized_binary<sycl::uchar4>(
                grid_pos.x() ^ signs0, signs0, std::minus<>());
            const int signs1 = dpct::vectorized_binary<sycl::uchar4>(
                signs & 0x80402010, 0, std::not_equal_to<>());
            r.g[l0 + 1] = dpct::vectorized_binary<sycl::uchar4>(
                grid_pos.y() ^ signs1, signs1, std::minus<>());
        }
        r.dw = sycl::vec<sycl::half, 1>(bq2->d)
                   .convert<float, sycl::rounding_mode::automatic>()[0];
        return r;
    }
};
template<> inline constexpr bool kSplit<22> = true;
template<> struct Split<22> : SplitLs2 {   // IQ2_S
    static W load(const void* __restrict__ vbq, int kbx, int iqs) {
        const block_iq2_s* bq2 = (const block_iq2_s*) vbq + kbx;
        const int qs_packed = get_int_b2(bq2->qs, iqs / 2);
        const uint8_t* qs = (const uint8_t*) &qs_packed;
        const int qh = bq2->qh[iqs / 2];
        const int signs_packed_32 = get_int_b2(bq2->qs, QK_K / 32 + iqs / 2);
        const uint8_t* signs_packed_8 = (const uint8_t*) &signs_packed_32;
        W r;
        r.ls0 = bq2->scales[iqs / 2] & 0x0F;
        r.ls1 = bq2->scales[iqs / 2] >> 4;
#pragma unroll
        for (int l0 = 0; l0 < 8; l0 += 2) {
            const int* grid_pos = (const int*) (iq2s_grid + (qs[l0 / 2] | ((qh << (8 - l0)) & 0x300)));
            const int signs0 = dpct::vectorized_binary<sycl::uchar4>(
                ((signs_packed_8[l0 / 2] & 0x03) << 7) |
                    ((signs_packed_8[l0 / 2] & 0x0C) << 21),
                0x00000000, std::not_equal_to<>());
            const int signs1 = dpct::vectorized_binary<sycl::uchar4>(
                ((signs_packed_8[l0 / 2] & 0x30) << 3) |
                    ((signs_packed_8[l0 / 2] & 0xC0) << 17),
                0x00000000, std::not_equal_to<>());
            r.g[l0 + 0] = dpct::vectorized_binary<sycl::uchar4>(
                grid_pos[0] ^ signs0, signs0, std::minus<>());
            r.g[l0 + 1] = dpct::vectorized_binary<sycl::uchar4>(
                grid_pos[1] ^ signs1, signs1, std::minus<>());
        }
        r.dw = sycl::vec<sycl::half, 1>(bq2->d)
                   .convert<float, sycl::rounding_mode::automatic>()[0];
        return r;
    }
};
template<> inline constexpr bool kSplit<18> = true;
template<> struct Split<18> {   // IQ3_XXS
    struct W { int g[8]; int ls; float dw; };
    static W load(const void* __restrict__ vbq, int kbx, int iqs) {
        const block_iq3_xxs* bq3 = (const block_iq3_xxs*) vbq + kbx;
        const sycl::int2 q3_packed =
            sycl::int2(get_int_b2(bq3->qs, iqs), get_int_b2(bq3->qs, iqs + 1));
        const uint8_t* q3 = (const uint8_t*) &q3_packed;
        const uint32_t aux32 = get_int_b2(bq3->qs, QK_K / 16 + iqs / 2);
        W r;
#pragma unroll
        for (int l0 = 0; l0 < 8; l0 += 2) {
            const sycl::int2 grid_pos =
                sycl::int2(iq3xxs_grid[q3[l0 + 0]], iq3xxs_grid[q3[l0 + 1]]);
            const uint32_t signs = unpack_ksigns(aux32 >> (7 * l0 / 2));
            const int signs0 = dpct::vectorized_binary<sycl::uchar4>(
                signs & 0x08040201, 0, std::not_equal_to<>());
            r.g[l0 + 0] = dpct::vectorized_binary<sycl::uchar4>(
                grid_pos.x() ^ signs0, signs0, std::minus<>());
            const int signs1 = dpct::vectorized_binary<sycl::uchar4>(
                signs & 0x80402010, 0, std::not_equal_to<>());
            r.g[l0 + 1] = dpct::vectorized_binary<sycl::uchar4>(
                grid_pos.y() ^ signs1, signs1, std::minus<>());
        }
        r.ls = aux32 >> 28;
        r.dw = sycl::vec<sycl::half, 1>(bq3->d)
                   .convert<float, sycl::rounding_mode::automatic>()[0];
        return r;
    }
    static float apply(const W& r, const block_q8_1* __restrict__ bq8_1, int iqs) {
        int sumi = 0;
#pragma unroll
        for (int j = 0; j < 8; ++j) sumi = ggml_cuda_dp4a(r.g[j], get_int_b4(bq8_1[iqs / 2].qs, j), sumi);
        sumi = (r.ls * sumi + sumi / 2) / 2;
        const float d = r.dw * bq8_1[iqs / 2].ds[0];
        return d * sumi;
    }
};
template<> inline constexpr bool kSplit<21> = true;
template<> struct Split<21> {   // IQ3_S
    struct W { int g[8]; int ls; float dw; };
    static W load(const void* __restrict__ vbq, int kbx, int iqs) {
        const block_iq3_s* bq3 = (const block_iq3_s*) vbq + kbx;
        const sycl::int2 qs_packed = sycl::int2(get_int_b2(bq3->qs, iqs + 0),
                                                get_int_b2(bq3->qs, iqs + 1));
        const uint8_t* qs = (const uint8_t*) &qs_packed;
        const int qh = bq3->qh[iqs / 2];
        const int signs_packed_32 = get_int_b2(bq3->signs, iqs / 2);
        const uint8_t* signs_packed_8 = (const uint8_t*) &signs_packed_32;
        W r;
#pragma unroll
        for (int l0 = 0; l0 < 8; l0 += 2) {
            const sycl::int2 grid_pos =
                sycl::int2(iq3s_grid[qs[l0 + 0] | ((qh << (8 - l0)) & 0x100)],
                           iq3s_grid[qs[l0 + 1] | ((qh << (7 - l0)) & 0x100)]);
            const int signs0 = dpct::vectorized_binary<sycl::uchar4>(
                ((signs_packed_8[l0 / 2] & 0x03) << 7) |
                    ((signs_packed_8[l0 / 2] & 0x0C) << 21),
                0x00000000, std::not_equal_to<>());
            const int signs1 = dpct::vectorized_binary<sycl::uchar4>(
                ((signs_packed_8[l0 / 2] & 0x30) << 3) |
                    ((signs_packed_8[l0 / 2] & 0xC0) << 17),
                0x00000000, std::not_equal_to<>());
            r.g[l0 + 0] = dpct::vectorized_binary<sycl::uchar4>(
                grid_pos.x() ^ signs0, signs0, std::minus<>());
            r.g[l0 + 1] = dpct::vectorized_binary<sycl::uchar4>(
                grid_pos.y() ^ signs1, signs1, std::minus<>());
        }
        r.ls = 1 + 2 * ((bq3->scales[iqs / 4] >> ((iqs << 1) & 0x04)) & 0x0F);
        r.dw = sycl::vec<sycl::half, 1>(bq3->d)
                   .convert<float, sycl::rounding_mode::automatic>()[0];
        return r;
    }
    static float apply(const W& r, const block_q8_1* __restrict__ bq8_1, int iqs) {
        int sumi = 0;
#pragma unroll
        for (int j = 0; j < 8; ++j) sumi = ggml_cuda_dp4a(r.g[j], get_int_b4(bq8_1[iqs / 2].qs, j), sumi);
        sumi *= r.ls;
        const float d = r.dw * bq8_1[iqs / 2].ds[0];
        return d * sumi;
    }
};
template<> inline constexpr bool kSplit<29> = true;
template<> struct Split<29> {   // IQ1_M
    struct W { int g[8]; float delta[4]; int sc0, sc1; float dw; };
    static W load(const void* __restrict__ vbq, int kbx, int iqs) {
        const block_iq1_m* bq1 = (const block_iq1_m*) vbq + kbx;
        const int qs_packed = get_int_b4(bq1->qs, iqs);
        const uint8_t* qs = (const uint8_t*) &qs_packed;
        W r;
#pragma unroll
        for (int l0 = 0; l0 < 8; l0 += 2) {
            const int qhl = bq1->qh[2 * iqs + l0 / 4] >> (4 * ((l0 / 2) % 2));
            const int grid = iq1s_grid_gpu[qs[l0 / 2] | ((qhl & 0x07) << 8)];
            r.g[l0 + 0] = (grid >> 0) & 0x0F0F0F0F;
            r.g[l0 + 1] = (grid >> 4) & 0x0F0F0F0F;
            r.delta[l0 / 2] = -1.0f + IQ1M_DELTA - (qhl & 0x08) * (2.0f * IQ1M_DELTA / 0x08);
        }
        const uint16_t* sc = (const uint16_t*) bq1->scales;
        iq1m_scale_t scale;
        scale.u16 = (sc[0] >> 12) | ((sc[1] >> 8) & 0x00F0) | ((sc[2] >> 4) & 0x0F00) | (sc[3] & 0xF000);
        r.dw = sycl::vec<sycl::half, 1>(scale.f16)
                   .convert<float, sycl::rounding_mode::automatic>()[0];
        const int tmp = sc[iqs / 2] >> (6 * (iqs % 2));
        r.sc0 = 2 * ((tmp >> 0) & 0x07) + 1;
        r.sc1 = 2 * ((tmp >> 3) & 0x07) + 1;
        return r;
    }
    static float apply(const W& r, const block_q8_1* __restrict__ bq8_1, int iqs) {
        int sumi[2] = {0, 0};
        float sumf[2] = {0.0f, 0.0f};
#pragma unroll
        for (int l0 = 0; l0 < 8; l0 += 2) {
            const int u0 = get_int_b4(bq8_1[iqs].qs, l0 + 0);
            const int u1 = get_int_b4(bq8_1[iqs].qs, l0 + 1);
            sumi[l0 / 4] = ggml_cuda_dp4a(r.g[l0 + 0], u0, sumi[l0 / 4]);
            sumi[l0 / 4] = ggml_cuda_dp4a(r.g[l0 + 1], u1, sumi[l0 / 4]);
            int sumy = 0;
            sumy = ggml_cuda_dp4a(u0, 0x01010101, sumy);
            sumy = ggml_cuda_dp4a(u1, 0x01010101, sumy);
            sumf[l0 / 4] += r.delta[l0 / 2] * sumy;
        }
        const float d = r.dw * bq8_1[iqs].ds[0];
        return d * ((sumi[0] + sumf[0]) * r.sc0 + (sumi[1] + sumf[1]) * r.sc1);
    }
};
template<> inline constexpr bool kSplit<20> = true;
template<> struct Split<20> {   // IQ4_NL
    struct W { sycl::int2 v[2]; float dw; };
    static W load(const void* __restrict__ vbq, int kbx, int iqs) {
        const block_iq4_nl* bq4 = (const block_iq4_nl*) vbq + kbx;
        W r;
#pragma unroll
        for (int l = 0; l < 2; ++l) r.v[l] = iq4nl_pair(get_int_b2(bq4->qs, iqs + l));   // SYCL port: = get_int_from_table_16(q)
        r.dw = sycl::vec<sycl::half, 1>(bq4->d)
                   .convert<float, sycl::rounding_mode::automatic>()[0];
        return r;
    }
    static float apply(const W& r, const block_q8_1* __restrict__ bq8_1, int iqs) {
        const int* q8 = (const int*) bq8_1->qs + iqs;
        int sumi = 0;
#pragma unroll
        for (int l = 0; l < 2; ++l) {
            sumi = ggml_cuda_dp4a(r.v[l].x(), q8[l + 0], sumi);
            sumi = ggml_cuda_dp4a(r.v[l].y(), q8[l + 4], sumi);
        }
        const float d = r.dw * bq8_1->ds[0];
        return d * sumi;
    }
};
template<> inline constexpr bool kSplit<23> = true;
template<> struct Split<23> {   // IQ4_XS
    struct W { sycl::int2 v[4]; int ls; float dw; };
    static W load(const void* __restrict__ vbq, int kbx, int iqs) {
        const block_iq4_xs* bq4 = (const block_iq4_xs*) vbq + kbx;
        W r;
#pragma unroll
        for (int j = 0; j < 4; ++j) r.v[j] = iq4nl_pair(get_int_b4(bq4->qs, iqs + j));   // SYCL port: = get_int_from_table_16(q)
        r.ls = ((bq4->scales_l[iqs / 8] >> (iqs & 0x04)) & 0x0F) | (((bq4->scales_h >> (iqs / 2)) & 0x03) << 4);
        r.dw = sycl::vec<sycl::half, 1>(bq4->d)
                   .convert<float, sycl::rounding_mode::automatic>()[0];
        return r;
    }
    static float apply(const W& r, const block_q8_1* __restrict__ bq8_1, int iqs) {
        int sumi = 0;
#pragma unroll
        for (int j = 0; j < 4; ++j) {
            const int u0 = get_int_b4(bq8_1[iqs / 4].qs, j + 0);
            const int u1 = get_int_b4(bq8_1[iqs / 4].qs, j + 4);
            sumi = ggml_cuda_dp4a(r.v[j].x(), u0, sumi);
            sumi = ggml_cuda_dp4a(r.v[j].y(), u1, sumi);
        }
        sumi *= r.ls - 32;
        const float d = r.dw * bq8_1[iqs / 4].ds[0];
        return d * sumi;
    }
};
template<> inline constexpr bool kSplit<42> = true;
template<> struct Split<42> {   // Q2_0
    struct W { int qx[4], qy[4]; float d2; };
    static W load(const void* __restrict__ vbq, int kbx, int iqs) {
        const block_q2_0* bq2_0 = (const block_q2_0*) vbq + kbx;
        W r;
        r.d2 = bq2_0->d;
        const int16_t* qs = (const int16_t*) bq2_0->qs + iqs * 4;
#pragma unroll
        for (int j = 0; j < 4; ++j) {
            const int q = qs[j];
            const int qe =
                dpct::byte_level_permute(0x020100FF, 0x020100FF, q >> 0);
            const int qo =
                dpct::byte_level_permute(0x020100FF, 0x020100FF, q >> 2);
            r.qx[j] = dpct::byte_level_permute(qe, qo, 0x5140);
            r.qy[j] = dpct::byte_level_permute(qe, qo, 0x7362);
        }
        return r;
    }
    static float apply(const W& r, const block_q8_1* __restrict__ bq8_1, int iqs) {
        const block_q8_1* bq8_1_chunk = bq8_1 + iqs;
        int sumi = 0;
#pragma unroll
        for (int j = 0; j < 4; ++j) {
            const int u = get_int_b4(bq8_1_chunk->qs, j * 2 + 0);
            const int v = get_int_b4(bq8_1_chunk->qs, j * 2 + 1);
            sumi = ggml_cuda_dp4a(u, r.qx[j], sumi);
            sumi = ggml_cuda_dp4a(v, r.qy[j], sumi);
        }
        const float d8 = bq8_1_chunk->ds[0];
        return r.d2 * d8 * sumi;
    }
};

// One row against the n <= NC activations x + off[0..n) (n >= 1, warp-uniform; offsets in q8_1 blocks, 32-bit to
// spare registers), the whole warp.  Per activation this is row_dot: the same calls k, lane-strided the same way,
// summed in the same order, then the same warp_sum.  Only the weight side moves out of the per-activation loop.
template <int TY, int NC>
__dpct_inline__ void row_dot_multi(const uint8_t *row, const block_q8_1 *x,
                                   const int (&off)[NC], int n, int nb,
                                   int lane, float (&s)[NC]) {
    using F = Fmt<TY>;
    using S = Split<TY>;
#pragma unroll
    for (int c = 0; c < NC; ++c) s[c] = 0.0f;
    for (int k = lane; k < nb * F::ipb; k += 32) {
        const int kbx = k / F::ipb, iqs = F::step * (k % F::ipb);
        /*
        DPCT1084: The function call "Split::load" has multiple migration
        results in different template instantiations that could not be unified.
        You may need to adjust the code.
        */
        const typename S::W w = S::load(row, kbx, iqs);
#pragma unroll
        for (int c = 0; c < NC; ++c)
            if (c < n) s[c] += S::apply(w, x + off[c] + kbx * (F::qk / 32), iqs);
    }
#pragma unroll
    for (int c = 0; c < NC; ++c)
        if (c < n) s[c] = warp_sum(s[c]);
}

// mmvq_kernel with the columns taken NC at a time.  After warp_sum every lane holds the same sum, so lane c stores
// column c.
template <int TY, int NC>
__dpct_inline__ void
mmvq_multi_kernel(const uint8_t *__restrict__ w, size_t row_bytes,
                  const block_q8_1 *__restrict__ x, float *__restrict__ y,
                  int n_in, int n_out, int ncols) {
    auto item_ct1 = sycl::ext::oneapi::this_work_item::get_nd_item<3>();
    const int row = item_ct1.get_group(2) * 4 + item_ct1.get_local_id(1);
    if (row >= n_out) return;
    const int lane = item_ct1.get_local_id(2);
    const int nb = n_in / Fmt<TY>::qk, xb = n_in / 32;
    const uint8_t* wr = w + (size_t) row * row_bytes;
    for (int c0 = 0; c0 < ncols; c0 += NC) {
        const int n = sycl::min(NC, ncols - c0);
        int off[NC];
#pragma unroll
        for (int c = 0; c < NC; ++c) off[c] = (c0 + sycl::min(c, n - 1)) * xb;
        float s[NC];
        row_dot_multi<TY, NC>(wr, x, off, n, nb, lane, s);
#pragma unroll
        for (int c = 0; c < NC; ++c)
            if (c < n && lane == c) y[(size_t) (c0 + c) * n_out + row] = s[c];
    }
}

// ---------------------------------------------------------------- grouped native experts

// ---------------------------------------------------------------- SYCL port: multi-entry dots
//
// A window routes the same expert for several tokens (up to 6). The single-entry dot above recomputes the
// dequantised weights - grid lookups, sign unpacking, the byte-wise compare/xor/subtract emulations - once per
// token; only the dp4a against the activation and the scale differ. Multi<TY> splits each format into `prep`
// (weights, once) and `acts` + `finish` (per token). NW packed int32s of weights per call.
struct MultiW { int w[8]; int a = 0, b = 0; float d = 0.f; };

template <int TY> struct Multi { static constexpr bool has = false; static constexpr int NW = 0; };

template <> struct Multi<18> {   // iq3_xxs
    static constexpr bool has = true; static constexpr int NW = 8;
    __dpct_inline__ static void prep(const void* vbq, int kbx, int iqs, MultiW& m, const void* grid = nullptr) {
        const block_iq3_xxs* bq3 = (const block_iq3_xxs*) vbq + kbx;
#if STRATA_IQ4NL_FAST
        const sycl::int2 q3_packed = load8_a2(bq3->qs + 4 * iqs);
        const uint32_t aux32 = (uint32_t) load4_a2(bq3->qs + 4 * (QK_K / 16 + iqs / 2));
#else
        const sycl::int2 q3_packed = sycl::int2(get_int_b2(bq3->qs, iqs), get_int_b2(bq3->qs, iqs + 1));
        const uint32_t aux32 = get_int_b2(bq3->qs, QK_K / 16 + iqs / 2);
#endif
        const uint8_t* q3 = (const uint8_t*) &q3_packed;
        const uint32_t* G = (const uint32_t*) grid;
#pragma unroll
        for (int l0 = 0; l0 < 8; l0 += 2) {
            const sycl::int2 grid_pos = G ? sycl::int2(G[q3[l0 + 0]], G[q3[l0 + 1]])
                                          : sycl::int2(iq3xxs_grid[q3[l0 + 0]], iq3xxs_grid[q3[l0 + 1]]);
            const uint32_t signs = unpack_ksigns(aux32 >> (7 * l0 / 2));
            const int signs0 = swar_ne4(signs & 0x08040201);
            const int signs1 = swar_ne4(signs & 0x80402010);
            m.w[l0 + 0] = swar_sub4(grid_pos.x() ^ signs0, signs0);
            m.w[l0 + 1] = swar_sub4(grid_pos.y() ^ signs1, signs1);
        }
        m.a = (int) (aux32 >> 28);
        m.d = sycl::vec<sycl::half, 1>(bq3->d).convert<float, sycl::rounding_mode::automatic>()[0];
    }
    __dpct_inline__ static void acts(const block_q8_1* y, int iqs, int* u, float& ds) {
#pragma unroll
        for (int j = 0; j < 8; ++j) u[j] = get_int_b4(y[iqs / 2].qs, j);
        ds = y[iqs / 2].ds[0];
    }
    __dpct_inline__ static float finish(int s0, int s1, const MultiW& m, float ds) {
        const int sumi = s0 + s1;
        return m.d * ds * (float) ((m.a * sumi + sumi / 2) / 2);
    }
};

template <> struct Multi<22> {   // iq2_s
    static constexpr bool has = true; static constexpr int NW = 8;
    __dpct_inline__ static void prep(const void* vbq, int kbx, int iqs, MultiW& m, const void* grid = nullptr) {
        const block_iq2_s* bq2 = (const block_iq2_s*) vbq + kbx;
#if STRATA_IQ4NL_FAST
        const int qs_packed = load4_a2(bq2->qs + 4 * (iqs / 2));
        const int signs_packed_32 = load4_a2(bq2->qs + 4 * (QK_K / 32 + iqs / 2));
#else
        const int qs_packed = get_int_b2(bq2->qs, iqs / 2);
        const int signs_packed_32 = get_int_b2(bq2->qs, QK_K / 32 + iqs / 2);
#endif
        const uint8_t* qs = (const uint8_t*) &qs_packed;
        const int qh = bq2->qh[iqs / 2];
        const uint8_t* sp = (const uint8_t*) &signs_packed_32;
        const uint64_t* G = (const uint64_t*) grid;
#pragma unroll
        for (int l0 = 0; l0 < 8; l0 += 2) {
            const int gi = qs[l0 / 2] | ((qh << (8 - l0)) & 0x300);
            const int* grid_pos = G ? (const int*) (G + gi) : (const int*) (iq2s_grid + gi);
            const int signs0 = swar_ne4(((sp[l0 / 2] & 0x03) << 7) | ((sp[l0 / 2] & 0x0C) << 21));
            const int signs1 = swar_ne4(((sp[l0 / 2] & 0x30) << 3) | ((sp[l0 / 2] & 0xC0) << 17));
            m.w[l0 + 0] = swar_sub4(grid_pos[0] ^ signs0, signs0);
            m.w[l0 + 1] = swar_sub4(grid_pos[1] ^ signs1, signs1);
        }
        m.a = bq2->scales[iqs / 2] & 0x0F;
        m.b = bq2->scales[iqs / 2] >> 4;
        m.d = sycl::vec<sycl::half, 1>(bq2->d).convert<float, sycl::rounding_mode::automatic>()[0];
    }
    __dpct_inline__ static void acts(const block_q8_1* y, int iqs, int* u, float& ds) {
#pragma unroll
        for (int j = 0; j < 8; ++j) u[j] = get_int_b4(y[iqs / 2].qs, j);
        ds = y[iqs / 2].ds[0];
    }
    __dpct_inline__ static float finish(int s0, int s1, const MultiW& m, float ds) {
        return m.d * ds * (float) ((s0 * m.a + s1 * m.b + (s0 + s1) / 2) / 4);
    }
};

template <> struct Multi<21> {   // iq3_s
    static constexpr bool has = true; static constexpr int NW = 8;
    __dpct_inline__ static void prep(const void* vbq, int kbx, int iqs, MultiW& m, const void* grid = nullptr) {
        const block_iq3_s* bq3 = (const block_iq3_s*) vbq + kbx;
#if STRATA_IQ4NL_FAST
        const sycl::int2 qs_packed = load8_a2(bq3->qs + 4 * iqs);
        const int signs_packed_32 = load4_a2(bq3->signs + 4 * (iqs / 2));
#else
        const sycl::int2 qs_packed = sycl::int2(get_int_b2(bq3->qs, iqs + 0), get_int_b2(bq3->qs, iqs + 1));
        const int signs_packed_32 = get_int_b2(bq3->signs, iqs / 2);
#endif
        const uint8_t* qs = (const uint8_t*) &qs_packed;
        const int qh = bq3->qh[iqs / 2];
        const uint8_t* sp = (const uint8_t*) &signs_packed_32;
        const uint32_t* G = (const uint32_t*) grid;
#pragma unroll
        for (int l0 = 0; l0 < 8; l0 += 2) {
            const int i0 = qs[l0 + 0] | ((qh << (8 - l0)) & 0x100), i1 = qs[l0 + 1] | ((qh << (7 - l0)) & 0x100);
            const sycl::int2 grid_pos = G ? sycl::int2(G[i0], G[i1]) : sycl::int2(iq3s_grid[i0], iq3s_grid[i1]);
            const int signs0 = swar_ne4(((sp[l0 / 2] & 0x03) << 7) | ((sp[l0 / 2] & 0x0C) << 21));
            const int signs1 = swar_ne4(((sp[l0 / 2] & 0x30) << 3) | ((sp[l0 / 2] & 0xC0) << 17));
            m.w[l0 + 0] = swar_sub4(grid_pos.x() ^ signs0, signs0);
            m.w[l0 + 1] = swar_sub4(grid_pos.y() ^ signs1, signs1);
        }
        m.a = 1 + 2 * ((bq3->scales[iqs / 4] >> ((iqs << 1) & 0x04)) & 0x0F);
        m.d = sycl::vec<sycl::half, 1>(bq3->d).convert<float, sycl::rounding_mode::automatic>()[0];
    }
    __dpct_inline__ static void acts(const block_q8_1* y, int iqs, int* u, float& ds) {
#pragma unroll
        for (int j = 0; j < 8; ++j) u[j] = get_int_b4(y[iqs / 2].qs, j);
        ds = y[iqs / 2].ds[0];
    }
    __dpct_inline__ static float finish(int s0, int s1, const MultiW& m, float ds) {
        return m.d * ds * (float) ((s0 + s1) * m.a);
    }
};

template <> struct Multi<20> {   // iq4_nl (down)
    static constexpr bool has = true; static constexpr int NW = 4;
    __dpct_inline__ static void prep(const void* vbq, int kbx, int iqs, MultiW& m, const void* = nullptr) {
        const block_iq4_nl* bq4 = (const block_iq4_nl*) vbq + kbx;
#if STRATA_IQ4NL_FAST
        const sycl::int2 qq = load8_a2(bq4->qs + 4 * iqs);
#endif
#pragma unroll
        for (int l = 0; l < 2; ++l) {
#if STRATA_IQ4NL_FAST
            const sycl::int2 v = iq4nl_pair(l ? qq.y() : qq.x());
#else
            const int aux_q4 = get_int_b2(bq4->qs, iqs + l);
            const sycl::int2 v = get_int_from_table_16(aux_q4, kvalues_iq4nl);
#endif
            m.w[2 * l + 0] = v.x();
            m.w[2 * l + 1] = v.y();
        }
        m.d = sycl::vec<sycl::half, 1>(bq4->d).convert<float, sycl::rounding_mode::automatic>()[0];
    }
    __dpct_inline__ static void acts(const block_q8_1* y, int iqs, int* u, float& ds) {
        const int* q8 = (const int*) y->qs + iqs;
#pragma unroll
        for (int l = 0; l < 2; ++l) { u[2 * l + 0] = q8[l + 0]; u[2 * l + 1] = q8[l + 4]; }
        ds = y->ds[0];
    }
    __dpct_inline__ static float finish(int s0, int s1, const MultiW& m, float ds) { return m.d * ds * (float) (s0 + s1); }
};

template <> struct Multi<42> {   // q2_0 (down)
    static constexpr bool has = true; static constexpr int NW = 8;
    __dpct_inline__ static void prep(const void* vbq, int kbx, int iqs, MultiW& m, const void* = nullptr) {
        const block_q2_0* bq2_0 = (const block_q2_0*) vbq + kbx;
        const int16_t* qs = (const int16_t*) bq2_0->qs + iqs * 4;
#pragma unroll
        for (int j = 0; j < 4; ++j) {
            const int q = qs[j];
            const int qe = dpct::byte_level_permute(0x020100FF, 0x020100FF, q >> 0);
            const int qo = dpct::byte_level_permute(0x020100FF, 0x020100FF, q >> 2);
            m.w[2 * j + 0] = dpct::byte_level_permute(qe, qo, 0x5140);
            m.w[2 * j + 1] = dpct::byte_level_permute(qe, qo, 0x7362);
        }
        m.d = bq2_0->d;
    }
    __dpct_inline__ static void acts(const block_q8_1* y, int iqs, int* u, float& ds) {
        const block_q8_1* chunk = y + iqs;
#pragma unroll
        for (int j = 0; j < 4; ++j) { u[2 * j + 0] = get_int_b4(chunk->qs, j * 2 + 0); u[2 * j + 1] = get_int_b4(chunk->qs, j * 2 + 1); }
        ds = chunk->ds[0];
    }
    __dpct_inline__ static float finish(int s0, int s1, const MultiW& m, float ds) { return m.d * ds * (float) (s0 + s1); }
};

// One row against E activations at once, LANES lanes per row.
template <int TY, int LANES, int E>
__dpct_inline__ void row_dot_multi(const uint8_t* row, const block_q8_1* const* xs, int nb, int sub, float* out,
                                   const void* grid = nullptr) {
    using F = Fmt<TY>;
    using M = Multi<TY>;
    float s[E];
#pragma unroll
    for (int e = 0; e < E; ++e) s[e] = 0.f;
    for (int k = sub; k < nb * F::ipb; k += LANES) {
        const int kbx = k / F::ipb, iqs = F::step * (k % F::ipb);
        MultiW m;
        M::prep(row, kbx, iqs, m, grid);
#pragma unroll
        for (int e = 0; e < E; ++e) {
            int u[8];
            float ds;
            M::acts(xs[e] + kbx * (F::qk / 32), iqs, u, ds);
            int s0 = 0, s1 = 0;
#pragma unroll
            for (int j = 0; j < M::NW / 2; ++j) s0 = ggml_cuda_dp4a(m.w[j], u[j], s0);
#pragma unroll
            for (int j = M::NW / 2; j < M::NW; ++j) s1 = ggml_cuda_dp4a(m.w[j], u[j], s1);
            s[e] += M::finish(s0, s1, m, ds);
        }
    }
#pragma unroll
    for (int e = 0; e < E; ++e) out[e] = lanes_sum<LANES>(s[e]);
}

// The entries e0..e1 of one row: chunks of 4 through the multi-entry dot, the rest one at a time.
template <int TY, int LANES>
__dpct_inline__ void row_entries(const uint8_t* wr, const block_q8_1* x, int x_stride, const int32_t* ent_idx,
                                 int e0, int e1, int nb, int sub, float* dst, size_t dst_stride,
                                 const void* grid = nullptr) {
    int e = e0;
    if constexpr (Multi<TY>::has) {
        for (; e + 4 <= e1; e += 4) {
            const block_q8_1* xs[4] = {x + (size_t) ent_idx[e] * x_stride, x + (size_t) ent_idx[e + 1] * x_stride,
                                       x + (size_t) ent_idx[e + 2] * x_stride, x + (size_t) ent_idx[e + 3] * x_stride};
            float o[4];
            row_dot_multi<TY, LANES, 4>(wr, xs, nb, sub, o, grid);
            if (sub == 0)
#pragma unroll
                for (int i = 0; i < 4; ++i) dst[(size_t) (e + i) * dst_stride] = o[i];
        }
        if (e + 2 <= e1) {
            const block_q8_1* xs[2] = {x + (size_t) ent_idx[e] * x_stride, x + (size_t) ent_idx[e + 1] * x_stride};
            float o[2];
            row_dot_multi<TY, LANES, 2>(wr, xs, nb, sub, o, grid);
            if (sub == 0) { dst[(size_t) e * dst_stride] = o[0]; dst[(size_t) (e + 1) * dst_stride] = o[1]; }
            e += 2;
        }
    }
    for (; e < e1; ++e) {
        const float v = row_dot_lanes<TY, LANES>(wr, x + (size_t) ent_idx[e] * x_stride, nb, sub);
        if (sub == 0) dst[(size_t) e * dst_stride] = v;
    }
}

constexpr int GU_ROWS = 8;     // rows per block (one warp each)

template <int TG, int LN = kExpertLanes>
__dpct_inline__ void native_gu_kernel(
    const unsigned long long *__restrict__ grp_ptr,
    const int32_t *__restrict__ grp_start, const int32_t *__restrict__ n_groups,
    const int32_t *__restrict__ ent_tok, const block_q8_1 *__restrict__ xq,
    NativeExpertLayout L, float *__restrict__ gate, float *__restrict__ up) {
    auto item_ct1 = sycl::ext::oneapi::this_work_item::get_nd_item<3>();
    // #363: group g is computed by block row g % gridDim.y (a launch sized for fewer rows than possible groups strides);
    // a block row with no group returns before the barrier below (uniform over the work-group)
    const int ng = *n_groups;
    if ((int) item_ct1.get_group(1) >= ng) return;
    const void* grid = nullptr;
#if STRATA_GRID_SLM
    if constexpr (TG == 18 || TG == 21 || TG == 22) {
        using GT = std::conditional_t<TG == 22, uint64_t, uint32_t>;
        constexpr int GN = TG == 22 ? 1024 : TG == 21 ? 512 : 256;
        auto& sg = *sycl::ext::oneapi::group_local_memory_for_overwrite<GT[GN]>(item_ct1.get_group());
        for (int i = (int) item_ct1.get_local_id(2); i < GN; i += (int) item_ct1.get_local_range(2)) {
            if constexpr (TG == 22) sg[i] = iq2s_grid[i];
            else if constexpr (TG == 21) sg[i] = iq3s_grid[i];
            else sg[i] = iq3xxs_grid[i];
        }
        item_ct1.barrier(sycl::access::fence_space::local_space);
        grid = &sg[0];
    }
#endif
    const int rib = item_ct1.get_local_id(2) / LN,
              sub = item_ct1.get_local_id(2) % LN;
    const int row = item_ct1.get_group(2) * (256 / LN) + rib; // 0 .. 2*n_ff
    if (row >= 2 * L.n_ff) return;
    const bool is_up = row >= L.n_ff;
    const int r = is_up ? row - (int) L.n_ff : row;
    const size_t off = (is_up ? L.up_off : 0) + (size_t) r * L.gu_row;
    const int nb = (int) (L.n_embd / Fmt<TG>::qk), xb = (int) (L.n_embd / 32);
    for (int g = (int) item_ct1.get_group(1); g < ng; g += (int) item_ct1.get_group_range(1)) {
        const uint8_t* wr = (const uint8_t*) grp_ptr[g] + off;
        const int e0 = grp_start[g], e1 = grp_start[g + 1];
        row_entries<TG, LN>(wr, xq, xb, ent_tok, e0, e1, nb, sub, (is_up ? up : gate) + r, (size_t) L.n_ff, grid);
    }
}

// native_gu_kernel with the group's entries taken GRP_NC at a time, each weight part decoded once per pass.
// A group has at most one entry per token of the window (kVerifyMaxT = 8; setup writes --spec 4, and a split window
// has halves of <= 4), so 4 takes such windows in one pass; 8 would take ~64-80 registers against ~48 (ptxas -v).
constexpr int GRP_NC = 4;

template <int TG>
__dpct_inline__ void native_gu_multi_kernel(
    const unsigned long long *__restrict__ grp_ptr,
    const int32_t *__restrict__ grp_start, const int32_t *__restrict__ n_groups,
    const int32_t *__restrict__ ent_tok, const block_q8_1 *__restrict__ xq,
    NativeExpertLayout L, float *__restrict__ gate, float *__restrict__ up) {
    auto item_ct1 = sycl::ext::oneapi::this_work_item::get_nd_item<3>();
    const int warp = item_ct1.get_local_id(2) >> 5,
              lane = item_ct1.get_local_id(2) & 31;
    const int row = item_ct1.get_group(2) * GU_ROWS + warp; // 0 .. 2*n_ff
    if (row >= 2 * L.n_ff) return;
    const bool is_up = row >= L.n_ff;
    const int r = is_up ? row - (int) L.n_ff : row;
    const size_t w_off = (is_up ? L.up_off : 0) + (size_t) r * L.gu_row;
    const int nb = (int) (L.n_embd / Fmt<TG>::qk), xb = (int) (L.n_embd / 32);
    float* dst = is_up ? up : gate;
    const int ng = *n_groups;
    for (int g = item_ct1.get_group(1); g < ng;
         g +=
         item_ct1.get_group_range(1)) { // the group stride, as native_gu_kernel
        const uint8_t* wr = (const uint8_t*) grp_ptr[g] + w_off;
        const int e0 = grp_start[g], e1 = grp_start[g + 1];
        for (int e = e0; e < e1; e += GRP_NC) {
            const int n = sycl::min(GRP_NC, e1 - e);
            int off[GRP_NC];
#pragma unroll
            for (int c = 0; c < GRP_NC; ++c)
                off[c] = ent_tok[e + sycl::min(c, n - 1)] * xb;
            float s[GRP_NC];
            row_dot_multi<TG, GRP_NC>(wr, xq, off, n, nb, lane, s);
#pragma unroll
            for (int c = 0; c < GRP_NC; ++c)
                if (c < n && lane == c) dst[(size_t) (e + c) * L.n_ff + r] = s[c];
        }
    }
}

__dpct_inline__ void swiglu_entries_kernel(const float *__restrict__ gate,
                                           const float *__restrict__ up,
                                           float *__restrict__ h, long long n) {
    auto item_ct1 = sycl::ext::oneapi::this_work_item::get_nd_item<3>();
    const long long i =
        (long long)item_ct1.get_group(2) * item_ct1.get_local_range(2) +
        item_ct1.get_local_id(2);
    if (i >= n) return;
    const float g = gate[i];
    h[i] = (g / (1.0f + sycl::native::exp(-g))) * up[i];
}

template <int TD, int LN = kExpertLanes>
__dpct_inline__ void native_down_kernel(
    const unsigned long long *__restrict__ grp_ptr,
    const int32_t *__restrict__ grp_start, const int32_t *__restrict__ n_groups,
    const int32_t *__restrict__ ent_dst, const block_q8_1 *__restrict__ hq,
    NativeExpertLayout L, float *__restrict__ out) {
    auto item_ct1 = sycl::ext::oneapi::this_work_item::get_nd_item<3>();
    const int ng = *n_groups;   // #363: the group stride, as native_gu_kernel
    const int rib = item_ct1.get_local_id(2) / LN,
              sub = item_ct1.get_local_id(2) % LN;
    const int r = item_ct1.get_group(2) * (256 / LN) + rib;
    if (r >= L.n_embd) return;
    const size_t off = L.down_off + (size_t) r * L.d_row;
    const int nb = (int) (L.n_ff / Fmt<TD>::qk), hb = (int) (L.n_ff / 32);
    // down: entry e's activation is row e of hq (identity index), its output row is ent_dst[e]
    for (int g = (int) item_ct1.get_group(1); g < ng; g += (int) item_ct1.get_group_range(1)) {
        const uint8_t* wr = (const uint8_t*) grp_ptr[g] + off;
        const int e0 = grp_start[g], e1 = grp_start[g + 1];
        int e = e0;
        if constexpr (Multi<TD>::has) {
            for (; e + 4 <= e1; e += 4) {
                const block_q8_1* xs[4] = {hq + (size_t) e * hb, hq + (size_t) (e + 1) * hb, hq + (size_t) (e + 2) * hb, hq + (size_t) (e + 3) * hb};
                float o[4];
                row_dot_multi<TD, LN, 4>(wr, xs, nb, sub, o);
                if (sub == 0)
#pragma unroll
                    for (int i = 0; i < 4; ++i) out[(size_t) ent_dst[e + i] * L.n_embd + r] = o[i];
            }
            if (e + 2 <= e1) {
                const block_q8_1* xs[2] = {hq + (size_t) e * hb, hq + (size_t) (e + 1) * hb};
                float o[2];
                row_dot_multi<TD, LN, 2>(wr, xs, nb, sub, o);
                if (sub == 0) { out[(size_t) ent_dst[e] * L.n_embd + r] = o[0]; out[(size_t) ent_dst[e + 1] * L.n_embd + r] = o[1]; }
                e += 2;
            }
        }
        for (; e < e1; ++e) {
            const float v = row_dot_lanes<TD, LN>(wr, hq + (size_t) e * hb, nb, sub);
            if (sub == 0) out[(size_t) ent_dst[e] * L.n_embd + r] = v;
        }
    }
}

// native_down_kernel with the entries taken GRP_NC at a time
template <int TD>
__dpct_inline__ void native_down_multi_kernel(
    const unsigned long long *__restrict__ grp_ptr,
    const int32_t *__restrict__ grp_start, const int32_t *__restrict__ n_groups,
    const int32_t *__restrict__ ent_dst, const block_q8_1 *__restrict__ hq,
    NativeExpertLayout L, float *__restrict__ out) {
    auto item_ct1 = sycl::ext::oneapi::this_work_item::get_nd_item<3>();
    const int warp = item_ct1.get_local_id(2) >> 5,
              lane = item_ct1.get_local_id(2) & 31;
    const int r = item_ct1.get_group(2) * 8 + warp;
    if (r >= L.n_embd) return;
    const size_t w_off = L.down_off + (size_t) r * L.d_row;
    const int nb = (int) (L.n_ff / Fmt<TD>::qk), hb = (int) (L.n_ff / 32);
    const int ng = *n_groups;
    for (int g = item_ct1.get_group(1); g < ng;
         g +=
         item_ct1.get_group_range(1)) { // the group stride, as native_gu_kernel
        const uint8_t* wr = (const uint8_t*) grp_ptr[g] + w_off;
        const int e0 = grp_start[g], e1 = grp_start[g + 1];
        for (int e = e0; e < e1; e += GRP_NC) {
            const int n = sycl::min(GRP_NC, e1 - e);
            int off[GRP_NC];
#pragma unroll
            for (int c = 0; c < GRP_NC; ++c)
                off[c] = (e + sycl::min(c, n - 1)) * hb;
            float s[GRP_NC];
            row_dot_multi<TD, GRP_NC>(wr, hq, off, n, nb, lane, s);
#pragma unroll
            for (int c = 0; c < GRP_NC; ++c)
                if (c < n && lane == c) out[(size_t) ent_dst[e + c] * L.n_embd + r] = s[c];
        }
    }
}

// ---------------------------------------------------------------- q8_1 (quantize.cu)
// value i of a q8_1 row set (the warp holds block i / 32, lane = i % 32)
__dpct_inline__ void q8_1_store(const float xi, block_q8_1 *__restrict__ y,
                                const long long i) {
    float amax = sycl::fabs(xi), sum = xi;
#pragma unroll
    for (int o = 16; o > 0; o >>= 1) {
        /*
        DPCT1108: '__shfl_xor_sync' was migrated with the experimental
        feature masked sub_group function which may not be supported by all
        compilers or runtimes. You may need to adjust the code.
        */
        amax = sycl::fmax(
            amax,
            dpct::experimental::permute_sub_group_by_xor(
                0xffffffffu, sycl::ext::oneapi::this_work_item::get_sub_group(),
                amax, o));
        /*
        DPCT1108: '__shfl_xor_sync' was migrated with the experimental
        feature masked sub_group function which may not be supported by all
        compilers or runtimes. You may need to adjust the code.
        */
        sum += dpct::experimental::permute_sub_group_by_xor(
            0xffffffffu, sycl::ext::oneapi::this_work_item::get_sub_group(),
            sum, o);
    }
    const float d = q8_1_finite(amax / 127.0f);   // #606: q8_1_finite.hpp - the same bits for every finite block
    const int8_t q = q8_1_quant(xi, d, amax);
    const long long ib = i / 32, iqs = i % 32;
    y[ib].qs[iqs] = q;
    if (iqs == 0) y[ib].ds = q8_1_ds(d, sum);
}

__dpct_inline__ void quantize_q8_1_kernel(const float *__restrict__ x,
                                          block_q8_1 *__restrict__ y,
                                          long long n) {
    auto item_ct1 = sycl::ext::oneapi::this_work_item::get_nd_item<3>();
    const long long i =
        (long long)item_ct1.get_group(2) * item_ct1.get_local_range(2) +
        item_ct1.get_local_id(2);
    if (i >= n) return;
    q8_1_store(x[i], y, i);
}

// swiglu_entries_kernel and quantize_q8_1_kernel in one pass, over the call's own entries only:
// [grp_start[0], grp_start[*n_groups]) (a call's entries are contiguous; the verify window's PCIe call starts after
// its VRAM call) instead of all cap_entries twice - nothing reads the others: the down kernel reads its groups'.
// n_ff is a multiple of 32, so a warp is one q8_1 block and the bounds are warp-uniform.  The values are the two
// kernels': the SwiGLU product is rounded on its own - it must not contract into the first add of q8_1_store's block
// sum, as it could not when it went through memory (CUDA: __fmul_rn; HIP's __fmul_rn is a plain product, so there
// the product is written here under contract(off)) - then q8_1_store unchanged: the same blocks, bit for bit.
__dpct_inline__ void swiglu_q8_1_entries_kernel(
    const float *__restrict__ gate, const float *__restrict__ up,
    const int32_t *__restrict__ grp_start, const int32_t *__restrict__ n_groups,
    int n_ff, block_q8_1 *__restrict__ hq) {
    // the product rounded on its own (no contraction into q8_1_store's block sum), as through memory in the v1 kernels
    // - HIP's form; icpx is clang too (SYCL port)
#pragma clang fp contract(off)
    auto item_ct1 = sycl::ext::oneapi::this_work_item::get_nd_item<3>();
    const long long lo = (long long)grp_start[0] * n_ff,
                    hi = (long long)grp_start[*n_groups] * n_ff;
    for (long long i =
             lo +
             (long long)item_ct1.get_group(2) * item_ct1.get_local_range(2) +
             item_ct1.get_local_id(2);
         i < hi; i += (long long)item_ct1.get_group_range(2) *
                      item_ct1.get_local_range(2)) {
        const float g = gate[i];
#if defined(__HIPCC__)
        const float h = (g / (1.0f + __expf(-g))) * up[i];
#else
        /*
        DPCT1013: The rounding mode could not be specified and the
        generated code may have different accuracy than the original code.
        Verify the correctness. SYCL math built-in function rounding mode is
        aligned with OpenCL C 1.2 standard.
        */
        const float h = (g / (1.0f + sycl::native::exp(-g))) * up[i];
#endif
        q8_1_store(h, hq, i);
    }
}

// ---------------------------------------------------------------- dequant (dequantize.cuh)
template <typename dst_t> __dpct_inline__ dst_t cvt(float v);
template <> __dpct_inline__ float cvt<float>(float v) { return v; }
template <>
__dpct_inline__ sycl::half cvt<sycl::half>(float v) {
    return sycl::vec<float, 1>(v)
        .convert<sycl::half, sycl::rounding_mode::automatic>()[0];
}

// SYCL port: the dequant functions below wrote one 2-byte value at a time (dq_* = llama.cpp's per-thread loops); on Xe
// the FP16 expert dequant ran at ~80 GB/s of writes, 40% of the prompt path. Each thread's run of values goes out as one
// vector store instead (the offsets are multiples of the run length, so the stores are aligned).
template <typename dst_t, int N>
__dpct_inline__ void store_run(dst_t* y, const float* v) {
    if constexpr (std::is_same_v<dst_t, sycl::half> && (N == 4 || N == 8)) {
        sycl::vec<sycl::half, N> h;
#pragma unroll
        for (int j = 0; j < N; ++j) h[j] = sycl::half(v[j]);
        *reinterpret_cast<sycl::vec<sycl::half, N>*>(y) = h;
    } else {
#pragma unroll
        for (int j = 0; j < N; ++j) y[j] = cvt<dst_t>(v[j]);
    }
}
template <typename dst_t>
inline void dq_iq2_xxs(const void *vx, int64_t ibs, dst_t *yy, int tid) {
    const block_iq2_xxs* x = (const block_iq2_xxs*) vx;
    const int64_t il = tid / 8, ib = tid % 8;
    dst_t* y = yy + 32 * ib + 8 * il;
    const uint16_t* q2 = x[ibs].qs + 4 * ib;
    const uint8_t* aux8 = (const uint8_t*) q2;
    const uint8_t* grid = (const uint8_t*) (iq2xxs_grid + aux8[il]);
    const uint32_t aux32 = q2[2] | (q2[3] << 16);
    const float d = (float) x[ibs].d * (0.5f + (aux32 >> 28)) * 0.25f;
    const uint8_t signs = ksigns_iq2xs[(aux32 >> 7 * il) & 127];
    float v[8];
#pragma unroll
    for (int j = 0; j < 8; ++j) v[j] = d * grid[j] * (signs & kmask_iq2xs[j] ? -1.f : 1.f);
    store_run<dst_t, 8>(y, v);
}
template <typename dst_t>
inline void dq_iq2_xs(const void *vx, int64_t ibs, dst_t *yy, int tid) {
    const block_iq2_xs* x = (const block_iq2_xs*) vx;
    const int64_t il = tid / 8, ib = tid % 8;
    dst_t* y = yy + 32 * ib + 8 * il;
    const uint16_t* q2 = x[ibs].qs + 4 * ib;
    const uint8_t* grid = (const uint8_t*) (iq2xs_grid + (q2[il] & 511));
    const float d = (float) x[ibs].d * (0.5f + ((x[ibs].scales[ib] >> 4 * (il / 2)) & 0xf)) * 0.25f;
    const uint8_t signs = ksigns_iq2xs[q2[il] >> 9];
    float v[8];
#pragma unroll
    for (int j = 0; j < 8; ++j) v[j] = d * grid[j] * (signs & kmask_iq2xs[j] ? -1.f : 1.f);
    store_run<dst_t, 8>(y, v);
}
template <typename dst_t>
inline void dq_iq2_s(const void *vx, int64_t ibs, dst_t *yy, int tid) {
    const block_iq2_s* x = (const block_iq2_s*) vx;
    const int64_t il = tid / 8, ib = tid % 8;
    dst_t* y = yy + 32 * ib + 8 * il;
    const uint8_t* grid = (const uint8_t*) (iq2s_grid + (x[ibs].qs[4 * ib + il] | ((x[ibs].qh[ib] << (8 - 2 * il)) & 0x300)));
    const float d = (float) x[ibs].d * (0.5f + ((x[ibs].scales[ib] >> 4 * (il / 2)) & 0xf)) * 0.25f;
    const uint8_t signs = x[ibs].qs[QK_K / 8 + 4 * ib + il];
    float v[8];
#pragma unroll
    for (int j = 0; j < 8; ++j) v[j] = d * grid[j] * (signs & kmask_iq2xs[j] ? -1.f : 1.f);
    store_run<dst_t, 8>(y, v);
}
template <typename dst_t>
inline void dq_iq3_xxs(const void *vx, int64_t ibs, dst_t *yy, int tid) {
    const block_iq3_xxs* x = (const block_iq3_xxs*) vx;
    const int64_t il = tid / 8, ib = tid % 8;
    dst_t* y = yy + 32 * ib + 8 * il;
    const uint8_t* q3 = x[ibs].qs + 8 * ib;
    const uint16_t* gas = (const uint16_t*) (x[ibs].qs + QK_K / 4) + 2 * ib;
    const uint8_t* grid1 = (const uint8_t*) (iq3xxs_grid + q3[2 * il + 0]);
    const uint8_t* grid2 = (const uint8_t*) (iq3xxs_grid + q3[2 * il + 1]);
    const uint32_t aux32 = gas[0] | (gas[1] << 16);
    const float d = (float) x[ibs].d * (0.5f + (aux32 >> 28)) * 0.5f;
    const uint8_t signs = ksigns_iq2xs[(aux32 >> 7 * il) & 127];
    float v[8];
#pragma unroll
    for (int j = 0; j < 4; ++j) {
        v[j + 0] = d * grid1[j] * (signs & kmask_iq2xs[j + 0] ? -1.f : 1.f);
        v[j + 4] = d * grid2[j] * (signs & kmask_iq2xs[j + 4] ? -1.f : 1.f);
    }
    store_run<dst_t, 8>(y, v);
}
template <typename dst_t>
inline void dq_iq3_s(const void *vx, int64_t ibs, dst_t *yy, int tid) {
    const block_iq3_s* x = (const block_iq3_s*) vx;
    const int64_t il = tid / 8, ib = tid % 8;
    dst_t* y = yy + 32 * ib + 8 * il;
    const uint8_t* qs = x[ibs].qs + 8 * ib;
    const uint8_t* grid1 = (const uint8_t*) (iq3s_grid + (qs[2 * il + 0] | ((x[ibs].qh[ib] << (8 - 2 * il)) & 256)));
    const uint8_t* grid2 = (const uint8_t*) (iq3s_grid + (qs[2 * il + 1] | ((x[ibs].qh[ib] << (7 - 2 * il)) & 256)));
    const float d = (float) x[ibs].d * (1 + 2 * ((x[ibs].scales[ib / 2] >> 4 * (ib % 2)) & 0xf));
    const uint8_t signs = x[ibs].signs[4 * ib + il];
    float v[8];
#pragma unroll
    for (int j = 0; j < 4; ++j) {
        v[j + 0] = d * grid1[j] * (signs & kmask_iq2xs[j + 0] ? -1.f : 1.f);
        v[j + 4] = d * grid2[j] * (signs & kmask_iq2xs[j + 4] ? -1.f : 1.f);
    }
    store_run<dst_t, 8>(y, v);
}
template <typename dst_t>
inline void dq_iq1_m(const void *vx, int64_t ibs, dst_t *yy, int tid) {
    const block_iq1_m* x = (const block_iq1_m*) vx;
    const int64_t il = tid / 8, ib = tid % 8;
    dst_t* y = yy + 32 * ib + 8 * il;
    const uint16_t* sc = (const uint16_t*) x[ibs].scales;
    iq1m_scale_t scale;
    scale.u16 = (sc[0] >> 12) | ((sc[1] >> 8) & 0x00f0) | ((sc[2] >> 4) & 0x0f00) | (sc[3] & 0xf000);
    const int64_t ib16 = 2 * ib + il / 2;
    const float d = (float) scale.f16 * (2 * ((sc[ib16 / 4] >> 3 * (ib16 % 4)) & 0x7) + 1);
    const float delta = x[ibs].qh[2 * ib + il / 2] & (0x08 << 4 * (il % 2)) ? -1 - IQ1M_DELTA : -1 + IQ1M_DELTA;
    uint32_t grid32[2];
    const int8_t* q = (const int8_t*) grid32;
    grid32[0] = iq1s_grid_gpu[x[ibs].qs[4 * ib + il] | (((x[ibs].qh[2 * ib + il / 2] >> 4 * (il % 2)) & 7) << 8)];
    grid32[1] = (grid32[0] >> 4) & 0x0f0f0f0f;
    grid32[0] &= 0x0f0f0f0f;
#pragma unroll
    for (int j = 0; j < 8; ++j) y[j] = cvt<dst_t>(d * (q[j] + delta));
}
template <typename dst_t>
inline void dq_iq4_nl(const void *vx, int64_t ibs, dst_t *yy, int tid) {
    const block_iq4_nl* x = (const block_iq4_nl*) vx + ibs * (QK_K / QK4_NL);
    // SYCL port: thread (ib, il) writes the block's values [8 il, 8 il + 8) as one run: il < 2 the low nibbles of
    // qs[8 il ..], il >= 2 the high nibbles of qs[8 (il - 2) ..] (the same values as llama.cpp's 4 + 4 split)
    const int64_t il = tid / 8, ib = tid % 8;
    dst_t* y = yy + 32 * ib + 8 * il;
    const uint8_t* q4 = x[ib].qs + 8 * (il & 1);
    const float d = (float) x[ib].d;
    float v[8];
    if (il < 2) {
#pragma unroll
        for (int j = 0; j < 8; ++j) v[j] = d * kvalues_iq4nl[q4[j] & 0xf];
    } else {
#pragma unroll
        for (int j = 0; j < 8; ++j) v[j] = d * kvalues_iq4nl[q4[j] >> 4];
    }
    store_run<dst_t, 8>(y, v);
}
// Q3_K (the Q2_0 file's token_embd): llama.cpp's dequantize_block_q3_K, its 64 threads folded onto 32
template <typename dst_t>
inline void dq_q3_k(const void *vx, int64_t ibs, dst_t *yy, int tid) {
    const block_q3_K* x = (const block_q3_K*) vx + ibs;
    for (int tt = tid; tt < 64; tt += 32) {
        const int r = tt / 4, t2 = r / 2, is0 = r % 2;
        const int l0 = 16 * is0 + 4 * (tt % 4);
        const int n = t2 / 4, j = t2 - 4 * n;
        const uint8_t m = (uint8_t) (1 << (4 * n + j));
        const int is = 8 * n + 2 * j + is0;
        const int shift = 2 * j;
        const int8_t us = is < 4  ? (int8_t) ((x->scales[is - 0] & 0xF) | (((x->scales[is + 8] >> 0) & 3) << 4)) :
                          is < 8  ? (int8_t) ((x->scales[is - 0] & 0xF) | (((x->scales[is + 4] >> 2) & 3) << 4)) :
                          is < 12 ? (int8_t) ((x->scales[is - 8] >> 4) | (((x->scales[is + 0] >> 4) & 3) << 4)) :
                                    (int8_t) ((x->scales[is - 8] >> 4) | (((x->scales[is - 4] >> 6) & 3) << 4));
        const float dl = (float) x->d * (us - 32);
        dst_t* y = yy + 128 * n + 32 * j;
        const uint8_t* q = x->qs + 32 * n;
        const uint8_t* hm = x->hmask;
#pragma unroll
        for (int l = l0; l < l0 + 4; ++l) y[l] = cvt<dst_t>(
            dl * ((int8_t)((q[l] >> shift) & 3) - ((hm[l] & m) ? 0 : 4)));
    }
}
template <typename dst_t>
inline void dq_iq4_xs(const void *vx, int64_t ibs, dst_t *yy, int tid) {
    const block_iq4_xs* x = (const block_iq4_xs*) vx + ibs;
    const int il = tid / 8, ib = tid % 8;
    dst_t* y = yy + 32 * ib + 4 * il;
    const uint8_t* q4 = x->qs + 16 * ib + 4 * il;
    const float d = (float) x->d * ((((x->scales_l[ib / 2] >> 4 * (ib % 2)) & 0xf) | (((x->scales_h >> 2 * ib) & 3) << 4)) - 32);
    float lo[4], hi[4];
#pragma unroll
    for (int j = 0; j < 4; ++j) {
        lo[j] = d * kvalues_iq4nl[q4[j] & 0xf];
        hi[j] = d * kvalues_iq4nl[q4[j] >> 4];
    }
    store_run<dst_t, 4>(y, lo);
    store_run<dst_t, 4>(y + 16, hi);
}
template <typename dst_t>
inline void dq_q2_0(const void *vx, int64_t ibs, dst_t *yy, int tid) {
    // one "superblock" = 256 values = 4 blocks of 64; thread tid writes 8 values
    const block_q2_0* x = (const block_q2_0*) vx + ibs * 4;
    const int b = tid / 8, part = tid % 8;          // block 0..3, 8 values each
    const float d = (float) x[b].d;
    float v[8];
#pragma unroll
    for (int j = 0; j < 8; ++j) {
        const int i = part * 8 + j;
        const int code = (x[b].qs[i / 4] >> ((i % 4) * 2)) & 3;
        v[j] = d * (float) (code - 1);
    }
    store_run<dst_t, 8>(yy + b * 64 + part * 8, v);
}

// llama.cpp's dequantize_q4_K / dequantize_q5_K (dequantize.cuh; q5_K's 64 threads folded onto 32) and the 32-value
// blocks of Q5_1 / Q8_0, 8 of them per 256-value "superblock" (thread tid writes 8 values of block tid % 8).
__dpct_inline__ void get_scale_min_k4(int j, const uint8_t *q, uint8_t &d,
                                      uint8_t &m) {
    if (j < 4) {
        d = q[j] & 63; m = q[j + 4] & 63;
    } else {
        d = (q[j + 4] & 0xF) | ((q[j - 4] >> 6) << 4);
        m = (q[j + 4] >> 4) | ((q[j - 0] >> 6) << 4);
    }
}
template <typename dst_t>
inline void dq_q4_k(const void *vx, int64_t ibs, dst_t *yy, int tid) {
    const block_q4_K* x = (const block_q4_K*) vx;
    const int64_t il = tid / 8, ir = tid % 8, is = 2 * il;
    const int n = 4;
    dst_t* y = yy + 64 * il + n * ir;
    const float dall = x[ibs].dm[0];
    const float dmin = x[ibs].dm[1];
    const uint8_t* q = x[ibs].qs + 32 * il + n * ir;
    uint8_t sc, m;
    get_scale_min_k4((int) is + 0, x[ibs].scales, sc, m);
    const float d1 = dall * sc, m1 = dmin * m;
    get_scale_min_k4((int) is + 1, x[ibs].scales, sc, m);
    const float d2 = dall * sc, m2 = dmin * m;
#pragma unroll
    for (int l = 0; l < n; ++l) {
        y[l + 0] = cvt<dst_t>(d1 * (q[l] & 0xF) - m1);
        y[l + 32] = cvt<dst_t>(d2 * (q[l] >> 4) - m2);
    }
}
template <typename dst_t>
inline void dq_q5_k(const void *vx, int64_t ibs, dst_t *yy, int tid) {
    const block_q5_K* x = (const block_q5_K*) vx;
    for (int tt = tid; tt < 64; tt += 32) {
        const int il = tt / 16, ir = tt % 16, is = 2 * il;
        dst_t* y = yy + 64 * il + 2 * ir;
        const float dall = x[ibs].dm[0];
        const float dmin = x[ibs].dm[1];
        const uint8_t* ql = x[ibs].qs + 32 * il + 2 * ir;
        const uint8_t* qh = x[ibs].qh + 2 * ir;
        uint8_t sc, m;
        get_scale_min_k4(is + 0, x[ibs].scales, sc, m);
        const float d1 = dall * sc, m1 = dmin * m;
        get_scale_min_k4(is + 1, x[ibs].scales, sc, m);
        const float d2 = dall * sc, m2 = dmin * m;
        uint8_t hm = (uint8_t) (1 << (2 * il));
        y[0] = cvt<dst_t>(d1 * ((ql[0] & 0xF) + (qh[0] & hm ? 16 : 0)) - m1);
        y[1] = cvt<dst_t>(d1 * ((ql[1] & 0xF) + (qh[1] & hm ? 16 : 0)) - m1);
        hm <<= 1;
        y[32] = cvt<dst_t>(d2 * ((ql[0] >> 4) + (qh[0] & hm ? 16 : 0)) - m2);
        y[33] = cvt<dst_t>(d2 * ((ql[1] >> 4) + (qh[1] & hm ? 16 : 0)) - m2);
    }
}
template <typename dst_t>
inline void dq_q5_0(const void *vx, int64_t ibs, dst_t *yy, int tid) {
    // symmetric: one delta (no min), the high bit folded in then centered by the format's fixed -16 offset
    const block_q5_0* x = (const block_q5_0*) vx + ibs * (QK_K / QK5_0);
    const int ib = tid % 8, il = tid / 8;
    const float d = sycl::vec<sycl::half, 1>(x[ib].d)
                        .convert<float, sycl::rounding_mode::automatic>()[0];
    uint32_t qh;
    memcpy(&qh, x[ib].qh, sizeof(qh));
    dst_t* y = yy + 32 * ib;
    for (int j = 0; j < 4; ++j) {
        const int iqs = 4 * il + j;
        const int xh_0 = ((qh >> (iqs + 0)) << 4) & 0x10;
        const int xh_1 = ((qh >> (iqs + 12))) & 0x10;
        y[iqs] = cvt<dst_t>(((float) ((x[ib].qs[iqs] & 0xf) | xh_0) - 16.0f) * d);
        y[iqs + 16] = cvt<dst_t>(((float) ((x[ib].qs[iqs] >> 4) | xh_1) - 16.0f) * d);
    }
}
template <typename dst_t>
inline void dq_q5_1(const void *vx, int64_t ibs, dst_t *yy, int tid) {
    const block_q5_1* x = (const block_q5_1*) vx + ibs * (QK_K / QK5_1);
    const int ib = tid % 8, il = tid / 8;
    const sycl::float2 dm =
        (x[ib].dm).template convert<float, sycl::rounding_mode::automatic>();
    uint32_t qh;
    memcpy(&qh, x[ib].qh, sizeof(qh));
    dst_t* y = yy + 32 * ib;
    for (int j = 0; j < 4; ++j) {
        const int iqs = 4 * il + j;                   // llama.cpp's dequantize_q5_1 for value pairs iqs, iqs + 16
        const int xh_0 = ((qh >> (iqs + 0)) << 4) & 0x10;
        const int xh_1 = ((qh >> (iqs + 12))) & 0x10;
        y[iqs] =
            cvt<dst_t>((float)((x[ib].qs[iqs] & 0xf) | xh_0) * dm.x() + dm.y());
        y[iqs + 16] =
            cvt<dst_t>((float)((x[ib].qs[iqs] >> 4) | xh_1) * dm.x() + dm.y());
    }
}
template <typename dst_t>
inline void dq_q8_0(const void *vx, int64_t ibs, dst_t *yy, int tid) {
    const block_q8_0* x = (const block_q8_0*) vx + ibs * (QK_K / QK8_0);
    const int ib = tid % 8, il = tid / 8;
    const float d = sycl::vec<sycl::half, 1>(x[ib].d)
                        .convert<float, sycl::rounding_mode::automatic>()[0];
    dst_t* y = yy + 32 * ib + 8 * il;
#pragma unroll
    for (int j = 0; j < 8; ++j)
        y[j] = cvt<dst_t>((float)x[ib].qs[8 * il + j] * d);
}

// BF16 (the token embedding as the checkpoint ships it, tools/embd_bf16_pack.py): 8 values per thread.
template <typename dst_t>
inline void dq_bf16(const void *vx, int64_t ibs, dst_t *yy, int tid) {
    const uint16_t* x = (const uint16_t*) vx + ibs * 256 + tid * 8;
#pragma unroll
    for (int j = 0; j < 8; ++j) yy[tid * 8 + j] =
        cvt<dst_t>(sycl::bit_cast<float>((uint32_t)x[j] << 16));
}

// Every type below must also be in is_iq() (BF16: embed_type_supported): the host entry points refuse the others,
// so the default is unreachable.
template <typename dst_t>
__dpct_inline__ void
dq_dispatch(int ty, const void *vx, int64_t ibs, dst_t *y, int tid) {
    switch (ty) {
        case 16: dq_iq2_xxs(vx, ibs, y, tid);
            break;
        case 17: dq_iq2_xs(vx, ibs, y, tid);
            break;
        case 18: dq_iq3_xxs(vx, ibs, y, tid);
            break;
        case 20: dq_iq4_nl(vx, ibs, y, tid); break;
        case 21: dq_iq3_s(vx, ibs, y, tid); break;
        case 22: dq_iq2_s(vx, ibs, y, tid); break;
        case 29: dq_iq1_m(vx, ibs, y, tid); break;
        case 23: dq_iq4_xs(vx, ibs, y, tid); break;
        case 11: dq_q3_k(vx, ibs, y, tid); break;
        case 42: dq_q2_0(vx, ibs, y, tid); break;
        case 12: dq_q4_k(vx, ibs, y, tid); break;
        case 13: dq_q5_k(vx, ibs, y, tid); break;
        case 7: dq_q5_1(vx, ibs, y, tid); break;
        case 6: dq_q5_0(vx, ibs, y, tid); break;
        case 8: dq_q8_0(vx, ibs, y, tid); break;
        case 30: dq_bf16(vx, ibs, y, tid); break;
        default: break;
    }
}

// flat: superblock i -> y + 256 i
template <typename dst_t>
__dpct_inline__ void dequant_flat_kernel(
    int ty, const void *__restrict__ vx, dst_t *__restrict__ y) {
    auto item_ct1 = sycl::ext::oneapi::this_work_item::get_nd_item<3>();
    const int64_t i = item_ct1.get_group(2);
    dq_dispatch<dst_t>(ty, vx, i, y + i * QK_K, item_ct1.get_local_id(2));
}
// gate/up: superblock i of a role matrix (n_embd/256 per row) -> interleaved row 2r + parity
__dpct_inline__ void dequant_gu_kernel(
    int ty, const void *__restrict__ gate, const void *__restrict__ up,
    int64_t per_row, sycl::half *__restrict__ y) {
    auto item_ct1 = sycl::ext::oneapi::this_work_item::get_nd_item<3>();
    const int64_t i = item_ct1.get_group(2);
    const int parity = item_ct1.get_group(1);
    const int64_t r = i / per_row, c = i % per_row;
    dq_dispatch<sycl::half>(ty, parity ? up : gate, i,
                            y + ((2 * r + parity) * per_row + c) * QK_K,
                            item_ct1.get_local_id(2));
}

// the types dq_dispatch dequantizes
bool is_iq(int t) {
    return t == 16 || t == 17 || t == 18 || t == 20 || t == 21 || t == 22 || t == 23 || t == 29 || t == 42 || t == 11 ||
           t == 12 || t == 13 || t == 7 || t == 6 || t == 8;
}
// values per block of the types the grouped expert kernels take (0 = none)
int gu_qk(int t) {
    switch (t) {
#define STRATA_QK(T) case T: return Fmt<T>::qk;
        STRATA_GU_FMTS(STRATA_QK)
#undef STRATA_QK
        default: return 0;
    }
}
int d_qk(int t) {
    switch (t) {
#define STRATA_QK(T) case T: return Fmt<T>::qk;
        STRATA_D_FMTS(STRATA_QK)
#undef STRATA_QK
        default: return 0;
    }
}

bool env_on(const char* name) {
    const char* v = std::getenv(name);
    return v != nullptr && v[0] != '\0' && v[0] != '0';
}
// STRATA_OLD_IQ_MMVQ=1 keeps the per-column kernels (bitwise equal to the new ones; kept for A/B timing)
bool g_old_kernels = env_on("STRATA_OLD_IQ_MMVQ");
// SYCL port (0.1.31 merge): upstream sends the split formats (the i-quants) to native_gu/down_multi_kernel, one warp
// per row, 8 rows a group. On the B70 the port's grouped kernels (8 lanes a row, 32 rows a group, the multi-entry
// dots with aligned loads and the IQ4 codebook in registers) are the measured ones, so they stay the default for
// every format; STRATA_EXPERT_SPLIT=1 takes upstream's multi kernels, launched with their own grid (GU_ROWS rows).
bool g_split_multi = env_on("STRATA_EXPERT_SPLIT");

template <int TY>
void launch_mmvq(const uint8_t *W, size_t rb, const block_q8_1 *X, float *y,
                 int n_in, int n_out, int ncols, dpct::queue_ptr s) {
    const dpct::dim3 grid((unsigned)((n_out + 3) / 4)), block(32, 4);
    /*
    DPCT1049: The work-group size passed to the SYCL kernel may exceed the
    limit. To get the device limit, query info::device::max_work_group_size.
    Adjust the work-group size if needed.
    */
    if constexpr (!kSplit<TY>) {

        auto exp_props = sycl::ext::oneapi::experimental::properties{
            sycl::ext::oneapi::experimental::use_root_sync};

        s->submit([&](sycl::handler &cgh) {

            cgh.parallel_for<dpct_kernel_name<class mmvq_kernel_87549a,
                                              dpct_kernel_scalar<TY>>>(
                sycl::nd_range<3>(grid * block, block), exp_props,
                [=](sycl::nd_item<3> item_ct1)
                    [[sycl::reqd_sub_group_size(32)]] {
                        mmvq_kernel<TY>(W, rb, X, y, n_in, n_out, ncols);
                    });
        });
    }
    /*
    DPCT1049: The work-group size passed to the SYCL kernel may exceed the
    limit. To get the device limit, query info::device::max_work_group_size.
    Adjust the work-group size if needed.
    */
    else if (g_old_kernels) {

        auto exp_props = sycl::ext::oneapi::experimental::properties{
            sycl::ext::oneapi::experimental::use_root_sync};

        s->submit([&](sycl::handler &cgh) {

            cgh.parallel_for<dpct_kernel_name<class mmvq_kernel_f617b1,
                                              dpct_kernel_scalar<TY>>>(
                sycl::nd_range<3>(grid * block, block), exp_props,
                [=](sycl::nd_item<3> item_ct1)
                    [[sycl::reqd_sub_group_size(32)]] {
                        mmvq_kernel<TY>(W, rb, X, y, n_in, n_out, ncols);
                    });
        });
    }
    /*
    DPCT1049: The work-group size passed to the SYCL kernel may exceed the
    limit. To get the device limit, query info::device::max_work_group_size.
    Adjust the work-group size if needed.
    */
    else if (ncols <= 1) {

        auto exp_props = sycl::ext::oneapi::experimental::properties{
            sycl::ext::oneapi::experimental::use_root_sync};

        s->submit([&](sycl::handler &cgh) {

            cgh.parallel_for<dpct_kernel_name<class mmvq_multi_kernel_77d6d0,
                                              dpct_kernel_scalar<TY>,
                                              dpct_kernel_scalar<1>>>(
                sycl::nd_range<3>(grid * block, block), exp_props,
                [=](sycl::nd_item<3> item_ct1)
                    [[sycl::reqd_sub_group_size(32)]] {
                        mmvq_multi_kernel<TY, 1>(W, rb, X, y, n_in, n_out,
                                                 ncols);
                    });
        });
    }
    /*
    DPCT1049: The work-group size passed to the SYCL kernel may exceed the
    limit. To get the device limit, query info::device::max_work_group_size.
    Adjust the work-group size if needed.
    */
    else if (ncols == 2) {

        auto exp_props = sycl::ext::oneapi::experimental::properties{
            sycl::ext::oneapi::experimental::use_root_sync};

        s->submit([&](sycl::handler &cgh) {

            cgh.parallel_for<dpct_kernel_name<class mmvq_multi_kernel_7ba01c,
                                              dpct_kernel_scalar<TY>,
                                              dpct_kernel_scalar<2>>>(
                sycl::nd_range<3>(grid * block, block), exp_props,
                [=](sycl::nd_item<3> item_ct1)
                    [[sycl::reqd_sub_group_size(32)]] {
                        mmvq_multi_kernel<TY, 2>(W, rb, X, y, n_in, n_out,
                                                 ncols);
                    });
        });
    }
    /*
    DPCT1049: The work-group size passed to the SYCL kernel may exceed the
    limit. To get the device limit, query info::device::max_work_group_size.
    Adjust the work-group size if needed.
    */
    else if (ncols <= 4) {

        auto exp_props = sycl::ext::oneapi::experimental::properties{
            sycl::ext::oneapi::experimental::use_root_sync};

        s->submit([&](sycl::handler &cgh) {

            cgh.parallel_for<dpct_kernel_name<class mmvq_multi_kernel_6fd6d9,
                                              dpct_kernel_scalar<TY>,
                                              dpct_kernel_scalar<4>>>(
                sycl::nd_range<3>(grid * block, block), exp_props,
                [=](sycl::nd_item<3> item_ct1)
                    [[sycl::reqd_sub_group_size(32)]] {
                        mmvq_multi_kernel<TY, 4>(W, rb, X, y, n_in, n_out,
                                                 ncols);
                    });
        });
    }
    /*
    DPCT1049: The work-group size passed to the SYCL kernel may exceed the
    limit. To get the device limit, query info::device::max_work_group_size.
    Adjust the work-group size if needed.
    */
    else {

        auto exp_props = sycl::ext::oneapi::experimental::properties{
            sycl::ext::oneapi::experimental::use_root_sync};

        s->submit([&](sycl::handler &cgh) {

            cgh.parallel_for<dpct_kernel_name<class mmvq_multi_kernel_6e0f6e,
                                              dpct_kernel_scalar<TY>,
                                              dpct_kernel_scalar<8>>>(
                sycl::nd_range<3>(grid * block, block), exp_props,
                [=](sycl::nd_item<3> item_ct1)
                    [[sycl::reqd_sub_group_size(32)]] {
                        mmvq_multi_kernel<TY, 8>(W, rb, X, y, n_in, n_out,
                                                 ncols);
                    });
        });
    } // 8 at a time past 8
}

// SYCL port: lanes per row of the port's grouped expert kernels at run time (STRATA_GU_LANES / STRATA_DOWN_LANES:
// 4, 8, 16 or 32; default STRATA_EXPERT_LANES), 256 / lanes rows per work-group. Gate/up rows are 80 dot calls
// (2560 wide), the IQ4_NL down rows 40 (640 wide), so the best split can differ by role.
inline int lanes_env(const char* name) {
    const char* v = std::getenv(name);
    const int n = v ? std::atoi(v) : kExpertLanes;
    return n == 4 || n == 8 || n == 16 || n == 32 ? n : kExpertLanes;
}
inline int gu_lanes() { static const int v = lanes_env("STRATA_GU_LANES"); return v; }
inline int down_lanes() { static const int v = lanes_env("STRATA_DOWN_LANES"); return v; }
template <int TG, int LN>
void launch_gu_port(unsigned groups, dpct::queue_ptr s, const unsigned long long *grp_ptr, const int32_t *grp_start,
                    const int32_t *n_groups, const int32_t *ent_tok, const block_q8_1 *X,
                    const NativeExpertLayout &L, float *gate, float *up) {
    constexpr int ROWS = 256 / LN;
    const unsigned gx = (unsigned) ((2 * L.n_ff + ROWS - 1) / ROWS);
    auto exp_props = sycl::ext::oneapi::experimental::properties{sycl::ext::oneapi::experimental::use_root_sync};
    s->parallel_for<dpct_kernel_name<class native_gu_port, dpct_kernel_scalar<TG>, dpct_kernel_scalar<LN>>>(
        sycl::nd_range<3>(sycl::range<3>(1, groups, (size_t) gx * 256), sycl::range<3>(1, 1, 256)), exp_props,
        [=](sycl::nd_item<3>) [[sycl::reqd_sub_group_size(32)]] {
            native_gu_kernel<TG, LN>(grp_ptr, grp_start, n_groups, ent_tok, X, L, gate, up);
        });
}
template <int TG>
void launch_gu_lanes(dpct::dim3 grid, dpct::queue_ptr s, const unsigned long long *grp_ptr, const int32_t *grp_start,
                     const int32_t *n_groups, const int32_t *ent_tok, const block_q8_1 *X,
                     const NativeExpertLayout &L, float *gate, float *up) {
    switch (gu_lanes()) {
        case 4: launch_gu_port<TG, 4>(grid.y, s, grp_ptr, grp_start, n_groups, ent_tok, X, L, gate, up); break;
        case 16: launch_gu_port<TG, 16>(grid.y, s, grp_ptr, grp_start, n_groups, ent_tok, X, L, gate, up); break;
        case 32: launch_gu_port<TG, 32>(grid.y, s, grp_ptr, grp_start, n_groups, ent_tok, X, L, gate, up); break;
        default: launch_gu_port<TG, 8>(grid.y, s, grp_ptr, grp_start, n_groups, ent_tok, X, L, gate, up); break;
    }
}
template <int TD, int LN>
void launch_down_port(unsigned groups, dpct::queue_ptr s, const unsigned long long *grp_ptr, const int32_t *grp_start,
                      const int32_t *n_groups, const int32_t *ent_dst, const block_q8_1 *hq,
                      const NativeExpertLayout &L, float *out) {
    constexpr int ROWS = 256 / LN;
    const unsigned gx = (unsigned) ((L.n_embd + ROWS - 1) / ROWS);
    auto exp_props = sycl::ext::oneapi::experimental::properties{sycl::ext::oneapi::experimental::use_root_sync};
    s->parallel_for<dpct_kernel_name<class native_down_port, dpct_kernel_scalar<TD>, dpct_kernel_scalar<LN>>>(
        sycl::nd_range<3>(sycl::range<3>(1, groups, (size_t) gx * 256), sycl::range<3>(1, 1, 256)), exp_props,
        [=](sycl::nd_item<3>) [[sycl::reqd_sub_group_size(32)]] {
            native_down_kernel<TD, LN>(grp_ptr, grp_start, n_groups, ent_dst, hq, L, out);
        });
}
template <int TD>
void launch_down_lanes(dpct::dim3 grid, dpct::queue_ptr s, const unsigned long long *grp_ptr, const int32_t *grp_start,
                       const int32_t *n_groups, const int32_t *ent_dst, const block_q8_1 *hq,
                       const NativeExpertLayout &L, float *out) {
    switch (down_lanes()) {
        case 4: launch_down_port<TD, 4>(grid.y, s, grp_ptr, grp_start, n_groups, ent_dst, hq, L, out); break;
        case 16: launch_down_port<TD, 16>(grid.y, s, grp_ptr, grp_start, n_groups, ent_dst, hq, L, out); break;
        case 32: launch_down_port<TD, 32>(grid.y, s, grp_ptr, grp_start, n_groups, ent_dst, hq, L, out); break;
        default: launch_down_port<TD, 8>(grid.y, s, grp_ptr, grp_start, n_groups, ent_dst, hq, L, out); break;
    }
}

template <int TG>
void launch_gu(dpct::dim3 grid, dpct::queue_ptr s,
               const unsigned long long *grp_ptr, const int32_t *grp_start,
               const int32_t *n_groups, const int32_t *ent_tok,
               const block_q8_1 *X, const NativeExpertLayout &L, float *gate,
               float *up) {
    if constexpr (!kSplit<TG>) {

        launch_gu_lanes<TG>(grid, s, grp_ptr, grp_start, n_groups, ent_tok, X, L, gate, up);
    } else if (g_old_kernels || !g_split_multi) {

        launch_gu_lanes<TG>(grid, s, grp_ptr, grp_start, n_groups, ent_tok, X, L, gate, up);
    } else {

        auto exp_props = sycl::ext::oneapi::experimental::properties{
            sycl::ext::oneapi::experimental::use_root_sync};

        s->submit([&](sycl::handler &cgh) {

            cgh.parallel_for<dpct_kernel_name<
                class native_gu_multi_kernel_37f479, dpct_kernel_scalar<TG>>>(
                sycl::nd_range<3>(dpct::dim3((unsigned) ((2 * L.n_ff + GU_ROWS - 1) / GU_ROWS), grid.y) * sycl::range(1, 1, 256),
                                  sycl::range(1, 1, 256)),
                exp_props,
                [=](sycl::nd_item<3> item_ct1)
                    [[sycl::reqd_sub_group_size(32)]] {
                        native_gu_multi_kernel<TG>(grp_ptr, grp_start, n_groups,
                                                   ent_tok, X, L, gate, up);
                    });
        });
    }
}

template <int TD>
void launch_down(dpct::dim3 grid, dpct::queue_ptr s,
                 const unsigned long long *grp_ptr, const int32_t *grp_start,
                 const int32_t *n_groups, const int32_t *ent_dst,
                 const block_q8_1 *hq, const NativeExpertLayout &L,
                 float *out) {
    if constexpr (!kSplit<TD>) {

        launch_down_lanes<TD>(grid, s, grp_ptr, grp_start, n_groups, ent_dst, hq, L, out);
    } else if (g_old_kernels || !g_split_multi) {

        launch_down_lanes<TD>(grid, s, grp_ptr, grp_start, n_groups, ent_dst, hq, L, out);
    } else {

        auto exp_props = sycl::ext::oneapi::experimental::properties{
            sycl::ext::oneapi::experimental::use_root_sync};

        s->submit([&](sycl::handler &cgh) {

            cgh.parallel_for<dpct_kernel_name<
                class native_down_multi_kernel_490cef, dpct_kernel_scalar<TD>>>(
                sycl::nd_range<3>(dpct::dim3((unsigned) ((L.n_embd + GU_ROWS - 1) / GU_ROWS), grid.y) * sycl::range(1, 1, 256),
                                  sycl::range(1, 1, 256)),
                exp_props,
                [=](sycl::nd_item<3> item_ct1)
                    [[sycl::reqd_sub_group_size(32)]] {
                        native_down_multi_kernel<TD>(grp_ptr, grp_start,
                                                     n_groups, ent_dst, hq, L,
                                                     out);
                    });
        });
    }
}

}  // namespace

void iq_set_old_kernels(bool old) { g_old_kernels = old; }
bool iq_old_kernels() { return g_old_kernels; }

bool iq_supported(int t) noexcept { return is_iq(t); }
bool embed_type_supported(int t) noexcept { return is_iq(t) || t == 30; }

size_t iq_row_bytes(int t, int64_t n) noexcept {
    switch (t) {
        case 16: return (size_t) (n / 256) * sizeof(block_iq2_xxs);
        case 17: return (size_t) (n / 256) * sizeof(block_iq2_xs);
        case 18: return (size_t) (n / 256) * sizeof(block_iq3_xxs);
        case 20: return (size_t) (n / 32) * sizeof(block_iq4_nl);
        case 21: return (size_t) (n / 256) * sizeof(block_iq3_s);
        case 22: return (size_t) (n / 256) * sizeof(block_iq2_s);
        case 29: return (size_t) (n / 256) * sizeof(block_iq1_m);
        case 23: return (size_t) (n / 256) * sizeof(block_iq4_xs);
        case 11: return (size_t) (n / 256) * sizeof(block_q3_K);
        case 42: return (size_t) (n / 64) * sizeof(block_q2_0);
        case 12: return (size_t) (n / 256) * sizeof(block_q4_K);
        case 13: return (size_t) (n / 256) * sizeof(block_q5_K);
        case 7: return (size_t) (n / 32) * sizeof(block_q5_1);
        case 6: return (size_t) (n / 32) * sizeof(block_q5_0);
        case 8: return (size_t) (n / 32) * sizeof(block_q8_0);
        case 30: return (size_t) n * 2;   // BF16: the token embedding only (iq_embed_rows, iq_dequant_f32)
        default: return 0;
    }
}

void quantize_q8_1_rows(const float* x, int64_t n_rows, int64_t n_cols, void* y, void* stream) {
    const long long n = (long long) n_rows * n_cols;
    if (n <= 0) return;
    {
        auto exp_props = sycl::ext::oneapi::experimental::properties{
            sycl::ext::oneapi::experimental::use_root_sync};

        strata::q_of(stream)
            ->parallel_for<dpct_kernel_name<class quantize_q8_1_kernel_9d62b4>>(
                sycl::nd_range<3>(
                    sycl::range(1, 1, (unsigned)((n + 255) / 256)) *
                        sycl::range(1, 1, 256),
                    sycl::range(1, 1, 256)),
                exp_props,
                [=](sycl::nd_item<3> item_ct1)
                    [[sycl::reqd_sub_group_size(32)]] {
                        quantize_q8_1_kernel(x, (block_q8_1 *)y, n);
                    });
    }
    check("quantize_q8_1_rows");
}

void iq_mmvq(int t, const void* w, const void* x_q8_1, float* y, int n_in, int n_out, int ncols, void* stream) {
    const size_t rb = iq_row_bytes(t, n_in);
    dpct::queue_ptr s = strata::q_of(stream);
    const auto* W = (const uint8_t*) w;
    const auto* X = (const block_q8_1*) x_q8_1;
    switch (t) {
#define STRATA_MMVQ(T) case T: launch_mmvq<T>(W, rb, X, y, n_in, n_out, ncols, s); break;
        STRATA_MMVQ_FMTS(STRATA_MMVQ)
#undef STRATA_MMVQ
        default: std::fprintf(stderr, "iq_mmvq: type %d is not supported\n", t); std::exit(1);
    }
    check("iq_mmvq");
}

void iq_dequant_f16(int t, const void* src, int64_t n, uint16_t* dst, void* stream) {
    if (n % 256 != 0 || !is_iq(t)) { std::fprintf(stderr, "iq_dequant_f16: bad arguments\n"); std::exit(1); }
    {

        auto exp_props = sycl::ext::oneapi::experimental::properties{
            sycl::ext::oneapi::experimental::use_root_sync};
        dpct::has_capability_or_fail(
            strata::q_of(stream)->get_device(),
            {sycl::aspect::fp16});

        strata::q_of(stream)
            ->submit([&](sycl::handler &cgh) {

                cgh.parallel_for<dpct_kernel_name<
                    class dequant_flat_kernel_d891c8, sycl::half>>(
                    sycl::nd_range<3>(sycl::range(1, 1, (unsigned)(n / 256)) *
                                          sycl::range(1, 1, 32),
                                      sycl::range(1, 1, 32)),
                    exp_props, [=](sycl::nd_item<3> item_ct1) {
                        dequant_flat_kernel<sycl::half>(
                            t, src, (sycl::half *)dst);
                    });
            });
    }
    check("iq_dequant_f16");
}

namespace {
__dpct_inline__ void embed_rows_kernel(
    int ty, const uint8_t *__restrict__ table, size_t row_bytes,
    const int32_t *__restrict__ tokens, int64_t n_embd, float *__restrict__ y) {
    auto item_ct1 = sycl::ext::oneapi::this_work_item::get_nd_item<3>();
    const int t = item_ct1.get_group(1);
    const int64_t b = item_ct1.get_group(2);
    const uint8_t* row = table + (size_t) tokens[t] * row_bytes;
    dq_dispatch<float>(ty, row, b, y + (size_t)t * n_embd + b * QK_K,
                       item_ct1.get_local_id(2));
}
}  // namespace

void iq_embed_rows(int t, const void* table, size_t row_bytes, const int32_t* tokens, int64_t n_tok, int64_t n_embd,
                   float* out, void* stream) {
    if (n_tok <= 0) return;
    if (n_embd % 256 != 0 || !embed_type_supported(t)) { std::fprintf(stderr, "iq_embed_rows: bad arguments\n"); std::exit(1); }
    {

        auto exp_props = sycl::ext::oneapi::experimental::properties{
            sycl::ext::oneapi::experimental::use_root_sync};
        dpct::has_capability_or_fail(
            strata::q_of(stream)->get_device(),
            {sycl::aspect::fp16});

        strata::q_of(stream)
            ->submit([&](sycl::handler &cgh) {

                cgh.parallel_for<
                    dpct_kernel_name<class embed_rows_kernel_284a43>>(
                    sycl::nd_range<3>(sycl::range(1, (unsigned)n_tok,
                                                  (unsigned)(n_embd / 256)) *
                                          sycl::range(1, 1, 32),
                                      sycl::range(1, 1, 32)),
                    exp_props, [=](sycl::nd_item<3> item_ct1) {
                        embed_rows_kernel(
                            t, (const uint8_t *)table, row_bytes, tokens,
                            n_embd, out);
                    });
            });
    }
    check("iq_embed_rows");
}

void iq_dequant_f32(int t, const void* src, int64_t n, float* dst, void* stream) {
    if (n % 256 != 0 || !embed_type_supported(t)) { std::fprintf(stderr, "iq_dequant_f32: bad arguments\n"); std::exit(1); }
    {

        auto exp_props = sycl::ext::oneapi::experimental::properties{
            sycl::ext::oneapi::experimental::use_root_sync};
        dpct::has_capability_or_fail(
            strata::q_of(stream)->get_device(),
            {sycl::aspect::fp16});

        strata::q_of(stream)
            ->submit([&](sycl::handler &cgh) {

                cgh.parallel_for<
                    dpct_kernel_name<class dequant_flat_kernel_d7b5ca, float>>(
                    sycl::nd_range<3>(sycl::range(1, 1, (unsigned)(n / 256)) *
                                          sycl::range(1, 1, 32),
                                      sycl::range(1, 1, 32)),
                    exp_props, [=](sycl::nd_item<3> item_ct1) {
                        dequant_flat_kernel<float>(
                            t, src, dst);
                    });
            });
    }
    check("iq_dequant_f32");
}

void iq_dequant_gu_f16(int t, const void* gate, const void* up, int64_t n_ff, int64_t n_embd, uint16_t* dst, void* stream) {
    // checked like the other entry points: an unknown type used to leave `dst` unwritten, a wrong prompt and no error
    if (n_embd % 256 != 0 || !is_iq(t)) { std::fprintf(stderr, "iq_dequant_gu_f16: type %d / %lld\n", t, (long long) n_embd); std::exit(1); }
    const int64_t per_row = n_embd / 256;
    {

        auto exp_props = sycl::ext::oneapi::experimental::properties{
            sycl::ext::oneapi::experimental::use_root_sync};
        dpct::has_capability_or_fail(
            strata::q_of(stream)->get_device(),
            {sycl::aspect::fp16});

        strata::q_of(stream)
            ->submit([&](sycl::handler &cgh) {

                cgh.parallel_for<
                    dpct_kernel_name<class dequant_gu_kernel_2397ca>>(
                    sycl::nd_range<3>(
                        sycl::range(1, 2, (unsigned)(n_ff * per_row)) *
                            sycl::range(1, 1, 32),
                        sycl::range(1, 1, 32)),
                    exp_props, [=](sycl::nd_item<3> item_ct1) {
                        dequant_gu_kernel(
                            t, gate, up, per_row, (sycl::half *)dst);
                    });
            });
    }
    check("iq_dequant_gu_f16");
}

bool native_expert_supported(int gu_type, int d_type, int64_t n_embd, int64_t n_ff) noexcept {
    const int qg = gu_qk(gu_type), qd = d_qk(d_type);
    return qg > 0 && qd > 0 && is_iq(gu_type) && is_iq(d_type) && n_embd % qg == 0 && n_ff % qd == 0 &&
           n_embd % 256 == 0 && (n_ff * n_embd) % 256 == 0;
}

NativeExpertLayout native_expert_layout(int gu_type, int d_type, int64_t n_embd, int64_t n_ff) {
    NativeExpertLayout L;
    L.gu_type = gu_type;
    L.d_type = d_type;
    L.n_embd = n_embd;
    L.n_ff = n_ff;
    L.gu_row = iq_row_bytes(gu_type, n_embd);
    L.d_row = iq_row_bytes(d_type, n_ff);
    L.up_off = (size_t) n_ff * L.gu_row;
    L.down_off = 2 * L.up_off;
    L.bytes = L.down_off + (size_t) n_embd * L.d_row;
    return L;
}

size_t native_expert_scratch_bytes(int64_t cap, int64_t n_ff) {
    const size_t f = (size_t) cap * (size_t) n_ff * sizeof(float);
    return 3 * ((f + 255) & ~(size_t) 255) + (((size_t) cap * (size_t) (n_ff / 32) * sizeof(block_q8_1) + 255) & ~(size_t) 255);
}

namespace {
bool g_grouped_v1 = env_on("STRATA_GROUPED_V1");
}  // namespace

void native_grouped_set_v1(bool v1) { g_grouped_v1 = v1; }

void native_expert_grouped(const NativeExpertLayout& L, const unsigned long long* grp_ptr, const int32_t* grp_start,
                           const int32_t* n_groups, const int32_t* ent_dst, const int32_t* ent_tok, int64_t cap_groups,
                           int64_t cap_entries, const void* x_q8_1, void* scratch, float* out, void* stream,
                           int64_t grid_groups) {
    if (cap_groups <= 0 || cap_entries <= 0) return;
    if (L.n_ff % 32 != 0) { std::fprintf(stderr, "native_expert_grouped: n_ff %lld\n", (long long) L.n_ff); std::exit(1); }
    dpct::queue_ptr s = strata::q_of(stream);
    const size_t f = (size_t) cap_entries * (size_t) L.n_ff * sizeof(float), fa = (f + 255) & ~(size_t) 255;
    float* gate = (float*) scratch;
    float* up = (float*) ((uint8_t*) scratch + fa);
    float* h = (float*) ((uint8_t*) scratch + 2 * fa);
    block_q8_1* hq = (block_q8_1*) ((uint8_t*) scratch + 3 * fa);
    const auto* X = (const block_q8_1*) x_q8_1;
    const bool v1 = g_grouped_v1;
    // #363: a verify window's PCIe call usually has no group: `grid_groups` block rows stride over the groups instead
    // of one row per possible group (the results do not depend on it)
    const int64_t gy = (v1 || grid_groups <= 0 || grid_groups > cap_groups) ? cap_groups : grid_groups;
    const dpct::dim3 ggu((unsigned)((2 * L.n_ff + kExpertRows - 1) / kExpertRows),
                         (unsigned)gy);
    switch (L.gu_type) {
#define STRATA_GU(T) case T: launch_gu<T>(ggu, s, grp_ptr, grp_start, n_groups, ent_tok, X, L, gate, up); break;
        STRATA_GU_FMTS(STRATA_GU)
#undef STRATA_GU
        default: std::fprintf(stderr, "native_expert_grouped: gate/up type %d\n", L.gu_type); std::exit(1);
    }
    check("native_expert_grouped/gu");
    const long long nh = (long long) cap_entries * L.n_ff;
    if (v1) {
        {
            auto exp_props = sycl::ext::oneapi::experimental::properties{
                sycl::ext::oneapi::experimental::use_root_sync};

            s->parallel_for<
                dpct_kernel_name<class swiglu_entries_kernel_a4959c>>(
                sycl::nd_range<3>(
                    sycl::range(1, 1, (unsigned)((nh + 255) / 256)) *
                        sycl::range(1, 1, 256),
                    sycl::range(1, 1, 256)),
                exp_props, [=](sycl::nd_item<3> item_ct1) {
                    swiglu_entries_kernel(gate, up, h, nh);
                });
        }
        {
            auto exp_props = sycl::ext::oneapi::experimental::properties{
                sycl::ext::oneapi::experimental::use_root_sync};

            s->parallel_for<
                dpct_kernel_name<class quantize_q8_1_kernel_cfb6c5>>(
                sycl::nd_range<3>(
                    sycl::range(1, 1, (unsigned)((nh + 255) / 256)) *
                        sycl::range(1, 1, 256),
                    sycl::range(1, 1, 256)),
                exp_props,
                [=](sycl::nd_item<3> item_ct1)
                    [[sycl::reqd_sub_group_size(32)]] {
                        quantize_q8_1_kernel(h, hq, nh);
                    });
        }
    } else {
        auto exp_props = sycl::ext::oneapi::experimental::properties{
            sycl::ext::oneapi::experimental::use_root_sync};

        s->parallel_for<
            dpct_kernel_name<class swiglu_q8_1_entries_kernel_115291>>(
            sycl::nd_range<3>(sycl::range(1, 1, (unsigned)((nh + 255) / 256)) *
                                  sycl::range(1, 1, 256),
                              sycl::range(1, 1, 256)),
            exp_props,
            [=](sycl::nd_item<3> item_ct1) [[sycl::reqd_sub_group_size(32)]] {
                swiglu_q8_1_entries_kernel(gate, up, grp_start, n_groups,
                                           (int)L.n_ff, hq);
            });
    }
    check("native_expert_grouped/swiglu");
    const dpct::dim3 gd((unsigned)((L.n_embd + kExpertRows - 1) / kExpertRows), (unsigned)gy);
    switch (L.d_type) {
#define STRATA_DOWN(T) case T: launch_down<T>(gd, s, grp_ptr, grp_start, n_groups, ent_dst, hq, L, out); break;
        STRATA_D_FMTS(STRATA_DOWN)
#undef STRATA_DOWN
        default: std::fprintf(stderr, "native_expert_grouped: down type %d\n", L.d_type); std::exit(1);
    }
    check("native_expert_grouped/down");
}


// ================================================================ SYCL port: XMX GEMM straight from the quantized rows
//
// The prompt path dequantizes every routed expert to FP16 (8 MB written, then read back by oneMKL) before its GEMMs:
// 39-41% of the prompt time on the B70 (2026-09-30), the GEMMs themselves another 16%. This kernel dequantizes a
// 16-feature strip of the weight matrix into local memory, packs it for the XMX B operand and multiplies it against
// every activation row with joint_matrix (16x16x16, FP16 in, FP32 out; sub-group 16 as XMX wants). The FP16 weights
// never touch global memory.
//
//   Y[M][n_out] = X[M][K] . W[n_out][K]^T      X FP16 row-major (lda K), Y FP32 (ldy), W quantized rows (a ggml IQ type)
//
// Gate/up: `up != nullptr`, virtual row 2r = gate row r, 2r+1 = up row r (the order iq_dequant_gu_f16 produces, which
// swiglu_interleaved expects). K must be a multiple of 256 (a QK_K block per slab) or the whole row must fit a slab
// (K = 640: 16 rows are 40 blocks, block-aligned since 16 * 640 / 256 = 40).
namespace xmx {
namespace jm = sycl::ext::oneapi::experimental::matrix;
constexpr int SG = 16;                 // XMX sub-group size
constexpr int NSG = 8;                 // sub-groups per work-group
constexpr int WG = SG * NSG;
constexpr int NT = 16;                 // features per work-group: one B tile
constexpr int MAXT = 4;                // C tiles per sub-group -> 512 rows per chunk
constexpr int APAD = 8;                // halves of padding on the staged A tile rows
struct Src { const uint8_t* gate; const uint8_t* up; int ty; int64_t per_row; };
struct Smem { sycl::half* stage; sycl::half* packed; sycl::half* atile; float* ctile; };

// the block index (x[ibs] in dq_dispatch) of virtual row v's k-range [k0, k0 + 256) - rows are per_row blocks long
__dpct_inline__ const void* block_of(const Src& s, int v, int cblk, int64_t& ibs) {
    if (s.up != nullptr) { ibs = (int64_t) (v >> 1) * s.per_row + cblk; return (v & 1) ? s.up : s.gate; }
    ibs = (int64_t) v * s.per_row + cblk;
    return s.gate;
}

// one work-group: features [f0, f0 + 16) of the weight matrix against all M rows; slab_k halves of K per pass
void gemm_kernel(Src src, const sycl::half* X, int K, int slab_k, float* Y, int64_t ldy, int M, int n_out, Smem sm) {
    auto it = sycl::ext::oneapi::this_work_item::get_nd_item<3>();
    auto sg = it.get_sub_group();
    const int sgid = (int) sg.get_group_id()[0], lane = (int) sg.get_local_id()[0], t = (int) it.get_local_id(2);
    const int f0 = (int) it.get_group(2) * NT;
    const int per_slab = slab_k / 256;                    // blocks of one row per slab (K % 256 == 0), or
    const int whole = (slab_k == K) ? 1 : 0;              // the whole 16-row strip as NT * K / 256 flat blocks
    const int nblk = whole ? NT * K / 256 : NT * per_slab;
    auto gptr = [](auto* p) { return sycl::address_space_cast<sycl::access::address_space::global_space, sycl::access::decorated::no>(p); };
    auto lptr = [](auto* p) { return sycl::address_space_cast<sycl::access::address_space::local_space, sycl::access::decorated::no>(p); };
    sycl::half* atile = sm.atile + (size_t) sgid * 16 * (16 + APAD);
    float* ctile = sm.ctile + (size_t) sgid * 256;
    for (int m_base = 0; m_base < M; m_base += MAXT * NSG * 16) {
        jm::joint_matrix<sycl::sub_group, float, jm::use::accumulator, 16, 16> C[MAXT];
#pragma unroll
        for (int k = 0; k < MAXT; ++k) jm::joint_matrix_fill(sg, C[k], 0.0f);
        const int tiles = std::min((M - m_base + 15) / 16, MAXT * NSG);
        for (int k0 = 0; k0 < K; k0 += slab_k) {
            it.barrier(sycl::access::fence_space::local_space);
            // dequantize the strip's blocks of this slab into stage[NT][slab_k] (flat, row-major): 32 "threads" per
            // block in dq_dispatch's terms, two calls per lane
            for (int b = sgid; b < nblk; b += NSG) {
                int v, kk;
                if (whole) { v = (b * 256) / K; kk = (b * 256) % K; }
                else { v = b / per_slab; kk = (b % per_slab) * 256; }
                sycl::half* dst = sm.stage + (size_t) v * slab_k + kk;
                const int row = f0 + v;
                if (row < n_out) {
                    int64_t ibs;
                    const void* blk = src.gate;
                    if (whole) ibs = ((int64_t) f0 * K) / 256 + b;   // flat blocks of the [n_out][K] matrix
                    else blk = block_of(src, row, k0 / 256 + b % per_slab, ibs);
                    dq_dispatch<sycl::half>(src.ty, blk, ibs, dst, lane);
                    dq_dispatch<sycl::half>(src.ty, blk, ibs, dst, lane + 16);
                } else {
                    for (int i = lane; i < 256; i += SG) dst[i] = sycl::half(0.0f);
                }
            }
            it.barrier(sycl::access::fence_space::local_space);
            // pack for the XMX B operand: element (k, n) at packed[(k / 2) * (NT * 2) + n * 2 + (k & 1)]
            for (int i = t; i < NT * slab_k; i += WG) {
                const int n = i / slab_k, k = i - n * slab_k;
                sm.packed[(k >> 1) * (NT * 2) + n * 2 + (k & 1)] = sm.stage[i];
            }
            it.barrier(sycl::access::fence_space::local_space);
            for (int ks = 0; ks < slab_k; ks += 16) {
                jm::joint_matrix<sycl::sub_group, sycl::half, jm::use::b, 16, 16, jm::layout::ext_intel_packed> B;
                jm::joint_matrix_load(sg, B, lptr(sm.packed + (ks >> 1) * (NT * 2)), NT * 2);
#pragma unroll
                for (int tt = 0; tt < MAXT; ++tt) {
                    const int tile = sgid + tt * NSG;
                    if (tile >= tiles) break;
                    const int m0 = m_base + tile * 16;
                    jm::joint_matrix<sycl::sub_group, sycl::half, jm::use::a, 16, 16, jm::layout::row_major> A;
                    if (m0 + 16 <= M) {
                        jm::joint_matrix_load(sg, A, gptr(X + (size_t) m0 * K + k0 + ks), (size_t) K);
                    } else {   // the tail tile: valid rows staged, the rest zero
                        for (int i = lane; i < 16 * 16; i += SG) {
                            const int r = i >> 4, c = i & 15;
                            atile[r * (16 + APAD) + c] = (m0 + r < M) ? X[(size_t) (m0 + r) * K + k0 + ks + c] : sycl::half(0.0f);
                        }
                        sycl::group_barrier(sg);
                        jm::joint_matrix_load(sg, A, lptr(atile), (size_t) (16 + APAD));
                        sycl::group_barrier(sg);
                    }
                    jm::joint_matrix_mad(sg, C[tt], A, B, C[tt]);
                }
            }
        }
        // the chunk's results
#pragma unroll
        for (int tt = 0; tt < MAXT; ++tt) {
            const int tile = sgid + tt * NSG;
            if (tile >= tiles) break;
            const int m0 = m_base + tile * 16;
            if (m0 + 16 <= M && f0 + NT <= n_out) {
                jm::joint_matrix_store(sg, C[tt], gptr(Y + (size_t) m0 * ldy + f0), (size_t) ldy, jm::layout::row_major);
            } else {
                jm::joint_matrix_store(sg, C[tt], lptr(ctile), (size_t) 16, jm::layout::row_major);
                sycl::group_barrier(sg);
                for (int i = lane; i < 256; i += SG) {
                    const int r = i >> 4, c = i & 15;
                    if (m0 + r < M && f0 + c < n_out) Y[(size_t) (m0 + r) * ldy + f0 + c] = ctile[i];
                }
                sycl::group_barrier(sg);
            }
        }
    }
}
}  // namespace xmx

bool xmx_gemm_iq(int ty, const void* gate, const void* up, int64_t K, int n_out, const uint16_t* X, int M, float* Y,
                 int64_t ldy, void* stream) {
    if (!is_iq(ty) || ty == 16 || ty == 17 || ty == 29 || ty == 23 || ty == 11) return false;   // dq_dispatch covers these; the rest untested here
    if (M <= 0 || n_out <= 0 || n_out % xmx::NT != 0 || K <= 0) return false;
    const bool by_block = (K % 256) == 0;
    const bool whole = !by_block && ((int64_t) xmx::NT * K) % 256 == 0 && K % 16 == 0;
    if (!by_block && !whole) return false;
    const int slab_k = by_block ? 256 : (int) K;
    if (up != nullptr && !by_block) return false;   // gate/up rows are per_row blocks each
    const size_t stage_halves = (size_t) xmx::NT * slab_k;
    const size_t bytes = stage_halves * 2 /*stage*/ + stage_halves * 2 /*packed*/ + (size_t) xmx::NSG * 16 * (16 + xmx::APAD) * 2 +
                         (size_t) xmx::NSG * 256 * 4;
    if (bytes > 96 * 1024) return false;
    dpct::queue_ptr q = strata::q_of(stream);
    xmx::Src src{(const uint8_t*) gate, (const uint8_t*) up, ty, by_block ? K / 256 : 0};
    const int groups = n_out / xmx::NT;
    q->submit([&](sycl::handler& cgh) {
        sycl::local_accessor<uint8_t, 1> slm(sycl::range<1>(bytes), cgh);
        cgh.parallel_for(
            sycl::nd_range<3>(sycl::range<3>(1, 1, (size_t) groups * xmx::WG), sycl::range<3>(1, 1, xmx::WG)),
            [=](sycl::nd_item<3>) [[sycl::reqd_sub_group_size(16)]] {
                uint8_t* base = slm.get_multi_ptr<sycl::access::decorated::no>().get();
                xmx::Smem sm;
                sm.stage = (sycl::half*) base;
                sm.packed = sm.stage + stage_halves;
                sm.atile = sm.packed + stage_halves;
                sm.ctile = (float*) (sm.atile + (size_t) xmx::NSG * 16 * (16 + xmx::APAD));
                xmx::gemm_kernel(src, (const sycl::half*) X, (int) K, slab_k, Y, ldy, M, n_out, sm);
            });
    });
    return true;
}
}  // namespace strata::kernels
