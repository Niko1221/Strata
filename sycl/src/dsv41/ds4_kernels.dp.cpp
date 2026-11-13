// Copyright (c) 2026 Peasant Smith and the Project Maya contributors (MIT; https://github.com/mw00/project-maya, v1.1.17).
// Ported into Strata (third_party/project-maya/NOTICE.md, LICENSE); changes are in the git history of this file.
// src/ds4/ds4_kernels.cu - DeepSeek-V4.1's device kernels (see ds4_kernels.cuh).  No fast math here: the rotary
// angles, the softmaxes and the roundings follow the reference's F32 arithmetic.  A window's token is blockIdx.y.
#define DPCT_PROFILING_ENABLED
#include <sycl/sycl.hpp>
#include <dpct/dpct.hpp>
#include "ds4_sycl_wait.hpp"
#include "ds4_launch.h" // strata_ds4_launch: <<<>>> on CUDA and HIP, a dispatch on Metal
#include "ds4_kernels.dp.hpp"
#if defined(STRATA_METAL)
#include <ds4_metal_host.h>
#endif
#include "ds4_numerics.dp.hpp"
#include "ds4_q6k.dp.hpp"
#include "ds4_attn_split.dp.hpp"
#include "ds4_wmma.dp.hpp"
#include "ds4_experts_rule.hpp"

#include <algorithm>
#include <atomic>
#include <cfloat>
#include <cstdlib>
#include <cstring>
#include <map>
#include <mutex>
#include <cstdio>
#include <time.h>

#include <cmath>

namespace strata::ds4 {
namespace {

constexpr int kThreads = 256;

__dpct_inline__ float bf16f(uint16_t b) {
    return sycl::bit_cast<float>((uint32_t)b << 16);
}

__dpct_inline__ uint16_t
f2bf16(float f) { // round to nearest even (the values here are exact anyway)
    uint32_t u = sycl::bit_cast<unsigned int>(f);
    if ((u & 0x7f800000u) == 0x7f800000u) return (uint16_t) (u >> 16);
    u += 0x7fffu + ((u >> 16) & 1u);
    return (uint16_t) (u >> 16);
}

__dpct_inline__ float warp_sum(float v) {
    /*
DPCT1108:125: '__shfl_xor_sync' was migrated with the experimental feature
masked sub_group function which may not be supported by all compilers or
runtimes. You may need to adjust the code.
*/
#pragma unroll
    for (int o = 16; o > 0; o >>= 1) v +=
        dpct::experimental::permute_sub_group_by_xor(
            0xffffffffu, sycl::ext::oneapi::this_work_item::get_sub_group(), v,
            o);
    return v;
}
__dpct_inline__ float warp_max(float v) {
    /*
DPCT1108:126: '__shfl_xor_sync' was migrated with the experimental feature
masked sub_group function which may not be supported by all compilers or
runtimes. You may need to adjust the code.
*/
#pragma unroll
    for (int o = 16; o > 0; o >>= 1) v = sycl::fmax(
        v, dpct::experimental::permute_sub_group_by_xor(
               0xffffffffu, sycl::ext::oneapi::this_work_item::get_sub_group(),
               v, o));
    return v;
}

// block-wide sum for blockDim.x <= 1024 (a multiple of 32); every thread gets the result
inline float block_sum(float v) {
    /*
    DPCT1115:414: The sycl::ext::oneapi::group_local_memory_for_overwrite is
    used to allocate group-local memory at the none kernel functor scope of a
    work-group data parallel kernel. You may need to adjust the code.
    */
    auto item_ct1 = sycl::ext::oneapi::this_work_item::get_nd_item<3>();
auto &red = *sycl::ext::oneapi::group_local_memory_for_overwrite<float[32]>(
    sycl::ext::oneapi::this_work_item::get_work_group<3>());
    /*
    DPCT1115:415: The sycl::ext::oneapi::group_local_memory_for_overwrite is
    used to allocate group-local memory at the none kernel functor scope of a
    work-group data parallel kernel. You may need to adjust the code.
    */
    auto &total = *sycl::ext::oneapi::group_local_memory_for_overwrite<float>(
        sycl::ext::oneapi::this_work_item::get_work_group<3>());
    const int lane = item_ct1.get_local_id(2) & 31,
              w = item_ct1.get_local_id(2) >> 5;
    v = warp_sum(v);
    if (lane == 0) red[w] = v;
    /*
    DPCT1065:411: Consider replacing sycl::nd_item::barrier() with
    sycl::nd_item::barrier(sycl::access::fence_space::local_space) for better
    performance if there is no access to global memory.
    */
    item_ct1.barrier();
    if (w == 0) {
        float t =
            lane < (int)(item_ct1.get_local_range(2) >> 5) ? red[lane] : 0.0f;
        t = warp_sum(t);
        if (lane == 0) total = t;
    }
    /*
    DPCT1065:412: Consider replacing sycl::nd_item::barrier() with
    sycl::nd_item::barrier(sycl::access::fence_space::local_space) for better
    performance if there is no access to global memory.
    */
    item_ct1.barrier();
    const float r = total;
    /*
    DPCT1065:413: Consider replacing sycl::nd_item::barrier() with
    sycl::nd_item::barrier(sycl::access::fence_space::local_space) for better
    performance if there is no access to global memory.
    */
    item_ct1.barrier();
    return r;
}

// 2^ceil(log2(x)) for a positive normal x (the reference's fast_round_scale)
__dpct_inline__ float pow2_ceil(float x) {
    const uint32_t b = sycl::bit_cast<unsigned int>(x);
    const int e = (int) ((b >> 23) & 0xffu) - 127 + ((b & 0x7fffffu) != 0u ? 1 : 0);
    return sycl::bit_cast<float>((uint32_t)(e + 127) << 23);
}

// a load of memory another device or the host writes, never from a stale cache line (CUDA's __ldcv; HIP: volatile
// 32-bit loads)
template <class T> __dpct_inline__ T ld_fresh(const T *p) {
#if defined(STRATA_USE_HIP)
    T v;
    const volatile uint32_t* s = (const volatile uint32_t*) p;
    uint32_t* d = (uint32_t*) &v;
#pragma unroll
    for (int i = 0; i < (int) (sizeof(T) / 4); ++i) d[i] = s[i];
    return v;
#else
    /*
    DPCT1098:416: The '*' expression is used instead of the __ldcv call. These
    two expressions do not provide the exact same functionality. Check the
    generated code for potential precision and/or performance issues.
    */
    return *p;
#endif
}

__dpct_inline__ float fp8_rt(float v) {
    return e4m3_round(v);
} // E4M3, nearest even (input clamped)

__dpct_inline__ float
e2m1_rt(float v) { // E2M1 magnitudes 0 .5 1 1.5 2 3 4 6, nearest, ties to even
    const float a = sycl::fabs(v);
    float r;
    if (a <= 0.25f) r = 0.0f;
    else if (a < 0.75f) r = 0.5f;
    else if (a <= 1.25f) r = 1.0f;
    else if (a < 1.75f) r = 1.5f;
    else if (a <= 2.5f) r = 2.0f;
    else if (a < 3.5f) r = 3.0f;
    else if (a <= 5.0f) r = 4.0f;
    else r = 6.0f;
    return sycl::copysign(r, v);
}

// ---------------------------------------------------------------- the caches' packed rows (kv_row_bytes)
// the E2M1 code of a value e2m1_rt returned (bit 3 the sign, -0 kept; magnitudes 0 .5 1 1.5 2 3 4 6 as 0 .. 7)
__dpct_inline__ uint32_t e2m1_code(float q) {
    const float a = sycl::fabs(q);
    const uint32_t m = a <= 1.5f ? (uint32_t) (a * 2.0f) : a <= 2.0f ? 4u : a <= 3.0f ? 5u : a <= 4.0f ? 6u : 7u;
    return m | ((sycl::bit_cast<unsigned int>(q) >> 31) << 3);
}
// the E4M3 code of v / 2^k for a v whose quotient is an E4M3 value (|v / 2^k| <= 448, at most 4 significant bits, a
// multiple of 2^-9; v 0 or a normal F32): integer arithmetic, exact, -0 kept
__dpct_inline__ uint32_t e4m3_code(float v, int k = 0) {
    const uint32_t b = sycl::bit_cast<unsigned int>(v), sg = (b >> 24) & 0x80u,
                   a = b & 0x7fffffffu;
    if (a == 0u) return sg;
    const int e = (int) (a >> 23) - 127 - k;   // the quotient's binade
    if (e >= -6) return sg | ((uint32_t) (e + 7) << 3) | ((a >> 20) & 7u);
    return sg | (((a & 0x7fffffu) | 0x800000u) >> (14 - e));   // a subnormal: m 2^-9
}
// an E2M1 code's value times 2^-14, by the conversion of an FP16 (the code's sign, exponent and mantissa bit are an
// FP16's at bits 15 and 11..9: exact, subnormals too) - the readers' decode: (value 2^-14) (scale 2^14) is value * scale
// exactly, both products exact
__dpct_inline__ float e2m1_val14(uint32_t c) {
    return sycl::vec<sycl::half, 1>(
               sycl::bit_cast<sycl::half, unsigned short>(
                   (unsigned short)(((c & 8u) << 12) | ((c & 7u) << 9))))
        .convert<float, sycl::rounding_mode::automatic>()[0];
}
// a UE8M0 scale byte's value (the byte the F32 exponent: 2^(b - 127); 0 -> 0, an unwritten row's)
__dpct_inline__ float ue8m0_val(uint32_t b) {
    return sycl::bit_cast<float>(b << 23);
}

// ---------------------------------------------------------------- embed
__dpct_inline__ void embed_kernel(const uint16_t *__restrict__ rows,
                                  float *__restrict__ h, int dim, int hc) {
    auto item_ct1 = sycl::ext::oneapi::this_work_item::get_nd_item<3>();
    const int e = item_ct1.get_group(2) * item_ct1.get_local_range(2) +
                  item_ct1.get_local_id(2),
              t = item_ct1.get_group(1);
    if (e >= dim) return;
    const float v = bf16f(rows[(size_t) t * dim + e]);
#pragma unroll
    for (int s = 0; s < hc; ++s) h[((size_t)t * hc + s) * dim + e] = v;
}

// ---------------------------------------------------------------- finish
constexpr int kFinMax = 32;   // values a thread holds: len <= 32 * 256

// kPack: the rounded rows packed to f.pdst (f.quant's cache format) instead of written to x or f.dst
template <bool kPack>
/*
DPCT1110:127: The total declared local variable size in device function
finish_kernel exceeds 128 bytes and may cause high register pressure. Consult
with your hardware vendor to find the total register size available and adjust
the code, or use smaller sub-group size to avoid high register pressure.
*/
__dpct_inline__ void finish_kernel(float *x, int len, long long stride,
                                   Finish f) {
    auto item_ct1 = sycl::ext::oneapi::this_work_item::get_nd_item<3>();
    float *v = x + (long long)item_ct1.get_group(2) * stride;
    float r[kFinMax];
    const int nj = (len + kThreads - 1) / kThreads;
    float ss = 0.0f;
#pragma unroll
    for (int j = 0; j < kFinMax; ++j) {
        if (j >= nj) break;
        const int e = item_ct1.get_local_id(2) + j * kThreads;
        r[j] = e < len ? v[e] : 0.0f;
        ss += r[j] * r[j];
    }
    if (f.w != nullptr) {
        const float tot = block_sum(ss);
        const float inv = sycl::rsqrt(tot / (float)len + f.eps);
#pragma unroll
        for (int j = 0; j < kFinMax; ++j) {
            if (j >= nj) break;
            const int e = item_ct1.get_local_id(2) + j * kThreads;
            if (e < len) r[j] = r[j] * inv * f.w[e];
        }
    }
    if (f.rd > 0) {
        const int r0 = len - f.rd;
        const int pos =
            f.pos + (int)(item_ct1.get_group(2) / f.vpp) * f.pos_step;
#pragma unroll
        for (int j = 0; j < kFinMax; ++j) {
            if (j >= nj) break;
            const int e = item_ct1.get_local_id(2) + j * kThreads;
            /*
            DPCT1108:128: '__shfl_xor_sync' was migrated with the experimental
            feature masked sub_group function which may not be supported by all
            compilers or runtimes. You may need to adjust the code.
            */
            const float other = dpct::experimental::permute_sub_group_by_xor(
                0xffffffffu, sycl::ext::oneapi::this_work_item::get_sub_group(),
                r[j], 1);
            if (e >= r0 && e < len) {
                const int i = (e - r0) >> 1;
                const float ang = (float) pos * f.freqs[i];
                const float c = sycl::cos((float)ang);
                const float sn =
                    f.inverse ? -sycl::sin((float)ang) : sycl::sin((float)ang);
                if (((e - r0) & 1) == 0) r[j] = r[j] * c - other * sn;   // re = a c - b s
                else r[j] = other * sn + r[j] * c;                          // im = a s + b c
            }
        }
    }
    if (f.quant != 0 && !kPack) {
        const int g = f.quant == 2 ? 16 : 32;
#pragma unroll
        for (int j = 0; j < kFinMax; ++j) {
            if (j >= nj) break;
            const int e = item_ct1.get_local_id(2) + j * kThreads;
            float a = e < len ? sycl::fabs(r[j]) : 0.0f;
            /*
DPCT1108:129: '__shfl_xor_sync' was migrated with the experimental feature
masked sub_group function which may not be supported by all compilers or
runtimes. You may need to adjust the code.
*/
#pragma unroll
            for (int o = g / 2; o > 0; o >>= 1) a = sycl::fmax(
                a,
                dpct::experimental::permute_sub_group_by_xor(
                    0xffffffffu,
                    sycl::ext::oneapi::this_work_item::get_sub_group(), a, o));
            if (e >= len) continue;
            if (f.quant == 1) {
                const float s =
                    pow2_ceil(sycl::fmax(a, 1e-4f) * (1.0f / 448.0f));
                r[j] =
                    fp8_rt(sycl::fmin(sycl::fmax(r[j] / s, -448.0f), 448.0f)) *
                    s;
            } else if (f.quant == 2) {
                const float s =
                    fp8_rt(sycl::fmax(a, 6.0f * 0.001953125f) / 6.0f);
                r[j] =
                    e2m1_rt(sycl::fmin(sycl::fmax(r[j] / s, -6.0f), 6.0f)) * s;
            } else {
                const float s =
                    pow2_ceil(sycl::fmax(a, 6.0f * 1.1754943508222875e-38f) *
                              (1.0f / 6.0f));
                r[j] =
                    e2m1_rt(sycl::fmin(sycl::fmax(r[j] / s, -6.0f), 6.0f)) * s;
            }
        }
    }
    if (kPack) {   // the same roundings, written as their codes and each group's scale (the readers' q * s: r exactly)
        const int g = f.quant == 2 ? 16 : 32;
        uint8_t *pr = f.pdst + (long long)item_ct1.get_group(2) * f.pdst_stride;
#pragma unroll
        for (int j = 0; j < kFinMax; ++j) {
            if (j >= nj) break;
            const int e = item_ct1.get_local_id(2) + j * kThreads;
            float a = e < len ? sycl::fabs(r[j]) : 0.0f;
            /*
DPCT1108:130: '__shfl_xor_sync' was migrated with the experimental feature
masked sub_group function which may not be supported by all compilers or
runtimes. You may need to adjust the code.
*/
#pragma unroll
            for (int o = g / 2; o > 0; o >>= 1) a = sycl::fmax(
                a,
                dpct::experimental::permute_sub_group_by_xor(
                    0xffffffffu,
                    sycl::ext::oneapi::this_work_item::get_sub_group(), a, o));
            float q = 0.0f, s = 1.0f;
            if (e < len) {
                if (f.quant == 1) {
                    s = pow2_ceil(sycl::fmax(a, 1e-4f) * (1.0f / 448.0f));
                    q = fp8_rt(
                        sycl::fmin(sycl::fmax(r[j] / s, -448.0f), 448.0f));
                } else if (f.quant == 2) {
                    s = fp8_rt(sycl::fmax(a, 6.0f * 0.001953125f) / 6.0f);
                    q = e2m1_rt(sycl::fmin(sycl::fmax(r[j] / s, -6.0f), 6.0f));
                } else {
                    s = pow2_ceil(
                        sycl::fmax(a, 6.0f * 1.1754943508222875e-38f) *
                        (1.0f / 6.0f));
                    q = e2m1_rt(sycl::fmin(sycl::fmax(r[j] / s, -6.0f), 6.0f));
                }
                r[j] = q * s;
            }
            if (f.quant == 1) {
                if (e < len) pr[e] = (uint8_t) e4m3_code(q);
            } else {
                const uint32_t c = e2m1_code(q);
                /*
                DPCT1108:131: '__shfl_down_sync' was migrated with the
                experimental feature masked sub_group function which may not be
                supported by all compilers or runtimes. You may need to adjust
                the code.
                */
                const uint32_t hi = dpct::experimental::shift_sub_group_left(
                    0xffffffffu,
                    sycl::ext::oneapi::this_work_item::get_sub_group(), c,
                    1); // (value e + 1: the next lane's)
                if (e < len && (e & 1) == 0) pr[e >> 1] = (uint8_t) (c | (hi << 4));
            }
            if (e < len && e % g == 0)
                pr[(f.quant == 1 ? len : len / 2) + e / g] =
                    (uint8_t)(f.quant == 2
                                  ? e4m3_code(s)
                                  : sycl::bit_cast<unsigned int>(s) >> 23);
        }
    }
    uint16_t *d = f.dst != nullptr
                      ? f.dst + (long long)item_ct1.get_group(2) * f.dst_stride
                      : nullptr;
#pragma unroll
    for (int j = 0; j < kFinMax; ++j) {
        if (j >= nj) break;
        const int e = item_ct1.get_local_id(2) + j * kThreads;
        if (e >= len) continue;
        if (d != nullptr) d[e] = f2bf16(r[j]);
        else if (!kPack) v[e] = r[j];
    }
}

// ---------------------------------------------------------------- hyper-connections
__dpct_inline__ void hc_mix_kernel(const float *__restrict__ h,
                                   const float *__restrict__ fn, int hcdim,
                                   float eps, float *__restrict__ mixes) {
    auto item_ct1 = sycl::ext::oneapi::this_work_item::get_nd_item<3>();
    const int m = item_ct1.get_group(2), t = item_ct1.get_group(1);
    const float* w = fn + (size_t) m * hcdim;
    const float* x = h + (size_t) t * hcdim;
    float acc = 0.0f, ss = 0.0f;
    for (int k = item_ct1.get_local_id(2) * 4; k < hcdim; k += kThreads * 4) {
        const sycl::float4 hv = *(const sycl::float4 *)(x + k);
        const sycl::float4 wv = *(const sycl::float4 *)(w + k);
        acc += hv.x() * wv.x() + hv.y() * wv.y() + hv.z() * wv.z() +
               hv.w() * wv.w();
        ss += hv.x() * hv.x() + hv.y() * hv.y() + hv.z() * hv.z() +
              hv.w() * hv.w();
    }
    acc = block_sum(acc);
    ss = block_sum(ss);
    if (item_ct1.get_local_id(2) == 0) mixes[(size_t)t * 32 + m] =
        acc * sycl::rsqrt(ss / (float)hcdim + eps);
}

// a warp a token: lane j * hc + k holds comb[j][k] (hc <= 4: lanes 0..15; wider streams fall back to lane 0 alone).  The body is
// a device function (hc_mix_kernel's last block runs it too); mx is the token's 32 mixes
/*
DPCT1110:132: The total declared local variable size in device function
hc_split_body exceeds 128 bytes and may cause high register pressure. Consult
with your hardware vendor to find the total register size available and adjust
the code, or use smaller sub-group size to avoid high register pressure.
*/
__dpct_inline__ void hc_split_body(const float *mx, int t, int lane,
                                   const float *__restrict__ scale,
                                   const float *__restrict__ base, int hc,
                                   int iters, float eps, float *pre,
                                   float *post, float *comb) {
    pre += (size_t) t * 8;
    post += (size_t) t * 8;
    comb += (size_t) t * 64;
    if (lane < hc) {
        pre[lane] = 1.0f / (1.0f + sycl::native::exp(
                                       -(mx[lane] * scale[0] + base[lane]))) +
                    eps;
        post[lane] =
            2.0f / (1.0f + sycl::native::exp(
                               -(mx[hc + lane] * scale[1] + base[hc + lane])));
    }
    if (hc == 4) {
        // rows are the 4-lane groups (j = lane / 4), columns the lanes 4 apart (k = lane % 4)
        const bool on = lane < 16;
        float c = on ? mx[8 + lane] * scale[2] + base[8 + lane] : -FLT_MAX;
        auto row_sum = [&](float v) {
            /*
            DPCT1108:133: '__shfl_xor_sync' was migrated with the experimental
            feature masked sub_group function which may not be supported by all
            compilers or runtimes. You may need to adjust the code.
            */
            v += dpct::experimental::permute_sub_group_by_xor(
                0xffffffffu, sycl::ext::oneapi::this_work_item::get_sub_group(),
                v, 1);
            /*
            DPCT1108:134: '__shfl_xor_sync' was migrated with the experimental
            feature masked sub_group function which may not be supported by all
            compilers or runtimes. You may need to adjust the code.
            */
            return v + dpct::experimental::permute_sub_group_by_xor(
                           0xffffffffu,
                           sycl::ext::oneapi::this_work_item::get_sub_group(),
                           v, 2);
        };
        auto col_sum = [&](float v) {
            /*
            DPCT1108:135: '__shfl_xor_sync' was migrated with the experimental
            feature masked sub_group function which may not be supported by all
            compilers or runtimes. You may need to adjust the code.
            */
            v += dpct::experimental::permute_sub_group_by_xor(
                0xffffffffu, sycl::ext::oneapi::this_work_item::get_sub_group(),
                v, 4);
            /*
            DPCT1108:136: '__shfl_xor_sync' was migrated with the experimental
            feature masked sub_group function which may not be supported by all
            compilers or runtimes. You may need to adjust the code.
            */
            return v + dpct::experimental::permute_sub_group_by_xor(
                           0xffffffffu,
                           sycl::ext::oneapi::this_work_item::get_sub_group(),
                           v, 8);
        };
        /*
        DPCT1108:137: '__shfl_xor_sync' was migrated with the experimental
        feature masked sub_group function which may not be supported by all
        compilers or runtimes. You may need to adjust the code.
        */
        float m = sycl::fmax(
            c, dpct::experimental::permute_sub_group_by_xor(
                   0xffffffffu,
                   sycl::ext::oneapi::this_work_item::get_sub_group(), c, 1));
        /*
        DPCT1108:138: '__shfl_xor_sync' was migrated with the experimental
        feature masked sub_group function which may not be supported by all
        compilers or runtimes. You may need to adjust the code.
        */
        m = sycl::fmax(
            m, dpct::experimental::permute_sub_group_by_xor(
                   0xffffffffu,
                   sycl::ext::oneapi::this_work_item::get_sub_group(), m, 2));
        c = on ? sycl::native::exp(c - m) : 0.0f;
        c = c / row_sum(c) + eps;                       // softmax over each row, + eps
        c = on ? c : 0.0f;
        c = c / (col_sum(c) + eps);                     // each destination column over its sources
#pragma unroll
        for (int it = 1; it < iters; ++it) {
            c = c / (row_sum(c) + eps);
            c = c / (col_sum(c) + eps);
        }
        if (on) comb[lane] = c;
        return;
    }
    if (lane != 0) return;
    float c[8][8];
    for (int j = 0; j < hc; ++j) {
        float m = -FLT_MAX;
#pragma unroll
        for (int k = 0; k < hc; ++k) {
            c[j][k] = mx[2 * hc + j * hc + k] * scale[2] + base[2 * hc + j * hc + k];
            m = sycl::fmax(m, c[j][k]);
        }
        float s = 0.0f;
#pragma unroll
        for (int k = 0; k < hc; ++k) {
            c[j][k] = sycl::native::exp(c[j][k] - m);
            s += c[j][k];
        }
#pragma unroll
        for (int k = 0; k < hc; ++k) c[j][k] = c[j][k] / s + eps;
    }
    for (int k = 0; k < hc; ++k) {
        float s = 0.0f;
#pragma unroll
        for (int j = 0; j < hc; ++j) s += c[j][k];
#pragma unroll
        for (int j = 0; j < hc; ++j) c[j][k] /= s + eps;
    }
    for (int it = 1; it < iters; ++it) {
        for (int j = 0; j < hc; ++j) {
            float s = 0.0f;
#pragma unroll
            for (int k = 0; k < hc; ++k) s += c[j][k];
#pragma unroll
            for (int k = 0; k < hc; ++k) c[j][k] /= s + eps;
        }
        for (int k = 0; k < hc; ++k) {
            float s = 0.0f;
#pragma unroll
            for (int j = 0; j < hc; ++j) s += c[j][k];
#pragma unroll
            for (int j = 0; j < hc; ++j) c[j][k] /= s + eps;
        }
    }
#pragma unroll
    for (int j = 0; j < hc; ++j)
#pragma unroll
        for (int k = 0; k < hc; ++k) comb[j * hc + k] = c[j][k];
}
__dpct_inline__ void hc_split_kernel(const float *__restrict__ mixes,
                                     const float *__restrict__ scale,
                                     const float *__restrict__ base, int hc,
                                     int iters, float eps, float *pre,
                                     float *post, float *comb) {
    auto item_ct1 = sycl::ext::oneapi::this_work_item::get_nd_item<3>();
    const int t = item_ct1.get_group(2);
    hc_split_body(mixes + (size_t)t * 32, t, item_ct1.get_local_id(2), scale,
                  base, hc, iters, eps, pre, post, comb);
}

// hc_mix_kernel with hc_split_kernel behind it: the block that finishes the token's last mix (an atomic count, reset by it) runs
// the split on warp 0 - the same arithmetic as the two launches, one launch less and no round trip through the launch gap
__dpct_inline__ void
hc_mix_split_kernel(const float *__restrict__ h, const float *__restrict__ fn,
                    int hcdim, float eps, float *mixes, unsigned *cnt,
                    const float *__restrict__ scale,
                    const float *__restrict__ base, int hc, int iters,
                    float hc_eps, float *pre, float *post, float *comb) {
    auto item_ct1 = sycl::ext::oneapi::this_work_item::get_nd_item<3>();
    const int m = item_ct1.get_group(2), t = item_ct1.get_group(1);
    const float* w = fn + (size_t) m * hcdim;
    const float* x = h + (size_t) t * hcdim;
    float acc = 0.0f, ss = 0.0f;
    for (int k = item_ct1.get_local_id(2) * 4; k < hcdim; k += kThreads * 4) {
        const sycl::float4 hv = *(const sycl::float4 *)(x + k);
        const sycl::float4 wv = *(const sycl::float4 *)(w + k);
        acc += hv.x() * wv.x() + hv.y() * wv.y() + hv.z() * wv.z() +
               hv.w() * wv.w();
        ss += hv.x() * hv.x() + hv.y() * hv.y() + hv.z() * hv.z() +
              hv.w() * hv.w();
    }
    acc = block_sum(acc);
    ss = block_sum(ss);
    auto &last = *sycl::ext::oneapi::group_local_memory_for_overwrite<unsigned>(
        sycl::ext::oneapi::this_work_item::get_work_group<3>());
    auto &smx = *sycl::ext::oneapi::group_local_memory_for_overwrite<float[32]>(
        sycl::ext::oneapi::this_work_item::get_work_group<3>());
    if (item_ct1.get_local_id(2) == 0) {
        mixes[(size_t)t * 32 + m] = acc * sycl::rsqrt(ss / (float)hcdim + eps);
        /*
        DPCT1078:140: Consider replacing memory_order::acq_rel with
        memory_order::seq_cst for correctness if strong memory order
        restrictions are needed.
        */
        sycl::atomic_fence(sycl::memory_order::acq_rel,
                           sycl::memory_scope::device);
        last =
            dpct::atomic_fetch_add<sycl::access::address_space::generic_space>(
                cnt + t, 1u) == item_ct1.get_group_range(2) - 1
                ? 1u
                : 0u;
    }
    /*
    DPCT1065:417: Consider replacing sycl::nd_item::barrier() with
    sycl::nd_item::barrier(sycl::access::fence_space::local_space) for better
    performance if there is no access to global memory.
    */
    item_ct1.barrier();
    if (!last) return;
    /*
    DPCT1078:139: Consider replacing memory_order::acq_rel with
    memory_order::seq_cst for correctness if strong memory order restrictions
    are needed.
    */
    sycl::atomic_fence(sycl::memory_order::acq_rel, sycl::memory_scope::device);
    if (item_ct1.get_local_id(2) < 32) {
        const int n = item_ct1.get_group_range(
            2); // the mixes of the token (n_mix <= 32); read past the L1
        smx[item_ct1.get_local_id(2)] =
            item_ct1.get_local_id(2) < n
                ? *(const volatile float *)(mixes + (size_t)t * 32 +
                                            item_ct1.get_local_id(2))
                : 0.0f;
        sycl::group_barrier(sycl::ext::oneapi::this_work_item::get_sub_group());
        hc_split_body(smx, t, item_ct1.get_local_id(2), scale, base, hc, iters,
                      hc_eps, pre, post, comb);
    }
    if (item_ct1.get_local_id(2) == 0) cnt[t] = 0u;
}

__dpct_inline__ void hc_pre_norm_kernel(const float *__restrict__ h,
                                        const float *__restrict__ mix,
                                        const float *__restrict__ w, int dim,
                                        int hc, float eps,
                                        float *__restrict__ x) {
    auto item_ct1 = sycl::ext::oneapi::this_work_item::get_nd_item<3>();
    const int t = item_ct1.get_group(2);
    h += (size_t) t * hc * dim;
    mix += (size_t) t * 8;
    x += (size_t) t * dim;
    float r[8];
    float ss = 0.0f;
    int n = 0;
    for (int e = item_ct1.get_local_id(2); e < dim;
         e += item_ct1.get_local_range(2), ++n) {
        float a = 0.0f;
#pragma unroll
        for (int s = 0; s < hc; ++s) a += mix[s] * h[(size_t)s * dim + e];
        r[n] = a;
        ss += a * a;
    }
    const float inv = sycl::rsqrt(block_sum(ss) / (float)dim + eps);
    n = 0;
#pragma unroll
    for (int e = item_ct1.get_local_id(2); e < dim;
         e += item_ct1.get_local_range(2), ++n)
        x[e] = w != nullptr ? r[n] * inv * w[e] : r[n];
}

__dpct_inline__ void hc_post_kernel(const float *__restrict__ y,
                                    const float *__restrict__ res,
                                    const float *__restrict__ post,
                                    const float *__restrict__ comb, int dim,
                                    int hc, float *__restrict__ out) {
    auto item_ct1 = sycl::ext::oneapi::this_work_item::get_nd_item<3>();
    const int i = item_ct1.get_group(2) * item_ct1.get_local_range(2) +
                  item_ct1.get_local_id(2),
              t = item_ct1.get_group(1);
    if (i >= dim * hc) return;
    const int k = i / dim, e = i - k * dim;
    const float* r = res + (size_t) t * hc * dim;
    float a = post[(size_t) t * 8 + k] * y[(size_t) t * dim + e];
#pragma unroll
    for (int j = 0; j < hc; ++j) a +=
        comb[(size_t)t * 64 + j * hc + k] * r[(size_t)j * dim + e];
    out[(size_t) t * hc * dim + i] = a;
}

// ---------------------------------------------------------------- gemv over F16 weights, a warp a row
__dpct_inline__ void gemv_f16_kernel(const uint16_t *__restrict__ W,
                                     const float *__restrict__ x, int n, int k,
                                     float alpha, float *__restrict__ y) {
    auto item_ct1 = sycl::ext::oneapi::this_work_item::get_nd_item<3>();
    const int row = item_ct1.get_group(2) * (kThreads / 32) +
                    (item_ct1.get_local_id(2) >> 5);
    const int lane = item_ct1.get_local_id(2) & 31, t = item_ct1.get_group(1);
    if (row >= n) return;
    const uint16_t* w = W + (size_t) row * k;
    const float* xv = x + (size_t) t * k;
    float acc = 0.0f;
    for (int c = lane * 8; c < k; c += 32 * 8) {
        const sycl::uint4 q = *(const sycl::uint4 *)(w + c);
        const sycl::half2 *hp = (const sycl::half2 *)&q;
#pragma unroll
        for (int i = 0; i < 4; ++i) {
            const sycl::float2 f =
                hp[i].template convert<float, sycl::rounding_mode::automatic>();
            acc += f.x() * xv[c + 2 * i] + f.y() * xv[c + 2 * i + 1];
        }
    }
    acc = warp_sum(acc);
    if (lane == 0) y[(size_t) t * n + row] = alpha * acc;
}
// ... for T activations (rows ldx apart, 16-byte aligned; outputs ldy apart), kRowsT of them a block (grid y): each
// weight row read once for the block's activations, theirs in float4s - each activation's sum gemv_f16_kernel's (the
// same products in the same order), so its bits too.  (A block for all of them, 16 at most, was slower than a launch
// a token: its accumulators and guards kept the loop from unrolling - 66 against 29 us at 5 tokens on an RTX 5060 Ti,
// this 9)
constexpr int kRowsT = 2;
/*
DPCT1110:141: The total declared local variable size in device function
gemv_f16_rows_kernel exceeds 128 bytes and may cause high register pressure.
Consult with your hardware vendor to find the total register size available and
adjust the code, or use smaller sub-group size to avoid high register pressure.
*/
__dpct_inline__ void gemv_f16_rows_kernel(const uint16_t *__restrict__ W,
                                          const float *__restrict__ x,
                                          size_t ldx, int n, int k, int T,
                                          float alpha, float *__restrict__ y,
                                          size_t ldy) {
    auto item_ct1 = sycl::ext::oneapi::this_work_item::get_nd_item<3>();
    const int row = item_ct1.get_group(2) * (kThreads / 32) +
                    (item_ct1.get_local_id(2) >> 5);
    const int lane = item_ct1.get_local_id(2) & 31;
    const int t0 = item_ct1.get_group(1) * kRowsT,
              nt = sycl::min(kRowsT, T - t0);
    if (row >= n) return;
    const uint16_t* w = W + (size_t) row * k;
    float acc[kRowsT];
#pragma unroll
    for (int t = 0; t < kRowsT; ++t) acc[t] = 0.0f;
    for (int c = lane * 8; c < k; c += 32 * 8) {
        const sycl::uint4 q = *(const sycl::uint4 *)(w + c);
        const sycl::half2 *hp = (const sycl::half2 *)&q;
        float xv[kRowsT][8];
#pragma unroll
        for (int t = 0; t < kRowsT; ++t)
            if (t < nt) {
                const float* xp = x + (size_t) (t0 + t) * ldx + c;
                const sycl::float4 a = *(const sycl::float4 *)xp,
                                   b = *(const sycl::float4 *)(xp + 4);
                xv[t][0] = a.x(), xv[t][1] = a.y(), xv[t][2] = a.z(),
                xv[t][3] = a.w();
                xv[t][4] = b.x(), xv[t][5] = b.y(), xv[t][6] = b.z(),
                xv[t][7] = b.w();
            }
#pragma unroll
        for (int i = 0; i < 4; ++i) {
            const sycl::float2 f =
                hp[i].template convert<float, sycl::rounding_mode::automatic>();
#pragma unroll
            for (int t = 0; t < kRowsT; ++t)
                if (t < nt) acc[t] +=
                    f.x() * xv[t][2 * i] + f.y() * xv[t][2 * i + 1];
        }
    }
#pragma unroll
    for (int t = 0; t < kRowsT; ++t)
        if (t < nt) {
            const float a = warp_sum(acc[t]);
            if (lane == 0) y[(size_t) (t0 + t) * ldy + row] = alpha * a;
        }
}

// ---------------------------------------------------------------- gemv over F32 weights, a warp a row (the router)
__dpct_inline__ void gemv_f32_kernel(const float *__restrict__ W,
                                     const float *__restrict__ x, int n, int k,
                                     float *__restrict__ y, int ldy) {
    auto item_ct1 = sycl::ext::oneapi::this_work_item::get_nd_item<3>();
    const int row = item_ct1.get_group(2) * (kThreads / 32) +
                    (item_ct1.get_local_id(2) >> 5);
    const int lane = item_ct1.get_local_id(2) & 31, t = item_ct1.get_group(1);
    if (row >= n) return;
    const sycl::float4 *w = (const sycl::float4 *)(W + (size_t)row * k);
    const sycl::float4 *xv = (const sycl::float4 *)(x + (size_t)t * k);
    float acc = 0.0f;
    for (int c = lane; c < k / 4; c += 32) {
        const sycl::float4 a = w[c], b = xv[c];
        acc = sycl::fma((float)(a.x()), (float)(b.x()), acc);
        acc = sycl::fma((float)(a.y()), (float)(b.y()), acc);
        acc = sycl::fma((float)(a.z()), (float)(b.z()), acc);
        acc = sycl::fma((float)(a.w()), (float)(b.w()), acc);
    }
    acc = warp_sum(acc);
    if (lane == 0) y[(size_t) t * ldy + row] = acc;
}

// ---------------------------------------------------------------- the router
// A pick that never came out: every score left is NaN (the gate's input went non-finite), so nothing beat -FLT_MAX and the
// index is still the 0x7fffffff it started as - and the next line read sc[0x7fffffff] (shared memory), then vram[0x7fffffff] /
// vs[0x7fffffff] in the kernels after it: a memory fault far from the cause (Project Maya v1.1.45's chain: NaN routing, then an
// illegal memory access; #4825).  The lowest expert not picked yet instead - a valid index, so nothing reads outside its tables;
// the weights stay NaN, the layer's output is garbage as it was, and the host-routed path says so (its picks are checked there).
// Only reached by a kernel's one thread when a pick is missing, so a finite router pays one compare a pick.
#if !defined(STRATA_METAL)
inline dpct::global_memory<int, 0>& g_router_bad = *new dpct::global_memory<int, 0>(0); // set by the fallback below, read and cleared by
                     // router_nonfinite_take (once a few tokens, not a token)
#endif
__dpct_inline__ int route_fallback_pick(const int *pick, int r, int n,
                                        int &g_router_bad) {
#if !defined(STRATA_METAL)
    g_router_bad = 1;
#endif
    int ix = 0;
#pragma unroll
    for (bool again = true; again;) { // (the picks are few and unsorted: start
                                      // over when one of them is the candidate)
        again = false;
#pragma unroll
        for (int q = 0; q < r && !again; ++q)
            if (pick[q] == ix) ++ix, again = true;
    }
    return ix < n ? ix : 0;
}
__dpct_inline__ void route_kernel(const float *__restrict__ logits,
                                  const float *__restrict__ bias, int n, int k,
                                  float route_scale, int *idx, float *w,
                                  const float *__restrict__ bias_vl,
                                  const VlBits vl, int &g_router_bad) {
    auto item_ct1 = sycl::ext::oneapi::this_work_item::get_nd_item<3>();
auto &sc = *sycl::ext::oneapi::group_local_memory_for_overwrite<float[1024]>(
    sycl::ext::oneapi::this_work_item::get_work_group<3>());
    auto &bs =
        *sycl::ext::oneapi::group_local_memory_for_overwrite<float[1024]>(
            sycl::ext::oneapi::this_work_item::get_work_group<3>());
    auto &bv = *sycl::ext::oneapi::group_local_memory_for_overwrite<float[32]>(
        sycl::ext::oneapi::this_work_item::get_work_group<3>());
    auto &bi = *sycl::ext::oneapi::group_local_memory_for_overwrite<int[32]>(
        sycl::ext::oneapi::this_work_item::get_work_group<3>());
    auto &pick = *sycl::ext::oneapi::group_local_memory_for_overwrite<int[16]>(
        sycl::ext::oneapi::this_work_item::get_work_group<3>());
    const int t = item_ct1.get_group(2);
    logits += (size_t) t * 1024;
    if (bias_vl != nullptr && ((vl.w[t >> 5] >> (t & 31)) & 1u)) bias = bias_vl;
    for (int i = item_ct1.get_local_id(2); i < n;
         i += item_ct1.get_local_range(2)) {
        const float l = logits[i];
        const float sp = l > 20.0f ? l : sycl::log1p(sycl::native::exp(l));
        sc[i] = sycl::sqrt((float)sp);
        bs[i] = sc[i] + bias[i];
    }
    /*
    DPCT1065:418: Consider replacing sycl::nd_item::barrier() with
    sycl::nd_item::barrier(sycl::access::fence_space::local_space) for better
    performance if there is no access to global memory.
    */
    item_ct1.barrier();
    const int lane = item_ct1.get_local_id(2) & 31,
              wp = item_ct1.get_local_id(2) >> 5;
    for (int r = 0; r < k; ++r) {
        float best = -FLT_MAX;
        int bix = 0x7fffffff;
#pragma unroll
        for (int i = item_ct1.get_local_id(2); i < n;
             i += item_ct1.get_local_range(2))
            if (bs[i] > best || (bs[i] == best && i < bix)) best = bs[i], bix = i;
        for (int o = 16; o > 0; o >>= 1) {
            /*
            DPCT1108:144: '__shfl_xor_sync' was migrated with the experimental
            feature masked sub_group function which may not be supported by all
            compilers or runtimes. You may need to adjust the code.
            */
            const float ov = dpct::experimental::permute_sub_group_by_xor(
                0xffffffffu, sycl::ext::oneapi::this_work_item::get_sub_group(),
                best, o);
            /*
            DPCT1108:145: '__shfl_xor_sync' was migrated with the experimental
            feature masked sub_group function which may not be supported by all
            compilers or runtimes. You may need to adjust the code.
            */
            const int oi = dpct::experimental::permute_sub_group_by_xor(
                0xffffffffu, sycl::ext::oneapi::this_work_item::get_sub_group(),
                bix, o);
            if (ov > best || (ov == best && oi < bix)) best = ov, bix = oi;
        }
        if (lane == 0) bv[wp] = best, bi[wp] = bix;
        /*
        DPCT1118:142: SYCL group functions and algorithms must be encountered in
        converged control flow. You may need to adjust the code.
        */
        /*
        DPCT1065:419: Consider replacing sycl::nd_item::barrier() with
        sycl::nd_item::barrier(sycl::access::fence_space::local_space) for
        better performance if there is no access to global memory.
        */
        item_ct1.barrier();
        if (item_ct1.get_local_id(2) == 0) {
            float b = -FLT_MAX;
            int ix = 0x7fffffff;
#pragma unroll
            for (int q = 0; q < (int)(item_ct1.get_local_range(2) >> 5); ++q)
                if (bv[q] > b || (bv[q] == b && bi[q] < ix)) b = bv[q], ix = bi[q];
            if (ix >= n) ix = route_fallback_pick(
                pick, r, n, g_router_bad); // (a non-finite gate input: see
                                           // route_fallback_pick)
            pick[r] = ix;
            bs[ix] = -FLT_MAX;
        }
        /*
        DPCT1118:143: SYCL group functions and algorithms must be encountered in
        converged control flow. You may need to adjust the code.
        */
        /*
        DPCT1065:420: Consider replacing sycl::nd_item::barrier() with
        sycl::nd_item::barrier(sycl::access::fence_space::local_space) for
        better performance if there is no access to global memory.
        */
        item_ct1.barrier();
    }
    if (item_ct1.get_local_id(2) == 0) {
        float s = 0.0f;
#pragma unroll
        for (int r = 0; r < k; ++r) s += sc[pick[r]];
#pragma unroll
        for (int r = 0; r < k; ++r) {
            idx[(size_t) t * kPicks + r] = pick[r];
            w[(size_t) t * kPicks + r] = sc[pick[r]] / (s + 1e-20f) * route_scale;
        }
    }
}

__dpct_inline__ void swiglu_kernel(const float *__restrict__ g,
                                   const float *__restrict__ u, int n,
                                   float lim, float *__restrict__ h) {
    auto item_ct1 = sycl::ext::oneapi::this_work_item::get_nd_item<3>();
    const int i = item_ct1.get_group(2) * item_ct1.get_local_range(2) +
                  item_ct1.get_local_id(2);
    if (i >= n) return;
    float gv = g[i], uv = u[i];
    if (lim > 0.0f) {
        gv = sycl::fmin(gv, lim);
        uv = sycl::fmin(sycl::fmax(uv, -lim), lim);
    }
    h[i] = gv / (1.0f + sycl::native::exp(-gv)) * uv;
}

// ---------------------------------------------------------------- the compressor (ratio > 1), a window in order
__dpct_inline__ void compress_kernel(const float *__restrict__ kv,
                                     const float *__restrict__ score,
                                     float *kv_state, float *score_state,
                                     int ratio, int p0, int T, int d,
                                     const float *__restrict__ w, float eps,
                                     float *__restrict__ latent) {
    auto item_ct1 = sycl::ext::oneapi::this_work_item::get_nd_item<3>();
    int g = 0;
    for (int t = 0; t < T; ++t) {
        const int slot = (p0 + t) % ratio;
        const bool pool = (p0 + t + 1) % ratio == 0;
#pragma unroll
        for (int e = item_ct1.get_local_id(2); e < d;
             e += item_ct1.get_local_range(2)) {
            kv_state[(size_t) slot * d + e] = kv[(size_t) t * d + e];
            score_state[(size_t) slot * d + e] = score[(size_t) t * d + e];
        }
        /*
        DPCT1118:146: SYCL group functions and algorithms must be encountered in
        converged control flow. You may need to adjust the code.
        */
        /*
        DPCT1065:421: Consider replacing sycl::nd_item::barrier() with
        sycl::nd_item::barrier(sycl::access::fence_space::local_space) for
        better performance if there is no access to global memory.
        */
        item_ct1.barrier();
        if (!pool) continue;
        // (kv * softmax(score, over the slots)).sum: each slot's probability first, then the sum, as torch does
        float val[2] = {0.0f, 0.0f};   // d <= 2 * blockDim
        float ss = 0.0f;
        int n = 0;
        for (int e = item_ct1.get_local_id(2); e < d;
             e += item_ct1.get_local_range(2), ++n) {
            float m = -FLT_MAX;
#pragma unroll
            for (int s = 0; s < ratio; ++s)
                m = sycl::fmax(m, score_state[(size_t)s * d + e]);
            float den = 0.0f;
#pragma unroll
            for (int s = 0; s < ratio; ++s) den +=
                sycl::native::exp(score_state[(size_t)s * d + e] - m);
            float acc = 0.0f;
#pragma unroll
            for (int s = 0; s < ratio; ++s)
                acc += kv_state[(size_t)s * d + e] *
                       (sycl::native::exp(score_state[(size_t)s * d + e] - m) /
                        den);
            val[n] = acc;
            ss += acc * acc;
        }
        const float inv = sycl::rsqrt(block_sum(ss) / (float)d + eps);
        n = 0;
#pragma unroll
        for (int e = item_ct1.get_local_id(2); e < d;
             e += item_ct1.get_local_range(2), ++n)
            latent[(size_t)g * d + e] = val[n] * inv * w[e];
        ++g;
        /*
        DPCT1118:147: SYCL group functions and algorithms must be encountered in
        converged control flow. You may need to adjust the code.
        */
        /*
        DPCT1065:422: Consider replacing sycl::nd_item::barrier() with
        sycl::nd_item::barrier(sycl::access::fence_space::local_space) for
        better performance if there is no access to global memory.
        */
        item_ct1.barrier();
    }
}

// ---------------------------------------------------------------- per-query sizes
__dpct_inline__ void rows_at(const Rows &r, int t, int &n, int &k) {
    if (r.gen == 0) {
        n = r.n[t];
        k = r.k[t];
        return;
    }
    const int nn = (int) ((r.p0 + t + 1) / r.R);
    const int cb = sycl::min(r.ccap, (nn + r.blk - 1) / r.blk);
    switch (r.gen) {
    case 1: n = nn, k = sycl::min(r.kcap, nn); break;
    case 2: n = nn, k = cb; break;
    case 3: n = cb * r.blk, k = sycl::min(r.kcap, nn); break;
    default: n = (nn + r.blk - 1) / r.blk, k = r.kcap; break;
    }
}

// ---------------------------------------------------------------- the indexer
// a thread a position: its key row (KF: kKvFp4p packed, or BF16) against every head's query (shared memory), relu,
// weighted sum
template <int HD, int KF>
/*
DPCT1110:148: The total declared local variable size in device function
index_score_kernel exceeds 128 bytes and may cause high register pressure.
Consult with your hardware vendor to find the total register size available and
adjust the code, or use smaller sub-group size to avoid high register pressure.
*/
__dpct_inline__ void
index_score_kernel(const float *__restrict__ q, const float *__restrict__ w,
                   const uint8_t *__restrict__ kc, int heads, int n,
                   const int *__restrict__ cand, int cand_stride, int blk,
                   Rows lim, float *__restrict__ out, long long out_stride,
                   uint8_t *dpct_local) {
    auto item_ct1 = sycl::ext::oneapi::this_work_item::get_nd_item<3>();
    auto qs = (float *)dpct_local; // heads * HD, then heads weights
    const int t = item_ct1.get_group(1);
#pragma unroll
    for (int i = item_ct1.get_local_id(2); i < heads * HD;
         i += item_ct1.get_local_range(2))
        qs[i] = q[(size_t)t * heads * HD + i];
#pragma unroll
    for (int i = item_ct1.get_local_id(2); i < heads;
         i += item_ct1.get_local_range(2))
        qs[heads * HD + i] = w[(size_t)t * 64 + i];
    /*
    DPCT1113:423: Consider replacing
    sycl::nd_item::barrier(sycl::access::fence_space::local_space) with
    sycl::nd_item::barrier() if function "index_score_kernel" is called in a
    multidimensional kernel.
    */
    item_ct1.barrier(sycl::access::fence_space::local_space);
    const int i = item_ct1.get_group(2) * item_ct1.get_local_range(2) +
                  item_ct1.get_local_id(2);
    if (i >= n) return;
    float* o = out + (long long) t * out_stride;
    int ln, lk;
    rows_at(lim, t, ln, lk);
    int p = i;
    if (cand != nullptr) {
        if (i >= lk * blk) {
            o[i] = -INFINITY;
            return;
        }
        p = cand[(size_t) t * cand_stride + i / blk] * blk + i % blk;
    }
    if (p >= ln) {
        o[i] = -INFINITY;
        return;
    }
    const uint8_t* kr = kc + (size_t) p * kv_row_bytes(KF, HD);
    float acc[32];
#pragma unroll
    for (int h = 0; h < 32; ++h) acc[h] = 0.0f;
#pragma unroll 2
    for (int c = 0; c < HD; c += 8) {
        float kv[8];
        if (KF == kKvFp4p) {   // 8 codes in 4 bytes, the group's power-of-two scale: the values the finish rounded to
            const uint32_t cw = *(const uint32_t*) (kr + c / 2);
            const float ks = ue8m0_val(kr[HD / 2 + c / 32]);
#pragma unroll
            for (int x = 0; x < 8; ++x) kv[x] = e2m1_val14(cw >> (4 * x)) * 16384.0f * ks;   // (any 2^k: exact)
        } else {
            const sycl::uint4 raw = *(const sycl::uint4 *)(kr + c * 2);
            const uint16_t* b = (const uint16_t*) &raw;
#pragma unroll
            for (int x = 0; x < 8; ++x) kv[x] = bf16f(b[x]);
        }
#pragma unroll
        for (int h = 0; h < 32; ++h) {
            if (h >= heads) break;
            const float* qh = qs + h * HD + c;
#pragma unroll
            for (int x = 0; x < 8; ++x) acc[h] += qh[x] * kv[x];
        }
    }
    float s = 0.0f;
#pragma unroll
    for (int h = 0; h < 32; ++h) {
        if (h >= heads) break;
        s += sycl::fmax(acc[h], 0.0f) * qs[heads * HD + h];
    }
    o[i] = s;
}

__dpct_inline__ uint32_t ord_key(float f) {
    const uint32_t b = sycl::bit_cast<unsigned int>(f);
    return (b & 0x80000000u) ? ~b : (b | 0x80000000u);
}

// a block a row: the k-th largest key by four 8-bit radix passes, then everything above it and enough of its ties
/*
DPCT1110:149: The total declared local variable size in device function
topk_kernel exceeds 128 bytes and may cause high register pressure. Consult with
your hardware vendor to find the total register size available and adjust the
code, or use smaller sub-group size to avoid high register pressure.
*/
__dpct_inline__ void topk_kernel(const float *__restrict__ vals,
                                 long long stride, Rows r,
                                 const int *__restrict__ cand, int cand_stride,
                                 int blk, int *__restrict__ out,
                                 int out_stride) {
    auto item_ct1 = sycl::ext::oneapi::this_work_item::get_nd_item<3>();
auto &hist =
    *sycl::ext::oneapi::group_local_memory_for_overwrite<unsigned[256]>(
        sycl::ext::oneapi::this_work_item::get_work_group<3>());
    auto &prefix_s =
        *sycl::ext::oneapi::group_local_memory_for_overwrite<uint32_t>(
            sycl::ext::oneapi::this_work_item::get_work_group<3>());
    auto &mask_s =
        *sycl::ext::oneapi::group_local_memory_for_overwrite<uint32_t>(
            sycl::ext::oneapi::this_work_item::get_work_group<3>());
    auto &remaining_s =
        *sycl::ext::oneapi::group_local_memory_for_overwrite<int>(
            sycl::ext::oneapi::this_work_item::get_work_group<3>());
    auto &cnt_gt = *sycl::ext::oneapi::group_local_memory_for_overwrite<int>(
        sycl::ext::oneapi::this_work_item::get_work_group<3>());
    auto &cnt_eq = *sycl::ext::oneapi::group_local_memory_for_overwrite<int>(
        sycl::ext::oneapi::this_work_item::get_work_group<3>());
    const int t = item_ct1.get_group(2);
    int n, k;
    rows_at(r, t, n, k);
    vals += (long long) t * stride;
    out += (size_t) t * out_stride;
    const int* cd = cand != nullptr ? cand + (size_t) t * cand_stride : nullptr;
    if (n <= 0) return;
    if (k >= n) {
#pragma unroll
        for (int i = item_ct1.get_local_id(2); i < n;
             i += item_ct1.get_local_range(2))
            out[i] = cd != nullptr ? cd[i / blk] * blk + i % blk : i;
        return;
    }
    if (item_ct1.get_local_id(2) == 0) {
        prefix_s = 0u;
        mask_s = 0u;
        remaining_s = k;
        cnt_gt = 0;
        cnt_eq = 0;
    }
    /*
    DPCT1065:424: Consider replacing sycl::nd_item::barrier() with
    sycl::nd_item::barrier(sycl::access::fence_space::local_space) for better
    performance if there is no access to global memory.
    */
    item_ct1.barrier();
    for (int shift = 24; shift >= 0; shift -= 8) {
#pragma unroll
        for (int i = item_ct1.get_local_id(2); i < 256;
             i += item_ct1.get_local_range(2)) hist[i] = 0u;
        /*
        DPCT1118:150: SYCL group functions and algorithms must be encountered in
        converged control flow. You may need to adjust the code.
        */
        /*
        DPCT1065:425: Consider replacing sycl::nd_item::barrier() with
        sycl::nd_item::barrier(sycl::access::fence_space::local_space) for
        better performance if there is no access to global memory.
        */
        item_ct1.barrier();
        const uint32_t prefix = prefix_s, mask = mask_s;
        for (int i = item_ct1.get_local_id(2); i < n;
             i += item_ct1.get_local_range(2)) {
            const uint32_t u = ord_key(vals[i]);
            if ((u & mask) == prefix) dpct::atomic_fetch_add<
                sycl::access::address_space::generic_space>(
                &hist[(u >> shift) & 0xffu], 1u);
        }
        /*
        DPCT1118:151: SYCL group functions and algorithms must be encountered in
        converged control flow. You may need to adjust the code.
        */
        /*
        DPCT1065:426: Consider replacing sycl::nd_item::barrier() with
        sycl::nd_item::barrier(sycl::access::fence_space::local_space) for
        better performance if there is no access to global memory.
        */
        item_ct1.barrier();
        if (item_ct1.get_local_id(2) == 0) {
            const int rem = remaining_s;
            int cum = 0;
            for (int d = 255; d >= 0; --d) {
                const int h = (int) hist[d];
                if (cum + h >= rem) {
                    prefix_s = prefix | ((uint32_t) d << shift);
                    mask_s = mask | (0xffu << shift);
                    remaining_s = rem - cum;
                    break;
                }
                cum += h;
            }
        }
        /*
        DPCT1118:152: SYCL group functions and algorithms must be encountered in
        converged control flow. You may need to adjust the code.
        */
        /*
        DPCT1065:427: Consider replacing sycl::nd_item::barrier() with
        sycl::nd_item::barrier(sycl::access::fence_space::local_space) for
        better performance if there is no access to global memory.
        */
        item_ct1.barrier();
    }
    const uint32_t thr = prefix_s;
    const int remaining = remaining_s;
    const int n_gt = k - remaining;
    // the picks in index order (a scan a pass of 1024 values), so the attention sums them in the same order every run
    // and the ties kept are the lowest positions
    auto &wgt = *sycl::ext::oneapi::group_local_memory_for_overwrite<int[32]>(
        sycl::ext::oneapi::this_work_item::get_work_group<3>());
    auto &weq = *sycl::ext::oneapi::group_local_memory_for_overwrite<int[32]>(
        sycl::ext::oneapi::this_work_item::get_work_group<3>());
    const int lane = item_ct1.get_local_id(2) & 31,
              wid = item_ct1.get_local_id(2) >> 5,
              nw = item_ct1.get_local_range(2) >> 5;
    const unsigned below = (1u << lane) - 1u;
    for (int i0 = 0; i0 < n; i0 += item_ct1.get_local_range(2)) {
        const int i = i0 + item_ct1.get_local_id(2);
        const uint32_t u = i < n ? ord_key(vals[i]) : 0u;
        const bool gt = i < n && u > thr, eq = i < n && u == thr;
        const unsigned
            bg = sycl::reduce_over_group(
                sycl::ext::oneapi::this_work_item::get_sub_group(),
                (0xffffffffu &
                 (0x1 << sycl::ext::oneapi::this_work_item::get_sub_group()
                             .get_local_linear_id())) &&
                        gt
                    ? (0x1 << sycl::ext::oneapi::this_work_item::get_sub_group()
                                  .get_local_linear_id())
                    : 0,
                sycl::ext::oneapi::plus<>()),
            be = sycl::reduce_over_group(
                sycl::ext::oneapi::this_work_item::get_sub_group(),
                (0xffffffffu &
                 (0x1 << sycl::ext::oneapi::this_work_item::get_sub_group()
                             .get_local_linear_id())) &&
                        eq
                    ? (0x1 << sycl::ext::oneapi::this_work_item::get_sub_group()
                                  .get_local_linear_id())
                    : 0,
                sycl::ext::oneapi::plus<>());
        if (lane == 0) {
            wgt[wid] = sycl::popcount(bg);
            weq[wid] = sycl::popcount(be);
        }
        /*
        DPCT1118:153: SYCL group functions and algorithms must be encountered in
        converged control flow. You may need to adjust the code.
        */
        /*
        DPCT1065:428: Consider replacing sycl::nd_item::barrier() with
        sycl::nd_item::barrier(sycl::access::fence_space::local_space) for
        better performance if there is no access to global memory.
        */
        item_ct1.barrier();
        if (wid == 0) {
            const int g0 = lane < nw ? wgt[lane] : 0, e0 = lane < nw ? weq[lane] : 0;
            int gs = g0, es = e0;
            for (int o = 1; o < 32; o <<= 1) {
                /*
                DPCT1108:156: '__shfl_up_sync' was migrated with the
                experimental feature masked sub_group function which may not be
                supported by all compilers or runtimes. You may need to adjust
                the code.
                */
                const int
                    a = dpct::experimental::shift_sub_group_right(
                        0xffffffffu,
                        sycl::ext::oneapi::this_work_item::get_sub_group(), gs,
                        o),
                    b = dpct::experimental::shift_sub_group_right(
                        0xffffffffu,
                        sycl::ext::oneapi::this_work_item::get_sub_group(), es,
                        o);
                if (lane >= o) {
                    gs += a;
                    es += b;
                }
            }
            if (lane < nw) {
                wgt[lane] = cnt_gt + gs - g0;
                weq[lane] = cnt_eq + es - e0;
            }
            sycl::group_barrier(
                sycl::ext::oneapi::this_work_item::get_sub_group());
            if (lane == 31) {
                cnt_gt += gs;
                cnt_eq += es;
            }
        }
        /*
        DPCT1118:154: SYCL group functions and algorithms must be encountered in
        converged control flow. You may need to adjust the code.
        */
        /*
        DPCT1065:429: Consider replacing sycl::nd_item::barrier() with
        sycl::nd_item::barrier(sycl::access::fence_space::local_space) for
        better performance if there is no access to global memory.
        */
        item_ct1.barrier();
        int slot = -1;
        if (gt) slot = wgt[wid] + sycl::popcount(bg & below);
        else if (eq) {
            const int e = weq[wid] + sycl::popcount(be & below);
            if (e < remaining) slot = n_gt + e;
        }
        if (slot >= 0) out[slot] = cd != nullptr ? cd[i / blk] * blk + i % blk : i;
        /*
        DPCT1118:155: SYCL group functions and algorithms must be encountered in
        converged control flow. You may need to adjust the code.
        */
        /*
        DPCT1065:430: Consider replacing sycl::nd_item::barrier() with
        sycl::nd_item::barrier(sycl::access::fence_space::local_space) for
        better performance if there is no access to global memory.
        */
        item_ct1.barrier();
    }
}

__dpct_inline__ void block_max_kernel(const float *__restrict__ score,
                                      long long stride, Rows r, int blk,
                                      float *__restrict__ bm,
                                      long long bm_stride) {
    auto item_ct1 = sycl::ext::oneapi::this_work_item::get_nd_item<3>();
    const int b = item_ct1.get_group(2) * item_ct1.get_local_range(2) +
                  item_ct1.get_local_id(2),
              t = item_ct1.get_group(1);
    int n, k_;
    rows_at(r, t, n, k_);
    const int nb = (n + blk - 1) / blk;
    if (b >= nb) return;
    float m = -INFINITY;
    for (int j = 0; j < blk; ++j) {
        const int p = b * blk + j;
        if (p < n) m = sycl::fmax(m, score[(long long)t * stride + p]);
    }
    if (b == (n - 1) / blk) m = INFINITY;   // the newest position's block is pinned in
    bm[(long long) t * bm_stride + b] = m;
}

// ---------------------------------------------------------------- sparse attention, a block a (head, query)
constexpr int kAttTile = 64;

// kContig: query t's selected rows are ckv's rows t * sel_stride .. (the scratch of rows decoded once a query) - read
// without sel, a warp's rows of a tile loaded kPre at a time before their dots (0: as the BF16 kernel); the
// arithmetic the same
template <bool kContig, int kPre = 0>
/*
DPCT1110:157: The total declared local variable size in device function
sparse_attn_kernel exceeds 128 bytes and may cause high register pressure.
Consult with your hardware vendor to find the total register size available and
adjust the code, or use smaller sub-group size to avoid high register pressure.
*/
__dpct_inline__ void sparse_attn_kernel(
    const float *__restrict__ q, const uint16_t *__restrict__ win,
    const uint16_t *__restrict__ ckv, const int *__restrict__ sel,
    int sel_stride, AttnRows ar, const float *__restrict__ sink, int heads,
    int d, float scale, float *__restrict__ out) {
    // d == 512: a thread owns 2 output values, a lane 16 of a row's dot
    auto item_ct1 = sycl::ext::oneapi::this_work_item::get_nd_item<3>();
    const int h = item_ct1.get_group(2), t = item_ct1.get_group(1);
    int n_win, n_sel, w0;
    if (ar.gen) {
        const long long ta = ar.t0 + t, pa = ar.p0 + ta;
        const long long first =
            sycl::max((long long)ar.win - ar.p0,
                      ta + 1); // the window buffer row of max(0, pa - win + 1)
        w0 = (int) first;
        n_win = (int) (ar.win + ta - first + 1);
        n_sel =
            ar.R > 0 ? (int)sycl::min((long long)ar.kcap, (pa + 1) / ar.R) : 0;
    } else {
        n_win = ar.nwin[t], n_sel = ar.nsel[t], w0 = ar.win0[t];
    }
    const uint16_t* wbase = win + (size_t) w0 * d;
    const int* sl = sel + (size_t) t * sel_stride;
    auto &qs = *sycl::ext::oneapi::group_local_memory_for_overwrite<float[512]>(
        sycl::ext::oneapi::this_work_item::get_work_group<3>());
    auto &sc =
        *sycl::ext::oneapi::group_local_memory_for_overwrite<float[kAttTile]>(
            sycl::ext::oneapi::this_work_item::get_work_group<3>());
    auto &rows = *sycl::ext::oneapi::group_local_memory_for_overwrite<
        const uint16_t *[kAttTile]>(
        sycl::ext::oneapi::this_work_item::get_work_group<3>());
    auto &m_run_s = *sycl::ext::oneapi::group_local_memory_for_overwrite<float>(
        sycl::ext::oneapi::this_work_item::get_work_group<3>());
    auto &tile_scale_s =
        *sycl::ext::oneapi::group_local_memory_for_overwrite<float>(
            sycl::ext::oneapi::this_work_item::get_work_group<3>());
    auto &den_s = *sycl::ext::oneapi::group_local_memory_for_overwrite<float>(
        sycl::ext::oneapi::this_work_item::get_work_group<3>());
#pragma unroll
    for (int i = item_ct1.get_local_id(2); i < d;
         i += item_ct1.get_local_range(2))
        qs[i] = q[((size_t)t * heads + h) * d + i];
    if (item_ct1.get_local_id(2) == 0) {
        m_run_s = -1e30f;
        den_s = 0.0f;
    }
    /*
    DPCT1065:431: Consider replacing sycl::nd_item::barrier() with
    sycl::nd_item::barrier(sycl::access::fence_space::local_space) for better
    performance if there is no access to global memory.
    */
    item_ct1.barrier();
    const int lane = item_ct1.get_local_id(2) & 31,
              wp = item_ct1.get_local_id(2) >> 5;
    float o0 = 0.0f, o1 = 0.0f;
    const int total = n_win + n_sel;
    for (int t0 = 0; t0 < total; t0 += kAttTile) {
        const int nt = sycl::min(kAttTile, total - t0);
        if (item_ct1.get_local_id(2) < nt) {
            const int j = t0 + item_ct1.get_local_id(2);
            rows[item_ct1.get_local_id(2)] =
                j < n_win
                    ? wbase + (size_t)j * d
                    : ckv + (kContig ? (size_t)t * sel_stride + (j - n_win)
                                     : (size_t)sl[j - n_win]) *
                                d;
        }
        /*
        DPCT1118:158: SYCL group functions and algorithms must be encountered in
        converged control flow. You may need to adjust the code.
        */
        /*
        DPCT1065:432: Consider replacing sycl::nd_item::barrier() with
        sycl::nd_item::barrier(sycl::access::fence_space::local_space) for
        better performance if there is no access to global memory.
        */
        item_ct1.barrier();
        if (kContig && kPre > 0) {   // the warp's rows of the tile (wp, wp + 8, ..), kPre loads at a time, then dots
#pragma unroll
            for (int k0 = 0; k0 < kAttTile / 8; k0 += (kPre > 0 ? kPre : 1)) {
                sycl::uint4 ra[kPre > 0 ? kPre : 1], rb[kPre > 0 ? kPre : 1];
#pragma unroll
                for (int k = 0; k < kPre; ++k) {
                    const int jj = wp + 8 * (k0 + k);
                    if (jj < nt) {
                        ra[k] = *(const sycl::uint4 *)(rows[jj] + lane * 16);
                        rb[k] =
                            *(const sycl::uint4 *)(rows[jj] + lane * 16 + 8);
                    }
                }
#pragma unroll
                for (int k = 0; k < kPre; ++k) {
                    const int jj = wp + 8 * (k0 + k);
                    if (jj < nt) {
                        const uint16_t* pa = (const uint16_t*) &ra[k];
                        const uint16_t* pb = (const uint16_t*) &rb[k];
                        float s = 0.0f;
#pragma unroll
                        for (int x = 0; x < 8; ++x) s += qs[lane * 16 + x] * bf16f(pa[x]);
#pragma unroll
                        for (int x = 0; x < 8; ++x) s += qs[lane * 16 + 8 + x] * bf16f(pb[x]);
                        s = warp_sum(s);
                        if (lane == 0) sc[jj] = s * scale;
                    }
                }
            }
        } else
            for (int jj = wp; jj < nt; jj += kThreads / 32) {
                const uint16_t* rr = rows[jj];
                const sycl::uint4 a = *(const sycl::uint4 *)(rr + lane * 16);
                const sycl::uint4 b =
                    *(const sycl::uint4 *)(rr + lane * 16 + 8);
                const uint16_t* pa = (const uint16_t*) &a;
                const uint16_t* pb = (const uint16_t*) &b;
                float s = 0.0f;
#pragma unroll
                for (int x = 0; x < 8; ++x) s += qs[lane * 16 + x] * bf16f(pa[x]);
#pragma unroll
                for (int x = 0; x < 8; ++x) s += qs[lane * 16 + 8 + x] * bf16f(pb[x]);
                s = warp_sum(s);
                if (lane == 0) sc[jj] = s * scale;
            }
        /*
        DPCT1118:159: SYCL group functions and algorithms must be encountered in
        converged control flow. You may need to adjust the code.
        */
        /*
        DPCT1065:433: Consider replacing sycl::nd_item::barrier() with
        sycl::nd_item::barrier(sycl::access::fence_space::local_space) for
        better performance if there is no access to global memory.
        */
        item_ct1.barrier();
        if (wp == 0) {
            float m = -INFINITY;
#pragma unroll
            for (int j = lane; j < nt; j += 32) m = sycl::fmax(m, sc[j]);
            m = warp_max(m);
            const float m_old = m_run_s;
            const float m_new = sycl::fmax(m_old, m);
            float s = 0.0f;
            for (int j = lane; j < nt; j += 32) {
                const float p = sycl::native::exp(sc[j] - m_new);
                sc[j] = p;
                s += p;
            }
            s = warp_sum(s);
            sycl::group_barrier(
                sycl::ext::oneapi::this_work_item::get_sub_group());
            if (lane == 0) {
                tile_scale_s = sycl::native::exp(m_old - m_new);
                den_s = den_s * tile_scale_s + s;
                m_run_s = m_new;
            }
        }
        /*
        DPCT1118:160: SYCL group functions and algorithms must be encountered in
        converged control flow. You may need to adjust the code.
        */
        /*
        DPCT1065:434: Consider replacing sycl::nd_item::barrier() with
        sycl::nd_item::barrier(sycl::access::fence_space::local_space) for
        better performance if there is no access to global memory.
        */
        item_ct1.barrier();
        const float cs = tile_scale_s;
        o0 *= cs;
        o1 *= cs;
        const int c = item_ct1.get_local_id(2) * 2;
        if (kContig && kPre > 0) {   // (eight rows' loads at once)
#pragma unroll 8
            for (int j = 0; j < nt; ++j) {
                const uint32_t pr = *(const uint32_t*) (rows[j] + c);
                o0 += sc[j] * bf16f((uint16_t) (pr & 0xffffu));
                o1 += sc[j] * bf16f((uint16_t) (pr >> 16));
            }
        } else
            for (int j = 0; j < nt; ++j) {
                const uint32_t pr = *(const uint32_t*) (rows[j] + c);
                o0 += sc[j] * bf16f((uint16_t) (pr & 0xffffu));
                o1 += sc[j] * bf16f((uint16_t) (pr >> 16));
            }
        /*
        DPCT1118:161: SYCL group functions and algorithms must be encountered in
        converged control flow. You may need to adjust the code.
        */
        /*
        DPCT1065:435: Consider replacing sycl::nd_item::barrier() with
        sycl::nd_item::barrier(sycl::access::fence_space::local_space) for
        better performance if there is no access to global memory.
        */
        item_ct1.barrier();
    }
    const float den = den_s + sycl::native::exp(sink[h] - m_run_s);
    const int c = item_ct1.get_local_id(2) * 2;
    float* ov = out + ((size_t) t * heads + h) * d;
    ov[c] = o0 / den;
    ov[c + 1] = o1 / den;
}

// ---------------------------------------------------------------- sparse attention over the packed compressed rows
// G heads of one query a block (8 / G warps a head): each pass stages 16 of a tile's rows in shared memory as BF16
// (the window's rows as they are; the packed rows decoded - exact in BF16), once for all G heads, through one 16-byte
// load a thread.  Each head's arithmetic is sparse_attn_kernel's, operation for operation: its tiles of 64 rows, its
// dots by a warp's lanes (16 values each, in order) and the same butterfly, its softmax by one warp the same way, each
// output value's weighted sum by the same thread in row order - the same bits for any G.
constexpr int kStage = 16;   // rows a pass stages (a thread: 32 values of one)

// values part * 32 .. + 32 of a packed compressed row (kKvFp4, 512 values) as BF16 into dst[0 .. 4) (exact: the
// values are E2M1 times an E4M3 scale)
__dpct_inline__ void unpack32(const uint8_t *rr, int part, sycl::uint4 *dst) {
    const sycl::uint4 cw =
        *(const sycl::uint4 *)(rr + part * 16); // 32 codes, two scale groups
    const float k0 = e4m3_value(rr[256 + part * 2]) * 16384.0f, k1 = e4m3_value(rr[256 + part * 2 + 1]) * 16384.0f;
    const uint32_t wv[4] = {cw.x(), cw.y(), cw.z(), cw.w()};
    uint32_t o[16];
#pragma unroll
    for (int i = 0; i < 16; ++i) {   // values 2i, 2i + 1 (the low and the high nibble of byte i)
        const uint32_t b = wv[i >> 2] >> (8 * (i & 3));
        const float ks = i < 8 ? k0 : k1;
        const float lo = e2m1_val14(b) * ks, hi = e2m1_val14(b >> 4) * ks;   // (exact: their BF16 bits)
        o[i] = (sycl::bit_cast<unsigned int>(lo) >> 16) |
               (sycl::bit_cast<unsigned int>(hi) & 0xffff0000u);
    }
    dst[0] = sycl::uint4(o[0], o[1], o[2], o[3]);
    dst[1] = sycl::uint4(o[4], o[5], o[6], o[7]);
    dst[2] = sycl::uint4(o[8], o[9], o[10], o[11]);
    dst[3] = sycl::uint4(o[12], o[13], o[14], o[15]);
}

// a few queries' attention (a decode step, a short window): the rows each query selects decoded once to BF16 rows -
// out[t * cap + i] = ckv[sel[t][i]] for i < the query's n_sel - for the BF16 kernel reading them in place (once a
// query, not once a head)
__dpct_inline__ void unpack_sel_kernel(const uint8_t *__restrict__ ckv,
                                       const int *__restrict__ sel,
                                       int sel_stride, AttnRows ar, int cap,
                                       uint16_t *__restrict__ out) {
    auto item_ct1 = sycl::ext::oneapi::this_work_item::get_nd_item<3>();
    const int t = item_ct1.get_group(1),
              i = item_ct1.get_group(2) * kStage +
                  (int)(item_ct1.get_local_id(2) >> 4),
              part = item_ct1.get_local_id(2) & 15;
    int n_sel;
    if (ar.gen) {
        const long long pa = ar.p0 + ar.t0 + t;
        n_sel =
            ar.R > 0 ? (int)sycl::min((long long)ar.kcap, (pa + 1) / ar.R) : 0;
    } else {
        n_sel = ar.nsel[t];
    }
    if (i >= n_sel) return;
    const size_t r = (size_t) t * cap + i;
    unpack32(ckv + (size_t)sel[(size_t)t * sel_stride + i] *
                       kv_row_bytes(kKvFp4, 512),
             part, (sycl::uint4 *)(out + r * 512 + part * 32));
}

template <int G>
/*
DPCT1110:162: The total declared local variable size in device function
sparse_attn_packed_kernel exceeds 128 bytes and may cause high register
pressure. Consult with your hardware vendor to find the total register size
available and adjust the code, or use smaller sub-group size to avoid high
register pressure.
*/
__dpct_inline__ void sparse_attn_packed_kernel(
    const float *__restrict__ q, const uint16_t *__restrict__ win,
    const uint8_t *__restrict__ ckv, const int *__restrict__ sel,
    int sel_stride, AttnRows ar, const float *__restrict__ sink, int heads,
    float scale, float *__restrict__ out) {
    auto item_ct1 = sycl::ext::oneapi::this_work_item::get_nd_item<3>();
    constexpr int d = 512, W = 8 / G; // (kThreads / 32 warps)
    const int hb = item_ct1.get_group(2) * G, t = item_ct1.get_group(1);
    int n_win, n_sel, w0;
    if (ar.gen) {
        const long long ta = ar.t0 + t, pa = ar.p0 + ta;
        const long long first = sycl::max((long long)ar.win - ar.p0, ta + 1);
        w0 = (int) first;
        n_win = (int) (ar.win + ta - first + 1);
        n_sel =
            ar.R > 0 ? (int)sycl::min((long long)ar.kcap, (pa + 1) / ar.R) : 0;
    } else {
        n_win = ar.nwin[t], n_sel = ar.nsel[t], w0 = ar.win0[t];
    }
    const uint16_t* wbase = win + (size_t) w0 * d;
    const int* sl = sel + (size_t) t * sel_stride;
    const size_t crow = kv_row_bytes(kKvFp4, d);
    auto &tile = *sycl::ext::oneapi::group_local_memory_for_overwrite<
        uint16_t[kStage * d]>(
        sycl::ext::oneapi::this_work_item::get_work_group<3>());
    auto &sc = *sycl::ext::oneapi::group_local_memory_for_overwrite<
        float[G][kAttTile]>(
        sycl::ext::oneapi::this_work_item::get_work_group<3>());
    auto &rows = *sycl::ext::oneapi::group_local_memory_for_overwrite<
        const uint8_t *[kAttTile]>(
        sycl::ext::oneapi::this_work_item::get_work_group<3>());
    auto &m_run_s =
        *sycl::ext::oneapi::group_local_memory_for_overwrite<float[G]>(
            sycl::ext::oneapi::this_work_item::get_work_group<3>());
    auto &tile_scale_s =
        *sycl::ext::oneapi::group_local_memory_for_overwrite<float[G]>(
            sycl::ext::oneapi::this_work_item::get_work_group<3>());
    auto &den_s =
        *sycl::ext::oneapi::group_local_memory_for_overwrite<float[G]>(
            sycl::ext::oneapi::this_work_item::get_work_group<3>());
    const int lane = item_ct1.get_local_id(2) & 31,
              wp = item_ct1.get_local_id(2) >> 5, gw = wp / W, wr = wp % W;
    float qr[16];   // this lane's 16 values of its warp's head
#pragma unroll
    for (int x = 0; x < 16; ++x) qr[x] = q[((size_t) t * heads + hb + gw) * d + lane * 16 + x];
    if (item_ct1.get_local_id(2) < G) {
        m_run_s[item_ct1.get_local_id(2)] = -1e30f;
        den_s[item_ct1.get_local_id(2)] = 0.0f;
    }
    float o0[G], o1[G];
#pragma unroll
    for (int g = 0; g < G; ++g) o0[g] = o1[g] = 0.0f;
    const int c = item_ct1.get_local_id(2) * 2;
    const int sr = item_ct1.get_local_id(2) >> 4,
              part = item_ct1.get_local_id(2) &
                     15; // the staging: a thread's row and 32 values
    // rows [s0, s0 + n) of the tile at t0 into tile[] as BF16
    auto stage = [&](int t0, int s0, int n) {
        if (sr >= n) return;
        const int j = s0 + sr;
        sycl::uint4 *dst = (sycl::uint4 *)(tile + sr * d + part * 32);
        if (t0 + j < n_win) {   // a window row: BF16 already
            const sycl::uint4 *src = (const sycl::uint4 *)(rows[j] + part * 64);
            dst[0] = src[0];
            dst[1] = src[1];
            dst[2] = src[2];
            dst[3] = src[3];
            return;
        }
        unpack32(rows[j], part, dst);
    };
    const int total = n_win + n_sel;
    for (int t0 = 0; t0 < total; t0 += kAttTile) {
        const int nt = sycl::min(kAttTile, total - t0);
        if (item_ct1.get_local_id(2) < nt) {
            const int j = t0 + item_ct1.get_local_id(2);
            rows[item_ct1.get_local_id(2)] =
                j < n_win ? (const uint8_t *)(wbase + (size_t)j * d)
                          : ckv + (size_t)sl[j - n_win] * crow;
        }
        /*
        DPCT1118:163: SYCL group functions and algorithms must be encountered in
        converged control flow. You may need to adjust the code.
        */
        /*
        DPCT1065:436: Consider replacing sycl::nd_item::barrier() with
        sycl::nd_item::barrier(sycl::access::fence_space::local_space) for
        better performance if there is no access to global memory.
        */
        item_ct1.barrier();
        for (int s0 = 0; s0 < nt; s0 += kStage) {   // the dots
            const int n = sycl::min(kStage, nt - s0);
            stage(t0, s0, n);
            /*
            DPCT1118:165: SYCL group functions and algorithms must be
            encountered in converged control flow. You may need to adjust the
            code.
            */
            /*
            DPCT1065:438: Consider replacing sycl::nd_item::barrier() with
            sycl::nd_item::barrier(sycl::access::fence_space::local_space) for
            better performance if there is no access to global memory.
            */
            item_ct1.barrier();
#if !defined(STRATA_USE_HIP)   // (HIP leaves a loop with shuffles and a remainder rolled anyway)
#pragma unroll 2
#endif
            for (int r = wr; r < n; r += W) {   // (two rows' chains at once)
                const uint16_t* vr = tile + r * d + lane * 16;
                const sycl::uint4 a = *(const sycl::uint4 *)vr;
                const sycl::uint4 b = *(const sycl::uint4 *)(vr + 8);
                const uint16_t* pa = (const uint16_t*) &a;
                const uint16_t* pb = (const uint16_t*) &b;
                float s = 0.0f;
#pragma unroll
                for (int x = 0; x < 8; ++x) s += qr[x] * bf16f(pa[x]);
#pragma unroll
                for (int x = 0; x < 8; ++x) s += qr[8 + x] * bf16f(pb[x]);
                s = warp_sum(s);
                if (lane == 0) sc[gw][s0 + r] = s * scale;
            }
            /*
            DPCT1118:166: SYCL group functions and algorithms must be
            encountered in converged control flow. You may need to adjust the
            code.
            */
            /*
            DPCT1065:439: Consider replacing sycl::nd_item::barrier() with
            sycl::nd_item::barrier(sycl::access::fence_space::local_space) for
            better performance if there is no access to global memory.
            */
            item_ct1.barrier();
        }
        if (wp < G) {   // head wp's softmax over the tile
            float* scg = sc[wp];
            float m = -INFINITY;
#pragma unroll
            for (int j = lane; j < nt; j += 32) m = sycl::fmax(m, scg[j]);
            m = warp_max(m);
            const float m_old = m_run_s[wp];
            const float m_new = sycl::fmax(m_old, m);
            float s = 0.0f;
            for (int j = lane; j < nt; j += 32) {
                const float p = sycl::native::exp(scg[j] - m_new);
                scg[j] = p;
                s += p;
            }
            s = warp_sum(s);
            sycl::group_barrier(
                sycl::ext::oneapi::this_work_item::get_sub_group());
            if (lane == 0) {
                tile_scale_s[wp] = sycl::native::exp(m_old - m_new);
                den_s[wp] = den_s[wp] * tile_scale_s[wp] + s;
                m_run_s[wp] = m_new;
            }
        }
        /*
        DPCT1118:164: SYCL group functions and algorithms must be encountered in
        converged control flow. You may need to adjust the code.
        */
        /*
        DPCT1065:437: Consider replacing sycl::nd_item::barrier() with
        sycl::nd_item::barrier(sycl::access::fence_space::local_space) for
        better performance if there is no access to global memory.
        */
        item_ct1.barrier();
#pragma unroll
        for (int g = 0; g < G; ++g) {
            const float cs = tile_scale_s[g];
            o0[g] *= cs;
            o1[g] *= cs;
        }
        for (int s0 = 0; s0 < nt; s0 += kStage) {   // the weighted sums
            const int n = sycl::min(kStage, nt - s0);
            stage(t0, s0, n);
            /*
            DPCT1118:167: SYCL group functions and algorithms must be
            encountered in converged control flow. You may need to adjust the
            code.
            */
            /*
            DPCT1065:440: Consider replacing sycl::nd_item::barrier() with
            sycl::nd_item::barrier(sycl::access::fence_space::local_space) for
            better performance if there is no access to global memory.
            */
            item_ct1.barrier();
            for (int r = 0; r < n; ++r) {
                const uint32_t pr = *(const uint32_t*) (tile + r * d + c);
                const float v0 = bf16f((uint16_t) (pr & 0xffffu)), v1 = bf16f((uint16_t) (pr >> 16));
#pragma unroll
                for (int g = 0; g < G; ++g) {
                    o0[g] += sc[g][s0 + r] * v0;
                    o1[g] += sc[g][s0 + r] * v1;
                }
            }
            /*
            DPCT1118:168: SYCL group functions and algorithms must be
            encountered in converged control flow. You may need to adjust the
            code.
            */
            /*
            DPCT1065:441: Consider replacing sycl::nd_item::barrier() with
            sycl::nd_item::barrier(sycl::access::fence_space::local_space) for
            better performance if there is no access to global memory.
            */
            item_ct1.barrier();
        }
    }
#pragma unroll
    for (int g = 0; g < G; ++g) {
        const float den =
            den_s[g] + sycl::native::exp(sink[hb + g] - m_run_s[g]);
        float* ov = out + ((size_t) t * heads + hb + g) * d;
        ov[c] = o0[g] / den;
        ov[c + 1] = o1[g] / den;
    }
}

// DS4_FAST_ATTN=1: the chunked prompt's packed-KV attention on the matrix cores (gfx11/gfx12 WMMA; the R9700's gfx12).
// The matrix's rows are 16 heads of ONE query, so a block stages a 16-key tile once for all of them: Q (16 heads x 512
// dims) is the A operand of the scores' product, the key tile is B, and the same tile is B of the values' product (the
// heads' softmaxes and their 512-dim outputs are its rows).  The QK and PV sums are the matrix cores' order, not
// sparse_attn_packed_kernel's scalar lanes' - hence opt-in (DS4_FAST_ATTN).
// (HIP only: the CUDA build keeps the scalar kernels, so this and the dispatch below compile out there.)
#if defined(STRATA_USE_HIP)
constexpr int kFaHeads = 16;   // heads a tile (the matrix's rows)
constexpr int kFaKeys = 16;    // keys a tile
constexpr int kFaD = 512;      // the head dim
constexpr int kFaLd = 520;     // halves a tile row (512 + 8: the pad keeps the 16 rows off one bank)
__global__ void __launch_bounds__(kThreads) sparse_attn_wmma_kernel(const float* __restrict__ q,
                                                                    const uint16_t* __restrict__ win,
                                                                    const uint8_t* __restrict__ ckv,
                                                                    const int* __restrict__ sel, int sel_stride,
                                                                    AttnRows ar, const float* __restrict__ sink,
                                                                    int heads, float scale, float* __restrict__ out) {
    __shared__ __align__(16) uint16_t qs[kFaHeads * kFaLd];
    __shared__ __align__(16) uint16_t ks[kFaKeys * kFaLd];
    __shared__ float sc[kFaHeads][kFaKeys];
    __shared__ float part[8][kFaHeads][kFaKeys];   // (the 8 warps' 64-dim slices of the scores)
    __shared__ uint16_t ps[kFaHeads][kFaKeys];
    __shared__ const uint8_t* rows[kFaKeys];
    __shared__ float m_run[kFaHeads], den[kFaHeads], ts[kFaHeads];
    const int tid = (int) threadIdx.x, wp = tid >> 5, lane = tid & 31, l16 = lane & 15, hi = lane >> 4;
    const int t = blockIdx.y, hb = blockIdx.x * kFaHeads;
    int n_win, n_sel, w0;
    if (ar.gen) {
        const long long ta = ar.t0 + t, pa = ar.p0 + ta;
        const long long first = max((long long) ar.win - ar.p0, ta + 1);
        w0 = (int) first;
        n_win = (int) (ar.win + ta - first + 1);
        n_sel = ar.R > 0 ? (int) min((long long) ar.kcap, (pa + 1) / ar.R) : 0;
    } else {
        n_win = ar.nwin[t], n_sel = ar.nsel[t], w0 = ar.win0[t];
    }
    const uint16_t* wbase = win + (size_t) w0 * kFaD;
    const int* sl = sel + (size_t) t * sel_stride;
    const size_t crow = kv_row_bytes(kKvFp4, kFaD);
    // the block's 16 heads' queries as FP16 (the matrix's A operand), and their running softmax state
    for (int i = tid; i < kFaHeads * kFaD; i += kThreads) {
        const int h = i / kFaD, c = i % kFaD;
        qs[h * kFaLd + c] = __builtin_bit_cast(uint16_t, (_Float16) q[((size_t) t * heads + hb + h) * kFaD + c]);
    }
    if (tid < kFaHeads) {
        m_run[tid] = -1e30f;
        den[tid] = 0.0f;
    }
    f8v o[4];
#pragma unroll
    for (int f = 0; f < 4; ++f)
#pragma unroll
        for (int i = 0; i < 8; ++i) o[f][i] = 0.0f;
    __syncthreads();
    auto stage = [&](int t0, int n) {   // the tile's n rows into ks (BF16 rows -> FP16)
        const int sr = tid >> 4, part = tid & 15;
        if (sr >= n) return;
        const uint8_t* row = rows[sr];
        uint16_t tmp[32];
        if (t0 + sr < n_win) {
            const uint4* src = (const uint4*) (row + part * 64);
            const uint4 a = src[0], b = src[1], c = src[2], d = src[3];
            ((uint4*) tmp)[0] = a;
            ((uint4*) tmp)[1] = b;
            ((uint4*) tmp)[2] = c;
            ((uint4*) tmp)[3] = d;
        } else {
            unpack32(row, part, (uint4*) tmp);
        }
        uint16_t* dst = ks + sr * kFaLd + part * 32;
#pragma unroll
        for (int i = 0; i < 32; ++i) dst[i] = __builtin_bit_cast(uint16_t, (_Float16) bf16f(tmp[i]));
    };
    const int total = n_win + n_sel;
    for (int t0 = 0; t0 < total; t0 += kFaKeys) {
        const int nt = min(kFaKeys, total - t0);
        if (tid < nt) {
            const int j = t0 + tid;
            rows[tid] = j < n_win ? (const uint8_t*) (wbase + (size_t) j * kFaD) : ckv + (size_t) sl[j - n_win] * crow;
        }
        __syncthreads();
        stage(t0, nt);
        __syncthreads();
        {   // the scores in 8 slices of 64 dims, a warp a slice, summed in shared
            f8v acc = {0, 0, 0, 0, 0, 0, 0, 0};
            const int k0 = wp * 64;
            for (int kk = k0; kk < k0 + 64; kk += 16) {
                h16v a, b;
                const uint32_t* qa = (const uint32_t*) (qs + l16 * kFaLd + kk);
                const uint32_t* kb = (const uint32_t*) (ks + l16 * kFaLd + kk);
#pragma unroll
                for (int i = 0; i < 8; ++i) {   // (two halves a load)
                    a[2 * i] = __builtin_bit_cast(_Float16, (uint16_t) (qa[i] & 0xffffu));
                    a[2 * i + 1] = __builtin_bit_cast(_Float16, (uint16_t) (qa[i] >> 16));
                    b[2 * i] = __builtin_bit_cast(_Float16, (uint16_t) (kb[i] & 0xffffu));
                    b[2 * i + 1] = __builtin_bit_cast(_Float16, (uint16_t) (kb[i] >> 16));
                }
                acc = set_wmma(a, b, acc);
            }
#pragma unroll
            for (int i = 0; i < 8; ++i) part[wp][set_wmma_row(i, hi)][l16] = acc[i];
            __syncthreads();
            const int h = tid >> 4, k = tid & 15;
            float a = 0.0f;
#pragma unroll
            for (int w = 0; w < 8; ++w) a += part[w][h][k];
            sc[h][k] = t0 + k < total ? a * scale : -INFINITY;
        }
        __syncthreads();
        {   // the online softmax: a warp per 2 heads, a lane a key (16 lanes a head)
            const int h = 2 * wp + (lane >> 4), k = lane & 15;
            float m = sc[h][k];
#pragma unroll
            for (int o = 8; o > 0; o >>= 1) m = fmaxf(m, __shfl_xor_sync(0xffffffffu, m, o));
            const float mo = m_run[h], mn = fmaxf(mo, m);
            const float p = expf(sc[h][k] - mn);
            ps[h][k] = __builtin_bit_cast(uint16_t, (_Float16) p);
            float s = p;
#pragma unroll
            for (int o = 8; o > 0; o >>= 1) s += __shfl_xor_sync(0xffffffffu, s, o);
            if (k == 0) {
                ts[h] = expf(mo - mn);
                den[h] = den[h] * ts[h] + s;
                m_run[h] = mn;
            }
        }
        __syncthreads();
        const int dim0 = wp * 64;
#pragma unroll
        for (int f = 0; f < 4; ++f)
#pragma unroll
            for (int i = 0; i < 8; ++i) o[f][i] *= ts[set_wmma_row(i, hi)];
#pragma unroll
        for (int f = 0; f < 4; ++f) {   // the values: P (16 heads x 16 keys) x V (16 keys x 64 dims this warp)
            h16v a, b;
            const uint32_t* pp = (const uint32_t*) ps[l16];
#pragma unroll
            for (int i = 0; i < 8; ++i) {
                a[2 * i] = __builtin_bit_cast(_Float16, (uint16_t) (pp[i] & 0xffffu));
                a[2 * i + 1] = __builtin_bit_cast(_Float16, (uint16_t) (pp[i] >> 16));
                b[2 * i] = __builtin_bit_cast(_Float16, ks[(2 * i) * kFaLd + dim0 + f * 16 + l16]);
                b[2 * i + 1] = __builtin_bit_cast(_Float16, ks[(2 * i + 1) * kFaLd + dim0 + f * 16 + l16]);
            }
            o[f] = set_wmma(a, b, o[f]);
        }
        __syncthreads();
    }
#pragma unroll
    for (int f = 0; f < 4; ++f)
#pragma unroll
        for (int i = 0; i < 8; ++i) {
            const int h = set_wmma_row(i, hi), dm = wp * 64 + f * 16 + l16;
            const float dv = den[h] + expf(sink[hb + h] - m_run[h]);
            out[((size_t) t * heads + hb + h) * kFaD + dm] = o[f][i] / dv;
        }
}
#endif   // STRATA_USE_HIP

// the ring's rows are kKvFp8: E4M3 codes, a UE8M0 scale per 32 (decoded: exactly the BF16 values they were made from)
__dpct_inline__ void ring_to_ext_kernel(const uint8_t *__restrict__ ring,
                                        uint16_t *__restrict__ ext,
                                        long long p0, int n, int d) {
    auto item_ct1 = sycl::ext::oneapi::this_work_item::get_nd_item<3>();
    const int r = item_ct1.get_group(2);
    const long long p = p0 - n + r;
    const uint8_t* row = ring + (size_t) (p >= 0 ? p % n : 0) * kv_row_bytes(kKvFp8, d);
#pragma unroll
    for (int e = item_ct1.get_local_id(2); e < d;
         e += item_ct1.get_local_range(2))
        ext[(size_t) r * d + e] = p >= 0 ? f2bf16(e4m3_value(row[e]) * ue8m0_val(row[d + e / 32])) : (uint16_t) 0;
}

// a window row's BF16 values (FP8 roundings: each an E4M3 value times its group's power of two) as E4M3 codes at the
// smallest power-of-two scale 2^k their group fits under (max |v| <= 448 * 2^k = 1.75 * 2^(8 + k)): never above the
// rounding's own scale, so each v / 2^k is that E4M3 value times a power of two >= 1, at most 448 - an E4M3 value again
__dpct_inline__ void ext_to_ring_kernel(const uint16_t *__restrict__ ext,
                                        uint8_t *__restrict__ ring,
                                        long long p0, int n, int T, int d,
                                        uint8_t *__restrict__ backup) {
    auto item_ct1 = sycl::ext::oneapi::this_work_item::get_nd_item<3>();
    const int t = item_ct1.get_group(2);
    if (T - t > n) return;   // overwritten by a later row of the window anyway
    const size_t rb = kv_row_bytes(kKvFp8, d);
    uint8_t* row = ring + (size_t) ((p0 + t) % n) * rb;
    if (backup != nullptr) {
#pragma unroll
        for (size_t i = item_ct1.get_local_id(2); i < rb;
             i += item_ct1.get_local_range(2))
            backup[(size_t)t * rb + i] = row[i];
        /*
        DPCT1118:169: SYCL group functions and algorithms must be encountered in
        converged control flow. You may need to adjust the code.
        */
        /*
        DPCT1065:442: Consider replacing sycl::nd_item::barrier() with
        sycl::nd_item::barrier(sycl::access::fence_space::local_space) for
        better performance if there is no access to global memory.
        */
        item_ct1.barrier();
    }
    const uint16_t* src = ext + (size_t) (n + t) * d;
    for (int e0 = 0; e0 < d;
         e0 += item_ct1.get_local_range(
             2)) { // (d a multiple of 32: a warp's 32 values one group)
        const int e = e0 + (int)item_ct1.get_local_id(2);
        const float v = e < d ? bf16f(src[e]) : 0.0f;
        uint32_t a = sycl::bit_cast<unsigned int>(v) &
                     0x7fffffffu; // (the magnitudes' order as bits)
        /*
DPCT1108:170: '__shfl_xor_sync' was migrated with the experimental feature
masked sub_group function which may not be supported by all compilers or
runtimes. You may need to adjust the code.
*/
#pragma unroll
        for (int o = 16; o > 0; o >>= 1) a = sycl::max(
            a, (unsigned int)((uint32_t)
                                  dpct::experimental::permute_sub_group_by_xor(
                                      0xffffffffu,
                                      sycl::ext::oneapi::this_work_item::
                                          get_sub_group(),
                                      a, o)));
        const int k = a == 0u ? 0 : (int) (a >> 23) - 135 + ((a & 0x7fffffu) > 0x600000u ? 1 : 0);
        if (e >= d) continue;
        row[e] = (uint8_t) e4m3_code(v, k);
        if ((e & 31) == 0) row[d + e / 32] = (uint8_t) (k + 127);
    }
}

__dpct_inline__ void ring_restore_kernel(uint8_t *__restrict__ ring,
                                         const uint8_t *__restrict__ backup,
                                         long long p0, int n, int keep,
                                         size_t rb) {
    auto item_ct1 = sycl::ext::oneapi::this_work_item::get_nd_item<3>();
    const int t = keep + item_ct1.get_group(2);
    uint8_t* row = ring + (size_t) ((p0 + t) % n) * rb;
#pragma unroll
    for (size_t i = item_ct1.get_local_id(2); i < rb;
         i += item_ct1.get_local_range(2)) row[i] = backup[(size_t)t * rb + i];
}
__dpct_inline__ void ring_restore_batch_kernel(const RingBatch b, long long p0,
                                               int n, int keep, size_t rb) {
    auto item_ct1 = sycl::ext::oneapi::this_work_item::get_nd_item<3>();
    const int t = keep + item_ct1.get_group(2), li = item_ct1.get_group(1);
    uint8_t* row = b.ring[li] + (size_t) ((p0 + t) % n) * rb;
    const uint8_t* src = b.bak[li] + (size_t) t * rb;
#pragma unroll
    for (size_t i = item_ct1.get_local_id(2); i < rb;
         i += item_ct1.get_local_range(2)) row[i] = src[i];
}

__dpct_inline__ void hc_mean_kernel(const float *__restrict__ h, int hc,
                                    int dim, float *__restrict__ out,
                                    long long out_stride) {
    auto item_ct1 = sycl::ext::oneapi::this_work_item::get_nd_item<3>();
    const int e = item_ct1.get_group(2) * item_ct1.get_local_range(2) +
                  item_ct1.get_local_id(2),
              t = item_ct1.get_group(1);
    if (e >= dim) return;
    float a = 0.0f;
#pragma unroll
    for (int s = 0; s < hc; ++s) a += h[((size_t)t * hc + s) * dim + e];
    out[(long long) t * out_stride + e] = a / (float) hc;
}

__dpct_inline__ void iota_rows_kernel(int *out, int n, int stride) {
    auto item_ct1 = sycl::ext::oneapi::this_work_item::get_nd_item<3>();
    const int t = item_ct1.get_group(2);
#pragma unroll
    for (int i = item_ct1.get_local_id(2); i < n;
         i += item_ct1.get_local_range(2)) out[(size_t)t * stride + i] = i;
}

// a warp a vocabulary row: E[*tok] (shared memory) . H[v]
__dpct_inline__ void markov_kernel(float *__restrict__ logits,
                                   const uint16_t *__restrict__ E,
                                   const uint16_t *__restrict__ H,
                                   const int *__restrict__ tok, int V, int R,
                                   uint8_t *dpct_local) {
    auto item_ct1 = sycl::ext::oneapi::this_work_item::get_nd_item<3>();
    auto er = (float *)dpct_local;
    const uint16_t* erow = E + (size_t) (*tok) * R;
#pragma unroll
    for (int i = item_ct1.get_local_id(2); i < R;
         i += item_ct1.get_local_range(2)) er[i] = bf16f(erow[i]);
    /*
    DPCT1065:443: Consider replacing sycl::nd_item::barrier() with
    sycl::nd_item::barrier(sycl::access::fence_space::local_space) for better
    performance if there is no access to global memory.
    */
    item_ct1.barrier();
    const int lane = item_ct1.get_local_id(2) & 31;
    const int v = item_ct1.get_group(2) * (kThreads / 32) +
                  (item_ct1.get_local_id(2) >> 5);
    if (v >= V) return;
    const uint16_t* hr = H + (size_t) v * R;
    float a = 0.0f;
    for (int c = lane * 8; c < R; c += 32 * 8) {
        const sycl::uint4 q = *(const sycl::uint4 *)(hr + c);
        const sycl::half2 *hp = (const sycl::half2 *)&q;
#pragma unroll
        for (int i = 0; i < 4; ++i) {
            const sycl::float2 f =
                hp[i].template convert<float, sycl::rounding_mode::automatic>();
            a += f.x() * er[c + 2 * i] + f.y() * er[c + 2 * i + 1];
        }
    }
    a = warp_sum(a);
    if (lane == 0) logits[v] += a;
}

// the same bias with the head quantised to Q8_0 blocks of 32 (34 B: an F16 d and 32 int8): half the bytes read for the same sums.
// A lane's 8 values sit inside one block (8 divides 32), so a lane reads one d and 8 codes; the sums are the same terms, in the
// block's own order, with the quantisation's error the only difference.
__dpct_inline__ void markov_kernel_q8(float *__restrict__ logits,
                                      const uint16_t *__restrict__ E,
                                      const uint8_t *__restrict__ Hq,
                                      const int *__restrict__ tok, int V, int R,
                                      uint8_t *dpct_local) {
    auto item_ct1 = sycl::ext::oneapi::this_work_item::get_nd_item<3>();
    auto er = (float *)dpct_local;
    const uint16_t* erow = E + (size_t) (*tok) * R;
#pragma unroll
    for (int i = item_ct1.get_local_id(2); i < R;
         i += item_ct1.get_local_range(2)) er[i] = bf16f(erow[i]);
    /*
    DPCT1065:444: Consider replacing sycl::nd_item::barrier() with
    sycl::nd_item::barrier(sycl::access::fence_space::local_space) for better
    performance if there is no access to global memory.
    */
    item_ct1.barrier();
    const int lane = item_ct1.get_local_id(2) & 31;
    const int v = item_ct1.get_group(2) * (kThreads / 32) +
                  (item_ct1.get_local_id(2) >> 5);
    if (v >= V) return;
    const uint8_t* hr = Hq + (size_t) v * (size_t) (R / 32) * 34;
    float a = 0.0f;
    for (int c = lane * 8; c < R; c += 32 * 8) {
        const uint8_t* blk = hr + (size_t) (c >> 5) * 34;
        const float d =
            sycl::vec<sycl::half, 1>(*(const sycl::half *)blk)
                .convert<float, sycl::rounding_mode::automatic>()[0];
        const int8_t* q = (const int8_t*) (blk + 2 + (c & 31));
        float t = 0.0f;
#pragma unroll
        for (int i = 0; i < 8; ++i) t += (float) q[i] * er[c + i];
        a += d * t;
    }
    a = warp_sum(a);
    if (lane == 0) logits[v] += a;
}

// a member's rows [v0, v0 + vn) of the bias (markov_kernel's sums, the prev token's E row already on the device, BF16)
__dpct_inline__ void markov_share_kernel(float *__restrict__ logits,
                                         const uint16_t *__restrict__ H,
                                         const uint16_t *__restrict__ erow,
                                         int vn, int R, uint8_t *dpct_local) {
    auto item_ct1 = sycl::ext::oneapi::this_work_item::get_nd_item<3>();
    auto er = (float *)dpct_local;
#pragma unroll
    for (int i = item_ct1.get_local_id(2); i < R;
         i += item_ct1.get_local_range(2)) er[i] = bf16f(erow[i]);
    /*
    DPCT1065:445: Consider replacing sycl::nd_item::barrier() with
    sycl::nd_item::barrier(sycl::access::fence_space::local_space) for better
    performance if there is no access to global memory.
    */
    item_ct1.barrier();
    const int lane = item_ct1.get_local_id(2) & 31;
    const int v = item_ct1.get_group(2) * (kThreads / 32) +
                  (item_ct1.get_local_id(2) >> 5);
    if (v >= vn) return;
    const uint16_t* hr = H + (size_t) v * R;
    float a = 0.0f;
    for (int c = lane * 8; c < R; c += 32 * 8) {
        const sycl::uint4 q = *(const sycl::uint4 *)(hr + c);
        const sycl::half2 *hp = (const sycl::half2 *)&q;
#pragma unroll
        for (int i = 0; i < 4; ++i) {
            const sycl::float2 f =
                hp[i].template convert<float, sycl::rounding_mode::automatic>();
            a += f.x() * er[c + 2 * i] + f.y() * er[c + 2 * i + 1];
        }
    }
    a = warp_sum(a);
    if (lane == 0) logits[v] += a;
}
// the share's argmax (the greedy sampler's rule: the larger value, on a tie the lower index) -> slot {value, v0 + i,
// seq}, the seq last
__dpct_inline__ void argmax_share_kernel(const float *__restrict__ row, int vn,
                                         int v0, unsigned *slot, unsigned seq) {
    auto item_ct1 = sycl::ext::oneapi::this_work_item::get_nd_item<3>();
auto &sv = *sycl::ext::oneapi::group_local_memory_for_overwrite<float[32]>(
    sycl::ext::oneapi::this_work_item::get_work_group<3>());
    auto &si = *sycl::ext::oneapi::group_local_memory_for_overwrite<int[32]>(
        sycl::ext::oneapi::this_work_item::get_work_group<3>());
    float bv = sycl::bit_cast<float, int>(0xff800000);
    int bi = 0x7fffffff;
    for (int i = item_ct1.get_local_id(2); i < vn;
         i += item_ct1.get_local_range(2)) {
        const float x = row[i];
        if (x > bv) bv = x, bi = i;   // (ascending: the lowest index of a tie stays)
    }
    for (int o = 16; o > 0; o >>= 1) {
        /*
        DPCT1108:171: '__shfl_xor_sync' was migrated with the experimental
        feature masked sub_group function which may not be supported by all
        compilers or runtimes. You may need to adjust the code.
        */
        const float ov = dpct::experimental::permute_sub_group_by_xor(
            0xffffffffu, sycl::ext::oneapi::this_work_item::get_sub_group(), bv,
            o);
        /*
        DPCT1108:172: '__shfl_xor_sync' was migrated with the experimental
        feature masked sub_group function which may not be supported by all
        compilers or runtimes. You may need to adjust the code.
        */
        const int oi = dpct::experimental::permute_sub_group_by_xor(
            0xffffffffu, sycl::ext::oneapi::this_work_item::get_sub_group(), bi,
            o);
        if (ov > bv || (ov == bv && oi < bi)) bv = ov, bi = oi;
    }
    if ((item_ct1.get_local_id(2) & 31) == 0)
        sv[item_ct1.get_local_id(2) >> 5] = bv,
                                       si[item_ct1.get_local_id(2) >> 5] = bi;
    item_ct1.barrier(sycl::access::fence_space::local_space);
    if (item_ct1.get_local_id(2) == 0) {
#pragma unroll
        for (int w = 1; w < (int)(item_ct1.get_local_range(2) >> 5); ++w)
            if (sv[w] > bv || (sv[w] == bv && si[w] < bi)) bv = sv[w], bi = si[w];
        slot[0] = sycl::bit_cast<unsigned int>(bv);
        slot[1] = (unsigned) (v0 + (bi == 0x7fffffff ? 0 : bi));
        /*
        DPCT1078:173: Consider replacing memory_order::acq_rel with
        memory_order::seq_cst for correctness if strong memory order
        restrictions are needed.
        */
        sycl::atomic_fence(sycl::memory_order::acq_rel,
                           sycl::memory_scope::system);
        *(volatile unsigned*) (slot + 2) = seq;
    }
}
// every member's share argmax of this step (slots nm apart) -> the global one into *tok, and its E row (host-mapped
// BF16 [V][R]) into erow for the next step's bias
__dpct_inline__ void markov_pick_kernel(const unsigned *slots, int nm,
                                        unsigned seq, int *tok,
                                        const uint16_t *E, uint16_t *erow,
                                        int R) {
    auto item_ct1 = sycl::ext::oneapi::this_work_item::get_nd_item<3>();
auto &pick = *sycl::ext::oneapi::group_local_memory_for_overwrite<int>(
    sycl::ext::oneapi::this_work_item::get_work_group<3>());
    if ((int)item_ct1.get_local_id(2) < nm)
        while (
            (int)(*(volatile const unsigned
                        *)(slots + (size_t)item_ct1.get_local_id(2) * 4 + 2) -
                  seq) < 0) {
#if defined(DPCT_COMPATIBILITY_TEMP) && DPCT_COMPATIBILITY_TEMP >= 700
            /*
            DPCT1008:446: __nanosleep function is not defined in SYCL. This is a
            hardware-specific feature. Consult with your hardware vendor to find
            a replacement.
            */
            /* CUDA pause hint; polling order is unchanged. */
#endif
        }
    item_ct1.barrier(sycl::access::fence_space::local_space);
    if (item_ct1.get_local_id(2) == 0) {
        float bv = sycl::bit_cast<float, int>(0xff800000);
        int bi = 0x7fffffff;
        for (int m = 0; m < nm; ++m) {   // (members in vocabulary order: a tie keeps the lower index)
            const float v = sycl::bit_cast<float>(
                *(volatile const unsigned *)(slots + (size_t)m * 4));
            const int i = (int) *(volatile const unsigned*) (slots + (size_t) m * 4 + 1);
            if (v > bv || (v == bv && i < bi)) bv = v, bi = i;
        }
        pick = bi;
        *tok = bi;
    }
    item_ct1.barrier(sycl::access::fence_space::local_space);
#if defined(STRATA_USE_HIP)   // (E is fixed once loaded: a plain load is as fresh)
    for (int i = threadIdx.x; i < R; i += blockDim.x) erow[i] = E[(size_t) pick * R + i];
#else
    /*
DPCT1098:447: The '*' expression is used instead of the __ldcv call. These two
expressions do not provide the exact same functionality. Check the generated
code for potential precision and/or performance issues.
*/
#pragma unroll
    for (int i = item_ct1.get_local_id(2); i < R;
         i += item_ct1.get_local_range(2))
        erow[i] = *(E + (size_t)pick * R + i);
#endif
}

__dpct_inline__ void confidence_kernel(const float *__restrict__ x,
                                       const uint16_t *__restrict__ E,
                                       const int *__restrict__ tok,
                                       const float *__restrict__ w, int dim,
                                       int R, float *__restrict__ out) {
    auto item_ct1 = sycl::ext::oneapi::this_work_item::get_nd_item<3>();
    const int t = item_ct1.get_group(2);
    const float* xt = x + (size_t) t * dim;
    const uint16_t* er = E + (size_t) tok[t] * R;
    float a = 0.0f;
#pragma unroll
    for (int i = item_ct1.get_local_id(2); i < dim + R;
         i += item_ct1.get_local_range(2)) a +=
        w[i] * (i < dim ? xt[i] : bf16f(er[i - dim]));
    a = block_sum(a);
    if (item_ct1.get_local_id(2) == 0) out[t] =
        1.0f / (1.0f + sycl::native::exp(-a));
}

// ---------------------------------------------------------------- engram
// ggml's IQ4_NL: a block of 32 values is 18 bytes, an fp16 d then 16 packed nibble bytes; the low nibble of qs[j] is
// value j and its high nibble value j + 16, each an index into this codebook (ggml-common.h's kvalues_iq4nl,
// dequantize_row_iq4_nl).  A row is hd / 32 of these blocks, the d of every one inside it - no separate scales.
const int8_t kIq4nl[16] = {-127, -104, -83, -65, -49, -35, -22, -10,
                           1,    13,   25,  38,  53,  69,  89,  113};
__dpct_inline__ float iq4nl_value(const uint8_t *b, int e) {
    const uint8_t q = b[2 + (e & 15)];
    const int v = (e & 16) ? (q >> 4) : (q & 15);
    return sycl::vec<sycl::half, 1>(
               sycl::bit_cast<sycl::half, unsigned short>(*(const uint16_t *)b))
               .convert<float, sycl::rounding_mode::automatic>()[0] *
           (float)kIq4nl[v];
}

__dpct_inline__ void engram_dequant_kernel(const uint8_t *__restrict__ rows,
                                           const uint8_t *__restrict__ scales,
                                           int hd, float *__restrict__ out,
                                           int fmt) {
    auto item_ct1 = sycl::ext::oneapi::this_work_item::get_nd_item<3>();
    const int r = item_ct1.get_group(2);
    if (fmt == 1) {   // (ggml Q6_K superblocks at the row's start)
#pragma unroll
        for (int e = item_ct1.get_local_id(2); e < hd;
             e += item_ct1.get_local_range(2))
            out[(size_t) r * hd + e] = q6k_value(rows + (size_t) r * hd + (size_t) (e / 256) * kQ6kBytes, e % 256);
        return;
    }
    if (fmt == 2) {   // (ggml IQ4_NL blocks at the row's start, their scales inside)
#pragma unroll
        for (int e = item_ct1.get_local_id(2); e < hd;
             e += item_ct1.get_local_range(2))
            out[(size_t) r * hd + e] = iq4nl_value(rows + (size_t) r * hd + (size_t) (e / 32) * 18, e % 32);
        return;
    }
    for (int e = item_ct1.get_local_id(2); e < hd;
         e += item_ct1.get_local_range(2)) {
        const float hv = e4m3_value(rows[(size_t) r * hd + e]);
        const uint8_t sb = scales[(size_t) r * (hd / 32) + e / 32];
        const float s =
            sb == 0xffu ? sycl::bit_cast<float>(0x7fc00000)
                        : (sb == 0 ? 5.877471754111438e-39f
                                   : sycl::bit_cast<float>((uint32_t)sb << 23));
        out[(size_t) r * hd + e] = hv * s;
    }
}

__dpct_inline__ void engram_gate_kernel(float *__restrict__ h,
                                        const float *__restrict__ kv,
                                        const float *__restrict__ w, int hc,
                                        int dim, float eps) {
    auto item_ct1 = sycl::ext::oneapi::this_work_item::get_nd_item<3>();
    const int k = item_ct1.get_group(2), t = item_ct1.get_group(1);
    float* hk = h + ((size_t) t * hc + k) * dim;
    const float* kvt = kv + (size_t) t * (hc + 1) * dim;
    const float* key = kvt + (size_t) k * dim;
    const float* val = kvt + (size_t) hc * dim;
    const float* wk = w + (size_t) k * dim;
    float sh = 0.0f, sk = 0.0f, sd = 0.0f;
    for (int e = item_ct1.get_local_id(2); e < dim;
         e += item_ct1.get_local_range(2)) {
        const float hv = hk[e], kv_ = key[e];
        sh += hv * hv;
        sk += kv_ * kv_;
        sd += hv * wk[e] * kv_;
    }
    sh = block_sum(sh);
    sk = block_sum(sk);
    sd = block_sum(sd);
    const float rstd =
        sycl::rsqrt(sh / (float)dim + eps) * sycl::rsqrt(sk / (float)dim + eps);
    const float dot = sd * rstd * sycl::rsqrt((float)dim);
    const float sq = sycl::copysign(
        sycl::sqrt(sycl::fmax(sycl::fabs(dot), 1e-6f)), (float)dot);
    const float gate = 1.0f / (1.0f + sycl::native::exp(-sq));
#pragma unroll
    for (int e = item_ct1.get_local_id(2); e < dim;
         e += item_ct1.get_local_range(2)) hk[e] = hk[e] + gate * val[e];
}

__dpct_inline__ void gather_kernel(const float *__restrict__ x, Gather g,
                                   int cols, float *__restrict__ out) {
    auto item_ct1 = sycl::ext::oneapi::this_work_item::get_nd_item<3>();
    const int r = item_ct1.get_group(1);
    const int c = item_ct1.get_group(2) * item_ct1.get_local_range(2) +
                  item_ct1.get_local_id(2);
    if (c < cols) out[(size_t) r * cols + c] = x[(size_t) g.src[r] * cols + c];
}

__dpct_inline__ void combine_kernel(const float *__restrict__ ed,
                                    const float *__restrict__ cpu,
                                    const float *__restrict__ sh, Combine cm,
                                    int dim, float *__restrict__ out) {
    auto item_ct1 = sycl::ext::oneapi::this_work_item::get_nd_item<3>();
    const int e = item_ct1.get_group(2) * item_ct1.get_local_range(2) +
                  item_ct1.get_local_id(2),
              t = item_ct1.get_group(1);
    if (e >= dim) return;
    float a = 0.0f;
#pragma unroll
    for (int j = 0; j < cm.n[t]; ++j) a +=
        cm.w[t][j] * ed[(size_t)cm.row[t][j] * dim + e];
    if (cpu != nullptr && cm.cpu[t]) a += cpu[(size_t) t * dim + e];
    if (sh != nullptr) a += sh[(size_t) t * dim + e];
    out[(size_t) t * dim + e] = a;
}

__dpct_inline__ void route_publish_kernel(
    const float *__restrict__ logits, const float *__restrict__ bias, int n,
    int k, float route_scale, const unsigned long long *__restrict__ vram,
    const float *__restrict__ x, int dim, int *idx, float *w,
    unsigned long long *plan_ptr, Req *req, const float *__restrict__ bias_vl,
    unsigned vl, int &g_router_bad) {
    auto item_ct1 = sycl::ext::oneapi::this_work_item::get_nd_item<3>();
auto &sc = *sycl::ext::oneapi::group_local_memory_for_overwrite<float[1024]>(
    sycl::ext::oneapi::this_work_item::get_work_group<3>());
    auto &bs =
        *sycl::ext::oneapi::group_local_memory_for_overwrite<float[1024]>(
            sycl::ext::oneapi::this_work_item::get_work_group<3>());
    auto &bv = *sycl::ext::oneapi::group_local_memory_for_overwrite<float[32]>(
        sycl::ext::oneapi::this_work_item::get_work_group<3>());
    auto &bi = *sycl::ext::oneapi::group_local_memory_for_overwrite<int[32]>(
        sycl::ext::oneapi::this_work_item::get_work_group<3>());
    auto &pick = *sycl::ext::oneapi::group_local_memory_for_overwrite<int[16]>(
        sycl::ext::oneapi::this_work_item::get_work_group<3>());
    auto &on_cpu = *sycl::ext::oneapi::group_local_memory_for_overwrite<int>(
        sycl::ext::oneapi::this_work_item::get_work_group<3>());
    const int t = item_ct1.get_group(2);
    logits += (size_t) t * 1024;
    if (bias_vl != nullptr && ((vl >> t) & 1u)) bias = bias_vl;
    for (int i = item_ct1.get_local_id(2); i < n;
         i += item_ct1.get_local_range(2)) {
        const float l = logits[i];
        const float sp = l > 20.0f ? l : sycl::log1p(sycl::native::exp(l));
        sc[i] = sycl::sqrt((float)sp);
        bs[i] = sc[i] + bias[i];
    }
    /*
    DPCT1065:448: Consider replacing sycl::nd_item::barrier() with
    sycl::nd_item::barrier(sycl::access::fence_space::local_space) for better
    performance if there is no access to global memory.
    */
    item_ct1.barrier();
    const int lane = item_ct1.get_local_id(2) & 31,
              wp = item_ct1.get_local_id(2) >> 5;
    for (int r = 0; r < k; ++r) {
        float best = -FLT_MAX;
        int bix = 0x7fffffff;
#pragma unroll
        for (int i = item_ct1.get_local_id(2); i < n;
             i += item_ct1.get_local_range(2))
            if (bs[i] > best || (bs[i] == best && i < bix)) best = bs[i], bix = i;
        for (int o = 16; o > 0; o >>= 1) {
            /*
            DPCT1108:176: '__shfl_xor_sync' was migrated with the experimental
            feature masked sub_group function which may not be supported by all
            compilers or runtimes. You may need to adjust the code.
            */
            const float ov = dpct::experimental::permute_sub_group_by_xor(
                0xffffffffu, sycl::ext::oneapi::this_work_item::get_sub_group(),
                best, o);
            /*
            DPCT1108:177: '__shfl_xor_sync' was migrated with the experimental
            feature masked sub_group function which may not be supported by all
            compilers or runtimes. You may need to adjust the code.
            */
            const int oi = dpct::experimental::permute_sub_group_by_xor(
                0xffffffffu, sycl::ext::oneapi::this_work_item::get_sub_group(),
                bix, o);
            if (ov > best || (ov == best && oi < bix)) best = ov, bix = oi;
        }
        if (lane == 0) bv[wp] = best, bi[wp] = bix;
        /*
        DPCT1118:174: SYCL group functions and algorithms must be encountered in
        converged control flow. You may need to adjust the code.
        */
        /*
        DPCT1065:450: Consider replacing sycl::nd_item::barrier() with
        sycl::nd_item::barrier(sycl::access::fence_space::local_space) for
        better performance if there is no access to global memory.
        */
        item_ct1.barrier();
        if (item_ct1.get_local_id(2) == 0) {
            float b = -FLT_MAX;
            int ix = 0x7fffffff;
#pragma unroll
            for (int q = 0; q < (int)(item_ct1.get_local_range(2) >> 5); ++q)
                if (bv[q] > b || (bv[q] == b && bi[q] < ix)) b = bv[q], ix = bi[q];
            if (ix >= n) ix = route_fallback_pick(
                pick, r, n, g_router_bad); // (a non-finite gate input: see
                                           // route_fallback_pick)
            pick[r] = ix;
            bs[ix] = -FLT_MAX;
        }
        /*
        DPCT1118:175: SYCL group functions and algorithms must be encountered in
        converged control flow. You may need to adjust the code.
        */
        /*
        DPCT1065:451: Consider replacing sycl::nd_item::barrier() with
        sycl::nd_item::barrier(sycl::access::fence_space::local_space) for
        better performance if there is no access to global memory.
        */
        item_ct1.barrier();
    }
    if (item_ct1.get_local_id(2) == 0) {
        float s = 0.0f;
#pragma unroll
        for (int r = 0; r < k; ++r) s += sc[pick[r]];
        int cpu = 0;
        for (int r = 0; r < k; ++r) {
            const float wv = sc[pick[r]] / (s + 1e-20f) * route_scale;
            idx[(size_t) t * kPicks + r] = pick[r];
            w[(size_t) t * kPicks + r] = wv;
            plan_ptr[(size_t) t * kPicks + r] = vram[pick[r]];
            cpu |= vram[pick[r]] == 0ull;
            if (req != nullptr) {
                req->ids[t][r] = pick[r];
                req->w[t][r] = wv;
            }
        }
        on_cpu = cpu;
    }
    // the token's FFN input to the host only where the CPU lane reads it (a pick of its on the CPU lane): each row
    // 20 KB over PCIe, where most windows have none
    /*
    DPCT1065:449: Consider replacing sycl::nd_item::barrier() with
    sycl::nd_item::barrier(sycl::access::fence_space::local_space) for better
    performance if there is no access to global memory.
    */
    item_ct1.barrier();
    if (req != nullptr && on_cpu)
#pragma unroll
        for (int i = item_ct1.get_local_id(2); i < dim;
             i += item_ct1.get_local_range(2))
            req->x[t][i] = x[(size_t)t * dim + i];
}

// a later layer's m likely picks (route_kernel's choice from its logits) into req->ahead[t][off ..)
__dpct_inline__ void route_ahead_kernel(const float *__restrict__ logits,
                                        const float *__restrict__ bias, int n,
                                        int m, Req *req, int off) {
    auto item_ct1 = sycl::ext::oneapi::this_work_item::get_nd_item<3>();
auto &bs = *sycl::ext::oneapi::group_local_memory_for_overwrite<float[1024]>(
    sycl::ext::oneapi::this_work_item::get_work_group<3>());
    auto &bv = *sycl::ext::oneapi::group_local_memory_for_overwrite<float[32]>(
        sycl::ext::oneapi::this_work_item::get_work_group<3>());
    auto &bi = *sycl::ext::oneapi::group_local_memory_for_overwrite<int[32]>(
        sycl::ext::oneapi::this_work_item::get_work_group<3>());
    const int t = item_ct1.get_group(2);
    logits += (size_t) t * 1024;
    for (int i = item_ct1.get_local_id(2); i < n;
         i += item_ct1.get_local_range(2)) {
        const float l = logits[i];
        const float sp = l > 20.0f ? l : sycl::log1p(sycl::native::exp(l));
        bs[i] = sycl::sqrt((float)sp) + bias[i];
    }
    /*
    DPCT1065:452: Consider replacing sycl::nd_item::barrier() with
    sycl::nd_item::barrier(sycl::access::fence_space::local_space) for better
    performance if there is no access to global memory.
    */
    item_ct1.barrier();
    const int lane = item_ct1.get_local_id(2) & 31,
              wp = item_ct1.get_local_id(2) >> 5;
    for (int r = 0; r < m; ++r) {
        float best = -FLT_MAX;
        int bix = 0x7fffffff;
#pragma unroll
        for (int i = item_ct1.get_local_id(2); i < n;
             i += item_ct1.get_local_range(2))
            if (bs[i] > best || (bs[i] == best && i < bix)) best = bs[i], bix = i;
        for (int o = 16; o > 0; o >>= 1) {
            /*
            DPCT1108:180: '__shfl_xor_sync' was migrated with the experimental
            feature masked sub_group function which may not be supported by all
            compilers or runtimes. You may need to adjust the code.
            */
            const float ov = dpct::experimental::permute_sub_group_by_xor(
                0xffffffffu, sycl::ext::oneapi::this_work_item::get_sub_group(),
                best, o);
            /*
            DPCT1108:181: '__shfl_xor_sync' was migrated with the experimental
            feature masked sub_group function which may not be supported by all
            compilers or runtimes. You may need to adjust the code.
            */
            const int oi = dpct::experimental::permute_sub_group_by_xor(
                0xffffffffu, sycl::ext::oneapi::this_work_item::get_sub_group(),
                bix, o);
            if (ov > best || (ov == best && oi < bix)) best = ov, bix = oi;
        }
        if (lane == 0) bv[wp] = best, bi[wp] = bix;
        /*
        DPCT1118:178: SYCL group functions and algorithms must be encountered in
        converged control flow. You may need to adjust the code.
        */
        /*
        DPCT1065:453: Consider replacing sycl::nd_item::barrier() with
        sycl::nd_item::barrier(sycl::access::fence_space::local_space) for
        better performance if there is no access to global memory.
        */
        item_ct1.barrier();
        if (item_ct1.get_local_id(2) == 0) {
            float b = -FLT_MAX;
            int ix = 0x7fffffff;
#pragma unroll
            for (int q = 0; q < (int)(item_ct1.get_local_range(2) >> 5); ++q)
                if (bv[q] > b || (bv[q] == b && bi[q] < ix)) b = bv[q], ix = bi[q];
            req->ahead[t][off + r] = ix < n ? ix : -1;
            if (ix < n) bs[ix] = -FLT_MAX;
        }
        /*
        DPCT1118:179: SYCL group functions and algorithms must be encountered in
        converged control flow. You may need to adjust the code.
        */
        /*
        DPCT1065:454: Consider replacing sycl::nd_item::barrier() with
        sycl::nd_item::barrier(sycl::access::fence_space::local_space) for
        better performance if there is no access to global memory.
        */
        item_ct1.barrier();
    }
}

__dpct_inline__ void publish_kernel(Req *req, int T, unsigned seq, int na0,
                                    int na1) {
    /*
    DPCT1078:182: Consider replacing memory_order::acq_rel with
    memory_order::seq_cst for correctness if strong memory order restrictions
    are needed.
    */
    sycl::atomic_fence(sycl::memory_order::acq_rel, sycl::memory_scope::system);
    req->na[0] = na0;
    req->na[1] = na1;
    req->T = T;
    /*
    DPCT1078:183: Consider replacing memory_order::acq_rel with
    memory_order::seq_cst for correctness if strong memory order restrictions
    are needed.
    */
    sycl::atomic_fence(sycl::memory_order::acq_rel, sycl::memory_scope::system);
    *(volatile unsigned*) &req->seq = seq;
}

__dpct_inline__ void init_pre_kernel(float *pre, int T) {
    auto item_ct1 = sycl::ext::oneapi::this_work_item::get_nd_item<3>();
    const int i = item_ct1.get_group(2) * item_ct1.get_local_range(2) +
                  item_ct1.get_local_id(2);
    if (i < T * 8) pre[i] = (i % 8) == 0 ? 1.0f : 0.0f;
}


// the hand-offs through host-mapped memory: a grid-stride copy of the segments (16 bytes a thread), then (put, flag
// given) the last block to finish sets the flag after a system fence; a get first waits for the flag
__dpct_inline__ void xfer_copy(const Xfer &x, bool from_host) {
    auto item_ct1 = sycl::ext::oneapi::this_work_item::get_nd_item<3>();
    size_t total = 0;
#pragma unroll
    for (int i = 0; i < x.n; ++i) total += x.bytes[i] / 16;
    for (size_t q =
             (size_t)item_ct1.get_group(2) * item_ct1.get_local_range(2) +
             item_ct1.get_local_id(2);
         q < total; q += (size_t)item_ct1.get_group_range(2) *
                         item_ct1.get_local_range(2)) {
        size_t r = q;
        int i = 0;
        while (r >= x.bytes[i] / 16) r -= x.bytes[i++] / 16;
        const sycl::uint4 *src = (const sycl::uint4 *)x.src[i] + r;
        ((sycl::uint4 *)x.dst[i])[r] = from_host ? ld_fresh(src) : *src;
    }
}

__dpct_inline__ void xfer_put_kernel(const Xfer x, unsigned *flag, unsigned seq,
                                     unsigned *cnt) {
    auto item_ct1 = sycl::ext::oneapi::this_work_item::get_nd_item<3>();
    xfer_copy(x, false);
    if (flag == nullptr) return;
    /*
    DPCT1078:185: Consider replacing memory_order::acq_rel with
    memory_order::seq_cst for correctness if strong memory order restrictions
    are needed.
    */
    sycl::atomic_fence(sycl::memory_order::acq_rel, sycl::memory_scope::system);
    /*
    DPCT1065:456: Consider replacing sycl::nd_item::barrier() with
    sycl::nd_item::barrier(sycl::access::fence_space::local_space) for better
    performance if there is no access to global memory.
    */
    item_ct1.barrier();
    if (item_ct1.get_local_id(2) == 0 &&
        dpct::atomic_fetch_add<sycl::access::address_space::generic_space>(
            cnt, 1u) == item_ct1.get_group_range(2) - 1) {
        *cnt = 0u;
        /*
        DPCT1078:186: Consider replacing memory_order::acq_rel with
        memory_order::seq_cst for correctness if strong memory order
        restrictions are needed.
        */
        sycl::atomic_fence(sycl::memory_order::acq_rel,
                           sycl::memory_scope::system);
        *(volatile unsigned*) flag = seq;
    }
}

__dpct_inline__ void wait_at_least(const unsigned *flag, unsigned seq) {
    while ((int) (*(volatile const unsigned*) flag - seq) < 0) {
#if defined(DPCT_COMPATIBILITY_TEMP) && DPCT_COMPATIBILITY_TEMP >= 700
        /*
        DPCT1008:457: __nanosleep function is not defined in SYCL. This is a
        hardware-specific feature. Consult with your hardware vendor to find a
        replacement.
        */
        /* CUDA pause hint; polling order is unchanged. */
#endif
    }
}

__dpct_inline__ void xfer_get_kernel(const Xfer x, const unsigned *flag,
                                     unsigned seq) {
    auto item_ct1 = sycl::ext::oneapi::this_work_item::get_nd_item<3>();
    if (item_ct1.get_local_id(2) == 0) wait_at_least(flag, seq);
    /*
    DPCT1065:458: Consider replacing sycl::nd_item::barrier() with
    sycl::nd_item::barrier(sycl::access::fence_space::local_space) for better
    performance if there is no access to global memory.
    */
    item_ct1.barrier();
    xfer_copy(x, true);
}

__dpct_inline__ void wait_flags_kernel(const unsigned *flags, int n, int stride,
                                       unsigned seq) {
    if (sycl::ext::oneapi::this_work_item::get_nd_item<3>().get_local_id(2) !=
        0) return;
#pragma unroll
    for (int i = 0; i < n; ++i) wait_at_least(flags + (size_t)i * stride, seq);
    /*
    DPCT1078:187: Consider replacing memory_order::acq_rel with
    memory_order::seq_cst for correctness if strong memory order restrictions
    are needed.
    */
    sycl::atomic_fence(sycl::memory_order::acq_rel, sycl::memory_scope::system);
}
// a batch of table entries: *pairs[2i] = pairs[2i + 1] (the expert cache's recent slots, between windows;
// Project Maya #344)
__dpct_inline__ void
tab_set_kernel(const unsigned long long *__restrict__ pairs, int n) {
    auto item_ct1 = sycl::ext::oneapi::this_work_item::get_nd_item<3>();
    const int i = item_ct1.get_group(2) * item_ct1.get_local_range(2) +
                  item_ct1.get_local_id(2);
    if (i < n) *(unsigned long long*) pairs[2 * i] = pairs[2 * i + 1];
}
__dpct_inline__ void set_flag_kernel(unsigned *flag, unsigned v) {
    /*
    DPCT1078:188: Consider replacing memory_order::acq_rel with
    memory_order::seq_cst for correctness if strong memory order restrictions
    are needed.
    */
    sycl::atomic_fence(sycl::memory_order::acq_rel, sycl::memory_scope::system);
    *(volatile unsigned*) flag = v;
}

// three members or more: a reduce-scatter and an all-gather in one launch, a block a contiguous range of the row cut
// in nm chunks.  The block puts its range of this member's partial out and flags it; once every member's flag of the
// range is up it sums chunk mi of the range in member order (ar_sum_kernel's sums, bit for bit), puts that out and
// flags it; once every member's sum flag is up it reads the other chunks' sums.  Each member reads about two rows over
// PCIe whatever nm is, where every member reading every other row grows with nm
/*
DPCT1110:189: The total declared local variable size in device function
ar_rs_kernel exceeds 128 bytes and may cause high register pressure. Consult
with your hardware vendor to find the total register size available and adjust
the code, or use smaller sub-group size to avoid high register pressure.
*/
__dpct_inline__ void ar_rs_kernel(float *x, int n4, float *rows, float *sums,
                                  size_t stride, unsigned *bflags, int nm,
                                  int mi, int slot, int slots, unsigned seq) {
    auto item_ct1 = sycl::ext::oneapi::this_work_item::get_nd_item<3>();
    sycl::float4 *x4 = (sycl::float4 *)x;
    const int per = (n4 + (int)item_ct1.get_group_range(2) - 1) /
                    (int)item_ct1.get_group_range(2);
    const int lo = item_ct1.get_group(2) * per, hi = sycl::min(n4, lo + per),
              cs = (hi - lo + nm - 1) / nm;
    auto rflag = [&](int m) {
                              return bflags +
                                     (((size_t)m * slots + slot) * kArBlocks +
                                      sycl::ext::oneapi::this_work_item::
                                          get_nd_item<3>()
                                              .get_group(2)) *
                                         8; };
    auto sflag = [&](int m) {
        return bflags +
               ((((size_t)nm + m) * slots + slot) * kArBlocks +
                sycl::ext::oneapi::this_work_item::get_nd_item<3>().get_group(
                    2)) *
                   8;
    };
    auto row = [&](int m) {
                            return (sycl::float4 *)(rows + ((size_t)m * slots + slot) * stride); };
    sycl::float4 *sum = (sycl::float4 *)(sums + (size_t)slot * stride);
    if (lo < hi) {
        sycl::float4 *mine = row(mi);
#pragma unroll
        for (int q = lo + item_ct1.get_local_id(2); q < hi;
             q += item_ct1.get_local_range(2)) mine[q] = x4[q];
    }
    /*
    DPCT1078:190: Consider replacing memory_order::acq_rel with
    memory_order::seq_cst for correctness if strong memory order restrictions
    are needed.
    */
    sycl::atomic_fence(sycl::memory_order::acq_rel, sycl::memory_scope::system);
    /*
    DPCT1065:459: Consider replacing sycl::nd_item::barrier() with
    sycl::nd_item::barrier(sycl::access::fence_space::local_space) for better
    performance if there is no access to global memory.
    */
    item_ct1.barrier();
    if (item_ct1.get_local_id(2) == 0) * (volatile unsigned *)rflag(mi) = seq;
    if ((int)item_ct1.get_local_id(2) < nm &&
        (int)item_ct1.get_local_id(2) != mi)
        wait_at_least(rflag(item_ct1.get_local_id(2)), seq);
    /*
    DPCT1065:460: Consider replacing sycl::nd_item::barrier() with
    sycl::nd_item::barrier(sycl::access::fence_space::local_space) for better
    performance if there is no access to global memory.
    */
    item_ct1.barrier();
    const int c0 = sycl::min(hi, lo + mi * cs), c1 = sycl::min(hi, c0 + cs);
    for (int q = c0 + item_ct1.get_local_id(2); q < c1;
         q += item_ct1.get_local_range(2)) {
        sycl::float4 v = sycl::float4(0.0f, 0.0f, 0.0f, 0.0f);
        for (int m = 0; m < nm; ++m) {
            const sycl::float4 p = m == mi ? x4[q] : ld_fresh(row(m) + q);
            if (m == 0) v = p;
            else {
                /*
                DPCT1013:463: The rounding mode could not be specified and the
                generated code may have different accuracy than the original
                code. Verify the correctness. SYCL math built-in function
                rounding mode is aligned with OpenCL C 1.2 standard.
                */
                v.x() = v.x() + p.x();
                /*
                DPCT1013:464: The rounding mode could not be specified and the
                generated code may have different accuracy than the original
                code. Verify the correctness. SYCL math built-in function
                rounding mode is aligned with OpenCL C 1.2 standard.
                */
                v.y() = v.y() + p.y();
                /*
                DPCT1013:465: The rounding mode could not be specified and the
                generated code may have different accuracy than the original
                code. Verify the correctness. SYCL math built-in function
                rounding mode is aligned with OpenCL C 1.2 standard.
                */
                v.z() = v.z() + p.z();
                /*
                DPCT1013:466: The rounding mode could not be specified and the
                generated code may have different accuracy than the original
                code. Verify the correctness. SYCL math built-in function
                rounding mode is aligned with OpenCL C 1.2 standard.
                */
                v.w() = v.w() + p.w();
            }
        }
        sum[q] = v;
        x4[q] = v;
    }
    /*
    DPCT1078:191: Consider replacing memory_order::acq_rel with
    memory_order::seq_cst for correctness if strong memory order restrictions
    are needed.
    */
    sycl::atomic_fence(sycl::memory_order::acq_rel, sycl::memory_scope::system);
    /*
    DPCT1065:461: Consider replacing sycl::nd_item::barrier() with
    sycl::nd_item::barrier(sycl::access::fence_space::local_space) for better
    performance if there is no access to global memory.
    */
    item_ct1.barrier();
    if (item_ct1.get_local_id(2) == 0) * (volatile unsigned *)sflag(mi) = seq;
    if ((int)item_ct1.get_local_id(2) < nm &&
        (int)item_ct1.get_local_id(2) != mi)
        wait_at_least(sflag(item_ct1.get_local_id(2)), seq);
    /*
    DPCT1065:462: Consider replacing sycl::nd_item::barrier() with
    sycl::nd_item::barrier(sycl::access::fence_space::local_space) for better
    performance if there is no access to global memory.
    */
    item_ct1.barrier();
#pragma unroll
    for (int q = lo + item_ct1.get_local_id(2); q < hi;
         q += item_ct1.get_local_range(2))
        if (q < c0 || q >= c1) x4[q] = ld_fresh(sum + q);
}

// ar_rs_kernel with the rows and the sums in FP16 (DS4_AR_F16): half the bytes over PCIe.  Every member's partial is
// rounded to FP16 (the owner's too), summed in F32 in member order, the sum rounded to FP16 - and every member, the
// owner too, takes that rounded sum, so the members' copies of the stream stay the same
__dpct_inline__ sycl::uint2 to_h4(sycl::float4 v) {
    const sycl::half2 a = sycl::float2(v.x(), v.y())
                              .convert<sycl::half, sycl::rounding_mode::rte>(),
                      b = sycl::float2(v.z(), v.w())
                              .convert<sycl::half, sycl::rounding_mode::rte>();
    return sycl::uint2(*(const unsigned *)&a, *(const unsigned *)&b);
}
__dpct_inline__ sycl::float4 from_h4(sycl::uint2 u) {
    const sycl::float2
        a = (*(const sycl::half2 *)&u.x())
                .template convert<float, sycl::rounding_mode::automatic>(),
        b = (*(const sycl::half2 *)&u.y())
                .template convert<float, sycl::rounding_mode::automatic>();
    return sycl::float4(a.x(), a.y(), b.x(), b.y());
}
__dpct_inline__ sycl::float4
sat4(sycl::float4
         v) { // (FP16's range: a partial past it saturates instead of inf)
    const float m = 65504.0f;
    return sycl::float4(sycl::fmin(sycl::fmax(v.x(), -m), m),
                        sycl::fmin(sycl::fmax(v.y(), -m), m),
                        sycl::fmin(sycl::fmax(v.z(), -m), m),
                        sycl::fmin(sycl::fmax(v.w(), -m), m));
}
/*
DPCT1110:192: The total declared local variable size in device function
ar_rs16_kernel exceeds 128 bytes and may cause high register pressure. Consult
with your hardware vendor to find the total register size available and adjust
the code, or use smaller sub-group size to avoid high register pressure.
*/
__dpct_inline__ void ar_rs16_kernel(float *x, int n4, float *rows, float *sums,
                                    size_t stride, unsigned *bflags, int nm,
                                    int mi, int slot, int slots, unsigned seq) {
    auto item_ct1 = sycl::ext::oneapi::this_work_item::get_nd_item<3>();
    sycl::float4 *x4 = (sycl::float4 *)x;
    const int per = (n4 + (int)item_ct1.get_group_range(2) - 1) /
                    (int)item_ct1.get_group_range(2);
    const int lo = item_ct1.get_group(2) * per, hi = sycl::min(n4, lo + per),
              cs = (hi - lo + nm - 1) / nm;
    auto rflag = [&](int m) {
                              return bflags +
                                     (((size_t)m * slots + slot) * kArBlocks +
                                      sycl::ext::oneapi::this_work_item::
                                          get_nd_item<3>()
                                              .get_group(2)) *
                                         8; };
    auto sflag = [&](int m) {
        return bflags +
               ((((size_t)nm + m) * slots + slot) * kArBlocks +
                sycl::ext::oneapi::this_work_item::get_nd_item<3>().get_group(
                    2)) *
                   8;
    };
    auto row = [&](int m) {
                            return (sycl::uint2 *)(rows + ((size_t)m * slots + slot) * stride); };   // (FP16: half a row)
    sycl::uint2 *sum = (sycl::uint2 *)(sums + (size_t)slot * stride);
    if (lo < hi) {
        sycl::uint2 *mine = row(mi);
#pragma unroll
        for (int q = lo + item_ct1.get_local_id(2); q < hi;
             q += item_ct1.get_local_range(2)) mine[q] = to_h4(sat4(x4[q]));
    }
    /*
    DPCT1078:193: Consider replacing memory_order::acq_rel with
    memory_order::seq_cst for correctness if strong memory order restrictions
    are needed.
    */
    sycl::atomic_fence(sycl::memory_order::acq_rel, sycl::memory_scope::system);
    /*
    DPCT1065:467: Consider replacing sycl::nd_item::barrier() with
    sycl::nd_item::barrier(sycl::access::fence_space::local_space) for better
    performance if there is no access to global memory.
    */
    item_ct1.barrier();
    if (item_ct1.get_local_id(2) == 0) * (volatile unsigned *)rflag(mi) = seq;
    if ((int)item_ct1.get_local_id(2) < nm &&
        (int)item_ct1.get_local_id(2) != mi)
        wait_at_least(rflag(item_ct1.get_local_id(2)), seq);
    /*
    DPCT1065:468: Consider replacing sycl::nd_item::barrier() with
    sycl::nd_item::barrier(sycl::access::fence_space::local_space) for better
    performance if there is no access to global memory.
    */
    item_ct1.barrier();
    const int c0 = sycl::min(hi, lo + mi * cs), c1 = sycl::min(hi, c0 + cs);
    for (int q = c0 + item_ct1.get_local_id(2); q < c1;
         q += item_ct1.get_local_range(2)) {
        sycl::float4 v = sycl::float4(0.0f, 0.0f, 0.0f, 0.0f);
        for (int m = 0; m < nm; ++m) {
            const sycl::uint2 u =
                m == mi ? to_h4(sat4(x4[q])) : ld_fresh(row(m) + q);
            const sycl::float4 p = from_h4(u);
            if (m == 0) v = p;
            else {
                /*
                DPCT1013:471: The rounding mode could not be specified and the
                generated code may have different accuracy than the original
                code. Verify the correctness. SYCL math built-in function
                rounding mode is aligned with OpenCL C 1.2 standard.
                */
                v.x() = v.x() + p.x();
                /*
                DPCT1013:472: The rounding mode could not be specified and the
                generated code may have different accuracy than the original
                code. Verify the correctness. SYCL math built-in function
                rounding mode is aligned with OpenCL C 1.2 standard.
                */
                v.y() = v.y() + p.y();
                /*
                DPCT1013:473: The rounding mode could not be specified and the
                generated code may have different accuracy than the original
                code. Verify the correctness. SYCL math built-in function
                rounding mode is aligned with OpenCL C 1.2 standard.
                */
                v.z() = v.z() + p.z();
                /*
                DPCT1013:474: The rounding mode could not be specified and the
                generated code may have different accuracy than the original
                code. Verify the correctness. SYCL math built-in function
                rounding mode is aligned with OpenCL C 1.2 standard.
                */
                v.w() = v.w() + p.w();
            }
        }
        const sycl::uint2 s16 = to_h4(sat4(v));
        sum[q] = s16;
        x4[q] = from_h4(s16);
    }
    /*
    DPCT1078:194: Consider replacing memory_order::acq_rel with
    memory_order::seq_cst for correctness if strong memory order restrictions
    are needed.
    */
    sycl::atomic_fence(sycl::memory_order::acq_rel, sycl::memory_scope::system);
    /*
    DPCT1065:469: Consider replacing sycl::nd_item::barrier() with
    sycl::nd_item::barrier(sycl::access::fence_space::local_space) for better
    performance if there is no access to global memory.
    */
    item_ct1.barrier();
    if (item_ct1.get_local_id(2) == 0) * (volatile unsigned *)sflag(mi) = seq;
    if ((int)item_ct1.get_local_id(2) < nm &&
        (int)item_ct1.get_local_id(2) != mi)
        wait_at_least(sflag(item_ct1.get_local_id(2)), seq);
    /*
    DPCT1065:470: Consider replacing sycl::nd_item::barrier() with
    sycl::nd_item::barrier(sycl::access::fence_space::local_space) for better
    performance if there is no access to global memory.
    */
    item_ct1.barrier();
#pragma unroll
    for (int q = lo + item_ct1.get_local_id(2); q < hi;
         q += item_ct1.get_local_range(2))
        if (q < c0 || q >= c1) x4[q] = from_h4(ld_fresh(sum + q));
}

// x[i] = the members' partials summed in member order (this member's from x, the others' from their host rows)
__dpct_inline__ void ar_sum_kernel(float *x, int n4, const float *peers,
                                   size_t stride, const unsigned *flags, int nm,
                                   int mi, int slot, int slots, unsigned seq) {
    auto item_ct1 = sycl::ext::oneapi::this_work_item::get_nd_item<3>();
    if (item_ct1.get_local_id(2) == 0)
#pragma unroll
        for (int m = 0; m < nm; ++m)
            if (m != mi) wait_at_least(flags + ((size_t) m * slots + slot) * 32, seq);
    /*
    DPCT1065:475: Consider replacing sycl::nd_item::barrier() with
    sycl::nd_item::barrier(sycl::access::fence_space::local_space) for better
    performance if there is no access to global memory.
    */
    item_ct1.barrier();
    for (int q = item_ct1.get_group(2) * item_ct1.get_local_range(2) +
                 item_ct1.get_local_id(2);
         q < n4;
         q += item_ct1.get_group_range(2) * item_ct1.get_local_range(2)) {
        sycl::float4 v = sycl::float4(0.0f, 0.0f, 0.0f, 0.0f);
        for (int m = 0; m < nm; ++m) {
            const sycl::float4 p =
                m == mi
                    ? ((const sycl::float4 *)x)[q]
                    : ld_fresh(
                          (const sycl::float4 *)(peers +
                                                 ((size_t)m * slots + slot) *
                                                     stride) +
                          q);
            if (m == 0) v = p;
            else {
                /*
                DPCT1013:476: The rounding mode could not be specified and the
                generated code may have different accuracy than the original
                code. Verify the correctness. SYCL math built-in function
                rounding mode is aligned with OpenCL C 1.2 standard.
                */
                v.x() = v.x() + p.x();
                /*
                DPCT1013:477: The rounding mode could not be specified and the
                generated code may have different accuracy than the original
                code. Verify the correctness. SYCL math built-in function
                rounding mode is aligned with OpenCL C 1.2 standard.
                */
                v.y() = v.y() + p.y();
                /*
                DPCT1013:478: The rounding mode could not be specified and the
                generated code may have different accuracy than the original
                code. Verify the correctness. SYCL math built-in function
                rounding mode is aligned with OpenCL C 1.2 standard.
                */
                v.z() = v.z() + p.z();
                /*
                DPCT1013:479: The rounding mode could not be specified and the
                generated code may have different accuracy than the original
                code. Verify the correctness. SYCL math built-in function
                rounding mode is aligned with OpenCL C 1.2 standard.
                */
                v.w() = v.w() + p.w();
            }
        }
        ((sycl::float4 *)x)[q] = v;
    }
}


// a thread a (token, value): every stream's value read before any is written
__dpct_inline__ void hc_post_inplace_kernel(const float *__restrict__ y,
                                            float *h,
                                            const float *__restrict__ post,
                                            const float *__restrict__ comb,
                                            int dim, int hc) {
    auto item_ct1 = sycl::ext::oneapi::this_work_item::get_nd_item<3>();
    const int e = item_ct1.get_group(2) * item_ct1.get_local_range(2) +
                  item_ct1.get_local_id(2),
              t = item_ct1.get_group(1);
    if (e >= dim) return;
    float* r = h + (size_t) t * hc * dim;
    float rv[8];
#pragma unroll
    for (int j = 0; j < hc; ++j) rv[j] = r[(size_t)j * dim + e];
    const float yv = y[(size_t) t * dim + e];
    for (int k = 0; k < hc; ++k) {
        float a = post[(size_t) t * 8 + k] * yv;
#pragma unroll
        for (int j = 0; j < hc; ++j) a +=
            comb[(size_t)t * 64 + j * hc + k] * rv[j];
        r[(size_t) k * dim + e] = a;
    }
}

__dpct_inline__ void hc_mix_scale_kernel(const float *__restrict__ h, int hcdim,
                                         int n_mix, float eps, float *mixes) {
    auto item_ct1 = sycl::ext::oneapi::this_work_item::get_nd_item<3>();
    const int t = item_ct1.get_group(2);
    const float* x = h + (size_t) t * hcdim;
    float ss = 0.0f;
    for (int k = item_ct1.get_local_id(2) * 4; k < hcdim; k += kThreads * 4) {
        const sycl::float4 hv = *(const sycl::float4 *)(x + k);
        ss += hv.x() * hv.x() + hv.y() * hv.y() + hv.z() * hv.z() +
              hv.w() * hv.w();
    }
    ss = block_sum(ss);
    const float inv = sycl::rsqrt(ss / (float)hcdim + eps);
    if (item_ct1.get_local_id(2) < n_mix)
        mixes[(size_t)t * 32 + item_ct1.get_local_id(2)] *= inv;
}

__dpct_inline__ void engram_dequant_tok_kernel(const uint8_t *__restrict__ src,
                                               size_t per, int cols, int hd,
                                               float *__restrict__ out,
                                               int fmt) {
    auto item_ct1 = sycl::ext::oneapi::this_work_item::get_nd_item<3>();
    const int r = item_ct1.get_group(2), t = r / cols, c = r - t * cols;
    const uint8_t* rows = src + (size_t) t * per + (size_t) c * hd;
    const uint8_t* scales = src + (size_t) t * per + (size_t) cols * hd + (size_t) c * (hd / 32);
    if (fmt == 1) {   // (ggml Q6_K superblocks at the row's start)
#pragma unroll
        for (int e = item_ct1.get_local_id(2); e < hd;
             e += item_ct1.get_local_range(2))
            out[(size_t) r * hd + e] = q6k_value(rows + (size_t) (e / 256) * kQ6kBytes, e % 256);
        return;
    }
    if (fmt == 2) {   // (ggml IQ4_NL blocks at the row's start, their scales inside)
#pragma unroll
        for (int e = item_ct1.get_local_id(2); e < hd;
             e += item_ct1.get_local_range(2))
            out[(size_t) r * hd + e] = iq4nl_value(rows + (size_t) (e / 32) * 18, e % 32);
        return;
    }
    for (int e = item_ct1.get_local_id(2); e < hd;
         e += item_ct1.get_local_range(2)) {
        const float hv = e4m3_value(rows[e]);
        const uint8_t sb = scales[e / 32];
        const float s =
            sb == 0xffu ? sycl::bit_cast<float>(0x7fc00000)
                        : (sb == 0 ? 5.877471754111438e-39f
                                   : sycl::bit_cast<float>((uint32_t)sb << 23));
        out[(size_t) r * hd + e] = hv * s;
    }
}

// a block a virtual slot: its picks counted (sort pass 1), or (fill) placed in token order by a block-wide scan
template <bool FILL>
__dpct_inline__ void moe_sort_kernel(const int *__restrict__ idx, int T, int k,
                                     const int *__restrict__ vs, int *bounds,
                                     int *row_tok, int *pos) {
    auto item_ct1 = sycl::ext::oneapi::this_work_item::get_nd_item<3>();
    const int slot = item_ct1.get_group(2);
    auto &warp_n = *sycl::ext::oneapi::group_local_memory_for_overwrite<
        int[kThreads / 32]>(
        sycl::ext::oneapi::this_work_item::get_work_group<3>());
    auto &base_s = *sycl::ext::oneapi::group_local_memory_for_overwrite<int>(
        sycl::ext::oneapi::this_work_item::get_work_group<3>());
    if (item_ct1.get_local_id(2) == 0) base_s = FILL ? bounds[slot] : 0;
    /*
    DPCT1065:480: Consider replacing sycl::nd_item::barrier() with
    sycl::nd_item::barrier(sycl::access::fence_space::local_space) for better
    performance if there is no access to global memory.
    */
    item_ct1.barrier();
    const int lane = item_ct1.get_local_id(2) & 31,
              wp = item_ct1.get_local_id(2) >> 5;
    const int n = T * k;
    for (int q0 = 0; q0 < n; q0 += kThreads) {
        const int q = q0 + item_ct1.get_local_id(2);
        const int t = q / k, i = q - t * k;
        const bool hit = q < n && vs[idx[(size_t) t * kPicks + i]] == slot;
        const unsigned m = sycl::reduce_over_group(
            sycl::ext::oneapi::this_work_item::get_sub_group(),
            (0xffffffffu &
             (0x1 << sycl::ext::oneapi::this_work_item::get_sub_group()
                         .get_local_linear_id())) &&
                    hit
                ? (0x1 << sycl::ext::oneapi::this_work_item::get_sub_group()
                              .get_local_linear_id())
                : 0,
            sycl::ext::oneapi::plus<>());
        if (lane == 0) warp_n[wp] = sycl::popcount(m);
        /*
        DPCT1118:195: SYCL group functions and algorithms must be encountered in
        converged control flow. You may need to adjust the code.
        */
        /*
        DPCT1065:481: Consider replacing sycl::nd_item::barrier() with
        sycl::nd_item::barrier(sycl::access::fence_space::local_space) for
        better performance if there is no access to global memory.
        */
        item_ct1.barrier();
        int before = 0, total = 0;
#pragma unroll
        for (int w = 0; w < kThreads / 32; ++w) {
            if (w < wp) before += warp_n[w];
            total += warp_n[w];
        }
        if (FILL && hit) {
            const int r =
                base_s + before + sycl::popcount(m & ((1u << lane) - 1u));
            row_tok[r] = t;
            pos[(size_t) t * kPicks + i] = r;
        }
        /*
        DPCT1118:196: SYCL group functions and algorithms must be encountered in
        converged control flow. You may need to adjust the code.
        */
        /*
        DPCT1065:482: Consider replacing sycl::nd_item::barrier() with
        sycl::nd_item::barrier(sycl::access::fence_space::local_space) for
        better performance if there is no access to global memory.
        */
        item_ct1.barrier();
        if (item_ct1.get_local_id(2) == 0) base_s += total;
        /*
        DPCT1118:197: SYCL group functions and algorithms must be encountered in
        converged control flow. You may need to adjust the code.
        */
        /*
        DPCT1065:483: Consider replacing sycl::nd_item::barrier() with
        sycl::nd_item::barrier(sycl::access::fence_space::local_space) for
        better performance if there is no access to global memory.
        */
        item_ct1.barrier();
    }
    if (!FILL && item_ct1.get_local_id(2) == 0) bounds[slot + 1] =
        base_s; // (the counts; the scan makes them bounds)
}

__dpct_inline__ void bounds_scan_kernel(
    int *bounds,
    int n_vs) { // one thread: bounds[0] = 0, bounds[s + 1] += bounds[s]
    if (sycl::ext::oneapi::this_work_item::get_nd_item<3>().get_local_id(2) !=
        0) return;
    bounds[0] = 0;
#pragma unroll
    for (int s = 0; s < n_vs; ++s) bounds[s + 1] += bounds[s];
}

__dpct_inline__ void moe_combine_kernel(float *acc, const float *__restrict__ y,
                                        const int *__restrict__ pos,
                                        const float *__restrict__ w, int k,
                                        int r0, int r1, int dim) {
    auto item_ct1 = sycl::ext::oneapi::this_work_item::get_nd_item<3>();
    const int e = item_ct1.get_group(2) * item_ct1.get_local_range(2) +
                  item_ct1.get_local_id(2),
              t = item_ct1.get_group(1);
    if (e >= dim) return;
    float a = acc[(size_t) t * dim + e];
    bool any = false;
    for (int i = 0; i < k; ++i) {
        const int r = pos[(size_t) t * kPicks + i];
        if (r < r0 || r >= r1) continue;
        /*
        DPCT1013:484: The rounding mode could not be specified and the generated
        code may have different accuracy than the original code. Verify the
        correctness. SYCL math built-in function rounding mode is aligned with
        OpenCL C 1.2 standard.
        */
        a = sycl::fma(w[(size_t)t * kPicks + i], y[(size_t)(r - r0) * dim + e],
                      a);
        any = true;
    }
    if (any) acc[(size_t) t * dim + e] = a;
}

int cdiv(int a, int b) { return (a + b - 1) / b; }

}  // namespace

namespace {

// ---- the split-KV attention's scratch: a device allocation a (device, stream), made on first use and kept
constexpr int kSplitG = 4, kSplitR = 40, kSplitCap = 24, kSplitMaxT = 5;
bool split_enabled() {
    static const bool on = [] {
        const char* e = std::getenv("DS4_ATTN_SPLIT");
        return e != nullptr && std::atoi(e) != 0;   // opt-in: no gain measured on 4x R9700 (21.8 vs 21.9 tok/s)
    }();
    return on;
}
// DS4_FAST_ATTN: the chunked prompt's packed-KV attention on the matrix cores (sparse_attn_wmma_kernel).  The
// QK/PV sums are the matrix cores' order, so the outputs differ from the scalar kernel's in the last places (the owner
// approved the trade for the prefill; the decode path stays on the scalar kernels).
// 0.2.7: ON by default on gfx1200 / gfx1201 (RDNA4: the R9700, RX 9000), where it was measured and gated (KL 0.0003 against the
// 0.006 gate, ids identical); gfx11 and gfx1151 (Strix Halo) are untested and keep the scalar kernels.  DS4_FAST_ATTN=0 is the
// off switch (exactly the 0.2.6 arithmetic), DS4_FAST_ATTN=1 turns it on elsewhere.  Decided per device (the first call there).
// The Marlin stream measured it ON on 4x R9700 (gfx1201), 2.25 bpw, 8K code prompts: 1180 -> 1354 ppts (+14.7%), the same ids,
// KL 0.10833 against 0.10925 for the scalar path.  gfx1151 on Windows was the untested part; measured on Strix Halo 2026-10-28
// (32K prompt, 2.00 bpw): 101.6 -> 70.8 s (+43.6%), the sparse-attention stage 38.3 -> 7.0 s, the same greedy first token
// (27408,), so it joins the default there and stays opt-in on every other gfx11 part.
#if defined(STRATA_USE_HIP)
bool device_is_gfx12() {
    static std::atomic<int> cache[64];   // 0 unknown, 1 no, 2 yes
    int dev = 0;
    cudaGetDevice(&dev);
    if (dev < 0 || dev >= 64) return false;
    int c = cache[dev].load();
    if (c == 0) {
        cudaDeviceProp prop{};
        const bool yes = cudaGetDeviceProperties(&prop, dev) == cudaSuccess &&
                         (std::strncmp(prop.gcnArchName, "gfx1200", 7) == 0 || std::strncmp(prop.gcnArchName, "gfx1201", 7) == 0);
        c = yes ? 2 : 1;
        cache[dev].store(c);
    }
    return c == 2;
}
#if defined(_WIN32)
bool device_is_gfx1151win() {
    static std::atomic<int> cache[64];   // 0 unknown, 1 no, 2 yes
    int dev = 0;
    cudaGetDevice(&dev);
    if (dev < 0 || dev >= 64) return false;
    int c = cache[dev].load();
    if (c == 0) {
        cudaDeviceProp prop{};
        const bool yes = cudaGetDeviceProperties(&prop, dev) == cudaSuccess && std::strncmp(prop.gcnArchName, "gfx1151", 7) == 0;
        c = yes ? 2 : 1;
        cache[dev].store(c);
    }
    return c == 2;
}
#endif
#endif
bool fast_attn_enabled() {
    static const int env = [] {
        const char* e = std::getenv("DS4_FAST_ATTN");
        return e == nullptr ? -1 : (std::atoi(e) != 0 ? 1 : 0);
    }();
#if defined(STRATA_USE_HIP)
#if defined(_WIN32)
    return strata::ds4::fast_attn_default(env, device_is_gfx12(), device_is_gfx1151win(), strata::ds4::defaults_026());
#else
    return strata::ds4::fast_attn_default(env, device_is_gfx12(), false, strata::ds4::defaults_026());
#endif
#else
    return strata::ds4::fast_attn_default(env, false, false, strata::ds4::defaults_026());
#endif
}
attnsplit::AttnSplitScratch split_scratch(dpct::queue_ptr s, int heads) try {
    static std::mutex mu;
    static std::map<std::pair<int, dpct::queue_ptr>,
                    attnsplit::AttnSplitScratch>
        m;
    int dev = 0;
    dev = dpct::get_current_device_id();
    std::lock_guard<std::mutex> lk(mu);
    auto it = m.find({dev, s});
    if (it != m.end()) return it->second;
    attnsplit::AttnSplitScratch sc;
    sc.nsplit = kSplitCap;
    sc.max_t = kSplitMaxT;
    const size_t np = (size_t) kSplitMaxT * heads * kSplitCap;
    if (DPCT_CHECK_ERROR(sc.part = sycl::malloc_device<float>(
                             np * 512, dpct::get_in_order_queue())) != 0 ||
        DPCT_CHECK_ERROR(sc.ml = sycl::malloc_device<float>(
                             np * 2, dpct::get_in_order_queue())) != 0) {
        /*
        DPCT1026:485: The call to cudaGetLastError was removed because this
        functionality is redundant in SYCL.
        */
        if (sc.part != nullptr) sycl::free(sc.part, dpct::get_in_order_queue());
        sc = attnsplit::AttnSplitScratch{};
    }
    m[{dev, s}] = sc;
    return sc;
}
catch (sycl::exception const &exc) {
  std::cerr << exc.what() << "Exception caught at file:" << __FILE__
            << ", line:" << __LINE__ << std::endl;
  std::exit(1);
}
}  // namespace

void route_publish(const float *logits, const float *bias, int n, int k, int T,
                   float route_scale, const unsigned long long *vram,
                   const float *x, int dim, int *idx, float *w,
                   unsigned long long *plan_ptr, Req *req, unsigned seq,
                   dpct::queue_ptr s, const float *bias_vl, unsigned vl,
                   int na0, int na1) {
#if defined(STRATA_METAL)
    // selection only: the prefetch schedule (req) is published by the CUDA kernel, and is turned on separately here
    if (strata::dsv41::metal_host::route_publish(logits, bias, n, k, T, route_scale, vram, idx, w, plan_ptr,
            vl != 0 ? bias_vl : nullptr, vl)) {
        // no prefetcher on the Mac path, so publish_kernel has no consumer: the selection is the whole job here
        (void) req; (void) seq; (void) na0; (void) na1; (void) x; (void) dim;
        return;
    }
#endif
    /*
    DPCT1049:198: The work-group size passed to the SYCL kernel may exceed the
    limit. To get the device limit, query info::device::max_work_group_size.
    Adjust the work-group size if needed.
    */
    {
        g_router_bad.init(*s);

        s->submit([&](sycl::handler &cgh) {
            auto g_router_bad_ptr_ct1 = g_router_bad.get_ptr();

            auto vl_bias_vl_nullptr_ct12 = vl != 0 ? bias_vl : nullptr;

            cgh.parallel_for(
                sycl::nd_range<3>(sycl::range(1, 1, T) * sycl::range(1, 1, 512),
                                  sycl::range(1, 1, 512)),
                [=](sycl::nd_item<3> item_ct1)
                    [[sycl::reqd_sub_group_size(32)]] {
                        route_publish_kernel(logits, bias, n, k, route_scale,
                                             vram, x, dim, idx, w, plan_ptr,
                                             req, vl_bias_vl_nullptr_ct12, vl,
                                             *g_router_bad_ptr_ct1);
                    });
        });
    }
        if (req != nullptr) s->parallel_for(
            sycl::nd_range<3>(sycl::range(1, 1, 1), sycl::range(1, 1, 1)),
            [=](sycl::nd_item<3> item_ct1) {
                publish_kernel(req, T, seq, na0, na1);
            });
}
void route_ahead(const float *logits, const float *bias, int n, int m, int T,
                 Req *req, int off, dpct::queue_ptr s) {
#if defined(STRATA_METAL)
    // no prefetch schedule on the Mac path yet: the router publishes no requests, so there is nothing to look ahead for
    (void) logits; (void) bias; (void) n; (void) m; (void) T; (void) req; (void) off; (void) s;
    return;
#endif
    if (req != nullptr && m > 0 && T > 0 && n <= 1024 && off + m <= kAhead)
        /*
        DPCT1049:199: The work-group size passed to the SYCL kernel may exceed
        the limit. To get the device limit, query
        info::device::max_work_group_size. Adjust the work-group size if needed.
        */
        s->parallel_for(
            sycl::nd_range<3>(sycl::range(1, 1, T) * sycl::range(1, 1, 512),
                              sycl::range(1, 1, 512)),
            [=](sycl::nd_item<3> item_ct1) [[sycl::reqd_sub_group_size(32)]] {
                route_ahead_kernel(logits, bias, n, m, req, off);
            });
}
bool router_nonfinite_take(dpct::queue_ptr s) try {
#if defined(STRATA_METAL)
    // the MSL router kernels raise a flag in a buffer of the Metal host (ds4_route_fallback_pick); the stream's work is done first
    if (cudaStreamSynchronize(s) != cudaSuccess) {
        (void) cudaGetLastError();
        return false;
    }
    return strata::dsv41::metal_host::router_bad_take();
#else
    int v = 0;
    if (DPCT_CHECK_ERROR(s->wait()) != 0 ||
        DPCT_CHECK_ERROR(dpct::get_in_order_queue()
                             .memcpy(&v, g_router_bad.get_ptr(), sizeof v)
                             .wait()) != 0) {
        /*
        DPCT1010:486: SYCL uses exceptions to report errors and does not use the
        error codes. The cudaGetLastError function call was replaced with 0. You
        need to rewrite this code.
        */
        (void)0;
        return false;
    }
    if (v != 0) {
        const int zero = 0;
        /*
        DPCT1010:487: SYCL uses exceptions to report errors and does not use the
        error codes. The cudaGetLastError function call was replaced with 0. You
        need to rewrite this code.
        */
        if (DPCT_CHECK_ERROR(
                dpct::get_in_order_queue()
                    .memcpy(g_router_bad.get_ptr(), &zero, sizeof zero)
                    .wait()) != 0)(void) 0;
    }
    return v != 0;
#endif
}
catch (sycl::exception const &exc) {
  std::cerr << exc.what() << "Exception caught at file:" << __FILE__
            << ", line:" << __LINE__ << std::endl;
  std::exit(1);
}
void wait_seq(const unsigned *seq_ptr, unsigned seq, dpct::queue_ptr s) {
    ds4_sycl_wait_host(seq_ptr, seq, *s);
}

// the RAM store's experts for the device (a decode token, window of one): a pick the VRAM table left at 0 whose expert the store holds
// (zc_tab[expert] = the slot's device address over PCIe, 0: not) is copied to staging slot i with wide coalesced loads (the PCIe
// rate; the expert kernels' own small loads over PCIe ran at a tenth of it), then the plan points at the copy
__dpct_inline__ void
zc_copy_kernel(const unsigned long long *__restrict__ plan,
               const int *__restrict__ idx,
               const unsigned long long *__restrict__ zc_tab,
               uint8_t *__restrict__ stage, size_t stride, size_t bytes) {
    auto item_ct1 = sycl::ext::oneapi::this_work_item::get_nd_item<3>();
    const int i = item_ct1.get_group(1);
    if (plan[i] != 0) return;
    const unsigned long long p = zc_tab[idx[i]];
    if (p == 0) return;
    const sycl::uint4 *src = (const sycl::uint4 *)p;
    sycl::uint4 *dst = (sycl::uint4 *)(stage + (size_t)i * stride);
    const size_t n = bytes / 16;
#pragma unroll
    for (size_t o =
             (size_t)item_ct1.get_group(2) * item_ct1.get_local_range(2) +
             item_ct1.get_local_id(2);
         o < n; o += (size_t)item_ct1.get_group_range(2) *
                     item_ct1.get_local_range(2)) dst[o] = src[o];
}
__dpct_inline__ void
zc_fix_kernel(unsigned long long *__restrict__ plan,
              const int *__restrict__ idx,
              const unsigned long long *__restrict__ zc_tab, int k,
              uint8_t *__restrict__ stage, size_t stride) {
    const int i =
        sycl::ext::oneapi::this_work_item::get_nd_item<3>().get_local_id(2);
    if (i >= k || plan[i] != 0) return;
    if (zc_tab[idx[i]] != 0) plan[i] = (unsigned long long) (stage + (size_t) i * stride);
}
void zc_stage(unsigned long long *plan, const int *idx,
              const unsigned long long *zc_tab, int k, uint8_t *stage,
              size_t stride, size_t bytes, dpct::queue_ptr s) {
    s->parallel_for(
        sycl::nd_range<3>(sycl::range(1, k, 64) * sycl::range(1, 1, 256),
                          sycl::range(1, 1, 256)),
        [=](sycl::nd_item<3> item_ct1) {
            zc_copy_kernel(plan, idx, zc_tab, stage, stride, bytes);
        });
    s->parallel_for(
        sycl::nd_range<3>(sycl::range(1, 1, 32), sycl::range(1, 1, 32)),
        [=](sycl::nd_item<3> item_ct1) {
            zc_fix_kernel(plan, idx, zc_tab, k, stage, stride);
        });
}

void init_pre(float *pre, int T, dpct::queue_ptr s) {
#if defined(STRATA_METAL)
    if (ds4_dbg("DS4_SKIP_INITPRE")) return;
#endif
#if defined(STRATA_METAL)
    if (strata::dsv41::metal_host::init_pre(pre, T, s)) return;   // the ported MSL kernel
#endif
    s->parallel_for(sycl::nd_range<3>(sycl::range(1, 1, cdiv(T * 8, 128)) *
                                          sycl::range(1, 1, 128),
                                      sycl::range(1, 1, 128)),
                    [=](sycl::nd_item<3> item_ct1) {
                        init_pre_kernel(pre, T);
                    });
}

void embed(const uint16_t *rows, int T, float *h, int dim, int hc,
           dpct::queue_ptr s) {
#if defined(STRATA_METAL)
    if (strata::dsv41::metal_host::embed_addr(rows, T, h, dim, hc)) return;
#endif
#if defined(STRATA_METAL)
    if (strata::dsv41::metal_host::embed(rows, T, h, dim, hc, s)) return;
#endif
    s->parallel_for(sycl::nd_range<3>(sycl::range(1, T, cdiv(dim, kThreads)) *
                                          sycl::range(1, 1, kThreads),
                                      sycl::range(1, 1, kThreads)),
                    [=](sycl::nd_item<3> item_ct1) {
                        embed_kernel(rows, h, dim, hc);
                    });
}

void finish(float *x, int n_vec, int len, long long stride, const Finish &f,
            dpct::queue_ptr s) {
    if (len > kFinMax * kThreads) {
        std::fprintf(stderr, "ds4 finish: a vector of %d values is more than %d\n", len, kFinMax * kThreads);
        return;
    }
    if (n_vec <= 0) return;
    if (f.pdst != nullptr) {
        if (f.quant < kKvFp8 || f.quant > kKvFp4p || len % 32 != 0) {
            std::fprintf(stderr, "ds4 finish: packed rows of quant %d, %d values not covered\n", f.quant, len);
            return;
        }
#if defined(STRATA_METAL)
        const auto& mp = *reinterpret_cast<const strata::dsv41::metal_host::FinishIn*>(&f);
        if (strata::dsv41::metal_host::finish(x, n_vec, len, stride, mp)) return;   // finish_kernel<true> (the KV pack)
#endif
        {
            dpct::has_capability_or_fail(s->get_device(), {sycl::aspect::fp16});

            s->parallel_for(sycl::nd_range<3>(sycl::range(1, 1, n_vec) *
                                                  sycl::range(1, 1, kThreads),
                                              sycl::range(1, 1, kThreads)),
                            [=](sycl::nd_item<3> item_ct1)
                                [[sycl::reqd_sub_group_size(32)]] {
                                    finish_kernel<true>(x, len, stride, f);
                                });
        }
    } else {
#if defined(STRATA_METAL)
        const auto& mf = *reinterpret_cast<const strata::dsv41::metal_host::FinishIn*>(&f);
        if (strata::dsv41::metal_host::finish(x, n_vec, len, stride, mf)) return;   // the ported MSL kernel
#endif
        dpct::has_capability_or_fail(s->get_device(), {sycl::aspect::fp16});

        s->parallel_for(sycl::nd_range<3>(sycl::range(1, 1, n_vec) *
                                              sycl::range(1, 1, kThreads),
                                          sycl::range(1, 1, kThreads)),
                        [=](sycl::nd_item<3> item_ct1)
                            [[sycl::reqd_sub_group_size(32)]] {
                                finish_kernel<false>(x, len, stride, f);
                            });
    }
}

void hc_mix(const float *h, const float *fn, int hcdim, int n_mix, int T,
            float eps, float *mixes, dpct::queue_ptr s) {
#if defined(STRATA_METAL)
    if (strata::dsv41::metal_host::hc_mix(h, fn, hcdim, n_mix, T, eps, mixes)) return;
#endif
    s->parallel_for(sycl::nd_range<3>(sycl::range(1, T, n_mix) *
                                          sycl::range(1, 1, kThreads),
                                      sycl::range(1, 1, kThreads)),
                    [=](sycl::nd_item<3> item_ct1)
                        [[sycl::reqd_sub_group_size(32)]] {
                            hc_mix_kernel(h, fn, hcdim, eps, mixes);
                        });
}
void hc_mix_split(const float *h, const float *fn, int hcdim, int n_mix, int T,
                  float eps, float *mixes, unsigned *cnt, const float *scale,
                  const float *base, int hc, int iters, float hc_eps,
                  float *pre, float *post, float *comb, dpct::queue_ptr s) {
    s->parallel_for(
        sycl::nd_range<3>(sycl::range(1, T, n_mix) *
                              sycl::range(1, 1, kThreads),
                          sycl::range(1, 1, kThreads)),
        [=](sycl::nd_item<3> item_ct1) [[sycl::reqd_sub_group_size(32)]] {
            hc_mix_split_kernel(h, fn, hcdim, eps, mixes, cnt, scale, base, hc,
                                iters, hc_eps, pre, post, comb);
        });
}
void hc_split(const float *mixes, const float *scale, const float *base, int hc,
              int iters, float hc_eps, int T, float *pre, float *post,
              float *comb, dpct::queue_ptr s) {
#if defined(STRATA_METAL)
    if (strata::dsv41::metal_host::hc_split(mixes, scale, base, hc, iters, hc_eps, T, pre, post, comb)) return;
#endif
    s->parallel_for(
        sycl::nd_range<3>(sycl::range(1, 1, T) * sycl::range(1, 1, 32),
                          sycl::range(1, 1, 32)),
        [=](sycl::nd_item<3> item_ct1) [[sycl::reqd_sub_group_size(32)]] {
            hc_split_kernel(mixes, scale, base, hc, iters, hc_eps, pre, post,
                            comb);
        });
}
void hc_mix_split(const float *h, const float *fn, int hcdim, int n_mix, int T,
                  float eps, float *mixes, const float *scale,
                  const float *base, int hc, int iters, float hc_eps,
                  float *pre, float *post, float *comb, dpct::queue_ptr s) {
#if defined(STRATA_METAL)
    // one dispatch: the mix and the sinkhorn split of its result (bit-identical to the two, see ds4_init.metal)
    if (strata::dsv41::metal_host::hc_mix_split(h, fn, hcdim, n_mix, T, eps, mixes, scale, base, hc, iters, hc_eps,
                                                pre, post, comb))
        return;
#endif
    hc_mix(h, fn, hcdim, n_mix, T, eps, mixes, s);
    hc_split(mixes, scale, base, hc, iters, hc_eps, T, pre, post, comb, s);
}

void hc_pre_norm(const float *h, const float *mix, const float *w, int dim,
                 int hc, int T, float eps, float *x, dpct::queue_ptr s) {
#if defined(STRATA_METAL)
    if (strata::dsv41::metal_host::hc_pre_norm(h, mix, w, dim, hc, T, eps, x)) return;
#endif
    /*
    DPCT1049:200: The work-group size passed to the SYCL kernel may exceed the
    limit. To get the device limit, query info::device::max_work_group_size.
    Adjust the work-group size if needed.
    */
    s->parallel_for(
        sycl::nd_range<3>(sycl::range(1, 1, T) * sycl::range(1, 1, 1024),
                          sycl::range(1, 1, 1024)),
        [=](sycl::nd_item<3> item_ct1) [[sycl::reqd_sub_group_size(32)]] {
            hc_pre_norm_kernel(h, mix, w, dim, hc, eps, x);
        });
}
void hc_post(const float *y, const float *res, const float *post,
             const float *comb, int dim, int hc, int T, float *out,
             dpct::queue_ptr s) {
#if defined(STRATA_METAL)
    if (strata::dsv41::metal_host::hc_post_v(y, res, post, comb, dim, hc, T, out)) return;
#endif
    s->parallel_for(
        sycl::nd_range<3>(sycl::range(1, T, cdiv(dim * hc, kThreads)) *
                              sycl::range(1, 1, kThreads),
                          sycl::range(1, 1, kThreads)),
        [=](sycl::nd_item<3> item_ct1) {
            hc_post_kernel(y, res, post, comb, dim, hc, out);
        });
}

void hc_post_inplace(const float *y, float *h, const float *post,
                     const float *comb, int dim, int hc, int T,
                     dpct::queue_ptr s) {
#if defined(STRATA_METAL)
    if (strata::dsv41::metal_host::hc_post(y, h, post, comb, dim, hc, T)) return;
#endif
    s->parallel_for(sycl::nd_range<3>(sycl::range(1, T, cdiv(dim, kThreads)) *
                                          sycl::range(1, 1, kThreads),
                                      sycl::range(1, 1, kThreads)),
                    [=](sycl::nd_item<3> item_ct1) {
                        hc_post_inplace_kernel(y, h, post, comb, dim, hc);
                    });
}
void hc_mix_scale(const float *h, int hcdim, int n_mix, int T, float eps,
                  float *mixes, dpct::queue_ptr s) {
#if defined(STRATA_METAL)
    if (strata::dsv41::metal_host::hc_mix_scale(h, hcdim, n_mix, T, eps, mixes)) return;
#endif
    s->parallel_for(
        sycl::nd_range<3>(sycl::range(1, 1, T) * sycl::range(1, 1, kThreads),
                          sycl::range(1, 1, kThreads)),
        [=](sycl::nd_item<3> item_ct1) [[sycl::reqd_sub_group_size(32)]] {
            hc_mix_scale_kernel(h, hcdim, n_mix, eps, mixes);
        });
}
void engram_dequant_tok(const uint8_t *src, size_t per, int cols, int T, int hd,
                        float *out, dpct::queue_ptr s, int fmt) {
#if defined(STRATA_METAL)
    if (strata::dsv41::metal_host::engram_dequant_tok(src, per, cols, T, hd, out, fmt)) return;
#endif
    if (T > 0) {
        dpct::has_capability_or_fail(s->get_device(), {sycl::aspect::fp16});

        s->parallel_for(sycl::nd_range<3>(sycl::range(1, 1, T * cols) *
                                              sycl::range(1, 1, kThreads),
                                          sycl::range(1, 1, kThreads)),
                        [=](sycl::nd_item<3> item_ct1) {
                            engram_dequant_tok_kernel(src, per, cols, hd, out,
                                                      fmt);
                        });
    }
}
void moe_sort(const int *idx, int T, int k, const int *vs, int n_vs,
              int *bounds, int *row_tok, int *pos, dpct::queue_ptr s) {
#if defined(STRATA_METAL)
    if (strata::dsv41::metal_host::moe_sort(idx, T, k, vs, n_vs, bounds, row_tok, pos)) return;
#endif
    s->parallel_for(
        sycl::nd_range<3>(sycl::range(1, 1, n_vs) * sycl::range(1, 1, kThreads),
                          sycl::range(1, 1, kThreads)),
        [=](sycl::nd_item<3> item_ct1) [[sycl::reqd_sub_group_size(32)]] {
            moe_sort_kernel<false>(idx, T, k, vs, bounds, row_tok, pos);
        });
    s->parallel_for(
        sycl::nd_range<3>(sycl::range(1, 1, 32), sycl::range(1, 1, 32)),
        [=](sycl::nd_item<3> item_ct1) {
            bounds_scan_kernel(bounds, n_vs);
        });
    s->parallel_for(
        sycl::nd_range<3>(sycl::range(1, 1, n_vs) * sycl::range(1, 1, kThreads),
                          sycl::range(1, 1, kThreads)),
        [=](sycl::nd_item<3> item_ct1) [[sycl::reqd_sub_group_size(32)]] {
            moe_sort_kernel<true>(idx, T, k, vs, bounds, row_tok, pos);
        });
}
void moe_combine(float *acc, const float *y, const int *pos, const float *w,
                 int T, int k, int r0, int r1, int dim, dpct::queue_ptr s) {
    if (r1 > r0 && T > 0) {
#if defined(STRATA_METAL)
        if (strata::dsv41::metal_host::moe_combine(acc, y, pos, w, T, k, r0, r1, dim)) return;
#endif
        s->parallel_for(
            sycl::nd_range<3>(sycl::range(1, T, cdiv(dim, kThreads)) *
                                  sycl::range(1, 1, kThreads),
                              sycl::range(1, 1, kThreads)),
            [=](sycl::nd_item<3> item_ct1) {
                moe_combine_kernel(acc, y, pos, w, k, r0, r1, dim);
            });
    }
}

void gemv_f16(const uint16_t *W, const float *x, int n, int k, int T,
              float alpha, float *y, dpct::queue_ptr s) {
    dpct::has_capability_or_fail(s->get_device(), {sycl::aspect::fp16});

    s->parallel_for(
        sycl::nd_range<3>(sycl::range(1, T, cdiv(n, kThreads / 32)) *
                              sycl::range(1, 1, kThreads),
                          sycl::range(1, 1, kThreads)),
        [=](sycl::nd_item<3> item_ct1) [[sycl::reqd_sub_group_size(32)]] {
            gemv_f16_kernel(W, x, n, k, alpha, y);
        });
}
void gemv_f16_rows(const uint16_t *W, const float *x, size_t ldx, int n, int k,
                   int T, float alpha, float *y, size_t ldy,
                   dpct::queue_ptr s) {
#if defined(STRATA_METAL)
    // gemv_f16_rows_kernel on the Metal dispatch; the launch below is the fallback
    if (strata::dsv41::metal_host::gemv_f16_rows(W, x, ldx, n, k, T, alpha, y, ldy)) return;
#endif
    if (T > 0)
    {
        dpct::has_capability_or_fail(s->get_device(), {sycl::aspect::fp16});

        s->parallel_for(
            sycl::nd_range<3>(
                sycl::range(1, cdiv(T, kRowsT), cdiv(n, kThreads / 32)) *
                    sycl::range(1, 1, kThreads),
                sycl::range(1, 1, kThreads)),
            [=](sycl::nd_item<3> item_ct1) [[sycl::reqd_sub_group_size(32)]] {
                gemv_f16_rows_kernel(W, x, ldx, n, k, T, alpha, y, ldy);
            });
    }
}

void gemv_f32(const float *W, const float *x, int n, int k, int T, float *y,
              int ldy, dpct::queue_ptr s) {
#if defined(STRATA_METAL)
    if (T > 0 && ::strata::dsv41::metal_host::gemv_f32(W, x, n, k, T, y, ldy)) return;
#endif
        if (T > 0) s->parallel_for(
            sycl::nd_range<3>(sycl::range(1, T, cdiv(n, kThreads / 32)) *
                                  sycl::range(1, 1, kThreads),
                              sycl::range(1, 1, kThreads)),
            [=](sycl::nd_item<3> item_ct1) [[sycl::reqd_sub_group_size(32)]] {
                gemv_f32_kernel(W, x, n, k, y, ldy);
            });
}

void route(const float *logits, const float *bias, int n, int k, int T,
           float route_scale, int *idx, float *w, dpct::queue_ptr s,
           const float *bias_vl, const VlBits *vl) {
    static const VlBits none = {};
#if defined(STRATA_METAL)
    // route_kernel on the Metal dispatch; the launch below is the fallback
    if (strata::dsv41::metal_host::route(logits, bias, n, k, T, route_scale, idx, w, bias_vl, vl)) return;
#endif
    /*
    DPCT1049:201: The work-group size passed to the SYCL kernel may exceed the
    limit. To get the device limit, query info::device::max_work_group_size.
    Adjust the work-group size if needed.
    */
    {
        g_router_bad.init(*s);

        s->submit([&](sycl::handler &cgh) {
            auto g_router_bad_ptr_ct1 = g_router_bad.get_ptr();

            auto vl_nullptr_bias_vl_nullptr_ct7 =
                vl != nullptr ? bias_vl : nullptr;
            const VlBits vl_nullptr_vl_none_ct8 = vl != nullptr ? *vl : none;

            cgh.parallel_for(
                sycl::nd_range<3>(sycl::range(1, 1, T) * sycl::range(1, 1, 512),
                                  sycl::range(1, 1, 512)),
                [=](sycl::nd_item<3> item_ct1)
                    [[sycl::reqd_sub_group_size(32)]] {
                        route_kernel(logits, bias, n, k, route_scale, idx, w,
                                     vl_nullptr_bias_vl_nullptr_ct7,
                                     vl_nullptr_vl_none_ct8,
                                     *g_router_bad_ptr_ct1);
                    });
        });
    }
}

void swiglu(const float *g, const float *u, int n, float lim, float *h,
            dpct::queue_ptr s) {
#if defined(STRATA_METAL)
    // declared here rather than in the header: the header carries the other entry points, this one is only used here
    namespace mh = strata::dsv41::metal_host;
    if (n > 0 && mh::swiglu(g, u, n, lim, h)) return;
#endif
    if (n > 0)
        s->parallel_for(sycl::nd_range<3>(sycl::range(1, 1, cdiv(n, kThreads)) *
                                              sycl::range(1, 1, kThreads),
                                          sycl::range(1, 1, kThreads)),
                        [=](sycl::nd_item<3> item_ct1) {
                            swiglu_kernel(g, u, n, lim, h);
                        });
}

void compress_win(const float *kv, const float *score, float *kv_state,
                  float *score_state, int ratio, int p0, int T, int d,
                  const float *w, float eps, float *latent, dpct::queue_ptr s) {
#if defined(STRATA_METAL)
    // compress_kernel on the Metal dispatch; the launch below is the fallback
    if (strata::dsv41::metal_host::compress_win(kv, score, kv_state, score_state, ratio, p0, T, d, w, eps, latent)) return;
#endif
    /*
    DPCT1049:202: The work-group size passed to the SYCL kernel may exceed the
    limit. To get the device limit, query info::device::max_work_group_size.
    Adjust the work-group size if needed.
    */
    s->parallel_for(
        sycl::nd_range<3>(sycl::range(1, 1, 512), sycl::range(1, 1, 512)),
        [=](sycl::nd_item<3> item_ct1) [[sycl::reqd_sub_group_size(32)]] {
            compress_kernel(kv, score, kv_state, score_state, ratio, p0, T, d,
                            w, eps, latent);
        });
}

void index_score(const float *q, const float *w, const void *k, int heads,
                 int hd, int T, int n, const int *cand, int cand_stride,
                 int blk, const Rows &lim, float *out, long long out_stride,
                 dpct::queue_ptr s, int kfmt) {
    if (hd != 128 || heads > 32 || (kfmt != kKvFp4p && kfmt != kKvBf16)) {
        std::fprintf(stderr, "ds4 index_score: %d heads of %d (keys %d) not covered\n", heads, hd, kfmt);
        return;
    }
    if (n <= 0) return;
#if defined(STRATA_METAL)
    // index_score_kernel<128, kfmt> on the Metal dispatch; the launches below are the fallback
    if (strata::dsv41::metal_host::index_score(q, w, (const uint8_t*) k, heads, hd, T, n, cand, cand_stride, blk, &lim,
                                               out, out_stride, kfmt))
        return;
#endif
    /*
    DPCT1083:203: The size of local memory in the migrated code may be different
    from the original code. Check that the allocated memory size in the migrated
    code is correct.
    */
    const size_t smem = (size_t)(heads * hd + heads) * sizeof(float);
    const dpct::dim3 grid(cdiv(n, kThreads), T);
    if (kfmt == kKvFp4p)
        s->submit([&](sycl::handler &cgh) {
            sycl::local_accessor<uint8_t, 1> dpct_local_acc_ct1(
                sycl::range(smem), cgh);

            cgh.parallel_for(
                sycl::nd_range<3>(grid * sycl::range(1, 1, kThreads),
                                  sycl::range(1, 1, kThreads)),
                [=](sycl::nd_item<3> item_ct1) {
                    index_score_kernel<128, kKvFp4p>(
                        q, w, (const uint8_t *)k, heads, n, cand, cand_stride,
                        blk, lim, out, out_stride,
                        dpct_local_acc_ct1
                            .get_multi_ptr<sycl::access::decorated::no>()
                            .get());
                });
        });
    else
        s->submit([&](sycl::handler &cgh) {
            sycl::local_accessor<uint8_t, 1> dpct_local_acc_ct1(
                sycl::range(smem), cgh);

            cgh.parallel_for(
                sycl::nd_range<3>(grid * sycl::range(1, 1, kThreads),
                                  sycl::range(1, 1, kThreads)),
                [=](sycl::nd_item<3> item_ct1) {
                    index_score_kernel<128, kKvBf16>(
                        q, w, (const uint8_t *)k, heads, n, cand, cand_stride,
                        blk, lim, out, out_stride,
                        dpct_local_acc_ct1
                            .get_multi_ptr<sycl::access::decorated::no>()
                            .get());
                });
        });
}

void topk(const float *vals, long long stride, int T, const Rows &r,
          const int *cand, int cand_stride, int blk, int *out, int out_stride,
          dpct::queue_ptr s) {
#if defined(STRATA_METAL)
    // topk_kernel on the Metal dispatch; the launch below is the fallback
    if (strata::dsv41::metal_host::topk(vals, stride, T, &r, cand, cand_stride, blk, out, out_stride)) return;
#endif
    /*
    DPCT1049:204: The work-group size passed to the SYCL kernel may exceed the
    limit. To get the device limit, query info::device::max_work_group_size.
    Adjust the work-group size if needed.
    */
    s->parallel_for(
        sycl::nd_range<3>(sycl::range(1, 1, T) * sycl::range(1, 1, 1024),
                          sycl::range(1, 1, 1024)),
        [=](sycl::nd_item<3> item_ct1) [[sycl::reqd_sub_group_size(32)]] {
            topk_kernel(vals, stride, r, cand, cand_stride, blk, out,
                        out_stride);
        });
}

void cand_blocks(const float *score, long long stride, int T, const Rows &r,
                 int blk, float *bm, long long bm_stride, int *out,
                 int out_stride, dpct::queue_ptr s) {
    int nb_max = 1;
    Rows rb = r;
    if (r.gen != 0) {   // (a chunk: the widest query's blocks, made on the device per query)
        rb.gen = 4;
        rb.blk = blk;
        nb_max = cdiv((int) ((r.p0 + T) / r.R), blk);
    } else
        for (int t = 0; t < T; ++t) {
            rb.n[t] = cdiv(r.n[t], blk);
            rb.k[t] = r.k[t];
            nb_max = rb.n[t] > nb_max ? rb.n[t] : nb_max;
        }
#if defined(STRATA_METAL)
    // block_max_kernel on the Metal dispatch; the launch below is the fallback.  The selection of the best blocks (topk) follows either way:
    // returning here skipped it, the candidate list stayed unwritten and every context past 16K tokens (4,096 compressed rows) read garbage.
    if (!strata::dsv41::metal_host::block_max(score, stride, T, &r, blk, bm, bm_stride, nb_max))
#endif
        s->parallel_for(
            sycl::nd_range<3>(sycl::range(1, T, cdiv(nb_max, kThreads)) *
                                  sycl::range(1, 1, kThreads),
                              sycl::range(1, 1, kThreads)),
            [=](sycl::nd_item<3> item_ct1) {
                block_max_kernel(score, stride, r, blk, bm, bm_stride);
            });
    topk(bm, bm_stride, T, rb, nullptr, 0, 1, out, out_stride, s);
}

// the multiprocessors of the current device (cached a device)
static int sm_count() try {
    static std::atomic<int> cached[64];
    int dev = 0;
    if (DPCT_CHECK_ERROR(dev = dpct::get_current_device_id()) != 0 || dev < 0 ||
        dev >= 64) dev = 0;
    int v = cached[dev].load(std::memory_order_relaxed);
    if (v <= 0) {
        if (DPCT_CHECK_ERROR(
                v = dpct::get_device(dev).get_max_compute_units()) != 0 ||
            v <= 0) v = 1;
        cached[dev].store(v, std::memory_order_relaxed);
    }
    return v;
}
catch (sycl::exception const &exc) {
  std::cerr << exc.what() << "Exception caught at file:" << __FILE__
            << ", line:" << __LINE__ << std::endl;
  std::exit(1);
}

void sparse_attn(const float *q, const uint16_t *win, const void *ckv,
                 const int *sel, int sel_stride, const AttnRows &a,
                 const float *sink, int heads, int d, int T, float scale,
                 float *out, dpct::queue_ptr s, int ckv_fmt,
                 int heads_per_block, const AttnScratch *scr) {
    if (d != 512 || (ckv_fmt != kKvFp4 && ckv_fmt != kKvBf16)) {
        std::fprintf(stderr, "ds4 sparse_attn: head dim %d (rows %d) not covered\n", d, ckv_fmt);
        return;
    }
#if defined(STRATA_METAL)
    // sparse_attn_packed_kernel<G> / sparse_attn_kernel<false> on the Metal dispatch; the launches below are the fallback
    if (strata::dsv41::metal_host::sparse_attn(q, win, ckv, sel, sel_stride, &a, sink, heads, d, T, scale, out, ckv_fmt))
        return;
#endif
    if (ckv_fmt == kKvBf16) {
        s->parallel_for(
            sycl::nd_range<3>(sycl::range(1, T, heads) *
                                  sycl::range(1, 1, kThreads),
                              sycl::range(1, 1, kThreads)),
            [=](sycl::nd_item<3> item_ct1) [[sycl::reqd_sub_group_size(32)]] {
                sparse_attn_kernel<false>(q, win, (const uint16_t *)ckv, sel,
                                          sel_stride, a, sink, heads, d, scale,
                                          out);
            });
        return;
    }
    // a decode step / short verify window of packed rows: the split-KV kernel (Strata's CUDA lab, ds4_attn_split.cuh; 2.9x at
    // one query on an A4000).  The same inputs; the merge sums the chunks in another order, so the bits differ in the last
    // place (4e-7 against a double reference).  DS4_ATTN_SPLIT=1 turns it on (off by default: no measurable decode gain on the R9700 box, the decode is bound elsewhere).
    if (ckv_fmt == kKvFp4 && a.gen == 0 && T >= 1 && T <= kSplitMaxT && heads % kSplitG == 0 && split_enabled()) {
        int total_max = 0;
        for (int t = 0; t < T; ++t) total_max = std::max(total_max, a.nwin[t] + a.nsel[t]);
        if (total_max > 0 && total_max <= kSplitCap * kSplitR) {
            const attnsplit::AttnSplitScratch sc = split_scratch(s, heads);
            if (sc.part != nullptr) {
                attnsplit::sparse_attn_split<kSplitG, kSplitR>(q, win, (const uint8_t*) ckv, kv_row_bytes(kKvFp4, d), sel,
                                                               sel_stride, a, sink, heads, T, scale, out, sc, total_max, s);
                return;
            }
        }
    }
    // a few queries (a decode step, a short window): their selected rows decoded once a query into the scratch, then
    // the BF16 rows' kernel itself (heads_per_block -1: this way at any T)
    const bool few = heads_per_block == -1 || (heads_per_block == 0 && T <= kAttnScratchT);
    if (few && scr != nullptr && scr->rows != nullptr && T <= scr->max_t && sel_stride <= scr->cap) {
#if defined(STRATA_METAL)
        if (strata::dsv41::metal_host::unpack_sel(ckv, sel, sel_stride, a.gen, a.p0, a.t0, a.R, a.kcap, a.nsel,
                scr->cap, scr->rows, T, a.gen ? 1 : (T < kWin ? T : kWin), kv_row_bytes(kKvFp4, 512))) return;
#endif
        {
            dpct::has_capability_or_fail(s->get_device(), {sycl::aspect::fp16});

            s->submit([&](sycl::handler &cgh) {
                auto scr_cap_ct4 = scr->cap;
                auto scr_rows_ct5 = scr->rows;

                cgh.parallel_for(
                    sycl::nd_range<3>(
                        sycl::range(1, T, cdiv(sel_stride, kStage)) *
                            sycl::range(1, 1, kThreads),
                        sycl::range(1, 1, kThreads)),
                    [=](sycl::nd_item<3> item_ct1) {
                        unpack_sel_kernel((const uint8_t *)ckv, sel, sel_stride,
                                          a, scr_cap_ct4, scr_rows_ct5);
                    });
            });
        }
#if !defined(STRATA_USE_HIP)
        if (T == 1)   // (V100: the loads ahead pay for the decode's kernel at one query, not at more)
            s->submit([&](sycl::handler &cgh) {
                const uint16_t *scr_rows_ct2 = scr->rows;
                auto scr_cap_ct4 = scr->cap;

                cgh.parallel_for(
                    sycl::nd_range<3>(sycl::range(1, T, heads) *
                                          sycl::range(1, 1, kThreads),
                                      sycl::range(1, 1, kThreads)),
                    [=](sycl::nd_item<3> item_ct1)
                        [[sycl::reqd_sub_group_size(32)]] {
                            sparse_attn_kernel<true, 8>(
                                q, win, scr_rows_ct2, nullptr, scr_cap_ct4, a,
                                sink, heads, d, scale, out);
                        });
            });
        else
#endif
            s->submit([&](sycl::handler &cgh) {
                const uint16_t *scr_rows_ct2 = scr->rows;
                auto scr_cap_ct4 = scr->cap;

                cgh.parallel_for(
                    sycl::nd_range<3>(sycl::range(1, T, heads) *
                                          sycl::range(1, 1, kThreads),
                                      sycl::range(1, 1, kThreads)),
                    [=](sycl::nd_item<3> item_ct1)
                        [[sycl::reqd_sub_group_size(32)]] {
                            sparse_attn_kernel<true, 0>(
                                q, win, scr_rows_ct2, nullptr, scr_cap_ct4, a,
                                sink, heads, d, scale, out);
                        });
            });
        return;
    }
    // DS4_FAST_ATTN=1: the chunked prompt's matrix-core path (16 heads a block, a 16-key tile).  The decode and the
    // verify windows keep the scalar kernels: one query's heads are not the matrix's rows there.
#if defined(STRATA_USE_HIP)
    if (fast_attn_enabled() && a.gen != 0 && T >= kFaKeys && heads % kFaHeads == 0) {
        sparse_attn_wmma_kernel<<<dim3(heads / kFaHeads, T), kThreads, 0, s>>>(q, win, (const uint8_t*) ckv, sel, sel_stride, a, sink, heads, scale, out);
        return;
    }
#endif
    // packed rows: as many heads a block as still leaves two blocks a multiprocessor (a decoded row serves them all)
    int G = heads_per_block;    if (G != 1 && G != 2 && G != 4 && G != 8) {
        G = 8;
        while (G > 1 && (heads % G != 0 || (heads / G) * T < 2 * sm_count())) G /= 2;
    }
    if (heads % G != 0) G = 1;
    const dpct::dim3 grid(heads / G, T);
    const uint8_t* c = (const uint8_t*) ckv;
    if (G == 8) {
        dpct::has_capability_or_fail(s->get_device(), {sycl::aspect::fp16});

        s->parallel_for(
            sycl::nd_range<3>(grid * sycl::range(1, 1, kThreads),
                              sycl::range(1, 1, kThreads)),
            [=](sycl::nd_item<3> item_ct1) [[sycl::reqd_sub_group_size(32)]] {
                sparse_attn_packed_kernel<8>(q, win, c, sel, sel_stride, a,
                                             sink, heads, scale, out);
            });
    } else if (G == 4) {
        dpct::has_capability_or_fail(s->get_device(), {sycl::aspect::fp16});

        s->parallel_for(
            sycl::nd_range<3>(grid * sycl::range(1, 1, kThreads),
                              sycl::range(1, 1, kThreads)),
            [=](sycl::nd_item<3> item_ct1) [[sycl::reqd_sub_group_size(32)]] {
                sparse_attn_packed_kernel<4>(q, win, c, sel, sel_stride, a,
                                             sink, heads, scale, out);
            });
    } else if (G == 2) {
        dpct::has_capability_or_fail(s->get_device(), {sycl::aspect::fp16});

        s->parallel_for(
            sycl::nd_range<3>(grid * sycl::range(1, 1, kThreads),
                              sycl::range(1, 1, kThreads)),
            [=](sycl::nd_item<3> item_ct1) [[sycl::reqd_sub_group_size(32)]] {
                sparse_attn_packed_kernel<2>(q, win, c, sel, sel_stride, a,
                                             sink, heads, scale, out);
            });
    } else {
        dpct::has_capability_or_fail(s->get_device(), {sycl::aspect::fp16});

        s->parallel_for(
            sycl::nd_range<3>(grid * sycl::range(1, 1, kThreads),
                              sycl::range(1, 1, kThreads)),
            [=](sycl::nd_item<3> item_ct1) [[sycl::reqd_sub_group_size(32)]] {
                sparse_attn_packed_kernel<1>(q, win, c, sel, sel_stride, a,
                                             sink, heads, scale, out);
            });
    }
}

void ring_to_ext(const uint8_t *ring, uint16_t *ext, int64_t p0, int n, int d,
                 dpct::queue_ptr s) {
#if defined(STRATA_METAL)
    if (strata::dsv41::metal_host::ring_to_ext(ring, ext, (long long) p0, n, d, kv_row_bytes(kKvFp8, d))) return;
#endif
    dpct::has_capability_or_fail(s->get_device(), {sycl::aspect::fp16});

    s->parallel_for(
        sycl::nd_range<3>(sycl::range(1, 1, n) * sycl::range(1, 1, kThreads),
                          sycl::range(1, 1, kThreads)),
        [=](sycl::nd_item<3> item_ct1) {
            ring_to_ext_kernel(ring, ext, (long long)p0, n, d);
        });
}
void ext_to_ring(const uint16_t *ext, uint8_t *ring, int64_t p0, int n, int T,
                 int d, dpct::queue_ptr s, uint8_t *backup) {
    if (d % 32 != 0) {
        std::fprintf(stderr, "ds4 ext_to_ring: row of %d not covered\n", d);
        return;
    }
#if defined(STRATA_METAL)
    if (strata::dsv41::metal_host::ext_to_ring(ext, ring, (long long) p0, n, T, d, kv_row_bytes(kKvFp8, d), backup)) return;
#endif
    s->parallel_for(
        sycl::nd_range<3>(sycl::range(1, 1, T) * sycl::range(1, 1, kThreads),
                          sycl::range(1, 1, kThreads)),
        [=](sycl::nd_item<3> item_ct1) [[sycl::reqd_sub_group_size(32)]] {
            ext_to_ring_kernel(ext, ring, (long long)p0, n, T, d, backup);
        });
}
void ring_restore(uint8_t *ring, const uint8_t *backup, int64_t p0, int n,
                  int keep, int T, int d, dpct::queue_ptr s) {
#if defined(STRATA_METAL)
    {   // (row copies on the context queue: the drafter's rollback is a few rows)
        const size_t rb = kv_row_bytes(kKvFp8, d);
        for (int t = keep; t < T; ++t) cudaMemcpyAsync(ring + (size_t) ((p0 + t) % n) * rb, backup + (size_t) t * rb, rb, cudaMemcpyDeviceToDevice, s);
        return;
    }
#endif
    if (T > keep)
        s->submit([&](sycl::handler &cgh) {
            auto kv_row_bytes_kKvFp8_d_ct5 = kv_row_bytes(kKvFp8, d);

            cgh.parallel_for(sycl::nd_range<3>(sycl::range(1, 1, T - keep) *
                                                   sycl::range(1, 1, kThreads),
                                               sycl::range(1, 1, kThreads)),
                             [=](sycl::nd_item<3> item_ct1) {
                                 ring_restore_kernel(ring, backup,
                                                     (long long)p0, n, keep,
                                                     kv_row_bytes_kKvFp8_d_ct5);
                             });
        });
}
void ring_restore_batch(const RingBatch &b, int64_t p0, int n, int keep, int T,
                        int d, dpct::queue_ptr s) {
#if defined(STRATA_METAL)
    {
        const size_t rb = kv_row_bytes(kKvFp8, d);
        for (int li = 0; li < b.n; ++li)
            for (int t = keep; t < T; ++t)
                cudaMemcpyAsync(b.ring[li] + (size_t) ((p0 + t) % n) * rb, b.bak[li] + (size_t) t * rb, rb, cudaMemcpyDeviceToDevice, s);
        return;
    }
#endif
    if (T > keep && b.n > 0)
        s->submit([&](sycl::handler &cgh) {
            auto kv_row_bytes_kKvFp8_d_ct4 = kv_row_bytes(kKvFp8, d);

            cgh.parallel_for(sycl::nd_range<3>(sycl::range(1, b.n, T - keep) *
                                                   sycl::range(1, 1, kThreads),
                                               sycl::range(1, 1, kThreads)),
                             [=](sycl::nd_item<3> item_ct1) {
                                 ring_restore_batch_kernel(
                                     b, (long long)p0, n, keep,
                                     kv_row_bytes_kKvFp8_d_ct4);
                             });
        });
}
void hc_mean(const float *h, int T, int hc, int dim, float *out,
             long long out_stride, dpct::queue_ptr s) {
#if defined(STRATA_METAL)
    if (ds4_dbg("DS4_SKIP_HCMEAN")) return;
#endif
#if defined(STRATA_METAL)
    // hc_mean_kernel on the Metal dispatch; the launch below is the fallback
    if (strata::dsv41::metal_host::hc_mean(h, T, hc, dim, out, out_stride)) return;
#endif
    s->parallel_for(sycl::nd_range<3>(sycl::range(1, T, cdiv(dim, kThreads)) *
                                          sycl::range(1, 1, kThreads),
                                      sycl::range(1, 1, kThreads)),
                    [=](sycl::nd_item<3> item_ct1) {
                        hc_mean_kernel(h, hc, dim, out, out_stride);
                    });
}
void iota_rows(int *out, int T, int n, int stride, dpct::queue_ptr s) {
#if defined(STRATA_METAL)
    // iota_rows_kernel on the Metal dispatch; the launch below is the fallback
    if (strata::dsv41::metal_host::iota_rows(out, T, n, stride)) return;
#endif
    s->parallel_for(
        sycl::nd_range<3>(sycl::range(1, 1, T) * sycl::range(1, 1, 64),
                          sycl::range(1, 1, 64)),
        [=](sycl::nd_item<3> item_ct1) {
            iota_rows_kernel(out, n, stride);
        });
}
void markov_add(float *logits, const uint16_t *E, const uint16_t *H,
                const int *tok, int V, int R, dpct::queue_ptr s) {
#if defined(STRATA_METAL)
    // markov_kernel on the Metal dispatch (the drafter's Markov bias); the launch below is the fallback
    if (strata::dsv41::metal_host::markov_add(logits, E, H, tok, V, R)) return;
#endif
    dpct::has_capability_or_fail(s->get_device(), {sycl::aspect::fp16});

    s->submit([&](sycl::handler &cgh) {
        /*
        DPCT1083:918: The size of local memory in the migrated code may be
        different from the original code. Check that the allocated memory size
        in the migrated code is correct.
        */
        sycl::local_accessor<uint8_t, 1> dpct_local_acc_ct1(
            sycl::range((size_t)R * sizeof(float)), cgh);

        cgh.parallel_for(
            sycl::nd_range<3>(sycl::range(1, 1, cdiv(V, kThreads / 32)) *
                                  sycl::range(1, 1, kThreads),
                              sycl::range(1, 1, kThreads)),
            [=](sycl::nd_item<3> item_ct1) [[sycl::reqd_sub_group_size(32)]] {
                markov_kernel(logits, E, H, tok, V, R,
                              dpct_local_acc_ct1
                                  .get_multi_ptr<sycl::access::decorated::no>()
                                  .get());
            });
    });
}
void markov_add_q8(float *logits, const uint16_t *E, const uint8_t *Hq,
                   const int *tok, int V, int R, dpct::queue_ptr s) {
    dpct::has_capability_or_fail(s->get_device(), {sycl::aspect::fp16});

    s->submit([&](sycl::handler &cgh) {
        /*
        DPCT1083:919: The size of local memory in the migrated code may be
        different from the original code. Check that the allocated memory size
        in the migrated code is correct.
        */
        sycl::local_accessor<uint8_t, 1> dpct_local_acc_ct1(
            sycl::range((size_t)R * sizeof(float)), cgh);

        cgh.parallel_for(
            sycl::nd_range<3>(sycl::range(1, 1, cdiv(V, kThreads / 32)) *
                                  sycl::range(1, 1, kThreads),
                              sycl::range(1, 1, kThreads)),
            [=](sycl::nd_item<3> item_ct1) [[sycl::reqd_sub_group_size(32)]] {
                markov_kernel_q8(
                    logits, E, Hq, tok, V, R,
                    dpct_local_acc_ct1
                        .get_multi_ptr<sycl::access::decorated::no>()
                        .get());
            });
    });
}
void markov_share(float *logits, const uint16_t *H, const uint16_t *erow,
                  int vn, int R, dpct::queue_ptr s) {
    dpct::has_capability_or_fail(s->get_device(), {sycl::aspect::fp16});

    s->submit([&](sycl::handler &cgh) {
        /*
        DPCT1083:920: The size of local memory in the migrated code may be
        different from the original code. Check that the allocated memory size
        in the migrated code is correct.
        */
        sycl::local_accessor<uint8_t, 1> dpct_local_acc_ct1(
            sycl::range((size_t)R * sizeof(float)), cgh);

        cgh.parallel_for(
            sycl::nd_range<3>(sycl::range(1, 1, cdiv(vn, kThreads / 32)) *
                                  sycl::range(1, 1, kThreads),
                              sycl::range(1, 1, kThreads)),
            [=](sycl::nd_item<3> item_ct1) [[sycl::reqd_sub_group_size(32)]] {
                markov_share_kernel(
                    logits, H, erow, vn, R,
                    dpct_local_acc_ct1
                        .get_multi_ptr<sycl::access::decorated::no>()
                        .get());
            });
    });
}
void argmax_share(const float *row, int vn, int v0, unsigned *slot,
                  unsigned seq, dpct::queue_ptr s) {
    /*
    DPCT1049:205: The work-group size passed to the SYCL kernel may exceed the
    limit. To get the device limit, query info::device::max_work_group_size.
    Adjust the work-group size if needed.
    */
    s->parallel_for(
        sycl::nd_range<3>(sycl::range(1, 1, 1024), sycl::range(1, 1, 1024)),
        [=](sycl::nd_item<3> item_ct1) [[sycl::reqd_sub_group_size(32)]] {
            argmax_share_kernel(row, vn, v0, slot, seq);
        });
}
void markov_pick(const unsigned *slots, int nm, unsigned seq, int *tok,
                 const uint16_t *E, uint16_t *erow, int R, dpct::queue_ptr s) {
    s->parallel_for(
        sycl::nd_range<3>(sycl::range(1, 1, 256), sycl::range(1, 1, 256)),
        [=](sycl::nd_item<3> item_ct1) {
            markov_pick_kernel(slots, nm, seq, tok, E, erow, R);
        });
}
void confidence(const float *x, const uint16_t *E, const int *tok,
                const float *w, int T, int dim, int R, float *out,
                dpct::queue_ptr s) {
#if defined(STRATA_METAL)
    // confidence_kernel on the Metal dispatch; the launch below is the fallback
    if (strata::dsv41::metal_host::confidence(x, E, tok, w, T, dim, R, out)) return;
#endif
    /*
    DPCT1049:206: The work-group size passed to the SYCL kernel may exceed the
    limit. To get the device limit, query info::device::max_work_group_size.
    Adjust the work-group size if needed.
    */
    s->parallel_for(
        sycl::nd_range<3>(sycl::range(1, 1, T) * sycl::range(1, 1, 1024),
                          sycl::range(1, 1, 1024)),
        [=](sycl::nd_item<3> item_ct1) [[sycl::reqd_sub_group_size(32)]] {
            confidence_kernel(x, E, tok, w, dim, R, out);
        });
}

void engram_dequant(const uint8_t *rows, const uint8_t *scales, int n, int hd,
                    float *out, dpct::queue_ptr s, int fmt) {
#if defined(STRATA_METAL)
    // engram_dequant_kernel on the Metal dispatch (fmt 2, this model's IQ4_NL tables); the launch below is the fallback
    if (strata::dsv41::metal_host::engram_dequant(rows, scales, n, hd, out, fmt)) return;
#endif
    dpct::has_capability_or_fail(s->get_device(), {sycl::aspect::fp16});

    s->parallel_for(
        sycl::nd_range<3>(sycl::range(1, 1, n) * sycl::range(1, 1, kThreads),
                          sycl::range(1, 1, kThreads)),
        [=](sycl::nd_item<3> item_ct1) {
            engram_dequant_kernel(rows, scales, hd, out, fmt);
        });
}

void engram_gate(float *h, const float *kv, const float *w, int hc, int dim,
                 int T, float eps, dpct::queue_ptr s) {
#if defined(STRATA_METAL)
    // engram_gate_kernel on the Metal dispatch; the launch below is the fallback
    if (strata::dsv41::metal_host::engram_gate(h, kv, w, hc, dim, T, eps)) return;
#endif
    /*
    DPCT1049:207: The work-group size passed to the SYCL kernel may exceed the
    limit. To get the device limit, query info::device::max_work_group_size.
    Adjust the work-group size if needed.
    */
    s->parallel_for(
        sycl::nd_range<3>(sycl::range(1, T, hc) * sycl::range(1, 1, 1024),
                          sycl::range(1, 1, 1024)),
        [=](sycl::nd_item<3> item_ct1) [[sycl::reqd_sub_group_size(32)]] {
            engram_gate_kernel(h, kv, w, hc, dim, eps);
        });
}

void gather_rows(const float *x, const Gather &g, int n, int cols, float *out,
                 dpct::queue_ptr s) {
#if defined(STRATA_METAL)
    // gather_kernel on the Metal dispatch; the launch below is the fallback
    if (strata::dsv41::metal_host::gather_rows(x, &g, n, cols, out)) return;
#endif
        if (n > 0) s->parallel_for(
            sycl::nd_range<3>(sycl::range(1, n, cdiv(cols, kThreads)) *
                                  sycl::range(1, 1, kThreads),
                              sycl::range(1, 1, kThreads)),
            [=](sycl::nd_item<3> item_ct1) {
                gather_kernel(x, g, cols, out);
            });
}

void combine(const float *ed, const float *cpu, const float *sh,
             const Combine &c, int T, int dim, float *out, dpct::queue_ptr s) {
#if defined(STRATA_METAL)
    // combine_kernel on the Metal dispatch; the launch below is the fallback
    if (strata::dsv41::metal_host::combine(ed, cpu, sh, &c, T, dim, out)) return;
#endif
    s->parallel_for(sycl::nd_range<3>(sycl::range(1, T, cdiv(dim, kThreads)) *
                                          sycl::range(1, 1, kThreads),
                                      sycl::range(1, 1, kThreads)),
                    [=](sycl::nd_item<3> item_ct1) {
                        combine_kernel(ed, cpu, sh, c, dim, out);
                    });
}

void xfer_put(const Xfer &x, unsigned *flag, unsigned seq, unsigned *cnt,
              dpct::queue_ptr s) {
    size_t total = 0;
    for (int i = 0; i < x.n; ++i) total += x.bytes[i] / 16;
    const int blocks = (int) std::max<size_t>(1, std::min<size_t>(64, (total + kThreads - 1) / kThreads));
    s->parallel_for(sycl::nd_range<3>(sycl::range(1, 1, blocks) *
                                          sycl::range(1, 1, kThreads),
                                      sycl::range(1, 1, kThreads)),
                    [=](sycl::nd_item<3> item_ct1) {
                        xfer_put_kernel(x, flag, seq, cnt);
                    });
    // (another GPU, or the host, waits for this flag: on Windows the put must not sit in the queue - #303: two RTX 3080s
    //  hung at the first hand-off, the receiver spinning on a flag whose writer had never been submitted)
    win_submit(s);
}
void win_submit(dpct::queue_ptr s) {
#if defined(_WIN32)
    static const bool on = [] {
        const char* v = std::getenv("DS4_WIN_SUBMIT");
        return v == nullptr || std::atoi(v) != 0;
    }();
    if (on) {
        (void) cudaStreamQuery(s);
        (void) cudaGetLastError();
    }
#else
    (void) s;
#endif
}
void xfer_get(const Xfer &x, const unsigned *flag, unsigned seq,
              dpct::queue_ptr s) {
    size_t total = 0;
    for (int i = 0; i < x.n; ++i) total += x.bytes[i] / 16;
    const int blocks = (int) std::max<size_t>(1, std::min<size_t>(64, (total + kThreads - 1) / kThreads));
    s->parallel_for(sycl::nd_range<3>(sycl::range(1, 1, blocks) *
                                          sycl::range(1, 1, kThreads),
                                      sycl::range(1, 1, kThreads)),
                    [=](sycl::nd_item<3> item_ct1) {
                        xfer_get_kernel(x, flag, seq);
                    });
}
void wait_flags(const unsigned *flags, int n, int stride, unsigned seq,
                dpct::queue_ptr s) {
        if (n > 0) s->parallel_for(
            sycl::nd_range<3>(sycl::range(1, 1, 32), sycl::range(1, 1, 32)),
            [=](sycl::nd_item<3> item_ct1) {
                wait_flags_kernel(flags, n, stride, seq);
            });
}
void tab_set(const unsigned long long *pairs, int n, dpct::queue_ptr s) {
    if (n > 0)
        s->parallel_for(sycl::nd_range<3>(sycl::range(1, 1, (n + 127) / 128) *
                                              sycl::range(1, 1, 128),
                                          sycl::range(1, 1, 128)),
                        [=](sycl::nd_item<3> item_ct1) {
                            tab_set_kernel(pairs, n);
                        });
}
void set_flag(unsigned *flag, unsigned v, dpct::queue_ptr s) {
    s->parallel_for(
        sycl::nd_range<3>(sycl::range(1, 1, 1), sycl::range(1, 1, 1)),
        [=](sycl::nd_item<3> item_ct1) {
            set_flag_kernel(flag, v);
        });
    win_submit(s);   // (the writer on another GPU waits for this acknowledgement before its next put - #384)
}
void allreduce(float *x, int n, const ArGroup &a, int mi, unsigned seq,
               unsigned *cnt, dpct::queue_ptr s) {
    const int slot = (int) (seq % (unsigned) a.slots);
    if (a.nm > 2 && a.sums != nullptr && a.bflags != nullptr) {   // (the reduce-scatter: ar_rs_kernel)
        static const bool f16 = [] {
            const char* e = std::getenv("DS4_AR_F16");
            return e != nullptr && std::atoi(e) != 0;   // opt-in: no gain measured on 4x R9700 (21.8 vs 21.9 tok/s)   // (on unless DS4_AR_F16=0)
        }();
        const int n4 = n / 4;
        const int blocks = std::max(1, std::min(kArBlocks, cdiv(n4, kThreads)));
        if (f16)
        {
            dpct::has_capability_or_fail(s->get_device(), {sycl::aspect::fp16});

            s->parallel_for(sycl::nd_range<3>(sycl::range(1, 1, blocks) *
                                                  sycl::range(1, 1, kThreads),
                                              sycl::range(1, 1, kThreads)),
                            [=](sycl::nd_item<3> item_ct1) {
                                ar_rs16_kernel(x, n4, a.data, a.sums, a.stride,
                                               a.bflags, a.nm, mi, slot,
                                               a.slots, seq);
                            });
        } else
            s->parallel_for(sycl::nd_range<3>(sycl::range(1, 1, blocks) *
                                                  sycl::range(1, 1, kThreads),
                                              sycl::range(1, 1, kThreads)),
                            [=](sycl::nd_item<3> item_ct1) {
                                ar_rs_kernel(x, n4, a.data, a.sums, a.stride,
                                             a.bflags, a.nm, mi, slot, a.slots,
                                             seq);
                            });
        return;
    }
    Xfer xp;
    xp.n = 1;
    xp.src[0] = x;
    xp.dst[0] = a.data + ((size_t) mi * a.slots + slot) * a.stride;
    xp.bytes[0] = (size_t) n * 4;
    xfer_put(xp, a.flags + ((size_t) mi * a.slots + slot) * 32, seq, cnt, s);
    const int n4 = n / 4;
    const int blocks = std::max(1, std::min(64, cdiv(n4, kThreads)));
    s->parallel_for(sycl::nd_range<3>(sycl::range(1, 1, blocks) *
                                          sycl::range(1, 1, kThreads),
                                      sycl::range(1, 1, kThreads)),
                    [=](sycl::nd_item<3> item_ct1) {
                        ar_sum_kernel(x, n4, a.data, a.stride, a.flags, a.nm,
                                      mi, slot, a.slots, seq);
                    });
}

}  // namespace strata::ds4
