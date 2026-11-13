// Copyright (c) 2026 Peasant Smith and the Project Maya contributors (MIT; https://github.com/mw00/project-maya, v1.1.17).
// Ported into Strata (third_party/project-maya/NOTICE.md, LICENSE); changes are in the git history of this file.
// src/ds4/ds4_numerics.cuh - the BF16 and FP8 (E4M3) conversions of maya-ds4's kernels, on CUDA through its
// intrinsics (cuda_bf16.h / cuda_fp8.h) and on HIP in plain arithmetic that gives the same bits: BF16 round to
// nearest even; E4M3 (OCP "fn": no infinities, max 448) decoded exactly, rounded to nearest even and saturated.
#pragma once

#define DPCT_PROFILING_ENABLED
#include <sycl/sycl.hpp>
#include <dpct/dpct.hpp>
#include <cstdint>
#if !defined(STRATA_USE_HIP)
#endif

namespace strata::ds4 {

/// F32 -> BF16 bits, round to nearest even (a NaN stays a NaN).
__dpct_inline__ uint16_t bf16_rn(float f) {
    // the bits by arithmetic on CUDA too, not __bfloat16_as_ushort(__float2bfloat16_rn()): under -std=c++20 with CUDA 12.1 the
    // bfloat16 conversion gave wrong bits and the vision tower came out zero (Project Maya #676, rilken); the same bits
    // in C++17 and C++20, and it needs no CUDA 12.2 intrinsic (#336)
    const uint32_t u = sycl::bit_cast<unsigned int>(f);
    if ((u & 0x7fffffffu) > 0x7f800000u) return (uint16_t) ((u >> 16) | 0x40u);
    return (uint16_t) ((u + 0x7fffu + ((u >> 16) & 1u)) >> 16);
}

// OCP E4M3 decode and round-to-nearest-even, using the portable upstream
// arithmetic instead of CUDA-only conversion intrinsics. Preserve signed zero.
inline float e4m3_value(uint8_t q) {
    const uint32_t e = (q >> 3) & 15u, m = q & 7u;
    float v;
    if (e == 15u && m == 7u) v = sycl::bit_cast<float>(0x7fc00000u);
    else if (e == 0u) v = float(m) * 0.001953125f;
    else v = sycl::bit_cast<float>(((e + 120u) << 23) | (m << 20));
    return (q & 0x80u) ? -v : v;
}
inline float e4m3_round(float v) {
    const float a = sycl::fabs(v);
    if (sycl::isnan(a)) return v;
    float r;
    if (a >= 448.0f) r = 448.0f;
    else {
        const int e = int((sycl::bit_cast<uint32_t>(a) >> 23) & 0xffu) - 127;
        const float step = e < -6 ? 0.001953125f
            : sycl::bit_cast<float>(uint32_t(e - 3 + 127) << 23);
        r = a == 0.0f ? 0.0f : sycl::rint(a / step) * step;
        r = sycl::fmin(r, 448.0f);
    }
    return sycl::copysign(r, v);
}

}  // namespace strata::ds4
