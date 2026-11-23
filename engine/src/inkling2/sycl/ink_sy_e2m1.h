// ink_sy_e2m1.h - E2M1 (FP4, the NVFP4 / SF4 element) code to float for the Intel Arc SYCL Inkling kernels.
// Header only and free of SYCL types, so a plain C++20 compiler can run the exactness test (tests/core/sy_e2m1_test.cpp).
#pragma once

#include <cstdint>

namespace ink_sycl {

// The reference: the 16-entry table (0, 0.5, 1, 1.5, 2, 3, 4, 6, then the same with the sign bit; code 8 is -0.0).
inline float e2m1_table(uint32_t nib) {
    constexpr float kLut[16] = {
        0.0f,  0.5f,  1.0f,  1.5f,  2.0f,  3.0f,  4.0f,  6.0f,
       -0.0f, -0.5f, -1.0f, -1.5f, -2.0f, -3.0f, -4.0f, -6.0f
    };
    return kLut[nib & 15u];
}

#ifdef INK_SY_E2M1_TABLE   // A/B builds only: the table, as before
inline float e2m1_lut(uint32_t nib) { return e2m1_table(nib); }
#else
// The same values from integer ops. A table indexed by a run-time value lives in per-thread private memory on Intel GPUs;
// this form is plain ALU work. Bit for bit equal to e2m1_table for all 16 codes.
inline float e2m1_lut(uint32_t nib) {
    const uint32_t m = nib & 7u, e = m >> 1, h = m & 1u;
    const uint32_t norm = ((e + 126u) << 23) | (h << 22);   // (1 + h/2) * 2^(e-1) for e >= 1
    const uint32_t sub = h ? 0x3F000000u : 0u;               // e == 0: 0 or 0.5
    return __builtin_bit_cast(float, ((e == 0u) ? sub : norm) | ((nib & 8u) << 28));
}
#endif

}  // namespace ink_sycl
