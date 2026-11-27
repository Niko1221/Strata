// ink_dflash2_mix.h - host decoder of the pre-quantized DFlash 2 file ("strata-dflash2-q1", variant "mix", the file Strata downloads next to the model).
// MIT License - Copyright (c) 2026 Strata contributors
//
// The mix file keeps the five layers' attention matrices (q_proj, o_proj) as int8 with one float32 scale per row
// (tensors NAME.q I8 [N*K], NAME.s F32 [N]) and the feed-forward matrices (gate / up / down) as SF4:
// E2M1 values, two per byte (first value in the low nibble), one UE4M3 scale per 16 values, one float32 outer scale
// (NAME.q U8 [N*K/2], NAME.s U8 [N*K/16], NAME.o F32 [1]); value = e2m1 * ue4m3 * outer. Everything else is as in the original model.
// The backends without an SF4 drafter kernel (Metal, SYCL) widen the SF4 matrices here and quantize them to their row-scaled int8 form.
#pragma once

#include <cmath>
#include <cstdint>
#include <cstring>

namespace ink2 {
namespace df2mix {

// E4M3 (finite, bias 7; the scale byte is unsigned, bit 7 is 0)
inline float e4m3_to_f32(uint8_t c) {
    const int e = (c >> 3) & 15;
    const int m = c & 7;
    if (e == 0) return std::ldexp((float)m, -9);                 // subnormal: m * 2^-9
    return std::ldexp(1.0f + (float)m * 0.125f, e - 7);
}

inline float e2m1_to_f32(uint8_t c) {
    static const float kV[8] = {0.0f, 0.5f, 1.0f, 1.5f, 2.0f, 3.0f, 4.0f, 6.0f};
    const float v = kV[c & 7];
    return (c & 8) ? -v : v;
}

inline uint16_t f32_to_bf16_rne(float f) {
    uint32_t u;
    std::memcpy(&u, &f, 4);
    return (uint16_t)((u + 0x7FFFu + ((u >> 16) & 1u)) >> 16);
}

// q: N*K/2 bytes, s: N*K/16 bytes, outer: the tensor's float32 outer scale. out: N*K bf16 values.
inline void sf4_to_bf16(const uint8_t* q, const uint8_t* s, float outer, int N, int K, uint16_t* out) {
    const size_t nb = (size_t)K / 16;
    for (int r = 0; r < N; ++r) {
        for (size_t b = 0; b < nb; ++b) {
            const float sc = e4m3_to_f32(s[(size_t)r * nb + b]) * outer;
            const uint8_t* qp = q + ((size_t)r * K + b * 16) / 2;
            uint16_t* op = out + (size_t)r * K + b * 16;
            for (int i = 0; i < 8; ++i) {
                op[2 * i]     = f32_to_bf16_rne(e2m1_to_f32(qp[i] & 15) * sc);
                op[2 * i + 1] = f32_to_bf16_rne(e2m1_to_f32(qp[i] >> 4) * sc);
            }
        }
    }
}

}  // namespace df2mix
}  // namespace ink2
