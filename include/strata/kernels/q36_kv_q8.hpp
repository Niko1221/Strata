// include/strata/kernels/q36_kv_q8.hpp - Qwen3.6-35B-A3B int8 KV primitive (docs/Q36_INT8_KV.md).
//
// The Qwen3.6 engine keeps one flat fp32 K/V allocation per layer, indexed `[kvh * ctx + pos] * HD`
// (head_dim = 256, 2 KV heads). `--kv int8` stores each HD-length K/V vector as q8_0 - llama.cpp's
// block layout of 32 int8 codes plus one fp16 scale:
//     scale = fp16(max|x| / 127),  code = clamp(rint(x / scale), -127, 127),  x' = code * scale
// 256 values become 256 codes + 8 fp16 scales (264 B) instead of 1,024 B of fp32.
//
// The math lives here as shared STRATA_HD helpers so the write kernel (k_attn_prep), the decode read
// (k_attn_part), the prompt staging kernel (k_kv_f16) and the parity test all run the SAME arithmetic.
// This is the engine's own primitive: the QSA path's `kv_q8.hpp` uses a 64-value group and page-table
// indexing that does not transplant, so it is only the reference for the block-scale idea, not the code.
#pragma once

#include "strata/kernels/f16_bits.hpp"

#include <cmath>
#include <cstdint>

namespace strata::kernels {

inline constexpr int Q36_KV_Q8_GROUP = 32;

/// fp16 bits of the per-group scale for a group whose max |value| is `amax`.
STRATA_HD inline uint16_t q36_kv_q8_scale_bits(float amax) { return f16_from_f32(amax / 127.0f); }

/// Round-to-nearest-even to an int. `__float2int_rn` is device-only; the host reference in the parity test
/// uses `std::nearbyint`, which is the same IEEE ties-to-even rounding.
STRATA_HD inline int q36_kv_q8_round(float r) {
#if defined(__CUDA_ARCH__) || defined(__HIP_DEVICE_COMPILE__)
    return __float2int_rn(r);
#else
    return (int) std::nearbyint(r);
#endif
}

/// Quantize one value against the STORED scale (f32_from_f16(scale)), not the unrounded amax/127.  The
/// fp16 rounding can push |x|/scale a hair past 127, so clamp rather than wrap.  A zero scale (all-zero
/// group) stores zero.
STRATA_HD inline int8_t q36_kv_q8_quantize(float x, uint16_t sbits) {
    const float sf = f32_from_f16(sbits);
    if (sf <= 0.0f) return 0;
    int q = q36_kv_q8_round(x / sf);
    q = q < -127 ? -127 : (q > 127 ? 127 : q);
    return (int8_t) q;
}

/// Dequantize one code: `code * scale`, exact in fp32.
STRATA_HD inline float q36_kv_q8_dequantize(int8_t q, uint16_t sbits) {
    return (float) q * f32_from_f16(sbits);
}

/// Quantize one HD-length vector `x` into `hd` int8 codes and `hd / Q36_KV_Q8_GROUP` fp16 scales.
/// `hd` must be a multiple of Q36_KV_Q8_GROUP.  One block of `hd` threads; each 32-thread warp reduces
/// its own group's max |x|, so the group is exactly one warp and needs no shared memory.
void q36_kv_q8_quantize(int8_t* q, uint16_t* scale, const float* x, int hd, void* stream);

/// Dequantize `hd` codes into fp32 (the round-trip for the parity test and the decode path's reference).
void q36_kv_q8_dequantize(float* y, const int8_t* q, const uint16_t* scale, int hd, void* stream);

}  // namespace strata::kernels
