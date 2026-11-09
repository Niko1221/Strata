// src/dsv41/ds4_q5k.cuh - one value of a ggml Q5_K superblock (256 values, 176 bytes: d F16, dmin F16, scales[12] (eight
// 6-bit scales and eight 6-bit mins), qh[32] the fifth bits, qs[128] the low 4 bits), as ggml's dequantize_row_q5_K
// computes it: (d * sc) * q - (dmin * m), each product and the difference rounded on their own in F32.  The Engram table of
// Unsloth's UD-Q2_K_XL DeepSeek-V4.1-Flash set is Q5_K (per_layer_token_embd, 135 GB).
#pragma once

#include <cuda_fp16.h>
#include <cstdint>

namespace strata::ds4 {

constexpr int kQ5kBytes = 176;

__device__ __forceinline__ float q5k_value(const uint8_t* __restrict__ b, int i) {
    // (the F16 scales' bits as halves: HIP's headers take no brace-initialized __half_raw from them)
    const float d = __half2float(__ushort_as_half((unsigned short) (b[0] | (b[1] << 8))));
    const float dmin = __half2float(__ushort_as_half((unsigned short) (b[2] | (b[3] << 8))));
    const uint8_t* sc = b + 4;
    const uint8_t* qh = b + 16;
    const uint8_t* qs = b + 48;
    // ggml's loop: 64-value groups j, each two 32-value halves (the low nibbles of qs[32 j ..], then the high ones);
    // half `is` = 2 j + hi has scale/min number is and its fifth bit at bit `is` of qh[l]
    const int j = i >> 6, hi = (i >> 5) & 1, l = i & 31, is = 2 * j + hi;
    int s, m;   // (ggml's get_scale_min_k4)
    if (is < 4) {
        s = sc[is] & 63;
        m = sc[is + 4] & 63;
    } else {
        s = (sc[is + 4] & 0xF) | ((sc[is - 4] >> 6) << 4);
        m = (sc[is + 4] >> 4) | ((sc[is] >> 6) << 4);
    }
    const int lo = qs[32 * j + l];
    const int q = (hi ? lo >> 4 : lo & 0xF) + (((qh[l] >> is) & 1) << 4);
    return __fsub_rn(__fmul_rn(__fmul_rn(d, (float) s), (float) q), __fmul_rn(dmin, (float) m));
}

}  // namespace strata::ds4
