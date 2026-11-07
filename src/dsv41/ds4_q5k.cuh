// Strata. One value of a ggml Q5_K superblock (256 values, 176 bytes: a half d and a half dmin, scales[12] the 6-bit
// scales and mins packed, qh[32] the fifth bit of every value, qs[128] the low four), as ggml's dequantize_row_q5_K
// computes it: d * sc, then * q, then minus dmin * m, every product in F32 on its own - the same bits (the Engram
// tables a GGUF may hold in Q5_K).
#pragma once

#include <cuda_fp16.h>
#include <cstdint>

namespace strata::ds4 {

constexpr int kQ5kBytes = 176;

__device__ __forceinline__ float q5k_value(const uint8_t* __restrict__ b, int i) {
    const uint8_t* sc = b + 4;
    const uint8_t* qh = b + 16;
    const uint8_t* ql = b + 48;
    // (the F16 scales' bits as halves: HIP's headers take no brace-initialized __half_raw from them)
    const float d = __half2float(__ushort_as_half((unsigned short) (b[0] | (b[1] << 8))));
    const float mn = __half2float(__ushort_as_half((unsigned short) (b[2] | (b[3] << 8))));
    const int step = i >> 6, hi = (i >> 5) & 1, l = i & 31;   // (ggml's loop: 64 values a step, its two 32 halves)
    const int j = 2 * step + hi;                             // (the step's two scales: get_scale_min_k4's j)
    // ggml's get_scale_min_k4: the first four scales and mins in the low 6 bits of scales[j] and scales[j + 4], the
    // last four from 4 bits there and the top 2 of the opposite byte
    int s6, m6;
    if (j < 4) {
        s6 = sc[j] & 63;
        m6 = sc[j + 4] & 63;
    } else {
        s6 = (sc[j + 4] & 0xF) | ((sc[j - 4] >> 6) << 4);
        m6 = (sc[j + 4] >> 4) | ((sc[j] >> 6) << 4);
    }
    const int lo = ql[step * 32 + l];
    const int q = (hi ? lo >> 4 : lo & 0xF) + (((qh[l] >> j) & 1) << 4);
    return __fsub_rn(__fmul_rn(__fmul_rn(d, (float) s6), (float) q), __fmul_rn(mn, (float) m6));
}

}  // namespace strata::ds4
