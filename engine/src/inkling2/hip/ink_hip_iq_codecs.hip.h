// engine/src/inkling2/hip/ink_hip_iq_codecs.hip.h
// GSQ-RCO expert types for the HIP backend (RDNA2 gfx103x and RDNA3 gfx110x; the code is wave-size and ISA neutral):
//   IQ2_XXS (16), IQ2_XS (17), IQ1_S (19), IQ3_S (21), IQ2_S (22), Q2_0 (42, Strata's own 64-value / 18-byte type).
// Same decode contract as hip_gguf_decode_8_f16 in ink_hip_codecs.hip.h: 8 consecutive weights from an 8-aligned column of one row, as
// fp32 values; every GGUF_RAW kernel (fdot2 linear, FP16 unpack for hipBLASLt, the grouped MoE kernels, the WMMA tiles) already goes through
// that one function, so these types need no kernels of their own.
// Block layouts follow the GGUF block specs (and strata::quant::dequant_row, checked to difference 0 by ink_gguf_quant_cross_test.cpp);
// the codebooks are the ones of the CUDA engine (cuda/ink_iq_grids.cuh, generated from strata-quant-common.h), kept in global memory.
// MIT License - Copyright (c) 2026 Niko1221 and the Strata contributors.
#pragma once

#include <cstdint>

#include "../cuda/ink_iq_grids.cuh"

namespace strata::inkling2::hip {

constexpr uint32_t kGgmlIq2Xxs = 16, kGgmlIq2Xs = 17, kGgmlIq1S = 19, kGgmlIq3S = 21, kGgmlIq2S = 22, kGgmlQ2_0 = 42;

__host__ __device__ inline bool gguf_type_is_hip_iq(uint32_t t) noexcept {
    return t == 16u || t == 17u || t == 19u || t == 21u || t == 22u || t == 42u;
}

// bytes of one row of K values (K a multiple of the block: 256, or 64 for Q2_0); 0 for another type
__host__ __device__ inline size_t hip_iq_row_bytes(uint32_t t, int K) noexcept {
    switch (t) {
        case 16: return static_cast<size_t>(K >> 8) * 66u;
        case 17: return static_cast<size_t>(K >> 8) * 74u;
        case 19: return static_cast<size_t>(K >> 8) * 50u;
        case 21: return static_cast<size_t>(K >> 8) * 110u;
        case 22: return static_cast<size_t>(K >> 8) * 82u;
        case 42: return static_cast<size_t>(K >> 6) * 18u;
        default: return 0u;
    }
}

namespace iq_detail {

__device__ __forceinline__ uint32_t ld_u16(const uint8_t* p) { return static_cast<uint32_t>(p[0]) | (static_cast<uint32_t>(p[1]) << 8); }
__device__ __forceinline__ uint32_t ld_u32(const uint8_t* p) { return ld_u16(p) | (ld_u16(p + 2) << 16); }
__device__ __forceinline__ float ld_f16(const uint8_t* p) {
    return static_cast<float>(__builtin_bit_cast(_Float16, static_cast<uint16_t>(ld_u16(p))));
}
// 8 magnitude bytes of a uint64 codebook entry, signed by the 8 bits of `sg` (bit j set = negative)
__device__ __forceinline__ void put8_u64(float* w, uint64_t g, uint32_t sg, float db) {
#pragma unroll
    for (int j = 0; j < 8; ++j) {
        const float v = static_cast<float>(static_cast<uint32_t>((g >> (8 * j)) & 255u)) * db;
        w[j] = ((sg >> j) & 1u) ? -v : v;
    }
}

}  // namespace iq_detail

// Decodes 8 weights (col % 8 == 0) of type t into w[0..7] (already multiplied by ext_scale). Returns false for a type that is not one of the six.
__device__ __forceinline__ bool hip_iq_decode_8(uint32_t t, const uint8_t* __restrict__ row, int col, float ext_scale, float* w) {
    using namespace iq_detail;
    if (t == 21u) {          // IQ3_S: 256 values = 110 B: d, qs[64], qh[8], signs[32], scales[4]
        const int b = col >> 8, idx = col & 255;
        const uint8_t* blk = row + static_cast<size_t>(b) * 110u;
        const float d = ld_f16(blk);
        const int ib = idx >> 5, l = (idx >> 3) & 3;
        const uint32_t sc = blk[106 + (ib >> 1)];
        const float db = d * static_cast<float>(1 + 2 * ((ib & 1) ? (sc >> 4) : (sc & 15u))) * ext_scale;
        const uint32_t qh = blk[66 + ib];
        const uint32_t g1 = strata::ink::kInkIq3SGrid[blk[2 + ib * 8 + 2 * l] | ((qh << (8 - 2 * l)) & 256u)];
        const uint32_t g2 = strata::ink::kInkIq3SGrid[blk[2 + ib * 8 + 2 * l + 1] | ((qh << (7 - 2 * l)) & 256u)];
        const uint32_t sg = blk[74 + ib * 4 + l];
#pragma unroll
        for (int j = 0; j < 4; ++j) {
            const float a = static_cast<float>((g1 >> (8 * j)) & 255u) * db;
            const float c = static_cast<float>((g2 >> (8 * j)) & 255u) * db;
            w[j]     = ((sg >> j) & 1u) ? -a : a;
            w[j + 4] = ((sg >> (j + 4)) & 1u) ? -c : c;
        }
    } else if (t == 16u) {   // IQ2_XXS: 256 values = 66 B: d, then per 32 values 8 B (4 grid bytes, then 28 bits of signs + 4 bits of scale)
        const int b = col >> 8, idx = col & 255;
        const uint8_t* blk = row + static_cast<size_t>(b) * 66u;
        const float d = ld_f16(blk);
        const int ib = idx >> 5, l = (idx >> 3) & 3;
        const uint32_t a0 = ld_u32(blk + 2 + 8 * ib), a1 = ld_u32(blk + 6 + 8 * ib);
        const float db = d * (0.5f + static_cast<float>(a1 >> 28)) * 0.25f * ext_scale;
        put8_u64(w, strata::ink::kInkIq2XxsGrid[(a0 >> (8 * l)) & 255u], strata::ink::kInkKsignsIq2[(a1 >> (7 * l)) & 127u], db);
    } else if (t == 17u) {   // IQ2_XS: 256 values = 74 B: d, qs[32] (u16: 9 bits grid + 7 bits signs), scales[8]
        const int b = col >> 8, idx = col & 255;
        const uint8_t* blk = row + static_cast<size_t>(b) * 74u;
        const float d = ld_f16(blk);
        const int ib = idx >> 5, l = (idx >> 3) & 3;
        const uint32_t qv = ld_u16(blk + 2 + 8 * ib + 2 * l);
        const uint32_t sc = blk[66 + ib];
        const float db = d * (0.5f + static_cast<float>((l < 2) ? (sc & 15u) : (sc >> 4))) * 0.25f * ext_scale;
        put8_u64(w, strata::ink::kInkIq2XsGrid[qv & 511u], strata::ink::kInkKsignsIq2[qv >> 9], db);
    } else if (t == 22u) {   // IQ2_S: 256 values = 82 B: d, qs[32] (grid low bits), signs[32], qh[8], scales[8]
        const int b = col >> 8, idx = col & 255;
        const uint8_t* blk = row + static_cast<size_t>(b) * 82u;
        const float d = ld_f16(blk);
        const int ib = idx >> 5, l = (idx >> 3) & 3;
        const uint32_t sc = blk[74 + ib];
        const float db = d * (0.5f + static_cast<float>((l < 2) ? (sc & 15u) : (sc >> 4))) * 0.25f * ext_scale;
        const uint32_t qh = blk[66 + ib];
        put8_u64(w, strata::ink::kInkIq2SGrid[blk[2 + ib * 4 + l] | ((qh << (8 - 2 * l)) & 0x300u)], blk[34 + ib * 4 + l], db);
    } else if (t == 19u) {   // IQ1_S: 256 values = 50 B: d, qs[32], qh[8] (u16: 3 x 3 high bits, 3 bits scale, sign of delta)
        const int b = col >> 8, idx = col & 255;
        const uint8_t* blk = row + static_cast<size_t>(b) * 50u;
        const float d = ld_f16(blk);
        const int ib = idx >> 5, l = (idx >> 3) & 3;
        const uint32_t qh = ld_u16(blk + 34 + 2 * ib);
        const float dl = d * static_cast<float>(2 * ((qh >> 12) & 7u) + 1) * ext_scale;
        const float delta = (qh & 0x8000u) ? -0.125f : 0.125f;
        const uint64_t g = strata::ink::kInkIq1SGrid[blk[2 + 4 * ib + l] | (((qh >> (3 * l)) & 7u) << 8)];
#pragma unroll
        for (int j = 0; j < 8; ++j) w[j] = dl * (static_cast<float>(static_cast<int8_t>((g >> (8 * j)) & 255u)) + delta);
    } else if (t == 42u) {   // Q2_0: 64 values = 18 B: d (fp16, either sign), 16 B of 2-bit codes, value = (q - 1) * d
        const int b = col >> 6, in64 = col & 63;
        const uint8_t* blk = row + static_cast<size_t>(b) * 18u;
        const float d = ld_f16(blk) * ext_scale;
        const uint32_t q = ld_u16(blk + 2 + (in64 >> 2));
#pragma unroll
        for (int j = 0; j < 8; ++j) w[j] = (static_cast<float>(static_cast<int>((q >> (2 * j)) & 3u)) - 1.0f) * d;
    } else {
        return false;
    }
    return true;
}

}  // namespace strata::inkling2::hip
