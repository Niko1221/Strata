// src/kernels/cpu/nvfp4_neon.cpp - see include/strata/kernels/cpu/nvfp4_avx2.hpp.
//
// ARM NEON NVFP4 expert rows, several tokens at once, for aarch64 (NVIDIA GB10).  The AVX-2 kernel
// (nvfp4_avx2.cpp) is x86-only; this file provides the same three symbols on ARM.
//
// It replicates ggml-cpu's ARM dot product (ggml-cpu/arch/arm/quants.c ggml_vec_dot_nvfp4_q8_0)
// bit for bit: per 64-value block, four 16-value sub-block dots (each = sum over 8 packed codes of
// the signed doubled E2M1 value of the low nibble times q8_0[j], plus the high nibble's value times
// q8_0[j+8]), then one fused multiply-add applies ue4m3(d[s]) * d_q8 per sub-block, and a horizontal
// add sums the four lanes at the end.  Each token's arithmetic is that dot product's NEON path step
// for step, so a row equals what ggml-cpu computes on this CPU - what an AVX-2 CPU computed before
// these existed.  Activations are Q8_0.
//
// Needs only baseline AArch64 NEON (vmull_s8 / vpadalq_s16 / vqtbl1q_s8); no dotprod, i8mm or FP16
// beyond the __fp16 scale conversion ggml's ARM path already uses.
#include "strata/kernels/cpu/nvfp4_avx2.hpp"

#define GGML_COMMON_DECL_CPP
#define GGML_COMMON_IMPL_CPP
#include "ggml-common.h"

#include <arm_neon.h>

#include <cmath>
#include <cstdlib>
#include <cstring>

namespace strata::kernels::cpu {
namespace {

// A compile-time table, not a constructor: this file is compiled for a portable ISA and a
// namespace-scope object with a constructor runs at program start on every CPU (#2324).
// Powers of two: exact.
constexpr float pow2i(int e) {
    float r = 1.0f;
    for (; e > 0; --e) r *= 2.0f;
    for (; e < 0; ++e) r *= 0.5f;
    return r;
}
struct Ue4m3Table {   // ggml_ue4m3_to_fp32 (ggml-impl.h): halved for the doubled kvalues_fp4, 0x7F read as 0
    float v[256];
    constexpr Ue4m3Table() : v() {
        for (int x = 0; x < 256; ++x) {
            if (x == 0 || x == 0x7F) { v[x] = 0.0f; continue; }
            const int e = (x >> 3) & 0xF, m = x & 0x7;
            const float raw = e == 0 ? (float) m * pow2i(-9) : (1.0f + (float) m / 8.0f) * pow2i(e - 7);
            v[x] = raw * 0.5f;
        }
    }
};
constexpr Ue4m3Table kUe4m3;

inline float h2f(uint16_t h) {   // ggml's ARM fp16 -> fp32, same rounding as its neon_compute_fp16_to_fp32
    __fp16 f16;
    std::memcpy(&f16, &h, 2);
    return (float) f16;
}

// software prefetch distance in bytes (STRATA_NVFP4_PREFETCH, 0 = off), as the AVX-2 rows
inline int prefetch_distance() {
    static const int v = [] {
        const char* e = std::getenv("STRATA_NVFP4_PREFETCH");
        return e ? std::atoi(e) : 2048;
    }();
    return v;
}

constexpr int kMaxBlocks = 64;   // rows of up to 4096 values (n_embd 2560)
// tokens per pass: the pool's MAXT, so a window is one pass.  Spilled accumulators cost less than
// another pass over the weights (see nvfp4_avx2.cpp).
constexpr int kMaxW = 8;

struct Acts {   // per token and block, the two Q8_0 blocks' scales; built once per call, reused by every row
    int nt = 0, nb = 0;
    const block_q8_0* y[kMaxW] = {};
    float dy[kMaxW * kMaxBlocks * 2];
    Acts(const void* const* act, int nt_, int nb_) : nt(nt_), nb(nb_) {
        for (int t = 0; t < nt; ++t) {
            y[t] = (const block_q8_0*) act[t];
            for (int ib = 0; ib < 2 * nb; ++ib) dy[(size_t) t * 2 * nb + ib] = h2f(y[t][ib].d);
        }
    }
};

// 8 packed codes -> the 16 signed doubled E2M1 values they hold (low nibble, then high), as two
// int8x8 lanes.  kvalues_fp4 is the signed table {0,1,2,3,4,6,8,12,0,-1,-2,-3,-4,-6,-8,-12}.
inline void decode8(const uint8_t* codes, int8x8_t& lo, int8x8_t& hi) {
    const int8x16_t tbl = vld1q_s8(kvalues_fp4);
    const uint8x8_t cb = vld1_u8(codes);
    // one table look-up for both halves: vqtbl1q_s8 wants a 16-lane index, so the low nibbles and the high nibbles
    // go into one vector (vtbl1_s8 could not: its table is 8 bytes, kvalues_fp4 is 16).
    const uint8x16_t idx = vcombine_u8(vand_u8(cb, vdup_n_u8(0x0F)), vshr_n_u8(cb, 4));
    const int8x16_t v = vqtbl1q_s8(tbl, idx);
    lo = vget_low_s8(v);
    hi = vget_high_s8(v);
}

// 16-value sub-block dot: the signed E2M1 values times the 16 q8_0 values, summed exactly in int32
// (vmull_s8 + vpaddlq_s16 + vpadalq_s16, ggml's ARM path).
inline int32_t dot16(const int8x8_t& lo, const int8x8_t& hi, const int8_t* q) {
    const int8x16_t x = vld1q_s8(q);
    const int16x8_t pl = vmull_s8(lo, vget_low_s8(x));    // low-nibble values * q[j]
    const int16x8_t ph = vmull_s8(hi, vget_high_s8(x));   // high-nibble values * q[j+8]
    int32x4_t acc = vpaddlq_s16(pl);
    acc = vpadalq_s16(acc, ph);
    return vaddvq_s32(acc);
}

template <int NT>
inline void row_dot(const uint8_t* row, const Acts& a, float* res) {
    float32x4_t acc[NT];
    for (int t = 0; t < NT; ++t) acc[t] = vdupq_n_f32(0.0f);
    const block_nvfp4* x = (const block_nvfp4*) row;
    const int prefetch_ahead = prefetch_distance();
    for (int ib = 0; ib < a.nb; ++ib) {
        if (prefetch_ahead > 0) __builtin_prefetch((const char*) (x + ib) + prefetch_ahead);
        // decode the block's four 16-value sub-blocks' E2M1 codes once, reuse across tokens
        int8x8_t lo0, hi0, lo1, hi1, lo2, hi2, lo3, hi3;
        decode8(x[ib].qs + 0,  lo0, hi0);
        decode8(x[ib].qs + 8,  lo1, hi1);
        decode8(x[ib].qs + 16, lo2, hi2);
        decode8(x[ib].qs + 24, lo3, hi3);
        // sub-block scales: ue4m3(d[s]) (same for every token)
        const float w0 = kUe4m3.v[x[ib].d[0]];
        const float w1 = kUe4m3.v[x[ib].d[1]];
        const float w2 = kUe4m3.v[x[ib].d[2]];
        const float w3 = kUe4m3.v[x[ib].d[3]];
        for (int t = 0; t < NT; ++t) {
            const block_q8_0* yb = a.y[t] + 2 * ib;
            // sub-block s=0..3: codes qs[8s..8s+7], activations yb[s/2].qs[(s%2)*16 .. +16)
            const int32_t d0 = dot16(lo0, hi0, yb[0].qs + 0);
            const int32_t d1 = dot16(lo1, hi1, yb[0].qs + 16);
            const int32_t d2 = dot16(lo2, hi2, yb[1].qs + 0);
            const int32_t d3 = dot16(lo3, hi3, yb[1].qs + 16);
            const int32_t vals[4] = {d0, d1, d2, d3};
            const int32x4_t dots = vld1q_s32(vals);
            const float dy0 = a.dy[(size_t) t * 2 * a.nb + 2 * ib + 0];
            const float dy1 = a.dy[(size_t) t * 2 * a.nb + 2 * ib + 1];
            const float sc[4] = {w0 * dy0, w1 * dy0, w2 * dy1, w3 * dy1};
            const float32x4_t scales = vld1q_f32(sc);
            acc[t] = vfmaq_f32(acc[t], vcvtq_f32_s32(dots), scales);
        }
    }
    for (int t = 0; t < NT; ++t) res[t] = vaddvq_f32(acc[t]);
}

template <int NT>
void gu_rows(const uint8_t* blob, size_t gu_row, size_t up_off, const Acts& a, float* const* ff,
             int r0, int r1, float sg, float su) {
    float g[NT], u[NT];
    for (int r = r0; r < r1; ++r) {
        row_dot<NT>(blob + (size_t) r * gu_row, a, g);
        row_dot<NT>(blob + up_off + (size_t) r * gu_row, a, u);
        for (int t = 0; t < NT; ++t) {
            // native_gu_rows' ggml path, operation for operation: silu(sg * g) * (su * u)
            const float gs = sg * g[t], us = su * u[t];
            ff[t][r] = (gs / (1.0f + std::exp(-gs))) * us;
        }
    }
}

template <int NT>
void dot_rows(const uint8_t* w, size_t row_bytes, const Acts& a, float* const* out, int r0, int r1,
              float scale) {
    float res[NT];
    for (int r = r0; r < r1; ++r) {
        row_dot<NT>(w + (size_t) r * row_bytes, a, res);
        for (int t = 0; t < NT; ++t) out[t][r] = res[t] * scale;
    }
}

// windows wider than kMaxW (not from the pool) run in balanced slices of at most kMaxW
template <typename F>
inline void slices(int nt, F&& run) {
    const int k = (nt + kMaxW - 1) / kMaxW;
    for (int i = 0, t0 = 0; i < k; ++i) {
        const int w = nt / k + (i < nt % k ? 1 : 0);
        run(t0, w);
        t0 += w;
    }
}

}  // namespace

bool nvfp4_256_fits(int n) { return n % QK_NVFP4 == 0 && n / QK_NVFP4 <= kMaxBlocks; }

void nvfp4_256_gu_rows(const uint8_t* blob, size_t gu_row, size_t up_off, int n, const void* const* act,
                       int nt, float* const* ff, int r0, int r1, float s_gate, float s_up) {
    slices(nt, [&](int t0, int w) {
        const Acts a(act + t0, w, n / QK_NVFP4);
        float* const* f = ff + t0;
        switch (w) {
            case 1: gu_rows<1>(blob, gu_row, up_off, a, f, r0, r1, s_gate, s_up); break;
            case 2: gu_rows<2>(blob, gu_row, up_off, a, f, r0, r1, s_gate, s_up); break;
            case 3: gu_rows<3>(blob, gu_row, up_off, a, f, r0, r1, s_gate, s_up); break;
            case 4: gu_rows<4>(blob, gu_row, up_off, a, f, r0, r1, s_gate, s_up); break;
            case 5: gu_rows<5>(blob, gu_row, up_off, a, f, r0, r1, s_gate, s_up); break;
            case 6: gu_rows<6>(blob, gu_row, up_off, a, f, r0, r1, s_gate, s_up); break;
            case 7: gu_rows<7>(blob, gu_row, up_off, a, f, r0, r1, s_gate, s_up); break;
            default: gu_rows<8>(blob, gu_row, up_off, a, f, r0, r1, s_gate, s_up); break;
        }
    });
}

void nvfp4_256_rows(const uint8_t* w, size_t row_bytes, int n, const void* const* act, int nt,
                    float* const* out, int r0, int r1, float scale) {
    slices(nt, [&](int t0, int k) {
        const Acts a(act + t0, k, n / QK_NVFP4);
        float* const* o = out + t0;
        switch (k) {
            case 1: dot_rows<1>(w, row_bytes, a, o, r0, r1, scale); break;
            case 2: dot_rows<2>(w, row_bytes, a, o, r0, r1, scale); break;
            case 3: dot_rows<3>(w, row_bytes, a, o, r0, r1, scale); break;
            case 4: dot_rows<4>(w, row_bytes, a, o, r0, r1, scale); break;
            case 5: dot_rows<5>(w, row_bytes, a, o, r0, r1, scale); break;
            case 6: dot_rows<6>(w, row_bytes, a, o, r0, r1, scale); break;
            case 7: dot_rows<7>(w, row_bytes, a, o, r0, r1, scale); break;
            default: dot_rows<8>(w, row_bytes, a, o, r0, r1, scale); break;
        }
    });
}

}  // namespace strata::kernels::cpu
