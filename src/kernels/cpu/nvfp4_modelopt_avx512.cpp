// src/kernels/cpu/nvfp4_modelopt_avx512.cpp - AVX-512 VNNI kernels for ModelOpt NVFP4 CPU experts.
//
// Compiled with AVX-512F/BW/VL/DQ/VNNI + FMA + F16C. Reached only behind the runtime CPU feature check in
// `src/kernels/cpu/nvfp4_modelopt_pool.cpp`. All helpers and templates live in the anonymous namespace so no
// weak/COMDAT symbol can leak EVEX instructions into callers on AVX2-only CPUs.
#include "strata/kernels/cpu/nvfp4_modelopt.hpp"

#include <immintrin.h>

#include <cmath>
#include <cstddef>
#include <cstdint>
#include <cstring>

namespace strata::kernels::cpu {
namespace {

inline uint16_t f32_to_bf16_bits(float v) {
    uint32_t bits = 0;
    std::memcpy(&bits, &v, sizeof(bits));
    uint32_t lsb = (bits >> 16) & 1u;
    bits += 0x7FFFu + lsb;
    return static_cast<uint16_t>(bits >> 16);
}

inline float bf16_bits_to_f32(uint16_t b) {
    uint32_t bits = static_cast<uint32_t>(b) << 16;
    float v = 0.0f;
    std::memcpy(&v, &bits, sizeof(v));
    return v;
}

// Hardware Zen 4 AVX-512_BF16 round-to-nearest-even FP32 -> BF16 -> FP32 across 16 lanes
inline __m512 bf16_round16_avx512(__m512 x) {
    __m256bh bh = _mm512_cvtneps_pbh(x);
    return _mm512_castsi512_ps(_mm512_slli_epi32(_mm512_cvtepu16_epi32((__m256i)bh), 16));
}

// Vectorized 16-wide exp(-x) using Cody-Waite range reduction + degree-5 minimax polynomial for 2^f + _mm512_scalef_ps
inline __m512 exp_neg16_avx512(__m512 x) {
    const __m512 v_neg_log2e = _mm512_set1_ps(-1.4426950408889634f);
    const __m512 v_max_arg = _mm512_set1_ps(126.0f);
    const __m512 v_min_arg = _mm512_set1_ps(-126.0f);
    __m512 t = _mm512_mul_ps(x, v_neg_log2e);
    t = _mm512_min_ps(_mm512_max_ps(t, v_min_arg), v_max_arg);
    __m512 n = _mm512_roundscale_ps(t, _MM_FROUND_TO_NEAREST_INT | _MM_FROUND_NO_EXC);
    __m512 f = _mm512_sub_ps(t, n);
    // Degree-5 minimax polynomial for 2^f on [-0.5, 0.5]
    const __m512 c5 = _mm512_set1_ps(1.33335581e-3f);
    const __m512 c4 = _mm512_set1_ps(9.61812910e-3f);
    const __m512 c3 = _mm512_set1_ps(5.55041086e-2f);
    const __m512 c2 = _mm512_set1_ps(2.40226507e-1f);
    const __m512 c1 = _mm512_set1_ps(6.93147182e-1f);
    const __m512 c0 = _mm512_set1_ps(1.0f);
    __m512 p = _mm512_fmadd_ps(c5, f, c4);
    p = _mm512_fmadd_ps(p, f, c3);
    p = _mm512_fmadd_ps(p, f, c2);
    p = _mm512_fmadd_ps(p, f, c1);
    p = _mm512_fmadd_ps(p, f, c0);
    return _mm512_scalef_ps(p, n);
}

// Vectorized 16-wide SwiGLU with exact BF16 intermediate rounding matching GPU FlyDSL kernels
inline __m512 swiglu16_bf16_avx512(__m512 g_raw, __m512 u_raw, __m512 v_gs2, __m512 v_us2) {
    __m512 gf = bf16_round16_avx512(_mm512_mul_ps(g_raw, v_gs2));
    __m512 uf = bf16_round16_avx512(_mm512_mul_ps(u_raw, v_us2));
    __m512 denom = _mm512_add_ps(_mm512_set1_ps(1.0f), exp_neg16_avx512(gf));
    __m512 silu_g = bf16_round16_avx512(_mm512_div_ps(gf, denom));
    return bf16_round16_avx512(_mm512_mul_ps(silu_g, uf));
}

// Unpack 32 ModelOpt NVFP4 bytes (low nibble = 2*i, high nibble = 2*i+1) into 64 unsigned u8 codes (2*val + 12)
inline __m512i codes64_modelopt(const uint8_t* wp, __m512i lut, __m512i mask) {
    __m512i w16 = _mm512_cvtepu8_epi16(_mm256_loadu_si256(reinterpret_cast<const __m256i*>(wp)));
    __m512i nibbles = _mm512_and_si512(_mm512_or_si512(w16, _mm512_slli_epi16(w16, 4)), mask);
    return _mm512_shuffle_epi8(lut, nibbles);
}

// Quantize 32 SwiGLU intermediate floats into int8 [32], float [2] (0.5 * amax / 127), and int32 [8] (-12 * sum4(q))
inline void quantize_act_f32_32_avx512(const float* h32, int8_t* q_h, float* d_h, int32_t* neg_bias_h) {
    const __m512 sign_mask = _mm512_castsi512_ps(_mm512_set1_epi32(0x7FFFFFFF));
    for (int sub = 0; sub < 2; ++sub) {
        __m512 xv = _mm512_loadu_ps(h32 + 16 * sub);
        float amax = _mm512_reduce_max_ps(_mm512_and_ps(xv, sign_mask));
        if (amax == 0.0f) {
            _mm_storeu_si128(reinterpret_cast<__m128i*>(q_h + 16 * sub), _mm_setzero_si128());
            d_h[sub] = 0.0f;
        } else {
            __m512 inv = _mm512_set1_ps(127.0f / amax);
            __m512i q32 = _mm512_cvtps_epi32(_mm512_mul_ps(xv, inv));
            __m128i q8 = _mm512_cvtepi32_epi8(q32);
            _mm_storeu_si128(reinterpret_cast<__m128i*>(q_h + 16 * sub), q8);
            d_h[sub] = amax * (0.5f / 127.0f);
        }
    }
    const __m256i zero = _mm256_setzero_si256();
    const __m256i c12 = _mm256_set1_epi8(12);
    __m256i qv = _mm256_loadu_si256(reinterpret_cast<const __m256i*>(q_h));
    __m256i nb = _mm256_sub_epi32(zero, _mm256_dpbusd_epi32(zero, c12, qv));
    _mm256_storeu_si256(reinterpret_cast<__m256i*>(neg_bias_h), nb);
}

// Vectorized 8-byte float8_e4m3fn -> 8x float32 (__m256) decoder in pure SIMD registers (zero LUT memory loads)
inline __m256 decode_f8_e4m3_8(const uint8_t* sp) {
    __m128i raw8 = _mm_loadl_epi64(reinterpret_cast<const __m128i*>(sp));
    __m256i b32 = _mm256_cvtepu8_epi32(raw8);
    __m256i sign = _mm256_slli_epi32(_mm256_and_si256(b32, _mm256_set1_epi32(0x80)), 24);
    __m256i exp = _mm256_and_si256(_mm256_srli_epi32(b32, 3), _mm256_set1_epi32(15));
    __m256i man = _mm256_and_si256(b32, _mm256_set1_epi32(7));
    __m256i norm_i = _mm256_or_si256(_mm256_slli_epi32(_mm256_add_epi32(exp, _mm256_set1_epi32(120)), 23),
                                     _mm256_slli_epi32(man, 20));
    __m256 sub = _mm256_mul_ps(_mm256_cvtepi32_ps(man), _mm256_set1_ps(0.001953125f));
    __mmask8 is_sub = _mm256_cmpeq_epi32_mask(exp, _mm256_setzero_si256());
    __m256 mag = _mm256_mask_blend_ps(is_sub, _mm256_castsi256_ps(norm_i), sub);
    return _mm256_castsi256_ps(_mm256_or_si256(_mm256_castps_si256(mag), sign));
}

// Vectorized 16-byte float8_e4m3fn -> 16x float32 (__m512) decoder in pure SIMD registers
inline __m512 decode_f8_e4m3_16(const uint8_t* sp) {
    __m128i raw16 = _mm_loadu_si128(reinterpret_cast<const __m128i*>(sp));
    __m512i b32 = _mm512_cvtepu8_epi32(raw16);
    __m512i sign = _mm512_slli_epi32(_mm512_and_si512(b32, _mm512_set1_epi32(0x80)), 24);
    __m512i exp = _mm512_and_si512(_mm512_srli_epi32(b32, 3), _mm512_set1_epi32(15));
    __m512i man = _mm512_and_si512(b32, _mm512_set1_epi32(7));
    __m512i norm_i = _mm512_or_si512(_mm512_slli_epi32(_mm512_add_epi32(exp, _mm512_set1_epi32(120)), 23),
                                     _mm512_slli_epi32(man, 20));
    __m512 sub = _mm512_mul_ps(_mm512_cvtepi32_ps(man), _mm512_set1_ps(0.001953125f));
    __mmask16 is_sub = _mm512_cmpeq_epi32_mask(exp, _mm512_setzero_si512());
    __m512 mag = _mm512_mask_blend_ps(is_sub, _mm512_castsi512_ps(norm_i), sub);
    return _mm512_castsi512_ps(_mm512_or_si512(_mm512_castps_si512(mag), sign));
}

template <int NT, int NB>
inline void nvfp4_row_dot_avx512(const uint8_t* wp, const uint8_t* sp,
                                 const int8_t* const* q_x, const float* const* d_x,
                                 const int32_t* const* neg_bias_x, float* out_f32) {
    static_assert(NB % 8 == 0, "NB must be a multiple of 8");
    const __m512i lut = _mm512_broadcast_i32x4(_mm_setr_epi8(
        12, 13, 14, 15, 16, 18, 20, 24, 12, 11, 10, 9, 8, 6, 4, 0));
    const __m512i mask = _mm512_set1_epi8(0x0F);
    const __m512i expand0 = _mm512_set_epi32(3, 3, 3, 3, 2, 2, 2, 2, 1, 1, 1, 1, 0, 0, 0, 0);
    const __m512i expand1 = _mm512_set_epi32(7, 7, 7, 7, 6, 6, 6, 6, 5, 5, 5, 5, 4, 4, 4, 4);

    __m512 acc0[NT];
    __m512 acc1[NT];
    for (int t = 0; t < NT; ++t) {
        acc0[t] = _mm512_setzero_ps();
        acc1[t] = _mm512_setzero_ps();
    }

    for (int b = 0; b < NB; b += 8) {
        _mm_prefetch(reinterpret_cast<const char*>(wp + 8 * b + 1024), _MM_HINT_T0);
        __m512i c0 = codes64_modelopt(wp + 8 * b, lut, mask);
        __m512i c1 = codes64_modelopt(wp + 8 * b + 32, lut, mask);
        __m256 ws8 = decode_f8_e4m3_8(sp + b);
        for (int t = 0; t < NT; ++t) {
            __m512 sc8_512 = _mm512_castps256_ps512(_mm256_mul_ps(ws8, _mm256_loadu_ps(d_x[t] + b)));
            __m512i qv0 = _mm512_loadu_si512(reinterpret_cast<const __m512i*>(q_x[t] + 16 * b));
            __m512i nb0 = _mm512_loadu_si512(reinterpret_cast<const __m512i*>(neg_bias_x[t] + 4 * b));
            __m512i raw0 = _mm512_dpbusd_epi32(nb0, c0, qv0);
            __m512 sc16_0 = _mm512_permutexvar_ps(expand0, sc8_512);
            acc0[t] = _mm512_fmadd_ps(_mm512_cvtepi32_ps(raw0), sc16_0, acc0[t]);

            __m512i qv1 = _mm512_loadu_si512(reinterpret_cast<const __m512i*>(q_x[t] + 16 * b + 64));
            __m512i nb1 = _mm512_loadu_si512(reinterpret_cast<const __m512i*>(neg_bias_x[t] + 4 * b + 16));
            __m512i raw1 = _mm512_dpbusd_epi32(nb1, c1, qv1);
            __m512 sc16_1 = _mm512_permutexvar_ps(expand1, sc8_512);
            acc1[t] = _mm512_fmadd_ps(_mm512_cvtepi32_ps(raw1), sc16_1, acc1[t]);
        }
    }
    for (int t = 0; t < NT; ++t) {
        out_f32[t] = _mm512_reduce_add_ps(_mm512_add_ps(acc0[t], acc1[t]));
    }
}

template <int NT>
void gu_tile32_nt_avx512(CpuExpertJob& jb, int r0, int t0) {
    const uint8_t* gw_base = jb.exp_base + EXP_OFF_GATE_W + static_cast<size_t>(r0) * 1280;
    const uint8_t* gs_base = jb.exp_base + EXP_OFF_GATE_S + static_cast<size_t>(r0) * 160;
    const uint8_t* uw_base = jb.exp_base + EXP_OFF_UP_W + static_cast<size_t>(r0) * 1280;
    const uint8_t* us_base = jb.exp_base + EXP_OFF_UP_S + static_cast<size_t>(r0) * 160;

    alignas(64) float g_all[NT][32];
    for (int r = 0; r < 32; ++r) {
        float g_f32[NT];
        nvfp4_row_dot_avx512<NT, 160>(gw_base + static_cast<size_t>(r) * 1280, gs_base + static_cast<size_t>(r) * 160,
                                      jb.q_x + t0, jb.d_x + t0, jb.neg_bias_x + t0, g_f32);
        for (int t = 0; t < NT; ++t) {
            g_all[t][r] = g_f32[t];
        }
    }

    alignas(64) float u_all[NT][32];
    for (int r = 0; r < 32; ++r) {
        float u_f32[NT];
        nvfp4_row_dot_avx512<NT, 160>(uw_base + static_cast<size_t>(r) * 1280, us_base + static_cast<size_t>(r) * 160,
                                      jb.q_x + t0, jb.d_x + t0, jb.neg_bias_x + t0, u_f32);
        for (int t = 0; t < NT; ++t) {
            u_all[t][r] = u_f32[t];
        }
    }

    const __m512 v_gs2 = _mm512_set1_ps(jb.gate_s2);
    const __m512 v_us2 = _mm512_set1_ps(jb.up_s2);
    alignas(64) float h32[NT][32];
    int b0 = r0 / 16;
    for (int t = 0; t < NT; ++t) {
        __m512 h0 = swiglu16_bf16_avx512(_mm512_load_ps(g_all[t] + 0), _mm512_load_ps(u_all[t] + 0), v_gs2, v_us2);
        __m512 h1 = swiglu16_bf16_avx512(_mm512_load_ps(g_all[t] + 16), _mm512_load_ps(u_all[t] + 16), v_gs2, v_us2);
        _mm512_store_ps(h32[t] + 0, h0);
        _mm512_store_ps(h32[t] + 16, h1);
        quantize_act_f32_32_avx512(h32[t], jb.q_h[t0 + t] + r0, jb.d_h[t0 + t] + b0, jb.neg_bias_h[t0 + t] + 4 * b0);
    }
}

template <int NT>
void down_tile80_nt_avx512(CpuExpertJob& jb, int r0, int t0) {
    const uint8_t* dw_base = jb.exp_base + EXP_OFF_DOWN_W + static_cast<size_t>(r0) * 320;
    const uint8_t* ds_base = jb.exp_base + EXP_OFF_DOWN_S + static_cast<size_t>(r0) * 40;

    const int8_t* q_h_ptrs[NT];
    const float* d_h_ptrs[NT];
    const int32_t* nb_h_ptrs[NT];
    for (int t = 0; t < NT; ++t) {
        q_h_ptrs[t] = jb.q_h[t0 + t];
        d_h_ptrs[t] = jb.d_h[t0 + t];
        nb_h_ptrs[t] = jb.neg_bias_h[t0 + t];
    }

    alignas(64) float d_all[NT][80];
    for (int r = 0; r < 80; ++r) {
        float d_f32[NT];
        nvfp4_row_dot_avx512<NT, 40>(dw_base + static_cast<size_t>(r) * 320, ds_base + static_cast<size_t>(r) * 40,
                                     q_h_ptrs, d_h_ptrs, nb_h_ptrs, d_f32);
        for (int t = 0; t < NT; ++t) {
            d_all[t][r] = d_f32[t];
        }
    }
    const __m512 v_ds2 = _mm512_set1_ps(jb.down_s2);
    for (int t = 0; t < NT; ++t) {
        uint16_t* dst = jb.y_out[t0 + t] + r0;
        for (int r = 0; r < 80; r += 16) {
            __m512 dv = _mm512_mul_ps(_mm512_load_ps(d_all[t] + r), v_ds2);
            _mm256_storeu_si256(reinterpret_cast<__m256i*>(dst + r), (__m256i)_mm512_cvtneps_pbh(dv));
        }
    }
}

}  // namespace

void quantize_act_bf16_2560_avx512(const uint16_t* x_bf16, int8_t* q_x, float* d_x, int32_t* neg_bias_x) {
    const __m512 sign_mask = _mm512_castsi512_ps(_mm512_set1_epi32(0x7FFFFFFF));
    const __m512i zero = _mm512_setzero_si512();
    const __m512i c12 = _mm512_set1_epi8(12);
    for (int b4 = 0; b4 < 160; b4 += 4) {
        for (int s = 0; s < 4; ++s) {
            const int b = b4 + s;
            __m256i bf = _mm256_loadu_si256(reinterpret_cast<const __m256i*>(x_bf16 + 16 * b));
            __m512 xv = _mm512_castsi512_ps(_mm512_slli_epi32(_mm512_cvtepu16_epi32(bf), 16));
            float amax = _mm512_reduce_max_ps(_mm512_and_ps(xv, sign_mask));
            if (amax == 0.0f) {
                _mm_storeu_si128(reinterpret_cast<__m128i*>(q_x + 16 * b), _mm_setzero_si128());
                d_x[b] = 0.0f;
            } else {
                __m512 inv = _mm512_set1_ps(127.0f / amax);
                __m512i q32 = _mm512_cvtps_epi32(_mm512_mul_ps(xv, inv));
                __m128i q8 = _mm512_cvtepi32_epi8(q32);
                _mm_storeu_si128(reinterpret_cast<__m128i*>(q_x + 16 * b), q8);
                d_x[b] = amax * (0.5f / 127.0f);
            }
        }
        __m512i qv = _mm512_loadu_si512(reinterpret_cast<const __m512i*>(q_x + 16 * b4));
        __m512i nb = _mm512_sub_epi32(zero, _mm512_dpbusd_epi32(zero, c12, qv));
        _mm512_storeu_si512(reinterpret_cast<__m512i*>(neg_bias_x + 4 * b4), nb);
    }
}

void gu_tile32_modelopt_avx512(CpuExpertJob& jb, int r0) {
    for (int t0 = 0; t0 < jb.nt; t0 += 4) {
        int rem = jb.nt - t0;
        switch (rem) {
            case 1: gu_tile32_nt_avx512<1>(jb, r0, t0); break;
            case 2: gu_tile32_nt_avx512<2>(jb, r0, t0); break;
            case 3: gu_tile32_nt_avx512<3>(jb, r0, t0); break;
            default: gu_tile32_nt_avx512<4>(jb, r0, t0); break;
        }
    }
}

void down_tile80_modelopt_avx512(CpuExpertJob& jb, int r0) {
    for (int t0 = 0; t0 < jb.nt; t0 += 4) {
        int rem = jb.nt - t0;
        switch (rem) {
            case 1: down_tile80_nt_avx512<1>(jb, r0, t0); break;
            case 2: down_tile80_nt_avx512<2>(jb, r0, t0); break;
            case 3: down_tile80_nt_avx512<3>(jb, r0, t0); break;
            default: down_tile80_nt_avx512<4>(jb, r0, t0); break;
        }
    }
}

void down_tile80_decode_modelopt_avx512(const CpuExpertJob& jb, int r0) {
    const uint8_t* dw_base = jb.exp_base + EXP_OFF_DOWN_W + static_cast<size_t>(r0) * 320;
    const uint8_t* ds_base = jb.exp_base + EXP_OFF_DOWN_S + static_cast<size_t>(r0) * 40;
    const int8_t* q_h_ptr = jb.q_h[0];
    const float* d_h_ptr = jb.d_h[0];
    const int32_t* nb_h_ptr = jb.neg_bias_h[0];
    alignas(64) float d_raw[80];
    for (int r = 0; r < 80; ++r) {
        nvfp4_row_dot_avx512<1, 40>(dw_base + static_cast<size_t>(r) * 320,
                                    ds_base + static_cast<size_t>(r) * 40,
                                    &q_h_ptr, &d_h_ptr, &nb_h_ptr, &d_raw[r]);
    }
    const __m512 v_sd = _mm512_set1_ps(jb.down_s2);
    const __m512 v_w = _mm512_set1_ps(jb.weight);
    float* dst = jb.y_f32 + r0;
    for (int r = 0; r < 80; r += 16) {
        __m512 y_bf = bf16_round16_avx512(_mm512_mul_ps(_mm512_load_ps(d_raw + r), v_sd));
        _mm512_storeu_ps(dst + r, bf16_round16_avx512(_mm512_mul_ps(y_bf, v_w)));
    }
}

void gu_tile32_fp8_avx512(CpuExpertJob& jb, int r0) {
    static constexpr size_t kOffGateW = 0;
    static constexpr size_t kOffUpW = 1638400;
    static constexpr size_t kOffGateS = 4915200;
    static constexpr size_t kOffUpS = 4915456;

    const uint8_t* gw_base = jb.exp_base + kOffGateW + static_cast<size_t>(r0) * 2560;
    const uint8_t* uw_base = jb.exp_base + kOffUpW + static_cast<size_t>(r0) * 2560;
    const uint16_t* gs = reinterpret_cast<const uint16_t*>(jb.exp_base + kOffGateS);
    const uint16_t* us = reinterpret_cast<const uint16_t*>(jb.exp_base + kOffUpS);
    const float* x_f32 = jb.x_f32;
    const int rb = (r0 >> 7);

    float sc_g_arr[20];
    float sc_u_arr[20];
    for (int b = 0; b < 20; ++b) {
        sc_g_arr[b] = bf16_bits_to_f32(gs[rb * 20 + b]);
        sc_u_arr[b] = bf16_bits_to_f32(us[rb * 20 + b]);
    }

    alignas(64) float g_raw[32];
    alignas(64) float u_raw[32];
    for (int r = 0; r < 32; ++r) {
        const uint8_t* gw_row = gw_base + static_cast<size_t>(r) * 2560;
        const uint8_t* uw_row = uw_base + static_cast<size_t>(r) * 2560;
        __m512 g_acc = _mm512_setzero_ps();
        __m512 u_acc = _mm512_setzero_ps();
        for (int b = 0; b < 20; ++b) {
            __m512 g_blk = _mm512_setzero_ps();
            __m512 u_blk = _mm512_setzero_ps();
            const size_t boff = static_cast<size_t>(b) * 128;
            _mm_prefetch(reinterpret_cast<const char*>(gw_row + boff + 512), _MM_HINT_T0);
            _mm_prefetch(reinterpret_cast<const char*>(uw_row + boff + 512), _MM_HINT_T0);
#pragma unroll
            for (int sub = 0; sub < 8; ++sub) {
                __m512 xv = _mm512_loadu_ps(x_f32 + boff + sub * 16);
                __m512 gwv = decode_f8_e4m3_16(gw_row + boff + sub * 16);
                __m512 uwv = decode_f8_e4m3_16(uw_row + boff + sub * 16);
                g_blk = _mm512_fmadd_ps(gwv, xv, g_blk);
                u_blk = _mm512_fmadd_ps(uwv, xv, u_blk);
            }
            g_acc = _mm512_fmadd_ps(g_blk, _mm512_set1_ps(sc_g_arr[b]), g_acc);
            u_acc = _mm512_fmadd_ps(u_blk, _mm512_set1_ps(sc_u_arr[b]), u_acc);
        }
        g_raw[r] = _mm512_reduce_add_ps(g_acc);
        u_raw[r] = _mm512_reduce_add_ps(u_acc);
    }
    const __m512 v_one = _mm512_set1_ps(1.0f);
    _mm512_storeu_ps(jb.h_f32 + r0 + 0, swiglu16_bf16_avx512(_mm512_load_ps(g_raw + 0), _mm512_load_ps(u_raw + 0), v_one, v_one));
    _mm512_storeu_ps(jb.h_f32 + r0 + 16, swiglu16_bf16_avx512(_mm512_load_ps(g_raw + 16), _mm512_load_ps(u_raw + 16), v_one, v_one));
}

void down_tile80_decode_fp8_avx512(const CpuExpertJob& jb, int r0) {
    static constexpr size_t kOffDownW = 3276800;
    static constexpr size_t kOffDownS = 4915712;

    const uint8_t* dw_base = jb.exp_base + kOffDownW + static_cast<size_t>(r0) * 640;
    const uint16_t* ds = reinterpret_cast<const uint16_t*>(jb.exp_base + kOffDownS);
    const float* h_f32 = jb.h_f32;
    alignas(64) float d_raw[80];

    for (int r = 0; r < 80; ++r) {
        const int rb = (r0 + r) >> 7;
        const uint8_t* dw_row = dw_base + static_cast<size_t>(r) * 640;
        __m512 d_acc = _mm512_setzero_ps();
        for (int b = 0; b < 5; ++b) {
            __m512 d_blk = _mm512_setzero_ps();
            const size_t boff = static_cast<size_t>(b) * 128;
            _mm_prefetch(reinterpret_cast<const char*>(dw_row + boff + 640), _MM_HINT_T0);
#pragma unroll
            for (int sub = 0; sub < 8; ++sub) {
                __m512 hv = _mm512_loadu_ps(h_f32 + boff + sub * 16);
                __m512 dwv = decode_f8_e4m3_16(dw_row + boff + sub * 16);
                d_blk = _mm512_fmadd_ps(dwv, hv, d_blk);
            }
            float sc_d = bf16_bits_to_f32(ds[rb * 5 + b]);
            d_acc = _mm512_fmadd_ps(d_blk, _mm512_set1_ps(sc_d), d_acc);
        }
        d_raw[r] = _mm512_reduce_add_ps(d_acc);
    }
    const __m512 v_w = _mm512_set1_ps(jb.weight);
    float* dst = jb.y_f32 + r0;
    for (int r = 0; r < 80; r += 16) {
        __m512 y_bf = bf16_round16_avx512(_mm512_load_ps(d_raw + r));
        _mm512_storeu_ps(dst + r, bf16_round16_avx512(_mm512_mul_ps(y_bf, v_w)));
    }
}

}  // namespace strata::kernels::cpu
