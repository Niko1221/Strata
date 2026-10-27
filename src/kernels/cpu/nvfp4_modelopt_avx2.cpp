// src/kernels/cpu/nvfp4_modelopt_avx2.cpp - AVX2 + FMA fallback kernels for ModelOpt NVFP4 CPU experts.
//
// Uses a constexpr F8_E4M3 table so there is no static initializer in this translation unit.
#include "strata/kernels/cpu/nvfp4_modelopt.hpp"

#include <immintrin.h>

#include <cmath>
#include <cstddef>
#include <cstdint>
#include <cstring>

namespace strata::kernels::cpu {
namespace {

constexpr float pow2i(int e) {
    float r = 1.0f;
    for (; e > 0; --e) r *= 2.0f;
    for (; e < 0; ++e) r *= 0.5f;
    return r;
}

struct F8E4m3Table {
    float v[256];
    constexpr F8E4m3Table() : v() {
        for (int i = 0; i < 256; ++i) {
            int s = (i >> 7) & 1;
            int e = (i >> 3) & 0xF;
            int m = i & 0x7;
            float val = 0.0f;
            if (e == 0) {
                val = static_cast<float>(m) * pow2i(-9);
            } else if (e == 15 && m == 7) {
                val = 0.0f;
            } else {
                val = (1.0f + static_cast<float>(m) * 0.125f) * pow2i(e - 7);
            }
            v[i] = s ? -val : val;
        }
    }
};

constexpr F8E4m3Table kF8E4m3;

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

inline void quantize_act_f32_32_avx2(const float* h32, int8_t* q_h, float* d_h, int32_t* neg_bias_h) {
    for (int sub = 0; sub < 2; ++sub) {
        const float* xv = h32 + 16 * sub;
        float amax = 0.0f;
        for (int i = 0; i < 16; ++i) {
            float a = std::fabs(xv[i]);
            if (a > amax) amax = a;
        }
        if (amax == 0.0f) {
            std::memset(q_h + 16 * sub, 0, 16);
            d_h[sub] = 0.0f;
            for (int g = 0; g < 4; ++g) neg_bias_h[4 * sub + g] = 0;
        } else {
            float inv = 127.0f / amax;
            d_h[sub] = amax * (0.5f / 127.0f);
            for (int g = 0; g < 4; ++g) {
                int32_t sum4 = 0;
                for (int j = 0; j < 4; ++j) {
                    int32_t q = static_cast<int32_t>(std::nearbyint(xv[4 * g + j] * inv));
                    if (q < -127) q = -127;
                    else if (q > 127) q = 127;
                    q_h[16 * sub + 4 * g + j] = static_cast<int8_t>(q);
                    sum4 += q;
                }
                neg_bias_h[4 * sub + g] = -12 * sum4;
            }
        }
    }
}

inline __m256i codes32_modelopt_avx2(const uint8_t* wp, __m256i lut, __m256i mask) {
    __m256i w16 = _mm256_cvtepu8_epi16(_mm_loadu_si128(reinterpret_cast<const __m128i*>(wp)));
    __m256i nibbles = _mm256_and_si256(_mm256_or_si256(w16, _mm256_slli_epi16(w16, 4)), mask);
    return _mm256_shuffle_epi8(lut, nibbles);
}

inline __m256i dpbusd_avx2(__m256i acc, __m256i u8_a, __m256i s8_b, __m256i ones16) {
    __m256i prod16 = _mm256_maddubs_epi16(u8_a, s8_b);
    __m256i prod32 = _mm256_madd_epi16(prod16, ones16);
    return _mm256_add_epi32(acc, prod32);
}

template <int NT, int NB>
inline void nvfp4_row_dot_avx2(const uint8_t* wp, const uint8_t* sp,
                               const int8_t* const* q_x, const float* const* d_x,
                               const int32_t* const* neg_bias_x, float* out_f32) {
    static_assert(NB % 8 == 0, "NB must be a multiple of 8");
    const __m256i lut = _mm256_broadcastsi128_si256(_mm_setr_epi8(
        12, 13, 14, 15, 16, 18, 20, 24, 12, 11, 10, 9, 8, 6, 4, 0));
    const __m256i mask = _mm256_set1_epi8(0x0F);
    const __m256i ones16 = _mm256_set1_epi16(1);
    const __m256i exp_lo = _mm256_set_epi32(1, 1, 1, 1, 0, 0, 0, 0);
    const __m256i exp_hi = _mm256_set_epi32(3, 3, 3, 3, 2, 2, 2, 2);

    __m256 acc0[NT];
    __m256 acc1[NT];
    for (int t = 0; t < NT; ++t) {
        acc0[t] = _mm256_setzero_ps();
        acc1[t] = _mm256_setzero_ps();
    }

    for (int b = 0; b < NB; b += 4) {
        __m256i c0 = codes32_modelopt_avx2(wp + 8 * b, lut, mask);
        __m256i c1 = codes32_modelopt_avx2(wp + 8 * b + 16, lut, mask);
        float ws4[4] = {
            kF8E4m3.v[sp[b + 0]],
            kF8E4m3.v[sp[b + 1]],
            kF8E4m3.v[sp[b + 2]],
            kF8E4m3.v[sp[b + 3]],
        };
        __m128 ws_v = _mm_loadu_ps(ws4);
        for (int t = 0; t < NT; ++t) {
            __m128 sc4 = _mm_mul_ps(ws_v, _mm_loadu_ps(d_x[t] + b));
            __m256 sc4_256 = _mm256_castps128_ps256(sc4);

            __m256i qv0 = _mm256_loadu_si256(reinterpret_cast<const __m256i*>(q_x[t] + 16 * b));
            __m256i nb0 = _mm256_loadu_si256(reinterpret_cast<const __m256i*>(neg_bias_x[t] + 4 * b));
            __m256i raw0 = dpbusd_avx2(nb0, c0, qv0, ones16);
            __m256 sc8_0 = _mm256_permutevar8x32_ps(sc4_256, exp_lo);
            acc0[t] = _mm256_fmadd_ps(_mm256_cvtepi32_ps(raw0), sc8_0, acc0[t]);

            __m256i qv1 = _mm256_loadu_si256(reinterpret_cast<const __m256i*>(q_x[t] + 16 * b + 32));
            __m256i nb1 = _mm256_loadu_si256(reinterpret_cast<const __m256i*>(neg_bias_x[t] + 4 * b + 8));
            __m256i raw1 = dpbusd_avx2(nb1, c1, qv1, ones16);
            __m256 sc8_1 = _mm256_permutevar8x32_ps(sc4_256, exp_hi);
            acc1[t] = _mm256_fmadd_ps(_mm256_cvtepi32_ps(raw1), sc8_1, acc1[t]);
        }
    }
    for (int t = 0; t < NT; ++t) {
        __m256 sum8 = _mm256_add_ps(acc0[t], acc1[t]);
        __m128 sum4 = _mm_add_ps(_mm256_castps256_ps128(sum8), _mm256_extractf128_ps(sum8, 1));
        __m128 sum2 = _mm_add_ps(sum4, _mm_movehl_ps(sum4, sum4));
        __m128 sum1 = _mm_add_ss(sum2, _mm_shuffle_ps(sum2, sum2, 1));
        out_f32[t] = _mm_cvtss_f32(sum1);
    }
}

template <int NT>
void gu_tile32_nt_avx2(CpuExpertJob& jb, int r0, int t0) {
    const uint8_t* gw_base = jb.exp_base + EXP_OFF_GATE_W + static_cast<size_t>(r0) * 1280;
    const uint8_t* gs_base = jb.exp_base + EXP_OFF_GATE_S + static_cast<size_t>(r0) * 160;
    const uint8_t* uw_base = jb.exp_base + EXP_OFF_UP_W + static_cast<size_t>(r0) * 1280;
    const uint8_t* us_base = jb.exp_base + EXP_OFF_UP_S + static_cast<size_t>(r0) * 160;

    alignas(64) float g_all[32][NT];
    for (int r = 0; r < 32; ++r) {
        nvfp4_row_dot_avx2<NT, 160>(gw_base + static_cast<size_t>(r) * 1280, gs_base + static_cast<size_t>(r) * 160,
                                    jb.q_x + t0, jb.d_x + t0, jb.neg_bias_x + t0, g_all[r]);
    }

    alignas(64) float h32[NT][32];
    for (int r = 0; r < 32; ++r) {
        float u_f32[NT];
        nvfp4_row_dot_avx2<NT, 160>(uw_base + static_cast<size_t>(r) * 1280, us_base + static_cast<size_t>(r) * 160,
                                    jb.q_x + t0, jb.d_x + t0, jb.neg_bias_x + t0, u_f32);
        for (int t = 0; t < NT; ++t) {
            float gf = bf16_bits_to_f32(f32_to_bf16_bits(g_all[r][t] * jb.gate_s2));
            float uf = bf16_bits_to_f32(f32_to_bf16_bits(u_f32[t] * jb.up_s2));
            float silu_g = bf16_bits_to_f32(f32_to_bf16_bits(gf / (1.0f + std::exp(-gf))));
            h32[t][r] = bf16_bits_to_f32(f32_to_bf16_bits(silu_g * uf));
        }
    }
    int b0 = r0 / 16;
    for (int t = 0; t < NT; ++t) {
        quantize_act_f32_32_avx2(h32[t], jb.q_h[t0 + t] + r0, jb.d_h[t0 + t] + b0, jb.neg_bias_h[t0 + t] + 4 * b0);
    }
}

template <int NT>
void down_tile80_nt_avx2(CpuExpertJob& jb, int r0, int t0) {
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

    for (int r = 0; r < 80; ++r) {
        float d_f32[NT];
        nvfp4_row_dot_avx2<NT, 40>(dw_base + static_cast<size_t>(r) * 320, ds_base + static_cast<size_t>(r) * 40,
                                   q_h_ptrs, d_h_ptrs, nb_h_ptrs, d_f32);
        for (int t = 0; t < NT; ++t) {
            jb.y_out[t0 + t][r0 + r] = f32_to_bf16_bits(d_f32[t] * jb.down_s2);
        }
    }
}

}  // namespace

void quantize_act_bf16_2560_avx2(const uint16_t* x_bf16, int8_t* q_x, float* d_x, int32_t* neg_bias_x) {
    for (int b = 0; b < 160; ++b) {
        float xv[16];
        float amax = 0.0f;
        for (int i = 0; i < 16; ++i) {
            xv[i] = bf16_bits_to_f32(x_bf16[16 * b + i]);
            float a = std::fabs(xv[i]);
            if (a > amax) amax = a;
        }
        if (amax == 0.0f) {
            std::memset(q_x + 16 * b, 0, 16);
            d_x[b] = 0.0f;
            for (int g = 0; g < 4; ++g) neg_bias_x[4 * b + g] = 0;
        } else {
            float inv = 127.0f / amax;
            d_x[b] = amax * (0.5f / 127.0f);
            for (int g = 0; g < 4; ++g) {
                int32_t sum4 = 0;
                for (int j = 0; j < 4; ++j) {
                    int32_t q = static_cast<int32_t>(std::nearbyint(xv[4 * g + j] * inv));
                    if (q < -127) q = -127;
                    else if (q > 127) q = 127;
                    q_x[16 * b + 4 * g + j] = static_cast<int8_t>(q);
                    sum4 += q;
                }
                neg_bias_x[4 * b + g] = -12 * sum4;
            }
        }
    }
}

void gu_tile32_modelopt_avx2(CpuExpertJob& jb, int r0) {
    for (int t0 = 0; t0 < jb.nt; t0 += 4) {
        int rem = jb.nt - t0;
        switch (rem) {
            case 1: gu_tile32_nt_avx2<1>(jb, r0, t0); break;
            case 2: gu_tile32_nt_avx2<2>(jb, r0, t0); break;
            case 3: gu_tile32_nt_avx2<3>(jb, r0, t0); break;
            default: gu_tile32_nt_avx2<4>(jb, r0, t0); break;
        }
    }
}

void down_tile80_modelopt_avx2(CpuExpertJob& jb, int r0) {
    for (int t0 = 0; t0 < jb.nt; t0 += 4) {
        int rem = jb.nt - t0;
        switch (rem) {
            case 1: down_tile80_nt_avx2<1>(jb, r0, t0); break;
            case 2: down_tile80_nt_avx2<2>(jb, r0, t0); break;
            case 3: down_tile80_nt_avx2<3>(jb, r0, t0); break;
            default: down_tile80_nt_avx2<4>(jb, r0, t0); break;
        }
    }
}

void down_tile80_decode_modelopt_avx2(const CpuExpertJob& jb, int r0) {
    const uint8_t* dw_base = jb.exp_base + EXP_OFF_DOWN_W + static_cast<size_t>(r0) * 320;
    const uint8_t* ds_base = jb.exp_base + EXP_OFF_DOWN_S + static_cast<size_t>(r0) * 40;
    const int8_t* q_h_ptr = jb.q_h[0];
    const float* d_h_ptr = jb.d_h[0];
    const int32_t* nb_h_ptr = jb.neg_bias_h[0];
    const float sd = jb.down_s2;
    const float w = jb.weight;
    float* dst = jb.y_f32 + r0;
    for (int r = 0; r < 80; ++r) {
        float d_f32 = 0.0f;
        nvfp4_row_dot_avx2<1, 40>(dw_base + static_cast<size_t>(r) * 320,
                                  ds_base + static_cast<size_t>(r) * 40,
                                  &q_h_ptr, &d_h_ptr, &nb_h_ptr, &d_f32);
        float y_bf = bf16_bits_to_f32(f32_to_bf16_bits(d_f32 * sd));
        dst[r] = bf16_bits_to_f32(f32_to_bf16_bits(y_bf * w));
    }
}

void gu_tile32_fp8_avx2(CpuExpertJob& jb, int r0) {
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

    for (int r = 0; r < 32; ++r) {
        const uint8_t* gw_row = gw_base + static_cast<size_t>(r) * 2560;
        const uint8_t* uw_row = uw_base + static_cast<size_t>(r) * 2560;
        float g_acc = 0.0f;
        float u_acc = 0.0f;
        for (int b = 0; b < 20; ++b) {
            float g_blk = 0.0f;
            float u_blk = 0.0f;
            const size_t boff = static_cast<size_t>(b) * 128;
            for (int j = 0; j < 128; ++j) {
                float xv = x_f32[boff + j];
                g_blk += kF8E4m3.v[gw_row[boff + j]] * xv;
                u_blk += kF8E4m3.v[uw_row[boff + j]] * xv;
            }
            g_acc += g_blk * bf16_bits_to_f32(gs[rb * 20 + b]);
            u_acc += u_blk * bf16_bits_to_f32(us[rb * 20 + b]);
        }
        float gf = bf16_bits_to_f32(f32_to_bf16_bits(g_acc));
        float uf = bf16_bits_to_f32(f32_to_bf16_bits(u_acc));
        float silu_g = bf16_bits_to_f32(f32_to_bf16_bits(gf / (1.0f + std::exp(-gf))));
        jb.h_f32[r0 + r] = bf16_bits_to_f32(f32_to_bf16_bits(silu_g * uf));
    }
}

void down_tile80_decode_fp8_avx2(const CpuExpertJob& jb, int r0) {
    static constexpr size_t kOffDownW = 3276800;
    static constexpr size_t kOffDownS = 4915712;

    const uint8_t* dw_base = jb.exp_base + kOffDownW + static_cast<size_t>(r0) * 640;
    const uint16_t* ds = reinterpret_cast<const uint16_t*>(jb.exp_base + kOffDownS);
    const float* h_f32 = jb.h_f32;
    const float w = jb.weight;
    float* dst = jb.y_f32 + r0;

    for (int r = 0; r < 80; ++r) {
        const int rb = (r0 + r) >> 7;
        const uint8_t* dw_row = dw_base + static_cast<size_t>(r) * 640;
        float d_acc = 0.0f;
        for (int b = 0; b < 5; ++b) {
            float d_blk = 0.0f;
            const size_t boff = static_cast<size_t>(b) * 128;
            for (int j = 0; j < 128; ++j) {
                d_blk += kF8E4m3.v[dw_row[boff + j]] * h_f32[boff + j];
            }
            d_acc += d_blk * bf16_bits_to_f32(ds[rb * 5 + b]);
        }
        float y_bf = bf16_bits_to_f32(f32_to_bf16_bits(d_acc));
        dst[r] = bf16_bits_to_f32(f32_to_bf16_bits(y_bf * w));
    }
}

}  // namespace strata::kernels::cpu
