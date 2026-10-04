// dsv4/dequant.cpp - ggml block formats, ported from llama.cpp ggml/src/ggml-quants.c
// (dequantise_row_* reference versions). Byte offsets are used instead of the ggml structs so
// nothing depends on alignment/padding; the layout is the on-disk GGUF layout.
#include "dsv4/dequant.hpp"

#include "dsv4/gguf_header.hpp"
#include "dsv4/iq_tables.hpp"

#include <cmath>
#include <cstring>

namespace dsv4 {
namespace {

constexpr int QK_K = 256;

inline uint16_t ld16(const uint8_t* p) { uint16_t v; std::memcpy(&v, p, 2); return v; }
inline uint32_t ld32(const uint8_t* p) { uint32_t v; std::memcpy(&v, p, 4); return v; }
inline float    ldff(const uint8_t* p) { float v;   std::memcpy(&v, p, 4); return v; }
inline double   lddd(const uint8_t* p) { double v;  std::memcpy(&v, p, 8); return v; }

inline float f16_to_f32(uint16_t h) {  // GGML_FP16_TO_FP32
    const uint32_t sign = (uint32_t)(h >> 15) & 1u, exp = (h >> 10) & 0x1fu, frac = h & 0x3ffu;
    float out;
    if (exp == 0) out = (float) frac * (1.0f / 16777216.0f);            // subnormal: frac * 2^-24
    else if (exp == 31) out = frac ? NAN : INFINITY;
    else { const uint32_t bits = (exp << 23) + (frac << 13) + ((127u - 15u) << 23); std::memcpy(&out, &bits, 4); }
    return sign ? -out : out;
}
inline float bf16_to_f32(uint16_t h) { const uint32_t bits = (uint32_t) h << 16; float f; std::memcpy(&f, &bits, 4); return f; }

inline float e8m0_to_fp32_half(uint8_t x) {  // GGML_E8M0_TO_FP32_HALF (0.5 * 2^(x-127))
    const uint32_t bits = x < 2 ? (0x00200000u << x) : ((uint32_t)(x - 1) << 23);
    float f; std::memcpy(&f, &bits, 4); return f;
}

// e2m1 values, doubled (shared by MXFP4)
const int8_t kvalues_mxfp4[16] = {0, 1, 2, 3, 4, 6, 8, 12, 0, -1, -2, -3, -4, -6, -8, -12};

inline void get_scale_min_k4(int j, const uint8_t* q, uint8_t* d, uint8_t* m) {
    if (j < 4) { *d = q[j] & 63; *m = q[j + 4] & 63; }
    else { *d = (q[j + 4] & 0xF) | ((q[j - 4] >> 6) << 4); *m = (q[j + 4] >> 4) | ((q[j - 0] >> 6) << 4); }
}

// ------------------------------------------------------------------ non-blocked types
void dq_f32(const uint8_t* w, int64_t n, float* y) { for (int64_t i = 0; i < n; ++i) y[i] = ldff(w + 4 * i); }
void dq_f64(const uint8_t* w, int64_t n, float* y) { for (int64_t i = 0; i < n; ++i) y[i] = (float) lddd(w + 8 * i); }
void dq_f16(const uint8_t* w, int64_t n, float* y) { for (int64_t i = 0; i < n; ++i) y[i] = f16_to_f32(ld16(w + 2 * i)); }
void dq_bf16(const uint8_t* w, int64_t n, float* y) { for (int64_t i = 0; i < n; ++i) y[i] = bf16_to_f32(ld16(w + 2 * i)); }
void dq_i8(const uint8_t* w, int64_t n, float* y) { for (int64_t i = 0; i < n; ++i) y[i] = (float)(int8_t) w[i]; }
void dq_i16(const uint8_t* w, int64_t n, float* y) { for (int64_t i = 0; i < n; ++i) y[i] = (float)(int16_t) ld16(w + 2 * i); }
void dq_i32(const uint8_t* w, int64_t n, float* y) { for (int64_t i = 0; i < n; ++i) y[i] = (float)(int32_t) ld32(w + 4 * i); }
void dq_i64(const uint8_t* w, int64_t n, float* y) { for (int64_t i = 0; i < n; ++i) { int64_t v; std::memcpy(&v, w + 8 * i, 8); y[i] = (float) v; } }

// ------------------------------------------------------------------ Q*_0 / Q*_1 / Q8_0
void dq_q4_0(const uint8_t* w, int64_t nb, float* y) {
    for (int64_t i = 0; i < nb; ++i) {
        const uint8_t* b = w + 18 * i; const float d = f16_to_f32(ld16(b)); const uint8_t* qs = b + 2;
        for (int j = 0; j < 16; ++j) { y[i * 32 + j] = d * ((int8_t)(qs[j] & 0xF) - 8); y[i * 32 + j + 16] = d * ((int8_t)(qs[j] >> 4) - 8); }
    }
}
void dq_q4_1(const uint8_t* w, int64_t nb, float* y) {
    for (int64_t i = 0; i < nb; ++i) {
        const uint8_t* b = w + 20 * i; const float d = f16_to_f32(ld16(b)), m = f16_to_f32(ld16(b + 2)); const uint8_t* qs = b + 4;
        for (int j = 0; j < 16; ++j) { y[i * 32 + j] = d * (qs[j] & 0xF) + m; y[i * 32 + j + 16] = d * (qs[j] >> 4) + m; }
    }
}
void dq_q5_0(const uint8_t* w, int64_t nb, float* y) {
    for (int64_t i = 0; i < nb; ++i) {
        const uint8_t* b = w + 22 * i; const float d = f16_to_f32(ld16(b)); const uint32_t qh = ld32(b + 2); const uint8_t* qs = b + 6;
        for (int j = 0; j < 16; ++j) {
            const uint8_t xh0 = ((qh >> (j + 0)) << 4) & 0x10, xh1 = ((qh >> (j + 12))) & 0x10;
            y[i * 32 + j] = d * (((qs[j] & 0xF) | xh0) - 16);
            y[i * 32 + j + 16] = d * (((qs[j] >> 4) | xh1) - 16);
        }
    }
}
void dq_q5_1(const uint8_t* w, int64_t nb, float* y) {
    for (int64_t i = 0; i < nb; ++i) {
        const uint8_t* b = w + 24 * i; const float d = f16_to_f32(ld16(b)), m = f16_to_f32(ld16(b + 2)); const uint32_t qh = ld32(b + 4); const uint8_t* qs = b + 8;
        for (int j = 0; j < 16; ++j) {
            const uint8_t xh0 = ((qh >> (j + 0)) << 4) & 0x10, xh1 = ((qh >> (j + 12))) & 0x10;
            y[i * 32 + j] = d * (((qs[j] & 0xF) | xh0)) + m;
            y[i * 32 + j + 16] = d * (((qs[j] >> 4) | xh1)) + m;
        }
    }
}
void dq_q8_0(const uint8_t* w, int64_t nb, float* y) {
    for (int64_t i = 0; i < nb; ++i) {
        const uint8_t* b = w + 34 * i; const float d = f16_to_f32(ld16(b)); const int8_t* qs = (const int8_t*)(b + 2);
        for (int j = 0; j < 32; ++j) y[i * 32 + j] = d * qs[j];
    }
}
void dq_q8_k(const uint8_t* w, int64_t nb, float* y) {
    for (int64_t i = 0; i < nb; ++i) {
        const uint8_t* b = w + 292 * i; const float d = ldff(b); const int8_t* qs = (const int8_t*)(b + 4);
        for (int j = 0; j < QK_K; ++j) y[i * QK_K + j] = d * qs[j];
    }
}

// ------------------------------------------------------------------ K-quants
void dq_q2_k(const uint8_t* w, int64_t nb, float* y) {  // { scales[16], qs[64], d, dmin }
    for (int64_t i = 0; i < nb; ++i) {
        const uint8_t* b = w + 84 * i;
        const float d = f16_to_f32(ld16(b + 80)), min = f16_to_f32(ld16(b + 82));
        const uint8_t* sc = b; const uint8_t* q = b + 16;
        float* out = y + i * QK_K; int is = 0;
        for (int n = 0; n < QK_K; n += 128) {
            for (int shift = 0, j = 0; j < 4; shift += 2, ++j) {
                uint8_t s = sc[is++]; float dl = d * (s & 0xF), ml = min * (s >> 4);
                for (int l = 0; l < 16; ++l) *out++ = dl * (int8_t)((q[l] >> shift) & 3) - ml;
                s = sc[is++]; dl = d * (s & 0xF); ml = min * (s >> 4);
                for (int l = 0; l < 16; ++l) *out++ = dl * (int8_t)((q[l + 16] >> shift) & 3) - ml;
            }
            q += 32;
        }
    }
}
void dq_q3_k(const uint8_t* w, int64_t nb, float* y) {
    for (int64_t i = 0; i < nb; ++i) {
        const uint8_t* b = w + 110 * i;
        const float d_all = f16_to_f32(ld16(b + 108));
        const uint8_t* hm = b;         // hmask[32]
        const uint8_t* q = b + 32;     // qs[64]
        uint32_t aux[4]; std::memcpy(aux, b + 96, 12);
        const uint32_t kmask1 = 0x03030303, kmask2 = 0x0f0f0f0f;
        const uint32_t tmp = aux[2];
        aux[2] = ((aux[0] >> 4) & kmask2) | (((tmp >> 4) & kmask1) << 4);
        aux[3] = ((aux[1] >> 4) & kmask2) | (((tmp >> 6) & kmask1) << 4);
        aux[0] = (aux[0] & kmask2) | (((tmp >> 0) & kmask1) << 4);
        aux[1] = (aux[1] & kmask2) | (((tmp >> 2) & kmask1) << 4);
        const int8_t* scales = (const int8_t*) aux;
        float* out = y + i * QK_K; int is = 0; uint8_t m = 1;
        for (int n = 0; n < QK_K; n += 128) {
            for (int shift = 0, j = 0; j < 4; shift += 2, ++j) {
                float dl = d_all * (scales[is++] - 32);
                for (int l = 0; l < 16; ++l) *out++ = dl * ((int8_t)((q[l + 0] >> shift) & 3) - ((hm[l + 0] & m) ? 0 : 4));
                dl = d_all * (scales[is++] - 32);
                for (int l = 0; l < 16; ++l) *out++ = dl * ((int8_t)((q[l + 16] >> shift) & 3) - ((hm[l + 16] & m) ? 0 : 4));
                m <<= 1;
            }
            q += 32;
        }
    }
}
void dq_q4_k(const uint8_t* w, int64_t nb, float* y) {
    for (int64_t i = 0; i < nb; ++i) {
        const uint8_t* b = w + 144 * i;
        const float d = f16_to_f32(ld16(b)), min = f16_to_f32(ld16(b + 2));
        const uint8_t* sc = b + 4; const uint8_t* q = b + 16;
        float* out = y + i * QK_K; int is = 0;
        for (int j = 0; j < QK_K; j += 64) {
            uint8_t s, mm;
            get_scale_min_k4(is + 0, sc, &s, &mm); const float d1 = d * s, m1 = min * mm;
            get_scale_min_k4(is + 1, sc, &s, &mm); const float d2 = d * s, m2 = min * mm;
            for (int l = 0; l < 32; ++l) *out++ = d1 * (q[l] & 0xF) - m1;
            for (int l = 0; l < 32; ++l) *out++ = d2 * (q[l] >> 4) - m2;
            q += 32; is += 2;
        }
    }
}
void dq_q5_k(const uint8_t* w, int64_t nb, float* y) {
    for (int64_t i = 0; i < nb; ++i) {
        const uint8_t* b = w + 176 * i;
        const float d = f16_to_f32(ld16(b)), min = f16_to_f32(ld16(b + 2));
        const uint8_t* sc = b + 4; const uint8_t* qh = b + 16; const uint8_t* ql = b + 48;
        float* out = y + i * QK_K; int is = 0; uint8_t u1 = 1, u2 = 2;
        for (int j = 0; j < QK_K; j += 64) {
            uint8_t s, mm;
            get_scale_min_k4(is + 0, sc, &s, &mm); const float d1 = d * s, m1 = min * mm;
            get_scale_min_k4(is + 1, sc, &s, &mm); const float d2 = d * s, m2 = min * mm;
            for (int l = 0; l < 32; ++l) *out++ = d1 * ((ql[l] & 0xF) + (qh[l] & u1 ? 16 : 0)) - m1;
            for (int l = 0; l < 32; ++l) *out++ = d2 * ((ql[l] >> 4) + (qh[l] & u2 ? 16 : 0)) - m2;
            ql += 32; is += 2; u1 <<= 2; u2 <<= 2;
        }
    }
}
void dq_q6_k(const uint8_t* w, int64_t nb, float* y) {
    for (int64_t i = 0; i < nb; ++i) {
        const uint8_t* b = w + 210 * i;
        const float d = f16_to_f32(ld16(b + 208));
        const uint8_t* ql = b; const uint8_t* qh = b + 128; const int8_t* sc = (const int8_t*)(b + 192);
        float* y0 = y + i * QK_K;
        for (int n = 0; n < QK_K; n += 128) {
            float* out = y0 + n;
            for (int l = 0; l < 32; ++l) {
                const int is = l / 16;
                const int8_t q1 = (int8_t)((ql[l + 0] & 0xF) | (((qh[l] >> 0) & 3) << 4)) - 32;
                const int8_t q2 = (int8_t)((ql[l + 32] & 0xF) | (((qh[l] >> 2) & 3) << 4)) - 32;
                const int8_t q3 = (int8_t)((ql[l + 0] >> 4) | (((qh[l] >> 4) & 3) << 4)) - 32;
                const int8_t q4 = (int8_t)((ql[l + 32] >> 4) | (((qh[l] >> 6) & 3) << 4)) - 32;
                out[l + 0] = d * sc[is + 0] * q1;
                out[l + 32] = d * sc[is + 2] * q2;
                out[l + 64] = d * sc[is + 4] * q3;
                out[l + 96] = d * sc[is + 6] * q4;
            }
            ql += 64; qh += 32; sc += 8;
        }
    }
}

// ------------------------------------------------------------------ IQ quants
void dq_iq2_xxs(const uint8_t* w, int64_t nb, float* y) {
    for (int64_t i = 0; i < nb; ++i) {
        const uint8_t* b = w + 66 * i; const float d = f16_to_f32(ld16(b)); const uint8_t* qs = b + 2;
        float* out = y + i * QK_K;
        for (int ib32 = 0; ib32 < QK_K / 32; ++ib32) {
            uint32_t aux32[2]; std::memcpy(aux32, qs + 8 * ib32, 2 * sizeof(uint32_t));  // qs is uint16_t[32] in ggml
            const uint8_t* aux8 = (const uint8_t*) aux32;
            const float db = d * (0.5f + (aux32[1] >> 28)) * 0.25f;
            for (int l = 0; l < 4; ++l) {
                const uint8_t* grid = (const uint8_t*)(iq::iq2xxs_grid + aux8[l]);
                const uint8_t signs = iq::ksigns_iq2xs[(aux32[1] >> 7 * l) & 127];
                for (int j = 0; j < 8; ++j) out[j] = db * grid[j] * (signs & iq::kmask_iq2xs[j] ? -1.f : 1.f);
                out += 8;
            }
        }
    }
}
void dq_iq2_xs(const uint8_t* w, int64_t nb, float* y) {
    for (int64_t i = 0; i < nb; ++i) {
        const uint8_t* b = w + 74 * i; const float d = f16_to_f32(ld16(b)); const uint8_t* qs = b + 2; const uint8_t* sc = b + 66;
        float* out = y + i * QK_K;
        for (int ib32 = 0; ib32 < QK_K / 32; ++ib32) {
            const float db[2] = {d * (0.5f + (sc[ib32] & 0xf)) * 0.25f, d * (0.5f + (sc[ib32] >> 4)) * 0.25f};
            for (int l = 0; l < 4; ++l) {
                const uint16_t v = ld16(qs + 2 * (4 * ib32 + l));
                const uint8_t* grid = (const uint8_t*)(iq::iq2xs_grid + (v & 511));
                const uint8_t signs = iq::ksigns_iq2xs[v >> 9];
                for (int j = 0; j < 8; ++j) out[j] = db[l / 2] * grid[j] * (signs & iq::kmask_iq2xs[j] ? -1.f : 1.f);
                out += 8;
            }
        }
    }
}
void dq_iq2_s(const uint8_t* w, int64_t nb, float* y) {
    for (int64_t i = 0; i < nb; ++i) {
        const uint8_t* b = w + 82 * i; const float d = f16_to_f32(ld16(b));
        const uint8_t* qs = b + 2; const uint8_t* qh = b + 66; const uint8_t* signs = qs + QK_K / 8; const uint8_t* sc = b + 74;
        float* out = y + i * QK_K;
        for (int ib32 = 0; ib32 < QK_K / 32; ++ib32) {
            const float db[2] = {d * (0.5f + (sc[ib32] & 0xf)) * 0.25f, d * (0.5f + (sc[ib32] >> 4)) * 0.25f};
            for (int l = 0; l < 4; ++l) {
                const float dl = db[l / 2];
                const uint8_t* grid = (const uint8_t*)(iq::iq2s_grid + (qs[l] | ((qh[ib32] << (8 - 2 * l)) & 0x300)));
                for (int j = 0; j < 8; ++j) out[j] = dl * grid[j] * (signs[l] & iq::kmask_iq2xs[j] ? -1.f : 1.f);
                out += 8;
            }
            qs += 4; signs += 4;
        }
    }
}
void dq_iq3_xxs(const uint8_t* w, int64_t nb, float* y) {
    for (int64_t i = 0; i < nb; ++i) {
        const uint8_t* b = w + 98 * i; const float d = f16_to_f32(ld16(b));
        const uint8_t* qs = b + 2; const uint8_t* scales_and_signs = qs + QK_K / 4;
        float* out = y + i * QK_K;
        for (int ib32 = 0; ib32 < QK_K / 32; ++ib32) {
            uint32_t aux32; std::memcpy(&aux32, scales_and_signs + 4 * ib32, sizeof(aux32));
            const float db = d * (0.5f + (aux32 >> 28)) * 0.5f;
            for (int l = 0; l < 4; ++l) {
                const uint8_t signs = iq::ksigns_iq2xs[(aux32 >> 7 * l) & 127];
                const uint8_t* g1 = (const uint8_t*)(iq::iq3xxs_grid + qs[2 * l + 0]);
                const uint8_t* g2 = (const uint8_t*)(iq::iq3xxs_grid + qs[2 * l + 1]);
                for (int j = 0; j < 4; ++j) {
                    out[j + 0] = db * g1[j] * (signs & iq::kmask_iq2xs[j + 0] ? -1.f : 1.f);
                    out[j + 4] = db * g2[j] * (signs & iq::kmask_iq2xs[j + 4] ? -1.f : 1.f);
                }
                out += 8;
            }
            qs += 8;
        }
    }
}
void dq_iq3_s(const uint8_t* w, int64_t nb, float* y) {
    for (int64_t i = 0; i < nb; ++i) {
        const uint8_t* b = w + 110 * i; const float d = f16_to_f32(ld16(b));
        const uint8_t* qs = b + 2; const uint8_t* qh = b + 66; const uint8_t* signs = b + 74; const uint8_t* sc = b + 106;
        float* out = y + i * QK_K;
        for (int ib32 = 0; ib32 < QK_K / 32; ib32 += 2) {
            const float db1 = d * (1 + 2 * (sc[ib32 / 2] & 0xf)), db2 = d * (1 + 2 * (sc[ib32 / 2] >> 4));
            for (int l = 0; l < 4; ++l) {
                const uint8_t* g1 = (const uint8_t*)(iq::iq3s_grid + (qs[2 * l + 0] | ((qh[0] << (8 - 2 * l)) & 256)));
                const uint8_t* g2 = (const uint8_t*)(iq::iq3s_grid + (qs[2 * l + 1] | ((qh[0] << (7 - 2 * l)) & 256)));
                for (int j = 0; j < 4; ++j) {
                    out[j + 0] = db1 * g1[j] * (signs[l] & iq::kmask_iq2xs[j + 0] ? -1.f : 1.f);
                    out[j + 4] = db1 * g2[j] * (signs[l] & iq::kmask_iq2xs[j + 4] ? -1.f : 1.f);
                }
                out += 8;
            }
            qs += 8; signs += 4;
            for (int l = 0; l < 4; ++l) {
                const uint8_t* g1 = (const uint8_t*)(iq::iq3s_grid + (qs[2 * l + 0] | ((qh[1] << (8 - 2 * l)) & 256)));
                const uint8_t* g2 = (const uint8_t*)(iq::iq3s_grid + (qs[2 * l + 1] | ((qh[1] << (7 - 2 * l)) & 256)));
                for (int j = 0; j < 4; ++j) {
                    out[j + 0] = db2 * g1[j] * (signs[l] & iq::kmask_iq2xs[j + 0] ? -1.f : 1.f);
                    out[j + 4] = db2 * g2[j] * (signs[l] & iq::kmask_iq2xs[j + 4] ? -1.f : 1.f);
                }
                out += 8;
            }
            qh += 2; qs += 8; signs += 4;
        }
    }
}
void dq_iq1_s(const uint8_t* w, int64_t nb, float* y) {
    for (int64_t i = 0; i < nb; ++i) {
        const uint8_t* b = w + 50 * i; const float d = f16_to_f32(ld16(b)); const uint8_t* qs = b + 2;
        const uint8_t* qhb = b + 34;
        float* out = y + i * QK_K;
        for (int ib = 0; ib < QK_K / 32; ++ib) {
            const uint16_t qh = ld16(qhb + 2 * ib);
            const float dl = d * (2 * ((qh >> 12) & 7) + 1);
            const float delta = qh & 0x8000 ? -iq::kIq1sDelta : iq::kIq1sDelta;
            for (int l = 0; l < 4; ++l) {
                const int8_t* grid = (const int8_t*)(iq::iq1s_grid + (qs[l] | (((qh >> 3 * l) & 7) << 8)));
                for (int j = 0; j < 8; ++j) out[j] = dl * (grid[j] + delta);
                out += 8;
            }
            qs += 4;
        }
    }
}
void dq_iq1_m(const uint8_t* w, int64_t nb, float* y) {
    for (int64_t i = 0; i < nb; ++i) {
        const uint8_t* b = w + 56 * i;
        const uint8_t* qs = b; const uint8_t* qhb = b + 32; const uint8_t* scb = b + 48;
        const uint16_t sc[4] = {ld16(scb), ld16(scb + 2), ld16(scb + 4), ld16(scb + 6)};
        const uint16_t su = (uint16_t)((sc[0] >> 12) | ((sc[1] >> 8) & 0x00f0) | ((sc[2] >> 4) & 0x0f00) | (sc[3] & 0xf000));
        const float d = f16_to_f32(su);
        float* out = y + i * QK_K;
        for (int ib = 0; ib < QK_K / 32; ++ib) {
            const uint8_t qh0 = qhb[2 * ib], qh1 = qhb[2 * ib + 1];   // qh is a uint8_t array in ggml
            const float dl1 = d * (2 * ((sc[ib / 2] >> (6 * (ib % 2) + 0)) & 0x7) + 1);
            const float dl2 = d * (2 * ((sc[ib / 2] >> (6 * (ib % 2) + 3)) & 0x7) + 1);
            const uint16_t idx[4] = {(uint16_t)(qs[0] | ((qh0 << 8) & 0x700)), (uint16_t)(qs[1] | ((qh0 << 4) & 0x700)),
                                     (uint16_t)(qs[2] | ((qh1 << 8) & 0x700)), (uint16_t)(qs[3] | ((qh1 << 4) & 0x700))};
            const float delta[4] = {qh0 & 0x08 ? -iq::kIq1sDelta : iq::kIq1sDelta, qh0 & 0x80 ? -iq::kIq1sDelta : iq::kIq1sDelta,
                                    qh1 & 0x08 ? -iq::kIq1sDelta : iq::kIq1sDelta, qh1 & 0x80 ? -iq::kIq1sDelta : iq::kIq1sDelta};
            for (int l = 0; l < 4; ++l) {
                const float dl = l < 2 ? dl1 : dl2;
                const int8_t* grid = (const int8_t*)(iq::iq1s_grid + idx[l]);
                for (int j = 0; j < 8; ++j) out[j] = dl * (grid[j] + delta[l]);
                out += 8;
            }
            qs += 4;
        }
    }
}
void dq_iq4_nl(const uint8_t* w, int64_t nb, float* y) {
    for (int64_t i = 0; i < nb; ++i) {
        const uint8_t* b = w + 18 * i; const float d = f16_to_f32(ld16(b)); const uint8_t* qs = b + 2;
        for (int j = 0; j < 16; ++j) { y[i * 32 + j] = d * iq::kvalues_iq4nl[qs[j] & 0xf]; y[i * 32 + j + 16] = d * iq::kvalues_iq4nl[qs[j] >> 4]; }
    }
}
void dq_iq4_xs(const uint8_t* w, int64_t nb, float* y) {
    for (int64_t i = 0; i < nb; ++i) {
        const uint8_t* b = w + 136 * i; const float d = f16_to_f32(ld16(b));
        const uint16_t scales_h = ld16(b + 2); const uint8_t* scales_l = b + 4; const uint8_t* qs = b + 8;
        float* out = y + i * QK_K;
        for (int ib = 0; ib < QK_K / 32; ++ib) {
            const int ls = ((scales_l[ib / 2] >> 4 * (ib % 2)) & 0xf) | (((scales_h >> 2 * ib) & 3) << 4);
            const float dl = d * (ls - 32);
            for (int j = 0; j < 16; ++j) { out[j + 0] = dl * iq::kvalues_iq4nl[qs[j] & 0xf]; out[j + 16] = dl * iq::kvalues_iq4nl[qs[j] >> 4]; }
            out += 32; qs += 16;
        }
    }
}
void dq_mxfp4(const uint8_t* w, int64_t nb, float* y) {
    for (int64_t i = 0; i < nb; ++i) {
        const uint8_t* b = w + 17 * i; const float d = e8m0_to_fp32_half(b[0]); const uint8_t* qs = b + 1;
        for (int j = 0; j < 16; ++j) { y[i * 32 + j] = d * kvalues_mxfp4[qs[j] & 0x0F]; y[i * 32 + j + 16] = d * kvalues_mxfp4[qs[j] >> 4]; }
    }
}

// One switch per ROW instead of one per block. row_dot() below used to call dequant_row() once per
// 256-element block: for a 4096-wide IQ1_M row that is 16 passes through dequant_block_elems()
// (a ggml_type_info lookup), the alignment checks and the dispatch switch, on top of ~1.5 us of real
// work. pick() resolves the decoder once and the loop just calls it.
typedef void (*dq_blocks_fn)(const uint8_t*, int64_t, float*);

dq_blocks_fn pick(uint32_t type) {
    switch (type) {
        case T_F32:   return dq_f32;
        case T_F16:   return dq_f16;
        case T_BF16:  return dq_bf16;
        case T_F64:   return dq_f64;
        case T_I8:    return dq_i8;
        case T_I16:   return dq_i16;
        case T_I32:   return dq_i32;
        case T_I64:   return dq_i64;
        case T_Q4_0:  return dq_q4_0;
        case T_Q4_1:  return dq_q4_1;
        case T_Q5_0:  return dq_q5_0;
        case T_Q5_1:  return dq_q5_1;
        case T_Q8_0:  return dq_q8_0;
        case T_Q8_K:  return dq_q8_k;
        case T_Q2_K:  return dq_q2_k;
        case T_Q3_K:  return dq_q3_k;
        case T_Q4_K:  return dq_q4_k;
        case T_Q5_K:  return dq_q5_k;
        case T_Q6_K:  return dq_q6_k;
        case T_IQ2_XXS: return dq_iq2_xxs;
        case T_IQ2_XS:  return dq_iq2_xs;
        case T_IQ2_S:   return dq_iq2_s;
        case T_IQ3_XXS: return dq_iq3_xxs;
        case T_IQ3_S:   return dq_iq3_s;
        case T_IQ1_S:   return dq_iq1_s;
        case T_IQ1_M:   return dq_iq1_m;
        case T_IQ4_NL:  return dq_iq4_nl;
        case T_IQ4_XS:  return dq_iq4_xs;
        case T_MXFP4:   return dq_mxfp4;
        default: return nullptr;
    }
}

}  // namespace

bool dequant_supported(uint32_t type) {
    switch (type) {
        case T_F32: case T_F16: case T_BF16: case T_F64:
        case T_I8: case T_I16: case T_I32: case T_I64:
        case T_Q4_0: case T_Q4_1: case T_Q5_0: case T_Q5_1: case T_Q8_0: case T_Q8_K:
        case T_Q2_K: case T_Q3_K: case T_Q4_K: case T_Q5_K: case T_Q6_K:
        case T_IQ2_XXS: case T_IQ2_XS: case T_IQ2_S: case T_IQ3_XXS: case T_IQ3_S:
        case T_IQ1_S: case T_IQ1_M: case T_IQ4_NL: case T_IQ4_XS: case T_MXFP4:
            return true;
        default: return false;
    }
}

int dequant_block_elems(uint32_t type) {
    const char* name = nullptr; int be = 0, bb = 0;
    if (!ggml_type_info(type, &name, &be, &bb)) return 0;
    return be;
}

size_t row_bytes(uint32_t type, int64_t in) {
    const char* name = nullptr; int be = 0, bb = 0;
    if (!ggml_type_info(type, &name, &be, &bb) || be <= 0 || in < 0 || in % be != 0) return 0;
    return (size_t)(in / be) * (size_t) bb;
}

void dequant_row(uint32_t type, const uint8_t* w, int64_t in, float* y) {
    if (in <= 0 || !w || !y) return;
    const int be = dequant_block_elems(type);
    if (be <= 0 || in % be != 0) return;
    const dq_blocks_fn fn = pick(type);
    if (!fn) return;
    fn(w, in / be, y);   // the plain types have be == 1, so in / be == in, exactly as the old switch passed it
}

float row_dot(uint32_t type, const uint8_t* w, const float* x, int64_t in) {
    if (in <= 0 || !w || !x) return 0.f;
    const int be = dequant_block_elems(type);
    if (be <= 0 || in % be != 0 || !dequant_supported(type)) return 0.f;
    if (be == 1) {  // plain element types: no need to materialise a block
        switch (type) {
            case T_F32: { double a = 0; for (int64_t i = 0; i < in; ++i) a += (double) x[i] * ldff(w + 4 * i); return (float) a; }
            case T_F16: { double a = 0; for (int64_t i = 0; i < in; ++i) a += (double) x[i] * f16_to_f32(ld16(w + 2 * i)); return (float) a; }
            case T_BF16: { double a = 0; for (int64_t i = 0; i < in; ++i) a += (double) x[i] * bf16_to_f32(ld16(w + 2 * i)); return (float) a; }
            case T_I8: { double a = 0; for (int64_t i = 0; i < in; ++i) a += (double) x[i] * (int8_t) w[i]; return (float) a; }
            case T_I16: { double a = 0; for (int64_t i = 0; i < in; ++i) a += (double) x[i] * (int16_t) ld16(w + 2 * i); return (float) a; }
            case T_I32: { double a = 0; for (int64_t i = 0; i < in; ++i) a += (double) x[i] * (int32_t) ld32(w + 4 * i); return (float) a; }
            default: break;
        }
    }
    // The decoder is resolved ONCE for the row. Going through dequant_row() per block meant a
    // ggml_type_info lookup, two modulo checks and a 30-way switch every 256 elements: on an IQ1_M row
    // of 4096 elements that overhead was a measurable share of the dot product itself.
    const dq_blocks_fn fn = pick(type);
    if (!fn) return 0.f;
    alignas(64) float blk[QK_K];
    const size_t rb = row_bytes(type, in);
    const size_t bb = rb / (size_t) (in / be);   // bytes per block, from the same source of truth
    double acc = 0;
    const uint8_t* p = w;
    for (int64_t off = 0; off < in; off += be, p += bb) {
        fn(p, 1, blk);
        for (int j = 0; j < be; ++j) acc += (double) blk[j] * x[off + j];
    }
    return (float) acc;
}

}  // namespace dsv4
