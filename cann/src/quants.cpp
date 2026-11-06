#include "strata/cann/quants.hpp"

namespace strata::cann {
namespace {

// ggml's GGML_FP16_TO_FP32 for the values stored in a block header.
float half_to_float(uint16_t h) {
    const uint32_t sign = (h >> 15) & 1u;
    const uint32_t exp = (h >> 10) & 0x1fu;
    const uint32_t man = h & 0x3ffu;
    float v;
    if (exp == 0) {
        v = (float) man * (1.0f / 16777216.0f);
    } else if (exp == 31) {
        v = man ? __builtin_nanf("") : __builtin_inff();
    } else {
        v = (float) (man | 0x400u) * (1.0f / 33554432.0f) * (float) (1 << exp);
    }
    return sign ? -v : v;
}

// ggml's get_scale_min_k4: 6-bit scale and min for sub-block j (0..7).
inline void get_scale_min_k4(int j, const uint8_t * q, uint8_t * d, uint8_t * m) {
    if (j < 4) {
        *d = q[j] & 63;
        *m = q[j + 4] & 63;
    } else {
        *d = (q[j + 4] & 0xF) | ((q[j - 4] >> 6) << 4);
        *m = (q[j + 4] >> 4) | ((q[j] >> 6) << 4);
    }
}

} // namespace

int quant_type_size(int ggml_type) {
    switch (ggml_type) {
        case 16: return 66;   // IQ2_XXS
        case 17: return 74;   // IQ2_XS
        case 18: return 98;   // IQ3_XXS
        case 20: return 18;   // IQ4_NL
        case 21: return 110;  // IQ3_S
        case 22: return 82;   // IQ2_S
        case 23: return 136;  // IQ4_XS
        case 42: return 18;   // Q2_0
        case 8: // GGML_TYPE_Q8_0
            return (int) sizeof(block_q8_0);
        case 12: // GGML_TYPE_Q4_K
            return (int) sizeof(block_q4_K);
        case 13: // GGML_TYPE_Q5_K
            return (int) sizeof(block_q5_K);
        case 14: // GGML_TYPE_Q6_K
            return (int) sizeof(block_q6_K);
        default:
            return 0;
    }
}

int quant_block_elements(int ggml_type) {
    if (ggml_type == 8 || ggml_type == 20) {
        return 32; // Q8_0 packs 32 values per block, not QK_K
    }
    if (ggml_type == 42) return 64;
    return quant_type_size(ggml_type) ? QK_K : 0;
}

void dequantize_q4_K(const block_q4_K * blocks, float * out, int64_t k) {
    const int64_t nb = k / QK_K;
    for (int64_t i = 0; i < nb; ++i) {
        const uint8_t * q = blocks[i].qs;
        const float d = half_to_float(blocks[i].d);
        const float min = half_to_float(blocks[i].dmin);

        int is = 0;
        uint8_t sc = 0, m = 0;
        for (int j = 0; j < QK_K; j += 64) {
            get_scale_min_k4(is + 0, blocks[i].scales, &sc, &m);
            const float d1 = d * sc;
            const float m1 = min * m;
            get_scale_min_k4(is + 1, blocks[i].scales, &sc, &m);
            const float d2 = d * sc;
            const float m2 = min * m;
            for (int l = 0; l < 32; ++l) {
                *out++ = d1 * (q[l] & 0xF) - m1;
            }
            for (int l = 0; l < 32; ++l) {
                *out++ = d2 * (q[l] >> 4) - m2;
            }
            q += 32;
            is += 2;
        }
    }
}

void dequantize_q5_K(const block_q5_K * blocks, float * out, int64_t k) {
    const int64_t nb = k / QK_K;
    for (int64_t i = 0; i < nb; ++i) {
        const uint8_t * ql = blocks[i].qs;
        const uint8_t * qh = blocks[i].qh;
        const float d = half_to_float(blocks[i].d);
        const float min = half_to_float(blocks[i].dmin);

        int is = 0;
        uint8_t sc = 0, m = 0;
        uint8_t u1 = 1, u2 = 2;
        for (int j = 0; j < QK_K; j += 64) {
            get_scale_min_k4(is + 0, blocks[i].scales, &sc, &m);
            const float d1 = d * sc;
            const float m1 = min * m;
            get_scale_min_k4(is + 1, blocks[i].scales, &sc, &m);
            const float d2 = d * sc;
            const float m2 = min * m;
            for (int l = 0; l < 32; ++l) {
                *out++ = d1 * ((ql[l] & 0xF) + ((qh[l] & u1) ? 16 : 0)) - m1;
            }
            for (int l = 0; l < 32; ++l) {
                *out++ = d2 * ((ql[l] >> 4) + ((qh[l] & u2) ? 16 : 0)) - m2;
            }
            ql += 32;
            is += 2;
            u1 = (uint8_t) (u1 << 2);
            u2 = (uint8_t) (u2 << 2);
        }
    }
}

void dequantize_q6_K(const block_q6_K * blocks, float * out, int64_t k) {
    const int64_t nb = k / QK_K;
    for (int64_t i = 0; i < nb; ++i) {
        const float d = half_to_float(blocks[i].d);
        const uint8_t * ql = blocks[i].ql;
        const uint8_t * qh = blocks[i].qh;
        const int8_t * sc = blocks[i].scales;

        for (int n = 0; n < QK_K; n += 128) {
            for (int l = 0; l < 32; ++l) {
                const int is = l / 16;
                const int8_t q1 = (int8_t) ((ql[l + 0] & 0xF) | (((qh[l] >> 0) & 3) << 4)) - 32;
                const int8_t q2 = (int8_t) ((ql[l + 32] & 0xF) | (((qh[l] >> 2) & 3) << 4)) - 32;
                const int8_t q3 = (int8_t) ((ql[l + 0] >> 4) | (((qh[l] >> 4) & 3) << 4)) - 32;
                const int8_t q4 = (int8_t) ((ql[l + 32] >> 4) | (((qh[l] >> 6) & 3) << 4)) - 32;
                out[l + 0] = d * sc[is + 0] * q1;
                out[l + 32] = d * sc[is + 2] * q2;
                out[l + 64] = d * sc[is + 4] * q3;
                out[l + 96] = d * sc[is + 6] * q4;
            }
            out += 128;
            ql += 64;
            qh += 32;
            sc += 8;
        }
    }
}

void dequantize_q8_0(const block_q8_0 * blocks, float * out, int64_t k) {
    const int64_t nb = k / 32;
    for (int64_t i = 0; i < nb; ++i) {
        const float d = half_to_float(blocks[i].d);
        for (int j = 0; j < 32; ++j) {
            *out++ = d * (float) blocks[i].qs[j];
        }
    }
}

bool dequantize(int ggml_type, const void * blocks, float * out, int64_t k) {
    if (k <= 0) {
        return false;
    }
    if (ggml_type == 16 || ggml_type == 17 || ggml_type == 18 ||
        ggml_type == 20 || ggml_type == 21 || ggml_type == 22 ||
        ggml_type == 23 || ggml_type == 42) {
        const int block = quant_block_elements(ggml_type);
        return k % block == 0 && dequantize_iq(ggml_type, blocks, out, k);
    }
    if (ggml_type == 8) {
        if (k % 32 != 0) {
            return false;
        }
        dequantize_q8_0((const block_q8_0 *) blocks, out, k);
        return true;
    }
    if (k % QK_K != 0 || quant_type_size(ggml_type) == 0) {
        return false;
    }
    switch (ggml_type) {
        case 12:
            dequantize_q4_K((const block_q4_K *) blocks, out, k);
            return true;
        case 13:
            dequantize_q5_K((const block_q5_K *) blocks, out, k);
            return true;
        case 14:
            dequantize_q6_K((const block_q6_K *) blocks, out, k);
            return true;
        default:
            return false;
    }
}

} // namespace strata::cann
