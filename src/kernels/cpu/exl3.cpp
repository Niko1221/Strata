// src/kernels/cpu/exl3.cpp - portable reference implementation of the EXL3 decode (see docs/EXL3.md).
//
// This file is the C++ twin of tools/exl3 (codebook.py, reconstruct.py).  It is deliberately scalar and
// dependency-free so it can run on any x86-64 and serve as the oracle for the AVX2 and HIP kernels.
#include "strata/kernels/cpu/exl3.hpp"

#include <cmath>
#include <cstring>
#include <vector>

namespace strata::kernels::cpu {
namespace {

// ---- fp16 bit patterns <-> float (round to nearest even), so the codebook matches the numpy spec ----

float h2f(uint16_t h) {
    uint32_t sign = (uint32_t)(h >> 15) << 31;
    uint32_t exp = (h >> 10) & 0x1F;
    uint32_t man = h & 0x3FF;
    uint32_t f;
    if (exp == 0) {
        if (man == 0) {
            f = sign;
        } else {                                   // subnormal half -> normal float
            exp = 127 - 15 + 1;
            while (!(man & 0x400)) { man <<= 1; --exp; }
            man &= 0x3FF;
            f = sign | (exp << 23) | (man << 13);
        }
    } else if (exp == 0x1F) {
        f = sign | 0x7F800000u | (man << 13);      // inf / nan
    } else {
        f = sign | ((exp + 127 - 15) << 23) | (man << 13);
    }
    float out;
    std::memcpy(&out, &f, 4);
    return out;
}

uint16_t f2h(float x) {
    uint32_t u;
    std::memcpy(&u, &x, 4);
    uint32_t sign = (u >> 16) & 0x8000u;
    uint32_t absu = u & 0x7FFFFFFFu;
    if (absu >= 0x7F800000u) return (uint16_t)(sign | (absu > 0x7F800000u ? 0x7E00u : 0x7C00u));
    int e = (int)((u >> 23) & 0xFF) - 127 + 15;
    uint32_t m = u & 0x7FFFFFu;
    if (e >= 0x1F) return (uint16_t)(sign | 0x7C00u);
    if (e <= 0) {
        if (e < -10) return (uint16_t)sign;
        m |= 0x800000u;
        uint32_t shift = 14 - (uint32_t)e;
        uint32_t h = m >> shift;
        uint32_t rem = m & ((1u << shift) - 1);
        uint32_t half = 1u << (shift - 1);
        if (rem > half || (rem == half && (h & 1))) ++h;
        return (uint16_t)(sign | h);
    }
    uint32_t h = ((uint32_t)e << 10) | (m >> 13);
    uint32_t rem = m & 0x1FFFu;
    if (rem > 0x1000u || (rem == 0x1000u && (h & 1))) ++h;
    return (uint16_t)(sign | h);
}

// ---- the procedural codebook (codebook.cuh) ----

uint32_t lop3(uint32_t a, uint32_t b, uint32_t c, uint32_t imm) {
    uint32_t r = 0;
    for (int i = 0; i < 8; ++i) {
        uint32_t t = ((imm >> i) & 1) ? (~0u) : 0u;
        uint32_t ta = (i & 4) ? a : ~a;
        uint32_t tb = (i & 2) ? b : ~b;
        uint32_t tc = (i & 1) ? c : ~c;
        r |= t & ta & tb & tc;
    }
    return r;
}

uint16_t decode_window(Exl3Codebook cb, uint16_t window) {
    uint32_t x = window;
    if (cb == Exl3Codebook::Mul1) {
        x *= 0x83DCD12Du;
        uint32_t s = (x & 0xFF) + ((x >> 8) & 0xFF) + ((x >> 16) & 0xFF) + ((x >> 24) & 0xFF) + 0x6400u;
        float h = h2f((uint16_t)(s & 0xFFFF));
        float inv = h2f(0x1EEEu);
        float bias = h2f(0xC931u);
        return f2h(std::fma(h, inv, bias));
    }
    if (cb == Exl3Codebook::Mcg) {
        x *= 0xCBAC1FEDu;
    } else {
        x = x * 89226354u + 64248484u;
    }
    x = lop3(x, 0x8FFF8FFFu, 0x3B603B60u, 0x6Au);
    return f2h(h2f((uint16_t)(x & 0xFFFF)) + h2f((uint16_t)(x >> 16)));
}

// ---- 16-bit sliding windows over one packed tile ----

uint16_t window16(const uint16_t* words, int nwords, int start) {
    int total = nwords * 16;
    uint32_t v = 0;
    for (int k = 0; k < 16; ++k) {
        int b = (start + k) % total;
        v |= (uint32_t)((words[b >> 4] >> (b & 15)) & 1) << k;
    }
    return (uint16_t)v;
}

// tensor_core_perm: tc[j] == row_major[perm[j]] (quantize.py).
void build_perm(int* perm) {
    for (int t = 0; t < 32; ++t) {
        int r0 = (t % 4) * 2, r1 = r0 + 1, r2 = r0 + 8, r3 = r0 + 9;
        int c0 = t / 4, c1 = c0 + 8;
        perm[t * 8 + 0] = r0 * 16 + c0;
        perm[t * 8 + 1] = r1 * 16 + c0;
        perm[t * 8 + 2] = r2 * 16 + c0;
        perm[t * 8 + 3] = r3 * 16 + c0;
        perm[t * 8 + 4] = r0 * 16 + c1;
        perm[t * 8 + 5] = r1 * 16 + c1;
        perm[t * 8 + 6] = r2 * 16 + c1;
        perm[t * 8 + 7] = r3 * 16 + c1;
    }
}

// ---- Hadamard 128 (natural-order Sylvester, scaled 1/sqrt(128)) ----

const float (*had128())[128] {
    static float h[128][128];
    static bool init = false;
    if (!init) {
        for (int i = 0; i < 128; ++i)
            for (int j = 0; j < 128; ++j)
                h[i][j] = ((__builtin_popcount((unsigned)(i & j)) & 1) ? -1.0f : 1.0f) /
                          std::sqrt(128.0f);
        init = true;
    }
    return (const float (*)[128])h;
}

void had_rows(std::vector<float>& m, int rows, int cols) {
    const float(*h)[128] = had128();
    std::vector<float> tmp(128);
    for (int b = 0; b < rows; b += 128) {
        for (int c = 0; c < cols; ++c) {
            for (int i = 0; i < 128; ++i) {
                float acc = 0;
                for (int j = 0; j < 128; ++j) acc += h[i][j] * m[(b + j) * cols + c];
                tmp[i] = acc;
            }
            for (int i = 0; i < 128; ++i) m[(b + i) * cols + c] = tmp[i];
        }
    }
}

void had_cols(std::vector<float>& m, int rows, int cols) {
    const float(*h)[128] = had128();
    std::vector<float> tmp(128);
    for (int r = 0; r < rows; ++r) {
        for (int b = 0; b < cols; b += 128) {
            for (int j = 0; j < 128; ++j) {
                float acc = 0;
                for (int k = 0; k < 128; ++k) acc += m[r * cols + b + k] * h[k][j];
                tmp[j] = acc;
            }
            for (int j = 0; j < 128; ++j) m[r * cols + b + j] = tmp[j];
        }
    }
}

const uint16_t* lut_for(Exl3Codebook cb) {
    static std::vector<uint16_t> luts[3];
    int idx = (int)cb;
    if (luts[idx].empty()) {
        luts[idx].resize(1u << 16);
        exl3_codebook_lut(cb, luts[idx].data());
    }
    return luts[idx].data();
}

}  // namespace

void exl3_codebook_lut(Exl3Codebook cb, uint16_t* lut) {
    for (int w = 0; w < (1 << 16); ++w) lut[w] = decode_window(cb, (uint16_t)w);
}

void exl3_decode_tile(const uint16_t* trellis_words, int bits, const uint16_t* lut,
                      uint16_t* out_row_major) {
    int perm[256];
    build_perm(perm);
    uint16_t tc[256];
    for (int t = 0; t < 256; ++t) {
        int start = t * bits + bits - 16;
        int total = 256 * bits;
        start = ((start % total) + total) % total;
        tc[t] = lut[window16(trellis_words, 256 * bits / 16, start)];
    }
    for (int j = 0; j < 256; ++j) out_row_major[perm[j]] = tc[j];
}

void exl3_decode_weight_hat(const uint16_t* trellis, int ki, int nj, int bits, Exl3Codebook cb,
                            uint16_t* out) {
    const uint16_t* lut = lut_for(cb);
    int words = 256 * bits / 16;
    int n = nj * 16;
    uint16_t tile[256];
    for (int i = 0; i < ki; ++i)
        for (int j = 0; j < nj; ++j) {
            exl3_decode_tile(trellis + ((size_t)(i * nj + j)) * words, bits, lut, tile);
            for (int r = 0; r < 16; ++r)
                for (int c = 0; c < 16; ++c)
                    out[(size_t)(i * 16 + r) * n + j * 16 + c] = tile[r * 16 + c];
        }
}

void exl3_reconstruct_weight(const uint16_t* trellis, int ki, int nj, int bits, Exl3Codebook cb,
                             const uint16_t* suh, const uint16_t* svh, uint16_t* out) {
    int k = ki * 16, n = nj * 16;
    std::vector<uint16_t> what((size_t)k * n);
    exl3_decode_weight_hat(trellis, ki, nj, bits, cb, what.data());
    std::vector<float> w((size_t)k * n);
    for (size_t i = 0; i < w.size(); ++i) w[i] = h2f(what[i]);
    had_rows(w, k, n);
    had_cols(w, k, n);
    for (int i = 0; i < k; ++i) {
        float su = h2f(suh[i]);
        for (int j = 0; j < n; ++j) out[(size_t)i * n + j] = f2h(w[(size_t)i * n + j] * su * h2f(svh[j]));
    }
}

void exl3_folded_gemv(const uint16_t* trellis, int ki, int nj, int bits, Exl3Codebook cb,
                      const uint16_t* suh, const uint16_t* svh, const uint16_t* x, int tokens,
                      float* y) {
    int k = ki * 16, n = nj * 16;
    std::vector<uint16_t> what((size_t)k * n);
    exl3_decode_weight_hat(trellis, ki, nj, bits, cb, what.data());
    std::vector<float> w((size_t)k * n);
    for (size_t i = 0; i < w.size(); ++i) w[i] = h2f(what[i]);
    std::vector<float> xh((size_t)tokens * k);
    for (int t = 0; t < tokens; ++t)
        for (int i = 0; i < k; ++i) xh[(size_t)t * k + i] = h2f(x[(size_t)t * k + i]) * h2f(suh[i]);
    had_cols(xh, tokens, k);
    for (int t = 0; t < tokens; ++t) {
        for (int j = 0; j < n; ++j) {
            float acc = 0;
            for (int i = 0; i < k; ++i) acc += xh[(size_t)t * k + i] * w[(size_t)i * n + j];
            y[(size_t)t * n + j] = acc;
        }
    }
    std::vector<float> z((size_t)tokens * n);
    for (size_t i = 0; i < z.size(); ++i) z[i] = y[i];
    had_cols(z, tokens, n);
    for (int t = 0; t < tokens; ++t)
        for (int j = 0; j < n; ++j) y[(size_t)t * n + j] = z[(size_t)t * n + j] * h2f(svh[j]);
}

void exl3_ngram_decode_row(const uint16_t* ring, int K, const uint16_t* lut, int dim,
                           const float* bias, float* out) {
    // word 0 is the fp16 scale; words 1.. hold the dim*K-bit ring.  Stream bits [p*K, (p+1)*K) are
    // the low K bits of position p's symbol; position i's 16-bit state stacks symbols i, i-1, ...
    const uint16_t* stream = ring + 1;
    int total_bits = dim * K;
    float scale = h2f(ring[0]);
    int nsym = (15 + K) / K;
    for (int i = 0; i < dim; ++i) {
        uint32_t state = 0;
        for (int j = 0; j < nsym; ++j) {
            int p = ((i - j) % dim + dim) % dim;
            int b0 = p * K;
            uint32_t sym = 0;
            for (int q = 0; q < K; ++q) {
                int b = (b0 + q) % total_bits;
                sym |= (uint32_t)((stream[b >> 4] >> (b & 15)) & 1) << q;
            }
            state |= sym << (j * K);
        }
        state &= 0xFFFF;
        float v = h2f(lut[state]) * scale;
        out[i] = bias ? v + bias[i] : v;
    }
}

}  // namespace strata::kernels::cpu
