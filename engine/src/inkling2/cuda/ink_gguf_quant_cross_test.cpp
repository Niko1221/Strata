// engine/src/inkling2/cuda/ink_gguf_quant_cross_test.cpp - host test: the spec-based reference dequantisation used by ink_gguf_types_test.cu against
// strata::quant::dequant_row (the engine's CPU path), on random but valid blocks of IQ2_XXS, IQ2_XS, IQ2_S, IQ3_S, IQ1_S, IQ3_XXS, IQ4_NL and Q2_0.
// Two independent readings of the formats must give the same numbers; a mismatch means one of them misreads a layout.
//
// build (from engine/): g++ -O2 -std=c++20 -I include -I src/quant src/inkling2/cuda/ink_gguf_quant_cross_test.cpp src/quant/dequant.cpp src/quant/types.cpp -o cross_test
//
// MIT License
// Copyright (c) 2026 Niko1221 and the Strata contributors
#define GGML_COMMON_IMPL_CPP
#include "strata/quant/strata-quant-common.h"
#include "strata/quant/dequant.hpp"

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <random>
#include <vector>

static float h_f16(uint16_t h) {
    const uint32_t s = (h >> 15) & 1u, e = (h >> 10) & 0x1Fu, m = h & 0x3FFu;
    double v;
    if (e == 0) v = std::ldexp((double)m, -24);
    else if (e == 31) v = 65504.0;
    else v = std::ldexp(1.0 + m / 1024.0, (int)e - 15);
    return (float)(s ? -v : v);
}
static uint16_t f16_bits(float f) {   // round to nearest, normal range only (test scales are 1e-3 .. 0.1)
    int ex; double fr = std::frexp((double)std::fabs(f), &ex);   // f = fr * 2^ex, fr in [0.5, 1)
    int e = ex - 1 + 15;
    int m = (int)std::lround((fr * 2.0 - 1.0) * 1024.0);
    if (m == 1024) { m = 0; ++e; }
    return (uint16_t)((f < 0 ? 0x8000u : 0u) | ((uint32_t)e << 10) | (uint32_t)m);
}

struct Tabs {
    std::vector<uint8_t> kmask, ksigns;
    std::vector<uint64_t> g2xxs, g2xs, g2s, g1s;
    std::vector<uint32_t> g3s, g3xxs;
};

static size_t blk_bytes(int t) {
    switch (t) { case 16: return 66; case 17: return 74; case 22: return 82; case 21: return 110; case 19: return 50; case 18: return 98; case 42: return 18; case 20: return 18; }
    return 0;
}
static int blk_elems(int t) { return t == 42 ? 64 : (t == 20 ? 32 : 256); }

// reference dequantisation of one row of K elements, from the block specs
static void ref_row(int t, const uint8_t* src, int K, const Tabs& T, double* y) {
    const int nb = K / blk_elems(t);
    for (int i = 0; i < nb; ++i) {
        const uint8_t* b = src + (size_t)i * blk_bytes(t);
        uint16_t dh; std::memcpy(&dh, b, 2);
        const double d = h_f16(dh);
        double* o = y + (size_t)i * blk_elems(t);
        if (t == 42) {
            for (int j = 0; j < 64; ++j) o[j] = (double)(((b[2 + j / 4] >> ((j % 4) * 2)) & 3) - 1) * d;
        } else if (t == 16) {
            for (int ib = 0; ib < 8; ++ib) {
                uint32_t a0, a1; std::memcpy(&a0, b + 2 + 8 * ib, 4); std::memcpy(&a1, b + 2 + 8 * ib + 4, 4);
                const double db = d * (0.5 + (a1 >> 28)) * 0.25;
                for (int l = 0; l < 4; ++l) {
                    const uint64_t g = T.g2xxs[(a0 >> (8 * l)) & 255];
                    const uint8_t sg = T.ksigns[(a1 >> (7 * l)) & 127];
                    for (int j = 0; j < 8; ++j) o[ib * 32 + l * 8 + j] = db * (double)((g >> (8 * j)) & 255) * ((sg & T.kmask[j]) ? -1.0 : 1.0);
                }
            }
        } else if (t == 17) {
            for (int ib = 0; ib < 8; ++ib) {
                const uint8_t sc = b[66 + ib];
                for (int l = 0; l < 4; ++l) {
                    uint16_t qv; std::memcpy(&qv, b + 2 + 8 * ib + 2 * l, 2);
                    const double db = d * (0.5 + ((l < 2) ? (sc & 15) : (sc >> 4))) * 0.25;
                    const uint64_t g = T.g2xs[qv & 511];
                    const uint8_t sg = T.ksigns[qv >> 9];
                    for (int j = 0; j < 8; ++j) o[ib * 32 + l * 8 + j] = db * (double)((g >> (8 * j)) & 255) * ((sg & T.kmask[j]) ? -1.0 : 1.0);
                }
            }
        } else if (t == 22) {
            const uint8_t* qs = b + 2; const uint8_t* qh = b + 66; const uint8_t* sgs = qs + 32; const uint8_t* scl = b + 74;
            for (int ib = 0; ib < 8; ++ib) {
                for (int l = 0; l < 4; ++l) {
                    const double db = d * (0.5 + ((l < 2) ? (scl[ib] & 15) : (scl[ib] >> 4))) * 0.25;
                    const uint64_t g = T.g2s[qs[ib * 4 + l] | ((qh[ib] << (8 - 2 * l)) & 0x300)];
                    const uint8_t sg = sgs[ib * 4 + l];
                    for (int j = 0; j < 8; ++j) o[ib * 32 + l * 8 + j] = db * (double)((g >> (8 * j)) & 255) * ((sg & T.kmask[j]) ? -1.0 : 1.0);
                }
            }
        } else if (t == 21) {
            const uint8_t* qs = b + 2; const uint8_t* qh = b + 66; const uint8_t* sgs = b + 74; const uint8_t* scl = b + 106;
            for (int ib = 0; ib < 8; ++ib) {
                const double db = d * (1 + 2 * ((ib & 1) ? (scl[ib / 2] >> 4) : (scl[ib / 2] & 15)));
                for (int l = 0; l < 4; ++l) {
                    const uint32_t g1 = T.g3s[qs[ib * 8 + 2 * l] | ((qh[ib] << (8 - 2 * l)) & 256)];
                    const uint32_t g2 = T.g3s[qs[ib * 8 + 2 * l + 1] | ((qh[ib] << (7 - 2 * l)) & 256)];
                    const uint8_t sg = sgs[ib * 4 + l];
                    for (int j = 0; j < 4; ++j) {
                        o[ib * 32 + l * 8 + j]     = db * (double)((g1 >> (8 * j)) & 255) * ((sg & T.kmask[j]) ? -1.0 : 1.0);
                        o[ib * 32 + l * 8 + j + 4] = db * (double)((g2 >> (8 * j)) & 255) * ((sg & T.kmask[j + 4]) ? -1.0 : 1.0);
                    }
                }
            }
        } else if (t == 19) {
            for (int ib = 0; ib < 8; ++ib) {
                uint16_t qh; std::memcpy(&qh, b + 34 + 2 * ib, 2);
                const double dl = d * (2 * ((qh >> 12) & 7) + 1);
                const double delta = (qh & 0x8000) ? -0.125 : 0.125;
                for (int l = 0; l < 4; ++l) {
                    const uint64_t g = T.g1s[b[2 + 4 * ib + l] | (((qh >> (3 * l)) & 7) << 8)];
                    for (int j = 0; j < 8; ++j) o[ib * 32 + l * 8 + j] = dl * ((double)(int8_t)((g >> (8 * j)) & 255) + delta);
                }
            }
        } else if (t == 18) {
            for (int ib = 0; ib < 8; ++ib) {
                uint32_t aux; std::memcpy(&aux, b + 66 + 4 * ib, 4);
                const double db = d * (0.5 + (aux >> 28)) * 0.5;
                for (int l = 0; l < 4; ++l) {
                    const uint8_t sg = T.ksigns[(aux >> (7 * l)) & 127];
                    const uint32_t g1 = T.g3xxs[b[2 + ib * 8 + 2 * l]], g2 = T.g3xxs[b[2 + ib * 8 + 2 * l + 1]];
                    for (int j = 0; j < 4; ++j) {
                        o[ib * 32 + l * 8 + j]     = db * (double)((g1 >> (8 * j)) & 255) * ((sg & T.kmask[j]) ? -1.0 : 1.0);
                        o[ib * 32 + l * 8 + j + 4] = db * (double)((g2 >> (8 * j)) & 255) * ((sg & T.kmask[j + 4]) ? -1.0 : 1.0);
                    }
                }
            }
        } else if (t == 20) {
            static const int lut[16] = {-127, -104, -83, -65, -49, -35, -22, -10, 1, 13, 25, 38, 53, 69, 89, 113};
            for (int j = 0; j < 16; ++j) { o[j] = d * lut[b[2 + j] & 15]; o[j + 16] = d * lut[b[2 + j] >> 4]; }
        }
    }
}


int main() {
    Tabs T;
    T.kmask.assign(kmask_iq2xs, kmask_iq2xs + 8);
    T.ksigns.assign(ksigns_iq2xs, ksigns_iq2xs + 128);
    T.g2xxs.assign(iq2xxs_grid, iq2xxs_grid + 256);
    T.g2xs.assign(iq2xs_grid, iq2xs_grid + 512);
    T.g2s.assign(iq2s_grid, iq2s_grid + 1024);
    T.g1s.assign(iq1s_grid, iq1s_grid + 2048);
    T.g3s.assign(iq3s_grid, iq3s_grid + 512);
    T.g3xxs.assign(iq3xxs_grid, iq3xxs_grid + 256);
    std::mt19937 rng(20261126);
    int fails = 0;
    const int types[] = {16, 17, 22, 21, 19, 18, 42, 20};
    for (int t : types) {
        const int K = (t == 42 || t == 20) ? 1152 : 2304;
        const size_t rb = (size_t)(K / blk_elems(t)) * blk_bytes(t);
        const int N = 64;
        std::vector<uint8_t> w(rb * N);
        for (auto& b : w) b = (uint8_t)(rng() & 255);
        std::uniform_real_distribution<float> dsc(0.002f, 0.05f);
        for (int r = 0; r < N; ++r)
            for (int i = 0; i < K / blk_elems(t); ++i) {
                float s = dsc(rng);
                if (t == 42 && (rng() & 1)) s = -s;
                const uint16_t h = f16_bits(s);
                std::memcpy(&w[(size_t)r * rb + (size_t)i * blk_bytes(t)], &h, 2);
            }
        double worst = 0, scale = 0;
        std::vector<double> y(K);
        std::vector<float> z(K);
        for (int r = 0; r < N; ++r) {
            ref_row(t, &w[(size_t)r * rb], K, T, y.data());
            if (!strata::quant::dequant_row((strata::quant::Type)t, &w[(size_t)r * rb], z.data(), K)) { std::printf("type %d: dequant_row refused\n", t); ++fails; break; }
            for (int i = 0; i < K; ++i) { worst = std::max(worst, std::fabs(y[i] - (double)z[i])); scale = std::max(scale, std::fabs(y[i])); }
        }
        std::printf("type %2d: max |reference - strata::quant| %.3g (largest value %.3g) %s\n", t, worst, scale, worst > 1e-5 * (scale + 1e-30) ? "MISMATCH" : "ok");
        if (worst > 1e-5 * (scale + 1e-30)) ++fails;
    }
    std::printf(fails ? "FAILED: %d\n" : "ALL TYPES AGREE\n", fails);
    return fails ? 1 : 0;
}
