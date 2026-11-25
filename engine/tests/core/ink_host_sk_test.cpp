// strata-inkling host S_K kernels (ink_host_avx2.cpp): the whole-row SIMD kernels give the same sums as the per-group kernel (bit for bit,
// same order of float adds) and agree with a plain scalar reference. Random payloads, scales and activations; the formats 2 .. 6; T = 1, 3, 16 tokens.
// No GPU. On a CPU without AVX2 (or a build without the AVX2 unit) the kernels are unavailable and the test passes by skipping.
#include "ink_host_avx2.h"

#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <vector>

using namespace strata::ink;

namespace {
int failed = 0;
#define CHECK(x) do { if (!(x)) { std::fprintf(stderr, "FAIL %s:%d: %s\n", __FILE__, __LINE__, #x); ++failed; } } while (0)

int sb_bytes(int fmt) {
    switch (fmt) { case 2: return 70; case 3: return 102; case 4: return 134; case 5: return 166; case 6: return 198; default: return 0; }
}
float cst_of(int fmt) {
    switch (fmt) { case 2: return 1.5f; case 3: return 3.5f; case 4: return 7.5f; case 5: return 15.5f; default: return 31.5f; }
}

struct Xq {                      // InkQ8_1_G64
    int8_t qs[64];
    float d, s;
    float pad[2];
};
static_assert(sizeof(Xq) == 80, "80 bytes");

uint32_t rng_state = 12345u;
uint32_t rnd() { rng_state = rng_state * 1664525u + 1013904223u; return rng_state >> 8; }

// scalar reference of one group (the formulas of ink_hybrid.cuh's dot_sk_sub64_h, integer part exact)
float ref_group(const uint8_t* pl, int fmt, int g, const Xq& x, float eff) {
    const int8_t* qs = x.qs;
    int32_t isum = 0;
    if (fmt == 2) {
        const uint8_t* p = pl + g * 16;
        for (int i = 0; i < 16; ++i) {
            const uint8_t b = p[i];
            isum += (int32_t)(b & 3u) * qs[4 * i] + (int32_t)((b >> 2) & 3u) * qs[4 * i + 1] + (int32_t)((b >> 4) & 3u) * qs[4 * i + 2] + (int32_t)(b >> 6) * qs[4 * i + 3];
        }
    } else if (fmt == 3) {
        const uint8_t* p = pl + g * 16;
        const uint8_t* hp = pl + 64 + g * 8;
        for (int i = 0; i < 8; ++i) {
            const uint8_t b0 = p[2 * i], b1 = p[2 * i + 1], h = hp[i];
            for (int k = 0; k < 4; ++k) {
                isum += (int32_t)(((b0 >> (2 * k)) & 3u) | (((h >> k) & 1u) << 2)) * qs[8 * i + k];
                isum += (int32_t)(((b1 >> (2 * k)) & 3u) | (((h >> (4 + k)) & 1u) << 2)) * qs[8 * i + 4 + k];
            }
        }
    } else if (fmt == 4) {
        const uint8_t* p = pl + g * 32;
        for (int i = 0; i < 32; ++i) isum += (int32_t)(p[i] & 0xFu) * qs[2 * i] + (int32_t)(p[i] >> 4) * qs[2 * i + 1];
    } else if (fmt == 5) {
        const uint8_t* p = pl + g * 32;
        const uint8_t* hp = pl + 128 + g * 8;
        for (int i = 0; i < 8; ++i)
            for (int k = 0; k < 4; ++k) {
                const uint8_t b = p[4 * i + k];
                isum += (int32_t)((b & 0xFu) | (((hp[i] >> (2 * k)) & 1u) << 4)) * qs[8 * i + 2 * k];
                isum += (int32_t)((b >> 4) | (((hp[i] >> (2 * k + 1)) & 1u) << 4)) * qs[8 * i + 2 * k + 1];
            }
    } else {
        const uint8_t* p = pl + g * 32;
        const uint8_t* hp = pl + 128 + g * 16;
        for (int i = 0; i < 16; ++i) {
            const uint8_t h = hp[i], b0 = p[2 * i], b1 = p[2 * i + 1];
            isum += (int32_t)((b0 & 0xFu) | ((h & 3u) << 4)) * qs[4 * i] + (int32_t)((b0 >> 4) | (((h >> 2) & 3u) << 4)) * qs[4 * i + 1]
                  + (int32_t)((b1 & 0xFu) | (((h >> 4) & 3u) << 4)) * qs[4 * i + 2] + (int32_t)((b1 >> 4) | ((h >> 6) << 4)) * qs[4 * i + 3];
        }
    }
    return eff * (x.d * (float)isum - cst_of(fmt) * x.s);
}

float f16_to_f32(uint16_t h) {
    const uint32_t s = (uint32_t)(h & 0x8000u) << 16, e = (h >> 10) & 0x1Fu, m = h & 0x03FFu;
    uint32_t f;
    if (e == 0) f = s;                                   // test scales are normal numbers
    else f = s | ((e + 112u) << 23) | (m << 13);
    float out;
    std::memcpy(&out, &f, 4);
    return out;
}

void fill_row(std::vector<uint8_t>& row, int sb) {
    for (auto& b : row) b = (uint8_t)rnd();
    for (int s = 0; s < 9; ++s) {                        // a normal f16 scale around 1e-2 .. 1
        const uint16_t h = (uint16_t)(0x2000u + (rnd() % 0x1800u));
        std::memcpy(row.data() + (size_t)s * sb, &h, 2);
    }
}

void fill_x(std::vector<Xq>& x) {
    for (auto& b : x) {
        int sum = 0;
        for (int i = 0; i < 64; ++i) { b.qs[i] = (int8_t)((int)(rnd() % 255u) - 127); sum += b.qs[i]; }
        b.d = 0.001f + (float)(rnd() % 1000u) * 1e-5f;
        b.s = b.d * (float)sum;
    }
}
}  // namespace

int main() {
    if (!host_avx2::available()) { std::printf("ink_host_sk_test: no AVX2 here, skipped\n"); return 0; }
    for (int fmt = 2; fmt <= 6; ++fmt) {
        const int sb = sb_bytes(fmt);
        for (int T : {1, 3, 16}) {
            std::vector<uint8_t> row((size_t)9 * sb + 64);
            std::vector<std::vector<Xq>> xs((size_t)T, std::vector<Xq>(36));
            const uint8_t* xp[16];
            for (int round = 0; round < 40; ++round) {
                fill_row(row, sb);
                for (int t = 0; t < T; ++t) { fill_x(xs[(size_t)t]); xp[t] = reinterpret_cast<const uint8_t*>(xs[(size_t)t].data()); }
                // ---- one row, 36 groups -> the activation of a gate/up row has 36 blocks
                float out[16];
                CHECK(host_avx2::dot_sk_row_multi(row.data(), fmt, sb, 9, xp, T, out));
                for (int t = 0; t < T; ++t) {
                    float chain = 0.0f, ref = 0.0f;
                    for (int s = 0; s < 9; ++s) {
                        const uint8_t* b = row.data() + (size_t)s * sb;
                        uint16_t h; std::memcpy(&h, b, 2);
                        const float D = f16_to_f32(h);
                        for (int sg = 0; sg < 4; ++sg) {
                            const float eff = D * ((float)b[2 + sg] * (1.0f / 255.0f));
                            const Xq& x = xs[(size_t)t][(size_t)(s * 4 + sg)];
                            chain += host_avx2::dot_sk_sub64(b + 6, fmt, sg, x.qs, x.d, x.s, eff, nullptr);
                            ref += ref_group(b + 6, fmt, sg, x, eff);
                        }
                    }
                    CHECK(out[t] == chain);                              // the same sums in the same order as the per-group kernel
                    CHECK(std::fabs(out[t] - ref) <= 1e-4f * std::fabs(ref) + 1e-6f);   // and the scalar reference (float contraction may differ in the last bit)
                }
                // ---- a down-projection row pair: groups 0..17 -> row 2r, 18..35 -> row 2r+1, activation blocks 0..17 each
                float o0[16], o1[16];
                CHECK(host_avx2::dot_sk_pair_multi(row.data(), fmt, sb, xp, T, o0, o1));
                for (int t = 0; t < T; ++t) {
                    float c0 = 0.0f, c1 = 0.0f;
                    for (int s = 0; s < 9; ++s) {
                        const uint8_t* b = row.data() + (size_t)s * sb;
                        uint16_t h; std::memcpy(&h, b, 2);
                        const float D = f16_to_f32(h);
                        for (int sg = 0; sg < 4; ++sg) {
                            const int sub = s * 4 + sg;
                            const float eff = D * ((float)b[2 + sg] * (1.0f / 255.0f));
                            const Xq& x = xs[(size_t)t][(size_t)(sub < 18 ? sub : sub - 18)];
                            const float v = host_avx2::dot_sk_sub64(b + 6, fmt, sg, x.qs, x.d, x.s, eff, nullptr);
                            if (sub < 18) c0 += v; else c1 += v;
                        }
                    }
                    CHECK(o0[t] == c0);
                    CHECK(o1[t] == c1);
                }
            }
        }
    }
    // which arithmetic the per-group kernel's final expression has (diagnostic: a = fma(x_d, isum, -(c*x_s)), b = fma(-c, x_s, x_d*isum), c = unfused)
    {
        int ma = 0, mb = 0, mc = 0, n = 0;
        const int fmt = 4, sb = sb_bytes(4);
        std::vector<uint8_t> row((size_t)9 * sb + 64);
        std::vector<Xq> x(36);
        for (int round = 0; round < 200; ++round) {
            fill_row(row, sb);
            fill_x(x);
            for (int s = 0; s < 9; ++s) {
                const uint8_t* b = row.data() + (size_t)s * sb;
                uint16_t h; std::memcpy(&h, b, 2);
                const float D = f16_to_f32(h);
                for (int sg = 0; sg < 4; ++sg) {
                    const float eff = D * ((float)b[2 + sg] * (1.0f / 255.0f));
                    const Xq& xb = x[(size_t)(s * 4 + sg)];
                    const float v = host_avx2::dot_sk_sub64(b + 6, fmt, sg, xb.qs, xb.d, xb.s, eff, nullptr);
                    int32_t isum = 0;
                    const uint8_t* p = b + 6 + sg * 32;
                    for (int i = 0; i < 32; ++i) isum += (int32_t)(p[i] & 0xFu) * xb.qs[2 * i] + (int32_t)(p[i] >> 4) * xb.qs[2 * i + 1];
                    const float fi = (float)isum;
                    volatile float cs = 7.5f;
                    const float ca = eff * std::fma(xb.d, fi, -(cs * xb.s));
                    const float cb = eff * std::fma(-cs, xb.s, xb.d * fi);
                    volatile float p1 = xb.d * fi, p2 = cs * xb.s;
                    const float cc = eff * (p1 - p2);
                    ++n; ma += (ca == v); mb += (cb == v); mc += (cc == v);
                }
            }
        }
        std::printf("per-group final expression matches: a %d / b %d / c %d of %d\n", ma, mb, mc, n);
        CHECK(ma == n);   // the explicit form fma(x_d, isum, -(c * x_s)), which is what GCC's contraction gave before it was written out
    }
    // formats without a SIMD row kernel say so
    {
        std::vector<uint8_t> row(1024);
        std::vector<Xq> x(36);
        const uint8_t* xp[1] = {reinterpret_cast<const uint8_t*>(x.data())};
        float out[1];
        CHECK(!host_avx2::dot_sk_row_multi(row.data(), 8, 262, 9, xp, 1, out));
        CHECK(!host_avx2::dot_sk_row_multi(row.data(), 1, 58, 9, xp, 1, out));
    }
    if (failed) { std::fprintf(stderr, "%d checks failed\n", failed); return 1; }
    std::printf("ink_host_sk_test OK\n");
    return 0;
}
