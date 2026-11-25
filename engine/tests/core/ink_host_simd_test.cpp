// ink_host_simd_test: the AVX2 and AVX-512 host kernels of the Inkling fast lane (CPU computes cold experts) against a plain C++ reference,
// for every format they handle: SF4, MX8 and the S1.5 / S2_K .. S8_K sub-groups.
//   - S_K sub-groups: the integer sums are exact in every tier and the float steps are the same, so the results agree to float rounding
//     (a compiler may fuse x_d * isum - c * x_s into an FMA in one TU and not in another: a tolerance of 2^-21 of the terms).
//   - SF4 / MX8: the lanes sum in a different order per tier, so the check is a tolerance (1e-5 of the sum of the absolute terms).
//   - The F8_E4M3 decode of the AVX-512 MX8 kernel is checked on all 256 codes (a row holding every code, x = 1, scale 1).
// A tier the CPU cannot run is reported as skipped; the test still passes on a CPU with AVX2 only or with neither (the kernels' own
// available() gate is what a real run uses, so this also proves the gate returns false and nothing is executed).  Run it under
// `qemu-x86_64 -cpu EPYC-Genoa` to execute the AVX-512 path on a CPU that has none.
#include "ink_host_avx2.h"
#include "ink_host_avx512.h"

#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <random>
#include <vector>

namespace {

int g_fail = 0;
#define CHECK(c, ...) do { if (!(c)) { ++g_fail; std::printf("FAIL %s:%d ", __FILE__, __LINE__); std::printf(__VA_ARGS__); std::printf("\n"); } } while (0)

float e4m3(int c) {
    const int s = c >> 7, e = (c >> 3) & 15, m = c & 7;
    float v;
    if (e == 15 && m == 7) v = 0.0f;
    else if (e == 0) v = (float)m * (1.0f / 512.0f);
    else v = (1.0f + (float)m * 0.125f) * std::ldexp(1.0f, e - 7);
    return s ? -v : v;
}

float ue8m0(int e) {
    if (e == 255) return 0.0f;
    if (e == 0) return 5.877471754111438e-39f;
    uint32_t u = (uint32_t)e << 23;
    float f;
    std::memcpy(&f, &u, 4);
    return f;
}

struct Luts {
    float e4[256];
    float u8[256];
    int8_t s15[256][5];
    Luts() {
        for (int c = 0; c < 256; ++c) { e4[c] = e4m3(c); u8[c] = ue8m0(c); }
        for (int b = 0; b < 256; ++b) {
            int v = b;
            for (int k = 0; k < 5; ++k) { s15[b][k] = (int8_t)((v % 3) - 1); v /= 3; }
        }
    }
};

// ---- plain C++ references (the generic branches of ink_hybrid.cuh) ----
// the exact integer sum of one 64-weight sub-group against the Q8 activations (every tier computes this exactly)
int32_t ref_sk_isum(const Luts& L, const uint8_t* pl, int fmt, int g, const int8_t* qs) {
    int32_t isum = 0;
    if (fmt == 1) {
        int8_t tr[64];
        const int base = g * 64;
        for (int i = 0; i < 64; ++i) {
            const int idx = base + i;
            tr[i] = L.s15[pl[idx / 5]][idx % 5];
        }
        // the kernels bias the trits to u8 (+1) and subtract x_s (= x_d * sum(qs) for real activations): isum is of the biased weights
        for (int i = 0; i < 64; ++i) isum += ((int32_t)tr[i] + 1) * (int32_t)qs[i];
        return isum;
    }
    if (fmt == 2) {
        const uint8_t* p = pl + g * 16;
        for (int i = 0; i < 16; ++i) {
            const uint8_t b = p[i];
            isum += (int32_t)(b & 3u) * qs[4 * i + 0] + (int32_t)((b >> 2) & 3u) * qs[4 * i + 1]
                  + (int32_t)((b >> 4) & 3u) * qs[4 * i + 2] + (int32_t)(b >> 6) * qs[4 * i + 3];
        }
        return isum;
    }
    if (fmt == 3) {
        const uint8_t* p = pl + g * 16;
        const uint8_t* hp = pl + 64 + g * 8;
        for (int i = 0; i < 8; ++i) {
            const uint8_t b0 = p[2 * i], b1 = p[2 * i + 1], h = hp[i];
            const uint8_t u[8] = {
                (uint8_t)((b0 & 3u) | ((h & 1u) << 2)), (uint8_t)(((b0 >> 2) & 3u) | (((h >> 1) & 1u) << 2)),
                (uint8_t)(((b0 >> 4) & 3u) | (((h >> 2) & 1u) << 2)), (uint8_t)((b0 >> 6) | (((h >> 3) & 1u) << 2)),
                (uint8_t)((b1 & 3u) | (((h >> 4) & 1u) << 2)), (uint8_t)(((b1 >> 2) & 3u) | (((h >> 5) & 1u) << 2)),
                (uint8_t)(((b1 >> 4) & 3u) | (((h >> 6) & 1u) << 2)), (uint8_t)((b1 >> 6) | (((h >> 7) & 1u) << 2))};
            for (int k = 0; k < 8; ++k) isum += (int32_t)u[k] * qs[8 * i + k];
        }
        return isum;
    }
    if (fmt == 4) {
        const uint8_t* p = pl + g * 32;
        for (int i = 0; i < 32; ++i) isum += (int32_t)(p[i] & 0xFu) * qs[2 * i] + (int32_t)(p[i] >> 4) * qs[2 * i + 1];
        return isum;
    }
    if (fmt == 5) {
        const uint8_t* p = pl + g * 32;
        const uint8_t* hp = pl + 128 + g * 8;
        for (int i = 0; i < 8; ++i) {
            const uint8_t h = hp[i];
            for (int k = 0; k < 4; ++k) {
                const uint8_t b = p[4 * i + k];
                isum += (int32_t)((b & 0xFu) | (((h >> (2 * k)) & 1u) << 4)) * qs[8 * i + 2 * k]
                      + (int32_t)((b >> 4) | (((h >> (2 * k + 1)) & 1u) << 4)) * qs[8 * i + 2 * k + 1];
            }
        }
        return isum;
    }
    if (fmt == 6) {
        const uint8_t* p = pl + g * 32;
        const uint8_t* hp = pl + 128 + g * 16;
        for (int i = 0; i < 16; ++i) {
            const uint8_t h = hp[i], b0 = p[2 * i], b1 = p[2 * i + 1];
            isum += (int32_t)((b0 & 0xFu) | ((h & 3u) << 4)) * qs[4 * i + 0]
                  + (int32_t)((b0 >> 4) | (((h >> 2) & 3u) << 4)) * qs[4 * i + 1]
                  + (int32_t)((b1 & 0xFu) | (((h >> 4) & 3u) << 4)) * qs[4 * i + 2]
                  + (int32_t)((b1 >> 4) | ((h >> 6) << 4)) * qs[4 * i + 3];
        }
        return isum;
    }
    const int8_t* p = reinterpret_cast<const int8_t*>(pl + g * 64);
    for (int i = 0; i < 64; ++i) isum += (int32_t)p[i] * qs[i];
    return isum;
}

// double-precision reference + the sum of the absolute terms (for the tolerance)
void ref_sf4(const Luts& L, const uint8_t* p, const uint8_t* sc, float w_outer, const float* x, int cols, double& val, double& mag) {
    static const double e2m1[16] = {0, .5, 1, 1.5, 2, 3, 4, 6, -0., -.5, -1, -1.5, -2, -3, -4, -6};
    val = 0; mag = 0;
    for (int b = 0; b < cols / 16; ++b) {
        double bs = 0, bm = 0;
        for (int i = 0; i < 8; ++i) {
            const uint8_t byte = p[b * 8 + i];
            const double a = e2m1[byte & 0xF] * x[b * 16 + 2 * i], c = e2m1[byte >> 4] * x[b * 16 + 2 * i + 1];
            bs += a + c; bm += std::fabs(a) + std::fabs(c);
        }
        val += bs * L.e4[sc[b]];
        mag += bm * std::fabs(L.e4[sc[b]]);
    }
    val *= w_outer; mag *= std::fabs(w_outer);
}

void ref_mx8(const Luts& L, const uint8_t* w, const uint8_t* sc, const float* x, int cols, double& val, double& mag) {
    val = 0; mag = 0;
    for (int b = 0; b < cols / 32; ++b) {
        double bs = 0, bm = 0;
        for (int i = 0; i < 32; ++i) {
            const double a = (double)L.e4[w[b * 32 + i]] * x[b * 32 + i];
            bs += a; bm += std::fabs(a);
        }
        val += bs * L.u8[sc[b]];
        mag += bm * std::fabs(L.u8[sc[b]]);
    }
}

}  // namespace

int main() {
    using namespace strata::ink;
    const bool has2 = host_avx2::available();
    const bool has5 = host_avx512::available();
    std::printf("ink_host_simd_test: avx2 %s, avx512 %s\n", has2 ? "yes" : "no (skipped)", has5 ? "yes" : "no (skipped)");
    if (!has2) { std::printf("the AVX2 gate is closed here: nothing to execute, nothing to compare\n"); }

    static const Luts L;
    std::mt19937 rng(12345);
    auto rnd8 = [&]() { return (uint8_t)(rng() & 0xFF); };
    auto rndf = [&](float lo, float hi) { return lo + (hi - lo) * (float)((rng() >> 8) & 0xFFFF) / 65535.0f; };

    // --- S_K sub-groups: the integer sum is exact in every tier; the float result is within the rounding of the last two float operations ---
    const int fmts[] = {1, 2, 3, 4, 5, 6, 8};
    long n_sk = 0;
    for (int fmt : fmts) {
        for (int iter = 0; iter < 400; ++iter) {
            alignas(64) uint8_t pl[256 + 64];
            for (auto& b : pl) b = rnd8();
            alignas(64) int8_t qs[64];
            for (auto& q : qs) q = (int8_t)((int)(rng() % 255) - 127);
            if (iter % 7 == 0) for (auto& q : qs) q = (iter % 14 == 0) ? 127 : -127;      // the extreme sums
            if (iter % 11 == 0) for (int i = 0; i < 64; ++i) pl[i + 0] = 0xFF;              // all-ones payload bytes
            const float x_d = rndf(1e-4f, 0.05f), x_s = rndf(-3.0f, 3.0f), eff = rndf(-0.5f, 0.5f);
            for (int g = 0; g < 4; ++g) {
                static const double kC[9] = {0, 1.0, 1.5, 3.5, 7.5, 15.5, 31.5, 0, 0};
                const double isum = (double)ref_sk_isum(L, pl, fmt, g, qs);
                const double refd = (double)eff * ((double)x_d * isum - kC[fmt] * (double)x_s);
                // 2^-21 of the terms: covers the one rounding that an FMA contraction (the compiler fuses x_d * isum - c * x_s) changes
                const double tol = 4.8e-7 * std::fabs((double)eff) * (std::fabs((double)x_d * isum) + kC[fmt] * std::fabs((double)x_s)) + 1e-30;
                float a = 0.0f, c = 0.0f;
                if (has2) {
                    a = host_avx2::dot_sk_sub64(pl, fmt, g, qs, x_d, x_s, eff, L.s15);
                    CHECK(std::fabs((double)a - refd) <= tol, "avx2 S_K fmt %d g %d: %.9g vs ref %.9g", fmt, g, a, refd);
                }
                if (has5) {
                    c = host_avx512::dot_sk_sub64(pl, fmt, g, qs, x_d, x_s, eff, L.s15);
                    CHECK(std::fabs((double)c - refd) <= tol, "avx512 S_K fmt %d g %d: %.9g vs ref %.9g", fmt, g, c, refd);
                }
                if (has2 && has5) CHECK(std::fabs((double)a - (double)c) <= tol, "avx2 %.9g vs avx512 %.9g (S_K fmt %d g %d)", a, c, fmt, g);
                ++n_sk;
            }
        }
    }
    std::printf("S_K sub-groups compared: %ld per tier\n", n_sk);

    // --- SF4 rows (cols = 16 * nb, odd nb too) ---
    int n_sf4 = 0;
    for (int nb : {1, 2, 3, 7, 8, 71, 72, 143, 144}) {
        for (int iter = 0; iter < 30; ++iter) {
            const int cols = nb * 16;
            std::vector<uint8_t> p(cols / 2 + 16), sc(nb + 2);
            for (auto& b : p) b = rnd8();
            for (auto& b : sc) b = (uint8_t)(100 + rng() % 40);   // finite, mid-range E4M3 scale codes
            std::vector<float> x(cols + 16);
            for (auto& v : x) v = rndf(-2.0f, 2.0f);
            const float w_outer = rndf(0.01f, 2.0f);
            double rv, rm;
            ref_sf4(L, p.data(), sc.data(), w_outer, x.data(), cols, rv, rm);
            const double tol = 1e-5 * rm + 1e-30;
            if (has2) { const double a = host_avx2::dot_sf4_row(p.data(), sc.data(), w_outer, x.data(), cols, L.e4); CHECK(std::fabs(a - rv) <= tol, "avx2 sf4 nb %d: %.9g vs %.9g", nb, a, rv); }
            if (has5) { const double c = host_avx512::dot_sf4_row(p.data(), sc.data(), w_outer, x.data(), cols, L.e4); CHECK(std::fabs(c - rv) <= tol, "avx512 sf4 nb %d: %.9g vs %.9g", nb, c, rv); }
            ++n_sf4;
        }
    }
    std::printf("SF4 rows compared: %d per tier\n", n_sf4);

    // --- MX8 rows ---
    int n_mx8 = 0;
    for (int nb : {1, 2, 3, 36, 72, 73}) {
        for (int iter = 0; iter < 30; ++iter) {
            const int cols = nb * 32;
            std::vector<uint8_t> w(cols + 32), sc(nb + 2);
            for (auto& b : w) b = rnd8();
            for (auto& b : sc) b = (uint8_t)(110 + rng() % 30);
            std::vector<float> x(cols + 32);
            for (auto& v : x) v = rndf(-2.0f, 2.0f);
            double rv, rm;
            ref_mx8(L, w.data(), sc.data(), x.data(), cols, rv, rm);
            const double tol = 1e-5 * rm + 1e-30;
            if (has2) { const double a = host_avx2::dot_mx8_row(w.data(), sc.data(), x.data(), cols, L.e4, L.u8); CHECK(std::fabs(a - rv) <= tol, "avx2 mx8 nb %d: %.9g vs %.9g", nb, a, rv); }
            if (has5) { const double c = host_avx512::dot_mx8_row(w.data(), sc.data(), x.data(), cols, L.e4, L.u8); CHECK(std::fabs(c - rv) <= tol, "avx512 mx8 nb %d: %.9g vs %.9g", nb, c, rv); }
            ++n_mx8;
        }
    }
    // every E4M3 code once: a row of the 256 codes against x = 1 with scale exp 127 (2^0): the sum is the sum of the table
    if (has5) {
        std::vector<uint8_t> w(256), sc(8, 127);
        for (int c = 0; c < 256; ++c) w[c] = (uint8_t)c;
        std::vector<float> x(256, 1.0f);
        double rv, rm;
        ref_mx8(L, w.data(), sc.data(), x.data(), 256, rv, rm);
        const double c = host_avx512::dot_mx8_row(w.data(), sc.data(), x.data(), 256, L.e4, L.u8);
        CHECK(std::fabs(c - rv) <= 1e-5 * rm, "avx512 mx8 all 256 codes: %.9g vs %.9g", c, rv);
        // and each code on its own (one nonzero x lane), so a wrong lane or a wrong single code cannot cancel out
        for (int code = 0; code < 256; ++code) {
            std::vector<uint8_t> w1(32, 0);
            std::vector<float> x1(32, 0.0f);
            w1[5] = (uint8_t)code; x1[5] = 1.0f;
            uint8_t sc1 = 127;
            const float got = host_avx512::dot_mx8_row(w1.data(), &sc1, x1.data(), 32, L.e4, L.u8);
            const float want = L.e4[code];
            CHECK(std::memcmp(&got, &want, 4) == 0 || (got == 0.0f && want == 0.0f), "avx512 e4m3 code %d: %.9g vs %.9g", code, got, want);
        }
    }
    std::printf("MX8 rows compared: %d per tier\n", n_mx8);

    if (g_fail) { std::printf("ink_host_simd_test: %d FAILED\n", g_fail); return 1; }
    std::printf("ink_host_simd_test: PASS\n");
    return 0;
}
