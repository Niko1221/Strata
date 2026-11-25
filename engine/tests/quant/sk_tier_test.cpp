// sk_tier_test: the run-time ISA tiers of the S_K / SF4 / MX8 codecs (src/quant/sk_codecs.cpp compiled as base / avx2 / avx512, dispatched by sk_dispatch.cpp)
// against each other.  Every tier the CPU can run is compared with the base tier (plain C++):
//   - S_K: dequant, single-row dot, GEMM (m = 1, 5, 18: the 16-token tile boundary) and the paired-2-row GEMV / GEMM of the down projection:
//     BIT-identical (the integer sums are exact and the tier files are -ffp-contract=off, so the float steps are the base tier's);
//   - the activation quantizer and the Hadamard transform: bit-identical;
//   - SF4 / MX8: the dequantizers bit-identical, the dots within 1e-5 of the sum of the absolute terms (lane order differs).
// And the public dispatcher (strata::quant::sk::*) must give the result of the tier it reports in active_sk_isa().
// Tiers the CPU cannot run are skipped; run under `qemu-x86_64 -cpu EPYC-Genoa` (AVX-512 + VNNI) to execute the avx512 tier anywhere.
#include "sk_tiers.hpp"
#include "strata/quant/isa.hpp"

#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <random>
#include <vector>

using namespace strata::quant::sk;

namespace {

int g_fail = 0;
#define CHECK(c, ...) do { if (!(c)) { ++g_fail; std::printf("FAIL %s:%d ", __FILE__, __LINE__); std::printf(__VA_ARGS__); std::printf("\n"); } } while (0)

std::mt19937 rng(20261123);

std::vector<uint8_t> make_rows(SkCode code, size_t nrows_equiv_sb, bool trit_ok = true) {
    const size_t sb = superblock_bytes(code);
    std::vector<uint8_t> w(nrows_equiv_sb * sb);
    for (size_t s = 0; s < nrows_equiv_sb; ++s) {
        uint8_t* b = w.data() + s * sb;
        for (size_t i = 0; i < sb; ++i) b[i] = (uint8_t)(rng() & 0xFF);
        const uint16_t d = (uint16_t)(0x2C00 | (rng() & 0x03FF));   // a positive normal fp16
        std::memcpy(b, &d, 2);
        for (int g = 0; g < 4; ++g) b[2 + g] = (uint8_t)(1 + rng() % 255);
        if (code == SkCode::S1_5 && trit_ok) for (size_t i = 6; i < sb; ++i) b[i] = (uint8_t)(rng() % 243);
    }
    return w;
}


bool same(const void* a, const void* b, size_t n) { return std::memcmp(a, b, n) == 0; }


}  // namespace

// every function of a tier, so one template body compares them all
#define TIER_API(NS)                                                                                          \
    struct NS##_t {                                                                                           \
        static void quantize(const float* x, BlockQ8_1_G64* o, size_t n) { NS::quantize_q8_1_g64(x, o, n); }  \
        static void fwht(const float* i, float* o, size_t n) { NS::fwht64_f32(i, o, n); }                     \
        static void deq(const uint8_t* r, SkCode c, float* o, size_t n) { NS::dequant_sk_row(r, c, o, n); }    \
        static float dot(const uint8_t* r, SkCode c, const BlockQ8_1_G64* x, size_t n) { return NS::dot_sk_q8_1_row(r, c, x, n); } \
        static void gemm(const uint8_t* w, SkCode c, const BlockQ8_1_G64* x, float* o, size_t nr, size_t nc, size_t m) { NS::gemm_sk_q8_1(w, c, x, o, nr, nc, m); } \
        static void gemm2(const uint8_t* w, SkCode c, const BlockQ8_1_G64* x, float* o, size_t nr, size_t nc, size_t m) { NS::gemm_paired2_sk_q8_1(w, c, x, o, nr, nc, m); } \
        static void deq2(const uint8_t* r, SkCode c, float* a, float* b, size_t n) { NS::dequant_paired2_sk_rows(r, c, a, b, n); } \
        static void deqsf4(const uint8_t* p, const uint8_t* s, float os, float* o, size_t n) { NS::dequant_sf4_row(p, s, os, o, n); } \
        static float dotsf4(const uint8_t* p, const uint8_t* s, float os, const float* x, size_t n) { return NS::dot_sf4_f32_row(p, s, os, x, n); } \
        static void deqmx8(const uint8_t* w, const uint8_t* s, float* o, size_t n) { NS::dequant_mx8_row(w, s, o, n); } \
        static float dotmx8(const uint8_t* w, const uint8_t* s, const float* x, size_t n) { return NS::dot_mx8_f32_row(w, s, x, n); } \
        static float dotf32(const float* a, const float* b, size_t n) { return NS::dot_f32(a, b, n); } \
    }
TIER_API(base);
TIER_API(avx2);
TIER_API(avx512);

template <class T>
void compare_tier(const char* name) {
    const SkCode codes[] = {SkCode::S1_25, SkCode::S1_5, SkCode::S2_K, SkCode::S3_K, SkCode::S4_K, SkCode::S5_K, SkCode::S6_K, SkCode::S8_K};
    for (SkCode code : codes) {
        const int ci = (int)code;
        // gate/up shaped: ncols 2304 (9 super-blocks), 8 rows
        {
            const size_t nc = 2304, nr = 8, nsb = nc / QK_S;
            auto w = make_rows(code, nr * nsb);
            for (size_t m : {1, 5, 18}) {
                std::vector<float> xs(nc * m);
                for (auto& v : xs) v = (float)((int)(rng() % 2001) - 1000) * 0.003f;
                std::vector<BlockQ8_1_G64> q(nc / SUB_G * m), q2(nc / SUB_G * m);
                base_t::quantize(xs.data(), q.data(), nc * m);
                T::quantize(xs.data(), q2.data(), nc * m);
                CHECK(same(q.data(), q2.data(), q.size() * sizeof(BlockQ8_1_G64)), "%s quantize_q8_1_g64 code %d m %zu", name, ci, m);
                std::vector<float> o0(nr * m), o1(nr * m);
                base_t::gemm(w.data(), code, q.data(), o0.data(), nr, nc, m);
                T::gemm(w.data(), code, q.data(), o1.data(), nr, nc, m);
                CHECK(same(o0.data(), o1.data(), o0.size() * 4), "%s gemm_sk_q8_1 code %d m %zu", name, ci, m);
                if (m == 1) {
                    for (size_t r = 0; r < nr; ++r) {
                        const float a = base_t::dot(w.data() + r * nsb * superblock_bytes(code), code, q.data(), nc);
                        const float b = T::dot(w.data() + r * nsb * superblock_bytes(code), code, q.data(), nc);
                        CHECK(same(&a, &b, 4), "%s dot_sk_q8_1_row code %d row %zu: %.9g vs %.9g", name, ci, r, b, a);
                    }
                }
            }
            std::vector<float> d0(nc), d1(nc);
            base_t::deq(w.data(), code, d0.data(), nc);
            T::deq(w.data(), code, d1.data(), nc);
            CHECK(same(d0.data(), d1.data(), nc * 4), "%s dequant_sk_row code %d", name, ci);
        }
        // down shaped: ncols 1152 (2 * 1152 / 256 = 9 super-blocks per row pair), 8 rows = 4 pairs
        {
            const size_t nc = 1152, nr = 8, pair_sb = 2 * nc / QK_S;
            auto w = make_rows(code, (nr / 2) * pair_sb);
            for (size_t m : {1, 3, 17}) {
                std::vector<float> xs(nc * m);
                for (auto& v : xs) v = (float)((int)(rng() % 2001) - 1000) * 0.003f;
                std::vector<BlockQ8_1_G64> q(nc / SUB_G * m);
                base_t::quantize(xs.data(), q.data(), nc * m);
                std::vector<float> o0(nr * m), o1(nr * m);
                base_t::gemm2(w.data(), code, q.data(), o0.data(), nr, nc, m);
                T::gemm2(w.data(), code, q.data(), o1.data(), nr, nc, m);
                CHECK(same(o0.data(), o1.data(), o0.size() * 4), "%s gemm_paired2_sk_q8_1 code %d m %zu", name, ci, m);
            }
            std::vector<float> a0(nc), b0(nc), a1(nc), b1(nc);
            base_t::deq2(w.data(), code, a0.data(), b0.data(), nc);
            T::deq2(w.data(), code, a1.data(), b1.data(), nc);
            CHECK(same(a0.data(), a1.data(), nc * 4) && same(b0.data(), b1.data(), nc * 4), "%s dequant_paired2_sk_rows code %d", name, ci);
        }
    }
    // Hadamard
    {
        std::vector<float> x(256), o0(256), o1(256);
        for (auto& v : x) v = (float)((int)(rng() % 2001) - 1000) * 0.01f;
        base_t::fwht(x.data(), o0.data(), 256);
        T::fwht(x.data(), o1.data(), 256);
        CHECK(same(o0.data(), o1.data(), 1024), "%s fwht64_f32", name);
    }
    // dense float dot: the base tier is the old sequential 4-accumulator sum bit for bit; the vector tiers sum in lane order (rounding only)
    for (size_t n : {1u, 3u, 4u, 7u, 16u, 33u, 64u, 100u, 128u, 2304u, 4097u}) {
        std::vector<float> a(n), b(n);
        for (auto& v : a) v = (float)((int)(rng() % 2001) - 1000) * 0.002f;
        for (auto& v : b) v = (float)((int)(rng() % 2001) - 1000) * 0.002f;
        float s0 = 0.0f, s1 = 0.0f, s2 = 0.0f, s3 = 0.0f;
        size_t i = 0;
        for (; i + 4 <= n; i += 4) { s0 += a[i] * b[i]; s1 += a[i + 1] * b[i + 1]; s2 += a[i + 2] * b[i + 2]; s3 += a[i + 3] * b[i + 3]; }
        float old_sum = (s0 + s1) + (s2 + s3);
        for (; i < n; ++i) old_sum += a[i] * b[i];
        const float base_sum = base_t::dotf32(a.data(), b.data(), n);
        CHECK(same(&old_sum, &base_sum, 4), "base dot_f32 n %zu is not the old sequential sum: %.9g vs %.9g", n, base_sum, old_sum);
        double mag = 0;
        for (size_t k = 0; k < n; ++k) mag += std::fabs((double)a[k] * b[k]);
        const float t = T::dotf32(a.data(), b.data(), n);
        CHECK(std::fabs((double)t - base_sum) <= 1e-5 * mag + 1e-30, "%s dot_f32 n %zu: %.9g vs %.9g", name, n, t, base_sum);
    }
    // SF4 and MX8
    for (size_t nc : {16u, 48u, 2304u, 1152u}) {
        std::vector<uint8_t> p(nc / 2), sc(nc / 16);
        for (auto& b : p) b = (uint8_t)(rng() & 0xFF);
        for (auto& b : sc) b = (uint8_t)(100 + rng() % 40);
        std::vector<float> x(nc), d0(nc), d1(nc);
        for (auto& v : x) v = (float)((int)(rng() % 2001) - 1000) * 0.002f;
        const float os = 0.37f;
        base_t::deqsf4(p.data(), sc.data(), os, d0.data(), nc);
        T::deqsf4(p.data(), sc.data(), os, d1.data(), nc);
        {   // values equal; the code for -0.0 decodes to +0.0 in the vector tiers (an integer 0 times the scale), so compare as floats, not bytes
            bool eq = true;
            for (size_t i = 0; i < nc; ++i) eq = eq && d0[i] == d1[i];
            CHECK(eq, "%s dequant_sf4_row nc %zu", name, nc);
        }
        double mag = 0;
        for (size_t i = 0; i < nc; ++i) mag += std::fabs((double)d0[i] * x[i]);
        const float a = base_t::dotsf4(p.data(), sc.data(), os, x.data(), nc), b = T::dotsf4(p.data(), sc.data(), os, x.data(), nc);
        CHECK(std::fabs((double)a - b) <= 1e-5 * mag + 1e-30, "%s dot_sf4_f32_row nc %zu: %.9g vs %.9g", name, nc, b, a);
    }
    for (size_t nc : {32u, 64u, 2304u, 1152u}) {
        std::vector<uint8_t> w(nc), sc(nc / 32);
        for (auto& b : w) b = (uint8_t)(rng() & 0xFF);
        for (auto& b : sc) b = (uint8_t)(110 + rng() % 30);
        std::vector<float> x(nc), d0(nc), d1(nc);
        for (auto& v : x) v = (float)((int)(rng() % 2001) - 1000) * 0.002f;
        base_t::deqmx8(w.data(), sc.data(), d0.data(), nc);
        T::deqmx8(w.data(), sc.data(), d1.data(), nc);
        CHECK(same(d0.data(), d1.data(), nc * 4), "%s dequant_mx8_row nc %zu", name, nc);
        double mag = 0;
        for (size_t i = 0; i < nc; ++i) mag += std::fabs((double)d0[i] * x[i]);
        const float a = base_t::dotmx8(w.data(), sc.data(), x.data(), nc), b = T::dotmx8(w.data(), sc.data(), x.data(), nc);
        CHECK(std::fabs((double)a - b) <= 1e-5 * mag + 1e-30, "%s dot_mx8_f32_row nc %zu: %.9g vs %.9g", name, nc, b, a);
    }
}

int main() {
    const auto& c = strata::quant::caps();
    const bool has2 = c.avx2, has5 = c.avx512 && c.avx512_vnni;
    std::printf("sk_tier_test: avx2 %s, avx512+vnni %s; dispatcher says '%s'\n", has2 ? "yes" : "no", has5 ? "yes" : "no", active_sk_isa());
    if (has2) compare_tier<avx2_t>("avx2");
    if (has5) compare_tier<avx512_t>("avx512");

    // the public dispatcher is the tier it reports
    {
        const char* isa = active_sk_isa();
        const size_t nc = 2304, nr = 4, nsb = nc / QK_S;
        auto w = make_rows(SkCode::S4_K, nr * nsb);
        std::vector<float> xs(nc);
        for (auto& v : xs) v = (float)((int)(rng() % 2001) - 1000) * 0.003f;
        std::vector<BlockQ8_1_G64> q(nc / SUB_G);
        quantize_q8_1_g64(xs.data(), q.data(), nc);
        std::vector<float> o(nr), r(nr);
        gemm_sk_q8_1(w.data(), SkCode::S4_K, q.data(), o.data(), nr, nc, 1);
        if (!std::strcmp(isa, "avx512-vnni")) avx512::gemm_sk_q8_1(w.data(), SkCode::S4_K, q.data(), r.data(), nr, nc, 1);
        else if (!std::strcmp(isa, "avx2")) avx2::gemm_sk_q8_1(w.data(), SkCode::S4_K, q.data(), r.data(), nr, nc, 1);
        else base::gemm_sk_q8_1(w.data(), SkCode::S4_K, q.data(), r.data(), nr, nc, 1);
        CHECK(same(o.data(), r.data(), nr * 4), "the dispatcher does not give the '%s' tier's result", isa);
        CHECK(!std::strcmp(isa, has5 ? "avx512-vnni" : (has2 ? "avx2" : "scalar")), "active_sk_isa '%s' does not match the CPU", isa);
    }
    if (g_fail) { std::printf("sk_tier_test: %d FAILED\n", g_fail); return 1; }
    std::printf("sk_tier_test: PASS\n");
    return 0;
}
