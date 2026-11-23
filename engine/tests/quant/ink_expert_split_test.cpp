// tests/quant/ink_expert_split_test.cpp - the row-split expert evaluation used by the hybrid CPU pool at decode
// (expert_m1_gu_rows / expert_m1_act / expert_m1_dn_rows) must give the same BITS as eval_inkling2_expert for every
// split of the rows, for SF4, MX8 and S2_K..S8_K experts, in any order the slices run.
// MIT License - Copyright (c) 2026 Strata contributors
#include "strata/inkling2/inkling2_loop.hpp"
#include "strata/quant/sk_codecs.hpp"

#include <algorithm>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <random>
#include <vector>

using namespace strata::inkling2;
using namespace strata::quant::sk;

namespace {

constexpr size_t kHidden = 2304, kInter = 1152;

struct Expert {
    std::vector<uint8_t> body;
    SkCode gu, dn;
    uint32_t dn_offset = 0;
};

void fill_random(std::vector<uint8_t>& v, size_t off, size_t n, std::mt19937& rng) {
    for (size_t i = 0; i < n; ++i) v[off + i] = static_cast<uint8_t>(rng() & 0xFF);
}

// Plausible FP16 scale 0.03125 (0x2800) at the start of every S_K super-block.
void set_sb_scales(std::vector<uint8_t>& v, size_t off, size_t n_sb, size_t sb_b) {
    for (size_t s = 0; s < n_sb; ++s) { v[off + s * sb_b] = 0x00; v[off + s * sb_b + 1] = 0x28; }
}

size_t gu_bytes(SkCode c) {
    if (c == SkCode::SF4) return 16 + (2 * kInter) * (kHidden / 2) + (2 * kInter) * (kHidden / 16);
    if (c == SkCode::MX8) return (2 * kInter) * kHidden + (2 * kInter) * (kHidden / 32);
    return 2 * kInter + (2 * kInter) * (kHidden / QK_S) * superblock_bytes(c);
}
size_t dn_bytes(SkCode c) {
    if (c == SkCode::SF4) return (kHidden) * (kInter / 2) + kHidden * (kInter / 16);
    if (c == SkCode::MX8) return kHidden * kInter + kHidden * (kInter / 32);
    return (kHidden / 2) * (((2 * kInter) / QK_S) * superblock_bytes(c));
}

Expert make_expert(SkCode gu, SkCode dn, std::mt19937& rng) {
    Expert e;
    e.gu = gu; e.dn = dn;
    const size_t g = gu_bytes(gu), d = dn_bytes(dn);
    e.body.assign(g + d + 64, 0);
    e.dn_offset = static_cast<uint32_t>(g);
    fill_random(e.body, 0, g + d, rng);
    // SF4 header: floats {w13 amax, w13 denom, w2 amax, w2 denom}
    if (gu == SkCode::SF4) { const float h[4] = {1.5f, 2.0f, 1.25f, 2.5f}; std::memcpy(e.body.data(), h, 16); }
    if (dn == SkCode::SF4) {
        float h[4] = {1.5f, 2.0f, 1.25f, 2.5f};
        if (gu == SkCode::SF4) std::memcpy(e.body.data(), h, 16);
        else std::memcpy(e.body.data(), h, 16);   // the dn code reads the header at body[0..16) as well
    }
    if (gu != SkCode::SF4 && gu != SkCode::MX8) {
        for (size_t j = 0; j < kInter; ++j) { e.body[2 * j] = 0x00; e.body[2 * j + 1] = 0x3C; }  // sq_xd_dn = 1.0 (FP16)
        set_sb_scales(e.body, 2 * kInter, (2 * kInter) * (kHidden / QK_S), superblock_bytes(gu));
    }
    if (dn != SkCode::SF4 && dn != SkCode::MX8) {
        if (gu == SkCode::SF4 || gu == SkCode::MX8) {   // the dn path reads sq_xd_dn from body[0..2*inter)
            for (size_t j = 0; j < kInter; ++j) { e.body[2 * j] = 0x00; e.body[2 * j + 1] = 0x3C; }
        }
        set_sb_scales(e.body, g, (kHidden / 2) * ((2 * kInter) / QK_S), superblock_bytes(dn));
    }
    return e;
}

const char* name(SkCode c) {
    switch (c) {
        case SkCode::S2_K: return "S2_K"; case SkCode::S3_K: return "S3_K"; case SkCode::S4_K: return "S4_K";
        case SkCode::S5_K: return "S5_K"; case SkCode::S6_K: return "S6_K"; case SkCode::S8_K: return "S8_K";
        case SkCode::SF4: return "SF4"; case SkCode::MX8: return "MX8"; default: return "?";
    }
}

// the same slicing as the pool: multiples of 8 rows
void slice(size_t total, size_t S, size_t sl, size_t& a, size_t& b) {
    size_t chunk = (total + S - 1) / S;
    chunk = (chunk + 7) & ~size_t(7);
    a = std::min(total, sl * chunk);
    b = std::min(total, a + chunk);
}

}  // namespace

int main() {
    std::mt19937 rng(12345);
    std::vector<float> x(kHidden);
    for (auto& v : x) v = (static_cast<int>(rng() % 2001) - 1000) / 500.0f;

    const SkCode codes[] = {SkCode::SF4, SkCode::MX8, SkCode::S2_K, SkCode::S3_K, SkCode::S4_K, SkCode::S5_K,
                            SkCode::S6_K, SkCode::S8_K};
    const size_t slices[] = {1, 2, 3, 5, 8, 16};
    int fails = 0, checks = 0;
    for (SkCode gu : codes) {
        for (SkCode dn : codes) {
            const bool lossless = (gu == SkCode::SF4 || gu == SkCode::MX8 || dn == SkCode::SF4 || dn == SkCode::MX8);
            if (lossless && gu != dn) continue;   // real files never mix these layouts
            if (!expert_m1_splittable(gu, dn)) { std::printf("FAIL splittable(%s,%s)\n", name(gu), name(dn)); ++fails; continue; }
            Expert e = make_expert(gu, dn, rng);
            std::vector<float> ref(kHidden);
            eval_inkling2_expert(e.body.data(), gu, dn, e.dn_offset, x.data(), kHidden, kInter, ref.data());
            for (size_t S : slices) {
                std::vector<float> g(2 * kInter, -7.0f), act(kInter, -7.0f), y(kHidden, -7.0f);
                for (size_t sl = S; sl-- > 0;) {   // reverse order on purpose
                    size_t a, b;
                    slice(2 * kInter, S, sl, a, b);
                    expert_m1_gu_rows(e.body.data(), gu, x.data(), kHidden, kInter, a, b, g.data());
                }
                for (size_t sl = 0; sl < S; ++sl) {
                    size_t a, b;
                    slice(kInter, S, sl, a, b);
                    expert_m1_act(g.data(), a, b, act.data());
                }
                for (size_t sl = S; sl-- > 0;) {
                    size_t a, b;
                    slice(kHidden, S, sl, a, b);
                    expert_m1_dn_rows(e.body.data(), dn, e.dn_offset, act.data(), kHidden, kInter, a, b, y.data());
                }
                ++checks;
                if (std::memcmp(y.data(), ref.data(), kHidden * sizeof(float)) != 0) {
                    size_t bad = 0;
                    while (bad < kHidden && std::memcmp(&y[bad], &ref[bad], 4) == 0) ++bad;
                    std::printf("FAIL gu=%s dn=%s slices=%zu first differing row %zu: split %.9g vs whole %.9g\n",
                                name(gu), name(dn), S, bad, y[bad], ref[bad]);
                    ++fails;
                }
            }
        }
    }
    std::printf("ink_expert_split_test: %d checks, %d failures\n", checks, fails);
    if (fails == 0) std::printf("ink_expert_split_test: OK\n");
    return fails == 0 ? 0 : 1;
}
