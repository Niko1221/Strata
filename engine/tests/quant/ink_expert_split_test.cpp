// tests/quant/ink_expert_split_test.cpp - the row-split expert evaluation used by the hybrid CPU pool at decode
// (expert_m1_gu_rows / expert_m1_act / expert_m1_dn_rows) must give the same BITS as eval_inkling2_expert for every
// split of the rows, for SF4, MX8, S2_K..S8_K and GGUF_RAW experts, in any order the slices run.
// MIT License - Copyright (c) 2026 Strata contributors
#include "strata/artifact/gguf_reader.hpp"
#include "strata/inkling2/inkling2_loop.hpp"
#include "strata/quant/sk_codecs.hpp"

#include <algorithm>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <random>
#include <string>
#include <vector>

using namespace strata::inkling2;
using namespace strata::quant::sk;

namespace {

constexpr size_t kHidden = 2304, kInter = 1152;

struct Expert {
    std::vector<uint8_t> body;
    SkCode gu, dn;
    uint32_t dn_offset = 0;
    std::string label;
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

const char* name(SkCode c) {
    switch (c) {
        case SkCode::S2_K: return "S2_K"; case SkCode::S3_K: return "S3_K"; case SkCode::S4_K: return "S4_K";
        case SkCode::S5_K: return "S5_K"; case SkCode::S6_K: return "S6_K"; case SkCode::S8_K: return "S8_K";
        case SkCode::SF4: return "SF4"; case SkCode::MX8: return "MX8"; default: return "?";
    }
}

Expert make_expert(SkCode gu, SkCode dn, std::mt19937& rng) {
    Expert e;
    e.gu = gu; e.dn = dn;
    e.label = std::string("gu=") + name(gu) + " dn=" + name(dn);
    const size_t g = gu_bytes(gu), d = dn_bytes(dn);
    e.body.assign(g + d + 64, 0);
    e.dn_offset = static_cast<uint32_t>(g);
    fill_random(e.body, 0, g + d, rng);
    if (gu == SkCode::SF4 || dn == SkCode::SF4) {   // header floats {w13 amax, w13 denom, w2 amax, w2 denom}
        const float h[4] = {1.5f, 2.0f, 1.25f, 2.5f};
        std::memcpy(e.body.data(), h, 16);
    }
    if (gu != SkCode::SF4 && gu != SkCode::MX8) {
        for (size_t j = 0; j < kInter; ++j) { e.body[2 * j] = 0x00; e.body[2 * j + 1] = 0x3C; }  // sq_xd_dn = 1.0 (FP16)
        set_sb_scales(e.body, 2 * kInter, (2 * kInter) * (kHidden / QK_S), superblock_bytes(gu));
    }
    if (dn != SkCode::SF4 && dn != SkCode::MX8) {
        set_sb_scales(e.body, g, (kHidden / 2) * ((2 * kInter) / QK_S), superblock_bytes(dn));
    }
    return e;
}

// GGUF_RAW expert: 32-byte header, gate/up rows (hidden columns), down rows (inter columns) of the given ggml types.
// fp16_at: byte offsets inside a block that hold fp16 scales (set to a plausible value so the result is finite).
struct GgufType { uint32_t t; const char* nm; std::vector<int> fp16_at; };

Expert make_gguf(const GgufType& gt, const GgufType& dt, bool interleaved, std::mt19937& rng) {
    Expert e;
    e.gu = e.dn = SkCode::GGUF_RAW;
    e.label = std::string("GGUF_RAW gu=") + gt.nm + " dn=" + dt.nm + (interleaved ? " interleaved" : " split");
    int gbe = 0, gbb = 0, dbe = 0, dbb = 0;
    strata::block_geometry(gt.t, gbe, gbb);
    strata::block_geometry(dt.t, dbe, dbb);
    const size_t grow = (kHidden / gbe) * gbb, drow = (kInter / dbe) * dbb;
    const size_t gu_total = 2 * kInter * grow, dn_total = kHidden * drow;
    const size_t dn_off = (32 + gu_total + 31) & ~size_t(31);
    e.dn_offset = static_cast<uint32_t>(dn_off);
    e.body.assign(dn_off + dn_total + 64, 0);
    fill_random(e.body, 32, gu_total, rng);
    fill_random(e.body, dn_off, dn_total, rng);
    auto fix = [&](size_t off, size_t rows, size_t row_b, size_t bb, const std::vector<int>& at) {
        for (size_t r = 0; r < rows; ++r)
            for (size_t b = 0; b < row_b / bb; ++b)
                for (int a : at) { e.body[off + r * row_b + b * bb + a] = 0x00; e.body[off + r * row_b + b * bb + a + 1] = 0x28; }
    };
    fix(32, 2 * kInter, grow, gbb, gt.fp16_at);
    fix(dn_off, kHidden, drow, dbb, dt.fp16_at);
    strata::quant::sk::GgufExpertHeader hdr{};
    hdr.magic = 0x47475546u; hdr.gu_type = gt.t; hdr.dn_type = dt.t; hdr.flags = interleaved ? 1u : 0u;
    hdr.gate_scale = 1.0f; hdr.up_scale = 0.75f; hdr.dn_scale = 0.5f;
    std::memcpy(e.body.data(), &hdr, sizeof(hdr));
    return e;
}

// the same slicing as the pool: multiples of 8 rows
void slice(size_t total, size_t S, size_t sl, size_t& a, size_t& b) {
    size_t chunk = (total + S - 1) / S;
    chunk = (chunk + 7) & ~size_t(7);
    a = std::min(total, sl * chunk);
    b = std::min(total, a + chunk);
}

int g_checks = 0, g_fails = 0, g_skipped = 0;

void check(const Expert& e, const std::vector<float>& x) {
    if (!expert_m1_splittable(e.gu, e.dn, e.body.data(), kHidden, kInter)) {
        std::printf("FAIL not splittable: %s\n", e.label.c_str());
        ++g_fails;
        return;
    }
    std::vector<float> ref(kHidden);
    eval_inkling2_expert(e.body.data(), e.gu, e.dn, e.dn_offset, x.data(), kHidden, kInter, ref.data());
    size_t nonfinite = 0;
    for (float v : ref) nonfinite += !(v == v) || v > 3e38f || v < -3e38f;
    const size_t slices[] = {1, 2, 3, 5, 8, 16};
    for (size_t S : slices) {
        std::vector<float> g(2 * kInter, -7.0f), act(kInter, -7.0f), y(kHidden, -7.0f);
        for (size_t sl = S; sl-- > 0;) {   // reverse order on purpose
            size_t a, b;
            slice(2 * kInter, S, sl, a, b);
            expert_m1_gu_rows(e.body.data(), e.gu, x.data(), kHidden, kInter, a, b, g.data());
        }
        for (size_t sl = 0; sl < S; ++sl) {
            size_t a, b;
            slice(kInter, S, sl, a, b);
            expert_m1_act(g.data(), a, b, act.data());
        }
        for (size_t sl = S; sl-- > 0;) {
            size_t a, b;
            slice(kHidden, S, sl, a, b);
            expert_m1_dn_rows(e.body.data(), e.dn, e.dn_offset, act.data(), kHidden, kInter, a, b, y.data());
        }
        ++g_checks;
        if (std::memcmp(y.data(), ref.data(), kHidden * sizeof(float)) != 0) {
            size_t bad = 0;
            while (bad < kHidden && std::memcmp(&y[bad], &ref[bad], 4) == 0) ++bad;
            std::printf("FAIL %s slices=%zu first differing row %zu: split %.9g vs whole %.9g\n",
                        e.label.c_str(), S, bad, y[bad], ref[bad]);
            ++g_fails;
        }
    }
    if (nonfinite > kHidden / 2) std::printf("note: %s: %zu of %zu outputs not finite (test is weak for it)\n", e.label.c_str(), nonfinite, kHidden);
}

}  // namespace

int main() {
    std::mt19937 rng(12345);
    std::vector<float> x(kHidden);
    for (auto& v : x) v = (static_cast<int>(rng() % 2001) - 1000) / 500.0f;

    const SkCode codes[] = {SkCode::SF4, SkCode::MX8, SkCode::S2_K, SkCode::S3_K, SkCode::S4_K, SkCode::S5_K,
                            SkCode::S6_K, SkCode::S8_K};
    for (SkCode gu : codes) {
        for (SkCode dn : codes) {
            const bool lossless = (gu == SkCode::SF4 || gu == SkCode::MX8 || dn == SkCode::SF4 || dn == SkCode::MX8);
            if (lossless && gu != dn) continue;   // real files never mix these layouts
            check(make_expert(gu, dn, rng), x);
        }
    }
    // GGUF_RAW: gate/up rows have 2304 columns (K-quants fit), down rows 1152 columns (32-element blocks only)
    const GgufType gus[] = {{12, "Q4_K", {0, 2}}, {13, "Q5_K", {0, 2}}, {14, "Q6_K", {208}}, {8, "Q8_0", {0}},
                            {11, "Q3_K", {108}}, {10, "Q2_K", {80, 82}}, {23, "IQ4_XS", {0}}, {2, "Q4_0", {0}}, {20, "IQ4_NL", {0}}};
    const GgufType dns[] = {{8, "Q8_0", {0}}, {2, "Q4_0", {0}}, {20, "IQ4_NL", {0}}};
    for (const auto& g : gus)
        for (const auto& d : dns)
            for (bool il : {true, false}) check(make_gguf(g, d, il, rng), x);
    std::printf("ink_expert_split_test: %d checks, %d failures\n", g_checks, g_fails);
    if (g_fails == 0) std::printf("ink_expert_split_test: OK\n");
    return g_fails == 0 ? 0 : 1;
}
