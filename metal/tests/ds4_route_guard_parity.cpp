// metal/tests/ds4_route_guard_parity.cpp - the DeepSeek router's MSL kernels (ds4_init.metal: ds4_route, ds4_route_publish) on a
// non-finite gate input.  The Metal twin of tests/dsv41/ds4_route_guard_test.cu: whatever the logits hold, the picks must be valid
// expert indices (no read past the expert tables - before the guard an all-NaN row picked 0x7fffffff and read sc[0x7fffffff] /
// vram[0x7fffffff]), the same as the host's top-k when the scores are finite, and the flag the host polls (ds4_finish.cpp's
// router_bad_take) must be raised exactly when a pick had to be made up.
//   cmake --build <build-metal> --target ds4_route_guard_parity && <build-metal>/ds4_route_guard_parity
#include "mtl_rt.h"

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <utility>
#include <vector>

static int fails = 0;
static void check(bool ok, const char* what) {
    std::printf("%s %s\n", ok ? "ok  " : "FAIL", what);
    if (!ok) ++fails;
}

// n routed experts: 384 (every set so far) and 192 (the ISTA-DAS Coder).  The entries past n hold a poison score (30, bias 5): a kernel that read one would pick it.
static void run(mtl::Context& cx, const int lib, const int n) {
    std::printf("-- %d experts\n", n);
    mtl::Pipeline p_route = cx.pipeline(lib, "ds4_route"), p_pub = cx.pipeline(lib, "ds4_route_publish");

    const int k = 6, T = 16, kPicks = 8;
    const float scale = 1.5f;
    mtl::Buffer b_logits = cx.alloc((size_t) T * 1024 * 4), b_bias = cx.alloc(1024 * 4), b_idx = cx.alloc((size_t) T * kPicks * 4),
                b_w = cx.alloc((size_t) T * kPicks * 4), b_vram = cx.alloc(1024 * 8), b_plan = cx.alloc((size_t) T * kPicks * 8),
                b_vlw = cx.alloc(1024), b_bad = cx.alloc(64);
    std::vector<float> bias(1024, 5.0f), logits((size_t) T * 1024, 30.0f);
    for (int i = 0; i < n; ++i) bias[(size_t) i] = 0.01f * (float) ((i * 37) % 17);
    std::vector<unsigned long long> vram(1024, 0ull);
    for (int i = 0; i < n; i += 2) vram[(size_t) i] = 0x1000ull + (unsigned long long) i;
    std::memcpy(b_bias.ptr, bias.data(), 1024 * 4);
    std::memcpy(b_vram.ptr, vram.data(), 1024 * 8);

    const float nan = std::nanf(""), inf = INFINITY;
    const char* names[] = {"finite", "all NaN", "+inf everywhere", "-inf everywhere", "half NaN", "fewer than k finite"};
    for (int mode = 0; mode < 6; ++mode) {
        unsigned s = 12345u;
        for (int t = 0; t < T; ++t)
            for (int i = 0; i < 1024; ++i) {
                s = s * 1664525u + 1013904223u;
                float v = ((float) (s >> 8) / (float) (1 << 24) - 0.5f) * 8.0f;
                if (i >= n) v = 30.0f;   // (the poison; the NaN / inf modes overwrite every entry)
                if (mode == 1) v = nan;
                if (mode == 2) v = inf;
                if (mode == 3) v = -inf;
                if (mode == 4 && (i & 1)) v = nan;
                if (mode == 5 && i >= 3) v = nan;
                logits[(size_t) t * 1024 + (size_t) i] = v;
            }
        std::memcpy(b_logits.ptr, logits.data(), logits.size() * 4);
        for (int which = 0; which < 2; ++which) {
            std::memset(b_idx.ptr, 0xff, (size_t) T * kPicks * 4);   // (-1: a pick the kernel never wrote is not a valid index)
            std::memset(b_plan.ptr, 0, (size_t) T * kPicks * 8);
            b_bad.as<int>()[0] = 0;
            mtl::Dispatch d;
            if (which == 0) {
                d.p = &p_route;
                d.args = {mtl::Arg::B(b_logits), mtl::Arg::B(b_bias), mtl::Arg::V((uint32_t) n), mtl::Arg::V((uint32_t) k),
                          mtl::Arg::V(scale), mtl::Arg::B(b_idx), mtl::Arg::B(b_w), mtl::Arg::B(b_bias), mtl::Arg::V(0u),
                          mtl::Arg::B(b_vlw), mtl::Arg::B(b_bad)};
            } else {
                d.p = &p_pub;
                d.args = {mtl::Arg::B(b_logits), mtl::Arg::B(b_bias), mtl::Arg::V((uint32_t) n), mtl::Arg::V((uint32_t) k),
                          mtl::Arg::V(scale), mtl::Arg::B(b_vram), mtl::Arg::B(b_idx), mtl::Arg::B(b_w), mtl::Arg::B(b_plan),
                          mtl::Arg::B(b_bias), mtl::Arg::V(0u), mtl::Arg::V(0u), mtl::Arg::B(b_bad)};
            }
            d.grid[0] = (uint32_t) T;
            d.tg[0] = 512;
            cx.run({d});
            const int* idx = b_idx.as<int>();
            const float* w = b_w.as<float>();
            const char* kn = which ? "route_publish" : "route";
            char what[200];
            bool valid = true, distinct = true, same = true;
            for (int t = 0; t < T; ++t) {
                for (int r = 0; r < k; ++r) {
                    const int e = idx[(size_t) t * kPicks + r];
                    valid = valid && e >= 0 && e < n;
                    for (int q = 0; q < r; ++q) distinct = distinct && idx[(size_t) t * kPicks + q] != e;
                }
                if (mode == 0) {   // the host's top-k of the same scores
                    std::vector<std::pair<float, int>> sc;
                    for (int i = 0; i < n; ++i) {
                        const float l = logits[(size_t) t * 1024 + (size_t) i];
                        const float sp = l > 20.0f ? l : std::log1p(std::exp(l));
                        sc.push_back({std::sqrt(sp) + bias[(size_t) i], i});
                    }
                    std::stable_sort(sc.begin(), sc.end(), [](const auto& a, const auto& b) { return a.first > b.first; });
                    for (int r = 0; r < k; ++r) same = same && sc[(size_t) r].second == idx[(size_t) t * kPicks + r];
                }
            }
            if (!valid || !distinct) {
                std::printf("  token 0 picks:");
                for (int r = 0; r < kPicks; ++r) std::printf(" %d", idx[r]);
                std::printf("\n");
            }
            std::snprintf(what, sizeof what, "%s: %s picks are all in [0, %d)", names[mode], kn, n);
            check(valid, what);
            // (with fewer than k finite scores the picked entries' start value ties with the empty pick, so the router re-takes the lowest of
            //  them: a repeat, harmless to the tables - and the weights are NaN there anyway - so only checked when k scores are usable)
            if (mode != 1 && mode != 5) {
                std::snprintf(what, sizeof what, "%s: %s picks are distinct", names[mode], kn);
                check(distinct, what);
            }
            if (mode == 0) {
                std::snprintf(what, sizeof what, "finite: %s picks equal the host's top-%d", kn, k);
                check(same, what);
                bool wsum = true;
                for (int t = 0; t < T; ++t) {
                    double sw = 0;
                    for (int r = 0; r < k; ++r) sw += w[(size_t) t * kPicks + r];
                    wsum = wsum && std::fabs(sw - scale) < 1e-4;
                }
                std::snprintf(what, sizeof what, "finite: %s weights sum to route_scale", kn);
                check(wsum, what);
            }
            const bool flagged = b_bad.as<int>()[0] != 0;
            std::snprintf(what, sizeof what, "%s: %s %s the non-finite flag", names[mode], kn, mode == 1 ? "raises" : "leaves clear");
            check(flagged == (mode == 1), what);   // (fewer than k finite: the picked entries' start value re-wins, no stand-in is needed)
            if (which == 1) {
                const unsigned long long* pl = b_plan.as<unsigned long long>();
                bool plan_ok = true;
                for (int t = 0; t < T; ++t)
                    for (int r = 0; r < k; ++r)
                        plan_ok = plan_ok && pl[(size_t) t * kPicks + r] == vram[(size_t) idx[(size_t) t * kPicks + r]];
                std::snprintf(what, sizeof what, "%s: route_publish's plan pointers are the table's entries of the picks", names[mode]);
                check(plan_ok, what);
            }
        }
    }
}

int main() {
    mtl::Context cx;
    const int lib = cx.compile({"../src/engine/msl/ds4_init.metal"}, "", false);
    for (int n : {384, 192}) run(cx, lib, n);
    std::printf("%s\n", fails ? "ds4_route_guard_parity: FAILED" : "ds4_route_guard_parity: all ok");
    return fails ? 1 : 0;
}
