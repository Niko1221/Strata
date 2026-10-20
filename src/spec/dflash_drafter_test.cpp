// src/spec/dflash_drafter_test.cpp - loads a --dflash GGUF, injects features, drafts; checks the bookkeeping.
// usage: dflash_drafter_test <draft.gguf>
#include "strata/spec/dflash_drafter.hpp"

#include <cassert>
#include <chrono>
#include <cstdio>
#include <vector>

int main(int argc, char** argv) {
    if (argc < 2) { std::fprintf(stderr, "usage: %s <draft.gguf>\n", argv[0]); return 2; }
    strata::spec::DflashDrafter d;
    std::string err;
    if (!d.load(argv[1], err)) { std::fprintf(stderr, "%s\n", err.c_str()); return 1; }

    assert(d.target_hidden() == 2560 && d.feat_width() == 12800 && d.block() == 8);
    std::vector<float> f((size_t) 16 * d.feat_width(), 0.01f);
    if (!d.inject(f.data(), 16, 0, err)) { std::fprintf(stderr, "%s\n", err.c_str()); return 1; }
    assert(d.pos_max() == 15);
    int32_t out[8];
    const auto t0 = std::chrono::steady_clock::now();
    const int n = d.draft(1234, 16, 7, 0.0f, out);
    const double ms = std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - t0).count();
    assert(n == 7 && d.pos_max() == 15);
    std::printf("draft %d tokens in %.2f ms:", n, ms);
    for (int i = 0; i < n; ++i) std::printf(" %d", out[i]);
    std::printf("\n");
    // the next commit lands at 16..18; a stale tail is replaced, a gap clears
    if (!d.inject(f.data(), 3, 16, err)) return 1;
    assert(d.pos_max() == 18);
    if (!d.inject(f.data(), 2, 17, err)) return 1;
    assert(d.pos_max() == 18);
    if (!d.inject(f.data(), 2, 120000, err)) return 1;
    assert(d.pos_max() == 120001);
    for (int k : {1, 2, 3, 7}) {
        for (int r = 0; r < 10; ++r) d.draft(1234, 120002, k, 0.0f, out);
        const auto t1 = std::chrono::steady_clock::now();
        for (int r = 0; r < 30; ++r) d.draft(1234, 120002, k, 0.0f, out);
        std::printf("steady draft(%d): %.2f ms\n", k,
                    std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - t1).count() / 30);
    }
    std::printf("ok\n");
    return 0;
}
