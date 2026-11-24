// engine/tests/core/ink_sy_fit_test.cpp - the SYCL runner's memory need per slot and how many slots a card can hold (no GPU, no SYCL).
// Arc A750 (8 GB), `--parallel 32 --max-context 4096`, UD-IQ1_S: the start died mid-load with "dalloc (268435456 bytes): out of device memory".
// MIT License - Copyright (c) 2026 Strata contributors
#include "ink_sy_fit.h"

#include <cstdio>
#include <string>

using namespace ink_sycl;

namespace {
int failed = 0;
#define CHECK(expr)                                                                         \
    do {                                                                                    \
        if (!(expr)) {                                                                      \
            std::fprintf(stderr, "FAIL %s:%d: %s\n", __FILE__, __LINE__, #expr);            \
            ++failed;                                                                       \
        }                                                                                   \
    } while (0)

// Inkling-2-Mini: 24 physical layers, core 6..18 looped 3 times -> 48 visits; global layers 5, 11, 17, 23 (11 and 17 are in the loop) -> 8 global visits.
InkSyFitGeom inkling(bool dspark, bool vision) {
    InkSyFitGeom g;
    const bool is_global[24] = {0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 1};
    for (int v = 0; v < 48; ++v) {
        const int ip = v < 6 ? v : v < 6 + 36 ? 6 + (v - 6) % 12 : 18 + (v - 42);
        g.visit_is_swa.push_back(is_global[ip] ? 0 : 1);
    }
    g.dspark = dspark;
    g.vision = vision;
    return g;
}
constexpr size_t MiB = 1ull << 20;
constexpr size_t GiB = 1ull << 30;
}  // namespace

int main() {
    const InkSyFitGeom g = inkling(false, false);
    int n_global = 0;
    for (uint8_t s : g.visit_is_swa) n_global += s ? 0 : 1;
    CHECK(g.visit_is_swa.size() == 48 && n_global == 8);

    // the K/V of one slot at 32768 tokens is what the engine logs ("KV=1184.0 MiB"): 8 global visits of 32768 tokens + 40 window visits of 1024
    const size_t one = ink_sy_runner_reserve(g, 1, 32768, 256);
    const size_t kv_only_1 = 2ull * 1 * (8ull * 32768 + 40ull * 1024) * 8 * 128 * 2;
    CHECK(kv_only_1 == 1184 * MiB);
    CHECK(one > kv_only_1 && one < kv_only_1 + 1 * GiB);

    // 32 slots at 4096 tokens: about 4.3 GiB of K/V alone (the 256 MiB allocation of the crash is one visit of it)
    const size_t kv32 = 2ull * 32 * (8ull * 4096 + 40ull * 1024) * 8 * 128 * 2;
    CHECK(kv32 == 32ull * 2 * (8 * 4096 + 40 * 1024) * 1024 * 2);
    CHECK(32ull * 4096 * 1024 * 2 == 256 * MiB);
    CHECK(ink_sy_runner_reserve(g, 32, 4096, 256) > kv32);

    // an 8 GB card with about 4 GiB left after the weights: 32 slots at 4096 do not fit, a smaller number does, and the message names it
    const size_t avail = 4 * GiB;
    CHECK(ink_sy_runner_reserve(g, 32, 4096, 256) > avail);
    const int most = ink_sy_max_parallel(g, 4096, 256, avail);
    CHECK(most >= 1 && most < 32);
    CHECK(ink_sy_runner_reserve(g, most, 4096, 256) <= avail);
    CHECK(ink_sy_runner_reserve(g, most + 1, 4096, 256) > avail);
    const std::string msg = ink_sy_fit_message(g, 32, 4096, 256, avail);
    CHECK(!msg.empty());
    CHECK(msg.find("--parallel 32") != std::string::npos);
    CHECK(msg.find("--parallel " + std::to_string(most)) != std::string::npos);

    // what fits gives no message; the reserve grows with slots and with the context; it never shrinks
    CHECK(ink_sy_fit_message(g, 1, 4096, 256, avail).empty());
    CHECK(ink_sy_fit_message(g, most, 4096, 256, avail).empty());
    CHECK(ink_sy_runner_reserve(g, 2, 4096, 256) > ink_sy_runner_reserve(g, 1, 4096, 256));
    CHECK(ink_sy_runner_reserve(g, 1, 8192, 256) > ink_sy_runner_reserve(g, 1, 4096, 256));

    // not even one slot: the message says so and asks for a shorter context
    const std::string none = ink_sy_fit_message(g, 1, 1000000, 256, avail);
    CHECK(!none.empty() && none.find("shorter --max-context") != std::string::npos);
    CHECK(ink_sy_max_parallel(g, 1000000, 256, avail) == 0);

    // DSpark and Vision add to the need
    CHECK(ink_sy_runner_reserve(inkling(true, false), 4, 8192, 256) > ink_sy_runner_reserve(g, 4, 8192, 256));
    CHECK(ink_sy_runner_reserve(inkling(false, true), 4, 8192, 256) == ink_sy_runner_reserve(g, 4, 8192, 256) + 700 * MiB);

    if (failed != 0) {
        std::fprintf(stderr, "ink_sy_fit_test: %d check(s) failed\n", failed);
        return 1;
    }
    std::printf("ink_sy_fit_test: OK\n");
    return 0;
}
