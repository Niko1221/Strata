// engine/tests/core/ink_hip_promote_policy_test.cpp - the HIP hybrid engine's use of a promoted expert must not depend on timing (no GPU, no HIP).
// The HIP twin of ink_sy_promote_policy_test: the elastic VRAM cache used a copy only if its event had already completed at the time of the
// check (hipEventQuery), so the same routing with other copy timing computed an expert once from VRAM and once on the CPU, which round differently.
// MIT License - Copyright (c) 2026 Strata contributors
#include "ink_hip_promote_policy.h"

#include <cstdio>
#include <random>
#include <vector>

using namespace ink_hip;

namespace {
int failed = 0;
#define CHECK(expr)                                                                         \
    do {                                                                                    \
        if (!(expr)) {                                                                      \
            std::fprintf(stderr, "FAIL %s:%d: %s\n", __FILE__, __LINE__, #expr);            \
            ++failed;                                                                       \
        }                                                                                   \
    } while (0)

std::vector<int> usable(bool async_poll, const std::vector<bool>& done) {
    std::vector<int> out;
    for (size_t i = 0; i < done.size(); ++i)
        if (ink_hip_promote_usable_now(async_poll, done[i])) out.push_back((int) i);
    return out;
}
}  // namespace

int main() {
    CHECK(!ink_hip_promote_async_poll(nullptr));
    CHECK(!ink_hip_promote_async_poll(""));
    CHECK(!ink_hip_promote_async_poll("0"));
    CHECK(ink_hip_promote_async_poll("1"));

    std::mt19937 rng(11);
    const std::vector<int> all = {0, 1, 2, 3, 4, 5};
    int poll_differs = 0;
    std::vector<int> first_poll;
    for (int t = 0; t < 200; ++t) {
        std::vector<bool> done(6);
        for (size_t i = 0; i < done.size(); ++i) done[i] = (rng() & 1u) != 0;
        CHECK(usable(false, done) == all);   // the default: the same set for every timing
        const std::vector<int> p = usable(true, done);
        if (t == 0) first_poll = p;
        else if (p != first_poll) ++poll_differs;
    }
    CHECK(poll_differs > 0);   // the poll is what depended on timing
    if (failed == 0) std::printf("ink_hip_promote_policy_test: ok\n");
    return failed == 0 ? 0 : 1;
}
