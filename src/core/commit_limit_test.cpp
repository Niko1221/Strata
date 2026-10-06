#include "strata/core/commit_limit.hpp"
#include <array>
#include <cstdio>

int main() {
    using strata::core::emitted_prefix_length;
    const std::array<int64_t, 2> eos{99, 100};
    const std::array<int32_t, 4> ordinary{1, 2, 3, 4};
    const std::array<int32_t, 4> first_eos{99, 2, 3, 4};
    const std::array<int32_t, 4> middle_eos{1, 100, 3, 4};
    int failures = 0;
    auto check = [&](bool condition) { if (!condition) ++failures; };
    check(emitted_prefix_length(ordinary, 100, eos) == 4);
    check(emitted_prefix_length(ordinary, 1, eos) == 1);
    check(emitted_prefix_length(ordinary, 0, eos) == 0);
    check(emitted_prefix_length(first_eos, 4, eos) == 1);
    check(emitted_prefix_length(middle_eos, 4, eos) == 2);
    check(emitted_prefix_length(middle_eos, 1, eos) == 1);
    check(emitted_prefix_length(ordinary, 2, {}) == 2);
    check(emitted_prefix_length({}, 2, eos) == 0);
    // Ending a request on a draft cannot feed a later draft into recurrent state.
    for (int budget = 1; budget <= 4; ++budget)
        check(emitted_prefix_length(middle_eos, budget, eos) == std::min(budget, 2));
    std::printf("commit_limit_test: %s\n", failures ? "FAILED" : "OK");
    return failures != 0;
}
