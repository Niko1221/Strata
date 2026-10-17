#include "strata/kernels/fused_gr_policy.hpp"

#include <cstdio>
#include <string_view>

struct Case {
    std::string_view selector;
    bool gfx12;
    int expected;
    const char* label;
};

int main() {
    using namespace strata::kernels::detail;
    constexpr Case cases[] = {
        {"", true, kGrVariantP4, "unset gfx12 selects default P4"},
        {"", false, kGrVariantStaged, "unset non-gfx12 keeps staged"},
        {"0", true, kGrVariantPlain, "0 selects plain on gfx12"},
        {"0", false, kGrVariantPlain, "0 selects plain off gfx12"},
        {"1", true, kGrVariantSplit, "1 selects split on gfx12"},
        {"1", false, kGrVariantSplit, "1 selects split off gfx12"},
        {"2", true, kGrVariantStaged, "2 selects staged on gfx12"},
        {"2", false, kGrVariantStaged, "2 selects staged off gfx12"},
        {"7", true, kGrVariantP4, "7 explicitly selects P4 on gfx12"},
        {"7", false, kGrVariantStaged, "7 falls back to staged off gfx12"},
        {"70", true, kGrVariantStaged, "selector must match exactly"},
        {"unknown", true, kGrVariantStaged, "unknown selector fails safe"},
    };
    int failures = 0;
    for (const auto& test : cases) {
        const int got = fused_gr_env_variant(test.selector, test.gfx12);
        if (got != test.expected) {
            ++failures;
            std::fprintf(stderr, "FAIL: %s: expected=%d got=%d\n", test.label, test.expected, got);
        }
    }
    std::printf("fused_gr_selector_test: %zu cases, %d failures\n",
                sizeof(cases) / sizeof(cases[0]), failures);
    return failures == 0 ? 0 : 1;
}
