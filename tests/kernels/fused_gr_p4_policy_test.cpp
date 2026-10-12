#include "strata/kernels/fused_gr_policy.hpp"

#include <cstdio>

struct TestCase {
    bool gfx12;
    bool staged_ok;
    bool p4_ok;
    bool expected;
    const char* name;
};

int main() {
    const TestCase cases[] = {
        {true, true, true, true, "gfx12 with staged and P4 parity"},
        {true, false, true, false, "reject P4 when staged reference fails"},
        {true, true, false, false, "reject P4 when candidate parity fails"},
        {false, true, true, false, "reject P4 outside gfx12"},
        {false, false, false, false, "reject all failed gates"},
    };
    int failures = 0;
    for (const auto& test : cases) {
        const bool got = strata::kernels::detail::fused_gr_p4_parity_gate(
            test.gfx12, test.staged_ok, test.p4_ok);
        if (got != test.expected) {
            ++failures;
            std::fprintf(stderr, "FAIL: %s expected=%d got=%d\n", test.name, test.expected, got);
        }
    }
    struct MarkerCase {
        bool selftest_ok;
        bool qfuse_hc;
        bool staged_ok;
        bool p4_ok;
        bool expected;
        const char* name;
    };
    const MarkerCase marker_cases[] = {
        {true, true, true, true, true, "report q32 parity after full self-test"},
        {false, true, true, true, false, "suppress q32 marker after unrelated self-test failure"},
        {true, false, true, true, false, "suppress q32 marker when QFUSE_HC is off"},
        {true, true, false, true, false, "suppress q32 marker when staged parity fails"},
        {true, true, true, false, false, "suppress q32 marker when P4 parity fails"},
    };
    for (const auto& test : marker_cases) {
        const bool got = strata::kernels::detail::fused_gr_p4_q32_parity_marker_ok(
            test.selftest_ok, test.qfuse_hc, test.staged_ok, test.p4_ok);
        if (got != test.expected) {
            ++failures;
            std::fprintf(stderr, "FAIL: %s expected=%d got=%d\n", test.name, test.expected, got);
        }
    }
    std::printf("fused_gr_p4_policy_test: %zu gate cases + %zu marker cases, %d failures\n",
                sizeof(cases) / sizeof(cases[0]), sizeof(marker_cases) / sizeof(marker_cases[0]), failures);
    return failures == 0 ? 0 : 1;
}
