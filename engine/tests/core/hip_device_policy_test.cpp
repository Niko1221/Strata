#include "strata/core/hip_device_policy.hpp"
#include "strata/kernels/wall_clock.hpp"

#include <cstdio>
#include <string>

int main() {
    using strata::core::hip_arch_problem;
    struct Case {
        const char* name;
        const char* raw;
        int wave;
        const char* compiled;
        int expected_wave;
        bool accepted;
    };
    const Case cases[] = {
        {"MI210", "gfx90a", 64, "gfx90a", 64, true},
        {"ROCm feature suffix", "gfx90a:sramecc+:xnack-", 64, "gfx90a", 64, true},
        {"compiled feature suffix", "gfx90a:xnack-", 64, "gfx90a:sramecc+:xnack-", 64, true},
        {"exact list member", "gfx90a", 64, "gfx906,gfx90a,gfx942", 64, true},
        {"different GCN", "gfx906", 64, "gfx90a", 64, false},
        {"different CDNA", "gfx908", 64, "gfx90a", 64, false},
        {"different CDNA generation", "gfx942", 64, "gfx90a", 64, false},
        {"RDNA", "gfx1100", 32, "gfx90a", 64, false},
        {"architecture prefix", "gfx90abc", 64, "gfx90a", 64, false},
        {"compiled token prefix", "gfx90a", 64, "gfx90abc,gfx906", 64, false},
        {"missing architecture", nullptr, 64, "gfx90a", 64, false},
        {"empty architecture", "", 64, "gfx90a", 64, false},
        {"empty base", ":xnack-", 64, "gfx90a", 64, false},
        {"architecture whitespace", "gfx90a ", 64, "gfx90a", 64, false},
        {"wrong wave", "gfx90a", 32, "gfx90a", 64, false},
        {"missing wave", "gfx90a", 0, "gfx90a", 64, false},
        {"missing compiled list", "gfx90a", 64, nullptr, 64, false},
        {"empty compiled list", "gfx90a", 64, "", 64, false},
        {"uncompiled architecture", "gfx90a", 64, "gfx906,gfx942", 64, false},
        {"RDNA policy remains possible", "gfx1100:xnack-", 32, "gfx1100,gfx1201", 32, true},
        {"RDNA rejects wave64", "gfx1100", 64, "gfx1100", 32, false},
    };
    int failures = 0;
    for (const auto& c : cases) {
        const std::string why = hip_arch_problem(c.raw, c.wave, c.compiled, c.expected_wave);
        if (why.empty() != c.accepted) {
            std::fprintf(stderr, "%s: expected %s, got '%s'\n", c.name,
                         c.accepted ? "accept" : "reject", why.c_str());
            ++failures;
        }
    }
    const std::string wrong_wave = hip_arch_problem("gfx90a:xnack-", 32, "gfx90a", 64);
    if (wrong_wave.find("64") == std::string::npos) {
        std::fprintf(stderr, "wave rejection must identify the expected wave width\n");
        ++failures;
    }
    struct ClockCase {
        uint64_t ticks;
        uint32_t rate_khz;
        uint64_t nanoseconds;
    };
    const ClockCase clock_cases[] = {
        {0, 25000, 0},
        {1, 25000, 40},
        {17, 25000, 680},
        {1, 100000, 10},
        {123456789, 100000, 1234567890},
        {1, 300, 3333},
        {3, 300, 10000},
        {4, 300, 13333},
        // Both final answers fit; multiplying ticks by 1,000,000 first would overflow.
        {1000000000000000000ull, 100000, 10000000000000000000ull},
        {UINT64_MAX, 1000000, UINT64_MAX},
    };
    for (const auto& c : clock_cases) {
        const uint64_t actual = strata::kernels::wall_clock_ticks_to_ns(c.ticks, c.rate_khz);
        if (actual != c.nanoseconds) {
            std::fprintf(stderr, "wall clock: %llu ticks at %u kHz yielded %llu ns, expected %llu ns\n",
                         (unsigned long long) c.ticks, c.rate_khz, (unsigned long long) actual,
                         (unsigned long long) c.nanoseconds);
            ++failures;
        }
    }
    if (failures) return 1;
    std::puts("HIP device policy and wall-clock conversion checks passed");
    return 0;
}
