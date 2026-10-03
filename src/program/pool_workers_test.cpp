#include "strata/program/pool_workers.hpp"

#include <cstdio>

int main() {
    using strata::program::select_pool_workers;
    constexpr auto cpu = "AMD Ryzen Threadripper 3990X 64-Core Processor";
    struct Case {
        const char* name;
        int requested, available;
        const char* cpu;
        const char* arch;
        bool windows_hip, single_gpu;
        int expected;
    };
    const Case cases[] = {
        {"measured PC", 0, 63, cpu, "gfx1030", true, true, 31},
        {"HIP feature suffix", 0, 63, cpu, "gfx1030:xnack-", true, true, 31},
        {"CPUID padding", 0, 63, "  AMD Ryzen Threadripper 3990X  ", "gfx1030", true, true, 31},
        {"explicit original count", 63, 63, cpu, "gfx1030", true, true, 63},
        {"explicit calibrated count", 15, 63, cpu, "gfx1030", true, true, 15},
        {"one worker", 1, 63, cpu, "gfx1030", true, true, 1},
        {"fewer available cores", 0, 15, cpu, "gfx1030", true, true, 0},
        {"already 31 available", 0, 31, cpu, "gfx1030", true, true, 0},
        {"unknown topology", 0, 0, cpu, "gfx1030", true, true, 0},
        {"Linux or CUDA", 0, 63, cpu, "gfx1030", false, true, 0},
        {"layer split or remote experts", 0, 63, cpu, "gfx1030", true, false, 0},
        {"other RDNA GPU", 0, 63, cpu, "gfx1100", true, true, 0},
        {"unknown GPU", 0, 63, cpu, "", true, true, 0},
        {"architecture boundary", 0, 63, cpu, "gfx10300", true, true, 0},
        {"unmeasured Zen2 SKU", 0, 31, "AMD Ryzen Threadripper 3970X 32-Core Processor", "gfx1030", true, true, 0},
        {"unmeasured Zen2 PRO", 0, 63, "AMD Ryzen Threadripper PRO 3995WX 64-Cores", "gfx1030", true, true, 0},
        {"newer Threadripper", 0, 63, "AMD Ryzen Threadripper PRO 5995WX 64-Cores", "gfx1030", true, true, 0},
        {"CPU model boundary", 0, 63, "AMD Ryzen Threadripper 3990XX", "gfx1030", true, true, 0},
        {"unknown CPU", 0, 63, "", "gfx1030", true, true, 0},
    };
    for (const auto& c : cases) {
        const int got = select_pool_workers(c.requested, c.available, c.cpu, c.arch,
                                            c.windows_hip, c.single_gpu);
        if (got != c.expected) {
            std::fprintf(stderr, "%s: expected %d, got %d\n", c.name, c.expected, got);
            return 1;
        }
    }
    std::puts("pool worker defaults: 19 cases passed");
    return 0;
}
