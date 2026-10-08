#include "strata/platform/shared_memory_budget.hpp"
#include <cstdio>
#include <limits>

int main() {
    using strata::platform::shared_allocation_budget;
    constexpr uint64_t GiB = 1ull << 30;
    struct Case { uint64_t available, total, headroom, expected; };
    const Case cases[] = {
        {32 * GiB, 32 * GiB, 6 * GiB, 26 * GiB},
        {2 * GiB, 32 * GiB, 6 * GiB, 0},
        {6 * GiB, 32 * GiB, 6 * GiB, 0},
        {0, 32 * GiB, 0, 0},
        {32 * GiB, 16 * GiB, 6 * GiB, 16 * GiB},
        {8 * GiB, 32 * GiB, 6 * GiB, 2 * GiB}, // cgroup-bounded availability
        {32 * GiB, 0, 6 * GiB, 0},
        {32 * GiB, 32 * GiB, 0, 32 * GiB},
        {std::numeric_limits<uint64_t>::max(), 32 * GiB, 6 * GiB, 32 * GiB},
        {32 * GiB, 32 * GiB, std::numeric_limits<uint64_t>::max(), 0},
    };
    for (const auto& c : cases) {
        if (shared_allocation_budget(c.available, c.total, c.headroom) != c.expected) {
            std::fprintf(stderr, "shared memory budget mismatch\n");
            return 1;
        }
    }
    std::puts("shared memory budget: PASS (10 cases)");
    return 0;
}
