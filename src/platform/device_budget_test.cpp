#include "strata/platform/device_budget.hpp"
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <limits>

int main() {
    strata::platform::DeviceBudget budget(10240);
    auto require = [](bool ok) { if (!ok) { std::fputs("budget invariant failed\n", stderr); std::exit(1); } };
    require(budget.reserve(9000));
    require(!budget.reserve(1241));
    require(budget.used() == 9000 && budget.peak() == 9000);
    require(budget.reserve(1240) && budget.available() == 0);
    require(!budget.reserve(std::numeric_limits<uint64_t>::max()));
    require(!budget.release(10241));
    require(budget.release(1240) && budget.reserve(1240)); // failed/grown buffers restore reservations
    require(budget.release(10240) && budget.used() == 0 && budget.peak() == 10240);
    strata::platform::DeviceBudget empty;
    require(!empty.reserve(1));
    std::puts("device budget invariants passed");
}
