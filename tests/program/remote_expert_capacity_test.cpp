#ifdef NDEBUG
#undef NDEBUG  // Keep assertions active in Release-mode CTest builds.
#endif
#include "strata/core/remote_expert_capacity.hpp"

#include <cassert>
#include <limits>

using strata::core::kRemoteExpertReserveBytes;
using strata::core::remote_expert_capacity_fits;

int main() {
    constexpr uint64_t reserve = kRemoteExpertReserveBytes;
    assert(remote_expert_capacity_fits(0, 0, reserve));
    assert(remote_expert_capacity_fits(0, 1, reserve + 1));
    assert(remote_expert_capacity_fits(128, 64, reserve + 192));
    assert(!remote_expert_capacity_fits(128, 65, reserve + 192));
    assert(!remote_expert_capacity_fits(0, 1, reserve));
    assert(!remote_expert_capacity_fits(9, 0, 8));
    constexpr uint64_t max = std::numeric_limits<uint64_t>::max();
    assert(!remote_expert_capacity_fits(max, 1, max));
}
