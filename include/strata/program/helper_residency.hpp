#pragma once

#include <cstddef>
#include <cstdint>
#include <vector>

namespace strata::program {

inline bool helper_expert_reserved(const std::vector<uint8_t>& owners, std::size_t index, bool enabled) {
    return enabled && index < owners.size() && owners[index] != 0;
}

} // namespace strata::program
