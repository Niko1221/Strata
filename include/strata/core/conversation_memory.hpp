#pragma once

#include <cstdint>
#include <istream>
#include <optional>

namespace strata::core {

// Host physical memory, not swap/commit or a container/job memory reservation.
// Unknown telemetry is deliberately distinct from a measured zero.
std::optional<uint64_t> conversation_available_memory();
std::optional<uint64_t> conversation_mem_available(std::istream& meminfo);

inline bool conversation_memory_admit(std::optional<uint64_t> available,
                                      uint64_t allocation, uint64_t floor) {
    return available && *available >= floor && allocation <= *available - floor;
}

} // namespace strata::core
