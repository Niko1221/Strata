// The SHARED CONVERSATION-CACHE CORE's physical-RAM admission guard (issue #57).  Imported as the DECLARED
// boundary only (docs/nvme-kv-cache-convergence.md step 2): nothing in this tree calls it, and its implementation
// (`src/core/conversation_memory.cpp`) is deliberately NOT imported yet - it belongs to the RAM-cache policy, whose
// wiring is a later decision.  Do not call these until that .cpp lands.
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
