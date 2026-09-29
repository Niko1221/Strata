// The SHARED CONVERSATION-CACHE CORE's physical-RAM admission guard (issue #57).  Imported as the DECLARED
// boundary only (docs/nvme-kv-cache-design.md step 2): nothing in the engine calls it.  Step 4 added its
// implementation (`src/core/conversation_memory.cpp`) for `conversation_memory_test` to link - a fixture target,
// not `strata_engine` - so the guard is still declared-and-unwired in the engine, and C10 (admitting a restore
// against physical RAM) stays open.
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
