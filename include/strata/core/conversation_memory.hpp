#pragma once

#include <cstdint>
#include <istream>
#include <optional>

namespace strata::core {

// Host physical memory, not swap/commit or a container/job memory reservation.
// Unknown telemetry is deliberately distinct from a measured zero.
std::optional<uint64_t> conversation_available_memory();
std::optional<uint64_t> conversation_mem_available(std::istream& meminfo);
// Hand the free heap the allocator still holds back to the kernel, so that a parked conversation that was dropped shows
// up in conversation_available_memory(). glibc keeps freed segments in its heap once its mmap threshold has risen;
// elsewhere freed segments of this size already go back at once and this does nothing.
void conversation_release_freed_memory();

inline bool conversation_memory_admit(std::optional<uint64_t> available,
                                      uint64_t allocation, uint64_t floor) {
    return available && *available >= floor && allocation <= *available - floor;
}

} // namespace strata::core
