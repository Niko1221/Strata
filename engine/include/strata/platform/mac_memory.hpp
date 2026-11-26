#pragma once

#include <algorithm>
#include <cstdint>

namespace strata::platform {

// HOST_VM_INFO64 counts speculative file pages in free_count. Only the remaining
// file-backed pages can contribute to the estimate for the inactive queue.
inline uint64_t mac_available_pages(uint64_t free, uint64_t purgeable, uint64_t inactive,
                                    uint64_t speculative, uint64_t file_backed) {
    const uint64_t other_file_pages = file_backed - std::min(file_backed, speculative);
    return free + purgeable + std::min(inactive, other_file_pages);
}

}  // namespace strata::platform
