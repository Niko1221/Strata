#pragma once

#include <charconv>
#include <cstdint>
#include <cstring>
#include <string>

namespace strata::core::detail {

// This allowance includes helper work buffers, not just free memory after loading.
inline bool remote_reserve_bytes(const char* value, uint64_t work_bytes,
                                 uint64_t& reserve_bytes, std::string& err) {
    uint32_t mib = 512;
    if (value) {
        const char* end = value + std::strlen(value);
        const auto parsed = std::from_chars(value, end, mib);
        if (parsed.ec != std::errc{} || parsed.ptr != end || mib > 8192) {
            err = "STRATA_REMOTE_RESERVE_MIB must be a decimal integer from 0 to 8192";
            return false;
        }
    }
    constexpr uint64_t MiB = 1ull << 20;
    // Round before adding headroom, so even an impossible work size cannot wrap.
    const uint64_t minimum_mib = work_bytes / MiB + (work_bytes % MiB != 0) + 16;
    if (mib < minimum_mib) {
        err = "helper allowance requires at least " + std::to_string(minimum_mib) +
              " MiB for work buffers and driver headroom";
        return false;
    }
    reserve_bytes = uint64_t(mib) * MiB;
    return true;
}

// Used for both automatic admission and an explicit helper slot count.
inline bool remote_cache_fits(uint64_t free_bytes, uint64_t selected_bytes,
                              uint64_t next_bytes, uint64_t reserve_bytes) {
    return reserve_bytes <= free_bytes && selected_bytes <= free_bytes - reserve_bytes &&
           next_bytes <= free_bytes - reserve_bytes - selected_bytes;
}

} // namespace strata::core::detail
