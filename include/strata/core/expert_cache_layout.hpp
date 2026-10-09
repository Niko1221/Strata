#pragma once

#include <cstddef>
#include <cstdint>
#include <cstring>
#include <vector>

namespace strata::core::detail {

inline constexpr uint64_t kCacheGemmLimit = 1ull << 32;
inline constexpr uint64_t kCacheLoanLimit = 39 * (1ull << 30) / 10;

inline bool cache_reverse(uint64_t bytes, const char* override_value) {
    if (override_value && std::strcmp(override_value, "0") == 0) return false;
    if (override_value && std::strcmp(override_value, "1") == 0) return true;
    return bytes > kCacheGemmLimit;
}

// Keep the total in the last entry; only slot addresses change.
inline void reverse_cache_offsets(std::vector<uint64_t>& off) {
    const uint64_t total = off.back();
    for (size_t i = 0; i + 1 < off.size(); ++i) off[i] = total - off[i + 1];
}

// Logical prefix size, also used by the resize API. A reversed prefix is at the high end of the arena.
inline uint64_t cache_prefix_bytes(int64_t slots, const uint64_t* off, bool reversed, int64_t n) {
    if (n <= 0) return 0;
    return reversed ? off[slots] - off[n - 1] : off[n];
}

inline uint64_t cache_tail_bytes(int64_t slots, uint64_t blob, const uint64_t* off,
                                 bool reversed, int64_t first) {
    if (first >= slots) return 0;
    if (!off) return (uint64_t) (slots - first) * blob;
    if (reversed) return first == 0 ? off[slots] : off[first - 1];
    return off[slots] - off[first];
}

inline int64_t cache_tail_slots(int64_t slots, uint64_t blob, const uint64_t* off,
                                bool reversed, uint64_t need) {
    if (need == 0) return 0;
    if (blob == 0 && !off) return slots + 1;
    if (!off) return (int64_t) ((need + blob - 1) / blob);
    int64_t k = 0;
    while (k < slots && cache_tail_bytes(slots, blob, off, reversed, slots - k) < need) ++k;
    if (reversed && (need > kCacheLoanLimit ||
        cache_tail_bytes(slots, blob, off, reversed, slots - k) > kCacheLoanLimit)) return slots + 1;
    return k;
}

}  // namespace strata::core::detail
