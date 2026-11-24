#pragma once
#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <limits>
#include <vector>

namespace strata::hy4_flash {
// Exact online counts, lazily halved by processed-token age. Single host owner.
// Only route IDs already observed by plan_slots contribute; no offline profile.
class ExpertHeat {
    struct Entry { uint16_t count = 0; uint64_t epoch = 0; };
    std::vector<Entry> entries_;
    uint64_t tokens_ = 0;
public:
    static constexpr uint64_t decay_tokens = 100;
    static constexpr int sample_slots = 32;
    void reset(std::size_t keys) { entries_.assign(keys, {}); tokens_ = 0; }
    void advance_tokens(uint64_t count) { tokens_ += count; }
    uint16_t score(std::size_t key) {
        Entry& e = entries_[key];
        const uint64_t epoch = tokens_ / decay_tokens;
        const uint64_t age = epoch - e.epoch;
        if (age) { e.count = age >= 16 ? 0 : uint16_t(e.count >> age); e.epoch = epoch; }
        return e.count;
    }
    void observe(std::size_t key) {
        (void) score(key);
        if (entries_[key].count != std::numeric_limits<uint16_t>::max()) ++entries_[key].count;
    }
    // Return a pool-local victim only if admission beats its frequency. Never
    // touch ownership, CLOCK bits, or the current group's protected readers.
    int victim(std::size_t incoming, int g0, int permanent, int& cursor,
               const std::vector<int32_t>& keys, const std::vector<uint32_t>& claimed,
               uint32_t mark) {
        int best = -1;
        uint16_t least = std::numeric_limits<uint16_t>::max();
        for (int i = 0, n = std::min(permanent, sample_slots); i < n; ++i) {
            const int candidate = cursor;
            cursor = (cursor + 1) % permanent;
            const std::size_t at = std::size_t(g0 + candidate);
            if (claimed[at] == mark || keys[at] < 0) continue;
            const uint16_t value = score(std::size_t(keys[at]));
            if (best < 0 || value < least) { best = candidate; least = value; }
        }
        return best >= 0 && score(incoming) > least ? best : -1;
    }
};
}
