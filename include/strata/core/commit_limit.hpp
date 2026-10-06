#pragma once
#include <algorithm>
#include <cstdint>
#include <span>

namespace strata::core {
// Only inputs corresponding to emitted outputs enter the committed session.
// A verified window may contain valid drafts beyond EOS or the request budget.
inline int emitted_prefix_length(std::span<const int32_t> picks, int64_t remaining,
                                 std::span<const int64_t> eos) {
    if (remaining <= 0) return 0;
    const size_t count = std::min(picks.size(), (size_t) remaining);
    for (size_t i = 0; i < count; ++i)
        if (std::find(eos.begin(), eos.end(), (int64_t) picks[i]) != eos.end()) return (int) i + 1;
    return (int) count;
}
}  // namespace strata::core
