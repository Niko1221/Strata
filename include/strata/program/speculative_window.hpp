// The target's equality policy and the single retained model/output boundary.
#pragma once
#include <algorithm>
#include <cstdint>
#include <stdexcept>
#include <vector>

namespace strata::program {

struct RetainedWindow {
    int count; // consumed inputs == emitted outputs; last output remains pending
    bool eos;
};

inline RetainedWindow retained_window(const int32_t* inputs, const int32_t* selected,
                                     int rows, int64_t remaining,
                                     const std::vector<int64_t>& end_ids, bool stop_eos = true) {
    if (!inputs || !selected || rows < 1 || remaining < 1)
        throw std::runtime_error("invalid speculative commit boundary");
    const int limit = (int) std::min<int64_t>(rows, remaining);
    for (int row = 0; row < limit; ++row) {
        const bool eos = stop_eos && std::find(end_ids.begin(), end_ids.end(), selected[row]) != end_ids.end();
        if (eos || row + 1 == limit || inputs[row + 1] != selected[row]) return {row + 1, eos};
    }
    throw std::runtime_error("missing speculative commit boundary");
}

} // namespace strata::program
