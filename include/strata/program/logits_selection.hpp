#pragma once

#include <charconv>
#include <cstdint>
#include <string_view>

namespace strata::program::logits_selection {

// Storage selection only: callers must still condition on every input token.
// Positive decimal int64, with no sign, whitespace, suffix or overflow.
inline bool parse_stride(std::string_view text, int64_t& result) {
    if (text.empty() || text.front() < '0' || text.front() > '9') return false;
    int64_t value = 0;
    const auto parsed = std::from_chars(text.data(), text.data() + text.size(), value);
    if (parsed.ec != std::errc{} || parsed.ptr != text.data() + text.size() || value <= 0) return false;
    result = value;
    return true;
}

// Zero-based multiples of stride, followed by the final input position once.
inline constexpr bool selected(int64_t position, int64_t total, int64_t stride) {
    return total > 0 && stride > 0 && position >= 0 && position < total &&
           (position % stride == 0 || position == total - 1);
}

inline constexpr int64_t row_count(int64_t total, int64_t stride) {
    return total > 0 && stride > 0 ? (total - 1) / stride + 1 + ((total - 1) % stride != 0) : 0;
}

// The same count over the range a path ACTUALLY emits: the selected positions in [first_emit, total - 1].
// `row_count(total, stride)` is the `first_emit = 0` case - the per-token loop, which runs from position 0.
// A native (IQ) pack skips that loop entirely and emits only from `n_prompt - 1`, so its header must count
// over that range or it promises rows the file never contains.  `first_emit = 0` reproduces `row_count`
// exactly, which is what keeps the non-native header byte-for-byte what it was.
inline constexpr int64_t row_count_from(int64_t first_emit, int64_t total, int64_t stride) {
    if (total <= 0 || stride <= 0 || first_emit < 0 || first_emit >= total) return 0;
    const int64_t last = total - 1;
    // multiples of stride in [first_emit, last]: floor division, so the -1 below floors to -1
    const int64_t rows = last / stride - (first_emit == 0 ? -1 : (first_emit - 1) / stride);
    return rows + (last % stride != 0);   // the final input position, counted once, as `selected` does
}

// The non-native header must not move: `first_emit = 0` IS `row_count`, over a spread of totals and strides.
static_assert(row_count_from(0, 1, 1) == row_count(1, 1) && row_count_from(0, 10, 3) == row_count(10, 3) &&
              row_count_from(0, 10, 4) == row_count(10, 4) && row_count_from(0, 4096, 7) == row_count(4096, 7) &&
              row_count_from(0, 100, 1) == row_count(100, 1) && row_count_from(0, 7, 5) == row_count(7, 5));
// A native pack's count is the tail only: `n_prompt - 1 + max_new` positions, emitted from `n_prompt - 1`.
static_assert(row_count_from(1, 2, 1) == 1 && row_count_from(1, 9, 1) == 8 && row_count_from(9, 13, 2) == 2);

} // namespace strata::program::logits_selection
