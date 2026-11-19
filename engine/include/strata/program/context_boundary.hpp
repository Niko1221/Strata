#pragma once

#include <algorithm>
#include <cstdint>

namespace strata::program {

// The server's usual eight-cell margin remains the default. Exact admission is
// opt-in until the full-context native run proves the final-cell device path.
constexpr bool serve_context_fits(int64_t prompt, int64_t max_new, int64_t capacity,
                                  bool exact_boundary) noexcept {
    if (prompt < 0 || max_new < 1 || capacity < 1 || prompt > capacity) return false;
    const int64_t margin = exact_boundary ? 0 : 8;
    return max_new <= capacity - prompt - margin;
}

// Speculative rows occupy positions [position, position + width). A last-cell
// request can use a narrower window without changing the requested output cap.
constexpr int serve_context_window(int width, int64_t position, int64_t capacity,
                                   bool exact_boundary) noexcept {
    if (!exact_boundary) return width;
    if (width < 1 || position < 0 || position >= capacity) return 0;
    return static_cast<int>(std::min<int64_t>(width, capacity - position));
}

}  // namespace strata::program
