#pragma once

#include <algorithm>

namespace strata::qwen35 {

// Polaris uses only the hybrid expert path; small row requests are raised to the
// engine's eight-row verification workspace minimum.
inline void clamp_gfx803_batch(int& chunk, int& hybrid_rows) {
    hybrid_rows = std::clamp(hybrid_rows, 8, 32);
    chunk = std::min(chunk, hybrid_rows);
}

}  // namespace strata::qwen35
