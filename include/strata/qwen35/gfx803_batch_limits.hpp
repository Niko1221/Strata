#pragma once

#include <algorithm>

namespace strata::qwen35 {

// Polaris uses only the hybrid expert path; nonpositive row requests keep the
// engine's established effective minimum of one row.
inline void clamp_gfx803_batch(int& chunk, int& hybrid_rows) {
    hybrid_rows = std::clamp(hybrid_rows, 1, 32);
    chunk = std::min(chunk, hybrid_rows);
}

}  // namespace strata::qwen35
