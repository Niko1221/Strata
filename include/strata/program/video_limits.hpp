#pragma once

#include <cstdint>

namespace strata::program::video_limits {
inline constexpr uint32_t max_frames = 4096;
inline constexpr uint64_t max_rows = 65536;
inline constexpr uint64_t max_rgb_bytes = 1ull << 30;
inline constexpr uint64_t max_wire_bytes = 768ull << 20;
inline constexpr double max_duration_s = 3600.0;
}  // namespace strata::program::video_limits
