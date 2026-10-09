// Reserve-based VRAM cap arithmetic. No device calls: also used by the CPU-only tests.
#pragma once

#include <algorithm>
#include <cerrno>
#include <cmath>
#include <cstdint>
#include <cstdlib>
#include <cstring>

namespace strata::program::vram_cap {

// A command-line value wins over the environment, including 1 (cap off).
// An absent value leaves exactly the old path; malformed/NaN/infinite values are errors.
inline bool parse_fraction(const char* cli, const char* env, double& fraction) {
    const char* text = cli != nullptr ? cli : env;
    if (text == nullptr) { fraction = 1.0; return true; }
    char* end = nullptr;
    errno = 0;
    const double f = std::strtod(text, &end);
    // ERANGE can also mean a positive subnormal; the finite/range checks admit those, but reject
    // overflow and underflow to zero, so the accepted domain really is every finite 0 < F <= 1.
    if (end == text || *end != '\0' || !std::isfinite(f) || !(f > 0.0 && f <= 1.0))
        return false;
    fraction = f;
    return true;
}

// Quality is opt-in. It also works with the cap off for a same-contract A/B reference.
// The fraction controls memory only; without the mode flag, Fast keeps the existing CPU/GPU split.
enum class Mode { Fast, Quality };

inline bool parse_mode(const char* text, Mode& mode) {
    if (text == nullptr || std::strcmp(text, "fast") == 0) { mode = Mode::Fast; return true; }
    if (std::strcmp(text, "quality") == 0) { mode = Mode::Quality; return true; }
    return false;
}

inline bool quality_active(Mode mode) {
    return mode == Mode::Quality;
}

// Use the hit kernel for every miss, not a link-probed fraction of them. Also applies to serve tuning keys.
inline double pcie_fraction(Mode mode, double requested) {
    return quality_active(mode) ? 1.0 : requested;
}

// One fixed pre-touch haircut; never derive the number of slots from a WDDM post-touch reading.
inline uint64_t startup_haircut_bytes(double fraction, bool wddm) {
    return fraction < 1.0 && wddm ? (1ull << 30) : 0;
}

// Round UP, not to nearest: even a fractional MiB must stay outside the budget.
inline int64_t floor_mib(uint64_t total_bytes, double fraction) {
    if (fraction >= 1.0) return 0;
    return (int64_t) std::ceil((long double) total_bytes * (1.0L - (long double) fraction) / 1048576.0L);
}

inline int64_t reserve_mib(int64_t current_mib, uint64_t total_bytes, double fraction) {
    if (fraction >= 1.0) return current_mib;   // cap off does not alter any existing reserve
    return std::max(current_mib, floor_mib(total_bytes, fraction));
}

// Late engine buffers are booked separately; they must not consume the cap's free-VRAM floor.
inline uint64_t cache_room(uint64_t free_bytes, int64_t reserve_mib, uint64_t late_bytes = 0) {
    const uint64_t reserve = (uint64_t) reserve_mib * 1048576;
    if (free_bytes <= reserve || free_bytes - reserve <= late_bytes) return 0;
    return free_bytes - reserve - late_bytes;
}

// A post-touch reading is an acceptance check only, NOT input to another sizing attempt.
// late_bytes excludes the pre-touch haircut: it compensates telemetry bias, not a later allocation.
inline bool post_touch_fits(uint64_t free_bytes, int64_t reserve_mib, uint64_t late_bytes) {
    const uint64_t reserve = (uint64_t) reserve_mib * 1048576;
    return free_bytes >= reserve && free_bytes - reserve >= late_bytes;
}

}  // namespace strata::program::vram_cap
