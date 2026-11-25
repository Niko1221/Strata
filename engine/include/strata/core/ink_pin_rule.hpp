#pragma once
// Inkling 2 Mini host expert arena: may it be pinned (page-locked)? A pinned arena lets the card read the cold experts itself (the exact lane's in-graph
// staging, the PCIe streaming of prefill), a pageable one goes through a bounce ring and a CPU copy. Pure function, no GPU, tested by ink_pin_rule_test.
//
// Windows (owner decision 11-23): the cap is min(16 GiB, 35% of the available RAM); STRATA_INK_PIN_GB=<n> replaces it by n GiB, =0 turns pinning off;
// whatever the cap, an arena that would leave less than 8 GiB of RAM available is not pinned (the desktop keeps its memory). Linux keeps its rule
// (the arena is pinned when it is under 45% of the available RAM, no size cap, no floor); STRATA_INK_PIN_GB applies to it as the cap in GiB, =0 turns it off.
// An unknown available RAM (0) pins up to 8 GiB on Windows and always on Linux, as before. The caller falls back to the pageable path with one INFO line
// when this says no, or when the allocation fails.
#include <algorithm>
#include <cstdint>
#include <cstdlib>

namespace strata {

struct InkPinRule {
    bool pin = false;
    uint64_t cap = 0;          // the size cap that applied (0: none)
    const char* why = "";      // the reason of a "no" / what decided a "yes"
};

inline InkPinRule ink_pin_rule(bool windows, uint64_t bytes, uint64_t avail, const char* env_gib) noexcept {
    constexpr uint64_t GiB = 1ull << 30;
    constexpr uint64_t kFloor = 8 * GiB;   // available RAM that must remain after pinning (Windows)
    InkPinRule r;
    if (env_gib != nullptr && env_gib[0] != 0) {
        const double n = std::atof(env_gib);
        if (!(n > 0.0)) { r.why = "STRATA_INK_PIN_GB=0 turns pinning off"; return r; }
        r.cap = (uint64_t)(n * (double)GiB);
        if (bytes > r.cap) { r.why = "larger than STRATA_INK_PIN_GB"; return r; }
        if (windows && avail != 0 && avail < bytes + kFloor) { r.why = "would leave less than 8 GiB of RAM available"; return r; }
        r.pin = true;
        r.why = "STRATA_INK_PIN_GB";
        return r;
    }
    if (windows) {
        r.cap = (avail != 0) ? std::min<uint64_t>(16 * GiB, (uint64_t)(0.35 * (double)avail)) : 8 * GiB;
        if (bytes > r.cap) { r.why = "larger than min(16 GiB, 35% of the available RAM)"; return r; }
        if (avail != 0 && avail < bytes + kFloor) { r.why = "would leave less than 8 GiB of RAM available"; return r; }
        r.pin = true;
        r.why = "within min(16 GiB, 35% of the available RAM)";
        return r;
    }
    if (avail == 0 || (double)bytes < 0.45 * (double)avail) { r.pin = true; r.why = "under 45% of the available RAM"; return r; }
    r.why = "larger than 45% of the available RAM";
    return r;
}

}  // namespace strata
