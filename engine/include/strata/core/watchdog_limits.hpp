// strata/core/watchdog_limits.hpp - the serve watchdog's two time limits, parsed without undefined behaviour (header-only, no GPU).
//
// STRATA_WATCHDOG_S (default 60, 0 = the watchdog is off) and STRATA_WATCHDOG_IO_S (default 10 x the first, 0 = no allowance while the file
// tier is read) were read with std::atoi.  atoi of a value outside int is undefined: #6861's server log printed "no progress for
// -1195029860 s" and "no step finished for -1907795513 s ... waiting up to 2079576072 s".  A negative limit made the first silent second
// "stuck"; so the limits are parsed with strtol, an unparsable or out-of-range value falls back to the default, and the messages print the
// seconds the engine really waited.
#pragma once

#include <cerrno>
#include <cstdint>
#include <cstdlib>

namespace strata::core {

/// Seconds of a watchdog variable: `env` null or empty -> `def`; a whole non-negative number up to `max_s` -> itself; anything else (text after
/// the digits, a negative number, a value past max_s or past long) -> `def`.  `*bad` (optional) is set when a value was given and refused.
inline int watchdog_seconds(const char* env, int def, bool* bad = nullptr, int max_s = 7 * 24 * 3600) noexcept {
    if (bad != nullptr) *bad = false;
    if (env == nullptr || env[0] == '\0') return def;
    char* end = nullptr;
    errno = 0;
    const long v = std::strtol(env, &end, 10);
    if (errno != 0 || end == env || *end != '\0' || v < 0 || v > (long) max_s) {
        if (bad != nullptr) *bad = true;
        return def;
    }
    return (int) v;
}

}  // namespace strata::core
