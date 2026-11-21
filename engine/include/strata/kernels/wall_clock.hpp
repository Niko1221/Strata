#pragma once

#include <cstdint>

#if defined(__CUDACC__) || defined(__HIPCC__)
#define STRATA_CLOCK_HD __host__ __device__
#else
#define STRATA_CLOCK_HD
#endif

namespace strata::kernels {

// HIP reports WallClockRate in kHz. The caller requires a nonzero rate.
// Divide first so a long-running counter does not overflow ticks * 1,000,000.
STRATA_CLOCK_HD inline uint64_t wall_clock_ticks_to_ns(uint64_t ticks, uint32_t rate_khz) {
    return (ticks / rate_khz) * 1000000ull + (ticks % rate_khz) * 1000000ull / rate_khz;
}

}  // namespace strata::kernels

#undef STRATA_CLOCK_HD
