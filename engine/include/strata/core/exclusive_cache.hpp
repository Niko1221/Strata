// include/strata/core/exclusive_cache.hpp - E1, the exclusive expert cache (STRATA_EXCLUSIVE_CACHE=1 /
// STRATA_ADAPT_FETCH_RESIDENT=1 with --resident-experts): fetch_blobs swaps evicted VRAM victims back into
// the promoted expert's RAM slot in-place with zero SSD reads.
#pragma once

#include <cstdint>

#if defined(__CUDACC__) || defined(__HIPCC__)
#define STRATA_EXCL_HD __host__ __device__
#else
#define STRATA_EXCL_HD
#endif

namespace strata::core::exclusive {

STRATA_EXCL_HD inline unsigned long long fetch_target(unsigned long long slot, bool swap) { return slot | (swap ? 1ull : 0ull); }
STRATA_EXCL_HD inline unsigned long long fetch_slot(unsigned long long target) { return target & ~1ull; }
STRATA_EXCL_HD inline bool fetch_swaps(unsigned long long target) { return (target & 1ull) != 0; }

inline const char* start_refusal(bool copy_ok, uint64_t pinned, uint64_t resident, int64_t lent_kept, int64_t lent) {
    if (!copy_ok) return "the RAM copy does not fit";
    if (pinned < resident) return "the RAM copy is not page-locked whole";
    if (lent_kept < lent) return "the prompt path's lend region does not fit in RAM beside it";
    return nullptr;
}

}  // namespace strata::core::exclusive

#undef STRATA_EXCL_HD
