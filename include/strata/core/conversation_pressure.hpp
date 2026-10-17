// RAM pressure on the parked conversations.  A park is admitted against --conversation-cache-min-free-mib, but once
// parked an image stayed in RAM until a later park or restore evicted it - also when the host ran short later (other
// programs opened, the engine's own memory grew).  The serve loop calls this before a request and every few seconds
// while idle: below the floor the cache gives RAM back, oldest first, until the floor plus a hysteresis is free.
#pragma once

#include "strata/core/conversation_cache.hpp"

#include <cstddef>
#include <cstdint>
#include <optional>

#if defined(__GLIBC__)
#include <malloc.h>   // malloc_trim
#endif

namespace strata::core {

/// Freed images back to the OS.  glibc: a freed mmap chunk raises the mmap threshold (up to 32 MiB), so later 16 MiB
/// segments come from the heap and stay there after free - malloc_trim gives the free pages back.  Windows: the UCRT
/// heap releases blocks this size on free (measured: private bytes back to ~1 MiB after 2.1 GiB of mixed sizes).
inline void conversation_return_freed_heap() {
#if defined(__GLIBC__)
    malloc_trim(0);
#endif
}

/// One step of a release: the bytes it held (the cache's own accounting) and whether it was a parked conversation
/// (else the retained K/V or an idle --batch slot's checkpoints).
struct PressureUnit {
    uint64_t bytes = 0;
    bool dialog = false;
};

struct PressureRelease {
    bool below = false;                  ///< the first reading was under the floor
    std::optional<uint64_t> before, after;   ///< available RAM at the first and the last reading
    size_t dialogs = 0, others = 0;      ///< parked conversations / other units released
    uint64_t bytes = 0;                  ///< what they held
    bool exhausted = false;              ///< nothing left to release while still under floor + hysteresis
    size_t released() const { return dialogs + others; }
};

/// `sample()`: the available RAM now (nullopt: unknown - nothing is released on an unknown reading, as the park gate
/// fails closed).  Only when it is under `floor`: `settle()` (the allocator's freed heap) and
/// then `next()` one unit at a time, oldest first, settling and sampling after each, until floor + hysteresis is
/// available, `next()` has nothing left (nullopt), or `bound` units went.
template<class Sample, class Settle, class Next>
PressureRelease conversation_pressure_release(uint64_t floor, uint64_t hysteresis, Sample sample, Settle settle,
                                              Next next, size_t bound) {
    PressureRelease out;
    std::optional<uint64_t> room = sample();
    out.before = out.after = room;
    if (!room || *room >= floor) return out;
    out.below = true;
    const uint64_t target = floor + hysteresis < floor ? UINT64_MAX : floor + hysteresis;
    settle();   // freed memory may simply not be visible yet
    room = sample();
    out.after = room;
    while (room && *room < target) {
        if (out.released() >= bound) break;
        const std::optional<PressureUnit> unit = next();
        if (!unit) { out.exhausted = true; break; }
        (unit->dialog ? out.dialogs : out.others) += 1;
        out.bytes += unit->bytes;
        settle();
        room = sample();
        out.after = room;
    }
    return out;
}

/// The cache's own units, oldest first: the retained K/V (an optimization only), then the parked conversation unused
/// the longest (evict_oldest, the cache's own LRU eviction: dropped; pinned shared prefixes stay).
inline std::optional<PressureUnit> conversation_cache_release_one(ConversationCache& cache) {
    if (const size_t r = cache.retained_bytes()) {
        { auto gone = cache.take_reuse(); }   // freed here
        return PressureUnit{r, false};
    }
    const size_t before = cache.bytes();
    if (!cache.evict_oldest()) return std::nullopt;
    return PressureUnit{before - cache.bytes(), true};
}

} // namespace strata::core
