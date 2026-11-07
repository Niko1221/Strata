// Optional SAVE and RESTORE admission: free reconstructible host caches, never the live device state.
#pragma once

#include "strata/core/conversation_file.hpp"
#include "strata/core/conversation_memory.hpp"

namespace strata::core {

struct SessionSaveReclaimed {
    size_t checkpoints = 0, parked = 0, bytes = 0;
};

namespace session_reclaim_detail {
// The shared ladder behind both admission hooks: retained K/V reuse first, then the oldest unpinned
// parked conversations, then the chain's own checkpoints from the back - preserving every explicit
// pin and exactly the checkpoint the file selection would have made.  `available()` is probed once
// before any eviction and re-probed after each one: freed cache capacity is not necessarily RAM the
// allocator returned, and eviction stops the moment a probe clears `need` over `floor`.  Unknown
// telemetry or an overflowed estimate never justifies evicting a cache.
template<class Probe>
SessionSaveReclaimed ladder(std::vector<ConversationCheckpoint>& chain, ConversationCache& cache,
                            uint64_t need, uint64_t floor, Probe available) {
    SessionSaveReclaimed out;
    const auto* selected = session_deepest_checkpoint(chain);
    auto room = available();
    auto done = [&] { return !room || need == UINT64_MAX || conversation_memory_admit(room, need, floor); };
    if (done()) return out;
    if (cache.retained_bytes()) {
        out.bytes += cache.retained_bytes();
        { auto unused = cache.take_reuse(); }
        room = available();
        if (done()) return out;
    }
    while (true) {
        const size_t before = cache.bytes();
        if (!cache.evict_oldest()) break; // pinned conversations stay parked
        ++out.parked;
        out.bytes += before - cache.bytes();
        room = available();
        if (done()) return out;
    }
    size_t keep = selected ? size_t(selected - chain.data()) : chain.size();
    // Erase from the back to preserve order, ties and the selected checkpoint's storage.
    for (size_t i = chain.size(); i-- > 0;) {
        if (i == keep || chain[i].pinned) continue;
        out.bytes += chain[i].bytes();
        chain.erase(chain.begin() + static_cast<std::ptrdiff_t>(i));
        if (i < keep) --keep;
        ++out.checkpoints;
        room = available();
        if (done()) break;
    }
    return out;
}
} // namespace session_reclaim_detail

// Called BEFORE session_save_checkpoints selects a pointer into chain; `need` is what that save
// would peak at (session_save_peak_bytes).
template<class Probe>
SessionSaveReclaimed session_save_reclaim(std::vector<ConversationCheckpoint>& chain,
                                          ConversationCache& cache, const SessionSaveLive& live,
                                          uint64_t floor, Probe available) {
    const uint64_t need = session_save_peak_bytes(session_deepest_checkpoint(chain), live);
    return session_reclaim_detail::ladder(chain, cache, need, floor, available);
}

// Called BEFORE a RESTORE's read admit (a session_file_read `limits.admit` that the probe says it
// would refuse): `need_bytes` is what parsing that file peaks at (session_read_peak_bytes).  The
// same ladder frees reconstructible host caches - never the live device state - so a big parked
// window can still be read back; when the ladder runs dry and the probe still refuses, the caller
// keeps its typed `memory` refusal.  The engine's checkpoint selection (the deepest one and every
// pin) survives eviction exactly as for a save: the file would restore the same resume point.
template<class Probe>
SessionSaveReclaimed session_read_reclaim(ConversationCache& cache, std::vector<ConversationCheckpoint>& chain,
                                          uint64_t need_bytes, uint64_t floor, Probe available) {
    return session_reclaim_detail::ladder(chain, cache, need_bytes, floor, available);
}

} // namespace strata::core
