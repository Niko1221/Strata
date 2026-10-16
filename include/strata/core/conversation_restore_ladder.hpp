// The restore ladder (opt-in, STRATA_RESTORE_LADDER=1): when the parked image the serve loop picked does not
// validate as a whole, the engine no longer throws the whole prefix away.  It descends the image's OWN checkpoints
// - the running-state prefixes it already carries - largest first, and restores the largest one that validates,
// re-reading only the tokens after it.  With the flag off nothing here is called: the reject path is the one that
// shipped (discard the image, fall back to the slot's own prefix).
//
// Why a shorter checkpoint is usable at all: attention is causal, so the K/V the image holds for tokens [0, N) is
// byte-for-byte the K/V of every shorter prefix; the running state (GDN/QSA/PLE) is checkpointed at each turn
// boundary, so one of them is a complete state at that prefix.  The restore therefore keeps the image's K/V whole
// and puts the checkpoint's state in place of the image's live state; the request rewrites every cell from the
// checkpoint on, so the stale tail is never read.
//
// This header is the ladder's PURE half: the switch, the ordering and the prefix/image filtering, no CUDA, no
// allocation beyond the returned vector.  The validation of a rung (running-state sizes, view_validate of the
// image, and the later stages' parts) and the restore itself stay in the engine (generate.cpp calls
// conversation_snapshot_validate_at / conversation_snapshot_restore_at, conversation_state.cpp owns them): they
// touch the session and CUDA.
#pragma once

#include "strata/core/conversation_cache.hpp"

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <cstdlib>
#include <utility>
#include <vector>

namespace strata::core {

// The switch: STRATA_RESTORE_LADDER set to a non-zero value enables the ladder.  Read on every call (getenv is
// cheap, and a test can flip it) rather than cached, so the serve loop and a test always agree.  With it off the
// reject path is the one that shipped: the image is discarded and the slot's own prefix is the fallback.
inline bool restore_ladder_enabled() {
    const char* v = std::getenv("STRATA_RESTORE_LADDER");
    return v != nullptr && std::atoi(v) != 0;
}

// One rung of the ladder: a checkpoint that is a strict prefix of the prompt, and the token count it restores.
struct RestoreLadderRung {
    size_t index = 0;      // index into the checkpoint vector
    int64_t tokens = 0;    // conversation_prefix(checkpoint, prompt): the restored prefix length
};

// The rungs a prompt can use, in DESCENDING token order: every checkpoint whose ids AND pictures are a strict
// prefix of `prompt` (conversation_prefix) and whose length is strictly greater than `floor` (the tokens this
// request already has without the image - its live state, a checkpoint, or an idle slot).  The sort is stable, so
// two checkpoints of the same length keep their stored order.  Pure: no CUDA, no K/V read, no image mutation.
template<class Token>
std::vector<RestoreLadderRung> restore_ladder_rungs(const std::vector<ConversationCheckpoint>& checkpoints,
                                                    const std::vector<Token>& prompt,
                                                    const std::vector<ConversationImageKey>& images,
                                                    int64_t floor) {
    std::vector<RestoreLadderRung> rungs;
    rungs.reserve(checkpoints.size());
    for (size_t i = 0; i < checkpoints.size(); ++i) {
        const int64_t n = conversation_prefix(checkpoints[i], prompt, images);
        if (n > floor) rungs.push_back({i, n});
    }
    // Stable so two rungs of the same length keep the image's own order (the deeper one, first).
    std::stable_sort(rungs.begin(), rungs.end(),
                     [](const RestoreLadderRung& a, const RestoreLadderRung& b) { return a.tokens > b.tokens; });
    return rungs;
}

// The first rung, largest first, that `usable` accepts.  `usable(index)` validates one checkpoint against the
// session (and, under --layer-split, every stage part); it may record the last rejection reason itself.  Returns
// true and sets `out` on the first acceptance, false (and `out` untouched) when no rung is usable.
template<class Token, class Usable>
bool restore_ladder_pick(const std::vector<ConversationCheckpoint>& checkpoints,
                         const std::vector<Token>& prompt,
                         const std::vector<ConversationImageKey>& images,
                         int64_t floor, Usable&& usable, RestoreLadderRung& out) {
    for (const RestoreLadderRung& rung : restore_ladder_rungs(checkpoints, prompt, images, floor))
        if (usable(rung.index)) { out = rung; return true; }
    return false;
}

// The ladder's decision for one image, gated by the switch: disabled returns false with no rung and does not even
// walk the rungs - exactly today's behaviour, where the caller falls back to its own prefix.  This is the single
// place the flag is honoured, so `enabled == false` is testable without the serve loop.
template<class Token, class Usable>
bool restore_ladder_plan(bool enabled, const std::vector<ConversationCheckpoint>& checkpoints,
                         const std::vector<Token>& prompt, const std::vector<ConversationImageKey>& images,
                         int64_t floor, Usable&& usable, RestoreLadderRung& out) {
    if (!enabled) return false;
    return restore_ladder_pick(checkpoints, prompt, images, floor, std::forward<Usable>(usable), out);
}

} // namespace strata::core
