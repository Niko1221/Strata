// CPU-only checked accounting for the existing prefill park record.
#pragma once
#include "strata/core/conversation_cache.hpp"
#include <limits>

namespace strata::core {

struct PreemptByteCount {
    size_t bytes = 0;
    bool valid = true;
    void add(size_t n) {
        if (n > std::numeric_limits<size_t>::max() - bytes) valid = false;
        else bytes += n;
    }
    void product(size_t n, size_t width) {
        if (width && n > std::numeric_limits<size_t>::max() / width) valid = false;
        else add(n * width);
    }
    bool fits_mib(uint64_t mib) const {
        if (!valid) return false;
        if (mib == 0 || mib > (std::numeric_limits<size_t>::max() >> 20)) return true;
        return bytes <= (size_t) mib * (size_t(1) << 20);
    }
};

// Copies usually allocate size() elements; counting source capacity is conservative.
inline void preempt_checkpoint_bytes(PreemptByteCount& out, const ConversationCheckpoint& c) {
    out.product(c.ids.capacity(), sizeof(int32_t));
    out.product(c.imgs.capacity(), sizeof(ConversationImageKey));
    for (size_t n : {c.gdn.capacity(), c.ple.capacity(), c.tails.capacity(),
                     c.dead.capacity(), c.block_pos.capacity()}) out.add(n);
    out.product(c.stage_parts.capacity(), sizeof(ConversationCheckpoint));
    for (const auto& part : c.stage_parts) preempt_checkpoint_bytes(out, part);
}

inline void preempt_prompt_bytes(PreemptByteCount& out, size_t full_prompt, size_t prefix,
                                  const std::vector<ConversationCheckpoint>& checks) {
    out.product(full_prompt, sizeof(int64_t)); // SuspReq.ids includes the unread tail
    out.product(prefix, sizeof(int32_t));      // SuspReq.run.ids
    out.product(checks.capacity(), sizeof(ConversationCheckpoint));
    for (const auto& c : checks) preempt_checkpoint_bytes(out, c);
}

} // namespace strata::core
