// The SHARED CONVERSATION-CACHE CORE's ownership types (issue #57, `jrich/feat/conversation-cache-shared-core`).
// Imported verbatim by the NVMe convergence branch (docs/nvme-kv-cache-convergence.md, step 2) so our serve loop
// and our disk adapter speak ONE checkpoint type.  The types are live; the RAM POLICY below them
// (`ConversationCache`: byte budget, slots, LRU eviction, `conversation_prefix` matching) is imported DORMANT -
// nothing in this tree calls it yet, and wiring it into the serve loop is a later decision.
//
// CPU-only ownership and matching policy for --serve's parked conversations.
// Token equality, image identity, and steering mode are all required for reuse.
#pragma once

#include <algorithm>
#include <array>
#include <cstddef>
#include <cstdint>
#include <deque>
#include <utility>
#include <vector>

namespace strata::core {

struct ConversationImageKey {
    int64_t start = 0;
    uint64_t hash = 0;
    bool operator==(const ConversationImageKey&) const = default;
};

struct ConversationCheckpoint {
    std::vector<int32_t> ids;
    std::vector<ConversationImageKey> imgs;
    std::vector<uint8_t> gdn, ple, tails, dead, block_pos;
    /// Last-use stamp for the retention policy (`conv_cache::eviction_victim`): the serve loop bumps it whenever
    /// a checkpoint is created or mounted through.  RUNTIME ONLY - nothing persists it, and the core's
    /// save/restore ignore it (0.1.21's layer split added it to the local struct this type replaced).
    uint64_t used = 0;
    /// A layer split's later stages: one checkpoint per stage, each saved by the serve loop with its own
    /// `conversation_checkpoint_save` against that stage's session (0.1.21).  The core's save/restore do not
    /// walk this - composition is the serve loop's - but the byte count and the disk adapter must know it:
    /// the NVMe envelope carries the PRIMARY stage only and refuses a split session's snapshot outright.
    std::vector<ConversationCheckpoint> stage_parts;

    size_t bytes() const {
        // The RETAINED PAYLOAD: what the retention policy keeps and what the core's persisted-payload
        // estimator (conversation_snapshot_bytes) must keep agreeing with - their fixture asserts
        // estimate == SavedConversation::bytes() (conversation_validation_test.cpp:114).  `used` is a runtime
        // stamp, not payload, and is excluded for exactly that reason; stage_parts are payload the cache
        // retains on a split engine and are counted recursively (zero when the vector is empty).
        size_t n = ids.capacity() * sizeof(int32_t) + imgs.capacity() * sizeof(ConversationImageKey) +
               gdn.capacity() + ple.capacity() + tails.capacity() + dead.capacity() + block_pos.capacity() +
               stage_parts.capacity() * sizeof(ConversationCheckpoint);
        for (const ConversationCheckpoint& s : stage_parts) n += s.bytes();
        return n;
    }
};

// Identity-layout K/V pages and completed indexer rows. For streamed layers the
// source is the authoritative host pool, NOT the replaceable VRAM slots.
struct ConversationKv {
    int format = 0;
    int64_t cells = 0, heads = 0, head_dim = 0, page_size = 0, pooled_rows = 0, idx_dim = 0;
    std::vector<uint8_t> k, v, k_scale, v_scale, pooled;
    size_t bytes() const {
        return k.capacity() + v.capacity() + k_scale.capacity() + v_scale.capacity() + pooled.capacity();
    }
};

struct SavedConversation {
    // Runtime compatibility only; NOT a model/weights identity or disk schema.
    std::array<int64_t, 18> geometry{};
    ConversationCheckpoint live;
    std::vector<ConversationCheckpoint> checkpoints;
    std::vector<ConversationKv> kv; // main layers followed by the draft layer
    bool cvec = true;

    size_t bytes() const {
        size_t n = live.bytes() + checkpoints.capacity() * sizeof(ConversationCheckpoint) +
                   kv.capacity() * sizeof(ConversationKv);
        for (const auto& c : checkpoints) n += c.bytes();
        for (const auto& k : kv) n += k.bytes();
        return n;
    }
};

template<class Token>
int64_t conversation_prefix(const ConversationCheckpoint& c, const std::vector<Token>& prompt,
                            const std::vector<ConversationImageKey>& images) {
    const size_t n = c.ids.size();
    // The last prompt token always starts the next verify window.
    if (n == 0 || n >= prompt.size() || !std::equal(c.ids.begin(), c.ids.end(), prompt.begin())) return 0;
    size_t j = 0;
    for (const auto& image : images) {
        if (image.start >= (int64_t) n) continue;
        if (j == c.imgs.size() || !(c.imgs[j++] == image)) return 0;
    }
    if (j != c.imgs.size()) return 0;
    return (int64_t) n;
}

class ConversationCache {
public:
    struct Match {
        size_t index = 0;
        int64_t tokens = 0;
        bool live = false;
    };

    ConversationCache(size_t budget, size_t slots) : budget_(budget), slots_(slots) {}
    bool enabled() const { return budget_ != 0 && slots_ != 0; }
    size_t bytes() const { return bytes_; }
    size_t size() const { return entries_.size(); }
    size_t evictions() const { return evictions_; }

    template<class Token>
    Match best(const std::vector<Token>& prompt, const std::vector<ConversationImageKey>& images, bool cvec) const {
        Match best;
        // Ties prefer the most recently parked branch. The caller prefers its
        // already-active state when that offers the same prefix length.
        for (size_t i = entries_.size(); i-- > 0;) {
            const auto& e = entries_[i];
            if (e.cvec != cvec) continue;
            auto consider = [&](const ConversationCheckpoint& c, bool live) {
                const int64_t n = conversation_prefix(c, prompt, images);
                if (n > best.tokens) best = {i, n, live};
            };
            consider(e.live, true);
            for (const auto& c : e.checkpoints) consider(c, false);
        }
        return best;
    }

    SavedConversation take(size_t index) {
        SavedConversation out = std::move(entries_.at(index));
        bytes_ -= out.bytes();
        entries_.erase(entries_.begin() + (std::ptrdiff_t) index);
        return out;
    }

    // Reserve before allocating a snapshot. held is an incoming image removed
    // with take() but still alive during the exchange; count it against RAM too.
    bool make_room(size_t incoming, size_t held = 0) {
        if (!enabled() || held > budget_ || incoming > budget_ - held) return false;
        while (!entries_.empty() && (entries_.size() >= slots_ || bytes_ > budget_ - held - incoming)) {
            bytes_ -= entries_.front().bytes();
            entries_.pop_front();
            ++evictions_;
        }
        return true;
    }

    bool put(SavedConversation&& image, size_t held = 0) {
        const size_t n = image.bytes();
        if (!make_room(n, held)) return false;
        entries_.push_back(std::move(image));
        bytes_ += n;
        return true;
    }

private:
    size_t budget_ = 0, slots_ = 0, bytes_ = 0, evictions_ = 0;
    std::deque<SavedConversation> entries_; // least recently active first
};

} // namespace strata::core
