// CPU-only ownership and matching policy for --serve's parked conversations.
// Token equality, image identity, and steering mode are all required for reuse.
#pragma once

#include "strata/core/conversation_buffer.hpp"

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
    uint64_t used = 0; // upstream root-pinned/LRU checkpoint retention
    // A shared-prefix pin (the request key pin=N): this checkpoint is the read-only prefix many suffix queries branch
    // from, so retention never evicts it (conv_cache.hpp) and a parked conversation holding it stays parked.  A run-time
    // mark only: it is not in the session file, a request that pins the same prefix again sets it.
    bool pinned = false;
    // Ordinary layer-split checkpoints retain each device's running state.
    // Whole-session parking is currently single-GPU and rejects these parts.
    std::vector<ConversationCheckpoint> stage_parts;

    size_t bytes() const {
        size_t n = ids.capacity() * sizeof(int32_t) + imgs.capacity() * sizeof(ConversationImageKey) +
               gdn.capacity() + ple.capacity() + tails.capacity() + dead.capacity() + block_pos.capacity() +
               stage_parts.capacity() * sizeof(ConversationCheckpoint);
        for (const auto& part : stage_parts) n += part.bytes();
        return n;
    }
};

// Identity-layout K/V pages and completed indexer rows. For streamed layers the
// source is the authoritative host pool, NOT the replaceable VRAM slots.
struct ConversationKv {
    int format = 0;
    int64_t cells = 0, heads = 0, head_dim = 0, page_size = 0, pooled_rows = 0, idx_dim = 0;
    ConversationBuffer k, v, k_scale, v_scale, pooled;
    size_t bytes() const {
        return k.bytes() + v.bytes() + k_scale.bytes() + v_scale.bytes() + pooled.bytes();
    }
};

struct ConversationKvReuse {
    std::vector<ConversationKv> kv;
    // Original image extent for validation, and the earliest subsequent rewrite.
    int64_t captured_tokens = 0, unchanged_tokens = 0;
    // With a layer split: the later stages' own retained K/V, one per stage, same extents (empty: none)
    std::vector<ConversationKvReuse> stages;
    size_t bytes() const {
        size_t n = kv.capacity() * sizeof(ConversationKv) + stages.capacity() * sizeof(ConversationKvReuse);
        for (const auto& layer : kv) n += layer.bytes();
        for (const auto& s : stages) n += s.bytes();
        return n;
    }
};

/// Layer-split parking: each running checkpoint holds the first stage's state and, in `stage_parts`, one part per
/// later stage; a parked conversation keeps them in one image per stage.  `split` MOVES the checkpoints apart (no
/// running state is copied): the complete ones (a part for each of `stages` stages) into `stage0` and `parts[k]`,
/// every part carrying the checkpoint's ids / images / LRU stamp, the others aside; `merge` puts them back in their
/// original order.  merge(split(x)) == x.
struct ConversationCheckpointSplit {
    std::vector<ConversationCheckpoint> stage0;
    std::vector<std::vector<ConversationCheckpoint>> parts;   ///< [stage][checkpoint]
    std::vector<ConversationCheckpoint> rest;                 ///< not split (no complete set of stage parts)
    std::vector<bool> complete;                               ///< per original position: in stage0/parts or in rest
};

inline ConversationCheckpointSplit conversation_checkpoints_split(std::vector<ConversationCheckpoint>&& checks,
                                                                  size_t stages) {
    ConversationCheckpointSplit out;
    out.parts.resize(stages);
    out.complete.reserve(checks.size());
    size_t whole = 0;
    for (const auto& c : checks) {
        const bool complete = c.stage_parts.size() == stages;
        out.complete.push_back(complete);
        whole += complete ? 1 : 0;
    }
    out.stage0.reserve(whole);
    out.rest.reserve(checks.size() - whole);
    for (auto& parts : out.parts) parts.reserve(whole);
    // Allocate and copy identity metadata before moving any running state.
    // If allocation fails, every input checkpoint remains usable by the caller.
    for (const auto& c : checks) {
        if (c.stage_parts.size() != stages) continue;
        for (size_t k = 0; k < stages; ++k) {
            ConversationCheckpoint part;
            part.ids = c.ids; part.imgs = c.imgs; part.used = c.used; part.pinned = c.pinned;
            out.parts[k].push_back(std::move(part));
        }
    }
    size_t w = 0;
    for (auto& c : checks) {
        if (c.stage_parts.size() != stages) { out.rest.push_back(std::move(c)); continue; }
        for (size_t k = 0; k < stages; ++k) {
            auto& dst = out.parts[k][w];
            auto& src = c.stage_parts[k];
            dst.gdn = std::move(src.gdn); dst.ple = std::move(src.ple);
            dst.tails = std::move(src.tails); dst.dead = std::move(src.dead);
            dst.block_pos = std::move(src.block_pos);
        }
        c.stage_parts.clear();
        out.stage0.push_back(std::move(c));
        ++w;
    }
    checks.clear();
    return out;
}

/// false (and `out` empty) when the parts do not line up: a stage with another number of checkpoints, or a part
/// whose ids differ from its checkpoint's.
inline bool conversation_checkpoints_merge(ConversationCheckpointSplit&& split, std::vector<ConversationCheckpoint>& out) {
    out.clear();
    for (const auto& p : split.parts)
        if (p.size() != split.stage0.size()) return false;
    size_t whole = 0, other = 0;
    for (bool c : split.complete) (c ? whole : other) += 1;
    if (!split.complete.empty() && (whole != split.stage0.size() || other != split.rest.size())) return false;
    if (split.complete.empty()) split.complete.assign(split.stage0.size(), true);   // a restored image: all whole
    for (size_t j = 0; j < split.stage0.size(); ++j)
        for (const auto& p : split.parts)
            if (p[j].ids != split.stage0[j].ids) return false;
    size_t w = 0, r = 0;
    for (bool c : split.complete) {
        if (!c) { out.push_back(std::move(split.rest[r++])); continue; }
        ConversationCheckpoint cp = std::move(split.stage0[w]);
        for (auto& p : split.parts) cp.stage_parts.push_back(std::move(p[w]));
        ++w;
        out.push_back(std::move(cp));
    }
    return true;
}

struct SavedConversation {
    // Runtime compatibility only; NOT a model/weights identity or disk schema.
    std::array<int64_t, 18> geometry{};
    // The session's layer carve the image was captured from ([0, n_layers) on one GPU); restore requires the same.
    int64_t layer_lo = 0, layer_hi = 0;
    ConversationCheckpoint live;
    std::vector<ConversationCheckpoint> checkpoints;
    std::vector<ConversationKv> kv; // main layers followed by the draft layer
    bool cvec = true;
    // with a layer split, the later stages' own images, one per stage, in stage order
    std::vector<SavedConversation> stage_images;

    /// Holds a pinned shared prefix (see ConversationCheckpoint::pinned): the parked-conversation budget keeps it.
    bool pinned() const {
        for (const auto& c : checkpoints) if (c.pinned) return true;
        return false;
    }

    size_t bytes() const {
        size_t n = live.bytes() + checkpoints.capacity() * sizeof(ConversationCheckpoint) +
                   kv.capacity() * sizeof(ConversationKv);
        for (const auto& s : stage_images) n += s.bytes();
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

// The same prefix test against only the first n ids of the checkpoint: a parked conversation's checkpoints are
// prefixes of its live ids, so a shorter resume point is checked without copying the ids.
template<class Token>
int64_t conversation_prefix_at(const ConversationCheckpoint& c, const std::vector<Token>& prompt,
                               const std::vector<ConversationImageKey>& images, size_t n) {
    // The last prompt token always starts the next verify window.
    if (n == 0 || n >= prompt.size() || n > c.ids.size() ||
        !std::equal(c.ids.begin(), c.ids.begin() + n, prompt.begin())) return 0;
    size_t a = 0, b = 0;
    while ((a < c.imgs.size() && c.imgs[a].start < (int64_t) n) ||
           (b < images.size() && images[b].start < (int64_t) n)) {
        if (a == c.imgs.size() || c.imgs[a].start >= (int64_t) n ||
            b == images.size() || images[b].start >= (int64_t) n || !(c.imgs[a] == images[b])) return 0;
        ++a;
        ++b;
    }
    return (int64_t) n;
}

// The longest common prefix of a cached conversation and the prompt, in tokens. Unlike conversation_prefix this
// does not require the whole cached ids to match: it is the resume point a similarity hit would give.
template<class Token>
int64_t conversation_lcp(const ConversationCheckpoint& cached, const std::vector<Token>& prompt,
                         const std::vector<ConversationImageKey>& images) {
    size_t n = std::min(cached.ids.size(), prompt.size());
    size_t i = 0;
    while (i < n && cached.ids[i] == prompt[i]) ++i;
    // Token ids alone do not identify image embeddings: stop before the first image boundary whose identity
    // differs, so no state after it is reused.
    size_t a = 0, b = 0;
    while ((a < cached.imgs.size() && cached.imgs[a].start < (int64_t) i) ||
           (b < images.size() && images[b].start < (int64_t) i)) {
        if (a == cached.imgs.size() || cached.imgs[a].start >= (int64_t) i ||
            b == images.size() || images[b].start >= (int64_t) i || !(cached.imgs[a] == images[b])) {
            const int64_t ca = a < cached.imgs.size() ? cached.imgs[a].start : INT64_MAX;
            const int64_t cb = b < images.size() ? images[b].start : INT64_MAX;
            const int64_t boundary = std::min(ca, cb);
            if (boundary < 0) return 0;
            i = std::min(i, (size_t) boundary);
            break;
        }
        ++a;
        ++b;
    }
    return (int64_t) i;
}

struct ConversationMatch {
    size_t index = 0;
    int64_t tokens = 0;
    bool live = false;
};

// The best resume a parked image offers for this prompt. similarity is the least LCP/new-prompt fraction accepted
// (0 keeps every exact-prefix hit); n_min the least common-prefix tokens. A hit longer than the LCP is never offered.
template<class Token>
ConversationMatch conversation_saved_match(const SavedConversation& saved,
                                           const std::vector<Token>& prompt,
                                           const std::vector<ConversationImageKey>& images,
                                           bool cvec, double similarity, int64_t n_min) {
    ConversationMatch best;
    if (saved.cvec != cvec || prompt.empty() || similarity < 0.0 || similarity >= 1.0) return best;
    const int64_t lcp = conversation_lcp(saved.live, prompt, images);
    if (lcp < n_min || (double) lcp / (double) prompt.size() <= similarity) return best;
    auto consider = [&](const ConversationCheckpoint& checkpoint, bool live) {
        const int64_t n = conversation_prefix(checkpoint, prompt, images);
        if (n > 0 && n <= lcp && n > best.tokens) best = {0, n, live};
    };
    consider(saved.live, true);
    for (const auto& checkpoint : saved.checkpoints) consider(checkpoint, false);
    return best;
}

// The same match from a spill directory's metadata only (no K/V read): the live ids and images plus each
// checkpoint's length, exactly what a parked image's checkpoints are prefixes of.
template<class Token>
ConversationMatch conversation_metadata_match(const ConversationCheckpoint& live,
                                              const std::vector<size_t>& checkpoint_lengths,
                                              const std::vector<Token>& prompt,
                                              const std::vector<ConversationImageKey>& images,
                                              bool cvec, bool cached_cvec,
                                              double similarity, int64_t n_min) {
    ConversationMatch best;
    if (cached_cvec != cvec || prompt.empty() || similarity < 0.0 || similarity >= 1.0) return best;
    const int64_t lcp = conversation_lcp(live, prompt, images);
    if (lcp < n_min || (double) lcp / (double) prompt.size() <= similarity) return best;
    const int64_t live_match = conversation_prefix(live, prompt, images);
    if (live_match > 0 && live_match <= lcp) best = {0, live_match, true};
    for (size_t length : checkpoint_lengths) {
        const int64_t n = conversation_prefix_at(live, prompt, images, length);
        if (n > 0 && n <= lcp && n > best.tokens) best = {0, n, false};
    }
    return best;
}

class ConversationCache {
public:
    using Match = ConversationMatch;

    ConversationCache(size_t budget, size_t slots) : budget_(budget), slots_(slots) {}
    bool enabled() const { return budget_ != 0 && slots_ != 0; }
    size_t bytes() const { return bytes_ + reuse_.bytes(); }
    size_t size() const { return entries_.size(); }
    size_t evictions() const { return evictions_; }
    // the longest parked conversation, in tokens (--kv-grow keeps the K/V that long while it could be restored)
    int64_t longest_tokens() const {
        int64_t n = 0;
        for (const auto& e : entries_) n = std::max<int64_t>(n, (int64_t) e.live.ids.size());
        return n;
    }

    // Retain only the restored K/V buffers, not duplicate running checkpoints.
    // This optimization never evicts a parked conversation to make itself fit.
    // `stage_kv`: with a layer split, the later stages' restored K/V (one per stage), retained with the first's.
    void retain(std::vector<ConversationKv>&& kv, int64_t tokens,
                std::vector<std::vector<ConversationKv>>&& stage_kv = {}) {
        reuse_ = {};
        ConversationKvReuse candidate{std::move(kv), tokens, tokens, {}};
        for (auto& k : stage_kv) candidate.stages.push_back(ConversationKvReuse{std::move(k), tokens, tokens, {}});
        if (enabled() && candidate.bytes() <= budget_ - bytes_) reuse_ = std::move(candidate);
    }
    void limit_reuse(int64_t first_dirty) {
        reuse_.unchanged_tokens = std::min(reuse_.unchanged_tokens, first_dirty);
        for (auto& s : reuse_.stages) s.unchanged_tokens = std::min(s.unchanged_tokens, first_dirty);
        if (reuse_.unchanged_tokens <= 0) reuse_ = {};
    }
    ConversationKvReuse take_reuse() { return std::exchange(reuse_, {}); }
    size_t retained_bytes() const { return reuse_.bytes(); }
    bool can_fit(size_t incoming, size_t held = 0) const {
        return enabled() && held <= budget_ && incoming <= budget_ - held &&
               entries_.size() < slots_ && bytes() <= budget_ - held - incoming;
    }

    template<class Token>
    Match best(const std::vector<Token>& prompt, const std::vector<ConversationImageKey>& images, bool cvec,
               double similarity = 0.0, int64_t n_min = 0) const {
        Match best;
        // Ties prefer the most recently parked branch. The caller prefers its
        // already-active state when that offers the same prefix length.
        for (size_t i = entries_.size(); i-- > 0;) {
            const auto& e = entries_[i];
            Match candidate = conversation_saved_match(e, prompt, images, cvec, similarity, n_min);
            if (candidate.tokens > best.tokens) { candidate.index = i; best = candidate; }
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
        return make_room(incoming, held, [](const SavedConversation&) {});
    }

    // The spill callback sees every conversation this call evicts from RAM, oldest first, before it is dropped, so
    // a durable disk tier can keep it. The default does nothing (the RAM cache alone).
    template<class Spill>
    bool make_room(size_t incoming, size_t held, Spill&& spill) {
        if (!enabled() || held > budget_ || incoming > budget_ - held) return false;
        if (bytes() > budget_ - held - incoming) reuse_ = {};
        while (!entries_.empty() && (entries_.size() >= slots_ || bytes_ > budget_ - held - incoming)) {
            // the oldest entry that does not hold a pinned shared prefix leaves; with only pinned ones left the new
            // image does not fit (the caller skips parking it - the pinned prefix is what the queries come back to)
            if (!evict_oldest(spill)) return false;
        }
        return true;
    }

    // The parked conversation that has gone unused the longest.  This is make_room()'s loop body, so
    // the parking path can also free RAM one entry at a time on demand (see the physical-RAM admission
    // gate in generate.cpp).  A ConversationBuffer is a list of 16 MiB segments and each segment is its
    // own allocation, far above glibc's mmap threshold, so dropping an entry returns the whole footprint
    // to the kernel at once - the next admission check reads it back from /proc/meminfo.
    // False when none can go (empty, or only entries that hold a pinned shared prefix are left).
    bool evict_oldest() { return evict_oldest([](const SavedConversation&) {}); }

    // The spill callback sees the conversation before it is dropped, so a durable disk tier keeps what the
    // physical-RAM gate evicts and not only what make_room() pushes out.
    template<class Spill>
    bool evict_oldest(Spill&& spill) {
        auto victim = std::find_if(entries_.begin(), entries_.end(), [](const SavedConversation& e) { return !e.pinned(); });
        if (victim == entries_.end()) return false;
        spill(*victim);
        bytes_ -= victim->bytes();
        entries_.erase(victim);
        ++evictions_;
        return true;
    }

    // The slot count, so a caller that evicts in a loop has a bound it did not invent.
    size_t slots() const { return slots_; }

    // #342: drop the parked entries an outgoing conversation (its live tokens and checkpoint chain) supersedes:
    // the same conversation a turn back, whose DEEPEST checkpoint the outgoing chain still holds, so all it adds
    // is the tail the client rewrote (the reply as it was generated, before the next request re-rendered it) and
    // checkpoints older than that one.  A subagent's successive turns parked one such copy each, and make_room's
    // oldest-first eviction then pushed the parent conversation out after `slots` turns.  An entry without
    // checkpoints, or whose deepest checkpoint the outgoing chain does not hold (another conversation that only
    // shares the system prompt's root with it), is kept.  Returns how many were dropped.
    size_t drop_superseded(const std::vector<int32_t>& ids, const std::vector<ConversationImageKey>& images,
                           const std::vector<ConversationCheckpoint>& checkpoints, bool cvec) {
        auto held = [&](const ConversationCheckpoint& c) {
            if (c.ids == ids && c.imgs == images) return true;
            for (const auto& k : checkpoints)
                if (k.ids == c.ids && k.imgs == c.imgs) return true;
            return false;
        };
        size_t dropped = 0;
        for (size_t i = 0; i < entries_.size();) {
            const auto& e = entries_[i];
            const ConversationCheckpoint* deepest = nullptr;
            for (const auto& c : e.checkpoints)
                if (!deepest || c.ids.size() > deepest->ids.size()) deepest = &c;
            if (e.cvec == cvec && deepest && !deepest->ids.empty() && held(*deepest)) {
                bytes_ -= e.bytes();
                entries_.erase(entries_.begin() + (std::ptrdiff_t) i);
                ++dropped;
                continue;
            }
            ++i;
        }
        superseded_ += dropped;
        return dropped;
    }
    size_t superseded() const { return superseded_; }

    // Spill every parked conversation (oldest first) and empty the cache: the shutdown path, so a restart finds
    // them on disk. Returns how many were handed to the callback.
    template<class Spill>
    size_t spill_all(Spill&& spill) {
        size_t count = 0;
        while (!entries_.empty()) {
            spill(entries_.front());
            bytes_ -= entries_.front().bytes();
            entries_.pop_front();
            ++evictions_;
            ++count;
        }
        reuse_ = {};
        return count;
    }

    bool put(SavedConversation&& image, size_t held = 0) {
        const size_t n = image.bytes();
        if (!enabled() || held > budget_ || n > budget_ - held) return false;   // make_room's refusal, first
        drop_superseded(image.live.ids, image.live.imgs, image.checkpoints, image.cvec);
        if (!make_room(n, held)) return false;
        entries_.push_back(std::move(image));
        bytes_ += n;
        return true;
    }

private:
    size_t budget_ = 0, slots_ = 0, bytes_ = 0, evictions_ = 0, superseded_ = 0;
    std::deque<SavedConversation> entries_; // least recently active first
    ConversationKvReuse reuse_;
};

} // namespace strata::core
