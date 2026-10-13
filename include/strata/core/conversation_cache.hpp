// CPU-only ownership and matching policy for --serve's parked conversations.
// Token equality, image identity, and steering mode are all required for reuse.
#pragma once

#include "strata/core/conversation_buffer.hpp"

#include <algorithm>
#include <array>
#include <cstddef>
#include <cstdint>
#include <cstdio>   // [SFX-DBG] probe: fprintf in best_with_partial's candidate scan
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
    // --prefill-pipe on a layer split: the plan a prompt segment [from, to) is read in - its chunk (0: none), the loan
    // its buffers were laid out for (which experts the prompt reads in place and which it streams: that sets bits
    // too), the pipe value it was planned with, whether it is one chunk and whether the plan changed the buffers'
    // chunks.  A periodic checkpoint taken inside such a segment keeps it, so a read that goes on from there reads the
    // rest in the same chunks with the same loan.  Run-time only, like `pinned` (a layer split saves no session files).
    struct PromptPlan {
        int64_t chunk = 0, from = -1, to = -1, lend = 0;
        double pipe = 0.0;
        bool single = false, chunked = false;
    } plan;
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

/// Phase 3b: longest common PREFIX of a checkpoint and a prompt, up to the checkpoint's length.
/// Returns:
///   0:  no LCP (token 0 mismatches, or image key mismatch below the LCP, or empty checkpoint)
///   n:  the checkpoint is a strict prefix of the prompt (== conversation_prefix, the "full match")
///   p:  partial match - 0 < p < n (the LCP length; the checkpoint's tail [p, n) diverges)
/// Image check mirrors conversation_prefix but bounded by the LCP (not the full checkpoint):
/// request images at positions < p must match the checkpoint's images at the same positions; the
/// checkpoint's images at positions >= p are irrelevant (the partial mount discards [p, n) of the
/// checkpoint).  This is the helper the flag-gated partial-prefix mount in
/// `best_with_partial` walks to find the longest LCP when no strict match exists.
template<class Token>
int64_t conversation_lcp(const ConversationCheckpoint& c, const std::vector<Token>& prompt,
                         const std::vector<ConversationImageKey>& images) {
    const size_t n = c.ids.size();
    if (n < 1 || n > prompt.size()) return 0;            // empty or larger than the prompt: no LCP
    int64_t p = 0;
    while (p < (int64_t) n && c.ids[(size_t) p] == (Token) prompt[(size_t) p]) ++p;
    if (p == 0) return 0;                                // token 0 mismatches
    // Image check: request images at positions < p must match the checkpoint's images.  The
    // checkpoint's images at positions >= p are irrelevant (we re-prefill [p, n) on the new
    // prompt).  The strict `conversation_prefix` check is the same logic with n -> p.
    size_t j = 0;
    for (const auto& image : images) {
        if (image.start >= p) continue;
        if (j == c.imgs.size() || !(c.imgs[j++] == image)) return 0;
    }
    for (size_t k = j; k < c.imgs.size(); ++k)
        if (c.imgs[k].start < p) return 0;               // checkpoint has an image below p not in the request
    return p;
}

class ConversationCache {
public:
    struct Match {
        size_t index = 0;
        int64_t tokens = 0;
        bool live = false;
        uint64_t serial = 0;   // the entry's identity; its index shifts when entries before it go
        // Phase 3b: flag-on partial-prefix match.  `partial == true` means the longest LCP of any
        // checkpoint with the request is strictly less than the checkpoint's length (a mid-prompt
        // edit past the LCP).  In that case `tokens` is the LCP (the resume length the splice hook
        // narrows to) and `checkpoint_length` is the full checkpoint's length (the restore anchor).
        // Strict match (== the existing behaviour): `partial == false`, `tokens == checkpoint_length`.
        // No match: `partial == false`, `tokens == 0`.  Off-flag (`flag <= 0`) and on-flag-with-
        // strict-match returns are byte-identical to `best()`.
        bool partial = false;
        int64_t checkpoint_length = 0;   ///< the full checkpoint's length (== tokens when !partial)
    };

    ConversationCache(size_t budget, size_t slots, int64_t min_tokens = 0)
        : budget_(budget), slots_(slots), min_tokens_(min_tokens) {}
    bool enabled() const { return budget_ != 0 && slots_ != 0; }
    /// `--conversation-cache-min-tokens`: whether a conversation of this length is worth one of `slots`.  Every
    /// parked conversation costs a whole slot whatever its length, so a client that interleaves short side requests
    /// with one long conversation spends the cache on the side requests and evicts the long one (oldest first) -
    /// exactly the reuse it wanted.  0 parks every conversation, whatever its size.
    bool wants(int64_t tokens) const { return min_tokens_ <= 0 || tokens >= min_tokens_; }
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
    Match best(const std::vector<Token>& prompt, const std::vector<ConversationImageKey>& images, bool cvec) const {
        Match best;
        // Ties prefer the most recently parked branch. The caller prefers its
        // already-active state when that offers the same prefix length.
        for (size_t i = entries_.size(); i-- > 0;) {
            const auto& e = entries_[i];
            if (e.cvec != cvec) continue;
            auto consider = [&](const ConversationCheckpoint& c, bool live) {
                const int64_t n = conversation_prefix(c, prompt, images);
                if (n > best.tokens) best = {i, n, live, serials_[i], false, n};
            };
            consider(e.live, true);
            for (const auto& c : e.checkpoints) consider(c, false);
        }
        return best;
    }

    /// Phase 3b: flag-gated partial-prefix match.  At `flag <= 0` (and on-flag when a strict
    /// match exists) this is byte-identical to `best()`: same loop, same `conversation_prefix`,
    /// same tie-break by most recently parked.  At `flag > 0` with NO strict match, falls back
    /// to the longest LCP `p` such that `0 < p < c.ids.size()` and the image check at positions
    /// < p matches; the returned Match has `partial = true, tokens = p, checkpoint_length = L`.
    /// Ties: longest LCP wins, then longest checkpoint (a deeper anchor gives the splice more
    /// K rows to rotate for the unchanged tail), then most recently parked (same as `best()`).
    template<class Token>
    Match best_with_partial(int64_t flag, const std::vector<Token>& prompt,
                            const std::vector<ConversationImageKey>& images, bool cvec) const {
        if (flag <= 0) return best(prompt, images, cvec);   // flag off: byte-identical
        Match m;
        // strict first (same as best()).
        for (size_t i = entries_.size(); i-- > 0;) {
            const auto& e = entries_[i];
            if (e.cvec != cvec) continue;
            auto consider = [&](const ConversationCheckpoint& c, bool live) {
                const int64_t n = conversation_prefix(c, prompt, images);
                if (n > m.tokens) m = {i, n, live, false, n};
            };
            consider(e.live, true);
            for (const auto& c : e.checkpoints) consider(c, false);
        }
        // FW-1 splice engagement (2026-10-13): do NOT return at the first strict
        // match.  One conversation's checkpoints are nested prefixes of ONE history,
        // so that history diverges from this request at a single point D and every
        // strict anchor s <= D equals the partial LCP p of the SAME conversation's
        // deeper checkpoints.  A partial mount therefore never reads from a
        // shallower point than the strict match would (p >= s), and its splice may
        // recover K/V up to the anchor length L.  Keep the partial candidate when
        // its checkpoint covers strictly more than the strict anchor (L > s);
        // otherwise the strict match stands (no splice cost when it cannot recover
        // more).  Without this, a mid-file edit always strict-mounts the anchor
        // just below the edit and the splice never engages (partial-branch
        // enter=0 in every steady-state trace - measured 2026-10-13).
        const int64_t strict_tokens = m.tokens;
        // partial pass: longest LCP > 0 and < c.ids.size(); tie-break by deeper anchor
        // (longer checkpoint) then by most recently parked (the loop iterates newest-first).
        Match pm;   // the best partial candidate (tokens = LCP, checkpoint_length = L)
        // [SFX-DBG] probe: per-candidate partial scan visibility.  The steady-state chain showed
        // parked-search reporting match{tokens=2598 partial=0} where the source says a partial
        // candidate (LCP 3424, anchor 5702) should dominate strict 2598 - log every candidate's
        // L/idx/LCP so the steady-state trace can attribute the miss to the candidate set, the
        // LCP guard, or the size guard.  Gated `flag > 0` (probe on the sfx path only).
        const bool dbg_partial = flag > 0;
        if (dbg_partial)
            std::fprintf(stderr, "[SFX-DBG] parked-scan: entries=%zu prompt_len=%zu strict_tokens=%lld:",
                         entries_.size(), prompt.size(), (long long) strict_tokens);
        for (size_t i = entries_.size(); i-- > 0;) {
            const auto& e = entries_[i];
            if (e.cvec != cvec) { if (dbg_partial) std::fprintf(stderr, " e%zu:cvec", i); continue; }
            auto consider = [&](const ConversationCheckpoint& c, bool live) {
                const int64_t L = (int64_t) c.ids.size();
                if (L < 1 || L > (int64_t) prompt.size()) {
                    if (dbg_partial) std::fprintf(stderr, " e%zu.%s:L=%lld:skip", i, live ? "live" : "ck", (long long) L);
                    return;   // must be <= prompt.size() (L == n is valid partial: same length, mid-file edit)
                }
                const int64_t p = conversation_lcp(c, prompt, images);
                if (dbg_partial)
                    std::fprintf(stderr, " e%zu.%s:L=%lld:lcp=%lld", i, live ? "live" : "ck", (long long) L, (long long) p);
                if (p <= 0 || p >= L) return;                         // 0 LCP or strict (already tried)
                if (p > pm.tokens || (p == pm.tokens && L > pm.checkpoint_length))
                    pm = {i, p, live, true, L};
            };
            consider(e.live, true);
            for (const auto& c : e.checkpoints) consider(c, false);
        }
        if (dbg_partial) std::fprintf(stderr, " -> pm{tokens=%lld L=%lld} strict=%lld\n",
                                      (long long) pm.tokens, (long long) pm.checkpoint_length,
                                      (long long) strict_tokens);
        if (pm.tokens > 0 && pm.checkpoint_length > strict_tokens) return pm;
        return m;                                            // strict match wins; partial ignored
    }

    SavedConversation take(size_t index) {
        SavedConversation out = std::move(entries_.at(index));
        bytes_ -= out.bytes();
        erase(index);
        return out;
    }

    // A checkpoint inside a parked conversation that a NEW conversation starts with - a sibling subagent sharing the
    // system prompt and tool list, a compacted history - is borrowed: restored from the entry, which stays parked.
    // take() would move the whole conversation into the session and the new request would overwrite everything after
    // that checkpoint, so the other conversation's next turn would read its history again.  A match on the entry's
    // live state or on its deepest checkpoint is the same conversation going on (the deepest one: its client
    // re-rendered the last reply), whose tail is stale: that is still taken.  So is a match that reaches a pinned
    // shared prefix (pin=N): a borrow copies the checkpoints up to the match, and taking it instead keeps one
    // conversation holding the pin, as pin=N expects.
    bool borrows(const Match& m) const {
        if (m.tokens <= 0 || m.live || m.index >= entries_.size() || serials_[m.index] != m.serial) return false;
        size_t deepest = 0;
        for (const auto& c : entries_[m.index].checkpoints) {
            if (c.pinned && (int64_t) c.ids.size() <= m.tokens) return false;
            deepest = std::max(deepest, c.ids.size());
        }
        return (size_t) m.tokens < deepest;
    }
    // The entry with this identity: its index (SIZE_MAX: gone), or the entry itself (nullptr: gone; the pointer is
    // valid until the cache changes).
    size_t index_of(uint64_t serial) const {
        for (size_t i = 0; i < entries_.size(); ++i)
            if (serials_[i] == serial) return i;
        return SIZE_MAX;
    }
    const SavedConversation* find(uint64_t serial) const {
        const size_t i = index_of(serial);
        return i == SIZE_MAX ? nullptr : &entries_[i];
    }
    // While a borrowed entry (the donor) is being restored, parking the outgoing conversation must not evict or drop
    // it: guard() it.  When the outgoing image fits only without it, make_room refuses before evicting anything and
    // guard_blocked() says so: the caller then takes the entry as before, so the conversation that was active a
    // moment ago keeps its place.
    void guard(uint64_t serial) { guarded_ = serial; guard_blocked_ = false; }
    void unguard() { guarded_ = 0; }
    bool guard_blocked() const { return guard_blocked_; }
    size_t borrowed() const { return borrowed_; }
    void count_borrow() { ++borrowed_; }

    // Reserve before allocating a snapshot. held is an incoming image removed
    // with take() but still alive during the exchange; count it against RAM too.
    bool make_room(size_t incoming, size_t held = 0) {
        if (!enabled() || held > budget_ || incoming > budget_ - held) return false;
        auto stays = [&](size_t i) { return entries_[i].pinned() || (guarded_ != 0 && serials_[i] == guarded_); };
        if (guarded_ != 0 && index_of(guarded_) != SIZE_MAX) {
            // with everything else gone, would it fit beside the entries that stay (the guarded donor and any pinned
            // shared prefix)?  If not, evict nothing
            size_t kept = 0, kept_bytes = 0;
            for (size_t i = 0; i < entries_.size(); ++i)
                if (stays(i)) { ++kept; kept_bytes += entries_[i].bytes(); }
            if (kept >= slots_ || kept_bytes > budget_ - held - incoming) { guard_blocked_ = true; return false; }
        }
        if (bytes() > budget_ - held - incoming) reuse_ = {};
        while (!entries_.empty() && (entries_.size() >= slots_ || bytes_ > budget_ - held - incoming)) {
            // the oldest entry that does not hold a pinned shared prefix leaves; with only pinned ones left the new
            // image does not fit (the caller skips parking it - the pinned prefix is what the queries come back to)
            if (!evict_oldest()) return false;
        }
        return true;
    }

    // The parked conversation that has gone unused the longest.  This is make_room()'s loop body, so
    // the parking path can also free RAM one entry at a time on demand (see the physical-RAM admission
    // gate in generate.cpp).  A ConversationBuffer is a list of 16 MiB segments and each segment is its
    // own allocation, far above glibc's mmap threshold, so dropping an entry returns the whole footprint
    // to the kernel at once - the next admission check reads it back from /proc/meminfo.
    // False when none can go (empty, or only entries that hold a pinned shared prefix are left).
    bool evict_oldest() {
        // the oldest entry that holds no pinned shared prefix and is not the guarded (borrowed) donor
        for (size_t i = 0; i < entries_.size(); ++i) {
            if (entries_[i].pinned() || (guarded_ != 0 && serials_[i] == guarded_)) continue;
            bytes_ -= entries_[i].bytes();
            erase(i);
            ++evictions_;
            return true;
        }
        return false;
    }

    // #879: every parked conversation (pinned ones too) and the retained K/V go - after non-finite logits any of them
    // may carry the poison.  Returns how many were parked.
    size_t clear() {
        const size_t n = entries_.size();
        entries_.clear();
        serials_.clear();
        bytes_ = 0;
        reuse_ = {};
        return n;
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
            if (e.cvec == cvec && deepest && !deepest->ids.empty() && held(*deepest) &&
                !(guarded_ != 0 && serials_[i] == guarded_)) {
                bytes_ -= e.bytes();
                erase(i);
                ++dropped;
                continue;
            }
            ++i;
        }
        superseded_ += dropped;
        return dropped;
    }
    size_t superseded() const { return superseded_; }
    /// --conversation-cache-park-gain N: the longest parked entry this conversation's outgoing state supersedes, in
    /// tokens, 0 when it has none.  It is the entry drop_superseded() would remove, so the caller can tell whether a
    /// park now would only rewrite what a turn from now drops: the prefix such an entry holds is this conversation's
    /// prefix, and the conversation only appends, so the copy stays usable for the reuse it was made for.
    size_t parked_prefix_tokens(const std::vector<int32_t>& ids, const std::vector<ConversationImageKey>& images,
                                const std::vector<ConversationCheckpoint>& checkpoints, bool cvec) const {
        size_t longest = 0;
        for (const auto& e : entries_) {
            if (e.cvec != cvec) continue;
            const ConversationCheckpoint* deepest = nullptr;
            for (const auto& c : e.checkpoints)
                if (!deepest || c.ids.size() > deepest->ids.size()) deepest = &c;
            if (!deepest || deepest->ids.empty()) continue;
            bool holds = deepest->ids == ids && deepest->imgs == images;
            for (const auto& k : checkpoints)
                if (!holds && k.ids == deepest->ids && k.imgs == deepest->imgs) holds = true;
            if (holds) longest = std::max(longest, deepest->ids.size());
        }
        return longest;
    }

    bool put(SavedConversation&& image, size_t held = 0) {
        const size_t n = image.bytes();
        if (!enabled() || held > budget_ || n > budget_ - held) return false;   // make_room's refusal, first
        drop_superseded(image.live.ids, image.live.imgs, image.checkpoints, image.cvec);
        if (!make_room(n, held)) return false;
        entries_.push_back(std::move(image));
        serials_.push_back(next_serial_++);
        bytes_ += n;
        return true;
    }

private:
    void erase(size_t index) {
        entries_.erase(entries_.begin() + (std::ptrdiff_t) index);
        serials_.erase(serials_.begin() + (std::ptrdiff_t) index);
    }
    size_t budget_ = 0, slots_ = 0, bytes_ = 0, evictions_ = 0, superseded_ = 0, borrowed_ = 0;
    int64_t min_tokens_ = 0;   // --conversation-cache-min-tokens: 0 parks every conversation, whatever its size
    std::deque<SavedConversation> entries_; // least recently active first
    std::deque<uint64_t> serials_;          // entries_[i]'s identity, never reused
    uint64_t next_serial_ = 1, guarded_ = 0;
    bool guard_blocked_ = false;
    ConversationKvReuse reuse_;
};

} // namespace strata::core
