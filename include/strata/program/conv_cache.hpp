// include/strata/program/conv_cache.hpp - the conversation cache's retention policy: which checkpoint
// leaves when the cache is over its slot budget.
//
// The cache holds conversation checkpoints: snapshots of the running state (the GDN recurrence, the PLE
// history, the QSA indexer tails - `ConvCheckpoint` in generate.cpp) that a request resumes its prompt
// from instead of reading those tokens again.  The request path keeps only the checkpoints whose tokens
// are a prefix of the current prompt - any other checkpoint's positional cells have been overwritten,
// because the KV cache is one arena that holds one branch of history at a time - so at any moment the
// retained checkpoints are a CHAIN: sorted by length, each a prefix of the next.  That is a radix cache's
// tree collapsed onto the one branch of history the session can hold.
//
// The chain's root is therefore the deepest point every request so far has shared - in practice the end
// of the system prompt, which every new chat of the same client mounts through - and it is exactly the
// node a radix cache keeps alive while its leaves rotate.  First-in-first-out kept it only by accident of
// being first: after `prompt_cache` newer checkpoints it was gone, and the next chat read the whole
// prefix again at prefill speed.  So:
//
//   * the root is PINNED: dropping it frees one slot's ~118 MB and costs every future conversation that
//     shares the prefix a full re-read of it (a 30K system prompt is ~30 s at ~1,000 tok/s);
//   * every other slot rotates by least recent use.  A checkpoint's stamp advances when it is created,
//     when a new checkpoint lands on its length, and when a request mounts through it.
//
// With a budget below two slots there is no room to keep root and leaf apart, so the pin switches off and
// the oldest checkpoint leaves - the previous behaviour, one slot = the newest point only.
#pragma once

#include <cstddef>
#include <cstdint>
#include <deque>
#include <utility>

namespace strata::program::conv_cache {

/// The index in `stamps` of the chain item to drop once the chain holds more than `cap` items.  `stamps`
/// are the items' last-use stamps; the caller owns the chain and erases the returned index.  Pure and
/// deterministic so conv_cache_test.cpp can walk the scenarios by hand.
inline size_t eviction_victim(const uint64_t* stamps, size_t n, int64_t cap) {
    if (cap < 2 || n < 2) return 0;   // no room for root and leaf: the pin is off, the oldest leaves
    size_t v = 1;                     // the root (0) is pinned; least recent use among the rest
    for (size_t i = 2; i < n; ++i)
        if (stamps[i] < stamps[v]) v = i;
    return v;
}

/// --prompt-cache-tail: the extra checkpoint near the prompt's end (`tail[i]`) is the first to go - it serves
/// only a branch of the last request - unless it is the newest item (the one just saved: `n > 1` and the newest
/// stamp).  With no such item this is the plain policy.  Never the root.
inline size_t eviction_victim(const uint64_t* stamps, size_t n, int64_t cap, const bool* tail) {
    if (tail != nullptr && n >= 2 && cap >= 2) {
        size_t newest = 0;
        for (size_t i = 1; i < n; ++i)
            if (stamps[i] > stamps[newest]) newest = i;
        for (size_t i = 1; i < n; ++i)
            if (tail[i] && i != newest) return i;
    }
    return eviction_victim(stamps, n, cap);
}

/// The pin=N shared prefixes (`pinned[i]`): they are never the victim, whatever their stamp - the queries that branch
/// from one keep coming back to it, and a least-recently-used rule would drop it while they still do.  The root and
/// the tail rule are as above.  When every other item is pinned too, the policy falls back to the oldest pinned one:
/// the cache still keeps to its `cap` (the caller limits how many pins it takes, so this is a backstop).
inline size_t eviction_victim(const uint64_t* stamps, size_t n, int64_t cap, const bool* tail, const bool* pinned) {
    if (pinned == nullptr) return eviction_victim(stamps, n, cap, tail);
    if (cap < 2 || n < 2) return 0;
    size_t newest = 0;
    for (size_t i = 1; i < n; ++i)
        if (stamps[i] > stamps[newest]) newest = i;
    if (tail != nullptr)
        for (size_t i = 1; i < n; ++i)
            if (tail[i] && !pinned[i] && i != newest) return i;
    size_t v = n;                    // the least recently used among the unpinned non-root items
    for (size_t i = 1; i < n; ++i)
        if (!pinned[i] && (v == n || stamps[i] < stamps[v])) v = i;
    if (v != n) return v;
    v = 1;                           // all pinned: the oldest of them (never the root, index 0)
    for (size_t i = 2; i < n; ++i)
        if (stamps[i] < stamps[v]) v = i;
    return v;
}

/// --prompt-cache-roots N (opt-in): the system-prompt roots (above) the engine made last, kept apart from any one
/// conversation's chain.  With --batch the chain a root was made in does not stay whole: a slot keeps only its turn
/// checkpoint, and a request that continues from a slot takes over that checkpoint alone, so the next NEW chat of
/// the same client found no root and read its whole system prompt again.  A root's state depends on its tokens
/// alone, so any session whose cells hold those tokens (the main one, an idle slot) can mount through a copy kept
/// here.  At most `cap` are kept (one per distinct system prompt, the least recently made or used leaves first); a
/// root equal to one kept (same tokens, images and control-vector mode) refreshes it instead of adding another copy.
template<class Ck>
class SharedRoots {
public:
    explicit SharedRoots(size_t cap) : cap_(cap) {}
    bool enabled() const { return cap_ > 0; }
    size_t size() const { return items_.size(); }
    void clear() { items_.clear(); }

    /// keep a copy of `c` (made with the control vector `cvec`); true when it was new
    bool keep(const Ck& c, bool cvec) {
        if (cap_ == 0) return false;
        for (size_t i = 0; i < items_.size(); ++i)
            if (items_[i].second == cvec && items_[i].first.ids == c.ids && items_[i].first.imgs == c.imgs) {
                touch(i);
                return false;
            }
        items_.emplace_back(c, cvec);
        while (items_.size() > cap_) items_.pop_front();
        return true;
    }

    /// the longest kept root with the control vector `cvec` that `fits` accepts (nullptr: none); a hit counts as a
    /// use.  The pointer stays valid until the next keep / find / clear.
    template<class Fits>
    const Ck* find(bool cvec, Fits&& fits) {
        size_t best = items_.size();
        for (size_t i = items_.size(); i-- > 0;)
            if (items_[i].second == cvec && (best == items_.size() || items_[i].first.ids.size() > items_[best].first.ids.size()) &&
                fits(items_[i].first))
                best = i;
        if (best == items_.size()) return nullptr;
        return &touch(best);
    }

private:
    Ck& touch(size_t i) {   // the most recently used goes last
        if (i + 1 != items_.size()) {
            std::pair<Ck, bool> it = std::move(items_[i]);
            items_.erase(items_.begin() + (std::ptrdiff_t) i);
            items_.push_back(std::move(it));
        }
        return items_.back().first;
    }
    size_t cap_ = 0;
    std::deque<std::pair<Ck, bool>> items_;   // least recently made or used first
};

}  // namespace strata::program::conv_cache
