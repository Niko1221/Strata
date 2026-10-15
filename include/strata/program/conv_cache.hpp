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
//
// --prompt-cache-policy log (eviction_victim_log, below) is the opt-in alternative for long conversations: the
// root pinned the same way, the rest chosen against a geometric grid of lengths instead of by stamp.
#pragma once

#include <cstddef>
#include <cstdint>

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

/// --prompt-cache-policy log (opt-in): the chain is chosen against a geometric grid behind the newest point - targets
/// at `every` * (2^k - 1) tokens behind it (0, 16K, 48K, 112K, 240K, ... with the default 16K), so the tail stays as
/// dense as the periodic checkpoints and the anchors behind it are farther apart the older they are.  The
/// least-recently-used rule keeps only the newest `cap` points: a 256K conversation with the defaults keeps the root
/// (~30K) and 192K-256K, and a request that branches between them reads up to ~162K tokens again.  Each eviction
/// re-picks against the grid from the chain it has, so the kept chain follows the grid where the history allows it
/// rather than landing on it exactly (conv_cache_test.cpp walks the 256K case: 30, 80, 128, 224, 240, 256K kept, the
/// largest gap 96K against the lru rule's 162K).
///
/// `lens[i]` is item i's length in tokens; the newest (longest) item is the grid's origin.  The root (index 0) is
/// not on the grid: it is pinned as above, and the grid stops at it - the targets that would fall behind it are
/// spent halving the largest gap left between the grid and the root instead.  Candidates are the items that are
/// neither the root, nor pinned, nor the newest-stamped one (just saved or just mounted); the tail rule is as above.
/// For each candidate, the chain without it is matched by rank (nearest first) to the targets and scored by the summed
/// relative distance; the candidate whose removal scores lowest leaves, ties to the older stamp, then the lower index.
/// With `every` <= 0 there is no grid, and a chain of more than 64 items (`--prompt-cache` 64 or more) is left to
/// the plain policy too.  Pure, like the others.
inline size_t eviction_victim_log(const uint64_t* stamps, const int64_t* lens, size_t n, int64_t cap, int64_t every,
                                  const bool* tail, const bool* pinned) {
    if (every <= 0 || lens == nullptr || cap < 2 || n < 3 || n > 64) return eviction_victim(stamps, n, cap, tail, pinned);
    size_t newest = 0;
    int64_t origin = lens[0];
    for (size_t i = 1; i < n; ++i) {
        if (stamps[i] > stamps[newest]) newest = i;
        if (lens[i] > origin) origin = lens[i];
    }
    const int64_t root_dist = origin > lens[0] ? origin - lens[0] : 0;
    auto is_pinned = [&](size_t i) { return pinned != nullptr && pinned[i]; };
    if (tail != nullptr)
        for (size_t i = 1; i < n; ++i)
            if (tail[i] && !is_pinned(i) && i != newest) return i;
    auto sort_asc = [](int64_t* a, size_t m) {   // insertion sort: a handful of items
        for (size_t i = 1; i < m; ++i) {
            const int64_t d = a[i];
            size_t b = i;
            while (b > 0 && a[b - 1] > d) { a[b] = a[b - 1]; --b; }
            a[b] = d;
        }
    };
    // the targets for m non-root items: the geometric grid up to the root, then the largest gap halved
    int64_t target[64];
    const size_t m = n - 2;   // non-root items once one has left
    {
        size_t t = 0;
        for (int64_t d = 0, gap = every; t < m && d < root_dist; d += gap, gap *= 2) target[t++] = d;
        while (t < m) {   // the points so far plus the root, sorted; halve the widest gap
            int64_t pts[66];
            size_t k = 0;
            for (size_t i = 0; i < t; ++i) pts[k++] = target[i];
            pts[k++] = root_dist;
            sort_asc(pts, k);
            int64_t lo = 0, hi = root_dist, widest = -1;
            for (size_t i = 1; i < k; ++i)
                if (pts[i] - pts[i - 1] > widest) { widest = pts[i] - pts[i - 1]; lo = pts[i - 1]; hi = pts[i]; }
            target[t++] = widest > 0 ? (lo + hi) / 2 : root_dist;
        }
        sort_asc(target, m);
    }
    size_t best = n;
    double best_cost = 0.0;
    for (size_t v = 1; v < n; ++v) {
        if (is_pinned(v) || v == newest) continue;
        int64_t dist[64];
        size_t k = 0;
        for (size_t i = 1; i < n; ++i)
            if (i != v) dist[k++] = origin - lens[i];
        sort_asc(dist, k);
        double cost = 0.0;
        for (size_t i = 0; i < k && i < m; ++i) {
            const int64_t diff = dist[i] > target[i] ? dist[i] - target[i] : target[i] - dist[i];
            cost += (double) diff / (double) (target[i] + every);
        }
        if (best == n || cost < best_cost || (cost == best_cost && stamps[v] < stamps[best])) { best = v; best_cost = cost; }
    }
    if (best != n) return best;
    return eviction_victim(stamps, n, cap, tail, pinned);   // nothing but pins and the newest: the backstop
}

}  // namespace strata::program::conv_cache
