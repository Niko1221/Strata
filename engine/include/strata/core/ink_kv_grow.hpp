// Inkling-2-Mini (strata-inkling CUDA): when may the global K/V grow in VRAM? (#6770, #6755)
//
// The 8 global visits grow one visit at a time: the new, larger buffers of a visit are made first, the data is copied, then
// the old buffers are freed. So the device never needs the whole new K/V next to the whole old one; it needs the net
// growth of every visit plus the one old visit that is still held while its replacement is made.
//
// The earlier check compared the free memory with the WHOLE new K/V (n_glob x slots x new_cap), as if the old K/V were not
// already in the card. With a plan that leaves room for the full context, that check starts to refuse when the K/V reaches
// about half of it, and the next long request fails with HTTP 503 while gigabytes are still free.
#pragma once

#include <algorithm>
#include <cstddef>
#include <cstdint>

namespace strata::core {

// Next global-K/V capacity (positions) when `needed_pos` does not fit in `old_cap`: whole 8K steps, at least what was asked,
// never past `max_ctx`, 64-aligned.
inline int ink_kv_next_cap(int old_cap, int needed_pos, int max_ctx) {
    int n = std::min(max_ctx, std::max(needed_pos, ((old_cap + 8191) / 8192 + 1) * 8192));
    return std::min(max_ctx, ((n + 63) & ~63));
}

// Peak extra device bytes while `n_glob` visits grow from `old_cap` to `new_cap` positions for `slots` slots, one visit at a time.
// per_pos = bytes of one visit for one position of one slot (K and V, and the FP8 scales when they are used).
inline size_t ink_kv_grow_peak_bytes(int n_glob, int slots, int old_cap, int new_cap, size_t per_pos) {
    if (n_glob <= 0 || new_cap <= old_cap) return 0;
    const size_t s = (size_t)std::max(1, slots);
    const size_t net = (size_t)n_glob * s * (size_t)(new_cap - old_cap) * per_pos;
    const size_t held_old_visit = s * (size_t)old_cap * per_pos;
    return net + held_old_visit;
}

// Does the growth fit in `vfree` bytes while `floor_b` stays free?
inline bool ink_kv_grow_fits(size_t vfree, size_t floor_b, int n_glob, int slots, int old_cap, int new_cap, size_t per_pos) {
    return vfree > floor_b + ink_kv_grow_peak_bytes(n_glob, slots, old_cap, new_cap, per_pos);
}

// Host tail (positions) to map when the card cannot grow the K/V any further and the rest of the K/V goes to pinned host RAM: the WHOLE rest of the
// context first, so the K/V never has to grow again (a spill sized to just the request that hit the limit made the next, longer request end the
// engine: "VRAM exhausted growing elastic global KV cache to 40961 tokens", #6905), and only when that does not fit in host memory, just the request.
inline int ink_kv_host_tail_full(int kv_dev_ctx, int max_ctx) { return std::max(0, max_ctx - kv_dev_ctx); }
inline int ink_kv_host_tail_need(int kv_dev_ctx, int new_cap) { return std::max(0, new_cap - kv_dev_ctx); }

// The capacity to grow to: the full step when it fits, else the tight one (just the request, 256-aligned), else 0 (no room:
// the caller spills the tail to mapped host memory or refuses).
inline int ink_kv_grow_target(size_t vfree, size_t floor_b, int n_glob, int slots, int old_cap, int needed_pos,
                              int max_ctx, size_t per_pos) {
    const int full = ink_kv_next_cap(old_cap, needed_pos, max_ctx);
    if (ink_kv_grow_fits(vfree, floor_b, n_glob, slots, old_cap, full, per_pos)) return full;
    const int tight = std::min(max_ctx, ((needed_pos + 255) & ~255));
    if (tight > old_cap && ink_kv_grow_fits(vfree, floor_b, n_glob, slots, old_cap, tight, per_pos)) return std::min(full, tight);
    return 0;
}

}  // namespace strata::core
