// Header-only peer-stream ring helpers for compact peer prompt groups. No GPU calls.
// Ensures deferred-release groups (peer_i8 / gather_native_group) never overwrite a held
// ring slot while unrouted streamed experts are skipped or across ring-wrap boundaries.
#pragma once

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <utility>
#include <vector>

namespace strata::prefill::detail {

/// Returns whether `p_release_to` may advance `p_issue_until` immediately after releasing an
/// unrouted peer-streamed expert inside a compact group.
///
/// When a group holds ring slots (`held_slots > 0`) for a deferred group kernel
/// (`peer_i8` or `gather_native_group`), advancing `p_issue_until(pk + RP)` inside the group
/// before the kernel records `pused` on those held slots wraps the ring around and overwrites
/// `pstage[pgg_slots[0]]` asynchronously. Only after the group kernel (or empty-row release)
/// records `pused` for `pgg_slots` may `p_issue_until` advance.
inline bool peer_ring_issue_allowed(bool deferred_group, int held_slots) noexcept {
    return !deferred_group || held_slots == 0;
}

/// Forms compact peer compute groups `[j0, j1)` over `order_peer`, honoring:
///  1. at most `group_max` experts per group,
///  2. total routed rows <= `row_cap` (unless one expert already starts the group),
///  3. when `ps_on`, every peer-streamed (`on_peer[e] == 2`) expert inside one group spans
///     strictly fewer than `ring_slots` entries in `pseq[ps_start .. ps_end)` (passed via
///     `pseq_expert(idx)` returning the expert id at `pseq[idx]`).
template <typename PseqExpertFn>
inline void build_peer_ring_groups(const std::vector<int32_t>& order_peer,
                                   const std::vector<int32_t>& cnt,
                                   const std::vector<char>& on_peer,
                                   size_t group_max,
                                   int64_t row_cap,
                                   bool ps_on,
                                   size_t ps_start,
                                   size_t ps_end,
                                   int ring_slots,
                                   PseqExpertFn&& pseq_expert,
                                   std::vector<std::pair<size_t, size_t>>& groups) {
    groups.clear();
    const size_t n = order_peer.size();
    const size_t rp = (size_t) std::max(ring_slots, 1);
    size_t g0 = 0;
    int64_t gr = 0;
    size_t ps_cur = ps_on ? ps_start : 0;
    size_t grp_ps0 = SIZE_MAX;
    for (size_t j = 0; j < n; ++j) {
        const int32_t e = order_peer[j];
        const int64_t c = cnt[(size_t) e];
        bool ps_split = false;
        if (ps_on && e >= 0 && (size_t) e < on_peer.size() && on_peer[(size_t) e] == 2) {
            while (ps_cur < ps_end && pseq_expert(ps_cur) < e) ++ps_cur;
            if (grp_ps0 != SIZE_MAX && ps_cur - grp_ps0 >= rp) ps_split = true;
        }
        if (j > g0 && (j - g0 == group_max || gr + c > row_cap || ps_split)) {
            groups.push_back({g0, j});
            g0 = j;
            gr = 0;
            grp_ps0 = SIZE_MAX;
        }
        if (ps_on && e >= 0 && (size_t) e < on_peer.size() && on_peer[(size_t) e] == 2 && grp_ps0 == SIZE_MAX) {
            grp_ps0 = ps_cur;
        }
        gr += c;
    }
    if (n > g0) groups.push_back({g0, n});
}

}  // namespace strata::prefill::detail
