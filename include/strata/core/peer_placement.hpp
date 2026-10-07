#pragma once

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <limits>
#include <stdexcept>
#include <utility>
#include <vector>

namespace strata::core {

inline bool peer_capacity_enabled(int peer_device, const char* value) {
    return peer_device >= 1 && value != nullptr && std::strcmp(value, "1") == 0;
}

struct PeerPlacement {
    using Pair = std::pair<int32_t, int32_t>;
    std::vector<Pair> ranked, primary, peer, missed;
    uint64_t primary_bytes = 0, peer_bytes = 0;
};

// STRP v1 records rank, not counts. Reciprocal square-root rank is a load proxy,
// not a measured frequency. Byte feasibility always takes precedence over it.
inline PeerPlacement plan_peer_placement(const std::vector<PeerPlacement::Pair>& ranked,
                                         const std::vector<uint64_t>& layer_bytes, int32_t n_expert,
                                         uint64_t primary_cap, uint64_t peer_cap,
                                         size_t peer_slots = std::numeric_limits<size_t>::max()) {
    if (n_expert <= 0 || layer_bytes.empty())
        throw std::invalid_argument("peer placement: empty model");
    std::vector<uint64_t> bytes;
    for (uint64_t b : layer_bytes) {
        if (b == 0 || b > std::numeric_limits<uint64_t>::max() - 255)
            throw std::invalid_argument("peer placement: invalid blob size");
        bytes.push_back((b + 255) / 256 * 256);
    }
    PeerPlacement plan;
    std::vector<uint8_t> seen(bytes.size() * (size_t) n_expert, 0);
    for (const auto& pr : ranked) {
        if (pr.first < 0 || (size_t) pr.first >= bytes.size() || pr.second < 0 || pr.second >= n_expert)
            throw std::invalid_argument("peer placement: pair out of range");
        const size_t i = (size_t) pr.first * (size_t) n_expert + (size_t) pr.second;
        if (!seen[i]) { seen[i] = 1; plan.ranked.push_back(pr); }
    }
    // Old profiles may name only 8,000 pairs. Fill the cold tail across layers.
    const size_t profiled = plan.ranked.size();
    for (int32_t e = 0; e < n_expert; ++e)
        for (size_t l = 0; l < bytes.size(); ++l)
            if (!seen[l * (size_t) n_expert + (size_t) e])
                plan.ranked.emplace_back((int32_t) l, e);

    uint64_t used[2] = {0, 0}, cap[2] = {primary_cap, peer_cap};
    size_t count[2] = {0, 0};
    double heat[2] = {0.0, 0.0};
    std::vector<int8_t> owner(plan.ranked.size(), -1);
    auto weight = [&](size_t r) { return r < profiled ? 1.0 / std::sqrt((double) r + 1.0) : 0.0; };
    auto fits = [&](int d, uint64_t b) {
        return b <= cap[d] - used[d] && (d == 0 || count[1] < peer_slots);
    };
    for (size_t r = 0; r < plan.ranked.size(); ++r) {
        const uint64_t b = bytes[(size_t) plan.ranked[r].first];
        bool a = fits(0, b), p = fits(1, b);
        // A bounded one-expert relocation avoids stranding a large blob behind a
        // smaller one when the other card can hold the smaller one.
        if (!a && !p) {
            const size_t begin = r > 64 ? r - 64 : 0;
            for (size_t j = r; j-- > begin;) {
                const int from = owner[j];
                if (from < 0) continue;
                const int to = 1 - from;
                const uint64_t old = bytes[(size_t) plan.ranked[j].first];
                if (!fits(to, old) || b > cap[from] - used[from] + old) continue;
                used[from] -= old; --count[from]; heat[from] -= weight(j);
                used[to] += old; ++count[to]; heat[to] += weight(j);
                owner[j] = (int8_t) to;
                a = fits(0, b); p = fits(1, b);
                break;
            }
        }
        if (!a && !p) continue;   // a later, smaller blob may still fit
        const int d = !a ? 1 : !p ? 0 : heat[0] != heat[1] ? (heat[0] < heat[1] ? 0 : 1)
                                                       : (cap[0] - used[0] >= cap[1] - used[1] ? 0 : 1);
        owner[r] = (int8_t) d;
        used[d] += b; ++count[d]; heat[d] += weight(r);
    }
    for (size_t r = 0; r < plan.ranked.size(); ++r)
        (owner[r] == 0 ? plan.primary : owner[r] == 1 ? plan.peer : plan.missed).push_back(plan.ranked[r]);
    plan.primary_bytes = used[0];
    plan.peer_bytes = used[1];
    return plan;
}

}  // namespace strata::core
