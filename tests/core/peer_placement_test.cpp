// CPU only; standalone compilation needs no CUDA headers or runtime.
#include "strata/core/peer_placement.hpp"

#include <cstdio>
#include <set>

namespace {
int fails = 0;
void check(bool ok, const char* what) {
    if (!ok) { std::fprintf(stderr, "FAIL: %s\n", what); ++fails; }
}
using Pair = strata::core::PeerPlacement::Pair;
using strata::core::plan_peer_placement;

void invariant(const strata::core::PeerPlacement& p, const std::vector<uint64_t>& bytes,
               int32_t ne, uint64_t a, uint64_t b, size_t slots = SIZE_MAX) {
    std::set<Pair> all;
    uint64_t used[2] = {0, 0};
    size_t d = 0;
    for (const auto* pairs : {&p.primary, &p.peer, &p.missed}) {
        for (const Pair& pr : *pairs) {
            check(all.insert(pr).second, "ownership has no duplication or overlap");
            if (d < 2) used[d] += (bytes[(size_t) pr.first] + 255) / 256 * 256;
        }
        ++d;
    }
    check(all.size() == bytes.size() * (size_t) ne, "resident and missed ownership covers every pair");
    check(used[0] == p.primary_bytes && used[1] == p.peer_bytes, "accounting uses aligned layer sizes");
    check(used[0] <= a && used[1] <= b && p.peer.size() <= slots, "byte capacities and peer slot cap respected");
    check(p.ranked.size() == all.size(), "complete unique ranking");
}
}  // namespace

int main() {
    using strata::core::peer_capacity_enabled;
    for (const char* flag : std::vector<const char*>{nullptr, "", "0", "true", "01", "2"})
        check(!peer_capacity_enabled(1, flag), "disabled or non-explicit flag preserves legacy selection");
    check(!peer_capacity_enabled(-1, "1") && !peer_capacity_enabled(0, "1"),
          "flag alone never changes single-GPU selection");
    check(peer_capacity_enabled(1, "1"), "only explicit peer and opt-in enable new ownership");
    for (const std::vector<Pair>& invalid : {std::vector<Pair>{{1, 0}}, std::vector<Pair>{{0, -1}}}) {
        bool rejected = false;
        try { (void) plan_peer_placement(invalid, {256}, 1, 256, 256); }
        catch (const std::invalid_argument&) { rejected = true; }
        check(rejected, "invalid ranked pair rejected before planning");
    }
    bool rejected = false;
    try { (void) plan_peer_placement({}, {UINT64_MAX}, 1, UINT64_MAX, UINT64_MAX); }
    catch (const std::invalid_argument&) { rejected = true; }
    check(rejected, "blob alignment overflow rejected before planning");

    const std::vector<uint64_t> bytes = {1, 257, 769};
    const std::vector<Pair> rank = {{2, 0}, {0, 0}, {1, 0}, {0, 0}, {2, 1}};
    for (const auto& caps : {std::pair<uint64_t, uint64_t>{0, 0}, {255, 255}, {256, 512},
                             {1536, 768}, {4096, 8192}, {8192, 0}}) {
        const auto p = plan_peer_placement(rank, bytes, 2, caps.first, caps.second);
        invariant(p, bytes, 2, caps.first, caps.second);
        if (caps.first + caps.second == 0 || caps.first == 255)
            check(p.primary.empty() && p.peer.empty(), "tiny capacity admits nothing");
        if (caps.first >= 4096)
            check(p.missed.empty(), "full byte capacity leaves no expert uncovered");
        check(p.ranked[0] == Pair(2, 0) && p.ranked[3] == Pair(2, 1), "rank order retained, duplicates removed");
    }
    const auto small = plan_peer_placement({{1, 0}, {0, 0}}, {1, 1025}, 1, 256, 256);
    invariant(small, {1, 1025}, 1, 256, 256);
    check(small.primary.size() + small.peer.size() == 1, "oversize hot pair does not leave a hole for smaller pairs");
    const auto repair = plan_peer_placement({{0, 0}, {1, 0}}, {512, 768}, 1, 768, 512);
    invariant(repair, {512, 768}, 1, 768, 512);
    check(repair.missed.empty() && repair.primary[0] == Pair(1, 0), "bounded relocation avoids stranding large blob");
    const auto capped = plan_peer_placement(rank, bytes, 2, 8192, 8192, 1);
    invariant(capped, bytes, 2, 8192, 8192, 1);
    check(capped.peer.size() == 1 && capped.missed.empty(), "explicit peer slot limit still covers through primary");

    // Regression: 21,076 primary slots swallowed the legacy 8,700-rank move.
    // Even an old 8,000-pair profile must place hot pairs on both owners and
    // complete the cold tail of this 48 x 512 model.
    std::vector<Pair> hot;
    for (int32_t e = 0; e < 512 && hot.size() < 8000; ++e)
        for (int32_t l = 0; l < 48 && hot.size() < 8000; ++l) hot.emplace_back(l, e);
    const std::vector<uint64_t> model(48, 2097152 - 31);
    const uint64_t primary = 21076ull * 2097152, peer = 21076ull * 2097152;
    const auto large = plan_peer_placement(hot, model, 512, primary, peer);
    invariant(large, model, 512, primary, peer);
    check(large.missed.empty(), "48 GiB-class combined capacities cover all 24,576 experts");
    std::set<Pair> peer_hot(large.peer.begin(), large.peer.end());
    size_t on_peer = 0;
    for (size_t r = 0; r < 100; ++r) on_peer += peer_hot.count(hot[r]);
    check(on_peer >= 40 && on_peer <= 60, "hottest ranks balanced despite primary capacity above 21,000");
    check(large.primary.front() == hot[0] && large.peer.front() == hot[1], "two hottest pairs get different owners");
    const auto unequal = plan_peer_placement(hot, model, 512, primary, 3500ull * 2097152);
    invariant(unequal, model, 512, primary, 3500ull * 2097152);
    check(unequal.primary.size() == 21076 && unequal.peer.size() == 3500 && unequal.missed.empty(),
          "exact unequal aligned capacity covers all pairs without holes");
    std::set<Pair> unequal_peer(unequal.peer.begin(), unequal.peer.end());
    on_peer = 0;
    for (size_t r = 0; r < 100; ++r) on_peer += unequal_peer.count(hot[r]);
    check(on_peer >= 40 && on_peer <= 60, "unequal capacities still balance the hottest ranks");

    uint32_t state = 7;
    for (int t = 0; t < 200; ++t) {
        auto next = [&]() { state = state * 1664525u + 1013904223u; return state; };
        std::vector<uint64_t> sizes;
        for (int l = 0; l < 5; ++l) sizes.push_back(1 + next() % 1800);
        const uint64_t a = next() % 20000, b = next() % 12000;
        const size_t limit = next() % 22;
        invariant(plan_peer_placement({}, sizes, 7, a, b, limit), sizes, 7, a, b, limit);
    }
    if (fails == 0) std::puts("peer_placement_test: OK");
    return fails == 0 ? 0 : 1;
}
