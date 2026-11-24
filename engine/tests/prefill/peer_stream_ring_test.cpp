// Peer-stream ring grouping and deferred-release slot safety (src/prefill/peer_stream_ring.hpp):
// header-only, no GPU.
#include "../../src/prefill/peer_stream_ring.hpp"

#include <cstdio>
#include <random>
#include <utility>
#include <vector>

namespace d = strata::prefill::detail;

static int failed = 0;
#define CHECK(x) do { if (!(x)) { std::fprintf(stderr, "FAIL %s:%d: %s\n", __FILE__, __LINE__, #x); ++failed; } } while (0)

namespace {

int simulate_peer_stream_ring(int n_layers, int n_expert, int rp, size_t group_max, int64_t row_cap,
                              bool sparse_gap_mode, bool apply_fix, unsigned seed) {
    std::mt19937 rng(seed);
    std::vector<std::pair<int32_t, int32_t>> pseq;   // (layer, expert)
    std::vector<size_t> pseq_start((size_t) n_layers + 1, 0);
    std::vector<std::vector<int32_t>> cnt((size_t) n_layers, std::vector<int32_t>((size_t) n_expert, 0));
    std::vector<std::vector<char>> on_peer((size_t) n_layers, std::vector<char>((size_t) n_expert, 0));

    for (int l = 0; l < n_layers; ++l) {
        pseq_start[(size_t) l] = pseq.size();
        for (int32_t e = 0; e < n_expert; ++e) {
            if (e % 3 == 0) {
                on_peer[(size_t) l][(size_t) e] = 1;   // peer-resident
                cnt[(size_t) l][(size_t) e] = (int32_t) (rng() % 32);
            } else if (e % 2 == 1) {
                on_peer[(size_t) l][(size_t) e] = 2;   // peer-streamed
                pseq.push_back({l, e});
                if (sparse_gap_mode) {
                    const int pos = e % (rp + 4);
                    cnt[(size_t) l][(size_t) e] = (pos == 1 || pos >= rp + 1) ? 16 : 0;
                } else {
                    cnt[(size_t) l][(size_t) e] = (rng() % 100 < 55) ? 0 : (int32_t) (1 + rng() % 48);
                }
            }
        }
    }
    pseq_start[(size_t) n_layers] = pseq.size();

    // Virtual async copy ring tracking which (layer, expert) owns each slot and whether pused was recorded.
    std::vector<int64_t> slot_entry((size_t) rp, -1);
    std::vector<bool> slot_in_use((size_t) rp, false);
    size_t p_issued = 0;
    size_t pk = 0;
    int corruptions = 0;

    auto issue_until = [&](size_t limit) {
        limit = std::min(limit, pseq.size());
        while (p_issued < limit) {
            const size_t sl = p_issued % (size_t) rp;
            if (slot_in_use[sl]) ++corruptions;   // async copy overwrote an un-released held slot
            slot_entry[sl] = (int64_t) p_issued;
            slot_in_use[sl] = true;
            ++p_issued;
        }
    };

    auto release_to = [&](int l, int32_t e_stop, bool issue_more) {
        while (pk < pseq_start[(size_t) l + 1] && pseq[pk].second < e_stop) {
            slot_in_use[pk % (size_t) rp] = false;
            ++pk;
            if (issue_more) issue_until(pk + (size_t) rp);
        }
    };

    issue_until((size_t) rp);

    for (int l = 0; l < n_layers; ++l) {
        std::vector<int32_t> order_peer;
        for (int32_t e = 0; e < n_expert; ++e)
            if (cnt[(size_t) l][(size_t) e] > 0 && on_peer[(size_t) l][(size_t) e] == 1)
                order_peer.push_back(e);
        for (int32_t e = 0; e < n_expert; ++e)
            if (cnt[(size_t) l][(size_t) e] > 0 && on_peer[(size_t) l][(size_t) e] == 2)
                order_peer.push_back(e);

        std::vector<std::pair<size_t, size_t>> groups;
        d::build_peer_ring_groups(
            order_peer, cnt[(size_t) l], on_peer[(size_t) l], group_max, row_cap,
            true, pseq_start[(size_t) l], pseq_start[(size_t) l + 1],
            apply_fix ? rp : 1000000000,
            [&](size_t idx) { return pseq[idx].second; },
            groups);

        for (const auto& grp : groups) {
            std::vector<size_t> held_slots;
            std::vector<int64_t> expected_entries;
            for (size_t j = grp.first; j < grp.second; ++j) {
                const int32_t e = order_peer[j];
                if (on_peer[(size_t) l][(size_t) e] == 2) {
                    const bool issue_more = apply_fix
                        ? d::peer_ring_issue_allowed(true, (int) held_slots.size())
                        : true;
                    release_to(l, e, issue_more);
                    CHECK(pk < pseq_start[(size_t) l + 1] && pseq[pk].second == e);
                    CHECK(pk < p_issued);
                    held_slots.push_back(pk % (size_t) rp);
                    expected_entries.push_back((int64_t) pk);
                    ++pk;
                }
            }
            // Verify every held slot still contains the exact expected streamed expert when the group kernel runs!
            for (size_t i = 0; i < held_slots.size(); ++i) {
                if (!slot_in_use[held_slots[i]] || slot_entry[held_slots[i]] != expected_entries[i]) {
                    ++corruptions;
                }
            }
            for (size_t sl : held_slots) slot_in_use[sl] = false;
            if (!held_slots.empty()) issue_until(pk + (size_t) rp);
        }
        release_to(l, n_expert, true);
    }
    return corruptions;
}

}  // namespace

int main() {
    // 1. Unit checks for deferred-group issue guard
    CHECK(d::peer_ring_issue_allowed(false, 0));
    CHECK(d::peer_ring_issue_allowed(false, 4));
    CHECK(d::peer_ring_issue_allowed(true, 0));
    CHECK(!d::peer_ring_issue_allowed(true, 1));
    CHECK(!d::peer_ring_issue_allowed(true, 16));

    // 2. Reproduction of the unfixed ring overwrite bug vs zero corruptions with the fix across RP sizes
    CHECK(simulate_peer_stream_ring(4, 128, 16, 16, 4096, true, false, 1u) > 0);
    for (int rp : {1, 2, 4, 8, 16, 48}) {
        for (unsigned seed = 1; seed <= 5; ++seed) {
            CHECK(simulate_peer_stream_ring(6, 256, rp, 16, 2048, false, true, seed) == 0);
            CHECK(simulate_peer_stream_ring(4, 128, rp, 16, 4096, true, true, seed) == 0);
        }
    }

    std::printf("peer_stream_ring_test: %s\n", failed ? "FAIL" : "ok");
    return failed ? 1 : 0;
}
