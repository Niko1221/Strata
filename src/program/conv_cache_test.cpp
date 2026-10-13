// src/program/conv_cache_test.cpp - the conversation cache's retention policy: the shared root survives,
// the leaves rotate by least recent use.
//
// The scenarios, in the serve loop's terms (generate.cpp): checkpoints are a prefix chain - stamps[0] is
// the root, the deepest point every request so far has shared (the end of the system prompt, in practice).
// A stamp advances on creation, when a new checkpoint lands on an existing length, and when a request
// mounts through the checkpoint.
//
//   1. one conversation over its budget: the oldest leaf leaves - with fresh stamps in creation order this
//      is exactly the previous first-in-first-out behaviour;
//   2. the root survives arbitrarily many newer checkpoints however old its stamp is;
//   3. a mount advances a checkpoint's stamp and the rotation then takes a different leaf;
//   4. a one-slot budget has no room for root and leaf, so the pin is off and the oldest leaves (the
//      newest point stays, as before);
//   5. the policy is pure: the same stamps, the same victim.
#include "strata/program/conv_cache.hpp"

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <cstdio>
#include <vector>

using strata::program::conv_cache::eviction_victim;

namespace {
int g_fail = 0;
void check(bool ok, const char* what) {
    std::printf("  %-66s %s\n", what, ok ? "ok" : "FAIL");
    if (!ok) ++g_fail;
}
}  // namespace

int main() {
    std::printf("conv_cache_test\n");
    {
        const std::vector<uint64_t> stamps = {1, 2, 3, 4, 5, 6, 7};   // chat A: root + six turns, one over
        check(eviction_victim(stamps.data(), stamps.size(), 6) == 1,
              "over budget with fresh stamps: the oldest leaf leaves (the FIFO it was)");
    }
    {
        // chat B mounts through the root long after it was created; its stamp stays ancient
        const std::vector<uint64_t> stamps = {1, 100, 101, 102, 103, 104, 105};
        const size_t v = eviction_victim(stamps.data(), stamps.size(), 6);
        check(v == 1, "the root's stale stamp never makes it the victim");
        check(v != 0, "the root - the shared prefix - is never the victim");
    }
    {
        // the same chain after mounting checkpoint 1: its stamp jumps, the rotation moves to the next leaf
        const std::vector<uint64_t> stamped = {1, 8, 3, 4, 5, 6, 7};
        check(eviction_victim(stamped.data(), stamped.size(), 6) == 2,
              "after a mount, the rotation takes the next least recently used leaf");
    }
    {
        const std::vector<uint64_t> stamps = {1, 90, 91, 92};
        check(eviction_victim(stamps.data(), stamps.size(), 2) == 1,
              "a two-slot budget keeps root + newest and drops the leaf in between");
        check(eviction_victim(stamps.data(), stamps.size(), 1) == 0,
              "a one-slot budget has no pin: the oldest leaves, the newest point stays");
    }
    {
        const std::vector<uint64_t> stamps = {7, 3, 3, 5};
        const size_t a = eviction_victim(stamps.data(), stamps.size(), 4);
        const size_t b = eviction_victim(stamps.data(), stamps.size(), 4);
        check(a == 1 && b == 1, "ties among equally stale leaves: the earlier index, deterministically");
    }
    {   // --prompt-cache-tail: the tail checkpoint goes first; never the root, never the newest item
        const std::vector<uint64_t> st = {1, 2, 3, 4, 5, 6, 7};
        const bool t2[7] = {false, false, false, true, false, false, false};
        check(eviction_victim(st.data(), st.size(), 6, t2) == 3, "a stale tail checkpoint leaves before the oldest leaf");
        const bool t3[7] = {false, false, false, false, false, false, true};
        check(eviction_victim(st.data(), st.size(), 6, t3) == 1, "the newest item, even a tail, is not the victim");
        const bool t4[7] = {true, false, false, false, false, false, false};
        check(eviction_victim(st.data(), st.size(), 6, t4) == 1, "the root is never the victim, tail flag or not");
        check(eviction_victim(st.data(), st.size(), 6, nullptr) == 1, "no tail flags: the plain policy");
    }
    {   // pin=N: a pinned shared prefix is never the victim, however stale its stamp
        const std::vector<uint64_t> st = {1, 2, 30, 4, 5, 6, 7};
        const bool pin1[7] = {false, true, false, false, false, false, false};
        check(eviction_victim(st.data(), st.size(), 6, nullptr, pin1) == 3,
              "the oldest checkpoint is pinned: the next least recently used leaves");
        check(eviction_victim(st.data(), st.size(), 6, nullptr, nullptr) == 1, "no pin flags: the plain policy");
        const bool pin2[7] = {false, true, false, true, false, false, false};
        check(eviction_victim(st.data(), st.size(), 6, nullptr, pin2) == 4, "two pins: the third oldest leaves");
        const bool tail3[7] = {false, false, false, true, false, false, false};
        check(eviction_victim(st.data(), st.size(), 6, tail3, pin2) == 4, "a pinned item is not taken as the tail either");
        const bool all[4] = {false, true, true, true};
        const std::vector<uint64_t> s4 = {1, 9, 3, 5};
        check(eviction_victim(s4.data(), s4.size(), 4, nullptr, all) == 2, "everything pinned: the oldest pin leaves, never the root");
        check(eviction_victim(s4.data(), s4.size(), 1, nullptr, all) == 0, "a one-slot budget keeps no pin: the oldest leaves");
    }
    {   // --prompt-cache-spread: the first checkpoint in every 16K tokens stays, the turns between them rotate
        using strata::program::conv_cache::eviction_victim_spread;
        const int64_t E = 16384;
        // an agent session: the root, a milestone at 17K, then turn after turn up to 40K (stamps in creation order)
        const std::vector<int64_t> len = {12819, 17000, 17500, 33000, 33400, 39000, 39600, 40200};
        const std::vector<uint64_t> st = {1, 2, 3, 4, 5, 6, 7, 8};
        check(eviction_victim_spread(st.data(), len.data(), len.size(), 7, E, nullptr, nullptr) == 2,
              "spread: the oldest turn that is not a milestone leaves");
        check(eviction_victim(st.data(), st.size(), 7, nullptr, nullptr) == 1, "plain policy, same chain: the 17K milestone leaves");
        // only milestones and the two newest are left: the milestone with the closest neighbours leaves
        const std::vector<int64_t> l2 = {12819, 17000, 33000, 49200, 49300, 66000, 82000, 82500};
        const std::vector<uint64_t> s2 = {1, 2, 3, 4, 9, 5, 6, 7};
        check(eviction_victim_spread(s2.data(), l2.data(), l2.size(), 7, E, nullptr, nullptr) == 3,
              "spread: a milestone with a newer point right above it leaves first");
        const std::vector<int64_t> l3 = {12819, 17000, 33000, 49200, 66000, 82000, 98500, 99000};
        const std::vector<uint64_t> s3 = {1, 2, 3, 4, 5, 6, 7, 8};
        check(eviction_victim_spread(s3.data(), l3.data(), l3.size(), 7, E, nullptr, nullptr) == 1,
              "spread: only milestones left - the one closest to its neighbours (17K, beside the root)");
        const std::vector<uint64_t> s3b = {1, 9, 3, 4, 5, 6, 7, 8};   // 17K is the newest now (a request resumed from it)
        const size_t v3b = eviction_victim_spread(s3b.data(), l3.data(), l3.size(), 7, E, nullptr, nullptr);
        check(v3b != 0 && v3b != 1 && v3b != 7, "spread: never the root nor the two most recently used");
        const bool pin3[8] = {false, true, false, false, false, false, false, false};
        check(eviction_victim_spread(s3.data(), l3.data(), l3.size(), 7, E, nullptr, pin3) != 1, "spread: a pinned item stays");
        const bool tail3[8] = {false, false, false, false, true, false, false, false};
        check(eviction_victim_spread(s3.data(), l3.data(), l3.size(), 7, E, tail3, nullptr) == 4, "spread: the tail checkpoint leaves first");
        // off conditions: the plain policy
        check(eviction_victim_spread(st.data(), len.data(), len.size(), 7, 0, nullptr, nullptr) == 1, "spread with every = 0: the plain policy");
        check(eviction_victim_spread(st.data(), nullptr, len.size(), 7, E, nullptr, nullptr) == 1, "spread without lengths: the plain policy");
        check(eviction_victim_spread(st.data(), len.data(), len.size(), 3, E, nullptr, nullptr) == 1, "spread with a 3-slot budget: the plain policy");
        // a whole session: 500-token turns from 13K to 131K with 14 slots - what a request that departs at X reads again
        std::vector<int64_t> L = {12819};
        std::vector<uint64_t> S = {1};
        uint64_t clock = 1;
        for (int64_t at = 13300; at <= 131000; at += 500) {
            L.push_back(at);
            S.push_back(++clock);
            while ((int64_t) L.size() > 14) {
                const size_t v = eviction_victim_spread(S.data(), L.data(), L.size(), 14, E, nullptr, nullptr);
                L.erase(L.begin() + (std::ptrdiff_t) v);
                S.erase(S.begin() + (std::ptrdiff_t) v);
            }
        }
        int64_t worst = 0;
        for (int64_t x = 13000; x <= 131000; x += 250) {
            int64_t below = 0;
            for (const int64_t l : L) if (l <= x && l > below) below = l;
            worst = std::max(worst, x - below);
        }
        check(L.size() == 14 && L[0] == 12819, "spread, a 131K session by 500-token turns: 14 kept, the root among them");
        check(worst <= E + 500, "spread: a departure anywhere reads at most one spacing and a turn again");
    }
    std::printf(g_fail ? "FAIL\n" : "PASS\n");
    return g_fail ? 1 : 0;
}
