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
//   5. the policy is pure: the same stamps, the same victim;
//   6. --prompt-cache-policy log: a conversation grown to 256K with the root at 30K and a checkpoint every 16K
//      keeps a dense tail and far anchors (the largest gap 96K against the LRU's 162K); the root, the newest
//      item and a pinned item are never the victim, the tail checkpoint still goes first, and without a period
//      (`every` 0), without lengths, or with more than 64 items it is the plain policy.
#include "strata/program/conv_cache.hpp"

#include <algorithm>
#include <cstdint>
#include <cstdio>
#include <vector>

using strata::program::conv_cache::eviction_victim;
using strata::program::conv_cache::eviction_victim_log;

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
    {   // --prompt-cache-policy log: grow a conversation checkpoint by checkpoint and see what stays
        struct Item { int64_t len; uint64_t stamp; };
        auto grow = [](bool log, int64_t L, int cap, int64_t every, int64_t root) {
            std::vector<Item> c;
            uint64_t clock = 0;
            c.push_back({root, ++clock});
            for (int64_t p = every; p <= L; p += every) {
                if (p <= root) continue;
                c.push_back({p, ++clock});
                while ((int) c.size() > cap) {
                    std::vector<uint64_t> st;
                    std::vector<int64_t> ln;
                    for (const Item& k : c) { st.push_back(k.stamp); ln.push_back(k.len); }
                    const size_t v = log ? eviction_victim_log(st.data(), ln.data(), c.size(), cap, every, nullptr, nullptr)
                                         : eviction_victim(st.data(), c.size(), cap, nullptr, nullptr);
                    c.erase(c.begin() + (std::ptrdiff_t) v);
                }
            }
            std::vector<int64_t> lens;
            for (const Item& k : c) lens.push_back(k.len);
            std::sort(lens.begin(), lens.end());
            return lens;
        };
        auto largest_gap = [](const std::vector<int64_t>& lens, int64_t L) {
            int64_t worst = 0, prev = 0;
            for (int64_t l : lens) { worst = std::max(worst, l - prev); prev = l; }
            return std::max(worst, L - prev);
        };
        const int64_t K = 1024, every = 16 * K, root = 30 * K;
        {
            const std::vector<int64_t> lru = grow(false, 256 * K, 6, every, root);
            const std::vector<int64_t> log = grow(true, 256 * K, 6, every, root);
            check(lru == std::vector<int64_t>({30 * K, 192 * K, 208 * K, 224 * K, 240 * K, 256 * K}),
                  "lru at 256K: the root and the newest five (the shape the docs describe)");
            check(log == std::vector<int64_t>({30 * K, 80 * K, 128 * K, 224 * K, 240 * K, 256 * K}),
                  "log at 256K: the root, two far anchors, and the newest three");
            check(largest_gap(lru, 256 * K) == 162 * K && largest_gap(log, 256 * K) == 96 * K,
                  "log at 256K: the largest blind gap is 96K against the lru's 162K");
            check(log[0] == root, "log: the root stays");
            check(log[log.size() - 1] == 256 * K && log[log.size() - 2] == 240 * K, "log: the tail keeps the last two periodic points");
        }
        {
            const std::vector<int64_t> lru = grow(false, 128 * K, 6, every, root);
            const std::vector<int64_t> log = grow(true, 128 * K, 6, every, root);
            check(largest_gap(log, 128 * K) <= largest_gap(lru, 128 * K) && log[0] == root && log.back() == 128 * K,
                  "log at 128K (the first evictions): no wider gap than lru, the root and the newest point kept");
        }
        {
            const std::vector<int64_t> lru = grow(false, 1024 * K, 6, every, root);
            const std::vector<int64_t> log = grow(true, 1024 * K, 6, every, root);
            check(largest_gap(log, 1024 * K) < largest_gap(lru, 1024 * K),
                  "log at 1M: the largest gap is smaller than the lru's");
            check(log[0] == root && log[log.size() - 1] == 1024 * K, "log at 1M: the root and the newest point stay");
        }
    }
    {   // the log policy's guards: never the root, never the newest stamp, never a pin; the tail goes first
        const int64_t K = 1024;
        const std::vector<uint64_t> st = {1, 2, 3, 4, 5, 9, 7};   // item 5 was just mounted (the newest stamp)
        const std::vector<int64_t> ln = {30 * K, 48 * K, 64 * K, 80 * K, 96 * K, 112 * K, 128 * K};
        const size_t v = eviction_victim_log(st.data(), ln.data(), st.size(), 6, 16 * K, nullptr, nullptr);
        check(v != 0 && v != 5, "log: neither the root nor the newest-stamped item is the victim");
        check(eviction_victim_log(st.data(), ln.data(), st.size(), 6, 16 * K, nullptr, nullptr) == v, "log: pure - the same chain, the same victim");
        const bool pin[7] = {false, false, false, false, false, false, false};
        bool pin_v[7] = {false, false, false, false, false, false, false};
        pin_v[v] = true;
        check(eviction_victim_log(st.data(), ln.data(), st.size(), 6, 16 * K, nullptr, pin_v) != v, "log: a pinned item is never the victim");
        check(eviction_victim_log(st.data(), ln.data(), st.size(), 6, 16 * K, nullptr, pin) == v, "log: pin flags all false: the same victim");
        const bool tail[7] = {false, false, false, true, false, false, false};
        check(eviction_victim_log(st.data(), ln.data(), st.size(), 6, 16 * K, tail, nullptr) == 3, "log: the tail checkpoint still goes first");
        check(eviction_victim_log(st.data(), ln.data(), st.size(), 6, 0, nullptr, nullptr) == eviction_victim(st.data(), st.size(), 6, nullptr, nullptr),
              "log without a period (every 0): the plain policy");
        check(eviction_victim_log(st.data(), nullptr, st.size(), 6, 16 * K, nullptr, nullptr) == eviction_victim(st.data(), st.size(), 6, nullptr, nullptr),
              "log without lengths: the plain policy");
        const std::vector<uint64_t> s2 = {1, 2};
        const std::vector<int64_t> l2 = {30 * K, 48 * K};
        check(eviction_victim_log(s2.data(), l2.data(), 2, 1, 16 * K, nullptr, nullptr) == 0, "log with a one-slot budget: the plain policy (the oldest leaves)");
        {   // index 0 is the root by creation order, not by length: a chain rebuilt from a slot may start longer
            const std::vector<uint64_t> st3 = {1, 2, 3, 4, 5, 6, 7};
            const std::vector<int64_t> ln3 = {64 * K, 32 * K, 48 * K, 80 * K, 96 * K, 112 * K, 128 * K};
            const size_t v3 = eviction_victim_log(st3.data(), ln3.data(), st3.size(), 6, 16 * K, nullptr, nullptr);
            check(v3 != 0 && v3 != 6 && v3 < 7, "log with a root that is not the shortest item: still never the root or the newest");
        }
        {   // the root is the longest item (nothing behind the origin): no grid to match, still a valid victim
            const std::vector<uint64_t> st4 = {1, 2, 3, 4};
            const std::vector<int64_t> ln4 = {128 * K, 32 * K, 48 * K, 64 * K};
            const size_t v4 = eviction_victim_log(st4.data(), ln4.data(), st4.size(), 3, 16 * K, nullptr, nullptr);
            check(v4 >= 1 && v4 <= 2, "log with the root as the longest item: a leaf other than the newest leaves");
        }
        {   // the plain policy for a chain the grid does not take: more than 64 items
            std::vector<uint64_t> st5;
            std::vector<int64_t> ln5;
            for (int i = 0; i < 70; ++i) { st5.push_back((uint64_t) (i + 1)); ln5.push_back((int64_t) (i + 2) * K); }
            check(eviction_victim_log(st5.data(), ln5.data(), st5.size(), 69, 16 * K, nullptr, nullptr) ==
                  eviction_victim(st5.data(), st5.size(), 69, nullptr, nullptr), "log with more than 64 items: the plain policy");
        }
    }
    std::printf(g_fail ? "FAIL\n" : "PASS\n");
    return g_fail ? 1 : 0;
}
