// src/program/conv_cache_test.cpp - the conversation cache's retention policy: the shared root survives,
// the leaves rotate by least recent use.
//
// The scenarios, in the serve loop's terms (generate.cpp): checkpoints are a prefix chain; the root is the
// checkpoint at the end of the system prompt, the deepest point every request so far has shared.  It is
// stamps[0] unless the system prompt ends past the first periodic chunk end, which leaves grid checkpoints
// ahead of it - then the caller names its index (root_length / root_index and the `root` overloads; #2228,
// #3090).  A stamp advances on creation, when a new checkpoint lands on an existing length, and when a
// request mounts through the checkpoint.
//
//   1. one conversation over its budget: the oldest leaf leaves - with fresh stamps in creation order this
//      is exactly the previous first-in-first-out behaviour;
//   2. the root survives arbitrarily many newer checkpoints however old its stamp is;
//   3. a mount advances a checkpoint's stamp and the rotation then takes a different leaf;
//   4. a one-slot budget has no room for root and leaf, so the pin is off and the oldest leaves (the
//      newest point stays, as before);
//   5. the policy is pure: the same stamps, the same victim;
//   6. with grid checkpoints ahead of the root, the pin follows the caller-named root and item 0 rotates;
//      with root 0 the `root` overloads are the plain policy, checked exhaustively on every small chain.
#include "strata/program/conv_cache.hpp"

#include <cstdint>
#include <cstdio>
#include <memory>
#include <vector>

using strata::program::conv_cache::eviction_victim;
using strata::program::conv_cache::root_index;
using strata::program::conv_cache::root_length;
using strata::program::conv_cache::spread_victim;

namespace {
int g_fail = 0;
void check(bool ok, const char* what) {
    std::printf("  %-66s %s\n", what, ok ? "ok" : "FAIL");
    if (!ok) ++g_fail;
}
}  // namespace

// a conversation of `turns` turns of `turn` tokens each after a `root`-token system prompt, a checkpoint at every turn's
// end and at most `cap` kept; returns how many tokens a request whose prompt differs `back` tokens before the end reuses
int64_t reuse_after_edit(bool spread, int64_t root, int64_t turn, int turns, int64_t cap, int64_t back) {
    std::vector<int64_t> lens = {root};
    std::vector<uint64_t> stamps = {1};
    for (int t = 1; t <= turns; ++t) {
        lens.push_back(root + t * turn);
        stamps.push_back(uint64_t(t + 1));
        while ((int64_t) lens.size() > cap) {
            const size_t v = spread ? spread_victim(lens.data(), stamps.data(), lens.size(), cap, nullptr, nullptr)
                                    : eviction_victim(stamps.data(), stamps.size(), cap);
            lens.erase(lens.begin() + (std::ptrdiff_t) v);
            stamps.erase(stamps.begin() + (std::ptrdiff_t) v);
        }
    }
    const int64_t edit = lens.back() - back;
    int64_t reuse = 0;
    for (int64_t l : lens)
        if (l <= edit) reuse = l;
    return reuse;
}

// the serve loop's chain after a first read that saved `first_read` checkpoints (grid points, then the root at
// root_len - #2228's shape), followed by `turns` appended turns of `turn` tokens, evicting to `cap` the way
// generate.cpp does: an always-present all-false tail array, no pins, and - when `named` - the root found through
// the production helpers (root_length over the conversation's tokens, root_index over the chain).  The conversation
// carries turn token 42 at the system prompt's end and at every appended turn's end.  Returns the lens kept.
std::vector<int64_t> run_chain(std::vector<int64_t> lens, int64_t root_len, int64_t turn, int turns, int64_t cap,
                               bool spread, bool named) {
    std::vector<uint64_t> stamps;
    uint64_t clock = 0;
    for (size_t i = 0; i < lens.size(); ++i) stamps.push_back(++clock);
    std::vector<int64_t> ids((size_t) lens.back() + 1, 7);
    ids[(size_t) root_len] = 42;
    int64_t end = lens.back();
    for (int t = 1; t <= turns; ++t) {
        end += turn;
        ids.resize((size_t) end + 1, 7);
        ids[(size_t) end] = 42;
        lens.push_back(end);
        stamps.push_back(++clock);
        while ((int64_t) lens.size() > cap) {
            const std::unique_ptr<bool[]> tail(new bool[lens.size()]());
            const size_t root = named ? root_index(lens.data(), lens.size(),
                                                   root_length(ids.data(), ids.size(), 42, 2048))
                                      : 0;
            const size_t v = spread ? spread_victim(lens.data(), stamps.data(), lens.size(), cap, tail.get(),
                                                    nullptr, root)
                                    : eviction_victim(stamps.data(), stamps.size(), cap, tail.get(), nullptr, root);
            lens.erase(lens.begin() + (std::ptrdiff_t) v);
            stamps.erase(stamps.begin() + (std::ptrdiff_t) v);
        }
    }
    return lens;
}

bool kept(const std::vector<int64_t>& lens, int64_t len) {
    for (int64_t l : lens)
        if (l == len) return true;
    return false;
}

namespace ref {
// references written from conv_cache.hpp's doc comments, independent of the implementation
size_t evict(const std::vector<uint64_t>& st, int64_t cap, const bool* tail, const bool* pin, size_t root) {
    const size_t n = st.size();
    if (cap < 2 || n < 2) return 0;   // no room for root and leaf: the oldest leaves
    size_t newest = 0;
    for (size_t i = 1; i < n; ++i)
        if (st[i] > st[newest]) newest = i;
    auto pinned = [&](size_t i) { return (pin != nullptr && pin[i]) || i == root; };
    if (tail != nullptr)              // the tail leaves first; never the newest or a pinned item
        for (size_t i = 0; i < n; ++i)
            if (tail[i] && !pinned(i) && i != newest) return i;
    size_t v = n;                     // the least recently used among the unpinned
    for (size_t i = 0; i < n; ++i)
        if (!pinned(i) && (v == n || st[i] < st[v])) v = i;
    if (v != n) return v;
    for (size_t i = 0; i < n; ++i)    // all pinned: the oldest of them, never the root
        if (i != root && (v == n || st[i] < st[v])) v = i;
    return v;
}
size_t spread(const std::vector<int64_t>& lens, const std::vector<uint64_t>& st, int64_t cap, const bool* tail,
              const bool* pin, size_t root) {
    const size_t n = st.size();
    if (n >= 3 && cap >= 2) {
        auto ok = [&](size_t i) { return !(pin != nullptr && pin[i]) && i != root; };
        if (tail != nullptr)
            for (size_t i = 1; i + 1 < n; ++i)
                if (tail[i] && ok(i)) return i;
        size_t v = n;
        double best = 0;
        for (size_t i = 1; i + 1 < n; ++i) {
            if (!ok(i)) continue;
            const double gap = double(lens[i + 1] - lens[i - 1]) / double(lens[n - 1] - lens[i - 1]);
            if (v == n || gap < best) { v = i; best = gap; }
        }
        if (v != n) return v;
    }
    return evict(st, cap, tail, pin, root);
}
}  // namespace ref

int main() {
    std::printf("conv_cache_test\n");
    {   // --prompt-cache-spread: 200 turns of 1,000 tokens after a 9,525-token system prompt, 6 checkpoints
        const int64_t end = 9525 + 200 * 1000;
        const int64_t lru6 = reuse_after_edit(false, 9525, 1000, 200, 6, 6000);
        const int64_t spread6 = reuse_after_edit(true, 9525, 1000, 200, 6, 6000);
        std::printf("  an edit 6K before the end: least recent use reuses %lld, spread %lld of %lld\n", (long long) lru6,
                    (long long) spread6, (long long) end);
        check(lru6 == 9525, "least recent use: an edit 6K back falls through to the root");
        check(end - spread6 < 4 * 6000, "spread: an edit 6K back reads less than 24K again");
        const int64_t spread42 = reuse_after_edit(true, 9525, 1000, 200, 6, 42000);
        check(end - spread42 < 4 * 42000, "spread: an edit 42K back reads less than 168K again");
        check(reuse_after_edit(true, 9525, 1000, 200, 6, 500) == end - 1000, "spread: the turn before is still kept");
    }
    {
        const std::vector<int64_t> lens = {100, 200, 300, 400};
        const std::vector<uint64_t> stamps = {1, 2, 3, 4};
        const bool pinned[] = {false, true, true, false};
        check(spread_victim(lens.data(), stamps.data(), 4, 3, nullptr, pinned) ==
                  eviction_victim(stamps.data(), 4, 3, nullptr, pinned),
              "spread: every candidate pinned - the plain policy");
        check(spread_victim(lens.data(), stamps.data(), 2, 1, nullptr, nullptr) == 0,
              "spread: two items, one slot - the plain policy");
    }
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
    {   // #2228 / #3090: a first read longer than --prompt-cache-every saves grid checkpoints before the root.
        // A 40K system prompt at the defaults (8K chunks, every 16384) leaves {16384, 32768, root 40000}.
        const std::vector<int64_t> first_read = {16384, 32768, 40000};
        check(kept(run_chain(first_read, 40000, 1200, 30, 6, false, true), 40000),
              "a root behind two grid checkpoints survives 30 turns when the caller names it");
        check(!kept(run_chain(first_read, 40000, 1200, 30, 6, false, false), 40000),
              "control: with no root named (root 0) item 0 stays the pin, and that root rotates out");
        // #2228's shape: 32K chunks, the root at 159,184 saved by the periodic rule itself, --prompt-cache 12
        const std::vector<int64_t> big = {32768, 65536, 98304, 131072, 159184};
        check(kept(run_chain(big, 159184, 1500, 30, 12, false, true), 159184),
              "#2228's chain: the 159K root survives 30 turns at --prompt-cache 12");
        // --prompt-cache-spread's gap rule also reaches the root once the newest point is far enough past it
        // (the 40K root above goes at turn 23 of 1,200 tokens, --prompt-cache 6)
        check(kept(run_chain(first_read, 40000, 1200, 40, 6, true, true), 40000),
              "the same under --prompt-cache-spread: the named root is never dropped");
        check(!kept(run_chain(first_read, 40000, 1200, 40, 6, true, false), 40000),
              "control: spread with no root named loses that root too");
        // the lookup drives eviction end to end: grids ahead of the root, evicted to two slots through the
        // production helpers - the root's index shifts 2 -> 1 -> 0 as the grid points leave
        std::vector<int64_t> two = {16384, 32768, 40000, 41200};
        std::vector<uint64_t> tst = {1, 2, 3, 4};
        std::vector<int64_t> tids(41201, 7);
        tids[40000] = 42;
        tids[41200] = 42;
        while ((int64_t) two.size() > 2) {
            const size_t root = root_index(two.data(), two.size(), root_length(tids.data(), tids.size(), 42, 2048));
            const size_t v = eviction_victim(tst.data(), tst.size(), 2, nullptr, nullptr, root);
            two.erase(two.begin() + (std::ptrdiff_t) v);
            tst.erase(tst.begin() + (std::ptrdiff_t) v);
        }
        check(two.size() == 2 && two[0] == 40000 && two[1] == 41200,
              "evicting to two slots through the real lookup keeps {root, newest}");
    }
    {   // the named root in the policy's own terms
        const std::vector<uint64_t> st = {5, 6, 7};
        check(eviction_victim(st.data(), 3, 2, nullptr, nullptr, 1) == 0,
              "a two-slot budget keeps the named root + newest; the grid point at item 0 leaves");
        const std::vector<uint64_t> st2 = {1, 2, 30, 4, 5, 6, 7};
        const bool tail0[7] = {true, false, false, false, false, false, false};
        check(eviction_victim(st2.data(), 7, 6, tail0, nullptr, 2) == 0,
              "a tail checkpoint at item 0 (ahead of the root) is the victim once the root is named");
        const bool all[4] = {true, false, true, true};
        const std::vector<uint64_t> s4 = {9, 1, 3, 5};
        check(eviction_victim(s4.data(), 4, 4, nullptr, all, 1) == 2,
              "everything else pinned: the oldest pin leaves, never the named root");
    }
    {   // root_length / root_index: the serve loop's lookup, from the conversation itself
        auto ids_with = [](std::initializer_list<int64_t> turns, int64_t n) {
            std::vector<int64_t> ids((size_t) n, 7);
            for (int64_t t : turns) ids[(size_t) t] = 42;
            return ids;
        };
        const auto two = ids_with({5000, 9000}, 12000);
        check(root_length(two.data(), two.size(), 42, 2048) == 5000,
              "root_length: the first turn boundary at or past --prompt-cache-root");
        check(root_length(two.data(), two.size(), -1, 2048) == 0, "root_length: no turn token - no root");
        const auto low = ids_with({1000, 9000}, 12000);
        check(root_length(low.data(), low.size(), 42, 2048) == 0,
              "root_length: a first boundary below --prompt-cache-root is not a root");
        const auto one = ids_with({5000}, 12000);
        check(root_length(one.data(), one.size(), 42, 2048) == 0,
              "root_length: a single boundary is the conversation's last turn, not a system prompt's end");
        const std::vector<int64_t> ls = {16384, 32768, 40000, 41200};
        check(root_index(ls.data(), ls.size(), 40000) == 2, "root_index: the chain item at the root's length");
        check(root_index(ls.data(), ls.size(), 39999) == 0, "root_index: no item at that length - item 0, as before");
        check(root_index(ls.data(), ls.size(), 0) == 0, "root_index: no root - item 0, as before");
    }
    {   // spread with a named root on an uneven chain: the smallest relative gap wins, not the earliest index
        const std::vector<int64_t> cl = {100, 10000, 10001, 10002, 20000};
        const std::vector<uint64_t> cst = {1, 2, 3, 4, 5};
        check(spread_victim(cl.data(), cst.data(), 5, 4, nullptr, nullptr, 3) == 2,
              "spread with a named root picks the smallest relative gap");
    }
    {   // the policies against references written from the doc comments: every chain of up to 5 items over three
        // length shapes (even, convex, one huge gap then tiny ones), every root in [0, n), stamps with ties,
        // caps 0..n+1, every tail and pin subset (and absent arrays)
        bool same = true;
        for (size_t n = 2; n <= 5 && same; ++n)
        for (int g = 0; g < 3 && same; ++g) {
            std::vector<uint64_t> st(n);
            std::vector<int64_t> lens(n);
            for (size_t i = 0; i < n; ++i)
                lens[i] = g == 0 ? int64_t(100 * (i + 1))
                        : g == 1 ? int64_t(100 * (i + 1) * (i + 1))
                                 : (i == 0 ? 100 : int64_t(10000 + i));
            size_t stamp_sets = 1;
            for (size_t i = 0; i < n; ++i) stamp_sets *= 3;
            for (size_t s = 0; s < stamp_sets && same; ++s) {
                size_t sv = s;
                for (size_t i = 0; i < n; ++i, sv /= 3) st[i] = 1 + sv % 3;
                for (int64_t cap = 0; cap <= (int64_t) n + 1 && same; ++cap)
                    for (size_t tb = 0; tb <= (1u << n) && same; ++tb)
                        for (size_t pb = 0; pb <= (1u << n) && same; ++pb) {
                            std::unique_ptr<bool[]> tail, pin;
                            if (tb < (1u << n)) {
                                tail.reset(new bool[n]);
                                for (size_t i = 0; i < n; ++i) tail[i] = (tb >> i) & 1;
                            }
                            if (pb < (1u << n)) {
                                pin.reset(new bool[n]);
                                for (size_t i = 0; i < n; ++i) pin[i] = (pb >> i) & 1;
                            }
                            for (size_t root = 0; root < n && same; ++root) {
                                const size_t e = eviction_victim(st.data(), n, cap, tail.get(), pin.get(), root);
                                const size_t sp = spread_victim(lens.data(), st.data(), n, cap, tail.get(),
                                                                pin.get(), root);
                                same = e == ref::evict(st, cap, tail.get(), pin.get(), root) &&
                                       sp == ref::spread(lens, st, cap, tail.get(), pin.get(), root) &&
                                       (root == 0 || (e != root && sp != root)) && e < n && sp < n;
                            }
                        }
            }
        }
        check(same, "eviction and spread match the doc-comment references on every small chain, shape and root");
    }
    {   // root 0 (or out of range) is the plain policy: every chain of up to 5 items, exhaustively
        bool same = true;
        std::vector<uint64_t> st;
        std::vector<int64_t> lens;
        std::vector<bool> bits;
        for (size_t n = 2; n <= 5 && same; ++n) {
            st.assign(n, 0);
            lens.resize(n);
            for (size_t i = 0; i < n; ++i) lens[i] = int64_t(100 * (i + 1));
            size_t stamp_sets = 1;
            for (size_t i = 0; i < n; ++i) stamp_sets *= 3;
            for (size_t s = 0; s < stamp_sets && same; ++s) {
                size_t sv = s;
                for (size_t i = 0; i < n; ++i, sv /= 3) st[i] = 1 + sv % 3;   // ties included
                for (int64_t cap = 0; cap <= (int64_t) n + 1 && same; ++cap)
                    for (size_t tb = 0; tb <= (1u << n) && same; ++tb)         // == 1<<n: no tail array
                        for (size_t pb = 0; pb <= (1u << n) && same; ++pb) {   // == 1<<n: no pin array
                            std::unique_ptr<bool[]> tail, pin;
                            if (tb < (1u << n)) {
                                tail.reset(new bool[n]);
                                for (size_t i = 0; i < n; ++i) tail[i] = (tb >> i) & 1;
                            }
                            if (pb < (1u << n)) {
                                pin.reset(new bool[n]);
                                for (size_t i = 0; i < n; ++i) pin[i] = (pb >> i) & 1;
                            }
                            for (const size_t root : {size_t{0}, n, n + 3}) {
                                same = same &&
                                       eviction_victim(st.data(), n, cap, tail.get(), pin.get(), root) ==
                                           eviction_victim(st.data(), n, cap, tail.get(), pin.get()) &&
                                       spread_victim(lens.data(), st.data(), n, cap, tail.get(), pin.get(), root) ==
                                           spread_victim(lens.data(), st.data(), n, cap, tail.get(), pin.get());
                            }
                        }
            }
        }
        check(same, "root 0 or out of range: the root overloads equal the plain policy on every small chain");
    }
    std::printf(g_fail ? "FAIL\n" : "PASS\n");
    return g_fail ? 1 : 0;
}
