// src/program/layer_split_test.cpp - the layer split's preparation: what the host arena pins, and the prompt path's
// buffers lent by every card's expert cache.
//
//   1. the arena is pinned whole unless WDDM maps it into several contexts (then 8 GiB, as measured on the 5080 +
//      3090 rig); one GPU or Linux: no cap;
//   2. the slots a cache lends for `need` bytes are taken from its end: uniform slots round up, sized slots stop at
//      the first run from the end that holds `need`;
//   3. the bytes lent from a first slot to the end, for both layouts;
//   4. one card: --prefill auto takes the largest chunk whose buffers leave 128 slots and stay under the lend share;
//      a fixed chunk is halved until it fits; nothing fits: chunk 0;
//   5. several cards: ONE chunk for all of them (the hand-off rows are that size), the largest every card can lend -
//      a card with a small cache decides it - and each card lends its own slot count;
//   6. a card that cannot lend even the smallest chunk: no plan (every card allocates its own buffers).
#include "strata/program/layer_split.hpp"

#include <cstdint>
#include <cstdio>
#include <vector>

namespace ls = strata::program::layer_split;

namespace {
int g_fail = 0;
void check(bool ok, const char* what) {
    std::printf("  %-74s %s\n", what, ok ? "ok" : "FAIL");
    if (!ok) ++g_fail;
}
constexpr uint64_t MiB = 1ull << 20, GiB = 1ull << 30;
// a prompt path's device bytes for a chunk: 200 MiB fixed + 0.25 MiB a token (the shape of Prefill::bytes_needed)
uint64_t need(int64_t chunk) { return 200 * MiB + (uint64_t) chunk * (MiB / 4); }
}  // namespace

int main() {
    std::printf("layer_split_test\n");
    {
        check(ls::arena_pin_cap(false, true) == 0, "one context, WDDM: the whole arena");
        check(ls::arena_pin_cap(false, false) == 0, "one context, Linux: the whole arena");
        check(ls::arena_pin_cap(true, true) == 8 * GiB, "several contexts under WDDM: 8 GiB");
        check(ls::arena_pin_cap(true, false) == 0, "several contexts, Linux: the whole arena");
    }
    {
        const ls::CacheSlots u{1000, 1000 * (int64_t) MiB, nullptr, (int64_t) MiB};   // uniform 1 MiB slots
        check(ls::slots_to_lend(u, 10 * MiB) == 10, "uniform slots: an exact multiple");
        check(ls::slots_to_lend(u, 10 * MiB + 1) == 11, "uniform slots: rounded up");
        check(ls::lent_bytes(u, 990) == 10 * MiB, "uniform slots: bytes from slot 990 to the end");
        // sized slots: 1 MiB each, the last four 2 MiB (offsets: each slot's start, then the end - as ExpertCache)
        std::vector<uint64_t> off;
        uint64_t at = 0;
        for (int i = 0; i < 100; ++i) { off.push_back(at); at += i >= 96 ? 2 * MiB : MiB; }
        off.push_back(at);
        const ls::CacheSlots s{100, (int64_t) at, off.data(), 2 * (int64_t) MiB};
        check(ls::slots_to_lend(s, 8 * MiB) == 4, "sized slots: the four 2 MiB slots hold 8 MiB");
        check(ls::slots_to_lend(s, 8 * MiB + 1) == 5, "sized slots: one more byte takes a fifth slot");
        check(ls::lent_bytes(s, 95) == 9 * MiB, "sized slots: bytes from slot 95 to the end");
        check(ls::slots_to_lend(s, 1000 * MiB) == 100, "sized slots: more than the cache holds: all of it");
    }
    {
        const ls::CacheSlots big{4000, 4000 * (int64_t) MiB, nullptr, (int64_t) MiB};
        const auto p = ls::plan_lend({big}, [](size_t, int64_t c) { return need(c); }, 8192, true, 90);
        check(p.chunk == 8192 && p.slots.size() == 1 && p.slots[0] == 2248, "one card, auto: 8192 (2248 of 4000 slots)");
        const ls::CacheSlots mid{3000, 3000 * (int64_t) MiB, nullptr, (int64_t) MiB};
        const auto q = ls::plan_lend({mid}, [](size_t, int64_t c) { return need(c); }, 8192, true, 60);
        check(q.chunk == 6144 && q.slots[0] == 1736, "one card, auto, 60% share: 6144 (1736 slots)");
        const auto f = ls::plan_lend({mid}, [](size_t, int64_t c) { return need(c); }, 16384, false, 90);
        check(f.chunk == 8192, "one card, fixed 16384: halved to 8192 (leaves 128 slots)");
        const ls::CacheSlots tiny{300, 300 * (int64_t) MiB, nullptr, (int64_t) MiB};
        const auto n = ls::plan_lend({tiny}, [](size_t, int64_t c) { return need(c); }, 8192, true, 90);
        check(n.chunk == 0 && n.slots.empty(), "one card, too small for any chunk: no plan");
    }
    {
        const ls::CacheSlots a{4000, 4000 * (int64_t) MiB, nullptr, (int64_t) MiB};
        const ls::CacheSlots b{2000, 2000 * (int64_t) MiB, nullptr, (int64_t) MiB};
        const auto p = ls::plan_lend({a, b}, [](size_t, int64_t c) { return need(c); }, 8192, true, 90);
        check(p.chunk == 6144 && p.slots.size() == 2, "two cards, auto: the smaller cache decides (6144, not 8192)");
        check(p.slots.size() == 2 && p.slots[0] == 1736 && p.slots[1] == 1736, "two cards, auto: each lends its slots");
        // the later card's prompt path needs more (its own buffers differ): per-stage need
        const auto q = ls::plan_lend({a, a}, [](size_t st, int64_t c) { return need(c) + (st == 1 ? 1024 * MiB : 0); },
                                     8192, true, 90);
        check(q.chunk == 8192 && q.slots.size() == 2 && q.slots[0] == 2248 && q.slots[1] == 3272,
              "two cards, per-stage need: slots differ");
        const ls::CacheSlots tiny{300, 300 * (int64_t) MiB, nullptr, (int64_t) MiB};
        const auto n = ls::plan_lend({a, tiny}, [](size_t, int64_t c) { return need(c); }, 8192, true, 90);
        check(n.chunk == 0 && n.slots.empty(), "two cards, one too small for any chunk: no plan");
        const auto e = ls::plan_lend({}, [](size_t, int64_t c) { return need(c); }, 8192, true, 90);
        check(e.chunk == 0 && e.slots.empty(), "no cards: no plan");
    }
    std::printf(g_fail ? "layer_split_test: %d FAILED\n" : "layer_split_test: all passed\n", g_fail);
    return g_fail ? 1 : 0;
}
