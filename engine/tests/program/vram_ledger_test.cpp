// tests/program/vram_ledger_test.cpp - the per-card lending ledger (no GPU), include/strata/program/vram_ledger.hpp.
#include "strata/program/vram_ledger.hpp"

#include <cstdio>
#include <random>
#include <string>
#include <vector>

using namespace strata::program;
using namespace strata::program::vram_ledger;

namespace {
int g_fail = 0;
void check(bool ok, const std::string& what) {
    if (ok) return;
    std::fprintf(stderr, "FAIL: %s\n", what.c_str());
    ++g_fail;
}
constexpr uint64_t MiB = 1ull << 20;
const Lease& lease(const Card& c, int id) { return c.leases()[(size_t) id]; }
}  // namespace

int main() {
    // 1. the tail loan is exactly the prompt path's loan today: split_lend_slots, uniform slots
    {
        const uint64_t blob = 1948 * 1024;   // ~1.86 MiB, the IQ3_S slot on this box
        Card c(7025, {}, blob);
        const uint64_t need = 900 * MiB;
        const int id = c.lend_tail("prompt", need, 128);
        const int64_t n = split_lend_slots(7025, (int64_t) (7025 * blob), nullptr, blob, need);
        check(id >= 0 && lease(c, id).lo == 7025 - n && lease(c, id).hi == 7025, "uniform: the loan = split_lend_slots' tail");
        check(c.expert_slots() == 7025 - n, "uniform: experts keep the rest");
    }
    // 2. ...and with per-slot sizes (a native pack), against random layouts
    {
        std::mt19937 rng(7);
        for (int t = 0; t < 50; ++t) {
            const int64_t slots = 500 + (int64_t) (rng() % 3000);
            std::vector<uint64_t> off(1, 0);
            for (int64_t i = 0; i < slots; ++i) off.push_back(off.back() + (1 + rng() % 4) * MiB);
            Card c(slots, off, 0);
            const uint64_t need = (uint64_t) (rng() % (off.back() / 2)) + 1;
            const int64_t n = split_lend_slots(slots, (int64_t) off.back(), off.data(), 0, need);
            const int id = c.lend_tail("prompt", need, 0);
            if (slots - n < 0 || n == 0) continue;
            check(id >= 0 && lease(c, id).lo == slots - n, "sized: the loan = split_lend_slots' tail (" + std::to_string(t) + ")");
            check(c.bytes_of(slots) - c.bytes_of(lease(c, id).lo) >= need, "sized: the loan covers what was asked");
        }
    }
    // 3. fixed regions stack from the head, only at start, within a budget
    {
        Card c(1000, {}, 2 * MiB);
        const int a = c.reserve_head("slot sessions", 100 * MiB, 200);   // 50 slots
        const int b = c.reserve_head("kv window", 60 * MiB, 200);        // 30 slots after it
        check(a >= 0 && lease(c, a).lo == 0 && lease(c, a).hi == 50, "head 1 at [0, 50)");
        check(b >= 0 && lease(c, b).lo == 50 && lease(c, b).hi == 80 && c.head_end() == 80, "head 2 stacked at [50, 80)");
        check(c.reserve_head("too much", 300 * MiB, 200) < 0, "over the head budget: refused");
        check(c.expert_slots() == 1000, "placed but not occupied: experts still use the head regions");
        c.occupy(a);
        check(c.expert_slots() == 950, "occupied: its slots leave the experts");
        c.vacate(a);
        check(c.expert_slots() == 1000, "vacated: back to the experts");
        const int l = c.lend_tail("prompt", 10 * MiB, 0);
        check(l >= 0 && c.reserve_head("late", 2 * MiB, 200) < 0, "no new fixed region once a loan is out");
        c.release(l);
        check(c.shrink_to(900) && c.reserve_head("late", 2 * MiB, 200) < 0, "no new fixed region after a shrink");
    }
    // 4. the loan keeps clear of the head regions (occupied or not) and leaves keep_min slots for experts
    {
        Card c(300, {}, 1 * MiB);
        c.reserve_head("slot sessions", 50 * MiB, 100);   // [0, 50): the loan may start at 50 + keep_min at the lowest
        check(c.lend_tail("prompt", 122 * MiB, 128) >= 0, "a loan leaving exactly 128 slots past the head: granted");
    }
    {
        Card c(300, {}, 1 * MiB);
        c.reserve_head("slot sessions", 50 * MiB, 100);
        check(c.lend_tail("prompt", 123 * MiB, 128) < 0, "a loan leaving 127 slots past the head: refused");
        check(c.lend_tail("prompt", 251 * MiB, 0) < 0, "a loan into the head region: refused");
    }
    // 5. sub-loans stack inside a loan and go back with it
    {
        Card c(1000, {}, 1 * MiB);
        const int p = c.lend_tail("prompt", 400 * MiB, 128);   // [600, 1000)
        check(p >= 0, "the loan [600, 1000)");
        const int s1 = c.sub(p, "dp stage ring", 76 * MiB);
        const int s2 = c.sub(p, "dp foreign session", 512 * MiB);
        const int s3 = c.sub(p, "dp foreign session", 300 * MiB);
        check(s1 >= 0 && lease(c, s1).lo == 600 && lease(c, s1).hi == 676, "sub 1 at the loan's low end");
        check(s2 < 0, "a sub-loan larger than what is left of the loan: refused");
        check(s3 >= 0 && lease(c, s3).lo == 676 && lease(c, s3).hi == 976, "sub 2 stacked after sub 1");
        check(c.expert_slots() == 600, "sub-loans do not count twice");
        c.release(p);
        check(!lease(c, s1).active && !lease(c, s3).active && c.expert_slots() == 1000, "releasing the loan releases its sub-loans");
        check(c.lend_tail("again", 10 * MiB, 0) >= 0, "a new loan after the release");
    }
    // 6. external room: only between loans, never below the head; the next loan follows the shorter tail
    {
        Card c(1000, {}, 1 * MiB);
        c.reserve_head("slot sessions", 50 * MiB, 100);
        const int p = c.lend_tail("prompt", 100 * MiB, 128);
        check(!c.shrink_to(800), "shrink while a loan is out: refused");
        c.release(p);
        check(c.shrink_to(800) && c.live() == 800, "shrink between requests (the picture reader's room)");
        check(!c.shrink_to(40), "shrink into the head regions: refused");
        const int q = c.lend_tail("prompt", 100 * MiB, 128);
        check(q >= 0 && lease(c, q).hi == 800 && lease(c, q).lo == 700, "the next loan takes the tail of what is left");
        c.release(q);
        check(c.shrink_to(5000) && c.live() == 1000, "grow back, clamped to the full cache");
    }

    // 7. several loans at once: stacked down from the tail, a released one leaves a gap the next fitting loan reuses
    {
        Card c(1000, {}, 1 * MiB);
        const int a = c.lend_tail("slot A prompt", 200 * MiB, 128);   // [800, 1000): today's loan exactly
        const int b = c.lend_tail("slot B prompt", 150 * MiB, 128);   // [650, 800)
        const int k = c.lend_tail("kv growth", 100 * MiB, 128);       // [550, 650)
        check(lease(c, a).lo == 800 && lease(c, a).hi == 1000, "loan 1 at the tail");
        check(lease(c, b).lo == 650 && lease(c, b).hi == 800, "loan 2 stacked below it");
        check(lease(c, k).lo == 550 && lease(c, k).hi == 650, "loan 3 below that");
        check(c.expert_slots() == 550, "three loans out: 450 slots lent");
        c.release(b);                                                   // a gap [650, 800)
        const int d = c.lend_tail("small", 100 * MiB, 128);
        check(d >= 0 && lease(c, d).lo == 700 && lease(c, d).hi == 800, "the next loan that fits reuses the gap, from its top");
        const int e = c.lend_tail("big", 300 * MiB, 128);               // the gap left [650, 700) is too small
        check(e >= 0 && lease(c, e).hi == 550 && lease(c, e).lo == 250, "a loan the gap cannot hold goes below the others");
        check(!c.shrink_to(900), "no shrink while loans are out");
    }
    // 8. a short card: the experts' floor holds across all loans, and `available` says what still fits
    {
        Card c(600, {}, 2 * MiB);                                       // ~1.2 GB of cache on an 8 GB card
        c.reserve_head("slot sessions", 100 * MiB, 100);                // [0, 50)
        const int a = c.lend_tail("prompt", 600 * MiB, 16);             // 300 slots: [300, 600)
        check(a >= 0, "the first loan fits");
        const uint64_t room = c.available(16);                          // 600 - 50 - 16 - 300 = 234 slots
        check(room == 234 * 2 * MiB, "available: what is left above the floor, in bytes");
        check(c.lend_tail("second", room + 1, 16) < 0, "a loan above what is available: refused");
        check(c.lend_tail("second", room, 16) >= 0, "a loan of exactly what is available: granted");
        check(c.available(16) == 0, "nothing left above the floor");
        check(c.expert_slots() == 66, "experts keep the floor (16) plus the head region while it is not occupied (50)");
    }
    // 9. the engine's call (generate.cpp lend()): the engine's prompt loan floor is need + max(lend_keep, head) <= slots (generate.cpp lend:
    //    the head region holds experts while no slot uses it, so it counts toward the floor); asked with
    //    keep_min = max(0, lend_keep - head), the ledger grants exactly the loans the engine makes, at the same slots.
    {
        std::mt19937 rng(7);
        int cases = 0, agree = 0;
        for (int64_t keep : {16, 128})
            for (int64_t head : {0, 1, 50, 127, 128, 129, 530})
                for (int t = 0; t < 40; ++t) {
                    const int64_t slots = 1200;
                    Card c(slots, {}, 1 * MiB);
                    if (head > 0 && c.reserve_head("slot sessions", (uint64_t) head * MiB, slots / 2) < 0) continue;
                    const int64_t need = (int64_t) (rng() % (uint64_t) slots) + 1;
                    const bool engine_ok = need + std::max<int64_t>(keep, head) <= slots;
                    const int id = c.lend_tail("prompt", (uint64_t) need * MiB, std::max<int64_t>(0, keep - head));
                    ++cases;
                    if (engine_ok == (id >= 0) && (id < 0 || (lease(c, id).lo == slots - need && lease(c, id).hi == slots))) ++agree;
                }
        check(cases > 0 && agree == cases, "shadow floor: the engine's and the ledger's answers agree in " + std::to_string(agree) +
                                               " of " + std::to_string(cases) + " cases");
    }

    if (g_fail == 0) std::printf("vram_ledger_test: all checks passed\n");
    return g_fail == 0 ? 0 : 1;
}
