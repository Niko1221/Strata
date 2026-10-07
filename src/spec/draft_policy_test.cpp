// src/spec/draft_policy_test.cpp - DraftPolicy: when does a lookup window beat the MTP's?
//
// Simulated rounds with the costs measured on the RTX 5070 (window-cost: ~10 ms more per token) check that
//   1. with no lookup proposal the MTP window is kept;
//   2. lookup drafts that are mostly rejected stop being taken (their bucket's rate falls);
//   3. lookup drafts that are always accepted are taken, and the window grows with them;
//   4. match-length buckets learn separately (short matches failing does not stop long ones);
//   5. the policy never proposes a window beyond its cap;
//   6. a transient expensive window is retried and relearned, with bounded, confidence-gated probes.
#include "strata/spec/draft_policy.hpp"

#include <cmath>
#include <cstdio>

using strata::spec::DraftPolicy;

namespace {
int g_fail = 0;
void check(bool ok, const char* what) {
    std::printf("  %-66s %s\n", what, ok ? "ok" : "FAIL");
    if (!ok) ++g_fail;
}
double cost(int t) { return 19.0 + 10.5 * (t - 1); }   // ms per round, measured shape

// Synthetic times isolate policy recovery; they are not a GPU throughput benchmark.
DraftPolicy slow_start(bool chained, bool smaller_lookup = false) {
    DraftPolicy p(6);
    for (int t : {2, 3, 5})
        for (int i = 0; i < 3; ++i)
            p.observe(false, t, t - 1, 0, smaller_lookup && t == 5 ? 42.0 : 1000.0);
    for (int i = 0; i < 3; ++i) p.observe(false, 4, 3, 0, 40.0);
    for (int i = 0; i < 3; ++i) {
        if (chained) p.observe_chain(4, 2, 5, 40, 1000.0);
        else p.observe(true, 6, 5, 40, 1000.0);
    }
    return p;
}

void check_recovery(bool chained, bool smaller_lookup, int t_mtp = 4) {
    DraftPolicy p = slow_start(chained, smaller_lookup);
    const int initial = chained ? 4 + p.chain(4, 1.0, 2, 40) : p.choose(t_mtp, 5, 40).t;
    check(initial == (smaller_lookup ? 5 : 4), "slow start: the expensive full window is initially avoided");
    int full = 0;
    bool settled = true;
    for (int i = 0; i < 256; ++i) {
        const auto pick = p.choose(t_mtp, 5, 40);
        const int k = chained ? p.chain(4, 1.0, 2, 40) : 0;
        const int t = chained ? 4 + k : pick.t;
        if (t == 6) ++full;
        if (i >= 128 && t != 6) settled = false;
        const double ms = t == 4 ? 40.0 : 40.0 + 2.0 * (t - 4);
        if (chained && k > 0) p.observe_chain(4, k, t - 1, 40, ms);
        else p.observe(chained ? false : pick.lookup, t, t - 1, 40, ms);
    }
    std::printf("    %s, MTP=%d, %s: %d / 256 full windows after the slow start\n",
                chained ? "chain" : "lookup", t_mtp, smaller_lookup ? "smaller lookup wins" : "MTP wins", full);
    check(settled, "stale cost: recovery settles on the now-cheap full window");
    check(p.cost_ms(6) < 45.0, "stale cost: the new measurement replaces the transient cost");
}

void check_expensive_probes(bool chained) {
    DraftPolicy p = slow_start(chained);
    int probes = 0, last_probe = -65;
    bool spaced = true;
    for (int i = 0; i < 256; ++i) {
        const auto pick = p.choose(4, 5, 40);
        const int k = chained ? p.chain(4, 1.0, 2, 40) : 0;
        const int t = chained ? 4 + k : pick.t;
        const bool full = t == 6;
        probes += full;
        if (full) {
            spaced &= i - last_probe >= 65;
            last_probe = i;
        }
        if (chained && k > 0) p.observe_chain(4, k, t - 1, 40, 1000.0);
        else p.observe(chained ? false : pick.lookup, t, t - 1, 40, full ? 1000.0 : 40.0);
    }
    check(probes > 0 && probes <= 4 && spaced,
          "persistently expensive: at most 4 isolated re-probes in 256 rounds");
    check(p.cost_ms(6) == 1000.0, "persistently expensive: re-probes keep the measured cost");
}
}  // namespace

int main() {
    std::printf("draft_policy_test\n");
    {
        DraftPolicy p(6);
        for (int i = 0; i < 50; ++i) p.observe(false, 4, 2, 0, cost(4));   // MTP windows of 4: 3 tokens each
        const DraftPolicy::Pick k = p.choose(4, 0, 0);
        check(!k.lookup && k.t == 4, "no proposal: the MTP window");
    }
    {
        DraftPolicy p(6);
        for (int i = 0; i < 50; ++i) p.observe(false, 4, 2, 0, cost(4));
        for (int t = 2; t <= 6; ++t) p.observe(false, t, 0, 0, cost(t));
        for (int i = 0; i < 40; ++i) p.observe(true, 6, 0, 4, cost(6));      // short matches, all rejected
        check(p.lookup_rate(4) < 0.15, "rejected short-match drafts: their rate falls below 0.15");
        check(!p.choose(4, 5, 4).lookup, "rejected short-match drafts: no longer taken");
        for (int i = 0; i < 40; ++i) p.observe(true, 6, 5, 30, cost(6));     // long matches, all accepted
        check(p.lookup_rate(30) > 0.9, "accepted long-match drafts: their rate rises above 0.9");
        const DraftPolicy::Pick k = p.choose(4, 5, 30);
        check(k.lookup && k.t == 6, "accepted long matches: the full lookup window is taken");
        check(!p.choose(4, 5, 4).lookup, "buckets are separate: short matches still not taken");
        check(p.choose(4, 20, 30).t <= 6, "never beyond the window cap");
    }
    {
        DraftPolicy p(8);
        for (int i = 0; i < 50; ++i) p.observe(false, 3, 2, 0, cost(3));      // a very good MTP: 3 of 3 tokens
        for (int t = 2; t <= 8; ++t) p.observe(false, t, t - 1, 0, cost(t));
        for (int i = 0; i < 40; ++i) p.observe(true, 4, 2, 8, cost(4));       // lookup at q ~ 0.67
        check(!p.choose(3, 7, 8).lookup, "a mediocre lookup does not replace a strong MTP window");
    }
    {
        DraftPolicy p(6);
        for (int i = 0; i < 50; ++i) p.observe(false, 4, 3, 0, cost(4));      // a near-perfect MTP, only size 4 seen
        const DraftPolicy::Pick k = p.choose(4, 5, 40);
        check(k.lookup && k.t == 6, "an unmeasured size is probed for a confident lookup");
        for (int i = 0; i < 3; ++i) p.observe(true, 6, 5, 40, 3.0 * cost(6));   // it turns out very expensive
        check(!p.choose(4, 5, 40).lookup, "after the probes, the measured cost decides");
    }
    {
        // --lookup-chain: rows cheap (UMA-like) and the chain always accepted -> chained; rows dear or the chain
        // always rejected -> not
        auto cheap = [](int t) { return 40.0 + 2.0 * (t - 1); };
        DraftPolicy p(8);
        for (int t = 2; t <= 8; ++t)
            for (int i = 0; i < 5; ++i) p.observe(false, t, 1, 0, cheap(t));
        for (int i = 0; i < 30; ++i) p.observe(false, 4, 2, 0, cheap(4));
        for (int i = 0; i < 30; ++i) p.observe_chain(4, 3, 6, 12, cheap(7));    // every chained token accepted
        check(p.chain(4, 0.9, 3, 12) == 3, "chain: cheap rows, always accepted -> all 3 chained");
        DraftPolicy q(8);
        for (int t = 2; t <= 8; ++t)
            for (int i = 0; i < 5; ++i) q.observe(false, t, 1, 0, cost(t));
        for (int i = 0; i < 30; ++i) q.observe_chain(4, 3, 3, 4, cost(7));       // reached, never accepted
        check(q.chain(4, 0.9, 3, 4) == 0, "chain: dear rows, never accepted -> none");
        check(q.chain(4, 0.9, 0, 4) == 0 && q.chain(8, 1.0, 3, 40) == 0, "chain: nothing proposed / no room -> none");
    }
    for (bool chained : {false, true}) {
        check_recovery(chained, false);
        check_recovery(chained, true);   // a profitable smaller lookup must not starve the full window
        check_expensive_probes(chained);
        DraftPolicy p = slow_start(chained);
        for (int i = 0; i < 256; ++i) p.observe(false, 4, 3, 0, 40.0);
        if (chained) p.observe_chain(4, 2, 5, 40, 0.0);
        else p.observe(true, 6, 5, 40, 0.0);
        const int t = chained ? 4 + p.chain(4, 1.0, 2, 40) : p.choose(4, 5, 40).t;
        check(t == 6 && p.cost_ms(6) == 1000.0, "untimed observation does not refresh a stale cost");
        if (chained) p.observe_chain(4, 2, 5, 40, 44.0);
        else p.observe(true, 6, 5, 40, 44.0);
        check(p.cost_ms(6) == 44.0, "first fresh observation replaces the stale cost exactly");
        if (chained) p.observe_chain(4, 2, 5, 40, 54.0);
        else p.observe(true, 6, 5, 40, 54.0);
        check(std::abs(p.cost_ms(6) - 45.0) < 1e-9, "subsequent fresh observation resumes the ordinary cost EMA");
    }
    check_recovery(false, true, 6);   // the full lookup need not be larger than the requested MTP window
    {
        DraftPolicy p = slow_start(false);
        for (int i = 0; i < 256; ++i) {
            p.choose(4, 5, 40);   // decisions without a timed observation do not age costs
            p.observe(false, 4, 3, 0, i % 2 ? 0.0 : -1.0);
        }
        check(!p.choose(4, 5, 40).lookup, "untimed rounds and repeated decisions do not trigger a re-probe");
        for (int i = 0; i < 256; ++i) p.observe(false, 4, 3, 0, 40.0);
        check(!p.choose(4, 0, 40).lookup, "stale cost: no proposal still keeps the MTP window");
        check(!p.choose(4, 5, 4).lookup, "stale cost: low-confidence lookup is not forced");
        check(p.chain(4, 0.0, 2, 40) == 0, "stale cost: low-probability MTP does not force a chain");
        check(p.chain(4, 1.0, 2, 4) == 0, "stale cost: low-confidence chain is not forced");
        check(p.chain(4, 1.0, 0, 40) == 0 && p.chain(6, 1.0, 2, 40) == 0,
              "stale cost: no chain proposal / no room still means none");
    }
    std::printf(g_fail ? "FAIL\n" : "PASS\n");
    return g_fail ? 1 : 0;
}
