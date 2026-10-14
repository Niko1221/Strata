// The auto layer split's stage-size rule (#1616): no GPU, no model.
#include <cstdio>
#include <cstdlib>
#include <vector>

#include "strata/program/split_rules.hpp"

using namespace strata::program::split_rules;

static int failed = 0;
#define CHECK(x) do { if (!(x)) { std::fprintf(stderr, "FAIL %s:%d: %s\n", __FILE__, __LINE__, #x); ++failed; } } while (0)

int main() {
    // the reporter's 4-GPU pick on a 48-layer model: layers 0-1, 2, 3-19, 20-47 -> cuts 2,3,20
    CHECK(smallest_stage({2, 3, 20}, 48) == 1);
    CHECK(has_tiny_stage({2, 3, 20}, 48));
    // the manual split that fixed it: 4,8,24 is not a cut list of 4 stages; as cuts 4,8,24 -> 4,4,16,24
    CHECK(smallest_stage({4, 8, 24}, 48) == 4);
    CHECK(!has_tiny_stage({4, 8, 24}, 48));
    // identical cards: the balanced split has no tiny stage, so the rule never fires and the pick stays
    CHECK(!has_tiny_stage({12, 24, 36}, 48));
    CHECK(!has_tiny_stage({24}, 48));
    CHECK(has_tiny_stage({47}, 48));
    CHECK(has_tiny_stage({2}, 48));
    CHECK(!has_tiny_stage({3}, 48));
    CHECK(smallest_stage({}, 48) == 48);
    // the restriction needs the layers for it
    CHECK(restriction_possible(48, 4));
    CHECK(restriction_possible(12, 4));
    CHECK(!restriction_possible(11, 4));
    CHECK(!restriction_possible(48, 1));

    // #1440: two equal cards, 48 layers x 512 experts of ~2 MiB (IQ3_S-like).  Card 0 has more room than card 1 (the
    // last stage also carries the drafter).  Every K whose two caches hold all their pairs predicts the same time.
    {
        const int64_t L = 48, E = 512, blob = 2 << 20, MiB = 1 << 20;
        const std::vector<int64_t> room{29000 * MiB, 26200 * MiB};
        std::vector<int64_t> need(L + 1, 0);
        for (int64_t l = 0; l < L; ++l) need[(size_t) l + 1] = need[(size_t) l] + E * blob;
        auto session = [&](int64_t lb, int64_t le) { return (le - lb) * 40 * MiB; };
        std::vector<Placement> c;
        for (int64_t k = 2; k < L; ++k) {
            Placement p;
            p.cuts = {k};
            p.min_spare = min_spare(p.cuts, L, room, need, session);
            // the miss term: the pairs a stage's cache cannot hold (each costs the same here)
            double missing = 0;
            for (int i = 0; i < 2; ++i) {
                const int64_t lb = i == 0 ? 0 : k, le = i == 0 ? k : L;
                const int64_t fit = (room[(size_t) i] - session(lb, le)) / blob;
                const int64_t want = (le - lb) * E;
                if (fit < want) missing += (double) (want - (fit > 0 ? fit : 0));
            }
            p.ms = 48 * 0.5 + 190.0 * missing / (double) (L * E);
            c.push_back(p);
        }
        const size_t plain = fastest(c), head = pick_with_headroom(c, 0.002);
        // the plain pick is the first full placement, and it leaves card 1 (almost) nothing
        CHECK(c[plain].ms == c[head].ms);
        CHECK(c[plain].cuts[0] < c[head].cuts[0]);
        CHECK(c[plain].min_spare < 1024 * MiB);
        CHECK(c[head].min_spare > 1536 * MiB);
        // the pick is the most balanced full placement: no other full one keeps more on its tightest card
        for (const Placement& p : c)
            if (p.ms == c[plain].ms) CHECK(p.min_spare <= c[head].min_spare);
        std::printf("split_rules_test: #1440 plain K=%lld (spare %lld MiB), headroom K=%lld (spare %lld MiB)\n",
                    (long long) c[plain].cuts[0], (long long) (c[plain].min_spare / MiB), (long long) c[head].cuts[0],
                    (long long) (c[head].min_spare / MiB));
        // a slower placement is never taken for its headroom beyond the tolerance
        std::vector<Placement> two{{{10}, 100.0, 0}, {{20}, 101.0, 5000 * MiB}};
        CHECK(pick_with_headroom(two, 0.002) == 0);
        CHECK(pick_with_headroom(two, 0.02) == 1);
        // exact ties with equal spare keep the first (the old pick)
        std::vector<Placement> same{{{10}, 100.0, 7}, {{20}, 100.0, 7}};
        CHECK(pick_with_headroom(same, 0.0) == 0);
        CHECK(pick_with_headroom({}, 0.1) == 0);
    }
    // #2007: the miss scale per link
    {
        // the case 0.1.41's scale was made for: CUDA1 on a physical x1 link beside an x16 one
        const std::vector<double> x1 = link_miss_scale({24.0, 1.6});
        CHECK(x1[0] == 1.0 && x1[1] > 14.9 && x1[1] < 15.1);
        // two equal cards on gen3 x16 (2x Tesla T10, P100s): the fitted formula, whatever the link speed
        const std::vector<double> eq = link_miss_scale({12.0, 12.0});
        CHECK(eq[0] == 1.0 && eq[1] == 1.0);
        // a moderate gap (x8 beside x16, or the probe's noise): no scale
        const std::vector<double> mod = link_miss_scale({11.4, 5.9});
        CHECK(mod[0] == 1.0 && mod[1] == 1.0);
        // a 4x gap and more scales by the ratio, capped at 64; unprobed stages stay 1
        const std::vector<double> big = link_miss_scale({26.0, 6.0, 0.2, -1.0});
        CHECK(big[0] == 1.0 && big[1] > 4.3 && big[1] < 4.4 && big[2] == 64.0 && big[3] == 1.0);
        CHECK(link_miss_scale({-1.0, -1.0})[1] == 1.0);
    }
    if (failed == 0) std::printf("split_rules_test: ok\n");
    return failed ? 1 : 0;
}
