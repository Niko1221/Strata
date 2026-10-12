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
    if (failed == 0) std::printf("split_rules_test: ok\n");
    return failed ? 1 : 0;
}
