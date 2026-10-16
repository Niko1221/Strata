#include "strata/program/slot_checkpoints.hpp"

#include <cstdio>

namespace {
struct Ck {
    std::vector<int32_t> ids;
    int tag = 0;
};
std::vector<int32_t> run(int32_t from, int32_t n) {
    std::vector<int32_t> v;
    for (int32_t i = 0; i < n; ++i) v.push_back(from + i);
    return v;
}
} // namespace

int main() {
    using strata::program::branch_checkpoint_boundary;
    using strata::program::deepest_prefix_checkpoints;
    using strata::program::shared_prefix;
    int checks = 0;
    auto expect = [&](bool ok, const char* what) {
        ++checks;
        if (!ok) std::fprintf(stderr, "slot_checkpoints_test: %s\n", what);
        return ok;
    };
    const std::vector<int32_t> held = run(0, 50000);
    std::vector<int32_t> other = run(0, 20000);
    other[19000] = -1;
    // root 8192 (twice: the first counts), periodic 16384 and 32768, turn 49990, the whole conversation (not a strict
    // start), another conversation's checkpoint
    const std::vector<Ck> cks = {{run(0, 32768), 1}, {run(0, 8192), 2}, {run(0, 49990), 3}, {run(0, 8192), 4},
                                 {run(0, 16384), 5}, {held, 6}, {other, 7}};
    auto tags = [&](std::size_t keep) {
        std::vector<int> t;
        for (const Ck* c : deepest_prefix_checkpoints(cks, held, keep)) t.push_back(c->tag);
        return t;
    };
    bool ok = expect(tags(1) == std::vector<int>{3}, "keep 1 = the single deepest (as before)") &&
              expect(tags(2) == std::vector<int>{1, 3}, "keep 2, shallowest first") &&
              expect(tags(4) == std::vector<int>{2, 5, 1, 3}, "keep 4: one per length, the first of equal ones") &&
              expect(tags(9) == std::vector<int>{2, 5, 1, 3}, "keep more than there are") &&
              expect(tags(0).empty(), "keep 0") &&
              expect(deepest_prefix_checkpoints(cks, run(0, 100), 3).empty(), "none is a start of a short history") &&
              expect(deepest_prefix_checkpoints(std::vector<Ck>{}, held, 2).empty(), "no checkpoints");
    std::vector<int64_t> prompt(held.begin(), held.end());
    prompt[49940] = -5;
    ok = ok && expect(shared_prefix(prompt, held) == 49940, "shared prefix at the rewrite") &&
         expect(shared_prefix(std::vector<int64_t>(prompt.begin(), prompt.begin() + 100), held) == 100, "prompt ends") &&
         expect(shared_prefix(prompt, run(0, 10)) == 10, "held ends") &&
         expect(shared_prefix({}, held) == 0, "empty prompt");
    // a classifier query: resume 32768, a sibling shares 49940, the turn at 50300, 50310 tokens
    ok = ok && expect(branch_checkpoint_boundary(49940, 32768, 50300, 50310, 2048) == 49940, "a deep branch") &&
         expect(branch_checkpoint_boundary(49940, 48000, 50300, 50310, 2048) == -1, "too close to the read start") &&
         expect(branch_checkpoint_boundary(49940, 47892, 50300, 50310, 2048) == 49940, "exactly the margin") &&
         expect(branch_checkpoint_boundary(50300, 0, 50300, 50310, 2048) == -1, "at the turn: taken anyway") &&
         expect(branch_checkpoint_boundary(50305, 0, 50300, 50310, 2048) == -1, "past the turn") &&
         expect(branch_checkpoint_boundary(50308, 0, -1, 50310, 2048) == 50308, "no turn: before the last token") &&
         expect(branch_checkpoint_boundary(50309, 0, -1, 50310, 2048) == -1, "no turn: the last token") &&
         expect(branch_checkpoint_boundary(49940, 0, 50300, 50310, 0) == -1, "margin 0: off") &&
         expect(branch_checkpoint_boundary(0, 0, 50300, 50310, 1) == -1, "nothing shared");
    if (!ok) return 1;
    std::printf("slot_checkpoints_test OK (%d checks)\n", checks);
}
