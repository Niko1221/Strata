// expert_pack_test.cpp - the resident-set determinism gate: an explicit --expert-cache N must pack exactly the
// first N ranked pairs (the same N on two engines is the same resident set), and the auto sizing must stay
// capped by what VRAM is free.  The free-VRAM-capped packing that once ran for both modes built 5621 vs 5616
// slots from one pinned value and split an interim request's answer (bench/results/prefill-preempt 2026-10-03).
#include "strata/core/expert_pack.hpp"
#include <cstdio>
#include <cstdlib>
#include <cstdint>
#include <vector>

using namespace strata::core;
static int checks = 0;
static void check(bool ok, const char* label) {
    ++checks;
    if (!ok) { std::fprintf(stderr, "FAIL: %s\n", label); std::exit(1); }
}
static uint64_t rounded_bytes(const std::vector<int64_t>& slots) {
    uint64_t sum = 0;
    for (const int64_t s : slots) sum += (uint64_t) (s + 255) / 256 * 256;
    return sum;
}

int main() {
    // a tiny profile: ranked pairs of differing blob sizes; the 256-byte rounded sizes are
    // 1024, 2048, 512, 4096, 512, 6144, 768, 8192
    const std::vector<int64_t> profile = {1000, 2000, 300, 4000, 500, 6000, 700, 8000};

    // explicit N: exactly the first N pairs, whatever the cap argument says, in rank order
    const auto five = pack_expert_slot_bytes(profile, UINT64_MAX, 5);
    check(five.size() == 5, "explicit N packs exactly N pairs");
    check(five == (std::vector<int64_t>{1000, 2000, 300, 4000, 500}), "explicit N keeps the profile's rank order");

    // the pack is a pure function of N: same N twice, same set (this is what pins a comparison's geometry)
    check(pack_expert_slot_bytes(profile, UINT64_MAX, 5) == five, "same N packs the same set");
    check(pack_expert_slot_bytes(profile, UINT64_MAX, 100).size() == profile.size(),
          "N beyond the profile packs the whole profile");

    // auto: capped by the byte cap even when more pairs would fit the count (1024+2048=3072; the next rounded
    // blob 512 does not fit 3500-3072)
    check(pack_expert_slot_bytes(profile, 3500, profile.size()) == (std::vector<int64_t>{1000, 2000}),
          "auto stops at the byte cap");
    check(rounded_bytes(pack_expert_slot_bytes(profile, 3500, profile.size())) <= 3500,
          "auto never exceeds the cap");

    // a pair that does not fit whole is skipped, not cut: 3584 takes the 300-byte blob, 3583 does not
    check(pack_expert_slot_bytes(profile, 3583, profile.size()).size() == 2,
          "a pair that would overflow the cap is skipped, not cut");
    check(pack_expert_slot_bytes(profile, 3584, profile.size()).size() == 3, "a pair that fits whole is taken");
    check(rounded_bytes(pack_expert_slot_bytes(profile, 3584, profile.size())) == 3584,
          "the cap accounts for 256-byte rounding");

    // degenerate inputs stay safe
    check(pack_expert_slot_bytes({}, UINT64_MAX, 5).empty(), "an empty profile packs nothing");
    check(pack_expert_slot_bytes(profile, 0, 5).empty(), "a zero cap packs nothing");
    check(pack_expert_slot_bytes(profile, UINT64_MAX, 0).empty(), "a zero count packs nothing");
    const std::vector<int64_t> bad = {1000, -5, 300};
    check(pack_expert_slot_bytes(bad, UINT64_MAX, 3).size() == 1, "a negative size stops the pack");

    // the repeat-3 arithmetic, miniaturised: a uniform budget of 2 largest-blob slots is 16000 bytes, free room
    // 12000 - the auto pack takes what fits the room (5 pairs here); an explicit 3 ignores both numbers and
    // takes 3 pairs, so two engines pinned to the same N cannot drift apart
    check(pack_expert_slot_bytes(profile, 12000, profile.size()).size() == 5,
          "auto packs free room, not the uniform budget");
    check(pack_expert_slot_bytes(profile, 999999, 3).size() == 3, "explicit N is free-room independent");
    std::printf("%d checks passed\n", checks);
}
