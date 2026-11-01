// tests/core/batch_dflash2_depth_test.cpp - adaptive DFlash2 batch proposal depth.
// Header-only CPU test: environment parsing, acceptance EMA, depth gate, and draft-probability trimming.
#include "strata/core/batch_dflash2_depth.hpp"

#include <cstdio>

using namespace strata::core;

namespace {
int failures = 0;
void check(bool ok, const char* what) {
    if (ok) return;
    ++failures;
    std::fprintf(stderr, "FAIL: %s\n", what);
}
}  // namespace

int main() {
    check(!batch_draft_depth_config(nullptr, nullptr, 7).on(), "unset depth is off");
    check(!batch_draft_depth_config("", "", 7).on(), "empty depth is off");
    check(!batch_draft_depth_config("1", nullptr, 7).on(), "depth 1 is off");
    check(!batch_draft_depth_config("junk", nullptr, 7).on(), "unparsable depth is off");
    check(batch_draft_depth_config("4", nullptr, 7).max_depth == 4, "depth parses");
    check(batch_draft_depth_config("99", nullptr, 7).max_depth == 7, "depth clamps to cap");
    check(batch_draft_depth_config("-3", nullptr, 7).max_depth == 1, "negative depth is off");
    check(batch_draft_depth_config("4", "0.7", 7).floor == 0.7f, "floor parses");
    check(batch_draft_depth_config("4", "0", 7).floor == 0.5f, "floor zero is rejected");
    check(batch_draft_depth_config("4", "1.5", 7).floor == 0.5f, "floor above one is rejected");

    check(batch_draft_depth_ema(0.5f, true, 0.25f) == 0.625f, "accepted EMA rises");
    check(batch_draft_depth_ema(0.5f, false, 0.25f) == 0.375f, "rejected EMA falls");
    const BatchDraftDepthConfig cfg = batch_draft_depth_config("3", "0.5", 6);
    check(cfg.on() && cfg.max_depth == 3 && cfg.floor == 0.5f, "DFlash2 depth config");
    check(batch_draft_depth_for(0.5f, cfg) == 1, "new slot remains at depth one");
    check(batch_draft_depth_for(0.75f, cfg) == 2, "EMA gate admits depth two");
    check(batch_draft_depth_for(0.80f, cfg) == 3, "EMA gate admits depth three");
    check(batch_draft_depth_for(0.95f, cfg) == 3, "depth is capped");
    check(batch_draft_depth_for(0.0f, cfg) == 1, "depth never drops below one");

    const float probabilities[] = {0.9f, 0.8f, 0.6f, 0.9f};
    check(batch_draft_depth_trim(probabilities, 4, 0.5f) == 2, "cumulative probability trims the tail");
    const float low_first[] = {0.1f, 0.99f, 0.99f};
    check(batch_draft_depth_trim(low_first, 3, 0.5f) == 1, "first proposal row is always kept");
    check(batch_draft_depth_trim(nullptr, 3, 0.5f) == 3, "missing probabilities keep all proposals");
    check(batch_draft_depth_trim(probabilities, 0, 0.5f) == 0, "empty draft keeps no rows");

    if (failures != 0) {
        std::fprintf(stderr, "batch_dflash2_depth_test: %d failure(s)\n", failures);
        return 1;
    }
    std::printf("batch_dflash2_depth_test: ok\n");
    return 0;
}
