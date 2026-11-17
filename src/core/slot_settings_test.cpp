// src/core/slot_settings_test.cpp - the batch windows' per-slot settings across a resize (header-only, CPU).
//
// The verifier rebuilds its batch layout on every grow/shrink of --parallel-recycle.  The slots that survive
// must come out of it EXACTLY as admission left them: with three requests decoding, growing to four silently
// re-assigned the sampling, the logit bias and the penalty tail of slots 0..2 from the global default.
//
//   1. a fresh layout (init_slots) starts every slot at the default;
//   2. a grow keeps the prefix and gives only the new slots the default;
//   3. a shrink keeps the prefix and hands back the bias arrays of the dropped slots (to be freed);
//   4. a layout built with no request's settings is the default one for every slot (the off path, unchanged).
#include "strata/core/slot_settings.hpp"

#include <cstdio>
#include <vector>

namespace {

struct Sp {   // stands in for strata::kernels::SamplerParams
    float temperature = 0.0f;
    float* logit_bias = nullptr;
    int seed = 0;
};

int g_fail = 0;
void check(bool ok, const char* what) {
    std::printf("  %-84s %s\n", what, ok ? "ok" : "FAIL");
    if (!ok) ++g_fail;
}
bool same(const Sp& a, const Sp& b) {
    return a.temperature == b.temperature && a.logit_bias == b.logit_bias && a.seed == b.seed;
}

}  // namespace

int main() {
    std::printf("slot_settings_test\n");
    using Slots = strata::core::SlotSettings<Sp>;
    Sp dflt;   // greedy, no bias
    dflt.temperature = 0.0f;

    // admission writes slots 0 and 1 (three requests would be slots 0..2)
    Slots live;
    live.reset(3, dflt);
    float bias0 = 0.0f, bias1 = 0.0f, bias2 = 0.0f;
    Sp a = dflt; a.temperature = 0.7f; a.logit_bias = &bias0; a.seed = 11;
    Sp b = dflt; b.temperature = 0.9f; b.logit_bias = &bias1; b.seed = 22;
    Sp c = dflt; c.temperature = 0.5f; c.logit_bias = &bias2; c.seed = 33;
    live.sp[0] = a; live.sp[1] = b; live.sp[2] = c;
    live.bias_dev[0] = &bias0; live.bias_dev[2] = &bias2;   // the per-slot device arrays set_slot_logit_bias owns
    live.tail[0] = {1, 2, 3};
    live.tail[1] = {4, 5};
    const Slots saved = live;   // resize_slots saves the prefix before init_slots rebuilds

    // 1. a fresh layout is the default for every slot
    check(live.size() == 3 && same(live.sp[0], a) && same(live.sp[1], b) && same(live.sp[2], c),
          "admission's three settings are stored as given");

    // 2. GROW 3 -> 4: init_slots rebuilds at four, then the prefix comes back
    live.reset(4, dflt);
    check(same(live.sp[0], dflt) && same(live.sp[3], dflt), "init_slots after the grow: every slot at the default");
    live.restore_prefix(saved);
    check(same(live.sp[0], a) && same(live.sp[1], b) && same(live.sp[2], c),
          "grow 3 -> 4: slots 0..2 keep their sampling, bias and tail");
    check(live.sp[0].logit_bias == &bias0 && live.sp[1].logit_bias == &bias1 && live.tail[0] == std::vector<int32_t>{1, 2, 3} &&
              live.tail[1] == std::vector<int32_t>{4, 5},
          "grow: the bias pointers and the penalty tails are the originals");
    check(same(live.sp[3], dflt), "grow: the new slot starts at the default");

    // 3. SHRINK 3 -> 2: the dropped slot's bias is reported (the caller frees it), the prefix is untouched
    live = saved;
    const std::vector<float*> dropped = live.dropped_biases(2);
    check(dropped.size() == 1 && dropped[0] == &bias2, "shrink 3 -> 2: the dropped slot's bias array is returned");
    check(live.dropped_biases(3).empty(), "shrink that drops nothing returns nothing");
    live.reset(2, dflt);
    live.restore_prefix(saved);
    check(same(live.sp[0], a) && same(live.sp[1], b), "shrink 3 -> 2: slots 0..1 keep their settings");
    check(live.size() == 2 && live.tail.size() == 2 && live.bias_dev.size() == 2, "shrink: the three vectors stay the same length");

    // 4. OFF (a plain --batch start): no per-slot setting was ever written, so the layout is the default one
    Slots plain;
    plain.reset(3, dflt);
    check(plain.size() == 3 && same(plain.sp[0], dflt) && same(plain.sp[1], dflt) && same(plain.sp[2], dflt) &&
              plain.tail[0].empty() && plain.bias_dev[0] == nullptr,
          "no per-slot write: every slot is the default (the off path)");

    std::printf("%s\n", g_fail == 0 ? "slot_settings_test: all passed" : "slot_settings_test: FAILED");
    return g_fail == 0 ? 0 : 1;
}
