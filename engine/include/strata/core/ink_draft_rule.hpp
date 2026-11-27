// The ONE drafter on/off rule of the Inkling engine (every backend states the same two principles; host-only, unit-tested, no GPU dependency).
//
// 1. RESIDENCY FIRST (load time). A default drafter is not loaded when its weights and rings would move the model from full residency (every expert in VRAM) to a hybrid load, or
//    would cut the expert tier noticeably (>= kDraftTierCutLimit of the VRAM left for experts). Measured: a 20 GB RX 7900 XT with SQ-20 fits fully; DFlash 2's 536 MiB made it hybrid:
//    decode 65.7 -> 52.3 tok/s, prefill 1715 -> 794 tok/s, with no drafting active. One line is printed (ink_draft_veto_line); `--draft on` (or STRATA_INK_DRAFTER=dflash2) loads it anyway.
// 2. PLAIN STEPS BY CONTEXT (per step). On a hybrid card (experts not all resident) a draft step costs extra expert reads for every verify row, and the K/V reads grow with the
//    context: from kDraftPlainCtxHybrid tokens on the step is a plain step (no draft), re-checked for every step of every request. On a fully resident card the existing window governor
//    decides alone, except on an architecture whose draft step is emulated (sm < 80, kDraftPlainCtxSlowArch). The exact form keeps its static model: this rule never turns drafting on, it only turns steps plain.
//
// The numbers behind the two constants are in phaseA-tests/rel0223/sm120rows/RESULTS.md (RTX 5060 8 GB, RTX 5070 12 GB, RX 7900 XT, V100 / P100 / 2080 Ti, 3060 / 4060 Ti).
#pragma once

#include <cstdio>
#include <string>

namespace strata::core {

// a drafter that takes at least this share of the VRAM left for experts is "noticeable" (a tier cut)
inline constexpr double kDraftTierCutLimit = 0.10;
// hybrid card: context (tokens) from which a draft step is a plain step.
// Measured on an RTX 5070 12 GB (SQ-20 v2, Ryzen 9 7900X, 3 rounds, DFlash 2 against --draft off): prompts of 33-62 tokens: DFlash 2 -11% on prose, +7% on code, +16% on tool with the
// compact hybrid load (old path: -25%, -10%, -9%); a 4K-token document prompt: 34.5 against 59.2 tok/s (-42%; old path 22.1 against 44.1); 32K: level (30.5 / 30.5). NV-OLDER: DFlash 2 on the
// V100, P100 and 2080 Ti is also below plain on a 4K document (V100 -7% / -10% at 4K / 8K, P100 and 2080 Ti about half). 2048 sits between the short prompts and the 4K document.
inline constexpr int kDraftPlainCtxHybrid = 2048;
// fully resident card on an architecture whose draft step is emulated (sm < 80: no dp4a / tensor-core Q8 GEMVs, 4.9 ms of a 15 ms plain step on the V100): 4K and 8K documents lose there too.
inline constexpr int kDraftPlainCtxSlowArch = 3072;
inline constexpr int kDraftSlowArchBelowCc = 80;

struct InkDraftLoadIn {
    bool full_without = false;        // all experts fit in VRAM without the drafter
    bool full_with = false;           // all experts still fit with the drafter's weights, rings and scratch
    double tier_budget_mib = 0.0;     // VRAM left for experts without the drafter (hybrid cards: the tier's size)
    double drafter_mib = 0.0;         // what the drafter takes (weights + rings + scratch)
    bool below_sm80 = false;          // the card is below compute capability 8.0 (V100, RTX 20, GTX 10, P100): see principle 1b
    bool sm60 = false;                // compute capability 6.0 exactly (Tesla P100): the one card of principle 1b that keeps the default drafter on a hybrid load (owner, 11-27)
    bool user_on = false;             // --draft on / STRATA_INK_DRAFTER=dflash2: the user asked for it by name
    bool user_off = false;            // --draft off
};

struct InkDraftLoadOut {
    bool load = true;
    bool residency_veto = false;      // the veto of principle 1 (the caller prints ink_draft_veto_line)
    bool to_hybrid = false;           // the drafter would move full residency to hybrid
    double tier_cut = 0.0;            // share of the tier the drafter takes (hybrid cards)
};

inline InkDraftLoadOut ink_draft_load_rule(const InkDraftLoadIn& in) {
    InkDraftLoadOut o;
    if (in.user_off) { o.load = false; return o; }
    o.to_hybrid = in.full_without && !in.full_with;
    o.tier_cut = (!in.full_without && in.tier_budget_mib > 0.0) ? in.drafter_mib / in.tier_budget_mib : 0.0;
    if (in.user_on) return o;                                          // asked for by name: never vetoed
    if (o.to_hybrid || o.tier_cut >= kDraftTierCutLimit) { o.load = false; o.residency_veto = true; }
    // 1b. BELOW sm_80: a hybrid load (some experts in host memory, drafter or not) never gets the default drafter, whatever the model. Measured on a 2080 Ti 11 GB, a P100 16 GB and a
    //     GTX 1080 Ti 11 GB (SQ-20 v2: plain decode 23.5 / 25.1 / 12.7 tok/s on prose against 16.3 / 22.1 / 9.2 with DFlash 2; IQ3_S on the 2080 Ti 32 against 23): every draft window reads more
    //     host experts than a plain step. A fully resident card (a V100 32 GB) keeps drafting. phaseA-tests/rel0223/nv-older/RESULTS.md.
    //     Exception, owner decision 11-27 ("Turn drafter on at P100"): the Tesla P100 (cc 6.0) keeps DFlash 2 by default on a hybrid load (SQ-20 v2 code 29.0 tok/s against 21.8 with the
    //     shipped DSpark and 12.8 plain, json 29.4 / 22.1, tool 19.7 / 15.3; prose and chat lower than plain). Principle 1 (the residency veto) and principle 2 (plain steps) still apply to it.
    if (in.below_sm80 && !in.sm60 && !in.full_without) { o.load = false; o.residency_veto = true; }
    return o;
}

// One line. The sentence start and the override are the same on every backend (the HIP line, AMD-RDNA32, reads "[ink-hip] DFlash 2 not loaded: it would move part of the model off the
// card (the experts need X GiB, Y GiB fit with the drafter's Z MiB, W GiB without it): decode and prompt reading get slower than the drafter's gain. `--draft on` loads it anyway.")
inline std::string ink_draft_veto_line(const char* prefix, const char* drafter, double experts_need_gib, double fit_with_gib, double drafter_mib, double fit_without_gib) {
    char b[480];
    std::snprintf(b, sizeof(b),
                  "%s %s not loaded: it would move part of the model off the card (the experts need %.1f GiB, %.1f GiB fit with the drafter's %.0f MiB, %.1f GiB without it): "
                  "decode and prompt reading get slower than the drafter's gain. `--draft on` loads it anyway.",
                  prefix, drafter, experts_need_gib, fit_with_gib, drafter_mib, fit_without_gib);
    return b;
}

// Principle 2. `hybrid`: some experts live in host memory. `ctx_pos`: the position of the step's first token.
// `sm_cc`: the card's compute capability (major * 10 + minor); AMD / other backends pass a value >= kDraftSlowArchBelowCc.
inline bool ink_draft_plain_step(bool hybrid, int ctx_pos, bool user_on, int sm_cc = 90) {
    if (user_on) return false;
    if (hybrid) return ctx_pos >= kDraftPlainCtxHybrid;
    if (sm_cc > 0 && sm_cc < kDraftSlowArchBelowCc) return ctx_pos >= kDraftPlainCtxSlowArch;
    return false;
}

}  // namespace strata::core
