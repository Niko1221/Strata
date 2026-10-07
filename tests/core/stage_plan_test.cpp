// tests/core/stage_plan_test.cpp - delta 3 (layer/series): the two pure decisions the delta touches, exercised
// host-only (no CUDA, no GPU, no model):
//
//   1. `batch_mtp_reason` - why --batch-mtp is left off.  The delta lifts the "one GPU" refusal, so the cases
//      below pin BOTH ends: every missing requirement still refuses with its own line, a layer split is accepted,
//      and the one-GPU answers are byte-identical to the base tag's (the layer is inert by default).
//   2. `order_with_head_device` - which device runs the output head and the MTP draft layer (--head-device).
//      -1 (the flag absent) must return the order UNCHANGED: that is the negative the delta asks for (with a
//      layer split and without the flag the engine behaves exactly as before).  A requested device that cannot
//      be reached is refused with a reason and leaves the order alone (recommend, never force), including the
//      primary device of a split, which runs the first stage and cannot host the head.
//
// The test links nothing but the header: it is built by STRATA_BUILD_CONVERSATION_TESTS alongside the other
// host-only tests (see _host_build.bat).
#include "strata/core/stage_plan.hpp"

#include <cstdio>
#include <string>
#include <vector>

using strata::core::batch_mtp_reason;
using strata::core::draft_head_type;
using strata::core::head_stage_of;
using strata::core::order_with_head_device;
using strata::core::slot_mtp_reason;

namespace {

int g_fail = 0;
int g_checks = 0;

void check(bool ok, const std::string& what) {
    ++g_checks;
    std::printf("  %-88s %s\n", what.c_str(), ok ? "ok" : "FAIL");
    if (!ok) ++g_fail;
}

const char* reason(int batch, bool mtp, int spec, bool serve, bool same, size_t n_stages) {
    return batch_mtp_reason(batch, mtp, spec, serve, same, n_stages);
}

std::string vec_s(const std::vector<int>& v) {
    std::string s = "[";
    for (size_t i = 0; i < v.size(); ++i) s += (i == 0 ? "" : ",") + std::to_string(v[i]);
    return s + "]";
}

}  // namespace

int main() {
    std::printf("stage_plan: batch-MTP eligibility\n");
    // ---- every missing requirement still refuses, with its own line (unchanged from the base tag)
    check(std::string(reason(0, true, 4, true, false, 1)) == "it needs --batch 2 or more",
          "no --batch: refused");
    check(std::string(reason(3, false, 4, true, false, 1)) == "it needs --mtp", "no --mtp: refused");
    check(std::string(reason(3, true, 1, true, false, 1)) == "it needs --spec T (T >= 2)",
          "no --spec T>=2: refused");
    check(std::string(reason(3, true, 4, false, false, 1)) == "it needs --serve", "no --serve: refused");
    check(std::string(reason(3, true, 4, true, true, 2)) ==
              "it needs each stage of a layer split on its own GPU",
          "--split-device 0 (same device stages): refused");
    // ---- the one-GPU case the base tag allowed: still allowed, unchanged
    check(reason(4, true, 4, true, false, 1) == nullptr,
          "one GPU, everything present: allowed (base tag unchanged)");
    // ---- DELTA 3: a layer split (2, 3 and 4 stages, each on its own GPU) is allowed too
    check(reason(4, true, 4, true, false, 2) == nullptr, "layer split, 2 stages: allowed");
    check(reason(8, true, 4, true, false, 3) == nullptr, "layer split, 3 stages: allowed");
    check(reason(8, true, 4, true, false, 4) == nullptr, "layer split, 4 stages: allowed");
    // ---- and a split still refuses when a requirement is missing (the split is not an exemption)
    check(reason(4, false, 4, true, false, 3) != nullptr, "layer split without --mtp: refused");
    check(reason(1, true, 4, true, false, 3) != nullptr, "layer split with one slot: refused");
    check(reason(4, true, 4, false, false, 3) != nullptr, "layer split without --serve: refused");
    check(reason(4, true, 4, true, true, 3) != nullptr, "split on one device with 3 stages: refused");
    // the answer does not depend on the stage count beyond the split itself
    check(reason(4, true, 4, true, false, 1) == reason(4, true, 4, true, false, 5),
          "one GPU and five stages agree (no split-shaped condition left in the rule)");

    std::printf("stage_plan: the head's stage\n");
    check(head_stage_of(1) == 0, "one stage: the head is stage 0");
    check(head_stage_of(2) == 1, "two stages: the head is the last (stage 1)");
    check(head_stage_of(4) == 3, "four stages: the head is the last (stage 3)");
    check(head_stage_of(0) == 0, "no stage: stage 0 (degenerate, never reached)");

    std::printf("stage_plan: --head-device\n");
    bool ok = false;
    std::string why;

    // ---- the flag absent (-1): the order is returned exactly as it came.  This is the delta's negative: with a
    // layer split and no --head-device the engine is the base tag's.
    for (const std::vector<int>& v : {std::vector<int>{0}, std::vector<int>{0, 1},
                                      std::vector<int>{0, 1, 2}, std::vector<int>{0, 2, 3}}) {
        ok = false;
        why.clear();
        const std::vector<int> out = order_with_head_device(v, -1, ok, why);
        check(ok && out == v && why.empty(),
              "no --head-device: " + vec_s(v) + " unchanged (" + vec_s(out) + ")");
    }

    // ---- a request that is already the last stage's device: no reorder, no complaint
    ok = false;
    check(order_with_head_device({0, 1}, 1, ok, why) == std::vector<int>{0, 1} && ok && why.empty(),
          "2 GPUs, --head-device 1 (already last): unchanged");
    ok = false;
    check(order_with_head_device({0}, 0, ok, why) == std::vector<int>{0} && ok,
          "one GPU, --head-device 0: unchanged");
    ok = false;
    check(order_with_head_device({0, 1}, 0, ok, why) == std::vector<int>{0, 1} && !ok && !why.empty(),
          "2 GPUs, --head-device 0 (the primary): refused with a reason");

    // ---- a request the pipeline can reach: that device goes last, the others keep their order
    ok = false;
    why.clear();
    std::vector<int> out = order_with_head_device({0, 1, 2}, 1, ok, why);
    check(ok && why.empty() && out == std::vector<int>{0, 2, 1},
          "3 GPUs, --head-device 1: later stages reorder to " + vec_s(out) + " (the head moves to CUDA1)");
    check(!out.empty() && out.back() == 1, "3 GPUs, --head-device 1: the requested device is last");
    ok = false;
    out = order_with_head_device({0, 1, 2, 3}, 2, ok, why);
    check(ok && out == std::vector<int>{0, 1, 3, 2} && out.back() == 2,
          "4 GPUs, --head-device 2: reorder to " + vec_s(out));
    // the primary device always keeps the first stage (it is the prompt path's)
    for (const std::vector<int>& v : {std::vector<int>{0, 1}, std::vector<int>{0, 1, 2}}) {
        for (int req = 1; req < (int) v.size(); ++req) {
            ok = false;
            const std::vector<int> o = order_with_head_device(v, req, ok, why);
            if (!ok || o.empty() || o[0] != 0) {
                check(false, "the primary device stays first for " + vec_s(v) + " req " + std::to_string(req));
            }
        }
    }
    check(true, "the primary device stays first whenever a request is honoured");

    // ---- a request nothing can honour: unchanged, said
    ok = true;
    why.clear();
    out = order_with_head_device({0, 1}, 2, ok, why);
    check(!ok && out == std::vector<int>{0, 1} && why.find("2") != std::string::npos,
          "2 GPUs, --head-device 2 (not in the pipeline): refused, order kept");
    ok = true;
    out = order_with_head_device({0}, 1, ok, why);
    check(!ok && out == std::vector<int>{0}, "one GPU, --head-device 1: refused, order kept");
    ok = true;
    out = order_with_head_device({}, 0, ok, why);
    check(ok && out.empty(), "no devices: nothing to order");
    // the primary refusal names the way out (the card order), which is what the user is pointed at
    ok = false;
    (void) order_with_head_device({0, 1}, 0, ok, why);
    check(!ok && why.find("primary") != std::string::npos && why.find("order the cards") != std::string::npos,
          "the primary refusal says why and names the card order as the way to move the head");

    // ── The draft head's type: the defect that took the server down (2026-10-07) ──────────────────────────────
    //
    // A batch slot's drafter copies the main drafter's draft head (`dhead_`/`dvocab_`) but, in the base tag, NOT
    // its format: `dhead_type_` stayed at its -1 initial value, `native_mmvq(-1, ...)` threw "unsupported native
    // MMVQ GGML type" on the first slot draft of an admission, and the engine exited (exit 1) - the server
    // restarted it on every request that admitted a slot.  The native types the head can have (the engine's own
    // `native_mmvq_supported`): Q4_0=2, Q5_0=6, Q8_0=8, Q3_K=11, Q4_K=12, Q5_K=13, Q6_K=14, Q8_K=15?, IQ4_NL=20,
    // IQ4_XS=23, Q2_0=42, and the IQ family 16/17/18/21/22/29.
    std::printf("stage_plan: the draft head's type (the 2026-10-07 defect)\n");
    const int native_types[] = {2, 6, 7, 8, 11, 12, 13, 14, 16, 17, 18, 20, 21, 22, 23, 29, 42};
    for (const int t : native_types) {
        const int got = draft_head_type(/*shares=*/true, /*shared_type=*/t, /*has_own=*/false, /*own=*/-1,
                                        /*head_type=*/14);
        check(got == t, "a slot drafter takes the main drafter's type " + std::to_string(t) +
                        " (got " + std::to_string(got) + ")");
    }
    check(draft_head_type(true, 8, false, -1, 14) != -1,
          "the shared answer is never -1 (the value that threw in native_mmvq)");
    // the main drafter had no subset of its own (no rt/draft_vocab.bin): the whole native head is the answer
    check(draft_head_type(true, -1, false, -1, 14) == 14,
          "a shared drafter with no subset falls back to the native head's type");
    // this drafter built the subset itself: head->type() (or Q4_0 under --mtp-q4)
    check(draft_head_type(false, -1, true, 14, 14) == 14, "its own subset: head->type()");
    check(draft_head_type(false, -1, true, 2, 14) == 2, "its own subset under --mtp-q4: Q4_0");
    check(draft_head_type(false, -1, false, -1, 14) == 14, "no subset at all: the native head's type");
    check(draft_head_type(false, -1, true, -1, 14) == 14, "an unresolved own subset falls back to the head");

    std::printf("stage_plan: the reason --batch-mtp stays off (the drafters' path)\n");
    check(slot_mtp_reason(14, true) == nullptr, "a supported type: the slots may draft");
    check(slot_mtp_reason(-1, true) != nullptr, "-1 (never resolved): refused");
    check(std::string(slot_mtp_reason(-1, true)).find("no GGML type resolved") != std::string::npos,
          "-1's reason says the type was never resolved");
    check(slot_mtp_reason(14, false) != nullptr, "a type with no native MMVQ path: refused");
    check(std::string(slot_mtp_reason(14, false)).find("native MMVQ") != std::string::npos,
          "that reason names the native MMVQ path");
    check(slot_mtp_reason(-1, false) != nullptr, "-1 and unsupported: still refused");

    std::printf("stage_plan: %d checks, %d failed\n", g_checks, g_fail);
    return g_fail == 0 ? 0 : 1;
}
