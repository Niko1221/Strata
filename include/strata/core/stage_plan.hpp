// include/strata/core/stage_plan.hpp - the pure decisions behind two of the engine's device rules:
//
//   1. whether --batch-mtp may run at all (and why not, in one line, when it may not); and
//   2. the device that owns the output head and the MTP draft layer (--head-device / STRATA_HEAD_DEVICE).
//
// Nothing here includes CUDA and nothing here allocates: the engine calls these functions and
// tests/core/stage_plan_test.cpp exercises the SAME code host-only, so the rule the engine applies is the
// rule the test asserts, and a moved threshold shows up as a failing test instead of a silent behaviour change.
//
// Why the head is a placement of its own: the head (and the MTP draft layer, which reads the last stage's
// residual) run on the LAST stage, and the layers are placed by the split search.  So the head's device is a
// placement decision the split search never makes - on a 4080 + 3090 pair `auto` puts the head on whichever
// card the pipeline ends on, which is not necessarily the faster one.  --head-device picks it.
//
// NO-AFIRMACIONES (docs/FLAGS.md keeps the same list):
//   * --head-device chooses among the devices the pipeline already uses.  It does NOT move the primary
//     context (device 0 always runs the first stage and the prompt path), so it cannot put the head on the
//     primary device of a layer split; a head-only tail stage would be needed for that, and it is not
//     implemented.  The card ORDER is what moves the head to another card (serve/server.py's "head_device"
//     and the config's "gpu" list order, which becomes CUDA_VISIBLE_DEVICES).
//   * It is not a speed claim: nothing here measures anything.  The installer's ranking (setup.py) estimates
//     the per-layer time from the card's SMs and clock, which is the same estimate the split search uses.

#pragma once

#include <algorithm>
#include <cstddef>
#include <string>
#include <vector>

namespace strata::core {

/// Why --batch-mtp cannot run, or nullptr when it can.  The engine prints the reason and leaves the flag off
/// (plain batching still starts); the default run (no flag) is exactly the base tag's.
///
/// `n_stages`: the pipeline's stages, 1 = one GPU, 2+ = a layer split into that many stages.
/// `same_device_stages`: --split-device 0, both stages on the primary device (the batch slots themselves need
/// each stage on its own GPU, so batch-MTP under it is refused for the same reason).
inline const char* batch_mtp_reason(int batch, bool has_mtp, int spec, bool serve, bool same_device_stages,
                                    std::size_t n_stages) {
    (void) n_stages;   // a layer split is supported: every stage keeps its own session per slot, and the slot
                       // drafters live on the stage that runs the head (the last one)
    if (batch < 2) return "it needs --batch 2 or more";
    if (!has_mtp) return "it needs --mtp";
    if (spec < 2) return "it needs --spec T (T >= 2)";
    if (!serve) return "it needs --serve";
    if (same_device_stages) return "it needs each stage of a layer split on its own GPU";
    return nullptr;
}

/// The stage that owns the output head and the MTP draft layer: the last of the pipeline.
inline int head_stage_of(std::size_t n_stages) { return n_stages == 0 ? 0 : (int) n_stages - 1; }

// ── The draft head's GGML type, and the reason --batch-mtp stays off when the drafters cannot run ──────────────
//
// A batch slot's drafter does not build the draft head's token subset: it COPIES the main drafter's (`dhead_`,
// `dvocab_`, `n_dvocab_`) and must copy its FORMAT with it.  The main drafter is the authority: it made the subset
// from the native head (`head->type()`), or as Q4_0 under `--mtp-q4`, or in the Q6_K packed layout under
// STRATA_Q6_PACKED.  `draft_head_type` is that resolution; bind() uses it.
//
// Why it is a named function and not a one-line field copy: leaving the type at its -1 initial value in a shared
// drafter reached `native_mmvq(-1, ...)` and threw "unsupported native MMVQ GGML type" on the FIRST slot draft of
// an admission - the engine then exited (exit 1) and the server restarted it on every request that admitted a slot.
// `draft_head_type` never answers -1 when it can answer anything else, and `slot_mtp_reason` says out loud when the
// answer cannot run, so the engine can leave --batch-mtp off at start (recommend, never force).

/// The ggml type of the draft head's row set.  `shares_subset`: this drafter copies the main one's subset (the case
/// of every batch slot's drafter).  `has_own_subset`: it built the subset itself.  `head_type`: the whole native
/// head's type, the fallback whenever the drafter has no subset of its own.
inline int draft_head_type(bool shares_subset, int shared_type, bool has_own_subset, int own_type, int head_type) {
    if (shares_subset) return shared_type >= 0 ? shared_type : head_type;
    if (has_own_subset) return own_type >= 0 ? own_type : head_type;
    return head_type;
}

/// Why --batch-mtp must stay off because the drafters it needs cannot run, or nullptr when they can.
/// `draft_head_type`: the type the drafters resolved (see above).  `supported`: the engine's
/// `native_mmvq_supported(type)` for it - a boolean here, so this stays CUDA-free and host-testable.
inline const char* slot_mtp_reason(int draft_head_type, bool supported) {
    if (draft_head_type < 0) return "a slot's draft head has no GGML type resolved";
    if (!supported) return "the draft head's GGML type has no native MMVQ path on this card";
    return nullptr;
}

/// The pipeline's devices, in order, with the one that must own the head LAST.
///
/// `devices`: the devices as the pipeline runs them (devices[0] is the primary context's, the rest are the
/// later stages' - the engine's --split-device list).
/// `requested`: --head-device / STRATA_HEAD_DEVICE, or -1 for "as placed" (the last device, unchanged).
///
/// `ok` is false when the request cannot be reached, and `why` says it in one line (the caller prints it and
/// keeps the placement, as it does for the other device knobs: recommend, never force).  When `ok` is true the
/// returned vector is the device order to run: its last element is the head's device.
inline std::vector<int> order_with_head_device(const std::vector<int>& devices, int requested, bool& ok,
                                               std::string& why) {
    ok = true;
    why.clear();
    if (devices.empty() || requested < 0) return devices;   // as placed
    const int last = devices.back();
    if (requested == last) return devices;
    const bool known = std::find(devices.begin(), devices.end(), requested) != devices.end();
    if (!known) {
        ok = false;
        why = "device " + std::to_string(requested) + " is not one of the pipeline's devices";
        return devices;
    }
    std::size_t at = 0;
    while (at < devices.size() && devices[at] != requested) ++at;
    if (at == 0) {
        // The primary context (device 0) runs the first stage and the prompt path, so the head cannot be
        // moved onto it while the model spans several cards: the last stage is where the residual lands.
        ok = false;
        why = "device " + std::to_string(requested) + " is the primary one and runs the first stage; the head "
              "cannot move there (a head-only tail stage is not implemented) - order the cards instead, so "
              "that card is listed last (serve/server.py's \"head_device\")";
        return devices;
    }
    // Move the requested device to the end; the others keep their relative order.
    std::vector<int> out;
    out.reserve(devices.size());
    for (const int d : devices) if (d != requested) out.push_back(d);
    out.push_back(requested);
    return out;
}

}  // namespace strata::core
