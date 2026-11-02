// Opt-in solo pipeline beside batched MTP. No GPU or environment reads here.
#pragma once

namespace strata::program::pipeline_batch {

struct Config {
    bool enabled = false;
    int slots = 0;
    int groups = 1;
    bool groups_explicit = false;
    int later_stages = 0;
    int windows = 0;
    bool adapt_async = false;
    bool batch_mtp = false;
    int piped_drafts = 0;
};

// Deliberately limited to the topology of the historical RDNA3 experiment.
// Other topologies keep the existing startup gates, even with the opt-in set.
inline bool supported(const Config& c) {
    return c.enabled && c.slots == 4 && c.groups_explicit && c.groups == 2 &&
           c.later_stages == 1 && c.windows == 2 &&
           !c.adapt_async && c.batch_mtp && c.piped_drafts == 2;
}

// Apply AFTER diagnostic overrides. A BGEN admission uses the serial verifier
// even before the first slot becomes active. Existing batch windows never
// borrow the solo pipeline. An ordinary GEN can use it again after they drain.
inline int request_windows(bool coexist, int requested, int admit_slot, bool active_slots) {
    return coexist && (admit_slot >= 0 || active_slots) ? 0 : requested;
}

}  // namespace strata::program::pipeline_batch
