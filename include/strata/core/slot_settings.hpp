// include/strata/core/slot_settings.hpp - the batch windows' PER-SLOT request settings.
//
// A batch slot carries the request it admitted: its sampling (temperature / top_p / penalties / seed), its logit
// bias and the reply tail its penalty window reads.  Admission writes them ONCE, for that slot.  The verifier used to
// keep them in three private vectors and re-assign all three from the global sampling whenever the batch layout was
// rebuilt, which was fine while the layout was built once.  With `--parallel-recycle` the layout is rebuilt on
// every grow and shrink: three requests decoding in slots 0..2 must not lose their settings because a fourth request
// grew the layout to four.  So a resize keeps the surviving prefix exactly as it is, and only the new slots get the
// default.  Header-only and templated on the sampler params, so a host test walks grow and shrink without a GPU.
#pragma once

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <vector>

namespace strata::core {

template <typename SamplerParams>
struct SlotSettings {
    std::vector<SamplerParams> sp;              ///< per slot: its sampling (logit_bias points at bias_dev)
    std::vector<float*> bias_dev;               ///< per slot: its logit bias on the device (null = none)
    std::vector<std::vector<int32_t>> tail;     ///< per slot: its reply tail for the penalty window

    size_t size() const { return sp.size(); }
    void clear() { sp.clear(); bias_dev.clear(); tail.clear(); }

    /// A fresh batch layout: every slot at the verifier's default sampling.  init_slots uses this.
    void reset(size_t n, const SamplerParams& dflt) {
        sp.assign(n, dflt);
        bias_dev.assign(n, nullptr);
        tail.assign(n, {});
    }

    /// The device bias arrays of the slots a resize to `n` drops.  The caller frees them; the kept prefix stays.
    std::vector<float*> dropped_biases(size_t n) const {
        std::vector<float*> out;
        for (size_t i = n; i < bias_dev.size(); ++i)
            if (bias_dev[i] != nullptr) out.push_back(bias_dev[i]);
        return out;
    }

    /// Put back the settings of the slots that survive, after init_slots rebuilt the layout at this size.
    /// Only the prefix [0, min(size(), saved.size())) comes back: a grow adds slots with the default, it never
    /// rewrites what a request already decoding set.
    void restore_prefix(const SlotSettings& saved) {
        const size_t keep = std::min(size(), saved.size());
        for (size_t i = 0; i < keep; ++i) {
            sp[i] = saved.sp[i];
            bias_dev[i] = saved.bias_dev[i];
            tail[i] = saved.tail[i];
        }
    }
};

}  // namespace strata::core
