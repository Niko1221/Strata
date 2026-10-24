// Generated windows commit only inputs paired with emitted outputs. A merged prompt
// window commits all prompt inputs and emits only its last row's prediction.
#pragma once

#include <cstdint>

namespace strata::program {

inline int serve_window_size(int proposed, int64_t remaining, bool merged_prompt = false) {
    if (remaining <= 0) return 0;
    if (merged_prompt) return proposed; // all prompt rows produce only one output token
    return remaining < proposed ? (int) remaining : proposed;
}

// The final emitted token (including EOS) remains the unconsumed head. Verifier::commit
// restores recurrent/indexer/PLE state to this many inputs; later KV cells are overwritten.
template<class IsEos>
int serve_output_count(const int32_t* outputs, int accepted, IsEos is_eos, bool merged_prompt = false) {
    if (merged_prompt) return accepted; // earlier predictions are not generated tokens, even if EOS
    for (int i = 0; i < accepted; ++i)
        if (is_eos(outputs[i])) return i + 1;
    return accepted;
}

} // namespace strata::program
