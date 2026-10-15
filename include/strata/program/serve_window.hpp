// A serving window may commit only the inputs paired with outputs that will be emitted.
#pragma once

#include <cstdint>

namespace strata::program {

inline int serve_window_size(int proposed, int64_t remaining) {
    if (remaining <= 0) return 0;
    return remaining < proposed ? (int) remaining : proposed;
}

// The final emitted token (including EOS) remains the unconsumed head. Verifier::commit
// restores recurrent/indexer/PLE state to this many inputs; later KV cells are overwritten.
template<class IsEos>
int serve_output_count(const int32_t* outputs, int accepted, IsEos is_eos) {
    for (int i = 0; i < accepted; ++i)
        if (is_eos(outputs[i])) return i + 1;
    return accepted;
}

} // namespace strata::program
