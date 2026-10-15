// STRATA_PIPELINE_ACTIVE_ROWS=1: active-row packing; unlisted slot state is never submitted to the verifier.
#pragma once
#include <cstdint>
#include <cstddef>
namespace strata::program::pipeline_rows {
template<class Slots>
int pack(const Slots& slots, int base, int count, int* rows, int32_t* tokens, int64_t* pos) {
    int n = 0;
    for (int b = base; b < base + count; ++b) {
        const auto& slot = slots[(std::size_t) b];
        if (!slot.active) continue;
        rows[n] = b; tokens[n] = slot.x; pos[n] = slot.p; ++n;
    }
    return n;
}
}  // namespace strata::program::pipeline_rows
