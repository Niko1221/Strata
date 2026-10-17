// File prediction can be omitted only when decode cannot acquire file-backed work.
// Evaluate after resource plans and prediction consumers are finalized, immediately
// before serving requests.
#pragma once
#include <cstddef>
#include <cstdint>
#include <vector>

namespace strata::core {
struct ResidentDecodePrefetch {
    bool serving = false;
    bool batched = false;             // prompts can lend GPU slots beside another request's decode
    bool elastic_kv = false;          // decode can give GPU expert slots to KV
    bool independent_consumer = false; // Foresight or routing statistics
    bool explicit_lookahead = false;  // preserve the user's existing environment override
};

// The owner calls this before concurrent residency changes. It is not an async query.
// The lifecycle proof, rather than a permanent copy of this table, permits the skip:
// swaps preserve GPU + RAM + staged copies; serial prompts return loans before decode.
template<class InRam>
bool resident_decode_needs_no_prediction(const ResidentDecodePrefetch& mode,
                                         const std::vector<int32_t>& gpu, size_t expected,
                                         InRam in_ram) {
    if (!mode.serving || mode.batched || mode.elastic_kv || mode.independent_consumer ||
        mode.explicit_lookahead || expected == 0 || gpu.size() != expected) return false;
    for (size_t i = 0; i < expected; ++i)
        if (gpu[i] < 0 && !in_ram(i)) return false;
    return true;
}
} // namespace strata::core
