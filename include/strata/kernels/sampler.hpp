// include/strata/kernels/sampler.hpp - the sampler chain, host-callable (P2.S2).
//
//     penalties -> top_k -> min_p -> top_p -> temperature -> pick
//
// The ORDER is llama.cpp's default chain (issue #53).  Three facts there are easy to get backwards: temperature
// comes AFTER the truncation filters, penalties apply EXACTLY ONCE - before the selection, never again after
// temperature - and min_p cuts BEFORE top_p (both on the pre-temperature logits).
#pragma once

#include <cstdint>

namespace strata::kernels {

struct SamplerParams {
    int top_k = 20;              // 0 disables the filter
    float top_p = 0.95f;         // 1.0 disables the filter
    float min_p = 0.0f;          // 0 disables; keeps tokens with p >= min_p * p_max (a prefix of the top_k list)
    float temperature = 1.0f;    // <= 0 means greedy
    int min_keep = 1;            // top_p keeps at least this many
    int penalty_last_n = 0;      // 0 disables; the window over `history` to count occurrences in
    float penalty_repeat = 1.0f;
    float penalty_freq = 0.0f;
    float penalty_present = 0.0f;
    uint64_t seed = 0;           // drives Philox, which is counter-based on (seed, token index)
    uint64_t counter = 0;        // absolute draw index of row 0; advance across decode calls
    bool greedy = false;
};

// logits (n_tokens, n_vocab) -> one sampled token id per row in `out`.
//
// **`logits` AND `out` ARE DEVICE POINTERS.**  This is a CUDA kernel launch, not a host function, and nothing
// in the parameter names or the types says so - `const float* logits` reads exactly like a host buffer.
// Measured: passing host memory for `logits` faults inside the kernel with an ILLEGAL MEMORY ACCESS that is
// reported by whatever synchronising call happens NEXT, which will be somewhere else entirely and will name a
// buffer that has nothing to do with it.  A sticky async fault does not know where it came from.
//
// `history` is (n_tokens, history_len) int32, the most recent tokens for each row with any unused slots set to
// -1; only the last `p.penalty_last_n` of each row are counted.  Pass nullptr and 0 when no penalties apply.
void sample_tokens(const float* logits, int n_tokens, int n_vocab, const int* history, int history_len,
                   const SamplerParams& p, int* out, void* stream);

}  // namespace strata::kernels
