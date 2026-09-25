#pragma once
#include "strata/kernels/ple.hpp"

namespace strata::kernels {

// Fixed qwen4exp single-token geometry: embd=2560, streams=4, history=9,
// convolution kernel=4/dilation=3. All buffers are caller-owned and capture-safe.
struct NativePlePostopsBuffers {
    float* key;        // 10240, normalized projected key
    float* query;      // 10240, temporary normalized hidden; may equal normalized
    float* gate;       // 4
    float* gated;      // 10240
    float* normalized; // 10240, append this to history AFTER this function
    float* conv;       // 10240
    float* result;     // 10240, may equal hidden exactly
};

// Matches pinned llama.cpp 3cf03257f219afbe7334045ff7c6a06ac68c627d CUDA
// build_ple AFTER the two projections. Only norm_key/query/conv and conv1d_f16
// fields of weights are used. Gamma is full [2560,4]; history is row-fastest
// [9,10240], and F16 taps are [4,10240]. Inputs must be finite.
// Require an explicit stream, nonnull aligned pointers and disjoint writable
// spans, with only the two exact aliases documented above allowed. No allocation
// or synchronization; retain every buffer through any captured graph execution.
void native_ple_postops(const float* projected_key, const float* hidden,
                        const float* value, const float* history,
                        const PleWeights& weights, const NativePlePostopsBuffers& buffers,
                        void* stream);

// n_tok consecutive tokens (the prompt path), row-major with the token slow: projected_key, hidden, key, query,
// gated, normalized and result [n_tok, 10240], value [n_tok, 2560], gate [n_tok, 4]. Bitwise n_tok calls of
// native_ple_postops, each followed by ple_history_advance: history is read, then left holding the last nine
// normalized rows. result may equal hidden and query may equal normalized; nothing else may overlap.
struct NativePleTokensBuffers {
    float* key;
    float* query;
    float* gate;
    float* gated;
    float* normalized;
    float* result;
};
void native_ple_postops_tokens(const float* projected_key, const float* hidden, const float* value,
                               float* history, const PleWeights& weights, const NativePleTokensBuffers& buffers,
                               int n_tok, void* stream);

} // namespace strata::kernels
