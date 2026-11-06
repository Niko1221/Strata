// cann/include/strata/cann/hc.hpp - hyper-connections.
//
// Qwen3.8 keeps four parallel residual streams instead of one, and the layer's normalisation lives
// inside the mixing that collapses them, not in a separate attn_norm tensor. A block therefore
// sees `hc_mix(...).input`, and its output goes back into all four streams through
// `hc_combine(...)`. Both are per token: the batch dimension is not modelled here.
#pragma once

#include <cstdint>
#include <vector>

#include "strata/cann/ops.hpp"

namespace strata::cann {

constexpr int64_t kHc = 4;        // number of residual streams
constexpr int64_t kHcLowRank = 320;

struct HcMixed {
    std::vector<float> input;  // [hidden] - what the block reads
    std::vector<float> inject; // [kHc]    - per-stream weights used after the block
};

// x, w_norm and gate-shaped weights are laid out [stream][hidden], stream slowest, as ggml stores
// a [hidden, hc] tensor.
std::vector<float> hc_group_norm(const std::vector<float> & x, const std::vector<float> & w_norm);

// w_down [hidden*hc][320], w_up [320][hidden*hc], w_inject [hidden*hc][kHc]
HcMixed hc_mix(Stream & stream, const std::vector<float> & x, const std::vector<float> & w_norm,
               const std::vector<float> & w_down, const std::vector<float> & w_up,
               const std::vector<float> & w_inject);

// residual += block_out * 2*sigmoid(inject[stream]), for every stream
void hc_combine(std::vector<float> & residual, const std::vector<float> & block_out,
                const std::vector<float> & inject);

} // namespace strata::cann
