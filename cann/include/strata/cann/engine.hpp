// cann/include/strata/cann/engine.hpp - the model over a token sequence.
//
// One instance owns the device, the weight cache and the per-layer state. `step` walks one token
// through the 40 layers and returns the last residual stream; `top` projects it to the vocabulary;
// `reset` clears what carries between requests (conv history, recurrent state, KV cache) without
// touching the weights.
#pragma once

#include <cstdint>
#include <memory>
#include <string>
#include <utility>
#include <vector>

namespace strata::cann {

class Model {
public:
    struct Top {
        std::vector<std::pair<float, int>> best; // logit and token id, highest first
        double log_z = 0.0;                      // log-partition, so logprob = logit - log_z
    };

    // capacity is how many positions the KV cache can hold
    Model(const std::string & path, int64_t capacity);
    ~Model();

    Model(const Model &) = delete;
    Model & operator=(const Model &) = delete;

    int64_t capacity() const;
    int64_t vocab() const;

    // One token at `pos` through every layer; returns the final residual stream (fp32, kHidden).
    std::vector<float> step(int64_t token, int64_t pos);

    // Clear the recurrent state, the conv history and the KV cache. Weights stay resident.
    void reset();

    // The `k` most likely next tokens and the log-partition for this hidden state.
    Top top(const std::vector<float> & hidden, int64_t k);

    int64_t cache_bytes() const;
    static void print_slab_stats();

private:
    struct Impl;
    std::unique_ptr<Impl> impl_;
};

} // namespace strata::cann
