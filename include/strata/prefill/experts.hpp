// include/strata/prefill/experts.hpp - the prompt path's routed experts on one GPU.
//
// A prompt chunk uses nearly every expert of every layer.  Its experts go through in batches of consecutive ids: per
// batch one staging copy for each run of adjacent arena blobs among the experts the GPU's cache does not hold, one
// dequantization launch, one gather of the batch's rows, the two GEMMs of each expert with a SwiGLU between them, and
// one ordered add of every row, times its routing weight, into its token's sum.  A token's experts add up in ascending
// id order, whatever the batches.
//
// Local: on the prompt path's own GPU and stream, reading the chunk's MoE input and adding into its MoE output there.
// Remote: on a second GPU.  The first GPU (the 3090) sits on a PCIe 4.0 x4 link, over which an expert its cache lacks
// streams at 6.4 GB/s (~0.65 GB (IQ3_XXS) to ~1.4 GB (UD-Q4_K_XL) a layer); the second GPU sits on x16.  The first GPU
// copies the chunk's MoE input (FP16) to pinned host memory; the second copies it in, computes the experts the first
// one's cache lacks - from its own cache when it holds them - and returns their sums (FP32) the same way, while the
// first computes the experts its cache holds.  Neither GPU reads the other's memory: Windows gives GeForce cards no
// peer access.
#pragma once

#include "strata/core/expert_cache.hpp"
#include "strata/core/expert_source.hpp"

#include <cuda_runtime.h>

#include <cstdint>
#include <memory>
#include <string>
#include <vector>

namespace strata::prefill {

class ExpertRunner {
public:
    ExpertRunner();
    ~ExpertRunner();
    ExpertRunner(const ExpertRunner&) = delete;
    ExpertRunner& operator=(const ExpertRunner&) = delete;

    /// For chunks of up to `max_chunk` tokens on `device`.  `stream`: the prompt path's stream on that device (local),
    /// or null (remote: its own streams; `main_device` is made current again after every call).  `cache`/`res`: the
    /// device's expert cache and its residency (n_layers x n_expert slots, -1 when not held), or null.  `streams`:
    /// whether experts its cache does not hold come its way (staging buffers).
    bool init(int device, int main_device, void* stream, core::ExpertSource* src, const core::ExpertCache* cache,
              const int32_t* res, int64_t n_expert, int64_t max_chunk, bool streams, std::string& err);
    /// Device bytes of the buffers for chunks of `chunk` tokens.
    static uint64_t bytes_needed(int64_t chunk, int64_t n_expert, bool remote, bool streams);
    /// The buffers for chunks of up to `chunk` tokens, carved from `region` (`bytes` long, e.g. lent cache slots), or
    /// with a null region allocated for `max_chunk` tokens (once).  Before the first layer of every prompt that
    /// uses a region.
    bool bind(void* region, uint64_t bytes, int64_t chunk, std::string& err);

    /// Remote: pinned [max_chunk, n_embd]: the MoE input (FP16) goes in here, the sums (FP32) come out here.
    uint16_t* host_input() const;
    float* host_sum() const;

    /// Layer `layer` of a chunk of T tokens: expert experts[j] serves rows [off[j], off[j + 1]), row r being token
    /// src[r] with routing weight w[r]; each row times its weight adds to its token's sum.  Local: reads `input`
    /// [T, n_embd] (FP16) and adds into `sum` [T, n_embd], queued on the prompt path's stream.  Remote: `input` and
    /// `sum` are null; its work waits for `input_ready` (an event of the first GPU, recorded once host_input holds
    /// the input), and done() is recorded once host_sum holds the sums.  Returns once the work is queued.
    bool run_layer(int64_t layer, int64_t T, const std::vector<int32_t>& experts, const std::vector<int32_t>& off,
                   const std::vector<int32_t>& src, const std::vector<float>& w, cudaEvent_t input_ready,
                   const uint16_t* input, float* sum, std::string& err);
    cudaEvent_t done() const;

    int64_t experts_resident = 0;   ///< expert-layer groups served from its cache
    int64_t experts_streamed = 0;   ///< expert blobs copied from the arena
    double ms_host = 0;             ///< host time queuing its work
    double ms_gpu = 0;              ///< remote: its GPU time from the input's arrival (all layers but the last)

private:
    struct Impl;
    std::unique_ptr<Impl> impl_;
};

}  // namespace strata::prefill
