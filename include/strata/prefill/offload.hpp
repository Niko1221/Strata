// include/strata/prefill/offload.hpp - the prompt path's experts on a second GPU.
//
// A prompt chunk uses nearly every expert of every layer, and the first GPU (the 3090, on a PCIe 4.0 x4 link) has
// to stream all the ones its cache does not hold: ~0.65 GB (IQ3_XXS) to ~1.4 GB (UD-Q4_K_XL) a layer at 6.4 GB/s.
// The second GPU sits on a x16 link.  Per layer the first GPU copies the chunk's MoE input (FP16) to pinned host
// memory while its router and shared expert run; the second GPU copies it in, computes each of its experts -
// from its own cache when it holds them, else streamed from the pinned arena through a ring of staging slots -
// and adds each expert's weighted rows into a per-token FP32 sum, in ascending expert order.  The sum returns
// through pinned host memory to the first GPU, which computes the experts its own cache holds meanwhile and adds
// the sum in its combine.  Neither GPU reads the other's memory: Windows gives GeForce cards no peer access.
#pragma once

#include "strata/core/expert_cache.hpp"
#include "strata/core/expert_source.hpp"

#include <cuda_runtime.h>

#include <cstdint>
#include <memory>
#include <string>
#include <vector>

namespace strata::prefill {

class Offload {
public:
    Offload();
    ~Offload();
    Offload(const Offload&) = delete;
    Offload& operator=(const Offload&) = delete;

    /// Buffers for chunks of up to `chunk` tokens on `device`; `main_device` is made current again after every
    /// call.  `cache`/`res`: the device's expert cache and its residency (n_layers x n_expert slots, -1 when not
    /// held), or null.  `borrow`/`borrow_bytes`: device memory for the buffers (lent cache slots), else allocated.
    bool init(int device, int main_device, core::ExpertSource* src, const core::ExpertCache* cache, const int32_t* res,
              int64_t n_expert, int64_t chunk, std::string& err, void* borrow = nullptr, uint64_t borrow_bytes = 0);
    /// Device bytes `init` needs for chunks of `chunk` tokens (what a borrowed region must hold).
    static uint64_t bytes_needed(int64_t chunk, int64_t n_expert);

    /// Pinned [chunk, n_embd]: the first GPU writes the MoE input (FP16) here, and reads the sums (FP32) here.
    uint16_t* host_input() const;
    float* host_sum() const;

    /// Layer `layer` of a chunk of T tokens: expert experts[j] serves rows [off[j], off[j + 1]), row r being token
    /// src[r] with routing weight w[r].  Queues the work behind `input_ready` (an event of the first GPU, recorded
    /// once host_input holds the input) and returns; done() is recorded once host_sum holds the sums.
    bool run_layer(int64_t layer, int64_t T, const std::vector<int32_t>& experts, const std::vector<int32_t>& off,
                   const std::vector<int32_t>& src, const std::vector<float>& w, cudaEvent_t input_ready, std::string& err);
    cudaEvent_t done() const;

    int64_t experts_resident = 0;   ///< expert-layer groups served from its cache
    int64_t experts_streamed = 0;   ///< expert blobs copied from the arena
    double ms_host = 0;             ///< host time queuing its work
    double ms_gpu = 0;              ///< its GPU time from the input's arrival (all layers but the last)

private:
    struct Impl;
    std::unique_ptr<Impl> impl_;
};

}  // namespace strata::prefill
