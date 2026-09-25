// include/strata/core/adaptive_tier.hpp - plan v0.3 P6: the VRAM expert tier follows the conversation.
//
// Every few rounds the missing experts routed most often (decayed counts, at least twice) take the slots of their
// layer's least-routed residents when they were routed clearly more often.  The copies run on their own stream
// between rounds; an evicted expert is a CPU miss at once, a new one resident once its copy has landed
// (`apply_pending`).  Both decode loops (--serve and the speculative loop) use it.
#pragma once

#include "strata/core/expert_cache.hpp"
#include "strata/core/expert_source.hpp"

#include <cuda_runtime.h>

#include <cstdint>
#include <string>
#include <utility>
#include <vector>

namespace strata::core {

class AdaptiveTier {
public:
    AdaptiveTier() = default;
    ~AdaptiveTier();
    AdaptiveTier(const AdaptiveTier&) = delete;
    AdaptiveTier& operator=(const AdaptiveTier&) = delete;

    /// `host_res` is the residency table (n_layers x n_expert, slot or kNotResident) the dispatch reads, `d_res`
    /// its device copy; up to `max_swaps` experts move per call.
    bool init(ExpertCache& cache, ExpertSource& src, std::vector<int32_t>& host_res, int32_t* d_res, int64_t n_layers,
              int64_t n_expert, int max_swaps, std::string& err);

    /// Ranks and submits this call's swaps, then decays `usage` (n_layers x n_expert).  Nothing to do while the
    /// previous swaps are in flight.  False when a copy could not be submitted.
    bool adapt(std::vector<float>& usage, std::string& err);
    /// Admits the landed swaps into `host_res` and `d_res`; `wait` blocks until they have landed.
    void apply_pending(bool wait);

    int64_t swaps = 0;   ///< experts swapped in
    double ms = 0;       ///< host time in `adapt`

private:
    ExpertCache* cache_ = nullptr;
    ExpertSource* src_ = nullptr;
    std::vector<int32_t>* res_ = nullptr;
    int32_t* d_res_ = nullptr;
    int64_t n_layers_ = 0, n_expert_ = 0;
    int max_swaps_ = 0;
    std::vector<std::pair<int32_t, int32_t>> pending_;    // (residency index, slot) once the copies have landed
    cudaStream_t stream_ = nullptr;
    cudaEvent_t ev_ = nullptr;
};

}  // namespace strata::core
