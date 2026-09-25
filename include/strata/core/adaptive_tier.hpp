// include/strata/core/adaptive_tier.hpp - plan v0.3 P6: a VRAM expert tier follows the conversation.
//
// Every few rounds the missing experts routed most often (decayed counts, at least twice) move into the tier: into
// their layer's empty slots first, then into the slots of the layer's least-routed residents when they were routed
// clearly more often.  The copies run on their own stream between rounds; an evicted expert is a miss at once, a new
// one resident once its copy has landed (`apply_pending`).  Both decode loops (--serve and the speculative loop) use
// it, for the first GPU's cache and for a second GPU's.
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
    /// its device copy (may be null); up to `max_moves` experts move per call.  `device`: the cache's GPU (its
    /// stream and copies), `main_device` current again afterwards.
    bool init(ExpertCache& cache, ExpertSource& src, std::vector<int32_t>& host_res, int32_t* d_res, int64_t n_layers,
              int64_t n_expert, int max_moves, std::string& err, int device = -1, int main_device = 0);
    bool on() const { return res_ != nullptr; }
    /// An empty slot sized for `layer`'s experts.
    void add_free(int64_t layer, int32_t slot) { free_[(size_t) layer].push_back(slot); }
    int64_t free_slots() const;
    /// A tier in front of this one: the experts it holds or is loading are not candidates here, and this tier's
    /// copies of them are evicted first.
    void set_upper(const AdaptiveTier* upper) { upper_ = upper; }

    /// Ranks and submits this call's moves, then decays `usage` (n_layers x n_expert) unless `decay` is false (a
    /// tier ranked before another on the same counts).  Nothing to do while the previous moves are in flight.
    /// False when a copy could not be submitted.
    bool adapt(std::vector<float>& usage, std::string& err, bool decay = true);
    /// Admits the landed moves into `host_res` (and `d_res`); `wait` blocks until they have landed.
    void apply_pending(bool wait);

    int64_t swaps = 0, fills = 0;   ///< experts moved into an occupied / an empty slot
    double ms = 0;                  ///< host time in `adapt`

private:
    ExpertCache* cache_ = nullptr;
    ExpertSource* src_ = nullptr;
    std::vector<int32_t>* res_ = nullptr;
    int32_t* d_res_ = nullptr;
    int64_t n_layers_ = 0, n_expert_ = 0;
    int max_moves_ = 0;
    int dev_ = -1, main_ = 0;
    const AdaptiveTier* upper_ = nullptr;
    std::vector<uint8_t> upper_has_;                      // per (layer, expert), rebuilt each call
    std::vector<std::vector<int32_t>> free_;              // per layer
    std::vector<std::pair<int32_t, int32_t>> pending_;    // (residency index, slot) once the copies have landed
    cudaStream_t stream_ = nullptr;
    cudaEvent_t ev_ = nullptr;
};

}  // namespace strata::core
