// include/strata/core/adaptive_tier.hpp - plan v0.3 P6: a VRAM expert tier follows the conversation.
//
// Every few rounds the missing experts routed most often (decayed counts, at least twice) move into the tier: into
// their layer's empty slots first, then into the slots of the layer's least-routed residents when they were routed
// clearly more often.  The copies run on their own stream while the rounds go on; an evicted expert is a miss at once,
// a new one resident once its copy has landed (`apply_pending`).  Both decode loops (--serve and the speculative loop)
// use it, for the first GPU's cache and for a second GPU's.
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

    /// `host_res` is the residency table (n_layers x n_expert, slot or kNotResident) the dispatch reads (the verify
    /// windows take a snapshot of it; a device copy of it is its owner's to refresh); up to `max_moves` experts move
    /// per call.  `device`: the cache's GPU (its stream and copies), `main_device` current again afterwards.
    bool init(ExpertCache& cache, ExpertSource& src, std::vector<int32_t>& host_res, int64_t n_layers, int64_t n_expert,
              int max_moves, std::string& err, int device = -1, int main_device = 0);
    bool on() const { return res_ != nullptr; }
    /// An empty slot sized for `layer`'s experts.
    void add_free(int64_t layer, int32_t slot) { free_[(size_t) layer].push_back(slot); }
    int64_t free_slots() const;
    /// A tier in front of this one: the experts it holds or is loading are not candidates here, and this tier's
    /// copies of them are evicted first.
    void set_upper(const AdaptiveTier* upper) { upper_ = upper; }
    /// A tier behind this one: a candidate it holds counts half its use here (moving it only shifts work from that
    /// tier's GPU to this one's).
    void set_lower(const AdaptiveTier* lower) { lower_ = lower; }
    /// An update waits while the previous one's moves are under way, instead of dropping the ones not staged yet (a
    /// tier whose moves all go out before the next window).
    void set_wait(bool on) { wait_ = on; }

    /// Ranks and queues this call's moves, then decays `usage` (n_layers x n_expert) unless `decay` is false (a tier
    /// ranked before another on the same counts).  The queued moves whose residents are still in place give way to
    /// this call's (ranked again if still worth it); the experts still under way are not candidates.
    bool adapt(std::vector<float>& usage, std::string& err, bool decay = true);

    /// By default the copies go out in pieces (the first GPU's tier: its copies cross the link that the verify
    /// window's own transfers need, and slowed them while they ran).  `stage` evicts the next queued moves' residents
    /// until `bytes` are staged and not sent: called before a window, whose residency snapshot then leaves their slots
    /// alone.  `pump_bytes` sends the staged moves' next `bytes`, a blob in as many calls as it takes.
    void stage(uint64_t bytes);
    bool pump_bytes(uint64_t bytes, std::string& err);

    /// Paced (the second GPU's tier, its copies between windows): whole moves, each resident evicted as its copy goes
    /// out.  `pump` submits them in order while they fit `budget` bytes, `sent` the bytes submitted, after `ev` (on
    /// the cache's GPU: the last work that may read the slots they overwrite).
    void set_paced(cudaEvent_t ev) { paced_ = true; after_ = ev; }
    bool pump(uint64_t budget, uint64_t& sent, std::string& err);

    /// A running average of the copies' measured rate (bytes per ms), 0 until the first timed ones have landed.
    double copy_rate() const { return rate_; }
    /// Admits the moves whose copies have landed into `host_res`; `wait` sends the queued ones and
    /// blocks until they have landed.
    void apply_pending(bool wait);

    int64_t swaps = 0, fills = 0;   ///< experts moved into an occupied / an empty slot
    int64_t dropped = 0;            ///< moves the next update dropped before their residents left
    uint64_t sent_bytes = 0;        ///< bytes copied
    double ms = 0;                  ///< host time in `adapt`

private:
    struct Move { int32_t layer, in, out, slot; };   // out: the evicted expert, < 0 for an empty slot
    ExpertCache* cache_ = nullptr;
    ExpertSource* src_ = nullptr;
    std::vector<int32_t>* res_ = nullptr;
    int64_t n_layers_ = 0, n_expert_ = 0;
    int max_moves_ = 0;
    int dev_ = -1, main_ = 0;
    bool paced_ = false;
    cudaEvent_t after_ = nullptr;
    const AdaptiveTier* upper_ = nullptr;
    const AdaptiveTier* lower_ = nullptr;
    bool wait_ = false;
    std::vector<uint8_t> blocked_;                        // per (layer, expert): not a candidate, rebuilt each call
    std::vector<uint8_t> upper_has_;                      // per (layer, expert), rebuilt each call
    std::vector<std::vector<int32_t>> free_;              // per layer
    // The queued moves in order and their (residency index, slot) once landed: [0, admitted_) admitted, up to sent_
    // sent (and off_ bytes of the next), up to staged_ with their residents evicted.
    std::vector<Move> queued_;
    std::vector<std::pair<int32_t, int32_t>> pending_;
    size_t admitted_ = 0, sent_ = 0, staged_ = 0;
    uint64_t off_ = 0;
    cudaEvent_t t0_ = nullptr, t1_ = nullptr;   // around the last timed copies, whose `timed_` bytes have not landed
    uint64_t timed_ = 0;
    double rate_ = 0;
    std::string failed_;                        // a copy apply_pending could not submit
    cudaStream_t stream_ = nullptr;
    // the batches in flight, oldest first: a ring of events, each with the moves sent when it was recorded
    static constexpr int kBatches = 64;
    cudaEvent_t evs_[kBatches] = {};
    size_t batch_sent_[kBatches] = {};
    int first_ = 0, flying_ = 0;
    uint64_t bytes_of(const Move& m) const;
    void evict(const Move& m);
    bool copy(const Move& m, uint64_t off, uint64_t n, std::string& err);
    bool time_begin(bool worth);
    bool end_batch(bool timed, uint64_t bytes, std::string& err);
};

}  // namespace strata::core
