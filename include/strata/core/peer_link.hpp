// include/strata/core/peer_link.hpp - multi-GPU, opt-in (--peer-link): two engine PROCESSES, one card each, that are
// each other's peer expert tier ("mutual help").
//
// `--peer-device` (#531) gives ONE engine a second card's VRAM as an extra expert tier.  When a second request
// arrives, the second card can instead run its own engine for it (one sequence per process, as always) - but then each
// engine sees only its own card's expert cache, and the experts the other card holds become CPU misses.  The link
// keeps both caches usable for both engines: each engine publishes which experts its card holds, and computes the
// other engine's rows for those experts on its own card, with the same kernels as the `--peer-device` tier (q8_1
// activations + native_expert_grouped), from the same activation values - so an expert gives the same rows on either
// card.  The two caches are filled from different ranks of the profile (deduplicated), so both engines see roughly the
// coverage of both cards.
//
// The two processes talk through one small shared-memory file (no CUDA IPC, no P2P, no MPS): the requesting engine's
// pool thread writes the layer's activations and the (expert -> rows) plan, the owning engine's service thread copies
// them to its card, computes, and writes the rows back; the requester waits for them where the `--peer-device` tier
// waits for its card (PeerExperts::finish).  Only the OWNER ever touches its card and its cache, so all ordering
// against the owner's own cache writes (adaptive swaps, the prompt path's loan, VRAM resizes) is local to the owner:
// before such a write it holds the link (`hold`: the service finishes what it launched and starts nothing new), and
// it publishes the new residency when the copies have landed.  An expert the owner no longer holds when a request
// arrives (the requester planned with a residency a moment old) is copied in from the owner's RAM copy for that one
// request - slower, the same rows.
#pragma once

#include <atomic>
#include <cstdint>
#include <mutex>
#include <string>
#include <thread>
#include <vector>

namespace strata::core {

class ExpertCache;
class ExpertSource;
struct PeerLinkShm;

class PeerLink {
public:
    PeerLink() = default;
    ~PeerLink();
    PeerLink(const PeerLink&) = delete;
    PeerLink& operator=(const PeerLink&) = delete;

    /// Maps the link file at `path`: the first engine creates it, the other joins (waiting up to `wait_s` for its
    /// size); both check the model geometry.  Remove a stale file before starting either engine.
    bool open(const std::string& path, int role, int64_t n_layers, int64_t n_expert, double wait_s, std::string& err);
    void close();
    bool valid() const { return shm_ != nullptr; }
    int role() const { return role_; }

    // ---- requester side (this engine's pool thread) ----
    /// Whether the other engine serves now and its card holds (layer, expert).
    bool partner_has(int64_t layer, int64_t expert) const;
    /// Posts the entries with kind[i] == 2 of one layer (x: n_tok rows of the model's width).
    bool request(int64_t layer, const float* x, const int32_t* ids, int64_t n_tok, int64_t k, const int32_t* kind,
                 std::string& err);
    /// Waits for the posted rows and writes them into `out` (row i = entry i).
    bool wait(float* out, std::string& err);
    int64_t entries() const { return entries_; }
    int64_t experts() const { return experts_; }
    double ms_wait = 0;

    // ---- owner side (this engine serves the other one) ----
    /// Starts the service thread on the calling thread's CUDA device, computing from `cache` (slots) with the
    /// residency `host_res` (published now) and `src` (the RAM copy, for an expert no longer resident).
    bool serve(ExpertCache& cache, ExpertSource& src, const std::vector<int32_t>& host_res, std::string& err);
    /// Before this engine writes into or unmaps its cache's slots: the service finishes what it launched and
    /// starts nothing new until `unhold`.  Nests on the same thread.
    /// Whether the other engine may plan rows for this card (it sees this before it plans each layer).  Off while
    /// this engine sleeps; a request already posted is still answered.
    void set_serving(bool on);
    void hold();
    void unhold();
    /// The residency the service (and the other engine's plans) use from now on.  Call when the copies behind it
    /// have landed (removals: under `hold`, before the slots are written).
    void publish(const std::vector<int32_t>& host_res);
    /// Routed entries this engine computed for the other one per (layer, expert), for the adaptive tier's victims
    /// (an expert the other engine uses is not the least used one).  Written by the service thread.
    const float* served_usage() const { return svc_usage_.empty() ? nullptr : svc_usage_.data(); }
    void decay_served(float f);
    int64_t served_entries() const { return served_entries_.load(std::memory_order_relaxed); }
    int64_t served_fallbacks() const { return served_fallback_.load(std::memory_order_relaxed); }
    void stop();
    /// timing of the service (owner side), cumulative: requests, ms from post to pickup, ms enqueueing (incl. lock),
    /// ms in handle (enqueue + GPU)
    int64_t n_handled() const { return n_handled_; }
    double t_recv_ms() const { return t_recv_ms_; }
    double t_enqueue_ms() const { return t_enqueue_ms_; }
    double t_handle_ms() const { return t_handle_ms_; }

private:
    int64_t n_handled_ = 0;
    double t_recv_ms_ = 0, t_enqueue_ms_ = 0, t_handle_ms_ = 0;
    void loop();
    bool handle(std::string& err);

    PeerLinkShm* shm_ = nullptr;
    size_t map_bytes_ = 0;
    int role_ = -1;
    int64_t n_layers_ = 0, n_expert_ = 0;
    int32_t* res_mine_ = nullptr;          // in the file: what this card holds (the other engine plans with it)
    const int32_t* res_partner_ = nullptr; // in the file: what the other card holds
    // requester
    uint64_t seq_ = 0;
    std::vector<int32_t> row_of_;
    int64_t rows_posted_ = 0, entries_ = 0, experts_ = 0;
    // owner
    ExpertCache* cache_ = nullptr;
    ExpertSource* src_ = nullptr;
    int device_ = -1;
    std::vector<int32_t> svc_res_;
    std::vector<float> svc_usage_;
    std::atomic<int64_t> served_entries_{0}, served_fallback_{0};
    std::recursive_mutex mu_;
    int hold_depth_ = 0;
    std::thread thread_;
    std::atomic<bool> stop_{false};
    bool registered_ = false;
    void* stream_ = nullptr;               // cudaStream_t, high priority
    void* d_shm_ = nullptr;                // the link file's device address (mapped)
    float* d_x_ = nullptr;                 // (unused: zero-copy)
    float* d_out_ = nullptr;
    void* d_meta_ = nullptr;               // device address of h_meta_ (mapped)
    void* h_meta_ = nullptr;
    uint8_t* d_q8_ = nullptr;
    void* d_scratch_ = nullptr;
    uint8_t* d_stage_ = nullptr;           // blobs of experts no longer resident, kStage at a time
    uint64_t stage_blob_ = 0;
    uint64_t last_seq_ = 0;
};

}  // namespace strata::core
