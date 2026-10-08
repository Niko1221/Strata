#pragma once

#include "dsv4/config.hpp"
#include "dsv4/mem_plan.hpp"
#include <string>
#include <vector>
#include <cstdint>
#include <memory>

namespace dsv4 {

// Definizione completa di RunOpts per permettere a Impl di allocarlo direttamente
struct RunOpts {
    int ctx = 2048;
    bool gpu = true;
    int threads = 0;              // 0: as many as the machine has cores (see Model::load)
    bool qat_sim = true;
    int max_layers = -1;
    double vram_pct = -1.0;
    long long abs_slots = -1;
    long long reserve_mib = 1024;
    long long vram_free_mib = -1;
    long long vram_total_mib = -1;
    std::string profile_in;
    std::string profile_out;
    bool ram_all = false;
    // LRU arena for the experts that are neither in the profile's RAM copy nor in a VRAM slot. Without it every
    // MISS re-reads its 6.8 MiB of weights from the mmap'ed GGUF, which on a model bigger than RAM means disk.
    long long ram_cache_mib = 8192;
    int adapt_swaps = 0;
    bool verify_slots = false;
    bool verbose = true;          // startup report: memory plan, profile, RAM/VRAM expert counts
    // Prefill chunk: how many prompt tokens one forward pass evaluates together. The weights of an
    // expert are the traffic, so a chunk that routes N tokens through the same expert pays for that
    // expert's 6.8 MiB once instead of N times. 0 disables batching (one token per pass, the old
    // behaviour). Capped at the sliding window and at the context at run time.
    int prefill_chunk = 256;
};

// Statistiche di runtime (lette da dsv4_run per il report finale).
struct Stats {
    uint64_t h2d_bytes = 0;   // bytes uploaded to VRAM
    uint64_t admits = 0;      // experts loaded into a free VRAM slot
    uint64_t swaps = 0;       // adaptive evict+load decisions
    uint64_t hits = 0;        // routed expert evaluations served from VRAM
    uint64_t misses = 0;      // routed expert evaluations served from RAM/disk
    uint64_t cache_hits = 0;      // MISS experts already in the RAM LRU arena
    uint64_t cache_loads = 0;     // MISS experts copied into the arena (from mmap)
    uint64_t cache_evictions = 0; // arena slots reused for another expert
    uint64_t cache_bytes = 0;     // bytes copied into the arena
    uint64_t tokens = 0;      // tokens processed
    uint64_t prefill_tokens = 0;   // tokens processed inside a batched chunk
    uint64_t prefill_chunks = 0;   // batched chunks run
    double total_s = 0;       // forward time
    double cpu_miss_s = 0;    // time spent computing MISS experts on the CPU
    double gpu_hit_s = 0;     // time spent waiting for the HIT experts on the GPU
};

class Model {
public:
    Model();
    ~Model();

    Model(const Model&) = delete;
    Model& operator=(const Model&) = delete;

    bool load(const std::vector<std::string>& shards, const RunOpts& opts, std::string& err_out);
    void reset();
    /// One token. Same result as forward_chunk(&token, 1, pos, logits_out).
    void forward(int token, int pos, std::vector<float>* logits_out);
    /// A chunk of n CONSECUTIVE tokens starting at pos0. Only the last token's logits are computed
    /// (that is the one that starts the answer). Returns the number of tokens processed.
    int forward_chunk(const int* tokens, int n, int pos0, std::vector<float>* logits_out);
    /// Routing of every token of the last forward_chunk(): [t][layer*K + k]. Empty after a plain
    /// forward(), which instead leaves the last token in last_routing.
    const std::vector<int>& chunk_routing() const { return chunk_routing_; }
    int chunk_routing_n() const { return chunk_routing_n_; }
    /// Largest chunk this build can take (0 = prefill batching disabled).
    int max_chunk() const;
    void end_token();
    void end_chunk(int n);
    bool save_profile(std::string& err) const;
    void selfcheck() const;
    
    uint64_t model_hash() const;
    const Dsv4Config& config() const;
    const std::vector<std::string>& tokens() const;

    std::vector<int> last_routing;
    std::vector<int> chunk_routing_;
    int chunk_routing_n_ = 0;

    Stats stats;  // p_->st points here (set in the Model constructor)

private:
    struct Impl;
    std::unique_ptr<Impl> p_;
};

/// Maps the model's byte-level token strings back to text (see docs / decode_token_text).
std::string decode_token_text(const std::string& tok);

} // namespace dsv4
