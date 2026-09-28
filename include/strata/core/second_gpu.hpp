// include/strata/core/second_gpu.hpp - a second GPU as another expert tier.
//
// It holds experts the first GPU's cache does not (the profile's next ranks, then the conversation's, through its
// own AdaptiveTier) and computes them for each layer of a verify window while the CPU pool computes the misses.
// It never spins on host memory: per layer the host launches captured graphs once the CPU pool's workers have
// started (copy the plan in, quantize the activations, the grouped experts, their rows out), and the last one writes
// the rows into the window's own rows in mapped host memory and raises a flag there, which the first GPU waits for;
// so it keeps its latency on a GPU that also drives a display (a graph spinning there stalls whenever the desktop
// draws).  The rows cross PCIe in 16-byte stores: written one float at a time they took ~0.2 ms a layer and slowed the
// CPU pool's RAM reads.  One graph per format pair, group-count bucket and part: the grouped kernels' grids are sized
// for the bucket.  Native packs only: the kernels are the first GPU's hit path.
//
// Prefetch: between a layer's CPU pool and the next layer's ring the RAM is idle (the first GPU's serial part, ~0.28 ms
// a layer), and this GPU's x16 link copies ~12 MiB in that time: 4 UD-Q4_K_XL or ~7 IQ3_XXS experts.  The next layer's
// likeliest experts that neither GPU holds go into a few slots of their own, and that layer's share takes them from
// there: its own experts start at once, the prefetched ones once their copies have landed.
#pragma once

#include "strata/core/expert_cache.hpp"

#include <cuda_runtime.h>

#include <cstdint>
#include <string>
#include <vector>

namespace strata::core {

class SecondGpu {
public:
    SecondGpu() = default;
    ~SecondGpu();
    SecondGpu(const SecondGpu&) = delete;
    SecondGpu& operator=(const SecondGpu&) = delete;

    /// Buffers for windows of up to `max_t` tokens with `k` routed experts each, on `device`; the caller then opens
    /// and fills `cache()` with that device current.  `main_device` is made current again after every call.
    bool init(int device, int main_device, int64_t n_embd, int64_t n_ff, int max_t, int64_t k, std::string& err);
    ExpertCache& cache() { return cache_; }
    int device() const { return dev_; }

    /// Room for `slots` prefetched experts (at most 16) of up to `blob_bytes`.  After init().
    bool init_prefetch(int slots, uint64_t blob_bytes, std::string& err);
    int prefetch_slots() const { return pre_max_; }
    /// Starts copying `layer`'s experts ids[0..n) (n <= prefetch_slots(), `bytes` each from the pinned blobs `src`) into
    /// the prefetch slots, once the last submitted layer is done with them.
    bool prefetch(int64_t layer, const int32_t* ids, const uint8_t* const* src, int n, uint64_t bytes, std::string& err);
    /// The prefetch slot holding (`layer`, `expert`), or -1; submit() takes prefetch slot p as slot -2 - p.
    int prefetched(int64_t layer, int32_t expert) const {
        if (layer != pre_layer_) return -1;
        for (int p = 0; p < pre_n_; ++p) if (pre_ids_[p] == expert) return p;
        return -1;
    }

    /// One layer's share: group g is the expert in slot `slots[g]`, serving the routed entries
    /// `entries[starts[g] .. starts[g+1])` (entry i = token i/k).  `x` holds the window's n_tok activations.  Entry
    /// i's row goes to row i of `out` (n_embd floats, mapped host memory), and then `*flag` (mapped) is set to `ring`.
    /// Waits for the last share's graphs first (the plan and staging are theirs until then).
    bool submit(int64_t layer, const float* x, int n_tok, int64_t k, const int32_t* slots, const int32_t* starts,
                const int32_t* entries, int n_groups, float* out, uint32_t* flag, uint32_t ring, std::string& err);
    /// False, with the reason, when its work so far failed; does not wait.
    bool healthy(std::string& err) const;
    /// Recorded after the last submitted share: the last work that may read its cache and prefetch slots.
    cudaEvent_t done_event() const { return ev_; }

    int64_t layers = 0, experts = 0, entries_done = 0;
    int64_t prefetch_copied = 0, prefetch_used = 0;   ///< experts copied ahead / of those, computed here

private:
    bool graph_for(int gu_type, int d_type, int groups, int part, bool prep, cudaGraphExec_t& exec, std::string& err);

    int dev_ = -1, main_ = 0, max_t_ = 0;
    int64_t n_embd_ = 0, n_ff_ = 0, cap_ = 0;
    ExpertCache cache_;
    cudaStream_t s_ = nullptr;
    cudaEvent_t ev_ = nullptr;
    // pinned staging (portable, mapped): the activations and the plan
    float *h_x_ = nullptr, *m_x_ = nullptr;
    uint8_t *h_plan_ = nullptr, *m_plan_ = nullptr;
    uint8_t *d_xq_ = nullptr, *d_plan_ = nullptr, *d_scratch_ = nullptr;
    float* d_rows_ = nullptr;        // the rows, then copied out in full transactions
    unsigned* d_count_ = nullptr;    // the rows-out kernel's block counter
    // part 0: the experts in its cache, launched at once; part 1: the prefetched ones, after their copies.  `prep`: the
    // graph copies the plan and quantizes the activations (the first launch of a layer)
    struct Graph { int gu, d, groups, part; bool prep; cudaGraphExec_t exec; };
    std::vector<Graph> graphs_;
    // prefetch slots, their copy stream, and the event of the last copies
    static constexpr int kPrefetchMax = 16;
    uint8_t* d_pre_ = nullptr;
    uint64_t pre_cap_ = 0;
    int pre_max_ = 0, pre_n_ = 0;
    int64_t pre_layer_ = -1;
    int32_t pre_ids_[kPrefetchMax] = {};
    cudaStream_t pre_s_ = nullptr;
    cudaEvent_t pre_ev_ = nullptr;
};

}  // namespace strata::core
