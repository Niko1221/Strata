// include/strata/core/second_gpu.hpp - a second GPU as another expert tier.
//
// It holds experts the first GPU's cache does not (the profile's next ranks, then the conversation's, through its
// own AdaptiveTier) and computes them for each layer of a verify window while the CPU pool computes the misses.
// It never spins on host memory: per layer the host launches one captured graph (copy the plan in, quantize the
// activations, the grouped experts; activations, plan and rows in mapped host memory) before the CPU pool and
// collects the rows after it, so it keeps its latency on a GPU that also drives a display (a graph spinning there
// stalls whenever the desktop draws).  One graph per format pair and group-count bucket: the grouped kernels'
// grids are sized for the bucket.  Native packs only: the kernels are the first GPU's hit path.
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

    /// One layer's share: group g is the expert in slot `slots[g]`, serving the routed entries
    /// `entries[starts[g] .. starts[g+1])` (entry i = token i/k).  `x` holds the window's n_tok activations.
    bool submit(int64_t layer, const float* x, int n_tok, int64_t k, const int32_t* slots, const int32_t* starts,
                const int32_t* entries, int n_groups, std::string& err);
    /// Waits for the submitted layer and writes its rows into `out` (row i = entry i, n_embd floats).
    bool finish(float* out, std::string& err);

    int64_t layers = 0, experts = 0, entries_done = 0;
    double ms_wait = 0;   ///< host time waiting for it after the CPU pool

private:
    bool graph_for(int gu_type, int d_type, int groups, cudaGraphExec_t& exec, std::string& err);

    int dev_ = -1, main_ = 0, max_t_ = 0;
    int64_t n_embd_ = 0, n_ff_ = 0, cap_ = 0;
    ExpertCache cache_;
    cudaStream_t s_ = nullptr;
    cudaEvent_t ev_ = nullptr;
    // pinned staging (portable, mapped): the activations, the plan, and the rows the down kernel writes
    float *h_x_ = nullptr, *m_x_ = nullptr, *h_out_ = nullptr, *m_out_ = nullptr;
    uint8_t *h_plan_ = nullptr, *m_plan_ = nullptr;
    uint8_t *d_xq_ = nullptr, *d_plan_ = nullptr, *d_scratch_ = nullptr;
    struct Graph { int gu, d, groups; cudaGraphExec_t exec; };
    std::vector<Graph> graphs_;      // per (gate/up, down) format pair and group-count bucket
    std::vector<int32_t> pending_;   // the submitted layer's entries, in row order
};

}  // namespace strata::core
