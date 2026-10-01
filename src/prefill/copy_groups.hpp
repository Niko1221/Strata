#pragma once
#include <cuda_runtime.h>
#include <algorithm>
#include <atomic>
#include <vector>

namespace strata::prefill::detail {
// Private long-prefill planner and event contract, shared with the GPU fixture.
// Entries expose `job`: -1 for direct pinned, otherwise the host staging index.
struct CopyGroups {
    std::vector<size_t> ends;
    int ring;
    struct Step {
        size_t end;
        int last_slot;
        bool first, terminal;
        bool fits(size_t consumed, int ring) const { return end <= consumed + (size_t) ring; }
    };

    template<class Entries>
    CopyGroups(const Entries& seq, const std::vector<char>& pinned, int gpu_ring,
               int host_ring, int direct_batch, int staged_batch) : ring(gpu_ring) {
        if (direct_batch == 1 && staged_batch == 1) return;
        ends.resize(seq.size());
        for (size_t begin = 0; begin < seq.size();) {
            size_t end = begin + 1;
            if (seq[begin].job < 0) {
                const size_t limit = std::min({seq.size(), begin + (size_t) direct_batch,
                    begin + (size_t) ring - begin % (size_t) ring});
                while (end < limit && seq[end].job < 0) ++end;
            } else if (staged_batch > 1 && pinned[seq[begin].job % host_ring]) {
                // Preparing this group must not require a source buffer still
                // owned by this group's DMA. Host ownership fences are separate.
                const size_t limit = std::min({seq.size(), begin + (size_t) staged_batch,
                    begin + (size_t) ring - begin % (size_t) ring,
                    begin + (size_t) host_ring - (size_t) seq[begin].job % (size_t) host_ring});
                while (end < limit && seq[end].job == seq[begin].job + (int) (end - begin) &&
                       pinned[seq[end].job % host_ring]) ++end;
            }
            for (size_t i = begin; i < end; ++i) ends[i] = end;
            begin = end;
        }
    }

    Step at(size_t idx) const {
        const size_t end = ends.empty() ? idx + 1 : ends[idx];
        return {end, (int) ((end - 1) % (size_t) ring),
                idx == 0 || ends.empty() || ends[idx - 1] != end, idx + 1 == end};
    }
    cudaError_t wait_reuse(const Step& step, bool live, const cudaEvent_t* used,
                           cudaStream_t copy) const {
        return step.first && live ? cudaStreamWaitEvent(copy, used[step.last_slot], 0) : cudaSuccess;
    }
    cudaError_t publish(const Step& step, const cudaEvent_t* copied,
                        cudaStream_t copy, std::atomic<size_t>& issued) const {
        if (!step.terminal) return cudaSuccess;
        // Ordered copy stream: record the terminal completion before publishing.
        const auto status = cudaEventRecord(copied[step.last_slot], copy);
        issued.store(step.end, std::memory_order_release);
        return status;
    }
    cudaError_t wait_copy(size_t idx, const cudaEvent_t* copied, cudaStream_t compute) const {
        // Caller first acquires publication. Its used record follows this wait;
        // ring-window release prevents re-recording before the old wait is queued.
        return cudaStreamWaitEvent(compute, copied[at(idx).last_slot], 0);
    }
};
}  // namespace strata::prefill::detail
