// include/strata/glm/experts.hpp - the routed experts: an LRU cache in RAM in front of the shards.
//
// 19,200 experts of 21,233,664 B are 407.7 GB; a 64 GB PC holds a few percent of them.  Each sparse layer of each
// token needs 8, so the cache's job is to (1) keep the experts that come back - routing is far from uniform and
// consecutive tokens share most of their choices - and (2) start a miss's read as early as possible and let the
// forward pass compute whatever is already resident while the rest arrive.
//
// One slot holds one expert exactly as it lies on disk: six plane regions (codes of down, gate, up; their scales),
// each with room for the alignment slack of an unbuffered read.  A run of adjacent planes (usually all three code
// planes, and all three scale planes) is one read into its first plane's region.  `view()` points Q4 matrices at
// the planes; nothing is converted.
//
// THREADING.  `request`, `view`, `release` and `prefetch` are called from the forward pass's thread only; the I/O
// threads only flip a slot from Loading to Ready (or Failed) and notify.
#pragma once

#include "strata/glm/container.hpp"
#include "strata/glm/io.hpp"
#include "strata/glm/kernels.hpp"

#include <atomic>
#include <condition_variable>
#include <cstdint>
#include <memory>
#include <mutex>
#include <unordered_map>
#include <vector>

namespace strata::glm {

struct ExpertView {
    Q4 gate, up, down;
};

struct ExpertStats {
    uint64_t requests = 0, hits = 0, misses = 0, prefetched = 0, prefetch_used = 0;
    uint64_t wait_us = 0;   ///< time the forward pass spent blocked on reads
};

class ExpertCache {
public:
    /// `spans[layer][e]` for every sparse layer (dense layers have an empty vector).  `budget` is the RAM the slots
    /// may take; at least `min_slots` are made regardless (a layer's request must fit).
    ExpertCache(IoPool& io, const GlmConfig& c, std::vector<std::vector<ExpertSpan>> spans, uint64_t budget);
    ~ExpertCache();

    /// Pin the slots of `n` experts of `layer` (starting reads for the misses); `slots[i]` gets expert i's slot.
    void request(int layer, const int* experts, int n, int* slots);
    /// True once the slot's bytes are in RAM; does not block.
    bool ready(int slot) const;
    /// Block until the slot is Ready (true) or its read failed (false).
    bool wait(int slot);
    ExpertView view(int slot) const;
    void release(int slot);
    /// Start reads for experts likely to be needed soon (not pinned; never evicts a pinned slot).
    void prefetch(int layer, const int* experts, int n);

    int slots() const { return (int) slots_.size(); }
    uint64_t slot_bytes() const { return slot_bytes_; }
    const ExpertStats& stats() const { return stats_; }

private:
    enum State : int { Empty = 0, Loading = 1, Ready = 2, Failed = 3 };
    struct Slot {
        uint8_t* mem = nullptr;
        int layer = -1, e = -1;
        std::atomic<int> state{Empty};
        std::atomic<int> parts{0};
        int pins = 0;
        uint64_t last_use = 0;
        const uint8_t* plane[6] = {};
        bool prefetched = false;
    };
    static uint64_t key(int layer, int e) { return ((uint64_t) layer << 32) | (uint32_t) e; }
    int find_victim();
    int acquire(int layer, int e, bool pin);
    void start_read(int s, int layer, int e);

    IoPool& io_;
    GlmConfig c_;
    std::vector<std::vector<ExpertSpan>> spans_;
    uint64_t plane_bytes_[6] = {}, region_[6] = {}, slot_bytes_ = 0;
    std::vector<std::unique_ptr<Slot>> slots_;
    std::unordered_map<uint64_t, int> where_;
    uint64_t clock_ = 0;
    mutable std::mutex m_;
    std::condition_variable cv_;
    ExpertStats stats_;
};

}  // namespace strata::glm
