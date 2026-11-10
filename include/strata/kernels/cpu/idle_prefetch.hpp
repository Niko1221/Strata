#pragma once
#include "strata/kernels/cpu/pool.hpp"
#include <array>
#include <algorithm>

namespace strata::kernels::cpu {
struct WarmBlob {
    const uint8_t* data = nullptr;
    size_t bytes = 0, down_off = 0, up_off = 0;
};

// Experimental CPU cache warming. Predictions affect reads only, never routing or expert arithmetic.
// One serialized host owns this object. Blob pointers must remain stable until stop().
class IdleExpertPrefetch {
public:
    IdleExpertPrefetch(ExpertPool& pool, int count) : pool_(pool), count_(std::clamp(count, 0, 4)) {}
    ~IdleExpertPrefetch() { stop(); if (starts_) report(); }
    bool init(const std::vector<std::vector<uint16_t>>& routers, int N, int NE, int k, std::string& error);
    void start(int layer, const float* x, int nt, const std::vector<WarmBlob>& blobs);
    void stop();
    void score(int layer, const int32_t* ids, int64_t n);
    std::vector<int> selected() const { return {chosen_.begin(), chosen_.begin() + chosen_n_}; }
    uint64_t predictions() const { return predicted_; }
    uint64_t read_lines() const { return lines_.load(std::memory_order_relaxed); }
    void report() const;

private:
    static void job(void*, int, int, const std::atomic<bool>&);
    void logits(int first, int end, const std::atomic<bool>&);
    void select();
    ExpertPool& pool_;
    int count_ = 0, N_ = 0, NE_ = 0, K_ = 0, L_ = 0, layer_ = -1, nt_ = 0;
    std::vector<uint8_t> rq_, xq_;
    std::vector<float> rs_, logits_;
    std::vector<WarmBlob> blobs_;
    std::array<int, 4> chosen_{};
    int chosen_n_ = 0;
    alignas(64) std::atomic<int> arrived_{0};
    alignas(64) std::atomic<bool> ready_{false};
    alignas(64) std::atomic<uint64_t> lines_{0};
    std::atomic<int> finished_{0};
    bool active_ = false;
    std::atomic<uint64_t> sink_{0};
    uint64_t starts_ = 0, predicted_ = 0, scored_ = 0, offered_ = 0, hit_ = 0, cancelled_ = 0;
};
}
