// src/core/adaptive_tier.cpp - see include/strata/core/adaptive_tier.hpp.
#include "strata/core/adaptive_tier.hpp"

#include "strata/kernels/cpu/expert_layout.hpp"

#include <algorithm>
#include <chrono>

namespace strata::core {

AdaptiveTier::~AdaptiveTier() {
    if (stream_) cudaStreamSynchronize(stream_);
    if (ev_) cudaEventDestroy(ev_);
    if (stream_) cudaStreamDestroy(stream_);
}

bool AdaptiveTier::init(ExpertCache& cache, ExpertSource& src, std::vector<int32_t>& host_res, int32_t* d_res,
                        int64_t n_layers, int64_t n_expert, int max_swaps, std::string& err) {
    cache_ = &cache;
    src_ = &src;
    res_ = &host_res;
    d_res_ = d_res;
    n_layers_ = n_layers;
    n_expert_ = n_expert;
    max_swaps_ = max_swaps;
    if (cudaStreamCreateWithFlags(&stream_, cudaStreamNonBlocking) != cudaSuccess ||
        cudaEventCreateWithFlags(&ev_, cudaEventDisableTiming) != cudaSuccess) {
        err = "adaptive tier: cannot create the refill stream";
        return false;
    }
    return true;
}

bool AdaptiveTier::adapt(std::vector<float>& usage, std::string& err) {
    const auto t0 = std::chrono::steady_clock::now();
    if (!pending_.empty()) return true;   // the previous swaps are still in flight
    struct Swap { float gain; int32_t layer, in, out; };
    std::vector<Swap> todo;
    std::vector<std::pair<float, int32_t>> cand, vict;
    std::vector<int32_t>& res = *res_;
    for (int64_t l = 0; l < n_layers_; ++l) {
        cand.clear();
        vict.clear();
        const float* u = usage.data() + l * n_expert_;
        const int32_t* r = res.data() + l * n_expert_;
        for (int32_t e = 0; e < (int32_t) n_expert_; ++e) {
            if (r[e] < 0) { if (u[e] >= 2.0f) cand.emplace_back(u[e], e); }
            else vict.emplace_back(u[e], e);
        }
        if (cand.empty() || vict.empty()) continue;
        std::sort(cand.begin(), cand.end(), [](auto& a, auto& b) { return a.first > b.first; });
        const size_t nc = std::min(cand.size(), vict.size());
        std::partial_sort(vict.begin(), vict.begin() + (ptrdiff_t) nc, vict.end(),
                          [](auto& a, auto& b) { return a.first < b.first; });
        for (size_t i = 0; i < nc; ++i) {
            if (cand[i].first < vict[i].first + 1.5f) break;
            todo.push_back({cand[i].first - vict[i].first, (int32_t) l, cand[i].second, vict[i].second});
        }
    }
    std::sort(todo.begin(), todo.end(), [](const Swap& a, const Swap& b) { return a.gain > b.gain; });
    if ((int) todo.size() > max_swaps_) todo.resize((size_t) max_swaps_);
    const auto& lay = strata::kernels::cpu::expert_layout();
    for (const Swap& s : todo) {
        const size_t in = (size_t) (s.layer * n_expert_ + s.in), out = (size_t) (s.layer * n_expert_ + s.out);
        const int32_t slot = res[out];
        const uint8_t* b = src_->blob(s.layer, s.in);
        // asynchronous: the copies run while the MTP drafts; the next window waits for them
        if (slot < 0 || b == nullptr || cudaMemcpyAsync(cache_->device_slot(slot), b, (size_t) lay.blob_bytes(s.layer),
                                                        cudaMemcpyHostToDevice, stream_) != cudaSuccess) {
            err = "adaptive tier: a refill copy failed";
            return false;
        }
        res[out] = kNotResident;                  // evicted now: the CPU computes it meanwhile
        pending_.emplace_back((int32_t) in, slot);   // resident once the copy has landed
    }
    if (!todo.empty()) cudaEventRecord(ev_, stream_);
    for (float& v : usage) v *= 0.7f;
    swaps += (int64_t) todo.size();
    ms += std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - t0).count();
    return true;
}

void AdaptiveTier::apply_pending(bool wait) {
    if (pending_.empty()) return;
    if (wait) cudaEventSynchronize(ev_);
    else if (cudaEventQuery(ev_) != cudaSuccess) return;
    std::vector<int32_t>& res = *res_;
    for (const auto& [i, slot] : pending_) res[(size_t) i] = slot;
    pending_.clear();
    if (d_res_ != nullptr) cudaMemcpy(d_res_, res.data(), res.size() * sizeof(int32_t), cudaMemcpyHostToDevice);
}

}  // namespace strata::core
