// src/core/adaptive_tier.cpp - see include/strata/core/adaptive_tier.hpp.
#include "strata/core/adaptive_tier.hpp"

#include "strata/kernels/cpu/expert_layout.hpp"

#include <algorithm>
#include <chrono>

namespace strata::core {
namespace {
struct OnDevice {   // the tier's GPU current for one call (when it has one), the main one again afterwards
    int main;
    bool set;
    OnDevice(int dev, int main_dev) : main(main_dev), set(dev >= 0) { if (set) cudaSetDevice(dev); }
    ~OnDevice() { if (set) cudaSetDevice(main); }
};
}  // namespace

AdaptiveTier::~AdaptiveTier() {
    OnDevice on(dev_, main_);
    if (stream_) cudaStreamSynchronize(stream_);
    cudaEvent_t evs[] = {ev_, t0_, t1_};
    for (cudaEvent_t e : evs) if (e) cudaEventDestroy(e);
    if (stream_) cudaStreamDestroy(stream_);
}

bool AdaptiveTier::set_paced() {
    OnDevice on(dev_, main_);
    paced_ = cudaEventCreate(&t0_) == cudaSuccess && cudaEventCreate(&t1_) == cudaSuccess;
    return paced_;
}

bool AdaptiveTier::init(ExpertCache& cache, ExpertSource& src, std::vector<int32_t>& host_res, int32_t* d_res,
                        int64_t n_layers, int64_t n_expert, int max_moves, std::string& err, int device, int main_device) {
    cache_ = &cache;
    src_ = &src;
    res_ = &host_res;
    d_res_ = d_res;
    n_layers_ = n_layers;
    n_expert_ = n_expert;
    max_moves_ = max_moves;
    dev_ = device;
    main_ = main_device;
    free_.assign((size_t) n_layers, {});
    OnDevice on(dev_, main_);
    if (cudaStreamCreateWithFlags(&stream_, cudaStreamNonBlocking) != cudaSuccess ||
        cudaEventCreateWithFlags(&ev_, cudaEventDisableTiming) != cudaSuccess) {
        res_ = nullptr;   // off
        err = "adaptive tier: cannot create the refill stream";
        return false;
    }
    return true;
}

int64_t AdaptiveTier::free_slots() const {
    int64_t n = 0;
    for (const auto& f : free_) n += (int64_t) f.size();
    return n;
}

bool AdaptiveTier::submit(const Move& m, std::string& err) {
    const uint8_t* b = src_->blob(m.layer, m.in);
    if (b == nullptr) { err = "adaptive tier: a refill copy failed"; return false; }
    if (m.out >= 0) (*res_)[(size_t) (m.layer * n_expert_ + m.out)] = kNotResident;   // evicted now: a miss meanwhile
    if (cudaMemcpyAsync(cache_->device_slot(m.slot), b, (size_t) strata::kernels::cpu::expert_layout().blob_bytes(m.layer),
                        cudaMemcpyHostToDevice, stream_) != cudaSuccess) {
        err = "adaptive tier: a refill copy failed";
        return false;
    }
    return true;
}

bool AdaptiveTier::pump_n(size_t n, std::string& err) {
    if (next_ >= queued_.size()) return true;
    OnDevice on(dev_, main_);
    if (after_ != nullptr && cudaStreamWaitEvent(stream_, after_, 0) != cudaSuccess) {
        err = "adaptive tier: a refill copy failed";
        return false;
    }
    for (size_t i = 0; i < n && next_ < queued_.size(); ++i)
        if (!submit(queued_[next_++], err)) return false;
    if (cudaEventRecord(ev_, stream_) != cudaSuccess) { err = "adaptive tier: a refill copy failed"; return false; }
    return true;
}

bool AdaptiveTier::pump(uint64_t budget, uint64_t& sent, std::string& err) {
    sent = 0;
    size_t n = 0;
    for (; next_ + n < queued_.size(); ++n) {
        const uint64_t b = strata::kernels::cpu::expert_layout().blob_bytes(queued_[next_ + n].layer);
        if (sent + b > budget) break;
        sent += b;
    }
    if (n == 0) return true;
    OnDevice on(dev_, main_);
    float took = 0;
    if (batch_ > 0 && cudaEventQuery(t1_) == cudaSuccess && cudaEventElapsedTime(&took, t0_, t1_) == cudaSuccess &&
        took > 0) {
        const double r = (double) batch_ / (double) took;   // the last batch has landed
        rate_ = rate_ > 0 ? 0.8 * rate_ + 0.2 * r : r;
        batch_ = 0;
    }
    const bool timed = batch_ == 0 && n >= 4;   // a batch of a few copies: its start and end dominate
    if (timed && cudaEventRecord(t0_, stream_) != cudaSuccess) { err = "adaptive tier: a refill copy failed"; return false; }
    if (!pump_n(n, err)) return false;
    if (timed) {
        if (cudaEventRecord(t1_, stream_) != cudaSuccess) { err = "adaptive tier: a refill copy failed"; return false; }
        batch_ = sent;
    }
    return true;
}

bool AdaptiveTier::adapt(std::vector<float>& usage, std::string& err, bool decay) {
    const auto t0 = std::chrono::steady_clock::now();
    if (!failed_.empty()) { err = failed_; return false; }
    if (paced_) {   // the moves not submitted by now give way to this call's (ranked again if still worth it)
        for (size_t i = next_; i < queued_.size(); ++i) {
            const Move& m = queued_[i];
            if (m.out >= 0) {
                --swaps;
            } else {
                free_[(size_t) m.layer].push_back(m.slot);
                --fills;
            }
        }
        dropped += (int64_t) (queued_.size() - next_);
        queued_.resize(next_);
        pending_.resize(next_);
        apply_pending(false);
    }
    if (!pending_.empty()) return true;   // the previous moves are still in flight
    if (upper_ != nullptr) {
        upper_has_.assign((size_t) (n_layers_ * n_expert_), 0);
        for (size_t i = 0; i < upper_has_.size(); ++i) upper_has_[i] = (*upper_->res_)[i] >= 0;
        for (const auto& pr : upper_->pending_) upper_has_[(size_t) pr.first] = 1;
    }
    struct Ranked { float gain; int32_t layer, in, out, slot; };   // out < 0: an empty slot
    std::vector<Ranked> moves;
    std::vector<std::pair<float, int32_t>> cand, vict;
    std::vector<int32_t>& res = *res_;
    for (int64_t l = 0; l < n_layers_; ++l) {
        cand.clear();
        vict.clear();
        const float* u = usage.data() + l * n_expert_;
        const int32_t* r = res.data() + l * n_expert_;
        const uint8_t* up = upper_ != nullptr ? upper_has_.data() + l * n_expert_ : nullptr;
        for (int32_t e = 0; e < (int32_t) n_expert_; ++e) {
            if (r[e] < 0) { if (u[e] >= 2.0f && (up == nullptr || !up[e])) cand.emplace_back(u[e], e); }
            else vict.emplace_back(up != nullptr && up[e] ? -1.0f : u[e], e);   // the upper tier's copy goes first
        }
        if (cand.empty()) continue;
        std::sort(cand.begin(), cand.end(), [](auto& a, auto& b) { return a.first > b.first; });
        const std::vector<int32_t>& fr = free_[(size_t) l];
        size_t c = 0;
        for (; c < cand.size() && c < fr.size(); ++c)
            moves.push_back({cand[c].first, (int32_t) l, cand[c].second, -1, fr[fr.size() - 1 - c]});
        const size_t nv = std::min(cand.size() - c, vict.size());
        std::partial_sort(vict.begin(), vict.begin() + (ptrdiff_t) nv, vict.end(),
                          [](auto& a, auto& b) { return a.first < b.first; });
        for (size_t i = 0; i < nv; ++i, ++c) {
            if (cand[c].first < vict[i].first + 1.5f) break;
            moves.push_back({cand[c].first - vict[i].first, (int32_t) l, cand[c].second, vict[i].second,
                             r[vict[i].second]});
        }
    }
    std::sort(moves.begin(), moves.end(), [](const Ranked& a, const Ranked& b) { return a.gain > b.gain; });
    if ((int) moves.size() > max_moves_) moves.resize((size_t) max_moves_);
    // the copies run beside the next windows; a move counts once admitted (apply_pending)
    queued_.clear();
    next_ = 0;
    for (const Ranked& m : moves) {
        if (m.out >= 0) {
            ++swaps;
        } else {
            std::vector<int32_t>& fr = free_[(size_t) m.layer];
            fr.erase(std::find(fr.begin(), fr.end(), m.slot));
            ++fills;
        }
        queued_.push_back({m.layer, m.in, m.out, m.slot});
        pending_.emplace_back((int32_t) (m.layer * n_expert_ + m.in), m.slot);   // resident once the copy has landed
    }
    if (!paced_ && !pump_n(queued_.size(), err)) return false;
    if (decay)
        for (float& v : usage) v *= 0.7f;
    ms += std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - t0).count();
    return true;
}

void AdaptiveTier::apply_pending(bool wait) {
    if (pending_.empty()) return;
    if (next_ < queued_.size()) {   // paced moves not submitted yet
        if (!wait) return;
        std::string err;
        if (!pump_n(queued_.size(), err)) { failed_ = err; return; }
    }
    if (wait) cudaEventSynchronize(ev_);
    else if (cudaEventQuery(ev_) != cudaSuccess) return;
    std::vector<int32_t>& res = *res_;
    for (const auto& [i, slot] : pending_) res[(size_t) i] = slot;
    pending_.clear();
    if (d_res_ != nullptr) cudaMemcpy(d_res_, res.data(), res.size() * sizeof(int32_t), cudaMemcpyHostToDevice);
}

}  // namespace strata::core
