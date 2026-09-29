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
    for (cudaEvent_t e : evs_) if (e) cudaEventDestroy(e);
    cudaEvent_t evs[] = {t0_, t1_};
    for (cudaEvent_t e : evs) if (e) cudaEventDestroy(e);
    if (stream_) cudaStreamDestroy(stream_);
}

bool AdaptiveTier::init(ExpertCache& cache, ExpertSource& src, std::vector<int32_t>& host_res, int64_t n_layers,
                        int64_t n_expert, int max_moves, std::string& err, int device, int main_device) {
    cache_ = &cache;
    src_ = &src;
    res_ = &host_res;
    n_layers_ = n_layers;
    n_expert_ = n_expert;
    max_moves_ = max_moves;
    dev_ = device;
    main_ = main_device;
    free_.assign((size_t) n_layers, {});
    OnDevice on(dev_, main_);
    bool ok = cudaStreamCreateWithFlags(&stream_, cudaStreamNonBlocking) == cudaSuccess &&
              cudaEventCreate(&t0_) == cudaSuccess && cudaEventCreate(&t1_) == cudaSuccess;
    for (cudaEvent_t& e : evs_) ok = ok && cudaEventCreateWithFlags(&e, cudaEventDisableTiming) == cudaSuccess;
    if (!ok) {
        res_ = nullptr;   // off
        err = "adaptive tier: cannot create the refill stream";
    }
    return ok;
}

int64_t AdaptiveTier::free_slots() const {
    int64_t n = 0;
    for (const auto& f : free_) n += (int64_t) f.size();
    return n;
}

uint64_t AdaptiveTier::bytes_of(const Move& m) const {
    return strata::kernels::cpu::expert_layout().blob_bytes(m.layer);
}

void AdaptiveTier::evict(const Move& m) {
    if (m.out >= 0) (*res_)[(size_t) (m.layer * n_expert_ + m.out)] = kNotResident;   // a miss from now on
}

bool AdaptiveTier::copy(const Move& m, uint64_t off, uint64_t n, std::string& err) {
    const uint8_t* b = src_->blob(m.layer, m.in);
    if (b == nullptr ||
        cudaMemcpyAsync(cache_->device_slot(m.slot) + off, b + off, (size_t) n, cudaMemcpyHostToDevice, stream_) !=
            cudaSuccess) {
        err = "adaptive tier: a refill copy failed";
        return false;
    }
    sent_bytes += n;
    return true;
}

// Takes the last timed copies' rate once they have landed; then starts timing these when `worth` (long enough for
// their start and end not to dominate) and none are being timed.
bool AdaptiveTier::time_begin(bool worth) {
    float took = 0;
    if (timed_ > 0 && cudaEventQuery(t1_) == cudaSuccess && cudaEventElapsedTime(&took, t0_, t1_) == cudaSuccess &&
        took > 0) {
        const double r = (double) timed_ / (double) took;
        rate_ = rate_ > 0 ? 0.8 * rate_ + 0.2 * r : r;
        timed_ = 0;
    }
    return worth && timed_ == 0 && cudaEventRecord(t0_, stream_) == cudaSuccess;
}

// After a batch of copies: its timing, and its event in the ring, with the moves sent so far (a full ring's newest
// event takes this batch too).
bool AdaptiveTier::end_batch(bool timed, uint64_t bytes, std::string& err) {
    if (timed) {
        if (cudaEventRecord(t1_, stream_) != cudaSuccess) { err = "adaptive tier: a refill copy failed"; return false; }
        timed_ = bytes;
    }
    if (flying_ < kBatches) ++flying_;
    const int i = (first_ + flying_ - 1) % kBatches;
    batch_sent_[i] = sent_;
    if (cudaEventRecord(evs_[i], stream_) != cudaSuccess) { err = "adaptive tier: a refill copy failed"; return false; }
    return true;
}

void AdaptiveTier::stage(uint64_t bytes) {
    uint64_t have = 0;
    for (size_t i = sent_; i < staged_; ++i) have += bytes_of(queued_[i]);
    have -= off_;
    for (; staged_ < queued_.size() && have < bytes; ++staged_) {
        evict(queued_[staged_]);
        have += bytes_of(queued_[staged_]);
    }
}

bool AdaptiveTier::pump_bytes(uint64_t bytes, std::string& err) {
    if (sent_ >= staged_ || bytes < (64u << 10)) return true;
    OnDevice on(dev_, main_);
    const bool timed = time_begin(bytes >= (256u << 10));
    uint64_t done = 0;
    while (sent_ < staged_ && done < bytes) {
        const Move& m = queued_[sent_];
        const uint64_t blob = bytes_of(m);
        uint64_t n = std::min(bytes - done, blob - off_);
        if (off_ + n < blob) n &= ~(uint64_t) 4095;   // whole pages, but the blob's end
        if (n == 0) break;
        if (!copy(m, off_, n, err)) return false;
        done += n;
        off_ += n;
        if (off_ == blob) {
            off_ = 0;
            ++sent_;
        }
    }
    return done == 0 || end_batch(timed, done, err);
}

bool AdaptiveTier::pump(uint64_t budget, uint64_t& sent, std::string& err) {
    sent = 0;
    size_t n = 0;
    for (; sent_ + n < queued_.size(); ++n) {
        const uint64_t b = bytes_of(queued_[sent_ + n]);
        if (sent + b > budget) break;
        sent += b;
    }
    if (n == 0) return true;
    OnDevice on(dev_, main_);
    // timed from when the work that may read its slots is done
    if (after_ != nullptr && cudaStreamWaitEvent(stream_, after_, 0) != cudaSuccess) {
        err = "adaptive tier: a refill copy failed";
        return false;
    }
    const bool timed = time_begin(n >= 4);   // a batch of a few copies: its start and end dominate
    for (size_t i = 0; i < n; ++i, ++sent_) {
        evict(queued_[sent_]);
        if (!copy(queued_[sent_], 0, bytes_of(queued_[sent_]), err)) return false;
    }
    staged_ = sent_;
    return end_batch(timed, sent, err);
}

bool AdaptiveTier::adapt(std::vector<float>& usage, std::string& err, bool decay) {
    const auto t0 = std::chrono::steady_clock::now();
    if (!failed_.empty()) { err = failed_; return false; }
    apply_pending(false);
    if (wait_ && admitted_ < queued_.size()) return true;
    // the moves whose residents are still in place give way to this call's (ranked again if still worth it)
    for (size_t i = staged_; i < queued_.size(); ++i) {
        const Move& m = queued_[i];
        if (m.out >= 0) {
            --swaps;
        } else {
            free_[(size_t) m.layer].push_back(m.slot);
            --fills;
        }
    }
    dropped += (int64_t) (queued_.size() - staged_);
    queued_.resize(staged_);
    pending_.resize(staged_);
    queued_.erase(queued_.begin(), queued_.begin() + (ptrdiff_t) admitted_);
    pending_.erase(pending_.begin(), pending_.begin() + (ptrdiff_t) admitted_);
    for (int b = 0; b < flying_; ++b) batch_sent_[(first_ + b) % kBatches] -= admitted_;
    sent_ -= admitted_;
    staged_ -= admitted_;
    admitted_ = 0;
    blocked_.assign((size_t) (n_layers_ * n_expert_), 0);   // the experts still under way
    for (const auto& pr : pending_) blocked_[(size_t) pr.first] = 1;
    if (upper_ != nullptr) {
        upper_has_.assign((size_t) (n_layers_ * n_expert_), 0);
        for (size_t i = 0; i < upper_has_.size(); ++i) upper_has_[i] = (*upper_->res_)[i] >= 0;
        for (size_t i = 0; i < upper_->staged_; ++i) upper_has_[(size_t) upper_->pending_[i].first] = 1;
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
        const uint8_t* bl = blocked_.data() + l * n_expert_;
        const uint8_t* up = upper_ != nullptr ? upper_has_.data() + l * n_expert_ : nullptr;
        const int32_t* lo = lower_ != nullptr ? lower_->res_->data() + l * n_expert_ : nullptr;
        for (int32_t e = 0; e < (int32_t) n_expert_; ++e) {
            const float ue = lo != nullptr && lo[e] >= 0 ? 0.5f * u[e] : u[e];
            if (r[e] < 0) { if (ue >= 2.0f && !bl[e] && (up == nullptr || !up[e])) cand.emplace_back(ue, e); }
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
    // a move counts once queued; its resident leaves when it is staged (or sent, paced), its expert comes once landed
    for (const Ranked& m : moves) {
        if (m.out >= 0) {
            ++swaps;
        } else {
            std::vector<int32_t>& fr = free_[(size_t) m.layer];
            fr.erase(std::find(fr.begin(), fr.end(), m.slot));
            ++fills;
        }
        queued_.push_back({m.layer, m.in, m.out, m.slot});
        pending_.emplace_back((int32_t) (m.layer * n_expert_ + m.in), m.slot);
    }
    if (decay)
        for (float& v : usage) v *= 0.7f;
    ms += std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - t0).count();
    return true;
}

void AdaptiveTier::apply_pending(bool wait) {
    if (wait && admitted_ < queued_.size()) {   // send the rest and wait for it
        std::string err;
        uint64_t sent = 0;
        if (!paced_) stage(UINT64_MAX);
        if (!(paced_ ? pump(UINT64_MAX, sent, err) : pump_bytes(UINT64_MAX, err))) {
            failed_ = err;
            return;
        }
        OnDevice on(dev_, main_);
        cudaStreamSynchronize(stream_);
    }
    size_t landed = admitted_;   // the batches landed, oldest first
    for (; flying_ > 0 && cudaEventQuery(evs_[first_]) == cudaSuccess; first_ = (first_ + 1) % kBatches, --flying_)
        landed = std::max(landed, batch_sent_[first_]);
    if (landed == admitted_) return;
    std::vector<int32_t>& res = *res_;
    for (; admitted_ < landed; ++admitted_) res[(size_t) pending_[admitted_].first] = pending_[admitted_].second;
}

}  // namespace strata::core
