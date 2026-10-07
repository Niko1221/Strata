// src/glm/experts.cpp - the RAM expert cache.  See the header.
#include "strata/glm/experts.hpp"

#include <chrono>
#include <stdexcept>

namespace strata::glm {

ExpertCache::ExpertCache(IoPool& io, const GlmConfig& c, std::vector<std::vector<ExpertSpan>> spans, uint64_t budget)
    : io_(io), c_(c), spans_(std::move(spans)) {
    const uint64_t plane = (uint64_t) c.hidden * c.moe_inter;
    for (int k = 0; k < 6; ++k) {
        plane_bytes_[k] = k < 3 ? plane / 2 : plane / 64 * 4;
        region_[k] = slot_bytes_;
        slot_bytes_ += (plane_bytes_[k] + 2 * kAlign + kAlign - 1) & ~(kAlign - 1);
    }
    // a layer's request (top-k experts for every row of a prompt chunk) must fit with room to evict: at least
    // n_experts slots, so even a long prompt that touches every expert of a layer can pin them all at once
    const uint64_t min_slots = (uint64_t) c.n_experts + 2ull * c.topk;
    const uint64_t n = std::max<uint64_t>(min_slots, budget / slot_bytes_);
    slots_.reserve((size_t) n);
    for (uint64_t i = 0; i < n; ++i) slots_.push_back(std::make_unique<Slot>());
}

ExpertCache::~ExpertCache() {
    // wait for reads in flight: their completion writes into the slots
    std::unique_lock<std::mutex> lk(m_);
    cv_.wait(lk, [&] {
        for (auto& s : slots_)
            if (s->state.load() == Loading) return false;
        return true;
    });
    for (auto& s : slots_) aligned_free_bytes(s->mem);
}

int ExpertCache::find_victim() {
    int best = -1;
    uint64_t best_use = ~0ull;
    for (int i = 0; i < (int) slots_.size(); ++i) {
        Slot& s = *slots_[i];
        if (s.mem == nullptr) return i;   // never used: free capacity first
        if (s.pins > 0 || s.state.load(std::memory_order_acquire) == Loading) continue;
        if (s.last_use < best_use) { best_use = s.last_use; best = i; }
    }
    return best;
}

void ExpertCache::start_read(int si, int layer, int e) {
    Slot& s = *slots_[si];
    const ExpertSpan& sp = spans_[layer][e];
    s.state.store(Loading, std::memory_order_release);
    s.parts.store(sp.n_runs, std::memory_order_relaxed);
    auto done = [this, si](bool ok) {
        Slot& d = *slots_[si];
        if (!ok) d.state.store(Failed, std::memory_order_release);
        if (d.parts.fetch_sub(1, std::memory_order_acq_rel) == 1) {
            int expect = Loading;
            d.state.compare_exchange_strong(expect, Ready, std::memory_order_acq_rel);
            std::lock_guard<std::mutex> lk(m_);
            cv_.notify_all();
        }
    };
    for (int r = 0; r < sp.n_runs; ++r) {
        const ExpertSpan::Run& run = sp.runs[r];
        const AlignedSpan a = aligned_span(run.off, run.bytes);
        uint8_t* dst = s.mem + region_[run.first];
        for (int k = 0; k < run.count; ++k) s.plane[run.first + k] = dst + a.skip + (uint64_t) k * plane_bytes_[run.first];
    }
    for (int r = 0; r < sp.n_runs; ++r) {
        const ExpertSpan::Run& run = sp.runs[r];
        const AlignedSpan a = aligned_span(run.off, run.bytes);
        io_.submit(ReadJob{run.shard, a.off, a.len, s.mem + region_[run.first], done});
    }
}

int ExpertCache::acquire(int layer, int e, bool pin) {
    if (layer < 0 || layer >= (int) spans_.size() || e < 0 || e >= (int) spans_[layer].size())
        throw std::runtime_error("expert request out of range");
    const uint64_t k = key(layer, e);
    auto it = where_.find(k);
    if (it != where_.end()) {
        Slot& s = *slots_[it->second];
        if (s.state.load(std::memory_order_acquire) != Failed) {
            if (pin) {
                ++s.pins;
                s.last_use = ++clock_;
                ++stats_.hits;
                if (s.prefetched) { ++stats_.prefetch_used; s.prefetched = false; }
            }
            return it->second;
        }
        where_.erase(it);   // a failed read is retried in a fresh slot
    }
    const int v = find_victim();
    if (v < 0) {
        if (!pin) return -1;   // a prefetch never blocks or evicts a pinned slot
        throw std::runtime_error("expert cache: every slot is pinned (raise the RAM budget)");
    }
    Slot& s = *slots_[v];
    if (s.mem == nullptr) {
        s.mem = (uint8_t*) aligned_alloc_bytes(slot_bytes_);
        if (!s.mem) throw std::runtime_error("expert cache: out of memory allocating a slot");
    }
    if (s.layer >= 0) where_.erase(key(s.layer, s.e));
    s.layer = layer;
    s.e = e;
    s.pins = pin ? 1 : 0;
    s.last_use = ++clock_;
    s.prefetched = !pin;
    where_[k] = v;
    if (pin) ++stats_.misses; else ++stats_.prefetched;
    start_read(v, layer, e);
    return v;
}

void ExpertCache::request(int layer, const int* experts, int n, int* slots) {
    std::lock_guard<std::mutex> lk(m_);   // only to order with the I/O threads' notify; callers are one thread
    for (int i = 0; i < n; ++i) {
        ++stats_.requests;
        slots[i] = acquire(layer, experts[i], true);
    }
}

void ExpertCache::prefetch(int layer, const int* experts, int n) {
    std::lock_guard<std::mutex> lk(m_);
    for (int i = 0; i < n; ++i) acquire(layer, experts[i], false);
}

bool ExpertCache::ready(int slot) const { return slots_[slot]->state.load(std::memory_order_acquire) == Ready; }

bool ExpertCache::wait(int slot) {
    Slot& s = *slots_[slot];
    int st = s.state.load(std::memory_order_acquire);
    if (st == Ready) return true;
    const auto t0 = std::chrono::steady_clock::now();
    std::unique_lock<std::mutex> lk(m_);
    cv_.wait(lk, [&] { st = s.state.load(std::memory_order_acquire); return st == Ready || st == Failed; });
    stats_.wait_us += (uint64_t) std::chrono::duration_cast<std::chrono::microseconds>(
                          std::chrono::steady_clock::now() - t0).count();
    return st == Ready;
}

ExpertView ExpertCache::view(int slot) const {
    const Slot& s = *slots_[slot];
    ExpertView v;   // plane order on disk: down, gate, up (codes), then the same for the scales
    v.down = Q4{c_.hidden, c_.moe_inter, s.plane[0], (const float*) s.plane[3]};
    v.gate = Q4{c_.moe_inter, c_.hidden, s.plane[1], (const float*) s.plane[4]};
    v.up = Q4{c_.moe_inter, c_.hidden, s.plane[2], (const float*) s.plane[5]};
    return v;
}

void ExpertCache::release(int slot) {
    std::lock_guard<std::mutex> lk(m_);
    Slot& s = *slots_[slot];
    if (s.pins > 0) --s.pins;
}

}  // namespace strata::glm
