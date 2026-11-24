#include "expert_cache.hpp"
#include "../sycl_memory.hpp"

#include <algorithm>
#include <limits>
#include <stdexcept>

namespace fnx::sycl_backend {

namespace {
constexpr size_t empty = std::numeric_limits<size_t>::max();
}

ExpertCache::ExpertCache(Runtime& runtime, size_t card, std::vector<ExpertSource> sources,
                         const std::vector<char>& resident, size_t window_floor)
    : runtime_(runtime), card_(card), sources_(std::move(sources)) {
    if (sources_.empty() || resident.size() != sources_.size() || !window_floor)
        throw std::runtime_error("SYCL expert cache: invalid geometry");
    for (size_t i = 0; i < sources_.size(); ++i) {
        const auto& s = sources_[i];
        if (!s.gate_up || !s.down || !s.gate_up_bytes || !s.down_bytes)
            throw std::runtime_error("SYCL expert cache: missing host mirror");
        auto it = std::find_if(pools_.begin(), pools_.end(), [&](const Pool& p) {
            return p.gu == s.gate_up_bytes && p.dn == s.down_bytes;
        });
        if (it == pools_.end()) {
            pools_.push_back(Pool{s.gate_up_bytes, s.down_bytes, {}, nullptr});
            it = pools_.end() - 1;
        }
        pool_of_.push_back((size_t) (it - pools_.begin()));
        if (resident[i]) it->owner.push_back(empty);
    }
    const size_t count = sources_.size();
    if (count > std::numeric_limits<size_t>::max() / sizeof(char*) / 2)
        throw std::runtime_error("SYCL expert cache: pointer table size overflow");
    bytes_ = 2 * count * sizeof(char*);
    // Validate the entire plan before allocating anything.
    for (const auto& p : pools_) {
        if (p.owner.size() < window_floor)
            throw std::runtime_error("SYCL expert cache: pool cannot hold two windows");
        const size_t stride = sycl_size_add(p.gu, p.dn);
        if (p.owner.size() > std::numeric_limits<size_t>::max() / stride)
            throw std::runtime_error("SYCL expert cache: pool size overflow");
        bytes_ = sycl_size_add(bytes_, p.owner.size() * stride);
    }
    gate_up_ = std::make_unique<DeviceBuffer>(runtime_, card_, count * sizeof(char*));
    down_ = std::make_unique<DeviceBuffer>(runtime_, card_, count * sizeof(char*));
    for (auto& p : pools_)
        p.data = std::make_unique<DeviceBuffer>(runtime_, card_, p.owner.size() * (p.gu + p.dn));
    slot_of_.assign(count, empty);
    gu_ptrs_.assign(count, nullptr);
    dn_ptrs_.assign(count, nullptr);
    std::vector<size_t> initial;
    for (size_t i = 0; i < count; ++i)
        if (resident[i]) initial.push_back(i);
    prepare(initial);
}

void ExpertCache::publish() {
    auto& q = runtime_.queue(card_);
    q.memcpy(gate_up_->data(), gu_ptrs_.data(), gu_ptrs_.size() * sizeof(char*));
    q.memcpy(down_->data(), dn_ptrs_.data(), dn_ptrs_.size() * sizeof(char*));
    q.wait_and_throw();
}

void ExpertCache::prepare(const std::vector<size_t>& experts) {
    if (failed_) throw std::runtime_error("SYCL expert cache: an earlier transfer failed");
    std::vector<char> needed(sources_.size(), 0);
    std::vector<size_t> counts(pools_.size(), 0);
    for (size_t i : experts) {
        if (i >= sources_.size()) throw std::runtime_error("SYCL expert cache: invalid expert index");
        if (!needed[i]) ++counts[pool_of_[i]];
        needed[i] = 1;
    }
    for (size_t p = 0; p < pools_.size(); ++p)
        if (counts[p] > pools_[p].owner.size())
            throw std::runtime_error("SYCL expert cache: working set exceeds its pool's capacity");
    auto& q = runtime_.queue(card_);
    try {
        // A previous window must finish before any slot it read is replaced.
        q.wait_and_throw();
        for (size_t i : experts) {
            if (slot_of_[i] != empty) continue;
            auto& p = pools_[pool_of_[i]];
            auto it = std::find_if(p.owner.begin(), p.owner.end(), [&](size_t old) {
                return old == empty || !needed[old];
            });
            if (it == p.owner.end()) throw std::runtime_error("SYCL expert cache: no replaceable slot");
            const size_t slot = (size_t) (it - p.owner.begin()), old = *it;
            if (old != empty) {
                slot_of_[old] = empty;
                gu_ptrs_[old] = dn_ptrs_[old] = nullptr;
            }
            char* at = (char*) p.data->data() + slot * (p.gu + p.dn);
            q.memcpy(at, sources_[i].gate_up, p.gu);
            q.memcpy(at + p.gu, sources_[i].down, p.dn);
            *it = i;
            slot_of_[i] = slot;
            gu_ptrs_[i] = at;
            dn_ptrs_[i] = at + p.gu;
        }
        publish();
    } catch (...) {
        failed_ = true;
        try { q.wait_and_throw(); } catch (...) {}
        throw;
    }
}

}  // namespace fnx::sycl_backend
