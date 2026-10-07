#pragma once

#include <atomic>
#include <cstddef>
#include <cstdlib>
#include <memory>
#include <utility>
#include <cstdint>
#include <limits>
#include <string>
#include <vector>

namespace strata::core::detail {

// Non-owning views: the complement and exchange allocations retain their original
// owners until FileExpertSource::close(). Ownership changes via slot IDs; the
// physical buffer addresses, bytes and CUDA aliases stay in place. This first path requires uniform, fully mapped/pinned expert slots.
class ExchangeStorage {
public:
    struct View {
        uint8_t* host = nullptr;
        const uint8_t* device = nullptr;
    };
    struct Change { size_t in, out, q; const uint8_t* staged; size_t bytes; };

    bool initialize(const std::vector<uint64_t>& offsets, uint8_t* resident_host,
                    const uint8_t* resident_device, uint64_t resident_bytes,
                    uint8_t* exchange_host, const uint8_t* exchange_device,
                    size_t count, size_t blob_bytes, std::string& error, size_t retained_slots = 0) {
        error.clear();
        if (active()) { error = "exchange storage already initialized"; return false; }
        if (!resident_host || !resident_device || !exchange_host || !exchange_device ||
            !blob_bytes || !count || !resident_bytes || resident_bytes % blob_bytes ||
            retained_slots > std::numeric_limits<size_t>::max() - count ||
            count + retained_slots > std::numeric_limits<size_t>::max() / blob_bytes) {
            error = "invalid uniform mapped exchange geometry"; return false;
        }
        const size_t resident_slots = (size_t)(resident_bytes / blob_bytes);
        if (resident_slots > std::numeric_limits<size_t>::max() - count - retained_slots) {
            error = "exchange slot count overflow"; return false;
        }
        auto experts = std::make_unique<std::atomic<size_t>[]>(offsets.size());
        std::vector<size_t> spares(count);
        std::vector<View> buffers(resident_slots + count + retained_slots);
        for (size_t slot = 0; slot < resident_slots; ++slot)
            buffers[slot] = {resident_host + slot * blob_bytes, resident_device + slot * blob_bytes};
        std::vector<bool> used(resident_slots, false);
        for (size_t i = 0; i < offsets.size(); ++i) {
            const uint64_t at = offsets[i];
            experts[i].store(kAbsent, std::memory_order_relaxed);
            if (at == ~uint64_t{0}) continue;
            if (at >= resident_bytes || at % blob_bytes || used[(size_t)(at / blob_bytes)]) {
                error = "invalid or duplicate resident slot"; return false;
            }
            used[(size_t)(at / blob_bytes)] = true;
            experts[i].store((size_t)(at / blob_bytes), std::memory_order_relaxed);
        }
        for (bool present : used) if (!present) {
            error = "unassigned resident slot"; return false;
        }
        for (size_t q = 0; q < count; ++q) {
            spares[q] = resident_slots + q;
            buffers[spares[q]] = {exchange_host + q * blob_bytes, exchange_device + q * blob_bytes};
        }
        std::vector<size_t> free;
        free.reserve(retained_slots + count);
        for (size_t q = count; q < count + retained_slots; ++q) {
            buffers[resident_slots + q] = {exchange_host + q * blob_bytes, exchange_device + q * blob_bytes};
            free.push_back(resident_slots + q);
        }
        if (retained_slots) {
            shadow_.assign(offsets.size(), false);
            prev_.assign(offsets.size(), kAbsent);
            next_.assign(offsets.size(), kAbsent);
        }
        experts_ = std::move(experts);
        expert_count_ = offsets.size();
        spares_.swap(spares);
        buffers_.swap(buffers);
        free_.swap(free);
        retained_capacity_ = retained_slots;
        required_slots_ = resident_slots;
        blob_bytes_ = blob_bytes;
        return true;
    }

    bool active() const { return blob_bytes_ != 0; }
    View resident(size_t expert) const {
        if (expert >= expert_count_) return {};
        const size_t slot = experts_[expert].load(std::memory_order_acquire);
        return slot != kAbsent ? buffers_[slot] : View{};
    }
    View spare(size_t q) const { return q < spares_.size() ? buffers_[spares_[q]] : View{}; }

    // Startup only: fill these unassigned buffers completely before publishing
    // their expert IDs. They are not visible through resident() yet.
    View seed_buffer(size_t q) const {
        return retained_capacity_ && !exchanges_ && !shadow_count_ && !fixed_seed_ && q < free_.size()
                   ? buffers_[free_[q]] : View{};
    }
    bool publish_seeded(const std::vector<size_t>& experts) {
        if (!active() || !retained_capacity_ || exchanges_ || shadow_count_ || fixed_seed_ ||
            experts.empty() || experts.size() > free_.size()) return false;
        std::vector<bool> selected(expert_count_, false);
        for (size_t e : experts) {
            if (e >= expert_count_ || selected[e] || experts_[e].load(std::memory_order_acquire) != kAbsent)
                return false;
            selected[e] = true;
        }
        // The caller has completed every seed copy. Validation above changed no
        // ownership. Retain only these profile-selected keys on later promotions.
        seeded_keys_.swap(selected);
        fixed_seed_ = true;
        for (size_t q = 0; q < experts.size(); ++q) {
            experts_[experts[q]].store(free_[q], std::memory_order_release);
            append_shadow(experts[q]);
        }
        free_.erase(free_.begin(), free_.begin() + experts.size());
        return true;
    }

    // Only after every CPU reader and H2D copy of `in` has completed, and `out`
    // has landed in spare(q). No allocation, memcpy, or CUDA operation here.
    bool commit(size_t in, size_t out, size_t q, const uint8_t* staged, size_t bytes) {
        if (retained_capacity_ || in >= expert_count_ || out >= expert_count_ || q >= spares_.size() ||
            !staged || bytes != blob_bytes_) return false;
        const size_t incoming_slot = experts_[in].load(std::memory_order_acquire);
        if (incoming_slot == kAbsent || experts_[out].load(std::memory_order_acquire) != kAbsent ||
            buffers_[spares_[q]].host != staged) return false;
        // A background router lookahead can query residency to skip redundant
        // file prefetches. Publish one atomic slot ID; the pointer/alias pair is
        // immutable. Actual compute readers still obey the completion contract.
        experts_[out].store(spares_[q], std::memory_order_release);
        spares_[q] = incoming_slot;
        experts_[in].store(kAbsent, std::memory_order_release);
        ++exchanges_;
        avoided_bytes_ += bytes;
        return true;
    }

    // Same completion contract as commit(), plus a whole batch: all outgoing
    // retained copies must become mandatory RAM before any replacement. This
    // prevents discarding an outgoing copy needed later in the batch. Default
    // policy retains promotions in FIFO order; a seeded policy retains only its
    // fixed keys. Extra slots hold immutable duplicates of current GPU experts.
    bool commit_retained(const std::vector<Change>& changes) {
        if (!retained_capacity_ || changes.size() > spares_.size()) return false;
        for (size_t i = 0; i < changes.size(); ++i) {
            const Change& c = changes[i];
            if (c.in >= expert_count_ || c.out >= expert_count_ || c.in == c.out ||
                c.q >= spares_.size() || c.bytes != blob_bytes_ || !c.staged ||
                experts_[c.in].load(std::memory_order_acquire) == kAbsent || shadow_[c.in]) return false;
            const size_t out = experts_[c.out].load(std::memory_order_acquire);
            if (out != kAbsent) {
                if (!shadow_[c.out] || buffers_[out].host != c.staged) return false;
            } else if (shadow_[c.out] || buffers_[spares_[c.q]].host != c.staged) return false;
            for (size_t j = 0; j < i; ++j) {
                const Change& p = changes[j];
                if (p.q == c.q || p.in == c.in || p.out == c.out || p.in == c.out || p.out == c.in) return false;
            }
        }
        // Validation above is read-only. Everything below is allocation-free.
        for (const Change& c : changes) {
            if (shadow_[c.out]) {
                unlink_shadow(c.out); // its bytes already are the authoritative RAM copy
                avoided_d2h_bytes_ += c.bytes;
            } else {
                experts_[c.out].store(spares_[c.q], std::memory_order_release);
                spares_[c.q] = kAbsent; // refill from the spare ownership pool below
            }
        }
        for (const Change& c : changes) {
            if (!fixed_seed_ || seeded_keys_[c.in]) append_shadow(c.in);
            else {
                const size_t slot = experts_[c.in].load(std::memory_order_acquire);
                experts_[c.in].store(kAbsent, std::memory_order_release);
                free_.push_back(slot);
            }
        }
        while (shadow_count_ > retained_capacity_) {
            const size_t e = oldest_;
            const size_t slot = experts_[e].load(std::memory_order_acquire);
            unlink_shadow(e);
            experts_[e].store(kAbsent, std::memory_order_release); // GPU still owns the immutable expert
            free_.push_back(slot);
        }
        for (const Change& c : changes) {
            if (spares_[c.q] == kAbsent) {
                if (free_.empty()) std::abort(); // an ownership invariant, never a recoverable cache miss
                spares_[c.q] = free_.back();
                free_.pop_back();
            }
            ++exchanges_;
            avoided_bytes_ += c.bytes;
        }
        return true;
    }

    size_t retained_capacity() const { return retained_capacity_; }
    size_t retained_count() const { return shadow_count_; }
    bool fixed_seeds() const { return fixed_seed_; }
    uint64_t avoided_d2h_bytes() const { return avoided_d2h_bytes_; }

    // Component/debug check; never called from the timed inference path.
    bool consistent() const {
        if (!active()) return false;
        std::vector<bool> used(buffers_.size(), false), linked(expert_count_, false);
        auto claim = [&](size_t slot) {
            if (slot >= used.size() || used[slot]) return false;
            used[slot] = true; return true;
        };
        size_t mandatory = 0, shadows = 0;
        for (size_t e = 0; e < expert_count_; ++e) {
            const size_t slot = experts_[e].load(std::memory_order_acquire);
            const bool shadow = retained_capacity_ && shadow_[e];
            if (shadow && fixed_seed_ && !seeded_keys_[e]) return false;
            if (slot == kAbsent) { if (shadow) return false; continue; }
            if (!claim(slot)) return false;
            if (shadow) ++shadows; else ++mandatory;
        }
        for (size_t slot : spares_) if (!claim(slot)) return false;
        for (size_t slot : free_) if (!claim(slot)) return false;
        for (bool u : used) if (!u) return false;
        size_t last = kAbsent, count = 0;
        for (size_t e = oldest_; e != kAbsent; e = next_[e]) {
            if (e >= expert_count_ || linked[e] || !shadow_[e] || prev_[e] != last) return false;
            linked[e] = true; last = e; ++count;
        }
        return mandatory == required_slots_ && shadows == shadow_count_ && count == shadows &&
               last == newest_ && shadows <= retained_capacity_ && free_.size() == retained_capacity_ - shadows;
    }

    uint64_t exchanges() const { return exchanges_; }
    uint64_t avoided_bytes() const { return avoided_bytes_; } // memcpy payload, not read+write total
    void clear() {
        experts_.reset(); expert_count_ = 0; buffers_.clear(); spares_.clear();
        blob_bytes_ = 0; exchanges_ = avoided_bytes_ = 0;
        free_.clear(); shadow_.clear(); prev_.clear(); next_.clear();
        seeded_keys_.clear(); fixed_seed_ = false;
        retained_capacity_ = required_slots_ = shadow_count_ = 0;
        oldest_ = newest_ = kAbsent; avoided_d2h_bytes_ = 0;
    }

private:
    static constexpr size_t kAbsent = std::numeric_limits<size_t>::max();
    std::unique_ptr<std::atomic<size_t>[]> experts_;
    size_t expert_count_ = 0;
    std::vector<View> buffers_; // immutable physical addresses after initialize
    std::vector<size_t> spares_; // touched only by the serialized cache manager
    size_t blob_bytes_ = 0;
    uint64_t exchanges_ = 0, avoided_bytes_ = 0;
    size_t retained_capacity_ = 0, required_slots_ = 0, shadow_count_ = 0;
    size_t oldest_ = kAbsent, newest_ = kAbsent;
    uint64_t avoided_d2h_bytes_ = 0;
    std::vector<size_t> free_, prev_, next_;
    std::vector<bool> shadow_; // only the serialized cache manager reads/writes this list
    std::vector<bool> seeded_keys_;
    bool fixed_seed_ = false;

    void unlink_shadow(size_t e) {
        if (prev_[e] != kAbsent) next_[prev_[e]] = next_[e]; else oldest_ = next_[e];
        if (next_[e] != kAbsent) prev_[next_[e]] = prev_[e]; else newest_ = prev_[e];
        shadow_[e] = false; prev_[e] = next_[e] = kAbsent; --shadow_count_;
    }
    void append_shadow(size_t e) {
        shadow_[e] = true; prev_[e] = newest_; next_[e] = kAbsent;
        if (newest_ != kAbsent) next_[newest_] = e; else oldest_ = e;
        newest_ = e; ++shadow_count_;
    }
};

} // namespace strata::core::detail
