#pragma once

#include <atomic>
#include <algorithm>
#include <cstddef>
#include <memory>
#include <utility>
#include <cstdint>
#include <limits>
#include <string>
#include <vector>

namespace strata::core::detail {

// Non-owning views: the complement and exchange allocations retain their original
// owners until FileExpertSource::close(). Ownership changes via slot IDs; the
// physical buffer addresses, capacities and CUDA aliases stay in place.
// Mixed-size buckets require a fully mapped/pinned complement.
class ExchangeStorage {
public:
    struct View {
        uint8_t* host = nullptr;
        const uint8_t* device = nullptr;
        size_t bytes = 0;
    };

    bool initialize(const std::vector<uint64_t>& offsets, uint8_t* resident_host,
                    const uint8_t* resident_device, uint64_t resident_bytes,
                    uint8_t* exchange_host, const uint8_t* exchange_device,
                    size_t count, size_t blob_bytes, std::string& error) {
        error.clear();
        if (active()) { error = "exchange storage already initialized"; return false; }
        if (!resident_host || !resident_device || !exchange_host || !exchange_device ||
            !blob_bytes || !count || !resident_bytes || resident_bytes % blob_bytes ||
            count > std::numeric_limits<size_t>::max() / blob_bytes) {
            error = "invalid uniform mapped exchange geometry"; return false;
        }
        const size_t resident_slots = (size_t)(resident_bytes / blob_bytes);
        if (resident_slots > std::numeric_limits<size_t>::max() - count) {
            error = "exchange slot count overflow"; return false;
        }
        auto experts = std::make_unique<std::atomic<size_t>[]>(offsets.size());
        std::vector<size_t> spares(count);
        std::vector<View> buffers(resident_slots + count);
        for (size_t slot = 0; slot < resident_slots; ++slot)
            buffers[slot] = {resident_host + slot * blob_bytes, resident_device + slot * blob_bytes, blob_bytes};
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
            buffers[spares[q]] = {exchange_host + q * blob_bytes, exchange_device + q * blob_bytes, blob_bytes};
        }
        experts_ = std::move(experts);
        expert_count_ = offsets.size();
        spares_.swap(spares);
        buffers_.swap(buffers);
        blob_bytes_ = blob_bytes;
        return true;
    }

    // Each immutable physical view has its own capacity. A same-size bucket
    // retains `count` spares even when ownership rotates into the compact arena.
    bool initialize_mixed(const std::vector<uint64_t>& offsets, const std::vector<size_t>& sizes,
                          uint8_t* rh, const uint8_t* rd, uint64_t resident_bytes,
                          uint8_t* sh, const uint8_t* sd, size_t count, std::string& error) {
        error.clear();
        if(active() || offsets.size()!=sizes.size() || !rh || !rd || !sh || !sd || !count || !resident_bytes) {
            error="invalid mixed mapped exchange geometry";return false;
        }
        std::vector<size_t> types=sizes;
        std::sort(types.begin(),types.end());types.erase(std::unique(types.begin(),types.end()),types.end());
        if(types.empty() || types[0]==0 || types.size()>std::numeric_limits<size_t>::max()/count) {
            error="invalid bucket count or size";return false;
        }
        size_t pool_bytes=0;
        for(size_t b:types) {
            if(b>std::numeric_limits<size_t>::max()/count || pool_bytes>std::numeric_limits<size_t>::max()-b*count) {
                error="bucket allocation overflow";return false;
            }
            pool_bytes+=b*count;
        }
        auto experts=std::make_unique<std::atomic<size_t>[]>(offsets.size());
        std::vector<View> buffers;std::vector<size_t> spares;
        std::vector<std::pair<uint64_t,uint64_t>> spans;
        for(size_t i=0;i<offsets.size();++i) {
            experts[i].store(kAbsent,std::memory_order_relaxed);
            if(offsets[i]==~uint64_t{0})continue;
            if(offsets[i]>resident_bytes || sizes[i]>resident_bytes-offsets[i]) {
                error="resident view out of bounds";return false;
            }
            spans.push_back({offsets[i],offsets[i]+sizes[i]});
            experts[i].store(buffers.size(),std::memory_order_relaxed);
            buffers.push_back({rh+offsets[i],rd+offsets[i],sizes[i]});
        }
        std::sort(spans.begin(),spans.end());
        for(size_t i=1;i<spans.size();++i)if(spans[i].first<spans[i-1].second) {
            error="overlapping resident views";return false;
        }
        size_t at=0;
        for(size_t b:types)for(size_t q=0;q<count;++q) {
            spares.push_back(buffers.size());buffers.push_back({sh+at,sd+at,b});at+=b;
        }
        experts_=std::move(experts);expert_count_=offsets.size();buffers_.swap(buffers);spares_.swap(spares);
        blob_bytes_=types.back();return true;
    }
    size_t spare_index_for(size_t bytes,size_t q) const {
        for(size_t i=0;i<spares_.size();++i)
            if(buffers_[spares_[i]].bytes==bytes && q--==0)return i;
        return kAbsent;
    }
    View spare_for(size_t bytes,size_t q) const {
        const size_t i=spare_index_for(bytes,q);return i==kAbsent?View{}:spare(i);
    }
    bool commit_for(size_t in,size_t out,size_t q,const uint8_t* staged,size_t bytes) {
        const size_t i=spare_index_for(bytes,q);return i!=kAbsent&&commit(in,out,i,staged,bytes);
    }

    bool active() const { return blob_bytes_ != 0; }
    View resident(size_t expert) const {
        if (expert >= expert_count_) return {};
        const size_t slot = experts_[expert].load(std::memory_order_acquire);
        return slot != kAbsent ? buffers_[slot] : View{};
    }
    View spare(size_t q) const { return q < spares_.size() ? buffers_[spares_[q]] : View{}; }

    // Only after every CPU reader and H2D copy of `in` has completed, and `out`
    // has landed in spare(q). No allocation, memcpy, or CUDA operation here.
    bool commit(size_t in, size_t out, size_t q, const uint8_t* staged, size_t bytes) {
        if (in >= expert_count_ || out >= expert_count_ || q >= spares_.size() ||
            !staged || !bytes) return false;
        const size_t incoming_slot = experts_[in].load(std::memory_order_acquire);
        if (incoming_slot == kAbsent || buffers_[incoming_slot].bytes != bytes ||
            buffers_[spares_[q]].bytes != bytes || experts_[out].load(std::memory_order_acquire) != kAbsent ||
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

    uint64_t exchanges() const { return exchanges_; }
    uint64_t avoided_bytes() const { return avoided_bytes_; } // memcpy payload, not read+write total
    void clear() {
        experts_.reset(); expert_count_ = 0; buffers_.clear(); spares_.clear();
        blob_bytes_ = 0; exchanges_ = avoided_bytes_ = 0;
    }

private:
    static constexpr size_t kAbsent = std::numeric_limits<size_t>::max();
    std::unique_ptr<std::atomic<size_t>[]> experts_;
    size_t expert_count_ = 0;
    std::vector<View> buffers_; // immutable physical addresses after initialize
    std::vector<size_t> spares_; // touched only by the serialized cache manager
    size_t blob_bytes_ = 0;
    uint64_t exchanges_ = 0, avoided_bytes_ = 0;
};

static_assert(sizeof(ExchangeStorage)==88,"preserve FileExpertSource ABI on this x86_64 experiment");
} // namespace strata::core::detail
