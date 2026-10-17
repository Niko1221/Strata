#pragma once

#include <cstddef>
#include <cstdint>
#include <functional>
#include <list>
#include <map>
#include <memory>
#include <stdexcept>
#include <vector>

namespace strata::core {

// Host-controller-owned cache. A lease keeps a blob alive through CPU jobs or asynchronous GPU copies;
// leased entries cannot be evicted and count against the same byte budget as idle entries.
class BoundedExpertCache {
public:
    class Blob {
    public:
        explicit Blob(size_t bytes)
            : bytes_(bytes), memory_(new uint8_t[bytes](), std::default_delete<uint8_t[]>()) {}
        Blob(size_t bytes, std::shared_ptr<uint8_t> memory) : bytes_(bytes), memory_(std::move(memory)) {}
        size_t size() const { return bytes_; }
        uint8_t* data() { return memory_.get(); }
        const uint8_t* data() const { return memory_.get(); }
        const uint8_t& operator[](size_t i) const { return memory_.get()[i]; }
    private:
        size_t bytes_;
        std::shared_ptr<uint8_t> memory_;
    };
    using Lease = std::shared_ptr<const Blob>;
    using Allocate = std::function<std::shared_ptr<Blob>(size_t)>;

    explicit BoundedExpertCache(size_t budget, Allocate allocate = {}) : budget_(budget), allocate_(std::move(allocate)) {
        if (!budget) throw std::invalid_argument("expert RAM cache budget must be positive");
    }

    template<class Fill>
    Lease get(uint64_t key, size_t bytes, Fill fill) {
        auto found = entries_.find(key);
        if (found != entries_.end()) {
            if (found->second.blob->size() != bytes) throw std::logic_error("expert cache key changed size");
            order_.splice(order_.begin(), order_, found->second.order);
            ++hits_;
            return found->second.blob;
        }
        if (bytes > budget_) throw std::runtime_error("one expert exceeds the RAM cache budget");
        while (resident_ > budget_ - bytes) {
            bool evicted = false;
            for (auto it = order_.end(); it != order_.begin();) {
                --it;
                auto victim = entries_.find(*it);
                if (victim->second.blob.use_count() != 1) continue;
                resident_ -= victim->second.blob->size();
                entries_.erase(victim);
                order_.erase(it);
                ++evictions_;
                evicted = true;
                break;
            }
            if (!evicted) throw std::runtime_error("expert RAM cache budget is exhausted by live readers");
        }
        auto blob = allocate_ ? allocate_(bytes) : std::make_shared<Blob>(bytes);
        if (!blob || blob->size() != bytes) throw std::runtime_error("expert cache allocator returned a wrong-sized blob");
        fill(blob->data());
        order_.push_front(key);
        try {
            entries_.emplace(key, Entry{blob, order_.begin()});
        } catch (...) {
            order_.pop_front();
            throw;
        }
        resident_ += bytes;
        if (resident_ > peak_) peak_ = resident_;
        ++misses_;
        fetched_ += bytes;
        return blob;
    }

    size_t budget() const { return budget_; }
    size_t resident() const { return resident_; }
    size_t peak() const { return peak_; }
    uint64_t hits() const { return hits_; }
    uint64_t misses() const { return misses_; }
    uint64_t evictions() const { return evictions_; }
    uint64_t fetched() const { return fetched_; }

private:
    struct Entry { Lease blob; std::list<uint64_t>::iterator order; };
    size_t budget_, resident_ = 0, peak_ = 0;
    uint64_t hits_ = 0, misses_ = 0, evictions_ = 0, fetched_ = 0;
    std::list<uint64_t> order_;
    std::map<uint64_t, Entry> entries_;
    Allocate allocate_;
};

} // namespace strata::core
