#pragma once
#include <algorithm>
#include <cstdint>
#include <limits>
#include <list>
#include <memory>
#include <stdexcept>

namespace strata::core {
struct BranchHandleCounters {
    uint64_t allocations = 0, releases = 0, live_bytes = 0, peak_bytes = 0;
    uint64_t live_entries = 0, peak_entries = 0, issued = 0, lookup_steps = 0;
};

// Counts actual allocator calls and requested byte extents. Allocator rounding,
// the images themselves and the inference engine's other memory are separate.
template<class T> struct BranchHandleAllocator {
    using value_type = T;
    BranchHandleCounters* counters;
    explicit BranchHandleAllocator(BranchHandleCounters& value) : counters(&value) {}
    template<class U> BranchHandleAllocator(const BranchHandleAllocator<U>& other) : counters(other.counters) {}
    T* allocate(size_t n) {
        T* value = std::allocator<T>{}.allocate(n);
        ++counters->allocations;
        counters->live_bytes += n * sizeof(T);
        counters->peak_bytes = std::max(counters->peak_bytes, counters->live_bytes);
        return value;
    }
    void deallocate(T* value, size_t n) {
        std::allocator<T>{}.deallocate(value, n);
        ++counters->releases;
        counters->live_bytes -= n * sizeof(T);
    }
    template<class U> bool operator==(const BranchHandleAllocator<U>& other) const { return counters == other.counters; }
    template<class U> bool operator!=(const BranchHandleAllocator<U>& other) const { return !(*this == other); }
};

// The driver owns this registry on one pipe thread. Images remain the existing
// immutable native image type; handles are only process-local ownership records.
template<class Image> class BranchHandles {
    struct Entry {
        uint64_t id;
        Image image;
    };
public:
    explicit BranchHandles(BranchHandleCounters& counters) : counters_(counters), images_(BranchHandleAllocator<Entry>(counters)) {}
    BranchHandles(const BranchHandles&) = delete;
    BranchHandles& operator=(const BranchHandles&) = delete;
    ~BranchHandles() { clear(); }
    uint64_t insert(Image image) {
        if (!image) throw std::invalid_argument("empty derived checkpoint");
        if (counters_.issued == std::numeric_limits<uint64_t>::max())
            throw std::overflow_error("derived checkpoint handle space exhausted");
        const auto id = counters_.issued;
        images_.push_front({id, std::move(image)});
        ++counters_.issued;
        ++counters_.live_entries;
        counters_.peak_entries = std::max(counters_.peak_entries, counters_.live_entries);
        return id;
    }
    const Image* find(uint64_t id) {
        if (id >= counters_.issued) return nullptr;
        for (const auto& entry : images_) {
            ++counters_.lookup_steps;
            if (entry.id == id) return &entry.image;
        }
        return nullptr;
    }
    bool drop(uint64_t id) {
        if (id >= counters_.issued) return false;
        for (auto it = images_.begin(); it != images_.end(); ++it) {
            ++counters_.lookup_steps;
            if (it->id != id) continue;
            images_.erase(it);
            --counters_.live_entries;
            break;
        }
        // IDs are issued consecutively, never reused. A missing earlier ID was
        // already released; retries need no growing tombstone collection.
        return true;
    }
    void clear() {
        images_.clear();
        counters_.live_entries = 0;
    }
private:
    BranchHandleCounters& counters_;
    std::list<Entry, BranchHandleAllocator<Entry>> images_;
};
} // namespace strata::core
