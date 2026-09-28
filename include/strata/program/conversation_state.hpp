// A bounded, pageable-RAM copy of mutable device and host sequence state.
// Model weights, expert-cache placement and CUDA graph addresses are never copied.
#pragma once
#include <cuda_runtime.h>
#include <cstdint>
#include <cstring>
#include <new>
#include <stdexcept>
#include <vector>

namespace strata::program {
struct ConversationRange {
    void* address;
    size_t bytes;
    bool device;
};

class ConversationState {
    std::vector<ConversationRange> ranges_;
    std::vector<uint8_t> data_;
public:
    size_t bytes() const { return data_.size(); }
    void clear() {
        std::vector<uint8_t>().swap(data_);
        ranges_.clear();
    }
    // available is measured immediately before allocation. Keep physical RAM for
    // the OS and the next request; do not turn a cache hit into a paging storm.
    bool save(const std::vector<ConversationRange>& ranges, uint64_t budget,
              uint64_t available, uint64_t margin) {
        clear();
        uint64_t total = 0;
        for (const auto& r : ranges) {
            if (r.bytes && !r.address) return false;
            if (r.bytes > budget || total > budget - r.bytes) return false;
            total += r.bytes;
        }
        if (total == 0 || total > data_.max_size()) return false;
        if (available < margin || total > available - margin) return false;
        try { data_.resize((size_t) total); ranges_ = ranges; }
        catch (const std::bad_alloc&) { clear(); return false; }
        catch (const std::length_error&) { clear(); return false; }
        if (cudaDeviceSynchronize() != cudaSuccess) { clear(); return false; }
        size_t at = 0;
        for (const auto& r : ranges_) {
            if (!r.bytes) continue;
            if (r.device) {
                if (cudaMemcpy(data_.data() + at, r.address, r.bytes, cudaMemcpyDeviceToHost) != cudaSuccess) {
                    clear(); return false;
                }
            } else std::memcpy(data_.data() + at, r.address, r.bytes);
            at += r.bytes;
        }
        return true;
    }
    bool restore() const {
        if (data_.empty() || cudaDeviceSynchronize() != cudaSuccess) return false;
        size_t at = 0;
        for (const auto& r : ranges_) {
            if (!r.bytes) continue;
            if (r.device) {
                if (cudaMemcpy(r.address, data_.data() + at, r.bytes, cudaMemcpyHostToDevice) != cudaSuccess)
                    return false;
            } else std::memcpy(r.address, data_.data() + at, r.bytes);
            at += r.bytes;
        }
        return cudaDeviceSynchronize() == cudaSuccess;
    }
};
} // namespace strata::program
