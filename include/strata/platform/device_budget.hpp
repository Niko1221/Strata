#pragma once

#include <cstdint>

namespace strata::platform {
// Callers serialize reservations, including allocations still in progress. A
// failed allocation returns its reservation; a free returns it only on success.
class DeviceBudget {
public:
    explicit DeviceBudget(uint64_t limit = 0) : limit_(limit) {}
    bool reserve(uint64_t bytes) {
        if (bytes > available()) return false;
        used_ += bytes;
        if (used_ > peak_) peak_ = used_;
        return true;
    }
    bool release(uint64_t bytes) {
        if (bytes > used_) return false;
        used_ -= bytes;
        return true;
    }
    uint64_t available() const { return limit_ - used_; }
    uint64_t used() const { return used_; }
    uint64_t peak() const { return peak_; }
private:
    uint64_t limit_, used_ = 0, peak_ = 0;
};
}  // namespace strata::platform
