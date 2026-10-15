#pragma once

#include <atomic>
#include <cstdint>

namespace strata::core {

// A protocol request records epoch() when queued. STOPs older than that request
// are stale; STOPs arriving after enqueue must survive delayed admission. The
// epoch recheck also closes the race with clearing the old request's flag.
class RequestStop {
public:
    uint64_t epoch() const { return epoch_.load(); }
    bool load() const { return stopped_.load(); }
    void request() {
        epoch_.fetch_add(1);
        stopped_.store(true);
    }
    void begin(uint64_t queued_epoch) {
        stopped_.store(false);
        if (epoch_.load() != queued_epoch) stopped_.store(true);
    }
private:
    std::atomic<uint64_t> epoch_{0};
    std::atomic<bool> stopped_{false};
};

} // namespace strata::core
