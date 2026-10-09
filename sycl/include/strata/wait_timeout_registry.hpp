#pragma once

#include <cstdint>
#include <mutex>
#include <stdexcept>
#include <unordered_map>

namespace strata {
inline constexpr int kWaitKinds = 4;
inline constexpr const char* kWaitNames[kWaitKinds] = {
    "other", "flag A (the plan)", "flag B (the PCIe copies)", "flag M (the CPU's results)"
};

class WaitTimeoutRegistry {
public:
    void add(const uint32_t* flag, uint32_t* counter) {
        if (flag == nullptr || counter == nullptr)
            throw std::runtime_error("verify: cannot register a null wait flag or timeout counter");
        std::lock_guard<std::mutex> lock(mutex_);
        if (!counters_.emplace(flag, counter).second)
            throw std::runtime_error("verify: wait flag already registered");
    }

    uint32_t* counter(const uint32_t* flag) {
        std::lock_guard<std::mutex> lock(mutex_);
        const auto it = counters_.find(flag);
        if (it == counters_.end())
            throw std::runtime_error("verify: wait flag has no registered timeout counter");
        return it->second;
    }

    void remove(const uint32_t* flag) {
        std::lock_guard<std::mutex> lock(mutex_);
        counters_.erase(flag);
    }

private:
    std::mutex mutex_;
    std::unordered_map<const uint32_t*, uint32_t*> counters_;
};

inline WaitTimeoutRegistry& wait_timeout_registry() {
    static WaitTimeoutRegistry registry;
    return registry;
}
}  // namespace strata
