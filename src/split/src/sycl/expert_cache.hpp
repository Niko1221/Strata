#pragma once

#include "runtime.hpp"

#include <cstddef>
#include <memory>
#include <utility>
#include <vector>

namespace fnx::sycl_backend {

struct ExpertSource {
    const void* gate_up;
    const void* down;
    size_t gate_up_bytes;
    size_t down_bytes;
};

// Host mirrors belong to the loader and must outlive the cache. Only device
// pointers are published: no kernel can accidentally dereference a cold mirror.
class ExpertCache {
public:
    ExpertCache(Runtime& runtime, size_t card, std::vector<ExpertSource> sources,
                const std::vector<char>& resident, size_t window_floor);
    void prepare(const std::vector<size_t>& experts);
    const char* const* gate_up() const { return (const char* const*) gate_up_->data(); }
    const char* const* down() const { return (const char* const*) down_->data(); }
    size_t bytes() const { return bytes_; }

private:
    struct Pool {
        size_t gu, dn;
        std::vector<size_t> owner;
        std::unique_ptr<DeviceBuffer> data;
    };
    Runtime& runtime_;
    size_t card_, bytes_ = 0;
    bool failed_ = false;
    std::vector<ExpertSource> sources_;
    std::vector<Pool> pools_;
    std::vector<size_t> pool_of_, slot_of_;
    std::vector<const char*> gu_ptrs_, dn_ptrs_;
    std::unique_ptr<DeviceBuffer> gate_up_, down_;
    void publish();
};

}  // namespace fnx::sycl_backend
