#pragma once
// src/dsv41/ds4_bounded_io.hpp - Bounded asynchronous streaming and drafter lifecycle control
#include <cstdint>
#include <string>
#include <atomic>

namespace strata::dsv41 {

enum class DrafterMode {
    DisabledUnloaded = 0,
    Auto = 1,
    AlwaysOn = 2
};

struct BoundedIoConfig {
    uint64_t max_pending_io_bytes = 256ULL * 1024 * 1024; // 256 MiB max in-flight
    uint32_t max_pinned_buffers   = 4;
    DrafterMode drafter_mode      = DrafterMode::DisabledUnloaded;
};

class BoundedIoManager {
public:
    explicit BoundedIoManager(const BoundedIoConfig& cfg = BoundedIoConfig{})
        : config_(cfg), current_in_flight_bytes_(0), current_generation_id_(1) {}

    bool is_drafter_unloaded() const {
        return config_.drafter_mode == DrafterMode::DisabledUnloaded;
    }

    uint64_t allocated_drafter_vram_bytes() const {
        if (is_drafter_unloaded()) return 0;
        return 1ULL * 1024 * 1024 * 1024; // 1 GiB for drafter if loaded
    }

    void unload_drafter() {
        config_.drafter_mode = DrafterMode::DisabledUnloaded;
    }

    void load_drafter() {
        config_.drafter_mode = DrafterMode::AlwaysOn;
    }

    bool reserve_io(uint64_t bytes) {
        uint64_t curr = current_in_flight_bytes_.load(std::memory_order_relaxed);
        do {
            // Guard against overflow or exceeding limit
            if (curr + bytes > config_.max_pending_io_bytes || (curr + bytes < curr)) {
                return false;
            }
        } while (!current_in_flight_bytes_.compare_exchange_weak(
            curr, curr + bytes, std::memory_order_acq_rel, std::memory_order_relaxed));
        return true;
    }

    void release_io(uint64_t bytes) {
        uint64_t curr = current_in_flight_bytes_.load(std::memory_order_relaxed);
        uint64_t next = 0;
        do {
            next = (curr >= bytes) ? (curr - bytes) : 0;
        } while (!current_in_flight_bytes_.compare_exchange_weak(
            curr, next, std::memory_order_acq_rel, std::memory_order_relaxed));
    }

    void cancel_pending(uint64_t expected_generation_id) {
        uint64_t expected = expected_generation_id;
        if (current_generation_id_.compare_exchange_strong(
                expected, expected + 1, std::memory_order_acq_rel)) {
            current_in_flight_bytes_.store(0, std::memory_order_release);
        }
    }

    uint64_t active_generation_id() const {
        return current_generation_id_.load(std::memory_order_acquire);
    }

    uint64_t in_flight_bytes() const {
        return current_in_flight_bytes_.load(std::memory_order_acquire);
    }

private:
    BoundedIoConfig config_;
    std::atomic<uint64_t> current_in_flight_bytes_;
    std::atomic<uint64_t> current_generation_id_;
};

} // namespace strata::dsv41
