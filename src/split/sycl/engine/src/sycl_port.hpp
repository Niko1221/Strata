#pragma once

#include <sycl/sycl.hpp>
#include <dpct/dpct.hpp>
#include "strata/sycl_doorbell.hpp"

#include <algorithm>
#include <atomic>
#include <chrono>
#include <condition_variable>
#include <cstdlib>
#include <exception>
#include <map>
#include <mutex>
#include <stdexcept>
#include <string>
#include <thread>
#include <type_traits>
#include <utility>
#include <unordered_map>
#include <vector>

namespace fnx::sycl_port {

struct HostTrace {
    struct Metric { size_t calls = 0; double ms = 0; };
    bool enabled = false;
    std::map<std::string, Metric> metrics;
};

class HostScope {
    HostTrace* trace_;
    const char* name_;
    std::chrono::steady_clock::time_point began_;
public:
    HostScope(HostTrace* trace, const char* name)
        : trace_(trace && trace->enabled ? trace : nullptr), name_(name),
          began_(trace_ ? std::chrono::steady_clock::now()
                        : std::chrono::steady_clock::time_point{}) {}
    ~HostScope() {
        if (trace_) {
            const double ms = std::chrono::duration<double, std::milli>(
                std::chrono::steady_clock::now() - began_).count();
            auto& metric = trace_->metrics[name_];
            ++metric.calls;
            metric.ms += ms;
        }
    }
};

constexpr unsigned spin_limit = 20000;

template <typename... Args>
inline void* alloc_device(size_t bytes, Args&&... args) {
    void* p = sycl::malloc_device(bytes, std::forward<Args>(args)...);
    if (!p && bytes)
        throw std::runtime_error("SYCL split: device USM allocation failed for " +
                                 std::to_string(bytes) + " bytes");
    return p;
}

template <typename... Args>
inline void* alloc_host(size_t bytes, Args&&... args) {
    void* p = sycl::malloc_host(bytes, std::forward<Args>(args)...);
    if (!p && bytes)
        throw std::runtime_error("SYCL split: host USM allocation failed for " +
                                 std::to_string(bytes) + " bytes");
    return p;
}

inline void mma_i8(int* c, const uint32_t* a, const uint32_t* b) {
    volatile void* out[4] = {c, c + 1, c + 2, c + 3};
    sycl::vec<uint32_t, 4> av(a[0], a[1], a[2], a[3]);
    sycl::vec<uint32_t, 2> bv(b[0], b[1]);
    sycl::vec<int32_t, 4> cv(c[0], c[1], c[2], c[3]);
    dpct::experimental::matrix::mma<16, 8, 32, int8_t, int32_t>(
        out, &av, &bv, &cv);
}

inline void mma_u8s8(int* c, const uint32_t* a, const uint32_t* b) {
    uint32_t low[4], high[4];
    for (int i = 0; i < 4; ++i) {
        low[i] = a[i] & 0x7f7f7f7f;
        high[i] = (a[i] >> 7) & 0x01010101;
    }
    int carry[4]{};
    mma_i8(c, low, b);
    mma_i8(carry, high, b);
    for (int i = 0; i < 4; ++i) c[i] += 128 * carry[i];
}

inline unsigned long long load_word(const unsigned long long* p) {
    sycl::atomic_fence(sycl::memory_order::acquire, sycl::memory_scope::system);
    return strata::sys_load_mapped(p);
}

inline void store_word(unsigned long long* p, unsigned long long v) {
    strata::sys_store_mapped(p, v);
}

class StreamBarrier;

inline sycl::event stream_barrier(sycl::queue* q,
                                 const std::vector<sycl::event>& dependencies = {}) {
    if (!q->has_property<sycl::property::queue::in_order>())
        throw std::invalid_argument("SYCL split: stream barriers require in-order queues");
    // Native barriers can block submission while holding the scheduler lock behind a host_task.
    return q->submit([&](sycl::handler& handler) {
        handler.depends_on(dependencies);
        handler.single_task<StreamBarrier>([] {});
    });
}

class ZeroDevice;

inline sycl::event stream_tail(sycl::queue* q) {
    if (!q->has_property<sycl::property::queue::in_order>())
        throw std::invalid_argument("SYCL split: tail events require in-order queues");
    auto last = q->ext_oneapi_get_last_event();
    return [&](auto&& event) -> sycl::event {
        if constexpr (std::is_same_v<std::decay_t<decltype(event)>, sycl::event>)
            return event;
        else {
            if (event.has_value()) return event.value();
            return stream_barrier(q);
        }
    }(last);
}

inline sycl::event zero_device(sycl::queue* q, void* pointer, size_t bytes) {
    const char* mode = std::getenv("FNX_KERNEL_ZERO");
    if (!mode || std::atoi(mode) == 0) return q->memset(pointer, 0, bytes);
    if (!bytes) return stream_barrier(q);
    auto* out = static_cast<unsigned char*>(pointer);
    const bool aligned = (reinterpret_cast<uintptr_t>(pointer) & 3) == 0;
    const size_t workers = std::min((bytes + 3) / 4, size_t(256 * 1024));
    return q->parallel_for<ZeroDevice>(sycl::range<1>(workers), [=](sycl::id<1> id) {
        for (size_t at = id[0] * 4; at < bytes; at += workers * 4) {
            if (aligned && bytes - at >= 4)
                *reinterpret_cast<uint32_t*>(out + at) = 0;
            else
                for (size_t j = at; j < bytes && j < at + 4; ++j) out[j] = 0;
        }
    });
}

class EventMailbox {
    struct Entry {
        unsigned sequence;
        sycl::event event;
    };
    std::mutex mutex_;
    std::condition_variable ready_;
    std::unordered_map<const unsigned*, Entry> entries_;
    std::exception_ptr error_;
public:
    void cancel(std::exception_ptr error) {
        {
            std::lock_guard<std::mutex> lock(mutex_);
            if (!error_) error_ = error;
        }
        ready_.notify_all();
    }

    void publish(const unsigned* address, unsigned sequence, const sycl::event& event) {
        {
            std::lock_guard<std::mutex> lock(mutex_);
            if (error_) std::rethrow_exception(error_);
            entries_.insert_or_assign(address, Entry{sequence, event});
        }
        ready_.notify_all();
    }

    void signal(sycl::queue* q, const unsigned* address, unsigned sequence,
                HostTrace* trace = nullptr) {
        const char* tail = std::getenv("FNX_PF_EVENT_TAIL");
        sycl::event recorded;
        {
            HostScope scope(trace, "mailbox.record");
            recorded = tail && std::atoi(tail) != 0 ? stream_tail(q) : stream_barrier(q);
        }
        HostScope scope(trace, "mailbox.publish");
        publish(address, sequence, recorded);
    }

    sycl::event event(const unsigned* address, unsigned sequence, HostTrace* trace = nullptr) {
        HostScope scope(trace, "mailbox.peer_publication");
        sycl::event event;
        {
            std::unique_lock<std::mutex> lock(mutex_);
            const bool published = ready_.wait_for(lock, std::chrono::seconds(120), [&] {
                const auto it = entries_.find(address);
                return error_ || (it != entries_.end() &&
                    int(it->second.sequence - sequence) >= 0);
            });
            if (error_) std::rethrow_exception(error_);
            if (!published)
                throw std::runtime_error("SYCL prefill: peer did not publish its event");
            const auto& entry = entries_.at(address);
            if (entry.sequence != sequence)
                throw std::runtime_error("SYCL prefill: peer event was overwritten before consumption");
            event = entry.event;
        }
        return event;
    }

    void wait(sycl::queue* q, const unsigned* address, unsigned sequence, HostTrace* trace = nullptr) {
        // Only event publication waits on the host; GPU completion stays asynchronous.
        const auto source = event(address, sequence, trace);
        HostScope scope(trace, "mailbox.wait_submit");
        stream_barrier(q, {source});
    }
};

inline std::mutex error_mutex;
inline std::vector<std::pair<sycl::context, unsigned*>> errors;

inline unsigned* error_word(const sycl::context& context) {
    std::lock_guard<std::mutex> lock(error_mutex);
    for (const auto& entry : errors)
        if (entry.first == context) return entry.second;
    auto* p = sycl::malloc_host<unsigned>(1, context);
    if (!p) throw std::bad_alloc();
    *p = 0;
    errors.emplace_back(context, p);
    return p;
}

inline void check_errors() {
    std::lock_guard<std::mutex> lock(error_mutex);
    for (const auto& entry : errors)
        if (__atomic_load_n(entry.second, __ATOMIC_ACQUIRE))
            throw std::runtime_error("SYCL split: stream flag timed out");
}

}  // namespace fnx::sycl_port

constexpr unsigned CU_STREAM_WAIT_VALUE_GEQ = 0;

inline int cuStreamWriteValue32(dpct::queue_ptr q, dpct::device_ptr address,
                                 unsigned value, unsigned flags) {
    if (flags) throw std::runtime_error("SYCL split: unsupported stream write flags");
    auto* p = reinterpret_cast<unsigned*>(address);
    q->single_task([=] { strata::sys_store(p, value); });
    return 0;
}

inline int cuStreamWaitValue32(dpct::queue_ptr q, dpct::device_ptr address,
                                unsigned value, unsigned flags) {
    if (flags) throw std::runtime_error("SYCL split: unsupported stream wait flags");
    auto* p = reinterpret_cast<const unsigned*>(address);
    const auto kind = sycl::get_pointer_type(p, q->get_context());
    if (kind != sycl::usm::alloc::host && kind != sycl::usm::alloc::shared)
        throw std::invalid_argument("SYCL split: stream flags must be host-accessible USM");
    unsigned* error = fnx::sycl_port::error_word(q->get_context());
    q->submit([=](sycl::handler& handler) {
        handler.host_task([=] {
            const auto deadline = std::chrono::steady_clock::now() + std::chrono::seconds(120);
            while ((int) (__atomic_load_n(p, __ATOMIC_ACQUIRE) - value) < 0) {
                if (__atomic_load_n(error, __ATOMIC_ACQUIRE)) return;
                if (std::chrono::steady_clock::now() >= deadline) {
                    __atomic_store_n(error, 1u, __ATOMIC_RELEASE);
                    return;
                }
                std::this_thread::sleep_for(std::chrono::microseconds(50));
            }
        });
    });
    return 0;
}
