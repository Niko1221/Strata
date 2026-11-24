#include "runtime.hpp"

#include <cmath>
#include <exception>
#include <limits>
#include <stdexcept>

namespace fnx::sycl_backend {

std::vector<sycl::device> Runtime::check_devices(const std::vector<sycl::device>& devices) {
    if (devices.size() < 2) throw std::runtime_error("SYCL split: at least two cards required");
    for (size_t c = 0; c < devices.size(); ++c) {
        const auto& d = devices[c];
        if (!d.is_gpu() || d.get_backend() != sycl::backend::ext_oneapi_level_zero ||
            d.get_info<sycl::info::device::vendor_id>() != 0x8086 ||
            d.get_platform() != devices[0].get_platform() ||
            !d.has(sycl::aspect::usm_host_allocations) ||
            !d.has(sycl::aspect::usm_device_allocations))
            throw std::runtime_error("SYCL split: cards must be Intel Level Zero GPUs on one platform with host and device USM");
        for (size_t p = 0; p < c; ++p)
            if (d == devices[p]) throw std::runtime_error("SYCL split: duplicate card");
    }
    return devices;
}

Runtime::Runtime(const std::vector<sycl::device>& devices)
    : devices_(check_devices(devices)), context_(devices_) {
    for (const auto& d : devices_)
        queues_.emplace_back(context_, d, [](sycl::exception_list errors) {
            for (const auto& e : errors) std::rethrow_exception(e);
        }, sycl::property::queue::in_order{});
}

DeviceBuffer::DeviceBuffer(Runtime& runtime, size_t card, size_t bytes)
    : queue_(runtime.queue(card)) {
    if (!bytes) throw std::runtime_error("SYCL split: zero-byte device allocation");
    data_ = sycl::malloc_device(bytes, queue_);
    if (!data_) throw std::bad_alloc();
}

DeviceBuffer::~DeviceBuffer() {
    // The owner must outlive the buffer, including its outstanding DMA operations.
    try { queue_.wait_and_throw(); } catch (...) {}
    sycl::free(data_, queue_.get_context());
}

void Runtime::sum(const std::vector<float*>& values, size_t n) {
    if (values.size() != cards())
        throw std::runtime_error("SYCL split: collective card count mismatch");
    if (!n) return;
    if (n > std::numeric_limits<size_t>::max() / sizeof(float) / (cards() + 1))
        throw std::runtime_error("SYCL split: collective allocation size overflow");
    for (size_t c = 0; c < cards(); ++c) {
        if (!values[c] ||
            sycl::get_pointer_type(values[c], context_) != sycl::usm::alloc::device ||
            sycl::get_pointer_device(values[c], context_) != devices_[c])
            throw std::runtime_error("SYCL split: collective buffer is not owned by its card");
    }
    float* staging = sycl::malloc_host<float>((cards() + 1) * n, context_);
    if (!staging) throw std::bad_alloc();
    auto free_host = [&](float* p) { sycl::free(p, context_); };
    std::unique_ptr<float, decltype(free_host)> host(staging, free_host);
    try {
        std::vector<sycl::event> copies;
        for (size_t c = 0; c < cards(); ++c)
            copies.push_back(queues_[c].memcpy(staging + c * n, values[c], n * sizeof(float)));
        sycl::event::wait_and_throw(copies);
        float* result = staging + cards() * n;
        for (size_t i = 0; i < n; ++i) {
            float v = staging[i];
            for (size_t c = 1; c < cards(); ++c) v += staging[c * n + i];
            if (!std::isfinite(v)) throw std::runtime_error("SYCL split: non-finite collective sum");
            result[i] = v;
        }
        copies.clear();
        for (size_t c = 0; c < cards(); ++c)
            copies.push_back(queues_[c].memcpy(values[c], result, n * sizeof(float)));
        sycl::event::wait_and_throw(copies);
    } catch (...) {
        // A submission or asynchronous failure must not free a buffer still in DMA.
        for (auto& q : queues_) {
            try { q.wait_and_throw(); } catch (...) {}
        }
        throw;
    }
}

}  // namespace fnx::sycl_backend
