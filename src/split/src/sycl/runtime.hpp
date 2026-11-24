#pragma once

#include <sycl/sycl.hpp>

#include <cstddef>
#include <memory>
#include <vector>

namespace fnx::sycl_backend {

class Runtime {
public:
    explicit Runtime(const std::vector<sycl::device>& devices);
    Runtime(const Runtime&) = delete;
    Runtime& operator=(const Runtime&) = delete;

    sycl::queue& queue(size_t card) { return queues_.at(card); }
    const sycl::context& context() const { return context_; }
    size_t cards() const { return queues_.size(); }

    // Copies are ordered after each card's preceding work. The CPU never polls a
    // device flag: host buffers are read only after their DMA events complete.
    void sum(const std::vector<float*>& values, size_t n);

private:
    static std::vector<sycl::device> check_devices(const std::vector<sycl::device>& devices);
    std::vector<sycl::device> devices_;
    sycl::context context_;
    std::vector<sycl::queue> queues_;
};

class DeviceBuffer {
public:
    DeviceBuffer(Runtime& runtime, size_t card, size_t bytes);
    ~DeviceBuffer();
    DeviceBuffer(const DeviceBuffer&) = delete;
    DeviceBuffer& operator=(const DeviceBuffer&) = delete;
    void* data() const { return data_; }

private:
    sycl::queue& queue_;
    void* data_ = nullptr;
};

}  // namespace fnx::sycl_backend
