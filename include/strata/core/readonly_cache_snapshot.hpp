#pragma once

#include <cuda_runtime.h>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <limits>

namespace strata::core {

// Metadata for immutable secondary expert copies, sampled at a completed
// verifier-window boundary. The owner invalidates it before another window
// can mutate cache tags/data. A returned device source must finish being read
// before that next mutation; this object does not pin a GPU cache entry.
class ReadonlyCacheSnapshot {
public:
    ReadonlyCacheSnapshot() = default;
    ReadonlyCacheSnapshot(const ReadonlyCacheSnapshot&) = delete;
    ReadonlyCacheSnapshot& operator=(const ReadonlyCacheSnapshot&) = delete;
    ~ReadonlyCacheSnapshot() { close(); }

    cudaError_t open(int64_t layers, int64_t experts, int ways, size_t stride,
                     const int32_t* device_tags, const uint8_t* device_data) {
        if (host_ || layers <= 0 || experts <= 0 || experts > std::numeric_limits<int32_t>::max() ||
                ways < 1 || ways > 16 ||
                !stride || !device_tags || !device_data) return cudaErrorInvalidValue;
        const auto limit = std::numeric_limits<size_t>::max();
        if ((uint64_t)layers > limit / (size_t)ways) return cudaErrorInvalidValue;
        const size_t slots = (size_t)layers * (size_t)ways;
        if (slots > limit / sizeof(int32_t) || slots > limit / stride)
            return cudaErrorInvalidValue;
        const cudaError_t e = cudaHostAlloc((void**)&host_, slots * sizeof(int32_t), cudaHostAllocDefault);
        if (e != cudaSuccess) return e;
        layers_ = layers; experts_ = experts; ways_ = ways; stride_ = stride;
        device_tags_ = device_tags; device_data_ = device_data;
        bytes_ = slots * sizeof(int32_t);
        std::memset(host_, 0xff, bytes_);
        return cudaSuccess;
    }

    bool enabled() const { return host_ != nullptr; }
    bool ready() const { return valid_ && !pending_; }
    size_t bytes() const { return bytes_; }
    size_t stride() const { return stride_; }
    uint64_t generations() const { return generations_; }

    // No read of a prior snapshot may authorize a refill once a new verifier
    // window can mutate the secondary cache. Refuse an unfinished observation.
    bool invalidate() {
        if (pending_) return false;
        valid_ = false;
        return true;
    }

    cudaError_t enqueue(cudaStream_t stream) {
        if (!host_ || pending_) return cudaErrorInvalidValue;
        valid_ = false;
        const cudaError_t e = cudaMemcpyAsync(host_, device_tags_, bytes_, cudaMemcpyDeviceToHost, stream);
        if (e == cudaSuccess) { pending_ = true; pending_stream_ = stream; }
        return e;
    }

    // The caller passes the result of synchronizing THIS enqueue's stream.
    // No new query/event/synchronization is added to the inference hot path.
    bool complete_after_stream_sync(cudaError_t result) {
        if (!pending_ || result != cudaSuccess) return false;
        pending_ = false;
        pending_stream_ = nullptr;
        valid_ = true;
        ++generations_;
        return true;
    }

    const uint8_t* find(int64_t layer, int64_t expert) const {
        if (!ready() || layer < 0 || layer >= layers_ || expert < 0 || expert >= experts_)
            return nullptr;
        const size_t start = (size_t)layer * (size_t)ways_;
        for (int slot = 0; slot < ways_; ++slot)
            if (host_[start + (size_t)slot] == expert)
                return device_data_ + (start + (size_t)slot) * stride_;
        return nullptr;
    }

    void close() {
        // The owner calls close while its stream still exists. In the verifier
        // this follows its stream drain, including on an error-return path.
        if (pending_) cudaStreamSynchronize(pending_stream_);
        if (host_) cudaFreeHost(host_);
        host_ = nullptr;
        device_tags_ = nullptr;
        device_data_ = nullptr;
        pending_stream_ = nullptr;
        pending_ = valid_ = false;
        layers_ = experts_ = ways_ = 0;
        bytes_ = stride_ = 0;
        generations_ = 0;
    }

private:
    int32_t* host_ = nullptr;
    const int32_t* device_tags_ = nullptr;
    const uint8_t* device_data_ = nullptr;
    cudaStream_t pending_stream_ = nullptr;
    int64_t layers_ = 0, experts_ = 0;
    int ways_ = 0;
    size_t bytes_ = 0, stride_ = 0;
    uint64_t generations_ = 0;
    bool pending_ = false, valid_ = false;
};

} // namespace strata::core
