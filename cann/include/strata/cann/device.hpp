// cann/include/strata/cann/device.hpp - the CANN device layer of the Ascend port.
//
// This is the smallest piece the rest of the port needs: create a device context, make a
// stream, allocate HBM, move bytes, and turn aclError into an exception that names the call.
// It deliberately does not touch Strata's engine types yet - the engine's kernels are ported
// on top of this layer, one op at a time (see cann/README.md).
#pragma once

#include <cstddef>
#include <cstdint>
#include <stdexcept>
#include <string>
#include <vector>

namespace strata::cann {

// Any ACL failure, with the numeric code and the call that produced it.
class Error : public std::runtime_error {
public:
    Error(const std::string & what, int code) : std::runtime_error(what), code_(code) {}
    int code() const { return code_; }

private:
    int code_;
};

// aclInit/aclFinalize once per process. Safe to call repeatedly.
void init();
void shutdown();

int device_count();

struct DeviceInfo {
    int ordinal = -1;
    std::string soc;          // "Ascend910B2C"
    uint64_t total_bytes = 0; // HBM total, from aclrtGetMemInfo
    uint64_t free_bytes = 0;
};

// Throws Error when the ordinal does not exist.
DeviceInfo device_info(int ordinal = 0);

// Owns aclrtSetDevice/aclrtResetDevice for one ordinal.
class Device {
public:
    explicit Device(int ordinal = 0);
    ~Device();
    Device(const Device &) = delete;
    Device & operator=(const Device &) = delete;

    int ordinal() const { return ordinal_; }

private:
    int ordinal_;
};

// Owns one aclrtStream.
class Stream {
public:
    explicit Stream(const Device & device);
    ~Stream();
    Stream(const Stream &) = delete;
    Stream & operator=(const Stream &) = delete;

    void synchronize() const;
    void * handle() const { return stream_; }

private:
    void * stream_ = nullptr;
};

// Owns one aclrtMalloc block.
class Buffer {
public:
    explicit Buffer(size_t bytes);
    ~Buffer();
    Buffer(const Buffer &) = delete;
    Buffer & operator=(const Buffer &) = delete;

    void * data() const { return data_; }
    size_t size() const { return bytes_; }

    void copy_from_host(const void * host, size_t bytes);
    void copy_to_host(void * host, size_t bytes) const;

private:
    void * data_ = nullptr;
    size_t bytes_ = 0;
};

} // namespace strata::cann
