#include "strata/cann/device.hpp"

#include <acl/acl.h>

#include <mutex>

namespace strata::cann {
namespace {

std::once_flag g_init_once;
bool g_initialized = false;

[[noreturn]] void fail(const char * what, int code) {
    throw Error(std::string("cann: ") + what + " failed (aclError " + std::to_string(code) + ")", code);
}

void check(aclError err, const char * what) {
    if (err != ACL_SUCCESS) {
        fail(what, (int) err);
    }
}

} // namespace

void init() {
    std::call_once(g_init_once, [] {
        aclError err = aclInit(nullptr);
        if (err != ACL_SUCCESS) {
            fail("aclInit", (int) err);
        }
        g_initialized = true;
    });
}

void shutdown() {
    if (g_initialized) {
        // the ops share one workspace buffer; release it while the runtime is still up
        extern void release_workspace();
        release_workspace();
        aclFinalize();
        g_initialized = false;
    }
}

int device_count() {
    init();
    uint32_t count = 0;
    if (aclrtGetDeviceCount(&count) != ACL_SUCCESS) {
        return 0;
    }
    return (int) count;
}

DeviceInfo device_info(int ordinal) {
    init();
    DeviceInfo info;
    info.ordinal = ordinal;

    // aclrtGetMemInfo reports the current device; select it without owning the reset here.
    if (aclrtSetDevice(ordinal) != ACL_SUCCESS) {
        fail("aclrtSetDevice", (int) ACL_ERROR_RT_INVALID_DEVICEID);
    }

    const char * soc = aclrtGetSocName();
    info.soc = soc ? soc : "";

    size_t free_bytes = 0, total_bytes = 0;
    check(aclrtGetMemInfo(ACL_HBM_MEM, &free_bytes, &total_bytes), "aclrtGetMemInfo");
    info.free_bytes = free_bytes;
    info.total_bytes = total_bytes;
    return info;
}

Device::Device(int ordinal) : ordinal_(ordinal) {
    init();
    if (aclrtSetDevice(ordinal) != ACL_SUCCESS) {
        fail("aclrtSetDevice", (int) ACL_ERROR_RT_INVALID_DEVICEID);
    }
}

Device::~Device() {
    aclrtResetDevice(ordinal_);
}

Stream::Stream(const Device & device) {
    (void) device;
    check(aclrtCreateStream((aclrtStream *) &stream_), "aclrtCreateStream");
}

Stream::~Stream() {
    if (stream_) {
        aclrtDestroyStream((aclrtStream) stream_);
    }
}

void Stream::synchronize() const {
    check(aclrtSynchronizeStream((aclrtStream) stream_), "aclrtSynchronizeStream");
}

Buffer::Buffer(size_t bytes) : bytes_(bytes) {
    check(aclrtMalloc(&data_, bytes, ACL_MEM_MALLOC_HUGE_FIRST), "aclrtMalloc");
}

Buffer::~Buffer() {
    if (data_) {
        aclrtFree(data_);
    }
}

void copy_to_host(const void * device_ptr, void * host, size_t bytes) {
    check(aclrtMemcpy(host, bytes, device_ptr, bytes, ACL_MEMCPY_DEVICE_TO_HOST), "aclrtMemcpy(D2H raw)");
}

void copy_to_device(void * device_ptr, const void * host, size_t bytes) {
    check(aclrtMemcpy(device_ptr, bytes, host, bytes, ACL_MEMCPY_HOST_TO_DEVICE), "aclrtMemcpy(H2D raw)");
}

void Buffer::copy_from_host(const void * host, size_t bytes) {
    if (bytes > bytes_) {
        throw Error("cann: copy_from_host larger than the buffer", -1);
    }
    check(aclrtMemcpy(data_, bytes_, host, bytes, ACL_MEMCPY_HOST_TO_DEVICE), "aclrtMemcpy(H2D)");
}

void Buffer::copy_to_host(void * host, size_t bytes) const {
    if (bytes > bytes_) {
        throw Error("cann: copy_to_host larger than the buffer", -1);
    }
    check(aclrtMemcpy(host, bytes, data_, bytes, ACL_MEMCPY_DEVICE_TO_HOST), "aclrtMemcpy(D2H)");
}

} // namespace strata::cann
