#include "strata/cann/device.hpp"

#include <acl/acl.h>

#include <chrono>
#include <mutex>
#include <unordered_map>
#include <vector>

namespace strata::cann {

void print_xfer_stats();

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
        extern void clear_buffer_pool();
        clear_buffer_pool();
        print_xfer_stats();
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

// Device memory is pooled. A single token allocates thousands of small temporaries, and one
// aclrtMalloc per temporary was the real cost of the whole model: 12.8 s per token on ops that
// each measure 0.043 ms.
namespace {

std::mutex g_pool_mutex;
std::unordered_map<size_t, std::vector<void *>> g_pool;

void * pool_take(size_t bytes) {
    {
        std::lock_guard<std::mutex> lock(g_pool_mutex);
        auto it = g_pool.find(bytes);
        if (it != g_pool.end() && !it->second.empty()) {
            void * p = it->second.back();
            it->second.pop_back();
            return p;
        }
    }
    void * p = nullptr;
    check(aclrtMalloc(&p, bytes, ACL_MEM_MALLOC_HUGE_FIRST), "aclrtMalloc");
    return p;
}

void pool_give(void * p, size_t bytes) {
    std::lock_guard<std::mutex> lock(g_pool_mutex);
    g_pool[bytes].push_back(p);
}

} // namespace

// Profiling: the model spends its time outside the ops, so transfers are counted too.
long long g_xfer_count = 0;
long long g_xfer_bytes = 0;
double g_xfer_seconds = 0.0;

void print_xfer_stats() {
    if (g_xfer_count > 0) {
        printf("xfer: %lld copies, %.1f MB, %.3f s total, %.3f ms each\n", g_xfer_count,
               (double) g_xfer_bytes / 1e6, g_xfer_seconds, 1000.0 * g_xfer_seconds / (double) g_xfer_count);
    }
}

void clear_buffer_pool() {
    std::lock_guard<std::mutex> lock(g_pool_mutex);
    for (auto & entry : g_pool) {
        for (void * p : entry.second) {
            aclrtFree(p);
        }
    }
    g_pool.clear();
}

Buffer::Buffer(size_t bytes) : bytes_(bytes) {
    data_ = pool_take(bytes);
}

Buffer::~Buffer() {
    if (data_) {
        pool_give(data_, bytes_);
    }
}

void copy_to_host(const void * device_ptr, void * host, size_t bytes) {
    check(aclrtMemcpy(host, bytes, device_ptr, bytes, ACL_MEMCPY_DEVICE_TO_HOST), "aclrtMemcpy(D2H raw)");
}

void copy_to_device(void * device_ptr, const void * host, size_t bytes) {
    check(aclrtMemcpy(device_ptr, bytes, host, bytes, ACL_MEMCPY_HOST_TO_DEVICE), "aclrtMemcpy(H2D raw)");
}

void Buffer::copy_from_host(const void * host, size_t bytes) {
    const auto t0 = std::chrono::steady_clock::now();
    struct Timer {
        std::chrono::steady_clock::time_point t0;
        size_t bytes;
        ~Timer() {
            g_xfer_count += 1;
            g_xfer_bytes += (long long) bytes;
            g_xfer_seconds += std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count();
        }
    } timer{ t0, bytes };
    if (bytes > bytes_) {
        throw Error("cann: copy_from_host larger than the buffer", -1);
    }
    check(aclrtMemcpy(data_, bytes_, host, bytes, ACL_MEMCPY_HOST_TO_DEVICE), "aclrtMemcpy(H2D)");
}

void Buffer::copy_to_host(void * host, size_t bytes) const {
    const auto t0 = std::chrono::steady_clock::now();
    struct Timer {
        std::chrono::steady_clock::time_point t0;
        size_t bytes;
        ~Timer() {
            g_xfer_count += 1;
            g_xfer_bytes += (long long) bytes;
            g_xfer_seconds += std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count();
        }
    } timer{ t0, bytes };
    if (bytes > bytes_) {
        throw Error("cann: copy_to_host larger than the buffer", -1);
    }
    check(aclrtMemcpy(host, bytes, data_, bytes, ACL_MEMCPY_DEVICE_TO_HOST), "aclrtMemcpy(D2H)");
}

} // namespace strata::cann
