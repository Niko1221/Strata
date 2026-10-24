// Exercise the actual header on host threads. No CUDA/HIP runtime call is made.
#include "strata/core/conversation_cache.hpp"
#include "strata/core/progress.hpp"
#include <cuda_runtime.h>

#include <atomic>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <mutex>
#include <thread>
#include <unordered_map>

namespace fixture {
constexpr cudaError_t failure = static_cast<cudaError_t>(1);
std::mutex mutex;
std::unordered_map<cudaStream_t, int> live;
int created = 0, destroyed = 0;
std::atomic<int> errors{0};
thread_local int device = 0;
thread_local bool fail_create = false, fail_get = false, fail_copy = false, fail_sync = false;

cudaError_t get_device(int* d) {
    if (fail_get) return failure;
    *d = device;
    return cudaSuccess;
}
cudaError_t set_device(int d) { device = d; return cudaSuccess; }
cudaError_t create(cudaStream_t* s, unsigned int) {
    std::lock_guard<std::mutex> lock(mutex);
    if (fail_create) {
        *s = reinterpret_cast<cudaStream_t>(uintptr_t(999999));
        return failure;
    }
    *s = reinterpret_cast<cudaStream_t>(uintptr_t(++created));
    live.emplace(*s, device);
    return cudaSuccess;
}
cudaError_t destroy(cudaStream_t s) {
    std::lock_guard<std::mutex> lock(mutex);
    auto it = live.find(s);
    if (it == live.end() || it->second != device) {
        ++errors;
        return failure;
    }
    live.erase(it);
    ++destroyed;
    return cudaSuccess;
}
cudaError_t copy(void*, const void*, size_t, decltype(cudaMemcpyDefault), cudaStream_t) {
    return fail_copy ? failure : cudaSuccess;
}
cudaError_t sync(cudaStream_t) { return fail_sync ? failure : cudaSuccess; }
void check(bool ok, const char* what) {
    if (!ok) { std::fprintf(stderr, "FAIL: %s\n", what); std::exit(1); }
}
struct CheckRestoredDevice {
    int expected = 0;
    ~CheckRestoredDevice() { if (device != expected) ++errors; }
};
}

// Dependencies are already included. Redirect only the operations of this helper.
#undef cudaGetDevice
#undef cudaSetDevice
#undef cudaStreamCreateWithFlags
#undef cudaStreamDestroy
#undef cudaMemcpyAsync
#undef cudaStreamSynchronize
#define cudaGetDevice fixture::get_device
#define cudaSetDevice fixture::set_device
#define cudaStreamCreateWithFlags fixture::create
#define cudaStreamDestroy fixture::destroy
#define cudaMemcpyAsync fixture::copy
#define cudaStreamSynchronize fixture::sync
#include "conversation_copy.hpp"

using strata::core::conversation_detail::copy_nonblocking;

int main() {
    using namespace fixture;
    // Models one fresh async stage thread per chunk: retained streams used to
    // increase by one each time, even after every worker had been joined.
    for (int i = 0; i < 32; ++i) {
        std::thread([] {
            check(copy_nonblocking(nullptr, nullptr, 0) == cudaSuccess, "first copy");
            check(copy_nonblocking(nullptr, nullptr, 0) == cudaSuccess, "reuse stream");
        }).join();
        check(live.empty(), "exited worker retains a stream");
    }
    check(created == 32 && destroyed == 32, "one stream per worker");
    const int before = created;
    std::thread([] {
        for (int i = 0; i < 32; ++i)
            check(copy_nonblocking(nullptr, nullptr, 0) == cudaSuccess, "persistent worker copy");
    }).join();
    check(created == before + 1 && live.empty(), "persistent worker reuses one stream");

    std::vector<std::thread> workers;
    for (int i = 0; i < 16; ++i) workers.emplace_back([] {
        // Constructed before the stream owner, hence inspected after its cleanup.
        thread_local CheckRestoredDevice after_owner;
        for (int d : {2, 0, 3, 2}) {
            device = d;
            check(copy_nonblocking(nullptr, nullptr, 0) == cudaSuccess, "multi-device copy");
        }
        after_owner.expected = device;
    });
    for (auto& worker : workers) worker.join();
    check(live.empty() && errors == 0, "multi-device cleanup and device restoration");

    std::thread([] {
        fail_get = true;
        check(copy_nonblocking(nullptr, nullptr, 0) == failure, "device query failure");
        fail_get = false;
        fail_create = true;
        check(copy_nonblocking(nullptr, nullptr, 0) == failure, "stream creation failure");
        fail_create = false;
        fail_copy = true;
        check(copy_nonblocking(nullptr, nullptr, 0) == failure, "copy failure");
        fail_copy = false;
        fail_sync = true;
        check(copy_nonblocking(nullptr, nullptr, 0) == failure, "sync failure");
        fail_sync = false;
    }).join();
    check(live.empty() && created == destroyed && errors == 0, "error paths release owned streams");
    std::puts("PASS: worker exit, stream reuse, concurrent devices, failure cleanup");
}
