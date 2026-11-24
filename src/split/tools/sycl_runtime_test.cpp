#include "../src/sycl/runtime.hpp"

#include <cmath>
#include <cstdio>
#include <limits>
#include <stdexcept>

using namespace fnx::sycl_backend;

static void check(bool ok, const char* what) {
    if (!ok) throw std::runtime_error(what);
}
template <typename F> static void refuses(F f) {
    try { f(); } catch (const std::runtime_error&) { return; }
    throw std::runtime_error("invalid collective accepted");
}

int main() try {
    const auto devices = sycl::device::get_devices(sycl::info::device_type::gpu);
    check(devices.size() == 2, "select exactly two GPUs with ONEAPI_DEVICE_SELECTOR=level_zero:0,1");
    Runtime runtime(devices);
    for (size_t c = 0; c < runtime.cards(); ++c)
        std::printf("card %zu: %s\n", c, devices[c].get_info<sycl::info::device::name>().c_str());
    refuses([&] { Runtime duplicate({devices[0], devices[0]}); });
    constexpr size_t n = 4097, bytes = n * sizeof(float);
    DeviceBuffer a(runtime, 0, bytes), b(runtime, 1, bytes);
    std::vector<float*> ptrs{(float*) a.data(), (float*) b.data()};
    std::vector<float> result(n);
    for (int round = 0; round < 8; ++round) {
        for (size_t c = 0; c < runtime.cards(); ++c) {
            float* dst = ptrs[c];
            runtime.queue(c).parallel_for(sycl::range<1>(n), [=](sycl::id<1> id) {
                dst[id[0]] = (float) ((id[0] % 19) * (c + 1)) + (float) round;
            });
        }
        runtime.sum(ptrs, n);
        for (size_t c = 0; c < runtime.cards(); ++c) {
            runtime.queue(c).memcpy(result.data(), ptrs[c], bytes).wait_and_throw();
            for (size_t i = 0; i < n; ++i)
                check(result[i] == (float) (3 * (i % 19) + 2 * round), "wrong two-card sum or stale buffer");
        }
    }
    refuses([&] { runtime.sum({ptrs[1], ptrs[0]}, n); });
    refuses([&] { runtime.sum({ptrs[0]}, n); });
    refuses([&] { runtime.sum(ptrs, SIZE_MAX); });
    runtime.sum(ptrs, 0);
    runtime.queue(0).single_task([p = ptrs[0]] {
        p[0] = std::numeric_limits<float>::infinity();
    });
    refuses([&] { runtime.sum(ptrs, n); });
    // An error must not poison the queues or retain staging allocations.
    runtime.queue(0).memset(ptrs[0], 0, bytes);
    runtime.queue(1).memset(ptrs[1], 0, bytes);
    runtime.sum(ptrs, n);
    std::puts("SYCL two-card host-staged collective: PASS");
    return 0;
} catch (const std::exception& e) {
    std::fprintf(stderr, "FAIL: %s\n", e.what());
    return 1;
}
