// hostread.cpp - does a kernel see data the host wrote into host USM before the launch?
#include <sycl/sycl.hpp>
#include <cstdio>
#include <cstring>
#include <atomic>
int main() {
    setvbuf(stdout, nullptr, _IONBF, 0);
    sycl::queue q{sycl::gpu_selector_v, sycl::property::queue::in_order()};
    const size_t n = 1 << 20;
    float* h = sycl::malloc_host<float>(n, q);          // host USM, written by the CPU
    float* d = sycl::malloc_device<float>(n, q);
    float* sum = sycl::malloc_shared<float>(4, q);
    for (int round = 0; round < 3; ++round) {
        for (size_t i = 0; i < n; ++i) h[i] = (float) (round + 1);   // host write AFTER the mapping exists
        std::atomic_thread_fence(std::memory_order_seq_cst);
        sum[0] = 0; sum[1] = 0;
        // (a) a kernel reads host USM directly
        q.parallel_for(sycl::range<1>(n), [=](sycl::id<1> i) {
            sycl::atomic_ref<float, sycl::memory_order::relaxed, sycl::memory_scope::device> a(sum[0]);
            a.fetch_add(h[i]);
        }).wait();
        // (b) memcpy H2D then a kernel reads device memory
        q.memcpy(d, h, n * 4).wait();
        q.parallel_for(sycl::range<1>(n), [=](sycl::id<1> i) {
            sycl::atomic_ref<float, sycl::memory_order::relaxed, sycl::memory_scope::device> a(sum[1]);
            a.fetch_add(d[i]);
        }).wait();
        std::printf("round %d: expected %.0f   kernel reading host USM: %.0f   after memcpy to device: %.0f\n",
                    round, (double) n * (round + 1), sum[0], sum[1]);
    }
    return 0;
}
