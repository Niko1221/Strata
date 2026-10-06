// bw.cpp - achievable device-memory read bandwidth: sum 512 MB with 4-, 16- and 64-byte loads per lane.
#include <sycl/sycl.hpp>
#include <chrono>
#include <cstdio>
int main() {
    setvbuf(stdout, nullptr, _IONBF, 0);
    sycl::queue q{sycl::gpu_selector_v, sycl::property::queue::in_order()};
    const size_t bytes = 512u << 20, n4 = bytes / 4;
    int* d = sycl::malloc_device<int>(n4, q);
    q.parallel_for(sycl::range<1>(n4), [=](sycl::id<1> i) { unsigned x = (unsigned) i[0] * 2654435761u; x ^= x >> 13; x *= 0x5bd1e995u; d[i] = (int) (x ^ (x >> 15)); }).wait();   // incompressible
    int* out = sycl::malloc_device<int>(1 << 16, q);
    auto run = [&](const char* what, auto kernel) {
        for (int w = 0; w < 3; ++w) { kernel(); q.wait(); }
        const auto t0 = std::chrono::steady_clock::now();
        for (int r = 0; r < 10; ++r) kernel();
        q.wait();
        const double ms = std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - t0).count() / 10;
        std::printf("%-28s %.2f ms  %.0f GB/s\n", what, ms, bytes / ms / 1e6);
    };
    const size_t wg = 256, groups = 4096;
    run("4-byte loads per lane", [&] { q.parallel_for(sycl::nd_range<1>(groups * wg, wg), [=](sycl::nd_item<1> it) {
        int acc = 0; for (size_t i = it.get_global_id(0); i < n4; i += groups * wg) acc += d[i];
        if (acc == 12345) out[it.get_group(0)] = acc; }); });
    run("16-byte loads per lane", [&] { q.parallel_for(sycl::nd_range<1>(groups * wg, wg), [=](sycl::nd_item<1> it) {
        const sycl::int4* v = (const sycl::int4*) d; int acc = 0;
        for (size_t i = it.get_global_id(0); i < n4 / 4; i += groups * wg) { auto x = v[i]; acc += x.x() + x.y() + x.z() + x.w(); }
        if (acc == 12345) out[it.get_group(0)] = acc; }); });
    run("64-byte loads per lane (4xint4)", [&] { q.parallel_for(sycl::nd_range<1>(groups * wg, wg), [=](sycl::nd_item<1> it) {
        const sycl::int4* v = (const sycl::int4*) d; int acc = 0;
        for (size_t i = it.get_global_id(0) * 4; i + 3 < n4 / 4; i += groups * wg * 4) { for (int k = 0; k < 4; ++k) { auto x = v[i + k]; acc += x.x() + x.y() + x.z() + x.w(); } }
        if (acc == 12345) out[it.get_group(0)] = acc; }); });
    run("memcpy d2d (512 MB)", [&] { q.memcpy(out, d, 0); q.memcpy(d + n4 / 2, d, bytes / 2); });
    return 0;
}
