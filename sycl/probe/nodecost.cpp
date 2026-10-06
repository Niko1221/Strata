// nodecost.cpp - per-node cost of small-but-real kernels in a graph vs on a queue (memory-touching, 8 work-groups).
#include <sycl/sycl.hpp>
#include <sycl/ext/oneapi/experimental/graph.hpp>
#include <chrono>
#include <cstdio>
namespace exp = sycl::ext::oneapi::experimental;
int main() {
    setvbuf(stdout, nullptr, _IONBF, 0);
    sycl::queue q{sycl::gpu_selector_v, sycl::property::queue::in_order()};
    const size_t n = 1 << 16;             // 64k floats = 256 KB per kernel: a typical layer-stage vector
    float* a = sycl::malloc_device<float>(n, q); float* b = sycl::malloc_device<float>(n, q);
    q.memset(a, 0, n * 4).wait(); q.memset(b, 0, n * 4).wait();
    const int N = 2000;
    for (int wg : {8, 64, 256}) {           // work-groups per kernel: 8 = a tiny norm, 256 = a small matvec
        auto launch = [&](sycl::queue& qq) {
            for (int i = 0; i < N; ++i)
                qq.parallel_for(sycl::nd_range<1>(wg * 256, 256), [=](sycl::nd_item<1> it) {
                    const size_t g = it.get_global_id(0);
                    for (size_t j = g; j < n; j += (size_t) wg * 256) b[j] = a[j] * 1.0001f + 1.0f;
                });
        };
        double tq = 1e9, tg = 1e9;
        for (int r = 0; r < 3; ++r) { auto t0 = std::chrono::steady_clock::now(); launch(q); q.wait();
            tq = std::min(tq, std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - t0).count()); }
        exp::command_graph<exp::graph_state::modifiable> g(q.get_context(), q.get_device());
        g.begin_recording(q); launch(q); g.end_recording();
        auto ge = g.finalize();
        for (int r = 0; r < 3; ++r) { auto t0 = std::chrono::steady_clock::now(); q.ext_oneapi_graph(ge).wait();
            tg = std::min(tg, std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - t0).count()); }
        std::printf("%3d work-groups x 256, 256 KB touched: queue %.1f us/kernel   graph %.1f us/node\n", wg, tq * 1000 / N, tg * 1000 / N);
    }
    return 0;
}
