// graphbench.cpp - what does one graph node cost on this card, with and without use_root_sync?
#include <sycl/sycl.hpp>
#include <sycl/ext/oneapi/experimental/graph.hpp>
#include <chrono>
#include <cstdio>
namespace exp = sycl::ext::oneapi::experimental;
int main() {
    setvbuf(stdout, nullptr, _IONBF, 0);
    sycl::queue q{sycl::gpu_selector_v, sycl::property::queue::in_order()};
    int* c = sycl::malloc_device<int>(1024, q);
    q.memset(c, 0, 4096).wait();
    const int N = 2000;
    for (int mode = 0; mode < 3; ++mode) {   // 0: plain queue, no graph; 1: graph, plain nodes; 2: graph, use_root_sync nodes
        auto launch = [&](sycl::queue& qq) {
            for (int i = 0; i < N; ++i) {
                if (mode == 2) {
                    auto props = exp::properties{exp::use_root_sync};
                    qq.parallel_for(sycl::nd_range<1>(256, 256), props, [=](sycl::nd_item<1> it) { if (it.get_global_id(0) == 0) c[i & 1023] += 1; });
                } else {
                    qq.parallel_for(sycl::nd_range<1>(256, 256), [=](sycl::nd_item<1> it) { if (it.get_global_id(0) == 0) c[i & 1023] += 1; });
                }
            }
        };
        double best = 1e9;
        if (mode == 0) {
            for (int r = 0; r < 3; ++r) { auto t0 = std::chrono::steady_clock::now(); launch(q); q.wait();
                best = std::min(best, std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - t0).count()); }
        } else {
            exp::command_graph<exp::graph_state::modifiable> g(q.get_context(), q.get_device());
            g.begin_recording(q); launch(q); g.end_recording();
            auto ge = g.finalize();
            for (int r = 0; r < 3; ++r) { auto t0 = std::chrono::steady_clock::now(); q.ext_oneapi_graph(ge).wait();
                best = std::min(best, std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - t0).count()); }
        }
        std::printf("mode %d (%s): %d kernels in %.1f ms = %.1f us per kernel\n", mode,
                    mode == 0 ? "queue, no graph" : mode == 1 ? "graph, plain" : "graph, use_root_sync", N, best, best * 1000 / N);
    }
    return 0;
}
