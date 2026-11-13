// tp_allreduce_sycl_test.cpp - a standalone check of the SYCL all-reduce on two devices.
//
//   icpx -fsycl -std=c++20 -I sycl/include -I include -I third_party/ggml \
//        tp_allreduce_sycl_test.cpp sycl/src/kernels/cuda/allreduce.dp.cpp -o tp_allreduce_test
//   ONEAPI_DEVICE_SELECTOR=level_zero:* ./tp_allreduce_test
//
// Rank 0 fills 1.0, rank 1 fills 2.0; both must end with 3.0 (F32 wire) or the bf16-rounded sum.
#include "strata/kernels/allreduce.hpp"
#include "strata/sycl_queue.hpp"

#include <sycl/sycl.hpp>
#include <dpct/dpct.hpp>

#include <cmath>
#include <cstdio>
#include <string>
#include <thread>
#include <vector>

int main() {
    strata::kernels::ArPair pair;
    std::string err;
    int ordinals[2] = {0, 1};
    if (!pair.init(ordinals, 2, 1024, strata::kernels::ArWire::F32, 8, err)) {
        std::fprintf(stderr, "init: %s\n", err.c_str());
        return 1;
    }
    const int n = 1024;
    float* x[2] = {nullptr, nullptr};
    for (int r = 0; r < 2; ++r) {
        dpct::select_device(ordinals[r]);
        x[r] = sycl::malloc_device<float>(n, dpct::get_in_order_queue());
        std::vector<float> host(n, r == 0 ? 1.0f : 2.0f);
        dpct::get_in_order_queue().memcpy(x[r], host.data(), n * sizeof(float)).wait();
    }
    std::thread t0([&] {
        dpct::select_device(0);
        pair.launch(0, x[0], n, nullptr);
    });
    std::thread t1([&] {
        dpct::select_device(1);
        pair.launch(1, x[1], n, nullptr);
    });
    t0.join();
    t1.join();
    for (int r = 0; r < 2; ++r) {
        dpct::select_device(ordinals[r]);
        dpct::get_in_order_queue().wait_and_throw();
    }
    int bad = 0;
    for (int r = 0; r < 2; ++r) {
        dpct::select_device(ordinals[r]);
        std::vector<float> host(n);
        dpct::get_in_order_queue().memcpy(host.data(), x[r], n * sizeof(float)).wait();
        double sum = 0.0;
        for (float v : host) sum += v;
        const double mean = sum / n;
        std::printf("rank %d: mean %.6f (expect 3.0)\n", r, mean);
        if (std::fabs(mean - 3.0) > 1e-4) ++bad;
    }
    std::printf("%s\n", bad == 0 ? "ALLREDUCE OK" : "ALLREDUCE FAILED");
    return bad == 0 ? 0 : 1;
}
