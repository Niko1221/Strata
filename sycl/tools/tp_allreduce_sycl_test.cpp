// tp_allreduce_sycl_test.cpp - a standalone check of the SYCL all-reduce on two to four devices.
//
//   icpx -fsycl -std=c++20 -I sycl/include -I include -I third_party/ggml \
//        tp_allreduce_sycl_test.cpp sycl/src/kernels/cuda/allreduce.dp.cpp -o tp_allreduce_test
//   ONEAPI_DEVICE_SELECTOR=level_zero:* ./tp_allreduce_test [ranks=2] [blocks=8] [n=1024] [wire=f32|bf16] [calls=4]
//
// Rank r fills element i with (r + 1) * (1 + i % 7).  Every element on every rank must end as the sum over the
// ranks, checked ELEMENT BY ELEMENT (a mean hides a sum that only covers some stripes), and every rank must
// hold the same bits.  The values are small integers, so the bf16 wire is exact too.  The call is repeated so
// both staging parities and the token counters are exercised.
#include "strata/kernels/allreduce.hpp"
#include "strata/sycl_queue.hpp"

#include <sycl/sycl.hpp>
#include <dpct/dpct.hpp>

#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>
#include <thread>
#include <vector>

int main(int argc, char** argv) {
    const int ranks = argc > 1 ? std::atoi(argv[1]) : 2;
    const int blocks = argc > 2 ? std::atoi(argv[2]) : 8;
    const int n = argc > 3 ? std::atoi(argv[3]) : 1024;
    const bool bf16 = argc > 4 && std::strcmp(argv[4], "bf16") == 0;
    const int calls = argc > 5 ? std::atoi(argv[5]) : 4;
    if (ranks < 2 || ranks > strata::kernels::kArRanks || n < 1 || calls < 1) {
        std::fprintf(stderr, "usage: tp_allreduce_test [ranks 2..4] [blocks] [n] [f32|bf16] [calls]\n");
        return 2;
    }
    strata::kernels::ArPair pair;
    std::string err;
    int ordinals[strata::kernels::kArRanks] = {0, 1, 2, 3};
    if (!pair.init(ordinals, ranks, n, bf16 ? strata::kernels::ArWire::Bf16 : strata::kernels::ArWire::F32, blocks,
                   err)) {
        std::fprintf(stderr, "init: %s\n", err.c_str());
        return 1;
    }
    std::vector<float*> x(ranks, nullptr);
    for (int r = 0; r < ranks; ++r) {
        dpct::select_device(ordinals[r]);
        x[r] = sycl::malloc_device<float>(n, dpct::get_in_order_queue());
    }
    int bad = 0;
    for (int c = 0; c < calls; ++c) {
        for (int r = 0; r < ranks; ++r) {
            dpct::select_device(ordinals[r]);
            std::vector<float> host(n);
            for (int i = 0; i < n; ++i) host[i] = (float) ((r + 1) * (1 + i % 7) + c);
            dpct::get_in_order_queue().memcpy(x[r], host.data(), n * sizeof(float)).wait();
        }
        std::vector<std::thread> th;
        for (int r = 0; r < ranks; ++r)
            th.emplace_back([&, r] {
                dpct::select_device(ordinals[r]);
                pair.launch(r, x[r], n, nullptr);
                dpct::get_in_order_queue().wait_and_throw();
            });
        for (auto& t : th) t.join();
        if (const int tr = pair.timed_out_rank(); tr >= 0) {
            std::printf("call %d: rank %d timed out\n", c, tr);
            ++bad;
            pair.clear_timeout();
        }
        std::vector<std::vector<float>> got(ranks, std::vector<float>(n));
        for (int r = 0; r < ranks; ++r) {
            dpct::select_device(ordinals[r]);
            dpct::get_in_order_queue().memcpy(got[r].data(), x[r], n * sizeof(float)).wait();
        }
        for (int r = 0; r < ranks; ++r) {
            int wrong = 0, first = -1;
            for (int i = 0; i < n; ++i) {
                const float want = (float) (ranks * (ranks + 1) / 2 * (1 + i % 7) + ranks * c);
                if (got[r][i] != want || std::memcmp(&got[r][i], &got[0][i], sizeof(float)) != 0) {
                    if (first < 0) first = i;
                    ++wrong;
                }
            }
            std::printf("call %d rank %d: %d of %d elements wrong%s\n", c, r, wrong, n,
                        wrong ? (" (first at " + std::to_string(first) + ", got " + std::to_string(got[r][first]) +
                                 ")").c_str() : "");
            if (wrong) ++bad;
        }
    }
    std::printf("%s\n", bad == 0 ? "ALLREDUCE OK" : "ALLREDUCE FAILED");
    return bad == 0 ? 0 : 1;
}
