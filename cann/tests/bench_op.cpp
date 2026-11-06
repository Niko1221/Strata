// cann/tests/bench_op.cpp - what one aclnn call costs.
//
// The whole model spends most of its time in calls that are individually tiny: a 1 x 2048 by
// 2048 x 512 GEMV is a rounding error on this card. This measures that call in a loop, with and
// without the per-op stream synchronize, so the next optimisation targets the real cost.
#include <acl/acl.h>

#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <vector>

#include "strata/cann/device.hpp"
#include "strata/cann/ops.hpp"

using namespace strata::cann;

int main(int argc, char ** argv) {
    setvbuf(stdout, nullptr, _IONBF, 0);
    const int64_t m = argc > 1 ? std::atoll(argv[1]) : 512;
    const int64_t k = argc > 2 ? std::atoll(argv[2]) : 2048;
    const int iters = argc > 3 ? std::atoi(argv[3]) : 200;

    Device device(0);
    Stream stream(device);

    std::vector<float> w((size_t) m * k, 0.01f);
    std::vector<uint16_t> wh(w.size());
    f32_to_f16(w.data(), wh.data(), w.size());
    Buffer bw(w.size() * 2);
    bw.copy_from_host(wh.data(), wh.size() * 2);

    std::vector<float> x((size_t) k, 0.5f);
    std::vector<uint16_t> xh((size_t) k);
    f32_to_f16(x.data(), xh.data(), k);
    Buffer bx((size_t) k * 2), by((size_t) m * 2);
    bx.copy_from_host(xh.data(), xh.size() * 2);

    Tensor tw{ bw.data(), DType::F16, k, m };
    Tensor tx{ bx.data(), DType::F16, 1, k };
    Tensor ty{ by.data(), DType::F16, 1, m };

    // warm up: the first call pays for op library loading
    for (int i = 0; i < 5; ++i) {
        matmul(stream, tw, tx, ty);
    }
    stream.synchronize();

    const auto t0 = std::chrono::steady_clock::now();
    for (int i = 0; i < iters; ++i) {
        matmul(stream, tw, tx, ty);
    }
    stream.synchronize();
    const auto t1 = std::chrono::steady_clock::now();
    const double total = std::chrono::duration<double>(t1 - t0).count();

    printf("gemv %lldx%lld, %d iterations\n", (long long) m, (long long) k, iters);
    printf("  total   %.3f s\n", total);
    printf("  per op  %.3f ms  (sync mode: %s)\n", total * 1000.0 / iters,
           getenv("STRATA_CANN_NO_SYNC") ? "off" : "on");

    // a wider matmul for comparison: same shape but 8 rows of output
    Buffer by8((size_t) m * 8 * 2);
    Tensor ty8{ by8.data(), DType::F16, 8, m };
    std::vector<float> x8((size_t) k * 8, 0.5f);
    std::vector<uint16_t> x8h(x8.size());
    f32_to_f16(x8.data(), x8h.data(), x8.size());
    Buffer bx8((size_t) k * 8 * 2);
    bx8.copy_from_host(x8h.data(), x8h.size() * 2);
    Tensor tx8{ bx8.data(), DType::F16, 8, k };
    for (int i = 0; i < 5; ++i) {
        matmul(stream, tw, tx8, ty8);
    }
    stream.synchronize();
    const auto t2 = std::chrono::steady_clock::now();
    for (int i = 0; i < iters; ++i) {
        matmul(stream, tw, tx8, ty8);
    }
    stream.synchronize();
    const auto t3 = std::chrono::steady_clock::now();
    printf("  same weight, 8 rows: %.3f ms per op\n", std::chrono::duration<double>(t3 - t2).count() * 1000.0 / iters);

    shutdown();
    return 0;
}
