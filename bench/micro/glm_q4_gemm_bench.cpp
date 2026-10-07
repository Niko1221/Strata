// bench/micro/glm_q4_gemm_bench.cpp - the GLM engine's int4-g64 matrix product at expert shapes.
//
// A prompt's routed expert sees about S * 8 / 256 of the prompt's rows (48 for 1,536 tokens), so the multi-row path
// of q4_rows is where a CPU prompt spends its expert time.  Prints GFLOP/s for a 2048 x 6144 (gate/up) and a
// 6144 x 2048 (down) matrix at several row counts.  Synthetic weights: no model.
#include "strata/glm/kernels.hpp"

#include <chrono>
#include <cstdio>
#include <random>
#include <thread>
#include <vector>

using namespace strata::glm;

int main(int argc, char** argv) {
    const int threads = argc > 1 ? std::atoi(argv[1]) : (int) std::thread::hardware_concurrency() / 2;
    Pool pool(threads);
    std::mt19937 rng(1);
    std::uniform_int_distribution<int> byte(0, 255);
    std::uniform_real_distribution<float> uni(-1.0f, 1.0f);
    for (auto [O, I] : {std::pair{2048, 6144}, std::pair{6144, 2048}}) {
        std::vector<uint8_t> codes((size_t) O * I / 2);
        std::vector<float> scales((size_t) O * I / 64);
        for (auto& c : codes) c = (uint8_t) byte(rng);
        for (auto& s : scales) s = 0.01f;
        const Q4 W{O, I, codes.data(), scales.data()};
        for (int S : {1, 8, 48, 256}) {
            std::vector<float> x((size_t) S * I), y((size_t) S * O);
            for (auto& v : x) v = uni(rng);
            q4_gemm(pool, W, x.data(), S, y.data());   // warm
            const int reps = S == 1 ? 200 : S < 64 ? 20 : 5;
            const auto t0 = std::chrono::steady_clock::now();
            for (int r = 0; r < reps; ++r) q4_gemm(pool, W, x.data(), S, y.data());
            const double s = std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count() / reps;
            std::printf("%5d x %5d  S = %4d  %8.3f ms  %7.1f GFLOP/s  %6.1f GB/s of weights\n", O, I, S, s * 1e3,
                        2.0 * O * I * S / s / 1e9, (codes.size() + scales.size() * 4.0) / s / 1e9);
        }
    }
    return 0;
}
