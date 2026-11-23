// SPDX-License-Identifier: MIT
// tests/core/dot_x4_bench.cpp - speed of the 4-row shared-activation kernels (dot_x4_kernel, STRATA_NO_DOT_X4 off) against four calls of the
// 1-row kernel (STRATA_NO_DOT_X4=1 reaches it) on the same rows, per weight type: one cache-resident block (4 MiB of rows) and one DRAM stream
// (512 MiB of rows, the shape of a cold expert read).  Prints GB/s and the ratio; always exits 0 (a measurement, not a gate).  #6552.
#include "strata/quant/dot.hpp"
#include "strata/quant/types.hpp"

#include <algorithm>
#include <chrono>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <random>
#include <vector>

using namespace strata::quant;

static double now_s() {
    return std::chrono::duration<double>(std::chrono::steady_clock::now().time_since_epoch()).count();
}

static void bench_type(Type wt, Type at, int n, size_t total_bytes, const char* regime) {
    const TypeInfo* wi = type_info(wt);
    const TypeInfo* ai = type_info(at);
    const DotKernel* k1 = dot_kernel(wt, Arith::Avx2);
    const DotX4Kernel* k4 = dot_x4_kernel(wt, Arith::Avx2);
    if (wi == nullptr || ai == nullptr || k1 == nullptr || k4 == nullptr) {
        std::printf("%-8s %-8s skipped (no x4 kernel on this CPU)\n", wi ? wi->name : "?", regime);
        return;
    }
    const size_t w_row = row_bytes((int) wt, n), a_row = row_bytes((int) at, n);
    size_t rows = total_bytes / w_row;
    rows -= rows % 4;
    std::vector<uint8_t> w(rows * w_row), a(a_row);
    std::mt19937 rng(1234u ^ (uint32_t) wt);
    for (auto& b : w) b = (uint8_t) (rng() & 0xFF);
    for (auto& b : a) b = (uint8_t) (rng() & 0xFF);
    // activation scales: small normal values so nothing runs through slow denormal / NaN paths
    if (at == Type::Q8_K) {
        for (int i = 0; i < n / 256; ++i) {
            uint8_t* bp = a.data() + (size_t) i * ai->bytes;
            const float d = 0.01f;
            std::memcpy(bp, &d, 4);
            int16_t* bs = (int16_t*) (bp + 4 + 256);
            for (int g = 0; g < 16; ++g) bs[g] = 0;
        }
    } else {
        for (int i = 0; i < n / ai->blck; ++i) {
            const uint16_t h = 0x2C00;   // 0.0625
            std::memcpy(a.data() + (size_t) i * ai->bytes, &h, 2);
        }
    }
    std::vector<float> out(rows);
    const int passes = total_bytes > (64u << 20) ? 3 : 200;
    double best1 = 1e30, best4 = 1e30;
    volatile float sink = 0.f;
    for (int rep = 0; rep < 4; ++rep) {
        double t = now_s();
        for (int p = 0; p < passes; ++p)
            for (size_t r = 0; r < rows; ++r) k1->fn(n, &out[r], w.data() + r * w_row, a.data());
        best1 = std::min(best1, now_s() - t);
        sink += out[0];
        t = now_s();
        for (int p = 0; p < passes; ++p)
            for (size_t r = 0; r < rows; r += 4) k4->fn(n, &out[r], w.data() + r * w_row, w_row, a.data());
        best4 = std::min(best4, now_s() - t);
        sink += out[1];
    }
    const double gb = (double) passes * (double) rows * (double) w_row / 1e9;
    std::printf("%-8s %-6s n=%d  1-row (%s) %6.2f GB/s   x4 (%s) %6.2f GB/s   x4/1-row %.3f\n", wi->name, regime, n, k1->name, gb / best1,
                k4->name, gb / best4, best1 / best4);
}

int main() {
    const int n = 2560;   // Qwen3.8's hidden size: a gate/up row (a down row has the 512-wide ff)
    const struct { Type w, a; } types[] = {{Type::Q4_0, Type::Q8_0}, {Type::Q8_0, Type::Q8_0}, {Type::Q4_K, Type::Q8_K}, {Type::Q5_K, Type::Q8_K},
                                           {Type::Q6_K, Type::Q8_K}, {Type::IQ4_XS, Type::Q8_K}, {Type::IQ3_XXS, Type::Q8_K}, {Type::IQ3_S, Type::Q8_K}};
    for (const auto& t : types) {
        bench_type(t.w, t.a, n, 4u << 20, "cache");
        bench_type(t.w, t.a, n, 512u << 20, "dram");
    }
    return 0;
}
