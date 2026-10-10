// src/artifact/exl3_pack_check.cpp - build the engine WeightTable from an EXL3 model and validate it.
#include "strata/artifact/exl3_pack.hpp"

#include "strata/core/layout.hpp"
#include "strata/kernels/exl3.hpp"

#include <cuda_runtime.h>

#include <cstdio>
#include <cstring>
#include <string>
#include <vector>

int main(int argc, char** argv) {
    if (argc < 2) { std::fprintf(stderr, "usage: %s <exl3_dir> [--run]\n", argv[0]); return 2; }
    try {
        strata::core::Exl3Pack pack(argv[1]);
        strata::core::WeightTable wt;
        std::string err;
        if (!pack.build(wt, err)) { std::fprintf(stderr, "build failed: %s\n", err.c_str()); return 1; }
        std::printf("roles: %zu   EXL3 linears: %.1f MiB   staged arena: %.1f MiB\n", wt.all().size(),
                    (double) pack.exl3_bytes() / (1 << 20), (double) pack.arena_bytes() / (1 << 20));
        std::printf("check_all: OK\n");
        if (argc < 3) return 0;
        // Run every dense EXL3 projection on the real weights, exactly as the engine's gemv_quantized does.
        strata::kernels::exl3_gemv_reserve(8192, 262144);
        int bad = 0, ran = 0;
        for (const auto& kv : wt.all()) {
            const strata::core::WeightRef& r = kv.second;
            if (r.exl3 == nullptr) continue;
            const auto* m = (const strata::kernels::Exl3Mat*) r.exl3;
            const int k = m->ki * 16, n = m->nj * 16;
            float* dx = nullptr; float* dy = nullptr;
            (void) cudaMalloc(&dx, (size_t) k * 4); (void) cudaMalloc(&dy, (size_t) n * 4);
            (void) cudaMemset(dx, 0, (size_t) k * 4);
            strata::kernels::exl3_gemv_f32(dx, m->suh, m->svh, m->trellis, m->ki, m->nj, m->bits, m->cb, dy, nullptr);
            const cudaError_t e = cudaDeviceSynchronize();
            if (e != cudaSuccess) { std::printf("  FAULT %-32s ki=%d nj=%d bits=%d: %s\n", kv.first.c_str(), m->ki, m->nj, m->bits, cudaGetErrorString(e)); ++bad; }
            (void) cudaFree(dx); (void) cudaFree(dy);
            ++ran;
        }
        std::printf("ran %d EXL3 projections, %d faults\n", ran, bad);
        return bad ? 1 : 0;
    } catch (const std::exception& e) {
        std::fprintf(stderr, "error: %s\n", e.what());
        return 1;
    }
}
