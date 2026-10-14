// src/kernels/exl3_native_check.cpp - what the dense EXL3 linears would cost if they were requantized to a
// ggml K-quant at load and served by the engine's native MMVQ instead of decoded from the trellis.
//
// Usage: strata-exl3-native <model_dir> <base_name> [q4k|q5k]
#include "strata/artifact/exl3_model.hpp"
#include "strata/kernels/exl3.hpp"
#include "strata/kernels/native_mmvq.hpp"
#include "strata/kernels/cpu/exl3.hpp"

#include <ggml.h>

#include <cuda_runtime.h>

#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <string>
#include <vector>

using namespace strata;
using namespace strata::kernels;

int main(int argc, char** argv) {
    if (argc < 3) { std::fprintf(stderr, "usage: %s <model_dir> <base_name> [q4k|q5k]\n", argv[0]); return 2; }
    try {
        Exl3Model m(argv[1]);
        Exl3Linear L = m.linear(argv[2]);
        const std::string kind = argc > 3 ? argv[3] : "q5k";
        const ggml_type type = kind == "q4k" ? GGML_TYPE_Q4_K : kind == "q8" ? GGML_TYPE_Q8_0 : GGML_TYPE_Q5_K;
        const int ki = L.ki, nj = L.nj;
        const int n_in = ki * 16, n_out = nj * 16;
        std::printf("%s: in=%d out=%d bits=%d cb=%d\n", argv[2], n_in, n_out, L.bits, L.cb);

        // ---- 1. reconstruct to fp16 on the GPU (the reconstruct takes DEVICE pointers: upload first)
        const int words = 256 * L.bits / 16;
        uint16_t *d_tr = nullptr, *d_suh = nullptr, *d_svh = nullptr, *d_w = nullptr;
        if (cudaMalloc(&d_tr, (size_t) ki * nj * words * 2) != cudaSuccess ||
            cudaMalloc(&d_suh, (size_t) n_in * 2) != cudaSuccess ||
            cudaMalloc(&d_svh, (size_t) n_out * 2) != cudaSuccess ||
            cudaMalloc(&d_w, (size_t) n_in * n_out * 2) != cudaSuccess) { std::fprintf(stderr, "malloc\n"); return 1; }
        (void) cudaMemcpy(d_tr, L.trellis, (size_t) ki * nj * words * 2, cudaMemcpyHostToDevice);
        (void) cudaMemcpy(d_suh, L.suh, (size_t) n_in * 2, cudaMemcpyHostToDevice);
        (void) cudaMemcpy(d_svh, L.svh, (size_t) n_out * 2, cudaMemcpyHostToDevice);
        exl3_reconstruct_weight(d_tr, ki, nj, L.bits, L.cb, d_suh, d_svh, d_w, nullptr);
        const cudaError_t re = cudaDeviceSynchronize();
        if (re != cudaSuccess) { std::fprintf(stderr, "reconstruct failed: %s\n", cudaGetErrorString(re)); return 1; }
        // the reconstruct is [in][out]; a block quant stores a row of n_in per output
        uint16_t* d_wt = nullptr;
        if (cudaMalloc(&d_wt, (size_t) n_in * n_out * 2) != cudaSuccess) { std::fprintf(stderr, "malloc\n"); return 1; }
        exl3_transpose_f16(d_w, d_wt, n_in, n_out, nullptr);
        std::vector<uint16_t> h_w((size_t) n_in * n_out);
        (void) cudaMemcpy(h_w.data(), d_wt, h_w.size() * 2, cudaMemcpyDeviceToHost);
        // ---- 1b. orientation check against the CPU reconstruct
        {
            std::vector<uint16_t> cpu_w((size_t) n_in * n_out);
            strata::kernels::cpu::exl3_reconstruct_weight(L.trellis, L.ki, L.nj, L.bits,
                                                          (strata::kernels::cpu::Exl3Codebook) L.cb, L.suh, L.svh,
                                                          cpu_w.data());
            auto h2f = [](uint16_t h) {
                unsigned sign = (unsigned) (h >> 15) << 31, exp = (h >> 10) & 0x1F, man = h & 0x3FF, f;
                if (exp == 0) { if (!man) f = sign; else { unsigned e2 = 127 - 15 + 1; unsigned mm = man; while (!(mm & 0x400)) { mm <<= 1; --e2; } f = sign | (e2 << 23) | ((mm & 0x3FF) << 13); } }
                else if (exp == 0x1F) f = sign | 0x7F800000u | (man << 13);
                else f = sign | ((exp + 127 - 15) << 23) | (man << 13);
                float o; std::memcpy(&o, &f, 4); return o;
            };
            double same = 0, transp = 0, den = 0;
            for (int i = 0; i < n_in; ++i)
                for (int j = 0; j < n_out; ++j) {
                    const double c = h2f(cpu_w[(size_t) i * n_out + j]);
                    const double a = h2f(h_w[(size_t) i * n_out + j]);
                    const double b = h2f(h_w[(size_t) j * n_in + i]);
                    same += (a - c) * (a - c);
                    transp += (b - c) * (b - c);
                    den += c * c;
                }
            std::printf("orientation: as-is rel %.3e, transposed rel %.3e\n", std::sqrt(same / den), std::sqrt(transp / den));
        }

        // ---- 2. requantize on the CPU, one row at a time (ggml's own from_float)
        const size_t row_bytes = ggml_row_size(type, n_in);
        std::vector<uint8_t> q(row_bytes * (size_t) n_out);
        const ggml_type_traits* tr = ggml_get_type_traits(type);
        std::vector<float> row((size_t) n_in);
        const auto tq0 = std::chrono::high_resolution_clock::now();
        for (int r = 0; r < n_out; ++r) {
            const uint16_t* src = h_w.data() + (size_t) r * n_in;
            for (int i = 0; i < n_in; ++i) {
                const unsigned h = src[i];
                const unsigned sign = (h >> 15) << 31, exp = (h >> 10) & 0x1F, man = h & 0x3FF;
                unsigned f;
                if (exp == 0) { if (!man) f = sign; else { unsigned e2 = 127 - 15 + 1; unsigned mm = man; while (!(mm & 0x400)) { mm <<= 1; --e2; } f = sign | (e2 << 23) | ((mm & 0x3FF) << 13); } }
                else if (exp == 0x1F) f = sign | 0x7F800000u | (man << 13);
                else f = sign | ((exp + 127 - 15) << 23) | (man << 13);
                std::memcpy(&row[(size_t) i], &f, 4);
            }
            tr->from_float_ref(row.data(), q.data() + (size_t) r * row_bytes, n_in);
        }
        const auto tq1 = std::chrono::high_resolution_clock::now();
        std::printf("requantize to %s: %.1f MiB (trellis %.1f MiB), %.1f s on one thread\n", kind.c_str(),
                    (double) q.size() / (1 << 20), (double) (L.ki * L.nj * (256 * L.bits / 16) * 2) / (1 << 20),
                    std::chrono::duration<double>(tq1 - tq0).count());
        (void) cudaFree(d_w);

        // ---- 3. the two GEMVs on the same matrix
        void* d_q = nullptr; void* d_scratch = nullptr; float* d_x = nullptr; float* d_y = nullptr;
        if (cudaMalloc(&d_q, q.size()) != cudaSuccess || cudaMalloc(&d_scratch, native_q8_1_bytes(n_in)) != cudaSuccess ||
            cudaMalloc(&d_x, (size_t) n_in * 4) != cudaSuccess || cudaMalloc(&d_y, (size_t) n_out * 4) != cudaSuccess) {
            std::fprintf(stderr, "cudaMalloc failed\n"); return 1;
        }
        (void) cudaMemcpy(d_q, q.data(), q.size(), cudaMemcpyHostToDevice);
        {
            // a NONZERO input: a zeroed one makes both GEMVs output zero and their difference meaningless
            std::vector<float> x((size_t) n_in);
            for (int i = 0; i < n_in; ++i) x[(size_t) i] = std::sin(0.37f * (float) i) * 0.3f;
            (void) cudaMemcpy(d_x, x.data(), x.size() * 4, cudaMemcpyHostToDevice);
        }

        cudaStream_t st = nullptr;
        if (cudaStreamCreate(&st) != cudaSuccess) { std::fprintf(stderr, "stream\n"); return 1; }
        const int iters = 200;
        (void) cudaDeviceSynchronize();
        auto t0 = std::chrono::high_resolution_clock::now();
        for (int it = 0; it < iters; ++it) native_q5_k_f32(d_q, d_x, d_scratch, d_y, n_in, n_out, 1, st);
        (void) cudaDeviceSynchronize();
        auto t1 = std::chrono::high_resolution_clock::now();
        const double native_ms = std::chrono::duration<double, std::milli>(t1 - t0).count() / iters;

        std::vector<float> y_native((size_t) n_out);
        (void) cudaMemcpy(y_native.data(), d_y, y_native.size() * 4, cudaMemcpyDeviceToHost);

        auto t2 = std::chrono::high_resolution_clock::now();
        for (int it = 0; it < iters; ++it)
            exl3_gemv_f32(d_x, d_suh, d_svh, d_tr, ki, nj, L.bits, L.cb, d_y, st);
        (void) cudaDeviceSynchronize();
        auto t3 = std::chrono::high_resolution_clock::now();
        const double exl3_ms = std::chrono::duration<double, std::milli>(t3 - t2).count() / iters;

        std::vector<float> y_exl3((size_t) n_out);
        (void) cudaMemcpy(y_exl3.data(), d_y, y_exl3.size() * 4, cudaMemcpyDeviceToHost);
        double num = 0, den = 0;
        for (int i = 0; i < n_out; ++i) { const double d = (double) y_native[(size_t) i] - y_exl3[(size_t) i]; num += d * d; den += (double) y_exl3[(size_t) i] * y_exl3[(size_t) i]; }
        std::printf("native %s: %.1f us   exl3_gemv_f32: %.1f us   speedup %.2fx   rel(y) %.2e\n", kind.c_str(),
                    native_ms * 1000.0, exl3_ms * 1000.0, exl3_ms / native_ms, std::sqrt(num) / (std::sqrt(den) + 1e-30));
        return 0;
    } catch (const std::exception& e) {
        std::fprintf(stderr, "error: %s\n", e.what());
        return 1;
    }
}
