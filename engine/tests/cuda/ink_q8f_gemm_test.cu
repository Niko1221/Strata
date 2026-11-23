// The Q8_0 dense GEMM of the CUDA Inkling engine (k_ink_gemm_q8f + k_ink_quant_x_q8f, format 105): bit-exact against a host emulation of the
// same arithmetic, and a token's result must not depend on M (prefill chunk, batch or alone).
//   nvcc -std=c++17 -arch=sm_86 -I engine/src/inkling2 -I engine/src/inkling2/cuda -I engine/include engine/tests/cuda/ink_q8f_gemm_test.cu -o ink_q8f_gemm_test
// Skips (exit 77) below compute capability 8.0.
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <random>
#include <vector>

#include "ink_cuda.cuh"
#include "ink_kernels.cuh"
#include "ink_q8f_pack.h"

using namespace strata::ink;

#define CK(x) do { cudaError_t e_ = (x); if (e_ != cudaSuccess) { std::printf("CUDA error %s at %d\n", cudaGetErrorString(e_), __LINE__); return 2; } } while (0)

static void run_gemm(const uint8_t* d_w, int N, int K, const float* d_x, int M, float* d_y, int8_t* d_xq, float* d_xs) {
    const int nMT = (M + 7) / 8, nKB = K >> 5;
    k_ink_quant_x_q8f<<<(nMT * 8 * nKB + 7) / 8, 256>>>(d_x, d_xq, d_xs, M, K, nMT);
    InkMultiProj mp{};
    mp.W[0] = d_w; mp.Y[0] = d_y; mp.N[0] = N; mp.cumN[0] = 0; mp.cumN[1] = N; mp.count = 1;
    const int gx = (N / 16 + 1) / 2;
    const int mtw = nMT >= 8 ? 8 : (nMT >= 4 ? 4 : (nMT >= 2 ? 2 : 1));
    const dim3 grid(gx, (nMT + mtw - 1) / mtw);
    if (mtw == 8)      k_ink_gemm_q8f<8><<<grid, 64>>>(mp, d_xq, d_xs, M, K, nMT);
    else if (mtw == 4) k_ink_gemm_q8f<4><<<grid, 64>>>(mp, d_xq, d_xs, M, K, nMT);
    else if (mtw == 2) k_ink_gemm_q8f<2><<<grid, 64>>>(mp, d_xq, d_xs, M, K, nMT);
    else               k_ink_gemm_q8f<1><<<grid, 64>>>(mp, d_xq, d_xs, M, K, nMT);
}

int main() {
    int major = 0;
    cudaDeviceGetAttribute(&major, cudaDevAttrComputeCapabilityMajor, 0);
    if (major < 8) { std::printf("ink_q8f_gemm_test: skipped (needs sm_80+)\n"); return 77; }
    std::mt19937 rng(12345);
    int bad = 0;
    for (int K : {2304, 3072}) {
        const int N = 160, MMAX = 70, nKB = K / 32;
        // random Q8_0 matrix: blocks of fp16 scale + 32 int8
        std::vector<uint8_t> q8(static_cast<size_t>(N) * nKB * 34);
        for (int r = 0; r < N; ++r)
            for (int b = 0; b < nKB; ++b) {
                uint8_t* blk = q8.data() + (static_cast<size_t>(r) * nKB + b) * 34;
                const __half h = __float2half(0.001f + 0.02f * (rng() % 1000) / 1000.0f);
                std::memcpy(blk, &h, 2);
                for (int i = 0; i < 32; ++i) blk[2 + i] = static_cast<uint8_t>(static_cast<int>(rng() % 255) - 127);
            }
        std::vector<uint8_t> packed(ink2::ink_q8f_bytes(N, K));
        ink2::ink_q8f_pack(q8.data(), N, K, packed.data());
        std::vector<float> x(static_cast<size_t>(MMAX) * K);
        for (int m = 0; m < MMAX; ++m)
            for (int k = 0; k < K; ++k) {
                float v = (static_cast<int>(rng() % 20001) - 10000) / 3000.0f;
                if (m % 11 == 3 && (k / 32) % 5 == 0) v = 0.0f;   // some all-zero blocks
                x[static_cast<size_t>(m) * K + k] = v * (1.0f + (k % 7));
            }
        uint8_t* d_w; float *d_x, *d_y, *d_xs; int8_t* d_xq;
        CK(cudaMalloc(&d_w, packed.size())); CK(cudaMalloc(&d_x, x.size() * 4)); CK(cudaMalloc(&d_y, static_cast<size_t>(MMAX) * N * 4));
        CK(cudaMalloc(&d_xq, static_cast<size_t>(MMAX + 8) * K)); CK(cudaMalloc(&d_xs, static_cast<size_t>(MMAX + 8) * nKB * 4));
        CK(cudaMemcpy(d_w, packed.data(), packed.size(), cudaMemcpyHostToDevice));
        CK(cudaMemcpy(d_x, x.data(), x.size() * 4, cudaMemcpyHostToDevice));

        // host emulation of the same arithmetic
        std::vector<float> ref(static_cast<size_t>(MMAX) * N);
        for (int m = 0; m < MMAX; ++m) {
            std::vector<int8_t> q(K);
            std::vector<float> sx(nKB);
            for (int b = 0; b < nKB; ++b) {
                float am = 0.0f;
                for (int i = 0; i < 32; ++i) am = std::fmax(am, std::fabs(x[static_cast<size_t>(m) * K + b * 32 + i]));
                const float d = am * (1.0f / 127.0f);
                const float inv = d > 0.0f ? (127.0f / am) : 0.0f;
                sx[b] = d;
                for (int i = 0; i < 32; ++i) {
                    int v = static_cast<int>(std::nearbyint(x[static_cast<size_t>(m) * K + b * 32 + i] * inv));
                    q[b * 32 + i] = static_cast<int8_t>(std::max(-127, std::min(127, v)));
                }
            }
            for (int r = 0; r < N; ++r) {
                float acc = 0.0f;
                for (int b = 0; b < nKB; ++b) {
                    const uint8_t* blk = q8.data() + (static_cast<size_t>(r) * nKB + b) * 34;
                    __half h; std::memcpy(&h, blk, 2);
                    const float sw = __half2float(h);
                    int32_t c = 0;
                    for (int i = 0; i < 32; ++i) c += static_cast<int32_t>(static_cast<int8_t>(blk[2 + i])) * static_cast<int32_t>(q[b * 32 + i]);
                    acc = std::fmaf(static_cast<float>(c), sw * sx[b], acc);
                }
                ref[static_cast<size_t>(m) * N + r] = acc;
            }
        }

        std::vector<float> y_full(static_cast<size_t>(MMAX) * N);
        for (int M : {70, 64, 33, 9, 8, 7, 1}) {
            CK(cudaMemset(d_y, 0xFF, static_cast<size_t>(MMAX) * N * 4));
            run_gemm(d_w, N, K, d_x, M, d_y, d_xq, d_xs);
            CK(cudaGetLastError()); CK(cudaDeviceSynchronize());
            std::vector<float> y(static_cast<size_t>(MMAX) * N);
            CK(cudaMemcpy(y.data(), d_y, y.size() * 4, cudaMemcpyDeviceToHost));
            int n_ref = 0, n_inv = 0;
            for (size_t i = 0; i < static_cast<size_t>(M) * N; ++i) {
                if (std::memcmp(&y[i], &ref[i], 4) != 0) ++n_ref;
                if (M == 70) y_full[i] = y[i];
                else if (std::memcmp(&y[i], &y_full[i], 4) != 0) ++n_inv;
            }
            // rows past M must stay untouched (0xFFFFFFFF = NaN)
            int n_over = 0;
            for (size_t i = static_cast<size_t>(M) * N; i < y.size(); ++i) {
                uint32_t u; std::memcpy(&u, &y[i], 4);
                if (u != 0xFFFFFFFFu) ++n_over;
            }
            std::printf("K=%d M=%2d: %d values differ from the host emulation, %d differ from the M=70 run, %d written past M\n", K, M, n_ref, n_inv, n_over);
            bad += n_ref + n_inv + n_over;
        }
        cudaFree(d_w); cudaFree(d_x); cudaFree(d_y); cudaFree(d_xq); cudaFree(d_xs);
    }
    if (bad) { std::printf("ink_q8f_gemm_test: FAIL (%d)\n", bad); return 1; }
    std::printf("ink_q8f_gemm_test: OK\n");
    return 0;
}
