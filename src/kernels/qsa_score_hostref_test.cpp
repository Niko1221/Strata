// V100 (sm_70) unit test for strata::kernels::native_qsa_score (the sm_70 fallback path)
// against an independent CPU reference.  Validates: per-head ReLU of 128-dim dot products with
// tf32-truncated inputs, left-associative F32 head addition, block bias, the 1e9 incomplete-tail
// bonus, writes to cells [0,n) only, and that cells [n, max_cells) stay untouched.
#include "strata/kernels/native_qsa_score.hpp"
#include <cuda_runtime.h>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cmath>
#include <cstring>
#include <random>
#include <vector>

#define CK(e) do { cudaError_t _e = (e); if (_e != cudaSuccess) { \
    std::fprintf(stderr, "cuda: %s at %s:%d\n", cudaGetErrorString(_e), __FILE__, __LINE__); return 1; } } while (0)

static float trunc_tf32(float x) {
    uint32_t u;
    std::memcpy(&u, &x, 4);
    u &= 0xFFFFE000u;
    std::memcpy(&x, &u, 4);
    return x;
}

int main() {
    const int D = 128, H = 4, R = 4;
    const int max_blocks = 577;              // 576 (multiple of 64) + 1 spare, per the header contract
    const int max_cells = (max_blocks - 1) * 4;  // = 2304 >= 2051
    std::mt19937 rng(12345);
    std::uniform_real_distribution<float> dist(-2.0f, 2.0f);

    std::vector<float> pooled((size_t)max_blocks * D), query((size_t)H * D), bias(max_blocks),
                        cells((size_t)max_cells), ref((size_t)max_cells);
    const float SENT = 1e30f;
    for (auto& x : cells) x = SENT;
    for (auto& x : pooled) x = dist(rng);
    for (auto& x : query) x = dist(rng);
    for (auto& x : bias) x = dist(rng);

    float *d_pooled, *d_query, *d_bias, *d_cells;
    int32_t* d_step;
    CK(cudaMalloc(&d_pooled, (size_t)max_blocks * D * 4));
    CK(cudaMalloc(&d_query, (size_t)H * D * 4));
    CK(cudaMalloc(&d_bias, (size_t)max_blocks * 4));
    CK(cudaMalloc(&d_cells, (size_t)max_cells * 4));
    CK(cudaMalloc(&d_step, 4 * sizeof(int32_t)));
    cudaStream_t stream;
    CK(cudaStreamCreate(&stream));
    CK(cudaMemcpy(d_pooled, pooled.data(), (size_t)max_blocks * D * 4, cudaMemcpyHostToDevice));
    CK(cudaMemcpy(d_query, query.data(), (size_t)H * D * 4, cudaMemcpyHostToDevice));
    CK(cudaMemcpy(d_bias, bias.data(), (size_t)max_blocks * 4, cudaMemcpyHostToDevice));
    CK(cudaMemcpy(d_cells, cells.data(), (size_t)max_cells * 4, cudaMemcpyHostToDevice));

    strata::kernels::QsaShapes s = strata::kernels::qsa_real_shapes();  // idx_dim 128 / 4 heads / block 4 / top-k 2048

    int bad = 0;
    const int nvals[] = {1, 2, 3, 4, 5, 7, 63, 64, 65, 129, 512, 2048, 2049, 2050, 2051};
    for (int n : nvals) {
        const int full = n / R;
        for (int cell = 0; cell < max_cells; ++cell) ref[(size_t)cell] = SENT;
        for (int row = 0; row <= full; ++row) {
            float sum = 0.0f;
            for (int h = 0; h < H; ++h) {
                double dot = 0.0;
                for (int d = 0; d < D; ++d)
                    dot += (double)trunc_tf32(pooled[(size_t)row * D + d]) *
                           (double)trunc_tf32(query[(size_t)h * D + d]);
                sum += fmaxf((float)dot, 0.0f);   // left-associative, like the kernel
            }
            sum += bias[row];
            sum += (row == full && n % R) ? 1e9f : 0.0f;
            for (int j = 0; j < R; ++j) {
                const int cell = row * R + j;
                if (cell < n) ref[(size_t)cell] = sum;
            }
        }
        int32_t host_step[4] = {n - 1, n, full, n < 2051 ? n : 2051};
        CK(cudaMemcpy(d_step, host_step, sizeof(host_step), cudaMemcpyHostToDevice));
        CK(cudaMemcpy(d_cells, cells.data(), (size_t)max_cells * 4, cudaMemcpyHostToDevice));
        strata::kernels::native_qsa_score(d_pooled, d_query, d_bias, s, d_step, max_blocks, max_cells,
                                          d_cells, (void*) stream);
        CK(cudaStreamSynchronize(stream));
        std::vector<float> got((size_t)max_cells);
        CK(cudaMemcpy(got.data(), d_cells, (size_t)max_cells * 4, cudaMemcpyDeviceToHost));
        double worst = 0.0;
        int mism = 0;
        for (int cell = 0; cell < max_cells; ++cell) {
            const float want = ref[(size_t)cell], have = got[(size_t)cell];
            if (want == SENT && have == SENT) continue;
            if (want != SENT && have == SENT) { ++mism; continue; }
            if (want == SENT && have != SENT) { ++mism; continue; }
            const double scale = std::max(1.0, std::fabs((double)want));
            const double err = std::fabs((double)want - (double)have) / scale;
            worst = std::max(worst, err);
            if (err > 2e-4) ++mism;
        }
        std::printf("n=%4d full=%4d: %s (worst rel err %.3e, mism %d)\n", n, full,
                    mism ? "FAIL" : "ok", worst, mism);
        if (mism) bad = 1;
    }
    std::printf(bad ? "qsa_score_test FAIL\n" : "qsa_score_test OK\n");
    return bad;
}
