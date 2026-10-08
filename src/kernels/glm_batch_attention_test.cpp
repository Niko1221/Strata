// Synthetic prompt MLA attention against an F32 softmax oracle; no model weights.
#include "strata/kernels/glm_batch.hpp"
#include <cuda_fp16.h>
#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <vector>

namespace gb = strata::kernels::glmb;
static void require(bool ok, const char* message) {
    if (!ok) { std::fprintf(stderr, "%s\n", message); std::exit(1); }
}
static void ck(cudaError_t e) { require(e == cudaSuccess, cudaGetErrorString(e)); }
template<class T> static T* upload(const std::vector<T>& x) {
    T* p = nullptr; ck(cudaMalloc((void**) &p, x.size() * sizeof(T)));
    ck(cudaMemcpy(p, x.data(), x.size() * sizeof(T), cudaMemcpyHostToDevice)); return p;
}

int main(int argc, char** argv) {
    const bool tensor = argc > 1 && std::strcmp(argv[1], "--tensor") == 0;
#ifdef _WIN32
    _putenv_s("STRATA_GLM_PREFILL_ATTN", tensor ? "" : "f32");
#else
    setenv("STRATA_GLM_PREFILL_ATTN", tensor ? "" : "f32", 1);
#endif
    cudaDeviceProp prop{}; ck(cudaGetDeviceProperties(&prop, 0));
    // WMMA needs the production kernel's 91 KiB shared arena. Refuse a silent F32 fallback.
    if (tensor && (prop.major < 7 || prop.sharedMemPerBlockOptin < 91136)) return 77;
    constexpr int T = 5, H = 32, KV = 512, ROWS = 101, SEL = 97;
    const float scale = 1.0f / std::sqrt(128.0f);
    std::vector<float> query(T * H * KV), latent(ROWS * KV), result(query.size());
    for (size_t i = 0; i < query.size(); ++i) query[i] = 0.3f * std::sin((float) i * 0.017f);
    for (size_t i = 0; i < latent.size(); ++i) latent[i] = 0.5f * std::cos((float) i * 0.013f);
    // Future rows contain a conspicuous poison: only causal cells [0, position] are selected.
    std::fill(latent.begin() + 97 * KV, latent.end(), 20.0f);
    std::vector<uint16_t> half(8 + latent.size() + 8, 0x3555);
    std::vector<int> cells(T * SEL, -1), counts = {1, 33, 97, 0, 33};
    for (int t = 0; t < 3; ++t)
        for (int s = 0; s < counts[t]; ++s) cells[t * SEL + s] = s % 7 == 6 ? -1 : s;
    float* dq = upload(query); float* dl = upload(latent); float* out = upload(result);
    uint16_t* dh = upload(half); int* dc = upload(cells); int* dn = upload(counts);
    gb::f32_to_f16(dl, dh + 8, latent.size(), nullptr);
    ck(cudaMemcpy(half.data(), dh, half.size() * 2, cudaMemcpyDeviceToHost));
    for (int i = 0; i < 8; ++i)
        require(half[i] == 0x3555 && half[half.size() - 1 - i] == 0x3555, "FP16 cache guard overwritten");
    for (size_t i = 0; i < latent.size(); ++i) {
        require(half[i + 8] == __half_as_ushort(__float2half_rn(latent[i])), "FP16 cache layout differs");
        latent[i] = __half2float(__ushort_as_half(half[i + 8]));
    }
    gb::mla_attn(dq, dh + 8, dc, dn, SEL, H, KV, scale, T, out, nullptr);
    ck(cudaDeviceSynchronize()); require(gb::launch_errors() == 0, "attention launch failed");
    ck(cudaMemcpy(result.data(), out, result.size() * sizeof(float), cudaMemcpyDeviceToHost));
    double worst = 0;
    for (int t = 0; t < T; ++t) for (int h = 0; h < H; ++h) {
        std::vector<double> scores(counts[t], -INFINITY);
        double maximum = -INFINITY, denominator = 0;
        for (int s = 0; s < counts[t]; ++s) if (cells[t * SEL + s] >= 0) {
            double dot = 0;
            for (int e = 0; e < KV; ++e)
                dot += query[(t * H + h) * KV + e] * (double) latent[cells[t * SEL + s] * KV + e];
            scores[s] = dot * scale; maximum = std::max(maximum, scores[s]);
        }
        for (double& score : scores) {
            score = score == -INFINITY ? 0.0 : std::exp(score - maximum);
            denominator += score;
        }
        for (int e = 0; e < KV; ++e) {
            double expected = 0;
            for (int s = 0; s < counts[t]; ++s) if (cells[t * SEL + s] >= 0)
                expected += scores[s] / denominator * latent[cells[t * SEL + s] * KV + e];
            const float got = result[(t * H + h) * KV + e];
            require(std::isfinite(got), "attention produced nonfinite output");
            worst = std::max(worst, std::fabs(got - expected));
        }
    }
    for (void* p : { (void*) dq, (void*) dl, (void*) out, (void*) dh, (void*) dc, (void*) dn }) ck(cudaFree(p));
    require(worst < 2e-3, "attention differs from F32 softmax oracle");
    std::printf("glm_batch_attention_test: PASS (%s, %zu values, max %.3e)\n", tensor ? "WMMA" : "F32", result.size(), worst);
}
