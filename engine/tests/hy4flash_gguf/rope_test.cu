// Check adjacent-pair RoPE against the CUDA angle formula and test norm preservation at long positions.
#include "strata/hy4flash_gguf/attention.hpp"

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <random>
#include <string>
#include <vector>

#if defined(__HIP_PLATFORM_AMD__) || defined(STRATA_USE_HIP)
#include <hip/hip_runtime.h>
#else
#include <cuda_runtime.h>
#endif

namespace {

int g_fail = 0;

void check(bool ok, const std::string& what) {
    std::printf("  %-70s %s\n", what.c_str(), ok ? "ok" : "FAIL");
    if (!ok) ++g_fail;
}

std::string sci(double v) {
    char b[32];
    std::snprintf(b, sizeof b, "%.3g", v);
    return b;
}

/// `recurrence = false` is ggml's CUDA rope (pos * powf), `true` is its CPU one. NORMAL's adjacent pairing
/// either way.
void cpu_rope(const std::vector<float>& x, int head_dim, int n_rot, int n_head, int n_tok,
              const std::vector<int32_t>& pos, float freq_base, std::vector<float>& out,
              bool recurrence = false) {
    const float theta_scale = std::pow(freq_base, -2.0f / (float) n_rot);
    out = x;
    for (int t = 0; t < n_tok; ++t)
        for (int h = 0; h < n_head; ++h) {
            float theta = (float) pos[(size_t) t];
            const size_t base = ((size_t) t * n_head + (size_t) h) * (size_t) head_dim;
            for (int j = 0; j < n_rot / 2; ++j) {
                if (!recurrence)
                    theta = (float) pos[(size_t) t] * std::pow(theta_scale, (float) j);
                const float c = std::cos(theta), s = std::sin(theta);
                const float x0 = x[base + (size_t) (2 * j)];
                const float x1 = x[base + (size_t) (2 * j + 1)];
                out[base + (size_t) (2 * j)] = x0 * c - x1 * s;
                out[base + (size_t) (2 * j + 1)] = x0 * s + x1 * c;
                theta *= theta_scale;
            }
        }
}

cudaStream_t g_stream{};

/// Allow host/device trigonometric rounding differences at large positions.
void one(const std::string& name, int head_dim, int n_rot, int n_head, int n_tok, float base, int first_pos,
         double tol = 1e-5) {
    std::mt19937 rng(20261106u + (unsigned) n_tok * 7u + (unsigned) first_pos);
    std::normal_distribution<float> nd(0.f, 1.f);
    std::vector<float> x((size_t) head_dim * n_head * n_tok);
    for (float& v : x) v = nd(rng);
    std::vector<int32_t> pos((size_t) n_tok);
    for (int i = 0; i < n_tok; ++i) pos[(size_t) i] = first_pos + i;

    std::vector<float> want, want_recurrence;
    cpu_rope(x, head_dim, n_rot, n_head, n_tok, pos, base, want);
    cpu_rope(x, head_dim, n_rot, n_head, n_tok, pos, base, want_recurrence, /*recurrence=*/true);

    float* dx = nullptr;
    float* dy = nullptr;
    int32_t* dp = nullptr;
    if (cudaMalloc(&dx, x.size() * 4) != cudaSuccess || cudaMalloc(&dy, x.size() * 4) != cudaSuccess ||
        cudaMalloc(&dp, pos.size() * 4) != cudaSuccess) {
        check(false, name + ": device allocation");
        return;
    }
    cudaMemcpy(dx, x.data(), x.size() * 4, cudaMemcpyHostToDevice);
    cudaMemcpy(dp, pos.data(), pos.size() * 4, cudaMemcpyHostToDevice);
    strata::hy4_flash::rope_norm(dx, dy, head_dim, n_rot, n_head, n_tok, dp, base, g_stream);
    cudaStreamSynchronize(g_stream);
    std::vector<float> got(x.size());
    cudaMemcpy(got.data(), dy, got.size() * 4, cudaMemcpyDeviceToHost);
    cudaFree(dx);
    cudaFree(dy);
    cudaFree(dp);

    double w = 0.0, w_rec = 0.0;
    for (size_t i = 0; i < got.size(); ++i) {
        w = std::max(w, std::fabs((double) got[i] - (double) want[i]));
        w_rec = std::max(w_rec, std::fabs((double) want[i] - (double) want_recurrence[i]));
    }
    check(w < tol, name + " (worst " + sci(w) + " vs CUDA contract, tol " + sci(tol) + ")");
    // Tight at every position: |(x0, x1)| is unchanged by a rotation, and the tail past n_rot is copied.
    double w_len = 0.0, w_tail = 0.0;
    for (int t = 0; t < n_tok; ++t)
        for (int h = 0; h < n_head; ++h) {
            const size_t b = ((size_t) t * (size_t) n_head + (size_t) h) * (size_t) head_dim;
            for (int j = 0; j < n_rot / 2; ++j) {
                const double i0 = x[b + (size_t) (2 * j)], i1 = x[b + (size_t) (2 * j + 1)];
                const double o0 = got[b + (size_t) (2 * j)], o1 = got[b + (size_t) (2 * j + 1)];
                const double in = std::sqrt(i0 * i0 + i1 * i1), on = std::sqrt(o0 * o0 + o1 * o1);
                w_len = std::max(w_len, std::fabs(on - in) / std::max(in, 1e-9));
            }
            for (int d = n_rot; d < head_dim; ++d)
                w_tail = std::max(w_tail, std::fabs((double) got[b + (size_t) d] - (double) x[b + (size_t) d]));
        }
    check(w_len < 1e-5, name + ": the pair length is preserved (worst " + sci(w_len) + ")");
    check(w_tail == 0.0, name + ": the tail past n_rot is copied through (worst " + sci(w_tail) + ")");
    // Recorded, not asserted: how far the CPU recurrence is from the CUDA formula at this position. Both
    // are ggml; the engine follows the one llama.cpp's CUDA backend runs.
    if (w_rec > 1e-6) std::printf("      (ggml CPU recurrence differs from ggml CUDA here by %s)\n", sci(w_rec).c_str());
}

void projection_guards() {
    float* d = nullptr;
    int32_t* dp = nullptr;
    cudaMalloc(&d, 1 << 20);
    cudaMalloc(&dp, 64);
    strata::hy4_flash::MlaProjected o{};
    auto throws = [&](int qk_head, int n_rot) {
        try {
            strata::hy4_flash::mla_project(d, d, 12, d, d, 12, d, 12, d, 64, 32, 16, qk_head, n_rot, 2, 1, dp,
                                         1e-5f, 1e7f, d, o, /*cur_in=*/strata::hy4_flash::MatIn{}, g_stream);
        } catch (const std::exception&) { return true; }
        return false;
    };
    check(throws(64, 64), "n_rot equal to qk_head is refused (there would be no nope half)");
    check(throws(64, 96), "n_rot beyond qk_head is refused");

    bool threw = false;
    try {
        strata::hy4_flash::gather_slice(d, d, 256, 200, 64, 1, 1, g_stream);
    } catch (const std::exception&) { threw = true; }
    check(threw, "gather_slice refuses a slice that runs past the row (200 + 64 > 256)");
    cudaFree(d);
    cudaFree(dp);
}

}  // namespace

int main() {
    if (cudaStreamCreate(&g_stream) != cudaSuccess) {
        std::printf("no CUDA stream: nothing to check\n");
        return 0;
    }
    std::printf("hy4-flash's geometry, both rope bases\n");
    one("sliding 1e4, 64 heads", 64, 64, 64, 5, 1e4f, 0);
    one("sparse 1e7, 32 heads", 64, 64, 32, 5, 1e7f, 0);
    std::printf("large positions, where the two ggml backends disagree and we follow CUDA\n");
    // Scale the tolerance for host/device trigonometric rounding at large positions.
    one("position 4096", 64, 64, 32, 3, 1e7f, 4096, 2e-3);
    one("position 1e6 (this model's context is 2^20)", 64, 64, 8, 2, 1e7f, 1000000, 3e-1);
    std::printf("a rotated slice inside a wider head: the tail is copied through\n");
    one("head_dim 256, n_rot 64", 256, 64, 4, 3, 1e7f, 7);

    projection_guards();
    std::printf("guards\n");
    {
        float* d = nullptr;
        int32_t* dp = nullptr;
        cudaMalloc(&d, 8192);
        cudaMalloc(&dp, 16);
        auto throws = [&](int head_dim, int n_rot, float base) {
            try {
                strata::hy4_flash::rope_norm(d, d, head_dim, n_rot, 1, 1, dp, base, g_stream);
            } catch (const std::exception&) { return true; }
            return false;
        };
        check(throws(64, 65, 1e7f), "an odd n_rot is refused (the rotation works in pairs)");
        check(throws(64, 128, 1e7f), "n_rot beyond head_dim is refused");
        check(throws(64, 64, 0.f), "a non-positive freq_base is refused");
        cudaFree(d);
        cudaFree(dp);
    }

    cudaStreamDestroy(g_stream);
    std::printf("%s\n", g_fail ? "FAILED" : "all ok");
    return g_fail ? 1 : 0;
}
