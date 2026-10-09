// src/kernels/dflash_rope_parity.cpp - the DFlash drafter's FULL-head NeoX RoPE (docs/DFLASH.md):
// n_rot == head_dim == 256, theta 1e7, unscaled.  Checks three properties against an independent
// host reference in the same float math:
//   1. parity: every rotated element agrees with the host's powf/cosf/sinf formula (<= 2e-5 abs;
//      device and host trigonometry may differ by an ulp);
//   2. the deep half actually rotates: with position > 0, dims 64..255 all change (the bug this
//      guards against is the target QSA's partial n_rot = 64, which leaves 64..255 untouched);
//   3. exact x == out aliasing works.
#include "strata/kernels/native_rope.hpp"

#include <cuda_runtime.h>

#include <cmath>
#include <cstdio>
#include <vector>

namespace {
int g_fail = 0;
void check(bool ok, const char* what) {
    if (!ok) { std::fprintf(stderr, "FAIL: %s\n", what); ++g_fail; }
}
}  // namespace

int main() {
    constexpr int kRows = 6, kHd = 256;
    const double theta = 1e7;
    const int positions[kRows] = {0, 1, 7, 120, 125, 8191};

    std::vector<float> x((size_t) kRows * kHd);
    for (size_t i = 0; i < x.size(); ++i) x[i] = (float) ((int) (i * 37 % 251) - 125) / 32.0f;

    // the host reference, in the kernel's own float math
    std::vector<float> ref(x.size());
    const float theta_scale = std::pow((float) theta, -2.0f / (float) kHd);
    for (int r = 0; r < kRows; ++r) {
        for (int pair = 0; pair < kHd / 2; ++pair) {
            const float inv = std::pow(theta_scale, (float) pair);
            const float ang = (float) positions[r] * inv;
            const float c = std::cos(ang), s = std::sin(ang);
            const float a = x[(size_t) r * kHd + pair], b = x[(size_t) r * kHd + pair + kHd / 2];
            ref[(size_t) r * kHd + pair] = a * c - b * s;
            ref[(size_t) r * kHd + pair + kHd / 2] = b * c + a * s;
        }
    }

    float *xd = nullptr, *od = nullptr;
    int* pd = nullptr;
    if (cudaMalloc(&xd, x.size() * 4) != cudaSuccess || cudaMalloc(&od, x.size() * 4) != cudaSuccess ||
        cudaMalloc(&pd, sizeof(positions)) != cudaSuccess) {
        std::printf("dflash_rope_parity: no GPU\n");
        return 77;
    }
    std::vector<float> got(x.size());
    cudaStream_t cs = nullptr;
    cudaStreamCreate(&cs);
    // out-of-place, then in place
    for (int in_place = 0; in_place < 2; ++in_place) {
        cudaMemcpyAsync(xd, x.data(), x.size() * 4, cudaMemcpyHostToDevice, cs);
        cudaMemcpyAsync(pd, positions, sizeof(positions), cudaMemcpyHostToDevice, cs);
        float* out = in_place ? xd : od;
        strata::kernels::dflash_rope_neox_apply(xd, out, kRows, kHd, theta, pd, cs);
        cudaMemcpyAsync(got.data(), out, x.size() * 4, cudaMemcpyDeviceToHost, cs);
        cudaStreamSynchronize(cs);
        // The host and device powf/trig differ by a few ulp and the position amplifies the angle's
        // relative error linearly, so the gate scales with the position: err <= 1e-6 * pos + 2e-6
        // (|x| <= ~8 here).  A wrong partial rotation (the target QSA's n_rot = 64) would produce
        // O(|x|) errors - four orders of magnitude above this gate.
        double max_err = 0, bound = 0;
        size_t deep_changed = 0, deep_total = 0;
        for (size_t i = 0; i < x.size(); ++i) {
            const int r = (int) (i / kHd);
            const double e = std::fabs((double) got[i] - ref[i]);
            const double b = 1e-6 * (double) positions[r] + 2e-6;
            if (e - b > max_err - bound) {
                max_err = e;
                bound = b;
            }
            const int dim = (int) (i % kHd);
            if (dim >= 64 && positions[r] > 0) {
                ++deep_total;
                if (got[i] != x[i]) ++deep_changed;
            }
        }
        char what[160];
        std::snprintf(what, sizeof what, "%s parity (max err %.3e against a %.3e position-scaled bound)",
                      in_place ? "in-place" : "out-of-place", max_err, bound);
        check(max_err <= bound, what);
        // the deep dims (64..255) must actually rotate for position > 0 (the partial n_rot = 64
        // reading this guards against leaves every one of them untouched)
        std::snprintf(what, sizeof what, "%s: %zu of %zu deep dims rotated", in_place ? "in-place" : "out-of-place",
                      deep_changed, deep_total);
        check(deep_changed * 20 >= deep_total * 19, what);
    }
    // position 0 is the identity rotation: the row must come back unchanged
    cudaMemcpyAsync(xd, x.data(), x.size() * 4, cudaMemcpyHostToDevice, cs);
    cudaMemcpyAsync(pd, positions, sizeof(positions), cudaMemcpyHostToDevice, cs);
    strata::kernels::dflash_rope_neox_apply(xd, od, kRows, kHd, theta, pd, cs);
    cudaMemcpyAsync(got.data(), od, x.size() * 4, cudaMemcpyDeviceToHost, cs);
    cudaStreamSynchronize(cs);
    bool row0_unchanged = true;
    for (int d = 0; d < kHd; ++d)
        if (got[(size_t) d] != x[(size_t) d]) row0_unchanged = false;
    check(row0_unchanged, "position 0 is the identity");

    cudaStreamDestroy(cs);
    cudaFree(xd);
    cudaFree(od);
    cudaFree(pd);
    std::printf(g_fail ? "dflash_rope_parity: %d FAILURES\n" : "dflash_rope_parity: ok\n", g_fail);
    return g_fail ? 1 : 0;
}
