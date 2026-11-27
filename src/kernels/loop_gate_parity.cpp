// src/kernels/loop_gate_parity.cpp - #7295 M5's test for the recurrent-depth loop's pass-boundary gate.
//
// There is no prior implementation to be bit-exact against (loop_gate.hpp/cu is new code - see its header
// comment), so the reference here is this file's OWN from-scratch, double-precision transcription of the
// formula, independent of the kernel's block/thread layout. Plus two structural properties that hold for any
// correct implementation of `R' = z*R + (1-z)*R0` regardless of what the weights happen to be:
//
//   1. R == R0  =>  R_out == R, for ANY weights - the blend of two equal things is that thing, whatever z is.
//   2. z is a sigmoid, so 0 < z < 1 strictly - R_out must therefore lie strictly between R and R0 on every
//      element where R != R0 (never outside that range, never equal to either endpoint).
#include "strata/kernels/loop_gate.hpp"
#include "strata/kernels/bf16_bits.hpp"

#include <cuda_runtime.h>

#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <random>
#include <vector>

namespace {

void check(cudaError_t e, const char* what) {
    if (e != cudaSuccess) {
        std::fprintf(stderr, "%s: %s\n", what, cudaGetErrorString(e));
        std::exit(1);
    }
}

uint16_t bf16_bits(float f) {
    uint32_t i;
    std::memcpy(&i, &f, 4);
    // round-to-nearest-even, same rule as strata::kernels::bf16_from_f32
    i = (i + ((i >> 16) & 1u) + 0x7FFFu) & 0xFFFF0000u;
    return (uint16_t) (i >> 16);
}

/// The formula in loop_gate.hpp's file comment, transcribed directly, in double.
void reference(const std::vector<float>& R, const std::vector<float>& R0, const std::vector<float>& down,
               const std::vector<float>& up, int64_t n_embd, int64_t hc, int64_t rank,
               std::vector<float>& R_out) {
    R_out.resize((size_t) (hc * n_embd));
    std::vector<double> g((size_t) (hc * rank));
    for (int64_t c = 0; c < hc; ++c)
        for (int64_t k = 0; k < rank; ++k) {
            double acc = 0.0;
            for (int64_t d = 0; d < n_embd; ++d)
                acc += (double) R[(size_t) (c * n_embd + d)] * (double) down[(size_t) (k * n_embd + d)];
            g[(size_t) (c * rank + k)] = acc;
        }
    for (int64_t c = 0; c < hc; ++c)
        for (int64_t d = 0; d < n_embd; ++d) {
            double acc = 0.0;
            for (int64_t k = 0; k < rank; ++k) acc += g[(size_t) (c * rank + k)] * (double) up[(size_t) (d * rank + k)];
            const double z = 1.0 / (1.0 + std::exp(-acc));
            const size_t i = (size_t) (c * n_embd + d);
            R_out[i] = (float) (z * (double) R[i] + (1.0 - z) * (double) R0[i]);
        }
}

}  // namespace

int main() {
    const int64_t n_embd = 96, hc = 4, rank = 8;   // small, deliberately not the real geometry - a unit test
    std::mt19937 rng(7295);
    std::uniform_real_distribution<float> dist(-1.0f, 1.0f);

    std::vector<float> R((size_t) (hc * n_embd)), R0((size_t) (hc * n_embd));
    std::vector<float> down_f((size_t) (rank * n_embd)), up_f((size_t) (n_embd * rank));
    for (auto& v : R) v = dist(rng);
    for (auto& v : R0) v = dist(rng);
    for (auto& v : down_f) v = dist(rng) * 0.1f;   // small weights: keeps the sigmoid away from saturation
    for (auto& v : up_f) v = dist(rng) * 0.1f;

    std::vector<uint16_t> down_bf(down_f.size()), up_bf(up_f.size());
    for (size_t i = 0; i < down_f.size(); ++i) down_bf[i] = bf16_bits(down_f[i]);
    for (size_t i = 0; i < up_f.size(); ++i) up_bf[i] = bf16_bits(up_f[i]);
    // the reference reads the SAME rounded values the kernel does, so the only remaining difference is
    // the reduction, not the weight precision
    for (size_t i = 0; i < down_f.size(); ++i) down_f[i] = strata::kernels::f32_from_bf16(down_bf[i]);
    for (size_t i = 0; i < up_f.size(); ++i) up_f[i] = strata::kernels::f32_from_bf16(up_bf[i]);

    std::vector<float> want;
    reference(R, R0, down_f, up_f, n_embd, hc, rank, want);

    float *d_R, *d_R0, *d_out, *d_ws;
    uint16_t *d_down, *d_up;
    const strata::kernels::LoopGateShapes shapes{n_embd, hc, rank};
    const size_t ws_bytes = strata::kernels::loop_gate_workspace_bytes(shapes);
    if (ws_bytes != (size_t) (hc * rank) * sizeof(float)) {
        std::fprintf(stderr, "loop_gate_workspace_bytes: got %zu, want %zu\n", ws_bytes,
                     (size_t) (hc * rank) * sizeof(float));
        return 1;
    }
    check(cudaMalloc(&d_R, R.size() * 4), "malloc R");
    check(cudaMalloc(&d_R0, R0.size() * 4), "malloc R0");
    check(cudaMalloc(&d_out, R.size() * 4), "malloc out");
    check(cudaMalloc(&d_ws, ws_bytes), "malloc ws");
    check(cudaMalloc(&d_down, down_bf.size() * 2), "malloc down");
    check(cudaMalloc(&d_up, up_bf.size() * 2), "malloc up");
    check(cudaMemcpy(d_R, R.data(), R.size() * 4, cudaMemcpyHostToDevice), "copy R");
    check(cudaMemcpy(d_R0, R0.data(), R0.size() * 4, cudaMemcpyHostToDevice), "copy R0");
    check(cudaMemcpy(d_down, down_bf.data(), down_bf.size() * 2, cudaMemcpyHostToDevice), "copy down");
    check(cudaMemcpy(d_up, up_bf.data(), up_bf.size() * 2, cudaMemcpyHostToDevice), "copy up");

    strata::kernels::loop_gate_mix(d_R, d_R0, d_down, d_up, shapes, d_ws, d_out, nullptr);

    std::vector<float> got(R.size());
    check(cudaMemcpy(got.data(), d_out, got.size() * 4, cudaMemcpyDeviceToHost), "copy out");

    double max_abs_err = 0.0;
    for (size_t i = 0; i < got.size(); ++i) max_abs_err = std::max(max_abs_err, (double) std::fabs(got[i] - want[i]));
    const double tol = 2.0e-4;   // bf16 weights plus a float accumulation vs. the double reference
    if (max_abs_err > tol) {
        std::fprintf(stderr, "loop_gate_parity: mixed vs reference max abs err %.6g (tol %.3g)\n", max_abs_err, tol);
        return 1;
    }
    std::fprintf(stderr, "loop_gate_parity: mixed vs reference OK, max abs err %.3g\n", max_abs_err);

    // Property 1: R == R0 => R_out == R, for the SAME weights, regardless of z.
    {
        check(cudaMemcpy(d_R0, R.data(), R.size() * 4, cudaMemcpyHostToDevice), "copy R as R0");
        strata::kernels::loop_gate_mix(d_R, d_R0, d_down, d_up, shapes, d_ws, d_out, nullptr);
        std::vector<float> same(R.size());
        check(cudaMemcpy(same.data(), d_out, same.size() * 4, cudaMemcpyDeviceToHost), "copy out (R==R0)");
        for (size_t i = 0; i < same.size(); ++i) {
            if (std::fabs(same[i] - R[i]) > 1.0e-5f) {
                std::fprintf(stderr, "loop_gate_parity: R==R0 property failed at %zu: %.6g vs %.6g\n", i, same[i], R[i]);
                return 1;
            }
        }
        std::fprintf(stderr, "loop_gate_parity: R==R0 identity property OK\n");
        check(cudaMemcpy(d_R0, R0.data(), R0.size() * 4, cudaMemcpyHostToDevice), "restore R0");
    }

    // Property 2: every output lies strictly between R and R0 (0 < z < 1 always).
    {
        bool ok = true;
        for (size_t i = 0; i < got.size(); ++i) {
            const float lo = std::min(R[i], R0[i]), hi = std::max(R[i], R0[i]);
            if (R[i] == R0[i]) continue;
            if (!(got[i] > lo && got[i] < hi)) {
                std::fprintf(stderr, "loop_gate_parity: betweenness failed at %zu: got %.6g, range (%.6g, %.6g)\n", i,
                             got[i], lo, hi);
                ok = false;
            }
        }
        if (!ok) return 1;
        std::fprintf(stderr, "loop_gate_parity: strict-betweenness property OK\n");
    }

    cudaFree(d_R); cudaFree(d_R0); cudaFree(d_out); cudaFree(d_ws); cudaFree(d_down); cudaFree(d_up);
    std::fprintf(stderr, "loop_gate_parity: PASS\n");
    return 0;
}
