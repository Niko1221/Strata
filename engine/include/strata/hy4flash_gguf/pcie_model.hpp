#pragma once
#include <algorithm>
#include <cmath>
#include <cstdint>

namespace strata::hy4_flash {
struct PcieModel {
    bool on = false;
    double a = 0, c = 0, d = 0, p0 = 0, g0 = 0, g = 0;   ///< the fits (read by `pick`)
    int np0 = 0;
    double A[3][3] = {}, B[3] = {};   ///< the CPU fit's normal equations, [1, n_c, n_p]
    double C[2][2] = {}, D[2] = {};   ///< the GPU fit's, [1, n_p]
    int ncpu = 0, ngpu = 0, ngpu_p = 0;   ///< samples in each (ngpu_p: with a PCIe expert)
    uint32_t explore = 0;
    static void mean(double& v, int& n, double x) {   // the first 16 evenly, then 1/20 each
        if (n < 16) v += (x - v) / ++n;
        else v += 0.05 * (x - v);
    }
    /// One layer's times: the pool's `p` with `n_c` CPU and `n_p` PCIe experts, the GPU's `gpu` (ms).
    void sample(int n_c, int n_p, double p, double gpu) {
        sample_cpu(n_c, n_p, p);
        sample_gpu(n_p, gpu);
    }
    void sample_cpu(int n_c, int n_p, double p, double weight = 1.0) {
        constexpr double lam = 0.995;   // ~200 layers' memory: a few windows
        if (n_c == 0) mean(p0, np0, p);
        else {
            const double x[3] = {1.0, (double) n_c, (double) n_p};
            for (int i = 0; i < 3; ++i) {
                const double wx = weight * x[i];
                B[i] = lam * B[i] + wx * p;
                for (int j = 0; j < 3; ++j) A[i][j] = lam * A[i][j] + wx * x[j];
            }
            ++ncpu;
        }
    }
    void sample_gpu(int n_p, double gpu) {
        constexpr double lam = 0.995;
        const double y[2] = {1.0, (double) n_p};
        for (int i = 0; i < 2; ++i) {
            D[i] = lam * D[i] + y[i] * gpu;
            for (int j = 0; j < 2; ++j) C[i][j] = lam * C[i][j] + y[i] * y[j];
        }
        ++ngpu;
        ngpu_p += n_p > 0;
    }
    /// Solve both fits (after a window). A ridge keeps a fit whose counts barely vary near its mean ratio.
    void solve() {
        solve_cpu();
        solve_gpu();
    }
    void solve_cpu() {
        if (ncpu >= 8) {
            double M[3][4];
            for (int i = 0; i < 3; ++i) {
                for (int j = 0; j < 3; ++j) M[i][j] = A[i][j] + (i == j && i > 0 ? 1e-3 * A[0][0] : 0.0);
                M[i][3] = B[i];
            }
            bool ok = true;
            for (int k = 0; k < 3 && ok; ++k) {   // Gauss-Jordan, partial pivot
                int piv = k;
                for (int i = k + 1; i < 3; ++i) if (std::abs(M[i][k]) > std::abs(M[piv][k])) piv = i;
                if (std::abs(M[piv][k]) < 1e-12) { ok = false; break; }
                for (int j = 0; j < 4; ++j) std::swap(M[k][j], M[piv][j]);
                for (int i = 0; i < 3; ++i) {
                    if (i == k) continue;
                    const double f = M[i][k] / M[k][k];
                    for (int j = k; j < 4; ++j) M[i][j] -= f * M[k][j];
                }
            }
            if (ok) {
                a = std::max(M[0][3] / M[0][0], 0.0);
                c = std::max(M[1][3] / M[1][1], 0.0);
                d = std::clamp(M[2][3] / M[2][2], 0.0, c);   // a PCIe expert slows the pool at most as one of its own
            }
        }
    }
    void solve_gpu() {
        if (ngpu >= 8 && ngpu_p >= 4) {
            const double det = C[0][0] * C[1][1] - C[0][1] * C[1][0];
            if (std::abs(det) > 1e-12) {
                g0 = std::max((D[0] * C[1][1] - C[0][1] * D[1]) / det, 0.0);
                g = std::max((C[0][0] * D[1] - C[1][0] * D[0]) / det, 0.0);
            }
        }
    }
    bool ready() const { return on && ncpu >= 8 && ngpu_p >= 4; }
    /// The PCIe count for a layer of `nmiss` misses (`fixed`: pcie_num's).  Every 64th layer with a miss that would
    /// keep them all on one side moves one to the other, so both fits keep seeing both.
    int pick(int nmiss, int fixed) {
        int m = fixed;
        if (ready()) {
            double best = 1e300;
            for (int k = 0; k <= nmiss; ++k) {
                const double cpu = nmiss - k > 0 ? a + c * (nmiss - k) + d * k : p0;
                const double t = std::max(g0 + g * k, cpu);
                if (t < best - 1e-6) { best = t; m = k; }
            }
        }
        if (on && nmiss >= 1 && (++explore & 63u) == 0) m = m == 0 ? 1 : m == nmiss ? nmiss - 1 : m;
        return m;
    }
};
}
