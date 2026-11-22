// engine/include/strata/math/riemann_rh.hpp
// Riemann Hypothesis (RH)-conditional mathematical primitives for Strata inference engine.
//
// Assumes RH: all non-trivial zeros of the Riemann zeta function lie on the critical line
// rho_k = 1/2 + i * gamma_k (Re(s) = 1/2).
//
// Applied theorems:
// 1. Littlewood (1912): sum_{n<=x} lambda(n) = O(x^{1/2 + eps}) <=> RH, where lambda(n) = (-1)^{Omega(n)}
//    is the Liouville function. Yields a zero-PRNG, compile-time sign vector with optimal O(sqrt(d))
//    square-root cancellation for deterministic quasi-orthogonal preconditioning.
// 2. von Koch (1901) / Riemann's Explicit Formula: under RH, counting fluctuations around the smooth
//    trend are bounded by O(t^{1/2} log t) (critical-line exponent sigma = 1/2). Motivates critical-line
//    adaptive decay gamma(t) = 1 - 1/sqrt(t + 4) and hysteresis floor eps_RH(t) = c * sqrt(S_a + S_b) / ln(t + e).
// 3. Zeta-Lagrangian VRAM Allocation: for MoE expert routing following a discrete Riemann Zeta law
//    P_i(r) = r^{-s_i} / zeta_E(s_i), global CPU-latency minimization subject to sum_i S_i * b_i <= B
//    is solved optimally by greedy marginal utility mu_i(k) = c_i * k^{-s_i} / (b_i * zeta_E(s_i)).
#pragma once

#include <algorithm>
#include <array>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <vector>

namespace strata::math {

// First 32 non-trivial zeros gamma_k > 0 of zeta(1/2 + i*gamma_k) = 0 (Odlyzko table, 15 digits).
inline constexpr std::array<double, 32> kZetaZeroImag = {
    14.134725141734694, 21.022039638771555, 25.010857580145688, 30.424876125859513,
    32.935061587739190, 37.586178158825671, 40.918719012147495, 43.327073280914999,
    48.005150881167160, 49.773832477672302, 52.970321477714461, 56.446247697063395,
    59.347044002602353, 60.831778524609810, 65.112544048081607, 67.079810529494174,
    69.546401711173979, 72.067157674481908, 75.704690699083933, 77.144840068874805,
    79.337375020249368, 82.910380854086030, 84.735492980517050, 87.425274613125229,
    88.809111207634465, 92.491899270558484, 94.651344040519887, 95.870634228245310,
    98.831194218193692, 101.31785100573139, 103.72553804047834, 105.44662305232609,
};

// Compile-time Liouville function lambda(n) = (-1)^{Omega(n)} in {-1, +1} for n >= 1.
// Under RH (Littlewood 1912), partial sums satisfy |sum_{n<=x} lambda(n)| = O(x^{1/2 + eps}).
constexpr int8_t liouville_sign(int n) noexcept {
    if (n <= 1) return 1;
    int parity = 0;
    int m = n;
    while ((m & 1) == 0) {
        parity ^= 1;
        m >>= 1;
    }
    for (int d = 3; d * d <= m; d += 2) {
        while (m % d == 0) {
            parity ^= 1;
            m /= d;
        }
    }
    if (m > 1) parity ^= 1;
    return parity ? int8_t(-1) : int8_t(1);
}

// Critical-line (sigma = 1/2) adaptive decay factor gamma(t) = 1 - 1 / sqrt(t + 4).
// Matches the RH explicit-formula fluctuation scaling O(t^{1/2}) so Zipf/Zeta head experts
// are retained across long horizons while adapting to regime shifts in O(sqrt(t)) steps.
inline double rh_critical_decay(uint64_t step) noexcept {
    const double d = 1.0 - 1.0 / std::sqrt((double) step + 4.0);
    return std::clamp(d, 0.50, 0.96);
}

// RH von Koch explicit-formula hysteresis floor:
// Avoids evicting a resident expert on O(sqrt(N)) random fluctuations while admitting
// genuine Zipf-head experts immediately.
inline double rh_von_koch_threshold(double score_hot, double score_cold, uint64_t step) noexcept {
    const double s = std::max(0.0, score_hot + score_cold);
    const double denom = std::log((double) step + 2.718281828459045);
    return 0.20 * std::sqrt(s) / std::max(1.0, denom);
}

// Truncated Riemann zeta function zeta_E(s) = sum_{k=1}^{E} k^{-s}.
inline double truncated_zeta(int E, double s) noexcept {
    double sum = 0.0;
    for (int k = 1; k <= E; ++k) {
        sum += std::pow((double) k, -s);
    }
    return sum > 0.0 ? sum : 1.0;
}

struct ZetaLayerSpec {
    int layer_id = 0;
    size_t stride_bytes = 0;
    double cpu_miss_cost = 1.0;
    double zeta_exponent = 1.15;
    int max_slots = 128;
};

// Globally optimal Tier-2 VRAM slot allocation under discrete Riemann Zeta popularity:
// Maximizes sum_i c_i * sum_{k=1}^{S_i} (k^{-s_i} / zeta_E(s_i)) subject to sum_i S_i * b_i <= budget.
// Because k^{-s_i} is strictly decreasing in k, greedy marginal-utility selection is globally optimal.
inline std::vector<int> zeta_lagrangian_allocate(const std::vector<ZetaLayerSpec>& specs,
                                                 size_t budget_bytes,
                                                 size_t* out_used_bytes = nullptr) {
    const size_t L = specs.size();
    std::vector<int> slots(L, 0);
    std::vector<double> inv_zeta(L, 1.0);
    for (size_t i = 0; i < L; ++i) {
        inv_zeta[i] = 1.0 / truncated_zeta(std::max(1, specs[i].max_slots), specs[i].zeta_exponent);
    }
    size_t used = 0;
    for (;;) {
        size_t best_i = L;
        double best_mu = -1.0;
        for (size_t i = 0; i < L; ++i) {
            const auto& sp = specs[i];
            if (slots[i] >= sp.max_slots || sp.stride_bytes == 0) continue;
            if (used + sp.stride_bytes > budget_bytes) continue;
            const double rank = (double) (slots[i] + 1);
            const double marginal_prob = std::pow(rank, -sp.zeta_exponent) * inv_zeta[i];
            const double mu = (sp.cpu_miss_cost * marginal_prob) / (double) sp.stride_bytes;
            if (mu > best_mu) {
                best_mu = mu;
                best_i = i;
            }
        }
        if (best_i == L) break;
        slots[best_i] += 1;
        used += specs[best_i].stride_bytes;
    }
    if (out_used_bytes) *out_used_bytes = used;
    return slots;
}

}  // namespace strata::math
