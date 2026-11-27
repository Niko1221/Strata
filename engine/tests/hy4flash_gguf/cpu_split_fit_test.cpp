#include "cpu_split_fit.hpp"

#include <cmath>
#include <cstdio>

int main() {
    using strata::hy4_flash::CpuSplitFit;
    strata::hy4_flash::PcieModel model;
    CpuSplitFit fit;
    if (!fit.configure_auto("ordinary") || fit.mode != CpuSplitFit::Mode::Ordinary) return 10;
    if (fit.configure_auto("unknown")) return 11;
    if (!fit.configure_auto(nullptr) || fit.mode != CpuSplitFit::Mode::Bounded) return 12;
    for (int i = 0; i < 32; ++i) {
        const int cpu = 1 + i % 2;
        fit.sample(model, cpu, 0, cpu == 1 ? 0.1 : 0.5, 0.2);
    }
    // The unconstrained intercept is negative. With a == d == 0 the
    // constrained optimum is a one-variable ridge regression.
    fit.solve(model);
    const double slope = model.B[1] / (model.A[1][1] + 0.001 * model.A[0][0]);
    if (model.a != 0 || model.d != 0 || std::abs(model.c - slope) > 1e-10) return 1;

    strata::hy4_flash::CpuSplitProbe probe;
    model.on = true;
    for (int j = 0; j < 64; ++j)
        if (probe.choose(0, 4, model) != 0) return 2;  // GPU fit still cold
    for (int j = 0; j < 8; ++j) model.sample_gpu(2, 0.2);
    for (int j = 0; j < 64; ++j)
        if (probe.choose(0, 1, model) != 0) return 3;  // cache-first or CPU cap
    for (int j = 0; j < 31; ++j)
        if (probe.choose(0, 4, model) != 0) return 4;
    if (probe.choose(0, 4, model) != 2) return 5;
    probe.observe(model, 2, 2, 0.05);
    for (int j = 0; j < 7; ++j)
        if (probe.choose(0, 4, model) != 0) return 6;
    if (probe.choose(0, 3, model) != 3) return 7;  // second probe targets four
    for (int j = 0; j < 256; ++j)
        if (probe.choose(3, 4, model) != 3) return 8;
    if (probe.trials != 2 || probe.overpredicted != 1) return 9;
    std::puts("bounded CPU fit and adaptive probes passed");
}
