#pragma once
#include "strata/hy4flash_gguf/pcie_model.hpp"


#include <algorithm>
#include <array>
#include <cmath>
#include <cstdint>
#include <cstring>

namespace strata::hy4_flash {

// A fit calibrated mostly at zero or one CPU expert cannot distinguish fixed
// overhead from marginal cost. Occasionally measure a wider batch on a layer
// where free GPU slots and the configured CPU cap actually permit it.
struct CpuSplitProbe {
    bool enabled = false;
    std::uint64_t trials = 0, overpredicted = 0;

    int choose(int planned, int capacity, const PcieModel& model) {
        pending = false;
        if (capacity < 2 || !model.ready()) return planned;
        if (planned >= 2) {
            remaining = interval = 32;
            return planned;
        }
        if (--remaining > 0) return planned;
        const int target = (trials % 2 == 0) ? 2 : 4;
        ++trials;
        pending = true;
        interval = std::min(256, 2 * interval);
        remaining = interval;
        return std::min(capacity, target);
    }

    void observe(const PcieModel& model, int cpu, int gpu, double actual_ms) {
        if (!pending) return;
        pending = false;
        const double expected_ms = model.a + model.c * cpu + model.d * gpu;
        if (expected_ms > 1.5 * actual_ms) {
            ++overpredicted;
            remaining = interval = 8;
        }
    }

private:
    int remaining = 32, interval = 32;
    bool pending = false;
};

// Used by the single-token async planner. GPU timing and the no-CPU baseline
// are observed immediately, even when the corresponding CPU sample is delayed.
struct CpuSplitFit {
    enum class Mode { Ordinary, SkipFirst, Huber, Bounded };
    Mode mode = Mode::Ordinary;
    std::uint64_t observed = 0, downweighted = 0;
    double first_ms = 0, first_weight = 0;

    bool configure_auto(const char* value) {
        if (!value || std::strcmp(value, "bounded") == 0) mode = Mode::Bounded;
        else if (std::strcmp(value, "ordinary") == 0) mode = Mode::Ordinary;
        else if (std::strcmp(value, "skip-first") == 0) mode = Mode::SkipFirst;
        else if (std::strcmp(value, "huber") == 0) mode = Mode::Huber;
        else return false;
        return true;
    }

    const char* name() const {
        switch (mode) {
            case Mode::Ordinary:  return "ordinary";
            case Mode::SkipFirst: return "skip-first";
            case Mode::Huber:     return "huber";
            case Mode::Bounded:   return "bounded";
        }
        return "unknown";
    }

    void solve(PcieModel& model) const {
        if (mode != Mode::Bounded) {
            model.solve();
            return;
        }
        model.solve_gpu();
        if (model.ncpu < 8) return;

        // Fit in coordinates (a, c-d, d) to enforce the cost bounds with the same ridge penalty.
        const double ridge = 1e-3 * model.A[0][0];
        const double cc = model.A[1][1] + ridge;
        const double cp = model.A[1][2];
        const double pp = model.A[2][2] + ridge;
        const double gram[3][3] = {
            {model.A[0][0], model.A[0][1], model.A[0][1] + model.A[0][2]},
            {model.A[0][1], cc, cc + cp},
            {model.A[0][1] + model.A[0][2], cc + cp, cc + 2*cp + pp}
        };
        const double rhs[] = {model.B[0], model.B[1], model.B[1] + model.B[2]};
        // With three coefficients there are only seven nonempty active sets.
        // Each reduced Gram matrix is positive definite thanks to the ridge.
        double best_gain = 0;
        double best[3] = {};
        for (unsigned mask = 1; mask < 8; ++mask) {
            int active[3], n = 0;
            for (int j = 0; j < 3; ++j)
                if (mask & (1u << j)) active[n++] = j;
            double matrix[3][4] = {};
            for (int j = 0; j < n; ++j) {
                for (int k = 0; k < n; ++k) matrix[j][k] = gram[active[j]][active[k]];
                matrix[j][n] = rhs[active[j]];
            }
            for (int j = 0; j < n; ++j) {
                for (int k = j + 1; k < n; ++k) {
                    const double factor = matrix[k][j] / matrix[j][j];
                    for (int col = j + 1; col <= n; ++col) matrix[k][col] -= factor * matrix[j][col];
                }
            }
            double x[3] = {}, gain = 0;
            bool feasible = true;
            for (int j = n - 1; j >= 0; --j) {
                double r = matrix[j][n];
                for (int k = j + 1; k < n; ++k) r -= matrix[j][k] * x[active[k]];
                x[active[j]] = r / matrix[j][j];
                feasible = feasible && x[active[j]] >= 0;
                gain += rhs[active[j]] * x[active[j]];
            }
            // At a face optimum x'Gram x == x'rhs, so gain is the loss reduction.
            if (feasible && gain > best_gain) {
                best_gain = gain;
                std::copy_n(x, 3, best);
            }
        }
        model.a = best[0];
        model.c = best[1] + best[2];
        model.d = best[2];
    }

    void sample(PcieModel& model, int cpu, int gpu, double cpu_ms, double gpu_ms) {
        model.sample_gpu(gpu, gpu_ms);
        if (!cpu) {
            model.sample_cpu(0, gpu, cpu_ms);
            return;
        }
        if (++observed == 1) first_ms = cpu_ms;
        if (mode != Mode::Huber) {
            if (observed == 1 && mode == Mode::SkipFirst) return;
            if (observed == 1) first_weight = 1;
            model.sample_cpu(cpu, gpu, cpu_ms);
            return;
        }
        batch[count++] = {cpu, gpu, cpu_ms};
        if (count == size) {
            fit_batch(model);
            count = 0;
        }
    }

private:
    static constexpr int size = 16;
    struct Observation { int cpu, gpu; double ms; };
    std::array<Observation, size> batch{};
    int count = 0;

    static double median(std::array<double, size> values) {
        std::sort(values.begin(), values.end());
        return 0.5 * (values[size / 2 - 1] + values[size / 2]);
    }

    void fit_batch(PcieModel& model) {
        auto fit = model;
        if (!model.ncpu) {
            std::array<double, size> per_expert{};
            for (int i = 0; i < size; ++i) per_expert[i] = batch[i].ms / batch[i].cpu;
            fit.c = median(per_expert);
        }
        // Reweight this batch against its workload-dependent CPU prediction.
        // Restart from the pre-batch sufficient statistics on every iteration so
        // each observation ages the history once, regardless of iteration count.
        for (int pass = 0; pass < 4; ++pass) {
            std::array<double, size> residual{};
            for (int i = 0; i < size; ++i) {
                const auto& row = batch[i];
                residual[i] = std::abs(row.ms - (fit.a + fit.c * row.cpu + fit.d * row.gpu));
            }
            // Median absolute residual about zero estimates the noise scale.
            // The branch clock has microsecond resolution.
            const double cutoff = std::max(0.001, 1.345 * 1.4826 * median(residual));
            fit = model;
            for (int i = 0; i < size; ++i) {
                const auto& row = batch[i];
                const double weight = residual[i] <= cutoff ? 1.0 : cutoff / residual[i];
                fit.sample_cpu(row.cpu, row.gpu, row.ms, weight);
                if (pass == 3) {
                    downweighted += weight < 1.0;
                    if (observed == size && i == 0) first_weight = weight;
                }
            }
            fit.solve_cpu();
        }
        model = fit;
    }
};

}  // namespace strata::hy4_flash
