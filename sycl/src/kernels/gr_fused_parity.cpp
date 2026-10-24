// sycl/src/kernels/gr_fused_parity.cpp - STRATA_GR_FUSED=1 on SYCL: the two-kernel BF16 hyper-connection read against
// the default read and a CPU FP64 reference.  Real geometry (n_embd 2560, hc 4, hc_lr 320), T = 1..8, with and
// without the pending write and the inject rows (a layer read / the final mixer), direct and as a captured graph
// replayed.  GPU, synthetic, no model.
//
// What it checks:
//   1. lo, rs, inject and mixed of the two-kernel read match an FP64 reference on the same BF16 weights within FP32
//      summation rounding (as the default read does: both are checked, the same tolerance);
//   2. R_out (the pending write) is bitwise the default read's; lo / rs / inject / mixed are NOT required bitwise
//      (another summation order) - their largest difference from the default read is printed and bounded;
//   3. a captured graph replayed twice gives bitwise the direct call's outputs (the arrival counters reset);
//   4. arguments not at the callers' fixed strides fall back to the default read, bitwise.
#define DPCT_PROFILING_ENABLED
#include <sycl/sycl.hpp>
#include <dpct/dpct.hpp>
#include "strata/kernels/fused_gr.hpp"
#include "strata/kernels/fused_gr_fused.hpp"

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <random>
#include <string>
#include <vector>

namespace k = strata::kernels;

namespace {
constexpr int N = 2560, HC = 4, LR = 320, D = N * HC, TM = k::kFusedGrMaxT;

template <typename T>
T* up(const std::vector<T>& h) {
    T* p = sycl::malloc_device<T>(h.size() + 64, dpct::get_in_order_queue());
    dpct::get_in_order_queue().memcpy(p, h.data(), h.size() * sizeof(T)).wait();
    return p;
}
template <typename T>
T* dev(size_t n) {
    T* p = sycl::malloc_device<T>(n + 64, dpct::get_in_order_queue());
    dpct::get_in_order_queue().memset(p, 0, (n + 64) * sizeof(T)).wait();
    return p;
}
template <typename T>
std::vector<T> down(const T* d, size_t n) {
    std::vector<T> h(n);
    dpct::get_in_order_queue().memcpy(h.data(), d, n * sizeof(T)).wait();
    return h;
}

uint16_t bf16_of(float f) {
    uint32_t i;
    std::memcpy(&i, &f, 4);
    i = (i + ((i >> 16) & 1u) + 0x7FFFu) & 0xFFFF0000u;
    return (uint16_t) (i >> 16);
}
float bf16_to_f(uint16_t b) {
    const uint32_t u = (uint32_t) b << 16;
    float f;
    std::memcpy(&f, &u, 4);
    return f;
}

struct Out {
    std::vector<float> r_out, lo, rs, inject, mixed;
};

/// ref/gr.py::gr_read in FP64 for T rows: R' = R + bo * 2 sigmoid(inj_prev / hc) (apply), rs per stream,
/// lo = silu(w_down . xn / hc), inject = w_inject . xn, mixed = mean_c xn * sigmoid(w_up . lo)
Out reference(int T, bool apply, bool inject, const std::vector<float>& R, const std::vector<float>& bo,
              const std::vector<float>& ip, const std::vector<float>& wn, const std::vector<float>& wd,
              const std::vector<float>& wu, const std::vector<float>& wi, float eps) {
    Out o;
    o.r_out.resize((size_t) T * D); o.lo.resize((size_t) T * LR); o.rs.resize((size_t) T * HC);
    o.inject.resize((size_t) T * HC); o.mixed.resize((size_t) T * N);
    std::vector<double> rp(D), xn(D), lo(LR);
    for (int t = 0; t < T; ++t) {
        for (int c = 0; c < HC; ++c) {
            const double gw = apply ? 2.0 / (1.0 + std::exp(-(double) ip[t * HC + c] / HC)) : 0.0;
            double ss = 0.0;
            for (int d = 0; d < N; ++d) {
                const int i = c * N + d;
                double r = R[(size_t) t * D + i];
                if (apply) r += (double) bo[(size_t) t * N + d] * gw;
                rp[i] = r;
                ss += r * r;
            }
            const double rs = 1.0 / std::sqrt(ss / N + (double) eps);
            o.rs[t * HC + c] = (float) rs;
            for (int d = 0; d < N; ++d) xn[c * N + d] = rp[c * N + d] * wn[c * N + d] * rs;
        }
        for (int i = 0; i < D; ++i) o.r_out[(size_t) t * D + i] = (float) rp[i];
        for (int j = 0; j < LR; ++j) {
            double s = 0.0;
            for (int i = 0; i < D; ++i) s += (double) wd[(size_t) j * D + i] * xn[i];
            const double x = s / HC;
            lo[j] = x / (1.0 + std::exp(-x));
            o.lo[t * LR + j] = (float) lo[j];
        }
        for (int c = 0; c < HC; ++c) {
            double s = 0.0;
            if (inject)
                for (int i = 0; i < D; ++i) s += (double) wi[(size_t) c * D + i] * xn[i];
            o.inject[t * HC + c] = (float) s;
        }
        for (int d = 0; d < N; ++d) {
            double m = 0.0;
            for (int c = 0; c < HC; ++c) {
                const int i = c * N + d;
                double u = 0.0;
                for (int j = 0; j < LR; ++j) u += (double) wu[(size_t) i * LR + j] * lo[j];
                m += xn[i] / (1.0 + std::exp(-u));
            }
            o.mixed[(size_t) t * N + d] = (float) (m / HC);
        }
    }
    return o;
}

/// worst |x - y| over max |y|
double rel_worst(const std::vector<float>& x, const std::vector<float>& y) {
    double worst = 0.0, mag = 1e-30;
    for (size_t i = 0; i < x.size(); ++i) {
        worst = std::max(worst, (double) std::fabs(x[i] - y[i]));
        mag = std::max(mag, (double) std::fabs(y[i]));
    }
    return worst / mag;
}
bool same(const std::vector<float>& x, const std::vector<float>& y) {
    return x.size() == y.size() && std::memcmp(x.data(), y.data(), x.size() * sizeof(float)) == 0;
}
bool all_same(const Out& a, const Out& b) {
    return same(a.r_out, b.r_out) && same(a.lo, b.lo) && same(a.rs, b.rs) && same(a.inject, b.inject) && same(a.mixed, b.mixed);
}
}  // namespace

int main(int argc, char** argv) {
    for (int i = 1; i < argc; ++i)
        if (std::string(argv[i]) != "--selftest") { std::fprintf(stderr, "usage: gr_fused_parity [--selftest]\n"); return 2; }
    const float eps = 1e-6f;
    std::mt19937 rng(20261016u);
    std::normal_distribution<float> gauss(0.0f, 1.0f);

    // the weights: BF16, as the pack holds them
    std::vector<float> w_norm((size_t) D);
    for (auto& x : w_norm) x = 1.0f + 0.1f * gauss(rng);
    std::vector<uint16_t> w_down((size_t) LR * D), w_up((size_t) D * LR), w_inj((size_t) HC * D);
    for (auto& x : w_down) x = bf16_of(gauss(rng) * 0.05f);
    for (auto& x : w_up) x = bf16_of(gauss(rng) * 0.05f);
    for (auto& x : w_inj) x = bf16_of(gauss(rng) * 0.02f);
    std::vector<float> wd_f(w_down.size()), wu_f(w_up.size()), wi_f(w_inj.size());
    for (size_t i = 0; i < w_down.size(); ++i) wd_f[i] = bf16_to_f(w_down[i]);
    for (size_t i = 0; i < w_up.size(); ++i) wu_f[i] = bf16_to_f(w_up[i]);
    for (size_t i = 0; i < w_inj.size(); ++i) wi_f[i] = bf16_to_f(w_inj[i]);
    float* d_norm = up(w_norm);
    uint16_t *d_down = up(w_down), *d_up = up(w_up), *d_inj = up(w_inj);

    // the rows: R with stream scales 1, 4, 16, 64 (so a per-stream norm is a claim with content), bo, inj_prev
    std::vector<float> R((size_t) TM * D), bo((size_t) TM * N), ip((size_t) TM * HC);
    for (int t = 0; t < TM; ++t)
        for (int c = 0; c < HC; ++c)
            for (int d = 0; d < N; ++d) R[(size_t) t * D + c * N + d] = gauss(rng) * (float) (1 << (2 * c));
    for (auto& x : bo) x = gauss(rng) * 0.3f;
    for (auto& x : ip) x = gauss(rng) * 0.3f;
    float *d_R = dev<float>((size_t) TM * D), *d_bo = up(bo), *d_ip = up(ip);
    float *d_lo = dev<float>((size_t) TM * LR), *d_rs = dev<float>((size_t) TM * HC), *d_io = dev<float>((size_t) TM * HC);
    float *d_mixed = dev<float>((size_t) TM * N), *d_xn = dev<float>((size_t) TM * D);

    dpct::queue_ptr stream = dpct::get_current_device().create_queue(true);
    auto reset = [&]() { stream->memcpy(d_R, R.data(), R.size() * sizeof(float)); };
    auto args_for = [&](int T, bool apply, bool inject) {
        std::vector<k::FusedGrArgs> a(T);
        for (int t = 0; t < T; ++t) {
            k::FusedGrArgs& x = a[t];
            x.R = d_R + (size_t) t * D; x.R_out = d_R + (size_t) t * D; x.apply = apply;
            x.bo_prev = d_bo + (size_t) t * N; x.inj_prev = d_ip + (size_t) t * HC;
            x.w_norm = d_norm; x.w_down = d_down; x.w_up = d_up; x.w_inject = inject ? d_inj : nullptr;
            x.eps = eps; x.lo = d_lo + (size_t) t * LR; x.rs = d_rs + (size_t) t * HC;
            x.inject_out = inject ? d_io + (size_t) t * HC : nullptr; x.mixed = d_mixed + (size_t) t * N;
        }
        return a;
    };
    auto snapshot = [&](int T, bool inject) {
        stream->wait();
        Out o;
        o.r_out = down(d_R, (size_t) T * D); o.lo = down(d_lo, (size_t) T * LR); o.rs = down(d_rs, (size_t) T * HC);
        o.inject = inject ? down(d_io, (size_t) T * HC) : std::vector<float>((size_t) T * HC, 0.0f);
        o.mixed = down(d_mixed, (size_t) T * N);
        return o;
    };
    auto clear = [&]() {
        stream->memset(d_lo, 0, (size_t) TM * LR * sizeof(float));
        stream->memset(d_rs, 0, (size_t) TM * HC * sizeof(float));
        stream->memset(d_io, 0, (size_t) TM * HC * sizeof(float));
        stream->memset(d_mixed, 0, (size_t) TM * N * sizeof(float));
    };
    // a read of `a` with the two-kernel read on or off
    auto run = [&](std::vector<k::FusedGrArgs>& a, int T, bool inject, bool fused) {
        clear(); reset();
        k::fused_gr_set_fused(fused ? 1 : 0);
        k::fused_gr_read_multi(a.data(), T, d_xn, stream);
        k::fused_gr_set_fused(-1);
        return snapshot(T, inject);
    };

    int bad = 0, cases = 0;
    const double tol = 1e-4;   // FP32 summation order against FP64: a structural error is orders of magnitude larger
    double worst_ref[4] = {}, worst_def[4] = {}, worst_default_ref = 0.0;
    for (int T = 1; T <= TM; ++T)
        for (int apply = 0; apply < 2; ++apply)
            for (int inject = 0; inject < 2; ++inject) {
                ++cases;
                const auto want = reference(T, apply, inject, R, bo, ip, w_norm, wd_f, wu_f, wi_f, eps);
                auto a = args_for(T, apply, inject);
                const Out def = run(a, T, inject, false);
                const Out fused = run(a, T, inject, true);
                const double r[4] = {rel_worst(fused.lo, want.lo), rel_worst(fused.rs, want.rs),
                                     inject ? rel_worst(fused.inject, want.inject) : 0.0, rel_worst(fused.mixed, want.mixed)};
                const double d[4] = {rel_worst(fused.lo, def.lo), rel_worst(fused.rs, def.rs),
                                     inject ? rel_worst(fused.inject, def.inject) : 0.0, rel_worst(fused.mixed, def.mixed)};
                const double dr = std::max({rel_worst(def.lo, want.lo), rel_worst(def.rs, want.rs),
                                            inject ? rel_worst(def.inject, want.inject) : 0.0, rel_worst(def.mixed, want.mixed)});
                worst_default_ref = std::max(worst_default_ref, dr);
                bool ok = dr <= tol;
                for (int i = 0; i < 4; ++i) {
                    worst_ref[i] = std::max(worst_ref[i], r[i]); worst_def[i] = std::max(worst_def[i], d[i]);
                    ok = ok && r[i] <= tol && d[i] <= tol;
                }
                if (!ok) std::printf("  T=%d apply=%d inject=%d: vs FP64 lo %.2e rs %.2e inject %.2e mixed %.2e; vs default lo %.2e rs %.2e "
                                     "inject %.2e mixed %.2e; default vs FP64 %.2e\n", T, apply, inject, r[0], r[1], r[2], r[3],
                                     d[0], d[1], d[2], d[3], dr);
                if (!same(fused.r_out, def.r_out)) {
                    std::printf("  T=%d apply=%d inject=%d: R_out of the two-kernel read differs from the default read's\n", T, apply, inject);
                    ok = false;
                }
                // a captured graph, replayed twice: bitwise the direct call
                dpct::experimental::command_graph_ptr g = nullptr;
                dpct::experimental::command_graph_exec_ptr gx = nullptr;
                k::fused_gr_set_fused(1);
                dpct::experimental::begin_recording(stream);
                k::fused_gr_read_multi(a.data(), T, d_xn, stream);
                dpct::experimental::end_recording(stream, &g);
                k::fused_gr_set_fused(-1);
                gx = new sycl::ext::oneapi::experimental::command_graph<sycl::ext::oneapi::experimental::graph_state::executable>(g->finalize());
                for (int rep = 0; rep < 2; ++rep) {
                    clear(); reset();
                    stream->ext_oneapi_graph(*gx);
                    const Out rp = snapshot(T, inject);
                    if (!all_same(rp, fused)) {
                        std::printf("  T=%d apply=%d inject=%d: graph replay %d differs from the direct two-kernel read\n", T, apply, inject, rep);
                        ok = false;
                    }
                }
                delete gx;
                delete g;
                if (!ok) ++bad;
            }
    std::printf("  two-kernel read vs FP64 reference, %d cases (T 1..8 x pending write x inject): worst rel lo %.2e rs %.2e inject %.2e "
                "mixed %.2e (the default read: %.2e; tolerance %.0e)\n", cases, worst_ref[0], worst_ref[1], worst_ref[2], worst_ref[3],
                worst_default_ref, tol);
    std::printf("  two-kernel read vs default read: R_out bitwise; lo %.2e rs %.2e inject %.2e mixed %.2e (another summation order: not "
                "bitwise); graph replays bitwise\n", worst_def[0], worst_def[1], worst_def[2], worst_def[3]);

    // rows not at the fixed strides (token 1's mixed before token 0's): the default read, bitwise
    {
        const int T = 2;
        auto a = args_for(T, true, true);
        std::swap(a[0].mixed, a[1].mixed);
        const Out def = run(a, T, true, false);
        const Out fused = run(a, T, true, true);
        const bool ok = all_same(fused, def);
        std::printf("  rows not at the fixed strides: the default read, bitwise: %s\n", ok ? "pass" : "FAIL");
        if (!ok) ++bad;
    }

    std::printf("gr_fused_parity: %s (%d failing of %d cases)\n", bad ? "FAIL" : "pass", bad, cases + 1);
    dpct::get_current_device().destroy_queue(stream);
    for (void* p : {(void*) d_norm, (void*) d_down, (void*) d_up, (void*) d_inj, (void*) d_R, (void*) d_bo, (void*) d_ip,
                    (void*) d_lo, (void*) d_rs, (void*) d_io, (void*) d_mixed, (void*) d_xn})
        sycl::free(p, dpct::get_in_order_queue());
    return bad ? 1 : 0;
}
