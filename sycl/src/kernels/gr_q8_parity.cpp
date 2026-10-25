// sycl/src/kernels/gr_q8_parity.cpp - STRATA_HC_Q8=1 on SYCL: fused_gr_read_multi's Q8_0 read (q8_down / q8_up
// set) against a CPU FP64 reference on the very Q8_0 values it reads, and against the port's BF16 read.  Real
// geometry (n_embd 2560, hc 4, hc_lr 320), T = 1..8, with and without the pending write and the inject rows (a layer
// read / the final mixer), direct and as a captured graph replayed.  GPU, synthetic, no model.
//
// What it checks:
//   1. the Q8_0 read is observable: its `mixed` differs from the BF16 read's by about the Q8_0 rounding of the
//      weights, and not more (the port used to ignore the Q8_0 copies and read BF16 whatever STRATA_HC_Q8 said);
//   2. lo, rs, inject and mixed match an FP64 reference computed from the Q8_0 bytes (d * q) within FP32 summation
//      rounding; R_out (the pending write, the same fma as the BF16 read) is bitwise the BF16 read's;
//   3. a captured graph replayed twice gives bitwise the direct call's outputs (the arrival counters reset);
//   4. arguments not at the callers' fixed strides fall back to the BF16 read, bitwise.
#define DPCT_PROFILING_ENABLED
#include <sycl/sycl.hpp>
#include <dpct/dpct.hpp>
#include "strata/kernels/fused_gr.hpp"
#include "strata/kernels/fused_gr_q8.hpp"

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
float half_to_f(uint16_t h) { return (float) sycl::bit_cast<sycl::half>(h); }

/// ggml's Q8_0 blocks of `n` bf16 values (d = amax / 127 as fp16, q = round(x / d)): the loader's rule
std::vector<uint8_t> q8_0_of(const std::vector<uint16_t>& w) {
    std::vector<uint8_t> out(w.size() / 32 * 34);
    for (size_t b = 0; b < w.size() / 32; ++b) {
        float amax = 0.0f;
        for (int j = 0; j < 32; ++j) amax = std::max(amax, std::fabs(bf16_to_f(w[b * 32 + j])));
        const float d = amax / 127.0f, id = d != 0.0f ? 1.0f / d : 0.0f;
        const uint16_t hb = sycl::bit_cast<uint16_t>(sycl::half(d));
        uint8_t* o = out.data() + b * 34;
        std::memcpy(o, &hb, 2);
        for (int j = 0; j < 32; ++j) o[2 + j] = (uint8_t) (int8_t) std::lround(bf16_to_f(w[b * 32 + j]) * id);
    }
    return out;
}
/// the values a Q8_0 block holds (the fp16 scale as the kernel reads it, times the quant)
std::vector<float> dequant(const std::vector<uint8_t>& blocks) {
    std::vector<float> w(blocks.size() / 34 * 32);
    for (size_t b = 0; b < blocks.size() / 34; ++b) {
        uint16_t hb;
        std::memcpy(&hb, blocks.data() + b * 34, 2);
        const float d = half_to_f(hb);
        for (int j = 0; j < 32; ++j) w[b * 32 + j] = d * (float) (int8_t) blocks[b * 34 + 2 + j];
    }
    return w;
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
/// RMS (x - y) over RMS y
double rel_rms(const std::vector<float>& x, const std::vector<float>& y) {
    double d = 0.0, m = 1e-60;
    for (size_t i = 0; i < x.size(); ++i) { d += (double) (x[i] - y[i]) * (x[i] - y[i]); m += (double) y[i] * y[i]; }
    return std::sqrt(d / m);
}
bool same(const std::vector<float>& x, const std::vector<float>& y) {
    return x.size() == y.size() && std::memcmp(x.data(), y.data(), x.size() * sizeof(float)) == 0;
}
}  // namespace

int main(int argc, char** argv) {
    for (int i = 1; i < argc; ++i)
        if (std::string(argv[i]) != "--selftest") { std::fprintf(stderr, "usage: gr_q8_parity [--selftest]\n"); return 2; }
    const float eps = 1e-6f;
    std::mt19937 rng(20261016u);
    std::normal_distribution<float> gauss(0.0f, 1.0f);

    // the weights: BF16 (as the pack holds them) and their Q8_0 blocks in the read's split layout
    std::vector<float> w_norm((size_t) D);
    for (auto& x : w_norm) x = 1.0f + 0.1f * gauss(rng);
    std::vector<uint16_t> w_down((size_t) LR * D), w_up((size_t) D * LR), w_inj((size_t) HC * D);
    for (auto& x : w_down) x = bf16_of(gauss(rng) * 0.05f);
    for (auto& x : w_up) x = bf16_of(gauss(rng) * 0.05f);
    for (auto& x : w_inj) x = bf16_of(gauss(rng) * 0.02f);
    const std::vector<uint8_t> q_down = q8_0_of(w_down), q_up = q8_0_of(w_up);
    std::vector<uint8_t> s_down(q_down.size()), s_up(q_up.size());
    k::fused_gr_q8_0_split_host(q_down.data(), s_down.data(), LR, D);
    k::fused_gr_q8_0_split_host(q_up.data(), s_up.data(), D, LR);
    // the reference's weights: the Q8_0 values for the Q8_0 read, the BF16 ones for the inject rows
    const std::vector<float> wd_q = dequant(q_down), wu_q = dequant(q_up);
    std::vector<float> wi_f(w_inj.size());
    for (size_t i = 0; i < w_inj.size(); ++i) wi_f[i] = bf16_to_f(w_inj[i]);

    float* d_norm = up(w_norm);
    uint16_t *d_down = up(w_down), *d_up = up(w_up), *d_inj = up(w_inj);
    uint8_t *d_qdown = up(s_down), *d_qup = up(s_up);

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
    auto args_for = [&](int T, bool apply, bool inject, bool q8) {
        std::vector<k::FusedGrArgs> a(T);
        for (int t = 0; t < T; ++t) {
            k::FusedGrArgs& x = a[t];
            x.R = d_R + (size_t) t * D; x.R_out = d_R + (size_t) t * D; x.apply = apply;
            x.bo_prev = d_bo + (size_t) t * N; x.inj_prev = d_ip + (size_t) t * HC;
            x.w_norm = d_norm; x.w_down = d_down; x.w_up = d_up; x.w_inject = inject ? d_inj : nullptr;
            x.q8_down = q8 ? d_qdown : nullptr; x.q8_up = q8 ? d_qup : nullptr;
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

    int bad = 0, cases = 0;
    const double tol = 1e-4;   // FP32 summation order against FP64: a structural error is orders of magnitude larger
    double worst_lo = 0, worst_rs = 0, worst_inj = 0, worst_mixed = 0, min_obs = 1e9, max_obs = 0;
    for (int T = 1; T <= TM; ++T)
        for (int apply = 0; apply < 2; ++apply)
            for (int inject = 0; inject < 2; ++inject) {
                ++cases;
                const auto want = reference(T, apply, inject, R, bo, ip, w_norm, wd_q, wu_q, wi_f, eps);
                // the BF16 read (the default) on the same rows
                clear(); reset();
                auto ab = args_for(T, apply, inject, false);
                k::fused_gr_read_multi(ab.data(), T, d_xn, stream);
                const Out bf16 = snapshot(T, inject);
                // the Q8_0 read
                clear(); reset();
                auto aq = args_for(T, apply, inject, true);
                k::fused_gr_read_multi(aq.data(), T, d_xn, stream);
                const Out q8 = snapshot(T, inject);
                const double rl = rel_worst(q8.lo, want.lo), rr = rel_worst(q8.rs, want.rs),
                             ri = inject ? rel_worst(q8.inject, want.inject) : 0.0, rm = rel_worst(q8.mixed, want.mixed);
                worst_lo = std::max(worst_lo, rl); worst_rs = std::max(worst_rs, rr);
                worst_inj = std::max(worst_inj, ri); worst_mixed = std::max(worst_mixed, rm);
                bool ok = rl <= tol && rr <= tol && ri <= tol && rm <= tol;
                if (!ok) std::printf("  T=%d apply=%d inject=%d: Q8_0 read vs FP64 reference: lo %.2e rs %.2e inject %.2e mixed %.2e\n",
                                     T, apply, inject, rl, rr, ri, rm);
                if (!same(q8.r_out, bf16.r_out)) {
                    std::printf("  T=%d apply=%d inject=%d: R_out of the Q8_0 read differs from the BF16 read's\n", T, apply, inject);
                    ok = false;
                }
                // observable: the Q8_0 weights, not the BF16 ones, were read
                const double obs = rel_rms(q8.mixed, bf16.mixed);
                min_obs = std::min(min_obs, obs); max_obs = std::max(max_obs, obs);
                if (obs < 1e-5 || obs > 1e-1) {
                    std::printf("  T=%d apply=%d inject=%d: Q8_0 read vs BF16 read: mixed rel RMS %.2e (expected the Q8_0 rounding, ~1e-3)\n",
                                T, apply, inject, obs);
                    ok = false;
                }
                // a captured graph, replayed twice: bitwise the direct call
                dpct::experimental::command_graph_ptr g = nullptr;
                dpct::experimental::command_graph_exec_ptr gx = nullptr;
                dpct::experimental::begin_recording(stream);
                k::fused_gr_read_multi(aq.data(), T, d_xn, stream);
                dpct::experimental::end_recording(stream, &g);
                gx = new sycl::ext::oneapi::experimental::command_graph<sycl::ext::oneapi::experimental::graph_state::executable>(g->finalize());
                for (int rep = 0; rep < 2; ++rep) {
                    clear(); reset();
                    stream->ext_oneapi_graph(*gx);
                    const Out rp = snapshot(T, inject);
                    if (!same(rp.r_out, q8.r_out) || !same(rp.lo, q8.lo) || !same(rp.rs, q8.rs) || !same(rp.inject, q8.inject) ||
                        !same(rp.mixed, q8.mixed)) {
                        std::printf("  T=%d apply=%d inject=%d: graph replay %d differs from the direct Q8_0 read\n", T, apply, inject, rep);
                        ok = false;
                    }
                }
                delete gx;
                delete g;
                if (!ok) ++bad;
            }
    std::printf("  Q8_0 read vs FP64 reference on its Q8_0 values, %d cases (T 1..8 x pending write x inject): worst rel lo %.2e rs %.2e "
                "inject %.2e mixed %.2e (tolerance %.0e); R_out bitwise the BF16 read's; graph replays bitwise\n",
                cases, worst_lo, worst_rs, worst_inj, worst_mixed, tol);
    std::printf("  Q8_0 read vs BF16 read, mixed rel RMS %.2e..%.2e: the Q8_0 weights are read (not bitwise, by their rounding)\n",
                min_obs, max_obs);

    // rows not at the fixed strides (token 1's mixed before token 0's): the BF16 read, bitwise
    {
        const int T = 2;
        clear(); reset();
        auto ab = args_for(T, true, true, false);
        std::swap(ab[0].mixed, ab[1].mixed);
        k::fused_gr_read_multi(ab.data(), T, d_xn, stream);
        const Out bf16 = snapshot(T, true);
        clear(); reset();
        auto aq = args_for(T, true, true, true);
        std::swap(aq[0].mixed, aq[1].mixed);
        k::fused_gr_read_multi(aq.data(), T, d_xn, stream);
        const Out q8 = snapshot(T, true);
        const bool ok = same(q8.r_out, bf16.r_out) && same(q8.lo, bf16.lo) && same(q8.rs, bf16.rs) &&
                        same(q8.inject, bf16.inject) && same(q8.mixed, bf16.mixed);
        std::printf("  rows not at the fixed strides: the BF16 read, bitwise: %s\n", ok ? "pass" : "FAIL");
        if (!ok) ++bad;
    }

    std::printf("gr_q8_parity: %s (%d failing of %d cases)\n", bad ? "FAIL" : "pass", bad, cases + 1);
    dpct::get_current_device().destroy_queue(stream);
    for (void* p : {(void*) d_norm, (void*) d_down, (void*) d_up, (void*) d_inj, (void*) d_qdown, (void*) d_qup, (void*) d_R,
                    (void*) d_bo, (void*) d_ip, (void*) d_lo, (void*) d_rs, (void*) d_io, (void*) d_mixed, (void*) d_xn})
        sycl::free(p, dpct::get_in_order_queue());
    return bad ? 1 : 0;
}
