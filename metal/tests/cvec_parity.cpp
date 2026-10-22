// metal/tests/cvec_parity.cpp - the control vector (src/engine/msl/cvec.metal) against double, and its switch:
//
//   project / add, with and without the pending write, 1..8 rows: against a double reference of cvec.hpp's math
//   switched off (on = 0, or s_l = 0): the residual it writes is bitwise gr_norm's own apply (gr.metal), so the next
//   read (apply = 0) sees the values the fused read would have made
#include "mtl_rt.h"
#include "ref_quant.h"

#include <algorithm>
#include <cmath>
#include <cstdlib>
#include <cstdio>
#include <cstring>
#include <string>
#include <vector>

namespace {
struct CvecArgs { uint32_t n_embd, hc, layer, write, mode; };
struct GrArgs { uint32_t n_embd, hc, hc_lr, n_tok, apply; float eps; };
}  // namespace

int main() {
    constexpr int N = 2560, HC = 4, L = 48, MAXT = 8;
    mtl::Context cx;
    // the engine's kernels (STRATA_METAL_ENGINE_KERNELS, else beside this file's source: ../src/engine/msl)
    const char* dir = std::getenv("STRATA_METAL_ENGINE_KERNELS");
    std::string here = __FILE__;
    here = here.substr(0, here.rfind('/'));
    const int lcv = cx.compile({(dir ? std::string(dir) : here + "/../src/engine/msl") + "/cvec.metal"}, "#define NO_GGML_COMMON 1\n");
    const int lgr = cx.compile({"quant.metal", "units.metal", "gr.metal"});   // gr.metal with the formats it reads
    mtl::Pipeline pcv = cx.pipeline(lcv, "cvec_apply"), pgr = cx.pipeline(lgr, "gr_norm");
    mtl::Buffer R = cx.alloc(4 * (size_t) MAXT * HC * N), R2 = cx.alloc(4 * (size_t) MAXT * HC * N),
                bo = cx.alloc(4 * (size_t) MAXT * N), inj = cx.alloc(4 * MAXT * HC), dv = cx.alloc(4 * (size_t) L * N),
                sv = cx.alloc(4 * L), on = cx.alloc(16), wn = cx.alloc(4 * HC * N), xn = cx.alloc(4 * (size_t) MAXT * HC * N);
    ref::Rng rng(7);
    // unit directions (project) / raw ones (add) per layer, a scale per layer
    std::vector<float> R0((size_t) MAXT * HC * N);
    for (int i = 0; i < HC * N; ++i) wn.as<float>()[i] = 1.0f;
    int fails = 0;
    double worst = 0;
    for (int trial = 0; trial < 24; ++trial) {
        const int mode = trial % 2, write = (trial / 2) % 2, nt = 1 + trial % MAXT, layer = 4 + trial;
        for (int l = 0; l < L; ++l) {
            double nrm = 0;
            for (int d = 0; d < N; ++d) { const float x = rng.sym(); dv.as<float>()[l * N + d] = x; nrm += (double) x * x; }
            if (mode == 0)
                for (int d = 0; d < N; ++d) dv.as<float>()[l * N + d] = (float) (dv.as<float>()[l * N + d] / std::sqrt(nrm));
            sv.as<float>()[l] = mode == 0 ? 0.5f + 3.0f * std::fabs(rng.sym()) : 1.0f;
        }
        for (size_t i = 0; i < R0.size(); ++i) R0[i] = 4.0f * rng.sym();
        for (int i = 0; i < MAXT * N; ++i) bo.as<float>()[i] = rng.sym();
        for (int i = 0; i < MAXT * HC; ++i) inj.as<float>()[i] = 3.0f * rng.sym();
        std::memcpy(R.ptr, R0.data(), R0.size() * 4);
        on.as<int>()[0] = 1;
        CvecArgs a{N, HC, (uint32_t) layer, (uint32_t) write, (uint32_t) mode};
        mtl::Dispatch d;
        d.p = &pcv;
        d.args = {mtl::Arg::B(R), mtl::Arg::B(bo), mtl::Arg::B(inj), mtl::Arg::B(dv), mtl::Arg::B(sv), mtl::Arg::B(on), mtl::Arg::V(a)};
        d.grid[0] = HC;
        d.grid[1] = nt;
        d.tg[0] = 256;
        cx.run({d});
        // the reference: cvec.hpp's math in double
        const float* v = dv.as<float>() + (size_t) layer * N;
        const double s = sv.as<float>()[layer];
        for (int t = 0; t < nt; ++t)
            for (int c = 0; c < HC; ++c) {
                std::vector<double> h(N);
                const double g = write ? 2.0 / (1.0 + std::exp(-(double) inj.as<float>()[t * HC + c] / HC)) : 0.0;
                double dot = 0, mag = 0;
                for (int i = 0; i < N; ++i) {
                    h[i] = R0[(size_t) (t * HC + c) * N + i] + (write ? bo.as<float>()[t * N + i] * g : 0.0);
                    dot += h[i] * v[i];
                    mag = std::max(mag, std::fabs(h[i]));
                }
                for (int i = 0; i < N; ++i) {
                    const double want = mode == 0 ? h[i] - s * dot * v[i] : h[i] + v[i];
                    const double got = R.as<float>()[(size_t) (t * HC + c) * N + i];
                    const double err = std::fabs(got - want) / (mag + 1e-30);
                    worst = std::max(worst, err);
                    if (err > 1e-5) ++fails;
                }
            }
        // switched off: the write alone, bitwise gr_norm's apply
        if (write) {
            for (int off = 0; off < 2; ++off) {
                std::memcpy(R.ptr, R0.data(), R0.size() * 4);
                std::memcpy(R2.ptr, R0.data(), R0.size() * 4);
                on.as<int>()[0] = off == 0 ? 0 : 1;
                const float keep = sv.as<float>()[layer];
                if (off == 1) sv.as<float>()[layer] = 0.0f;   // a layer with no direction
                cx.run({d});
                GrArgs ga{N, HC, 320, (uint32_t) nt, 1u, 1e-6f};
                mtl::Dispatch g;
                g.p = &pgr;
                g.args = {mtl::Arg::B(R2), mtl::Arg::B(bo), mtl::Arg::B(inj), mtl::Arg::B(wn), mtl::Arg::B(xn), mtl::Arg::V(ga)};
                g.grid[0] = HC;
                g.grid[1] = nt;
                g.tg[0] = 256;
                cx.run({g});
                sv.as<float>()[layer] = keep;
                if (std::memcmp(R.ptr, R2.ptr, (size_t) nt * HC * N * 4) != 0) {
                    std::printf("trial %d: switched off (%s) is not gr_norm's write bitwise\n", trial, off ? "s_l = 0" : "on = 0");
                    ++fails;
                }
            }
        }
    }
    std::printf("control vector parity, 24 windows (project/add, with/without the write, 1..8 rows): worst rel err %.2e, "
                "off = gr_norm's write bitwise: %d failures  %s\n", worst, fails, fails ? "FAIL" : "PASS");
    return fails ? 1 : 0;
}
