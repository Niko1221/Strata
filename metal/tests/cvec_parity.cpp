// metal/tests/cvec_parity.cpp - the control vector kernel (src/engine/msl/cvec.metal) against a host reference, as
// src/kernels/cvec_parity.cpp checks the CUDA one:
//   1. project: h - s (h.v) v per stream and row, against double precision; s = 2 reflects the component, s = 1
//      removes it.
//   2. the pending FFN write: BITWISE gr_norm's folded write (gr.metal with apply), which is what the window's next
//      read would have applied; then the write and the projection together.
//   3. add: h + d (bitwise).
// For 1, 3 and 8 rows (a decode token, a verify window, a full window); the prompt chunks use the same kernel.
#include "mtl_rt.h"

#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <random>
#include <string>
#include <vector>

namespace {

constexpr int N = 2560, HC = 4, D = HC * N, MAXT = 8;
struct CvecArgs { uint32_t n_embd, hc, mode, write; float s; };
struct GrArgs { uint32_t n_embd, hc, hc_lr, n_tok, apply; float eps; };

int g_fail = 0;
void check(bool ok, const std::string& what) {
    std::printf("  %-66s %s\n", what.c_str(), ok ? "ok" : "FAIL");
    if (!ok) ++g_fail;
}

std::string engine_msl() {
    if (const char* e = std::getenv("STRATA_METAL_ENGINE_KERNELS")) return e;
    std::string f = __FILE__;   // metal/tests/cvec_parity.cpp -> metal/src/engine/msl
    return f.substr(0, f.rfind('/')) + "/../src/engine/msl";
}

}  // namespace

int main() {
    mtl::Context cx;
    std::printf("device %s\n", cx.device_name().c_str());
    const int lcv = cx.compile({engine_msl() + "/cvec.metal"}, "#define NO_GGML_COMMON 1\n");
    const int lgr = cx.compile({"quant.metal", "units.metal", "gr.metal"});
    mtl::Pipeline pcv = cx.pipeline(lcv, "cvec_apply"), pgr = cx.pipeline(lgr, "gr_norm");

    std::mt19937 rng(1234);
    std::normal_distribution<float> nd(0.0f, 1.0f);
    std::vector<float> v((size_t) N);
    {
        double nrm = 0.0;
        for (auto& x : v) { x = nd(rng); nrm += (double) x * x; }
        nrm = std::sqrt(nrm);
        for (auto& x : v) x = (float) (x / nrm);
    }
    std::vector<float> R0((size_t) MAXT * D), bo0((size_t) MAXT * N), inj0((size_t) MAXT * HC), wn((size_t) D);
    for (auto& x : R0) x = nd(rng) * 3.0f;
    for (auto& x : bo0) x = nd(rng);
    for (auto& x : inj0) x = nd(rng);
    for (auto& x : wn) x = 1.0f + 0.1f * nd(rng);

    mtl::Buffer R = cx.alloc(4 * R0.size()), F = cx.alloc(4 * R0.size()), bo = cx.alloc(4 * bo0.size()),
                inj = cx.alloc(4 * inj0.size()), vb = cx.alloc(4 * (size_t) N), zb = cx.alloc(4 * (size_t) N),
                wnb = cx.alloc(4 * wn.size()), xn = cx.alloc(4 * R0.size());
    std::memcpy(bo.ptr, bo0.data(), 4 * bo0.size());
    std::memcpy(inj.ptr, inj0.data(), 4 * inj0.size());
    std::memcpy(vb.ptr, v.data(), 4 * v.size());
    std::memcpy(wnb.ptr, wn.data(), 4 * wn.size());

    auto cvec = [&](const mtl::Buffer& dir, int T, uint32_t mode, bool write, float s) {
        CvecArgs a{N, HC, mode, write ? 1u : 0u, s};
        mtl::Dispatch d;
        d.p = &pcv;
        d.args = {mtl::Arg::B(R), mtl::Arg::B(bo), mtl::Arg::B(inj), mtl::Arg::B(dir), mtl::Arg::V(a)};
        d.grid[0] = HC;
        d.grid[1] = (uint32_t) T;
        d.tg[0] = 256;
        mtl::require_fit(d, "cvec_apply");
        cx.run({d});
    };
    auto load = [&](mtl::Buffer& b) { std::memcpy(b.ptr, R0.data(), 4 * R0.size()); };
    // h - s (h.v) v over every stream and row of `src`, against what R holds
    auto project_err = [&](const float* src, int T, double s, double* worst_dot) {
        double worst = 0.0;
        *worst_dot = 0.0;
        for (int t = 0; t < T; ++t)
            for (int c = 0; c < HC; ++c) {
                const float* h = src + (size_t) t * D + (size_t) c * N;
                const float* g = R.as<float>() + (size_t) t * D + (size_t) c * N;
                double dot = 0.0, after = 0.0;
                for (int j = 0; j < N; ++j) dot += (double) h[j] * v[(size_t) j];
                for (int j = 0; j < N; ++j) {
                    worst = std::fmax(worst, std::fabs((double) g[j] - ((double) h[j] - s * dot * v[(size_t) j])));
                    after += (double) g[j] * v[(size_t) j];
                }
                *worst_dot = std::fmax(*worst_dot, std::fabs(after - (1.0 - s) * dot));
            }
        return worst;
    };

    for (int T : {1, 3, 8}) {
        std::printf("rows %d\n", T);
        // ---- 1. project, no pending write
        double wd = 0.0;
        load(R);
        cvec(vb, T, 0, false, 2.0f);
        double we = project_err(R0.data(), T, 2.0, &wd);
        std::printf("    s = 2: max |err| %.3g, max |h'.v + h.v| %.3g\n", we, wd);
        check(we < 1e-4, "project matches h - s (h.v) v (s = 2)");
        check(wd < 1e-3, "the component along v is reflected (s = 2)");
        load(R);
        cvec(vb, T, 0, false, 1.0f);
        we = project_err(R0.data(), T, 1.0, &wd);
        std::printf("    s = 1: max |err| %.3g, max |h'.v| %.3g\n", we, wd);
        check(we < 1e-4 && wd < 1e-3, "s = 1 projects the direction out");

        // ---- 2. the pending write: gr_norm's fold (apply) on F, the kernel's write alone (add a zero vector) on R
        load(F);
        {
            GrArgs a{N, HC, 320, (uint32_t) T, 1u, 1e-6f};
            mtl::Dispatch d;
            d.p = &pgr;
            d.args = {mtl::Arg::B(F), mtl::Arg::B(bo), mtl::Arg::B(inj), mtl::Arg::B(wnb), mtl::Arg::B(xn), mtl::Arg::V(a)};
            d.grid[0] = HC;
            d.grid[1] = (uint32_t) T;
            d.tg[0] = 256;
            cx.run({d});
        }
        load(R);
        cvec(zb, T, 1, true, 1.0f);
        check(std::memcmp(R.ptr, F.ptr, 4 * (size_t) T * D) == 0, "the pending write is bitwise gr_norm's folded write");
        std::vector<float> folded(F.as<float>(), F.as<float>() + (size_t) T * D);
        load(R);
        cvec(vb, T, 0, true, 1.0f);
        we = project_err(folded.data(), T, 1.0, &wd);
        std::printf("    write + project: max |err| %.3g, max |h'.v| %.3g\n", we, wd);
        check(we < 1e-4 && wd < 1e-3, "pending write, then the projection");
        check(std::memcmp(R.ptr, F.ptr, 4 * (size_t) T * D) != 0, "the projection changes the rows");

        // ---- 3. add: h + d in every stream
        std::vector<float> d01(v);
        for (auto& x : d01) x *= 0.1f;
        mtl::Buffer db = cx.alloc(4 * (size_t) N);
        std::memcpy(db.ptr, d01.data(), 4 * d01.size());
        load(R);
        cvec(db, T, 1, false, 1.0f);
        bool same = true;
        for (int t = 0; t < T; ++t)
            for (int c = 0; c < HC; ++c)
                for (int j = 0; j < N; ++j) {
                    const size_t i = (size_t) t * D + (size_t) c * N + (size_t) j;
                    same = same && R.as<float>()[i] == R0[i] + d01[(size_t) j];
                }
        check(same, "add is h + d in every stream (bitwise)");
        // rows past T are untouched
        bool rest = true;
        for (size_t i = (size_t) T * D; i < R0.size(); ++i) rest = rest && R.as<float>()[i] == R0[i];
        check(rest, "rows past the window are untouched");
        cx.free(db);
    }

    std::printf(g_fail ? "cvec_parity: %d FAILED\n" : "cvec_parity: all passed\n", g_fail);
    return g_fail ? 1 : 0;
}
