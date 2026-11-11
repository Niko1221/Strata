// metal/tests/q36_gdn_parity.cpp - the GDN kernels (gdn.metal gdn_step, gdn_prompt.metal gdn_post) at Qwen3.6-35B-A3B's
// shapes (h_k 16, h_v 32, state 128, channels 8192 = q 2048 | k 2048 | v 4096, z width 4096, conv 4 taps) with the
// opt-in function constant GDN_SILU_Z (index 12): the output gate silu(z) instead of Flash-Next's sigmoid(z),
//     y = rms(o) * gamma * silu(z),  o scaled by 1/sqrt(128)   (q36_engine.cu k_gdn_seq / k_gdn_norm, llama.cpp
//     build_norm_gated).
// Same structure as gdn_parity.cpp and gdn_prompt_parity.cpp, same tolerances (y / state rel L2 < 1e-5, history exact,
// the window properties bitwise).  Checks:
//   per window T = 1..8 (GDN_NTOK):
//     * y, the committed state and the conv history against a double-precision CPU reference with silu(z)
//     * a verify window (commit 0) leaves state and history untouched and gives bitwise the commit window's y
//     * T single-token steps (NTOK 1, commit 1, history ping-pong) == the T-token commit window, bitwise: every y,
//       the final state and history
//     * verify window T, then the fold of keep = 1..T tokens (NTOK keep, commit 1) == the sequential path after keep
//       steps, bitwise; the in-place partial commit (commit = 0x100 | keep) of the T-token window likewise
//     * plane output mode (prepared activations): plane 0 is y, planes 1..3 and the per-16 sums are bitwise the
//       arithmetic of y
//     * the default (constant unset, and set to false) is still the sigmoid gate: CPU reference with sigmoid(z); the
//       state and history are the gate's (z never touches them) and bitwise the silu run's; silu != sigmoid
//   gdn_post alone (the prompt path's last pass) against the CPU reference, silu and default
//   the prompt chain gdn_pre -> gdn_scanr_4 -> gdn_post (silu) for T = 1, 3, 37, 300 against the CPU reference and
//   against gdn_step (NTOK = T, silu)
//   the gdn_step pipelines (GDN_NTOK 1..8, silu on) must allow 512 threads
//
// The z gate does not touch the recurrence, so everything but y is bitwise the same with the constant on or off.
#include "mtl_rt.h"
#include "ref_quant.h"

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <map>
#include <string>
#include <vector>

namespace {
constexpr int S = 128, HK = 16, HV = 32, C = (2 * HK + HV) * S, ZW = HV * S, MAXT = 8, TMAX = 300;
constexpr int SILU_Z = 12;   // function constant GDN_SILU_Z
constexpr float EPS = 1e-6f;
static_assert(C == 8192 && ZW == 4096, "Qwen3.6 GDN shapes");
struct GdnArgs { uint32_t h_k, h_v, channels, commit; float eps; uint32_t plane; };
struct GdnPArgs { uint32_t n_tok, h_k, h_v, channels; float eps; uint32_t f32; };

struct Err {
    double se = 0, sr = 0, mx = 0, rmax = 0;
    void add(double g, double r) {
        se += (g - r) * (g - r);
        sr += r * r;
        mx = std::max(mx, std::fabs(g - r));
        rmax = std::max(rmax, std::fabs(r));
    }
    double l2() const { return sr > 0 ? std::sqrt(se / sr) : std::sqrt(se); }
    double linf() const { return rmax > 0 ? mx / rmax : mx; }   // max |err| / max |ref|
};

double silu_d(double z) { return z / (1.0 + std::exp(-z)); }
double sigm_d(double z) { return 1.0 / (1.0 + std::exp(-z)); }

struct Ref {
    std::vector<double> y, ysig, S, H;   // y: silu gate, ysig: sigmoid gate (the same o), [T][ZW]; state; history
};

// The double-precision contract of gdn.metal (conv + SiLU, q/k L2 norm, gate / beta, delta rule, rms(o) * gamma * gate(z)).
Ref cpu_ref(int T, const float* st0, const float* h0, const float* qkv, const float* al, const float* be, const float* z,
            const float* cw, const float* dt, const float* sa, const float* gm) {
    const size_t st_elems = (size_t)S * HV * S;
    Ref r;
    r.y.assign((size_t)T * ZW, 0.0);
    r.ysig.assign((size_t)T * ZW, 0.0);
    r.S.resize(st_elems);
    r.H.resize((size_t)C * 3);
    for (size_t i = 0; i < st_elems; ++i) r.S[i] = st0[i];
    for (int i = 0; i < C * 3; ++i) r.H[i] = h0[i];
    std::vector<double>& Sd = r.S;
    std::vector<double>& Hd = r.H;
    for (int t = 0; t < T; ++t) {
        std::vector<double> cv(C);
        for (int c = 0; c < C; ++c) {
            const double in = qkv[(size_t)t * C + c];
            const double s = Hd[c * 3] * cw[c * 4] + Hd[c * 3 + 1] * cw[c * 4 + 1] + Hd[c * 3 + 2] * cw[c * 4 + 2] +
                             in * cw[c * 4 + 3];
            cv[c] = silu_d(s);   // the conv's own SiLU (not the z gate)
            Hd[c * 3] = Hd[c * 3 + 1];
            Hd[c * 3 + 1] = Hd[c * 3 + 2];
            Hd[c * 3 + 2] = in;
        }
        for (int hh = 0; hh < 2 * HK; ++hh) {
            double ss = 0;
            for (int i = 0; i < S; ++i) ss += cv[hh * S + i] * cv[hh * S + i];
            const double rs = 1.0 / std::sqrt(ss / S + 1e-6 / S);
            for (int i = 0; i < S; ++i) cv[hh * S + i] = cv[hh * S + i] * rs / std::sqrt((double)S);
        }
        for (int h = 0; h < HV; ++h) {
            const int hk = h % HK;
            const double* q = &cv[hk * S];
            const double* k = &cv[HK * S + hk * S];
            const double* v = &cv[2 * HK * S + h * S];
            const double av = (double)al[(size_t)t * HV + h] + dt[h];
            const double sp = av > 20 ? av : std::log1p(std::exp(av));
            const double g = std::exp(sp * sa[h]);
            const double b = sigm_d((double)be[(size_t)t * HV + h]);
            std::vector<double> o(S);
            for (int jj = 0; jj < S; ++jj) {
                double kv = 0;
                for (int i = 0; i < S; ++i) kv += Sd[((size_t)i * HV + h) * S + jj] * k[i];
                const double delta = (v[jj] - g * kv) * b;
                double acc = 0;
                for (int i = 0; i < S; ++i) {
                    double& sv = Sd[((size_t)i * HV + h) * S + jj];
                    sv = g * sv + k[i] * delta;
                    acc += sv * q[i];
                }
                o[jj] = acc / std::sqrt((double)S);
            }
            double ss = 0;
            for (int jj = 0; jj < S; ++jj) ss += o[jj] * o[jj];
            const double rs = 1.0 / std::sqrt(ss / S + 1e-6);
            for (int jj = 0; jj < S; ++jj) {
                const double zz = z[(size_t)t * ZW + h * S + jj];
                const double base = o[jj] * rs * gm[jj];
                r.y[(size_t)t * ZW + h * S + jj] = base * silu_d(zz);
                r.ysig[(size_t)t * ZW + h * S + jj] = base * sigm_d(zz);
            }
        }
    }
    return r;
}

template <class T> bool same(const T* a, const T* b, size_t n) { return std::memcmp(a, b, n * sizeof(T)) == 0; }
const char* yn(bool b) { return b ? "yes" : "NO"; }
}  // namespace

int main() {
    mtl::Context cx;
    const int lib = cx.compile({"gdn.metal", "gdn_prompt.metal"}, "#define NO_GGML_COMMON 1\n");
    std::printf("device %s\n", cx.device_name().c_str());
    std::printf("Qwen3.6 GDN shapes: h_k %d, h_v %d, state %d, channels %d, z width %d; GDN_SILU_Z = function constant %d\n", HK, HV,
                S, C, ZW, SILU_Z);

    const size_t st_elems = (size_t)S * HV * S;
    auto F = [&](size_t n) { return cx.alloc(4 * n); };
    mtl::Buffer st0 = F(st_elems), state = F(st_elems), h0 = F((size_t)C * 3), hA = F((size_t)C * 3), hB = F((size_t)C * 3),
                qkv = F((size_t)TMAX * C), al = F((size_t)TMAX * HV), be = F((size_t)TMAX * HV), z = F((size_t)TMAX * ZW),
                cw = F((size_t)C * 4), dt = F(HV), sa = F(HV), gm = F(S), yv = F((size_t)MAXT * ZW), yc = F((size_t)MAXT * ZW),
                yq = F((size_t)MAXT * ZW), yf = F((size_t)MAXT * ZW), yd = F((size_t)MAXT * ZW), ydf = F((size_t)MAXT * ZW);
    const size_t PL = (size_t)ZW * MAXT;   // floats between the planes of the plane output mode
    mtl::Buffer yp = F(4 * PL + (size_t)MAXT * (ZW / 16));

    ref::Rng r(36);
    for (size_t i = 0; i < st_elems; ++i) st0.as<float>()[i] = r.sym() * 0.1f;
    for (int i = 0; i < C * 3; ++i) h0.as<float>()[i] = r.sym();
    for (size_t i = 0; i < (size_t)TMAX * C; ++i) qkv.as<float>()[i] = r.sym() * 2.0f;
    for (int i = 0; i < TMAX * HV; ++i) { al.as<float>()[i] = r.sym() * 2.0f; be.as<float>()[i] = r.sym() * 2.0f; }
    for (size_t i = 0; i < (size_t)TMAX * ZW; ++i) z.as<float>()[i] = r.sym() * 4.0f;   // both signs, past +-3: silu != sigmoid
    for (int i = 0; i < C * 4; ++i) cw.as<float>()[i] = r.sym() * 0.5f;
    for (int i = 0; i < HV; ++i) { dt.as<float>()[i] = r.sym(); sa.as<float>()[i] = -std::exp(r.sym()); }
    for (int i = 0; i < S; ++i) gm.as<float>()[i] = 1.0f + 0.2f * r.sym();

    // gdn_step pipelines, cached: mode 0 = constant unset (the default), 1 = set to false, 2 = silu (set to true)
    std::map<int, mtl::Pipeline> cache;
    auto step_pl = [&](int nt, int mode, const char* kn = "gdn_step") -> const mtl::Pipeline& {
        const int key = (nt * 3 + mode) * 2 + (kn[8] == '8');
        auto it = cache.find(key);
        if (it == cache.end()) {
            mtl::Constants c;
            c.i(2, nt);
            if (mode == 1) c.b(SILU_Z, false);
            if (mode == 2) c.b(SILU_Z, true);
            it = cache.emplace(key, cx.pipeline(lib, kn, c)).first;
        }
        return it->second;
    };
    // one dispatch of gdn_step over tokens tok0.. (inputs at their token offsets), y written at token out_tok0
    auto mk = [&](const mtl::Pipeline& p, int tok0, uint32_t commit, uint32_t plane, const mtl::Buffer& hin,
                  const mtl::Buffer& hout, const mtl::Buffer& out, int out_tok0) {
        GdnArgs a{HK, HV, C, commit, EPS, plane};
        mtl::Dispatch d;
        d.p = &p;
        d.args = {mtl::Arg::B(state), mtl::Arg::B(hin), mtl::Arg::B(qkv, 4 * (size_t)C * tok0), mtl::Arg::B(al, 4 * (size_t)HV * tok0),
                  mtl::Arg::B(be, 4 * (size_t)HV * tok0), mtl::Arg::B(z, 4 * (size_t)ZW * tok0), mtl::Arg::B(cw), mtl::Arg::B(dt),
                  mtl::Arg::B(sa), mtl::Arg::B(gm), mtl::Arg::B(out, 4 * (size_t)ZW * out_tok0), mtl::Arg::V(a), mtl::Arg::B(hout)};
        d.grid[0] = HV;
        d.tg[0] = 512;
        return d;
    };
    auto reset = [&]() {
        std::memcpy(state.ptr, st0.ptr, 4 * st_elems);
        std::memcpy(hA.ptr, h0.ptr, 4 * (size_t)C * 3);
        std::memset(hB.ptr, 0xFF, 4 * (size_t)C * 3);
    };
    auto snap = [](const mtl::Buffer& b, size_t n) { return std::vector<float>(b.as<float>(), b.as<float>() + n); };

    int fails = 0;

    // ---- the thread limit: every specialization the engine launches (GDN_NTOK 1..8), silu on ----------------------------
    {
        int low = 0, min_threads = 1 << 30;
        for (int nt = 1; nt <= 8; ++nt)
            for (int mode : {0, 2}) {
                const mtl::Pipeline& pn = step_pl(nt, mode);
                std::printf("gdn_step GDN_NTOK %d %-6s: maxTotalThreadsPerThreadgroup %d (the kernel needs 512)%s\n", nt,
                            mode == 2 ? "silu" : "sigmoid", pn.max_threads, pn.max_threads < 512 ? "  TOO FEW" : "");
                low += pn.max_threads < 512;
                min_threads = std::min(min_threads, pn.max_threads);
            }
        const mtl::Pipeline& p8 = step_pl(8, 2, "gdn_step8");
        std::printf("(gdn_step8, GDN_NTOK 8 silu: max %d threads, needs 1024: informational)\n", p8.max_threads);
        std::printf("thread limit: min %d over GDN_NTOK 1..8, silu and sigmoid: %s\n", min_threads, low ? "FAIL" : "ok (>= 512)");
        fails += low != 0;
    }

    // ---- gdn_step windows T = 1..8 ---------------------------------------------------------------------------------------
    for (int T = 1; T <= MAXT; ++T) {
        const mtl::Pipeline& pS = step_pl(T, 2);
        const size_t yn_ = (size_t)T * ZW;

        // verify window: state and history untouched; deterministic
        reset();
        cx.run({mk(pS, 0, 0, 0, hA, hB, yv, 0)});
        const bool untouched = same(state.as<float>(), st0.as<float>(), st_elems) && same(hA.as<float>(), h0.as<float>(), (size_t)C * 3) &&
                               [&] { for (int i = 0; i < C * 3; ++i) if (hB.as<uint32_t>()[i] != 0xFFFFFFFFu) return false; return true; }();
        const std::vector<float> yverify = snap(yv, yn_);
        bool deterministic = true;
        for (int k = 0; k < 2; ++k) {
            cx.run({mk(pS, 0, 0, 0, hA, hB, yq, 0)});
            deterministic = deterministic && same(yq.as<float>(), yverify.data(), yn_);
        }

        // commit window: the reference, and y == the verify window's
        reset();
        cx.run({mk(pS, 0, 1, 0, hA, hB, yc, 0)});
        const std::vector<float> sw = snap(state, st_elems), hw = snap(hB, (size_t)C * 3);
        const bool same_y = same(yc.as<float>(), yverify.data(), yn_);
        const Ref ref = cpu_ref(T, st0.as<float>(), h0.as<float>(), qkv.as<float>(), al.as<float>(), be.as<float>(), z.as<float>(),
                                cw.as<float>(), dt.as<float>(), sa.as<float>(), gm.as<float>());
        Err ey, es, eh;
        for (size_t i = 0; i < yn_; ++i) ey.add(yc.as<float>()[i], ref.y[i]);
        for (size_t i = 0; i < st_elems; ++i) es.add(sw[i], ref.S[i]);
        for (int i = 0; i < C * 3; ++i) eh.add(hw[i], ref.H[i]);

        // T single-token steps (NTOK 1, commit 1, the history ping-pongs hA <-> hB, the state is in place)
        std::vector<std::vector<float>> seqS(T + 1), seqH(T + 1);
        reset();
        {
            const mtl::Pipeline& p1 = step_pl(1, 2);
            const mtl::Buffer* in = &hA;
            const mtl::Buffer* out = &hB;
            for (int t = 0; t < T; ++t) {
                cx.run({mk(p1, t, 1, 0, *in, *out, yf, t)});
                seqS[t + 1] = snap(state, st_elems);
                seqH[t + 1] = snap(*out, (size_t)C * 3);
                std::swap(in, out);
            }
        }
        const bool seq_y = same(yf.as<float>(), yc.as<float>(), yn_), seq_s = seqS[T] == sw, seq_h = seqH[T] == hw;

        // verify window T, then the fold of keep tokens (NTOK keep, commit 1) from the untouched state; and the in-place
        // partial commit (0x100 | keep) of the T-token window
        bool fold_ok = true, partial_ok = true;
        double fold_rel = 0;
        for (int k = 1; k <= T; ++k) {
            reset();
            cx.run({mk(pS, 0, 0, 0, hA, hB, yq, 0)});   // the verify window
            cx.run({mk(step_pl(k, 2), 0, 1, 0, hA, hB, yf, 0)});   // the fold
            fold_ok = fold_ok && snap(state, st_elems) == seqS[k] && snap(hB, (size_t)C * 3) == seqH[k] &&
                      same(yf.as<float>(), yverify.data(), (size_t)k * ZW);
            Err e;
            for (size_t i = 0; i < st_elems; ++i) e.add(state.as<float>()[i], seqS[k][i]);
            fold_rel = std::max(fold_rel, e.l2());
            reset();
            cx.run({mk(pS, 0, 0x100u | (uint32_t)k, 0, hA, hB, yq, 0)});
            partial_ok = partial_ok && snap(state, st_elems) == seqS[k] && snap(hB, (size_t)C * 3) == seqH[k] &&
                         same(yq.as<float>(), yverify.data(), yn_);
        }

        // plane output mode: plane 0 = y, planes 1..3 = y x (2^-2 jj, 2^-8 jj, odd lanes 2^-8), the per-16 sums
        bool plane_ok = true;
        {
            reset();
            cx.run({mk(pS, 0, 0, (uint32_t)PL, hA, hB, yp, 0)});
            const float q2s[8] = {1.0f, 0.25f, 0.0625f, 0.015625f, 0.00390625f, 0.0009765625f, 0.000244140625f, 0.00006103515625f};
            const float p4s[4] = {1.0f, 0.00390625f, 1.52587890625e-05f, 5.9604644775390625e-08f};
            const float* P = yp.as<float>();
            for (int t = 0; t < T; ++t)
                for (int i = 0; i < ZW; ++i) {
                    const float y = yverify[(size_t)t * ZW + i];
                    const int jj = i & 15;   // (h * 128 + j) & 15 == j & 15
                    plane_ok = plane_ok && P[(size_t)t * ZW + i] == y && P[PL + (size_t)t * ZW + i] == y * q2s[jj & 7] &&
                               P[2 * PL + (size_t)t * ZW + i] == y * p4s[jj & 3] &&
                               P[3 * PL + (size_t)t * ZW + i] == ((jj & 1) ? y * 0.00390625f : y);
                    if (jj == 0) {
                        float s0 = 0.0f, s1 = 0.0f;
                        for (int k = 0; k < 16; k += 2) { s0 += yverify[(size_t)t * ZW + i + k]; s1 += yverify[(size_t)t * ZW + i + k + 1]; }
                        plane_ok = plane_ok && P[4 * PL + (size_t)t * (ZW / 16) + i / 16] == s0 + s1;
                    }
                }
        }

        // the default (constant unset, and false) is the sigmoid gate; the recurrence does not depend on the gate
        reset();
        cx.run({mk(step_pl(T, 0), 0, 1, 0, hA, hB, yd, 0)});
        const bool dflt_state = snap(state, st_elems) == sw && snap(hB, (size_t)C * 3) == hw;
        Err eyd, edif;
        for (size_t i = 0; i < yn_; ++i) { eyd.add(yd.as<float>()[i], ref.ysig[i]); edif.add(yc.as<float>()[i], yd.as<float>()[i]); }
        reset();
        cx.run({mk(step_pl(T, 1), 0, 1, 0, hA, hB, ydf, 0)});
        const bool false_is_default = same(ydf.as<float>(), yd.as<float>(), yn_) && snap(state, st_elems) == sw;
        const bool differs = std::sqrt(edif.se / std::max(edif.sr, 1e-30)) > 1e-2;   // silu vs sigmoid y: far apart

        const bool ok1 = untouched && deterministic && same_y && ey.l2() < 1e-5 && es.l2() < 1e-5 && eh.l2() == 0 && seq_y && seq_s && seq_h &&
                         fold_ok && partial_ok && plane_ok;
        const bool ok2 = eyd.l2() < 1e-5 && dflt_state && false_is_default && differs;
        const bool ok = ok1 && ok2;
        fails += !ok;
        std::printf("q36 gdn ntok %d (silu): y rel L2 %.2e (max %.2e of |y|max), state %.2e (max %.2e), history %.1e; verify leaves state: %s, "
                    "deterministic: %s, verify y == commit y: %s\n", T, ey.l2(), ey.linf(), es.l2(), es.linf(), eh.l2(), yn(untouched),
                    yn(deterministic), yn(same_y));
        std::printf("                  %d single-token steps == window (bitwise) y: %s, state: %s, history: %s; fold keep 1..%d (state/hist/y "
                    "bitwise): %s (max rel %.1e); partial commits 0x100|keep bitwise: %s; plane mode bitwise: %s\n", T, yn(seq_y), yn(seq_s),
                    yn(seq_h), T, yn(fold_ok), fold_rel, yn(partial_ok), yn(plane_ok));
        std::printf("                  default sigmoid: y rel L2 %.2e (max %.2e), state/hist == silu run's: %s, constant false == unset: %s, "
                    "silu vs sigmoid y rel diff %.2e: %s  %s\n", eyd.l2(), eyd.linf(), yn(dflt_state), yn(false_is_default),
                    std::sqrt(edif.se / std::max(edif.sr, 1e-30)), differs ? "differ" : "SAME", ok ? "PASS" : "FAIL");
    }

    // ---- gdn_post alone (the prompt path's last pass): o random, y = o * rsqrt(mean(o^2) + eps) * gamma * gate(z) -----------
    {
        mtl::Buffer o = F((size_t)TMAX * ZW), yP = F((size_t)TMAX * ZW), yP2 = F((size_t)TMAX * ZW), yh = cx.alloc(2 * (size_t)TMAX * ZW),
                    yh2 = cx.alloc(2 * (size_t)TMAX * ZW);
        for (size_t i = 0; i < (size_t)TMAX * ZW; ++i) o.as<float>()[i] = r.sym() * (1.0f + 3.0f * ((i / S) % 5 == 0));
        const mtl::Pipeline psilu = cx.pipeline(lib, "gdn_post", mtl::Constants().b(SILU_Z, true));
        const mtl::Pipeline pdef = cx.pipeline(lib, "gdn_post"), pfalse = cx.pipeline(lib, "gdn_post", mtl::Constants().b(SILU_Z, false));
        for (int T : {1, 5, 37}) {
            const size_t n = (size_t)T * ZW;
            auto run_post = [&](const mtl::Pipeline& p, mtl::Buffer& y32, mtl::Buffer& y16) {
                GdnPArgs a{(uint32_t)T, HK, HV, C, EPS, 1u};
                mtl::Dispatch d;
                d.p = &p;
                d.args = {mtl::Arg::B(o), mtl::Arg::B(z), mtl::Arg::B(gm), mtl::Arg::B(y16), mtl::Arg::B(y32), mtl::Arg::V(a)};
                d.grid[0] = HV;
                d.grid[1] = T;
                d.tg[0] = 128;
                cx.run({d});
            };
            run_post(psilu, yP, yh);
            run_post(pdef, yP2, yh2);
            std::vector<float> y_false(n);
            {
                mtl::Buffer yF = F(n), yhF = cx.alloc(2 * n);
                run_post(pfalse, yF, yhF);
                std::memcpy(y_false.data(), yF.ptr, 4 * n);
            }
            Err es_, ed_, eh16;
            for (int t = 0; t < T; ++t)
                for (int h = 0; h < HV; ++h) {
                    double ss = 0;
                    for (int j = 0; j < S; ++j) { const double v = o.as<float>()[(size_t)t * ZW + h * S + j]; ss += v * v; }
                    const double rs = 1.0 / std::sqrt(ss / 128.0 + 1e-6);
                    for (int j = 0; j < S; ++j) {
                        const size_t i = (size_t)t * ZW + h * S + j;
                        const double base = (double)o.as<float>()[i] * rs * gm.as<float>()[j], zz = z.as<float>()[i];
                        es_.add(yP.as<float>()[i], base * silu_d(zz));
                        ed_.add(yP2.as<float>()[i], base * sigm_d(zz));
                        eh16.add(ref::f16_to_f32(yh.as<uint16_t>()[i]), yP.as<float>()[i]);
                    }
                }
            const bool deflt_same = same(y_false.data(), yP2.as<float>(), n);
            const bool ok = es_.l2() < 1e-5 && ed_.l2() < 1e-5 && eh16.l2() < 1e-3 && deflt_same;
            fails += !ok;
            std::printf("q36 gdn_post T %3d: silu y rel L2 %.2e (max %.2e), fp16 vs fp32 %.2e; default sigmoid y rel L2 %.2e (max %.2e); constant "
                        "false == unset: %s  %s\n", T, es_.l2(), es_.linf(), eh16.l2(), ed_.l2(), ed_.linf(), yn(deflt_same), ok ? "PASS" : "FAIL");
        }
    }

    // ---- the prompt chain gdn_pre -> gdn_scanr_4 -> gdn_post (silu) against the CPU reference and against gdn_step ----------
    {
        mtl::Buffer qn = F((size_t)TMAX * HK * S), kn = F((size_t)TMAX * HK * S), vc = F((size_t)TMAX * ZW), gb = F(2 * (size_t)TMAX * HV),
                    kq = F((size_t)TMAX * HK), o = F((size_t)TMAX * ZW), yP = F((size_t)TMAX * ZW), yh = cx.alloc(2 * (size_t)TMAX * ZW),
                    stP = F(st_elems), hP = F((size_t)C * 3), yS = F((size_t)TMAX * ZW), hS = F((size_t)C * 3);
        const mtl::Pipeline pre = cx.pipeline(lib, "gdn_pre"), scan = cx.pipeline(lib, "gdn_scanr_4", mtl::Constants().i(6, 0)),
                            post = cx.pipeline(lib, "gdn_post", mtl::Constants().b(SILU_Z, true));
        for (int T : {1, 3, 37, 300}) {
            std::memcpy(stP.ptr, st0.ptr, 4 * st_elems);
            std::memcpy(state.ptr, st0.ptr, 4 * st_elems);
            std::memcpy(hA.ptr, h0.ptr, 4 * (size_t)C * 3);
            GdnPArgs a{(uint32_t)T, HK, HV, C, EPS, 1u};
            mtl::Dispatch d1, d2, d3;
            d1.p = &pre;
            d1.args = {mtl::Arg::B(qkv), mtl::Arg::B(hA), mtl::Arg::B(cw), mtl::Arg::B(al), mtl::Arg::B(be), mtl::Arg::B(dt), mtl::Arg::B(sa),
                       mtl::Arg::B(qn), mtl::Arg::B(kn), mtl::Arg::B(vc), mtl::Arg::B(gb), mtl::Arg::B(kq), mtl::Arg::B(hP), mtl::Arg::V(a)};
            d1.grid[0] = HK + HV / 2;
            d1.grid[1] = T;
            d1.tg[0] = 256;
            d2.p = &scan;
            d2.args = {mtl::Arg::B(stP), mtl::Arg::B(qn), mtl::Arg::B(kn), mtl::Arg::B(vc), mtl::Arg::B(gb), mtl::Arg::B(kq), mtl::Arg::B(o),
                       mtl::Arg::V(a)};
            d2.grid[0] = HV * 4;
            d2.tg[0] = 128;
            d3.p = &post;
            d3.args = {mtl::Arg::B(o), mtl::Arg::B(z), mtl::Arg::B(gm), mtl::Arg::B(yh), mtl::Arg::B(yP), mtl::Arg::V(a)};
            d3.grid[0] = HV;
            d3.grid[1] = T;
            d3.tg[0] = 128;
            cx.run({d1, d2, d3});
            // gdn_step over the same T tokens (NTOK = T, commit), silu
            {
                mtl::Dispatch d = mk(step_pl(T, 2), 0, 1, 0, hA, hS, yS, 0);
                cx.run({d});
            }
            const Ref ref = cpu_ref(T, st0.as<float>(), h0.as<float>(), qkv.as<float>(), al.as<float>(), be.as<float>(), z.as<float>(),
                                    cw.as<float>(), dt.as<float>(), sa.as<float>(), gm.as<float>());
            Err ey, es, eh, eys, ess, ehh;
            const size_t n = (size_t)T * ZW;
            for (size_t i = 0; i < n; ++i) {
                ey.add(yP.as<float>()[i], ref.y[i]);
                eys.add(yP.as<float>()[i], yS.as<float>()[i]);
            }
            for (size_t i = 0; i < st_elems; ++i) { es.add(stP.as<float>()[i], ref.S[i]); ess.add(stP.as<float>()[i], state.as<float>()[i]); }
            for (int i = 0; i < C * 3; ++i) { eh.add(hP.as<float>()[i], ref.H[i]); ehh.add(hP.as<float>()[i], hS.as<float>()[i]); }
            Err e16;
            for (size_t i = 0; i < n; ++i) e16.add(ref::f16_to_f32(yh.as<uint16_t>()[i]), yP.as<float>()[i]);
            const bool ok = ey.l2() < 1e-5 && es.l2() < 1e-5 && eh.l2() == 0 && eys.l2() < 1e-5 && ess.l2() < 1e-5 && ehh.l2() == 0 && e16.l2() < 1e-3;
            fails += !ok;
            std::printf("q36 prompt chain T %3d (silu): vs CPU ref: y rel L2 %.2e (max %.2e), state %.2e, history %s; vs gdn_step: y %.2e, state %.2e, "
                        "history %s; fp16 y %.2e  %s\n", T, ey.l2(), ey.linf(), es.l2(), eh.l2() == 0 ? "exact" : "DIFFERS", eys.l2(), ess.l2(),
                        ehh.l2() == 0 ? "exact" : "DIFFERS", e16.l2(), ok ? "PASS" : "FAIL");
        }
    }

    std::printf("q36_gdn_parity: %s\n", fails ? "FAIL" : "PASS");
    return fails ? 1 : 0;
}
