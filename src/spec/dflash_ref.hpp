// src/spec/dflash_ref.hpp - a plain CPU reference of the DFlash2 draft model (native-spec/graph.md) with f64
// accumulation, for the parity test: it measures how far each backend is from the arithmetic truth, so a tolerance is
// calibrated rather than guessed.  Slow (a feature row costs 60M MACs): use short contexts.  Test code only.
#pragma once

#include "gguf_lite.h"
#include "ref_quant.h"

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <functional>
#include <map>
#include <thread>
#include <vector>

namespace dfref {

using Vec = std::vector<double>;
constexpr int D = 2560, NL = 5, HD = 128, NH = 32, NKV = 8, FF = 7680, TOPK = 16, WIN = 2048;

inline void parallel_for(int n, const std::function<void(int, int)>& fn) {
    const int nt = std::max(1u, std::min(16u, std::thread::hardware_concurrency()));
    std::vector<std::thread> th;
    for (int t = 0; t < nt; ++t) th.emplace_back([&, t] { fn(t * n / nt, (t + 1) * n / nt); });
    for (auto& x : th) x.join();
}

struct Model {
    gguf::File f;
    std::vector<int32_t> d2t;
    int mask = 248077;
    std::vector<Vec> hfin, logits;   // the last draft's final-norm rows 1.. and their draft-vocab logits
    // K/V cache of the injected rows, per layer: position -> (K 8x128 after norm + RoPE, V), rounded to f16 as llama.cpp's cache is
    std::map<int, std::vector<float>> k[NL], v[NL];

    bool load(const std::string& path, std::string& err) {
        if (!f.open(path, err)) return false;
        const gguf::Tensor& t = *f.find("d2t");
        const int64_t* s = (const int64_t*) ((const uint8_t*) f.map + t.offset);
        for (uint64_t i = 0; i < t.shape[0]; ++i) d2t.push_back((int32_t) s[i]);
        return true;
    }
    const gguf::Tensor& T(const std::string& n) const { return *f.find(n); }
    std::vector<float> w32(const std::string& n) const {
        const gguf::Tensor& t = T(n);
        const float* p = (const float*) ((const uint8_t*) f.map + t.offset);
        return std::vector<float>(p, p + t.bytes / 4);
    }
    void row(const gguf::Tensor& t, uint64_t r, float* out) const {
        ref::dequant(t.type, (const uint8_t*) f.map + t.offset + r * ref::row_bytes(t.type, (int) t.shape[0]), out, t.shape[0]);
    }
    // Y[n][o] = sum_i W[o][i] X[n][i]
    void mm(const gguf::Tensor& t, const std::vector<Vec>& X, std::vector<Vec>& Y) const {
        const int n_in = (int) t.shape[0], n_out = (int) t.shape[1];
        Y.assign(X.size(), Vec(n_out));
        parallel_for(n_out, [&](int a, int b) {
            std::vector<float> w(n_in);
            for (int o = a; o < b; ++o) {
                row(t, o, w.data());
                for (size_t n = 0; n < X.size(); ++n) {
                    double s = 0;
                    for (int i = 0; i < n_in; ++i) s += (double) w[i] * X[n][i];
                    Y[n][o] = s;
                }
            }
        });
    }
    static Vec rms(const double* x, const float* w, int n) {
        double ss = 0;
        for (int i = 0; i < n; ++i) ss += x[i] * x[i];
        const double r = 1.0 / std::sqrt(ss / n + (double) 1e-6f);
        Vec y(n);
        for (int i = 0; i < n; ++i) y[i] = x[i] * r * w[i];
        return y;
    }
    // a 128-wide head: rmsnorm then NEOX RoPE at position p
    static void head_norm_rope(double* x, const float* w, int p) {
        Vec n = rms(x, w, HD);
        for (int k = 0; k < HD / 2; ++k) {
            const double th = p * std::pow(10000.0, -2.0 * k / HD), c = std::cos(th), s = std::sin(th);
            x[k] = n[k] * c - n[k + 64] * s;
            x[k + 64] = n[k] * s + n[k + 64] * c;
        }
    }
    static float f16r(double x) { return ref::f16_to_f32(ref::f32_to_f16((float) x)); }
    static std::string L(int il, const char* n) { return "blk." + std::to_string(il) + "." + n; }

    void clear() { for (int il = 0; il < NL; ++il) { k[il].clear(); v[il].clear(); } }

    void inject(const float* feats, int n, int pos0) {
        std::vector<Vec> X(n, Vec(5 * D)), G;
        for (int r = 0; r < n; ++r) std::copy(feats + (size_t) r * 5 * D, feats + (size_t) (r + 1) * 5 * D, X[r].begin());
        mm(T("fc.weight"), X, G);
        const auto en = w32("enc.output_norm.weight");
        for (auto& g : G) g = rms(g.data(), en.data(), D);
        for (int il = 0; il < NL; ++il) {
            std::vector<Vec> K, V;
            mm(T(L(il, "attn_k.weight")), G, K);
            mm(T(L(il, "attn_v.weight")), G, V);
            const auto kn = w32(L(il, "attn_k_norm.weight"));
            for (int r = 0; r < n; ++r) {
                std::vector<float> ko(1024), vo(1024);
                for (int h = 0; h < NKV; ++h) head_norm_rope(&K[r][h * HD], kn.data(), pos0 + r);
                for (int i = 0; i < 1024; ++i) { ko[i] = f16r(K[r][i]); vo[i] = f16r(V[r][i]); }
                k[il][pos0 + r] = ko;
                v[il][pos0 + r] = vo;
            }
        }
    }
    static Vec conv(const std::vector<Vec>& x, const std::vector<Vec>& dyn, const std::vector<float>& base, int side, int t, int skip = 0) {
        (void) skip;
        Vec y(D);
        for (int ch = 0; ch < D; ++ch) {
            const int g = ch / 16;
            const double w0 = base[ch + 5120 * side] + dyn[t][g + 320 * side], w1 = base[ch + 2560 + 5120 * side] + dyn[t][g + 160 + 320 * side];
            y[ch] = w0 * x[t][ch] + (t > 0 ? w1 * x[t - 1][ch] : 0.0);
        }
        return y;
    }
    // the lattice rows (nb x 2560, row 0 zero) of the block [anchor, mask...] at pos0..
    std::vector<float> draft(int32_t anchor, int pos0, int nb) {
        const gguf::Tensor& te = T("token_embd.weight");
        std::vector<Vec> x(nb, Vec(D));
        std::vector<float> e(D);
        for (int r = 0; r < nb; ++r) {
            row(te, r == 0 ? anchor : mask, e.data());
            for (int i = 0; i < D; ++i) x[r][i] = e[i];
        }
        for (int il = 0; il < NL; ++il) {
            std::vector<Vec> h(nb), dyn, hc(nb), Q, K, V, att(nb, Vec(NH * HD)), O, oc(nb), ffn_inp(nb), f(nb), fdyn, fc(nb), Gt, Up, hh(nb), M, mc(nb);
            const auto an = w32(L(il, "attn_norm.weight")), ab = w32(L(il, "attn_conv_base")), qn = w32(L(il, "attn_q_norm.weight")),
                       kn = w32(L(il, "attn_k_norm.weight")), fnw = w32(L(il, "ffn_norm.weight")), fb = w32(L(il, "ffn_conv_base"));
            for (int r = 0; r < nb; ++r) h[r] = rms(x[r].data(), an.data(), D);
            mm(T(L(il, "attn_conv_proj.weight")), h, dyn);
            for (int r = 0; r < nb; ++r) hc[r] = conv(h, dyn, ab, 0, r);
            mm(T(L(il, "attn_q.weight")), hc, Q);
            mm(T(L(il, "attn_k.weight")), hc, K);
            mm(T(L(il, "attn_v.weight")), hc, V);
            for (int r = 0; r < nb; ++r) {
                for (int hd = 0; hd < NH; ++hd) head_norm_rope(&Q[r][hd * HD], qn.data(), pos0 + r);
                for (int hd = 0; hd < NKV; ++hd) head_norm_rope(&K[r][hd * HD], kn.data(), pos0 + r);
                for (int i = 0; i < 1024; ++i) { K[r][i] = f16r(K[r][i]); V[r][i] = f16r(V[r][i]); }
            }
            for (int r = 0; r < nb; ++r) {
                const int p1 = pos0 + r;
                for (int hd = 0; hd < NH; ++hd) {
                    const int kv = hd / 4;
                    std::vector<double> sc;
                    std::vector<const float*> vp;
                    std::vector<double> vd;   // block rows' V as double
                    for (auto& [p, kk] : k[il]) {
                        if (p1 - p >= WIN) continue;
                        double s = 0;
                        for (int i = 0; i < HD; ++i) s += Q[r][hd * HD + i] * kk[kv * HD + i];
                        sc.push_back(s / std::sqrt((double) HD));
                        vp.push_back(&v[il][p][kv * HD]);
                    }
                    const size_t nctx = sc.size();
                    for (int j = 0; j < nb; ++j) {
                        double s = 0;
                        for (int i = 0; i < HD; ++i) s += Q[r][hd * HD + i] * K[j][kv * HD + i];
                        sc.push_back(s / std::sqrt((double) HD));
                    }
                    double m = *std::max_element(sc.begin(), sc.end()), l = 0;
                    for (double& s : sc) { s = std::exp(s - m); l += s; }
                    for (int i = 0; i < HD; ++i) {
                        double a = 0;
                        for (size_t j = 0; j < nctx; ++j) a += sc[j] * vp[j][i];
                        for (int j = 0; j < nb; ++j) a += sc[nctx + j] * V[j][kv * HD + i];
                        att[r][hd * HD + i] = a / l;
                    }
                }
            }
            mm(T(L(il, "attn_output.weight")), att, O);
            for (int r = 0; r < nb; ++r) {
                oc[r] = conv(O, dyn, ab, 1, r);
                ffn_inp[r].resize(D);
                for (int i = 0; i < D; ++i) ffn_inp[r][i] = oc[r][i] + x[r][i];
                f[r] = rms(ffn_inp[r].data(), fnw.data(), D);
            }
            mm(T(L(il, "ffn_conv_proj.weight")), f, fdyn);
            for (int r = 0; r < nb; ++r) fc[r] = conv(f, fdyn, fb, 0, r);
            mm(T(L(il, "ffn_gate.weight")), fc, Gt);
            mm(T(L(il, "ffn_up.weight")), fc, Up);
            for (int r = 0; r < nb; ++r) {
                hh[r].resize(FF);
                for (int i = 0; i < FF; ++i) hh[r][i] = Gt[r][i] / (1.0 + std::exp(-Gt[r][i])) * Up[r][i];
            }
            mm(T(L(il, "ffn_down.weight")), hh, M);
            for (int r = 0; r < nb; ++r) {
                mc[r] = conv(M, fdyn, fb, 1, r);
                for (int i = 0; i < D; ++i) x[r][i] = mc[r][i] + ffn_inp[r][i];
            }
        }
        const auto on = w32("output_norm.weight");
        std::vector<Vec> gate;
        hfin.assign(nb - 1, Vec());
        logits.clear();
        for (int r = 1; r < nb; ++r) hfin[r - 1] = rms(x[r].data(), on.data(), D);
        mm(T("output.weight"), hfin, logits);
        mm(T("selector_hidden.weight"), hfin, gate);
        std::vector<float> lat((size_t) nb * D, 0.0f);
        std::vector<std::vector<int>> cand(nb);
        std::vector<std::vector<double>> un(nb);
        for (int r = 1; r < nb; ++r) {
            std::vector<int> idx(logits[r - 1].size());
            for (size_t i = 0; i < idx.size(); ++i) idx[i] = (int) i;
            const Vec& lg = logits[r - 1];
            std::partial_sort(idx.begin(), idx.begin() + TOPK, idx.end(), [&](int a, int b) { return lg[a] != lg[b] ? lg[a] > lg[b] : a < b; });
            for (int s = 0; s < TOPK; ++s) { cand[r].push_back(d2t[idx[s]]); un[r].push_back(lg[idx[s]]); }
        }
        const gguf::Tensor &pre = T("selector_predecessor.weight"), &nxt = T("selector_successor.weight");
        std::vector<float> a(256), b(256);
        for (int r = 1; r < nb; ++r) {
            for (int s = 0; s < TOPK; ++s) lat[(size_t) r * D + s] = (float) cand[r][s];
            for (int ai = 0; ai < TOPK; ++ai) {
                row(pre, r == 1 ? anchor : cand[r - 1][ai], a.data());
                for (int s = 0; s < TOPK; ++s) {
                    row(nxt, cand[r][s], b.data());
                    double acc = 0;
                    for (int i = 0; i < 256; ++i) acc += (double) b[i] * ((double) a[i] * gate[r - 1][i]);
                    lat[(size_t) r * D + 16 + 16 * ai + s] = (float) (acc + un[r][s]);
                }
            }
        }
        return lat;
    }
};

// Scores of two lattices matched by (predecessor id, successor id): max / mean |difference| over the pairs both hold
// (the candidate order of a backend is its own), and how many candidates per row they share (min over rows).
inline void lattice_diff(const float* A, const float* B, int nb, int anchor, double& mx, double& mean, int& shared_min) {
    mx = 0;
    mean = 0;
    shared_min = 16;
    long cnt = 0;
    for (int p = 1; p < nb; ++p) {
        std::map<std::pair<int, int>, float> ma, mb;
        auto fill = [&](const float* L, std::map<std::pair<int, int>, float>& m) {
            for (int a = 0; a < 16; ++a)
                for (int s = 0; s < 16; ++s) {
                    const int pid = p == 1 ? anchor : (int) L[(size_t) (p - 1) * D + a];
                    m[{pid, (int) L[(size_t) p * D + s]}] = L[(size_t) p * D + 16 + 16 * a + s];
                }
        };
        fill(A, ma);
        fill(B, mb);
        int shared = 0;
        for (int s = 0; s < 16; ++s) {
            const int id = (int) A[(size_t) p * D + s];
            for (int t = 0; t < 16; ++t) shared += id == (int) B[(size_t) p * D + t];
        }
        shared_min = std::min(shared_min, shared);
        for (auto& [key, va] : ma) {
            auto it = mb.find(key);
            if (it == mb.end()) continue;
            const double d = std::fabs((double) va - it->second);
            mx = std::max(mx, d);
            mean += d;
            ++cnt;
        }
    }
    if (cnt) mean /= cnt;
}

}  // namespace dfref
