#include "dsv4/ops.hpp"

#include <algorithm>
#include <cmath>
#include <numeric>
#ifdef _OPENMP
#include <omp.h>
#endif

namespace dsv4 {

void rmsnorm(const float* x, const float* w, float eps, int n, float* y) {
    double ss = 0;
    for (int i = 0; i < n; ++i) ss += (double) x[i] * x[i];
    const float r = 1.0f / std::sqrt((float) (ss / n) + eps);
    for (int i = 0; i < n; ++i) y[i] = (w ? w[i] : 1.0f) * (x[i] * r);
}

void yarn_table(int dim, int seqlen, int orig_len, double base, double factor, double beta_fast, double beta_slow,
                std::vector<float>& cos_t, std::vector<float>& sin_t) {
    const int h = dim / 2;
    std::vector<float> freqs((size_t) h);
    for (int i = 0; i < h; ++i) freqs[(size_t) i] = (float) (1.0 / std::pow(base, (double) (2 * i) / dim));
    if (orig_len > 0) {
        auto cdim = [&](double rot) { return dim * std::log(orig_len / (rot * 2 * M_PI)) / (2 * std::log(base)); };
        double low = std::max(std::floor(cdim(beta_fast)), 0.0);
        double high = std::min(std::ceil(cdim(beta_slow)), (double) (dim - 1));
        double mx = (low == high) ? high + 0.001 : high;
        for (int i = 0; i < h; ++i) {
            double ramp = std::min(std::max((i - low) / (mx - low), 0.0), 1.0);
            double smooth = 1.0 - ramp;
            freqs[(size_t) i] = (float) (freqs[(size_t) i] / factor * (1 - smooth) + freqs[(size_t) i] * smooth);
        }
    }
    cos_t.assign((size_t) seqlen * h, 0);
    sin_t.assign((size_t) seqlen * h, 0);
    for (int t = 0; t < seqlen; ++t)
        for (int i = 0; i < h; ++i) {
            const double a = (double) t * freqs[(size_t) i];
            cos_t[(size_t) t * h + i] = (float) std::cos(a);
            sin_t[(size_t) t * h + i] = (float) std::sin(a);
        }
}

void rotary(float* x, int n, int rope_dim, const float* c, const float* s, bool inverse) {
    float* p = x + (n - rope_dim);
    for (int i = 0; i < rope_dim / 2; ++i) {
        const float a = p[2 * i], b = p[2 * i + 1], sn = inverse ? -s[i] : s[i];
        p[2 * i] = a * c[i] - b * sn;
        p[2 * i + 1] = a * sn + b * c[i];
    }
}

void gate_route(const float* logits, int n, int topk, Score fn, const float* bias, const int32_t* hash_idx,
                float route_scale, int32_t* idx, float* w) {
    std::vector<double> sc((size_t) n);
    if (fn == Score::Softmax) {
        double m = *std::max_element(logits, logits + n), z = 0;
        for (int i = 0; i < n; ++i) { sc[(size_t) i] = std::exp(logits[i] - m); z += sc[(size_t) i]; }
        for (auto& v : sc) v /= z;
    } else if (fn == Score::Sigmoid) {
        for (int i = 0; i < n; ++i) sc[(size_t) i] = 1.0 / (1.0 + std::exp(-(double) logits[i]));
    } else {
        for (int i = 0; i < n; ++i) {
            const double x = logits[i];
            sc[(size_t) i] = std::sqrt(x > 30 ? x : std::log1p(std::exp(x)));  // sqrt(softplus(x))
        }
    }
    if (hash_idx) {
        for (int k = 0; k < topk; ++k) idx[k] = hash_idx[k];
    } else {
        std::vector<int> o((size_t) n);
        std::iota(o.begin(), o.end(), 0);
        auto key = [&](int i) { return sc[(size_t) i] + (bias ? (double) bias[i] : 0.0); };
        std::partial_sort(o.begin(), o.begin() + topk, o.end(), [&](int a, int b) {
            const double ka = key(a), kb = key(b);
            return ka != kb ? ka > kb : a < b;
        });
        for (int k = 0; k < topk; ++k) idx[k] = o[(size_t) k];
    }
    double sum = 0;
    for (int k = 0; k < topk; ++k) { w[k] = (float) sc[(size_t) idx[k]]; sum += sc[(size_t) idx[k]]; }
    for (int k = 0; k < topk; ++k) {
        double v = sc[(size_t) idx[k]];
        if (fn != Score::Softmax) v /= sum;
        w[k] = (float) (v * route_scale);
    }
}

void swiglu_clamped(const float* g, const float* u, int n, float limit, float* out) {
    for (int i = 0; i < n; ++i) {
        float a = g[i], b = u[i];
        if (limit > 0) { a = std::min(a, limit); b = std::min(std::max(b, -limit), limit); }
        out[i] = (float) ((double) a / (1.0 + std::exp(-(double) a))) * b;
    }
}

void window_topk(int win, int seqlen, int start_pos, std::vector<int>& out, int& rows, int& cols) {
    out.clear();
    if (start_pos >= win - 1) {
        const int sp = start_pos % win;
        rows = 1; cols = win;
        for (int i = sp + 1; i < win; ++i) out.push_back(i);
        for (int i = 0; i <= sp; ++i) out.push_back(i);
    } else if (start_pos > 0) {
        rows = 1; cols = win;
        for (int i = 0; i <= start_pos; ++i) out.push_back(i);
        while ((int) out.size() < win) out.push_back(-1);
    } else {
        rows = seqlen; cols = std::min(seqlen, win);
        for (int b = 0; b < seqlen; ++b)
            for (int j = 0; j < cols; ++j) {
                const int m = std::max(b - win + 1, 0) + j;
                out.push_back(m > b ? -1 : m);
            }
    }
}

void compress_topk(int ratio, int seqlen, int start_pos, int offset, std::vector<int>& out, int& rows, int& cols) {
    out.clear();
    if (start_pos > 0) {
        rows = 1; cols = (start_pos + 1) / ratio;
        for (int i = 0; i < cols; ++i) out.push_back(i + offset);
    } else {
        rows = seqlen; cols = seqlen / ratio;
        for (int b = 0; b < seqlen; ++b)
            for (int j = 0; j < cols; ++j) out.push_back(j >= (b + 1) / ratio ? -1 : j + offset);
    }
}

namespace {
inline double sigm(double v) { return 1.0 / (1.0 + std::exp(-v)); }
}

void hc_split_sinkhorn(const float* mixes, const float* scale, const float* base, int hc, int iters, float eps,
                       float* pre, float* post, float* comb) {
    for (int j = 0; j < hc; ++j) {
        pre[j] = (float) (sigm((double) mixes[j] * scale[0] + base[j]) + eps);
        post[j] = (float) (2.0 * sigm((double) mixes[j + hc] * scale[1] + base[j + hc]));
    }
    std::vector<double> c((size_t) hc * hc);
    for (int j = 0; j < hc; ++j) {
        double mx = -1e300;
        for (int k = 0; k < hc; ++k) {
            c[(size_t) (j * hc + k)] = (double) mixes[2 * hc + j * hc + k] * scale[2] + base[2 * hc + j * hc + k];
            mx = std::max(mx, c[(size_t) (j * hc + k)]);
        }
        double z = 0;
        for (int k = 0; k < hc; ++k) { c[(size_t) (j * hc + k)] = std::exp(c[(size_t) (j * hc + k)] - mx); z += c[(size_t) (j * hc + k)]; }
        for (int k = 0; k < hc; ++k) c[(size_t) (j * hc + k)] = c[(size_t) (j * hc + k)] / z + eps;
    }
    auto col_norm = [&]() {
        for (int k = 0; k < hc; ++k) {
            double s = 0;
            for (int j = 0; j < hc; ++j) s += c[(size_t) (j * hc + k)];
            for (int j = 0; j < hc; ++j) c[(size_t) (j * hc + k)] /= (s + eps);
        }
    };
    auto row_norm = [&]() {
        for (int j = 0; j < hc; ++j) {
            double s = 0;
            for (int k = 0; k < hc; ++k) s += c[(size_t) (j * hc + k)];
            for (int k = 0; k < hc; ++k) c[(size_t) (j * hc + k)] /= (s + eps);
        }
    };
    col_norm();
    for (int it = 0; it < iters - 1; ++it) { row_norm(); col_norm(); }
    for (int i = 0; i < hc * hc; ++i) comb[i] = (float) c[(size_t) i];
}

// hc=4, d=4096: hc_pre is (2+hc)*hc = 24 dot products over 16384 doubles, twice per layer, 43 layers.
// Every output row is independent and its summation order is untouched, so the pragma is free correctness-wise.
void hc_pre(const float* x, int hc, int d, const float* fn, const float* scale, const float* base, float norm_eps,
            int iters, float eps, float* y, float* post, float* comb) {
    const int n = hc * d, mix = (2 + hc) * hc;
    double ss = 0;
    for (int i = 0; i < n; ++i) ss += (double) x[i] * x[i];
    const double rs = 1.0 / std::sqrt(ss / n + norm_eps);
    std::vector<float> mixes((size_t) mix), pre((size_t) hc);
#ifdef _OPENMP
#pragma omp parallel for schedule(static)
#endif
    for (int m = 0; m < mix; ++m) {
        double a = 0;
        for (int i = 0; i < n; ++i) a += (double) fn[(size_t) m * n + i] * x[i];
        mixes[(size_t) m] = (float) (a * rs);
    }
    hc_split_sinkhorn(mixes.data(), scale, base, hc, iters, eps, pre.data(), post, comb);
#ifdef _OPENMP
#pragma omp parallel for schedule(static)
#endif
    for (int i = 0; i < d; ++i) {
        double a = 0;
        for (int j = 0; j < hc; ++j) a += (double) pre[(size_t) j] * x[j * d + i];
        y[i] = (float) a;
    }
}

void hc_post(const float* x, const float* residual, const float* post, const float* comb, int hc, int d, float* out) {
#ifdef _OPENMP
#pragma omp parallel for schedule(static)
#endif
    for (int k = 0; k < hc; ++k)
        for (int i = 0; i < d; ++i) {
            double a = (double) post[k] * x[i];
            for (int j = 0; j < hc; ++j) a += (double) comb[j * hc + k] * residual[j * d + i];
            out[k * d + i] = (float) a;
        }
}

void hc_head(const float* x, int hc, int d, const float* fn, const float* scale, const float* base, float norm_eps,
             float hc_eps, float* y) {
    const int n = hc * d;
    double ss = 0;
    for (int i = 0; i < n; ++i) ss += (double) x[i] * x[i];
    const double rs = 1.0 / std::sqrt(ss / n + norm_eps);
    std::vector<double> pre((size_t) hc);
#ifdef _OPENMP
#pragma omp parallel for schedule(static)
#endif
    for (int m = 0; m < hc; ++m) {
        double a = 0;
        for (int i = 0; i < n; ++i) a += (double) fn[(size_t) m * n + i] * x[i];
        pre[(size_t) m] = sigm(a * rs * scale[0] + base[m]) + hc_eps;
    }
#ifdef _OPENMP
#pragma omp parallel for schedule(static)
#endif
    for (int i = 0; i < d; ++i) {
        double a = 0;
        for (int j = 0; j < hc; ++j) a += pre[(size_t) j] * x[j * d + i];
        y[i] = (float) a;
    }
}

// The most expensive op of a token: n_head(64) x topk(128+) x head_dim(512), walked twice, in double.
// Heads are fully independent (own q row, own output row), so parallelising over them changes no head's
// arithmetic. The score scratch is thread_local: 64 heads x 43 layers x every token was a malloc per head.
void sparse_attn_token(const float* q, int h, int d, const float* kv, const int* idxs, int topk, const float* sink,
                       float scale, float* o) {
    std::vector<int> v;
    for (int t = 0; t < topk; ++t) if (idxs[t] >= 0) v.push_back(idxs[t]);
#ifdef _OPENMP
#pragma omp parallel for schedule(static)
#endif
    for (int hh = 0; hh < h; ++hh) {
        float* oo = o + (size_t) hh * d;
        std::fill(oo, oo + d, 0.f);
        if (v.empty()) continue;
        static thread_local std::vector<double> s;
        s.resize(v.size());
        const float* qq = q + (size_t) hh * d;
        double mx = -1e300;
        for (size_t t = 0; t < v.size(); ++t) {
            const float* kk = kv + (size_t) v[t] * d;
            double a = 0;
            for (int i = 0; i < d; ++i) a += (double) qq[i] * kk[i];
            s[t] = a * scale;
            mx = std::max(mx, s[t]);
        }
        double den = std::exp((double) sink[hh] - mx);
        for (auto& e : s) { e = std::exp(e - mx); den += e; }
        for (size_t t = 0; t < v.size(); ++t) {
            const float* kk = kv + (size_t) v[t] * d;
            for (int i = 0; i < d; ++i) oo[i] += (float) (s[t] / den * kk[i]);
        }
    }
}

}  // namespace dsv4
