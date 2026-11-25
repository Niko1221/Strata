#pragma once

#include <cmath>
#include <cstddef>
#include <vector>

namespace strata::qwen35::memt {

inline float sigmoid(float x) { return 1.0f / (1.0f + std::exp(-x)); }

inline float softplus(float x) {
    // Match PyTorch softplus's linear branch above 20.
    return x > 20.0f ? x : std::log1p(std::exp(x));
}

inline float logsigmoid(float x) {
    return x >= 0.0f ? -std::log1p(std::exp(-x)) : x - std::log1p(std::exp(x));
}

inline float silu(float x) { return x * sigmoid(x); }

inline void l2norm_in_place(float* x, int n) {
    float sum = 0.0f;
    for (int i = 0; i < n; ++i) sum += x[i] * x[i];
    const float inv = 1.0f / std::sqrt(sum + 1e-6f);
    for (int i = 0; i < n; ++i) x[i] *= inv;
}

struct Gates {
    float erase;    // per (h, k)
    float write;    // per (h, v)
    float log_decay;// per (h, k)
};

inline Gates token_gates(float erase_raw, float write_raw, float decay_value,
                         float decay_scale_log_h) {
    Gates g;
    g.erase = std::exp(0.5f * logsigmoid(erase_raw));
    g.write = sigmoid(write_raw);
    g.log_decay = -std::exp(decay_scale_log_h) * softplus(decay_value);
    return g;
}

inline void memory_read(const float* q, const float* snapshot, float* out,
                        int heads, int dim) {
    const float scale = 1.0f / std::sqrt((float) dim);
    for (int h = 0; h < heads; ++h) {
        const float* qh = q + (size_t) h * dim;
        const float* sh = snapshot + (size_t) h * dim * dim;
        float* oh = out + (size_t) h * dim;
        for (int v = 0; v < dim; ++v) {
            float acc = 0.0f;
            for (int k = 0; k < dim; ++k) acc += qh[k] * sh[(size_t) k * dim + v];
            oh[v] = acc * scale;
        }
    }
}

inline void gated_rmsnorm(const float* x, const float* gate, const float* weight,
                          float* out, int heads, int dim, float eps,
                          float softcap_limit) {
    for (int h = 0; h < heads; ++h) {
        const float* xh = x + (size_t) h * dim;
        const float* gh = gate + (size_t) h * dim;
        float* oh = out + (size_t) h * dim;
        float ss = 0.0f;
        for (int i = 0; i < dim; ++i) ss += xh[i] * xh[i];
        const float inv = 1.0f / std::sqrt(ss / (float) dim + eps);
        for (int i = 0; i < dim; ++i) {
            const float gated = softcap_limit * std::tanh(silu(gh[i]) / softcap_limit);
            oh[i] = xh[i] * inv * weight[i] * gated;
        }
    }
}

// One token's stable-GDN-2 update.  `k` already carries the erase gate; `beta` is
// the write gate, applied to v only (delta = beta*v - S^T k).  The state is
// [H][K][V], K == V == dim.
inline void memory_update(float* state, const float* k, const float* v,
                          const float* log_decay, const float* beta, int heads,
                          int dim) {
    for (int h = 0; h < heads; ++h) {
        float* sh = state + (size_t) h * dim * dim;
        const float* kh = k + (size_t) h * dim;
        const float* vh = v + (size_t) h * dim;
        const float* gh = log_decay + (size_t) h * dim;
        const float* bh = beta + (size_t) h * dim;
        for (int i = 0; i < dim; ++i) {
            const float decay = std::exp(gh[i]);
            for (int j = 0; j < dim; ++j) sh[(size_t) i * dim + j] *= decay;
        }
        std::vector<float> delta((size_t) dim);
        for (int j = 0; j < dim; ++j) {
            float kv = 0.0f;
            for (int i = 0; i < dim; ++i) kv += sh[(size_t) i * dim + j] * kh[i];
            delta[(size_t) j] = vh[j] * bh[j] - kv;
        }
        for (int i = 0; i < dim; ++i) {
            for (int j = 0; j < dim; ++j) sh[(size_t) i * dim + j] += kh[i] * delta[(size_t) j];
        }
    }
}

}  // namespace strata::qwen35::memt
