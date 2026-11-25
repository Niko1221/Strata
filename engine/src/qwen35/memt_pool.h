// CPU reference for MemT session memory.
#pragma once

#include "memt_math.h"

#include <algorithm>
#include <cstddef>
#include <vector>

namespace strata::qwen35::memt {

struct Pool {
    int sessions = 0;
    int layers = 0;
    int heads = 0;
    int dim = 0;
    std::vector<float> data;

    Pool() = default;
    Pool(int sessions_, int layers_, int heads_, int dim_)
        : sessions(sessions_), layers(layers_), heads(heads_), dim(dim_),
          data((size_t) sessions_ * layers_ * heads_ * dim_ * dim_, 0.0f) {}

    size_t layer_stride() const { return (size_t) heads * dim * dim; }
    size_t session_stride() const { return (size_t) layers * layer_stride(); }

    float* at(int session, int layer) {
        return data.data() + (size_t) session * session_stride() +
               (size_t) layer * layer_stride();
    }
    const float* at(int session, int layer) const {
        return data.data() + (size_t) session * session_stride() +
               (size_t) layer * layer_stride();
    }
    void zero_session(int session) {
        std::fill_n(at(session, 0), session_stride(), 0.0f);
    }
    std::vector<float> export_session(int session) const {
        return std::vector<float>(at(session, 0), at(session, 0) + session_stride());
    }
    void import_session(int session, const float* values) {
        std::copy_n(values, session_stride(), at(session, 0));
    }
};

// out[t, h, :] = gated_rmsnorm((l2norm(q[t,h]) @ state[h]) * D^-0.5, out_gate[t,h]).
inline void query(const float* state, const float* q, const float* out_gate,
                  const float* weight, float* out, int tokens, int heads, int dim,
                  float eps, float softcap) {
    const size_t per_token = (size_t) heads * dim;
    std::vector<float> normalized(per_token);
    std::vector<float> raw(per_token);
    for (int t = 0; t < tokens; ++t) {
        for (int h = 0; h < heads; ++h) {
            for (int d = 0; d < dim; ++d) normalized[h * dim + d] = q[t * per_token + h * dim + d];
            l2norm_in_place(normalized.data() + h * dim, dim);
        }
        memory_read(normalized.data(), state, raw.data(), heads, dim);
        gated_rmsnorm(raw.data(), out_gate + t * per_token, weight,
                      out + t * per_token, heads, dim, eps, softcap);
    }
}

// Commit `tokens` tokens: every token reads the state at the call's start, then
// `state` advances in place in write events of `chunk_tokens`.  `out` receives
// the read result of every token.  `erase` is the release's gate,
// exp(0.5 * logsigmoid(erase_raw)), applied to the normalized key as is.
inline void commit(float* state, const float* q, const float* k, const float* v,
                   const float* erase, const float* write, const float* log_decay,
                   const float* out_gate, const float* o_norm, float* out,
                   int tokens, int heads, int dim, int chunk_tokens, float eps,
                   float softcap) {
    const size_t per_token = (size_t) heads * dim;
    const size_t state_size = per_token * dim;
    std::vector<float> snapshot(state_size);
    std::vector<float> normalized(per_token);
    std::vector<float> k_eff(per_token);
    std::vector<float> raw(per_token);
    std::copy_n(state, state_size, snapshot.data());   // the call's frozen read state
    int pos = 0;
    while (pos < tokens) {
        const int length = std::min(chunk_tokens > 0 ? chunk_tokens : tokens, tokens - pos);
        for (int i = 0; i < length; ++i) {
            const int t = pos + i;
            for (int h = 0; h < heads; ++h) {
                for (int d = 0; d < dim; ++d) {
                    normalized[h * dim + d] = q[t * per_token + h * dim + d];
                    k_eff[h * dim + d] = k[t * per_token + h * dim + d];
                }
                l2norm_in_place(normalized.data() + h * dim, dim);
                l2norm_in_place(k_eff.data() + h * dim, dim);
            }
            for (int i2 = 0; i2 < (int) per_token; ++i2) k_eff[i2] *= erase[t * per_token + i2];
            memory_read(normalized.data(), snapshot.data(), raw.data(), heads, dim);
            gated_rmsnorm(raw.data(), out_gate + t * per_token, o_norm,
                          out + t * per_token, heads, dim, eps, softcap);
            memory_update(state, k_eff.data(), v + t * per_token,
                          log_decay + t * per_token, write + t * per_token, heads, dim);
        }
        pos += length;
    }
}

}  // namespace strata::qwen35::memt
