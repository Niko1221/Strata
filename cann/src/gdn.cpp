#include "strata/cann/gdn.hpp"

#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <vector>

#include "strata/cann/ops.hpp"

namespace strata::cann {
namespace {

// One fp16 scalar from device memory. Copying two bytes straight into a float leaves the
// upper bytes of the float untouched, which reads as a denormal near zero.
float read_half(const void * device_ptr) {
    uint16_t raw = 0;
    copy_to_host(device_ptr, &raw, 2);
    float f = 0.0f;
    f16_to_f32(&raw, &f, 1);
    return f;
}

// aclnn ops write into a distinct output tensor; copying with a scale of 1 keeps the port
// free of a separate "copy" op.
void assign(Stream & stream, const Tensor & src, Tensor & dst) {
    muls(stream, src, 1.0f, dst);
}

} // namespace

void gdn_ref(const float * q, const float * k, const float * v, const float * g, const float * beta, float * state,
             float * attn, const GdnShape & shape) {
    const int64_t S = shape.S;
    const float scale = 1.0f / std::sqrt((float) S);

    if (shape.kda) {
        // The per-channel gate path is wrong for S > 64 (gdn_parity shows 0.70 relative error at
        // S=128): the decay has to scale column i of the state by exp(g[i]), and the broadcast we
        // use does not express that. qwen3.5-moe uses the scalar gate, so rather than ship a path
        // that silently returns wrong numbers this refuses.
        throw Error("cann: gdn with a per-channel gate (KDA) is not implemented", -1);
    }

    for (int64_t t = 0; t < shape.T; ++t) {
        const float * kt = k + t * S;
        const float * qt = q + t * S;
        const float * vt = v + t * S;
        const float bt = beta[t];

        if (shape.kda) {
            const float * gt = g + t * S;
            for (int64_t j = 0; j < S; ++j) {
                float * row = state + j * S;
                for (int64_t i = 0; i < S; ++i) {
                    row[i] *= std::exp(gt[i]);
                }
            }
        } else {
            const float decay = std::exp(g[t]);
            for (int64_t i = 0; i < S * S; ++i) {
                state[i] *= decay;
            }
        }

        std::vector<float> delta((size_t) S);
        for (int64_t j = 0; j < S; ++j) {
            const float * row = state + j * S;
            float dot = 0.0f;
            for (int64_t i = 0; i < S; ++i) {
                dot += row[i] * kt[i];
            }
            delta[(size_t) j] = (vt[j] - dot) * bt;
        }

        for (int64_t j = 0; j < S; ++j) {
            float * row = state + j * S;
            const float d = delta[(size_t) j];
            for (int64_t i = 0; i < S; ++i) {
                row[i] += kt[i] * d;
            }
        }

        float * at = attn + t * S;
        for (int64_t j = 0; j < S; ++j) {
            const float * row = state + j * S;
            float dot = 0.0f;
            for (int64_t i = 0; i < S; ++i) {
                dot += row[i] * qt[i];
            }
            at[j] = dot * scale;
        }
    }
}

void gdn(Stream & stream, const Tensor & q, const Tensor & k, const Tensor & v, const Tensor & g,
         const Tensor & beta, Tensor & state, Tensor & attn, const GdnShape & shape) {
    const int64_t S = shape.S;
    const size_t row_bytes = (size_t) S * 2;
    const size_t mat_bytes = (size_t) S * S * 2;
    const float scale = 1.0f / std::sqrt((float) S);

    if (shape.kda) {
        // The per-channel gate path is wrong for S > 64 (gdn_parity shows 0.70 relative error at
        // S=128): the decay has to scale column i of the state by exp(g[i]), and the broadcast we
        // use does not express that. qwen3.5-moe uses the scalar gate, so rather than ship a path
        // that silently returns wrong numbers this refuses.
        throw Error("cann: gdn with a per-channel gate (KDA) is not implemented", -1);
    }

    // working state M and two scratch matrices; ops never alias input and output
    Buffer m_b(mat_bytes), tmp_b(mat_bytes), tmp2_b(mat_bytes);
    Buffer vec_b(row_bytes), kv_b(row_bytes), delta_b(row_bytes), expg_b(row_bytes);
    Tensor M{ m_b.data(), DType::F16, S, S };
    Tensor tmp{ tmp_b.data(), DType::F16, S, S };
    Tensor tmp2{ tmp2_b.data(), DType::F16, S, S };
    Tensor vec{ vec_b.data(), DType::F16, 1, S };
    Tensor kv{ kv_b.data(), DType::F16, 1, S };
    Tensor delta{ delta_b.data(), DType::F16, 1, S };
    Tensor expg{ expg_b.data(), DType::F16, S, 1 };

    assign(stream, state, M);

    for (int64_t t = 0; t < shape.T; ++t) {
        Tensor k_col{ (char *) k.data + t * row_bytes, DType::F16, 1, S };
        Tensor k_row{ (char *) k.data + t * row_bytes, DType::F16, S, 1 };
        Tensor q_col{ (char *) q.data + t * row_bytes, DType::F16, 1, S };
        Tensor v_col{ (char *) v.data + t * row_bytes, DType::F16, 1, S };
        Tensor attn_row{ (char *) attn.data + t * row_bytes, DType::F16, 1, S }; // the row seen as an [S,1] result

        // decay
        if (shape.kda) {
            Tensor g_row{ (char *) g.data + t * row_bytes, DType::F16, S, 1 };
            exp(stream, g_row, expg);
            mul(stream, M, expg, tmp); // column i scaled by exp(g[i])
        } else {
            const float gv = read_half((char *) g.data + t * 2);
            muls(stream, M, std::exp(gv), tmp);
        }
        assign(stream, tmp, M);

        // delta = (v - M k) * beta
        matmul(stream, M, k_col, kv);
        sub(stream, v_col, kv, vec);
        const float bt = read_half((char *) beta.data + t * 2);
        muls(stream, vec, bt, delta);


        // M += delta (x) k, as a 1-column times 1-row matmul (no broadcast rules involved)
        matmul(stream, delta, k_row, tmp);
        add(stream, M, tmp, tmp2);
        assign(stream, tmp2, M);

        // attn = M q / sqrt(S)
        matmul(stream, M, q_col, kv);
        muls(stream, kv, scale, attn_row);
    }

    assign(stream, M, state);
}

} // namespace strata::cann
