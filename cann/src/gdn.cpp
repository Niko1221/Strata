#include "aclnn_call.hpp"
#include "strata/cann/gdn.hpp"

#include <aclnnop/aclnn_recurrent_gated_delta_rule.h>

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

// ---- batched over heads ----
//
// One token at a time, all H heads in one call each. The layouts are chosen so that every operand
// is a plain view of the caller's buffers:
//   q/k/v/attn: [T][H*S]   -> per token a [H][S] slice, seen as [H][S][1] or [H][1][S]
//   decay/beta: [T][H]     -> per token an [H] slice, seen as [H][1][1]
//   state:      [H][S][S]  -> the recurrent matrix per head, rows contiguous
void gdn_batched(Stream & stream, const Tensor & q, const Tensor & k, const Tensor & v, const Tensor & decay,
                 const Tensor & beta, Tensor & state, Tensor & attn, int64_t S, int64_t H, int64_t T) {
    const size_t hvec_bytes = (size_t) H * (size_t) S * 2;
    const size_t hmat_bytes = (size_t) H * (size_t) S * (size_t) S * 2;
    const size_t hscalar_bytes = (size_t) H * 2;

    // M is the working state, tmp receives M after the decay, outer the rank-1 update
    Buffer m_b(hmat_bytes), tmp_b(hmat_bytes), outer_b(hmat_bytes);
    Buffer kv_b(hvec_bytes), d_b(hvec_bytes), delta_b(hvec_bytes), decay_b(hscalar_bytes);
    Buffer beta_b(hscalar_bytes), scale_b(hscalar_bytes);

    Tensor M{ state.data, DType::F16, S, S, H };
    Tensor tmp{ tmp_b.data(), DType::F16, S, S, H };
    Tensor outer{ outer_b.data(), DType::F16, S, S, H };
    Tensor kv{ kv_b.data(), DType::F16, 1, S, H };     // [H][S][1]
    Tensor d{ d_b.data(), DType::F16, 1, S, H };
    Tensor delta{ delta_b.data(), DType::F16, 1, S, H };
    Tensor decay_h{ decay_b.data(), DType::F16, 1, 1, H };
    Tensor beta_h{ beta_b.data(), DType::F16, 1, 1, H };
    Tensor scale_h{ scale_b.data(), DType::F16, 1, 1, H };

    // the scale is constant, so one upload serves every token
    {
        const float s = 1.0f / std::sqrt((float) S);
        std::vector<float> ones((size_t) H, s);
        std::vector<uint16_t> h((size_t) H);
        f32_to_f16(ones.data(), h.data(), h.size());
        scale_b.copy_from_host(h.data(), (size_t) H * 2);
    }

    for (int64_t t = 0; t < T; ++t) {
        const size_t base = (size_t) t * hvec_bytes;
        Tensor q_t{ (char *) q.data + base, DType::F16, 1, S, H };    // [H][S][1]
        Tensor k_t{ (char *) k.data + base, DType::F16, 1, S, H };
        Tensor k_row{ (char *) k.data + base, DType::F16, S, 1, H };  // [H][1][S]
        Tensor v_t{ (char *) v.data + base, DType::F16, 1, S, H };
        Tensor a_t{ (char *) attn.data + base, DType::F16, 1, S, H };
        Tensor dec_t{ (char *) decay.data + (size_t) t * hscalar_bytes, DType::F16, 1, 1, H };
        Tensor bet_t{ (char *) beta.data + (size_t) t * hscalar_bytes, DType::F16, 1, 1, H };

        // M *= decay
        mul_bcast(stream, M, dec_t, tmp);

        // d = (v - M k) * beta, with the decayed state (tmp), not the pre-decay one
        bmm(stream, tmp, k_t, kv);
        sub_bcast(stream, v_t, kv, d);
        mul_bcast(stream, d, bet_t, delta);

        // M += delta (x) k
        bmm(stream, delta, k_row, outer);
        add_bcast(stream, tmp, outer, M);

        // o = (M q) / sqrt(S)
        bmm(stream, M, q_t, kv);
        mul_bcast(stream, kv, scale_h, a_t);
        (void) a_t;
    }
    (void) scale_h;
}

// The same recurrence in fp32. llama.cpp's CPU kernel is float, so an fp16 accumulator is the main
// source of drift; here only the caller's q/k/v and the output stay fp16, and the state is float.
void gdn_batched_f32(Stream & stream, const Tensor & q, const Tensor & k, const Tensor & v, const Tensor & decay,
                     const Tensor & beta, Tensor & state, Tensor & attn, int64_t S, int64_t H, int64_t T) {
    const size_t hvec_bytes = (size_t) H * (size_t) S * 4;
    const size_t hmat_bytes = (size_t) H * (size_t) S * (size_t) S * 4;
    const size_t hscalar_bytes = (size_t) H * 4;

    // the whole sequence is cast once, not per token
    Buffer qf((size_t) T * hvec_bytes), kf((size_t) T * hvec_bytes), vf((size_t) T * hvec_bytes);
    Buffer af((size_t) T * hvec_bytes);
    Buffer tmp_b(hmat_bytes), outer_b(hmat_bytes);
    Buffer kv_b(hvec_bytes), d_b(hvec_bytes), delta_b(hvec_bytes);

    {
        Tensor tq{ qf.data(), DType::F32, H * S, T };
        Tensor tk{ kf.data(), DType::F32, H * S, T };
        Tensor tv{ vf.data(), DType::F32, H * S, T };
        cast(stream, q, DType::F32, tq);
        cast(stream, k, DType::F32, tk);
        cast(stream, v, DType::F32, tv);
    }

    Tensor M{ state.data, DType::F32, S, S, H };
    Tensor tmp{ tmp_b.data(), DType::F32, S, S, H };
    Tensor outer{ outer_b.data(), DType::F32, S, S, H };
    Tensor kv{ kv_b.data(), DType::F32, 1, S, H };
    Tensor d{ d_b.data(), DType::F32, 1, S, H };
    Tensor delta{ delta_b.data(), DType::F32, 1, S, H };

    Buffer scale_b(hscalar_bytes);
    {
        std::vector<float> s((size_t) H, 1.0f / std::sqrt((float) S));
        scale_b.copy_from_host(s.data(), (size_t) H * 4);
    }
    Tensor scale_h{ scale_b.data(), DType::F32, 1, 1, H };

    for (int64_t t = 0; t < T; ++t) {
        const size_t base = (size_t) t * hvec_bytes;
        Tensor q_t{ (char *) qf.data() + base, DType::F32, 1, S, H };
        Tensor k_t{ (char *) kf.data() + base, DType::F32, 1, S, H };
        Tensor k_row{ (char *) kf.data() + base, DType::F32, S, 1, H };
        Tensor v_t{ (char *) vf.data() + base, DType::F32, 1, S, H };
        Tensor a_t{ (char *) af.data() + base, DType::F32, 1, S, H };
        Tensor dec_t{ (char *) decay.data + (size_t) t * hscalar_bytes, DType::F32, 1, 1, H };
        Tensor bet_t{ (char *) beta.data + (size_t) t * hscalar_bytes, DType::F32, 1, 1, H };

        mul_bcast(stream, M, dec_t, tmp);
        bmm(stream, tmp, k_t, kv);
        sub_bcast(stream, v_t, kv, d);
        mul_bcast(stream, d, bet_t, delta);
        bmm(stream, delta, k_row, outer);
        add_bcast(stream, tmp, outer, M);
        bmm(stream, M, q_t, kv);
        mul_bcast(stream, kv, scale_h, a_t);
    }

    Tensor taf{ af.data(), DType::F32, H * S, T };
    cast(stream, taf, DType::F16, attn);
}

// ---- CANN's fused operator ----
//
// Everything is bfloat16 except the gates (fp32) and the three int32 side inputs. The state is
// updated in place, so a caller that keeps its state in this buffer pays nothing per token beyond
// the three casts around the call.
void gdn_fused(Stream & stream, const Tensor & q, const Tensor & k, const Tensor & v, const Tensor & beta,
               const Tensor & g, Tensor & state, Tensor & out, int64_t T, int64_t Nk, int64_t Nv, int64_t Dk,
               int64_t Dv) {
    int64_t qshape[3] = { T, Nk, Dk };
    int64_t vshape[3] = { T, Nv, Dv };
    int64_t bshape[2] = { T, Nv };
    int64_t sshape[4] = { 1, Nv, Dv, Dk };

    auto mk = [](void * data, int64_t * shape, int64_t dims, aclDataType dt) {
        int64_t stride[4] = { 1, 1, 1, 1 };
        int64_t acc = 1;
        for (int64_t i = dims - 1; i >= 0; --i) {
            stride[i] = acc;
            acc *= shape[i];
        }
        int64_t storage = acc;
        return aclCreateTensor(shape, (uint64_t) dims, dt, stride, 0, ACL_FORMAT_ND, &storage, 1, data);
    };

    aclTensor * tq = mk(q.data, qshape, 3, ACL_BF16);
    aclTensor * tk = mk(k.data, qshape, 3, ACL_BF16);
    aclTensor * tv = mk(v.data, vshape, 3, ACL_BF16);
    aclTensor * tb = mk(beta.data, bshape, 2, ACL_BF16);
    aclTensor * tg = mk(g.data, bshape, 2, ACL_FLOAT);
    aclTensor * tst = mk(state.data, sshape, 4, ACL_BF16);
    aclTensor * to = mk(out.data, vshape, 3, ACL_BF16);

    // one sequence, one token, no speculative rows. These three are device tensors like the rest:
    // passing host pointers here faults the kernel instead of reporting a parameter error.
    Buffer b_len((size_t) 1 * 4), b_acc((size_t) 1 * 4), b_idx((size_t) T * 4);
    {
        const int32_t len_val = (int32_t) T;
        const int32_t acc_val = 1;
        std::vector<int32_t> idx_host((size_t) T, 0);
        b_len.copy_from_host(&len_val, 4);
        b_acc.copy_from_host(&acc_val, 4);
        b_idx.copy_from_host(idx_host.data(), idx_host.size() * 4);
    }
    int64_t lshape[1] = { 1 };
    int64_t ishape[1] = { T };
    int64_t ashape[1] = { 1 };
    aclTensor * tlen = mk(b_len.data(), lshape, 1, ACL_INT32);
    aclTensor * tidx = mk(b_idx.data(), ishape, 1, ACL_INT32);
    aclTensor * tacc = mk(b_acc.data(), ashape, 1, ACL_INT32);

    const float scale = 1.0f / std::sqrt((float) Dk);
    execute(
        stream,
        [&](uint64_t * ws, aclOpExecutor ** ex) {
            return aclnnRecurrentGatedDeltaRuleGetWorkspaceSize(tq, tk, tv, tb, tst, tlen, tidx, tg, tg, tacc, scale,
                                                               to, ws, ex);
        },
        [&](void * ws, uint64_t ws_size, aclOpExecutor * ex, aclrtStream s) {
            return aclnnRecurrentGatedDeltaRule(ws, ws_size, ex, s);
        },
        "aclnnRecurrentGatedDeltaRule");

    aclDestroyTensor(tq);
    aclDestroyTensor(tk);
    aclDestroyTensor(tv);
    aclDestroyTensor(tb);
    aclDestroyTensor(tg);
    aclDestroyTensor(tst);
    aclDestroyTensor(to);
    aclDestroyTensor(tlen);
    aclDestroyTensor(tidx);
    aclDestroyTensor(tacc);
}

} // namespace strata::cann
