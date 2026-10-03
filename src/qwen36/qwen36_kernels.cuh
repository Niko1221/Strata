// qwen36_kernels.cuh - launchers for the kernels strata-qwen36 needs that Strata does not have (or has for Qwen3.8).
// All activations are F32.  Every launcher is asynchronous on `stream` and never allocates.
#pragma once
#include <cstdint>
#include <cuda_fp16.h>

namespace q36 {

// y[r*n+i] = x[r*n+i] * rsqrt(mean_i(x^2) + eps) * w[i]   (w is used as stored: the GGUF has +1 baked in). y==x allowed.
void rmsnorm(float* y, const float* x, const float* w, int rows, int n, float eps, void* stream);
// y[r] = dot(W[r, :], x) for W row-major [n_out][n_in] in F32 / F16 / BF16 (type: 0, 1, 30).
void gemv_float(int ggml_type, const void* W, const float* x, float* y, int n_in, int n_out, void* stream);
void add_inplace(float* y, const float* x, int n, void* stream);                 // y += x
void silu_mul(float* out, const float* gate, const float* up, int n, void* stream); // out = silu(gate) * up
// logits [n_expert] -> softmax -> top-k -> renormalised weights.  ids/wts have k entries.
void router_topk(const float* logits, int n_expert, int k, int* ids, float* wts, void* stream);
// x += sum_j wts[j] * down[j*n+i] + sigmoid(sgate_logit[0]) * shexp[i]
void moe_combine(float* x, const float* down, const float* wts, const float* shexp, const float* sgate_logit,
                 int k, int n, void* stream);
void dot_f32(float* out, const float* a, const float* b, int n, void* stream);   // out[0] = a . b
// Gated RMSNorm of the GDN: dst = rms_norm(o) * gamma * silu(z), per head row of `cols` (Qwen3.6: silu, not sigmoid).
void gdn_out_norm_silu(float* dst, const float* o, const float* z, const float* gamma, int heads, int cols,
                       float eps, void* stream);

// Attention.  q2 = [n_head][2*hd] (q | gate) straight from attn_q.
// q_out = rope(rmsnorm(q)), gate_out = gate.   pos = absolute position, rot = rotary dims (even), theta = rope base.
void attn_prep_q(float* q_out, float* gate_out, const float* q2, const float* qnorm, int n_head, int hd, int rot,
                 float theta, int pos, float eps, void* stream);
// k = rope(rmsnorm(k)); writes K and V for `pos` into the F16 cache [n_kv][max_ctx][hd].
void attn_prep_kv(__half* kcache, __half* vcache, const float* k, const float* v, const float* knorm, int n_kv, int hd,
                  int rot, float theta, int pos, int max_ctx, float eps, void* stream);
// out[h*hd+d] = softmax(q.K^T * scale) V * sigmoid(gate), over positions 0..n_ctx-1.  hd must be 128 or 256.
void attn_decode(float* out, const float* q, const float* gate, const __half* kcache, const __half* vcache,
                 int n_head, int n_kv, int hd, int n_ctx, int max_ctx, void* stream);


// ------------------------------------------------------------------------------------------------------------------
// Batched variants for block prefill.  Layouts: activations [B][width] row-major (token-major).
// Y[col*n_out + row] = dot(W[row,:], X[col*n_in : (col+1)*n_in]) for W in F32 / F16 / BF16.
void gemv_float_cols(int ggml_type, const void* W, const float* X, float* Y, int n_in, int n_out, int ncols, void* stream);
// F32 activations -> F16 bits, for the tensor-core GEMMs (strata::prefill::Gemm takes F16/BF16 inputs).  The
// conversion is Strata's own round-to-nearest-even `f16_from_f32`, not `__float2half` (see f16_bits.hpp).
void to_f16(const float* x, uint16_t* out, int n, void* stream);
// Like router_topk for `rows` independent rows: logits [rows][n_expert], ids/wts [rows][k].
void router_topk_rows(const float* logits, int rows, int n_expert, int k, int* ids, float* wts, void* stream);
// Causal conv (k=4) + SiLU over B consecutive tokens, same history layout as Strata's native op (hist[c*3 + tap]).
void gdn_conv_silu_b(float* hist, const float* X, const float* w, float* out, int channels, int B, void* stream);
// L2-normalise the first `rows_per_token` rows of S floats of every token (q heads then k heads), exactly as Strata's.
void gdn_l2norm_qk_b(float* qkv, int channels, int rows_per_token, int S, int B, float eps, void* stream);
// gate = softplus(alpha + dt) * ssm_a ; beta = sigmoid(beta)  (in place), for B tokens x hv heads.
void gdn_gate_beta_b(const float* alpha, const float* dt, const float* ssm_a, float* gate, float* beta, int hv, int B,
                     void* stream);
// The delta-rule recurrence over B tokens, same state layout and arithmetic as native_gdn_step (S = 128).
void gdn_step_b(float* state, const float* qkv, const float* gate, const float* beta, float* out, int hk, int hv,
                int channels, int B, void* stream);
void gather_rows(float* dst, const float* src, const int* idx, int n_rows, int n, void* stream);   // dst[r] = src[idx[r]]
// x[b] += sum_j wts[b*k+j] * D[pos[b*k+j]] + sigmoid(sg[b]) * shexp[b]       (rows of n floats)
void moe_combine_b(float* x, const float* D, const float* wts, const int* pos, const float* shexp, const float* sg, int B,
                   int k, int n, void* stream);
void dot_rows(float* out, const float* w, const float* X, int rows, int n, void* stream);          // out[r] = w . X[r]
void attn_prep_q_b(float* q_out, float* gate_out, const float* q2, const float* qnorm, int n_head, int hd, int rot,
                   float theta, int pos0, int B, float eps, void* stream);
void attn_prep_kv_b(__half* kcache, __half* vcache, const float* k, const float* v, const float* knorm, int n_kv, int hd,
                    int rot, float theta, int pos0, int B, int max_ctx, float eps, void* stream);
// Token b attends to positions 0 .. pos0+b (K/V of the whole batch must already be in the cache).
void attn_decode_b(float* out, const float* q, const float* gate, const __half* kcache, const __half* vcache, int n_head,
                   int n_kv, int hd, int pos0, int B, int max_ctx, void* stream);

}  // namespace q36
