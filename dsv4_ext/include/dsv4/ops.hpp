// dsv4/ops.hpp - CPU reference math of DeepSeek-V4, transcribed from the official inference/model.py.
// Float32 reference semantics; these are the oracles the CUDA kernels must match.
#pragma once
#include "dsv4/dequant.hpp"  // ggml types + row_bytes()/row_dot()/dequant_row()/dequant_supported()

#include <cstddef>
#include <cstdint>
#include <vector>

namespace dsv4 {
/// y = x * rsqrt(mean(x^2)+eps) * w   (w may be nullptr: no learned weight)
void rmsnorm(const float* x, const float* w, float eps, int n, float* y);

/// YaRN rope table, model.py precompute_freqs_cis. orig_len == 0 disables YaRN (sliding-window-only layers
/// use base rope_theta without YaRN; compressed layers use compress_rope_theta WITH YaRN).
/// cos/sin are [seqlen][dim/2].
void yarn_table(int dim, int seqlen, int orig_len, double base, double factor, double beta_fast, double beta_slow,
                std::vector<float>& cos_t, std::vector<float>& sin_t);

/// In-place rotary on the LAST `rope_dim` elements of a head vector of length n (pairs (2i, 2i+1) = complex).
/// inverse = true de-rotates (used on the attention output).
void rotary(float* x, int n, int rope_dim, const float* cos_row, const float* sin_row, bool inverse);

enum class Score { Softmax, Sigmoid, SqrtSoftplus };

/// model.py Gate.forward for ONE token.  bias (nullable) only shifts the selection, never the weights.
/// hash_idx (nullable): predetermined experts for this token (first n_hash_layers) - replaces the top-k.
/// Outputs idx[topk] (descending score, ties -> lower index) and w[topk] (normalised unless softmax, times scale).
void gate_route(const float* logits, int n_expert, int topk, Score fn, const float* bias, const int32_t* hash_idx,
                float route_scale, int32_t* idx, float* w);

/// out = silu(min(gate, limit)) * clamp(up, -limit, limit); limit <= 0 disables the clamp.
void swiglu_clamped(const float* gate, const float* up, int n, float limit, float* out);

/// model.py get_window_topk_idxs / get_compress_topk_idxs (bsz = 1). Row-major [rows][cols], -1 = masked.
void window_topk(int win, int seqlen, int start_pos, std::vector<int>& out, int& rows, int& cols);
void compress_topk(int ratio, int seqlen, int start_pos, int offset, std::vector<int>& out, int& rows, int& cols);

/// kernel.py hc_split_sinkhorn for ONE token. mixes[(2+hc)*hc], scale[3], base[(2+hc)*hc] -> pre[hc], post[hc], comb[hc*hc]
/// (comb[j*hc+k]; Sinkhorn: row softmax + eps, then column norm, then (iters-1) x (row norm, column norm)).
void hc_split_sinkhorn(const float* mixes, const float* scale, const float* base, int hc, int iters, float eps,
                       float* pre, float* post, float* comb);

/// model.py Block.hc_pre for ONE token. x[hc*d] (the hc streams, stream-major), fn[(2+hc)*hc][hc*d].
/// y[d] = sum_j pre[j]*x[j]; also returns post[hc], comb[hc*hc] for hc_post.
void hc_pre(const float* x, int hc, int d, const float* fn, const float* scale, const float* base, float norm_eps,
            int iters, float eps, float* y, float* post, float* comb);

/// model.py Block.hc_post: out[k] = post[k]*x + sum_j comb[j*hc+k]*residual[j]   (x[d], residual/out[hc*d]).
void hc_post(const float* x, const float* residual, const float* post, const float* comb, int hc, int d, float* out);

/// model.py ParallelHead.hc_head: pre = sigmoid(mixes*scale+base)+hc_eps ; y = sum_j pre[j]*x[j].  fn[hc][hc*d], scale[1].
void hc_head(const float* x, int hc, int d, const float* fn, const float* scale, const float* base, float norm_eps,
             float hc_eps, float* y);

/// kernel.py sparse_attn for ONE query token: q[h*d], kv[n*d] (K = V, MQA), idxs[topk] (-1 = masked), sink[h].
/// softmax over the gathered scores PLUS the sink logit (the sink adds to the denominator only). o[h*d].
/// A row with no valid index returns zeros (the GPU kernel would produce NaN there).
void sparse_attn_token(const float* q, int h, int d, const float* kv, const int* idxs, int topk, const float* sink,
                       float scale, float* o);

}  // namespace dsv4
