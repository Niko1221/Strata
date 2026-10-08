/**
 * @file ops.hpp
 * @brief CPU reference math of DeepSeek-V4, transcribed from the official inference/model.py.
 *
 * This module implements all the mathematical operations that make up a DeepSeek-V4-Flash forward pass,
 * as float32 reference semantics. These are the oracles that CUDA kernels must match — if ops.hpp and
 * the GPU kernel disagree, there is a bug in the kernel (or in this transcription).
 *
 * @par Operations implemented
 *   - Normalization: RMSNorm (`rmsnorm()`)
 *   - Positional encoding: YaRN RoPE tables + forward/inverse rotary (`yarn_table()`, `rotary()`)
 *   - Expert routing: gate scoring (softmax/sigmoid/sqrtsoftplus), top-k selection, hash experts (`gate_route()`)
 *   - Activation: clamped SwiGLU (`swiglu_clamped()`)
 *   - Attention indexing: windowed and compressed top-k (`window_topk()`, `compress_topk()`)
 *   - Hyper-connections: Sinkhorn normalization, pre/post mixing, head combination (`hc_split_sinkhorn()`,
 *     `hc_pre()`, `hc_post()`, `hc_head()`)
 *   - Sparse attention: gather-top-k + softmax over indexed positions + sink logit (`sparse_attn_token()`)
 *
 * @par Parallelism
 *   All loops that operate over independent elements (heads, streams, tokens) are annotated with
 *   `#pragma omp parallel for` for OpenMP auto-parallelization. The arithmetic is deterministic:
 *   no reduction order dependency means the pragma changes speed but not correctness.
 */
#pragma once

#include "dsv4/dequant.hpp"  // ggml types + row_bytes()/row_dot()/dequant_row()/dequant_supported()

#include <cstddef>
#include <cstdint>
#include <vector>

namespace dsv4 {

/**
 * @brief RMSNorm: Root Mean Square Layer Normalization with learned weight.
 *
 * Computes y[i] = x[i] * rsqrt(mean(x^2) + eps) * w[i], where w is an optional learned weight vector.
 * If w is nullptr, acts as plain RMSNorm (no weight scaling).
 *
 * @param x  Input vector [n].
 * @param w  Learned weight vector [n] (nullable: no weight if null).
 * @param eps  Epsilon for numerical stability (from config: attention.layer_norm_rms_epsilon).
 * @param n  Number of elements.
 * @param y  Output vector [n], normalized values.
 */
void rmsnorm(const float* x, const float* w, float eps, int n, float* y);

/**
 * @brief Precompute YaRN RoPE (Rotary Positional Embedding) frequency tables.
 *
 * Generates cos/sin lookup tables for rotary position encoding with YaRN context extension.
 * orig_len == 0 disables YaRN (sliding-window-only layers use base rope_theta without YaRN;
 * compressed layers use compress_rope_theta WITH YaRN).
 *
 * @param dim        Model embedding dimension (must be even).
 * @param seqlen     Maximum sequence length for the table.
 * @param orig_len   Original context length during training (0 = disable YaRN).
 * @param base       RoPE base frequency (rope.freq_base from config).
 * @param factor     YaRN scaling factor (rope.scaling.factor from config).
 * @param beta_fast  YaRN low-frequency bound for interpolation.
 * @param beta_slow  YaRN high-frequency bound for interpolation.
 * @param cos_t      Output: [seqlen × dim/2] cosine table.
 * @param sin_t      Output: [seqlen × dim/2] sine table.
 */
void yarn_table(int dim, int seqlen, int orig_len, double base, double factor, double beta_fast, double beta_slow,
                std::vector<float>& cos_t, std::vector<float>& sin_t);

/**
 * @brief In-place rotary position encoding on the last `rope_dim` elements of a head vector.
 *
 * Applies 2D rotation to pairs (2i, 2i+1) using precomputed cos/sin rows. Operates in-place on x.
 * Used for both forward (position encoding during prefill/decode) and inverse (de-rotating attention output).
 *
 * @param x         Input/output vector [n], modified in place on its last rope_dim elements.
 * @param n         Total vector length (rope_dim <= n).
 * @param rope_dim  Number of elements to rotate (must be even, from config: rope.dimension_count).
 * @param cos_row   Cosine lookup row [rope_dim/2].
 * @param sin_row   Sine lookup row [rope_dim/2].
 * @param inverse   true = de-rotate (apply negative rotation), false = rotate.
 */
void rotary(float* x, int n, int rope_dim, const float* cos_row, const float* sin_row, bool inverse);

/**
 * @enum class Score
 * @brief Gating function types for expert routing score computation.
 *
 * Maps to the `expert_gating_func` GGUF key (id 4 = SqrtSoftplus in HF's implementation).
 */
enum class Score { Softmax, Sigmoid, SqrtSoftplus };

/**
 * @brief Expert gate routing: select top-k experts for one token.
 *
 * Implements model.py Gate.forward for a single token. Computes scores using the specified gating
 * function (softmax/sigmoid/sqrtsoftplus), adds optional bias, and selects the top-k experts by score.
 *
 * @par Hash experts
 *   If `hash_idx` is provided, it overrides the top-k selection for the first n_hash_layers: those
 *   layers use predetermined hash-based expert assignment instead of scoring.
 *
 * @param logits       Expert routing logits [n_expert] from the gate projection.
 * @param n_expert     Total number of experts per layer.
 * @param topk         Number of experts to select (n_expert_used from config).
 * @param fn           Gating function: softmax, sigmoid, or sqrtsoftplus.
 * @param bias         Optional bias vector [n_expert] that shifts scores (never changes weights).
 * @param hash_idx     Optional predetermined expert indices [topk]; replaces top-k if not null.
 * @param route_scale  Multiplicative scale applied to output weights (from config: expert_weights_scale).
 * @param idx          Output: selected expert indices [topk], sorted by descending score (ties → lower index).
 * @param w            Output: normalized routing weights [topk] (times scale; not normalized for softmax).
 */
void gate_route(const float* logits, int n_expert, int topk, Score fn, const float* bias, const int32_t* hash_idx,
                float route_scale, int32_t* idx, float* w);

/**
 * @brief SwiGLU with clamping: silu(min(gate, limit)) * clamp(up, -limit, limit).
 *
 * Implements the gated activation function used in DeepSeek-V4's MoE feed-forward layers.
 * Clamping is controlled per-layer by `swiglu_clamp_exp` and `swiglu_clamp_shexp` from config.
 * If limit <= 0, clamping is disabled (plain SwiGLU).
 *
 * @param gate     Gate input vector [n] (from gate projection).
 * @param up       Up-projected input vector [n] (from up projection).
 * @param n        Vector length (expert FFN width: expert_feed_forward_length from config).
 * @param limit    Clamp threshold (>0 enables clamping, <=0 disables).
 * @param out      Output vector [n], computed as silu(min(gate,limit)) * clamp(up,-limit,limit).
 */
void swiglu_clamped(const float* gate, const float* up, int n, float limit, float* out);

/**
 * @brief Window-based top-k index generation for sliding window attention.
 *
 * Implements model.py get_window_topk_idxs / get_compress_topk_idxs (bsz = 1). Generates a row-major
 * index matrix where -1 indicates masked positions (future tokens not yet seen).
 *
 * @param win        Sliding window size (from config: attention.sliding_window).
 * @param seqlen     Current sequence length.
 * @param start_pos  Position in the sequence (for ring buffer offset).
 * @param out        Output: index matrix [rows × cols], -1 = masked.
 * @param rows       Output: number of rows in the index matrix.
 * @param cols       Output: number of columns per row (= win).
 */
void window_topk(int win, int seqlen, int start_pos, std::vector<int>& out, int& rows, int& cols);

/**
 * @brief Compressed top-k index generation for compressed attention layers.
 *
 * Generates indices for layers with compression ratio > 1. Each compressed key-value pair
 * represents multiple original positions, reducing the attention computation.
 *
 * @param ratio      Compression ratio (from config: attention.compress_ratios[layer]).
 * @param seqlen     Current sequence length.
 * @param start_pos  Position in the sequence.
 * @param offset     Base offset for indices (for MTP layers).
 * @param out        Output: index matrix [rows × cols], -1 = masked.
 * @param rows       Output: number of rows.
 * @param cols       Output: number of columns.
 */
void compress_topk(int ratio, int seqlen, int start_pos, int offset, std::vector<int>& out, int& rows, int& cols);

/**
 * @brief Sinkhorn normalization for hyper-connection mixing coefficients.
 *
 * Implements kernel.py hc_split_sinkhorn for one token. Computes pre (pre-mixing), post (post-mixing),
 * and comb (combination matrix) from the learned mix/scale/base parameters using Sinkhorn iteration:
 * row softmax + eps, column norm + eps, repeated `iters` times.
 *
 * @param mixes      Learned mixing coefficients [(2+hc)*hc].
 * @param scale      Scale factors [3]: one each for pre, post, comb.
 * @param base       Base offsets [(2+hc)*hc], added before sigmoid/softmax.
 * @param hc         Number of hyper-connection streams (from config: hyper_connection.count).
 * @param iters      Sinkhorn iteration count (from config: hyper_connection.sinkhorn_iterations).
 * @param eps        Epsilon for numerical stability in softmax/norm (from config: hyper_connection.epsilon).
 * @param pre        Output: pre-mixing coefficients [hc].
 * @param post       Output: post-mixing coefficients [hc].
 * @param comb       Output: combination matrix [hc × hc], row-major.
 */
void hc_split_sinkhorn(const float* mixes, const float* scale, const float* base, int hc, int iters, float eps,
                       float* pre, float* post, float* comb);

/**
 * @brief Hyper-connection pre-mixing: combine hc streams into one.
 *
 * Implements model.py Block.hc_pre for one token. Takes hc independent streams of dimension d,
 * applies learned mixing via Sinkhorn-normalized coefficients, and produces a single output stream.
 * Also returns post and comb for the subsequent hc_post step.
 *
 * @par Parallelism
 *   hc=4, d=4096: 24 dot products over 16384 doubles, twice per layer, 43 layers. Every output row
 *   is independent — OpenMP parallelization changes speed but not arithmetic order.
 *
 * @param x        Input: hc × d streams (stream-major layout).
 * @param hc       Number of hyper-connection streams.
 * @param d        Dimension per stream.
 * @param fn       Learned mixing functions [(2+hc)*hc × hc*d].
 * @param scale    Scale factors [3] for pre/post/comb computation.
 * @param base     Base offsets [(2+hc)*hc], added before sigmoid.
 * @param norm_eps  RMSNorm epsilon for input normalization.
 * @param iters    Sinkhorn iteration count.
 * @param eps      Epsilon for Sinkhorn stability.
 * @param y        Output: single stream [d].
 * @param post     Output: post-mixing coefficients [hc] (for hc_post).
 * @param comb     Output: combination matrix [hc × hc] (for hc_post).
 */
void hc_pre(const float* x, int hc, int d, const float* fn, const float* scale, const float* base, float norm_eps,
            int iters, float eps, float* y, float* post, float* comb);

/**
 * @brief Hyper-connection post-mixing: combine streams with residual.
 *
 * Implements model.py Block.hc_post: out[k] = post[k]*x + sum_j(comb[j*hc+k]*residual[j])
 * for each stream k. Applies per-stream scaling and cross-stream combination to produce hc output streams.
 *
 * @param x         Input: single stream [d] (from hc_pre output).
 * @param residual  Residual input: hc × d streams (stream-major layout).
 * @param post      Post-mixing coefficients [hc] (from hc_pre or hc_split_sinkhorn).
 * @param comb      Combination matrix [hc × hc], row-major (from hc_pre or hc_split_sinkhorn).
 * @param hc        Number of hyper-connection streams.
 * @param d         Dimension per stream.
 * @param out       Output: hc × d streams (stream-major layout).
 */
void hc_post(const float* x, const float* residual, const float* post, const float* comb, int hc, int d, float* out);

/**
 * @brief Hyper-connection head combination for ParallelHead layers.
 *
 * Implements model.py ParallelHead.hc_head: computes pre-mixing coefficients via sigmoid(mixes*scale+base)+hc_eps,
 * then combines hc streams into a single output using these coefficients.
 *
 * @param x         Input: hc × d streams (stream-major layout).
 * @param hc        Number of hyper-connection streams.
 * @param d         Dimension per stream.
 * @param fn        Learned mixing functions [hc × hc*d].
 * @param scale     Scale factor [1] (scalar applied to all mixings).
 * @param base      Base offsets [hc], added before sigmoid.
 * @param norm_eps  RMSNorm epsilon for input normalization.
 * @param hc_eps    Epsilon added to pre-mixing coefficients after sigmoid.
 * @param y         Output: single stream [d].
 */
void hc_head(const float* x, int hc, int d, const float* fn, const float* scale, const float* base, float norm_eps,
             float hc_eps, float* y);

/**
 * @brief Sparse attention for one query token using indexed key-value pairs.
 *
 * Implements kernel.py sparse_attn for one query: gathers top-k key-value pairs via pre-computed indices,
 * computes softmax over gathered scores PLUS a sink logit (the sink adds to the denominator only,
 * providing a baseline for long-range attention). A row with no valid index returns zeros.
 *
 * @par Complexity
 *   The most expensive op per token: n_head(64) × topk(128+) × head_dim(512), walked twice in double precision.
 *   Heads are fully independent (own q row, own output row), so parallelization over heads is safe.
 *
 * @param q        Query vector [h × d] (one per head).
 * @param h        Number of attention heads.
 * @param d        Head dimension.
 * @param kv       Key-value matrix [n_seq × d] (K = V, multi-query attention).
 * @param idxs     Top-k indices [topk]: -1 = masked position.
 * @param topk     Number of indexed positions to gather.
 * @param sink     Sink logit [h]: baseline for each head's softmax denominator.
 * @param scale    Attention scaling factor (1 / sqrt(d)).
 * @param o        Output: attention result [h × d].
 */
void sparse_attn_token(const float* q, int h, int d, const float* kv, const int* idxs, int topk, const float* sink,
                       float scale, float* o);

}  // namespace dsv4
