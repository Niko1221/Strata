// include/strata/kernels/verify_kernels.hpp - plan v0.3 P6: kernels for the speculative VERIFY window, where T
// tokens (the last accepted token and T-1 drafts) go through a layer at once.
//
// The GDN recurrence is the one piece of state that cannot simply be overwritten by the next window, so its
// kernels come in two halves:
//
//   * the verify half reads the conv history and the recurrent state, runs all T tokens in order (bitwise the
//     single-token kernels' arithmetic per token) and writes ONLY the per-token outputs - the state is left
//     untouched;
//   * the commit half, launched once acceptance is known, replays the first `n_keep` tokens from the inputs the
//     verify half stored and writes the state.  Rejected tokens therefore never touch the state and there is no
//     snapshot of the 113 MB recurrent state to keep.
//
// Everything that varies per window (token ids, `n_keep`) is read from DEVICE memory so the kernels can be captured.
#pragma once

#include <cstdint>

namespace strata::kernels {

inline constexpr int kVerifyMaxT = 8;

/// For token t of T: conv over [history(3) | qkv_0 .. qkv_t] -> SiLU -> L2 norm of the q/k heads -> h[t].
/// `history` is NOT written, but with `commit` (one token): it then keeps the token, as gdn_conv_commit.  Bitwise
/// `fused_gdn_conv_l2` per token.  Up to 65535 tokens (the prompt path runs a whole chunk through it).
void gdn_conv_l2_multi(const float* history, const float* qkv, const float* conv_w, float* h, int channels,
                       int qk_heads, float eps, int n_tok, void* stream, int t_begin = 0, bool commit = false);
/// history <- the last 3 entries of [history | qkv_0 .. qkv_{n-1}], n = *n_keep (0 leaves it as it was).
void gdn_conv_commit(float* history, const float* qkv, int channels, const int32_t* n_keep, void* stream);
/// alpha/beta for T columns of x (T, n_embd): gate (T, h_v), beta (T, h_v).  Bitwise `fused_gdn_ab` per column.
void gdn_ab_multi(const float* x, const uint16_t* w_alpha, const uint16_t* w_beta, const float* dt, const float* ssm_a,
                  float* gate, float* beta, int n_embd, int h_v, int n_tok, void* stream);
/// The recurrence + output norm for T tokens (h = (T, conv_channels) as q|k|v, gate/beta (T, h_v), z/y
/// (T, value_dim)).  With `n_keep == nullptr` the state is read and NOT written (verify); otherwise the first
/// *n_keep tokens are run and the state is written (commit; `y` may be scratch).  Bitwise `fused_gdn_step_norm`.
void gdn_step_norm_multi(float* state, const float* h, int conv_channels, const float* gate, const float* beta,
                         const float* z, const float* gamma, float eps, float* y, int h_k, int h_v, int n_tok,
                         const int32_t* n_keep, void* stream, int t_out_begin = 0);
/// Spin until *flag >= value (a mapped host flag).  The value is fixed at capture, so several rings can be
/// outstanding at once (the split verify window keeps two).
void wait_flag_ge(const uint32_t* flag, uint32_t value, void* stream);
/// *dst = the GPU's global nanosecond timer when the stream reaches this point (`--window-profile`).
void gpu_stamp(unsigned long long* dst, void* stream);

/// Rows of the S2/S4/S8 embedding for T token ids read from DEVICE memory; out (T, n).  Bitwise `embedding_gather`.
void embedding_gather_dev(const uint8_t* codes, const float* scales, const float* offsets, const int32_t* tokens,
                          int n_tok, int64_t n, int code_bits, int code_bias, int group_elems, uint64_t row_codes,
                          uint64_t row_groups, float* out, void* stream);
/// R[t][c][:] = x[t][:] for the hc streams.
void broadcast_streams(const float* x, float* R, int64_t n_embd, int hc, int n_tok, void* stream);
/// dst[:n] = src[(*index) * stride + :n]  (index read from device memory; a negative index copies nothing).
void copy_indexed(float* dst, const float* src, int64_t stride, const int32_t* index, int64_t n, void* stream);

/// Plan v0.3 P6: copy *n (device memory) blobs of `blob_bytes` from mapped host memory (src[k], device aliases)
/// into dst + k * blob_bytes with coalesced 16-byte loads - the PCIe share of a layer's missed experts, staged
/// into VRAM before the grouped expert kernel reads them.  Launched for a capacity of `cap` blobs.
void fetch_blobs(const unsigned long long* src, const int32_t* n, uint8_t* dst, int64_t blob_bytes, int cap, void* stream);
/// A list in mapped host memory, [n, pad x3, rows...]: each listed row of `src` (mapped, n_embd floats, a multiple of
/// 4) to the same row of `dst`, and the list to `list_dev` (device, the same layout).  At most `cap` rows; rows
/// outside [0, cap) are not copied.  The second GPU's rows of a verify window's layer, taken while the CPU works.
void fetch_listed_rows(const int32_t* list, const float* src, float* dst, int32_t* list_dev, int cap, int64_t n_embd,
                       void* stream);
/// ptr[k] = base + k * blob_bytes for k < *n (the staged copies `fetch_blobs` made).
void rebase_ptrs(unsigned long long* ptr, const int32_t* n, uint8_t* base, int64_t blob_bytes, void* stream);
/// The main GPU's share of a verify window's layer, decided on the device: the entries of `ids` (n = tokens * k,
/// n <= 128) whose expert its VRAM tier holds (`res[e]` = slot, else < 0; `slot_ptr[slot]` = the slot's address).  One
/// group per distinct resident expert, in the order of their first entries, its entries in routing order; written in
/// the plan block's layout (`Verifier`): counts [groups, entries, 0, 0] | start (cap + 1) | dst (cap) | tok (cap) |
/// ptr (cap u64, at int32 offset `ptr_off`).
void verify_hit_plan(const int32_t* ids, int n, int k, const int32_t* res, int n_expert,
                     const unsigned long long* slot_ptr, int32_t* plan, int64_t cap, int64_t ptr_off, void* stream);

// ---- the MTP draft layer (src/core/mtp.cpp)
/// R[t][c][:] = h[t][c][:] + e[t][:]  (the embedding branch added to every stream).
void add_streams_broadcast(const float* h, const float* e, float* R, int64_t n_embd, int hc, int n_tok, void* stream);
/// Every routed expert is resident (slot = expert id): slot[i] = ids[i], dst[i] = i, *count = n.
void ident_hits(const int32_t* ids, int n, int32_t* slot, int32_t* dst, int32_t* count, void* stream);
/// The draft chain's next input: R_dst[:] = R_src[row], tok_dst[0] = ids[row], out[j] = ids[row], with
/// row = *row_dev (device memory).  `out` may be mapped host memory.
void mtp_select(const float* R_src, int64_t R_stride, const int32_t* ids, const int32_t* row_dev, float* R_dst,
                int32_t* tok_dst, int32_t* out, int j, void* stream, const float* probs = nullptr,
                float* out_p = nullptr);
/// Row *row_dev of a, b and c (a_n, b_n, c_n floats a row) copied to their row 0 (the draft layer's rest runs on
/// row 0).  Graph-capturable: the row is read on the device.
void copy_row_to_first(const int32_t* row_dev, float* a, int64_t a_n, float* b, int64_t b_n, float* c, int64_t c_n,
                       void* stream);
/// dst row i = src row ids[i] (row_bytes each, a multiple of 4), for n rows.
void gather_rows(const uint8_t* src, int64_t row_bytes, const int32_t* ids, int64_t n, uint8_t* dst, void* stream);
/// ids[t] = table[ids[t]] for n entries (a subset index back to a token id).
void map_ids(int32_t* ids, const int32_t* table, int n, void* stream);
/// probs[t] = softmax(logits[t])[ids[t]] for n_rows rows of n_vocab (the probability of each row's argmax).
void row_top_prob(const float* logits, int n_rows, int n_vocab, const int32_t* ids, float* probs, void* stream);
/// out[t] = the argmax of row t (n floats a row; ties to the lowest index, NaN never picked, 0 when no value is above
/// -inf): bitwise sample_tokens' greedy pick.  Up to 128 blocks a row scan slices of it, the row's last block merges
/// their picks.  `scratch`: argmax_rows_scratch_bytes(n_rows), zero before the first launch (each launch leaves it
/// so).  Graph-capturable.
/// With `out_val` each pick's logit too (-inf when no value is above it).
uint64_t argmax_rows_scratch_bytes(int n_rows);
void argmax_rows(const float* logits, int n_rows, int n, void* scratch, int32_t* out, void* stream,
                 float* out_val = nullptr);
/// *flag += 1 once the stream's earlier work is visible to the host (mapped memory).  Graph-capturable.
void mapped_bump(uint32_t* flag, void* stream);
/// dst row t (n floats) = a row t (na floats) then b row t (n - na floats), for `rows` rows.
void join_rows(float* dst, int64_t n, const float* a, int64_t na, const float* b, int rows, void* stream);
/// row_top_prob over 8 blocks a row, each computing 4 of its 32 warps' sums, the row's last block adding the 32 in
/// order: bitwise row_top_prob.  `scratch`: row_top_prob_scratch_bytes(n_rows), zero before the first launch.
uint64_t row_top_prob_scratch_bytes(int n_rows);
void row_top_prob_split(const float* logits, int n_rows, int n_vocab, const int32_t* ids, float* probs, void* scratch,
                        void* stream);
/// Dense-attention step records for `n` cells: [cell, cell+1, (cell+1)/4, cell+1] from cells[i] (device memory).
void dense_steps(const int32_t* cells, int n, int32_t* steps, void* stream);
/// A sliding attention window: for `n` step records (kStepCount ints each, n_kv at [1]) the selection becomes the
/// last `window` cells: ids[q * ids_stride + j] = max(0, n_kv - window) + j and the record's width = the count.
void window_ids(int32_t* steps, int n, int window, int32_t* ids, int64_t ids_stride, void* stream);

}  // namespace strata::kernels
