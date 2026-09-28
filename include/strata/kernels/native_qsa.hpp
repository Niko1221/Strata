#pragma once

namespace strata::kernels {

// Set before session construction/capture; existing CUDA graphs retain their
// selected kernels. Disabled by default. Integration chooses the call sites.
void native_qsa_set_enabled(bool enabled);
bool native_qsa_enabled();

// Pinned single-token CUDA F32 arithmetic from llama.cpp
// 3cf03257f219afbe7334045ff7c6a06ac68c627d. No allocation or synchronization;
// both functions require a nonnull caller-owned stream and four-byte alignment.
// Positive dimensions with n_cols*n_rows (or head_dim*n_head) <= INT_MAX.
// Finite inputs and finite intermediate sums/products are caller preconditions.

// Contiguous input/output [n_cols,n_rows], gamma[n_cols] broadcast over rows.
// Each row has independent RMS statistics. Exact output==input is supported;
// otherwise all three spans must be disjoint. Epsilon must be finite and >= 0;
// epsilon==0 additionally requires a nonzero sum of squares in every row.
// Pinned 256-thread (width<1024) or 1024-thread XOR reduction, (scale*x)*gamma.
void native_qsa_rms_norm_weighted(const float* input, const float* gamma, float* output,
                                  int n_cols, int n_rows, float epsilon, void* stream);

// native_qsa_rms_norm_weighted, then native_rope_apply_tokens (64 rotated columns) on its output, in one launch and
// bitwise both: n_rows rows of n_cols (128 or 256) read at input + r * in_stride, written contiguously to output;
// row r's position is positions[(r / heads) * pos_stride + r % heads].  output == input needs in_stride == n_cols.
void native_qsa_norm_rope_tokens(const float* input, int in_stride, const float* gamma, float* output, int n_cols,
                                 int n_rows, float epsilon, int n_rot, float freq_base, const int* positions, int heads,
                                 int pos_stride, void* stream);

// attn/output [head_dim,n_head]; q_full [2*head_dim,n_head], each row is
// [query channels, gate channels]. Output is attn*sigmoid(second-half gate).
// Exact output==attn is supported; all other spans must be disjoint.
void native_qsa_gate_apply(const float* attn, const float* q_full, float* output,
                           int n_head, int head_dim, void* stream);

} // namespace strata::kernels
