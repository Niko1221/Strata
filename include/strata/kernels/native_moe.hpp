#pragma once
#include <cstdint>

namespace strata::kernels {
// Configure before constructing/capturing sessions. Existing captures keep their
// selected kernels; changing this flag does not rewrite an existing graph.
void native_moe_combine_set_enabled(bool enabled);
bool native_moe_combine_enabled();

// Pinned CUDA weighted-reduction contract for one token, k in [1, 15].
// The backend fuses k=2..15; k=1 is its ordinary multiply/contiguous/add path.
// parts: k contiguous F32 rows of n_embd; weights: k F32 values. The first
// product rounds to F32, following products accumulate with FMA in expert order,
// and optional shared is added once afterward. Shared is not router weighted.
// Requires a nonnull ordered stream and disjoint output. No allocation or sync.
void native_moe_combine(const float* parts, const float* weights, const float* shared,
                        float* output, int64_t n_embd, int64_t k, void* stream);

// The verify window's combine for n_tok tokens, bitwise `native_moe_combine` per token over the rows the window
// used to assemble: row t*k + j comes from `gpu_rows` (row-indexed) as the host's zeroed row plus the hit when an
// entry of dst[0..*count) or dst2[0..*count2) names it, as it is when an entry of dst3[0..*count3) does, and
// otherwise from `host_rows` (mapped memory: only these rows cross PCIe).  `dst2` and `dst3` may be null.
// weights (n_tok, k), shared and output (n_tok, n_embd).  Graph-capturable.
void native_moe_gather_combine(const float* gpu_rows, const float* host_rows, const int32_t* dst,
                               const int32_t* count, const int32_t* dst2, const int32_t* count2, const int32_t* dst3,
                               const int32_t* count3, const float* weights, const float* shared, float* output,
                               int64_t n_embd, int64_t k, int n_tok, void* stream);
}
