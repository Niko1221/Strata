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
/// n_tok rows (parts [n,k,N], weights [n,k], shared/output [n,N]) in one launch, each as the single call.
/// 143-amdk: native_moe_combine_multi with moe_hit_add folded in: for the rows listed in dst[0, *count) (entry index = token * k + expert,
/// relative to `parts` and `hits`) the expert's part is parts + hits (the add moe_hit_add made in place), then the combine as before.
/// Only the k = 10 vec4 case; `native_moe_combine_hits_supported` says whether a call may use it.
bool native_moe_combine_hits_supported(const float* parts, const float* shared, const float* output, int64_t n_embd, int64_t k, int n_tok);
void native_moe_combine_multi_hits_added(const float* parts, const float* hits, const int32_t* dst, const int32_t* count,
                                         const float* weights, const float* shared, float* output, int64_t n_embd, int64_t k,
                                         int n_tok, void* stream);
void native_moe_combine_multi(const float* parts, const float* weights, const float* shared, float* output,
                              int64_t n_embd, int64_t k, int n_tok, void* stream);
/// STRATA_VERIFY_RESIDENT: the same, reading every part as `0.0f + hit[i]` - bitwise what a zeroed part row plus
/// moe_hit_add (parts += hit) gave, without the zero fill and the add launch.
void native_moe_combine_multi_hits(const float* hits, const float* weights, const float* shared, float* output,
                                   int64_t n_embd, int64_t k, int n_tok, void* stream);
/// STRATA_COMBINE_HITS=1: native_moe_combine_multi with moe_hit_add folded in - rows dst[0, *count) of `parts` read as
/// `parts[i] + hit[i]` (the add moe_hit_add writes back, the same single rounding), so bitwise the two launches it
/// replaces; parts itself is left as it was. k = 10, 16-byte aligned rows only: false (nothing launched) otherwise.
bool native_moe_combine_multi_add_hits(const float* parts, const float* hit, const int32_t* dst, const int32_t* count,
                                       const float* weights, const float* shared, float* output, int64_t n_embd,
                                       int64_t k, int n_tok, void* stream);
/// STRATA_WAIT_COMBINE: wait_copy_rows_from_mapped (the CPU's rows after `flag` reaches `value`) and
/// native_moe_combine_multi_add_hits in one launch: the same output bit for bit.  `cpu_rows` is the mapped rows.
bool native_moe_wait_combine_ok(const float* cpu_rows, const float* hit, const float* shared, const float* output,
                                int64_t n_embd, int64_t k, int n_tok);
void native_moe_wait_combine_multi_add_hits(const uint32_t* flag, uint32_t value, const float* cpu_rows, const float* hit,
                                            const int32_t* dst, const int32_t* count, const float* weights,
                                            const float* shared, float* output, int64_t n_embd, int64_t k, int n_tok,
                                            void* stream);
/// S26 STRATA_LFUSE: the same with `shared` unscaled and shared_gate[t] the shared expert's raw gate logit: the
/// row is scaled by sigmoid(shared_gate[t]) here, bitwise what shared_expert_multi's sigmoid + scale gave.
void native_moe_combine_multi_hits_gated(const float* hits, const float* weights, const float* shared,
                                         const float* shared_gate, float* output, int64_t n_embd, int64_t k, int n_tok,
                                         void* stream);
}
