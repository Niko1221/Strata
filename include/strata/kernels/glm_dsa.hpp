// include/strata/kernels/glm_dsa.hpp - glm5-next's DSA (k-pool indexer) and the sparse attention it selects for.
//
// **THIS IS AN OPT-IN PATH AND THE MODEL'S DEFAULT IS NOT TO USE IT.**  Off, an MLA layer attends every cell up
// to its own position - which is what `glm_mla_attn` does today and what the reference does when `cparams.dsa`
// is false.  On, it attends the `idx_top_k` cells the indexer picks.  The two agree exactly while the cache is
// short enough that everything is selected, so this is a feature above ~2048 cells and not a correctness fix.
//
// The four pieces, per MLA layer, transcribed from upstream llama.cpp's `build_glm5next_dsa_top_k`
// (`src/graphs/build_glm5next.cpp`) and ik_llama.cpp's host fill for it:
//
//   pool      pooled[e, p] = sum_m softmax_m(ig[e, m_p] + ape[e, m]) * ik[e, m_p]
//             **THE SOFTMAX RUNS SEPARATELY PER KEY-DIM ELEMENT.**  The reference permutes (kpool, key_dim) and
//             calls `soft_max` over ne[0], so the normaliser is the kpool members of ONE key dim and not the
//             whole pool row.  One softmax over the row is the natural misreading, it is finite, it is
//             normalised, and it is a different selection.
//             `ape` is a learned per-member additive embedding; pool p's members are the cells
//             p*kpool .. p*kpool+kpool-1.
//   score     score[t, p] = sum_h relu(iq_h . pooled_p) * weights[h, t], with `weights` ALREADY prescaled by
//             `1/sqrt(key_dim * idx_heads)` - the reference divides by that where it builds them, not here.
//   select    pool p is visible to query t only when its LAST member is <= t.  The `top_pools` visible pools
//             with the highest scores are expanded to their kpool cells in descending score order, then the
//             incomplete tail is appended, and the row is padded to `n_sel` with -1.  The rank formula is
//             `glm_router_sigmoid_topk`'s: stable descending, ties to the LOWER index.
//   attend    prob_s = softmax_s((q_abs_h . latent_s) * scale) over the VALID cells (the -1 padding carries
//             -inf, which is the graph's additive mask), ctx = sum_s prob_s * latent_s, which our de-absorption
//             then takes through `attn_v_b` one head at a time.
//
// THE SCALE IS `1/sqrt(qk_nope)` - the per-head width BEFORE absorption, 256 here - and NOT `1/sqrt(kv_lora)`.
// 512 is what the dot products actually run over, so `1/sqrt(512)` is the plausible constant that is wrong.
//
// ---- `select_tail`, WHERE THE TWO ORACLES DISAGREE.
//
// The incomplete tail is the cells after the last complete pool (`n_vis*kpool .. pos`).  At positions 0..kpool-1
// it is the ONLY thing a query can see, so with it off the first kpool-1 tokens of a sequence attend to nothing
// at all.
//
//   * ik_llama.cpp (`build_glm5next.cpp`: `if (r > 1) { ... inp_kpool_tail = ... }`) creates it whenever
//     kpool > 1 and reads NO key for it - unconditionally on.
//   * upstream llama.cpp reads `%s.attention.indexer.kpool_select_tail`, defaulting to FALSE.
//   * The model carries NEITHER key.  `l4.gguf`'s 70 metadata keys were read directly: the indexer keys present
//     are head_count, key_length, kpool, top_k and index_share_mtp, and there is no kpool_select_tail.
//
// So the oracles differ by default, and the ladder oracle is ik.  **We follow ik: `select_tail` is a parameter
// and every caller passes 1**, which is also the only reading under which the first kpool-1 tokens see anything
// the model was trained to see.  It is a parameter and not a constant so the other reading stays reachable and
// testable - `glm_parity` runs both.
//
// ---- A SECOND, SMALLER DIVERGENCE: ik's `n_sel` reserves room for the tail, and we do not.
//
// ik's selection width is a MINIMUM OF THREE terms (`build_glm5next.cpp:151-155`), not our two:
//
//     n_sel = min(n_pool, indexer_top_k / r, (n_kv - tail_cnt) / r)      tail_cnt = tail_cells ? r-1 : 0
//
// and `tail_cells` is non-null whenever `r > 1` (the same unconditional construction as `select_tail` above), so
// on this model the third term is `(n_kv - 3) / 4`.  ik states its purpose in the line above it: "reserve room
// for the tail (r-1 cells) so r*n_sel + (r-1) <= n_kv; dense fallback if the cache is too small".  It is a bound
// on a STATIC graph shape - ik's cell list is a concat of two `ggml` tensors whose sizes are fixed at build time.
//
// We have no such shape to satisfy: `cells` is a row of `n_sel` ints sized once from `top_k/kpool` and the tail,
// and a selection that ran past the cache is not representable-but-wrong here, it simply is shorter.  We select
// `min(top_k/kpool, pools completed)`.  Copying the guard would import a batching artifact into the model's
// attention: under it ik's cells depend on `n_kv`, which is the ubatch's END (`llama-build-context.cpp:66`,
// `n_kv = worst_case ? kv_self.size : kv_self.n`) - so identical tokens at the same position would select
// differently under `-b 64` and `-b 512`.
//
// MEASURED, and the reason this is a note rather than a code change: at `-c 512` with the default `top_k` 2048,
// ik's `--dsa` output is BYTE-IDENTICAL to its own no-`--dsa` output over all 255 scored rows of chunk 0, and
// stays byte-identical at `-b 64` (llama-perplexity on `l4.gguf`, PPL 91142.1138 in every one of the four runs).
// ik's cap there is (512-3)/4 = 127, and no scored row has 128 visible pools - the 128th appears only at the
// cache's final cell, whose row perplexity cannot score because it has no next token.  So the term binds, if it
// ever does, at exactly one position per full cache, and above ~2051 cells `top_k/kpool` binds for both engines
// and the guard is inert.
//
// ---- LAYOUT CONTRACTS.  All f32, ggml order (features fastest); the token is the outermost axis.
//
//   ik, ig        [key_dim, n_cells]        e + key_dim*c      the cache is indexed by CELL, one row per token
//   pooled        [key_dim, n_pools]        e + key_dim*p
//   ape           [key_dim, kpool]          e + key_dim*m
//   iq            [key_dim, idx_heads, T]   e + key_dim*h + key_dim*idx_heads*t
//   weights       [idx_heads, T]            h + idx_heads*t
//   score         [n_pools, T]              p + n_pools*t
//   cells         [n_sel, T]                s + n_sel*t, -1 = padding
//   latents       [kv_lora, n_cells]        e + kv_lora*c     (the MLA cache, fp16 in the engine)
//   q_abs, out    [kv_lora, n_head, T]      e + kv_lora*h + kv_lora*n_head*t
//
// `ik`/`ig` are indexed by the cell the token occupies and so are the MLA latents: this engine's cache is
// indexed by ABSOLUTE position and never rotates, so cell == position and `pos[t]` in `glm_dsa_select` is just
// the token's absolute position.  A rotating cache would need a cell map here, which is exactly the `cell_of_pos`
// the reference's host fill carries.
//
// **`glm_dsa_pool` READS `ik`/`ig` FROM POOL 0**, so pooling a range that does not start at 0 must offset both
// pointers by `lo * kpool * key_dim` and `pooled` by `lo * key_dim`.
//
// **AND THE ENGINE EXPLOITS THAT: IT KEEPS ONLY THE POOL IN PROGRESS.**  A cell's `ik`/`ig` are read exactly once
// - by the pool that contains them - and never again, so the engine stores `kpool` cells of key and gate rather
// than the whole `[key_dim, n_cells]` history the reference's cache holds.  Each token's row is copied into the
// slot `pos % kpool`, and the pool runs on completion with `n_pools = 1`.  At a 16K context that is 4 KB of
// partial state a layer instead of 16 MB of history, and `pooled` (the only part anything reads later) is the
// same size either way.  The arithmetic is identical: `glm_dsa_pool` reads members 0..kpool-1 of one pool in
// both forms, and the only difference is which rows of memory are contiguous.
#pragma once

#include <cstdint>

namespace strata::kernels {

/// **WHETHER THE ENGINE RUNS THE INDEXER AT ALL.**  Off is the reference's own default (`cparams.dsa = false`)
/// and it is what this port did before the indexer existed: `glm_mla_attn` over every cell up to the query.
///
/// It is a process-wide switch set once from the CLI (`--dsa`), not a parameter threaded through the block, for
/// the reason `native_qsa_indexer_set_enabled` is: it reaches `mla_layer` without every layer's signature
/// growing an argument, and it is the same shape as the family's other opt-in kernels.  The session's STATE
/// SIZE does not depend on it (the indexer's carve is taken either way), so nothing that is sized once can
/// disagree with something that is read later.
void glm_dsa_set_enabled(bool enabled);
bool glm_dsa_enabled();

/// The number of pools a query selects from: `idx_top_k / kpool` (2048/4 = 512 here).
inline int glm_dsa_top_pools(int64_t idx_top_k, int64_t idx_kpool) {
    return idx_kpool > 0 ? (int) (idx_top_k / idx_kpool) : 0;
}

/// The width of a `cells` row: kpool cells for each selected pool, plus the tail when `select_tail`.
/// The reference's `n_sel` (`n_sel_max()` in glm_model.hpp), and what `glm_mla_attention`'s shared memory is
/// sized from.
inline int glm_dsa_n_sel(int64_t idx_top_k, int64_t idx_kpool, int select_tail) {
    const int tp = glm_dsa_top_pools(idx_top_k, idx_kpool);
    return (int) (idx_kpool * tp + (select_tail ? idx_kpool - 1 : 0));
}

/// `pooled[e, p]` for `n_pools` COMPLETED pools (pool p's members are cells p*kpool .. p*kpool+kpool-1).
void glm_dsa_pool(const float* ik, const float* ig, const float* ape, int key_dim, int kpool, int n_pools,
                  float* pooled, void* stream);

/// `score[t, p]` over every (query, completed pool) pair.  Visibility is the select kernel's business - this
/// scores all of them, and a pool a query cannot see is scored and then ignored.
void glm_dsa_score(const float* iq, const float* pooled, const float* weights, int key_dim, int idx_heads,
                   int n_tokens, int n_pools, float* score, void* stream);

/// The selection: `cells[s + n_sel*t]`, kpool cells per selected pool then the tail, -1 padding.
/// `pos[t]` is each query's absolute cache cell; `n_pools` is the score row's stride and must be at least
/// every query's visible count.
///
/// `top_pools == 0` is LEGITIMATE - no pool has completed yet - and the kernel still has to run, because with
/// `select_tail` the current token is visible through the tail cells.  Returning early would leave the previous
/// call's cells in place and the attention would read whatever they pointed at.
void glm_dsa_select(const float* score, int n_pools, int kpool, int top_pools, int select_tail, int n_tokens,
                    int n_sel, const int* pos, int* cells, void* stream);

/// The sparse attention, WITHOUT the de-absorption: `out[e, h, t] = sum_s prob_s * latent[cells[s, t]][e]`.
/// `out` has the same shape and layout as `q_abs` (kv_lora wide), which is the pre-de-absorption `kqv` our
/// `attn_v_b` band projection reads.  `latents` is the fp16 MLA cache; `cells` rows are -1 padded.
///
/// `qk_nope` is the per-head width BEFORE absorption (256) - the scale is its `rsqrt`, and passing `kv_lora`
/// here is the one-constant error that changes every attention output and looks entirely reasonable.
void glm_dsa_attn(const float* q_abs, const uint16_t* latents, const int* cells, int kv_lora, int n_head,
                  int qk_nope, int n_tokens, int n_sel, float* out, void* stream);

}  // namespace strata::kernels
