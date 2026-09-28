// include/strata/kernels/qsa_select.hpp - plan v0.3 P5/P7: the QSA indexer's block scores and top-k selection,
// for many queries at once and at long context.
//
// The per-token path scores every pooled block with one FP64 block per row (qsa_index_kernel) and selects with a
// single-block kernel that makes 32 bit-serial passes over every CELL (topk_kernel): ~0.1-0.2 ms per query at 32K,
// once per QSA layer and token - 1-2 ms of a decode token and most of a 32K prompt's time.  Here:
//
//   qsa_block_scores : FP32; relu per indexer head, summed; block n_bid (the incomplete tail) scores the `dead` key
//                      and gets +1e9 when it has cells - which is exactly what the per-token path reads there,
//                      because pooled[n_bid] holds `dead` at that time (in a chunk it may already hold a block
//                      completed later, so it is not read).  Each score sums in the order one warp per
//                      (query, block) did.  Decode windows (< 12 queries): an 8-lane group per key block reads the
//                      key once for all queries.  Batches: a per-thread GEMM form, a lane's 2 queries x 4 key
//                      blocks over all heads, the tree on a stack of partial sums, the queries read by the whole
//                      warp at once; one persistent block an SM with an equal share of the (key tile, query chunk)
//                      units.
//   qsa_block_topk   : one 1024-thread block per query; a 4-pass radix select over the query's n_bid + 1 blocks,
//                      each weighted by its cell count, then the cells emitted in ascending order with ties to the
//                      lowest index - the same selection as topk_kernel, over a quarter of the elements.
//
// Queries carry their own step record (pos, n_kv, n_bid, width) as everywhere else in QSA.
#pragma once

#include "strata/kernels/qsa.hpp"

#include <cstdint>

namespace strata::kernels {

/// scores [nq, max_blocks]; q_idx [nq, idx_n_head, idx_dim] (normed and rotated); steps [nq, kStepCount].
/// The grid covers the first `grid_blocks` blocks and goes round for the rest: a captured graph passes a fixed size
/// (qsa_score_grid_blocks), the prompt path the batch's largest n_bid + 1.
void qsa_block_scores(const float* pooled, const float* dead, const float* q_idx, const int32_t* steps, int64_t nq,
                      int64_t max_blocks, const QsaShapes& s, float* scores, void* stream, int64_t grid_blocks);

/// The block-score grid of the decode paths, whatever the context: 164 thread blocks of 64 key blocks, one wave on
/// the RTX 3090 (two per SM); longer contexts go round.
constexpr int64_t qsa_score_grid_blocks = 10496;

/// ids [nq, cap] (cells, ascending); `cap` >= the largest selection width.
void qsa_block_topk(const float* scores, const int32_t* steps, int64_t nq, int64_t max_blocks, int64_t cap,
                    const QsaShapes& s, int32_t* ids, void* stream);

}  // namespace strata::kernels
