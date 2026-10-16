// include/strata/qsa_decode_attn_multi.hpp - the SYCL port: qsa_decode_attn_batch over several KV pool sets in one
// launch.
//
// A batch verify window (--batch) holds the rows of several slots, and every slot session has its own K/V pools, so
// verify.cpp runs the QSA selection and attention of such a window one row at a time: four launches per row and
// layer.  With STRATA_BATCH_QSA_ATTN=1 (opt-in) it runs the scores once per slot, the top-k once per window and the
// attention of the whole window through this call: rows [rows[0] + ... + rows[k-1], + rows[k]) read pools[k]; q, ids,
// steps, scratch and attn are row-contiguous exactly as for qsa_decode_attn_batch.  The work-group of row z reads the
// pools of its own set and nothing else differs, so every row gets qsa_decode_attn_batch's bits.  One chunk launch
// and one merge launch per four pool sets; pool sets of different KV formats, or STRATA_ATTN_LANECELL=1, take one
// qsa_decode_attn_batch per set instead.
#pragma once

#include "strata/kernels/qsa_decode_attn.hpp"

#include <cstdint>

namespace strata::kernels {

void qsa_decode_attn_multi(const float* q, const QsaAttnPools* pools, const int* rows, int n_pools, const int32_t* ids,
                           const int32_t* steps, int64_t cap, const QsaShapes& s, float* scratch, float* attn,
                           void* stream);

}  // namespace strata::kernels
