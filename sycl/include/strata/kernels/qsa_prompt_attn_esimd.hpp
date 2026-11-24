// SYCL port (ARC-W): the QSA prompt attention on the int8 DPAS (ESIMD, Battlemage) - see qsa_prompt_attn_esimd.dp.cpp.
#pragma once
#include "strata/kernels/qsa.hpp"
#include "strata/kernels/qsa_decode_attn.hpp"
#include <cstdint>
namespace strata::kernels {
/// The contract of qsa_prompt_attn_batch (q [n_q, n_head, 256], ids [n_q, cap], steps, attn [n_q, n_head, 256]) for int8 KV.
/// Returns false (nothing launched) when the device, shape or pools are not handled, so the caller keeps its fallback.
/// On by default on Battlemage; STRATA_PROMPT_ATTN_ESIMD=0 turns it off.
bool qsa_prompt_attn_esimd(const float* q, const QsaAttnPools& pools, const int32_t* ids, const int32_t* steps, int64_t cap,
                           const QsaShapes& s, float* attn, int64_t n_q, void* stream);

struct QsaSplitAttnLayout {
    const int32_t* page_table = nullptr;
    const int32_t* counts = nullptr;
    const int32_t* position = nullptr;
    uint64_t page_halves = 0;
    int page_rows = 0, kv_heads = 0;
    int card = 0, cards = 0;
    int query_rows = 0, first_query = 0, query_count = 0, selection_capacity = 0;
};

// KV32 pages: [page][token][head][codes], followed by scales within each layer.
// qg has query/gate pairs; output is [head-card][query][local-head][acc256,max_ln,sum].
bool qsa_prompt_attn_esimd_split(const float* qg, const uint16_t* k, const uint16_t* v,
                                 const int32_t* cells, const QsaSplitAttnLayout& layout,
                                 float* partial, void* stream);
}  // namespace strata::kernels
