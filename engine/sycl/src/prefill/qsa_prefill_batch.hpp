#pragma once
#include "strata/kernels/qsa_prompt_attn.hpp"
#ifdef STRATA_SYCL_TLA_ATTN
#include "strata/kernels/qsa_prompt_attn_tla.hpp"
#include <cstdlib>
#endif

namespace strata::kernels {
enum class QsaPrefillBatch { declined, tla, batch };

#ifdef STRATA_SYCL_TLA_ATTN
inline bool qsa_use_tla() {
    static const bool enabled = [] {
        const char* value = std::getenv("STRATA_SYCL_TLA_ATTN");
        return !value || value[0] != '0';
    }();
    return enabled;
}
#endif

// A declined backend has not submitted any work. The caller retains its per-cell/warp fallback.
inline QsaPrefillBatch qsa_prefill_batch(const float* q, const QsaAttnPools& pools,
                                         const int32_t* ids, const int32_t* steps, int64_t cap,
                                         const QsaShapes& shapes, float* scratch, float* out,
                                         int64_t queries, int64_t scratch_queries, void* stream) {
#ifdef STRATA_SYCL_TLA_ATTN
    if (qsa_use_tla() && qsa_prompt_attn_tla_prefill(q, pools, ids, steps, cap, shapes, scratch,
                                                    out, queries, scratch_queries, stream))
        return QsaPrefillBatch::tla;
#else
    (void)scratch;
    (void)scratch_queries;
#endif
    return qsa_prompt_attn_batch(q, pools, ids, steps, cap, shapes, out, queries, stream)
               ? QsaPrefillBatch::batch : QsaPrefillBatch::declined;
}
} // namespace strata::kernels
