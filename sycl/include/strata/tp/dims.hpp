// THE ONE HOME for the per-rank GDN/QSA counts that tensor parallelism needs.  Every consumer - the session
// state strides (`gdn_state_floats`), the prefill fast path, the verify windows and the buffer carving in
// layer.cpp - MUST take these counts from here.  Before this header they were three copies that could
// drift, and one did drift in the other direction: the batched KV append kept the MODEL's page stride
// while the pools were allocated with the RANK's counts, which is the non-finite-logits bug fixed in
// commit 7f55fd3 (docs/TP2-HANDOVER.md section 0).  Every function here is the identity when
// `tp_ranks_current()` is 1, so a single-card run gets the model's global counts byte for byte.
#pragma once
#include <cstdint>
#include "strata/core/geometry.hpp"
#include "strata/core/layout.hpp"    // core::qsa_shapes
#include "strata/core/shard.hpp"     // core::tp_ranks_current
#include "strata/kernels/qsa.hpp"    // kernels::QsaShapes

namespace strata::tp {

inline int64_t gdn_v_heads_rank(const core::ModelGeometry& g) {
    const int64_t r = (int64_t) core::tp_ranks_current();
    return r > 0 ? g.ssm_v_heads / r : g.ssm_v_heads;
}
inline int64_t gdn_channels_rank(const core::ModelGeometry& g) {
    const int64_t r = (int64_t) core::tp_ranks_current();
    return r > 0 ? g.ssm_conv_channels / r : g.ssm_conv_channels;
}
inline int64_t gdn_k_heads_rank(const core::ModelGeometry& g) {
    const int64_t r = (int64_t) core::tp_ranks_current();
    return r > 0 ? g.ssm_k_heads / r : g.ssm_k_heads;
}

// A QSA rank holds its OWN attention heads (and its own slice of the indexer's queries), so every shape
// the QSA kernels take is the per-rank one.
inline kernels::QsaShapes qsa_shapes_rank(const core::ModelGeometry& g) {
    kernels::QsaShapes s = core::qsa_shapes(g);
    const int64_t tpr = (int64_t) core::tp_ranks_current();
    if (tpr > 1) {
        s.n_head = (int) ((int64_t) g.n_head / tpr);
        s.n_head_kv = (int) ((int64_t) g.n_head_kv / tpr);
    }
    return s;
}

}  // namespace strata::tp
