// SYCL-only sparse FP32 prompt attention; no change to the common CUDA/HIP API.
#pragma once
#include <cstddef>
#include "strata/kernels/qsa_decode_attn.hpp"

namespace strata::kernels {
// Enqueues core and merge into the supplied queue. Returns false before enqueue
// for unsupported pools, geometry, alignment or device. Scratch uses the existing
// qsa_decode_attn_scratch_floats layout; weights and KV storage are unchanged.
bool qsa_prompt_attn_tla_batch(const float* q, const QsaAttnPools& pools,
                             const int32_t* ids, const int32_t* steps, int64_t cap,
                             const QsaShapes& shapes, float* scratch, float* out,
                             int64_t n_q, void* stream);
// Processes a prompt using at most scratch_queries slices of the existing scratch.
bool qsa_prompt_attn_tla_prefill(const float* q, const QsaAttnPools& pools,
                                 const int32_t* ids, const int32_t* steps, int64_t cap,
                                 const QsaShapes& shapes, float* scratch, float* out,
                                 int64_t n_q, int64_t scratch_queries, void* stream);
struct RopeScaling;
// Prepare the query part of the interleaved query/gate projection on a Battlemage GPU.
bool qsa_tla_prepare_q(const float* query_and_gate, const float* norm_weights, float* query, int64_t queries,
                       int64_t position, float eps, const RopeScaling& rope, void* stream);
// Pre-existing storage which is idle on the in-order compute queue until the output projection.
// The caller supplies the initialized logical prefix for an identity-mapped KV pool.
// The workspace may alias the later output projection only on its in-order queue.
// Keep it available to attention through the merge, before the projection overwrites it.
struct QsaTlaKeyWorkspace {
    void* data = nullptr;
    std::size_t bytes = 0;
    int64_t identity_cells = 0;
    // Set for consecutive causal queries ending at identity_cells.
    int64_t first_query_position = -1;
};
struct QsaTlaGate {
    const float* query_and_gate;
    uint16_t* output;
    int64_t leading_dimension;
};
// Use the supplied workspace when it fits. An optional gate writes the input for the next projection.
bool qsa_tla_sparse_prefill(const float* q, const QsaAttnPools& pools, const int32_t* ids, const int32_t* steps,
                            int64_t cap, const QsaShapes& shapes, float* scratch, float* output, int64_t queries,
                            int64_t scratch_queries, void* stream, QsaTlaKeyWorkspace workspace,
                            const QsaTlaGate* gate);
} // namespace strata::kernels
