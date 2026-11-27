// Batch prompt expert projections by expert across tokens.
#pragma once

#include "strata/hy4flash_gguf/attention.hpp"

#include <cstddef>
#include <cstdint>
#include <vector>

namespace strata::hy4_flash {

/// The chunk at which batching starts to pay; `latent_moe_mmq_scratch_bytes` is zero at or below it, so a
/// decode step pays nothing for a path it cannot take.
extern const int kMoeBatchMin;

/// Check whether this layer's weights and layout are supported by batched MMQ.
bool moe_mmq_usable(int gate_type, int up_type, int down_type, int expert_dim, int expert_ff, int n_tok);

/// Device bytes `latent_moe_experts_mmq` needs. Dominated by the four [n_tok * k, expert_ff or expert_dim]
/// F32 intermediates, so it grows with the chunk: about 118 MiB at 256 tokens for this model.
std::size_t latent_moe_mmq_scratch_bytes(int expert_dim, int expert_ff, int n_expert, int k, int n_tok, int row_cap = 0);

// Prepared once per layer/chunk from the routing already read by the supply scheduler. Each batch
// owns disjoint (token,pick) entries; device views live until the next prepare on this thread/stream.
struct MoeBatchSpec {
    std::vector<int32_t> slot_of;
    int n_slots = 0;
};
struct MoeBatchPlan {
    int32_t* bounds = nullptr;
    int32_t* absolute_bounds = nullptr;
    int32_t* quant_ids = nullptr;
    int total_rows = 0;
    int32_t* src = nullptr;
    int32_t* entries = nullptr;
    int32_t* identity = nullptr;
    void* xq = nullptr;
    int n_slots = 0, n_rows = 0, max_rows = 0;
    int workspace_rows = 0, weight_base = 0;
    std::int64_t tile_rows_hint = -1;
    std::vector<MoeBatchPlan> children;
};
std::vector<MoeBatchPlan> moe_prepare_batches(const std::vector<int32_t>& ids,
    const std::vector<MoeBatchSpec>& specs, int n_expert, int k, int n_tok,
    const float* routed_in, int gate_type, int expert_dim, void* stream);
void moe_finish_batches(void* stream);
int moe_group_rows(int k, int n_tok);

/// Write the routed expert contributions for a prompt chunk into latent.
bool latent_moe_experts_mmq(const float* routed_in, const ExpertGroup& grp, int resident_base,
                            int gate_type, int up_type, int down_type, int expert_dim, int expert_ff,
                            int n_expert, int k, int n_tok, const int32_t* ids, const float* rw,
                            float swiglu_clamp, float* latent, void* scratch, std::size_t scratch_bytes,
                            void* stream, const int32_t* slot_of = nullptr, int n_slot = 0,
                            bool accumulate = false, float* parts_out = nullptr, const MoeBatchPlan* plan = nullptr);

// Batch the per-head attention projections over a prompt chunk.

/// Whether the batched per-head path can run: MMQ covers the type, its tile fits `n_out` rows here, and
/// there is more than one token to batch.
bool grouped_mmq_usable(int ggml_type, int n_out, int n_tok);

/// Device bytes `grouped_product_mmq` needs for `n_group` groups of `n_tok` rows of `n_in` values.
std::size_t grouped_mmq_scratch_bytes(int ggml_type, int n_in, int n_group, int n_tok);

/// `y[(g * n_tok + t) * n_out ..]` = matrix g times row (g, t) of `x`, for every group and token in one
/// launch. Returns false and launches nothing when it cannot run, so the caller keeps its loop.
bool grouped_product_mmq(const void* w, int ggml_type, std::size_t head_stride, const float* x, int n_in,
                         int n_out, int n_group, int n_tok, float* y, void* scratch,
                         std::size_t scratch_bytes, void* stream);

/// Free the MMQ launch context and the pinned readback this path keeps per thread.
void prefill_moe_release();

}  // namespace strata::hy4_flash
