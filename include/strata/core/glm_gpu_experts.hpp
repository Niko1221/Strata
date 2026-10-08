// include/strata/core/glm_gpu_experts.hpp - glm5-next's routed experts in VRAM.
//
// **WHY.**  `GlmExpertPool` computes every routed expert on the CPU.  On `glm-packs/full` (UD-IQ4_XS) that is
// 42 MoE layers x 8 experts x 11.67 MB = **3.92 GB read per token**, and it is the whole of the decode cost:
// the measured 6.8 tok/s warm is ~147 ms a token, and at ~40 GB/s the bytes alone are ~98 ms of it.  Nothing
// else in the arch is close.  The fix is the one the first family already has: stop reading the bytes.
//
// **WHAT THIS IS.**  A per-layer set of experts lives in VRAM (`ExpertCache`), the card computes the ones it
// holds with `native_expert_grouped`, and the CPU pool only gets the misses.  The hits are launched BEFORE the
// pool runs, so the two overlap, and only the misses' rows cross PCIe back.  Slots fill as experts are routed
// (an even quota a layer), then LFU: a slot goes to an expert routed at least 2 more times than the least
// routed one the layer holds, at most one swap a layer a token, counts halved every 4096 tokens so the set
// follows the conversation rather than the prompt.  A run warms up over its first tokens.
//
// **THE QUOTA IS PER STAGE, NOT PER MODEL.**  A layer split gives each card `layer_hi - layer_lo` of the
// trunk, so this holds slots for ITS OWN MoE layers only.  The reference implementation sized over the whole
// model, which on a 4-way split spends three quarters of the VRAM on layers the card is never asked to
// compute - and, because the quota is `budget / total blob bytes`, shrinks every stage's quota by 4x on the
// way.  `lo_`/`hi_` are the layer's own slot range and are empty for a layer outside the stage's range.
//
// **WHAT IT DOES NOT DO.**  Prefill.  `session_token_chunk` still sends every expert of a chunk to the CPU
// pool; the tier is warmed by the decode tokens that follow, which for a ~40-slot layer is single-digit
// tokens.  And it computes nothing itself: `native_expert_grouped` and `native_down_rows` do, so a hit is the
// same arithmetic the pool would have done, in a different order - see STRATA_GLM_GPU_CHECK below.
//
// Off unless `--glm-gpu-experts N` is given, so every run that does not ask for it is bit-identical to before
// it existed.  `--glm-gpu-experts 0` takes what the card has free less STRATA_GLM_GPU_RESERVE_MIB (2048);
// a positive N caps it at N MiB.
//
// **STRATA_GLM_GPU_CHECK=1 recomputes every hit on the CPU and prints max |gpu - cpu|.**  That is the only
// thing that says a slot the card computed holds the expert it is named for; a slot table that is right about
// indices and wrong about bytes produces a plausible token, which is the failure this project pays for most.
// The gap is not zero and is not meant to be: the GPU path quantizes the activation to q8_1 and ggml-cpu's
// `vec_dot` to q8_K, so ~2% of the largest value is the rounding, and a slot holding the WRONG expert is not
// 2% of anything.
#pragma once

#include "strata/core/expert_cache.hpp"
#include "strata/core/expert_source.hpp"
#include "strata/kernels/iq_kernels.hpp"

#include <cstdint>
#include <memory>
#include <string>
#include <vector>

namespace strata::core {

class GlmGpuExperts {
public:
    GlmGpuExperts() = default;
    ~GlmGpuExperts();
    GlmGpuExperts(const GlmGpuExperts&) = delete;
    GlmGpuExperts& operator=(const GlmGpuExperts&) = delete;

    /// `gu_type` / `d_type` / `blob_bytes` are per layer, indexed by the GLOBAL ordinal, as
    /// `kernels::cpu::expert_layout()` reports them (`blob_bytes[l] == 0` for a dense lead layer and for any
    /// layer with no row).  `[layer_lo, layer_hi)` is this stage's trunk range and the only range that gets
    /// slots; `budget_bytes` is the VRAM they may take, split evenly over the range's MoE layers.
    bool init(ExpertSource* src, const std::vector<int>& gu_type, const std::vector<int>& d_type,
              const std::vector<uint64_t>& blob_bytes, int64_t layer_lo, int64_t layer_hi, int64_t n_expert,
              int64_t k, int64_t n_embd, int64_t n_ff, int64_t budget_bytes, std::string& err);
    bool valid() const { return cache_.valid(); }

    /// One decode token, one MoE layer.  `ids` are the routed experts (HOST, `k` of them, and `k` may not
    /// exceed the `k` this was sized for - `job`s past it would write past `slot_`).  The hits are launched on
    /// `stream` and write their UNWEIGHTED rows `i` of `parts_dev` (`k` x `n_embd` floats); `miss` receives the
    /// positions `i` the CPU still has to compute.  `cur_dev` is the layer's normed FFN input on the device -
    /// the same tensor `glm_x_host` is a copy of.
    bool run_hits(int64_t layer, const float* cur_dev, const int32_t* ids, int64_t k, float* parts_dev,
                  void* stream, std::vector<int32_t>& miss, std::string& err);

    /// After the CPU computed the misses and their rows were staged into `parts_dev`: move each miss into a
    /// slot (a free one, or the layer's least-routed expert when the newcomer is at least 2 hotter), reading
    /// the blob through `src_`.  The copies are queued on `stream` behind this layer's hit kernel, which may
    /// still be reading a slot being replaced, and ahead of the next token's kernel that reads it.
    bool admit(int64_t layer, const int32_t* ids, const std::vector<int32_t>& miss, void* stream, std::string& err);

    /// STRATA_GLM_GPU_CHECK: the positions in `parts_dev` the last `run_hits` filled from VRAM.
    const std::vector<int32_t>& last_hits() const { return hit_pos_; }

    int64_t hits() const { return hits_; }
    int64_t misses() const { return misses_; }
    int64_t admitted() const { return admitted_; }
    int64_t slots() const { return cache_.full_slots(); }
    int64_t per_layer() const { return per_layer_; }
    int64_t swaps() const { return swaps_; }
    double gib() const { return cache_.gib(); }

private:
    ExpertCache cache_;
    ExpertSource* src_ = nullptr;
    int dev_ = -1;                    ///< the device the buffers live on; the dtor frees under it
    int64_t n_layers_ = 0, n_expert_ = 0, k_ = 0, n_embd_ = 0, n_ff_ = 0, per_layer_ = 0;
    std::vector<strata::kernels::NativeExpertLayout> lay_;
    std::vector<int64_t> lo_, hi_, next_;      ///< per layer: its slot range and the next free slot in it
    std::vector<int32_t> slot_;                ///< [layer * n_expert + expert] -> slot or kNotResident
    std::vector<int32_t> owner_;               ///< [slot] -> the expert it holds, or -1
    std::vector<uint32_t> count_;              ///< [layer * n_expert + expert]: how often it was routed
    int64_t moe_layers_ = 0, calls_ = 0, swaps_ = 0, admitted_ = 0;
    int64_t hits_ = 0, misses_ = 0, rep_hits_ = 0, rep_miss_ = 0;
    void* xq_ = nullptr;                       ///< the token's q8_1 activation
    void* scratch_ = nullptr;                  ///< native_expert_grouped's
    unsigned long long* d_ptr_ = nullptr;      ///< device: one group per hit, its blob's address
    int32_t* d_idx_ = nullptr;                 ///< device: start[k+1] | n_groups | dst[k] | tok[k]
    unsigned long long* h_ptr_ = nullptr;      ///< pinned host mirrors of the two above
    int32_t* h_idx_ = nullptr;
    std::vector<uint8_t*> stage_;              ///< pinned staging, one blob per admission in flight
    std::vector<int32_t> hit_pos_;
};

}  // namespace strata::core
