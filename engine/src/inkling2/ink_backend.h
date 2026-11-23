// engine/src/inkling2/ink_backend.h - Pluggable Backend ABI for strata-inkling (CPU / CUDA / HIP / Metal / SYCL)
// MIT License - Copyright (c) 2026 Strata contributors
#pragma once

#include <cstdint>
#include <memory>
#include <string>
#include <vector>

#include "ink_batch.h"
#include "ink_manifest.h"

namespace ink2 {

struct ImageRec;  // defined in ink_vision.h

struct BackendOpts {
    int max_ctx = 32768;
    int parallel = 1;            // 1..64 concurrent slots
    int chunk = 512;             // prompt tokens per step; the CUDA backend picks a larger one by itself while chunk_auto (see strata/core/ink_chunk.hpp)
    bool chunk_auto = true;      // false once --chunk is given; only the CUDA backend reads it, every other backend uses chunk as it is
    int gpu_index = 0;
    int gpus = 1;                // 1..4 (--gpus 2 / --gpus 0,1 / --layer-split auto)
    std::vector<int> gpu_ids;    // optional explicit GPU ordinals (--gpus 0,1 / --devices 0,1 / IK2_GPUS)
    double ep_split = 0.5;       // GPU 1 expert share in (0, 1) (--ep-split / IK2_EP_SPLIT)
    std::string layer_split;     // "", "auto", or "K1,K2,..." (2..4 GPUs pipeline layer split, zero duplicated KV/experts!)
    int vram_reserve_mib = -1;   // >= 0 reserves N MiB free VRAM & activates hybrid offload (#6520)
    int cpu_threads = 0;         // > 0 activates CUDA-Graph mapped-memory doorbell NUMA CPU expert pool (#6520)
    int threads = 0;             // 0 = auto
    double vram_experts_gb = -1.0; // <0 = auto (3x loop-weighted vram_load_order!)
    double ram_experts_gb = -1.0;  // <0 = auto, 0 = GPU+SSD only (bounded 1-2 GiB DMA ring, 0 host arena)
    bool ssd_experts = false;      // first-class GPU+SSD / CPU+SSD direct NVMe streaming mode
    KvDtype kv_dtype = KvDtype::FP16;
    bool kv_dtype_auto = true;   // default lossless FP16; Host KV Offload handles VRAM capacity (#552)
    std::string kv_offload = "auto"; // "auto" | "on" | "off" (zero-copy mapped pinned host global KV spill)
    bool enable_dspark = true;
    int spec_beam = -1;          // --draft-beam N: 1 = one linear chain (the block drafter), 4 = the beam tree (default); < 0 = STRATA_INK_SPEC_BEAM or 4
    double spec_cost = -1.0;     // --draft-cost R: the MoE share of a verify step in the window-size model (default 4.2)
    double spec_plain = -1.0;    // --draft-plain X: > 0 lets a step fall back to one plain token when the drafter's expected tokens do not pay for the window (1.0 = break-even)
    bool spec_adapt = false;     // --draft-adapt: per request, drafting goes off for 40 steps when its measured gain falls below 0.93 (all backends, see ink_serve.h)
    std::string dspark_path;     // optional external mtp.safetensors or dspark-*.gguf
    std::string vision_path;
};

struct BackendStats {
    size_t vram_experts_loaded = 0;
    size_t ram_experts_loaded = 0;
    size_t total_experts = 0;
    size_t vram_used_bytes = 0;
    size_t vram_total_bytes = 0;
    int active_slots = 0;
    int max_slots = 1;
    KvDtype kv_dtype = KvDtype::FP16;
    uint64_t expert_selections = 0;   // routed experts counted on the batched path (decode), 0 when the backend does not count
    uint64_t expert_misses = 0;       // of those, the ones not in the card
};

// Candidate rows of forward_batch_cand: row r (in the order of the wanted-logits rows) holds `k` = min(kInkCandidates, vocab)
// ids and logits in val / idx [r * kInkCandidates + 0 .. k), and finite[r] (1 = every 67th logit of the row is finite).
struct CandRows {
    int rows = 0;
    int k = 0;
    std::vector<float> val;
    std::vector<int32_t> idx;
    std::vector<uint8_t> finite;
};

class InkBackend {
public:
    virtual ~InkBackend() = default;

    virtual const char* backend_name() const noexcept = 0;
    virtual std::string device_info() const = 0;
    virtual BackendStats stats() const = 0;
    virtual int effective_max_ctx() const { return 0; }
    virtual int effective_chunk() const { return 0; }     // the prefill chunk the backend really runs with (0: opts.chunk)
    // The last forward_batch failed only because the global K/V could not grow (no room in the card, none to spill to): the engine is fine, the
    // requests whose positions reach kv_capacity_tokens() are the ones to end (#6905); every other failure is fatal.
    virtual bool last_failure_is_kv_growth() const { return false; }
    virtual int kv_capacity_tokens() const { return 0; }

    // Load Section 1 trunk + DSpark + 3x loop-weighted Section 4 experts into device/host memory
    virtual bool init(const InkModelFile& model, const BackendOpts& opts, std::string& err) = 0;

    // Slot lifecycle (0 <= slot < opts.parallel)
    virtual bool alloc_slot(int slot, int max_ctx, KvDtype kv_dtype, std::string& err) = 0;
    virtual void free_slot(int slot) = 0;
    virtual void reset_slot(int slot) = 0;
    // Speculative rollback: restore KV + ShortConv state to `keep_len` (requires snapshot taken before verify!)
    virtual void snapshot_slot(int slot) = 0;
    virtual void rollback_slot(int slot, int keep_len) = 0;

    // Optional prompt-end conversation checkpoint (#6627): saves/restores SWA KV + ShortConv + DSpark state at `seq_len`
    // so multi-turn chats and exact repeats reuse the cached prompt even when hidden thinking or trailing tokens diverge.
    virtual bool save_prompt_checkpoint(int slot, int seq_len) {
        (void)slot; (void)seq_len;
        return false;
    }
    virtual bool restore_prompt_checkpoint(int slot, int seq_len) {
        (void)slot; (void)seq_len;
        return false;
    }

    // Separate shared-prefix snapshot; valid only while this slot's global KV prefix is intact.
    virtual bool supports_shared_checkpoint() const { return false; }
    virtual bool save_shared_checkpoint(int slot, int seq_len) {
        (void)slot; (void)seq_len;
        return false;
    }
    virtual bool restore_shared_checkpoint(int slot, int seq_len) {
        (void)slot; (void)seq_len;
        return false;
    }

    // Returns true when the on-device DSpark drafter is initialized and ready to serve drafts (#6633).
    virtual bool dspark_ready() const { return false; }

    // Batched 48-visit forward across 1..64 active slots!
    // logits_out: [batch.count_logits(), vocab_unpadded] FP32 (already softcapped by 30*tanh(z/30) and sliced to 200058).
    virtual bool forward_batch(
        const StepBatch& batch,
        float* logits_out,
        TapCapture* taps_out,
        std::string& err) = 0;

    // The same step, but a row of logits is not copied to the host: the backend returns, per wanted row, the kInkCandidates best
    // token ids with their logits (larger logit first, the smaller id first on equal logits - the order the sampler uses) and a
    // flag telling whether the row's every 67th logit is finite (what the host check looked at).  That is 128 x 8 bytes per row
    // instead of 200058 x 4, which is what limits many concurrent slots.  Exact by construction: the values are the very floats
    // forward_batch would have copied.  A backend that cannot do it returns false from supports_candidates() (the default).
    static constexpr int kInkCandidates = 128;
    virtual bool supports_candidates() const noexcept { return false; }
    virtual bool forward_batch_cand(const StepBatch& batch, CandRows& out, std::string& err) {
        (void)batch; (void)out;
        err = "this backend does not return candidate rows";
        return false;
    }

    // On-device DSpark Block-8 Recurrent Markov drafting for one slot.
    // Returns number of drafted tokens (0..max_draft, typically up to 7); returns 0 if unavailable/disabled.
    virtual int dspark_draft(
        int slot,
        int32_t anchor_tok,
        int anchor_pos,
        int max_draft,
        int32_t* out_draft_ids,
        std::string& err) {
        (void)slot; (void)anchor_tok; (void)anchor_pos; (void)max_draft; (void)out_draft_ids; (void)err;
        return 0;
    }

    // Optional fused on-device greedy speculative step (W=4 Adaptive Beam Tree + zero-logits-D2H verify + commit).
    // Returns n_emit in 1..min(8, max_emit) on success, 0 to fall back to dspark_draft + forward_batch, or -1 on error.
    virtual int dspark_spec_step_greedy(
        int slot,
        int32_t anchor_tok,
        int anchor_pos,
        int max_emit,
        int32_t* out_emitted,
        int* out_drafted,
        std::string& err) {
        (void)slot; (void)anchor_tok; (void)anchor_pos; (void)max_emit;
        (void)out_emitted; (void)out_drafted; (void)err;
        return 0;
    }

    // Optional GPU Vision ViT encoder: returns true if encoded on device, false to fall back to CPU ViT.
    virtual bool encode_image(const ImageRec& img, std::vector<float>& out_embeds, std::string& err) {
        (void)img; (void)out_embeds; (void)err;
        return false;
    }

    // Verify Section 5 selfcheck / golden ops parity on this backend
    virtual bool selfcheck(std::string& err) = 0;
};

// Factory functions (CPU always built; GPU backends plug in via CMake compile definitions!)
std::unique_ptr<InkBackend> make_cpu_backend();

#ifdef INKLING_ENABLE_CUDA
std::unique_ptr<InkBackend> make_cuda_backend();
// true when the GPU can run the Q8_0 dense path (int8 tensor-core GEMM: compute capability 8.0 or newer)
bool ink_cuda_q8f_supported(int gpu_index);
#endif
#ifdef INKLING_ENABLE_EP_CUDA
std::unique_ptr<InkBackend> make_ep_cuda_backend();
#endif
#ifdef INKLING_ENABLE_HIP
std::unique_ptr<InkBackend> make_hip_backend();
#endif
#ifdef INKLING_ENABLE_METAL
std::unique_ptr<InkBackend> make_metal_backend();
#endif
#ifdef INKLING_ENABLE_SYCL
std::unique_ptr<InkBackend> make_sycl_backend(const BackendOpts* opts = nullptr);
#endif

inline std::unique_ptr<InkBackend> make_default_backend(bool force_cpu = false, const BackendOpts* opts = nullptr) {
    (void)opts;
    if (!force_cpu) {
#ifdef INKLING_ENABLE_CUDA
        // Primary InkCudaBackend natively supports .strata, .gguf, safetensors, FP16/FP8/K8V8/K8V4 KV,
        // hybrid offload + GPU+SSD streaming, Dual-GPU EP, AND multi-GPU --layer-split!
        return make_cuda_backend();
#elif defined(INKLING_ENABLE_EP_CUDA)
        return make_ep_cuda_backend();
#elif defined(INKLING_ENABLE_HIP)
        return make_hip_backend();
#elif defined(INKLING_ENABLE_METAL)
        return make_metal_backend();
#elif defined(INKLING_ENABLE_SYCL)
        return make_sycl_backend(opts);
#endif
    }
    return make_cpu_backend();
}

}  // namespace ink2
