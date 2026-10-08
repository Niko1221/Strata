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
// **IT SERVES A PREFILL CHUNK TOO, AND BY A DIFFERENT RULE.**  `run_chunk` streams every expert the chunk
// routes to through the layer's slots and computes it on the card, so `session_token_chunk` no longer has to
// send the chunk to the CPU pool.  The two rules are different because the two loads are: a decode token
// routes to `k` experts and a cache is the right answer (the ~40-slot layer is warmed by the tokens that
// follow), while a chunk routes to essentially ALL of the layer's 288 (88 tokens already touch 263), so the
// slots are not a cache at all there - they are a ring, refilled `per_layer` at a time until the layer's
// experts have all passed through.  A chunk therefore does not read fewer bytes by being bigger; it pays the
// same fixed cost fewer times, which is what the chunk-size curve in the PR body measures.
//
// **A CHUNK CANNOT BE PARTLY RESIDENT.**  The wave rule above means `run_chunk` rewrites which expert each of
// the layer's slots holds, so it leaves the `slot_` table describing what it actually left there.  Decode's
// LFU counts are NOT touched - `admit` continues from where it was - so a prefill does not teach the tier
// anything false about the conversation.
//
// And it computes nothing itself: `native_expert_grouped` and `native_down_rows` do, so a hit is the
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
    ///
    /// `chunk_tokens` is the largest prefill chunk this will be asked to serve with `run_chunk`, or 0 for a
    /// decode-only tier (which is the whole of the cost this adds, and what every run that does not ask for
    /// prefill gets).  It sizes the chunk plan, the activations and `native_expert_grouped`'s scratch, all of
    /// which come out of `budget_bytes` BEFORE the slots are divided - so asking for it costs a few slots a
    /// layer, not a few slots of nothing.
    bool init(ExpertSource* src, const std::vector<int>& gu_type, const std::vector<int>& d_type,
              const std::vector<uint64_t>& blob_bytes, int64_t layer_lo, int64_t layer_hi, int64_t n_expert,
              int64_t k, int64_t n_embd, int64_t n_ff, int64_t budget_bytes, int64_t chunk_tokens,
              std::string& err);
    bool valid() const { return cache_.valid(); }
    /// Whether this stage holds slots for `layer`'s experts at all.  False for a layer outside its range, and for
    /// a dense lead layer.  `run_chunk` can only serve a layer this says yes to; for any other the caller takes
    /// the CPU pool exactly as it did before the tier existed.
    bool serves_layer(int64_t layer) const {
        return layer >= 0 && layer < n_layers_ && hi_[(size_t) layer] > lo_[(size_t) layer];
    }

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

    /// A whole prefill chunk's MoE for ONE layer, computed entirely on the card: every distinct expert the
    /// chunk routes to is streamed through this layer's slots `per_layer` at a time and grouped-kernel'd against
    /// all the tokens that chose it, writing each `(token, expert)` pair's row of `parts_dev`.
    ///
    /// `ids` are the chunk's routed experts, HOST memory, `T * k` of them token-major (`ids[t*k + i]`), `cur_dev`
    /// is the layer's normed FFN input for all `T` tokens (`T * n_embd` floats back to back, DEVICE), and
    /// `parts_dev` is the session's own `T * k` output rows.  Unlike `run_hits` nothing is staged through the
    /// host: `cur_dev` is read where it lies and `parts_dev` is written where it lies, which is the point.
    ///
    /// **ALL OR NOTHING.**  `served` says whether the card did the layer.  It is false - and nothing is written -
    /// when this stage holds no slots for the layer, when a routed id is out of range, or when `T` or `k` is past
    /// what this was sized for; the caller then sends the whole chunk to the CPU pool as it always did.  There is
    /// no partial answer, because a partial one would need the pool's `T*k` host rows staged back up, which is
    /// most of what this exists to avoid.
    ///
    /// The card's rows are NOT bitwise the pool's: the activation is quantized to q8_1 here and ggml-cpu's
    /// `vec_dot` quantizes it to q8_K there, the same difference `run_hits` has.  `STRATA_GLM_GPU_CHECK` is the
    /// gate, and it is the same one.
    bool run_chunk(int64_t layer, const float* cur_dev, const int32_t* ids, int64_t T, int64_t k, float* parts_dev,
                   void* stream, bool& served, std::string& err);

    /// The distinct experts the last `run_chunk` served, ascending - the CPU check needs the list to recompute.
    const std::vector<int32_t>& last_chunk_experts() const { return plan_e_; }
    /// The last `run_chunk`'s plan, entry by entry: `last_chunk_dst()[p]` is the ROW of `parts` entry `p` owns
    /// and `last_chunk_tok()[p]` the token whose activation it needed.  Entries are grouped by expert through
    /// `last_chunk_off()`, which is why the check can walk `dst` alone: a row names its own (token, expert)
    /// pair, and that is all the CPU needs to recompute it.
    const std::vector<int32_t>& last_chunk_dst() const { return plan_dst_; }
    const std::vector<int32_t>& last_chunk_tok() const { return plan_tok_; }
    const std::vector<int32_t>& last_chunk_off() const { return plan_off_; }
    /// How many `(token, expert)` entries the last `run_chunk` computed - 0 for a chunk that routed nowhere,
    /// which is also what makes the check a no-op rather than a walk over a stale plan.
    int64_t last_chunk_entries() const { return last_chunk_entries_; }
    /// STRATA_GLM_GPU_CHUNK_TIME: where a chunk's wall went, as `run_chunk` saw it.  `host` is the CPU
    /// assembling blobs out of the source, `wait` is this thread blocking on the staging ring, `wave` is the
    /// blocking done at the end of a wave waiting for the card, and `plan` is the host-side plan and uploads.
    /// The split is what says whether the ring is too shallow, the PCIe is too slow, or the card is the wall.
    struct ChunkTimes {
        double host = 0, wait = 0, wave = 0, plan = 0, total = 0;
        int64_t blobs = 0, waves = 0, entries = 0;
    };
    const ChunkTimes& last_chunk_times() const { return chunk_; }

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

    // ---- the chunk path (`run_chunk`).  Its own buffers, deliberately: `d_idx_`/`h_idx_` above are laid out
    // against `k_` and drive the live decode path, and re-sizing them for a chunk would put the decode tier's
    // correctness behind a second caller's geometry.  Everything here is null/empty on a decode-only tier.
    void* cxq_ = nullptr;                      ///< device: the chunk's q8_1 activations, `chunk_tokens_` rows
    void* cscratch_ = nullptr;                 ///< native_expert_grouped's, for `cap_entry_` entries
    unsigned long long* c_ptr_ = nullptr;      ///< device: one group per expert in the wave
    int32_t* c_idx_ = nullptr;                 ///< device: start[] | n_groups | dst[] | tok[], sized on the caps
    unsigned long long* hc_ptr_ = nullptr;     ///< pinned host mirrors
    int32_t* hc_idx_ = nullptr;
    int64_t chunk_tokens_ = 0;                 ///< the largest chunk sized for; 0 = a decode-only tier
    int64_t cap_group_ = 0;                    ///< the most groups, and so the most slots, one wave may use
    int64_t cap_entry_ = 0;                    ///< the most (token, expert) entries one wave may hold
    int64_t chunk_calls_ = 0, chunk_entries_ = 0, last_chunk_entries_ = 0;
    std::vector<int32_t> plan_e_;              ///< host: the distinct experts of the chunk, ascending
    std::vector<int32_t> plan_off_;            ///< host: plan_off_[e]..plan_off_[e+1) index the two below
    std::vector<int32_t> plan_cur_;            ///< host: the counting sort's running cursor, one an expert
    std::vector<int32_t> plan_dst_, plan_tok_; ///< host: the entry's row in `parts` and the token it came from
    // The chunk path's staging ring.  `copy_blob` assembles a GGUF-in-place blob on the CPU (three slices of a
    // mmap, ~11 MB and ~2 ms an expert), so the ring exists to keep that memcpy RUNNING AHEAD of the DMA rather
    // than taking turns with it: a buffer is reused only after the copy that read it has landed, which the event
    // in the same slot says.  The decode path's `stage_` above is k deep because an admission is at most k blobs
    // a layer; a wave is up to `cap_group_` of them.
    std::vector<uint8_t*> stage_chunk_;
    std::vector<void*> stage_ev_;              ///< `cudaEvent_t`s, held as void* so this header needs no CUDA
    ChunkTimes chunk_;                         ///< the last `run_chunk`'s breakdown (STRATA_GLM_GPU_CHUNK_TIME)
    int64_t par_threads_ = 1;                  ///< threads that assemble one wave (STRATA_GLM_GPU_CHUNK_THREADS)
    int64_t last_chunk_threads_ = 0;           ///< the width the last wave's assembly actually ran at
};

}  // namespace strata::core
