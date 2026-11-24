#pragma once
#include "strata/hy4flash_gguf/vision.hpp"
#include <cstdint>
#include <string>
#include <vector>
#include <functional>
#include "strata/hy4flash_gguf/round_state.hpp"

namespace strata::hy4_flash {

struct Geometry;
struct DenseResidency;

/// Wall-clock time per phase, including host work and waits.
struct PhaseTime {
    double engram_read = 0.0;   ///< reading Engram rows out of the 67 GB table
    double engram_gpu = 0.0;    ///< the Engram layer itself
    double weights = 0.0;       ///< uploading or looking up this block's dense weights
    double attention = 0.0;     ///< hc_pre, the MLA chain, hc_post
    double moe = 0.0;           ///< the router, the experts, the shared expert - and the slice fetch, which
                                ///< is inside the expert loop by design and cannot be split out without
                                ///< serialising the thing the fetch exists to overlap
    double dense_ffn = 0.0;     ///< the one block that has one
    double head = 0.0;          ///< the output head
    double total = 0.0;         ///< the whole call, so the unattributed remainder is visible
    int passes = 0;
    // Inside `moe`: the slice fetch, split three ways, because "the expert loop is slow" is not an answer.
    double fetch_wait = 0.0;    ///< blocked on a way's previous copy - i.e. on the GPU, not on the file
    double fetch_read = 0.0;    ///< the memcpy out of the mapping - i.e. on the file
    double fetch_post = 0.0;    ///< enqueueing the copies and the events
    long fetches = 0;
    // The fetcher counts cumulatively; these hold what has already been attributed.
    double seen_wait = 0.0, seen_read = 0.0, seen_post = 0.0;
    long seen_fetch = 0;
};

/// Capture intermediate target activations on device when requested by a drafter.
struct Geometry;
struct DflyCapture {
    enum class Mode { prefill_tail, verify_window };
    // ---- set by the caller before `init` / each pass
    Mode mode = Mode::verify_window;
    std::vector<int> sel_blocks;     ///< block ids whose selection is kept; each must be a sparse block
    // ---- set by `init`
    int capacity_rows = 0, n_embd = 0, key_dim = 0, kpool = 0, sel_cap = 0, n_idx_layers = 0;
    void* hidden = nullptr;          ///< device f32 [capacity_rows, n_embd]
    std::vector<void*> sel;          ///< device i32 [capacity_rows, sel_cap], one per `sel_blocks`
    std::vector<void*> vg;           ///< device f32 [kpool - 1 + capacity_rows, 2 * key_dim], per indexer layer
    // ---- set by a successful pass
    bool valid = false;
    bool has_window = false;         ///< `vg` holds this pass's window (verify_window)
    bool engaged = false;            ///< the selection was in force for the kept row(s)
    int pos_begin = 0;               ///< absolute position of the first kept row
    int n_rows = 0;                  ///< rows kept in `hidden`
    int n_sel = 0;                   ///< cells per row in `sel`
    int win_start = 0;               ///< absolute position of `vg` row 0
    int win_rows = 0;                ///< rows valid in `vg`
    uint64_t generation = 0;         ///< bumped by every successful fill

    bool init(const Geometry& g, int capacity_rows, std::string& err);
    void release();
    ~DflyCapture() { release(); }
    DflyCapture() = default;
    DflyCapture(const DflyCapture&) = delete;
    DflyCapture& operator=(const DflyCapture&) = delete;
};

/// Run a chunk of target tokens through the trunk and output head.
struct ForwardOptions {
    VisionPrompt* picture = nullptr;
    /// Called when the trunk expert tier is allocated or resized (slots and usable expert bytes).
    void (*on_expert_tier)(int slots, std::size_t bytes) = nullptr;
    /// Called after each block with the 4-stream state on the device, [hidden, hc, n_tok]. A test uses it to
    /// compare against the model; return false to stop the pass.
    bool (*on_block)(int block, const float* state, void* user) = nullptr;
    /// Observe the device attention output before gating and the output projection.
    void (*on_attn)(int block, const float* kqv, void* user) = nullptr;
    /// Called with the block's routed expert ids on the device, [top_k, n_tok], for the blocks that route.
    void (*on_routing)(int block, const int32_t* ids, void* user) = nullptr;
    void* user = nullptr;
    /// Replace the state after each block with this (the model's own `l_out`), so no drift accumulates. A
    /// diagnostic: it attributes error per block instead of measuring the accumulation. nullptr = off.
    const float* (*teacher_state)(int block, void* user) = nullptr;
    /// Fill scratch with a NaN pattern rather than zero, so a read-before-write shows up as NaN in the
    /// logits instead of a quiet 0.
    bool poison_scratch = false;
    /// The latent KV cache. nullptr: single-chunk, `n_kv == n_tok`. Given one, the pass appends at
    /// `cache->len` and attends over everything cached, and advances `len` by `n_tok` on success - so the
    /// caller just calls again with the next token.
    struct KvCache* cache = nullptr;
    /// The position of the first token. 0 for a prefill; for decode it is the cache's length, so RoPE and the
    /// causal mask see the real position rather than 0 again. Ignored when `cache` is set, which knows it.
    int first_pos = 0;
    /// Retain dense weights on device across calls.
    struct DenseResidency* dense = nullptr;
    /// Device budget for expert weights in bytes. Requires a persistent dense residency.
    std::size_t expert_vram_bytes = 0;
    /// Keep each trunk expert in a permanent GPU slot or the complementary RAM pool.
    bool resident_experts = false;

    /// Every token of the sequence so far, by absolute position. The Engram hash reads TWO PREDECESSORS, and
    /// during decode they are behind the one token being passed in - without this a decode step would hash as
    /// though it were at the start of a sequence, which is a different row and so a different answer.
    const std::vector<int32_t>* history = nullptr;
    /// Accumulate per-phase wall clock here. nullptr: no timing, and no synchronisation added for it.
    PhaseTime* timing = nullptr;
    /// false: run the trunk and append to the cache, but do not project the output head - `logits` comes
    /// back empty. What a prompt chunk that is not the last one wants: nothing reads its rows, and the head
    /// is 371 MB of weight (120,832 x 3,072), 0.24 ms a chunk that would otherwise be spent per chunk.
    bool want_logits = true;
    /// Return only the last vocabulary row when the caller needs the next-token logits.
    bool logits_last_only = false;
    // Optional caller-owned device rows after hc_head + output_norm, for MTP.
    // Same row selection as logits. The buffer must outlive any captured graph.
    float* hidden_out = nullptr;
    bool hidden_all_rows = false;
    bool speculative = false;
    bool decode_equivalent = false; // preserve single-column projection arithmetic in verify windows
    // Opt-in sampler adapter: return synchronized device logits instead of D2H.
    // Borrowed until the next forward; callers must finish sampling first.
    const float** device_logits_out = nullptr;
    /// Copy what a drafter needs out of this pass (see `DflyCapture`). nullptr: nothing. Needs a cache, a
    /// resident dense set and `want_logits`. Exact verify windows with resident experts can use
    /// graphs cached by (width, indexer engagement); STRATA_HY4_FLASH_DFLY_GRAPH=0 keeps eager execution.
    DflyCapture* dfly = nullptr;
    /// Exact 1–4-row verification requires a cache and one indexer plan for the entire window.
    bool exact_columns = false;
};

/// Preload the ordinary host copy or seed the bounded GPU/RAM pools before the first request.
bool prepare_experts(const std::string& path, const Geometry& g, const ForwardOptions& opt,
                              std::string& err);

/// Return whether every routed expert has a device slot. The first forward pass allocates
/// the tier from the available device memory, up to the configured budget. Before then
/// this returns false. Partial residency can select different prefill and decode paths.
bool dense_experts_complete(const DenseResidency& res);

bool expert_window_enabled();

/// Borrowed embedding and output head views are valid only while the residency is alive.
struct DflyTargetWeights {
    const void* token_embd = nullptr;
    int token_embd_type = 0;
    std::vector<int64_t> token_embd_shape;
    const void* output = nullptr;
    int output_type = 0;
    std::vector<int64_t> output_shape;
};
bool dfly_target_weights(const DenseResidency& res, DflyTargetWeights& out, std::string& err);

/// Rewind the indexer after verification to the prefix the caller accepted.
bool dfly_rewind(DenseResidency& res, struct KvCache& cache, const DflyCapture& cap, int new_len,
                 std::string& err);

/// DIAGNOSTIC: the indexer's live state as bytes, for an exact comparison between two histories that should
/// be the same: host `{n_pool, carry, valid_pos}`, the device `{carry, n_pool}`, every layer's pooled rows
/// `[0, n_pool)` and in-flight rows `[0, carry)`. Dead tails are deliberately not included.
bool dfly_indexer_snapshot(const DenseResidency& res, std::vector<unsigned char>& out, std::string& err);

enum class DflyResetResult { clean, restart_worker };
/// After ANY failure that may have left the device ahead of the host: wait for all device work, write the
/// indexer's device pair to zero unconditionally (the ordinary rewind returns early when host == target),
/// zero the host mirror and `cache.len`. `restart_worker` when the CUDA context cannot be trusted.
DflyResetResult dfly_abort_reset(DenseResidency& res, struct KvCache& cache, std::string& err);

/// Maximum tokens accepted by one forward_prefill call.
int prefill_chunk_max();

// Borrow a device weight already owned by the target residency (embedding/head).
const void* forward_weight(DenseResidency& dense, const std::string& name, int& type);
struct IndexerCache* forward_indexer(DenseResidency& dense);
void forward_commit_device(DenseResidency& dense,const struct RoundVerdict* verdict,void* stream);
void forward_commit_mirror(DenseResidency& dense,KvCache& cache,const struct RoundVerdict& verdict);
// Experimental bounded GPU loop. False without error means not warmed/supported;
// it has queued no work and the caller may use the normal serving path.
bool forward_device_loop(DenseResidency& dense,KvCache& cache,const Geometry& g,
    class Mtp& mtp,class GpuSampler& sampler,const std::vector<int32_t>& history,float* hidden,
    int remaining,int draft_tokens,const std::vector<int32_t>& stops,
    const std::function<bool()>& cancel,const std::function<void(int32_t)>& emit,
    DeviceLoopState& result,std::string& error);

/// `logits` comes back as [vocab, n_tok]. False on error, with `err` set.
bool forward_prefill(const std::string& gguf_path, const Geometry& g, const std::string& pack_dir,
                     const std::vector<int32_t>& tokens, std::vector<float>& logits, std::string& err,
                     const ForwardOptions& opt = {});

/// Persistent device storage for dense weights, populated on first use.
struct OffloadCounters {
    uint64_t expert_bytes=0,transfers=0,hits=0,misses=0;
    uint64_t window_layers=0,window_entries=0,window_unique=0;
};
struct DenseResidency {
    struct Slot;
    /// Keep the model metadata and its backing mapping alive with the residency.
    struct Open;
    Open* file = nullptr;
    /// Retain device scratch and the expert fetcher across passes.
    struct Work;
    Work* work = nullptr;
    /// The Engram token map, 120,832 entries, read from the pack once instead of once per step.
    std::vector<int32_t> token_map;
    /// One per trunk block, and one more at the end for the embedding and output head.
    std::vector<Slot*> slots;
    bool kv_pos_filled = false;    ///< kv_pos[i] == i, so it is written once for the whole reservation
    std::size_t host_bytes = 0;    ///< trunk expert bytes preloaded into the ordinary host file cache
    std::size_t bytes = 0;         ///< the dense weights it holds on the device, for a budget line
    // Actual allocated expert capacity and cache counters.
    int tier_slots = 0;            ///< experts the tier can hold
    int tier_filled = 0;           ///< how many it holds
    std::size_t tier_bytes = 0;
    long tier_hits = 0, tier_misses = 0;
    unsigned long long cpu_experts = 0, cpu_result_bytes = 0; // CPU-computed misses, separate from weight DMA
    unsigned long long h2d_bytes = 0;   ///< read from the host for the misses
    unsigned long long d2d_bytes = 0;   ///< copied on the card for the hits
    // Published after a completed forward; includes prefill-ring weight DMA.
    // Metadata, Engram, dense weights and MTP residency are excluded.
    OffloadCounters offload;
    DenseResidency() = default;
    ~DenseResidency();
    DenseResidency(const DenseResidency&) = delete;
    DenseResidency& operator=(const DenseResidency&) = delete;
};

/// Collect per-step Engram row-read timing and reuse counts.
void engram_read_cost(double& secs, long& steps, long& requests, long& recur, long& distinct);

}  // namespace strata::hy4_flash
