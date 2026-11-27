// engine/metal/src/hy4flash_gguf/forward.h - the decode forward pass and the chunked prefill.
//
// WHAT RUNS AND IN WHAT ORDER.  One window of T <= 8 positions is ONE command buffer, split into segments at the
// MoE routers (mtl_rt's run_graph_seg): the GPU computes a block up to and including its router, signals, and waits
// while the host takes its turn - reads the routed ids, puts the missing experts in the arena (experts.h at_cut),
// and writes the expert plan the next segment's kernels read.  So the SSD reads of ordinal o overlap nothing on the
// first pass and the read-ahead of ordinal o + 1 runs while the GPU computes o.
//
// PREFILL IS THE SAME CODE WITH A BIGGER T.  A prompt runs in chunks of `Prefill::chunk` positions (default 1024);
// each chunk walks the blocks in order and each BLOCK is its own command buffer with the same cut after its router,
// so the host groups that block's 1024 x 16 selections by expert and `at_layer` reads each expert from the SSD ONCE
// for the whole chunk (ExpertStream::at_layer's prompt path, T > 8).  Why a command buffer per block rather than one
// for the chunk, as decode does: the dense GEMV covers at most 8 positions per dispatch (gemv.metal's acc[NR][8]),
// so 1024 positions are 128 dispatches of every weight shape and a whole chunk would be some 770,000 dispatches in
// one encoder.
//
// Three stages inside a block run in SUB-BATCHES of positions, because their scratch is what would not fit: the
// attention's score rows (n_tok x n_head x 2051 floats is 537 GB at T = 1024), the MoE's entry-indexed buffers
// (635 MB at T = 1024, see MoeSlots) and the head's logits (T x 120832). A sub-batch is a pure dispatch split: every
// one of these kernels is one thread group per (position, ...) whose arithmetic cannot see the dispatch's token
// count, which is the T invariance this engine requires and the kernel tests check.
// So prefill and decode are bitwise identical, and `--prefill-consistency` checks exactly that end to end.
//
// THE ENGRAM WAIT POINT.  Everything is encoded before the command buffer is committed, so the four tables' rows
// must be in their buffer by then: `issue()` starts the reads at the top of the window and all four are joined
// before the first block is encoded.  A per-block wait point would also be possible; this takes the simpler one and
// reports the wait (`engram_wait_ms`).
//
// WHAT THE STATE IS.  Per block: the MLA latent cache, [max_context][kv_lora + rope_dim] halves, full length
// (sliding blocks are limited by the mask, not by a ring, so a truncation is safe).  Per sparse block: the
// indexer cache, [max_context][3 * idx_key_dim] halves holding value | gate | pooled, the pooled segment written
// into the row of its group's FIRST position when the group fills.
#pragma once

#include "dispatch.h"
#include "dump.h"
#include "engram.h"
#include "experts.h"
#include "model.h"
#include "mtl_rt.h"
#include "prefill_snap.h"

#include <cstdint>
#include <functional>
#include <map>
#include <memory>
#include <string>
#include <vector>

namespace strata::hy4_flash {

/// What one run may inject instead of computing it.  A null pointer means "the engine's own".
struct Inject {
    const float* hc = nullptr;       ///< the block's input HC state, [T][hc_mult][hidden] floats
    const int32_t* topk = nullptr;   ///< [T][top_k] expert ids; the weights are recomputed from OUR probabilities
    const int32_t* sel = nullptr;    ///< [T][sel_cap] attended cell indices per position
    const int32_t* n_sel = nullptr;  ///< [T] how many of them are valid (at most 2051)
};

/// How a prompt is run.  `chunk` = 0 leaves prefill off and the scratch at a decode window's size; anything else
/// allocates for `chunk` positions (about 1.5 GB at 1024).
struct Prefill {
    int chunk = 0;       ///< positions per chunk (HY4_FLASH_PREFILL_CHUNK, --prefill-chunk)
    int attn_sub = 64;   ///< positions per attention dispatch (HY4_FLASH_PREFILL_ATTN): the score scratch
    int moe_sub = 256;   ///< positions per MoE plan (HY4_FLASH_PREFILL_MOE): the entry-indexed scratch
    int head_sub = 8;    ///< positions per head dispatch (the logits buffer)
    int idx_sub = 256;   ///< positions per indexer score dispatch (HY4_FLASH_PREFILL_IDX): the score scratch
    /// HY4_FLASH_PREFILL_MMA (default 1): the prompt's attention and indexer score run on the simdgroup-matrix kernels
    /// (hy4_flash_attn_pf.metal) instead of decode's.  Prefill only - a chunk of more than kMaxWindow positions - so no
    /// decode window ever reaches them; 0 runs the decode kernels for the prompt instead.  Their arithmetic is
    /// NOT decode's (matrix products, an online softmax), which the prefill tolerance allows; nothing is rounded
    /// below fp32 except the K / V operands, which are the cache's own halves.
    int mma = 1;
    /// A WRONG DECISION, ON PURPOSE, IN THE PREFILL PATH ONLY (HY4_FLASH_PREFILL_PROBE), so that
    /// `--prefill-consistency` has a case it must FAIL on.  A bitwise check that has never been seen to fail is
    /// not evidence; these two are the shapes the check exists to catch - an offset that forgets which
    /// sub-batch it is in - and they are invisible to decode, which is one sub-batch of everything.
    ///   1 = the attention sub-batch reads the positions of sub-batch 0 (so every later one attends at the
    ///       wrong absolute position, and the sliding window's lower bound is wrong with it)
    ///   2 = the MoE sub-batch reads sub-batch 0's expert input x
    ///   3 = the sliding window one position too wide in prefill only (a SUBTLE wrongness: one extra key for
    ///       positions >= the window; the tolerance judge must still FAIL it - prefill_tol.h)
    /// A run with it set says so on stdout, like the third gate's mutations do.
    int probe = 0;
    /// A BENIGN perturbation, prefill only (HY4_FLASH_PREFILL_NOISE=<relative size>): every element of each block's
    /// output stream is multiplied by (1 + noise * r), r in [-1, 1) from a fixed hash of (chunk position, block,
    /// element).  It stands in for the rounding of a different fp32 summation order (an MMA prefill), so the
    /// tolerance judge's thresholds can be derived from a perturbation that is known to be harmless.  0 = off.
    double noise = 0;
    /// The three sub-batch sizes and the probe from the environment, with the defaults above.
    static Prefill from_env(int chunk);
};

/// Per-token timings of the last window, in milliseconds.
struct Timing {
    double gpu_ms = 0, wall_ms = 0, host_cut_ms = 0, engram_wait_ms = 0, expert_wait_ms = 0;
    uint64_t windows = 0, positions = 0;
    // prefill, kept apart from decode so the I/O and the compute can be timed separately
    double pf_gpu_ms = 0;      ///< the GPU time of the chunks' command buffers
    double pf_encode_ms = 0;   ///< the host time spent building the dispatch lists
    double pf_host_ms = 0;     ///< the host's turn at the cuts, expert waiting included
    double pf_wait_ms = 0;     ///< of pf_host_ms: waiting for the SSD (ExpertStream's prompt wait)
    /// The prompt's Engram waits, KEPT OUT of `engram_wait_ms`.  Adding them there while the prompt counts
    /// `pf_positions` would make `engram_wait_ms / positions` divide the WHOLE PROMPT's Engram waits by the
    /// DECODE position count, for a table that does not grow with the context, and the inflation would rise
    /// with the number of chunks.  `expert_wait_ms` keeps the same split - ExpertStream's `wait_ms` against
    /// its `prompt_wait_ms`.
    double pf_engram_ms = 0;
    double pf_wall_ms = 0;     ///< wall time of the chunks, the head included
    double pf_head_ms = 0;     ///< of pf_wall_ms: the output head and the logits copy-out
    uint64_t pf_chunks = 0, pf_positions = 0;
    uint64_t pf_bytes_read = 0;   ///< what the SSD delivered while the prompt ran
    std::vector<double> pf_block_ms;   ///< GPU ms per trunk block, summed over the chunks
    // Engram's own split (engram.h's EngramSupply::Timing, mirrored here so main.cpp needs one accessor).
    // `engram_wait_ms` above is the SUM of these three plus prepare_window.
    double eng_tail_ms = 0;    ///< issue()'s opening join_all(): the PREVIOUS window's unfinished readers
    double eng_hash_ms = 0;    ///< engram_row_ids for the four tables + creating the four threads (pure CPU)
    double eng_join_ms = 0;    ///< this window's own read wait, sharing the SSD queue with the expert stream
};

class Forward {
public:
    Forward() = default;
    ~Forward();
    Forward(const Forward&) = delete;
    Forward& operator=(const Forward&) = delete;

    /// Compiles the kernels, resolves every weight and allocates the state.  `pack_dir` may be empty, and then the
    /// Engram tables cannot be read - which is an error only when a block that has one is actually run.
    std::string open(Model& m, mtl::Context& cx, const std::string& pack_dir, const Prefill& pf = {});

    static constexpr int kMaxWindow = 8;        ///< a decode window
    static constexpr int kSelCap = 2051;        ///< the widest selection a query may have

    /// Forgets the caches and the positions.  Call before a new sequence.
    ///
    /// ROLLBACK.  There is no separate truncate: `window` / `prefill` at p0 treat every
    /// position >= p0 as gone.  The MLA and indexer rows are rewritten by the window that reruns them, and the
    /// one piece of state that is NOT per position - how many kpool groups have their pooled key - is clamped to
    /// p0 / kpool at the start of the run, so a group that a rejected draft completed is pooled again from the
    /// rewritten rows.  A speculative verify that is rejected at p is therefore undone by the next window at p.
    void reset();

    /// Runs the whole trunk and the head for the T tokens at positions [p0, p0 + T), with T <= kMaxWindow.
    /// `all_tokens` is the sequence from position 0 (the Engram hash needs two predecessors).  With `dump`, every
    /// MoE block's routing and router logits go into it, which is what a per-layer agreement check needs: without
    /// the logits it can only report the rate, not say whether a difference is a near-tie or an operator error.
    std::string window(const int32_t* all_tokens, int p0, int T, Dump* dump = nullptr);

    /// What a prefill chunk hands back.  `logits` is called once per head sub-batch with the positions' logits,
    /// so a caller can stream 8192 x 120832 floats to a file without the engine holding 4 GB.
    struct PrefillSink {
        std::function<void(int pos, int n, const float* logits)> logits;
        bool all_positions = false;   ///< run the head for every position (teacher forced), not only the last
        /// The trunk's final hidden state (output_norm of the HC head, the MTP block's `h` input) of every
        /// position, a head sub-batch at a time: [n][n_embd] floats.  Runs the HC head and the norm only - not
        /// the vocabulary GEMV - so asking for it costs no logits.
        std::function<void(int pos, int n, const float* h)> hidden;
    };

    /// Runs the trunk for the T positions at [p0, p0 + T) in one go, T <= the chunk size `open` was given.
    /// `prefill_chunk()` is that size, 0 when prefill was not configured.
    std::string prefill(const int32_t* all_tokens, int p0, int T, const PrefillSink& sink, Dump* dump = nullptr);
    int prefill_chunk() const;

    /// After the arena is built and before the first prompt: puts the prompt's scratch into a region of the arena
    /// (HY4_FLASH_SCRATCH_SPLIT=0: it was allocated beside the arena at open, and this does nothing).
    std::string attach_prefill();

    /// Runs the whole prompt in chunks: prefill() over [0, n), the last chunk's last position included.  After it
    /// `last_logits()` / `last_token()` are that position's.
    std::string prompt(const int32_t* tokens, int n, const PrefillSink& sink, Dump* dump = nullptr);

    /// Runs ONLY block L for the T positions at [p0, p0 + T) from the injected HC state (--block-test).
    std::string block_window(int L, const int32_t* all_tokens, int p0, int T, const Inject& inj, Dump* dump);

    /// Runs only the head, from an injected final HC state (--head-test).
    std::string head_window(const float* hc_in, int T, Dump* dump);

    /// The prefix every "a non-finite value reached an f16 cache" error carries, so a caller can tell this apart
    /// from an engine failure: the REQUEST ends (`finish` = `nonfinite`), the process does not.  `window`,
    /// `block_window` and `head_window` all return it, checked after the command buffer that wrote the cache has
    /// completed - the only point at which the flag is readable.
    static constexpr const char* kNonfinite = "nonfinite";
    static bool is_nonfinite(const std::string& err) { return err.rfind(kNonfinite, 0) == 0; }

    /// SAVE / RESTORE THE STATE A PROMPT LEAVES BEHIND (prefill_snap.h), so a decode measurement does not have to
    /// run the prompt again.  What persists between windows is the MLA latent cache, the indexer cache, the
    /// pooled-key counts, the last token and the non-finite tally - the hyper-connection state does not
    /// (`embed_window` rebuilds it from the tokens every window), which is why this is 0.5 GB and not the whole
    /// scratch.
    ///
    /// `b` carries the caller's half of the binding - the model file and the prompt - and the geometry, the
    /// position count and the last token are filled in here, because only this class knows them.  Loading
    /// REFUSES unless every field matches: a decode continued from the wrong KV cache yields plausible tokens,
    /// so it would read as a regression in whatever was being measured rather than as a broken snapshot.
    /// `extra` is state the CALLER keeps that a decode continuation needs - the MTP block's `h` history, which
    /// leaves through `PrefillSink::hidden` and lives in the Mtp object, not here.  It is appended
    /// to the regions and must be identical on load, so a snapshot taken with the MTP cannot be loaded without
    /// it or the other way round; `b.mtp_rows` carries that in the binding so the refusal names it.
    std::string snapshot_save(const std::string& path, SnapBinding b,
                              const std::vector<SnapSlice>& extra = {}) const;
    std::string snapshot_load(const std::string& path, SnapBinding want,
                              const std::vector<SnapSlice>& extra = {});
    /// Positions in the cache: where the next window starts.  -1 before any prompt or load.
    int32_t snapshot_used() const;

    const float* logits() const;                ///< [T][vocab] of the last window
    /// [T][n_embd] of the last window: the trunk's `result_norm`, which is the MTP block's `h` input
    const float* result_norm() const;
    /// The compiled hy4-flash library, for a module that shares the kernels (the MTP block)
    int library() const;
    const int32_t* argmax() const;              ///< [T] of the last window
    /// The last position the head ran for, whichever path ran it: the token a greedy continuation appends.
    int32_t last_token() const;
    const float* last_logits() const;           ///< [vocab] of that position
    const Timing& timing() const;
    int32_t vocab() const;
    /// 0 when no non-finite value ever reached an f16 cache, else how many did, over the whole run.
    /// The per-window check is `is_nonfinite()` on what the window returned; this is the running total.
    uint32_t cache_nonfinite() const;
    /// The last window's routed ids and the next ordinal's predicted ids, for the read-ahead accuracy
    /// measurement: both are [n_ordinal][T][top_k], and entry o of `predicted` is what ordinal o - 1's FFN input
    /// said ordinal o would want (so o = 0 is unused).  The caller compares the two sets per (ordinal, token).
    const std::vector<int32_t>& routed_ids() const;
    const std::vector<int32_t>& predicted_ids() const;
    /// --prefill-consistency only: while prompt() runs, record every chunk's routed ids into *record
    /// ([n_ordinal][n][top_k], resized here) and/or REPLACE them by force's (the same layout; the weights are
    /// recomputed from this run's probabilities, as --force-topk does).  (nullptr, nullptr, 0) turns both off.
    void set_prompt_routing(std::vector<int32_t>* record, const int32_t* force, int n);

    /// One request's pictures, attached before prompt() and cleared after the request.
    ///
    /// `rows` is the tower's output for every picture concatenated in prompt order, `out_hidden` floats a
    /// row; `spans` says which prompt positions each picture's rows replace. `axes` is [3][axes_len] -
    /// the 3-axis M-RoPE positions for the PROMPT - and `delta` is what every position past the prompt
    /// adds, which upstream's contract (include/strata/hy4flash_gguf/vision.hpp) states as "the KV and causal
    /// positions stay sequential; only the RoPE angles use these three planes, with a constant position
    /// offset after the prompt". So pos_b keeps `p0 + t` for the indexer and the cache, and only the two
    /// rope sites read the planes.
    struct ImageSpanRows {
        int64_t start = 0, count = 0;
        const float* rows = nullptr;      ///< count * out_hidden floats
    };
    void set_image_prompt(const std::vector<ImageSpanRows>* spans, const int32_t* axes, int64_t axes_len,
                          int32_t delta, int32_t mrope_len1, int32_t mrope_len2);
    /// Whether a picture is attached: the text path must be able to assert it is not.
    bool has_image_prompt() const;

    struct Impl;   // public so the run_graph_seg trampoline in forward.cpp can name it

private:
    Impl* im_ = nullptr;
};

}  // namespace strata::hy4_flash
