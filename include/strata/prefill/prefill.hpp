// include/strata/prefill/prefill.hpp - plan v0.3 P5: batched prompt processing.
//
// The prompt's positions [pos0, pos0 + n) are processed in chunks of `chunk` tokens through all 48 layers, leaving
// the session state (GDN recurrence and conv state, QSA KV pools and indexer, PLE history) where the token path
// would have left it; the decode loop then continues with the next token.  Per layer: the projections are
// tensor-core GEMMs (quantized weights dequantized to FP16 on the fly, BF16 weights as they are), the recurrences
// walk the chunk inside one kernel, and the routed experts are grouped by expert: resident ones are read from the
// VRAM tier, the others streamed from the host arena through a pinned ring on a copy stream.
//
// Requires the native weights (`--native`): every quantized projection must carry its GGUF blocks.
#pragma once

#include "strata/core/expert_cache.hpp"
#include "strata/core/expert_source.hpp"
#include "strata/core/layer.hpp"
#include "strata/core/session.hpp"
#include "strata/core/weights.hpp"

#include <cstdint>
#include <functional>
#include <future>
#include <memory>
#include <string>

namespace strata::core { class PeerExperts; }

namespace strata::prefill {

struct PrefillStats {
    int64_t tokens = 0;
    int64_t chunks = 0;
    double ms_total = 0;
    double ms_experts_host = 0;     ///< host time staging non-resident experts
    int64_t experts_streamed = 0;   ///< expert blobs copied host -> device
    int64_t experts_dma = 0;        ///< ...of which straight from the pinned arena (no CPU copy)
    int64_t experts_resident = 0;   ///< expert-layer groups served from the VRAM tier
    double ms_ple = 0;
};

}  // namespace strata::prefill
namespace strata::core { class MtpDrafter; }
namespace strata::prefill {

class Prefill {
public:
    /// E-9: the draft layer's K/V for prompt cells [cell0, cell0 + n) from their final residual rows `R_rows`
    /// (device) and `next_tokens` (host: the token at cell+1), in batches through this path's GEMMs and its idle
    /// scratch - call it from on_chunk.  false with `err` empty: not applicable here (a ring or hybrid K/V, another
    /// device, too little scratch; STRATA_MTP_BATCH=0), the caller runs the drafter's own pass.  Not bit-identical to
    /// that pass (FP16 GEMMs instead of Q8_1 activations): the drafts may differ, never the target's tokens' logits.
    bool draft_kv(core::MtpDrafter& mtp, const float* R_rows, const int32_t* next_tokens, int64_t n, int64_t cell0,
                  std::string& err);
    Prefill();
    ~Prefill();
    Prefill(const Prefill&) = delete;
    Prefill& operator=(const Prefill&) = delete;

    /// Frees every buffer, stream and event `init` made (as the destructor does) and starts over empty, so `init` can
    /// run again - with a smaller chunk when the first one did not fit.  The stage range (`set_stage`) and the
    /// callbacks stay.  The device `init` ran on must be current.
    void reset();

    /// `host_res`: the static residency table (n_layers x n_expert, slot or -1) or null; `cache` its slots.
    /// `borrow`/`borrow_bytes`: device memory to carve every buffer from (the top slots of the expert cache,
    /// lent for the prompt and refilled after it); null = allocate normally.
    bool init(const core::WeightTable& wt, const core::ModelGeometry& g, core::SessionState& ss,
              core::ExpertSource* src, const core::ExpertCache* cache, const int32_t* host_res, int64_t chunk,
              void* stream, std::string& err, void* borrow = nullptr, uint64_t borrow_bytes = 0);

    /// With borrowed buffers: lay them out again for chunks of `chunk` tokens (at most `init`'s) in `borrow` - a
    /// request lends only the slots its prompt needs.  The stream must be idle (between prompts).
    bool relayout(int64_t chunk, void* borrow, uint64_t borrow_bytes, std::string& err);
    int64_t chunk() const;

    /// The share of the streamed experts' bytes DMA-able straight from pinned RAM (1 = all).  Sizes the streamed
    /// ring (a big one only pays when the copy engine, not the host copies, is the limit); set before bytes_needed.
    static void set_pinned_share(double share);
    static double pinned_share();
    /// #340: the streamed ring's slot count for chunks that stream every expert, instead of the pinned-share rule
    /// (0 = that rule). Set before any `bytes_needed`/`init` (both count the ring); STRATA_PREFILL_RING still wins.
    static void set_ring_override(int slots);
    /// #583: the auto chunk scan's byte-budget ring, for chunks above `small_max` (0.1.39's auto chunk: a prompt that
    /// fits it keeps 0.1.39's ring).  0 slots = none.  A layer split's set_ring_override and STRATA_PREFILL_RING win.
    static void set_ring_budget(int slots, int64_t small_max);

    /// Device bytes `init` needs for a chunk of `chunk` tokens (what a borrowed region must hold).
    static uint64_t bytes_needed(const core::ModelGeometry& g, const core::SessionState& ss, int64_t chunk);

    /// The same without the streamed ring: what the chunk's own buffers cost.  The auto chunk scan sizes the chunk
    /// first and hands the ring what the chunk leaves over, so it needs the chunk priced on its own.
    static uint64_t bytes_needed_no_ring(const core::ModelGeometry& g, const core::SessionState& ss, int64_t chunk);

    /// The streamed ring's byte budget as a slot count for this pack (the measured slot count x Q2_0's blob, over
    /// max_blob, never past ring_cap()):
    /// what the auto chunk scan treats as a full ring.  A slot is one whole blob, so a pack with bigger blobs than
    /// Q2_0's gets fewer of them for the same bytes - 384 on Q2_0, 199 on a 2.54 MiB-blob IQ3_S pack.
    static int64_t ring_max_slots();
    /// 0.1.39's ring for this PC (1024 fused / 384 pinned / 96), within ring_cap().
    static int64_t ring_default_slots();
    /// #583: the ring the auto scan keeps full, given the chunk 0.1.39's rule picked: its byte budget where that rule's
    /// chunk was small (< 6144), else 0.1.39's ring (measured: shrinking it for a bigger chunk lost there).
    static int64_t ring_cap_for(int64_t old_chunk);

    /// What the ring actually resolves to for a chunk of `chunk` tokens, after the override, STRATA_PREFILL_RING
    /// and the pinned-share rule - the slot count `init` lays out.  The engine reports it on its INFO line so the
    /// Monitor tab shows the pair the run really got, not what it asked for.
    static int64_t ring_slots_for(int64_t chunk);

    /// 0.1.39b (#583, the default): the ring as a byte budget, the loan's corrected count and the auto chunk scan that
    /// keeps the ring full.  STRATA_RING_BYTES=0: 0.1.39's ring, loan and chunk list.
    static bool ring_bytes_enabled();

    /// Positions [pos0, pos0 + n) holding `tokens`; `ss.ple_prev` must be the two tokens before pos0 (oldest
    /// first, -1 for none) and is advanced to the last two of these.
    bool run(const int64_t* tokens, int64_t n, int64_t pos0, std::string& err);

    const PrefillStats& stats() const { return stats_; }

    /// multi-GPU: the experts the peer GPU holds are computed THERE for every prompt chunk (up to `cap_rows` routed
    /// rows per layer; the rest of the peer's experts are read by this GPU over P2P).  Allocates the peer's buffers for
    /// chunks of up to init's `chunk` tokens.  Needs P2P between the two cards.
    bool set_peer(core::PeerExperts* peer, int64_t cap_rows, std::string& err);

    /// Plan v0.3 P6: called after every chunk with the chunk's final multi-stream residual rows (device,
    /// T x hc*n_embd, valid until the next chunk) and the chunk's first position; the MTP draft layer builds its
    /// K/V from them.  The prefill stream is synchronized before the call.
    std::function<bool(const float* R_rows, int64_t T, int64_t pos0, std::string& err)> on_chunk;

    /// Layer split: called by every stage when it has read a chunk, with the position reached, while its own state
    /// is still at that chunk's end (its stream synchronized; the last stage calls it just before `on_chunk`).  An
    /// earlier stage is a chunk or more ahead of the last one by the time `on_chunk` runs, so this is where a
    /// mid-prompt checkpoint takes each stage's part.  Runs on that stage's thread, with its device current.
    std::function<bool(int64_t done, std::string& err)> on_stage_chunk;

    /// Checked before every chunk: true stops the prompt early (`run` returns false with err "cancelled").
    std::function<bool()> should_stop;

    /// The vision path: HOST rows (n_embd floats) indexed by absolute position, read in place of the token
    /// embedding where non-null (an image's <|image_pad|> cells).  Null (default): every position embeds its token.
    const float* const* embd_rows = nullptr;

    /// LAYER SPLIT (multi-GPU): this prompt path runs layers [layer_begin, layer_end) (-1: to the last) on the
    /// device `init` runs on.  A stage that does not start at layer 0 reads each chunk's residual rows from the
    /// previous stage instead of embedding the tokens; a stage that does not end at the last layer copies its rows
    /// to pinned host buffers (two, allocated by `init`) and runs `next` on them - on a thread, so the next stage
    /// reads chunk c while this one reads chunk c + 1.  `on_chunk` belongs on the last stage.  Set before `init`.
    void set_stage(int64_t layer_begin, int64_t layer_end, Prefill* next) {
        stage_lb_ = layer_begin; stage_le_ = layer_end; next_ = next;
    }

    /// LAYER SPLIT: `helper` is another stage's prompt path (another GPU).  A prompt of one chunk runs the stages one
    /// after the other, so while this stage reads it the helper's GPU idles: it then streams a share of this stage's
    /// non-resident experts over its own PCIe link into its own (lent) prompt buffers, computes their rows, and sends
    /// them back - the --peer-device peer's streaming, without P2P (activations and rows through mapped host memory,
    /// read by copy kernels).  Only on the MMQ prompt path, only for chunks of stream_all_min() tokens and more, up to
    /// the size where the measured share stops paying.  Opt-in: STRATA_PREFILL_HELP=1 (not bit-identical to the default).  Both stages must have
    /// run `init`.
    bool set_stage_helper(Prefill* helper, std::string& err);

private:
    // Stage-1 pipeline: intermediate stages return after handing their chunk to
    // the direct successor. The public run() drains the chain once at prompt end.
    bool run_impl(const int64_t* tokens, int64_t n, int64_t pos0, std::string& err);
    bool drain_pipeline(std::string& err);

    int64_t stage_lb_ = 0, stage_le_ = -1;
    Prefill* next_ = nullptr;
    Prefill* helper_ = nullptr;         ///< set_stage_helper
    bool single_chunk_ = false;         ///< a later stage: the prompt is one chunk (set by the stage before)
    bool bind_stage_helper(int64_t T);  // binds the helper's buffers for a one-chunk prompt of T tokens
    const float* hand_in_ = nullptr;    ///< the previous stage's rows of the chunk being read (host, pinned)

    std::string next_err_;
    std::future<bool> next_run_;
    int hand_buf_ = 0;

    bool carve(std::size_t T, void* alloc);   // the device buffers of a chunk (prefill.cpp's Alloc)
    void release();                          // the destructor's cleanup (also `reset`'s)
    struct Impl;
    std::unique_ptr<Impl> impl_;
    PrefillStats stats_;
};

}  // namespace strata::prefill
