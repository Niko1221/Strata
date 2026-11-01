// include/strata/core/dflash2.hpp - the DFlash2 drafter for Qwen3.8-Flash-Next on the GPU (CUDA and HIP).
//
// incoai/Qwen3.8-Flash-Next-DFlash2 (Apache-2.0) is a block drafter: 5 transformer layers of hidden 2560 that read the target's hidden
// states (the mean of the 4 hyper-connection streams after layers 5, 15, 25, 35, 45) as context and, from the last accepted token plus
// 7 mask tokens, produce 7 draft tokens in one forward pass.  A candidate selector (the head's top-16 per row, a rank-256 bilinear
// score between the previous pick and each candidate) picks the path left to right.  tools/dflash2_pack.py packs it as Q8_0.
//
//   context    taps (5 x 2560 per accepted row) -> fc (12800 -> 2560) -> hidden_norm -> per layer K (norm + rope) and V -> ring slot pos % R
//   block      8 rows: embed(anchor), embed(mask) x 7; per layer
//                rms -> dynamic 2-tap conv (prepare) -> q, k, v -> q/k norm + rope -> attention over the ring + the block's own keys
//                -> o -> conv (finish) + residual -> rms -> conv (prepare) -> gate/up -> silu*up -> down -> conv (finish) + residual
//   selector   final norm -> head over the draft vocabulary (rows 1..7) -> top-16 -> walk
//
// The ring holds only COMMITTED rows (an accepted window row's taps arrive after the verify window), and the block's keys stay outside
// it, so a rejected draft leaves nothing to roll back.  One CUDA graph per context-row count n = 1..8 holds the context update and the
// block; the host stages the positions and the anchor and waits for the 7 drafts.
//
// The weights, head and block scratch belong to the card; the ring K/V and the captured round graphs belong to one conversation.  One
// drafter can therefore hold the state of several conversations in slots (include/strata/core/dflash2_slots.hpp): it drafts the active
// slot, and a batch loop switches slots between rounds.  One slot - the default - is exactly the drafter as it has always run.
#pragma once

#include "strata/core/dflash2_slots.hpp"
#include "strata/core/session.hpp"

#include <cuda_runtime.h>

#include <array>
#include <cstdint>
#include <functional>
#include <initializer_list>
#include <memory>
#include <string>
#include <vector>

namespace strata::core {

class NativeHead;

class Dflash2Drafter {
public:
    static constexpr int kRows = 8;           ///< block rows: the anchor + 7 masks
    static constexpr int kDrafts = kRows - 1;
    static constexpr int kBatchMaxRows = kDf2MaxSlots * kDrafts;
    static constexpr int kDefaultWindow = 2048;
    static constexpr int kMaxWindow = 8192;   ///< opt-in pack metadata limit; longer windows need larger ring/kernel buffers
    static constexpr int kRingSlack = 16;     ///< rows written ahead of a rejection
    static constexpr int kRing = kDefaultWindow + kRingSlack;  ///< default pack ring size, kept for existing bench callers
    static constexpr int kMaxRing = kMaxWindow + kRingSlack;

    struct BatchItem {
        int slot = 0;                ///< conversation slot index (0 .. n_slots() - 1)
        const float* taps = nullptr; ///< device pointer to this slot's committed tap rows [n_new][n_taps * N]
        int n_new = 0;               ///< committed tap rows (1 .. kRows)
        int64_t pos0 = 0;            ///< position of the first committed row
        int32_t anchor = 0;          ///< last committed token (the block's row 0 token)
        int k_want = kDrafts;        ///< proposals to produce for this slot (1 .. kDrafts)
        int32_t* drafts = nullptr;   ///< host output [k_want]
        float* probs = nullptr;      ///< optional host output [k_want]
    };

    Dflash2Drafter() { slots_.push_back(std::make_unique<Slot>()); }   // slot 0: the drafter as it runs today
    ~Dflash2Drafter();
    Dflash2Drafter(const Dflash2Drafter&) = delete;
    Dflash2Drafter& operator=(const Dflash2Drafter&) = delete;

    /// Reads the pack (tools/dflash2_pack.py) to the GPU `device` and allocates the ring and the buffers (about 0.8 GB).
    bool load(const std::string& pack_path, int device, std::string& err);
    /// The target's embedding and head (and the draft vocabulary file, "" or missing: the whole head).
    bool bind(const WeightTable& wt, const NativeHead* head, const std::string& vocab_file, std::string& err);
    /// Device bytes bind() will allocate for the draft head subset, index map, and logits buffer.  `vocab_file` is
    /// the draft vocabulary (int32 ids, as tools/dflash2_pack.py writes it); "" or an unreadable file means the whole
    /// head, which is what bind() falls back to.  The engine needs this before bind() to size the expert cache.
    uint64_t bind_bytes(uint64_t head_row_bytes, int64_t n_vocab, const std::string& vocab_file) const;
    /// Tests: an embedding callback and a head given as raw rows of `head_type` (a native_mmvq type).
    using EmbedFn = std::function<void(const int32_t* tokens_dev, int n, float* out, cudaStream_t)>;
    bool bind_raw(EmbedFn embed, const void* head_w, int head_type, int64_t head_rows, const std::vector<int32_t>& vocab, std::string& err);

    /// A new request: no row in the old request's ring is a valid context cell.
    void reset();
    /// Rewinds the live ring in place to `prefix_len`, zeroing any slots written at positions >= prefix_len.
    void rewind(int64_t prefix_len);
    /// Context rows without a draft: `n` rows of taps (device, [n][5 * n_embd]) for positions pos0 ... pos0 + n - 1.
    bool context(const float* taps_dev, int64_t n, int64_t pos0, std::string& err);
    /// One round: the new context rows (n_new in 1..8: the committed window rows, taps device [n_new][5 * n_embd], positions pos0 ...),
    /// then the block for `anchor` (the last committed token, at pos0 + n_new).  `drafts` / `probs` get kDrafts entries (the
    /// selector's pick and its probability).  The taps pointer must be the device buffer this slot's graphs were captured on
    /// (the guard is per slot: df2_graphs_valid_for against Slot::taps_); the engine keeps one shared staging buffer for all slots.
    bool draft(const float* taps_dev, int n_new, int64_t pos0, int32_t anchor, int32_t* drafts, float* probs, std::string& err);

    /// Co-drafts `n_items` active slots (1 .. kDf2MaxSlots) in one batch: each slot's context update and 5-layer
    /// transformer body run on its own captured body graph (`Slot::body_exec_[n_new]`) into a packed batch buffer,
    /// then the draft head GEMM, top-16 reduction, and candidate selector run together across all slots with a
    /// single stream synchronization at the end.  Bitwise equal to calling draft() on each slot for `k_want` rows.
    /// `draft_batch_begin` queues all GPU work onto `stream()` without blocking the host; `draft_batch_end` waits
    /// for `stream()` and copies the selected tokens and probabilities into `items[idx].drafts` / `probs`.
    bool draft_batch_begin(const BatchItem* items, int n_items, float* stage_taps, size_t stage_stride, std::string& err);
    bool draft_batch_end(const BatchItem* items, int n_items, std::string& err);
    bool draft_batch(const BatchItem* items, int n_items, float* stage_taps, size_t stage_stride, std::string& err);
    /// Precomputes the window's DFlash2 context projections (`fc.weight` + 5-layer `k_proj`/`v_proj`) on `cs_` as soon
    /// as the target verifier finishes its last tap layer (layer 36), overlapping with target layers 37..47.
    void precompute_batch_context(const float* taps, int n_rows, const int64_t* pos);

    /// Captures the eight round graphs (n_new = 1..8) on `taps` now, so no round pays for a capture.
    bool prepare(const float* taps_dev, std::string& err);

    // ---- the drafter's context: save at park, restore at restore ----------------------------------
    // The ring's K/V cells, the absolute position every slot was written at, and the last committed
    // position: everything a parked conversation needs so its next turn starts with the drafter warm
    // instead of proposing from a cold (or, worse, another conversation's) ring. The blob self-describes:
    // magic, version, the geometry ints, the pack's path/size/mtime, `have`, and the kRing slot table;
    // restore_state() refuses a blob from another geometry or pack (the conversation starts cold).
    static constexpr uint32_t kStateMagic = 0x53324644;   // "DF2S"
    static constexpr uint32_t kStateVersion = 1;
    /// Host bytes one save_state() produces (header + slot table + L layers of planar K/V).
    size_t ring_elem_bytes() const { return fp16_ring_ ? 2 : 4; }
    void set_fp16_ring(bool on) { fp16_ring_ = on; }
    /// The header + slot table + rings arithmetic lives in dflash2_slots.hpp so the VRAM planner and the test read the same formula.
    size_t state_bytes() const { return (size_t) df2_state_bytes(ring_slots_, L_, NKV_, HD_, (int) ring_elem_bytes()); }
    int attention_window() const { return window_; }
    int ring_slots() const { return ring_slots_; }
    /// Dumps the ring (device quiesced: draft() synchronized before returning) into `out`.
    bool save_state(std::vector<uint8_t>& out, std::string& err, int64_t conversation_len = -1);
    /// Reinstalls a saved ring as the context of a conversation of `prefix_len` tokens: slots written at
    /// positions >= prefix_len (a branch this park took past the resumed checkpoint) are zeroed, every
    /// slot keeps the absolute position it was written at, `have_ctx_` becomes prefix_len.
    bool restore_state(const std::vector<uint8_t>& in, int64_t prefix_len, std::string& err);
    /// Copies `from_slot`'s ring up to `prefix_len` directly into `to_slot` on the device (`cs_`), avoiding a
    /// host `save_state`/`restore_state` round-trip on batch admission and slot restore.
    bool copy_slot(int from_slot, int to_slot, int64_t prefix_len, std::string& err);
    int64_t have_ctx() const { return cur().have_ctx_; }

    // ---- one drafter, several conversations (sizing and the cap in dflash2_slots.hpp) ------------------
    // A round's captured graph bakes the ring pointers AND the taps staging pointer it was recorded against, so
    // each slot owns its rings, its graph set, and the taps buffer those graphs read (Slot::taps_): launching
    // another slot's graph would draft against the wrong ring, and launching a graph on a different taps buffer
    // would read another buffer's rows.  By design all slots draft into ONE shared taps staging buffer (the
    // batch loop stages each slot's committed rows into it before the round), so one buffer serves every slot
    // and it stays valid for every slot's graphs until the engine rebuilds it - then each slot's own guard
    // fails until that slot recaptures.  Everything above - context, draft, reset, rewind, save_state,
    // restore_state, capture - acts on the active slot (slot() / set_slot); with the default single slot it is
    // all exactly as before.  set_n_slots grows the drafter to serve more conversations on the same card: it
    // allocates the extra slots' rings on the loaded device (or, before load, just records the count and lets
    // alloc() give every slot its rings), never past kDf2MaxSlots, and refuses to drop a slot that has drafted -
    // its ring holds a conversation.  A growth that runs out of VRAM partway is all-or-nothing: the pushed slots
    // are freed and the drafter keeps the slot count it had before the call.
    int n_slots() const { return (int) slots_.size(); }
    bool set_n_slots(int n, std::string& err);
    int slot() const { return cur_; }
    bool set_slot(int i, std::string& err);

    uint64_t vram_bytes() const { return vram_; }
    int64_t n_draft_vocab() const { return n_dvocab_; }
    int n_embd() const { return N_; }
    int device() const { return device_; }
    const std::vector<int>& tap_layers() const { return taps_; }
    cudaStream_t stream() const { return cs_; }
    /// Diagnostics (STRATA_DF2_DUMP): the final normalized block rows of the last draft(), host copy.
    bool debug_rows(std::vector<float>& rows, std::string& err);

    double ms_draft = 0, ms_gpu = 0, ms_ctx = 0, ms_capture = 0;
    int64_t rounds = 0;

private:
    /// One conversation's drafter state.  The captured round graphs live here with the ring because the graphs bake
    /// the ring pointers and the taps staging pointer they recorded: a ring needs its own graph set, and a batch
    /// loop that switched rings (or the taps buffer) under a shared graph would draft against another
    /// conversation's context.  All slots still share ONE taps staging buffer - the batch loop stages each slot's
    /// rows into it - so the pointer only moves when the engine rebuilds that buffer; each slot's taps_ then
    /// refuses rounds until that slot recaptures.
    struct Slot {
        float *kring_[8] = {}, *vring_[8] = {};   ///< ring K/V per layer (L <= 8), head-major [NKV][ring_slots][HD]
        cudaGraphExec_t exec_[kRows + 1] = {};   ///< this slot's round graphs (n_new = 1..8), captured against its own rings
        cudaGraphExec_t body_exec_[kRows + 1] = {}; ///< this slot's body-only graphs (context + 5 layers + hsel) for draft_batch
        const float* taps_ = nullptr;            ///< the taps staging buffer this slot's exec_ graphs were captured on; draft() refuses another
        size_t ring_plane_bytes_ = 0;            ///< bytes one ring plane (K or V, one layer) was cudaMalloc'd with; free_slot subtracts exactly these from vram_
        int64_t have_ctx_ = 0;                   ///< last committed position (exclusive) for this conversation
        /// The absolute position every ring slot was last written at (-1: never). Host-side mirror of the
        /// device writes; maintained by context()/draft(), saved/restored with the ring (see save_state).
        /// Sized from ring_slots_ wherever a slot comes to exist - load() sizes every slot, set_n_slots() sizes
        /// each one it adds (df2_ring_mirror_entries) - and starts empty, so a slot whose sizing was missed
        /// cannot hide behind a default that happens to fit (an 8k-window ring needs 8208, the default 2064 did not).
        std::vector<int32_t> ring_pos_;
    };
    Slot& cur() { return *slots_[(size_t) cur_]; }
    const Slot& cur() const { return *slots_[(size_t) cur_]; }
    bool alloc_slot_rings(Slot& sl);   ///< cudaMalloc + zero this slot's rings on device_ (vram_ grows); unwinds itself on failure
    void free_slot(Slot& sl);          ///< destroy this slot's graphs, free its rings (vram_ shrinks by the bytes recorded at allocation)
    int32_t& mirror_cell(Slot& sl, int64_t pos);  ///< sl.ring_pos_[pos % ring_slots_] with the bounds asserted (df2_ring_mirror_index)

    struct Tensor { std::string name, dtype; int64_t rows = 0, cols = 0; uint64_t off = 0, bytes = 0; };
    const Tensor& find(const std::string& name) const;
    const void* q8(const std::string& name) const { return blob_ + find(name).off; }
    const float* f32(const std::string& name) const { return reinterpret_cast<const float*>(blob_ + find(name).off); }
    /// planar Q8 matrices (1..3 of them, same input width) times the planar activations in xqp_ / xdp_ (ncols columns)
    void gemm(std::initializer_list<const char*> names, std::initializer_list<float*> outs, int ncols, cudaStream_t cs, const uint8_t* xq, const float* xd);
    bool record_context(const float* taps, int n, cudaStream_t cs, std::string& err);
    bool record_block_body(cudaStream_t cs, std::string& err);
    bool record_block_head(cudaStream_t cs, std::string& err);
    bool record_block(cudaStream_t cs, std::string& err);
    bool capture(int n, const float* taps, std::string& err);
    bool capture_body(int n, const float* taps, std::string& err);
    bool alloc(std::string& err);
    struct Prof;
    Prof* prof_ = nullptr;
    void stamp(const char* name);
    void prof_collect();
    void prof_report() const;
    void prof_free();

    int device_ = -1;
    cudaStream_t cs_ = nullptr;
    std::vector<Tensor> tensors_;
    uint8_t* blob_ = nullptr;
    uint64_t vram_ = 0;
    int N_ = 2560, L_ = 5, NH_ = 32, NKV_ = 8, HD_ = 128, I_ = 7680, window_ = kDefaultWindow, ring_slots_ = kRing, group_ = 16, rank_ = 256, mask_token_ = 0;
    float theta_ = 10000.0f, eps_ = 1e-6f;
    std::vector<int> taps_;
    int64_t vocab_ = 0;

    // target pieces
    EmbedFn embed_;
    const void* head_w_ = nullptr;
    int head_type_ = -1;
    int64_t n_dvocab_ = 0;
    int32_t* dvocab_ = nullptr;
    uint8_t* dhead_ = nullptr;      ///< the draft vocabulary's head rows (owned), or null with head_w_ the whole head
    bool bound_ = false;

    // device buffers
    int32_t* d_meta_ = nullptr;     ///< [0..7] context positions, [8..15] block positions, [16] anchor token, [17] anchor position, [24..31] block tokens
    int32_t* h_meta_ = nullptr;     ///< host staging of d_meta_ (mapped; m_meta_ is its device alias)
    int32_t* m_meta_ = nullptr;
    int32_t* h_out_ = nullptr;      ///< pinned: 7 draft tokens then 7 probabilities
    float* hp_out_ = nullptr;
    float *rt_ctx_ = nullptr, *rt_blk_ = nullptr;
    uint8_t* xq_ = nullptr;         ///< the head's q8_1 activations
    uint8_t* xqp_ = nullptr;        ///< planar activations of the drafter's own GEMMs (kernels/dflash2.hpp)
    float* xdp_ = nullptr;
    uint8_t* xqc_ = nullptr;        ///< the same for the context update, which runs beside the block
    float* xdc_ = nullptr;
    cudaStream_t cs2_ = nullptr;    ///< the context update's branch of the round graph
    cudaEvent_t ev_fork_ = nullptr, ev_join_ = nullptr, ev_ctx_[8] = {};
    bool fork_ = false;             ///< recording the forked graph: layer l's attention waits for ev_ctx_[l]
    float *fcy_ = nullptr, *hctx_ = nullptr, *kc_ = nullptr, *vc_ = nullptr;
    const float* precomp_taps_ = nullptr;
    int precomp_rows_ = 0;
    int64_t precomp_pos_[32] = {};
    float *h_ = nullptr, *xn_ = nullptr, *xc_ = nullptr, *dyn_ = nullptr, *q_ = nullptr, *qn_ = nullptr, *k_ = nullptr, *v_ = nullptr, *kblk_ = nullptr;
    float *ao_ = nullptr, *o_ = nullptr, *gate_ = nullptr, *up_ = nullptr, *mlp_ = nullptr, *part_ = nullptr, *fn_ = nullptr;
    float *fn_batch_ = nullptr, *hsel_batch_ = nullptr;
    float *logits_ = nullptr, *topval_ = nullptr, *hsel_ = nullptr, *out_prob_ = nullptr;
    int32_t *toptok_ = nullptr, *out_tok_ = nullptr;
    uint32_t* topscratch_ = nullptr;
    std::vector<std::unique_ptr<Slot>> slots_;   ///< one per conversation; the constructor makes slot 0
    int cur_ = 0;                                ///< the active slot (set_slot); each slot keeps its own captured taps pointer (Slot::taps_)
    std::string pack_path_;          ///< the pack's path at load (restore refuses a moved pack)
    uint64_t pack_size_ = 0;         ///< the pack's size and mtime at load: a swapped pack is a foreign blob
    int64_t pack_mtime_ = 0;
    cudaEvent_t ev_a_ = nullptr, ev_b_ = nullptr;
    bool timing_ = false;
    bool fp16_ring_ = false;   ///< opt-in only: the FP16 K/V ring changes DFlash2 draft numerics
};

}  // namespace strata::core
