// include/strata/core/verify.hpp - plan v0.3 P6: the speculative VERIFY window.
//
// T tokens at consecutive positions p0 .. p0+T-1 - the last accepted token and T-1 drafts - go through all 48
// layers in ONE captured graph, and the head's argmax is produced for every one of them.  Token t's argmax is
// what plain greedy decode would produce after token t, BIT FOR BIT: every kernel here is either the single-token
// kernel applied per token, or a multi-token kernel whose per-token arithmetic is the single-token kernel's
// (multi-column MMVQ in exact mode, the T-token GDN kernels, the per-token hit activation, the multi-token CPU
// expert rows).  So a draft is accepted exactly when greedy decode would have produced it.
//
// What the window costs is the dense weights read ONCE for T tokens and the union of the T tokens' missed
// experts on the CPU (measured on decode traces: 1.75x one token's misses for T=2, 2.4x for 3, 3.05x for 4).
//
// STATE.  The window appends K/V and indexer keys for all T positions and leaves the GDN state untouched.
// `commit(n_keep)` then makes the first `n_keep` tokens permanent: the GDN conv history and recurrent state are
// advanced by replaying those tokens from inputs the window stored, the indexer's key tail is restored from a
// snapshot and the accepted keys re-appended (a rejected key can land in a slot the current block still needs),
// and the PLE history is set to its snapshot after token n_keep-1.  K/V cells past the accepted prefix are simply
// overwritten when those positions are processed again, before any query can read them.
//
// EXPERTS.  The main GPU computes the routed experts its VRAM tier holds, decided in the graph from a snapshot of the
// residency table that the pool reads too (`residency`), so neither waits for the other; its shared expert runs on a
// branch of its own, forked before the router, and the wait for the CPU's rows on another.  The pool computes the rest
// on the CPU and the second GPU, and with a PCIe share publishes the missed experts the GPU reads over PCIe.  With the
// native router and a native pack the router kernel writes their input in the forms they take: ggml's Q8_K rows for
// the CPU (`pool_takes_q8k` layers) and q8_1 rows for the GPUs, so neither quantizes it.
//
// Requires the default native decode configuration (native projections, fused GR, fused GDN, fast attention and
// selection, native indexer, QSA norms, RoPE, combine, shared expert gate and PLE block) and a profile-filled VRAM
// expert tier.
#pragma once

#include "strata/core/expert_source.hpp"
#include "strata/core/layer.hpp"
#include "strata/core/session.hpp"
#include "strata/core/split_head.hpp"
#include "strata/kernels/sampler.hpp"

#include <cuda_runtime.h>

#include <cstdint>
#include <string>
#include <vector>

namespace strata::core {

class NativeHead;

/// The CPU pool for a window: x_f (n_tok, n_embd), ids (n_tok, k) -> out (n_tok * k, n_embd): the CPU's rows, the
/// main GPU's zeroed unless `GpuPlanSink::host_rows_only`; a second GPU writes its own (`set_gpu2`).
using PoolMultiFn = void (*)(void* user, const float* x_f, const int32_t* ids, int64_t n_tok, int64_t k, float* out,
                             int64_t layer);
/// False, with the reason, when work the window waits for failed (the second GPU's).
using WatchFn = bool (*)(void* user, std::string& err);
/// Layer `layer`'s likely experts, its router applied to an estimate of its FFN input (`Verifier::set_predict`): ids
/// and weights (n_tok, k), host memory.
using PredictFn = void (*)(void* user, int64_t layer, const int32_t* ids, const float* w, int64_t n_tok, int64_t k);
/// The window's last CPU rows are in.
using TailFn = void (*)(void* user);
/// Layer `layer`'s CPU rows are in (`Verifier::set_rows_in`).
using RowsFn = void (*)(void* user, int64_t layer);

/// The main GPU's VRAM expert tier.
struct VerifyHits {
    const int32_t* res = nullptr;          ///< host [n_layers * n_expert]: slot or -1, the live residency table
    const uint8_t* cache_base = nullptr;   ///< slot 0 of the VRAM expert arena
    const uint64_t* slot_off = nullptr;    ///< [slots]: each slot's byte offset from `cache_base`; null: slot * blob
    int64_t slots = 0;
    int64_t blob = 0;
};

class Verifier {
public:
    Verifier() = default;
    ~Verifier();
    Verifier(const Verifier&) = delete;
    Verifier& operator=(const Verifier&) = delete;

    /// `max_t` <= kVerifyMaxT.  `head` may be null (the canonical head is then run per token).
    bool init(const WeightTable& wt, const ModelGeometry& g, SessionState& ss, const VerifyHits& hits,
              const NativeHead* head, int max_t, std::string& err);

    /// One window: `tokens[0..T)` at positions pos0.., the pool served per layer; `out[t]` = argmax after token t.
    /// The PLE rows follow `ss.ple_prev` and the tokens: read while layer 0 runs (the graph waits for them before
    /// layer 1), unless `ple_ahead` already started them.  Captures the T-token graph on first use.
    bool run(int T, const int32_t* tokens, int64_t pos0, PoolMultiFn pool, void* user, int32_t* out, std::string& err);
    /// Starts reading the PLE rows of the next window's token t (0, 1, ... in order, after `commit`): `run` takes
    /// them if its token t, and the ones before it, are these.  For the drafts, while the draft layer runs.
    void ple_ahead(int t, int32_t token);

    /// The sampling the head applies (temperature / top_p / top_k / min_p / seed), per request; greedy by default.
    /// It runs after the captured graph (a captured sampler would bake its parameters in), so it can change between
    /// requests freely: row t of a window at pos0 draws Philox(seed, pos0 + t), tied to the position it samples, not
    /// to how the text was cut into windows.  A rejected row's draw is discarded.
    void set_sampling(const strata::kernels::SamplerParams& sp) { sampling_ = sp; }
    /// The penalty histories for `penalty_last_n`: one row per window row, T rows of `history_len` int32 slots
    /// (`strata::kernels::penalty_rows`), device memory staged before every window; null: no penalties.
    void set_history(const int32_t* history, int history_len) { hist_d_ = history; hist_len_ = history_len; }
    /// Off: `run` skips the head sampling and `out` is the greedy pick (windows whose picks are discarded: a prompt
    /// read through windows commits every token).
    void set_head_sampling(bool on) { head_sampling_ = on; }

    /// Keep the first `n_keep` (1..T) tokens of the last window; advances `ss.ple_prev` by them.  With `wait` false
    /// the commit graph is only launched (the MTP draft, on its own stream, reads nothing it writes): `wait_commit`
    /// before anything outside the next `run` reads the sequence state.
    bool commit(int n_keep, std::string& err, bool wait = true);
    bool wait_commit(std::string& err);

    /// The residency the window decides the main GPU's hits by: `VerifyHits::res` as `init` and the last `run` found
    /// it.  The pool reads this one (`ExpertDispatch::host_res`), so it leaves exactly the GPU's rows to the GPU.
    const int32_t* residency() const { return h_res_; }

    /// Token t's residual after the last layer, (hc, n_embd) on the device, valid until the next `run`.
    const float* final_R(int t) const;
    const float* final_R_all() const { return R_; }
    /// With `set_host_rows(true)` (before the first `run`), the same rows in host memory once `run` returns: the
    /// graph writes them there itself, so reading them waits for no copy engine.
    void set_host_rows(bool on) { host_rows_ = on; }
    const float* final_R_host() const { return h_rows_; }
    /// The head's logits of the last window's first `T` tokens, T * n_vocab floats, copied to host memory `out`.
    bool copy_logits(int T, float* out, std::string& err) const;
    /// The head's rows [split, n_vocab) on another GPU (`head` then holds rows [0, split)): the window hands the head's
    /// input over, and `run` keeps each token's larger pick.  Set before `init`.
    void set_split_head(SplitHead* sh) { shead_ = sh; }
    /// Whole logits rows after every window (copy_logits, --window-logits): with a split head the second GPU's part
    /// then copies its rows back.  Set before `init`.
    void set_logits_wanted(bool on) { logits_wanted_ = on; }

    /// Where the pool publishes each layer's PCIe share of the misses; give it to the dispatch
    /// (`ExpertDispatch::plan`) before the first `run`.
    GpuPlanSink* plan_sink() { return &sink_; }
    /// Plan v0.3 P6: split the window into two token groups and pipeline the CPU experts of one with the GPU work
    /// of the other (default on).  Set before the first `run`.
    void set_split(bool on) { split_ = on; }
    /// Plan v0.3 P6: how the PCIe share of the misses reaches the GPU: 0 = DMA into staging (the copy engine works
    /// beside the CPU; best when the CPU is compute-bound, the i-quants), 1 = the grouped kernel reads the mapped
    /// arena directly, 2 = a copy kernel stages it inside the graph (no API calls on the pool's thread; best when
    /// the CPU is RAM-bound, Q2_0).  Set before the first `run`.
    void set_pcie_mode(int mode) { sink_.pcie_mode = mode; }
    /// Whether the pool may give the GPU a PCIe share of the misses (`--pcie-frac` > 0); without one the window
    /// has no PCIe stage.  Set before the first `run`.
    void set_pcie_share(bool on) { pcie_share_ = on; sink_.pcie = on; }
    /// Diagnostics: GPU timestamps between the stages of every layer (`--window-profile`; each costs a kernel
    /// launch, ~2 us).  Set before the first `run`; `print_profile` writes the per-stage table to stdout.
    void set_profile(bool on) { profile_ = on; }
    void print_profile() const;
    /// Each layer's router also runs on an estimate of its FFN input from the previous layer's state, on a side branch
    /// of the graph: the layer's FFN read of that state, its gates their running average over the windows, and the
    /// router's logits corrected by a running average of their error.  Once the CPU's rows of a layer are in, `fn`
    /// gets the next layer's prediction (the RAM is idle until the next ring), waited for when it comes later
    /// (`predict_late`).  Without `learn` the averages stay at their start (gates of one half, no correction), so the
    /// prediction depends on the window alone.  Windows of one token group.  Set before the first `run`.
    void set_predict(PredictFn fn, void* user, bool learn) { predict_ = fn; predict_user_ = user; learn_ = learn; }
    /// `fn` runs once the window's last CPU rows are in: the RAM stays idle until the next window's first pool (the
    /// head, the commit, the drafts).  The main GPU may still run the window then: `window_done` follows its graph.
    /// Set before the first `run`.
    void set_tail(TailFn fn, void* user) { tail_ = fn; tail_user_ = user; }
    /// `fn` runs once each layer's CPU rows are in and the next layer's prediction has gone out, but the last's (the
    /// tail's): the main GPU's link then has nothing to carry until the next layer's router writes its rows to the
    /// host.  Set before the first `run`.
    void set_rows_in(RowsFn fn, void* user) { rows_in_ = fn; rows_user_ = user; }
    cudaEvent_t window_done() const { return done_; }
    /// The pool gives a second GPU a share of each layer (`GpuPlanSink::gpu2_flag`): it writes its rows into the
    /// pool's rows and raises its token group's flag, and a branch of the window, forked at the ring, takes them
    /// into VRAM while the CPU works.  Set before the first `run`.
    void set_gpu2(bool on) { gpu2_ = on; }
    /// `fn` is asked every ~2 ms while the host waits for a ring: on failure the window's waits are released and `run`
    /// returns the reason.  Set before the first `run`.
    void set_watch(WatchFn fn, void* user) { watch_ = fn; watch_user_ = user; }

    double ms_wait = 0, ms_pool = 0, ms_host = 0, ms_commit = 0, ms_predict = 0;
    int64_t windows = 0, predict_late = 0;
    /// Layers whose second-GPU rows came after the CPU's, and the host time from the CPU's rows to theirs.
    int64_t late2 = 0;
    double ms_late2 = 0;

private:
    bool capture(int T, std::string& err);
    bool capture_commit(std::string& err);
    bool record_window(int T, cudaStream_t cs, std::string& err);

    strata::kernels::SamplerParams sampling_ = [] {
        strata::kernels::SamplerParams s;
        s.greedy = true;
        s.temperature = 0.0f;
        return s;
    }();
    const int32_t* hist_d_ = nullptr;
    int hist_len_ = 0;
    bool head_sampling_ = true;

    const WeightTable* wt_ = nullptr;
    const ModelGeometry* g_ = nullptr;
    SessionState* ss_ = nullptr;
    VerifyHits hits_;
    const NativeHead* head_ = nullptr;
    int max_t_ = 0;
    int last_t_ = 0;
    int64_t last_pos0_ = 0;
    int32_t last_tokens_[8] = {};
    int64_t n_vocab_ = 0;
    cudaStream_t cs_ = nullptr;
    cudaGraphExec_t exec_[9] = {};
    cudaGraphExec_t commit_exec_ = nullptr;
    bool commit_pending_ = false;

    // mapped staging (host pointer, device alias)
    int32_t* h_tok_ = nullptr;   int32_t* m_tok_ = nullptr;     // T
    int32_t* h_step_ = nullptr;  int32_t* m_step_ = nullptr;    // T * kStepCount
    int32_t* h_pos_ = nullptr;   int32_t* m_pos_ = nullptr;     // T * n_head
    int32_t* h_commit_ = nullptr; int32_t* m_commit_ = nullptr; // [n_keep, n_keep-1, pos_0 .. pos_{T-1}]
    float* h_ple_ = nullptr;     float* m_ple_ = nullptr;       // T * n_embd
    int32_t* h_out_ = nullptr;   int32_t* m_out_ = nullptr;     // T argmax ids
    float* h_val_ = nullptr;     float* m_val_ = nullptr;       // their logits (a split head's merge)
    SplitHead* shead_ = nullptr;
    bool logits_wanted_ = false;
    int64_t head_rows_ = 0;                                     // the head's rows here: n_vocab, or the split
    float* head_full_ = nullptr;                                // T whole logits rows (a split head, when wanted)
    uint32_t hand_n_ = 0, done_n_ = 0;                          // the split head's counts so far
    float* h_x_ = nullptr;       float* m_x_ = nullptr;         // doorbell payload: T * n_embd
    int32_t* h_ids_ = nullptr;   int32_t* m_ids_ = nullptr;     // T * k
    float* h_w_ = nullptr;       float* m_w_ = nullptr;         // T * k
    uint32_t* h_seq_ = nullptr;  uint32_t* m_seq_ = nullptr;
    uint32_t* h_flag_ = nullptr; uint32_t* m_flag_ = nullptr;
    uint32_t* h_flagA_ = nullptr; uint32_t* m_flagA_ = nullptr;  // the PCIe share's plan is in place
    uint32_t* h_flagB_ = nullptr; uint32_t* m_flagB_ = nullptr;  // the PCIe share's DMA copies have landed
    uint32_t* h_pleflag_ = nullptr; uint32_t* m_pleflag_ = nullptr;  // the PLE rows are in h_ple_
    bool host_rows_ = false;
    float* h_rows_ = nullptr;    float* m_rows_ = nullptr;      // T * hc * n_embd: the final rows (set_host_rows)
    int32_t ahead_tok_[8] = {};                                   // the tokens ple_ahead read for
    cudaStream_t copy_ = nullptr;                                 // the copy engine's stream (DMA of missed experts)
    struct FlagSet { uint32_t* flag; uint32_t value; };
    FlagSet flag_sets_[2 * 64 * 2] = {};                          // host-function arguments, one per (layer, group)
    static void fetch_dma(void* ctx, const uint8_t* const* src, int n, size_t bytes);
    static void raise_flag(uint32_t* flag, uint32_t value);
    int32_t* h_plan_ = nullptr;  int32_t* m_plan_ = nullptr;     // the PCIe share's plan block per token group
    int64_t plan_i32_ = 0, plan_ptr_off_ = 0, plan_stride_ = 0;   // a plan block's int32 words, its ptr field, the
                                                                  // device blocks' stride (see `init`)
    int32_t* h_res_ = nullptr;   int32_t* m_res_ = nullptr;      // the residency snapshot (`residency`)
    int64_t res_words_ = 0;
    GpuPlanSink sink_;
    uint32_t cur_layer_ = 0;
    static void publish_plan(void* ctx);
    void set_plan_slot(int grp);
    bool split_ = false;   // opt-in (--spec-split): exact but slower, see the overlap study
    bool pcie_share_ = true;
    int groups_[9] = {};
    float* h_ymiss_ = nullptr;   float* m_ymiss_ = nullptr;     // T * k * n_embd
    // the FFN input as the router kernel writes it (quant_router_): Q8_K rows `xk_stride_` bytes apart for the layers
    // the pool takes them (xk_layer_), q8_1 rows for the second GPU
    bool quant_router_ = false;
    std::vector<uint8_t> xk_layer_;
    int64_t xk_stride_ = 0;
    uint8_t* h_xk_ = nullptr;    uint8_t* m_xk_ = nullptr;      // T rows
    uint8_t* h_xq1_ = nullptr;   uint8_t* m_xq1_ = nullptr;     // T rows of n_embd / 32 blocks
    // the second GPU's share (set_gpu2), per token group: the flag it raises (64 bytes apart) and its entries,
    // [n, pad x3, entries] (mapped; `list2_` their device copies, which the combine reads)
    bool gpu2_ = false;
    uint32_t* h_flag2_ = nullptr; uint32_t* m_flag2_ = nullptr;
    int32_t* h_list2_ = nullptr; int32_t* m_list2_ = nullptr;
    int32_t* list2_ = nullptr;
    int64_t list2_stride_ = 0;
    WatchFn watch_ = nullptr;
    void* watch_user_ = nullptr;
    // the next layer's predicted experts: two halves by layer parity, so a prediction stays put until the host has
    // taken it; the counter holds the predictions published in this window
    PredictFn predict_ = nullptr;
    void* predict_user_ = nullptr;
    bool learn_ = false;
    TailFn tail_ = nullptr;
    void* tail_user_ = nullptr;
    RowsFn rows_in_ = nullptr;
    void* rows_user_ = nullptr;
    int32_t* h_pids_ = nullptr;  int32_t* m_pids_ = nullptr;     // 2 x T * k
    float* h_pw_ = nullptr;      float* m_pw_ = nullptr;         // 2 x T * k
    uint32_t* h_pseq_ = nullptr; uint32_t* m_pseq_ = nullptr;
    // short branches beside the main stream: on side_ the GDN gates or the QSA indexer (the mixer) and the experts'
    // activations (beside the router), then the prediction; on shs_ the GDN gate projection or the QSA queries (the
    // mixer), then the shared expert
    cudaStream_t side_ = nullptr;
    cudaEvent_t join_ = nullptr;                                  // the prediction's
    cudaEvent_t bfork_ = nullptr, bjoin_ = nullptr;               // side_'s others
    cudaEvent_t pfork_ = nullptr, pjoin_ = nullptr;               // shs_'s projections
    cudaStream_t shs_ = nullptr;                                  // the shared expert's branch (and the snapshot's copy)
    cudaEvent_t shfork_ = nullptr, shjoin_ = nullptr, res_ready_ = nullptr;
    cudaStream_t g2s_ = nullptr;                                  // the second GPU's rows, per token group
    cudaEvent_t g2fork_ = nullptr, g2join_[2] = {};
    cudaStream_t cps_ = nullptr;                                  // the wait for the CPU's rows, per token group
    cudaEvent_t cpfork_ = nullptr, cpjoin_[2] = {};
    cudaStream_t kst_ = nullptr;                                  // the QSA mixer's keys
    cudaEvent_t kjoin_ = nullptr, iproj_ = nullptr;               // and the end of its indexer's projections
    cudaStream_t sgs_ = nullptr;                                  // the shared expert's scalar gate
    cudaEvent_t sgjoin_ = nullptr;
    cudaEvent_t efork_ = nullptr, rdone_ = nullptr;               // the FFN read's end, the router's
    cudaEvent_t done_ = nullptr;                                  // after the last window's graph

    // device
    void* arena_ = nullptr;
    int32_t *tok_ = nullptr, *step_ = nullptr, *pos_ = nullptr, *commit_ = nullptr;
    float *ple_ = nullptr, *emb_ = nullptr, *R_ = nullptr, *mixed_ = nullptr, *bo_ = nullptr;
    float *inj_ = nullptr, *inj2_ = nullptr, *lo_ = nullptr, *rs_ = nullptr;
    float* grs_ = nullptr;                                    // the hyper-connection reads' scratch (zeroed with the arena)
    uint8_t* xq_ = nullptr;                                   // T columns of q8_1
    uint8_t* xil_ = nullptr;                                  // their interleaved copy (2+ tokens)
    float *qkv_L_ = nullptr, *h_L_ = nullptr, *gate_L_ = nullptr, *beta_L_ = nullptr;   // per GDN layer
    float *z_ = nullptr, *y_ = nullptr, *y_dummy_ = nullptr;
    float *qfull_ = nullptr, *qcur_ = nullptr, *kcur_ = nullptr, *vcur_ = nullptr, *idx_raw_L_ = nullptr;
    float *qidx_ = nullptr, *scores_ = nullptr, *attn32_ = nullptr, *attn_scratch_ = nullptr;
    float* tail_snap_ = nullptr;                              // per QSA layer
    int32_t* sel_ = nullptr;
    float *logits_ = nullptr, *w_ = nullptr, *shared_ = nullptr, *hit_out_ = nullptr;
    int32_t *ids_ = nullptr, *hit_slot_ = nullptr, *hit_dst_ = nullptr, *hit_count_ = nullptr;
    unsigned* ring_count_ = nullptr;                          // verify_router's block counters: the layer's, the prediction's
    float* plogits_ = nullptr;                                    // the predictions' logits, by layer parity
    float* pw_ = nullptr;                                         // the next layer's router (prediction)
    int32_t* pids_ = nullptr;
    float* pmixed_ = nullptr;                                     // its input: the next layer's FFN read, estimated
    float* gate_ema_ = nullptr;                                   // [layer][hc][n_embd]: the FFN reads' gates, averaged
    float* pbias_ = nullptr;                                      // [layer][n_expert]: the predictions' logit bias
    int32_t* plan_ = nullptr;                                     // device copies of the PCIe share's plan blocks
    int32_t* dplan_ = nullptr;                                    // the main GPU's hit plans, decided on the device
    int32_t* res_ = nullptr;                                      // the residency snapshot, copied as a window starts
    unsigned long long* slot_ptr_ = nullptr;                      // each VRAM slot's address
    uint8_t* staging_ = nullptr;                                  // VRAM slots for the PCIe share of the misses
    static constexpr int64_t kStagingBlobs = 16;
    uint8_t* hit_xq_ = nullptr;
    uint8_t* nat_xq_ = nullptr;   // plan v0.3 P6: q8_1 activations for a native pack's grouped experts
    float* hit_xs_ = nullptr;
    void* hit_scratch_ = nullptr;
    float *head_mixed_ = nullptr, *head_inj_ = nullptr, *head_logits_ = nullptr;
    uint8_t* arg_scratch_ = nullptr;   ///< argmax_rows' partials and counters
    int32_t* one_ = nullptr;           ///< device {1}: the n_keep of a one-token window, which commits itself
    float *sh_gate_ = nullptr, *sh_up_ = nullptr, *sh_g_ = nullptr;
    float* hist_snap_ = nullptr;                              // T * NG_HIST * NG_HC_DIM
    // the PLE block's projections (keys T * NG_HC_DIM, values T * n_embd) and the rest of the block's rows
    float *ple_key_ = nullptr, *ple_val_ = nullptr, *ple_nkey_ = nullptr, *ple_norm_ = nullptr, *ple_gated_ = nullptr;
    float* ple_gate_ = nullptr;
    int64_t cap_ = 0, max_blocks_ = 0, attn_scratch_floats_ = 0;

    // --window-profile: stamps [window size][layer, then one row for the window][stage]: a layer's main-stream stages,
    // then the ends of its shared expert's branch and of the second GPU's rows' branch
    static constexpr int kLayerStamps = 13, kStamps = 15;
    bool profile_ = false;
    unsigned long long* stamps_ = nullptr;
    std::vector<unsigned long long> h_stamps_;
    std::vector<double> prof_ns_;                             // [mixer kind (GDN, QSA)][stage], summed
    double prof_window_ns_[4] = {};                           // inputs + embeddings, head, whole window, stamp gap
    int64_t prof_windows_ = 0, prof_layers_[2] = {};
    static constexpr int kWaitBuckets = 7;                    // the wait for the CPU's rows, by bucket (us)
    static constexpr double kWaitEdges[kWaitBuckets - 1] = {6, 8, 12, 20, 50, 100};
    int64_t prof_wait_hist_[2][kWaitBuckets] = {};
};

}  // namespace strata::core
