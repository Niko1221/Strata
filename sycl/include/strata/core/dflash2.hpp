// sycl/include/strata/core/dflash2.hpp - the DFlash2 drafter for Qwen3.8-Flash-Next on an Intel GPU (SYCL).  Port of include/strata/core/dflash2.hpp; see that
// file for the model.  Differences:
//   * Q4_0 pack only; no MTP beside it (--draft-fallback none) and no shared head.
//   * One in-order queue (no context/block fork).  The round is one sycl graph (per context-row count n = 1..8) of compute kernels only; the host stages the
//     positions, the anchor and the rope tables (computed on the host: a device FP64 kernel slows every later kernel on an Arc Alchemist) with one memcpy
//     before the graph and takes the 7 picks back with one memcpy after it (a kernel that touches the engine's uncached host memory does the same).
#pragma once

#include <sycl/sycl.hpp>
#include <dpct/dpct.hpp>

#include "strata/core/session.hpp"

#include <cstdint>
#include <functional>
#include <initializer_list>
#include <string>
#include <vector>

namespace strata::core {

class NativeHead;

class Dflash2Drafter {
public:
    static constexpr int kRows = 8;           ///< block rows: the anchor + 7 masks
    static constexpr int kDrafts = kRows - 1;
    static constexpr int kRing = 2064;        ///< K/V slots per layer: the 2048 window + slack for rows written ahead of a rejection

    Dflash2Drafter() = default;
    ~Dflash2Drafter();
    Dflash2Drafter(const Dflash2Drafter&) = delete;
    Dflash2Drafter& operator=(const Dflash2Drafter&) = delete;

    /// Reads the pack (tools/dflash2_pack.py --q4_0) to the GPU `device` and allocates the ring and the buffers (about 0.5 GB of weights, 0.1 GB of buffers).
    bool load(const std::string& pack_path, int device, std::string& err);
    /// The target's embedding and head (and the draft vocabulary file, "" or missing: the whole head).
    bool bind(const WeightTable& wt, const NativeHead* head, const std::string& vocab_file, std::string& err);
    using EmbedFn = std::function<void(const int32_t* tokens_dev, int n, float* out, sycl::queue*)>;
    bool bind_raw(EmbedFn embed, const void* head_w, int head_type, int64_t head_rows, const std::vector<int32_t>& vocab, std::string& err);

    /// A new request: the ring is empty again.
    void reset() { have_ctx_ = 0; hist_.clear(); ++epoch_; ring_hi_ = 0; }
    /// Serve: a request starts with prompt `ids` (n tokens) and the session continues at `resume`; see the CUDA header.
    bool begin_request(const int32_t* ids, int64_t n, int64_t resume, bool* cold);
    /// Serve: rows [pos0, pos0 + n) were committed with these tokens.
    void note_rows(const int32_t* toks, int64_t n, int64_t pos0);
    /// Context rows without a draft: `n` rows of taps (device, [n][5 * n_embd]) for positions pos0 ... pos0 + n - 1.
    bool context(const float* taps_dev, int64_t n, int64_t pos0, std::string& err);
    /// One round: the new context rows (n_new in 1..8: the committed window rows, taps device [n_new][5 * n_embd], positions pos0 ...),
    /// then the block for `anchor` (the last committed token, at pos0 + n_new).  `drafts` / `probs` get kDrafts entries.  The taps pointer must be the
    /// same device buffer on every call (graphs are captured on it).
    bool draft(const float* taps_dev, int n_new, int64_t pos0, int32_t anchor, int32_t* drafts, float* probs, std::string& err);

    /// Captures the eight round graphs (n_new = 1..8) on `taps` now, so no round pays for a capture.
    bool prepare(const float* taps_dev, std::string& err);

    /// The most drafts a window will use (the engine's widest window - 1, 1..kDrafts; default kDrafts).  The head, the top-k and the selector's walk run for
    /// that many rows only (DF-04): the first `n` drafts and probabilities are bitwise the ones a full block makes; draft() returns token 0 with
    /// probability 0 for the rest.  Before prepare() / the first draft(); a later change drops the captured graphs.
    void set_max_drafts(int n);

    uint64_t vram_bytes() const { return vram_; }
    int64_t n_draft_vocab() const { return n_dvocab_; }
    int n_embd() const { return N_; }
    const std::vector<int>& tap_layers() const { return taps_; }
    sycl::queue* stream() const { return cs_; }

    double ms_draft = 0, ms_gpu = 0, ms_ctx = 0, ms_capture = 0;
    int64_t rounds = 0;

private:
    struct Tensor { std::string name, dtype; int64_t rows = 0, cols = 0; uint64_t off = 0, bytes = 0; };
    const Tensor& find(const std::string& name) const;
    const float* f32(const std::string& name) const { return reinterpret_cast<const float*>(blob_ + find(name).off); }
    void gemm(std::initializer_list<const char*> names, std::initializer_list<float*> outs, int ncols, const uint8_t* xq, const float* xd);
    bool record_context(const float* taps, int n, std::string& err);
    bool record_block(std::string& err);
    bool capture(int n, const float* taps, std::string& err);
    bool alloc(std::string& err);
    /// the host's staging of one launch: positions and rope tables (rows 0..7 context, 8..15 block), the anchor, then ONE copy to the device
    void stage(int n_ctx, int64_t pos0, int block_n, int64_t block_s, int32_t anchor);

    int device_ = -1;
    sycl::queue* cs_ = nullptr;
    std::vector<Tensor> tensors_;
    uint8_t* blob_ = nullptr;
    uint64_t vram_ = 0;
    int N_ = 2560, L_ = 5, NH_ = 32, NKV_ = 8, HD_ = 128, I_ = 7680, window_ = 2048, group_ = 16, rank_ = 256, mask_token_ = 0;
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
    int32_t* d_meta_ = nullptr;     ///< ints [0..7] context positions, [8..15] block positions, [16] anchor token, [17] anchor position, [18] epoch, [24..31] block tokens; then 16 rope tables of 128 floats
    int32_t* h_meta_ = nullptr;     ///< host staging of d_meta_ (the engine's uncached host memory: only memcpy touches it)
    float* rt_ctx_ = nullptr;       ///< d_meta_ + 32 (as floats); rt_blk_ = rt_ctx_ + 8 * 128
    float* rt_blk_ = nullptr;
    int32_t* d_out_ = nullptr;      ///< the select kernel's 7 picks, then 9 pad, then 7 probabilities
    int32_t* h_out_ = nullptr;
    float *kring_[8] = {}, *vring_[8] = {};
    uint8_t* xq_ = nullptr;         ///< the head's q8_1 activations
    uint8_t* xqp_ = nullptr;        ///< planar activations of the drafter's own GEMMs (kernels/dflash2.hpp)
    float* xdp_ = nullptr;
    uint8_t* xqc_ = nullptr;        ///< the same for the context update
    float* xdc_ = nullptr;
    float *fcy_ = nullptr, *kc_ = nullptr, *vc_ = nullptr;
    float *h_ = nullptr, *xn_ = nullptr, *dyn_ = nullptr, *q_ = nullptr, *qn_ = nullptr, *k_ = nullptr, *v_ = nullptr, *kblk_ = nullptr;
    float *o_ = nullptr, *gate_ = nullptr, *up_ = nullptr, *mlp_ = nullptr, *part_ = nullptr, *fn_ = nullptr;
    float *logits_ = nullptr, *topval_ = nullptr, *hsel_ = nullptr;
    int32_t* toptok_ = nullptr;
    uint32_t* topscratch_ = nullptr;
    dpct::experimental::command_graph_exec_ptr exec_[kRows + 1] = {};
    const float* graph_taps_ = nullptr;
    int ndraft_ = kDrafts;          ///< drafts a window can use (set_max_drafts)
    int64_t have_ctx_ = 0;
    std::vector<int32_t> hist_;     ///< the token of every position the ring was filled for (serve)
    int64_t ring_hi_ = 0;           ///< one past the highest position written in this epoch
    int32_t epoch_ = 0;             ///< stamped into the ring tags: a new epoch makes every older slot invalid
    int64_t* tag_ = nullptr;        ///< per ring slot: (epoch << 32 | position) of the row it holds, -1 none
    bool timing_ = false;
};

}  // namespace strata::core
