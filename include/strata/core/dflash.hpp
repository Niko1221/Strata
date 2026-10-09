// include/strata/core/dflash.hpp - the standalone DFlash drafter's artifact.
//
// A DFlash artifact is a standalone GGUF v3 file holding a small dense block drafter (the DeepSpec
// DFlash design) for a fixed target: five transformer layers that turn a fused projection of the
// target's per-layer residual taps into a block of candidate tokens in ONE parallel pass.  The
// checkpoint ships neither an embedding nor an LM head - both are bound from the target at load.
//
//   PixelML/Qwen3.8-Flash-Next-NVFP4-DFlash: 5 layers, hidden 2560, GQA 24Q/2KV x 256, MLP 7680,
//   58 BF16 tensors, ~498M params, trained block (query count) 7, mask token 248077,
//   rope theta 1e7, target taps [3, 15, 23, 35, 43].
//
// The geometry is PARSED from the artifact (never hardcoded at the call sites) and validated
// twice: structurally here (metadata <-> tensor shapes agree, supported types, in bounds), and against what
// the runtime fast path supports (validate_supported) when the engine wires the drafter up.
// Semantics and the DeepSpec anchor layout: docs/DFLASH.md.
#pragma once

#include "strata/artifact/gguf_reader.hpp"
#include "strata/core/session.hpp"
#include "strata/core/vmm.hpp"

#include <array>
#include <cstdint>
#include <functional>
#include <memory>
#include <string>
#include <vector>

namespace strata::core {

class NativeHead;

/// The batch attention's identity selection, one [0, cap) row per query (testable; see
/// DFlashDrafter::upload).
void dflash_identity_fill(int32_t* host, int rows, int64_t cap);

/// Everything the artifact's metadata says about the drafter.  Defaults are the unloaded state;
/// `load` fills every field or fails.
struct DFlashGeometry {
    int64_t hidden = 0;                     ///< dflash.embedding_length (2560)
    int64_t layers = 0;                     ///< dflash.block_count (5)
    int64_t n_head = 0;                     ///< dflash.attention.head_count (24)
    int64_t n_head_kv = 0;                  ///< dflash.attention.head_count_kv (2)
    int64_t head_dim = 0;                   ///< dflash.attention.key_length (256)
    int64_t intermediate = 0;               ///< dflash.feed_forward_length (7680)
    int64_t vocab = 0;                      ///< dflash.vocab_size (248320)
    int64_t block_size = 0;                 ///< trained query count (7): max candidates per pass
    int64_t mask_token_id = -1;             ///< dflash.mask_token_id (248077), -1 = unset
    double rope_theta = 0;                  ///< dflash.rope.frequency_base (1e7)
    double rms_eps = 1e-6;                  ///< fixed by the architecture, not stored
    std::vector<int32_t> target_layers;     ///< dflash.target_layers, strictly increasing ([3,15,23,35,43])
    bool sample_from_anchor = true;         ///< DeepSpec anchor layout; false is refused at load
    bool causal = false;                    ///< block attention causality; true is refused at load
    int64_t markov_rank = 0;                ///< > 0 (DSpark Markov head) is refused at load
    int64_t selector_top_k = 0;             ///< > 0 (DFlash2 selector) is refused at load
    bool confidence_head = false;           ///< true (DSpark confidence gate) is refused at load
    int64_t fusion_in() const { return hidden * (int64_t) target_layers.size(); }
};

/// One device weight, row-major rows of `cols` values in its GGML type (the GGUF file layout, ne0 varies
/// fastest, is preserved: row r is output neuron r, y[r] = dot(row, x)).
struct DFlashTensor {
    std::string name;      ///< canonical Strata-side name ("layers.3.self_attn.q_proj")
    std::string file_name; ///< the name it carries in the GGUF (either naming family)
    int64_t rows = 0, cols = 0;
    int type = 30;        ///< GGML BF16 or quantized matrix type
    uint64_t bytes = 0;   ///< packed payload bytes
    const uint16_t* d = nullptr;   ///< device payload, valid after upload()
};

/// The parsed artifact: metadata, geometry and the tensor inventory of an opened GGUF.  No device
/// work happens here - `upload` moves the payloads in the runtime that owns a CUDA context.
class DFlashArtifact {
public:
    /// Opens `path`, parses the metadata and the tensor directory, resolves every required tensor
    /// (llama.cpp GGUF names and raw-HF names are both accepted) and checks each shape against the
    /// geometry the metadata declares.  Precise `err` on anything unexpected.
    bool open(const std::string& path, std::string& err);
    /// Target-independent check that the parsed geometry is one this runtime implements.
    /// The canonical fast path is the published PixelML Flash-Next drafter geometry.
    static bool validate_supported(const DFlashGeometry& g, std::string& err);

    bool loaded() const { return file_ != nullptr; }
    const std::string& path() const { return path_; }
    const DFlashGeometry& geom() const { return geom_; }

    /// Aligned packed payload bytes of every required tensor together (weights VRAM before scratch).
    uint64_t weight_bytes() const { return weight_bytes_; }
    /// The resolved tensor `name` (Strata-side names, e.g. "layers.0.self_attn.q_proj"), or nullptr.
    const DFlashTensor* tensor(const std::string& name) const;
    const std::vector<DFlashTensor>& tensors() const { return tensors_; }

    /// The host address of `t`'s payload inside the mmap (upload() copies from here).
    const uint8_t* host_data(const DFlashTensor& t) const;

private:
    bool require(const char* canon, std::vector<const char*> aliases, std::vector<uint64_t> shape,
                 DFlashTensor* out, std::string& err);
    std::unique_ptr<strata::GgufFile> file_;
    std::string path_;
    DFlashGeometry geom_;
    std::vector<DFlashTensor> tensors_;
    uint64_t weight_bytes_ = 0;
};

/// The drafter owns its weights, its own K/V pools and the block forward; it never owns target
/// verifier state. Query graphs reuse compatible attention layouts; docs/DFLASH.md
/// holds the semantics.  The embedding and the LM head are the TARGET's, bound here, never copied.
class DFlashDrafter {
public:
    /// Parse + validate the artifact (no device work, no inference).
    bool load(const std::string& gguf_path, std::string& err) {
        return artifact_.open(gguf_path, err);
    }
    const DFlashArtifact& artifact() const { return artifact_; }
    /// Allocated logical draft context capacity (independent of target streaming).
    int64_t capacity() const { return cap_; }
    /// Maximum candidates this artifact may propose per pass (its trained query count).
    int max_block() const { return (int) artifact_.geom().block_size; }
    bool load_head(const std::string& path, std::string& err);  ///< experimental draft-only head
    /// The device bytes this drafter holds right now: the packed weights, the widened norm vectors,
    /// the K/V arenas and every scratch buffer - each counted once, at allocation.  The target's
    /// embedding table and shared native head are not counted. An optional draft-owned head
    /// and its activation scratch are counted. 0 before upload().
    uint64_t vram_bytes() const;
    /// Attention scratch grows alongside the owned K/V using the same cache chunks.
    int64_t scratch_chunks_needed(int64_t cells) const;
    bool grow_scratch(int64_t cells, const std::function<VmmChunk()>& take);
    int64_t shrink_scratch(int64_t cells, const std::function<void(VmmChunk)>& give);
    int64_t scratch_cells() const { return scratch_range_ ? scratch_cap_ : INT64_MAX; }
    uint64_t scratch_mapped_bytes() const;
    uint64_t scratch_reserved_bytes() const;

    /// Uploads the packed weights and carves the drafter's own K/V pools and scratch.  Call BEFORE
    /// the expert cache is sized, like MtpDrafter::load.  `target_g` is the target's geometry (its
    /// QSA pool shapes); the pools hold the artifact's `layers` draft layers.  `window` sizes the
    /// drafter's context: the attention sees cells [0, window) and the pools hold exactly those
    /// cells (the reference attends to every cell; 0 = every cell the session's cache could hold,
    /// which prices five FP16 pools at 10 KiB per cell). On supported CUDA cache
    /// configurations, physical K/V and attention scratch grow from 8192 cells; other
    /// backends allocate them up front. The attention always covers the same cells.
    /// `mask_override`: the CLI's --dflash-mask-token (-1 = the artifact's metadata decides; the
    /// effective id is the override when given, else the metadata, and must sit inside the target's
    /// vocabulary - generate.cpp validates it against n_vocab).
    bool upload(const ModelGeometry& target_g, SessionState& ss, int device, int64_t window,
                int64_t mask_override, std::string& err, bool elastic_kv = false);
    /// Binds the target's LM head.  The query rows' embeddings are gathered per cycle from the
    /// target's table - nothing is cached here.
    bool bind(const WeightTable& wt, const NativeHead* head, std::string& err, const std::string& vocab = {});
    /// Frees every resource this drafter holds (device buffers, the five K/V states' arenas and
    /// their pinned staging, the stream) and leaves it unloaded.  Safe on a partially-uploaded
    /// drafter; called by the destructor and on upload()'s failure paths.
    void release();

    /// Context cells for committed positions [pos0, pos0 + rows): `taps` is the prompt path's
    /// capture, BF16 rows [n_taps x stride_rows x n_embd], `stride_rows` the per-tap row stride.
    /// Runs in batches of up to 8 rows through the fusion and one K/V append per draft layer.
    bool add_context(const uint16_t* taps, int n_taps, int64_t stride_rows, int64_t pos0, int64_t rows,
                     std::string& err);
    /// The same from the verify window's capture: f32 taps [n_taps][stride_floats] (the window's
    /// max_t * n_embd), rows [0, rows) of each tap valid (at most 8).
    bool add_context_f32(const float* taps, int n_taps, int64_t stride_floats, int64_t pos0, int64_t rows,
                         std::string& err);

    /// One block (docs/DFLASH.md): the anchor token `x` at position `pos` (its own context cell
    /// does not exist - query 0 carries it), `block` query rows at rope positions [pos, pos+block),
    /// inputs [x, mask x (block-1)], attention non-causal over [0, pos+block) once the queries'
    /// own cells are appended.  Greedy argmax per row into `out`: candidate j is the token at
    /// pos+1+j.  Synchronizes the drafter's stream before returning.
    bool propose(int32_t x, int64_t pos, int block, int32_t* out, std::string& err, float* probabilities = nullptr);

    cudaStream_t stream() const { return cs_; }
    /// The drafter's per-layer K/V states (read-only: STRATA_STATE_HASH hashes them like MTP's).
    const std::vector<QsaState>& kv_states() const { return st_; }
    int device() const { return device_; }
    bool idle(std::string& err) {
        if (cs_ && cudaStreamSynchronize(cs_) != cudaSuccess) { err = "dflash: its stream failed"; return false; }
        return true;
    }
    /// The mask token id this drafter proposes with (metadata, or the CLI override set at upload).
    int64_t mask_token() const { return mask_; }

private:
    /// ctx = hidden_norm(fc(tapin_)) for `rows` rows already staged in `tapin_` ([rows][F] bf16),
    /// then each draft layer's context K/V appended at [pos0, pos0+rows).
    bool fusion_rows(int64_t pos0, int rows, std::string& err);

    void project(const uint16_t* x, const uint16_t* w, float* y, int ni, int no, int rows, void* stream, bool prepared = false);
    void prepare_projection(const uint16_t* x, int ni, int rows, void* stream);
    std::vector<std::pair<const uint16_t*, int>> quant_types_;
    float* proj_float_ = nullptr;
    uint8_t* proj_q8_ = nullptr;

    DFlashArtifact artifact_;
    uint64_t vram_ = 0;
    int64_t mask_ = -1;
    int device_ = -1;

public:
    ~DFlashDrafter() { release(); }   // deterministic cleanup; release() is also safe on a partial upload

private:
    cudaStream_t cs_ = nullptr;
    std::array<cudaGraphExec_t, 9> graph_exec_{};
    std::array<int64_t, 9> graph_chunks_{}, graph_cap_{};
    bool graph_prob_ = false;
    void clear_graphs();

    // the target's shared modules
    const NativeHead* head_ = nullptr;
    NativeHead* owned_head_ = nullptr;
    int32_t* head_ids_ = nullptr;
    int head_rows_ = 0;
    const WeightRef* emb_ref_ = nullptr;

    // weights (device GGML payloads, the GGUF layout) and the rms_norm gammas widened to F32
    uint16_t* w_ = nullptr;
    std::vector<std::pair<std::string, const uint16_t*>> wt_;   ///< canonical name -> device pointer
    std::vector<std::pair<std::string, const float*>> wf_;      ///< canonical name -> device f32 (norms)

    // the drafter's own K/V: ONE STATE PER DRAFT LAYER (a QsaState is one layer's pools)
    std::vector<QsaState> st_;
    std::vector<void*> arenas_;
    ModelGeometry pool_g_ = {};      ///< the pool geometry (the target's QSA shapes, 5 layers)
    strata::kernels::QsaShapes shapes_ = {};
    void* state_arena_ = nullptr;
    int64_t window_ = 0, cap_ = 0, attn_scratch_floats_ = 0;

    // buffers: at most 8 rows ride through the forward at once
    int32_t *tok_ = nullptr, *step_ = nullptr, *attn_step_ = nullptr;
    int32_t *pos_ = nullptr, *poskv_ = nullptr;   ///< the rope positions, built on the DEVICE once
                                                  ///< per forward (dflash_build_positions): the query
                                                  ///< rows' [row][head] at pos+r, and the KV rows'
                                                  ///< [row][head_kv].  No host staging - the pinned
                                                  ///< h_pos_/h_step_ pair and its two-region race
                                                  ///< dance existed only for the staging copies
    uint16_t *tapin_ = nullptr, *xn16_ = nullptr, *attn16_ = nullptr;
    float *tapf_ = nullptr, *emb_ = nullptr, *h_ = nullptr, *xn_ = nullptr, *ctx_ = nullptr;
    float *q_ = nullptr, *kc_ = nullptr, *vc_ = nullptr, *attn_ = nullptr, *bo_ = nullptr;
    float *gate_ = nullptr, *up_ = nullptr, *logits_ = nullptr;
    uint8_t *arg_scratch_ = nullptr, *top_scratch_ = nullptr, *xq_ = nullptr;
    int32_t* out_ = nullptr;         ///< the block's picks, device alias
    int32_t* h_out_ = nullptr;       ///< ... and its mapped host memory
    int32_t* h_tok_ = nullptr;       ///< host-side token ids staged to tok_
    void* attn_scratch_ = nullptr;
    std::unique_ptr<VmmRange> scratch_range_;
    int64_t scratch_cap_ = 0;
    uint64_t scratch_bytes(int64_t cells) const;
    int64_t max_rows_ = 0;
    int64_t cycle_ = 0;              ///< proposes so far (the parity fixture's cycle selector)
    char parity_dir_[512] = {};      ///< STRATA_DF_PARITY: the stage-dump directory (empty: off)
};

}  // namespace strata::core
