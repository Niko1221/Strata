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
// twice: structurally here (metadata <-> tensor shapes agree, BF16, in bounds), and against what
// the runtime fast path supports (validate_supported) when the engine wires the drafter up.
// Semantics and the DeepSpec anchor layout: docs/DFLASH.md.
#pragma once

#include "strata/artifact/gguf_reader.hpp"

#include <cstdint>
#include <string>
#include <vector>

namespace strata::core {

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

/// One device weight, row-major rows of `cols` BF16 values (the GGUF file layout, ne0 varies
/// fastest, is preserved: row r is output neuron r, y[r] = dot(row, x)).
struct DFlashTensor {
    std::string name;      ///< canonical Strata-side name ("layers.3.self_attn.q_proj")
    std::string file_name; ///< the name it carries in the GGUF (either naming family)
    int64_t rows = 0, cols = 0;
    const uint16_t* d = nullptr;   ///< device BF16, valid after upload()
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

    /// bf16 payload bytes of every required tensor together (weights VRAM before scratch).
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

/// The drafter owns its weights, its own KV pools and the block forward; it never owns target
/// verifier state.  (The forward lands with the DFlash decode commits; the artifact API above is
/// the load-time contract.)
class DFlashDrafter {
public:
    /// Parse + validate the artifact (no device work, no inference).
    bool load(const std::string& gguf_path, std::string& err) {
        return artifact_.open(gguf_path, err);
    }
    const DFlashArtifact& artifact() const { return artifact_; }
    /// Maximum candidates this artifact may propose per pass (its trained query count).
    int max_block() const { return (int) artifact_.geom().block_size; }
    /// VRAM the drafter's weights take (exact once uploaded, the GGUF payload size before that).
    uint64_t vram_bytes() const { return vram_ + artifact_.weight_bytes(); }
    /// VRAM the later bind() will add (draft logits + head scratch) so the expert cache can be
    /// sized with it reserved; the weights are already counted in vram_bytes().
    uint64_t bind_bytes(int64_t n_vocab, int max_t) const;

private:
    DFlashArtifact artifact_;
    uint64_t vram_ = 0;
};

}  // namespace strata::core
