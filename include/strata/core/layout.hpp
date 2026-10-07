// include/strata/core/layout.hpp - the pack's LAYOUT, resolved by name and checked against the kernels.
//
// `WeightTable` answers "where are the bytes".  This answers "is this the tensor I think it is", and it is
// the layer where a pack and a kernel disagree - which is the failure this project keeps paying for.  Two
// examples, both real:
//
//   * round 194 sized the indexer key store from `indexer.head_count = 4`, which counts QUERY heads.  The
//     cached key is ONE shared head of 128 (`indexer.k_proj` is [2560, 128]), so the term was 4x too big and
//     the error survived a round because it moved in the direction that TIGHTENS the budget.
//   * `docs/semantics.md` and the tensor manifest agree on every dimension here, but nothing CHECKED that a
//     named tensor had the shape the kernel reading it assumes.
//
// So this header does two things: it names the tensors per layer type, and it asserts every shape the
// kernels depend on.  A mismatch is reported with the tensor name, the shape found and the shape required,
// at LOAD time - not as a wrong number inside a GEMV at token 4000.
#pragma once

#include "strata/core/arch.hpp"
#include "strata/core/weights.hpp"

#include <cstdint>
#include <string>

namespace strata {
class GgufFile;   // the model file; defined in the artifact layer, which this header does not include
}  // namespace strata

namespace strata::core {

/// The model's geometry, taken from `docs/semantics.md` and the artifact's own metadata.  Every field here
/// is a number a kernel depends on, so a change is a change to a kernel contract and not a tuning knob.
///
/// The defaults are the first family's (`qwen4exp`).  A second family is a FACTORY - `glm5next_geometry()` -
/// and every field the file carries is then overlaid from the file, so a fine-tune that changes a dimension
/// is read rather than assumed.
struct ModelGeometry {
    Arch arch = Arch::Qwen4Exp;

    int64_t n_embd = 2560;
    int64_t n_layers = 48;
    int64_t qsa_interval = 4;      ///< every 4th layer is full attention: layers 3, 7, ... 47
    int64_t n_vocab = 248320;      ///< the head is untied on both families: `output.weight` is present

    // GDN (36 layers)
    int64_t ssm_state_size = 128;
    int64_t ssm_k_heads = 16;
    int64_t ssm_v_heads = 48;
    int64_t ssm_d_conv = 4;
    int64_t ssm_conv_channels = 10240;   ///< 2*128*16 + 128*48
    int64_t ssm_value_dim = 6144;        ///< 128 * 48

    // QSA (12 layers)
    int64_t n_head = 24;
    int64_t n_head_kv = 2;
    int64_t head_dim = 256;
    int64_t idx_q_heads = 4;
    int64_t idx_key_dim = 128;

    // gated residual, on every layer
    int64_t hc = 4;
    int64_t hc_lr = 320;

    // MoE, on every layer
    int64_t n_expert = 512;
    int64_t n_ff = 640;

    // ---- glm5-next ----
    // Absorbed NoPE MLA.  The cache holds the kv_lora latent, so for these layers `n_head_kv` is 1 and
    // `head_dim` is `kv_lora_rank`; `mla_head_dim` is the per-head width the output projection consumes.
    int64_t q_lora_rank = 0;        // 1536: 4096 -> 1536 -> 64*256, with the norm ON the latent
    int64_t kv_lora_rank = 0;       // 512: what the cache stores per token, and K == V
    int64_t mla_head_dim = 0;       // 256

    // KDA (the linear layers), whose state is per-head and sequential over tokens
    int64_t kda_head_dim = 0;       // 128
    int64_t kda_conv_kernel = 0;    // 4, applied per stream (q, k, v each have their own conv)
    double kda_gate_floor = 0.0;    // -5.0: the decay gate is bounded to (floor, 0)

    // dense-lead layers, and the shared expert
    int64_t n_layer_dense_lead = 0; // 3
    int64_t n_ff_dense = 0;         // 12288; 0 means every layer is MoE
    int64_t n_expert_shared = 0;    // 1
    int64_t n_nextn = 0;            // 1 MTP block; block_count counts it, n_layers does not

    // scalars the graph reads per layer
    int64_t hc_mix = 0;             // 24 = hc*(2+hc): pre(4) + post(4) + comb(16)
    int64_t idx_top_k = 0;          // 2048
    int64_t idx_kpool = 0;          // 4
    double expert_weights_scale = 0.0;  // 2.5, applied after the sum-normalisation
    double swiglu_clamp = 0.0;      // 10.0 - per-layer in the file, uniform on every artifact seen
    /// `attention.layer_norm_rms_epsilon`: **1e-6 on qwen4exp and 1e-5 on glm5-next**, so a layer file must
    /// read it from here rather than use `gemv::RMS_EPS`, which is the first family's number.  KDA's L2
    /// normalisation reuses it too (there it is a floor on the norm, not a term in the sum).
    double rms_eps = 1e-6;

    int64_t hc_dim() const { return hc * n_embd; }
    /// `layer % qsa_interval == qsa_interval - 1` is full attention.  Derived, not a second list.
    int64_t n_qsa_layers() const { return n_layers / qsa_interval; }
    int64_t n_gdn_layers() const { return n_layers - n_qsa_layers(); }
    /// True for the layers that carry a dense SwiGLU FFN instead of MoE.
    bool is_dense_ffn_layer(int64_t layer) const { return layer < n_layer_dense_lead; }
};

/// The first family's shape.  Every field is that family's default, so this is the identity on a
/// default-constructed `ModelGeometry` - it exists so the two factories read the same way at the call site.
inline ModelGeometry qwen4exp_geometry() { return ModelGeometry{}; }

/// GLM-5.3-Flash: 45 trunk layers plus one MTP block, 11 absorbed-NoPE MLA layers at `layer % 4 == 3` and 34
/// KDA layers, dense SwiGLU on the first three.  These are the numbers `general.architecture = glm5-next`
/// declares and the ones the shipped Unsloth IQ4_XS artifact was measured to carry; the loader overlays the
/// file's own keys on top, so this is a starting point and not a second source of truth.
inline ModelGeometry glm5next_geometry() {
    ModelGeometry g;
    g.arch = Arch::Glm5Next;
    g.n_embd = 4096;
    g.n_layers = 45;               // the trunk; the MTP block at 45 is `n_nextn`
    g.n_nextn = 1;
    g.qsa_interval = 4;            // the uniform rule holds: MLA at 3, 7, ... 43
    g.n_vocab = 154880;

    g.n_head = 64;
    g.n_head_kv = 1;               // one shared latent per token, not a KV head count
    g.head_dim = 512;              // = kv_lora_rank: the cache is 512 wide
    g.q_lora_rank = 1536;
    g.kv_lora_rank = 512;
    g.mla_head_dim = 256;

    g.kda_head_dim = 128;
    g.kda_conv_kernel = 4;
    g.kda_gate_floor = -5.0;

    g.n_layer_dense_lead = 3;
    g.n_ff_dense = 12288;
    g.n_expert = 288;
    g.n_ff = 2048;                 // expert FF; the shared expert is the same width
    g.n_expert_shared = 1;

    g.idx_q_heads = 32;
    g.idx_key_dim = 128;
    g.idx_top_k = 2048;
    g.idx_kpool = 4;

    g.hc = 4;
    g.hc_mix = 24;
    g.hc_lr = 0;                   // no low-rank hyper-connection path: hc_*_fn is a full 16384x24 map

    g.expert_weights_scale = 2.5;
    g.swiglu_clamp = 10.0;
    g.rms_eps = 1e-5;              // measured from `l4.gguf`'s own `attention.layer_norm_rms_epsilon`
    return g;
}

/// The factory for an arch.  `Unknown` is a programming error at a call site that has not resolved the file
/// yet, and returns the first family's shape rather than something uninitialised.
inline ModelGeometry model_geometry_for(Arch a) {
    return a == Arch::Glm5Next ? glm5next_geometry() : qwen4exp_geometry();
}

/// Overlay a model file's own metadata onto `g`, and resolve `arch` from it.  Defined in model_arch.cpp so
/// this header stays free of the artifact layer (the file is forward-declared, not included); declared here
/// because the geometry and the file that describes it belong together.  Returns "" on success, or a precise
/// message.  `K` is the expert count per token.
std::string apply_model_geometry(const strata::GgufFile& f, ModelGeometry& g, int64_t& K);

/// True for the full-attention layers.  `docs/semantics.md` gives this twice over - `full_attention_interval
/// = 4` and an explicit `attention.compress_ratios` array - and this is the first of the two.
inline bool is_qsa_layer(const ModelGeometry& g, int64_t layer) {
    return layer % g.qsa_interval == g.qsa_interval - 1;
}

/// One layer's tensors, resolved by NAME.  `get("attn_qkv.weight")` looks up `blk.<layer>.attn_qkv.weight`
/// and returns null if the pack does not have it - a null is information, because a GDN layer has no
/// `attn_q` and a QSA layer has no `attn_qkv`.
///
/// Holds a reference to the table; the table must outlive it.
class LayerView {
public:
    LayerView(const WeightTable& table, int64_t layer) : table_(&table), layer_(layer) {}

    int64_t layer() const { return layer_; }
    std::string name(const char* suffix) const;
    const WeightRef* get(const char* suffix) const { return table_->find(name(suffix)); }

private:
    const WeightTable* table_;
    int64_t layer_;
};

/// Every shape the kernels depend on, asserted for ONE layer.  Returns false and fills `err` with the first
/// mismatch, naming the tensor, what it has and what is required.
///
/// This is deliberately a separate function from `LayerView`: a caller that only wants the pointers should
/// not pay for the checks, and a caller that wants the checks should get ALL of them rather than the ones
/// its own call site happens to touch.
bool check_layer(const WeightTable& table, const ModelGeometry& g, int64_t layer, std::string& err);

/// `check_layer` over every layer, plus the cross-layer properties: exactly 12 QSA layers at the right
/// indices, every GDN layer having the GDN set and no QSA tensor, and vice versa.  Returns the first failure.
bool check_all(const WeightTable& table, const ModelGeometry& g, std::string& err);

}  // namespace strata::core
