// include/strata/core/dense_model.hpp - a DENSE qwen35 model on Strata's kernels: Qwen3.8-27B (unsloth GGUF).
//
// WHY THIS IS A SEPARATE RUNNER AND NOT A MODE OF `strata`.  The MoE engine is specialised end to end
// (check_architecture: `qwen4exp`, 2560 wide, 48 blocks; expert arena; QSA indexer; hyper-connections; PLE;
// n-gram table).  A dense model has none of it, so `strata` refuses its GGUF at the first guard.  What the two
// models DO share is the GDN mixer (S = 128, 16 QK heads, 48 V heads, 10240 conv channels, d_conv 4 - the same
// numbers), the quantized GEMVs, RoPE and the attention output gate.  Those kernels are reused as they are.
//
// THE MODEL (qwen35, 64 layers, n_embd 5120, n_ff 17408), per layer:
//
//     xn = rms_norm(x, attn_norm)
//     GDN layer  (3 of every 4):  qkv, z = attn_qkv xn, attn_gate xn
//                                 conv+silu -> l2(q), l2(k) -> alpha, beta = ssm_alpha xn, ssm_beta xn
//                                 gate = softplus(alpha + dt) * ssm_a ; beta = sigmoid(beta)
//                                 o = delta-rule(state, q, k, v) ; y = rms_norm(o) * ssm_norm * sigmoid(z)
//                                 mix = ssm_out y
//     attention layer (1 of 4):   q_full, k, v = attn_q xn, attn_k xn, attn_v xn      q_full = [q | gate] per head
//                                 q, k = rope(rms_norm_head(q), rms_norm_head(k))       24 Q heads, 4 KV heads, 256
//                                 attn = softmax(q K^T / sqrt(256)) V ; attn *= sigmoid(gate)
//                                 mix = attn_output attn
//     x += mix
//     x += ffn_down( silu(ffn_gate xn2) * ffn_up xn2 ),   xn2 = rms_norm(x, post_attention_norm)
//
// then logits = output( rms_norm(x, output_norm) ).  Which of the two mixers a layer has is read from the tensors
// it carries (attn_qkv.weight -> GDN, attn_q.weight -> attention), not from a formula.
//
// Weights stay in their GGUF blocks on the GPU and are multiplied by native_mmvq, which takes every type an
// Unsloth UD file uses - Q4_K, Q5_K, Q6_K, Q3_K, Q8_0 AND IQ3_S / IQ3_XXS / IQ4_XS / IQ2_*.  There is no pack step
// and no per-type conversion; a GGUF with a type outside that list is refused with the list.
#pragma once

#include <cstddef>
#include <cstdint>
#include <memory>
#include <string>
#include <vector>

namespace strata::core {

struct DenseConfig {
    std::string arch;            ///< general.architecture ("qwen35")
    int n_layer = 0, n_embd = 0, n_ff = 0, n_vocab = 0;
    int n_head = 0, n_head_kv = 0, head_dim = 0, n_rot = 0;
    int ssm_d_conv = 0, ssm_state = 0, ssm_k_heads = 0, ssm_v_heads = 0;
    float rms_eps = 1e-6f, rope_base = 1e7f;
    int32_t eos_id = -1;         ///< tokenizer.ggml.eos_token_id: the id that ends an answer
    int conv_channels() const { return 2 * ssm_state * ssm_k_heads + ssm_state * ssm_v_heads; }
    int value_dim() const { return ssm_state * ssm_v_heads; }
};

class DenseModel {
public:
    DenseModel();
    ~DenseModel();
    DenseModel(const DenseModel&) = delete;
    DenseModel& operator=(const DenseModel&) = delete;

    /// Reads the GGUF (single file or the first shard of a split), checks every tensor it needs, uploads the
    /// weights and allocates the KV cache for `max_context` cells and the recurrent state.
    bool load(const std::string& gguf_path, int64_t max_context, std::string& err);

    const DenseConfig& config() const { return cfg_; }
    int64_t max_context() const { return max_context_; }
    uint64_t weight_bytes() const { return weight_bytes_; }
    uint64_t kv_bytes() const { return kv_bytes_; }

    /// Forgets the conversation: zeroes the recurrent state and the position.
    void reset();
    /// Number of tokens the state has consumed (= the next position).
    int64_t position() const { return pos_; }

    /// Feeds one token at position() and advances it.  With `want_logits` the vocabulary logits are left in
    /// logits() (device, n_vocab floats); without, the final norm and head are skipped (prompt tokens).
    bool step(int32_t token, bool want_logits, std::string& err);
    /// Blocks until everything queued has run.
    bool sync(std::string& err);

    float* logits() const;             ///< device pointer, valid after a step(want_logits=true)
    void* stream() const;

private:
    struct Impl;
    DenseConfig cfg_;
    int64_t max_context_ = 0, pos_ = 0;
    uint64_t weight_bytes_ = 0, kv_bytes_ = 0;
    std::unique_ptr<Impl> impl_;
};

}  // namespace strata::core
