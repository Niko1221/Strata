#pragma once
#include "strata/qwen38/quant_tensor.hpp"
#include <cstdint>

namespace strata::qwen38 {

enum class SequenceMixerType : uint8_t {
    GATED_DELTANET_GROUPED = 0,       ///< Qwen3.8 GDN (H_qk=16, H_v=48, d=128)
    GATED_ATTENTION_PARTIAL_ROPE = 1, ///< Qwen3.8 GQA (H_q=24, H_kv=4, d=256, rope=64)
    STANDARD_GQA_SLIDING = 2,         ///< Gemma-4 / Llama sliding + full GQA
};

enum class FfnBlockType : uint8_t {
    DENSE_SWIGLU = 0,
    DENSE_GEGLU = 1,
    SPARSE_MOE_PANTRY = 2,
};

/// Metadata-driven layer descriptor holding per-tensor mixed quantization metadata.
struct UniversalTransformerLayer {
    int               layer_index = 0;
    SequenceMixerType mixer_type  = SequenceMixerType::GATED_DELTANET_GROUPED;
    FfnBlockType      ffn_type    = FfnBlockType::DENSE_SWIGLU;

    // Norms (F32 / F16)
    StrataQuantTensor attn_norm;
    StrataQuantTensor post_attn_norm;

    // GDN / Attention mixer weights
    StrataQuantTensor w_qkv;
    StrataQuantTensor w_out_gate;
    StrataQuantTensor w_out_proj;
    StrataQuantTensor w_ssm_conv1d;
    StrataQuantTensor w_ssm_alpha;
    StrataQuantTensor w_ssm_beta;
    StrataQuantTensor w_ssm_dt_bias;
    StrataQuantTensor w_ssm_a;
    StrataQuantTensor w_ssm_norm;

    // GQA weights
    StrataQuantTensor w_q;
    StrataQuantTensor w_k;
    StrataQuantTensor w_v;
    StrataQuantTensor w_q_norm;
    StrataQuantTensor w_k_norm;

    // FFN weights (supports per-layer mixed types e.g. IQ3_S + IQ2_S)
    StrataQuantTensor w_ffn_gate;
    StrataQuantTensor w_ffn_up;
    StrataQuantTensor w_ffn_down;
};

} // namespace strata::qwen38
