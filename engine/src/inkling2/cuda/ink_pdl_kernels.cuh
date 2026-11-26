// engine/src/inkling2/cuda/ink_pdl_kernels.cuh - which kernels may start early (PERF-STEP lane)
//
// MIT License
// Copyright (c) 2026 Niko1221 and the Strata contributors
//
// EVERY kernel registered here starts with INK_PDL_WAIT() (the first statement, before any early return). Add a kernel to this list only together with that
// line; a kernel that is not listed is launched with its ordinary edges and nothing else changes. See ink_pdl.cuh.
#pragma once

#include "ink_pdl.cuh"
#include "ink_topk.cuh"

namespace strata::ink::pdl {

inline void register_all() {
    // trunk glue shared by the old path and the fast path
    add(k_ink_attn_res_mix);
    add(k_ink_attn_res_3x_cat_rms);
    add(k_ink_mlp_res_cat_rms);
    add(k_ink_rel_table);
    add(k_ink_rel_table_md);
    add(k_ink_post_qkv_fused);
    add(k_ink_rms_norm);
    add(k_ink_rms_norm_3x);
    add(k_ink_sconv4_seq);
    add(k_ink_moe_reduce8_sconv_m1);
    add(k_ink_embed_norm);
    add(k_ink_softcap);
    add(k_ink_topk_pass1);
    add(k_ink_topk_merge);
    add(k_ink_router_select_topk8);
    add(k_ink_moe_bin);
    // old path (BF16 trunk, SF4/MX8 experts, F32 controllers)
    add(k_ink_gemv_multi_bf16<1, false>);
    add(k_ink_gemv_multi_bf16<4, false>);
    add(k_ink_gemv_multi_bf16<8, false>);
    add(k_ink_gemv_multi_bf16<16, false>);
    add(k_ink_gemv_multi_bf16<1, true>);
    add(k_ink_gemv_multi_bf16<4, true>);
    add(k_ink_gemv_f32<1>);
    add(k_ink_gemv_f32<4>);
    add(k_ink_gemv_f32<8>);
    add(k_ink_gemv_f32<16>);
    add(k_ink_router_gemv_smallM<1>);
    add(k_ink_router_gemv_smallM<4>);
    add(k_ink_router_gemv_smallM<8>);
    add(k_ink_router_gemv_smallM<16>);
    add(k_ink_moe_gu_sf4_mx8<1>);
    add(k_ink_moe_dn_sf4_mx8<1>);
    add(k_ink_moe_gu_sf4_mx8<4>);
    add(k_ink_moe_dn_sf4_mx8<4>);
    add(k_ink_ctrl_3x_down_smallM<1>);
    add(k_ink_ctrl_3x_down_smallM<4>);
    add(k_ink_ctrl_3x_down_smallM<8>);
    add(k_ink_ctrl_3x_down_smallM<16>);
    add(k_ink_ctrl_3x_apply_smallM<1>);
    add(k_ink_ctrl_3x_apply_smallM<4>);
    add(k_ink_ctrl_3x_apply_smallM<8>);
    add(k_ink_ctrl_3x_apply_smallM<16>);
    add(k_ink_ctrl_down_smallM<1>);
    add(k_ink_ctrl_down_smallM<4>);
    add(k_ink_ctrl_down_smallM<8>);
    add(k_ink_ctrl_down_smallM<16>);
    add(k_ink_ctrl_apply_smallM<1>);
    add(k_ink_ctrl_apply_smallM<4>);
    add(k_ink_ctrl_apply_smallM<8>);
    add(k_ink_ctrl_apply_smallM<16>);
    // 5090 fast path (FP8 trunk, FP16 router, M=1 L1-direct experts, FP8 controllers)
    add(rtx5090::k_ink_gemv_multi_fp8_m1_l1);
    add(rtx5090::k_ink_gemv_multi_fp8_smallM_l1<4>);
    add(rtx5090::k_ink_gemv_multi_fp8_smallM_l1<8>);
    add(rtx5090::k_ink_f32_to_h2);
    add(rtx5090::k_ink_f32_sink_sconv_fused<1>);
    add(rtx5090::k_ink_f32_sink_sconv_fused<4>);
    add(rtx5090::k_ink_f32_sink_sconv_fused<16>);
    add(rtx5090::k_ink_rms_norm_3x_h2);
    add(rtx5090::k_ink_router_gemv_fp16<1>);
    add(rtx5090::k_ink_router_gemv_fp16<4>);
    add(rtx5090::k_ink_router_gemv_fp16<8>);
    add(rtx5090::k_ink_router_gemv_fp16<16>);
    add(rtx5090::k_ink_moe_gu_m1_l1);
    add(rtx5090::k_ink_moe_dn_m1_l1);
    add(rtx5090::k_ink_ctrl_3x_down_fp8<1>);
    add(rtx5090::k_ink_ctrl_3x_down_fp8<4>);
    add(rtx5090::k_ink_ctrl_3x_down_fp8<8>);
    add(rtx5090::k_ink_ctrl_3x_down_fp8<16>);
    add(rtx5090::k_ink_ctrl_3x_down_fp8<24>);
    add(rtx5090::k_ink_ctrl_3x_apply_fp8<1>);
    add(rtx5090::k_ink_ctrl_3x_apply_fp8<4>);
    add(rtx5090::k_ink_ctrl_3x_apply_fp8<8>);
    add(rtx5090::k_ink_ctrl_3x_apply_fp8<16>);
    add(rtx5090::k_ink_ctrl_3x_apply_fp8<24>);
    add(rtx5090::k_ink_ctrl_1x_down_fp8<1>);
    add(rtx5090::k_ink_ctrl_1x_down_fp8<4>);
    add(rtx5090::k_ink_ctrl_1x_down_fp8<8>);
    add(rtx5090::k_ink_ctrl_1x_down_fp8<16>);
    add(rtx5090::k_ink_ctrl_1x_apply_fp8<1>);
    add(rtx5090::k_ink_ctrl_1x_apply_fp8<4>);
    add(rtx5090::k_ink_ctrl_1x_apply_fp8<8>);
    add(rtx5090::k_ink_ctrl_1x_apply_fp8<16>);
}

}  // namespace strata::ink::pdl
