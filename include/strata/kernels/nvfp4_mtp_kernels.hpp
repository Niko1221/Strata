// include/strata/kernels/nvfp4_mtp_kernels.hpp - RDNA4 (gfx1201) FlyDSL HSACO kernel launchers for native Multi-Token Prediction (MTP),
// target logit confidence/margin extraction, NSA/CSA sink+window pinning, and capped hybrid small-batch MoE routing.
#pragma once

#include "strata/kernels/gfx1201_nvfp4_kernels.hpp"

#include <hip/hip_runtime_api.h>

#include <algorithm>
#include <cstddef>
#include <cstdint>

namespace strata::kernels {

// Per-expert byte layout for the 512 FP8 MTP experts (mtp.layers.0.mlp.experts.{0..511}):
//   gate_proj.weight [640, 2560] F8_E4M3 (1,638,400 B) + weight_scale_inv [5, 20] BF16 (200 B, padded to 256 B)
//   up_proj.weight   [640, 2560] F8_E4M3 (1,638,400 B) + weight_scale_inv [5, 20] BF16 (200 B, padded to 256 B)
//   down_proj.weight [2560, 640] F8_E4M3 (1,638,400 B) + weight_scale_inv [20, 5] BF16 (200 B, padded to 576 B)
inline constexpr size_t MTP_EXP_W_BYTES = 1638400;
inline constexpr size_t MTP_EXP_S_BYTES = 200;
inline constexpr size_t MTP_OFF_GATE_W = 0;
inline constexpr size_t MTP_OFF_UP_W = 1638400;
inline constexpr size_t MTP_OFF_DOWN_W = 3276800;
inline constexpr size_t MTP_OFF_GATE_S = 4915200;
inline constexpr size_t MTP_OFF_UP_S = 4915456;
inline constexpr size_t MTP_OFF_DOWN_S = 4915712;
inline constexpr size_t MTP_EXP_STRIDE_BYTES = 4916288;

struct Nvfp4ArgmaxSummary {
    int32_t tok_id;
    float prob;
    float margin;
};

// Write M rows of BF16 [M, 10240] final 4-stream residuals into the 512-slot ring buffer at positions [pos0, pos0 + M).
inline void launch_nvfp4_mtp_ring_write(Gfx1201KernelCache& kernels, const void* src_x_bf16, void* dst_ring_bf16,
                                        int32_t pos0, int32_t ring_mask, int32_t M, hipStream_t stream = nullptr) {
    if (M <= 0) return;
    void* args[] = {const_cast<void**>(&src_x_bf16), &dst_ring_bf16, &pos0, &ring_mask};
    kernels.launch("mtp_ring_write", 5, M, 1, args, stream);
}

// Gather M token embeddings from embed_tokens [248320, 2560] and apply (1 + w) RMSNorm over 2560 elements.
inline void launch_nvfp4_mtp_emb_rms_norm(Gfx1201KernelCache& kernels, const int32_t* d_tok_ids,
                                          const void* d_embed_tokens, const void* d_norm_w, void* d_out_bf16,
                                          int32_t M, float eps, hipStream_t stream = nullptr) {
    if (M <= 0) return;
    void* args[] = {const_cast<int32_t**>(&d_tok_ids), const_cast<void**>(&d_embed_tokens),
                    const_cast<void**>(&d_norm_w), &d_out_bf16, &eps};
    kernels.launch("mtp_emb_rms_norm", M, 1, 1, args, stream);
}

// Apply (1 + w) RMSNorm over 10240 elements per row (reading row t from d_r_base[((ring_pos0 + t) & ring_mask) * 10240]).
inline void launch_nvfp4_mtp_hidden_rms_norm(Gfx1201KernelCache& kernels, const void* d_r_base, int32_t ring_pos0,
                                             int32_t ring_mask, const void* d_norm_w, void* d_out_bf16, int32_t M,
                                             float eps, bool per_stream = false, hipStream_t stream = nullptr) {
    if (M <= 0) return;
    int32_t ps = per_stream ? 1 : 0;
    void* args[] = {const_cast<void**>(&d_r_base), const_cast<void**>(&d_norm_w), &d_out_bf16,
                    &ring_pos0, &ring_mask, &eps, &ps};
    kernels.launch("mtp_hidden_rms_norm", M, 1, 1, args, stream);
}

// Broadcast-add e2 [M, 2560] to each of the 4 streams of h2 [M, 4, 2560] -> r_out [M, 4, 2560].
inline void launch_nvfp4_mtp_stream_broadcast_add(Gfx1201KernelCache& kernels, const void* d_h2_bf16,
                                                  const void* d_e2_bf16, void* d_r_out_bf16, int32_t M,
                                                  hipStream_t stream = nullptr) {
    if (M <= 0) return;
    void* args[] = {const_cast<void**>(&d_h2_bf16), const_cast<void**>(&d_e2_bf16), &d_r_out_bf16};
    kernels.launch("mtp_stream_broadcast_add", 40, M, 1, args, stream);
}

// Compute K/V + indexer QK projections for M tokens: x [M, 2560] x w_kv_idx [1664, 2560]
// -> k_out [M, 512], v_out [M, 512], idx_qk_out [M, 640].
inline void launch_nvfp4_mtp_kv_idx_proj(Gfx1201KernelCache& kernels, const void* d_x_bf16,
                                         const void* d_w_kv_idx_bf16, void* d_k_out_bf16, void* d_v_out_bf16,
                                         void* d_idx_qk_out_bf16, int32_t M, hipStream_t stream = nullptr) {
    if (M <= 0) return;
    void* args[] = {const_cast<void**>(&d_x_bf16), const_cast<void**>(&d_w_kv_idx_bf16),
                    &d_k_out_bf16, &d_v_out_bf16, &d_idx_qk_out_bf16};
    kernels.launch("mtp_kv_idx_proj", 208, M, 1, args, stream);
}

// Execute GPU-cold FP8 MTP experts (gate_proj + up_proj + SwiGLU) for decode (M = 1, top-k GPU-routed slots).
inline void launch_nvfp4_mtp_moe_decode_gate_up(Gfx1201KernelCache& kernels, const void* d_mixed_bf16,
                                                const int32_t* d_cur_ids, const int64_t* d_ptrs,
                                                void* d_moe_inter_bf16, int32_t gpu_slots = 3,
                                                hipStream_t stream = nullptr) {
    if (gpu_slots <= 0) return;
    void* args[] = {const_cast<void**>(&d_mixed_bf16), const_cast<int32_t**>(&d_cur_ids),
                    const_cast<int64_t**>(&d_ptrs), &d_moe_inter_bf16};
    kernels.launch("mtp_moe_decode_gate_up", 80, gpu_slots, 1, args, stream);
}

// Execute GPU-cold FP8 MTP experts (down_proj) for decode (M = 1, top-k GPU-routed slots).
inline void launch_nvfp4_mtp_moe_decode_down(Gfx1201KernelCache& kernels, const void* d_moe_inter_bf16,
                                             const int32_t* d_cur_ids, const int64_t* d_ptrs,
                                             void* d_moe_y_bf16, int32_t gpu_slots = 3,
                                             hipStream_t stream = nullptr) {
    if (gpu_slots <= 0) return;
    void* args[] = {const_cast<void**>(&d_moe_inter_bf16), const_cast<int32_t**>(&d_cur_ids),
                    const_cast<int64_t**>(&d_ptrs), &d_moe_y_bf16};
    kernels.launch("mtp_moe_decode_down", 320, gpu_slots, 1, args, stream);
}

// Single-pass argmax + exact full-vocab softmax probability + top-1/top-2 logit margin over [M, 248320] BF16 logits.
inline void launch_nvfp4_argmax_prob_margin_248320(Gfx1201KernelCache& kernels, const void* d_logits_bf16,
                                                   int32_t* d_out_tok, Nvfp4ArgmaxSummary* d_out_summary,
                                                   int32_t M, hipStream_t stream = nullptr) {
    if (M <= 0) return;
    void* args[] = {const_cast<void**>(&d_logits_bf16), &d_out_tok, &d_out_summary};
    kernels.launch("argmax_prob_margin_248320", M, 1, 1, args, stream);
}

// Pin initial attention sink blocks (0..n_sink-1) and local sliding-window blocks ([nb - n_local, nb)) in QSA block scores.
inline void launch_nvfp4_qsa_pin_sink_and_window(Gfx1201KernelCache& kernels, float* d_scores, int32_t pos0,
                                                 int32_t M, int32_t score_stride, int32_t n_sink, int32_t n_local,
                                                 hipStream_t stream = nullptr) {
    if (M <= 0) return;
    void* args[] = {&d_scores, &pos0, &score_stride, &n_sink, &n_local};
    kernels.launch("qsa_pin_sink_and_window", 1, M, 1, args, stream);
}

// Cap GPU-cold PCIe experts to max_gpu_cold in small-batch verification (M <= 4), marking excess cold experts with 0x10000
// for parallel execution on the 16-core AVX-512 CPU pool and producing a GPU-filtered ID list for moe_align.
inline void launch_nvfp4_moe_cap_gpu_cold(Gfx1201KernelCache& kernels, int32_t* d_cur_ids, int32_t* d_gpu_align_ids,
                                          const int64_t* d_ptrs_decode, const int64_t* d_ptrs_vram,
                                          int32_t pairs, int32_t max_gpu_cold, hipStream_t stream = nullptr) {
    if (pairs <= 0) return;
    void* args[] = {&d_cur_ids, &d_gpu_align_ids, const_cast<int64_t**>(&d_ptrs_decode),
                    const_cast<int64_t**>(&d_ptrs_vram), &pairs, &max_gpu_cold};
    kernels.launch("moe_cap_gpu_cold", 1, 1, 1, args, stream);
}

// Safe small-batch (M <= 4, pairs <= 40) MoE tile alignment that packs valid GPU experts (e >= 0) into 16-row tiles
// and cleanly skips CPU-routed slots (e == -1) without out-of-bounds LDS indexing or cross-wave races.
inline void launch_nvfp4_moe_small_align(Gfx1201KernelCache& kernels, const int32_t* d_gpu_align_ids,
                                         int32_t* d_moe_sorted, int32_t* d_moe_tile_e,
                                         int32_t pairs, int32_t max_tiles, hipStream_t stream = nullptr) {
    if (pairs <= 0 || max_tiles <= 0) return;
    void* args[] = {const_cast<int32_t**>(&d_gpu_align_ids), &d_moe_sorted, &d_moe_tile_e, &pairs, &max_tiles};
    kernels.launch("moe_small_align", 1, 1, 1, args, stream);
}

// Pruned draft vocabulary GEMV: computes logits_out [n_dvocab] = x [2560] x lm_head[dvocab[0..n_dvocab-1], 2560]^T.
inline void launch_nvfp4_mtp_dvocab_gemv(Gfx1201KernelCache& kernels, const void* d_x_bf16,
                                         const void* d_lm_head_bf16, const int32_t* d_dvocab,
                                         void* d_logits_out_bf16, int32_t n_dvocab, hipStream_t stream = nullptr) {
    if (n_dvocab <= 0) return;
    int32_t blocks = (n_dvocab + 7) / 8;
    void* args[] = {const_cast<void**>(&d_x_bf16), const_cast<void**>(&d_lm_head_bf16),
                    const_cast<int32_t**>(&d_dvocab), &d_logits_out_bf16, &n_dvocab};
    kernels.launch("mtp_dvocab_gemv", blocks, 1, 1, args, stream);
}

// Single-pass argmax + softmax probability + top-2 margin over n_dvocab logits, mapping winner via d_dvocab.
inline void launch_nvfp4_argmax_prob_margin_dvocab(Gfx1201KernelCache& kernels, const void* d_logits_bf16,
                                                   const int32_t* d_dvocab, int32_t n_dvocab, int32_t* d_out_tok,
                                                   Nvfp4ArgmaxSummary* d_out_summary, hipStream_t stream = nullptr) {
    if (n_dvocab <= 0) return;
    void* args[] = {const_cast<void**>(&d_logits_bf16), const_cast<int32_t**>(&d_dvocab),
                    &d_out_tok, &d_out_summary, &n_dvocab};
    kernels.launch("argmax_prob_margin_dvocab", 1, 1, 1, args, stream);
}

// Truncate MTP top-10 routed experts to the top keep_k highest-weight experts and renormalize their weights.
inline void launch_nvfp4_mtp_topk_truncate(Gfx1201KernelCache& kernels, int32_t* d_cur_ids, float* d_cur_wts,
                                           int32_t keep_k, hipStream_t stream = nullptr) {
    if (keep_k <= 0 || keep_k >= 10) return;
    void* args[] = {&d_cur_ids, &d_cur_wts, &keep_k};
    kernels.launch("mtp_topk_truncate", 1, 1, 1, args, stream);
}

// Compute QSA block keys for prefill batches of any size M (up to 4096+) without RAW_K_CACHE 1024-token ring wrap:
// reads tokens < pos0 from d_raw_k_cache (read-only) and tokens >= pos0 directly from d_idx_qk_b[(tok - pos0) * 640 + 512],
// and writes block keys to d_block_k_cache (tiered). Must be launched BEFORE qsa_cache_write_nvfp4.
inline void launch_nvfp4_qsa_block_keys_from_batch(Gfx1201KernelCache& kernels, const void* d_raw_k_cache,
                                                   const void* d_idx_qk_b, const void* d_idx_k_ln,
                                                   const void* d_rope_cos_bf16, const void* d_rope_sin_bf16,
                                                   void* d_block_k_cache, int64_t bk_host_delta_elems,
                                                   int32_t pos0, int32_t M, int32_t b_start, int32_t b_cnt,
                                                   int32_t vram_blocks, float eps, hipStream_t stream = nullptr) {
    if (b_cnt <= 0) return;
    int32_t grid_x = (b_cnt + 3) / 4;
    void* args[] = {const_cast<void**>(&d_raw_k_cache), const_cast<void**>(&d_idx_qk_b),
                    const_cast<void**>(&d_idx_k_ln), const_cast<void**>(&d_rope_cos_bf16),
                    const_cast<void**>(&d_rope_sin_bf16), &d_block_k_cache,
                    &bk_host_delta_elems, &pos0, &M, &b_start, &b_cnt, &vram_blocks, &eps};
    kernels.launch("qsa_block_keys_from_batch", grid_x, 1, 1, args, stream);
}

// Deterministically refresh the tail min(M, 1024) slots of d_raw_k_cache after qsa_cache_write_nvfp4 when M > 1024.
inline void launch_nvfp4_qsa_refresh_raw_k_tail(Gfx1201KernelCache& kernels, void* d_raw_k_cache,
                                                const void* d_idx_qk_b, int32_t pos0, int32_t M,
                                                hipStream_t stream = nullptr) {
    if (M <= 1024) return;
    const int32_t tail_cnt = 1024;
    int32_t t_start = M - tail_cnt;
    void* args[] = {&d_raw_k_cache, const_cast<void**>(&d_idx_qk_b), &pos0, &t_start};
    kernels.launch("qsa_refresh_raw_k_tail", tail_cnt, 1, 1, args, stream);
}

// Batched QSA block-key scorer that tiles up to 8 query tokens per workgroup in LDS (reading each 128-BF16 block key
// once per 8 query tokens) and fuses NSA initial attention sink + local sliding window pinning.
inline void launch_nvfp4_qsa_score_blocks_batched(Gfx1201KernelCache& kernels, const void* d_idx_qk_b,
                                                  const void* d_idx_q_ln, const void* d_rope_cos_bf16,
                                                  const void* d_rope_sin_bf16, const void* d_block_k_cache,
                                                  float* d_scores, int64_t bk_host_delta_elems, int32_t pos_cur,
                                                  int32_t m_cur, int32_t vram_blocks, int32_t score_stride,
                                                  int32_t n_sink, int32_t n_local, float eps,
                                                  hipStream_t stream = nullptr) {
    if (m_cur <= 0) return;
    int32_t max_nb = (pos_cur + m_cur) / 4;
    int32_t score_grid = std::max(1, (max_nb + 255) / 256);
    int32_t tok_tiles = (m_cur + 7) / 8;
    void* args[] = {const_cast<void**>(&d_idx_qk_b), const_cast<void**>(&d_idx_q_ln),
                    const_cast<void**>(&d_rope_cos_bf16), const_cast<void**>(&d_rope_sin_bf16),
                    const_cast<void**>(&d_block_k_cache), &d_scores, &bk_host_delta_elems,
                    &pos_cur, &m_cur, &vram_blocks, &score_stride, &n_sink, &n_local, &eps};
    kernels.launch("qsa_score_blocks_batched", score_grid, tok_tiles, 1, args, stream);
}

// KRN-2: Automatic small-batch BF16 linear dispatcher:
//   M == 1    -> bf16_gemv_<N>_<K> (wave32-coalesced single-row GEMV across all 64 CUs)
//   2 <= M<=4 -> bf16_gemv_m4_<N>_<K> (wave32-coalesced 4-row GEMV reading W once)
//   M > 4     -> bf16_linear_<N>_<K> (16-row WMMA tile GEMM)
inline void launch_nvfp4_linear_auto(Gfx1201KernelCache& kernels, const char* lin_key, const char* gemv_m1_key,
                                     const char* gemv_m4_key, const void* d_x_bf16, const void* d_w_bf16,
                                     void* d_y_bf16, int32_t N, int32_t M, hipStream_t stream = nullptr) {
    if (M <= 0) return;
    if (M == 1 && gemv_m1_key != nullptr) {
        void* args[] = {const_cast<void**>(&d_x_bf16), const_cast<void**>(&d_w_bf16), &d_y_bf16};
        kernels.launch_default(gemv_m1_key, args, stream);
    } else if (M <= 4 && gemv_m4_key != nullptr) {
        void* args[] = {const_cast<void**>(&d_x_bf16), const_cast<void**>(&d_w_bf16), &d_y_bf16, &M};
        kernels.launch_default(gemv_m4_key, args, stream);
    } else {
        int32_t zero = 0;
        int32_t w = (N % 64 == 0) ? 4 : 2;
        int32_t gx = N / (16 * w);
        int32_t gy = (M + 15) / 16;
        void* args[] = {const_cast<void**>(&d_x_bf16), const_cast<void**>(&d_w_bf16), &d_y_bf16, &M, &zero};
        kernels.launch(lin_key, gx, gy, 1, args, stream);
    }
}

// KRN-1: Single-pass LM Head dispatcher for M in 1..16+ (reads 1.271 GB lm_head.weight ONCE for any M):
//   M == 1    -> bf16_gemv_248320_2560
//   2 <= M<=4 -> bf16_gemv_m4_248320_2560
//   5 <= M<=8 -> bf16_gemv_m8_248320_2560
//   M > 8     -> bf16_linear_248320_2560
inline void launch_nvfp4_lm_head_auto(Gfx1201KernelCache& kernels, const void* d_x_bf16, const void* d_lm_head_bf16,
                                      void* d_logits_bf16, int32_t M, hipStream_t stream = nullptr) {
    if (M <= 0) return;
    if (M == 1) {
        void* args[] = {const_cast<void**>(&d_x_bf16), const_cast<void**>(&d_lm_head_bf16), &d_logits_bf16};
        kernels.launch_default("bf16_gemv_248320_2560", args, stream);
    } else if (M <= 4) {
        void* args[] = {const_cast<void**>(&d_x_bf16), const_cast<void**>(&d_lm_head_bf16), &d_logits_bf16, &M};
        kernels.launch_default("bf16_gemv_m4_248320_2560", args, stream);
    } else if (M <= 8) {
        void* args[] = {const_cast<void**>(&d_x_bf16), const_cast<void**>(&d_lm_head_bf16), &d_logits_bf16, &M};
        kernels.launch_default("bf16_gemv_m8_248320_2560", args, stream);
    } else {
        int32_t zero = 0;
        int32_t gy = (M + 15) / 16;
        void* args[] = {const_cast<void**>(&d_x_bf16), const_cast<void**>(&d_lm_head_bf16), &d_logits_bf16, &M, &zero};
        kernels.launch("bf16_linear_248320_2560", 3880, gy, 1, args, stream);
    }
}

// KRN-3: Fused single-launch tiered NVFP4 Sink + Sliding-Window decode attention + sigmoid gate for MTP draft module.
inline void launch_nvfp4_qsa_swa_sink_decode_fused(Gfx1201KernelCache& kernels, const void* d_q_bf16,
                                                   const void* d_k_cache, const void* d_v_cache,
                                                   const void* d_qg_bf16, void* d_out_bf16,
                                                   int64_t kc_host_delta, int64_t vc_host_delta,
                                                   int32_t pos, int32_t vram_prefix, int32_t sink_tokens,
                                                   int32_t window_tokens, hipStream_t stream = nullptr) {
    void* args[] = {const_cast<void**>(&d_q_bf16), const_cast<void**>(&d_k_cache),
                    const_cast<void**>(&d_v_cache), const_cast<void**>(&d_qg_bf16),
                    &d_out_bf16, &kc_host_delta, &vc_host_delta, &pos, &vram_prefix,
                    &sink_tokens, &window_tokens};
    kernels.launch("qsa_swa_sink_decode_fused_nvfp4", 24, 1, 1, args, stream);
}

// KRN-4: Row-strided SwiGLU over [rows, in_stride] reading gate at col 0..639 and up at col 640..1279.
inline void launch_nvfp4_silu_mul_640_strided(Gfx1201KernelCache& kernels, const void* d_gu_strided_bf16,
                                              void* d_out_bf16, int32_t rows, int32_t in_stride,
                                              hipStream_t stream = nullptr) {
    if (rows <= 0) return;
    int32_t gx = (rows * 80 + 255) / 256;
    void* args[] = {const_cast<void**>(&d_gu_strided_bf16), &d_out_bf16, &rows, &in_stride};
    kernels.launch("silu_mul_640_strided", gx, 1, 1, args, stream);
}

// KRN-5: Fused Gated DeltaNet with configurable input stride (16480 or 10240) and out-of-place state (S_IN -> S_OUT).
inline void launch_nvfp4_gated_deltanet_fused_strided(
    Gfx1201KernelCache& kernels, const void* d_x_bf16, const void* d_z_bf16, const void* d_a_bf16,
    const void* d_b_bf16, const void* d_alog_bf16, const void* d_dtb_bf16, const void* d_cw_bf16,
    const void* d_csi_bf16, void* d_cso_bf16, const float* d_s_in_f32, float* d_s_out_f32,
    const void* d_nw_bf16, void* d_out_bf16, int32_t seq_len, int32_t in_stride, float eps,
    hipStream_t stream = nullptr) {
    if (seq_len <= 0) return;
    void* args[] = {const_cast<void**>(&d_x_bf16),   const_cast<void**>(&d_z_bf16),
                    const_cast<void**>(&d_a_bf16),   const_cast<void**>(&d_b_bf16),
                    const_cast<void**>(&d_alog_bf16), const_cast<void**>(&d_dtb_bf16),
                    const_cast<void**>(&d_cw_bf16),  const_cast<void**>(&d_csi_bf16),
                    &d_cso_bf16,                     const_cast<float**>(&d_s_in_f32),
                    &d_s_out_f32,                    const_cast<void**>(&d_nw_bf16),
                    &d_out_bf16,                     &seq_len,
                    &in_stride,                      &eps};
    kernels.launch("gated_deltanet_fused_strided", 48, 1, 1, args, stream);
}

// KRN-6: Batched PLE conv state update for M tokens in a single launch.
inline void launch_nvfp4_ple_state_update_batch(Gfx1201KernelCache& kernels, void* d_ple_conv_state_bf16,
                                                const void* d_gvn_bf16, int32_t M, hipStream_t stream = nullptr) {
    if (M <= 0) return;
    void* args[] = {&d_ple_conv_state_bf16, const_cast<void**>(&d_gvn_bf16), &M};
    kernels.launch("ple_state_update_batch", 40, 1, 1, args, stream);
}

// KRN-7: Tiered QSA block scorer with fused sink + sliding-window pinning (eliminates qsa_pin_sink_and_window in decode).
inline void launch_nvfp4_qsa_score_blocks_tiered_pinned(
    Gfx1201KernelCache& kernels, const void* d_idx_qk, const void* d_idx_q_ln, const void* d_rope_cos_bf16,
    const void* d_rope_sin_bf16, const void* d_block_k_cache, float* d_scores, int64_t bk_host_delta_elems,
    int32_t pos, int32_t vram_blocks, int32_t score_stride, int32_t n_sink, int32_t n_local, float eps,
    int32_t M_toks = 1, hipStream_t stream = nullptr) {
    if (M_toks <= 0) return;
    int32_t max_nb = (pos + M_toks) / 4;
    int32_t score_grid = std::max(1, (max_nb + 255) / 256);
    void* args[] = {const_cast<void**>(&d_idx_qk),        const_cast<void**>(&d_idx_q_ln),
                    const_cast<void**>(&d_rope_cos_bf16), const_cast<void**>(&d_rope_sin_bf16),
                    const_cast<void**>(&d_block_k_cache), &d_scores,
                    &bk_host_delta_elems,                 &pos,
                    &vram_blocks,                         &score_stride,
                    &n_sink,                              &n_local,
                    &eps};
    kernels.launch("qsa_score_blocks_tiered_pinned", score_grid, M_toks, 1, args, stream);
}

// KRN-8: Batched 5-in-1 QSA prep + partial RoPE + tiered NVFP4 KV cache write for M tokens.
inline void launch_nvfp4_qsa_prep_and_cache_write_batch(
    Gfx1201KernelCache& kernels, const void* d_proj_bf16, const void* d_q_norm_bf16, const void* d_k_norm_bf16,
    const void* d_rope_cos_f32, const void* d_rope_sin_f32, void* d_qg_out_bf16, void* d_q_out_bf16,
    void* d_idx_qk_out_bf16, void* d_k_cache, void* d_v_cache, void* d_raw_k_cache, int64_t kc_host_delta,
    int64_t vc_host_delta, int32_t pos0, int32_t M, int32_t vram_prefix, float eps, hipStream_t stream = nullptr) {
    if (M <= 0) return;
    void* args[] = {const_cast<void**>(&d_proj_bf16),      const_cast<void**>(&d_q_norm_bf16),
                    const_cast<void**>(&d_k_norm_bf16),    const_cast<void**>(&d_rope_cos_f32),
                    const_cast<void**>(&d_rope_sin_f32),   &d_qg_out_bf16,
                    &d_q_out_bf16,                         &d_idx_qk_out_bf16,
                    &d_k_cache,                            &d_v_cache,
                    &d_raw_k_cache,                        &kc_host_delta,
                    &vc_host_delta,                        &pos0,
                    &M,                                    &vram_prefix,
                    &eps};
    kernels.launch("qsa_prep_and_cache_write_batch_nvfp4", 4, M, 1, args, stream);
}

}  // namespace strata::kernels
