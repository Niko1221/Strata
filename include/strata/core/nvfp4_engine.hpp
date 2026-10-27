// include/strata/core/nvfp4_engine.hpp - RDNA4 (gfx1201) ModelOpt NVFP4 engine for Qwen3.8-Flash-Next (qwen4_exp).
#pragma once

#include "strata/core/vmm.hpp"
#include "strata/kernels/cpu/nvfp4_modelopt.hpp"
#include "strata/kernels/gfx1201_nvfp4_kernels.hpp"
#include "strata/kernels/ngram.hpp"
#include "strata/kernels/nvfp4_mtp_kernels.hpp"
#include "strata/kernels/rope_scaling.hpp"

#include <hip/hip_runtime_api.h>

#include <cstddef>
#include <cstdint>
#include <random>
#include <string>
#include <vector>

namespace strata::core {

struct Nvfp4HcWeights {
    void* norm = nullptr;   // BF16 [10240]
    void* down = nullptr;   // BF16 [320, 10240]
    void* up = nullptr;     // BF16 [10240, 320]
    void* inj = nullptr;    // BF16 [4, 10240] (nullptr for final mixer)
    int64_t wi_delta = 0;
};

struct Nvfp4GdnWeights {
    void* in_proj_fused = nullptr; // BF16 [16480, 2560] = [qkv(10240) | z(6144) | a(48) | b(48)]
    void* out_proj = nullptr;      // BF16 [2560, 6144]
    void* conv1d = nullptr;        // BF16 [10240, 1, 4]
    void* A_log = nullptr;         // BF16 [48]
    void* dt_bias = nullptr;       // BF16 [48]
    void* norm = nullptr;          // BF16 [128]
};

struct Nvfp4QsaWeights {
    void* in_proj_fused = nullptr;   // BF16 [13952, 2560] = [q(12288) | k(512) | v(512) | idx_qk(640)]
    void* o_proj = nullptr;          // BF16 [2560, 6144]
    void* q_norm = nullptr;          // BF16 [256]
    void* k_norm = nullptr;          // BF16 [256]
    void* idx_q_ln = nullptr;        // BF16 [128]
    void* idx_k_ln = nullptr;        // BF16 [128]
};

struct Nvfp4MoeWeights {
    void* in_proj_fused = nullptr; // BF16 [1856, 2560] = [sh_gate(640) | sh_up(640) | router(512) | sh_sc(1) | pad(63)]
    void* sh_down = nullptr;       // BF16 [2560, 640]
    void* d_ptrs = nullptr;        // I64 [512, 6] on GPU (decode: non-zero for in_vram || in_gpu_cold)
    void* d_ptrs_prefill = nullptr;// I64 [512, 6] on GPU (prefill: non-zero ONLY for in_vram)
    void* d_s2 = nullptr;          // F32 [512, 3] on GPU
    int vram_experts = 0;          // Number of experts in this layer resident in VRAM

    // Zero-D2H Adaptive VRAM Expert Cache state (routing hot experts into VRAM and cold ones out to host RAM)
    int n_pinned = 0;
    int n_dyn = 0;
    uint8_t* vram_base = nullptr;
    uint8_t* dyn_vram_base = nullptr;
    uint8_t* host_cpu_ptr[512]{};
    uint8_t* host_dev_ptr[512]{};
    bool in_vram[512]{};
    bool in_gpu_cold[512]{};       // True if cold expert is assigned to GPU PCIe in decode (pcie_frac)
    int16_t dyn_slot[512]{};       // -1 if pinned or in host RAM; 0..n_dyn-1 if in a dynamic VRAM slot
    int32_t slot_exp[64]{};        // Expert ID currently occupying dynamic slot s
    uint32_t slot_last_used[64]{}; // Step counter when dynamic slot s was last used
    uint32_t hit_count[512]{};     // Cumulative routing hits for expert e
    int64_t* h_ptrs = nullptr;          // Pinned host mirror of d_ptrs [512 * 6]
    int64_t* h_ptrs_prefill = nullptr;  // Pinned host mirror of d_ptrs_prefill [512 * 6]
    std::vector<float> h_s2;       // Host copy of s2 [512 * 3] (gate_s2, up_s2, down_s2)
};

struct Nvfp4PleWeights {
    void* key_proj = nullptr;      // BF16 [10240, 2560]
    void* value_proj = nullptr;    // BF16 [2560, 2560]
    void* norm_key = nullptr;      // BF16 [10240]
    void* norm_query = nullptr;    // BF16 [10240]
    void* norm_conv = nullptr;     // BF16 [10240]
    void* conv_weight = nullptr;   // BF16 [10240, 1, 4]
    void* weight_scale = nullptr;  // BF16 [1]
    int ple_fd = -1;
    uint64_t shard_offsets[128]{};
};

struct Nvfp4LayerWeights {
    bool is_qsa = false;
    Nvfp4HcWeights attn_hc;
    Nvfp4HcWeights mlp_hc;
    Nvfp4GdnWeights gdn;
    Nvfp4QsaWeights qsa;
    Nvfp4MoeWeights moe;
};

struct Nvfp4MtpWeights {
    bool available = false;
    void* pre_fc_norm_embedding = nullptr; // BF16 [2560]
    void* pre_fc_norm_hidden = nullptr;    // BF16 [10240]
    void* fc_embedding = nullptr;          // BF16 [2560, 2560]
    void* fc_hidden = nullptr;             // BF16 [2560, 2560]
    Nvfp4HcWeights attn_hc;
    Nvfp4QsaWeights qsa;
    Nvfp4HcWeights mlp_hc;
    Nvfp4MoeWeights moe;
    Nvfp4HcWeights final_hc;
};

struct Nvfp4SamplingParams {
    float temperature = 0.0f;
    float top_p = 1.0f;
    int32_t top_k = 0;
    float presence_penalty = 0.0f;
    float frequency_penalty = 0.0f;
    float repeat_penalty = 1.0f;
    uint64_t seed = 42;

    bool is_greedy() const {
        return temperature <= 0.0f || top_k == 1 || top_p <= 0.0f;
    }
};

class Qwen4ExpGfx1201Engine {
public:
    Qwen4ExpGfx1201Engine(std::string model_dir, std::string cache_dir, std::string expert_profile_path,
                          int vram_expert_layers, int max_seq, int kv_resident, bool kv_grow,
                          strata::kernels::RopeScaling rope_scaling, bool rope_auto_yarn,
                          int vram_reserve_mib);
    ~Qwen4ExpGfx1201Engine();

    Qwen4ExpGfx1201Engine(const Qwen4ExpGfx1201Engine&) = delete;
    Qwen4ExpGfx1201Engine& operator=(const Qwen4ExpGfx1201Engine&) = delete;

    void init(bool serve_mode = false);

    int32_t step_token(int32_t token_id, int32_t prev0, int32_t prev1, int32_t pos, bool compute_logits,
                       std::vector<float>* dump_logits_out = nullptr,
                       const Nvfp4SamplingParams* sp = nullptr,
                       const std::vector<int32_t>* history = nullptr,
                       std::mt19937_64* rng = nullptr);

    int32_t step_prefill_batch(const int32_t* tokens, const int32_t* prev0s, const int32_t* prev1s,
                               int32_t M, int32_t pos0, bool compute_logits,
                               std::vector<float>* dump_logits_out = nullptr,
                               int32_t* verify_next_toks_out = nullptr,
                               const Nvfp4SamplingParams* sp = nullptr,
                               const std::vector<int32_t>* history = nullptr,
                               std::mt19937_64* rng = nullptr,
                               const int32_t* verify_draft_toks = nullptr);

    void save_recurrent_ckpt();
    void restore_recurrent_ckpt();
    void finish_verify_step(int32_t M, int32_t n_adv);
    void reset_session();
    void save_turn_ckpt(const std::vector<int32_t>& ids, int32_t next_tok);
    void restore_turn_ckpt(std::vector<int32_t>& history_out, int32_t& next_tok_out);

    bool has_mtp() const { return mtp_.available; }
    void mtp_catchup_kv(const std::vector<int32_t>& history, int32_t target_kv_pos);
    void draft_mtp(int32_t next_tok, const std::vector<int32_t>& history, int32_t max_drafts,
                   float min_prob, std::vector<int32_t>& draft_out,
                   std::vector<float>* probs_out = nullptr);

    float last_target_prob() const { return last_target_prob_; }
    float last_target_margin() const { return last_target_margin_; }
    const std::vector<int32_t>& turn_ckpt_ids() const { return turn_ckpt_ids_; }
    int total_vram_experts() const { return total_vram_experts_; }
    int32_t max_seq() const { return max_seq_; }
    int32_t vram_cap() const { return vram_cap_; }

private:
    static constexpr int32_t kMaxBatch = 4096;
    static constexpr int32_t kNumQsaCaches = 13; // 12 main model QSA layers + 1 MTP QSA layer (index 12)
    static constexpr int32_t kMtpQsaIdx = 12;
    static constexpr int32_t kMtpRingCap = 4096;
    static constexpr int32_t kMtpRingMask = kMtpRingCap - 1;

    int32_t sample_from_device_logits(const Nvfp4SamplingParams& sp,
                                      const std::vector<int32_t>* history,
                                      std::mt19937_64& rng);
    int32_t sample_row_from_host_bf16_logits(const uint16_t* h_bf_row,
                                             const Nvfp4SamplingParams& sp,
                                             const std::vector<int32_t>* history,
                                             size_t hist_len,
                                             std::mt19937_64& rng,
                                             int32_t draft_tok = -1,
                                             float* out_prob = nullptr,
                                             float* out_margin = nullptr);
    void fetch_ple_rows_cached(const int32_t* tokens, const int32_t* prev0s, const int32_t* prev1s,
                               int32_t M, uint8_t* h_ple_dst);
    int32_t run_cpu_moe_layer(int l, int32_t M, bool is_prefill, bool use_gpu_cold_cap = false);
    int32_t run_mtp_cpu_moe_decode();
    void run_mtp_kv_batch(const int32_t* next_toks, int32_t cell0, int32_t M);
    int32_t run_mtp_step_token(int32_t next_tok, int32_t cell, bool from_ring,
                               float* prob_out = nullptr, float* margin_out = nullptr,
                               int32_t ple_prev0 = -2, int32_t ple_prev1 = -2);
    void adapt_vram_experts(const int32_t* routed_ids, int32_t M, uint32_t min_hits, int max_swaps,
                            int32_t valid_toks = -1);
    void* gpu_alloc(size_t bytes, bool zero = false);
    void* host_pinned_alloc(size_t bytes);
    void alloc_state_and_scratch();
    void populate_rope_range(int32_t p_begin, int32_t p_end);
    void init_rope_tables();
    void ensure_rope_capacity(int32_t needed_seq);
    strata::core::VmmChunk take_one_dynamic_expert_chunk();
    bool map_kv_vmm_with_reclaim(size_t offset, size_t length);
    void ensure_context_capacity(int32_t needed_seq);
    void ensure_verify_capacity(int32_t needed_m);
    void load_weights_from_safetensors();

    std::string model_dir_;
    std::string expert_profile_path_;
    strata::kernels::Gfx1201KernelCache kernels_;
    strata::kernels::cpu::Nvfp4CpuPool cpu_pool_;
    float pcie_frac_ = 0.60f;
    int vram_expert_layers_ = 14;
    int vram_reserve_mib_ = 1536;
    int total_vram_experts_ = 0;
    int32_t max_seq_ = 1048576;
    int32_t max_blocks_ = 262144;
    int32_t kv_resident_ = 32768;
    int32_t vram_cap_ = 32768;
    int32_t vram_prefix_ = 28672;
    int32_t vram_blocks_ = 32768;
    int32_t kv_mapped_seq_ = 0;
    bool kv_grow_ = true;
    uint32_t step_counter_ = 0;

    strata::kernels::RopeScaling rope_scaling_{};
    bool rope_auto_yarn_ = true;
    int32_t rope_populated_seq_ = 0;
    size_t rope_stride_f32_bytes_ = 0;
    size_t rope_stride_bf16_bytes_ = 0;
    size_t rope_mapped_bytes_ = 0;
    uint8_t* host_rope_mmap_ = nullptr;
    size_t host_rope_mmap_bytes_ = 0;

    strata::core::VmmRange kv_vmm_range_;
    bool kv_vmm_ok_ = false;
    size_t vram_kv_stride_bytes_ = 0;
    size_t vram_bk_stride_bytes_ = 0;
    size_t vram_kv_mapped_bytes_ = 0;
    size_t vram_bk_mapped_bytes_ = 0;

    strata::core::VmmRange exp_vmm_range_;
    bool exp_vmm_ok_ = false;
    size_t exp_vmm_mapped_bytes_ = 0;
    size_t exp_layer_dyn_slot_off_[48]{};

    uint8_t* host_kv_mmap_ = nullptr;
    size_t host_kv_mmap_bytes_ = 0;
    size_t host_kv_stride_bytes_ = 0;
    size_t host_kv_mapped_bytes_ = 0;

    uint8_t* host_bk_mmap_ = nullptr;
    size_t host_bk_mmap_bytes_ = 0;
    size_t host_bk_stride_bytes_ = 0;
    size_t host_bk_mapped_bytes_ = 0;

    int64_t kc_host_delta_[kNumQsaCaches]{};
    int64_t vc_host_delta_[kNumQsaCaches]{};
    int64_t bk_host_delta_elems_[kNumQsaCaches]{};
    void* d_qsa_scores_ = nullptr;
    double t_wait_acc_ms_ = 0.0;
    double t_cpu_acc_ms_ = 0.0;
    int n_cpu_exp_acc_ = 0;
    int n_gpu_pcie_acc_ = 0;
    int n_vram_exp_acc_ = 0;
    hipEvent_t ev_prof_[9]{};

    strata::kernels::PleConsts ple_consts_{};
    Nvfp4PleWeights ple_;
    void* d_embed_tokens_ = nullptr;
    void* d_lm_head_ = nullptr;
    Nvfp4HcWeights final_hc_;
    Nvfp4LayerWeights layers_[48];
    Nvfp4MtpWeights mtp_;
    bool mtp_hnorm_per_stream_ = false;
    int32_t mtp_topk_ = 6;
    int32_t mtp_kv_pos_ = 0;
    int32_t turn_ckpt_mtp_kv_pos_ = 0;
    float last_target_prob_ = 1.0f;
    float last_target_margin_ = 10.0f;
    int32_t verify_cap_ = 0;
    std::vector<strata::kernels::Nvfp4ArgmaxSummary> verify_summaries_;
    void* d_mtp_r_ring_ = nullptr;
    void* d_mtp_r_state_ = nullptr;
    void* d_mtp_emb_norm_ = nullptr;
    void* d_mtp_e2_ = nullptr;
    void* d_mtp_h_norm_ = nullptr;
    void* d_mtp_h2_ = nullptr;
    void* d_mtp_dvocab_ = nullptr;
    int32_t n_mtp_dvocab_ = 0;
    std::vector<int32_t> h_mtp_dvocab_;
    std::vector<uint8_t> mtp_in_dvocab_;
    void* d_argmax_summary_ = nullptr;
    void* d_moe_gpu_align_ids_ = nullptr;
    alignas(64) float mtp_act_f32_[2560];

    static constexpr size_t kPleCacheSets = 65536;
    static constexpr size_t kPleCacheWays = 4;
    static constexpr size_t kPleCacheSlots = kPleCacheSets * kPleCacheWays;
    struct PleCacheEntry {
        uint64_t off = ~0ULL;
        uint8_t data[160];
    };
    std::vector<PleCacheEntry> ple_cache_;
    uint32_t ple_cache_clock_ = 0;

    // State & Scratch
    hipEvent_t ev_mixed_ = nullptr;
    hipEvent_t ev_route_ = nullptr;
    void* d_route_bell_ = nullptr;
    void* h_route_bell_ = nullptr;
    void* d_cpu_upload_ = nullptr;
    void* d_cpu_upload_host_dev_ = nullptr;
    void* h_cpu_upload_ = nullptr;
    std::vector<strata::kernels::cpu::CpuExpertJob> cpu_jobs_;
    std::vector<int32_t> h_all_routed_ids_;
    int8_t* act_q_ = nullptr;
    float* act_d_ = nullptr;
    int32_t* act_nb_ = nullptr;
    std::vector<uint64_t> ple_miss_offs_;
    std::vector<uint8_t*> ple_miss_cache_dsts_;
    std::vector<uint8_t*> ple_miss_user_dsts_;
    struct MoeMissItem {
        int32_t e;
        int32_t pair;
    };
    std::vector<MoeMissItem> moe_misses_;
    std::vector<MoeMissItem> moe_misses_sorted_;
    std::vector<uint8_t> moe_tok_needed_;
    std::vector<int32_t> moe_upload_pairs_;
    void* d_phase_arena_ = nullptr;

    void* d_step_meta_ = nullptr;
    void* h_step_meta_ = nullptr;
    void* d_tok_id_ = nullptr;
    void* d_pos_i32_ = nullptr;
    void* d_seq_k_i32_ = nullptr;
    void* d_step_out_ = nullptr;
    void* d_next_tok_ = nullptr;
    void* d_x_stream_a_ = nullptr;
    void* d_x_stream_b_ = nullptr;
    void* d_hc_h_ = nullptr;
    void* d_hc_rstd_ = nullptr;
    void* d_attn_inj_ = nullptr;
    void* d_mlp_inj_ = nullptr;
    void* d_mixed_ = nullptr;
    void* d_attn_out_ = nullptr;
    void* d_mlp_out_ = nullptr;
    void* d_final_hidden_ = nullptr;
    void* d_logits_bf16_ = nullptr;
    void* d_proj_scratch_ = nullptr;

    void* d_ple_rows_u8_ = nullptr;
    void* d_ple_embed_ = nullptr;
    void* d_ple_k_ = nullptr;
    void* d_ple_v_ = nullptr;
    void* d_ple_gv_ = nullptr;
    void* d_ple_gvn_ = nullptr;
    void* d_ple_verify_gvn_ = nullptr;
    void* d_ple_conv_state_ = nullptr;

    void* d_gdn_qkv_ = nullptr;
    void* d_gdn_z_ = nullptr;
    void* d_gdn_a_ = nullptr;
    void* d_gdn_b_ = nullptr;
    void* d_gdn_z_b_ = nullptr;
    void* d_gdn_a_b_ = nullptr;
    void* d_gdn_b_b_ = nullptr;
    void* d_gdn_out_ = nullptr;
    void* d_gdn_conv_in_[48]{};
    void* d_gdn_conv_out_[48]{};
    void* d_gdn_state_[48]{};
    void* d_gdn_state_ckpt_[48]{};
    void* d_recurrent_arena_ = nullptr;
    void* d_recurrent_ckpt_ = nullptr;
    void* d_raw_k_arena_ = nullptr;
    void* h_turn_ckpt_ = nullptr;
    size_t recurrent_arena_bytes_ = 0;
    size_t recurrent_ckpt_bytes_ = 0;
    size_t conv_ckpt_bytes_ = 0;
    size_t raw_k_arena_bytes_ = 0;
    void* init_gdn_conv_in_[48]{};
    void* init_gdn_conv_out_[48]{};
    void* ckpt_gdn_conv_in_[48]{};
    void* ckpt_gdn_conv_out_[48]{};
    void* turn_ckpt_gdn_conv_in_[48]{};
    void* turn_ckpt_gdn_conv_out_[48]{};
    bool turn_ckpt_conv_swapped_[48]{};
    std::vector<int32_t> turn_ckpt_ids_;
    int32_t turn_ckpt_next_tok_ = -1;
    bool serve_mode_ = false;
    void* d_gdn_verify_arena_ = nullptr;
    void* d_gdn_verify_qkv_[48]{};

    void* d_qsa_qg_ = nullptr;
    void* d_qsa_q_ = nullptr;
    void* d_qsa_k_ = nullptr;
    void* d_qsa_v_ = nullptr;
    void* d_qsa_idx_qk_ = nullptr;
    void* d_qsa_k_b_ = nullptr;
    void* d_qsa_v_b_ = nullptr;
    void* d_qsa_idx_qk_b_ = nullptr;
    void* d_qsa_attn_raw_ = nullptr;
    void* d_qsa_sel_ids_ = nullptr;
    void* d_zero_mask_ = nullptr;
    void* d_attn_pm_ = nullptr;
    void* d_attn_pl_ = nullptr;
    void* d_attn_pa_ = nullptr;
    void* d_k_cache_[kNumQsaCaches]{};
    void* d_v_cache_[kNumQsaCaches]{};
    void* d_raw_k_cache_[kNumQsaCaches]{};
    void* d_block_k_cache_[kNumQsaCaches]{};

    void* d_rope_cos_f32_ = nullptr;
    void* d_rope_sin_f32_ = nullptr;
    void* d_rope_cos_bf16_ = nullptr;
    void* d_rope_sin_bf16_ = nullptr;

    void* d_sh_gu_ = nullptr;
    void* d_sh_inter_ = nullptr;
    void* d_sh_out_ = nullptr;
    void* d_sh_scale_ = nullptr;
    void* d_router_logits_ = nullptr;
    void* d_moe_ids_ = nullptr;
    void* d_moe_wts_ = nullptr;
    void* d_moe_sorted_ = nullptr;
    void* d_moe_tile_e_ = nullptr;
    void* d_moe_inter_ = nullptr;
    void* d_moe_y_ = nullptr;

    std::vector<void*> vram_allocs_;
    std::vector<void*> host_pinned_;
    std::vector<void*> host_registered_;
    std::vector<void*> host_slab_registered_;
    std::vector<void*> host_mallocs_;
};

}  // namespace strata::core
