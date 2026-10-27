// src/core/nvfp4_engine.cpp - RDNA4 (gfx1201) ModelOpt NVFP4 engine for Qwen3.8-Flash-Next (qwen4_exp).
#include "strata/core/nvfp4_engine.hpp"
#include "strata/artifact/safetensors_reader.hpp"
#include "strata/core/expert_cache.hpp"
#include "strata/kernels/cpu/pool.hpp"

#include <immintrin.h>

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <fcntl.h>
#include <filesystem>
#include <sys/mman.h>
#include <thread>
#include <unistd.h>
#include <unordered_map>
#include <utility>

namespace strata::core {
namespace {

using strata::artifact::TensorSlice;
using strata::artifact::parse_safetensors_header;
using strata::artifact::pread_exact;
using strata::kernels::cpu::CpuExpertJob;
using strata::kernels::cpu::EXP_OFF_DOWN_S;
using strata::kernels::cpu::EXP_OFF_DOWN_W;
using strata::kernels::cpu::EXP_OFF_GATE_S;
using strata::kernels::cpu::EXP_OFF_GATE_W;
using strata::kernels::cpu::EXP_OFF_UP_S;
using strata::kernels::cpu::EXP_OFF_UP_W;
using strata::kernels::cpu::EXP_S_BYTES;
using strata::kernels::cpu::EXP_STRIDE_BYTES;
using strata::kernels::cpu::EXP_W_BYTES;
using strata::kernels::cpu::kMaxJobToks;
using strata::kernels::cpu::quantize_act_bf16_2560;

#define HIP_CHECK(call)                                                                                     \
    do {                                                                                                    \
        hipError_t err__ = (call);                                                                          \
        if (err__ != hipSuccess) {                                                                          \
            std::fprintf(stderr, "HIP error at %s:%d: %s (%d)\n", __FILE__, __LINE__,                       \
                         hipGetErrorString(err__), (int) err__);                                            \
            std::exit(1);                                                                                   \
        }                                                                                                   \
    } while (0)

inline uint16_t f32_to_bf16_bits(float v) {
    uint32_t bits = 0;
    std::memcpy(&bits, &v, sizeof(bits));
    uint32_t lsb = (bits >> 16) & 1u;
    bits += 0x7FFFu + lsb;
    return static_cast<uint16_t>(bits >> 16);
}

inline float bf16_bits_to_f32(uint16_t b) {
    uint32_t bits = static_cast<uint32_t>(b) << 16;
    float v = 0.0f;
    std::memcpy(&v, &bits, sizeof(v));
    return v;
}

}  // namespace

Qwen4ExpGfx1201Engine::Qwen4ExpGfx1201Engine(std::string model_dir, std::string cache_dir,
                                             std::string expert_profile_path, int vram_expert_layers,
                                             int max_seq, int kv_resident, bool kv_grow,
                                             strata::kernels::RopeScaling rope_scaling, bool rope_auto_yarn,
                                             int vram_reserve_mib)
    : model_dir_(std::move(model_dir)),
      expert_profile_path_(std::move(expert_profile_path)),
      kernels_(std::move(cache_dir)),
      vram_expert_layers_(vram_expert_layers),
      vram_reserve_mib_(std::max(256, vram_reserve_mib)),
      max_seq_(std::max(4096, (max_seq + 3) & ~3)),
      max_blocks_(max_seq_ / 4),
      kv_resident_(std::max(8192, (kv_resident + 4095) & ~4095)),
      kv_grow_(kv_grow),
      rope_scaling_(rope_scaling),
      rope_auto_yarn_(rope_auto_yarn) {
    vram_cap_ = std::min(max_seq_, kv_resident_);
    vram_prefix_ = (max_seq_ > vram_cap_) ? (vram_cap_ - 4096) : vram_cap_;
    vram_blocks_ = strata::core::vmm_available()
                       ? max_blocks_
                       : std::min<int32_t>(max_blocks_, std::max<int32_t>(vram_cap_ / 4, 32768));
}

Qwen4ExpGfx1201Engine::~Qwen4ExpGfx1201Engine() {
    cpu_pool_.stop();
    if (ev_mixed_) (void) hipEventDestroy(ev_mixed_);
    if (ev_route_) (void) hipEventDestroy(ev_route_);
    if (ple_.ple_fd >= 0) ::close(ple_.ple_fd);
    for (void* p : host_slab_registered_) {
        (void) hipHostUnregister(p);
    }
    if (host_kv_mmap_ && host_kv_mmap_ != MAP_FAILED) {
        (void) ::munmap(host_kv_mmap_, host_kv_mmap_bytes_);
    }
    if (host_bk_mmap_ && host_bk_mmap_ != MAP_FAILED) {
        (void) ::munmap(host_bk_mmap_, host_bk_mmap_bytes_);
    }
    if (host_rope_mmap_ && host_rope_mmap_ != MAP_FAILED) {
        (void) ::munmap(host_rope_mmap_, host_rope_mmap_bytes_);
    }
    kv_vmm_range_.release();
    exp_vmm_range_.release();
    for (void* p : vram_allocs_) (void) hipFree(p);
    for (void* p : host_pinned_) (void) hipHostFree(p);
    for (void* p : host_registered_) {
        (void) hipHostUnregister(p);
        std::free(p);
    }
    for (void* p : host_mallocs_) {
        std::free(p);
    }
}

void Qwen4ExpGfx1201Engine::init(bool serve_mode) {
    serve_mode_ = serve_mode;
    FILE* log_fp = serve_mode_ ? stderr : stdout;
    auto t0 = std::chrono::steady_clock::now();
    if (const char* env_pf = std::getenv("STRATA_NVFP4_PCIE_FRAC")) {
        pcie_frac_ = std::clamp(static_cast<float>(std::atof(env_pf)), 0.0f, 1.0f);
    }
    cpu_pool_.init();
    std::fprintf(log_fp,
                 "[gfx1201-nvfp4] Initializing embedded gfx1201 kernels, %d-core AVX-512 VNNI pool (pcie_frac=%.2f), and model buffers...\n",
                 cpu_pool_.num_threads(), pcie_frac_);
    std::fflush(log_fp);

    for (size_t i = 0; i < strata::kernels::embedded_kernel_count(); ++i) {
        std::string_view key = strata::kernels::embedded_kernels_begin()[i].key;
        if (key == "argmax_248320" || key == "bf16_gemv_12800_2560" ||
            key == "bf16_gemv_m4_12800_2560" || key == "bf16_linear_12800_2560" ||
            key == "gdn_proj_split" || key == "moe_sh_gu_extract" ||
            key == "ple_state_update" || key == "qsa_score_blocks_tiered" ||
            key == "silu_mul_640") {
            continue;
        }
        (void) kernels_.get(std::string(key));
    }

    alloc_state_and_scratch();
    init_rope_tables();
    load_weights_from_safetensors();

    double dt = std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count();
    std::fprintf(log_fp,
                 "[gfx1201-nvfp4] Model ready in %.2f s (%d profile-ranked experts in VRAM, %d in pinned host RAM; "
                 "context window: %d -> %d max [VRAM resident cap: %d, prefix: %d + 4096 ring, host overflow: %d])\n",
                 dt, total_vram_experts_, 48 * 512 - total_vram_experts_,
                 kv_mapped_seq_, max_seq_, vram_cap_, vram_prefix_, std::max(0, max_seq_ - vram_prefix_));
    std::fflush(log_fp);
}

int32_t Qwen4ExpGfx1201Engine::step_token(int32_t token_id, int32_t prev0, int32_t prev1, int32_t pos,
                                          bool compute_logits, std::vector<float>* dump_logits_out,
                                          const Nvfp4SamplingParams* sp,
                                          const std::vector<int32_t>* history,
                                          std::mt19937_64* rng) {
    cpu_pool_.ensure_host_pinned();
    ensure_context_capacity(pos + 1);
    int32_t* meta_i32 = static_cast<int32_t*>(h_step_meta_);
    meta_i32[0] = token_id;
    meta_i32[1] = pos;
    meta_i32[2] = pos + 1;
    meta_i32[3] = 0;
    HIP_CHECK(hipMemcpyAsync(d_step_meta_, h_step_meta_, 16, hipMemcpyHostToDevice, nullptr));

    int32_t one = 1;
    float eps = 1e-6f;
    void* d_cur_ids = static_cast<uint8_t*>(d_route_bell_) + 2560 * sizeof(uint16_t);
    void* d_cur_wts = static_cast<uint8_t*>(d_cur_ids) + 10 * sizeof(int32_t);

    void* cur_x = d_x_stream_a_;
    void* alt_x = d_x_stream_b_;
    bool has_prev_update = false;
    void* prev_out = nullptr;
    void* prev_inj = nullptr;

    const bool prof = (std::getenv("STRATA_PROFILE_GFX1201") != nullptr && compute_logits);
    for (int l = 0; l < 48; ++l) {
        Nvfp4LayerWeights& lw = layers_[l];
        if (prof) kernels_.set_profile(true, "");

        if (l == 1) {
            {
                void* args[] = {&cur_x, &prev_out, &prev_inj, &alt_x};
                kernels_.launch_default("hc_residual_add", args);
                std::swap(cur_x, alt_x);
                has_prev_update = false;
            }
            void* d_ple_v_dec = static_cast<uint16_t*>(d_ple_k_) + 10240;
            {
                void* args_kv[] = {&d_ple_rows_u8_, &ple_.weight_scale, &ple_.key_proj, &d_ple_k_};
                kernels_.launch_default("ple_dequant_gemv_12800_2560", args_kv);
            }
            {
                void* args[] = {&d_ple_k_, &d_ple_v_dec, &cur_x, &ple_.norm_key, &ple_.norm_query,
                                &ple_.norm_conv, &d_ple_gv_, &d_ple_gvn_, &eps};
                kernels_.launch_default("ple_gate", args);
            }
            {
                void* args[] = {&cur_x, &d_ple_gv_, &d_ple_gvn_, &d_ple_conv_state_, &ple_.conv_weight, &alt_x};
                kernels_.launch_default("ple_conv_add_decode", args);
                std::swap(cur_x, alt_x);
            }
        }

        if (l == 20 && ev_prof_[0]) HIP_CHECK(hipEventRecord(ev_prof_[0], nullptr));
        {
            int64_t wi_delta = lw.attn_hc.wi_delta;
            if (l == 0) {
                void* args[] = {&d_tok_id_, &d_embed_tokens_, &cur_x, &lw.attn_hc.norm, &lw.attn_hc.down,
                                &d_hc_h_, &d_attn_inj_, &d_hc_rstd_, &wi_delta, &eps};
                kernels_.launch_default("embed_hc_premix_decode_l0", args);
            } else if (!has_prev_update) {
                void* args[] = {&cur_x, &cur_x, &cur_x, &cur_x, &lw.attn_hc.norm, &lw.attn_hc.down,
                                &d_hc_h_, &d_attn_inj_, &d_hc_rstd_, &wi_delta, &eps};
                kernels_.launch_default("hc_premix_decode_c1_u0", args);
            } else {
                void* args[] = {&cur_x, &prev_out, &prev_inj, &alt_x, &lw.attn_hc.norm, &lw.attn_hc.down,
                                &d_hc_h_, &d_attn_inj_, &d_hc_rstd_, &wi_delta, &eps};
                kernels_.launch_default("hc_premix_decode_c1_u1", args);
                std::swap(cur_x, alt_x);
            }
            void* m_args[] = {&cur_x, &lw.attn_hc.norm, &d_hc_rstd_, &d_hc_h_, &lw.attn_hc.up, &d_mixed_, &one};
            kernels_.launch_default("hc_mix_decode", m_args);
        }
        if (l == 20 && ev_prof_[1]) HIP_CHECK(hipEventRecord(ev_prof_[1], nullptr));

        if (!lw.is_qsa) {
            {
                void* a1[] = {&d_mixed_, &lw.gdn.in_proj_fused, &d_gdn_qkv_};
                kernels_.launch_default("bf16_gemv_16480_2560", a1);
            }
            if (l == 20 && ev_prof_[2]) HIP_CHECK(hipEventRecord(ev_prof_[2], nullptr));
            {
                void* csi = d_gdn_conv_in_[l];
                void* cso = d_gdn_conv_out_[l];
                void* st = d_gdn_state_[l];
                void* args[] = {&d_gdn_qkv_, &d_gdn_z_, &d_gdn_a_, &d_gdn_b_, &lw.gdn.A_log, &lw.gdn.dt_bias,
                                &lw.gdn.conv1d, &csi, &cso, &st, &lw.gdn.norm, &d_gdn_out_, &one, &eps};
                kernels_.launch_default("gated_deltanet_fused", args);
                std::swap(d_gdn_conv_in_[l], d_gdn_conv_out_[l]);
            }
            if (l == 20 && ev_prof_[3]) HIP_CHECK(hipEventRecord(ev_prof_[3], nullptr));
            {
                void* args[] = {&d_gdn_out_, &lw.gdn.out_proj, &d_attn_out_};
                kernels_.launch_default("bf16_gemv_2560_6144", args);
            }
            if (l == 20 && ev_prof_[4]) HIP_CHECK(hipEventRecord(ev_prof_[4], nullptr));
        } else {
            const int qsa_idx = l / 4;
            {
                void* a1[] = {&d_mixed_, &lw.qsa.in_proj_fused, &d_qsa_qg_};
                kernels_.launch_default("bf16_gemv_13952_2560", a1);
            }
            void* kc = d_k_cache_[qsa_idx];
            void* vc = d_v_cache_[qsa_idx];
            void* rkc = d_raw_k_cache_[qsa_idx];
            void* bkc = d_block_k_cache_[qsa_idx];
            int64_t kc_delta = kc_host_delta_[qsa_idx];
            int64_t vc_delta = vc_host_delta_[qsa_idx];
            int64_t bk_delta = bk_host_delta_elems_[qsa_idx];
            {
                void* args[] = {&d_qsa_qg_, &lw.qsa.q_norm, &lw.qsa.k_norm, &d_rope_cos_f32_, &d_rope_sin_f32_,
                                &d_qsa_q_, &kc, &vc, &rkc, &kc_delta, &vc_delta, &pos, &vram_prefix_, &eps};
                kernels_.launch_default("qsa_prep_and_cache_write_nvfp4", args);
            }
            if ((pos + 1) % 4 == 0) {
                int32_t b_start = pos / 4;
                int32_t count = 1;
                void* args[] = {&rkc, &lw.qsa.idx_k_ln, &d_rope_cos_bf16_, &d_rope_sin_bf16_, &bkc,
                                &bk_delta, &b_start, &count, &vram_blocks_, &eps};
                kernels_.launch("qsa_block_keys_tiered", 1, 1, 1, args);
            }
            int32_t Lk = pos + 1;
            int32_t chunk = 256;
            if (Lk <= 256) {
                void* af[] = {&d_qsa_q_, &kc, &vc, &d_qsa_qg_, &d_qsa_attn_raw_, &d_seq_k_i32_};
                kernels_.launch_default("attn_decode_fused_nvfp4", af);
            } else if (Lk <= 2048) {
                int32_t splits = (Lk + chunk - 1) / chunk;
                void* ap[] = {&d_qsa_q_, &kc, &vc, &d_zero_mask_, &d_attn_pm_, &d_attn_pl_, &d_attn_pa_,
                              &d_seq_k_i32_, &chunk, &splits};
                kernels_.launch("attn_decode_partial_nvfp4", splits * 24, 1, 1, ap);
                void* ac[] = {&d_attn_pm_, &d_attn_pl_, &d_attn_pa_, &d_qsa_qg_, &d_qsa_attn_raw_, &splits};
                kernels_.launch("attn_decode_combine", 24, 1, 1, ac);
            } else {
                if ((pos + 1) / 4 > 512) {
                    strata::kernels::launch_nvfp4_qsa_score_blocks_tiered_pinned(
                        kernels_, d_qsa_idx_qk_, lw.qsa.idx_q_ln, d_rope_cos_bf16_, d_rope_sin_bf16_,
                        bkc, static_cast<float*>(d_qsa_scores_), bk_delta, pos, vram_blocks_,
                        max_blocks_, 16, 64, eps, 1, nullptr);
                }
                void* atk[] = {&d_qsa_scores_, &d_qsa_sel_ids_, &pos, &max_blocks_};
                kernels_.launch("qsa_topk_blocks", 1, 1, 1, atk);
                int32_t splits = 9;
                void* ap[] = {&d_qsa_q_, &kc, &vc, &d_qsa_sel_ids_, &d_pos_i32_, &d_attn_pm_, &d_attn_pl_,
                              &d_attn_pa_, &kc_delta, &vc_delta, &vram_prefix_, &chunk, &splits};
                kernels_.launch("qsa_sparse_decode_partial_nvfp4", splits * 24, 1, 1, ap);
                void* ac[] = {&d_attn_pm_, &d_attn_pl_, &d_attn_pa_, &d_qsa_qg_, &d_qsa_attn_raw_, &splits};
                kernels_.launch("attn_decode_combine", 24, 1, 1, ac);
            }
            {
                void* args[] = {&d_qsa_attn_raw_, &lw.qsa.o_proj, &d_attn_out_};
                kernels_.launch_default("bf16_gemv_2560_6144", args);
            }
        }

        {
            int64_t wi_delta = lw.mlp_hc.wi_delta;
            void* args[] = {&cur_x, &d_attn_out_, &d_attn_inj_, &alt_x, &lw.mlp_hc.norm, &lw.mlp_hc.down,
                            &d_hc_h_, &d_mlp_inj_, &d_hc_rstd_, &wi_delta, &eps};
            kernels_.launch_default("hc_premix_decode_c1_u1", args);
            std::swap(cur_x, alt_x);
            void* m_args[] = {&cur_x, &lw.mlp_hc.norm, &d_hc_rstd_, &d_hc_h_, &lw.mlp_hc.up, &d_mixed_, &one};
            kernels_.launch_default("hc_mix_decode", m_args);
        }
        if (l == 20 && ev_prof_[5]) HIP_CHECK(hipEventRecord(ev_prof_[5], nullptr));

        {
            void* a_gu[] = {&d_mixed_, &lw.moe.in_proj_fused, &d_sh_gu_};
            kernels_.launch_default("bf16_gemv_1793_2560", a_gu);

            void* ptrs = lw.moe.d_ptrs;
            void* ptrs_vram = lw.moe.d_ptrs_prefill;
            void* a_ts[] = {&d_sh_gu_, &ptrs, &ptrs_vram, &d_cur_ids, &d_cur_wts, &d_sh_inter_};
            kernels_.launch_default("moe_topk_and_silu_mul_640", a_ts);

            HIP_CHECK(hipMemcpyAsync(h_route_bell_, d_route_bell_,
                                     2560 * sizeof(uint16_t) + 10 * sizeof(int32_t) + 10 * sizeof(float),
                                     hipMemcpyDeviceToHost, nullptr));
            HIP_CHECK(hipEventRecord(ev_route_, nullptr));
            if (l == 20 && ev_prof_[6]) HIP_CHECK(hipEventRecord(ev_prof_[6], nullptr));

            void* a_d[] = {&d_sh_inter_, &lw.moe.sh_down, &d_sh_out_};
            kernels_.launch_default("bf16_gemv_2560_640", a_d);

            void* s2 = lw.moe.d_s2;
            void* a_mgu[] = {&d_mixed_, &d_cur_ids, &ptrs, &s2, &d_moe_inter_};
            kernels_.launch_default("moe_decode_gate_up", a_mgu);
            void* a_dn[] = {&d_moe_inter_, &d_cur_ids, &ptrs, &s2, &d_moe_y_};
            kernels_.launch_default("moe_decode_down", a_dn);
            if (l == 20 && ev_prof_[7]) HIP_CHECK(hipEventRecord(ev_prof_[7], nullptr));

            if (l == 0) {
                uint8_t* h_ple_rows = static_cast<uint8_t*>(h_step_meta_) + 16;
                fetch_ple_rows_cached(&token_id, &prev0, &prev1, 1, h_ple_rows);
                HIP_CHECK(hipMemcpyAsync(d_ple_rows_u8_, h_ple_rows, 2560, hipMemcpyHostToDevice, nullptr));
            }

            int32_t has_cpu = run_cpu_moe_layer(l, 1, /*is_prefill=*/false);

            void* a_cb[] = {&d_moe_y_, &d_cur_ids, &ptrs, &d_cur_wts, &d_cpu_upload_,
                            &d_sh_out_, &d_sh_scale_, &d_mlp_out_, &has_cpu};
            kernels_.launch_default("moe_combine_decode", a_cb);
            if (l == 20 && ev_prof_[8]) HIP_CHECK(hipEventRecord(ev_prof_[8], nullptr));
        }

        has_prev_update = true;
        prev_out = d_mlp_out_;
        prev_inj = d_mlp_inj_;
    }

    if (!compute_logits) {
        if (mtp_.available) {
            void* final_xo = static_cast<uint16_t*>(d_mtp_r_ring_) + static_cast<size_t>(pos & kMtpRingMask) * 10240;
            void* args[] = {&cur_x, &prev_out, &prev_inj, &final_xo};
            kernels_.launch_default("hc_residual_add", args);
        }
        adapt_vram_experts(h_all_routed_ids_.data(), 1, /*min_hits=*/2, /*max_swaps=*/2);
        t_wait_acc_ms_ = 0.0;
        t_cpu_acc_ms_ = 0.0;
        n_cpu_exp_acc_ = 0;
        n_gpu_pcie_acc_ = 0;
        n_vram_exp_acc_ = 0;
        return -1;
    }
    if (prof) kernels_.set_profile(true, "head:");

    {
        int64_t zero_delta = 0;
        void* final_xo = mtp_.available
                             ? (static_cast<uint16_t*>(d_mtp_r_ring_) + static_cast<size_t>(pos & kMtpRingMask) * 10240)
                             : alt_x;
        void* args[] = {&cur_x, &prev_out, &prev_inj, &final_xo, &final_hc_.norm, &final_hc_.down,
                        &d_hc_h_, &d_hc_h_, &d_hc_rstd_, &zero_delta, &eps};
        kernels_.launch_default("hc_premix_decode_c0_u1", args);
        void* m_args[] = {&final_xo, &final_hc_.norm, &d_hc_rstd_, &d_hc_h_, &final_hc_.up, &d_final_hidden_, &one};
        kernels_.launch_default("hc_mix_decode", m_args);
    }

    const bool use_sampling = (sp && !sp->is_greedy() && rng);
    {
        void* a_lm[] = {&d_final_hidden_, &d_lm_head_, &d_logits_bf16_};
        kernels_.launch_default("bf16_gemv_248320_2560", a_lm);
        if (!use_sampling) {
            strata::kernels::launch_nvfp4_argmax_prob_margin_248320(
                kernels_, d_logits_bf16_, static_cast<int32_t*>(d_next_tok_),
                static_cast<strata::kernels::Nvfp4ArgmaxSummary*>(d_argmax_summary_), 1, nullptr);
        }
    }

    int32_t next_id = -1;
    auto t_sync0 = std::chrono::steady_clock::now();
    if (use_sampling) {
        next_id = sample_from_device_logits(*sp, history, *rng);
    } else {
        strata::kernels::Nvfp4ArgmaxSummary s{};
        HIP_CHECK(hipMemcpy(&s, d_argmax_summary_, sizeof(s), hipMemcpyDeviceToHost));
        next_id = s.tok_id;
        last_target_prob_ = s.prob;
        last_target_margin_ = s.margin;
    }
    auto t_adapt0 = std::chrono::steady_clock::now();
    adapt_vram_experts(h_all_routed_ids_.data(), 1, /*min_hits=*/2, /*max_swaps=*/2);
    auto t_adapt1 = std::chrono::steady_clock::now();
    if (std::getenv("STRATA_NVFP4_TIMING") != nullptr && compute_logits) {
        double sync_ms = std::chrono::duration<double, std::milli>(t_adapt0 - t_sync0).count();
        double adapt_ms = std::chrono::duration<double, std::milli>(t_adapt1 - t_adapt0).count();
        float g_hc1 = 0, g_in = 0, g_mid = 0, g_out = 0, g_hc2 = 0, g_rt = 0, g_moe = 0, g_cb = 0;
        if (ev_prof_[0]) {
            (void) hipEventElapsedTime(&g_hc1, ev_prof_[0], ev_prof_[1]);
            (void) hipEventElapsedTime(&g_in,  ev_prof_[1], ev_prof_[2]);
            (void) hipEventElapsedTime(&g_mid, ev_prof_[2], ev_prof_[3]);
            (void) hipEventElapsedTime(&g_out, ev_prof_[3], ev_prof_[4]);
            (void) hipEventElapsedTime(&g_hc2, ev_prof_[4], ev_prof_[5]);
            (void) hipEventElapsedTime(&g_rt,  ev_prof_[5], ev_prof_[6]);
            (void) hipEventElapsedTime(&g_moe, ev_prof_[6], ev_prof_[7]);
            (void) hipEventElapsedTime(&g_cb,  ev_prof_[7], ev_prof_[8]);
        }
        std::printf("[timing] wait=%.2f ms | cpu_moe=%.2f ms (cpu=%d, pcie=%d, vram=%d) | sync=%.2f | adapt=%.2f | L20(us): hc1=%.0f in=%.0f gdn=%.0f out=%.0f hc2=%.0f rt=%.0f moe=%.0f cb=%.0f\n",
                    t_wait_acc_ms_, t_cpu_acc_ms_, n_cpu_exp_acc_, n_gpu_pcie_acc_, n_vram_exp_acc_, sync_ms, adapt_ms,
                    g_hc1 * 1000.0f, g_in * 1000.0f, g_mid * 1000.0f, g_out * 1000.0f,
                    g_hc2 * 1000.0f, g_rt * 1000.0f, g_moe * 1000.0f, g_cb * 1000.0f);
        t_wait_acc_ms_ = 0.0;
        t_cpu_acc_ms_ = 0.0;
        n_cpu_exp_acc_ = 0;
        n_gpu_pcie_acc_ = 0;
        n_vram_exp_acc_ = 0;
    }

    if (prof) {
        kernels_.set_profile(false);
        kernels_.dump_profile();
    }

    if (dump_logits_out) {
        std::vector<uint16_t> bf(248320);
        HIP_CHECK(hipMemcpy(bf.data(), d_logits_bf16_, 248320 * sizeof(uint16_t), hipMemcpyDeviceToHost));
        dump_logits_out->resize(248320);
        for (size_t i = 0; i < 248320; ++i) {
            (*dump_logits_out)[i] = bf16_bits_to_f32(bf[i]);
        }
    }
    return next_id;
}

int32_t Qwen4ExpGfx1201Engine::step_prefill_batch(const int32_t* tokens, const int32_t* prev0s, const int32_t* prev1s,
                                                  int32_t M, int32_t pos0, bool compute_logits,
                                                  std::vector<float>* dump_logits_out,
                                                  int32_t* verify_next_toks_out,
                                                  const Nvfp4SamplingParams* sp,
                                                  const std::vector<int32_t>* history,
                                                  std::mt19937_64* rng,
                                                  const int32_t* verify_draft_toks) {
    if (M == 1 && !verify_next_toks_out) {
        return step_token(tokens[0], prev0s[0], prev1s[0], pos0, compute_logits, dump_logits_out, sp, history, rng);
    }
    cpu_pool_.ensure_host_pinned();
    ensure_context_capacity(pos0 + M);
    if (verify_next_toks_out) {
        ensure_verify_capacity(M);
    }

    uint8_t* h_ple_rows = static_cast<uint8_t*>(h_step_meta_) + 16;
    uint8_t* h_batch_toks = h_ple_rows + static_cast<size_t>(M) * 2560;
    std::memcpy(h_batch_toks, tokens, static_cast<size_t>(M) * sizeof(int32_t));
    void* d_batch_toks = static_cast<uint8_t*>(d_ple_rows_u8_) + static_cast<size_t>(M) * 2560;
    HIP_CHECK(hipMemcpyAsync(d_batch_toks, h_batch_toks, static_cast<size_t>(M) * sizeof(int32_t),
                             hipMemcpyHostToDevice, nullptr));

    int32_t one = 1;
    float eps = 1e-6f;
    const int32_t m_tiles64 = (M + 63) / 64;
    const bool use_wmma_hc = (M >= 16);
    void* d_cur_ids = static_cast<uint8_t*>(d_route_bell_) + static_cast<size_t>(M) * 2560 * sizeof(uint16_t);

    if (use_wmma_hc) {
        void* args[] = {&d_batch_toks, &d_embed_tokens_, &d_x_stream_a_};
        kernels_.launch("embed_hc_gather", M, 1, 1, args);
    }

    void* cur_x = d_x_stream_a_;
    void* alt_x = d_x_stream_b_;
    bool has_prev_update = false;
    void* prev_out = nullptr;
    void* prev_inj = nullptr;

    for (int l = 0; l < 48; ++l) {
        Nvfp4LayerWeights& lw = layers_[l];

        if (l == 1) {
            {
                void* args[] = {&cur_x, &prev_out, &prev_inj, &alt_x};
                kernels_.launch("hc_residual_add", 40, M, 1, args);
                std::swap(cur_x, alt_x);
                has_prev_update = false;
            }
            if (M <= 4) {
                void* args_kv[] = {&d_ple_rows_u8_, &ple_.weight_scale, &ple_.key_proj, &d_ple_k_, &d_ple_v_, &M};
                kernels_.launch_default("ple_dequant_gemv_m4_12800_2560", args_kv);
            } else {
                {
                    void* args_dq[] = {&d_ple_rows_u8_, &ple_.weight_scale, &d_ple_embed_, &M};
                    kernels_.launch("ple_embed_dequant", M, 1, 1, args_dq);
                }
                {
                    strata::kernels::launch_nvfp4_linear_auto(
                        kernels_, "bf16_linear_10240_2560", "bf16_gemv_10240_2560", "bf16_gemv_m4_10240_2560",
                        d_ple_embed_, ple_.key_proj, d_ple_k_, 10240, M);
                    strata::kernels::launch_nvfp4_linear_auto(
                        kernels_, "bf16_linear_2560_2560", "bf16_gemv_2560_2560", "bf16_gemv_m4_2560_2560",
                        d_ple_embed_, ple_.value_proj, d_ple_v_, 2560, M);
                }
            }
            {
                void* args[] = {&d_ple_k_, &d_ple_v_, &cur_x, &ple_.norm_key, &ple_.norm_query,
                                &ple_.norm_conv, &d_ple_gv_, &d_ple_gvn_, &eps};
                kernels_.launch("ple_gate", M, 1, 1, args);
            }
            {
                void* args[] = {&cur_x, &d_ple_gv_, &d_ple_gvn_, &d_ple_conv_state_, &ple_.conv_weight, &alt_x, &M};
                kernels_.launch("ple_conv_add", (M * 1280 + 255) / 256, 1, 1, args);
                std::swap(cur_x, alt_x);
            }
            strata::kernels::launch_nvfp4_ple_state_update_batch(kernels_, d_ple_conv_state_, d_ple_gvn_, M);
        }

        {
            int64_t wi_delta = lw.attn_hc.wi_delta;
            if (!has_prev_update) {
                if (use_wmma_hc) {
                    void* args[] = {&cur_x, &cur_x, &cur_x, &cur_x, &lw.attn_hc.norm, &lw.attn_hc.down,
                                    &d_hc_h_, &d_attn_inj_, &d_hc_rstd_, &wi_delta, &M, &eps};
                    kernels_.launch("hc_premix_wmma_c1_u0", 21, m_tiles64, 1, args);
                } else if (l == 0) {
                    void* args[] = {&d_batch_toks, &d_embed_tokens_, &cur_x, &lw.attn_hc.norm, &lw.attn_hc.down,
                                    &d_hc_h_, &d_attn_inj_, &d_hc_rstd_, &wi_delta, &M, &eps};
                    kernels_.launch("embed_hc_premix_l0", 81, M, 1, args);
                } else {
                    void* args[] = {&cur_x, &cur_x, &cur_x, &cur_x, &lw.attn_hc.norm, &lw.attn_hc.down,
                                    &d_hc_h_, &d_attn_inj_, &d_hc_rstd_, &wi_delta, &M, &eps};
                    kernels_.launch("hc_premix_c1_u0", 81, M, 1, args);
                }
            } else {
                void* args[] = {&cur_x, &prev_out, &prev_inj, &alt_x, &lw.attn_hc.norm, &lw.attn_hc.down,
                                &d_hc_h_, &d_attn_inj_, &d_hc_rstd_, &wi_delta, &M, &eps};
                if (use_wmma_hc) {
                    kernels_.launch("hc_premix_wmma_c1_u1", 21, m_tiles64, 1, args);
                } else {
                    kernels_.launch("hc_premix_c1_u1", 81, M, 1, args);
                }
                std::swap(cur_x, alt_x);
            }
            void* m_args[] = {&cur_x, &lw.attn_hc.norm, &d_hc_rstd_, &d_hc_h_, &lw.attn_hc.up, &d_mixed_, &M};
            if (use_wmma_hc) {
                kernels_.launch("hc_mix_wmma", 40, m_tiles64, 1, m_args);
            } else {
                kernels_.launch("hc_mix", 80, M, 1, m_args);
            }
        }

        const bool is_verify_mode = (verify_next_toks_out != nullptr);
        const bool is_small_batch = (is_verify_mode || M <= 16);
        if (!lw.is_qsa) {
            void* gdn_proj_buf = is_verify_mode ? d_gdn_verify_qkv_[l] : d_proj_scratch_;
            strata::kernels::launch_nvfp4_linear_auto(
                kernels_, "bf16_linear_16480_2560", "bf16_gemv_16480_2560", "bf16_gemv_m4_16480_2560",
                d_mixed_, lw.gdn.in_proj_fused, gdn_proj_buf, 16480, M);
            void* qkv_buf = gdn_proj_buf;
            void* z_buf = static_cast<uint16_t*>(gdn_proj_buf) + 10240;
            void* a_buf = static_cast<uint16_t*>(gdn_proj_buf) + 16384;
            void* b_buf = static_cast<uint16_t*>(gdn_proj_buf) + 16432;
            void* st_in = d_gdn_state_[l];
            void* st_out = is_verify_mode ? d_gdn_state_ckpt_[l] : d_gdn_state_[l];
            strata::kernels::launch_nvfp4_gated_deltanet_fused_strided(
                kernels_, qkv_buf, z_buf, a_buf, b_buf, lw.gdn.A_log, lw.gdn.dt_bias,
                lw.gdn.conv1d, d_gdn_conv_in_[l], d_gdn_conv_out_[l],
                static_cast<const float*>(st_in), static_cast<float*>(st_out),
                lw.gdn.norm, d_gdn_out_, M, 16480, eps);
            std::swap(d_gdn_conv_in_[l], d_gdn_conv_out_[l]);
            strata::kernels::launch_nvfp4_linear_auto(
                kernels_, "bf16_linear_2560_6144", "bf16_gemv_2560_6144", "bf16_gemv_m4_2560_6144",
                d_gdn_out_, lw.gdn.out_proj, d_attn_out_, 2560, M);
        } else {
            const int qsa_idx = l / 4;
            strata::kernels::launch_nvfp4_linear_auto(
                kernels_, "bf16_linear_13952_2560", "bf16_gemv_13952_2560", "bf16_gemv_m4_13952_2560",
                d_mixed_, lw.qsa.in_proj_fused, d_proj_scratch_, 13952, M);
            void* kc = d_k_cache_[qsa_idx];
            void* vc = d_v_cache_[qsa_idx];
            void* rkc = d_raw_k_cache_[qsa_idx];
            void* bkc = d_block_k_cache_[qsa_idx];
            int64_t kc_delta = kc_host_delta_[qsa_idx];
            int64_t vc_delta = vc_host_delta_[qsa_idx];
            int64_t bk_delta = bk_host_delta_elems_[qsa_idx];
            int32_t b_start = pos0 / 4;
            int32_t b_end = (pos0 + M) / 4;
            int32_t b_cnt = b_end - b_start;
            if (M <= 1020) {
                strata::kernels::launch_nvfp4_qsa_prep_and_cache_write_batch(
                    kernels_, d_proj_scratch_, lw.qsa.q_norm, lw.qsa.k_norm, d_rope_cos_f32_, d_rope_sin_f32_,
                    d_qsa_qg_, d_qsa_q_, d_qsa_idx_qk_b_, kc, vc, rkc, kc_delta, vc_delta,
                    pos0, M, vram_prefix_, eps, nullptr);
                strata::kernels::launch_nvfp4_qsa_block_keys_from_batch(
                    kernels_, rkc, d_qsa_idx_qk_b_, lw.qsa.idx_k_ln, d_rope_cos_bf16_, d_rope_sin_bf16_,
                    bkc, bk_delta, pos0, M, b_start, b_cnt, vram_blocks_, eps, nullptr);
            } else {
                {
                    void* as[] = {&d_proj_scratch_, &d_qsa_qg_, &d_qsa_k_b_, &d_qsa_v_b_, &d_qsa_idx_qk_b_};
                    kernels_.launch("qsa_proj_split", 55, M, 1, as);
                }
                {
                    void* args[] = {&d_qsa_qg_, &d_qsa_q_};
                    kernels_.launch("qsa_q_extract", 24, M, 1, args);
                }
                {
                    void* cos_p = static_cast<uint8_t*>(d_rope_cos_f32_) + static_cast<size_t>(pos0) * 32 * sizeof(float);
                    void* sin_p = static_cast<uint8_t*>(d_rope_sin_f32_) + static_cast<size_t>(pos0) * 32 * sizeof(float);
                    int32_t r24 = M * 24, h24 = 24;
                    void* aq[] = {&d_qsa_q_, &lw.qsa.q_norm, &cos_p, &sin_p, &d_qsa_q_, &r24, &h24, &eps};
                    kernels_.launch("qk_norm_partial_rope", M * 3, 1, 1, aq);
                    int32_t r2 = M * 2, h2 = 2;
                    void* ak[] = {&d_qsa_k_b_, &lw.qsa.k_norm, &cos_p, &sin_p, &d_qsa_k_b_, &r2, &h2, &eps};
                    kernels_.launch("qk_norm_partial_rope", (M * 2 + 7) / 8, 1, 1, ak);
                }
                strata::kernels::launch_nvfp4_qsa_block_keys_from_batch(
                    kernels_, rkc, d_qsa_idx_qk_b_, lw.qsa.idx_k_ln, d_rope_cos_bf16_, d_rope_sin_bf16_,
                    bkc, bk_delta, pos0, M, b_start, b_cnt, vram_blocks_, eps, nullptr);
                {
                    void* args[] = {&d_qsa_k_b_, &d_qsa_v_b_, &d_qsa_idx_qk_b_, &kc, &vc, &rkc,
                                    &kc_delta, &vc_delta, &pos0, &vram_prefix_};
                    kernels_.launch("qsa_cache_write_nvfp4", 1, M, 1, args);
                }
                if (M > 1024) {
                    strata::kernels::launch_nvfp4_qsa_refresh_raw_k_tail(
                        kernels_, rkc, d_qsa_idx_qk_b_, pos0, M, nullptr);
                }
            }
            if (pos0 + M > 2048) {
                int32_t t_first = std::max(0, 2048 - pos0);
                int32_t max_nb = (pos0 + M) / 4;
                int32_t score_grid = std::max(1, (max_nb + 255) / 256);
                int32_t score_stride = score_grid * 256;
                int32_t max_m_per_launch = std::max(1, static_cast<int32_t>(4194304 / score_stride));
                for (int32_t t0 = t_first; t0 < M; t0 += max_m_per_launch) {
                    int32_t m_cur = std::min(M - t0, max_m_per_launch);
                    int32_t pos_cur = pos0 + t0;
                    void* qk_cur = static_cast<uint16_t*>(d_qsa_idx_qk_b_) + static_cast<size_t>(t0) * 640;
                    void* sel_cur = static_cast<int32_t*>(d_qsa_sel_ids_) + static_cast<size_t>(t0) * 512;
                    if ((pos_cur + m_cur) / 4 > 512) {
                        strata::kernels::launch_nvfp4_qsa_score_blocks_batched(
                            kernels_, qk_cur, lw.qsa.idx_q_ln, d_rope_cos_bf16_, d_rope_sin_bf16_,
                            bkc, static_cast<float*>(d_qsa_scores_), bk_delta, pos_cur, m_cur,
                            vram_blocks_, score_stride, 16, 64, eps, nullptr);
                    }
                    void* atk[] = {&d_qsa_scores_, &sel_cur, &pos_cur, &score_stride};
                    kernels_.launch("qsa_topk_blocks", m_cur, 1, 1, atk);
                }
            }
            {
                void* ap[] = {&d_qsa_q_, &kc, &vc, &d_qsa_qg_, &d_qsa_sel_ids_, &d_qsa_attn_raw_,
                              &kc_delta, &vc_delta, &pos0, &M, &vram_prefix_};
                kernels_.launch("qsa_prefill_causal_nvfp4", 24, M, 1, ap);
            }
            strata::kernels::launch_nvfp4_linear_auto(
                kernels_, "bf16_linear_2560_6144", "bf16_gemv_2560_6144", "bf16_gemv_m4_2560_6144",
                d_qsa_attn_raw_, lw.qsa.o_proj, d_attn_out_, 2560, M);
        }

        {
            int64_t wi_delta = lw.mlp_hc.wi_delta;
            void* args[] = {&cur_x, &d_attn_out_, &d_attn_inj_, &alt_x, &lw.mlp_hc.norm, &lw.mlp_hc.down,
                            &d_hc_h_, &d_mlp_inj_, &d_hc_rstd_, &wi_delta, &M, &eps};
            if (use_wmma_hc) {
                kernels_.launch("hc_premix_wmma_c1_u1", 21, m_tiles64, 1, args);
            } else {
                kernels_.launch("hc_premix_c1_u1", 81, M, 1, args);
            }
            std::swap(cur_x, alt_x);
            void* m_args[] = {&cur_x, &lw.mlp_hc.norm, &d_hc_rstd_, &d_hc_h_, &lw.mlp_hc.up, &d_mixed_, &M};
            if (use_wmma_hc) {
                kernels_.launch("hc_mix_wmma", 40, m_tiles64, 1, m_args);
            } else {
                kernels_.launch("hc_mix", 80, M, 1, m_args);
            }
        }

        {
            static const int32_t verify_gpu_cold_override = []() {
                if (const char* e = std::getenv("STRATA_VERIFY_GPU_COLD")) {
                    return std::clamp(std::atoi(e), -1, 40);
                }
                return 0;
            }();
            static const int32_t verify_decode_moe_max_m = []() {
                if (const char* e = std::getenv("STRATA_VERIFY_DECODE_MOE")) {
                    return std::clamp(std::atoi(e), 0, 16);
                }
                return 0;
            }();
            const int32_t cold_cap = (verify_gpu_cold_override >= 0) ? verify_gpu_cold_override : (M * 2);
            const bool is_verify_batch = is_verify_mode && (cold_cap > 0);
            const bool use_decode_moe = (M <= verify_decode_moe_max_m);
            strata::kernels::launch_nvfp4_linear_auto(
                kernels_, "bf16_linear_1856_2560", "bf16_gemv_1856_2560", "bf16_gemv_m4_1856_2560",
                d_mixed_, lw.moe.in_proj_fused, d_proj_scratch_, 1856, M);

            void* r_logits = static_cast<uint16_t*>(d_proj_scratch_) + 1280;
            void* sh_sc = static_cast<uint16_t*>(d_proj_scratch_) + 1792;
            int32_t stride1856 = 1856;
            void* a_tk[] = {&r_logits, &d_cur_ids, &d_moe_wts_, &M, &stride1856};
            kernels_.launch("moe_topk", (M + 3) / 4, 1, 1, a_tk);

            int32_t pairs = M * 10;
            if (is_verify_batch) {
                strata::kernels::launch_nvfp4_moe_cap_gpu_cold(
                    kernels_,
                    static_cast<int32_t*>(d_cur_ids),
                    static_cast<int32_t*>(d_moe_gpu_align_ids_),
                    static_cast<const int64_t*>(lw.moe.d_ptrs),
                    static_cast<const int64_t*>(lw.moe.d_ptrs_prefill),
                    pairs, cold_cap, nullptr);
            }

            HIP_CHECK(hipMemcpyAsync(h_route_bell_, d_route_bell_,
                                     static_cast<size_t>(M) * (2560 * sizeof(uint16_t) + 10 * sizeof(int32_t)),
                                     hipMemcpyDeviceToHost, nullptr));
            HIP_CHECK(hipEventRecord(ev_route_, nullptr));

            const bool use_gpu_cold_batch = !is_small_batch || is_verify_batch;
            const bool has_gpu_experts = use_gpu_cold_batch || (lw.moe.vram_experts > 0);
            if (has_gpu_experts && is_small_batch && !is_verify_batch && !use_decode_moe) {
                // Filter out non-VRAM experts into d_moe_gpu_align_ids_ after ev_route_ has fired
                // so moe_small_align only creates 16-row WMMA tiles for VRAM-resident experts.
                strata::kernels::launch_nvfp4_moe_cap_gpu_cold(
                    kernels_,
                    static_cast<int32_t*>(d_cur_ids),
                    static_cast<int32_t*>(d_moe_gpu_align_ids_),
                    static_cast<const int64_t*>(lw.moe.d_ptrs),
                    static_cast<const int64_t*>(lw.moe.d_ptrs_prefill),
                    pairs, /*max_gpu_cold=*/0, nullptr);
            }

            strata::kernels::launch_nvfp4_silu_mul_640_strided(
                kernels_, d_proj_scratch_, d_sh_inter_, M, 1856, nullptr);
            strata::kernels::launch_nvfp4_linear_auto(
                kernels_, "bf16_linear_2560_640", "bf16_gemv_2560_640", "bf16_gemv_m4_2560_640",
                d_sh_inter_, lw.moe.sh_down, d_sh_out_, 2560, M);

            if (has_gpu_experts) {
                void* ptrs = use_gpu_cold_batch ? lw.moe.d_ptrs : lw.moe.d_ptrs_prefill;
                void* s2 = lw.moe.d_s2;
                if (use_decode_moe) {
                    void* a_mgu[] = {&d_mixed_, &d_cur_ids, &ptrs, &s2, &d_moe_inter_};
                    kernels_.launch("moe_decode_gate_up", 20, pairs, 1, a_mgu);
                    void* a_mdn[] = {&d_moe_inter_, &d_cur_ids, &ptrs, &s2, &d_moe_y_};
                    kernels_.launch("moe_decode_down", 20, pairs, 1, a_mdn);
                } else {
                    const int32_t max_moe_tiles = (kMaxBatch * 10 + 15) / 16 + 512;
                    int32_t max_tiles = is_small_batch
                                            ? std::min<int32_t>({pairs, max_moe_tiles,
                                                                 use_gpu_cold_batch ? pairs : lw.moe.vram_experts})
                                            : ((pairs + 15) / 16 + std::min(512, pairs));
                    if (is_small_batch) {
                        strata::kernels::launch_nvfp4_moe_small_align(
                            kernels_,
                            static_cast<const int32_t*>(d_moe_gpu_align_ids_),
                            static_cast<int32_t*>(d_moe_sorted_),
                            static_cast<int32_t*>(d_moe_tile_e_),
                            pairs, max_tiles, nullptr);
                    } else {
                        void* a_al[] = {&d_cur_ids, &d_moe_sorted_, &d_moe_tile_e_, &pairs, &max_tiles};
                        kernels_.launch("moe_align", 1, 1, 1, a_al);
                    }

                    int32_t topk10 = 10;
                    void* a_gu[] = {&d_mixed_, &d_moe_sorted_, &d_moe_tile_e_, &ptrs, &s2, &d_moe_inter_, &topk10};
                    kernels_.launch("moe_gate_up", 5, max_tiles, 1, a_gu);
                    void* a_dn[] = {&d_moe_inter_, &d_moe_sorted_, &d_moe_tile_e_, &ptrs, &s2, &d_moe_y_};
                    kernels_.launch("moe_down", 10, max_tiles, 1, a_dn);
                }
            }

            if (l == 0) {
                fetch_ple_rows_cached(tokens, prev0s, prev1s, M, h_ple_rows);
                HIP_CHECK(hipMemcpyAsync(d_ple_rows_u8_, h_ple_rows, static_cast<size_t>(M) * 2560,
                                         hipMemcpyHostToDevice, nullptr));
            }

            (void) run_cpu_moe_layer(l, M, /*is_prefill=*/true, use_gpu_cold_batch);

            void* a_cb[] = {&d_moe_y_, &d_moe_wts_, &d_sh_out_, &sh_sc, &d_mlp_out_, &stride1856};
            kernels_.launch("moe_combine", M, 1, 1, a_cb);
        }

        has_prev_update = true;
        prev_out = d_mlp_out_;
        prev_inj = d_mlp_inj_;
    }

    if (!compute_logits) {
        if (mtp_.available) {
            void* args[] = {&cur_x, &prev_out, &prev_inj, &alt_x};
            kernels_.launch("hc_residual_add", 40, M, 1, args);
            std::swap(cur_x, alt_x);
            strata::kernels::launch_nvfp4_mtp_ring_write(kernels_, cur_x, d_mtp_r_ring_, pos0, kMtpRingMask, M, nullptr);
        }
        adapt_vram_experts(h_all_routed_ids_.data(), M, /*min_hits=*/2, /*max_swaps=*/12);
        return -1;
    }

    if (verify_next_toks_out) {
        int64_t zero_delta = 0;
        void* args[] = {&cur_x, &prev_out, &prev_inj, &alt_x, &final_hc_.norm, &final_hc_.down,
                        &d_hc_h_, &d_hc_h_, &d_hc_rstd_, &zero_delta, &M, &eps};
        kernels_.launch("hc_premix_c0_u1", 80, M, 1, args);
        std::swap(cur_x, alt_x);
        if (mtp_.available) {
            strata::kernels::launch_nvfp4_mtp_ring_write(kernels_, cur_x, d_mtp_r_ring_, pos0, kMtpRingMask, M, nullptr);
        }
        void* m_args[] = {&cur_x, &final_hc_.norm, &d_hc_rstd_, &d_hc_h_, &final_hc_.up, &d_attn_out_, &M};
        kernels_.launch("hc_mix", 80, M, 1, m_args);

        static constexpr size_t kVocab = 248320;
        const bool use_ver_sampling = (sp && !sp->is_greedy() && rng);
        const size_t base_hist_len = (history && history->size() >= static_cast<size_t>(M))
                                         ? (history->size() - static_cast<size_t>(M))
                                         : 0;
        if (!use_ver_sampling && M <= 16) {
            strata::kernels::launch_nvfp4_lm_head_auto(kernels_, d_attn_out_, d_lm_head_, d_logits_bf16_, M);
            strata::kernels::launch_nvfp4_argmax_prob_margin_248320(
                kernels_, d_logits_bf16_, static_cast<int32_t*>(d_next_tok_),
                static_cast<strata::kernels::Nvfp4ArgmaxSummary*>(d_argmax_summary_), M, nullptr);
        } else {
            const int32_t chunk_step = 16;
            for (int32_t t0 = 0; t0 < M; t0 += chunk_step) {
                int32_t m_chunk = std::min<int32_t>(chunk_step, M - t0);
                void* h_in = static_cast<uint16_t*>(d_attn_out_) + static_cast<size_t>(t0) * 2560;
                strata::kernels::launch_nvfp4_lm_head_auto(kernels_, h_in, d_lm_head_, d_logits_bf16_, m_chunk);
                if (!use_ver_sampling) {
                    strata::kernels::launch_nvfp4_argmax_prob_margin_248320(
                        kernels_, d_logits_bf16_, static_cast<int32_t*>(d_next_tok_) + t0,
                        static_cast<strata::kernels::Nvfp4ArgmaxSummary*>(d_argmax_summary_) + t0,
                        m_chunk, nullptr);
                } else {
                    uint16_t* h_bf_row = static_cast<uint16_t*>(h_cpu_upload_);
                    for (int32_t tc = 0; tc < m_chunk; ++tc) {
                        const int32_t t = t0 + tc;
                        HIP_CHECK(hipMemcpy(
                            h_bf_row,
                            static_cast<const uint16_t*>(d_logits_bf16_) + static_cast<size_t>(tc) * kVocab,
                            kVocab * sizeof(uint16_t), hipMemcpyDeviceToHost));
                        int32_t d_tok = (verify_draft_toks && t + 1 < M) ? verify_draft_toks[t] : -1;
                        float prob_t = 1.0f, margin_t = 0.0f;
                        int32_t sampled = sample_row_from_host_bf16_logits(
                            h_bf_row, *sp, history,
                            base_hist_len + static_cast<size_t>(t + 1), *rng, d_tok, &prob_t, &margin_t);
                        verify_next_toks_out[t] = sampled;
                        verify_summaries_[static_cast<size_t>(t)].tok_id = sampled;
                        verify_summaries_[static_cast<size_t>(t)].prob = prob_t;
                        verify_summaries_[static_cast<size_t>(t)].margin = margin_t;
                        if (d_tok >= 0 && sampled != d_tok) {
                            for (int32_t rem = t + 1; rem < M; ++rem) {
                                verify_next_toks_out[rem] = sampled;
                                verify_summaries_[static_cast<size_t>(rem)] = verify_summaries_[static_cast<size_t>(t)];
                            }
                            return verify_next_toks_out[M - 1];
                        }
                    }
                }
            }
        }

        if (!use_ver_sampling) {
            HIP_CHECK(hipMemcpy(verify_summaries_.data(), d_argmax_summary_,
                                static_cast<size_t>(M) * sizeof(strata::kernels::Nvfp4ArgmaxSummary),
                                hipMemcpyDeviceToHost));
            for (int32_t t = 0; t < M; ++t) {
                verify_next_toks_out[t] = verify_summaries_[static_cast<size_t>(t)].tok_id;
            }
        }
        return verify_next_toks_out[M - 1];
    }

    {
        if (mtp_.available && M > 1) {
            int32_t m_prev = M - 1;
            void* args_r[] = {&cur_x, &prev_out, &prev_inj, &alt_x};
            kernels_.launch("hc_residual_add", 40, m_prev, 1, args_r);
        }
        void* last_x = static_cast<uint16_t*>(cur_x) + static_cast<size_t>(M - 1) * 10240;
        void* last_alt = static_cast<uint16_t*>(alt_x) + static_cast<size_t>(M - 1) * 10240;
        void* last_out = static_cast<uint16_t*>(prev_out) + static_cast<size_t>(M - 1) * 2560;
        void* last_inj = static_cast<uint16_t*>(prev_inj) + static_cast<size_t>(M - 1) * 4;
        int64_t zero_delta = 0;
        void* args[] = {&last_x, &last_out, &last_inj, &last_alt, &final_hc_.norm, &final_hc_.down,
                        &d_hc_h_, &d_hc_h_, &d_hc_rstd_, &zero_delta, &eps};
        kernels_.launch_default("hc_premix_decode_c0_u1", args);
        if (mtp_.available) {
            strata::kernels::launch_nvfp4_mtp_ring_write(kernels_, alt_x, d_mtp_r_ring_, pos0, kMtpRingMask, M, nullptr);
        }
        void* m_args[] = {&last_alt, &final_hc_.norm, &d_hc_rstd_, &d_hc_h_, &final_hc_.up, &d_final_hidden_, &one};
        kernels_.launch_default("hc_mix_decode", m_args);
    }

    const bool use_sampling = (sp && !sp->is_greedy() && rng);
    {
        void* a_lm[] = {&d_final_hidden_, &d_lm_head_, &d_logits_bf16_};
        kernels_.launch_default("bf16_gemv_248320_2560", a_lm);
        if (!use_sampling) {
            strata::kernels::launch_nvfp4_argmax_prob_margin_248320(
                kernels_, d_logits_bf16_, static_cast<int32_t*>(d_next_tok_),
                static_cast<strata::kernels::Nvfp4ArgmaxSummary*>(d_argmax_summary_), 1, nullptr);
        }
    }

    int32_t next_id = -1;
    if (use_sampling) {
        next_id = sample_from_device_logits(*sp, history, *rng);
    } else {
        strata::kernels::Nvfp4ArgmaxSummary s{};
        HIP_CHECK(hipMemcpy(&s, d_argmax_summary_, sizeof(s), hipMemcpyDeviceToHost));
        next_id = s.tok_id;
        last_target_prob_ = s.prob;
        last_target_margin_ = s.margin;
    }
    adapt_vram_experts(h_all_routed_ids_.data(), M, /*min_hits=*/2, /*max_swaps=*/12);

    if (dump_logits_out) {
        std::vector<uint16_t> bf(248320);
        HIP_CHECK(hipMemcpy(bf.data(), d_logits_bf16_, 248320 * sizeof(uint16_t), hipMemcpyDeviceToHost));
        dump_logits_out->resize(248320);
        for (size_t i = 0; i < 248320; ++i) {
            (*dump_logits_out)[i] = bf16_bits_to_f32(bf[i]);
        }
    }
    return next_id;
}

void Qwen4ExpGfx1201Engine::fetch_ple_rows_cached(const int32_t* tokens, const int32_t* prev0s, const int32_t* prev1s,
                                                  int32_t M, uint8_t* h_ple_dst) {
    uint64_t* miss_offs = ple_miss_offs_.data();
    uint8_t** miss_cache_dsts = ple_miss_cache_dsts_.data();
    uint8_t** miss_user_dsts = ple_miss_user_dsts_.data();
    int n_miss = 0;

    auto set_index_for = [](uint64_t off) -> size_t {
        return static_cast<size_t>(((off / 160ULL) * 0x9E3779B97F4A7C15ULL) >> 48) & (kPleCacheSets - 1);
    };

    if (h_ple_dst != nullptr) {
        for (int32_t t = 0; t < M; ++t) {
            uint8_t* dst_2560 = h_ple_dst + static_cast<size_t>(t) * 2560;
            uint32_t rows16[16]{};
            const int32_t prev_pair[2] = {prev0s[t], prev1s[t]};
            strata::kernels::ngram_rows(&tokens[t], prev_pair, 1, ple_consts_, rows16);
            for (int h = 0; h < 16; ++h) {
                uint32_t r = rows16[h];
                uint32_t shard = r / 2500012u;
                uint32_t row_in_shard = r % 2500012u;
                uint64_t off = ple_.shard_offsets[shard] + static_cast<uint64_t>(row_in_shard) * 160ULL;
                uint8_t* user_dst = dst_2560 + h * 160;
                if (!ple_cache_.empty()) {
                    const PleCacheEntry* set_base = &ple_cache_[set_index_for(off) * kPleCacheWays];
                    bool hit = false;
                    for (size_t w = 0; w < kPleCacheWays; ++w) {
                        if (set_base[w].off == off) {
                            std::memcpy(user_dst, set_base[w].data, 160);
                            hit = true;
                            break;
                        }
                    }
                    if (hit) continue;
                }
                miss_offs[n_miss] = off;
                miss_user_dsts[n_miss] = user_dst;
                n_miss++;
            }
        }
        if (n_miss > 0) {
            cpu_pool_.run_ple_reads(ple_.ple_fd, miss_offs, miss_user_dsts, n_miss);
            if (!ple_cache_.empty()) {
                for (int i = 0; i < n_miss; ++i) {
                    uint64_t off = miss_offs[i];
                    PleCacheEntry* set_base = &ple_cache_[set_index_for(off) * kPleCacheWays];
                    size_t target_w = kPleCacheWays;
                    for (size_t w = 0; w < kPleCacheWays; ++w) {
                        if (set_base[w].off == off) {
                            target_w = w;
                            break;
                        }
                        if (target_w == kPleCacheWays && set_base[w].off == ~0ULL) {
                            target_w = w;
                        }
                    }
                    if (target_w == kPleCacheWays) {
                        target_w = (ple_cache_clock_++) & (kPleCacheWays - 1);
                    }
                    PleCacheEntry& ce = set_base[target_w];
                    ce.off = off;
                    std::memcpy(ce.data, miss_user_dsts[i], 160);
                }
            }
        }
        return;
    }

    for (int32_t t = 0; t < M; ++t) {
        uint32_t rows16[16]{};
        const int32_t prev_pair[2] = {prev0s[t], prev1s[t]};
        strata::kernels::ngram_rows(&tokens[t], prev_pair, 1, ple_consts_, rows16);
        for (int h = 0; h < 16; ++h) {
            uint32_t r = rows16[h];
            uint32_t shard = r / 2500012u;
            uint32_t row_in_shard = r % 2500012u;
            uint64_t off = ple_.shard_offsets[shard] + static_cast<uint64_t>(row_in_shard) * 160ULL;
            if (!ple_cache_.empty()) {
                PleCacheEntry* set_base = &ple_cache_[set_index_for(off) * kPleCacheWays];
                bool hit = false;
                size_t empty_w = kPleCacheWays;
                for (size_t w = 0; w < kPleCacheWays; ++w) {
                    if (set_base[w].off == off) {
                        hit = true;
                        break;
                    }
                    if (empty_w == kPleCacheWays && set_base[w].off == ~0ULL) {
                        empty_w = w;
                    }
                }
                if (hit) continue;
                size_t target_w = (empty_w < kPleCacheWays) ? empty_w : ((ple_cache_clock_++) & (kPleCacheWays - 1));
                PleCacheEntry& ce = set_base[target_w];
                ce.off = off;
                miss_offs[n_miss] = off;
                miss_cache_dsts[n_miss] = ce.data;
                n_miss++;
            }
        }
    }

    if (n_miss > 0) {
        cpu_pool_.run_ple_reads(ple_.ple_fd, miss_offs, miss_cache_dsts, n_miss);
    }
}

int32_t Qwen4ExpGfx1201Engine::sample_row_from_host_bf16_logits(const uint16_t* h_bf,
                                                                const Nvfp4SamplingParams& sp,
                                                                const std::vector<int32_t>* history,
                                                                size_t hist_len,
                                                                std::mt19937_64& rng,
                                                                int32_t draft_tok,
                                                                float* top1_prob_out,
                                                                float* margin_out) {
    static constexpr int32_t kVocab = 248320;

    thread_local std::vector<float> logits(kVocab);
    float* __restrict logits_ptr = logits.data();
    for (int32_t i = 0; i < kVocab; ++i) {
        logits_ptr[i] = bf16_bits_to_f32(h_bf[i]);
    }

    if (history && !history->empty() && hist_len > 0 &&
        (sp.presence_penalty != 0.0f || sp.frequency_penalty != 0.0f || sp.repeat_penalty != 1.0f)) {
        std::unordered_map<int32_t, int32_t> counts;
        const size_t n_hist = std::min(history->size(), hist_len);
        const size_t start = (n_hist > 1024) ? (n_hist - 1024) : 0;
        for (size_t i = start; i < n_hist; ++i) {
            int32_t tid = (*history)[i];
            if (tid >= 0 && tid < kVocab) counts[tid]++;
        }
        for (const auto& kv : counts) {
            int32_t tid = kv.first;
            int32_t cnt = kv.second;
            float v = logits_ptr[tid];
            if (sp.repeat_penalty != 1.0f && sp.repeat_penalty > 0.0f) {
                v = (v < 0.0f) ? (v * sp.repeat_penalty) : (v / sp.repeat_penalty);
            }
            v -= sp.presence_penalty;
            v -= sp.frequency_penalty * static_cast<float>(cnt);
            logits_ptr[tid] = v;
        }
    }

    int32_t K = (sp.top_k > 0) ? std::min<int32_t>(sp.top_k, kVocab) : 256;
    struct Cand {
        float val;
        int32_t id;
    };
    float max_l = -1e30f;
    float max2_l = -1e30f;
    int32_t best_id = 0;
    for (int32_t i = 0; i < kVocab; ++i) {
        float v = logits_ptr[i];
        if (v > max_l) {
            max2_l = max_l;
            max_l = v;
            best_id = i;
        } else if (v > max2_l) {
            max2_l = v;
        }
    }
    if (margin_out) *margin_out = max_l - max2_l;
    if (K <= 1 || sp.temperature <= 0.0f) {
        if (top1_prob_out) *top1_prob_out = 1.0f;
        return best_id;
    }

    const float cutoff = max_l - 20.0f * std::max(1.0f, sp.temperature);
    std::vector<Cand> cands;
    cands.reserve(2048);
    for (int32_t i = 0; i < kVocab; ++i) {
        if (logits_ptr[i] >= cutoff) {
            cands.push_back(Cand{logits_ptr[i], i});
        }
    }
    if (static_cast<int32_t>(cands.size()) < K && sp.top_k > 0) {
        cands.resize(kVocab);
        for (int32_t i = 0; i < kVocab; ++i) cands[i] = Cand{logits[i], i};
    }
    if (cands.empty()) {
        if (top1_prob_out) *top1_prob_out = 1.0f;
        return best_id;
    }

    auto cmp = [](const Cand& a, const Cand& b) {
        return (a.val != b.val) ? (a.val > b.val) : (a.id < b.id);
    };
    if (static_cast<int32_t>(cands.size()) > K) {
        std::partial_sort(cands.begin(), cands.begin() + K, cands.end(), cmp);
        cands.resize(K);
    } else {
        std::sort(cands.begin(), cands.end(), cmp);
    }

    const float inv_temp = 1.0f / std::max(1e-5f, sp.temperature);
    const float top_val = cands[0].val;
    double sum_p = 0.0;
    std::vector<double> probs(cands.size());
    for (size_t i = 0; i < cands.size(); ++i) {
        double p = std::exp(static_cast<double>((cands[i].val - top_val) * inv_temp));
        probs[i] = p;
        sum_p += p;
    }
    if (sum_p <= 0.0) {
        if (top1_prob_out) *top1_prob_out = 1.0f;
        return cands[0].id;
    }

    size_t keep = cands.size();
    if (sp.top_p > 0.0f && sp.top_p < 1.0f) {
        double cum = 0.0;
        const double thresh = static_cast<double>(sp.top_p) * sum_p;
        for (size_t i = 0; i < cands.size(); ++i) {
            cum += probs[i];
            if (cum >= thresh) {
                keep = i + 1;
                sum_p = cum;
                break;
            }
        }
    }
    if (top1_prob_out) {
        *top1_prob_out = static_cast<float>(probs[0] / std::max(1e-30, sum_p));
    }

    // Exact lossless point-mass speculative rejection sampling (DSpark §3.3.1 Eq. 8):
    // For a deterministic draft token \hat{y} (q(\hat{y}) = 1, q(v != \hat{y}) = 0):
    //   1. Accept \hat{y} with probability p_target(\hat{y}) = probs[idx_draft] / sum_p.
    //   2. On rejection, sample from residual distribution p'(v) \propto max(0, p_target(v) - q(v)),
    //      which for point-mass q(\hat{y}) = 1 equals p_target(v) restricted to v != \hat{y}.
    if (draft_tok >= 0) {
        ssize_t draft_idx = -1;
        for (size_t i = 0; i < keep; ++i) {
            if (cands[i].id == draft_tok) {
                draft_idx = static_cast<ssize_t>(i);
                break;
            }
        }
        std::uniform_real_distribution<double> u01(0.0, 1.0);
        const double r_acc = u01(rng);
        const double p_draft = (draft_idx >= 0) ? (probs[draft_idx] / sum_p) : 0.0;
        if (r_acc < p_draft) {
            return draft_tok;
        }
        // Rejected: zero out draft_tok mass and sample from the residual distribution
        double res_sum = sum_p - ((draft_idx >= 0) ? probs[draft_idx] : 0.0);
        if (res_sum <= 1e-30) {
            return cands[0].id;
        }
        std::uniform_real_distribution<double> dist_res(0.0, res_sum);
        double r_res = dist_res(rng);
        double acc = 0.0;
        for (size_t i = 0; i < keep; ++i) {
            if (static_cast<ssize_t>(i) == draft_idx) continue;
            acc += probs[i];
            if (r_res <= acc) return cands[i].id;
        }
        for (size_t i = keep; i > 0; --i) {
            if (static_cast<ssize_t>(i - 1) != draft_idx) return cands[i - 1].id;
        }
        return cands[0].id;
    }

    std::uniform_real_distribution<double> dist(0.0, sum_p);
    double r = dist(rng);
    double acc = 0.0;
    for (size_t i = 0; i < keep; ++i) {
        acc += probs[i];
        if (r <= acc) return cands[i].id;
    }
    return cands[keep - 1].id;
}

int32_t Qwen4ExpGfx1201Engine::sample_from_device_logits(const Nvfp4SamplingParams& sp,
                                                         const std::vector<int32_t>* history,
                                                         std::mt19937_64& rng) {
    static constexpr int32_t kVocab = 248320;
    uint16_t* h_bf = static_cast<uint16_t*>(h_cpu_upload_);
    HIP_CHECK(hipMemcpy(h_bf, d_logits_bf16_, kVocab * sizeof(uint16_t), hipMemcpyDeviceToHost));
    const size_t hist_len = history ? history->size() : 0;
    return sample_row_from_host_bf16_logits(h_bf, sp, history, hist_len, rng, -1,
                                            &last_target_prob_, &last_target_margin_);
}

void Qwen4ExpGfx1201Engine::save_recurrent_ckpt() {
    HIP_CHECK(hipMemcpyAsync(d_recurrent_ckpt_, d_recurrent_arena_, conv_ckpt_bytes_,
                             hipMemcpyDeviceToDevice, nullptr));
    for (int l = 0; l < 48; ++l) {
        ckpt_gdn_conv_in_[l] = d_gdn_conv_in_[l];
        ckpt_gdn_conv_out_[l] = d_gdn_conv_out_[l];
    }
}

void Qwen4ExpGfx1201Engine::restore_recurrent_ckpt() {
    // Copy only the 4.39 MiB PLE + GDN conv1d state back; GDN 108 MiB state is preserved out-of-place.
    HIP_CHECK(hipMemcpyAsync(d_recurrent_arena_, d_recurrent_ckpt_, conv_ckpt_bytes_,
                             hipMemcpyDeviceToDevice, nullptr));
    for (int l = 0; l < 48; ++l) {
        if (l % 4 != 3) {
            d_gdn_conv_in_[l] = ckpt_gdn_conv_in_[l];
            d_gdn_conv_out_[l] = ckpt_gdn_conv_out_[l];
        }
    }
}

void Qwen4ExpGfx1201Engine::finish_verify_step(int32_t M, int32_t n_adv) {
    int32_t acc_idx = std::clamp(n_adv - 1, 0, std::max(0, M - 1));
    last_target_prob_ = verify_summaries_[static_cast<size_t>(acc_idx)].prob;
    last_target_margin_ = verify_summaries_[static_cast<size_t>(acc_idx)].margin;
    if (n_adv == M) {
        for (int l = 0; l < 48; ++l) {
            if (!layers_[l].is_qsa) {
                std::swap(d_gdn_state_[l], d_gdn_state_ckpt_[l]);
            }
        }
    } else if (n_adv < M) {
        restore_recurrent_ckpt();
        strata::kernels::launch_nvfp4_ple_state_update_batch(kernels_, d_ple_conv_state_, d_ple_gvn_, n_adv);
        float eps = 1e-6f;
        for (int l = 0; l < 48; ++l) {
            Nvfp4LayerWeights& lw = layers_[l];
            if (!lw.is_qsa) {
                void* v_qkv = d_gdn_verify_qkv_[l];
                void* v_z = static_cast<uint16_t*>(v_qkv) + 10240;
                void* v_a = static_cast<uint16_t*>(v_qkv) + 16384;
                void* v_b = static_cast<uint16_t*>(v_qkv) + 16432;
                strata::kernels::launch_nvfp4_gated_deltanet_fused_strided(
                    kernels_, v_qkv, v_z, v_a, v_b, lw.gdn.A_log, lw.gdn.dt_bias,
                    lw.gdn.conv1d, d_gdn_conv_in_[l], d_gdn_conv_out_[l],
                    static_cast<const float*>(d_gdn_state_[l]),
                    static_cast<float*>(d_gdn_state_[l]),
                    lw.gdn.norm, d_gdn_out_, n_adv, 16480, eps);
                std::swap(d_gdn_conv_in_[l], d_gdn_conv_out_[l]);
            }
        }
    }
    adapt_vram_experts(h_all_routed_ids_.data(), M, /*min_hits=*/2, /*max_swaps=*/2, n_adv);
    t_wait_acc_ms_ = 0.0;
    t_cpu_acc_ms_ = 0.0;
    n_cpu_exp_acc_ = 0;
    n_gpu_pcie_acc_ = 0;
    n_vram_exp_acc_ = 0;
}

void Qwen4ExpGfx1201Engine::reset_session() {
    HIP_CHECK(hipMemsetAsync(d_recurrent_arena_, 0, conv_ckpt_bytes_, nullptr));
    const size_t gdn_state_bytes = 48 * 128 * 128 * sizeof(float);
    for (int l = 0; l < 48; ++l) {
        if (l % 4 != 3 && d_gdn_state_[l]) {
            HIP_CHECK(hipMemsetAsync(d_gdn_state_[l], 0, gdn_state_bytes, nullptr));
        }
    }
    if (d_raw_k_arena_ && raw_k_arena_bytes_ > 0) {
        HIP_CHECK(hipMemsetAsync(d_raw_k_arena_, 0, raw_k_arena_bytes_, nullptr));
    }
    for (int l = 0; l < 48; ++l) {
        d_gdn_conv_in_[l] = init_gdn_conv_in_[l];
        d_gdn_conv_out_[l] = init_gdn_conv_out_[l];
        Nvfp4MoeWeights& mw = layers_[l].moe;
        for (int e = 0; e < 512; ++e) {
            mw.hit_count[e] = mw.in_vram[e] ? 4 : 0;
        }
        for (int s = 0; s < 32; ++s) {
            mw.slot_last_used[s] = 0;
        }
    }
    step_counter_ = 0;
    mtp_kv_pos_ = 0;
}

void Qwen4ExpGfx1201Engine::save_turn_ckpt(const std::vector<int32_t>& ids, int32_t next_tok) {
    if (!h_turn_ckpt_) return;
    HIP_CHECK(hipMemcpy(h_turn_ckpt_, d_recurrent_arena_, conv_ckpt_bytes_, hipMemcpyDeviceToHost));
    const size_t gdn_state_bytes = 48 * 128 * 128 * sizeof(float);
    uint8_t* h_gdn_dst = static_cast<uint8_t*>(h_turn_ckpt_) + conv_ckpt_bytes_;
    for (int l = 0; l < 48; ++l) {
        if (l % 4 != 3) {
            HIP_CHECK(hipMemcpy(h_gdn_dst, d_gdn_state_[l], gdn_state_bytes, hipMemcpyDeviceToHost));
            h_gdn_dst += gdn_state_bytes;
        }
    }
    if (d_raw_k_arena_ && raw_k_arena_bytes_ > 0) {
        HIP_CHECK(hipMemcpy(static_cast<uint8_t*>(h_turn_ckpt_) + recurrent_ckpt_bytes_,
                            d_raw_k_arena_, raw_k_arena_bytes_, hipMemcpyDeviceToHost));
    }
    for (int l = 0; l < 48; ++l) {
        turn_ckpt_conv_swapped_[l] = (d_gdn_conv_in_[l] != init_gdn_conv_in_[l]);
    }
    turn_ckpt_ids_ = ids;
    turn_ckpt_next_tok_ = next_tok;
    turn_ckpt_mtp_kv_pos_ = mtp_kv_pos_;
}

void Qwen4ExpGfx1201Engine::restore_turn_ckpt(std::vector<int32_t>& history_out, int32_t& next_tok_out) {
    if (!h_turn_ckpt_) return;
    HIP_CHECK(hipMemcpy(d_recurrent_arena_, h_turn_ckpt_, conv_ckpt_bytes_, hipMemcpyHostToDevice));
    const size_t gdn_state_bytes = 48 * 128 * 128 * sizeof(float);
    const uint8_t* h_gdn_src = static_cast<const uint8_t*>(h_turn_ckpt_) + conv_ckpt_bytes_;
    for (int l = 0; l < 48; ++l) {
        if (l % 4 != 3) {
            HIP_CHECK(hipMemcpy(d_gdn_state_[l], h_gdn_src, gdn_state_bytes, hipMemcpyHostToDevice));
            h_gdn_src += gdn_state_bytes;
        }
    }
    if (d_raw_k_arena_ && raw_k_arena_bytes_ > 0) {
        HIP_CHECK(hipMemcpy(d_raw_k_arena_,
                            static_cast<const uint8_t*>(h_turn_ckpt_) + recurrent_ckpt_bytes_,
                            raw_k_arena_bytes_, hipMemcpyHostToDevice));
    }
    for (int l = 0; l < 48; ++l) {
        if (l % 4 != 3) {
            d_gdn_conv_in_[l] = turn_ckpt_conv_swapped_[l] ? init_gdn_conv_out_[l] : init_gdn_conv_in_[l];
            d_gdn_conv_out_[l] = turn_ckpt_conv_swapped_[l] ? init_gdn_conv_in_[l] : init_gdn_conv_out_[l];
        }
    }
    history_out = turn_ckpt_ids_;
    next_tok_out = turn_ckpt_next_tok_;
    mtp_kv_pos_ = turn_ckpt_mtp_kv_pos_;
}

int32_t Qwen4ExpGfx1201Engine::run_mtp_cpu_moe_decode() {
    while (hipEventQuery(ev_route_) == hipErrorNotReady) {
        _mm_pause();
    }
    const uint16_t* h_mixed = reinterpret_cast<const uint16_t*>(h_route_bell_);
    const int32_t* h_ids = reinterpret_cast<const int32_t*>(
        static_cast<const uint8_t*>(h_route_bell_) + 2560 * sizeof(uint16_t));
    const float* h_wts = reinterpret_cast<const float*>(h_ids + 10);

    struct MissItem {
        int32_t e;
        int32_t pair;
    };
    MissItem misses[10];
    int32_t n_miss = 0;
    for (int32_t p = 0; p < 10; ++p) {
        int32_t raw_id = h_ids[p];
        int32_t e = (raw_id >= 0) ? (raw_id & 0xFFFF) : -1;
        if (e < 0 || e >= 512) continue;
        bool forced_cpu = (raw_id & 0x10000) != 0;
        if (forced_cpu || !mtp_.moe.in_gpu_cold[e]) {
            misses[n_miss++] = MissItem{e, p};
        }
    }
    if (n_miss == 0) return 0;

    for (int32_t k = 0; k < 2560; ++k) {
        mtp_act_f32_[k] = bf16_bits_to_f32(h_mixed[k]);
    }
    float* h_accum_f32 = reinterpret_cast<float*>(h_cpu_upload_);
    for (int32_t i = 0; i < n_miss; ++i) {
        int32_t e = misses[i].e;
        int32_t p = misses[i].pair;
        CpuExpertJob& jb = cpu_jobs_[i];
        jb.exp_base = mtp_.moe.host_cpu_ptr[e];
        jb.weight = h_wts[p];
        jb.nt = 1;
        jb.x_f32 = mtp_act_f32_;
        jb.d_x[0] = mtp_act_f32_;
        jb.y_f32 = jb.y_buf_f32;
    }
    cpu_pool_.run_mtp_decode_batch(cpu_jobs_.data(), n_miss, h_accum_f32);
    HIP_CHECK(hipMemcpyAsync(d_cpu_upload_, h_accum_f32, 2560 * sizeof(float), hipMemcpyHostToDevice, nullptr));
    return 1;
}

void Qwen4ExpGfx1201Engine::run_mtp_kv_batch(const int32_t* next_toks, int32_t cell0, int32_t M) {
    if (!mtp_.available || M <= 0) return;
    ensure_context_capacity(cell0 + M);

    uint8_t* h_ple_rows = static_cast<uint8_t*>(h_step_meta_) + 16;
    uint8_t* h_batch_toks = h_ple_rows + static_cast<size_t>(M) * 2560;
    std::memcpy(h_batch_toks, next_toks, static_cast<size_t>(M) * sizeof(int32_t));
    void* d_batch_toks = static_cast<uint8_t*>(d_ple_rows_u8_) + static_cast<size_t>(M) * 2560;
    HIP_CHECK(hipMemcpyAsync(d_batch_toks, h_batch_toks, static_cast<size_t>(M) * sizeof(int32_t),
                             hipMemcpyHostToDevice, nullptr));

    float eps = 1e-6f;
    const int32_t m_tiles64 = (M + 63) / 64;
    const bool use_wmma_hc = (M >= 16);

    strata::kernels::launch_nvfp4_mtp_emb_rms_norm(
        kernels_, static_cast<const int32_t*>(d_batch_toks), d_embed_tokens_,
        mtp_.pre_fc_norm_embedding, d_mtp_emb_norm_, M, eps, nullptr);
    strata::kernels::launch_nvfp4_mtp_hidden_rms_norm(
        kernels_, d_mtp_r_ring_, cell0, kMtpRingMask, mtp_.pre_fc_norm_hidden,
        d_mtp_h_norm_, M, eps, mtp_hnorm_per_stream_, nullptr);

    {
        strata::kernels::launch_nvfp4_linear_auto(
            kernels_, "bf16_linear_2560_2560", "bf16_gemv_2560_2560", "bf16_gemv_m4_2560_2560",
            d_mtp_emb_norm_, mtp_.fc_embedding, d_mtp_e2_, 2560, M);
        int32_t M4 = 4 * M;
        strata::kernels::launch_nvfp4_linear_auto(
            kernels_, "bf16_linear_2560_2560", "bf16_gemv_2560_2560", "bf16_gemv_m4_2560_2560",
            d_mtp_h_norm_, mtp_.fc_hidden, d_mtp_h2_, 2560, M4);
    }
    strata::kernels::launch_nvfp4_mtp_stream_broadcast_add(
        kernels_, d_mtp_h2_, d_mtp_e2_, d_x_stream_a_, M, nullptr);

    void* cur_x = d_x_stream_a_;
    {
        int64_t wi_delta = mtp_.attn_hc.wi_delta;
        void* args_hc[] = {&cur_x, &cur_x, &cur_x, &cur_x, &mtp_.attn_hc.norm, &mtp_.attn_hc.down,
                           &d_hc_h_, &d_attn_inj_, &d_hc_rstd_, &wi_delta, &M, &eps};
        if (use_wmma_hc) {
            kernels_.launch("hc_premix_wmma_c1_u0", 21, m_tiles64, 1, args_hc);
        } else {
            kernels_.launch("hc_premix_c1_u0", 81, M, 1, args_hc);
        }
        void* m_args[] = {&cur_x, &mtp_.attn_hc.norm, &d_hc_rstd_, &d_hc_h_, &mtp_.attn_hc.up, &d_mixed_, &M};
        if (use_wmma_hc) {
            kernels_.launch("hc_mix_wmma", 40, m_tiles64, 1, m_args);
        } else {
            kernels_.launch("hc_mix", 80, M, 1, m_args);
        }
    }

    const void* w_kv_idx = static_cast<const uint16_t*>(mtp_.qsa.in_proj_fused) + 12288ULL * 2560;
    strata::kernels::launch_nvfp4_mtp_kv_idx_proj(
        kernels_, d_mixed_, w_kv_idx, d_qsa_k_b_, d_qsa_v_b_, d_qsa_idx_qk_b_, M, nullptr);

    {
        void* cos_p = static_cast<uint8_t*>(d_rope_cos_f32_) + static_cast<size_t>(cell0) * 32 * sizeof(float);
        void* sin_p = static_cast<uint8_t*>(d_rope_sin_f32_) + static_cast<size_t>(cell0) * 32 * sizeof(float);
        int32_t r2 = M * 2, h2 = 2;
        void* ak[] = {&d_qsa_k_b_, &mtp_.qsa.k_norm, &cos_p, &sin_p, &d_qsa_k_b_, &r2, &h2, &eps};
        kernels_.launch("qk_norm_partial_rope", (M * 2 + 7) / 8, 1, 1, ak);
    }

    void* kc = d_k_cache_[kMtpQsaIdx];
    void* vc = d_v_cache_[kMtpQsaIdx];
    void* rkc = d_raw_k_cache_[kMtpQsaIdx];
    int64_t kc_delta = kc_host_delta_[kMtpQsaIdx];
    int64_t vc_delta = vc_host_delta_[kMtpQsaIdx];
    {
        void* args[] = {&d_qsa_k_b_, &d_qsa_v_b_, &d_qsa_idx_qk_b_, &kc, &vc, &rkc,
                        &kc_delta, &vc_delta, &cell0, &vram_prefix_};
        kernels_.launch("qsa_cache_write_nvfp4", 1, M, 1, args);
    }
}

int32_t Qwen4ExpGfx1201Engine::run_mtp_step_token(int32_t next_tok, int32_t cell, bool from_ring,
                                                  float* prob_out, float* margin_out,
                                                  int32_t ple_prev0, int32_t ple_prev1) {
    if (!mtp_.available) return -1;
    ensure_context_capacity(cell + 1);

    int32_t* meta_i32 = static_cast<int32_t*>(h_step_meta_);
    meta_i32[0] = next_tok;
    meta_i32[1] = cell;
    meta_i32[2] = cell + 1;
    meta_i32[3] = 0;
    HIP_CHECK(hipMemcpyAsync(d_step_meta_, h_step_meta_, 16, hipMemcpyHostToDevice, nullptr));

    int32_t one = 1;
    int32_t four = 4;
    float eps = 1e-6f;
    void* d_cur_ids = static_cast<uint8_t*>(d_route_bell_) + 2560 * sizeof(uint16_t);
    void* d_cur_wts = static_cast<uint8_t*>(d_cur_ids) + 10 * sizeof(int32_t);

    strata::kernels::launch_nvfp4_mtp_emb_rms_norm(
        kernels_, static_cast<const int32_t*>(d_tok_id_), d_embed_tokens_,
        mtp_.pre_fc_norm_embedding, d_mtp_emb_norm_, 1, eps, nullptr);
    if (from_ring) {
        strata::kernels::launch_nvfp4_mtp_hidden_rms_norm(
            kernels_, d_mtp_r_ring_, cell, kMtpRingMask, mtp_.pre_fc_norm_hidden,
            d_mtp_h_norm_, 1, eps, mtp_hnorm_per_stream_, nullptr);
    } else {
        strata::kernels::launch_nvfp4_mtp_hidden_rms_norm(
            kernels_, d_mtp_r_state_, 0, 0, mtp_.pre_fc_norm_hidden,
            d_mtp_h_norm_, 1, eps, mtp_hnorm_per_stream_, nullptr);
    }
    {
        void* a_fe[] = {&d_mtp_emb_norm_, &mtp_.fc_embedding, &d_mtp_e2_};
        kernels_.launch_default("bf16_gemv_2560_2560", a_fe);
        void* a_fh[] = {&d_mtp_h_norm_, &mtp_.fc_hidden, &d_mtp_h2_, &four};
        kernels_.launch_default("bf16_gemv_m4_2560_2560", a_fh);
    }
    strata::kernels::launch_nvfp4_mtp_stream_broadcast_add(
        kernels_, d_mtp_h2_, d_mtp_e2_, d_x_stream_a_, 1, nullptr);

    void* cur_x = d_x_stream_a_;
    void* alt_x = d_x_stream_b_;
    {
        int64_t wi_delta = mtp_.attn_hc.wi_delta;
        void* args[] = {&cur_x, &cur_x, &cur_x, &cur_x, &mtp_.attn_hc.norm, &mtp_.attn_hc.down,
                        &d_hc_h_, &d_attn_inj_, &d_hc_rstd_, &wi_delta, &eps};
        kernels_.launch_default("hc_premix_decode_c1_u0", args);
        void* m_args[] = {&cur_x, &mtp_.attn_hc.norm, &d_hc_rstd_, &d_hc_h_, &mtp_.attn_hc.up, &d_mixed_, &one};
        kernels_.launch_default("hc_mix_decode", m_args);
    }

    {
        void* a1[] = {&d_mixed_, &mtp_.qsa.in_proj_fused, &d_qsa_qg_};
        kernels_.launch_default("bf16_gemv_13952_2560", a1);
    }
    void* kc = d_k_cache_[kMtpQsaIdx];
    void* vc = d_v_cache_[kMtpQsaIdx];
    void* rkc = d_raw_k_cache_[kMtpQsaIdx];
    int64_t kc_delta = kc_host_delta_[kMtpQsaIdx];
    int64_t vc_delta = vc_host_delta_[kMtpQsaIdx];
    {
        void* args[] = {&d_qsa_qg_, &mtp_.qsa.q_norm, &mtp_.qsa.k_norm, &d_rope_cos_f32_, &d_rope_sin_f32_,
                        &d_qsa_q_, &kc, &vc, &rkc, &kc_delta, &vc_delta, &cell, &vram_prefix_, &eps};
        kernels_.launch_default("qsa_prep_and_cache_write_nvfp4", args);
    }
    int32_t Lk = cell + 1;
    int32_t chunk = 256;
    if (Lk <= 256) {
        void* af[] = {&d_qsa_q_, &kc, &vc, &d_qsa_qg_, &d_qsa_attn_raw_, &d_seq_k_i32_};
        kernels_.launch_default("attn_decode_fused_nvfp4", af);
    } else if (Lk <= 1088) {
        int32_t splits = (Lk + chunk - 1) / chunk;
        void* ap[] = {&d_qsa_q_, &kc, &vc, &d_zero_mask_, &d_attn_pm_, &d_attn_pl_, &d_attn_pa_,
                      &d_seq_k_i32_, &chunk, &splits};
        kernels_.launch("attn_decode_partial_nvfp4", splits * 24, 1, 1, ap);
        void* ac[] = {&d_attn_pm_, &d_attn_pl_, &d_attn_pa_, &d_qsa_qg_, &d_qsa_attn_raw_, &splits};
        kernels_.launch("attn_decode_combine", 24, 1, 1, ac);
    } else {
        strata::kernels::launch_nvfp4_qsa_swa_sink_decode_fused(
            kernels_, d_qsa_q_, kc, vc, d_qsa_qg_, d_qsa_attn_raw_,
            kc_delta, vc_delta, cell, vram_prefix_, /*sink_tokens=*/64, /*window_tokens=*/1024, nullptr);
    }
    {
        void* args[] = {&d_qsa_attn_raw_, &mtp_.qsa.o_proj, &d_attn_out_};
        kernels_.launch_default("bf16_gemv_2560_6144", args);
    }

    {
        int64_t wi_delta = mtp_.mlp_hc.wi_delta;
        void* args[] = {&cur_x, &d_attn_out_, &d_attn_inj_, &alt_x, &mtp_.mlp_hc.norm, &mtp_.mlp_hc.down,
                        &d_hc_h_, &d_mlp_inj_, &d_hc_rstd_, &wi_delta, &eps};
        kernels_.launch_default("hc_premix_decode_c1_u1", args);
        std::swap(cur_x, alt_x);
        void* m_args[] = {&cur_x, &mtp_.mlp_hc.norm, &d_hc_rstd_, &d_hc_h_, &mtp_.mlp_hc.up, &d_mixed_, &one};
        kernels_.launch_default("hc_mix_decode", m_args);
    }

    {
        void* a_gu[] = {&d_mixed_, &mtp_.moe.in_proj_fused, &d_sh_gu_};
        kernels_.launch_default("bf16_gemv_1793_2560", a_gu);

        void* ptrs = mtp_.moe.d_ptrs;
        void* ptrs_vram = mtp_.moe.d_ptrs_prefill;
        void* a_ts[] = {&d_sh_gu_, &ptrs, &ptrs_vram, &d_cur_ids, &d_cur_wts, &d_sh_inter_};
        kernels_.launch_default("moe_topk_and_silu_mul_640", a_ts);

        if (mtp_topk_ > 0 && mtp_topk_ < 10) {
            strata::kernels::launch_nvfp4_mtp_topk_truncate(
                kernels_, static_cast<int32_t*>(d_cur_ids), static_cast<float*>(d_cur_wts), mtp_topk_, nullptr);
        }

        HIP_CHECK(hipMemcpyAsync(h_route_bell_, d_route_bell_,
                                 2560 * sizeof(uint16_t) + 10 * sizeof(int32_t) + 10 * sizeof(float),
                                 hipMemcpyDeviceToHost, nullptr));
        HIP_CHECK(hipEventRecord(ev_route_, nullptr));

        void* a_d[] = {&d_sh_inter_, &mtp_.moe.sh_down, &d_sh_out_};
        kernels_.launch_default("bf16_gemv_2560_640", a_d);

        const int32_t mtp_gpu_slots = (mtp_topk_ > 0) ? std::min<int32_t>(3, mtp_topk_) : 3;
        strata::kernels::launch_nvfp4_mtp_moe_decode_gate_up(
            kernels_, d_mixed_, static_cast<const int32_t*>(d_cur_ids),
            static_cast<const int64_t*>(ptrs), d_moe_inter_, mtp_gpu_slots, nullptr);
        strata::kernels::launch_nvfp4_mtp_moe_decode_down(
            kernels_, d_moe_inter_, static_cast<const int32_t*>(d_cur_ids),
            static_cast<const int64_t*>(ptrs), d_moe_y_, mtp_gpu_slots, nullptr);

        int32_t has_cpu = run_mtp_cpu_moe_decode();

        void* a_cb[] = {&d_moe_y_, &d_cur_ids, &ptrs, &d_cur_wts, &d_cpu_upload_,
                        &d_sh_out_, &d_sh_scale_, &d_mlp_out_, &has_cpu};
        kernels_.launch_default("moe_combine_decode", a_cb);
    }

    {
        int64_t zero_delta = 0;
        void* prev_out = d_mlp_out_;
        void* prev_inj = d_mlp_inj_;
        void* args[] = {&cur_x, &prev_out, &prev_inj, &d_mtp_r_state_, &mtp_.final_hc.norm, &mtp_.final_hc.down,
                        &d_hc_h_, &d_hc_h_, &d_hc_rstd_, &zero_delta, &eps};
        kernels_.launch_default("hc_premix_decode_c0_u1", args);
        void* m_args[] = {&d_mtp_r_state_, &mtp_.final_hc.norm, &d_hc_rstd_, &d_hc_h_, &mtp_.final_hc.up, &d_final_hidden_, &one};
        kernels_.launch_default("hc_mix_decode", m_args);
    }

    if (d_mtp_dvocab_ != nullptr && n_mtp_dvocab_ > 0) {
        strata::kernels::launch_nvfp4_mtp_dvocab_gemv(
            kernels_, d_final_hidden_, d_lm_head_, static_cast<const int32_t*>(d_mtp_dvocab_),
            d_logits_bf16_, n_mtp_dvocab_, nullptr);
        strata::kernels::launch_nvfp4_argmax_prob_margin_dvocab(
            kernels_, d_logits_bf16_, static_cast<const int32_t*>(d_mtp_dvocab_), n_mtp_dvocab_,
            static_cast<int32_t*>(d_next_tok_),
            static_cast<strata::kernels::Nvfp4ArgmaxSummary*>(d_argmax_summary_), nullptr);
    } else {
        void* a_lm[] = {&d_final_hidden_, &d_lm_head_, &d_logits_bf16_};
        kernels_.launch_default("bf16_gemv_248320_2560", a_lm);
        strata::kernels::launch_nvfp4_argmax_prob_margin_248320(
            kernels_, d_logits_bf16_, static_cast<int32_t*>(d_next_tok_),
            static_cast<strata::kernels::Nvfp4ArgmaxSummary*>(d_argmax_summary_), 1, nullptr);
    }

    // Overlap CPU PLE N-gram shard row prefetch with GPU MTP execution before blocking on hipMemcpy
    if (ple_prev0 != -2 && ple_prev1 != -2) {
        fetch_ple_rows_cached(&next_tok, &ple_prev0, &ple_prev1, 1, nullptr);
    }

    strata::kernels::Nvfp4ArgmaxSummary s{};
    HIP_CHECK(hipMemcpy(&s, d_argmax_summary_, sizeof(s), hipMemcpyDeviceToHost));
    if (prob_out) *prob_out = s.prob;
    if (margin_out) *margin_out = s.margin;
    return s.tok_id;
}

void Qwen4ExpGfx1201Engine::mtp_catchup_kv(const std::vector<int32_t>& history, int32_t target_kv_pos) {
    if (!mtp_.available) return;
    int32_t max_valid = static_cast<int32_t>(history.size()) - 1;
    int32_t end_pos = std::min(target_kv_pos, max_valid);
    if (end_pos <= mtp_kv_pos_) return;
    if (d_mtp_dvocab_ != nullptr && !mtp_in_dvocab_.empty()) {
        int32_t sync_from = n_mtp_dvocab_;
        for (int32_t p = mtp_kv_pos_; p <= end_pos; ++p) {
            int32_t tid = history[p];
            if (tid >= 0 && tid < 248320 && !mtp_in_dvocab_[tid] && n_mtp_dvocab_ < 65536) {
                mtp_in_dvocab_[tid] = 1;
                h_mtp_dvocab_.push_back(tid);
                n_mtp_dvocab_++;
            }
        }
        if (n_mtp_dvocab_ > sync_from) {
            HIP_CHECK(hipMemcpyAsync(static_cast<int32_t*>(d_mtp_dvocab_) + sync_from,
                                     h_mtp_dvocab_.data() + sync_from,
                                     static_cast<size_t>(n_mtp_dvocab_ - sync_from) * sizeof(int32_t),
                                     hipMemcpyHostToDevice, nullptr));
        }
    }
    if (end_pos - mtp_kv_pos_ > kMtpRingCap) {
        mtp_kv_pos_ = end_pos - kMtpRingCap;
    }
    while (mtp_kv_pos_ < end_pos) {
        int32_t chunk = std::min<int32_t>(kMaxBatch, end_pos - mtp_kv_pos_);
        run_mtp_kv_batch(&history[mtp_kv_pos_ + 1], mtp_kv_pos_, chunk);
        mtp_kv_pos_ += chunk;
    }
}

void Qwen4ExpGfx1201Engine::draft_mtp(int32_t next_tok, const std::vector<int32_t>& history, int32_t max_drafts,
                                      float min_prob, std::vector<int32_t>& draft_out,
                                      std::vector<float>* probs_out) {
    draft_out.clear();
    if (probs_out) probs_out->clear();
    if (!mtp_.available || max_drafts <= 0 || history.empty()) return;

    // Target certainty & margin pre-gating (DS-2):
    // Skip neural MTP step 0 when the target model's last token had very high entropy
    // (low top-1 probability and narrow top-2 logit margin).
    if (last_target_prob_ < 0.22f && last_target_margin_ < 0.85f) {
        return;
    }

    const int32_t cur_pos = static_cast<int32_t>(history.size());
    if (mtp_kv_pos_ < cur_pos - 1) {
        mtp_catchup_kv(history, cur_pos - 1);
    }

    if (d_mtp_dvocab_ != nullptr && !mtp_in_dvocab_.empty()) {
        int32_t sync_from = n_mtp_dvocab_;
        auto ensure_in_dvocab = [&](int32_t tid) {
            if (tid >= 0 && tid < 248320 && !mtp_in_dvocab_[tid] && n_mtp_dvocab_ < 65536) {
                mtp_in_dvocab_[tid] = 1;
                h_mtp_dvocab_.push_back(tid);
                n_mtp_dvocab_++;
            }
        };
        ensure_in_dvocab(history.back());
        ensure_in_dvocab(next_tok);
        if (n_mtp_dvocab_ > sync_from) {
            HIP_CHECK(hipMemcpyAsync(static_cast<int32_t*>(d_mtp_dvocab_) + sync_from,
                                     h_mtp_dvocab_.data() + sync_from,
                                     static_cast<size_t>(n_mtp_dvocab_ - sync_from) * sizeof(int32_t),
                                     hipMemcpyHostToDevice, nullptr));
        }
    }

    int32_t cur_tok = next_tok;
    int32_t p0 = (history.size() >= 2) ? history[history.size() - 2] : strata::kernels::TOKEN_NULL;
    int32_t p1 = history.back();
    double cum_surv = 1.0;
    for (int32_t j = 0; j < max_drafts; ++j) {
        float prob = 0.0f;
        float margin = 0.0f;
        const bool from_ring = (j == 0);
        const int32_t cell = (cur_pos - 1) + j;
        int32_t d_tok = run_mtp_step_token(cur_tok, cell, from_ring, &prob, &margin, p0, p1);
        if (j == 0) {
            mtp_kv_pos_ = cur_pos;
        }
        if (d_tok < 0 || d_tok == 248044 || d_tok == 248046) {
            break;
        }
        // Calibrated step survival score combining top-1 softmax probability and top-2 logit margin
        const float step_conf = std::clamp(prob + 0.04f * std::min(margin, 3.0f), 0.0f, 0.99f);
        const float keep_floor = (j == 0) ? std::min(min_prob, 0.22f) : 0.18f;
        if (step_conf < keep_floor) {
            break;
        }
        draft_out.push_back(d_tok);
        if (probs_out) probs_out->push_back(prob);
        p0 = p1;
        p1 = cur_tok;
        cur_tok = d_tok;
        cum_surv *= static_cast<double>(step_conf);
        // Speculative throughput scheduler (STS) stopping rule for launching step j+1:
        // Require cumulative prefix survival >= 0.48 and current step confidence >= step_chain_thresh.
        const float step_chain_thresh = (j == 0) ? std::max(min_prob, 0.58f) : std::max(min_prob, 0.72f);
        if (step_conf < step_chain_thresh || cum_surv < 0.48) {
            break;
        }
    }
}

int32_t Qwen4ExpGfx1201Engine::run_cpu_moe_layer(int l, int32_t M, bool is_prefill, bool use_gpu_cold_cap) {
    auto t0 = std::chrono::steady_clock::now();
    while (hipEventQuery(ev_route_) == hipErrorNotReady) {
        _mm_pause();
    }
    auto t1 = std::chrono::steady_clock::now();
    t_wait_acc_ms_ += std::chrono::duration<double, std::milli>(t1 - t0).count();
    const uint16_t* h_mixed = reinterpret_cast<const uint16_t*>(h_route_bell_);
    const int32_t* h_ids = reinterpret_cast<const int32_t*>(
        static_cast<const uint8_t*>(h_route_bell_) + static_cast<size_t>(M) * 2560 * sizeof(uint16_t));
    const int32_t pairs = M * 10;
    int32_t* dst_routed = h_all_routed_ids_.data() + static_cast<size_t>(l) * pairs;

    const Nvfp4MoeWeights& mw = layers_[l].moe;
    MoeMissItem* misses = moe_misses_.data();
    uint8_t* tok_needed = moe_tok_needed_.data();
    std::memset(tok_needed, 0, static_cast<size_t>(M));
    int32_t n_miss = 0;
    const bool use_gpu_cold = !is_prefill || use_gpu_cold_cap;
    int32_t exp_cnt[512]{};
    for (int32_t p = 0; p < pairs; ++p) {
        int32_t raw_id = h_ids[p];
        int32_t e = (raw_id >= 0) ? (raw_id & 0xFFFF) : -1;
        dst_routed[p] = e;
        if (e < 0 || e >= 512) continue;
        bool forced_cpu = (raw_id & 0x10000) != 0;
        if (mw.in_vram[e]) {
            n_vram_exp_acc_++;
        } else if (use_gpu_cold && !forced_cpu && mw.in_gpu_cold[e]) {
            n_gpu_pcie_acc_++;
        }
        bool on_cpu = !mw.in_vram[e] && (!use_gpu_cold || forced_cpu || !mw.in_gpu_cold[e]);
        if (on_cpu) {
            misses[n_miss++] = MoeMissItem{e, p};
            exp_cnt[e]++;
            tok_needed[p / 10] = 1;
        }
    }
    n_cpu_exp_acc_ += n_miss;
    if (n_miss == 0) return 0;

    if (!is_prefill) {
        quantize_act_bf16_2560(h_mixed, act_q_, act_d_, act_nb_);
        const float* h_wts = reinterpret_cast<const float*>(h_ids + 10);
        float* h_accum_f32 = reinterpret_cast<float*>(h_cpu_upload_);
        for (int32_t i = 0; i < n_miss; ++i) {
            int32_t e = misses[i].e;
            int32_t p = misses[i].pair;
            CpuExpertJob& jb = cpu_jobs_[i];
            jb.exp_base = mw.host_cpu_ptr[e];
            jb.gate_s2 = mw.h_s2[e * 3 + 0];
            jb.up_s2 = mw.h_s2[e * 3 + 1];
            jb.down_s2 = mw.h_s2[e * 3 + 2];
            jb.weight = h_wts[p];
            jb.nt = 1;
            jb.q_x[0] = act_q_;
            jb.d_x[0] = act_d_;
            jb.neg_bias_x[0] = act_nb_;
            jb.y_f32 = jb.y_buf_f32;
        }
        cpu_pool_.run_decode_batch(cpu_jobs_.data(), n_miss, h_accum_f32);
        HIP_CHECK(hipMemcpyAsync(d_cpu_upload_, h_accum_f32, 2560 * sizeof(float), hipMemcpyHostToDevice, nullptr));
        auto t2 = std::chrono::steady_clock::now();
        t_cpu_acc_ms_ += std::chrono::duration<double, std::milli>(t2 - t1).count();
        return 1;
    }

    if (n_miss > 1) {
        int32_t exp_off[512];
        int32_t cur_off = 0;
        for (int32_t e = 0; e < 512; ++e) {
            exp_off[e] = cur_off;
            cur_off += exp_cnt[e];
        }
        MoeMissItem* sorted_misses = moe_misses_sorted_.data();
        for (int32_t i = 0; i < n_miss; ++i) {
            int32_t e = misses[i].e;
            sorted_misses[exp_off[e]++] = misses[i];
        }
        misses = sorted_misses;
    }

    cpu_pool_.run_quantize_batch(h_mixed, tok_needed, act_q_, act_d_, act_nb_, M);

    uint16_t* h_y_out = static_cast<uint16_t*>(h_cpu_upload_);
    int32_t* upload_pairs = moe_upload_pairs_.data();
    int32_t n_jobs = 0;
    for (int32_t i = 0; i < n_miss;) {
        int32_t e = misses[i].e;
        int nt = 0;
        CpuExpertJob& jb = cpu_jobs_[n_jobs++];
        jb.exp_base = mw.host_cpu_ptr[e];
        jb.gate_s2 = mw.h_s2[e * 3 + 0];
        jb.up_s2 = mw.h_s2[e * 3 + 1];
        jb.down_s2 = mw.h_s2[e * 3 + 2];
        while (i < n_miss && misses[i].e == e && nt < kMaxJobToks) {
            int32_t p = misses[i].pair;
            int32_t t = p / 10;
            jb.q_x[nt] = act_q_ + static_cast<size_t>(t) * 2560;
            jb.d_x[nt] = act_d_ + static_cast<size_t>(t) * 160;
            jb.neg_bias_x[nt] = act_nb_ + static_cast<size_t>(t) * 640;
            jb.y_out[nt] = h_y_out + static_cast<size_t>(i) * 2560;
            upload_pairs[i] = p;
            nt++;
            i++;
        }
        jb.nt = nt;
    }

    cpu_pool_.run_batch(cpu_jobs_.data(), n_jobs);

    if (n_miss == 1) {
        void* dst = static_cast<uint16_t*>(d_moe_y_) + static_cast<size_t>(upload_pairs[0]) * 2560;
        HIP_CHECK(hipMemcpyAsync(dst, h_y_out, 2560 * sizeof(uint16_t), hipMemcpyHostToDevice, nullptr));
    } else {
        size_t y_bytes = static_cast<size_t>(n_miss) * 2560 * sizeof(uint16_t);
        size_t idx_bytes = static_cast<size_t>(n_miss) * sizeof(int32_t);
        std::memcpy(static_cast<uint8_t*>(h_cpu_upload_) + y_bytes, upload_pairs, idx_bytes);
        HIP_CHECK(hipMemcpyAsync(d_cpu_upload_, h_cpu_upload_, y_bytes + idx_bytes, hipMemcpyHostToDevice, nullptr));
        void* d_src = d_cpu_upload_;
        void* d_idx = static_cast<uint8_t*>(d_cpu_upload_) + y_bytes;
        void* s_args[] = {&d_src, &d_idx, &d_moe_y_, &n_miss};
        kernels_.launch("moe_scatter_rows", n_miss, 1, 1, s_args);
    }
    return 1;
}

void Qwen4ExpGfx1201Engine::adapt_vram_experts(const int32_t* routed_ids, int32_t M, uint32_t min_hits, int max_swaps,
                                               int32_t valid_toks) {
    step_counter_++;
    int swaps_done = 0;
    const int32_t pairs_per_layer = M * 10;
    const int32_t valid_pairs = (valid_toks > 0 && valid_toks < M) ? (valid_toks * 10) : pairs_per_layer;
    for (int l = 0; l < 48; ++l) {
        Nvfp4MoeWeights& mw = layers_[l].moe;
        if (mw.n_dyn <= 0) continue;
        const int32_t* l_ids = routed_ids + static_cast<size_t>(l) * pairs_per_layer;
        bool ptrs_dirty = false;

        // Pass 1: record hit_count and mark slot_last_used for all accepted tokens in the batch
        // before considering any slot evictions, so token 0 never evicts a VRAM expert used by token 1..N-1.
        for (int32_t i = 0; i < valid_pairs; ++i) {
            int32_t e = l_ids[i];
            if (e < 0 || e >= 512) continue;
            mw.hit_count[e]++;
            if (mw.in_vram[e]) {
                int16_t s = mw.dyn_slot[e];
                if (s >= 0) {
                    mw.slot_last_used[s] = step_counter_;
                }
            }
        }

        // Pass 2: promote hot cold-pinned experts into genuinely stale dynamic VRAM slots.
        if (swaps_done < max_swaps) {
            for (int32_t i = 0; i < valid_pairs && swaps_done < max_swaps; ++i) {
                int32_t e = l_ids[i];
                if (e < 0 || e >= 512) continue;
                if (!mw.in_vram[e] && mw.in_gpu_cold[e] && mw.hit_count[e] >= min_hits) {
                    int best_s = -1;
                    uint32_t best_age = (step_counter_ > 8) ? (step_counter_ - 8) : 0;
                    for (int s = 0; s < mw.n_dyn; ++s) {
                        int32_t cand_old = mw.slot_exp[s];
                        if (mw.slot_last_used[s] < best_age &&
                            (cand_old < 0 || mw.hit_count[e] >= mw.hit_count[cand_old] + min_hits)) {
                            best_age = mw.slot_last_used[s];
                            best_s = s;
                        }
                    }
                    if (best_s >= 0) {
                        int32_t old_e = mw.slot_exp[best_s];
                        uint32_t old_hits = (old_e >= 0 && old_e < 512) ? mw.hit_count[old_e] : 0;
                        for (int k = 0; k < 6; ++k) mw.h_ptrs_prefill[old_e * 6 + k] = 0;
                        if (mw.in_gpu_cold[old_e] && mw.host_dev_ptr[old_e]) {
                            uint8_t* old_hdev = mw.host_dev_ptr[old_e];
                            mw.h_ptrs[old_e * 6 + 0] = reinterpret_cast<int64_t>(old_hdev + EXP_OFF_GATE_W);
                            mw.h_ptrs[old_e * 6 + 1] = reinterpret_cast<int64_t>(old_hdev + EXP_OFF_GATE_S);
                            mw.h_ptrs[old_e * 6 + 2] = reinterpret_cast<int64_t>(old_hdev + EXP_OFF_UP_W);
                            mw.h_ptrs[old_e * 6 + 3] = reinterpret_cast<int64_t>(old_hdev + EXP_OFF_UP_S);
                            mw.h_ptrs[old_e * 6 + 4] = reinterpret_cast<int64_t>(old_hdev + EXP_OFF_DOWN_W);
                            mw.h_ptrs[old_e * 6 + 5] = reinterpret_cast<int64_t>(old_hdev + EXP_OFF_DOWN_S);
                        } else {
                            for (int k = 0; k < 6; ++k) mw.h_ptrs[old_e * 6 + k] = 0;
                        }
                        mw.in_vram[old_e] = false;
                        mw.dyn_slot[old_e] = -1;

                        uint8_t* vslot = mw.dyn_vram_base
                                             ? (mw.dyn_vram_base + static_cast<size_t>(best_s) * EXP_STRIDE_BYTES)
                                             : (mw.vram_base + static_cast<size_t>(mw.n_pinned + best_s) * EXP_STRIDE_BYTES);
                        HIP_CHECK(hipMemcpyAsync(vslot, mw.host_cpu_ptr[e], EXP_STRIDE_BYTES,
                                                 hipMemcpyHostToDevice, nullptr));
                        mw.h_ptrs[e * 6 + 0] = reinterpret_cast<int64_t>(vslot + EXP_OFF_GATE_W);
                        mw.h_ptrs[e * 6 + 1] = reinterpret_cast<int64_t>(vslot + EXP_OFF_GATE_S);
                        mw.h_ptrs[e * 6 + 2] = reinterpret_cast<int64_t>(vslot + EXP_OFF_UP_W);
                        mw.h_ptrs[e * 6 + 3] = reinterpret_cast<int64_t>(vslot + EXP_OFF_UP_S);
                        mw.h_ptrs[e * 6 + 4] = reinterpret_cast<int64_t>(vslot + EXP_OFF_DOWN_W);
                        mw.h_ptrs[e * 6 + 5] = reinterpret_cast<int64_t>(vslot + EXP_OFF_DOWN_S);
                        for (int k = 0; k < 6; ++k) mw.h_ptrs_prefill[e * 6 + k] = mw.h_ptrs[e * 6 + k];
                        mw.in_vram[e] = true;
                        mw.dyn_slot[e] = static_cast<int16_t>(best_s);
                        mw.slot_exp[best_s] = e;
                        mw.slot_last_used[best_s] = step_counter_;
                        mw.hit_count[e] = std::max(mw.hit_count[e], old_hits + min_hits);
                        ptrs_dirty = true;
                        swaps_done++;
                    }
                }
            }
        }
        if (ptrs_dirty) {
            HIP_CHECK(hipMemcpyAsync(mw.d_ptrs, mw.h_ptrs, 512 * 6 * sizeof(int64_t),
                                     hipMemcpyHostToDevice, nullptr));
            HIP_CHECK(hipMemcpyAsync(mw.d_ptrs_prefill, mw.h_ptrs_prefill,
                                     512 * 6 * sizeof(int64_t), hipMemcpyHostToDevice, nullptr));
        }
    }
}

void* Qwen4ExpGfx1201Engine::gpu_alloc(size_t bytes, bool zero) {
    void* p = nullptr;
    HIP_CHECK(hipMalloc(&p, bytes));
    if (zero) HIP_CHECK(hipMemset(p, 0, bytes));
    vram_allocs_.push_back(p);
    return p;
}

void* Qwen4ExpGfx1201Engine::host_pinned_alloc(size_t bytes) {
    void* p = nullptr;
    HIP_CHECK(hipHostMalloc(&p, bytes, hipHostMallocDefault));
    std::memset(p, 0, bytes);
    host_pinned_.push_back(p);
    return p;
}

void Qwen4ExpGfx1201Engine::alloc_state_and_scratch() {
    const size_t B = static_cast<size_t>(kMaxBatch);
    HIP_CHECK(hipEventCreateWithFlags(&ev_mixed_, hipEventDisableTiming));
    HIP_CHECK(hipEventCreateWithFlags(&ev_route_, hipEventDisableTiming));
    if (std::getenv("STRATA_NVFP4_TIMING") != nullptr) {
        for (int i = 0; i < 9; ++i) HIP_CHECK(hipEventCreate(&ev_prof_[i]));
    }
    int64_t* h_ptrs_arena =
        static_cast<int64_t*>(host_pinned_alloc(49ULL * 2ULL * 512ULL * 6ULL * sizeof(int64_t)));
    for (int l = 0; l < 48; ++l) {
        layers_[l].moe.h_ptrs = h_ptrs_arena + static_cast<size_t>(l * 2 + 0) * (512 * 6);
        layers_[l].moe.h_ptrs_prefill = h_ptrs_arena + static_cast<size_t>(l * 2 + 1) * (512 * 6);
    }
    mtp_.moe.h_ptrs = h_ptrs_arena + 96ULL * (512 * 6);
    mtp_.moe.h_ptrs_prefill = h_ptrs_arena + 97ULL * (512 * 6);
    cpu_jobs_.resize(512 + (B * 10) / static_cast<size_t>(kMaxJobToks) + 32);
    h_all_routed_ids_.resize(48 * B * 10, 0);
    ple_miss_offs_.resize(B * 16);
    ple_miss_cache_dsts_.resize(B * 16);
    ple_miss_user_dsts_.resize(B * 16);
    moe_misses_.resize(B * 10);
    moe_misses_sorted_.resize(B * 10);
    moe_tok_needed_.resize(B, 0);
    moe_upload_pairs_.resize(B * 10);

    auto alloc_aligned_host = [&](size_t bytes) -> void* {
        void* p = nullptr;
        if (::posix_memalign(&p, 64, bytes) != 0 || !p) {
            std::fprintf(stderr, "posix_memalign failed for %zu bytes\n", bytes);
            std::exit(1);
        }
        std::memset(p, 0, bytes);
        host_mallocs_.push_back(p);
        return p;
    };
    act_q_ = static_cast<int8_t*>(alloc_aligned_host(B * 2560 * sizeof(int8_t)));
    act_d_ = static_cast<float*>(alloc_aligned_host(B * 160 * sizeof(float)));
    act_nb_ = static_cast<int32_t*>(alloc_aligned_host(B * 640 * sizeof(int32_t)));

    d_step_meta_ = gpu_alloc(16 + B * 2560 + B * sizeof(int32_t));
    h_step_meta_ = host_pinned_alloc(16 + B * 2560 + B * sizeof(int32_t));
    d_tok_id_ = d_step_meta_;
    d_pos_i32_ = static_cast<uint8_t*>(d_step_meta_) + 4;
    d_seq_k_i32_ = static_cast<uint8_t*>(d_step_meta_) + 8;
    d_ple_rows_u8_ = static_cast<uint8_t*>(d_step_meta_) + 16;

    d_step_out_ = gpu_alloc((B + 48 * B * 10) * sizeof(int32_t));
    d_next_tok_ = d_step_out_;
    d_moe_ids_ = static_cast<int32_t*>(d_step_out_) + B;

    const size_t bell_bytes = B * (2560 * sizeof(uint16_t) + 10 * sizeof(int32_t) + 10 * sizeof(float));
    d_route_bell_ = gpu_alloc(bell_bytes);
    h_route_bell_ = host_pinned_alloc(bell_bytes);
    d_mixed_ = d_route_bell_;

    const size_t upload_bytes = B * 10 * (2560 * sizeof(uint16_t) + sizeof(int32_t));
    d_cpu_upload_ = gpu_alloc(upload_bytes);
    h_cpu_upload_ = host_pinned_alloc(upload_bytes);
    HIP_CHECK(hipHostGetDevicePointer(&d_cpu_upload_host_dev_, h_cpu_upload_, 0));

    d_x_stream_a_ = gpu_alloc(B * 10240 * sizeof(uint16_t));
    d_x_stream_b_ = gpu_alloc(B * 10240 * sizeof(uint16_t));
    d_hc_h_ = gpu_alloc(B * 320 * sizeof(uint16_t));
    d_hc_rstd_ = gpu_alloc(B * 4 * sizeof(float));
    d_attn_inj_ = gpu_alloc(B * 4 * sizeof(uint16_t));
    d_mlp_inj_ = gpu_alloc(B * 4 * sizeof(uint16_t));
    d_attn_out_ = gpu_alloc(B * 2560 * sizeof(uint16_t));
    d_mlp_out_ = gpu_alloc(B * 2560 * sizeof(uint16_t));
    d_final_hidden_ = gpu_alloc(2560 * sizeof(uint16_t));
    d_logits_bf16_ = gpu_alloc(16ULL * 248320 * sizeof(uint16_t));
    d_proj_scratch_ = gpu_alloc(B * 16480 * sizeof(uint16_t));
    d_ple_gvn_ = gpu_alloc(B * 10240 * sizeof(uint16_t));

    // Unified intra-phase scratch arena (aliases mutually exclusive PLE, GDN, QSA, MoE, and MTP pre-FC buffers)
    auto align256 = [](size_t x) -> size_t { return (x + 255ULL) & ~255ULL; };
    const size_t max_moe_tiles = (B * 10 + 15) / 16 + 512;
    const size_t qsa_scores_bytes = std::max<size_t>(4194304, static_cast<size_t>(max_blocks_)) * sizeof(float);

    const size_t ph_ple_bytes = align256(B * 2560 * sizeof(uint16_t)) +
                                align256(B * 12800 * sizeof(uint16_t)) +
                                align256(B * 2560 * sizeof(uint16_t)) +
                                align256(B * 10240 * sizeof(uint16_t));
    const size_t ph_gdn_bytes = align256(B * 16480 * sizeof(uint16_t)) +
                                align256(B * 6144 * sizeof(uint16_t));
    const size_t ph_qsa_bytes = align256(B * 13952 * sizeof(uint16_t)) +
                                align256(B * 6144 * sizeof(uint16_t)) +
                                align256(B * 6144 * sizeof(uint16_t)) +
                                align256(B * 512 * sizeof(int32_t)) +
                                align256(qsa_scores_bytes);
    const size_t ph_moe_bytes = align256(B * 640 * sizeof(uint16_t)) +
                                align256(B * 2560 * sizeof(uint16_t)) +
                                align256(B * 16 * sizeof(float)) +
                                align256(max_moe_tiles * 16 * sizeof(int32_t)) +
                                align256(max_moe_tiles * sizeof(int32_t)) +
                                align256(max_moe_tiles * 16 * 640 * sizeof(uint16_t)) +
                                align256(B * 16 * 2560 * sizeof(uint16_t));
    const size_t ph_mtp_bytes = align256(B * 2560 * sizeof(uint16_t)) +
                                align256(B * 2560 * sizeof(uint16_t)) +
                                align256(B * 10240 * sizeof(uint16_t)) +
                                align256(B * 10240 * sizeof(uint16_t));
    const size_t phase_arena_bytes = std::max({ph_ple_bytes, ph_gdn_bytes, ph_qsa_bytes, ph_moe_bytes, ph_mtp_bytes});
    d_phase_arena_ = gpu_alloc(phase_arena_bytes, true);
    uint8_t* ph_base = static_cast<uint8_t*>(d_phase_arena_);

    // Phase 1: PLE
    {
        size_t off = 0;
        d_ple_embed_ = ph_base + off; off += align256(B * 2560 * sizeof(uint16_t));
        d_ple_k_ = ph_base + off;     off += align256(B * 12800 * sizeof(uint16_t));
        d_ple_v_ = ph_base + off;     off += align256(B * 2560 * sizeof(uint16_t));
        d_ple_gv_ = ph_base + off;    off += align256(B * 10240 * sizeof(uint16_t));
    }

    const size_t ple_conv_bytes = 9 * 10240 * sizeof(uint16_t);
    const size_t gdn_conv_bytes = 3 * 10240 * sizeof(uint16_t);
    const size_t gdn_state_bytes = 48 * 128 * 128 * sizeof(float);
    const size_t raw_k_bytes = 1024ULL * 128 * sizeof(uint16_t);
    conv_ckpt_bytes_ = ple_conv_bytes + 36ULL * 2 * gdn_conv_bytes;
    recurrent_ckpt_bytes_ = conv_ckpt_bytes_ + 36ULL * gdn_state_bytes;
    raw_k_arena_bytes_ = static_cast<size_t>(kNumQsaCaches) * raw_k_bytes;
    recurrent_arena_bytes_ = recurrent_ckpt_bytes_ + raw_k_arena_bytes_;
    d_recurrent_arena_ = gpu_alloc(recurrent_ckpt_bytes_, true);
    d_recurrent_ckpt_ = gpu_alloc(recurrent_ckpt_bytes_, false);
    d_raw_k_arena_ = gpu_alloc(raw_k_arena_bytes_, true);
    void* h_turn = nullptr;
    if (::posix_memalign(&h_turn, 2 * 1024 * 1024, recurrent_arena_bytes_) == 0 && h_turn) {
        (void) ::madvise(h_turn, recurrent_arena_bytes_, MADV_HUGEPAGE);
        h_turn_ckpt_ = h_turn;
        host_mallocs_.push_back(h_turn);
    }
    uint8_t* rec_conv_cur = static_cast<uint8_t*>(d_recurrent_arena_);
    uint8_t* rec_state_cur = static_cast<uint8_t*>(d_recurrent_arena_) + conv_ckpt_bytes_;
    uint8_t* ckpt_state_cur = static_cast<uint8_t*>(d_recurrent_ckpt_) + conv_ckpt_bytes_;
    d_ple_conv_state_ = rec_conv_cur;
    rec_conv_cur += ple_conv_bytes;

    // Phase 2: GDN
    {
        size_t off = 0;
        d_gdn_qkv_ = ph_base + off; off += align256(B * 16480 * sizeof(uint16_t));
        d_gdn_out_ = ph_base + off; off += align256(B * 6144 * sizeof(uint16_t));
    }
    d_gdn_z_ = static_cast<uint16_t*>(d_gdn_qkv_) + 10240;
    d_gdn_a_ = static_cast<uint16_t*>(d_gdn_qkv_) + 16384;
    d_gdn_b_ = static_cast<uint16_t*>(d_gdn_qkv_) + 16432;
    d_gdn_z_b_ = static_cast<uint16_t*>(d_gdn_qkv_) + B * 10240;
    d_gdn_a_b_ = static_cast<uint16_t*>(d_gdn_qkv_) + B * 16384;
    d_gdn_b_b_ = static_cast<uint16_t*>(d_gdn_qkv_) + B * 16432;
    for (int l = 0; l < 48; ++l) {
        if (l % 4 != 3) {
            d_gdn_conv_in_[l] = rec_conv_cur;
            init_gdn_conv_in_[l] = rec_conv_cur;
            rec_conv_cur += gdn_conv_bytes;
            d_gdn_conv_out_[l] = rec_conv_cur;
            init_gdn_conv_out_[l] = rec_conv_cur;
            rec_conv_cur += gdn_conv_bytes;
            d_gdn_state_[l] = rec_state_cur;
            rec_state_cur += gdn_state_bytes;
            d_gdn_state_ckpt_[l] = ckpt_state_cur;
            ckpt_state_cur += gdn_state_bytes;
        }
    }

    // Phase 3: QSA
    {
        size_t off = 0;
        d_qsa_qg_ = ph_base + off;       off += align256(B * 13952 * sizeof(uint16_t));
        d_qsa_q_ = ph_base + off;        off += align256(B * 6144 * sizeof(uint16_t));
        d_qsa_attn_raw_ = ph_base + off; off += align256(B * 6144 * sizeof(uint16_t));
        d_qsa_sel_ids_ = ph_base + off;  off += align256(B * 512 * sizeof(int32_t));
    }
    d_qsa_scores_ = gpu_alloc(qsa_scores_bytes, true);
    d_qsa_k_ = static_cast<uint16_t*>(d_qsa_qg_) + 12288;
    d_qsa_v_ = static_cast<uint16_t*>(d_qsa_qg_) + 12800;
    d_qsa_idx_qk_ = static_cast<uint16_t*>(d_qsa_qg_) + 13312;
    d_qsa_k_b_ = static_cast<uint16_t*>(d_qsa_qg_) + B * 12288;
    d_qsa_v_b_ = static_cast<uint16_t*>(d_qsa_qg_) + B * 12800;
    d_qsa_idx_qk_b_ = static_cast<uint16_t*>(d_qsa_qg_) + B * 13312;
    d_zero_mask_ = gpu_alloc(2048 * sizeof(float), true);
    int max_splits = 10;
    d_attn_pm_ = gpu_alloc(24 * max_splits * sizeof(float));
    d_attn_pl_ = gpu_alloc(24 * max_splits * sizeof(float));
    d_attn_pa_ = gpu_alloc(24 * max_splits * 256 * sizeof(float));

    uint8_t* raw_k_cur = static_cast<uint8_t*>(d_raw_k_arena_);
    for (int q = 0; q < kNumQsaCaches; ++q) {
        d_raw_k_cache_[q] = raw_k_cur;
        raw_k_cur += raw_k_bytes;
    }

    const size_t gran = strata::core::vmm_available()
                            ? static_cast<size_t>(strata::core::vmm_granularity())
                            : (2ULL * 1024 * 1024);
    vram_kv_stride_bytes_ = ((static_cast<size_t>(vram_cap_) * 2 * 160) + gran - 1) & ~(gran - 1);
    vram_bk_stride_bytes_ = ((static_cast<size_t>(vram_blocks_) * 128 * sizeof(uint16_t)) + gran - 1) & ~(gran - 1);
    const size_t total_kv_vmm_bytes = static_cast<size_t>(kNumQsaCaches) * (2 * vram_kv_stride_bytes_ + vram_bk_stride_bytes_);

    kv_vmm_ok_ = strata::core::vmm_available() && kv_vmm_range_.reserve(total_kv_vmm_bytes);
    if (!kv_vmm_ok_) {
        vram_blocks_ = std::min<int32_t>(max_blocks_, std::max<int32_t>(vram_cap_ / 4, 32768));
        vram_bk_stride_bytes_ = ((static_cast<size_t>(vram_blocks_) * 128 * sizeof(uint16_t)) + gran - 1) & ~(gran - 1);
    }
    const int32_t init_vram_seq = kv_grow_ ? std::min<int32_t>(vram_cap_, 6553) : vram_cap_;
    const size_t init_kv_map = ((static_cast<size_t>(init_vram_seq) * 2 * 160) + gran - 1) & ~(gran - 1);
    const int32_t init_bk_blks = kv_grow_ ? std::min<int32_t>(vram_blocks_, (init_vram_seq + 3) / 4) : vram_blocks_;
    const size_t init_bk_map = ((static_cast<size_t>(init_bk_blks) * 128 * sizeof(uint16_t)) + gran - 1) & ~(gran - 1);

    auto no_take = []() -> strata::core::VmmChunk { return 0; };
    if (kv_vmm_ok_) {
        uint8_t* vbase = kv_vmm_range_.base();
        for (int q = 0; q < kNumQsaCaches; ++q) {
            size_t k_off = static_cast<size_t>(q) * 2 * vram_kv_stride_bytes_;
            size_t v_off = k_off + vram_kv_stride_bytes_;
            size_t b_off = 2ULL * static_cast<size_t>(kNumQsaCaches) * vram_kv_stride_bytes_ +
                           static_cast<size_t>(q) * vram_bk_stride_bytes_;
            d_k_cache_[q] = vbase + k_off;
            d_v_cache_[q] = vbase + v_off;
            d_block_k_cache_[q] = vbase + b_off;
            int64_t k_lo = static_cast<int64_t>(k_off / gran);
            int64_t k_hi = static_cast<int64_t>((k_off + init_kv_map) / gran);
            int64_t v_lo = static_cast<int64_t>(v_off / gran);
            int64_t v_hi = static_cast<int64_t>((v_off + init_kv_map) / gran);
            int64_t b_lo = static_cast<int64_t>(b_off / gran);
            int64_t b_hi = static_cast<int64_t>((b_off + init_bk_map) / gran);
            if (!kv_vmm_range_.map_range(k_lo, k_hi, no_take) ||
                !kv_vmm_range_.map_range(v_lo, v_hi, no_take) ||
                (init_bk_map > 0 && !kv_vmm_range_.map_range(b_lo, b_hi, no_take))) {
                std::fprintf(stderr, "[gfx1201-nvfp4] VmmRange initial KV map failed\n");
                std::exit(1);
            }
            HIP_CHECK(hipMemset(d_k_cache_[q], 0, init_kv_map));
            HIP_CHECK(hipMemset(d_v_cache_[q], 0, init_kv_map));
            if (init_bk_map > 0) HIP_CHECK(hipMemset(d_block_k_cache_[q], 0, init_bk_map));
        }
        vram_kv_mapped_bytes_ = init_kv_map;
        vram_bk_mapped_bytes_ = init_bk_map;
        kv_mapped_seq_ = static_cast<int32_t>(std::min<size_t>(vram_prefix_, (init_kv_map / 320)));
    } else {
        for (int q = 0; q < kNumQsaCaches; ++q) {
            d_k_cache_[q] = gpu_alloc(vram_kv_stride_bytes_, true);
            d_v_cache_[q] = gpu_alloc(vram_kv_stride_bytes_, true);
            if (vram_bk_stride_bytes_ > 0) {
                d_block_k_cache_[q] = gpu_alloc(vram_bk_stride_bytes_, true);
            }
        }
        vram_kv_mapped_bytes_ = vram_kv_stride_bytes_;
        vram_bk_mapped_bytes_ = vram_bk_stride_bytes_;
        kv_mapped_seq_ = vram_prefix_;
    }

    const int32_t host_overflow_toks = std::max<int32_t>(0, max_seq_ - vram_prefix_);
    const int32_t host_overflow_blks = std::max<int32_t>(0, max_blocks_ - vram_blocks_);
    host_kv_stride_bytes_ = (host_overflow_toks > 0)
                                ? (((static_cast<size_t>(host_overflow_toks) * 2 * 160) + gran - 1) & ~(gran - 1))
                                : 0;
    host_bk_stride_bytes_ = (host_overflow_blks > 0)
                                ? (((static_cast<size_t>(host_overflow_blks) * 128 * sizeof(uint16_t)) + gran - 1) & ~(gran - 1))
                                : 0;

    if (host_kv_stride_bytes_ > 0) {
        host_kv_mmap_bytes_ = 2ULL * static_cast<size_t>(kNumQsaCaches) * host_kv_stride_bytes_;
        void* p = ::mmap(nullptr, host_kv_mmap_bytes_, PROT_READ | PROT_WRITE,
                         MAP_PRIVATE | MAP_ANONYMOUS | MAP_NORESERVE, -1, 0);
        if (p == MAP_FAILED) {
            std::fprintf(stderr, "[gfx1201-nvfp4] mmap failed for elastic host KV overflow range\n");
            std::exit(1);
        }
        host_kv_mmap_ = static_cast<uint8_t*>(p);
        for (int q = 0; q < kNumQsaCaches; ++q) {
            uint8_t* hk = host_kv_mmap_ + static_cast<size_t>(2 * q) * host_kv_stride_bytes_;
            uint8_t* hv = host_kv_mmap_ + static_cast<size_t>(2 * q + 1) * host_kv_stride_bytes_;
            kc_host_delta_[q] = reinterpret_cast<int64_t>(hk) - reinterpret_cast<int64_t>(d_k_cache_[q]);
            vc_host_delta_[q] = reinterpret_cast<int64_t>(hv) - reinterpret_cast<int64_t>(d_v_cache_[q]);
        }
    }
    if (host_bk_stride_bytes_ > 0) {
        host_bk_mmap_bytes_ = static_cast<size_t>(kNumQsaCaches) * host_bk_stride_bytes_;
        void* p = ::mmap(nullptr, host_bk_mmap_bytes_, PROT_READ | PROT_WRITE,
                         MAP_PRIVATE | MAP_ANONYMOUS | MAP_NORESERVE, -1, 0);
        if (p == MAP_FAILED) {
            std::fprintf(stderr, "[gfx1201-nvfp4] mmap failed for elastic host block-key range\n");
            std::exit(1);
        }
        host_bk_mmap_ = static_cast<uint8_t*>(p);
        for (int q = 0; q < kNumQsaCaches; ++q) {
            uint8_t* hb = host_bk_mmap_ + static_cast<size_t>(q) * host_bk_stride_bytes_;
            if (!d_block_k_cache_[q]) d_block_k_cache_[q] = hb;
            bk_host_delta_elems_[q] = (reinterpret_cast<int64_t>(hb) - reinterpret_cast<int64_t>(d_block_k_cache_[q])) / 2;
        }
    }

    d_sh_gu_ = gpu_alloc(B * 1856 * sizeof(uint16_t), true);
    d_router_logits_ = static_cast<uint16_t*>(d_sh_gu_) + 1280;
    d_sh_scale_ = static_cast<uint16_t*>(d_sh_gu_) + 1792;

    // Phase 4: MoE
    {
        size_t off = 0;
        d_sh_inter_ = ph_base + off;   off += align256(B * 640 * sizeof(uint16_t));
        d_sh_out_ = ph_base + off;     off += align256(B * 2560 * sizeof(uint16_t));
        d_moe_wts_ = ph_base + off;    off += align256(B * 16 * sizeof(float));
        d_moe_sorted_ = ph_base + off; off += align256(max_moe_tiles * 16 * sizeof(int32_t));
        d_moe_tile_e_ = ph_base + off; off += align256(max_moe_tiles * sizeof(int32_t));
        d_moe_inter_ = ph_base + off;  off += align256(max_moe_tiles * 16 * 640 * sizeof(uint16_t));
        d_moe_y_ = ph_base + off;      off += align256(B * 16 * 2560 * sizeof(uint16_t));
    }

    // Phase 5: MTP pre-FC
    {
        size_t off = 0;
        d_mtp_emb_norm_ = ph_base + off; off += align256(B * 2560 * sizeof(uint16_t));
        d_mtp_e2_ = ph_base + off;       off += align256(B * 2560 * sizeof(uint16_t));
        d_mtp_h_norm_ = ph_base + off;   off += align256(B * 10240 * sizeof(uint16_t));
        d_mtp_h2_ = ph_base + off;       off += align256(B * 10240 * sizeof(uint16_t));
    }

    d_mtp_r_ring_ = gpu_alloc(static_cast<size_t>(kMtpRingCap) * 10240 * sizeof(uint16_t), true);
    d_mtp_r_state_ = gpu_alloc(10240 * sizeof(uint16_t), true);
    ensure_verify_capacity(32);
    d_moe_gpu_align_ids_ = gpu_alloc(B * 10 * sizeof(int32_t));
    d_mtp_dvocab_ = static_cast<int32_t*>(gpu_alloc(65536 * sizeof(int32_t), true));
    ple_cache_.resize(kPleCacheSlots);
}

void Qwen4ExpGfx1201Engine::ensure_verify_capacity(int32_t needed_m) {
    if (needed_m <= verify_cap_) return;
    auto free_tracked = [&](void* ptr) {
        if (!ptr) return;
        (void) hipFree(ptr);
        auto it = std::find(vram_allocs_.begin(), vram_allocs_.end(), ptr);
        if (it != vram_allocs_.end()) vram_allocs_.erase(it);
    };
    free_tracked(d_gdn_verify_arena_);
    free_tracked(d_argmax_summary_);
    int32_t new_cap = std::max<int32_t>(32, (needed_m + 15) & ~15);
    const size_t gdn_verify_layer_bytes = static_cast<size_t>(new_cap) * 16480 * sizeof(uint16_t);
    d_gdn_verify_arena_ = gpu_alloc(36ULL * gdn_verify_layer_bytes);
    uint8_t* ver_cur = static_cast<uint8_t*>(d_gdn_verify_arena_);
    for (int l = 0; l < 48; ++l) {
        if (l % 4 != 3) {
            d_gdn_verify_qkv_[l] = ver_cur;
            ver_cur += gdn_verify_layer_bytes;
        }
    }
    d_argmax_summary_ = gpu_alloc(static_cast<size_t>(new_cap) * sizeof(strata::kernels::Nvfp4ArgmaxSummary));
    verify_summaries_.resize(static_cast<size_t>(new_cap));
    verify_cap_ = new_cap;
}

void Qwen4ExpGfx1201Engine::populate_rope_range(int32_t p_begin, int32_t p_end) {
    if (p_end <= p_begin) return;
    double inv[32];
    for (int i = 0; i < 32; ++i) {
        inv[i] = std::pow(rope_scaling_.freq_base, -static_cast<double>(2 * i) / 64.0);
    }
    float* cf32 = static_cast<float*>(d_rope_cos_f32_);
    float* sf32 = static_cast<float*>(d_rope_sin_f32_);
    uint16_t* cbf16 = static_cast<uint16_t*>(d_rope_cos_bf16_);
    uint16_t* sbf16 = static_cast<uint16_t*>(d_rope_sin_bf16_);

    const strata::kernels::RopeKernelArgs static_ka = rope_scaling_.kernel_args(64);
    const int32_t orig_ctx = static_cast<int32_t>(rope_scaling_.orig_ctx);

    for (int32_t p = p_begin; p < p_end; ++p) {
        strata::kernels::RopeKernelArgs ka = static_ka;
        bool unscaled = (rope_scaling_.type == strata::kernels::RopeScalingType::None);
        if (rope_auto_yarn_) {
            if (p < orig_ctx) {
                unscaled = true;
            } else {
                unscaled = false;
                strata::kernels::RopeScaling dyn_rs = rope_scaling_;
                dyn_rs.type = strata::kernels::RopeScalingType::YaRN;
                dyn_rs.factor = std::max(1.0, static_cast<double>(p + 1) / static_cast<double>(orig_ctx));
                dyn_rs.ext_factor = 1.0;
                ka = dyn_rs.kernel_args(64);
            }
        }
        for (int i = 0; i < 32; ++i) {
            float c = 1.0f, s = 0.0f;
            if (unscaled) {
                double ang = static_cast<double>(p) * inv[i];
                c = static_cast<float>(std::cos(ang));
                s = static_cast<float>(std::sin(ang));
            } else {
                float theta_extrap = static_cast<float>(static_cast<double>(p) * inv[i]);
                strata::kernels::rope_scaled_angle(theta_extrap, ka.freq_scale, ka.corr_low, ka.corr_high,
                                                   ka.ext_factor, ka.attn_factor, i, c, s);
            }
            cf32[static_cast<size_t>(p) * 32 + i] = c;
            sf32[static_cast<size_t>(p) * 32 + i] = s;
            uint16_t cb = f32_to_bf16_bits(c);
            uint16_t sb = f32_to_bf16_bits(s);
            cbf16[static_cast<size_t>(p) * 64 + i] = cb;
            cbf16[static_cast<size_t>(p) * 64 + 32 + i] = cb;
            sbf16[static_cast<size_t>(p) * 64 + i] = sb;
            sbf16[static_cast<size_t>(p) * 64 + 32 + i] = sb;
        }
    }
}

void Qwen4ExpGfx1201Engine::init_rope_tables() {
    const size_t gran = strata::core::vmm_available()
                            ? static_cast<size_t>(strata::core::vmm_granularity())
                            : (2ULL * 1024 * 1024);
    rope_stride_f32_bytes_ = ((static_cast<size_t>(max_seq_) * 32 * sizeof(float)) + gran - 1) & ~(gran - 1);
    rope_stride_bf16_bytes_ = ((static_cast<size_t>(max_seq_) * 64 * sizeof(uint16_t)) + gran - 1) & ~(gran - 1);
    host_rope_mmap_bytes_ = 2 * rope_stride_f32_bytes_ + 2 * rope_stride_bf16_bytes_;

    void* p = ::mmap(nullptr, host_rope_mmap_bytes_, PROT_READ | PROT_WRITE,
                     MAP_PRIVATE | MAP_ANONYMOUS | MAP_NORESERVE, -1, 0);
    if (p == MAP_FAILED) {
        std::fprintf(stderr, "[gfx1201-nvfp4] mmap failed for elastic RoPE tables\n");
        std::exit(1);
    }
    host_rope_mmap_ = static_cast<uint8_t*>(p);
    d_rope_cos_f32_ = host_rope_mmap_;
    d_rope_sin_f32_ = host_rope_mmap_ + rope_stride_f32_bytes_;
    d_rope_cos_bf16_ = host_rope_mmap_ + 2 * rope_stride_f32_bytes_;
    d_rope_sin_bf16_ = host_rope_mmap_ + 2 * rope_stride_f32_bytes_ + rope_stride_bf16_bytes_;

    rope_populated_seq_ = 0;
    rope_mapped_bytes_ = 0;
    ensure_rope_capacity(kv_mapped_seq_);
}

void Qwen4ExpGfx1201Engine::ensure_rope_capacity(int32_t needed_seq) {
    int32_t target_seq = std::min<int32_t>(max_seq_, std::max<int32_t>(needed_seq, 16384));
    const size_t gran = strata::core::vmm_available()
                            ? static_cast<size_t>(strata::core::vmm_granularity())
                            : (2ULL * 1024 * 1024);
    size_t need_bytes = ((static_cast<size_t>(target_seq) * 32 * sizeof(float)) + gran - 1) & ~(gran - 1);
    need_bytes = std::min(need_bytes, rope_stride_f32_bytes_);
    if (need_bytes > rope_mapped_bytes_) {
        size_t add_bytes = need_bytes - rope_mapped_bytes_;
        uint8_t* ptrs[4] = {
            static_cast<uint8_t*>(d_rope_cos_f32_) + rope_mapped_bytes_,
            static_cast<uint8_t*>(d_rope_sin_f32_) + rope_mapped_bytes_,
            static_cast<uint8_t*>(d_rope_cos_bf16_) + rope_mapped_bytes_,
            static_cast<uint8_t*>(d_rope_sin_bf16_) + rope_mapped_bytes_,
        };
        for (int k = 0; k < 4; ++k) {
            (void) ::madvise(ptrs[k], add_bytes, MADV_HUGEPAGE);
            std::memset(ptrs[k], 0, add_bytes);
            HIP_CHECK(hipHostRegister(ptrs[k], add_bytes, hipHostRegisterMapped));
            host_slab_registered_.push_back(ptrs[k]);
        }
        rope_mapped_bytes_ = need_bytes;
    }
    int32_t max_pop = static_cast<int32_t>(std::min<size_t>(max_seq_, rope_mapped_bytes_ / (32 * sizeof(float))));
    if (rope_populated_seq_ < max_pop) {
        populate_rope_range(rope_populated_seq_, max_pop);
        rope_populated_seq_ = max_pop;
    }
}

strata::core::VmmChunk Qwen4ExpGfx1201Engine::take_one_dynamic_expert_chunk() {
    if (!exp_vmm_ok_ || exp_vmm_mapped_bytes_ == 0) return 0;
    const size_t gran = strata::core::vmm_available()
                            ? static_cast<size_t>(strata::core::vmm_granularity())
                            : (2ULL * 1024 * 1024);
    if (exp_vmm_mapped_bytes_ < gran) return 0;
    const size_t new_mapped_bytes = exp_vmm_mapped_bytes_ - gran;
    const size_t max_valid_slots = new_mapped_bytes / EXP_STRIDE_BYTES;

    for (int l = 0; l < 48; ++l) {
        Nvfp4MoeWeights& mw = layers_[l].moe;
        bool dirty = false;
        for (int s = mw.n_dyn - 1; s >= 0; --s) {
            size_t g_slot = exp_layer_dyn_slot_off_[l] + static_cast<size_t>(s);
            if (g_slot >= max_valid_slots) {
                int32_t old_e = mw.slot_exp[s];
                if (old_e >= 0 && old_e < 512) {
                    for (int k = 0; k < 6; ++k) mw.h_ptrs_prefill[old_e * 6 + k] = 0;
                    if (mw.in_gpu_cold[old_e]) {
                        uint8_t* old_hdev = mw.host_dev_ptr[old_e];
                        mw.h_ptrs[old_e * 6 + 0] = reinterpret_cast<int64_t>(old_hdev + EXP_OFF_GATE_W);
                        mw.h_ptrs[old_e * 6 + 1] = reinterpret_cast<int64_t>(old_hdev + EXP_OFF_GATE_S);
                        mw.h_ptrs[old_e * 6 + 2] = reinterpret_cast<int64_t>(old_hdev + EXP_OFF_UP_W);
                        mw.h_ptrs[old_e * 6 + 3] = reinterpret_cast<int64_t>(old_hdev + EXP_OFF_UP_S);
                        mw.h_ptrs[old_e * 6 + 4] = reinterpret_cast<int64_t>(old_hdev + EXP_OFF_DOWN_W);
                        mw.h_ptrs[old_e * 6 + 5] = reinterpret_cast<int64_t>(old_hdev + EXP_OFF_DOWN_S);
                    } else {
                        for (int k = 0; k < 6; ++k) mw.h_ptrs[old_e * 6 + k] = 0;
                    }
                    mw.in_vram[old_e] = false;
                    mw.dyn_slot[old_e] = -1;
                    mw.slot_exp[s] = -1;
                    total_vram_experts_--;
                    dirty = true;
                }
                mw.n_dyn = s;
                mw.vram_experts = mw.n_pinned + mw.n_dyn;
            }
        }
        if (dirty) {
            HIP_CHECK(hipMemcpy(mw.d_ptrs, mw.h_ptrs, 512 * 6 * sizeof(int64_t),
                                hipMemcpyHostToDevice));
            HIP_CHECK(hipMemcpy(mw.d_ptrs_prefill, mw.h_ptrs_prefill,
                                512 * 6 * sizeof(int64_t), hipMemcpyHostToDevice));
        }
    }
    HIP_CHECK(hipDeviceSynchronize());
    int64_t chunk_idx = static_cast<int64_t>(new_mapped_bytes / gran);
    strata::core::VmmChunk h = exp_vmm_range_.unmap(chunk_idx);
    if (h == 0) return 0;
    exp_vmm_mapped_bytes_ = new_mapped_bytes;
    return h;
}

bool Qwen4ExpGfx1201Engine::map_kv_vmm_with_reclaim(size_t offset, size_t length) {
    const size_t gran = strata::core::vmm_available()
                            ? static_cast<size_t>(strata::core::vmm_granularity())
                            : (2ULL * 1024 * 1024);
    int64_t lo = static_cast<int64_t>(offset / gran);
    int64_t hi = static_cast<int64_t>((offset + length) / gran);
    auto take_fn = [&]() -> strata::core::VmmChunk {
        strata::core::VmmChunk h = take_one_dynamic_expert_chunk();
        if (h != 0) return h;
        size_t free_vram = 0, total_vram = 0;
        const size_t min_free = static_cast<size_t>(std::max(512, vram_reserve_mib_)) * 1024ULL * 1024ULL;
        if (hipMemGetInfo(&free_vram, &total_vram) == hipSuccess &&
            free_vram >= gran + min_free) {
            return 0;
        }
        return 0;
    };
    return kv_vmm_range_.map_range(lo, hi, take_fn);
}

void Qwen4ExpGfx1201Engine::ensure_context_capacity(int32_t needed_seq) {
    if (needed_seq > max_seq_) {
        std::fprintf(stderr, "[gfx1201-nvfp4] Context limit reached (%d > max_seq %d)\n", needed_seq, max_seq_);
        std::exit(1);
    }
    ensure_rope_capacity(needed_seq);
    if (needed_seq <= kv_mapped_seq_) return;

    const size_t gran = strata::core::vmm_available()
                            ? static_cast<size_t>(strata::core::vmm_granularity())
                            : (2ULL * 1024 * 1024);
    if (kv_vmm_ok_ && vram_kv_mapped_bytes_ < vram_kv_stride_bytes_) {
        int32_t want_vram_seq = (needed_seq > vram_prefix_) ? vram_cap_ : std::min<int32_t>(vram_cap_, needed_seq);
        size_t want_kv_bytes = ((static_cast<size_t>(want_vram_seq) * 2 * 160) + gran - 1) & ~(gran - 1);
        want_kv_bytes = std::min(want_kv_bytes, vram_kv_stride_bytes_);
        if (want_kv_bytes > vram_kv_mapped_bytes_) {
            size_t add_kv = want_kv_bytes - vram_kv_mapped_bytes_;
            uint8_t* vbase = kv_vmm_range_.base();
            for (int q = 0; q < kNumQsaCaches; ++q) {
                size_t k_off = static_cast<size_t>(q) * 2 * vram_kv_stride_bytes_ + vram_kv_mapped_bytes_;
                size_t v_off = static_cast<size_t>(q) * 2 * vram_kv_stride_bytes_ + vram_kv_stride_bytes_ + vram_kv_mapped_bytes_;
                if (!map_kv_vmm_with_reclaim(k_off, add_kv) || !map_kv_vmm_with_reclaim(v_off, add_kv)) {
                    std::fprintf(stderr, "[gfx1201-nvfp4] Failed to grow VRAM KV VmmRange to %zu bytes\n", want_kv_bytes);
                    std::exit(1);
                }
                HIP_CHECK(hipMemset(vbase + k_off, 0, add_kv));
                HIP_CHECK(hipMemset(vbase + v_off, 0, add_kv));
            }
            vram_kv_mapped_bytes_ = want_kv_bytes;
        }
    }
    if (kv_vmm_ok_ && vram_bk_mapped_bytes_ < vram_bk_stride_bytes_) {
        int32_t want_vram_blks = std::min<int32_t>(vram_blocks_, (needed_seq + 3) / 4);
        size_t want_bk_bytes = ((static_cast<size_t>(want_vram_blks) * 128 * sizeof(uint16_t)) + gran - 1) & ~(gran - 1);
        want_bk_bytes = std::min(want_bk_bytes, vram_bk_stride_bytes_);
        if (want_bk_bytes > vram_bk_mapped_bytes_) {
            size_t add_bk = want_bk_bytes - vram_bk_mapped_bytes_;
            uint8_t* vbase = kv_vmm_range_.base();
            for (int q = 0; q < kNumQsaCaches; ++q) {
                size_t b_off = 2ULL * static_cast<size_t>(kNumQsaCaches) * vram_kv_stride_bytes_ +
                               static_cast<size_t>(q) * vram_bk_stride_bytes_ + vram_bk_mapped_bytes_;
                if (!map_kv_vmm_with_reclaim(b_off, add_bk)) {
                    std::fprintf(stderr, "[gfx1201-nvfp4] Failed to grow VRAM Block-K VmmRange to %zu bytes\n", want_bk_bytes);
                    std::exit(1);
                }
                HIP_CHECK(hipMemset(vbase + b_off, 0, add_bk));
            }
            vram_bk_mapped_bytes_ = want_bk_bytes;
        }
    }

    if (needed_seq > vram_prefix_ && host_kv_mmap_) {
        int32_t over_toks = needed_seq - vram_prefix_;
        size_t want_hkv_bytes = ((static_cast<size_t>(over_toks) * 2 * 160) + gran - 1) & ~(gran - 1);
        want_hkv_bytes = std::min(want_hkv_bytes, host_kv_stride_bytes_);
        if (want_hkv_bytes > host_kv_mapped_bytes_) {
            size_t add_hkv = want_hkv_bytes - host_kv_mapped_bytes_;
            for (int q = 0; q < kNumQsaCaches; ++q) {
                uint8_t* hk = host_kv_mmap_ + static_cast<size_t>(2 * q) * host_kv_stride_bytes_ + host_kv_mapped_bytes_;
                uint8_t* hv = host_kv_mmap_ + static_cast<size_t>(2 * q + 1) * host_kv_stride_bytes_ + host_kv_mapped_bytes_;
                (void) ::madvise(hk, add_hkv, MADV_HUGEPAGE);
                (void) ::madvise(hv, add_hkv, MADV_HUGEPAGE);
                std::memset(hk, 0, add_hkv);
                std::memset(hv, 0, add_hkv);
                HIP_CHECK(hipHostRegister(hk, add_hkv, hipHostRegisterMapped));
                HIP_CHECK(hipHostRegister(hv, add_hkv, hipHostRegisterMapped));
                host_slab_registered_.push_back(hk);
                host_slab_registered_.push_back(hv);
            }
            host_kv_mapped_bytes_ = want_hkv_bytes;
        }
    }
    if (needed_seq > vram_blocks_ * 4 && host_bk_mmap_ && host_bk_stride_bytes_ > 0) {
        int32_t over_blks = ((needed_seq + 3) / 4) - vram_blocks_;
        size_t want_hbk_bytes = ((static_cast<size_t>(over_blks) * 128 * sizeof(uint16_t)) + gran - 1) & ~(gran - 1);
        want_hbk_bytes = std::min(want_hbk_bytes, host_bk_stride_bytes_);
        if (want_hbk_bytes > host_bk_mapped_bytes_) {
            size_t add_hbk = want_hbk_bytes - host_bk_mapped_bytes_;
            for (int q = 0; q < kNumQsaCaches; ++q) {
                uint8_t* hb = host_bk_mmap_ + static_cast<size_t>(q) * host_bk_stride_bytes_ + host_bk_mapped_bytes_;
                (void) ::madvise(hb, add_hbk, MADV_HUGEPAGE);
                std::memset(hb, 0, add_hbk);
                HIP_CHECK(hipHostRegister(hb, add_hbk, hipHostRegisterMapped));
                host_slab_registered_.push_back(hb);
            }
            host_bk_mapped_bytes_ = want_hbk_bytes;
        }
    }

    int32_t vram_cov = static_cast<int32_t>(std::min<size_t>(vram_prefix_, vram_kv_mapped_bytes_ / 320));
    int32_t host_cov = static_cast<int32_t>(host_kv_mapped_bytes_ / 320);
    int32_t bk_cov = (static_cast<int32_t>(vram_bk_mapped_bytes_ / 256) + static_cast<int32_t>(host_bk_mapped_bytes_ / 256)) * 4;
    kv_mapped_seq_ = std::min(bk_cov, (host_cov > 0) ? (vram_prefix_ + host_cov) : vram_cov);
}

void Qwen4ExpGfx1201Engine::load_weights_from_safetensors() {
    ple_consts_ = strata::kernels::ple_artifact_consts();

    std::vector<std::string> shard_paths;
    std::error_code ec;
    for (const auto& entry : std::filesystem::directory_iterator(model_dir_, ec)) {
        if (!entry.is_regular_file(ec)) continue;
        const auto p = entry.path();
        if (p.extension() == ".safetensors") {
            shard_paths.push_back(p.string());
        }
    }
    std::sort(shard_paths.begin(), shard_paths.end());
    if (shard_paths.empty()) {
        std::fprintf(stderr, "No .safetensors shards found in %s\n", model_dir_.c_str());
        std::exit(1);
    }

    std::unordered_map<std::string, TensorSlice> main_index;
    std::vector<int> shard_fds(shard_paths.size(), -1);
    for (size_t s = 0; s < shard_paths.size(); ++s) {
        shard_fds[s] = ::open(shard_paths[s].c_str(), O_RDONLY);
        if (shard_fds[s] < 0) {
            std::fprintf(stderr, "Cannot open shard: %s\n", shard_paths[s].c_str());
            std::exit(1);
        }
        parse_safetensors_header(shard_fds[s], static_cast<int>(s), main_index);
        (void) ::posix_fadvise(shard_fds[s], 0, 0, POSIX_FADV_DONTNEED);
    }

    int ple_shard_idx = -1;
    for (int s = 0; s < 128; ++s) {
        std::string tname = "model.language_model.layers.1.ple.ple_embedding.ngram_embedding.shard_" +
                            std::to_string(s) + ".weight";
        auto it = main_index.find(tname);
        if (it == main_index.end()) {
            std::fprintf(stderr, "Missing PLE shard tensor: %s\n", tname.c_str());
            std::exit(1);
        }
        ple_.shard_offsets[s] = it->second.file_offset;
        ple_shard_idx = it->second.shard_idx;
    }
    ple_.ple_fd = shard_fds[ple_shard_idx];
    {
        auto it = main_index.find("model.language_model.layers.1.ple.ple_embedding.ngram_embedding.weight_scale");
        if (it == main_index.end()) {
            std::fprintf(stderr, "Missing PLE weight_scale\n");
            std::exit(1);
        }
        uint16_t sc_bf16 = 0;
        pread_exact(shard_fds[it->second.shard_idx], &sc_bf16, sizeof(uint16_t), it->second.file_offset);
        ple_.weight_scale = gpu_alloc(sizeof(uint16_t));
        HIP_CHECK(hipMemcpy(ple_.weight_scale, &sc_bf16, sizeof(uint16_t), hipMemcpyHostToDevice));
    }

    std::vector<uint8_t> stage_buf(16 * 1024 * 1024);
    auto load_vram_tensor = [&](const std::string& name) -> void* {
        auto it = main_index.find(name);
        if (it == main_index.end()) {
            std::fprintf(stderr, "Missing tensor in safetensors: %s\n", name.c_str());
            std::exit(1);
        }
        const TensorSlice& ts = it->second;
        void* d_ptr = gpu_alloc(ts.byte_len);
        if (stage_buf.size() < ts.byte_len) stage_buf.resize(ts.byte_len);
        pread_exact(shard_fds[ts.shard_idx], stage_buf.data(), ts.byte_len, ts.file_offset);
        HIP_CHECK(hipMemcpy(d_ptr, stage_buf.data(), ts.byte_len, hipMemcpyHostToDevice));
        return d_ptr;
    };

    auto load_vram_tensor_concat = [&](const std::vector<std::string>& names, size_t pad_bytes = 0) -> void* {
        size_t total_bytes = pad_bytes;
        for (const auto& name : names) {
            auto it = main_index.find(name);
            if (it == main_index.end()) {
                std::fprintf(stderr, "Missing tensor in safetensors: %s\n", name.c_str());
                std::exit(1);
            }
            total_bytes += it->second.byte_len;
        }
        void* d_ptr = gpu_alloc(total_bytes, pad_bytes > 0);
        size_t cur_off = 0;
        for (const auto& name : names) {
            const TensorSlice& ts = main_index.at(name);
            if (stage_buf.size() < ts.byte_len) stage_buf.resize(ts.byte_len);
            pread_exact(shard_fds[ts.shard_idx], stage_buf.data(), ts.byte_len, ts.file_offset);
            HIP_CHECK(hipMemcpy(static_cast<uint8_t*>(d_ptr) + cur_off, stage_buf.data(), ts.byte_len,
                                hipMemcpyHostToDevice));
            cur_off += ts.byte_len;
        }
        return d_ptr;
    };

    auto load_hc = [&](const std::string& pfx, bool combine) -> Nvfp4HcWeights {
        Nvfp4HcWeights hc;
        hc.norm = load_vram_tensor(pfx + ".hc_norm.weight");
        hc.down = load_vram_tensor(pfx + ".input_mix_weight_down.weight");
        hc.up = load_vram_tensor(pfx + ".input_mix_weight_up.weight");
        if (combine) {
            hc.inj = load_vram_tensor(pfx + ".block_inject_weight.weight");
            hc.wi_delta = (reinterpret_cast<int64_t>(hc.inj) - reinterpret_cast<int64_t>(hc.down)) / 2;
        }
        return hc;
    };

    d_embed_tokens_ = load_vram_tensor("model.language_model.embed_tokens.weight");
    d_lm_head_ = load_vram_tensor("lm_head.weight");
    final_hc_ = load_hc("model.language_model.hyper_connection_mixer", false);

    ple_.key_proj = load_vram_tensor_concat({
        "model.language_model.layers.1.ple.key_proj.weight",
        "model.language_model.layers.1.ple.value_proj.weight",
    });
    ple_.value_proj = static_cast<uint16_t*>(ple_.key_proj) + 10240ULL * 2560;
    ple_.norm_key = load_vram_tensor("model.language_model.layers.1.ple.norm_key.weight");
    ple_.norm_query = load_vram_tensor("model.language_model.layers.1.ple.norm_query.weight");
    ple_.norm_conv = load_vram_tensor("model.language_model.layers.1.ple.norm_conv.weight");
    ple_.conv_weight = load_vram_tensor("model.language_model.layers.1.ple.conv1d.weight");

    for (int l = 0; l < 48; ++l) {
        Nvfp4LayerWeights& lw = layers_[l];
        lw.is_qsa = (l % 4 == 3);
        std::string lpfx = "model.language_model.layers." + std::to_string(l);
        lw.attn_hc = load_hc(lpfx + ".attn_hyper_connection", true);
        lw.mlp_hc = load_hc(lpfx + ".mlp_hyper_connection", true);

        if (!lw.is_qsa) {
            std::string apfx = lpfx + ".linear_attn";
            lw.gdn.in_proj_fused = load_vram_tensor_concat({
                apfx + ".in_proj_qkv.weight",
                apfx + ".in_proj_z.weight",
                apfx + ".in_proj_a.weight",
                apfx + ".in_proj_b.weight",
            });
            lw.gdn.out_proj = load_vram_tensor(apfx + ".out_proj.weight");
            lw.gdn.conv1d = load_vram_tensor(apfx + ".conv1d.weight");
            lw.gdn.A_log = load_vram_tensor(apfx + ".A_log");
            lw.gdn.dt_bias = load_vram_tensor(apfx + ".dt_bias");
            lw.gdn.norm = load_vram_tensor(apfx + ".norm.weight");
        } else {
            std::string apfx = lpfx + ".self_attn";
            lw.qsa.in_proj_fused = load_vram_tensor_concat({
                apfx + ".q_proj.weight",
                apfx + ".k_proj.weight",
                apfx + ".v_proj.weight",
                apfx + ".indexer.index_qk_proj.weight",
            });
            lw.qsa.o_proj = load_vram_tensor(apfx + ".o_proj.weight");
            lw.qsa.q_norm = load_vram_tensor(apfx + ".q_norm.weight");
            lw.qsa.k_norm = load_vram_tensor(apfx + ".k_norm.weight");
            lw.qsa.idx_q_ln = load_vram_tensor(apfx + ".indexer.q_layernorm.weight");
            lw.qsa.idx_k_ln = load_vram_tensor(apfx + ".indexer.k_layernorm.weight");
        }

        std::string mpfx = lpfx + ".mlp";
        lw.moe.in_proj_fused = load_vram_tensor_concat(
            {
                mpfx + ".shared_expert.gate_proj.weight",
                mpfx + ".shared_expert.up_proj.weight",
                mpfx + ".gate.weight",
                mpfx + ".shared_expert_gate.weight",
            },
            63ULL * 2560 * sizeof(uint16_t));
        lw.moe.sh_down = load_vram_tensor(mpfx + ".shared_expert.down_proj.weight");
    }

    const bool want_mtp = (std::getenv("STRATA_DISABLE_MTP") == nullptr) &&
                          (main_index.count("mtp.fc_embedding.weight") > 0) &&
                          (main_index.count("mtp.layers.0.mlp.experts.0.gate_proj.weight") > 0);
    if (const char* env_hn = std::getenv("STRATA_MTP_HNORM_STREAM")) {
        mtp_hnorm_per_stream_ = (std::atoi(env_hn) != 0);
    }
    if (want_mtp) {
        mtp_.pre_fc_norm_embedding = load_vram_tensor("mtp.pre_fc_norm_embedding.weight");
        mtp_.pre_fc_norm_hidden = load_vram_tensor("mtp.pre_fc_norm_hidden.weight");
        mtp_.fc_embedding = load_vram_tensor("mtp.fc_embedding.weight");
        mtp_.fc_hidden = load_vram_tensor("mtp.fc_hidden.weight");
        mtp_.attn_hc = load_hc("mtp.layers.0.attn_hyper_connection", true);
        std::string apfx = "mtp.layers.0.self_attn";
        mtp_.qsa.in_proj_fused = load_vram_tensor_concat({
            apfx + ".q_proj.weight",
            apfx + ".k_proj.weight",
            apfx + ".v_proj.weight",
            apfx + ".indexer.index_qk_proj.weight",
        });
        mtp_.qsa.o_proj = load_vram_tensor(apfx + ".o_proj.weight");
        mtp_.qsa.q_norm = load_vram_tensor(apfx + ".q_norm.weight");
        mtp_.qsa.k_norm = load_vram_tensor(apfx + ".k_norm.weight");
        mtp_.qsa.idx_q_ln = load_vram_tensor(apfx + ".indexer.q_layernorm.weight");
        mtp_.qsa.idx_k_ln = load_vram_tensor(apfx + ".indexer.k_layernorm.weight");
        mtp_.mlp_hc = load_hc("mtp.layers.0.mlp_hyper_connection", true);
        std::string mpfx = "mtp.layers.0.mlp";
        mtp_.moe.in_proj_fused = load_vram_tensor_concat(
            {
                mpfx + ".shared_expert.gate_proj.weight",
                mpfx + ".shared_expert.up_proj.weight",
                mpfx + ".gate.weight",
                mpfx + ".shared_expert_gate.weight",
            },
            63ULL * 2560 * sizeof(uint16_t));
        mtp_.moe.sh_down = load_vram_tensor(mpfx + ".shared_expert.down_proj.weight");
        mtp_.final_hc = load_hc("mtp.hyper_connection_mixer", false);
    }

    for (int s = 0; s < 10; ++s) {
        (void) ::posix_fadvise(shard_fds[s], 0, 0, POSIX_FADV_DONTNEED);
    }

    int32_t vram_rank[48][512];
    for (int l = 0; l < 48; ++l) {
        for (int e = 0; e < 512; ++e) vram_rank[l][e] = -1;
    }
    int layer_vram_cnt[48]{};
    size_t free_vram = 0, total_vram = 0;
    HIP_CHECK(hipMemGetInfo(&free_vram, &total_vram));
    // Account for post-query per-layer pointer tables, VMM 2 MiB slab rounding, and DRM graphics headroom
    const size_t reserve_vram = static_cast<size_t>(vram_reserve_mib_ + 512) * 1024ULL * 1024ULL;
    int max_vram_experts = (free_vram > reserve_vram)
                               ? static_cast<int>((free_vram - reserve_vram) / EXP_STRIDE_BYTES)
                               : (vram_expert_layers_ * 512);
    if (vram_expert_layers_ > 0 && vram_expert_layers_ != 14) {
        max_vram_experts = std::min(max_vram_experts, vram_expert_layers_ * 512);
    }

    if (expert_profile_path_.empty()) {
        std::error_code ec;
        std::string cand_model = model_dir_ + "/expert-profile.bin";
        if (std::filesystem::exists(cand_model, ec)) {
            expert_profile_path_ = cand_model;
        } else if (std::filesystem::exists("data/expert-profile.bin", ec)) {
            expert_profile_path_ = "data/expert-profile.bin";
        } else {
            std::filesystem::path exe_p = std::filesystem::read_symlink("/proc/self/exe", ec);
            if (!ec) {
                std::filesystem::path p1 = exe_p.parent_path() / "data" / "expert-profile.bin";
                std::filesystem::path p2 = exe_p.parent_path().parent_path() / "data" / "expert-profile.bin";
                if (std::filesystem::exists(p1, ec)) {
                    expert_profile_path_ = p1.string();
                } else if (std::filesystem::exists(p2, ec)) {
                    expert_profile_path_ = p2.string();
                }
            }
        }
    }

    std::vector<std::pair<int32_t, int32_t>> ranked_pairs;
    int64_t prof_slots = 0;
    std::string prof_err;
    if (!expert_profile_path_.empty() &&
        strata::core::read_expert_profile(expert_profile_path_, 48, 512, ranked_pairs, prof_slots, prof_err)) {
        int placed = 0;
        double cold_acc = 0.0;
        for (const auto& [ly, ex] : ranked_pairs) {
            if (ly < 0 || ly >= 48 || ex < 0 || ex >= 512 || vram_rank[ly][ex] >= 0) continue;
            if (placed < max_vram_experts) {
                vram_rank[ly][ex] = layer_vram_cnt[ly]++;
                placed++;
            } else {
                cold_acc += static_cast<double>(pcie_frac_);
                if (cold_acc >= 1.0) {
                    cold_acc -= 1.0;
                    layers_[ly].moe.in_gpu_cold[ex] = true;
                }
            }
        }
        total_vram_experts_ = placed;
    } else {
        if (!expert_profile_path_.empty()) {
            std::fprintf(stderr, "[gfx1201-nvfp4] WARNING: failed to read expert profile '%s' (%s); using uniform per-layer placement\n",
                         expert_profile_path_.c_str(), prof_err.c_str());
        }
        int placed = 0;
        double cold_acc = 0.0;
        for (int e = 0; e < 512; ++e) {
            for (int l = 0; l < 48; ++l) {
                if (placed < max_vram_experts) {
                    vram_rank[l][e] = layer_vram_cnt[l]++;
                    placed++;
                } else {
                    cold_acc += static_cast<double>(pcie_frac_);
                    if (cold_acc >= 1.0) {
                        cold_acc -= 1.0;
                        layers_[l].moe.in_gpu_cold[e] = true;
                    }
                }
            }
        }
        total_vram_experts_ = placed;
    }

    size_t total_gpu_cold_experts = 0;
    size_t total_cpu_host_experts = 0;
    int max_layer_pinned = 0;
    int max_layer_vram = 0;
    size_t layer_gpu_cold_off[48]{};
    size_t layer_cpu_host_off[48]{};
    for (int l = 0; l < 48; ++l) {
        int n_vram = layer_vram_cnt[l];
        int n_dyn = std::min(32, std::max(0, n_vram / 4));
        int n_pinned = n_vram - n_dyn;
        layer_gpu_cold_off[l] = total_gpu_cold_experts;
        layer_cpu_host_off[l] = total_cpu_host_experts;
        for (int e = 0; e < 512; ++e) {
            int32_t rk = vram_rank[l][e];
            bool is_pinned = (rk >= 0 && rk < n_pinned);
            if (is_pinned) continue;
            if (layers_[l].moe.in_gpu_cold[e]) {
                total_gpu_cold_experts++;
            } else {
                total_cpu_host_experts++;
            }
        }
        max_layer_pinned = std::max(max_layer_pinned, n_pinned);
        max_layer_vram = std::max(max_layer_vram, n_vram);
    }

    const int n_io_threads = std::max(1, cpu_pool_.num_threads());
    auto prefault_hugepages = [&](uint8_t* base, size_t total_bytes) {
        if (!base || total_bytes == 0) return;
        const size_t step = 2 * 1024 * 1024;
        const size_t n_pages = (total_bytes + step - 1) / step;
        std::vector<std::thread> pf_workers;
        pf_workers.reserve(n_io_threads - 1);
        auto touch_range = [&](int tid) {
            if (tid > 0) {
                int c = cpu_pool_.core_for_thread(tid);
                if (c >= 0) strata::kernels::cpu::pin_current_thread(c);
            }
            size_t p0 = (static_cast<size_t>(tid) * n_pages) / static_cast<size_t>(n_io_threads);
            size_t p1 = (static_cast<size_t>(tid + 1) * n_pages) / static_cast<size_t>(n_io_threads);
            for (size_t p = p0; p < p1; ++p) {
                base[p * step] = 0;
            }
        };
        for (int t = 1; t < n_io_threads; ++t) {
            pf_workers.emplace_back(touch_range, t);
        }
        touch_range(0);
        for (auto& w : pf_workers) w.join();
    };

    uint8_t* gpu_cold_host_base = nullptr;
    const size_t total_gpu_cold_bytes = total_gpu_cold_experts * EXP_STRIDE_BYTES;
    if (total_gpu_cold_bytes > 0) {
        void* hptr = nullptr;
        if (::posix_memalign(&hptr, 2 * 1024 * 1024, total_gpu_cold_bytes) != 0 || !hptr) {
            std::fprintf(stderr, "posix_memalign failed for %zu MB GPU-cold host arena\n",
                         total_gpu_cold_bytes / (1024 * 1024));
            std::exit(1);
        }
        (void) ::madvise(hptr, total_gpu_cold_bytes, MADV_HUGEPAGE);
        gpu_cold_host_base = static_cast<uint8_t*>(hptr);
        prefault_hugepages(gpu_cold_host_base, total_gpu_cold_bytes);
    }

    uint8_t* cpu_host_base = nullptr;
    const size_t total_cpu_host_bytes = total_cpu_host_experts * EXP_STRIDE_BYTES;
    if (total_cpu_host_bytes > 0) {
        void* hptr = nullptr;
        if (::posix_memalign(&hptr, 2 * 1024 * 1024, total_cpu_host_bytes) != 0 || !hptr) {
            std::fprintf(stderr, "posix_memalign failed for %zu MB CPU-only host arena\n",
                         total_cpu_host_bytes / (1024 * 1024));
            std::exit(1);
        }
        (void) ::madvise(hptr, total_cpu_host_bytes, MADV_HUGEPAGE);
        cpu_host_base = static_cast<uint8_t*>(hptr);
        host_mallocs_.push_back(cpu_host_base);
        prefault_hugepages(cpu_host_base, total_cpu_host_bytes);
    }

    uint8_t* pin_vram_stage[2] = {nullptr, nullptr};
    hipStream_t upload_stream = nullptr;
    hipEvent_t ev_stage_done[2] = {nullptr, nullptr};
    if (max_layer_vram > 0) {
        HIP_CHECK(hipStreamCreateWithFlags(&upload_stream, hipStreamNonBlocking));
        for (int b = 0; b < 2; ++b) {
            HIP_CHECK(hipHostMalloc(reinterpret_cast<void**>(&pin_vram_stage[b]),
                                    static_cast<size_t>(max_layer_vram) * EXP_STRIDE_BYTES,
                                    hipHostMallocDefault));
            HIP_CHECK(hipEventCreateWithFlags(&ev_stage_done[b], hipEventDisableTiming));
        }
    }

    struct ExpSlice9 {
        TensorSlice gw, gs, gs2, uw, us, us2, dw, ds, ds2;
    };
    std::vector<ExpSlice9> exp_slices(512);
    int32_t gpu_cold_slot[48][512];

    size_t total_dyn_experts = 0;
    for (int l = 0; l < 48; ++l) {
        int n_vram = layer_vram_cnt[l];
        int n_dyn = std::min(32, std::max(0, n_vram / 4));
        exp_layer_dyn_slot_off_[l] = total_dyn_experts;
        total_dyn_experts += static_cast<size_t>(n_dyn);
    }
    const size_t gran = strata::core::vmm_available()
                            ? static_cast<size_t>(strata::core::vmm_granularity())
                            : (2ULL * 1024 * 1024);
    const size_t total_dyn_vmm_bytes = ((total_dyn_experts * EXP_STRIDE_BYTES) + gran - 1) & ~(gran - 1);
    exp_vmm_ok_ = false;
    exp_vmm_mapped_bytes_ = 0;
    if (total_dyn_vmm_bytes > 0 && strata::core::vmm_available() && exp_vmm_range_.reserve(total_dyn_vmm_bytes)) {
        int64_t n_chunks = static_cast<int64_t>(total_dyn_vmm_bytes / gran);
        if (exp_vmm_range_.map_range(0, n_chunks, []() -> strata::core::VmmChunk { return 0; })) {
            exp_vmm_ok_ = true;
            exp_vmm_mapped_bytes_ = total_dyn_vmm_bytes;
        } else {
            exp_vmm_range_.release();
        }
    }

    for (int l = 0; l < 48; ++l) {
        Nvfp4LayerWeights& lw = layers_[l];
        int n_vram = layer_vram_cnt[l];
        int n_dyn = std::min(32, std::max(0, n_vram / 4));
        int n_pinned = n_vram - n_dyn;

        lw.moe.vram_experts = n_vram;
        lw.moe.n_pinned = n_pinned;
        lw.moe.n_dyn = n_dyn;

        uint8_t* vram_base = nullptr;
        uint8_t* dyn_vram_base = nullptr;
        if (exp_vmm_ok_) {
            if (n_pinned > 0) {
                vram_base = static_cast<uint8_t*>(gpu_alloc(static_cast<size_t>(n_pinned) * EXP_STRIDE_BYTES));
            }
            if (n_dyn > 0) {
                dyn_vram_base = static_cast<uint8_t*>(exp_vmm_range_.base()) +
                                exp_layer_dyn_slot_off_[l] * EXP_STRIDE_BYTES;
            }
        } else {
            if (n_vram > 0) {
                vram_base = static_cast<uint8_t*>(gpu_alloc(static_cast<size_t>(n_vram) * EXP_STRIDE_BYTES));
            }
            if (n_dyn > 0 && vram_base) {
                dyn_vram_base = vram_base + static_cast<size_t>(n_pinned) * EXP_STRIDE_BYTES;
            }
        }
        lw.moe.vram_base = vram_base;
        lw.moe.dyn_vram_base = dyn_vram_base;

        const int stage_idx = l & 1;
        uint8_t* cur_stage = pin_vram_stage[stage_idx];
        if (cur_stage && l >= 2) {
            HIP_CHECK(hipEventSynchronize(ev_stage_done[stage_idx]));
        }

        uint8_t* layer_gpu_cold = gpu_cold_host_base + layer_gpu_cold_off[l] * EXP_STRIDE_BYTES;
        uint8_t* layer_cpu_host = cpu_host_base + layer_cpu_host_off[l] * EXP_STRIDE_BYTES;
        std::memset(lw.moe.h_ptrs, 0, 512 * 6 * sizeof(int64_t));
        std::memset(lw.moe.h_ptrs_prefill, 0, 512 * 6 * sizeof(int64_t));
        lw.moe.h_s2.resize(512 * 3);
        std::string epfx_base = "model.language_model.layers." + std::to_string(l) + ".mlp.experts.";

        int sorted_e[512];
        for (int e = 0; e < 512; ++e) {
            std::string epfx = epfx_base + std::to_string(e) + ".";
            exp_slices[e] = ExpSlice9{
                main_index.at(epfx + "gate_proj.weight"),
                main_index.at(epfx + "gate_proj.weight_scale"),
                main_index.at(epfx + "gate_proj.weight_scale_2"),
                main_index.at(epfx + "up_proj.weight"),
                main_index.at(epfx + "up_proj.weight_scale"),
                main_index.at(epfx + "up_proj.weight_scale_2"),
                main_index.at(epfx + "down_proj.weight"),
                main_index.at(epfx + "down_proj.weight_scale"),
                main_index.at(epfx + "down_proj.weight_scale_2"),
            };
            sorted_e[e] = e;
            gpu_cold_slot[l][e] = -1;
        }
        std::sort(sorted_e, sorted_e + 512, [&](int a, int b) {
            const auto& da = exp_slices[a].dw;
            const auto& db = exp_slices[b].dw;
            return (da.shard_idx != db.shard_idx) ? (da.shard_idx < db.shard_idx)
                                                  : (da.file_offset < db.file_offset);
        });

        uint8_t* dst_ptr[512]{};
        int gc_idx = 0;
        int ch_idx = 0;
        for (int e = 0; e < 512; ++e) {
            int32_t rk = vram_rank[l][e];
            bool is_pinned = (rk >= 0 && rk < n_pinned);
            if (is_pinned) {
                dst_ptr[e] = cur_stage + static_cast<size_t>(rk) * EXP_STRIDE_BYTES;
            } else if (lw.moe.in_gpu_cold[e]) {
                int s = gc_idx++;
                gpu_cold_slot[l][e] = s;
                dst_ptr[e] = layer_gpu_cold + static_cast<size_t>(s) * EXP_STRIDE_BYTES;
                lw.moe.host_cpu_ptr[e] = dst_ptr[e];
            } else {
                int s = ch_idx++;
                dst_ptr[e] = layer_cpu_host + static_cast<size_t>(s) * EXP_STRIDE_BYTES;
                lw.moe.host_cpu_ptr[e] = dst_ptr[e];
            }
        }

        std::vector<std::thread> io_workers;
        io_workers.reserve(n_io_threads - 1);
        auto load_expert_chunk = [&](int tid) {
            if (tid > 0) {
                int c = cpu_pool_.core_for_thread(tid);
                if (c >= 0) strata::kernels::cpu::pin_current_thread(c);
            }
            int idx_begin = (tid * 512) / n_io_threads;
            int idx_end = ((tid + 1) * 512) / n_io_threads;
            for (int idx = idx_begin; idx < idx_end; ++idx) {
                int e = sorted_e[idx];
                const ExpSlice9& es = exp_slices[e];
                pread_exact(shard_fds[es.gs2.shard_idx], &lw.moe.h_s2[e * 3 + 0], sizeof(float), es.gs2.file_offset);
                pread_exact(shard_fds[es.us2.shard_idx], &lw.moe.h_s2[e * 3 + 1], sizeof(float), es.us2.file_offset);
                pread_exact(shard_fds[es.ds2.shard_idx], &lw.moe.h_s2[e * 3 + 2], sizeof(float), es.ds2.file_offset);
            }
            for (int idx = idx_begin; idx < idx_end; ++idx) {
                int e = sorted_e[idx];
                const ExpSlice9& es = exp_slices[e];
                uint8_t* dst = dst_ptr[e];
                pread_exact(shard_fds[es.ds.shard_idx], dst + EXP_OFF_DOWN_S, EXP_S_BYTES, es.ds.file_offset);
                pread_exact(shard_fds[es.gs.shard_idx], dst + EXP_OFF_GATE_S, EXP_S_BYTES, es.gs.file_offset);
                pread_exact(shard_fds[es.us.shard_idx], dst + EXP_OFF_UP_S, EXP_S_BYTES, es.us.file_offset);
            }
            for (int idx = idx_begin; idx < idx_end; ++idx) {
                int e = sorted_e[idx];
                const ExpSlice9& es = exp_slices[e];
                uint8_t* dst = dst_ptr[e];
                pread_exact(shard_fds[es.dw.shard_idx], dst + EXP_OFF_DOWN_W, EXP_W_BYTES, es.dw.file_offset);
                pread_exact(shard_fds[es.gw.shard_idx], dst + EXP_OFF_GATE_W, EXP_W_BYTES, es.gw.file_offset);
                pread_exact(shard_fds[es.uw.shard_idx], dst + EXP_OFF_UP_W, EXP_W_BYTES, es.uw.file_offset);
                int32_t rk = vram_rank[l][e];
                if (rk >= n_pinned && rk < n_vram && cur_stage) {
                    std::memcpy(cur_stage + static_cast<size_t>(rk) * EXP_STRIDE_BYTES, dst, EXP_STRIDE_BYTES);
                }
            }
        };
        for (int t = 1; t < n_io_threads; ++t) {
            io_workers.emplace_back(load_expert_chunk, t);
        }
        load_expert_chunk(0);
        for (auto& w : io_workers) w.join();

        for (int fd : shard_fds) {
            (void) ::posix_fadvise(fd, 0, 0, POSIX_FADV_DONTNEED);
        }

        if (n_pinned > 0 && cur_stage) {
            HIP_CHECK(hipMemcpyAsync(vram_base, cur_stage,
                                     static_cast<size_t>(n_pinned) * EXP_STRIDE_BYTES,
                                     hipMemcpyHostToDevice, upload_stream));
        }
        if (n_dyn > 0 && cur_stage && lw.moe.dyn_vram_base) {
            HIP_CHECK(hipMemcpyAsync(lw.moe.dyn_vram_base,
                                     cur_stage + static_cast<size_t>(n_pinned) * EXP_STRIDE_BYTES,
                                     static_cast<size_t>(n_dyn) * EXP_STRIDE_BYTES,
                                     hipMemcpyHostToDevice, upload_stream));
        }
        if (cur_stage) {
            HIP_CHECK(hipEventRecord(ev_stage_done[stage_idx], upload_stream));
        }
        for (int e = 0; e < 512; ++e) {
            int32_t rk = vram_rank[l][e];
            bool is_pinned = (rk >= 0 && rk < n_pinned);
            bool is_dyn = (rk >= n_pinned && rk < n_vram);
            if (is_pinned) {
                lw.moe.in_vram[e] = true;
                lw.moe.dyn_slot[e] = -1;
            } else if (is_dyn) {
                int s = rk - n_pinned;
                lw.moe.in_vram[e] = true;
                lw.moe.dyn_slot[e] = static_cast<int16_t>(s);
                lw.moe.slot_exp[s] = e;
                lw.moe.slot_last_used[s] = 0;
                lw.moe.hit_count[e] = 4;
            } else {
                lw.moe.in_vram[e] = false;
                lw.moe.dyn_slot[e] = -1;
            }
        }

        if ((l + 1) % 8 == 0 || l == 47) {
            FILE* log_fp = serve_mode_ ? stderr : stdout;
            std::fprintf(log_fp,
                         "[gfx1201-nvfp4] Loaded layers 0..%d / 47 (layer %d: %d VRAM [%d pinned + %d dynamic], %d host)\n",
                         l, l, n_vram, n_pinned, n_dyn, 512 - n_vram);
            std::fflush(log_fp);
        }
    }

    if (upload_stream) {
        HIP_CHECK(hipStreamSynchronize(upload_stream));
        for (int b = 0; b < 2; ++b) {
            if (ev_stage_done[b]) HIP_CHECK(hipEventDestroy(ev_stage_done[b]));
            if (pin_vram_stage[b]) HIP_CHECK(hipHostFree(pin_vram_stage[b]));
        }
        HIP_CHECK(hipStreamDestroy(upload_stream));
    }

    uint8_t* gpu_cold_host_dev = nullptr;
    if (total_gpu_cold_bytes > 0) {
        HIP_CHECK(hipHostRegister(gpu_cold_host_base, total_gpu_cold_bytes, hipHostRegisterMapped));
        host_registered_.push_back(gpu_cold_host_base);
        void* dptr = nullptr;
        HIP_CHECK(hipHostGetDevicePointer(&dptr, gpu_cold_host_base, 0));
        gpu_cold_host_dev = static_cast<uint8_t*>(dptr);
    }

    for (int l = 0; l < 48; ++l) {
        Nvfp4LayerWeights& lw = layers_[l];
        int n_vram = lw.moe.vram_experts;
        int n_pinned = lw.moe.n_pinned;
        uint8_t* vram_base = lw.moe.vram_base;
        uint8_t* layer_gpu_cold_dev = gpu_cold_host_dev + layer_gpu_cold_off[l] * EXP_STRIDE_BYTES;
        for (int e = 0; e < 512; ++e) {
            int32_t rk = vram_rank[l][e];
            bool is_pinned = (rk >= 0 && rk < n_pinned);
            bool is_dyn = (rk >= n_pinned && rk < n_vram);
            uint8_t* edev = nullptr;
            if (is_pinned) {
                edev = vram_base + static_cast<size_t>(rk) * EXP_STRIDE_BYTES;
            } else if (is_dyn) {
                edev = lw.moe.dyn_vram_base + static_cast<size_t>(rk - n_pinned) * EXP_STRIDE_BYTES;
            } else if (lw.moe.in_gpu_cold[e]) {
                uint8_t* hdev = layer_gpu_cold_dev + static_cast<size_t>(gpu_cold_slot[l][e]) * EXP_STRIDE_BYTES;
                lw.moe.host_dev_ptr[e] = hdev;
                edev = hdev;
            }
            if (lw.moe.in_vram[e] || lw.moe.in_gpu_cold[e]) {
                lw.moe.h_ptrs[e * 6 + 0] = reinterpret_cast<int64_t>(edev + EXP_OFF_GATE_W);
                lw.moe.h_ptrs[e * 6 + 1] = reinterpret_cast<int64_t>(edev + EXP_OFF_GATE_S);
                lw.moe.h_ptrs[e * 6 + 2] = reinterpret_cast<int64_t>(edev + EXP_OFF_UP_W);
                lw.moe.h_ptrs[e * 6 + 3] = reinterpret_cast<int64_t>(edev + EXP_OFF_UP_S);
                lw.moe.h_ptrs[e * 6 + 4] = reinterpret_cast<int64_t>(edev + EXP_OFF_DOWN_W);
                lw.moe.h_ptrs[e * 6 + 5] = reinterpret_cast<int64_t>(edev + EXP_OFF_DOWN_S);
            }
            if (lw.moe.in_vram[e]) {
                for (int k = 0; k < 6; ++k) lw.moe.h_ptrs_prefill[e * 6 + k] = lw.moe.h_ptrs[e * 6 + k];
            }
        }
        lw.moe.d_ptrs = gpu_alloc(512 * 6 * sizeof(int64_t));
        lw.moe.d_ptrs_prefill = gpu_alloc(512 * 6 * sizeof(int64_t));
        lw.moe.d_s2 = gpu_alloc(lw.moe.h_s2.size() * sizeof(float));
        HIP_CHECK(hipMemcpy(lw.moe.d_ptrs, lw.moe.h_ptrs, 512 * 6 * sizeof(int64_t),
                            hipMemcpyHostToDevice));
        HIP_CHECK(hipMemcpy(lw.moe.d_ptrs_prefill, lw.moe.h_ptrs_prefill,
                            512 * 6 * sizeof(int64_t), hipMemcpyHostToDevice));
        HIP_CHECK(hipMemcpy(lw.moe.d_s2, lw.moe.h_s2.data(), lw.moe.h_s2.size() * sizeof(float),
                            hipMemcpyHostToDevice));
    }

    if (want_mtp) {
        using strata::kernels::MTP_EXP_S_BYTES;
        using strata::kernels::MTP_EXP_STRIDE_BYTES;
        using strata::kernels::MTP_EXP_W_BYTES;
        using strata::kernels::MTP_OFF_DOWN_S;
        using strata::kernels::MTP_OFF_DOWN_W;
        using strata::kernels::MTP_OFF_GATE_S;
        using strata::kernels::MTP_OFF_GATE_W;
        using strata::kernels::MTP_OFF_UP_S;
        using strata::kernels::MTP_OFF_UP_W;

        const size_t total_mtp_exp_bytes = 512ULL * MTP_EXP_STRIDE_BYTES;
        void* hptr = nullptr;
        if (::posix_memalign(&hptr, 2 * 1024 * 1024, total_mtp_exp_bytes) != 0 || !hptr) {
            std::fprintf(stderr, "posix_memalign failed for %zu MB MTP FP8 expert host arena\n",
                         total_mtp_exp_bytes / (1024 * 1024));
            std::exit(1);
        }
        (void) ::madvise(hptr, total_mtp_exp_bytes, MADV_HUGEPAGE);
        uint8_t* mtp_host_base = static_cast<uint8_t*>(hptr);
        prefault_hugepages(mtp_host_base, total_mtp_exp_bytes);

        struct MtpExpSlice6 {
            TensorSlice gw, gs, uw, us, dw, ds;
        };
        std::vector<MtpExpSlice6> mtp_slices(512);
        int sorted_e[512];
        for (int e = 0; e < 512; ++e) {
            std::string epfx = "mtp.layers.0.mlp.experts." + std::to_string(e) + ".";
            mtp_slices[e] = MtpExpSlice6{
                main_index.at(epfx + "gate_proj.weight"),
                main_index.at(epfx + "gate_proj.weight_scale_inv"),
                main_index.at(epfx + "up_proj.weight"),
                main_index.at(epfx + "up_proj.weight_scale_inv"),
                main_index.at(epfx + "down_proj.weight"),
                main_index.at(epfx + "down_proj.weight_scale_inv"),
            };
            sorted_e[e] = e;
        }
        std::sort(sorted_e, sorted_e + 512, [&](int a, int b) {
            const auto& da = mtp_slices[a].dw;
            const auto& db = mtp_slices[b].dw;
            return (da.shard_idx != db.shard_idx) ? (da.shard_idx < db.shard_idx)
                                                  : (da.file_offset < db.file_offset);
        });

        std::vector<std::thread> io_workers;
        io_workers.reserve(n_io_threads - 1);
        auto load_mtp_chunk = [&](int tid) {
            if (tid > 0) {
                int c = cpu_pool_.core_for_thread(tid);
                if (c >= 0) strata::kernels::cpu::pin_current_thread(c);
            }
            int idx_begin = (tid * 512) / n_io_threads;
            int idx_end = ((tid + 1) * 512) / n_io_threads;
            for (int idx = idx_begin; idx < idx_end; ++idx) {
                int e = sorted_e[idx];
                const MtpExpSlice6& es = mtp_slices[e];
                uint8_t* dst = mtp_host_base + static_cast<size_t>(e) * MTP_EXP_STRIDE_BYTES;
                pread_exact(shard_fds[es.gs.shard_idx], dst + MTP_OFF_GATE_S, MTP_EXP_S_BYTES, es.gs.file_offset);
                pread_exact(shard_fds[es.us.shard_idx], dst + MTP_OFF_UP_S, MTP_EXP_S_BYTES, es.us.file_offset);
                pread_exact(shard_fds[es.ds.shard_idx], dst + MTP_OFF_DOWN_S, MTP_EXP_S_BYTES, es.ds.file_offset);
                pread_exact(shard_fds[es.gw.shard_idx], dst + MTP_OFF_GATE_W, MTP_EXP_W_BYTES, es.gw.file_offset);
                pread_exact(shard_fds[es.uw.shard_idx], dst + MTP_OFF_UP_W, MTP_EXP_W_BYTES, es.uw.file_offset);
                pread_exact(shard_fds[es.dw.shard_idx], dst + MTP_OFF_DOWN_W, MTP_EXP_W_BYTES, es.dw.file_offset);
            }
        };
        for (int t = 1; t < n_io_threads; ++t) {
            io_workers.emplace_back(load_mtp_chunk, t);
        }
        load_mtp_chunk(0);
        for (auto& w : io_workers) w.join();

        for (int fd : shard_fds) {
            (void) ::posix_fadvise(fd, 0, 0, POSIX_FADV_DONTNEED);
        }

        HIP_CHECK(hipHostRegister(mtp_host_base, total_mtp_exp_bytes, hipHostRegisterMapped));
        host_registered_.push_back(mtp_host_base);
        void* dptr = nullptr;
        HIP_CHECK(hipHostGetDevicePointer(&dptr, mtp_host_base, 0));
        uint8_t* mtp_host_dev = static_cast<uint8_t*>(dptr);

        std::memset(mtp_.moe.h_ptrs, 0, 512 * 6 * sizeof(int64_t));
        std::memset(mtp_.moe.h_ptrs_prefill, 0, 512 * 6 * sizeof(int64_t));
        for (int e = 0; e < 512; ++e) {
            mtp_.moe.in_vram[e] = false;
            mtp_.moe.in_gpu_cold[e] = true;
            mtp_.moe.host_cpu_ptr[e] = mtp_host_base + static_cast<size_t>(e) * MTP_EXP_STRIDE_BYTES;
            uint8_t* edev = mtp_host_dev + static_cast<size_t>(e) * MTP_EXP_STRIDE_BYTES;
            mtp_.moe.host_dev_ptr[e] = edev;
            mtp_.moe.h_ptrs[e * 6 + 0] = reinterpret_cast<int64_t>(edev + MTP_OFF_GATE_W);
            mtp_.moe.h_ptrs[e * 6 + 1] = reinterpret_cast<int64_t>(edev + MTP_OFF_GATE_S);
            mtp_.moe.h_ptrs[e * 6 + 2] = reinterpret_cast<int64_t>(edev + MTP_OFF_UP_W);
            mtp_.moe.h_ptrs[e * 6 + 3] = reinterpret_cast<int64_t>(edev + MTP_OFF_UP_S);
            mtp_.moe.h_ptrs[e * 6 + 4] = reinterpret_cast<int64_t>(edev + MTP_OFF_DOWN_W);
            mtp_.moe.h_ptrs[e * 6 + 5] = reinterpret_cast<int64_t>(edev + MTP_OFF_DOWN_S);
        }
        mtp_.moe.d_ptrs = gpu_alloc(512 * 6 * sizeof(int64_t));
        mtp_.moe.d_ptrs_prefill = gpu_alloc(512 * 6 * sizeof(int64_t), true);
        HIP_CHECK(hipMemcpy(mtp_.moe.d_ptrs, mtp_.moe.h_ptrs, 512 * 6 * sizeof(int64_t),
                            hipMemcpyHostToDevice));
        if (const char* env_tk = std::getenv("STRATA_MTP_TOPK")) {
            mtp_topk_ = std::clamp(std::atoi(env_tk), 1, 10);
        }
        const char* env_dv = std::getenv("STRATA_MTP_DVOCAB");
        bool want_dvocab = (env_dv == nullptr) || (std::strcmp(env_dv, "0") != 0 && std::strcmp(env_dv, "full") != 0);
        if (want_dvocab && d_mtp_dvocab_ != nullptr) {
            std::string dv_path = (env_dv && env_dv[0] != '\0' && std::strcmp(env_dv, "1") != 0)
                                      ? std::string(env_dv)
                                      : std::string("data/draft_vocab_en.bin");
            std::error_code dv_ec;
            if (!std::filesystem::exists(dv_path, dv_ec)) {
                std::string cand2 = model_dir_ + "/draft_vocab_en.bin";
                if (std::filesystem::exists(cand2, dv_ec)) {
                    dv_path = cand2;
                } else {
                    std::filesystem::path exe_p = std::filesystem::read_symlink("/proc/self/exe", dv_ec);
                    if (!dv_ec) {
                        std::filesystem::path p1 = exe_p.parent_path() / "data" / "draft_vocab_en.bin";
                        std::filesystem::path p2 = exe_p.parent_path().parent_path() / "data" / "draft_vocab_en.bin";
                        if (std::filesystem::exists(p1, dv_ec)) {
                            dv_path = p1.string();
                        } else if (std::filesystem::exists(p2, dv_ec)) {
                            dv_path = p2.string();
                        }
                    }
                }
            }
            int dv_fd = ::open(dv_path.c_str(), O_RDONLY);
            if (dv_fd >= 0) {
                off_t fsz = ::lseek(dv_fd, 0, SEEK_END);
                int32_t n_ids = static_cast<int32_t>(fsz / static_cast<off_t>(sizeof(int32_t)));
                if (n_ids > 0 && n_ids <= 60000) {
                    h_mtp_dvocab_.resize(n_ids);
                    h_mtp_dvocab_.reserve(65536);
                    pread_exact(dv_fd, h_mtp_dvocab_.data(), static_cast<size_t>(n_ids) * sizeof(int32_t), 0);
                    mtp_in_dvocab_.assign(248320, 0);
                    int32_t valid_cnt = 0;
                    for (int32_t i = 0; i < n_ids; ++i) {
                        int32_t tid = h_mtp_dvocab_[i];
                        if (tid >= 0 && tid < 248320 && !mtp_in_dvocab_[tid]) {
                            mtp_in_dvocab_[tid] = 1;
                            h_mtp_dvocab_[valid_cnt++] = tid;
                        }
                    }
                    // Ensure EOS / stop tokens are always present in draft vocabulary
                    for (int32_t eos_id : {248044, 248046}) {
                        if (!mtp_in_dvocab_[eos_id] && valid_cnt < 65536) {
                            mtp_in_dvocab_[eos_id] = 1;
                            h_mtp_dvocab_.push_back(eos_id);
                            valid_cnt++;
                        }
                    }
                    h_mtp_dvocab_.resize(valid_cnt);
                    n_mtp_dvocab_ = valid_cnt;
                    HIP_CHECK(hipMemcpy(d_mtp_dvocab_, h_mtp_dvocab_.data(),
                                        static_cast<size_t>(n_mtp_dvocab_) * sizeof(int32_t),
                                        hipMemcpyHostToDevice));
                }
                ::close(dv_fd);
            }
        }
        mtp_.available = true;
        FILE* log_fp = serve_mode_ ? stderr : stdout;
        std::fprintf(log_fp,
                     "[gfx1201-nvfp4] Loaded native MTP draft layer (BF16 dense + 512 FP8 experts in %.2f GiB mapped hugepage RAM, topk=%d, dvocab=%d)\n",
                     static_cast<double>(total_mtp_exp_bytes) / (1024.0 * 1024.0 * 1024.0),
                     mtp_topk_, n_mtp_dvocab_);
        std::fflush(log_fp);
    }

    for (size_t s = 0; s < shard_fds.size(); ++s) {
        if (static_cast<int>(s) != ple_shard_idx && shard_fds[s] >= 0) {
            ::close(shard_fds[s]);
        }
    }
}

}  // namespace strata::core
