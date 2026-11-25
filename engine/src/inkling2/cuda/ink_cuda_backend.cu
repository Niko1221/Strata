// engine/src/inkling2/cuda/ink_cuda_backend.cu - InkBackend CUDA Plugin for `strata-inkling` (sm_70..sm_121a)
//
// MIT License
// Copyright (c) 2026 Niko1221 and the Strata contributors
#include "ink_backend.h"
#include "ink_batch.h"
#include "ink_manifest.h"

#include "ink_cuda.cuh"
#include "ink_dspark.cuh"
#include "ink_runner.cuh"
#include "ink_spec.cuh"
#include "ink_topk.cuh"
#include "strata/core/arch_probe.hpp"
#include "strata/core/ink_chunk.hpp"
#include "strata/core/volta_forever.hpp"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstring>
#include <memory>
#include <string>
#include <unordered_map>
#include <vector>

namespace strata::ink {

// Copy 1 slot's 48-visit ShortConv state [48 x 3 x 6656] <-> snapshot buffer (2.5 us on GDDR7!)
__global__ void __launch_bounds__(256) k_ink_sconv_slot_copy(
    float* const* __restrict__ sconv_pools,  // [48]
    float* __restrict__ snap_buf,            // [48, 3 * 6656]
    int slot,
    int max_slots,
    bool to_snap) {
    const int v = blockIdx.x;
    const int strm = blockIdx.y;
    const int tid = threadIdx.x;
    if (v >= kInkVisits || strm >= 4) return;
    constexpr int c_off[4] = {0, kInkKVDim, 2 * kInkKVDim, 2 * kInkKVDim + kInkHidden};
    constexpr int c_len[4] = {kInkKVDim, kInkKVDim, kInkHidden, kInkHidden};
    const int ch_off = c_off[strm];
    const int C = c_len[strm];

    float* slot_h = sconv_pools[v] + (size_t)ch_off * 3 * max_slots + (size_t)slot * 3 * C;
    float* snap_h = snap_buf + (size_t)v * 3 * kInkSconvCh + (size_t)ch_off * 3;
    const int n = 3 * C;
    if (to_snap) {
        for (int i = tid; i < n; i += 256) snap_h[i] = slot_h[i];
    } else {
        for (int i = tid; i < n; i += 256) slot_h[i] = snap_h[i];
    }
}

// Gather selected hidden rows d_h_norm[idx[k], 2304] -> d_h_sel[k, 2304]
__global__ void __launch_bounds__(256) k_ink_gather_rows(
    const float* __restrict__ src,
    const int32_t* __restrict__ idx,
    float* __restrict__ dst,
    int n_sel) {
    const int k = blockIdx.x;
    if (k >= n_sel) return;
    const int row = idx[k];
    const float4* s4 = reinterpret_cast<const float4*>(src + (size_t)row * kInkHidden);
    float4* d4 = reinterpret_cast<float4*>(dst + (size_t)k * kInkHidden);
    for (int i = threadIdx.x; i < (kInkHidden / 4); i += 256) d4[i] = s4[i];
}

// Never launched: cudaFuncGetAttributes() on it says whether this binary holds GPU code for the card (#6759).
__global__ void k_ink_arch_probe(int* p) {
    if (p) *p = 1;
}

}  // namespace strata::ink

namespace ink2 {

class InkCudaBackend final : public InkBackend {
public:
    ~InkCudaBackend() override { cleanup(); }

    const char* backend_name() const noexcept override {
        return model_.ls_active ? "cuda-ls" : (model_.ep2_active ? "cuda-ep2" : "cuda");
    }

    int effective_max_ctx() const override { return initialized_ ? eff_max_ctx_ : 0; }
    int effective_chunk() const override { return initialized_ ? eff_chunk_run_ : 0; }
    bool last_failure_is_kv_growth() const override { return kv_grow_failed_; }
    int kv_capacity_tokens() const override { return (initialized_ && runner_) ? runner_->kv_alloc_ctx : 0; }

    bool dspark_ready() const override { return initialized_ && dspark_ && dspark_->ready; }

    std::string device_info() const override { return dev_info_; }

    BackendStats stats() const override {
        BackendStats st{};
        if (initialized_) {
            st.vram_experts_loaded = model_.n_vram_experts + model_.n_peer_experts;
            st.ram_experts_loaded  = model_.n_host_experts;
            st.total_experts       = model_.n_vram_experts + model_.n_peer_experts + model_.n_host_experts;
            size_t vf = 0, vt = 0;
            if (cudaMemGetInfo(&vf, &vt) == cudaSuccess) {
                st.vram_total_bytes = vt;
                st.vram_used_bytes  = vt - vf;
            }
            st.max_slots = max_slots_;
            int act = 0;
            for (bool a : slot_alloc_) if (a) ++act;
            st.active_slots = act;
            st.kv_dtype = kv_dtype_;
            model_.hybrid.stats_total(&st.expert_selections, &st.expert_misses);   // M = 1 (device counters) + batched path
        }
        return st;
    }

    bool init(const InkModelFile& mf, const BackendOpts& opts, std::string& err) override {
        if (const char* e = std::getenv("STRATA_INK_SPEC_BEAM")) spec_cfg_.beam = std::atoi(e);
        if (const char* e = std::getenv("STRATA_INK_SPEC_COST")) spec_cfg_.cost = (float)std::atof(e);
        if (const char* e = std::getenv("STRATA_INK_SPEC_PLAIN")) spec_cfg_.plain = (float)std::atof(e);
        if (opts.spec_beam > 0) spec_cfg_.beam = opts.spec_beam;
        if (opts.spec_cost > 0.0) spec_cfg_.cost = (float)opts.spec_cost;
        if (opts.spec_plain >= 0.0) spec_cfg_.plain = (float)opts.spec_plain;
        spec_cfg_.beam = std::max(1, std::min(8, spec_cfg_.beam));
        spec_cfg_.log = std::getenv("STRATA_INK_SPEC_LOG") != nullptr;
        cleanup();
        int n_dev = 0;
        cudaError_t ce = cudaGetDeviceCount(&n_dev);
        if (ce != cudaSuccess || n_dev <= 0) {
            err = std::string("cudaGetDeviceCount failed: ") + cudaGetErrorString(ce);
            return false;
        }
        // Resolve primary + optional multi-GPU list (up to 4 GPUs!) from opts.gpu_ids / opts.gpu_index / opts.gpus
        gpu_id_ = (opts.gpu_index >= 0 && opts.gpu_index < n_dev) ? opts.gpu_index : 0;
        std::vector<int> all_gpus;
        if (!opts.gpu_ids.empty()) {
            for (int g : opts.gpu_ids) {
                if (g >= 0 && g < n_dev && std::find(all_gpus.begin(), all_gpus.end(), g) == all_gpus.end() && all_gpus.size() < 4) {
                    all_gpus.push_back(g);
                }
            }
            if (!all_gpus.empty()) gpu_id_ = all_gpus[0];
        } else if (opts.gpus >= 2 && n_dev >= 2) {
            const int want = std::min(std::min(opts.gpus, n_dev), 4);
            all_gpus.push_back(gpu_id_);
            for (int g = 0; g < n_dev && (int)all_gpus.size() < want; ++g) {
                if (g != gpu_id_) all_gpus.push_back(g);
            }
        } else {
            all_gpus.push_back(gpu_id_);
        }
        const int peer_gpu_id = (all_gpus.size() >= 2) ? all_gpus[1] : -1;

        ce = cudaSetDevice(gpu_id_);
        if (ce != cudaSuccess) {
            err = std::string("cudaSetDevice failed: ") + cudaGetErrorString(ce);
            return false;
        }
        cudaDeviceProp prop{};
        ce = cudaGetDeviceProperties(&prop, gpu_id_);
        if (ce != cudaSuccess) {
            err = std::string("cudaGetDeviceProperties failed: ") + cudaGetErrorString(ce);
            return false;
        }
        const int sm_cc = prop.major * 10 + prop.minor;

        // A binary built without code for this card (the CUDA 13 packages hold sm_75 and newer) loads fine, then every kernel launch
        // fails with "no kernel image" without a message and the logits stay 0: token id 0 forever (#6759). Say so before anything loads.
        {
            cudaFuncAttributes probe_attr{};
            const cudaError_t pe = cudaFuncGetAttributes(&probe_attr, reinterpret_cast<const void*>(strata::ink::k_ink_arch_probe));
            if (pe != cudaSuccess) {
                cudaGetLastError();
                err = strata::core::no_gpu_code_message("strata-inkling", prop.name, prop.major, prop.minor, cudaGetErrorString(pe));
                return false;
            }
        }

        // Compute exact minimum KV + ShortConv + workspace headroom BEFORE pinning experts!
        // Keep lossless FP16 KV as default when Host KV Offload handles capacity (#552)!
        max_slots_ = std::max(1, std::min(kMaxParallelSlots, opts.parallel));
        const int kv_offload_mode = (opts.kv_offload == "on") ? 1 : ((opts.kv_offload == "off") ? -1 : 0);
        kv_dtype_ = opts.kv_dtype;
        if (opts.kv_dtype_auto) {
            if (max_slots_ >= 16) {
                kv_dtype_ = KvDtype::FP8_E4M3;
            } else if (kv_offload_mode < 0) {
                if (prop.totalGlobalMem <= (13ull << 30)) {
                    kv_dtype_ = KvDtype::FP8_E4M3;
                } else if (sm_cc >= 89 && opts.max_ctx > 16384 && prop.totalGlobalMem < (40ull << 30)) {
                    kv_dtype_ = KvDtype::FP8_E4M3;
                }
            }
        }
        auto map_kvd = [](KvDtype d) -> strata::ink::InkKvDtype {
            if (d == KvDtype::FP8_E4M3) return strata::ink::InkKvDtype::FP8;
            if (d == KvDtype::K8V8)     return strata::ink::InkKvDtype::K8V8;
            if (d == KvDtype::K8V4)     return strata::ink::InkKvDtype::K8V4;
            return strata::ink::InkKvDtype::FP16;
        };
        const auto cuda_kvd = map_kvd(kv_dtype_);

        const int eff_chunk = std::max(64, opts.chunk);
        const size_t swa_r = (max_slots_ >= 16) ? (size_t)(strata::ink::kInkSwaWin + 32)
                                                : (size_t)(strata::ink::kInkSwaWin + std::max(eff_chunk, 64));
        const size_t per_visit_pos = strata::ink::InkCudaRunner::kv_bytes_per_visit_pos(cuda_kvd);
        const int init_glob = (kv_offload_mode == 1)
            ? 0
            : ((kv_offload_mode == 0 && prop.totalGlobalMem <= (13ull << 30))
                ? std::min(std::max(256, opts.max_ctx), 2048)
                : std::max(256, std::min(std::max(256, opts.max_ctx), (max_slots_ == 1) ? 8192 : 4096)));
        const size_t min_kv_bytes = (size_t)40 * max_slots_ * swa_r * per_visit_pos
                                  + (size_t)strata::ink::kInkVisits * max_slots_ * 3 * strata::ink::kInkSconvCh * 4
                                  + (size_t)8 * max_slots_ * init_glob * per_visit_pos;
        // Include O(1) prefill ws (~114 MiB) + DSpark (~224 MiB when enabled) + activation ws (WDDM floor added inside loader!)
        const bool build_screen = opts.enable_dspark && mf.m.dspark.present;
        const size_t dspark_pad = build_screen ? (224ull << 20) : 0ull;
        const size_t ws_pad = ((max_slots_ >= 8) ? (1280ull << 20)
                             : ((prop.totalGlobalMem >= (20ull << 30)) ? (960ull << 20)
                              : (prop.totalGlobalMem <= (13ull << 30)) ? (512ull << 20)
                                                                       : (832ull << 20))) + dspark_pad;
        size_t min_reserve_bytes = min_kv_bytes + ws_pad;
        if (opts.vram_reserve_mib >= 0) {
            min_reserve_bytes = std::max(min_reserve_bytes, (size_t)opts.vram_reserve_mib << 20);
        }

        // Load .strata v1.1 / .gguf (Unsloth + ggml-org) / HF safetensors trunk + experts (Single-GPU, EP2, Layer-Split, or 4-tier GPU+RAM+SSD!)
        if (!model_.load_from_model_file(mf, gpu_id_, peer_gpu_id, (float)opts.ep_split, build_screen, err,
                                         min_reserve_bytes, gguf_detail::gguf_dequant_row_f32, gguf_detail::gguf_row_bytes,
                                         opts.vram_experts_gb, opts.ram_experts_gb, opts.ssd_experts,
                                         all_gpus, opts.layer_split, opts.threads)) {
            return false;
        }

        {
            char buf[512];
            if (model_.ls_active) {
                std::string s = "LayerSplit[";
                for (int i = 0; i < model_.n_stages; ++i) {
                    cudaDeviceProp sp{};
                    cudaGetDeviceProperties(&sp, model_.stage_gpu[i]);
                    if (i > 0) s += " + ";
                    char part[128];
                    std::snprintf(part, sizeof(part), "GPU%d:%.48s(%.0fG,L%d-%d)",
                                  model_.stage_gpu[i], sp.name, (double)sp.totalGlobalMem / 1073741824.0,
                                  model_.stage_l0[i], model_.stage_l1[i] - 1);
                    s += part;
                }
                s += "]";
                dev_info_ = s;
            } else if (peer_gpu_id >= 0 && model_.ep2_active) {
                cudaDeviceProp prop1{};
                cudaGetDeviceProperties(&prop1, peer_gpu_id);
                std::snprintf(buf, sizeof(buf), "GPU%d:%.120s + GPU%d:%.120s (sm_%d%d, %.1f+%.1f GiB VRAM, ep2=1)",
                              gpu_id_, prop.name, peer_gpu_id, prop1.name, prop.major, prop.minor,
                              (double)prop.totalGlobalMem / 1073741824.0, (double)prop1.totalGlobalMem / 1073741824.0);
                dev_info_ = buf;
            } else {
                std::snprintf(buf, sizeof(buf), "%.200s (sm_%d%d, %d SMs, %.1f GiB VRAM, sm120_tc=%d)",
                              prop.name, prop.major, prop.minor, prop.multiProcessorCount,
                              (double)prop.totalGlobalMem / 1073741824.0, (int)strata::ink::sm120::available());
                dev_info_ = buf;
            }
        }

        const bool vf_graph_ok = strata::core::allow_1024_thread_cuda_graphs(
            prop.major, prop.minor, strata::core::volta_forever_active());
        if (!vf_graph_ok) {
            std::fprintf(stderr, "[inkling-cuda] sm_70 under volta-forever: disabling 1024-thread CUDA Graph capture by default\n");
        }
        // Memory ledger (what each start-up step took from the card), so a plan that leaves too little can be seen in the log (#6644 #6792).
        auto free_mib = [&]() -> double {
            size_t vf = 0, vt = 0;
            if (strata::ink::ink_cuda_mem_info(&vf, &vt, gpu_id_) != cudaSuccess) return -1.0;
            return (double)vf / 1048576.0;
        };
        // Test hook: STRATA_INK_TEST_STEAL_MIB=N holds N MiB of the card after the experts are placed, to imitate a plan that left too little.
        if (const char* steal = std::getenv("STRATA_INK_TEST_STEAL_MIB")) {
            const long long mib = std::atoll(steal);
            if (mib > 0 && cudaMalloc(&d_test_steal_, (size_t)mib << 20) != cudaSuccess) { d_test_steal_ = nullptr; cudaGetLastError(); }
        }
        const double led_loaded = free_mib();

        // The experts are already in the card, so a start-up allocation that does not fit is met by a smaller workspace (the
        // prefill chunk is halved down to 64) instead of ending the engine (#6644 #6792); the drafter is dropped last of all.
        // STRATA_INK_OOM_RETRY=0 turns the retry off (the engine then ends on the first allocation that does not fit, as before).
        const char* retry_env = std::getenv("STRATA_INK_OOM_RETRY");
        strata::ink::ink_alloc_throws() = !(retry_env && std::atoi(retry_env) == 0);
        struct ThrowGuard { ~ThrowGuard() { strata::ink::ink_alloc_throws() = false; } } throw_guard;
        int chunk_try = std::max(64, opts.chunk);
        // No --chunk: the memory left after the experts decides (strata/core/ink_chunk.hpp); STRATA_INK_CHUNK_AUTO=0 keeps the old 512.
        if (opts.chunk_auto && !model_.ls_active) {
            const char* auto_env = std::getenv("STRATA_INK_CHUNK_AUTO");
            if (!(auto_env && std::atoi(auto_env) == 0)) {
                // the K/V still to be allocated for every slot to reach the context stays free of the chunk's working memory (kv offload on: it is host RAM)
                const double kv_grow_mib = (kv_offload_mode == 1) ? 0.0
                    : (double)8 * max_slots_ * std::max(0, std::max(256, opts.max_ctx) - init_glob) * (double)per_visit_pos / 1048576.0;
                chunk_try = std::max(chunk_try, strata::core::ink_auto_chunk(led_loaded, max_slots_, kv_grow_mib));
                std::fprintf(stderr, "[InkCUDA] prefill chunk %d (auto: %.0f MiB free after the experts, %.0f MiB of K/V still to grow, %d slot(s); --chunk N sets it)\n",
                             chunk_try, led_loaded, kv_grow_mib, max_slots_);
            }
        }
        for (;;) {
            runner_ = std::make_unique<strata::ink::InkCudaRunner>(model_);
            runner_->use_cuda_graph = vf_graph_ok && !model_.ep2_active && !model_.ls_active;
            if (const char* ng = std::getenv("STRATA_INK_NO_GRAPH"); ng && ng[0] == '1' && runner_->use_cuda_graph) {
                // diagnostic: launch every kernel of a step eagerly, as the layer-split / EP2 paths always do
                runner_->use_cuda_graph = false;
                std::fprintf(stderr, "[InkCUDA] STRATA_INK_NO_GRAPH=1: CUDA Graph capture off (eager launches)\n");
            }
            if (const char* pe = std::getenv("STRATA_INK_PROF"); pe && (pe[0] == '1' || pe[0] == '2')) {   // per-phase decode times, eager (no graph); 2 = prefill chunks too
                runner_->prof_enabled = true;
                runner_->prof_prefill = (pe[0] == '2');
                runner_->use_cuda_graph = false;
                strata::ink::g_ink_moe_hook_ctx = runner_.get();
                strata::ink::g_ink_moe_hook = [](int tag, void* ctx) {
                    static_cast<strata::ink::InkCudaRunner*>(ctx)->prof_mark_end(8 + tag);
                };
                std::fprintf(stderr, "[InkCUDA] STRATA_INK_PROF=1: eager launches, per-phase timing (runner phases every 50 batched steps, host phases every 32)\n");
            }
            try {
                runner_->init(max_slots_, std::max(256, opts.max_ctx), chunk_try, cuda_kvd, kv_offload_mode);
                break;
            } catch (const strata::ink::InkAllocFailure& ex) {
                runner_->free_all();
                runner_.reset();
                cudaGetLastError();
                if (chunk_try <= 64) {
                    err = std::string(ex.what()) + "; not enough free GPU memory for the working buffers even at the smallest size. "
                          "Use a smaller --max-context, a lower --vram-experts-gb, or close other programs that use the card.";
                    return false;
                }
                const int next = std::max(64, chunk_try / 2);
                std::fprintf(stderr, "[InkCUDA] not enough free GPU memory for the working buffers at chunk %d (%s); retrying with chunk %d\n",
                             chunk_try, ex.what(), next);
                chunk_try = next;
            }
        }
        eff_max_ctx_ = runner_->max_ctx;
        eff_chunk_run_ = std::max(1, std::min(chunk_try, runner_->max_batch));
        const double led_runner = free_mib();

        if (opts.enable_dspark && model_.man.has_dspark && !model_.ls_active) {
            dspark_ = std::make_unique<strata::ink::InkDSpark>(*runner_);
            try {
                dspark_->init(eff_max_ctx_, 512);
            } catch (const strata::ink::InkAllocFailure& ex) {
                dspark_->free_all();
                dspark_.reset();
                cudaGetLastError();
                std::fprintf(stderr, "[InkCUDA] DSpark drafter turned off: not enough free GPU memory for it (%s)\n", ex.what());
            }
        }
        strata::ink::ink_alloc_throws() = false;
        std::fprintf(stderr, "[InkCUDA] memory: %.0f MiB free after load, runner+KV took %.0f MiB (max_batch=%d), DSpark took %.0f MiB, %.0f MiB free now\n",
                     led_loaded, led_loaded - led_runner, runner_->max_batch, led_runner - free_mib(), free_mib());

        runner_->pick_cold_lane();   // STRATA_INK_EXACT unset: time the CPU lane and the copy-in lane for the cold experts once, keep the faster

        // Allocate snapshot buffer (3.83 MB) + selected hidden gather buffers
        if (!model_.ls_active) {
            d_sconv_snap_ = strata::ink::dalloc<float>((size_t)strata::ink::kInkVisits * 3 * strata::ink::kInkSconvCh);
        }
        d_h_sel_      = strata::ink::dalloc<float>((size_t)runner_->max_batch * strata::ink::kInkHidden);
        d_sel_idx_    = strata::ink::dalloc<int32_t>(runner_->max_batch);
        slot_alloc_.assign(max_slots_, false);
        snap_active_.assign(max_slots_, false);
        snap_p0_.assign(max_slots_, 0);
        snap_M_.assign(max_slots_, 0);
        prompt_ckpts_.assign(max_slots_, PromptCkpt{});
        shared_ckpts_.assign(max_slots_, PromptCkpt{});
        initialized_ = true;
        return true;
    }

    bool alloc_slot(int slot, int /*max_ctx*/, KvDtype /*kv_dtype*/, std::string& err) override {
        if (!initialized_ || slot < 0 || slot >= max_slots_) {
            err = "invalid slot " + std::to_string(slot);
            return false;
        }
        slot_alloc_[slot] = true;
        reset_slot(slot);
        return true;
    }

    void free_slot(int slot) override {
        if (slot >= 0 && slot < max_slots_) {
            slot_alloc_[slot] = false;
            snap_active_[slot] = false;
            if (slot < (int)prompt_ckpts_.size()) prompt_ckpts_[slot] = PromptCkpt{};
            if (slot < (int)shared_ckpts_.size()) shared_ckpts_[slot] = PromptCkpt{};
            if (slot == 0) tree_snap_ready_ = false;
        }
    }

    void reset_slot(int slot) override {
        if (!initialized_ || slot < 0 || slot >= max_slots_) return;
        runner_->reset_slot(slot);
        snap_active_[slot] = false;
        if (slot < (int)prompt_ckpts_.size()) prompt_ckpts_[slot].valid = false;
        if (slot < (int)shared_ckpts_.size()) shared_ckpts_[slot].valid = false;
        if (slot == 0) { dspark_slot_ = -1; tree_snap_ready_ = false; }
    }

    // #6627: Save/restore slot's ShortConv (3.83 MB) + 40 SWA KV ring windows (<=160 MiB in host RAM) + DSpark SWA KV
    // so multi-turn chats and exact repeats reuse the full prior prompt even if hidden thinking or trailing tokens diverged!
    bool save_prompt_checkpoint(int slot, int seq_len) override {
        return save_checkpoint(slot, seq_len, false);
    }
    bool restore_prompt_checkpoint(int slot, int seq_len) override {
        return restore_checkpoint(slot, seq_len, false);
    }
    bool supports_shared_checkpoint() const override {
        return initialized_ && (model_.ls_active || d_sconv_snap_ != nullptr) && max_slots_ <= 8;   // (a layer split: the per-stage copies of save_checkpoint)
    }
    bool save_shared_checkpoint(int slot, int seq_len) override {
        return save_checkpoint(slot, seq_len, true);
    }
    bool restore_shared_checkpoint(int slot, int seq_len) override {
        return restore_checkpoint(slot, seq_len, true);
    }

    // Layer split (--layer-split, 2+ GPUs): visit v's ShortConv state and SWA ring live on the GPU of the stage that owns its layer, so the
    // checkpoint copies run per visit on that stage's device and stream (the one-kernel gather of the single-GPU path would read other
    // cards' memory).  `visit_stream(v)` selects them (single GPU: the runner's stream, nothing else changes); `ckpt_sync()` waits for
    // every stream the copies used and returns to the device the runner works on.
    cudaStream_t visit_stream(int v) {
        if (!model_.ls_active) return runner_->st;
        const int s = model_.layer_owner[strata::ink::ink_visit_phys(v)];
        INK_CK(cudaSetDevice(model_.stage_gpu[s]));
        return runner_->ls_stages[s].st;
    }
    void ckpt_sync() {
        if (!model_.ls_active) { INK_CK(cudaStreamSynchronize(runner_->st)); return; }
        for (int s = 0; s < model_.n_stages; ++s) {
            INK_CK(cudaSetDevice(model_.stage_gpu[s]));
            INK_CK(cudaStreamSynchronize(runner_->ls_stages[s].st));
        }
        INK_CK(cudaSetDevice(model_.stage_gpu[runner_->cur_stage]));
    }
    // ShortConv state of one slot, host layout [48 visits][3 x 6656] (the snapshot buffer's), per visit and channel group on its own stage
    void ls_sconv_io(int slot, float* host, bool to_host) {
        constexpr int c_off[4] = {0, strata::ink::kInkKVDim, 2 * strata::ink::kInkKVDim, 2 * strata::ink::kInkKVDim + strata::ink::kInkHidden};
        constexpr int c_len[4] = {strata::ink::kInkKVDim, strata::ink::kInkKVDim, strata::ink::kInkHidden, strata::ink::kInkHidden};
        for (int v = 0; v < strata::ink::kInkVisits; ++v) {
            cudaStream_t vs = visit_stream(v);
            for (int g = 0; g < 4; ++g) {
                float* dev = runner_->d_sconv_pool[v] + (size_t)c_off[g] * 3 * runner_->max_slots + (size_t)slot * 3 * c_len[g];
                float* hst = host + (size_t)v * 3 * strata::ink::kInkSconvCh + (size_t)c_off[g] * 3;
                INK_CK(cudaMemcpyAsync(to_host ? (void*)hst : (void*)dev, to_host ? (const void*)dev : (const void*)hst,
                                       (size_t)3 * c_len[g] * sizeof(float), to_host ? cudaMemcpyDeviceToHost : cudaMemcpyHostToDevice, vs));
            }
        }
    }

    bool save_checkpoint(int slot, int seq_len, bool shared) {
        if (!initialized_ || slot < 0 || slot >= max_slots_ || (!model_.ls_active && !d_sconv_snap_)) return false;
        if (seq_len <= 0 || max_slots_ > 8) return false;
        auto& ck = (shared ? shared_ckpts_ : prompt_ckpts_)[slot];
        ck.valid = false;
        ck.seq_len = seq_len;
        if (model_.ls_active) ckpt_sync();             // the stage streams finished the step before the state is read
        // 1. Copy 48-visit ShortConv state via d_sconv_snap_ -> host RAM (3.83 MB)
        const size_t sc_elems = (size_t)strata::ink::kInkVisits * 3 * strata::ink::kInkSconvCh;
        ck.sconv.resize(sc_elems);
        if (model_.ls_active) {
            ls_sconv_io(slot, ck.sconv.data(), true);
        } else {
            strata::ink::k_ink_sconv_slot_copy<<<dim3(strata::ink::kInkVisits, 4), 256, 0, runner_->st>>>(
                runner_->d_sconv_pool_arr, d_sconv_snap_, slot, runner_->max_slots, true);
            INK_CK(cudaMemcpyAsync(ck.sconv.data(), d_sconv_snap_, sc_elems * sizeof(float), cudaMemcpyDeviceToHost, runner_->st));
        }
        // 2. Copy active SWA KV ring window [start_pos .. seq_len-1] for all 40 SWA visits
        const int start_pos = std::max(0, seq_len - strata::ink::kInkSwaWin);
        const int n_pos = seq_len - start_pos;
        ck.swa_start = start_pos;
        ck.swa_len = n_pos;
        ck.swa_k.clear(); ck.swa_v.clear(); ck.swa_ksc.clear(); ck.swa_vsc.clear();
        for (int v = 0; v < strata::ink::kInkVisits; ++v) {
            const auto& kv = runner_->kv_desc[v];
            if (!kv.is_swa) continue;
            cudaStream_t vs = visit_stream(v);
            const int cap = kv.cap_pos;
            const int p0 = start_pos % cap;
            const int seg1 = std::min(n_pos, cap - p0);
            const int seg2 = n_pos - seg1;
            const size_t slot_base = (size_t)slot * cap;
            auto copy_d2h = [&](const void* base, size_t elem_b, size_t stride_elems, std::vector<uint8_t>& dst) {
                const size_t off = dst.size();
                dst.resize(off + (size_t)n_pos * stride_elems * elem_b);
                const uint8_t* bp = reinterpret_cast<const uint8_t*>(base);
                INK_CK(cudaMemcpyAsync(dst.data() + off, bp + (slot_base + p0) * stride_elems * elem_b,
                                       (size_t)seg1 * stride_elems * elem_b, cudaMemcpyDeviceToHost, vs));
                if (seg2 > 0) {
                    INK_CK(cudaMemcpyAsync(dst.data() + off + (size_t)seg1 * stride_elems * elem_b,
                                           bp + slot_base * stride_elems * elem_b,
                                           (size_t)seg2 * stride_elems * elem_b, cudaMemcpyDeviceToHost, vs));
                }
            };
            if (kv.kv_fmt == 0) {
                copy_d2h(kv.k_f16, sizeof(__half), strata::ink::kInkKVDim, ck.swa_k);
                copy_d2h(kv.v_f16, sizeof(__half), strata::ink::kInkKVDim, ck.swa_v);
            } else {
                const size_t v_stride = (kv.kv_fmt == 3) ? (strata::ink::kInkKVDim / 2) : strata::ink::kInkKVDim;
                copy_d2h(kv.k_f8, 1, strata::ink::kInkKVDim, ck.swa_k);
                copy_d2h(kv.v_f8, 1, v_stride, ck.swa_v);
                copy_d2h(kv.k_sc, sizeof(float), strata::ink::kInkKVHeads, ck.swa_ksc);
                copy_d2h(kv.v_sc, sizeof(float), strata::ink::kInkKVHeads, ck.swa_vsc);
            }
        }
        // 3. Save DSpark SWA-1025 context KV (layers 0 & 2) + f_rnn row at seq_len-1 for slot 0
        ck.has_dspark = false;
        if (slot == 0 && dspark_ && dspark_->ready && dspark_slot_ == 0) {
            ck.dspark_k.clear(); ck.dspark_v.clear();
            for (int il = 0; il < 3; ++il) {
                if (model_.man.c.dspark_is_swa[il] == 0) continue;
                const int cap = dspark_->ctx_cap[il];
                const int s0 = std::max(0, seq_len - model_.man.c.dspark_swa_win);
                const int np = seq_len - s0;
                const int p0 = s0 % cap;
                const int seg1 = std::min(np, cap - p0);
                const int seg2 = np - seg1;
                auto copy_ds = [&](const __half* src, std::vector<uint8_t>& dst) {
                    const size_t row_b = (size_t)strata::ink::kInkKVDim * sizeof(__half);
                    const size_t off = dst.size();
                    dst.resize(off + (size_t)np * row_b);
                    const uint8_t* bp = reinterpret_cast<const uint8_t*>(src);
                    INK_CK(cudaMemcpyAsync(dst.data() + off, bp + (size_t)p0 * row_b, (size_t)seg1 * row_b, cudaMemcpyDeviceToHost, runner_->st));
                    if (seg2 > 0) {
                        INK_CK(cudaMemcpyAsync(dst.data() + off + (size_t)seg1 * row_b, bp, (size_t)seg2 * row_b, cudaMemcpyDeviceToHost, runner_->st));
                    }
                };
                copy_ds(dspark_->d_ctx_k[il], ck.dspark_k);
                copy_ds(dspark_->d_ctx_v[il], ck.dspark_v);
            }
            ck.dspark_frnn.resize(strata::ink::kInkHidden);
            const int fr_slot = (seq_len - 1) % dspark_->frnn_cap;
            INK_CK(cudaMemcpyAsync(ck.dspark_frnn.data(), dspark_->d_f_rnn + (size_t)fr_slot * strata::ink::kInkHidden,
                                   (size_t)strata::ink::kInkHidden * sizeof(float), cudaMemcpyDeviceToHost, runner_->st));
            ck.has_dspark = true;
        }
        ckpt_sync();
        ck.valid = true;
        return true;
    }

    bool restore_checkpoint(int slot, int seq_len, bool shared) {
        if (!initialized_ || slot < 0 || slot >= max_slots_ || (!model_.ls_active && !d_sconv_snap_)) return false;
        const auto& ckpts = shared ? shared_ckpts_ : prompt_ckpts_;
        if (slot >= (int)ckpts.size()) return false;
        const auto& ck = ckpts[slot];
        if (!ck.valid || ck.seq_len != seq_len) return false;
        snap_active_[slot] = false;
        if (slot == 0) tree_snap_ready_ = false;
        // 1. Restore ShortConv state
        const size_t sc_elems = (size_t)strata::ink::kInkVisits * 3 * strata::ink::kInkSconvCh;
        if (model_.ls_active) {
            ckpt_sync();                                // (the stage streams are idle: this request's read has not started)
            ls_sconv_io(slot, const_cast<float*>(ck.sconv.data()), false);
        } else {
            INK_CK(cudaMemcpyAsync(d_sconv_snap_, ck.sconv.data(), sc_elems * sizeof(float), cudaMemcpyHostToDevice, runner_->st));
            strata::ink::k_ink_sconv_slot_copy<<<dim3(strata::ink::kInkVisits, 4), 256, 0, runner_->st>>>(
                runner_->d_sconv_pool_arr, d_sconv_snap_, slot, runner_->max_slots, false);
        }
        // 2. Restore 40 SWA KV ring windows (Global KV is append-only and already intact at [0..seq_len-1]!)
        size_t ok = 0, ov = 0, oks = 0, ovs = 0;
        const int start_pos = ck.swa_start;
        const int n_pos = ck.swa_len;
        for (int v = 0; v < strata::ink::kInkVisits; ++v) {
            const auto& kv = runner_->kv_desc[v];
            if (!kv.is_swa) continue;
            cudaStream_t vs = visit_stream(v);
            const int cap = kv.cap_pos;
            const int p0 = start_pos % cap;
            const int seg1 = std::min(n_pos, cap - p0);
            const int seg2 = n_pos - seg1;
            const size_t slot_base = (size_t)slot * cap;
            auto copy_h2d = [&](void* base, size_t elem_b, size_t stride_elems, const std::vector<uint8_t>& src, size_t& off) {
                uint8_t* bp = reinterpret_cast<uint8_t*>(base);
                INK_CK(cudaMemcpyAsync(bp + (slot_base + p0) * stride_elems * elem_b, src.data() + off,
                                       (size_t)seg1 * stride_elems * elem_b, cudaMemcpyHostToDevice, vs));
                if (seg2 > 0) {
                    INK_CK(cudaMemcpyAsync(bp + slot_base * stride_elems * elem_b,
                                           src.data() + off + (size_t)seg1 * stride_elems * elem_b,
                                           (size_t)seg2 * stride_elems * elem_b, cudaMemcpyHostToDevice, vs));
                }
                off += (size_t)n_pos * stride_elems * elem_b;
            };
            if (kv.kv_fmt == 0) {
                copy_h2d(kv.k_f16, sizeof(__half), strata::ink::kInkKVDim, ck.swa_k, ok);
                copy_h2d(kv.v_f16, sizeof(__half), strata::ink::kInkKVDim, ck.swa_v, ov);
            } else {
                const size_t v_stride = (kv.kv_fmt == 3) ? (strata::ink::kInkKVDim / 2) : strata::ink::kInkKVDim;
                copy_h2d(kv.k_f8, 1, strata::ink::kInkKVDim, ck.swa_k, ok);
                copy_h2d(kv.v_f8, 1, v_stride, ck.swa_v, ov);
                copy_h2d(kv.k_sc, sizeof(float), strata::ink::kInkKVHeads, ck.swa_ksc, oks);
                copy_h2d(kv.v_sc, sizeof(float), strata::ink::kInkKVHeads, ck.swa_vsc, ovs);
            }
        }
        // 3. Restore DSpark SWA-1025 context KV + f_rnn row for slot 0
        if (slot == 0 && ck.has_dspark && dspark_ && dspark_->ready) {
            size_t dk_o = 0, dv_o = 0;
            for (int il = 0; il < 3; ++il) {
                if (model_.man.c.dspark_is_swa[il] == 0) continue;
                const int cap = dspark_->ctx_cap[il];
                const int s0 = std::max(0, seq_len - model_.man.c.dspark_swa_win);
                const int np = seq_len - s0;
                const int p0 = s0 % cap;
                const int seg1 = std::min(np, cap - p0);
                const int seg2 = np - seg1;
                auto copy_ds_h2d = [&](__half* dst, const std::vector<uint8_t>& src, size_t& off) {
                    const size_t row_b = (size_t)strata::ink::kInkKVDim * sizeof(__half);
                    uint8_t* bp = reinterpret_cast<uint8_t*>(dst);
                    INK_CK(cudaMemcpyAsync(bp + (size_t)p0 * row_b, src.data() + off, (size_t)seg1 * row_b, cudaMemcpyHostToDevice, runner_->st));
                    if (seg2 > 0) {
                        INK_CK(cudaMemcpyAsync(bp, src.data() + off + (size_t)seg1 * row_b, (size_t)seg2 * row_b, cudaMemcpyHostToDevice, runner_->st));
                    }
                    off += (size_t)np * row_b;
                };
                copy_ds_h2d(dspark_->d_ctx_k[il], ck.dspark_k, dk_o);
                copy_ds_h2d(dspark_->d_ctx_v[il], ck.dspark_v, dv_o);
            }
            const int fr_slot = (seq_len - 1) % dspark_->frnn_cap;
            INK_CK(cudaMemcpyAsync(dspark_->d_f_rnn + (size_t)fr_slot * strata::ink::kInkHidden, ck.dspark_frnn.data(),
                                   (size_t)strata::ink::kInkHidden * sizeof(float), cudaMemcpyHostToDevice, runner_->st));
            dspark_slot_ = 0;
        } else if (slot == 0) {
            dspark_slot_ = -1;
        }
        ckpt_sync();
        return true;
    }

    void snapshot_slot(int slot) override {
        if (!initialized_ || slot < 0 || slot >= max_slots_ || model_.ls_active || !d_sconv_snap_) return;
        strata::ink::k_ink_sconv_slot_copy<<<dim3(strata::ink::kInkVisits, 4), 256, 0, runner_->st>>>(
            runner_->d_sconv_pool_arr, d_sconv_snap_, slot, runner_->max_slots, true);
        snap_active_[slot] = true;
        snap_M_[slot] = 0;
        if (slot == 0) tree_snap_ready_ = false;
    }

    void rollback_slot(int slot, int keep_len) override {
        if (!initialized_ || slot < 0 || slot >= max_slots_ || model_.ls_active || !d_sconv_snap_) return;
        if (snap_active_[slot]) {
            const int n_keep = std::max(0, std::min(snap_M_[slot], keep_len - snap_p0_[slot]));
            strata::ink::k_ink_sconv_slot_copy<<<dim3(strata::ink::kInkVisits, 4), 256, 0, runner_->st>>>(
                runner_->d_sconv_pool_arr, d_sconv_snap_, slot, runner_->max_slots, false);
            if (n_keep > 0) {
                int32_t path[16];
                for (int i = 0; i < n_keep; ++i) path[i] = i;
                INK_CK(cudaMemcpyAsync(runner_->d_acc_path, path, (size_t)n_keep * 4, cudaMemcpyHostToDevice, runner_->st));
                strata::ink::k_ink_commit_sconv_all48<<<dim3(strata::ink::kInkVisits, 4), 256, 0, runner_->st>>>(
                    runner_->d_sconv_pool_arr, runner_->d_stage_sconv, runner_->d_acc_path, n_keep, slot, runner_->max_slots, runner_->tree_cap);
            }
            snap_active_[slot] = false;
            return;
        }
        if (slot == 0 && tree_snap_ready_) {
            // Exact multi-branch tree rewind on early EOS / stop_tok / anti-loop!
            const int n_keep = std::max(0, std::min(tree_snap_M_, keep_len - tree_snap_p0_));
            strata::ink::k_ink_sconv_slot_copy<<<dim3(strata::ink::kInkVisits, 4), 256, 0, runner_->st>>>(
                runner_->d_sconv_pool_arr, d_sconv_snap_, 0, runner_->max_slots, false);
            if (n_keep > 0) {
                INK_CK(cudaMemcpyAsync(runner_->d_acc_path, tree_snap_path_, (size_t)n_keep * 4, cudaMemcpyHostToDevice, runner_->st));
                strata::ink::k_ink_commit_sconv_all48<<<dim3(strata::ink::kInkVisits, 4), 256, 0, runner_->st>>>(
                    runner_->d_sconv_pool_arr, runner_->d_stage_sconv, runner_->d_acc_path, n_keep, 0, runner_->max_slots, runner_->tree_cap);
            }
            tree_snap_ready_ = false;
        }
    }

    bool forward_batch(
        const StepBatch& batch,
        float* logits_out,
        TapCapture* taps_out,
        std::string& err) override {
        return forward_batch_impl(batch, logits_out, nullptr, taps_out, err);
    }

    bool supports_candidates() const noexcept override { return initialized_; }

    bool forward_batch_cand(const StepBatch& batch, CandRows& out, std::string& err) override {
        if (!initialized_) { err = "CUDA backend not initialized"; return false; }
        // the scratch is allocated before anything runs, so a failure here leaves the step untouched (the caller keeps the full-row path)
        if (!topk_.ensure(std::max(64, batch.count_logits()))) { err = "no memory for the candidate scratch"; return false; }
        const int V = model_.man.c.vocab_unpadded;
        const int nrows = batch.count_logits();
        out.rows = nrows;
        out.k = std::min(kInkCandidates, V);
        out.val.assign((size_t)nrows * kInkCandidates, 0.0f);
        out.idx.assign((size_t)nrows * kInkCandidates, 0);
        out.finite.assign((size_t)nrows, 1);
        return forward_batch_impl(batch, nullptr, &out, nullptr, err);
    }

    bool forward_batch_impl(
        const StepBatch& batch,
        float* logits_out,
        CandRows* cand_out,
        TapCapture* taps_out,
        std::string& err) {
        if (!initialized_) {
            err = "CUDA backend not initialized";
            return false;
        }
        const int N = batch.n_tokens;
        if (N <= 0) return true;
        // STRATA_INK_STEPTIME=1: where one decode token of the serving loop goes, with the graph kept (diagnostic, off by default):
        // host time between two steps (sampling, protocol, batch building), the step call's wall time (launches + head + top-K + sync)
        // and the GPU span of the call (first event to the event after the top-K, before the sync).
        struct StepTimeGuard {
            static bool on() { static const bool v = [] { const char* e = std::getenv("STRATA_INK_STEPTIME"); return e && e[0] == '1'; }(); return v; }
            bool act = false;
            std::chrono::steady_clock::time_point t0;
            cudaEvent_t e0 = nullptr;
            cudaStream_t s = nullptr;
            cudaEvent_t* e1_slot = nullptr;
            StepTimeGuard(int N_, cudaStream_t st, cudaEvent_t* e1) : act(on() && N_ == 1), s(st), e1_slot(e1) {
                if (!act) return;
                t0 = std::chrono::steady_clock::now();
                static cudaEvent_t ev0 = nullptr;
                if (!ev0) cudaEventCreate(&ev0);
                e0 = ev0;
                cudaEventRecord(e0, s);
            }
            ~StepTimeGuard() {
                if (!act) return;
                static std::chrono::steady_clock::time_point last_end{};
                static double s_between = 0, s_wall = 0, s_gpu = 0; static long s_n = 0, s_gpu_n = 0;
                const auto t1 = std::chrono::steady_clock::now();
                if (last_end.time_since_epoch().count() != 0) s_between += std::chrono::duration<double, std::milli>(t0 - last_end).count();
                s_wall += std::chrono::duration<double, std::milli>(t1 - t0).count();
                if (e1_slot && *e1_slot) {
                    float ms = 0.0f;
                    if (cudaEventElapsedTime(&ms, e0, *e1_slot) == cudaSuccess) { s_gpu += ms; ++s_gpu_n; }
                }
                ++s_n;
                last_end = std::chrono::steady_clock::now();
                if (s_n % 100 == 0) {
                    std::fprintf(stderr, "[InkCUDA STEPTIME] %ld one-token steps: host between steps %.3f ms, step call wall %.3f ms, GPU span of the call %.3f ms (n=%ld)\n",
                                 s_n, s_between / s_n, s_wall / s_n, s_gpu_n ? s_gpu / s_gpu_n : 0.0, s_gpu_n);
                    s_between = s_wall = s_gpu = 0; s_n = s_gpu_n = 0;
                }
            }
        };
        static cudaEvent_t steptime_e1 = nullptr;
        if (StepTimeGuard::on() && !steptime_e1) cudaEventCreate(&steptime_e1);
        StepTimeGuard steptime_guard(N, runner_->st, &steptime_e1);
        kv_grow_failed_ = false;
        // Defensive KV position bounds check + elastic global KV growth (fixes #6590B CUDA error 700 & CA1JEeAx!)
        int max_pos_req = 0;
        for (int i = 0; i < N; ++i) {
            if (batch.pos[i] < 0 || batch.pos[i] >= eff_max_ctx_) {
                err = "CUDA KV position " + std::to_string(batch.pos[i]) + " out of bounds [0, " + std::to_string(eff_max_ctx_) + ")";
                return false;
            }
            if (batch.pos[i] > max_pos_req) max_pos_req = batch.pos[i];
        }
        if (!runner_->ensure_global_kv_cap(max_pos_req + 1)) {
            err = "VRAM exhausted growing elastic global KV cache to " + std::to_string(max_pos_req + 1) + " tokens";
            kv_grow_failed_ = true;
            return false;
        }
        if (batch.capture_taps && dspark_ && dspark_->ready) {
            if (!dspark_->ensure_ctx_cap(max_pos_req + 1)) {
                dspark_->ready = false;
            }
        }
        tree_snap_ready_ = false;
        const int V = model_.man.c.vocab_unpadded;

        // Pre-compute packed output row index for every token with want_logits[i] != 0
        std::vector<int> out_row(N, -1);
        int total_want = 0;
        for (int i = 0; i < N; ++i) {
            if (batch.want_logits[i]) out_row[i] = total_want++;
        }

        if (taps_out) {
            for (int t = 0; t < 4; ++t) taps_out->taps[t].assign((size_t)N * strata::ink::kInkHidden, 0.0f);
        }

        // Helper: extract wanted logits rows from runner_->d_h_norm[0..M-1] into logits_out
        auto emit_selected_logits = [&](const std::vector<int>& local_idx, const std::vector<int>& dst_rows) {
            const int nw = (int)local_idx.size();
            if (nw == 0 || (!logits_out && !cand_out)) return;
            const float* h_src = runner_->d_h_norm;
            if (nw == 1) {
                h_src = runner_->d_h_norm + (size_t)local_idx[0] * strata::ink::kInkHidden;
            } else {
                bool contiguous_0 = true;
                for (int k = 0; k < nw; ++k) if (local_idx[k] != k) { contiguous_0 = false; break; }
                if (!contiguous_0) {
                    INK_CK(cudaMemcpyAsync(d_sel_idx_, local_idx.data(), (size_t)nw * 4, cudaMemcpyHostToDevice, runner_->st));
                    strata::ink::k_ink_gather_rows<<<nw, 256, 0, runner_->st>>>(runner_->d_h_norm, d_sel_idx_, d_h_sel_, nw);
                    h_src = d_h_sel_;
                }
            }
            const bool hp = runner_->prof_enabled;
            auto hnow = [] { return std::chrono::steady_clock::now(); };
            auto hms = [](std::chrono::steady_clock::time_point a, std::chrono::steady_clock::time_point b) { return std::chrono::duration<double, std::milli>(b - a).count(); };
            auto hq0 = hnow();
            runner_->head_logits_from(h_src, nw);
            if (hp) INK_CK(cudaStreamSynchronize(runner_->st));
            auto hq1 = hnow();
            if (cand_out) {
                // candidate rows: the 128 best (id, logit) of each wanted row and its finite flag; no full row leaves the card
                INK_CK(topk_.run(runner_->d_logits, nw, V, runner_->st));
                if (steptime_e1 && nw == 1 && StepTimeGuard::on()) cudaEventRecord(steptime_e1, runner_->st);
                INK_CK(cudaStreamSynchronize(runner_->st));
                const int kc = std::min(kInkCandidates, V);
                for (int k = 0; k < nw; ++k) {
                    const int dst = dst_rows[k];
                    const uint64_t* keys = topk_.h_out + (size_t)k * strata::ink::kTopkCand;
                    for (int j = 0; j < kc; ++j) {
                        int32_t id = strata::ink::ink_topk_key_index(keys[j]);
                        if (id < 0 || id >= V) id = 0;     // only a NaN logit can leave a pad key here
                        cand_out->val[(size_t)dst * kInkCandidates + j] = strata::ink::ink_topk_key_value(keys[j]);
                        cand_out->idx[(size_t)dst * kInkCandidates + j] = id;
                    }
                    cand_out->finite[dst] = topk_.h_fin[k] != 0 ? 1 : 0;
                }
                return;
            }
            // Use pinned host staging buffer for full PCIe 5.0 DMA bandwidth (~50 GB/s vs ~12 GB/s pageable)!
            ensure_pinned_logits(nw, V);
            INK_CK(cudaMemcpyAsync(h_logits_pin_, runner_->d_logits,
                                   (size_t)nw * V * sizeof(float), cudaMemcpyDeviceToHost, runner_->st));
            INK_CK(cudaStreamSynchronize(runner_->st));
            auto hq2 = hnow();
            struct HostProfGuard {
                bool on; double head, d2h; std::chrono::steady_clock::time_point t2; int nw;
                ~HostProfGuard() {
                    if (!on) return;
                    static double s_head = 0, s_d2h = 0, s_cpy = 0; static long s_n = 0, s_rows = 0;
                    s_head += head; s_d2h += d2h; s_cpy += std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - t2).count(); ++s_n; s_rows += nw;
                    if (s_n % 32 == 0) {
                        std::fprintf(stderr, "[InkCUDA PROF host] %ld logits batches (%.1f rows avg): unembed+softcap %.2f ms, D2H %.2f ms, copy to caller %.2f ms per batch\n",
                                     s_n, (double)s_rows / s_n, s_head / s_n, s_d2h / s_n, s_cpy / s_n);
                    }
                }
            } hpg{hp, hms(hq0, hq1), hms(hq1, hq2), hq2, nw};
            bool contiguous_dst = true;
            for (int k = 1; k < nw; ++k) if (dst_rows[k] != dst_rows[k - 1] + 1) { contiguous_dst = false; break; }
            if (contiguous_dst) {
                std::memcpy(logits_out + (size_t)dst_rows[0] * V, h_logits_pin_, (size_t)nw * V * sizeof(float));
            } else {
                for (int k = 0; k < nw; ++k) {
                    std::memcpy(logits_out + (size_t)dst_rows[k] * V,
                                h_logits_pin_ + (size_t)k * V, (size_t)V * sizeof(float));
                }
            }
        };

        auto copy_host_taps = [&](const std::vector<int>& orig_indices) {
            if (!taps_out) return;
            const int M = (int)orig_indices.size();
            for (int t = 0; t < 4; ++t) {
                std::vector<float> tmp((size_t)M * strata::ink::kInkHidden);
                INK_CK(cudaMemcpyAsync(tmp.data(), runner_->d_taps[t], tmp.size() * 4, cudaMemcpyDeviceToHost, runner_->st));
                INK_CK(cudaStreamSynchronize(runner_->st));
                for (int k = 0; k < M; ++k) {
                    std::memcpy(taps_out->taps[t].data() + (size_t)orig_indices[k] * strata::ink::kInkHidden,
                                tmp.data() + (size_t)k * strata::ink::kInkHidden,
                                strata::ink::kInkHidden * sizeof(float));
                }
            }
        };

        // Helper: run a contiguous single-slot sequence (chunked to runner_->max_batch if needed!)
        auto run_single_slot_seq = [&](int slot, const std::vector<int>& idxs) {
            const int total_m = (int)idxs.size();
            int off = 0;
            while (off < total_m) {
                const int M = std::min(total_m - off, runner_->max_batch);
                const int p0 = batch.pos[idxs[off]];
                std::vector<int32_t> toks(M);
                std::vector<uint8_t> is_img(M, 0);
                bool any_img = false;
                for (int k = 0; k < M; ++k) {
                    const int oi = idxs[off + k];
                    toks[k] = batch.token_ids[oi];
                    if (batch.img_embeds && oi < (int)batch.img_embed_idx.size() && batch.img_embed_idx[oi] >= 0) {
                        is_img[k] = 1;
                        any_img = true;
                    }
                }
                if (any_img) {
                    std::vector<float> img_flat((size_t)M * strata::ink::kInkHidden, 0.0f);
                    for (int k = 0; k < M; ++k) {
                        if (is_img[k]) {
                            const int img_r = batch.img_embed_idx[idxs[off + k]];
                            std::memcpy(img_flat.data() + (size_t)k * strata::ink::kInkHidden,
                                        batch.img_embeds + (size_t)img_r * strata::ink::kInkHidden,
                                        strata::ink::kInkHidden * sizeof(float));
                        }
                    }
                    INK_CK(cudaMemcpyAsync(runner_->d_is_img, is_img.data(), M, cudaMemcpyHostToDevice, runner_->st));
                    INK_CK(cudaMemcpyAsync(runner_->d_img_emb, img_flat.data(), img_flat.size() * 4, cudaMemcpyHostToDevice, runner_->st));
                    runner_->has_img_step = true;
                } else {
                    runner_->has_img_step = false;
                }

                const bool want_taps = (taps_out != nullptr) || (batch.capture_taps && dspark_ && dspark_->ready && slot == 0);

                // If snapshot_slot(slot) is active for DSpark verify (M <= 16, no images):
                // run forward_tree_verify (stages ShortConv + KV) & commit all M tokens so rollback_slot can replay exact n_keep!
                if (snap_active_[slot] && !any_img && M <= 16 && total_m <= 16) {
                    strata::ink::InkTreeSpec tr{};
                    tr.M = M;
                    for (int k = 0; k < M; ++k) {
                        tr.tok[k] = toks[k];
                        tr.parent[k] = (int8_t)(k - 1);
                        tr.depth[k] = (uint8_t)k;
                    }
                    tr.build();
                    runner_->forward_tree_verify(tr, p0, slot, want_taps);
                    int32_t path[16];
                    for (int k = 0; k < M; ++k) path[k] = k;
                    INK_CK(cudaMemcpyAsync(runner_->d_acc_path, path, (size_t)M * 4, cudaMemcpyHostToDevice, runner_->st));
                    strata::ink::k_ink_commit_sconv_all48<<<dim3(strata::ink::kInkVisits, 4), 256, 0, runner_->st>>>(
                        runner_->d_sconv_pool_arr, runner_->d_stage_sconv, runner_->d_acc_path, M, slot, runner_->max_slots, runner_->tree_cap);
                    strata::ink::k_ink_commit_kv_all48<<<dim3(strata::ink::kInkVisits, strata::ink::kInkKVHeads), 128, 0, runner_->st>>>(
                        runner_->d_kv_desc_arr, runner_->d_stage_k, runner_->d_stage_v, runner_->d_acc_path, p0, M, slot, runner_->tree_cap);
                    snap_p0_[slot] = p0;
                    snap_M_[slot] = M;
                } else {
                    runner_->forward_seq(toks.data(), p0, M, slot, want_taps);
                }
                runner_->has_img_step = false;

                if (batch.capture_taps && dspark_ && dspark_->ready && slot == 0 && p0 + M <= dspark_->max_ctx) {
                    dspark_->ingest_taps(p0, M);
                    dspark_slot_ = 0;
                }
                if (taps_out) {
                    std::vector<int> slice(idxs.begin() + off, idxs.begin() + off + M);
                    copy_host_taps(slice);
                }

                std::vector<int> loc_idx, dst_rows;
                for (int k = 0; k < M; ++k) {
                    const int oi = idxs[off + k];
                    if (out_row[oi] >= 0) { loc_idx.push_back(k); dst_rows.push_back(out_row[oi]); }
                }
                emit_selected_logits(loc_idx, dst_rows);
                off += M;
            }
        };

        // Partition tokens by slot count:
        // - Distinct 1-token-per-slot items with no images -> ONE batched CUDA-Graph forward_batched(B) call!
        // - Multi-token or image-bearing slot runs -> run_single_slot_seq(slot, idxs)!
        std::unordered_map<int, std::vector<int>> by_slot;
        std::vector<int> slot_order;
        for (int i = 0; i < N; ++i) {
            int s = batch.slot_ids[i];
            if (by_slot.find(s) == by_slot.end()) slot_order.push_back(s);
            by_slot[s].push_back(i);
        }

        // Fast-path A: Single slot (prefill, M=1 decode, or DSpark verify)
        if (slot_order.size() == 1) {
            run_single_slot_seq(slot_order[0], by_slot[slot_order[0]]);
            INK_CK(cudaStreamSynchronize(runner_->st));
            return true;
        }

        // Fast-path B: Separate 1-token-per-slot decode items (coalesced into 1 forward_batched!) and multi-token prefill runs!
        std::vector<int> dec_orig;
        std::vector<int32_t> dec_toks, dec_slots, dec_pos;
        std::vector<int> multi_slots;
        for (int s : slot_order) {
            const auto& idxs = by_slot[s];
            const bool is_img = (batch.img_embeds && idxs[0] < (int)batch.img_embed_idx.size() && batch.img_embed_idx[idxs[0]] >= 0);
            if (idxs.size() == 1 && !is_img && !snap_active_[s]) {
                dec_orig.push_back(idxs[0]);
                dec_toks.push_back(batch.token_ids[idxs[0]]);
                dec_slots.push_back(s);
                dec_pos.push_back(batch.pos[idxs[0]]);
            } else {
                multi_slots.push_back(s);
            }
        }

        if (!dec_orig.empty()) {
            const int B = (int)dec_orig.size();
            const bool want_taps = (taps_out != nullptr);
            runner_->has_img_step = false;
            {
                static double p_step_ms = 0; static long p_n = 0, p_rows = 0;
                const bool pf = runner_->prof_enabled;
                auto tb0 = std::chrono::steady_clock::now();
                runner_->forward_batched(dec_toks.data(), dec_slots.data(), dec_pos.data(), B, want_taps);
                if (pf) {
                    INK_CK(cudaStreamSynchronize(runner_->st));
                    p_step_ms += std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - tb0).count();
                    ++p_n; p_rows += B;
                    if (p_n % 32 == 0) {
                        std::fprintf(stderr, "[InkCUDA PROF step] %ld batched steps, B avg %.1f, eager 48-visit step %.2f ms\n", p_n, (double)p_rows / p_n, p_step_ms / p_n);
                    }
                }
            }
            if (taps_out) copy_host_taps(dec_orig);
            std::vector<int> loc_idx, dst_rows;
            for (int k = 0; k < B; ++k) {
                const int oi = dec_orig[k];
                if (out_row[oi] >= 0) { loc_idx.push_back(k); dst_rows.push_back(out_row[oi]); }
            }
            emit_selected_logits(loc_idx, dst_rows);
        }

        for (int s : multi_slots) {
            run_single_slot_seq(s, by_slot[s]);
        }

        INK_CK(cudaStreamSynchronize(runner_->st));
        return true;
    }

    int dspark_draft(
        int slot,
        int32_t anchor_tok,
        int anchor_pos,
        int max_draft,
        int32_t* out_draft_ids,
        std::string& /*err*/) override {
        if (!initialized_ || model_.ls_active || !dspark_ || !dspark_->ready || slot != 0 || dspark_slot_ != 0) return 0;
        if (anchor_pos <= 0 || anchor_pos + 8 >= eff_max_ctx_ || anchor_pos + 8 >= dspark_->max_ctx) return 0;
        if (!runner_->ensure_global_kv_cap(anchor_pos + 8) || !dspark_->ensure_ctx_cap(anchor_pos + 8)) return 0;
        int32_t ids7[8] = {};
        float conf7[8] = {};
        dspark_->draft_block8(anchor_tok, anchor_pos, ids7, conf7);
        const int cap = std::max(0, std::min(7, max_draft));
        if (cap <= 0) return 0;
        // Sparse-MoE Bayes-Optimal Adaptive Draft Length (128 experts, Top-8):
        // Pick nd in [2..cap] maximizing expected emitted tokens / (1 + 4.2 * (1 - (15/16)^(nd+1)))!
        double best_rate = -1.0;
        int best_nd = std::min(2, cap);
        double cum_p = 1.0;
        double e_tok = 1.0;
        for (int k = 1; k <= cap; ++k) {
            cum_p *= std::max(0.0, std::min(1.0, (double)conf7[k - 1]));
            e_tok += cum_p;
            if (k >= 2 || k == cap) {
                const double exp_frac = 1.0 - std::pow(15.0 / 16.0, (double)(k + 1));
                const double rate = e_tok / (1.0 + 4.2 * exp_frac);
                if (rate > best_rate) { best_rate = rate; best_nd = k; }
            }
        }
        for (int i = 0; i < best_nd; ++i) out_draft_ids[i] = ids7[i];
        return best_nd;
    }

    int dspark_spec_step_greedy(
        int slot,
        int32_t anchor_tok,
        int anchor_pos,
        int max_emit,
        int32_t* out_emitted,
        int* out_drafted,
        std::string& /*err*/) override {
        if (!initialized_ || model_.ls_active || !dspark_ || !dspark_->ready || slot != 0 || dspark_slot_ != 0) return 0;
        if (anchor_pos <= 0 || anchor_pos + 12 >= eff_max_ctx_ || anchor_pos + 12 >= dspark_->max_ctx || max_emit < 1 || !out_emitted) return 0;
        if (!runner_->ensure_global_kv_cap(anchor_pos + 12) || !dspark_->ensure_ctx_cap(anchor_pos + 12)) return 0;

        // 1. Snapshot slot-0 ShortConv pre-verify state (2.5 us) for exact mid-tree EOS rollback if needed!
        strata::ink::k_ink_sconv_slot_copy<<<dim3(strata::ink::kInkVisits, 4), 256, 0, runner_->st>>>(
            runner_->d_sconv_pool_arr, d_sconv_snap_, 0, runner_->max_slots, true);

        // 2. True W=4 Multi-Branch Sparse-MoE Bayes-Optimal Adaptive Beam Tree (M* in {3,4,5,6,8,10,12})!
        strata::ink::InkTreeSpec tr{};
        const int max_nodes = std::min(12, runner_->tree_cap);
        dspark_->draft_beam_tree(anchor_tok, anchor_pos, spec_cfg_.beam, max_nodes, tr, spec_cfg_.cost, spec_cfg_.plain);

        // 3. Single CUDA-Graph 48-visit tree verify + exact GPU screened argmax (ZERO 6.4 MB logits D2H!)
        runner_->forward_tree_verify(tr, anchor_pos, 0, true);
        int32_t target_next[strata::ink::kInkMaxTree] = {};
        runner_->head_argmax_exact(tr.M, target_next);

        // 4. Exact greedy tree walk (clamped to min(max_emit, 8))
        const int emit_cap = std::min(max_emit, 8);
        int32_t path[strata::ink::kInkMaxTree] = {};
        int n_emit = 0;
        int cur_node = 0;
        while (n_emit < emit_cap) {
            path[n_emit] = cur_node;
            const int32_t tok_next = target_next[cur_node];
            out_emitted[n_emit++] = tok_next;
            int next_child = -1;
            for (int j = cur_node + 1; j < tr.M; ++j) {
                if (tr.parent[j] == cur_node && tr.tok[j] == tok_next) {
                    next_child = j;
                    break;
                }
            }
            if (next_child >= 0) cur_node = next_child;
            else break;
        }

        // 5. Commit ONLY the accepted tree path across all 48 visits (2 launches) + compact & ingest DSpark taps!
        INK_CK(cudaMemcpyAsync(runner_->d_acc_path, path, (size_t)n_emit * 4, cudaMemcpyHostToDevice, runner_->st));
        strata::ink::k_ink_commit_sconv_all48<<<dim3(strata::ink::kInkVisits, 4), 256, 0, runner_->st>>>(
            runner_->d_sconv_pool_arr, runner_->d_stage_sconv, runner_->d_acc_path, n_emit, 0, runner_->max_slots, runner_->tree_cap);
        strata::ink::k_ink_commit_kv_all48<<<dim3(strata::ink::kInkVisits, strata::ink::kInkKVHeads), 128, 0, runner_->st>>>(
            runner_->d_kv_desc_arr, runner_->d_stage_k, runner_->d_stage_v, runner_->d_acc_path, anchor_pos, n_emit, 0, runner_->tree_cap);
        for (int t = 0; t < 4; ++t) {
            strata::ink::k_ink_compact_taps<<<n_emit, 256, 0, runner_->st>>>(
                runner_->d_taps[t], runner_->d_taps_compact[t], runner_->d_acc_path, n_emit);
        }
        dspark_->ingest_taps_from(runner_->d_taps_compact, anchor_pos, n_emit);

        if (spec_cfg_.log) {
            static long st_steps = 0, st_emit = 0, st_nodes = 0; static double st_pred = 0.0; static long st_plain = 0;
            ++st_steps; st_emit += n_emit; st_nodes += tr.M; st_pred += dspark_->last_pred_etok; if (tr.M == 1) ++st_plain;
            if (st_steps % 200 == 0) std::fprintf(stderr, "[SPEC-STATS] steps=%ld mean_M=%.2f mean_pred_tokens=%.2f mean_actual_tokens=%.2f plain_steps=%ld\n", st_steps, (double)st_nodes / st_steps, st_pred / st_steps, (double)st_emit / st_steps, st_plain);
        }
        tree_snap_ready_ = true;
        tree_snap_p0_ = anchor_pos;
        tree_snap_M_ = n_emit;
        for (int i = 0; i < n_emit; ++i) tree_snap_path_[i] = path[i];
#ifdef INK_MOE_STATS
        {
            static int stat_calls = 0;
            if (++stat_calls % 100 == 0) {
                unsigned long long c[64], a[64];
                cudaMemcpyFromSymbol(c, strata::ink::g_ink_moe_calls, sizeof(c));
                cudaMemcpyFromSymbol(a, strata::ink::g_ink_moe_nact, sizeof(a));
                std::fprintf(stderr, "[MOE-STATS] M:visits:mean-unique-experts");
                for (int m = 1; m < 64; ++m) if (c[m]) std::fprintf(stderr, " %d:%llu:%.1f", m, c[m], (double)a[m] / (double)c[m]);
                std::fputc(10, stderr);
            }
        }
#endif
        if (out_drafted) *out_drafted = tr.M - 1;
        return n_emit;
    }

    bool selfcheck(std::string& err) override {
        cudaDeviceProp prop{};
        cudaError_t ce = cudaGetDeviceProperties(&prop, gpu_id_);
        if (ce != cudaSuccess) {
            err = std::string("CUDA device query failed: ") + cudaGetErrorString(ce);
            return false;
        }
        // Verify SITU + softcap + basic device execution
        float* d_tmp = strata::ink::dalloc<float>(64);
        std::vector<float> h_in(64), h_out(64);
        for (int i = 0; i < 64; ++i) h_in[i] = (float)(i - 32) * 1.5f;
        INK_CK(cudaMemcpy(d_tmp, h_in.data(), 64 * 4, cudaMemcpyHostToDevice));
        strata::ink::k_ink_softcap<<<1, 256>>>(d_tmp, 64, 30.0f);
        INK_CK(cudaMemcpy(h_out.data(), d_tmp, 64 * 4, cudaMemcpyDeviceToHost));
        INK_CK(cudaFree(d_tmp));
        for (int i = 0; i < 64; ++i) {
            float ref = 30.0f * std::tanh(h_in[i] / 30.0f);
            if (std::fabs(h_out[i] - ref) > 1e-4f) {
                err = "CUDA softcap selfcheck mismatch";
                return false;
            }
        }
        return true;
    }

private:
    void ensure_pinned_logits(int nw, int V) {
        if (nw <= h_logits_cap_ && h_logits_pin_) return;
        if (h_logits_pin_) { cudaFreeHost(h_logits_pin_); h_logits_pin_ = nullptr; }
        h_logits_cap_ = std::max(nw, 8);
        h_logits_pin_ = strata::ink::halloc_pinned<float>((size_t)h_logits_cap_ * V);
    }

    void cleanup() {
        if (dspark_) { dspark_->free_all(); dspark_.reset(); }
        if (runner_) { runner_->free_all(); runner_.reset(); }
        if (d_sconv_snap_) { cudaFree(d_sconv_snap_); d_sconv_snap_ = nullptr; }
        if (d_test_steal_) { cudaFree(d_test_steal_); d_test_steal_ = nullptr; }
        if (d_h_sel_)      { cudaFree(d_h_sel_); d_h_sel_ = nullptr; }
        if (d_sel_idx_)    { cudaFree(d_sel_idx_); d_sel_idx_ = nullptr; }
        if (h_logits_pin_) { cudaFreeHost(h_logits_pin_); h_logits_pin_ = nullptr; h_logits_cap_ = 0; }
        topk_.free_all();
        model_.free_all();
        prompt_ckpts_.clear();
        shared_ckpts_.clear();
        initialized_ = false;
    }

    struct PromptCkpt {
        bool valid = false;
        bool has_dspark = false;
        int seq_len = 0;
        int swa_start = 0;
        int swa_len = 0;
        std::vector<float> sconv;
        std::vector<uint8_t> swa_k, swa_v, swa_ksc, swa_vsc;
        std::vector<uint8_t> dspark_k, dspark_v;
        std::vector<float> dspark_frnn;
    };

    bool initialized_ = false;
    int gpu_id_ = 0;
    int max_slots_ = 1;
    int eff_max_ctx_ = 4096;
    // The drafter's window shape: --draft-beam / --draft-cost / --draft-plain (or STRATA_INK_SPEC_BEAM / _COST / _PLAIN), STRATA_INK_SPEC_LOG=1 prints stats
    struct SpecCfg { int beam = 4; float cost = 4.2f; float plain = 0.0f; bool log = false; } spec_cfg_;
    bool kv_grow_failed_ = false;    // the last forward_batch ended only because the global K/V could not grow
    int eff_chunk_run_ = 0;          // the prefill chunk the runner really got (after auto / halve-on-fail)
    int dspark_slot_ = -1;
    KvDtype kv_dtype_ = KvDtype::FP16;
    std::string dev_info_;

    strata::ink::InkCudaModel model_;
    std::unique_ptr<strata::ink::InkCudaRunner> runner_;
    std::unique_ptr<strata::ink::InkDSpark> dspark_;
    float* d_sconv_snap_ = nullptr;
    void* d_test_steal_ = nullptr;   // STRATA_INK_TEST_STEAL_MIB test hook
    float* d_h_sel_ = nullptr;
    int32_t* d_sel_idx_ = nullptr;
    float* h_logits_pin_ = nullptr;
    strata::ink::InkTopkScratch topk_;   // candidate rows (ink_topk.cuh)
    int h_logits_cap_ = 0;
    std::vector<bool> slot_alloc_;
    std::vector<bool> snap_active_;
    std::vector<int> snap_p0_;
    std::vector<int> snap_M_;
    std::vector<PromptCkpt> prompt_ckpts_;
    std::vector<PromptCkpt> shared_ckpts_;
    bool tree_snap_ready_ = false;
    int tree_snap_p0_ = 0;
    int tree_snap_M_ = 0;
    int32_t tree_snap_path_[strata::ink::kInkMaxTree] = {};
};

std::unique_ptr<InkBackend> make_cuda_backend() {
    return std::make_unique<InkCudaBackend>();
}

bool ink_cuda_q8f_supported(int gpu_index) {
    int major = 0, minor = 0;
    const int dev = gpu_index < 0 ? 0 : gpu_index;
    if (cudaDeviceGetAttribute(&major, cudaDevAttrComputeCapabilityMajor, dev) != cudaSuccess ||
        cudaDeviceGetAttribute(&minor, cudaDevAttrComputeCapabilityMinor, dev) != cudaSuccess) {
        cudaGetLastError();
        return false;
    }
    // The int8 tensor-core dense GEMM wins at one token on consumer Ampere (sm_86: +7..10% decode on an RTX A6000) and loses on the cards whose
    // memory system the BF16 GEMV already saturates: RTX 4090 (sm_89) 95.4 against 103.7 tok/s at C1 (UD-Q4_K_XL, 8 slots equal), RTX 5090 (sm_120)
    // 98 against 133. sm_80 (A100) is unmeasured and stays on BF16. The terminal planner (inkling_fit.py, Q8_DENSE_SMS) uses the same set.
    return major == 8 && minor == 6;
}

}  // namespace ink2
