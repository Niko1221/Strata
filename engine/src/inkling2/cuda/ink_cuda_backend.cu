// engine/src/inkling2/cuda/ink_cuda_backend.cu - InkBackend CUDA Plugin for `strata-inkling` (sm_70..sm_121a)
//
// MIT License
// Copyright (c) 2026 Niko1221 and the Strata contributors
#include "ink_backend.h"
#include "ink_batch.h"
#include "ink_manifest.h"

#include "ink_cuda.cuh"
#include "ink_dspark.cuh"
#include "ink_rtx5090_bspec.cuh"
#include "ink_runner.cuh"
#include "ink_spec.cuh"
#include "ink_topk.cuh"
#include "ink_vision_cuda.cuh"
#include "ink_vision_policy.h"
#include "strata/core/arch_probe.hpp"
#include "strata/core/ink_chunk.hpp"
#include "strata/core/ink_draft_rule.hpp"
#include "strata/core/volta_forever.hpp"

#include <algorithm>
#include <chrono>
#include <filesystem>
#include <cmath>
#include <cstdio>
#include <cstring>
#include <memory>
#include <string>
#include <unordered_map>
#include <vector>

namespace strata::ink {

// PERF-STEP: the INT8-screened exact head (ink_head_screen.cuh): STRATA_INK_HEAD_SCREEN=0/1 decides; unset = on in the fast set of a card with the FP8 trunk cache (the 5090 row), off in exact mode and on every other card
static bool ink_head_screen_wanted(int gpu_id) {
    const char* e = std::getenv("STRATA_INK_HEAD_SCREEN");
    if (e != nullptr && e[0] != 0) return e[0] == '1';
    const auto& c = strata::ink::sm120_cfg(gpu_id);
    return c.fast && c.fp8_cache;
}

// Host buffers of the prompt checkpoints (PREFILL lane): page-locked, so the device-to-host copies of save_checkpoint are real DMA copies (the ~85 MB of SWA windows
// of a long prompt took ~24 ms through pageable memory: staged copies, a zero-fill of the vectors and page faults, in the time to the first token). Falls back to
// ordinary memory when the page-locked allocation fails. The allocator default-initialises (no zero-fill on resize): every byte is written by a copy before it is read.
template <class T>
struct PinnedAlloc {
    using value_type = T;
    PinnedAlloc() = default;
    template <class U> PinnedAlloc(const PinnedAlloc<U>&) noexcept {}
    T* allocate(size_t n) {
        void* p = nullptr;
        if (n == 0) return nullptr;
        if (cudaHostAlloc(&p, n * sizeof(T), cudaHostAllocPortable) != cudaSuccess) {
            cudaGetLastError();
            p = ::operator new(n * sizeof(T));
        }
        return static_cast<T*>(p);
    }
    void deallocate(T* p, size_t) noexcept {
        if (!p) return;
        if (cudaFreeHost(p) != cudaSuccess) { cudaGetLastError(); ::operator delete(p); }
    }
    template <class U> void construct(U*) noexcept {}
    template <class U, class... A> void construct(U* p, A&&... a) { ::new (static_cast<void*>(p)) U(std::forward<A>(a)...); }
    template <class U> bool operator==(const PinnedAlloc<U>&) const noexcept { return true; }
    template <class U> bool operator!=(const PinnedAlloc<U>&) const noexcept { return false; }
};
template <class T> using PinnedVec = std::vector<T, PinnedAlloc<T>>;

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

// ---------------------------------------------------------------------------------------------------------------------------------------------
// VRAM sizing, in ONE place.
//   plan_vram_reserve()   before a single expert is placed: the K/V type, and how much of the card is set aside for the K/V, the ShortConv state and
//                         the workspaces (min_reserve_bytes, handed to the model loader that then places the experts around it);
//   plan_prefill_chunk()  once the experts are in: the prefill chunk, from the memory that is left.
// Nothing else in this file sizes memory. The long-context lane (LCTX: "K/V before experts") changes the placement order by changing these two
// functions and nothing else. With every RTX 50 switch off both return the shipped numbers, bit for bit (the 5060 Ti hand-over's changes to them
// moved the K/V split on a 16 GB sm_86 card: review finding B1; they apply only through the table row now).
// ---------------------------------------------------------------------------------------------------------------------------------------------
struct VramPlanIn {
    const BackendOpts* opts = nullptr;
    int gpu_id = 0;
    int sm_cc = 0;
    size_t total_vram = 0;
    int max_slots = 1;
    int kv_offload_mode = 0;    // 1 on, -1 off, 0 auto
    bool build_screen = false;  // the DSpark drafter will be built
    bool compact_gguf = false;  // a GGUF with the compact trunk (GSQ-RCO file): on a card of 13 GB or less the prefill chunk is 256 so that every expert fits (the plan with 512 left 212 on the host)
    const strata::ink::Sm120Cfg* sm120 = nullptr;
    const strata::ink::InkRtx5060TiCfg* c5060 = nullptr;
};

struct VramPlan {
    KvDtype kv_dtype = KvDtype::FP16;
    strata::ink::InkKvDtype cuda_kvd = strata::ink::InkKvDtype::FP16;
    int eff_chunk = 64;
    size_t swa_r = 0;
    size_t per_visit_pos = 0;
    bool c64_host_kv = false;
    int init_glob = 0;
    size_t min_kv_bytes = 0;
    size_t dspark_pad = 0;
    size_t ws_pad = 0;
    size_t min_reserve_bytes = 0;
};

static VramPlan plan_vram_reserve(const VramPlanIn& in) {
    const BackendOpts& opts = *in.opts;
    const auto& opt5060 = *in.c5060;
    VramPlan p;
    // Keep lossless FP16 KV as default when Host KV Offload handles capacity (#552)!
    p.kv_dtype = opts.kv_dtype;
    if (opts.kv_dtype_auto) {
        if (in.max_slots >= 16) {
            p.kv_dtype = KvDtype::FP8_E4M3;
        } else if (in.kv_offload_mode < 0) {
            if (in.total_vram <= (13ull << 30)) {
                p.kv_dtype = KvDtype::FP8_E4M3;
            } else if (in.sm_cc >= 89 && opts.max_ctx > 16384 && in.total_vram < (40ull << 30)) {
                p.kv_dtype = KvDtype::FP8_E4M3;
            }
        }
    }
    if (opt5060.c64_spec && in.max_slots >= 16 && opts.kv_dtype_auto) {
        p.kv_dtype = KvDtype::FP8_E4M3;
    } else if (opt5060.compact_vram && opt5060.compact_kv && in.max_slots == 1 && opts.kv_dtype_auto && strata::ink::ink_elastic_ram_cap_bytes() == 0) {
        p.kv_dtype = KvDtype::FP8_E4M3;
    } else if (strata::ink::ink_elastic_1m_enabled(in.gpu_id) && opts.kv_dtype_auto) {
        auto dt = strata::ink::ink_elastic_select_kv_dtype(
            p.kv_dtype == KvDtype::FP8_E4M3 ? strata::ink::InkKvDtype::FP8 : strata::ink::InkKvDtype::FP16,
            true, opts.max_ctx, in.max_slots, strata::ink::ink_ram_available_bytes(), in.gpu_id);
        if (dt == strata::ink::InkKvDtype::FP8) p.kv_dtype = KvDtype::FP8_E4M3;
        else if (dt == strata::ink::InkKvDtype::K8V4) p.kv_dtype = KvDtype::K8V4;
    }
    auto map_kvd = [](KvDtype d) -> strata::ink::InkKvDtype {
        if (d == KvDtype::FP8_E4M3) return strata::ink::InkKvDtype::FP8;
        if (d == KvDtype::K8V8)     return strata::ink::InkKvDtype::K8V8;
        if (d == KvDtype::K8V4)     return strata::ink::InkKvDtype::K8V4;
        return strata::ink::InkKvDtype::FP16;
    };
    p.cuda_kvd = map_kvd(p.kv_dtype);

    const bool gguf_small_card = in.compact_gguf && in.max_slots == 1 && opts.chunk_auto && in.total_vram <= (13ull << 30);
    p.eff_chunk = ((opt5060.prefill_lm && in.max_slots == 1 && opts.chunk_auto) || gguf_small_card) ? 256 : std::max(64, opts.chunk);
    p.swa_r = (in.max_slots >= 16) ? (size_t)(strata::ink::kInkSwaWin + 32)
                                   : (size_t)(strata::ink::kInkSwaWin + std::max(p.eff_chunk, 64));
    p.per_visit_pos = strata::ink::InkCudaRunner::kv_bytes_per_visit_pos(p.cuda_kvd);
    p.c64_host_kv = (opt5060.c64_spec && in.max_slots >= 16);
    p.init_glob = (in.kv_offload_mode == 1 || p.c64_host_kv)
        ? 0
        : (opt5060.pack && in.max_slots == 1)
            ? std::min(std::max(256, opts.max_ctx), 2048)
            : ((in.kv_offload_mode == 0 && in.total_vram <= (13ull << 30))
                ? std::min(std::max(256, opts.max_ctx), 2048)
                : std::max(256, std::min(std::max(256, opts.max_ctx), (in.max_slots == 1) ? 8192 : 4096)));
    p.min_kv_bytes = (p.c64_host_kv ? 0ull : (size_t)40 * in.max_slots * p.swa_r * p.per_visit_pos)
                   + (size_t)strata::ink::kInkVisits * in.max_slots * 3 * strata::ink::kInkSconvCh * 4
                   + (size_t)8 * in.max_slots * p.init_glob * p.per_visit_pos;
    // Include O(1) prefill ws (~114 MiB) + DSpark (~224 MiB when enabled) + activation ws (WDDM floor added inside loader!)
    // Note: unembed_screen (441 MiB) is allocated INSIDE load_from_model_file BEFORE expert packing.
    p.dspark_pad = in.build_screen ? (opt5060.pack ? (540ull << 20) : (224ull << 20)) : 0ull;
    // STRATA_INK_DRAFTER=dflash2: the DFlash 2 drafter's weights (Q8 + bf16) and scratch, ~480 MiB more than DSpark's buffers, are held back
    if (in.build_screen && strata::ink::InkDFlash2::requested()) p.dspark_pad += (strata::ink::sm120_cfg(in.gpu_id).fast ? (strata::ink::df2_total_mib(strata::ink::df2_variant(in.total_vram < (24ull << 30)), in.max_slots) - 200ull) : (strata::ink::df2_f32_path() ? strata::ink::df2_total_mib(strata::ink::df2_variant(in.total_vram < (24ull << 30)), in.max_slots) - 100ull : 42ull * (size_t)std::min(in.max_slots, 8))) << 20;   // NV-OLDER (below sm_80): the whole drafter (FP32 matrices, block weights, scratch, rings), as the RTX 50 fast branch counts it: with 4 slots the planned K/V + workspace side exceeds the 1850 MiB figure below, and the drafter then took the last MiB (2080 Ti, --parallel 4: out of memory at the first request)   // the compact (pack) loader leaves DSpark's weights off the card
    p.ws_pad = opt5060.pack
        ? ((in.max_slots >= 16 ? (700ull << 20) : (420ull << 20)) + p.dspark_pad)
        : (((in.max_slots >= 8) ? (1280ull << 20)
           : ((in.total_vram >= (20ull << 30)) ? (960ull << 20)
            : (in.total_vram <= (13ull << 30)) ? (512ull << 20)
                                               : (832ull << 20))) + p.dspark_pad);
    // LCTX (PERF-LONGCTX), "K/V before experts" (STRATA_INK_KV_FIRST=1, off by default): on a card whose experts do not all fit, the K/V of the WHOLE chosen context, the
    // DSpark state of every context token and a growth margin are set aside before the experts get the rest, so the global K/V stays in VRAM (no host tail: 64K 9.9 -> 28.5 tok/s,
    // 128K 5.6 -> 23.0 on a 5060 Ti 16 GB, f-lctx RESULTS-LCTX.md section 3). The rule keeps at least 1.5 GiB for the experts; when the K/V does not fit with that, the shipped
    // window rule stays (a partial window is never planned). kv_offload on keeps its host K/V; one slot only.
    if (in.kv_offload_mode != 1 && in.max_slots == 1 && !p.c64_host_kv && in.total_vram <= (20ull << 30) && strata::ink::sm120_cfg_detail::env_get("STRATA_INK_KV_FIRST") &&
        strata::ink::sm120_cfg_detail::env_get("STRATA_INK_KV_FIRST")[0] == '1') {
        const size_t ctx = (size_t)std::max(256, opts.max_ctx);
        const size_t glob_all = (size_t)8 * ctx * p.per_visit_pos;
        const size_t glob_rest = (size_t)8 * (ctx > (size_t)p.init_glob ? ctx - (size_t)p.init_glob : 0) * p.per_visit_pos;
        const size_t dspark_ctx = in.build_screen ? ctx * 4096ull : 0ull;
        const size_t margin = glob_all / 8 + (256ull << 20);
        const size_t kv_extra = glob_rest + dspark_ctx + margin;
        // what the card keeps for everything else, from LCTX's measured rule: dense trunk 3.67 GB, driver 1.30, hybrid workspace 0.85 (decimal GB), experts floor 1.5 GiB
        const size_t fixed = (size_t)((3.67 + 1.30 + 0.85) * 1e9);
        const size_t floor_experts = 3ull << 29;
        if (in.total_vram > fixed + p.min_kv_bytes + p.ws_pad + kv_extra + floor_experts) {
            p.min_kv_bytes += kv_extra;
            std::fprintf(stderr, "[InkCUDA] STRATA_INK_KV_FIRST: %.2f GiB reserved for the K/V of the whole %zu-token context (+ DSpark state and growth margin) before the experts\n",
                         (double)kv_extra / 1073741824.0, ctx);
        } else {
            std::fprintf(stderr, "[InkCUDA] STRATA_INK_KV_FIRST: the %zu-token context does not fit next to at least 1.5 GiB of experts on this card; the shipped placement stays\n", ctx);
        }
    }
    p.min_reserve_bytes = p.min_kv_bytes + p.ws_pad;
    // DFlash 2 on a card with hybrid experts (not the compact 16 GB mode): hold back what its weights, rings and graphs need, as --vram-reserve-mib 1850 does
    if (in.build_screen && !strata::ink::sm120_cfg(in.gpu_id).fast && strata::ink::InkDFlash2::requested()) p.min_reserve_bytes = std::max<size_t>(p.min_reserve_bytes, (size_t)(1850.0 + strata::ink::df2_big_mib(strata::ink::df2_variant(in.total_vram < (24ull << 30))) - strata::ink::df2_big_mib(strata::ink::Df2Variant{2, 2})) << 20);
    // RTX 50 fast path (the big-card rows, not the 16 GB compact mode, which has its own precheck with the same margin): the start must end with a crash margin free. The reserve above
    // is what the K/V, workspaces and drafter are PLANNED to take; the real runner, K/V and DFlash 2 took ~370 MiB more, which left 24-32 MiB free on a 5090 whose other programs held
    // 11.8-12.7 GB, and the first graph capture / top-k / tier swap then died (CUDA error 1 or out of memory). The old path keeps its sizing (same memory as main).
    if (strata::ink::sm120_cfg(in.gpu_id).fast && !opt5060.compact_vram && !opt5060.pack && in.max_slots <= 8) {
        p.min_reserve_bytes += (size_t)256 << 20;
    }
    if (opts.vram_reserve_mib >= 0) {
        p.min_reserve_bytes = std::max(p.min_reserve_bytes, (size_t)opts.vram_reserve_mib << 20);
    }
    return p;
}

// The prefill chunk once the experts are placed. `led_loaded` = free MiB after load. Prints the "[InkCUDA] prefill chunk" line when it was decided
// from the free memory (as the shipped code did).
static int plan_prefill_chunk(const VramPlanIn& in, const VramPlan& p, bool all_sf4_vram, bool ls_active, double led_loaded) {
    const BackendOpts& opts = *in.opts;
    const auto& opt5060 = *in.c5060;
    int chunk_try = std::max(64, opts.chunk);
    if (opt5060.compact_vram && all_sf4_vram && opts.chunk_auto) {
        if (in.max_slots == 1) {
            chunk_try = 256;
            if (const char* ec = strata::ink::sm120_cfg_detail::env2("STRATA_INK_SM120_CHUNK", "STRATA_INK_5060TI_CHUNK")) {
                const int v = std::atoi(ec);
                if (v >= 64 && v <= 4096) chunk_try = v;
            }
        } else if (in.max_slots >= 16) {
            chunk_try = std::min(512, std::max(192, in.max_slots * (opt5060.c64_draft_k + 1)));
            if (const char* ec = strata::ink::sm120_cfg_detail::env2("STRATA_INK_SM120_CHUNK", "STRATA_INK_5060TI_CHUNK")) {
                const int v = std::atoi(ec);
                if (v >= 64 && v <= 4096) chunk_try = v;
            }
        }
    } else if (opt5060.prefill_lm && !all_sf4_vram && in.max_slots == 1 && opts.chunk_auto) {
        chunk_try = 256;
    }
    const bool gguf_small_card = in.compact_gguf && in.max_slots == 1 && opts.chunk_auto && in.total_vram <= (13ull << 30);
    if (gguf_small_card) chunk_try = 256;
    // No --chunk: the memory left after the experts decides (strata/core/ink_chunk.hpp); STRATA_INK_CHUNK_AUTO=0 keeps the old 512.
    if (opts.chunk_auto && !ls_active && !opt5060.prefill_lm && !gguf_small_card) {
        const char* auto_env = std::getenv("STRATA_INK_CHUNK_AUTO");
        if (!(auto_env && std::atoi(auto_env) == 0)) {
            // the K/V still to be allocated for every slot to reach the context stays free of the chunk's working memory (kv offload on: it is host RAM)
            const double kv_grow_mib = (in.kv_offload_mode == 1) ? 0.0
                : (double)8 * in.max_slots * std::max(0, std::max(256, opts.max_ctx) - p.init_glob) * (double)p.per_visit_pos / 1048576.0;
            chunk_try = std::max(chunk_try, strata::core::ink_auto_chunk(led_loaded, in.max_slots, kv_grow_mib));
            const auto& s120 = *in.sm120;
            if (s120.fast && s120.c1_chunk_big > 0 && in.max_slots == 1 && (led_loaded - kv_grow_mib) >= (double)s120.c1_chunk_big_headroom_mib) {
                chunk_try = std::max(chunk_try, s120.c1_chunk_big);
            }
            std::fprintf(stderr, "[InkCUDA] prefill chunk %d (auto: %.0f MiB free after the experts, %.0f MiB of K/V still to grow, %d slot(s); --chunk N sets it)\n",
                         chunk_try, led_loaded, kv_grow_mib, in.max_slots);
        }
    }
    return chunk_try;
}

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

    bool set_tail_skip(const std::string& spec, std::string& err) override {
        if (!initialized_) { err = "CUDA backend not initialized"; return false; }
        strata::ink::InkTailCfg c;
        if (!strata::ink::ink_tail_parse(spec, c, err)) return false;
        (void)strata::ink::ink_tail_cfg();   // read the environment first, so it cannot overwrite this later
        if (c.stats && !strata::ink::ink_tail_cnt_dev()) {
            void* p = nullptr;
            if (cudaMalloc(&p, sizeof(unsigned long long) * strata::ink::kInkTailVisits * strata::ink::kInkTailStatW) != cudaSuccess) { err = "no memory for the tail-skip counters"; return false; }
            cudaMemset(p, 0, sizeof(unsigned long long) * strata::ink::kInkTailVisits * strata::ink::kInkTailStatW);
            strata::ink::ink_tail_cnt_dev() = static_cast<unsigned long long*>(p);
        }
        cudaStreamSynchronize(runner_->st);
        strata::ink::ink_tail_cfg_ref() = c;
        if (c.on) strata::ink::ink_tail_sync_tables(c);   // load the tables now, outside any graph capture
        // the CUDA graphs of the one-token step carry the old arguments: drop them so the next step captures the new ones
        for (auto& e : runner_->graph_cache) if (e.exec) cudaGraphExecDestroy(e.exec);
        runner_->graph_cache.clear();
        return true;
    }

    bool tail_calib(std::vector<double>& sum, std::vector<uint64_t>& cnt, bool reset) override {
        auto& t = strata::ink::ink_tail_dev();
        sum.assign((size_t)24 * strata::ink::kInkHidden, 0.0);
        cnt.assign(24, 0);
        if (!initialized_ || !t.acc) return false;
        cudaStreamSynchronize(runner_->st);
        if (cudaMemcpy(sum.data(), t.acc, sum.size() * sizeof(double), cudaMemcpyDeviceToHost) != cudaSuccess) return false;
        if (cudaMemcpy(cnt.data(), t.cnt, 24 * sizeof(uint64_t), cudaMemcpyDeviceToHost) != cudaSuccess) return false;
        if (reset) { cudaMemset(t.acc, 0, sum.size() * sizeof(double)); cudaMemset(t.cnt, 0, 24 * sizeof(uint64_t)); }
        return true;
    }

    bool tail_stats(std::vector<uint64_t>& out, bool reset) override {
        out.assign((size_t)strata::ink::kInkTailVisits * strata::ink::kInkTailStatW, 0);
        unsigned long long* d = strata::ink::ink_tail_cnt_dev();
        if (!initialized_ || !d) return false;
        cudaStreamSynchronize(runner_->st);
        if (cudaMemcpy(out.data(), d, out.size() * sizeof(unsigned long long), cudaMemcpyDeviceToHost) != cudaSuccess) return false;
        if (reset) cudaMemset(d, 0, out.size() * sizeof(unsigned long long));
        return true;
    }

    std::string device_info() const override { return dev_info_; }

    size_t primary_device_bytes() const override { return primary_mem_bytes_; }

    // The GPU picture reader (ink_vision_cuda.cuh).  It is put on the card with the first picture and, with one GPU, taken off again after
    // vision_idle_s seconds without one (Qwen's elastic reader); on a machine whose GPU shares the system's RAM (DGX Spark), with several GPUs or with
    // --vision-idle-s 0 it stays once loaded.  Its room is part of the memory held back from the experts at start (init), so a card does not run out.
    bool encode_image(const ImageRec& img, std::vector<float>& out_embeds, std::string& err) override {
        if (!initialized_ || !vit_planned_ || vit_path_.empty()) return false;
        cudaSetDevice(gpu_id_);
        if (!vit_) vit_ = std::make_unique<strata::ink::InkVisionCuda>();
        if (!vit_->loaded()) {
            const double t0 = strata::ink::now_s();
            if (!vit_->load(vit_path_, gpu_id_, err)) { vit_->release(); return false; }
            std::fprintf(stderr, "[InkCUDA] picture reader put on the card: %.0f MiB of weights in %.2f s (%s)\n", (double)vit_->weight_bytes() / 1048576.0,
                         strata::ink::now_s() - t0, vit_->precision() == strata::ink::VitPrec::TC2 ? "BF16 tensor cores, two-part activation" :
                         (vit_->precision() == strata::ink::VitPrec::BF16 ? "BF16 tensor cores" : "FP32"));
        }
        const bool ok = vit_->encode(img, out_embeds, err);
        vit_last_ = std::chrono::steady_clock::now();
        if (!ok) vit_->release();    // a picture that did not fit: give everything back, the CPU reader reads it
        return ok;
    }
    int idle_release_in_ms() const override {
        if (!vit_ || !vit_->loaded() || !vit_elastic_) return -1;
        const double left = vit_idle_s_ * 1000.0 - std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - vit_last_).count();
        return left <= 0.0 ? 0 : (int)left + 1;
    }
    void idle_release() override {
        if (!vit_ || !vit_->loaded()) return;
        const double mib = (double)vit_->weight_bytes() / 1048576.0;
        vit_->release();
        std::fprintf(stderr, "[InkCUDA] picture reader taken off the card after %.0f s without a picture (%.0f MiB given back)\n", vit_idle_s_, mib);
    }

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
        strata::ink::df2_model_path() = mf.path;   // where DFlash 2 looks for its sidecar: needed by the reserves below as well
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
        {
            // the card's table row (RTX 50 fast mode); every flag is false and every number the shipped one when no switch is set
            const auto& s120 = strata::ink::sm120_cfg(gpu_id_);
            track_pld90_ = s120.dspark_opt;
            if (s120.dspark_opt && !s120.exact) {   // the row's drafting numbers (beam, cost, plain margin) are FAST mode only: the exact form keeps off's drafting policy
                if (!std::getenv("STRATA_INK_SPEC_BEAM") && opts.spec_beam <= 0) spec_cfg_.beam = s120.spec_beam;
                if (!std::getenv("STRATA_INK_SPEC_COST") && opts.spec_cost <= 0.0) spec_cfg_.cost = s120.spec_cost;
                if (!std::getenv("STRATA_INK_SPEC_PLAIN") && opts.spec_plain < 0.0) spec_cfg_.plain = s120.spec_plain;
            }
        }
        {   // the opt-in tail skip (environment): its tables are loaded before anything is captured into a CUDA graph (the runner may capture while it initialises)
            const strata::ink::InkTailCfg& tc0 = strata::ink::ink_tail_cfg();
            if (tc0.on) strata::ink::ink_tail_sync_tables(tc0);
        }
        cudaDeviceProp prop{};
        ce = cudaGetDeviceProperties(&prop, gpu_id_);
        if (ce != cudaSuccess) {
            err = std::string("cudaGetDeviceProperties failed: ") + cudaGetErrorString(ce);
            return false;
        }
        const int sm_cc = prop.major * 10 + prop.minor;
        dev_cc_ = sm_cc;
        if (sm_cc < 80) strata::ink::df2_f32_path() = true;   // NV-OLDER: DFlash 2 runs its FP32 / dp4a form below sm_80 (sizing below needs to know)

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
        strata::ink::df2_small_card() = prop.totalGlobalMem < (24ull << 30);
        strata::ink::df2_parallel() = max_slots_;   // DFlash 2 as the default drafter only up to 16 slots (see ink_dflash2_path.cuh)
        const int kv_offload_mode = (opts.kv_offload == "on") ? 1 : ((opts.kv_offload == "off") ? -1 : 0);
        // all of the sizing below is plan_vram_reserve() / plan_prefill_chunk() (top of this file)
        VramPlanIn plan_in;
        plan_in.opts = &opts;
        plan_in.gpu_id = gpu_id_;
        plan_in.sm_cc = sm_cc;
        plan_in.total_vram = prop.totalGlobalMem;
        plan_in.max_slots = max_slots_;
        plan_in.kv_offload_mode = kv_offload_mode;
        plan_in.build_screen = (opts.enable_dspark && ((mf.m.dspark.present && strata::ink::df2_dspark_selected()) || strata::ink::df2_replaces_dspark(mf.path))) || strata::ink::ink_head_screen_wanted(gpu_id_);   // DFlash 2 (a sidecar) also needs the screen; PERF-STEP: the head screen needs the INT8 index of the unembedding
        plan_in.compact_gguf = mf.m.embed_host_tail && !(std::getenv("STRATA_INK_GGUF_CHUNK256") && std::getenv("STRATA_INK_GGUF_CHUNK256")[0] == '0');
        plan_in.sm120 = &strata::ink::sm120_cfg(gpu_id_);
        plan_in.c5060 = &strata::ink::ink_rtx5060ti_cfg(gpu_id_);
        // RESIDENCY FIRST (strata/core/ink_draft_rule.hpp): DFlash 2 as the DEFAULT drafter is not loaded when its weights and rings would move the model from full residency to a hybrid
        // load, or cut the expert tier by 10% or more. The two plans (with and without the drafter) are compared against the free VRAM, the trunk and the expert section the loader will see.
        // `--draft on`, STRATA_INK_DFLASH2_FORCE=1 and STRATA_INK_DRAFTER=dflash2 ask for it by name and are never vetoed. The COMPACT path has its own accounting (its precheck) and is skipped.
        auto plan_for = [&](bool veto) {
            strata::ink::df2_residency_veto() = veto;
            plan_in.build_screen = (opts.enable_dspark && ((mf.m.dspark.present && strata::ink::df2_dspark_selected()) || strata::ink::df2_replaces_dspark(mf.path))) || strata::ink::ink_head_screen_wanted(gpu_id_);
            return plan_vram_reserve(plan_in);
        };
        draft_forced_ = opts.draft_forced || strata::ink::df2_forced_by_env();
        VramPlan plan = plan_for(false);
        // below sm_80 (ink_draft_rule.hpp principle 1b) the rule also covers a .gguf: its expert bytes are estimated from the file (minus the dense trunk: 1.4 GiB compact, 3.4 GiB otherwise)
        const bool legacy_arch = sm_cc < 80;
        if (!draft_forced_ && opts.enable_dspark && strata::ink::df2_drafter_env().empty() && (mf.kind == ModelSourceKind::Strata || (legacy_arch && mf.kind == ModelSourceKind::Gguf)) &&
            !plan_in.sm120->compact && strata::ink::df2_replaces_dspark(mf.path)) {
            const VramPlan plan_no = plan_for(true);
            size_t vf = 0, vt = 0;
            if (cudaMemGetInfo(&vf, &vt) == cudaSuccess && vt > 0) {
                size_t trunk = 0, sec4 = 0;
                if (mf.kind == ModelSourceKind::Strata) {
                    trunk = (size_t)mf.rd.header().sections[1].byte_length;
                    sec4 = (size_t)mf.rd.header().sections[4].byte_length;
                } else {
                    std::error_code fec;
                    const size_t fsz = (size_t)std::filesystem::file_size(mf.path, fec);
                    trunk = (mf.m.embed_host_tail ? 1400ull : 3400ull) << 20;
                    sec4 = (fec || fsz <= trunk) ? 0u : fsz - trunk;
                }
                const size_t wf = strata::ink::InkCudaModel::wddm_safety_floor(vt);
                auto reserve_of = [&](const VramPlan& p) {
                    const size_t base = ((vt >= (20ull << 30)) ? (1600ull << 20) : (vt <= (13ull << 30)) ? (960ull << 20) : (1280ull << 20)) + (wf > (512ull << 20) ? (wf - (512ull << 20)) : 0ull);
                    return std::max(base, p.min_reserve_bytes + wf);
                };
                const size_t r_with = reserve_of(plan), r_no = reserve_of(plan_no);
                const size_t avail_no = (vf > trunk + r_no) ? (vf - trunk - r_no) : 0u, avail_with = (vf > trunk + r_with) ? (vf - trunk - r_with) : 0u;
                strata::core::InkDraftLoadIn li;
                li.below_sm80 = legacy_arch;
                li.sm60 = (sm_cc == 60);   // the Tesla P100 keeps the default drafter on a hybrid load (ink_draft_rule.hpp, 1b)
                li.full_without = avail_no >= sec4;
                li.full_with = avail_with >= sec4;
                li.tier_budget_mib = (double)avail_no / 1048576.0;
                li.drafter_mib = (double)(r_with > r_no ? r_with - r_no : 0) / 1048576.0;
                const strata::core::InkDraftLoadOut lo = strata::core::ink_draft_load_rule(li);
                if (lo.residency_veto) {
                    std::fprintf(stderr, "%s\n", strata::core::ink_draft_veto_line("[InkCUDA]", "DFlash 2", (double)sec4 / 1073741824.0, (double)avail_with / 1073741824.0, li.drafter_mib,
                                                                                    (double)avail_no / 1073741824.0).c_str());
                    plan = plan_no;                      // the veto stays set: no drafter pad in the reserve below
                } else {
                    plan = plan_for(false);
                }
            } else {
                plan = plan_for(false);
            }
        }
        kv_dtype_ = plan.kv_dtype;
        const auto cuda_kvd = plan.cuda_kvd;
        const bool build_screen = plan_in.build_screen;
        size_t min_reserve_bytes = plan.min_reserve_bytes;
        const auto& opt5060 = *plan_in.c5060;
        // The GPU picture reader (ink_vision_policy.h: 24 GB or more, or --vision-reader gpu): its weights and the working buffers of a 1,024-token picture
        // come out of the experts' budget like the KV and the workspace do, so the card is not filled to the last byte before the first picture.
        primary_mem_bytes_ = prop.totalGlobalMem;
        vit_planned_ = false;
        vit_elastic_ = false;
        vit_path_.clear();
        vit_idle_s_ = ink2::vision_idle_seconds(opts.vision_idle_s);
        if (!opts.vision_path.empty()) {
            const ink2::VisionPlan vp = ink2::plan_vision_reader(ink2::vision_ask_from(opts.vision_reader), "cuda", (double)prop.totalGlobalMem / 1073741824.0);
            ink2::VisionShardInfo vinfo;
            std::string verr0;
            if (vp.gpu && !ink2::vision_probe(opts.vision_path, vinfo, verr0)) {
                std::fprintf(stderr, "[InkCUDA] picture reader not planned: %s\n", verr0.c_str());
            } else if (vp.gpu) {
                vit_planned_ = true;
                vit_path_ = opts.vision_path;
                const bool shares_ram = prop.integrated != 0;
                vit_elastic_ = (all_gpus.size() == 1) && !shares_ram && opts.vision_idle_s != 0;
                const size_t room = (size_t)vinfo.sec1_len + strata::ink::InkVisionCuda::activation_bytes(
                                        vinfo.c, 1024, strata::ink::InkVisionCuda::default_precision(sm_cc)) + (64ull << 20);
                min_reserve_bytes += room;
                std::fprintf(stderr, "[InkCUDA] picture reader on the card: %.0f MiB held back from the experts (weights + a 1,024-token picture); %s\n",
                             (double)room / 1048576.0,
                             vit_elastic_ ? "it is put on the card with the first picture and taken off after the idle time" : "it stays on the card once loaded");
            }
        }

        strata::ink::sm120_plan_slots_ref().store(max_slots_);
        // Load .strata v1.1 / .gguf (Unsloth + ggml-org) / HF safetensors trunk + experts (Single-GPU, EP2, Layer-Split, or 4-tier GPU+RAM+SSD!)
        if (!model_.load_from_model_file(mf, gpu_id_, peer_gpu_id, (float)opts.ep_split, build_screen, err,
                                         min_reserve_bytes, gguf_detail::gguf_dequant_row_f32, gguf_detail::gguf_row_bytes,
                                         opts.vram_experts_gb, opts.ram_experts_gb, opts.ssd_experts,
                                         all_gpus, opts.layer_split, opts.threads)) {
            return false;
        }

        {   // --turbo / --turbo-max (STRATA_INK_TAIL_SKIP): the RTX 50 fast-path MoE kernels (the 5090's, and the 5060 Ti's all-VRAM COMPACT kernels) do not carry the tail-skip arguments:
            // say so once instead of being silently ineffective (the old path, STRATA_INK_SM120=0, runs the tail skip)
            const strata::ink::InkTailCfg& tcn = strata::ink::ink_tail_cfg();
            const auto& sxn = strata::ink::sm120_cfg(gpu_id_);
            if (tcn.on && ((sxn.moe_opt && !sxn.exact) || (model_.all_sf4_vram && sxn.compact))) {
                std::fprintf(stderr, "[InkCUDA] --turbo / STRATA_INK_TAIL_SKIP has no effect on this card's RTX 50 fast path; STRATA_INK_SM120=0 runs the tail skip\n");
            }
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
        int chunk_try = plan_prefill_chunk(plan_in, plan, model_.all_sf4_vram, model_.ls_active, led_loaded);
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
        if (opt5060.prefill_lm && max_slots_ == 1 && runner_->can_layer_major_prefill()) {
            eff_chunk_run_ = 8704;
            std::fprintf(stderr, "[Ink5060Ti] Layer-Major Single-Sweep Coda Prefill active (effective_chunk=%d, micro_batch=%d, first_host_coda=L%d)\n",
                         eff_chunk_run_, runner_->max_batch, runner_->first_host_coda_layer());
        }
        const double led_runner = free_mib();

        if (opts.enable_dspark && model_.man.has_dspark && !model_.ls_active) {
            strata::ink::df2_model_path() = mf.path;   // where DFlash 2 looks for its sidecar weights (STRATA_INK_DRAFTER=dflash2)
            dspark_ = std::make_unique<strata::ink::InkDSpark>(*runner_);
            dspark_->adapt.on = strata::ink::sm120_cfg(gpu_id_).spec_adapt && !strata::ink::rtx5090::exact_mode_on();   // fast only; the exact form keeps the static window model
            try {
                dspark_->init(eff_max_ctx_, 512);
            } catch (const strata::ink::InkAllocFailure& ex) {
                dspark_->free_all();
                dspark_.reset();
                cudaGetLastError();
                std::fprintf(stderr, "[InkCUDA] DSpark drafter turned off: not enough free GPU memory for it (%s)\n", ex.what());
            }
            if (dspark_ && dspark_->ready && max_slots_ >= 2 && strata::ink::rtx5090::dspark_opt_on()) {
                bspec_ = std::make_unique<strata::ink::rtx5090::Rtx5090BatchedSpec>();
                try {
                    if (!bspec_->init(*runner_, max_slots_)) bspec_.reset();
                } catch (const strata::ink::InkAllocFailure& ex) {
                    bspec_->free_all();
                    bspec_.reset();
                    cudaGetLastError();
                    std::fprintf(stderr, "[InkRTX5090] Mode-3 Batched Spec disabled (%s)\n", ex.what());
                }
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
        slot_hists_.assign(max_slots_, {});
        c64_gov_steps_ = 0; c64_gov_emit_sum_ = 0.0; c64_gov_disabled_ = false;
        initialized_ = true;
        {   // the opt-in tail skip (environment): its tables are loaded now, never inside a CUDA graph capture
            const strata::ink::InkTailCfg& tc0 = strata::ink::ink_tail_cfg();
            if (tc0.on) strata::ink::ink_tail_sync_tables(tc0);
        }
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
        if (slot >= 0 && slot < (int)slot_hists_.size()) slot_hists_[slot].clear();
        if (slot == 0) { dspark_slot_ = -1; tree_snap_ready_ = false; pld_hist_.clear(); slot0_hist_.clear(); }
        if (bspec_) bspec_->reset_slot(slot);
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
        {   // one allocation per array (the copies below are asynchronous into page-locked memory: the vectors must not move while they run)
            size_t nk = 0, nv = 0, nks = 0, nvs = 0;
            for (int v = 0; v < strata::ink::kInkVisits; ++v) {
                const auto& kv = runner_->kv_desc[v];
                if (!kv.is_swa) continue;
                if (kv.kv_fmt == 0) {
                    nk += (size_t)n_pos * strata::ink::kInkKVDim * sizeof(__half);
                    nv += (size_t)n_pos * strata::ink::kInkKVDim * sizeof(__half);
                } else {
                    const size_t v_stride = (kv.kv_fmt == 3) ? (strata::ink::kInkKVDim / 2) : strata::ink::kInkKVDim;
                    nk += (size_t)n_pos * strata::ink::kInkKVDim;
                    nv += (size_t)n_pos * v_stride;
                    nks += (size_t)n_pos * strata::ink::kInkKVHeads * sizeof(float);
                    nvs += (size_t)n_pos * strata::ink::kInkKVHeads * sizeof(float);
                }
            }
            ck.swa_k.reserve(nk); ck.swa_v.reserve(nv); ck.swa_ksc.reserve(nks); ck.swa_vsc.reserve(nvs);
        }
        for (int v = 0; v < strata::ink::kInkVisits; ++v) {
            const auto& kv = runner_->kv_desc[v];
            if (!kv.is_swa) continue;
            cudaStream_t vs = visit_stream(v);
            const int cap = kv.cap_pos;
            const int p0 = start_pos % cap;
            const int seg1 = std::min(n_pos, cap - p0);
            const int seg2 = n_pos - seg1;
            const size_t slot_base = (size_t)slot * cap;
            auto copy_d2h = [&](const void* base, size_t elem_b, size_t stride_elems, strata::ink::PinnedVec<uint8_t>& dst) {
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
        if (slot == 0 && dspark_ && dspark_->ready && !dspark_->df2 && dspark_slot_ == 0) {
            ck.dspark_k.clear(); ck.dspark_v.clear();
            {
                size_t nd = 0;
                for (int il = 0; il < 3; ++il) {
                    if (model_.man.c.dspark_is_swa[il] == 0) continue;
                    nd += (size_t)(seq_len - std::max(0, seq_len - model_.man.c.dspark_swa_win)) * strata::ink::kInkKVDim * sizeof(__half);
                }
                ck.dspark_k.reserve(nd); ck.dspark_v.reserve(nd);
            }
            for (int il = 0; il < 3; ++il) {
                if (model_.man.c.dspark_is_swa[il] == 0) continue;
                const int cap = dspark_->ctx_cap[il];
                const int s0 = std::max(0, seq_len - model_.man.c.dspark_swa_win);
                const int np = seq_len - s0;
                const int p0 = s0 % cap;
                const int seg1 = std::min(np, cap - p0);
                const int seg2 = np - seg1;
                auto copy_ds = [&](const __half* src, strata::ink::PinnedVec<uint8_t>& dst) {
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
            auto copy_h2d = [&](void* base, size_t elem_b, size_t stride_elems, const strata::ink::PinnedVec<uint8_t>& src, size_t& off) {
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
        if (slot == 0 && ck.has_dspark && dspark_ && dspark_->ready && !dspark_->df2) {
            size_t dk_o = 0, dv_o = 0;
            for (int il = 0; il < 3; ++il) {
                if (model_.man.c.dspark_is_swa[il] == 0) continue;
                const int cap = dspark_->ctx_cap[il];
                const int s0 = std::max(0, seq_len - model_.man.c.dspark_swa_win);
                const int np = seq_len - s0;
                const int p0 = s0 % cap;
                const int seg1 = std::min(np, cap - p0);
                const int seg2 = np - seg1;
                auto copy_ds_h2d = [&](__half* dst, const strata::ink::PinnedVec<uint8_t>& src, size_t& off) {
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
            if (dspark_ && dspark_->df2) dspark_->df2->reset_slot(0, seq_len);   // DFlash 2: the ring is rebuilt from the next ingest on
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
        if (slot == 0 && keep_len >= 0 && (size_t)keep_len < pld_hist_.size()) {
            pld_hist_.resize((size_t)keep_len);
        }
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
        const bool track_pld = strata::ink::ink_rtx5060ti_cfg(gpu_id_).pld;
        for (int i = 0; i < N; ++i) {
            if (batch.pos[i] < 0 || batch.pos[i] >= eff_max_ctx_) {
                err = "CUDA KV position " + std::to_string(batch.pos[i]) + " out of bounds [0, " + std::to_string(eff_max_ctx_) + ")";
                return false;
            }
            if (batch.pos[i] > max_pos_req) max_pos_req = batch.pos[i];
            if (track_pld && batch.slot_ids[i] >= 0 && batch.slot_ids[i] < (int)slot_hists_.size()) {
                auto& sh = slot_hists_[batch.slot_ids[i]];
                const size_t p = (size_t)batch.pos[i];
                if (p == 0) sh.clear();
                if (sh.size() <= p) sh.resize(p + 1, 0);
                sh[p] = batch.token_ids[i];
                if (batch.slot_ids[i] == 0) {
                    if (p == 0) slot0_hist_.clear();
                    if (slot0_hist_.size() <= p) slot0_hist_.resize(p + 1, 0);
                    slot0_hist_[p] = batch.token_ids[i];
                }
            }
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
            if (model_.all_sf4_vram && nw > 4) {
                for (int o0 = 0; o0 < nw; o0 += 4) {
                    const int oc = std::min(4, nw - o0);
                    runner_->head_logits_from(h_src + (size_t)o0 * strata::ink::kInkHidden, oc);
                    if (cand_out && topk_.ensure(oc)) {
                        INK_CK(topk_.run(runner_->d_logits, oc, V, runner_->st));
                        INK_CK(cudaStreamSynchronize(runner_->st));
                        const int kc = std::min(kInkCandidates, V);
                        for (int k = 0; k < oc; ++k) {
                            const int dst = dst_rows[o0 + k];
                            const uint64_t* keys = topk_.h_out + (size_t)k * strata::ink::kTopkCand;
                            for (int j = 0; j < kc; ++j) {
                                int32_t id = strata::ink::ink_topk_key_index(keys[j]);
                                if (id < 0 || id >= V) id = 0;
                                cand_out->val[(size_t)dst * kInkCandidates + j] = strata::ink::ink_topk_key_value(keys[j]);
                                cand_out->idx[(size_t)dst * kInkCandidates + j] = id;
                            }
                            cand_out->finite[dst] = topk_.h_fin[k] != 0 ? 1 : 0;
                        }
                    } else {
                        ensure_pinned_logits(oc, V);
                        INK_CK(cudaMemcpyAsync(h_logits_pin_, runner_->d_logits,
                                               (size_t)oc * V * sizeof(float), cudaMemcpyDeviceToHost, runner_->st));
                        INK_CK(cudaStreamSynchronize(runner_->st));
                        for (int k = 0; k < oc; ++k) {
                            std::memcpy(logits_out + (size_t)dst_rows[o0 + k] * V,
                                        h_logits_pin_ + (size_t)k * V, (size_t)V * sizeof(float));
                        }
                    }
                }
                return;
            }
            // PERF-STEP: STRATA_INK_HEAD_SCREEN=1: the exact top-128 of a single row through the INT8 screening index (the full BF16 head and top-k are skipped);
            // STRATA_INK_HEAD_SCREEN_CHECK=1 also runs the full path and compares every key (the full path's answer is the one returned)
            static const bool hs_on = strata::ink::ink_head_screen_wanted(gpu_id_);
            static const bool hs_chk = [] { const char* e = std::getenv("STRATA_INK_HEAD_SCREEN_CHECK"); return e != nullptr && e[0] == '1'; }();
            static long hs_cmp = 0, hs_bad = 0;
            if (hs_on && nw == 1 && cand_out && !hs_chk && topk_.ensure(1) && runner_->head_screen_cands(h_src, topk_)) {
                const int kc = std::min(kInkCandidates, V);
                const int dst = dst_rows[0];
                for (int j = 0; j < kc; ++j) {
                    int32_t id = strata::ink::ink_topk_key_index(topk_.h_out[j]);
                    if (id < 0 || id >= V) id = 0;
                    cand_out->val[(size_t)dst * kInkCandidates + j] = strata::ink::ink_topk_key_value(topk_.h_out[j]);
                    cand_out->idx[(size_t)dst * kInkCandidates + j] = id;
                }
                cand_out->finite[dst] = topk_.h_fin[0] != 0 ? 1 : 0;
                return;
            }
            uint64_t hs_ref[strata::ink::kTopkCand];
            int hs_ref_fin = 1;
            bool hs_do_cmp = false;
            if (hs_on && hs_chk && nw == 1 && cand_out && topk_.ensure(1)) {
                runner_->head_logits_from(h_src, 1);
                INK_CK(topk_.run(runner_->d_logits, 1, V, runner_->st));
                INK_CK(cudaStreamSynchronize(runner_->st));
                std::memcpy(hs_ref, topk_.h_out, sizeof(hs_ref));
                hs_ref_fin = topk_.h_fin[0];
                hs_do_cmp = true;
                if (runner_->head_screen_cands(h_src, topk_)) {
                    ++hs_cmp;
                    int bad = ((topk_.h_fin[0] != 0) != (hs_ref_fin != 0)) ? 1 : 0;   // the full path leaves the flag as 0x01010101, not 1
                    for (int j = 0; j < strata::ink::kTopkCand; ++j) if (topk_.h_out[j] != hs_ref[j]) ++bad;
                    if (bad) { ++hs_bad; std::fprintf(stderr, "[InkHeadScreen] CHECK step %ld: %d keys differ from the full head\n", hs_cmp, bad); }
                }
                if (hs_cmp % 100 == 1) std::fprintf(stderr, "[InkHeadScreen] CHECK %ld steps compared, %ld differ\n", hs_cmp, hs_bad);
                std::memcpy(topk_.h_out, hs_ref, sizeof(hs_ref));
                topk_.h_fin[0] = hs_ref_fin;
            }
            (void)hs_do_cmp;
            if (!hs_do_cmp) runner_->head_logits_from(h_src, nw);
            if (hp) INK_CK(cudaStreamSynchronize(runner_->st));
            auto hq1 = hnow();
            if (cand_out && topk_.ensure(nw)) {
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
            // Opt-in RTX 5060 Ti Visit-Major Macro-Prefill (All-VRAM W=4096 or Layer-Major Coda) for long prompts (M >= 1024)!
            if (total_m >= 1024 && !snap_active_[slot] && taps_out == nullptr &&
                (runner_->can_allvram_macro_prefill() || runner_->can_layer_major_prefill())) {
                bool any_img_all = false;
                int first_want = total_m;
                for (int k = 0; k < total_m; ++k) {
                    const int oi = idxs[k];
                    if (batch.img_embeds && oi < (int)batch.img_embed_idx.size() && batch.img_embed_idx[oi] >= 0) {
                        any_img_all = true;
                        break;
                    }
                    if (out_row[oi] >= 0 && k < first_want) first_want = k;
                }
                if (!any_img_all && first_want >= total_m - runner_->max_batch) {
                    std::vector<int32_t> all_toks(total_m);
                    for (int k = 0; k < total_m; ++k) all_toks[k] = batch.token_ids[idxs[k]];
                    const int p0_all = batch.pos[idxs[0]];
                    const bool want_taps_lm = (batch.capture_taps && dspark_ && dspark_->ready &&
                                               ((slot == 0) || dspark_->multislot_ready));
                    int last_mc = 0;
                    runner_->has_img_step = false;
                    auto tap_cb = [&](int cp0, int cmc) {
                        if (dspark_ && dspark_->ready && !dspark_->multislot_ready && slot == 0 && cp0 + cmc <= dspark_->max_ctx) {
                            dspark_->ingest_taps(cp0, cmc);
                            dspark_slot_ = 0;
                        }
                        if (dspark_ && dspark_->multislot_ready) {
                            std::vector<int32_t> sls(cmc, slot), ps(cmc);
                            for (int i = 0; i < cmc; ++i) ps[i] = cp0 + i;
                            dspark_->ingest_taps_slots(runner_->d_taps, sls.data(), ps.data(), cmc);
                        }
                    };
                    const bool ok_lm = runner_->can_allvram_macro_prefill()
                        ? runner_->forward_seq_allvram_macro(all_toks.data(), p0_all, total_m, slot, want_taps_lm, last_mc, tap_cb)
                        : runner_->forward_seq_layer_major(all_toks.data(), p0_all, total_m, slot, want_taps_lm, last_mc, tap_cb);
                    if (ok_lm && last_mc > 0) {
                        const int base_off = total_m - last_mc;
                        std::vector<int> loc_idx, dst_rows;
                        for (int k = 0; k < last_mc; ++k) {
                            const int oi = idxs[base_off + k];
                            if (out_row[oi] >= 0) { loc_idx.push_back(k); dst_rows.push_back(out_row[oi]); }
                        }
                        emit_selected_logits(loc_idx, dst_rows);
                        return;
                    }
                }
            }
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

                if (track_pld90_ && slot == 0 && !any_img && p0 >= 0) {   // only when the 5090-family drafting part is on (review finding: this copy ran for everyone)
                    const size_t need_len = (size_t)(p0 + M);
                    if (pld_hist_.size() < need_len) pld_hist_.resize(need_len, -1);
                    for (int k = 0; k < M; ++k) pld_hist_[(size_t)(p0 + k)] = toks[k];
                }
                const bool want_ms = (bspec_ && bspec_->ready && dspark_ && dspark_->ready && slot < bspec_->max_slots);
                const bool want_taps = (taps_out != nullptr) || want_ms ||
                    (batch.capture_taps && dspark_ && dspark_->ready && (slot == 0 || dspark_->multislot_ready));

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
                    runner_->elastic_kv.commit_tier2_tree(
                        runner_->d_stage_k, runner_->d_stage_v, runner_->d_acc_path, p0, M, slot, runner_->tree_cap, runner_->st);
                    snap_p0_[slot] = p0;
                    snap_M_[slot] = M;
                } else {
                    runner_->forward_seq(toks.data(), p0, M, slot, want_taps);
                }
                runner_->has_img_step = false;

                if (batch.capture_taps && dspark_ && dspark_->ready && !dspark_->multislot_ready && slot == 0 && p0 + M <= dspark_->max_ctx) {
                    dspark_->ingest_taps(p0, M);
                    dspark_slot_ = 0;
                }
                if (want_ms) {
                    bspec_->ingest_slot_seq(*dspark_, slot, p0, M);
                    bspec_->record_slot_tokens(slot, p0, toks.data(), M);
                }
                if (batch.capture_taps && dspark_ && dspark_->multislot_ready) {
                    std::vector<int32_t> sls(M, slot), ps(M);
                    for (int k = 0; k < M; ++k) ps[k] = p0 + k;
                    dspark_->ingest_taps_slots(runner_->d_taps, sls.data(), ps.data(), M);
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
            const bool want_ms = (bspec_ && bspec_->ready && dspark_ && dspark_->ready);
            const bool want_taps = (taps_out != nullptr) || want_ms || (batch.capture_taps && dspark_ && dspark_->multislot_ready);
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
            if (want_ms) {
                bspec_->ingest_taps_batched(*dspark_, dec_slots.data(), dec_pos.data(), nullptr, B, 1);
                for (int k = 0; k < B; ++k) bspec_->record_slot_tokens(dec_slots[k], dec_pos[k], &dec_toks[k], 1);
            }
            if (batch.capture_taps && dspark_ && dspark_->multislot_ready) {
                dspark_->ingest_taps_slots(runner_->d_taps, dec_slots.data(), dec_pos.data(), B);
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

    bool draft_plain_now(int ctx_pos) const override {
        return strata::core::ink_draft_plain_step(model_.n_host_experts > 0, ctx_pos, draft_forced_, dev_cc_);
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
        // the ONE drafter rule, principle 2 (strata/core/ink_draft_rule.hpp): on a hybrid card (experts in host memory) a step from kDraftPlainCtxHybrid tokens of context on is a plain step.
        // Per step, so it follows every request; `--draft on` turns it off; with all experts in VRAM the window governor decides alone.
        if (strata::core::ink_draft_plain_step(model_.n_host_experts > 0, anchor_pos, draft_forced_, dev_cc_)) return 0;
        const bool dspark_opt = strata::ink::rtx5090::dspark_opt_on() && (runner_->d_tree_anc32 != nullptr);
        const bool deep_tree = (runner_->d_tree_anc32 != nullptr);   // deep verify tree allocated (5090 builder or the 5060 Ti PLD graft)
        const int want_cap = (deep_tree && anchor_pos + 24 < eff_max_ctx_ && anchor_pos + 24 < dspark_->max_ctx)
                                 ? std::min(24, runner_->tree_cap)
                                 : std::min(12, runner_->tree_cap);
        if (anchor_pos <= 0 || anchor_pos + 12 >= eff_max_ctx_ || anchor_pos + 12 >= dspark_->max_ctx || max_emit < 1 || !out_emitted) return 0;
        if (!runner_->ensure_global_kv_cap(anchor_pos + want_cap) || !dspark_->ensure_ctx_cap(anchor_pos + want_cap)) return 0;

        // Track slot-0 token history for zero-cost PLD suffix lookup
        if (dspark_opt) {
            if (pld_hist_.size() <= (size_t)anchor_pos) pld_hist_.resize((size_t)anchor_pos + 1, -1);
            pld_hist_[(size_t)anchor_pos] = anchor_tok;
        }

        // 1. Snapshot slot-0 ShortConv pre-verify state (2.5 us) for exact mid-tree EOS rollback if needed!
        strata::ink::k_ink_sconv_slot_copy<<<dim3(strata::ink::kInkVisits, 4), 256, 0, runner_->st>>>(
            runner_->d_sconv_pool_arr, d_sconv_snap_, 0, runner_->max_slots, true);

        // 2. Build the verify tree. The card table picks the builder: the 5090-family one (Root-PLD-First spine, leaf graft, 2-block chain)
        //    when dspark_opt, otherwise the generic / 5060 Ti one. With every switch off the second branch is exactly the shipped code path.
        const bool sp_log = spec_cfg_.log;
        auto sp_now = [] { return std::chrono::steady_clock::now(); };
        auto sp_ms  = [](std::chrono::steady_clock::time_point a, std::chrono::steady_clock::time_point b) {
            return std::chrono::duration<double, std::milli>(b - a).count();
        };
        auto st_t0 = sp_now();
        strata::ink::InkTreeSpec tr{};
        const auto& opt5060_sp = strata::ink::ink_rtx5060ti_cfg(gpu_id_);
        int max_nodes = 0;
        int real_M = 0;
        auto st_t1 = st_t0;
        if (dspark_opt) {
        // 2. Root-PLD-First (zero-cost exact >=3-gram suffix spine) + W=3..4 Multi-Branch Bayes-Optimal Beam Tree!
        max_nodes = want_cap;
        const int base_max = dspark_opt ? std::min(12, max_nodes) : max_nodes;
        static const bool pld_on = [] {
            const char* e = std::getenv("STRATA_INK_SM120_PLD");
            return !e || e[0] != '0';
        }();
        auto find_pld = [&](const int32_t* key, int klen, int max_cont, int32_t* out_cont, int* out_n = nullptr) -> int {
            const int hi = anchor_pos - 1;
            const int lo = std::max(0, hi - 16384);
            const int32_t last_t = key[klen - 1];
            for (int n = std::min(5, klen); n >= 3; --n) {
                const int cap_c = std::min(max_cont, (n >= 4) ? 14 : 10);
                for (int p = hi - 1; p >= lo + n - 1; --p) {
                    if (pld_hist_[(size_t)p] != last_t) continue;
                    bool eq = true;
                    for (int j = 1; j < n; ++j) {
                        if (pld_hist_[(size_t)(p - j)] != key[klen - 1 - j]) { eq = false; break; }
                    }
                    if (eq) {
                        int cnt = 0;
                        for (int k = 1; k <= cap_c && (p + k) < anchor_pos; ++k) {
                            const int32_t ct = pld_hist_[(size_t)(p + k)];
                            if (ct < 0) break;
                            out_cont[cnt++] = ct;
                        }
                        if (cnt > 0) {
                            if (out_n) *out_n = n;
                            return cnt;
                        }
                    }
                }
            }
            return 0;
        };

        bool root_pld_used = false;
        if (dspark_opt && pld_on && anchor_pos >= 6 && (int)pld_hist_.size() > anchor_pos) {
            int32_t rkey[5];
            for (int j = 0; j < 5; ++j) rkey[j] = pld_hist_[(size_t)(anchor_pos - 4 + j)];
            int32_t cont[16];
            int match_n = 0;
            const int n_c = find_pld(rkey, 5, std::min(14, max_nodes - 3), cont, &match_n);
            if (match_n >= 4 && n_c >= 3) {
                // Build a 10..14-token linear PLD spine FIRST, then attach lean 2-node CUDA-graphed DSpark fallback!
                tr.M = 1;
                tr.tok[0] = anchor_tok;
                tr.parent[0] = -1;
                tr.depth[0] = 0;
                int par = 0;
                for (int k = 0; k < n_c && tr.M < max_nodes && k < 22; ++k) {
                    const int nid = tr.M++;
                    tr.tok[nid]    = cont[k];
                    tr.parent[nid] = (int8_t)par;
                    tr.depth[nid]  = (uint8_t)(k + 1);
                    par = nid;
                }
                int32_t ids7[8] = {};
                float conf7[8] = {};
                dspark_->draft_block8(anchor_tok, anchor_pos, ids7, conf7);
                int cur = 0;
                int k = 0;
                while (k < 7 && k < n_c && ids7[k] == cont[k]) { cur = k + 1; ++k; }
                // Graft up to 2 top-1 DSpark fallback nodes at the first divergence point (total M <= 16!)
                const int fb_cap = std::min(16, max_nodes);
                int fb_added = 0;
                for (; k < 7 && fb_added < 2 && tr.M < fb_cap; ++k) {
                    if (fb_added > 0 && conf7[k] < 0.30f) break;
                    const int nid = tr.M++;
                    tr.tok[nid]    = ids7[k];
                    tr.parent[nid] = (int8_t)cur;
                    tr.depth[nid]  = (uint8_t)(tr.depth[cur] + 1);
                    cur = nid;
                    ++fb_added;
                }
                tr.build();
                root_pld_used = true;
            }
        }

        if (!root_pld_used) {
            dspark_->draft_beam_tree(anchor_tok, anchor_pos, spec_cfg_.beam, base_max, tr, spec_cfg_.cost, spec_cfg_.plain);
        }

        real_M = tr.M;
        if (dspark_opt && !root_pld_used && tr.M < max_nodes) {
            bool tree_mod = false;
            const int leaf_id = dspark_->last_deepest_node;
            bool leaf_extended = false;

            // 2b. Leaf Suffix Lookup (PLD) Graft (< 5 us CPU, 0 VRAM!)
            if (pld_on && anchor_pos >= 6 && (int)pld_hist_.size() > anchor_pos) {
                // Extend deepest highest-prob DSpark leaf via suffix match!
                if (leaf_id > 0 && leaf_id < tr.M && tr.depth[leaf_id] < 23 && tr.M < max_nodes) {
                    int32_t key[16];
                    int klen = 0;
                    const int d_leaf = (int)tr.depth[leaf_id];
                    if (d_leaf < 5) {
                        const int need_h = std::min(5 - d_leaf, anchor_pos);
                        for (int i = anchor_pos - need_h; i < anchor_pos; ++i) key[klen++] = pld_hist_[(size_t)i];
                    }
                    for (int d = std::max(0, d_leaf - 4); d <= d_leaf && klen < 16; ++d) {
                        key[klen++] = tr.tok[tr.anc32[leaf_id][d]];
                    }
                    int32_t cont[16];
                    const int room = std::min(max_nodes - tr.M, std::min(12, 23 - d_leaf));
                    const int n_c = (room > 0) ? find_pld(key, klen, room, cont) : 0;
                    if (n_c > 0) {
                        int par = leaf_id;
                        int dep = d_leaf;
                        for (int k = 0; k < n_c && tr.M < max_nodes && dep < 23; ++k) {
                            const int nid = tr.M++;
                            tr.tok[nid]    = cont[k];
                            tr.parent[nid] = (int8_t)par;
                            tr.depth[nid]  = (uint8_t)(dep + 1);
                            par = nid;
                            ++dep;
                        }
                        tree_mod = true;
                        leaf_extended = true;
                    }
                }
                // Graft 3-gram root suffix matches onto the beam tree without displacing beam siblings!
                if (tr.M < max_nodes && anchor_pos >= 5) {
                    int32_t rkey[5];
                    for (int j = 0; j < 5; ++j) rkey[j] = pld_hist_[(size_t)(anchor_pos - 4 + j)];
                    int32_t cont[16];
                    const int n_c = find_pld(rkey, 5, std::min(12, max_nodes - 1), cont);
                    if (n_c > 0) {
                        int cur = 0;
                        int dep = 0;
                        for (int k = 0; k < n_c && dep < 23; ++k) {
                            int found = -1;
                            for (int j = cur + 1; j < tr.M; ++j) {
                                if (tr.parent[j] == cur && tr.tok[j] == cont[k]) { found = j; break; }
                            }
                            if (found >= 0) {
                                cur = found;
                                dep = (int)tr.depth[cur];
                            } else {
                                if (tr.M >= max_nodes) break;
                                const int nid = tr.M++;
                                tr.tok[nid]    = cont[k];
                                tr.parent[nid] = (int8_t)cur;
                                tr.depth[nid]  = (uint8_t)(dep + 1);
                                cur = nid;
                                ++dep;
                                tree_mod = true;
                            }
                        }
                    }
                }
            }

            // 2c. Optional 2-Block Chained DSpark (depth 8..14) when Block-1 spine is high-confidence!
            static const bool chain2_on = [] {
                const char* e = std::getenv("STRATA_INK_SM120_CHAIN2");
                return !e || e[0] != '0';
            }();
            static const float chain2_thr = [] {
                const char* e = std::getenv("STRATA_INK_SM120_CHAIN2_THR");
                return (e && *e) ? (float)std::atof(e) : 0.55f;
            }();
            if (chain2_on && !leaf_extended && leaf_id > 0 && leaf_id < tr.M &&
                tr.depth[leaf_id] == 7 && tr.M + 2 <= max_nodes &&
                dspark_->last_deepest_logp > -10.0f && std::exp((double)dspark_->last_deepest_logp) >= (double)chain2_thr) {
                int32_t ids2[8] = {};
                float conf2[8] = {};
                dspark_->draft_block8_at(tr.tok[leaf_id], anchor_pos, anchor_pos + 7, ids2, conf2);
                int par = leaf_id;
                int dep = 7;
                double cum2 = 1.0;
                for (int k = 0; k < 7 && tr.M < max_nodes && dep < 23; ++k) {
                    if (conf2[k] < 0.45f) break;
                    cum2 *= std::max(0.0, std::min(1.0, (double)conf2[k]));
                    if (cum2 < 0.30) break;
                    const int nid = tr.M++;
                    tr.tok[nid]    = ids2[k];
                    tr.parent[nid] = (int8_t)par;
                    tr.depth[nid]  = (uint8_t)(dep + 1);
                    par = nid;
                    ++dep;
                    tree_mod = true;
                }
            }

            real_M = tr.M;
            if (tree_mod) {
                tr.build();
            }
        }
            st_t1 = sp_now();
        } else {
        // 2. Root-PLD-First (n >= 3) -> skips DSpark draft (0.005 ms!) on long exact suffix matches; else lean DSpark + leaf PLD graft!
        const int eff_beam  = (opt5060_sp.core_graph && !std::getenv("STRATA_INK_SPEC_BEAM")) ? opt5060_sp.c1_beam : spec_cfg_.beam;
        max_nodes = std::min(opt5060_sp.core_graph ? opt5060_sp.c1_max_nodes : 12, runner_->tree_cap);
        const float eff_cost = (opt5060_sp.core_graph && !std::getenv("STRATA_INK_SPEC_COST")) ? opt5060_sp.c1_spec_cost : spec_cfg_.cost;
        const float eff_plain = (opt5060_sp.core_graph && !std::getenv("STRATA_INK_SPEC_PLAIN")) ? opt5060_sp.c1_spec_plain : spec_cfg_.plain;
        bool used_root_pld = false;
        const bool df2_long = dspark_->df2 && dspark_->df2_plain_ctx(anchor_pos);   // DFlash 2 on a small card at long context: plain steps
        if (opt5060_sp.pld && !slot0_hist_.empty() && !df2_long) {
            int32_t pld_cont[16];
            const int max_d = std::min(14, std::max(0, eff_max_ctx_ - anchor_pos - 2));
            const int nc = strata::ink::ink_pld_find_chain(slot0_hist_, anchor_tok, std::min(max_d, runner_->tree_cap - 1), pld_cont);
            if (nc >= 3) {
                tr.M = 1 + nc;
                tr.tok[0] = anchor_tok;
                tr.parent[0] = -1;
                tr.depth[0] = 0;
                for (int k = 0; k < nc; ++k) {
                    tr.tok[1 + k] = pld_cont[k];
                    tr.parent[1 + k] = (int8_t)k;
                    tr.depth[1 + k] = (uint8_t)(k + 1);
                }
                tr.build();
                used_root_pld = true;
            }
        }
        if (!used_root_pld) {
            dspark_->draft_beam_tree(anchor_tok, anchor_pos, eff_beam, max_nodes, tr, eff_cost, eff_plain);
            if (opt5060_sp.pld && !slot0_hist_.empty() && !df2_long) {
                const int max_d = std::min(strata::ink::kInkDeepTreeDepth - 1, std::max(1, eff_max_ctx_ - anchor_pos - 2));
                const int pld_cap = std::min(opt5060_sp.c1_pld_max_m, runner_->tree_cap);
                strata::ink::ink_pld_graft_tree(slot0_hist_, anchor_tok, tr, pld_cap, max_d);
            }
        }
        st_t1 = sp_now();
        real_M = tr.M;

        // Fast M == 1 plain CUDA-Graph path when confidence gate chose M=1 (zero tree staging/commit overhead!)
        if (opt5060_sp.compact_vram && tr.M == 1) {
            runner_->forward_seq(&anchor_tok, anchor_pos, 1, 0, true);
            auto st_t2 = sp_now();
            int32_t tok_next = 0;
            runner_->head_argmax_exact(1, &tok_next);
            auto st_t3 = sp_now();
            out_emitted[0] = tok_next;
            if (opt5060_sp.pld) {
                if (slot0_hist_.size() > (size_t)anchor_pos + 1) slot0_hist_.resize((size_t)anchor_pos + 1);
                slot0_hist_.push_back(tok_next);
            }
            dspark_->ingest_taps(anchor_pos, 1);
            tree_snap_ready_ = false;
            if (sp_log) {
                static long m1_n = 0; static double m1_d = 0, m1_v = 0, m1_a = 0, m1_c = 0;
                INK_CK(cudaStreamSynchronize(runner_->st));
                auto st_t4 = sp_now();
                ++m1_n; m1_d += sp_ms(st_t0, st_t1); m1_v += sp_ms(st_t1, st_t2); m1_a += sp_ms(st_t2, st_t3); m1_c += sp_ms(st_t3, st_t4);
                if (m1_n <= 5 || m1_n % 50 == 0) {
                    std::fprintf(stderr, "[SPEC-M1 n=%ld] draft=%.2fms fwd=%.2fms argmax=%.2fms ing=%.2fms\n",
                                 m1_n, m1_d / m1_n, m1_v / m1_n, m1_a / m1_n, m1_c / m1_n);
                }
            }
            if (out_drafted) *out_drafted = 0;
            return 1;
        }
        }

        // 3. Single CUDA-Graph 48-visit tree verify + exact GPU screened argmax (ZERO 6.4 MB logits D2H!)
        runner_->forward_tree_verify(tr, anchor_pos, 0, true);
        if (sp_log) INK_CK(cudaStreamSynchronize(runner_->st));
        auto st_t2 = sp_now();
        int32_t target_next[strata::ink::kInkMaxTree] = {};
        runner_->head_argmax_exact(tr.M, target_next);
        auto st_t3 = sp_now();

        // 4. Exact greedy tree walk (clamped to min(max_emit, 24 on an sm_120 card with the drafting switches on, else 8); never walks dummy padding nodes >= real_M!)
        const int emit_cap = std::min(max_emit, dspark_opt ? std::min(24, max_nodes) : (deep_tree ? 24 : 8));
        int32_t path[strata::ink::kInkMaxTree] = {};
        int n_emit = 0;
        int cur_node = 0;
        while (n_emit < emit_cap) {
            path[n_emit] = cur_node;
            const int32_t tok_next = target_next[cur_node];
            out_emitted[n_emit++] = tok_next;
            int next_child = -1;
            for (int j = cur_node + 1; j < real_M; ++j) {
                if (tr.parent[j] == cur_node && tr.tok[j] == tok_next) {
                    next_child = j;
                    break;
                }
            }
            if (next_child >= 0) cur_node = next_child;
            else break;
        }
        if (dspark_->adapt.on) dspark_->adapt.observe(real_M, sp_ms(st_t0, st_t3), n_emit, dspark_->last_pred_etok, dspark_->last_m);
        if (dspark_opt && n_emit > 1) {
            const size_t end_len = (size_t)(anchor_pos + n_emit);
            if (pld_hist_.size() < end_len) pld_hist_.resize(end_len, -1);
            for (int i = 0; i < n_emit - 1; ++i) pld_hist_[(size_t)(anchor_pos + 1 + i)] = out_emitted[i];
        }
        if (opt5060_sp.pld) {
            if (slot0_hist_.size() > (size_t)anchor_pos + 1) slot0_hist_.resize((size_t)anchor_pos + 1);
            for (int i = 0; i < n_emit; ++i) slot0_hist_.push_back(out_emitted[i]);
        }

        // 5. Commit ONLY the accepted tree path across all 48 visits (2 launches) + compact & ingest DSpark taps!
        INK_CK(cudaMemcpyAsync(runner_->d_acc_path, path, (size_t)n_emit * 4, cudaMemcpyHostToDevice, runner_->st));
        strata::ink::k_ink_commit_sconv_all48<<<dim3(strata::ink::kInkVisits, 4), 256, 0, runner_->st>>>(
            runner_->d_sconv_pool_arr, runner_->d_stage_sconv, runner_->d_acc_path, n_emit, 0, runner_->max_slots, runner_->tree_cap);
        strata::ink::k_ink_commit_kv_all48<<<dim3(strata::ink::kInkVisits, strata::ink::kInkKVHeads), 128, 0, runner_->st>>>(
            runner_->d_kv_desc_arr, runner_->d_stage_k, runner_->d_stage_v, runner_->d_acc_path, anchor_pos, n_emit, 0, runner_->tree_cap);
        runner_->elastic_kv.commit_tier2_tree(
            runner_->d_stage_k, runner_->d_stage_v, runner_->d_acc_path, anchor_pos, n_emit, 0, runner_->tree_cap, runner_->st);
        for (int t = 0; t < 4; ++t) {
            strata::ink::k_ink_compact_taps<<<n_emit, 256, 0, runner_->st>>>(
                runner_->d_taps[t], runner_->d_taps_compact[t], runner_->d_acc_path, n_emit);
        }
        dspark_->ingest_taps_from(runner_->d_taps_compact, anchor_pos, n_emit);

        if (spec_cfg_.log) {
            INK_CK(cudaStreamSynchronize(runner_->st));
            auto st_t4 = sp_now();
            static long st_steps = 0, st_emit = 0, st_nodes = 0; static double st_pred = 0.0; static long st_plain = 0;
            static double st_d = 0, st_v = 0, st_a = 0, st_c = 0;
            ++st_steps; st_emit += n_emit; st_nodes += tr.M; st_pred += dspark_->last_pred_etok; if (tr.M == 1) ++st_plain;
            st_d += sp_ms(st_t0, st_t1); st_v += sp_ms(st_t1, st_t2); st_a += sp_ms(st_t2, st_t3); st_c += sp_ms(st_t3, st_t4);
            if (st_steps <= 5 || st_steps % 50 == 0) {
                std::fprintf(stderr, "[SPEC-STATS steps=%ld] M=%.2f emit=%.2f plain=%ld | draft=%.2fms verify=%.2fms argmax=%.2fms commit=%.2fms\n",
                             st_steps, (double)st_nodes / st_steps, (double)st_emit / st_steps, st_plain,
                             st_d / st_steps, st_v / st_steps, st_a / st_steps, st_c / st_steps);
            }
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

    // ---- The one multi-user (C2..C64) drafting hook (ink_backend.h). The card table picks the implementation:
    //      the 5090-family Mode-3 batched spec (bspec_) or the 5060 Ti-family K8V4 multi-slot DSpark chain verify. ----
    bool c64_k8v4_ready() const noexcept {
        return initialized_ && !model_.ls_active && runner_ && dspark_ && dspark_->multislot_ready &&
               !c64_gov_disabled_ && strata::ink::ink_rtx5060ti_cfg(gpu_id_).c64_spec;
    }
    bool bspec_ready() const noexcept {
        return initialized_ && !model_.ls_active && dspark_ && dspark_->ready && bspec_ && bspec_->ready;
    }
    bool supports_batched_dspark() const noexcept override { return bspec_ready() || c64_k8v4_ready(); }
    bool prefers_candidate_rows() const noexcept override { return initialized_ && strata::ink::sm120_cfg(gpu_id_).fast; }
    bool batched_dspark_prefill_first() const noexcept override { return c64_k8v4_ready(); }

    int dspark_spec_step_batched_greedy(
        int B,
        const int* slots,
        const int32_t* anchor_toks,
        const int* anchor_pos,
        const int* max_emits,
        int max_stride,
        int32_t* out_emitted,
        int* out_n_emit,
        int* out_drafted,
        std::string& err) override {
        if (bspec_ready()) {
            return spec_step_batched_bspec(B, slots, anchor_toks, anchor_pos, max_emits, max_stride, out_emitted, out_n_emit, out_drafted, err);
        }
        if (c64_k8v4_ready()) {
            // the K8V4 chain verify writes 8 tokens per slot
            if (max_stride < 8) return 0;
            return spec_step_batched_c64(B, slots, anchor_toks, anchor_pos, max_emits, out_emitted, out_n_emit, out_drafted, err);
        }
        return 0;
    }



    int max_macro_prefill_tokens() const noexcept override {
        if (!initialized_ || !runner_) return 0;
        if (runner_->can_allvram_macro_prefill() || runner_->can_layer_major_prefill()) return 8704;
        return 0;
    }

    int spec_step_batched_bspec(
        int B,
        const int* slots,
        const int32_t* anchor_toks,
        const int* anchor_pos,
        const int* max_emits,
        int max_stride,
        int32_t* out_emitted,
        int* out_n_emit,
        int* out_drafted,
        std::string& /*err*/) {
        if (!initialized_ || model_.ls_active || !dspark_ || !dspark_->ready || !bspec_ || !bspec_->ready) return 0;
        if (B < 2 || B > bspec_->max_slots || max_stride < 4 || !slots || !anchor_toks || !anchor_pos || !max_emits || !out_emitted || !out_n_emit) return 0;
        if (dspark_->df2 && B > dspark_->df2->max_slots) return 0;   // DFlash 2 keeps context rings for at most STRATA_INK_DFLASH2_SLOTS (16) slots
        if (bspec_->gov_plain_steps > 0) { --bspec_->gov_plain_steps; return 0; }

        static const int k_draft_cfg = [] {
            const char* e = std::getenv("STRATA_INK_SM120_BSPEC_KD");
            int v = (e && *e) ? std::atoi(e) : 3;
            return std::max(1, std::min(3, v));
        }();
        int K_draft = k_draft_cfg;
        while (K_draft > 1 && B * (K_draft + 1) > bspec_->m_cap) --K_draft;
        const int K_slot = K_draft + 1;
        const int M = B * K_slot;

        int max_need_pos = 0;
        for (int b = 0; b < B; ++b) {
            const int sl = slots[b];
            const int ap = anchor_pos[b];
            if (sl < 0 || sl >= bspec_->max_slots || ap <= 0 || ap + K_slot + 2 >= eff_max_ctx_) return 0;
            if (bspec_->slot_synced_pos[sl] != ap) return 0;
            if (ap + K_slot > max_need_pos) max_need_pos = ap + K_slot;
        }
        if (!runner_->ensure_global_kv_cap(max_need_pos)) return 0;

        // 1. Batched 64-slot DSpark draft (exact Block-8 backbone + INT8 TC base logits + Markov RNN!)
        const double t_b0 = spec_cfg_.log ? strata::ink::now_s() : 0.0;
        int32_t draft_buf[strata::ink::rtx5090::kBspecMaxM] = {};
        bspec_->draft_batched(*dspark_, B, K_draft, slots, anchor_toks, anchor_pos, draft_buf);
        const double t_b1 = spec_cfg_.log ? strata::ink::now_s() : 0.0;

        // 2. Per-slot >=2/3-gram PLD override (zero VRAM, boosts JSON/code/chat batches!)
        for (int b = 0; b < B; ++b) {
            const int sl = slots[b];
            const int ap = anchor_pos[b];
            bspec_->record_slot_tokens(sl, ap, &anchor_toks[b], 1);
            int32_t pld_c[4] = {};
            const int nc = bspec_->find_slot_pld(sl, ap, K_draft, pld_c);
            for (int k = 0; k < nc && k < K_draft; ++k) draft_buf[b * K_draft + k] = pld_c[k];
        }

        // 3. Build flat [B * K_slot] verify batch -> ONE CUDA-Graphed Mode-3 48-visit pass!
        int32_t toks[strata::ink::rtx5090::kBspecMaxM];
        int32_t sl_arr[strata::ink::rtx5090::kBspecMaxM];
        int32_t pos_arr[strata::ink::rtx5090::kBspecMaxM];
        for (int b = 0; b < B; ++b) {
            const int base = b * K_slot;
            toks[base]    = anchor_toks[b];
            sl_arr[base]  = slots[b];
            pos_arr[base] = anchor_pos[b];
            for (int k = 0; k < K_draft; ++k) {
                toks[base + 1 + k]    = draft_buf[b * K_draft + k];
                sl_arr[base + 1 + k]  = slots[b];
                pos_arr[base + 1 + k] = anchor_pos[b] + 1 + k;
            }
        }
        runner_->forward_batched_spec(toks, sl_arr, pos_arr, B, K_slot, true);
        if (spec_cfg_.log) INK_CK(cudaStreamSynchronize(runner_->st));
        const double t_b2 = spec_cfg_.log ? strata::ink::now_s() : 0.0;

        // 4. Fast exact target argmax (0.28 ms via INT8 TC Top-32 screen + exact Q8ROW/S6_K eval!)
        int32_t argmax_buf[strata::ink::rtx5090::kBspecMaxM] = {};
        bspec_->head_argmax_fast(*runner_, M, argmax_buf);
        const double t_b3 = spec_cfg_.log ? strata::ink::now_s() : 0.0;

        // 5. Exact lossless greedy verification per slot!
        int n_acc_b[strata::ink::rtx5090::kBspecMaxSlots] = {};
        int total_emit = 0;
        for (int b = 0; b < B; ++b) {
            const int base = b * K_slot;
            const int cap  = std::min(max_emits[b], K_slot);
            int32_t* dst   = out_emitted + (size_t)b * max_stride;
            int na = 0;
            for (int k = 0; k < cap; ++k) {
                const int32_t t_next = argmax_buf[base + k];
                dst[na++] = t_next;
                if (k < K_draft && t_next == draft_buf[b * K_draft + k]) continue;
                break;
            }
            n_acc_b[b] = na;
            out_n_emit[b] = na;
            if (out_drafted) out_drafted[b] = K_draft;
            total_emit += na;
            if (na > 1) bspec_->record_slot_tokens(slots[b], anchor_pos[b] + 1, dst, na - 1);
        }

        // 6. Commit accepted KV + ShortConv across all 48 visits (2 launches!) + ingest accepted DSpark taps!
        runner_->commit_bspec(slots, anchor_pos, n_acc_b, B, K_slot);
        bspec_->ingest_taps_batched(*dspark_, slots, anchor_pos, n_acc_b, B, K_slot);
        INK_CK(cudaStreamSynchronize(runner_->st));
        const double t_b4 = spec_cfg_.log ? strata::ink::now_s() : 0.0;

        // 7. 2-step adaptive governor (falls back cleanly if synthetic random acceptance < 1.25 tok/slot!)
        const double mean_emit = (double)total_emit / (double)B;
        if (spec_cfg_.log) {
            static int bs_n = 0; static double s_d = 0, s_v = 0, s_a = 0, s_c = 0, s_e = 0;
            ++bs_n;
            s_d += (t_b1 - t_b0) * 1000.0; s_v += (t_b2 - t_b1) * 1000.0;
            s_a += (t_b3 - t_b2) * 1000.0; s_c += (t_b4 - t_b3) * 1000.0;
            s_e += mean_emit;
            if (bs_n % 10 == 0) {
                const double tot_ms = (s_d + s_v + s_a + s_c) / bs_n;
                std::fprintf(stderr, "[BSPEC-STATS] n=%d B=%d KD=%d mean_emit=%.2f | draft=%.2f verify=%.2f argmax=%.2f commit=%.2f total=%.2f ms (%.0f tok/s)\n",
                             bs_n, B, K_draft, s_e / bs_n, s_d / bs_n, s_v / bs_n, s_a / bs_n, s_c / bs_n, tot_ms,
                             (s_e / bs_n) * B * 1000.0 / std::max(0.01, tot_ms));
            }
        }
        if (mean_emit < 1.25) {
            if (++bspec_->gov_low_streak >= 2) {
                bspec_->gov_plain_steps = 20;
                bspec_->gov_low_streak = 0;
            }
        } else {
            bspec_->gov_low_streak = 0;
        }
        return total_emit;
    }

    int spec_step_batched_c64(
        int B,
        const int32_t* slots,
        const int32_t* anchor_toks,
        const int32_t* anchor_pos,
        const int32_t* max_emit,
        int32_t* out_emitted,
        int32_t* out_n_emit,
        int32_t* out_drafted,
        std::string& err) {
        if (B <= 0 || B > max_slots_) return 0;
        if (dspark_->df2 && B > dspark_->df2->max_slots) return 0;   // DFlash 2 keeps context rings for at most STRATA_INK_DFLASH2_SLOTS (16) slots
        const auto& opt5060 = strata::ink::ink_rtx5060ti_cfg(gpu_id_);
        const int base_kd = std::max(1, std::min(7, opt5060.c64_draft_k));
        const int max_kd_fit = std::max(1, std::min(7, (runner_->max_batch / B) - 1));
        const int tail_mode = []() { const char* e = strata::ink::sm120_cfg_detail::env2("STRATA_INK_SM120_C64_TAIL_MODE", "STRATA_INK_5060TI_C64_TAIL_MODE"); return (e && *e) ? std::atoi(e) : 1; }();
        int n_multi = 0;
        for (int b = 0; b < B; ++b) if (max_emit[b] >= 2) ++n_multi;
        int want_kd = base_kd;
        if (tail_mode == 1) {
            if (B <= 16 || n_multi * 2 < B) return 0;
            if (B <= 32) want_kd = 1;
        } else if (tail_mode == 3) {
            if (B <= 16) return 0;
            else if (B <= 32) want_kd = 1;
            else if (B <= 48) want_kd = std::min(base_kd, 2);
        } else if (tail_mode == 4) {
            if (B <= 24) return 0;
            else if (B <= 44) want_kd = 1;
        } else if (tail_mode == 2) {
            want_kd = (B <= 30) ? std::max(base_kd, 5)
                    : ((B <= 38) ? std::max(base_kd, 4)
                    : ((B <= 48) ? std::max(base_kd, 3) : base_kd));
        }
        const int K_d = std::min(want_kd, max_kd_fit);
        const int K   = 1 + K_d;
        if (B * K > runner_->max_batch) return 0;
        int max_need = 0;
        for (int b = 0; b < B; ++b) {
            if (anchor_pos[b] <= 0 || anchor_pos[b] + K + 1 >= eff_max_ctx_) return 0;
            max_need = std::max(max_need, anchor_pos[b] + K + 1);
        }
        if (!runner_->ensure_global_kv_cap(max_need)) {
            kv_grow_failed_ = true;
            err = "global KV cap reached during batched speculative step";
            return 0;
        }
        if (!runner_->ensure_mc_buffers(std::max(B * K, max_slots_ * (base_kd + 1)))) return 0;

        const bool pf = runner_->prof_enabled;
        auto now_tp = [] { return std::chrono::steady_clock::now(); };
        auto dt_ms  = [](std::chrono::steady_clock::time_point a, std::chrono::steady_clock::time_point b) {
            return std::chrono::duration<double, std::milli>(b - a).count();
        };
        auto t0 = now_tp();
        // 1. Batched Multi-Slot DSpark Block-8 Draft across all B concurrent slots (only K_d draft rows/steps!)
        std::vector<int32_t> drafts((size_t)B * 7);
        dspark_->draft_block8_batched(B, slots, anchor_toks, anchor_pos, drafts.data(), K_d);
        if (opt5060.pld) {
            int32_t pld_cont[8];
            for (int b = 0; b < B; ++b) {
                const int sl = slots[b];
                if (sl >= 0 && sl < (int)slot_hists_.size() && !slot_hists_[sl].empty()) {
                    const int nc = strata::ink::ink_pld_find_chain(slot_hists_[sl], anchor_toks[b], K_d, pld_cont);
                    for (int k = 0; k < nc; ++k) drafts[(size_t)b * 7 + k] = pld_cont[k];
                }
            }
        }
        auto t1 = now_tp();

        // 2. Pack [B * K] verify token chain & run ONE Mode-3 48-visit forward pass!
        const int M = B * K;
        std::vector<int32_t> v_toks(M);
        for (int b = 0; b < B; ++b) {
            v_toks[b * K + 0] = anchor_toks[b];
            for (int k = 0; k < K_d; ++k) v_toks[b * K + 1 + k] = drafts[(size_t)b * 7 + k];
        }
        runner_->forward_multichain_verify(v_toks.data(), slots, anchor_pos, B, K, true);
        if (pf) INK_CK(cudaStreamSynchronize(runner_->st));
        auto t2 = now_tp();

        // 3. Exact GPU screened argmax over all M = B * K rows (ZERO 400 MiB logits D2H!)
        std::vector<int32_t> target_next(M);
        runner_->head_argmax_exact_batch(M, target_next.data());
        auto t3 = now_tp();

        // 4. Exact greedy per-slot walk (100% mathematically lossless!)
        std::vector<int32_t> n_acc(B, 1);
        std::vector<int32_t> tap_rows, tap_slots, tap_pos;
        tap_rows.reserve(M);
        tap_slots.reserve(M);
        tap_pos.reserve(M);
        int tot_emit = 0;
        for (int b = 0; b < B; ++b) {
            const int cap = std::max(1, std::min(K, max_emit[b]));
            int ne = 0;
            for (int k = 0; k < cap; ++k) {
                const int32_t tn = target_next[b * K + k];
                out_emitted[(size_t)b * 8 + ne] = tn;
                tap_rows.push_back(b * K + k);
                tap_slots.push_back(slots[b]);
                tap_pos.push_back(anchor_pos[b] + k);
                ++ne;
                if (k < K_d && tn == drafts[(size_t)b * 7 + k]) continue;
                break;
            }
            n_acc[b] = ne;
            out_n_emit[b] = ne;
            tot_emit += ne;
            if (opt5060.pld && slots[b] >= 0 && slots[b] < (int)slot_hists_.size()) {
                auto& sh = slot_hists_[slots[b]];
                const size_t want_len = (size_t)anchor_pos[b] + 1;
                if (sh.size() > want_len) sh.resize(want_len);
                if (sh.size() < want_len) sh.resize(want_len, anchor_toks[b]);
                for (int k = 0; k < ne; ++k) sh.push_back(out_emitted[(size_t)b * 8 + k]);
            }
            if (out_drafted) out_drafted[b] = K_d;
        }

        // 5. Commit exact ShortConv 3-lag state per slot across all 48 visits + ingest accepted taps into multi-slot DSpark!
        runner_->commit_multichain_sconv(n_acc.data(), B, K);
        if (!tap_rows.empty()) {
            dspark_->ingest_taps_selected(tap_rows.data(), tap_slots.data(), tap_pos.data(), (int)tap_rows.size());
        }
        // 2-step C64 Spec Governor: if random/unpredictable prompts emit < 1.35 tok/slot after 2 steps, cleanly defer to plain B=64!
        if (++c64_gov_steps_ <= 2) {
            c64_gov_emit_sum_ += (double)tot_emit / (double)B;
            if (c64_gov_steps_ == 2 && (c64_gov_emit_sum_ * 0.5) < 1.35) {
                c64_gov_disabled_ = true;
                std::fprintf(stderr, "[Ink5060Ti] C64 Spec Governor: mean %.2f tok/slot < 1.35 -> switching to plain B=%d\n",
                             c64_gov_emit_sum_ * 0.5, B);
            }
        }
        if (pf) {
            INK_CK(cudaStreamSynchronize(runner_->st));
            auto t4 = now_tp();
            static int mc_n = 0;
            if (++mc_n <= 8 || (mc_n % 10 == 0)) {
                std::fprintf(stderr, "[C64-SPEC step=%d B=%d K=%d M=%d] draft=%.1fms verify=%.1fms argmax=%.1fms commit+ing=%.1fms tot=%.1fms emit=%d (%.2f/slot, %.1f tok/s)\n",
                             mc_n, B, K, M, dt_ms(t0, t1), dt_ms(t1, t2), dt_ms(t2, t3), dt_ms(t3, t4), dt_ms(t0, t4),
                             tot_emit, (double)tot_emit / B, (double)tot_emit * 1000.0 / std::max(0.1, dt_ms(t0, t4)));
            }
        }
        return std::max(1, tot_emit);
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
        if (bspec_)  { bspec_->free_all(); bspec_.reset(); }
        if (vit_) { vit_->release(); vit_.reset(); }
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
        strata::ink::PinnedVec<float> sconv;
        strata::ink::PinnedVec<uint8_t> swa_k, swa_v, swa_ksc, swa_vsc;
        strata::ink::PinnedVec<uint8_t> dspark_k, dspark_v;
        strata::ink::PinnedVec<float> dspark_frnn;
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
    size_t primary_mem_bytes_ = 0;
    bool vit_planned_ = false;
    bool vit_elastic_ = false;
    double vit_idle_s_ = 120.0;
    std::string vit_path_;
    std::unique_ptr<strata::ink::InkVisionCuda> vit_;
    std::chrono::steady_clock::time_point vit_last_ = std::chrono::steady_clock::now();
    std::unique_ptr<strata::ink::InkCudaRunner> runner_;
    std::unique_ptr<strata::ink::InkDSpark> dspark_;
    std::unique_ptr<strata::ink::rtx5090::Rtx5090BatchedSpec> bspec_;
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
    bool track_pld90_ = false;       // sm_120 table: the 5090-family deep-tree drafting part is on for this card
    std::vector<int32_t> pld_hist_;  // Slot-0 token history for zero-cost PLD suffix lookup (5090-family drafting switch)
    std::vector<int32_t> slot0_hist_;
    int dev_cc_ = 90;             // compute capability of the device (the drafter rule: emulated draft steps below sm_80)
    bool draft_forced_ = false;   // --draft on / STRATA_INK_DFLASH2_FORCE=1: the residency-first rule never vetoes the drafter (strata/core/ink_draft_rule.hpp)
    std::vector<std::vector<int32_t>> slot_hists_;
    int    c64_gov_steps_    = 0;
    double c64_gov_emit_sum_ = 0.0;
    bool   c64_gov_disabled_ = false;
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

bool ink_cuda_compact_trunk_supported(int gpu_index) {
    int major = 0, minor = 0;
    const int dev = gpu_index < 0 ? 0 : gpu_index;
    if (cudaDeviceGetAttribute(&major, cudaDevAttrComputeCapabilityMajor, dev) != cudaSuccess ||
        cudaDeviceGetAttribute(&minor, cudaDevAttrComputeCapabilityMinor, dev) != cudaSuccess) {
        cudaGetLastError();
        return false;
    }
    return major * 10 + minor >= 61;
}

}  // namespace ink2
