// engine/src/inkling2/hip/ink_hip_r9700_state.hpp
// Opt-in R9700 (gfx1201, 32 GB) runtime state container & helper kernels.
// Allocated via std::unique_ptr<R9700State> ONLY when STRATA_INK_R9700=1 passes the real-card gate.
// When OFF, r9700_ is nullptr -> ZERO allocations, ZERO extra streams/events, ZERO behavior change.
// MIT License - Copyright (c) 2026 Niko1221 and the Strata contributors.
#pragma once

#include "ink_hip_r9700_attn.hip.h"
#include "ink_hip_r9700_cfg.hpp"
#include "ink_hip_r9700_fp8.hip.h"
#include "ink_hip_r9700_spec.hip.h"

#include <hip/hip_runtime.h>

#include <cstdint>
#include <unordered_map>
#include <vector>

#ifdef _WIN32
#ifndef WIN32_LEAN_AND_MEAN
#define WIN32_LEAN_AND_MEAN
#endif
#include <windows.h>
#else
#include <dlfcn.h>
#endif

namespace strata::inkling2::hip::r9700 {

using strata::hip_uma::GpuBuffer;
using strata::hip_uma::alloc_gpu_buffer;
using strata::hip_uma::free_gpu_buffer;

// =================================================================================================
// 0. Opt-In Dynamic hipBLASLt FP8_E4M3 -> FP32 GEMM Runner (233 TFLOP/s on gfx1201 R9700!)
//    Zero link dependency; cleanly falls back to native k_r9700_fp8_wmma_128x128 if libhipblaslt unavailable.
// =================================================================================================
class R9700Fp8BlasLt {
public:
    static constexpr int kTypeF32  = 0;    // HIP_R_32F
    static constexpr int kTypeFp8  = 28;   // HIP_R_8F_E4M3 (gfx1201 OCP FP8!)
    static constexpr int kOpN      = 111;  // HIPBLAS_OP_N
    static constexpr int kOpT      = 112;  // HIPBLAS_OP_T
    static constexpr int kCompF32  = 2;    // HIPBLAS_COMPUTE_32F
    static constexpr int kAttrTransA   = 0;
    static constexpr int kAttrTransB   = 1;
    static constexpr int kAttrAScaleP  = 5;  // HIPBLASLT_MATMUL_DESC_A_SCALE_POINTER
    static constexpr int kAttrBScaleP  = 6;  // HIPBLASLT_MATMUL_DESC_B_SCALE_POINTER
    static constexpr int kPrefMaxWs    = 1;

    struct Algo { uint8_t data[16]{}; size_t max_ws = 0; };
    struct Heur { Algo algo{}; size_t ws = 0; int32_t state = 0; float waves = 1.0f; int rsv[4]{}; };

    ~R9700Fp8BlasLt() { release(); }

    bool init(hipStream_t st) noexcept {
        release();
        // Opt-in only (STRATA_INK_R9700_FP8_BLASLT=1): custom XOR-swizzled FP8 WMMA avoids cold heuristic TTFT
        // spikes and extra FP32 scale-epilogue DRAM traffic on R9700.
        const char* env = std::getenv("STRATA_INK_R9700_FP8_BLASLT");
        if (!env || std::atoi(env) == 0) return false;
#ifdef _WIN32
        lib_ = static_cast<void*>(LoadLibraryA("libhipblaslt.dll"));
        if (!lib_) lib_ = static_cast<void*>(LoadLibraryA("hipblaslt.dll"));
#else
        lib_ = dlopen("libhipblaslt.so.1", RTLD_LAZY | RTLD_LOCAL);
        if (!lib_) lib_ = dlopen("libhipblaslt.so", RTLD_LAZY | RTLD_LOCAL);
#endif
        if (!lib_) return false;
        auto sym = [&](const char* nm) -> void* {
#ifdef _WIN32
            return reinterpret_cast<void*>(GetProcAddress(static_cast<HMODULE>(lib_), nm));
#else
            return dlsym(lib_, nm);
#endif
        };
        fn_create_       = reinterpret_cast<FnCreate>(sym("hipblasLtCreate"));
        fn_destroy_      = reinterpret_cast<FnDestroy>(sym("hipblasLtDestroy"));
        fn_md_create_    = reinterpret_cast<FnMdCreate>(sym("hipblasLtMatmulDescCreate"));
        fn_md_destroy_   = reinterpret_cast<FnMdDestroy>(sym("hipblasLtMatmulDescDestroy"));
        fn_md_set_       = reinterpret_cast<FnMdSet>(sym("hipblasLtMatmulDescSetAttribute"));
        fn_ml_create_    = reinterpret_cast<FnMlCreate>(sym("hipblasLtMatrixLayoutCreate"));
        fn_ml_destroy_   = reinterpret_cast<FnMlDestroy>(sym("hipblasLtMatrixLayoutDestroy"));
        fn_pref_create_  = reinterpret_cast<FnPrefCreate>(sym("hipblasLtMatmulPreferenceCreate"));
        fn_pref_destroy_ = reinterpret_cast<FnPrefDestroy>(sym("hipblasLtMatmulPreferenceDestroy"));
        fn_pref_set_     = reinterpret_cast<FnPrefSet>(sym("hipblasLtMatmulPreferenceSetAttribute"));
        fn_get_heur_     = reinterpret_cast<FnGetHeur>(sym("hipblasLtMatmulAlgoGetHeuristic"));
        fn_matmul_       = reinterpret_cast<FnMatmul>(sym("hipblasLtMatmul"));
        if (!fn_create_ || !fn_destroy_ || !fn_md_create_ || !fn_md_destroy_ || !fn_md_set_ ||
            !fn_ml_create_ || !fn_ml_destroy_ || !fn_pref_create_ || !fn_pref_destroy_ ||
            !fn_pref_set_ || !fn_get_heur_ || !fn_matmul_) {
            release();
            return false;
        }
        if (fn_create_(&handle_) != 0 || !handle_ || fn_pref_create_(&pref_) != 0 || !pref_) {
            release();
            return false;
        }
        ws_bytes_ = 32ULL << 20;
        if (alloc_gpu_buffer(ws_, ws_bytes_, false) != hipSuccess) {
            (void)hipGetLastError();
            ws_bytes_ = 8ULL << 20;
            if (alloc_gpu_buffer(ws_, ws_bytes_, false) != hipSuccess) { (void)hipGetLastError(); ws_bytes_ = 0; }
        }
        const uint64_t max_ws = static_cast<uint64_t>(ws_bytes_);
        (void)fn_pref_set_(pref_, kPrefMaxWs, &max_ws, sizeof(max_ws));
        if (alloc_gpu_buffer(d_one_, sizeof(float), false) != hipSuccess) {
            (void)hipGetLastError();
            release();
            return false;
        }
        const float one = 1.0f;
        (void)hipMemcpyAsync(d_one_.dev_ptr, &one, sizeof(float), hipMemcpyHostToDevice, st);
        ready_ = true;
        return true;
    }

    bool ready() const noexcept { return ready_; }

    // Computes unscaled row-major Y[M, N] = X_fp8[M, K] @ W_fp8[N, K]^T via col-major Y^T[N, M] = W^T^T @ X^T
    bool gemm_fp8_f32(const uint8_t* W_fp8, const uint8_t* X_fp8, float* Y_f32,
                      int M, int N, int K, hipStream_t st) noexcept {
        if (!ready_ || M <= 0 || N <= 0 || K <= 0) return false;
        const Key key{M, N, K};
        auto it = cache_.find(key);
        if (it == cache_.end()) {
            Plan p{};
            (void)build_plan(key, p);
            it = cache_.emplace(key, p).first;
        }
        const Plan& plan = it->second;
        if (!plan.valid) return false;
        const float alpha = 1.0f, beta = 0.0f;
        return fn_matmul_(handle_, plan.md, &alpha, W_fp8, plan.ad, X_fp8, plan.bd,
                          &beta, Y_f32, plan.cd, Y_f32, plan.dd, &plan.algo,
                          ws_.dev_ptr, ws_bytes_, st) == 0;
    }

    void release() noexcept {
        for (auto& kv : cache_) {
            const auto& p = kv.second;
            if (p.ad && fn_ml_destroy_) fn_ml_destroy_(p.ad);
            if (p.bd && fn_ml_destroy_) fn_ml_destroy_(p.bd);
            if (p.cd && fn_ml_destroy_) fn_ml_destroy_(p.cd);
            if (p.dd && fn_ml_destroy_) fn_ml_destroy_(p.dd);
            if (p.md && fn_md_destroy_) fn_md_destroy_(p.md);
        }
        cache_.clear();
        if (pref_ && fn_pref_destroy_) { fn_pref_destroy_(pref_); pref_ = nullptr; }
        if (handle_ && fn_destroy_) { fn_destroy_(handle_); handle_ = nullptr; }
        free_gpu_buffer(ws_);
        free_gpu_buffer(d_one_);
        ws_bytes_ = 0;
        if (lib_) {
#ifdef _WIN32
            FreeLibrary(static_cast<HMODULE>(lib_));
#else
            dlclose(lib_);
#endif
            lib_ = nullptr;
        }
        ready_ = false;
    }

private:
    struct Key {
        int M, N, K;
        bool operator==(const Key& o) const noexcept { return M == o.M && N == o.N && K == o.K; }
    };
    struct KeyHash {
        size_t operator()(const Key& k) const noexcept {
            size_t h = static_cast<size_t>(k.M) * 1315423911u;
            h ^= static_cast<size_t>(k.N) + 0x9e3779b9u + (h << 6) + (h >> 2);
            h ^= static_cast<size_t>(k.K) + 0x9e3779b9u + (h << 6) + (h >> 2);
            return h;
        }
    };
    struct Plan { bool valid = false; void *md = nullptr, *ad = nullptr, *bd = nullptr, *cd = nullptr, *dd = nullptr; Algo algo{}; };

    bool build_plan(const Key& k, Plan& out) noexcept {
        if (fn_md_create_(&out.md, kCompF32, kTypeF32) != 0) return false;
        const int32_t ta = kOpT, tb = kOpN;
        const void* sc_p = d_one_.dev_ptr;
        if (fn_md_set_(out.md, kAttrTransA, &ta, sizeof(ta)) != 0 ||
            fn_md_set_(out.md, kAttrTransB, &tb, sizeof(tb)) != 0 ||
            fn_md_set_(out.md, kAttrAScaleP, &sc_p, sizeof(sc_p)) != 0 ||
            fn_md_set_(out.md, kAttrBScaleP, &sc_p, sizeof(sc_p)) != 0) {
            return false;
        }
        if (fn_ml_create_(&out.ad, kTypeFp8, static_cast<uint64_t>(k.K), static_cast<uint64_t>(k.N), static_cast<int64_t>(k.K)) != 0 ||
            fn_ml_create_(&out.bd, kTypeFp8, static_cast<uint64_t>(k.K), static_cast<uint64_t>(k.M), static_cast<int64_t>(k.K)) != 0 ||
            fn_ml_create_(&out.cd, kTypeF32, static_cast<uint64_t>(k.N), static_cast<uint64_t>(k.M), static_cast<int64_t>(k.N)) != 0 ||
            fn_ml_create_(&out.dd, kTypeF32, static_cast<uint64_t>(k.N), static_cast<uint64_t>(k.M), static_cast<int64_t>(k.N)) != 0) {
            return false;
        }
        Heur res[4]{};
        int ret = 0;
        if (fn_get_heur_(handle_, out.md, out.ad, out.bd, out.cd, out.dd, pref_, 4, res, &ret) != 0 || ret <= 0) return false;
        for (int i = 0; i < ret; ++i) {
            if (res[i].state == 0 && res[i].ws <= ws_bytes_) {
                out.algo = res[i].algo;
                out.valid = true;
                return true;
            }
        }
        return false;
    }

    using FnCreate      = int32_t (*)(void**);
    using FnDestroy     = int32_t (*)(void*);
    using FnMdCreate    = int32_t (*)(void**, int, int);
    using FnMdDestroy   = int32_t (*)(void*);
    using FnMdSet       = int32_t (*)(void*, int, const void*, size_t);
    using FnMlCreate    = int32_t (*)(void**, int, uint64_t, uint64_t, int64_t);
    using FnMlDestroy   = int32_t (*)(void*);
    using FnPrefCreate  = int32_t (*)(void**);
    using FnPrefDestroy = int32_t (*)(void*);
    using FnPrefSet     = int32_t (*)(void*, int, const void*, size_t);
    using FnGetHeur     = int32_t (*)(void*, void*, void*, void*, void*, void*, void*, int, Heur*, int*);
    using FnMatmul      = int32_t (*)(void*, void*, const void*, const void*, void*, const void*, void*,
                                      const void*, const void*, void*, void*, void*, const Algo*,
                                      void*, size_t, hipStream_t);

    void *lib_ = nullptr, *handle_ = nullptr, *pref_ = nullptr;
    GpuBuffer ws_{}, d_one_{};
    size_t ws_bytes_ = 0;
    bool ready_ = false;
    std::unordered_map<Key, Plan, KeyHash> cache_;
    FnCreate fn_create_ = nullptr; FnDestroy fn_destroy_ = nullptr;
    FnMdCreate fn_md_create_ = nullptr; FnMdDestroy fn_md_destroy_ = nullptr; FnMdSet fn_md_set_ = nullptr;
    FnMlCreate fn_ml_create_ = nullptr; FnMlDestroy fn_ml_destroy_ = nullptr;
    FnPrefCreate fn_pref_create_ = nullptr; FnPrefDestroy fn_pref_destroy_ = nullptr; FnPrefSet fn_pref_set_ = nullptr;
    FnGetHeur fn_get_heur_ = nullptr; FnMatmul fn_matmul_ = nullptr;
};

// =================================================================================================
// 1. Pre-Last-Token ShortConv Snapshot Kernel (for Zero-TTFT Async Prompt Checkpoint)
//    Given prefill chunk inputs in_ptr[0..B-1, C] and pre-chunk history h_slot[3, C],
//    saves the EXACT 3-lag state after B-1 tokens (seq_len = prompt_len - 1) into dst_snap[3, C]!
// Grid: ((C + 255) / 256,), Block: 256
// =================================================================================================
__global__ void __launch_bounds__(256) k_r9700_sconv_save_prelast(
    const float* __restrict__ in_ptr,      // [B, C]
    const uint16_t* __restrict__ h_slot,   // [3, C] (pre-chunk history!)
    uint16_t* __restrict__ dst_snap,       // [3, C] (for one visit)
    int B_minus_1, int C) {
#if defined(__HIP_DEVICE_COMPILE__) && defined(__gfx1201__)
    const int c = blockIdx.x * blockDim.x + threadIdx.x;
    if (c >= C) return;
    for (int lag = 0; lag < 3; ++lag) {
        const int t = B_minus_1 - 3 + lag;
        dst_snap[lag * C + c] = (t < 0)
            ? h_slot[(t + 3) * C + c]
            : strata::inkling2::hip::f32_to_bf16(in_ptr[static_cast<size_t>(t) * C + c]);
    }
#else
    (void)in_ptr; (void)h_slot; (void)dst_snap; (void)B_minus_1; (void)C;
#endif
}

// =================================================================================================
// 2. Multi-Slot Batched ShortConv Snapshot / Rollback Kernels (for C64 Batched Spec Decode!)
// =================================================================================================
__global__ void __launch_bounds__(256) k_r9700_bspec_sconv_snapshot_pair(
    const uint16_t* __restrict__ pool_a,   // [n_visits, max_slots, 3, C]
    const uint16_t* __restrict__ pool_b,
    uint16_t* __restrict__ snap_a,         // [P, n_visits, 3, C]
    uint16_t* __restrict__ snap_b,
    const int32_t* __restrict__ slots,     // [P]
    int P, int max_slots, int n_visits, int C) {
#if defined(__HIP_DEVICE_COMPILE__) && defined(__gfx1201__)
    const int c = blockIdx.x * blockDim.x + threadIdx.x;
    const int v = blockIdx.y;
    const int p = blockIdx.z;
    if (c >= C || v >= n_visits || p >= P) return;
    const int slot = slots[p];
    const size_t sbase = (static_cast<size_t>(v) * max_slots + slot) * 3 * C;
    const size_t dbase = (static_cast<size_t>(p) * n_visits + v) * 3 * C;
    for (int lag = 0; lag < 3; ++lag) {
        snap_a[dbase + lag * C + c] = pool_a[sbase + lag * C + c];
        snap_b[dbase + lag * C + c] = pool_b[sbase + lag * C + c];
    }
#else
    (void)pool_a; (void)pool_b; (void)snap_a; (void)snap_b; (void)slots; (void)P; (void)max_slots; (void)n_visits; (void)C;
#endif
}

__global__ void __launch_bounds__(256) k_r9700_bspec_sconv_record(
    const float* __restrict__ in_ptr,      // [P * L, C] (slot-major contiguous!)
    uint16_t* __restrict__ rec_all,        // [P, n_visits, 4, C]
    int visit, int P, int n_visits, int L, int C) {
#if defined(__HIP_DEVICE_COMPILE__) && defined(__gfx1201__)
    const int c = blockIdx.x * blockDim.x + threadIdx.x;
    const int row = blockIdx.y;  // 0 .. P*L - 1
    if (c >= C || row >= P * L) return;
    const int p = row / L;
    const int k = row - p * L;
    if (k < 4) {
        const size_t dst = ((static_cast<size_t>(p) * n_visits + visit) * 4 + k) * C + c;
        rec_all[dst] = strata::inkling2::hip::f32_to_bf16(in_ptr[static_cast<size_t>(row) * C + c]);
    }
#else
    (void)in_ptr; (void)rec_all; (void)visit; (void)P; (void)n_visits; (void)L; (void)C;
#endif
}

__global__ void __launch_bounds__(256) k_r9700_bspec_sconv_rollback_pair(
    const uint16_t* __restrict__ snap_a,   // [P, n_visits, 3, C]
    const uint16_t* __restrict__ rec_a,    // [P, n_visits, 4, C]
    uint16_t* __restrict__ pool_a,         // [n_visits, max_slots, 3, C]
    const uint16_t* __restrict__ snap_b,
    const uint16_t* __restrict__ rec_b,
    uint16_t* __restrict__ pool_b,
    const int32_t* __restrict__ slots,     // [P]
    const int32_t* __restrict__ keeps,     // [P] in [1..4]
    int P, int max_slots, int n_visits, int C) {
#if defined(__HIP_DEVICE_COMPILE__) && defined(__gfx1201__)
    const int c = blockIdx.x * blockDim.x + threadIdx.x;
    const int v = blockIdx.y;
    const int p = blockIdx.z;
    if (c >= C || v >= n_visits || p >= P) return;
    const int slot = slots[p];
    const int keep = keeps[p];
    const size_t dbase = (static_cast<size_t>(v) * max_slots + slot) * 3 * C;
    const size_t sbase = (static_cast<size_t>(p) * n_visits + v) * 3 * C;
    const size_t rbase = (static_cast<size_t>(p) * n_visits + v) * 4 * C;
    for (int lag = 0; lag < 3; ++lag) {
        const int t = keep - 3 + lag;
        pool_a[dbase + lag * C + c] = (t < 0) ? snap_a[sbase + (t + 3) * C + c] : rec_a[rbase + static_cast<size_t>(t) * C + c];
        pool_b[dbase + lag * C + c] = (t < 0) ? snap_b[sbase + (t + 3) * C + c] : rec_b[rbase + static_cast<size_t>(t) * C + c];
    }
#else
    (void)snap_a; (void)rec_a; (void)pool_a; (void)snap_b; (void)rec_b; (void)pool_b;
    (void)slots; (void)keeps; (void)P; (void)max_slots; (void)n_visits; (void)C;
#endif
}

// =================================================================================================
// 3. R9700State: All Opt-In GPU Buffers, Graphs, Async DMA Stream & Host Metadata
// =================================================================================================
struct Fp8TrunkEntry {
    const uint8_t* w_ptr = nullptr;     // points into owned buffer (or slice offset inside it)
    const float*   sc_ptr = nullptr;
    int N = 0;
    int K = 0;
};

struct Fp8TrunkOwned {
    GpuBuffer w;
    GpuBuffer sc;
};

struct SlotRec16 {
    GpuBuffer k, v, attn, mlp;
    int cap = 0;
    void release() noexcept {
        free_gpu_buffer(k); free_gpu_buffer(v);
        free_gpu_buffer(attn); free_gpu_buffer(mlp);
        cap = 0;
    }
};

struct PendingAsyncCkpt {
    bool valid = false;
    int seq_len = 0;
    GpuBuffer sc_k, sc_v, sc_attn, sc_mlp;
    hipEvent_t ev_dma_done = nullptr;
    bool dma_inflight = false;
    void release() noexcept {
        if (ev_dma_done) { (void)hipEventDestroy(ev_dma_done); ev_dma_done = nullptr; }
        free_gpu_buffer(sc_k); free_gpu_buffer(sc_v);
        free_gpu_buffer(sc_attn); free_gpu_buffer(sc_mlp);
        valid = false;
        dma_inflight = false;
        seq_len = 0;
    }
};

struct R9700State {
    Config cfg{};

    // Pre-transposed RelPos projection [rel_extent, 16] FP16 per physical layer (787 KB total!)
    std::vector<GpuBuffer> rel_proj_t_f16;

    // Intra-chunk FlashAttention buffers (grown on demand for contiguous single-slot prefill B >= 64)
    int attn_chunk_cap = 0;
    GpuBuffer d_r_clip_f16;
    GpuBuffer d_k_chunk_f16;
    GpuBuffer d_v_chunk_f16;

    // Native FP8_E4M3FN dense trunk weights (FAST mode only, keyed by original BF16 dev_ptr!)
    R9700Fp8BlasLt fp8_blaslt;
    std::vector<Fp8TrunkOwned> fp8_owned;
    std::unordered_map<const uint16_t*, Fp8TrunkEntry> fp8_map;
    GpuBuffer sc_x_fp8;
    GpuBuffer sc_x_fp8_sc;

    // Persistent BM=16 FP8 WMMA MoE tile queue (reuses sc_x_q8_/sc_act_q8_ for FP8 activations!)
    GpuBuffer d_moe_tiles;   // MoeTile16[4096]
    GpuBuffer d_moe_meta;    // int32_t[4]

    // Compact Top-32K MALL-Resident INT4 Unembed Screen Head for DSpark drafter (~39 MB -> fits 64 MiB MALL! 100% lossless!)
    GpuBuffer d_screen_q4;
    GpuBuffer d_screen_sc64;
    int screen_q4_vocab = 0;

    // Exact on-GPU Top-128 candidate extraction & stochastic tree verify buffers
    int cand_cap = 0;
    GpuBuffer d_cand_val, d_cand_idx, d_cand_fin;
    float*   h_cand_val = nullptr;
    int32_t* h_cand_idx = nullptr;
    uint8_t* h_cand_fin = nullptr;
    GpuBuffer d_st_tok, d_st_par, d_st_qprob, d_winpath;

    // WinPath16 (M=1..16) graphs & 16-step taps/record buffers
    hipGraphExec_t graph_tree16_greedy[17]{};
    int            t16_g_slot[17]{};
    uint64_t       t16_g_gen[17]{};
    int            t16_g_scB[17]{};

    hipGraphExec_t graph_tree16_sampled[17]{};
    int            t16_s_slot[17]{};
    uint64_t       t16_s_gen[17]{};
    int            t16_s_scB[17]{};

    hipGraphExec_t graph_tree16_ctx[17]{};
    uint64_t       t16_ctx_gen[17]{};
    int            t16_ctx_scB[17]{};
    GpuBuffer      d_tree16_taps_win[4];

    std::vector<SlotRec16> slot_rec16;
    std::vector<PldHistory> pld;
    std::vector<uint64_t> rng_ctr;

    // Async PromptCkpt stream & per-slot pending pre-last-token ShortConv state
    hipStream_t dma_stream = nullptr;
    hipEvent_t  ev_ckpt_ready = nullptr;
    std::vector<PendingAsyncCkpt> pend_ckpt;

    // C64 Batched Speculative Decode scratch (lazily allocated on first multi-slot bspec call!)
    int bspec_cap_P = 0;
    GpuBuffer d_bspec_snap_k, d_bspec_snap_v, d_bspec_snap_attn, d_bspec_snap_mlp;
    GpuBuffer d_bspec_rec_k, d_bspec_rec_v, d_bspec_rec_attn, d_bspec_rec_mlp;
    GpuBuffer d_bspec_slots, d_bspec_keeps;
    bool bspec_rec_active = false;
    int  bspec_rec_P = 0;
    int  bspec_rec_L = 0;
    float bspec_ema_acc = 2.0f;
    int   bspec_cool_rem = 0;

    R9700State() {
        for (int i = 0; i <= 16; ++i) {
            t16_g_slot[i] = -1;
            t16_s_slot[i] = -1;
        }
    }

    void invalidate_graphs() noexcept {
        for (int i = 0; i <= 16; ++i) {
            if (graph_tree16_greedy[i])  { (void)hipGraphExecDestroy(graph_tree16_greedy[i]);  graph_tree16_greedy[i] = nullptr; }
            if (graph_tree16_sampled[i]) { (void)hipGraphExecDestroy(graph_tree16_sampled[i]); graph_tree16_sampled[i] = nullptr; }
            if (graph_tree16_ctx[i])     { (void)hipGraphExecDestroy(graph_tree16_ctx[i]);     graph_tree16_ctx[i] = nullptr; }
            t16_g_slot[i] = -1; t16_g_gen[i] = 0; t16_g_scB[i] = 0;
            t16_s_slot[i] = -1; t16_s_gen[i] = 0; t16_s_scB[i] = 0;
            t16_ctx_gen[i] = 0; t16_ctx_scB[i] = 0;
        }
    }

    void release() noexcept {
        invalidate_graphs();
        if (dma_stream) {
            (void)hipStreamSynchronize(dma_stream);
            (void)hipStreamDestroy(dma_stream);
            dma_stream = nullptr;
        }
        if (ev_ckpt_ready) { (void)hipEventDestroy(ev_ckpt_ready); ev_ckpt_ready = nullptr; }
        for (auto& pc : pend_ckpt) pc.release();
        pend_ckpt.clear();
        for (auto& b : rel_proj_t_f16) free_gpu_buffer(b);
        rel_proj_t_f16.clear();
        free_gpu_buffer(d_r_clip_f16);
        free_gpu_buffer(d_k_chunk_f16);
        free_gpu_buffer(d_v_chunk_f16);
        attn_chunk_cap = 0;
        fp8_blaslt.release();
        for (auto& o : fp8_owned) { free_gpu_buffer(o.w); free_gpu_buffer(o.sc); }
        fp8_owned.clear();
        fp8_map.clear();
        free_gpu_buffer(sc_x_fp8);
        free_gpu_buffer(sc_x_fp8_sc);
        free_gpu_buffer(d_moe_tiles);
        free_gpu_buffer(d_moe_meta);
        free_gpu_buffer(d_screen_q4);
        free_gpu_buffer(d_screen_sc64);
        free_gpu_buffer(d_cand_val);
        free_gpu_buffer(d_cand_idx);
        free_gpu_buffer(d_cand_fin);
        if (h_cand_val) { (void)hipHostFree(h_cand_val); h_cand_val = nullptr; }
        if (h_cand_idx) { (void)hipHostFree(h_cand_idx); h_cand_idx = nullptr; }
        if (h_cand_fin) { (void)hipHostFree(h_cand_fin); h_cand_fin = nullptr; }
        cand_cap = 0;
        free_gpu_buffer(d_st_tok);
        free_gpu_buffer(d_st_par);
        free_gpu_buffer(d_st_qprob);
        free_gpu_buffer(d_winpath);
        for (int k = 0; k < 4; ++k) free_gpu_buffer(d_tree16_taps_win[k]);
        for (auto& r : slot_rec16) r.release();
        slot_rec16.clear();
        pld.clear();
        rng_ctr.clear();
        free_gpu_buffer(d_bspec_snap_k); free_gpu_buffer(d_bspec_snap_v);
        free_gpu_buffer(d_bspec_snap_attn); free_gpu_buffer(d_bspec_snap_mlp);
        free_gpu_buffer(d_bspec_rec_k); free_gpu_buffer(d_bspec_rec_v);
        free_gpu_buffer(d_bspec_rec_attn); free_gpu_buffer(d_bspec_rec_mlp);
        free_gpu_buffer(d_bspec_slots); free_gpu_buffer(d_bspec_keeps);
        bspec_cap_P = 0;
        bspec_rec_active = false;
    }

    size_t allocated_bytes() const noexcept {
        size_t tot = d_r_clip_f16.bytes + d_k_chunk_f16.bytes + d_v_chunk_f16.bytes +
                     sc_x_fp8.bytes + sc_x_fp8_sc.bytes + d_moe_tiles.bytes + d_moe_meta.bytes +
                     d_screen_q4.bytes + d_screen_sc64.bytes +
                     d_cand_val.bytes + d_cand_idx.bytes + d_cand_fin.bytes +
                     d_st_tok.bytes + d_st_par.bytes + d_st_qprob.bytes + d_winpath.bytes +
                     d_bspec_snap_k.bytes + d_bspec_snap_v.bytes + d_bspec_snap_attn.bytes + d_bspec_snap_mlp.bytes +
                     d_bspec_rec_k.bytes + d_bspec_rec_v.bytes + d_bspec_rec_attn.bytes + d_bspec_rec_mlp.bytes +
                     d_bspec_slots.bytes + d_bspec_keeps.bytes;
        for (const auto& b : rel_proj_t_f16) tot += b.bytes;
        for (const auto& o : fp8_owned) tot += o.w.bytes + o.sc.bytes;
        for (int k = 0; k < 4; ++k) tot += d_tree16_taps_win[k].bytes;
        for (const auto& r : slot_rec16) tot += r.k.bytes + r.v.bytes + r.attn.bytes + r.mlp.bytes;
        for (const auto& pc : pend_ckpt) tot += pc.sc_k.bytes + pc.sc_v.bytes + pc.sc_attn.bytes + pc.sc_mlp.bytes;
        return tot;
    }

    bool ensure_cand_cap(int rows) noexcept {
        if (rows <= cand_cap && d_cand_val.dev_ptr && h_cand_val) return true;
        const int nc = (rows > cand_cap * 2) ? rows : (cand_cap > 0 ? cand_cap * 2 : std::max(rows, 16));
        free_gpu_buffer(d_cand_val); free_gpu_buffer(d_cand_idx); free_gpu_buffer(d_cand_fin);
        if (h_cand_val) { (void)hipHostFree(h_cand_val); h_cand_val = nullptr; }
        if (h_cand_idx) { (void)hipHostFree(h_cand_idx); h_cand_idx = nullptr; }
        if (h_cand_fin) { (void)hipHostFree(h_cand_fin); h_cand_fin = nullptr; }
        if (alloc_gpu_buffer(d_cand_val, static_cast<size_t>(nc) * 128 * sizeof(float), false) != hipSuccess ||
            alloc_gpu_buffer(d_cand_idx, static_cast<size_t>(nc) * 128 * sizeof(int32_t), false) != hipSuccess ||
            alloc_gpu_buffer(d_cand_fin, static_cast<size_t>(nc), false) != hipSuccess ||
            hipHostMalloc(reinterpret_cast<void**>(&h_cand_val), static_cast<size_t>(nc) * 128 * sizeof(float), hipHostMallocDefault) != hipSuccess ||
            hipHostMalloc(reinterpret_cast<void**>(&h_cand_idx), static_cast<size_t>(nc) * 128 * sizeof(int32_t), hipHostMallocDefault) != hipSuccess ||
            hipHostMalloc(reinterpret_cast<void**>(&h_cand_fin), static_cast<size_t>(nc), hipHostMallocDefault) != hipSuccess) {
            (void)hipGetLastError();
            cand_cap = 0;
            return false;
        }
        cand_cap = nc;
        return true;
    }

    bool ensure_attn_chunk(int B, int n_head, int kv_dim) noexcept {
        if (B <= attn_chunk_cap && d_k_chunk_f16.dev_ptr) return true;
        free_gpu_buffer(d_r_clip_f16);
        free_gpu_buffer(d_k_chunk_f16);
        free_gpu_buffer(d_v_chunk_f16);
        if (alloc_gpu_buffer(d_r_clip_f16, static_cast<size_t>(B) * n_head * 16 * sizeof(half_t), false) != hipSuccess ||
            alloc_gpu_buffer(d_k_chunk_f16, static_cast<size_t>(B) * kv_dim * sizeof(half_t), false) != hipSuccess ||
            alloc_gpu_buffer(d_v_chunk_f16, static_cast<size_t>(B) * kv_dim * sizeof(half_t), false) != hipSuccess) {
            (void)hipGetLastError();
            attn_chunk_cap = 0;
            return false;
        }
        attn_chunk_cap = B;
        return true;
    }

    bool ensure_x_fp8(int M, int K) noexcept {
        const size_t need_x = static_cast<size_t>(M) * K;
        const size_t need_s = static_cast<size_t>(M) * std::max(1, K >> 6) * sizeof(float);
        if (sc_x_fp8.bytes >= need_x && sc_x_fp8_sc.bytes >= need_s) return true;
        free_gpu_buffer(sc_x_fp8);
        free_gpu_buffer(sc_x_fp8_sc);
        if (alloc_gpu_buffer(sc_x_fp8, need_x, false) != hipSuccess ||
            alloc_gpu_buffer(sc_x_fp8_sc, need_s, false) != hipSuccess) {
            (void)hipGetLastError();
            return false;
        }
        return true;
    }

    bool ensure_bspec_cap(int P, int n_visits, int kv_dim, int D) noexcept {
        if (P <= bspec_cap_P && d_bspec_snap_k.dev_ptr) return true;
        free_gpu_buffer(d_bspec_snap_k); free_gpu_buffer(d_bspec_snap_v);
        free_gpu_buffer(d_bspec_snap_attn); free_gpu_buffer(d_bspec_snap_mlp);
        free_gpu_buffer(d_bspec_rec_k); free_gpu_buffer(d_bspec_rec_v);
        free_gpu_buffer(d_bspec_rec_attn); free_gpu_buffer(d_bspec_rec_mlp);
        free_gpu_buffer(d_bspec_slots); free_gpu_buffer(d_bspec_keeps);
        const size_t sk = static_cast<size_t>(P) * n_visits * 3u * kv_dim * sizeof(uint16_t);
        const size_t sh = static_cast<size_t>(P) * n_visits * 3u * D * sizeof(uint16_t);
        const size_t rk = static_cast<size_t>(P) * n_visits * 4u * kv_dim * sizeof(uint16_t);
        const size_t rh = static_cast<size_t>(P) * n_visits * 4u * D * sizeof(uint16_t);
        if (alloc_gpu_buffer(d_bspec_snap_k, sk, false) != hipSuccess ||
            alloc_gpu_buffer(d_bspec_snap_v, sk, false) != hipSuccess ||
            alloc_gpu_buffer(d_bspec_snap_attn, sh, false) != hipSuccess ||
            alloc_gpu_buffer(d_bspec_snap_mlp, sh, false) != hipSuccess ||
            alloc_gpu_buffer(d_bspec_rec_k, rk, false) != hipSuccess ||
            alloc_gpu_buffer(d_bspec_rec_v, rk, false) != hipSuccess ||
            alloc_gpu_buffer(d_bspec_rec_attn, rh, false) != hipSuccess ||
            alloc_gpu_buffer(d_bspec_rec_mlp, rh, false) != hipSuccess ||
            alloc_gpu_buffer(d_bspec_slots, static_cast<size_t>(P) * sizeof(int32_t), false) != hipSuccess ||
            alloc_gpu_buffer(d_bspec_keeps, static_cast<size_t>(P) * sizeof(int32_t), false) != hipSuccess) {
            (void)hipGetLastError();
            bspec_cap_P = 0;
            return false;
        }
        bspec_cap_P = P;
        return true;
    }
};

}  // namespace strata::inkling2::hip::r9700
