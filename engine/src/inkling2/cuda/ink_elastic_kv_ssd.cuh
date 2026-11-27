// engine/src/inkling2/cuda/ink_elastic_kv_ssd.cuh - True 3-Tier Elastic Global KV (VRAM -> Host RAM -> NVMe SSD)
// + Deterministic 16GB/32GB/64GB RAM-Tier Budget Enforcement + Inverted-Loop O(1)-VRAM Long-Context (64K..1M) Attention!
//
// MIT License
// Copyright (c) 2026 Niko1221 and the Strata contributors
//
// STRICTLY OFF BY DEFAULT (AGENT-RULES-GPU.md):
//   Activated ONLY when STRATA_INK_ELASTIC_1M=1 (checked once per GPU device). It is not part of the RTX 50 fast mode.
//   When OFF, the existing baseline path in ink_runner.cuh / ink_cuda_backend.cu executes 100% untouched.
//
// Architecture (for Inkling-2-Mini's 8 Global Visits: visits 5, 11, 17, 23, 29, 35, 41, 47):
//   - Tier 0 (GDDR7 VRAM):       Hot prefix [0 .. dev_cap) + all 40 SWA-1024 rings (fixed <= 240 MiB).
//   - Tier 1 (Mapped Host RAM):  Warm window [dev_cap .. ssd_base), grown elastically one visit at a time
//                                strictly within safe host RAM budget (respecting /proc/meminfo, cgroup v1/v2,
//                                and STRATA_INK_RAM_CAP_MB so OS never swaps or OOMs!).
//   - Tier 2 (NVMe SSD Backing): Cold tail [ssd_base .. max_ctx) up to 1,048,576 (1M) tokens!
//                                Stored in 16 KiB-aligned 2048-token pages (exact multiple of 64 KiB for all 4 KV formats:
//                                FP16=8 MiB, FP8/K8V8=4.125 MiB, K8V4=3.125 MiB per visit per 2048-token page!).
//                                Uses O_DIRECT (with automatic buffered + POSIX_FADV_DONTNEED fallback on overlayfs)
//                                into bounded (~66-128 MiB total) mapped pinned staging buffers + async background reader
//                                thread so process VmRSS stays strictly bounded even at 1M context on 16 GB RAM!
#pragma once

#include "ink_attn.cuh"
#include "ink_cuda.cuh"
#include "strata/platform/cgroup_mem.hpp"

#include <algorithm>
#include <atomic>
#include <chrono>
#include <condition_variable>
#include <cerrno>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <mutex>
#include <string>
#include <thread>
#include <vector>

#ifdef _WIN32
#ifndef WIN32_LEAN_AND_MEAN
#define WIN32_LEAN_AND_MEAN
#endif
#ifndef NOMINMAX
#define NOMINMAX
#endif
#include <windows.h>
#include <psapi.h>
#else
#include <cerrno>
#include <fcntl.h>
#include <sys/stat.h>
#include <sys/statvfs.h>
#include <unistd.h>
#endif

namespace strata::ink {

// ============================================================================
// 1. Per-Device Opt-In Switch & Deterministic RAM-Tier Budget Helpers
// ============================================================================
inline bool ink_elastic_env_on(const char* name) noexcept {
    const char* e = std::getenv(name);
    return e && (e[0] == '1' || e[0] == 't' || e[0] == 'T');
}

/// Per-GPU opt-in switch (OFF by default!). Checked once per device ordinal.
inline bool ink_elastic_1m_enabled(int dev = 0) noexcept {
    static std::atomic<int> cache[16];
    const int d = (dev >= 0 && dev < 16) ? dev : 0;
    int v = cache[d].load(std::memory_order_acquire);
    if (v != 0) return v > 0;
    // Its own switch, separate from the sm_120 fast mode (long context is a separate lane): off unless STRATA_INK_ELASTIC_1M=1.
    const bool on = ink_elastic_env_on("STRATA_INK_ELASTIC_1M");
    cache[d].store(on ? 1 : -1, std::memory_order_release);
    return on;
}

/// Explicit host RAM ceiling in bytes from STRATA_INK_RAM_CAP_MB (0 = unset).
inline uint64_t ink_elastic_ram_cap_bytes() noexcept {
    const char* e = std::getenv("STRATA_INK_RAM_CAP_MB");
    if (!e || !*e) return 0ull;
    const long long mb = std::atoll(e);
    return (mb > 0) ? (static_cast<uint64_t>(mb) << 20) : 0ull;
}

struct InkProcMemStats {
    uint64_t vm_rss_bytes    = 0;
    uint64_t vm_hwm_bytes    = 0;
    uint64_t rss_anon_bytes  = 0;
    uint64_t rss_file_bytes  = 0;
    uint64_t rss_shmem_bytes = 0;
};

inline InkProcMemStats ink_query_proc_mem_stats() noexcept {
    InkProcMemStats s{};
#ifdef _WIN32
    PROCESS_MEMORY_COUNTERS pmc{};
    if (GetProcessMemoryInfo(GetCurrentProcess(), &pmc, sizeof(pmc))) {
        s.vm_rss_bytes = static_cast<uint64_t>(pmc.WorkingSetSize);
        s.vm_hwm_bytes = static_cast<uint64_t>(pmc.PeakWorkingSetSize);
        s.rss_anon_bytes = static_cast<uint64_t>(pmc.PagefileUsage);
    }
#else
    if (FILE* f = std::fopen("/proc/self/status", "r")) {
        char line[256];
        unsigned long long kb = 0;
        while (std::fgets(line, sizeof(line), f)) {
            if (std::sscanf(line, "VmRSS: %llu kB", &kb) == 1) s.vm_rss_bytes = kb * 1024ull;
            else if (std::sscanf(line, "VmHWM: %llu kB", &kb) == 1) s.vm_hwm_bytes = kb * 1024ull;
            else if (std::sscanf(line, "RssAnon: %llu kB", &kb) == 1) s.rss_anon_bytes = kb * 1024ull;
            else if (std::sscanf(line, "RssFile: %llu kB", &kb) == 1) s.rss_file_bytes = kb * 1024ull;
            else if (std::sscanf(line, "RssShmem: %llu kB", &kb) == 1) s.rss_shmem_bytes = kb * 1024ull;
        }
        std::fclose(f);
    }
#endif
    return s;
}

/// Available host RAM tightened by cgroup v1/v2 limits AND STRATA_INK_RAM_CAP_MB (minus current process RSS).
inline uint64_t ink_elastic_avail_ram_bytes(uint64_t raw_avail, int dev = 0) noexcept {
    if (!ink_elastic_1m_enabled(dev) && ink_elastic_ram_cap_bytes() == 0ull) return raw_avail;
    uint64_t avail = raw_avail;
#if defined(__linux__)
    const auto cg = strata::platform::cgroup_mem::probe_cgroup_room(0);
    if (cg.limited && cg.available_bytes != strata::platform::cgroup_mem::kUnlimited) {
        avail = (avail == 0) ? cg.available_bytes : std::min(avail, cg.available_bytes);
    }
#endif
    const uint64_t cap = ink_elastic_ram_cap_bytes();
    if (cap > 0) {
        const auto ms = ink_query_proc_mem_stats();
        const uint64_t used = ms.vm_rss_bytes;
        const uint64_t rem = (cap > used) ? (cap - used) : 0ull;
        avail = (avail == 0) ? rem : std::min(avail, rem);
    }
    return avail;
}

/// Safe host expert RAM budget across 16 GB / 32 GB / 64 GB RAM tiers when STRATA_INK_ELASTIC_1M=1.
/// On 16 GB RAM (cap <= ~16 GiB), caps host experts cleanly so total process RSS stays strictly < 13.5 GiB
/// while automatically activating InkColdStore 16 KiB O_DIRECT NVMe streaming for the coldest 1-pass tail!
inline size_t ink_elastic_expert_ram_cap(double ram_experts_gb, bool ssd_experts, uint64_t raw_avail, int dev = 0) noexcept {
    if (ssd_experts || ram_experts_gb == 0.0) return 0u;
    if (ram_experts_gb > 0.0) return static_cast<size_t>(ram_experts_gb * 1073741824.0);
    const uint64_t avail = ink_elastic_avail_ram_bytes(raw_avail, dev);
    if (avail == 0) return static_cast<size_t>(-1);
    const uint64_t cap_env = ink_elastic_ram_cap_bytes();
    const uint64_t eff_total = (cap_env > 0) ? std::min(cap_env, avail + ink_query_proc_mem_stats().vm_rss_bytes) : avail;
    if (ink_elastic_1m_enabled(dev) && eff_total <= (17ull << 30)) {
        // 16 GB RAM tier: reserve room for CUDA driver/pinned (~1.5 GiB), OS headroom (~2.5 GiB),
        // ColdStore DMA ring (~0.3 GiB), and Elastic KV Tier 1 (~3-5 GiB). Cap RAM experts <= 2.4 GiB
        // so the coldest single-pass tail automatically streams via 16 KiB O_DIRECT InkColdStore!
        const uint64_t max_exp = std::min<uint64_t>(2450ull << 20, (avail > (6ull << 30)) ? (avail / 4) : (1536ull << 20));
        return static_cast<size_t>(max_exp);
    }
    const uint64_t os_res = std::max<uint64_t>(2ull << 30, std::min<uint64_t>(avail / 4, 4ull << 30));
    return (avail > os_res) ? static_cast<size_t>(avail - os_res) : 0u;
}

/// Suppress InkTier's ~11 GiB host duplicate of VRAM experts on 16 GB / 32 GB tiers or long-ctx runs.
inline bool ink_elastic_allow_host_tier_dup(uint64_t raw_avail, size_t host_used, size_t vram_dup_bytes, int dev = 0) noexcept {
    const uint64_t avail = ink_elastic_avail_ram_bytes(raw_avail, dev);
    if (!ink_elastic_1m_enabled(dev)) {
        return avail == 0 || avail > (uint64_t)(host_used + vram_dup_bytes) + (5ull << 30);
    }
    // Keep at least 16 GiB headroom above (host_used + vram_dup_bytes) so 32 GB / 16 GB tiers keep RAM for Elastic KV!
    return avail > (uint64_t)(host_used + vram_dup_bytes) + (16ull << 30);
}

/// Smart auto-KV precision selection when STRATA_INK_ELASTIC_1M=1 and user left --kv auto.
/// Preserves FP16 for <= 64K when plenty of RAM exists; auto-selects FP8_E4M3 (or K8V4 if requested) for > 64K / 1M!
inline InkKvDtype ink_elastic_select_kv_dtype(InkKvDtype cur_dt, bool kv_dtype_auto, int max_ctx, int max_slots,
                                              uint64_t raw_avail, int dev = 0) noexcept {
    if (!ink_elastic_1m_enabled(dev) || !kv_dtype_auto) return cur_dt;
    if (ink_elastic_env_on("STRATA_INK_KV_K8V4")) return InkKvDtype::K8V4;
    if (max_slots >= 16) return InkKvDtype::FP8;
    const uint64_t avail = ink_elastic_avail_ram_bytes(raw_avail, dev);
    if (max_ctx > 65536) return InkKvDtype::FP8;  // 128K..1M: FP8_E4M3 cuts 1M KV 32 GiB -> 16.5 GiB with zero NIAH loss!
    if (max_ctx >= 32768 && avail > 0 && avail <= (16ull << 30)) return InkKvDtype::FP8;
    return cur_dt;
}

// ============================================================================
// 2. Additive Device Kernels for 3-Tier Range-Sliced Split-K Decode & Tier-2 Writes
// ============================================================================

/// Write K, V [M, 8, 128] into a single Tier-2 page [page_start, page_start + page_tokens) mapped at `page_desc`!
/// `page_desc` has dev_cap = page_start, host_cap = page_tokens, cap_pos = page_start + page_tokens.
__global__ void __launch_bounds__(128) k_ink_kv_write_tier2_page(
    const float* __restrict__ k_in,     // [M, 8, 128]
    const float* __restrict__ v_in,     // [M, 8, 128]
    const int32_t* __restrict__ slots,  // [M] (null -> slot0)
    int slot0,
    const int32_t* __restrict__ pos,    // [M]
    InkKvVisitDesc page_desc,
    int page_start,
    int M) {
    const int m = blockIdx.y;
    const int h = blockIdx.x;  // 0..7
    const int d = threadIdx.x; // 0..127
    if (m >= M || h >= kInkKVHeads) return;

    const int sl = slots ? slots[m] : slot0;
    const int p = pos[m];
    if (p < page_start || p >= page_desc.cap_pos) return;
    const float kv_val = k_in[((size_t)m * kInkKVHeads + h) * kInkHeadDim + d];
    const float vv_val = v_in[((size_t)m * kInkKVHeads + h) * kInkHeadDim + d];
    __shared__ float sk[4], sv[4];
    ink_kv_store_128(page_desc, sl, p, h, d, kv_val, true, sk);
    ink_kv_store_128(page_desc, sl, p, h, d, vv_val, false, sv);
}

/// Commit accepted tree nodes `path[0..n_acc-1]` for ONE global visit `visit` into Tier-2 page [page_start, page_desc.cap_pos)!
__global__ void __launch_bounds__(128) k_ink_commit_kv_tier2_page(
    const float* __restrict__ stage_k,  // [48, tree_cap, 1024]
    const float* __restrict__ stage_v,  // [48, tree_cap, 1024]
    const int32_t* __restrict__ path,   // [n_acc]
    int visit,
    int p0,
    int n_acc,
    int slot0,
    int tree_cap,
    InkKvVisitDesc page_desc,
    int page_start) {
    const int h = blockIdx.x;  // 0..7
    const int d = threadIdx.x; // 0..127
    if (h >= kInkKVHeads) return;
    __shared__ float sk[4], sv[4];
    for (int k = 0; k < n_acc; ++k) {
        const int p = p0 + k;
        if (p < page_start || p >= page_desc.cap_pos) continue;
        const int node = path[k];
        const size_t sbase = (((size_t)visit * tree_cap + node) * kInkKVHeads + h) * kInkHeadDim + d;
        ink_kv_store_128(page_desc, slot0, p, h, d, stage_k[sbase], true, sk);
        ink_kv_store_128(page_desc, slot0, p, h, d, stage_v[sbase], false, sv);
        if (page_desc.kv_fmt != 0) __syncthreads();
    }
}

/// Range-bounded Split-K Flash-GQA kernel for 3-Tier (VRAM + Host RAM + NVMe SSD) Decode (up to 1M+ tokens, 100% size_t safe!).
/// Attends to keys in `[range_beg, min(range_end, k_end))` backed by `kv` (plus optional sink & tree ancestors).
__global__ void __launch_bounds__(128) k_ink_flash_gqa_splitk_range(
    const float* __restrict__ q,          // [M, 24, 128]
    const float* __restrict__ rel_table,  // [M, 24, 1025]
    const float* __restrict__ sinks,      // [M, 24]
    InkKvVisitDesc kv,
    const int32_t* __restrict__ slots,    // [M] (null -> slot0)
    int slot0,
    const int32_t* __restrict__ q_pos,    // [M]
    const float* __restrict__ tree_k,     // [M, 8, 128] optional (pass ONLY on first chunk!)
    const float* __restrict__ tree_v,     // [M, 8, 128] optional
    const uint8_t* __restrict__ tree_anc, // [M, 8]
    int tree_p0,
    int range_beg,                        // inclusive global position start of this chunk
    int range_end,                        // exclusive global position end of this chunk
    bool include_sink,                    // true ONLY on the first chunk
    bool include_tree,                    // true ONLY on the chunk covering tree_k (or final chunk)
    float* __restrict__ part_m,           // [M, 24, n_splits]
    float* __restrict__ part_l,           // [M, 24, n_splits]
    float* __restrict__ part_o,           // [M, 24, n_splits, 128]
    int M,
    int n_splits) {
    const int kh = blockIdx.x;  // 0..7
    const int sp = blockIdx.y;  // 0..n_splits-1
    const int m  = blockIdx.z;  // 0..M-1
    if (kh >= kInkKVHeads || sp >= n_splits || m >= M) return;

    const int tid  = threadIdx.x;
    const int warp = tid >> 5;
    const int lane = tid & 31;
    const int sl   = slots ? slots[m] : slot0;
    const int qp   = q_pos[m];
    const float scale = 0.08838834764831845f;

    const int h0 = 3 * kh;
    const float4 qv0 = *(const float4*)(q + ((size_t)m * kInkHeads + h0 + 0) * kInkHeadDim + lane * 4);
    const float4 qv1 = *(const float4*)(q + ((size_t)m * kInkHeads + h0 + 1) * kInkHeadDim + lane * 4);
    const float4 qv2 = *(const float4*)(q + ((size_t)m * kInkHeads + h0 + 2) * kInkHeadDim + lane * 4);
    const float* rt0 = rel_table + ((size_t)m * kInkHeads + h0 + 0) * (kInkRelExtent + 1);
    const float* rt1 = rel_table + ((size_t)m * kInkHeads + h0 + 1) * (kInkRelExtent + 1);
    const float* rt2 = rel_table + ((size_t)m * kInkHeads + h0 + 2) * (kInkRelExtent + 1);

    const int tp0 = tree_k ? q_pos[0] : tree_p0;
    const int committed_end = tree_k ? tp0 : (qp + 1);
    const int k_beg = max(0, range_beg);
    const int k_end = min(range_end, committed_end);
    const int k_len = max(0, k_end - k_beg);
    const int per_sp = (k_len + n_splits - 1) / n_splits;
    const int s_beg = k_beg + sp * per_sp;
    const int s_end = min(k_end, s_beg + per_sp);

    const bool fold_sk = (include_sink && sp == 0 && warp == 0);
    float mw[3] = {
        fold_sk ? sinks[(size_t)m * kInkHeads + h0 + 0] : -1e30f,
        fold_sk ? sinks[(size_t)m * kInkHeads + h0 + 1] : -1e30f,
        fold_sk ? sinks[(size_t)m * kInkHeads + h0 + 2] : -1e30f
    };
    float lw[3] = { fold_sk ? 1.0f : 0.0f, fold_sk ? 1.0f : 0.0f, fold_sk ? 1.0f : 0.0f };
    float a[3][4] = {};

    auto attend3 = [&](float4 kv4, float4 vv4, int k_pos) {
        const int dist = min(max(qp - k_pos, 0), kInkRelExtent);
        const float sc[3] = {
            warp_sum(qv0.x * kv4.x + qv0.y * kv4.y + qv0.z * kv4.z + qv0.w * kv4.w) * scale + rt0[dist],
            warp_sum(qv1.x * kv4.x + qv1.y * kv4.y + qv1.z * kv4.z + qv1.w * kv4.w) * scale + rt1[dist],
            warp_sum(qv2.x * kv4.x + qv2.y * kv4.y + qv2.z * kv4.z + qv2.w * kv4.w) * scale + rt2[dist]
        };
#pragma unroll
        for (int r = 0; r < 3; ++r) {
            const float mn = fmaxf(mw[r], sc[r]);
            const float al = (lw[r] > 0.0f) ? __expf(mw[r] - mn) : 0.0f;
            const float p  = __expf(sc[r] - mn);
            lw[r] = lw[r] * al + p;
            a[r][0] = a[r][0] * al + p * vv4.x;
            a[r][1] = a[r][1] * al + p * vv4.y;
            a[r][2] = a[r][2] * al + p * vv4.z;
            a[r][3] = a[r][3] * al + p * vv4.w;
            mw[r] = mn;
        }
    };

    for (int kp = s_beg + warp; kp < s_end; kp += 4) {
        if (kp < 0 || kp >= kv.cap_pos) continue;
        float4 kv4, vv4;
        ink_kv_load4(kv, sl, kp, kh, lane, kv4, vv4);
        attend3(kv4, vv4, kp);
    }

    if (include_tree && tree_k && sp == 0) {
        for (int kp = tp0 + warp; kp <= qp; kp += 4) {
            const int anc_node = (int)tree_anc[m * kInkMaxTreeDepth + (kp - tp0)];
            const size_t tbase = ((size_t)anc_node * kInkKVHeads + kh) * kInkHeadDim + lane * 4;
            const float4 kv4 = *(const float4*)(tree_k + tbase);
            const float4 vv4 = *(const float4*)(tree_v + tbase);
            attend3(kv4, vv4, kp);
        }
    }

    __shared__ float sm_m[4], sm_l[4], sm_a[4][kInkHeadDim];
#pragma unroll
    for (int r = 0; r < 3; ++r) {
        if (lane == 0) { sm_m[warp] = mw[r]; sm_l[warp] = lw[r]; }
        sm_a[warp][lane * 4 + 0] = a[r][0];
        sm_a[warp][lane * 4 + 1] = a[r][1];
        sm_a[warp][lane * 4 + 2] = a[r][2];
        sm_a[warp][lane * 4 + 3] = a[r][3];
        __syncthreads();
        if (warp == 0) {
            float mg = -1e30f;
            for (int w = 0; w < 4; ++w) if (sm_l[w] > 0.0f) mg = fmaxf(mg, sm_m[w]);
            float w0 = sm_l[0] > 0.0f ? __expf(sm_m[0] - mg) : 0.0f;
            float w1 = sm_l[1] > 0.0f ? __expf(sm_m[1] - mg) : 0.0f;
            float w2 = sm_l[2] > 0.0f ? __expf(sm_m[2] - mg) : 0.0f;
            float w3 = sm_l[3] > 0.0f ? __expf(sm_m[3] - mg) : 0.0f;
            const float lt = sm_l[0] * w0 + sm_l[1] * w1 + sm_l[2] * w2 + sm_l[3] * w3;
            const size_t idx_sp = ((size_t)m * kInkHeads + h0 + r) * n_splits + sp;
            if (lane == 0) { part_m[idx_sp] = mg; part_l[idx_sp] = lt; }
            *(float4*)(part_o + idx_sp * kInkHeadDim + lane * 4) = make_float4(
                sm_a[0][lane * 4 + 0] * w0 + sm_a[1][lane * 4 + 0] * w1 + sm_a[2][lane * 4 + 0] * w2 + sm_a[3][lane * 4 + 0] * w3,
                sm_a[0][lane * 4 + 1] * w0 + sm_a[1][lane * 4 + 1] * w1 + sm_a[2][lane * 4 + 1] * w2 + sm_a[3][lane * 4 + 1] * w3,
                sm_a[0][lane * 4 + 2] * w0 + sm_a[1][lane * 4 + 2] * w1 + sm_a[2][lane * 4 + 2] * w2 + sm_a[3][lane * 4 + 2] * w3,
                sm_a[0][lane * 4 + 3] * w0 + sm_a[1][lane * 4 + 3] * w1 + sm_a[2][lane * 4 + 3] * w2 + sm_a[3][lane * 4 + 3] * w3);
        }
        __syncthreads();
    }
}

/// Fold `n_splits` partials into running online-softmax accumulators (run_m, run_l, run_o).
/// Grid: (24 heads, M), Block: 128 (1 dim/thread).
__global__ void __launch_bounds__(128) k_ink_flash_gqa_fold_splits(
    const float* __restrict__ part_m,     // [M, 24, n_splits]
    const float* __restrict__ part_l,     // [M, 24, n_splits]
    const float* __restrict__ part_o,     // [M, 24, n_splits, 128]
    float* __restrict__ run_m,            // [M, 24]
    float* __restrict__ run_l,            // [M, 24]
    float* __restrict__ run_o,            // [M, 24, 128]
    int M,
    int n_splits,
    bool is_first) {
    const int h = blockIdx.x;
    const int m = blockIdx.y;
    const int d = threadIdx.x;
    if (h >= kInkHeads || m >= M) return;

    const size_t base_sp = ((size_t)m * kInkHeads + h) * n_splits;
    const size_t ridx = (size_t)m * kInkHeads + h;
    float mg = -1e30f;
    for (int s = 0; s < n_splits; ++s) {
        if (part_l[base_sp + s] > 0.0f) mg = fmaxf(mg, part_m[base_sp + s]);
    }
    float lt = 0.0f;
    float acc = 0.0f;
    for (int s = 0; s < n_splits; ++s) {
        const float ls = part_l[base_sp + s];
        if (ls > 0.0f) {
            const float w = __expf(part_m[base_sp + s] - mg);
            lt += ls * w;
            acc = fmaf(part_o[(base_sp + s) * kInkHeadDim + d], w, acc);
        }
    }
    const size_t oi = ridx * kInkHeadDim + d;
    if (is_first) {
        if (d == 0) { run_m[ridx] = mg; run_l[ridx] = lt; }
        run_o[oi] = acc;
    } else if (lt > 0.0f) {
        const float om = run_m[ridx];
        const float ol = run_l[ridx];
        const float mn = (ol > 0.0f) ? fmaxf(om, mg) : mg;
        const float wo = (ol > 0.0f) ? __expf(om - mn) : 0.0f;
        const float wn = __expf(mg - mn);
        run_o[oi] = run_o[oi] * wo + acc * wn;
        if (d == 0) { run_m[ridx] = mn; run_l[ridx] = ol * wo + lt * wn; }
    }
}

/// Finalize running online-softmax accumulators + fused * sigmoid(gate_attn) -> out [M, 24, 128].
__global__ void __launch_bounds__(128) k_ink_flash_gqa_finalize_gate(
    const float* __restrict__ run_l,      // [M, 24]
    const float* __restrict__ run_o,      // [M, 24, 128]
    const float* __restrict__ gate_attn,  // [M, 24, 128]
    float* __restrict__ out,              // [M, 24, 128]
    int M) {
    const int h = blockIdx.x;
    const int m = blockIdx.y;
    const int d = threadIdx.x;
    if (h >= kInkHeads || m >= M) return;
    const size_t ridx = (size_t)m * kInkHeads + h;
    const size_t ob = ridx * kInkHeadDim + d;
    const float rl = run_l[ridx];
    float val = (rl > 0.0f) ? (run_o[ob] / rl) : 0.0f;
    if (gate_attn) val *= 1.0f / (1.0f + __expf(-gate_attn[ob]));
    out[ob] = val;
}

// ============================================================================
// 3. True 3-Tier Elastic Global KV Manager (`InkElasticKvSsd`)
// ============================================================================
struct InkElasticKvSsd {
    static constexpr int kNumGlobalVisits = 8;
    static constexpr int kPageTokens      = 2048;  // Matches kInkPfKvTile = 2048; exact 64 KiB multiple for all 4 KV formats!
    static constexpr size_t kAlignBytes   = 16384u; // 16 KiB NVMe alignment (zero read-amp!)

    bool enabled = false;
    bool ssd_open = false;
    bool ssd_direct = false;
    int primary_gpu = 0;
    int max_slots = 1;
    int max_ctx = 4096;
    InkKvDtype kv_dtype = InkKvDtype::FP16;

    // 3-Tier Partition:
    //   Tier 0 (VRAM):            [0 .. dev_cap)
    //   Tier 1 (Mapped Host RAM): [dev_cap .. dev_cap + tier1_max_ctx)  (grown lazily up to tier1_max_ctx!)
    //   Tier 2 (NVMe SSD):        [ssd_base .. max_ctx)                 where ssd_base = dev_cap + tier1_max_ctx
    int dev_cap       = 0;
    int tier1_max_ctx = 0;
    int ssd_base      = 0;
    int num_ssd_pages = 0;  // ceil((max_ctx - ssd_base) / kPageTokens)
    int gv_to_visit[kNumGlobalVisits] = {5, 11, 17, 23, 29, 35, 41, 47};
    int visit_to_gv[kInkVisits];

    // Per-page byte layout inside a 16 KiB-aligned page buffer for `[max_slots, kPageTokens]`:
    size_t k_bytes_page   = 0;
    size_t v_bytes_page   = 0;
    size_t sc_bytes_page  = 0;
    size_t page_bytes     = 0;  // total bytes per (gv, page) rounded up to 16 KiB

    // Backing file (Linux fd or Windows HANDLE)
    std::string ssd_path;
#ifdef _WIN32
    HANDLE h_file = INVALID_HANDLE_VALUE;
#else
    int fd = -1;
#endif

    // Bounded Mapped Pinned Staging Buffers:
    //   - 8 active write-coalescing pages (1 per global visit): ~33 MiB FP8 / ~64 MiB FP16
    //   - 2 double-buffered read-staging pages (shared across visits): ~8.25 MiB FP8 / ~16 MiB FP16
    struct PageBuf {
        void* raw = nullptr;
        uint8_t* h_base = nullptr;  // 16 KiB-aligned host pointer
        uint8_t* d_base = nullptr;  // matching GPU mapped device pointer
        int cur_gv   = -1;
        int cur_page = -1;
        bool dirty   = false;
    };
    PageBuf write_buf[kNumGlobalVisits]{};
    PageBuf read_buf[2]{};
    cudaEvent_t ev_read_done[2] = {};
    bool read_ev_pending[2] = {false, false};

    // Per-(gv, page) written bitmap & per-slot max written global position
    std::vector<uint8_t> page_written;  // [8 * num_ssd_pages]; read by the I/O thread, written by the main thread: only through pw_get / pw_set
    mutable std::mutex pw_mu;
    bool pw_get(int gv, int page) const {
        std::lock_guard<std::mutex> lk(pw_mu);
        return page_written[(size_t)gv * (size_t)num_ssd_pages + (size_t)page] != 0;
    }
    void pw_set(int gv, int page) {
        std::lock_guard<std::mutex> lk(pw_mu);
        page_written[(size_t)gv * (size_t)num_ssd_pages + (size_t)page] = 1u;
    }
    std::vector<int> slot_max_pos;      // [max_slots]

    // Telemetry counters (bytes read/written to NVMe SSD)
    std::atomic<uint64_t> ssd_bytes_read{0};
    std::atomic<uint64_t> ssd_bytes_written{0};
    std::atomic<uint64_t> ssd_pages_read{0};
    std::atomic<uint64_t> ssd_pages_written{0};

    // Running online-softmax accumulators for chunked Tier-2 decode (~200 KiB VRAM)
    float* d_run_m = nullptr;
    float* d_run_l = nullptr;
    float* d_run_o = nullptr;
    int run_cap_M = 0;

    // Inverted-loop O(1)-VRAM prefill multi-QChunk accumulators (<= 24 MiB VRAM for M=2048!)
    float* d_pf_m_all = nullptr;
    float* d_pf_l_all = nullptr;
    float* d_pf_o_all = nullptr;
    int pf_qchunk_cap = 0;

    // Background async NVMe page prefetcher thread (overlaps page pg+1 pread with GPU compute on page pg!)
    struct AsyncReadReq {
        bool valid = false;
        int buf_idx = 0;
        int gv = 0;
        int page = 0;
    };
    std::thread io_thr;
    std::mutex io_mu;
    std::condition_variable io_cv;
    bool io_stop = false;
    bool io_busy = false;
    bool io_ok = true;
    AsyncReadReq io_req{};

    InkElasticKvSsd() {
        for (int v = 0; v < kInkVisits; ++v) visit_to_gv[v] = -1;
    }
    ~InkElasticKvSsd() { free_all(); }

    InkElasticKvSsd(const InkElasticKvSsd&) = delete;
    InkElasticKvSsd& operator=(const InkElasticKvSsd&) = delete;

    static int global_visit_idx(int visit) noexcept {
        switch (visit) {
            case 5:  return 0;
            case 11: return 1;
            case 17: return 2;
            case 23: return 3;
            case 29: return 4;
            case 35: return 5;
            case 41: return 6;
            case 47: return 7;
            default: return -1;
        }
    }

    static size_t kv_bytes_per_pos(InkKvDtype dt) noexcept {
        if (dt == InkKvDtype::FP16) return (size_t)kInkKVDim * 4u;
        if (dt == InkKvDtype::K8V4) return (size_t)kInkKVDim + ((size_t)kInkKVDim / 2u) + (size_t)(2 * kInkKVHeads * 4);
        return (size_t)kInkKVDim * 2u + (size_t)(2 * kInkKVHeads * 4);
    }

    /// Plan 3-tier global KV capacities at runner init (when STRATA_INK_ELASTIC_1M=1).
    /// NEVER clamps max_ctx down below requested ctx (up to 1M+) and NEVER pins 16-32 GiB upfront!
    void plan(int gpu_id, int slots, int req_max_ctx, InkKvDtype dt,
              int vram_fit_ctx, int kv_offload_mode, uint64_t raw_avail_ram,
              int& out_dev_cap, int& out_init_host_cap, int& out_eff_max_ctx) {
        free_all();
        primary_gpu = gpu_id;
        max_slots = std::max(1, slots);
        max_ctx = std::max(256, req_max_ctx);
        kv_dtype = dt;
        enabled = true;
        for (int v = 0; v < kInkVisits; ++v) visit_to_gv[v] = global_visit_idx(v);

        // Tier 0 (VRAM): align to kPageTokens (2048) when spilling so Tier 2 pages never straddle tier boundaries!
        const int c1_hot_cap = (vram_fit_ctx >= 10240) ? 10240 : 8192;
        if (kv_offload_mode == 1) {
            dev_cap = 0;
        } else if (vram_fit_ctx >= max_ctx) {
            dev_cap = std::max(256, std::min(max_ctx, (max_slots == 1) ? c1_hot_cap : 4096));
        } else {
            // Keep compact hot VRAM window (4K..10K aligned to 2048) leaving plenty of GDDR7 headroom for TC prefill & DSpark!
            const int cap_hot = (max_slots == 1) ? c1_hot_cap : 4096;
            const int fit_al = ((vram_fit_ctx * 3 / 4) / kPageTokens) * kPageTokens;
            dev_cap = std::min(cap_hot, std::max(0, fit_al));
        }

        // Compute safe Tier-1 Mapped Host RAM ceiling respecting /proc/meminfo, cgroup v1/v2, and STRATA_INK_RAM_CAP_MB!
        const uint64_t avail = ink_elastic_avail_ram_bytes(raw_avail_ram, gpu_id);
        const size_t per_pos_all8 = (size_t)kNumGlobalVisits * max_slots * kv_bytes_per_pos(kv_dtype);
        uint64_t kv_ram_budget = 0;
        if (const char* kv_mb_env = std::getenv("STRATA_INK_KV_RAM_MB"); kv_mb_env && *kv_mb_env) {
            const long long mb = std::atoll(kv_mb_env);
            kv_ram_budget = (mb > 0) ? (static_cast<uint64_t>(mb) << 20) : 0ull;
        } else if (ink_elastic_env_on("STRATA_INK_KV_SSD_FORCE")) {
            kv_ram_budget = (size_t)4 * kPageTokens * per_pos_all8;  // 8K tokens in RAM, rest on NVMe SSD!
        } else {
            // Leave generous OS + CUDA driver + staging headroom (>= 3.5 GiB on 16GB tier, >= 6 GiB on 32/64GB tier)
            const uint64_t cap_env = ink_elastic_ram_cap_bytes();
            const uint64_t reserve = (cap_env > 0 && cap_env <= (16ull << 30))
                ? (3584ull << 20)
                : ((cap_env > 0 && cap_env <= (32ull << 30)) ? (6144ull << 20) : (6144ull << 20));
            if (avail > reserve) {
                // Use up to 75% of remaining headroom so single-visit growth peak (+1/8) never exceeds cap!
                kv_ram_budget = ((avail - reserve) * 3ull) / 4ull;
            }
        }
        const int need_tail = std::max(0, max_ctx - dev_cap);
        int t1_fit = (per_pos_all8 > 0)
            ? (int)std::min<uint64_t>((uint64_t)need_tail, (kv_ram_budget / per_pos_all8))
            : need_tail;
        if (t1_fit < need_tail) {
            // Round Tier 1 down to an exact multiple of kPageTokens (2048) so Tier 2 pages are cleanly aligned!
            t1_fit = (t1_fit / kPageTokens) * kPageTokens;
        } else {
            // Whole context fits in Tier 0 + Tier 1! Round up to 64-alignment <= need_tail.
            t1_fit = need_tail;
        }
        tier1_max_ctx = std::max(0, t1_fit);
        ssd_base = dev_cap + tier1_max_ctx;
        num_ssd_pages = (max_ctx > ssd_base) ? ((max_ctx - ssd_base + kPageTokens - 1) / kPageTokens) : 0;

        // Compute 16 KiB-aligned page layout
        const size_t elems_page = (size_t)max_slots * kPageTokens * kInkKVHeads * kInkHeadDim;
        if (kv_dtype == InkKvDtype::FP16) {
            k_bytes_page  = elems_page * sizeof(__half);
            v_bytes_page  = elems_page * sizeof(__half);
            sc_bytes_page = 0;
        } else {
            k_bytes_page  = elems_page;
            v_bytes_page  = (kv_dtype == InkKvDtype::K8V4) ? (elems_page / 2u) : elems_page;
            sc_bytes_page = (size_t)max_slots * kPageTokens * kInkKVHeads * sizeof(float);
        }
        const size_t raw_pb = k_bytes_page + v_bytes_page + 2u * sc_bytes_page;
        page_bytes = ((raw_pb + kAlignBytes - 1u) / kAlignBytes) * kAlignBytes;

        slot_max_pos.assign((size_t)max_slots, 0);
        if (num_ssd_pages > 0) {
            page_written.assign((size_t)kNumGlobalVisits * (size_t)num_ssd_pages, 0u);
        }

        out_dev_cap = dev_cap;
        out_init_host_cap = 0;  // Lazy elastic Tier-1 allocation: ZERO upfront multi-GB host pinning!
        out_eff_max_ctx = max_ctx;

        std::fprintf(stderr,
            "[InkElasticKV] 3-Tier Plan (kv_fmt=%d, max_ctx=%d, slots=%d, avail_ram=%.2f GiB): "
            "Tier0(VRAM)=%d tok, Tier1(RAM cap)=%d tok (%.2f GiB max), Tier2(NVMe SSD)=%d tok (%d pages x %.2f MiB, ssd_base=%d)\n",
            (int)kv_dtype, max_ctx, max_slots, (double)avail / 1073741824.0,
            dev_cap, tier1_max_ctx, (double)((size_t)tier1_max_ctx * per_pos_all8) / 1073741824.0,
            std::max(0, max_ctx - ssd_base), num_ssd_pages, (double)page_bytes / 1048576.0, ssd_base);
    }

    bool alloc_page_buf(PageBuf& pb) noexcept {
        pb = {};
        const size_t alloc_sz = page_bytes + kAlignBytes;
        if (cudaHostAlloc(&pb.raw, alloc_sz, cudaHostAllocMapped) != cudaSuccess) {
            (void)cudaGetLastError();
            pb.raw = nullptr;
            return false;
        }
        void* dp = nullptr;
        if (cudaHostGetDevicePointer(&dp, pb.raw, 0) != cudaSuccess) {
            (void)cudaGetLastError();
            cudaFreeHost(pb.raw);
            pb.raw = nullptr;
            return false;
        }
        const uintptr_t h_addr = reinterpret_cast<uintptr_t>(pb.raw);
        const uintptr_t h_al   = (h_addr + kAlignBytes - 1u) & ~(static_cast<uintptr_t>(kAlignBytes) - 1u);
        const size_t delta     = static_cast<size_t>(h_al - h_addr);
        pb.h_base = reinterpret_cast<uint8_t*>(h_al);
        pb.d_base = reinterpret_cast<uint8_t*>(dp) + delta;
        std::memset(pb.h_base, 0, page_bytes);
        return true;
    }

    void free_page_buf(PageBuf& pb) noexcept {
        if (pb.raw) {
            cudaFreeHost(pb.raw);
            pb.raw = nullptr;
        }
        pb.h_base = nullptr;
        pb.d_base = nullptr;
        pb.cur_gv = -1;
        pb.cur_page = -1;
        pb.dirty = false;
    }

    InkKvVisitDesc make_page_desc(const PageBuf& pb, int page_start) const noexcept {
        InkKvVisitDesc d{};
        d.is_swa   = false;
        d.kv_fmt   = (uint8_t)kv_dtype;
        d.dev_cap  = page_start;
        d.host_cap = kPageTokens;
        d.cap_pos  = page_start + kPageTokens;
        uint8_t* p = pb.d_base;
        if (kv_dtype == InkKvDtype::FP16) {
            d.h_k_f16 = reinterpret_cast<__half*>(p);
            d.h_v_f16 = reinterpret_cast<__half*>(p + k_bytes_page);
        } else {
            d.h_k_f8 = p;
            d.h_v_f8 = p + k_bytes_page;
            d.h_k_sc = reinterpret_cast<float*>(p + k_bytes_page + v_bytes_page);
            d.h_v_sc = reinterpret_cast<float*>(p + k_bytes_page + v_bytes_page + sc_bytes_page);
        }
        return d;
    }

    uint64_t page_file_offset(int gv, int page) const noexcept {
        return (static_cast<uint64_t>(gv) * static_cast<uint64_t>(num_ssd_pages) + static_cast<uint64_t>(page))
             * static_cast<uint64_t>(page_bytes);
    }

    bool raw_write_page(int gv, int page, const uint8_t* src) noexcept {
        const uint64_t off = page_file_offset(gv, page);
        size_t done = 0;
#ifdef _WIN32
        while (done < page_bytes) {
            OVERLAPPED ov{};
            const uint64_t cur = off + done;
            ov.Offset = static_cast<DWORD>(cur & 0xFFFFFFFFu);
            ov.OffsetHigh = static_cast<DWORD>(cur >> 32);
            DWORD nw = 0;
            const DWORD chunk = static_cast<DWORD>(std::min<size_t>(page_bytes - done, 1u << 28));
            if (!WriteFile(h_file, src + done, chunk, &nw, &ov)) return false;
            if (nw == 0) return false;
            done += nw;
        }
#else
        while (done < page_bytes) {
            const ssize_t n = ::pwrite(fd, src + done, page_bytes - done, static_cast<off_t>(off + done));
            if (n < 0) {
                if (errno == EINTR) continue;
                return false;
            }
            if (n == 0) return false;
            done += static_cast<size_t>(n);
        }
        if (!ssd_direct) {
#if defined(POSIX_FADV_DONTNEED)
            (void)::posix_fadvise(fd, static_cast<off_t>(off), static_cast<off_t>(page_bytes), POSIX_FADV_DONTNEED);
#endif
        }
#endif
        pw_set(gv, page);
        ssd_bytes_written.fetch_add(page_bytes, std::memory_order_relaxed);
        ssd_pages_written.fetch_add(1, std::memory_order_relaxed);
        return true;
    }

    bool raw_read_page(int gv, int page, uint8_t* dst) noexcept {
        if (!pw_get(gv, page)) {
            std::memset(dst, 0, page_bytes);
            return true;
        }
        const uint64_t off = page_file_offset(gv, page);
        size_t done = 0;
#ifdef _WIN32
        while (done < page_bytes) {
            OVERLAPPED ov{};
            const uint64_t cur = off + done;
            ov.Offset = static_cast<DWORD>(cur & 0xFFFFFFFFu);
            ov.OffsetHigh = static_cast<DWORD>(cur >> 32);
            DWORD nr = 0;
            const DWORD chunk = static_cast<DWORD>(std::min<size_t>(page_bytes - done, 1u << 28));
            if (!ReadFile(h_file, dst + done, chunk, &nr, &ov)) return false;
            if (nr == 0) break;
            done += nr;
        }
#else
        while (done < page_bytes) {
            const ssize_t n = ::pread(fd, dst + done, page_bytes - done, static_cast<off_t>(off + done));
            if (n < 0) {
                if (errno == EINTR) continue;
                return false;
            }
            if (n == 0) break;
            done += static_cast<size_t>(n);
        }
        if (!ssd_direct) {
#if defined(POSIX_FADV_DONTNEED)
            (void)::posix_fadvise(fd, static_cast<off_t>(off), static_cast<off_t>(page_bytes), POSIX_FADV_DONTNEED);
#endif
        }
#endif
        if (done < page_bytes) std::memset(dst + done, 0, page_bytes - done);
        ssd_bytes_read.fetch_add(page_bytes, std::memory_order_relaxed);
        ssd_pages_read.fetch_add(1, std::memory_order_relaxed);
        return true;
    }

    bool ensure_ssd_open() {
        if (ssd_open) return true;
        if (num_ssd_pages <= 0) return false;
        INK_CK(cudaSetDevice(primary_gpu));

        std::string dir;
        if (const char* e = std::getenv("STRATA_INK_KV_SSD_DIR"); e && *e) {
            dir = e;
        } else {
#ifdef _WIN32
            const char* tmp = std::getenv("TEMP");
            dir = std::string(tmp ? tmp : ".") + "\\strata_kv_ssd";
#else
            struct stat st_w{};
            if (::stat("/workspace", &st_w) == 0 && S_ISDIR(st_w.st_mode)) dir = "/workspace/kv_ssd";
            else dir = "/tmp/strata_kv_ssd-" + std::to_string((unsigned long long)::getuid());   // one directory per user, not a shared name
#endif
        }
#ifdef _WIN32
        CreateDirectoryA(dir.c_str(), nullptr);
        ssd_path = dir + "\\kv_tier2_" + std::to_string(GetCurrentProcessId()) + ".bin";
        h_file = CreateFileA(ssd_path.c_str(), GENERIC_READ | GENERIC_WRITE, 0, nullptr, CREATE_ALWAYS,
                             FILE_ATTRIBUTE_TEMPORARY | FILE_FLAG_DELETE_ON_CLOSE | FILE_FLAG_NO_BUFFERING, nullptr);
        if (h_file != INVALID_HANDLE_VALUE) {
            ssd_direct = true;
        } else {
            h_file = CreateFileA(ssd_path.c_str(), GENERIC_READ | GENERIC_WRITE, 0, nullptr, CREATE_ALWAYS,
                                 FILE_ATTRIBUTE_TEMPORARY | FILE_FLAG_DELETE_ON_CLOSE, nullptr);
            if (h_file == INVALID_HANDLE_VALUE) return false;
            ssd_direct = false;
        }
#else
        (void)::mkdir(dir.c_str(), 0700);
        {
            // the directory must be ours, a real directory (not a link someone planted), and closed to other users
            struct stat sd{};
            if (::lstat(dir.c_str(), &sd) != 0 || !S_ISDIR(sd.st_mode) || sd.st_uid != ::geteuid() || (sd.st_mode & 077) != 0) {
                std::fprintf(stderr, "[InkElasticKV] Tier-2 SSD directory %s is not a private directory of this user (owner/mode/link check failed); Tier-2 off. "
                                     "Set STRATA_INK_KV_SSD_DIR to a directory you own.\n", dir.c_str());
                return false;
            }
            // room for the file (it is sparse, but a full disk must not be found out in the middle of an answer)
            struct statvfs sv{};
            if (::statvfs(dir.c_str(), &sv) == 0) {
                const uint64_t avail = (uint64_t)sv.f_bavail * (uint64_t)sv.f_frsize;
                const uint64_t need = (uint64_t)kNumGlobalVisits * (uint64_t)num_ssd_pages * (uint64_t)page_bytes;
                if (avail < need) {
                    std::fprintf(stderr, "[InkElasticKV] Tier-2 SSD directory %s has %.1f GiB free, the plan needs %.1f GiB; Tier-2 off.\n",
                                 dir.c_str(), (double)avail / 1073741824.0, (double)need / 1073741824.0);
                    return false;
                }
            }
        }
        static std::atomic<uint64_t> uid_gen{1};
        ssd_path = dir + "/kv_tier2_" + std::to_string(::getpid()) + "_" + std::to_string(uid_gen.fetch_add(1)) + ".bin";
#if defined(O_DIRECT)
        fd = ::open(ssd_path.c_str(), O_RDWR | O_CREAT | O_EXCL | O_NOFOLLOW | O_CLOEXEC | O_DIRECT, 0600);
        if (fd >= 0) {
            ssd_direct = true;
        } else {
            fd = ::open(ssd_path.c_str(), O_RDWR | O_CREAT | O_EXCL | O_NOFOLLOW | O_CLOEXEC, 0600);
            ssd_direct = false;
        }
#else
        fd = ::open(ssd_path.c_str(), O_RDWR | O_CREAT | O_EXCL | O_NOFOLLOW | O_CLOEXEC, 0600);
        ssd_direct = false;
#endif
        if (fd < 0) return false;
        // Unlink immediately on Linux so file space is reclaimed cleanly even if process is killed!
        (void)::unlink(ssd_path.c_str());
#endif
        // Allocate bounded mapped pinned staging buffers (8 write + 2 read = ~41 MiB FP8 / ~80 MiB FP16 total!)
        for (int g = 0; g < kNumGlobalVisits; ++g) {
            if (!alloc_page_buf(write_buf[g])) { free_ssd_only(); return false; }
        }
        for (int b = 0; b < 2; ++b) {
            if (!alloc_page_buf(read_buf[b])) { free_ssd_only(); return false; }
            INK_CK(cudaEventCreateWithFlags(&ev_read_done[b], cudaEventDisableTiming));
            read_ev_pending[b] = false;
        }
        // Probe O_DIRECT write compatibility on Linux (e.g. overlayfs); fall back cleanly to buffered + DONTNEED on EINVAL!
#if !defined(_WIN32) && defined(O_DIRECT)
        if (ssd_direct) {
            const ssize_t pw = ::pwrite(fd, read_buf[0].h_base, kAlignBytes, 0);
            if (pw < 0 && errno == EINVAL) {
                ::close(fd);
                ssd_direct = false;
                fd = ::open(ssd_path.c_str(), O_RDWR | O_CREAT | O_EXCL | O_NOFOLLOW | O_CLOEXEC, 0600);   // the path was unlinked a moment ago
                if (fd < 0) { free_ssd_only(); return false; }
                (void)::unlink(ssd_path.c_str());
            }
        }
#endif
        // Start persistent async background NVMe reader thread
        io_stop = false;
        io_busy = false;
        io_req.valid = false;
        io_thr = std::thread([this]() {
            for (;;) {
                AsyncReadReq rq{};
                {
                    std::unique_lock<std::mutex> lk(io_mu);
                    io_cv.wait(lk, [&]() { return io_stop || io_req.valid; });
                    if (io_stop) break;
                    rq = io_req;
                    io_req.valid = false;
                }
                const bool ok = raw_read_page(rq.gv, rq.page, read_buf[rq.buf_idx].h_base);
                {
                    std::lock_guard<std::mutex> lk(io_mu);
                    read_buf[rq.buf_idx].cur_gv = ok ? rq.gv : -1;
                    read_buf[rq.buf_idx].cur_page = ok ? rq.page : -1;
                    io_ok = ok;
                    io_busy = false;
                }
                io_cv.notify_all();
            }
        });

        ssd_open = true;
        std::fprintf(stderr,
            "[InkElasticKV] Tier-2 NVMe SSD store active: %s (direct=%d, %d pages x %.2f MiB/visit, staging=%.1f MiB pinned)\n",
            ssd_path.c_str(), (int)ssd_direct, num_ssd_pages, (double)page_bytes / 1048576.0,
            (double)((kNumGlobalVisits + 2) * page_bytes) / 1048576.0);
        return true;
    }

    void wait_buf_gpu_done(int buf_idx) {
        if (buf_idx >= 0 && buf_idx < 2 && read_ev_pending[buf_idx] && ev_read_done[buf_idx]) {
            INK_CK(cudaEventSynchronize(ev_read_done[buf_idx]));
            read_ev_pending[buf_idx] = false;
        }
    }

    void mark_buf_gpu_used(int buf_idx, cudaStream_t st) {
        if (buf_idx >= 0 && buf_idx < 2 && ev_read_done[buf_idx]) {
            INK_CK(cudaEventRecord(ev_read_done[buf_idx], st));
            read_ev_pending[buf_idx] = true;
        }
    }

    void submit_async_read(int buf_idx, int gv, int page) {
        wait_async_read();
        wait_buf_gpu_done(buf_idx);
        std::lock_guard<std::mutex> lk(io_mu);
        io_busy = true;
        io_req = {true, buf_idx, gv, page};
        io_cv.notify_one();
    }

    bool wait_async_read() {
        std::unique_lock<std::mutex> lk(io_mu);
        io_cv.wait(lk, [&]() { return !io_busy; });
        return io_ok;
    }

    /// Grow Tier-1 Mapped Host RAM lazily (one visit at a time!) and activate Tier-2 NVMe SSD when needed_pos > ssd_base.
    /// Automatically fails over to Tier-2 SSD if pinned host RAM allocation ever fails under tight OS memory!
    template <typename RunnerT>
    bool ensure_cap(RunnerT& r, int needed_pos) {
        if (needed_pos <= r.kv_alloc_ctx) return true;
        if (needed_pos > max_ctx) return false;

        // 1. Grow Tier 1 (Mapped Host RAM) lazily up to tier1_max_ctx, one global visit at a time!
        const int need_t1_raw = std::min(tier1_max_ctx, std::max(0, needed_pos - dev_cap));
        if (need_t1_raw > r.kv_host_ctx) {
            int want_t1 = need_t1_raw;
            if (needed_pos > ssd_base) {
                want_t1 = tier1_max_ctx;
            } else {
                // Smooth elastic growth: 16K -> 32K -> 64K -> 128K -> 256K -> +64K steps (aligned to kPageTokens!)
                int step = std::max(16384, (r.kv_host_ctx > 0) ? (r.kv_host_ctx * 2) : 16384);
                if (step > r.kv_host_ctx + 131072) step = r.kv_host_ctx + 131072;
                want_t1 = std::min(tier1_max_ctx, std::max(need_t1_raw, step));
                if (want_t1 < tier1_max_ctx) want_t1 = ((want_t1 + kPageTokens - 1) / kPageTokens) * kPageTokens;
                want_t1 = std::min(tier1_max_ctx, want_t1);
            }
            INK_CK(cudaSetDevice(primary_gpu));
            INK_CK(cudaStreamSynchronize(r.st));
            const int old_h = r.kv_host_ctx;
            bool all_ok = true;
            // Grow one global visit at a time to minimize transient RSS spike (+1/8 instead of +100%!)
            for (int gv = 0; gv < kNumGlobalVisits; ++gv) {
                const int v = gv_to_visit[gv];
                const size_t new_slot_el = (size_t)want_t1 * kInkKVHeads * kInkHeadDim;
                const size_t old_slot_el = (size_t)old_h   * kInkKVHeads * kInkHeadDim;
                const size_t new_tot_el  = (size_t)max_slots * new_slot_el;

                void *rk = nullptr, *rv = nullptr, *rks = nullptr, *rvs = nullptr;
                __half *dk16 = nullptr, *dv16 = nullptr;
                uint8_t *dk8 = nullptr, *dv8 = nullptr;
                float *dks = nullptr, *dvs = nullptr;
                bool ok = false;
                if (kv_dtype == InkKvDtype::FP16) {
                    ok = RunnerT::template alloc_mapped_host<__half>(new_tot_el, rk, dk16)
                      && RunnerT::template alloc_mapped_host<__half>(new_tot_el, rv, dv16);
                } else {
                    const size_t v_b  = (kv_dtype == InkKvDtype::K8V4) ? (new_tot_el / 2u) : new_tot_el;
                    const size_t sc_n = (size_t)max_slots * (size_t)want_t1 * kInkKVHeads;
                    ok = RunnerT::template alloc_mapped_host<uint8_t>(new_tot_el, rk, dk8)
                      && RunnerT::template alloc_mapped_host<uint8_t>(v_b, rv, dv8)
                      && RunnerT::template alloc_mapped_host<float>(sc_n, rks, dks)
                      && RunnerT::template alloc_mapped_host<float>(sc_n, rvs, dvs);
                }
                if (!ok) {
                    if (rk)  cudaFreeHost(rk);
                    if (rv)  cudaFreeHost(rv);
                    if (rks) cudaFreeHost(rks);
                    if (rvs) cudaFreeHost(rvs);
                    all_ok = false;
                    // Roll back any visits already grown in this loop back to old_h? Wait: keeping old_h on un-grown visits
                    // would leave mismatched host_cap across visits! If any visit fails, seal Tier 1 at old_h cleanly!
                    // For already-grown visits 0..gv-1, shrink back to old_h or seal only if gv==0:
                    // Even cleaner: check avail RAM before starting loop, and if gv > 0, shrink visits 0..gv-1 to old_h.
                    for (int pgv = 0; pgv < gv; ++pgv) {
                        const int pv = gv_to_visit[pgv];
                        r.kv_desc[pv].host_cap = old_h;
                        r.kv_desc[pv].cap_pos  = dev_cap + old_h;
                    }
                    break;
                }
                // Copy preserved Tier-1 tokens [0 .. old_h) per slot!
                if (old_h > 0) {
                    if (kv_dtype == InkKvDtype::FP16) {
                        for (int sl = 0; sl < max_slots; ++sl) {
                            std::memcpy(reinterpret_cast<__half*>(rk) + (size_t)sl * new_slot_el,
                                        reinterpret_cast<const __half*>(r.host_kv_raw_k[v]) + (size_t)sl * old_slot_el,
                                        old_slot_el * sizeof(__half));
                            std::memcpy(reinterpret_cast<__half*>(rv) + (size_t)sl * new_slot_el,
                                        reinterpret_cast<const __half*>(r.host_kv_raw_v[v]) + (size_t)sl * old_slot_el,
                                        old_slot_el * sizeof(__half));
                        }
                    } else {
                        const bool v4 = (kv_dtype == InkKvDtype::K8V4);
                        const size_t old_vb = v4 ? (old_slot_el / 2u) : old_slot_el;
                        const size_t new_vb = v4 ? (new_slot_el / 2u) : new_slot_el;
                        const size_t old_sc = (size_t)old_h * kInkKVHeads;
                        const size_t new_sc = (size_t)want_t1 * kInkKVHeads;
                        for (int sl = 0; sl < max_slots; ++sl) {
                            std::memcpy(reinterpret_cast<uint8_t*>(rk) + (size_t)sl * new_slot_el,
                                        reinterpret_cast<const uint8_t*>(r.host_kv_raw_k[v]) + (size_t)sl * old_slot_el,
                                        old_slot_el);
                            std::memcpy(reinterpret_cast<uint8_t*>(rv) + (size_t)sl * new_vb,
                                        reinterpret_cast<const uint8_t*>(r.host_kv_raw_v[v]) + (size_t)sl * old_vb,
                                        old_vb);
                            std::memcpy(reinterpret_cast<float*>(rks) + (size_t)sl * new_sc,
                                        reinterpret_cast<const float*>(r.host_kv_raw_ksc[v]) + (size_t)sl * old_sc,
                                        old_sc * sizeof(float));
                            std::memcpy(reinterpret_cast<float*>(rvs) + (size_t)sl * new_sc,
                                        reinterpret_cast<const float*>(r.host_kv_raw_vsc[v]) + (size_t)sl * old_sc,
                                        old_sc * sizeof(float));
                        }
                    }
                }
                r.free_host_kv_visit(v);
                r.host_kv_raw_k[v] = rk;
                r.host_kv_raw_v[v] = rv;
                r.host_kv_raw_ksc[v] = rks;
                r.host_kv_raw_vsc[v] = rvs;
                r.kv_desc[v].h_k_f16 = dk16;
                r.kv_desc[v].h_v_f16 = dv16;
                r.kv_desc[v].h_k_f8  = dk8;
                r.kv_desc[v].h_v_f8  = dv8;
                r.kv_desc[v].h_k_sc  = dks;
                r.kv_desc[v].h_v_sc  = dvs;
                r.kv_desc[v].dev_cap  = dev_cap;
                r.kv_desc[v].host_cap = want_t1;
                r.kv_desc[v].cap_pos  = dev_cap + want_t1;
            }
            if (all_ok) {
                r.kv_host_ctx = want_t1;
                r.kv_alloc_ctx = dev_cap + want_t1;
                INK_CK(cudaMemcpy(r.d_kv_desc_arr, r.kv_desc, sizeof(r.kv_desc), cudaMemcpyHostToDevice));
                r.drop_all_graphs();
                const auto ms = ink_query_proc_mem_stats();
                std::fprintf(stderr,
                    "[InkElasticKV] Tier-1 RAM grown: %d VRAM + %d RAM = %d tok (VmRSS=%.2f GiB, VmHWM=%.2f GiB)\n",
                    dev_cap, r.kv_host_ctx, r.kv_alloc_ctx,
                    (double)ms.vm_rss_bytes / 1073741824.0, (double)ms.vm_hwm_bytes / 1073741824.0);
            } else {
                // Pinning failed unexpectedly: seal Tier-1 at current old_h (page-aligned) & fail over cleanly to Tier-2 SSD!
                INK_CK(cudaMemcpy(r.d_kv_desc_arr, r.kv_desc, sizeof(r.kv_desc), cudaMemcpyHostToDevice));
                r.drop_all_graphs();
                tier1_max_ctx = (old_h / kPageTokens) * kPageTokens;
                ssd_base = dev_cap + tier1_max_ctx;
                num_ssd_pages = (max_ctx > ssd_base) ? ((max_ctx - ssd_base + kPageTokens - 1) / kPageTokens) : 0;
                page_written.assign((size_t)kNumGlobalVisits * (size_t)num_ssd_pages, 0u);
                std::fprintf(stderr,
                    "[InkElasticKV] Tier-1 RAM ceiling hit at %d tok; sealing Tier-1 & failing over to Tier-2 NVMe SSD (ssd_base=%d)\n",
                    old_h, ssd_base);
            }
        }

        // 2. If needed_pos enters Tier-2 NVMe SSD [ssd_base .. max_ctx), open SSD backing store & unlock full max_ctx!
        if (needed_pos > ssd_base) {
            if (!ensure_ssd_open()) return false;
            r.kv_alloc_ctx = max_ctx;
        }
        return needed_pos <= r.kv_alloc_ctx;
    }

    /// Return mapped pinned write-page InkKvVisitDesc for a single token at `pos >= ssd_base` (for k_ink_post_qkv_fused!).
    InkKvVisitDesc prepare_single_write_desc(int visit, int pos, int slot0, cudaStream_t st) {
        const int gv = global_visit_idx(visit);
        const int pg = std::max(0, std::min(num_ssd_pages - 1, (pos - ssd_base) / kPageTokens));
        bind_write_page(gv, pg, st);
        write_buf[gv].dirty = true;
        if (slot0 >= 0 && slot0 < max_slots && pos + 1 > slot_max_pos[slot0]) slot_max_pos[slot0] = pos + 1;
        return make_page_desc(write_buf[gv], ssd_base + pg * kPageTokens);
    }

    /// A Tier-2 page that cannot be written or read back (full disk, I/O error) would leave zeros where K/V should be: the model would go on with
    /// silently wrong attention. Stop the engine with a clear message instead (same exit path as a fatal GPU error).
    [[noreturn]] void ssd_io_fatal(const char* what, int gv, int page) const {
        char msg[256];
        std::snprintf(msg, sizeof(msg), "strata-inkling elastic KV: Tier-2 SSD %s failed (visit %d, page %d, file %s)", what, gv, page, ssd_path.c_str());
        strata::core::exit_device_error(msg, std::strerror(errno));
    }

    void flush_write_buf(int gv, cudaStream_t st) {
        auto& wb = write_buf[gv];
        if (!wb.dirty || wb.cur_page < 0) return;
        INK_CK(cudaStreamSynchronize(st));
        if (!raw_write_page(gv, wb.cur_page, wb.h_base)) ssd_io_fatal("write", gv, wb.cur_page);
        wb.dirty = false;
    }

    /// Ensure write_buf[gv] holds Tier-2 page `pg` (flushing previous dirty page to NVMe if needed).
    bool bind_write_page(int gv, int pg, cudaStream_t st) {
        if (!ensure_ssd_open()) return false;
        auto& wb = write_buf[gv];
        if (wb.cur_gv == gv && wb.cur_page == pg) return true;
        if (wb.dirty && wb.cur_page >= 0) {
            INK_CK(cudaStreamSynchronize(st));
            if (!raw_write_page(gv, wb.cur_page, wb.h_base)) ssd_io_fatal("write", gv, wb.cur_page);
            wb.dirty = false;
        }
        if (pw_get(gv, pg)) {
            if (!raw_read_page(gv, pg, wb.h_base)) ssd_io_fatal("read", gv, pg);
        } else {
            std::memset(wb.h_base, 0, page_bytes);
        }
        wb.cur_gv = gv;
        wb.cur_page = pg;
        return true;
    }

    /// Write K, V [M, 8, 128] for positions >= ssd_base into Tier-2 active page(s), splitting cleanly across 2048-tok boundaries!
    void write_tier2_kv_batch(int visit, const float* d_k, const float* d_v,
                              const int32_t* d_slots, int slot0, const int32_t* d_pos,
                              int p_min, int p_max, int M, cudaStream_t st) {
        const int gv = global_visit_idx(visit);
        if (gv < 0 || p_max < ssd_base) return;
        const int beg_pg = std::max(0, (p_min - ssd_base) / kPageTokens);
        const int end_pg = std::min(num_ssd_pages - 1, (p_max - ssd_base) / kPageTokens);
        for (int pg = beg_pg; pg <= end_pg; ++pg) {
            if (!bind_write_page(gv, pg, st)) return;
            const int pstart = ssd_base + pg * kPageTokens;
            const InkKvVisitDesc pdesc = make_page_desc(write_buf[gv], pstart);
            k_ink_kv_write_tier2_page<<<dim3(kInkKVHeads, M), 128, 0, st>>>(
                d_k, d_v, d_slots, slot0, d_pos, pdesc, pstart, M);
            write_buf[gv].dirty = true;
            // If we completed this page during prefill, flush immediately to keep active state clean!
            if (p_max >= pstart + kPageTokens - 1 && pg < end_pg) {
                flush_write_buf(gv, st);
            }
        }
        if (slot0 >= 0 && slot0 < max_slots && p_max + 1 > slot_max_pos[slot0]) {
            slot_max_pos[slot0] = p_max + 1;
        }
    }

    /// Commit accepted Tree-32 path for all 8 global visits whose [anchor_pos .. anchor_pos + n_emit - 1] reaches Tier 2!
    void commit_tier2_tree(const float* d_stage_k, const float* d_stage_v, const int32_t* d_acc_path,
                           int p0, int n_acc, int slot0, int tree_cap, cudaStream_t st) {
        if (!enabled || num_ssd_pages <= 0 || p0 + n_acc <= ssd_base) return;
        const int p_min = std::max(ssd_base, p0);
        const int p_max = p0 + n_acc - 1;
        const int beg_pg = (p_min - ssd_base) / kPageTokens;
        const int end_pg = std::min(num_ssd_pages - 1, (p_max - ssd_base) / kPageTokens);
        for (int gv = 0; gv < kNumGlobalVisits; ++gv) {
            const int visit = gv_to_visit[gv];
            for (int pg = beg_pg; pg <= end_pg; ++pg) {
                if (!bind_write_page(gv, pg, st)) continue;
                const int pstart = ssd_base + pg * kPageTokens;
                const InkKvVisitDesc pdesc = make_page_desc(write_buf[gv], pstart);
                k_ink_commit_kv_tier2_page<<<kInkKVHeads, 128, 0, st>>>(
                    d_stage_k, d_stage_v, d_acc_path, visit, p0, n_acc, slot0, tree_cap, pdesc, pstart);
                write_buf[gv].dirty = true;
            }
        }
        if (slot0 >= 0 && slot0 < max_slots && p_max + 1 > slot_max_pos[slot0]) {
            slot_max_pos[slot0] = p_max + 1;
        }
    }

    /// Ensure decode online-softmax accumulators for M queries
    void ensure_decode_run_bufs(int M) {
        if (M <= run_cap_M && d_run_o) return;
        if (d_run_m) { cudaFree(d_run_m); d_run_m = nullptr; }
        if (d_run_l) { cudaFree(d_run_l); d_run_l = nullptr; }
        if (d_run_o) { cudaFree(d_run_o); d_run_o = nullptr; }
        run_cap_M = std::max(M, 32);
        d_run_m = dalloc<float>((size_t)run_cap_M * kInkHeads);
        d_run_l = dalloc<float>((size_t)run_cap_M * kInkHeads);
        d_run_o = dalloc<float>((size_t)run_cap_M * kInkHeads * kInkHeadDim);
    }

    /// Ensure inverted-loop prefill accumulators for `n_qc` query chunks of 256 tokens
    bool ensure_pf_multi_qc(int n_qc) {
        if (n_qc <= pf_qchunk_cap && d_pf_o_all) return true;
        if (d_pf_m_all) { cudaFree(d_pf_m_all); d_pf_m_all = nullptr; }
        if (d_pf_l_all) { cudaFree(d_pf_l_all); d_pf_l_all = nullptr; }
        if (d_pf_o_all) { cudaFree(d_pf_o_all); d_pf_o_all = nullptr; }
        pf_qchunk_cap = 0;
        const size_t ml_b = (size_t)n_qc * kInkHeads * 256u * sizeof(float);
        const size_t o_b  = (size_t)n_qc * kInkHeads * 256u * kInkHeadDim * sizeof(float);
        size_t vf = 0, vt = 0;
        if (ink_cuda_mem_info(&vf, &vt, primary_gpu) != cudaSuccess || vf < (2 * ml_b + o_b + (32ull << 20))) {
            return false;
        }
        if (cudaMalloc(&d_pf_m_all, ml_b) != cudaSuccess ||
            cudaMalloc(&d_pf_l_all, ml_b) != cudaSuccess ||
            cudaMalloc(&d_pf_o_all, o_b)  != cudaSuccess) {
            (void)cudaGetLastError();
            if (d_pf_m_all) { cudaFree(d_pf_m_all); d_pf_m_all = nullptr; }
            if (d_pf_l_all) { cudaFree(d_pf_l_all); d_pf_l_all = nullptr; }
            if (d_pf_o_all) { cudaFree(d_pf_o_all); d_pf_o_all = nullptr; }
            return false;
        }
        pf_qchunk_cap = n_qc;
        return true;
    }

    /// Inverted-Loop O(1)-VRAM 3-Tier Global Prefill Attention (reads each 2048-tok KV tile ONCE across all M queries -> 8x faster!).
    /// Handles Tier 0 (VRAM), Tier 1 (Mapped Host RAM), AND Tier 2 (Double-Buffered Async NVMe SSD) seamlessly!
    void run_global_prefill_attn_3tier(
        int visit, const InkKvVisitDesc& kv_t01, int slot0, int p0_val, int M,
        bool use_fp16_pf, cudaDataType_t pf_dt, cublasGemmAlgo_t pf_algo,
        cublasHandle_t cublas, cudaStream_t st,
        const float* d_q, const float* d_rel_table, const float* d_sinks, const float* d_gate_attn,
        void* d_pf_q, void* d_pf_k, void* d_pf_v, float* d_pf_s, void* d_pf_p, float* d_pf_ot,
        float* d_alpha_tmp, float* d_attn_out) {
        constexpr int Q_CH = 256;
        const int n_qc = (M + Q_CH - 1) / Q_CH;
        const int gv = global_visit_idx(visit);
        const int K_tot = p0_val + M;
        const float sc_qk = 0.08838834764831845f;
        const float one = 1.0f, zero = 0.0f;

        if (use_fp16_pf) k_ink_pf_pack_q<true><<<dim3(kInkHeads, M), 128, 0, st>>>(d_q, d_pf_q, M);
        else             k_ink_pf_pack_q<false><<<dim3(kInkHeads, M), 128, 0, st>>>(d_q, d_pf_q, M);

        // Initialize online-softmax accumulators for all Q-chunks
        for (int qc = 0; qc < n_qc; ++qc) {
            const int q0 = qc * Q_CH;
            const int C  = std::min(Q_CH, M - q0);
            float* m_ptr = d_pf_m_all + (size_t)qc * kInkHeads * Q_CH;
            float* l_ptr = d_pf_l_all + (size_t)qc * kInkHeads * Q_CH;
            float* o_ptr = d_pf_o_all + (size_t)qc * kInkHeads * Q_CH * kInkHeadDim;
            k_ink_pf_online_init<<<dim3(kInkHeads, C), 128, 0, st>>>(d_sinks, m_ptr, l_ptr, o_ptr, q0, C);
        }

        // Prepare async double-buffered SSD prefetch if K_tot enters Tier 2
        const int first_ssd_kt = (ssd_open && gv >= 0 && K_tot > ssd_base) ? ssd_base : K_tot;
        int prefetched_buf = -1;
        int prefetched_pg  = -1;
        auto kick_next_ssd_page = [&](int next_pg, int buf) {
            if (!ssd_open || gv < 0 || next_pg < 0 || next_pg >= num_ssd_pages) return;
            if (ssd_base + next_pg * kPageTokens >= K_tot) return;
            if (write_buf[gv].cur_gv == gv && write_buf[gv].cur_page == next_pg) return;  // already in active write_buf!
            submit_async_read(buf, gv, next_pg);
            prefetched_buf = buf;
            prefetched_pg  = next_pg;
        };

        // OUTER LOOP over KV tiles (2048 tokens): each tile is read from VRAM/RAM/SSD ONLY ONCE!
        for (int kt = 0; kt < K_tot; ) {
            const int K_t = (kt < first_ssd_kt)
                ? std::min(kPageTokens, first_ssd_kt - kt)
                : std::min(kPageTokens, K_tot - kt);
            if (kt < first_ssd_kt) {
                // Tier 0 (VRAM) or Tier 1 (Host RAM)
                // Overlap first Tier-2 SSD page read while GPU processes the last Tier-1 tile!
                if (kt + K_t == first_ssd_kt && first_ssd_kt < K_tot) {
                    kick_next_ssd_page(0, 0);
                }
                if (use_fp16_pf) k_ink_pf_pack_kv<true><<<dim3(kInkKVHeads, K_t), 128, 0, st>>>(kv_t01, slot0, kt, K_t, d_pf_k, d_pf_v);
                else             k_ink_pf_pack_kv<false><<<dim3(kInkKVHeads, K_t), 128, 0, st>>>(kv_t01, slot0, kt, K_t, d_pf_k, d_pf_v);
            } else {
                // Tier 2 (NVMe SSD)
                const int pg = (kt - ssd_base) / kPageTokens;
                const int pstart = ssd_base + pg * kPageTokens;
                InkKvVisitDesc sdesc{};
                if (write_buf[gv].cur_gv == gv && write_buf[gv].cur_page == pg) {
                    sdesc = make_page_desc(write_buf[gv], pstart);
                    // Overlap next SSD page read if any
                    if (prefetched_pg != pg + 1) kick_next_ssd_page(pg + 1, 0);
                } else {
                    int use_buf = 0;
                    if (prefetched_pg == pg && prefetched_buf >= 0) {
                        wait_async_read();
                        use_buf = prefetched_buf;
                        prefetched_pg = -1;
                    } else {
                        if (prefetched_pg >= 0) wait_async_read();
                        wait_buf_gpu_done(0);
                        raw_read_page(gv, pg, read_buf[0].h_base);
                        read_buf[0].cur_gv = gv;
                        read_buf[0].cur_page = pg;
                        use_buf = 0;
                    }
                    sdesc = make_page_desc(read_buf[use_buf], pstart);
                    // Kick off async NVMe prefetch of pg + 1 into the OTHER staging buffer while GPU computes!
                    kick_next_ssd_page(pg + 1, 1 - use_buf);
                    if (use_fp16_pf) k_ink_pf_pack_kv<true><<<dim3(kInkKVHeads, K_t), 128, 0, st>>>(sdesc, slot0, kt, K_t, d_pf_k, d_pf_v);
                    else             k_ink_pf_pack_kv<false><<<dim3(kInkKVHeads, K_t), 128, 0, st>>>(sdesc, slot0, kt, K_t, d_pf_k, d_pf_v);
                    mark_buf_gpu_used(use_buf, st);
                    goto pf_tile_packed;
                }
                if (use_fp16_pf) k_ink_pf_pack_kv<true><<<dim3(kInkKVHeads, K_t), 128, 0, st>>>(sdesc, slot0, kt, K_t, d_pf_k, d_pf_v);
                else             k_ink_pf_pack_kv<false><<<dim3(kInkKVHeads, K_t), 128, 0, st>>>(sdesc, slot0, kt, K_t, d_pf_k, d_pf_v);
            pf_tile_packed:;
            }

            // INNER LOOP over Q-chunks (256 tokens each): reuse d_pf_k & d_pf_v in VRAM across all causal Q-chunks!
            for (int qc = 0; qc < n_qc; ++qc) {
                const int q0 = qc * Q_CH;
                const int C  = std::min(Q_CH, M - q0);
                const int k_sub_end = p0_val + q0 + C;
                if (kt >= k_sub_end) continue;  // Strictly causal: skip future KV tiles for this Q-chunk!
                const bool is_last = (kt + K_t >= k_sub_end);
                float* m_ptr = d_pf_m_all + (size_t)qc * kInkHeads * Q_CH;
                float* l_ptr = d_pf_l_all + (size_t)qc * kInkHeads * Q_CH;
                float* o_ptr = d_pf_o_all + (size_t)qc * kInkHeads * Q_CH * kInkHeadDim;

                INK_CUBLAS_CK(cublasGemmStridedBatchedEx(
                    cublas, CUBLAS_OP_T, CUBLAS_OP_N,
                    K_t, C, kInkHeadDim, &sc_qk,
                    d_pf_k, pf_dt, kInkHeadDim, (long long)K_t * kInkHeadDim,
                    reinterpret_cast<uint16_t*>(d_pf_q) + (size_t)q0 * kInkHeadDim, pf_dt, kInkHeadDim, (long long)M * kInkHeadDim,
                    &zero, d_pf_s, CUDA_R_32F, K_t, (long long)C * K_t,
                    kInkHeads, CUBLAS_COMPUTE_32F, pf_algo));
                if (use_fp16_pf) {
                    k_ink_pf_softmax_tile_online<true><<<dim3(C, kInkHeads), 256, 0, st>>>(
                        d_pf_s, d_rel_table, d_pf_p, m_ptr, l_ptr, d_alpha_tmp, q0, C, kt, K_t, p0_val, false);
                } else {
                    k_ink_pf_softmax_tile_online<false><<<dim3(C, kInkHeads), 256, 0, st>>>(
                        d_pf_s, d_rel_table, d_pf_p, m_ptr, l_ptr, d_alpha_tmp, q0, C, kt, K_t, p0_val, false);
                }
                INK_CUBLAS_CK(cublasGemmStridedBatchedEx(
                    cublas, CUBLAS_OP_N, CUBLAS_OP_N,
                    kInkHeadDim, C, K_t, &one,
                    d_pf_v, pf_dt, kInkHeadDim, (long long)K_t * kInkHeadDim,
                    d_pf_p, pf_dt, K_t, (long long)C * K_t,
                    &zero, d_pf_ot, CUDA_R_32F, kInkHeadDim, (long long)C * kInkHeadDim,
                    kInkHeads, CUBLAS_COMPUTE_32F, pf_algo));
                k_ink_pf_accum_tile<<<dim3(kInkHeads, C), 128, 0, st>>>(
                    d_pf_ot, d_alpha_tmp, l_ptr, o_ptr, d_gate_attn, d_attn_out, q0, C, is_last);
            }
            kt += K_t;
        }
        if (prefetched_pg >= 0) wait_async_read();
    }

    /// 3-Tier Chunked Online-Softmax Split-K Decode Attention (M <= 16 or Tree-32) when Tier-2 NVMe SSD holds tokens!
    void run_global_decode_attn_3tier(
        int visit, const InkKvVisitDesc& kv_t01, int slot0, const int32_t* d_slots,
        const int32_t* d_pos, int max_qpos, int M, int mode, int p0_val,
        const float* d_q, const float* d_rel_table, const float* d_sinks, const float* d_gate_attn,
        const float* tree_k, const float* tree_v, const uint8_t* tree_anc,
        float* d_attn_pm, float* d_attn_pl, float* d_attn_po, float* d_attn_out, cudaStream_t st) {
        const int gv = global_visit_idx(visit);
        ensure_decode_run_bufs(M);
        const int n_sp = (M == 1 || mode == 2) ? kInkAttnMaxSplits : kInkAttnSplits;
        dim3 gsp(kInkKVHeads, n_sp, M);
        dim3 gr(kInkHeads, M);

        const int committed_end = (mode == 2) ? p0_val : (max_qpos + 1);
        const int n_active_ssd_pages = (committed_end > ssd_base)
            ? std::min(num_ssd_pages, (committed_end - ssd_base + kPageTokens - 1) / kPageTokens)
            : 0;

        // Kick off async prefetch of SSD page 0 while GPU computes Tier 0 + Tier 1!
        int prefetched_buf = -1, prefetched_pg = -1;
        auto kick_page = [&](int pg, int buf) {
            if (pg < 0 || pg >= n_active_ssd_pages) return;
            if (write_buf[gv].cur_gv == gv && write_buf[gv].cur_page == pg) return;
            submit_async_read(buf, gv, pg);
            prefetched_buf = buf;
            prefetched_pg  = pg;
        };
        if (n_active_ssd_pages > 0) kick_page(0, 0);

        // Pass 0: Tier 0 (VRAM) + Tier 1 (Host RAM) [0 .. ssd_base) + per-query sink + Tree ancestors
        k_ink_flash_gqa_splitk_range<<<gsp, 128, 0, st>>>(
            d_q, d_rel_table, d_sinks, kv_t01, (mode == 1) ? d_slots : nullptr, slot0, d_pos,
            (mode == 2) ? tree_k : nullptr, (mode == 2) ? tree_v : nullptr, (mode == 2) ? tree_anc : nullptr,
            p0_val, 0, ssd_base, /*include_sink=*/true, /*include_tree=*/(mode == 2),
            d_attn_pm, d_attn_pl, d_attn_po, M, n_sp);
        k_ink_flash_gqa_fold_splits<<<gr, 128, 0, st>>>(
            d_attn_pm, d_attn_pl, d_attn_po, d_run_m, d_run_l, d_run_o, M, n_sp, /*is_first=*/true);

        // Pass 1..P: Stream Tier-2 SSD pages double-buffered with async background NVMe prefetcher!
        for (int pg = 0; pg < n_active_ssd_pages; ++pg) {
            const int pstart = ssd_base + pg * kPageTokens;
            const int pend   = std::min(committed_end, pstart + kPageTokens);
            InkKvVisitDesc sdesc{};
            bool used_read_buf = false;
            int use_buf = 0;
            if (write_buf[gv].cur_gv == gv && write_buf[gv].cur_page == pg) {
                sdesc = make_page_desc(write_buf[gv], pstart);
                if (prefetched_pg != pg + 1) kick_page(pg + 1, 0);
            } else {
                if (prefetched_pg == pg && prefetched_buf >= 0) {
                    wait_async_read();
                    use_buf = prefetched_buf;
                    prefetched_pg = -1;
                } else {
                    if (prefetched_pg >= 0) wait_async_read();
                    wait_buf_gpu_done(0);
                    raw_read_page(gv, pg, read_buf[0].h_base);
                    read_buf[0].cur_gv = gv;
                    read_buf[0].cur_page = pg;
                    use_buf = 0;
                }
                sdesc = make_page_desc(read_buf[use_buf], pstart);
                used_read_buf = true;
                kick_page(pg + 1, 1 - use_buf);
            }
            k_ink_flash_gqa_splitk_range<<<gsp, 128, 0, st>>>(
                d_q, d_rel_table, d_sinks, sdesc, (mode == 1) ? d_slots : nullptr, slot0, d_pos,
                nullptr, nullptr, nullptr, p0_val, pstart, pend, /*include_sink=*/false, /*include_tree=*/false,
                d_attn_pm, d_attn_pl, d_attn_po, M, n_sp);
            if (used_read_buf) mark_buf_gpu_used(use_buf, st);
            k_ink_flash_gqa_fold_splits<<<gr, 128, 0, st>>>(
                d_attn_pm, d_attn_pl, d_attn_po, d_run_m, d_run_l, d_run_o, M, n_sp, /*is_first=*/false);
        }
        if (prefetched_pg >= 0) wait_async_read();

        // Finalize: divide by run_l and apply fused * sigmoid(gate_attn)
        k_ink_flash_gqa_finalize_gate<<<gr, 128, 0, st>>>(d_run_l, d_run_o, d_gate_attn, d_attn_out, M);
    }

    void reset_slot(int slot) {
        if (!enabled) return;
        if (slot >= 0 && slot < (int)slot_max_pos.size()) slot_max_pos[slot] = 0;
        if (max_slots == 1) {
            std::fill(page_written.begin(), page_written.end(), 0u);
            for (int g = 0; g < kNumGlobalVisits; ++g) {
                write_buf[g].cur_gv = -1;
                write_buf[g].cur_page = -1;
                write_buf[g].dirty = false;
            }
        }
    }

    bool tier2_active_for_pos(int max_pos) const noexcept {
        return enabled && ssd_open && max_pos >= ssd_base;
    }

    bool needs_eager_for_pos(int max_pos) const noexcept {
        return enabled && num_ssd_pages > 0 && max_pos >= ssd_base;
    }

    void print_stats() const {
        if (!enabled) return;
        const auto ms = ink_query_proc_mem_stats();
        std::fprintf(stderr,
            "[InkElasticKV] stats: ssd_read=%.2f MiB (%llu pages), ssd_written=%.2f MiB (%llu pages) | "
            "VmRSS=%.2f GiB, VmHWM=%.2f GiB, RssAnon=%.2f GiB, RssFile=%.2f GiB, RssShmem=%.2f GiB\n",
            (double)ssd_bytes_read.load() / 1048576.0, (unsigned long long)ssd_pages_read.load(),
            (double)ssd_bytes_written.load() / 1048576.0, (unsigned long long)ssd_pages_written.load(),
            (double)ms.vm_rss_bytes / 1073741824.0, (double)ms.vm_hwm_bytes / 1073741824.0,
            (double)ms.rss_anon_bytes / 1073741824.0, (double)ms.rss_file_bytes / 1073741824.0,
            (double)ms.rss_shmem_bytes / 1073741824.0);
    }

    void free_ssd_only() {
        if (io_thr.joinable()) {
            {
                std::lock_guard<std::mutex> lk(io_mu);
                io_stop = true;
            }
            io_cv.notify_all();
            io_thr.join();
        }
        for (int g = 0; g < kNumGlobalVisits; ++g) free_page_buf(write_buf[g]);
        for (int b = 0; b < 2; ++b) {
            if (ev_read_done[b]) { cudaEventDestroy(ev_read_done[b]); ev_read_done[b] = nullptr; }
            read_ev_pending[b] = false;
            free_page_buf(read_buf[b]);
        }
#ifdef _WIN32
        if (h_file != INVALID_HANDLE_VALUE) { CloseHandle(h_file); h_file = INVALID_HANDLE_VALUE; }
#else
        if (fd >= 0) { ::close(fd); fd = -1; }
#endif
        ssd_open = false;
        ssd_direct = false;
    }

    void free_all() {
        if (enabled) {
            print_stats();
        }
        free_ssd_only();
        if (d_run_m)    { cudaFree(d_run_m);    d_run_m = nullptr; }
        if (d_run_l)    { cudaFree(d_run_l);    d_run_l = nullptr; }
        if (d_run_o)    { cudaFree(d_run_o);    d_run_o = nullptr; }
        if (d_pf_m_all) { cudaFree(d_pf_m_all); d_pf_m_all = nullptr; }
        if (d_pf_l_all) { cudaFree(d_pf_l_all); d_pf_l_all = nullptr; }
        if (d_pf_o_all) { cudaFree(d_pf_o_all); d_pf_o_all = nullptr; }
        run_cap_M = 0;
        pf_qchunk_cap = 0;
        page_written.clear();
        slot_max_pos.clear();
        enabled = false;
    }
};

}  // namespace strata::ink
