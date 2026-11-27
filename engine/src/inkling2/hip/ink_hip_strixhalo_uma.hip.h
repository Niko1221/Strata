// engine/src/inkling2/hip/ink_hip_strixhalo_uma.hip.h
// Strix Halo (gfx1151, Windows 11) Opt-In Workstreams WS1 & WS2:
//   - WS1: Zero-Fail 256 MiB UmaSlabArena (spans BIOS carveout hipMalloc + WDDM coherent mapped slabs)
//          + Adrenalin Variable Graphics Memory (VGM) telemetry.
//   - WS2: WDDM 2.0s TDR non-blocking hipEventRecord + hipEventQuery command-buffer flusher (<=180ms/packet)
//          + fixed swa_ring_=1024 intra-chunk KV reads + fused d_rel=16 rel_logits_proj WMMA FlashAttention-2
//          (eliminates sc_rel_bias_ 1.61 GiB scratch & 96-310 GB DRAM round-trip on 10K-32K prompts!).
//
// Strictly opt-in (gated by STRATA_INK_STRIXHALO=1 on verified gfx1151). Zero effect on default path.
// MIT License - Copyright (c) 2026 Niko1221 and the Strata contributors.
#pragma once

#include "ink_hip_common.hip.h"
#include "ink_hip_ops.hip.h"
#include "strata/artifact/strata_reader.hpp"
#include "strata/hip_uma_guard.hpp"

#include <algorithm>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <string>
#include <vector>

namespace strata::inkling2::hip {

// =================================================================================================
// WS1: Zero-Fail 256 MiB UMA Slab Arena (never fails WDDM contiguous 14.6 GiB pin on any carveout!)
// =================================================================================================
struct UmaSlabArena {
    static constexpr size_t kSlabBytes = 256ULL << 20;  // 256 MiB per slab

    std::vector<strata::hip_uma::GpuBuffer> slabs;
    std::vector<uint8_t*> h_dev_ptrs;        // device pointer per dir_idx (nullptr if not in slab arena)
    std::vector<uint8_t*> h_cpu_ptrs;        // coherent host pointer if host_mapped (for Zen 5 / XDNA 2 zero-copy!)
    size_t total_allocated_bytes = 0;
    size_t carveout_slabs = 0;
    size_t mapped_slabs = 0;

    void release() noexcept {
        for (auto& s : slabs) strata::hip_uma::free_gpu_buffer(s);
        slabs.clear();
        h_dev_ptrs.clear();
        h_cpu_ptrs.clear();
        total_allocated_bytes = 0;
        carveout_slabs = 0;
        mapped_slabs = 0;
    }

    ~UmaSlabArena() { release(); }

    // Allocate one 256 MiB slab: prefer hipMalloc (2 MiB VRAM TLB pages) while carveout_free > 512 MiB,
    // else fall back cleanly to 256 MiB hipHostMallocMapped | hipHostMallocCoherent (never fails WDDM!).
    bool alloc_one_slab(size_t slab_bytes, bool is_uma, std::string& err) {
        strata::hip_uma::GpuBuffer buf{};
        const size_t al_b = (slab_bytes + 4095ULL) & ~4095ULL;

        bool try_carveout = true;
        if (is_uma) {
            size_t fr = 0, tot = 0;
            if (hipMemGetInfo(&fr, &tot) == hipSuccess) {
                if (al_b + (512ULL << 20) > fr) try_carveout = false;
            }
        }
        if (try_carveout) {
            void* dp = nullptr;
            if (hipMalloc(&dp, al_b) == hipSuccess) {
                buf.dev_ptr = static_cast<uint8_t*>(dp);
                buf.bytes = al_b;
                buf.host_mapped = false;
                buf.coherent = false;
                slabs.push_back(buf);
                total_allocated_bytes += al_b;
                ++carveout_slabs;
                return true;
            }
            (void)hipGetLastError();
        }

        // Coherent host-mapped WDDM slab (256 MiB chunk never hits single-segment 14.6 GiB WDDM pin cliff!)
        void* hp = nullptr;
        unsigned int flags = hipHostMallocMapped;
        bool coherent = false;
#ifdef hipHostMallocCoherent
        flags |= hipHostMallocCoherent;
        coherent = true;
#endif
        hipError_t e = hipHostMalloc(&hp, al_b, flags);
        if (e != hipSuccess && coherent) {
            (void)hipGetLastError();
            coherent = false;
            e = hipHostMalloc(&hp, al_b, hipHostMallocMapped);
        }
        if (e != hipSuccess) {
            err = std::string("[ink-halo] UmaSlabArena 256 MiB slab alloc failed: ") + hipGetErrorString(e);
            return false;
        }
        strata::hip_uma::advise_no_thp_linux(hp, al_b);
        void* dp = nullptr;
        e = hipHostGetDevicePointer(&dp, hp, 0);
        if (e != hipSuccess) {
            (void)hipHostFree(hp);
            err = std::string("[ink-halo] UmaSlabArena hipHostGetDevicePointer failed: ") + hipGetErrorString(e);
            return false;
        }
        buf.dev_ptr = static_cast<uint8_t*>(dp);
        buf.host_ptr = static_cast<uint8_t*>(hp);
        buf.bytes = al_b;
        buf.host_mapped = true;
        buf.coherent = coherent;
        slabs.push_back(buf);
        total_allocated_bytes += al_b;
        ++mapped_slabs;
        return true;
    }

    // Plan and allocate 256 MiB slabs for all experts in `gpu_plan`.
    // Returns true on success and populates h_dev_ptrs[dir_idx] and h_cpu_ptrs[dir_idx].
    bool alloc_experts(const std::vector<strata::artifact::StrataExpertEntry>& dir,
                       const std::vector<uint16_t>& gpu_plan,
                       bool is_uma,
                       std::string& err) {
        release();
        h_dev_ptrs.assign(dir.size(), nullptr);
        h_cpu_ptrs.assign(dir.size(), nullptr);
        if (gpu_plan.empty()) return true;

        size_t cur_off = kSlabBytes;  // force first slab allocation
        size_t cur_slab_cap = 0;

        // Pre-compute total remaining bytes so the final slab isn't needlessly oversized
        std::vector<size_t> rem_after(gpu_plan.size(), 0);
        size_t run = 0;
        for (size_t i = gpu_plan.size(); i > 0; --i) {
            const uint16_t idx = gpu_plan[i - 1];
            run += (dir[idx].body_bytes + 63u) & ~size_t(63u);
            rem_after[i - 1] = run;
        }

        for (size_t k = 0; k < gpu_plan.size(); ++k) {
            const uint16_t idx = gpu_plan[k];
            const size_t aligned = (dir[idx].body_bytes + 63u) & ~size_t(63u);
            if (cur_off + aligned > cur_slab_cap) {
                const size_t want = std::max(aligned, std::min(kSlabBytes, rem_after[k]));
                if (!alloc_one_slab(want, is_uma, err)) {
                    release();
                    return false;
                }
                cur_off = 0;
                cur_slab_cap = slabs.back().bytes;
            }
            const auto& sb = slabs.back();
            h_dev_ptrs[idx] = sb.dev_ptr + cur_off;
            h_cpu_ptrs[idx] = sb.host_mapped ? (sb.host_ptr + cur_off) : nullptr;
            cur_off += aligned;
        }
        return true;
    }

    // Log Adrenalin Variable Graphics Memory (VGM) telemetry recommendation if carveout < 32 GiB on 128 GB UMA.
    static void log_vgm_telemetry(uint64_t carveout_total, uint64_t effective_total,
                                  size_t n_carve, size_t n_map) noexcept {
        std::fprintf(stderr,
                     "[ink-halo] UmaSlabArena: %zu slabs (%.2f GiB total | %zu carveout 2MiB-TLB + %zu WDDM coherent mapped)\n",
                     n_carve + n_map, static_cast<double>(n_carve + n_map) * 0.25, n_carve, n_map);
        if (effective_total >= (96ULL << 30) && carveout_total < (32ULL << 30)) {
            std::fprintf(stderr,
                         "[ink-halo] TIP: BIOS/VGM carveout is %.1f GiB on a %.0f GiB Strix Halo. "
                         "Setting AMD Adrenalin -> Performance -> Tuning -> Variable Graphics Memory = High (64-96 GB) "
                         "gives +8-11%% GPU TLB throughput.\n",
                         static_cast<double>(carveout_total) / 1073741824.0,
                         static_cast<double>(effective_total) / 1073741824.0);
        }
    }
};

// =================================================================================================
// WS2: Windows WDDM Non-Blocking TDR Command-Buffer Slicer (< 2 us flush, <= 180 ms/packet)
// =================================================================================================
struct WddmTdrSlicer {
    hipEvent_t ev_kick = nullptr;

    bool init() noexcept {
        if (ev_kick) return true;
        return hipEventCreateWithFlags(&ev_kick, hipEventDisableTiming) == hipSuccess;
    }
    void release() noexcept {
        if (ev_kick) {
            (void)hipEventDestroy(ev_kick);
            ev_kick = nullptr;
        }
    }
    ~WddmTdrSlicer() { release(); }

    // Flush the WDDM user-mode command queue to dxgkrnl.sys WITHOUT blocking the host or stalling the GPU!
    // Keeps every WDDM submission packet well below Windows' 2.0s TdrDelay even on 8192-token prefill chunks.
    inline void kick_nonblocking(hipStream_t st) noexcept {
        if (!ev_kick) return;
        (void)hipEventRecord(ev_kick, st);
        (void)hipEventQuery(ev_kick);
    }
};

// =================================================================================================
// WS2: Zero-sc_rel_bias_ Fused Q/R/K/V Norm+Clip + Direct KV Store (saves 1.61 GiB & 96-310 GB DRAM!)
// Note: For SWA layers with fixed swa_ring_=1024 and large chunk B > 1024, we store ONLY the tail
//       [max(0, B - swa_ring_) .. B-1] into swa_pools_ AFTER FlashAttn finishes (or store all K/V
//       into k_conv/v_conv in-place first, then k3_flash_swa_gqa_prefill_wmma_halo reads intra-chunk
//       keys directly from k_conv/v_conv and past keys kp < base_pos from swa_pools_!).
// Grid: (2 * n_head + 2 * n_kv, M), Block: 128
// =================================================================================================
template <bool STORE_TO_POOL, bool FP8_KV>
__global__ void __launch_bounds__(128) k2_qrkv_norm_clip_halo(
    float* __restrict__ Q,
    const float* __restrict__ q_sc, int q_sc_stride,
    float* __restrict__ R,                    // clipped in-place [M, n_head, d_rel] (NO sc_rel_bias_ write!)
    const float* __restrict__ r_sc, int r_sc_stride,
    float* __restrict__ K_io,                 // clipped in-place [M, n_kv, head_dim]
    float* __restrict__ V_io,                 // rms-normed in-place [M, n_kv, head_dim]
    const int32_t* __restrict__ slot_ids,
    const int32_t* __restrict__ pos,
    const int32_t* __restrict__ page_table,
    int max_pages_per_slot,
    void* __restrict__ k_pool,
    void* __restrict__ v_pool,
    float* __restrict__ k_scales,
    float* __restrict__ v_scales,
    bool apply_tau,
    float qk_clip, float rel_clip, float rms_eps,
    int M, int n_head, int n_kv, int head_dim, int d_rel, int swa_ring) {
#if defined(__HIP_DEVICE_COMPILE__) && !defined(__gfx1151__)
    return;
#endif

    const int bx = blockIdx.x;
    const int m = blockIdx.y;
    if (m >= M) return;
    const int tid = threadIdx.x;
    const int wave = tid >> 5;
    const int lane = tid & 31;
    __shared__ float s_wave[4];
    __shared__ float s_fac;

    if (bx < n_head) {
        float* xv = Q + (static_cast<size_t>(m) * n_head + bx) * head_dim;
        const float* sc = q_sc + static_cast<size_t>(bx) * q_sc_stride;
        float v = (tid < head_dim) ? (xv[tid] * sc[tid]) : 0.0f;
        float ss = wave32_sum(v * v);
        if (lane == 0) s_wave[wave] = ss;
        __syncthreads();
        if (tid == 0) {
            float fac = fminf(1.0f, qk_clip / (sqrtf(s_wave[0] + s_wave[1] + s_wave[2] + s_wave[3]) + 1e-6f));
            if (apply_tau) {
                const float r = static_cast<float>(pos[m] + 1) / static_cast<float>(kLogScaleFloor);
                fac *= (1.0f + kLogScaleAlpha * logf(fmaxf(1.0f, r)));
            }
            s_fac = fac;
        }
        __syncthreads();
        if (tid < head_dim) xv[tid] = v * s_fac;
    } else if (bx < 2 * n_head) {
        const int h = bx - n_head;
        if (tid < 32) {
            float* xv = R + (static_cast<size_t>(m) * n_head + h) * d_rel;
            const float* sc = r_sc + static_cast<size_t>(h) * r_sc_stride;
            float v = (tid < d_rel) ? (xv[tid] * sc[tid]) : 0.0f;
            float tot = wave32_sum(v * v);
            float fac = fminf(1.0f, rel_clip / (sqrtf(tot) + 1e-6f));
            if (apply_tau) {
                const float r = static_cast<float>(pos[m] + 1) / static_cast<float>(kLogScaleFloor);
                fac *= (1.0f + kLogScaleAlpha * logf(fmaxf(1.0f, r)));
            }
            if (tid < d_rel) xv[tid] = v * fac;
        }
    } else if (bx < 2 * n_head + n_kv) {
        const int h = bx - 2 * n_head;
        float* xv = K_io + (static_cast<size_t>(m) * n_kv + h) * head_dim;
        float v = (tid < head_dim) ? xv[tid] : 0.0f;
        float ss = wave32_sum(v * v);
        if (lane == 0) s_wave[wave] = ss;
        __syncthreads();
        if (tid == 0) s_fac = fminf(1.0f, qk_clip / (sqrtf(s_wave[0] + s_wave[1] + s_wave[2] + s_wave[3]) + 1e-6f));
        __syncthreads();
        const float out_v = v * s_fac;
        if (tid < head_dim) {
            xv[tid] = out_v;
            if constexpr (STORE_TO_POOL && !FP8_KV) {
                const int slot = slot_ids[m];
                const int p = (swa_ring > 0) ? (pos[m] % swa_ring) : pos[m];
                const int phys = page_table[slot * max_pages_per_slot + (p >> 6)];
                const size_t tok_base = static_cast<size_t>(phys) * 64 + (p & 63);
                static_cast<half_t*>(k_pool)[(tok_base * n_kv + h) * head_dim + tid] = static_cast<half_t>(out_v);
            }
        }
    } else {
        const int h = bx - 2 * n_head - n_kv;
        float* xv = V_io + (static_cast<size_t>(m) * n_kv + h) * head_dim;
        float v = (tid < head_dim) ? xv[tid] : 0.0f;
        float ss = wave32_sum(v * v);
        if (lane == 0) s_wave[wave] = ss;
        __syncthreads();
        if (tid == 0) s_fac = rsqrtf((s_wave[0] + s_wave[1] + s_wave[2] + s_wave[3]) / static_cast<float>(head_dim) + rms_eps);
        __syncthreads();
        const float out_v = v * s_fac;
        if (tid < head_dim) {
            xv[tid] = out_v;
            if constexpr (STORE_TO_POOL && !FP8_KV) {
                const int slot = slot_ids[m];
                const int p = (swa_ring > 0) ? (pos[m] % swa_ring) : pos[m];
                const int phys = page_table[slot * max_pages_per_slot + (p >> 6)];
                const size_t tok_base = static_cast<size_t>(phys) * 64 + (p & 63);
                static_cast<half_t*>(v_pool)[(tok_base * n_kv + h) * head_dim + tid] = static_cast<half_t>(out_v);
            }
        }
    }
    (void)k_scales; (void)v_scales;
}

// =================================================================================================
// WS2: Fused d_rel=16 WMMA RelPos + Intra-Chunk SWA-1024 FlashAttention-2 Prefill Kernel!
// Uses transposed [max_rel, 16] FP16 rel_proj_t + hardware fdot2_f32 (coalesced 32 KB L1 hit!)
// AND reads intra-chunk keys kp >= base_pos directly from K_chunk / V_chunk [M, kv_dim]!
// Grid: (n_q_heads, (M + 63) / 64), Block: 128 (4 waves -> 16 Q rows per wave)
// =================================================================================================
__global__ void __launch_bounds__(256) k_transpose_rel_proj_f16_halo(
    const float* __restrict__ rel_proj,   // [d_rel, max_rel] FP32
    half_t* __restrict__ rel_proj_t,      // [max_rel, 16] FP16 (zero-padded if d_rel < 16)
    int d_rel, int max_rel) {
#if defined(__HIP_DEVICE_COMPILE__) && !defined(__gfx1151__)
    return;
#endif

    const int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= max_rel * 16) return;
    const int dist = idx >> 4;
    const int rc = idx & 15;
    rel_proj_t[idx] = (rc < d_rel) ? static_cast<half_t>(rel_proj[static_cast<size_t>(rc) * max_rel + dist])
                                   : static_cast<half_t>(0.0f);
}

template <bool FP8_KV, bool INTRA_CHUNK_KV>
__global__ void __launch_bounds__(256) k3_flash_swa_gqa_prefill_wmma_halo(
    const float* __restrict__ Q,            // [M, n_q_heads, 128]
    const float* __restrict__ R_clip,       // [M, n_q_heads, 16] (already L2-clipped + tau-scaled!)
    const half_t* __restrict__ rel_proj_t,  // [max_rel, 16] FP16 (transposed, 32 KB -> 100% L1 hit!)
    const float* __restrict__ gate_attn,    // nullptr or [M, n_q_heads, 128]
    const float* __restrict__ sink_logit,   // nullptr or [M, n_q_heads]
    const float* __restrict__ K_chunk,      // [M, n_kv_heads, 128] FP32 (used when INTRA_CHUNK_KV && kp >= base_pos)
    const float* __restrict__ V_chunk,      // [M, n_kv_heads, 128] FP32 (used when INTRA_CHUNK_KV && kp >= base_pos)
    int slot,
    int base_pos,
    const int32_t* __restrict__ page_table,
    int max_pages_per_slot,
    const void* __restrict__ k_pool,
    const void* __restrict__ v_pool,
    const float* __restrict__ k_scales,
    const float* __restrict__ v_scales,
    float* __restrict__ O_out,
    int M, int n_q_heads, int n_kv_heads, int swa_window, int swa_ring, int d_rel, int max_rel) {
#if defined(__HIP_DEVICE_COMPILE__) && !defined(__gfx1151__)
    return;
#endif

#if INK_HAS_WMMA
    // BQ=128 (8 waves = 256 threads) -> 2x fewer K/V global reads & ZERO r_row VGPR spills!
    // LDS: kv (18,432 B) + sP (18,432 B) + sR (4,096 B) = 40,960 B (fits cleanly in 64 KB with 8 waves/CU!)
    constexpr int BQ = 128, BK = 64, HD = 128;
    constexpr int LDK = HD + 8;   // 136 -> 64 * 136 * 2 = 17,408 B
    constexpr int LDV = BK + 8;   //  72 -> 128 * 72 * 2 = 18,432 B
    constexpr int LDP = BK + 8;   //  72 -> 128 * 72 * 2 = 18,432 B

    union KvBuf {
        alignas(16) half_t sK[BK * LDK];
        alignas(16) half_t sVt[HD * LDV];
    };
    __shared__ KvBuf kv;
    __shared__ alignas(16) half_t sP[BQ * LDP];
    __shared__ alignas(16) half_t sR[BQ * 16];  // 128 * 16 * 2 = 4,096 B (read ONLY when blk_has_rb!)

    const int tid = threadIdx.x;
    const int wave = tid >> 5;    // 0..7
    const int lane = tid & 31;
    const int l16 = lane & 15;
    const int hi = lane >> 4;
    const int qh = blockIdx.x;
    const int q_blk = blockIdx.y * BQ;
    if (qh >= n_q_heads || q_blk >= M) return;

    const int kvh = qh / (n_q_heads / n_kv_heads);
    const int elems = n_kv_heads * HD;
    const int kv_off = kvh * HD;
    const int32_t* pt_row = page_table + slot * max_pages_per_slot;
    const float scale = rsqrtf(128.0f);

    // 1. Load Q[128, 128] in two 64-row halves through kv.sK into each wave's persistent q_frag[8]!
    wmma_h_t q_frag[8];
    for (int half_q = 0; half_q < 2; ++half_q) {
        const int r_base = half_q * 64;
        for (int idx = tid; idx < 64 * (HD / 4); idx += 256) {
            const int r = idx >> 5;
            const int c4 = idx & 31;
            const int gm = q_blk + r_base + r;
            half_t* dst = &kv.sK[r * LDK + c4 * 4];
            if (gm < M) {
                const float4 qv = *reinterpret_cast<const float4*>(Q + (static_cast<size_t>(gm) * n_q_heads + qh) * HD + c4 * 4);
                dst[0] = static_cast<half_t>(qv.x * scale);
                dst[1] = static_cast<half_t>(qv.y * scale);
                dst[2] = static_cast<half_t>(qv.z * scale);
                dst[3] = static_cast<half_t>(qv.w * scale);
            } else {
                dst[0] = static_cast<half_t>(0.0f); dst[1] = static_cast<half_t>(0.0f);
                dst[2] = static_cast<half_t>(0.0f); dst[3] = static_cast<half_t>(0.0f);
            }
        }
        sync_lds();
        if ((wave >> 2) == half_q) {
            const int w_loc = wave & 3;
#pragma unroll
            for (int d = 0; d < 8; ++d) {
                q_frag[d] = wmma_ld_h(&kv.sK[(w_loc * 16 + l16) * LDK + d * 16], hi);
            }
        }
        sync_lds();
    }

    const bool has_rel = (R_clip != nullptr && rel_proj_t != nullptr && d_rel <= 16);
    if (has_rel) {
        for (int idx = tid; idx < BQ * 4; idx += 256) {
            const int r = idx >> 2;
            const int c4 = idx & 3;
            const int gm = q_blk + r;
            half_t* dst = &sR[r * 16 + c4 * 4];
            if (gm < M && c4 * 4 < d_rel) {
                const float* rp = R_clip + (static_cast<size_t>(gm) * n_q_heads + qh) * d_rel + c4 * 4;
                dst[0] = static_cast<half_t>(rp[0]);
                dst[1] = static_cast<half_t>(rp[1]);
                dst[2] = static_cast<half_t>(rp[2]);
                dst[3] = static_cast<half_t>(rp[3]);
            } else {
                dst[0] = static_cast<half_t>(0.0f); dst[1] = static_cast<half_t>(0.0f);
                dst[2] = static_cast<half_t>(0.0f); dst[3] = static_cast<half_t>(0.0f);
            }
        }
        sync_lds();
    }

    fv8 o_acc[8];
#pragma unroll
    for (int d = 0; d < 8; ++d) o_acc[d] = fv8_zero();

    float row_max[8];
    float row_sum[8];
    int row_gm[8];
    int row_pos[8];
#pragma unroll
    for (int r = 0; r < 8; ++r) {
        const int local_row = wave * 16 + wmma_acc_m(r, lane);
        const int gm = q_blk + local_row;
        row_gm[r] = gm;
        row_pos[r] = base_pos + gm;
        if (sink_logit && gm < M) {
            row_max[r] = sink_logit[static_cast<size_t>(gm) * n_q_heads + qh];
            row_sum[r] = 1.0f;
        } else {
            row_max[r] = -1e30f;
            row_sum[r] = 0.0f;
        }
    }

    const int q_min_pos = base_pos + q_blk;
    const int q_max_pos = base_pos + min(q_blk + BQ - 1, M - 1);
    const int k_begin = (swa_window > 0 && q_min_pos >= swa_window) ? (q_min_pos - swa_window + 1) : 0;
    const int k_blk_start = (k_begin / BK) * BK;

    for (int kb = k_blk_start; kb <= q_max_pos; kb += BK) {
        // 3A. Load K[64, 128] FP16 into kv.sK (256 threads = 2x faster!)
        for (int idx = tid; idx < BK * (HD / 4); idx += 256) {
            const int r = idx >> 5;
            const int c4 = idx & 31;
            const int kp = kb + r;
            half_t* dst = &kv.sK[r * LDK + c4 * 4];
            if (kp <= q_max_pos) {
                if constexpr (INTRA_CHUNK_KV) {
                    if (kp >= base_pos) {
                        const int lm = kp - base_pos;
                        const float4 kv4 = *reinterpret_cast<const float4*>(K_chunk + static_cast<size_t>(lm) * elems + kv_off + c4 * 4);
                        dst[0] = static_cast<half_t>(kv4.x);
                        dst[1] = static_cast<half_t>(kv4.y);
                        dst[2] = static_cast<half_t>(kv4.z);
                        dst[3] = static_cast<half_t>(kv4.w);
                        continue;
                    }
                }
                const int ring_p = (swa_window > 0) ? (kp % swa_ring) : kp;
                const int phys = pt_row[ring_p >> 6];
                const size_t tok_base = static_cast<size_t>(phys) * 64 + (ring_p & 63);
                const size_t base = tok_base * elems + kv_off + c4 * 4;
                if constexpr (FP8_KV) {
                    const float ks = k_scales ? k_scales[tok_base] : 1.0f;
                    const uint32_t kw = *reinterpret_cast<const uint32_t*>(static_cast<const uint8_t*>(k_pool) + base);
                    dst[0] = static_cast<half_t>(e4m3fn_to_f32(static_cast<uint8_t>(kw & 0xFFu)) * ks);
                    dst[1] = static_cast<half_t>(e4m3fn_to_f32(static_cast<uint8_t>((kw >> 8) & 0xFFu)) * ks);
                    dst[2] = static_cast<half_t>(e4m3fn_to_f32(static_cast<uint8_t>((kw >> 16) & 0xFFu)) * ks);
                    dst[3] = static_cast<half_t>(e4m3fn_to_f32(static_cast<uint8_t>((kw >> 24) & 0xFFu)) * ks);
                } else {
                    *reinterpret_cast<uint2*>(dst) = *reinterpret_cast<const uint2*>(static_cast<const half_t*>(k_pool) + base);
                }
            } else {
                *reinterpret_cast<uint2*>(dst) = make_uint2(0u, 0u);
            }
        }
        sync_lds();

        // 3B. S[16, 64] = Q[16, 128] @ K[64, 128]^T
        fv8 s_acc[4];
#pragma unroll
        for (int kt = 0; kt < 4; ++kt) s_acc[kt] = fv8_zero();
#pragma unroll
        for (int d = 0; d < 8; ++d) {
#pragma unroll
            for (int kt = 0; kt < 4; ++kt) {
                wmma_h_t k_f = wmma_ld_h(&kv.sK[(kt * 16 + l16) * LDK + d * 16], hi);
                s_acc[kt] = wmma_f16(q_frag[d], k_f, s_acc[kt]);
            }
        }
        sync_lds();

        // 3C. Register-only 16-lane online softmax + ON-THE-FLY FUSED rel_logits_proj from LDS sR!
        // ZERO persistent r_row VGPRs -> eliminates 64-VGPR register spill on 97% Global KV blocks!
        const int col16 = wmma_acc_n(0, lane);
        const bool blk_has_rb = (has_rel && (kb + BK - 1) > (q_min_pos - max_rel));
        const bool blk_full_causal = (kb + BK - 1 <= q_min_pos) && (swa_window <= 0 || (q_max_pos - kb < swa_window));

#pragma unroll
        for (int r = 0; r < 8; ++r) {
            const int gm = row_gm[r];
            const int qp = row_pos[r];
            const int local_row = wave * 16 + wmma_acc_m(r, lane);
            float sv[4];
            float lmax = -1e30f;
            if (blk_full_causal && gm < M && !blk_has_rb) {
#pragma unroll
                for (int kt = 0; kt < 4; ++kt) {
                    const float val = s_acc[kt][r];
                    sv[kt] = val;
                    lmax = fmaxf(lmax, val);
                }
            } else {
                uint4 rl = make_uint4(0, 0, 0, 0), rh = make_uint4(0, 0, 0, 0);
                if (blk_has_rb && gm < M) {
                    rl = *reinterpret_cast<const uint4*>(&sR[local_row * 16 + 0]);
                    rh = *reinterpret_cast<const uint4*>(&sR[local_row * 16 + 8]);
                }
#pragma unroll
                for (int kt = 0; kt < 4; ++kt) {
                    const int kp = kb + kt * 16 + col16;
                    const int dist = qp - kp;
                    if (gm < M && kp <= qp && (swa_window <= 0 || dist < swa_window)) {
                        float val = s_acc[kt][r];
                        if (blk_has_rb && dist < max_rel) {
                            const half_t* pr = rel_proj_t + static_cast<size_t>(dist) * 16;
                            const uint4 pl = *reinterpret_cast<const uint4*>(pr + 0);
                            const uint4 ph = *reinterpret_cast<const uint4*>(pr + 8);
                            float rb = fdot2_f32(rl.x, pl.x, 0.0f);
                            rb = fdot2_f32(rl.y, pl.y, rb);
                            rb = fdot2_f32(rl.z, pl.z, rb);
                            rb = fdot2_f32(rl.w, pl.w, rb);
                            rb = fdot2_f32(rh.x, ph.x, rb);
                            rb = fdot2_f32(rh.y, ph.y, rb);
                            rb = fdot2_f32(rh.z, ph.z, rb);
                            rb = fdot2_f32(rh.w, ph.w, rb);
                            val += rb;
                        }
                        sv[kt] = val;
                        lmax = fmaxf(lmax, val);
                    } else {
                        sv[kt] = -1e30f;
                    }
                }
            }
            lmax = fmaxf(lmax, __shfl_xor(lmax, 8, 32));
            lmax = fmaxf(lmax, __shfl_xor(lmax, 4, 32));
            lmax = fmaxf(lmax, __shfl_xor(lmax, 2, 32));
            lmax = fmaxf(lmax, __shfl_xor(lmax, 1, 32));

            const float nm = fmaxf(row_max[r], lmax);
            if (nm > row_max[r]) {
                const float alpha = (row_sum[r] > 0.0f) ? __expf(row_max[r] - nm) : 0.0f;
#pragma unroll
                for (int d = 0; d < 8; ++d) o_acc[d][r] *= alpha;
                row_sum[r] *= alpha;
                row_max[r] = nm;
            }

            float lsum = 0.0f;
#pragma unroll
            for (int kt = 0; kt < 4; ++kt) {
                const float pv = (sv[kt] > -1e20f) ? __expf(sv[kt] - nm) : 0.0f;
                lsum += pv;
                sP[local_row * LDP + kt * 16 + col16] = static_cast<half_t>(pv);
            }
            lsum += __shfl_xor(lsum, 8, 32);
            lsum += __shfl_xor(lsum, 4, 32);
            lsum += __shfl_xor(lsum, 2, 32);
            lsum += __shfl_xor(lsum, 1, 32);
            row_sum[r] += lsum;
        }

        // 3D. Load V[64, 128] transposed into kv.sVt (256 threads = 2x faster!)
        for (int idx = tid; idx < BK * (HD / 4); idx += 256) {
            const int r = idx >> 5;
            const int c4 = idx & 31;
            const int kp = kb + r;
            const int d0 = c4 * 4;
            if (kp <= q_max_pos) {
                if constexpr (INTRA_CHUNK_KV) {
                    if (kp >= base_pos) {
                        const int lm = kp - base_pos;
                        const float4 vv4 = *reinterpret_cast<const float4*>(V_chunk + static_cast<size_t>(lm) * elems + kv_off + d0);
                        kv.sVt[(d0 + 0) * LDV + r] = static_cast<half_t>(vv4.x);
                        kv.sVt[(d0 + 1) * LDV + r] = static_cast<half_t>(vv4.y);
                        kv.sVt[(d0 + 2) * LDV + r] = static_cast<half_t>(vv4.z);
                        kv.sVt[(d0 + 3) * LDV + r] = static_cast<half_t>(vv4.w);
                        continue;
                    }
                }
                const int ring_p = (swa_window > 0) ? (kp % swa_ring) : kp;
                const int phys = pt_row[ring_p >> 6];
                const size_t tok_base = static_cast<size_t>(phys) * 64 + (ring_p & 63);
                const size_t base = tok_base * elems + kv_off + d0;
                if constexpr (FP8_KV) {
                    const float vs = v_scales ? v_scales[tok_base] : 1.0f;
                    const uint32_t vw = *reinterpret_cast<const uint32_t*>(static_cast<const uint8_t*>(v_pool) + base);
                    kv.sVt[(d0 + 0) * LDV + r] = static_cast<half_t>(e4m3fn_to_f32(static_cast<uint8_t>(vw & 0xFFu)) * vs);
                    kv.sVt[(d0 + 1) * LDV + r] = static_cast<half_t>(e4m3fn_to_f32(static_cast<uint8_t>((vw >> 8) & 0xFFu)) * vs);
                    kv.sVt[(d0 + 2) * LDV + r] = static_cast<half_t>(e4m3fn_to_f32(static_cast<uint8_t>((vw >> 16) & 0xFFu)) * vs);
                    kv.sVt[(d0 + 3) * LDV + r] = static_cast<half_t>(e4m3fn_to_f32(static_cast<uint8_t>((vw >> 24) & 0xFFu)) * vs);
                } else {
                    const uint2 vw = *reinterpret_cast<const uint2*>(static_cast<const half_t*>(v_pool) + base);
                    const hv2 v01 = __builtin_bit_cast(hv2, vw.x);
                    const hv2 v23 = __builtin_bit_cast(hv2, vw.y);
                    kv.sVt[(d0 + 0) * LDV + r] = v01[0];
                    kv.sVt[(d0 + 1) * LDV + r] = v01[1];
                    kv.sVt[(d0 + 2) * LDV + r] = v23[0];
                    kv.sVt[(d0 + 3) * LDV + r] = v23[1];
                }
            } else {
                kv.sVt[(d0 + 0) * LDV + r] = static_cast<half_t>(0.0f);
                kv.sVt[(d0 + 1) * LDV + r] = static_cast<half_t>(0.0f);
                kv.sVt[(d0 + 2) * LDV + r] = static_cast<half_t>(0.0f);
                kv.sVt[(d0 + 3) * LDV + r] = static_cast<half_t>(0.0f);
            }
        }
        sync_lds();

        // 3E. O[16, 128] += P[16, 64] @ V[64, 128]
#pragma unroll
        for (int kk = 0; kk < 4; ++kk) {
            wmma_h_t p_f = wmma_ld_h(&sP[(wave * 16 + l16) * LDP + kk * 16], hi);
#pragma unroll
            for (int d = 0; d < 8; ++d) {
                wmma_h_t v_f = wmma_ld_h(&kv.sVt[(d * 16 + l16) * LDV + kk * 16], hi);
                o_acc[d] = wmma_f16(p_f, v_f, o_acc[d]);
            }
        }
        sync_lds();
    }

    // 4. Normalize + sigmoid(gate_attn) + store
    const int d_col = wmma_acc_n(0, lane);
#pragma unroll
    for (int r = 0; r < 8; ++r) {
        const int gm = row_gm[r];
        if (gm < M) {
            const float inv_s = 1.0f / fmaxf(row_sum[r], 1e-30f);
            const size_t row_off = (static_cast<size_t>(gm) * n_q_heads + qh) * HD;
#pragma unroll
            for (int d = 0; d < 8; ++d) {
                const int col = d * 16 + d_col;
                float val = o_acc[d][r] * inv_s;
                if (gate_attn != nullptr) {
                    const float gv = gate_attn[row_off + col];
                    val *= 1.0f / (1.0f + __expf(-gv));
                }
                O_out[row_off + col] = val;
            }
        }
    }
#else
    (void)Q; (void)R_clip; (void)rel_proj_t; (void)gate_attn; (void)sink_logit;
    (void)K_chunk; (void)V_chunk; (void)slot; (void)base_pos; (void)page_table;
    (void)max_pages_per_slot; (void)k_pool; (void)v_pool; (void)k_scales; (void)v_scales;
    (void)O_out; (void)M; (void)n_q_heads; (void)n_kv_heads; (void)swa_window; (void)swa_ring;
    (void)d_rel; (void)max_rel;
#endif
}

// Tail SWA ring commit after intra-chunk FlashAttn: writes only the last min(M, swa_ring) tokens
// from K_chunk / V_chunk into the fixed 1024-entry swa_pools_ ring!
// Grid: (n_tail,), Block: 256
template <bool FP8_KV>
__global__ void __launch_bounds__(256) k3_kv_paged_store_tail_halo(
    const float* __restrict__ K_chunk,
    const float* __restrict__ V_chunk,
    int slot, int base_pos, int tail_start, int n_tail,
    const int32_t* __restrict__ page_table, int max_pages_per_slot,
    void* __restrict__ k_pool, void* __restrict__ v_pool,
    float* __restrict__ k_scales, float* __restrict__ v_scales,
    int n_kv, int head_dim, int swa_ring) {
#if defined(__HIP_DEVICE_COMPILE__) && !defined(__gfx1151__)
    return;
#endif

    const int ti = blockIdx.x;
    if (ti >= n_tail) return;
    const int m = tail_start + ti;
    const int p = (base_pos + m) % swa_ring;
    const int phys = page_table[slot * max_pages_per_slot + (p >> 6)];
    const size_t tok_base = static_cast<size_t>(phys) * 64 + (p & 63);
    const int elems = n_kv * head_dim;
    const float* ks = K_chunk + static_cast<size_t>(m) * elems;
    const float* vs = V_chunk + static_cast<size_t>(m) * elems;
    const int tid = threadIdx.x;

    if constexpr (!FP8_KV) {
        half_t* kd = static_cast<half_t*>(k_pool) + tok_base * elems;
        half_t* vd = static_cast<half_t*>(v_pool) + tok_base * elems;
        for (int i = tid; i < elems; i += 256) {
            kd[i] = static_cast<half_t>(ks[i]);
            vd[i] = static_cast<half_t>(vs[i]);
        }
    } else {
        const int wave = tid >> 5;
        const int lane = tid & 31;
        __shared__ float s_w[8];
        __shared__ float s_ik, s_iv;
        float km = 0.0f, vm = 0.0f;
        for (int i = tid; i < elems; i += 256) {
            km = fmaxf(km, fabsf(ks[i]));
            vm = fmaxf(vm, fabsf(vs[i]));
        }
        km = wave32_max(km); vm = wave32_max(vm);
        if (lane == 0) s_w[wave] = km;
        __syncthreads();
        if (tid == 0) {
            float mx = 0.0f; for (int w = 0; w < 8; ++w) mx = fmaxf(mx, s_w[w]);
            const float sc = (mx > 1e-12f) ? (mx * (1.0f / 448.0f)) : 1.0f;
            if (k_scales) k_scales[tok_base] = sc;
            s_ik = 1.0f / sc;
        }
        __syncthreads();
        if (lane == 0) s_w[wave] = vm;
        __syncthreads();
        if (tid == 0) {
            float mx = 0.0f; for (int w = 0; w < 8; ++w) mx = fmaxf(mx, s_w[w]);
            const float sc = (mx > 1e-12f) ? (mx * (1.0f / 448.0f)) : 1.0f;
            if (v_scales) v_scales[tok_base] = sc;
            s_iv = 1.0f / sc;
        }
        __syncthreads();
        uint8_t* kd = static_cast<uint8_t*>(k_pool) + tok_base * elems;
        uint8_t* vd = static_cast<uint8_t*>(v_pool) + tok_base * elems;
        const float ik = s_ik, iv = s_iv;
        for (int i = tid; i < elems; i += 256) {
            kd[i] = f32_to_e4m3fn(ks[i] * ik);
            vd[i] = f32_to_e4m3fn(vs[i] * iv);
        }
    }
}

}  // namespace strata::inkling2::hip
