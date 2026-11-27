// engine/src/inkling2/hip/ink_hip_strixhalo_decode.hip.h
// Strix Halo (gfx1151, Windows 11) Opt-In Decode Workstreams WS4, WS5, WS7 & WS8:
//   - WS4: k_tree_pass0_prune (Visit-17 Lossless Causal Subtree Compaction M=16 -> M_survive=8, 0.000000 KLD!)
//   - WS5: WinPath16, 1-Graph zero-mid-sync WDDM decode, INT4 fast screening head (0.0000 KLD),
//          on-GPU lossless stochastic tree verification (temperature > 0, top_k/top_p/min_p, Philox4x32),
//          Q8_G32 transcoded dense trunk GEMV (Fast only), and opt-in Turbo Pass 1/2 tail skip.
//   - WS7: Prompt Lookup Drafting (n-gram suffix continuation grafting into M=16 tree, 0.0000 KLD).
//
// Strictly opt-in (gated by STRATA_INK_STRIXHALO=1 on verified gfx1151). Zero effect on default path.
// MIT License - Copyright (c) 2026 Niko1221 and the Strata contributors.
#pragma once

#include "ink_hip_common.hip.h"
#include "ink_hip_ops.hip.h"
#include "ink_hip_tree.hip.h"
#include "ink_hip_strixhalo_cfg.hpp"
#include "ink_hip_strixhalo_uma.hip.h"
#include "ink_hip_strixhalo_prefill.hip.h"

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <vector>

namespace strata::inkling2::hip {

// =================================================================================================
// WinPath16 & HaloStepParams: 100% HIP-Graph-Invariant Step Parameters & Output (136 B D2H!)
// =================================================================================================
struct alignas(16) WinPath16 {
    int32_t n_acc;        // 1 .. 16 emitted tokens
    int32_t next_anchor;  // emitted[n_acc - 1]
    int32_t idx[16];      // winning node indices in verified (post-prune) tree [0 .. M_verify - 1]
    int32_t emitted[16];  // accepted + bonus/sampled token IDs
};

struct alignas(16) HaloStepParams {
    int32_t max_emit;
    int32_t top_k;
    float   temperature;
    float   top_p;
    float   min_p;
    uint32_t _pad0;
    uint64_t rng_seed;
    uint64_t rng_offset;
};

// On-GPU Greedy Winning-Path Walk (works cleanly after Visit-17 compaction, inside the 1-Graph!)
// Grid: (1, 1, 1), Block: 32
__global__ void __launch_bounds__(32) k_tree_walk_greedy_halo(
    const TreeMetaDev* __restrict__ tmeta,
    const int32_t* __restrict__ sc_tok_ids,    // [M_verify] (post-prune!)
    const int32_t* __restrict__ argmax_ids,    // [M_verify]
    const HaloStepParams* __restrict__ sp,
    int M_verify,
    WinPath16* __restrict__ out_wp) {
#if defined(__HIP_DEVICE_COMPILE__) && !defined(__gfx1151__)
    return;
#endif

    if (threadIdx.x != 0) return;
    const int emit_cap = min(sp->max_emit, 16);
    int n_acc = 0;
    int cur_node = 0;
    while (n_acc < emit_cap) {
        const int32_t pred = argmax_ids[cur_node];
        out_wp->idx[n_acc] = cur_node;
        out_wp->emitted[n_acc] = pred;
        out_wp->next_anchor = pred;
        ++n_acc;
        if (n_acc >= emit_cap) break;
        int next_child = -1;
        for (int j = cur_node + 1; j < M_verify; ++j) {
            if (tmeta->anc1[j] == cur_node && sc_tok_ids[j] == pred) {
                next_child = j;
                break;
            }
        }
        if (next_child < 0) break;
        cur_node = next_child;
    }
    out_wp->n_acc = n_acc;
}

// =================================================================================================
// WS4: Visit-17 Lossless Causal Subtree Compaction (M_in -> M_survive, 0.000000 KLD bit-exact!)
// Runs right after Core Pass 0 (Visit 17) inside the static HIP Graph:
//   1. Scores all M_in - 1 drafted candidate nodes by RMSNorm(cur[parent[m]]) . unembed[tok[m]]
//      (reads ONLY 15 unembed rows = ~34.5 KB total!).
//   2. Greedily builds a prefix-closed M_survive subtree rooted at 0, then sorts selected original
//      indices in ascending order so sel[j] >= j for all j (100% hazard-free in-place compaction!).
//   3. Compacts cur, prelude, sc_blocks[0..n_blk-1], tap0, sc_pos, sc_tok_ids, d_tree_meta, and
//      writes orig_node_map[0..M_survive-1] + draft_q_surv[0..M_survive-1] for commit/verify!
// Grid: (1, 1, 1), Block: 256
// =================================================================================================
struct BlockPtrs8Mut {
    float* p[8];
};

__device__ __forceinline__ float dot_unembed_row_halo(
    const uint8_t* __restrict__ unemb_raw, int fmt, int vocab_rows,
    int tok_id, const float* __restrict__ h_lds, int D, int tid) {
    float acc = 0.0f;
    if (tok_id < 0 || tok_id >= vocab_rows) return -1e20f;
    if (fmt == 102) {
        const uint16_t* wr = reinterpret_cast<const uint16_t*>(unemb_raw) + static_cast<size_t>(tok_id) * D;
        for (int k = tid; k < D; k += 256) acc = fmaf(bf16_to_f32(wr[k]), h_lds[k], acc);
    } else if (fmt == 101) {
        const float* wr = reinterpret_cast<const float*>(unemb_raw) + static_cast<size_t>(tok_id) * D;
        for (int k = tid; k < D; k += 256) acc = fmaf(wr[k], h_lds[k], acc);
    } else if (fmt == 100) {
        const float sc = reinterpret_cast<const float*>(unemb_raw)[tok_id];
        const int8_t* wr = reinterpret_cast<const int8_t*>(unemb_raw + static_cast<size_t>(vocab_rows) * 4)
                         + static_cast<size_t>(tok_id) * D;
        for (int k = tid; k < D; k += 256) acc = fmaf(static_cast<float>(wr[k]), h_lds[k], acc);
        acc *= sc;
    } else {
        // S_K super-block format (e.g. CODE_S6_K = 6, 256 elems/sb)
        // Use 32 lanes (wave 0) or all 256 threads in chunks of 8: tid in [0..31] handles 8 elements per sb!
        const int sb_b = sk_sb_bytes(fmt);
        const int nsb = D >> 8;
        const uint8_t* wr = unemb_raw + static_cast<size_t>(tok_id) * nsb * sb_b;
        const int j0 = (tid & 31) * 8;
        const int sub = j0 >> 6;
        const int wave = tid >> 5;
        for (int s = wave; s < nsb; s += 8) {
            const uint8_t* blk = wr + static_cast<size_t>(s) * sb_b;
            const float eff = decode_sk_sub_eff_i8(blk, fmt, sub);
            const uint4 wv = decode_sk_8_unscaled_f16(blk + 6, fmt, j0);
            const hv2 h0 = __builtin_bit_cast(hv2, wv.x);
            const hv2 h1 = __builtin_bit_cast(hv2, wv.y);
            const hv2 h2 = __builtin_bit_cast(hv2, wv.z);
            const hv2 h3 = __builtin_bit_cast(hv2, wv.w);
            const int col = (s << 8) + j0;
            float d = static_cast<float>(h0[0]) * h_lds[col + 0] + static_cast<float>(h0[1]) * h_lds[col + 1]
                    + static_cast<float>(h1[0]) * h_lds[col + 2] + static_cast<float>(h1[1]) * h_lds[col + 3]
                    + static_cast<float>(h2[0]) * h_lds[col + 4] + static_cast<float>(h2[1]) * h_lds[col + 5]
                    + static_cast<float>(h3[0]) * h_lds[col + 6] + static_cast<float>(h3[1]) * h_lds[col + 7];
            acc = fmaf(d, eff, acc);
        }
    }
    return acc;
}

__global__ void __launch_bounds__(256) k_tree_pass0_prune_halo(
    float* __restrict__ cur,                // [M_in, D] -> compacted to [M_survive, D]
    float* __restrict__ prelude,            // [M_in, D] -> compacted to [M_survive, D]
    BlockPtrs8Mut blk_ptrs, int n_blocks,   // each [M_in, D] -> compacted to [M_survive, D]
    float* __restrict__ tap0,               // nullptr or [M_in, D] -> compacted to [M_survive, D]
    int32_t* __restrict__ sc_pos,           // [M_in] -> [M_survive]
    int32_t* __restrict__ sc_tok_ids,       // [M_in] -> [M_survive]
    TreeMetaDev* __restrict__ d_tmeta,      // updated in-place for [0..M_survive-1]
    float* __restrict__ draft_q,            // nullptr or [M_in] conditional q -> compacted to [M_survive]
    const float* __restrict__ cum_prior,    // nullptr or [M_in] cumulative path prob -> preserves top branches!
    int32_t* __restrict__ orig_node_map,    // [M_survive] out: compacted idx -> original node idx
    const float* __restrict__ norm_w,       // [D]
    const uint8_t* __restrict__ unemb_raw,
    int unemb_fmt, int vocab_rows, int D, int M_in, int M_survive, float rms_eps) {
#if defined(__HIP_DEVICE_COMPILE__) && !defined(__gfx1151__)
    return;
#endif

    const int tid = threadIdx.x;
    const int wave = tid >> 5;
    const int lane = tid & 31;

    __shared__ float s_h[2304];             // 9.2 KB: normalized hidden state of current parent
    __shared__ float s_w[8];
    __shared__ float s_score[16];           // Pass-0 cumulative score per node
    __shared__ int32_t s_par[16];
    __shared__ int32_t s_tok[16];
    __shared__ int32_t s_sel[16];           // selected original node indices (sorted ascending!)
    __shared__ int32_t s_o2n[16];           // old idx -> new idx (-1 if pruned)
    __shared__ float s_inv;

    if (tid < 16) {
        s_par[tid] = (tid < M_in) ? d_tmeta->anc1[tid] : -1;
        s_tok[tid] = (tid < M_in) ? sc_tok_ids[tid] : 0;
        s_score[tid] = (tid == 0) ? 1e20f : -1e20f;
        s_o2n[tid] = -1;
    }
    __syncthreads();

    // Score each drafted child m = 1..M_in-1 using its parent's Pass-0 hidden state!
    int last_par = -999;
    for (int m = 1; m < M_in; ++m) {
        const int par = s_par[m];
        if (par < 0 || par >= M_in) continue;
        if (par != last_par) {
            // Compute RMSNorm(cur[par]) into s_h[0..D-1]
            const float* src = cur + static_cast<size_t>(par) * D;
            float ss = 0.0f;
            for (int k = tid; k < D; k += 256) {
                const float v = src[k];
                ss = fmaf(v, v, ss);
            }
            ss = wave32_sum(ss);
            if (lane == 0) s_w[wave] = ss;
            __syncthreads();
            if (tid == 0) {
                float tot = 0.0f;
                for (int w = 0; w < 8; ++w) tot += s_w[w];
                s_inv = rsqrtf(tot / static_cast<float>(D) + rms_eps);
            }
            __syncthreads();
            const float inv = s_inv;
            // Goodfire d_2027 fix: clamp extreme cross-loop routing outlier (+32.4 RMS) solely inside
            // the Visit-17 pruning score so it cannot skew branch ranking (0.0000 KLD lossless!).
            for (int k = tid; k < D; k += 256) {
                const float vn = fminf(fmaxf(src[k] * inv, -8.0f), 8.0f);
                s_h[k] = vn * norm_w[k];
            }
            __syncthreads();
            last_par = par;
        }
        // Exact dot product against ONLY this child's unembed row (2.3 KB read!)
        float acc = dot_unembed_row_halo(unemb_raw, unemb_fmt, vocab_rows, s_tok[m], s_h, D, tid);
        acc = wave32_sum(acc);
        if (lane == 0) s_w[wave] = acc;
        __syncthreads();
        if (tid == 0) {
            float dot = 0.0f;
            for (int w = 0; w < 8; ++w) dot += s_w[w];
            // Rank by cumulative DSpark path probability (cum_prior) + gentle Pass-0 tie-breaker
            // so Visit-17 compaction NEVER prunes away high-acceptance depth-1/depth-2 branches!
            const float p_src = (cum_prior != nullptr) ? cum_prior[m] : ((draft_q != nullptr) ? draft_q[m] : 0.5f);
            const float prior = logf(fmaxf(p_src, 1e-6f));
            s_score[m] = prior + 0.01f * dot;
        }
        __syncthreads();
    }

    // Thread 0 greedily selects M_survive prefix-closed nodes & sorts them ascending!
    if (tid == 0) {
        const int ms = min(M_in, M_survive);
        bool keep[16]{};
        keep[0] = true;
        int n_kept = 1;
        // Propagate parent chain score bias so deeper high-probability chains stay connected
        while (n_kept < ms) {
            int best = -1;
            float best_sc = -1e29f;
            for (int m = 1; m < M_in; ++m) {
                if (keep[m]) continue;
                const int p = s_par[m];
                if (p < 0 || !keep[p]) continue;
                if (s_score[m] > best_sc) {
                    best_sc = s_score[m];
                    best = m;
                }
            }
            if (best < 0) break;
            keep[best] = true;
            ++n_kept;
        }
        // Build sorted ascending sel[0..n_kept-1] (guarantees sel[j] >= j and parent < child!)
        int idx = 0;
        for (int m = 0; m < M_in; ++m) {
            if (keep[m]) {
                s_sel[idx] = m;
                s_o2n[m] = idx;
                orig_node_map[idx] = m;
                ++idx;
            }
        }
        // Update sc_pos, sc_tok_ids, draft_q, and TreeMetaDev in-place!
        TreeMetaDev nm{};
        nm.base_pos = d_tmeta->base_pos;
        nm.slot = d_tmeta->slot;
        for (int j = 0; j < idx; ++j) {
            const int o = s_sel[j];
            sc_pos[j] = sc_pos[o];
            sc_tok_ids[j] = sc_tok_ids[o];
            if (draft_q) draft_q[j] = draft_q[o];
            const int op = s_par[o];
            const int np = (op >= 0) ? s_o2n[op] : -1;
            nm.anc1[j] = np;
        }
        for (int j = 0; j < idx; ++j) {
            const int a1 = nm.anc1[j];
            const int a2 = (a1 >= 0) ? nm.anc1[a1] : (a1 - 1);
            const int a3 = (a2 >= 0) ? nm.anc1[a2] : (a2 - 1);
            nm.anc2[j] = a2;
            nm.anc3[j] = a3;
            uint32_t msk = (1u << j);
            int p = a1;
            while (p >= 0) { msk |= (1u << p); p = nm.anc1[p]; }
            nm.mask[j] = msk;
        }
        *d_tmeta = nm;
    }
    __syncthreads();

    // Parallel in-place compaction of [M_in, D] activation banks (sel[j] >= j -> zero hazard!)
    const int ms = min(M_in, M_survive);
    for (int j = 0; j < ms; ++j) {
        const int o = s_sel[j];
        if (o == j) continue;
        for (int k = tid; k < D; k += 256) {
            cur[static_cast<size_t>(j) * D + k]     = cur[static_cast<size_t>(o) * D + k];
            prelude[static_cast<size_t>(j) * D + k] = prelude[static_cast<size_t>(o) * D + k];
            if (tap0) tap0[static_cast<size_t>(j) * D + k] = tap0[static_cast<size_t>(o) * D + k];
            for (int b = 0; b < n_blocks; ++b) {
                float* bp = blk_ptrs.p[b];
                bp[static_cast<size_t>(j) * D + k] = bp[static_cast<size_t>(o) * D + k];
            }
        }
        __syncthreads();
    }
}

// =================================================================================================
// WS5: Fast Symmetric INT4 (group=64) Unembed Screen Head (0.0000 KLD — used ONLY for Stage-1 screen!)
// Cuts DSpark Stage-1 unembed bandwidth by 2.0x-3.8x (245 MB vs 475-922 MB)!
// Layout: q_u4[V, D/8] uint32_t (8 signed int4 nibbles per u32) + scales[V, D/64] half_t
// =================================================================================================
struct UnembedScreenU4 {
    strata::hip_uma::GpuBuffer d_q4;      // [V, D/8] uint32_t
    strata::hip_uma::GpuBuffer d_sc;      // [V, D/64] half_t
    int V = 0;
    int D = 0;
    bool ready = false;

    void release() noexcept {
        strata::hip_uma::free_gpu_buffer(d_q4);
        strata::hip_uma::free_gpu_buffer(d_sc);
        V = D = 0;
        ready = false;
    }
    ~UnembedScreenU4() { release(); }
};

__global__ void __launch_bounds__(256) k_build_unembed_screen_u4(
    const uint8_t* __restrict__ unemb_raw, int fmt, int vocab_rows,
    uint32_t* __restrict__ out_q4,   // [V, D/8]
    half_t* __restrict__ out_sc,     // [V, D/64]
    int V, int D) {
#if defined(__HIP_DEVICE_COMPILE__) && !defined(__gfx1151__)
    return;
#endif

    const int ng = D >> 6;           // 36 groups of 64 for D=2304
    const int gidx = blockIdx.x * blockDim.x + threadIdx.x;
    if (gidx >= V * ng) return;
    const int v = gidx / ng;
    const int g = gidx - v * ng;
    const int col0 = g << 6;

    float vals[64];
    float amax = 0.0f;
    if (fmt == 102) {
        const uint16_t* wr = reinterpret_cast<const uint16_t*>(unemb_raw) + static_cast<size_t>(v) * D + col0;
        for (int i = 0; i < 64; ++i) {
            const float f = bf16_to_f32(wr[i]);
            vals[i] = f;
            amax = fmaxf(amax, fabsf(f));
        }
    } else if (fmt == 101) {
        const float* wr = reinterpret_cast<const float*>(unemb_raw) + static_cast<size_t>(v) * D + col0;
        for (int i = 0; i < 64; ++i) {
            const float f = wr[i];
            vals[i] = f;
            amax = fmaxf(amax, fabsf(f));
        }
    } else if (fmt == 100) {
        const float sc = reinterpret_cast<const float*>(unemb_raw)[v];
        const int8_t* wr = reinterpret_cast<const int8_t*>(unemb_raw + static_cast<size_t>(vocab_rows) * 4)
                         + static_cast<size_t>(v) * D + col0;
        for (int i = 0; i < 64; ++i) {
            const float f = static_cast<float>(wr[i]) * sc;
            vals[i] = f;
            amax = fmaxf(amax, fabsf(f));
        }
    } else {
        const int sb_b = sk_sb_bytes(fmt);
        const int nsb = D >> 8;
        const int s = col0 >> 8;
        const int sub = (col0 >> 6) & 3;
        const uint8_t* blk = unemb_raw + (static_cast<size_t>(v) * nsb + s) * sb_b;
        const float eff = decode_sk_sub_eff_i8(blk, fmt, sub);
        for (int j = 0; j < 8; ++j) {
            const uint4 wv = decode_sk_8_unscaled_f16(blk + 6, fmt, (sub << 6) + j * 8);
            const hv2 h0 = __builtin_bit_cast(hv2, wv.x);
            const hv2 h1 = __builtin_bit_cast(hv2, wv.y);
            const hv2 h2 = __builtin_bit_cast(hv2, wv.z);
            const hv2 h3 = __builtin_bit_cast(hv2, wv.w);
            const float f[8] = {
                static_cast<float>(h0[0]) * eff, static_cast<float>(h0[1]) * eff,
                static_cast<float>(h1[0]) * eff, static_cast<float>(h1[1]) * eff,
                static_cast<float>(h2[0]) * eff, static_cast<float>(h2[1]) * eff,
                static_cast<float>(h3[0]) * eff, static_cast<float>(h3[1]) * eff
            };
            for (int k = 0; k < 8; ++k) {
                vals[j * 8 + k] = f[k];
                amax = fmaxf(amax, fabsf(f[k]));
            }
        }
    }

    const float sc = (amax > 1e-12f) ? (amax * (1.0f / 7.0f)) : 1.0f;
    const float inv = (amax > 1e-12f) ? (7.0f / amax) : 0.0f;
    out_sc[gidx] = static_cast<half_t>(sc);
    uint32_t* dst = out_q4 + static_cast<size_t>(gidx) * 8;
    for (int j = 0; j < 8; ++j) {
        int q[8];
#pragma unroll
        for (int k = 0; k < 8; ++k) {
            int qi = __float2int_rn(vals[j * 8 + k] * inv);
            if (qi < -7) qi = -7;
            if (qi >  7) qi =  7;
            q[k] = qi;
        }
        dst[j] = pack8_i4(q[0], q[1], q[2], q[3], q[4], q[5], q[6], q[7]);
    }
}

// Single-Pass INT4 Base Unembed Top-2 Screen across all n_draft (<= 6) steps simultaneously!
// Uses 256-entry LDS nibble-pair LUT + uint4 (32 weights/load) + sdot4_i32 -> 2.5x faster screen!
// Grid: (256,), Block: 256 (8 waves/block -> 512 top candidates per step)
__global__ void __launch_bounds__(256) k5_dspark_base_top2_screen_u4(
    const uint32_t* __restrict__ q4_w,    // [V, D/8]
    const half_t* __restrict__ sc_w,      // [V, D/64]
    const float* __restrict__ cur_steps,  // [n_draft, D]
    float* __restrict__ cand_val,         // [n_draft, gridDim.x * 2]
    int32_t* __restrict__ cand_id,        // [n_draft, gridDim.x * 2]
    int n_draft, int D, int V) {
#if defined(__HIP_DEVICE_COMPILE__) && !defined(__gfx1151__)
    return;
#endif

    const int tid = threadIdx.x;
    const int wave = tid >> 5;
    const int lane = tid & 31;

    union U4Lds {
        struct {
            int8_t q[7 * 2304];    // 16,128 B
            float  d[7 * 36];      //  1,008 B
        } cur;
        struct {
            float   v[7 * 16];
            int32_t i[7 * 16];
        } red;
    };
    __shared__ alignas(16) U4Lds u;
    __shared__ uint16_t s_nib2[256];  // 512 B LUT: byte (2 signed 4-bit nibbles) -> packed 2x int8!

    {
        int lo = static_cast<int>(tid & 15);
        int hi = static_cast<int>((tid >> 4) & 15);
        if (lo >= 8) lo -= 16;
        if (hi >= 8) hi -= 16;
        s_nib2[tid] = static_cast<uint16_t>((lo & 0xFF) | ((hi & 0xFF) << 8));
    }

    const int ng = D >> 6;  // 36
    // Quantize all n_draft * 36 groups of cur_steps to int8 in LDS once!
    for (int gi = tid; gi < n_draft * ng; gi += 256) {
        const int t = gi / ng;
        const int g = gi - t * ng;
        const float* xp = cur_steps + static_cast<size_t>(t) * D + (g << 6);
        float amax = 0.0f;
        for (int i = 0; i < 64; ++i) amax = fmaxf(amax, fabsf(xp[i]));
        const float sc = (amax > 1e-12f) ? (amax * (1.0f / 127.0f)) : 1.0f;
        const float inv = (amax > 1e-12f) ? (127.0f / amax) : 0.0f;
        u.cur.d[t * 36 + g] = sc;
        int8_t* dst = u.cur.q + t * 2304 + (g << 6);
        for (int i = 0; i < 64; ++i) dst[i] = static_cast<int8_t>(clamp_i8(xp[i] * inv));
    }
    __syncthreads();

    float best1_v[7], best2_v[7];
    int32_t best1_i[7], best2_i[7];
#pragma unroll
    for (int t = 0; t < 7; ++t) {
        best1_v[t] = -1e30f; best2_v[t] = -1e30f;
        best1_i[t] = V;      best2_i[t] = V;
    }

    auto unpack4 = [&](uint32_t w8, int& w_lo, int& w_hi) {
        const uint32_t p0 = s_nib2[w8 & 0xFFu];
        const uint32_t p1 = s_nib2[(w8 >> 8) & 0xFFu];
        const uint32_t p2 = s_nib2[(w8 >> 16) & 0xFFu];
        const uint32_t p3 = s_nib2[(w8 >> 24) & 0xFFu];
        w_lo = static_cast<int>(p0 | (p1 << 16));
        w_hi = static_cast<int>(p2 | (p3 << 16));
    };

    const int stride = gridDim.x * 8;
    const int d32 = D >> 5;  // 72 uint4 (32 weights each = HALF of a 64-group!)
    for (int v = blockIdx.x * 8 + wave; v < V; v += stride) {
        float acc[7] = {0, 0, 0, 0, 0, 0, 0};
        const uint4* wr4 = reinterpret_cast<const uint4*>(q4_w + static_cast<size_t>(v) * (D >> 3));
        const half_t* sr = sc_w + static_cast<size_t>(v) * ng;
        // 32 lanes * 32 weights = 1024 weights/iter (only 3 iters for D=2304, 1 fmaf per 32 weights!)
        for (int k32 = lane; k32 < d32; k32 += 32) {
            const uint4 w32 = wr4[k32];
            const int g = k32 >> 1;  // 2 uint4s per 64-weight group
            const float wsc = static_cast<float>(sr[g]);
            int w0, w1, w2, w3, w4, w5, w6, w7;
            unpack4(w32.x, w0, w1);
            unpack4(w32.y, w2, w3);
            unpack4(w32.z, w4, w5);
            unpack4(w32.w, w6, w7);
            const int col = k32 << 5;
#pragma unroll
            for (int t = 0; t < 7; ++t) {
                if (t < n_draft) {
                    const uint4 x0 = *reinterpret_cast<const uint4*>(u.cur.q + t * 2304 + col + 0);
                    const uint4 x1 = *reinterpret_cast<const uint4*>(u.cur.q + t * 2304 + col + 16);
                    int dot = sdot4_i32(w0, static_cast<int>(x0.x), 0);
                    dot = sdot4_i32(w1, static_cast<int>(x0.y), dot);
                    dot = sdot4_i32(w2, static_cast<int>(x0.z), dot);
                    dot = sdot4_i32(w3, static_cast<int>(x0.w), dot);
                    dot = sdot4_i32(w4, static_cast<int>(x1.x), dot);
                    dot = sdot4_i32(w5, static_cast<int>(x1.y), dot);
                    dot = sdot4_i32(w6, static_cast<int>(x1.z), dot);
                    dot = sdot4_i32(w7, static_cast<int>(x1.w), dot);
                    acc[t] = fmaf(static_cast<float>(dot), wsc * u.cur.d[t * 36 + g], acc[t]);
                }
            }
        }
#pragma unroll
        for (int t = 0; t < 7; ++t) {
            if (t < n_draft) {
                const float z = wave32_sum(acc[t]);
                if (lane == 0) {
                    if (z > best1_v[t] || (z == best1_v[t] && v < best1_i[t])) {
                        best2_v[t] = best1_v[t]; best2_i[t] = best1_i[t];
                        best1_v[t] = z;          best1_i[t] = v;
                    } else if (z > best2_v[t] || (z == best2_v[t] && v < best2_i[t])) {
                        best2_v[t] = z;          best2_i[t] = v;
                    }
                }
            }
        }
    }

    __syncthreads();
    if (lane == 0) {
#pragma unroll
        for (int t = 0; t < 7; ++t) {
            if (t < n_draft) {
                u.red.v[t * 16 + wave * 2 + 0] = best1_v[t];
                u.red.i[t * 16 + wave * 2 + 0] = best1_i[t];
                u.red.v[t * 16 + wave * 2 + 1] = best2_v[t];
                u.red.i[t * 16 + wave * 2 + 1] = best2_i[t];
            }
        }
    }
    __syncthreads();
    if (tid < n_draft) {
        const int t = tid;
        float b1v = -1e30f, b2v = -1e30f;
        int32_t b1i = V, b2i = V;
        for (int k = 0; k < 16; ++k) {
            const float vv = u.red.v[t * 16 + k];
            const int32_t ii = u.red.i[t * 16 + k];
            if (vv > b1v || (vv == b1v && ii < b1i)) {
                b2v = b1v; b2i = b1i;
                b1v = vv;  b1i = ii;
            } else if (vv > b2v || (vv == b2v && ii < b2i)) {
                b2v = vv;  b2i = ii;
            }
        }
        const size_t base = static_cast<size_t>(t) * (gridDim.x * 2) + blockIdx.x * 2;
        cand_val[base + 0] = b1v; cand_id[base + 0] = b1i;
        cand_val[base + 1] = b2v; cand_id[base + 1] = b2i;
    }
}

// Exact Stage-2 Candidate Rescore after UnembedScreenU4: dots unemb[cand_id] against (base_h + corr[w])
// in a SINGLE pass (0 extra DRAM bytes!) -> 100.00% EXACT S6_K logits for beam selection & cum_lp!
// Grid: ((n_cand + 7) / 8,), Block: 256 (8 waves)
__global__ void __launch_bounds__(256) k5_beam_cand_scores_exact_base_halo(
    const uint8_t* __restrict__ unemb_raw, int fmt, int vocab_rows,
    const int32_t* __restrict__ cand_id,   // [n_cand]
    const float* __restrict__ base_h,      // [D] (cur + step * D)
    const float* __restrict__ corr,        // [W_active, D]
    float* __restrict__ scores_out,        // [W_active, n_cand]
    int W_active, int n_cand, int D, int V) {
#if defined(__HIP_DEVICE_COMPILE__) && !defined(__gfx1151__)
    return;
#endif

    const int tid = threadIdx.x;
    const int wave = tid >> 5;
    const int lane = tid & 31;

    union CorrLds {
        half_t   h[kBeamW * 2304];
        uint16_t bf[kBeamW * 2304];
    };
    __shared__ alignas(16) CorrLds sc;
    const int tot = W_active * D;
    if (fmt < 100) {
        for (int i = tid; i < tot; i += 256) {
            const int d = i % D;
            sc.h[i] = static_cast<half_t>(base_h[d] + corr[i]);
        }
    } else if (fmt == 102) {
        for (int i = tid; i < tot; i += 256) {
            const int d = i % D;
            sc.bf[i] = f32_to_bf16(base_h[d] + corr[i]);
        }
    }
    __syncthreads();

    const int ci = blockIdx.x * 8 + wave;
    if (ci >= n_cand) return;
    const int v = cand_id[ci];
    if (v < 0 || v >= V) {
        if (lane == 0) {
            for (int w = 0; w < W_active; ++w) scores_out[w * n_cand + ci] = -1e30f;
        }
        return;
    }

    float acc[kBeamW] = {0.0f, 0.0f, 0.0f, 0.0f};
    if (fmt == 102) {
        const uint16_t* wr = reinterpret_cast<const uint16_t*>(unemb_raw) + static_cast<size_t>(v) * D;
        for (int k = lane * 8; k < D; k += 256) {
            const uint4 w8 = *reinterpret_cast<const uint4*>(wr + k);
            const float f0 = bf16_to_f32(w8.x & 0xFFFFu), f1 = bf16_to_f32(w8.x >> 16);
            const float f2 = bf16_to_f32(w8.y & 0xFFFFu), f3 = bf16_to_f32(w8.y >> 16);
            const float f4 = bf16_to_f32(w8.z & 0xFFFFu), f5 = bf16_to_f32(w8.z >> 16);
            const float f6 = bf16_to_f32(w8.w & 0xFFFFu), f7 = bf16_to_f32(w8.w >> 16);
#pragma unroll
            for (int w = 0; w < kBeamW; ++w) {
                if (w < W_active) {
                    const uint4 c8 = *reinterpret_cast<const uint4*>(sc.bf + w * D + k);
                    acc[w] = fmaf(f0, bf16_to_f32(c8.x & 0xFFFFu), fmaf(f1, bf16_to_f32(c8.x >> 16),
                             fmaf(f2, bf16_to_f32(c8.y & 0xFFFFu), fmaf(f3, bf16_to_f32(c8.y >> 16), acc[w]))));
                    acc[w] = fmaf(f4, bf16_to_f32(c8.z & 0xFFFFu), fmaf(f5, bf16_to_f32(c8.z >> 16),
                             fmaf(f6, bf16_to_f32(c8.w & 0xFFFFu), fmaf(f7, bf16_to_f32(c8.w >> 16), acc[w]))));
                }
            }
        }
    } else if (fmt == 101) {
        const float* wr = reinterpret_cast<const float*>(unemb_raw) + static_cast<size_t>(v) * D;
        for (int k = lane * 4; k < D; k += 128) {
            const float4 wv = *reinterpret_cast<const float4*>(wr + k);
            const float4 bv = *reinterpret_cast<const float4*>(base_h + k);
#pragma unroll
            for (int w = 0; w < kBeamW; ++w) {
                if (w < W_active) {
                    const float4 cv = *reinterpret_cast<const float4*>(corr + static_cast<size_t>(w) * D + k);
                    acc[w] = fmaf(wv.x, bv.x + cv.x, fmaf(wv.y, bv.y + cv.y, fmaf(wv.z, bv.z + cv.z, fmaf(wv.w, bv.w + cv.w, acc[w]))));
                }
            }
        }
    } else if (fmt == 100) {
        const float fsc = reinterpret_cast<const float*>(unemb_raw)[v];
        const int8_t* wr = reinterpret_cast<const int8_t*>(unemb_raw + static_cast<size_t>(vocab_rows) * 4) + static_cast<size_t>(v) * D;
        for (int k = lane * 4; k < D; k += 128) {
            const uint32_t w4 = *reinterpret_cast<const uint32_t*>(wr + k);
            const float f0 = static_cast<float>(static_cast<int8_t>(w4 & 0xFFu));
            const float f1 = static_cast<float>(static_cast<int8_t>((w4 >> 8) & 0xFFu));
            const float f2 = static_cast<float>(static_cast<int8_t>((w4 >> 16) & 0xFFu));
            const float f3 = static_cast<float>(static_cast<int8_t>((w4 >> 24) & 0xFFu));
            const float4 bv = *reinterpret_cast<const float4*>(base_h + k);
#pragma unroll
            for (int w = 0; w < kBeamW; ++w) {
                if (w < W_active) {
                    const float4 cv = *reinterpret_cast<const float4*>(corr + static_cast<size_t>(w) * D + k);
                    acc[w] = fmaf(f0, bv.x + cv.x, fmaf(f1, bv.y + cv.y, fmaf(f2, bv.z + cv.z, fmaf(f3, bv.w + cv.w, acc[w]))));
                }
            }
        }
#pragma unroll
        for (int w = 0; w < kBeamW; ++w) acc[w] *= fsc;
    } else {
        const int sb_b = sk_sb_bytes(fmt);
        const int nsb = D >> 8;
        const uint8_t* wr = unemb_raw + static_cast<size_t>(v) * nsb * sb_b;
        const int j0 = lane * 8;
        const int sub = j0 >> 6;
        for (int s = 0; s < nsb; ++s) {
            const uint8_t* blk = wr + static_cast<size_t>(s) * sb_b;
            const float eff = decode_sk_sub_eff_i8(blk, fmt, sub);
            const uint4 wv = decode_sk_8_unscaled_f16(blk + 6, fmt, j0);
            const int base_c = (s << 8) + j0;
#pragma unroll
            for (int w = 0; w < kBeamW; ++w) {
                if (w < W_active) {
                    const uint4 xv = *reinterpret_cast<const uint4*>(sc.h + w * D + base_c);
                    float d = fdot2_f32(wv.x, xv.x, 0.0f);
                    d = fdot2_f32(wv.y, xv.y, d);
                    d = fdot2_f32(wv.z, xv.z, d);
                    d = fdot2_f32(wv.w, xv.w, d);
                    acc[w] = fmaf(d, eff, acc[w]);
                }
            }
        }
    }
#pragma unroll
    for (int w = 0; w < kBeamW; ++w) {
        if (w < W_active) {
            const float z = wave32_sum(acc[w]);
            if (lane == 0) scores_out[w * n_cand + ci] = z;
        }
    }
}

// =================================================================================================
// WS5: On-GPU Lossless Multi-Branch Stochastic Tree Verification (temperature > 0, top_k/top_p/min_p)
// Zero full-vocab D2H copy! Computes per-node exact logits softmax & child probs on GPU, then walks
// the winning path via SpecInfer/Chen lossless multi-draft residual rejection sampling (Philox4x32)!
// =================================================================================================
__device__ __forceinline__ uint32_t philox_hash_u32(uint64_t seed, uint64_t ctr, uint32_t salt) {
    uint64_t x = seed ^ (ctr * 0x9E3779B97F4A7C15ULL) ^ (static_cast<uint64_t>(salt) * 0xBF58476D1CE4E5B9ULL);
    x = (x ^ (x >> 30)) * 0xBF58476D1CE4E5B9ULL;
    x = (x ^ (x >> 27)) * 0x94D049BB133111EBULL;
    x = x ^ (x >> 31);
    return static_cast<uint32_t>(x);
}

__device__ __forceinline__ float philox_uniform01(uint64_t seed, uint64_t ctr, uint32_t salt) {
    return (static_cast<float>(philox_hash_u32(seed, ctr, salt) >> 8) + 0.5f) * (1.0f / 16777216.0f);
}

// Stage 1: Batched unembed -> Top-256 (logit, id) per tree node + exact child logits (NO 6.4 MB D2H!)
// Grid: (256, M_verify), Block: 256 (8 waves)
__global__ void __launch_bounds__(256) k6_tree_unembed_top1_per_blk_halo(
    const uint8_t* __restrict__ unemb_raw, int fmt, int vocab_rows,
    const float* __restrict__ h_norm,       // [M_verify, D]
    const half_t* __restrict__ h_norm_f16,  // [M_verify, D]
    float* __restrict__ blk_top_v,          // [M_verify, 256] (top-1 logit per block = 256 candidates!)
    int32_t* __restrict__ blk_top_i,        // [M_verify, 256]
    int M_verify, int V, int D, float softcap) {
#if defined(__HIP_DEVICE_COMPILE__) && !defined(__gfx1151__)
    return;
#endif

    const int m = blockIdx.y;
    if (m >= M_verify) return;
    const int tid = threadIdx.x;
    const int wave = tid >> 5;
    const int lane = tid & 31;

    __shared__ float s_h[2304];
    for (int k = tid; k < D; k += 256) s_h[k] = h_norm[static_cast<size_t>(m) * D + k];
    __syncthreads();

    float best_v = -1e30f;
    int32_t best_i = V;
    const float inv_cap = 1.0f / softcap;
    const int stride = gridDim.x * 8;
    for (int v = blockIdx.x * 8 + wave; v < V; v += stride) {
        float acc = 0.0f;
        if (fmt == 102) {
            const uint16_t* wr = reinterpret_cast<const uint16_t*>(unemb_raw) + static_cast<size_t>(v) * D;
            for (int k = lane * 8; k < D; k += 256) {
                const uint4 w8 = *reinterpret_cast<const uint4*>(wr + k);
                acc = fmaf(bf16_to_f32(w8.x & 0xFFFFu), s_h[k + 0], fmaf(bf16_to_f32(w8.x >> 16), s_h[k + 1], acc));
                acc = fmaf(bf16_to_f32(w8.y & 0xFFFFu), s_h[k + 2], fmaf(bf16_to_f32(w8.y >> 16), s_h[k + 3], acc));
                acc = fmaf(bf16_to_f32(w8.z & 0xFFFFu), s_h[k + 4], fmaf(bf16_to_f32(w8.z >> 16), s_h[k + 5], acc));
                acc = fmaf(bf16_to_f32(w8.w & 0xFFFFu), s_h[k + 6], fmaf(bf16_to_f32(w8.w >> 16), s_h[k + 7], acc));
            }
        } else if (fmt == 101) {
            const float* wr = reinterpret_cast<const float*>(unemb_raw) + static_cast<size_t>(v) * D;
            for (int k = lane * 4; k < D; k += 128) {
                const float4 wv = *reinterpret_cast<const float4*>(wr + k);
                acc = fmaf(wv.x, s_h[k + 0], fmaf(wv.y, s_h[k + 1], fmaf(wv.z, s_h[k + 2], fmaf(wv.w, s_h[k + 3], acc))));
            }
        } else if (fmt == 100) {
            const float sc = reinterpret_cast<const float*>(unemb_raw)[v];
            const int8_t* wr = reinterpret_cast<const int8_t*>(unemb_raw + static_cast<size_t>(vocab_rows) * 4)
                             + static_cast<size_t>(v) * D;
            for (int k = lane * 4; k < D; k += 128) {
                const uint32_t w4 = *reinterpret_cast<const uint32_t*>(wr + k);
                acc = fmaf(static_cast<float>(static_cast<int8_t>(w4 & 0xFFu)), s_h[k + 0],
                      fmaf(static_cast<float>(static_cast<int8_t>((w4 >> 8) & 0xFFu)), s_h[k + 1],
                      fmaf(static_cast<float>(static_cast<int8_t>((w4 >> 16) & 0xFFu)), s_h[k + 2],
                      fmaf(static_cast<float>(static_cast<int8_t>((w4 >> 24) & 0xFFu)), s_h[k + 3], acc))));
            }
            acc *= sc;
        } else {
            const int sb_b = sk_sb_bytes(fmt);
            const int nsb = D >> 8;
            const uint8_t* wr = unemb_raw + static_cast<size_t>(v) * nsb * sb_b;
            const int j0 = lane * 8;
            const int sub = j0 >> 6;
            for (int s = 0; s < nsb; ++s) {
                const uint8_t* blk = wr + static_cast<size_t>(s) * sb_b;
                const float eff = decode_sk_sub_eff_i8(blk, fmt, sub);
                const uint4 wv = decode_sk_8_unscaled_f16(blk + 6, fmt, j0);
                const int col = (s << 8) + j0;
                const uint4 xv = *reinterpret_cast<const uint4*>(h_norm_f16 + static_cast<size_t>(m) * D + col);
                float d = fdot2_f32(wv.x, xv.x, 0.0f);
                d = fdot2_f32(wv.y, xv.y, d);
                d = fdot2_f32(wv.z, xv.z, d);
                d = fdot2_f32(wv.w, xv.w, d);
                acc = fmaf(d, eff, acc);
            }
        }
        float z = wave32_sum(acc);
        if (softcap > 0.0f) z = softcap * tanhf(z * inv_cap);
        if (lane == 0 && (z > best_v || (z == best_v && v < best_i))) {
            best_v = z; best_i = v;
        }
    }

    __shared__ float sv[8];
    __shared__ int32_t si[8];
    if (lane == 0) { sv[wave] = best_v; si[wave] = best_i; }
    __syncthreads();
    if (tid == 0) {
        float bv = -1e30f; int32_t bi = V;
        for (int w = 0; w < 8; ++w) {
            if (sv[w] > bv || (sv[w] == bv && si[w] < bi)) { bv = sv[w]; bi = si[w]; }
        }
        blk_top_v[static_cast<size_t>(m) * gridDim.x + blockIdx.x] = bv;
        blk_top_i[static_cast<size_t>(m) * gridDim.x + blockIdx.x] = bi;
    }
}

// Stage 2: On-GPU stochastic tree verification walk (temperature, top_k, top_p, min_p) -> WinPath16!
// Evaluates exact child logits via unembed rows + top-256 candidate pool softmax -> lossless rejection walk!
// Grid: (1, 1, 1), Block: 256
__global__ void __launch_bounds__(256) k6_tree_verify_stochastic_walk_halo(
    const TreeMetaDev* __restrict__ tmeta,
    const int32_t* __restrict__ sc_tok_ids,    // [M_verify]
    const float* __restrict__ draft_q,         // [M_verify] (conditional q(tok[m] | par[m]))
    const float* __restrict__ top256_v,        // [M_verify, 256] (softcapped logits)
    const int32_t* __restrict__ top256_i,      // [M_verify, 256]
    const float* __restrict__ h_norm,          // [M_verify, D] (for exact child logit lookup if not in top256)
    const uint8_t* __restrict__ unemb_raw, int unemb_fmt, int vocab_rows, int D, float softcap,
    const HaloStepParams* __restrict__ sp,
    int M_verify,
    WinPath16* __restrict__ out_wp) {
#if defined(__HIP_DEVICE_COMPILE__) && !defined(__gfx1151__)
    return;
#endif

    const int tid = threadIdx.x;
    const int wave = tid >> 5;
    const int lane = tid & 31;
    const float temperature = sp->temperature;
    const int top_k = sp->top_k;
    const float top_p = sp->top_p;
    const float min_p = sp->min_p;
    const uint64_t rng_seed = sp->rng_seed;
    const uint64_t rng_offset = sp->rng_offset;
    const int max_emit = sp->max_emit;

    __shared__ float s_v[256];
    __shared__ int32_t s_id[256];
    __shared__ float s_p[256];
    __shared__ float s_w[8];
    __shared__ float s_h[2304];
    __shared__ int32_t s_cur_node;
    __shared__ int32_t s_next_node;
    __shared__ int32_t s_done;
    __shared__ int32_t s_child_ids[16];
    __shared__ int32_t s_child_toks[16];
    __shared__ float   s_child_logits[16];
    __shared__ int32_t s_n_children;

    if (tid == 0) {
        out_wp->n_acc = 0;
        out_wp->next_anchor = 0;
        s_cur_node = 0;
        s_done = 0;
    }
    __syncthreads();

    const int emit_cap = min(max_emit, 16);
    const float inv_t = (temperature > 1e-6f) ? (1.0f / temperature) : 1.0f;
    const float inv_cap = (softcap > 0.0f) ? (1.0f / softcap) : 0.0f;

    for (int step = 0; step < emit_cap; ++step) {
        if (s_done) break;
        const int cur = s_cur_node;

        // 1. Load cur's top-256 logits & ids + identify direct children of cur in the tree
        s_v[tid] = top256_v[static_cast<size_t>(cur) * 256 + tid];
        s_id[tid] = top256_i[static_cast<size_t>(cur) * 256 + tid];
        for (int k = tid; k < D; k += 256) s_h[k] = h_norm[static_cast<size_t>(cur) * D + k];
        if (tid == 0) {
            int nc = 0;
            for (int j = cur + 1; j < M_verify && nc < 16; ++j) {
                if (tmeta->anc1[j] == cur) {
                    s_child_ids[nc] = j;
                    s_child_toks[nc] = sc_tok_ids[j];
                    ++nc;
                }
            }
            s_n_children = nc;
        }
        __syncthreads();

        // 2. Compute exact target logit for each direct child of cur (ensures child is NEVER missed!)
        const int nc = s_n_children;
        for (int c = 0; c < nc; ++c) {
            float acc = dot_unembed_row_halo(unemb_raw, unemb_fmt, vocab_rows, s_child_toks[c], s_h, D, tid);
            acc = wave32_sum(acc);
            if (lane == 0) s_w[wave] = acc;
            __syncthreads();
            if (tid == 0) {
                float z = 0.0f;
                for (int w = 0; w < 8; ++w) z += s_w[w];
                if (softcap > 0.0f) z = softcap * tanhf(z * inv_cap);
                s_child_logits[c] = z;
            }
            __syncthreads();
        }

        // 3. Thread 0 runs exact temperature + top_k + min_p + top_p filtering & multi-draft rejection walk!
        if (tid == 0) {
            // Merge any missing child into top-256 slot with lowest logit
            for (int c = 0; c < nc; ++c) {
                const int32_t ctok = s_child_toks[c];
                bool found = false;
                int min_idx = 0;
                for (int i = 0; i < 256; ++i) {
                    if (s_id[i] == ctok) { s_v[i] = s_child_logits[c]; found = true; break; }
                    if (s_v[i] < s_v[min_idx]) min_idx = i;
                }
                if (!found) {
                    s_id[min_idx] = ctok;
                    s_v[min_idx] = s_child_logits[c];
                }
            }

            // Sort descending (partial insertion/selection over 256 items is ~2us on 1 thread)
            const int eff_k = (top_k > 0 && top_k < 256) ? top_k : 256;
            for (int i = 0; i < eff_k; ++i) {
                int best = i;
                for (int j = i + 1; j < 256; ++j) {
                    if (s_v[j] > s_v[best] || (s_v[j] == s_v[best] && s_id[j] < s_id[best])) best = j;
                }
                if (best != i) {
                    float tv = s_v[i]; s_v[i] = s_v[best]; s_v[best] = tv;
                    int32_t ti = s_id[i]; s_id[i] = s_id[best]; s_id[best] = ti;
                }
            }

            // Softmax with temperature over eff_k
            const float mx = s_v[0] * inv_t;
            float sum = 0.0f;
            for (int i = 0; i < eff_k; ++i) {
                const float p = expf(s_v[i] * inv_t - mx);
                s_p[i] = p;
                sum += p;
            }
            const float inv_s = (sum > 1e-30f) ? (1.0f / sum) : 1.0f;
            for (int i = 0; i < eff_k; ++i) s_p[i] *= inv_s;

            // Apply min_p & top_p truncation
            const float mp_cut = (min_p > 0.0f) ? (s_p[0] * min_p) : 0.0f;
            float cum = 0.0f;
            int keep_n = 0;
            for (int i = 0; i < eff_k; ++i) {
                if (i > 0 && mp_cut > 0.0f && s_p[i] < mp_cut) break;
                cum += s_p[i];
                ++keep_n;
                if (top_p > 0.0f && top_p < 1.0f && cum >= top_p) break;
            }
            if (keep_n < 1) keep_n = 1;
            const float renorm = (cum > 1e-30f) ? (1.0f / cum) : 1.0f;
            for (int i = 0; i < keep_n; ++i) s_p[i] *= renorm;

            // Sequential sibling lossless rejection sampling (SpecInfer/Chen)
            int accepted_child = -1;
            int32_t chosen_tok = s_id[0];
            for (int c = 0; c < nc; ++c) {
                const int32_t ctok = s_child_toks[c];
                const int cnode = s_child_ids[c];
                float p_target = 0.0f;
                int p_slot = -1;
                for (int i = 0; i < keep_n; ++i) {
                    if (s_id[i] == ctok) { p_target = s_p[i]; p_slot = i; break; }
                }
                const float q_draft = fmaxf(draft_q[cnode], 1e-6f);
                const float acc_prob = fminf(1.0f, p_target / q_draft);
                const float u = philox_uniform01(rng_seed, rng_offset + static_cast<uint64_t>(step), static_cast<uint32_t>(c + 1));
                if (u < acc_prob) {
                    accepted_child = cnode;
                    chosen_tok = ctok;
                    break;
                }
                // Residual adjustment: p <- norm(max(0, p - q_child)) for rejected sibling
                if (p_slot >= 0) {
                    const float rem = fmaxf(0.0f, s_p[p_slot] - q_draft);
                    const float diff = s_p[p_slot] - rem;
                    s_p[p_slot] = rem;
                    float nsum = 0.0f;
                    for (int i = 0; i < keep_n; ++i) nsum += s_p[i];
                    if (nsum > 1e-30f) {
                        const float ri = 1.0f / nsum;
                        for (int i = 0; i < keep_n; ++i) s_p[i] *= ri;
                    }
                    (void)diff;
                }
            }

            if (accepted_child < 0) {
                // Sample bonus/corrected token from residual target distribution s_p[0..keep_n-1]
                const float u = philox_uniform01(rng_seed, rng_offset + static_cast<uint64_t>(step), 997u);
                float csum = 0.0f;
                chosen_tok = s_id[keep_n - 1];
                for (int i = 0; i < keep_n; ++i) {
                    csum += s_p[i];
                    if (u <= csum) { chosen_tok = s_id[i]; break; }
                }
            }

            const int na = out_wp->n_acc;
            out_wp->idx[na] = cur;
            out_wp->emitted[na] = chosen_tok;
            out_wp->n_acc = na + 1;
            out_wp->next_anchor = chosen_tok;

            if (accepted_child >= 0 && na + 1 < emit_cap) {
                s_next_node = accepted_child;
            } else {
                s_done = 1;
            }
        }
        __syncthreads();
        if (tid == 0 && !s_done) s_cur_node = s_next_node;
        __syncthreads();
    }
}

// =================================================================================================
// WS4+WS5: Prune-Aware Winning-Path Commit Kernels (supports WinPath16 AND Visit-17 orig_node_map!)
// For visits v < prune_visit: reads trec/tree_k/v at orig_node_map[wp.idx[k]]!
// For visits v >= prune_visit: reads trec/tree_k/v at wp.idx[k]!
// =================================================================================================
__global__ void __launch_bounds__(256) k_sconv_commit_tree16_pruned_pair(
    uint16_t* __restrict__ pool_a,
    uint16_t* __restrict__ snap_a,
    uint16_t* __restrict__ rec_a,         // [n_visits, 16, C]
    const uint16_t* __restrict__ trec_a,  // [n_visits, kTreeMaxM, C]
    uint16_t* __restrict__ pool_b,
    uint16_t* __restrict__ snap_b,
    uint16_t* __restrict__ rec_b,         // [n_visits, 16, C]
    const uint16_t* __restrict__ trec_b,  // [n_visits, kTreeMaxM, C]
    const WinPath16* __restrict__ wp_ptr,
    const int32_t* __restrict__ orig_node_map,  // nullptr if not pruned, else [M_survive]
    int prune_visit, int rec_stride, int slot, int max_slots, int n_visits, int C) {
#if defined(__HIP_DEVICE_COMPILE__) && !defined(__gfx1151__)
    return;
#endif

    const int c = blockIdx.x * blockDim.x + threadIdx.x;
    const int v = blockIdx.y;
    if (c >= C || v >= n_visits) return;

    const WinPath16 wp = *wp_ptr;
    const int n_acc = wp.n_acc;
    const size_t pbase = (static_cast<size_t>(v) * max_slots + slot) * 3 * C;
    const size_t sbase = static_cast<size_t>(v) * 3 * C;
    const size_t rbase = static_cast<size_t>(v) * rec_stride * C;
    const size_t tbase = static_cast<size_t>(v) * kTreeMaxM * C;
    const bool use_orig = (orig_node_map != nullptr && v < prune_visit);

    uint16_t old_a[3], old_b[3];
    for (int lag = 0; lag < 3; ++lag) {
        old_a[lag] = pool_a[pbase + lag * C + c];
        old_b[lag] = pool_b[pbase + lag * C + c];
        snap_a[sbase + lag * C + c] = old_a[lag];
        snap_b[sbase + lag * C + c] = old_b[lag];
    }
    for (int k = 0; k < n_acc; ++k) {
        const int wi = wp.idx[k];
        const int nid = use_orig ? orig_node_map[wi] : wi;
        if (k < rec_stride) {
            rec_a[rbase + k * C + c] = trec_a[tbase + nid * C + c];
            rec_b[rbase + k * C + c] = trec_b[tbase + nid * C + c];
        }
    }
    for (int lag = 0; lag < 3; ++lag) {
        const int t = n_acc - 3 + lag;
        if (t < 0) {
            pool_a[pbase + lag * C + c] = old_a[t + 3];
            pool_b[pbase + lag * C + c] = old_b[t + 3];
        } else {
            const int wi = wp.idx[t];
            const int nid = use_orig ? orig_node_map[wi] : wi;
            pool_a[pbase + lag * C + c] = trec_a[tbase + nid * C + c];
            pool_b[pbase + lag * C + c] = trec_b[tbase + nid * C + c];
        }
    }
}

template <bool FP8_KV>
__global__ void __launch_bounds__(256) k3_kv_commit_tree16_pruned_all(
    const KvVisitDesc* __restrict__ visits,
    const half_t* __restrict__ tree_k_all,
    const half_t* __restrict__ tree_v_all,
    const WinPath16* __restrict__ wp_ptr,
    const TreeMetaDev* __restrict__ tmeta,
    const int32_t* __restrict__ orig_node_map,
    int prune_visit, int slot, int n_visits, int elems) {
#if defined(__HIP_DEVICE_COMPILE__) && !defined(__gfx1151__)
    return;
#endif

    const int k = blockIdx.x;
    const int v = blockIdx.y;
    const WinPath16 wp = *wp_ptr;
    if (k >= wp.n_acc || v >= n_visits) return;
    const int tid = threadIdx.x;
    const int base_pos = tmeta->base_pos;

    const KvVisitDesc vd = visits[v];
    const int p = (vd.swa_ring > 0) ? ((base_pos + k) % vd.swa_ring) : (base_pos + k);
    const int phys = vd.d_pt[slot * vd.max_pages_per_slot + (p >> 6)];
    const size_t tok_base = static_cast<size_t>(phys) * 64 + (p & 63);
    const size_t dst_base = tok_base * elems;
    const int wi = wp.idx[k];
    const int nid = (orig_node_map != nullptr && v < prune_visit) ? orig_node_map[wi] : wi;
    const size_t src_base = (static_cast<size_t>(v) * kTreeMaxM + nid) * elems;
    const half_t* ks = tree_k_all + src_base;
    const half_t* vs = tree_v_all + src_base;

    if constexpr (FP8_KV) {
        const int wave = tid >> 5;
        const int lane = tid & 31;
        __shared__ float s_w[8];
        __shared__ float s_ik, s_iv;
        float km = 0.0f, vm = 0.0f;
        for (int i = tid; i < elems; i += 256) {
            km = fmaxf(km, fabsf(static_cast<float>(ks[i])));
            vm = fmaxf(vm, fabsf(static_cast<float>(vs[i])));
        }
        km = wave32_max(km); vm = wave32_max(vm);
        if (lane == 0) s_w[wave] = km;
        __syncthreads();
        if (tid == 0) {
            float mx = 0.0f; for (int w = 0; w < 8; ++w) mx = fmaxf(mx, s_w[w]);
            const float sc = (mx > 1e-12f) ? (mx * (1.0f / 448.0f)) : 1.0f;
            if (vd.k_scales) vd.k_scales[tok_base] = sc;
            s_ik = 1.0f / sc;
        }
        __syncthreads();
        if (lane == 0) s_w[wave] = vm;
        __syncthreads();
        if (tid == 0) {
            float mx = 0.0f; for (int w = 0; w < 8; ++w) mx = fmaxf(mx, s_w[w]);
            const float sc = (mx > 1e-12f) ? (mx * (1.0f / 448.0f)) : 1.0f;
            if (vd.v_scales) vd.v_scales[tok_base] = sc;
            s_iv = 1.0f / sc;
        }
        __syncthreads();
        uint8_t* kd = static_cast<uint8_t*>(vd.k_pool) + dst_base;
        uint8_t* vd_p = static_cast<uint8_t*>(vd.v_pool) + dst_base;
        const float ik = s_ik, iv = s_iv;
        for (int i = tid; i < elems; i += 256) {
            kd[i] = f32_to_e4m3fn(static_cast<float>(ks[i]) * ik);
            vd_p[i] = f32_to_e4m3fn(static_cast<float>(vs[i]) * iv);
        }
    } else {
        half_t* kd = static_cast<half_t*>(vd.k_pool) + dst_base;
        half_t* vd_p = static_cast<half_t*>(vd.v_pool) + dst_base;
        for (int i = tid; i < elems; i += 256) {
            kd[i] = ks[i];
            vd_p[i] = vs[i];
        }
    }
}

__global__ void __launch_bounds__(256) k_gather_tree16_taps4(
    const float* __restrict__ t0, const float* __restrict__ t1,
    const float* __restrict__ t2, const float* __restrict__ t3,
    float* __restrict__ d0, float* __restrict__ d1,
    float* __restrict__ d2, float* __restrict__ d3,
    const WinPath16* __restrict__ wp_ptr, int D) {
#if defined(__HIP_DEVICE_COMPILE__) && !defined(__gfx1151__)
    return;
#endif

    const int col = blockIdx.x * blockDim.x + threadIdx.x;
    const int k = blockIdx.y;
    const WinPath16 wp = *wp_ptr;
    if (col >= D || k >= wp.n_acc) return;
    const int nid = wp.idx[k];
    const size_t so = static_cast<size_t>(nid) * D + col;
    const size_t doff = static_cast<size_t>(k) * D + col;
    d0[doff] = t0[so]; d1[doff] = t1[so]; d2[doff] = t2[so]; d3[doff] = t3[so];
}

// =================================================================================================
// WS5 (Fast Mode Only): Q8_G32 Transcoded Dense Trunk GEMV (-48% Trunk Bytes/Step = -5.8 ms/step!)
// Layout: q8[N, K] int8_t + sc[N, K/32] half_t (1.0625 B/param vs 2.0 B/param BF16!)
// =================================================================================================
struct Q8TrunkMatrix {
    strata::hip_uma::GpuBuffer d_q8;
    strata::hip_uma::GpuBuffer d_sc;
    int N = 0;
    int K = 0;
    bool ready = false;

    void release() noexcept {
        strata::hip_uma::free_gpu_buffer(d_q8);
        strata::hip_uma::free_gpu_buffer(d_sc);
        N = K = 0;
        ready = false;
    }
    ~Q8TrunkMatrix() { release(); }
};

__global__ void __launch_bounds__(256) k_quant_bf16_to_q8_g32(
    const uint16_t* __restrict__ W_bf16,
    int8_t* __restrict__ W_q8,
    half_t* __restrict__ W_sc,
    int N, int K) {
#if defined(__HIP_DEVICE_COMPILE__) && !defined(__gfx1151__)
    return;
#endif

    const int ng = K >> 5;
    const int gidx = blockIdx.x * blockDim.x + threadIdx.x;
    if (gidx >= N * ng) return;
    const int n = gidx / ng;
    const int g = gidx - n * ng;
    const uint16_t* src = W_bf16 + static_cast<size_t>(n) * K + (g << 5);
    float vals[32];
    float amax = 0.0f;
    for (int i = 0; i < 32; ++i) {
        const float f = bf16_to_f32(src[i]);
        vals[i] = f;
        amax = fmaxf(amax, fabsf(f));
    }
    const float sc = (amax > 1e-12f) ? (amax * (1.0f / 127.0f)) : 1.0f;
    const float inv = (amax > 1e-12f) ? (127.0f / amax) : 0.0f;
    W_sc[gidx] = static_cast<half_t>(sc);
    int8_t* dst = W_q8 + static_cast<size_t>(n) * K + (g << 5);
    for (int i = 0; i < 32; ++i) dst[i] = static_cast<int8_t>(clamp_i8(vals[i] * inv));
}

// Pre-quantize X[M, K] -> X_q8[M, K] + X_d[M, K/32] ONCE per matmul (eliminates 960x redundant quants!)
// Grid: ((M * (K/32) + 255) / 256,), Block: 256 (1 thread per 32-element group)
__global__ void __launch_bounds__(256) k_halo_q8_quant_x_g32(
    const float* __restrict__ X,
    int8_t* __restrict__ X_q8,
    float* __restrict__ X_d,
    int M, int K) {
#if defined(__HIP_DEVICE_COMPILE__) && !defined(__gfx1151__)
    return;
#endif

    const int ng = K >> 5;
    const int gi = blockIdx.x * blockDim.x + threadIdx.x;
    if (gi >= M * ng) return;
    const int m = gi / ng;
    const int g = gi - m * ng;
    const float* xp = X + static_cast<size_t>(m) * K + (g << 5);
    float amax = 0.0f;
#pragma unroll
    for (int i = 0; i < 32; ++i) amax = fmaxf(amax, fabsf(xp[i]));
    const float sc = (amax > 1e-12f) ? (amax * (1.0f / 127.0f)) : 1.0f;
    const float inv = (amax > 1e-12f) ? (127.0f / amax) : 0.0f;
    X_d[gi] = sc;
    int8_t* dst = X_q8 + static_cast<size_t>(m) * K + (g << 5);
#pragma unroll
    for (int i = 0; i < 32; ++i) dst[i] = static_cast<int8_t>(clamp_i8(xp[i] * inv));
}

// Fast Wave32 Q8_G32 Multi-Row GEMV for M in 1..16 reading pre-quantized (X_q8, X_d) + hardware sdot4_i32!
// Streams W_q8 at full 230 GB/s with coalesced uint4 loads (16 bytes/lane) + 4x v_dot4_i32_i8!
// Grid: ((N + 7) / 8,), Block: 256 (8 waves -> 8 rows/block)
template <int MT>
__global__ void __launch_bounds__(256) k_q8_g32_gemv_halo(
    const int8_t* __restrict__ W_q8,   // [N, K]
    const half_t* __restrict__ W_sc,   // [N, K/32]
    const int8_t* __restrict__ X_q8,   // [M, K] (pre-quantized once -> L1/L2 cache hit!)
    const float* __restrict__ X_d,     // [M, K/32]
    float* __restrict__ Y,             // [M, N]
    int M, int N, int K) {
#if defined(__HIP_DEVICE_COMPILE__) && !defined(__gfx1151__)
    return;
#endif

    const int tid = threadIdx.x;
    const int wave = tid >> 5;
    const int lane = tid & 31;
    const int row = blockIdx.x * 8 + wave;
    if (row >= N) return;

    const int ng = K >> 5;
    const int8_t* wr = W_q8 + static_cast<size_t>(row) * K;
    const half_t* sr = W_sc + static_cast<size_t>(row) * ng;
    float acc[MT];
#pragma unroll
    for (int m = 0; m < MT; ++m) acc[m] = 0.0f;

    // Each lane handles 16 consecutive int8 weights (4x sdot4_i32!) per step -> 32 lanes * 16 = 512/iter!
    for (int k = lane * 16; k < K; k += 512) {
        const uint4 w16 = *reinterpret_cast<const uint4*>(wr + k);
        const int g = k >> 5;
        const float wsc = static_cast<float>(sr[g]);
#pragma unroll
        for (int m = 0; m < MT; ++m) {
            if (m < M) {
                const uint4 x16 = *reinterpret_cast<const uint4*>(X_q8 + static_cast<size_t>(m) * K + k);
                int dot = sdot4_i32(static_cast<int>(w16.x), static_cast<int>(x16.x), 0);
                dot = sdot4_i32(static_cast<int>(w16.y), static_cast<int>(x16.y), dot);
                dot = sdot4_i32(static_cast<int>(w16.z), static_cast<int>(x16.z), dot);
                dot = sdot4_i32(static_cast<int>(w16.w), static_cast<int>(x16.w), dot);
                acc[m] = fmaf(static_cast<float>(dot), wsc * X_d[static_cast<size_t>(m) * ng + g], acc[m]);
            }
        }
    }
#pragma unroll
    for (int m = 0; m < MT; ++m) {
        if (m < M) {
            const float v = wave32_sum(acc[m]);
            if (lane == 0) Y[static_cast<size_t>(m) * N + row] = v;
        }
    }
}

// =================================================================================================
// WS5 (Opt-In Turbo): Slack-Bounded Pass 1 & 2 Tail Expert Skip (<= 6% cumulative sigmoid mass!)
// Sets skipped expert IDs to -1 (ignored cleanly by k_moe_build_buckets & direct MoE kernels!)
// Grid: (B,), Block: 32
// =================================================================================================
__global__ void __launch_bounds__(32) k_halo_turbo_tail_skip(
    int32_t* __restrict__ topk_ids,  // [B, K_top]
    float* __restrict__ topk_w,      // [B, K_top]
    int B, int K_top, float mass_thresh) {
#if defined(__HIP_DEVICE_COMPILE__) && !defined(__gfx1151__)
    return;
#endif

    const int b = blockIdx.x;
    if (b >= B || threadIdx.x != 0) return;
    int32_t* ids = topk_ids + b * K_top;
    float* ws = topk_w + b * K_top;
    float tot = 0.0f;
    for (int k = 0; k < K_top; ++k) tot += ws[k];
    if (tot <= 1e-12f) return;
    // Find highest-weight expert slot so zeroed tail slots point to an already-active expert
    // (0 extra unique experts in grouped MoE, and 100% OOB-safe in direct MoE!)
    int max_k = 0;
    for (int k = 1; k < K_top; ++k) if (ws[k] > ws[max_k]) max_k = k;
    const int32_t keep_id = ids[max_k];
    // Drop up to 2 smallest-weight experts if their combined share <= mass_thresh (renormalize kept weights)
    float dropped = 0.0f;
    int n_drop = 0;
    for (int pass = 0; pass < 2; ++pass) {
        int min_k = -1;
        float min_w = 1e30f;
        for (int k = 0; k < K_top; ++k) {
            if (k != max_k && ws[k] > 0.0f && ws[k] < min_w) { min_w = ws[k]; min_k = k; }
        }
        if (min_k >= 0 && (dropped + min_w) / tot <= mass_thresh) {
            dropped += min_w;
            ids[min_k] = keep_id;
            ws[min_k] = 0.0f;
            ++n_drop;
        } else {
            break;
        }
    }
    if (n_drop > 0) {
        const float kept = tot - dropped;
        if (kept > 1e-12f) {
            const float scale = tot / kept;
            for (int k = 0; k < K_top; ++k) if (ids[k] >= 0) ws[k] *= scale;
        }
    }
}

// =================================================================================================
// WS8: Batched Multi-Slot DSpark Speculative Decode Kernels (C64 --serve --parallel 64!)
// =================================================================================================
__global__ void __launch_bounds__(256) k_dspark_interleave_masks_halo(
    const float* __restrict__ anchor_emb,  // [n_slots, D]
    const float* __restrict__ mask_emb,    // [D]
    float* __restrict__ out_cur,           // [n_slots * L, D]
    int n_slots, int L, int D) {
#if defined(__HIP_DEVICE_COMPILE__) && !defined(__gfx1151__)
    return;
#endif

    const int idx = blockIdx.x * blockDim.x + threadIdx.x;
    const int tot = n_slots * L * D;
    if (idx >= tot) return;
    const int row = idx / D;
    const int d = idx - row * D;
    const int s = row / L;
    const int off = row - s * L;
    out_cur[idx] = (off == 0) ? anchor_emb[static_cast<size_t>(s) * D + d] : mask_emb[d];
}

__global__ void __launch_bounds__(128) k_dspark_split_qkv_norm_rope_multi_halo(
    const float* __restrict__ qkv,         // [n_slots * L, q_dim + 2*kv_dim]
    const float* __restrict__ q_norm_w,    // [hd]
    const float* __restrict__ k_norm_w,    // [hd]
    const int32_t* __restrict__ pos_arr,   // [n_slots * L]
    float* __restrict__ Q_out,
    float* __restrict__ K_out,
    float* __restrict__ V_out,
    int B, int H, int KV, int hd, float eps, float theta) {
#if defined(__HIP_DEVICE_COMPILE__) && !defined(__gfx1151__)
    return;
#endif

    const int h = blockIdx.x;
    const int b = blockIdx.y;
    const int tid = threadIdx.x;
    if (b >= B || tid >= hd) return;
    const int q_dim = H * hd;
    const int kv_dim = KV * hd;
    const float* row = qkv + static_cast<size_t>(b) * (q_dim + 2 * kv_dim);

    if (h >= H + KV) {
        const int vh = h - H - KV;
        if (vh < KV) V_out[(static_cast<size_t>(b) * KV + vh) * hd + tid] = row[q_dim + kv_dim + vh * hd + tid];
        return;
    }
    const bool is_q = (h < H);
    const float val = is_q ? row[h * hd + tid] : row[q_dim + (h - H) * hd + tid];
    const float nw  = is_q ? q_norm_w[tid] : k_norm_w[tid];
    const int wave = tid >> 5;
    const int lane = tid & 31;
    float ss = wave32_sum(val * val);
    __shared__ float s_w[4];
    __shared__ float s_x[128];
    if (lane == 0) s_w[wave] = ss;
    __syncthreads();
    const int n_waves = (hd + 31) >> 5;
    float tot = 0.0f;
    for (int w = 0; w < n_waves; ++w) tot += s_w[w];
    const float inv = rsqrtf(tot / static_cast<float>(hd) + eps);
    s_x[tid] = val * inv * nw;
    __syncthreads();

    const int half = hd >> 1;
    if (tid < half) {
        const int p = pos_arr[b];
        const float freq = 1.0f / powf(theta, static_cast<float>(2 * tid) / static_cast<float>(hd));
        float s, c;
        __sincosf(static_cast<float>(p) * freq, &s, &c);
        const float a = s_x[tid], bv = s_x[tid + half];
        float* dst = is_q ? (Q_out + (static_cast<size_t>(b) * H + h) * hd)
                          : (K_out + (static_cast<size_t>(b) * KV + (h - H)) * hd);
        dst[tid]        = a * c - bv * s;
        dst[tid + half] = a * s + bv * c;
    }
}

__global__ void __launch_bounds__(128) k_dspark_block_attn_multi_halo(
    const float* __restrict__ Q,                  // [n_slots * L, H, 128]
    uint16_t* const* __restrict__ ctx_K_ptrs,     // [max_slots] -> [cap, KV, 128] BF16
    uint16_t* const* __restrict__ ctx_V_ptrs,     // [max_slots] -> [cap, KV, 128] BF16
    const float* __restrict__ drf_K,              // [n_slots * L, KV, 128]
    const float* __restrict__ drf_V,              // [n_slots * L, KV, 128]
    const int32_t* __restrict__ slot_arr,         // [n_slots]
    const int32_t* __restrict__ anchor_pos_arr,   // [n_slots]
    float* __restrict__ O_out,                    // [n_slots * L, H, 128]
    int n_slots, int L, int H, int KV, int swa_win) {
#if defined(__HIP_DEVICE_COMPILE__) && !defined(__gfx1151__)
    return;
#endif

    const int wave = threadIdx.x >> 5;
    const int lane = threadIdx.x & 31;
    const int hh = blockIdx.x * 4 + wave;
    const int b = blockIdx.y;
    const int B = n_slots * L;
    if (hh >= H || b >= B) return;
    const int si = b / L;
    const int slot = slot_arr[si];
    const int anchor_pos = anchor_pos_arr[si];
    const uint16_t* ctx_K = ctx_K_ptrs[slot];
    const uint16_t* ctx_V = ctx_V_ptrs[slot];
    const int c0 = (swa_win > 0 && anchor_pos >= swa_win) ? (anchor_pos - swa_win + 1) : 0;
    const int c_len = anchor_pos - c0;

    const int d0 = lane * 4;
    const int kvh = hh / (H / KV);
    const float scale = rsqrtf(128.0f);
    const float4 qv = *reinterpret_cast<const float4*>(Q + (static_cast<size_t>(b) * H + hh) * 128 + d0);
    const float q0 = qv.x * scale, q1 = qv.y * scale, q2 = qv.z * scale, q3 = qv.w * scale;

    float max_l = -1e30f, sum_e = 0.0f;
    float o0 = 0.0f, o1 = 0.0f, o2 = 0.0f, o3 = 0.0f;
    const int drf_base = si * L;
    const int tot = c_len + L;
    for (int t = 0; t < tot; ++t) {
        float k0, k1, k2, k3, v0, v1, v2, v3;
        if (t < c_len) {
            const size_t idx = (static_cast<size_t>(c0 + t) * KV + kvh) * 128 + d0;
            const uint2 kw = *reinterpret_cast<const uint2*>(ctx_K + idx);
            const uint2 vw = *reinterpret_cast<const uint2*>(ctx_V + idx);
            k0 = bf16_to_f32(kw.x & 0xFFFFu); k1 = bf16_to_f32(kw.x >> 16);
            k2 = bf16_to_f32(kw.y & 0xFFFFu); k3 = bf16_to_f32(kw.y >> 16);
            v0 = bf16_to_f32(vw.x & 0xFFFFu); v1 = bf16_to_f32(vw.x >> 16);
            v2 = bf16_to_f32(vw.y & 0xFFFFu); v3 = bf16_to_f32(vw.y >> 16);
        } else {
            const int di = drf_base + (t - c_len);
            const size_t idx = (static_cast<size_t>(di) * KV + kvh) * 128 + d0;
            const float4 kw = *reinterpret_cast<const float4*>(drf_K + idx);
            const float4 vw = *reinterpret_cast<const float4*>(drf_V + idx);
            k0 = kw.x; k1 = kw.y; k2 = kw.z; k3 = kw.w;
            v0 = vw.x; v1 = vw.y; v2 = vw.z; v3 = vw.w;
        }
        const float sc = wave32_sum(fmaf(q0, k0, fmaf(q1, k1, fmaf(q2, k2, q3 * k3))));
        const float nm = fmaxf(max_l, sc);
        const float a = __expf(max_l - nm);
        const float eb = __expf(sc - nm);
        o0 = fmaf(o0, a, eb * v0); o1 = fmaf(o1, a, eb * v1);
        o2 = fmaf(o2, a, eb * v2); o3 = fmaf(o3, a, eb * v3);
        sum_e = fmaf(sum_e, a, eb);
        max_l = nm;
    }
    const float inv = 1.0f / fmaxf(sum_e, 1e-30f);
    *reinterpret_cast<float4*>(O_out + (static_cast<size_t>(b) * H + hh) * 128 + d0) =
        make_float4(o0 * inv, o1 * inv, o2 * inv, o3 * inv);
}

__global__ void __launch_bounds__(256) k_sconv_rec_multi_halo(
    const float* __restrict__ in_f32,          // [n_groups * L, C]
    uint16_t* const* __restrict__ rec_ptrs,    // [n_groups]
    int visit, int n_groups, int L, int C) {
#if defined(__HIP_DEVICE_COMPILE__) && !defined(__gfx1151__)
    return;
#endif

    const int col = blockIdx.x * blockDim.x + threadIdx.x;
    const int b = blockIdx.y;
    if (col >= C || b >= n_groups * L) return;
    const int g = b / L;
    const int off = b - g * L;
    uint16_t* dst = rec_ptrs[g];
    if (!dst) return;
    const size_t one_visit = 8u * static_cast<size_t>(C);
    dst[static_cast<size_t>(visit) * one_visit + static_cast<size_t>(off) * C + col] =
        f32_to_bf16(in_f32[static_cast<size_t>(b) * C + col]);
}

__global__ void __launch_bounds__(256) k_logits_argmax_rows_halo(
    const float* __restrict__ logits,  // [N_rows, V]
    int32_t* __restrict__ out_ids,     // [N_rows]
    int N_rows, int V) {
#if defined(__HIP_DEVICE_COMPILE__) && !defined(__gfx1151__)
    return;
#endif

    const int r = blockIdx.x;
    if (r >= N_rows) return;
    const int tid = threadIdx.x;
    const float* row = logits + static_cast<size_t>(r) * V;
    float bv = -1e30f;
    int32_t bi = 0;
    for (int v = tid; v < V; v += 256) {
        const float val = row[v];
        if (val > bv || (val == bv && v < bi)) { bv = val; bi = v; }
    }
    __shared__ float sv[256];
    __shared__ int32_t si[256];
    sv[tid] = bv; si[tid] = bi;
    __syncthreads();
    for (int s = 128; s > 0; s >>= 1) {
        if (tid < s) {
            if (sv[tid + s] > sv[tid] || (sv[tid + s] == sv[tid] && si[tid + s] < si[tid])) {
                sv[tid] = sv[tid + s]; si[tid] = si[tid + s];
            }
        }
        __syncthreads();
    }
    if (tid == 0) out_ids[r] = si[0];
}

// =================================================================================================
// WS7: Prompt Lookup Drafting (PLD) — Zero-Cost When Off, < 5 us Host Suffix Match for Code/Tools!
// =================================================================================================
struct PldSlotHistory {
    std::vector<int32_t> toks;

    void clear() { toks.clear(); }

    void record(const int32_t* ids, int n, int pos0) {
        if (!ids || n <= 0 || pos0 < 0) return;
        const size_t need = static_cast<size_t>(pos0 + n);
        if (toks.size() < need) toks.resize(need, -1);
        std::memcpy(toks.data() + pos0, ids, static_cast<size_t>(n) * sizeof(int32_t));
    }

    void truncate(int keep_len) {
        if (keep_len >= 0 && static_cast<size_t>(keep_len) < toks.size()) {
            toks.resize(static_cast<size_t>(keep_len));
        }
    }

    // Find longest suffix match (n_gram in 4..2) ending at anchor_pos (with toks[anchor_pos] = anchor_tok)
    // and write up to max_cont continuation tokens into out_cont. Returns number of matched tokens (0..max_cont).
    int match_suffix(int anchor_pos, int32_t anchor_tok, int max_cont, int32_t* out_cont) const noexcept {
        if (max_cont <= 0 || anchor_pos < 4 || static_cast<int>(toks.size()) < anchor_pos) return 0;
        const int search_end = anchor_pos - 1;
        const int search_beg = std::max(0, search_end - 8192);
        for (int ng = 4; ng >= 2; --ng) {
            if (anchor_pos < ng) continue;
            // Suffix pattern is toks[anchor_pos - ng + 1 .. anchor_pos - 1] followed by anchor_tok
            for (int p = search_end - 1; p >= search_beg + ng - 1; --p) {
                if (toks[p] != anchor_tok) continue;
                bool ok = true;
                for (int k = 1; k < ng; ++k) {
                    if (toks[p - k] != toks[anchor_pos - k]) { ok = false; break; }
                }
                if (ok) {
                    int got = 0;
                    for (int j = 1; j <= max_cont && p + j < anchor_pos; ++j) {
                        const int32_t t = toks[p + j];
                        if (t < 0) break;
                        out_cont[got++] = t;
                    }
                    if (got > 0) return got;
                }
            }
        }
        return 0;
    }
};

// =================================================================================================
// WS4+WS5 Wave 6: Compact Active-Expert Deduplicated Grouped-MoE GEMV for B in 2..16!
//   - Eliminates 2.5x-3.5x duplicate expert reads from k_moe_direct (B<=6) AND 70% empty WGs / BM=64
//     padding waste from k_moe_grouped_w13_situ_wmma (B=7..16)!
//   - Streams every unique active expert strictly ONCE from LPDDR5X and decodes SF4 nibbles ONCE
//     across all m_e tree nodes sharing that expert.
//   - Uses the EXACT same fdot2_f32 + fmaf + wave32_sum K-reduction order as k_moe_direct_w13_situ
//     and k_moe_direct_w2_par -> 100% BIT-IDENTICAL in EXACT mode AND blazing fast in FAST mode!
// =================================================================================================
__global__ void __launch_bounds__(128) k_moe_build_buckets_compact_halo(
    const int32_t* __restrict__ topk_ids,    // [B, top_k]
    int32_t* __restrict__ exp_off,           // [129]
    int32_t* __restrict__ bucket_tok,        // [B * top_k]
    int32_t* __restrict__ slotk_bidx,        // [B * top_k]
    int32_t* __restrict__ active_eids,       // [129]: [0..n_act-1] = unique eids, [128] = n_act
    int B, int top_k) {
#if defined(__HIP_DEVICE_COMPILE__) && !defined(__gfx1151__)
    return;
#endif

    const int tid = threadIdx.x;
    __shared__ int32_t s_cnt[128];
    __shared__ int32_t s_off[129];
    __shared__ int32_t s_cur[128];
    s_cnt[tid] = 0;
    __syncthreads();

    const int tot = B * top_k;
    for (int i = tid; i < tot; i += 128) {
        const int e = topk_ids[i];
        if (e >= 0 && e < 128) atomicAdd(&s_cnt[e], 1);
    }
    __syncthreads();

    if (tid == 0) {
        int acc = 0;
        int n_act = 0;
        for (int e = 0; e < 128; ++e) {
            s_off[e] = acc;
            s_cur[e] = acc;
            if (s_cnt[e] > 0) active_eids[n_act++] = e;
            acc += s_cnt[e];
        }
        s_off[128] = acc;
        active_eids[128] = n_act;
    }
    __syncthreads();

    exp_off[tid] = s_off[tid];
    if (tid == 0) exp_off[128] = s_off[128];
    __syncthreads();

    if (tid == 0) {
        // Deterministic scatter order matching token index 0..B-1
        for (int i = 0; i < tot; ++i) {
            const int e = topk_ids[i];
            if (e >= 0 && e < 128) {
                const int pos = s_cur[e]++;
                bucket_tok[pos] = i / top_k;
                slotk_bidx[i] = pos;
            } else {
                slotk_bidx[i] = -1;
            }
        }
    }
}

// =================================================================================================
// Wave 7 WDDM Launch-Collapse Fusions (-560+ barrier-serialized micro-launches per decode step!)
// =================================================================================================

// Fusion 1A: Quantize 3 inputs (q_in, k_in, v_in) -> X_q8[3, M, K] + X_d[3, M, K/32] in ONE launch!
// Grid: ((M * (K/32) + 255) / 256, 3), Block: 256
__global__ void __launch_bounds__(256) k_halo_q8_quant_x3_g32(
    const float* __restrict__ X0,
    const float* __restrict__ X1,
    const float* __restrict__ X2,
    int8_t* __restrict__ X_q8,   // stride = M * K per branch
    float* __restrict__ X_d,     // stride = M * (K/32) per branch
    int M, int K) {
#if defined(__HIP_DEVICE_COMPILE__) && !defined(__gfx1151__)
    return;
#endif

    const int br = blockIdx.y;
    const int ng = K >> 5;
    const int gi = blockIdx.x * blockDim.x + threadIdx.x;
    if (br >= 3 || gi >= M * ng) return;
    const float* X = (br == 0) ? X0 : ((br == 1) ? X1 : X2);
    const int m = gi / ng;
    const int g = gi - m * ng;
    const float* xp = X + static_cast<size_t>(m) * K + (g << 5);
    float amax = 0.0f;
#pragma unroll
    for (int i = 0; i < 32; ++i) amax = fmaxf(amax, fabsf(xp[i]));
    const float sc = (amax > 1e-12f) ? (amax * (1.0f / 127.0f)) : 1.0f;
    const float inv = (amax > 1e-12f) ? (127.0f / amax) : 0.0f;
    X_d[static_cast<size_t>(br) * M * ng + gi] = sc;
    int8_t* dst = X_q8 + (static_cast<size_t>(br) * M + m) * K + (g << 5);
#pragma unroll
    for (int i = 0; i < 32; ++i) dst[i] = static_cast<int8_t>(clamp_i8(xp[i] * inv));
}

// Fusion 1B: Single-Launch Q8_G32 QKVRG + Direct Split + Optional FP32 Attn-Sink Proj!
// Eliminates k_split_qkvrg + 2 extra K/V GEMVs + attn_sink_proj = up to 5 launches saved per visit!
// Grid: ((tot_n + n_sink + 7) / 8,), Block: 256
template <int MT, bool THREE_IN>
__global__ void __launch_bounds__(256) k_q8_g32_qkvrg_split_halo(
    const int8_t* __restrict__ W_q8,    // [tot_n, K]
    const half_t* __restrict__ W_sc,    // [tot_n, K/32]
    const int8_t* __restrict__ X_q8,    // [THREE_IN ? 3 : 1, M, K]
    const float* __restrict__ X_d,      // [THREE_IN ? 3 : 1, M, K/32]
    const float* __restrict__ X0_f32,   // [M, K] (for optional FP32 W_sink)
    const float* __restrict__ W_sink,   // nullptr or [n_sink, K] FP32
    float* __restrict__ q_out,
    float* __restrict__ r_out,
    float* __restrict__ g_out,
    float* __restrict__ k_out,
    float* __restrict__ v_out,
    float* __restrict__ sinks_out,
    int M, int q_dim, int r_dim, int kv_dim, int n_sink, int K) {
#if defined(__HIP_DEVICE_COMPILE__) && !defined(__gfx1151__)
    return;
#endif

    const int tid = threadIdx.x;
    const int wave = tid >> 5;
    const int lane = tid & 31;
    const int row = blockIdx.x * 8 + wave;
    const int o_r = q_dim;
    const int o_g = q_dim + r_dim;
    const int o_k = o_g + q_dim;
    const int o_v = o_k + kv_dim;
    const int tot_n = o_v + kv_dim;
    if (row >= tot_n + n_sink) return;

    float acc[MT];
#pragma unroll
    for (int m = 0; m < MT; ++m) acc[m] = 0.0f;

    if (row >= tot_n) {
        // Optional fused FP32 attn_sink_proj row!
        const int sr_idx = row - tot_n;
        const float* ws = W_sink + static_cast<size_t>(sr_idx) * K;
        for (int k = lane * 4; k < K; k += 128) {
            const float4 wv = *reinterpret_cast<const float4*>(ws + k);
#pragma unroll
            for (int m = 0; m < MT; ++m) {
                if (m < M) {
                    const float4 xv = *reinterpret_cast<const float4*>(X0_f32 + static_cast<size_t>(m) * K + k);
                    acc[m] = fmaf(wv.x, xv.x, fmaf(wv.y, xv.y, fmaf(wv.z, xv.z, fmaf(wv.w, xv.w, acc[m]))));
                }
            }
        }
#pragma unroll
        for (int m = 0; m < MT; ++m) {
            if (m < M) {
                const float v = wave32_sum(acc[m]);
                if (lane == 0) sinks_out[static_cast<size_t>(m) * n_sink + sr_idx] = v;
            }
        }
        return;
    }

    const int ng = K >> 5;
    int br = 0;
    if constexpr (THREE_IN) {
        br = (row < o_k) ? 0 : ((row < o_v) ? 1 : 2);
    }
    const int8_t* xb = X_q8 + static_cast<size_t>(br) * M * K;
    const float*  db = X_d  + static_cast<size_t>(br) * M * ng;
    const int8_t* wr = W_q8 + static_cast<size_t>(row) * K;
    const half_t* sr = W_sc + static_cast<size_t>(row) * ng;

    for (int k = lane * 16; k < K; k += 512) {
        const uint4 w16 = *reinterpret_cast<const uint4*>(wr + k);
        const int g = k >> 5;
        const float wsc = static_cast<float>(sr[g]);
#pragma unroll
        for (int m = 0; m < MT; ++m) {
            if (m < M) {
                const uint4 x16 = *reinterpret_cast<const uint4*>(xb + static_cast<size_t>(m) * K + k);
                int dot = sdot4_i32(static_cast<int>(w16.x), static_cast<int>(x16.x), 0);
                dot = sdot4_i32(static_cast<int>(w16.y), static_cast<int>(x16.y), dot);
                dot = sdot4_i32(static_cast<int>(w16.z), static_cast<int>(x16.z), dot);
                dot = sdot4_i32(static_cast<int>(w16.w), static_cast<int>(x16.w), dot);
                acc[m] = fmaf(static_cast<float>(dot), wsc * db[static_cast<size_t>(m) * ng + g], acc[m]);
            }
        }
    }

    float* dst_base = nullptr;
    int dst_stride = 0;
    int dst_col = 0;
    if (row < o_r) {
        dst_base = q_out; dst_stride = q_dim; dst_col = row;
    } else if (row < o_g) {
        dst_base = r_out; dst_stride = r_dim; dst_col = row - o_r;
    } else if (row < o_k) {
        dst_base = g_out; dst_stride = q_dim; dst_col = row - o_g;
    } else if (row < o_v) {
        dst_base = k_out; dst_stride = kv_dim; dst_col = row - o_k;
    } else {
        dst_base = v_out; dst_stride = kv_dim; dst_col = row - o_v;
    }
#pragma unroll
    for (int m = 0; m < MT; ++m) {
        if (m < M) {
            const float v = wave32_sum(acc[m]);
            if (lane == 0) dst_base[static_cast<size_t>(m) * dst_stride + dst_col] = v;
        }
    }
}

// Fusion 2: Bit-Exact Fused ShortConv4 K+V Pair for Tree Verify (1 launch instead of 2 -> -48/step!)
// Grid: ((kv_dim + 255) / 256, M, 2), Block: 256
__global__ void __launch_bounds__(256) k2_shortconv4_kv_pair_tree_halo(
    const float* __restrict__ k_cur, const float* __restrict__ v_cur,
    const float* __restrict__ k_sw,  const float* __restrict__ v_sw,
    const float* __restrict__ k_dyn, const float* __restrict__ v_dyn,
    const TreeMetaDev* __restrict__ tmeta,
    const uint16_t* __restrict__ k_hist, const uint16_t* __restrict__ v_hist,
    uint16_t* __restrict__ k_rec, uint16_t* __restrict__ v_rec,
    float* __restrict__ k_out, float* __restrict__ v_out,
    int M, int C, int n_heads) {
#pragma clang fp contract(off)
#if defined(__HIP_DEVICE_COMPILE__) && !defined(__gfx1151__)
    return;
#endif
    const int c = blockIdx.x * blockDim.x + threadIdx.x;
    const int m = blockIdx.y;
    const int is_v = blockIdx.z;
    if (c >= C || m >= M) return;
    const float* cur = is_v ? v_cur : k_cur;
    const float* static_w = is_v ? v_sw : k_sw;
    const float* dyn_taps = is_v ? v_dyn : k_dyn;
    const uint16_t* hist_pool = is_v ? v_hist : k_hist;
    uint16_t* rec_out = is_v ? v_rec : k_rec;
    float* out = is_v ? v_out : k_out;

    const int slot = tmeta->slot;
    const uint16_t* h_slot = hist_pool + static_cast<size_t>(slot) * 3 * C;
    const int a1 = tmeta->anc1[m];
    const int a2 = tmeta->anc2[m];
    const int a3 = tmeta->anc3[m];

    const float x0 = cur[static_cast<size_t>(m) * C + c];
    rec_out[static_cast<size_t>(m) * C + c] = f32_to_bf16(x0);

    const float h2 = (a1 >= 0) ? bf16_to_f32(f32_to_bf16(cur[static_cast<size_t>(a1) * C + c]))
                               : bf16_to_f32(h_slot[2 * C + c]);
    const float h1 = (a2 >= 0) ? bf16_to_f32(f32_to_bf16(cur[static_cast<size_t>(a2) * C + c]))
                               : bf16_to_f32(h_slot[(3 + a2) * C + c]);
    const float h0 = (a3 >= 0) ? bf16_to_f32(f32_to_bf16(cur[static_cast<size_t>(a3) * C + c]))
                               : bf16_to_f32(h_slot[(3 + a3) * C + c]);

    const float4 sw = *reinterpret_cast<const float4*>(static_w + static_cast<size_t>(c) * 4);
    float y = x0 + sw.x * h0 + sw.y * h1 + sw.z * h2 + sw.w * x0;
    if (dyn_taps != nullptr) {
        const int head = c / (C / n_heads);
        const float4 dt = *reinterpret_cast<const float4*>(dyn_taps + (static_cast<size_t>(m) * n_heads + head) * 4);
        y += dt.x * x0 + dt.y * h2 + dt.z * h1 + dt.w * h0;
        y *= (1.0f / 3.0f);
    } else {
        y *= 0.44721359549995793f;
    }
    out[static_cast<size_t>(m) * C + c] = y;
}

// Fusion 3: Bit-Exact ShortConv4 + Residual Add OR Block-Boundary Save+Overwrite (-104 launches/step!)
// If save_block_dst != nullptr: save_block_dst[idx] = cur_state[idx]; cur_state[idx] = y;
// Else: cur_state[idx] = cur_state[idx] + y;
// Grid: ((D + 255) / 256, M), Block: 256
__global__ void __launch_bounds__(256) k2_shortconv4_tree_post_halo(
    const float* __restrict__ in_proj,      // [M, D] (oproj or moeout)
    const float* __restrict__ static_w,     // [D, 4]
    const TreeMetaDev* __restrict__ tmeta,
    const uint16_t* __restrict__ hist_pool, // [max_slots, 3, D]
    uint16_t* __restrict__ rec_visit_out,   // [kTreeMaxM, D]
    float* __restrict__ cur_state,          // [M, D] (updated in-place!)
    float* __restrict__ save_block_dst,     // nullptr (add mode) or sc_blocks_[n_blocks] (block start!)
    int M, int D) {
#pragma clang fp contract(off)
#if defined(__HIP_DEVICE_COMPILE__) && !defined(__gfx1151__)
    return;
#endif
    const int c = blockIdx.x * blockDim.x + threadIdx.x;
    const int m = blockIdx.y;
    if (c >= D || m >= M) return;

    const int slot = tmeta->slot;
    const uint16_t* h_slot = hist_pool + static_cast<size_t>(slot) * 3 * D;
    const int a1 = tmeta->anc1[m];
    const int a2 = tmeta->anc2[m];
    const int a3 = tmeta->anc3[m];
    const size_t idx = static_cast<size_t>(m) * D + c;

    const float x0 = in_proj[idx];
    rec_visit_out[idx] = f32_to_bf16(x0);

    const float h2 = (a1 >= 0) ? bf16_to_f32(f32_to_bf16(in_proj[static_cast<size_t>(a1) * D + c]))
                               : bf16_to_f32(h_slot[2 * D + c]);
    const float h1 = (a2 >= 0) ? bf16_to_f32(f32_to_bf16(in_proj[static_cast<size_t>(a2) * D + c]))
                               : bf16_to_f32(h_slot[(3 + a2) * D + c]);
    const float h0 = (a3 >= 0) ? bf16_to_f32(f32_to_bf16(in_proj[static_cast<size_t>(a3) * D + c]))
                               : bf16_to_f32(h_slot[(3 + a3) * D + c]);

    const float4 sw = *reinterpret_cast<const float4*>(static_w + static_cast<size_t>(c) * 4);
    const float y = (x0 + sw.x * h0 + sw.y * h1 + sw.z * h2 + sw.w * x0) * 0.44721359549995793f;

    if (save_block_dst != nullptr) {
        save_block_dst[idx] = cur_state[idx];
        cur_state[idx] = y;
    } else {
        cur_state[idx] = cur_state[idx] + y;
    }
}

// Fusion 4: Single-Launch Loop Controller (RMS + Down + Up) + Post-RMSNorm! (-144 launches/step!)
// Collapses k1_loop_ctrl_rms + down + up + k_rms_norm_rows (4 launches -> 1!) for both 3-branch
// attention (grid=(3, M)) and 1-branch MLP (grid=(1, M))! Uses 18.7 KB LDS -> 2 WG/CU!
template <bool BF16_W>
__global__ void __launch_bounds__(256) k1_loop_ctrl_fused_norm_halo(
    FPtr3 src_ptrs,
    const float* __restrict__ msg_base,  // [M, 4, D]
    int msg_ch_offset,                   // 0 for attn/k/v (ch = br), 3 for mlp (ch = 3)
    Ptr3 w_down, Ptr3 w_gain, Ptr3 w_forget,
    FPtr3 norm_w,
    MutFPtr3 dst_ptrs,
    int M, int D, int rank, float rms_eps) {
#if defined(__HIP_DEVICE_COMPILE__) && !defined(__gfx1151__)
    return;
#endif

    const int br = blockIdx.x;
    const int m = blockIdx.y;
    if (m >= M || br >= 3) return;
    const int tid = threadIdx.x;
    const int wave = tid >> 5;
    const int lane = tid & 31;
    const int msg_ch = msg_ch_offset + br;

    __shared__ alignas(16) float s_c[2304];  // 9,216 B
    __shared__ alignas(16) float s_m[2304];  // 9,216 B
    __shared__ float s_gt[64];               //   256 B
    __shared__ float s_w[8];                 //    32 B
    __shared__ float s_inv[3];               //    12 B -> 18.7 KB total

    const float* c_row = src_ptrs.p[br] + static_cast<size_t>(m) * D;
    const float* m_row = msg_base + (static_cast<size_t>(m) * 4 + msg_ch) * D;
    float ssc = 0.0f, ssm = 0.0f;
    for (int d = tid; d < D; d += 256) {
        const float cv = c_row[d];
        const float mv = m_row[d];
        s_c[d] = cv;
        s_m[d] = mv;
        ssc = fmaf(cv, cv, ssc);
        ssm = fmaf(mv, mv, ssm);
    }
    ssc = wave32_sum(ssc);
    if (lane == 0) s_w[wave] = ssc;
    __syncthreads();
    if (tid == 0) {
        float t = 0.0f;
        for (int w = 0; w < 8; ++w) t += s_w[w];
        s_inv[0] = rsqrtf(t / static_cast<float>(D) + kRmsEps);
    }
    __syncthreads();
    ssm = wave32_sum(ssm);
    if (lane == 0) s_w[wave] = ssm;
    __syncthreads();
    if (tid == 0) {
        float t = 0.0f;
        for (int w = 0; w < 8; ++w) t += s_w[w];
        s_inv[1] = rsqrtf(t / static_cast<float>(D) + kRmsEps);
    }
    __syncthreads();

    const float inv_c = s_inv[0];
    const float inv_m = s_inv[1];
    // 8 waves handle rank (<=64) rows -> 8 ranks/wave, each wave dots 2*D across 32 lanes!
    const void* wd_p = w_down.p[br];
    for (int r = wave; r < rank; r += 8) {
        float dc = 0.0f, dm = 0.0f;
        constexpr int kB = 6;  // Pattern a: 6x vector loads (24 elems) in flight, bit-exact order!
        if constexpr (BF16_W) {
            const uint16_t* wr = static_cast<const uint16_t*>(wd_p) + static_cast<size_t>(r) * (2 * D);
            int d = lane * 4;
            for (; d + 128 * (kB - 1) < D; d += 128 * kB) {
                uint2 wc[kB], wm[kB];
                float4 xc[kB], xm[kB];
#pragma unroll
                for (int b = 0; b < kB; ++b) {
                    const int dd = d + 128 * b;
                    wc[b] = *reinterpret_cast<const uint2*>(wr + dd);
                    wm[b] = *reinterpret_cast<const uint2*>(wr + D + dd);
                    xc[b] = *reinterpret_cast<const float4*>(s_c + dd);
                    xm[b] = *reinterpret_cast<const float4*>(s_m + dd);
                }
#pragma unroll
                for (int b = 0; b < kB; ++b) {
                    dc = fmaf(bf16_to_f32(wc[b].x & 0xFFFFu), xc[b].x, fmaf(bf16_to_f32(wc[b].x >> 16), xc[b].y,
                         fmaf(bf16_to_f32(wc[b].y & 0xFFFFu), xc[b].z, fmaf(bf16_to_f32(wc[b].y >> 16), xc[b].w, dc))));
                    dm = fmaf(bf16_to_f32(wm[b].x & 0xFFFFu), xm[b].x, fmaf(bf16_to_f32(wm[b].x >> 16), xm[b].y,
                         fmaf(bf16_to_f32(wm[b].y & 0xFFFFu), xm[b].z, fmaf(bf16_to_f32(wm[b].y >> 16), xm[b].w, dm))));
                }
            }
            for (; d < D; d += 128) {
                const uint2 wc = *reinterpret_cast<const uint2*>(wr + d);
                const uint2 wm = *reinterpret_cast<const uint2*>(wr + D + d);
                const float4 xc = *reinterpret_cast<const float4*>(s_c + d);
                const float4 xm = *reinterpret_cast<const float4*>(s_m + d);
                dc = fmaf(bf16_to_f32(wc.x & 0xFFFFu), xc.x, fmaf(bf16_to_f32(wc.x >> 16), xc.y,
                     fmaf(bf16_to_f32(wc.y & 0xFFFFu), xc.z, fmaf(bf16_to_f32(wc.y >> 16), xc.w, dc))));
                dm = fmaf(bf16_to_f32(wm.x & 0xFFFFu), xm.x, fmaf(bf16_to_f32(wm.x >> 16), xm.y,
                     fmaf(bf16_to_f32(wm.y & 0xFFFFu), xm.z, fmaf(bf16_to_f32(wm.y >> 16), xm.w, dm))));
            }
        } else {
            const float* wr = static_cast<const float*>(wd_p) + static_cast<size_t>(r) * (2 * D);
            int d = lane * 4;
            for (; d + 128 * (kB - 1) < D; d += 128 * kB) {
                float4 wc[kB], wm[kB], xc[kB], xm[kB];
#pragma unroll
                for (int b = 0; b < kB; ++b) {
                    const int dd = d + 128 * b;
                    wc[b] = *reinterpret_cast<const float4*>(wr + dd);
                    wm[b] = *reinterpret_cast<const float4*>(wr + D + dd);
                    xc[b] = *reinterpret_cast<const float4*>(s_c + dd);
                    xm[b] = *reinterpret_cast<const float4*>(s_m + dd);
                }
#pragma unroll
                for (int b = 0; b < kB; ++b) {
                    dc = fmaf(wc[b].x, xc[b].x, fmaf(wc[b].y, xc[b].y, fmaf(wc[b].z, xc[b].z, fmaf(wc[b].w, xc[b].w, dc))));
                    dm = fmaf(wm[b].x, xm[b].x, fmaf(wm[b].y, xm[b].y, fmaf(wm[b].z, xm[b].z, fmaf(wm[b].w, xm[b].w, dm))));
                }
            }
            for (; d < D; d += 128) {
                const float4 wc = *reinterpret_cast<const float4*>(wr + d);
                const float4 wm = *reinterpret_cast<const float4*>(wr + D + d);
                const float4 xc = *reinterpret_cast<const float4*>(s_c + d);
                const float4 xm = *reinterpret_cast<const float4*>(s_m + d);
                dc = fmaf(wc.x, xc.x, fmaf(wc.y, xc.y, fmaf(wc.z, xc.z, fmaf(wc.w, xc.w, dc))));
                dm = fmaf(wm.x, xm.x, fmaf(wm.y, xm.y, fmaf(wm.z, xm.z, fmaf(wm.w, xm.w, dm))));
            }
        }
        const float gt = wave32_sum(dc * inv_c + dm * inv_m);
        if (lane == 0) s_gt[r] = gt;
    }
    __syncthreads();

    const void* wg_p = w_gain.p[br];
    const void* wf_p = w_forget.p[br];
    float sso = 0.0f;
    for (int d = tid; d < D; d += 256) {
        float ga = 0.0f, fa = 0.0f;
        if constexpr (BF16_W) {
            const uint16_t* wg = static_cast<const uint16_t*>(wg_p) + static_cast<size_t>(d) * rank;
            const uint16_t* wf = static_cast<const uint16_t*>(wf_p) + static_cast<size_t>(d) * rank;
            for (int r = 0; r < rank; r += 4) {
                const uint2 g4 = *reinterpret_cast<const uint2*>(wg + r);
                const uint2 f4 = *reinterpret_cast<const uint2*>(wf + r);
                ga = fmaf(bf16_to_f32(g4.x & 0xFFFFu), s_gt[r + 0], fmaf(bf16_to_f32(g4.x >> 16), s_gt[r + 1],
                     fmaf(bf16_to_f32(g4.y & 0xFFFFu), s_gt[r + 2], fmaf(bf16_to_f32(g4.y >> 16), s_gt[r + 3], ga))));
                fa = fmaf(bf16_to_f32(f4.x & 0xFFFFu), s_gt[r + 0], fmaf(bf16_to_f32(f4.x >> 16), s_gt[r + 1],
                     fmaf(bf16_to_f32(f4.y & 0xFFFFu), s_gt[r + 2], fmaf(bf16_to_f32(f4.y >> 16), s_gt[r + 3], fa))));
            }
        } else {
            const float* wg = static_cast<const float*>(wg_p) + static_cast<size_t>(d) * rank;
            const float* wf = static_cast<const float*>(wf_p) + static_cast<size_t>(d) * rank;
            for (int r = 0; r < rank; ++r) {
                const float gr = s_gt[r];
                ga = fmaf(wg[r], gr, ga);
                fa = fmaf(wf[r], gr, fa);
            }
        }
        const float out_v = fmaf(1.0f - tanhf(fa), s_c[d], tanhf(ga) * s_m[d]);
        s_c[d] = out_v;
        sso = fmaf(out_v, out_v, sso);
    }
    sso = wave32_sum(sso);
    if (lane == 0) s_w[wave] = sso;
    __syncthreads();
    if (tid == 0) {
        float t = 0.0f;
        for (int w = 0; w < 8; ++w) t += s_w[w];
        s_inv[2] = rsqrtf(t / static_cast<float>(D) + rms_eps);
    }
    __syncthreads();

    const float inv_o = s_inv[2];
    const float* nw = norm_w.p[br];
    float* dst = dst_ptrs.p[br] + static_cast<size_t>(m) * D;
    for (int d = tid; d < D; d += 256) {
        dst[d] = s_c[d] * inv_o * nw[d];
    }
}

// Fusion 5: Single-Launch Router Gate GEMV (128 experts) + F16 Cast + Top-K Selection! (-48 launches/step!)
// Grid: (M,), Block: 256 (8 waves -> 16 experts/wave, s_x[2304] in 9.2 KB LDS!)
template <bool BF16_W>
__global__ void __launch_bounds__(256) k_halo_router_topk_f16_fused(
    const void* __restrict__ w_gate,    // [128, D] BF16 or FP32
    const float* __restrict__ bias,     // nullptr or [128]
    const float* __restrict__ mlpin,    // [M, D]
    half_t* __restrict__ mlpf16,        // [M, D]
    int32_t* __restrict__ topk_ids,     // [M, K_top]
    float* __restrict__ topk_weights,   // [M, K_top]
    int M, int D, int K_top, float route_scale) {
#if defined(__HIP_DEVICE_COMPILE__) && !defined(__gfx1151__)
    return;
#endif

    const int m = blockIdx.x;
    if (m >= M) return;
    const int tid = threadIdx.x;
    const int wave = tid >> 5;
    const int lane = tid & 31;

    __shared__ alignas(16) float s_x[2304];  // 9,216 B
    __shared__ float s_prob[128];
    __shared__ float s_sc[128];
    __shared__ float s_wmax[4];
    __shared__ int32_t s_wid[4];
    __shared__ int32_t s_chosen[8];

    const int d4 = D >> 2;
    const float4* src4 = reinterpret_cast<const float4*>(mlpin + static_cast<size_t>(m) * D);
    float4* sx4 = reinterpret_cast<float4*>(s_x);
    uint2* dst2 = reinterpret_cast<uint2*>(mlpf16 + static_cast<size_t>(m) * D);
    for (int i = tid; i < d4; i += 256) {
        const float4 v = src4[i];
        sx4[i] = v;
        dst2[i] = make_uint2(h2_pack(v.x, v.y), h2_pack(v.z, v.w));
    }
    __syncthreads();

    for (int e = wave; e < 128; e += 8) {
        float acc = 0.0f;
        constexpr int kB = 6;  // Pattern a: 6x vector loads (24 elems) in flight, bit-exact order!
        if constexpr (BF16_W) {
            const uint16_t* wr = static_cast<const uint16_t*>(w_gate) + static_cast<size_t>(e) * D;
            int d = lane * 4;
            for (; d + 128 * (kB - 1) < D; d += 128 * kB) {
                uint2 wv[kB];
                float4 xv[kB];
#pragma unroll
                for (int b = 0; b < kB; ++b) {
                    const int dd = d + 128 * b;
                    wv[b] = *reinterpret_cast<const uint2*>(wr + dd);
                    xv[b] = *reinterpret_cast<const float4*>(s_x + dd);
                }
#pragma unroll
                for (int b = 0; b < kB; ++b) {
                    acc = fmaf(bf16_to_f32(wv[b].x & 0xFFFFu), xv[b].x, fmaf(bf16_to_f32(wv[b].x >> 16), xv[b].y,
                          fmaf(bf16_to_f32(wv[b].y & 0xFFFFu), xv[b].z, fmaf(bf16_to_f32(wv[b].y >> 16), xv[b].w, acc))));
                }
            }
            for (; d < D; d += 128) {
                const uint2 w4 = *reinterpret_cast<const uint2*>(wr + d);
                const float4 xv = *reinterpret_cast<const float4*>(s_x + d);
                acc = fmaf(bf16_to_f32(w4.x & 0xFFFFu), xv.x, fmaf(bf16_to_f32(w4.x >> 16), xv.y,
                      fmaf(bf16_to_f32(w4.y & 0xFFFFu), xv.z, fmaf(bf16_to_f32(w4.y >> 16), xv.w, acc))));
            }
        } else {
            const float* wr = static_cast<const float*>(w_gate) + static_cast<size_t>(e) * D;
            int d = lane * 4;
            for (; d + 128 * (kB - 1) < D; d += 128 * kB) {
                float4 wv[kB], xv[kB];
#pragma unroll
                for (int b = 0; b < kB; ++b) {
                    const int dd = d + 128 * b;
                    wv[b] = *reinterpret_cast<const float4*>(wr + dd);
                    xv[b] = *reinterpret_cast<const float4*>(s_x + dd);
                }
#pragma unroll
                for (int b = 0; b < kB; ++b) {
                    acc = fmaf(wv[b].x, xv[b].x, fmaf(wv[b].y, xv[b].y, fmaf(wv[b].z, xv[b].z, fmaf(wv[b].w, xv[b].w, acc))));
                }
            }
            for (; d < D; d += 128) {
                const float4 wv = *reinterpret_cast<const float4*>(wr + d);
                const float4 xv = *reinterpret_cast<const float4*>(s_x + d);
                acc = fmaf(wv.x, xv.x, fmaf(wv.y, xv.y, fmaf(wv.z, xv.z, fmaf(wv.w, xv.w, acc))));
            }
        }
        const float lg = wave32_sum(acc);
        if (lane == 0) {
            float p = 1.0f / (1.0f + __expf(-lg));
            if (!(p == p)) p = 0.0f;
            s_prob[e] = p;
            float sc = p + (bias ? bias[e] : 0.0f);
            if (!(sc == sc)) sc = -1e29f;
            s_sc[e] = sc;
        }
    }
    __syncthreads();

    float my_sc = (tid < 128) ? s_sc[tid] : -1e30f;
    for (int k = 0; k < K_top; ++k) {
        if (tid < 128) {
            float bv = my_sc;
            int32_t bi = tid;
#pragma unroll
            for (int off = 16; off > 0; off >>= 1) {
                const float ov = __shfl_xor(bv, off, 32);
                const int32_t oi = __shfl_xor(bi, off, 32);
                if (ov > bv || (ov == bv && oi < bi)) { bv = ov; bi = oi; }
            }
            if (lane == 0) { s_wmax[wave] = bv; s_wid[wave] = bi; }
        }
        __syncthreads();
        if (tid == 0) {
            float gv = s_wmax[0]; int32_t gi = s_wid[0];
            for (int w = 1; w < 4; ++w) {
                if (s_wmax[w] > gv || (s_wmax[w] == gv && s_wid[w] < gi)) { gv = s_wmax[w]; gi = s_wid[w]; }
            }
            s_chosen[k] = gi;
        }
        __syncthreads();
        if (tid == s_chosen[k]) my_sc = -1e30f;
    }
    if (tid == 0) {
        float wsum = 0.0f;
        for (int k = 0; k < K_top; ++k) {
            const int best_e = s_chosen[k];
            topk_ids[m * K_top + k] = best_e;
            const float w = s_prob[best_e];
            topk_weights[m * K_top + k] = w;
            wsum += w;
        }
        const float inv_w = route_scale / (wsum + 1e-20f);
        for (int k = 0; k < K_top; ++k) topk_weights[m * K_top + k] *= inv_w;
    }
}

// Compile-time unrolled zero-spill SF4 grouped-GEMV inner loop (0 bytes of scratch/private memory spill!)
template <int MT>
__device__ __forceinline__ void halo_sf4_grouped_dot_u(
    const uint8_t* __restrict__ pr,
    const uint8_t* __restrict__ sr,
    const uint32_t* __restrict__ s_sf4_pair,
    const float* __restrict__ s_sf4_sc,
    const half_t* const (&xptrs)[MT],
    float (&acc)[MT],
    int nblk, float outer, int lane) {
#pragma unroll
    for (int mi = 0; mi < MT; ++mi) acc[mi] = 0.0f;
    for (int blk = lane; blk < nblk; blk += 32) {
        const float sc = s_sf4_sc[sr[blk]] * outer;
        const uint2 raw = *reinterpret_cast<const uint2*>(pr + blk * 8);
        const uint32_t w0 = s_sf4_pair[raw.x & 0xFFu];
        const uint32_t w1 = s_sf4_pair[(raw.x >> 8) & 0xFFu];
        const uint32_t w2 = s_sf4_pair[(raw.x >> 16) & 0xFFu];
        const uint32_t w3 = s_sf4_pair[(raw.x >> 24) & 0xFFu];
        const uint32_t w4 = s_sf4_pair[raw.y & 0xFFu];
        const uint32_t w5 = s_sf4_pair[(raw.y >> 8) & 0xFFu];
        const uint32_t w6 = s_sf4_pair[(raw.y >> 16) & 0xFFu];
        const uint32_t w7 = s_sf4_pair[(raw.y >> 24) & 0xFFu];
        const int col = blk << 4;
#pragma unroll
        for (int mi = 0; mi < MT; ++mi) {
            const half_t* xsrc = xptrs[mi] + col;
            const uint4 x0 = *reinterpret_cast<const uint4*>(xsrc);
            const uint4 x1 = *reinterpret_cast<const uint4*>(xsrc + 8);
            float d = fdot2_f32(w0, x0.x, 0.0f);
            d = fdot2_f32(w1, x0.y, d);
            d = fdot2_f32(w2, x0.z, d);
            d = fdot2_f32(w3, x0.w, d);
            d = fdot2_f32(w4, x1.x, d);
            d = fdot2_f32(w5, x1.y, d);
            d = fdot2_f32(w6, x1.z, d);
            d = fdot2_f32(w7, x1.w, d);
            acc[mi] = fmaf(d, sc, acc[mi]);
        }
    }
}

// Grid: ((inter + 3) / 4, max_unique), Block: 256 (8 waves -> 4 inter indices/block)
__global__ void __launch_bounds__(256) k_halo_moe_grouped_gemv_w13_situ(
    const GpuExpertDesc* __restrict__ exps,     // [128]
    const int32_t* __restrict__ exp_off,        // [129]
    const int32_t* __restrict__ bucket_tok,     // [B * top_k]
    const int32_t* __restrict__ active_eids,    // [129]: [128] = n_active
    const half_t* __restrict__ X_f16,           // [B, D]
    half_t* __restrict__ bucket_act,            // [B * top_k, inter]
    int D, int inter, float beta, float lb) {
#if defined(__HIP_DEVICE_COMPILE__) && !defined(__gfx1151__)
    return;
#endif

    const int n_act = active_eids[128];
    if (static_cast<int>(blockIdx.y) >= n_act) return;
    const int eid = active_eids[blockIdx.y];
    const int beg = exp_off[eid];
    const int m_e = exp_off[eid + 1] - beg;
    if (m_e <= 0 || m_e > 16) return;
    const GpuExpertDesc desc = exps[eid];
    if (!desc.body) return;

    const int tid = threadIdx.x;
    const int wave = tid >> 5;
    const int lane = tid & 31;
    const int j = blockIdx.x * 4 + (wave >> 1);
    const int row = (j << 1) | (wave & 1);

    __shared__ int32_t sTok[16];
    __shared__ half_t s_e4m3[256];
    __shared__ uint32_t s_sf4_pair[256];
    __shared__ float s_sf4_sc[256];
    __shared__ float s_gu[16][8];

    if (tid < m_e) sTok[tid] = bucket_tok[beg + tid];
    if (desc.gu_code == CODE_MX8) {
        s_e4m3[tid] = static_cast<half_t>(e4m3fn_to_f32(static_cast<uint8_t>(tid)));
    } else if (desc.gu_code == CODE_SF4) {
        const hv2 hp = {sf4_e2m1_f16(static_cast<uint32_t>(tid) & 15u), sf4_e2m1_f16(static_cast<uint32_t>(tid) >> 4)};
        s_sf4_pair[tid] = __builtin_bit_cast(uint32_t, hp);
        s_sf4_sc[tid] = e4m3fn_to_f32(static_cast<uint8_t>(tid));
    }
    __syncthreads();
    if (j >= inter) return;

    if (desc.gu_code == CODE_SF4) {
        const float* hdr = reinterpret_cast<const float*>(desc.body);
        const float outer = hdr[0] / hdr[1];
        const uint8_t* packed = desc.body + 16;
        const uint8_t* scales = packed + static_cast<size_t>(2 * inter) * (D >> 1);
        const uint8_t* pr = packed + static_cast<size_t>(row) * (D >> 1);
        const uint8_t* sr = scales + static_cast<size_t>(row) * (D >> 4);
        const int nblk = D >> 4;

        auto finish_chunk = [&](int base_mi, int cnt, const float* vals) {
            for (int i = 0; i < cnt; ++i) {
                const float v = wave32_sum(vals[i]);
                if (lane == 0) s_gu[base_mi + i][wave] = v;
            }
        };

        if (m_e == 1) {
            const half_t* xp[1] = {X_f16 + static_cast<size_t>(sTok[0]) * D};
            float a[1];
            halo_sf4_grouped_dot_u<1>(pr, sr, s_sf4_pair, s_sf4_sc, xp, a, nblk, outer, lane);
            finish_chunk(0, 1, a);
        } else if (m_e == 2) {
            const half_t* xp[2] = {
                X_f16 + static_cast<size_t>(sTok[0]) * D,
                X_f16 + static_cast<size_t>(sTok[1]) * D};
            float a[2];
            halo_sf4_grouped_dot_u<2>(pr, sr, s_sf4_pair, s_sf4_sc, xp, a, nblk, outer, lane);
            finish_chunk(0, 2, a);
        } else if (m_e <= 4) {
            const half_t* xp[4] = {
                X_f16 + static_cast<size_t>(sTok[0]) * D,
                X_f16 + static_cast<size_t>(sTok[1]) * D,
                X_f16 + static_cast<size_t>(sTok[(m_e > 2) ? 2 : 0]) * D,
                X_f16 + static_cast<size_t>(sTok[(m_e > 3) ? 3 : 0]) * D};
            float a[4];
            halo_sf4_grouped_dot_u<4>(pr, sr, s_sf4_pair, s_sf4_sc, xp, a, nblk, outer, lane);
            finish_chunk(0, m_e, a);
        } else if (m_e <= 8) {
            const half_t* xp[8] = {
                X_f16 + static_cast<size_t>(sTok[0]) * D,
                X_f16 + static_cast<size_t>(sTok[1]) * D,
                X_f16 + static_cast<size_t>(sTok[2]) * D,
                X_f16 + static_cast<size_t>(sTok[3]) * D,
                X_f16 + static_cast<size_t>(sTok[4]) * D,
                X_f16 + static_cast<size_t>(sTok[(m_e > 5) ? 5 : 0]) * D,
                X_f16 + static_cast<size_t>(sTok[(m_e > 6) ? 6 : 0]) * D,
                X_f16 + static_cast<size_t>(sTok[(m_e > 7) ? 7 : 0]) * D};
            float a[8];
            halo_sf4_grouped_dot_u<8>(pr, sr, s_sf4_pair, s_sf4_sc, xp, a, nblk, outer, lane);
            finish_chunk(0, m_e, a);
        } else {
            for (int b0 = 0; b0 < m_e; b0 += 8) {
                const int cnt = (m_e - b0 < 8) ? (m_e - b0) : 8;
                const half_t* xp[8] = {
                    X_f16 + static_cast<size_t>(sTok[b0 + 0]) * D,
                    X_f16 + static_cast<size_t>(sTok[b0 + ((cnt > 1) ? 1 : 0)]) * D,
                    X_f16 + static_cast<size_t>(sTok[b0 + ((cnt > 2) ? 2 : 0)]) * D,
                    X_f16 + static_cast<size_t>(sTok[b0 + ((cnt > 3) ? 3 : 0)]) * D,
                    X_f16 + static_cast<size_t>(sTok[b0 + ((cnt > 4) ? 4 : 0)]) * D,
                    X_f16 + static_cast<size_t>(sTok[b0 + ((cnt > 5) ? 5 : 0)]) * D,
                    X_f16 + static_cast<size_t>(sTok[b0 + ((cnt > 6) ? 6 : 0)]) * D,
                    X_f16 + static_cast<size_t>(sTok[b0 + ((cnt > 7) ? 7 : 0)]) * D};
                float a[8];
                halo_sf4_grouped_dot_u<8>(pr, sr, s_sf4_pair, s_sf4_sc, xp, a, nblk, outer, lane);
                finish_chunk(b0, cnt, a);
            }
        }
    } else if (desc.gu_code == CODE_MX8) {
        const uint8_t* W = desc.body;
        const uint8_t* S = W + static_cast<size_t>(2 * inter) * D;
        const uint8_t* wr = W + static_cast<size_t>(row) * D;
        const uint8_t* sr = S + static_cast<size_t>(row) * (D >> 5);
        const int nblk = D >> 5;
        for (int mi = 0; mi < m_e; ++mi) {
            float acc = 0.0f;
            const half_t* xrow = X_f16 + static_cast<size_t>(sTok[mi]) * D;
            for (int blk = lane; blk < nblk; blk += 32) {
                const float sc = ue8m0_to_f32(sr[blk]);
                const uint8_t* wb = wr + (blk << 5);
                const uint4* xp = reinterpret_cast<const uint4*>(xrow + (blk << 5));
                float d = 0.0f;
#pragma unroll
                for (int q = 0; q < 4; ++q) {
                    const uint4 xv = xp[q];
                    const hv2 h0 = {s_e4m3[wb[q*8+0]], s_e4m3[wb[q*8+1]]};
                    const hv2 h1 = {s_e4m3[wb[q*8+2]], s_e4m3[wb[q*8+3]]};
                    const hv2 h2 = {s_e4m3[wb[q*8+4]], s_e4m3[wb[q*8+5]]};
                    const hv2 h3 = {s_e4m3[wb[q*8+6]], s_e4m3[wb[q*8+7]]};
                    d = fdot2_f32(__builtin_bit_cast(uint32_t, h0), xv.x, d);
                    d = fdot2_f32(__builtin_bit_cast(uint32_t, h1), xv.y, d);
                    d = fdot2_f32(__builtin_bit_cast(uint32_t, h2), xv.z, d);
                    d = fdot2_f32(__builtin_bit_cast(uint32_t, h3), xv.w, d);
                }
                acc = fmaf(d, sc, acc);
            }
            const float val = wave32_sum(acc);
            if (lane == 0) s_gu[mi][wave] = val;
        }
    } else if (desc.gu_code == CODE_GGUF_RAW) {
        const HipGgufExpertHeader* hdr = reinterpret_cast<const HipGgufExpertHeader*>(desc.body);
        const uint32_t gt = hdr->gu_type;
        const size_t rb = hip_gguf_row_bytes(gt, D);
        const int src_row = (hdr->flags & 1u) ? row : (((row & 1) ? inter : 0) + (row >> 1));
        const float ext_sc = (row & 1) ? hdr->up_scale : hdr->gate_scale;
        const uint8_t* wr = desc.body + 32 + static_cast<size_t>(src_row) * rb;
        const int n8 = D >> 3;
        for (int mi = 0; mi < m_e; ++mi) {
            float acc = 0.0f;
            const half_t* xrow = X_f16 + static_cast<size_t>(sTok[mi]) * D;
            for (int u = lane; u < n8; u += 32) {
                const uint4 wv = hip_gguf_decode_8_f16(gt, wr, u << 3, ext_sc);
                const uint4 xv = *reinterpret_cast<const uint4*>(xrow + (u << 3));
                float d = fdot2_f32(wv.x, xv.x, acc);
                d = fdot2_f32(wv.y, xv.y, d);
                d = fdot2_f32(wv.z, xv.z, d);
                acc = fdot2_f32(wv.w, xv.w, d);
            }
            const float val = wave32_sum(acc);
            if (lane == 0) s_gu[mi][wave] = val;
        }
    } else {
        const uint8_t* gu_ptr = desc.body + static_cast<size_t>(inter) * 2;
        const int nsb = D >> 8;
        const int sb_b = sk_sb_bytes(desc.gu_code);
        const uint8_t* wr = gu_ptr + static_cast<size_t>(row) * nsb * sb_b;
        const int j0 = lane * 8;
        const int sub = j0 >> 6;
        for (int mi = 0; mi < m_e; ++mi) {
            float acc = 0.0f;
            const half_t* xrow = X_f16 + static_cast<size_t>(sTok[mi]) * D;
            for (int s = 0; s < nsb; ++s) {
                const uint8_t* blk = wr + static_cast<size_t>(s) * sb_b;
                const float eff = decode_sk_sub_eff_i8(blk, desc.gu_code, sub);
                const uint4 wv = decode_sk_8_unscaled_f16(blk + 6, desc.gu_code, j0);
                const uint4 xv = *reinterpret_cast<const uint4*>(xrow + (s << 8) + j0);
                float d = fdot2_f32(wv.x, xv.x, 0.0f);
                d = fdot2_f32(wv.y, xv.y, d);
                d = fdot2_f32(wv.z, xv.z, d);
                d = fdot2_f32(wv.w, xv.w, d);
                acc = fmaf(d, eff, acc);
            }
            const float val = wave32_sum(acc);
            if (lane == 0) s_gu[mi][wave] = val;
        }
    }

    __syncthreads();
    if (lane == 0 && (wave & 1) == 0) {
        const bool div_sq = (desc.dn_code != CODE_SF4 && desc.dn_code != CODE_MX8 && desc.dn_code != CODE_GGUF_RAW);
        const float inv_sq = div_sq ? (1.0f / static_cast<float>(reinterpret_cast<const half_t*>(desc.body)[j])) : 1.0f;
        for (int mi = 0; mi < m_e; ++mi) {
            float v = situ_f32(s_gu[mi][wave], s_gu[mi][wave + 1], beta, lb);
            if (div_sq) v *= inv_sq;
            bucket_act[static_cast<size_t>(beg + mi) * inter + j] = static_cast<half_t>(v);
        }
    }
}

// Grid: ((D + 7) / 8, max_unique), Block: 256 (8 waves -> 8 output rows/block)
__global__ void __launch_bounds__(256) k_halo_moe_grouped_gemv_w2(
    const GpuExpertDesc* __restrict__ exps,     // [128]
    const int32_t* __restrict__ exp_off,        // [129]
    const int32_t* __restrict__ active_eids,    // [129]: [128] = n_active
    const half_t* __restrict__ bucket_act,      // [B * top_k, inter]
    float* __restrict__ bucket_y,               // [B * top_k, D]
    int D, int inter) {
#if defined(__HIP_DEVICE_COMPILE__) && !defined(__gfx1151__)
    return;
#endif

    const int n_act = active_eids[128];
    if (static_cast<int>(blockIdx.y) >= n_act) return;
    const int eid = active_eids[blockIdx.y];
    const int beg = exp_off[eid];
    const int m_e = exp_off[eid + 1] - beg;
    if (m_e <= 0 || m_e > 16) return;
    const GpuExpertDesc desc = exps[eid];

    const int tid = threadIdx.x;
    const int wave = tid >> 5;
    const int lane = tid & 31;
    const int row = blockIdx.x * 8 + wave;
    if (!desc.body) {
        if (lane == 0 && row < D) {
            for (int mi = 0; mi < m_e; ++mi) bucket_y[static_cast<size_t>(beg + mi) * D + row] = 0.0f;
        }
        return;
    }

    __shared__ half_t s_e4m3[256];
    __shared__ uint32_t s_sf4_pair[256];
    __shared__ float s_sf4_sc[256];
    if (desc.dn_code == CODE_MX8) {
        s_e4m3[tid] = static_cast<half_t>(e4m3fn_to_f32(static_cast<uint8_t>(tid)));
    } else if (desc.dn_code == CODE_SF4) {
        const hv2 hp = {sf4_e2m1_f16(static_cast<uint32_t>(tid) & 15u), sf4_e2m1_f16(static_cast<uint32_t>(tid) >> 4)};
        s_sf4_pair[tid] = __builtin_bit_cast(uint32_t, hp);
        s_sf4_sc[tid] = e4m3fn_to_f32(static_cast<uint8_t>(tid));
    }
    __syncthreads();
    if (row >= D) return;

    if (desc.dn_code == CODE_SF4) {
        const float* hdr = reinterpret_cast<const float*>(desc.body);
        const float outer = hdr[2] / hdr[3];
        const uint8_t* packed = desc.body + desc.dn_offset;
        const uint8_t* scales = packed + static_cast<size_t>(D) * (inter >> 1);
        const uint8_t* pr = packed + static_cast<size_t>(row) * (inter >> 1);
        const uint8_t* sr = scales + static_cast<size_t>(row) * (inter >> 4);
        const int nblk = inter >> 4;
        const half_t* abase = bucket_act + static_cast<size_t>(beg) * inter;

        auto write_chunk = [&](int base_mi, int cnt, const float* vals) {
            for (int i = 0; i < cnt; ++i) {
                const float v = wave32_sum(vals[i]);
                if (lane == 0) bucket_y[static_cast<size_t>(beg + base_mi + i) * D + row] = v;
            }
        };

        if (m_e == 1) {
            const half_t* ap[1] = {abase};
            float a[1];
            halo_sf4_grouped_dot_u<1>(pr, sr, s_sf4_pair, s_sf4_sc, ap, a, nblk, outer, lane);
            write_chunk(0, 1, a);
        } else if (m_e == 2) {
            const half_t* ap[2] = {abase, abase + inter};
            float a[2];
            halo_sf4_grouped_dot_u<2>(pr, sr, s_sf4_pair, s_sf4_sc, ap, a, nblk, outer, lane);
            write_chunk(0, 2, a);
        } else if (m_e <= 4) {
            const half_t* ap[4] = {
                abase,
                abase + inter,
                abase + static_cast<size_t>((m_e > 2) ? 2 : 0) * inter,
                abase + static_cast<size_t>((m_e > 3) ? 3 : 0) * inter};
            float a[4];
            halo_sf4_grouped_dot_u<4>(pr, sr, s_sf4_pair, s_sf4_sc, ap, a, nblk, outer, lane);
            write_chunk(0, m_e, a);
        } else if (m_e <= 8) {
            const half_t* ap[8] = {
                abase + 0 * inter, abase + 1 * inter, abase + 2 * inter, abase + 3 * inter,
                abase + 4 * inter,
                abase + static_cast<size_t>((m_e > 5) ? 5 : 0) * inter,
                abase + static_cast<size_t>((m_e > 6) ? 6 : 0) * inter,
                abase + static_cast<size_t>((m_e > 7) ? 7 : 0) * inter};
            float a[8];
            halo_sf4_grouped_dot_u<8>(pr, sr, s_sf4_pair, s_sf4_sc, ap, a, nblk, outer, lane);
            write_chunk(0, m_e, a);
        } else {
            for (int b0 = 0; b0 < m_e; b0 += 8) {
                const int cnt = (m_e - b0 < 8) ? (m_e - b0) : 8;
                const half_t* ab = abase + static_cast<size_t>(b0) * inter;
                const half_t* ap[8] = {
                    ab,
                    ab + static_cast<size_t>((cnt > 1) ? 1 : 0) * inter,
                    ab + static_cast<size_t>((cnt > 2) ? 2 : 0) * inter,
                    ab + static_cast<size_t>((cnt > 3) ? 3 : 0) * inter,
                    ab + static_cast<size_t>((cnt > 4) ? 4 : 0) * inter,
                    ab + static_cast<size_t>((cnt > 5) ? 5 : 0) * inter,
                    ab + static_cast<size_t>((cnt > 6) ? 6 : 0) * inter,
                    ab + static_cast<size_t>((cnt > 7) ? 7 : 0) * inter};
                float a[8];
                halo_sf4_grouped_dot_u<8>(pr, sr, s_sf4_pair, s_sf4_sc, ap, a, nblk, outer, lane);
                write_chunk(b0, cnt, a);
            }
        }
    } else if (desc.dn_code == CODE_MX8) {
        const uint8_t* W = desc.body + desc.dn_offset;
        const uint8_t* S = W + static_cast<size_t>(D) * inter;
        const uint8_t* wr = W + static_cast<size_t>(row) * inter;
        const uint8_t* sr = S + static_cast<size_t>(row) * (inter >> 5);
        const int nblk = inter >> 5;
        for (int mi = 0; mi < m_e; ++mi) {
            float acc = 0.0f;
            const half_t* arow = bucket_act + static_cast<size_t>(beg + mi) * inter;
            for (int blk = lane; blk < nblk; blk += 32) {
                const float sc = ue8m0_to_f32(sr[blk]);
                const uint8_t* wb = wr + (blk << 5);
                const uint4* xp = reinterpret_cast<const uint4*>(arow + (blk << 5));
                float d = 0.0f;
#pragma unroll
                for (int q = 0; q < 4; ++q) {
                    const uint4 xv = xp[q];
                    const hv2 h0 = {s_e4m3[wb[q*8+0]], s_e4m3[wb[q*8+1]]};
                    const hv2 h1 = {s_e4m3[wb[q*8+2]], s_e4m3[wb[q*8+3]]};
                    const hv2 h2 = {s_e4m3[wb[q*8+4]], s_e4m3[wb[q*8+5]]};
                    const hv2 h3 = {s_e4m3[wb[q*8+6]], s_e4m3[wb[q*8+7]]};
                    d = fdot2_f32(__builtin_bit_cast(uint32_t, h0), xv.x, d);
                    d = fdot2_f32(__builtin_bit_cast(uint32_t, h1), xv.y, d);
                    d = fdot2_f32(__builtin_bit_cast(uint32_t, h2), xv.z, d);
                    d = fdot2_f32(__builtin_bit_cast(uint32_t, h3), xv.w, d);
                }
                acc = fmaf(d, sc, acc);
            }
            const float val = wave32_sum(acc);
            if (lane == 0) bucket_y[static_cast<size_t>(beg + mi) * D + row] = val;
        }
    } else if (desc.dn_code == CODE_GGUF_RAW) {
        const HipGgufExpertHeader* hdr = reinterpret_cast<const HipGgufExpertHeader*>(desc.body);
        const uint32_t dt = hdr->dn_type;
        const size_t rb = hip_gguf_row_bytes(dt, inter);
        const uint8_t* wr = desc.body + desc.dn_offset + static_cast<size_t>(row) * rb;
        const int n8 = inter >> 3;
        for (int mi = 0; mi < m_e; ++mi) {
            float acc = 0.0f;
            const half_t* arow = bucket_act + static_cast<size_t>(beg + mi) * inter;
            for (int u = lane; u < n8; u += 32) {
                const uint4 wv = hip_gguf_decode_8_f16(dt, wr, u << 3, hdr->dn_scale);
                const uint4 xv = *reinterpret_cast<const uint4*>(arow + (u << 3));
                float d = fdot2_f32(wv.x, xv.x, acc);
                d = fdot2_f32(wv.y, xv.y, d);
                d = fdot2_f32(wv.z, xv.z, d);
                acc = fdot2_f32(wv.w, xv.w, d);
            }
            const float val = wave32_sum(acc);
            if (lane == 0) bucket_y[static_cast<size_t>(beg + mi) * D + row] = val;
        }
    } else {
        const int nsb = inter >> 8;
        const int sb_b = sk_sb_bytes(desc.dn_code);
        const uint8_t* wr = desc.body + desc.dn_offset + static_cast<size_t>(row) * nsb * sb_b;
        const int j0 = lane * 8;
        const int sub = j0 >> 6;
        for (int mi = 0; mi < m_e; ++mi) {
            float acc = 0.0f;
            const half_t* arow = bucket_act + static_cast<size_t>(beg + mi) * inter;
            for (int s = 0; s < nsb; ++s) {
                const uint8_t* blk = wr + static_cast<size_t>(s) * sb_b;
                const float eff = decode_sk_sub_eff_i8(blk, desc.dn_code, sub);
                const uint4 wv = decode_sk_8_unscaled_f16(blk + 6, desc.dn_code, j0);
                const uint4 xv = *reinterpret_cast<const uint4*>(arow + (s << 8) + j0);
                float d = fdot2_f32(wv.x, xv.x, 0.0f);
                d = fdot2_f32(wv.y, xv.y, d);
                d = fdot2_f32(wv.z, xv.z, d);
                d = fdot2_f32(wv.w, xv.w, d);
                acc = fmaf(d, eff, acc);
            }
            const float val = wave32_sum(acc);
            if (lane == 0) bucket_y[static_cast<size_t>(beg + mi) * D + row] = val;
        }
    }
}

}  // namespace strata::inkling2::hip
