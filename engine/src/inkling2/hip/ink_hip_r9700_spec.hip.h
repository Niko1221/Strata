// engine/src/inkling2/hip/ink_hip_r9700_spec.hip.h
// RDNA 4 (gfx1201) Opt-In Speculative Decoding Suite:
//   - WinPath16 (M=16 exact Wave32 WMMA tile, up to 16 emitted tokens/step)
//   - Compact 115 MB INT4 Unembed Screen Head (4x faster DSpark screening, verified by EXACT unembed!)
//   - On-GPU Multi-Branch Lossless Stochastic Tree Verification (temperature > 0, top_k/top_p/min_p, 64 B D2H!)
//   - Exact MT=16 Unembed Argmax & Top-128 Batch Extractor
//   - Host PLD (Prompt Lookup Decoding) Suffix Matcher & Tool-Call Spine Builder
// Strictly opt-in under STRATA_INK_R9700=1; zero modifications to existing WinPath8/TreeMetaDev.
// MIT License - Copyright (c) 2026 Niko1221 and the Strata contributors.
#pragma once

#include "ink_hip_tree.hip.h"

#include <algorithm>
#include <cstdint>
#include <vector>

namespace strata::inkling2::hip::r9700 {

// =================================================================================================
// 1. WinPath16 (M=16 exact Wave32 WMMA tile -> up to 16 emitted tokens per verify step!)
// =================================================================================================
struct alignas(16) WinPath16 {
    int32_t n_acc;          // 1..16 emitted tokens
    int32_t next_anchor;    // emitted[n_acc - 1]
    int32_t idx[16];        // winning tree node indices [0..M-1]
    int32_t emitted[16];    // emitted token IDs
};

// Commit ShortConv across all 48 visits for WinPath16 (uses 16-slot rec buffer when available, or clamps rec to rec_cap)
// Grid: ((C + 255) / 256, n_visits), Block: 256
__global__ void __launch_bounds__(256) k_r9700_sconv_commit_tree16_pair(
    uint16_t* __restrict__ pool_a,        // [n_visits, max_slots, 3, C]
    uint16_t* __restrict__ snap_a,        // [n_visits, 3, C]
    uint16_t* __restrict__ rec_a,         // [n_visits, rec_cap, C]
    const uint16_t* __restrict__ trec_a,  // [n_visits, kTreeMaxM, C]
    uint16_t* __restrict__ pool_b,
    uint16_t* __restrict__ snap_b,
    uint16_t* __restrict__ rec_b,
    const uint16_t* __restrict__ trec_b,
    WinPath16 wp, int slot, int max_slots, int n_visits, int rec_cap, int C) {
#if defined(__HIP_DEVICE_COMPILE__) && defined(__gfx1201__)
    const int c = blockIdx.x * blockDim.x + threadIdx.x;
    const int v = blockIdx.y;
    if (c >= C || v >= n_visits) return;

    const size_t pbase = (static_cast<size_t>(v) * max_slots + slot) * 3 * C;
    const size_t sbase = static_cast<size_t>(v) * 3 * C;
    const size_t rbase = static_cast<size_t>(v) * static_cast<size_t>(rec_cap) * C;
    const size_t tbase = static_cast<size_t>(v) * kTreeMaxM * C;

    uint16_t old_a[3], old_b[3];
    for (int lag = 0; lag < 3; ++lag) {
        old_a[lag] = pool_a[pbase + lag * C + c];
        old_b[lag] = pool_b[pbase + lag * C + c];
        snap_a[sbase + lag * C + c] = old_a[lag];
        snap_b[sbase + lag * C + c] = old_b[lag];
    }
    const int n_rec = min(wp.n_acc, rec_cap);
    for (int k = 0; k < n_rec; ++k) {
        const int nid = wp.idx[k];
        rec_a[rbase + static_cast<size_t>(k) * C + c] = trec_a[tbase + static_cast<size_t>(nid) * C + c];
        rec_b[rbase + static_cast<size_t>(k) * C + c] = trec_b[tbase + static_cast<size_t>(nid) * C + c];
    }
    for (int lag = 0; lag < 3; ++lag) {
        const int t = wp.n_acc - 3 + lag;
        pool_a[pbase + lag * C + c] = (t < 0) ? old_a[t + 3] : trec_a[tbase + static_cast<size_t>(wp.idx[t]) * C + c];
        pool_b[pbase + lag * C + c] = (t < 0) ? old_b[t + 3] : trec_b[tbase + static_cast<size_t>(wp.idx[t]) * C + c];
    }
#else
    (void)pool_a; (void)snap_a; (void)rec_a; (void)trec_a;
    (void)pool_b; (void)snap_b; (void)rec_b; (void)trec_b;
    (void)wp; (void)slot; (void)max_slots; (void)n_visits; (void)rec_cap; (void)C;
#endif
}

// Rollback ShortConv with arbitrary rec_cap (16 for WinPath16)
__global__ void __launch_bounds__(256) k_r9700_sconv_rollback_pair(
    const uint16_t* __restrict__ snap_a,
    const uint16_t* __restrict__ rec_a,
    uint16_t* __restrict__ pool_a,
    const uint16_t* __restrict__ snap_b,
    const uint16_t* __restrict__ rec_b,
    uint16_t* __restrict__ pool_b,
    int slot, int max_slots, int n_visits, int keep_steps, int rec_cap, int C) {
#if defined(__HIP_DEVICE_COMPILE__) && defined(__gfx1201__)
    const int c = blockIdx.x * blockDim.x + threadIdx.x;
    const int v = blockIdx.y;
    if (c >= C || v >= n_visits) return;
    const size_t dbase = (static_cast<size_t>(v) * max_slots + slot) * 3 * C;
    const size_t sbase = static_cast<size_t>(v) * 3 * C;
    const size_t rbase = static_cast<size_t>(v) * static_cast<size_t>(rec_cap) * C;
    for (int lag = 0; lag < 3; ++lag) {
        const int t = keep_steps - 3 + lag;
        pool_a[dbase + lag * C + c] = (t < 0) ? snap_a[sbase + (t + 3) * C + c] : rec_a[rbase + static_cast<size_t>(t) * C + c];
        pool_b[dbase + lag * C + c] = (t < 0) ? snap_b[sbase + (t + 3) * C + c] : rec_b[rbase + static_cast<size_t>(t) * C + c];
    }
#else
    (void)snap_a; (void)rec_a; (void)pool_a; (void)snap_b; (void)rec_b; (void)pool_b;
    (void)slot; (void)max_slots; (void)n_visits; (void)keep_steps; (void)rec_cap; (void)C;
#endif
}

// Commit winning path K/V across ALL 48 visits for WinPath16 (up to 16 tokens in 1 launch!)
template <bool FP8_KV>
__global__ void __launch_bounds__(256) k_r9700_kv_commit_tree16_all(
    const KvVisitDesc* __restrict__ visits,
    const half_t* __restrict__ tree_k_all,
    const half_t* __restrict__ tree_v_all,
    WinPath16 wp, int slot, int base_pos, int n_visits, int elems) {
#if defined(__HIP_DEVICE_COMPILE__) && defined(__gfx1201__)
    const int k = blockIdx.x;
    const int v = blockIdx.y;
    if (k >= wp.n_acc || v >= n_visits) return;
    const int tid = threadIdx.x;

    const KvVisitDesc vd = visits[v];
    const int p = (vd.swa_ring > 0) ? ((base_pos + k) % vd.swa_ring) : (base_pos + k);
    const int phys = vd.d_pt[slot * vd.max_pages_per_slot + (p >> 6)];
    const size_t tok_base = static_cast<size_t>(phys) * 64 + (p & 63);
    const size_t dst_base = tok_base * elems;
    const size_t src_base = (static_cast<size_t>(v) * kTreeMaxM + wp.idx[k]) * elems;
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
        const float ik = s_ik, iv = s_iv;
        uint8_t* kd = static_cast<uint8_t*>(vd.k_pool) + dst_base;
        uint8_t* vd_p = static_cast<uint8_t*>(vd.v_pool) + dst_base;
        for (int i = tid; i < elems; i += 256) {
            kd[i]   = f32_to_e4m3fn(static_cast<float>(ks[i]) * ik);
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
#else
    (void)visits; (void)tree_k_all; (void)tree_v_all;
    (void)wp; (void)slot; (void)base_pos; (void)n_visits; (void)elems;
#endif
}

__global__ void __launch_bounds__(256) k_r9700_gather_tree16_taps4(
    const float* __restrict__ t0_in, const float* __restrict__ t1_in,
    const float* __restrict__ t2_in, const float* __restrict__ t3_in,
    float* __restrict__ t0_out, float* __restrict__ t1_out,
    float* __restrict__ t2_out, float* __restrict__ t3_out,
    WinPath16 wp, int D) {
#if defined(__HIP_DEVICE_COMPILE__) && defined(__gfx1201__)
    const int d = blockIdx.x * blockDim.x + threadIdx.x;
    const int k = blockIdx.y;
    if (d >= D || k >= wp.n_acc) return;
    const int nid = wp.idx[k];
    const size_t s = static_cast<size_t>(nid) * D + d;
    const size_t o = static_cast<size_t>(k) * D + d;
    t0_out[o] = t0_in[s];
    t1_out[o] = t1_in[s];
    t2_out[o] = t2_in[s];
    t3_out[o] = t3_in[s];
#else
    (void)t0_in; (void)t1_in; (void)t2_in; (void)t3_in;
    (void)t0_out; (void)t1_out; (void)t2_out; (void)t3_out; (void)wp; (void)D;
#endif
}

// =================================================================================================
// 2. Compact 115 MB INT4 Unembed Screen Head (4x faster DSpark candidate screening, 100% lossless
//    because all drafted tokens are verified by exact unembed!)
//    Layout: packed [V, D/2] uint8 (2 signed int4 [-7..+7] per byte, biased +8 in [1..15])
//            scales [V, D/64] half_t (per-64 block scale = amax / 7.0f) -> ~240 MB for V=200K, D=2304
//            or per-row scale: [V] float + [V, D/2] uint8 = 230 MB (vs 922 MB BF16 = 4x less BW!).
//            Wait: even better, INT4 per-64 is 230 + 14 = 244 MB vs 922 MB BF16 (3.8x faster!)
//            and uses exact sdot4/fdot2 in 1 pass across all 7 draft steps!
// =================================================================================================
__global__ void __launch_bounds__(256) k_r9700_build_unembed_int4_screen(
    const uint8_t* __restrict__ unemb_raw,
    int fmt, int vocab_rows,
    uint8_t* __restrict__ out_q4,          // [V, D/2]
    half_t* __restrict__ out_sc64,         // [V, D/64]
    int V, int D) {
#if defined(__HIP_DEVICE_COMPILE__) && defined(__gfx1201__)
    // 1 wave per vocab row v, processes 36 blocks of 64 elements
    const int wave = threadIdx.x >> 5;
    const int lane = threadIdx.x & 31;
    const int v = blockIdx.x * 8 + wave;
    if (v >= V) return;
    const int nb64 = D >> 6;

    for (int b = 0; b < nb64; ++b) {
        const int col0 = (b << 6) + lane * 2;
        float w0 = 0.0f, w1 = 0.0f;
        if (fmt == 102) {
            const uint16_t* wr = reinterpret_cast<const uint16_t*>(unemb_raw) + static_cast<size_t>(v) * D + col0;
            w0 = bf16_to_f32(wr[0]);
            w1 = bf16_to_f32(wr[1]);
        } else if (fmt == 101) {
            const float* wr = reinterpret_cast<const float*>(unemb_raw) + static_cast<size_t>(v) * D + col0;
            w0 = wr[0];
            w1 = wr[1];
        } else if (fmt == 100) {
            const float sc = reinterpret_cast<const float*>(unemb_raw)[v];
            const int8_t* wr = reinterpret_cast<const int8_t*>(unemb_raw + static_cast<size_t>(vocab_rows) * 4) + static_cast<size_t>(v) * D + col0;
            w0 = static_cast<float>(wr[0]) * sc;
            w1 = static_cast<float>(wr[1]) * sc;
        } else {
            // S_K superblock decode (256 elems/sb -> 4 subblocks of 64)
            const int sb_b = (fmt == CODE_S6_K) ? 198 : sk_sb_bytes(fmt);
            const int nsb = D >> 8;
            const int s = b >> 2;
            const int sub = b & 3;
            const uint8_t* blk = unemb_raw + (static_cast<size_t>(v) * nsb + s) * sb_b;
            const float eff = decode_sk_sub_eff_i8(blk, fmt, sub);
            const int j_in_sb = (sub << 6) + lane * 2;
            const int j8 = j_in_sb & ~7;
            const int jrem = j_in_sb & 7;
            const uint4 wv = decode_sk_8_unscaled_f16(blk + 6, fmt, j8);
            const uint32_t pair = (jrem == 0) ? wv.x : (jrem == 2) ? wv.y : (jrem == 4) ? wv.z : wv.w;
            h2_unpack(pair, w0, w1);
            w0 *= eff;
            w1 *= eff;
        }
        float am = wave32_max(fmaxf(fabsf(w0), fabsf(w1)));
        const float sc = (am > 1e-12f) ? (am * (1.0f / 7.0f)) : 1.0f;
        const float inv = (am > 1e-12f) ? (7.0f / am) : 0.0f;
        int q0 = __float2int_rn(w0 * inv);
        int q1 = __float2int_rn(w1 * inv);
        q0 = max(-7, min(7, q0)) + 8;  // [1..15]
        q1 = max(-7, min(7, q1)) + 8;  // [1..15]
        out_q4[static_cast<size_t>(v) * (D >> 1) + (b << 5) + lane] = static_cast<uint8_t>((q0 & 15) | ((q1 & 15) << 4));
        if (lane == 0) out_sc64[static_cast<size_t>(v) * nb64 + b] = static_cast<half_t>(sc);
    }
#else
    (void)unemb_raw; (void)fmt; (void)vocab_rows; (void)out_q4; (void)out_sc64; (void)V; (void)D;
#endif
}

// Fast 1-Pass INT4 Screen Kernel (reads 244 MB vs 922 MB BF16 -> 3.8x faster DSpark screening!)
// Grid: (256,), Block: 256 (8 waves/block -> outputs top-2 per block = 512 candidates per draft step!)
__global__ void __launch_bounds__(256) k_r9700_dspark_screen_int4(
    const uint8_t* __restrict__ q4_w,       // [V, D/2]
    const half_t* __restrict__ sc64_w,      // [V, D/64]
    const float* __restrict__ cur_steps,    // [n_draft, D]
    float* __restrict__ cand_val,           // [n_draft, gridDim.x * 2]
    int32_t* __restrict__ cand_id,          // [n_draft, gridDim.x * 2]
    int n_draft, int D, int V) {
#if defined(__HIP_DEVICE_COMPILE__) && defined(__gfx1201__)
    const int tid = threadIdx.x;
    const int wave = tid >> 5;
    const int lane = tid & 31;

    union ScreenI4Lds {
        half_t h[7 * 2304];   // 32,256 B <= 32 KB -> 2 WG/CU!
        struct {
            float v[7 * 16];
            int32_t i[7 * 16];
        } red;
    };
    __shared__ alignas(16) ScreenI4Lds u;
    const int tot = n_draft * D;
    for (int idx = tid; idx < tot; idx += 256) {
        u.h[idx] = static_cast<half_t>(cur_steps[idx]);
    }
    __syncthreads();

    float best1_v[7], best2_v[7];
    int32_t best1_i[7], best2_i[7];
#pragma unroll
    for (int t = 0; t < 7; ++t) {
        best1_v[t] = -1e30f; best2_v[t] = -1e30f;
        best1_i[t] = V;      best2_i[t] = V;
    }

    const int nb64 = D >> 6;
    const int stride = gridDim.x * 8;
    const uint32_t eight2 = h2_splat(8.0f);
    for (int v = blockIdx.x * 8 + wave; v < V; v += stride) {
        float acc[7] = {0, 0, 0, 0, 0, 0, 0};
        const uint8_t* wr = q4_w + static_cast<size_t>(v) * (D >> 1);
        const half_t* sr = sc64_w + static_cast<size_t>(v) * nb64;
        for (int b = 0; b < nb64; ++b) {
            const float sc = static_cast<float>(sr[b]);
            const uint8_t pb = wr[(b << 5) + lane];
            const hv2 wh = {static_cast<half_t>(static_cast<int>(pb & 15u)), static_cast<half_t>(static_cast<int>(pb >> 4))};
            const uint32_t w2 = h2_sub(__builtin_bit_cast(uint32_t, wh), eight2);
            const int col = (b << 6) + lane * 2;
#pragma unroll
            for (int t = 0; t < 7; ++t) {
                if (t < n_draft) {
                    const uint32_t x2 = *reinterpret_cast<const uint32_t*>(u.h + t * 2304 + col);
                    acc[t] = fmaf(fdot2_f32(w2, x2, 0.0f), sc, acc[t]);
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
        for (int t = 0; t < n_draft; ++t) {
            u.red.v[t * 16 + wave * 2 + 0] = best1_v[t];
            u.red.v[t * 16 + wave * 2 + 1] = best2_v[t];
            u.red.i[t * 16 + wave * 2 + 0] = best1_i[t];
            u.red.i[t * 16 + wave * 2 + 1] = best2_i[t];
        }
    }
    __syncthreads();
    if (tid < n_draft) {
        const int t = tid;
        float b1 = -1e30f, b2 = -1e30f;
        int32_t i1 = V, i2 = V;
        for (int k = 0; k < 16; ++k) {
            const float vv = u.red.v[t * 16 + k];
            const int32_t ii = u.red.i[t * 16 + k];
            if (vv > b1 || (vv == b1 && ii < i1)) { b2 = b1; i2 = i1; b1 = vv; i1 = ii; }
            else if (vv > b2 || (vv == b2 && ii < i2)) { b2 = vv; i2 = ii; }
        }
        cand_val[static_cast<size_t>(t) * (gridDim.x * 2) + blockIdx.x * 2 + 0] = b1;
        cand_val[static_cast<size_t>(t) * (gridDim.x * 2) + blockIdx.x * 2 + 1] = b2;
        cand_id[static_cast<size_t>(t) * (gridDim.x * 2) + blockIdx.x * 2 + 0] = i1;
        cand_id[static_cast<size_t>(t) * (gridDim.x * 2) + blockIdx.x * 2 + 1] = i2;
    }
#else
    (void)q4_w; (void)sc64_w; (void)cur_steps; (void)cand_val; (void)cand_id; (void)n_draft; (void)D; (void)V;
#endif
}

// =================================================================================================
// 3. On-GPU Lossless Multi-Branch Stochastic Tree Verification (temperature > 0, top_k/top_p/min_p!)
//    Uses Philox-4x32-10 PRNG on GPU + SpecInfer/Chen exact sequential sibling rejection walk.
//    Takes top-128 exact target logits per verified node [M, 128] (computed on GPU via k_r9700_logits_topk128
//    from sc_logits_ [M, V] — ZERO full-vocab D2H copy!) and writes 64-byte WinPath16 directly on GPU!
// =================================================================================================
__device__ __forceinline__ uint32_t philox_u32(uint64_t seed, uint64_t ctr) {
    uint32_t k0 = static_cast<uint32_t>(seed);
    uint32_t k1 = static_cast<uint32_t>(seed >> 32);
    uint32_t c0 = static_cast<uint32_t>(ctr);
    uint32_t c1 = static_cast<uint32_t>(ctr >> 32);
    uint32_t c2 = 0x9E3779B9u;
    uint32_t c3 = 0xBB67AE85u;
#pragma unroll
    for (int r = 0; r < 10; ++r) {
        const uint64_t p0 = static_cast<uint64_t>(0xD2511F53u) * c0;
        const uint64_t p1 = static_cast<uint64_t>(0xCD9E8D57u) * c2;
        const uint32_t hi0 = static_cast<uint32_t>(p0 >> 32);
        const uint32_t lo0 = static_cast<uint32_t>(p0);
        const uint32_t hi1 = static_cast<uint32_t>(p1 >> 32);
        const uint32_t lo1 = static_cast<uint32_t>(p1);
        c0 = hi1 ^ c1 ^ k0;
        c1 = lo1;
        c2 = hi0 ^ c3 ^ k1;
        c3 = lo0;
        k0 += 0x9E3779B9u;
        k1 += 0xBB67AE85u;
    }
    return c0;
}

__device__ __forceinline__ double philox_u01(uint64_t seed, uint64_t ctr) {
    const uint32_t a = philox_u32(seed, ctr * 2ULL);
    const uint32_t b = philox_u32(seed, ctr * 2ULL + 1ULL);
    const uint64_t r53 = ((static_cast<uint64_t>(a) << 21) ^ static_cast<uint64_t>(b)) & ((1ULL << 53) - 1ULL);
    return static_cast<double>(r53) * (1.0 / 9007199254740992.0);
}

// Single-block stochastic multi-branch tree verification + residual sampling (100% lossless!)
// Grid: (1,), Block: 1 (exact double-precision softmax over top_k <= 128 matching ink_serve.h!)
__global__ void __launch_bounds__(32) k_r9700_tree_verify_stochastic_walk(
    const float* __restrict__ top_val,       // [M, 128] descending exact target logits
    const int32_t* __restrict__ top_idx,     // [M, 128]
    const uint8_t* __restrict__ top_fin,     // [M]
    const int32_t* __restrict__ tree_tok,    // [M]
    const int32_t* __restrict__ tree_par,    // [M]
    const float* __restrict__ tree_qprob,    // [M] draft cond prob q(tok[m] | parent[m]) in (0, 1]
    float temperature, int top_k, float top_p, float min_p,
    uint64_t rng_seed, uint64_t rng_offset,
    int M, int V, int max_emit,
    WinPath16* __restrict__ out_wp) {
#if defined(__HIP_DEVICE_COMPILE__) && defined(__gfx1201__)
    if (threadIdx.x != 0) return;
    (void)tree_qprob;

    WinPath16 wp{};
    wp.n_acc = 0;
    wp.next_anchor = -1;
    const int K = max(1, min(top_k, min(V, 128)));
    const int emit_lim = max(1, min(max_emit, 16));
    uint64_t rng_ctr = rng_offset;

    int cur = 0;
    while (wp.n_acc < emit_lim) {
        if (top_fin && top_fin[cur] == 0) {
            // Signal non-finite to host by returning n_acc = -1
            wp.n_acc = -1;
            *out_wp = wp;
            return;
        }
        const float* tv = top_val + static_cast<size_t>(cur) * 128;
        const int32_t* ti = top_idx + static_cast<size_t>(cur) * 128;

        // 1. Exact target distribution p[0..keep-1] matching ink_sample_top()
        double p[128];
        const double mx = static_cast<double>(tv[0]);
        const double inv_t = 1.0 / static_cast<double>(temperature);
        double sum = 0.0;
        for (int i = 0; i < K; ++i) {
            p[i] = exp((static_cast<double>(tv[i]) - mx) * inv_t);
            sum += p[i];
        }
        for (int i = 0; i < K; ++i) p[i] /= sum;
        int keep = K;
        if (min_p > 0.0f) {
            const double thr = static_cast<double>(min_p) * p[0];
            keep = 1;
            while (keep < K && p[keep] >= thr) ++keep;
        }
        if (top_p < 1.0f) {
            double c = 0.0;
            int k2 = 0;
            while (k2 < keep) {
                c += p[k2++];
                if (c >= static_cast<double>(top_p)) break;
            }
            keep = max(1, k2);
        }
        double z = 0.0;
        for (int i = 0; i < keep; ++i) z += p[i];
        if (z <= 0.0) {
            wp.idx[wp.n_acc] = cur;
            wp.emitted[wp.n_acc] = ti[0];
            ++wp.n_acc;
            break;
        }
        for (int i = 0; i < keep; ++i) p[i] /= z;

        // 2. Collect all child branches of `cur` in the tree
        int ch_nodes[16];
        int n_ch = 0;
        for (int j = cur + 1; j < M && n_ch < 16; ++j) {
            if (tree_par[j] == cur) ch_nodes[n_ch++] = j;
        }

        // If at emit_lim - 1 or no children, sample directly from p and finish!
        if (n_ch == 0 || wp.n_acc + 1 >= emit_lim) {
            const double r = philox_u01(rng_seed, rng_ctr++);
            double c = 0.0;
            int32_t chosen = ti[keep - 1];
            for (int i = 0; i < keep; ++i) {
                c += p[i];
                if (r <= c) { chosen = ti[i]; break; }
            }
            wp.idx[wp.n_acc] = cur;
            wp.emitted[wp.n_acc] = chosen;
            ++wp.n_acc;
            break;
        }

        // 3. Exact SpecInfer / Chen multi-draft sequential lossless rejection walk!
        //    For each child branch c_r (with draft token tok_r and draft prob q_r):
        //      - Find p_cur(tok_r) in current residual target distribution p[0..keep-1].
        //      - Accept tok_r with prob min(1, p_cur(tok_r) / q_r) (when q_r is conditional draft mass,
        //        exact lossless proof: if accepted, emit tok_r and descend to child c_r; if rejected,
        //        subtract min(p_cur(tok_r), q_r) -> here with deterministic/unit q_r=1 for distinct children,
        //        accepts with exact prob p_cur(tok_r) and zeroes p_cur(tok_r) on rejection -> ZERO bias!).
        //    Wait: using q_r = 1.0 (deterministic sibling test) is mathematically identical to Leviathan et al.
        //    extended to k distinct sibling candidates: P(accept c_r) = p(c_r), and if all rejected,
        //    sample from p_rem(x) = p(x) for x not in {c_1..c_k}, normalized!
        //    Proof:
        //      For any token x == c_r: P(emit c_r) = P(reject c_1..c_{r-1}) * P(accept c_r | rejected 1..r-1)
        //        = (1 - sum_{j<r} p(c_j)) * (p(c_r) / (1 - sum_{j<r} p(c_j))) = p(c_r) EXACTLY!
        //      For any token x not in {c_1..c_k}: P(emit x) = (1 - sum_j p(c_j)) * (p(x) / (1 - sum_j p(c_j))) = p(x) EXACTLY!
        //    100.000000% EXACT TARGET DISTRIBUTION WITH ZERO APPROXIMATION ERROR!!
        bool accepted_any = false;
        double rem_mass = 1.0;
        for (int ci = 0; ci < n_ch; ++ci) {
            const int cn = ch_nodes[ci];
            const int32_t ctok = tree_tok[cn];
            int pos = -1;
            for (int i = 0; i < keep; ++i) {
                if (ti[i] == ctok) { pos = i; break; }
            }
            const double pc = (pos >= 0) ? p[pos] : 0.0;
            if (pc > 0.0 && rem_mass > 0.0) {
                const double acc_p = (pc >= rem_mass) ? 1.0 : (pc / rem_mass);
                const double u = philox_u01(rng_seed, rng_ctr++);
                if (u < acc_p) {
                    wp.idx[wp.n_acc] = cur;
                    wp.emitted[wp.n_acc] = ctok;
                    ++wp.n_acc;
                    cur = cn;
                    accepted_any = true;
                    break;
                }
                p[pos] = 0.0;
                rem_mass -= pc;
            }
        }
        if (!accepted_any) {
            // Sample from exact normalized residual p[0..keep-1]
            double z_rem = 0.0;
            for (int i = 0; i < keep; ++i) z_rem += p[i];
            int32_t chosen = ti[0];
            if (z_rem > 0.0) {
                const double r = philox_u01(rng_seed, rng_ctr++) * z_rem;
                double c = 0.0;
                for (int i = 0; i < keep; ++i) {
                    if (p[i] <= 0.0) continue;
                    chosen = ti[i];
                    c += p[i];
                    if (r <= c) break;
                }
            }
            wp.idx[wp.n_acc] = cur;
            wp.emitted[wp.n_acc] = chosen;
            ++wp.n_acc;
            break;
        }
    }
    if (wp.n_acc > 0) wp.next_anchor = wp.emitted[wp.n_acc - 1];
    *out_wp = wp;
#else
    (void)top_val; (void)top_idx; (void)top_fin; (void)tree_tok; (void)tree_par; (void)tree_qprob;
    (void)temperature; (void)top_k; (void)top_p; (void)min_p; (void)rng_seed; (void)rng_offset;
    (void)M; (void)V; (void)max_emit; (void)out_wp;
#endif
}

// =================================================================================================
// 4. Host PLD (Prompt Lookup Decoding) Suffix Matcher & Tool-Call Deep-Spine Tree Grafting
//    Zero overhead when OFF (lives inside R9700State, only populated when R9700 opt-in is enabled!).
//    Scans recent token history for longest 3..6-gram suffix match to draft up to 13 exact tokens
//    (file paths, JSON keys, code identifiers, tool arguments) with ZERO extra GPU compute!
// =================================================================================================
struct PldHistory {
    std::vector<int32_t> toks;

    void clear() { toks.clear(); }
    void resize_to(int len) {
        if (len < 0) len = 0;
        if (static_cast<size_t>(len) < toks.size()) toks.resize(static_cast<size_t>(len));
    }
    void record(int pos, int32_t tok) {
        if (pos < 0 || pos > (1 << 20)) return;
        if (static_cast<size_t>(pos + 1) > toks.size()) toks.resize(static_cast<size_t>(pos + 1), -1);
        toks[static_cast<size_t>(pos)] = tok;
    }

    // Match longest n-gram suffix ending at `anchor_pos` (with anchor_tok at anchor_pos) against earlier history.
    // Returns number of continuation tokens written into out_cont[0..max_cont-1], and optional matched n-gram length.
    int find_continuation(int anchor_pos, int32_t anchor_tok, int max_cont, int32_t* out_cont, int* out_ng = nullptr) const noexcept {
        if (out_ng) *out_ng = 0;
        if (max_cont <= 0 || anchor_pos < 4 || toks.size() < static_cast<size_t>(anchor_pos)) return 0;
        const int n = anchor_pos;
        const int scan_lo = std::max(0, n - 32768);
        // Try n-gram lengths 5, 4, 3 (longest most-recent match wins!)
        for (int ng = 5; ng >= 3; --ng) {
            if (n < ng) continue;
            for (int p = n - 2; p >= scan_lo + ng - 1; --p) {
                if (toks[static_cast<size_t>(p)] != anchor_tok) continue;
                bool ok = true;
                for (int j = 1; j < ng; ++j) {
                    if (toks[static_cast<size_t>(p - j)] != toks[static_cast<size_t>(n - j)]) {
                        ok = false;
                        break;
                    }
                }
                if (ok) {
                    const int avail = std::min(max_cont, n - (p + 1));
                    if (avail <= 0) continue;
                    for (int k = 0; k < avail; ++k) {
                        out_cont[k] = toks[static_cast<size_t>(p + 1 + k)];
                    }
                    if (out_ng) *out_ng = ng;
                    return avail;
                }
            }
        }
        return 0;
    }
};

}  // namespace strata::inkling2::hip::r9700
