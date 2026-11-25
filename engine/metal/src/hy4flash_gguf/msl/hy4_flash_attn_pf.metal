// engine/metal/src/hy4flash_gguf/msl/hy4_flash_attn_pf.metal - the MLA attention and the indexer score of a PREFILL chunk on simdgroup
// matrices (MMA is allowed for prefill only, with fp32 operands and fp32 accumulation; decode is untouched).
//
// The same attention as hy4_flash_attn.metal's three dispatches - one shared 576-wide key per position, V = K[0:512], a
// sink that joins the maximum and the denominator, a sliding range or the indexer's selection - but NOT bitwise with
// them: the dots and the PV sums are 8x8 matrix products (another summation order) and the softmax is ONLINE (a
// running maximum and sum, the partial output rescaled when the maximum moves).  The prefill tolerance allows both
// (prefill-consistency is a KL / top-1 judgement for prefill).
//
// PRECISION.  Every matrix operand is fp32 and every product accumulates in fp32: the cache's half K / V rows are
// widened to fp32 (exactly) when they are staged in threadgroup memory, Q and the weights P are fp32.  Nothing is
// rounded to a narrower type.
//
// TILING (hy4_flash_mla_pf5, NSG = 32 simdgroups = 1024 threads).  One threadgroup = one position t and 32 heads (they
// share the key rows, MLA).  A tile is 64 keys, gathered through `sel` for a sparse block:
//   S[32 x 64]   per 64-dim chunk, Q (fp32) and K (the cache's halves) are staged and one 8x8 score block a
//                simdgroup is accumulated over all 576 dims
//   softmax      online, NSG threads a head row, 64 / NSG keys each
//   O += P . V   in sub-tiles of 16 keys; simdgroup g owns output dims (512 / NSG) * g and holds 4 x (64 / NSG)
//                accumulators, rescaled by diag(corr) first when one of its rows' maximum moved
// At the end O is stored and each element DIVIDED by its row's sum.
// grid (n_tok, ceil(n_head / 32)), threads 1024.  Requires d_qk == 576, d_v == 512, kv_stride % 4 == 0.
//
// WHY 32 SIMDGROUPS: the accumulators one simdgroup holds scale as 1 / NSG, so with 8 (256 threads) a simdgroup
// holds 32 of them and spills - the cost of that lands on the PV phase.

#include <metal_simdgroup_matrix>

#define HY4_FLASH_PF_HB 32
#define HY4_FLASH_PF_KT 8
#define HY4_FLASH_PF_DQK 576
#define HY4_FLASH_PF_DV 512

// ---- the attention -------------------------------------------------------------------------------------------------
template <int NSG>
kernel void hy4_flash_mla_pf5(device const float* q [[buffer(0)]], device const half* kv [[buffer(1)]],
                        device const float* sinks [[buffer(2)]], device const int* sel [[buffer(3)]],
                        device const int* n_sel [[buffer(4)]], device const int* pos [[buffer(5)]],
                        device float* o [[buffer(6)]], constant Hy4FlashAttnArgs& a [[buffer(7)]],
                        uint3 tgp [[threadgroup_position_in_grid]], ushort tid [[thread_index_in_threadgroup]],
                        ushort sg [[simdgroup_index_in_threadgroup]], ushort lane [[thread_index_in_simdgroup]]) {
    constexpr int KT = 64, DC = 64, NT = NSG * 32;
    constexpr int VB = 64 / NSG;          // 8-dim output blocks a simdgroup
    constexpr int SB = 32 / NSG;          // score blocks a simdgroup
    constexpr int TPR = NSG;              // softmax threads a head row
    constexpr int KPT = KT / TPR;         // keys a softmax thread
    threadgroup float st[HY4_FLASH_PF_HB * KT];
    threadgroup uchar stage[16384];
    threadgroup float* qs = (threadgroup float*)stage;
    threadgroup half* ks = (threadgroup half*)(stage + 8192);
    threadgroup half* vs = (threadgroup half*)stage;
    threadgroup float dg[4 * 64];
    threadgroup float rm[HY4_FLASH_PF_HB], rl[HY4_FLASH_PF_HB];
    threadgroup int kid[KT];
    threadgroup int moved[HY4_FLASH_PF_HB];   // per ROW: the row's own softmax leader writes it, so no two
                                       // simdgroups write one word (a head row is one simdgroup here)

    const uint t = tgp.x;
    const uint h0 = tgp.y * HY4_FLASH_PF_HB;
    const int p = pos[t];
    const int lo = hy4_flash_attn_sel ? 0 : (a.window != 0 ? max(0, p - int(a.window) + 1) : 0);
    const int n = hy4_flash_attn_sel ? n_sel[t] : p - lo + 1;
    device const int* sr = sel + (ulong)t * a.sel_stride;
    device const float* qt = q + (ulong)t * a.q_stride + (ulong)h0 * HY4_FLASH_PF_DQK;
    const uint nh = min(uint(HY4_FLASH_PF_HB), a.n_head - h0);

    if (tid < HY4_FLASH_PF_HB) {
        rm[tid] = tid < nh ? sinks[h0 + tid] : 0.0f;
        rl[tid] = 1.0f;
    }
    for (uint i = tid; i < 4 * 64; i += NT) dg[i] = 0.0f;
    if (tid < HY4_FLASH_PF_HB) moved[tid] = 0;

    simdgroup_float8x8 acc[4][VB];
    for (int i = 0; i < 4; ++i)
        for (int j = 0; j < VB; ++j) acc[i][j] = simdgroup_float8x8(0.0f);
    const uint dv0 = (512 / NSG) * sg;

    for (int j0 = 0; j0 < n; j0 += KT) {
        threadgroup_barrier(mem_flags::mem_threadgroup);
        if (tid < uint(KT)) {
            const int j = j0 + int(tid);
            kid[tid] = j < n ? (hy4_flash_attn_sel ? sr[j] : lo + j) : -1;
        }
        simdgroup_float8x8 s[SB];
        for (int b = 0; b < SB; ++b) s[b] = simdgroup_float8x8(0.0f);
        for (int d0 = 0; d0 < HY4_FLASH_PF_DQK; d0 += DC) {
            threadgroup_barrier(mem_flags::mem_threadgroup);
            for (uint i = tid; i < HY4_FLASH_PF_HB * DC / 4; i += NT) {
                const uint r = i / (DC / 4), c = i % (DC / 4);
                *((threadgroup float4*)(qs + r * DC) + c) =
                    r < nh ? *((device const float4*)(qt + (ulong)r * HY4_FLASH_PF_DQK + d0) + c) : float4(0.0f);
            }
            for (uint i = tid; i < KT * DC / 4; i += NT) {
                const uint r = i / (DC / 4), c = i % (DC / 4);
                const int cell = kid[r];
                *((threadgroup half4*)(ks + r * DC) + c) =
                    cell >= 0 ? *((device const half4*)(kv + (ulong)cell * a.kv_stride + d0) + c) : half4(0.0h);
            }
            threadgroup_barrier(mem_flags::mem_threadgroup);
            for (int b = 0; b < SB; ++b) {
                const int blk = sg * SB + b, hb = blk / 8, kb = blk % 8;
                for (int d = 0; d < DC; d += 8) {
                    simdgroup_float8x8 qa;
                    simdgroup_half8x8 k8;
                    simdgroup_load(qa, qs + (8 * hb) * DC + d, DC);
                    simdgroup_load(k8, ks + (8 * kb) * DC + d, DC, ulong2(0, 0), true);
                    simdgroup_multiply_accumulate(s[b], qa, k8, s[b]);
                }
            }
        }
        for (int b = 0; b < SB; ++b) {
            const int blk = sg * SB + b, hb = blk / 8, kb = blk % 8;
            simdgroup_store(s[b], st + (8 * hb) * KT + 8 * kb, KT);
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
        {   // online softmax: TPR consecutive lanes a head row (TPR <= 32: one simdgroup), KPT keys each
            const int r = int(tid) / TPR, sub = int(tid) % TPR;
            const float m = rm[r];
            float sv[KPT];
            float mx = m;
            for (int i = 0; i < KPT; ++i) {
                const int k = sub * KPT + i;
                sv[i] = kid[k] >= 0 ? st[r * KT + k] * a.kq_scale : -INFINITY;
                mx = max(mx, sv[i]);
            }
            for (ushort off = TPR / 2; off > 0; off >>= 1) mx = max(mx, simd_shuffle_xor(mx, off));
            float ls = 0.0f;
            for (int i = 0; i < KPT; ++i) {
                const float e = precise::exp(sv[i] - mx);
                st[r * KT + sub * KPT + i] = e;
                ls += e;
            }
            for (ushort off = TPR / 2; off > 0; off >>= 1) ls += simd_shuffle_xor(ls, off);
            simdgroup_barrier(mem_flags::mem_threadgroup);
            if (sub == 0) {
                const float corr = precise::exp(m - mx);
                rl[r] = rl[r] * corr + ls;
                rm[r] = mx;
                dg[(r / 8) * 64 + (r % 8) * 9] = corr;
                moved[r] = mx != m ? 1 : 0;
            }
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
        for (int i = 0; i < 4; ++i) {
            int mv = 0;   // any of this 8-row block's maxima moved
            for (int r = 0; r < 8; ++r) mv |= moved[8 * i + r];
            if (mv) {
                simdgroup_float8x8 dm;
                simdgroup_load(dm, dg + i * 64, 8);
                for (int j = 0; j < VB; ++j) {
                    simdgroup_float8x8 tmp;
                    simdgroup_multiply(tmp, dm, acc[i][j]);
                    acc[i][j] = tmp;
                }
            }
        }
        for (int k0 = 0; k0 < KT; k0 += 16) {
            if (j0 + k0 >= n) break;
            threadgroup_barrier(mem_flags::mem_threadgroup);
            for (uint i = tid; i < 16 * (HY4_FLASH_PF_DV / 4); i += NT) {
                const uint r = i / (HY4_FLASH_PF_DV / 4), c = i % (HY4_FLASH_PF_DV / 4);
                const int cell = kid[k0 + r];
                *((threadgroup half4*)(vs + r * HY4_FLASH_PF_DV) + c) =
                    cell >= 0 ? *((device const half4*)(kv + (ulong)cell * a.kv_stride) + c) : half4(0.0h);
            }
            threadgroup_barrier(mem_flags::mem_threadgroup);
            for (int kk = 0; kk < 16; kk += 8)
                for (int i = 0; i < 4; ++i) {
                    simdgroup_float8x8 pm;
                    simdgroup_load(pm, st + (8 * i) * KT + k0 + kk, KT);
                    for (int j = 0; j < VB; ++j) {
                        simdgroup_half8x8 vm;
                        simdgroup_load(vm, vs + kk * HY4_FLASH_PF_DV + dv0 + 8 * j, HY4_FLASH_PF_DV);
                        simdgroup_multiply_accumulate(acc[i][j], pm, vm, acc[i][j]);
                    }
                }
        }
    }
    device float* ot = o + (ulong)t * a.o_stride + (ulong)h0 * HY4_FLASH_PF_DV;
    // a simdgroup_store writes a FULL 8 x 8 block, so a head block that is only partly inside this block's heads
    // would write the next token's rows (o is [token][head][d_v]).  The host only dispatches head counts that are a
    // multiple of 32 (forward.cpp), so nh is 32 here; the guard refuses anything else rather than corrupting.
    for (int i = 0; i < 4; ++i)
        for (int j = 0; j < VB; ++j)
            if (uint(8 * i + 8) <= nh) simdgroup_store(acc[i][j], ot + (ulong)(8 * i) * HY4_FLASH_PF_DV + dv0 + 8 * j, HY4_FLASH_PF_DV);
    threadgroup_barrier(mem_flags::mem_device | mem_flags::mem_threadgroup);
    for (uint i = tid; i < HY4_FLASH_PF_HB * HY4_FLASH_PF_DV; i += NT) {
        const uint r = i / HY4_FLASH_PF_DV;
        if (r < nh) ot[i] = ot[i] / rl[r];
    }
}
template [[host_name("hy4_flash_mla_pf5_s32")]] kernel decltype(hy4_flash_mla_pf5<32>) hy4_flash_mla_pf5<32>;

// ---- the indexer's score for a prefill chunk, on simdgroup matrices ------------------------------------------------
// score[t][g] = sum_h max(iq[t][h] . pool[g], 0) * w[t][h]   (-inf for a pool that ends past the token's position)
// The function of hy4_flash_indexer_score (hy4_flash_indexer.metal), not bitwise with it: the dots are 8x8 matrix products of fp32
// operands (the pooled keys widened from the cache's halves, exactly) with fp32 accumulation; the head sum runs over h
// ascending in one thread.  One threadgroup = 4 tokens (64 (token, head) rows) x 32 pools, 8 simdgroups with 4 of the
// 8 x 4 score blocks each.  grid (ceil(n_pool / 32), ceil(n_tok / 4)), threads 256.
// Requires n_head == 16 and dim == 128 (the staged pool tile is 32 x 128).
kernel void hy4_flash_indexer_score_pf(device const float* iq [[buffer(0)]], device const half* pooled [[buffer(1)]],
                                 device const float* wgt [[buffer(2)]], device const int* pos [[buffer(3)]],
                                 device float* score [[buffer(4)]], constant Hy4FlashScoreArgs& a [[buffer(5)]],
                                 uint3 tgp [[threadgroup_position_in_grid]], ushort tid [[thread_index_in_threadgroup]],
                                 ushort sg [[simdgroup_index_in_threadgroup]]) {
    threadgroup float pt[32 * 128];   // 16 KB: the pool tile, fp32
    threadgroup float s[64 * 32];     // 8 KB
    const uint t0 = tgp.y * 4, g0 = tgp.x * 32;
    for (uint i = tid; i < 32 * 32; i += 256) {   // 32 pools x 32 half4
        const uint r = i / 32, c = i % 32;
        const uint g = g0 + r;
        float4 v = float4(0.0f);
        if (g < a.n_pool) v = float4(*((device const half4*)(pooled + (ulong)g * a.pool_stride) + c));
        *((threadgroup float4*)(pt + r * 128) + c) = v;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    for (int blk = sg; blk < 32; blk += 8) {
        const int rb = blk / 4, cb = blk % 4;   // (token, head) rows 8 rb .. +7 - one token's heads -, pools 8 cb .. +7
        const uint tq = min(t0 + uint(rb) / 2, a.n_tok - 1);   // a token past the end reads the last one, dropped below
        device const float* qa = iq + (ulong)tq * a.q_stride + (uint(rb) % 2) * 8 * a.dim;
        simdgroup_float8x8 acc(0.0f);
        for (uint d = 0; d < 128; d += 8) {
            simdgroup_float8x8 qm, pm;
            simdgroup_load(qm, qa + d, a.dim);
            simdgroup_load(pm, pt + (8 * cb) * 128 + d, 128, ulong2(0, 0), true);
            simdgroup_multiply_accumulate(acc, qm, pm, acc);
        }
        simdgroup_store(acc, s + (8 * rb) * 32 + 8 * cb, 32);
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (tid < 128) {
        const uint tl = tid / 32, gl = tid % 32;
        const uint t = t0 + tl, g = g0 + gl;
        if (t < a.n_tok && g < a.n_pool) {
            device const float* wr = wgt + (ulong)t * a.w_stride;
            float tot = 0.0f;
            for (uint h = 0; h < 16; ++h) tot += max(s[(16 * tl + h) * 32 + gl], 0.0f) * wr[h];
            const int pool_end = int(a.kpool) * int(g) + int(a.kpool) - 1;
            score[(ulong)t * a.s_stride + g] = pool_end <= pos[t] ? tot : -INFINITY;
        }
    }
}

