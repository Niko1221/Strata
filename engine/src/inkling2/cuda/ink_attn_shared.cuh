// engine/src/inkling2/cuda/ink_attn_shared.cuh - split-K verify attention that reads every cached K/V cell ONCE for a group of verify rows.
//
// Opt-in (STRATA_INK_VERIFY_SHARED_KV=1|2|4|6|8 = rows per CTA, 1 = 4), default off.  The old k_ink_flash_gqa_splitk launches one CTA per (kv head, split, ROW), so a
// verify step of M rows streams the whole global KV M times.  This kernel launches one CTA per (kv head, split, group of RG rows): each K/V cell is loaded once and
// used by all RG rows of the group (M = 12 -> 2..3 CTAs read it instead of 12).
//
// It is bit-identical to the old kernel by construction.  Per row m the old kernel
//   - cuts the keys [0, qp_m] into n_splits ranges of per_sp_m = ceil((qp_m + 1) / n_splits) keys,
//   - gives virtual warp w (0..3) of split sp the keys s_beg_m + w, + 4, + 8, ... (increasing), starting from the sink (split 0, warp 0) or empty,
//   - takes each key through the same attend3 arithmetic and merges the four warps in the order 0, 1, 2, 3.
// Here a physical warp pw walks the keys u_beg + pw, + 4, ... of the UNION of the group's ranges, and a row only takes the keys inside its own range.  For one row the keys
// of physical warp pw are exactly the keys of its virtual warp vw = (u_beg + pw - s_beg_m) & 3, in the same increasing order, so every accumulator sees the same values
// in the same order with the same rounding; the four accumulators are put back in virtual-warp order for the merge.  The attend arithmetic is the old lambda, expression for
// expression.  Global (non-SWA) visits only; tree rows read their own tree ancestors for kp >= tree_p0 exactly like the old kernel.
#pragma once

#include "ink_attn.cuh"
#include "ink_cuda.cuh"

#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <vector>

namespace strata::ink {

template <int RG>
__global__ void __launch_bounds__(128) k_ink_flash_gqa_splitk_shared(
    const float* __restrict__ q,          // [M, 24, 128]
    const float* __restrict__ rel_table,  // [M, 24, 1025]
    const float* __restrict__ sinks,      // [M, 24]
    InkKvVisitDesc kv,
    int slot0,
    const int32_t* __restrict__ q_pos,    // [M]
    const float* __restrict__ tree_k,     // [M, 8, 128] optional
    const float* __restrict__ tree_v,     // [M, 8, 128] optional
    const uint8_t* __restrict__ tree_anc, // [M, 16]
    float* __restrict__ part_m,           // [M, 24, n_splits]
    float* __restrict__ part_l,           // [M, 24, n_splits]
    float* __restrict__ part_o,           // [M, 24, n_splits, 128]
    int M,
    int n_splits) {
    // 1-D grid with the row group FASTEST: the CTAs that read the same keys (one per row group) are launched next to each other, so they run in step and the L2 serves
    // the later ones (the old kernel's row index is the slowest, so its rows run one after the other and every row re-reads the KV from VRAM or over PCIe)
    const int n_groups = (M + RG - 1) / RG;
    const int g  = blockIdx.x % n_groups;
    const int kh = (blockIdx.x / n_groups) % kInkKVHeads;
    const int sp = blockIdx.x / (n_groups * kInkKVHeads);
    if (sp >= n_splits) return;

    const int tid  = threadIdx.x;
    const int warp = tid >> 5;
    const int lane = tid & 31;
    const float scale = 0.08838834764831845f;
    const int h0 = 3 * kh;
    const int tp0 = tree_k ? q_pos[0] : 0;

    // ---- per-row ranges (the old kernel's, global visit: k_start = 0) ----
    int sb[RG], se[RG], qpr[RG];
    int u_beg = 0x7fffffff, u_end = 0;
#pragma unroll
    for (int i = 0; i < RG; ++i) {
        const int m = g * RG + i;
        if (m < M) {
            const int qp = q_pos[m];
            const int k_end = qp + 1;
            const int k_len = max(0, k_end);
            const int per_sp = (k_len + n_splits - 1) / n_splits;
            sb[i] = sp * per_sp;
            se[i] = min(k_end, sb[i] + per_sp);
            qpr[i] = qp;
            u_beg = min(u_beg, sb[i]);
            u_end = max(u_end, se[i]);
        } else {
            sb[i] = 0x3fffffff;
            se[i] = 0;
            qpr[i] = 0;
        }
    }
    if (u_beg == 0x7fffffff) return;   // no row of this group (cannot happen: grid.z = ceil(M / RG))

    // q of the group's rows lives in shared memory (RG * 3 float4 per lane would not fit the registers of the larger groups)
    __shared__ float4 sq[RG][3][32];
    int rto[RG];
    if (warp == 0) {
#pragma unroll
        for (int i = 0; i < RG; ++i) {
            const int m = min(g * RG + i, M - 1);
            sq[i][0][lane] = *(const float4*)(q + ((size_t)m * kInkHeads + h0 + 0) * kInkHeadDim + lane * 4);
            sq[i][1][lane] = *(const float4*)(q + ((size_t)m * kInkHeads + h0 + 1) * kInkHeadDim + lane * 4);
            sq[i][2][lane] = *(const float4*)(q + ((size_t)m * kInkHeads + h0 + 2) * kInkHeadDim + lane * 4);
        }
    }
#pragma unroll
    for (int i = 0; i < RG; ++i) rto[i] = (min(g * RG + i, M - 1) * kInkHeads + h0) * (kInkRelExtent + 1);
    __syncthreads();

    float mw[RG][3], lw[RG][3], a[RG][3][4];
#pragma unroll
    for (int i = 0; i < RG; ++i) {
        const int m = min(g * RG + i, M - 1);
        const int vw = (warp + u_beg - sb[i]) & 3;   // the virtual warp of the old kernel this physical warp stands for
        const bool first = (sp == 0 && vw == 0);
#pragma unroll
        for (int r = 0; r < 3; ++r) {
            mw[i][r] = first ? sinks[(size_t)m * kInkHeads + h0 + r] : -1e30f;
            lw[i][r] = first ? 1.0f : 0.0f;
            a[i][r][0] = 0.0f; a[i][r][1] = 0.0f; a[i][r][2] = 0.0f; a[i][r][3] = 0.0f;
        }
    }

    for (int kp = u_beg + warp; kp < u_end; kp += 4) {
        const bool from_tree = (tree_k != nullptr) && (kp >= tp0);
        float4 ck4 = make_float4(0.f, 0.f, 0.f, 0.f), cv4 = ck4;
        if (!from_tree) {
            if (kp < 0 || kp >= kv.cap_pos) continue;   // the old kernel skips the key for every row
            ink_kv_load4(kv, slot0, kp, kh, lane, ck4, cv4);
        }
#pragma unroll
        for (int i = 0; i < RG; ++i) {
            if (kp < sb[i] || kp >= se[i]) continue;
            float4 kv4 = ck4, vv4 = cv4;
            if (from_tree) {
                const int m = g * RG + i;
                const int anc_node = (int)tree_anc[m * kInkMaxTreeDepth + (kp - tp0)];
                const size_t tbase = ((size_t)anc_node * kInkKVHeads + kh) * kInkHeadDim + lane * 4;
                kv4 = *(const float4*)(tree_k + tbase);
                vv4 = *(const float4*)(tree_v + tbase);
            }
            const int dist = min(max(qpr[i] - kp, 0), kInkRelExtent);
            // The arithmetic of the old kernel's attend3, with the fused / unfused choices written out (what its compiled code does: x*x, then three FMAs, a * al is a
            // separate multiply and p * v the FMA) so that the contraction the compiler picks for this kernel cannot differ from the old one.
            float sc[3];
#pragma unroll
            for (int r = 0; r < 3; ++r) {
                const float4 q4 = sq[i][r][lane];
                float t = __fmul_rn(q4.x, kv4.x);
                t = __fmaf_rn(q4.y, kv4.y, t);
                t = __fmaf_rn(q4.z, kv4.z, t);
                t = __fmaf_rn(q4.w, kv4.w, t);
                sc[r] = __fmaf_rn(warp_sum(t), scale, rel_table[rto[i] + r * (kInkRelExtent + 1) + dist]);
            }
#pragma unroll
            for (int r = 0; r < 3; ++r) {
                const float mn = fmaxf(mw[i][r], sc[r]);
                const float al = __expf(mw[i][r] - mn);
                const float p  = __expf(sc[r] - mn);
                lw[i][r] = __fmaf_rn(lw[i][r], al, p);
                a[i][r][0] = __fmaf_rn(p, vv4.x, __fmul_rn(a[i][r][0], al));
                a[i][r][1] = __fmaf_rn(p, vv4.y, __fmul_rn(a[i][r][1], al));
                a[i][r][2] = __fmaf_rn(p, vv4.z, __fmul_rn(a[i][r][2], al));
                a[i][r][3] = __fmaf_rn(p, vv4.w, __fmul_rn(a[i][r][3], al));
                mw[i][r] = mn;
            }
        }
    }

    // ---- merge the four (virtual) warps of each row in the old order and write the partials ----
    __shared__ float sm_m[4], sm_l[4], sm_a[4][kInkHeadDim];
#pragma unroll
    for (int i = 0; i < RG; ++i) {
        const int m = g * RG + i;
        const int vw = (warp + u_beg - sb[i]) & 3;
#pragma unroll
        for (int r = 0; r < 3; ++r) {
            if (lane == 0) { sm_m[vw] = mw[i][r]; sm_l[vw] = lw[i][r]; }
            sm_a[vw][lane * 4 + 0] = a[i][r][0];
            sm_a[vw][lane * 4 + 1] = a[i][r][1];
            sm_a[vw][lane * 4 + 2] = a[i][r][2];
            sm_a[vw][lane * 4 + 3] = a[i][r][3];
            __syncthreads();
            if (warp == 0 && m < M) {
                float mg = fmaxf(fmaxf(sm_m[0], sm_m[1]), fmaxf(sm_m[2], sm_m[3]));
                float w0 = sm_l[0] > 0.0f ? __expf(sm_m[0] - mg) : 0.0f;
                float w1 = sm_l[1] > 0.0f ? __expf(sm_m[1] - mg) : 0.0f;
                float w2 = sm_l[2] > 0.0f ? __expf(sm_m[2] - mg) : 0.0f;
                float w3 = sm_l[3] > 0.0f ? __expf(sm_m[3] - mg) : 0.0f;
                // the old merge: X0 * w0 + X1 * w1 + X2 * w2 + X3 * w3 is compiled as fma(X3, w3, fma(X2, w2, fma(X0, w0, X1 * w1)))
                const float lt = __fmaf_rn(sm_l[3], w3, __fmaf_rn(sm_l[2], w2, __fmaf_rn(sm_l[0], w0, __fmul_rn(sm_l[1], w1))));
                const size_t idx_sp = ((size_t)m * kInkHeads + h0 + r) * n_splits + sp;
                if (lane == 0) { part_m[idx_sp] = mg; part_l[idx_sp] = lt; }
                float o4[4];
#pragma unroll
                for (int c = 0; c < 4; ++c)
                    o4[c] = __fmaf_rn(sm_a[3][lane * 4 + c], w3, __fmaf_rn(sm_a[2][lane * 4 + c], w2, __fmaf_rn(sm_a[0][lane * 4 + c], w0, __fmul_rn(sm_a[1][lane * 4 + c], w1))));
                *(float4*)(part_o + idx_sp * kInkHeadDim + lane * 4) = make_float4(o4[0], o4[1], o4[2], o4[3]);
            }
            __syncthreads();
        }
    }
}

// ---------------------------------------------------------------------------------------------------------------------------------------------------------------
// Switch, dispatch and start-up self check.
//
// STRATA_INK_VERIFY_SHARED_KV   unset / 0 = off (default; the old kernel runs, nothing else changes)
//                               1         = on for the global visits whose K/V has a host tail (host_cap > 0): there every verify row re-reads the tail over PCIe
//                               all       = on for every global visit (A/B only: with the whole K/V in VRAM the old kernel is as fast or faster)
// STRATA_INK_VERIFY_SHARED_ROWS rows per CTA, 1 / 2 / 4 / 6 / 8 (default 8): a step of M rows reads each cell ceil(M / rows) times instead of M times
//
// On first use (the runner's init calls ink_verify_shared_cfg()) the kernel is checked against the old one on synthetic data; a single differing bit switches the
// feature off again with a message, so "on" can never mean "different answers" on a compiler / card where the two would not round the same way.
// ---------------------------------------------------------------------------------------------------------------------------------------------------------------
inline bool ink_launch_splitk_shared_raw(
    cudaStream_t st, int rg, const float* q, const float* rel_table, const float* sinks, const InkKvVisitDesc& kv, int slot0, const int32_t* q_pos,
    const float* tree_k, const float* tree_v, const uint8_t* tree_anc, float* part_m, float* part_l, float* part_o, int M, int n_splits);

__global__ void k_shared_selfcheck_fill(float* p, size_t n, uint32_t seed, float amp) {
    for (size_t i = blockIdx.x * (size_t)blockDim.x + threadIdx.x; i < n; i += (size_t)gridDim.x * blockDim.x) {
        uint64_t x = (uint64_t)i * 0x9E3779B97F4A7C15ull + seed * 0xBF58476D1CE4E5B9ull;
        x = (x ^ (x >> 30)) * 0xBF58476D1CE4E5B9ull;
        x = (x ^ (x >> 27)) * 0x94D049BB133111EBull;
        p[i] = (((uint32_t)((x ^ (x >> 31)) >> 40) & 0xFFFFFF) * (1.0f / 16777216.0f) - 0.5f) * amp;
    }
}
__global__ void k_shared_selfcheck_f16(__half* p, const float* src, size_t n) {
    for (size_t i = blockIdx.x * (size_t)blockDim.x + threadIdx.x; i < n; i += (size_t)gridDim.x * blockDim.x) p[i] = __float2half_rn(src[i]);
}

inline bool ink_shared_selfcheck(int rg) {
    constexpr int kCap = 6000, kM = 8, kSp = kInkAttnMaxSplits, kTp0 = 5000;
    const size_t kv_n = (size_t)kCap * kInkKVHeads * kInkHeadDim, np = (size_t)kM * kInkHeads * kSp;
    float *fk = nullptr, *fv = nullptr, *q = nullptr, *rel = nullptr, *sinks = nullptr, *tk = nullptr, *tv = nullptr, *pm[2] = {}, *pl[2] = {}, *po[2] = {};
    __half *hk = nullptr, *hv = nullptr;
    int32_t* pos = nullptr;
    uint8_t *anc = nullptr, *dep = nullptr;
    bool ok = true;
    ok = ok && cudaMalloc(&fk, kv_n * 4) == cudaSuccess && cudaMalloc(&fv, kv_n * 4) == cudaSuccess && cudaMalloc(&hk, kv_n * 2) == cudaSuccess && cudaMalloc(&hv, kv_n * 2) == cudaSuccess;
    ok = ok && cudaMalloc(&q, (size_t)kM * kInkHeads * kInkHeadDim * 4) == cudaSuccess && cudaMalloc(&rel, (size_t)kM * kInkHeads * (kInkRelExtent + 1) * 4) == cudaSuccess;
    ok = ok && cudaMalloc(&sinks, (size_t)kM * kInkHeads * 4) == cudaSuccess && cudaMalloc(&tk, (size_t)kM * kInkKVHeads * kInkHeadDim * 4) == cudaSuccess;
    ok = ok && cudaMalloc(&tv, (size_t)kM * kInkKVHeads * kInkHeadDim * 4) == cudaSuccess && cudaMalloc(&pos, kM * 4) == cudaSuccess;
    ok = ok && cudaMalloc(&anc, 32 * kInkMaxTreeDepth) == cudaSuccess && cudaMalloc(&dep, 32) == cudaSuccess;
    for (int i = 0; i < 2 && ok; ++i)
        ok = ok && cudaMalloc(&pm[i], np * 4) == cudaSuccess && cudaMalloc(&pl[i], np * 4) == cudaSuccess && cudaMalloc(&po[i], np * kInkHeadDim * 4) == cudaSuccess;
    bool same = false;
    if (ok) {
        k_shared_selfcheck_fill<<<256, 256>>>(fk, kv_n, 1, 6.0f);
        k_shared_selfcheck_fill<<<256, 256>>>(fv, kv_n, 2, 4.0f);
        k_shared_selfcheck_f16<<<256, 256>>>(hk, fk, kv_n);
        k_shared_selfcheck_f16<<<256, 256>>>(hv, fv, kv_n);
        k_shared_selfcheck_fill<<<256, 256>>>(q, (size_t)kM * kInkHeads * kInkHeadDim, 3, 6.0f);
        k_shared_selfcheck_fill<<<256, 256>>>(rel, (size_t)kM * kInkHeads * (kInkRelExtent + 1), 4, 4.0f);
        k_shared_selfcheck_fill<<<8, 256>>>(sinks, (size_t)kM * kInkHeads, 5, 8.0f);
        k_shared_selfcheck_fill<<<64, 256>>>(tk, (size_t)kM * kInkKVHeads * kInkHeadDim, 6, 4.0f);
        k_shared_selfcheck_fill<<<64, 256>>>(tv, (size_t)kM * kInkKVHeads * kInkHeadDim, 7, 4.0f);
        int32_t hp[kM];
        uint8_t ha[32 * kInkMaxTreeDepth] = {}, hd[32] = {};
        for (int i = 0; i < kM; ++i) {   // a chain of 8 nodes: node i sees ancestors 0..i
            hp[i] = kTp0 + i;
            hd[i] = (uint8_t)i;
            for (int d = 0; d <= i && d < kInkMaxTreeDepth; ++d) ha[i * kInkMaxTreeDepth + d] = (uint8_t)d;
        }
        cudaMemcpy(pos, hp, sizeof(hp), cudaMemcpyHostToDevice);
        cudaMemcpy(anc, ha, sizeof(ha), cudaMemcpyHostToDevice);
        cudaMemcpy(dep, hd, sizeof(hd), cudaMemcpyHostToDevice);
        InkKvVisitDesc kv{};
        kv.k_f16 = hk; kv.v_f16 = hv; kv.cap_pos = kCap; kv.dev_cap = kCap; kv.host_cap = 0; kv.is_swa = false; kv.kv_fmt = 0;
        cudaMemset(pm[0], 0xFF, np * 4); cudaMemset(pl[0], 0xFF, np * 4); cudaMemset(po[0], 0xFF, np * kInkHeadDim * 4);
        cudaMemset(pm[1], 0x00, np * 4); cudaMemset(pl[1], 0x00, np * 4); cudaMemset(po[1], 0x00, np * kInkHeadDim * 4);
        k_ink_flash_gqa_splitk<<<dim3(kInkKVHeads, kSp, kM), 128>>>(q, rel, sinks, kv, nullptr, 0, pos, tk, tv, anc, dep, 0, pm[0], pl[0], po[0], kM, kSp);
        const bool launched = ink_launch_splitk_shared_raw(0, rg, q, rel, sinks, kv, 0, pos, tk, tv, anc, pm[1], pl[1], po[1], kM, kSp);
        if (launched && cudaDeviceSynchronize() == cudaSuccess) {
            std::vector<float> a(np), b(np), c(np), d(np), e(np * kInkHeadDim), f(np * kInkHeadDim);
            cudaMemcpy(a.data(), pm[0], np * 4, cudaMemcpyDeviceToHost); cudaMemcpy(b.data(), pm[1], np * 4, cudaMemcpyDeviceToHost);
            cudaMemcpy(c.data(), pl[0], np * 4, cudaMemcpyDeviceToHost); cudaMemcpy(d.data(), pl[1], np * 4, cudaMemcpyDeviceToHost);
            cudaMemcpy(e.data(), po[0], np * kInkHeadDim * 4, cudaMemcpyDeviceToHost); cudaMemcpy(f.data(), po[1], np * kInkHeadDim * 4, cudaMemcpyDeviceToHost);
            same = std::memcmp(a.data(), b.data(), np * 4) == 0 && std::memcmp(c.data(), d.data(), np * 4) == 0 && std::memcmp(e.data(), f.data(), np * kInkHeadDim * 4) == 0;
        }
    }
    cudaGetLastError();
    for (void* p : {(void*)fk, (void*)fv, (void*)hk, (void*)hv, (void*)q, (void*)rel, (void*)sinks, (void*)tk, (void*)tv, (void*)pos, (void*)anc, (void*)dep,
                    (void*)pm[0], (void*)pm[1], (void*)pl[0], (void*)pl[1], (void*)po[0], (void*)po[1]})
        if (p) cudaFree(p);
    return ok && same;
}

struct InkSharedCfg {
    int rg = 0;         // 0 = off
    bool all = false;   // also the visits whose K/V is entirely in VRAM
};

inline const InkSharedCfg& ink_verify_shared_cfg() {
    static const InkSharedCfg cfg = [] {
        InkSharedCfg c;
        const char* e = std::getenv("STRATA_INK_VERIFY_SHARED_KV");
        if (!e || !e[0] || e[0] == '0') return c;
        c.all = (e[0] == 'a' || e[0] == 'A');
        int rg = 8;
        if (const char* r = std::getenv("STRATA_INK_VERIFY_SHARED_ROWS")) rg = std::atoi(r);
        if (!(rg == 1 || rg == 2 || rg == 4 || rg == 6 || rg == 8)) rg = 8;
        if (!ink_shared_selfcheck(rg)) {
            std::fprintf(stderr, "[InkCUDA] STRATA_INK_VERIFY_SHARED_KV: the one-read verify kernel does not reproduce the old kernel's bits on this build / card: left OFF\n");
            return c;
        }
        c.rg = rg;
        std::fprintf(stderr, "[InkCUDA] STRATA_INK_VERIFY_SHARED_KV: verify steps read each KV cell once per group of %d rows (%s); bit-identical to the old kernel (start-up check passed)\n",
                     rg, c.all ? "every global visit" : "global visits with a host tail");
        return c;
    }();
    return cfg;
}

// Launch the shared kernel for the given row-group size.  Returns false when it does not apply (the caller launches the old kernel).
inline bool ink_launch_splitk_shared_raw(
    cudaStream_t st, int rg, const float* q, const float* rel_table, const float* sinks, const InkKvVisitDesc& kv, int slot0, const int32_t* q_pos,
    const float* tree_k, const float* tree_v, const uint8_t* tree_anc, float* part_m, float* part_l, float* part_o, int M, int n_splits) {
    if (rg <= 0 || kv.is_swa || M < 2) return false;
#define INK_SHARED_CASE(R)                                                                                                       \
    case R: {                                                                                                                    \
        const unsigned grid = (unsigned)(kInkKVHeads * n_splits * ((M + R - 1) / R));                                            \
        k_ink_flash_gqa_splitk_shared<R><<<grid, 128, 0, st>>>(q, rel_table, sinks, kv, slot0, q_pos, tree_k, tree_v, tree_anc, \
                                                               part_m, part_l, part_o, M, n_splits);                             \
        return true;                                                                                                             \
    }
    switch (rg) {
        INK_SHARED_CASE(1)
        INK_SHARED_CASE(2)
        INK_SHARED_CASE(4)
        INK_SHARED_CASE(6)
        INK_SHARED_CASE(8)
        default: return false;
    }
#undef INK_SHARED_CASE
}

// The runner's call (verify step, global visit): applies the switch and the host-tail rule.
inline bool ink_launch_splitk_shared(
    cudaStream_t st, const float* q, const float* rel_table, const float* sinks, const InkKvVisitDesc& kv, int slot0, const int32_t* q_pos,
    const float* tree_k, const float* tree_v, const uint8_t* tree_anc, float* part_m, float* part_l, float* part_o, int M, int n_splits) {
    const InkSharedCfg& c = ink_verify_shared_cfg();
    if (c.rg <= 0 || (!c.all && kv.host_cap <= 0)) return false;
    return ink_launch_splitk_shared_raw(st, c.rg, q, rel_table, sinks, kv, slot0, q_pos, tree_k, tree_v, tree_anc, part_m, part_l, part_o, M, n_splits);
}

}  // namespace strata::ink
