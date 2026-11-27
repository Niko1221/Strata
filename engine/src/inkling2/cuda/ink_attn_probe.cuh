// engine/src/inkling2/cuda/ink_attn_probe.cuh - DEBUG ONLY: how much of the softmax mass of a global-layer decode step sits in the top-k KV blocks.
//
// Off unless STRATA_INK_ATTN_PROBE=<output.jsonl> is set; it never changes an output (it reads q / rel table / sinks / K / V after the real attention ran,
// writes only its own scratch buffers, and synchronises the stream, so it must run with STRATA_INK_NO_GRAPH=1 / --no-graph).
//
// For every probed (decode step, global visit) it recomputes, in FP32 from the KV cache, the 24 heads' scores (q.k * 1/sqrt(128) + rel bias), the softmax
// with the per-query sink, the mass of every 64-token block, and the Quest bound of every 64 / 128-token block (sum_d max(q_d kmin_d, q_d kmax_d) *
// scale + the largest rel bias in the block).  Then, on the host, for budgets 1K / 2K / 4K / 8K cells and block sizes 64 / 128 and three selectors
//   oracle  : the blocks with the largest true mass,
//   quest   : the blocks with the largest Quest bound,
//   questf  : block 0 and the last two blocks forced, the rest by Quest bound (budget counts the forced blocks),
// it reports per head: the captured share of the KV attention mass, the relative L2 change of the head's attention output (the selected blocks only,
// softmax renormalised over them + the sink), and, for a needle at token positions lo:hi, the needle's own mass and the part of it inside the selection.
//
// Env: STRATA_INK_ATTN_PROBE=<file>            output (JSON lines), appended
//      STRATA_INK_ATTN_PROBE_EVERY=N           probe every Nth decode step (default 1)
//      STRATA_INK_ATTN_PROBE_MAXREC=N          stop after N records (default 4096)
//      STRATA_INK_ATTN_PROBE_MINCTX=N          skip steps with fewer than N cached positions (default 1024)
//      STRATA_INK_ATTN_PROBE_NEEDLE=lo:hi      needle token positions
//      STRATA_INK_ATTN_PROBE_DUMP=<dir>        also write the raw q / rel / sink / gate / K / V (fp32) of the FIRST record of one visit and its masks / outputs,
//      STRATA_INK_ATTN_PROBE_DUMP_VISIT=v      so a host script can recompute the numbers independently (default visit 47)
#pragma once

#include "ink_attn.cuh"
#include "ink_cuda.cuh"

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <numeric>
#include <string>
#include <vector>

namespace strata::ink {

constexpr int kProbeBlk = 64;   // finest block; 128-blocks are pairs

// scores[h][j] = q_h . k_j / sqrt(128) + rel_h[dist]       grid (8 kv heads, ceil(L/64)), block 128
__global__ void __launch_bounds__(128) k_probe_scores(
    const float* __restrict__ q, const float* __restrict__ rel, InkKvVisitDesc kv, int sl, int qp, int L, float* __restrict__ scores) {
    const int kh = blockIdx.x, warp = threadIdx.x >> 5, lane = threadIdx.x & 31;
    const int h0 = 3 * kh;
    const float4 qv0 = *(const float4*)(q + (size_t)(h0 + 0) * kInkHeadDim + lane * 4);
    const float4 qv1 = *(const float4*)(q + (size_t)(h0 + 1) * kInkHeadDim + lane * 4);
    const float4 qv2 = *(const float4*)(q + (size_t)(h0 + 2) * kInkHeadDim + lane * 4);
    const float scale = 0.08838834764831845f;
    for (int i = 0; i < 16; ++i) {
        const int kp = blockIdx.y * 64 + warp + 4 * i;
        if (kp >= L) break;
        float4 k4, v4;
        ink_kv_load4(kv, sl, kp, kh, lane, k4, v4);
        const int dist = min(max(qp - kp, 0), kInkRelExtent);
        const float s0 = warp_sum(qv0.x * k4.x + qv0.y * k4.y + qv0.z * k4.z + qv0.w * k4.w) * scale + rel[(size_t)(h0 + 0) * (kInkRelExtent + 1) + dist];
        const float s1 = warp_sum(qv1.x * k4.x + qv1.y * k4.y + qv1.z * k4.z + qv1.w * k4.w) * scale + rel[(size_t)(h0 + 1) * (kInkRelExtent + 1) + dist];
        const float s2 = warp_sum(qv2.x * k4.x + qv2.y * k4.y + qv2.z * k4.z + qv2.w * k4.w) * scale + rel[(size_t)(h0 + 2) * (kInkRelExtent + 1) + dist];
        if (lane == 0) {
            scores[(size_t)(h0 + 0) * L + kp] = s0;
            scores[(size_t)(h0 + 1) * L + kp] = s1;
            scores[(size_t)(h0 + 2) * L + kp] = s2;
        }
    }
}

// per head: the max of (scores, sink) and Z = e^(sink-max) + sum e^(s-max)     grid 24, block 256
__global__ void __launch_bounds__(256) k_probe_stats(
    const float* __restrict__ scores, const float* __restrict__ sinks, int L, float* __restrict__ mx_out, float* __restrict__ z_out) {
    const int h = blockIdx.x, t = threadIdx.x;
    __shared__ float red[256];
    float mx = sinks[h];
    for (int j = t; j < L; j += 256) mx = fmaxf(mx, scores[(size_t)h * L + j]);
    red[t] = mx;
    __syncthreads();
    for (int s = 128; s > 0; s >>= 1) { if (t < s) red[t] = fmaxf(red[t], red[t + s]); __syncthreads(); }
    mx = red[0];
    __syncthreads();
    float sum = 0.0f;
    for (int j = t; j < L; j += 256) sum += expf(scores[(size_t)h * L + j] - mx);
    red[t] = sum;
    __syncthreads();
    for (int s = 128; s > 0; s >>= 1) { if (t < s) red[t] += red[t + s]; __syncthreads(); }
    if (t == 0) { mx_out[h] = mx; z_out[h] = red[0] + expf(sinks[h] - mx); }
}

// per (kv head, block of B tokens): mass[h][b] (softmax share of the 3 heads) and Quest bound[h][b]     grid (8, ceil(L/B)), block 128
__global__ void __launch_bounds__(128) k_probe_blocks(
    const float* __restrict__ q, const float* __restrict__ rel, const float* __restrict__ scores, const float* __restrict__ mx, const float* __restrict__ z,
    InkKvVisitDesc kv, int sl, int qp, int L, int B, int nb, float* __restrict__ mass, float* __restrict__ bound) {
    const int kh = blockIdx.x, b = blockIdx.y, t = threadIdx.x, warp = t >> 5, lane = t & 31;
    const int p0 = b * B, p1 = min(L, p0 + B);
    float mn[4] = {INFINITY, INFINITY, INFINITY, INFINITY}, mxv[4] = {-INFINITY, -INFINITY, -INFINITY, -INFINITY};
    for (int kp = p0 + warp; kp < p1; kp += 4) {
        float4 k4, v4;
        ink_kv_load4(kv, sl, kp, kh, lane, k4, v4);
        mn[0] = fminf(mn[0], k4.x); mn[1] = fminf(mn[1], k4.y); mn[2] = fminf(mn[2], k4.z); mn[3] = fminf(mn[3], k4.w);
        mxv[0] = fmaxf(mxv[0], k4.x); mxv[1] = fmaxf(mxv[1], k4.y); mxv[2] = fmaxf(mxv[2], k4.z); mxv[3] = fmaxf(mxv[3], k4.w);
    }
    __shared__ float smn[4][kInkHeadDim], smx[4][kInkHeadDim], red[3][4];
    for (int c = 0; c < 4; ++c) { smn[warp][lane * 4 + c] = mn[c]; smx[warp][lane * 4 + c] = mxv[c]; }
    __syncthreads();
    const float kmin = fminf(fminf(smn[0][t], smn[1][t]), fminf(smn[2][t], smn[3][t]));
    const float kmax = fmaxf(fmaxf(smx[0][t], smx[1][t]), fmaxf(smx[2][t], smx[3][t]));
    const float scale = 0.08838834764831845f;
    for (int r = 0; r < 3; ++r) {
        const int h = 3 * kh + r;
        const float qd = q[(size_t)h * kInkHeadDim + t];
        float ub = fmaxf(qd * kmin, qd * kmax);
        ub = warp_sum(ub);
        if (lane == 0) red[r][warp] = ub;
    }
    __syncthreads();
    __shared__ float rt_m[3][4], ms_m[3][4];
    for (int r = 0; r < 3; ++r) {
        const int h = 3 * kh + r;
        float rm = -INFINITY, ms = 0.0f;
        for (int kp = p0 + t; kp < p1; kp += 128) {
            const int dist = min(max(qp - kp, 0), kInkRelExtent);
            rm = fmaxf(rm, rel[(size_t)h * (kInkRelExtent + 1) + dist]);
            ms += expf(scores[(size_t)h * L + kp] - mx[h]);
        }
        rm = warp_max(rm);
        ms = warp_sum(ms);
        if (lane == 0) { rt_m[r][warp] = rm; ms_m[r][warp] = ms; }
    }
    __syncthreads();
    if (t < 3) {
        const int h = 3 * kh + t;
        const float sub = red[t][0] + red[t][1] + red[t][2] + red[t][3];
        const float rm = fmaxf(fmaxf(rt_m[t][0], rt_m[t][1]), fmaxf(rt_m[t][2], rt_m[t][3]));
        const float msum = ms_m[t][0] + ms_m[t][1] + ms_m[t][2] + ms_m[t][3];
        bound[(size_t)h * nb + b] = sub * scale + rm;
        mass[(size_t)h * nb + b] = msum / z[h];
    }
}

// o[c][h][:] = sum over the 64-blocks marked in masks[c][h][:] of e^(s-max) v / (e^(sink-max) + sum e^(s-max))     grid (24, nc), block 128
__global__ void __launch_bounds__(128) k_probe_out(
    const float* __restrict__ scores, const float* __restrict__ mx, const float* __restrict__ sinks, InkKvVisitDesc kv, int sl, int L, int nb64,
    const uint8_t* __restrict__ masks, float* __restrict__ out) {
    const int h = blockIdx.x, c = blockIdx.y, kh = h / 3, warp = threadIdx.x >> 5, lane = threadIdx.x & 31;
    const uint8_t* mk = masks + ((size_t)c * kInkHeads + h) * nb64;
    float a0 = 0, a1 = 0, a2 = 0, a3 = 0, l = 0;
    const float m = mx[h];
    for (int b = 0; b < nb64; ++b) {
        if (!mk[b]) continue;
        const int p0 = b * kProbeBlk, p1 = min(L, p0 + kProbeBlk);
        for (int kp = p0 + warp; kp < p1; kp += 4) {
            float4 k4, v4;
            ink_kv_load4(kv, sl, kp, kh, lane, k4, v4);
            const float e = expf(scores[(size_t)h * L + kp] - m);
            l += e;
            a0 += e * v4.x; a1 += e * v4.y; a2 += e * v4.z; a3 += e * v4.w;
        }
    }
    __shared__ float sl_[4], sa[4][kInkHeadDim];
    if (lane == 0) sl_[warp] = l;
    sa[warp][lane * 4 + 0] = a0; sa[warp][lane * 4 + 1] = a1; sa[warp][lane * 4 + 2] = a2; sa[warp][lane * 4 + 3] = a3;
    __syncthreads();
    if (warp == 0) {
        const float den = sl_[0] + sl_[1] + sl_[2] + sl_[3] + expf(sinks[h] - m);
        for (int c4 = 0; c4 < 4; ++c4) {
            const int d = lane * 4 + c4;
            out[((size_t)c * kInkHeads + h) * kInkHeadDim + d] = (sa[0][d] + sa[1][d] + sa[2][d] + sa[3][d]) / den;
        }
    }
}

// fp32 copy of the cached K / V of one KV head sweep (dump only)     grid (8, ceil(L/4)), block 128
__global__ void __launch_bounds__(128) k_probe_dump_kv(InkKvVisitDesc kv, int sl, int L, float* __restrict__ kout, float* __restrict__ vout) {
    const int kh = blockIdx.x, lane = threadIdx.x & 31, warp = threadIdx.x >> 5;
    const int kp = blockIdx.y * 4 + warp;
    if (kp >= L) return;
    float4 k4, v4;
    ink_kv_load4(kv, sl, kp, kh, lane, k4, v4);
    *(float4*)(kout + ((size_t)kp * kInkKVHeads + kh) * kInkHeadDim + lane * 4) = k4;
    *(float4*)(vout + ((size_t)kp * kInkKVHeads + kh) * kInkHeadDim + lane * 4) = v4;
}

struct InkAttnProbe {
    bool on = false;
    std::string path, dump_dir;
    int every = 1, max_rec = 4096, min_ctx = 1024, dump_visit = 47, needle_lo = -1, needle_hi = -1;
    FILE* fp = nullptr;
    int n_rec = 0;
    long long calls[64] = {};
    bool dumped = false;
    bool warned_graph = false;
    // device scratch
    size_t cap_L = 0;
    float *d_scores = nullptr, *d_mx = nullptr, *d_z = nullptr, *d_mass64 = nullptr, *d_bound64 = nullptr, *d_mass128 = nullptr, *d_bound128 = nullptr;
    uint8_t* d_masks = nullptr;
    float* d_out = nullptr;

    static InkAttnProbe& get() {
        static InkAttnProbe p;
        static bool init = false;
        if (!init) {
            init = true;
            if (const char* e = std::getenv("STRATA_INK_ATTN_PROBE"); e && e[0]) {
                p.on = true;
                p.path = e;
                if (const char* v = std::getenv("STRATA_INK_ATTN_PROBE_EVERY")) p.every = std::max(1, std::atoi(v));
                if (const char* v = std::getenv("STRATA_INK_ATTN_PROBE_MAXREC")) p.max_rec = std::max(1, std::atoi(v));
                if (const char* v = std::getenv("STRATA_INK_ATTN_PROBE_MINCTX")) p.min_ctx = std::max(1, std::atoi(v));
                if (const char* v = std::getenv("STRATA_INK_ATTN_PROBE_NEEDLE")) std::sscanf(v, "%d:%d", &p.needle_lo, &p.needle_hi);
                if (const char* v = std::getenv("STRATA_INK_ATTN_PROBE_DUMP")) p.dump_dir = v;
                if (const char* v = std::getenv("STRATA_INK_ATTN_PROBE_DUMP_VISIT")) p.dump_visit = std::atoi(v);
                p.fp = std::fopen(p.path.c_str(), "a");
                if (!p.fp) { std::fprintf(stderr, "[attn-probe] cannot open %s: probe off\n", p.path.c_str()); p.on = false; }
                else std::fprintf(stderr, "[attn-probe] ON -> %s (every %d, max %d records, min ctx %d, needle %d:%d)\n", p.path.c_str(), p.every, p.max_rec, p.min_ctx, p.needle_lo, p.needle_hi);
            }
        }
        return p;
    }

    void ensure(int L) {
        if ((size_t)L <= cap_L) return;
        free_all();
        cap_L = (size_t)L + 4096;
        const size_t nb64 = (cap_L + 63) / 64, nb128 = (cap_L + 127) / 128;
        INK_CK(cudaMalloc(&d_scores, (size_t)kInkHeads * cap_L * 4));
        INK_CK(cudaMalloc(&d_mx, kInkHeads * 4));
        INK_CK(cudaMalloc(&d_z, kInkHeads * 4));
        INK_CK(cudaMalloc(&d_mass64, kInkHeads * nb64 * 4));
        INK_CK(cudaMalloc(&d_bound64, kInkHeads * nb64 * 4));
        INK_CK(cudaMalloc(&d_mass128, kInkHeads * nb128 * 4));
        INK_CK(cudaMalloc(&d_bound128, kInkHeads * nb128 * 4));
        INK_CK(cudaMalloc(&d_masks, 32 * kInkHeads * nb64));
        INK_CK(cudaMalloc(&d_out, (size_t)32 * kInkHeads * kInkHeadDim * 4));
    }
    void free_all() {
        for (void* p : {(void*)d_scores, (void*)d_mx, (void*)d_z, (void*)d_mass64, (void*)d_bound64, (void*)d_mass128, (void*)d_bound128, (void*)d_masks, (void*)d_out})
            if (p) cudaFree(p);
        d_scores = d_mx = d_z = d_mass64 = d_bound64 = d_mass128 = d_bound128 = d_out = nullptr;
        d_masks = nullptr;
        cap_L = 0;
    }

    static void dump_file(const std::string& dir, const char* name, const void* data, size_t bytes) {
        const std::string fn = dir + "/" + name;
        if (FILE* f = std::fopen(fn.c_str(), "wb")) { std::fwrite(data, 1, bytes, f); std::fclose(f); }
    }

    // One global-layer decode step.  d_q [24,128] (after clip/tau), d_rel [24,1025], d_sinks [24], d_gate [24,128], d_attn_out [24,128] (the engine's own result).
    void run(cudaStream_t st, int visit, const InkKvVisitDesc& kv, int slot, int qp, const float* d_q, const float* d_rel, const float* d_sinks,
             const float* d_gate, const float* d_attn_out) {
        if (!on || n_rec >= max_rec || visit < 0 || visit >= 64) return;
        cudaStreamCaptureStatus cs = cudaStreamCaptureStatusNone;
        if (cudaStreamIsCapturing(st, &cs) == cudaSuccess && cs != cudaStreamCaptureStatusNone) {
            if (!warned_graph) { std::fprintf(stderr, "[attn-probe] stream is being captured into a CUDA graph: probe skipped (use STRATA_INK_NO_GRAPH=1)\n"); warned_graph = true; }
            return;
        }
        const long long step = calls[visit]++;
        if (step % every != 0) return;
        const int L = std::min(qp + 1, kv.cap_pos);
        if (L < min_ctx) return;
        ensure(L);
        const int nb64 = (L + 63) / 64, nb128 = (L + 127) / 128;
        k_probe_scores<<<dim3(kInkKVHeads, (L + 63) / 64), 128, 0, st>>>(d_q, d_rel, kv, slot, qp, L, d_scores);
        k_probe_stats<<<kInkHeads, 256, 0, st>>>(d_scores, d_sinks, L, d_mx, d_z);
        k_probe_blocks<<<dim3(kInkKVHeads, nb64), 128, 0, st>>>(d_q, d_rel, d_scores, d_mx, d_z, kv, slot, qp, L, 64, nb64, d_mass64, d_bound64);
        k_probe_blocks<<<dim3(kInkKVHeads, nb128), 128, 0, st>>>(d_q, d_rel, d_scores, d_mx, d_z, kv, slot, qp, L, 128, nb128, d_mass128, d_bound128);
        INK_CK(cudaStreamSynchronize(st));
        INK_CK(cudaGetLastError());

        std::vector<float> mass64((size_t)kInkHeads * nb64), bound64((size_t)kInkHeads * nb64), bound128((size_t)kInkHeads * nb128), mass128((size_t)kInkHeads * nb128);
        std::vector<float> hmx(kInkHeads), hz(kInkHeads), hsink(kInkHeads), hgate((size_t)kInkHeads * kInkHeadDim), hattn((size_t)kInkHeads * kInkHeadDim);
        INK_CK(cudaMemcpy(mass64.data(), d_mass64, mass64.size() * 4, cudaMemcpyDeviceToHost));
        INK_CK(cudaMemcpy(bound64.data(), d_bound64, bound64.size() * 4, cudaMemcpyDeviceToHost));
        INK_CK(cudaMemcpy(mass128.data(), d_mass128, mass128.size() * 4, cudaMemcpyDeviceToHost));
        INK_CK(cudaMemcpy(bound128.data(), d_bound128, bound128.size() * 4, cudaMemcpyDeviceToHost));
        INK_CK(cudaMemcpy(hmx.data(), d_mx, kInkHeads * 4, cudaMemcpyDeviceToHost));
        INK_CK(cudaMemcpy(hz.data(), d_z, kInkHeads * 4, cudaMemcpyDeviceToHost));
        INK_CK(cudaMemcpy(hsink.data(), d_sinks, kInkHeads * 4, cudaMemcpyDeviceToHost));
        INK_CK(cudaMemcpy(hgate.data(), d_gate, hgate.size() * 4, cudaMemcpyDeviceToHost));
        INK_CK(cudaMemcpy(hattn.data(), d_attn_out, hattn.size() * 4, cudaMemcpyDeviceToHost));

        // needle mass per head (dense), from the scores of the needle's positions
        std::vector<double> needle_mass(kInkHeads, 0.0);
        std::vector<char> needle_blk64(nb64, 0);
        const bool has_needle = needle_lo >= 0 && needle_hi >= needle_lo && needle_hi < L;
        std::vector<float> nsc;
        if (has_needle) {
            const int nl = needle_hi - needle_lo + 1;
            nsc.resize((size_t)kInkHeads * nl);
            INK_CK(cudaMemcpy2D(nsc.data(), (size_t)nl * 4, d_scores + needle_lo, (size_t)L * 4, (size_t)nl * 4, kInkHeads, cudaMemcpyDeviceToHost));
            for (int h = 0; h < kInkHeads; ++h)
                for (int j = 0; j < nl; ++j) needle_mass[h] += std::exp((double)nsc[(size_t)h * nl + j] - hmx[h]) / hz[h];
            for (int b = needle_lo / 64; b <= needle_hi / 64; ++b) needle_blk64[b] = 1;
        }

        // ---- selections ----
        struct Cfg { int B, budget, sel; };   // sel 0 oracle, 1 quest, 2 questf
        std::vector<Cfg> cfgs;
        for (int sel = 0; sel < 3; ++sel)
            for (int B : {64, 128})
                for (int budget : {1024, 2048, 4096, 8192}) cfgs.push_back({B, budget, sel});
        const int nc = 1 + (int)cfgs.size();
        std::vector<uint8_t> masks((size_t)nc * kInkHeads * nb64, 0);
        std::fill(masks.begin(), masks.begin() + (size_t)kInkHeads * nb64, (uint8_t)1);   // c = 0: dense
        for (int ci = 0; ci < (int)cfgs.size(); ++ci) {
            const Cfg& cf = cfgs[ci];
            const int nbB = (cf.B == 64) ? nb64 : nb128;
            const int k = std::max(1, cf.budget / cf.B);
            for (int h = 0; h < kInkHeads; ++h) {
                uint8_t* mk = masks.data() + ((size_t)(ci + 1) * kInkHeads + h) * nb64;
                std::vector<char> pick(nbB, 0);
                if (k >= nbB) {
                    std::fill(pick.begin(), pick.end(), 1);
                } else {
                    const float* key = (cf.sel == 0) ? ((cf.B == 64) ? &mass64[(size_t)h * nb64] : &mass128[(size_t)h * nb128])
                                                     : ((cf.B == 64) ? &bound64[(size_t)h * nb64] : &bound128[(size_t)h * nb128]);
                    int left = k;
                    if (cf.sel == 2) {
                        for (int b : {0, nbB - 1, nbB - 2}) if (b >= 0 && !pick[b] && left > 0) { pick[b] = 1; --left; }
                    }
                    std::vector<int> idx;
                    idx.reserve(nbB);
                    for (int b = 0; b < nbB; ++b) if (!pick[b]) idx.push_back(b);
                    if (left > 0 && left < (int)idx.size()) {
                        std::nth_element(idx.begin(), idx.begin() + left, idx.end(), [&](int a, int b2) { return key[a] > key[b2]; });
                        idx.resize(left);
                    }
                    if (left > 0) for (int b : idx) pick[b] = 1;
                }
                const int ratio = cf.B / 64;
                for (int b = 0; b < nbB; ++b) if (pick[b]) for (int r = 0; r < ratio; ++r) if (b * ratio + r < nb64) mk[b * ratio + r] = 1;
            }
        }
        INK_CK(cudaMemcpy(d_masks, masks.data(), masks.size(), cudaMemcpyHostToDevice));
        k_probe_out<<<dim3(kInkHeads, nc), 128, 0, st>>>(d_scores, d_mx, d_sinks, kv, slot, L, nb64, d_masks, d_out);
        INK_CK(cudaStreamSynchronize(st));
        INK_CK(cudaGetLastError());
        std::vector<float> out((size_t)nc * kInkHeads * kInkHeadDim);
        INK_CK(cudaMemcpy(out.data(), d_out, out.size() * 4, cudaMemcpyDeviceToHost));

        // self check: the dense recompute times sigmoid(gate) against the engine's own attention output (relative L2 over all 24 heads)
        double num = 0, den = 0;
        for (int h = 0; h < kInkHeads; ++h)
            for (int d = 0; d < kInkHeadDim; ++d) {
                const size_t i = (size_t)h * kInkHeadDim + d;
                const double mine = (double)out[i] * (1.0 / (1.0 + std::exp(-(double)hgate[i])));
                num += (mine - hattn[i]) * (mine - hattn[i]);
                den += (double)hattn[i] * hattn[i];
            }
        const double selfcheck = std::sqrt(num / std::max(den, 1e-30));

        if (fp) {
            std::fprintf(fp, "{\"step\":%lld,\"visit\":%d,\"qp\":%d,\"L\":%d,\"selfcheck\":%.3e,", step, visit, qp, L, selfcheck);
            std::fprintf(fp, "\"sink\":[");
            for (int h = 0; h < kInkHeads; ++h) std::fprintf(fp, "%s%.5g", h ? "," : "", std::exp((double)hsink[h] - hmx[h]) / hz[h]);
            std::fprintf(fp, "],\"needle_mass\":[");
            for (int h = 0; h < kInkHeads; ++h) std::fprintf(fp, "%s%.5g", h ? "," : "", needle_mass[h]);
            std::fprintf(fp, "],\"cfg\":[");
            for (int ci = 0; ci < (int)cfgs.size(); ++ci) {
                const Cfg& cf = cfgs[ci];
                std::fprintf(fp, "%s{\"sel\":\"%s\",\"B\":%d,\"budget\":%d,\"cap\":[", ci ? "," : "", cf.sel == 0 ? "oracle" : (cf.sel == 1 ? "quest" : "questf"), cf.B, cf.budget);
                std::vector<double> err(kInkHeads), ncap(kInkHeads, 0.0);
                std::vector<int> nhit(kInkHeads, 0);
                for (int h = 0; h < kInkHeads; ++h) {
                    const uint8_t* mk = masks.data() + ((size_t)(ci + 1) * kInkHeads + h) * nb64;
                    double cap = 0, tot = 0;
                    for (int b = 0; b < nb64; ++b) { tot += mass64[(size_t)h * nb64 + b]; if (mk[b]) cap += mass64[(size_t)h * nb64 + b]; }
                    std::fprintf(fp, "%s%.5g", h ? "," : "", tot > 0 ? cap / tot : 1.0);
                    const float* od = out.data() + (size_t)h * kInkHeadDim;
                    const float* os = out.data() + ((size_t)(ci + 1) * kInkHeads + h) * kInkHeadDim;
                    double e2 = 0, n2 = 0;
                    for (int d = 0; d < kInkHeadDim; ++d) { e2 += (double)(os[d] - od[d]) * (os[d] - od[d]); n2 += (double)od[d] * od[d]; }
                    err[h] = std::sqrt(e2 / std::max(n2, 1e-30));
                    if (has_needle) {
                        bool all = true;
                        for (int b = needle_lo / 64; b <= needle_hi / 64; ++b) all = all && mk[b];
                        nhit[h] = all ? 1 : 0;
                        const int nl = needle_hi - needle_lo + 1;
                        for (int j = 0; j < nl; ++j) if (mk[(needle_lo + j) / 64]) ncap[h] += std::exp((double)nsc[(size_t)h * nl + j] - hmx[h]) / hz[h];
                    }
                }
                std::fprintf(fp, "],\"err\":[");
                for (int h = 0; h < kInkHeads; ++h) std::fprintf(fp, "%s%.5g", h ? "," : "", err[h]);
                std::fprintf(fp, "]");
                if (has_needle) {
                    std::fprintf(fp, ",\"nhit\":[");
                    for (int h = 0; h < kInkHeads; ++h) std::fprintf(fp, "%s%d", h ? "," : "", nhit[h]);
                    std::fprintf(fp, "],\"ncap\":[");
                    for (int h = 0; h < kInkHeads; ++h) std::fprintf(fp, "%s%.5g", h ? "," : "", ncap[h]);
                    std::fprintf(fp, "]");
                }
                std::fprintf(fp, "}");
            }
            std::fprintf(fp, "]}\n");
            std::fflush(fp);
        }
        ++n_rec;

        if (!dump_dir.empty() && !dumped && visit == dump_visit) {
            dumped = true;
            std::vector<float> hq((size_t)kInkHeads * kInkHeadDim), hrel((size_t)kInkHeads * (kInkRelExtent + 1));
            INK_CK(cudaMemcpy(hq.data(), d_q, hq.size() * 4, cudaMemcpyDeviceToHost));
            INK_CK(cudaMemcpy(hrel.data(), d_rel, hrel.size() * 4, cudaMemcpyDeviceToHost));
            float *dk = nullptr, *dv = nullptr;
            const size_t kvn = (size_t)L * kInkKVHeads * kInkHeadDim;
            INK_CK(cudaMalloc(&dk, kvn * 4));
            INK_CK(cudaMalloc(&dv, kvn * 4));
            k_probe_dump_kv<<<dim3(kInkKVHeads, (L + 3) / 4), 128, 0, st>>>(kv, slot, L, dk, dv);
            INK_CK(cudaStreamSynchronize(st));
            std::vector<float> hk(kvn), hv(kvn);
            INK_CK(cudaMemcpy(hk.data(), dk, kvn * 4, cudaMemcpyDeviceToHost));
            INK_CK(cudaMemcpy(hv.data(), dv, kvn * 4, cudaMemcpyDeviceToHost));
            cudaFree(dk); cudaFree(dv);
            dump_file(dump_dir, "q.f32", hq.data(), hq.size() * 4);
            dump_file(dump_dir, "rel.f32", hrel.data(), hrel.size() * 4);
            dump_file(dump_dir, "sink.f32", hsink.data(), hsink.size() * 4);
            dump_file(dump_dir, "gate.f32", hgate.data(), hgate.size() * 4);
            dump_file(dump_dir, "attn_out.f32", hattn.data(), hattn.size() * 4);
            dump_file(dump_dir, "K.f32", hk.data(), hk.size() * 4);
            dump_file(dump_dir, "V.f32", hv.data(), hv.size() * 4);
            dump_file(dump_dir, "masks.u8", masks.data(), masks.size());
            dump_file(dump_dir, "out.f32", out.data(), out.size() * 4);
            dump_file(dump_dir, "mass64.f32", mass64.data(), mass64.size() * 4);
            dump_file(dump_dir, "bound64.f32", bound64.data(), bound64.size() * 4);
            char meta[256];
            std::snprintf(meta, sizeof(meta), "{\"visit\":%d,\"qp\":%d,\"L\":%d,\"nb64\":%d,\"nc\":%d}\n", visit, qp, L, nb64, nc);
            dump_file(dump_dir, "meta.json", meta, std::strlen(meta));
            std::fprintf(stderr, "[attn-probe] raw dump of visit %d (L=%d) in %s\n", visit, L, dump_dir.c_str());
        }
    }
};

}  // namespace strata::ink
