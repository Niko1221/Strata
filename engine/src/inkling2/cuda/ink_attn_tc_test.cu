// engine/src/inkling2/cuda/ink_attn_tc_test.cu - accuracy and kernel-level timing of the tensor-core split-K decode attention (ink_attn_tc.cuh)
// against the old scalar kernel (k_ink_flash_gqa_splitk + merge), on synthetic KV (FP16 and FP8), plain rows and verify trees.
//
// Build: nvcc -O3 -std=c++20 --expt-relaxed-constexpr -gencode arch=compute_120,code=sm_120 -I<repo>/engine/src/inkling2/cuda -I<repo>/engine/include ink_attn_tc_test.cu -o ink_attn_tc_test
// Run:   ink_attn_tc_test [--ctx N] [--reps R] [--nbuf 1|2] [--depth 1|2|3] [--quick]
// Output: one line per case: relative L2 error, max abs error, time of old and new (ms, kernel + merge), GB/s of the new one (one pass over the KV).
// Exit code 0 when every case is below the error bound (rel L2 < 2e-3).
#include "ink_attn.cuh"
#include "ink_attn_tc.cuh"
#include "ink_cuda.cuh"
#include "ink_rtx5090_bspec_kernels.cuh"

#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <random>
#include <string>
#include <vector>

using namespace strata::ink;

static __device__ __forceinline__ uint32_t mix32(uint64_t x) {
    x += 0x9E3779B97F4A7C15ull;
    x = (x ^ (x >> 30)) * 0xBF58476D1CE4E5B9ull;
    x = (x ^ (x >> 27)) * 0x94D049BB133111EBull;
    return (uint32_t)((x ^ (x >> 31)) >> 16);
}
static __device__ __forceinline__ float rnd(uint64_t i, uint64_t seed) { return ((mix32(i * 2654435761ull + seed) & 0xFFFFFF) * (1.0f / 16777216.0f)) - 0.5f; }

__global__ void k_fill_f16(__half* p, size_t n, uint64_t seed, float amp) {
    for (size_t i = blockIdx.x * (size_t)blockDim.x + threadIdx.x; i < n; i += (size_t)gridDim.x * blockDim.x) p[i] = __float2half_rn(rnd(i, seed) * amp);
}
__global__ void k_fill_f8(uint8_t* p, size_t n, uint64_t seed, float amp) {
    for (size_t i = blockIdx.x * (size_t)blockDim.x + threadIdx.x; i < n; i += (size_t)gridDim.x * blockDim.x) {
        const __nv_fp8_e4m3 v(rnd(i, seed) * amp);
        p[i] = v.__x;
    }
}
__global__ void k_fill_f32(float* p, size_t n, uint64_t seed, float amp, float off) {
    for (size_t i = blockIdx.x * (size_t)blockDim.x + threadIdx.x; i < n; i += (size_t)gridDim.x * blockDim.x) p[i] = rnd(i, seed) * amp + off;
}

struct Tree { int M; int tp0; uint8_t anc[32 * 8]; uint8_t dep[32]; int32_t pos[32]; };

static Tree make_tree(int M, int tp0, uint32_t seed) {
    std::mt19937 rng(seed);
    Tree t{};
    t.M = M; t.tp0 = tp0;
    int parent[32] = {};
    t.dep[0] = 0;
    for (int i = 1; i < M; ++i) {
        int p;
        do { p = (int)(rng() % i); } while (t.dep[p] >= 7);
        parent[i] = p;
        t.dep[i] = t.dep[p] + 1;
    }
    for (int i = 0; i < M; ++i) {
        int chain[8], n = 0;
        for (int c = i;; c = parent[c]) { chain[n++] = c; if (c == 0) break; }
        for (int d = 0; d < n; ++d) t.anc[i * 8 + d] = (uint8_t)chain[n - 1 - d];
        t.pos[i] = tp0 + t.dep[i];
    }
    return t;
}

struct Bufs {
    InkKvVisitDesc kv{};
    float *q = nullptr, *rel = nullptr, *sinks = nullptr, *tk = nullptr, *tv = nullptr;
    int32_t* pos = nullptr;
    uint8_t* anc = nullptr; uint8_t* dep = nullptr;
    float *pm = nullptr, *pl = nullptr, *po = nullptr, *out[2] = {};
    int cap = 0;
};

static int g_nbuf = 2, g_depth = 2, g_reps = 20;

static double now_ms_between(cudaEvent_t a, cudaEvent_t b) { float ms = 0; cudaEventElapsedTime(&ms, a, b); return ms; }


// Bandwidth probe: the same access pattern as the TC kernel (a block per (KV head, slice), 32-key tiles, 16-byte pieces, depth-D register prefetch), no arithmetic but an XOR.
template <int FMT, int D>
__global__ void __launch_bounds__(128, 2) k_bw_probe(InkKvVisitDesc kv, int nc, int nsp, uint32_t* sink) {
    using namespace strata::ink::tcattn;
    const int kh = blockIdx.x, sp = blockIdx.y, tid = threadIdx.x;
    const int per = slice_len(nc, nsp);
    const int lo = sp * per, hi = min(nc, lo + per);
    if (lo >= hi) return;
    RawTile<FMT> r0, r1, r2;
    load_tile<FMT>(r0, kv, 0, kh, lo, hi, tid);
    if (D >= 2) load_tile<FMT>(r1, kv, 0, kh, lo + kTile, hi, tid);
    if (D >= 3) load_tile<FMT>(r2, kv, 0, kh, lo + 2 * kTile, hi, tid);
    uint32_t x = 0;
    auto fold = [&](RawTile<FMT>& r, int tile) {
        for (int i = 0; i < Fmt<FMT>::kPasses; ++i) x ^= r.k[i].x ^ r.k[i].y ^ r.k[i].z ^ r.k[i].w ^ r.v[i].x ^ r.v[i].y ^ r.v[i].z ^ r.v[i].w ^ __float_as_uint(r.ks[i] + r.vs[i]);
        load_tile<FMT>(r, kv, 0, kh, tile + D * kTile, hi, tid);
    };
    for (int tile = lo; tile < hi; tile += D * kTile) {
        fold(r0, tile);
        if (D >= 2 && tile + kTile < hi) fold(r1, tile + kTile);
        if (D >= 3 && tile + 2 * kTile < hi) fold(r2, tile + 2 * kTile);
    }
    if (x == 0x12345678u) sink[0] = x;
}

static void run_probe(Bufs& b, int fmt, int nc) {
    uint32_t* sink = nullptr;
    cudaMalloc(&sink, 4);
    b.kv.kv_fmt = (uint8_t)fmt;
    cudaEvent_t e0, e1;
    cudaEventCreate(&e0); cudaEventCreate(&e1);
    const double bytes = (double)nc * kInkKVHeads * ((fmt == 0 ? 256.0 : 128.0 + 4.0) * 2);
    for (int D = 1; D <= 3; ++D) {
        auto go = [&]() {
            const dim3 grid(kInkKVHeads, strata::ink::tcattn::splits_cur());
            if (fmt == 1) { if (D == 1) k_bw_probe<1, 1><<<grid, 128>>>(b.kv, nc, strata::ink::tcattn::splits_cur(), sink); else if (D == 2) k_bw_probe<1, 2><<<grid, 128>>>(b.kv, nc, strata::ink::tcattn::splits_cur(), sink); else k_bw_probe<1, 3><<<grid, 128>>>(b.kv, nc, strata::ink::tcattn::splits_cur(), sink); }
            else { if (D == 1) k_bw_probe<0, 1><<<grid, 128>>>(b.kv, nc, strata::ink::tcattn::splits_cur(), sink); else if (D == 2) k_bw_probe<0, 2><<<grid, 128>>>(b.kv, nc, strata::ink::tcattn::splits_cur(), sink); else k_bw_probe<0, 3><<<grid, 128>>>(b.kv, nc, strata::ink::tcattn::splits_cur(), sink); }
        };
        for (int i = 0; i < 3; ++i) go();
        cudaDeviceSynchronize();
        cudaEventRecord(e0);
        for (int i = 0; i < g_reps; ++i) go();
        cudaEventRecord(e1); cudaEventSynchronize(e1);
        const double ms = now_ms_between(e0, e1) / g_reps;
        std::printf("probe %s nc=%d depth %d: %.3f ms  %.0f GB/s\n", fmt ? "fp8 " : "fp16", nc, D, ms, bytes / (ms * 1e-3) / 1e9);
    }
    cudaEventDestroy(e0); cudaEventDestroy(e1);
    cudaFree(sink);
}


// batched drafting: B slots x K chain rows, against k_ink_flash_gqa_bspec (the K/V of the chain rows is the tail)
static int run_chain(Bufs& b, int fmt, int B, int K, int nc, const char* tag) {
    const int M = B * K;
    std::vector<int32_t> hp(M), hs(M, 0);
    for (int s = 0; s < B; ++s) for (int k = 0; k < K; ++k) hp[s * K + k] = nc - 37 * s + k;
    int32_t* dslots = nullptr;
    cudaMalloc(&dslots, 64 * 4);
    cudaMemcpy(b.pos, hp.data(), M * 4, cudaMemcpyHostToDevice);
    cudaMemcpy(dslots, hs.data(), M * 4, cudaMemcpyHostToDevice);
    k_fill_f32<<<256, 256>>>(b.q, (size_t)M * kInkHeads * kInkHeadDim, 901, 3.0f, 0.0f);
    k_fill_f32<<<256, 256>>>(b.rel, (size_t)M * kInkHeads * (kInkRelExtent + 1), 902, 2.0f, 0.0f);
    k_fill_f32<<<4, 256>>>(b.sinks, (size_t)M * kInkHeads, 903, 4.0f, 0.0f);
    k_fill_f32<<<64, 256>>>(b.tk, (size_t)M * kInkKVHeads * kInkHeadDim, 904, 4.0f, 0.0f);
    k_fill_f32<<<64, 256>>>(b.tv, (size_t)M * kInkKVHeads * kInkHeadDim, 905, 4.0f, 0.0f);
    cudaDeviceSynchronize();
    b.kv.kv_fmt = (uint8_t)fmt;
    auto old_path = [&](cudaStream_t st) {
        rtx5090::k_ink_flash_gqa_bspec<<<dim3(kInkKVHeads, M), 128, 0, st>>>(b.q, b.rel, b.sinks, nullptr, b.kv, dslots, b.pos, b.tk, b.tv, b.out[0], K, M);
    };
    auto new_path = [&](cudaStream_t st) -> bool {
        const bool ok = strata::ink::tcattn::launch(st, b.q, b.rel, b.sinks, b.kv, dslots, 0, b.pos, b.tk, b.tv, nullptr, nullptr, 0, b.pm, b.pl, b.po, M, K, 0, g_nbuf, g_depth);
        if (ok) strata::ink::tcattn::merge(st, b.pm, b.pl, b.po, nullptr, b.out[1], nullptr, b.pos, true, b.kv.cap_pos, M, K);
        return ok;
    };
    old_path(0);
    if (!new_path(0)) { std::printf("%-34s  NOT APPLICABLE\n", tag); cudaFree(dslots); return 1; }
    cudaError_t e = cudaDeviceSynchronize();
    if (e != cudaSuccess) { std::printf("%-34s  CUDA ERROR %s\n", tag, cudaGetErrorString(e)); return 1; }
    const size_t n = (size_t)M * kInkHeads * kInkHeadDim;
    std::vector<float> r(n), c(n);
    cudaMemcpy(r.data(), b.out[0], n * 4, cudaMemcpyDeviceToHost);
    cudaMemcpy(c.data(), b.out[1], n * 4, cudaMemcpyDeviceToHost);
    double d2 = 0, r2 = 0;
    bool finite = true;
    for (size_t i = 0; i < n; ++i) { if (!std::isfinite(c[i])) finite = false; const double d = (double)c[i] - r[i]; d2 += d * d; r2 += (double)r[i] * r[i]; }
    const double rel_l2 = std::sqrt(d2 / std::max(r2, 1e-30));
    cudaEvent_t e0, e1;
    cudaEventCreate(&e0); cudaEventCreate(&e1);
    for (int i = 0; i < 3; ++i) { old_path(0); new_path(0); }
    cudaDeviceSynchronize();
    cudaEventRecord(e0);
    for (int i = 0; i < g_reps; ++i) old_path(0);
    cudaEventRecord(e1); cudaEventSynchronize(e1);
    const double t_old = now_ms_between(e0, e1) / g_reps;
    cudaEventRecord(e0);
    for (int i = 0; i < g_reps; ++i) new_path(0);
    cudaEventRecord(e1); cudaEventSynchronize(e1);
    const double t_new = now_ms_between(e0, e1) / g_reps;
    cudaFree(dslots);
    const bool pass = finite && rel_l2 < 2e-3;
    std::printf("%-34s  relL2 %.2e  old %.3f ms  new %.3f ms  x%.2f  %s\n", tag, rel_l2, t_old, t_new, t_old / t_new, pass ? "ok" : "FAIL");
    return pass ? 0 : 1;
}

// returns 0 ok, 1 error above bound
static int run_case(Bufs& b, int fmt, int M, int nc, bool tree, uint32_t seed, const char* tag) {
    Tree t = make_tree(M, nc, seed);
    if (!tree) { for (int i = 0; i < M; ++i) t.pos[i] = nc - 1; }   // plain rows: M rows each its own slot-less decode at the same position (rows are independent)
    cudaMemcpy(b.pos, t.pos, M * 4, cudaMemcpyHostToDevice);
    cudaMemcpy(b.anc, t.anc, 32 * 8, cudaMemcpyHostToDevice);
    cudaMemcpy(b.dep, t.dep, 32, cudaMemcpyHostToDevice);
    const size_t nrel = (size_t)M * kInkHeads * (kInkRelExtent + 1);
    k_fill_f32<<<256, 256>>>(b.q, (size_t)M * kInkHeads * kInkHeadDim, seed * 3 + 1, 3.0f, 0.0f);
    k_fill_f32<<<256, 256>>>(b.rel, nrel, seed * 3 + 2, 2.0f, 0.0f);
    k_fill_f32<<<4, 256>>>(b.sinks, (size_t)M * kInkHeads, seed * 3 + 3, 4.0f, 0.0f);
    k_fill_f32<<<64, 256>>>(b.tk, (size_t)M * kInkKVHeads * kInkHeadDim, seed * 3 + 4, 4.0f, 0.0f);
    k_fill_f32<<<64, 256>>>(b.tv, (size_t)M * kInkKVHeads * kInkHeadDim, seed * 3 + 5, 4.0f, 0.0f);
    cudaDeviceSynchronize();
    b.kv.kv_fmt = (uint8_t)fmt;

    const int n_old = kInkAttnMaxSplits;
    const int n_new = strata::ink::tcattn::kSlots;   // partial slots per (row, head)
    auto old_path = [&](cudaStream_t st) {
        dim3 gsp(kInkKVHeads, n_old, M);
        k_ink_flash_gqa_splitk<<<gsp, 128, 0, st>>>(b.q, b.rel, b.sinks, b.kv, nullptr, 0, b.pos, tree ? b.tk : nullptr, tree ? b.tv : nullptr, tree ? b.anc : nullptr,
                                                      tree ? b.dep : nullptr, nc, b.pm, b.pl, b.po, M, n_old);
        k_ink_flash_gqa_merge<<<dim3(kInkHeads, M), 128, 0, st>>>(b.pm, b.pl, b.po, nullptr, b.out[0], M, n_old);
    };
    auto new_path = [&](cudaStream_t st) -> bool {
        const bool ok = strata::ink::tcattn::launch(st, b.q, b.rel, b.sinks, b.kv, nullptr, 0, b.pos, tree ? b.tk : nullptr, tree ? b.tv : nullptr, tree ? b.anc : nullptr,
                                                    tree ? b.dep : nullptr, kInkMaxTreeDepth, b.pm, b.pl, b.po, M, 0, 0, g_nbuf, g_depth);
        if (ok) strata::ink::tcattn::merge(st, b.pm, b.pl, b.po, nullptr, b.out[1], nullptr, b.pos, tree, b.kv.cap_pos, M);
        return ok;
    };
    old_path(0);
    cudaDeviceSynchronize();
    cudaMemset(b.pm, 0xFF, (size_t)M * kInkHeads * n_new * 4);   // garbage in the partials: inactive slots must be written by the kernel
    cudaMemset(b.pl, 0xFF, (size_t)M * kInkHeads * n_new * 4);
    if (!new_path(0)) { std::printf("%-34s  NOT APPLICABLE\n", tag); return 1; }
    cudaError_t e = cudaDeviceSynchronize();
    if (e != cudaSuccess) { std::printf("%-34s  CUDA ERROR %s\n", tag, cudaGetErrorString(e)); return 1; }
    const size_t n = (size_t)M * kInkHeads * kInkHeadDim;
    std::vector<float> r(n), c(n);
    cudaMemcpy(r.data(), b.out[0], n * 4, cudaMemcpyDeviceToHost);
    cudaMemcpy(c.data(), b.out[1], n * 4, cudaMemcpyDeviceToHost);
    double d2 = 0, r2 = 0, mx = 0;
    bool finite = true;
    for (size_t i = 0; i < n; ++i) {
        if (!std::isfinite(c[i])) finite = false;
        const double d = (double)c[i] - (double)r[i];
        d2 += d * d; r2 += (double)r[i] * r[i];
        mx = std::max(mx, std::fabs(d));
    }
    const double rel_l2 = std::sqrt(d2 / std::max(r2, 1e-30));
    // timing
    cudaEvent_t e0, e1;
    cudaEventCreate(&e0); cudaEventCreate(&e1);
    for (int i = 0; i < 3; ++i) { old_path(0); new_path(0); }
    cudaDeviceSynchronize();
    cudaEventRecord(e0);
    for (int i = 0; i < g_reps; ++i) old_path(0);
    cudaEventRecord(e1); cudaEventSynchronize(e1);
    const double t_old = now_ms_between(e0, e1) / g_reps;
    cudaEventRecord(e0);
    for (int i = 0; i < g_reps; ++i) new_path(0);
    cudaEventRecord(e1); cudaEventSynchronize(e1);
    const double t_new = now_ms_between(e0, e1) / g_reps;
    cudaEventDestroy(e0); cudaEventDestroy(e1);
    const double row_bytes = (fmt == 0 ? 256.0 : 128.0 + 4.0) * 2;   // K + V (+ scales) of one cell of one head
    const double gbs = (double)nc * kInkKVHeads * row_bytes / (t_new * 1e-3) / 1e9;
    const bool pass = finite && rel_l2 < 2e-3;
    std::printf("%-34s  relL2 %.2e  maxabs %.2e  old %.3f ms  new %.3f ms  x%.2f  %.0f GB/s  %s\n", tag, rel_l2, mx, t_old, t_new, t_old / t_new, gbs, pass ? "ok" : "FAIL");
    return pass ? 0 : 1;
}

int main(int argc, char** argv) {
    int ctx = 131072;
    bool quick = false;
    for (int i = 1; i < argc; ++i) {
        if (!std::strcmp(argv[i], "--ctx") && i + 1 < argc) ctx = std::atoi(argv[++i]);
        else if (!std::strcmp(argv[i], "--reps") && i + 1 < argc) g_reps = std::atoi(argv[++i]);
        else if (!std::strcmp(argv[i], "--nbuf") && i + 1 < argc) g_nbuf = std::atoi(argv[++i]);
        else if (!std::strcmp(argv[i], "--depth") && i + 1 < argc) g_depth = std::atoi(argv[++i]);
        else if (!std::strcmp(argv[i], "--quick")) quick = true;
    }
    cudaDeviceProp prop;
    cudaGetDeviceProperties(&prop, 0);
    std::printf("device %s sm_%d%d, %d SMs, ctx %d, nbuf %d, reps %d\n", prop.name, prop.major, prop.minor, prop.multiProcessorCount, ctx, g_nbuf, g_reps);
    Bufs b;
    b.cap = ctx + 64;
    const size_t cells = (size_t)b.cap * kInkKVHeads;
    const size_t n8 = cells * kInkHeadDim;
    cudaMalloc(&b.kv.k_f16, n8 * 2); cudaMalloc(&b.kv.v_f16, n8 * 2);
    cudaMalloc(&b.kv.k_f8, n8); cudaMalloc(&b.kv.v_f8, n8);
    cudaMalloc(&b.kv.k_sc, cells * 4); cudaMalloc(&b.kv.v_sc, cells * 4);
    k_fill_f16<<<1024, 256>>>(b.kv.k_f16, n8, 11, 4.0f);
    k_fill_f16<<<1024, 256>>>(b.kv.v_f16, n8, 12, 4.0f);
    k_fill_f8<<<1024, 256>>>(b.kv.k_f8, n8, 13, 400.0f);
    k_fill_f8<<<1024, 256>>>(b.kv.v_f8, n8, 14, 400.0f);
    k_fill_f32<<<256, 256>>>(b.kv.k_sc, cells, 15, 0.01f, 0.02f);   // 0.015 .. 0.025 (real K up to about +-5)
    k_fill_f32<<<256, 256>>>(b.kv.v_sc, cells, 16, 0.01f, 0.02f);
    b.kv.cap_pos = b.cap; b.kv.dev_cap = b.cap; b.kv.host_cap = 0; b.kv.is_swa = false;
    const int MMAX = 32;
    cudaMalloc(&b.q, (size_t)MMAX * kInkHeads * kInkHeadDim * 4);
    cudaMalloc(&b.rel, (size_t)MMAX * kInkHeads * (kInkRelExtent + 1) * 4);
    cudaMalloc(&b.sinks, (size_t)MMAX * kInkHeads * 4);
    cudaMalloc(&b.tk, (size_t)MMAX * kInkKVHeads * kInkHeadDim * 4);
    cudaMalloc(&b.tv, (size_t)MMAX * kInkKVHeads * kInkHeadDim * 4);
    cudaMalloc(&b.pos, MMAX * 4); cudaMalloc(&b.anc, 32 * 8); cudaMalloc(&b.dep, 32);
    const size_t np = (size_t)MMAX * kInkHeads * strata::ink::tcattn::kSlots;
    cudaMalloc(&b.pm, np * 4); cudaMalloc(&b.pl, np * 4); cudaMalloc(&b.po, np * kInkHeadDim * 4);
    cudaMalloc(&b.out[0], (size_t)MMAX * kInkHeads * kInkHeadDim * 4); cudaMalloc(&b.out[1], (size_t)MMAX * kInkHeads * kInkHeadDim * 4);
    cudaDeviceSynchronize();
    if (cudaGetLastError() != cudaSuccess) { std::printf("setup failed\n"); return 2; }

    for (int fmt = 1; fmt >= 0; --fmt) run_probe(b, fmt, ctx);
    int fails = 0;
    std::vector<int> ctxs;
    if (quick) ctxs = {ctx};
    else ctxs = {300, 4099, 32768, ctx};
    const int Ms[] = {1, 3, 5, 6, 10, 11, 16, 24};
    for (int fmt = 1; fmt >= 0; --fmt) {
        for (int c : ctxs) {
            if (c > ctx) continue;
            for (int M : Ms) {
                for (int pass = 0; pass < 2; ++pass) {   // pass 0: verify tree, pass 1: plain rows (mode 0 with M <= 16, mode 1)
                    const bool tree = (pass == 0) && M > 1;
                    if (pass == 1 && (M == 1 || M > 16 || (quick && M != 5))) continue;
                    char tag[96];
                    std::snprintf(tag, sizeof tag, "%s %s nc=%d M=%d", fmt ? "fp8 " : "fp16", tree ? "tree" : "plain", c, M);
                    if (quick && pass == 0 && !(M == 1 || M == 5 || M == 10 || M == 16)) continue;
                    fails += run_case(b, fmt, M, c, tree, 100 + M, tag);
                }
            }
        }
    }
    for (int fmt = 1; fmt >= 0; --fmt) {
        for (int c : {4099, 32768, 131072}) {
            if (c > ctx) continue;
            const int BK[][2] = {{2, 3}, {4, 3}, {4, 4}, {8, 3}, {3, 5}};
            for (auto& bk : BK) {
                char tag[96];
                std::snprintf(tag, sizeof tag, "chain %s nc=%d B=%d K=%d", fmt ? "fp8 " : "fp16", c, bk[0], bk[1]);
                fails += run_chain(b, fmt, bk[0], bk[1], c, tag);
            }
        }
    }
    std::printf("TC-TEST-RC=%d\n", fails ? 1 : 0);
    return fails ? 1 : 0;
}
