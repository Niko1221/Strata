// The APR trellis routed experts (TQ2_T, TQK6, TQK7: Gyro-M, Gyro-S) and their Hadamard input rotation in the split engine, against the
// main engine's CUDA kernels (strata::kernels: hadamard_rows, quantize_q8_1_rows, iq_mmvq, iq_dequant_f16) on real tensors of the file.
//   split_trellis_test MODEL.gguf [MODEL.gguf ...]      exit 0: every check within its bound; 1: a check failed; 2: usage / file error
// Per file: layers 0, 23 and 47, experts 0, 255 and 511, T = 5 random rows (seeded), each card of the two-card split (card_split.hpp):
//   rotate f16   had_rows_f16(x)           == f16 of hadamard_rows(x)                      mismatches: 0
//   rotate q8    had_quantize_q8_1(x)      == quantize_q8_1_rows(hadamard_rows(x)) (qs, d)  mismatches: 0
//   tile         the prompt tiles' f16 weights (MmTile, moe_mma_iq.cuh) == iq_dequant_f16    mismatches: 0
//   gu decode    moe_gate_up (q8_1 x int8 codebook) vs iq_mmvq on the same q8_1 rows          max|d| / max|ref| < 1e-4
//   gu prompt    moe_gate_up_mma (f16 tiles) vs a double dot of iq_dequant_f16 weights and f16(hadamard_rows(x))   < 1e-3
//   down decode  moe_down on both cards (each its cover), summed, vs a double dot of the f16 weights and hadamard_rows(h):
//                relative L2 < 3e-2 (the q8_1 / int8-codebook bound native_expert_parity uses); the main engine's iq_mmvq on the
//                whole h is printed beside it
//   down prompt  swiglu_h16_had + moe_down_mma on both cards, summed, vs the same double reference: relative L2 < 5e-3
// Once per file, swiglu_h16_had over 7000 rows x TOPK = 70000 pairs (more than a grid's y extent), each card: sampled pairs,
// some past 65535, vs hadamard_rows of the same h made on the host: max|d| / max|ref| < 1e-3 (f16 rounding).
// A non-finite difference fails every bound.
// h is silu(gate) * up of the main engine's own gate|up rows, so the down inputs have the model's real spread.
#include "kernels.hpp"

#include <cuda_fp16.h>
#include <cuda_runtime.h>
#include <mma.h>

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <random>
#include <stdexcept>
#include <string>
#include <vector>

#include "gguf.hpp"
#include "hadamard_meta.hpp"
#include "qdot.cuh"
#include "strata/kernels/iq_kernels.hpp"

namespace fnx {
namespace {

__device__ __forceinline__ float silu_(float x) { return x / (1.0f + __expf(-x)); }

#include "moe_w.cuh"
#include "moe_mma.cuh"
#include "moe_mma_iq.cuh"

template <int TYPE>
__global__ void __launch_bounds__(MM_THREADS) k_tiles(const char* __restrict__ we, size_t row_a, int nrows, int ncols, __half* __restrict__ out) {
    extern __shared__ __align__(32) __half smem[];
    constexpr int NP = MM_BM / MM_PASS;
    const int row0 = blockIdx.x * MM_BM;
    for (int k0 = 0; k0 < ncols; k0 += MM_KC) {
        const int kc = min(MM_KC, ncols - k0);
        MmTile<TYPE> wt[NP];
#pragma unroll
        for (int p = 0; p < NP; ++p) wt[p].fetch(we, row_a, nrows, row0 + p * MM_PASS, k0, kc);
#pragma unroll
        for (int p = 0; p < NP; ++p) wt[p].put(k0, kc, smem + p * MM_PASS * MM_LD);
        __syncthreads();
        for (int e = threadIdx.x; e < MM_BM * kc; e += MM_THREADS) {
            const int r = e / kc, c = e % kc, row = row0 + r;
            if (row < nrows) out[(size_t) row * ncols + k0 + c] = smem[r * MM_LD + c];
        }
        __syncthreads();
    }
}
void tiles(uint32_t type, const char* d_we, size_t row_a, int nrows, int ncols, __half* d_out) {
    const dim3 g((nrows + MM_BM - 1) / MM_BM);
    const size_t sm = MM_BM * MM_LD * sizeof(__half);
    switch (type) {
        case T_TQ2_T: k_tiles<T_TQ2_T><<<g, MM_THREADS, sm>>>(d_we, row_a, nrows, ncols, d_out); break;
        case T_TQK6: k_tiles<T_TQK6><<<g, MM_THREADS, sm>>>(d_we, row_a, nrows, ncols, d_out); break;
        case T_TQK7: k_tiles<T_TQK7><<<g, MM_THREADS, sm>>>(d_we, row_a, nrows, ncols, d_out); break;
        default: throw std::runtime_error("tiles: not a trellis type");
    }
}

void ck(cudaError_t e, const char* what) {
    if (e != cudaSuccess) throw std::runtime_error(std::string(what) + ": " + cudaGetErrorString(e));
}
template <typename T> T* dev(size_t n) {
    void* p = nullptr;
    ck(cudaMalloc(&p, n * sizeof(T) + 16), "cudaMalloc");
    ck(cudaMemset(p, 0, n * sizeof(T) + 16), "cudaMemset");
    return (T*) p;
}
template <typename T> T* up(const T* h, size_t n) {
    T* d = dev<T>(n);
    ck(cudaMemcpy(d, h, n * sizeof(T), cudaMemcpyHostToDevice), "upload");
    return d;
}
template <typename T> std::vector<T> down(const T* d, size_t n) {
    ck(cudaDeviceSynchronize(), "sync");
    std::vector<T> h(n);
    ck(cudaMemcpy(h.data(), d, n * sizeof(T), cudaMemcpyDeviceToHost), "download");
    return h;
}
float h2f(uint16_t b) { return __half2float(__ushort_as_half(b)); }
uint16_t f2h(float f) { return __half_as_ushort(__float2half_rn(f)); }

struct Report {
    int fails = 0;
    void bound(const std::string& what, double v, double lim) {
        const bool ok = v < lim;
        std::printf("  %-44s %.3e  (< %.0e) %s\n", what.c_str(), v, lim, ok ? "ok" : "FAIL");
        if (!ok) ++fails;
    }
    void equal(const std::string& what, size_t mism, size_t n) {
        std::printf("  %-44s %zu of %zu differ %s\n", what.c_str(), mism, n, mism ? "FAIL" : "ok");
        if (mism) ++fails;
    }
};
// max |a - ref| / max |ref|; infinity when any difference is not finite (a NaN output must fail its bound)
double rel_max(const std::vector<float>& a, const std::vector<double>& ref) {
    double m = 0, r = 0;
    for (size_t i = 0; i < a.size(); ++i) {
        const double d = std::fabs(a[i] - ref[i]);
        if (!std::isfinite(d)) return INFINITY;
        m = std::max(m, d), r = std::max(r, std::fabs(ref[i]));
    }
    return m / std::max(r, 1e-30);
}
double rel_l2(const std::vector<float>& a, const std::vector<double>& ref) {
    double d = 0, r = 0;
    for (size_t i = 0; i < a.size(); ++i) d += (a[i] - ref[i]) * (a[i] - ref[i]), r += ref[i] * ref[i];
    return std::sqrt(d / std::max(r, 1e-30));
}
// out[t][r] = sum_k w[r][k] x[t][k], in double
std::vector<double> dot(const std::vector<float>& w, const std::vector<float>& x, int rows, int cols, int T) {
    std::vector<double> o((size_t) T * rows);
    for (int t = 0; t < T; ++t)
        for (int r = 0; r < rows; ++r) {
            double s = 0;
            for (int k = 0; k < cols; ++k) s += (double) w[(size_t) r * cols + k] * x[(size_t) t * cols + k];
            o[(size_t) t * rows + r] = s;
        }
    return o;
}

constexpr int T = 5;

// swiglu_h16_had over more pairs than a grid's y extent (65535): a prompt chunk of WIDE_ROWS rows has WIDE_ROWS * TOPK pairs.
// Each card's rotated, zero-padded f16 h of sampled pairs (some past 65535) vs hadamard_rows of the same h made on the host.
constexpr int WIDE_ROWS = 7000;
void wide_swiglu(const float* d_sf, Report& rep) {
    const int P = WIDE_ROWS * TOPK;
    const std::vector<int> sample = {0, 1, 4095, 32767, 65534, 65535, 65536, 65537, 68000, P - 1};
    std::mt19937 rng(99);
    std::normal_distribution<float> nd(0.f, 1.f);
    for (int c = 0; c < 2; ++c) {
        const int ff0 = ff_lo(2, c), ff1 = ff_hi(2, c), ffs = ff1 - ff0;
        const int c0 = cover_lo(2, c, 128), c1 = cover_hi(2, c, 128), cw = c1 - c0;
        std::vector<float> gus((size_t) P * 2 * ffs);
        for (float& v : gus) v = nd(rng);
        float* d_gus = up(gus.data(), gus.size());
        uint16_t* d_h16 = dev<uint16_t>((size_t) P * cw);
        DW Dw;
        Dw.n_in = cw, Dw.had = d_sf + c0, Dw.had_off = ff0 - c0, Dw.had_ffs = ffs;
        const std::string what = "swiglu_h16_had " + std::to_string(P) + " pairs card " + std::to_string(c);
        try {
            swiglu_h16_had(d_gus, ffs, P, d_h16, Dw, nullptr);
            ck(cudaDeviceSynchronize(), "swiglu_h16_had");
        } catch (const std::exception& e) {
            std::printf("  %-44s %s FAIL\n", what.c_str(), e.what());
            ++rep.fails;
            cudaGetLastError();
            cudaFree(d_gus), cudaFree(d_h16);
            continue;
        }
        const int S = (int) sample.size();
        std::vector<float> hs((size_t) S * cw, 0.f);
        for (int s = 0; s < S; ++s)
            for (int k = 0; k < ffs; ++k) {
                const float* g = gus.data() + (size_t) sample[s] * 2 * ffs;
                hs[(size_t) s * cw + Dw.had_off + k] = g[k] / (1.0f + std::exp(-g[k])) * g[ffs + k];
            }
        float* d_hs = up(hs.data(), hs.size());
        float* d_hsr = dev<float>(hs.size());
        strata::kernels::hadamard_rows(d_hs, d_hsr, S, cw, HAD_BLOCK, d_sf + c0, nullptr);
        const std::vector<float> hr = down(d_hsr, hs.size());
        const std::vector<uint16_t> all = down(d_h16, (size_t) P * cw);
        std::vector<float> got(hs.size());
        std::vector<double> ref(hs.size());
        for (int s = 0; s < S; ++s)
            for (int k = 0; k < cw; ++k) got[(size_t) s * cw + k] = h2f(all[(size_t) sample[s] * cw + k]), ref[(size_t) s * cw + k] = hr[(size_t) s * cw + k];
        rep.bound(what, rel_max(got, ref), 1e-3);
        cudaFree(d_gus), cudaFree(d_h16), cudaFree(d_hs), cudaFree(d_hsr);
    }
}

void run_file(const std::string& path, Report& rep) {
    Gguf g(path);
    const HadSpec hs = had_spec(g, FF, N);
    std::printf("%s: rotation block %d, %zu rotated tensors\n", path.c_str(), hs.block, hs.names.size());
    float* d_sn = hs.signs_n.empty() ? nullptr : up(hs.signs_n.data(), hs.signs_n.size());
    float* d_sf = hs.signs_ff.empty() ? nullptr : up(hs.signs_ff.data(), hs.signs_ff.size());
    std::mt19937 rng(1234);
    std::normal_distribution<float> nd(0.f, 1.f);
    // jobs: one expert (index 0 of the uploaded matrices), T rows y = out = t
    Job jb{};
    jb.w = 0, jb.n = T;
    BigJob bj{};
    bj.w = 0, bj.n = T;
    for (int t = 0; t < T; ++t) jb.y[t] = jb.out[t] = bj.y[t] = bj.out[t] = t;
    const int one = 1;
    Job* d_jb = up(&jb, 1);
    BigJob* d_bj = up(&bj, 1);
    int* d_nj = up(&one, 1);
    for (const int l : {0, 23, 47}) {
        const std::string b = "blk." + std::to_string(l) + ".";
        const Tensor& tg = g.t(b + "ffn_gate_exps.weight");
        const Tensor& tu = g.t(b + "ffn_up_exps.weight");
        const Tensor& td = g.t(b + "ffn_down_exps.weight");
        if (!is_trellis(tg.type) || !is_trellis(td.type) || tu.type != tg.type) throw std::runtime_error(b + ": routed experts not in a trellis type");
        if (tg.ne[0] != N || tg.ne[1] != FF || td.ne[0] != FF || td.ne[1] != N) throw std::runtime_error(b + ": routed expert shapes");
        const bool rot_gu = hs.rotated(tg.name), rot_dn = hs.rotated(td.name);
        const size_t gu_row = tg.row_bytes(), dn_row = td.row_bytes(), gmat = (size_t) FF * gu_row, dmat = (size_t) N * dn_row;
        for (const int e : {0, 255, 511}) {
            std::printf(" layer %d expert %d (gate/up type %u%s, down type %u%s)\n", l, e, tg.type, rot_gu ? " rotated" : "", td.type,
                        rot_dn ? " rotated" : "");
            const uint8_t* hg = tg.data + (size_t) e * gmat;
            const uint8_t* hu = tu.data + (size_t) e * gmat;
            const uint8_t* hd = td.data + (size_t) e * dmat;
            uint8_t* d_g = up(hg, gmat);
            uint8_t* d_u = up(hu, gmat);
            uint8_t* d_d = up(hd, dmat);
            // ---- the activations, rotated by the main engine
            std::vector<float> x((size_t) T * N);
            for (float& v : x) v = nd(rng);
            float* d_x = up(x.data(), x.size());
            float* d_xr = dev<float>(x.size());
            if (rot_gu) strata::kernels::hadamard_rows(d_x, d_xr, T, N, HAD_BLOCK, d_sn, nullptr);
            else ck(cudaMemcpy(d_xr, d_x, x.size() * 4, cudaMemcpyDeviceToDevice), "copy");
            const std::vector<float> xr = down(d_xr, x.size());
            void* d_xq_ref = dev<uint8_t>(q8_1_bytes(N) * T);
            strata::kernels::quantize_q8_1_rows(d_xr, T, N, d_xq_ref, nullptr);
            // ---- rotation: split vs main
            uint16_t* d_xh = dev<uint16_t>(x.size());
            void* d_xq = dev<uint8_t>(q8_1_bytes(N) * T);
            if (rot_gu) {
                had_rows_f16(d_x, d_xh, N, T, d_sn, nullptr);
                had_quantize_q8_1(d_x, d_xq, N, T, d_sn, nullptr);
            } else {
                std::vector<uint16_t> xh(x.size());
                for (size_t i = 0; i < x.size(); ++i) xh[i] = f2h(x[i]);
                ck(cudaMemcpy(d_xh, xh.data(), xh.size() * 2, cudaMemcpyHostToDevice), "copy");
                quantize_q8_1(d_x, d_xq, N, T, nullptr);
            }
            const std::vector<uint16_t> xh = down(d_xh, x.size());
            std::vector<float> xh_f(x.size());
            size_t mism = 0;
            for (size_t i = 0; i < x.size(); ++i) mism += xh[i] != f2h(xr[i]), xh_f[i] = h2f(xh[i]);
            rep.equal("rotate f16 (had_rows_f16)", mism, x.size());
            {
                const std::vector<block_q8_1> a = down((const block_q8_1*) d_xq, (size_t) T * N / 32), r = down((const block_q8_1*) d_xq_ref, (size_t) T * N / 32);
                mism = 0;
                for (size_t i = 0; i < a.size(); ++i)
                    mism += std::memcmp(a[i].qs, r[i].qs, 32) != 0 || __low2float(a[i].ds) != __low2float(r[i].ds);
                rep.equal("rotate q8_1 (had_quantize_q8_1: qs, d)", mism, a.size());
            }
            // ---- the f16 weights: tiles vs the main engine's dequantizer
            uint16_t* d_w16 = dev<uint16_t>((size_t) FF * N);
            __half* d_t16 = dev<__half>((size_t) FF * N);
            std::vector<float> wg16, wu16, wd16;
            for (int m = 0; m < 3; ++m) {
                const uint8_t* dw = m == 0 ? d_g : m == 1 ? d_u : d_d;
                const uint32_t ty = m == 2 ? td.type : tg.type;
                const int rows = m == 2 ? N : FF, cols = m == 2 ? FF : N;
                strata::kernels::iq_dequant_f16((int) ty, dw, (int64_t) FF * N, d_w16, nullptr);
                tiles(ty, (const char*) dw, m == 2 ? dn_row : gu_row, rows, cols, d_t16);
                const std::vector<uint16_t> a = down(d_w16, (size_t) FF * N), t16 = down((const uint16_t*) d_t16, (size_t) FF * N);
                mism = 0;
                std::vector<float> wf(a.size());
                for (size_t i = 0; i < a.size(); ++i) mism += a[i] != t16[i], wf[i] = h2f(a[i]);
                rep.equal(std::string("tile ") + (m == 0 ? "gate" : m == 1 ? "up" : "down") + " (MmTile vs iq_dequant_f16)", mism, a.size());
                (m == 0 ? wg16 : m == 1 ? wu16 : wd16) = std::move(wf);
            }
            // ---- the main engine's gate|up rows (decode arithmetic): the reference for decode and the source of h
            float* d_gref = dev<float>((size_t) T * FF);
            float* d_uref = dev<float>((size_t) T * FF);
            strata::kernels::iq_mmvq((int) tg.type, d_g, d_xq_ref, d_gref, N, FF, T, nullptr);
            strata::kernels::iq_mmvq((int) tg.type, d_u, d_xq_ref, d_uref, N, FF, T, nullptr);
            const std::vector<float> gref = down(d_gref, (size_t) T * FF), uref = down(d_uref, (size_t) T * FF);
            const std::vector<double> gp = dot(wg16, xh_f, FF, N, T), upd = dot(wu16, xh_f, FF, N, T);
            // h and its rotation (main), the down references
            std::vector<float> h((size_t) T * FF);
            for (size_t i = 0; i < h.size(); ++i) h[i] = gref[i] / (1.0f + std::exp(-gref[i])) * uref[i];
            float* d_h = up(h.data(), h.size());
            float* d_hr = dev<float>(h.size());
            if (rot_dn) strata::kernels::hadamard_rows(d_h, d_hr, T, FF, HAD_BLOCK, d_sf, nullptr);
            else ck(cudaMemcpy(d_hr, d_h, h.size() * 4, cudaMemcpyDeviceToDevice), "copy");
            const std::vector<double> dref = dot(wd16, down(d_hr, h.size()), N, FF, T);
            void* d_hq = dev<uint8_t>(q8_1_bytes(FF) * T);
            strata::kernels::quantize_q8_1_rows(d_hr, T, FF, d_hq, nullptr);
            float* d_dmain = dev<float>((size_t) T * N);
            strata::kernels::iq_mmvq((int) td.type, d_d, d_hq, d_dmain, FF, N, T, nullptr);
            const double main_dn = rel_l2(down(d_dmain, (size_t) T * N), dref);
            // ---- the split, card by card
            std::vector<float> dsum((size_t) T * N, 0.f), msum((size_t) T * N, 0.f);
            for (int c = 0; c < 2; ++c) {
                const int ff0 = ff_lo(2, c), ff1 = ff_hi(2, c), ffs = ff1 - ff0;
                const int c0 = cover_lo(2, c, 128), c1 = cover_hi(2, c, 128), cw = c1 - c0;
                const std::string cs = " card " + std::to_string(c);
                // gate|up as the loader stacks them: [gate rows ff0..ff1; up rows ff0..ff1]
                std::vector<uint8_t> gu((size_t) 2 * ffs * gu_row);
                std::memcpy(gu.data(), hg + (size_t) ff0 * gu_row, (size_t) ffs * gu_row);
                std::memcpy(gu.data() + (size_t) ffs * gu_row, hu + (size_t) ff0 * gu_row, (size_t) ffs * gu_row);
                DW A;
                A.p = up(gu.data(), gu.size());
                A.type = tg.type, A.n_in = N, A.n_out = 2 * ffs, A.row_bytes = gu_row;
                A.had = rot_gu ? d_sn : nullptr;
                float* d_gu = dev<float>((size_t) T * 2 * ffs);
                moe_gate_up(A, gu.size(), d_jb, d_nj, T * TOPK, d_xq, d_gu, nullptr);
                std::vector<double> ref_gu((size_t) T * 2 * ffs), ref_gup((size_t) T * 2 * ffs);
                for (int t = 0; t < T; ++t)
                    for (int i = 0; i < ffs; ++i) {
                        ref_gu[(size_t) t * 2 * ffs + i] = gref[(size_t) t * FF + ff0 + i];
                        ref_gu[(size_t) t * 2 * ffs + ffs + i] = uref[(size_t) t * FF + ff0 + i];
                        ref_gup[(size_t) t * 2 * ffs + i] = gp[(size_t) t * FF + ff0 + i];
                        ref_gup[(size_t) t * 2 * ffs + ffs + i] = upd[(size_t) t * FF + ff0 + i];
                    }
                rep.bound("gu decode vs iq_mmvq" + cs, rel_max(down(d_gu, ref_gu.size()), ref_gu), 1e-4);
                float* d_gum = dev<float>((size_t) T * 2 * ffs);
                moe_gate_up_mma(A, gu.size(), d_bj, d_nj, 1, d_xh, d_gum, nullptr);
                rep.bound("gu prompt vs double dot" + cs, rel_max(down(d_gum, ref_gup.size()), ref_gup), 1e-3);
                // down, the card's cover of every row; gu from the main engine's rows (the same h on every path)
                std::vector<uint8_t> dn((size_t) N * (cw / 128) * (dn_row / (FF / 128)));
                const size_t brow = dn_row / (FF / 128), crow = (size_t) (cw / 128) * brow;
                for (int r = 0; r < N; ++r) std::memcpy(dn.data() + (size_t) r * crow, hd + (size_t) r * dn_row + (size_t) (c0 / 128) * brow, crow);
                DW Dw;
                Dw.p = up(dn.data(), dn.size());
                Dw.type = td.type, Dw.n_in = cw, Dw.n_out = N, Dw.row_bytes = crow;
                Dw.had = rot_dn ? d_sf + c0 : nullptr, Dw.had_off = ff0 - c0, Dw.had_ffs = ffs;
                // moe_down reads njobs_max = T * TOPK rows of gu (as engine.cu sizes b.gu): the T real rows, repeated
                std::vector<float> gus((size_t) T * TOPK * 2 * ffs);
                for (size_t i = 0; i < gus.size(); ++i) gus[i] = (float) ref_gu[i % ref_gu.size()];
                float* d_gus = up(gus.data(), gus.size());
                float* d_part = dev<float>((size_t) T * N);
                moe_down(Dw, dn.size(), d_jb, d_nj, T * TOPK, d_gus, d_part, nullptr);
                uint16_t* d_h16 = dev<uint16_t>((size_t) T * cw);
                swiglu_h16_had(d_gus, ffs, T, d_h16, Dw, nullptr);
                float* d_partm = dev<float>((size_t) T * N);
                moe_down_mma(Dw, dn.size(), d_bj, d_nj, 1, d_h16, d_partm, nullptr);
                const std::vector<float> pd = down(d_part, (size_t) T * N), pm = down(d_partm, (size_t) T * N);
                for (size_t i = 0; i < pd.size(); ++i) dsum[i] += pd[i], msum[i] += pm[i];
                for (void* p : {(void*) A.p, (void*) Dw.p, (void*) d_gu, (void*) d_gum, (void*) d_gus, (void*) d_part, (void*) d_h16, (void*) d_partm})
                    cudaFree(p);
            }
            std::printf("  %-44s %.3e\n", "down decode, main engine (iq_mmvq) vs double", main_dn);
            rep.bound("down decode, 2 cards summed vs double", rel_l2(dsum, dref), 3e-2);
            rep.bound("down prompt, 2 cards summed vs double", rel_l2(msum, dref), 5e-3);
            for (void* p : {(void*) d_g, (void*) d_u, (void*) d_d, (void*) d_x, (void*) d_xr, d_xq_ref, (void*) d_xh, d_xq, (void*) d_w16, (void*) d_t16,
                            (void*) d_gref, (void*) d_uref, (void*) d_h, (void*) d_hr, d_hq, (void*) d_dmain})
                cudaFree(p);
        }
    }
    if (d_sf) {
        std::printf(" the prompt's rotated h over %d rows x %d experts\n", WIDE_ROWS, TOPK);
        wide_swiglu(d_sf, rep);
    }
    cudaFree(d_jb), cudaFree(d_bj), cudaFree(d_nj), cudaFree(d_sn), cudaFree(d_sf);
}

}  // namespace
}  // namespace fnx

int main(int argc, char** argv) {
    if (argc < 2) {
        std::fprintf(stderr, "usage: split_trellis_test MODEL.gguf [MODEL.gguf ...]\n");
        return 2;
    }
    fnx::Report rep;
    // a NaN kernel output must not pass a bound
    rep.equal("rel_max of a NaN output is infinite", std::isinf(fnx::rel_max({NAN, 1.f}, {1.0, 1.0})) ? 0 : 1, 1);
    try {
        for (int i = 1; i < argc; ++i) fnx::run_file(argv[i], rep);
    } catch (const std::exception& e) {
        std::fprintf(stderr, "split_trellis_test: %s\n", e.what());
        return 2;
    }
    std::printf("%s: %d check(s) failed\n", rep.fails ? "FAIL" : "PASS", rep.fails);
    return rep.fails ? 1 : 0;
}
