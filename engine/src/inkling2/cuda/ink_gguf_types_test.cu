// engine/src/inkling2/cuda/ink_gguf_types_test.cu - parity test of the native GGUF expert row dots (ink_gguf_row_dot_multi) against a
// CPU reference written from the GGUF block specs (IQ2_XXS, IQ2_XS, IQ2_S, IQ3_S, IQ1_S, Q2_0; IQ3_XXS and IQ4_NL as controls).
//
// For random but valid blocks and random activations (quantised to Q8_1_G32 on the device), the dot of every weight row with every activation
// is computed on the GPU (one warp per row, MAX_M = 1 and 4 tokens) and on the host in double: weights dequantised from the spec, activations
// dequantised from the same Q8_1_G32 codes. Both use the same activation codes, so the only differences are float accumulation order.
// The codebook tables are read back from the device arrays (they are the GGUF format's data tables, generated from strata-quant-common.h).
//
// build: nvcc -O2 -std=c++20 --expt-relaxed-constexpr -arch=sm_86 -I engine/src/inkling2/cuda -I engine/include ink_gguf_types_test.cu -lcublas -o ink_gguf_types_test
//
// MIT License
// Copyright (c) 2026 Niko1221 and the Strata contributors
#include "ink_cuda.cuh"
#include "ink_kernels.cuh"
#include "ink_moe_cuda.cuh"

#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <random>
#include <vector>

using namespace strata::ink;

static float h_f16(uint16_t h) {
    const uint32_t s = (h >> 15) & 1u, e = (h >> 10) & 0x1Fu, m = h & 0x3FFu;
    double v;
    if (e == 0) v = std::ldexp((double)m, -24);
    else if (e == 31) v = 65504.0;
    else v = std::ldexp(1.0 + m / 1024.0, (int)e - 15);
    return (float)(s ? -v : v);
}
static uint16_t f16_bits(float f) {   // round to nearest, normal range only (test scales are 1e-3 .. 0.1)
    int ex; double fr = std::frexp((double)std::fabs(f), &ex);   // f = fr * 2^ex, fr in [0.5, 1)
    int e = ex - 1 + 15;
    int m = (int)std::lround((fr * 2.0 - 1.0) * 1024.0);
    if (m == 1024) { m = 0; ++e; }
    return (uint16_t)((f < 0 ? 0x8000u : 0u) | ((uint32_t)e << 10) | (uint32_t)m);
}

struct Tabs {
    std::vector<uint8_t> kmask, ksigns;
    std::vector<uint64_t> g2xxs, g2xs, g2s, g1s;
    std::vector<uint32_t> g3s, g3xxs;
};

static size_t blk_bytes(int t) {
    switch (t) { case 16: return 66; case 17: return 74; case 22: return 82; case 21: return 110; case 19: return 50; case 18: return 98; case 42: return 18; case 20: return 18; }
    return 0;
}
static int blk_elems(int t) { return t == 42 ? 64 : (t == 20 ? 32 : 256); }

// reference dequantisation of one row of K elements, from the block specs
static void ref_row(int t, const uint8_t* src, int K, const Tabs& T, double* y) {
    const int nb = K / blk_elems(t);
    for (int i = 0; i < nb; ++i) {
        const uint8_t* b = src + (size_t)i * blk_bytes(t);
        uint16_t dh; std::memcpy(&dh, b, 2);
        const double d = h_f16(dh);
        double* o = y + (size_t)i * blk_elems(t);
        if (t == 42) {
            for (int j = 0; j < 64; ++j) o[j] = (double)(((b[2 + j / 4] >> ((j % 4) * 2)) & 3) - 1) * d;
        } else if (t == 16) {
            for (int ib = 0; ib < 8; ++ib) {
                uint32_t a0, a1; std::memcpy(&a0, b + 2 + 8 * ib, 4); std::memcpy(&a1, b + 2 + 8 * ib + 4, 4);
                const double db = d * (0.5 + (a1 >> 28)) * 0.25;
                for (int l = 0; l < 4; ++l) {
                    const uint64_t g = T.g2xxs[(a0 >> (8 * l)) & 255];
                    const uint8_t sg = T.ksigns[(a1 >> (7 * l)) & 127];
                    for (int j = 0; j < 8; ++j) o[ib * 32 + l * 8 + j] = db * (double)((g >> (8 * j)) & 255) * ((sg & T.kmask[j]) ? -1.0 : 1.0);
                }
            }
        } else if (t == 17) {
            for (int ib = 0; ib < 8; ++ib) {
                const uint8_t sc = b[66 + ib];
                for (int l = 0; l < 4; ++l) {
                    uint16_t qv; std::memcpy(&qv, b + 2 + 8 * ib + 2 * l, 2);
                    const double db = d * (0.5 + ((l < 2) ? (sc & 15) : (sc >> 4))) * 0.25;
                    const uint64_t g = T.g2xs[qv & 511];
                    const uint8_t sg = T.ksigns[qv >> 9];
                    for (int j = 0; j < 8; ++j) o[ib * 32 + l * 8 + j] = db * (double)((g >> (8 * j)) & 255) * ((sg & T.kmask[j]) ? -1.0 : 1.0);
                }
            }
        } else if (t == 22) {
            const uint8_t* qs = b + 2; const uint8_t* qh = b + 66; const uint8_t* sgs = qs + 32; const uint8_t* scl = b + 74;
            for (int ib = 0; ib < 8; ++ib) {
                for (int l = 0; l < 4; ++l) {
                    const double db = d * (0.5 + ((l < 2) ? (scl[ib] & 15) : (scl[ib] >> 4))) * 0.25;
                    const uint64_t g = T.g2s[qs[ib * 4 + l] | ((qh[ib] << (8 - 2 * l)) & 0x300)];
                    const uint8_t sg = sgs[ib * 4 + l];
                    for (int j = 0; j < 8; ++j) o[ib * 32 + l * 8 + j] = db * (double)((g >> (8 * j)) & 255) * ((sg & T.kmask[j]) ? -1.0 : 1.0);
                }
            }
        } else if (t == 21) {
            const uint8_t* qs = b + 2; const uint8_t* qh = b + 66; const uint8_t* sgs = b + 74; const uint8_t* scl = b + 106;
            for (int ib = 0; ib < 8; ++ib) {
                const double db = d * (1 + 2 * ((ib & 1) ? (scl[ib / 2] >> 4) : (scl[ib / 2] & 15)));
                for (int l = 0; l < 4; ++l) {
                    const uint32_t g1 = T.g3s[qs[ib * 8 + 2 * l] | ((qh[ib] << (8 - 2 * l)) & 256)];
                    const uint32_t g2 = T.g3s[qs[ib * 8 + 2 * l + 1] | ((qh[ib] << (7 - 2 * l)) & 256)];
                    const uint8_t sg = sgs[ib * 4 + l];
                    for (int j = 0; j < 4; ++j) {
                        o[ib * 32 + l * 8 + j]     = db * (double)((g1 >> (8 * j)) & 255) * ((sg & T.kmask[j]) ? -1.0 : 1.0);
                        o[ib * 32 + l * 8 + j + 4] = db * (double)((g2 >> (8 * j)) & 255) * ((sg & T.kmask[j + 4]) ? -1.0 : 1.0);
                    }
                }
            }
        } else if (t == 19) {
            for (int ib = 0; ib < 8; ++ib) {
                uint16_t qh; std::memcpy(&qh, b + 34 + 2 * ib, 2);
                const double dl = d * (2 * ((qh >> 12) & 7) + 1);
                const double delta = (qh & 0x8000) ? -0.125 : 0.125;
                for (int l = 0; l < 4; ++l) {
                    const uint64_t g = T.g1s[b[2 + 4 * ib + l] | (((qh >> (3 * l)) & 7) << 8)];
                    for (int j = 0; j < 8; ++j) o[ib * 32 + l * 8 + j] = dl * ((double)(int8_t)((g >> (8 * j)) & 255) + delta);
                }
            }
        } else if (t == 18) {
            for (int ib = 0; ib < 8; ++ib) {
                uint32_t aux; std::memcpy(&aux, b + 66 + 4 * ib, 4);
                const double db = d * (0.5 + (aux >> 28)) * 0.5;
                for (int l = 0; l < 4; ++l) {
                    const uint8_t sg = T.ksigns[(aux >> (7 * l)) & 127];
                    const uint32_t g1 = T.g3xxs[b[2 + ib * 8 + 2 * l]], g2 = T.g3xxs[b[2 + ib * 8 + 2 * l + 1]];
                    for (int j = 0; j < 4; ++j) {
                        o[ib * 32 + l * 8 + j]     = db * (double)((g1 >> (8 * j)) & 255) * ((sg & T.kmask[j]) ? -1.0 : 1.0);
                        o[ib * 32 + l * 8 + j + 4] = db * (double)((g2 >> (8 * j)) & 255) * ((sg & T.kmask[j + 4]) ? -1.0 : 1.0);
                    }
                }
            }
        } else if (t == 20) {
            static const int lut[16] = {-127, -104, -83, -65, -49, -35, -22, -10, 1, 13, 25, 38, 53, 69, 89, 113};
            for (int j = 0; j < 16; ++j) { o[j] = d * lut[b[2 + j] & 15]; o[j + 16] = d * lut[b[2 + j] >> 4]; }
        }
    }
}

// KK > 0: K is a compile-time constant, as in the engine's kernels (k_ink_moe_gu_gguf_dp4a passes kInkHidden); KK == 0: K is the run-time argument
template <int MAXM, int KK = 0>
__global__ void k_row_dot_test(int type, const uint8_t* rows, size_t rb, const InkQ8_1_G32* xq, int nbx, int K, int n_e, float* out, int N) {
    const int row = blockIdx.x * 4 + (threadIdx.x >> 5);
    const int lane = threadIdx.x & 31;
    if (row >= N) return;
    const InkQ8_1_G32* xp[MAXM];
    const float* xf[MAXM];
    for (int m = 0; m < MAXM; ++m) { xp[m] = xq + (size_t)m * nbx; xf[m] = nullptr; }
    float acc[MAXM] = {};
    ink_gguf_row_dot_multi<MAXM>((uint32_t)type, rows + (size_t)row * rb, xp, xf, n_e, KK > 0 ? KK : K, lane, acc);
    if (lane == 0) for (int m = 0; m < MAXM; ++m) if (m < n_e) out[(size_t)m * N + row] = acc[m];
}

#define FROM_SYM(vec, sym, n) do { (vec).resize(n); INK_CK(cudaMemcpyFromSymbol((vec).data(), sym, (n) * sizeof((vec)[0]))); } while (0)

static int run_type(int t, int K, const Tabs& T, std::mt19937& rng) {
    const int N = 96, M = 4;
    const size_t rb = (size_t)(K / blk_elems(t)) * blk_bytes(t);
    std::vector<uint8_t> w(rb * N);
    std::uniform_int_distribution<int> du8(0, 255);
    for (auto& b : w) b = (uint8_t)du8(rng);
    std::uniform_real_distribution<float> dsc(0.002f, 0.05f);
    for (int r = 0; r < N; ++r)
        for (int i = 0; i < K / blk_elems(t); ++i) {
            float s = dsc(rng);
            if (t == 42 && (rng() & 1)) s = -s;   // Q2_0 scale is signed
            const uint16_t h = f16_bits(s);
            std::memcpy(&w[(size_t)r * rb + (size_t)i * blk_bytes(t)], &h, 2);
        }
    std::normal_distribution<float> dn(0.0f, 1.0f);
    std::vector<float> x((size_t)M * K);
    for (auto& v : x) v = dn(rng);
    // a few big outliers like the real activations
    for (int m = 0; m < M; ++m) x[(size_t)m * K + 2027 % K] *= 30.0f;

    uint8_t* d_w = dalloc<uint8_t>(w.size());
    float* d_x = dalloc<float>(x.size());
    const int nbx = K / 32;
    InkQ8_1_G32* d_xq = dalloc<InkQ8_1_G32>((size_t)M * nbx);
    float* d_out = dalloc<float>((size_t)M * N);
    INK_CK(cudaMemcpy(d_w, w.data(), w.size(), cudaMemcpyHostToDevice));
    INK_CK(cudaMemcpy(d_x, x.data(), x.size() * 4, cudaMemcpyHostToDevice));
    const int nblk_x = M * nbx;
    k_ink_quant_q8_1_g32<<<(nblk_x + 7) / 8, 256>>>(d_x, d_xq, M, K);
    INK_CK(cudaGetLastError());
    std::vector<InkQ8_1_G32> xq((size_t)M * nbx);
    INK_CK(cudaMemcpy(xq.data(), d_xq, xq.size() * sizeof(InkQ8_1_G32), cudaMemcpyDeviceToHost));
    std::vector<double> xd((size_t)M * K);
    for (int m = 0; m < M; ++m)
        for (int b = 0; b < nbx; ++b)
            for (int i = 0; i < 32; ++i) xd[(size_t)m * K + b * 32 + i] = (double)xq[(size_t)m * nbx + b].d * (double)xq[(size_t)m * nbx + b].qs[i];

    std::vector<double> ref((size_t)M * N), refabs((size_t)M * N);
    std::vector<double> y(K);
    for (int r = 0; r < N; ++r) {
        ref_row(t, &w[(size_t)r * rb], K, T, y.data());
        for (int m = 0; m < M; ++m) {
            double s = 0, sa = 0;
            for (int i = 0; i < K; ++i) { const double p = y[i] * xd[(size_t)m * K + i]; s += p; sa += std::fabs(p); }
            ref[(size_t)m * N + r] = s; refabs[(size_t)m * N + r] = sa;
        }
    }
    int fails = 0;
    std::vector<float> out((size_t)M * N);
    for (int variant = 0; variant < 2; ++variant) {
        const int n_e = variant == 0 ? 1 : M;
        INK_CK(cudaMemset(d_out, 0, out.size() * 4));
        if (variant == 0) k_row_dot_test<1><<<(N + 3) / 4, 128>>>(t, d_w, rb, d_xq, nbx, K, n_e, d_out, N);
        else              k_row_dot_test<4><<<(N + 3) / 4, 128>>>(t, d_w, rb, d_xq, nbx, K, n_e, d_out, N);
        INK_CK(cudaGetLastError());
        INK_CK(cudaDeviceSynchronize());
        INK_CK(cudaMemcpy(out.data(), d_out, out.size() * 4, cudaMemcpyDeviceToHost));
        double worst = 0; int bad = 0;
        for (int m = 0; m < n_e; ++m)
            for (int r = 0; r < N; ++r) {
                const double e = std::fabs((double)out[(size_t)m * N + r] - ref[(size_t)m * N + r]) / (refabs[(size_t)m * N + r] + 1e-30);
                worst = std::max(worst, e);
                if (e > 2e-5) ++bad;
            }
        std::printf("  type %2d K=%4d n_e=%d: worst relative error (vs sum|w x|) %.3g, %d of %d dots above 2e-5 %s\n", t, K, n_e, worst, bad, n_e * N, bad ? "FAIL" : "ok");
        fails += bad ? 1 : 0;
    }
    cudaFree(d_w); cudaFree(d_x); cudaFree(d_xq); cudaFree(d_out);
    return fails;
}

// The tensor-core prefill kernels against the dp4a kernels on one synthetic expert: 40 tokens routed to expert 0, random weights of the type, random activations.
// GU: k_ink_moe_gu_gguf_dp4a<16> against k_ink_moe_iq_tc<true, WT>; DN: k_ink_moe_dn_gguf_dp4a<16> against k_ink_moe_iq_tc<false, 42> / k_ink_moe_q80_tc<false, 20>.
static int run_tc_test(int wt, bool gu, std::mt19937& rng) {
    const int P = 40;
    const int Kg = kInkHidden, Kd = kInkInter;
    const uint32_t gu_type = gu ? (uint32_t)wt : 21u, dn_type = gu ? 20u : (uint32_t)wt;
    const size_t rb_gu = ink_gguf_row_bytes(gu_type, Kg), rb_dn = ink_gguf_row_bytes(dn_type, Kd);
    const size_t gu_bytes = (size_t)(2 * kInkInter) * rb_gu, dn_bytes = (size_t)kInkHidden * rb_dn;
    const uint32_t dn_off = (uint32_t)((32 + gu_bytes + 31) & ~(size_t)31);
    std::vector<uint8_t> body(dn_off + dn_bytes + 64);
    for (auto& b : body) b = (uint8_t)(rng() & 255);
    std::uniform_real_distribution<float> dsc(0.002f, 0.05f);
    auto fix_scales = [&](size_t start, size_t rows, size_t rb, int type) {
        const int be = (type == 42) ? 64 : ((type == 20 || type == 8) ? 32 : 256);
        const size_t bb = (type == 42 || type == 20) ? 18 : (type == 16 ? 66 : type == 17 ? 74 : type == 22 ? 82 : type == 21 ? 110 : type == 19 ? 50 : type == 18 ? 98 : 34);
        for (size_t r = 0; r < rows; ++r)
            for (size_t i = 0; i < rb / bb; ++i) {
                float s = dsc(rng);
                if (type == 42 && (rng() & 1)) s = -s;
                const uint16_t h = f16_bits(s);
                std::memcpy(&body[start + r * rb + i * bb], &h, 2);
            }
        (void)be;
    };
    fix_scales(32, 2 * kInkInter, rb_gu, (int)gu_type);
    fix_scales(dn_off, kInkHidden, rb_dn, (int)dn_type);
    InkGgufExpertHeader hdr{0x47475546u, gu_type, dn_type, 0u, 1.0f, 1.0f, 1.0f, 0u};
    std::memcpy(body.data(), &hdr, sizeof(hdr));
    uint8_t* d_body = dalloc<uint8_t>(body.size());
    INK_CK(cudaMemcpy(d_body, body.data(), body.size(), cudaMemcpyHostToDevice));
    InkExpertDev ex{0, dn_off, (uint8_t)FMT_GGUF_RAW, (uint8_t)FMT_GGUF_RAW, 0};
    InkExpertDev* d_ex = dalloc<InkExpertDev>(128);
    std::vector<InkExpertDev> hex(128, ex);
    INK_CK(cudaMemcpy(d_ex, hex.data(), 128 * sizeof(InkExpertDev), cudaMemcpyHostToDevice));
    InkMoeTables tb{};
    tb.cnt = dalloc<int32_t>(128); tb.off = dalloc<int32_t>(129); tb.active = dalloc<int32_t>(128); tb.nact = dalloc<int32_t>(1);
    tb.pairs = dalloc<int32_t>(P); tb.tok_idx = dalloc<int32_t>(P);
    std::vector<int32_t> h_cnt(128, 0), h_off(129, 0), h_act(128, 0), h_pairs(P), h_tok(P);
    h_cnt[0] = P; for (int i = 1; i <= 128; ++i) h_off[i] = P;
    for (int i = 0; i < P; ++i) { h_pairs[i] = i; h_tok[i] = i; }
    int32_t one = 1;
    INK_CK(cudaMemcpy(tb.cnt, h_cnt.data(), 128 * 4, cudaMemcpyHostToDevice));
    INK_CK(cudaMemcpy(tb.off, h_off.data(), 129 * 4, cudaMemcpyHostToDevice));
    INK_CK(cudaMemcpy(tb.active, h_act.data(), 128 * 4, cudaMemcpyHostToDevice));
    INK_CK(cudaMemcpy(tb.nact, &one, 4, cudaMemcpyHostToDevice));
    INK_CK(cudaMemcpy(tb.pairs, h_pairs.data(), P * 4, cudaMemcpyHostToDevice));
    INK_CK(cudaMemcpy(tb.tok_idx, h_tok.data(), P * 4, cudaMemcpyHostToDevice));
    std::normal_distribution<float> dn(0.0f, 1.0f);
    const int K = gu ? Kg : Kd, NBX = K / 32;
    std::vector<float> x((size_t)P * K);
    for (auto& v : x) v = dn(rng);
    float* d_x = dalloc<float>(x.size());
    INK_CK(cudaMemcpy(d_x, x.data(), x.size() * 4, cudaMemcpyHostToDevice));
    InkQ8_1_G32* d_xq = dalloc<InkQ8_1_G32>((size_t)P * NBX);
    k_ink_quant_q8_1_g32<<<(P * NBX + 7) / 8, 256>>>(d_x, d_xq, P, K);
    const size_t n_out = gu ? (size_t)P * kInkInter : (size_t)P * kInkHidden;
    float* d_a = dalloc<float>(n_out); float* d_b = dalloc<float>(n_out);
    float* d_w = dalloc<float>(P);
    std::vector<float> ones(P, 1.0f);
    INK_CK(cudaMemcpy(d_w, ones.data(), P * 4, cudaMemcpyHostToDevice));
    INK_CK(cudaMemset(d_a, 0, n_out * 4)); INK_CK(cudaMemset(d_b, 0, n_out * 4));
    if (gu) {
        k_ink_moe_gu_gguf_dp4a<16><<<dim3(kInkInter / 4, 1), 128>>>(d_body, d_ex, tb, d_xq, d_x, d_a, 0);
        const dim3 g(kInkInter / 64, 1);
        switch (wt) {
            case 16: k_ink_moe_iq_tc<true, 16><<<g, 128>>>(d_body, d_ex, tb, d_xq, nullptr, d_b); break;
            case 17: k_ink_moe_iq_tc<true, 17><<<g, 128>>>(d_body, d_ex, tb, d_xq, nullptr, d_b); break;
            case 18: k_ink_moe_iq_tc<true, 18><<<g, 128>>>(d_body, d_ex, tb, d_xq, nullptr, d_b); break;
            case 19: k_ink_moe_iq_tc<true, 19><<<g, 128>>>(d_body, d_ex, tb, d_xq, nullptr, d_b); break;
            case 21: k_ink_moe_iq_tc<true, 21><<<g, 128>>>(d_body, d_ex, tb, d_xq, nullptr, d_b); break;
            default: k_ink_moe_iq_tc<true, 22><<<g, 128>>>(d_body, d_ex, tb, d_xq, nullptr, d_b); break;
        }
    } else {
        k_ink_moe_dn_gguf_dp4a<16><<<dim3(kInkHidden / 4, 1), 128>>>(d_body, d_ex, tb, d_xq, d_x, d_w, d_a, 0);
        const dim3 g(kInkHidden / 64, 1);
        if (wt == 42) k_ink_moe_iq_tc<false, 42><<<g, 128>>>(d_body, d_ex, tb, d_xq, d_w, d_b);
        else          k_ink_moe_q80_tc<false, 20><<<g, 128>>>(d_body, d_ex, tb, d_xq, d_w, d_b);
    }
    INK_CK(cudaGetLastError());
    INK_CK(cudaDeviceSynchronize());
    std::vector<float> a(n_out), b(n_out);
    INK_CK(cudaMemcpy(a.data(), d_a, n_out * 4, cudaMemcpyDeviceToHost));
    INK_CK(cudaMemcpy(b.data(), d_b, n_out * 4, cudaMemcpyDeviceToHost));
    double ss = 0, worst = 0;
    for (size_t i = 0; i < n_out; ++i) ss += (double)a[i] * a[i];
    const double rms = std::sqrt(ss / n_out);
    for (size_t i = 0; i < n_out; ++i) worst = std::max(worst, std::fabs((double)a[i] - b[i]));
    std::printf("  tensor-core %s type %2d: max |tc - dp4a| %.3g, rms of the output %.3g -> %.2g relative %s\n", gu ? "gate/up" : "down", wt, worst, rms, worst / (rms + 1e-30),
                worst / (rms + 1e-30) < 2e-4 ? "ok" : "FAIL");
    cudaFree(d_body); cudaFree(d_ex); cudaFree(tb.cnt); cudaFree(tb.off); cudaFree(tb.active); cudaFree(tb.nact); cudaFree(tb.pairs); cudaFree(tb.tok_idx);
    cudaFree(d_x); cudaFree(d_xq); cudaFree(d_a); cudaFree(d_b); cudaFree(d_w);
    return worst / (rms + 1e-30) < 2e-4 ? 0 : 1;
}

// ./ink_gguf_types_test bench: time of the row-dot kernel per type on a weight block larger than the L2, for 1 token (decode) and 16 tokens (prefill group).
static size_t bench_rb(int t, int K) {
    switch (t) {
        case 12: return (size_t)(K / 256) * 144; case 14: return (size_t)(K / 256) * 210; case 8: return (size_t)(K / 32) * 34;
        case 23: return (size_t)(K / 256) * 136; default: return (size_t)(K / blk_elems(t)) * blk_bytes(t);
    }
}
static void bench_all() {
    const int types[] = {16, 17, 22, 21, 19, 18, 42, 20, 23, 12, 14, 8};
    const int N = 147456;
    for (int t : types) {
        const int K = (t == 42 || t == 20) ? 1152 : 2304;
        const size_t rb = bench_rb(t, K);
        std::vector<uint8_t> w(rb * (size_t)N);
        std::mt19937 rng(1);
        for (auto& b : w) b = (uint8_t)(rng() & 255);
        uint8_t* d_w = dalloc<uint8_t>(w.size());
        INK_CK(cudaMemcpy(d_w, w.data(), w.size(), cudaMemcpyHostToDevice));
        const int M = 16, nbx = K / 32;
        std::vector<float> x((size_t)M * K);
        std::normal_distribution<float> dn(0.0f, 1.0f);
        for (auto& v : x) v = dn(rng);
        float* d_x = dalloc<float>(x.size());
        INK_CK(cudaMemcpy(d_x, x.data(), x.size() * 4, cudaMemcpyHostToDevice));
        InkQ8_1_G32* d_xq = dalloc<InkQ8_1_G32>((size_t)M * nbx);
        k_ink_quant_q8_1_g32<<<(M * nbx + 7) / 8, 256>>>(d_x, d_xq, M, K);
        float* d_out = dalloc<float>((size_t)M * N);
        cudaEvent_t e0, e1;
        cudaEventCreate(&e0); cudaEventCreate(&e1);
        for (int variant = 0; variant < 2; ++variant) {
            const int n_e = variant == 0 ? 1 : 16;
            auto launch = [&]() {
                if (K == 2304) {
                    if (variant == 0) k_row_dot_test<1, 2304><<<(N + 3) / 4, 128>>>(t, d_w, rb, d_xq, nbx, K, n_e, d_out, N);
                    else              k_row_dot_test<16, 2304><<<(N + 3) / 4, 128>>>(t, d_w, rb, d_xq, nbx, K, n_e, d_out, N);
                } else {
                    if (variant == 0) k_row_dot_test<1, 1152><<<(N + 3) / 4, 128>>>(t, d_w, rb, d_xq, nbx, K, n_e, d_out, N);
                    else              k_row_dot_test<16, 1152><<<(N + 3) / 4, 128>>>(t, d_w, rb, d_xq, nbx, K, n_e, d_out, N);
                }
            };
            launch(); INK_CK(cudaDeviceSynchronize());
            cudaEventRecord(e0);
            for (int i = 0; i < 10; ++i) launch();
            cudaEventRecord(e1); INK_CK(cudaEventSynchronize(e1));
            float ms = 0; cudaEventElapsedTime(&ms, e0, e1); ms /= 10;
            std::printf("type %2d K=%4d %2d tokens: %7.3f ms  %7.1f GB/s of weights  %6.2f Gdot-elem/s/token-row\n", t, K, n_e, ms, (double)w.size() / (ms * 1e-3) / 1e9,
                        (double)N * K * n_e / (ms * 1e-3) / 1e9);
        }
        cudaFree(d_w); cudaFree(d_x); cudaFree(d_xq); cudaFree(d_out);
        cudaEventDestroy(e0); cudaEventDestroy(e1);
    }
}

int main(int argc, char** argv) {
    cudaDeviceProp p{};
    INK_CK(cudaGetDeviceProperties(&p, 0));
    std::printf("Device: %s (sm_%d%d)\n", p.name, p.major, p.minor);
    if (argc > 1 && std::strcmp(argv[1], "bench") == 0) { bench_all(); return 0; }
    Tabs T;
    FROM_SYM(T.kmask, kInkKmaskIq2, 8);
    FROM_SYM(T.ksigns, kInkKsignsIq2, 128);
    FROM_SYM(T.g2xxs, kInkIq2XxsGrid, 256);
    FROM_SYM(T.g2xs, kInkIq2XsGrid, 512);
    FROM_SYM(T.g2s, kInkIq2SGrid, 1024);
    FROM_SYM(T.g1s, kInkIq1SGrid, 2048);
    FROM_SYM(T.g3s, kInkIq3SGrid, 512);
    {   // the existing IQ3_XXS table is a __constant__ array
        T.g3xxs.resize(256);
        INK_CK(cudaMemcpyFromSymbol(T.g3xxs.data(), kInkIq3XxsGrid, 256 * 4));
    }
    std::mt19937 rng(20261126);
    int fails = 0;
    const int types[] = {16, 17, 22, 21, 19, 18};
    for (int t : types) fails += run_type(t, 2304, T, rng);
    fails += run_type(42, 1152, T, rng);
    fails += run_type(20, 1152, T, rng);
    std::printf("tensor-core prefill kernels against the dp4a kernels:\n");
    for (int t : {16, 17, 18, 19, 21, 22}) fails += run_tc_test(t, true, rng);
    fails += run_tc_test(42, false, rng);
    fails += run_tc_test(20, false, rng);
    if (fails == 0) std::printf("ALL GGUF TYPE ROW-DOT TESTS PASSED\n");
    else            std::printf("FAILED: %d\n", fails);
    return fails ? 1 : 0;
}
