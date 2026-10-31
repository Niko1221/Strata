// src/kernels/dense_tc_parity.cpp - dense_tc (INT8 tensor-core projections, BW-DENSE P2.1) against
//   (1) a host scalar reference written from the GGUF format and the arithmetic contract of dense_tc.hpp (bitwise),
//   (2) a double-precision reference on the same q8_1 bytes (how far old and new each are from the true dot),
//   (3) native_mmvq (the pre-existing dp4a path), reported as a difference, never as a parity claim,
// plus: T consistency (a column's bits do not depend on the column count), zero / extreme inputs, ineligible shapes
// and pointers, repeated CUDA-graph replay, two concurrent sessions, and with --bench the speed against the path the
// engine takes today (native_mmvq_il where its table says so, else native_mmvq).
//
//     build/dense_tc_parity [--bench] [--only substr] [--tile rt,ks] [--rounds N] [--heavy]
#include "strata/kernels/dense_tc.hpp"
#include "strata/kernels/iq_kernels.hpp"
#include "strata/kernels/native_mmvq.hpp"

#include <cuda_runtime.h>

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <random>
#include <string>
#include <vector>

using namespace strata::kernels;

namespace {

float h2f(uint16_t h) {   // IEEE half -> float, subnormals included
    const uint32_t s = (h >> 15) & 1, e = (h >> 10) & 31, m = h & 1023;
    float v;
    if (e == 0) v = std::ldexp(float(m), -24);
    else if (e == 31) v = m ? NAN : INFINITY;
    else v = std::ldexp(float(m + 1024), int(e) - 25);
    return s ? -v : v;
}
bool ck(cudaError_t e, const char* what) {
    if (e == cudaSuccess) return true;
    std::printf("CUDA: %s: %s\n", what, cudaGetErrorString(e));
    return false;
}

constexpr int kQ6 = 210;
struct Q6Row {   // one decoded row of blocks (natural value order, as dequantize_row_q6_K)
    std::vector<int8_t> q;      // nb * 256
    std::vector<int8_t> sc;     // nb * 16
    std::vector<float> d;       // nb
};
void decode_q6_row(const uint8_t* row, int nb, Q6Row& o) {
    o.q.assign(size_t(nb) * 256, 0);
    o.sc.assign(size_t(nb) * 16, 0);
    o.d.assign(nb, 0.f);
    for (int b = 0; b < nb; ++b) {
        const uint8_t* blk = row + size_t(b) * kQ6;
        const uint8_t *ql = blk, *qh = blk + 128;
        const int8_t* sc = reinterpret_cast<const int8_t*>(blk + 192);
        uint16_t dh;
        std::memcpy(&dh, blk + 208, 2);
        o.d[b] = h2f(dh);
        for (int i = 0; i < 16; ++i) o.sc[size_t(b) * 16 + i] = sc[i];
        int8_t* y = &o.q[size_t(b) * 256];
        for (int n = 0; n < 256; n += 128) {
            for (int l = 0; l < 32; ++l) {
                y[l + 0] = int8_t(((ql[l] & 0xF) | (((qh[l] >> 0) & 3) << 4)) - 32);
                y[l + 32] = int8_t(((ql[l + 32] & 0xF) | (((qh[l] >> 2) & 3) << 4)) - 32);
                y[l + 64] = int8_t(((ql[l] >> 4) | (((qh[l] >> 4) & 3) << 4)) - 32);
                y[l + 96] = int8_t(((ql[l + 32] >> 4) | (((qh[l] >> 6) & 3) << 4)) - 32);
            }
            y += 128;
            ql += 64;
            qh += 32;
        }
    }
}

// the contract's order for one (row, column); `ks` warps own blocks b = k, k + ks, ...
float contract_dot(const Q6Row& r, const uint8_t* xq_col, int nb, int ks) {
    float part[16] = {};
    for (int k = 0; k < ks; ++k) {
        float acc = 0.f;
        for (int b = k; b < nb; b += ks) {
            float inner = 0.f;
            for (int a = 0; a < 8; ++a) {
                const uint8_t* xb = xq_col + size_t(8 * b + a) * 36;
                uint16_t dh;
                std::memcpy(&dh, xb, 2);
                const float d8 = h2f(dh);
                const int8_t* u = reinterpret_cast<const int8_t*>(xb + 4);
                int S = 0;
                for (int half = 0; half < 2; ++half) {
                    int D = 0;
                    for (int j = 0; j < 16; ++j) D += int(r.q[size_t(b) * 256 + a * 32 + half * 16 + j]) * int(u[half * 16 + j]);
                    S += int(r.sc[size_t(b) * 16 + 2 * a + half]) * D;
                }
                inner = std::fmaf(d8, float(S), inner);
            }
            acc = std::fmaf(r.d[b], inner, acc);
        }
        part[k] = acc;
    }
    float v = part[0];
    for (int k = 1; k < ks; ++k) v += part[k];
    return v;
}
double exact_dot(const Q6Row& r, const uint8_t* xq_col, int nb) {
    double tot = 0;
    for (int b = 0; b < nb; ++b) {
        double inner = 0;
        for (int a = 0; a < 8; ++a) {
            const uint8_t* xb = xq_col + size_t(8 * b + a) * 36;
            uint16_t dh;
            std::memcpy(&dh, xb, 2);
            const double d8 = h2f(dh);
            const int8_t* u = reinterpret_cast<const int8_t*>(xb + 4);
            long long S = 0;
            for (int half = 0; half < 2; ++half) {
                long long D = 0;
                for (int j = 0; j < 16; ++j) D += (long long) r.q[size_t(b) * 256 + a * 32 + half * 16 + j] * u[half * 16 + j];
                S += (long long) r.sc[size_t(b) * 16 + 2 * a + half] * D;
            }
            inner += d8 * double(S);
        }
        tot += double(r.d[b]) * inner;
    }
    return tot;
}

struct Case { const char* name; int n_in, n_out; };
const Case CASES[] = {
    {"Q6_K 2560x10240", 2560, 10240}, {"Q6_K 2560x12288", 2560, 12288}, {"Q6_K 6144x2560", 6144, 2560},
    {"Q6_K 2560x6144", 2560, 6144},   {"Q6_K 2560x640", 2560, 640},     {"Q6_K 2560x512", 2560, 512},
    {"Q6_K odd rows 2560x2051", 2560, 2051}, {"Q6_K tail 256x37", 256, 37}, {"Q6_K 512x16", 512, 16},
    {"Q6_K head 2560x248320", 2560, 248320},
};

enum WMode { W_RANDOM, W_MAXCODE, W_MINCODE, W_EXTREME, W_SUBNORMAL_D };
enum XMode { X_NORMAL, X_ZERO, X_LARGE, X_SPIKE };
const char* WN[] = {"random", "max-codes", "min-codes", "extremes", "subnormal-d"};
const char* XN[] = {"normal", "zero", "large", "spike"};

void fill_weights(std::vector<uint8_t>& w, size_t n_blocks, WMode m, std::mt19937& rng) {
    for (size_t k = 0; k < n_blocks; ++k) {
        uint8_t* b = &w[k * kQ6];
        for (int i = 0; i < 208; ++i) {
            switch (m) {
            case W_MAXCODE: b[i] = (i < 192) ? 0xFF : 0x7F; break;
            case W_MINCODE: b[i] = (i < 192) ? 0x00 : 0x80; break;
            case W_EXTREME: b[i] = (i < 192) ? ((rng() & 1) ? 0xFF : 0x00) : ((rng() & 1) ? 0x7F : 0x80); break;
            default: b[i] = uint8_t(rng() & 0xff); break;
            }
        }
        uint16_t d;
        if (m == W_SUBNORMAL_D) d = uint16_t(1 + rng() % 1023);   // positive subnormal halves
        else {
            const uint32_t r = rng();
            d = uint16_t(((r >> 31) << 15) | ((5u + (r >> 10) % 5u) << 10) | (r & 0x3ffu));
        }
        std::memcpy(b + 208, &d, 2);
    }
}
void fill_x(std::vector<float>& x, int T, int n_in, XMode m, std::mt19937& rng) {
    std::normal_distribution<float> nd(0.f, 1.f);
    for (int c = 0; c < T; ++c)
        for (int i = 0; i < n_in; ++i) {
            float v = nd(rng);
            if (m == X_ZERO) v = 0.f;
            else if (m == X_LARGE) v = (rng() & 1) ? 1000.f : -1000.f;
            else if (m == X_SPIKE) v = (i % 64 == 3) ? 50.f * v : 0.01f * v;
            x[size_t(c) * n_in + i] = v;
        }
}

struct Stats { double max_rel_new_vs_exact = 0, max_rel_old_vs_exact = 0, max_rel_new_vs_old = 0; };

int bad_total = 0;
#define EXPECT(cond, ...) do { if (!(cond)) { ++bad_total; std::printf("FAIL: "); std::printf(__VA_ARGS__); std::printf("\n"); } } while (0)

// returns the number of failures
int run_case(const Case& c, WMode wm, XMode xm, bool heavy, cudaStream_t s) {
    const int nb = c.n_in / 256;
    const size_t wbytes = native_mmvq_weight_bytes(14, c.n_in, c.n_out);
    std::mt19937 rng(1234u + unsigned(c.n_out) * 7u + unsigned(wm) * 101u + unsigned(xm) * 13u);
    std::vector<uint8_t> w(wbytes);
    fill_weights(w, size_t(c.n_out) * nb, wm, rng);
    uint8_t* dw = nullptr;
    if (!ck(cudaMalloc(&dw, wbytes + 64), "malloc w") || !ck(cudaMemcpy(dw, w.data(), wbytes, cudaMemcpyHostToDevice), "copy w") || !ck(cudaDeviceSynchronize(), "sync w")) return 1;

    // rows to check on the host: all of them, or a sample for the big shapes
    std::vector<int> rows;
    const int max_rows = heavy ? 100000000 : 600;
    if (c.n_out <= max_rows) for (int r = 0; r < c.n_out; ++r) rows.push_back(r);
    else {
        for (int r = 0; r < 24; ++r) { rows.push_back(r); rows.push_back(c.n_out - 1 - r); }
        for (int i = 0; i < 400; ++i) rows.push_back(int(rng() % unsigned(c.n_out)));
    }
    std::vector<Q6Row> dec(rows.size());
    for (size_t i = 0; i < rows.size(); ++i) decode_q6_row(w.data() + size_t(rows[i]) * nb * kQ6, nb, dec[i]);

    const int ks = dense_tc_split_for(14, c.n_in, c.n_out);
    int fails = 0;
    std::vector<std::vector<float>> ycols_by_T;   // for the T-consistency check: column 0..T-1 of the T=8 call
    std::vector<float> y8;
    for (int T = 1; T <= 8; ++T) {
        std::vector<float> x(size_t(T) * c.n_in);
        // the same first column for every T (embedding the token in a window): columns are generated per column index
        {
            std::mt19937 rx(777u + unsigned(xm));
            std::vector<float> all(size_t(8) * c.n_in);
            fill_x(all, 8, c.n_in, xm, rx);
            std::copy(all.begin(), all.begin() + size_t(T) * c.n_in, x.begin());
        }
        float* dx = nullptr;
        void* xq = nullptr;
        float *ynew = nullptr, *yold = nullptr;
        const size_t ybytes = size_t(T) * c.n_out * 4;
        if (!ck(cudaMalloc(&dx, x.size() * 4), "x") || !ck(cudaMalloc(&xq, native_q8_1_bytes(c.n_in, T) + 64), "xq") ||
            !ck(cudaMalloc(&ynew, ybytes), "ynew") || !ck(cudaMalloc(&yold, ybytes), "yold"))
            return 1;
        cudaMemcpy(dx, x.data(), x.size() * 4, cudaMemcpyHostToDevice);
        cudaDeviceSynchronize();   // a pageable H2D copy returns before its DMA lands; the kernels run on a non-blocking stream
        quantize_q8_1_rows(dx, T, c.n_in, xq, s);
        cudaMemsetAsync(ynew, 0xff, ybytes, s);   // on the kernel's own stream: a legacy-stream memset does not order against a non-blocking stream
        const bool launched = dense_tc(14, dw, xq, ynew, c.n_in, c.n_out, T, s);
        EXPECT(launched, "%s T=%d: dense_tc did not launch (%s)", c.name, T, dense_tc_reason(dense_tc_check(14, c.n_in, c.n_out, T, dw, xq, ynew)));
        native_mmvq(14, dw, xq, yold, c.n_in, c.n_out, T, s);
        ck(cudaStreamSynchronize(s), "sync");
        ck(cudaGetLastError(), "launch");
        std::vector<float> hn(size_t(T) * c.n_out), ho(hn.size());
        cudaMemcpy(hn.data(), ynew, ybytes, cudaMemcpyDeviceToHost);
        cudaMemcpy(ho.data(), yold, ybytes, cudaMemcpyDeviceToHost);
        std::vector<uint8_t> hq(native_q8_1_bytes(c.n_in, T));
        cudaMemcpy(hq.data(), xq, hq.size(), cudaMemcpyDeviceToHost);

        long long bitdiff = 0, nonfinite = 0;
        double maxrel_new = 0, maxrel_old = 0, maxrel_nv = 0, scale = 0;
        // an output-scale for relative errors: the largest exact magnitude of the checked rows of this column
        for (int col = 0; col < T; ++col) {
            const uint8_t* xc = hq.data() + size_t(col) * (c.n_in / 32) * 36;
            double colmax = 0;
            std::vector<double> ex(rows.size());
            for (size_t i = 0; i < rows.size(); ++i) { ex[i] = exact_dot(dec[i], xc, nb); colmax = std::max(colmax, std::fabs(ex[i])); }
            scale = std::max(colmax, 1e-30);
            for (size_t i = 0; i < rows.size(); ++i) {
                const int r = rows[i];
                const float got = hn[size_t(col) * c.n_out + r];
                const float want = contract_dot(dec[i], xc, nb, ks);
                uint32_t a, b;
                std::memcpy(&a, &got, 4);
                std::memcpy(&b, &want, 4);
                if (a != b) ++bitdiff;
                if (!std::isfinite(got)) ++nonfinite;
                maxrel_new = std::max(maxrel_new, std::fabs(double(got) - ex[i]) / scale);
                maxrel_old = std::max(maxrel_old, std::fabs(double(ho[size_t(col) * c.n_out + r]) - ex[i]) / scale);
                maxrel_nv = std::max(maxrel_nv, std::fabs(double(got) - double(ho[size_t(col) * c.n_out + r])) / scale);
            }
        }
        if (nonfinite) {   // which columns / rows hold the non-finite values
            for (int col = 0; col < T; ++col) {
                long long nf = 0, first = -1;
                for (int r = 0; r < c.n_out; ++r) if (!std::isfinite(hn[size_t(col) * c.n_out + r])) { if (first < 0) first = r; ++nf; }
                if (nf) std::printf("    column %d: %lld of %d rows non-finite (first row %lld)\n", col, nf, c.n_out, first);
            }
        }
        if (bitdiff || nonfinite) {
            ++fails;
            EXPECT(false, "%s [%s w, %s x] T=%d: %lld outputs differ from the contract reference, %lld non-finite", c.name, WN[wm], XN[xm], T, bitdiff, nonfinite);
        }
        // the error against the exact dot must be of the old path's size (both are fp32 sums of exact integer parts)
        EXPECT(maxrel_new <= std::max(4.0 * maxrel_old, 2e-6), "%s [%s w, %s x] T=%d: new error %.3g vs old %.3g (of the column max)", c.name, WN[wm], XN[xm], T, maxrel_new, maxrel_old);
        if (wm == W_RANDOM && xm == X_NORMAL && (T == 1 || T == 4 || T == 8))
            std::printf("  %-26s T=%d  err vs exact (of column max): new %.2e  old %.2e   new vs old %.2e   [k_split %d, %zu rows checked]\n", c.name, T,
                        maxrel_new, maxrel_old, maxrel_nv, ks, rows.size());
        // column-count independence: the T=8 call is the reference for every smaller T
        if (T == 8) y8 = hn;
        ycols_by_T.push_back(hn);
        cudaFree(dx); cudaFree(xq); cudaFree(ynew); cudaFree(yold);
    }
    // each column's bits at every T equal its bits in the T=8 call (T=8 columns are the same data)
    for (int T = 1; T <= 7; ++T) {
        long long diff = 0;
        for (int col = 0; col < T; ++col)
            for (int r = 0; r < c.n_out; ++r)
                if (std::memcmp(&ycols_by_T[T - 1][size_t(col) * c.n_out + r], &y8[size_t(col) * c.n_out + r], 4) != 0) {
                    if (diff < 4) std::printf("    T=%d col %d row %d: %.9g vs %.9g in the T=8 call\n", T, col, r, ycols_by_T[T - 1][size_t(col) * c.n_out + r], y8[size_t(col) * c.n_out + r]);
                    ++diff;
                }
        EXPECT(diff == 0, "%s [%s w, %s x]: T=%d differs from the same columns in T=8 in %lld outputs", c.name, WN[wm], XN[xm], T, diff);
        if (diff) ++fails;
    }
    cudaFree(dw);
    return fails;
}

void check_ineligible() {
    // pointers and shapes the kernel must refuse, with the stable reason
    void *w = nullptr, *x = nullptr, *y = nullptr;
    cudaMalloc(&w, 1 << 20);
    cudaMalloc(&x, 1 << 20);
    cudaMalloc(&y, 1 << 20);
    auto chk = [&](const char* what, int got, int want) { EXPECT(got == want, "%s: reason %s, wanted %s", what, dense_tc_reason(got), dense_tc_reason(want)); };
    chk("ok", dense_tc_check(14, 2560, 640, 4, w, x, y), kDenseTcOk);
    chk("ncols 0", dense_tc_check(14, 2560, 640, 0, w, x, y), kDenseTcCols);
    chk("ncols 9", dense_tc_check(14, 2560, 640, 9, w, x, y), kDenseTcCols);
    chk("n_in tail", dense_tc_check(14, 2560 + 32, 640, 4, w, x, y), kDenseTcShape);
    chk("n_out 0", dense_tc_check(14, 2560, 0, 4, w, x, y), kDenseTcShape);
    chk("type Q8_0", dense_tc_check(8, 2560, 640, 4, w, x, y), kDenseTcType);
    chk("weights odd pointer", dense_tc_check(14, 2560, 640, 4, static_cast<char*>(w) + 1, x, y), kDenseTcAlign);
    chk("activation pointer +2", dense_tc_check(14, 2560, 640, 4, w, static_cast<char*>(x) + 2, y), kDenseTcAlign);
    chk("output pointer +2", dense_tc_check(14, 2560, 640, 4, w, x, static_cast<char*>(y) + 2), kDenseTcAlign);
    chk("weights 2-byte aligned is fine", dense_tc_check(14, 2560, 640, 4, static_cast<char*>(w) + 2, x, y), kDenseTcOk);
    EXPECT(!dense_tc(14, w, x, static_cast<float*>(y), 2560 + 32, 640, 4, nullptr), "a refused call must not launch");
    dense_tc_enable(0);
    chk("switch off", dense_tc_check(14, 2560, 640, 4, w, x, y), kDenseTcOff);
    dense_tc_enable(1);
    cudaFree(w); cudaFree(x); cudaFree(y);
}

// repeated graph replay and two concurrent sessions on distinct inputs
void check_graph_and_sessions() {
    const int n_in = 2560, n_out = 6144, T = 5;
    const size_t wbytes = native_mmvq_weight_bytes(14, n_in, n_out);
    std::mt19937 rng(99);
    std::vector<uint8_t> w(wbytes);
    fill_weights(w, size_t(n_out) * (n_in / 256), W_RANDOM, rng);
    uint8_t* dw;
    cudaMalloc(&dw, wbytes);
    cudaMemcpy(dw, w.data(), wbytes, cudaMemcpyHostToDevice);
    cudaStream_t s[2];
    cudaStreamCreateWithFlags(&s[0], cudaStreamNonBlocking);
    cudaStreamCreateWithFlags(&s[1], cudaStreamNonBlocking);
    float* dx[2];
    void* xq[2];
    float *ysolo[2], *yg[2];
    std::vector<float> hsolo[2];
    for (int k = 0; k < 2; ++k) {
        std::vector<float> x(size_t(T) * n_in);
        fill_x(x, T, n_in, X_NORMAL, rng);
        cudaMalloc(&dx[k], x.size() * 4);
        cudaMalloc(&xq[k], native_q8_1_bytes(n_in, T));
        cudaMalloc(&ysolo[k], size_t(T) * n_out * 4);
        cudaMalloc(&yg[k], size_t(T) * n_out * 4);
        cudaMemcpy(dx[k], x.data(), x.size() * 4, cudaMemcpyHostToDevice);
        quantize_q8_1_rows(dx[k], T, n_in, xq[k], s[k]);
        dense_tc(14, dw, xq[k], ysolo[k], n_in, n_out, T, s[k]);
        cudaStreamSynchronize(s[k]);
        hsolo[k].resize(size_t(T) * n_out);
        cudaMemcpy(hsolo[k].data(), ysolo[k], hsolo[k].size() * 4, cudaMemcpyDeviceToHost);
    }
    cudaGraph_t g[2];
    cudaGraphExec_t ge[2];
    for (int k = 0; k < 2; ++k) {
        cudaStreamBeginCapture(s[k], cudaStreamCaptureModeThreadLocal);
        EXPECT(dense_tc(14, dw, xq[k], yg[k], n_in, n_out, T, s[k]), "launch under capture");
        cudaStreamEndCapture(s[k], &g[k]);
        EXPECT(ck(cudaGraphInstantiate(&ge[k], g[k], 0), "instantiate"), "graph instantiate");
    }
    long long diff = 0;
    for (int rep = 0; rep < 200; ++rep) {   // both sessions replay concurrently, distinct inputs
        for (int k = 0; k < 2; ++k) cudaGraphLaunch(ge[k], s[k]);
        if (rep % 20 == 19) {
            for (int k = 0; k < 2; ++k) {
                cudaStreamSynchronize(s[k]);
                std::vector<float> h(hsolo[k].size());
                cudaMemcpy(h.data(), yg[k], h.size() * 4, cudaMemcpyDeviceToHost);
                diff += std::memcmp(h.data(), hsolo[k].data(), h.size() * 4) != 0;
                cudaMemsetAsync(yg[k], 0xff, h.size() * 4, s[k]);
            }
        }
    }
    cudaDeviceSynchronize();
    EXPECT(diff == 0, "graph replay / two sessions: %lld replays differed from the solo launch", diff);
    EXPECT(ck(cudaGetLastError(), "graph run"), "graph run error");
    std::printf("  graph replay x200, two concurrent sessions: %s\n", diff == 0 ? "identical to the solo launch" : "DIFFERENT");
}

// ---- bench ---------------------------------------------------------------------------------------------------------
template <class F>
double time_us(F fn, int reps, cudaStream_t s) {
    cudaEvent_t e0, e1;
    cudaEventCreate(&e0);
    cudaEventCreate(&e1);
    float ms = 0;
    for (int rep = 0; rep < 2; ++rep) {
        cudaEventRecord(e0, s);
        for (int i = 0; i < reps; ++i) fn(i);
        cudaEventRecord(e1, s);
        cudaEventSynchronize(e1);
        cudaEventElapsedTime(&ms, e0, e1);
    }
    cudaEventDestroy(e0);
    cudaEventDestroy(e1);
    return 1000.0 * ms / reps;
}

void bench_case(const Case& c, int rounds, cudaStream_t s) {
    const size_t wbytes = native_mmvq_weight_bytes(14, c.n_in, c.n_out);
    const int copies = int(std::max<size_t>(1, std::min<size_t>(24, (size_t(256) << 20) / wbytes)));
    std::mt19937 rng(5);
    std::vector<uint8_t> w(wbytes);
    fill_weights(w, size_t(c.n_out) * (c.n_in / 256), W_RANDOM, rng);
    std::vector<uint8_t*> dw(copies);
    for (auto& p : dw) { cudaMalloc(&p, wbytes); cudaMemcpy(p, w.data(), wbytes, cudaMemcpyHostToDevice); }
    for (int T = 1; T <= 8; ++T) {
        std::vector<float> x(size_t(T) * c.n_in);
        fill_x(x, T, c.n_in, X_NORMAL, rng);
        float* dx;
        void *xq, *xil;
        float *yn, *yo;
        cudaMalloc(&dx, x.size() * 4);
        cudaMalloc(&xq, native_q8_1_bytes(c.n_in, T));
        cudaMalloc(&xil, native_q8_1_il_bytes(c.n_in, T));
        cudaMalloc(&yn, size_t(T) * c.n_out * 4);
        cudaMalloc(&yo, size_t(T) * c.n_out * 4);
        cudaMemcpy(dx, x.data(), x.size() * 4, cudaMemcpyHostToDevice);
        cudaDeviceSynchronize();   // a pageable H2D copy returns before its DMA lands; the kernels run on a non-blocking stream
        quantize_q8_1_rows(dx, T, c.n_in, xq, s);
        if (T >= 2) native_q8_1_interleave(xq, xil, c.n_in, T, s);
        const bool il = native_mmvq_il_supported(14, T, c.n_out);
        auto old_call = [&](int i) {
            if (il) native_mmvq_il(14, dw[i % copies], xq, xil, yo, c.n_in, c.n_out, T, s);
            else native_mmvq(14, dw[i % copies], xq, yo, c.n_in, c.n_out, T, s);
        };
        auto new_call = [&](int i) { dense_tc(14, dw[i % copies], xq, yn, c.n_in, c.n_out, T, s); };
        std::vector<double> to, tn;
        for (int r = 0; r < rounds; ++r) {   // interleaved rounds
            to.push_back(time_us(old_call, 100, s));
            tn.push_back(time_us(new_call, 100, s));
        }
        std::sort(to.begin(), to.end());
        std::sort(tn.begin(), tn.end());
        const double mo = to[to.size() / 2], mn = tn[tn.size() / 2];
        std::printf("BENCH %-26s T=%d  old(%s) %7.1f us %5.0f GB/s | dense_tc %7.1f us %5.0f GB/s | ratio %.3f\n", c.name, T, il ? "il " : "mmvq", mo,
                    wbytes / mo / 1e3, mn, wbytes / mn / 1e3, mn / mo);
        cudaFree(dx); cudaFree(xq); cudaFree(xil); cudaFree(yn); cudaFree(yo);
    }
    for (auto p : dw) cudaFree(p);
}

}  // namespace

int main(int argc, char** argv) {
    bool bench = false, heavy = false;
    int rounds = 5;
    std::string only;
    for (int i = 1; i < argc; ++i) {
        if (!std::strcmp(argv[i], "--bench")) bench = true;
        else if (!std::strcmp(argv[i], "--heavy")) heavy = true;
        else if (!std::strcmp(argv[i], "--only") && i + 1 < argc) only = argv[++i];
        else if (!std::strcmp(argv[i], "--rounds") && i + 1 < argc) rounds = std::atoi(argv[++i]);
        else if (!std::strcmp(argv[i], "--tile") && i + 1 < argc) { int a = 0, b = 0; std::sscanf(argv[++i], "%d,%d", &a, &b); dense_tc_tile(a, b); }
    }
    dense_tc_enable(1);
    cudaStream_t s;
    cudaStreamCreateWithFlags(&s, cudaStreamNonBlocking);
    if (dense_tc_check(14, 2560, 640, 4, nullptr, nullptr, nullptr) != kDenseTcOk) {
        std::printf("dense_tc_parity: SKIP (%s: not a 12.0 card or not built)\n", dense_tc_reason(dense_tc_check(14, 2560, 640, 4, nullptr, nullptr, nullptr)));
        return 0;
    }
    dense_tc_print_resources();
    if (bench) {
        for (const Case& c : CASES) if (only.empty() || std::strstr(c.name, only.c_str())) bench_case(c, rounds, s);
        return 0;
    }
    check_ineligible();
    for (const Case& c : CASES) {
        if (!only.empty() && !std::strstr(c.name, only.c_str())) continue;
        std::printf("%s\n", c.name);
        const bool big = c.n_out > 100000;
        for (int wm = 0; wm <= W_SUBNORMAL_D; ++wm)
            for (int xm = 0; xm <= X_SPIKE; ++xm) {
                if (big && (wm != W_RANDOM || xm != X_NORMAL) && !(wm == W_EXTREME && xm == X_LARGE)) continue;   // the head: two modes (it is 521 MB)
                run_case(c, WMode(wm), XMode(xm), heavy, s);
            }
    }
    check_graph_and_sessions();
    if (bad_total) {
        std::printf("dense_tc_parity: %d FAILURES\n", bad_total);
        return 1;
    }
    std::printf("dense_tc_parity: OK\n");
    return 0;
}
