// src/prefill/rp_expert_bench.cpp - gfx906: the repacked IQ4_NL expert GEMM (rp_moe.hpp) against llama.cpp's MMQ at
// the prompt path's shapes, for correctness and time.
//
//     rp_expert_bench [rows per expert ...]
//
// One group as the prompt path launches it: 16 experts, R rows each, gate/up [1280 x 2560] and down [2560 x 640], in
// IQ4_NL.  Synthetic weights (random codes, small positive f16 scales), random activations of row-varying magnitude,
// the same q8_1 activations for both.  Per product: the largest |rp - mmq| relative to the largest |mmq| (the
// products sum in another order, so not bitwise), then medians of 25 runs each.  Default R: 12, 80 (a 4096-token
// chunk: 4096 x 10 / 512), 160, 626.
#include "strata/prefill/moe_mmq.hpp"
#include "strata/prefill/rp_moe.hpp"

#include <cuda_runtime.h>

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <random>
#include <vector>

namespace mmq = strata::prefill::mmq;
namespace rp = strata::prefill::rp;

namespace {

constexpr int kIq4nl = 20;   // GGML_TYPE_IQ4_NL

void ck(cudaError_t e, const char* what) {
    if (e != cudaSuccess) {
        std::fprintf(stderr, "rp_expert_bench: %s: %s\n", what, cudaGetErrorString(e));
        std::exit(1);
    }
}

std::vector<uint8_t> weights(int64_t rows, int64_t cols, int experts, std::mt19937& rng) {
    const size_t per = mmq::matrix_bytes(kIq4nl, rows, cols);
    std::vector<uint8_t> w((size_t) experts * per + 4096);
    std::uniform_int_distribution<int> byte(0, 255);
    for (auto& b : w) b = (uint8_t) byte(rng);
    const size_t nblk = (size_t) experts * per / 18;
    for (size_t i = 0; i < nblk; ++i) {
        const uint16_t d = 0x2000 + (uint16_t) byte(rng);   // f16 ~ 0.0078..0.016
        std::memcpy(w.data() + i * 18, &d, 2);
    }
    std::fill(w.end() - 4096, w.end(), 0);   // MMQ reads past the last matrix: zeroed, as the prompt path's MMQ_TAIL
    return w;
}

float median_ms(int n, const auto& fn) {
    cudaEvent_t e0, e1;
    cudaEventCreate(&e0);
    cudaEventCreate(&e1);
    std::vector<float> ts;
    for (int i = 0; i < n + 3; ++i) {
        cudaEventRecord(e0);
        fn();
        cudaEventRecord(e1);
        ck(cudaEventSynchronize(e1), "run");
        float ms;
        cudaEventElapsedTime(&ms, e0, e1);
        if (i >= 3) ts.push_back(ms);
    }
    cudaEventDestroy(e0);
    cudaEventDestroy(e1);
    std::sort(ts.begin(), ts.end());
    return ts[ts.size() / 2];
}

struct Case {
    const char* name;
    int64_t rows, cols;
};

bool run_case(const Case& c, int R, std::mt19937& rng) {
    constexpr int E = 16;
    const int T = E * R;
    const size_t per = mmq::matrix_bytes(kIq4nl, c.rows, c.cols), rper = rp::matrix_bytes(kIq4nl, c.rows, c.cols);
    std::vector<uint8_t> w = weights(c.rows, c.cols, E, rng);
    std::vector<float> x((size_t) T * c.cols);
    std::normal_distribution<float> nd(0.f, 1.f);
    std::uniform_real_distribution<float> mag(0.1f, 10.f);
    for (int t = 0; t < T; ++t) {
        const float m = mag(rng);
        for (int64_t k = 0; k < c.cols; ++k) x[(size_t) t * c.cols + k] = nd(rng) * m;
    }
    std::vector<int32_t> bounds(E + 1);
    for (int e = 0; e <= E; ++e) bounds[e] = e * R;

    uint8_t *dw, *drw;
    float *dx, *d_mmq, *d_rp;
    void* dq;
    int32_t *dids, *db;
    ck(cudaMalloc(&dw, w.size()), "w");
    ck(cudaMemcpy(dw, w.data(), w.size(), cudaMemcpyHostToDevice), "w");
    ck(cudaMalloc(&drw, (size_t) E * rper), "rw");
    ck(cudaMalloc(&dx, x.size() * 4), "x");
    ck(cudaMemcpy(dx, x.data(), x.size() * 4, cudaMemcpyHostToDevice), "x");
    ck(cudaMalloc(&dq, mmq::q8_bytes(T, c.cols)), "xq");
    ck(cudaMalloc(&d_mmq, (size_t) T * c.rows * 4), "dst");
    ck(cudaMalloc(&d_rp, (size_t) T * c.rows * 4), "dst");
    ck(cudaMalloc(&dids, (size_t) T * 4), "ids");
    ck(cudaMalloc(&db, (E + 1) * 4), "bounds");
    ck(cudaMemcpy(db, bounds.data(), (E + 1) * 4, cudaMemcpyHostToDevice), "bounds");
    mmq::iota(dids, T, nullptr);
    mmq::quantize(dx, nullptr, dq, kIq4nl, c.cols, c.cols, T, nullptr);
    const float rep_ms = median_ms(5, [&] { rp::repack_iq4nl(dw, per, drw, rper, c.rows, c.cols, E, nullptr); });

    mmq::Product p;
    p.w = dw; p.type = kIq4nl; p.w_rows = c.rows; p.w_cols = c.cols; p.expert_bytes = per; p.n = E;
    p.xq = dq; p.bounds = db; p.ids = dids; p.total_rows = T; p.max_rows = R; p.dst = d_mmq; p.ld_dst = c.rows;
    mmq::Product q = p;
    q.w = drw; q.expert_bytes = rper; q.dst = d_rp;
    mmq::Context mctx;
    rp::Context rctx;
    ck(cudaMemset(d_mmq, 0, (size_t) T * c.rows * 4), "zero");
    ck(cudaMemset(d_rp, 0, (size_t) T * c.rows * 4), "zero");
    mctx.run(p, nullptr);
    rctx.run(q, nullptr);
    ck(cudaDeviceSynchronize(), "first runs");
    std::vector<float> a((size_t) T * c.rows), b(a.size());
    ck(cudaMemcpy(a.data(), d_mmq, a.size() * 4, cudaMemcpyDeviceToHost), "out");
    ck(cudaMemcpy(b.data(), d_rp, b.size() * 4, cudaMemcpyDeviceToHost), "out");
    double amax = 0, dmax = 0;
    size_t bad = 0;
    for (size_t i = 0; i < a.size(); ++i) {
        amax = std::max(amax, (double) std::fabs(a[i]));
        const double d = std::fabs((double) a[i] - b[i]);
        if (!std::isfinite(b[i])) ++bad;
        dmax = std::max(dmax, d);
    }
    const double rel = amax > 0 ? dmax / amax : dmax;
    const bool ok = bad == 0 && rel < 1e-4;
    const float t_mmq = median_ms(25, [&] { mctx.run(p, nullptr); });
    const float t_rp = median_ms(25, [&] { rctx.run(q, nullptr); });
    std::printf("  R %4d  %-13s mmq %7.3f ms  rp %7.3f ms  (%+5.0f%%)  repack %6.3f ms  max|rp-mmq|/max|mmq| %.2e%s %s\n",
                R, c.name, t_mmq, t_rp, 100.0 * (t_mmq / t_rp - 1.0), rep_ms, rel, bad ? " (non-finite)" : "",
                ok ? "ok" : "FAIL");
    cudaFree(dw); cudaFree(drw); cudaFree(dx); cudaFree(dq); cudaFree(d_mmq); cudaFree(d_rp); cudaFree(dids);
    cudaFree(db);
    return ok;
}

// A group as the prompt path really has it: `n` experts (fewer than 16 in a layer's last group) with uneven rows,
// some none, starting at row `off` of a larger q8_1 array (gate/up: the whole chunk's rows), every written row
// compared.  Bitwise against MMQ.
bool check_ragged(const Case& c, int n, int off, int extra, int hi, std::mt19937& rng, rp::Context& rctx) {
    std::uniform_int_distribution<int> cnt(0, hi);
    std::vector<int32_t> bounds(n + 1);
    bounds[0] = off;
    int maxr = 0;
    for (int e = 0; e < n; ++e) {
        const int r = (e % 5 == 3) ? 0 : cnt(rng);
        bounds[e + 1] = bounds[e] + r;
        maxr = std::max(maxr, r);
    }
    const int T = bounds[n] + extra;
    const size_t per = mmq::matrix_bytes(kIq4nl, c.rows, c.cols), rper = rp::matrix_bytes(kIq4nl, c.rows, c.cols);
    std::vector<uint8_t> w = weights(c.rows, c.cols, n, rng);
    std::vector<float> x((size_t) T * c.cols);
    std::normal_distribution<float> nd(0.f, 1.f);
    for (auto& v : x) v = nd(rng) * 3.f;
    uint8_t *dw, *drw; float *dx, *a, *b; void* dq; int32_t *dids, *db;
    ck(cudaMalloc(&dw, w.size()), "w"); ck(cudaMemcpy(dw, w.data(), w.size(), cudaMemcpyHostToDevice), "w");
    ck(cudaMalloc(&drw, (size_t) n * rper), "rw");
    ck(cudaMalloc(&dx, x.size() * 4), "x"); ck(cudaMemcpy(dx, x.data(), x.size() * 4, cudaMemcpyHostToDevice), "x");
    ck(cudaMalloc(&dq, mmq::q8_bytes(T, c.cols)), "xq");
    ck(cudaMalloc(&a, (size_t) T * c.rows * 4), "a"); ck(cudaMalloc(&b, (size_t) T * c.rows * 4), "b");
    ck(cudaMemset(a, 0, (size_t) T * c.rows * 4), "a"); ck(cudaMemset(b, 0, (size_t) T * c.rows * 4), "b");
    ck(cudaMalloc(&dids, (size_t) T * 4), "ids"); ck(cudaMalloc(&db, (n + 1) * 4), "bounds");
    ck(cudaMemcpy(db, bounds.data(), (n + 1) * 4, cudaMemcpyHostToDevice), "bounds");
    mmq::iota(dids, T, nullptr);
    mmq::quantize(dx, nullptr, dq, kIq4nl, c.cols, c.cols, T, nullptr);
    rp::repack_iq4nl(dw, per, drw, rper, c.rows, c.cols, n, nullptr);
    mmq::Product p;
    p.w = dw; p.type = kIq4nl; p.w_rows = c.rows; p.w_cols = c.cols; p.expert_bytes = per; p.n = n;
    p.xq = dq; p.bounds = db; p.ids = dids; p.total_rows = T; p.max_rows = maxr; p.dst = a; p.ld_dst = c.rows;
    mmq::Product q = p;
    q.w = drw; q.expert_bytes = rper; q.dst = b;
    mmq::Context mctx;
    mctx.run(p, nullptr); rctx.run(q, nullptr);
    ck(cudaDeviceSynchronize(), "ragged");
    std::vector<float> ha((size_t) T * c.rows), hb(ha.size());
    ck(cudaMemcpy(ha.data(), a, ha.size() * 4, cudaMemcpyDeviceToHost), "a");
    ck(cudaMemcpy(hb.data(), b, hb.size() * 4, cudaMemcpyDeviceToHost), "b");
    size_t diff = 0, first = (size_t) -1;
    for (size_t i = (size_t) off * c.rows; i < (size_t) bounds[n] * c.rows; ++i)
        if (std::memcmp(&ha[i], &hb[i], 4) != 0) { ++diff; if (first == (size_t) -1) first = i; }
    std::printf("  ragged %-8s n %2d off %5d rows %5d max %3d: %zu of %zu values differ%s", c.name, n, off, bounds[n] - off,
                maxr, diff, (size_t) (bounds[n] - off) * c.rows, diff ? "" : " (bitwise)");
    if (diff) std::printf(" - first at row %zu col %zu: mmq %.6g rp %.6g", first / c.rows, first % c.rows, ha[first], hb[first]);
    std::printf("\n");
    for (void* v : {(void*) dw, (void*) drw, (void*) dx, dq, (void*) a, (void*) b, (void*) dids, (void*) db}) cudaFree(v);
    return diff == 0;
}

// The prompt path's group gather (rp::gather_group_iq4nl, from expert blobs [gate | up | down]) against the plain
// repack of the same GGUF matrices: the same bytes.
bool check_gather(std::mt19937& rng) {
    constexpr int E = 16;
    const int64_t N = 2560, FF = 640;
    const size_t gu = mmq::matrix_bytes(kIq4nl, 2 * FF, N), dn = mmq::matrix_bytes(kIq4nl, N, FF);
    const size_t rgu = rp::matrix_bytes(kIq4nl, 2 * FF, N), rdn = rp::matrix_bytes(kIq4nl, N, FF);
    std::vector<uint8_t> g = weights(2 * FF, N, E, rng), d = weights(N, FF, E, rng);
    std::vector<uint8_t> blobs((size_t) E * (gu + dn));
    for (int e = 0; e < E; ++e) {
        std::memcpy(blobs.data() + (size_t) e * (gu + dn), g.data() + (size_t) e * gu, gu);
        std::memcpy(blobs.data() + (size_t) e * (gu + dn) + gu, d.data() + (size_t) e * dn, dn);
    }
    uint8_t *db, *dg, *dd, *a_gu, *a_d, *b_gu, *b_d;
    ck(cudaMalloc(&db, blobs.size()), "b"); ck(cudaMemcpy(db, blobs.data(), blobs.size(), cudaMemcpyHostToDevice), "b");
    ck(cudaMalloc(&dg, g.size()), "g"); ck(cudaMemcpy(dg, g.data(), g.size(), cudaMemcpyHostToDevice), "g");
    ck(cudaMalloc(&dd, d.size()), "d"); ck(cudaMemcpy(dd, d.data(), d.size(), cudaMemcpyHostToDevice), "d");
    for (uint8_t** p : {&a_gu, &b_gu}) { ck(cudaMalloc(p, (size_t) E * rgu), "o"); ck(cudaMemset(*p, 0xAB, (size_t) E * rgu), "o"); }
    for (uint8_t** p : {&a_d, &b_d}) { ck(cudaMalloc(p, (size_t) E * rdn), "o"); ck(cudaMemset(*p, 0xAB, (size_t) E * rdn), "o"); }
    rp::repack_iq4nl(dg, gu, a_gu, rgu, 2 * FF, N, E, nullptr);
    rp::repack_iq4nl(dd, dn, a_d, rdn, N, FF, E, nullptr);
    mmq::GatherGroup gg;
    for (int e = 0; e < E; ++e) gg.blob[e] = db + (size_t) e * (gu + dn);
    gg.first = 0; gg.n = E;
    rp::gather_group_iq4nl(gg, gu / 2, gu, N, FF, b_gu, rgu, b_d, rdn, nullptr);
    ck(cudaDeviceSynchronize(), "gather");
    std::vector<uint8_t> x((size_t) E * rgu), y(x.size()), u((size_t) E * rdn), v(u.size());
    ck(cudaMemcpy(x.data(), a_gu, x.size(), cudaMemcpyDeviceToHost), "x"); ck(cudaMemcpy(y.data(), b_gu, y.size(), cudaMemcpyDeviceToHost), "y");
    ck(cudaMemcpy(u.data(), a_d, u.size(), cudaMemcpyDeviceToHost), "u"); ck(cudaMemcpy(v.data(), b_d, v.size(), cudaMemcpyDeviceToHost), "v");
    const bool ok = x == y && u == v;
    std::printf("  group gather vs repack of the same matrices: %s\n", ok ? "same bytes" : "MISMATCH");
    for (uint8_t* p : {db, dg, dd, a_gu, a_d, b_gu, b_d}) cudaFree(p);
    return ok;
}

}  // namespace

int main(int argc, char** argv) {
    if (!rp::built()) {
        std::printf("rp_expert_bench: this build has no repacked GEMM (gfx906 only)\n");
        return 0;
    }
    std::vector<int> Rs;
    for (int i = 1; i < argc; ++i) Rs.push_back(std::atoi(argv[i]));
    if (Rs.empty()) Rs = {12, 80, 160, 626};
    std::printf("rp_expert_bench: IQ4_NL, 16 experts a launch, mmq = llama.cpp MMQ, rp = mx repacked GEMM, medians of 25"
                " (%% = rp speedup)\n");
    const Case cases[] = {{"gate/up", 1280, 2560}, {"down", 2560, 640}};
    std::mt19937 rng(7);
    int fails = check_gather(rng) ? 0 : 1;
    for (const Case& c : {Case{"gate/up", 1280, 2560}, Case{"down", 2560, 640}})
        for (int hi : {220, 900}) {   // 900: rows past 256 take the 64-wide tile
            rp::Context rctx;
            for (int n : {16, 7, 1}) fails += check_ragged(c, n, c.cols == 2560 ? 1000 : 0, 37, hi, rng, rctx) ? 0 : 1;
        }
    {   // one context, the prompt path's order: a whole chunk's gate/up rows, then a group's smaller down rows, then a
        // larger one, then gate/up again - the routing scratch must not carry one call's tables into the next
        rp::Context shared;
        const Case gu{"gate/up", 1280, 2560}, dn{"down", 2560, 640};
        fails += check_ragged(gu, 16, 3000, 37, 300, rng, shared) ? 0 : 1;
        fails += check_ragged(dn, 9, 0, 0, 150, rng, shared) ? 0 : 1;
        fails += check_ragged(dn, 16, 0, 0, 400, rng, shared) ? 0 : 1;
        fails += check_ragged(gu, 16, 5000, 11, 500, rng, shared) ? 0 : 1;
    }
    for (int R : Rs)
        for (const Case& c : cases) fails += run_case(c, R, rng) ? 0 : 1;
    std::printf("rp_expert_bench: %d failures\n", fails);
    return fails ? 1 : 0;
}
