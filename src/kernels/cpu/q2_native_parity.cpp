// src/kernels/cpu/q2_native_parity.cpp - a native pack's Q2_0 rows (ggml type 42) on the path the pool takes for
// them, q2_native_kernels() -> act_quant_any + q2_rows_any (AVX-512, AVX2 or AVX1), against ggml-cpu's own Q2_0
// vec_dot, both measured against the exact dot of the dequantized weights with the float activations.
//
// Strata's activation quantizer and ggml's q8_0 round differently, so neither is bitwise the other: the check is
// that Strata's rows are as accurate as ggml's (max error over all rows and tokens, relative to the largest exact
// value, within 2x ggml's plus a small floor).  Gate/up shape (n_ff x n_embd) and down shape (n_embd x n_ff), one
// and four tokens, and the rows cut into ranges as the pool's workers cut them (each range must be bitwise the rows
// of the whole call).  The case that motivated it: before q2_native_kernels() admitted the AVX1 rung, a Sandy / Ivy
// Bridge CPU never ran q2_rows_any inside the pool, so nothing checked it there against ggml.
// Synthetic data, CPU only; skipped (pass) on a CPU where q2_native_kernels(42) is false.
#include "strata/kernels/cpu/expert.hpp"
#include "strata/kernels/cpu/expert_layout.hpp"

#include "ggml.h"
#include "ggml-cpu.h"

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <random>
#include <vector>

namespace c = strata::kernels::cpu;

namespace {

constexpr int kQ2 = 42;   // GGML_TYPE_Q2_0 in Strata's ggml

struct Result {
    double rel_strata = 0, rel_ggml = 0;
    bool ranges_exact = true;
};

Result check(int rows, int k, int nt, std::mt19937& rng) {
    const ggml_type t = (ggml_type) kQ2;
    const size_t rb = ggml_row_size(t, k);
    std::normal_distribution<float> nd(0.f, 0.02f);
    std::vector<float> row(k), deq((size_t) rows * k);
    std::vector<uint8_t> w(rb * rows);
    for (int r = 0; r < rows; ++r) {
        for (float& v : row) v = nd(rng);
        ggml_quantize_chunk(t, row.data(), w.data() + rb * r, 0, 1, k, nullptr);
        ggml_get_type_traits(t)->to_float(w.data() + rb * r, deq.data() + (size_t) r * k, k);
    }
    std::normal_distribution<float> xd(0.f, 1.f);
    std::vector<std::vector<float>> x(nt, std::vector<float>(k));
    for (auto& v : x)
        for (float& e : v) e = xd(rng);

    // exact
    std::vector<double> exact((size_t) nt * rows);
    double amax = 0;
    for (int tk = 0; tk < nt; ++tk)
        for (int r = 0; r < rows; ++r) {
            double s = 0;
            for (int i = 0; i < k; ++i) s += (double) deq[(size_t) r * k + i] * x[tk][i];
            exact[(size_t) tk * rows + r] = s;
            amax = std::max(amax, std::fabs(s));
        }

    // ggml-cpu: its vec_dot with the activation in its vec_dot_type
    const ggml_type_traits_cpu* tc = ggml_get_type_traits_cpu(t);
    const ggml_type vt = tc->vec_dot_type;
    std::vector<uint8_t> xq(ggml_row_size(vt, k));
    double eg = 0;
    for (int tk = 0; tk < nt; ++tk) {
        ggml_get_type_traits_cpu(vt)->from_float(x[tk].data(), xq.data(), k);
        for (int r = 0; r < rows; ++r) {
            float y = 0;
            tc->vec_dot(k, &y, 0, w.data() + rb * r, 0, xq.data(), 0, 1);
            eg = std::max(eg, std::fabs(y - exact[(size_t) tk * rows + r]));
        }
    }

    // Strata: the pool's path
    std::vector<c::ActQ> a(nt);
    const c::ActQ* ap[8];
    std::vector<float> out((size_t) nt * rows), cut((size_t) nt * rows);
    float* op[8];
    float* cp[8];
    for (int tk = 0; tk < nt; ++tk) {
        c::act_quant_any(x[tk].data(), k, a[tk]);
        ap[tk] = &a[tk];
        op[tk] = out.data() + (size_t) tk * rows;
        cp[tk] = cut.data() + (size_t) tk * rows;
    }
    c::q2_rows_any(w.data(), rb, k / 64, ap, nt, op, 0, rows);
    double es = 0;
    for (size_t i = 0; i < out.size(); ++i) es = std::max(es, std::fabs(out[i] - exact[i]));
    // the same rows cut into ragged ranges, as the pool's tasks cut them
    for (int r0 = 0; r0 < rows;) {
        const int r1 = std::min(rows, r0 + 1 + (int) (rng() % 97));
        c::q2_rows_any(w.data(), rb, k / 64, ap, nt, cp, r0, r1);
        r0 = r1;
    }
    Result res;
    res.rel_strata = es / amax;
    res.rel_ggml = eg / amax;
    res.ranges_exact = std::memcmp(out.data(), cut.data(), out.size() * sizeof(float)) == 0;
    return res;
}

}  // namespace

int main() {
    if (!c::q2_native_kernels(kQ2)) {
        std::printf("q2_native_parity: this CPU runs Q2_0 on ggml-cpu (q2_native_kernels false), skipped\n");
        return 0;
    }
    ggml_cpu_init();
    std::printf("q2_native_parity: q2_rows_any on %s\n",
                c::cpu_avx512_ok() ? "AVX-512" : c::cpu_avx2_ok() ? "AVX2" : "AVX1");
    std::mt19937 rng(20261030);
    int failures = 0;
    struct Shape { const char* name; int rows, k; };
    const Shape shapes[] = {{"gate/up", c::FF, c::H}, {"down", c::H, c::FF}};
    for (const Shape& s : shapes)
        for (int nt : {1, 4}) {
            const Result r = check(s.rows, s.k, nt, rng);
            const bool ok = r.rel_strata <= 2 * r.rel_ggml + 1e-4 && r.rel_strata < 2e-2 && r.ranges_exact;
            failures += !ok;
            std::printf("  %-7s %4d x %4d, %d token(s): Strata rel %.2e, ggml rel %.2e, ranges %s  %s\n", s.name, s.rows,
                        s.k, nt, r.rel_strata, r.rel_ggml, r.ranges_exact ? "bitwise" : "DIFFER", ok ? "ok" : "FAIL");
        }
    std::printf("q2_native_parity: %d failure(s)\n", failures);
    return failures != 0;
}
