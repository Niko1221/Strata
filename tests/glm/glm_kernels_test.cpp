// tests/glm/glm_kernels_test.cpp - the GLM engine's CPU kernels against their scalar definitions.  No model, no GPU.
//
// What it pins down: the int4-g64 nibble order (element 2j low, 2j+1 high), the +8 code bias, signed int8 rows, the
// permuted-activation kernel for one and for several rows (and a row sub-range written at an offset), the
// transposed product MLA absorption uses, the int8 row kernels, and the interleaved RoPE's pairing.
#include "strata/glm/kernels.hpp"

#include <cmath>
#include <cstdio>
#include <random>
#include <vector>

using namespace strata::glm;

static int g_fail = 0;
static void check(bool ok, const char* what) {
    std::printf("  %-72s %s\n", what, ok ? "ok" : "FAIL");
    if (!ok) ++g_fail;
}

static float max_rel(const std::vector<float>& a, const std::vector<float>& b) {
    float m = 0.0f, ref = 1e-6f;
    for (size_t i = 0; i < a.size(); ++i) ref = std::max(ref, std::fabs(b[i]));
    for (size_t i = 0; i < a.size(); ++i) m = std::max(m, std::fabs(a[i] - b[i]) / ref);
    return m;
}

int main() {
    std::mt19937 rng(7);
    std::uniform_int_distribution<int> byte(0, 255);
    std::uniform_real_distribution<float> uni(-1.0f, 1.0f);
    const int O = 37, I = 512;
    std::vector<uint8_t> codes((size_t) O * I / 2);
    std::vector<float> scales((size_t) O * I / 64);
    for (auto& c : codes) c = (uint8_t) byte(rng);
    for (auto& s : scales) s = 0.01f + 0.1f * std::fabs(uni(rng));
    const Q4 W{O, I, codes.data(), scales.data()};

    // the format itself: byte 0 of row 0 holds element 0 (low nibble) and element 1 (high nibble), minus 8
    check(q4_weight(W, 0, 0) == (float) ((codes[0] & 15) - 8) * scales[0] &&
              q4_weight(W, 0, 1) == (float) ((codes[0] >> 4) - 8) * scales[0],
          "element 2j is the low nibble, 2j+1 the high one, both code - 8");
    check(q4_weight(W, 1, 64) == (float) ((codes[(size_t) I / 2 + 32] & 15) - 8) * scales[I / 64 + 1],
          "row stride I/2 bytes, one scale per group of 64");

    Pool pool(4);
    for (int S : {1, 5, 19}) {
        std::vector<float> x((size_t) S * I);
        for (auto& v : x) v = uni(rng);
        std::vector<float> y((size_t) S * O), r((size_t) S * O);
        q4_gemm(pool, W, x.data(), S, y.data());
        q4_rows_ref(W, x.data(), S, 0, O, r.data(), O);
        char what[96];
        std::snprintf(what, sizeof what, "q4_gemm matches the scalar definition, S = %d (rel %.1e)", S, max_rel(y, r));
        check(max_rel(y, r) < 2e-5f, what);
    }
    {   // a row sub-range written at an offset (the value half of kv_b)
        std::vector<float> x(I);
        for (auto& v : x) v = uni(rng);
        Act a;
        a.prepare(x.data(), 1, I);
        std::vector<float> y(10), r((size_t) O);
        q4_rows(W, a, 20, 30, y.data(), 10, 20);
        q4_rows_ref(W, x.data(), 1, 0, O, r.data(), O);
        bool ok = true;
        for (int i = 0; i < 10; ++i) ok = ok && std::fabs(y[i] - r[20 + i]) <= 2e-5f * (std::fabs(r[20 + i]) + 1.0f);
        check(ok, "q4_rows writes rows [20, 30) to y[0..10) with ybase = 20");
    }
    {   // transposed product: out[i] += sum_k coef[k] W[r0 + k, i]
        const int r0 = 3, n = 11;
        std::vector<float> coef(n), out(I, 0.5f), ref(I, 0.5f);
        for (auto& v : coef) v = uni(rng);
        q4_rows_t(W, r0, n, coef.data(), out.data());
        for (int i = 0; i < I; ++i) {
            double acc = 0.0;
            for (int k = 0; k < n; ++k) acc += (double) coef[k] * q4_weight(W, r0 + k, i);
            ref[i] += (float) acc;
        }
        char what[96];
        std::snprintf(what, sizeof what, "q4_rows_t is W^T coef over a row range, accumulated (rel %.1e)", max_rel(out, ref));
        check(max_rel(out, ref) < 2e-5f, what);
    }
    {   // int8 rows: signed bytes, one scale per row
        const int O8 = 300, I8 = 6144 / 8;
        std::vector<uint8_t> c8((size_t) O8 * I8);
        std::vector<float> s8(O8), x(I8), y(O8);
        for (auto& c : c8) c = (uint8_t) byte(rng);
        for (auto& s : s8) s = 0.001f + 0.01f * std::fabs(uni(rng));
        for (auto& v : x) v = uni(rng);
        const Q8R W8{O8, I8, c8.data(), s8.data()};
        q8r_gemv(pool, W8, x.data(), y.data());
        std::vector<float> ref(O8);
        for (int r = 0; r < O8; ++r) {
            double acc = 0.0;
            for (int i = 0; i < I8; ++i) acc += (double) (int8_t) c8[(size_t) r * I8 + i] * s8[r] * x[i];
            ref[r] = (float) acc;
        }
        check(max_rel(y, ref) < 2e-5f, "q8r_gemv: (int8_t) code * row scale");
        std::vector<float> row(I8);
        q8r_row(W8, 7, row.data());
        check(row[5] == (float) (int8_t) c8[(size_t) 7 * I8 + 5] * s8[7] && (int8_t) (uint8_t) 0x80 == -128,
              "q8r_row is the dequantized embedding row (0x80 is -128, 0x00 is 0)");
    }
    {   // RoPE: pairs (2j, 2j+1), written to (j, j + n/2); position 0 is the identity up to the reorder
        Rope rope;
        rope.init(64, 8000000.0);
        std::vector<float> v(64), in(64);
        for (int i = 0; i < 64; ++i) in[i] = v[i] = (float) i;
        rope.apply(v.data(), 0);
        bool ok = true;
        for (int j = 0; j < 32; ++j) ok = ok && v[j] == in[2 * j] && v[32 + j] == in[2 * j + 1];
        check(ok, "rope at position 0 de-interleaves: v[j] = x[2j], v[32 + j] = x[2j + 1]");
        for (int i = 0; i < 64; ++i) v[i] = in[i];
        rope.apply(v.data(), 1000);
        const float ang = 1000.0f * std::pow(8000000.0f, -2.0f * 3 / 64.0f);
        ok = std::fabs(v[3] - (in[6] * std::cos(ang) - in[7] * std::sin(ang))) < 1e-4f &&
             std::fabs(v[35] - (in[7] * std::cos(ang) + in[6] * std::sin(ang))) < 1e-4f;
        check(ok, "rope rotates pair j = 3 by pos * theta^(-2j/64)");
        // the dot product of two roped vectors depends only on the position difference
        std::vector<float> a(64), b(64), a2(64), b2(64);
        for (int i = 0; i < 64; ++i) { a[i] = a2[i] = uni(rng); b[i] = b2[i] = uni(rng); }
        rope.apply(a.data(), 10);
        rope.apply(b.data(), 3);
        rope.apply(a2.data(), 110);
        rope.apply(b2.data(), 103);
        check(std::fabs(dot_f32(a.data(), b.data(), 64) - dot_f32(a2.data(), b2.data(), 64)) < 1e-3f,
              "q.k after rope depends only on the position difference");
    }
    {   // prompt MLA, 4 queries at a time, against the per-query definition (S = 7: a block of 4 and a short one)
        const int kvl = 64, nr = 16, nope = 24, vh = 32, S = 7, pos0 = 5, row = kvl + nr, qstride = nope + nr + 3;
        std::vector<float> wk((size_t) nope * kvl), wv((size_t) vh * kvl), kv((size_t) (pos0 + S) * row),
            q((size_t) S * qstride), got((size_t) S * vh), want((size_t) S * vh);
        for (auto* v : {&wk, &wv, &kv, &q})
            for (auto& x : *v) x = uni(rng);
        const float scale = 0.125f;
        mla_head_prompt(wk.data(), wv.data(), q.data(), qstride, kv.data(), kvl, nr, nope, vh, S, pos0, scale, got.data(), vh);
        for (int s = 0; s < S; ++s) {
            const float* qs = &q[(size_t) s * qstride];
            const int nt = pos0 + s + 1;
            std::vector<double> qabs(kvl, 0.0), sc(nt), clat(kvl, 0.0);
            for (int d = 0; d < nope; ++d)
                for (int i = 0; i < kvl; ++i) qabs[i] += (double) qs[d] * wk[(size_t) d * kvl + i];
            double mx = -1e300, sum = 0.0;
            for (int t = 0; t < nt; ++t) {
                double a = 0.0;
                for (int i = 0; i < kvl; ++i) a += qabs[i] * kv[(size_t) t * row + i];
                for (int i = 0; i < nr; ++i) a += (double) qs[nope + i] * kv[(size_t) t * row + kvl + i];
                sc[t] = a * scale;
                mx = std::max(mx, sc[t]);
            }
            for (int t = 0; t < nt; ++t) { sc[t] = std::exp(sc[t] - mx); sum += sc[t]; }
            for (int t = 0; t < nt; ++t)
                for (int i = 0; i < kvl; ++i) clat[i] += sc[t] / sum * kv[(size_t) t * row + i];
            for (int r = 0; r < vh; ++r) {
                double a = 0.0;
                for (int i = 0; i < kvl; ++i) a += (double) wv[(size_t) r * kvl + i] * clat[i];
                want[(size_t) s * vh + r] = (float) a;
            }
        }
        char what[96];
        std::snprintf(what, sizeof what, "mla_head_prompt is causal absorbed attention, S = 7 (rel %.1e)", max_rel(got, want));
        check(max_rel(got, want) < 1e-5f, what);
        std::vector<float> d(W.I);   // a whole row
        q4_dequant_rows(W, 3, 1, d.data());
        bool ok = true;
        for (int i = 0; i < 64; ++i) ok = ok && d[i] == q4_weight(W, 3, i);
        check(ok, "q4_dequant_rows gives the format's values in natural order");
    }
    {   // the pool covers every index exactly once
        std::vector<int> hit(10007, 0);
        pool.parallel_for(10007, 13, [&](int64_t b, int64_t e) { for (int64_t i = b; i < e; ++i) ++hit[i]; });
        bool ok = true;
        for (int h : hit) ok = ok && h == 1;
        check(ok, "Pool::parallel_for runs every index once");
    }
    std::printf(g_fail ? "glm_kernels_test: %d FAILED\n" : "glm_kernels_test: all passed\n", g_fail);
    return g_fail ? 1 : 0;
}
