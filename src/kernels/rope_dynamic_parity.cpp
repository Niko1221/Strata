// src/kernels/rope_dynamic_parity.cpp - --rope-scaling dynamic: cached keys turned to a new factor against the same
// keys written at that factor (GPU, synthetic, no model).
//
// K is cached POST-RoPE, so a factor change turns every stored key by p * (w_new - w_old) and mscale_new/mscale_old
// (kernels/rope_dynamic.hpp).  A key written at factor A and turned to B has to be the key the engine would have
// written at B.  Random pre-rope keys at positions 0 .. 519,999 (the real page size, pages spread over 2x the trained
// context) are rotated on the host by A and by B as the table path does (float cos/sin from build_rope_table's
// float64 expressions, float products) and written by the engine's own kernels: kv_append_step (fp16),
// kv_append_q8_step (int8, one fp16 scale per 64), and the int8 rows through fwht256 first (STRATA_KV_ROT=1).  A's
// pool is turned on the GPU and compared with B's pool and with the float64 key at B:
//   1. the error of the turned key is at most 1.6x the error of the key written at B (RMS over the rotary dims; with
//      the Hadamard over the whole row): a turn re-codes once more, so its error is the written key's plus A's carried
//      over, sqrt(2) = 1.41x for equal steps;
//   2. its largest error per element is at most 2 quantization steps of B's group (int8; with the Hadamard 3, since H
//      spreads A's error over the row) or 3 fp16 ulps of the row's largest value (fp16);
//   3. without the Hadamard, dims 64-255 (codes and their scales) are bitwise what was written: only the rope group
//      is re-coded;
//   4. the indexer's pooled keys: the turned float rows within 8 float ulps of the row's largest value of the float64
//      key at B; the spare row and `dead` (position 0) are exactly mscale_B/mscale_A times their old values;
//   5. build_rope_table_mt is build_rope_table bit for bit (none and yarn, 1/7/16 threads): a dialog that goes back
//      below the trained context gets the startup table, so it runs the startup bits.
// The factor pairs: none -> yarn 2, yarn 1.5 -> 2, yarn 2 -> none, none -> yarn 1.125.
#include "strata/kernels/f16_bits.hpp"
#include "strata/kernels/kv_q4.hpp"
#include "strata/kernels/kv_q8.hpp"
#include "strata/kernels/qsa.hpp"
#include "strata/kernels/rope.hpp"
#include "strata/kernels/rope_dynamic.hpp"
#include "strata/kernels/rope_scaling.hpp"

#include <cuda_runtime.h>

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <random>
#include <vector>

namespace k = strata::kernels;

namespace {
int g_fail = 0;
void ck(cudaError_t e, const char* w) {
    if (e != cudaSuccess) { std::fprintf(stderr, "%s: %s\n", w, cudaGetErrorString(e)); std::exit(2); }
}
template <typename T> T* dalloc(size_t n) {
    T* p = nullptr;
    ck(cudaMalloc(&p, n * sizeof(T) + 64), "malloc");
    ck(cudaMemset(p, 0, n * sizeof(T) + 64), "memset");
    return p;
}
void fail(const char* what) { std::fprintf(stderr, "FAIL: %s\n", what); ++g_fail; }

// the engine's yarn for the trained 262,144 (ggml's defaults: ext_factor 1, attn_factor 1, beta 32/1); f = 1: none
k::RopeScaling scaling(double f) {
    k::RopeScaling s;
    s.freq_base = 1e7;
    if (f <= 1.0) return s;
    s.type = k::RopeScalingType::YaRN;
    s.factor = f;
    s.orig_ctx = 262144;
    s.ext_factor = 1.0;
    return s;
}

// build_rope_table's float64 angle and magnitude of pair i at position p (rope.cu, transcribed)
void angle(const k::RopeScaling& sc, int64_t p, int i, double& ang, double& ms) {
    const double inv = std::pow(sc.freq_base, -2.0 * (double) i / 64.0);
    ms = 1.0;
    ang = (double) p * inv;
    if (sc.type == k::RopeScalingType::None) return;
    const double fs = sc.freq_scale();
    ms = sc.mscale();
    double cd[2];
    sc.corr_dims(64, cd);
    const double extrap = (double) p * inv, interp = fs * extrap;
    ang = interp;
    if (sc.ext_factor != 0) {
        const double ramp = (double) k::rope_yarn_ramp((float) cd[0], (float) cd[1], i) * sc.ext_factor;
        ang = interp * (1.0 - ramp) + extrap * ramp;
    }
}

// a row of `dim` floats rotated at p: NEOX pairs (i, i + 32) of dims 0..63.  `exact`: float64 throughout; else the
// table path's float cos/sin and float products (what the engine writes)
void rotate(const k::RopeScaling& sc, int64_t p, const float* x, int dim, bool exact, double* out) {
    for (int d = 0; d < dim; ++d) out[d] = x[d];
    for (int i = 0; i < 32; ++i) {
        double ang, ms;
        angle(sc, p, i, ang, ms);
        if (exact) {
            const double c = std::cos(ang) * ms, s = std::sin(ang) * ms;
            out[i] = (double) x[i] * c - (double) x[i + 32] * s;
            out[i + 32] = (double) x[i] * s + (double) x[i + 32] * c;
        } else {
            const float c = (float) (std::cos(ang) * ms), s = (float) (std::sin(ang) * ms);
            out[i] = (double) (x[i] * c - x[i + 32] * s);
            out[i + 32] = (double) (x[i] * s + x[i + 32] * c);
        }
    }
}

// fwht256 with scale 1/16, natural order: its own inverse
void hadamard(double* v) {
    for (int h = 1; h < 256; h *= 2)
        for (int i = 0; i < 256; i += 2 * h)
            for (int j = i; j < i + h; ++j) {
                const double a = v[j], b = v[j + h];
                v[j] = a + b;
                v[j + h] = a - b;
            }
    for (int i = 0; i < 256; ++i) v[i] *= 0.0625;
}

struct Stats {
    double sq_turn = 0, sq_built = 0, worst_turn = 0, worst_built = 0;
    long n = 0;
};

}  // namespace

int main() {
    k::QsaShapes s = k::qsa_real_shapes();
    const int H = (int) s.n_head_kv, D = (int) s.head_dim, P = (int) s.page_size, G = D / k::KV_Q8_GROUP;
    const int64_t positions = 520000, lpages = positions / P;
    std::mt19937 rng(23);
    // 48 logical pages held, the first and last of the context and both sides of 262,144 among them
    std::vector<int64_t> held = {0, 1, 262144 / P - 1, 262144 / P, lpages - 1};
    while (held.size() < 48) {
        const int64_t lp = (int64_t) (rng() % (uint32_t) lpages);
        if (std::find(held.begin(), held.end(), lp) == held.end()) held.push_back(lp);
    }
    const int pages = (int) held.size();
    std::vector<int32_t> table((size_t) lpages, -1);
    for (int i = 0; i < pages; ++i) table[(size_t) held[(size_t) i]] = i;
    int32_t* d_table = dalloc<int32_t>((size_t) lpages);
    ck(cudaMemcpy(d_table, table.data(), table.size() * 4, cudaMemcpyHostToDevice), "table");
    std::vector<int64_t> cell_pos;   // physical cell c -> its position
    for (int i = 0; i < pages; ++i)
        for (int j = 0; j < P; ++j) cell_pos.push_back(held[(size_t) i] * P + j);
    // pre-rope keys: unit normal, some rows small or large, some rope groups zero
    std::normal_distribution<float> nd(0.f, 1.f);
    std::vector<std::vector<float>> raw(cell_pos.size(), std::vector<float>((size_t) H * D));
    for (size_t c = 0; c < raw.size(); ++c) {
        const float sc = c % 7 == 0 ? 1e-3f : (c % 11 == 0 ? 40.f : 1.f);
        for (float& x : raw[c]) x = nd(rng) * sc;
        if (c % 13 == 0) std::fill(raw[c].begin(), raw[c].begin() + 64, 0.f);
    }
    float *kcur = dalloc<float>((size_t) H * D), *vcur = dalloc<float>((size_t) H * D);
    int32_t* step = dalloc<int32_t>(k::kStepCount);
    const size_t cells = (size_t) pages * P;
    int8_t *kq = dalloc<int8_t>(cells * H * D), *vq = dalloc<int8_t>(cells * H * D);
    uint16_t *ks = dalloc<uint16_t>(cells * H * G), *vs = dalloc<uint16_t>(cells * H * G);
    uint16_t *kp = dalloc<uint16_t>(cells * H * D), *vp = dalloc<uint16_t>(cells * H * D);

    enum Kind { F16, I8, I8H };
    // the keys of every held cell at factor f written by the engine's kernels into the pool of `kind`
    auto write = [&](Kind kind, double f) {
        const k::RopeScaling sc = scaling(f);
        std::vector<double> rot((size_t) D);
        std::vector<float> row((size_t) H * D);
        for (size_t c = 0; c < cell_pos.size(); ++c) {
            for (int h = 0; h < H; ++h) {
                rotate(sc, cell_pos[c], raw[c].data() + (size_t) h * D, D, false, rot.data());
                for (int d = 0; d < D; ++d) row[(size_t) h * D + d] = (float) rot[(size_t) d];
            }
            const int32_t hstep[k::kStepCount] = {(int32_t) cell_pos[c], (int32_t) cell_pos[c] + 1, 0, 0};
            ck(cudaMemcpy(step, hstep, sizeof hstep, cudaMemcpyHostToDevice), "step");
            ck(cudaMemcpy(kcur, row.data(), row.size() * 4, cudaMemcpyHostToDevice), "k");
            ck(cudaMemcpy(vcur, row.data(), row.size() * 4, cudaMemcpyHostToDevice), "v");
            if (kind == I8H) k::fwht256_inplace_cuda(kcur, H, nullptr);
            if (kind == F16) k::kv_append_step(kp, vp, d_table, step, kcur, vcur, s, nullptr);
            else k::kv_append_q8_step(kq, vq, ks, vs, d_table, step, kcur, vcur, s, nullptr);
            ck(cudaDeviceSynchronize(), "append");
        }
    };
    struct Pool { std::vector<int8_t> q; std::vector<uint16_t> sc, f16; };
    auto read = [&](Kind kind) {
        Pool p;
        if (kind == F16) {
            p.f16.resize(cells * H * D);
            ck(cudaMemcpy(p.f16.data(), kp, p.f16.size() * 2, cudaMemcpyDeviceToHost), "d2h");
        } else {
            p.q.resize(cells * H * D);
            p.sc.resize(cells * H * G);
            ck(cudaMemcpy(p.q.data(), kq, p.q.size(), cudaMemcpyDeviceToHost), "d2h");
            ck(cudaMemcpy(p.sc.data(), ks, p.sc.size() * 2, cudaMemcpyDeviceToHost), "d2h");
        }
        return p;
    };
    // stored row (physical cell c, head h) -> values in the rotated domain (the Hadamard undone)
    auto value = [&](Kind kind, const Pool& p, size_t c, int h, double* out) {
        const size_t row = ((c / P) * H + h) * P + c % P;
        for (int d = 0; d < D; ++d)
            out[d] = kind == F16 ? (double) k::f32_from_f16(p.f16[row * D + d])
                                 : (double) p.q[row * D + d] * (double) k::f32_from_f16(p.sc[row * G + d / 64]);
        if (kind == I8H) hadamard(out);
    };

    const double pairs[][2] = {{1.0, 2.0}, {1.5, 2.0}, {2.0, 1.0}, {1.0, 1.125}};
    const char* names[] = {"fp16", "int8", "int8+H"};
    for (const auto& fp : pairs) {
        const double fa = fp[0], fb = fp[1];
        const k::RopeRates ra = k::rope_rates(64, scaling(fa)), rb = k::rope_rates(64, scaling(fb));
        for (Kind kind : {F16, I8, I8H}) {
            write(kind, fb);
            const Pool built = read(kind);
            write(kind, fa);
            const Pool before = read(kind);
            k::RopeKPool kpool;
            if (kind == F16) kpool.k_pool = kp;
            else { kpool.k_q = kq; kpool.k_scale = ks; }
            kpool.page_table = d_table;
            kpool.n_kv_heads = H;
            kpool.page_size = P;
            kpool.head_dim = D;
            kpool.hadamard = kind == I8H;
            k::rope_rerotate_k(kpool, positions, ra, rb, nullptr);
            ck(cudaDeviceSynchronize(), "rerotate");
            const Pool turned = read(kind);
            Stats st;
            std::vector<double> want((size_t) D), vt((size_t) D), vb((size_t) D);
            const int dims = kind == I8H ? D : 64;
            long untouched_bad = 0;
            for (size_t c = 0; c < cells; ++c)
                for (int h = 0; h < H; ++h) {
                    rotate(scaling(fb), cell_pos[c], raw[c].data() + (size_t) h * D, D, true, want.data());
                    value(kind, turned, c, h, vt.data());
                    value(kind, built, c, h, vb.data());
                    double rmax = 0;
                    for (int d = 0; d < D; ++d) rmax = std::max(rmax, std::fabs(want[(size_t) d]));
                    const size_t row = ((c / P) * H + h) * P + c % P;
                    for (int d = 0; d < dims; ++d) {
                        // the unit: B's quantization step of this group (int8; with H the row's largest step), or
                        // an fp16 ulp of the row's largest value
                        double unit;
                        if (kind == F16) unit = rmax > 0 ? std::ldexp(1.0, std::ilogb(rmax) - 10) : 1.0;
                        else if (kind == I8) unit = k::f32_from_f16(built.sc[row * G + d / 64]);
                        else {
                            unit = 0;
                            for (int g = 0; g < G; ++g) unit = std::max(unit, (double) k::f32_from_f16(built.sc[row * G + g]));
                        }
                        if (unit <= 0) unit = 1e-30;
                        const double et = vt[(size_t) d] - want[(size_t) d], eb = vb[(size_t) d] - want[(size_t) d];
                        st.sq_turn += et * et;
                        st.sq_built += eb * eb;
                        st.worst_turn = std::max(st.worst_turn, std::fabs(et) / unit);
                        st.worst_built = std::max(st.worst_built, std::fabs(eb) / unit);
                        ++st.n;
                    }
                    if (kind != I8H)   // only the rope group is re-coded
                        for (int d = 64; d < D; ++d) {
                            if (kind == F16 && turned.f16[row * D + d] != before.f16[row * D + d]) ++untouched_bad;
                            if (kind == I8 && turned.q[row * D + d] != before.q[row * D + d]) ++untouched_bad;
                        }
                    if (kind == I8)
                        for (int g = 1; g < G; ++g)
                            if (turned.sc[row * G + g] != before.sc[row * G + g]) ++untouched_bad;
                }
            const double rt = std::sqrt(st.sq_turn / (double) st.n), rbl = std::sqrt(st.sq_built / (double) st.n);
            const double ratio = rbl > 0 ? rt / rbl : 0.0;
            const double max_worst = kind == F16 ? 3.0 : kind == I8 ? 2.0 : 3.0;
            std::printf("%-6s %.4g -> %.4g: RMS error turned %.3g, written at B %.3g (x%.3f); worst turned %.3f, written "
                        "%.3f %s%s\n", names[kind], fa, fb, rt, rbl, ratio, st.worst_turn, st.worst_built,
                        kind == F16 ? "ulps" : "steps", untouched_bad ? " UNTOUCHED DIMS CHANGED" : "");
            if (!(ratio <= 1.6)) fail("the turned keys' RMS error is more than 1.6x the written keys'");
            if (!(st.worst_turn <= max_worst)) fail("a turned key is off by more than its bound");
            if (untouched_bad) fail("dims 64-255 changed without the Hadamard");
        }
        // the indexer's pooled keys: rows [0, n) at b * idx_block, the spare row n and `dead` at position 0
        {
            const int ID = (int) s.idx_dim, IB = (int) s.idx_block;
            const int64_t n = positions / IB;
            std::vector<float> raw_p((size_t) (n + 2) * ID), cur((size_t) (n + 2) * ID);
            for (float& x : raw_p) x = nd(rng);
            std::vector<double> rot((size_t) ID);
            for (int64_t r = 0; r < n + 2; ++r) {
                const int64_t pos = r < n ? r * IB : 0;
                rotate(scaling(fa), pos, raw_p.data() + (size_t) r * ID, ID, false, rot.data());
                for (int d = 0; d < ID; ++d) cur[(size_t) r * ID + d] = (float) rot[(size_t) d];
            }
            float* d_p = dalloc<float>((size_t) (n + 2) * ID);
            ck(cudaMemcpy(d_p, cur.data(), cur.size() * 4, cudaMemcpyHostToDevice), "pooled");
            k::rope_rerotate_pooled(d_p, d_p + (size_t) (n + 1) * ID, n, IB, ID, ra, rb, nullptr);
            std::vector<float> got(cur.size());
            ck(cudaMemcpy(got.data(), d_p, got.size() * 4, cudaMemcpyDeviceToHost), "d2h");
            ck(cudaFree(d_p), "free");
            double worst = 0;
            long exact_bad = 0;
            const float mratio = (float) (rb.mscale / ra.mscale);
            for (int64_t r = 0; r < n + 2; ++r) {
                if (r >= n) {   // position 0: the magnitude alone
                    for (int d = 0; d < 64; ++d)
                        if (got[(size_t) r * ID + d] != mratio * cur[(size_t) r * ID + d]) ++exact_bad;
                    continue;
                }
                rotate(scaling(fb), r * IB, raw_p.data() + (size_t) r * ID, ID, true, rot.data());
                double rmax = 0;
                for (int d = 0; d < ID; ++d) rmax = std::max(rmax, std::fabs(rot[(size_t) d]));
                const double ulp = rmax > 0 ? std::ldexp(1.0, std::ilogb(rmax) - 23) : 1.0;
                for (int d = 0; d < 64; ++d)
                    worst = std::max(worst, std::fabs((double) got[(size_t) r * ID + d] - rot[(size_t) d]) / ulp);
                for (int d = 64; d < ID; ++d)
                    if (got[(size_t) r * ID + d] != cur[(size_t) r * ID + d]) ++exact_bad;
            }
            std::printf("pooled %.4g -> %.4g: %lld rows, worst %.2f float ulps of the row's largest value%s\n", fa, fb,
                        (long long) n, worst, exact_bad ? ", SPARE/DEAD/UNROTATED DIMS DIFFER" : "");
            if (!(worst <= 8.0)) fail("a turned pooled key is off by more than 8 ulps");
            if (exact_bad) fail("the spare row, `dead` or dims 64-127 of a pooled key");
        }
    }
    // 5. the threaded table build, bit for bit
    {
        const int max_pos = 70000;
        const size_t n = (size_t) max_pos * 32;
        std::vector<float> c1(n), s1(n), c2(n), s2(n);
        for (double f : {1.0, 1.125, 2.0}) {
            k::build_rope_table(64, scaling(f), max_pos, c1.data(), s1.data());
            for (int threads : {1, 7, 16}) {
                std::fill(c2.begin(), c2.end(), -9.f);
                std::fill(s2.begin(), s2.end(), -9.f);
                k::build_rope_table_mt(64, scaling(f), max_pos, c2.data(), s2.data(), threads);
                if (std::memcmp(c1.data(), c2.data(), n * 4) != 0 || std::memcmp(s1.data(), s2.data(), n * 4) != 0)
                    fail("build_rope_table_mt differs from build_rope_table");
            }
        }
        std::printf("table: build_rope_table_mt = build_rope_table, bitwise (factors 1, 1.125, 2; 1/7/16 threads)\n");
    }
    std::printf("rope_dynamic_parity: %s\n", g_fail ? "FAILED" : "OK");
    return g_fail ? 1 : 0;
}
