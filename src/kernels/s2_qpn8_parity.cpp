// src/kernels/s2_qpn8_parity.cpp - the V100 QPN8 repack + m8n8k4 path against the DP4A kernels (GPU,
// synthetic, no model).
//
//     build/s2_qpn8_parity --selftest     exit 0 when the QPN8 outputs are within the float-order bound
//
// Skips (and passes) when `s2_qpn8_active()` is false - the path is STRATA_QPN8=1 opt-in and Volta-only, so
// the default tree and every other architecture run the DP4A kernels and see this test as a no-op.
//
// WHAT IS CHECKED.  Same blobs (one canonical, one repacked through `s2_qpn8_repack_blob`), same routing,
// same activations; `moe_grouped_s2` and `moe_grouped_s2_qpn8` run the whole gate/up -> swiglu -> quantize ->
// down pipeline and their outputs are compared, along the way the gate/up intermediate and the quantized
// intermediate as well.  The per-chunk terms of the two are bit-identical (the 1025-bias trick gives the
// exact integer `s - hx` and the readout is the same `dw * dx * (s - hx)` expression); only the order the
// chunk terms are ADDED differs, so the outputs must agree to float-order level - `compare` states the
// metric and the bound (absolute 1e-4 + L1 1e-6; measured noise 9.5e-6, a structural bug lands at 1..10).
// The quantized intermediates must be BYTE-identical, which pins the down stages to the same inputs.
// Both activation contracts are covered: fp32 scales (`quantize_q8_0_scaled`) and the block's fp16 `d`.
// Groups of 1..8 entries exercise the m8 tile's partial occupancy, which is where a wrong m/entry map
// shows up first.  **`ent_dst` is a permutation** - the engines' plans give every entry its own output row,
// and duplicated destinations make both kernels race on `out[]` (that cost this test one afternoon).
#include "strata/kernels/f16_bits.hpp"
#include "strata/kernels/s2_expert_grouped.hpp"
#include "strata/kernels/s2_qpn8.hpp"

#include <cuda_runtime.h>

#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <random>
#include <string>
#include <vector>

namespace k = strata::kernels;

namespace {
constexpr int H = 2560, FF = 640, NCH_GU = H / 32, NCH_D = FF / 32;
constexpr size_t ROW_GU = H / 4, ROW_D = FF / 4, SC_GU = H / 64, SC_D = FF / 64;
constexpr size_t O_D_CODES = 2ull * FF * ROW_GU;
constexpr size_t O_GU_SCALES = O_D_CODES + (size_t) H * ROW_D;
constexpr size_t O_D_SCALES = O_GU_SCALES + 2ull * FF * SC_GU * 2;
constexpr size_t BLOB = O_D_SCALES + (size_t) H * SC_D * 2;
static_assert(BLOB == 1382400, "the Q2_0 expert blob is 1,382,400 bytes");
constexpr size_t XROW = (size_t) NCH_GU * 34;

void ck(cudaError_t e, const char* w) {
    if (e != cudaSuccess) { std::fprintf(stderr, "%s: %s\n", w, cudaGetErrorString(e)); std::exit(2); }
}
template <typename T> T* dalloc(size_t n) {
    T* p = nullptr;
    ck(cudaMalloc(&p, n * sizeof(T) + 256), "malloc");
    ck(cudaMemset(p, 0, n * sizeof(T) + 256), "memset");
    return p;
}
template <typename T> void up(T* d, const std::vector<T>& h) {
    ck(cudaMemcpy(d, h.data(), h.size() * sizeof(T), cudaMemcpyHostToDevice), "h2d");
}
template <typename T> std::vector<T> down(const T* d, size_t n) {
    std::vector<T> h(n);
    ck(cudaMemcpy(h.data(), d, n * sizeof(T), cudaMemcpyDeviceToHost), "d2h");
    return h;
}

void put16(uint8_t* p, float f) {
    const uint16_t h = k::f16_from_f32(f);
    p[0] = (uint8_t) h;
    p[1] = (uint8_t) (h >> 8);
}

void fill_blob(uint8_t* b, std::mt19937& rng) {
    for (size_t i = 0; i < O_GU_SCALES; ++i) b[i] = (uint8_t) rng();
    std::uniform_real_distribution<float> mag(1e-3f, 3e-2f);
    for (size_t i = O_GU_SCALES; i < BLOB; i += 2) put16(b + i, (rng() & 1 ? -1.0f : 1.0f) * mag(rng));
}

void fill_x(std::vector<uint8_t>& x, std::vector<float>& xs, int n_tok, std::mt19937& rng) {
    x.assign((size_t) n_tok * XROW, 0);
    xs.assign((size_t) n_tok * NCH_GU, 0.0f);
    std::uniform_real_distribution<float> sc(2e-3f, 5e-2f);
    for (size_t c = 0; c < (size_t) n_tok * NCH_GU; ++c) {
        uint8_t* b = x.data() + c * 34;
        xs[c] = sc(rng);
        put16(b, xs[c]);
        for (int j = 2; j < 34; ++j) b[j] = (uint8_t) (int8_t) (rng() % 255 - 127);
    }
}

int g_fail = 0;

// worst |a - b| / (|a| + |b| + floor) and worst absolute difference
// The comparison metric and bound.  The per-chunk terms of the two paths are bit-identical (the 1025-bias
// trick gives the exact integer `s - hx`, and the readout is the same `dw * dx * (float)(s - hx)`); only
// the order the chunk terms are ADDED differs (lane-strided shuffle tree vs k order through the split-K
// reduce).  So the error is float-order: bounded by ~n * eps * sum|term| and visible as ~1e-5 ABSOLUTE at
// this test's magnitudes (measured: 9.5e-6 on the gate/up intermediate, 2.6e-6 on the outputs) - which a
// per-element relative metric misreads as 5e-4 wherever the result nearly cancels.  The assertions are
// therefore absolute (1e-4, two orders below a structural bug's 1..10 and one above the measured noise)
// plus an L1-relative global (1e-6); the numbers printed are the full picture.
void compare(const char* what, const std::vector<float>& a, const std::vector<float>& b) {
    double worst = 0.0, wabs = 0.0, l1a = 0.0, l1d = 0.0;
    size_t at = 0;
    int bad = 0, shown = 0;
    for (size_t i = 0; i < a.size(); ++i) {
        const double d = std::fabs((double) a[i] - (double) b[i]);
        const double r = d / (std::fabs((double) a[i]) + std::fabs((double) b[i]) + 1e-3);
        if (r > worst) { worst = r; at = i; }
        if (d > wabs) wabs = d;
        l1a += std::fabs((double) a[i]);
        l1d += d;
        if (d > 1e-3) {
            ++bad;
            if (shown < 12) {
                std::printf("    differs at entry %zu row %zu: %.6f vs %.6f\n", i / H, i % H, (double) a[i],
                            (double) b[i]);
                ++shown;
            }
        }
    }
    std::printf("  %-36s worst relative %.3e (at %zu: %.6f vs %.6f), worst abs %.3e, L1 relative %.3e, "
                "%d of %zu above 1e-3\n",
                what, worst, at, (double) a[at], (double) b[at], wabs, l1d / (l1a + 1e-30), bad, a.size());
    if (wabs > 1e-4 || l1d / (l1a + 1e-30) > 1e-6) {
        std::printf("  %-36s FAIL: above the float-order bound\n", what);
        g_fail = 1;
    }
}

}  // namespace

// ---- `--bench`: the two entry points on the V100, CUDA events, no model ------------------------------------
//
// Window shapes as (groups G, entries per group ne): the shipped `moe_grouped_s2` (in-kernel 2-bit unpack)
// against `moe_grouped_s2_qpn8` (load-time repack + m8n8k4).  The first family keeps G = 10 and sweeps ne
// (the task's M = 1/2/4/8); the second is the verify window's shape family at 16 entries (T = 8 tokens x
// top-k 2): ne = 16 / G.  The m8 tile always computes 8 rows, so its unused rows are pure loss when
// ne < 8 - that is what the sweep prices.  Blobs cycle through a set larger than L2 so the codes come from
// DRAM as the engine's do.
namespace bench {
void run() {
    if (!k::s2_qpn8_active()) {
        std::printf("s2_qpn8_parity --bench: skipped (STRATA_QPN8 != 1 or not a Volta device)\n");
        return;
    }
    std::mt19937 rng(11);
    const int nb = 48;
    std::vector<uint8_t> slots((size_t) nb * 2 * BLOB);
    for (int i = 0; i < nb; ++i) fill_blob(slots.data() + (size_t) i * 2 * BLOB, rng);
    uint8_t* d_slots = nullptr;
    ck(cudaMalloc(&d_slots, slots.size()), "bench slots");
    ck(cudaMemcpy(d_slots, slots.data(), slots.size(), cudaMemcpyHostToDevice), "bench slots up");
    for (int i = 0; i < nb; ++i)
        k::s2_qpn8_repack_blob(d_slots + (size_t) i * 2 * BLOB + BLOB, d_slots + (size_t) i * 2 * BLOB,
                               (int64_t) BLOB, nullptr);
    ck(cudaDeviceSynchronize(), "bench repack");

    const int T = 8;
    std::vector<uint8_t> x;
    std::vector<float> xs;
    fill_x(x, xs, T, rng);
    uint8_t* d_x = nullptr;
    float* d_xs = nullptr;
    ck(cudaMalloc(&d_x, x.size()), "bench x");
    ck(cudaMemcpy(d_x, x.data(), x.size(), cudaMemcpyHostToDevice), "bench x up");
    ck(cudaMalloc(&d_xs, xs.size() * 4), "bench xs");
    ck(cudaMemcpy(d_xs, xs.data(), xs.size() * 4, cudaMemcpyHostToDevice), "bench xs up");

    struct Shape { int g, ne; const char* what; };
    const Shape shapes[] = {
        {10, 1, "10 experts x 1 entry "},   {10, 2, "10 experts x 2 entries"},
        {10, 4, "10 experts x 4 entries"},  {10, 8, "10 experts x 8 entries"},
        {16, 1, "window 16e: 16 x 1"},      {8, 2, "window 16e: 8 x 2"},
        {4, 4, "window 16e: 4 x 4"},        {2, 8, "window 16e: 2 x 8"},
    };
    cudaDeviceProp prop{};
    ck(cudaGetDeviceProperties(&prop, 0), "props");
    const char* only = std::getenv("STRATA_QPN8_BENCH_SHAPE");
    std::printf("s2_qpn8_parity --bench: the entry points on %s (us per window call, 20 reps)\n", prop.name);
    std::printf("  %-24s %10s %10s %8s\n", "shape (all entries shared)", "unpack DP4A", "repack m8n8k4", "ratio");
    for (const Shape& s : shapes) {
        if (only != nullptr && std::strstr(s.what, only) == nullptr) continue;
        const int groups = s.g, entries = groups * s.ne;
        std::vector<unsigned long long> gptr(groups);
        std::vector<int32_t> gstart(groups + 1), ng(1), edst(entries), etok(entries);
        for (int g = 0; g < groups; ++g) {
            gptr[g] = (unsigned long long) (d_slots + (size_t) (g % nb) * 2 * BLOB);
            gstart[g] = g * s.ne;
            for (int e = 0; e < s.ne; ++e) {
                edst[g * s.ne + e] = (int32_t) ((g * s.ne + e) * 13 + 5) % entries;   // a permutation
                etok[g * s.ne + e] = (int32_t) ((g * s.ne + e) % T);
            }
        }
        gstart[groups] = entries;
        ng[0] = groups;
        unsigned long long* d_gptr = dalloc<unsigned long long>(gptr.size());
        int32_t* d_gstart = dalloc<int32_t>(gstart.size());
        int32_t* d_ng = dalloc<int32_t>(1);
        int32_t* d_edst = dalloc<int32_t>(edst.size());
        int32_t* d_etok = dalloc<int32_t>(etok.size());
        up(d_gptr, gptr);
        up(d_gstart, gstart);
        up(d_ng, ng);
        up(d_edst, edst);
        up(d_etok, etok);
        const uint64_t scratch_n = k::moe_hit_grouped_scratch_bytes(entries, H, FF);
        void* s_a = nullptr;
        void* s_b = nullptr;
        ck(cudaMalloc(&s_a, scratch_n), "bench scratch a");
        ck(cudaMalloc(&s_b, scratch_n), "bench scratch b");
        float* out_a = dalloc<float>((size_t) entries * H);
        float* out_b = dalloc<float>((size_t) entries * H);

        auto time = [&](bool qpn8) {
            for (int i = 0; i < 5; ++i) {
                if (qpn8)
                    k::moe_grouped_s2_qpn8(d_gptr, d_gstart, d_ng, d_edst, d_etok, groups, entries, (int64_t) BLOB,
                                           d_x, d_xs, s_b, out_b, nullptr);
                else
                    k::moe_grouped_s2(d_gptr, d_gstart, d_ng, d_edst, d_etok, groups, entries, d_x, d_xs, s_a,
                                      out_a, nullptr);
            }
            ck(cudaDeviceSynchronize(), "bench warmup");
            cudaEvent_t a, b;
            ck(cudaEventCreate(&a), "ev");
            ck(cudaEventCreate(&b), "ev");
            ck(cudaEventRecord(a), "ev rec");
            for (int i = 0; i < 20; ++i) {
                if (qpn8)
                    k::moe_grouped_s2_qpn8(d_gptr, d_gstart, d_ng, d_edst, d_etok, groups, entries, (int64_t) BLOB,
                                           d_x, d_xs, s_b, out_b, nullptr);
                else
                    k::moe_grouped_s2(d_gptr, d_gstart, d_ng, d_edst, d_etok, groups, entries, d_x, d_xs, s_a,
                                      out_a, nullptr);
            }
            ck(cudaEventRecord(b), "ev rec");
            ck(cudaEventSynchronize(b), "ev sync");
            float ms = 0.0f;
            ck(cudaEventElapsedTime(&ms, a, b), "ev elapsed");
            cudaEventDestroy(a);
            cudaEventDestroy(b);
            return (double) ms * 1000.0 / 20.0;
        };
        const double us_dp4a = time(false);
        const double us_qpn8 = time(true);
        std::printf("  %-24s %10.1f %10.1f %7.2fx\n", s.what, us_dp4a, us_qpn8, us_dp4a / us_qpn8);
        cudaFree(s_a);
        cudaFree(s_b);
        cudaFree(out_a);
        cudaFree(out_b);
        cudaFree(d_gptr);
        cudaFree(d_gstart);
        cudaFree(d_ng);
        cudaFree(d_edst);
        cudaFree(d_etok);
    }
    std::printf("  (full pipeline: gate/up -> swiglu -> quantize -> down; the shared stages are in both "
                "columns)\n");
    cudaFree(d_slots);
    cudaFree(d_x);
    cudaFree(d_xs);
}
}  // namespace bench

int main(int argc, char** argv) {
    bool selftest = false, do_bench = false;
    for (int i = 1; i < argc; ++i) {
        if (std::strcmp(argv[i], "--selftest") == 0) selftest = true;
        if (std::strcmp(argv[i], "--bench") == 0) do_bench = true;
    }
    if (do_bench) {
        bench::run();
        return 0;
    }
    if (!selftest) {
        std::fprintf(stderr, "usage: s2_qpn8_parity --selftest | --bench\n");
        return 2;
    }
    if (!k::s2_qpn8_active()) {
        std::printf("s2_qpn8_parity: skipped (STRATA_QPN8 != 1 or not a Volta device); PASS\n");
        return 0;
    }
    std::printf("s2_qpn8_parity: the QPN8 m8n8k4 path vs the DP4A kernels (float-order bound: see compare)\n");

    std::mt19937 rng(7);
    // dual-form slots: canonical at i * 2 * BLOB, the repacked copy at + BLOB (the fill hook's layout)
    const int nb = 6;
    std::vector<uint8_t> canon((size_t) nb * BLOB), slots((size_t) nb * 2 * BLOB);
    for (int i = 0; i < nb; ++i) {
        fill_blob(canon.data() + (size_t) i * BLOB, rng);
        std::memcpy(slots.data() + (size_t) i * 2 * BLOB, canon.data() + (size_t) i * BLOB, BLOB);
    }
    uint8_t* d_slots = nullptr;
    ck(cudaMalloc(&d_slots, slots.size()), "slots");
    ck(cudaMemcpy(d_slots, slots.data(), slots.size(), cudaMemcpyHostToDevice), "slots up");
    for (int i = 0; i < nb; ++i)
        k::s2_qpn8_repack_blob(d_slots + (size_t) i * 2 * BLOB + BLOB, d_slots + (size_t) i * 2 * BLOB,
                               (int64_t) BLOB, nullptr);
    ck(cudaDeviceSynchronize(), "repack");

    // groups of 1..8 entries over the nb blobs; entries read token rows of x.
    //
    // **`ent_dst` MUST BE A PERMUTATION.**  The first version used `entries % (T * K)`, which maps 45
    // entries onto 32 destinations: both kernels ASSIGN `out[dst]`, so two blocks writing one row race and
    // the comparison then measured the scheduler instead of the kernels (nondeterministic counts across
    // runs).  The engine's plans give every entry its own row, so the test must too - shuffled for coverage.
    const int T = 8, K = 4;                       // 32 token rows of x
    std::vector<int> ge{1, 2, 3, 4, 5, 6, 7, 8, 1, 8};
    int entries = 0;
    for (int n : ge) entries += n;
    const int total = entries;
    std::vector<unsigned long long> gptr(ge.size());
    std::vector<int32_t> gstart(ge.size() + 1), ng(1), edst(entries), etok(entries);
    entries = 0;
    for (size_t g = 0; g < ge.size(); ++g) {
        gptr[g] = (unsigned long long) (d_slots + (size_t) (g % nb) * 2 * BLOB);
        gstart[g] = entries;
        for (int e = 0; e < ge[g]; ++e) {
            edst[entries] = (int32_t) ((entries * 13 + 5) % total);   // gcd(13, 45) = 1: a permutation
            etok[entries] = (int32_t) (entries % T);
            ++entries;
        }
    }
    gstart[ge.size()] = entries;
    ng[0] = (int32_t) ge.size();
    const int cap = entries;

    unsigned long long* d_gptr = dalloc<unsigned long long>(gptr.size());
    int32_t* d_gstart = dalloc<int32_t>(gstart.size());
    int32_t* d_ng = dalloc<int32_t>(1);
    int32_t* d_edst = dalloc<int32_t>(edst.size());
    int32_t* d_etok = dalloc<int32_t>(etok.size());
    up(d_gptr, gptr);
    up(d_gstart, gstart);
    up(d_ng, ng);
    up(d_edst, edst);
    up(d_etok, etok);

    std::vector<uint8_t> x;
    std::vector<float> xs;
    fill_x(x, xs, T, rng);
    uint8_t* d_x = dalloc<uint8_t>(x.size());
    float* d_xs = dalloc<float>(xs.size());
    up(d_x, x);
    up(d_xs, xs);

    const uint64_t scratch_n = k::moe_hit_grouped_scratch_bytes(cap, H, FF);
    void* s_a = nullptr;
    void* s_b = nullptr;
    ck(cudaMalloc(&s_a, scratch_n), "scratch a");
    ck(cudaMalloc(&s_b, scratch_n), "scratch b");
    ck(cudaMemset(s_a, 0, scratch_n), "zero a");
    ck(cudaMemset(s_b, 0, scratch_n), "zero b");
    float* out_a = dalloc<float>((size_t) cap * H);
    float* out_b = dalloc<float>((size_t) cap * H);

    for (int with_xs = 0; with_xs < 2; ++with_xs) {
        const float* xs_arg = with_xs ? d_xs : nullptr;
        ck(cudaMemset(out_a, 0, (size_t) cap * H * 4), "zero oa");
        ck(cudaMemset(out_b, 0, (size_t) cap * H * 4), "zero ob");
        k::moe_grouped_s2(d_gptr, d_gstart, d_ng, d_edst, d_etok, cap, cap, d_x, xs_arg, s_a, out_a, nullptr);
        k::moe_grouped_s2_qpn8(d_gptr, d_gstart, d_ng, d_edst, d_etok, cap, cap, (int64_t) BLOB, d_x, xs_arg,
                               s_b, out_b, nullptr);
        ck(cudaDeviceSynchronize(), "run");
        {
            const size_t n = (size_t) cap * 2 * FF;
            const std::vector<float> ga = down((const float*) s_a, n);
            const std::vector<float> gb = down((const float*) s_b, n);
            compare(with_xs ? "gate_up intermediate, fp32 scales" : "gate_up intermediate, fp16 d", ga, gb);
        }
        {
            // the quantized intermediate each pipeline feeds its down stage (int8, block_q8_0 bodies)
            const uint64_t gu_bytes = ((uint64_t) cap * (uint64_t) (2 * FF) * 4 + 15) & ~15ull;
            const size_t nb = (size_t) cap * (size_t) NCH_D * 34;
            const std::vector<uint8_t> qa = down((const uint8_t*) ((const uint8_t*) s_a + gu_bytes), nb);
            const std::vector<uint8_t> qb = down((const uint8_t*) ((const uint8_t*) s_b + gu_bytes), nb);
            int worst = 0, nd = 0;
            for (size_t i = 0; i < nb; ++i) {
                const int d = std::abs((int) qa[i] - (int) qb[i]);
                if (d > worst) worst = d;
                if (d > 0) ++nd;
            }
            std::printf("  %-36s %d of %zu bytes differ, worst byte delta %d\n", "quantized intermediate", nd, nb,
                        worst);
            if (nd != 0) {
                std::printf("  %-36s FAIL: the pipelines' quantized intermediates differ\n", "quantized intermediate");
                g_fail = 1;
            }
        }
        const std::vector<float> a = down(out_a, (size_t) cap * H);
        const std::vector<float> b = down(out_b, (size_t) cap * H);
        compare(with_xs ? "moe_grouped_s2, fp32 scales" : "moe_grouped_s2, fp16 d", a, b);
    }

    if (g_fail == 0) std::printf("s2_qpn8_parity: PASS\n");
    else std::printf("s2_qpn8_parity: FAIL\n");
    return g_fail ? 1 : 0;
}
