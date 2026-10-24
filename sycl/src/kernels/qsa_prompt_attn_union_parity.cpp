// sycl/src/kernels/qsa_prompt_attn_union_parity.cpp - port-only: the union-tile XMX prompt attention
// (qsa_prompt_attn_union.hpp, STRATA_PROMPT_ATTN_UNION=1) against the FP32 kernel the prompt path runs without it
// (qsa_decode_attn_batch, 32 queries per launch, as prefill.cpp calls it) and an FP64 host reference (GPU, synthetic,
// no model).
//
// INT8 pools with random codes and scales behind a shuffled page table; selections shaped like the prompt path's:
// the 512 most recent cells plus older cells, of which each query redraws a fraction `drift` (0.03: neighbours share
// most of their cells; 1: every query its own set, the largest unions); short contexts take every cell. Optionally a
// share of the older pages is not resident (page -1), which both kernels must mask. Checks:
//   1. against FP64 on a sample of queries (the last one included): the union kernel's error is within 1e-2 of the
//      output scale (its products take FP16 inputs);
//   2. union vs FP32 on every output: max difference within 1e-2, mean within 1e-3 of the output scale, no
//      non-finite value;
//   3. a second run of the union kernel is bitwise equal to the first;
// then the time of both over the chunk (best of `reps`).
// Usage: qsa_prompt_attn_union_parity --selftest
//        qsa_prompt_attn_union_parity [context=32768] [queries=2048] [reps=5] [--drift F] [--sel FILE]
//   --sel FILE: a STRATA_DUMP_SEL dump of the engine (int32 T, cap, then T x cap cells and T widths) as the
//   selections of the last T positions of `context` (the engine's real overlap between neighbouring positions).
// Exit code 0 pass, 1 fail, 77 skipped (the kernel does not run on this card).
#include <sycl/sycl.hpp>
#include <dpct/dpct.hpp>
#include "strata/kernels/qsa.hpp"
#include "strata/kernels/qsa_decode_attn.hpp"
#include "strata/kernels/qsa_prompt_attn_union.hpp"

#include <algorithm>
#include <chrono>
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
sycl::queue& Q() { return dpct::get_in_order_queue(); }
template <typename T> T* up(const std::vector<T>& h) {
    T* d = sycl::malloc_device<T>(h.size() + 64, Q());
    if (!d) { std::fprintf(stderr, "malloc_device failed (%zu)\n", h.size()); std::exit(2); }
    Q().memcpy(d, h.data(), h.size() * sizeof(T)).wait();
    return d;
}
float h2f(uint16_t b) { return (float) sycl::bit_cast<sycl::half>(b); }
uint16_t f2h(float f) { return sycl::bit_cast<uint16_t>(sycl::half(f)); }
double now_ms() { return std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now().time_since_epoch()).count(); }
uint64_t hash_bits(const std::vector<float>& v) {
    uint64_t h = 1469598103934665603ull;
    for (float x : v) { uint32_t u; std::memcpy(&u, &x, 4); h = (h ^ u) * 1099511628211ull; }
    return h;
}

struct Case {
    int64_t ctx, nq;
    double drift;      // share of the older cells a query redraws (>= 1: a fresh set per query)
    double unmapped;   // share of the older pages that are not resident
    int reps;
    const char* sel;   // STRATA_DUMP_SEL file, or null
};

int run(const Case& c) {
    const k::QsaShapes s = k::qsa_real_shapes();
    const int64_t HD = s.head_dim, NKV = s.n_head_kv, NH = s.n_head, PS = s.page_size, ZV = NH * HD;
    const int64_t ctx = c.ctx, nq = c.nq, recent = 512;
    const int64_t pages = (ctx + PS - 1) / PS, rows = pages * NKV * PS;
    int64_t cap = k::qsa_selection_width(k::kTopkMaxCells, s);   // the prompt path's (prefill.cpp)
    std::mt19937 rng((unsigned) (ctx * 31 + nq));
    std::normal_distribution<float> nd(0.f, 1.f);
    std::uniform_int_distribution<int> code(-127, 127);
    std::uniform_real_distribution<float> sc(0.005f, 0.03f), u01(0.f, 1.f);
    std::vector<int8_t> kq((size_t) (rows * HD)), vq((size_t) (rows * HD));
    std::vector<uint16_t> ks((size_t) (rows * 4)), vs((size_t) (rows * 4));
    for (auto& x : kq) x = (int8_t) code(rng);
    for (auto& x : vq) x = (int8_t) code(rng);
    for (auto& x : ks) x = f2h(sc(rng));
    for (auto& x : vs) x = f2h(sc(rng));
    std::vector<int32_t> table((size_t) pages);
    for (int64_t i = 0; i < pages; ++i) table[i] = (int32_t) i;
    std::shuffle(table.begin(), table.end(), rng);
    // not resident: pages wholly older than every query's recent window
    for (int64_t pg = 0; c.unmapped > 0 && (pg + 1) * PS <= ctx - nq - recent; ++pg)
        if (u01(rng) < c.unmapped) table[pg] = -1;
    std::vector<int32_t> steps((size_t) (nq * k::kStepCount));
    for (int64_t i = 0; i < nq; ++i) k::qsa_step_fill(steps.data() + i * k::kStepCount, ctx - nq + i, s);
    std::vector<int32_t> ids;
    if (c.sel) {
        std::FILE* f = std::fopen(c.sel, "rb");
        int32_t hdr[2] = {0, 0};
        const bool head = f && std::fread(hdr, 4, 2, f) == 2 && hdr[0] == nq && hdr[1] > 0;
        cap = head ? hdr[1] : 0;
        ids.resize((size_t) (nq * cap));
        std::vector<int32_t> w((size_t) nq);
        const bool body = head && std::fread(ids.data(), 4, ids.size(), f) == ids.size() &&
                          std::fread(w.data(), 4, w.size(), f) == w.size();
        if (f) std::fclose(f);
        if (!body) {
            std::fprintf(stderr, "--sel %s: want a STRATA_DUMP_SEL dump of %lld positions\n", c.sel, (long long) nq);
            return 1;
        }
        for (int64_t i = 0; i < nq; ++i) {
            const int64_t nkv = ctx - nq + i + 1;
            bool in_range = w[i] == steps[i * k::kStepCount + k::kStepWidth];
            for (int64_t x = 0; in_range && x < w[i]; ++x) in_range = ids[i * cap + x] >= 0 && ids[i * cap + x] < nkv;
            if (!in_range) {
                std::fprintf(stderr, "--sel %s: position %lld does not fit context %lld (its width or cells)\n", c.sel,
                             (long long) i, (long long) ctx);
                return 1;
            }
        }
    } else {
        ids.assign((size_t) (nq * cap), 0);
        std::vector<int32_t> old;
        std::vector<uint8_t> in((size_t) ctx, 0);
        for (int64_t i = 0; i < nq; ++i) {
            const int64_t nkv = ctx - nq + i + 1, w = steps[i * k::kStepCount + k::kStepWidth];
            int32_t* sel = ids.data() + i * cap;
            if (w == nkv) {
                for (int64_t x = 0; x < w; ++x) sel[x] = (int32_t) x;
                continue;
            }
            const int64_t older = w - recent, span = nkv - recent;
            std::uniform_int_distribution<int64_t> any(0, span - 1);
            if ((int64_t) old.size() != older || c.drift >= 1.0) {   // a fresh set
                for (int32_t x : old) in[x] = 0;
                old.clear();
                while ((int64_t) old.size() < older) {
                    const int32_t x = (int32_t) any(rng);
                    if (!in[x]) { in[x] = 1; old.push_back(x); }
                }
            } else {   // redraw `drift` of them
                std::uniform_int_distribution<int64_t> pick(0, older - 1);
                const int64_t n = std::max<int64_t>(1, (int64_t) std::llround(c.drift * (double) older));
                for (int64_t r = 0; r < n; ++r) {
                    const int32_t x = (int32_t) any(rng);
                    if (in[x]) continue;
                    int32_t& slot = old[(size_t) pick(rng)];
                    in[slot] = 0; slot = x; in[x] = 1;
                }
            }
            std::vector<int32_t> v(old);
            for (int64_t x = span; x < nkv; ++x) v.push_back((int32_t) x);
            std::sort(v.begin(), v.end());
            std::copy(v.begin(), v.end(), sel);
        }
    }
    // the union of 8 neighbouring queries against one query's cells (docs/INTEL.md: 3.3x on an 80K prompt)
    double su = 0, sw = 0;
    for (int64_t t0 = 0; t0 + 8 <= nq; t0 += 8) {
        std::vector<int32_t> u;
        for (int64_t i = t0; i < t0 + 8; ++i) {
            const int64_t w = steps[i * k::kStepCount + k::kStepWidth];
            u.insert(u.end(), ids.begin() + i * cap, ids.begin() + i * cap + w);
            sw += (double) w;
        }
        std::sort(u.begin(), u.end());
        su += (double) (std::unique(u.begin(), u.end()) - u.begin());
    }
    const double union8 = sw > 0 ? su / (sw / 8) : 0.0;
    std::vector<float> q((size_t) (nq * ZV));
    for (auto& x : q) x = nd(rng) * 2.0f;

    k::QsaAttnPools pl;
    pl.k_q = up(kq); pl.v_q = up(vq); pl.k_scale = up(ks); pl.v_scale = up(vs); pl.page_table = up(table);
    const int32_t* d_ids = up(ids);
    const int32_t* d_steps = up(steps);
    const float* d_q = up(q);
    const int64_t batch = 32;
    float* d_old = sycl::malloc_device<float>((size_t) (nq * ZV), Q());
    float* d_new = sycl::malloc_device<float>((size_t) (nq * ZV), Q());
    float* scratch = sycl::malloc_device<float>((size_t) (batch * k::qsa_decode_attn_scratch_floats(cap, s)), Q());
    if (!d_old || !d_new || !scratch) { std::fprintf(stderr, "malloc_device failed\n"); std::exit(2); }
    auto old_run = [&]() {
        for (int64_t t0 = 0; t0 < nq; t0 += batch)
            k::qsa_decode_attn_batch(d_q + t0 * ZV, pl, d_ids + t0 * cap, d_steps + t0 * k::kStepCount, cap, s, scratch,
                                     d_old + t0 * ZV, std::min(batch, nq - t0), &Q());
    };
    auto new_run = [&]() { return k::qsa_prompt_attn_union(d_q, pl, d_ids, d_steps, cap, s, d_new, nq, &Q()); };
    auto release = [&]() {
        for (const void* p : {(const void*) d_ids, (const void*) d_steps, (const void*) d_q, (const void*) d_old,
                              (const void*) d_new, (const void*) scratch, (const void*) pl.k_q, (const void*) pl.v_q,
                              (const void*) pl.k_scale, (const void*) pl.v_scale, (const void*) pl.page_table})
            sycl::free(const_cast<void*>(p), Q());
    };
    old_run();
    if (!new_run()) {
        Q().wait();
        release();
        std::printf("SKIP ctx %lld: qsa_prompt_attn_union refused (see its stderr line)\n", (long long) ctx);
        return 77;
    }
    Q().wait();
    std::vector<float> o((size_t) (nq * ZV)), nw(o.size()), nw2(o.size());
    Q().memcpy(o.data(), d_old, o.size() * 4).wait();
    Q().memcpy(nw.data(), d_new, nw.size() * 4).wait();
    Q().memset(d_new, 0xff, nw.size() * 4).wait();   // NaN: the second run must write every output again
    new_run();
    Q().wait();
    Q().memcpy(nw2.data(), d_new, nw2.size() * 4).wait();
    const uint64_t h1 = hash_bits(nw), h2 = hash_bits(nw2);
    std::printf("UNIONHASH %016llx\n", (unsigned long long) h1);
    // 1. FP64 on a sample of queries
    double err_old = 0, err_new = 0, ref_scale = 0;
    std::vector<int64_t> sample;
    for (int64_t i = 0; i < nq; i += std::max<int64_t>(1, nq / 16)) sample.push_back(i);
    if (sample.back() != nq - 1) sample.push_back(nq - 1);
    for (int64_t i : sample) {
        const int64_t w = steps[i * k::kStepCount + k::kStepWidth];
        const int32_t* sel = ids.data() + i * cap;
        for (int64_t h = 0; h < NH; ++h) {
            const int64_t kvh = h / (NH / NKV);
            std::vector<double> sco((size_t) w, 0.0);
            std::vector<int64_t> row((size_t) w, -1);
            double mx = -1e300;
            for (int64_t x = 0; x < w; ++x) {
                const int64_t cell = sel[x], pg = table[cell / PS];
                if (pg < 0) continue;
                row[x] = (pg * NKV + kvh) * PS + cell % PS;
                double a = 0;
                for (int64_t d = 0; d < HD; ++d)
                    a += (double) q[(i * NH + h) * HD + d] * (double) kq[row[x] * HD + d] * h2f(ks[row[x] * 4 + d / 64]);
                sco[x] = a / 16.0;
                mx = std::max(mx, sco[x]);
            }
            double l = 0;
            for (int64_t x = 0; x < w; ++x) { sco[x] = row[x] < 0 ? 0.0 : std::exp(sco[x] - mx); l += sco[x]; }
            for (int64_t d = 0; d < HD; ++d) {
                double a = 0;
                for (int64_t x = 0; x < w; ++x)
                    if (row[x] >= 0) a += sco[x] * (double) vq[row[x] * HD + d] * h2f(vs[row[x] * 4 + d / 64]);
                const double r = a / l;
                const size_t at = (size_t) ((i * NH + h) * HD + d);
                ref_scale = std::max(ref_scale, std::fabs(r));
                err_old = std::max(err_old, std::fabs(o[at] - r));
                err_new = std::max(err_new, std::fabs(nw[at] - r));
            }
        }
    }
    // 2. union vs FP32 everywhere
    double diff = 0, sum = 0, scale = 0, row_rel = 0;
    size_t bad = 0;
    for (int64_t r = 0; r < nq * NH; ++r) {
        double rmax = 0, rdiff = 0;
        for (int64_t d = 0; d < HD; ++d) {
            const size_t i = (size_t) (r * HD + d);
            if (!std::isfinite(nw[i])) { ++bad; continue; }
            const double df = std::fabs((double) o[i] - (double) nw[i]);
            rmax = std::max(rmax, (double) std::fabs(o[i]));
            rdiff = std::max(rdiff, df);
            sum += df;
        }
        diff = std::max(diff, rdiff);
        scale = std::max(scale, rmax);
        if (rmax > 0) row_rel = std::max(row_rel, rdiff / rmax);
    }
    const double mean = sum / (double) o.size();
    // 3. time, best of reps
    auto best = [&](auto&& f) {
        double b = 1e30;
        for (int r = 0; r < c.reps; ++r) { const double t0 = now_ms(); f(); Q().wait(); b = std::min(b, now_ms() - t0); }
        return b;
    };
    const double ms_old = best(old_run), ms_new = best(new_run);
    const bool ok = bad == 0 && err_new <= 1e-2 * ref_scale && diff <= 1e-2 * scale && mean <= 1e-3 * scale && h1 == h2;
    std::printf("%s ctx %lld, %lld queries, %s (union of 8: %.2fx one query's cells), unmapped %.0f%%: vs FP64 fp32 %.3g "
                "union %.3g (output scale %.3g); union vs fp32 max %.3g mean %.3g (%.2g / %.2g of scale), max per-row rel "
                "%.3g, non-finite %zu, rerun %s; %.2f -> %.2f ms per chunk (%.2fx)\n",
                ok ? "PASS" : "FAIL", (long long) ctx, (long long) nq,
                c.sel ? "--sel" : (c.drift >= 1.0 ? "fresh selections" : "drifting selections"), union8, 100.0 * c.unmapped,
                err_old, err_new, ref_scale, diff, mean, diff / scale, mean / scale, row_rel, bad,
                h1 == h2 ? "bitwise equal" : "DIFFERS", ms_old, ms_new, ms_old / ms_new);
    release();
    return ok ? 0 : 1;
}
}  // namespace

int main(int argc, char** argv) {
    std::vector<Case> cases;
    if (argc > 1 && !std::strcmp(argv[1], "--selftest")) {
        cases = {
            {32768, 2600, 0.03, 0.0, 3, nullptr},   // two launch pairs (325 tiles), two bitmap passes
            {32768, 512, 1.0, 0.0, 3, nullptr},     // unions of ~8 selections
            {40000, 268, 0.3, 0.02, 3, nullptr},    // three bitmap passes, pages that are not resident, a 4-query tile
            {1500, 1500, 0.03, 0.0, 3, nullptr},    // every cell (causal masks only), a 4-query last tile
            {2100, 253, 0.03, 0.0, 3, nullptr},     // the identity-to-sparse edge, a 5-query last tile
        };
    } else {
        Case c{32768, 2048, 0.03, 0.0, 5, nullptr};
        int pos = 0;
        for (int i = 1; i < argc; ++i) {
            if (!std::strcmp(argv[i], "--drift") && i + 1 < argc) c.drift = std::atof(argv[++i]);
            else if (!std::strcmp(argv[i], "--sel") && i + 1 < argc) c.sel = argv[++i];
            else if (pos == 0) { c.ctx = std::atoll(argv[i]); ++pos; }
            else if (pos == 1) { c.nq = std::atoll(argv[i]); ++pos; }
            else if (pos == 2) { c.reps = std::max(1, std::atoi(argv[i])); ++pos; }
        }
        if (c.nq > c.ctx || c.nq <= 0) { std::fprintf(stderr, "need 0 < queries <= context\n"); return 2; }
        cases.push_back(c);
    }
    std::printf("device %s\n", Q().get_device().get_info<sycl::info::device::name>().c_str());
    int fails = 0, skips = 0;
    for (const Case& c : cases) {
        const int r = run(c);
        if (r == 77) ++skips;
        else fails += r;
    }
    if (skips == (int) cases.size()) return 77;
    std::printf("FAILURES: %d\n", fails);
    return fails ? 1 : 0;
}
