// prefill_fp16tc_test - the sm_70 FP16 WMMA routed experts of the prompt path (moe_fp16tc.hpp, built with
// STRATA_PREFILL_FP16TC=1) on random native Q2_0 expert blobs and random routing, against
//   - a double-precision reference: the dequantized weights times the FP32 activations (gate/up, SwiGLU, down), and
//   - the MMQ path in prefill.cpp's sequence (gather_native into 16-expert groups, q8_1 of the slots' activations,
//     gate/up, SwiGLU, q8_1 of H, down).
// Both reconstruct the q8_1 activations and the weights differently, so this is not a bitwise test: the FP16TC
// path's error against the reference has to be comparable to MMQ's own (at most 1.5x its RMS and 2x its worst row).
// Part 1 (reference): 256 tokens over 64 experts, skewed routing - experts with 0 rows, with one, with several 64-row
// tiles - an all-zero token, per-expert blobs at unrelated addresses.
// Part 2 (compact): a sub-group of one group run on its own must reproduce the
// full-group rows exactly.  Every live group is covered, and at least one sub-group of a group whose absolute start
// is nonzero has to be exercised: with the first group (absolute start 0) an absolute row base and a group-relative
// one are the same number, so a caller that confused the two would still pass.
// The parity has to hold on every visible Volta device (both V100s).  Exit 77 without a Volta device.
#include "strata/prefill/moe_fp16tc.hpp"
#include "strata/prefill/moe_mmq.hpp"

#include "ggml.h"

#include <cuda_runtime.h>

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <memory>
#include <random>
#include <stdexcept>
#include <string>
#include <thread>
#include <vector>

namespace {
namespace mmq = strata::prefill::mmq;
namespace fp16tc = strata::prefill::fp16tc;

constexpr int N = 2560, FF = 640, K = 10, GROUP = 16, Q2 = 42;   // GGML_TYPE_Q2_0
constexpr size_t GU_ROW = 720, D_ROW = 180;                     // Q2_0 row bytes (18 B per 64 values)
constexpr size_t UP_OFF = FF * GU_ROW, GU_BYTES = 2 * UP_OFF, D_BYTES = N * D_ROW, BLOB = GU_BYTES + D_BYTES;
constexpr size_t O_DOWN = GU_BYTES;
static_assert(GROUP == fp16tc::kMaxBatch, "a group must fit the batch's blob/down pointer arrays");

void ck(cudaError_t e, const char* what) {
    if (e != cudaSuccess) throw std::runtime_error(std::string(what) + ": " + cudaGetErrorString(e));
}

struct Dev {
    void* p = nullptr;
    explicit Dev(size_t n) { ck(cudaMalloc(&p, n), "cudaMalloc"); }
    ~Dev() { cudaFree(p); }
    Dev(const Dev&) = delete;
    Dev& operator=(const Dev&) = delete;
    template <typename T> T* as() const { return (T*) p; }
};

// A native Q2_0 expert blob: gate [640 rows][720 B], up [640][720], down [2560][180]; each row 40 (10) blocks of
// {fp16 scale; 16 code bytes}.  Scales in [0.004, 0.03] (a 2-bit expert's range) and random code bytes, so all four
// 2-bit codes - the dequant's signed (q - 1) - are covered.
std::vector<uint8_t> make_blob(std::mt19937& rng) {
    std::vector<uint8_t> b(BLOB);
    std::uniform_int_distribution<int> byte(0, 255);
    std::uniform_real_distribution<float> sc(0.004f, 0.03f);
    auto rows = [&](size_t base, int nrows, size_t row_bytes) {
        const int nb = (int) (row_bytes / 18);
        for (int r = 0; r < nrows; ++r)
            for (int q = 0; q < nb; ++q) {
                const size_t at = base + (size_t) r * row_bytes + (size_t) q * 18;
                const ggml_fp16_t h = ggml_fp32_to_fp16(sc(rng));
                std::memcpy(&b[at], &h, 2);
                for (int j = 0; j < 16; ++j) b[at + 2 + j] = (uint8_t) byte(rng);
            }
    };
    rows(0, FF, GU_ROW);         // gate
    rows(UP_OFF, FF, GU_ROW);    // up
    rows(O_DOWN, N, D_ROW);      // down
    return b;
}

// the blob's weights as floats: gate/up [1280][2560] (gate rows then up rows), down [2560][640]
void dequant(const std::vector<uint8_t>& b, std::vector<float>& gu, std::vector<float>& dn) {
    auto row = [&](size_t base, size_t row_bytes, int nb, int r, float* out) {
        for (int k = 0; k < nb * 64; ++k) {
            ggml_fp16_t h;
            std::memcpy(&h, &b[base + (size_t) r * row_bytes + (size_t) (k / 64) * 18], 2);
            const int q = (b[base + (size_t) r * row_bytes + (size_t) (k / 64) * 18 + 2 + (k % 64) / 4] >>
                           (2 * (k % 4))) & 3;
            out[k] = ggml_fp16_to_fp32(h) * (float) (q - 1);
        }
    };
    gu.resize((size_t) 1280 * N);
    dn.resize((size_t) N * FF);
    for (int r = 0; r < FF; ++r) row(0, GU_ROW, 40, r, gu.data() + (size_t) r * N);
    for (int r = 0; r < FF; ++r) row(UP_OFF, GU_ROW, 40, r, gu.data() + (size_t) (FF + r) * N);
    for (int r = 0; r < N; ++r) row(O_DOWN, D_ROW, 10, r, dn.data() + (size_t) r * FF);
}

// activations: N(0, 1) with a few large values (the hidden state has outliers); token `zero` all zero
std::vector<float> make_x(int T, std::mt19937& rng, int zero) {
    std::vector<float> x((size_t) T * N);
    std::normal_distribution<float> nd(0.0f, 1.0f);
    std::uniform_int_distribution<int> pick(0, 199);
    for (float& v : x) v = nd(rng) * (pick(rng) == 0 ? 12.0f : 1.0f);
    if (zero >= 0) std::fill(x.begin() + (size_t) zero * N, x.begin() + (size_t) (zero + 1) * N, 0.0f);
    return x;
}

// K distinct experts per token; `hot` > 0 skews the first choices towards the lowest ids
std::vector<int32_t> make_ids(int T, int E, std::mt19937& rng, int hot, int unused) {
    std::vector<int32_t> ids((size_t) T * K);
    std::uniform_int_distribution<int> any(0, E - 1 - unused);
    for (int t = 0; t < T; ++t)
        for (int k = 0; k < K; ++k) {
            int e;
            bool dup;
            do {
                e = (hot > 0 && k < 3 && (int) (rng() % 4) != 0) ? (int) (rng() % (unsigned) hot) : any(rng);
                dup = false;
                for (int j = 0; j < k; ++j) dup |= ids[(size_t) t * K + j] == e;
            } while (dup);
            ids[(size_t) t * K + k] = e;
        }
    return ids;
}

struct Routing {
    std::vector<int32_t> cnt, off, src;   // per expert; sorted row -> token
    std::vector<int32_t> row_of;          // pair -> sorted row
};
Routing sort_rows(const std::vector<int32_t>& ids, int E) {
    Routing r;
    r.cnt.assign((size_t) E, 0);
    for (int32_t e : ids) ++r.cnt[(size_t) e];
    r.off.assign((size_t) E + 1, 0);
    for (int e = 0; e < E; ++e) r.off[(size_t) e + 1] = r.off[(size_t) e] + r.cnt[(size_t) e];
    std::vector<int32_t> fill(r.off.begin(), r.off.end() - 1);
    r.src.resize(ids.size());
    r.row_of.resize(ids.size());
    for (size_t i = 0; i < ids.size(); ++i) {
        const int32_t p = fill[(size_t) ids[i]]++;
        r.row_of[i] = p;
        r.src[(size_t) p] = (int32_t) (i / K);
    }
    return r;
}

struct Bufs {
    Dev xq, gu, h, hq, dm, ident, bounds, grp_gu, grp_d;
    Bufs(int64_t rows, int E)
        : xq(mmq::q8_bytes(rows, N)), gu((size_t) rows * 1280 * 4), h((size_t) rows * FF * 4),
          hq(mmq::q8_bytes(rows, FF)), dm((size_t) rows * N * 4), ident((size_t) rows * 4),
          bounds((size_t) (2 * (E + E / GROUP + 2)) * 4), grp_gu(GROUP * GU_BYTES + 4096),
          grp_d(GROUP * D_BYTES + 4096) {}
};

// The routing -> the device bounds + the per-group schedule both paths share (order, offsets, group index).
struct Plan {
    std::vector<int32_t> order, bh;
    size_t n = 0, ng = 0;
};
Plan plan_groups(const Routing& r, int64_t rows) {
    Plan p;
    const int E = (int) r.cnt.size();
    for (int e = 0; e < E; ++e) if (r.cnt[(size_t) e] > 0) p.order.push_back(e);
    p.n = p.order.size();
    p.ng = (p.n + GROUP - 1) / GROUP;
    p.bh.assign(p.n + 1 + p.ng * (GROUP + 1), 0);
    for (size_t j = 0; j < p.n; ++j) p.bh[j] = r.off[(size_t) p.order[j]];
    p.bh[p.n] = (int32_t) rows;
    for (size_t g = 0; g < p.ng; ++g)
        for (size_t i = 0; i <= GROUP; ++i)
            p.bh[p.n + 1 + g * (GROUP + 1) + i] = p.bh[std::min(p.n, g * GROUP + i)] - p.bh[g * GROUP];
    return p;
}

// MMQ, as prefill.cpp runs it: per group of 16 routed experts a native gather, gate/up, SwiGLU, q8_1 of H, down.
void run_mmq(mmq::Context& ctx, Bufs& b, const Plan& p, const Routing& r, const float* x_dev, const int32_t* src_dev,
             const std::vector<const uint8_t*>& blob, int64_t rows, cudaStream_t s) {
    ck(cudaMemcpyAsync(b.bounds.p, p.bh.data(), p.bh.size() * 4, cudaMemcpyHostToDevice, s), "bounds");
    mmq::iota(b.ident.as<int32_t>(), rows, s);
    mmq::quantize(x_dev, src_dev, b.xq.p, Q2, N, N, rows, s);
    for (size_t j = 0; j < p.n; ++j) {
        const size_t q = j % GROUP;
        mmq::gather_native(blob[(size_t) p.order[j]], blob[(size_t) p.order[j]] + UP_OFF, UP_OFF,
                           blob[(size_t) p.order[j]] + O_DOWN, D_BYTES, b.grp_gu.as<uint8_t>() + q * GU_BYTES,
                           b.grp_d.as<uint8_t>() + q * D_BYTES, s);
        if (q + 1 < GROUP && j + 1 < p.n) continue;
        const size_t j0 = j - q, gi = j0 / GROUP;
        const int ngx = (int) (q + 1);
        const int64_t r0 = p.bh[j0], nr = p.bh[j + 1] - r0;
        int64_t maxr = 0;
        for (size_t i = j0; i <= j; ++i) maxr = std::max<int64_t>(maxr, r.cnt[(size_t) p.order[i]]);
        ck(cudaMemsetAsync(b.grp_gu.as<uint8_t>() + (size_t) ngx * GU_BYTES, 0, 4096, s), "tail");
        ck(cudaMemsetAsync(b.grp_d.as<uint8_t>() + (size_t) ngx * D_BYTES, 0, 4096, s), "tail");
        mmq::Product gu;
        gu.w = b.grp_gu.p; gu.type = Q2; gu.w_rows = 1280; gu.w_cols = N; gu.expert_bytes = GU_BYTES;
        gu.n = ngx; gu.xq = b.xq.p; gu.bounds = b.bounds.as<int32_t>() + j0; gu.ids = b.ident.as<int32_t>();
        gu.total_rows = rows; gu.max_rows = maxr; gu.dst = b.gu.as<float>(); gu.ld_dst = 1280;
        ctx.run(gu, s);
        mmq::swiglu(b.gu.as<float>() + r0 * 1280, b.h.as<float>() + r0 * FF, nr, FF, false, s);
        mmq::quantize(b.h.as<float>() + r0 * FF, nullptr, b.hq.p, Q2, FF, FF, nr, s);
        mmq::Product dn;
        dn.w = b.grp_d.p; dn.type = Q2; dn.w_rows = N; dn.w_cols = FF; dn.expert_bytes = D_BYTES;
        dn.n = ngx; dn.xq = b.hq.p; dn.bounds = b.bounds.as<int32_t>() + p.n + 1 + gi * (GROUP + 1);
        dn.ids = b.ident.as<int32_t>(); dn.total_rows = nr; dn.max_rows = maxr; dn.dst = b.dm.as<float>() + r0 * N;
        dn.ld_dst = N;
        ctx.run(dn, s);
    }
}

// the candidate: the same gather and bounds, then fp16tc::gu / swiglu / quantize / fp16tc::down
void run_fp16tc(Bufs& b, const Plan& p, const Routing& r, const float* x_dev, const int32_t* src_dev,
                const std::vector<const uint8_t*>& blob, int64_t rows, cudaStream_t s) {
    ck(cudaMemcpyAsync(b.bounds.p, p.bh.data(), p.bh.size() * 4, cudaMemcpyHostToDevice, s), "bounds");
    mmq::iota(b.ident.as<int32_t>(), rows, s);
    mmq::quantize(x_dev, src_dev, b.xq.p, Q2, N, N, rows, s);
    fp16tc::Geom g{};
    g.n_embd = N; g.n_ff = FF; g.gu_row = GU_ROW; g.d_row = D_ROW; g.up_off = UP_OFF;
    for (size_t j = 0; j < p.n; ++j) {
        const size_t q = j % GROUP;
        mmq::gather_native(blob[(size_t) p.order[j]], blob[(size_t) p.order[j]] + UP_OFF, UP_OFF,
                           blob[(size_t) p.order[j]] + O_DOWN, D_BYTES, b.grp_gu.as<uint8_t>() + q * GU_BYTES,
                           b.grp_d.as<uint8_t>() + q * D_BYTES, s);
        if (q + 1 < GROUP && j + 1 < p.n) continue;
        const size_t j0 = j - q, gi = j0 / GROUP;
        const int ngx = (int) (q + 1);
        const int64_t r0 = p.bh[j0], nr = p.bh[j + 1] - r0;
        int64_t maxr = 0;
        for (size_t i = j0; i <= j; ++i) maxr = std::max<int64_t>(maxr, r.cnt[(size_t) p.order[i]]);
        fp16tc::Batch bt;
        bt.n = ngx;
        bt.max_rows = (int) maxr;
        for (int i = 0; i < ngx; ++i) {
            bt.blob[i] = b.grp_gu.as<uint8_t>() + (size_t) i * GU_BYTES;
            bt.down[i] = b.grp_d.as<uint8_t>() + (size_t) i * D_BYTES;
        }
        fp16tc::gu(bt, g, b.bounds.as<int32_t>() + j0, b.xq.p, rows, b.gu.as<float>(), 1280, s);
        mmq::swiglu(b.gu.as<float>() + r0 * 1280, b.h.as<float>() + r0 * FF, nr, FF, false, s);
        mmq::quantize(b.h.as<float>() + r0 * FF, nullptr, b.hq.p, Q2, FF, FF, nr, s);
        fp16tc::down(bt, g, b.bounds.as<int32_t>() + p.n + 1 + gi * (GROUP + 1), b.hq.p, nr, b.dm.as<float>(), N, r0,
                     s);
    }
}

struct Err {
    double rms = 0, worst = 0;   // RMS error / RMS of the reference; the worst row's max |error| / its max |ref|
};
struct Stats {
    size_t nonfinite = 0;
    double absmax = 0, mn = 0, mx = 0;
};
Stats stats(const std::vector<float>& v) {
    Stats s;
    bool first = true;
    for (float x : v) {
        if (!std::isfinite(x)) { ++s.nonfinite; continue; }
        s.absmax = std::max(s.absmax, std::fabs((double) x));
        if (first) { s.mn = s.mx = x; first = false; }
        else { s.mn = std::min(s.mn, (double) x); s.mx = std::max(s.mx, (double) x); }
    }
    return s;
}
// A buffer's finiteness and range: a non-finite value names the buffer instead of only failing a later comparison.
void report(const char* what, const std::vector<float>& v) {
    const Stats s = stats(v);
    std::printf("  %-12s nonfinite %zu  absmax %.4e  range [%.4e, %.4e]\n", what, s.nonfinite, s.absmax, s.mn, s.mx);
}

// y_a[row_a(i)] against y_b[row_b(i)] for every pair i, `width` values per row
template <typename RA, typename RB>
Err compare(const std::vector<float>& a, RA row_a, const std::vector<float>& b, RB row_b, size_t pairs,
            int width = N) {
    double e2 = 0, r2 = 0;
    Err out;
    for (size_t i = 0; i < pairs; ++i) {
        const float* ya = a.data() + (size_t) row_a(i) * width;
        const float* yb = b.data() + (size_t) row_b(i) * width;
        double me = 0, mr = 0;
        for (int o = 0; o < width; ++o) {
            if (!std::isfinite(ya[o])) throw std::runtime_error("a non-finite output");
            const double d = (double) ya[o] - yb[o];
            e2 += d * d;
            r2 += (double) yb[o] * yb[o];
            me = std::max(me, std::fabs(d));
            mr = std::max(mr, (double) std::fabs(yb[o]));
        }
        if (mr > 0) out.worst = std::max(out.worst, me / mr);
        else if (me > 0) out.worst = 1e30;   // a zero row must stay exactly zero
    }
    out.rms = r2 > 0 ? std::sqrt(e2 / r2) : 0;
    return out;
}

std::vector<float> download(const Dev& d, size_t n) {
    std::vector<float> h(n);
    ck(cudaMemcpy(h.data(), d.p, n * 4, cudaMemcpyDeviceToHost), "download");
    return h;
}

// The compact-group row bases (mode 3): a sub-group of one group is run on its own - GU with gu_dst_row_base =
// -absolute_start so its outputs land at row 0 of its own buffer, down with down_act_row_base = -group_relative_start
// so it reads its own q8_1 from row 0 - and every row must reproduce the full-group run.  This is what catches a row
// base that is wrong when the sub-group does not start at the group's first row.
//
// Every live group is exercised, and the run is only accepted when a sub-group of a group whose absolute start is
// nonzero was covered: the first group starts at row 0, where an absolute row base and a group-relative one are the
// same number, so a caller that confused the two would still pass there.  A silent skip - no sub-group at all, or no
// nonzero-absolute group - is a failure, not a pass.
void compact_part(cudaStream_t s, const Routing& r, const Plan& p, const std::vector<const uint8_t*>& blob,
                  const void* xq_dev, int64_t rows, const Bufs& ref, int cap) {
    fp16tc::Geom g{};
    g.n_embd = N; g.n_ff = FF; g.gu_row = GU_ROW; g.d_row = D_ROW; g.up_off = UP_OFF;
    Dev grp_gu(GROUP * GU_BYTES + 4096), grp_d(GROUP * D_BYTES + 4096);
    int subgroups = 0, abs_groups = 0, rel_subgroups = 0;
    double worst_gu = 0, worst_h = 0, worst_dm = 0;
    for (size_t gi = 0; gi < p.ng; ++gi) {
        const size_t j0 = gi * GROUP, n = std::min<size_t>(GROUP, p.n - j0);
        if (n < 3 || p.bh[j0 + n] <= p.bh[j0]) continue;   // one expert cannot split; an empty group has no rows
        const int64_t grel0 = p.bh[j0];   // the group's absolute row start: 0 only for the first group
        if (grel0 != 0) ++abs_groups;
        for (size_t s0 = 0; s0 < n;) {
            size_t s1 = s0;
            while (s1 < n && p.bh[j0 + s1 + 1] - p.bh[j0 + s0] <= (int64_t) cap) ++s1;
            if (s1 == s0) s1 = s0 + 1;
            const int nsub = (int) (s1 - s0);
            const int64_t abs0 = p.bh[j0 + s0], rel0 = p.bh[j0 + s0] - grel0;
            if (rel0 != 0) ++rel_subgroups;
            const int64_t nr = p.bh[j0 + s1] - abs0;
            int64_t maxr = 0;
            for (size_t i = s0; i < s1; ++i) maxr = std::max<int64_t>(maxr, r.cnt[(size_t) p.order[j0 + i]]);
            for (int i = 0; i < nsub; ++i) {
                const size_t e = (size_t) p.order[j0 + s0 + (size_t) i];
                mmq::gather_native(blob[e], blob[e] + UP_OFF, UP_OFF, blob[e] + O_DOWN, D_BYTES,
                                   grp_gu.as<uint8_t>() + (size_t) i * GU_BYTES,
                                   grp_d.as<uint8_t>() + (size_t) i * D_BYTES, s);
            }
            ck(cudaMemsetAsync(grp_gu.as<uint8_t>() + (size_t) nsub * GU_BYTES, 0, 4096, s), "tail");
            ck(cudaMemsetAsync(grp_d.as<uint8_t>() + (size_t) nsub * D_BYTES, 0, 4096, s), "tail");
            std::vector<int32_t> bh((size_t) (2 * (nsub + 1)), 0);
            for (int i = 0; i <= nsub; ++i) {
                bh[(size_t) i] = (int32_t) p.bh[j0 + s0 + (size_t) i];                        // GU: absolute rows
                bh[(size_t) (nsub + 1 + i)] = (int32_t) (p.bh[j0 + s0 + (size_t) i] - grel0);  // down: group-relative
            }
            Dev bounds(bh.size() * 4), hq(mmq::q8_bytes(nr, FF));
            ck(cudaMemcpyAsync(bounds.p, bh.data(), bh.size() * 4, cudaMemcpyHostToDevice, s), "compact bounds");
            Dev gu_sub((size_t) nr * 1280 * 4), h_sub((size_t) nr * FF * 4);
            Dev dm_sub((size_t) (rel0 + nr) * N * 4);   // the sub-group's down writes at its group-relative rows
            fp16tc::Batch bt;
            bt.n = nsub;
            bt.max_rows = (int) maxr;
            bt.gu_dst_row_base = -abs0;     // the sub-group's GU rows land at row 0 of gu_sub
            bt.down_act_row_base = -rel0;   // and its down reads hq from row 0
            for (int i = 0; i < nsub; ++i) {
                bt.blob[i] = grp_gu.as<uint8_t>() + (size_t) i * GU_BYTES;
                bt.down[i] = grp_d.as<uint8_t>() + (size_t) i * D_BYTES;
            }
            fp16tc::gu(bt, g, bounds.as<int32_t>(), xq_dev, rows, gu_sub.as<float>(), 1280, s);
            mmq::swiglu(gu_sub.as<float>(), h_sub.as<float>(), nr, FF, false, s);
            mmq::quantize(h_sub.as<float>(), nullptr, hq.p, Q2, FF, FF, nr, s);
            fp16tc::down(bt, g, bounds.as<int32_t>() + nsub + 1, hq.p, nr, dm_sub.as<float>(), N, 0, s);
            ck(cudaStreamSynchronize(s), "compact");
            auto slice = [](const Dev& d, size_t elems) {
                std::vector<float> v(elems);
                ck(cudaMemcpy(v.data(), d.p, elems * 4, cudaMemcpyDeviceToHost), "compact download");
                return v;
            };
            auto ref_rows = [](const Dev& d, int64_t row0, int64_t nrows, int width) {
                std::vector<float> v((size_t) nrows * width);
                ck(cudaMemcpy(v.data(), (const float*) d.p + (size_t) row0 * width, v.size() * 4,
                              cudaMemcpyDeviceToHost), "reference download");
                return v;
            };
            const std::vector<float> gu_s = slice(gu_sub, (size_t) nr * 1280);
            const std::vector<float> h_s = slice(h_sub, (size_t) nr * FF);
            const std::vector<float> dm_s = slice(dm_sub, (size_t) (rel0 + nr) * N);
            const std::vector<float> r_gu = ref_rows(ref.gu, abs0, nr, 1280);
            const std::vector<float> r_h = ref_rows(ref.h, abs0, nr, FF);
            const std::vector<float> r_dm = ref_rows(ref.dm, grel0 + rel0, nr, N);
            auto ident = [](size_t i) { return (int64_t) i; };
            auto shift = [&](size_t i) { return rel0 + (int64_t) i; };
            const Err egu = compare(gu_s, ident, r_gu, ident, (size_t) nr, 1280);
            const Err eh = compare(h_s, ident, r_h, ident, (size_t) nr, FF);
            const Err ed = compare(dm_s, shift, r_dm, ident, (size_t) nr, N);
            worst_gu = std::max(worst_gu, egu.rms);
            worst_h = std::max(worst_h, eh.rms);
            worst_dm = std::max(worst_dm, ed.rms);
            std::printf("  compact group %zu sub-group %d: experts %d, rows %lld (abs %lld, group-rel %lld)"
                        " GU RMS %.3e  H %.3e  Dm %.3e\n",
                        gi, subgroups, nsub, (long long) nr, (long long) abs0, (long long) rel0, egu.rms, eh.rms,
                        ed.rms);
            ++subgroups;
            s0 = s1;
        }
    }
    if (subgroups < 1) throw std::runtime_error("the compact-group row bases were not exercised");
    if (abs_groups < 1) throw std::runtime_error("no compact sub-group had a nonzero group-absolute start");
    if (rel_subgroups < 1) throw std::runtime_error("no compact sub-group had a nonzero group-relative start");
    if (worst_gu > 1e-6 || worst_h > 1e-6 || worst_dm > 1e-6)
        throw std::runtime_error("the compact row bases do not reproduce the full-group rows");
}

// part 1: the two paths against the double-precision reference, then the compact-group row bases
void reference_part(cudaStream_t s) {
    constexpr int T = 256, E = 64, ZERO = 77;
    std::mt19937 rng(136);
    std::vector<std::vector<uint8_t>> host((size_t) E);
    std::vector<std::unique_ptr<Dev>> dev;
    std::vector<const uint8_t*> blob((size_t) E);
    for (int e = 0; e < E; ++e) {
        host[(size_t) e] = make_blob(rng);
        dev.push_back(std::make_unique<Dev>(BLOB + 4096 * (size_t) (e % 3)));   // unrelated addresses
        uint8_t* p = dev.back()->as<uint8_t>() + 256 * (size_t) (e % 3);       // 16-byte aligned, not 512
        ck(cudaMemcpy(p, host[(size_t) e].data(), BLOB, cudaMemcpyHostToDevice), "blob");
        blob[(size_t) e] = p;
    }
    const std::vector<float> x = make_x(T, rng, ZERO);
    const std::vector<int32_t> ids = make_ids(T, E, rng, 2, 4);   // experts 0, 1 hot (several tiles), 60-63 unused
    const int64_t rows = (int64_t) T * K;
    const Routing r = sort_rows(ids, E);
    const Plan p = plan_groups(r, rows);
    std::printf("reference part: %d tokens, %d experts, rows per expert: max %d, experts without rows %d\n", T, E,
                *std::max_element(r.cnt.begin(), r.cnt.end()), (int) std::count(r.cnt.begin(), r.cnt.end(), 0));

    Dev x_dev(x.size() * 4), src_dev((size_t) rows * 4);
    ck(cudaMemcpy(x_dev.p, x.data(), x.size() * 4, cudaMemcpyHostToDevice), "x");
    ck(cudaMemcpy(src_dev.p, r.src.data(), r.src.size() * 4, cudaMemcpyHostToDevice), "src");
    mmq::Context ctx;
    Bufs mb(rows, E), tb(rows, E);
    ck(cudaMemset(tb.gu.p, 0xff, (size_t) rows * 1280 * 4), "sentinel gu");   // an unwritten row is NaN
    ck(cudaMemset(tb.dm.p, 0xff, (size_t) rows * N * 4), "sentinel dm");
    // the uploads (legacy stream) must land before the non-blocking stream reads them, or the kernels could run on
    // all-zero routing (every pair on one expert: fast and wrong)
    ck(cudaDeviceSynchronize(), "uploads");
    run_mmq(ctx, mb, p, r, x_dev.as<float>(), src_dev.as<int32_t>(), blob, rows, s);
    run_fp16tc(tb, p, r, x_dev.as<float>(), src_dev.as<int32_t>(), blob, rows, s);
    ck(cudaStreamSynchronize(s), "sync");
    const std::vector<float> y_mmq = download(mb.dm, (size_t) rows * N), y_tc = download(tb.dm, (size_t) rows * N);
    const std::vector<float> gu_mmq = download(mb.gu, (size_t) rows * 1280), gu_tc = download(tb.gu, (size_t) rows * 1280);
    const std::vector<float> h_mmq = download(mb.h, (size_t) rows * FF), h_tc = download(tb.h, (size_t) rows * FF);
    report("MMQ GU", gu_mmq); report("FP16TC GU", gu_tc);
    report("MMQ H", h_mmq); report("FP16TC H", h_tc);
    report("MMQ Dm", y_mmq); report("FP16TC Dm", y_tc);
    {
        auto ident = [](size_t i) { return (int64_t) i; };
        const Err egu = compare(gu_tc, ident, gu_mmq, ident, (size_t) rows, 1280);
        const Err eh = compare(h_tc, ident, h_mmq, ident, (size_t) rows, FF);
        std::printf("  FP16TC vs MMQ intermediates: GU rel RMS %.3e  H rel RMS %.3e\n", egu.rms, eh.rms);
    }

    // the reference, an expert at a time on all cores
    std::vector<float> ref((size_t) rows * N);
    std::vector<std::thread> th;
    const int nth = std::max(1, std::min(16, (int) std::thread::hardware_concurrency()));
    for (int w = 0; w < nth; ++w)
        th.emplace_back([&, w] {
            std::vector<float> gu, dn, h(FF);
            for (int e = w; e < E; e += nth) {
                if (r.cnt[(size_t) e] == 0) continue;
                dequant(host[(size_t) e], gu, dn);
                for (int64_t i = 0; i < rows; ++i) {
                    if (ids[(size_t) i] != e) continue;
                    const float* xr = x.data() + (size_t) (i / K) * N;
                    for (int f = 0; f < FF; ++f) {
                        double gt = 0, up = 0;
                        const float* wg = gu.data() + (size_t) f * N;
                        const float* wu = gu.data() + (size_t) (FF + f) * N;
                        for (int k = 0; k < N; ++k) { gt += (double) wg[k] * xr[k]; up += (double) wu[k] * xr[k]; }
                        h[(size_t) f] = (float) (gt / (1.0 + std::exp(-gt)) * up);
                    }
                    for (int o = 0; o < N; ++o) {
                        double a = 0;
                        const float* wd = dn.data() + (size_t) o * FF;
                        for (int f = 0; f < FF; ++f) a += (double) wd[f] * h[(size_t) f];
                        ref[(size_t) i * N + o] = (float) a;
                    }
                }
            }
        });
    for (auto& t : th) t.join();

    auto pair = [](size_t i) { return (int64_t) i; };
    auto srow = [&](size_t i) { return (int64_t) r.row_of[i]; };
    const Err em = compare(y_mmq, srow, ref, pair, (size_t) rows);
    const Err et = compare(y_tc, srow, ref, pair, (size_t) rows);
    const Err etm = compare(y_tc, srow, y_mmq, srow, (size_t) rows);
    std::printf("  MMQ    vs FP32 reference: rel RMS %.3e  worst row max rel %.3e\n", em.rms, em.worst);
    std::printf("  FP16TC vs FP32 reference: rel RMS %.3e  worst row max rel %.3e\n", et.rms, et.worst);
    std::printf("  FP16TC vs MMQ           : rel RMS %.3e  worst row max rel %.3e\n", etm.rms, etm.worst);
    double zmax = 0;
    for (int k = 0; k < K; ++k)
        for (int o = 0; o < N; ++o)
            zmax = std::max(zmax, (double) std::fabs(y_tc[(size_t) r.row_of[(size_t) (ZERO * K + k)] * N + o]));
    if (zmax != 0) throw std::runtime_error("the all-zero token's outputs are not zero");
    if (et.rms > 1.5 * em.rms || et.worst > 2.0 * em.worst)
        throw std::runtime_error("the FP16TC path's error is not comparable to MMQ's");

    // part 2: the compact row bases must reproduce the full-group rows, on every group
    compact_part(s, r, p, blob, tb.xq.p, rows, tb, 256);
}

}  // namespace

int main() {
    try {
        int n = 0;
        if (cudaGetDeviceCount(&n) != cudaSuccess || n == 0) { std::printf("no CUDA device: skipped\n"); return 77; }
        if (!mmq::built()) { std::printf("no MMQ in this build\n"); return 1; }
        int ran = 0;
        for (int d = 0; d < n; ++d) {
            ck(cudaSetDevice(d), "set device");
            if (!fp16tc::available()) { std::printf("device %d is not Volta: skipped\n", d); continue; }
            cudaStream_t s = nullptr;
            ck(cudaStreamCreateWithFlags(&s, cudaStreamNonBlocking), "stream");
            std::printf("device %d:\n", d);
            reference_part(s);
            ck(cudaStreamDestroy(s), "destroy");
            ++ran;
        }
        if (ran == 0) { std::printf("no Volta device: skipped\n"); return 77; }
        std::printf("prefill fp16tc parity passed\n");
        return 0;
    } catch (const std::exception& e) {
        std::fprintf(stderr, "prefill fp16tc parity failed: %s\n", e.what());
        return 1;
    }
}
