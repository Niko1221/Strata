#include "ink_sy_codecs.h"
#include "ink_sy_ops.h"
#include "ink_sy_xmx.h"
#include <random>
#include <stdexcept>
using namespace ink_sycl;

template <class T> struct Array {
    T* p;
    explicit Array(size_t n) : p(sycl::malloc_shared<T>(n, Q())) {
        if (!p)
            throw std::bad_alloc();
    }
    ~Array() { sycl::free(p, Q()); }
};
static void equal(const char* kind, const float* a, const float* b, size_t n) {
    for (size_t i = 0; i < n; ++i)
        if (std::memcmp(a + i, b + i, sizeof(float))) {
            std::fprintf(stderr, "%s index=%zu base=%a candidate=%a\n", kind, i, a[i], b[i]);
            throw std::runtime_error(kind);
        }
}
static void routing_and_tail_check() {
    Array<float> logits(128), bias(128), ref(8), weights(8);
    Array<int> a(8), b(8);
    for (int test = 0; test < 5; ++test) {
        for (int e = 0; e < 128; ++e) {
            logits.p[e] = test == 0 ? 0.f : (e % 19 - 9) * 0.3125f;
            bias.p[e] = test == 0 ? 0.f : (e % 7 - 3) * 0.0625f;
            if (test == 2 && e % 4 == 0)
                logits.p[e] = NAN;
            if (test == 3 && e % 5 == 0)
                bias.p[e] = INFINITY;
            if (test == 4)
                bias.p[e] = e == 1 ? 0.f : -1e31f;
        }
        k_router_topk8_sg16(logits.p, bias.p, a.p, ref.p, 1.375f, 1);
        k_sq20_decode_top8(logits.p, bias.p, b.p, weights.p, 1.375f);
        wait();
        if (std::memcmp(a.p, b.p, 8 * sizeof(int)))
            throw std::runtime_error("parallel top8 IDs");
        equal("parallel top8 weights", ref.p, weights.p, 8);
    }
    constexpr int d = 2304;
    Array<float> y(8 * d), raw(d), conv(d), w(4 * d), residual(d), ref_out(d), out(d), h0(6 * d), h1(6 * d);
    Array<int> slot(1), offset(1), count(1), row(1);
    slot.p[0] = count.p[0] = 1;
    offset.p[0] = row.p[0] = 0;
    for (int i = 0; i < 8 * d; i++)
        y.p[i] = (i % 23 - 12) * .0173f;
    for (int i = 0; i < 4 * d; i++)
        w.p[i] = (i % 11 - 5) * .31f;
    for (int i = 0; i < d; i++)
        residual.p[i] = (i % 7 - 3) * .07f;
    for (int i = 0; i < 6 * d; i++)
        h1.p[i] = h0.p[i] = (i % 17 - 8) * .051f;
    for (int i = 0; i < 3; i++) {
        k_moe_reduce8_sg16(y.p, raw.p, 1);
        k_sconv4_multi_sg16(raw.p, w.p, nullptr, h0.p, slot.p, offset.p, count.p, row.p, conv.p, 1, d, 0);
        k_add_sg16(residual.p, conv.p, ref_out.p, d);
        k_sq20_decode_tail(y.p, w.p, h1.p, slot.p, residual.p, out.p);
        wait();
        equal("single-token residual", ref_out.p, out.p, d);
        equal("ShortConv history", h0.p, h1.p, 6 * d);
    }
}

#ifdef INK_SYCL_USE_ONEMKL
static void expert_check(const char* model) {
    strata::artifact::StrataFileReader f;
    std::string error;
    if (!f.open(model, error))
        throw std::runtime_error(error);
    const auto& entries = f.expert_directory();
    Array<InkSyExpertDev> experts(128);
    std::vector<uint8_t*> allocated;
    int n4 = 0, n8 = 0;
    for (size_t i = 0; i < entries.size(); ++i) {
        const auto& en = entries[i];
        if (en.gu_code_id != en.dn_code_id)
            continue;
        bool sf4 = en.gu_code_id == FMT_SF4 && n4 < 4, mx8 = en.gu_code_id == FMT_MX8 && n8 < 4;
        if (!sf4 && !mx8)
            continue;
        std::vector<uint8_t> body;
        if (!f.read_expert_body(i, body, error))
            throw std::runtime_error(error);
        uint8_t* dev = sycl::malloc_device<uint8_t>(body.size(), Q());
        if (!dev)
            throw std::bad_alloc();
        Q().memcpy(dev, body.data(), body.size()).wait_and_throw();
        const int id = sf4 ? n4++ : 4 + n8++;
        experts.p[id] = {dev, en.dn_offset, en.gu_code_id, en.dn_code_id, 1, 0};
        allocated.push_back(dev);
        if (n4 + n8 == 8)
            break;
    }
    if (n4 != 4 || n8 != 4)
        throw std::runtime_error("missing pack formats");
    for (int e = 0; e < n4; ++e) {
        constexpr size_t gu = 2304ul * 2304, dn = 2304ul * 1152;
        Array<half> ref(gu + dn), got(gu + dn);
        k_dequant_moe_gu_f16_sg16(experts.p[e], ref.p);
        k_dequant_moe_dn_f16_sg16(experts.p[e], ref.p + gu);
        k_unpack_sf4_expert_panels(experts.p[e], got.p);
        wait();
        if (std::memcmp(ref.p, got.p, (gu + dn) * sizeof(half)) != 0)
            throw std::runtime_error("combined SF4 unpack differs from separate panels");
    }
    std::puts("PASS bitwise: all FP16 gate/up and down weights from four real SF4 experts");
    Array<int> idx(8), ue(128), off(128), cnt(128), pairs(128), nu(1);
    Array<float> weights(128), x(16 * 2304), h(8 * 1152), h_ref(8 * 1152), y(8 * 2304), y_ref(8 * 2304);
    for (int i = 0; i < 16 * 2304; ++i)
        x.p[i] = float((i * 37) % 101 - 50) * .001f;
    for (int i = 0; i < 128; ++i)
        weights.p[i] = float(i % 8 + 1) * .03125f;
    for (int i = 0; i < 8; ++i)
        idx.p[i] = (7 - i) * 3 % 8;
    k_sq20_expert_gemv<false>(experts.p, idx.p, weights.p, x.p, h.p);
    k_sq20_expert_gemv<true>(experts.p, idx.p, weights.p, h.p, y.p);
    k_moe_plan_sg16(idx.p, 8, ue.p, off.p, cnt.p, pairs.p, nu.p);
    k_moe_gu_situ_sg16(experts.p, x.p, ue.p, off.p, cnt.p, pairs.p, nu.p, 8, h_ref.p);
    k_moe_dn_sg16(experts.p, weights.p, ue.p, off.p, cnt.p, pairs.p, nu.p, 8, h_ref.p, y_ref.p);
    wait();
    equal("packed dot GU", h_ref.p, h.p, 8 * 1152);
    equal("packed dot down", y_ref.p, y.p, 8 * 2304);
    for (int i = 0; i < 8; ++i)
        idx.p[i] = (7 - i) % 4;
    k_moe_plan_sg16(idx.p, 8, ue.p, off.p, cnt.p, pairs.p, nu.p);
    k_moe_gu_situ_sg16(experts.p, x.p, ue.p, off.p, cnt.p, pairs.p, nu.p, 8, h_ref.p);
    k_sq20_sf4_pair<false>(experts.p, idx.p, nullptr, x.p, h.p);
    k_moe_dn_sg16(experts.p, weights.p, ue.p, off.p, cnt.p, pairs.p, nu.p, 8, h_ref.p, y_ref.p);
    k_sq20_sf4_pair<true>(experts.p, idx.p, weights.p, h.p, y.p);
    wait();
    equal("paired SF4 decode gate/up", h_ref.p, h.p, 8 * 1152);
    equal("paired SF4 decode down", y_ref.p, y.p, 8 * 2304);
    for (int n = 1; n <= 12; ++n) {
        const int rows = 8 * n;
        Array<float> gold_h(rows * 1152), test_h(rows * 1152), gold_y(rows * 2304), test_y(rows * 2304);
        nu.p[0] = 8;
        for (int e = 0; e < 8; ++e) {
            ue.p[e] = e;
            off.p[e] = e * n;
            cnt.p[e] = n;
            for (int j = 0; j < n; ++j)
                pairs.p[e * n + j] = j * 8 + e;
        }
        k_moe_gu_situ_sg16(experts.p, x.p, ue.p, off.p, cnt.p, pairs.p, nu.p, 8, gold_h.p);
        k_moe_dn_sg16(experts.p, weights.p, ue.p, off.p, cnt.p, pairs.p, nu.p, 8, gold_h.p, gold_y.p);
        sq20_small_moe(experts.p, ue.p, off.p, cnt.p, pairs.p, nu.p, 8, weights.p, x.p, test_h.p, test_y.p);
        wait();
        equal("SQ-20 small GU", gold_h.p, test_h.p, rows * 1152);
        equal("SQ-20 small down", gold_y.p, test_y.p, rows * 2304);
    }
    // More than sixteen distinct experts forces another pass through the
    // bounded dispatch grid. A twelve-row expert also crosses the route tile.
    constexpr int unique = 33, active_rows = 12, routed = active_rows * 8;
    Array<InkSyExpertDev> aliases(unique);
    Array<float> factors(unique * sf4_expert_scales), ref_h(routed * 1152), got_h(routed * 1152),
                 ref_y(routed * 2304), got_y(routed * 2304);
    for (int i = 0; i < unique; ++i) aliases.p[i] = experts.p[i % 4];
    k_sf4_build_scales(aliases.p, unique, factors.p);
    nu.p[0] = unique;
    int n_used = 0;
    for (int i = 0; i < unique; ++i) {
        ue.p[i] = (i * 7) % unique;
        off.p[i] = n_used;
        cnt.p[i] = i == 0 ? 12 : 1 + i % 4;
        for (int j = 0; j < cnt.p[i]; ++j) {
            pairs.p[n_used] = (n_used * 5) % routed;
            ++n_used;
        }
    }
    if (n_used > routed) throw std::runtime_error("invalid SF4 fixture");
    std::fill_n(ref_h.p, routed * 1152, -99.f); std::fill_n(got_h.p, routed * 1152, -99.f);
    std::fill_n(ref_y.p, routed * 2304, -99.f); std::fill_n(got_y.p, routed * 2304, -99.f);
    k_moe_gu_situ_sg16(aliases.p, x.p, ue.p, off.p, cnt.p, pairs.p, nu.p, routed, ref_h.p);
    k_moe_dn_sg16(aliases.p, weights.p, ue.p, off.p, cnt.p, pairs.p, nu.p, routed, ref_h.p, ref_y.p);
    k_sf4_cached<false>(aliases.p, ue.p, off.p, cnt.p, pairs.p, nu.p, routed, nullptr, x.p, got_h.p, factors.p);
    k_sf4_cached<true>(aliases.p, ue.p, off.p, cnt.p, pairs.p, nu.p, routed, weights.p, got_h.p, got_y.p, factors.p);
    wait();
    equal("cached SF4 gate/up and unassigned rows", ref_h.p, got_h.p, routed * 1152);
    equal("cached SF4 down and unassigned rows", ref_y.p, got_y.p, routed * 2304);
    // The packed route moves only rows. Compare its complete projected and
    // scattered output to main's one-expert-at-a-time path on the same routes.
    constexpr std::array<int, 8> counts{12, 15, 16, 17, 32, 33, 65, 96};
    constexpr int pairs_count = 286, tokens = (pairs_count + 7) / 8;
    Array<int> order(pairs_count);
    Array<float> xx(tokens * 2304), rw(pairs_count), separate(pairs_count * 2304),
        panel(pairs_count * 2304), packed_out(pairs_count * 2304);
    Array<half> packed_x(pairs_count * 2304);
    std::mt19937 rng(17);
    for (int i = 0; i < tokens * 2304; ++i) xx.p[i] = float(int(rng() % 20001) - 10000) / 8192.f;
    for (int i = 0; i < pairs_count; ++i) { order.p[i] = i; rw.p[i] = float(1 + i % 8) / 36.f; }
    std::shuffle(order.p, order.p + pairs_count, rng);
    k_moe_gather_f16_sg16(xx.p, order.p, 0, pairs_count, packed_x.p);
    int start = 0;
    for (int i = 0; i < 8; ++i) {
        ink_xmx_moe_single(experts.p[i], xx.p, rw.p, order.p, start, counts[i], separate.p);
        ink_xmx_moe_project(experts.p[i], packed_x.p + start * 2304, counts[i], panel.p + start * 2304, true);
        start += counts[i];
    }
    k_moe_scatter_w_sg16(panel.p, rw.p, order.p, 0, pairs_count, packed_out.p);
    wait();
    equal("packed oneMKL routed output", separate.p, packed_out.p, pairs_count * 2304);
    // Share an expansion across the same expert's independent tile shapes.
    for (int expert : {0, 4}) {
        int offset = 0;
        for (int rows : {12, 33, 65}) {
            ink_xmx_moe_project(experts.p[expert], packed_x.p + offset * 2304,
                                rows, separate.p + offset * 2304, true);
            offset += rows;
        }
        ink_xmx_expand_expert(experts.p[expert]);
        offset = 0;
        for (int rows : {12, 33, 65}) {
            ink_xmx_moe_project(experts.p[expert], packed_x.p + offset * 2304,
                                rows, panel.p + offset * 2304, false, true);
            offset += rows;
        }
        wait();
        equal("shared expert expansion across prefill tiles", separate.p, panel.p, offset * 2304);
    }
    std::puts("PASS bitwise: shared expert expansion across prefill tiles");
    std::puts("PASS bitwise: paired SF4 single-token gate/up and down vs generic compressed dot");
    for (auto* v : allocated)
        sycl::free(v, Q());
    std::puts("PASS bitwise: small SF4/MX8 experts and packed oneMKL routed output");

}
#endif

// Optional model argument enables the tests with real SF4 and MX8 weights.
int main(int argc, char** argv) {
    if (argc > 2) {
        std::fprintf(stderr, "usage: %s [SQ-20.strata]\n", argv[0]);
        return 2;
    }
    if (!ink_battlemage()) {
        std::fprintf(stderr, "This test requires a Battlemage GPU.\n");
        return 77;
    }
    std::mt19937 rng(42);
    auto draw = [&]() { return float(int(rng() % 20001) - 10000) / 8192.f; };
    Array<float> mx8_ref(256), mx8_out(256);
    float* expected = mx8_ref.p;
    float* actual = mx8_out.p;
    Q().parallel_for(sycl::range<1>(256), [=](sycl::id<1> id) {
        const auto code = uint32_t(id[0]);
        expected[code] = e4m3_to_f32(code);
        actual[code] = mx8_coefficient_f32(code);
    });
    wait();
    equal("all E4M3 encodings", expected, actual, 256);
    std::puts("PASS bitwise: all E4M3 encodings");
    {
        constexpr int rows = 11, pairs = rows * 8, dim = 2304;
        Array<float> packed(pairs * dim), small(pairs * dim), weight(pairs),
                     scattered(pairs * dim), original(rows * dim), fused(rows * dim);
        Array<int> order(pairs), inverse(pairs);
        for (int i = 0; i < pairs * dim; ++i) {
            packed.p[i] = draw();
            small.p[i] = scattered.p[i] = draw();
        }
        int packed_rows = 0;
        for (int pid = 0; pid < pairs; ++pid) {
            weight.p[pid] = draw();
            inverse.p[pid] = -1;
        }
        for (int j = 0; j < pairs; ++j) {
            const int pid = (j * 23) % pairs;
            if (j % 3 == 0) continue;
            inverse.p[pid] = packed_rows;
            order.p[packed_rows++] = pid;
        }
        k_moe_scatter_w_sg16(packed.p, weight.p, order.p, 0, packed_rows, scattered.p);
        k_moe_reduce8_sg16(scattered.p, original.p, rows);
        k_moe_reduce_packed(packed.p, small.p, weight.p, inverse.p, fused.p, rows);
        wait();
        equal("packed expert reduction and separately rounded routing weights", original.p, fused.p, rows * dim);
    }
    std::puts("PASS bitwise: packed expert reduction");
    routing_and_tail_check();
    std::puts("PASS bitwise: top-8 IDs and weights, three consecutive outputs and both histories");
    if (argc == 2) {
#ifdef INK_SYCL_USE_ONEMKL
        expert_check(argv[1]);
#else
        std::fprintf(stderr, "SQ-20 comparison requires oneMKL.\n");
        return 77;
#endif
    }
}
