#include "ink_sy_codecs.h"
#include "ink_sy_controller.h"
#include "ink_sy_xmx.h"
#include "main_reference.h"
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
static void controller_batch_check() {
    constexpr int d = INK_D, rank = ctrl_rank, dr = ctrl_k * rank, gr = d * rank;
    Array<float> down(4 * dr), gain(4 * gr), forget(4 * gr), norm(4 * d);
    Array<float> original(3 * d), msg(3 * d), ref(3 * d), actual(3 * d), cat(3 * ctrl_k);
    Array<float> gates(3 * rank), want_gates(3 * rank);
    std::mt19937 rng(51);
    auto draw = [&] { return float(int(rng() % 20001) - 10000) / 15237.0f; };
    auto weight = [&] { return bf16_to_f32(sycl::bit_cast<uint32_t>(draw()) >> 16); };
    for (int i = 0; i < 4 * dr; ++i) down.p[i] = weight();
    for (int i = 0; i < 4 * gr; ++i) { gain.p[i] = weight(); forget.p[i] = weight(); }
    for (int i = 0; i < 4 * d; ++i) norm.p[i] = draw();
    for (int i = 0; i < 3 * d; ++i) { original.p[i] = draw(); msg.p[i] = draw(); }

    InkSyModel model;
    auto& c = model.man.c;
    c.n_phys = 2; c.core_start = 1; c.core_end = 2; c.is_swa = {0, 0};
    auto add = [&](const std::string& name, float* ptr, std::vector<int64_t> shape) {
        auto& tensor = model.man.T["model.llm.layers.1." + name];
        tensor.shape = std::move(shape);
        tensor.dev_ptr = reinterpret_cast<uint8_t*>(ptr);
        tensor.bytes = tensor.rows() * tensor.cols() * sizeof(float);
    };
    const char* names[] = {"attn", "k", "v", "mlp"};
    const char* norms[] = {"q_input_norm.weight", "k_input_norm.weight", "v_input_norm.weight", "mlp_norm.weight"};
    for (int j = 0; j < 4; ++j) {
        const std::string p = std::string("controllers.") + names[j] + ".";
        add(p + "down_weight", down.p + j * dr, {rank, ctrl_k});
        add(p + "gain_weight", gain.p + j * gr, {d, rank});
        add(p + "forget_weight", forget.p + j * gr, {d, rank});
        add(norms[j], norm.p + j * d, {d});
    }
    add("attn_norm.weight", norm.p, {d});
    sy_setenv("INK_SYCL_CTRL_BATCH", "1");
    sy_setenv("INK_SYCL_STREAM_GEMV", "1");
    InkControllerCache cache;
    for (int swa : {0, 1}) {
        c.is_swa[1] = swa;
        cache.prepare(model);
        if (cache.layers.size() != 1) throw std::runtime_error("expected exact controller cache");
        for (int count : {1, 3}) {
            std::copy_n(original.p, 3 * d, actual.p);
            std::copy_n(original.p, 3 * d, ref.p);
            if (count == 3 && swa) {
                std::copy_n(original.p, d, ref.p + d);
                std::copy_n(original.p, d, ref.p + 2 * d);
            }
            for (int j = 0; j < count; ++j) {
                const int bank = count == 1 ? 3 : j;
                float* row = ref.p + j * d;
                main_reference::k_ctrl_cat_rms_sg16(row, msg.p + j * d, cat.p, 1);
                main_reference::k_linear_f32_sg16(down.p + bank * dr, rank, ctrl_k, cat.p, want_gates.p + j * rank, 1);
                main_reference::k_ctrl_apply_norm_sg16(row, msg.p + j * d, want_gates.p + j * rank,
                    gain.p + bank * gr, forget.p + bank * gr, norm.p + (count == 3 && swa ? 0 : bank * d),
                    row, 1, 1);
            }
            const auto& layer = cache.layers[0];
            if (count == 3) k_ctrl_decode_batch(layer.attention, actual.p, msg.p, swa, cat.p, gates.p);
            else k_ctrl_decode_batch(layer.mlp, actual.p, msg.p, false, cat.p, gates.p);
            wait();
            equal("controller gate vector", want_gates.p, gates.p, count * rank);
            equal("in-place controller output and guard rows", ref.p, actual.p, 3 * d);
        }
        cache.release();
    }
    // A single inexact coefficient in even the last bank must reject the cache.
    for (float* tail : {down.p + 4 * dr - 1, gain.p + 4 * gr - 1, forget.p + 4 * gr - 1}) {
        const float saved = *tail;
        *tail = sycl::bit_cast<float>(sycl::bit_cast<uint32_t>(saved) | 1u);
        cache.prepare(model);
        if (cache.storage || !cache.layers.empty()) throw std::runtime_error("inexact controller accepted");
        *tail = saved;
    }
    auto& down_info = model.man.T.at("model.llm.layers.1.controllers.mlp.down_weight");
    down_info.fmt = FMT_BF16;
    cache.prepare(model);
    if (cache.storage || !cache.layers.empty()) throw std::runtime_error("unhandled controller format accepted");
    down_info.fmt = FMT_F32;
    sy_setenv("INK_SYCL_CTRL_BATCH", "0");
    cache.prepare(model);
    if (cache.storage || !cache.layers.empty()) throw std::runtime_error("controller override ignored");
    sy_setenv("INK_SYCL_STREAM_GEMV", "0");
}

static void single_request_ops_check() {
    constexpr int d = INK_D, max_rows = 512;
    std::mt19937 rng(97);
    auto draw = [&]() { return (int(rng() % 2001) - 1000) / 1531.0f; };
    Array<float> x(max_rows * d), w(4 * d), taps(max_rows * 32), y(max_rows * d), ref(max_rows * d);
    Array<float> hist(2 * 3 * d), ref_hist(2 * 3 * d);
    Array<int> slot(1), off(1), count(1), rows(max_rows);
    slot.p[0] = 1; off.p[0] = 0;
    for (int i = 0; i < max_rows; ++i) rows.p[i] = i;
    for (int i = 0; i < max_rows * d; ++i) x.p[i] = draw();
    for (int i = 0; i < 4 * d; ++i) w.p[i] = draw();
    for (int i = 0; i < max_rows * 32; ++i) taps.p[i] = draw();
    for (int i = 0; i < 6 * d; ++i) hist.p[i] = ref_hist.p[i] = draw();
    for (int c : {1024, d}) {
        for (int n : {32, 33, 127, 511, 512}) {
            count.p[0] = n;
            const float* dyn = c == 1024 ? taps.p : nullptr;
            main_reference::k_sconv4_multi_sg16(x.p, w.p, dyn, ref_hist.p, slot.p, off.p, count.p, rows.p, ref.p, 1, c, dyn != nullptr);
            k_sconv4_prefill(x.p, w.p, dyn, hist.p, slot.p, y.p, n, c);
            wait();
            equal("ShortConv output", ref.p, y.p, n * c);
            equal("ShortConv history, including inactive slot", ref_hist.p, hist.p, 6 * c);
        }
    }
    Array<float> cur(2 * d), msg(2 * d), gate(128), gain(d * 64), forget(d * 64), norm(d);
    for (int i = 0; i < d * 64; ++i) { gain.p[i] = draw(); forget.p[i] = draw(); }
    for (int i = 0; i < 128; ++i) gate.p[i] = draw();
    for (int i = 0; i < d; ++i) norm.p[i] = draw();
    for (int i = 0; i < 2 * d; ++i) msg.p[i] = draw();
    for (int normalize : {0, 1}) {
        for (int i = 0; i < 2 * d; ++i) cur.p[i] = draw();
        std::copy_n(cur.p, 2 * d, ref.p);
        // Two independent rows use the original fused kernel and reduction order.
        main_reference::k_ctrl_apply_norm_sg16(ref.p, msg.p, gate.p, gain.p, forget.p, norm.p, ref.p, 2, normalize);
        k_ctrl_apply_norm_sg16(cur.p, msg.p, gate.p, gain.p, forget.p, norm.p, cur.p, 1, normalize);
        wait();
        equal("in-place controller", ref.p, cur.p, d);
    }
    Array<float> blocks(7 * 2 * d), queries(4 * d), stats(64);
    for (int i = 0; i < 14 * d; ++i) blocks.p[i] = draw();
    for (int i = 0; i < 4 * d; ++i) queries.p[i] = draw();
    for (int n : {0, 1, 4, 7}) {
        for (int nq : {1, 3, 4}) {
            main_reference::k_attn_res_mix_sg16(blocks.p, cur.p, queries.p, ref.p, n, 2 * d, 1, nq);
            k_res_mix_decode(blocks.p, cur.p, queries.p, y.p, n, 2 * d, nq, stats.p);
            wait();
            equal("decode residual mix", ref.p, y.p, nq * d);
        }
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
    for (int n = 1; n <= 11; ++n) {
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
    // Compare the public entry points with their existing SG16 implementation.
    sy_setenv("INK_SYCL_STREAM_GEMV", "0", 1);
    if (!ink_battlemage()) {
        std::fprintf(stderr, "This test requires a Battlemage GPU.\n");
        return 77;
    }
    std::mt19937 rng(42);
    auto draw = [&]() { return float(int(rng() % 20001) - 10000) / 8192.f; };
    int nlinear = 0, nattn = 0;
    for (auto shape : {std::pair{24, 2304}, {32, 2304}, {64, 4608}, {128, 2304}, {17, 256}, {129, 512}}) {
        const auto [n, k] = shape;
        Array<float> w(n * k), x(k), a(n), b(n);
        Array<uint16_t> wb(n * k);
        for (int i = 0; i < n * k; ++i) {
            w.p[i] = draw();
            uint32_t bits;
            std::memcpy(&bits, w.p + i, 4);
            wb.p[i] = bits >> 16;
        }
        for (int i = 0; i < k; ++i)
            x.p[i] = draw();
#ifdef INK_HAS_GROUP_LOAD
        if (k % 128 == 0) {
            main_reference::k_linear_f32_sg16(w.p, n, k, x.p, a.p, 1);
            k_streaming_gemv<false>(w.p, n, k, x.p, b.p);
            wait();
            equal("FP32 GEMV", a.p, b.p, n);
            if (n % 8 == 0 && n <= 128) {
                constexpr int rows = 33;
                Array<float> xx(rows * k), yy(rows * n), ref(rows * n);
                for (int i = 0; i < rows * k; ++i) xx.p[i] = draw();
                for (int t = 0; t < rows; ++t)
                    main_reference::k_linear_f32_sg16(w.p, n, k, xx.p + t * k, ref.p + t * n, 1);
                k_linear_f32_sg16(w.p, n, k, xx.p, yy.p, rows);
                wait();
                equal("FP32 prefill paired rows and tail", ref.p, yy.p, rows * n);
            }
            ++nlinear;
        }
        if (k % 256 == 0) {
            main_reference::k_linear_bf16_sg16(wb.p, n, k, x.p, a.p, 1);
            k_streaming_gemv<true>(wb.p, n, k, x.p, b.p);
            wait();
            equal("BF16 GEMV", a.p, b.p, n);
            ++nlinear;
        }
#endif
    }
    constexpr int m = 5, nh = 24, hd = 128, stride = 2080;
    Array<float> q(m * nh * hd), r(m * nh * 16), rp(16 * 1025), s(m * nh), g(m * nh * hd), a(m * nh * hd),
        b(m * nh * hd);
    Array<half> k(2 * stride * 8 * hd), v(2 * stride * 8 * hd);
    Array<int> pos(m), slot(m);
    for (int i = 0; i < m * nh * hd; ++i) {
        q.p[i] = draw();
        g.p[i] = draw();
    }
    for (int i = 0; i < m * nh * 16; ++i)
        r.p[i] = draw();
    for (int i = 0; i < m * nh; ++i)
        s.p[i] = draw();
    for (int i = 0; i < 16 * 1025; ++i)
        rp.p[i] = (i % 1025 == 1024) ? 0.f : draw();
    for (int i = 0; i < 2 * stride * 8 * hd; ++i) {
        k.p[i] = half(draw());
        v.p[i] = half(draw());
    }
    for (int i = 0; i < m; ++i) {
        pos.p[i] = std::array<int, m>{0, 7, 1023, 1024, 2067}[i];
        slot.p[i] = i % 2;
    }
    for (int swa : {0, 1})
        for (int extent : {63, 1024}) {
            main_reference::k_relpos_sink_gqa_sg16(q.p, r.p, rp.p, s.p, g.p, k.p, v.p, slot.p, pos.p, a.p, m, swa, 1024,
                                   extent, stride);
            ++nattn;
            k_relpos_prefill_shared_kv(q.p, r.p, rp.p, s.p, g.p, k.p, v.p, slot.p, pos.p, b.p, m, swa, 1024,
                                       extent, stride);
            wait();
            equal("shared KV", a.p, b.p, m * nh * hd);
            Array<float> partial(nh * 8 * 130);
            for (int i = 0; i < m; ++i)
                k_relpos_decode_split_kv(q.p + i * nh * hd, r.p + i * nh * 16, rp.p, s.p + i * nh,
                                         g.p + i * nh * hd, k.p, v.p, slot.p[i], pos.p[i], b.p + i * nh * hd,
                                         partial.p, swa, 1024, extent, stride);
            wait();
            equal("split KV", a.p, b.p, m * nh * hd);
        }
    single_request_ops_check();
    std::puts("PASS bitwise: tiled ShortConv and carried history, aliased controller and single-token residual mixing");
    controller_batch_check();
    std::puts("PASS bitwise: attention/K/V and MLP controllers, shared SWA input, inexact-weight fallback");
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
    std::printf("PASS bitwise: %d projection cases, %d attention configurations (5 queries each)\n", nlinear,
                nattn);
}
