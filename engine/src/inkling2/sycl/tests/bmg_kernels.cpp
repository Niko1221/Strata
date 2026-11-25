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
#ifdef INK_SYCL_USE_ONEMKL
static void dspark_q8_projection_check() {
    constexpr int n = 128, k = 2304, m = 8;
    Array<uint8_t> packed(n * (k + sizeof(float)));
    auto* w_scale = reinterpret_cast<float*>(packed.p);
    auto* w = reinterpret_cast<int8_t*>(w_scale + n);
    Array<int8_t> x(m * k);
    Array<float> x_scale(m), expected(m * n), actual(m * n);
    Array<int32_t> dots(m * n), zero(1);
    zero.p[0] = 0;

    for (int i = 0; i < m; ++i) {
        x_scale.p[i] = float(i + 1) / 1037.0f;
        for (int j = 0; j < k; ++j)
            x.p[i * k + j] = i == 0 ? 127 : (i * 29 + j * 43) % 255 - 127;
    }
    for (int i = 0; i < n; ++i) {
        w_scale[i] = float(i + 1) / 1303.0f;
        for (int j = 0; j < k; ++j)
            w[i * k + j] = i == 0 ? 127 : (i * 31 + j * 47) % 255 - 127;
    }
    // Exercise the conversion of a non-exact S32 sum beyond the 24-bit
    // mantissa of the FP32 output used by the original DP4A path.
    w[k - 1] = 126;
    k_linear_q8row_dp4a_sg16(packed.p, n, k, x.p, x_scale.p, expected.p, m);
    ink_xmx_dspark_q8(packed.p, n, k, x.p, x_scale.p, actual.p, dots.p, zero.p);
    wait();
    equal("DSpark eight-row INT8 projection", expected.p, actual.p, m * n);
}
#endif

#ifdef INK_HAS_GROUP_LOAD
static void tree_narrow_projection_check() {
    constexpr int k = 2304, max_n = 128, max_nodes = 12;
    Array<float> w(max_n * k), x(max_nodes * k), ref(max_nodes * max_n), got(max_nodes * max_n);
    for (int i = 0; i < max_n * k; ++i) w.p[i] = float((i * 19) % 2503 - 1250) / 3807.0f;
    for (int i = 0; i < max_nodes * k; ++i) x.p[i] = float((i * 31) % 3137 - 1520) / 5733.0f;
    for (int n : {23, 24, 32, 128}) {
        for (int nodes : {3, 8, 12}) {
            k_tree_linear_f32_sg16(w.p, n, k, x.p, ref.p, nodes);
            k_tree_linear_f32_striped(w.p, n, k, x.p, got.p, nodes);
            wait();
            equal("DSpark striped FP32 projection", ref.p, got.p, size_t(nodes) * n);
        }
    }
}
#endif

static void controller_batch_check() {
    constexpr int d = INK_D, rank = ctrl_rank, dr = ctrl_k * rank, gr = d * rank;
    Array<float> down(4 * dr), gain(4 * gr), forget(4 * gr), norm(4 * d);
    constexpr int max_nodes = 12, capacity = 3 * max_nodes * d;
    Array<float> original(capacity), msg(capacity), ref(capacity), actual(capacity), cat(3 * max_nodes * ctrl_k);
    Array<float> gates(3 * max_nodes * rank), want_gates(3 * max_nodes * rank);
    std::mt19937 rng(51);
    auto draw = [&] { return float(int(rng() % 20001) - 10000) / 15237.0f; };
    auto weight = [&] { return bf16_to_f32(sycl::bit_cast<uint32_t>(draw()) >> 16); };
    for (int i = 0; i < 4 * dr; ++i) down.p[i] = weight();
    for (int i = 0; i < 4 * gr; ++i) { gain.p[i] = weight(); forget.p[i] = weight(); }
    for (int i = 0; i < 4 * d; ++i) norm.p[i] = draw();
    for (int i = 0; i < capacity; ++i) { original.p[i] = draw(); msg.p[i] = draw(); }

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
    sy_setenv("INK_SYCL_BMG_DENSE", "1");
    InkControllerCache cache;
    for (int swa : {0, 1}) {
        c.is_swa[1] = swa;
        cache.prepare(model);
        if (cache.layers.size() != 1) throw std::runtime_error("expected exact controller cache");
        for (int nodes : {1, 3, max_nodes}) {
            const int span = nodes * d;
            for (int count : {1, 3}) {
                std::copy_n(original.p, capacity, actual.p);
                std::copy_n(original.p, capacity, ref.p);
                if (count == 3 && swa) {
                    std::copy_n(original.p, span, ref.p + span);
                    std::copy_n(original.p, span, ref.p + 2 * span);
                }
                for (int j = 0; j < count; ++j) {
                    const int bank = count == 1 ? 3 : j;
                    float* row = ref.p + j * span;
                    const float* message = msg.p + j * span;
                    float* gate = want_gates.p + j * nodes * rank;
                    main_reference::k_ctrl_cat_rms_sg16(row, message, cat.p, nodes);
                    main_reference::k_linear_f32_sg16(down.p + bank * dr, rank, ctrl_k, cat.p, gate, nodes);
                    main_reference::k_ctrl_apply_norm_sg16(row, message, gate,
                        gain.p + bank * gr, forget.p + bank * gr, norm.p + (count == 3 && swa ? 0 : bank * d),
                        row, nodes, 1);
                }
                const auto& layer = cache.layers[0];
                if (count == 3 && nodes == 1) k_ctrl_decode_batch(layer.attention, actual.p, msg.p, swa, cat.p, gates.p);
                else if (count == 1 && nodes == 1) k_ctrl_decode_batch(layer.mlp, actual.p, msg.p, false, cat.p, gates.p);
                else if (count == 3) k_ctrl_decode_batch<3, true>(layer.attention, actual.p, msg.p, swa, cat.p, gates.p, nodes);
                else k_ctrl_decode_batch<1, true>(layer.mlp, actual.p, msg.p, false, cat.p, gates.p, nodes);
                wait();
                equal("controller gate vector", want_gates.p, gates.p, count * nodes * rank);
                equal("in-place controller output and guard rows", ref.p, actual.p, capacity);
            }
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
    sy_setenv("INK_SYCL_BMG_DENSE", "0");
    cache.prepare(model);
    if (cache.storage || !cache.layers.empty()) throw std::runtime_error("controller override ignored");
    sy_setenv("INK_SYCL_BMG_DENSE", "1");
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
    constexpr int max_nodes = 12, stride = (max_nodes + 1) * d;
    Array<float> blocks(7 * stride), queries(4 * d), stats(4 * max_nodes * 8 * 2);
    for (int i = 0; i < 7 * stride; ++i) blocks.p[i] = draw();
    for (int i = 0; i < 4 * d; ++i) queries.p[i] = draw();
    for (int nodes : {1, 3, max_nodes}) {
        for (int n : {0, 1, 4, 7}) {
            for (int nq : {1, 3, 4}) {
                main_reference::k_attn_res_mix_sg16(blocks.p, x.p, queries.p, ref.p, n, stride, nodes, nq);
                if (nodes == 1) k_res_mix_decode(blocks.p, x.p, queries.p, y.p, n, stride, nq, stats.p);
                else k_res_mix_decode<true>(blocks.p, x.p, queries.p, y.p, n, stride, nq, stats.p, nodes);
                wait();
                equal("decode residual mix", ref.p, y.p, nodes * nq * d);
            }
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
        // The first four experts are SF4. Keep their original scattered row IDs.
        std::fill_n(test_h.p, rows * 1152, NAN);
        std::fill_n(test_y.p, rows * 2304, NAN);
        nu.p[0] = 4;
        k_tree_sf4_packed<false>(experts.p, ue.p, off.p, cnt.p, pairs.p, nu.p, 8, nullptr, x.p, test_h.p);
        k_tree_sf4_packed<true>(experts.p, ue.p, off.p, cnt.p, pairs.p, nu.p, 8, weights.p, test_h.p, test_y.p);
        wait();
        for (int token = 0; token < n; ++token) {
            equal("tree SF4 gate/up", gold_h.p + token * 8 * 1152, test_h.p + token * 8 * 1152, 4 * 1152);
            equal("tree SF4 down", gold_y.p + token * 8 * 2304, test_y.p + token * 8 * 2304, 4 * 2304);
        }

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
            if (n == 17 || n == 24) {
                InkTensorInfo weight;
                weight.fmt = FMT_BF16;
                weight.shape = {n, k};
                weight.dev_ptr = reinterpret_cast<uint8_t*>(wb.p);
                Array<float> xx(12 * k), yy(12 * n), ref(12 * n);
                for (int i = 0; i < 12 * k; ++i) xx.p[i] = draw();
                for (int nodes : {3, 4, 5, 6, 8, 10, 12}) {
                    main_reference::k_linear_bf16_sg16(wb.p, n, k, xx.p, ref.p, nodes);
                    if (!tree_bf16_projection(weight, xx.p, yy.p, nodes)) throw std::runtime_error("tree BF16 dispatch");
                    wait();
                    equal("tree BF16 projection", ref.p, yy.p, nodes * n);
                }
                if (tree_bf16_projection(weight, xx.p, yy.p, 7)) throw std::runtime_error("unexpected tree width");
                weight.fmt = FMT_F32;
                if (tree_bf16_projection(weight, xx.p, yy.p, 3)) throw std::runtime_error("unexpected tree dtype");
            }
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
            pos.p[m - 1] = swa ? stride + 33 : stride - 13;
            if (swa) {
                // The frozen reference uses a 1024-row ring. Reconstruct each
                // node's visible window from the actual batched cache stride.
                Array<half> rk(1024 * 8 * hd), rv(1024 * 8 * hd);
                Array<int> zero(1);
                zero.p[0] = 0;
                for (int i = 0; i < m; ++i) {
                    for (int j = std::max(0, pos.p[i] - 1023); j <= pos.p[i]; ++j) {
                        const int src = (slot.p[i] * stride + j % stride) * 8 * hd;
                        const int dst = (j % 1024) * 8 * hd;
                        std::copy_n(k.p + src, 8 * hd, rk.p + dst);
                        std::copy_n(v.p + src, 8 * hd, rv.p + dst);
                    }
                    main_reference::k_relpos_sink_gqa_sg16(q.p + i * nh * hd, r.p + i * nh * 16, rp.p,
                        s.p + i * nh, g.p + i * nh * hd, rk.p, rv.p, zero.p, pos.p + i, a.p + i * nh * hd,
                        1, swa, 1024, extent, 1024);
                    wait();
                }
            } else {
                main_reference::k_relpos_sink_gqa_sg16(q.p, r.p, rp.p, s.p, g.p, k.p, v.p, slot.p, pos.p,
                                                       a.p, m, swa, 1024, extent, stride);
            }
            ++nattn;
            k_relpos_prefill_shared_kv(q.p, r.p, rp.p, s.p, g.p, k.p, v.p, slot.p, pos.p, b.p, m, swa, 1024,
                                       extent, stride);
            wait();
            equal("shared KV", a.p, b.p, m * nh * hd);
            k_relpos_prefill_shared_kv<true>(q.p, r.p, rp.p, s.p, g.p, k.p, v.p, slot.p, pos.p, b.p, m,
                                            swa, 1024, extent, stride);
            wait();
            equal("shared KV with one-exp softmax and clipped bias", a.p, b.p, m * nh * hd);
            Array<float> partial(nh * 8 * 130);
            for (int i = 0; i < m; ++i)
                k_relpos_decode_split_kv(q.p + i * nh * hd, r.p + i * nh * 16, rp.p, s.p + i * nh,
                                         g.p + i * nh * hd, k.p, v.p, slot.p[i], pos.p[i], b.p + i * nh * hd,
                                         partial.p, swa, 1024, extent, stride);
            wait();
            equal("split KV", a.p, b.p, m * nh * hd);
        }
    // Materialize each branch's logical cache separately for the independent
    // SG16 reference; siblings at the same depth must never see one another.
    Array<int> ancestry(m * 8), zero(1);
    zero.p[0] = 0;
    const int depth[m] = {0, 1, 1, 2, 2}, parent[m] = {-1, 0, 0, 1, 2};
    for (int i = 0; i < m; ++i) {
        if (i) std::copy_n(ancestry.p + parent[i] * 8, depth[i], ancestry.p + i * 8);
        ancestry.p[i * 8 + depth[i]] = i;
        slot.p[i] = 1;
    }
    for (int root : {7, 1031, stride - 2, stride + 29}) {
        for (bool sliding : {false, true}) {
            if (!sliding && root + m > stride) continue;
            const int ref_stride = sliding ? 1024 : stride;
            Array<half> rk(ref_stride * 8 * hd), rv(ref_stride * 8 * hd);
            for (int extent : {63, 1024}) {
                for (int i = 0; i < m; ++i) {
                    const int pos_i = pos.p[i] = root + depth[i];
                    for (int j = sliding ? std::max(0, pos_i - 1023) : 0; j <= pos_i; ++j) {
                        const int physical = j < root ? j : root + ancestry.p[i * 8 + j - root];
                        const int src = (stride + physical % stride) * 8 * hd, dst = (j % ref_stride) * 8 * hd;
                        std::copy_n(k.p + src, 8 * hd, rk.p + dst);
                        std::copy_n(v.p + src, 8 * hd, rv.p + dst);
                    }
                    main_reference::k_relpos_sink_gqa_sg16(q.p + i * nh * hd, r.p + i * nh * 16, rp.p,
                        s.p + i * nh, g.p + i * nh * hd, rk.p, rv.p, zero.p, pos.p + i, a.p + i * nh * hd,
                        1, sliding, 1024, extent, ref_stride);
                    wait();
                }
                k_relpos_tree_stream(q.p, r.p, rp.p, s.p, g.p, k.p, v.p, slot.p, pos.p, m, sliding,
                                     1024, extent, stride, root, ancestry.p, b.p);
                wait();
                equal("tree KV vs independently materialized branch", a.p, b.p, m * nh * hd);
                if (sliding) {
                    k_relpos_tree_stream<true>(q.p, r.p, rp.p, s.p, g.p, k.p, v.p, slot.p, pos.p, m, true,
                                               1024, extent, stride, root, ancestry.p, b.p);
                    wait();
                    equal("sliding tree KV vs independently materialized branch", a.p, b.p, m * nh * hd);
                } else {
                    k_relpos_tree_stream<false, true>(q.p, r.p, rp.p, s.p, g.p, k.p, v.p, slot.p, pos.p, m, false,
                                                      1024, extent, stride, root, ancestry.p, b.p);
                    wait();
                    equal("full-context tree KV vs independently materialized branch", a.p, b.p, m * nh * hd);
                }
            }
        }
    }
    {
        constexpr int cases = 7;
        Array<float> inputs(cases), observed(cases * cases * 2), reference(cases * cases * 2);
        const float values[cases] = {-INFINITY, -12.f, -0.f, 0.f, 12.f, INFINITY, NAN};
        std::copy_n(values, cases, inputs.p);
        auto* in = inputs.p; auto* got = observed.p; auto* ref = reference.p;
        Q().parallel_for(sycl::range<1>(cases * cases), [=](sycl::id<1> id) {
            const int i = id[0];
            const float peak = in[i / cases], score = in[i % cases], mx = sycl::fmax(peak, score);
            const auto f = swa_online_factors(peak, score, mx);
            got[2*i] = f.x(); got[2*i+1] = f.y();
            ref[2*i] = sycl::exp(peak - mx); ref[2*i+1] = sycl::exp(score - mx);
        });
        wait();
        for (int i = 0; i < cases * cases * 2; ++i) {
            if (std::isnan(ref[i]) && std::isnan(got[i])) continue;
            equal("SWA softmax factor", ref + i, got + i, 1);
        }
    }
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
    single_request_ops_check();
    std::puts("PASS bitwise: tiled ShortConv and carried history, aliased controller and single-token residual mixing");
#ifdef INK_SYCL_USE_ONEMKL
    dspark_q8_projection_check();
    std::puts("PASS bitwise: DSpark eight-row INT8 XMX projection versus DP4A");
#endif
#ifdef INK_HAS_GROUP_LOAD
    tree_narrow_projection_check();
    std::puts("PASS bitwise: DSpark striped narrow FP32 projections");
#endif
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
