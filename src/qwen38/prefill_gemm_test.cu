#include "prefill_gemm.cuh"
#include "ops.cuh"
#include <cmath>
#include <cstdio>
#include <cstring>
#include <limits>
#include <vector>

namespace Q = strata::qwen38;

namespace {
uint16_t f32_to_f16_bits(float v) {
    uint32_t x;
    std::memcpy(&x, &v, 4);
    uint32_t sign = (x >> 16) & 0x8000u;
    int32_t exp = ((x >> 23) & 0xff) - 127 + 15;
    uint32_t mant = x & 0x7fffffu;
    if (exp <= 0) return static_cast<uint16_t>(sign);
    if (exp >= 31) return static_cast<uint16_t>(sign | 0x7c00u);
    return static_cast<uint16_t>(sign | (exp << 10) | (mant >> 13));
}

std::vector<uint8_t> build_test_weights(int type_id, int K, int N, float f32_scale = 0.02f) {
    const auto t = static_cast<Q::GgmlType>(type_id);
    const size_t rb = (K / Q::StrataQuantTensor::block_elements(t)) * Q::StrataQuantTensor::block_bytes(t);
    std::vector<uint8_t> h_w(static_cast<size_t>(N) * rb, 0);
    for (int r = 0; r < N; ++r) {
        uint8_t* row = h_w.data() + static_cast<size_t>(r) * rb;
        if (type_id == 0) {
            auto* fp = reinterpret_cast<float*>(row);
            for (int c = 0; c < K; ++c) fp[c] = f32_scale * std::cos(0.07f * (r * K + c));
        } else if (type_id == 1) {
            auto* hp = reinterpret_cast<uint16_t*>(row);
            for (int c = 0; c < K; ++c) hp[c] = f32_to_f16_bits(f32_scale * std::cos(0.07f * (r * K + c)));
        } else if (type_id == 30) {
            auto* bp = reinterpret_cast<uint16_t*>(row);
            for (int c = 0; c < K; ++c) {
                float v = f32_scale * std::cos(0.07f * (r * K + c));
                uint32_t u;
                std::memcpy(&u, &v, 4);
                bp[c] = static_cast<uint16_t>(u >> 16);
            }
        } else {
            const size_t bb = Q::StrataQuantTensor::block_bytes(t);
            const size_t nb = rb / bb;
            for (size_t b = 0; b < nb; ++b) {
                uint8_t* blk = row + b * bb;
                for (size_t i = 0; i < bb; ++i) {
                    blk[i] = static_cast<uint8_t>((r * 31 + b * 17 + i * 13) & 0xFF);
                }
                uint16_t d16 = f32_to_f16_bits(0.01f);
                if (type_id == 14) {
                    blk[208] = static_cast<uint8_t>(d16 & 0xFF);
                    blk[209] = static_cast<uint8_t>(d16 >> 8);
                } else {
                    blk[0] = static_cast<uint8_t>(d16 & 0xFF);
                    blk[1] = static_cast<uint8_t>(d16 >> 8);
                    if (type_id == 12 || type_id == 13) {
                        blk[2] = static_cast<uint8_t>(d16 & 0xFF);
                        blk[3] = static_cast<uint8_t>(d16 >> 8);
                    }
                }
            }
        }
    }
    return h_w;
}
} // namespace

int main() {
    static_assert(Q::PrefillGemmWorkspace::kDefaultTileBufBytes == 20ull * 1024 * 1024,
                  "kDefaultTileBufBytes must be 20 MiB");
    Q::ck(cudaSetDevice(0));
    constexpr int K = 512;
    constexpr int N = 64;
    constexpr int T = 16;

    const int types[] = {0, 1, 8, 12, 13, 14, 21, 22, 23, 30, 142};

    // 1. Direct dequant_rows_fp16 exactness check (including ragged tile_rows=19 across 8-row CTA boundary).
    for (int type_id : types) {
        constexpr int RaggedRows = 19;
        auto h_w = build_test_weights(type_id, K, N);
        Q::Buffer d_w(h_w.size());
        Q::ck(cudaMemcpy(d_w.data, h_w.data(), d_w.bytes, cudaMemcpyHostToDevice));
        const auto qt = Q::make_quant_tensor(d_w.data, type_id, K, N);

        Q::Buffer d_deq_fp16(static_cast<size_t>(RaggedRows) * K * sizeof(half));
        Q::Buffer d_deq_ref(static_cast<size_t>(RaggedRows) * K * sizeof(float));
        Q::dequant_rows_fp16(qt, 5, RaggedRows, d_deq_fp16.as<half>(), nullptr);
        const uint8_t* src_off = static_cast<const uint8_t*>(d_w.data) + 5 * qt.row_bytes;
        for (int r = 0; r < RaggedRows; ++r) {
            Q::decode(type_id, src_off + static_cast<size_t>(r) * qt.row_bytes,
                      d_deq_ref.as<float>() + static_cast<size_t>(r) * K, K, nullptr);
        }
        Q::ck(cudaDeviceSynchronize());

        std::vector<half> got_h(static_cast<size_t>(RaggedRows) * K);
        std::vector<float> ref_f(static_cast<size_t>(RaggedRows) * K);
        Q::ck(cudaMemcpy(got_h.data(), d_deq_fp16.data, got_h.size() * sizeof(half), cudaMemcpyDeviceToHost));
        Q::ck(cudaMemcpy(ref_f.data(), d_deq_ref.data, ref_f.size() * sizeof(float), cudaMemcpyDeviceToHost));
        for (size_t i = 0; i < ref_f.size(); ++i) {
            const float expected = __half2float(__float2half_rn(ref_f[i]));
            const float actual = __half2float(got_h[i]);
            if (std::abs(expected - actual) > 1e-6f) {
                std::fprintf(stderr, "dequant_rows_fp16 mismatch type=%d idx=%zu exp=%g got=%g\n",
                             type_id, i, expected, actual);
                return 1;
            }
        }
    }

    // 2. All 11 GGUF types matmul_batch parity vs matvec.
    Q::PrefillGemmWorkspace ws;
    ws.init(T, K, 1024 * 1024);

    std::vector<float> h_x(T * K);
    for (int i = 0; i < T * K; ++i) {
        h_x[i] = 0.05f * std::sin(0.13f * i);
    }
    Q::Buffer d_x(h_x.size() * sizeof(float));
    Q::ck(cudaMemcpy(d_x.data, h_x.data(), d_x.bytes, cudaMemcpyHostToDevice));

    Q::Buffer d_y_ref(T * N * sizeof(float));
    Q::Buffer d_y_gemm(T * N * sizeof(float));

    for (int type_id : types) {
        auto h_w = build_test_weights(type_id, K, N);
        Q::Buffer d_w(h_w.size());
        Q::ck(cudaMemcpy(d_w.data, h_w.data(), d_w.bytes, cudaMemcpyHostToDevice));
        const auto qt = Q::make_quant_tensor(d_w.data, type_id, K, N);

        for (int tok = 0; tok < T; ++tok) {
            Q::matvec(type_id, d_w.data, d_x.as<float>() + tok * K, d_y_ref.as<float>() + tok * N, K, N, nullptr);
        }
        ws.matmul_batch(qt, d_x.as<float>(), d_y_gemm.as<float>(), T, nullptr);
        Q::ck(cudaDeviceSynchronize());

        std::vector<float> ref(T * N), got(T * N);
        Q::ck(cudaMemcpy(ref.data(), d_y_ref.data, ref.size() * sizeof(float), cudaMemcpyDeviceToHost));
        Q::ck(cudaMemcpy(got.data(), d_y_gemm.data, got.size() * sizeof(float), cudaMemcpyDeviceToHost));

        double max_err = 0.0, max_ref = 1e-6;
        for (size_t i = 0; i < ref.size(); ++i) {
            double r = std::abs(static_cast<double>(ref[i]));
            double e = std::abs(static_cast<double>(ref[i]) - static_cast<double>(got[i]));
            if (r > max_ref) max_ref = r;
            if (e > max_err) max_err = e;
            if (!std::isfinite(got[i])) {
                std::fprintf(stderr, "non-finite output for type=%d\n", type_id);
                return 1;
            }
        }
        double rel_err = max_err / max_ref;
        std::printf("type=%3d max_abs_err=%.6g rel_err=%.6g\n", type_id, max_err, rel_err);
        if (max_err > 0.05 && rel_err > 0.06) {
            std::fprintf(stderr, "parity failure for type=%d max_err=%g rel_err=%g\n", type_id, max_err, rel_err);
            return 1;
        }
    }
    if (ws.allocated_bytes() != Q::PrefillGemmWorkspace::estimate_bytes(T, K, 1024 * 1024)) {
        std::fprintf(stderr, "estimate_bytes mismatch\n");
        return 1;
    }
    const auto st0 = ws.stats();
    std::printf("small_shapes: fp16=%llu fp8=%llu fb_shape=%llu fb_algo=%llu\n",
                (unsigned long long)st0.fp16_gemm_calls, (unsigned long long)st0.fp8_gemm_calls,
                (unsigned long long)st0.fp8_fallback_shape, (unsigned long long)st0.fp8_fallback_no_algo);

    // 3. prepare_activation + matmul_batch_prepared + reset_l2_window + <4 token fallback check.
    {
        constexpr int N0 = 48, N1 = 136;
        auto hw0 = build_test_weights(8, K, N0);
        auto hw1 = build_test_weights(22, K, N1);
        Q::Buffer dw0(hw0.size()), dw1(hw1.size());
        Q::ck(cudaMemcpy(dw0.data, hw0.data(), dw0.bytes, cudaMemcpyHostToDevice));
        Q::ck(cudaMemcpy(dw1.data, hw1.data(), dw1.bytes, cudaMemcpyHostToDevice));
        const auto qt0 = Q::make_quant_tensor(dw0.data, 8, K, N0);
        const auto qt1 = Q::make_quant_tensor(dw1.data, 22, K, N1);

        Q::Buffer dy0_prep(T * N0 * sizeof(float)), dy0_solo(T * N0 * sizeof(float));
        Q::Buffer dy1_prep(T * N1 * sizeof(float)), dy1_solo(T * N1 * sizeof(float));

        ws.prepare_activation(d_x.as<float>(), T, K, nullptr);
        ws.matmul_batch_prepared(qt0, dy0_prep.as<float>(), T, nullptr);
        ws.reset_l2_window(nullptr); // Exercise L2 persisting window reset & subsequent re-attach
        ws.matmul_batch_prepared(qt1, dy1_prep.as<float>(), T, nullptr);
        ws.matmul_batch(qt0, d_x.as<float>(), dy0_solo.as<float>(), T, nullptr);
        ws.matmul_batch(qt1, d_x.as<float>(), dy1_solo.as<float>(), T, nullptr);
        Q::ck(cudaDeviceSynchronize());
        ws.reset_l2_window(nullptr);

        std::vector<float> a0(T * N0), b0(T * N0), a1(T * N1), b1(T * N1);
        Q::ck(cudaMemcpy(a0.data(), dy0_prep.data, a0.size() * 4, cudaMemcpyDeviceToHost));
        Q::ck(cudaMemcpy(b0.data(), dy0_solo.data, b0.size() * 4, cudaMemcpyDeviceToHost));
        Q::ck(cudaMemcpy(a1.data(), dy1_prep.data, a1.size() * 4, cudaMemcpyDeviceToHost));
        Q::ck(cudaMemcpy(b1.data(), dy1_solo.data, b1.size() * 4, cudaMemcpyDeviceToHost));
        for (size_t i = 0; i < a0.size(); ++i) {
            if (a0[i] != b0[i]) {
                std::fprintf(stderr, "prepared vs standalone mismatch on qt0 at %zu\n", i);
                return 1;
            }
        }
        for (size_t i = 0; i < a1.size(); ++i) {
            if (a1[i] != b1[i]) {
                std::fprintf(stderr, "prepared vs standalone mismatch on qt1 at %zu\n", i);
                return 1;
            }
        }

        // Verify n_tokens = 3 (< 4) matvec fallback via prepare_activation + matmul_batch_prepared.
        ws.reset_stats();
        ws.prepare_activation(d_x.as<float>(), 3, K, nullptr);
        ws.matmul_batch_prepared(qt0, dy0_prep.as<float>(), 3, nullptr);
        Q::ck(cudaDeviceSynchronize());
        if (ws.stats().matvec_fallback_calls != 3) {
            std::fprintf(stderr, "expected 3 matvec_fallback_calls, got %llu\n",
                         (unsigned long long)ws.stats().matvec_fallback_calls);
            return 1;
        }
    }

    // 4. Token-scaled dynamic range (overflow/underflow immunity), gm==0, NaN/Inf non-masking, and extreme F32 weights.
    {
        constexpr int TD = 6;
        std::vector<float> h_xd(TD * K, 0.0f);
        for (int k = 0; k < K; ++k) {
            const float pat = std::sin(0.11f * (k + 1));
            h_xd[0 * K + k] = 100000.0f * pat; // > 65504 FP16 max
            h_xd[1 * K + k] = 1.0e-7f * pat;   // < FP16 normal min
            h_xd[2 * K + k] = 0.0f;            // exact zero -> act_scales == 0.0f
            h_xd[3 * K + k] = 0.5f * pat;
            h_xd[4 * K + k] = 0.5f * pat;
            h_xd[5 * K + k] = 0.5f * pat;
        }
        h_xd[3 * K + 17] = std::numeric_limits<float>::quiet_NaN();
        h_xd[4 * K + 33] = std::numeric_limits<float>::infinity();

        Q::Buffer d_xd(h_xd.size() * sizeof(float));
        Q::ck(cudaMemcpy(d_xd.data, h_xd.data(), d_xd.bytes, cudaMemcpyHostToDevice));

        auto hw_q4 = build_test_weights(12, K, N);
        auto hw_f32_big = build_test_weights(0, K, N, 500.0f); // large F32 weights
        Q::Buffer dw_q4(hw_q4.size()), dw_f32(hw_f32_big.size());
        Q::ck(cudaMemcpy(dw_q4.data, hw_q4.data(), dw_q4.bytes, cudaMemcpyHostToDevice));
        Q::ck(cudaMemcpy(dw_f32.data, hw_f32_big.data(), dw_f32.bytes, cudaMemcpyHostToDevice));
        const auto qt_q4 = Q::make_quant_tensor(dw_q4.data, 12, K, N);
        const auto qt_f32 = Q::make_quant_tensor(dw_f32.data, 0, K, N);

        for (const auto* qt_ptr : {&qt_q4, &qt_f32}) {
            Q::Buffer d_yd_ref(TD * N * sizeof(float)), d_yd_got(TD * N * sizeof(float));
            for (int tok = 0; tok < TD; ++tok) {
                Q::matvec(static_cast<int>(qt_ptr->type), qt_ptr->d_data,
                          d_xd.as<float>() + tok * K, d_yd_ref.as<float>() + tok * N, K, N, nullptr);
            }
            ws.matmul_batch(*qt_ptr, d_xd.as<float>(), d_yd_got.as<float>(), TD, nullptr);
            Q::ck(cudaDeviceSynchronize());

            std::vector<float> yd_ref(TD * N), yd_got(TD * N);
            Q::ck(cudaMemcpy(yd_ref.data(), d_yd_ref.data, yd_ref.size() * 4, cudaMemcpyDeviceToHost));
            Q::ck(cudaMemcpy(yd_got.data(), d_yd_got.data, yd_got.size() * 4, cudaMemcpyDeviceToHost));

            for (int tok : {0, 1, 2, 5}) {
                double max_e = 0.0, max_r = 0.0;
                for (int c = 0; c < N; ++c) {
                    const float rv = yd_ref[tok * N + c];
                    const float gv = yd_got[tok * N + c];
                    if (!std::isfinite(gv)) {
                        std::fprintf(stderr, "unexpected non-finite on tok=%d type=%d\n", tok, (int)qt_ptr->type);
                        return 1;
                    }
                    max_r = std::fmax(max_r, std::abs(static_cast<double>(rv)));
                    max_e = std::fmax(max_e, std::abs(static_cast<double>(rv) - static_cast<double>(gv)));
                }
                if (tok == 2 && max_e != 0.0) {
                    std::fprintf(stderr, "zero token produced non-zero output: %g\n", max_e);
                    return 1;
                }
                if (max_r > 0.0 && (max_e / max_r) > 0.01) {
                    std::fprintf(stderr, "dynamic range rel_err=%g too high on tok=%d type=%d\n",
                                 max_e / max_r, tok, (int)qt_ptr->type);
                    return 1;
                }
            }
            for (int tok : {3, 4}) {
                for (int c = 0; c < N; ++c) {
                    if (std::isfinite(yd_got[tok * N + c])) {
                        std::fprintf(stderr, "NaN/Inf was masked at tok=%d c=%d val=%g\n",
                                     tok, c, yd_got[tok * N + c]);
                        return 1;
                    }
                }
            }
        }
    }
    ws.release();

    // 5. Real Qwen3.8-27B shapes (including T=140 N=48 ssm_alpha, K=6144 balanced tiles, and K=17408 target_max=4.0f).
    struct ShapeCase { int k; int n; int t; int type_id; };
    const ShapeCase real_cases[] = {
        {5120, 48, 140, 8},
        {5120, 5120, 64, 12},
        {6144, 5120, 64, 14},
        {17408, 5120, 64, 22},
        {5120, 5120, 12, 12},
    };
    cudaStream_t test_stream = nullptr;
    Q::ck(cudaStreamCreateWithFlags(&test_stream, cudaStreamNonBlocking));
    Q::PrefillGemmWorkspace rws;
    rws.init(140, 17408);
    if (rws.allocated_bytes() != Q::PrefillGemmWorkspace::estimate_bytes(140, 17408)) {
        std::fprintf(stderr, "real workspace estimate_bytes mismatch\n");
        return 1;
    }
    for (const auto& sc : real_cases) {
        rws.reset_stats();
        auto h_rw = build_test_weights(sc.type_id, sc.k, sc.n);
        std::vector<float> h_rx(static_cast<size_t>(sc.t) * sc.k);
        for (size_t i = 0; i < h_rx.size(); ++i) {
            h_rx[i] = 0.08f * std::sin(0.017f * static_cast<float>(i));
        }
        Q::Buffer d_rw(h_rw.size()), d_rx(h_rx.size() * sizeof(float)), d_ry(static_cast<size_t>(sc.t) * sc.n * sizeof(float));
        Q::ck(cudaMemcpy(d_rw.data, h_rw.data(), d_rw.bytes, cudaMemcpyHostToDevice));
        Q::ck(cudaMemcpy(d_rx.data, h_rx.data(), d_rx.bytes, cudaMemcpyHostToDevice));
        const auto rqt = Q::make_quant_tensor(d_rw.data, sc.type_id, sc.k, sc.n);
        rws.matmul_batch(rqt, d_rx.as<float>(), d_ry.as<float>(), sc.t, test_stream);
        Q::ck(cudaStreamSynchronize(test_stream));
        const auto rst = rws.stats();
        std::printf("real K=%d N=%d T=%d type=%d -> fp16=%llu fp8=%llu wmma=%llu fb_shape=%llu fb_algo=%llu\n",
                    sc.k, sc.n, sc.t, sc.type_id,
                    (unsigned long long)rst.fp16_gemm_calls, (unsigned long long)rst.fp8_gemm_calls,
                    (unsigned long long)rst.fused_wmma_calls,
                    (unsigned long long)rst.fp8_fallback_shape, (unsigned long long)rst.fp8_fallback_no_algo);
        if (rst.fp16_gemm_calls + rst.fp8_gemm_calls + rst.fused_wmma_calls == 0) {
            std::fprintf(stderr, "no GEMM calls recorded for K=%d N=%d T=%d\n", sc.k, sc.n, sc.t);
            return 1;
        }

        // Spot-check first and last token against Q::matvec reference
        Q::Buffer d_ref0(sc.n * sizeof(float)), d_ref1(sc.n * sizeof(float));
        Q::matvec(sc.type_id, d_rw.data, d_rx.as<float>(), d_ref0.as<float>(), sc.k, sc.n, test_stream);
        Q::matvec(sc.type_id, d_rw.data, d_rx.as<float>() + static_cast<size_t>(sc.t - 1) * sc.k,
                  d_ref1.as<float>(), sc.k, sc.n, test_stream);
        Q::ck(cudaStreamSynchronize(test_stream));
        std::vector<float> got0(sc.n), got1(sc.n), ref0(sc.n), ref1(sc.n);
        Q::ck(cudaMemcpy(got0.data(), d_ry.as<float>(), sc.n * 4, cudaMemcpyDeviceToHost));
        Q::ck(cudaMemcpy(got1.data(), d_ry.as<float>() + static_cast<size_t>(sc.t - 1) * sc.n, sc.n * 4, cudaMemcpyDeviceToHost));
        Q::ck(cudaMemcpy(ref0.data(), d_ref0.data, sc.n * 4, cudaMemcpyDeviceToHost));
        Q::ck(cudaMemcpy(ref1.data(), d_ref1.data, sc.n * 4, cudaMemcpyDeviceToHost));
        double max_e = 0.0, max_r = 1e-6;
        for (int c = 0; c < sc.n; ++c) {
            max_r = std::fmax(max_r, std::fmax(std::abs(ref0[c]), std::abs(ref1[c])));
            max_e = std::fmax(max_e, std::fmax(std::abs(ref0[c] - got0[c]), std::abs(ref1[c] - got1[c])));
        }
        if (max_e > 0.05 && (max_e / max_r) > 0.02) {
            std::fprintf(stderr, "real shape parity failure K=%d N=%d T=%d max_e=%g rel=%g\n",
                         sc.k, sc.n, sc.t, max_e, max_e / max_r);
            return 1;
        }
    }
    rws.reset_l2_window(test_stream);
    rws.release();
    Q::ck(cudaStreamDestroy(test_stream));
    std::puts("prefill_gemm_test: ALL 11 TYPES + DYNAMIC RANGE + REAL SHAPES PASSED");
    return 0;
}
