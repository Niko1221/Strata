#include "prefill_gemm.cuh"
#include "ops.cuh"
#include <cmath>
#include <cstdio>
#include <cstring>
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
} // namespace

int main() {
    Q::ck(cudaSetDevice(0));
    constexpr int K = 512;
    constexpr int N = 64;
    constexpr int T = 16;

    const int types[] = {0, 1, 8, 12, 13, 14, 21, 22, 23, 30, 142};
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
        const auto t = static_cast<Q::GgmlType>(type_id);
        const size_t rb = (K / Q::StrataQuantTensor::block_elements(t)) * Q::StrataQuantTensor::block_bytes(t);
        std::vector<uint8_t> h_w(N * rb, 0);
        // Populate deterministic non-zero block scales & payloads for every format.
        for (int r = 0; r < N; ++r) {
            uint8_t* row = h_w.data() + static_cast<size_t>(r) * rb;
            if (type_id == 0) {
                auto* fp = reinterpret_cast<float*>(row);
                for (int c = 0; c < K; ++c) fp[c] = 0.02f * std::cos(0.07f * (r * K + c));
            } else if (type_id == 1) {
                auto* hp = reinterpret_cast<uint16_t*>(row);
                for (int c = 0; c < K; ++c) hp[c] = f32_to_f16_bits(0.02f * std::cos(0.07f * (r * K + c)));
            } else if (type_id == 30) {
                auto* bp = reinterpret_cast<uint16_t*>(row);
                for (int c = 0; c < K; ++c) {
                    float v = 0.02f * std::cos(0.07f * (r * K + c));
                    uint32_t u; std::memcpy(&u, &v, 4);
                    bp[c] = static_cast<uint16_t>(u >> 16);
                }
            } else {
                const size_t bb = Q::StrataQuantTensor::block_bytes(t);
                const size_t nb = rb / bb;
                for (size_t b = 0; b < nb; ++b) {
                    uint8_t* blk = row + b * bb;
                    for (size_t i = 0; i < bb; ++i) blk[i] = static_cast<uint8_t>((r * 31 + b * 17 + i * 13) & 0xFF);
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
    ws.release();

    // Real Qwen3.8-27B shapes with multi-tile 16 MiB buffer (checks 16-multiple tile alignment & tail).
    struct ShapeCase { int k; int n; int t; int type_id; };
    const ShapeCase real_cases[] = {
        {5120, 5120, 64, 12},
        {6144, 5120, 64, 14},
        {17408, 5120, 64, 22},
        {5120, 5120, 12, 12},
    };
    Q::PrefillGemmWorkspace rws;
    rws.init(64, 17408, 16ull * 1024 * 1024);
    if (rws.allocated_bytes() != Q::PrefillGemmWorkspace::estimate_bytes(64, 17408, 16ull * 1024 * 1024)) {
        std::fprintf(stderr, "real workspace estimate_bytes mismatch\n");
        return 1;
    }
    for (const auto& sc : real_cases) {
        rws.reset_stats();
        const auto t = static_cast<Q::GgmlType>(sc.type_id);
        const size_t rb = (sc.k / Q::StrataQuantTensor::block_elements(t)) * Q::StrataQuantTensor::block_bytes(t);
        std::vector<uint8_t> h_rw(static_cast<size_t>(sc.n) * rb, 0x11);
        std::vector<float> h_rx(static_cast<size_t>(sc.t) * sc.k, 0.01f);
        Q::Buffer d_rw(h_rw.size()), d_rx(h_rx.size() * sizeof(float)), d_ry(static_cast<size_t>(sc.t) * sc.n * sizeof(float));
        Q::ck(cudaMemcpy(d_rw.data, h_rw.data(), d_rw.bytes, cudaMemcpyHostToDevice));
        Q::ck(cudaMemcpy(d_rx.data, h_rx.data(), d_rx.bytes, cudaMemcpyHostToDevice));
        const auto rqt = Q::make_quant_tensor(d_rw.data, sc.type_id, sc.k, sc.n);
        rws.matmul_batch(rqt, d_rx.as<float>(), d_ry.as<float>(), sc.t, nullptr);
        Q::ck(cudaDeviceSynchronize());
        const auto rst = rws.stats();
        std::printf("real K=%d N=%d T=%d type=%d -> fp16=%llu fp8=%llu fb_shape=%llu fb_algo=%llu\n",
                    sc.k, sc.n, sc.t, sc.type_id,
                    (unsigned long long)rst.fp16_gemm_calls, (unsigned long long)rst.fp8_gemm_calls,
                    (unsigned long long)rst.fp8_fallback_shape, (unsigned long long)rst.fp8_fallback_no_algo);
        if (rst.fp16_gemm_calls + rst.fp8_gemm_calls == 0) {
            std::fprintf(stderr, "no GEMM calls recorded for K=%d N=%d T=%d\n", sc.k, sc.n, sc.t);
            return 1;
        }
    }
    std::puts("prefill_gemm_test: ALL 11 TYPES + REAL SHAPES PASSED");
    return 0;
}
