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
    std::puts("prefill_gemm_test: ALL 11 TYPES PASSED");
    return 0;
}
