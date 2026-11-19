#include "spin8_triality.cuh"
#include "ops.cuh"
#include <cmath>
#include <cstdio>
#include <vector>

namespace Q = strata::qwen38;

__global__ void test_fp8_dot_kernel(
    const __nv_fp8_e4m3* x_fp8, const float* x_sc,
    const __nv_fp8_e4m3* w_fp8, const float* w_sc,
    float* out, int M, int N, int K
) {
    const int m = blockIdx.y;
    const int n = blockIdx.x;
    if (m >= M || n >= N) return;
    float acc = 0.0f;
    for (int k = threadIdx.x; k < K; k += blockDim.x) {
        acc += static_cast<float>(x_fp8[m * K + k]) * static_cast<float>(w_fp8[n * K + k]);
    }
    for (int mask = 16; mask > 0; mask >>= 1) acc += __shfl_xor_sync(0xffffffffu, acc, mask);
    __shared__ float sm[8];
    if ((threadIdx.x & 31) == 0) sm[threadIdx.x >> 5] = acc;
    __syncthreads();
    if (threadIdx.x == 0) {
        float s = 0.0f;
        for (int i = 0; i < 8; ++i) s += sm[i];
        out[m * N + n] = s * x_sc[m] * w_sc[n];
    }
}

__global__ void test_k8v4_dot_kernel(
    const float* q_rot, const int8_t* k_q8, const half* k_sc,
    float* out, int n_tokens, int n_kv_heads, int ctx
) {
    const int tok = blockIdx.x;
    const int kh  = blockIdx.y;
    if (tok >= n_tokens || kh >= n_kv_heads) return;
    const size_t q_off = (static_cast<size_t>(tok) * n_kv_heads + kh) * 256;
    const size_t c_cell = static_cast<size_t>(kh) * ctx + tok;
    const float sc = __half2float(k_sc[c_cell]);
    float acc = 0.0f;
    for (int i = threadIdx.x; i < 256; i += blockDim.x) {
        acc += q_rot[q_off + i] * (static_cast<float>(k_q8[c_cell * 256 + i]) * sc);
    }
    for (int mask = 16; mask > 0; mask >>= 1) acc += __shfl_xor_sync(0xffffffffu, acc, mask);
    if (threadIdx.x == 0) out[tok * n_kv_heads + kh] = acc;
}

int main() {
    Q::ck(cudaSetDevice(0));

    // 1. Test spin8_fp8_hadamard_quant_kernel orthogonality: (X H_8)(W H_8)^T == X W^T
    constexpr int M = 8, N = 8, K = 256;
    std::vector<half> h_x(M * K), h_w(N * K);
    for (int i = 0; i < M * K; ++i) h_x[i] = __float2half(0.1f * std::sin(0.11f * i));
    for (int i = 0; i < N * K; ++i) h_w[i] = __float2half(0.1f * std::cos(0.07f * i));
    // Inject a sharp outlier in X to verify Spin(8) flattening
    h_x[0] = __float2half(12.5f);

    Q::Buffer d_x(M * K * sizeof(half)), d_w(N * K * sizeof(half));
    Q::ck(cudaMemcpy(d_x.data, h_x.data(), d_x.bytes, cudaMemcpyHostToDevice));
    Q::Buffer d_w_buf(N * K * sizeof(half));
    Q::ck(cudaMemcpy(d_w_buf.data, h_w.data(), d_w_buf.bytes, cudaMemcpyHostToDevice));

    Q::Buffer d_x_fp8(M * K), d_w_fp8(N * K), d_x_sc(M * 4), d_w_sc(N * 4), d_out(M * N * 4);
    Q::spin8_fp8_hadamard_quant_kernel<<<M, 256>>>(d_x.as<half>(), d_x_fp8.as<__nv_fp8_e4m3>(), d_x_sc.as<float>(), M, K);
    Q::spin8_fp8_hadamard_quant_kernel<<<N, 256>>>(d_w_buf.as<half>(), d_w_fp8.as<__nv_fp8_e4m3>(), d_w_sc.as<float>(), N, K);
    test_fp8_dot_kernel<<<dim3(N, M), 256>>>(d_x_fp8.as<__nv_fp8_e4m3>(), d_x_sc.as<float>(),
                                             d_w_fp8.as<__nv_fp8_e4m3>(), d_w_sc.as<float>(),
                                             d_out.as<float>(), M, N, K);
    Q::ck(cudaDeviceSynchronize());

    std::vector<float> got(M * N);
    Q::ck(cudaMemcpy(got.data(), d_out.data, got.size() * 4, cudaMemcpyDeviceToHost));
    double max_fp8_err = 0.0;
    for (int m = 0; m < M; ++m) {
        for (int n = 0; n < N; ++n) {
            double ref = 0.0;
            for (int k = 0; k < K; ++k) ref += __half2float(h_x[m * K + k]) * __half2float(h_w[n * K + k]);
            double e = std::abs(ref - got[m * N + n]);
            if (e > max_fp8_err) max_fp8_err = e;
        }
    }
    std::printf("spin8_fp8_hadamard_quant max_abs_err=%.6g\n", max_fp8_err);
    if (max_fp8_err > 0.15) return 1;

    // 2. Test spin8_kv_write_batch_kernel + spin8_q_precondition_kernel dot parity
    constexpr int T = 4, KH = 4, CTX = 16;
    std::vector<float> h_q(T * KH * 256), h_k(T * KH * 256), h_v(T * KH * 256);
    for (size_t i = 0; i < h_q.size(); ++i) {
        h_q[i] = 0.2f * std::sin(0.05f * i);
        h_k[i] = 0.2f * std::cos(0.03f * i);
        h_v[i] = 0.2f * std::sin(0.09f * i);
    }
    Q::Buffer dq(h_q.size() * 4), dq_rot(h_q.size() * 4), dk(h_k.size() * 4), dv(h_v.size() * 4);
    Q::ck(cudaMemcpy(dq.data, h_q.data(), dq.bytes, cudaMemcpyHostToDevice));
    Q::ck(cudaMemcpy(dk.data, h_k.data(), dk.bytes, cudaMemcpyHostToDevice));
    Q::ck(cudaMemcpy(dv.data, h_v.data(), dv.bytes, cudaMemcpyHostToDevice));

    Q::Buffer dk_q8(KH * CTX * 256), dk_sc(KH * CTX * 2), dv_q4(KH * CTX * 128), dv_sc(KH * CTX * 2), d_dot(T * KH * 4);
    Q::spin8_kv_write_batch_kernel<<<dim3(T, KH), 32>>>(
        dk.as<float>(), dv.as<float>(), dk_q8.as<int8_t>(), dk_sc.as<half>(),
        dv_q4.as<uint8_t>(), dv_sc.as<half>(), 0, T, KH, CTX, 10000000.0f);
    Q::spin8_q_precondition_kernel<<<dim3(T, KH), 32>>>(
        dq.as<float>(), dq_rot.as<float>(), 0, T, KH, 10000000.0f);
    test_k8v4_dot_kernel<<<dim3(T, KH), 32>>>(
        dq_rot.as<float>(), dk_q8.as<int8_t>(), dk_sc.as<half>(), d_dot.as<float>(), T, KH, CTX);
    Q::ck(cudaDeviceSynchronize());

    std::vector<float> got_dot(T * KH);
    Q::ck(cudaMemcpy(got_dot.data(), d_dot.data, got_dot.size() * 4, cudaMemcpyDeviceToHost));
    double max_kv_err = 0.0;
    for (int t = 0; t < T; ++t) {
        for (int kh = 0; kh < KH; ++kh) {
            double ref = 0.0;
            const size_t off = (static_cast<size_t>(t) * KH + kh) * 256;
            for (int d = 0; d < 256; ++d) ref += h_q[off + d] * h_k[off + d];
            double e = std::abs(ref - got_dot[t * KH + kh]);
            if (e > max_kv_err) max_kv_err = e;
        }
    }
    std::printf("spin8_kv_write_batch + q_precondition max_abs_err=%.6g\n", max_kv_err);
    if (max_kv_err > 0.05) return 1;

    std::puts("spin8_triality_test: ALL PASSED");
    return 0;
}
