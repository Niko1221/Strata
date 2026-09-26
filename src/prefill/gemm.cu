// src/prefill/gemm.cu - see include/strata/prefill/gemm.hpp.
#include "strata/prefill/gemm.hpp"
#include "strata/kernels/dequant_bf16.hpp"

#include <cublas_v2.h>
#include <cuda_runtime.h>

#include <algorithm>
#include <cstdio>
#include <cstdlib>
#include <vector>

namespace strata::prefill {
namespace {

void ck(cublasStatus_t s, const char* what) {
    if (s != CUBLAS_STATUS_SUCCESS) {
        std::fprintf(stderr, "prefill gemm: %s: cuBLAS status %d\n", what, (int) s);
        std::exit(1);
    }
}

}  // namespace

Gemm::~Gemm() {
    if (handle_) cublasDestroy((cublasHandle_t) handle_);
    for (int i = 0; i < AUX; ++i) {
        if (aux_handle_[i]) cublasDestroy((cublasHandle_t) aux_handle_[i]);
        if (join_[i]) cudaEventDestroy((cudaEvent_t) join_[i]);
        if (aux_stream_[i]) {
            cudaStreamSynchronize((cudaStream_t) aux_stream_[i]);
            cudaStreamDestroy((cudaStream_t) aux_stream_[i]);
        }
    }
    if (fork_) cudaEventDestroy((cudaEvent_t) fork_);
}

bool Gemm::init(void* stream, std::string& err, bool side_streams) {
    cublasHandle_t h = nullptr;
    if (cublasCreate(&h) != CUBLAS_STATUS_SUCCESS) { err = "prefill gemm: cublasCreate failed"; return false; }
    handle_ = h;
    stream_ = stream;
    cublasSetStream(h, (cudaStream_t) stream);
    cublasSetMathMode(h, CUBLAS_DEFAULT_MATH);
    if (!side_streams) return true;
    if (cudaEventCreateWithFlags((cudaEvent_t*) &fork_, cudaEventDisableTiming) != cudaSuccess) {
        err = "prefill gemm: events";
        return false;
    }
    for (int i = 0; i < AUX; ++i) {
        cublasHandle_t a = nullptr;
        if (cudaStreamCreateWithFlags((cudaStream_t*) &aux_stream_[i], cudaStreamNonBlocking) != cudaSuccess ||
            cudaEventCreateWithFlags((cudaEvent_t*) &join_[i], cudaEventDisableTiming) != cudaSuccess ||
            cublasCreate(&a) != CUBLAS_STATUS_SUCCESS) {
            err = "prefill gemm: side streams";
            return false;
        }
        aux_handle_[i] = a;
        cublasSetStream(a, (cudaStream_t) aux_stream_[i]);
        cublasSetMathMode(a, CUBLAS_DEFAULT_MATH);
    }
    return true;
}

void Gemm::set_buffers(uint16_t* scratch, int64_t scratch_elems, void* workspace, size_t ws_bytes) {
    // a fixed workspace, so the handles never allocate on the way: a slice for each stream
    const int ns = aux_handle_[0] ? AUX + 1 : 1;
    const size_t slice = ws_bytes / (size_t) ns / 256 * 256;
    cublasSetWorkspace((cublasHandle_t) handle_, workspace, slice);
    for (int i = 0; i + 1 < ns; ++i)
        cublasSetWorkspace((cublasHandle_t) aux_handle_[i], (uint8_t*) workspace + (size_t) (i + 1) * slice, slice);
    scratch_ = scratch;
    scratch_elems_ = scratch_elems;
}

void Gemm::bf16(const uint16_t* X, const uint16_t* W, float* Y, int64_t T, int64_t N, int64_t K, int64_t ldy,
                float beta) {
    if (T <= 0 || N <= 0) return;
    if (ldy <= 0) ldy = N;
    const float alpha = 1.0f;
    // Column-major view: Y^T[N, T] = W[N, K] (stored K x N col-major, transposed) . X^T[K, T].
    ck(cublasGemmEx((cublasHandle_t) handle_, CUBLAS_OP_T, CUBLAS_OP_N, (int) N, (int) T, (int) K, &alpha, W,
                    CUDA_R_16BF, (int) K, X, CUDA_R_16BF, (int) K, &beta, Y, CUDA_R_32F, (int) ldy,
                    CUBLAS_COMPUTE_32F, CUBLAS_GEMM_DEFAULT),
       "cublasGemmEx");
}

void Gemm::f16(const uint16_t* X, const uint16_t* W, float* Y, int64_t T, int64_t N, int64_t K, int64_t ldy,
               float beta) {
    if (T <= 0 || N <= 0) return;
    if (ldy <= 0) ldy = N;
    const float alpha = 1.0f;
    ck(cublasGemmEx((cublasHandle_t) handle_, CUBLAS_OP_T, CUBLAS_OP_N, (int) N, (int) T, (int) K, &alpha, W,
                    CUDA_R_16F, (int) K, X, CUDA_R_16F, (int) K, &beta, Y, CUDA_R_32F, (int) ldy,
                    CUBLAS_COMPUTE_32F, CUBLAS_GEMM_DEFAULT),
       "cublasGemmEx f16");
}

void Gemm::f16_grouped(const uint16_t* const* X, const uint16_t* const* W, float* const* Y, const uint16_t* const* X_host,
                       const uint16_t* const* W_host, float* const* Y_host, const int* rows, int n, int64_t N, int64_t K) {
    if (n <= 0) return;
    if (grouped_) {
        // column-major, as f16(): Y_i^T[N, rows] = W_i (K x N col-major, transposed) . X_i^T[K, rows]; a group per problem
        const std::vector<cublasOperation_t> ta((size_t) n, CUBLAS_OP_T), tb((size_t) n, CUBLAS_OP_N);
        const std::vector<int> m((size_t) n, (int) N), k((size_t) n, (int) K), ld_wx((size_t) n, (int) K),
            ld_y((size_t) n, (int) N), size((size_t) n, 1);
        const std::vector<float> alpha((size_t) n, 1.0f), beta((size_t) n, 0.0f);
        const cublasStatus_t st = cublasGemmGroupedBatchedEx(
            (cublasHandle_t) handle_, ta.data(), tb.data(), m.data(), rows, k.data(), alpha.data(), (const void* const*) W,
            CUDA_R_16F, ld_wx.data(), (const void* const*) X, CUDA_R_16F, ld_wx.data(), beta.data(), (void* const*) Y,
            CUDA_R_32F, ld_y.data(), n, size.data(), CUBLAS_COMPUTE_32F);
        if (st == CUBLAS_STATUS_SUCCESS) return;
        if (st != CUBLAS_STATUS_NOT_SUPPORTED && st != CUBLAS_STATUS_INVALID_VALUE) ck(st, "cublasGemmGroupedBatchedEx");
        grouped_ = false;
        std::fprintf(stderr, "prefill gemm: cuBLAS does not group FP16 -> FP32 GEMMs here; one call per expert\n");
    }
    // round robin over the stream and its side streams, which join it again
    int64_t sum_rows = 0;
    for (int i = 0; i < n; ++i) sum_rows += rows[i];
    const int ns = aux_handle_[0] && sum_rows >= 128 * (int64_t) n ? std::min(n, AUX + 1) : 1;
    if (ns > 1) {
        cudaEventRecord((cudaEvent_t) fork_, (cudaStream_t) stream_);
        for (int s = 1; s < ns; ++s) cudaStreamWaitEvent((cudaStream_t) aux_stream_[s - 1], (cudaEvent_t) fork_, 0);
    }
    void* const main = handle_;
    for (int i = 0; i < n; ++i) {
        handle_ = i % ns == 0 ? main : aux_handle_[i % ns - 1];
        f16(X_host[i], W_host[i], Y_host[i], rows[i], N, K);
    }
    handle_ = main;
    for (int s = 1; s < ns; ++s) {
        cudaEventRecord((cudaEvent_t) join_[s - 1], (cudaStream_t) aux_stream_[s - 1]);
        cudaStreamWaitEvent((cudaStream_t) stream_, (cudaEvent_t) join_[s - 1], 0);
    }
}

void Gemm::native(const uint16_t* X, int ggml_type, const void* W_blocks, float* Y, int64_t T, int64_t N, int64_t K,
                  int64_t ldy, float beta) {
    if (N * K > scratch_elems_) {
        // Too large for the scratch at once: in row slices.
        const int64_t rows = scratch_elems_ / K;
        if (rows <= 0) { std::fprintf(stderr, "prefill gemm: scratch too small for K=%lld\n", (long long) K); std::exit(1); }
        if (ldy <= 0) ldy = N;
        for (int64_t r0 = 0; r0 < N; r0 += rows) {
            const int64_t n = (N - r0 < rows) ? N - r0 : rows;
            strata::kernels::dequant_f16(ggml_type, W_blocks, r0, n, K, scratch_, stream_);
            f16(X, scratch_, Y + r0, T, n, K, ldy, beta);
        }
        return;
    }
    strata::kernels::dequant_f16(ggml_type, W_blocks, 0, N, K, scratch_, stream_);
    f16(X, scratch_, Y, T, N, K, ldy, beta);
}

}  // namespace strata::prefill
