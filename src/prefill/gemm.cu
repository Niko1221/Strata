// src/prefill/gemm.cu - see include/strata/prefill/gemm.hpp.
#include "strata/prefill/gemm.hpp"
#include "strata/kernels/dequant_bf16.hpp"

#include <cublas_v2.h>
#include <cuda_runtime.h>

#include <algorithm>
#include <cstdio>
#include <cstdlib>
#include <stdexcept>

namespace strata::prefill {
namespace {

__global__ void bf16_expand(const uint16_t* src, float* dst, int64_t n) {
    const int64_t i = int64_t(blockIdx.x) * blockDim.x + threadIdx.x;
    if (i < n) dst[i] = __uint_as_float(uint32_t(src[i]) << 16);
}

// Narrow products and low-memory fallback: retain BF16's exponent range with FP32 arithmetic.
__global__ void bf16_simt(const uint16_t* X, const uint16_t* W, float* Y,
                         int64_t T, int64_t N, int64_t K, int64_t ldy, float beta) {
    __shared__ float a[16][16], b[16][16];
    const int tx = threadIdx.x, ty = threadIdx.y;
    const int64_t row = int64_t(blockIdx.y) * 16 + ty;
    const int64_t col = int64_t(blockIdx.x) * 16 + tx;
    float sum = 0.0f;
    for (int64_t k = 0; k < K; k += 16) {
        a[ty][tx] = row < T && k + tx < K ? __uint_as_float(uint32_t(X[row * K + k + tx]) << 16) : 0.0f;
        b[ty][tx] = col < N && k + ty < K ? __uint_as_float(uint32_t(W[col * K + k + ty]) << 16) : 0.0f;
        __syncthreads();
        for (int j = 0; j < 16; ++j) sum = fmaf(a[ty][j], b[j][tx], sum);
        __syncthreads();
    }
    if (row < T && col < N) {
        const int64_t i = row * ldy + col;
        Y[i] = beta == 0.0f ? sum : fmaf(beta, Y[i], sum);
    }
}

bool supports_bf16() {
    int device = 0;
    cudaDeviceProp prop{};
    if (cudaGetDevice(&device) != cudaSuccess || cudaGetDeviceProperties(&prop, device) != cudaSuccess)
        throw std::runtime_error("prefill gemm: cannot query CUDA device");
    return prop.major >= 8;
}

void ck(cublasStatus_t s, const char* what) {
    if (s != CUBLAS_STATUS_SUCCESS) {
        std::fprintf(stderr, "prefill gemm: %s: cuBLAS status %d\n", what, (int) s);
        std::exit(1);
    }
}

}  // namespace

Gemm::~Gemm() {
    if (handle_) cublasDestroy((cublasHandle_t) handle_);
    if (fp32_) cudaFree(fp32_);
    if (!external_) {
        if (scratch_) cudaFree(scratch_);
        if (workspace_) cudaFree(workspace_);
    }
}

bool Gemm::init_external(void* stream, uint16_t* scratch, int64_t scratch_elems, void* workspace, size_t ws_bytes,
                         std::string& err) {
    native_bf16_ = supports_bf16();
    cublasHandle_t h = nullptr;
    if (cublasCreate(&h) != CUBLAS_STATUS_SUCCESS) { err = "prefill gemm: cublasCreate failed"; return false; }
    handle_ = h;
    stream_ = stream;
    external_ = true;
    cublasSetStream(h, (cudaStream_t) stream);
    workspace_ = workspace;
    cublasSetWorkspace(h, workspace_, ws_bytes);
    cublasSetMathMode(h, CUBLAS_DEFAULT_MATH);
    scratch_ = scratch;
    scratch_elems_ = scratch_elems;
    return true;
}

void Gemm::rebind(uint16_t* scratch, int64_t scratch_elems, void* workspace, size_t ws_bytes) {
    scratch_ = scratch;
    scratch_elems_ = scratch_elems;
    workspace_ = workspace;
    cublasSetWorkspace((cublasHandle_t) handle_, workspace_, ws_bytes);
}

bool Gemm::init(void* stream, int64_t scratch_elems, std::string& err) {
    native_bf16_ = supports_bf16();
    cublasHandle_t h = nullptr;
    if (cublasCreate(&h) != CUBLAS_STATUS_SUCCESS) { err = "prefill gemm: cublasCreate failed"; return false; }
    handle_ = h;
    stream_ = stream;
    cublasSetStream(h, (cudaStream_t) stream);
    // A fixed workspace so the handle never allocates on the way (and graphs could capture it later).
    const size_t ws = 32u << 20;
    if (cudaMalloc(&workspace_, ws) != cudaSuccess) { err = "prefill gemm: workspace"; return false; }
    cublasSetWorkspace(h, workspace_, ws);
    cublasSetMathMode(h, CUBLAS_DEFAULT_MATH);
    if (scratch_elems > 0 && cudaMalloc((void**) &scratch_, (size_t) scratch_elems * 2) != cudaSuccess) {
        err = "prefill gemm: dequant scratch of " + std::to_string(scratch_elems * 2 >> 20) + " MiB";
        return false;
    }
    scratch_elems_ = scratch_elems;
    return true;
}

void Gemm::bf16(const uint16_t* X, const uint16_t* W, float* Y, int64_t T, int64_t N, int64_t K, int64_t ldy,
                float beta) {
    if (T <= 0 || N <= 0) return;
    if (ldy <= 0) ldy = N;
    if (!native_bf16_) {
        // Volta/Turing lack native BF16 GEMM. Expand exactly, then use SGEMM for wide products.
        // Cap temporary storage at 64 MiB per Gemm instance; keep narrow products on SIMT.
        constexpr int64_t capacity = (64ll << 20) / sizeof(float);
        const int64_t rows = K > 0 ? std::min<int64_t>(T, std::min<int64_t>(1024, capacity / K / 2)) : 0;
        if (N >= 32 && rows > 0) {
            if (!fp32_) {
                if (cudaMalloc((void**) &fp32_, capacity * sizeof(float)) != cudaSuccess)
                    cudaGetLastError(); // Optional allocation: use SIMT if it does not fit.
            }
            if (fp32_) {
                float* xf = fp32_;
                float* wf = fp32_ + rows * K;
                const int64_t columns = (capacity - rows * K) / K;
                const float alpha = 1.0f;
                for (int64_t t0 = 0; t0 < T; t0 += rows) {
                    const int64_t t = std::min(rows, T - t0);
                    bf16_expand<<<(unsigned) ((t * K + 255) / 256), 256, 0, (cudaStream_t) stream_>>>(X + t0 * K, xf, t * K);
                    for (int64_t n0 = 0; n0 < N; n0 += columns) {
                        const int64_t n = std::min(columns, N - n0);
                        bf16_expand<<<(unsigned) ((n * K + 255) / 256), 256, 0, (cudaStream_t) stream_>>>(W + n0 * K, wf, n * K);
                        ck(cublasSgemm((cublasHandle_t) handle_, CUBLAS_OP_T, CUBLAS_OP_N, (int) n, (int) t, (int) K,
                                      &alpha, wf, (int) K, xf, (int) K, &beta, Y + t0 * ldy + n0, (int) ldy),
                           "BF16 via SGEMM");
                    }
                }
                if (cudaGetLastError() != cudaSuccess) throw std::runtime_error("prefill gemm: BF16 expansion failed");
                return;
            }
        }
        bf16_simt<<<dim3((N + 15) / 16, (T + 15) / 16), dim3(16, 16), 0, (cudaStream_t) stream_>>>(
            X, W, Y, T, N, K, ldy, beta);
        if (cudaGetLastError() != cudaSuccess) throw std::runtime_error("prefill gemm: BF16 fallback launch failed");
        return;
    }
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
