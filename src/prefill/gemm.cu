// src/prefill/gemm.cu - see include/strata/prefill/gemm.hpp.
#include "strata/prefill/gemm.hpp"
#include "strata/kernels/dequant_bf16.hpp"

#include <cublas_v2.h>
#include <cuda_runtime.h>

#include <cstdio>
#include <cstdlib>

namespace strata::prefill {
namespace {

void ck(cublasStatus_t s, const char* what) {
    if (s != CUBLAS_STATUS_SUCCESS) {
        std::fprintf(stderr, "prefill gemm: %s: cuBLAS status %d\n", what, (int) s);
        std::exit(1);
    }
}

__device__ __forceinline__ float bf16_to_float(uint16_t value) {
    return __uint_as_float(static_cast<uint32_t>(value) << 16);
}

__global__ void bf16_gemm_fallback(const uint16_t* __restrict__ x, const uint16_t* __restrict__ w,
                                   float* __restrict__ y, int t_count, int n_count, int k_count, int ldy,
                                   float beta) {
    constexpr int tile_size = 16;
    __shared__ float x_tile[tile_size][tile_size];
    __shared__ float w_tile[tile_size][tile_size];
    const int t = blockIdx.y * tile_size + threadIdx.y;
    const int n = blockIdx.x * tile_size + threadIdx.x;
    float sum = 0.0f;
    for (int k0 = 0; k0 < k_count; k0 += tile_size) {
        const int xk = k0 + threadIdx.x;
        const int wk = k0 + threadIdx.y;
        x_tile[threadIdx.y][threadIdx.x] =
            t < t_count && xk < k_count ? bf16_to_float(x[(size_t) t * k_count + xk]) : 0.0f;
        w_tile[threadIdx.y][threadIdx.x] =
            n < n_count && wk < k_count ? bf16_to_float(w[(size_t) n * k_count + wk]) : 0.0f;
        __syncthreads();
#pragma unroll
        for (int k = 0; k < tile_size; ++k) sum = fmaf(x_tile[threadIdx.y][k], w_tile[k][threadIdx.x], sum);
        __syncthreads();
    }
    if (t < t_count && n < n_count) {
        float* out = y + (size_t) t * ldy + n;
        *out = beta == 0.0f ? sum : fmaf(beta, *out, sum);
    }
}

bool device_has_native_bf16() {
    int ordinal = 0;
    cudaDeviceProp properties{};
    return cudaGetDevice(&ordinal) == cudaSuccess &&
           cudaGetDeviceProperties(&properties, ordinal) == cudaSuccess && properties.major >= 8;
}

}  // namespace

Gemm::~Gemm() {
    if (handle_) cublasDestroy((cublasHandle_t) handle_);
    if (!external_) {
        if (scratch_) cudaFree(scratch_);
        if (workspace_) cudaFree(workspace_);
    }
}

bool Gemm::init_external(void* stream, uint16_t* scratch, int64_t scratch_elems, void* workspace, size_t ws_bytes,
                         std::string& err) {
    cublasHandle_t h = nullptr;
    if (cublasCreate(&h) != CUBLAS_STATUS_SUCCESS) { err = "prefill gemm: cublasCreate failed"; return false; }
    handle_ = h;
    stream_ = stream;
    external_ = true;
    cublasSetStream(h, (cudaStream_t) stream);
    workspace_ = workspace;
    cublasSetWorkspace(h, workspace_, ws_bytes);
    cublasSetMathMode(h, CUBLAS_DEFAULT_MATH);
    native_bf16_ = device_has_native_bf16();
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
    native_bf16_ = device_has_native_bf16();
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
    const float alpha = 1.0f;
    if (!native_bf16_) {
        const dim3 threads(16, 16);
        const dim3 blocks((unsigned) ((N + 15) / 16), (unsigned) ((T + 15) / 16));
        bf16_gemm_fallback<<<blocks, threads, 0, (cudaStream_t) stream_>>>(
            X, W, Y, (int) T, (int) N, (int) K, (int) ldy, beta);
        const cudaError_t error = cudaGetLastError();
        if (error != cudaSuccess) {
            std::fprintf(stderr, "prefill gemm: Volta BF16 fallback: %s\n", cudaGetErrorString(error));
            std::exit(1);
        }
        return;
    }
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
