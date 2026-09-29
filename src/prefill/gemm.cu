// src/prefill/gemm.cu - see include/strata/prefill/gemm.hpp.
#include "strata/prefill/gemm.hpp"
#include "strata/prefill/kernels.hpp"
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

// Volta (sm_70) has no BF16 path in cuBLAS.  A BF16 value is exact in FP16 (7 mantissa bits fit in 10), so the
// BF16 product runs identically on FP16 tensor cores: W is converted here, X must already be FP16 bits (the
// prompt producers write FP16 images on such devices).  The conversion rounds to FP16; every BF16 value in
// FP16's range (all model weights and activations) is represented exactly.
__global__ void bf16_to_f16_kernel(const uint16_t* __restrict__ in, uint16_t* __restrict__ out, int64_t n) {
    const int64_t i = (int64_t) blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;
    const uint32_t bits = (uint32_t) in[i] << 16;   // the exact f32 image of the bf16 value
    out[i] = __half_as_ushort(__float2half_rn(__uint_as_float(bits)));
}

void launch_bf16_to_f16(const uint16_t* in, uint16_t* out, int64_t n, cudaStream_t s) {
    if (n <= 0) return;
    const unsigned blocks = (unsigned) ((n + 255) / 256);
    bf16_to_f16_kernel<<<blocks, 256, 0, s>>>(in, out, n);
    const cudaError_t error = cudaGetLastError();
    if (error != cudaSuccess) {
        std::fprintf(stderr, "prefill gemm: bf16->fp16: %s\n", cudaGetErrorString(error));
        std::exit(1);
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
    if (w16_) cudaFree(w16_);
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
    strata::prefill::set_fp16_bits(!native_bf16_);
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
    strata::prefill::set_fp16_bits(!native_bf16_);
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
        // Volta: convert the BF16 W to FP16 (exact) and run an FP16 tensor-core GEMM; X is already FP16 bits
        // (the producers write FP16 images when `prefill::fp16_bits()`, which `init` set for this device).
        const int64_t w_elems = N * K;
        uint16_t* w16 = scratch_;
        if (w16 == nullptr || w_elems > scratch_elems_) {
            // No scratch (the parity test) or W larger than it: stage in an owned buffer.
            if (w_elems > w16_elems_) {
                if (w16_) cudaFree(w16_);
                w16_ = nullptr;
                w16_elems_ = 0;
                if (cudaMalloc(&w16_, (size_t) w_elems * 2) != cudaSuccess) {
                    std::fprintf(stderr, "prefill gemm: Volta W staging of %lld elems\n", (long long) w_elems);
                    std::exit(1);
                }
                w16_elems_ = w_elems;
            }
            w16 = w16_;
        }
        launch_bf16_to_f16(W, w16, w_elems, (cudaStream_t) stream_);
        ck(cublasGemmEx((cublasHandle_t) handle_, CUBLAS_OP_T, CUBLAS_OP_N, (int) N, (int) T, (int) K, &alpha, w16,
                        CUDA_R_16F, (int) K, X, CUDA_R_16F, (int) K, &beta, Y, CUDA_R_32F, (int) ldy,
                        CUBLAS_COMPUTE_32F, CUBLAS_GEMM_DEFAULT),
           "cublasGemmEx fp16 (Volta bf16 path)");
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
