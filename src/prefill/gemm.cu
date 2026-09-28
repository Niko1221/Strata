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

__global__ void bf16_to_f32_kernel(const uint16_t* in, float* out, size_t n) {
    const size_t i = (size_t) blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) out[i] = __uint_as_float(((unsigned) in[i]) << 16);
}

static void bf16_to_f32(const uint16_t* in, float* out, size_t n, void* stream) {
    if (n == 0) return;
    const unsigned threads = 256;
    const size_t blocks = (n + threads - 1) / threads;
    bf16_to_f32_kernel<<<(unsigned) blocks, threads, 0, (cudaStream_t) stream>>>(in, out, n);
}

}  // namespace

Gemm::~Gemm() {
    if (handle_) cublasDestroy((cublasHandle_t) handle_);
    for (auto& kv : w32_cache_)
        if (kv.second.buf) cudaFree(kv.second.buf);
    if (x32_) cudaFree(x32_);
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
    static int cc_major = -1;
    if (cc_major < 0) {
        int dev = 0;
        cc_major = 99;  // on a failed query assume Ampere and let cuBLAS speak for itself
        if (cudaGetDevice(&dev) == cudaSuccess &&
            cudaDeviceGetAttribute(&cc_major, cudaDevAttrComputeCapabilityMajor, dev) != cudaSuccess)
            cc_major = 99;
    }
    if (cc_major < 70) {
        // Pascal port: cuBLAS has no BF16 GEMM below sm_80.  BF16 -> FP32 is a lossless 16-bit shift per
        // element; convert the weight once per device pointer (weights are resident in the arena and their
        // pointers stable) and the activations per call, then run the same GEMM over FP32 operands.
        float* wf = nullptr;
        const auto it = w32_cache_.find(W);
        if (it != w32_cache_.end() && it->second.elems == N * K) {
            wf = it->second.buf;
        } else {
            w32_cache_.erase(W);
            if (cudaMalloc((void**) &wf, (size_t) N * K * sizeof(float)) != cudaSuccess || wf == nullptr) {
                std::fprintf(stderr, "prefill gemm: bf16->f32 weight cache of %lld KiB\n",
                             (long long) ((N * K * sizeof(float)) >> 10));
                std::exit(1);
            }
            w32_cache_[W] = {wf, N * K};
            bf16_to_f32(W, wf, (size_t) N * K, stream_);
        }
        const int64_t x_elems = T * K;
        if (x_elems > x32_elems_) {
            if (x32_) cudaFree(x32_);
            if (cudaMalloc((void**) &x32_, (size_t) x_elems * sizeof(float)) != cudaSuccess || x32_ == nullptr) {
                std::fprintf(stderr, "prefill gemm: bf16->f32 activation buffer of %lld KiB\n",
                             (long long) ((x_elems * sizeof(float)) >> 10));
                std::exit(1);
            }
            x32_elems_ = x_elems;
        }
        bf16_to_f32(X, x32_, (size_t) x_elems, stream_);
        const float alpha = 1.0f;
        // Column-major view as below; all operands FP32 here.
        ck(cublasGemmEx((cublasHandle_t) handle_, CUBLAS_OP_T, CUBLAS_OP_N, (int) N, (int) T, (int) K, &alpha, wf,
                        CUDA_R_32F, (int) K, x32_, CUDA_R_32F, (int) K, &beta, Y, CUDA_R_32F, (int) ldy,
                        CUBLAS_COMPUTE_32F, CUBLAS_GEMM_DEFAULT),
           "cublasGemmEx bf16->f32 (pre-sm_80)");
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
