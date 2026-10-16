#include "cuda_matvec.hpp"
#include "strata/muse/model.hpp"
#include <cuda_runtime_api.h>
#include <cublas_v2.h>
#include <stdexcept>
#include <string>

namespace strata::muse {
namespace {
void cuda_check(cudaError_t e) {
    if (e != cudaSuccess) throw std::runtime_error(std::string("Muse CUDA: ") + cudaGetErrorString(e));
}
void blas_check(cublasStatus_t e) {
    if (e != CUBLAS_STATUS_SUCCESS) throw std::runtime_error("Muse cuBLAS error " + std::to_string(int(e)));
}
}
struct CudaMatvec::Impl {
    int device;
    cublasHandle_t handle = nullptr;
    float *tile = nullptr, *x = nullptr, *y = nullptr;
    explicit Impl(int d) : device(d) {}
    ~Impl() {
        // Construction failures also release any successful partial allocations.
        cudaSetDevice(device);
        if (handle) cublasDestroy(handle);
        if (y) cudaFree(y);
        if (x) cudaFree(x);
        if (tile) cudaFree(tile);
    }
};
CudaMatvec::CudaMatvec(int device) : impl_(new Impl(device)) {
    static_assert(2 * tile_bytes + 64 * sizeof(float) < device_budget_bytes, "bounded CUDA scratch");
    cuda_check(cudaSetDevice(device));
    blas_check(cublasCreate(&impl_->handle));
    blas_check(cublasSetMathMode(impl_->handle, CUBLAS_DEFAULT_MATH));
    cuda_check(cudaMalloc(reinterpret_cast<void**>(&impl_->tile), tile_bytes));
    cuda_check(cudaMalloc(reinterpret_cast<void**>(&impl_->x), tile_bytes));
    cuda_check(cudaMalloc(reinterpret_cast<void**>(&impl_->y), 64 * sizeof(float)));
}
CudaMatvec::~CudaMatvec() = default;
void CudaMatvec::multiply(const float* tile, const float* x, float* y, size_t cols, size_t rows) {
    if (!cols || !rows || cols > tile_bytes / sizeof(float) || rows > 64 ||
        checked_mul(checked_mul(cols, rows), sizeof(float)) > tile_bytes)
        throw std::runtime_error("Muse CUDA tile exceeds bounded scratch");
    cuda_check(cudaSetDevice(impl_->device));
    cuda_check(cudaMemcpy(impl_->tile, tile, cols * rows * sizeof(float), cudaMemcpyHostToDevice));
    cuda_check(cudaMemcpy(impl_->x, x, cols * sizeof(float), cudaMemcpyHostToDevice));
    const float alpha = 1, beta = 0;
    // Row-major [rows,cols] is column-major [cols,rows], transposed for GEMV.
    blas_check(cublasSgemv(impl_->handle, CUBLAS_OP_T, int(cols), int(rows), &alpha,
                          impl_->tile, int(cols), impl_->x, 1, &beta, impl_->y, 1));
    cuda_check(cudaMemcpy(y, impl_->y, rows * sizeof(float), cudaMemcpyDeviceToHost));
}
}
