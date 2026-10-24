#pragma once
#include <cstddef>
#include <memory>
namespace strata::muse {
// Input is a decoded ROW TILE, never a complete model matrix or expert.
class CudaMatvec {
public:
    explicit CudaMatvec(int device);
    ~CudaMatvec();
    CudaMatvec(const CudaMatvec&) = delete;
    CudaMatvec& operator=(const CudaMatvec&) = delete;
    void multiply(const float* tile, const float* x, float* y, size_t cols, size_t rows);
private:
    struct Impl;
    std::unique_ptr<Impl> impl_;
};
}
