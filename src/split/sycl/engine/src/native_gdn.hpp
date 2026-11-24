#pragma once
#include "kernels.hpp"

#include <vector>

namespace fnx {
class NativeGdn {
public:
    static constexpr int nk = 8, nv = 24, hs = 128;
    NativeGdn(dpct::queue_ptr queue, int rows);
    ~NativeGdn();
    NativeGdn(const NativeGdn&) = delete;
    NativeGdn& operator=(const NativeGdn&) = delete;
    size_t bytes() const { return bytes_; }
    void run(const float* qkvz, const float* ab, float* history,
             const float* conv, const float* dt, const float* a,
             const float* norm, float* state, const int* ctl, int groups,
             int rows, float eps, float* out);
private:
    template<class T> T* alloc(size_t n) {
        auto* p = sycl::malloc_device<T>(n, *queue_);
        if (!p) throw std::bad_alloc();
        buffers_.push_back(p);
        bytes_ += n * sizeof(T);
        return p;
    }
    dpct::queue_ptr queue_;
    int rows_;
    size_t bytes_ = 0;
    std::vector<void*> buffers_;
    float *qkv_, *z_, *h_, *gate_, *beta_, *history_, *state_;
    uint16_t* output16_;
};
}
