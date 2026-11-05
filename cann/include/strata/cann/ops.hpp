// cann/include/strata/cann/ops.hpp - the first CANN ops of the port, with a CPU oracle.
//
// Every function takes FP16 operands on the device (the format the 910B Cube unit wants) and
// has a matching `*_ref` CPU implementation in ops.cpp that the parity test compares against.
#pragma once

#include <cstdint>
#include <vector>

#include "strata/cann/device.hpp"

namespace strata::cann {

enum class DType { F16, F32 };

// A contiguous device tensor, at most 2D (the shapes the first ops need).
struct Tensor {
    void * data = nullptr;
    DType dtype = DType::F16;
    int64_t ne0 = 0; // columns (contiguous)
    int64_t ne1 = 1; // rows

    int64_t elements() const { return ne0 * ne1; }
    size_t bytes() const { return (size_t) elements() * (dtype == DType::F16 ? 2 : 4); }
};

// out[M,N] = a[M,K] * b[K,N]
void matmul(Stream & stream, const Tensor & a, const Tensor & b, Tensor & out);

// out[row,:] = x[row,:] * rsqrt(mean(x[row,:]^2) + eps) * gamma
void rms_norm(Stream & stream, const Tensor & x, const Tensor & gamma, double eps, Tensor & out);

// out = x * sigmoid(x)
void silu(Stream & stream, const Tensor & x, Tensor & out);

// out = a + b (same shape)
void add(Stream & stream, const Tensor & a, const Tensor & b, Tensor & out);

// out = exp(x), elementwise
void exp(Stream & stream, const Tensor & x, Tensor & out);

// out = x * scalar
void muls(Stream & stream, const Tensor & x, float scalar, Tensor & out);

// out = a - b (same shape)
void sub(Stream & stream, const Tensor & a, const Tensor & b, Tensor & out);

// out = a * b with numpy-style broadcast (e.g. [S,1] x [1,S] -> [S,S])
void mul(Stream & stream, const Tensor & a, const Tensor & b, Tensor & out);

// out[row,:] = softmax(x[row,:]) over the last axis
void softmax(Stream & stream, const Tensor & x, Tensor & out);

// ---- CPU oracle (float, row-major) ----

std::vector<float> matmul_ref(const std::vector<float> & a, const std::vector<float> & b, int64_t m, int64_t k,
                              int64_t n);
std::vector<float> rms_norm_ref(const std::vector<float> & x, const std::vector<float> & gamma, int64_t rows,
                                int64_t cols, double eps);
std::vector<float> silu_ref(const std::vector<float> & x);
std::vector<float> add_ref(const std::vector<float> & a, const std::vector<float> & b);
std::vector<float> softmax_ref(const std::vector<float> & x, int64_t rows, int64_t cols);

// ---- fp16 helpers (host side) ----

void f32_to_f16(const float * src, uint16_t * dst, size_t n);
void f16_to_f32(const uint16_t * src, float * dst, size_t n);

} // namespace strata::cann
