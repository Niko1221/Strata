#pragma once
#include "strata/glm/experts.hpp"
#include <memory>
#include <string>

namespace strata::glm {
struct CudaStats { uint64_t hits = 0, requests = 0, promotions = 0; };
// All methods run on the model thread. Synchronous return keeps host weight lifetimes explicit.
class CudaBackend {
public:
    CudaBackend();
    ~CudaBackend();
    bool init(int device, int hidden, int intermediate, uint64_t budget, uint64_t reserve,
              int promote_after, std::string& err);
    bool contains(int layer, int expert) const;
    void promote(int layer, int expert, const ExpertView& v);
    void expert(int layer, int expert, const ExpertView* host, const float* x, int rows, float* y);
    void gemm(const Q4& w, const float* x, int rows, float* y);
    void attention(const Q4& kv_b, const float* q, const float* kv, int rows, int pos0,
                   int heads, int nope, int rope, int value, float* ctx);
    int slots() const;
    uint64_t bytes() const;
    const CudaStats& stats() const;
private:
    struct Impl;
    std::unique_ptr<Impl> p_;
};
} // namespace strata::glm
