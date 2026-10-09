// src/kernels/cuda/exl3_experts.cu - one layer's routed EXL3 experts on the GPU (see the header).
#include "strata/kernels/exl3_experts.hpp"
#include "strata/artifact/exl3_model.hpp"

#include <cuda_runtime.h>

#include <cstdio>
#include <cstdlib>
#include <string>
#include <vector>

namespace strata::kernels {

struct Exl3ExpertStore::Impl {
    std::vector<Exl3Mat> gate, up, down;
    std::vector<void*> allocs;
};

namespace {

Exl3Mat upload(const Exl3Linear& L, std::vector<void*>& allocs, cudaStream_t st, size_t& bytes) {
    auto alloc = [&](size_t n, const void* src) -> void* {
        void* p = nullptr;
        if (cudaMalloc(&p, n) != cudaSuccess) { std::fprintf(stderr, "exl3 store: cudaMalloc failed\n"); std::abort(); }
        (void) st;
        if (cudaMemcpy(p, src, n, cudaMemcpyHostToDevice) != cudaSuccess) { std::fprintf(stderr, "exl3 store: memcpy failed\n"); std::abort(); }
        allocs.push_back(p);
        bytes += n;
        return p;
    };
    int words = 256 * L.bits / 16;
    size_t tr_bytes = (size_t) L.ki * L.nj * words * 2;
    Exl3Mat M;
    M.trellis = (const uint16_t*) alloc(tr_bytes, L.trellis);
    M.suh = (const uint16_t*) alloc((size_t) L.in() * 2, L.suh);
    M.svh = (const uint16_t*) alloc((size_t) L.out() * 2, L.svh);
    M.ki = L.ki; M.nj = L.nj; M.bits = L.bits; M.cb = L.cb;
    return M;
}

}  // namespace

Exl3ExpertStore::Exl3ExpertStore(const std::string& model_dir, int layer, int n_experts, void* stream) {
    impl_ = new Impl;
    n_experts_ = n_experts;
    cudaStream_t st = (cudaStream_t) stream;
    Exl3Model m(model_dir);
    const std::string prefix = "model.language_model.layers." + std::to_string(layer) + ".mlp.experts.";
    for (int e = 0; e < n_experts; ++e) {
        const std::string b = prefix + std::to_string(e);
        impl_->gate.push_back(upload(m.linear(b + ".gate_proj"), impl_->allocs, st, bytes_));
        impl_->up.push_back(upload(m.linear(b + ".up_proj"), impl_->allocs, st, bytes_));
        impl_->down.push_back(upload(m.linear(b + ".down_proj"), impl_->allocs, st, bytes_));
    }
}

Exl3ExpertStore::~Exl3ExpertStore() {
    if (impl_) {
        for (void* p : impl_->allocs) (void) cudaFree(p);
        delete impl_;
    }
}

void Exl3ExpertStore::run(const uint16_t* x, const int* ids, const float* weights, int k, uint16_t* out,
                          void* stream) const {
    std::vector<Exl3Mat> g(k), u(k), d(k);
    for (int i = 0; i < k; ++i) {
        int e = ids[i];
        g[i] = impl_->gate[e];
        u[i] = impl_->up[e];
        d[i] = impl_->down[e];
    }
    exl3_moe_ffn(g.data(), u.data(), d.data(), weights, k, x, out, stream);
}

}  // namespace strata::kernels
