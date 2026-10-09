// src/kernels/cuda/exl3_experts.cu - one layer's routed EXL3 experts on the GPU (see the header).
#include "strata/kernels/exl3_experts.hpp"
#include "strata/artifact/exl3_model.hpp"

#include <cuda_runtime.h>
#include <cuda_fp16.h>

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


// ================================ streaming experts ================================

namespace {

__global__ void f16_to_f32_rows_kernel(const uint16_t* __restrict__ h, float* __restrict__ y, long n) {
    long i = (long) blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) y[i] = __half2float(*(const __half*) &h[i]);
}

__global__ void f32_to_f16_rows_kernel(const float* __restrict__ x, uint16_t* __restrict__ h, long n) {
    long i = (long) blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) h[i] = __half_as_ushort(__float2half_rn(x[i]));
}

void upload16(uint16_t* dst, const uint16_t* src, size_t bytes, cudaStream_t st) {
    if (dst == nullptr || bytes == 0) return;
    (void) cudaMemcpyAsync(dst, src, bytes, cudaMemcpyHostToDevice, st);
}

}  // namespace

struct Exl3ExpertStream::Impl {
    std::string dir;
    std::unique_ptr<Exl3Model> model;
    int n_embd = 2560;
    int max_k = 64;
    uint16_t* d_x = nullptr;
    uint16_t* d_rows = nullptr;
    float* d_out = nullptr;

    struct Slot { uint16_t* tr = nullptr; uint16_t* suh = nullptr; uint16_t* svh = nullptr;
                  size_t trb = 0, suhb = 0, svhb = 0; };
    std::vector<Slot> g, u, d;
    std::vector<Exl3Mat> gm, um, dm;

    ~Impl() {
        for (void* p : { (void*) d_x, (void*) d_rows, (void*) d_out }) (void) cudaFree(p);
        for (auto& v : { &g, &u, &d })
            for (Impl::Slot& s : *v) { (void) cudaFree(s.tr); (void) cudaFree(s.suh); (void) cudaFree(s.svh); }
    }
};

Exl3ExpertStream::Exl3ExpertStream(const std::string& model_dir, int n_embd, void* stream) {
    impl_ = new Impl;
    impl_->dir = model_dir;
    impl_->model = std::make_unique<Exl3Model>(model_dir);
    impl_->n_embd = n_embd;
    const int mk = impl_->max_k;
    auto alloc = [](uint16_t** p, size_t bytes) { if (cudaMalloc((void**) p, bytes ? bytes : 1) != cudaSuccess) std::abort(); };
    alloc(&impl_->d_x, (size_t) n_embd * 2);
    alloc(&impl_->d_rows, (size_t) mk * n_embd * 2);
    if (cudaMalloc((void**) &impl_->d_out, (size_t) mk * n_embd * 4) != cudaSuccess) std::abort();
    // size the per-slot weight buffers from expert 0 of layer 0 (all experts in a layer share a shape)
    const std::string b0 = "model.language_model.layers.0.mlp.experts.0.";
    auto setup = [&](std::vector<Impl::Slot>& slots, std::vector<Exl3Mat>& mats, const char* proj) {
        Exl3Linear L = impl_->model->linear(b0 + proj);
        const size_t trb = (size_t) L.ki * L.nj * (256 * L.bits / 16) * 2;
        const size_t suhb = (size_t) L.in() * 2, svhb = (size_t) L.out() * 2;
        slots.resize(mk); mats.resize(mk);
        for (int i = 0; i < mk; ++i) {
            alloc(&slots[i].tr, trb); alloc(&slots[i].suh, suhb); alloc(&slots[i].svh, svhb);
            slots[i].trb = trb; slots[i].suhb = suhb; slots[i].svhb = svhb;
            mats[i].trellis = slots[i].tr; mats[i].suh = slots[i].suh; mats[i].svh = slots[i].svh;
            mats[i].ki = L.ki; mats[i].nj = L.nj; mats[i].bits = L.bits; mats[i].cb = L.cb;
        }
    };
    setup(impl_->g, impl_->gm, "gate_proj");
    setup(impl_->u, impl_->um, "up_proj");
    setup(impl_->d, impl_->dm, "down_proj");
    (void) stream;
}

Exl3ExpertStream::~Exl3ExpertStream() { delete impl_; }

size_t Exl3ExpertStream::bytes() const {
    size_t n = (size_t) impl_->n_embd * 2 + (size_t) impl_->max_k * impl_->n_embd * (2 + 4);
    for (auto& s : impl_->g) n += s.trb + s.suhb + s.svhb;
    for (auto& s : impl_->u) n += s.trb + s.suhb + s.svhb;
    for (auto& s : impl_->d) n += s.trb + s.suhb + s.svhb;
    return n;
}

bool Exl3ExpertStream::run(int layer, const float* x_f32, const int* ids, int k, float* out, void* stream) {
    if (k <= 0) return true;
    if (k > impl_->max_k) return false;
    cudaStream_t st = (cudaStream_t) stream;
    const long n = impl_->n_embd;
    f32_to_f16_rows_kernel<<<(unsigned)((n + 255) / 256), 256, 0, st>>>(x_f32, impl_->d_x, n);
    const std::string pfx = "model.language_model.layers." + std::to_string(layer) + ".mlp.experts.";
    for (int i = 0; i < k; ++i) {
        const std::string b = pfx + std::to_string(ids[i]) + ".";
        Exl3Linear Lg = impl_->model->linear(b + "gate_proj");
        Exl3Linear Lu = impl_->model->linear(b + "up_proj");
        Exl3Linear Ld = impl_->model->linear(b + "down_proj");
        upload16(impl_->g[i].tr, Lg.trellis, impl_->g[i].trb, st);
        upload16(impl_->g[i].suh, Lg.suh, impl_->g[i].suhb, st);
        upload16(impl_->g[i].svh, Lg.svh, impl_->g[i].svhb, st);
        upload16(impl_->u[i].tr, Lu.trellis, impl_->u[i].trb, st);
        upload16(impl_->u[i].suh, Lu.suh, impl_->u[i].suhb, st);
        upload16(impl_->u[i].svh, Lu.svh, impl_->u[i].svhb, st);
        upload16(impl_->d[i].tr, Ld.trellis, impl_->d[i].trb, st);
        upload16(impl_->d[i].suh, Ld.suh, impl_->d[i].suhb, st);
        upload16(impl_->d[i].svh, Ld.svh, impl_->d[i].svhb, st);
    }
    exl3_moe_rows(impl_->gm.data(), impl_->um.data(), impl_->dm.data(), ids, k, impl_->d_x, impl_->d_rows, stream);
    f16_to_f32_rows_kernel<<<(unsigned)((n * k + 255) / 256), 256, 0, st>>>(impl_->d_rows, impl_->d_out, n * k);
    if (cudaMemcpyAsync(out, impl_->d_out, (size_t) n * k * 4, cudaMemcpyDeviceToHost, st) != cudaSuccess) return false;
    const cudaError_t se = cudaStreamSynchronize(st);
    if (se != cudaSuccess) std::fprintf(stderr, "exl3 stream run(layer=%d): %s\n", layer, cudaGetErrorString(se));
    return true;
}

}  // namespace strata::kernels
