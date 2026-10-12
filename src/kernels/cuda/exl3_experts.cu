// src/kernels/cuda/exl3_experts.cu - one layer's routed EXL3 experts on the GPU (see the header).
#include "strata/kernels/exl3_experts.hpp"
#include "strata/artifact/exl3_model.hpp"

#include <cuda_runtime.h>
#include <cuda_fp16.h>

#include <algorithm>
#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <string>
#include <unordered_map>
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

void upload16(uint16_t* dst, const uint16_t* src, size_t bytes, cudaStream_t st) {
    if (dst == nullptr || bytes == 0) return;
    (void) cudaMemcpyAsync(dst, src, bytes, cudaMemcpyHostToDevice, st);
}

__global__ void f16_to_f32_rows_kernel(const uint16_t* __restrict__ h, float* __restrict__ y, long n) {
    long i = (long) blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) y[i] = __half2float(*(const __half*) &h[i]);
}

__global__ void f32_to_f16_rows_kernel(const float* __restrict__ x, uint16_t* __restrict__ h, long n) {
    long i = (long) blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) h[i] = __half_as_ushort(__float2half_rn(x[i]));
}

}  // namespace

struct Exl3ExpertStream::Impl {
    std::string dir;
    std::unique_ptr<Exl3Model> model;
    int n_embd = 2560;
    int n_layers = 48;
    int max_k = 64;
    int cache = 0;                     // slots per layer; 0 = the per-token scratch
    uint16_t* d_x = nullptr;
    uint16_t* d_rows = nullptr;
    float* d_out = nullptr;

    struct Slot { uint16_t* tr = nullptr; uint16_t* suh = nullptr; uint16_t* svh = nullptr;
                  size_t trb = 0, suhb = 0, svhb = 0; };
    struct Blocks { std::vector<Slot> g, u, d; std::vector<Exl3Mat> gm, um, dm; };

    Blocks scratch;                    // cache == 0: `max_k` slots, every use overwrites them
    struct Layer : Blocks {
        std::vector<void*> allocs;         // one block per (projection, plane), `cache` slots in it
        std::unordered_map<int, int> at;   // expert -> slot
        std::vector<uint32_t> tick;        // per slot: the last use, for the LRU
        int filled = 0;                    // slots handed out so far; the map's size does not track it after an eviction
        uint32_t clock = 0;
    };
    std::vector<Layer> layers;         // cache > 0: `cache` slots per layer, resident across tokens

    // STRATA_EXL3_TIME=1: per-token upload/ffn/d2h milliseconds and the cache hit rate
    long long hits = 0, misses = 0;
    double up_ms = 0, ffn_ms = 0, d2h_ms = 0, call_ms = 0;

    ~Impl() {
        for (void* p : { (void*) d_x, (void*) d_rows, (void*) d_out }) (void) cudaFree(p);
        for (auto& v : { &scratch.g, &scratch.u, &scratch.d })
            for (Slot& s : *v) { (void) cudaFree(s.tr); (void) cudaFree(s.suh); (void) cudaFree(s.svh); }
        for (Layer& l : layers)
            for (void* p : l.allocs) (void) cudaFree(p);
    }
};

Exl3ExpertStream::Exl3ExpertStream(const std::string& model_dir, int n_embd, int n_layers, int slots_per_layer,
                                   void* stream) {
    impl_ = new Impl;
    impl_->dir = model_dir;
    impl_->model = std::make_unique<Exl3Model>(model_dir);
    impl_->n_embd = n_embd;
    impl_->n_layers = n_layers;
    const int mk = impl_->max_k;
    auto alloc = [](uint16_t** p, size_t bytes) { if (cudaMalloc((void**) p, bytes ? bytes : 1) != cudaSuccess) std::abort(); };
    alloc(&impl_->d_x, (size_t) n_embd * 2);
    alloc(&impl_->d_rows, (size_t) mk * n_embd * 2);
    if (cudaMalloc((void**) &impl_->d_out, (size_t) mk * n_embd * 4) != cudaSuccess) std::abort();
    // size the per-slot weight buffers from expert 0 of layer 0 (all experts in a layer share a shape)
    const std::string b0 = "model.language_model.layers.0.mlp.experts.0.";
    auto setup_scratch = [&](std::vector<Impl::Slot>& slots, std::vector<Exl3Mat>& mats, const char* proj) {
        Exl3Linear L = impl_->model->linear(b0 + proj);
        const size_t trb = (size_t) L.ki * L.nj * (256 * L.bits / 16) * 2;
        const size_t suhb = (size_t) L.in() * 2, svhb = (size_t) L.out() * 2;
        slots.resize((size_t) mk); mats.resize((size_t) mk);
        for (int i = 0; i < mk; ++i) {
            alloc(&slots[(size_t) i].tr, trb);
            alloc(&slots[(size_t) i].suh, suhb);
            alloc(&slots[(size_t) i].svh, svhb);
            slots[(size_t) i].trb = trb; slots[(size_t) i].suhb = suhb; slots[(size_t) i].svhb = svhb;
            mats[(size_t) i].trellis = slots[(size_t) i].tr;
            mats[(size_t) i].suh = slots[(size_t) i].suh;
            mats[(size_t) i].svh = slots[(size_t) i].svh;
            mats[(size_t) i].ki = L.ki; mats[(size_t) i].nj = L.nj; mats[(size_t) i].bits = L.bits; mats[(size_t) i].cb = L.cb;
        }
    };
    setup_scratch(impl_->scratch.g, impl_->scratch.gm, "gate_proj");
    setup_scratch(impl_->scratch.u, impl_->scratch.um, "up_proj");
    setup_scratch(impl_->scratch.d, impl_->scratch.dm, "down_proj");
    (void) stream;

    // ---- the resident cache.  `-1` sizes it from the free VRAM: one slot in every layer costs
    // `per_slot` bytes, and what is left after this instance's own buffers is the budget.  The GGUF
    // path's `--expert-cache` is a global count because its blobs are one pool; here the stream is per
    // layer, so the count is per layer.
    int slots = slots_per_layer;
    if (slots < 0) {
        size_t per_slot = 0;
        for (int l = 0; l < n_layers; ++l) {
            const std::string b = "model.language_model.layers." + std::to_string(l) + ".mlp.experts.0.";
            for (const char* p : {"gate_proj", "up_proj", "down_proj"}) {
                Exl3Linear L = impl_->model->linear(b + p);
                per_slot += (size_t) L.ki * L.nj * (256 * L.bits / 16) * 2 + (size_t) (L.in() + L.out()) * 2;
            }
        }
        size_t free_b = 0, total_b = 0;
        (void) cudaMemGetInfo(&free_b, &total_b);
        const size_t reserve = (size_t) 512 << 20;   // the session's buffers and the capture
        slots = (per_slot > 0 && free_b > reserve) ? (int) std::min<size_t>((free_b - reserve) / per_slot, 4096) : 0;
        std::fprintf(stderr, "strata generate: EXL3 expert cache auto: %.2f GiB free -> %d slots/layer\n",
                     (double) free_b / 1073741824.0, slots);
    }
    impl_->cache = slots > 0 ? slots : 0;
    if (impl_->cache > 0) {
        impl_->layers.resize((size_t) n_layers);
        for (int l = 0; l < n_layers; ++l) {
            Impl::Layer& L = impl_->layers[(size_t) l];
            L.tick.assign((size_t) impl_->cache, 0);
            const std::string b = "model.language_model.layers." + std::to_string(l) + ".mlp.experts.0.";
            std::vector<Impl::Slot>* vs[3] = { &L.g, &L.u, &L.d };
            std::vector<Exl3Mat>* ms[3] = { &L.gm, &L.um, &L.dm };
            const char* projs[3] = { "gate_proj", "up_proj", "down_proj" };
            for (int i = 0; i < 3; ++i) {
                Exl3Linear S = impl_->model->linear(b + projs[i]);
                const size_t trb = (size_t) S.ki * S.nj * (256 * S.bits / 16) * 2;
                const size_t suhb = (size_t) S.in() * 2, svhb = (size_t) S.out() * 2;
                uint16_t *tr = nullptr, *suh = nullptr, *svh = nullptr;
                alloc(&tr, trb * (size_t) impl_->cache);
                alloc(&suh, suhb * (size_t) impl_->cache);
                alloc(&svh, svhb * (size_t) impl_->cache);
                L.allocs.push_back(tr); L.allocs.push_back(suh); L.allocs.push_back(svh);
                vs[i]->resize((size_t) impl_->cache);
                ms[i]->resize((size_t) impl_->cache);
                for (int s = 0; s < impl_->cache; ++s) {
                    Impl::Slot& sl = (*vs[i])[(size_t) s];
                    sl.tr = tr + (size_t) s * (trb / 2);
                    sl.suh = suh + (size_t) s * (suhb / 2);
                    sl.svh = svh + (size_t) s * (svhb / 2);
                    sl.trb = trb; sl.suhb = suhb; sl.svhb = svhb;
                    Exl3Mat& M = (*ms[i])[(size_t) s];
                    M.trellis = sl.tr; M.suh = sl.suh; M.svh = sl.svh;
                    M.ki = S.ki; M.nj = S.nj; M.bits = S.bits; M.cb = S.cb;
                }
            }
        }
    }
}

Exl3ExpertStream::~Exl3ExpertStream() { delete impl_; }

int Exl3ExpertStream::slots_per_layer() const { return impl_->cache; }

size_t Exl3ExpertStream::bytes() const {
    size_t n = (size_t) impl_->n_embd * 2 + (size_t) impl_->max_k * impl_->n_embd * (2 + 4);
    for (auto& s : impl_->scratch.g) n += s.trb + s.suhb + s.svhb;
    for (auto& s : impl_->scratch.u) n += s.trb + s.suhb + s.svhb;
    for (auto& s : impl_->scratch.d) n += s.trb + s.suhb + s.svhb;
    for (const Impl::Layer& l : impl_->layers)
        for (const auto& v : { &l.g, &l.u, &l.d })
            for (const Impl::Slot& s : *v) n += s.trb + s.suhb + s.svhb;
    return n;
}

bool Exl3ExpertStream::run(int layer, const float* x_f32, const int* ids, int k, float* out, void* stream) {
    if (k <= 0) return true;
    if (k > impl_->max_k) return false;
    cudaStream_t st = (cudaStream_t) stream;
    const long n = impl_->n_embd;
    static const bool timing = std::getenv("STRATA_EXL3_TIME") != nullptr;
    const auto t_call0 = std::chrono::steady_clock::now();
    const auto t_up0 = std::chrono::steady_clock::now();
    f32_to_f16_rows_kernel<<<(unsigned)((n + 255) / 256), 256, 0, st>>>(x_f32, impl_->d_x, n);

    // ---- which slots this token's k experts live in.  A cache hit costs nothing; a miss reads the
    // expert out of the mmap and uploads it into the least recently used slot of this layer.
    std::vector<Exl3Mat> gm, um, dm;
    Impl::Layer* cache = (impl_->cache > 0 && layer >= 0 && layer < (int) impl_->layers.size() &&
                          k <= impl_->cache) ? &impl_->layers[(size_t) layer] : nullptr;
    const std::string pfx = "model.language_model.layers." + std::to_string(layer) + ".mlp.experts.";
    if (cache != nullptr) {
        gm.resize((size_t) k); um.resize((size_t) k); dm.resize((size_t) k);
        for (int i = 0; i < k; ++i) {
            const int e = ids[i];
            auto it = cache->at.find(e);
            int s = -1;
            if (it != cache->at.end()) {
                s = it->second;
                ++impl_->hits;
            } else {
                ++impl_->misses;
                if (cache->filled < impl_->cache) {
                    s = cache->filled++;                 // still filling: the slots are handed out in order
                } else {
                    s = (int) (std::min_element(cache->tick.begin(), cache->tick.end()) - cache->tick.begin());
                    for (auto victim = cache->at.begin(); victim != cache->at.end(); ++victim)
                        if (victim->second == s) { cache->at.erase(victim); break; }
                }
                const std::string b = pfx + std::to_string(e) + ".";
                Exl3Linear Lg = impl_->model->linear(b + "gate_proj");
                Exl3Linear Lu = impl_->model->linear(b + "up_proj");
                Exl3Linear Ld = impl_->model->linear(b + "down_proj");
                upload16(cache->g[(size_t) s].tr, Lg.trellis, cache->g[(size_t) s].trb, st);
                upload16(cache->g[(size_t) s].suh, Lg.suh, cache->g[(size_t) s].suhb, st);
                upload16(cache->g[(size_t) s].svh, Lg.svh, cache->g[(size_t) s].svhb, st);
                upload16(cache->u[(size_t) s].tr, Lu.trellis, cache->u[(size_t) s].trb, st);
                upload16(cache->u[(size_t) s].suh, Lu.suh, cache->u[(size_t) s].suhb, st);
                upload16(cache->u[(size_t) s].svh, Lu.svh, cache->u[(size_t) s].svhb, st);
                upload16(cache->d[(size_t) s].tr, Ld.trellis, cache->d[(size_t) s].trb, st);
                upload16(cache->d[(size_t) s].suh, Ld.suh, cache->d[(size_t) s].suhb, st);
                upload16(cache->d[(size_t) s].svh, Ld.svh, cache->d[(size_t) s].svhb, st);
                cache->at[e] = s;
            }
            cache->tick[(size_t) s] = ++cache->clock;
            gm[(size_t) i] = cache->gm[(size_t) s];
            um[(size_t) i] = cache->um[(size_t) s];
            dm[(size_t) i] = cache->dm[(size_t) s];
        }
    } else {
        gm.resize((size_t) k); um.resize((size_t) k); dm.resize((size_t) k);
        for (int i = 0; i < k; ++i) {
            const std::string b = pfx + std::to_string(ids[i]) + ".";
            Exl3Linear Lg = impl_->model->linear(b + "gate_proj");
            Exl3Linear Lu = impl_->model->linear(b + "up_proj");
            Exl3Linear Ld = impl_->model->linear(b + "down_proj");
            upload16(impl_->scratch.g[(size_t) i].tr, Lg.trellis, impl_->scratch.g[(size_t) i].trb, st);
            upload16(impl_->scratch.g[(size_t) i].suh, Lg.suh, impl_->scratch.g[(size_t) i].suhb, st);
            upload16(impl_->scratch.g[(size_t) i].svh, Lg.svh, impl_->scratch.g[(size_t) i].svhb, st);
            upload16(impl_->scratch.u[(size_t) i].tr, Lu.trellis, impl_->scratch.u[(size_t) i].trb, st);
            upload16(impl_->scratch.u[(size_t) i].suh, Lu.suh, impl_->scratch.u[(size_t) i].suhb, st);
            upload16(impl_->scratch.u[(size_t) i].svh, Lu.svh, impl_->scratch.u[(size_t) i].svhb, st);
            upload16(impl_->scratch.d[(size_t) i].tr, Ld.trellis, impl_->scratch.d[(size_t) i].trb, st);
            upload16(impl_->scratch.d[(size_t) i].suh, Ld.suh, impl_->scratch.d[(size_t) i].suhb, st);
            upload16(impl_->scratch.d[(size_t) i].svh, Ld.svh, impl_->scratch.d[(size_t) i].svhb, st);
            gm[(size_t) i] = impl_->scratch.gm[(size_t) i];
            um[(size_t) i] = impl_->scratch.um[(size_t) i];
            dm[(size_t) i] = impl_->scratch.dm[(size_t) i];
        }
    }
    const auto t_up1 = std::chrono::steady_clock::now();
    exl3_moe_rows(gm.data(), um.data(), dm.data(), ids, k, impl_->d_x, impl_->d_rows, stream);
    const auto t_ffn1 = std::chrono::steady_clock::now();
    f16_to_f32_rows_kernel<<<(unsigned)((n * k + 255) / 256), 256, 0, st>>>(impl_->d_rows, impl_->d_out, n * k);
    if (cudaMemcpyAsync(out, impl_->d_out, (size_t) n * k * 4, cudaMemcpyDeviceToHost, st) != cudaSuccess) return false;
    const cudaError_t se = cudaStreamSynchronize(st);
    if (se != cudaSuccess) std::fprintf(stderr, "exl3 stream run(layer=%d): %s\n", layer, cudaGetErrorString(se));
    if (timing) {
        const auto t_d2h1 = std::chrono::steady_clock::now();
        impl_->up_ms += std::chrono::duration<double, std::milli>(t_up1 - t_up0).count();
        impl_->ffn_ms += std::chrono::duration<double, std::milli>(t_ffn1 - t_up1).count();
        impl_->d2h_ms += std::chrono::duration<double, std::milli>(t_d2h1 - t_ffn1).count();
        impl_->call_ms += std::chrono::duration<double, std::milli>(t_d2h1 - t_call0).count();
        if (layer == impl_->n_layers - 1) {
            const long long uses = impl_->hits + impl_->misses;
            std::fprintf(stderr, "exl3 stream: %lld experts, %.1f%% cached | upload %.1f ms, ffn %.1f ms, "
                                 "d2h %.1f ms, pool %.1f ms per token\n", uses,
                         100.0 * (double) impl_->hits / (double) std::max<long long>(uses, 1),
                         impl_->up_ms, impl_->ffn_ms, impl_->d2h_ms, impl_->call_ms);
            impl_->hits = impl_->misses = 0;
            impl_->up_ms = impl_->ffn_ms = impl_->d2h_ms = impl_->call_ms = 0;
        }
    }
    return true;
}

}  // namespace strata::kernels
