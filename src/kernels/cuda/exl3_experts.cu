// src/kernels/cuda/exl3_experts.cu - one layer's routed EXL3 experts on the GPU (see the header).
#include "strata/kernels/exl3_experts.hpp"
#include "strata/artifact/exl3_model.hpp"

#include <cuda_runtime.h>
#include <cuda_fp16.h>

#include <algorithm>
#include <atomic>
#include <chrono>
#include <condition_variable>
#include <mutex>
#include <cstdio>
#include <cstdlib>
#include <string>
#include <thread>
#include <unordered_map>
#include <vector>

#if !defined(_WIN32)
#include <fcntl.h>
#include <unistd.h>
#endif

#if defined(__x86_64__) || defined(_M_X64)
#include <immintrin.h>
#define STRATA_EXL3_PAUSE() _mm_pause()
#else
#define STRATA_EXL3_PAUSE() ((void) 0)
#endif

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

    // ---- the host-side cache (see the header).  Per layer, like the VRAM one: the routing is layer-specific,
    // and a slot holds that layer's nine pieces at fixed offsets, so a hit is a memcpy into the staging buffer.
    std::vector<uint8_t> ram;                     // pageable: pinning this much needs a memlock limit
    std::vector<size_t> ram_base;                 // per layer: where its slots start in `ram`
    std::vector<size_t> ram_slot_bytes;           // per layer: one expert's nine pieces
    std::vector<std::array<size_t, 9>> ram_off;   // per layer: the nine piece offsets inside a slot
    int ram_slots = 0;                            // slots per layer
    struct RamLayer {
        std::unordered_map<int, int> at;          // expert -> slot
        std::vector<uint32_t> tick;               // per slot: the last use, for the LRU
        std::vector<int> key;                     // slot -> expert
        int filled = 0;
        uint32_t clock = 0;
    };
    std::vector<RamLayer> ram_layers;
    std::mutex ram_mu;
    long long ram_hits = 0, ram_misses = 0;
    int cur_layer = 0;                            // the layer `run` is on (the queue is host-side, single-threaded)

    // STRATA_EXL3_TIME=1: per-token upload/ffn/d2h milliseconds and the cache hit rate
    long long hits = 0, misses = 0;
    double up_ms = 0, ffn_ms = 0, d2h_ms = 0, call_ms = 0;

    // ---- the miss reads
    //
    // **A MISS IS READ WITH `pread`, NOT OUT OF THE MMAP.**  Copying a tensor out of the mmap faults every
    // 4 KiB page on the way in: measured on 60 real expert tensors, 0.3 GB/s through the mmap against
    // 10.4 GB/s with `preadv` into a reused buffer, both from the page cache.  The read is the whole of the
    // doorbell: the token graph spins while the host fetches the expert.
    //
    // The destination is pinned staging, one buffer per worker, so the kernel writes straight into memory
    // the H2D can DMA from: no pageable staging copy, and the buffer's own pages never fault again.
    struct Job { uint16_t* dst; const uint16_t* src; int fd; size_t off; size_t bytes;
                 uint8_t* ram_dst = nullptr;       // a miss: fill this slot as a side effect of the read
                 const uint8_t* ram_src = nullptr; // a hit: copy this out of RAM through the pinned staging
               };
    std::vector<Job> jobs;
    std::vector<std::thread> workers;
    // **TWO BUFFERS PER WORKER, WITH AN EVENT ON EACH.**  The DMA out of a staging buffer is asynchronous,
    // so the next pread into the same buffer would overwrite data the DMA has not read yet (that is how the
    // first version produced NaNs).  Two buffers let a worker read into one while the other drains, and the
    // event orders the reuse.
    struct Stage { uint8_t* buf[2] = {nullptr, nullptr}; cudaEvent_t done[2] = {nullptr, nullptr}; int turn = 0; };
    std::vector<Stage> stage;          // one per worker (one for the sequential fallback)
    size_t stage_bytes = 0;
    std::map<std::string, int> fds;    // shard path -> fd, opened once
    std::mutex api_mu;                 // concurrent enqueues to one stream are not safe on the null stream
    // the resident/miss split: two output blocks (the FFN of each part writes its own) and the row maps
    uint16_t* d_hit = nullptr;
    uint16_t* d_miss = nullptr;
    int* d_hit_rows = nullptr;
    int* d_miss_rows = nullptr;
    std::atomic<size_t> job_at{0};
    std::atomic<uint64_t> work_done{0};
    std::mutex work_mu;
    std::condition_variable work_cv;
    uint64_t work_gen = 0;             // guarded by work_mu
    bool work_quit = false;            // guarded by work_mu
    int n_workers = 0;
    cudaStream_t work_stream = nullptr;   // the staging copies' own stream (see the ctor)
    cudaEvent_t stage_done = nullptr;     // recorded on it, waited on the FFN's stream

    void stage_one(const Job& j, Stage& stg) {
        if (j.bytes == 0) return;
        const uint16_t* src = j.src;
        if (j.ram_src != nullptr && stg.buf[0] != nullptr) {
            // **A HIT GOES THROUGH THE PINNED BUFFER TOO.**  `cudaMemcpyAsync` from pageable memory is
            // synchronous: it would stall each worker for the length of the copy, which is exactly what the
            // double buffer and the events exist to avoid.
            const int b = stg.turn;
            stg.turn ^= 1;
            if (stg.done[b] != nullptr) (void) cudaEventSynchronize(stg.done[b]);
            std::memcpy(stg.buf[b], j.ram_src, j.bytes);
            const std::lock_guard<std::mutex> lk(api_mu);
            (void) cudaMemcpyAsync(j.dst, stg.buf[b], j.bytes, cudaMemcpyHostToDevice, work_stream);
            if (stg.done[b] != nullptr) (void) cudaEventRecord(stg.done[b], work_stream);
            return;
        }
#if !defined(_WIN32)
        if (j.fd >= 0 && stg.buf[0] != nullptr) {
            const int b = stg.turn;
            stg.turn ^= 1;
            if (stg.done[b] != nullptr) (void) cudaEventSynchronize(stg.done[b]);
            const ssize_t got = ::pread(j.fd, stg.buf[b], j.bytes, (off_t) j.off);
            if (got == (ssize_t) j.bytes) {
                src = (const uint16_t*) stg.buf[b];
                // the host cache is filled as a side effect of the read that had to happen anyway; the copy is
                // synchronous, so it is complete before this buffer is read into again (the double buffer's
                // event only guards the DMA)
                if (j.ram_dst != nullptr) std::memcpy(j.ram_dst, stg.buf[b], j.bytes);
                {
                    // **ONE ENQUEUE AT A TIME.**  The workers read in parallel, but two threads enqueueing
                    // `cudaMemcpyAsync` on the same (legacy default) stream raced: the same run produced
                    // different tokens with 4 workers and identical ones with 1.  The copies are asynchronous,
                    // so serialising only the call costs nothing.
                    const std::lock_guard<std::mutex> lk(api_mu);
                    (void) cudaMemcpyAsync(j.dst, src, j.bytes, cudaMemcpyHostToDevice, work_stream);
                    if (stg.done[b] != nullptr) (void) cudaEventRecord(stg.done[b], work_stream);
                }
                return;
            }
            std::fprintf(stderr, "strata generate: EXL3 expert read: pread %zu B at %zu: %zd\n", j.bytes, j.off, got);
        }
#endif
        {
            const std::lock_guard<std::mutex> lk(api_mu);
            (void) cudaMemcpyAsync(j.dst, src, j.bytes, cudaMemcpyHostToDevice, work_stream);
        }
    }

    // Queue one expert's nine pieces: the FILE location (for the pread), the mmap pointer (the fallback), and
    // the host cache: on a hit the pieces come out of RAM, on a miss the read fills the slot as a side effect.
    void queue_expert(int expert, const Slot* gs, const Slot* us, const Slot* ds,
                      const Exl3Linear& Lg, const Exl3Linear& Lu, const Exl3Linear& Ld) {
        const std::string b = "model.language_model.layers." + std::to_string(cur_layer) + ".mlp.experts." +
                              std::to_string(expert) + ".";
        const uint8_t* ram_src = nullptr;
        uint8_t* ram_dst = nullptr;
        if (ram_slots > 0) {
            const std::lock_guard<std::mutex> lk(ram_mu);
            RamLayer& rl = ram_layers[(size_t) cur_layer];
            const size_t base = ram_base[(size_t) cur_layer];
            const size_t slot_bytes = ram_slot_bytes[(size_t) cur_layer];
            auto it = rl.at.find(expert);
            int s = -1;
            if (it != rl.at.end()) {
                s = it->second;
                ram_src = ram.data() + base + (size_t) s * slot_bytes;
                ++ram_hits;
            } else {
                ++ram_misses;
                if (rl.filled < ram_slots) {
                    s = rl.filled++;
                } else {
                    s = (int) (std::min_element(rl.tick.begin(), rl.tick.end()) - rl.tick.begin());
                    rl.at.erase(rl.key[(size_t) s]);
                }
                rl.at[expert] = s;
                rl.key[(size_t) s] = expert;
                ram_dst = ram.data() + base + (size_t) s * slot_bytes;
            }
            rl.tick[(size_t) s] = ++rl.clock;
        }
        const std::array<size_t, 9>& ro = ram_off[(size_t) cur_layer];
        std::string path; size_t off = 0, bytes = 0;
        int piece = 0;
        auto push = [&](uint16_t* dst, const std::string& base, const char* plane, const uint16_t* src, size_t n) {
            // STRATA_EXL3_NO_PREAD=1 reads through the mmap instead (an A/B arm for the pread path)
            static const bool no_pread = std::getenv("STRATA_EXL3_NO_PREAD") != nullptr;
            if (ram_src != nullptr) {   // a hit: no file read at all
                jobs.push_back({dst, src, -1, 0, n, nullptr, ram_src + ro[(size_t) piece]});
                ++piece;
                return;
            }
            const int fd = (!no_pread && model->locate(base + plane, path, off, bytes) && bytes == n) ? fd_of(path) : -1;
            jobs.push_back({dst, src, fd, off, n, ram_dst != nullptr ? ram_dst + ro[(size_t) piece] : nullptr});
            ++piece;
        };
        const std::string bg = b + "gate_proj", bu = b + "up_proj", bd = b + "down_proj";
        push(gs->tr, bg, ".trellis", Lg.trellis, gs->trb);
        push(gs->suh, bg, ".suh", Lg.suh, gs->suhb);
        push(gs->svh, bg, ".svh", Lg.svh, gs->svhb);
        push(us->tr, bu, ".trellis", Lu.trellis, us->trb);
        push(us->suh, bu, ".suh", Lu.suh, us->suhb);
        push(us->svh, bu, ".svh", Lu.svh, us->svhb);
        push(ds->tr, bd, ".trellis", Ld.trellis, ds->trb);
        push(ds->suh, bd, ".suh", Ld.suh, ds->suhb);
        push(ds->svh, bd, ".svh", Ld.svh, ds->svhb);
    }

    int fd_of(const std::string& path) {
#if defined(_WIN32)
        (void) path;
        return -1;
#else
        auto it = fds.find(path);
        if (it != fds.end()) return it->second;
        const int fd = ::open(path.c_str(), O_RDONLY);
        fds[path] = fd;
        return fd;
#endif
    }

    void stage_parallel() {
        if (jobs.empty()) return;
        if (workers.empty()) {
            for (const Job& j : jobs) stage_one(j, stage[0]);
            return;
        }
        job_at.store(0, std::memory_order_relaxed);
        work_done.store(0, std::memory_order_relaxed);
        {
            const std::lock_guard<std::mutex> lk(work_mu);
            ++work_gen;
        }
        work_cv.notify_all();
        while (work_done.load(std::memory_order_acquire) < (uint64_t) n_workers) STRATA_EXL3_PAUSE();
    }

    void worker_loop(int wid) {
        uint64_t seen = 0;
        for (;;) {
            {
                std::unique_lock<std::mutex> lk(work_mu);
                work_cv.wait(lk, [&] { return work_quit || work_gen != seen; });
                if (work_quit) return;
                seen = work_gen;
            }
            for (;;) {
                const size_t i = job_at.fetch_add(1, std::memory_order_relaxed);
                if (i >= jobs.size()) break;
                stage_one(jobs[i], stage[(size_t) wid]);
            }
            work_done.fetch_add(1, std::memory_order_release);
        }
    }

    ~Impl() {
        {
            const std::lock_guard<std::mutex> lk(work_mu);
            work_quit = true;
        }
        work_cv.notify_all();
        for (std::thread& t : workers) if (t.joinable()) t.join();
        for (Stage& st : stage) {
            for (uint8_t* p : st.buf) if (p != nullptr) (void) cudaFreeHost(p);
            for (cudaEvent_t e : st.done) if (e != nullptr) (void) cudaEventDestroy(e);
        }
#if !defined(_WIN32)
        for (auto& kv : fds) if (kv.second >= 0) (void) ::close(kv.second);
#endif
        for (void* p : { (void*) d_x, (void*) d_rows, (void*) d_out, (void*) d_hit, (void*) d_miss,
                         (void*) d_hit_rows, (void*) d_miss_rows }) (void) cudaFree(p);
        if (stage_done != nullptr) (void) cudaEventDestroy(stage_done);
        if (work_stream != nullptr) (void) cudaStreamDestroy(work_stream);
        for (auto& v : { &scratch.g, &scratch.u, &scratch.d })
            for (Slot& s : *v) { (void) cudaFree(s.tr); (void) cudaFree(s.suh); (void) cudaFree(s.svh); }
        for (Layer& l : layers)
            for (void* p : l.allocs) (void) cudaFree(p);
    }
};

Exl3ExpertStream::Exl3ExpertStream(const std::string& model_dir, int n_embd, int n_layers, int slots_per_layer,
                                   void* stream, size_t ram_cache_bytes) {
    impl_ = new Impl;
    impl_->dir = model_dir;
    impl_->model = std::make_unique<Exl3Model>(model_dir);
    impl_->n_embd = n_embd;
    impl_->n_layers = n_layers;
    // **THE STAGING COPIES GET THEIR OWN STREAM.**  They used to share the caller's, which put them behind
    // the resident experts' FFN: the double-buffered workers then blocked on their own DMA events for the
    // length of that FFN (measured: the staging went from 24 to 37 ms/token).  With a stream of their own
    // the copies start at once, and the miss FFN waits on one event instead.
    if (cudaStreamCreateWithFlags(&impl_->work_stream, cudaStreamNonBlocking) != cudaSuccess ||
        cudaEventCreateWithFlags(&impl_->stage_done, cudaEventDisableTiming) != cudaSuccess) {
        std::fprintf(stderr, "strata generate: EXL3 expert stream: no staging stream\n");
        std::abort();
    }
    (void) stream;
    // The staging workers: enough to saturate the host's memory bandwidth on the miss reads, not more.
    const char* nw = std::getenv("STRATA_EXL3_WORKERS");
    impl_->n_workers = nw != nullptr ? std::atoi(nw) : 4;
    // the largest piece any layer's expert has (the bitrate can differ per layer), and one expert's nine
    // pieces per layer for the host cache's slot layout
    impl_->ram_slot_bytes.assign((size_t) n_layers, 0);
    impl_->ram_off.assign((size_t) n_layers, {});
    for (int l = 0; l < n_layers; ++l) {
        const std::string b = "model.language_model.layers." + std::to_string(l) + ".mlp.experts.0.";
        size_t off = 0;
        int piece = 0;
        for (const char* p : {"gate_proj", "up_proj", "down_proj"}) {
            Exl3Linear L = impl_->model->linear(b + p);
            const size_t trb = (size_t) L.ki * L.nj * (256 * L.bits / 16) * 2;
            impl_->stage_bytes = std::max(impl_->stage_bytes, std::max(trb, (size_t) (L.in() + L.out()) * 2));
            for (const size_t n : {trb, (size_t) L.in() * 2, (size_t) L.out() * 2}) {
                impl_->ram_off[(size_t) l][(size_t) piece++] = off;
                off += n;
            }
        }
        impl_->ram_slot_bytes[(size_t) l] = off;
    }
    if (ram_cache_bytes > 0) {
        size_t per_layer = 0;
        for (size_t n : impl_->ram_slot_bytes) per_layer += n;
        if (per_layer > 0) impl_->ram_slots = (int) std::min<size_t>(ram_cache_bytes / per_layer, 1u << 20);
        if (impl_->ram_slots > 0) {
            impl_->ram_base.assign((size_t) n_layers, 0);
            size_t at = 0;
            for (int l = 0; l < n_layers; ++l) {
                impl_->ram_base[(size_t) l] = at;
                at += (size_t) impl_->ram_slots * impl_->ram_slot_bytes[(size_t) l];
            }
            try { impl_->ram.assign(at, 0); } catch (const std::exception&) { impl_->ram_slots = 0; }
            if (impl_->ram_slots > 0) {
                impl_->ram_layers.resize((size_t) n_layers);
                for (Impl::RamLayer& rl : impl_->ram_layers) {
                    rl.tick.assign((size_t) impl_->ram_slots, 0);
                    rl.key.assign((size_t) impl_->ram_slots, -1);
                }
            }
        }
    }
    impl_->stage.resize((size_t) (impl_->n_workers > 0 ? impl_->n_workers : 1));
    for (Impl::Stage& stg : impl_->stage)
        for (int b = 0; b < 2; ++b) {
            if (cudaMallocHost((void**) &stg.buf[b], impl_->stage_bytes) != cudaSuccess) stg.buf[b] = nullptr;
            else if (cudaEventCreateWithFlags(&stg.done[b], cudaEventDisableTiming) != cudaSuccess) stg.done[b] = nullptr;
        }
    if (impl_->n_workers > 0) {
        impl_->workers.reserve((size_t) impl_->n_workers);
        for (int i = 0; i < impl_->n_workers; ++i) impl_->workers.emplace_back([im = impl_, i] { im->worker_loop(i); });
    }
    const int mk = impl_->max_k;
    auto alloc = [](uint16_t** p, size_t bytes) { if (cudaMalloc((void**) p, bytes ? bytes : 1) != cudaSuccess) std::abort(); };
    alloc(&impl_->d_x, (size_t) n_embd * 2);
    alloc(&impl_->d_rows, (size_t) mk * n_embd * 2);
    if (cudaMalloc((void**) &impl_->d_out, (size_t) mk * n_embd * 4) != cudaSuccess) std::abort();
    if (cudaMalloc((void**) &impl_->d_hit, (size_t) mk * n_embd * 2) != cudaSuccess ||
        cudaMalloc((void**) &impl_->d_miss, (size_t) mk * n_embd * 2) != cudaSuccess ||
        cudaMalloc((void**) &impl_->d_hit_rows, (size_t) mk * sizeof(int)) != cudaSuccess ||
        cudaMalloc((void**) &impl_->d_miss_rows, (size_t) mk * sizeof(int)) != cudaSuccess) std::abort();
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
int Exl3ExpertStream::ram_slots_per_layer() const { return impl_->ram_slots; }
size_t Exl3ExpertStream::ram_bytes() const { return impl_->ram.size(); }

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

    impl_->jobs.clear();
    // ---- which slots this token's k experts live in.  A cache hit costs nothing; a miss reads the
    // expert out of the mmap and uploads it into the least recently used slot of this layer.
    std::vector<Exl3Mat> gm, um, dm;                 // the no-split path (no cache): one list of k
    std::vector<Exl3Mat> hg, hu, hd, mg, mu, md;     // the split path: resident rows and miss rows
    std::vector<int> hrow, mrow;
    Impl::Layer* cache = (impl_->cache > 0 && layer >= 0 && layer < (int) impl_->layers.size() &&
                          k <= impl_->cache) ? &impl_->layers[(size_t) layer] : nullptr;
    impl_->cur_layer = layer;
    const std::string pfx = "model.language_model.layers." + std::to_string(layer) + ".mlp.experts.";
    if (cache != nullptr) {
        hg.reserve((size_t) k); hu.reserve((size_t) k); hd.reserve((size_t) k);
        mg.reserve((size_t) k); mu.reserve((size_t) k); md.reserve((size_t) k);
        hrow.reserve((size_t) k); mrow.reserve((size_t) k);
        for (int i = 0; i < k; ++i) {
            const int e = ids[i];
            auto it = cache->at.find(e);
            int s = -1;
            const bool resident = it != cache->at.end();
            if (resident) {
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
                Exl3Linear Lg = impl_->model->linear(pfx + std::to_string(e) + ".gate_proj");
                Exl3Linear Lu = impl_->model->linear(pfx + std::to_string(e) + ".up_proj");
                Exl3Linear Ld = impl_->model->linear(pfx + std::to_string(e) + ".down_proj");
                impl_->queue_expert(e, &cache->g[(size_t) s], &cache->u[(size_t) s], &cache->d[(size_t) s],
                                    Lg, Lu, Ld);
                cache->at[e] = s;
            }
            cache->tick[(size_t) s] = ++cache->clock;
            if (resident) {
                hrow.push_back(i);
                hg.push_back(cache->gm[(size_t) s]); hu.push_back(cache->um[(size_t) s]); hd.push_back(cache->dm[(size_t) s]);
            } else {
                mrow.push_back(i);
                mg.push_back(cache->gm[(size_t) s]); mu.push_back(cache->um[(size_t) s]); md.push_back(cache->dm[(size_t) s]);
            }
        }
    } else {
        gm.resize((size_t) k); um.resize((size_t) k); dm.resize((size_t) k);
        for (int i = 0; i < k; ++i) {
            Exl3Linear Lg = impl_->model->linear(pfx + std::to_string(ids[i]) + ".gate_proj");
            Exl3Linear Lu = impl_->model->linear(pfx + std::to_string(ids[i]) + ".up_proj");
            Exl3Linear Ld = impl_->model->linear(pfx + std::to_string(ids[i]) + ".down_proj");
            impl_->queue_expert(ids[i], &impl_->scratch.g[(size_t) i], &impl_->scratch.u[(size_t) i],
                                &impl_->scratch.d[(size_t) i], Lg, Lu, Ld);
            gm[(size_t) i] = impl_->scratch.gm[(size_t) i];
            um[(size_t) i] = impl_->scratch.um[(size_t) i];
            dm[(size_t) i] = impl_->scratch.dm[(size_t) i];
        }
        // no cache: nothing is resident, so the whole FFN follows the staging (the split has nothing to run early)
        impl_->stage_parallel();
        exl3_moe_rows(gm.data(), um.data(), dm.data(), ids, k, impl_->d_x, impl_->d_rows, stream);
    }
    // **THE RESIDENT HALF'S FFN RUNS WHILE THE MISSES ARE READ.**  Their weights are already in VRAM, so
    // nothing has to wait for the host: the kernel is enqueued before the workers start, and the miss FFN
    // follows the staging copies on the same stream.  Without the split the whole FFN waited for the last
    // miss, which is most of the doorbell.
    if (!hrow.empty())
        exl3_moe_rows_part(hg.data(), hu.data(), hd.data(), (int) hrow.size(), impl_->d_x, impl_->d_hit, stream);
    impl_->stage_parallel();
    const auto t_up1 = std::chrono::steady_clock::now();
    // the miss FFN (and the scatter) read what the staging wrote: order them after it
    if (!impl_->jobs.empty()) {
        (void) cudaEventRecord(impl_->stage_done, impl_->work_stream);
        (void) cudaStreamWaitEvent(st, impl_->stage_done, 0);
    }
    if (!mrow.empty()) {
        (void) cudaMemcpyAsync(impl_->d_hit_rows, hrow.data(), hrow.size() * sizeof(int), cudaMemcpyHostToDevice, st);
        (void) cudaMemcpyAsync(impl_->d_miss_rows, mrow.data(), mrow.size() * sizeof(int), cudaMemcpyHostToDevice, st);
        exl3_moe_rows_part(mg.data(), mu.data(), md.data(), (int) mrow.size(), impl_->d_x, impl_->d_miss, stream);
        exl3_rows_scatter(impl_->d_hit, impl_->d_hit_rows, (int) hrow.size(), impl_->d_miss, impl_->d_miss_rows,
                          (int) mrow.size(), (int) n, impl_->d_rows, stream);
    } else if (!hrow.empty()) {
        (void) cudaMemcpyAsync(impl_->d_hit_rows, hrow.data(), hrow.size() * sizeof(int), cudaMemcpyHostToDevice, st);
        exl3_rows_scatter(impl_->d_hit, impl_->d_hit_rows, (int) hrow.size(), nullptr, nullptr, 0, (int) n,
                          impl_->d_rows, stream);
    }
    const auto t_ffn1 = std::chrono::steady_clock::now();
    f16_to_f32_rows_kernel<<<(unsigned)((n * k + 255) / 256), 256, 0, st>>>(impl_->d_rows, impl_->d_out, n * k);
    if (cudaMemcpyAsync(out, impl_->d_out, (size_t) n * k * 4, cudaMemcpyDeviceToHost, st) != cudaSuccess) return false;
    const cudaError_t se = cudaStreamSynchronize(st);
    if (impl_->work_stream != nullptr) (void) cudaStreamSynchronize(impl_->work_stream);   // its copies feed this layer's FFN
    if (se != cudaSuccess) std::fprintf(stderr, "exl3 stream run(layer=%d): %s\n", layer, cudaGetErrorString(se));
    if (timing) {
        const auto t_d2h1 = std::chrono::steady_clock::now();
        impl_->up_ms += std::chrono::duration<double, std::milli>(t_up1 - t_up0).count();
        impl_->ffn_ms += std::chrono::duration<double, std::milli>(t_ffn1 - t_up1).count();
        impl_->d2h_ms += std::chrono::duration<double, std::milli>(t_d2h1 - t_ffn1).count();
        impl_->call_ms += std::chrono::duration<double, std::milli>(t_d2h1 - t_call0).count();
        if (layer == impl_->n_layers - 1) {
            const long long uses = impl_->hits + impl_->misses;
            std::fprintf(stderr, "exl3 stream: %lld experts, %.1f%% cached, host %.1f%% (%lld reads) | upload %.1f ms, "
                                 "ffn %.1f ms, d2h %.1f ms, pool %.1f ms per token\n", uses,
                         100.0 * (double) impl_->hits / (double) std::max<long long>(uses, 1),
                         100.0 * (double) impl_->ram_hits / (double) std::max<long long>(impl_->ram_hits + impl_->ram_misses, 1),
                         impl_->ram_misses, impl_->up_ms, impl_->ffn_ms, impl_->d2h_ms, impl_->call_ms);
            impl_->hits = impl_->misses = 0;
            impl_->ram_hits = impl_->ram_misses = 0;
            impl_->up_ms = impl_->ffn_ms = impl_->d2h_ms = impl_->call_ms = 0;
        }
    }
    return true;
}

}  // namespace strata::kernels
