#pragma once
// Private host staging ring, shared with the HIP ownership regression test.
#include <cuda_runtime.h>
#include <algorithm>
#include <atomic>
#include <condition_variable>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <memory>
#include <mutex>
#include <thread>
#include <vector>

namespace strata::prefill::detail {
struct Stager {
    // D-5: the pinned ring's depth (STRATA_STAGER_RING, default 16) - how far the host copies can run ahead of the
    // DMAs of the unpinned experts' blobs
    int kRing = 16;
    struct Job { const uint8_t* src; size_t bytes; };
    std::vector<uint8_t*> buf;
    std::vector<char> pinned;
    std::vector<std::vector<uint8_t>> pageable;   // the fallback when no more RAM can be pinned
    std::vector<cudaEvent_t> dma_done;
    std::vector<Job> jobs;
    std::unique_ptr<std::atomic<int>[]> ready;
    size_t ready_cap = 0;
    // gen << 32 | n << 16 | next index: a claim is a CAS on the generation it woke for (a thread late from the
    // previous layer can never take a job of this one - the expert pool's issue #29 lesson)
    std::atomic<uint64_t> head{0};
    std::atomic<int> issued{0}, active{0};
    uint32_t gen = 0;
    bool quit = false;
    std::mutex mu;
    std::condition_variable cv;
    bool profile_host = std::getenv("STRATA_PREFILL_TIMING") != nullptr;
    int fence_batch = 1;
    int last_submitted = -1;
    cudaStream_t fence_stream = nullptr;
    uint64_t fence_records = 0;
    std::atomic<uint64_t> fence_waits{0}, fence_errors{0};
    std::vector<std::thread> threads;
    int device = 0;

    bool init(size_t blob_bytes, int nthreads) {
        if (const char* v = std::getenv("STRATA_STAGER_RING")) kRing = std::clamp(std::atoi(v), 2, 256);
        buf.assign((size_t) kRing, nullptr);
        pinned.assign((size_t) kRing, 0);
        dma_done.assign((size_t) kRing, nullptr);
        pageable.resize(kRing);
        for (int i = 0; i < kRing; ++i) {
            pinned[i] = cudaHostAlloc((void**) &buf[i], blob_bytes, cudaHostAllocDefault) == cudaSuccess;
            if (!pinned[i]) {
                cudaGetLastError();
                pageable[(size_t) i].resize(blob_bytes);
                buf[i] = pageable[(size_t) i].data();
            }
            if (cudaEventCreateWithFlags(&dma_done[i], cudaEventDisableTiming) != cudaSuccess) return false;
        }
        cudaGetDevice(&device);
        for (int t = 0; t < nthreads; ++t) threads.emplace_back([this] { work(); });
        return true;
    }
    ~Stager() {
        finish();
        { std::lock_guard<std::mutex> lk(mu); quit = true; }
        cv.notify_all();
        for (auto& t : threads) t.join();
        for (int i = 0; i < kRing; ++i) {
            if (dma_done[i]) cudaEventDestroy(dma_done[i]);
            if (buf[i] && pinned[i]) cudaFreeHost(buf[i]);
        }
    }
    void work() {
        cudaSetDevice(device);
        uint32_t seen = 0;
        for (;;) {
            {
                std::unique_lock<std::mutex> lk(mu);
                cv.wait(lk, [&] { return quit || gen != seen; });
                if (quit) return;
                seen = gen;
            }
            for (;;) {
                active.fetch_add(1, std::memory_order_acq_rel);
                const int j = claim(seen);
                if (j < 0) { active.fetch_sub(1, std::memory_order_acq_rel); break; }
                const int b = j % kRing;
                if (j >= kRing) {
                    const int previous_end = group_end(j - kRing);
                    while (issued.load(std::memory_order_acquire) < previous_end) std::this_thread::yield();
                    const auto status = cudaEventSynchronize(dma_done[(previous_end - 1) % kRing]);
                    if (profile_host) ++fence_waits;
                    if (status != cudaSuccess) ++fence_errors;
                }
                std::memcpy(buf[b], jobs[(size_t) j].src, jobs[(size_t) j].bytes);
                ready[(size_t) j].store(1, std::memory_order_release);
                active.fetch_sub(1, std::memory_order_acq_rel);
            }
        }
    }
    int claim(uint32_t g) {
        uint64_t cur = head.load(std::memory_order_acquire);
        for (;;) {
            if ((uint32_t) (cur >> 32) != g) return -1;
            const int n = (int) ((cur >> 16) & 0xffff), j = (int) (cur & 0xffff);
            if (j >= n) return -1;
            if (head.compare_exchange_weak(cur, cur + 1, std::memory_order_acq_rel, std::memory_order_acquire)) return j;
        }
    }
    /// A layer's jobs; the previous layer's are finished (finish()).
    void start(std::vector<Job>&& js, bool long_pipeline = false) {
        std::lock_guard<std::mutex> lk(mu);
        jobs = std::move(js);
        fence_batch = 1;
#if defined(STRATA_USE_HIP) && defined(_WIN32)
        if (long_pipeline) if (const char* v = std::getenv("STRATA_HIP_STAGE_BATCH"))
            fence_batch = std::clamp(std::atoi(v), 1, kRing);
#endif
        fence_records = 0;
        fence_waits.store(0);
        fence_errors.store(0);
        last_submitted = -1;
        fence_stream = nullptr;
        if (ready_cap < jobs.size()) {
            ready_cap = jobs.size() * 2;
            ready.reset(new std::atomic<int>[ready_cap]);
        }
        for (size_t i = 0; i < jobs.size(); ++i) ready[i].store(0, std::memory_order_relaxed);
        issued.store(0);
        ++gen;
        head.store((uint64_t) gen << 32 | (uint64_t) jobs.size() << 16, std::memory_order_release);
        cv.notify_all();
    }
    /// Job j's bytes, in a pinned buffer (waits for the copy).
    const uint8_t* wait(int j) {
        while (!ready[(size_t) j].load(std::memory_order_acquire)) std::this_thread::yield();
        return buf[j % kRing];
    }
    /// Exclusive end of j's ownership group; never crosses the host ring wrap.
    int group_end(int j) const {
        const int ring_begin = (j / kRing) * kRing;
        return std::min({ring_begin + ((j % kRing) / fence_batch + 1) * fence_batch,
                         ring_begin + kRing, (int) jobs.size()});
    }
    /// The launching thread queued job j's DMA on `copy`: its buffer is free once that is done.
    void issued_one(int j, cudaStream_t copy) {
        last_submitted = j;
        fence_stream = copy;
        // Every DMA in the group uses this same ordered copy stream. Publish only
        // after its final event record. A next-ring worker waits that group event
        // before writing its own buffer. Re-recording this event requires all next
        // group sources ready, which proves all prior waits have already returned.
        if (j + 1 == group_end(j)) {
            const auto status = cudaEventRecord(dma_done[j % kRing], copy);
            ++fence_records;
            if (status != cudaSuccess) ++fence_errors;
            issued.store(j + 1, std::memory_order_release);
        }
    }
    /// No job is running after this (the end of a layer, or an early return in the middle of one).
    void finish() {
        head.store((uint64_t) gen << 32, std::memory_order_release);   // n = 0: nothing more to claim
        // The issuer has joined before finish(). An early return may leave a
        // partially issued group. Fence its queued DMAs before unblocking workers;
        // an unrecorded group event must never permit overwriting an active source.
        if (fence_batch > 1 && last_submitted >= 0 && last_submitted + 1 != group_end(last_submitted)) {
            const auto status = cudaEventRecord(dma_done[(group_end(last_submitted) - 1) % kRing], fence_stream);
            ++fence_records;
            if (status != cudaSuccess) ++fence_errors;
        }
        last_submitted = -1;
        issued.store(1 << 30, std::memory_order_release);
        while (active.load(std::memory_order_acquire) != 0) std::this_thread::yield();
    }
};

}  // namespace strata::prefill::detail
