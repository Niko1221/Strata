// PinnedRing: small and large uploads interleaved, the source overwritten the moment each call returns, the stream held
// behind a kernel that waits for the host so the small ring wraps (and waits on its own slots) while the copies are
// still queued; then every destination byte is checked, through d2h_sync and a plain copy.
#include "strata/core/pinned_ring.hpp"

#include <atomic>
#include <chrono>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <thread>
#include <vector>

static void check(bool ok, const char* what) {
    if (!ok) { std::fprintf(stderr, "FAIL: %s\n", what); std::exit(1); }
}
static void gpu(cudaError_t e) { check(e == cudaSuccess, cudaGetErrorString(e)); }

__global__ void wait_for_host(volatile unsigned* flag) {
    while (*flag == 0) { }
}

static uint8_t pattern(int upload, size_t at) { return (uint8_t) (upload * 131u + at * 7u + (at >> 9)); }

int main() {
    using strata::core::PinnedRing;
    cudaStream_t s;
    gpu(cudaStreamCreateWithFlags(&s, cudaStreamNonBlocking));
    unsigned *flag_h, *flag_d;
    gpu(cudaHostAlloc((void**) &flag_h, sizeof(unsigned), cudaHostAllocMapped));
    gpu(cudaHostGetDevicePointer((void**) &flag_d, flag_h, 0));

    // sizes: the first 200 small (1 B .. 64 KiB: the small ring wraps three times while the device is held), then every
    // fifth large (64 KiB + 1 .. 1 MiB, the growing slots)
    const int n = 600;
    std::vector<size_t> size(n), at(n);
    size_t total = 0;
    uint32_t r = 12345;
    auto rnd = [&] { r = r * 1103515245u + 12345u; return r >> 8; };
    for (int i = 0; i < n; ++i) {
        size[i] = i >= 200 && i % 5 == 4 ? 64 * 1024 + 1 + rnd() % (1 << 20) : 1 + rnd() % (64 * 1024);
        at[i] = total;
        total += (size[i] + 255) & ~(size_t) 255;
    }
    uint8_t* dev = nullptr;
    gpu(cudaMalloc((void**) &dev, total));
    gpu(cudaMemset(dev, 0, total));

    for (int pass = 0; pass < 2; ++pass) {   // pass 1: after reserve(), as STRATA_PIPED_CONCURRENT does
        PinnedRing ring;
        if (pass == 1) check(ring.reserve(256 * 1024), "reserve");
        *flag_h = 0;
        wait_for_host<<<1, 1, 0, s>>>(flag_d);
        gpu(cudaGetLastError());
        std::atomic<bool> go{false};
        std::thread release([&] {   // the host fills more than kSmallSlots slots before the device may copy any
            while (!go) std::this_thread::sleep_for(std::chrono::milliseconds(1));
            std::this_thread::sleep_for(std::chrono::milliseconds(200));
            __atomic_store_n(flag_h, 1u, __ATOMIC_SEQ_CST);
        });
        go = true;
        const auto t0 = std::chrono::steady_clock::now();
        std::vector<uint8_t> src;
        for (int i = 0; i < n; ++i) {
            src.resize(size[i]);
            for (size_t b = 0; b < size[i]; ++b) src[b] = pattern(i + pass * n, b);
            if (i % 3 == 1) {   // the gather form: acquire a slot, fill it, submit
                void* st = ring.acquire(size[i]);
                check(st != nullptr, "acquire");
                std::memcpy(st, src.data(), size[i]);
                gpu(ring.submit(dev + at[i], st, size[i], s));
            } else {
                gpu(ring.h2d(dev + at[i], src.data(), size[i], s));
            }
            for (size_t b = 0; b < size[i]; ++b) src[b] = 0xA5;   // the caller may reuse its memory at once
        }
        const double fill_ms = std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - t0).count();
        const bool held = __atomic_load_n(flag_h, __ATOMIC_SEQ_CST) == 0;
        release.join();
        std::printf("pass %d: the uploads were queued in %.1f ms, device still held at the end: %s\n", pass, fill_ms, held ? "yes" : "no");
        // d2h_sync of one region (it drains the stream), then a plain copy of all of it
        std::vector<uint8_t> one(size[7]);
        gpu(ring.d2h_sync(one.data(), dev + at[7], size[7], s));
        for (size_t b = 0; b < size[7]; ++b) check(one[b] == pattern(7 + pass * n, b), "d2h_sync bytes");
        std::vector<uint8_t> back(total);
        gpu(cudaMemcpy(back.data(), dev, total, cudaMemcpyDeviceToHost));
        size_t bad = 0;
        for (int i = 0; i < n; ++i)
            for (size_t b = 0; b < size[i]; ++b)
                if (back[at[i] + b] != pattern(i + pass * n, b)) ++bad;
        if (bad) { std::fprintf(stderr, "FAIL: pass %d, %zu wrong bytes\n", pass, bad); return 1; }
        std::printf("pass %d: %d uploads (%d small slots, %d growing), %zu bytes, all correct\n", pass, n,
                    PinnedRing::kSmallSlots, PinnedRing::kSlots, total);
    }
    gpu(cudaFree(dev));
    gpu(cudaFreeHost(flag_h));
    gpu(cudaStreamDestroy(s));
    std::printf("pinned_ring_test: OK\n");
    return 0;
}
