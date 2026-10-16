// bench/dram_probe.cpp - what this box's DRAM can do for the expert access pattern.
// Two arms over the same buffer, same thread count:
//   seq    : each thread streams a contiguous region (the pattern PCIe/DMA-friendly code sees)
//   random : each thread reads whole random blobs of --blob bytes (what the CPU expert pool does)
// The gap is the cost of page walks + DRAM page locality at the engine's blob size.
#include <windows.h>
#include <cstdio>
#include <cstdint>
#include <cstring>
#include <cstdlib>
#include <vector>
#include <thread>
#include <atomic>
#include <chrono>

static std::atomic<bool> stop_flag{false};
static std::atomic<uint64_t> bytes_done{0};

static uint64_t rnd(uint64_t& s) { s ^= s << 13; s ^= s >> 7; s ^= s << 17; return s; }

static void worker(const uint8_t* buf, size_t buf_bytes, size_t blob, int mode, int id, int nthreads) {
    uint64_t seed = 0x9E3779B97F4A7C15ull + (uint64_t) id * 0xFF51AFD7ED558CCDull;
    const size_t n_blobs = buf_bytes / blob;
    const size_t chunk = (buf_bytes / nthreads) & ~(blob - 1);
    const uint64_t* words = nullptr;
    uint64_t acc = 0;
    uint64_t local = 0;
    while (!stop_flag.load(std::memory_order_relaxed)) {
        size_t off;
        if (mode == 0) {                       // seq: walk this thread's own slice, wrap at its end
            static thread_local size_t pos = (size_t) id * chunk;
            off = pos;
            pos += blob;
            if (pos >= (size_t) (id + 1) * chunk) pos = (size_t) id * chunk;
        } else {                               // random: a fresh blob anywhere in the buffer
            off = (size_t) (rnd(seed) % n_blobs) * blob;
        }
        words = (const uint64_t*) (buf + off);
        const size_t n = blob / 8;
        for (size_t i = 0; i < n; i += 8) {
            uint64_t v0 = words[i], v1 = words[i + 1], v2 = words[i + 2], v3 = words[i + 3];
            uint64_t v4 = words[i + 4], v5 = words[i + 5], v6 = words[i + 6], v7 = words[i + 7];
            acc += v0 + v1 + v2 + v3 + v4 + v5 + v6 + v7;
        }
        local += blob;
        if ((local & 0x3FFFFFF) == 0) bytes_done.fetch_add(local, std::memory_order_relaxed), local = 0;
    }
    bytes_done.fetch_add(local, std::memory_order_relaxed);
    if (acc == 0x1234567812345678ull) std::printf("x");
}

int main(int argc, char** argv) {
    size_t gib = argc > 1 ? (size_t) std::atoll(argv[1]) : 24;
    int nthreads = argc > 2 ? std::atoi(argv[2]) : 21;
    size_t blob = argc > 3 ? (size_t) std::atoll(argv[3]) : (1956 * 1024);
    const size_t bytes = gib * 1024ull * 1024ull * 1024ull;
    std::printf("buffer %zu GiB, %d threads, blob %zu KiB, large-page minimum %zu KiB\n",
                gib, nthreads, blob / 1024, GetLargePageMinimum() / 1024);
    void* p = VirtualAlloc(nullptr, bytes, MEM_RESERVE | MEM_COMMIT, PAGE_READWRITE);
    if (p == nullptr) { std::printf("VirtualAlloc failed: %lu\n", (unsigned long) GetLastError()); return 1; }
    // Touch every page so the buffer is really resident before it is timed.
    std::vector<std::thread> touch;
    for (int t = 0; t < nthreads; ++t)
        touch.emplace_back([p, bytes, t, nthreads] {
            uint8_t* b = (uint8_t*) p;
            for (size_t i = (size_t) t * 4096; i < bytes; i += (size_t) nthreads * 4096) b[i] = 1;
        });
    for (auto& th : touch) th.join();
    std::printf("buffer touched\n");

    for (int mode = 0; mode < 2; ++mode) {
        for (int pass = 0; pass < 3; ++pass) {
            bytes_done.store(0);
            stop_flag.store(false);
            std::vector<std::thread> th;
            for (int t = 0; t < nthreads; ++t) th.emplace_back(worker, (const uint8_t*) p, bytes, blob, mode, t, nthreads);
            const auto t0 = std::chrono::steady_clock::now();
            std::this_thread::sleep_for(std::chrono::seconds(3));
            stop_flag.store(true);
            for (auto& x : th) x.join();
            const double s = std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count();
            const double gb = (double) bytes_done.load() / 1e9;
            std::printf("  %-7s pass %d  %8.2f GB/s\n", mode == 0 ? "seq" : "random", pass, gb / s);
        }
    }
    VirtualFree(p, 0, MEM_RELEASE);
    return 0;
}
