// include/strata/glm/io.hpp - unbuffered reads from the shards, from a pool of I/O threads.
//
// WHY UNBUFFERED, AND WHY ONE HANDLE PER THREAD.  Every token of GLM-5.3 reads up to 600 routed experts of 21 MB
// from the SSD.  Through the OS file cache those bytes would be copied twice and would push the engine's own RAM
// expert cache out of memory, so the reads bypass the cache (FILE_FLAG_NO_BUFFERING / O_DIRECT).  On Windows a
// synchronous handle serialises concurrent ReadFile calls on one object, so each I/O thread opens its own handle
// per shard: measured on this checkpoint (colibri's iobench, 20.25 MB random reads, Kingston NV3 on PCIe 4.0),
// 3.69 GB/s with one thread and 4.54 GB/s with eight.
//
// Contract: offsets, lengths and buffers given to `submit` are multiples of `kAlign`.  `aligned_span` turns any
// byte range into the covering aligned range.
#pragma once

#include <atomic>
#include <condition_variable>
#include <cstdint>
#include <deque>
#include <functional>
#include <mutex>
#include <string>
#include <thread>
#include <vector>

namespace strata::glm {

constexpr uint64_t kAlign = 4096;

struct AlignedSpan {
    uint64_t off = 0;    ///< aligned start
    uint64_t len = 0;    ///< aligned length
    uint64_t skip = 0;   ///< where the wanted bytes start inside the aligned read
};
inline AlignedSpan aligned_span(uint64_t off, uint64_t bytes) {
    AlignedSpan a;
    a.off = off & ~(kAlign - 1);
    a.skip = off - a.off;
    a.len = (a.skip + bytes + kAlign - 1) & ~(kAlign - 1);
    return a;
}

void* aligned_alloc_bytes(size_t bytes);   ///< page-aligned, zero-initialised lazily by the OS
void aligned_free_bytes(void* p);

/// One read job.  `done(ok)` runs on the I/O thread that finished it.
struct ReadJob {
    int shard = 0;
    uint64_t off = 0, len = 0;
    void* dst = nullptr;
    std::function<void(bool)> done;
};

class IoPool {
public:
    IoPool(const std::vector<std::string>& shards, int threads);
    ~IoPool();
    IoPool(const IoPool&) = delete;
    IoPool& operator=(const IoPool&) = delete;

    void submit(ReadJob job);
    /// Read synchronously on the calling thread (used by the loader); same alignment contract.
    bool read_now(int shard, uint64_t off, uint64_t len, void* dst, std::string& err);
    int threads() const { return (int) workers_.size(); }
    uint64_t bytes_read() const { return bytes_.load(std::memory_order_relaxed); }

private:
    struct Handles;   // per-thread file handles, opened on first use
    void loop(int id);

    std::vector<std::string> shards_;
    std::vector<std::thread> workers_;
    std::mutex m_;
    std::condition_variable cv_;
    std::deque<ReadJob> q_;
    bool quit_ = false;
    std::atomic<uint64_t> bytes_{0};
};

}  // namespace strata::glm
