// src/core/trunk_stream.cpp
#include "strata/core/trunk_stream.hpp"
#include "strata/core/weights.hpp"
#include "strata/platform/memory.hpp"
#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <algorithm>

#if defined(_WIN32)
#include <windows.h>
#include <io.h>
#else
#include <fcntl.h>
#include <unistd.h>
#include <sys/mman.h>
#endif

namespace strata::core {

static double now_s() {
    return std::chrono::duration<double>(std::chrono::steady_clock::now().time_since_epoch()).count();
}

TrunkStreamer::~TrunkStreamer() {
    {
        std::lock_guard<std::mutex> g(mu_);
        stop_ = true;
    }
    cv_.notify_all();
    if (io_thread_.joinable()) io_thread_.join();

#if defined(_WIN32)
    if (fd_ >= 0) _close(fd_);
#else
    if (fd_ >= 0) ::close(fd_);
#endif

    // Free slots
    for (void* p : pin_slots_) {
        if (p) std::free(p);
    }
    for (void* p : ring_slots_) {
        if (p) std::free(p);
    }
}

bool TrunkStreamer::open(const std::string& pack_dir, WeightTable& wt, uint64_t budget_bytes, std::string& err) {
    pack_dir_ = pack_dir;
    
    // Not implemented fully for this snippet
    // In a real implementation we would parse index.txt here or rely on wt
    // Then allocate ring_slots_ using strata::platform::memory_alloc
    return true;
}

bool TrunkStreamer::bind_layer(int L, WeightTable& wt, std::string& err) {
    // If not pinned, wait for ring slot
    return true;
}

void TrunkStreamer::prefetch_layer(int L) {
    std::lock_guard<std::mutex> g(mu_);
    req_layer_ = L;
    req_busy_ = true;
    cv_.notify_one();
}

void TrunkStreamer::io_loop() {
    while (true) {
        int L = -1;
        {
            std::unique_lock<std::mutex> lock(mu_);
            cv_.wait(lock, [this] { return stop_ || req_busy_; });
            if (stop_) break;
            L = req_layer_;
        }
        
        std::string err;
        bool ok = read_layer_sync(L, nullptr, err);
        
        {
            std::lock_guard<std::mutex> g(mu_);
            req_busy_ = false;
            req_done_ = true;
            io_err_ = ok ? "" : err;
        }
        cv_.notify_all();
    }
}

bool TrunkStreamer::read_layer_sync(int L, void* dst, std::string& err) {
    // Read from fd_ to dst
    return true;
}

void TrunkStreamer::report() const {
    std::printf("TrunkStreamer: hits=%llu, misses=%llu, read=%.2f GB in %.2f s\n",
                (unsigned long long)hits_, (unsigned long long)misses_,
                (double)bytes_read_ / 1e9, load_seconds_);
}

} // namespace strata::core
