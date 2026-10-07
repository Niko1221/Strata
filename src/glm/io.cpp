// src/glm/io.cpp - unbuffered shard reads.  See the header.
#include "strata/glm/io.hpp"

#include <algorithm>
#include <cerrno>
#include <cstring>

#if defined(_WIN32)
#ifndef NOMINMAX
#define NOMINMAX
#endif
#include <windows.h>
#else
#include <fcntl.h>
#include <stdlib.h>
#include <unistd.h>
#endif

namespace strata::glm {

void* aligned_alloc_bytes(size_t bytes) {
#if defined(_WIN32)
    return VirtualAlloc(nullptr, bytes, MEM_RESERVE | MEM_COMMIT, PAGE_READWRITE);
#else
    void* q = nullptr;
    if (posix_memalign(&q, kAlign, bytes) != 0) return nullptr;
    return q;
#endif
}

void aligned_free_bytes(void* p) {
    if (!p) return;
#if defined(_WIN32)
    VirtualFree(p, 0, MEM_RELEASE);
#else
    free(p);
#endif
}

namespace {

#if defined(_WIN32)
using Fd = HANDLE;
const Fd kBad = INVALID_HANDLE_VALUE;
Fd open_direct(const std::string& path) {
    const int n = MultiByteToWideChar(CP_UTF8, 0, path.c_str(), -1, nullptr, 0);
    std::wstring w((size_t) n, L'\0');
    MultiByteToWideChar(CP_UTF8, 0, path.c_str(), -1, w.data(), n);
    return CreateFileW(w.c_str(), GENERIC_READ, FILE_SHARE_READ, nullptr, OPEN_EXISTING, FILE_FLAG_NO_BUFFERING, nullptr);
}
void close_fd(Fd f) { CloseHandle(f); }
/// Read len bytes at off; bytes past the end of the file are left as they are.  Returns false on an I/O error.
bool pread_direct(Fd f, uint64_t off, uint64_t len, void* dst) {
    uint8_t* d = (uint8_t*) dst;
    while (len > 0) {
        const DWORD want = (DWORD) std::min<uint64_t>(len, 64ull << 20);
        OVERLAPPED ov{};
        ov.Offset = (DWORD) (off & 0xFFFFFFFFull);
        ov.OffsetHigh = (DWORD) (off >> 32);
        DWORD got = 0;
        if (!ReadFile(f, d, want, &got, &ov)) {
            if (GetLastError() == ERROR_HANDLE_EOF) return true;
            return false;
        }
        if (got == 0) return true;   // end of file
        d += got;
        off += got;
        len -= got;
        if (got < want) return true;
    }
    return true;
}
#else
using Fd = int;
const Fd kBad = -1;
Fd open_direct(const std::string& path) {
#ifdef O_DIRECT
    int fd = ::open(path.c_str(), O_RDONLY | O_DIRECT);
    if (fd >= 0) return fd;
#endif
    return ::open(path.c_str(), O_RDONLY);
}
void close_fd(Fd f) { ::close(f); }
bool pread_direct(Fd f, uint64_t off, uint64_t len, void* dst) {
    uint8_t* d = (uint8_t*) dst;
    while (len > 0) {
        const size_t want = (size_t)std::min<uint64_t>(len, 64ull << 20);
        const ssize_t got = ::pread(f, d, want, (off_t) off);
        if (got < 0) {
            if (errno == EINTR) continue;
#ifdef O_DIRECT
            // Some WSL/network filesystems accept O_DIRECT at open but refuse reads.
            const int read_error = errno;
            const int flags = ::fcntl(f, F_GETFL);
            if (read_error == EINVAL && flags >= 0 && (flags & O_DIRECT) && ::fcntl(f, F_SETFL, flags & ~O_DIRECT) == 0) continue;
#endif
            return false;
        }
        if (got == 0) return true;
        d += got;
        off += (uint64_t) got;
        len -= (uint64_t) got;
        if ((size_t)got < want) return true; // alignment padding beyond EOF, as on Windows
    }
    return true;
}
#endif

}  // namespace

struct IoPool::Handles {
    std::vector<Fd> fd;
    explicit Handles(size_t n) : fd(n, kBad) {}
    ~Handles() {
        for (Fd f : fd)
            if (f != kBad) close_fd(f);
    }
    Fd get(const std::vector<std::string>& shards, int s) {
        if (fd[s] == kBad) fd[s] = open_direct(shards[s]);
        return fd[s];
    }
};

IoPool::IoPool(const std::vector<std::string>& shards, int threads) : shards_(shards) {
    for (int i = 0; i < std::max(1, threads); ++i) workers_.emplace_back([this, i] { loop(i); });
}

IoPool::~IoPool() {
    {
        std::lock_guard<std::mutex> lk(m_);
        quit_ = true;
    }
    cv_.notify_all();
    for (auto& t : workers_) t.join();
}

void IoPool::submit(ReadJob job) {
    {
        std::lock_guard<std::mutex> lk(m_);
        q_.push_back(std::move(job));
    }
    cv_.notify_one();
}

bool IoPool::read_now(int shard, uint64_t off, uint64_t len, void* dst, std::string& err) {
    static thread_local Handles* h = nullptr;
    static thread_local const IoPool* owner = nullptr;
    if (owner != this) {
        delete h;
        h = new Handles(shards_.size());
        owner = this;
    }
    Fd f = h->get(shards_, shard);
    if (f == kBad) { err = "cannot open " + shards_[shard]; return false; }
    if (!pread_direct(f, off, len, dst)) { err = "read failed in " + shards_[shard]; return false; }
    bytes_.fetch_add(len, std::memory_order_relaxed);
    return true;
}

void IoPool::loop(int) {
    Handles h(shards_.size());
    for (;;) {
        ReadJob job;
        {
            std::unique_lock<std::mutex> lk(m_);
            cv_.wait(lk, [&] { return quit_ || !q_.empty(); });
            if (quit_ && q_.empty()) return;
            job = std::move(q_.front());
            q_.pop_front();
        }
        Fd f = h.get(shards_, job.shard);
        const bool ok = f != kBad && pread_direct(f, job.off, job.len, job.dst);
        if (ok) bytes_.fetch_add(job.len, std::memory_order_relaxed);
        if (job.done) job.done(ok);
    }
}

}  // namespace strata::glm
