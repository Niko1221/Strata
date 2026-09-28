// src/platform/direct_file.cpp - see include/strata/platform/direct_file.hpp.
#include "strata/platform/direct_file.hpp"

#include <chrono>
#include <cstdlib>
#include <cstring>
#include <deque>
#include <algorithm>
#include <thread>
#include <condition_variable>
#include <mutex>
#include <vector>

#if defined(_WIN32)
#define WIN32_LEAN_AND_MEAN
#define NOMINMAX
#include <windows.h>
#else
#include <fcntl.h>
#include <sys/stat.h>
#include <unistd.h>
#endif

namespace strata::platform {

double now_us() {
    using namespace std::chrono;
    return (double) duration_cast<nanoseconds>(steady_clock::now().time_since_epoch()).count() / 1000.0;
}

void* DirectFile::alloc_aligned(size_t bytes) {
#if defined(_WIN32)
    return VirtualAlloc(nullptr, bytes, MEM_RESERVE | MEM_COMMIT, PAGE_READWRITE);
#else
    void* p = nullptr;
    return posix_memalign(&p, alignment(), bytes) == 0 ? p : nullptr;
#endif
}

void DirectFile::free_aligned(void* p) {
    if (p == nullptr) return;
#if defined(_WIN32)
    VirtualFree(p, 0, MEM_RELEASE);
#else
    std::free(p);
#endif
}

#if defined(_WIN32)
// ------------------------------------------------------------------------------------------------ Windows
namespace {
/// One in-flight request. The OVERLAPPED must be the first member so a completion packet's OVERLAPPED*
/// converts back to the request.
struct Req {
    OVERLAPPED ov;
    uint64_t tag;
};
}  // namespace

struct DirectFile::Impl {
    HANDLE file = INVALID_HANDLE_VALUE;
    HANDLE port = nullptr;
    uint64_t size = 0;
    std::deque<Req*> free_reqs;
    std::vector<Req*> all_reqs;
    std::deque<Completion> immediate;   // requests that failed after being counted as queued

    Req* take() {
        if (free_reqs.empty()) {
            Req* r = new Req();
            all_reqs.push_back(r);
            return r;
        }
        Req* r = free_reqs.front();
        free_reqs.pop_front();
        return r;
    }
};

DirectFile::DirectFile() : impl_(new Impl) {}
DirectFile::~DirectFile() {
    close();
    for (Req* r : impl_->all_reqs) delete r;
    delete impl_;
}

bool DirectFile::open(const std::string& path, std::string& err) {
    close();
    const int wlen = MultiByteToWideChar(CP_UTF8, 0, path.c_str(), -1, nullptr, 0);
    std::wstring wpath((size_t) (wlen > 0 ? wlen : 1), L'\0');
    MultiByteToWideChar(CP_UTF8, 0, path.c_str(), -1, wpath.data(), wlen);
    // NO_BUFFERING: the cache manager never holds these pages, so the table cannot grow into RAM.
    // RANDOM_ACCESS: no read-ahead. OVERLAPPED: many reads in flight, completed through the port below.
    impl_->file = CreateFileW(wpath.c_str(), GENERIC_READ, FILE_SHARE_READ, nullptr, OPEN_EXISTING,
                              FILE_FLAG_NO_BUFFERING | FILE_FLAG_OVERLAPPED | FILE_FLAG_RANDOM_ACCESS, nullptr);
    if (impl_->file == INVALID_HANDLE_VALUE) {
        err = "DirectFile: cannot open " + path + " (error " + std::to_string(GetLastError()) + ")";
        return false;
    }
    LARGE_INTEGER sz;
    if (!GetFileSizeEx(impl_->file, &sz)) {
        err = "DirectFile: cannot size " + path;
        close();
        return false;
    }
    impl_->size = (uint64_t) sz.QuadPart;
    impl_->port = CreateIoCompletionPort(impl_->file, nullptr, 0, 1);
    if (impl_->port == nullptr) {
        err = "DirectFile: CreateIoCompletionPort failed (error " + std::to_string(GetLastError()) + ")";
        close();
        return false;
    }
    // Completions of reads that finish synchronously must still be queued to the port, so every submit
    // produces exactly one packet and `wait` is the only completion path.
    return true;
}

void DirectFile::close() {
    if (impl_->port != nullptr) CloseHandle(impl_->port);
    if (impl_->file != INVALID_HANDLE_VALUE) CloseHandle(impl_->file);
    impl_->port = nullptr;
    impl_->file = INVALID_HANDLE_VALUE;
    impl_->size = 0;
    impl_->immediate.clear();
}

bool DirectFile::is_open() const { return impl_->file != INVALID_HANDLE_VALUE; }
uint64_t DirectFile::size() const { return impl_->size; }

bool DirectFile::submit(uint64_t offset, void* buffer, uint32_t length, uint64_t tag, std::string& err) {
    if (!is_open()) { err = "DirectFile: not open"; return false; }
    if (offset % alignment() || length % alignment() || ((uintptr_t) buffer) % alignment() || length == 0) {
        err = "DirectFile: unaligned request";
        return false;
    }
    Req* r = impl_->take();
    std::memset(&r->ov, 0, sizeof r->ov);
    r->ov.Offset = (DWORD) (offset & 0xFFFFFFFFull);
    r->ov.OffsetHigh = (DWORD) (offset >> 32);
    r->tag = tag;
    if (!ReadFile(impl_->file, buffer, length, nullptr, &r->ov)) {
        const DWORD e = GetLastError();
        if (e != ERROR_IO_PENDING) {
            impl_->free_reqs.push_back(r);
            if (e == ERROR_HANDLE_EOF) {           // at or past end of file: a zero-byte completion
                impl_->immediate.push_back(Completion{tag, 0, true});
                return true;
            }
            err = "DirectFile: ReadFile failed (error " + std::to_string(e) + ")";
            return false;
        }
    }
    return true;
}

int DirectFile::wait(Completion* out, int max, int timeout_ms) {
    int n = 0;
    while (n < max && !impl_->immediate.empty()) {
        out[n++] = impl_->immediate.front();
        impl_->immediate.pop_front();
    }
    if (n == max || !is_open()) return n;
    OVERLAPPED_ENTRY entries[64];
    const ULONG want = (ULONG) (max - n < 64 ? max - n : 64);
    ULONG got = 0;
    const DWORD t = timeout_ms < 0 ? INFINITE : (DWORD) timeout_ms;
    if (!GetQueuedCompletionStatusEx(impl_->port, entries, want, &got, n > 0 ? 0 : t, FALSE)) return n;
    for (ULONG i = 0; i < got; ++i) {
        if (entries[i].lpOverlapped == nullptr) {          // a wake() packet, not a read
            out[n++] = Completion{WAKE_TAG, 0, true};
            continue;
        }
        Req* r = (Req*) entries[i].lpOverlapped;
        const uint32_t status = (uint32_t) r->ov.Internal;   // an NTSTATUS
        Completion c;
        c.tag = r->tag;
        c.bytes = entries[i].dwNumberOfBytesTransferred;
        // STATUS_END_OF_FILE (0xC0000011) is a legal short read at the table's last page.
        c.ok = status == 0 || status == 0xC0000011u;
        out[n++] = c;
        impl_->free_reqs.push_back(r);
    }
    return n;
}

void DirectFile::wake() {
    if (impl_->port != nullptr) PostQueuedCompletionStatus(impl_->port, 0, 0, nullptr);
}

#else
// ------------------------------------------------------------------------------------------------ POSIX
// A read used to complete inside `submit` (a plain pread), so on Linux the PLE reader's "in flight" reads ran one
// after another on the caller's thread: ~16 table rows per token read back to back, 8-9% of decode spent blocked in
// "submit".  Now `submit` queues the request and a pool of threads issues the preads concurrently, which gives the
// NVMe the queue depth the Windows overlapped path always had.  Same contract: `wait` returns completions (blocking
// up to its timeout), and `wake` makes a blocked `wait` return one WAKE_TAG completion.
struct DirectFile::Impl {
    struct Req { uint64_t offset; void* buffer; uint32_t length; uint64_t tag; };
    int fd = -1;
    uint64_t size = 0;
    std::mutex mu;
    std::condition_variable cv_req, cv_done;
    std::deque<Req> reqs;
    std::deque<Completion> done;
    std::vector<std::thread> workers;
    int wakes = 0;
    bool stop = false;

    void work() {
        for (;;) {
            Req r;
            {
                std::unique_lock<std::mutex> lk(mu);
                cv_req.wait(lk, [&] { return stop || !reqs.empty(); });
                if (stop && reqs.empty()) return;
                r = reqs.front();
                reqs.pop_front();
            }
            uint32_t got = 0;
            bool ok = true;
            while (got < r.length) {
                const ssize_t n = pread(fd, (uint8_t*) r.buffer + got, r.length - got, (off_t) (r.offset + got));
                if (n < 0) { ok = false; break; }
                if (n == 0) break;                  // end of file: a legal short read at the table's last page
                got += (uint32_t) n;
            }
            {
                std::lock_guard<std::mutex> lk(mu);
                done.push_back(Completion{r.tag, got, ok});
            }
            cv_done.notify_all();
        }
    }
};

DirectFile::DirectFile() : impl_(new Impl) {}
DirectFile::~DirectFile() { close(); delete impl_; }

bool DirectFile::open(const std::string& path, std::string& err) {
    close();
    impl_->fd = ::open(path.c_str(), O_RDONLY | O_DIRECT);
    if (impl_->fd < 0) { err = "DirectFile: cannot open " + path; return false; }
    struct stat st;
    if (fstat(impl_->fd, &st) != 0) { err = "DirectFile: cannot size " + path; close(); return false; }
    impl_->size = (uint64_t) st.st_size;
    const char* env = std::getenv("STRATA_PLE_IO_THREADS");
    const int n = env ? std::max(1, std::atoi(env)) : 32;
    impl_->stop = false;
    for (int i = 0; i < n; ++i) impl_->workers.emplace_back([this] { impl_->work(); });
    return true;
}

void DirectFile::close() {
    {
        std::lock_guard<std::mutex> lk(impl_->mu);
        impl_->stop = true;
    }
    impl_->cv_req.notify_all();
    for (auto& t : impl_->workers) t.join();
    impl_->workers.clear();
    if (impl_->fd >= 0) ::close(impl_->fd);
    impl_->fd = -1;
    impl_->size = 0;
    impl_->reqs.clear();
    impl_->done.clear();
    impl_->wakes = 0;
}

bool DirectFile::is_open() const { return impl_->fd >= 0; }
uint64_t DirectFile::size() const { return impl_->size; }

bool DirectFile::submit(uint64_t offset, void* buffer, uint32_t length, uint64_t tag, std::string& err) {
    if (offset % alignment() || length % alignment() || ((uintptr_t) buffer) % alignment() || length == 0) {
        err = "DirectFile: unaligned request";
        return false;
    }
    {
        std::lock_guard<std::mutex> lk(impl_->mu);
        impl_->reqs.push_back(Impl::Req{offset, buffer, length, tag});
    }
    impl_->cv_req.notify_one();
    return true;
}

void DirectFile::wake() {
    {
        std::lock_guard<std::mutex> lk(impl_->mu);
        ++impl_->wakes;
    }
    impl_->cv_done.notify_all();
}

int DirectFile::wait(Completion* out, int max, int timeout_ms) {
    std::unique_lock<std::mutex> lk(impl_->mu);
    auto ready = [&] { return !impl_->done.empty() || impl_->wakes > 0; };
    if (!ready()) {
        if (timeout_ms < 0) impl_->cv_done.wait(lk, ready);
        else if (timeout_ms > 0) impl_->cv_done.wait_for(lk, std::chrono::milliseconds(timeout_ms), ready);
    }
    int n = 0;
    if (impl_->wakes > 0 && n < max) {
        --impl_->wakes;
        out[n++] = Completion{WAKE_TAG, 0, true};
    }
    while (n < max && !impl_->done.empty()) {
        out[n++] = impl_->done.front();
        impl_->done.pop_front();
    }
    return n;
}
#endif

}  // namespace strata::platform
