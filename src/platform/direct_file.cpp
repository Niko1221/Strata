// src/platform/direct_file.cpp - see include/strata/platform/direct_file.hpp.
#include "strata/platform/direct_file.hpp"

#include <chrono>
#include <cstdlib>
#include <cstring>
#include <deque>
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
#include <atomic>
#include <cerrno>
#include <linux/io_uring.h>
#include <sys/mman.h>
#include <sys/syscall.h>

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
// Phase L: O_DIRECT reads through io_uring with the caller's own queue depth (the PleReader's inflight count),
// instead of one synchronous pread per read.  A 4K-token prompt chunk references ~70 K scattered 4 KiB pages
// of the 28.8 GiB n-gram table; the depth lets the NVMe pipeline them (~70 K reads in under a second) where
// pread-at-depth-1 took several seconds.  Ring setup is best-effort: if it fails, reads fall back to the
// synchronous path, which is what the older code did.
namespace {
int uring_setup(unsigned entries, struct io_uring_params* p) { return (int) syscall(SYS_io_uring_setup, entries, p); }
int uring_enter(int fd, unsigned to_submit, unsigned min_complete, unsigned flags) {
    return (int) syscall(SYS_io_uring_enter, fd, to_submit, min_complete, flags, nullptr, 0);
}
}  // namespace

struct DirectFile::Impl {
    int fd = -1;
    uint64_t size = 0;
    std::deque<Completion> done;          // completions of the synchronous fallback path
    // io_uring (best effort; `uring` false runs the fallback)
    bool uring = false;
    int ring_fd = -1;
    unsigned pending = 0;                 // SQEs enqueued but not yet submitted
    void* sq_ring = nullptr;
    void* cq_ring = nullptr;
    struct io_uring_sqe* sqes = nullptr;
    size_t sq_ring_sz = 0, cq_ring_sz = 0, sqes_sz = 0;
    unsigned* sq_head = nullptr;
    unsigned* sq_tail = nullptr;
    unsigned* sq_mask = nullptr;
    unsigned* sq_array = nullptr;
    unsigned* cq_head = nullptr;
    unsigned* cq_tail = nullptr;
    unsigned* cq_mask = nullptr;
    unsigned entries = 0;

    void teardown() {
        if (uring) {
            while (pending > 0) {           // never leave SQEs that nobody will reap
                uring_enter(ring_fd, pending, 0, 0);
                pending = 0;
            }
            if (sq_ring) munmap(sq_ring, sq_ring_sz);
            if (cq_ring) munmap(cq_ring, cq_ring_sz);
            if (sqes) munmap(sqes, sqes_sz);
            sq_ring = cq_ring = nullptr;
            sqes = nullptr;
            if (ring_fd >= 0) ::close(ring_fd);
            ring_fd = -1;
            uring = false;
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
    // io_uring: one ring per open file (the engine opens one PLE table; MTP experts use a different path).
    struct io_uring_params p{};
    const int rfd = uring_setup(1024, &p);
    if (rfd < 0) return true;             // fallback to synchronous reads
    impl_->ring_fd = rfd;
    impl_->entries = p.sq_entries;
    impl_->sq_ring_sz = p.sq_off.array + (size_t) p.sq_entries * sizeof(unsigned);
    impl_->cq_ring_sz = p.cq_off.cqes + (size_t) p.cq_entries * sizeof(struct io_uring_cqe);
    impl_->sqes_sz = (size_t) p.sq_entries * sizeof(struct io_uring_sqe);
    void* sq = mmap(nullptr, impl_->sq_ring_sz, PROT_READ | PROT_WRITE, MAP_SHARED | MAP_POPULATE, rfd, IORING_OFF_SQ_RING);
    void* cq = mmap(nullptr, impl_->cq_ring_sz, PROT_READ | PROT_WRITE, MAP_SHARED | MAP_POPULATE, rfd, IORING_OFF_CQ_RING);
    void* sqes = mmap(nullptr, impl_->sqes_sz, PROT_READ | PROT_WRITE, MAP_SHARED | MAP_POPULATE, rfd, IORING_OFF_SQES);
    if (sq == MAP_FAILED || cq == MAP_FAILED || sqes == MAP_FAILED) { close(); err = "DirectFile: mmap io_uring rings"; return false; }
    impl_->sq_ring = sq;
    impl_->cq_ring = cq;
    impl_->sqes = (struct io_uring_sqe*) sqes;
    impl_->sq_head = (unsigned*) ((char*) sq + p.sq_off.head);
    impl_->sq_tail = (unsigned*) ((char*) sq + p.sq_off.tail);
    impl_->sq_mask = (unsigned*) ((char*) sq + p.sq_off.ring_mask);
    impl_->sq_array = (unsigned*) ((char*) sq + p.sq_off.array);
    impl_->cq_head = (unsigned*) ((char*) cq + p.cq_off.head);
    impl_->cq_tail = (unsigned*) ((char*) cq + p.cq_off.tail);
    impl_->cq_mask = (unsigned*) ((char*) cq + p.cq_off.ring_mask);
    impl_->uring = true;
    impl_->pending = 0;
    return true;
}

void DirectFile::close() {
    if (impl_ == nullptr) return;
    impl_->teardown();
    if (impl_->fd >= 0) ::close(impl_->fd);
    impl_->fd = -1;
    impl_->size = 0;
    impl_->done.clear();
}

bool DirectFile::is_open() const { return impl_->fd >= 0; }
uint64_t DirectFile::size() const { return impl_->size; }

bool DirectFile::submit(uint64_t offset, void* buffer, uint32_t length, uint64_t tag, std::string& err) {
    if (offset % alignment() || length % alignment() || ((uintptr_t) buffer) % alignment() || length == 0) {
        err = "DirectFile: unaligned request";
        return false;
    }
    Impl& m = *impl_;
    if (!m.uring) {
        const ssize_t got = pread(m.fd, buffer, length, (off_t) offset);
        m.done.push_back(Completion{tag, got < 0 ? 0u : (uint32_t) got, got >= 0});
        return true;
    }
    const unsigned tail = __atomic_load_n(m.sq_tail, __ATOMIC_RELAXED);
    const unsigned idx = tail & *m.sq_mask;
    struct io_uring_sqe* sqe = &m.sqes[idx];
    std::memset(sqe, 0, sizeof(*sqe));
    sqe->opcode = IORING_OP_READ;
    sqe->fd = m.fd;
    sqe->addr = (uint64_t) (uintptr_t) buffer;
    sqe->len = length;
    sqe->off = offset;
    sqe->user_data = tag;
    m.sq_array[idx] = idx;
    __atomic_store_n(m.sq_tail, tail + 1, __ATOMIC_RELEASE);
    ++m.pending;
    return true;
}

void DirectFile::wake() {}   // completions arrive on their own; a blocked wait re-arms after every batch

int DirectFile::wait(Completion* out, int max, int timeout_ms) {
    Impl& m = *impl_;
    if (!m.uring) {
        int n = 0;
        while (n < max && !m.done.empty()) {
            out[n++] = m.done.front();
            m.done.pop_front();
        }
        return n;
    }
    if (m.pending > 0) {
        const unsigned n = m.pending;
        m.pending = 0;
        if (uring_enter(m.ring_fd, n, 0, 0) < 0) {
            // Enter failed: nothing will ever complete.  Report the reads as failed so the caller errors out
            // instead of waiting forever.
            int written = 0;
            for (unsigned i = 0; i < n && written < max; ++i) out[written++] = Completion{~0ull, 0, false};
            return written;
        }
    }
    unsigned head = __atomic_load_n(m.cq_head, __ATOMIC_ACQUIRE);
    unsigned tail = __atomic_load_n(m.cq_tail, __ATOMIC_ACQUIRE);
    if (head == tail && timeout_ms != 0) {
        // Block for at least one completion (GETEVENTS also flushes anything still pending).
        uring_enter(m.ring_fd, 0, 1, IORING_ENTER_GETEVENTS);
        head = __atomic_load_n(m.cq_head, __ATOMIC_ACQUIRE);
        tail = __atomic_load_n(m.cq_tail, __ATOMIC_ACQUIRE);
    }
    int n = 0;
    while (head != tail && n < max) {
        struct io_uring_cqe* cqe = (struct io_uring_cqe*) ((char*) m.cq_ring +
                                                            ((size_t) (head & *m.cq_mask) * sizeof(struct io_uring_cqe)));
        Completion c{cqe->user_data, cqe->res < 0 ? 0u : (uint32_t) cqe->res, cqe->res >= 0};
        out[n++] = c;
        ++head;
    }
    __atomic_store_n(m.cq_head, head, __ATOMIC_RELEASE);
    return n;
}
#endif

}  // namespace strata::platform
