// src/core/gguf_expert_source.cpp - see the header. Plain C++, no device code.
#include "strata/core/gguf_expert_source.hpp"
#include "strata/kernels/cpu/expert_layout.hpp"

#if defined(_WIN32)
// before <sycl/sycl.hpp>, which (through dpct/device.hpp) includes <windows.h>: no min/max macros, no shell headers
// (guarded: sycl/CMakeLists.txt defines both for every target on Windows)
#ifndef WIN32_LEAN_AND_MEAN
#define WIN32_LEAN_AND_MEAN
#endif
#ifndef NOMINMAX
#define NOMINMAX
#endif
#endif
#include <cstring>
#include <algorithm>
#include <dpct/dpct.hpp>
#include <sycl/sycl.hpp>
#include <thread>
#include <atomic>
#if defined(_WIN32)
#include <windows.h>
#include <fcntl.h>
#include <io.h>
#else
#include <fcntl.h>
#include <unistd.h>
#endif

namespace strata::core {

namespace {
constexpr size_t kRing = 512;   // blobs alive at once: the prompt path holds a layer's worth of streamed experts

#if defined(_WIN32)
// Windows: the three POSIX calls this file is built on (open, pread, close), as Windows has them.  The reads are
// positional and concurrent - the profile fill's reader threads and the prompt path's stager share one handle per
// shard - so the handle is opened overlapped and every read carries its own offset and its thread's event; on a
// synchronous handle Windows would serialize them on the file object.  The HANDLE travels as a CRT fd
// (_open_osfhandle), so the header's `std::vector<int> fds_` stays as it is; nothing ever calls _read on it.
// open(2) read-only: the UTF-8 path as UTF-16 (a model under a folder with a non-ASCII name), as expert_source.cpp
int fd_open_ro(const std::string& path) {
    const int wide = MultiByteToWideChar(CP_UTF8, 0, path.c_str(), -1, nullptr, 0);
    std::vector<wchar_t> w((size_t) (wide > 0 ? wide : 1), L'\0');
    if (wide > 0) MultiByteToWideChar(CP_UTF8, 0, path.c_str(), -1, w.data(), wide);
    const HANDLE h = CreateFileW(w.data(), GENERIC_READ, FILE_SHARE_READ, nullptr, OPEN_EXISTING,
                                 FILE_ATTRIBUTE_NORMAL | FILE_FLAG_OVERLAPPED, nullptr);
    if (h == INVALID_HANDLE_VALUE) return -1;
    const int fd = _open_osfhandle((intptr_t) h, _O_RDONLY);
    if (fd < 0) CloseHandle(h);
    return fd;
}
// pread(2): one read at `off`, at most 1 GiB (ReadFile counts in a DWORD; the callers loop on short reads anyway)
int64_t fd_pread(int fd, void* dst, size_t n, uint64_t off) {
    struct Event {   // this thread's completion event, closed when the thread ends
        HANDLE h = CreateEventW(nullptr, TRUE, FALSE, nullptr);
        ~Event() { if (h != nullptr) CloseHandle(h); }
    };
    static thread_local Event ev;
    const HANDLE h = (HANDLE) _get_osfhandle(fd);
    if (ev.h == nullptr || h == INVALID_HANDLE_VALUE) return -1;
    OVERLAPPED ov{};
    ov.Offset = (DWORD) off;
    ov.OffsetHigh = (DWORD) (off >> 32);
    ov.hEvent = ev.h;
    DWORD got = 0;
    if (!ReadFile(h, dst, (DWORD) std::min<size_t>(n, 1u << 30), nullptr, &ov) && GetLastError() != ERROR_IO_PENDING)
        return -1;   // a read past the end of the file lands here too (ERROR_HANDLE_EOF): the callers want every byte
    if (!GetOverlappedResult(h, &ov, &got, TRUE)) return -1;   // waits on this thread's event
    return (int64_t) got;
}
#define STRATA_FD_OPEN_RO(path) fd_open_ro(path)
#define STRATA_FD_PREAD(fd, dst, n, off) fd_pread((fd), (dst), (n), (off))
#define STRATA_FD_CLOSE(fd) _close(fd)
#else
#define STRATA_FD_OPEN_RO(path) ::open((path).c_str(), O_RDONLY | O_CLOEXEC)
#define STRATA_FD_PREAD(fd, dst, n, off) ::pread((fd), (dst), (n), (off_t) (off))
#define STRATA_FD_CLOSE(fd) ::close(fd)
#endif
}

GgufExpertSource::~GgufExpertSource() { close(); }

void GgufExpertSource::close() {
    if (mirror_) { sycl::free(mirror_, dpct::get_in_order_queue()); mirror_ = nullptr; }
    mirror_bytes_ = 0; mirror_off_.clear(); layer_first_.clear();
    for (int fd : fds_) if (fd >= 0) STRATA_FD_CLOSE(fd);
    fds_.clear(); names_.clear(); layer_fd_.clear(); ring_.clear(); ring_key_.clear(); where_.clear();
    ring_next_ = 0;
}

bool GgufExpertSource::open(const std::string& shard1, int64_t n_layers, int64_t n_expert, std::string& err) {
    close();
    const auto& lay = strata::kernels::cpu::expert_layout();
    if (!lay.native || lay.gguf_off.size() < (size_t) (3 * n_layers)) {
        err = "--stream-experts needs a native (IQ) pack whose native_experts.txt carries the GGUF tensor offsets";
        return false;
    }
    for (int64_t l = 0; l < n_layers; ++l)
        for (int r = 0; r < 3; ++r)
            if (lay.gguf_off[(size_t) (3 * l + r)] == 0) {
                err = "--stream-experts: layer " + std::to_string(l) + " has no GGUF offset for its experts";
                return false;
            }
    shard_ = shard1;
    const size_t cut = shard1.find_last_of("/\\");
    dir_ = cut == std::string::npos ? std::string() : shard1.substr(0, cut + 1);
    n_layers_ = n_layers;
    n_expert_ = n_expert;
    layer_fd_.assign((size_t) (3 * n_layers), -1);
    for (int64_t l = 0; l < n_layers; ++l)
        for (int r = 0; r < 3; ++r)
            if (fd_of(l, r, err) < 0) { close(); return false; }
    ring_.resize(kRing);
    for (auto& b : ring_) b.resize((size_t) lay.max_blob);
    ring_key_.assign(kRing, -1);
    return true;
}

// The file of role `role` (0 gate, 1 up, 2 down) of `layer`. ExpertLayout::gguf_file is per layer AND role
// (`3 * layer + role`) since upstream 0.1.31; indexing it by layer alone read a shard-2 layer from shard 1 (Swift 1.5,
// whose layers 13-47 are in shard 2: garbage IQ1_M scales, NaN logits - 2026-10-01).
int GgufExpertSource::fd_of(int64_t layer, int role, std::string& err) {
    const size_t i = (size_t) (3 * layer + role);
    if (layer_fd_[i] >= 0) return fds_[(size_t) layer_fd_[i]];
    const auto& lay = strata::kernels::cpu::expert_layout();
    std::string name = shard_;
    if (lay.gguf_file.size() > i && !lay.gguf_file[i].empty()) name = dir_ + lay.gguf_file[i];
    for (size_t k = 0; k < names_.size(); ++k)
        if (names_[k] == name) { layer_fd_[i] = (int) k; return fds_[k]; }
    const int fd = STRATA_FD_OPEN_RO(name);
    if (fd < 0) { err = "--stream-experts: cannot open " + name; return -1; }
    names_.push_back(name);
    fds_.push_back(fd);
    layer_fd_[i] = (int) fds_.size() - 1;
    return fd;
}

const uint8_t* GgufExpertSource::blob(int64_t layer, int64_t expert) {
    if (layer < 0 || layer >= n_layers_ || expert < 0 || expert >= n_expert_ || ring_.empty()) return nullptr;
    if (mirror_ != nullptr) {
        const int64_t o = mirror_off_[(size_t) (layer * n_expert_ + expert)];
        if (o >= 0) return mirror_ + o;
    }
    const auto& lay = strata::kernels::cpu::expert_layout();
    const auto& fm = lay.fmt[(size_t) layer];
    const uint64_t blob = lay.bytes[(size_t) layer];
    // the arena loader's gather, for one expert: [gate | up | down] from the three tensors
    const uint64_t per[3] = {fm.up_off, fm.up_off, blob - fm.down_off};
    const uint64_t at[3] = {0, fm.up_off, fm.down_off};
    const int64_t key = ((int64_t) layer << 20) | expert;
    size_t slot;
    {
        std::lock_guard<std::mutex> lk(mu_);
        auto it = where_.find(key);
        if (it != where_.end()) { ++reads_; return ring_[it->second].data(); }   // still resident
        slot = ring_next_;
        ring_next_ = (ring_next_ + 1) % ring_.size();
        if (ring_key_[slot] >= 0) where_.erase(ring_key_[slot]);
        ring_key_[slot] = key;
        where_[key] = slot;
    }
    std::vector<uint8_t>& buf = ring_[slot];
    for (int r = 0; r < 3; ++r) {
        const int fd = fds_[(size_t) layer_fd_[(size_t) (3 * layer + r)]];
        const uint64_t src = lay.gguf_off[(size_t) (3 * layer + r)] + per[r] * (uint64_t) expert;
        uint64_t done = 0;
        while (done < per[r]) {
            const int64_t n = STRATA_FD_PREAD(fd, buf.data() + at[r] + done, (size_t) (per[r] - done), src + done);
            if (n <= 0) { std::lock_guard<std::mutex> lk(mu_); where_.erase(key); ring_key_[slot] = -1; return nullptr; }
            done += (uint64_t) n;
        }
    }
    ++reads_;
    return buf.data();
}

int64_t GgufExpertSource::mirror(const std::vector<std::pair<int64_t, int64_t>>& pairs, uint64_t cap, int threads,
                                 std::string& err) {
    const auto& lay = strata::kernels::cpu::expert_layout();
    std::vector<std::pair<int64_t, int64_t>> take;
    std::vector<uint64_t> offs;
    uint64_t total = 0;
    for (const auto& [l, e] : pairs) {
        if (l < 0 || l >= n_layers_ || e < 0 || e >= n_expert_) continue;
        const uint64_t b = (lay.bytes[(size_t) l] + 255) / 256 * 256;
        if (total + b > cap) break;
        take.push_back({l, e});
        offs.push_back(total);
        total += b;
    }
    if (take.empty()) return 0;
    uint8_t* base = nullptr;
    try {
        base = (uint8_t*) sycl::malloc_host(total, dpct::get_in_order_queue());
    } catch (const sycl::exception& ex) {
        err = std::string("mirror: ") + ex.what();
    }
    if (base == nullptr) {
        if (err.empty()) err = "mirror: no pinned host memory for " + std::to_string(total >> 20) + " MiB";
        return -1;
    }
    std::atomic<size_t> next{0};
    std::atomic<bool> bad{false};
    std::vector<std::thread> ts;
    for (int i = 0; i < std::max(1, threads); ++i)
        ts.emplace_back([&] {
            for (size_t j; (j = next.fetch_add(1)) < take.size() && !bad.load();) {
                const auto [l, e] = take[j];
                if (!read_into(l, e, base + offs[j], (size_t) lay.bytes[(size_t) l])) bad.store(true);
            }
        });
    for (auto& th : ts) th.join();
    if (bad.load()) {
        sycl::free(base, dpct::get_in_order_queue());
        err = "mirror: reading an expert from the GGUF failed";
        return -1;
    }
    if (mirror_) sycl::free(mirror_, dpct::get_in_order_queue());
    mirror_ = base;
    mirror_bytes_ = total;
    mirror_off_.assign((size_t) (n_layers_ * n_expert_), -1);
    layer_first_.assign((size_t) n_layers_, -1);
    for (size_t j = 0; j < take.size(); ++j) {
        const auto [l, e] = take[j];
        mirror_off_[(size_t) (l * n_expert_ + e)] = (int64_t) offs[j];
        if (layer_first_[(size_t) l] < 0) layer_first_[(size_t) l] = (int64_t) offs[j];
    }
    return (int64_t) take.size();
}

bool GgufExpertSource::pinned(int64_t layer, int64_t expert) const {
    if (mirror_ == nullptr || layer < 0 || layer >= n_layers_ || expert < 0 || expert >= n_expert_) return false;
    return mirror_off_[(size_t) (layer * n_expert_ + expert)] >= 0;
}

const uint8_t* GgufExpertSource::device_alias(int64_t layer, int64_t expert) const {
    if (mirror_ == nullptr || layer < 0 || layer >= n_layers_) return nullptr;
    if (expert >= 0 && expert < n_expert_) {
        const int64_t o = mirror_off_[(size_t) (layer * n_expert_ + expert)];
        if (o >= 0) return mirror_ + o;
    }
    const int64_t f = layer_first_[(size_t) layer];
    return f >= 0 ? mirror_ + f : nullptr;
}

bool GgufExpertSource::read_into(int64_t layer, int64_t expert, uint8_t* dst, size_t bytes) const {
    if (layer < 0 || layer >= n_layers_ || expert < 0 || expert >= n_expert_ || dst == nullptr) return false;
    const auto& lay = strata::kernels::cpu::expert_layout();
    const auto& fm = lay.fmt[(size_t) layer];
    const uint64_t blob = lay.bytes[(size_t) layer];
    if (bytes < blob) return false;
    const uint64_t per[3] = {fm.up_off, fm.up_off, blob - fm.down_off};
    const uint64_t at[3] = {0, fm.up_off, fm.down_off};
    for (int r = 0; r < 3; ++r) {
        const int fd = fds_[(size_t) layer_fd_[(size_t) (3 * layer + r)]];
        const uint64_t src = lay.gguf_off[(size_t) (3 * layer + r)] + per[r] * (uint64_t) expert;
        uint64_t done = 0;
        while (done < per[r]) {
            const int64_t n = STRATA_FD_PREAD(fd, dst + at[r] + done, (size_t) (per[r] - done), src + done);
            if (n <= 0) return false;
            done += (uint64_t) n;
        }
    }
    return true;
}

}  // namespace strata::core
