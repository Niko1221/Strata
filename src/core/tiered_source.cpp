// src/core/tiered_source.cpp - the VRAM / pinned RAM / SSD expert tiers.  See `TieredExpertSource` in the header.
#include "strata/core/expert_source.hpp"
#include "strata/kernels/cpu/expert_layout.hpp"
#include "strata/core/pinned.hpp"

#include <cuda_runtime.h>

#include <algorithm>
#include <atomic>
#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <deque>
#include <condition_variable>
#include <mutex>
#include <string>
#include <thread>
#include <vector>

#if !defined(_WIN32)
#include <fcntl.h>
#include <sys/mman.h>
#include <sys/stat.h>
#include <unistd.h>
#endif

#if defined(_WIN32)
// The tiers use mmap / madvise / O_DIRECT; on Windows the class exists so the engine links, and refuses to open.
namespace strata::core {
TieredExpertSource::~TieredExpertSource() = default;
bool TieredExpertSource::open(const std::string&, int64_t, int64_t, std::string& err) {
    err = "--tiered-experts is not available on Windows";
    return false;
}
bool TieredExpertSource::settle(const ExpertCache*, const std::vector<std::pair<int32_t, int32_t>>&, int64_t, int64_t,
                                int, std::string& err) {
    err = "--tiered-experts is not available on Windows";
    return false;
}
void TieredExpertSource::close() {}
const uint8_t* TieredExpertSource::blob(int64_t, int64_t) { return nullptr; }
bool TieredExpertSource::pinned(int64_t, int64_t) const { return false; }
const uint8_t* TieredExpertSource::device_alias(int64_t, int64_t) const { return nullptr; }
void TieredExpertSource::begin_layer(int64_t, const int32_t*, int64_t) {}
void TieredExpertSource::read_into(const uint8_t* src, uint8_t* dst, size_t n) const { std::memcpy(dst, src, n); }
}  // namespace strata::core
#else
namespace strata::core {

// expert_source.cpp: the arena's loader from shard 1 (not in a header; the arena was its only caller)
LoadStats load_experts_gguf(const std::string& gguf, uint8_t* dst, const strata::kernels::cpu::ExpertLayout& lay,
                            int threads);

namespace {

// Converting a native pack: the experts in the layout's blob order, written once to experts.bin.  The arena's own
// loader does the per-expert gather (gate | up | down per blob), so the bytes are the ones the arena would hold.
bool write_experts_bin(const std::string& gguf, const std::string& path, std::string& err) {
    const strata::kernels::cpu::ExpertLayout& lay = strata::kernels::cpu::expert_layout();
    const std::string part = path + ".part";
    const int fd = ::open(part.c_str(), O_RDWR | O_CREAT | O_TRUNC, 0644);
    if (fd < 0) { err = "TieredExpertSource: cannot create " + part; return false; }
    if (ftruncate(fd, (off_t) lay.total) != 0) { ::close(fd); err = "TieredExpertSource: cannot size " + part; return false; }
    void* dst = mmap(nullptr, (size_t) lay.total, PROT_READ | PROT_WRITE, MAP_SHARED, fd, 0);
    if (dst == MAP_FAILED) { ::close(fd); err = "TieredExpertSource: cannot map " + part; return false; }
    std::fprintf(stderr, "strata generate: writing %s from %s (%.2f GiB, one time) ...\n", path.c_str(), gguf.c_str(),
                 (double) lay.total / 1073741824.0);
    const LoadStats st = load_experts_gguf(gguf, (uint8_t*) dst, lay, /*threads=*/6);
    const bool ok = st.seconds >= 0 && msync(dst, (size_t) lay.total, MS_SYNC) == 0;
    munmap(dst, (size_t) lay.total);
    ::close(fd);
    if (!ok || std::rename(part.c_str(), path.c_str()) != 0) {
        err = "TieredExpertSource: writing " + path + " failed";
        return false;
    }
    std::fprintf(stderr, "strata generate: wrote %s in %.0f s\n", path.c_str(), st.seconds);
    return true;
}

uint64_t page_size() {
    static const uint64_t p = (uint64_t) sysconf(_SC_PAGESIZE);
    return p;
}

// MemAvailable, in bytes: the kernel's own estimate of what can be allocated without swapping, which counts the
// reclaimable page cache - including the resident experts' pages this source has just dropped.
int64_t mem_available() {
    std::ifstream f("/proc/meminfo");
    std::string key;
    int64_t kb = 0;
    std::string unit;
    while (f >> key >> kb >> unit)
        if (key == "MemAvailable:") return kb * 1024;
    return -1;
}

}  // namespace

TieredExpertSource::~TieredExpertSource() { close(); }

bool TieredExpertSource::open(const std::string& pack_dir, int64_t n_layers, int64_t n_expert, std::string& err) {
    close();
    const strata::kernels::cpu::ExpertLayout& lay = strata::kernels::cpu::expert_layout();
    if (lay.n_layers != n_layers || lay.n_expert != n_expert) {
        err = "TieredExpertSource: the expert layout was loaded for a different geometry";
        return false;
    }
    const std::string path = pack_dir + "/experts.bin";
    if (lay.native && !gguf_.empty() && !std::ifstream(path, std::ios::binary) && !write_experts_bin(gguf_, path, err))
        return false;
    const int fd = ::open(path.c_str(), O_RDONLY);
    if (fd < 0) {
        err = "TieredExpertSource: cannot open " + path + " (the tiered source needs a pack with experts.bin)";
        return false;
    }
    struct stat st{};
    if (fstat(fd, &st) != 0 || (uint64_t) st.st_size != lay.total) {
        char buf[400];
        std::snprintf(buf, sizeof buf, "TieredExpertSource: %s is %llu B but the layout makes %llu B", path.c_str(),
                      (unsigned long long) st.st_size, (unsigned long long) lay.total);
        ::close(fd);
        err = buf;
        return false;
    }
    // The arena is one largest blob longer than the file (a copy of a whole VRAM slot may start at any expert);
    // past the file's last page a file mapping raises SIGBUS, so the file is mapped over an anonymous reservation
    // and the tail stays anonymous zeros.
    const uint64_t pg = page_size();
    file_bytes_ = lay.total;
    map_bytes_ = (lay.total + lay.max_blob + pg - 1) / pg * pg;
    void* res = mmap(nullptr, (size_t) map_bytes_, PROT_READ, MAP_PRIVATE | MAP_ANONYMOUS | MAP_NORESERVE, -1, 0);
    if (res == MAP_FAILED) {
        ::close(fd);
        err = "TieredExpertSource: the address reservation failed";
        return false;
    }
    void* view = mmap(res, (size_t) file_bytes_, PROT_READ, MAP_SHARED | MAP_FIXED, fd, 0);
    if (view == MAP_FAILED) {
        munmap(res, (size_t) map_bytes_);
        ::close(fd);
        err = "TieredExpertSource: mmap failed on " + path;
        return false;
    }
    fd_ = fd;
    // a second descriptor for streaming reads that should not go through the page cache (read_into)
    dfd_ = std::getenv("STRATA_NO_DIRECT_STREAM") ? -1 : ::open(path.c_str(), O_RDONLY | O_DIRECT);
    base_ = (const uint8_t*) res;
    n_layers_ = n_layers;
    n_expert_ = n_expert;
    // Until `settle` runs every expert is cold: the cache fill reads through the mapping like any other reader.
    tier_.assign((size_t) (n_layers * n_expert), (uint8_t) kCold);
    note_ = "mapped " + path;
    pf_stop_ = false;
    for (int i = 0; i < 4; ++i)
        pf_threads_.emplace_back([this] {
            for (;;) {
                std::pair<uint64_t, uint64_t> r;
                {
                    std::unique_lock<std::mutex> lk(pf_mu_);
                    pf_cv_.wait(lk, [&] { return pf_stop_ || !pf_q_.empty(); });
                    if (pf_stop_) return;
                    r = pf_q_.front();
                    pf_q_.pop_front();
                }
                madvise((void*) (base_ + r.first), (size_t) r.second, MADV_WILLNEED);
            }
        });
    return true;
}

bool TieredExpertSource::settle(const ExpertCache* cache, const std::vector<std::pair<int32_t, int32_t>>& profile,
                                int64_t budget_bytes, int64_t reserve_bytes, int threads, std::string& err) {
    if (base_ == nullptr) { err = "TieredExpertSource::settle: not open"; return false; }
    const strata::kernels::cpu::ExpertLayout& lay = strata::kernels::cpu::expert_layout();
    const uint64_t pg = page_size();
    const auto t0 = std::chrono::steady_clock::now();

    // ---- 1. the VRAM tier: drop its pages.  Only whole pages inside the blob, so a neighbour's bytes that share
    // an edge page stay mapped (dropping a clean page is harmless anyway - it would only be read again).
    uint64_t vram_bytes = 0;
    int64_t n_vram = 0;
    for (int64_t l = 0; l < n_layers_; ++l)
        for (int64_t e = 0; e < n_expert_; ++e) {
            if (cache == nullptr || cache->slot_of(l, e) == kNotResident) continue;
            tier_[(size_t) (l * n_expert_ + e)] = kVram;
            const uint64_t off = lay.blob_offset(l, e), n = lay.blob_bytes(l);
            const uint64_t a = (off + pg - 1) / pg * pg, b = (off + n) / pg * pg;
            if (b > a) {
                madvise((void*) (base_ + a), (size_t) (b - a), MADV_DONTNEED);
                posix_fadvise(fd_, (off_t) a, (off_t) (b - a), POSIX_FADV_DONTNEED);
            }
            vram_bytes += n;
            ++n_vram;
        }

    // ---- 2. the budget, read AFTER the drop so the cache fill's pages count as free
    if (budget_bytes < 0) {
        const int64_t avail = mem_available();
        budget_bytes = avail > reserve_bytes ? avail - reserve_bytes : 0;
    }

    // ---- 3. the PINNED tier: the profile's order past the cache, while the budget lasts.  A pair missing from
    // the profile is never routed in the profiling traces, so it is the right one to leave cold.
    uint64_t pinned_bytes = 0;
    int64_t n_pinned = 0;
    for (const auto& pr : profile) {
        const int64_t l = pr.first, e = pr.second;
        if (l < 0 || l >= n_layers_ || e < 0 || e >= n_expert_) continue;
        uint8_t& t = tier_[(size_t) (l * n_expert_ + e)];
        if (t != kCold) continue;
        const uint64_t n = lay.blob_bytes(l);
        if (pinned_bytes + n > (uint64_t) budget_bytes) break;
        t = kPinned;
        pinned_bytes += n;
        ++n_pinned;
    }

    // ---- 4. merge the pinned blobs into runs in FILE order, across layer boundaries, and widen each to whole
    // pages.  Two registrations must not share a page (`cudaHostRegister` refuses an overlap), and runs split by
    // at least one other blob (>= 1 MB) never do; runs that would touch are merged by construction.
    struct Run { uint64_t a, b; };
    std::vector<Run> runs;
    for (int64_t l = 0; l < n_layers_; ++l)
        for (int64_t e = 0; e < n_expert_; ++e) {
            if (tier_[(size_t) (l * n_expert_ + e)] != kPinned) continue;
            const uint64_t off = lay.blob_offset(l, e), end = off + lay.blob_bytes(l);
            const uint64_t a = off / pg * pg, b = (end + pg - 1) / pg * pg;
            if (!runs.empty() && a <= runs.back().b) runs.back().b = std::max(runs.back().b, b);
            else runs.push_back({a, b});
        }

    // ---- 5. turn each run into anonymous memory IN PLACE, fill it from the file, then lock it.
    //
    // **THE DRIVER DOES NOT PIN FILE-BACKED PAGES.**  Measured on this machine (driver 610, kernel 7.2):
    // `cudaHostRegister` on the `MAP_SHARED` file mapping returns "operation not supported" for every run, with
    // or without `cudaHostRegisterReadOnly` - the kernel refuses long-term pins of page-cache pages.  So the run's
    // slice of the mapping is replaced (`MAP_FIXED`) by anonymous memory at the SAME address and filled with
    // `pread`: `blob_offset` still lands on the same bytes, and anonymous memory is what the arena always pinned.
    // Edge pages shared with a neighbouring blob are filled from the file too, so the neighbour's bytes stay right.
    std::vector<uint8_t> anon_ok(runs.size(), 0);
    {
        std::atomic<size_t> next{0};
        auto worker = [&]() {
            for (;;) {
                const size_t i = next.fetch_add(1);
                if (i >= runs.size()) return;
                uint8_t* p = (uint8_t*) base_ + runs[i].a;
                // the run's last page may extend past the file: only the file's bytes are read, the rest is zero
                const uint64_t end = std::min(runs[i].b, file_bytes_);
                void* m = mmap(p, (size_t) (runs[i].b - runs[i].a), PROT_READ | PROT_WRITE,
                               MAP_PRIVATE | MAP_ANONYMOUS | MAP_FIXED, -1, 0);
                if (m == MAP_FAILED) continue;
                bool good = true;
                for (uint64_t off = runs[i].a; off < end;) {
                    const ssize_t got = pread(fd_, p + (off - runs[i].a), (size_t) std::min<uint64_t>(end - off, 64u << 20),
                                              (off_t) off);
                    if (got <= 0) { good = false; break; }
                    off += (uint64_t) got;
                }
                if (good) anon_ok[i] = 1;
            }
        };
        std::vector<std::thread> pool;
        for (int i = 1; i < std::max(threads, 1); ++i) pool.emplace_back(worker);
        worker();
        for (auto& t : pool) t.join();
    }
    uint64_t locked = 0;
    int64_t failed_runs = 0;
    std::string first_fail;
    for (size_t i = 0; i < runs.size(); ++i) {
        const Run& r = runs[i];
        void* p = (void*) (base_ + r.a);
        const cudaError_t e = anon_ok[i] ? cudaHostRegister(p, (size_t) (r.b - r.a),
                                                            cudaHostRegisterPortable | cudaHostRegisterMapped)
                                         : cudaErrorInvalidValue;
        if (e != cudaSuccess) {
            // Consume the error (see pinned.cu: a sticky error lies about the next launch), and put the file
            // mapping back: an unpinned anonymous copy would only be swapped (zram) instead of dropped.
            (void) cudaGetLastError();
            if (first_fail.empty()) first_fail = anon_ok[i] ? cudaGetErrorString(e) : "the anonymous fill failed";
            mmap(p, (size_t) (std::min(r.b, file_bytes_) - r.a), PROT_READ, MAP_SHARED | MAP_FIXED, fd_, (off_t) r.a);
            ++failed_runs;
            continue;
        }
        mprotect(p, (size_t) (r.b - r.a), PROT_READ);   // the kernels only read; a stray write should fault
        regs_.push_back({(uint8_t*) p, r.b - r.a});
        locked += r.b - r.a;
    }
    // the fill's reads went through the page cache as well: those copies are now duplicates of the anonymous ones
    for (const Run& r : runs) posix_fadvise(fd_, (off_t) r.a, (off_t) (std::min(r.b, file_bytes_) - r.a), POSIX_FADV_DONTNEED);
    if (failed_runs > 0) {
        // demote every pinned blob that is not inside a registered run
        for (int64_t l = 0; l < n_layers_; ++l)
            for (int64_t e = 0; e < n_expert_; ++e) {
                uint8_t& t = tier_[(size_t) (l * n_expert_ + e)];
                if (t != kPinned) continue;
                if (!pinned(l, e)) {
                    t = kCold;
                    pinned_bytes -= lay.blob_bytes(l);
                    --n_pinned;
                }
            }
    }

    // `pinned()` above consults `regs_` only while settling; from here on the tier table is the answer.
    uint64_t cold_bytes = 0;
    int64_t n_cold = 0;
    for (int64_t l = 0; l < n_layers_; ++l)
        for (int64_t e = 0; e < n_expert_; ++e)
            if (tier_[(size_t) (l * n_expert_ + e)] == kCold) {
                cold_bytes += lay.blob_bytes(l);
                ++n_cold;
            }

    const double secs = std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count();
    auto gib = [](uint64_t b) { return (double) b / 1073741824.0; };
    char buf[600];
    std::snprintf(buf, sizeof buf,
                  "tiers: VRAM %lld experts (%.2f GiB, dropped from RAM) | PINNED %lld (%.2f GiB, %zu runs, "
                  "%.2f GiB locked) | COLD %lld (%.2f GiB, paged from SSD) | budget %.2f GiB | %.1f s%s%s",
                  (long long) n_vram, gib(vram_bytes), (long long) n_pinned, gib(pinned_bytes), regs_.size(),
                  gib(locked), (long long) n_cold, gib(cold_bytes), gib((uint64_t) budget_bytes), secs,
                  failed_runs ? "; cudaHostRegister refused runs: " : "", failed_runs ? first_fail.c_str() : "");
    note_ = buf;
    return true;
}

void TieredExpertSource::close() {
    {
        std::lock_guard<std::mutex> lk(pf_mu_);
        pf_stop_ = true;
        pf_q_.clear();
    }
    pf_cv_.notify_all();
    for (auto& t : pf_threads_) t.join();
    pf_threads_.clear();
    for (const auto& r : regs_) cudaHostUnregister(r.first);
    regs_.clear();
    if (base_ != nullptr) munmap((void*) base_, (size_t) map_bytes_);
    if (fd_ >= 0) ::close(fd_);
    if (dfd_ >= 0) ::close(dfd_);
    base_ = nullptr;
    fd_ = -1;
    dfd_ = -1;
    tier_.clear();
}

const uint8_t* TieredExpertSource::blob(int64_t layer, int64_t expert) {
    if (base_ == nullptr || layer < 0 || layer >= n_layers_ || expert < 0 || expert >= n_expert_) return nullptr;
    ++reads_;
    return base_ + strata::kernels::cpu::expert_layout().blob_offset(layer, expert);
}

bool TieredExpertSource::pinned(int64_t layer, int64_t expert) const {
    if (base_ == nullptr || layer < 0 || layer >= n_layers_ || expert < 0 || expert >= n_expert_) return false;
    if (tier_[(size_t) (layer * n_expert_ + expert)] != kPinned) return false;
    // A pinned blob lies inside one registered run; checked against the runs so a refused registration can never
    // hand the GPU an unlocked address.  The runs are in address order: binary search.
    const auto& lay = strata::kernels::cpu::expert_layout();
    const uint8_t* a = base_ + lay.blob_offset(layer, expert);
    const uint8_t* b = a + lay.blob_bytes(layer);
    auto it = std::upper_bound(regs_.begin(), regs_.end(), a,
                               [](const uint8_t* p, const std::pair<uint8_t*, uint64_t>& r) { return p < r.first; });
    if (it == regs_.begin()) return false;
    --it;
    return a >= it->first && b <= it->first + it->second;
}

const uint8_t* TieredExpertSource::device_alias(int64_t layer, int64_t expert) const {
    if (!pinned(layer, expert)) return nullptr;
    // Unified addressing (every 64-bit Linux CUDA context): a mapped registration's device address is its host
    // address.  Asked of the driver rather than assumed, once per call - this path runs a few times per layer.
    const uint8_t* h = base_ + strata::kernels::cpu::expert_layout().blob_offset(layer, expert);
    void* d = nullptr;
    if (cudaHostGetDevicePointer(&d, (void*) h, 0) != cudaSuccess) {
        (void) cudaGetLastError();
        return nullptr;
    }
    return (const uint8_t*) d;
}

void TieredExpertSource::read_into(const uint8_t* src, uint8_t* dst, size_t n) const {
    // Inside the file-backed part and not in a locked run: one pread (a single large request to the NVMe) instead
    // of faulting ~400 pages in through the mapping with 128 KB of readahead.  Pinned or anonymous bytes are in
    // RAM already: memcpy.
    if (base_ != nullptr && src >= base_ && src + n <= base_ + file_bytes_) {
        const uint8_t* p = src;
        auto it = std::upper_bound(regs_.begin(), regs_.end(), p,
                                   [](const uint8_t* q, const std::pair<uint8_t*, uint64_t>& r) { return q < r.first; });
        const bool locked = it != regs_.begin() && p < (std::prev(it))->first + (std::prev(it))->second;
        if (!locked) {
            // O_DIRECT: a streamed expert is read once per prompt chunk, and through the page cache ~20 GB per long
            // prompt would evict the cold-tier pages decode relies on.  Direct I/O wants the offset, length and
            // buffer page-aligned, so the aligned superset lands in a per-thread buffer and the blob is copied out.
            if (dfd_ >= 0) {
                const uint64_t off = (uint64_t) (src - base_), pg = 4096;
                const uint64_t a = off / pg * pg, b = std::min<uint64_t>((off + n + pg - 1) / pg * pg,
                                                                          (file_bytes_ + pg - 1) / pg * pg);
                thread_local uint8_t* bounce = nullptr;
                thread_local size_t cap = 0;
                if (cap < b - a) {
                    std::free(bounce);
                    bounce = (uint8_t*) std::aligned_alloc(pg, (size_t) (b - a));
                    cap = bounce ? (size_t) (b - a) : 0;
                }
                size_t done = 0;
                while (bounce != nullptr && done < b - a) {
                    const ssize_t got = pread(dfd_, bounce + done, (size_t) (b - a) - done, (off_t) (a + done));
                    if (got <= 0) break;
                    done += (size_t) got;
                }
                if (bounce != nullptr && done >= off - a + n) {
                    std::memcpy(dst, bounce + (off - a), n);
                    return;
                }
            }
            size_t done = 0;
            while (done < n) {
                const ssize_t got = pread(fd_, dst + done, n - done, (off_t) (src - base_) + (off_t) done);
                if (got <= 0) break;
                done += (size_t) got;
            }
            if (done == n) return;
        }
    }
    std::memcpy(dst, src, n);
}

void TieredExpertSource::begin_layer(int64_t layer, const int32_t* ids, int64_t k) {
    if (base_ == nullptr || ids == nullptr || layer < 0 || layer >= n_layers_) return;
    static const bool off = std::getenv("STRATA_NO_COLD_PREFETCH") != nullptr;   // the A/B arm
    if (off) return;
    const auto& lay = strata::kernels::cpu::expert_layout();
    const uint64_t pg = page_size();
    for (int64_t i = 0; i < k; ++i) {
        const int64_t e = ids[i];
        if (e < 0 || e >= n_expert_) continue;
        const size_t idx = (size_t) (layer * n_expert_ + e);
        const uint8_t t = tier_[idx];
        if (t == kPinned) continue;
        // a VRAM-tier expert has no RAM copy; it needs one only once the adaptive swap has evicted it
        if (t == kVram && (res_ == nullptr || res_[idx] >= 0)) continue;
        // One read for the whole blob instead of ~340 page faults in the pool.  Asynchronous: it overlaps the
        // pool's work on the experts already in RAM.
        const uint64_t off = lay.blob_offset(layer, e);
        const uint64_t a = off / pg * pg, b = (off + lay.blob_bytes(layer) + pg - 1) / pg * pg;
        ++cold_prefetches_;
        // STRATA_SYNC_PREFETCH: the old arm, madvise on the pool's own thread (it can block while the device queue
        // is full, and the pool's plan phase waited for it).  Otherwise a helper thread takes the range.
        static const bool sync_pf = std::getenv("STRATA_SYNC_PREFETCH") != nullptr;
        if (sync_pf) {
            madvise((void*) (base_ + a), (size_t) (b - a), MADV_WILLNEED);
            continue;
        }
        {
            std::lock_guard<std::mutex> lk(pf_mu_);
            pf_q_.push_back({a, b - a});
        }
        pf_cv_.notify_one();
    }
}

}  // namespace strata::core
#endif  // !_WIN32
