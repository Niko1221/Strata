// src/core/expert_source.cpp - the adapter.  See the header for the three clauses of the contract.
#include "strata/core/expert_source.hpp"
#include "strata/core/remote_experts.hpp"
#include "strata/kernels/cpu/expert_layout.hpp"

#include "strata/core/pinned.hpp"
#include "strata/kernels/elementwise.hpp"
#include "strata/kernels/quantize_act.hpp"
#include "strata/kernels/s2_expert_grouped.hpp"

#include <cuda_runtime.h>

#include <algorithm>
#include <atomic>
#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <filesystem>
#include <sstream>
#include <limits>
#include <mutex>
#include <thread>
#include <utility>
#include <vector>

#if defined(_WIN32)
#define WIN32_LEAN_AND_MEAN
#ifndef NOMINMAX
#define NOMINMAX   // std::numeric_limits<T>::max() below
#endif
#include <windows.h>
#else
#include <fcntl.h>
#include <sys/mman.h>
#include <sys/stat.h>
#include <unistd.h>
#endif

namespace strata::core {

namespace detail {

bool cgroup_available_bytes(uint64_t limit, const CgroupMemoryStat& stat, uint64_t& bytes) {
    bytes = 0;
    if (!stat.valid) return false;

    // memory.stat's inactive_file can race memory.current, so bound it to charged usage first.
    uint64_t reclaimable = std::min(stat.inactive_file, stat.current);
    reclaimable = stat.file_dirty >= reclaimable ? 0 : reclaimable - stat.file_dirty;
    reclaimable = stat.file_writeback >= reclaimable ? 0 : reclaimable - stat.file_writeback;

    // Reclaiming clean file pages reduces usage; saturating subtraction also handles a transient over-limit read.
    const uint64_t usage_after_reclaim = stat.current - reclaimable;
    bytes = usage_after_reclaim < limit ? limit - usage_after_reclaim : 0;
    return true;
}

bool make_cache_complement_plan(
    int64_t n_layers, int64_t n_expert, const std::vector<uint64_t>& layer_blob_bytes,
    const std::vector<std::pair<int32_t, int32_t>>& primary_gpu_pairs,
    const std::vector<std::pair<int32_t, int32_t>>& additional_gpu_pairs,
    std::vector<uint64_t>& offsets, uint64_t& bytes, std::string& err) {
    offsets.clear();
    bytes = 0;
    err.clear();
    if (n_layers <= 0 || n_expert <= 0 || layer_blob_bytes.size() != (size_t) n_layers) {
        err = "FileExpertSource: invalid geometry for the cache complement plan";
        return false;
    }
    if ((uint64_t) n_layers > (uint64_t) std::numeric_limits<size_t>::max() / (uint64_t) n_expert) {
        err = "FileExpertSource: cache complement index table is too large";
        return false;
    }
    const size_t count = (size_t) n_layers * (size_t) n_expert;
    std::vector<uint8_t> omitted(count, 0);
    auto mark_pairs = [&](const std::vector<std::pair<int32_t, int32_t>>& pairs, uint8_t bit,
                          const char* label) -> bool {
        for (const auto& pair : pairs) {
            if (pair.first < 0 || pair.second < 0 || pair.first >= n_layers || pair.second >= n_expert) {
                err = std::string("FileExpertSource: ") + label + " pair is outside the expert geometry";
                return false;
            }
            const size_t index = (size_t) pair.first * (size_t) n_expert + (size_t) pair.second;
            if ((omitted[index] & bit) != 0) {
                err = std::string("FileExpertSource: duplicate ") + label + " pair in the cache complement plan";
                return false;
            }
            if (bit == 2 && (omitted[index] & 1) != 0) {
                err = "FileExpertSource: the primary and additional GPU expert tiers overlap";
                return false;
            }
            omitted[index] |= bit;
        }
        return true;
    };
    if (!mark_pairs(primary_gpu_pairs, 1, "primary GPU") ||
        !mark_pairs(additional_gpu_pairs, 2, "additional GPU")) return false;
    for (uint64_t blob_bytes : layer_blob_bytes) {
        if (blob_bytes == 0) {
            err = "FileExpertSource: cache complement layer has zero-sized expert blobs";
            return false;
        }
    }

    offsets.assign(count, kNoCacheComplement);
    for (int64_t layer = 0; layer < n_layers; ++layer) {
        const uint64_t blob_bytes = layer_blob_bytes[(size_t) layer];
        for (int64_t expert = 0; expert < n_expert; ++expert) {
            const size_t index = (size_t) layer * (size_t) n_expert + (size_t) expert;
            if (omitted[index] != 0) continue;
            if (bytes > std::numeric_limits<uint64_t>::max() - blob_bytes) {
                offsets.clear();
                bytes = 0;
                err = "FileExpertSource: cache complement size overflows";
                return false;
            }
            offsets[index] = bytes;
            bytes += blob_bytes;
        }
    }
    if (bytes > (uint64_t) std::numeric_limits<size_t>::max()) {
        offsets.clear();
        bytes = 0;
        err = "FileExpertSource: cache complement exceeds the host address space";
        return false;
    }
    return true;
}

const uint8_t* cache_complement_blob_or_fallback(
    size_t index, const std::vector<uint64_t>& offsets, const uint8_t* complement_host,
    const uint8_t* mapped_fallback) {
    if (complement_host != nullptr && index < offsets.size() && offsets[index] != kNoCacheComplement)
        return complement_host + (size_t) offsets[index];
    return mapped_fallback;
}

}  // namespace detail

namespace {
constexpr uint64_t kPinnedMemoryHeadroom = 8ull << 30;

#if defined(__linux__)
bool read_cgroup_memory_stat(const std::filesystem::path& path, uint64_t current,
                             detail::CgroupMemoryStat& stat) {
    std::ifstream input(path / "memory.stat");
    if (!input) return false;

    bool inactive_file = false, file_dirty = false, file_writeback = false;
    std::string line;
    while (std::getline(input, line)) {
        std::istringstream fields(line);
        std::string key;
        uint64_t value = 0;
        if (!(fields >> key >> value)) return false;
        fields >> std::ws;
        if (!fields.eof()) return false;

        if (key == "inactive_file") {
            if (inactive_file) return false;
            inactive_file = true;
            stat.inactive_file = value;
        } else if (key == "file_dirty") {
            if (file_dirty) return false;
            file_dirty = true;
            stat.file_dirty = value;
        } else if (key == "file_writeback") {
            if (file_writeback) return false;
            file_writeback = true;
            stat.file_writeback = value;
        }
    }
    if (!input.eof() || !inactive_file || !file_dirty || !file_writeback) return false;
    stat.current = current;
    stat.valid = true;
    return true;
}
#endif

bool available_memory_bytes(uint64_t& bytes) {
#if defined(_WIN32)
    MEMORYSTATUSEX status{};
    status.dwLength = sizeof(status);
    if (!GlobalMemoryStatusEx(&status)) return false;
    bytes = (uint64_t) status.ullAvailPhys;
    return bytes > 0;
#elif defined(__linux__)
    // MemAvailable includes reclaimable page cache, unlike _SC_AVPHYS_PAGES.
    std::ifstream info("/proc/meminfo");
    std::string line;
    bytes = 0;
    while (std::getline(info, line)) {
        std::istringstream fields(line);
        std::string key, unit;
        uint64_t value = 0;
        if (fields >> key >> value >> unit && key == "MemAvailable:" && unit == "kB" &&
            value <= std::numeric_limits<uint64_t>::max() / 1024) bytes = value * 1024;
    }
    if (bytes == 0) return false;
    // Account for the tightest cgroup-v2 ancestor limit when its normal mount is visible.
    // This is a point-in-time guard, not a reservation against concurrent allocations.
    std::ifstream groups("/proc/self/cgroup");
    if (!groups) return false;
    bool resolved_v2 = false;
    while (std::getline(groups, line)) {
        if (line.rfind("0::/", 0) != 0) continue;
        const std::filesystem::path root("/sys/fs/cgroup");
        auto path = (root / line.substr(4)).lexically_normal();
        if (path.string().rfind(root.string(), 0) != 0 || !std::filesystem::is_directory(path)) return false;
        resolved_v2 = true;
        while (path.string().rfind(root.string(), 0) == 0) {
            std::ifstream limit_file(path / "memory.max"), current_file(path / "memory.current");
            std::string limit;
            uint64_t current = 0;
            const bool readable = bool(limit_file >> limit) && bool(current_file >> current);
            // The host's root cgroup has no memory.max; ordinary child groups must expose their limits.
            if (!readable && !(path == root && !std::filesystem::exists(path / "memory.max") &&
                               std::filesystem::exists(path / "cgroup.controllers"))) return false;
            if (readable && limit != "max") {
                try {
                    size_t consumed = 0;
                    const uint64_t cap = std::stoull(limit, &consumed);
                    if (consumed != limit.size()) return false;
                    detail::CgroupMemoryStat stat;
                    if (!read_cgroup_memory_stat(path, current, stat)) return false;
                    uint64_t cgroup_available = 0;
                    if (!detail::cgroup_available_bytes(cap, stat, cgroup_available)) return false;
                    bytes = std::min(bytes, cgroup_available);
                } catch (...) { return false; }
            }
            if (path == root) break;
            path = path.parent_path();
        }
    }
    return resolved_v2;
#else
    const long pages = sysconf(_SC_AVPHYS_PAGES);
    const long page_bytes = sysconf(_SC_PAGESIZE);
    if (pages <= 0 || page_bytes <= 0 ||
        (uint64_t) pages > std::numeric_limits<uint64_t>::max() / (uint64_t) page_bytes) return false;
    bytes = (uint64_t) pages * (uint64_t) page_bytes;
    return bytes > 0;
#endif
}

}  // namespace

// ================================ THE FILE-BACKED SOURCE ================================

FileExpertSource::~FileExpertSource() { close(); }

bool FileExpertSource::open(const std::string& pack_dir, int64_t n_layers, int64_t n_expert, std::string& err) {
    close();
    if (n_layers <= 0 || n_expert <= 0) { err = "FileExpertSource: the geometry is empty"; return false; }
    const auto& layout = strata::kernels::cpu::expert_layout();
    if (layout.n_layers != n_layers || layout.n_expert != n_expert) {
        err = "FileExpertSource: the requested geometry does not match the loaded expert layout";
        return false;
    }
    if ((uint64_t) n_layers > (uint64_t) std::numeric_limits<int64_t>::max() / (uint64_t) n_expert) {
        err = "FileExpertSource: the expert count overflows";
        return false;
    }
    const uint64_t blob_count = (uint64_t) n_layers * (uint64_t) n_expert;
    if (blob_count > (uint64_t) std::numeric_limits<int64_t>::max() ||
        (uint64_t) n_layers > (uint64_t) std::numeric_limits<size_t>::max()) {
        err = "FileExpertSource: the expert count overflows";
        return false;
    }

    std::vector<uint64_t> layer_offsets((size_t) n_layers), layer_blob_bytes((size_t) n_layers);
    const uint64_t want = layout.total;
    if (want == 0 || want > (uint64_t) std::numeric_limits<size_t>::max()) {
        err = "FileExpertSource: the loaded expert layout has an invalid size";
        return false;
    }
    if (!layout.native) {
        if (blob_count > std::numeric_limits<uint64_t>::max() / (uint64_t) strata::kernels::cpu::BLOB) {
            err = "FileExpertSource: the canonical expert size overflows";
            return false;
        }
        const uint64_t canonical_size = blob_count * (uint64_t) strata::kernels::cpu::BLOB;
        if (want != canonical_size) {
            err = "FileExpertSource: the canonical expert layout has an inconsistent size";
            return false;
        }
        const uint64_t bytes = (uint64_t) strata::kernels::cpu::BLOB;
        const uint64_t layer_bytes = (uint64_t) n_expert * bytes;
        for (int64_t layer = 0; layer < n_layers; ++layer) {
            layer_offsets[(size_t) layer] = (uint64_t) layer * layer_bytes;
            layer_blob_bytes[(size_t) layer] = bytes;
        }
    } else {
        if (layout.offset.size() != (size_t) n_layers || layout.bytes.size() != (size_t) n_layers ||
            layout.fmt.size() != (size_t) n_layers) {
            err = "FileExpertSource: the native expert layout is incomplete";
            return false;
        }
        uint64_t at = 0;
        for (int64_t layer = 0; layer < n_layers; ++layer) {
            const size_t i = (size_t) layer;
            const uint64_t bytes = (uint64_t) layout.fmt[i].bytes;
            if (layout.offset[i] != at || bytes == 0 || layout.bytes[i] != bytes ||
                bytes > std::numeric_limits<uint64_t>::max() / (uint64_t) n_expert) {
                err = "FileExpertSource: the native expert layout is invalid at layer " + std::to_string(layer);
                return false;
            }
            const uint64_t layer_bytes = bytes * (uint64_t) n_expert;
            if (at > want || layer_bytes > want - at) {
                err = "FileExpertSource: the native expert layout exceeds its declared size at layer " +
                      std::to_string(layer);
                return false;
            }
            layer_offsets[i] = layout.offset[i];
            layer_blob_bytes[i] = bytes;
            at += layer_bytes;
        }
        if (at != want) {
            err = "FileExpertSource: the native expert layout has an inconsistent size";
            return false;
        }
    }
    const std::string path = pack_dir + "/experts.bin";

#if defined(_WIN32)
    // UTF-8 -> UTF-16: the pack may live under a path with non-ASCII characters, and `CreateFileA` would
    // silently mangle it into a file-not-found.
    const int wide = MultiByteToWideChar(CP_UTF8, 0, path.c_str(), -1, nullptr, 0);
    std::vector<wchar_t> wpath((size_t) (wide > 0 ? wide : 1));
    if (wide > 0) MultiByteToWideChar(CP_UTF8, 0, path.c_str(), -1, wpath.data(), wide);
    // **`FILE_FLAG_RANDOM_ACCESS` WAS HERE AND IT COST 14x.**
    //
    // The design depends on the OS page cache holding the whole 34 GB expert set, because this machine has
    // 64 GB of DDR5 and `L9` measured the CPU path at 44.14 GB/s from DRAM.  `FILE_FLAG_RANDOM_ACCESS` tells
    // the cache manager the opposite: it disables read-ahead AND it lets the manager drop the pages again
    // quickly, on the assumption that a large randomly-accessed file will not be re-read.  Measured, on
    // `strata generate --max-new 24`: **1.93 GB/s** - disk speed, 344 ms/token, and it never warmed up over 25
    // tokens, because the pages were being evicted as fast as they were faulted in.
    //
    // The correct flag is NO flag.  The access pattern IS random (10 of 512 experts per layer, a different 10
    // each layer), but every byte read is read again on the next token, so retention is the whole game.
    HANDLE f = CreateFileW(wpath.data(), GENERIC_READ, FILE_SHARE_READ, nullptr, OPEN_EXISTING,
                           FILE_ATTRIBUTE_NORMAL, nullptr);
    if (f == INVALID_HANDLE_VALUE) {
        err = "FileExpertSource: cannot open " + path;
        return false;
    }
    LARGE_INTEGER sz{};
    if (!GetFileSizeEx(f, &sz)) {
        CloseHandle(f);
        err = "FileExpertSource: cannot size " + path;
        return false;
    }
    if ((uint64_t) sz.QuadPart != want) {
        char buf[400];
        std::snprintf(buf, sizeof buf,
                      "FileExpertSource: %s is %llu B but the loaded expert layout requires %llu B - this is not "
                      "the pack this geometry came from",
                      path.c_str(), (unsigned long long) sz.QuadPart, (unsigned long long) want);
        CloseHandle(f);
        err = buf;
        return false;
    }
    HANDLE m = CreateFileMappingW(f, nullptr, PAGE_READONLY, 0, 0, nullptr);
    if (m == nullptr) {
        CloseHandle(f);
        err = "FileExpertSource: CreateFileMapping failed on " + path;
        return false;
    }
    void* view = MapViewOfFile(m, FILE_MAP_READ, 0, 0, 0);
    if (view == nullptr) {
        CloseHandle(m);
        CloseHandle(f);
        err = "FileExpertSource: MapViewOfFile failed on " + path;
        return false;
    }
    file_ = f;
    mapping_ = m;
    base_ = (const uint8_t*) view;
#else
    const int fd = ::open(path.c_str(), O_RDONLY);
    if (fd < 0) { err = "FileExpertSource: cannot open " + path; return false; }
    struct stat st{};
    if (fstat(fd, &st) != 0) { ::close(fd); err = "FileExpertSource: cannot stat " + path; return false; }
    if (st.st_size < 0 || (uint64_t) st.st_size != want) {
        char buf[400];
        std::snprintf(buf, sizeof buf,
                      "FileExpertSource: %s is %llu B but the loaded expert layout requires %llu B - this is not "
                      "the pack this geometry came from",
                      path.c_str(), (unsigned long long) (st.st_size < 0 ? 0 : st.st_size),
                      (unsigned long long) want);
        ::close(fd);
        err = buf;
        return false;
    }
    void* view = mmap(nullptr, (size_t) want, PROT_READ, MAP_SHARED, fd, 0);
    if (view == MAP_FAILED) { ::close(fd); err = "FileExpertSource: mmap failed on " + path; return false; }
    fd_ = fd;
    base_ = (const uint8_t*) view;
#endif
    blobs_ = (int64_t) blob_count;
    n_layers_ = n_layers;
    n_expert_ = n_expert;
    mapped_bytes_ = want;
    layer_offsets_ = std::move(layer_offsets);
    layer_blob_bytes_ = std::move(layer_blob_bytes);
    return true;
}

void FileExpertSource::close() {
    if (complement_arena_ != nullptr) {
        if (complement_pinned_) (void) cudaFreeHost(complement_arena_);
        else std::free(complement_arena_);
    }
    complement_arena_ = nullptr;
    complement_host_ = nullptr;
    complement_device_ = nullptr;
    complement_bytes_ = 0;
    complement_offsets_.clear();
    complement_pinned_ = false;
    complement_ready_ = false;
#if defined(_WIN32)
    if (base_ != nullptr) UnmapViewOfFile((LPCVOID) base_);
    if (mapping_ != nullptr) CloseHandle((HANDLE) mapping_);
    if (file_ != nullptr) CloseHandle((HANDLE) file_);
    mapping_ = nullptr;
    file_ = nullptr;
#else
    if (base_ != nullptr) munmap((void*) base_, (size_t) mapped_bytes_);
    if (fd_ >= 0) ::close(fd_);
    fd_ = -1;
#endif
    base_ = nullptr;
    blobs_ = 0;
    n_layers_ = 0;
    n_expert_ = 0;
    mapped_bytes_ = 0;
    layer_offsets_.clear();
    layer_blob_bytes_.clear();
    reads_ = 0;
}

const uint8_t* FileExpertSource::mapped_blob(int64_t layer, int64_t expert) const {
    if (base_ == nullptr || layer < 0 || expert < 0 || layer >= n_layers_ || expert >= n_expert_) return nullptr;
    const size_t i = (size_t) layer;
    if (i >= layer_offsets_.size() || i >= layer_blob_bytes_.size()) return nullptr;
    const uint64_t blob_bytes = layer_blob_bytes_[i];
    if (blob_bytes == 0 || (uint64_t) expert > std::numeric_limits<uint64_t>::max() / blob_bytes) return nullptr;
    const uint64_t expert_offset = (uint64_t) expert * blob_bytes;
    const uint64_t layer_offset = layer_offsets_[i];
    if (layer_offset > mapped_bytes_ || expert_offset > mapped_bytes_ - layer_offset) return nullptr;
    const uint64_t offset = layer_offset + expert_offset;
    if (blob_bytes > mapped_bytes_ - offset) return nullptr;
    return base_ + (size_t) offset;
}

bool FileExpertSource::pin_cache_complement(
    const ExpertCache& cache, std::string& err, bool pin,
    const std::vector<std::pair<int32_t, int32_t>>& additional_gpu_pairs) {
    err.clear();
    if (base_ == nullptr) { err = "FileExpertSource: open the mapped experts before pinning a complement"; return false; }
    if (complement_ready_) { err = "FileExpertSource: the cache complement is already pinned"; return false; }
    if (!cache.valid()) { err = "FileExpertSource: the GPU expert cache is not open"; return false; }
    if (cache.fills() != cache.resident()) {
        err = "FileExpertSource: the GPU expert cache is not fully filled";
        return false;
    }
    const cudaError_t sync = cudaDeviceSynchronize();
    if (sync != cudaSuccess) {
        err = std::string("FileExpertSource: GPU expert cache is not ready: ") + cudaGetErrorString(sync);
        (void) cudaGetLastError();
        return false;
    }

    std::vector<std::pair<int32_t, int32_t>> primary_gpu_pairs;
    primary_gpu_pairs.reserve((size_t) cache.resident());
    for (int64_t layer = 0; layer < n_layers_; ++layer) {
        for (int64_t expert = 0; expert < n_expert_; ++expert) {
            if (cache.slot_of(layer, expert) != kNotResident)
                primary_gpu_pairs.emplace_back((int32_t) layer, (int32_t) expert);
        }
    }
    std::vector<uint64_t> offsets;
    uint64_t bytes = 0;
    if (!detail::make_cache_complement_plan(n_layers_, n_expert_, layer_blob_bytes_, primary_gpu_pairs,
                                            additional_gpu_pairs, offsets, bytes, err)) return false;

    if (bytes > 0) {
        uint64_t physical = 0;
        if (!available_memory_bytes(physical)) {
            err = "FileExpertSource: cannot determine available RAM for the resident-memory safety check";
            return false;
        }
        if (physical <= kPinnedMemoryHeadroom || bytes > physical - kPinnedMemoryHeadroom) {
            char message[256];
            std::snprintf(message, sizeof message,
                          "FileExpertSource: resident complement %.2f GiB exceeds available RAM minus the 8 GiB safety headroom",
                          (double) bytes / 1073741824.0);
            err = message;
            return false;
        }
    }

    void* arena = nullptr;
    const uint8_t* host = nullptr;
    const uint8_t* device = nullptr;
    if (bytes > 0) {
        std::fprintf(stderr, "FileExpertSource: allocating %.2f GiB %s cache complement\n",
                     (double) bytes / 1073741824.0, pin ? "mapped pinned" : "pageable resident");
        std::fflush(stderr);
        if (pin) {
            const cudaError_t allocated = cudaHostAlloc(&arena, (size_t) bytes,
                                                         cudaHostAllocMapped | cudaHostAllocPortable);
            if (allocated != cudaSuccess) {
                err = std::string("FileExpertSource: mapped pinned complement allocation failed: ") +
                      cudaGetErrorString(allocated);
                std::fprintf(stderr, "%s\n", err.c_str());
                std::fflush(stderr);
                (void) cudaGetLastError();
                return false;
            }
            host = (const uint8_t*) arena;
            void* alias = nullptr;
            const cudaError_t aliased = cudaHostGetDevicePointer(&alias, arena, 0);
            if (aliased != cudaSuccess || alias == nullptr) {
                err = std::string("FileExpertSource: mapped pinned complement device alias failed: ") +
                      cudaGetErrorString(aliased);
                std::fprintf(stderr, "%s\n", err.c_str());
                std::fflush(stderr);
                (void) cudaGetLastError();
                (void) cudaFreeHost(arena);
                return false;
            }
            device = (const uint8_t*) alias;
        } else {
            arena = std::malloc((size_t) bytes);
            if (arena == nullptr) {
                err = "FileExpertSource: pageable resident complement allocation failed";
                return false;
            }
            host = (const uint8_t*) arena;
        }
    }

    uint64_t copied = 0;
    for (int64_t layer = 0; layer < n_layers_; ++layer) {
        const uint64_t blob_bytes = layer_blob_bytes_[(size_t) layer];
        for (int64_t expert = 0; expert < n_expert_; ++expert) {
            const size_t index = (size_t) layer * (size_t) n_expert_ + (size_t) expert;
            const uint64_t offset = offsets[index];
            if (offset == kNoComplement) continue;
            const uint8_t* source = mapped_blob(layer, expert);
            if (source == nullptr || offset > bytes || blob_bytes > bytes - offset) {
                err = "FileExpertSource: invalid blob bounds while building the cache complement";
                if (arena != nullptr) {
                    if (pin) (void) cudaFreeHost(arena);
                    else std::free(arena);
                }
                return false;
            }
            std::memcpy((uint8_t*) host + (size_t) offset, source, (size_t) blob_bytes);
            copied += blob_bytes;
        }
#if !defined(_WIN32)
        const uint64_t layer_offset = layer_offsets_[(size_t) layer];
        const uint64_t layer_bytes = blob_bytes * (uint64_t) n_expert_;
        const uint64_t layer_end = layer_offset + layer_bytes;
        const long page_bytes = sysconf(_SC_PAGESIZE);
        if (page_bytes <= 0) {
            err = "FileExpertSource: cannot determine page size for mapped-page release";
            if (arena != nullptr) {
                if (pin) (void) cudaFreeHost(arena);
                else std::free(arena);
            }
            return false;
        }
        const uint64_t page = (uint64_t) page_bytes;
        const uint64_t advice_start = layer_offset - layer_offset % page;
        const uint64_t end_remainder = layer_end % page;
        const uint64_t extra = end_remainder == 0 ? 0 : page - end_remainder;
        const uint64_t advice_end = extra > mapped_bytes_ - layer_end ? mapped_bytes_ : layer_end + extra;
        if (advice_end > advice_start &&
            madvise((void*) (base_ + (size_t) advice_start), (size_t) (advice_end - advice_start), MADV_DONTNEED) != 0) {
            err = "FileExpertSource: madvise could not release mapped expert layer " + std::to_string(layer);
            if (arena != nullptr) {
                if (pin) (void) cudaFreeHost(arena);
                else std::free(arena);
            }
            return false;
        }
        const int advice = posix_fadvise(fd_, (off_t) layer_offset, (off_t) layer_bytes, POSIX_FADV_DONTNEED);
        if (advice != 0) {
            err = "FileExpertSource: posix_fadvise could not release expert layer " + std::to_string(layer);
            if (arena != nullptr) {
                if (pin) (void) cudaFreeHost(arena);
                else std::free(arena);
            }
            return false;
        }
#endif
        if ((layer + 1) % 8 == 0 || layer + 1 == n_layers_) {
            std::fprintf(stderr, "FileExpertSource: copied cache complement through layer %lld/%lld (%.2f GiB)\n",
                         (long long) (layer + 1), (long long) n_layers_, (double) copied / 1073741824.0);
            std::fflush(stderr);
        }
    }

    complement_arena_ = arena;
    complement_host_ = host;
    complement_device_ = device;
    complement_bytes_ = bytes;
    complement_offsets_ = std::move(offsets);
    complement_pinned_ = pin && bytes > 0;
    complement_ready_ = true;
    std::fprintf(stderr, "FileExpertSource: %s cache complement ready: resident %.2f GiB, pinned %.2f GiB\n",
                 complement_pinned_ ? "mapped pinned" : "pageable resident",
                 (double) resident_bytes() / 1073741824.0, (double) pinned_bytes() / 1073741824.0);
    if (!additional_gpu_pairs.empty()) {
        std::fprintf(stderr, "FileExpertSource: %zu verified additional-GPU experts remain on the mmap fallback\n",
                     additional_gpu_pairs.size());
    }
    std::fflush(stderr);
    return true;
}

const uint8_t* FileExpertSource::blob(int64_t layer, int64_t expert) {
    if (base_ == nullptr || layer < 0 || expert < 0 || layer >= n_layers_ || expert >= n_expert_) return nullptr;
    const size_t index = (size_t) layer * (size_t) n_expert_ + (size_t) expert;
    const uint8_t* mapped_fallback = mapped_blob(layer, expert);
    const uint8_t* result = complement_ready_
        ? detail::cache_complement_blob_or_fallback(index, complement_offsets_, complement_host_, mapped_fallback)
        : mapped_fallback;
    if (result != nullptr) ++reads_;
    return result;
}

bool FileExpertSource::pinned(int64_t layer, int64_t expert) const {
    if (!complement_ready_ || !complement_pinned_ || complement_host_ == nullptr || layer < 0 || expert < 0 ||
        layer >= n_layers_ || expert >= n_expert_) return false;
    const size_t index = (size_t) layer * (size_t) n_expert_ + (size_t) expert;
    return index < complement_offsets_.size() && complement_offsets_[index] != kNoComplement;
}

const uint8_t* FileExpertSource::device_alias(int64_t layer, int64_t expert) const {
    if (!pinned(layer, expert) || complement_device_ == nullptr) return nullptr;
    const size_t index = (size_t) layer * (size_t) n_expert_ + (size_t) expert;
    return complement_device_ + (size_t) complement_offsets_[index];
}

// ================================ THE ADAPTER ================================

void expert_pool_dispatch(void* user, const float* x_f, const int32_t* ids, const float* weights, int64_t n_embd,
                          int64_t k, float* out) {
    (void) weights;   // clause 2: `moe_combine` applies it on the device.  Not an oversight.
    ExpertDispatch& d = *(ExpertDispatch*) user;
    if (d.failed) return;   // a previous layer already failed; do not make it worse

    using namespace strata::kernels::cpu;
    // Clause 3: the blob's internal offsets are compile-time constants, so a mismatched geometry does not
    // produce a wrong answer - it produces a walk off the end of the blob into the next expert's bytes, which
    // is finite and plausible.  Refuse, name the number, and let the driver report it.
    if (n_embd != H) {
        d.failed = true;
        d.fail = "the expert kernel is compiled for a 2560-wide activation";
        d.fail_layer = d.layers;
        return;
    }
    if (expert_layout().native) {
        // plan v0.3 P6: a native pack runs its experts in verify windows only (the driver guarantees it)
        d.failed = true;
        d.fail = "the single-token expert path does not take a native (IQ) pack";
        d.fail_layer = d.layers;
        return;
    }
    if (k > (int64_t) d.jobs.size()) d.jobs.resize((size_t) k);

    d.src->begin_layer(d.layers, ids, k);

    // Clause 1: rebuilt from `x_f` on EVERY call.  `x_f` is mapped pinned memory whose address never changes,
    // so anything cached against it would be layer 0's activation reused 48 times.
    act_quant_q8_1(x_f, H, d.act);

    // ---- R4.2c: THE POOL'S HALF OF THE SPLIT.  **IT DOES NOT DECIDE ANYTHING - `Launch` ALREADY DID.**
    //
    // The decision has to be made on THIS layer's ids, and `Launch` is the only callback that runs before the
    // pool while the ids are known (the doorbell publishes them when the ring fires).  So `expert_hit_run`
    // decides, and this consumes `d.is_hit`.  The first version decided here instead, which meant `Launch`
    // computed the PREVIOUS layer's experts into this layer's rows: C1 went from mean KL 9.69e-02 to 1.03e+00.
    //
    // `njobs` indexes the JOB ARRAY and `i` indexes the OUTPUT - they are the same only when nothing is a hit.
    const bool graph_hits = d.host_res != nullptr;
    const bool use_hits = graph_hits || (d.hits_ready() && d.decided);
    int64_t njobs = 0;

    if (d.remote_count > 0) {
        int32_t kind[32];
        if (k > 32) {
            d.failed = true; d.fail = "remote experts: routing width exceeds 32"; return;
        }
        for (int64_t i = 0; i < k; ++i)
            kind[i] = use_hits && ids[i] >= 0 && ids[i] < d.n_expert && (graph_hits
                ? d.host_res[(size_t) d.layers * (size_t) d.n_expert + (size_t) ids[i]] >= 0
                : d.is_hit[(size_t) i] != 0) ? 0 : -1;
        static thread_local std::string remote_error;
        for (int r = 0; r < d.remote_count; ++r)
            if (!d.remote[r]->begin(d.layers, x_f, ids, 1, k, kind, d.host_res, remote_error)) {
                d.failed = true; d.fail = remote_error.c_str(); d.fail_layer = d.layers; return;
            }
    }

    for (int64_t i = 0; i < k; ++i) {
        const int64_t e = ids[i];
        if (e < 0 || e >= d.n_expert) {
            d.failed = true;
            d.fail = "a routed expert id is out of range";
            d.fail_layer = d.layers;
            d.fail_expert = e;
            return;
        }
        const uint8_t* b = d.src->blob(d.layers, e);
        if (b == nullptr) {
            // The one failure the loop cannot see.  Leaving `out` at its previous contents would feed the NEXT
            // layer a stale expert vector, which `moe_combine` would weight and add - the token would still be
            // finite and would still be wrong, 48 layers deep.
            d.failed = true;
            d.fail = "the expert source could not produce a blob";
            d.fail_layer = d.layers;
            d.fail_expert = e;
            ++d.missing;
            return;
        }
        // A hit's row was zeroed by `Launch` and belongs to the GPU; the pool must not touch it.
        if (use_hits && (graph_hits ? d.host_res[(size_t) d.layers * (size_t) d.n_expert + (size_t) e] >= 0
                                    : d.is_hit[(size_t) i] != 0)) {
            if (graph_hits) ++d.cache_hits;
            // The GPU owns this row and `hit_out` is zeroed, so the CPU's contribution is zero - but
            // `y_miss` is a REUSED pinned buffer, so the row must be written, not merely skipped.
            std::memset(out + (size_t) i * (size_t) n_embd, 0, (size_t) n_embd * sizeof(float));
            continue;
        }
        if (graph_hits) ++d.cache_refused;   // token graph: a miss (nothing is admitted during a token)

        bool remote_owns = false;
        for (int r = 0; r < d.remote_count; ++r) remote_owns |= d.remote[r]->owns(i);
        if (remote_owns) {
            std::memset(out + (size_t) i * (size_t) n_embd, 0, (size_t) n_embd * sizeof(float));
            continue;
        }

        // `njobs` indexes the JOB ARRAY and `i` indexes the OUTPUT - they are the same only when nothing is a
        // hit, and using one for the other is how a hit's row would get two experts summed into it.
        ExpertJob& j = d.jobs[(size_t) njobs++];
        j.blob = b;
        j.act = &d.act;             // SHARED across the batch: one conversion serves all ten experts
        j.out = out + (size_t) i * (size_t) n_embd;
        j.weight = 1.0f;            // clause 2: a diagnostic field, NOT the router weight
        j.slot = (int) i;
    }

    // Plan v0.3 P4: rows of every expert across all threads (bitwise the same as `run`).
    if (d.split_rows) d.pool->run_split(d.jobs.data(), (int) njobs);
    else d.pool->run(d.jobs.data(), (int) njobs);
    if (d.remote_count > 0) {
        static thread_local std::string remote_error;
        for (int r = 0; r < d.remote_count; ++r)
            if (!d.remote[r]->finish(out, remote_error)) {
                d.failed = true; d.fail = remote_error.c_str(); d.fail_layer = d.layers; return;
            }
    }
    ++d.layers;
    d.experts += k;
}

namespace {
// the verify window's per-entry tables in `expert_pool_dispatch_multi` (`kind`, `distinct`, `first_of`)
// are fixed arrays of this many entries: MAXT tokens of the model's 10 routed experts must fit, and a larger k is
// refused at run time rather than written past them.
constexpr int64_t kMaxWindowEntries = 128;
static_assert(strata::kernels::cpu::MAXT * 10 <= kMaxWindowEntries, "a verify window's entries overflow the tables");
}  // namespace

void expert_pool_dispatch_multi(ExpertDispatch& d, const float* x_f, const int32_t* ids, int64_t n_tok, int64_t k,
                                float* out) {
    using namespace strata::kernels::cpu;
    if (d.failed) return;
    if (n_tok < 1 || n_tok > MAXT) {
        d.failed = true;
        d.fail = "a verify window has more tokens than the multi-token expert kernel takes";
        d.fail_layer = d.layers;
        return;
    }
    if (k < 1 || n_tok * k > kMaxWindowEntries) {
        d.failed = true;
        d.fail = "a verify window routes more entries than the expert pool's window tables hold";
        d.fail_layer = d.layers;
        return;
    }
    if ((int64_t) d.act_multi.size() < n_tok) d.act_multi.resize((size_t) MAXT);
    const ExpertLayout& lay = expert_layout();
    const bool native = lay.native;
    if (native && d.nact_multi.size() < (size_t) MAXT * kNativeActBytes) d.nact_multi.resize((size_t) MAXT * kNativeActBytes);
    if (d.job_of.size() != (size_t) d.n_expert) d.job_of.assign((size_t) d.n_expert, (int16_t) -1);
    if (d.jobs_multi.size() < (size_t) (n_tok * k)) d.jobs_multi.resize((size_t) (MAXT * k));
    static const bool ptrace = std::getenv("STRATA_POOL_TRACE") != nullptr;
    auto pt = [&](const char* what, long long a = -1) {
        if (ptrace) { std::fprintf(stderr, "pool trace: layer %lld %s %lld\n", (long long) d.layers, what, a); std::fflush(stderr); }
    };
    const auto c0 = std::chrono::steady_clock::now();
    pt("begin");
    d.src->begin_layer(d.layers, ids, n_tok * k);
    pt("begun");
    if (!d.usage.empty())
        for (int64_t i = 0; i < n_tok * k; ++i)
            if (ids[i] >= 0 && ids[i] < d.n_expert) d.usage[(size_t) d.layers * (size_t) d.n_expert + (size_t) ids[i]] += 1.0f;
    // ---- plan v0.3 P6: the GPU's share, decided and published FIRST so the GPU starts while the CPU works.
    // Distinct experts in routing order; resident ones and the last pcie_num/256 of the missed ones go to the GPU.
    const int64_t n = n_tok * k;
    int32_t kind[kMaxWindowEntries];       // per entry: -1 CPU, 0 VRAM, 1 PCIe
    if (d.plan != nullptr && n <= kMaxWindowEntries && n <= d.plan->cap) {
        int64_t distinct[kMaxWindowEntries], first_of[kMaxWindowEntries];
        int nd = 0, nmiss = 0;
        for (int64_t i = 0; i < n; ++i) {
            first_of[i] = i;
            for (int64_t j = 0; j < i; ++j)
                if (ids[j] == ids[i]) { first_of[i] = first_of[j]; break; }
            if (first_of[i] == i) {
                distinct[nd++] = i;
                const int32_t e = ids[i];
                if (e >= 0 && e < d.n_expert && d.host_res[(size_t) d.layers * (size_t) d.n_expert + (size_t) e] < 0) ++nmiss;
            }
        }
        const bool pcie_ok = d.pcie_num > 0 && d.src->dma_capable(d.layers);
        const int m = pcie_ok ? (nmiss * d.pcie_num) >> 8 : 0;
        int miss_rank = 0, groups = 0, entries = 0, fetches = 0;
        GpuPlanSink& P = *d.plan;
        const uint8_t* dma_src[64];
        int64_t pcie_i0[64];
        for (int q = 0; q < nd; ++q) {
            const int64_t i0 = distinct[q];
            const int32_t e = ids[i0];
            int kd = -1;
            unsigned long long ptr = 0;
            if (e >= 0 && e < d.n_expert) {
                const int32_t slot = d.host_res[(size_t) d.layers * (size_t) d.n_expert + (size_t) e];
                if (slot >= 0) {
                    kd = 0;
                    ptr = (unsigned long long) (d.cache_base + (d.cache_slot_off ? (size_t) d.cache_slot_off[slot]
                                                                                 : (size_t) slot * (size_t) d.cache_blob));
                } else {
                    if (miss_rank >= nmiss - m && fetches < P.staging_cap && fetches < 64) {
                        const uint8_t* src = d.src->pinned(d.layers, e) ? d.src->blob(d.layers, e) : nullptr;
                        if (src != nullptr) {
                            kd = 1;
                            dma_src[fetches] = src;
                            pcie_i0[fetches] = i0;
                            ++fetches;
                        }
                    }
                    ++miss_rank;
                }
            }
            for (int64_t i = i0; i < n; ++i)
                if (first_of[i] == i0) kind[i] = kd;
            if (kd != 0) continue;                 // the VRAM groups first; the PCIe groups below
            P.ptr[groups] = ptr;
            P.start[groups] = entries;
            for (int64_t i = i0; i < n; ++i)
                if (first_of[i] == i0) {
                    P.dst[entries] = (int32_t) i;
                    P.tok[entries] = (int32_t) (i / k);
                    ++entries;
                }
            ++groups;
        }
        P.start[groups] = entries;
        const uint64_t bb = lay.blob_bytes(d.layers);
        for (int q = 0; q < fetches; ++q) {       // the PCIe groups: staging slot q, entries after the VRAM ones
            const int64_t i0 = pcie_i0[q];
            P.ptr2[q] = P.pcie_mode != 0 ? (unsigned long long) d.src->device_alias(d.layers, ids[i0])
                                 : P.staging + (unsigned long long) q * (unsigned long long) bb;
            P.start2[q] = entries;
            for (int64_t i = i0; i < n; ++i)
                if (first_of[i] == i0) {
                    P.dst[entries] = (int32_t) i;
                    P.tok[entries] = (int32_t) (i / k);
                    ++entries;
                }
            ++d.pcie_experts;
        }
        P.start2[fetches] = entries;
        P.counts[0] = groups;
        P.counts[1] = entries;
        P.counts[2] = fetches;
        std::atomic_thread_fence(std::memory_order_seq_cst);
        pt("publish", fetches);
        if (P.publish) P.publish(P.ctx);
        pt("fetch", fetches);
        if (P.fetch) P.fetch(P.ctx, dma_src, P.pcie_mode != 0 ? 0 : fetches, (size_t) bb);   // the copy engine, beside the CPU's work
    } else {
        for (int64_t i = 0; i < n; ++i) {
            const int32_t e = ids[i];
            kind[i] = (e >= 0 && e < d.n_expert && d.host_res != nullptr &&
                       d.host_res[(size_t) d.layers * (size_t) d.n_expert + (size_t) e] >= 0) ? 0 : -1;
        }
    }
    if (d.remote_count > 0) {
        static thread_local std::string remote_error;
        for (int r = 0; r < d.remote_count; ++r) {
            if (!d.remote[r]->begin(d.layers, x_f, ids, n_tok, k, kind, d.host_res, remote_error)) {
                d.failed = true; d.fail = remote_error.c_str(); d.fail_layer = d.layers; return;
            }
            for (int64_t i = 0; i < n; ++i) if (d.remote[r]->owns(i)) kind[i] = 2;
        }
    }
    const auto c1 = std::chrono::steady_clock::now();
    if (native && lay.fmt[(size_t) d.layers].gu_type == 42)   // a native Q2_0 pack: the Q2_0 kernels' activations
        for (int64_t t = 0; t < n_tok; ++t) act_quant_any(x_f + (size_t) t * H, H, d.act_multi[(size_t) t]);
    else if (native)
        for (int64_t t = 0; t < n_tok; ++t)
            native_quant_act(lay.fmt[(size_t) d.layers], x_f + (size_t) t * H, d.nact_multi.data() + (size_t) t * kNativeActBytes);
    else
        for (int64_t t = 0; t < n_tok; ++t) act_quant_q8_1(x_f + (size_t) t * H, H, d.act_multi[(size_t) t]);
    const auto c2 = std::chrono::steady_clock::now();
    int njobs = 0;
    for (int64_t t = 0; t < n_tok; ++t)
        for (int64_t j = 0; j < k; ++j) {
            const int64_t i = t * k + j;
            const int64_t e = ids[i];
            float* row = out + (size_t) i * H;
            if (e < 0 || e >= d.n_expert) {
                d.failed = true;
                d.fail = "a routed expert id is out of range";
                d.fail_layer = d.layers;
                d.fail_expert = e;
                return;
            }
            if (kind[i] >= 0) {             // CUDA0, PCIe, or a remote result staged into this row below
                if (kind[i] == 0) ++d.cache_hits;
                std::memset(row, 0, (size_t) H * sizeof(float));
                continue;
            }
            ++d.cache_refused;
            int16_t& jo = d.job_of[(size_t) e];
            if (jo < 0) {
                const uint8_t* b = d.src->blob(d.layers, e);
                if (b == nullptr) {
                    d.failed = true;
                    d.fail = "the expert source could not produce a blob";
                    d.fail_layer = d.layers;
                    d.fail_expert = e;
                    ++d.missing;
                    return;
                }
                jo = (int16_t) njobs++;
                ExpertJobMulti& nj = d.jobs_multi[(size_t) jo];
                nj.blob = b;
                nj.nt = 0;
            }
            ExpertJobMulti& jb = d.jobs_multi[(size_t) jo];
            jb.act[jb.nt] = &d.act_multi[(size_t) t];
            jb.nact[jb.nt] = native ? d.nact_multi.data() + (size_t) t * kNativeActBytes : nullptr;
            jb.out[jb.nt] = row;
            ++jb.nt;
            ++d.multi_entries;
        }
    const auto c3 = std::chrono::steady_clock::now();
    pt("run", njobs);
    if (native) d.pool->run_split_multi_native(lay.fmt[(size_t) d.layers], d.jobs_multi.data(), njobs);
    else d.pool->run_split_multi(d.jobs_multi.data(), njobs);
    if (d.remote_count > 0) {
        static thread_local std::string remote_error;
        for (int r = 0; r < d.remote_count; ++r)
            if (!d.remote[r]->finish(out, remote_error)) {
                d.failed = true; d.fail = remote_error.c_str(); d.fail_layer = d.layers; return;
            }
    }
    const auto c4 = std::chrono::steady_clock::now();
    pt("ran");
    auto ms = [](auto a, auto b) { return std::chrono::duration<double, std::milli>(b - a).count(); };
    d.ms_plan += ms(c0, c1);
    d.ms_actq += ms(c1, c2);
    d.ms_jobs += ms(c2, c3);
    d.ms_run += ms(c3, c4);
    for (int64_t i = 0; i < n_tok * k; ++i) {
        const int64_t e = ids[i];
        if (e >= 0 && e < d.n_expert) d.job_of[(size_t) e] = -1;
    }
    d.multi_misses += njobs;
    ++d.layers;
    d.experts += n_tok * k;
}

void expert_hit_run(void* user, void* stream, HitPhase phase, const int32_t* ids, int64_t k) {
    ExpertDispatch& d = *(ExpertDispatch*) user;
    if (d.failed) return;
    cudaStream_t cs = (cudaStream_t) stream;

    if (phase == HitPhase::Launch) {
        d.decided = false;
        d.hit_pending = false;
        if (!d.hits_ready() || ids == nullptr || k <= 0) return;
        if ((int64_t) d.is_hit.size() < k) d.is_hit.resize((size_t) k);

        // ================================ THE DECISION, ONCE, ON THIS LAYER'S IDS ================================
        //
        // Every routed expert is asked of the cache.  Resident -> the GPU computes it.  Not resident -> it is
        // admitted and filled if there is room (which makes it a hit on THIS call, because the fill and the
        // kernel are on one stream in that order), and otherwise it stays a miss for the CPU.
        d.n_hits = 0;
        for (int64_t i = 0; i < k; ++i) {
            const int64_t e = ids[i];
            d.is_hit[(size_t) i] = 0;
            if (e < 0 || e >= d.n_expert) continue;   // out of range: the pool refuses it, with a message
            int32_t slot = d.cache->slot_of(d.layers, e);
            if (slot == kNotResident) {
                const int32_t cand = d.cache->admit(d.layers, e);
                if (cand == kNotResident) {
                    ++d.cache_refused;
                    continue;
                }
                // `blob` is asked ONLY for an expert about to be filled, so the source's read counter stays a
                // count of distinct experts moved rather than of looks.
                const uint8_t* b = d.src->blob(d.layers, e);
                std::string ferr;
                if (b == nullptr || !d.cache->fill_slot(cand, b, cs, ferr, (int64_t) strata::kernels::cpu::expert_layout().blob_bytes(d.layers))) {
                    d.failed = true;
                    d.fail = "the expert cache could not fill a slot";
                    d.fail_layer = d.layers;
                    d.fail_expert = e;
                    return;
                }
                ++d.cache_admitted;
                slot = cand;
            } else {
                ++d.cache_hits;
            }
            d.is_hit[(size_t) i] = 1;
            d.h_slot[(size_t) d.n_hits] = slot;
            d.h_dst[(size_t) d.n_hits] = (int32_t) i;
            ++d.n_hits;
        }
        d.decided = true;
        if (d.n_hits <= 0) return;   // nothing resident yet: no GPU work, and nothing for `Combine` to add

        const size_t list_bytes = (size_t) d.n_hits * sizeof(int32_t);
        // `hit_out` is ZEROED rather than overwritten: the kernel writes only the rows this layer's hits own,
        // so a row that was a hit last layer and a miss this one would still hold last layer's expert and
        // `add_inplace` would sum it in.  Finite, plausible, wrong.
        if (cudaMemsetAsync(d.hit_out, 0, (size_t) d.parts_elems * sizeof(float), cs) != cudaSuccess ||
            cudaMemcpyAsync(d.d_slot, d.h_slot.data(), list_bytes, cudaMemcpyHostToDevice, cs) != cudaSuccess ||
            cudaMemcpyAsync(d.d_dst, d.h_dst.data(), list_bytes, cudaMemcpyHostToDevice, cs) != cudaSuccess) {
            d.hit_fail = "the hit list could not be staged";
            d.failed = true;
            d.fail = d.hit_fail;
            return;
        }
        // The activation is quantized HERE rather than reused from `s.moe.x_q8_0`, which `post[l-1]` wrote from
        // the PREVIOUS layer's `mixed`.  `pre[l]` has since overwritten `mixed`, so that buffer is a layer stale
        // - and a stale activation produces a perfectly finite expert for the wrong input.
        // **R4.2h: THE SCALED QUANTIZER, SO A HIT REPRODUCES A MISS.**  The CPU pool quantizes this same
        // activation with `act_quant_q8_1` and multiplies by the fp32 `ActQ::scale`; `quantize_q8_0` writes
        // an fp16 `d` instead, and `bench/micro/act_quant_parity.cu` measured **80 of 80 chunks differing by
        // up to 4.761e-04 relative**.  `quantize_q8_0_scaled` adopts the CPU's rule and scale, and the kernel
        // takes the fp32 array.  Falling back to the old path would silently reintroduce the divergence, so
        // the scales are required here rather than optional.
        if (d.x_q8_0_hit_scale == nullptr) {
            d.failed = true;
            d.fail = "the hit path has no fp32 activation scales (R4.2h)";
            return;
        }
        strata::kernels::quantize_q8_0_scaled(d.mixed, d.x_q8_0_hit, d.x_q8_0_hit_scale, strata::kernels::cpu::H,
                                              cs);
        if (d.hit_cpu_order)
            strata::kernels::moe_hit_grouped_s2_cpu_order(d.cache_base, d.d_slot, d.d_dst, d.n_hits,
                d.cache_blob, d.x_q8_0_hit, d.hit_scratch, d.hit_out, cs, d.x_q8_0_hit_scale);
        else
            strata::kernels::moe_hit_grouped_s2(d.cache_base, d.d_slot, d.d_dst, d.n_hits, d.cache_blob,
                d.x_q8_0_hit, d.hit_scratch, d.hit_out, cs, d.x_q8_0_hit_scale);
        d.hit_pending = true;
        if (d.hit_done != nullptr) cudaEventRecord((cudaEvent_t) d.hit_done, cs);
        // The A/B arm: ONE driver entry here, and nothing else changes.  If the work was waiting for the host
        // to enter the driver, this is what lets it start while the pool runs.
        if (d.hit_poke && d.hit_done != nullptr) (void) cudaEventQuery((cudaEvent_t) d.hit_done);
        return;
    }

    // Combine: `parts += hit_out`, stream-ordered after the misses were copied into `parts`.
    if (!d.hit_pending) return;
    d.hit_pending = false;
    // Did the GPU get the hit work done while the CPU was in the pool?  This query is itself a driver entry,
    // so it is the LAST chance to observe a late start: a NOT-READY here means the work had not finished by the
    // time the pool returned, and with no poke in front of it that can only be because it began after.
    if (d.hit_done != nullptr) {
        if (cudaEventQuery((cudaEvent_t) d.hit_done) == cudaSuccess) ++d.hit_ready;
        else ++d.hit_late;
    }
    strata::kernels::add_inplace(d.parts_out, d.hit_out, d.parts_elems, cs);
}

// ================================ THE RESIDENT ARENA (R2.1) ================================

// Plan v0.3 P6: the arena from the model's shard 1.  Each layer's gate, up and down tensors hold the 512 experts
// one after another; they are read in chunks and each expert's slice lands at its place in the blob
// [gate rows | up rows | down rows] - the layout tools/iq_pack.py would have written to experts.bin.
//
// `unbuffered` (Windows): each chunk's 4 KiB-aligned window is read with FILE_FLAG_NO_BUFFERING into an aligned
// buffer (#230: MSVC's std::ifstream reads in 4095-byte pieces, 0.02 GiB/s on a normal install).  `ready`: layer L
// is written only once *ready > L (a PinnedArena registering its slices meanwhile: writing a slice while
// cudaHostRegister runs on it corrupts it).
LoadStats load_experts_gguf(const std::string& gguf, uint8_t* dst, const strata::kernels::cpu::ExpertLayout& lay,
                            int threads, const std::atomic<int>* ready = nullptr, bool unbuffered = false) {
    LoadStats st;
    st.layers = (uint64_t) lay.n_layers;
    const auto t0 = std::chrono::steady_clock::now();
    std::atomic<int64_t> next{0};
    std::atomic<bool> bad{false};
    auto wait_ready = [&](int64_t l) {
        if (ready != nullptr)
            while (ready->load(std::memory_order_acquire) <= l) std::this_thread::yield();
    };
    // a layer's experts may sit in another shard of the model (native_experts.txt v3): a name beside `gguf`
    const size_t cut = gguf.find_last_of("/\\");
    const std::string dir = cut == std::string::npos ? std::string() : gguf.substr(0, cut + 1);
    auto file_of = [&](int64_t l) -> std::string {
        if (lay.gguf_file.empty() || lay.gguf_file[(size_t) l].empty()) return gguf;
        return dir + lay.gguf_file[(size_t) l];
    };
#if defined(_WIN32)
    uint64_t max_chunk = 0;
    for (int64_t l = 0; l < lay.n_layers; ++l) {
        const auto& fm = lay.fmt[(size_t) l];
        max_chunk = std::max<uint64_t>(max_chunk, std::max<uint64_t>(fm.up_off, lay.bytes[(size_t) l] - fm.down_off) * 16);
    }
    std::mutex err_mu;
    std::string err;
    auto worker_unbuffered = [&]() {
        constexpr uint64_t kSector = 4096;
        const uint64_t cap = (max_chunk + 2 * kSector + kSector - 1) / kSector * kSector;
        uint8_t* buf = (uint8_t*) VirtualAlloc(nullptr, (size_t) cap, MEM_COMMIT | MEM_RESERVE, PAGE_READWRITE);
        HANDLE h = INVALID_HANDLE_VALUE;
        std::string open_name;
        auto fail = [&](const std::string& what) {
            std::lock_guard<std::mutex> g(err_mu);
            if (err.empty()) err = what;
            bad = true;
        };
        if (buf == nullptr) fail("cannot allocate a read buffer");
        for (;;) {
            const int64_t l = next.fetch_add(1);
            if (l >= lay.n_layers || bad) break;
            const std::string name = file_of(l);
            if (name != open_name) {
                if (h != INVALID_HANDLE_VALUE) CloseHandle(h);
                // UTF-8 -> UTF-16, as FileExpertSource::open does
                const int wide = MultiByteToWideChar(CP_UTF8, 0, name.c_str(), -1, nullptr, 0);
                std::vector<wchar_t> w((size_t) std::max(wide, 1), L'\0');
                if (wide > 0) MultiByteToWideChar(CP_UTF8, 0, name.c_str(), -1, w.data(), wide);
                h = CreateFileW(w.data(), GENERIC_READ, FILE_SHARE_READ, nullptr, OPEN_EXISTING,
                                FILE_FLAG_NO_BUFFERING | FILE_FLAG_SEQUENTIAL_SCAN, nullptr);
                if (h == INVALID_HANDLE_VALUE) {
                    fail("cannot open " + name + " (error " + std::to_string((unsigned long long) GetLastError()) + ")");
                    break;
                }
                open_name = name;
            }
            wait_ready(l);
            const auto& fm = lay.fmt[(size_t) l];
            const uint64_t blob = lay.bytes[(size_t) l];
            const uint64_t per[3] = {fm.up_off, fm.up_off, blob - fm.down_off};
            const uint64_t at[3] = {0, fm.up_off, fm.down_off};
            for (int r = 0; r < 3 && !bad; ++r) {
                const uint64_t src = lay.gguf_off[(size_t) (3 * l + r)];
                const uint64_t total = per[r] * (uint64_t) lay.n_expert;
                const uint64_t chunk = per[r] * 16;           // 16 experts per read
                for (uint64_t done = 0; done < total; done += chunk) {
                    const uint64_t n = std::min<uint64_t>(chunk, total - done);
                    const uint64_t a0 = (src + done) / kSector * kSector;
                    const uint64_t a1 = (src + done + n + kSector - 1) / kSector * kSector;
                    OVERLAPPED ov{};
                    ov.Offset = (DWORD) a0;
                    ov.OffsetHigh = (DWORD) (a0 >> 32);
                    DWORD got = 0;
                    // the window may run past the end of the file: only the tensor's own bytes have to arrive
                    if (!ReadFile(h, buf, (DWORD) (a1 - a0), &got, &ov) || (uint64_t) got < src + done - a0 + n) {
                        fail("short unbuffered read of layer " + std::to_string(l) + " in " + name + " (error " +
                             std::to_string((unsigned long long) GetLastError()) + ")");
                        break;
                    }
                    const uint8_t* p = buf + (src + done - a0);
                    for (uint64_t k = 0; k < n / per[r]; ++k) {
                        const uint64_t e = done / per[r] + k;
                        std::memcpy(dst + lay.blob_offset(l, (int64_t) e) + at[r], p + k * per[r], (size_t) per[r]);
                    }
                }
            }
        }
        if (h != INVALID_HANDLE_VALUE) CloseHandle(h);
        if (buf != nullptr) VirtualFree(buf, 0, MEM_RELEASE);
    };
    if (unbuffered) {
        std::vector<std::thread> pool;
        for (int i = 1; i < threads; ++i) pool.emplace_back(worker_unbuffered);
        worker_unbuffered();
        for (auto& t : pool) t.join();
        if (bad) {
            st.seconds = -1.0;
            st.ok = false;
            st.error = err.empty() ? "unreadable shard while reading the experts from the GGUF" : err;
            return st;
        }
        st.bytes = lay.total;
        st.seconds = std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count();
        return st;
    }
#else
    (void) unbuffered;
#endif
    auto worker = [&]() {
        std::ifstream f;
        std::string open_name;
        std::vector<uint8_t> buf;
        for (;;) {
            const int64_t l = next.fetch_add(1);
            if (l >= lay.n_layers || bad) break;
            const std::string name = file_of(l);
            if (name != open_name) {
                f.close();
                f.clear();
                f.open(name, std::ios::binary);
                if (!f) { bad = true; return; }
                open_name = name;
            }
            wait_ready(l);
            const auto& fm = lay.fmt[(size_t) l];
            const uint64_t blob = lay.bytes[(size_t) l];
            const uint64_t per[3] = {fm.up_off, fm.up_off, blob - fm.down_off};
            const uint64_t at[3] = {0, fm.up_off, fm.down_off};
            for (int r = 0; r < 3; ++r) {
                const uint64_t src = lay.gguf_off[(size_t) (3 * l + r)];
                const uint64_t total = per[r] * (uint64_t) lay.n_expert;
                const uint64_t chunk = per[r] * 16;           // 16 experts per read
                buf.resize((size_t) chunk);
                for (uint64_t done = 0; done < total; done += chunk) {
                    const uint64_t n = std::min<uint64_t>(chunk, total - done);
                    f.seekg((std::streamoff) (src + done));
                    f.read((char*) buf.data(), (std::streamsize) n);
                    if ((uint64_t) f.gcount() != n) { bad = true; return; }
                    for (uint64_t k = 0; k < n / per[r]; ++k) {
                        const uint64_t e = done / per[r] + k;
                        std::memcpy(dst + lay.blob_offset(l, (int64_t) e) + at[r], buf.data() + k * per[r], (size_t) per[r]);
                    }
                }
            }
        }
    };
    std::vector<std::thread> pool;
    for (int i = 1; i < threads; ++i) pool.emplace_back(worker);
    worker();
    for (auto& t : pool) t.join();
    if (bad) {
        st.seconds = -1.0;
        st.ok = false;
        st.error = "short read or unreadable shard while reading the experts from the GGUF";
        return st;
    }
    st.bytes = lay.total;
    st.seconds = std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count();
    return st;
}

ArenaExpertSource::~ArenaExpertSource() { close(); }

bool ArenaExpertSource::open(const std::string& pack_dir, int64_t n_layers, int64_t n_expert, int threads,
                             std::string& err, uint64_t max_pinned_bytes) {
    close();
    const std::string path = pack_dir + "/experts.bin";
    // plan v0.3 P6: the layout (canonical, or a native pack's per-layer blobs) was loaded by the driver
    const strata::kernels::cpu::ExpertLayout& lay = strata::kernels::cpu::expert_layout();
    if (lay.n_layers != n_layers || lay.n_expert != n_expert) {
        err = "ArenaExpertSource: the expert layout was loaded for a different geometry";
        return false;
    }
    const int64_t blob = (int64_t) lay.max_blob;
    const uint64_t want = lay.total;

    // plan v0.3 P6: no experts.bin in a native pack -> the experts come straight from the GGUF
    const bool from_gguf = !std::ifstream(path, std::ios::binary) && lay.native && !lay.gguf_off.empty() && !gguf_.empty();
    // SIZE CHECK BEFORE THE ALLOCATION, not after.  A wrong pack should name the two numbers rather than spend
    // 34 GB and a minute of loading first.
    if (!from_gguf) {
        std::ifstream f(path, std::ios::binary | std::ios::ate);
        if (!f) { err = "ArenaExpertSource: cannot open " + path; return false; }
        const uint64_t got = (uint64_t) f.tellg();
        if (got != want) {
            char buf[400];
            std::snprintf(buf, sizeof buf,
                          "ArenaExpertSource: %s is %llu B but %lld layers x %lld experts (blobs up to %lld B) "
                          "make %llu B - this is not the pack this geometry came from",
                          path.c_str(), (unsigned long long) got, (long long) n_layers, (long long) n_expert,
                          (long long) blob, (unsigned long long) want);
            err = buf;
            return false;
        }
    }

    // one layer per registration slice, so no expert straddles two registrations.  The arena is one blob
    // longer than the file: a copy of a whole VRAM slot (the largest blob) may then start at any expert.
    std::vector<uint64_t> bounds, loff, lbytes;
    for (int64_t l = 0; l < n_layers; ++l) {
        bounds.push_back(lay.layer_offset(l));
        loff.push_back(lay.layer_offset(l));
        lbytes.push_back(lay.blob_bytes(l) * (uint64_t) n_expert);
    }
    bounds.push_back(want);
    // Unbuffered straight into the arena when the pack's ranges are sector-aligned, with the per-layer CUDA
    // registration running on a thread ahead of the readers (STRATA_BUFFERED_LOAD=1: the buffered reader after
    // a whole-arena registration, the A/B arm). 16 readers keep a PCIe 5 drive's queue full.
    // The GGUF branch (a native pack without experts.bin, setup's default) the same way: its chunks' windows read
    // unbuffered into a buffer and scattered into the blobs (#230), the registration on the thread ahead of them.
    const bool try_direct = max_pinned_bytes == 0 && std::getenv("STRATA_BUFFERED_LOAD") == nullptr;
    const auto t_reserve = std::chrono::steady_clock::now();
    PinnedArena* a = try_direct ? new PinnedArena(want + (uint64_t) blob, bounds, PinnedArena::Deferred{})
                                : new PinnedArena(want + (uint64_t) blob, bounds, max_pinned_bytes);
    if (!a->valid()) {
        delete a;
        err = "ArenaExpertSource: the arena could not be reserved (" + std::to_string(want) + " B)";
        return false;
    }
    const double reserve_s = std::chrono::duration<double>(std::chrono::steady_clock::now() - t_reserve).count();
    LoadStats st;
    bool direct = false;
    if (try_direct) {
        std::atomic<int> ready{0};
        std::thread reg([&] { a->register_slices(ready); });
        // The readers wait for each slice's registration EVEN ON LARGE PAGES: reading while cudaHostRegister runs on
        // the same slice corrupted the arena (measured 2026-09-30, WDDM, 2 MB pages: STRATA_VERIFY_ARENA gave a
        // different wrong checksum on every run, 11.8 vs 10.2 GiB/s). The ~0.9 s it would save is not available.
        st = from_gguf ? load_experts_gguf(gguf_, a->data(), lay, /*threads=*/16, &ready, /*unbuffered=*/true)
                       : load_experts_direct(path, a->data(), loff, lbytes, /*threads=*/16, /*chunk=*/8u << 20, &ready);
        if (!st.ok) ready.store((int) n_layers);        // a refused read: let the registration thread finish
        reg.join();
#if defined(_WIN32)
        direct = st.ok;
#else
        direct = st.ok && !from_gguf;   // off Windows the GGUF branch reads buffered
#endif
        if (!st.ok && (from_gguf || !st.error.empty())) {
            delete a;
            err = "ArenaExpertSource: the unbuffered load failed: " + st.error;
            return false;
        }
        if (!st.ok) {   // not sector-aligned: registered already, so the buffered reader fills it as before
            st = load_experts_ranges(path, a->data(), loff, lbytes, threads, /*chunk=*/8u << 20);
        }
    }
    if (!try_direct)
        st = from_gguf ? load_experts_gguf(gguf_, a->data(), lay, threads)
                       : load_experts_ranges(path, a->data(), loff, lbytes, threads, /*chunk=*/8u << 20);
    if (!st.ok) {
        delete a;
        err = "ArenaExpertSource: the expert load was refused: " + (st.error.empty() ? std::string("unknown") : st.error);
        return false;
    }
    if (st.bytes != want) {
        delete a;
        err = "ArenaExpertSource: the load read " + std::to_string(st.bytes) + " B of " + std::to_string(want);
        return false;
    }
    arena_ = a;
    base_ = a->data();
    pinned_bytes_ = a->registered_bytes;
    // plan v0.3 P6: device aliases of the mapped registration, for the PCIe share of the misses
    dev_slice_.clear();
    slice_bytes_ = a->slice_bytes;
    if (a->registered_bytes > 0) {
        std::vector<uint64_t> starts = a->slice_bytes > 0 ? a->slice_starts : std::vector<uint64_t>{0};
        for (uint64_t off : starts) {
            void* d = nullptr;
            if (cudaHostGetDevicePointer(&d, (void*) (base_ + off), 0) != cudaSuccess) {
                (void) cudaGetLastError();
                dev_slice_.clear();
                break;
            }
            dev_slice_.push_back((const uint8_t*) d);
        }
    }
    blobs_ = n_layers * n_expert;
    n_expert_ = n_expert;
    reads_ = 0;
    // STRATA_VERIFY_ARENA=1: a checksum of the loaded arena (per layer, combined), to compare two loaders
    if (std::getenv("STRATA_VERIFY_ARENA") != nullptr) {
        std::vector<uint64_t> h((size_t) n_layers, 0);
        std::atomic<int64_t> nl{0};
        std::vector<std::thread> ths;
        for (int t = 0; t < 16; ++t)
            ths.emplace_back([&] {
                for (int64_t l; (l = nl.fetch_add(1)) < n_layers;) {
                    const uint64_t* w = (const uint64_t*) (a->data() + loff[(size_t) l]);
                    uint64_t s = 0;
                    for (uint64_t i = 0; i < lbytes[(size_t) l] / 8; ++i) s += w[i] * (2 * i + 1);
                    h[(size_t) l] = s;
                }
            });
        for (auto& t : ths) t.join();
        uint64_t all = 0;
        for (int64_t l = 0; l < n_layers; ++l) all = all * 1099511628211ull + h[(size_t) l];
        std::fprintf(stderr, "strata generate: STRATA_VERIFY_ARENA %016llx (%s load)\n", (unsigned long long) all,
                     direct ? "unbuffered" : "buffered");
    }
    char timing[160];
    std::snprintf(timing, sizeof timing, "; reserved%s in %.1f s, read %s in %.1f s", try_direct ? "" : "+registered",
                  reserve_s, direct ? "unbuffered (registration alongside)" : "buffered", st.seconds);
    note_ = a->note + timing;
    gib_per_s_ = st.gib_per_second();
    load_seconds_ = st.seconds;
    load_read_s_ = st.read_seconds;
    load_copy_s_ = st.copy_seconds;
    return true;
}

void ArenaExpertSource::close() {
    if (arena_ != nullptr) {
        delete (PinnedArena*) arena_;
        arena_ = nullptr;
    }
    base_ = nullptr;
    blobs_ = 0;
    n_expert_ = 0;
}

bool ArenaExpertSource::pinned(int64_t layer, int64_t expert) const {
    if (base_ == nullptr || layer < 0 || expert < 0 || expert >= n_expert_) return false;
    const auto& lay = strata::kernels::cpu::expert_layout();
    return lay.blob_offset(layer, expert) + lay.blob_bytes(layer) <= pinned_bytes_;
}

const uint8_t* ArenaExpertSource::device_alias(int64_t layer, int64_t expert) const {
    if (dev_slice_.empty() || !pinned(layer, expert)) return nullptr;
    const auto& lay = strata::kernels::cpu::expert_layout();
    if (slice_bytes_ == 0) return dev_slice_[0] + lay.blob_offset(layer, expert);
    // one registration slice per layer
    if ((size_t) layer >= dev_slice_.size()) return nullptr;
    return dev_slice_[(size_t) layer] + (uint64_t) expert * lay.blob_bytes(layer);
}

const uint8_t* ArenaExpertSource::blob(int64_t layer, int64_t expert) {
    if (base_ == nullptr) return nullptr;
    if (layer < 0 || expert < 0 || expert >= n_expert_) return nullptr;
    const int64_t idx = layer * n_expert_ + expert;
    if (idx < 0 || idx >= blobs_) return nullptr;
    ++reads_;
    // Pointer arithmetic into resident memory.  No fault, no copy, no mapping - which is the entire point of
    // this class over `FileExpertSource`.
    return base_ + strata::kernels::cpu::expert_layout().blob_offset(layer, expert);
}

uint64_t available_ram_bytes() {
    uint64_t b = 0;
    return available_memory_bytes(b) ? b : 0;
}

uint64_t total_ram_bytes() {
#if defined(_WIN32)
    MEMORYSTATUSEX status{};
    status.dwLength = sizeof(status);
    return GlobalMemoryStatusEx(&status) ? (uint64_t) status.ullTotalPhys : 0;
#else
    const long pages = sysconf(_SC_PHYS_PAGES), page = sysconf(_SC_PAGESIZE);
    return pages > 0 && page > 0 ? (uint64_t) pages * (uint64_t) page : 0;
#endif
}

uint64_t installed_ram_bytes() {
    // STRATA_EMULATE_RAM_GIB (tests): answer as a PC with that much RAM, for the --low-ram default's rule
    if (const char* e = std::getenv("STRATA_EMULATE_RAM_GIB")) return (uint64_t) (std::atof(e) * 1073741824.0);
#if defined(_WIN32)
    ULONGLONG kib = 0;
    if (GetPhysicallyInstalledSystemMemory(&kib) && kib > 0) return (uint64_t) kib * 1024;
#endif
    return total_ram_bytes();
}

// ================================ THE TIERED SOURCE ================================

namespace {
constexpr uint64_t kSector = 4096;   // unbuffered reads: offset, length and buffer 4 KiB-aligned

#if defined(_WIN32)
// One blob, read unbuffered into `dst` (4 KiB-aligned, `off % 4096 + bytes` rounded up to 4 KiB long at least):
// the blob starts at dst + off % 4096.
bool read_blob_window(HANDLE h, uint64_t off, uint64_t bytes, uint8_t* dst) {
    const uint64_t a0 = off / kSector * kSector, a1 = (off + bytes + kSector - 1) / kSector * kSector;
    OVERLAPPED ov{};
    ov.Offset = (DWORD) a0;
    ov.OffsetHigh = (DWORD) (a0 >> 32);
    DWORD got = 0;
    // the window may run past the end of the file: only the blob itself has to arrive
    return ReadFile(h, dst, (DWORD) (a1 - a0), &got, &ov) && (uint64_t) got >= off - a0 + bytes;
}
// UTF-8 path -> UTF-16, as FileExpertSource::open does (CreateFileA would take the ANSI code page)
HANDLE open_path(const std::string& path, DWORD flags) {
    const int wide = MultiByteToWideChar(CP_UTF8, 0, path.c_str(), -1, nullptr, 0);
    if (wide <= 0) return INVALID_HANDLE_VALUE;
    std::vector<wchar_t> w((size_t) wide);
    MultiByteToWideChar(CP_UTF8, 0, path.c_str(), -1, w.data(), wide);
    return CreateFileW(w.data(), GENERIC_READ, FILE_SHARE_READ, nullptr, OPEN_EXISTING, flags, nullptr);
}
HANDLE open_unbuffered(const std::string& path) {
    return open_path(path, FILE_FLAG_NO_BUFFERING | FILE_FLAG_SEQUENTIAL_SCAN);
}
#endif
}  // namespace

TieredExpertSource::~TieredExpertSource() {
    if (std::getenv("STRATA_TIER_STATS") != nullptr)
        std::fprintf(stderr, "strata tier: %lld blob reads, %lld of them from the file; %lld spare slots left\n",
                     (long long) reads_, (long long) file_reads_, (long long) free_.size());
    // STRATA_TIER_VERIFY=1 (tests): every host copy - loaded, or copied back from VRAM - against the file
    if (std::getenv("STRATA_TIER_VERIFY") != nullptr && base_ != nullptr) {
        int64_t n = 0, bad = 0;
        for (int64_t l = 0; l < n_layers_; ++l)
            for (int64_t e = 0; e < n_expert_; ++e) {
                const uint8_t* h = host_[idx(l, e)];
                if (h == nullptr) continue;
                ++n;
                const uint8_t* f = file_.blob(l, e);
                if (f == nullptr || std::memcmp(h, f, strata::kernels::cpu::expert_layout().blob_bytes(l)) != 0) ++bad;
            }
        std::fprintf(stderr, "strata tier: STRATA_TIER_VERIFY %lld host copies, %lld differ from the file\n",
                     (long long) n, (long long) bad);
    }
    close();
}

void TieredExpertSource::close() {
    {
        std::lock_guard<std::mutex> g(jobs_mu_);
        readers_quit_ = true;
    }
    jobs_cv_.notify_all();
    for (auto& t : readers_) t.join();
    readers_.clear();
    readers_quit_ = false;
    if (scratch_ != nullptr) cudaFreeHost(scratch_);
    scratch_ = nullptr;
    delete (PinnedArena*) arena_;
    arena_ = nullptr;
    base_ = nullptr;
    file_.close();
    host_.clear();
    slot_of_.clear();
    pending_.clear();
    free_.clear();
    slice_start_.clear();
    dev_slice_.clear();
    registered_ = false;
    slots_ = members_ = 0;
}

bool TieredExpertSource::open(const std::string& pack_dir, int64_t n_layers, int64_t n_expert, std::string& err) {
    close();
    if (!file_.open(pack_dir, n_layers, n_expert, err)) return false;
    path_ = pack_dir + "/experts.bin";
    n_layers_ = n_layers;
    n_expert_ = n_expert;
    const size_t n = (size_t) n_layers * (size_t) n_expert;
    host_.assign(n, nullptr);
    slot_of_.assign(n, -1);
    pending_.assign(n, -1);
    const auto& lay = strata::kernels::cpu::expert_layout();
    stride_ = ((uint64_t) lay.max_blob + 2 * kSector + kSector - 1) / kSector * kSector;
    return true;
}

uint64_t TieredExpertSource::file_offset(int64_t layer, int64_t expert) const {
    return strata::kernels::cpu::expert_layout().blob_offset(layer, expert);
}

bool TieredExpertSource::load(const std::vector<std::pair<int32_t, int32_t>>& order, uint64_t budget_bytes,
                              int64_t spare, int threads, std::string& err) {
#if defined(_WIN32)
    if (!file_.mapped()) { err = "TieredExpertSource: open the pack first"; return false; }
    if (arena_ != nullptr) { err = "TieredExpertSource: the tier is loaded already"; return false; }
    const auto t0 = std::chrono::steady_clock::now();
    const int64_t fit = (int64_t) (budget_bytes / stride_);
    spare = std::max<int64_t>(0, std::min(spare, fit));
    // the members: `order` as far as the budget goes, each pair once
    std::vector<std::pair<int32_t, int32_t>> mem;
    std::vector<uint8_t> seen(host_.size(), 0);
    for (const auto& p : order) {
        if ((int64_t) mem.size() >= fit - spare) break;
        if (p.first < 0 || p.second < 0 || p.first >= n_layers_ || p.second >= n_expert_) continue;
        const size_t i = idx(p.first, p.second);
        if (seen[i]) continue;
        seen[i] = 1;
        mem.push_back(p);
    }
    members_ = (int64_t) mem.size();
    slots_ = members_ + spare;
    if (slots_ == 0) { note_ = "empty: every expert comes from the file"; return true; }
    // in file order, so the reads walk the drive forward; slot i holds mem[i]
    std::sort(mem.begin(), mem.end(), [&](const auto& a, const auto& b) {
        return file_offset(a.first, a.second) < file_offset(b.first, b.second);
    });
    // registration slices of 256 slots: the readers wait for a slice's registration before touching it (reading
    // while cudaHostRegister runs on the same pages corrupted the resident arena, see ArenaExpertSource::open)
    std::vector<uint64_t> bounds;
    for (int64_t s = 0; s < slots_; s += 256) bounds.push_back((uint64_t) s * stride_);
    bounds.push_back((uint64_t) slots_ * stride_);
    PinnedArena* a = new PinnedArena((uint64_t) slots_ * stride_, bounds, PinnedArena::Deferred{});
    if (!a->valid()) {
        delete a;
        char m[160];
        std::snprintf(m, sizeof m, "TieredExpertSource: cannot reserve the %.2f GiB host tier",
                      (double) slots_ * (double) stride_ / 1073741824.0);
        err = m;
        return false;
    }
    std::atomic<int> ready{0};
    std::thread reg([&] { a->register_slices(ready); });
    std::atomic<int64_t> next{0};
    std::atomic<bool> bad{false};
    std::string bad_msg;
    std::mutex bad_mu;
    const auto& lay = strata::kernels::cpu::expert_layout();
    auto worker = [&] {
        HANDLE h = open_unbuffered(path_);
        if (h == INVALID_HANDLE_VALUE) {
            std::lock_guard<std::mutex> g(bad_mu);
            bad_msg = "cannot open " + path_ + " unbuffered";
            bad = true;
            return;
        }
        for (int64_t s; !bad && (s = next.fetch_add(1)) < members_;) {
            while (ready.load(std::memory_order_acquire) <= (int) (s / 256)) std::this_thread::yield();
            const auto& p = mem[(size_t) s];
            const uint64_t off = file_offset(p.first, p.second);
            uint8_t* dst = a->data() + (uint64_t) s * stride_;
            if (!read_blob_window(h, off, lay.blob_bytes(p.first), dst)) {
                std::lock_guard<std::mutex> g(bad_mu);
                bad_msg = "short unbuffered read at offset " + std::to_string(off);
                bad = true;
                break;
            }
            host_[idx(p.first, p.second)] = dst + off % kSector;
            slot_of_[idx(p.first, p.second)] = (int32_t) s;
        }
        CloseHandle(h);
    };
    std::vector<std::thread> pool;
    for (int t = 1; t < std::max(1, threads); ++t) pool.emplace_back(worker);
    worker();
    for (auto& t : pool) t.join();
    if (bad) ready.store((int) bounds.size());
    reg.join();
    if (bad) {
        delete a;
        std::fill(host_.begin(), host_.end(), nullptr);
        std::fill(slot_of_.begin(), slot_of_.end(), -1);
        err = "TieredExpertSource: " + bad_msg;
        return false;
    }
    arena_ = a;
    base_ = a->data();
    for (int64_t s = members_; s < slots_; ++s) free_.push_back((int32_t) s);
    if (!rank_.empty()) {   // the members, coldest on top: what a demotion evicts once the spares are gone
        for (const auto& p : mem) {
            const size_t i = idx(p.first, p.second);
            heap_.emplace_back(i < rank_.size() ? rank_[i] : INT32_MAX, (int32_t) i);
        }
        std::make_heap(heap_.begin(), heap_.end());
    }
    // the decode prefetch: pinned buffers and eight readers (an NVMe wants several reads in flight)
    if (cudaHostAlloc((void**) &scratch_, (size_t) kScratch * stride_, cudaHostAllocPortable) == cudaSuccess) {
        scratch_of_.assign(host_.size(), -1);
        scratch_pair_.assign(kScratch, -1);
        scratch_pad_.assign(kScratch, 0);
        scratch_state_.reset(new std::atomic<int>[kScratch]);
        for (int b = 0; b < kScratch; ++b) scratch_state_[b].store(1);
        for (int t = 0; t < 8; ++t) readers_.emplace_back([this] { reader_loop(); });
    } else {
        (void) cudaGetLastError();
        scratch_ = nullptr;
    }
    // device aliases of the registered slices (the PCIe share can read the tier directly)
    registered_ = a->registered_bytes >= (uint64_t) slots_ * stride_;
    slice_start_ = a->slice_starts;
    for (uint64_t off : slice_start_) {
        void* d = nullptr;
        if (cudaHostGetDevicePointer(&d, (void*) (base_ + off), 0) != cudaSuccess) {
            (void) cudaGetLastError();
            dev_slice_.clear();
            registered_ = false;
            break;
        }
        dev_slice_.push_back((const uint8_t*) d);
    }
    load_s_ = std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count();
    char m[256];
    std::snprintf(m, sizeof m, "%lld experts + %lld spare slots, %.2f GiB, read in %.1f s (%.1f GiB/s); %s",
                  (long long) members_, (long long) spare, (double) tier_bytes() / 1073741824.0, load_s_,
                  load_s_ > 0 ? (double) members_ * (double) lay.max_blob / 1073741824.0 / load_s_ : 0.0,
                  a->note.c_str());
    note_ = m;
    return true;
#else
    (void) order; (void) budget_bytes; (void) spare; (void) threads;
    err = "TieredExpertSource: the low-RAM tier is Windows-only for now";
    return false;
#endif
}

bool TieredExpertSource::stream(const std::vector<std::pair<int32_t, int32_t>>& pairs,
                                const std::function<bool(size_t, const uint8_t*)>& sink, int threads,
                                std::string& err) {
#if defined(_WIN32)
    if (pairs.empty()) return true;
    threads = std::max(1, threads);
    const int ring = 4 * threads;
    uint8_t* buf = nullptr;
    if (cudaHostAlloc((void**) &buf, (size_t) ring * stride_, cudaHostAllocPortable) != cudaSuccess) {
        (void) cudaGetLastError();
        err = "TieredExpertSource: cannot allocate the pinned read buffers";
        return false;
    }
    const auto& lay = strata::kernels::cpu::expert_layout();
    std::vector<HANDLE> hs;
    for (int t = 0; t < threads; ++t) {
        HANDLE h = open_unbuffered(path_);
        if (h == INVALID_HANDLE_VALUE) break;
        hs.push_back(h);
    }
    bool ok = !hs.empty();
    if (!ok) err = "TieredExpertSource: cannot open " + path_ + " unbuffered";
    // a pipeline: the readers run up to `ring` blobs ahead of the sink, which takes them in order
    const size_t n = pairs.size();
    std::vector<std::atomic<uint8_t>> ready(n);
    for (auto& r : ready) r.store(0, std::memory_order_relaxed);
    std::atomic<size_t> next{0}, consumed{0};
    std::atomic<bool> bad{false};
    std::vector<std::thread> th;
    for (size_t t = 0; ok && t < hs.size(); ++t)
        th.emplace_back([&, t] {
            for (size_t j; !bad && (j = next.fetch_add(1)) < n;) {
                while (!bad && consumed.load(std::memory_order_acquire) + (size_t) ring <= j) std::this_thread::yield();
                if (bad) break;
                const auto& p = pairs[j];
                if (!read_blob_window(hs[t], file_offset(p.first, p.second), lay.blob_bytes(p.first),
                                      buf + (j % (size_t) ring) * stride_)) {
                    bad = true;
                    break;
                }
                ready[j].store(1, std::memory_order_release);
            }
        });
    for (size_t j = 0; ok && j < n; ++j) {
        while (!bad && !ready[j].load(std::memory_order_acquire)) std::this_thread::yield();
        if (bad) { err = "TieredExpertSource: a streamed read failed"; ok = false; break; }
        const auto& p = pairs[j];
        ok = sink(j, buf + (j % (size_t) ring) * stride_ + file_offset(p.first, p.second) % kSector);
        consumed.store(j + 1, std::memory_order_release);
    }
    if (!ok) bad = true;   // a failed sink stops the readers too
    for (auto& x : th) x.join();
    for (HANDLE h : hs) CloseHandle(h);
    cudaFreeHost(buf);
    return ok;
#else
    (void) pairs; (void) sink; (void) threads;
    err = "TieredExpertSource: streaming is Windows-only for now";
    return false;
#endif
}

void TieredExpertSource::reader_loop() {
#if defined(_WIN32)
    // buffered, unlike the bulk reads: the few cold experts a token needs are read again and again, and the OS
    // cache keeps them in RAM nobody else wants (standby pages, handed back on demand) - not in this process
    HANDLE h = open_path(path_, FILE_ATTRIBUTE_NORMAL);   // no RANDOM_ACCESS: it lets the cache drop them (open())
    const auto& lay = strata::kernels::cpu::expert_layout();
    for (;;) {
        int b = -1;
        {
            std::unique_lock<std::mutex> g(jobs_mu_);
            jobs_cv_.wait(g, [&] { return readers_quit_ || !jobs_.empty(); });
            if (readers_quit_) break;
            b = jobs_.back();
            jobs_.pop_back();
        }
        const int32_t i = scratch_pair_[(size_t) b];
        const int64_t l = i / n_expert_, e = i % n_expert_;
        const uint64_t off = file_offset(l, e);
        OVERLAPPED ov{};
        ov.Offset = (DWORD) off;
        ov.OffsetHigh = (DWORD) (off >> 32);
        DWORD got = 0;
        const DWORD n = (DWORD) lay.blob_bytes(l);
        const bool ok = h != INVALID_HANDLE_VALUE && ReadFile(h, scratch_ + (size_t) b * stride_, n, &got, &ov) && got == n;
        scratch_pad_[(size_t) b] = 0;
        scratch_state_[b].store(ok ? 1 : -1, std::memory_order_release);
        scratch_busy_.fetch_sub(1, std::memory_order_acq_rel);
    }
    if (h != INVALID_HANDLE_VALUE) CloseHandle(h);
#endif
}

const uint8_t* TieredExpertSource::stable_blob(int64_t layer, int64_t expert) {
    if (layer < 0 || expert < 0 || layer >= n_layers_ || expert >= n_expert_) return nullptr;
    const uint8_t* h = host_[idx(layer, expert)];
    return h != nullptr ? h : file_.blob(layer, expert);
}

const uint8_t* TieredExpertSource::blob(int64_t layer, int64_t expert) {
    if (layer < 0 || expert < 0 || layer >= n_layers_ || expert >= n_expert_) return nullptr;
    ++reads_;
    const size_t pi = idx(layer, expert);
    const uint8_t* h = host_[pi];
    if (h != nullptr) return h;
    ++file_reads_;
    if (scratch_ != nullptr && scratch_of_[pi] >= 0) {   // prefetched at begin_layer: wait for its read
        const int b = scratch_of_[pi];
        int s;
        while ((s = scratch_state_[b].load(std::memory_order_acquire)) == 0) std::this_thread::yield();
        if (s == 1) return scratch_ + (size_t) b * stride_ + scratch_pad_[(size_t) b];
    }
    const uint8_t* f = file_.blob(layer, expert);
    if (f != nullptr) {   // its mapped pages leave the working set at the next layer (begin_layer)
        std::lock_guard<std::mutex> g(handed_mu_);
        handed_.emplace_back(f, (size_t) strata::kernels::cpu::expert_layout().blob_bytes(layer));
    }
    return f;
}

void TieredExpertSource::begin_layer(int64_t layer, const int32_t* ids, int64_t k) {
    (void) layer; (void) ids; (void) k;
    // The file-backed blobs handed out for the previous layer are computed by now: their pages go back to the
    // standby list (VirtualUnlock on unlocked pages drops them from the working set), so reading the experts the
    // tier could not hold never grows the process until Windows starts trimming everything else.
    std::lock_guard<std::mutex> g(handed_mu_);
#if defined(_WIN32)
    for (const auto& [p, n] : handed_) {
        const uintptr_t a = (uintptr_t) p / kSector * kSector;
        VirtualUnlock((LPVOID) a, (SIZE_T) ((uintptr_t) p + n - a));
    }
#endif
    handed_.clear();
    if (scratch_ == nullptr || ids == nullptr) return;
    // the previous layer is computed: its buffers are free once no read is still landing in them
    while (scratch_busy_.load(std::memory_order_acquire) > 0) std::this_thread::yield();
    for (int b = 0; b < kScratch; ++b)
        if (scratch_pair_[(size_t) b] >= 0) {
            scratch_of_[(size_t) scratch_pair_[(size_t) b]] = -1;
            scratch_pair_[(size_t) b] = -1;
        }
    if (layer < 0 || layer >= n_layers_) return;
    int used = 0;
    {
        std::lock_guard<std::mutex> g2(jobs_mu_);
        for (int64_t j = 0; j < k && used < kScratch; ++j) {
            const int32_t e = ids[j];
            if (e < 0 || e >= n_expert_) continue;
            const size_t pi = idx(layer, e);
            if (host_[pi] != nullptr || scratch_of_[pi] >= 0) continue;   // in the tier, or already reading
            if (res_ != nullptr && res_[pi] >= 0) continue;                // in VRAM: the GPU computes it
            scratch_of_[pi] = (int16_t) used;
            scratch_pair_[(size_t) used] = (int32_t) pi;
            scratch_state_[used].store(0, std::memory_order_relaxed);
            scratch_busy_.fetch_add(1, std::memory_order_acq_rel);
            jobs_.push_back(used);
            ++used;
        }
    }
    if (used > 0) jobs_cv_.notify_all();
}

bool TieredExpertSource::read_blob(int64_t layer, int64_t expert, uint8_t* dst, size_t cap, size_t& pad) {
#if defined(_WIN32)
    if (layer < 0 || expert < 0 || layer >= n_layers_ || expert >= n_expert_ || host_[idx(layer, expert)] != nullptr ||
        cap < stride_ || ((uintptr_t) dst % kSector) != 0)
        return false;
    // one unbuffered handle per thread (the stager's and the refill's threads call this concurrently)
    thread_local HANDLE h = INVALID_HANDLE_VALUE;
    thread_local std::string h_path;
    if (h == INVALID_HANDLE_VALUE || h_path != path_) {
        if (h != INVALID_HANDLE_VALUE) CloseHandle(h);
        h = open_unbuffered(path_);
        h_path = path_;
        if (h == INVALID_HANDLE_VALUE) return false;
    }
    const uint64_t off = file_offset(layer, expert);
    if (!read_blob_window(h, off, strata::kernels::cpu::expert_layout().blob_bytes(layer), dst)) return false;
    pad = (size_t) (off % kSector);
    ++reads_;
    ++file_reads_;
    return true;
#else
    (void) layer; (void) expert; (void) dst; (void) cap; (void) pad;
    return false;
#endif
}

bool TieredExpertSource::pinned(int64_t layer, int64_t expert) const {
    if (!registered_ || layer < 0 || expert < 0 || layer >= n_layers_ || expert >= n_expert_) return false;
    return host_[idx(layer, expert)] != nullptr;
}

const uint8_t* TieredExpertSource::device_alias(int64_t layer, int64_t expert) const {
    if (!pinned(layer, expert) || dev_slice_.empty()) return nullptr;
    const uint64_t off = (uint64_t) (host_[idx(layer, expert)] - base_);
    // the slice holding it: slices start at multiples of 256 slots
    const size_t s = (size_t) (off / (256 * stride_));
    if (s >= dev_slice_.size()) return nullptr;
    return dev_slice_[s] + (off - slice_start_[s]);
}

bool TieredExpertSource::has_copy(int64_t layer, int64_t expert) const {
    return layer >= 0 && expert >= 0 && layer < n_layers_ && expert < n_expert_ && host_[idx(layer, expert)] != nullptr;
}

uint8_t* TieredExpertSource::demote_begin(int64_t layer, int64_t expert) {
    if (base_ == nullptr || has_copy(layer, expert)) return nullptr;
    const size_t i = idx(layer, expert);
    if (pending_[i] >= 0) return nullptr;
    const int32_t yr = i < rank_.size() ? rank_[i] : INT32_MAX;
    // no spare: the member ranked below this expert gives its slot up (it is in the file too)
    while (free_.empty() && !heap_.empty()) {
        const auto [zr, z] = heap_.front();
        const bool stale = slot_of_[(size_t) z] < 0 || host_[(size_t) z] == nullptr ||
                           (i < rank_.size() && (size_t) z < rank_.size() && zr != rank_[(size_t) z]);
        if (!stale && zr <= yr) break;   // every member is ranked above it: it stays file-backed
        std::pop_heap(heap_.begin(), heap_.end());
        heap_.pop_back();
        if (stale) continue;
        free_.push_back(slot_of_[(size_t) z]);
        host_[(size_t) z] = nullptr;
        slot_of_[(size_t) z] = -1;
    }
    if (free_.empty()) return nullptr;
    const int32_t s = free_.back();
    free_.pop_back();
    pending_[i] = s;
    return base_ + (uint64_t) s * stride_;
}

void TieredExpertSource::demote_commit(int64_t layer, int64_t expert) {
    if (layer < 0 || expert < 0 || layer >= n_layers_ || expert >= n_expert_) return;
    const size_t i = idx(layer, expert);
    if (pending_[i] < 0) return;
    slot_of_[i] = pending_[i];
    host_[i] = base_ + (uint64_t) pending_[i] * stride_;
    pending_[i] = -1;
    if (!rank_.empty()) {
        heap_.emplace_back(i < rank_.size() ? rank_[i] : INT32_MAX, (int32_t) i);
        std::push_heap(heap_.begin(), heap_.end());
        // promotions leave entries behind: rebuilt from the members once they are half of it
        if (heap_.size() > 2 * (size_t) std::max<int64_t>(members_, 64)) {
            heap_.clear();
            for (size_t p = 0; p < host_.size(); ++p)
                if (slot_of_[p] >= 0 && host_[p] != nullptr) heap_.emplace_back(p < rank_.size() ? rank_[p] : INT32_MAX, (int32_t) p);
            std::make_heap(heap_.begin(), heap_.end());
        }
    }
}

void TieredExpertSource::demote_abort(int64_t layer, int64_t expert) {
    if (layer < 0 || expert < 0 || layer >= n_layers_ || expert >= n_expert_) return;
    const size_t i = idx(layer, expert);
    if (pending_[i] < 0) return;
    free_.push_back(pending_[i]);
    pending_[i] = -1;
}

void TieredExpertSource::promote_done(int64_t layer, int64_t expert) {
    if (layer < 0 || expert < 0 || layer >= n_layers_ || expert >= n_expert_) return;
    const size_t i = idx(layer, expert);
    if (slot_of_[i] < 0) return;
    host_[i] = nullptr;
    free_.push_back(slot_of_[i]);
    slot_of_[i] = -1;
}

}  // namespace strata::core
