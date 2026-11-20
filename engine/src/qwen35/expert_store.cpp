#include "strata/qwen35/expert_store.hpp"
#include "strata/qwen35/bounded_experts.hpp"

#include "strata/qwen35/startup_trace.hpp"
#include "strata/qwen35/phase_trace.hpp"
#include <algorithm>
#include <chrono>
#include <cstdlib>
#include <cwchar>
#include <array>
#include <cstdio>
#include <cstring>
#include <limits>
#include <memory>
#include <mutex>
#include <vector>

#if defined(_WIN32)
#ifndef NOMINMAX
#define NOMINMAX
#endif
#include <windows.h>
#include <winioctl.h>
#include <wincrypt.h>
#include <dpapi.h>
#include <bcrypt.h>
#else
#include <errno.h>
#include <fcntl.h>
#include <signal.h>
#include <sys/mman.h>
#include <sys/stat.h>
#include <sys/statvfs.h>
#include <unistd.h>
#endif

namespace strata::qwen35 {
bool q36_require_host_headroom(uint64_t additional, std::string& err) {
#if defined(_WIN32)
    MEMORYSTATUSEX status{};
    status.dwLength = sizeof(status);
    if (!GlobalMemoryStatusEx(&status)) {
        err = "Q36 bounded expert memory observation failed";
        return false;
    }
    if (q36_host_headroom(additional, status.ullAvailPhys, status.ullAvailPageFile)) return true;
    err = "Q36 bounded expert admission refused: additional=" + std::to_string(additional) +
          " ram_available=" + std::to_string(status.ullAvailPhys) +
          " commit_available=" + std::to_string(status.ullAvailPageFile) +
          " bytes; requires 3 GiB RAM and 4 GiB commit remaining";
#else
    (void) additional;
    err = "Q36 bounded expert admission is Windows-only";
#endif
    return false;
}

void q36_startup_physical_memory(const char* stage) {
    if (!q36_startup_detail_enabled()) return;
    uint64_t free_bytes = 0, total_bytes = 0;
#if defined(_WIN32)
    MEMORYSTATUSEX status{};
    status.dwLength = sizeof(status);
    if (!GlobalMemoryStatusEx(&status)) return;
    free_bytes = status.ullAvailPhys; total_bytes = status.ullTotalPhys;
#elif defined(__linux__)
    std::FILE* file = std::fopen("/proc/meminfo", "r");
    if (!file) return;
    char line[256];
    unsigned long long kib = 0;
    while (std::fgets(line, sizeof(line), file)) {
        if (std::sscanf(line, "MemTotal: %llu kB", &kib) == 1) total_bytes = kib * 1024;
        if (std::sscanf(line, "MemAvailable: %llu kB", &kib) == 1) free_bytes = kib * 1024;
    }
    std::fclose(file);
    if (!total_bytes) return;
#else
    return;
#endif
    std::fprintf(stderr, "Q36_EFFECTIVE schema=1 generation=%s memory_stage=%s ram_free_bytes=%llu ram_total_bytes=%llu\n",
                 q36_startup_generation(), stage, (unsigned long long) free_bytes, (unsigned long long) total_bytes);
}
namespace {

uint64_t g_scratch_bytes = 0;
bool g_verification_cached = false;

void put_u32(uint8_t* p, uint32_t v) {
    p[0] = (uint8_t) v;
    p[1] = (uint8_t) (v >> 8);
    p[2] = (uint8_t) (v >> 16);
    p[3] = (uint8_t) (v >> 24);
}

void put_u64(uint8_t* p, uint64_t v) {
    for (int i = 0; i < 8; ++i) p[i] = (uint8_t) (v >> (8 * i));
}

bool read_u32(const uint8_t* base, uint64_t size, uint64_t at, uint32_t& out) {
    if (at > size || size - at < 4) return false;
    out = (uint32_t) base[at] | ((uint32_t) base[at + 1] << 8) | ((uint32_t) base[at + 2] << 16) |
          ((uint32_t) base[at + 3] << 24);
    return true;
}

bool read_u64(const uint8_t* base, uint64_t size, uint64_t at, uint64_t& out) {
    if (at > size || size - at < 8) return false;
    out = 0;
    for (int i = 0; i < 8; ++i) out |= (uint64_t) base[at + i] << (8 * i);
    return true;
}

uint32_t crc32_update(uint32_t crc, const uint8_t* p, size_t n) {
    // Slicing by eight preserves the IEEE polynomial and streaming state of
    // existing packs, while breaking the byte-at-a-time dependency chain.
    // Explicit little-endian loads also work on unaligned and big-endian hosts.
    static const std::array<std::array<uint32_t, 256>, 8> table = [] {
        std::array<std::array<uint32_t, 256>, 8> values{};
        for (uint32_t i = 0; i < values[0].size(); ++i) {
            uint32_t v = i;
            for (int b = 0; b < 8; ++b) v = (v >> 1) ^ (0xEDB88320u & (0u - (v & 1u)));
            values[0][i] = v;
        }
        for (size_t slice = 1; slice < values.size(); ++slice)
            for (size_t i = 0; i < values[slice].size(); ++i) {
                const uint32_t v = values[slice - 1][i];
                values[slice][i] = (v >> 8) ^ values[0][v & 0xffu];
            }
        return values;
    }();
    crc = ~crc;
    while (n >= 8) {
        const uint32_t first =
            crc ^ (uint32_t) p[0] ^ ((uint32_t) p[1] << 8) ^ ((uint32_t) p[2] << 16) ^ ((uint32_t) p[3] << 24);
        crc = table[7][first & 0xffu] ^ table[6][(first >> 8) & 0xffu] ^ table[5][(first >> 16) & 0xffu] ^
              table[4][first >> 24] ^ table[3][p[4]] ^ table[2][p[5]] ^ table[1][p[6]] ^ table[0][p[7]];
        p += 8;
        n -= 8;
    }
    while (n--) crc = table[0][(crc ^ *p++) & 0xffu] ^ (crc >> 8);
    return ~crc;
}

struct PortableSha256 {
    uint32_t h[8] = {0x6a09e667u, 0xbb67ae85u, 0x3c6ef372u, 0xa54ff53au,
                     0x510e527fu, 0x9b05688cu, 0x1f83d9abu, 0x5be0cd19u};
    uint64_t bits = 0;
    uint8_t buf[64]{};
    size_t n = 0;
    bool bad = false;

    static uint32_t rotr(uint32_t x, uint32_t r) { return (x >> r) | (x << (32 - r)); }

    void block(const uint8_t* p) {
        static const uint32_t k[64] = {
            0x428a2f98u, 0x71374491u, 0xb5c0fbcfu, 0xe9b5dba5u, 0x3956c25bu, 0x59f111f1u, 0x923f82a4u, 0xab1c5ed5u,
            0xd807aa98u, 0x12835b01u, 0x243185beu, 0x550c7dc3u, 0x72be5d74u, 0x80deb1feu, 0x9bdc06a7u, 0xc19bf174u,
            0xe49b69c1u, 0xefbe4786u, 0x0fc19dc6u, 0x240ca1ccu, 0x2de92c6fu, 0x4a7484aau, 0x5cb0a9dcu, 0x76f988dau,
            0x983e5152u, 0xa831c66du, 0xb00327c8u, 0xbf597fc7u, 0xc6e00bf3u, 0xd5a79147u, 0x06ca6351u, 0x14292967u,
            0x27b70a85u, 0x2e1b2138u, 0x4d2c6dfcu, 0x53380d13u, 0x650a7354u, 0x766a0abbu, 0x81c2c92eu, 0x92722c85u,
            0xa2bfe8a1u, 0xa81a664bu, 0xc24b8b70u, 0xc76c51a3u, 0xd192e819u, 0xd6990624u, 0xf40e3585u, 0x106aa070u,
            0x19a4c116u, 0x1e376c08u, 0x2748774cu, 0x34b0bcb5u, 0x391c0cb3u, 0x4ed8aa4au, 0x5b9cca4fu, 0x682e6ff3u,
            0x748f82eeu, 0x78a5636fu, 0x84c87814u, 0x8cc70208u, 0x90befffau, 0xa4506cebu, 0xbef9a3f7u, 0xc67178f2u};
        uint32_t w[64];
        for (int i = 0; i < 16; ++i) {
            w[i] = ((uint32_t) p[i * 4] << 24) | ((uint32_t) p[i * 4 + 1] << 16) | ((uint32_t) p[i * 4 + 2] << 8) |
                   (uint32_t) p[i * 4 + 3];
        }
        for (int i = 16; i < 64; ++i) {
            const uint32_t s0 = rotr(w[i - 15], 7) ^ rotr(w[i - 15], 18) ^ (w[i - 15] >> 3);
            const uint32_t s1 = rotr(w[i - 2], 17) ^ rotr(w[i - 2], 19) ^ (w[i - 2] >> 10);
            w[i] = w[i - 16] + s0 + w[i - 7] + s1;
        }
        uint32_t a = h[0], b = h[1], c = h[2], d = h[3], e = h[4], f = h[5], g = h[6], hh = h[7];
        for (int i = 0; i < 64; ++i) {
            const uint32_t s1 = rotr(e, 6) ^ rotr(e, 11) ^ rotr(e, 25);
            const uint32_t ch = (e & f) ^ (~e & g);
            const uint32_t t1 = hh + s1 + ch + k[i] + w[i];
            const uint32_t s0 = rotr(a, 2) ^ rotr(a, 13) ^ rotr(a, 22);
            const uint32_t maj = (a & b) ^ (a & c) ^ (b & c);
            const uint32_t t2 = s0 + maj;
            hh = g;
            g = f;
            f = e;
            e = d + t1;
            d = c;
            c = b;
            b = a;
            a = t1 + t2;
        }
        h[0] += a;
        h[1] += b;
        h[2] += c;
        h[3] += d;
        h[4] += e;
        h[5] += f;
        h[6] += g;
        h[7] += hh;
    }

    void update(const void* data, size_t len) {
        if (bad) return;
        if (len > std::numeric_limits<uint64_t>::max() / 8 || bits > std::numeric_limits<uint64_t>::max() - (uint64_t) len * 8) {
            bad = true;
            return;
        }
        bits += (uint64_t) len * 8;
        const auto* p = static_cast<const uint8_t*>(data);
        while (len) {
            const size_t take = std::min(len, (size_t) 64 - n);
            std::memcpy(buf + n, p, take);
            n += take;
            p += take;
            len -= take;
            if (n == 64) {
                block(buf);
                n = 0;
            }
        }
    }

    bool final(uint8_t out[32]) {
        if (bad) return false;
        const uint64_t bitlen = bits;
        buf[n++] = 0x80;
        if (n > 56) {
            while (n < 64) buf[n++] = 0;
            block(buf);
            n = 0;
        }
        while (n < 56) buf[n++] = 0;
        for (int i = 7; i >= 0; --i) buf[n++] = (uint8_t) (bitlen >> (i * 8));
        block(buf);
        for (int i = 0; i < 8; ++i) {
            out[i * 4] = (uint8_t) (h[i] >> 24);
            out[i * 4 + 1] = (uint8_t) (h[i] >> 16);
            out[i * 4 + 2] = (uint8_t) (h[i] >> 8);
            out[i * 4 + 3] = (uint8_t) h[i];
        }
        return true;
    }
};

#if defined(_WIN32)
// Same SHA-256 and full-file coverage as the portable implementation. CNG can
// use the CPU's accelerated hash implementation; no digest/CRC check is skipped.
struct Sha256 {
    BCRYPT_ALG_HANDLE algorithm = nullptr;
    BCRYPT_HASH_HANDLE hash = nullptr;
    std::vector<uint8_t> object;
    uint64_t bytes = 0;
    bool bad = false;

    Sha256() {
        DWORD object_size = 0, hash_size = 0, got = 0;
        if (BCryptOpenAlgorithmProvider(&algorithm, BCRYPT_SHA256_ALGORITHM, nullptr, 0) < 0 ||
            BCryptGetProperty(algorithm, BCRYPT_OBJECT_LENGTH, (PUCHAR) &object_size, sizeof(object_size), &got, 0) < 0 ||
            BCryptGetProperty(algorithm, BCRYPT_HASH_LENGTH, (PUCHAR) &hash_size, sizeof(hash_size), &got, 0) < 0 ||
            hash_size != 32 || object_size == 0) { bad = true; return; }
        object.resize(object_size);
        if (BCryptCreateHash(algorithm, &hash, object.data(), object_size, nullptr, 0, 0) < 0) bad = true;
    }
    Sha256(const Sha256&) = delete;
    Sha256& operator=(const Sha256&) = delete;
    ~Sha256() {
        if (hash) BCryptDestroyHash(hash);
        if (algorithm) BCryptCloseAlgorithmProvider(algorithm, 0);
    }
    void update(const void* data, size_t len) {
        if (bad) return;
        if (len > std::numeric_limits<uint64_t>::max() / 8 - bytes) { bad = true; return; }
        bytes += len;
        const auto* p = static_cast<const uint8_t*>(data);
        while (len) {
            const ULONG take = (ULONG) std::min<size_t>(len, 1u << 20);
            if (BCryptHashData(hash, const_cast<PUCHAR>(p), take, 0) < 0) { bad = true; return; }
            p += take;
            len -= take;
        }
    }
    bool final(uint8_t out[32]) { return !bad && BCryptFinishHash(hash, out, 32, 0) >= 0; }
};
#else
using Sha256 = PortableSha256;
#endif

std::string to_hex(const uint8_t* p, size_t n) {
    static const char* hexd = "0123456789abcdef";
    std::string s(n * 2, '0');
    for (size_t i = 0; i < n; ++i) {
        s[i * 2] = hexd[p[i] >> 4];
        s[i * 2 + 1] = hexd[p[i] & 0x0f];
    }
    return s;
}

bool layout_ok(const ExpertBlobLayout& layout, const ExpertSource& src, std::string& err) {
    if (layout.n_expert == 0 || layout.stride == 0 || src.src_gsz == 0 || src.src_dsz == 0) {
        err = "expert pack layout is empty";
        return false;
    }
    if (layout.up_off < src.src_gsz || layout.down_off < layout.up_off + src.src_gsz ||
        layout.blob_bytes < layout.down_off + src.src_dsz || layout.stride < layout.blob_bytes) {
        err = "expert pack copies overlap or run past the stride";
        return false;
    }
    if (layout.stride > (std::numeric_limits<uint64_t>::max() - kQ36ExpertTail) / layout.n_expert) {
        err = "expert pack layer size overflows";
        return false;
    }
    return true;
}

std::string pack_path_for(const std::string& model) { return model + ".q36epk"; }
std::string partial_path_for(const std::string& pack) { return pack + ".partial"; }
std::string rejected_path_for(const std::string& pack) { return pack + ".rejected"; }
std::string lock_path_for(const std::string& pack) { return pack + ".lock"; }

#if defined(_WIN32)
std::wstring widen(const std::string& path) {
    if (path.empty()) return {};
    const int n = MultiByteToWideChar(CP_UTF8, MB_ERR_INVALID_CHARS, path.c_str(), -1, nullptr, 0);
    if (n <= 0) return {};
    std::wstring w((size_t) n, L'\0');
    MultiByteToWideChar(CP_UTF8, MB_ERR_INVALID_CHARS, path.c_str(), -1, w.data(), n);
    if (!w.empty() && w.back() == L'\0') w.pop_back();
    return w;
}

bool remove_file(const std::string& path) {
    const std::wstring w = widen(path);
    if (w.empty()) return false;
    return DeleteFileW(w.c_str()) || GetLastError() == ERROR_FILE_NOT_FOUND;
}

bool rename_over(const std::string& from, const std::string& to) {
    const std::wstring a = widen(from), b = widen(to);
    if (a.empty() || b.empty()) return false;
    return MoveFileExW(a.c_str(), b.c_str(), MOVEFILE_REPLACE_EXISTING | MOVEFILE_WRITE_THROUGH) != 0;
}

bool process_alive(uint32_t pid) {
    if (pid == 0) return false;
    HANDLE h = OpenProcess(PROCESS_QUERY_LIMITED_INFORMATION, FALSE, pid);
    if (!h) return GetLastError() == ERROR_ACCESS_DENIED;
    DWORD code = 0;
    const BOOL ok = GetExitCodeProcess(h, &code);
    CloseHandle(h);
    return ok && code == STILL_ACTIVE;
}

int64_t disk_free(const std::string& path) {
    const std::wstring w = widen(path);
    if (w.empty()) return -1;
    std::wstring dir = w;
    const auto slash = dir.find_last_of(L"\\/");
    if (slash == std::wstring::npos) dir = L".";
    else if (slash == 0) dir = L"\\";
    else dir.resize(slash);
    ULARGE_INTEGER freeb{};
    if (!GetDiskFreeSpaceExW(dir.c_str(), &freeb, nullptr, nullptr)) return -1;
    return (int64_t) freeb.QuadPart;
}
#else
bool remove_file(const std::string& path) { return ::unlink(path.c_str()) == 0 || errno == ENOENT; }

bool rename_over(const std::string& from, const std::string& to) { return ::rename(from.c_str(), to.c_str()) == 0; }

bool process_alive(uint32_t pid) {
    if (pid == 0) return false;
    return ::kill((pid_t) pid, 0) == 0 || errno == EPERM;
}

int64_t disk_free(const std::string& path) {
    std::string dir = path;
    const auto slash = dir.find_last_of('/');
    if (slash == std::string::npos) dir = ".";
    else if (slash == 0) dir = "/";
    else dir.resize(slash);
    struct statvfs st{};
    if (statvfs(dir.c_str(), &st) != 0) return -1;
    return (int64_t) st.f_bavail * (int64_t) st.f_frsize;
}
#endif

struct MappedView {
    const uint8_t* data = nullptr;
    uint64_t size = 0;
#if defined(_WIN32)
    HANDLE file = INVALID_HANDLE_VALUE;
    HANDLE mapping = nullptr;
    HANDLE direct_file = INVALID_HANDLE_VALUE;
    uint8_t* direct_buffer = nullptr;
    DWORD sector_bytes = 0;
    bool bounded_io = false;
    std::mutex read_mutex;
#else
    int fd = -1;
#endif

    ~MappedView() { close(); }

    void close() {
#if defined(_WIN32)
        if (data) UnmapViewOfFile(data);
        if (mapping) CloseHandle(mapping);
        if (file != INVALID_HANDLE_VALUE) CloseHandle(file);
        if (direct_file != INVALID_HANDLE_VALUE) CloseHandle(direct_file);
        if (direct_buffer) VirtualFree(direct_buffer, 0, MEM_RELEASE);
        direct_file = INVALID_HANDLE_VALUE;
        direct_buffer = nullptr;
        sector_bytes = 0;
        bounded_io = false;
        data = nullptr;
        mapping = nullptr;
        file = INVALID_HANDLE_VALUE;
#else
        if (data && size) ::munmap(const_cast<uint8_t*>(data), (size_t) size);
        if (fd >= 0) ::close(fd);
        data = nullptr;
        fd = -1;
#endif
        size = 0;
    }

    bool open(const std::string& path, std::string& err, bool sequential = false) {
        close();
#if defined(_WIN32)
        const std::wstring w = widen(path);
        if (w.empty()) {
            err = "expert pack path is not valid UTF-8";
            return false;
        }
        file = CreateFileW(w.c_str(), GENERIC_READ, FILE_SHARE_READ | FILE_SHARE_WRITE | FILE_SHARE_DELETE, nullptr,
                           OPEN_EXISTING, FILE_ATTRIBUTE_NORMAL | (sequential ? FILE_FLAG_SEQUENTIAL_SCAN : 0), nullptr);
        if (file == INVALID_HANDLE_VALUE) {
            err = "expert pack could not be opened";
            return false;
        }
        LARGE_INTEGER sz{};
        if (!GetFileSizeEx(file, &sz) || sz.QuadPart < 0) {
            err = "expert pack size could not be read";
            close();
            return false;
        }
        size = (uint64_t) sz.QuadPart;
        if (size == 0) return true;
        if (sequential) {
            // Full verification is a one-pass scan, not inference. Buffered
            // ReadFile can populate the system cache with both weight files
            // while CUDA already owns host memory. Keep one aligned MiB for
            // this scan; inference continues to use the existing mapped cache.
            FILE_STORAGE_INFO storage{};
            if (GetFileInformationByHandleEx(file, FileStorageInfo, &storage, sizeof(storage))) {
                const DWORD alignment = std::max(storage.LogicalBytesPerSector, storage.PhysicalBytesPerSectorForAtomicity);
                if (alignment && alignment <= 65536 && (alignment & (alignment - 1)) == 0) {
                    direct_file = CreateFileW(w.c_str(), GENERIC_READ, FILE_SHARE_READ | FILE_SHARE_WRITE | FILE_SHARE_DELETE,
                                              nullptr, OPEN_EXISTING, FILE_ATTRIBUTE_NORMAL | FILE_FLAG_NO_BUFFERING, nullptr);
                    if (direct_file != INVALID_HANDLE_VALUE) {
                        BY_HANDLE_FILE_INFORMATION original{}, direct{};
                        if (!GetFileInformationByHandle(file, &original) || !GetFileInformationByHandle(direct_file, &direct) ||
                            original.dwVolumeSerialNumber != direct.dwVolumeSerialNumber || original.nFileIndexHigh != direct.nFileIndexHigh ||
                            original.nFileIndexLow != direct.nFileIndexLow) {
                            err = "expert pack file changed while opening the verification reader";
                            close();
                            return false;
                        }
                        direct_buffer = static_cast<uint8_t*>(VirtualAlloc(nullptr, 1u << 20, MEM_RESERVE | MEM_COMMIT, PAGE_READWRITE));
                        if (direct_buffer) {
                            sector_bytes = alignment;
                            std::fprintf(stderr, "strata-q36: full-file verification uses bounded direct I/O (%lu-byte alignment)\n", (unsigned long) sector_bytes);
                        }
                        else { CloseHandle(direct_file); direct_file = INVALID_HANDLE_VALUE; }
                    }
                }
            }
        }
        mapping = CreateFileMappingW(file, nullptr, PAGE_READONLY, 0, 0, nullptr);
        if (!mapping) {
            err = "expert pack could not be mapped";
            close();
            return false;
        }
        data = static_cast<const uint8_t*>(MapViewOfFile(mapping, FILE_MAP_READ, 0, 0, 0));
        if (!data) {
            err = "expert pack view failed";
            close();
            return false;
        }
#else
        (void) sequential;
        fd = ::open(path.c_str(), O_RDONLY);
        if (fd < 0) {
            err = "expert pack could not be opened";
            return false;
        }
        struct stat st{};
        if (fstat(fd, &st) != 0 || st.st_size < 0) {
            err = "expert pack size could not be read";
            close();
            return false;
        }
        size = (uint64_t) st.st_size;
        if (size == 0) return true;
        void* view = ::mmap(nullptr, (size_t) size, PROT_READ, MAP_PRIVATE, fd, 0);
        if (view == MAP_FAILED) {
            err = "expert pack could not be mapped";
            data = nullptr;
            close();
            return false;
        }
        data = static_cast<const uint8_t*>(view);
#endif
        return true;
    }

    // Hashing through a second full mapped view populates tens of GiB of the
    // process working set on Windows. Bounded buffered reads leave room for the
    // engine and the resident experts, while still checking every byte.
    bool read(uint64_t offset, uint8_t* dst, size_t n) {
        Q36ReadTrace trace(n);
        const bool success = read_impl(offset, dst, n);
        trace.complete(success);
        return success;
    }

    bool read_impl(uint64_t offset, uint8_t* dst, size_t n) {
        if (offset > size || n > size - offset || (n && !dst)) return false;
#if defined(_WIN32)
        if (bounded_io) {
            // Both callers (CPU waves and H2D fills) may share this reader. The
            // bounce bytes cannot be reused until copied into caller storage.
            std::lock_guard<std::mutex> lock(read_mutex);
            while (n) {
                const uint64_t aligned = offset - offset % sector_bytes;
                const size_t skip = (size_t) (offset - aligned);
                const size_t take = std::min(n, MappedExpertPack::kBoundedReadScratchBytes - skip);
                const DWORD request = (DWORD) ((skip + take + sector_bytes - 1) / sector_bytes * sector_bytes);
                OVERLAPPED at{};
                at.Offset = (DWORD) aligned;
                at.OffsetHigh = (DWORD) (aligned >> 32);
                DWORD got = 0;
                if (!ReadFile(direct_file, direct_buffer, request, &got, &at) || got < skip + take) return false;
                std::memcpy(dst, direct_buffer + skip, take);
                dst += take; n -= take; offset += take;
            }
            return true;
        }
        while (n) {
            DWORD take = (DWORD) std::min<size_t>(n, 1u << 20);
            // Unaligned records and short tails use the buffered handle. No
            // padding is included in the hash of a non-sector-sized GGUF.
            const bool direct = direct_buffer && offset % sector_bytes == 0 && take >= sector_bytes;
            if (direct) take -= take % sector_bytes;
            DWORD got = 0;
            // This is a synchronous handle. Supplying the offset on each read
            // avoids a shared seek/read cursor race between staging callers.
            OVERLAPPED at{};
            at.Offset = (DWORD) offset;
            at.OffsetHigh = (DWORD) (offset >> 32);
            if (!ReadFile(direct ? direct_file : file, direct ? direct_buffer : dst, take, &got, &at) || got == 0) return false;
            if (direct) std::memcpy(dst, direct_buffer, got);
            dst += got; n -= got; offset += got;
        }
#else
        while (n) {
            const ssize_t got = pread(fd, dst, n, (off_t) offset);
            if (got < 0 && errno == EINTR) continue;
            if (got <= 0) return false;
            dst += got; n -= (size_t) got; offset += (uint64_t) got;
        }
#endif
        return true;
    }
};

bool hash_range(const uint8_t* p, uint64_t n, uint8_t out[32], std::string& err) {
    Sha256 sha;
    const uint64_t chunk = 1u << 20;
    uint64_t at = 0;
    while (at < n) {
        const uint64_t take = std::min(chunk, n - at);
        sha.update(p + at, (size_t) take);
        at += take;
    }
    if (!sha.final(out)) {
        err = "expert pack SHA-256 failed";
        return false;
    }
    return true;
}

#if defined(_WIN32)
class BinaryWriter {
public:
    ~BinaryWriter() { close(); }

    bool open(const std::string& path, std::string& err) {
        const std::wstring w = widen(path);
        if (w.empty()) {
            err = "expert pack path is not valid UTF-8";
            return false;
        }
        handle_ = CreateFileW(w.c_str(), GENERIC_READ | GENERIC_WRITE, FILE_SHARE_READ, nullptr, CREATE_ALWAYS,
                              FILE_ATTRIBUTE_NORMAL, nullptr);
        if (handle_ == INVALID_HANDLE_VALUE) {
            err = "expert pack partial file could not be created";
            return false;
        }
        return true;
    }

    bool write(const void* data, size_t n) {
        const auto* p = static_cast<const uint8_t*>(data);
        size_t left = n;
        while (left) {
            DWORD wrote = 0;
            const DWORD chunk = (DWORD) std::min(left, (size_t) 1 << 20);
            if (!WriteFile(handle_, p, chunk, &wrote, nullptr) || wrote != chunk) return false;
            p += chunk;
            left -= chunk;
            pos_ += chunk;
        }
        return true;
    }

    bool seek(uint64_t at) {
        LARGE_INTEGER li{};
        li.QuadPart = (LONGLONG) at;
        if (!SetFilePointerEx(handle_, li, nullptr, FILE_BEGIN)) return false;
        pos_ = at;
        return true;
    }

    bool flush() { return FlushFileBuffers(handle_) != 0; }

    void close() {
        if (handle_ != INVALID_HANDLE_VALUE) CloseHandle(handle_);
        handle_ = INVALID_HANDLE_VALUE;
    }

private:
    HANDLE handle_ = INVALID_HANDLE_VALUE;
    uint64_t pos_ = 0;
};
#else
class BinaryWriter {
public:
    ~BinaryWriter() { close(); }

    bool open(const std::string& path, std::string& err) {
        fp_ = std::fopen(path.c_str(), "w+b");
        if (!fp_) {
            err = "expert pack partial file could not be created";
            return false;
        }
        return true;
    }

    bool write(const void* data, size_t n) {
        if (std::fwrite(data, 1, n, fp_) != n) return false;
        pos_ += n;
        return true;
    }

    bool seek(uint64_t at) {
        if (fseeko(fp_, (off_t) at, SEEK_SET) != 0) return false;
        pos_ = at;
        return true;
    }

    bool flush() { return std::fflush(fp_) == 0 && fsync(fileno(fp_)) == 0; }

    void close() {
        if (fp_) std::fclose(fp_);
        fp_ = nullptr;
    }

private:
    std::FILE* fp_ = nullptr;
    uint64_t pos_ = 0;
};
#endif

bool hash_file(const std::string& path, uint8_t out[32], std::string& err, bool report_progress = false,
               const char* detail_stage = "model_hash") {
    Q36StartupDetail detail(detail_stage);
    MappedView view;
    if (!view.open(path, err, true)) { detail.error("file_open_failed"); return false; }
    detail.total(view.size);
    Sha256 sha;
    std::vector<uint8_t> buffer(1u << 20);
    const auto start = std::chrono::steady_clock::now();
    auto last = start;
    for (uint64_t at = 0; at < view.size;) {
        const size_t take = (size_t) std::min<uint64_t>(buffer.size(), view.size - at);
        if (!view.read(at, buffer.data(), take)) { detail.error("hash_read_failed"); err = "model hash read failed"; return false; }
        sha.update(buffer.data(), take);
        at += take;
        detail.progress(at);
        if (report_progress) {
            const auto now = std::chrono::steady_clock::now();
            if (now - last >= std::chrono::seconds(2)) {
                const auto ms = std::chrono::duration_cast<std::chrono::milliseconds>(now - start).count();
                q36_startup("expert_model_verify", "progress", ms, ms, nullptr);
                last = now;
            }
        }
    }
    if (!sha.final(out)) { err = "expert pack SHA-256 failed"; return false; }
    detail.end();
    return true;
}

struct VerificationProgress {
    const char* stage;
    std::chrono::steady_clock::time_point start = std::chrono::steady_clock::now();
    std::chrono::steady_clock::time_point last = start;
    bool finished = false;
    explicit VerificationProgress(const char* name) : stage(name) { q36_startup(stage, "begin", 0, 0, nullptr); }
    void tick() {
        const auto now = std::chrono::steady_clock::now();
        if (now - last < std::chrono::seconds(2)) return;
        const auto ms = std::chrono::duration_cast<std::chrono::milliseconds>(now - start).count();
        q36_startup(stage, "progress", ms, ms, nullptr);
        last = now;
    }
    void end() {
        const auto ms = std::chrono::duration_cast<std::chrono::milliseconds>(std::chrono::steady_clock::now() - start).count();
        q36_startup(stage, "end", ms, ms, nullptr);
        finished = true;
    }
    ~VerificationProgress() {
        if (!finished) q36_startup(stage, "error", 0, 0, "pack_verification_failed");
    }
};

// A local verification receipt is an optimisation for unchanged files, not a
// publisher signature. Only NTFS files with a readable change journal qualify.
// Read locks exclude writers/deletion while checking the receipt and hashing.
// Size/mtime alone are insufficient: restored timestamps must invalidate reuse.
struct VerificationCache {
    std::string path;
    std::array<uint8_t, 136> identity{};
    bool available = false;
#if defined(_WIN32)
    HANDLE model = INVALID_HANDLE_VALUE, pack = INVALID_HANDLE_VALUE;
    static bool fingerprint(HANDLE file, uint8_t* out) {
        if (file == INVALID_HANDLE_VALUE) return false;
        wchar_t filesystem[32]{};
        if (!GetVolumeInformationByHandleW(file, nullptr, 0, nullptr, nullptr, nullptr, filesystem, 32) ||
            std::wcscmp(filesystem, L"NTFS") != 0) return false;
        FILE_ID_INFO id{};
        FILE_BASIC_INFO basic{};
        LARGE_INTEGER size{};
        alignas(USN_RECORD_V2) uint8_t journal[4096]{};
        DWORD got = 0;
        if (!GetFileInformationByHandleEx(file, FileIdInfo, &id, sizeof(id)) ||
            !GetFileInformationByHandleEx(file, FileBasicInfo, &basic, sizeof(basic)) || !GetFileSizeEx(file, &size) ||
            !DeviceIoControl(file, FSCTL_READ_FILE_USN_DATA, nullptr, 0, journal, sizeof(journal), &got, nullptr) ||
            got < sizeof(USN_RECORD_V2)) return false;
        const auto* usn = reinterpret_cast<const USN_RECORD_V2*>(journal);
        if (usn->MajorVersion != 2 || usn->Usn <= 0 || basic.ChangeTime.QuadPart <= 0 || size.QuadPart < 0) return false;
        put_u64(out, id.VolumeSerialNumber);
        std::memcpy(out + 8, id.FileId.Identifier, 16);
        put_u64(out + 24, (uint64_t) size.QuadPart);
        put_u64(out + 32, (uint64_t) basic.CreationTime.QuadPart);
        put_u64(out + 40, (uint64_t) basic.LastWriteTime.QuadPart);
        put_u64(out + 48, (uint64_t) basic.ChangeTime.QuadPart);
        put_u64(out + 56, (uint64_t) usn->Usn);
        return true;
    }
#endif
    VerificationCache(const std::string& model_path, const std::string& pack_path, bool enabled) : path(pack_path + ".verified") {
#if defined(_WIN32)
        if (!enabled) return;
        const auto a = widen(model_path), b = widen(pack_path);
        if (a.empty() || b.empty()) return;
        model = CreateFileW(a.c_str(), GENERIC_READ, FILE_SHARE_READ, nullptr, OPEN_EXISTING, FILE_ATTRIBUTE_NORMAL, nullptr);
        pack = CreateFileW(b.c_str(), GENERIC_READ, FILE_SHARE_READ, nullptr, OPEN_EXISTING, FILE_ATTRIBUTE_NORMAL, nullptr);
        std::memcpy(identity.data(), "Q36VC2\0", 8);
        available = fingerprint(model, identity.data() + 8) && fingerprint(pack, identity.data() + 72);
#else
        (void) model_path; (void) enabled;
#endif
    }
    bool matches() const {
#if defined(_WIN32)
        if (!available) return false;
        const char* force = std::getenv("STRATA_Q36_VERIFY_PACK");
        if (force && std::strcmp(force, "1") == 0) return false;
        MappedView receipt;
        std::string ignored;
        if (!receipt.open(path, ignored) || !receipt.size || receipt.size > 4096) return false;
        DATA_BLOB sealed{(DWORD) receipt.size, const_cast<BYTE*>(receipt.data)}, plain{};
        if (!CryptUnprotectData(&sealed, nullptr, nullptr, nullptr, nullptr, CRYPTPROTECT_UI_FORBIDDEN, &plain)) return false;
        const bool same = plain.cbData == identity.size() && std::memcmp(plain.pbData, identity.data(), identity.size()) == 0;
        LocalFree(plain.pbData);
        return same;
#else
        return false;
#endif
    }
    void publish() const {
#if defined(_WIN32)
        if (!available) return;
        auto current = identity;
        if (!fingerprint(model, current.data() + 8) || !fingerprint(pack, current.data() + 72) || current != identity) return;
        // A downloaded or another user's receipt cannot grant a cache hit.
        // DPAPI binds this integrity-protected record to the current account.
        DATA_BLOB plain{(DWORD) identity.size(), const_cast<BYTE*>(identity.data())}, sealed{};
        if (!CryptProtectData(&plain, L"Strata Qwen3.6 verification", nullptr, nullptr, nullptr, CRYPTPROTECT_UI_FORBIDDEN, &sealed)) return;
        const std::string temp = path + "." + std::to_string(GetCurrentProcessId()) + "." + std::to_string(GetTickCount64()) + ".tmp";
        BinaryWriter writer;
        std::string ignored;
        const bool ok = writer.open(temp, ignored) && writer.write(sealed.pbData, sealed.cbData) && writer.flush();
        LocalFree(sealed.pbData);
        writer.close();
        if (ok) rename_over(temp, path);
        remove_file(temp);
#endif
    }
    ~VerificationCache() {
#if defined(_WIN32)
        if (model != INVALID_HANDLE_VALUE) CloseHandle(model);
        if (pack != INVALID_HANDLE_VALUE) CloseHandle(pack);
#endif
    }
};

struct HeldLock {
    std::string path;
    bool owned = false;
    ~HeldLock() {
        if (owned) remove_file(path);
    }
};

bool take_lock(const std::string& path, HeldLock& lock, std::string& err) {
    lock.path = path;
#if defined(_WIN32)
    const std::wstring w = widen(path);
    HANDLE h = CreateFileW(w.c_str(), GENERIC_READ | GENERIC_WRITE, 0, nullptr, CREATE_NEW, FILE_ATTRIBUTE_NORMAL, nullptr);
    if (h == INVALID_HANDLE_VALUE) {
        MappedView existing;
        std::string ignored;
        uint32_t pid = 0;
        if (existing.open(path, ignored) && existing.size >= 4) read_u32(existing.data, existing.size, 0, pid);
        existing.close();
        if (process_alive(pid)) {
            err = "another expert pack build holds the lock. This launch did not allocate the full host arena.";
            return false;
        }
        remove_file(path);
        h = CreateFileW(w.c_str(), GENERIC_READ | GENERIC_WRITE, 0, nullptr, CREATE_NEW, FILE_ATTRIBUTE_NORMAL, nullptr);
        if (h == INVALID_HANDLE_VALUE) {
            err = "expert pack lock could not be taken. This launch did not allocate the full host arena.";
            return false;
        }
    }
    const uint32_t pid = GetCurrentProcessId();
    uint8_t bytes[4];
    put_u32(bytes, pid);
    DWORD wrote = 0;
    const BOOL ok = WriteFile(h, bytes, 4, &wrote, nullptr);
    CloseHandle(h);
    if (!ok || wrote != 4) {
        remove_file(path);
        err = "expert pack lock could not be written";
        return false;
    }
#else
    const int fd = ::open(path.c_str(), O_CREAT | O_EXCL | O_WRONLY, 0644);
    if (fd < 0) {
        FILE* fp = std::fopen(path.c_str(), "rb");
        uint32_t pid = 0;
        if (fp) {
            uint8_t bytes[4]{};
            if (std::fread(bytes, 1, 4, fp) == 4) pid = (uint32_t) bytes[0] | ((uint32_t) bytes[1] << 8) |
                                                       ((uint32_t) bytes[2] << 16) | ((uint32_t) bytes[3] << 24);
            std::fclose(fp);
        }
        if (process_alive(pid)) {
            err = "another expert pack build holds the lock. This launch did not allocate the full host arena.";
            return false;
        }
        remove_file(path);
        const int again = ::open(path.c_str(), O_CREAT | O_EXCL | O_WRONLY, 0644);
        if (again < 0) {
            err = "expert pack lock could not be taken. This launch did not allocate the full host arena.";
            return false;
        }
        const uint32_t self = (uint32_t) ::getpid();
        uint8_t bytes[4];
        put_u32(bytes, self);
        const bool ok = ::write(again, bytes, 4) == 4;
        ::close(again);
        if (!ok) {
            remove_file(path);
            err = "expert pack lock could not be written";
            return false;
        }
        lock.owned = true;
        return true;
    }
    const uint32_t self = (uint32_t) ::getpid();
    uint8_t bytes[4];
    put_u32(bytes, self);
    const bool ok = ::write(fd, bytes, 4) == 4;
    ::close(fd);
    if (!ok) {
        remove_file(path);
        err = "expert pack lock could not be written";
        return false;
    }
#endif
    lock.owned = true;
    return true;
}

struct LayerPlan {
    PackLayer layer;
    uint64_t layer_bytes = 0;
    uint64_t file_offset = 0;
};

bool plan_layers(const PackBuildRequest& req, std::vector<LayerPlan>& out, uint64_t& payload_bytes, uint64_t& payload_offset,
                 std::string& err) {
    if (req.model_path.empty() || req.layers.empty() || req.layers.size() > 256) {
        err = "expert pack request has no model or an unsupported layer count";
        return false;
    }
    if (req.tail != kQ36ExpertTail) {
        err = "expert pack tail is not the engine tail";
        return false;
    }
    payload_bytes = 0;
    out.clear();
    out.reserve(req.layers.size());
    for (const PackLayer& layer : req.layers) {
        if (!layout_ok(layer.layout, layer.source, err)) return false;
        LayerPlan plan;
        plan.layer = layer;
        plan.layer_bytes = layer.layout.stride * layer.layout.n_expert + req.tail;
        if (payload_bytes > std::numeric_limits<uint64_t>::max() - plan.layer_bytes) {
            err = "expert pack payload overflows";
            return false;
        }
        payload_bytes += plan.layer_bytes;
        out.push_back(plan);
    }
    const uint64_t header = kExpertPackPrefixBytes + (uint64_t) out.size() * kExpertPackRecordBytes;
    payload_offset = (header + kExpertPackAlign - 1) & ~(kExpertPackAlign - 1);
    uint64_t cursor = payload_offset;
    for (LayerPlan& plan : out) {
        plan.file_offset = cursor;
        cursor += plan.layer_bytes;
    }
    return true;
}

}  // namespace

struct MappedExpertPack::Impl {
    // In bounded mode retain read locks across verification, mapping and use.
    // The ordinary mapped-pack path keeps its existing lifecycle.
    std::unique_ptr<VerificationCache> lifetime_lock;
    MappedView view;
    std::string path;
    struct Span {
        const uint8_t* data = nullptr;
        uint64_t bytes = 0;
    };
    std::vector<Span> layers;
};

MappedExpertPack::MappedExpertPack() = default;
MappedExpertPack::~MappedExpertPack() = default;
MappedExpertPack::MappedExpertPack(MappedExpertPack&&) noexcept = default;
MappedExpertPack& MappedExpertPack::operator=(MappedExpertPack&&) noexcept = default;

const uint8_t* MappedExpertPack::layer_data(uint32_t layer) const {
    if (!impl_ || layer >= impl_->layers.size()) return nullptr;
    return impl_->layers[layer].data;
}

uint64_t MappedExpertPack::layer_bytes(uint32_t layer) const {
    if (!impl_ || layer >= impl_->layers.size()) return 0;
    return impl_->layers[layer].bytes;
}

uint32_t MappedExpertPack::layer_count() const { return impl_ ? (uint32_t) impl_->layers.size() : 0; }

const std::string& MappedExpertPack::path() const {
    static const std::string empty;
    return impl_ ? impl_->path : empty;
}

bool MappedExpertPack::enable_bounded_reads(std::string& err) {
#if defined(_WIN32)
    if (!impl_ || impl_->view.file == INVALID_HANDLE_VALUE) {
        err = "Q36 bounded reader has no open pack";
        return false;
    }
    auto& view = impl_->view;
    if (view.bounded_io) return true;
    FILE_STORAGE_INFO storage{};
    if (!GetFileInformationByHandleEx(view.file, FileStorageInfo, &storage, sizeof(storage))) {
        err = "Q36 bounded reader cannot observe sector alignment";
        return false;
    }
    const DWORD alignment = std::max(storage.LogicalBytesPerSector, storage.PhysicalBytesPerSectorForAtomicity);
    if (!alignment || alignment > 65536 || (alignment & (alignment - 1)) != 0) {
        err = "Q36 bounded reader unsupported sector alignment";
        return false;
    }
    if (!q36_require_host_headroom(kBoundedReadScratchBytes, err)) return false;
    HANDLE file = ReOpenFile(view.file, GENERIC_READ, FILE_SHARE_READ | FILE_SHARE_WRITE | FILE_SHARE_DELETE,
                             FILE_FLAG_NO_BUFFERING);
    if (file == INVALID_HANDLE_VALUE) {
        err = "Q36 bounded reader cannot reopen pack for unbuffered I/O";
        return false;
    }
    auto* buffer = static_cast<uint8_t*>(VirtualAlloc(nullptr, kBoundedReadScratchBytes, MEM_RESERVE | MEM_COMMIT, PAGE_READWRITE));
    if (!buffer || reinterpret_cast<uintptr_t>(buffer) % alignment != 0) {
        if (buffer) VirtualFree(buffer, 0, MEM_RELEASE);
        CloseHandle(file);
        err = "Q36 bounded reader aligned buffer allocation failed";
        return false;
    }
    view.direct_file = file;
    view.direct_buffer = buffer;
    view.sector_bytes = alignment;
    view.bounded_io = true;
    return true;
#else
    err = "Q36 bounded reader is Windows-only";
    return false;
#endif
}

uint64_t MappedExpertPack::bounded_read_scratch_bytes() const {
#if defined(_WIN32)
    return impl_ && impl_->view.bounded_io ? kBoundedReadScratchBytes : 0;
#else
    return 0;
#endif
}

bool MappedExpertPack::read_mapped(const uint8_t* src, uint8_t* dst, size_t bytes) const {
    if (!impl_ || !impl_->view.data || !src || (bytes && !dst)) return false;
    // Integer range checks also reject pointers outside this mapping without
    // subtracting unrelated C++ pointers or overflowing src + bytes.
    const uintptr_t base = reinterpret_cast<uintptr_t>(impl_->view.data);
    const uintptr_t address = reinterpret_cast<uintptr_t>(src);
    if (address < base) return false;
    const uint64_t offset = (uint64_t) (address - base);
    if (offset > impl_->view.size || bytes > impl_->view.size - offset) return false;
    return impl_->view.read(offset, dst, bytes);
}

bool pack_expert_blob(const ExpertBlobLayout& layout, const ExpertSource& src, uint32_t expert, uint8_t* dst) {
    std::string err;
    if (!dst || expert >= layout.n_expert || !src.gate || !src.up || !src.down || !layout_ok(layout, src, err)) return false;
    std::memset(dst, 0, (size_t) layout.stride);
    std::memcpy(dst, src.gate + (uint64_t) expert * src.src_gsz, (size_t) src.src_gsz);
    std::memcpy(dst + layout.up_off, src.up + (uint64_t) expert * src.src_gsz, (size_t) src.src_gsz);
    std::memcpy(dst + layout.down_off, src.down + (uint64_t) expert * src.src_dsz, (size_t) src.src_dsz);
    return true;
}

std::string expert_pack_sha256_file(const std::string& path, std::string& err) {
    uint8_t dig[32];
    if (!hash_file(path, dig, err, false, nullptr)) return {};
    return to_hex(dig, 32);
}

uint64_t expert_pack_last_scratch_bytes() { return g_scratch_bytes; }
bool expert_pack_last_verification_cached() { return g_verification_cached; }

std::string expert_pack_verify(const std::string& pack_path, const std::string& model_path,
                               const std::vector<PackLayer>* expected, uint64_t tail, bool reuse_verified) {
    g_verification_cached = false;
    VerificationCache cache(model_path, pack_path, reuse_verified);
    const bool cached = cache.matches();
    std::string err;
    MappedView view;
    if (!view.open(pack_path, err, true)) return err;
    if (view.size < kExpertPackPrefixBytes) return "expert pack is shorter than its header";
    if (std::memcmp(view.data, kExpertPackMagic, 8) != 0) return "expert pack magic does not match";
    uint32_t version = 0, endian = 0, n_layers = 0;
    if (!read_u32(view.data, view.size, kExpertPackVersionField, version)) return "expert pack version could not be read";
    if (version != kExpertPackVersion) return "expert pack version is not supported";
    if (!read_u32(view.data, view.size, 12, endian) || endian != 1) return "expert pack endian is not supported";
    uint8_t model_sha[32];
    VerificationProgress model_progress("expert_model_verify");
    if (!cached) {
        if (!hash_file(model_path, model_sha, err, true)) return err.empty() ? "model file could not be hashed" : err;
        if (std::memcmp(view.data + 16, model_sha, 32) != 0) return "expert pack is not for this model file";
    }
    model_progress.end();
    if (!read_u32(view.data, view.size, 48, n_layers) || n_layers == 0 || n_layers > 256) return "expert pack layer count is not supported";
    uint64_t file_tail = 0, payload_offset = 0, payload_bytes = 0;
    if (!read_u64(view.data, view.size, 56, file_tail) || file_tail != tail) return "expert pack tail does not match";
    if (!read_u64(view.data, view.size, kExpertPackPayloadOffsetField, payload_offset)) return "expert pack payload offset could not be read";
    if (!read_u64(view.data, view.size, 72, payload_bytes)) return "expert pack payload size could not be read";
    if (payload_offset > view.size || payload_bytes > view.size - payload_offset || payload_offset + payload_bytes != view.size) {
        return "expert pack payload does not fill the file";
    }
    const uint64_t header = kExpertPackPrefixBytes + (uint64_t) n_layers * kExpertPackRecordBytes;
    if (payload_offset < header || (payload_offset & (kExpertPackAlign - 1)) != 0) return "expert pack payload is not aligned";
    Sha256 payload_hash;
    VerificationProgress payload_progress("expert_pack_verify");
    // A reused receipt did not read the payload in this process. Do not report invented read bytes.
    Q36StartupDetail payload_detail("pack_payload_verify", cached ? 0 : payload_bytes);
    std::vector<uint8_t> verify_buffer(1u << 20);
    bool checksum_mismatch = false;
    if (expected && expected->size() != n_layers) return "expert pack layer count does not match this model";
    uint64_t cursor = payload_offset;
    for (uint32_t i = 0; i < n_layers; ++i) {
        const uint64_t rec = kExpertPackPrefixBytes + (uint64_t) i * kExpertPackRecordBytes;
        uint32_t gu_type = 0, d_type = 0, n_embd = 0, n_ff = 0, n_expert = 0, crc = 0;
        uint64_t gu_row = 0, d_row = 0, up_off = 0, down_off = 0, blob_bytes = 0, stride = 0, src_gsz = 0, src_dsz = 0,
                 layer_bytes = 0, file_offset = 0;
        if (!read_u32(view.data, view.size, rec + 0, gu_type) || !read_u32(view.data, view.size, rec + 4, d_type) ||
            !read_u32(view.data, view.size, rec + 8, n_embd) || !read_u32(view.data, view.size, rec + 12, n_ff) ||
            !read_u32(view.data, view.size, rec + 16, n_expert) || !read_u64(view.data, view.size, rec + 24, gu_row) ||
            !read_u64(view.data, view.size, rec + 32, d_row) || !read_u64(view.data, view.size, rec + 40, up_off) ||
            !read_u64(view.data, view.size, rec + 48, down_off) || !read_u64(view.data, view.size, rec + 56, blob_bytes) ||
            !read_u64(view.data, view.size, rec + 64, stride) || !read_u64(view.data, view.size, rec + 72, src_gsz) ||
            !read_u64(view.data, view.size, rec + 80, src_dsz) || !read_u64(view.data, view.size, rec + 88, layer_bytes) ||
            !read_u64(view.data, view.size, rec + 96, file_offset) || !read_u32(view.data, view.size, rec + 104, crc)) {
            return "expert pack layer record is truncated";
        }
        if (n_expert == 0 || stride == 0 || layer_bytes != stride * n_expert + tail || file_offset != cursor) {
            return "expert pack layer geometry is inconsistent";
        }
        if (cursor > view.size || layer_bytes > view.size - cursor) return "expert pack layer runs past the file";
        // SHA-256 and each layer's CRC cover the same bytes in one sequential
        // pass. A 20-GiB pack must not be read twice on a 32-GiB machine.
        uint32_t actual_crc = 0;
        for (uint64_t at = 0; !cached && at < layer_bytes;) {
            const size_t take = (size_t) std::min<uint64_t>(verify_buffer.size(), layer_bytes - at);
            if (!view.read(cursor + at, verify_buffer.data(), take)) return "expert pack payload read failed";
            payload_hash.update(verify_buffer.data(), take);
            actual_crc = crc32_update(actual_crc, verify_buffer.data(), take);
            at += take;
            payload_progress.tick();
            payload_detail.progress(cursor - payload_offset + at, (int) i);
        }
        checksum_mismatch = checksum_mismatch || (!cached && actual_crc != crc);
        if (expected) {
            const ExpertBlobLayout& want = (*expected)[i].layout;
            const ExpertSource& src = (*expected)[i].source;
            if (want.gu_type != gu_type || want.d_type != d_type || want.n_embd != n_embd || want.n_ff != n_ff ||
                want.n_expert != n_expert || want.gu_row != gu_row || want.d_row != d_row || want.up_off != up_off ||
                want.down_off != down_off || want.blob_bytes != blob_bytes || want.stride != stride ||
                src.src_gsz != src_gsz || src.src_dsz != src_dsz) {
                return "expert pack geometry does not match this model";
            }
        }
        cursor += layer_bytes;
    }
    if (cursor != view.size) return "expert pack has bytes after the last layer";
    uint8_t payload_sha[32];
    if (!cached && !payload_hash.final(payload_sha)) return "expert pack SHA-256 failed";
    if (!cached && std::memcmp(view.data + 80, payload_sha, 32) != 0) return "expert pack payload hash does not match";
    if (checksum_mismatch) return "expert pack layer checksum does not match";
    payload_progress.end();
    payload_detail.end();
    if (cached) std::fprintf(stderr, "strata-q36: reusing full expert-pack verification (unchanged read-locked NTFS files and USNs)\n");
    else cache.publish();
    g_verification_cached = cached;
    return {};
}

std::unique_ptr<MappedExpertPack> open_or_build_expert_pack(const PackBuildRequest& req, std::string& err) {
    g_scratch_bytes = 0;
    err.clear();
    std::vector<LayerPlan> plans;
    uint64_t payload_bytes = 0, payload_offset = 0;
    if (!plan_layers(req, plans, payload_bytes, payload_offset, err)) return nullptr;
    const std::string final_path = pack_path_for(req.model_path);
    std::unique_ptr<VerificationCache> lifetime_lock;
    if (req.bounded_reads) {
#if defined(_WIN32)
        lifetime_lock = std::make_unique<VerificationCache>(req.model_path, final_path, true);
        if (lifetime_lock->model == INVALID_HANDLE_VALUE || lifetime_lock->pack == INVALID_HANDLE_VALUE) {
            err = "Q36 bounded mode requires an existing read-lockable model and expert pack; no pack was built";
            return nullptr;
        }
#else
        err = "Q36 bounded reader is Windows-only";
        return nullptr;
#endif
    }
    const std::string reason = expert_pack_verify(final_path, req.model_path, &req.layers, req.tail, true);
    if (reason.empty()) {
        auto pack = std::unique_ptr<MappedExpertPack>(new MappedExpertPack());
        pack->impl_.reset(new MappedExpertPack::Impl());
        pack->impl_->lifetime_lock = std::move(lifetime_lock);
        pack->impl_->path = final_path;
        if (!pack->impl_->view.open(final_path, err)) return nullptr;
        if (req.bounded_reads && !pack->enable_bounded_reads(err)) return nullptr;
        pack->impl_->layers.resize(plans.size());
        for (size_t i = 0; i < plans.size(); ++i) {
            pack->impl_->layers[i].data = pack->impl_->view.data + plans[i].file_offset;
            pack->impl_->layers[i].bytes = plans[i].layer_bytes;
        }
        return pack;
    }
    if (req.bounded_reads) {
        err = "Q36 bounded mode requires a valid existing expert pack; no pack was built: " + reason;
        return nullptr;
    }
    for (const LayerPlan& plan : plans) {
        if (!plan.layer.source.gate || !plan.layer.source.up || !plan.layer.source.down) {
            err = reason;
            return nullptr;
        }
    }
    uint8_t model_sha[32];
    if (!hash_file(req.model_path, model_sha, err)) return nullptr;
    const uint64_t need = payload_offset + payload_bytes + (64ull << 20);
    const int64_t freeb = disk_free(final_path);
    if (freeb < 0 || (uint64_t) freeb < need) {
        err = "not enough free space for the expert pack. The full host arena was not allocated.";
        return nullptr;
    }
    HeldLock lock;
    if (!take_lock(lock_path_for(final_path), lock, err)) return nullptr;
    const std::string partial = partial_path_for(final_path);
    remove_file(partial);
    BinaryWriter out;
    if (!out.open(partial, err)) return nullptr;
    std::vector<uint8_t> header((size_t) payload_offset, 0);
    std::memcpy(header.data(), kExpertPackMagic, 8);
    put_u32(header.data() + kExpertPackVersionField, kExpertPackVersion);
    put_u32(header.data() + 12, 1);
    std::memcpy(header.data() + 16, model_sha, 32);
    put_u32(header.data() + 48, (uint32_t) plans.size());
    put_u64(header.data() + 56, req.tail);
    put_u64(header.data() + kExpertPackPayloadOffsetField, payload_offset);
    put_u64(header.data() + 72, payload_bytes);
    put_u32(header.data() + 112, (uint32_t) payload_offset);
    for (size_t i = 0; i < plans.size(); ++i) {
        uint8_t* rec = header.data() + kExpertPackPrefixBytes + i * kExpertPackRecordBytes;
        const ExpertBlobLayout& layout = plans[i].layer.layout;
        const ExpertSource& src = plans[i].layer.source;
        put_u32(rec + 0, layout.gu_type);
        put_u32(rec + 4, layout.d_type);
        put_u32(rec + 8, layout.n_embd);
        put_u32(rec + 12, layout.n_ff);
        put_u32(rec + 16, layout.n_expert);
        put_u64(rec + 24, layout.gu_row);
        put_u64(rec + 32, layout.d_row);
        put_u64(rec + 40, layout.up_off);
        put_u64(rec + 48, layout.down_off);
        put_u64(rec + 56, layout.blob_bytes);
        put_u64(rec + 64, layout.stride);
        put_u64(rec + 72, src.src_gsz);
        put_u64(rec + 80, src.src_dsz);
        put_u64(rec + 88, plans[i].layer_bytes);
        put_u64(rec + 96, plans[i].file_offset);
    }
    if (!out.write(header.data(), header.size())) {
        out.close();
        remove_file(partial);
        err = "expert pack header could not be written. The full host arena was not allocated.";
        return nullptr;
    }
    Sha256 payload_sha;
    std::vector<uint8_t> zeros((size_t) std::min(req.tail, (uint64_t) 4096), 0);
    for (size_t i = 0; i < plans.size(); ++i) {
        const ExpertBlobLayout& layout = plans[i].layer.layout;
        const ExpertSource& src = plans[i].layer.source;
        if (layout.stride > g_scratch_bytes) g_scratch_bytes = layout.stride;
        std::vector<uint8_t> scratch((size_t) layout.stride, 0);
        uint32_t crc = 0;
        for (uint32_t e = 0; e < layout.n_expert; ++e) {
            if (!pack_expert_blob(layout, src, e, scratch.data())) {
                out.close();
                remove_file(partial);
                err = "expert pack blob could not be written. The full host arena was not allocated.";
                return nullptr;
            }
            crc = crc32_update(crc, scratch.data(), scratch.size());
            payload_sha.update(scratch.data(), scratch.size());
            if (!out.write(scratch.data(), scratch.size())) {
                out.close();
                remove_file(partial);
                err = "expert pack payload could not be written. The full host arena was not allocated.";
                return nullptr;
            }
        }
        uint64_t tail_left = req.tail;
        while (tail_left) {
            const size_t take = (size_t) std::min(tail_left, (uint64_t) zeros.size());
            crc = crc32_update(crc, zeros.data(), take);
            payload_sha.update(zeros.data(), take);
            if (!out.write(zeros.data(), take)) {
                out.close();
                remove_file(partial);
                err = "expert pack tail could not be written. The full host arena was not allocated.";
                return nullptr;
            }
            tail_left -= take;
        }
        uint8_t crc_bytes[4];
        put_u32(crc_bytes, crc);
        const uint64_t crc_at = kExpertPackPrefixBytes + (uint64_t) i * kExpertPackRecordBytes + 104;
        if (!out.seek(crc_at) || !out.write(crc_bytes, 4) || !out.seek(plans[i].file_offset + plans[i].layer_bytes)) {
            out.close();
            remove_file(partial);
            err = "expert pack checksum could not be stored. The full host arena was not allocated.";
            return nullptr;
        }
    }
    uint8_t dig[32];
    if (!payload_sha.final(dig) || !out.seek(80) || !out.write(dig, 32) || !out.flush()) {
        out.close();
        remove_file(partial);
        err = "expert pack could not be flushed. The full host arena was not allocated.";
        return nullptr;
    }
    out.close();
    if (!remove_file(rejected_path_for(final_path))) {
        err = "the previous rejected expert pack could not be removed";
        remove_file(partial);
        return nullptr;
    }
    MappedView existing;
    std::string present;
    if (existing.open(final_path, present)) {
        existing.close();
        if (!rename_over(final_path, rejected_path_for(final_path))) {
            remove_file(partial);
            err = "the existing expert pack could not be moved aside. The full host arena was not allocated.";
            return nullptr;
        }
    }
    if (!rename_over(partial, final_path)) {
        remove_file(partial);
        err = "expert pack could not be published. The full host arena was not allocated.";
        return nullptr;
    }
    const std::string published = expert_pack_verify(final_path, req.model_path, &req.layers, req.tail, true);
    if (!published.empty()) {
        rename_over(final_path, rejected_path_for(final_path));
        err = published;
        return nullptr;
    }
    auto pack = std::unique_ptr<MappedExpertPack>(new MappedExpertPack());
    pack->impl_.reset(new MappedExpertPack::Impl());
    pack->impl_->path = final_path;
    if (!pack->impl_->view.open(final_path, err)) return nullptr;
    pack->impl_->layers.resize(plans.size());
    for (size_t i = 0; i < plans.size(); ++i) {
        pack->impl_->layers[i].data = pack->impl_->view.data + plans[i].file_offset;
        pack->impl_->layers[i].bytes = plans[i].layer_bytes;
    }
    return pack;
}

struct ExpertPinPool::Impl {
    uint64_t slot_bytes = 0;
    bool owns = true;
    std::vector<std::unique_ptr<uint8_t[]>> owned;
    std::vector<uint8_t*> ptrs;
    std::vector<PinSlotState> states;
};

ExpertPinPool::ExpertPinPool(uint32_t slots, uint64_t slot_bytes) : impl_(new Impl()) {
    if (slots == 0) slots = kDefaultSlots;
    if (slot_bytes == 0) slot_bytes = kDefaultSlotBytes;
    impl_->owns = true;
    impl_->slot_bytes = slot_bytes;
    impl_->owned.resize(slots);
    impl_->ptrs.resize(slots);
    impl_->states.assign(slots, PinSlotState::Reusable);
    for (uint32_t i = 0; i < slots; ++i) {
        impl_->owned[i].reset(new uint8_t[(size_t) slot_bytes]());
        impl_->ptrs[i] = impl_->owned[i].get();
    }
}

ExpertPinPool::ExpertPinPool(uint8_t* const* external, uint32_t slots, uint64_t slot_bytes) : impl_(new Impl()) {
    if (slots == 0 || slot_bytes == 0 || !external) {
        impl_->owns = true;
        impl_->slot_bytes = 0;
        return;
    }
    impl_->owns = false;
    impl_->slot_bytes = slot_bytes;
    impl_->ptrs.assign(external, external + slots);
    impl_->states.assign(slots, PinSlotState::Reusable);
}

ExpertPinPool::ExpertPinPool(ExpertPinPool&&) noexcept = default;
ExpertPinPool& ExpertPinPool::operator=(ExpertPinPool&&) noexcept = default;
ExpertPinPool::~ExpertPinPool() = default;

uint32_t ExpertPinPool::slot_count() const { return (uint32_t) impl_->states.size(); }
uint64_t ExpertPinPool::slot_bytes() const { return impl_->slot_bytes; }
uint64_t ExpertPinPool::committed_bytes() const {
    return (uint64_t) impl_->states.size() * impl_->slot_bytes;
}
bool ExpertPinPool::owns_buffers() const { return impl_->owns; }

PinSlotState ExpertPinPool::state(int slot) const {
    if (slot < 0 || (size_t) slot >= impl_->states.size()) return PinSlotState::Reusable;
    return impl_->states[(size_t) slot];
}

uint8_t* ExpertPinPool::data(int slot) {
    if (slot < 0 || (size_t) slot >= impl_->ptrs.size()) return nullptr;
    return impl_->ptrs[(size_t) slot];
}

const uint8_t* ExpertPinPool::data(int slot) const {
    if (slot < 0 || (size_t) slot >= impl_->ptrs.size()) return nullptr;
    return impl_->ptrs[(size_t) slot];
}

int ExpertPinPool::begin_fill() {
    for (size_t i = 0; i < impl_->states.size(); ++i) {
        if (impl_->states[i] != PinSlotState::Reusable) continue;
        impl_->states[i] = PinSlotState::HostFilling;
        return (int) i;
    }
    return -1;
}

bool ExpertPinPool::mark_h2d(int slot) {
    if (slot < 0 || (size_t) slot >= impl_->states.size()) return false;
    if (impl_->states[(size_t) slot] != PinSlotState::HostFilling) return false;
    impl_->states[(size_t) slot] = PinSlotState::H2dInFlight;
    return true;
}

bool ExpertPinPool::release(int slot) {
    if (slot < 0 || (size_t) slot >= impl_->states.size()) return false;
    if (impl_->states[(size_t) slot] == PinSlotState::Reusable) return false;
    impl_->states[(size_t) slot] = PinSlotState::Reusable;
    return true;
}

}  // namespace strata::qwen35
