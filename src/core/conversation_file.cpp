#include "strata/core/conversation_file.hpp"
#include "strata/core/rht_state_codec.hpp"

#include <algorithm>
#include <bit>
#include <cerrno>
#include <chrono>
#include <climits>
#include <condition_variable>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <mutex>
#include <new>
#include <unordered_map>
#include <optional>
#include <filesystem>
#include <limits>
#include <memory>
#include <random>
#include <string>
#include <system_error>
#include <thread>
#include <type_traits>
#ifdef _WIN32
#ifndef WIN32_LEAN_AND_MEAN
#define WIN32_LEAN_AND_MEAN
#endif
#ifndef NOMINMAX
#define NOMINMAX
#endif
#include <windows.h>
#include <malloc.h>
#include <process.h>
#else
#include <fcntl.h>
#include <sys/stat.h>
#include <sys/statvfs.h>
#include <sys/types.h>
#include <unistd.h>
#endif

// Format v1 stores integers and floats as this machine holds them in memory; it is defined as little-endian with
// IEEE-754 floats, so a build where that is not the memory layout is refused rather than writing another format.
static_assert(std::endian::native == std::endian::little, "session files (format v1) are little-endian only");
static_assert(std::numeric_limits<float>::is_iec559 && std::numeric_limits<double>::is_iec559,
              "session files (format v1) need IEEE-754 floats");

namespace strata::core {
namespace {
constexpr char kMagic[8] = {'S', 'T', 'R', 'S', 'E', 'S', 'S', '\x01'};
constexpr char kEnd[8] = {'S', 'T', 'R', 'S', 'E', 'N', 'D', '\x01'};
constexpr uint32_t kVersion = 1;
// Version 2 uses descriptor-coded state arrays (codec, logical, stored, bytes): a prefix checkpoint may use REF
// descriptors, and an ordinary disk save may use RHT for GDN. Without either a REF descriptor or RHT state, bytes
// remain the version-1 format.
constexpr uint32_t kVersionV2 = 2;
// Version 3 is the K/V-first layout for the disk-only background-state path (--conversation-cache-disk-background-
// state): geometry, layer range and cvec, then the K/V headers and payload, then the same v2 state descriptors, so
// the K/V sources are read (and released) before any state is serialized. Opt-in: default writes stay v1/v2.
constexpr uint32_t kVersionV3 = 3;
constexpr size_t kHeader = 64, kTrailer = 16;

using TimingClock = std::chrono::steady_clock;
uint64_t timing_ns(TimingClock::time_point since) {
    return (uint64_t) std::chrono::duration_cast<std::chrono::nanoseconds>(TimingClock::now() - since).count();
}
void timing_add(uint64_t* counter, TimingClock::time_point since) {
    if (counter) *counter += timing_ns(since);
}

// per checkpoint the v2 descriptors add 5 x 16 B over v1's five 8 B length words (live included)
constexpr uint64_t kV2DescriptorSlack = 80;

// Progress for a transfer: one report after every block that was handed to (or read from) the OS, whatever its size
// - the last, partial block included - so a small file reports too, and each report follows real movement.
class Ticker {
public:
    explicit Ticker(const SessionProgress& fn) : fn_(fn) {}
    void operator()(uint64_t done, uint64_t total) { if (fn_) fn_(done, total); }
private:
    SessionProgress fn_;
};

SessionError errno_kind(int e) {
    switch (e) {
        case ENOSPC:
#ifdef EDQUOT
        case EDQUOT:
#endif
        case EFBIG: return SessionError::storage;
        case ENOMEM: return SessionError::memory;
        default: return SessionError::io;
    }
}

constexpr uint64_t P1 = 0x9E3779B185EBCA87ull, P2 = 0xC2B2AE3D27D4EB4Full, P3 = 0x165667B19E3779F9ull,
                   P4 = 0x85EBCA77C2B2AE63ull, P5 = 0x27D4EB2F165667C5ull;
uint64_t rotl(uint64_t x, int r) { return (x << r) | (x >> (64 - r)); }
uint64_t round1(uint64_t acc, uint64_t w) { return rotl(acc + w * P2, 31) * P1; }
uint64_t load64(const uint8_t* p) { uint64_t v; std::memcpy(&v, p, 8); return v; }

// ---- file I/O in aligned 16 MiB blocks.  Linux: O_DIRECT when the filesystem takes it (no page cache, so a
// 1.2 GB session neither evicts other files nor is charged to the engine's memory cgroup), buffered I/O otherwise
// (tmpfs, a filesystem that refuses an aligned direct transfer, or STRATA_SESSION_BUFFERED=1).  Windows and other
// systems: ordinary buffered I/O, same blocks.
constexpr size_t kAlign = 4096, kBlock = 16u << 20;

uint64_t sat_add(uint64_t a, uint64_t b) { return a > UINT64_MAX - b ? UINT64_MAX : a + b; }
uint64_t sat_mul(uint64_t a, uint64_t b) { return a && b > UINT64_MAX / a ? UINT64_MAX : a * b; }

uint64_t host_page_size() {
    static const uint64_t page = [] {
#ifdef _WIN32
        SYSTEM_INFO info{};
        GetSystemInfo(&info);
        return info.dwPageSize ? (uint64_t) info.dwPageSize : (uint64_t) kAlign;
#else
        const long value = ::sysconf(_SC_PAGESIZE);
        return value > 0 ? (uint64_t) value : (uint64_t) kAlign;
#endif
    }();
    return page;
}

uint64_t round_allocation_pages(uint64_t bytes) {
    if (bytes == 0) return 0;
    const uint64_t page = host_page_size();
    const uint64_t remainder = bytes % page;
    return remainder == 0 ? bytes : sat_add(bytes, page - remainder);
}

uint64_t allocation_bytes(uint64_t count, size_t element_size) {
    return round_allocation_pages(sat_mul(count, (uint64_t) element_size));
}

#ifdef _WIN32
struct AlignedFree { void operator()(uint8_t* p) const { _aligned_free(p); } };
using Aligned = std::unique_ptr<uint8_t, AlignedFree>;
Aligned aligned_block() { return Aligned(static_cast<uint8_t*>(_aligned_malloc(kBlock, kAlign))); }
#else
struct AlignedFree { void operator()(uint8_t* p) const { std::free(p); } };
using Aligned = std::unique_ptr<uint8_t, AlignedFree>;
Aligned aligned_block() { return Aligned(static_cast<uint8_t*>(std::aligned_alloc(kAlign, kBlock))); }
#endif

uint64_t random64() {
    std::random_device rd;
    uint64_t r = (uint64_t(rd()) << 32) ^ rd();
    r ^= (uint64_t) std::chrono::steady_clock::now().time_since_epoch().count() * P1;
#ifdef _WIN32
    r ^= (uint64_t) _getpid() * P2;
#else
    r ^= (uint64_t) ::getpid() * P2;
#endif
    return r;
}

// The temporary name beside the destination: hidden (a leading dot, which the server never accepts as a session
// name), unique, and short enough for any filesystem whatever the destination's length.
std::string temp_name(const std::string& leaf) {
    size_t keep = std::min<size_t>(leaf.size(), 64);
    while (keep > 0 && keep < leaf.size() && (uint8_t(leaf[keep]) & 0xC0) == 0x80) --keep;   // whole UTF-8 chars
    char hex[17];
    std::snprintf(hex, sizeof hex, "%016llx", (unsigned long long) random64());
    return "." + leaf.substr(0, keep) + "." + hex + ".tmp";
}

// a test's injected failure at `step`: its errno value (0 = none)
int injected(const std::function<int(const char*)>* fault, const char* step) {
    return fault && *fault ? (*fault)(step) : 0;
}

bool split_path(const std::string& path, std::string& dir, std::string& leaf, std::string& error) {
    const size_t cut = path.find_last_of(
#ifdef _WIN32
        "/\\"
#else
        "/"
#endif
    );
    dir = cut == std::string::npos ? std::string() : path.substr(0, cut == 0 ? 1 : cut);
    leaf = cut == std::string::npos ? path : path.substr(cut + 1);
    if (leaf.empty() || leaf == "." || leaf == "..") { error = "session file: " + path + " names no file"; return false; }
    return true;
}

#ifdef _WIN32
// paths reach the engine as UTF-8 (the server writes its commands as UTF-8); invalid UTF-8 is refused, not replaced
bool wide(const std::string& s, std::wstring& w) {
    w.clear();
    if (s.empty()) return true;
    if (s.size() > (size_t) INT_MAX) return false;
    const int n = MultiByteToWideChar(CP_UTF8, MB_ERR_INVALID_CHARS, s.data(), (int) s.size(), nullptr, 0);
    if (n <= 0) return false;
    w.assign((size_t) n, L'\0');
    return MultiByteToWideChar(CP_UTF8, MB_ERR_INVALID_CHARS, s.data(), (int) s.size(), w.data(), n) == n;
}

SessionError win_kind(DWORD code) {
    switch (code) {
        case ERROR_DISK_FULL: case ERROR_HANDLE_DISK_FULL: case ERROR_FILE_TOO_LARGE: return SessionError::storage;
        case ERROR_NOT_ENOUGH_MEMORY: case ERROR_OUTOFMEMORY: return SessionError::memory;
        default: return SessionError::io;
    }
}

std::string last_error(DWORD code = GetLastError()) {
    char msg[256] = {};
    const DWORD n = FormatMessageA(FORMAT_MESSAGE_FROM_SYSTEM | FORMAT_MESSAGE_IGNORE_INSERTS, nullptr, code, 0, msg,
                                   sizeof msg, nullptr);
    std::string s(msg, n);
    while (!s.empty() && (s.back() == '\n' || s.back() == '\r' || s.back() == ' ' || s.back() == '.')) s.pop_back();
    return s.empty() ? "error " + std::to_string(code) : s;
}

class RawFile {
public:
    ~RawFile() { close(); }
    // a new file that did not exist (CREATE_NEW: never an existing file or link)
    bool create_new(const std::wstring& path) {
        h_ = CreateFileW(path.c_str(), GENERIC_WRITE, 0, nullptr, CREATE_NEW,
                         FILE_ATTRIBUTE_NORMAL | FILE_FLAG_SEQUENTIAL_SCAN, nullptr);
        if (h_ == INVALID_HANDLE_VALUE) { { const DWORD c_ = GetLastError(); kind_ = win_kind(c_); error_ = last_error(c_); } return false; }
        return true;
    }
    // a session to restore: the name itself is opened, never what a link or reparse point there points to; it must be
    // an ordinary disk file with one name
    bool open_session(const std::string& path, uint64_t& size, bool = true) {
        std::wstring w;
        if (!wide(path, w)) { refused_ = true; error_ = "the path is not valid UTF-8"; return false; }
        h_ = CreateFileW(w.c_str(), GENERIC_READ, FILE_SHARE_READ, nullptr, OPEN_EXISTING,
                         FILE_FLAG_OPEN_REPARSE_POINT | FILE_FLAG_SEQUENTIAL_SCAN, nullptr);
        if (h_ == INVALID_HANDLE_VALUE) { { const DWORD c_ = GetLastError(); kind_ = win_kind(c_); error_ = last_error(c_); } return false; }
        BY_HANDLE_FILE_INFORMATION info{};
        if (!GetFileInformationByHandle(h_, &info)) { { const DWORD c_ = GetLastError(); kind_ = win_kind(c_); error_ = last_error(c_); } return false; }
        if (info.dwFileAttributes & FILE_ATTRIBUTE_REPARSE_POINT) { refused_ = true; error_ = "a link or reparse point, not a file"; return false; }
        if ((info.dwFileAttributes & FILE_ATTRIBUTE_DIRECTORY) || GetFileType(h_) != FILE_TYPE_DISK) {
            refused_ = true;
            error_ = "not a regular file";
            return false;
        }
        if (info.nNumberOfLinks != 1) { refused_ = true; error_ = "the file has more than one name (hard link)"; return false; }
        size = (uint64_t(info.nFileSizeHigh) << 32) | info.nFileSizeLow;
        return true;
    }
    // a model input: links followed (model folders are often linked)
    bool open_model(const std::string& path, uint64_t& size) {
        std::wstring w;
        if (!wide(path, w)) { refused_ = true; error_ = "the path is not valid UTF-8"; return false; }
        h_ = CreateFileW(w.c_str(), GENERIC_READ, FILE_SHARE_READ, nullptr, OPEN_EXISTING, FILE_ATTRIBUTE_NORMAL,
                         nullptr);
        if (h_ == INVALID_HANDLE_VALUE) {
            const DWORD code = GetLastError();
            missing_ = code == ERROR_FILE_NOT_FOUND || code == ERROR_PATH_NOT_FOUND;
            kind_ = win_kind(code); error_ = last_error(code);
            return false;
        }
        LARGE_INTEGER s{};
        if (!GetFileSizeEx(h_, &s)) { { const DWORD c_ = GetLastError(); kind_ = win_kind(c_); error_ = last_error(c_); } return false; }
        size = (uint64_t) s.QuadPart;
        return true;
    }
    // a spill chain's append log: an existing file of ours opened read+write, or created (the append path).  No
    // reparse-point follow: a link where the log should be is refused, never written through.
    bool open_append(const std::string& path, uint64_t& size) {
        std::wstring w;
        if (!wide(path, w)) { refused_ = true; error_ = "the path is not valid UTF-8"; return false; }
        h_ = CreateFileW(w.c_str(), GENERIC_READ | GENERIC_WRITE, FILE_SHARE_READ, nullptr, OPEN_ALWAYS,
                         FILE_ATTRIBUTE_NORMAL | FILE_FLAG_OPEN_REPARSE_POINT | FILE_FLAG_SEQUENTIAL_SCAN, nullptr);
        if (h_ == INVALID_HANDLE_VALUE) { { const DWORD c_ = GetLastError(); kind_ = win_kind(c_); error_ = last_error(c_); } return false; }
        BY_HANDLE_FILE_INFORMATION info{};
        if (!GetFileInformationByHandle(h_, &info)) { { const DWORD c_ = GetLastError(); kind_ = win_kind(c_); error_ = last_error(c_); } return false; }
        if (info.dwFileAttributes & FILE_ATTRIBUTE_REPARSE_POINT) { refused_ = true; error_ = "a link or reparse point, not a file"; return false; }
        if ((info.dwFileAttributes & FILE_ATTRIBUTE_DIRECTORY) || GetFileType(h_) != FILE_TYPE_DISK) {
            refused_ = true;
            error_ = "not a regular file";
            return false;
        }
        if (info.nNumberOfLinks != 1) { refused_ = true; error_ = "the file has more than one name (hard link)"; return false; }
        size = (uint64_t(info.nFileSizeHigh) << 32) | info.nFileSizeLow;
        return true;
    }
    bool is_open() const { return h_ != INVALID_HANDLE_VALUE; }
    bool direct() const { return false; }
    bool seek(uint64_t offset) {
        LARGE_INTEGER at{};
        at.QuadPart = (LONGLONG) offset;
        if (SetFilePointerEx(h_, at, nullptr, FILE_BEGIN)) return true;
        { const DWORD c_ = GetLastError(); kind_ = win_kind(c_); error_ = last_error(c_); }
        return false;
    }
    bool write_all(const uint8_t* p, size_t n) {
        while (n) {
            DWORD w = 0;
            const DWORD want = (DWORD) std::min<size_t>(n, size_t(1) << 30);
            if (!WriteFile(h_, p, want, &w, nullptr) || w == 0) { { const DWORD c_ = GetLastError(); kind_ = win_kind(c_); error_ = last_error(c_); } return false; }
            p += w; n -= w;
        }
        return true;
    }
    // bytes read, 0 at the end of the file, -1 on error
    long long read_some(uint8_t* p, size_t n) {
        DWORD r = 0;
        if (!ReadFile(h_, p, (DWORD) std::min<size_t>(n, size_t(1) << 30), &r, nullptr)) {
            { const DWORD c_ = GetLastError(); kind_ = win_kind(c_); error_ = last_error(c_); }
            return -1;
        }
        return (long long) r;
    }
    bool read_at(uint64_t offset, uint8_t* p, size_t n) {
        LARGE_INTEGER at{};
        at.QuadPart = (LONGLONG) offset;
        if (!SetFilePointerEx(h_, at, nullptr, FILE_BEGIN)) { { const DWORD c_ = GetLastError(); kind_ = win_kind(c_); error_ = last_error(c_); } return false; }
        while (n) {
            const long long r = read_some(p, n);
            if (r <= 0) { if (r == 0) error_ = "file ends early"; return false; }
            p += r; n -= (size_t) r;
        }
        return true;
    }
    bool truncate(uint64_t n) {
        LARGE_INTEGER at{};
        at.QuadPart = (LONGLONG) n;
        if (SetFilePointerEx(h_, at, nullptr, FILE_BEGIN) && SetEndOfFile(h_)) return true;
        { const DWORD c_ = GetLastError(); kind_ = win_kind(c_); error_ = last_error(c_); }
        return false;
    }
    bool sync() {
        if (FlushFileBuffers(h_)) return true;
        { const DWORD c_ = GetLastError(); kind_ = win_kind(c_); error_ = last_error(c_); }
        return false;
    }
    bool close() {
        if (h_ == INVALID_HANDLE_VALUE) return true;
        const bool ok = CloseHandle(h_) != 0;
        if (!ok) { const DWORD c_ = GetLastError(); kind_ = win_kind(c_); error_ = last_error(c_); }
        h_ = INVALID_HANDLE_VALUE;
        return ok;
    }
    const std::string& error() const { return error_; }
    bool missing() const { return missing_; }
    SessionError kind() const { return kind_; }
    // a validation refusal (a link, not a regular file, ...): bad input, not an I/O failure
    bool refused() const { return refused_; }
private:
    HANDLE h_ = INVALID_HANDLE_VALUE;
    SessionError kind_ = SessionError::io;
    bool refused_ = false;
    bool missing_ = false;
    std::string error_;
};

// Where the file is published.  MoveFileExW replaces an existing destination (std::rename does not on Windows); the
// replacement is a rename on one volume, but Windows promises no transaction on every filesystem.  A directory cannot
// be flushed on Windows: the file's data was flushed (FlushFileBuffers) before the move, and the move is write-through.
class Publisher {
public:
    bool open(const std::string& path, std::string& error) {
        std::string dir, leaf;
        if (!split_path(path, dir, leaf, error)) return false;
        std::wstring wdir, wleaf;
        if (!wide(dir, wdir) || !wide(leaf, wleaf)) { error = "session file: the path is not valid UTF-8"; return false; }
        prefix_ = wdir.empty() ? std::wstring() : (wdir.back() == L'\\' || wdir.back() == L'/' ? wdir : wdir + L"\\");
        final_ = prefix_ + wleaf;
        leaf_ = leaf;
        return true;
    }
    bool create_temp(RawFile& f, std::string& error) {
        for (int attempt = 0; attempt < 8; ++attempt) {
            std::wstring name;
            wide(temp_name(leaf_), name);
            temp_ = prefix_ + name;
            if (f.create_new(temp_)) { own_ = true; return true; }
            if (GetLastError() != ERROR_FILE_EXISTS && GetLastError() != ERROR_ALREADY_EXISTS) break;
        }
        error = f.error();
        return false;
    }
    bool publish(std::string& error, bool, SessionStatus& st, const std::function<int(const char*)>* fault) {
        if (const int e = injected(fault, "rename")) { error = std::strerror(e); st.error = errno_kind(e); return false; }
        if (MoveFileExW(temp_.c_str(), final_.c_str(), MOVEFILE_REPLACE_EXISTING | MOVEFILE_WRITE_THROUGH)) {
            own_ = false;
            st.published = true;
            // Windows has no folder flush; a test can still make this step fail after the move
            if (const int e = injected(fault, "dir_flush")) {
                error = std::string("flushing the directory: ") + std::strerror(e);
                st.error = errno_kind(e);
                return false;
            }
            return true;
        }
        const DWORD c = GetLastError();
        st.error = win_kind(c);
        error = last_error(c);
        return false;
    }
    SessionError kind() const { return SessionError::io; }
    ~Publisher() { if (own_) DeleteFileW(temp_.c_str()); }   // only the temporary file this call created
private:
    std::wstring prefix_, final_, temp_;
    std::string leaf_;
    bool own_ = false;
};

bool free_space(const std::string& path, uint64_t& out) {
    std::wstring w;
    if (!wide(session_free_space_dir(path, true), w)) return false;
    ULARGE_INTEGER avail{};
    if (!GetDiskFreeSpaceExW(w.empty() ? nullptr : w.c_str(), &avail, nullptr, nullptr)) return false;
    out = avail.QuadPart;
    return true;
}
#else
#ifndef O_DIRECT
#define O_DIRECT 0            // not offered by this system: buffered I/O
#endif

bool buffered_forced() {
    const char* e = std::getenv("STRATA_SESSION_BUFFERED");
    return e != nullptr && e[0] == '1';
}

static_assert(sizeof(off_t) >= 8, "session files need 64-bit file offsets (large-file support)");

class RawFile {
public:
    ~RawFile() { close(); }
    // a new file that did not exist (O_CREAT|O_EXCL: never an existing file or link), readable by its owner only
    bool create_new(int dirfd, const std::string& name) {
        fd_ = ::openat(dirfd, name.c_str(), O_WRONLY | O_CREAT | O_EXCL | O_NOFOLLOW | O_CLOEXEC, 0600);
        if (fd_ < 0) { kind_ = errno_kind(errno); error_ = std::strerror(errno); return false; }
        try_direct();
        return true;
    }
    // a session to restore: the name itself is opened, never what a symbolic link there points to; it must be a
    // regular file with one name (no hard link to a file elsewhere).  Checked on the opened descriptor. Delta readers
    // pass false: their scalar fields use arbitrary byte offsets that cannot satisfy O_DIRECT alignment.
    bool open_session(const std::string& path, uint64_t& size, bool allow_direct = true) {
        fd_ = ::open(path.c_str(), O_RDONLY | O_NOFOLLOW | O_NONBLOCK | O_CLOEXEC);   // NONBLOCK: a FIFO cannot hang
        if (fd_ < 0) {
            refused_ = errno == ELOOP;
            kind_ = errno_kind(errno);
            error_ = errno == ELOOP ? std::string("a symbolic link, not a file") : std::string(std::strerror(errno));
            return false;
        }
        struct stat st {};
        if (::fstat(fd_, &st) != 0) { error_ = std::strerror(errno); return false; }
        if (!S_ISREG(st.st_mode)) { refused_ = true; error_ = "not a regular file"; return false; }
        if (st.st_nlink != 1) { refused_ = true; error_ = "the file has more than one name (hard link)"; return false; }
        const int fl = ::fcntl(fd_, F_GETFL);
        if (fl >= 0) ::fcntl(fd_, F_SETFL, fl & ~O_NONBLOCK);
        size = (uint64_t) st.st_size;
        if (allow_direct) try_direct();
        return true;
    }
    // a model input: links followed (model folders are often linked)
    bool open_model(const std::string& path, uint64_t& size) {
        fd_ = ::open(path.c_str(), O_RDONLY | O_CLOEXEC);
        if (fd_ < 0) { missing_ = errno == ENOENT; kind_ = errno_kind(errno); error_ = std::strerror(errno); return false; }
        struct stat st {};
        if (::fstat(fd_, &st) != 0) { error_ = std::strerror(errno); return false; }
        if (!S_ISREG(st.st_mode)) { error_ = "not a regular file"; return false; }
        size = (uint64_t) st.st_size;
        return true;
    }
    // a spill chain's append log: an existing file of ours opened read+write, or created (the append path).  O_NOFOLLOW
    // (a symbolic link where the log should be is refused), no O_DIRECT (records start on arbitrary offsets).
    bool open_append(const std::string& path, uint64_t& size) {
        fd_ = ::open(path.c_str(), O_RDWR | O_CREAT | O_NOFOLLOW | O_CLOEXEC, 0600);
        if (fd_ < 0) {
            refused_ = errno == ELOOP;
            kind_ = errno_kind(errno);
            error_ = errno == ELOOP ? std::string("a symbolic link, not a file") : std::string(std::strerror(errno));
            return false;
        }
        struct stat st {};
        if (::fstat(fd_, &st) != 0) { kind_ = errno_kind(errno); error_ = std::strerror(errno); return false; }
        if (!S_ISREG(st.st_mode)) { refused_ = true; error_ = "not a regular file"; return false; }
        if (st.st_nlink != 1) { refused_ = true; error_ = "the file has more than one name (hard link)"; return false; }
        size = (uint64_t) st.st_size;
        return true;
    }
    bool is_open() const { return fd_ >= 0; }
    bool direct() const { return direct_; }
    bool seek(uint64_t offset) {
        if (::lseek(fd_, (off_t) offset, SEEK_SET) >= 0) return true;
        kind_ = errno_kind(errno);
        error_ = std::strerror(errno);
        return false;
    }
    bool write_all(const uint8_t* p, size_t n) {
        while (n) {
            const ssize_t w = ::write(fd_, p, n);
            if (w < 0 && errno == EINTR) continue;
            // a filesystem that took O_DIRECT at open but refuses this transfer: nothing was written, go buffered
            if (w < 0 && errno == EINVAL && direct_ && leave_direct()) continue;
            if (w <= 0) { kind_ = w < 0 ? errno_kind(errno) : SessionError::io; error_ = w < 0 ? std::strerror(errno) : "short write"; return false; }
            p += w; n -= (size_t) w;
        }
        return true;
    }
    // bytes read, 0 at the end of the file, -1 on error
    long long read_some(uint8_t* p, size_t n) {
        for (;;) {
            const ssize_t r = ::read(fd_, p, n);
            if (r < 0 && errno == EINTR) continue;
            if (r < 0 && errno == EINVAL && direct_ && leave_direct()) continue;
            if (r < 0) { kind_ = errno_kind(errno); error_ = std::strerror(errno); }
            return (long long) r;
        }
    }
    bool read_at(uint64_t offset, uint8_t* p, size_t n) {
        while (n) {
            const ssize_t r = ::pread(fd_, p, n, (off_t) offset);
            if (r < 0 && errno == EINTR) continue;
            if (r <= 0) { kind_ = r < 0 ? errno_kind(errno) : SessionError::io; error_ = r < 0 ? std::strerror(errno) : "file ends early"; return false; }
            p += r; n -= (size_t) r; offset += (uint64_t) r;
        }
        return true;
    }
    bool truncate(uint64_t n) {
        if (::ftruncate(fd_, (off_t) n) == 0) return true;
        kind_ = errno_kind(errno);
        error_ = std::strerror(errno);
        return false;
    }
    bool sync() {
        if (::fsync(fd_) == 0) return true;
        kind_ = errno_kind(errno);
        error_ = std::strerror(errno);
        return false;
    }
    bool close() {
        if (fd_ < 0) return true;
        const bool ok = ::close(fd_) == 0;
        if (!ok) { kind_ = errno_kind(errno); error_ = std::strerror(errno); }
        fd_ = -1;
        return ok;
    }
    const std::string& error() const { return error_; }
    bool missing() const { return missing_; }
    SessionError kind() const { return kind_; }
    // a validation refusal (a link, not a regular file, ...): bad input, not an I/O failure
    bool refused() const { return refused_; }
private:
    // O_DIRECT is switched on the open descriptor: a filesystem that refuses it (EINVAL, e.g. tmpfs) keeps buffered
    // I/O, and the file is never opened twice
    void try_direct() {
        if (O_DIRECT == 0 || buffered_forced()) return;
        const int fl = ::fcntl(fd_, F_GETFL);
        direct_ = fl >= 0 && ::fcntl(fd_, F_SETFL, fl | O_DIRECT) == 0;
    }
    bool leave_direct() {
        const int fl = ::fcntl(fd_, F_GETFL);
        if (fl < 0 || ::fcntl(fd_, F_SETFL, fl & ~O_DIRECT) != 0) return false;
        direct_ = false;
        return true;
    }
    int fd_ = -1;
    bool direct_ = false, missing_ = false, refused_ = false;
    SessionError kind_ = SessionError::io;
    std::string error_;
};

// The destination directory is opened once; the temporary file is created in it exclusively, renamed over the
// final name in the same directory (a symbolic link at the final name is replaced, never written through), and the
// directory is flushed after the rename so the new name survives a power loss.
class Publisher {
public:
    bool open(const std::string& path, std::string& error) {
        std::string dir;
        if (!split_path(path, dir, leaf_, error)) return false;
        dirfd_ = ::open(dir.empty() ? "." : dir.c_str(), O_RDONLY | O_DIRECTORY | O_CLOEXEC);
        if (dirfd_ < 0) {
            kind_ = errno_kind(errno);
            error = "session file: cannot open the directory of " + path + ": " + std::strerror(errno);
            return false;
        }
        return true;
    }
    bool create_temp(RawFile& f, std::string& error) {
        for (int attempt = 0; attempt < 8; ++attempt) {
            temp_ = temp_name(leaf_);
            if (f.create_new(dirfd_, temp_)) { own_ = true; return true; }
            if (errno != EEXIST) break;
        }
        error = f.error();
        return false;
    }
    bool publish(std::string& error, bool durable, SessionStatus& st, const std::function<int(const char*)>* fault) {
        int e = injected(fault, "rename");
        if (e || ::renameat(dirfd_, temp_.c_str(), dirfd_, leaf_.c_str()) != 0) {
            if (!e) e = errno;
            error = std::strerror(e);
            st.error = errno_kind(e);
            return false;
        }
        own_ = false;
        st.published = true;   // the new file is at its name now: a failure below cannot bring the old one back
        if (!durable) return true;
        e = injected(fault, "dir_flush");
        if (!e && ::fsync(dirfd_) != 0) e = errno;
        if (e == EINVAL) { st.dir_flush_unsupported = true; return true; }   // this filesystem cannot flush a folder
        if (e) {
            error = std::string("flushing the directory: ") + std::strerror(e);
            st.error = errno_kind(e);
            return false;
        }
        return true;
    }
    SessionError kind() const { return kind_; }
    ~Publisher() {
        if (own_) ::unlinkat(dirfd_, temp_.c_str(), 0);   // only the temporary file this call created
        if (dirfd_ >= 0) ::close(dirfd_);
    }
private:
    int dirfd_ = -1;
    std::string leaf_, temp_;
    bool own_ = false;
    SessionError kind_ = SessionError::io;
};

bool free_space(const std::string& path, uint64_t& out) {
    struct statvfs s {};
    if (::statvfs(session_free_space_dir(path, false).c_str(), &s) != 0) return false;
    out = (uint64_t) s.f_bavail * (uint64_t) s.f_frsize;
    return true;
}
#endif

class FileSink {
public:
    explicit FileSink(RawFile& f) : buf_(aligned_block()), f_(f) {}
    bool ok() const { return buf_ != nullptr; }
    std::string error() const {
        if (!buf_) return "out of memory for the write buffer";
        return fault_ ? fault_msg_ : f_.error();
    }
    SessionError kind() const { return buf_ ? (fault_ ? fault_kind_ : f_.kind()) : SessionError::memory; }
    // room in the block for in-place production (avoids a staging copy)
    uint8_t* span(size_t& room) { room = kBlock - used_; return buf_.get() + used_; }
    bool commit(size_t n) { used_ += n; total_ += n; return used_ < kBlock || drain(); }
    bool write(const void* p, size_t n) {
        const uint8_t* s = static_cast<const uint8_t*>(p);
        while (n) {
            size_t room = 0;
            uint8_t* d = span(room);
            const size_t c = std::min(room, n);
            std::memcpy(d, s, c);
            if (!commit(c)) return false;
            s += c; n -= c;
        }
        return true;
    }
    // last block padded to the alignment for direct I/O, then the file cut back to its exact size and flushed
    bool finish(bool durable) {
        if (!f_.is_open()) return false;
        if (fault_) { f_.close(); return false; }
        const auto finish_started = timings ? TimingClock::now() : TimingClock::time_point{};
        const size_t padded = f_.direct() ? (used_ + kAlign - 1) / kAlign * kAlign : used_;
        std::memset(buf_.get() + used_, 0, padded - used_);
        const auto write_started = timings ? TimingClock::now() : TimingClock::time_point{};
        bool ok = f_.write_all(buf_.get(), padded);
        if (timings) timing_add(&timings->os_write_ns, write_started);
        if (ok && used_ && progress) (*progress)(total_, expected);   // the last, partial block
        ok = ok && (padded == used_ || f_.truncate(total_));
        if (ok && durable && phase) phase("flush", total_);
        if (ok && durable && flush_fault) { fail_with(flush_fault); ok = false; }
        ok = ok && (!durable || f_.sync());
        ok = f_.close() && ok;
        if (timings) timing_add(&timings->finish_ns, finish_started);
        return ok;
    }
    std::optional<Ticker> progress;
    SessionPhase phase;
    uint64_t expected = 0;
    int write_fault = 0, flush_fault = 0;   // tests: errno values injected at the first block write / the file flush
    SessionWriteTimings* timings = nullptr;
private:
    void fail_with(int e) { fault_ = true; fault_kind_ = errno_kind(e); fault_msg_ = std::strerror(e); }
    bool drain() {
        if (write_fault) { fail_with(write_fault); return false; }
        const auto started = timings ? TimingClock::now() : TimingClock::time_point{};
        const bool ok = f_.write_all(buf_.get(), used_);
        if (timings) timing_add(&timings->os_write_ns, started);
        if (!ok) return false;
        used_ = 0;
        if (progress) (*progress)(total_, expected);
        return true;
    }
    Aligned buf_;
    RawFile& f_;
    bool fault_ = false;
    SessionError fault_kind_ = SessionError::io;
    std::string fault_msg_;
    size_t used_ = 0;
    uint64_t total_ = 0;
};

// The background-state write sink (SessionWriteOptions::background_state): FileSink's producer face (span/commit/
// write/finish) over two pre-allocated 16 MiB aligned buffers. A single writer thread owns every OS write, the tail
// truncate/flush/close and the per-block progress reports; the producer fills and pads one buffer while the writer
// drains the other and blocks when both are queued (backpressure). Every wait's predicate includes failed_, so a
// writer-thread failure wakes a blocked producer instead of deadlocking; finish() and the destructor both join the
// thread (it is never detached). The writer reports block progress and the flush phase; the caller observes
// `timings` only after finish() returns, and the join orders the threads.
class AsyncFileSink {
public:
    explicit AsyncFileSink(RawFile& f) : f_(f) {
        slots_[0].buf = aligned_block();
        slots_[1].buf = aligned_block();
        if (!slots_[0].buf || !slots_[1].buf) return;   // ok() == false: report memory, start no thread
        started_ = true;
        th_ = std::thread([this] { run(); });
    }
    ~AsyncFileSink() {
        {
            std::lock_guard<std::mutex> lk(m_);
            quit_ = true;
            abandon_ = true;   // the file is unfinished: the Publisher (or the caller) removes the temporary
        }
        cv_.notify_all();
        if (th_.joinable()) th_.join();
    }
    AsyncFileSink(const AsyncFileSink&) = delete;
    AsyncFileSink& operator=(const AsyncFileSink&) = delete;
    bool ok() const { return started_; }
    std::string error() const {
        if (!started_) return "out of memory for the write buffers";
        std::lock_guard<std::mutex> lk(m_);
        return failed_ ? fault_msg_ : f_.error();
    }
    SessionError kind() const {
        if (!started_) return SessionError::memory;
        std::lock_guard<std::mutex> lk(m_);
        return failed_ ? fault_kind_ : f_.kind();
    }
    // room in the fill buffer for in-place production (avoids a staging copy), FileSink's contract
    uint8_t* span(size_t& room) {
        room = kBlock - slots_[fill_].used;
        return slots_[fill_].buf.get() + slots_[fill_].used;
    }
    bool commit(size_t n) {
        slots_[fill_].used += n;
        total_ += n;
        std::unique_lock<std::mutex> lk(m_);
        if (failed_) return false;
        if (slots_[fill_].used < kBlock) return true;
        pend_[(pend_head_ + pend_count_) % 2] = fill_;
        ++pend_count_;
        cv_.notify_one();
        cv_.wait(lk, [&] { return failed_ || free_count_ != 0; });
        if (failed_) return false;
        fill_ = free_[free_head_];
        free_head_ = (free_head_ + 1) % 2;
        --free_count_;
        return true;
    }
    bool write(const void* p, size_t n) {
        const uint8_t* s = static_cast<const uint8_t*>(p);
        while (n) {
            size_t room = 0;
            uint8_t* d = span(room);
            const size_t c = std::min(room, n);
            std::memcpy(d, s, c);
            if (!commit(c)) return false;
            s += c; n -= c;
        }
        return true;
    }
    // the tail block (padded for direct I/O) is queued here; the truncate/flush/close run on the writer thread while
    // this waits for it, then the thread is joined
    bool finish(bool durable) {
        if (!started_) return false;
        const auto finish_started = timings ? TimingClock::now() : TimingClock::time_point{};
        {
            std::unique_lock<std::mutex> lk(m_);
            if (!failed_) {
                const size_t used = slots_[fill_].used;
                const size_t padded = f_.direct() ? (used + kAlign - 1) / kAlign * kAlign : used;
                if (padded) {
                    std::memset(slots_[fill_].buf.get() + used, 0, padded - used);
                    slots_[fill_].used = padded;
                    tail_slot_ = fill_;
                    pend_[(pend_head_ + pend_count_) % 2] = fill_;
                    ++pend_count_;
                }
                tail_used_ = used;
                finish_padded_ = padded;
                finish_total_ = total_;
                finish_durable_ = durable;
            }
            quit_ = true;
            cv_.notify_all();
            cv_.wait(lk, [&] { return done_; });
        }
        if (th_.joinable()) th_.join();
        const bool ok = !failed_ && finish_ok_;
        if (timings) timing_add(&timings->finish_ns, finish_started);
        return ok;
    }
    std::optional<Ticker> progress;
    SessionPhase phase;
    uint64_t expected = 0;
    int write_fault = 0, flush_fault = 0;   // tests: errno values injected at the first block write / the file flush
    SessionWriteTimings* timings = nullptr;
private:
    struct Slot { Aligned buf; size_t used = 0; };
    void fail_locked(SessionError kind, std::string msg) {
        if (!failed_) {
            failed_ = true;
            fault_kind_ = kind;
            fault_msg_ = std::move(msg);
        }
        cv_.notify_all();
    }
    void fail(SessionError kind, std::string msg) {
        std::lock_guard<std::mutex> lk(m_);
        fail_locked(kind, std::move(msg));
    }
    void run() {
        bool fault_applied = false;
        std::unique_lock<std::mutex> lk(m_);
        try {
            for (;;) {
                cv_.wait(lk, [&] { return quit_ || pend_count_ != 0; });
                if (pend_count_ == 0) break;   // quit with the queue drained
                const size_t idx = pend_[pend_head_];
                pend_head_ = (pend_head_ + 1) % 2;
                --pend_count_;
                const size_t used = slots_[idx].used;
                const bool tail = idx == tail_slot_;
                if (!failed_ && write_fault && !fault_applied) {
                    fault_applied = true;
                    fail_locked(errno_kind(write_fault), std::strerror(write_fault));
                }
                if (!failed_) {
                    const auto started = timings ? TimingClock::now() : TimingClock::time_point{};
                    lk.unlock();
                    const bool ok = f_.write_all(slots_[idx].buf.get(), used);
                    lk.lock();
                    if (timings) timing_add(&timings->os_write_ns, started);
                    if (!ok) fail_locked(f_.kind(), f_.error());
                    else {
                        written_ = tail ? finish_total_ : written_ + used;
                        if (progress) {
                            lk.unlock();
                            (*progress)(written_, expected);
                            lk.lock();
                        }
                    }
                }
                slots_[idx].used = 0;
                free_[(free_head_ + free_count_) % 2] = idx;
                ++free_count_;
                cv_.notify_all();
            }
            if (!abandon_) {
                lk.unlock();
                finish_steps();
                lk.lock();
            }
        } catch (...) {
            if (!lk.owns_lock()) lk.lock();
            fail_locked(SessionError::io, "session file: the background writer failed");
        }
        done_ = true;
        lk.unlock();
        if (f_.is_open()) f_.close();
        cv_.notify_all();
    }
    // the tail block's write happened in run(); here the file is cut back to its exact size, flushed and closed.
    // Called without m_ held (it invokes callbacks and fail()).
    void finish_steps() {
        bool ok = !failed_ && f_.is_open();
        if (ok && finish_padded_ != tail_used_) ok = f_.truncate(finish_total_);
        if (ok && finish_durable_ && phase) phase("flush", finish_total_);
        if (ok && finish_durable_ && flush_fault) {
            ok = false;
            fail(errno_kind(flush_fault), std::strerror(flush_fault));
        }
        if (ok && finish_durable_) ok = f_.sync();
        ok = f_.close() && ok;
        finish_ok_ = ok;
    }
    Slot slots_[2];
    RawFile& f_;
    std::thread th_;
    mutable std::mutex m_;   // error()/kind() are const like FileSink's
    std::condition_variable cv_;
    size_t fill_ = 0;
    size_t pend_[2] = {}, pend_head_ = 0, pend_count_ = 0;       // queued, unwritten slots (FIFO)
    size_t free_[2] = {1, 0}, free_head_ = 0, free_count_ = 1;   // slots the producer may fill (slot 0 is first)
    size_t tail_slot_ = SIZE_MAX;
    size_t tail_used_ = 0, finish_padded_ = 0;
    uint64_t total_ = 0, written_ = 0, finish_total_ = 0;
    bool started_ = false, quit_ = false, abandon_ = false, done_ = false, failed_ = false, finish_durable_ = false,
         finish_ok_ = false;
    SessionError fault_kind_ = SessionError::io;
    std::string fault_msg_;
};

class FileSource {
public:
    explicit FileSource(RawFile& f) : buf_(aligned_block()), f_(f) {}
    bool ok() const { return buf_ != nullptr; }
    std::string error() const { return buf_ ? f_.error() : std::string("out of memory for the read buffer"); }
    SessionError kind() const { return buf_ ? f_.kind() : SessionError::memory; }
    bool read(void* p, size_t n) {
        uint8_t* d = static_cast<uint8_t*>(p);
        while (n) {
            if (at_ == have_ && !fill()) return false;
            const size_t c = std::min(n, have_ - at_);
            std::memcpy(d, buf_.get() + at_, c);
            at_ += c; d += c; n -= c;
        }
        return true;
    }
    std::optional<Ticker> progress;
    uint64_t expected = 0;
private:
    bool fill() {
        at_ = have_ = 0;
        while (have_ < kBlock) {
            const long long r = f_.read_some(buf_.get() + have_, kBlock - have_);
            if (r < 0) return false;
            if (r == 0) break;
            have_ += (size_t) r;
            if (f_.direct() && have_ % kAlign) break;   // short aligned read: end of file
        }
        const uint64_t before = total_;
        total_ += have_;
        if (progress && total_ > before) (*progress)(total_, expected);
        return have_ > 0;
    }
    Aligned buf_;
    RawFile& f_;
    size_t at_ = 0, have_ = 0;
    uint64_t total_ = 0;
};

// ---- serialization: the same walk counts (f == nullptr) and writes
// The sources-released notification (SessionWriteOptions::on_sources_released): fires exactly once, at the point
// where no further SessionKvSource::read call can be made.  The v3 payload calls fire() right after its K/V section;
// every later exit (a failed source read, a write fault, a failed publish) relies on the destructor, so a caller
// blocked on the notification is woken on every path.  Never constructed for a refusal before any disk access.
struct SourcesReleased {
    explicit SourcesReleased(const std::function<void()>& fn) : fn_(&fn) {}
    ~SourcesReleased() { fire(); }
    void fire() {
        if (fn_ && *fn_ && !fired_) { fired_ = true; (*fn_)(); }
    }
private:
    const std::function<void()>* fn_;
    bool fired_ = false;
};

struct Out {
    FileSink* f = nullptr;
    AsyncFileSink* af = nullptr;   // set instead of f when the background-state sink owns the OS writes
    SessionHasher* hash = nullptr;
    SessionWriteTimings* timings = nullptr;
    SourcesReleased* released = nullptr;
    uint64_t n = 0;
    bool ok = true, overflow = false;
    void count(uint64_t c) {
        if (c > UINT64_MAX - n) { overflow = true; ok = false; return; }
        n += c;
    }
    // the walk addresses either sink through the same face; a sizing pass sets neither (counting only)
    bool writing() const { return f != nullptr || af != nullptr; }
    uint8_t* span(size_t& room) { return f ? f->span(room) : af->span(room); }
    bool commit(size_t n2) { return f ? f->commit(n2) : af->commit(n2); }
    bool sink_write(const void* p, size_t c) { return f ? f->write(p, c) : af->write(p, c); }
    void raw(const void* p, size_t c) {
        count(c);
        if (!writing() || !ok || !c) return;
        if (hash) hash->update(p, c);
        ok = sink_write(p, c);
    }
    void u64(uint64_t v) { raw(&v, 8); }
    void i64(int64_t v) { raw(&v, 8); }
    template<class T> void vec(const std::vector<T>& v) { u64(v.size()); raw(v.data(), v.size() * sizeof(T)); }
    void buffer(const ConversationBuffer& b) {
        u64(b.size());
        b.visit(0, b.size(), [&](const uint8_t* p, size_t c, size_t) { raw(p, c); return ok; });
    }
    // a K/V part pulled from its owner straight into the file block (sizing pass: counted, never read)
    void source(const SessionKvSource& s, size_t part) {
        const size_t size = s.sizes[part];
        u64(size);
        if (!writing()) { count(size); return; }
        read_from(s, part, 0, size);
    }
    // a delta record's K/V part: the offset it starts at, its length, and the bytes from there
    void source_tail(const SessionKvSource& s, size_t part) {
        const size_t full = s.sizes[part];
        const size_t first = std::min(s.first_bytes[part], full);
        const size_t size = full - first;
        u64(first);
        u64(size);
        if (!writing()) { count(size); return; }
        read_from(s, part, first, size);
    }
    bool read_failed = false;
    std::string read_error;
private:
    void read_from(const SessionKvSource& s, size_t part, size_t first, size_t size) {
        for (size_t at = 0; at < size && ok;) {
            size_t room = 0;
            uint8_t* d = span(room);
            const size_t c = std::min(room, size - at);
            const auto started = timings ? TimingClock::now() : TimingClock::time_point{};
            const bool read = s.read && s.read(part, first + at, d, c);
            if (timings) timing_add(&timings->kv_source_ns, started);
            if (!read) {
                ok = false;
                read_failed = true;
                if (s.error) read_error = s.error();
                return;
            }
            if (hash) hash->update(d, c);
            count(c);
            ok = ok && commit(c);
            at += c;
        }
    }
};

void put_kv_header(Out& o, int format, int64_t cells, int64_t heads, int64_t head_dim, int64_t page_size,
                   int64_t pooled_rows, int64_t idx_dim) {
    for (int64_t v : {int64_t(format), cells, heads, head_dim, page_size, pooled_rows, idx_dim}) o.i64(v);
}

// the meta prefix every layout shares: geometry, layer range, cvec.  In the v1/v2 payloads it opens the state
// section; in the K/V-first v3 payload it stands alone before the K/V.
void put_geo_prefix(Out& o, const SavedConversation& s) {
    for (int64_t g : s.geometry) o.i64(g);
    o.i64(s.layer_lo); o.i64(s.layer_hi); o.u64(s.cvec ? 1 : 0);
}

void put_checkpoint(Out& o, const ConversationCheckpoint& c) {
    o.vec(c.ids);
    o.u64(c.imgs.size());
    for (const auto& k : c.imgs) { o.i64(k.start); o.u64(k.hash); }
    o.vec(c.gdn); o.vec(c.ple); o.vec(c.tails); o.vec(c.dead); o.vec(c.block_pos);
    o.u64(c.used);
}

template<class A, class B>
bool bytes_equal(const std::vector<uint8_t, A>& a, const std::vector<uint8_t, B>& b) {
    return a.size() == b.size() && (a.empty() || std::memcmp(a.data(), b.data(), a.size()) == 0);
}

// the write side's all-five homomorphy rule: every running-state array equal in length AND bytes (direct comparison,
// no digest); ids/imgs/used do not take part
bool checkpoint_state_homomorphic(const ConversationCheckpoint& c, const ConversationCheckpoint& live) {
    return bytes_equal(c.gdn, live.gdn) && bytes_equal(c.ple, live.ple) && bytes_equal(c.tails, live.tails) &&
           bytes_equal(c.dead, live.dead) && bytes_equal(c.block_pos, live.block_pos);
}

using StateCodecArray = std::array<uint8_t, 5>;

// Codec choices are decided once before either sizing or writing.  REF takes precedence over RHT for a checkpoint;
// unsupported geometry, a mismatched GDN buffer, and non-finite state all remain RAW.
std::vector<StateCodecArray> state_codecs(const SavedConversation& s, bool enabled, const std::vector<char>& ref,
                                          rht::Layout& layout) {
    std::vector<StateCodecArray> codec(s.checkpoints.size() + 1);
    if (!enabled || !rht::layout_from(s.geometry, s.layer_lo, s.layer_hi, layout)) return codec;
    const auto select_gdn = [&](const ConversationCheckpoint& checkpoint, size_t index, bool is_ref) {
        if (is_ref || checkpoint.gdn.size() != layout.logical ||
            !rht::all_finite(checkpoint.gdn.data(), layout.logical)) return;
        codec[index][0] = 2;
    };
    select_gdn(s.live, 0, false);
    for (size_t i = 0; i < s.checkpoints.size(); ++i)
        select_gdn(s.checkpoints[i], i + 1, i < ref.size() && ref[i] != 0);
    return codec;
}

bool any_state_codec2(const std::vector<StateCodecArray>& codec) {
    return std::any_of(codec.begin(), codec.end(), [](const StateCodecArray& state) { return state[0] == 2; });
}

// v2: one running-state array as (codec, logical, stored, bytes[stored]); REF (codec 1) stores no bytes, its logical
// size is the live array's length and the reader copies the live bytes into an independent vector; RHT (codec 2)
// stores the encoded GDN array and decodes it into an independent logical-size vector.
template<class Vec>
void put_state_v2(Out& o, const Vec& v, uint8_t codec, const rht::Layout& layout) {
    o.u64(codec);
    if (codec == 2) {
        o.u64(layout.logical);
        o.u64(layout.stored);
        if (!o.writing()) {
            o.count(layout.stored);
        } else {
            const auto started = o.timings ? TimingClock::now() : TimingClock::time_point{};
            const bool encoded = rht::encode(v.data(), layout, [&](const void* data, size_t bytes) {
                o.raw(data, bytes);
                return o.ok;
            });
            if (o.timings) timing_add(&o.timings->rht_ns, started);
            if (!encoded) o.ok = false;
        }
        return;
    }
    o.u64((uint64_t) v.size());
    o.u64(codec == 1 ? 0 : (uint64_t) v.size());
    if (codec != 1) o.raw(v.data(), v.size());
}

void put_checkpoint_v2(Out& o, const ConversationCheckpoint& c, const StateCodecArray& codec,
                       const rht::Layout& layout) {
    o.vec(c.ids);
    o.u64(c.imgs.size());
    for (const auto& k : c.imgs) { o.i64(k.start); o.u64(k.hash); }
    put_state_v2(o, c.gdn, codec[0], layout);
    put_state_v2(o, c.ple, codec[1], layout);
    put_state_v2(o, c.tails, codec[2], layout);
    put_state_v2(o, c.dead, codec[3], layout);
    put_state_v2(o, c.block_pos, codec[4], layout);
    o.u64(c.used);
}

// the running state, v1 raw coding: the live checkpoint, then the count and the checkpoints
void put_states(Out& o, const SavedConversation& s) {
    put_checkpoint(o, s.live);
    o.u64(s.checkpoints.size());
    for (const auto& c : s.checkpoints) put_checkpoint(o, c);
}

// the running state, v2 descriptor coding.  `ref` holds one flag per checkpoint: all five arrays homomorphic to
// the live state's -> the whole checkpoint's five arrays become REF descriptors of it.
void put_states_v2(Out& o, const SavedConversation& s, const std::vector<StateCodecArray>& codec,
                   const std::vector<char>& ref, const rht::Layout& layout) {
    put_checkpoint_v2(o, s.live, codec[0], layout);
    o.u64(s.checkpoints.size());
    for (size_t i = 0; i < s.checkpoints.size(); ++i) {
        const bool is_ref = i < ref.size() && ref[i] != 0;
        StateCodecArray checkpoint_codec = codec[i + 1];
        if (is_ref) checkpoint_codec.fill(1);
        put_checkpoint_v2(o, s.checkpoints[i], checkpoint_codec, layout);
    }
}

// the K/V section: the layer count and, per layer, its header plus the five parts - streamed from `sources`, or
// the captured buffers of `kv`
void put_kv_sources(Out& o, const std::vector<SessionKvSource>& sources) {
    o.u64(sources.size());
    for (const auto& k : sources) {
        put_kv_header(o, k.format, k.cells, k.heads, k.head_dim, k.page_size, k.pooled_rows, k.idx_dim);
        for (size_t part = 0; part < 5; ++part) o.source(k, part);
    }
}

void put_kv_buffers(Out& o, const std::vector<ConversationKv>& kv) {
    o.u64(kv.size());
    for (const auto& k : kv) {
        put_kv_header(o, k.format, k.cells, k.heads, k.head_dim, k.page_size, k.pooled_rows, k.idx_dim);
        o.buffer(k.k); o.buffer(k.v); o.buffer(k.k_scale); o.buffer(k.v_scale); o.buffer(k.pooled);
    }
}

// a delta record's K/V section: per part the offset it starts at, its length, and the bytes from there
void put_kv_tails(Out& o, const std::vector<SessionKvSource>& kv) {
    o.u64(kv.size());
    for (const auto& k : kv) {
        put_kv_header(o, k.format, k.cells, k.heads, k.head_dim, k.page_size, k.pooled_rows, k.idx_dim);
        for (size_t part = 0; part < 5; ++part) o.source_tail(k, part);
    }
}

// `ref` holds one flag per checkpoint: all five arrays homomorphic to the live state's -> the whole checkpoint's
// five arrays become REF descriptors.  The K/V part is byte-identical to v1 (RAW, streamed from `sources` or the
// captured buffers); only the state arrays are descriptor-coded.  v1/v2 keep the state-first section order.
void put_payload_v2(Out& o, const SavedConversation& s, const std::vector<SessionKvSource>* sources,
                    const std::vector<char>& ref, const std::vector<StateCodecArray>& codec,
                    const rht::Layout& layout) {
    const auto state_started = o.timings ? TimingClock::now() : TimingClock::time_point{};
    put_geo_prefix(o, s);
    put_states_v2(o, s, codec, ref, layout);
    if (o.timings) timing_add(&o.timings->state_ns, state_started);
    const auto kv_started = o.timings ? TimingClock::now() : TimingClock::time_point{};
    if (sources) put_kv_sources(o, *sources);
    else put_kv_buffers(o, s.kv);
    if (o.timings) timing_add(&o.timings->kv_ns, kv_started);
}

// the K/V-first v3 payload: the shared geometry prefix, the K/V section (whose source reads finish here), the
// sources-released notification, then the same v2 state descriptors.
void put_payload_v3(Out& o, const SavedConversation& s, const std::vector<SessionKvSource>* sources,
                    const std::vector<char>& ref, const std::vector<StateCodecArray>& codec,
                    const rht::Layout& layout) {
    const auto state_started = o.timings ? TimingClock::now() : TimingClock::time_point{};
    put_geo_prefix(o, s);
    if (o.timings) timing_add(&o.timings->state_ns, state_started);
    const auto kv_started = o.timings ? TimingClock::now() : TimingClock::time_point{};
    if (sources) put_kv_sources(o, *sources);
    else put_kv_buffers(o, s.kv);
    if (o.timings) timing_add(&o.timings->kv_ns, kv_started);
    // every K/V source read is done and no more will be made: notify before the state section is serialized
    if (o.ok && o.released) o.released->fire();
    const auto states_started = o.timings ? TimingClock::now() : TimingClock::time_point{};
    put_states_v2(o, s, codec, ref, layout);
    if (o.timings) timing_add(&o.timings->state_ns, states_started);
}

void put_payload(Out& o, const SavedConversation& s, const std::vector<SessionKvSource>* sources = nullptr) {
    const auto state_started = o.timings ? TimingClock::now() : TimingClock::time_point{};
    put_geo_prefix(o, s);
    put_states(o, s);
    if (o.timings) timing_add(&o.timings->state_ns, state_started);
    const auto kv_started = o.timings ? TimingClock::now() : TimingClock::time_point{};
    if (sources) put_kv_sources(o, *sources);
    else put_kv_buffers(o, s.kv);
    if (o.timings) timing_add(&o.timings->kv_ns, kv_started);
}

// ---- delta records (--conversation-cache-disk-delta): the same framing as a session file, then the K/V each
// part as (offset, size, bytes).  The payload hash covers the record's header fields too, so an edit anywhere in it
// is refused by the read pass.  Versions 1/2 put the state section first (like the v1/v2 files); version 3 is the
// K/V-first layout, the record form of put_payload_v3.
constexpr uint64_t kDeltaVersionV1 = 1;
constexpr uint64_t kDeltaVersion = 2;
constexpr uint64_t kDeltaVersionV3 = 3;
constexpr char kDeltaMagic[8] = {'S', 'T', 'R', 'S', 'D', 'L', 'T', '\x01'};
// one K/V layer's header (7 words)
constexpr uint64_t kKvHeaderBytes = 7 * 8;

void put_delta_record(Out& o, const SessionFileIdentity& id, int64_t first_token, const SavedConversation& s,
                      const std::vector<SessionKvSource>& kv, uint64_t version,
                      const std::vector<StateCodecArray>& codec, const rht::Layout& layout) {
    const auto state_started = o.timings ? TimingClock::now() : TimingClock::time_point{};
    o.raw(kDeltaMagic, 8);
    o.u64(version);
    o.u64(id.model); o.u64(id.config);
    o.i64(first_token);
    put_geo_prefix(o, s);
    if (o.timings) timing_add(&o.timings->state_ns, state_started);
    const auto kv_started = o.timings ? TimingClock::now() : TimingClock::time_point{};
    if (version == kDeltaVersionV3) {
        put_kv_tails(o, kv);
        if (o.timings) timing_add(&o.timings->kv_ns, kv_started);
        // every K/V source read is done: notify before the state section is serialized
        if (o.ok && o.released) o.released->fire();
        const auto states_started = o.timings ? TimingClock::now() : TimingClock::time_point{};
        put_states_v2(o, s, codec, {}, layout);
        if (o.timings) timing_add(&o.timings->state_ns, states_started);
        return;
    }
    // versions 1 and 2 keep the states-first section order, exactly as written before
    const auto states_started = o.timings ? TimingClock::now() : TimingClock::time_point{};
    if (version == kDeltaVersion) put_states_v2(o, s, codec, {}, layout);
    else put_states(o, s);
    if (o.timings) timing_add(&o.timings->state_ns, states_started);
    const auto kv2_started = o.timings ? TimingClock::now() : TimingClock::time_point{};
    put_kv_tails(o, kv);
    if (o.timings) timing_add(&o.timings->kv_ns, kv2_started);
}

// ---- parsing: every count is checked against the bytes left and the caller's limits before anything is allocated
struct In {
    FileSource* f;
    SessionHasher hash;
    uint64_t left;
    std::string& error;
    const SessionReadLimits& limits;
    // Streaming K/V sizes: `record_kv` receives them (the read pass); `verify_kv`, set from the read pass's list,
    // makes this walk refuse a file whose layer count or any part size differs (the apply pass).  At most one set.
    std::vector<std::array<uint64_t, 5>>* record_kv = nullptr;
    const std::vector<std::array<uint64_t, 5>>* verify_kv = nullptr;
    SessionError kind = SessionError::invalid;
    size_t kv_layer = 0;
    // format v2: the upfront read ask is an initial screen, not a reservation. REF and RHT allocate full logical
    // outputs, which resize_admitted prices once per allocation; descriptor handling must not repeat the aggregate
    // peak ask after earlier allocations. REF arrays copy live bytes into independent vectors.
    const ConversationCheckpoint* ref_live = nullptr;
    // the digest's: set when a state descriptor is REF or RHT (RHT decodes lossily, so its state cannot be compared)
    bool* saw_ref = nullptr;
    bool* saw_rht = nullptr;
    bool fail(const std::string& m) { if (error.empty()) error = "session file: " + m; return false; }
    bool raw(void* p, size_t c) {
        if (c > left) return fail("payload ends early");
        if (c && !f->read(p, c)) { kind = f->kind(); return fail("read error: " + f->error()); }
        hash.update(p, c);
        left -= c;
        return true;
    }
    bool u64(uint64_t& v) { return raw(&v, 8); }
    bool i64(int64_t& v) { return raw(&v, 8); }
    // `elem`: the fewest file bytes one element takes; `limit`: the most elements the caller accepts
    bool count(uint64_t& n, size_t elem, uint64_t limit = UINT64_MAX) {
        if (!u64(n)) return false;
        if (elem && n > left / elem) return fail("element count exceeds the payload");
        if (n > limit) return fail("element count " + std::to_string(n) + " exceeds this engine's limit " +
                                   std::to_string(limit));
        if (n > (uint64_t) std::numeric_limits<size_t>::max() / (elem ? elem : 1)) return fail("element count too large");
        return true;
    }
    // the same checks as count() for a number already read from the stream (v2 descriptors carry their counts):
    // validates it without reading anything, so the caller's decoded value is never overwritten by a second read
    bool bound(uint64_t n, size_t elem, uint64_t limit = UINT64_MAX) {
        if (elem && n > left / elem) return fail("element count exceeds the payload");
        if (n > limit) return fail("element count " + std::to_string(n) + " exceeds this engine's limit " +
                                   std::to_string(limit));
        if (n > (uint64_t) std::numeric_limits<size_t>::max() / (elem ? elem : 1)) return fail("element count too large");
        return true;
    }
    // The v3 apply walk's states section: `n` bytes feed the running hash with NO decode and no destination - the
    // same bytes a full decode would read and hash, so the trailing-hash check covers them identically.  `scratch` is
    // a reusable read block (the K/V walk's 16 MiB block); `n` must not exceed the payload bytes left.
    bool hash_only(uint8_t* scratch, size_t cap, uint64_t n) {
        if (n > left) return fail("payload ends early");
        while (n) {
            const size_t c = (size_t) std::min<uint64_t>(cap, n);
            if (!f->read(scratch, c)) { kind = f->kind(); return fail("read error: " + f->error()); }
            hash.update(scratch, c);
            left -= c;
            n -= c;
        }
        return true;
    }
    template<class T, class A> bool resize(std::vector<T, A>& v, uint64_t n) {
        v.resize((size_t) n);
        return true;
    }
    template<class T> bool vec(std::vector<T>& v, uint64_t limit = UINT64_MAX) {
        uint64_t n = 0;
        if (!count(n, sizeof(T), limit) || !resize(v, n)) return false;
        return raw(v.data(), (size_t) n * sizeof(T));
    }
    // ---- format v2 only: admission before every count-based allocation (the DeltaIn reader's checking style),
    // covering ids/imgs, the live RAW and the checkpoint RAW/REF state arrays, the checkpoint and K/V directories,
    // the whole-read ConversationBuffer parts and the streamed working block.  Page rounding, multiply-add overflow
    // and the container's max_size are checked before the allocation.
    bool admit_need(uint64_t need) {
        if (need && limits.admit) {
            std::string why;
            if (!limits.admit(need, why)) {
                kind = SessionError::memory;
                if (error.empty()) error = "session file: " +
                    (why.empty() ? std::string("RAM admission refused") : why);
                return false;
            }
        }
        return true;
    }
    bool prepare_allocation(uint64_t count, size_t element_size) {
        if (element_size && count > (uint64_t) std::numeric_limits<size_t>::max() / element_size)
            return fail("allocation size exceeds this build");
        const uint64_t need = allocation_bytes(count, element_size);
        if (need == UINT64_MAX) return fail("allocation size overflows");
        return admit_need(need);
    }
    template<class T, class A> bool resize_admitted(std::vector<T, A>& v, uint64_t n) {
        if (n > v.max_size()) return fail("element count exceeds this container's maximum");
        if (!prepare_allocation(n, sizeof(T))) return false;
        const size_t count = (size_t) n;
        if (count) v.reserve(count);   // these parser-owned vectors start empty; no geometric growth or old/new overlap
        v.resize(count);
        return true;
    }
    template<class T, class A> bool vec_admitted(std::vector<T, A>& v, uint64_t limit = UINT64_MAX) {
        uint64_t n = 0;
        if (!count(n, sizeof(T), limit) || !resize_admitted(v, n)) return false;
        return raw(v.data(), (size_t) n * sizeof(T));
    }
    bool buffer_admitted(ConversationBuffer& b, size_t part) {
        uint64_t n = 0;
        const uint64_t limit = kv_layer < limits.max_kv_bytes.size() ? limits.max_kv_bytes[kv_layer][part] : UINT64_MAX;
        if (!count(n, 1, limit)) return false;
        const size_t peak = b.allocation_peak((size_t) n);
        if (!admit_need(peak == SIZE_MAX ? UINT64_MAX : (uint64_t) peak)) return false;
        b = {};
        b.resize((size_t) n);
        return b.visit(0, b.size(), [&](uint8_t* p, size_t c, size_t) { return raw(p, c); });
    }
    bool buffer(ConversationBuffer& b, size_t part) {
        uint64_t n = 0;
        const uint64_t limit = kv_layer < limits.max_kv_bytes.size() ? limits.max_kv_bytes[kv_layer][part] : UINT64_MAX;
        if (!count(n, 1, limit)) return false;
        b = {};
        b.resize((size_t) n);
        return b.visit(0, b.size(), [&](uint8_t* p, size_t c, size_t) { return raw(p, c); });
    }
};

// The sections are shared by every layout and by delta records: any reader with raw/u64/i64/count/vec/fail/limits
// works (In over a session file, DeltaIn over a record).  The geometry prefix carries the fail-closed checks: a file
// of another geometry or layer range allocates nothing.
template<class R>
bool get_checkpoint_t(R& in, ConversationCheckpoint& c) {
    if (!in.vec(c.ids, in.limits.max_tokens)) return false;
    uint64_t n = 0;
    if (!in.count(n, 16, in.limits.max_tokens) || !in.resize(c.imgs, n)) return false;
    for (auto& k : c.imgs) if (!in.i64(k.start) || !in.u64(k.hash)) return false;
    const auto& m = in.limits.max_state_bytes;   // byte arrays: element and byte counts are the same
    return in.vec(c.gdn, m[0]) && in.vec(c.ple, m[1]) && in.vec(c.tails, m[2]) && in.vec(c.dead, m[3]) &&
           in.vec(c.block_pos, m[4]) && in.u64(c.used);
}

template<class R>
bool get_geo_prefix_t(R& in, SavedConversation& s) {
    for (auto& g : s.geometry) if (!in.i64(g)) return false;
    // checked before any state array is read: a file of another geometry allocates nothing
    if (in.limits.geometry && *in.limits.geometry != s.geometry)
        return in.fail("saved with another model geometry than this runtime's");
    uint64_t cvec = 0;
    if (!in.i64(s.layer_lo) || !in.i64(s.layer_hi) || !in.u64(cvec)) return false;
    if (in.limits.layer_range &&
        (in.limits.layer_range->first != s.layer_lo || in.limits.layer_range->second != s.layer_hi))
        return in.fail("saved with another layer range than this runtime's");
    if (cvec > 1) return in.fail("invalid cvec flag");
    s.cvec = cvec == 1;
    return true;
}

// the running state, v1 raw coding: the live checkpoint first, then the count and the checkpoints
template<class R>
bool get_states_raw_t(R& in, SavedConversation& s) {
    if (!get_checkpoint_t(in, s.live)) return false;
    uint64_t n = 0;
    // a checkpoint is at least 8 counts + used = 72 bytes
    if (!in.count(n, 72, in.limits.max_checkpoints) || !in.resize(s.checkpoints, n)) return false;
    for (auto& c : s.checkpoints) if (!get_checkpoint_t(in, c)) return false;
    return true;
}

template<class R>
bool get_meta_prefix_t(R& in, SavedConversation& s) {
    return get_geo_prefix_t(in, s) && get_states_raw_t(in, s);
}

bool get_meta_prefix(In& in, SavedConversation& s) { return get_meta_prefix_t(in, s); }

// the K/V section, materialized into the image: the layer count, per-layer header, the five buffers.  <admit>
// prices every count-based allocation (the descriptor formats); v1 stays plain, as before.
template<bool admit>
bool get_kv_full(In& in, SavedConversation& s) {
    uint64_t n = 0;
    if (!in.count(n, 96, in.limits.max_kv_layers)) return false;
    if (admit) { if (!in.resize_admitted(s.kv, n)) return false; }
    else s.kv.resize((size_t) n);
    for (size_t l = 0; l < (size_t) n; ++l) {
        auto& k = s.kv[l];
        in.kv_layer = l;
        int64_t format = 0;
        if (!in.i64(format) || !in.i64(k.cells) || !in.i64(k.heads) || !in.i64(k.head_dim) ||
            !in.i64(k.page_size) || !in.i64(k.pooled_rows) || !in.i64(k.idx_dim)) return false;
        if (format < INT32_MIN || format > INT32_MAX) return in.fail("invalid K/V format");
        k.format = (int) format;
        if (admit) {
            if (!in.buffer_admitted(k.k, 0) || !in.buffer_admitted(k.v, 1) || !in.buffer_admitted(k.k_scale, 2) ||
                !in.buffer_admitted(k.v_scale, 3) || !in.buffer_admitted(k.pooled, 4)) return false;
        } else {
            if (!in.buffer(k.k, 0) || !in.buffer(k.v, 1) || !in.buffer(k.k_scale, 2) ||
                !in.buffer(k.v_scale, 3) || !in.buffer(k.pooled, 4)) return false;
        }
    }
    return true;
}

bool get_payload(In& in, SavedConversation& s) {
    if (!get_meta_prefix(in, s)) return false;
    return get_kv_full<false>(in, s);
}

// defined with the v2 readers below
template<class R> bool get_states_v2_t(R& in, SavedConversation& s);

// the K/V-first v3 payload: the shared geometry prefix, the K/V, then the v2 state descriptors
bool get_payload_v3(In& in, SavedConversation& s) {
    if (!get_geo_prefix_t(in, s)) return false;
    if (!get_kv_full<true>(in, s)) return false;
    return get_states_v2_t(in, s);
}

// A delta record's reader: bounded reads at an absolute file offset, hashing every payload byte (the trailer is read
// without hashing, as in read_impl).  The read pass records each part's offset and size; the apply pass (verify) makes
// this walk refuse a record whose layer set, K/V headers, offsets or sizes differ from the read pass's record before
// that part is touched.
struct DeltaIn {
    RawFile* f;
    uint64_t at, end;
    SessionHasher hash;
    std::string& error;
    const SessionReadLimits& limits;
    std::vector<std::array<uint64_t, 5>>* record_kv = nullptr;
    std::vector<std::array<uint64_t, 5>>* record_offset = nullptr;
    const SessionDeltaRecord* verify = nullptr;
    SessionError kind = SessionError::invalid;
    size_t kv_layer = 0;
    const ConversationCheckpoint* ref_live = nullptr;
    // the digest's: see In's (no current delta caller sets them)
    bool* saw_ref = nullptr;
    bool* saw_rht = nullptr;
    const char* what = "delta record";   // the error prefix
    bool fail(const std::string& m) { if (error.empty()) error = std::string(what) + ": " + m; return false; }
    bool account_state_codec(uint64_t codec, uint64_t, uint64_t) { return codec <= 2; }
    bool raw(void* p, size_t c) {
        if (c > end - at) return fail("record ends early");
        if (c && !f->read_at(at, static_cast<uint8_t*>(p), c)) { kind = f->kind(); return fail("read error: " + f->error()); }
        hash.update(p, c);
        at += c;
        return true;
    }
    bool tail(void* p, size_t c) {   // the trailer: read, not hashed
        if (c > end - at) return fail("record ends early");
        if (c && !f->read_at(at, static_cast<uint8_t*>(p), c)) { kind = f->kind(); return fail("read error: " + f->error()); }
        at += c;
        return true;
    }
    bool u64(uint64_t& v) { return raw(&v, 8); }
    bool i64(int64_t& v) { return raw(&v, 8); }
    bool count(uint64_t& n, size_t elem, uint64_t limit = UINT64_MAX) {
        if (!u64(n)) return false;
        if (elem && n > (end - at) / elem) return fail("element count exceeds the record");
        if (n > limit) return fail("element count " + std::to_string(n) + " exceeds this engine's limit " +
                                   std::to_string(limit));
        if (n > (uint64_t) std::numeric_limits<size_t>::max() / (elem ? elem : 1)) return fail("element count too large");
        return true;
    }
    bool bound(uint64_t n, size_t elem, uint64_t limit = UINT64_MAX) {
        if (elem && n > (end - at) / elem) return fail("element count exceeds the record");
        if (n > limit) return fail("element count " + std::to_string(n) + " exceeds this engine's limit " +
                                   std::to_string(limit));
        if (n > (uint64_t) std::numeric_limits<size_t>::max() / (elem ? elem : 1)) return fail("element count too large");
        return true;
    }
    // The v3 apply walk's states section: `n` bytes feed the running hash with NO decode and no destination - the
    // same bytes a full decode would read and hash, so the trailing-hash check covers them identically.  `scratch` is
    // a reusable read block (the K/V walk's 16 MiB block); `n` must not exceed the record bytes left.
    bool hash_only(uint8_t* scratch, size_t cap, uint64_t n) {
        if (n > end - at) return fail("record ends early");
        while (n) {
            const size_t c = (size_t) std::min<uint64_t>(cap, n);
            if (!f->read_at(at, scratch, c)) { kind = f->kind(); return fail("read error: " + f->error()); }
            hash.update(scratch, c);
            at += c;
            n -= c;
        }
        return true;
    }
    bool prepare_allocation(uint64_t count, size_t element_size) {
        if (element_size && count > (uint64_t) std::numeric_limits<size_t>::max() / element_size)
            return fail("allocation size exceeds this build");
        const uint64_t need = allocation_bytes(count, element_size);
        if (need == UINT64_MAX) return fail("allocation size overflows");
        if (need && limits.admit) {
            std::string why;
            if (!limits.admit(need, why)) {
                kind = SessionError::memory;
                if (error.empty()) error = std::string(what) + ": " +
                    (why.empty() ? std::string("RAM admission refused") : why);
                return false;
            }
        }
        return true;
    }
    template<class T> bool resize(std::vector<T>& v, uint64_t n) {
        if (n > v.max_size()) return fail("element count exceeds this container's maximum");
        if (!prepare_allocation(n, sizeof(T))) return false;
        const size_t count = (size_t) n;
        if (count) v.reserve(count);   // these parser-owned vectors start empty; no geometric growth or old/new overlap
        v.resize(count);
        return true;
    }
    template<class T> bool resize_zero(std::vector<T>& v, uint64_t n) {
        if (!resize(v, n)) return false;
        std::fill(v.begin(), v.end(), T{});
        return true;
    }
    template<class T> bool vec(std::vector<T>& v, uint64_t limit = UINT64_MAX) {
        uint64_t n = 0;
        if (!count(n, sizeof(T), limit) || !resize(v, n)) return false;
        return raw(v.data(), (size_t) n * sizeof(T));
    }
    template<class T, class A> bool resize_admitted(std::vector<T, A>& v, uint64_t n) { return resize(v, n); }
    template<class T, class A> bool vec_admitted(std::vector<T, A>& v, uint64_t limit = UINT64_MAX) {
        return vec(v, limit);
    }
};

// The K/V section, streamed: every part's bytes pass through one 16 MiB block and the hasher, never kept; a `sink`
// (the apply pass) gets each block as it is read.  When `in.record_kv` is set the part sizes are recorded for the
// caller; when `in.verify_kv` is set (the apply pass) the file must have exactly the layer count and part sizes the
// read pass saw, checked before that layer or part is applied.  <admit> prices the layer directory and the block
// (the descriptor formats); v1 stays plain, as before.
template<bool admit>
bool get_kv_streamed(In& in, SavedConversation& s, const SessionKvBlock& sink) {
    uint64_t n = 0;
    if (!in.count(n, 96, in.limits.max_kv_layers)) return false;
    if (in.verify_kv && in.verify_kv->size() != (size_t) n)
        return in.fail("K/V layer count differs from the read pass");
    if (admit) { if (!in.resize_admitted(s.kv, n)) return false; }
    else s.kv.resize((size_t) n);
    if (in.record_kv) in.record_kv->assign((size_t) n, {});
    Aligned block;
    if (n) {
        if (admit && !in.admit_need(kBlock)) return false;
        block = aligned_block();
        if (!block) { in.kind = SessionError::memory; return in.fail("out of memory for the K/V read block"); }
    }
    for (size_t l = 0; l < (size_t) n; ++l) {
        auto& k = s.kv[l];
        in.kv_layer = l;
        int64_t format = 0;
        if (!in.i64(format) || !in.i64(k.cells) || !in.i64(k.heads) || !in.i64(k.head_dim) ||
            !in.i64(k.page_size) || !in.i64(k.pooled_rows) || !in.i64(k.idx_dim)) return false;
        if (format < INT32_MIN || format > INT32_MAX) return in.fail("invalid K/V format");
        k.format = (int) format;
        for (size_t part = 0; part < 5; ++part) {
            uint64_t size = 0;
            const uint64_t limit = l < in.limits.max_kv_bytes.size() ? in.limits.max_kv_bytes[l][part] : UINT64_MAX;
            if (!in.count(size, 1, limit)) return false;
            if (in.record_kv) (*in.record_kv)[l][part] = size;
            if (in.verify_kv && size != (*in.verify_kv)[l][part])
                return in.fail("K/V part size differs from the read pass");
            for (uint64_t at = 0; at < size; at += kBlock) {
                const size_t c = (size_t) std::min<uint64_t>(kBlock, size - at);
                if (!in.raw(block.get(), c)) return false;
                if (sink && !sink(l, part, at, block.get(), c)) {
                    if (in.error.empty()) in.error = "session file: the K/V sink failed";
                    in.kind = SessionError::io;
                    return false;
                }
            }
        }
    }
    return true;
}

bool get_payload_streamed(In& in, SavedConversation& s, const SessionKvBlock& sink) {
    if (!get_meta_prefix(in, s)) return false;
    return get_kv_streamed<false>(in, s, sink);
}

bool has_stage_parts(const SavedConversation& s) {
    if (!s.live.stage_parts.empty()) return true;
    for (const auto& c : s.checkpoints) if (!c.stage_parts.empty()) return true;
    return false;
}

// ---- format v2 readers: descriptor-coded state arrays with per-allocation admission.  REF resolves against the
// file's live checkpoint (parsed first) into an INDEPENDENT vector; RHT decodes into an independent logical-size
// vector.  ids/imgs/used are never referenced and keep their stored values.
template<class R, class Vec>
bool get_state_v2(R& in, Vec& dst, const Vec& live_src, uint64_t limit, bool live, const char* name,
                  bool gdn, const std::array<int64_t, 18>& geometry, int64_t layer_lo, int64_t layer_hi) {
    uint64_t codec = 0, logical = 0, stored = 0;
    if (!in.u64(codec) || !in.u64(logical) || !in.u64(stored)) return false;
    if (codec > 2) return in.fail(std::string("unknown state array codec ") + std::to_string(codec) + " (" + name + ")");
    if (codec == 1) {
        // REF: only a non-live checkpoint, no stored bytes, the live same-order array's exact length (a logical size
        // may exceed the file's remaining bytes - that is what REF is for)
        if (in.saw_ref) *in.saw_ref = true;
        if (live) return in.fail("the live state cannot reference itself");
        if (stored != 0) return in.fail(std::string("a REF state array stores no bytes (") + name + ")");
        if (logical != (uint64_t) live_src.size())
            return in.fail(std::string("a REF state array's length differs from the live state's (") + name + ")");
        if (!in.bound(logical, 0, limit) || !in.resize_admitted(dst, logical)) return false;
        if (logical) std::memcpy(dst.data(), live_src.data(), (size_t) logical);
        return true;
    }
    if (codec == 2) {
        if (in.saw_rht) *in.saw_rht = true;
        if (!gdn) return in.fail(std::string("RHT codec is only valid for gdn state (") + name + ")");
        rht::Layout layout{};
        if (!rht::layout_from(geometry, layer_lo, layer_hi, layout))
            return in.fail("RHT codec is invalid for this model geometry and layer range");
        if (logical != layout.logical)
            return in.fail("RHT state logical size does not match the model geometry");
        if (stored != layout.stored)
            return in.fail("RHT state stored size does not match the model geometry");
        if (stored > logical) return in.fail("RHT stored size exceeds its logical size");
        if (!in.bound(logical, 0, limit) || !in.bound(stored, 1) || !in.resize_admitted(dst, logical)) return false;
        const auto read = [&in](void* data, size_t bytes) { return in.raw(data, bytes); };
        if (!rht::decode(read, dst.data(), layout))
            return in.fail(std::string("invalid RHT state payload (") + name + ")");
        return true;
    }
    if (stored != logical)
        return in.fail(std::string("a RAW state array's stored size differs from its logical size (") + name + ")");
    // `stored` was read with the descriptor: validate it against the bytes left and the engine limit, never read it
    // again (count() would consume the next u64 of the payload and overwrite `stored`)
    if (!in.bound(stored, 1, limit)) return false;
    if (!in.resize_admitted(dst, logical)) return false;
    return in.raw(dst.data(), (size_t) stored);
}

template<class R>
bool get_checkpoint_v2(R& in, ConversationCheckpoint& c, bool live,
                       const std::array<int64_t, 18>& geometry, int64_t layer_lo, int64_t layer_hi) {
    if (!in.vec_admitted(c.ids, in.limits.max_tokens)) return false;
    uint64_t n = 0;
    if (!in.count(n, 16, in.limits.max_tokens) || !in.resize_admitted(c.imgs, n)) return false;
    for (auto& k : c.imgs) if (!in.i64(k.start) || !in.u64(k.hash)) return false;
    const auto& m = in.limits.max_state_bytes;   // byte arrays: element and byte counts are the same
    if (!get_state_v2(in, c.gdn, in.ref_live->gdn, m[0], live, "gdn", true, geometry, layer_lo, layer_hi)) return false;
    if (!get_state_v2(in, c.ple, in.ref_live->ple, m[1], live, "ple", false, geometry, layer_lo, layer_hi)) return false;
    if (!get_state_v2(in, c.tails, in.ref_live->tails, m[2], live, "tails", false, geometry, layer_lo, layer_hi)) return false;
    if (!get_state_v2(in, c.dead, in.ref_live->dead, m[3], live, "dead", false, geometry, layer_lo, layer_hi)) return false;
    return get_state_v2(in, c.block_pos, in.ref_live->block_pos, m[4], live, "block_pos", false,
                        geometry, layer_lo, layer_hi) && in.u64(c.used);
}

// the running state, v2 descriptor coding.  REF resolves against the file's live checkpoint (parsed first) into an
// INDEPENDENT vector; RHT decodes into an independent logical-size vector.  ids/imgs/used are never referenced and
// keep their stored values.  Every layout with descriptor states parses this section the same way (the v2 file
// states-first, the v3 file states-last); it follows the geo prefix.
template<class R>
bool get_states_v2_t(R& in, SavedConversation& s) {
    in.ref_live = &s.live;   // set before the live parse: the descriptor reader resolves REF arrays against it
    if (!get_checkpoint_v2(in, s.live, true, s.geometry, s.layer_lo, s.layer_hi)) return false;
    uint64_t n = 0;
    // a v2 checkpoint is at least 2 counts + 5 descriptors + used = 144 bytes
    if (!in.count(n, 144, in.limits.max_checkpoints) || !in.resize_admitted(s.checkpoints, n)) return false;
    for (auto& c : s.checkpoints)
        if (!get_checkpoint_v2(in, c, false, s.geometry, s.layer_lo, s.layer_hi)) return false;
    return true;
}

template<class R>
bool get_meta_prefix_v2(R& in, SavedConversation& s) {
    return get_geo_prefix_t(in, s) && get_states_v2_t(in, s);
}

bool get_payload_v2(In& in, SavedConversation& s) {
    if (!get_meta_prefix_v2(in, s)) return false;
    return get_kv_full<true>(in, s);
}

// The streaming v2 read: the state arrays resolve as in get_payload_v2; the K/V part streams through one admitted
// 16 MiB block exactly as the v1 streamed walk.
bool get_payload_streamed_v2(In& in, SavedConversation& s, const SessionKvBlock& sink) {
    if (!get_meta_prefix_v2(in, s)) return false;
    return get_kv_streamed<true>(in, s, sink);
}

// the streaming v3 read: the geo prefix, the streamed K/V, then the state descriptors
bool get_payload_streamed_v3(In& in, SavedConversation& s, const SessionKvBlock& sink) {
    if (!get_geo_prefix_t(in, s)) return false;
    if (!get_kv_streamed<true>(in, s, sink)) return false;
    return get_states_v2_t(in, s);
}

// the v3 APPLY walk: the geo prefix, the streamed K/V to the sink, then the state section consumed HASH-ONLY - every
// remaining payload byte feeds the running hash with no decode, exactly the bytes the read pass decoded and hashed.
// The states are protected exactly like the K/V: the read pass's full decode plus this same trailing payload hash,
// so a state corruption surfaces no later than the trailer check - after any K/V reached the sink - and the caller
// fails stops the same way it does for K/V corruption.
bool get_payload_apply_v3(In& in, SavedConversation& s, const SessionKvBlock& sink) {
    if (!get_geo_prefix_t(in, s)) return false;
    if (!get_kv_streamed<true>(in, s, sink)) return false;
    if (!in.left) return true;
    Aligned block = aligned_block();
    if (!block) { in.kind = SessionError::memory; return in.fail("out of memory for the state read block"); }
    return in.hash_only(block.get(), kBlock, in.left);
}

// header, little-endian: magic[8] version:u32 header_size:u32 model:u64 config:u64 payload:u64 reserved:u64[2]
// header_hash:u64 (session_hash64 of bytes 0..55, seed 0); version 2 is the descriptor-coded state format, version 3
// the K/V-first layout with the same state descriptors (below)
void header_bytes(uint8_t* h, const SessionFileIdentity& id, uint64_t payload, uint32_t version) {
    std::memset(h, 0, kHeader);
    std::memcpy(h, kMagic, 8);
    const uint32_t size = kHeader;
    std::memcpy(h + 8, &version, 4);
    std::memcpy(h + 12, &size, 4);
    std::memcpy(h + 16, &id.model, 8);
    std::memcpy(h + 24, &id.config, 8);
    std::memcpy(h + 32, &payload, 8);
    const uint64_t hh = session_hash64(h, 56, 0);
    std::memcpy(h + 56, &hh, 8);
}
} // namespace

SessionHasher::SessionHasher(uint64_t seed)
    : lane_{seed + P1 + P2, seed + P2, seed, seed - P1}, pending_{} {}

void SessionHasher::block(const uint8_t* p) {
    for (int i = 0; i < 4; ++i) lane_[i] = round1(lane_[i], load64(p + 8 * i));
}

void SessionHasher::update(const void* data, size_t n) {
    const uint8_t* p = static_cast<const uint8_t*>(data);
    total_ += n;
    if (npending_) {
        const size_t take = std::min(n, sizeof pending_ - npending_);
        std::memcpy(pending_ + npending_, p, take);
        npending_ += take; p += take; n -= take;
        if (npending_ < sizeof pending_) return;
        block(pending_);
        npending_ = 0;
    }
    for (; n >= 32; p += 32, n -= 32) block(p);
    if (n) std::memcpy(pending_, p, n);
    npending_ = n;
}

uint64_t SessionHasher::digest() const {
    uint64_t h = rotl(lane_[0], 1) + rotl(lane_[1], 7) + rotl(lane_[2], 12) + rotl(lane_[3], 18);
    for (uint64_t l : lane_) { h ^= round1(0, l); h = h * P1 + P4; }
    h += total_;
    size_t i = 0;
    for (; i + 8 <= npending_; i += 8) { h ^= round1(0, load64(pending_ + i)); h = rotl(h, 27) * P1 + P4; }
    for (; i < npending_; ++i) { h ^= pending_[i] * P5; h = rotl(h, 11) * P1; }
    h ^= h >> 33; h *= P2; h ^= h >> 29; h *= P3; h ^= h >> 32;
    return h;
}

uint64_t session_hash64(const void* data, size_t n, uint64_t seed) {
    SessionHasher h(seed);
    h.update(data, n);
    return h.digest();
}

void SessionIdentityBuilder::field(const std::string& tag, uint8_t type, const void* data, uint64_t n) {
    const uint64_t tn = tag.size();
    hash_.update(&tn, 8);
    hash_.update(tag.data(), tag.size());
    hash_.update(&type, 1);
    hash_.update(&n, 8);
    if (n) hash_.update(data, (size_t) n);
}
void SessionIdentityBuilder::str(const std::string& tag, const std::string& v) { field(tag, 's', v.data(), v.size()); }
void SessionIdentityBuilder::i64(const std::string& tag, int64_t v) { field(tag, 'i', &v, 8); }
void SessionIdentityBuilder::u64(const std::string& tag, uint64_t v) { field(tag, 'u', &v, 8); }
void SessionIdentityBuilder::f64(const std::string& tag, double v) {
    uint64_t bits = 0;
    std::memcpy(&bits, &v, 8);
    field(tag, 'f', &bits, 8);
}
void SessionIdentityBuilder::bytes(const std::string& tag, const void* data, size_t n) { field(tag, 'b', data, n); }

uint64_t session_config_fingerprint(const SessionConfig& c) {
    SessionIdentityBuilder b(0x434f4e4649473031ull);   // "CONFIG01"
    b.str("engine", c.engine_version);
    b.str("backend", c.backend);
    b.str("kv", c.kv);
    b.i64("max_context", c.max_context);
    b.i64("kv_resident", c.kv_resident);
    b.i64("mtp_window", c.mtp_window);
    b.i64("kv_rot", c.kv_rot ? 1 : 0);
    b.i64("rope.type", c.rope.type);
    b.f64("rope.freq_base", c.rope.freq_base);
    b.f64("rope.factor", c.rope.factor);
    b.f64("rope.freq_scale", c.rope.freq_scale);
    b.f64("rope.orig_ctx", c.rope.orig_ctx);
    b.f64("rope.ext_factor", c.rope.ext_factor);
    b.f64("rope.attn_factor", c.rope.attn_factor);
    b.f64("rope.beta_fast", c.rope.beta_fast);
    b.f64("rope.beta_slow", c.rope.beta_slow);
    b.u64("cvec", c.cvec);
    b.u64("switches", c.switches.size());
    for (const auto& [name, value] : c.switches) b.i64("switch." + name, value);
    return b.digest();
}

const char* session_error_name(SessionError e) {
    switch (e) {
        case SessionError::none: return "none";
        case SessionError::invalid: return "invalid";
        case SessionError::storage: return "storage";
        case SessionError::memory: return "memory";
        case SessionError::io: return "io";
    }
    return "io";
}

std::string session_free_space_dir(const std::string& path, bool windows_rules) {
    if (!windows_rules) {
        const size_t cut = path.find_last_of('/');
        if (cut == std::string::npos) return ".";
        return path.substr(0, cut == 0 ? 1 : cut);
    }
    std::string p = path;
    for (char& c : p) if (c == '/') c = '\\';
    const size_t cut = p.find_last_of('\\');
    if (cut == std::string::npos) {
        // "C:chat.bin" (drive-relative): the drive root answers for the same volume
        if (p.size() >= 2 && p[1] == ':') return p.substr(0, 2) + "\\";
        return std::string();   // the current folder
    }
    return p.substr(0, cut + 1);   // with its trailing backslash: "\\server\share\dir\", "C:\"
}

std::vector<SessionModelFile> session_model_inputs(const SessionInputs& in) {
    std::vector<SessionModelFile> f;
    for (size_t i = 0; i < in.native_shards.size(); ++i) f.push_back({"native shard " + std::to_string(i), in.native_shards[i]});
    for (size_t i = 0; i < in.native_dense.size(); ++i) f.push_back({"native dense " + std::to_string(i), in.native_dense[i]});
    for (size_t i = 0; i < in.native_head.size(); ++i) f.push_back({"native head " + std::to_string(i), in.native_head[i]});
    if (!in.embedding.empty()) f.push_back({"embedding", in.embedding});
    if (!in.ple.empty()) f.push_back({"ple", in.ple});
    if (!in.mtp.empty()) f.push_back({"mtp", in.mtp});
    if (!in.pack.empty()) f.push_back({"pack native_experts.txt", in.pack + "/native_experts.txt", true});
    for (const auto& [role, path] : in.experts) f.push_back({role, path});
    return f;
}

bool session_model_fingerprint(const std::vector<SessionModelFile>& files, uint64_t& fingerprint, std::string& error,
                               const std::function<void()>& beat) {
    SessionIdentityBuilder h(0x5354524154414d44ull);
    std::vector<uint8_t> buf(1u << 20);
    // a file in several roles is read once; its sample enters the fingerprint under every role
    std::unordered_map<std::string, std::pair<bool, uint64_t>> seen;   // path -> (present, sample digest)
    h.u64("files", files.size());
    for (const auto& file : files) {
        auto it = seen.find(file.path);
        if (it == seen.end()) {
            RawFile f;
            uint64_t size = 0;
            std::pair<bool, uint64_t> d{false, 0};
            if (!f.open_model(file.path, size)) {
                if (!(file.optional && f.missing())) {
                    error = "session fingerprint: " + file.role + " " + file.path + ": " + f.error();
                    return false;
                }
            } else {
                SessionIdentityBuilder one(0x46494c4553414d50ull);   // "FILESAMP"
                one.u64("size", size);
                const uint64_t head = std::min<uint64_t>(size, buf.size());
                if (!f.read_at(0, buf.data(), (size_t) head)) {
                    error = "session fingerprint: reading " + file.path + ": " + f.error();
                    return false;
                }
                one.bytes("head", buf.data(), (size_t) head);
                if (size > buf.size()) {
                    const uint64_t tail = std::min<uint64_t>(size - head, buf.size());
                    if (!f.read_at(size - tail, buf.data(), (size_t) tail)) {
                        error = "session fingerprint: reading " + file.path + ": " + f.error();
                        return false;
                    }
                    one.bytes("tail", buf.data(), (size_t) tail);
                }
                d = {true, one.digest()};
            }
            it = seen.emplace(file.path, d).first;
            if (beat) beat();
        }
        if (!it->second.first) {
            if (!file.optional) { error = "session fingerprint: " + file.role + " " + file.path + " is missing"; return false; }
            h.str("absent", file.role);
            continue;
        }
        h.str("role", file.role);
        h.u64("sample", it->second.second);
    }
    fingerprint = h.digest();
    return true;
}

uint64_t session_read_max_file_bytes(const SessionReadLimits& l) {
    const uint64_t open = UINT64_MAX;
    if (l.max_tokens == open || l.max_checkpoints == open || l.max_kv_layers == open) return l.max_file_bytes;
    for (uint64_t b : l.max_state_bytes) if (b == open) return l.max_file_bytes;
    if (l.max_kv_bytes.size() < l.max_kv_layers) return l.max_file_bytes;
    // one checkpoint: ids (8 + 4 per token), images (8 + 16 per image), five state arrays (8 + bytes), used (8)
    uint64_t cp = sat_add(16, sat_mul(l.max_tokens, 4 + 16));
    for (uint64_t b : l.max_state_bytes) cp = sat_add(cp, sat_add(8, b));
    cp = sat_add(cp, 8);
    uint64_t n = kHeader + kTrailer + 18 * 8 + 3 * 8;             // geometry, layer range, cvec
    n = sat_add(n, cp);                                            // live
    n = sat_add(n, sat_add(8, sat_mul(l.max_checkpoints, cp)));    // checkpoints
    n = sat_add(n, 8);
    for (uint64_t i = 0; i < l.max_kv_layers; ++i) {
        n = sat_add(n, 7 * 8);
        for (uint64_t b : l.max_kv_bytes[(size_t) i]) n = sat_add(n, sat_add(8, b));
    }
    return std::min(n, l.max_file_bytes);
}

uint64_t session_read_peak_bytes(uint64_t file_bytes, const SessionReadLimits& limits, uint32_t file_version,
                                 uint64_t decoded_state_extra) {
    (void) limits;
    // Optional descriptor expansion augments this estimate when a caller already knows it. Readers keep this as an
    // initial screen, then admit each actual logical destination allocation once while parsing.
    const uint64_t parsed_bytes = sat_add(file_bytes,
        file_version >= kVersionV2 ? decoded_state_extra : 0);
    const uint64_t segments = parsed_bytes / kBlock + 1;
    return sat_add(sat_add(sat_add(parsed_bytes, kBlock), sat_mul(segments, 4096)), uint64_t(1) << 20);
}

namespace {
uint64_t session_meta_peak_bytes(const SessionReadLimits& limits, bool include_kv_layers) {
    const uint64_t states = sat_add(limits.max_checkpoints, 1);
    uint64_t one_checkpoint = allocation_bytes(limits.max_tokens, sizeof(int32_t));
    one_checkpoint = sat_add(one_checkpoint, allocation_bytes(limits.max_tokens, sizeof(ConversationImageKey)));
    for (uint64_t bytes : limits.max_state_bytes) one_checkpoint = sat_add(one_checkpoint, round_allocation_pages(bytes));

    uint64_t total = sizeof(SavedConversation);
    total = sat_add(total, allocation_bytes(limits.max_checkpoints, sizeof(ConversationCheckpoint)));
    if (include_kv_layers)
        total = sat_add(total, allocation_bytes(limits.max_kv_layers, sizeof(ConversationKv)));
    total = sat_add(total, sat_mul(states, one_checkpoint));
    return total;
}
} // namespace

uint64_t session_read_meta_peak_bytes(const SessionReadLimits& limits) {
    // The live checkpoint and every accepted checkpoint; their vectors allocate once from empty and are bounded by
    // the same counts and byte limits the parser checks.  Page rounding covers each independent vector allocation.
    return session_meta_peak_bytes(limits, true);
}

uint64_t session_delta_walk_peak_bytes(const SessionReadLimits& limits) {
    // The read pass has one populated result plus its empty caller-owned output object. During apply, the already-read
    // expected record coexists with a parser SavedConversation. The base/latest metadata is already live and remains
    // accounted for by available memory; this incremental need never includes log length or streamed K/V payload.
    uint64_t record = session_meta_peak_bytes(limits, true);
    record = sat_add(record, sizeof(SessionDeltaRecord) - sizeof(SavedConversation));
    const uint64_t layer_parts = allocation_bytes(limits.max_kv_layers, sizeof(std::array<uint64_t, 5>));
    record = sat_add(record, sat_mul(2, layer_parts));

    const uint64_t scratch = sat_add(kBlock, sizeof(DeltaIn) + sizeof(RawFile) + sizeof(Aligned) + 64);
    const uint64_t read_peak = sat_add(sat_add(record, sizeof(SessionDeltaRecord)), scratch);
    const uint64_t apply_peak = sat_add(sat_add(record, session_meta_peak_bytes(limits, false)), scratch);
    return std::max(read_peak, apply_peak);
}

namespace {
uint64_t session_stream_metadata_bytes(uint64_t file_bytes, const SessionReadLimits& limits,
                                       bool& includes_state_allocations) {
    includes_state_allocations = false;
    const bool state_open = std::find(limits.max_state_bytes.begin(), limits.max_state_bytes.end(), UINT64_MAX) !=
                            limits.max_state_bytes.end();
    if (limits.max_tokens == UINT64_MAX || limits.max_checkpoints == UINT64_MAX || state_open) return file_bytes;

    uint64_t one_checkpoint = 64;
    const uint64_t token_bytes = sizeof(int32_t) + sizeof(ConversationImageKey);
    if (file_bytes < one_checkpoint || limits.max_tokens > (file_bytes - one_checkpoint) / token_bytes) return file_bytes;
    one_checkpoint += limits.max_tokens * token_bytes;
    for (uint64_t bytes : limits.max_state_bytes) {
        if (bytes > file_bytes - one_checkpoint) return file_bytes;
        one_checkpoint += bytes;
    }
    if (one_checkpoint && limits.max_checkpoints > (file_bytes - one_checkpoint) / one_checkpoint) return file_bytes;
    includes_state_allocations = true;
    return one_checkpoint + limits.max_checkpoints * one_checkpoint;
}
} // namespace

uint64_t session_read_stream_peak_bytes(uint64_t file_bytes, const SessionReadLimits& limits, uint32_t file_version,
                                        uint64_t decoded_state_extra) {
    // the reader's block and the block a K/V part is moved through, plus the small vectors
    uint64_t n = sat_add(sat_mul(2, kBlock), uint64_t(1) << 20);
    // A bounded metadata estimate includes the allowed logical state arrays when it fits below the file-size cap.
    // Callers that know descriptors may add expansion not represented by the metadata estimate; the read path instead
    // admits each destination allocation separately after this initial screen.
    bool includes_state_allocations = false;
    const uint64_t metadata_bytes = session_stream_metadata_bytes(file_bytes, limits, includes_state_allocations);
    n = sat_add(n, metadata_bytes);
    if (file_version >= kVersionV2 && !includes_state_allocations) n = sat_add(n, decoded_state_extra);
    // one hash per <= 16 MiB block moved, and the per-layer directories (a tiny term, bounded by the file)
    n = sat_add(n, sat_mul(file_bytes / kBlock + 1, 6 * sizeof(uint64_t)));
    return n;
}

namespace {
bool write_impl(const std::string& path, const SavedConversation& image, const std::vector<SessionKvSource>* sources,
                const SessionFileIdentity& id, size_t& bytes, std::string& error, const SessionWriteOptions& opt,
                SessionStatus& st) {
    st = {};
    st.error = SessionError::invalid;   // refusals before any disk access
    if (has_stage_parts(image)) { error = "session file: layer-split state cannot be saved"; return false; }
    if (sources && !image.kv.empty()) { error = "session file: K/V given both in the image and as sources"; return false; }
    if (opt.background_state && !opt.layout_v3) {
        error = "session file: background-state writes require the v3 layout (layout_v3)";
        return false;   // the K/V-first order is what makes the early sources release safe
    }
    // the state-ref decision, once, before sizing: with state_ref on (a prefix save), a non-live checkpoint whose
    // five running-state arrays are ALL byte-identical to the live state's takes REF in a version-2 file; any
    // difference in any array keeps that checkpoint RAW, and absent both REF and RHT descriptors the file stays v1
    std::vector<char> ref;
    if (opt.state_ref && !image.checkpoints.empty()) {
        ref.assign(image.checkpoints.size(), 0);
        for (size_t i = 0; i < image.checkpoints.size(); ++i)
            ref[i] = checkpoint_state_homomorphic(image.checkpoints[i], image.live) ? 1 : 0;
        if (std::find(ref.begin(), ref.end(), char(1)) == ref.end()) ref.clear();
    }
    rht::Layout codec_layout{};
    const std::vector<StateCodecArray> codec = state_codecs(image, opt.state_codec_rht, ref, codec_layout);
    const bool any_codec2 = any_state_codec2(codec);
    // v3 is opt-in (the K/V-first layout); otherwise a REF or RHT descriptor makes the file v2, and with neither
    // the bytes stay v1
    const uint32_t version = opt.layout_v3 ? kVersionV3
                                           : (ref.empty() && !any_codec2 ? kVersion : kVersionV2);
    Out sizing;
    if (version == kVersion) put_payload(sizing, image, sources);
    else if (version == kVersionV2) put_payload_v2(sizing, image, sources, ref, codec, codec_layout);
    else put_payload_v3(sizing, image, sources, ref, codec, codec_layout);
    if (sizing.overflow || sizing.n > UINT64_MAX - kHeader - kTrailer ||
        sizing.n + kHeader + kTrailer > (uint64_t) std::numeric_limits<size_t>::max()) {
        error = "session file: the state is too large for this build";
        return false;
    }
    const uint64_t payload = sizing.n, total = kHeader + payload + kTrailer;
    st.error = SessionError::io;
    if (opt.min_free_bytes) {
        uint64_t avail = 0;
        if (!free_space(path, avail)) { error = "session file: cannot read the free disk space for " + path; return false; }
        if (avail < total || avail - total < opt.min_free_bytes) {
            st.error = SessionError::storage;
            error = "session file: not enough disk space (" + std::to_string(total >> 20) + " MiB plus a reserve of " +
                    std::to_string(opt.min_free_bytes >> 20) + " MiB needed, " + std::to_string(avail >> 20) +
                    " MiB free)";
            return false;
        }
    }
    Publisher pub;
    if (!pub.open(path, error)) { st.error = pub.kind(); return false; }
    RawFile raw;
    if (!pub.create_temp(raw, error)) {
        st.error = raw.kind();
        error = "session file: cannot create a temporary file beside " + path + ": " + error;
        return false;
    }
    SourcesReleased released(opt.on_sources_released);
    const std::function<int(const char*)>* fault = opt.fault ? &opt.fault : nullptr;
    std::string why;
    // the same production walk over either sink: with the background sink the K/V source reads overlap the OS
    // writes, and the sources-released notification fires while the writer thread is still draining the buffers
    const auto drive = [&](auto& sink) -> bool {
        using Sink = std::remove_reference_t<decltype(sink)>;
        if (opt.progress) sink.progress.emplace(opt.progress);
        sink.phase = opt.phase;
        sink.expected = total;
        sink.timings = opt.timings;
        if (sink.progress) (*sink.progress)(0, total);   // the temporary file exists: the transfer starts
        sink.write_fault = injected(fault, "write");
        sink.flush_fault = injected(fault, "file_flush");
        uint8_t h[kHeader];
        header_bytes(h, id, payload, version);
        bool ok = sink.write(h, kHeader);
        SessionHasher hash(0);
        Out out;
        if constexpr (std::is_same_v<Sink, AsyncFileSink>) out.af = &sink;
        else out.f = &sink;
        out.hash = &hash;
        out.timings = opt.timings;
        out.released = &released;
        if (ok) {
            if (version == kVersion) put_payload(out, image, sources);
            else if (version == kVersionV2) put_payload_v2(out, image, sources, ref, codec, codec_layout);
            else put_payload_v3(out, image, sources, ref, codec, codec_layout);
        }
        ok = ok && out.ok && out.n == payload;
        if (ok) {
            const uint64_t ph = hash.digest();
            ok = sink.write(&ph, 8) && sink.write(kEnd, 8);
        }
        ok = sink.finish(opt.durable) && ok;
        if (!ok) {
            why = out.read_failed ? "K/V source read" + (out.read_error.empty() ? std::string() : ": " + out.read_error)
                                  : sink.error();
            st.error = out.read_failed ? SessionError::io : sink.kind();
            return false;
        }
        return true;
    };
    bool ok;
    if (opt.background_state) {
        AsyncFileSink sink(raw);
        if (!sink.ok()) { st.error = SessionError::memory; error = "session file: " + sink.error(); return false; }
        ok = drive(sink);
    } else {
        FileSink sink(raw);
        if (!sink.ok()) { st.error = SessionError::memory; error = "session file: " + sink.error(); return false; }
        ok = drive(sink);
    }
    if (ok) {
        if (opt.phase) opt.phase("publish", total);
        const auto started = opt.timings ? TimingClock::now() : TimingClock::time_point{};
        const bool published = pub.publish(why, opt.durable, st, fault);
        if (opt.timings) timing_add(&opt.timings->publish_ns, started);
        if (published) {
            st.error = SessionError::none;
            bytes = (size_t) total;
            return true;
        }
    }
    error = "session file: writing " + path + " failed" + (why.empty() ? std::string() : ": " + why);
    if (st.published) error += " (the new file has already replaced the old one; its folder entry may not survive a power loss)";
    return false;   // Publisher removes this call's temporary file, and only that
}

bool write_guarded(const std::string& path, const SavedConversation& image, const std::vector<SessionKvSource>* sources,
                   const SessionFileIdentity& id, size_t& bytes, std::string& error, const SessionWriteOptions& opt,
                   SessionStatus* status) {
    SessionStatus st;
    bool ok = false;
    try {
        ok = write_impl(path, image, sources, id, bytes, error, opt, st);
    } catch (const std::bad_alloc&) {
        error = "session file: out of memory while writing " + path;
        st.error = SessionError::memory;
        ok = false;
    }
    if (status) *status = st;
    return ok;
}
} // namespace

std::vector<ConversationCheckpoint> session_checkpoints_to_save(const std::vector<ConversationCheckpoint>& chain) {
    const ConversationCheckpoint* deepest = session_deepest_checkpoint(chain);
    if (!deepest) return {};
    return {*deepest};
}

const ConversationCheckpoint* session_deepest_checkpoint(const std::vector<ConversationCheckpoint>& chain) {
    // A pinned shared prefix (pin=N) is the resume point the next queries want, so with one in the chain the file
    // keeps the deepest PINNED checkpoint, not a deeper periodic one a long private suffix made past it.  No pin:
    // the deepest, as before.
    const ConversationCheckpoint* deepest = nullptr;
    bool any_pinned = false;
    for (const auto& c : chain) any_pinned = any_pinned || c.pinned;
    for (const auto& c : chain)
        if ((!any_pinned || c.pinned) && (!deepest || c.ids.size() > deepest->ids.size())) deepest = &c;
    return deepest;
}

uint64_t session_save_peak_bytes(const ConversationCheckpoint* deepest, const SessionSaveLive& live) {
    const uint64_t buffers = live.background_state ? 2 * kBlock : kBlock;
    uint64_t n = sat_add(buffers, uint64_t(1) << 20);                         // write buffers, small vectors
    if (deepest) n = sat_add(n, sat_mul(live.background_state ? 1 : 2, deepest->bytes()));
    n = sat_add(n, live.state_bytes);                                         // the live running state
    n = sat_add(n, sat_mul(live.tokens, sizeof(int32_t)));                    // its token list
    n = sat_add(n, sat_mul(live.images, sizeof(ConversationImageKey)));       // its images
    n = sat_add(n, sat_mul(live.kv_layers, sizeof(SessionKvSource) + 256));   // the K/V source directory
    n = sat_add(n, live.codec_scratch);                                       // bounded state-codec working set
    return n;
}

bool session_save_checkpoints(const std::vector<ConversationCheckpoint>& chain, const SessionSaveLive& live,
                              const std::function<bool(uint64_t need_bytes, std::string& why)>& admit,
                              std::vector<ConversationCheckpoint>& out, std::string& why) {
    out.clear();
    const ConversationCheckpoint* deepest = session_deepest_checkpoint(chain);
    if (admit && !admit(session_save_peak_bytes(deepest, live), why)) return false;
    if (deepest) out.push_back(*deepest);
    return true;
}

int64_t session_phase_limit_s(uint64_t bytes) {
    return std::min<int64_t>(3600, 60 + (int64_t) std::min<uint64_t>(bytes >> 22, 3600));
}

bool session_file_write(const std::string& path, const SavedConversation& meta, const std::vector<SessionKvSource>& kv,
                        const SessionFileIdentity& id, size_t& bytes, std::string& error,
                        const SessionWriteOptions& options, SessionStatus* status) {
    return write_guarded(path, meta, &kv, id, bytes, error, options, status);
}

bool session_file_write(const std::string& path, const SavedConversation& image, const SessionFileIdentity& id,
                        size_t& bytes, std::string& error, const SessionWriteOptions& options, SessionStatus* status) {
    return write_guarded(path, image, nullptr, id, bytes, error, options, status);
}

namespace {
// The header every read checks first: magic, version, fields, size and both fingerprints, leaving `payload` and
// the format `version` (1 or 2) set.
bool header_ok(const uint8_t* h, uint64_t size, const SessionFileIdentity& id, uint64_t accept_config,
               uint32_t& version, uint64_t& payload, std::string& error) {
    uint32_t hsize = 0;
    uint64_t model = 0, config = 0, r0 = 0, r1 = 0, hh = 0;
    std::memcpy(&version, h + 8, 4); std::memcpy(&hsize, h + 12, 4);
    std::memcpy(&model, h + 16, 8); std::memcpy(&config, h + 24, 8); std::memcpy(&payload, h + 32, 8);
    std::memcpy(&r0, h + 40, 8); std::memcpy(&r1, h + 48, 8); std::memcpy(&hh, h + 56, 8);
    if (std::memcmp(h, kMagic, 8) != 0) { error = "session file: not a Strata session file (magic)"; return false; }
    if (hh != session_hash64(h, 56, 0)) { error = "session file: header checksum mismatch"; return false; }
    if (version < kVersion || version > kVersionV3) {
        error = "session file: unsupported version " + std::to_string(version);
        return false;
    }
    if (hsize != kHeader || r0 || r1) { error = "session file: invalid header fields"; return false; }
    if (model != id.model) { error = "session file: saved with another model (model fingerprint differs)"; return false; }
    // the opt-in accepted config (a gate-approved cache entry): the model still has to match
    if (config != id.config && !(accept_config && config == accept_config)) {
        error = "session file: saved with another engine configuration (config fingerprint differs)";
        return false;
    }
    if (payload > size || size - kHeader - kTrailer != payload) {
        error = "session file: size " + std::to_string(size) + " does not match the header (" +
                std::to_string(kHeader + payload + kTrailer) + ")";
        return false;
    }
    return true;
}

// A read's prologue: open the file by its name (never what a symbolic link there points to), bound its size, and
// read and check the header.  On success the source is positioned after the header and `payload` is set.
struct ReadFile {
    RawFile raw;
    FileSource source;
    uint64_t size = 0;
    uint32_t version = 0;   // the verified header's format version (1, 2 or 3)
    ReadFile() : source(raw) {}
    bool open(const std::string& path, const SessionFileIdentity& id, const SessionReadLimits& limits,
              uint64_t& payload, std::string& error, SessionStatus& st) {
        if (!raw.open_session(path, size)) {
            st.error = raw.refused() ? SessionError::invalid : raw.kind();
            error = "session file: " + path + ": " + raw.error();
            return false;
        }
        if (size < kHeader + kTrailer) {
            error = "session file: size " + std::to_string(size) + " is below the minimum";
            return false;
        }
        const uint64_t max_size = session_read_max_file_bytes(limits);   // the v1-derived bound, as before
        bool over_v1 = false;
        if (size > max_size || size > (uint64_t) std::numeric_limits<size_t>::max()) {
            if (size > limits.max_file_bytes || size > (uint64_t) std::numeric_limits<size_t>::max()) {
                error = "session file: size " + std::to_string(size) + " exceeds this engine's limit" +
                        (max_size == UINT64_MAX ? std::string() : " of " + std::to_string(max_size));
                return false;
            }
            // within the absolute cap but past the v1-derived bound: a descriptor-format file (v2 or v3) may legally
            // be larger (its descriptor overhead is not in the v1 count), so the verified header decides
            over_v1 = true;
        }
        if (!source.ok()) { st.error = SessionError::memory; error = "session file: " + source.error(); return false; }
        // installed before the first block (the header's) is read: every block reports, the first one included
        if (limits.progress) source.progress.emplace(limits.progress);
        source.expected = size;
        uint8_t h[kHeader];
        if (!source.read(h, kHeader)) {
            st.error = source.kind();
            error = "session file: header read error: " + source.error();
            return false;
        }
        if (!header_ok(h, size, id, limits.accept_config, version, payload, error)) return false;
        if (over_v1) {
            if (version == kVersion) {
                error = "session file: size " + std::to_string(size) + " exceeds this engine's limit of " +
                        std::to_string(max_size);
                return false;
            }
            // v2/v3's own physical bound: the untruncated v1-derived bound plus the per-checkpoint descriptor slack
            // (live included), still under the absolute cap
            const uint64_t v2_max = sat_add(max_size, sat_mul(kV2DescriptorSlack, sat_add(1, limits.max_checkpoints)));
            const uint64_t v2_bound = std::min(v2_max, limits.max_file_bytes);
            if (size > v2_bound) {
                error = "session file: size " + std::to_string(size) + " exceeds this engine's limit of " +
                        std::to_string(v2_bound);
                return false;
            }
        }
        st.file_version = version;   // the header is verified: the caller may tell a descriptor-format RAM refusal
                                     // from a bad file
        return true;
    }
};

bool read_impl(const std::string& path, const SessionFileIdentity& id, SavedConversation& image,
               size_t& bytes, std::string& error, const SessionReadLimits& limits, SessionStatus& st) {
    error.clear();
    st = {};
    st.error = SessionError::invalid;   // every refusal below, unless the step says otherwise
    ReadFile rf;
    uint64_t payload = 0;
    if (!rf.open(path, id, limits, payload, error, st)) return false;
    if (limits.admit) {
        std::string why;
        if (!limits.admit(session_read_peak_bytes(rf.size, limits, rf.version), why)) {
            st.error = SessionError::memory;
            error = "session file: " + why;
            return false;
        }
    }
    SavedConversation parsed;
    In in{&rf.source, SessionHasher(0), payload, error, limits};
    const bool ok = rf.version == kVersion ? get_payload(in, parsed)
                  : rf.version == kVersionV2 ? get_payload_v2(in, parsed)
                                             : get_payload_v3(in, parsed);
    if (!ok) { st.error = in.kind; return false; }
    if (in.left) { error = "session file: payload has " + std::to_string(in.left) + " unparsed bytes"; return false; }
    uint8_t t[kTrailer];
    if (!rf.source.read(t, kTrailer)) { st.error = rf.source.kind(); error = "session file: trailer read error: " + rf.source.error(); return false; }
    uint64_t ph = 0;
    std::memcpy(&ph, t, 8);
    if (ph != in.hash.digest()) { error = "session file: payload checksum mismatch"; return false; }
    if (std::memcmp(t + 8, kEnd, 8) != 0) { error = "session file: bad end marker"; return false; }
    image = std::move(parsed);
    bytes = (size_t) rf.size;
    st.error = SessionError::none;
    return true;
}

// The read pass of a streaming restore: the whole file is validated without materializing the K/V (see
// conversation_file.hpp).  The apply pass re-walks the same file with a sink, bound by the K/V sizes recorded here.
bool read_stream_impl(const std::string& path, const SessionFileIdentity& id, SavedConversation& meta,
                      size_t& bytes, std::string& error, const SessionReadLimits& limits,
                      std::vector<std::array<uint64_t, 5>>* kv_part_bytes, SessionStatus& st) {
    error.clear();
    st = {};
    st.error = SessionError::invalid;
    ReadFile rf;
    uint64_t payload = 0;
    if (!rf.open(path, id, limits, payload, error, st)) return false;
    if (limits.admit) {
        std::string why;
        if (!limits.admit(session_read_stream_peak_bytes(rf.size, limits, rf.version), why)) {
            st.error = SessionError::memory;
            error = "session file: " + why;
            return false;
        }
    }
    SavedConversation parsed;
    In in{&rf.source, SessionHasher(0), payload, error, limits};
    in.record_kv = kv_part_bytes;
    const bool ok = rf.version == kVersion ? get_payload_streamed(in, parsed, {})
                  : rf.version == kVersionV2 ? get_payload_streamed_v2(in, parsed, {})
                                             : get_payload_streamed_v3(in, parsed, {});
    if (!ok) { st.error = in.kind; return false; }
    if (in.left) { error = "session file: payload has " + std::to_string(in.left) + " unparsed bytes"; return false; }
    uint8_t t[kTrailer];
    if (!rf.source.read(t, kTrailer)) { st.error = rf.source.kind(); error = "session file: trailer read error: " + rf.source.error(); return false; }
    uint64_t ph = 0;
    std::memcpy(&ph, t, 8);
    if (ph != in.hash.digest()) { error = "session file: payload checksum mismatch"; return false; }
    if (std::memcmp(t + 8, kEnd, 8) != 0) { error = "session file: bad end marker"; return false; }
    meta = std::move(parsed);
    bytes = (size_t) rf.size;
    st.error = SessionError::none;
    return true;
}

// The apply pass: the file is walked again and every K/V block goes to the sink.  `kv_part_bytes` is the read
// pass's list: this walk refuses a file whose layer count or part sizes differ before applying that layer or part,
// so the sink only ever sees the bytes the read pass admitted.  The payload hash is recomputed and compared to the
// trailer at the end - a same-size divergence since the read pass ends the caller (fail-stop).  Any failure past
// the first sink call leaves the caller's state partly written - the caller must fail-stop.  A v3 file's state
// section is consumed HASH-ONLY in this walk (no second decode): its protection is the read pass's full decode plus
// this same trailing hash, so a v3 state corruption surfaces no later than the trailer check - after any K/V
// reached the sink - and takes the identical fail-stop path.
bool apply_stream_impl(const std::string& path, const SessionFileIdentity& id, const SessionKvBlock& sink,
                       const std::vector<std::array<uint64_t, 5>>& kv_part_bytes, std::string& error,
                       SessionStatus& st, const SessionReadLimits* bounds) {
    error.clear();
    st = {};
    st.error = SessionError::invalid;
    SessionReadLimits open;   // the fallback: the read pass checked the engine's bounds (the pre-state-ref form)
    ReadFile rf;
    uint64_t payload = 0;
    if (!rf.open(path, id, bounds ? *bounds : open, payload, error, st)) return false;
    const bool desc = rf.version >= kVersionV2;   // descriptor-coded states (v2 states-first, v3 K/V-first)
    if (desc && bounds && bounds->admit) {
        // the apply pass re-parses the metadata while the read pass's copy is still resident: admit ONLY this
        // additional peak (parser metadata and one 16 MiB I/O block), before the first byte can reach the sink.  The
        // per-allocation checks below stay as overflow/container guards; the callback is not asked twice for the
        // same staging.
        std::string why;
        if (!bounds->admit(session_stream_apply_peak_bytes(*bounds), why)) {
            st.error = SessionError::memory;
            error = "session file: " + (why.empty() ? std::string("RAM admission refused") : why);
            return false;
        }
    }
    SavedConversation parsed;   // the geo/K-V headers the walk needs; v3 states are not re-decoded here (hash-only)
    SessionReadLimits apply_limits = bounds ? *bounds : open;
    apply_limits.admit = {};    // the descriptor extra peak above covered this pass's staging; a caller passing
                                // bounds MUST set admit itself (both current callers do), or the apply pass runs
                                // unguarded
    In in{&rf.source, SessionHasher(0), payload, error, apply_limits};
    in.verify_kv = &kv_part_bytes;
    const bool ok = rf.version == kVersion ? get_payload_streamed(in, parsed, sink)
                  : rf.version == kVersionV2 ? get_payload_streamed_v2(in, parsed, sink)
                                             : get_payload_apply_v3(in, parsed, sink);
    if (!ok) { st.error = in.kind; return false; }
    if (in.left) { error = "session file: payload has " + std::to_string(in.left) + " unparsed bytes"; return false; }
    uint8_t t[kTrailer];
    if (!rf.source.read(t, kTrailer)) { st.error = rf.source.kind(); error = "session file: trailer read error: " + rf.source.error(); return false; }
    uint64_t ph = 0;
    std::memcpy(&ph, t, 8);
    if (ph != in.hash.digest()) { error = "session file: the file changed since the read pass (payload checksum)"; return false; }
    if (std::memcmp(t + 8, kEnd, 8) != 0) { error = "session file: bad end marker"; return false; }
    st.error = SessionError::none;
    return true;
}
} // namespace

bool session_file_read(const std::string& path, const SessionFileIdentity& id, SavedConversation& image,
                       size_t& bytes, std::string& error, const SessionReadLimits& limits, SessionStatus* status) {
    SessionStatus st;
    bool ok = false;
    try {
        ok = read_impl(path, id, image, bytes, error, limits, st);
    } catch (const std::bad_alloc&) {
        error = "session file: out of memory while reading " + path;
        st.error = SessionError::memory;
        ok = false;
    }
    if (status) *status = st;
    return ok;
}

bool session_file_read_streamed(const std::string& path, const SessionFileIdentity& id, SavedConversation& meta,
                                size_t& bytes, std::string& error, const SessionReadLimits& limits,
                                std::vector<std::array<uint64_t, 5>>* kv_part_bytes, SessionStatus* status) {
    SessionStatus st;
    bool ok = false;
    try {
        ok = read_stream_impl(path, id, meta, bytes, error, limits, kv_part_bytes, st);
    } catch (const std::bad_alloc&) {
        error = "session file: out of memory while reading " + path;
        st.error = SessionError::memory;
        ok = false;
    }
    if (status) *status = st;
    return ok;
}

bool session_file_apply_streamed(const std::string& path, const SessionFileIdentity& id, const SessionKvBlock& sink,
                                 const std::vector<std::array<uint64_t, 5>>& kv_part_bytes, std::string& error,
                                 SessionStatus* status, const SessionReadLimits* bounds) {
    SessionStatus st;
    bool ok = false;
    try {
        ok = apply_stream_impl(path, id, sink, kv_part_bytes, error, st, bounds);
    } catch (const std::bad_alloc&) {
        error = "session file: out of memory while reading " + path;
        st.error = SessionError::memory;
        ok = false;
    }
    if (status) *status = st;
    return ok;
}

void session_read_bounds_from_meta(const SavedConversation& meta, SessionReadLimits& out) {
    out = SessionReadLimits{};
    out.max_checkpoints = meta.checkpoints.size();
    out.max_kv_layers = meta.kv.size();
    out.geometry = meta.geometry;
    out.layer_range = std::make_pair(meta.layer_lo, meta.layer_hi);
    out.max_state_bytes.fill(0);
    // SessionReadLimits{} leaves every bound open (UINT64_MAX); this pass needs exact maxima so the apply pass's
    // admission prices a finite peak - an open bound would price UINT64_MAX and refuse every file
    out.max_tokens = 0;
    const auto include = [&](const ConversationCheckpoint& c) {
        out.max_tokens = std::max<uint64_t>(out.max_tokens, c.ids.size());
        out.max_tokens = std::max<uint64_t>(out.max_tokens, c.imgs.size());
        out.max_state_bytes[0] = std::max<uint64_t>(out.max_state_bytes[0], c.gdn.size());
        out.max_state_bytes[1] = std::max<uint64_t>(out.max_state_bytes[1], c.ple.size());
        out.max_state_bytes[2] = std::max<uint64_t>(out.max_state_bytes[2], c.tails.size());
        out.max_state_bytes[3] = std::max<uint64_t>(out.max_state_bytes[3], c.dead.size());
        out.max_state_bytes[4] = std::max<uint64_t>(out.max_state_bytes[4], c.block_pos.size());
    };
    include(meta.live);
    for (const auto& c : meta.checkpoints) include(c);
}

uint64_t session_stream_apply_peak_bytes(const SessionReadLimits& bounds) {
    // The apply pass's freshly parsed metadata (bounded by `bounds`; session_read_meta_peak_bytes prices one decoded
    // metadata state), one 16 MiB K/V block and the small I/O state - what the apply pass adds while the read pass's
    // metadata is still resident.  Checked/saturating; streamed K/V payload length is excluded.  The apply pass's
    // ReadFile (its 16 MiB FileSource block) is constructed BEFORE this peak is admitted, so that block is already
    // allocated and reflected in the available memory the callback sees - only this one block is priced here.
    return sat_add(session_read_meta_peak_bytes(bounds), sat_add(kBlock, uint64_t(1) << 20));
}

// ---- delta records: the append, the read pass and the apply pass (see conversation_file.hpp)
namespace {
bool delta_write_impl(const std::string& path, uint64_t at, const SavedConversation& meta,
                      const std::vector<SessionKvSource>& kv, int64_t first_token, const SessionFileIdentity& id,
                      uint64_t& bytes, std::string& error, const SessionWriteOptions& opt, SessionStatus& st) {
    st = {};
    st.error = SessionError::invalid;   // refusals before any disk access
    if (has_stage_parts(meta)) { error = "delta record: layer-split state cannot be saved"; return false; }
    if (!meta.kv.empty()) { error = "delta record: the metadata must not carry K/V buffers"; return false; }
    if (opt.background_state && !opt.layout_v3) {
        error = "delta record: background-state writes require the v3 layout (layout_v3)";
        return false;   // st.error is still 'invalid'
    }
    for (const auto& s : kv)
        for (size_t p = 0; p < 5; ++p)
            if (s.first_bytes[p] > s.sizes[p]) { error = "delta record: a K/V offset is past its part"; return false; }
    rht::Layout codec_layout{};
    const std::vector<StateCodecArray> codec = state_codecs(meta, opt.state_codec_rht, {}, codec_layout);
    const uint64_t record_version = opt.layout_v3 ? kDeltaVersionV3
                                    : any_state_codec2(codec) ? kDeltaVersion : kDeltaVersionV1;
    Out sizing;
    put_delta_record(sizing, id, first_token, meta, kv, record_version, codec, codec_layout);
    if (sizing.overflow || sizing.n > UINT64_MAX - kTrailer) {
        error = "delta record: the state is too large for this build";
        return false;
    }
    const uint64_t total = sizing.n + kTrailer;   // the payload hash and the end marker
    st.error = SessionError::io;
    if (opt.min_free_bytes) {
        uint64_t avail = 0;
        if (!free_space(path, avail)) { error = "delta record: cannot read the free disk space for " + path; return false; }
        if (avail < total || avail - total < opt.min_free_bytes) {
            st.error = SessionError::storage;
            error = "delta record: not enough disk space (" + std::to_string(total >> 20) + " MiB plus a reserve of " +
                    std::to_string(opt.min_free_bytes >> 20) + " MiB needed, " + std::to_string(avail >> 20) +
                    " MiB free)";
            return false;
        }
    }
    RawFile raw;
    uint64_t have = 0;
    if (!raw.open_append(path, have)) {
        st.error = raw.refused() ? SessionError::invalid : raw.kind();
        error = "delta record: cannot open " + path + ": " + raw.error();
        return false;
    }
    SourcesReleased released(opt.on_sources_released);
    // the committed end is the append point: anything past it (an interrupted append) is dropped before writing
    if (!raw.truncate(at) || !raw.seek(at)) {
        st.error = raw.kind();
        error = "delta record: cannot position " + path + " at " + std::to_string(at) + ": " + raw.error();
        return false;
    }
    const std::function<int(const char*)>* fault = opt.fault ? &opt.fault : nullptr;
    const auto drive = [&](auto& sink) -> bool {
        using Sink = std::remove_reference_t<decltype(sink)>;
        if (opt.progress) sink.progress.emplace(opt.progress);
        sink.phase = opt.phase;
        sink.expected = total;
        sink.timings = opt.timings;
        if (sink.progress) (*sink.progress)(0, total);
        sink.write_fault = injected(fault, "write");
        sink.flush_fault = injected(fault, "file_flush");
        SessionHasher hash(0);
        Out out;
        if constexpr (std::is_same_v<Sink, AsyncFileSink>) out.af = &sink;
        else out.f = &sink;
        out.hash = &hash;
        out.timings = opt.timings;
        out.released = &released;
        put_delta_record(out, id, first_token, meta, kv, record_version, codec, codec_layout);
        bool ok = out.ok && out.n == sizing.n;
        if (ok) {
            const uint64_t ph = hash.digest();
            ok = sink.write(&ph, 8) && sink.write(kEnd, 8);
        }
        ok = sink.finish(opt.durable) && ok;
        if (!ok) {
            const std::string why = out.read_failed
                                        ? "K/V source read" + (out.read_error.empty() ? std::string() : ": " + out.read_error)
                                        : sink.error();
            st.error = out.read_failed ? SessionError::io : sink.kind();
            error = "delta record: writing " + path + " failed" + (why.empty() ? std::string() : ": " + why);
            return false;
        }
        st.error = SessionError::none;
        bytes = total;
        return true;
    };
    if (opt.background_state) {
        AsyncFileSink sink(raw);
        if (!sink.ok()) { st.error = SessionError::memory; error = "delta record: " + sink.error(); return false; }
        return drive(sink);
    }
    FileSink sink(raw);
    if (!sink.ok()) { st.error = SessionError::memory; error = "delta record: " + sink.error(); return false; }
    return drive(sink);
}

// a delta record's K/V directory walk (the read pass): the layer count, each layer's header, each part's
// (offset, size), then the part's bytes read for the hash only.  Fills `parsed`.kv's headers and the record's
// offset/size directory.  Shared by every record version; the caller walks it after the state section (v1/v2
// records) or before it (v3 records).
bool delta_kv_dir_read(DeltaIn& in, SavedConversation& parsed, SessionDeltaRecord& result) {
    uint64_t n = 0;
    if (!in.count(n, 96, in.limits.max_kv_layers)) return false;
    if (!in.resize(parsed.kv, n) || !in.resize_zero(result.kv_offset, n) || !in.resize_zero(result.kv_bytes, n))
        return false;
    Aligned block;
    if (n) {
        if (!in.prepare_allocation(kBlock, 1)) return false;
        block = aligned_block();
        if (!block) { in.kind = SessionError::memory; return in.fail("out of memory for the K/V read block"); }
    }
    for (size_t l = 0; l < (size_t) n; ++l) {
        auto& k = parsed.kv[l];
        in.kv_layer = l;
        int64_t format = 0;
        if (!in.i64(format) || !in.i64(k.cells) || !in.i64(k.heads) || !in.i64(k.head_dim) ||
            !in.i64(k.page_size) || !in.i64(k.pooled_rows) || !in.i64(k.idx_dim)) return false;
        if (format < INT32_MIN || format > INT32_MAX) { in.fail("invalid K/V format"); return false; }
        k.format = (int) format;
        for (size_t part = 0; part < 5; ++part) {
            uint64_t part_at = 0, part_bytes = 0;
            if (!in.u64(part_at) || !in.u64(part_bytes)) return false;
            const uint64_t max = l < in.limits.max_kv_bytes.size() ? in.limits.max_kv_bytes[l][part] : UINT64_MAX;
            if (part_at > max || part_bytes > max - part_at) {
                in.fail("K/V part beyond this engine's limit");
                in.kind = SessionError::invalid;
                return false;
            }
            result.kv_offset[l][part] = part_at;
            result.kv_bytes[l][part] = part_bytes;
            for (uint64_t done = 0; done < part_bytes;) {
                const size_t c = (size_t) std::min<uint64_t>(kBlock, part_bytes - done);
                if (!in.raw(block.get(), c)) return false;
                done += (uint64_t) c;
            }
        }
    }
    return true;
}

// the apply pass's K/V directory walk: re-reads the directory and refuses a record that differs from `expected`
// (a K/V header, an offset or a size) BEFORE that part's first block reaches the sink, then hands every block to
// the sink at (offset + done within the part).
bool delta_kv_dir_apply(DeltaIn& in, const SessionDeltaRecord& expected, const SessionKvBlock& sink) {
    uint64_t n = 0;
    if (!in.count(n, 96, in.limits.max_kv_layers)) return false;
    if (n != expected.kv_offset.size()) return in.fail("K/V layer count differs from the read pass");
    Aligned block;
    if (n) {
        if (!in.prepare_allocation(kBlock, 1)) return false;
        block = aligned_block();
        if (!block) { in.kind = SessionError::memory; return in.fail("out of memory for the K/V read block"); }
    }
    for (size_t l = 0; l < (size_t) n; ++l) {
        in.kv_layer = l;
        int64_t format = 0, cells = 0, heads = 0, head_dim = 0, page_size = 0, pooled_rows = 0, idx_dim = 0;
        if (!in.i64(format) || !in.i64(cells) || !in.i64(heads) || !in.i64(head_dim) ||
            !in.i64(page_size) || !in.i64(pooled_rows) || !in.i64(idx_dim)) return false;
        const ConversationKv& want = expected.meta.kv[l];
        if (format != want.format || cells != want.cells || heads != want.heads || head_dim != want.head_dim ||
            page_size != want.page_size || pooled_rows != want.pooled_rows || idx_dim != want.idx_dim)
            return in.fail("a K/V header differs from the read pass");
        for (size_t part = 0; part < 5; ++part) {
            uint64_t part_at = 0, part_bytes = 0;
            if (!in.u64(part_at) || !in.u64(part_bytes)) return false;
            if (part_at != expected.kv_offset[l][part] || part_bytes != expected.kv_bytes[l][part])
                return in.fail("a K/V part size differs from the read pass");
            for (uint64_t done = 0; done < part_bytes;) {
                const size_t c = (size_t) std::min<uint64_t>(kBlock, part_bytes - done);
                if (!in.raw(block.get(), c)) return false;
                if (sink && !sink(l, part, part_at + done, block.get(), c)) {
                    if (in.error.empty()) in.error = std::string(in.what) + ": the K/V sink failed";
                    in.kind = SessionError::io;
                    return false;
                }
                done += (uint64_t) c;
            }
        }
    }
    return true;
}

bool delta_read_impl(const std::string& path, uint64_t at0, uint64_t limit, const SessionFileIdentity& id,
                     const SessionReadLimits& limits, SessionDeltaRecord& out, uint64_t& next, std::string& error,
                     SessionStatus& st) {
    error.clear();
    st = {};
    st.error = SessionError::invalid;
    RawFile f;
    uint64_t size = 0;
    if (!f.open_session(path, size, false)) {
        st.error = f.refused() ? SessionError::invalid : f.kind();
        error = "delta record: " + path + ": " + f.error();
        return false;
    }
    const uint64_t end = std::min(size, limit);
    if (at0 >= end || end - at0 < 8 * 6) { error = "delta record: " + path + ": no record at this offset"; return false; }
    DeltaIn in{&f, at0, end, SessionHasher(0), error, limits};
    char magic[8];
    uint64_t version = 0, model = 0, config = 0;
    int64_t first_token = 0;
    if (!in.raw(magic, 8) || std::memcmp(magic, kDeltaMagic, 8) != 0 || !in.u64(version) ||
        (version != kDeltaVersionV1 && version != kDeltaVersion && version != kDeltaVersionV3) ||
        !in.u64(model) || !in.u64(config) || !in.i64(first_token)) {
        st.error = in.kind;
        return false;
    }
    // the optional accepted config (a gate-approved cache entry): the model still has to match
    if (model != id.model || (config != id.config && !(limits.accept_config && config == limits.accept_config))) {
        error = "delta record: saved with another model or engine configuration";
        return false;
    }
    // built aside and moved into `out` only once the whole record checked out: a failure leaves the caller's record
    // as it was
    SessionDeltaRecord result;
    SavedConversation& parsed = result.meta;
    if (version == kDeltaVersionV3) {
        // the K/V-first record: geometry prefix, the K/V directory, then the state descriptors
        if (!get_geo_prefix_t(in, parsed)) { st.error = in.kind; return false; }
        if (!delta_kv_dir_read(in, parsed, result)) { st.error = in.kind; return false; }
        if (!get_states_v2_t(in, parsed)) { st.error = in.kind; return false; }
    } else {
        const bool parsed_meta = version == kDeltaVersion ? get_meta_prefix_v2(in, parsed)
                                                          : get_meta_prefix_t(in, parsed);
        if (!parsed_meta) { st.error = in.kind; return false; }
        if (!delta_kv_dir_read(in, parsed, result)) { st.error = in.kind; return false; }
    }
    uint64_t ph = 0;
    char endm[8];
    result.payload_end = in.at;   // the record's payload end, for the v3 apply walk's hash-only states consumption
    if (!in.tail(&ph, 8) || !in.tail(endm, 8)) { st.error = in.kind; return false; }
    if (ph != in.hash.digest()) { error = "delta record: payload checksum mismatch"; return false; }
    if (std::memcmp(endm, kEnd, 8) != 0) { error = "delta record: bad end marker"; return false; }
    result.first_token = first_token;
    out = std::move(result);
    next = in.at;
    st.error = SessionError::none;
    return true;
}

bool delta_apply_limits(const SessionDeltaRecord& expected, SessionReadLimits& limits, std::string& error) {
    if (expected.meta.kv.size() != expected.kv_offset.size() ||
        expected.kv_offset.size() != expected.kv_bytes.size()) {
        error = "delta record: the expected layer directories differ";
        return false;
    }
    limits.max_tokens = 0;
    limits.max_checkpoints = expected.meta.checkpoints.size();
    limits.max_kv_layers = expected.meta.kv.size();
    limits.geometry = expected.meta.geometry;
    limits.layer_range = std::make_pair(expected.meta.layer_lo, expected.meta.layer_hi);
    limits.max_state_bytes.fill(0);
    auto include = [&](const ConversationCheckpoint& checkpoint) {
        limits.max_tokens = std::max<uint64_t>(limits.max_tokens, checkpoint.ids.size());
        limits.max_tokens = std::max<uint64_t>(limits.max_tokens, checkpoint.imgs.size());
        limits.max_state_bytes[0] = std::max<uint64_t>(limits.max_state_bytes[0], checkpoint.gdn.size());
        limits.max_state_bytes[1] = std::max<uint64_t>(limits.max_state_bytes[1], checkpoint.ple.size());
        limits.max_state_bytes[2] = std::max<uint64_t>(limits.max_state_bytes[2], checkpoint.tails.size());
        limits.max_state_bytes[3] = std::max<uint64_t>(limits.max_state_bytes[3], checkpoint.dead.size());
        limits.max_state_bytes[4] = std::max<uint64_t>(limits.max_state_bytes[4], checkpoint.block_pos.size());
    };
    include(expected.meta.live);
    for (const ConversationCheckpoint& checkpoint : expected.meta.checkpoints) include(checkpoint);
    return true;
}

bool delta_apply_impl(const std::string& path, uint64_t at0, const SessionFileIdentity& id,
                      const SessionDeltaRecord& expected, const SessionKvBlock& sink, std::string& error,
                      uint64_t accept_config, SessionStatus& st) {
    error.clear();
    st = {};
    st.error = SessionError::invalid;
    RawFile f;
    uint64_t size = 0;
    if (!f.open_session(path, size, false)) {
        st.error = f.refused() ? SessionError::invalid : f.kind();
        error = "delta record: " + path + ": " + f.error();
        return false;
    }
    if (at0 >= size || size - at0 < 8 * 6) { error = "delta record: " + path + ": no record at this offset"; return false; }
    SessionReadLimits bounded;
    if (!delta_apply_limits(expected, bounded, error)) { st.error = SessionError::invalid; return false; }
    DeltaIn in{&f, at0, size, SessionHasher(0), error, bounded};
    char magic[8];
    uint64_t version = 0, model = 0, config = 0;
    int64_t first_token = 0;
    if (!in.raw(magic, 8) || std::memcmp(magic, kDeltaMagic, 8) != 0 || !in.u64(version) ||
        (version != kDeltaVersionV1 && version != kDeltaVersion && version != kDeltaVersionV3) ||
        !in.u64(model) || !in.u64(config) || !in.i64(first_token)) { st.error = in.kind; return false; }
    if (model != id.model || (config != id.config && !(accept_config && config == accept_config))) {
        error = "delta record: saved with another model or engine configuration";
        return false;
    }
    if (first_token != expected.first_token) { in.fail("the record's first token differs from the read pass"); st.error = in.kind; return false; }
    SavedConversation parsed;
    if (version == kDeltaVersionV3) {
        // the K/V-first record: only the geometry prefix is parsed here; the state section is consumed HASH-ONLY
        // after the K/V directory (below) - exactly the protection the K/V gets: the read pass's full decode plus
        // the trailing payload hash.  A live/checkpoint divergence is a state-section change, caught by that hash
        // no later than the trailer check, after the record's K/V reached the sink (the caller fails stops, as for
        // a K/V change).
        if (!get_geo_prefix_t(in, parsed)) { st.error = in.kind; return false; }
    } else {
        const bool parsed_meta = version == kDeltaVersion ? get_meta_prefix_v2(in, parsed)
                                                          : get_meta_prefix_t(in, parsed);
        if (!parsed_meta) { st.error = in.kind; return false; }
    }
    // the v3 record's live/checkpoint counts come from its (hash-only) state section: they cannot be compared here
    const bool meta_differs =
        version == kDeltaVersionV3
            ? (parsed.geometry != expected.meta.geometry || parsed.layer_lo != expected.meta.layer_lo ||
               parsed.layer_hi != expected.meta.layer_hi || parsed.cvec != expected.meta.cvec)
            : (parsed.live.ids.size() != expected.meta.live.ids.size() || parsed.geometry != expected.meta.geometry ||
               parsed.layer_lo != expected.meta.layer_lo || parsed.layer_hi != expected.meta.layer_hi ||
               parsed.cvec != expected.meta.cvec || parsed.checkpoints.size() != expected.meta.checkpoints.size());
    if (meta_differs) {
        in.fail("the record differs from the read pass");
        st.error = in.kind;
        return false;
    }
    if (!delta_kv_dir_apply(in, expected, sink)) { st.error = in.kind; return false; }
    if (version == kDeltaVersionV3) {
        // the state section, in record order: consumed HASH-ONLY - every byte between the K/V directory and the
        // record's trailer feeds the running hash with no decode, exactly the bytes the read pass decoded and hashed
        if (expected.payload_end < in.at) { in.fail("the record's state section overlaps its trailer"); st.error = in.kind; return false; }
        const uint64_t states = expected.payload_end - in.at;
        if (states) {
            Aligned block = aligned_block();
            if (!block) { in.kind = SessionError::memory; st.error = in.kind; error = "delta record: out of memory for the state read block"; return false; }
            if (!in.hash_only(block.get(), kBlock, states)) { st.error = in.kind; return false; }
        }
    }
    uint64_t ph = 0;
    char endm[8];
    if (!in.tail(&ph, 8) || !in.tail(endm, 8)) { st.error = in.kind; return false; }
    if (ph != in.hash.digest()) { error = "delta record: the record changed since the read pass (payload checksum)"; return false; }
    if (std::memcmp(endm, kEnd, 8) != 0) { error = "delta record: bad end marker"; return false; }
    st.error = SessionError::none;
    return true;
}
} // namespace

bool session_delta_write(const std::string& path, uint64_t at, const SavedConversation& meta,
                         const std::vector<SessionKvSource>& kv, int64_t first_token, const SessionFileIdentity& id,
                         uint64_t& bytes, std::string& error, const SessionWriteOptions& options, SessionStatus* status) {
    SessionStatus st;
    bool ok = false;
    try {
        ok = delta_write_impl(path, at, meta, kv, first_token, id, bytes, error, options, st);
    } catch (const std::bad_alloc&) {
        error = "delta record: out of memory while writing " + path;
        st.error = SessionError::memory;
        ok = false;
    }
    if (status) *status = st;
    return ok;
}

bool session_delta_read(const std::string& path, uint64_t at, uint64_t limit, const SessionFileIdentity& id,
                        const SessionReadLimits& limits, SessionDeltaRecord& out, uint64_t& next, std::string& error,
                        SessionStatus* status) {
    SessionStatus st;
    bool ok = false;
    try {
        ok = delta_read_impl(path, at, limit, id, limits, out, next, error, st);
    } catch (const std::bad_alloc&) {
        error = "delta record: out of memory while reading " + path;
        st.error = SessionError::memory;
        ok = false;
    }
    if (status) *status = st;
    return ok;
}

bool session_delta_apply(const std::string& path, uint64_t at, const SessionFileIdentity& id,
                         const SessionDeltaRecord& expected, const SessionKvBlock& sink, std::string& error,
                         SessionStatus* status, uint64_t accept_config) {
    SessionStatus st;
    bool ok = false;
    try {
        ok = delta_apply_impl(path, at, id, expected, sink, error, accept_config, st);
    } catch (const std::bad_alloc&) {
        error = "delta record: out of memory while reading " + path;
        st.error = SessionError::memory;
        ok = false;
    }
    if (status) *status = st;
    return ok;
}

uint64_t session_delta_record_bytes(const SavedConversation& meta, const std::vector<SessionKvSource>& kv,
                                    bool* overflow, bool state_codec_rht, bool layout_v3) {
    if (overflow) *overflow = false;
    rht::Layout codec_layout{};
    const std::vector<StateCodecArray> codec = state_codecs(meta, state_codec_rht, {}, codec_layout);
    const uint64_t record_version = layout_v3 ? kDeltaVersionV3
                                    : any_state_codec2(codec) ? kDeltaVersion : kDeltaVersionV1;
    Out sizing;
    put_delta_record(sizing, SessionFileIdentity{}, 0, meta, kv, record_version, codec, codec_layout);
    if (sizing.overflow || sizing.n > UINT64_MAX - kTrailer) {
        if (overflow) *overflow = true;
        return 0;
    }
    return sizing.n + kTrailer;
}

bool session_file_full_bytes(const SavedConversation& image, const std::vector<SessionKvSource>& sources,
                             uint64_t& bytes, std::string& error) {
    bytes = 0;
    error.clear();
    if (has_stage_parts(image)) {
        error = "session file: layer-split state cannot be saved";
        return false;
    }
    if (!image.kv.empty()) {
        error = "session file: K/V given both in the image and as sources";
        return false;
    }
    Out sizing;
    put_payload(sizing, image, &sources);
    if (sizing.overflow || sizing.n > UINT64_MAX - kHeader - kTrailer ||
        sizing.n + kHeader + kTrailer > (uint64_t) std::numeric_limits<size_t>::max()) {
        error = "session file: the state is too large for this build";
        return false;
    }
    bytes = sizing.n + kHeader + kTrailer;
    return true;
}

bool session_delta_first_bytes(SessionKvSource& s, int64_t kept_tokens, bool index, int64_t idx_block,
                               int64_t idx_key_dim, std::string& error) {
    if (kept_tokens < 0 || s.page_size <= 0 || s.cells < 0 || (index && (idx_block <= 0 || idx_key_dim <= 0))) {
        error = "delta record: invalid geometry for a K/V offset";
        return false;
    }
    const int64_t whole_cells = (kept_tokens / s.page_size) * s.page_size;
    for (size_t part = 0; part < 5; ++part) {
        const size_t keep = part == 4
            ? (index ? size_t(kept_tokens / idx_block) * (size_t) idx_key_dim * 4 : 0)
            : (s.cells ? (s.sizes[part] / (size_t) s.cells) * (size_t) whole_cells : 0);
        if (keep > s.sizes[part]) {
            error = "delta record: a K/V offset is past its part";
            return false;
        }
        s.first_bytes[part] = keep;
    }
    return true;
}

namespace {
// A delta record stream beside a base file: its magic is not a session file's, so the digest can name it instead of
// reporting "not a Strata session file".
bool is_delta_stream(const std::string& path) {
    RawFile raw;
    uint64_t size = 0;
    char magic[8];
    // buffered: the eight bytes are read at offset 0, which need not satisfy O_DIRECT's alignment
    return raw.open_session(path, size, false) && size >= 8 &&
           raw.read_at(0, reinterpret_cast<uint8_t*>(magic), 8) && std::memcmp(magic, kDeltaMagic, 8) == 0;
}

// The digest's token part: the ids as stored (int32, little-endian) and every image key, framed by the builder.
void digest_tokens(SessionIdentityBuilder& b, const ConversationCheckpoint& c) {
    b.u64("ids.size", c.ids.size());
    b.bytes("ids", c.ids.data(), c.ids.size() * sizeof(int32_t));
    b.u64("imgs.size", c.imgs.size());
    for (const auto& k : c.imgs) {
        b.i64("img.start", k.start);
        b.u64("img.hash", k.hash);
    }
}

// The digest's state part: the arrays a restore decodes (REF already resolved to the live bytes, RHT already decoded),
// in a fixed order so a v1, a v2 and a v3 file of the same state agree.
void digest_state(SessionIdentityBuilder& b, const ConversationCheckpoint& c) {
    b.bytes("gdn", c.gdn.data(), c.gdn.size());
    b.bytes("ple", c.ple.data(), c.ple.size());
    b.bytes("tails", c.tails.data(), c.tails.size());
    b.bytes("dead", c.dead.data(), c.dead.size());
    b.bytes("block_pos", c.block_pos.data(), c.block_pos.size());
    b.u64("used", c.used);
}

bool digest_impl(const std::string& path, const SessionFileIdentity& id, SessionFileDigest& out, std::string& error,
                 SessionStatus& st) {
    error.clear();
    st = {};
    st.error = SessionError::invalid;   // every refusal below, unless the step says otherwise
    if (is_delta_stream(path)) {
        error = "session digest: " + path + " is a delta record stream, not a base session file";
        return false;
    }
    ReadFile rf;
    uint64_t payload = 0;
    const SessionReadLimits limits;   // the file's own counts bound it; a digest has no runtime geometry
    if (!rf.open(path, id, limits, payload, error, st)) return false;
    SavedConversation meta;
    SessionHasher blocks(0);
    const SessionKvBlock sink = [&blocks](size_t layer, size_t part, uint64_t offset, const void* data, size_t n) {
        const uint64_t frame[4] = {(uint64_t) layer, (uint64_t) part, offset, (uint64_t) n};
        blocks.update(frame, sizeof frame);
        blocks.update(data, n);
        return true;
    };
    In in{&rf.source, SessionHasher(0), payload, error, limits};
    bool saw_ref = false, saw_rht = false;
    in.saw_ref = &saw_ref;
    in.saw_rht = &saw_rht;
    const bool ok = rf.version == kVersion ? get_payload_streamed(in, meta, sink)
                  : rf.version == kVersionV2 ? get_payload_streamed_v2(in, meta, sink)
                                             : get_payload_streamed_v3(in, meta, sink);
    if (!ok) { st.error = in.kind; return false; }
    if (in.left) { error = "session file: payload has " + std::to_string(in.left) + " unparsed bytes"; return false; }
    uint8_t t[kTrailer];
    if (!rf.source.read(t, kTrailer)) {
        st.error = rf.source.kind();
        error = "session file: trailer read error: " + rf.source.error();
        return false;
    }
    uint64_t ph = 0;
    std::memcpy(&ph, t, 8);
    if (ph != in.hash.digest()) { error = "session file: payload checksum mismatch"; return false; }
    if (std::memcmp(t + 8, kEnd, 8) != 0) { error = "session file: bad end marker"; return false; }

    SessionFileDigest digest;
    digest.file_version = rf.version;
    SessionIdentityBuilder tokens(0), state(0);
    digest_tokens(tokens, meta.live);
    digest_state(state, meta.live);
    for (const auto& c : meta.checkpoints) {
        digest_tokens(tokens, c);
        digest_state(state, c);
    }
    digest.tokens = tokens.digest();
    digest.state = state.digest();
    // the layer headers, then the block stream (which frames each layer/part/offset), so a different header can never
    // pass for the same bytes
    SessionHasher kv(0);
    const uint64_t layers = meta.kv.size();
    kv.update(&layers, sizeof layers);
    for (const auto& k : meta.kv) {
        const int64_t header[7] = {k.format, k.cells, k.heads, k.head_dim, k.page_size, k.pooled_rows, k.idx_dim};
        kv.update(header, sizeof header);
    }
    const uint64_t streamed = blocks.digest();
    kv.update(&streamed, sizeof streamed);
    digest.kv = kv.digest();
    digest.state_ref = saw_ref;
    digest.state_rht = saw_rht;
    digest.state_exact = !saw_rht;
    out = digest;
    st.error = SessionError::none;
    return true;
}
} // namespace

bool session_file_digest(const std::string& path, const SessionFileIdentity& id, SessionFileDigest& out,
                         std::string& error, SessionStatus* status) {
    SessionStatus st;
    bool ok = false;
    try {
        ok = digest_impl(path, id, out, error, st);
    } catch (const std::bad_alloc&) {
        error = "session digest: out of memory while reading " + path;
        st.error = SessionError::memory;
        ok = false;
    }
    if (status) *status = st;
    return ok;
}

} // namespace strata::core
