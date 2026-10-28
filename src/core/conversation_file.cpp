#include "strata/core/conversation_file.hpp"

#include <algorithm>
#include <atomic>
#include <bit>
#include <cerrno>
#include <chrono>
#include <climits>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <condition_variable>
#include <new>
#include <unordered_map>
#include <optional>
#include <filesystem>
#include <limits>
#include <memory>
#include <mutex>
#include <random>
#include <string>
#include <system_error>
#include <thread>
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

// Format v2 stores integers and floats as this machine holds them in memory; it is defined as little-endian with
// IEEE-754 floats, so a build where that is not the memory layout is refused rather than writing another format.
static_assert(std::endian::native == std::endian::little, "session files (format v2) are little-endian only");
static_assert(std::numeric_limits<float>::is_iec559 && std::numeric_limits<double>::is_iec559,
              "session files (format v2) need IEEE-754 floats");

namespace strata::core {
namespace {
constexpr char kMagic[8] = {'S', 'T', 'R', 'S', 'E', 'S', 'S', '\x01'};
constexpr char kEnd[8] = {'S', 'T', 'R', 'S', 'E', 'N', 'D', '\x01'};
constexpr uint32_t kVersion = 2;
constexpr size_t kHeader = 64, kTrailer = 16;

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
    bool open_session(const std::string& path, uint64_t& size) {
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
    // The append target of an existing session (v3 append path): the name itself, never a link or reparse
    // point, a regular disk file with one name; the pointer sits at EOF for the group's block writes.
    bool open_append(const std::string& path, uint64_t& size) {
        std::wstring w;
        if (!wide(path, w)) { refused_ = true; error_ = "the path is not valid UTF-8"; return false; }
        h_ = CreateFileW(w.c_str(), GENERIC_WRITE, FILE_SHARE_READ, nullptr, OPEN_EXISTING,
                         FILE_FLAG_OPEN_REPARSE_POINT, nullptr);
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
        LARGE_INTEGER end{};
        if (!SetFilePointerEx(h_, end, nullptr, FILE_END)) { { const DWORD c_ = GetLastError(); kind_ = win_kind(c_); error_ = last_error(c_); } return false; }
        return true;
    }
    // in-place update at a fixed offset (the 64 B header rewrite); the pointer returns to EOF for the appends
    bool write_at(uint64_t offset, const uint8_t* p, size_t n) {
        LARGE_INTEGER at{};
        at.QuadPart = (LONGLONG) offset;
        if (!SetFilePointerEx(h_, at, nullptr, FILE_BEGIN)) { { const DWORD c_ = GetLastError(); kind_ = win_kind(c_); error_ = last_error(c_); } return false; }
        while (n) {
            DWORD w = 0;
            const DWORD want = (DWORD) std::min<size_t>(n, size_t(1) << 30);
            if (!WriteFile(h_, p, want, &w, nullptr) || w == 0) { { const DWORD c_ = GetLastError(); kind_ = win_kind(c_); error_ = last_error(c_); } return false; }
            p += w; n -= w;
        }
        LARGE_INTEGER end{};
        if (!SetFilePointerEx(h_, end, nullptr, FILE_END)) { { const DWORD c_ = GetLastError(); kind_ = win_kind(c_); error_ = last_error(c_); } return false; }
        return true;
    }
    // skip n bytes forward (the framing walk seeks over segment zones instead of reading them)
    bool seek(uint64_t n) {
        LARGE_INTEGER d{};
        d.QuadPart = (LONGLONG) n;
        if (!SetFilePointerEx(h_, d, nullptr, FILE_CURRENT)) { { const DWORD c_ = GetLastError(); kind_ = win_kind(c_); error_ = last_error(c_); } return false; }
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
    bool is_open() const { return h_ != INVALID_HANDLE_VALUE; }
    bool direct() const { return false; }
    void force_buffered() {}
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
            return dir_flush(error, true, st, fault);
        }
        const DWORD c = GetLastError();
        st.error = win_kind(c);
        error = last_error(c);
        return false;
    }
    // Windows has no folder flush (the rename is WRITE_THROUGH); the seam stands so the append path keeps
    // the same testable honesty semantics as POSIX.
    bool dir_flush(std::string& error, bool, SessionStatus& st, const std::function<int(const char*)>* fault,
                   const char* step = "dir_flush") {
        if (const int e = injected(fault, step)) {
            error = std::string("flushing the directory: ") + std::strerror(e);
            st.error = errno_kind(e);
            return false;
        }
        return true;
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
    // regular file with one name (no hard link to a file elsewhere).  Checked on the opened descriptor.
    bool open_session(const std::string& path, uint64_t& size) {
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
        try_direct();
        return true;
    }
    // The append target of an existing session (v3 append path): the name itself is opened, never what a
    // symbolic link there points to; a regular file with one name, and a FIFO cannot hang it.  The descriptor
    // is deliberately left buffered: a v3 append group starts 64 bytes into a 512-byte grid - an alignment no
    // direct transfer could ever use.  NOT O_APPEND: Linux's generic write path would apply O_APPEND to
    // pwrite too, and the header rewrites must land at offset 0 - the file offset is positioned at EOF here,
    // so the group's sequential writes still append.
    bool open_append(const std::string& path, uint64_t& size) {
        fd_ = ::open(path.c_str(), O_WRONLY | O_NOFOLLOW | O_NONBLOCK | O_CLOEXEC);
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
        if (::lseek(fd_, (off_t) st.st_size, SEEK_SET) < 0) { error_ = std::strerror(errno); return false; }
        size = (uint64_t) st.st_size;
        return true;   // no try_direct(): the append position is not alignment-friendly by construction
    }
    // in-place update at a fixed offset (the 64 B header rewrite): O_APPEND does not apply to pwrite, so
    // later write() calls still land at EOF
    bool write_at(uint64_t offset, const uint8_t* p, size_t n) {
        while (n) {
            const ssize_t w = ::pwrite(fd_, p, n, (off_t) offset);
            if (w < 0 && errno == EINTR) continue;
            if (w <= 0) { kind_ = w < 0 ? errno_kind(errno) : SessionError::io; error_ = w < 0 ? std::strerror(errno) : "short write"; return false; }
            p += w; n -= (size_t) w; offset += (uint64_t) w;
        }
        return true;
    }
    // skip n bytes forward (the framing walk seeks over segment zones instead of reading them)
    bool seek(uint64_t n) {
        if (::lseek(fd_, (off_t) n, SEEK_CUR) < 0) { kind_ = errno_kind(errno); error_ = std::strerror(errno); return false; }
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
    bool is_open() const { return fd_ >= 0; }
    bool direct() const { return direct_; }
    // small random reads (the v3 checkpoint probes) do not benefit from O_DIRECT and must
    // not be constrained by its alignment rules
    void force_buffered() { if (direct_) leave_direct(); }
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
        return dir_flush(error, durable, st, fault);
    }
    // Folder flush, shared with the v3 append path's "append_dir_flush" seam: same honesty semantics.
    bool dir_flush(std::string& error, bool durable, SessionStatus& st, const std::function<int(const char*)>* fault,
                   const char* step = "dir_flush") {
        if (!durable) return true;
        int e = injected(fault, step);
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
    explicit FileSink(RawFile& f) : buf_{aligned_block(), aligned_block()}, f_(f) {}
    ~FileSink() { join_writer(); }
    bool ok() const { return buf_[0] && buf_[1]; }
    std::string error() const {
        if (!buf_[0] || !buf_[1]) return "out of memory for the write buffer";
        return fault_ ? fault_msg_ : f_.error();
    }
    SessionError kind() const { return ok() ? (fault_ ? fault_kind_ : f_.kind()) : SessionError::memory; }
    // room in the active block for in-place production (avoids a staging copy)
    uint8_t* span(size_t& room) { room = kBlock - used_; return buf_[active_].get() + used_; }
    bool commit(size_t n) { used_ += n; total_ += n; return used_ < kBlock || hand_off(); }
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
    // joins the writer, then the last block padded to the alignment for direct I/O, the file cut back to its
    // exact size and flushed: after join_writer() nothing here and no writer touches RawFile concurrently
    bool finish(bool durable) {
        join_writer();
        if (!f_.is_open()) return false;
        if (!fault_ && write_after_fault && written_.load() > 0) fail_with(write_after_fault);   // append seam: the block carrying the trailer never lands
        if (fault_) { if (!keep_open) f_.close(); return false; }
        const size_t padded = f_.direct() ? (used_ + kAlign - 1) / kAlign * kAlign : used_;
        std::memset(buf_[active_].get() + used_, 0, padded - used_);
        bool ok = f_.write_all(buf_[active_].get(), padded);
        if (ok && used_ && progress) (*progress)(total_, expected);   // the last, partial block
        ok = ok && (padded == used_ || f_.truncate(total_));
        if (ok && durable && phase) phase("flush", total_);
        if (ok && durable && flush_fault) { fail_with(flush_fault); ok = false; }
        ok = ok && (!durable || f_.sync());
        if (keep_open) { /* append: the fd stays open for the post-commit header update; the caller closes */ }
        else ok = f_.close() && ok;
        return ok;
    }
    std::optional<Ticker> progress;
    std::optional<std::function<void(uint64_t, uint64_t, uint64_t)>> block_span;   // tests only, see SessionWriteOptions
    SessionPhase phase;
    uint64_t expected = 0;
    int write_fault = 0, flush_fault = 0;   // tests: errno values injected at the first block write / the file flush
    int writer_thread_fault = 0;            // tests: nonzero forces the synchronous draining fallback
    int write_after_fault = 0;              // tests (append path): fires at the next block write once at least one
                                            // block has landed - the torn-tail seam a real crashed append leaves
    bool keep_open = false;                 // append: finish() flushes and fsyncs but keeps the fd for the header update
private:
    static uint64_t now_ns() {
        return uint64_t(std::chrono::duration_cast<std::chrono::nanoseconds>(
            std::chrono::steady_clock::now().time_since_epoch()).count());
    }
    void fail_with(int e) { fault_ = true; fault_kind_ = errno_kind(e); fault_msg_ = std::strerror(e); }
    // hands the full block to the writer and flips buffers: the producer only waits on disk when BOTH blocks
    // are working, which is how the source copy of the next block overlaps this block's write
    bool hand_off() {
        if (!writer_.joinable() && !no_writer_) {
            if (writer_thread_fault) no_writer_ = true;   // tests: the starvation seam below, forced
            else {
                try { writer_ = std::thread([this] { run_writer(); }); }
                catch (const std::system_error&) { no_writer_ = true; }   // no thread available (EAGAIN
            }                                                             // under process limits): drain
        }                                                                 // synchronously, as before
        if (no_writer_) return drain_sync();
        {
            std::unique_lock<std::mutex> lk(m_);
            if (writer_failed_) return false;
            const int next = 1 - active_;
            slot_free_.wait(lk, [&] { return ready_ != next && inflight_ != next; });
            ready_ = active_;
            active_ = next;
            used_ = 0;
            work_ready_.notify_all();
        }
        if (progress) report_written();
        return !writer_failed_;
    }
    // the un-pipelined fallback: full blocks are written in the caller's thread, exactly as the single-buffer
    // FileSink did; every save path and error channel stays the same
    bool drain_sync() {
        if (write_fault) { fail_with(write_fault); return false; }
        if (write_after_fault && written_.load() > 0) { fail_with(write_after_fault); return false; }   // append seam: a block has already landed
        const uint64_t t0 = now_ns();
        const bool ok = f_.write_all(buf_[active_].get(), kBlock);
        const uint64_t t1 = now_ns();
        if (!ok) {
            fault_ = true;
            fault_kind_ = f_.kind();
            fault_msg_ = f_.error();
            return false;
        }
        const uint64_t w = written_.fetch_add(kBlock) + kBlock;
        if (block_span) (*block_span)(w, t0, t1);
        used_ = 0;
        active_ = 1 - active_;
        if (progress) report_written();
        return true;
    }
    void run_writer() {
        for (;;) {
            int slot;
            {
                std::unique_lock<std::mutex> lk(m_);
                work_ready_.wait(lk, [&] { return ready_ >= 0 || producer_done_; });
                if (ready_ < 0) return;
                slot = ready_;
                ready_ = -1;
                inflight_ = slot;
            }
            if (write_fault) {
                std::lock_guard<std::mutex> lk(m_);
                fail_with(write_fault);
                writer_failed_ = true;
                ready_ = -1;
                inflight_ = -1;   // abandon any other pending block: the write is failing, and a producer
                slot_free_.notify_all();   // waiting on a slot must not outlive the writer that cannot free it
                return;
            }
            if (write_after_fault && written_.load() > 0) {   // append seam: a block has already landed
                std::lock_guard<std::mutex> lk(m_);
                fail_with(write_after_fault);
                writer_failed_ = true;
                ready_ = -1;
                inflight_ = -1;
                slot_free_.notify_all();
                return;
            }
            const uint64_t t0 = now_ns();
            const bool ok = f_.write_all(buf_[slot].get(), kBlock);
            const uint64_t t1 = now_ns();
            if (!ok) {
                std::lock_guard<std::mutex> lk(m_);
                fault_ = true;
                fault_kind_ = f_.kind();
                fault_msg_ = f_.error();
                writer_failed_ = true;
                ready_ = -1;
                inflight_ = -1;
                slot_free_.notify_all();
                return;
            }
            const uint64_t w = written_.fetch_add(kBlock) + kBlock;
            if (block_span) (*block_span)(w, t0, t1);   // tests: this block's OS write span
            { std::lock_guard<std::mutex> lk(m_); inflight_ = -1; }
            slot_free_.notify_all();
        }
    }
    // progress reports stay on the calling thread and follow real movement: bytes the writer has already handed
    // to the OS, never ahead, lagging at most the blocks still in flight
    void report_written() {
        const uint64_t w = written_.load();
        if (w > reported_) { reported_ = w; (*progress)(w, expected); }
    }
    void join_writer() {
        if (!writer_.joinable()) return;
        { std::lock_guard<std::mutex> lk(m_); producer_done_ = true; }
        work_ready_.notify_all();
        writer_.join();
        if (progress) report_written();   // catch up on blocks the writer finished while the producer ran ahead
    }
    Aligned buf_[2];
    RawFile& f_;
    int active_ = 0;
    bool no_writer_ = false;   // set when the writer cannot start: every block drains in the caller's thread
    std::thread writer_;
    std::mutex m_;
    std::condition_variable work_ready_, slot_free_;
    int ready_ = -1, inflight_ = -1;
    bool producer_done_ = false;
    std::atomic<bool> writer_failed_{false};
    std::atomic<uint64_t> written_{0};
    uint64_t reported_ = 0;
    bool fault_ = false;
    SessionError fault_kind_ = SessionError::io;
    std::string fault_msg_;
    size_t used_ = 0;
    uint64_t total_ = 0;
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
    // skip n bytes forward without reading them (the append-plan framing walk seeks over segment zones).
    // Buffered leftovers are dropped: the seek lands the descriptor exactly where sequential reading resumes.
    bool skip(size_t n) {
        if (n <= have_ - at_) { at_ += n; return true; }
        n -= have_ - at_;
        at_ = have_ = 0;
        if (!f_.seek(n)) return false;
        total_ += n;   // accounting only: a skip reports no progress (nothing crossed)
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
struct Out {
    FileSink* f = nullptr;
    SessionHasher* hash = nullptr;
    uint64_t n = 0;
    bool ok = true, overflow = false;
    void count(uint64_t c) {
        if (c > UINT64_MAX - n) { overflow = true; ok = false; return; }
        n += c;
    }
    void raw(const void* p, size_t c) {
        count(c);
        if (!f || !ok || !c) return;
        if (hash) hash->update(p, c);
        ok = f->write(p, c);
    }
    void u64(uint64_t v) { raw(&v, 8); }
    void i64(int64_t v) { raw(&v, 8); }
    template<class T> void vec(const std::vector<T>& v) { u64(v.size()); raw(v.data(), v.size() * sizeof(T)); }
    void buffer(const ConversationBuffer& b) {
        u64(b.size());
        buffer_raw(b);
    }
    void buffer_raw(const ConversationBuffer& b) {  // v3 segment streams: the data only, the size lives in the group framing
        b.visit(0, b.size(), [&](const uint8_t* p, size_t c, size_t) { raw(p, c); return ok; });
    }
    // a K/V part pulled from its owner straight into the file block (sizing pass: counted, never read)
    void source(const SessionKvSource& s, size_t part) {
        u64(s.sizes[part]);
        source_range(s, part, 0, s.sizes[part]);
    }
    // the bytes [from, to) of one part: v3 segments start at a rounded committed boundary
    void source_range(const SessionKvSource& s, size_t part, size_t from, size_t to) {
        if (!f) { count(to - from); return; }
        for (size_t at = from; at < to && ok;) {
            size_t room = 0;
            uint8_t* d = f->span(room);
            const size_t c = std::min(room, to - at);
            if (!s.read || !s.read(part, at, d, c)) {
                ok = false;
                read_failed = true;
                if (s.error) read_error = s.error();
                return;
            }
            if (hash) hash->update(d, c);
            count(c);
            ok = ok && f->commit(c);
            at += c;
        }
    }
    bool read_failed = false;
    std::string read_error;
};

void put_kv_header(Out& o, int format, int64_t cells, int64_t heads, int64_t head_dim, int64_t page_size,
                   int64_t pooled_rows, int64_t idx_dim) {
    for (int64_t v : {int64_t(format), cells, heads, head_dim, page_size, pooled_rows, idx_dim}) o.i64(v);
}
// v3 adds the engine's pooled-row block size (idx_block; 0 when the writer does not know it,
// which also makes that layer's pooled prefix append-refused later)
void put_kv_header_v3(Out& o, int format, int64_t cells, int64_t heads, int64_t head_dim, int64_t page_size,
                      int64_t pooled_rows, int64_t idx_dim, int64_t idx_block) {
    for (int64_t v : {int64_t(format), cells, heads, head_dim, page_size, pooled_rows, idx_dim, idx_block}) o.i64(v);
}

void put_checkpoint(Out& o, const ConversationCheckpoint& c) {
    o.vec(c.ids);
    o.u64(c.imgs.size());
    for (const auto& k : c.imgs) { o.i64(k.start); o.u64(k.hash); }
    o.vec(c.gdn); o.vec(c.ple); o.vec(c.tails); o.vec(c.dead); o.vec(c.block_pos);
    o.u64(c.used);
    o.u64(c.stage_parts.size());                                 // v2: one part per later stage, 0 for parts themselves
    for (const auto& p : c.stage_parts) put_checkpoint(o, p);
}

void put_payload(Out& o, const SavedConversation& s, const std::vector<SessionKvSource>* sources = nullptr) {
    for (int64_t g : s.geometry) o.i64(g);
    o.i64(s.layer_lo); o.i64(s.layer_hi); o.u64(s.cvec ? 1 : 0);
    // format v2: the later-stage geometry the file was written with (empty table: a single-device session)
    o.u64(s.stage_table.size());
    for (const auto& g : s.stage_table) {
        o.i64(g.layer_lo); o.i64(g.layer_hi); o.i64(g.qsa_alloc); o.i64(g.gdn_alloc); o.u64(g.draft ? 1 : 0);
    }
    put_checkpoint(o, s.live);
    o.u64(s.checkpoints.size());
    for (const auto& c : s.checkpoints) put_checkpoint(o, c);
    if (sources) {
        o.u64(sources->size());
        for (const auto& k : *sources) {
            put_kv_header(o, k.format, k.cells, k.heads, k.head_dim, k.page_size, k.pooled_rows, k.idx_dim);
            for (size_t part = 0; part < 5; ++part) o.source(k, part);
        }
        return;
    }
    o.u64(s.kv.size());
    for (const auto& k : s.kv) {
        put_kv_header(o, k.format, k.cells, k.heads, k.head_dim, k.page_size, k.pooled_rows, k.idx_dim);
        o.buffer(k.k); o.buffer(k.v); o.buffer(k.k_scale); o.buffer(k.v_scale); o.buffer(k.pooled);
    }
}

// ---- format v3: one generation group (see the header comment and the plan)
//
// Canonical group order, fixed for every generation and pinned by a golden test:
// preamble (generation 1 only: geometry, carve, cvec, stage table, K/V layer count - the
// identity digests in the header already pin them); mutable block (live record: counts and
// running-state arrays, with the live ids/imgs DATA as segment streams; full checkpoint
// records, re-emitted because the saved checkpoint's identity can change between saves; per-
// layer K/V headers and part sizes); segment block (live ids/imgs and per-stage-part mirrors,
// then per-layer K/V streams, each in this fixed order); zero pad so group+trailer is a
// multiple of 512 (the pad is part of the hashed zone); the 128 B trailer record.
void put_live_v3(Out& o, const ConversationCheckpoint& c) {  // counts + running state; ids/imgs go as segments
    o.u64(c.ids.size());
    o.u64(c.imgs.size());
    o.vec(c.gdn); o.vec(c.ple); o.vec(c.tails); o.vec(c.dead); o.vec(c.block_pos);
    o.u64(c.used);
    o.u64(c.stage_parts.size());
    for (const auto& p : c.stage_parts) put_live_v3(o, p);  // parts of parts are refused before any disk access
}
// Where each append-only stream of a generation group starts: a full write starts every stream at 0;
// an append starts ids/imgs (and each stage part's) at the previous generation's record counts.
struct V3LiveFrom {
    uint64_t ids = 0, imgs = 0;
    std::vector<V3LiveFrom> parts;   // one per stage part, in stage order (parts hold no parts)
};
void put_live_segments(Out& o, const ConversationCheckpoint& c, const V3LiveFrom* from = nullptr) {
    const uint64_t fi = from ? from->ids : 0, fg = from ? from->imgs : 0;
    if (fi > c.ids.size() || fg > c.imgs.size()) { o.ok = false; return; }   // refused long before here
    o.raw(c.ids.data() + fi, (c.ids.size() - fi) * sizeof(int32_t));
    o.raw(c.imgs.data() + fg, (c.imgs.size() - fg) * sizeof(ConversationImageKey));
    for (size_t k = 0; k < c.stage_parts.size(); ++k)
        put_live_segments(o, c.stage_parts[k], from && k < from->parts.size() ? &from->parts[k] : nullptr);
}
// The group's K/V part framing records the append window [from, to) per part: the segment zone
// then carries exactly the bytes [from, to) of that part, and `to` is the part's total size in
// this generation.  A full write (generation 1) uses from = 0.  An append starts at the rounded
// committed boundary, so its fresh copy of the residue region (the final partial page and the
// pooled spare row) SHADOWS the older group's stale bytes: the reader truncates the accumulated
// bytes to `from` before appending, never concatenating the residue twice.
void put_group_v3(Out& o, const SavedConversation& s, const std::vector<SessionKvSource>* sources = nullptr,
                  const std::vector<std::array<uint64_t, 5>>* froms = nullptr, const V3LiveFrom* live_from = nullptr,
                  bool with_preamble = true) {
    if (with_preamble) {   // an append group is framed by generation 1's preamble: only a full write re-emits it
        for (int64_t g : s.geometry) o.i64(g);
        o.i64(s.layer_lo); o.i64(s.layer_hi); o.u64(s.cvec ? 1 : 0);
        o.u64(s.stage_table.size());
        for (const auto& g : s.stage_table) {
            o.i64(g.layer_lo); o.i64(g.layer_hi); o.i64(g.qsa_alloc); o.i64(g.gdn_alloc); o.u64(g.draft ? 1 : 0);
        }
        o.u64(sources ? sources->size() : s.kv.size());
    }
    put_live_v3(o, s.live);
    o.u64(s.checkpoints.size());
    for (const auto& c : s.checkpoints) put_checkpoint(o, c);  // whole record: counts AND data (identity can switch)
    if (sources) {
        for (size_t j = 0; j < sources->size(); ++j) {
            const auto& k = (*sources)[j];
            put_kv_header_v3(o, k.format, k.cells, k.heads, k.head_dim, k.page_size, k.pooled_rows, k.idx_dim,
                             k.idx_block);
            for (size_t part = 0; part < 5; ++part) {
                o.u64(froms ? (*froms)[j][part] : 0);
                o.u64(k.sizes[part]);
            }
        }
    } else {
        for (const auto& k : s.kv) {
            // a captured image does not carry the engine's idx_block: record 0 (append refuses that layer);
            // a captured write is always full, so every window starts at 0
            put_kv_header_v3(o, k.format, k.cells, k.heads, k.head_dim, k.page_size, k.pooled_rows, k.idx_dim, 0);
            const std::array<uint64_t, 5> sz = {(uint64_t) k.k.size(), (uint64_t) k.v.size(), (uint64_t) k.k_scale.size(),
                                                (uint64_t) k.v_scale.size(), (uint64_t) k.pooled.size()};
            for (size_t part = 0; part < 5; ++part) { o.u64(0); o.u64(sz[part]); }
        }
    }
    put_live_segments(o, s.live, live_from);
    if (sources) {
        for (size_t j = 0; j < sources->size(); ++j) {
            const auto& k = (*sources)[j];
            for (size_t part = 0; part < 5; ++part)
                o.source_range(k, part, froms ? (size_t) (*froms)[j][part] : 0, k.sizes[part]);
        }
        return;
    }
    for (const auto& k : s.kv) {
        o.buffer_raw(k.k); o.buffer_raw(k.v); o.buffer_raw(k.k_scale); o.buffer_raw(k.v_scale); o.buffer_raw(k.pooled);
    }
}

// ---- parsing: every count is checked against the bytes left and the caller's limits before anything is allocated
struct In {
    FileSource* f;
    SessionHasher hash;
    uint64_t left;
    std::string& error;
    const SessionReadLimits& limits;
    SessionError kind = SessionError::invalid;
    size_t kv_layer = 0;
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
    // advance a framing walk past bytes it does not need: same bounds discipline as raw(), no read, no hash
    bool skip(size_t c) {
        if (c > left) return fail("payload ends early");
        if (c && !f->skip(c)) { kind = f->kind(); return fail("read error: " + f->error()); }
        left -= c;
        return true;
    }
    // `elem`: the fewest file bytes one element takes; `limit`: the most elements the caller accepts
    bool count(uint64_t& n, size_t elem, uint64_t limit = UINT64_MAX) {
        if (!u64(n)) return false;
        if (elem && n > left / elem) return fail("element count exceeds the payload");
        if (n > limit) return fail("element count " + std::to_string(n) + " exceeds this engine's limit " +
                                   std::to_string(limit));
        if (n > (uint64_t) std::numeric_limits<size_t>::max() / (elem ? elem : 1)) return fail("element count too large");
        return true;
    }
    template<class T> bool vec(std::vector<T>& v, uint64_t limit = UINT64_MAX) {
        uint64_t n = 0;
        if (!count(n, sizeof(T), limit)) return false;
        v.resize((size_t) n);
        return raw(v.data(), (size_t) n * sizeof(T));
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

bool get_checkpoint(In& in, ConversationCheckpoint& c, uint64_t expected_parts,
                    const std::array<uint64_t, 5>& state_max, bool framing = false) {
    static constexpr std::array<uint64_t, 5> kOpen{UINT64_MAX, UINT64_MAX, UINT64_MAX, UINT64_MAX, UINT64_MAX};
    if (framing) {  // the framing walk needs only the record's SIZE: every field counted, its bytes skipped
        uint64_t n = 0;  // (review finding M3: counts are capped by the runtime's limits before anything allocates)
        if (!in.count(n, 4, in.limits.max_tokens) || !in.skip((size_t) n * 4)) return false;  // ids
        if (!in.count(n, 16, in.limits.max_tokens) || !in.skip((size_t) n * 16)) return false;  // images
        for (size_t a = 0; a < 5; ++a)
            if (!in.count(n, 1, state_max[a]) || !in.skip((size_t) n)) return false;
        uint64_t used = 0;
        if (!in.u64(used)) return false;
    } else {
        if (!in.vec(c.ids, in.limits.max_tokens)) return false;
        uint64_t n = 0;
        if (!in.count(n, 16, in.limits.max_tokens)) return false;
        c.imgs.resize((size_t) n);
        for (auto& k : c.imgs) if (!in.i64(k.start) || !in.u64(k.hash)) return false;
        // byte arrays: element and byte counts are the same; main checkpoints use the runtime's arrays,
        // stage parts their own stage's (max_stage_state_bytes, by part index)
        if (!in.vec(c.gdn, state_max[0]) || !in.vec(c.ple, state_max[1]) || !in.vec(c.tails, state_max[2]) ||
            !in.vec(c.dead, state_max[3]) || !in.vec(c.block_pos, state_max[4]) || !in.u64(c.used))
            return false;
    }
    uint64_t parts = 0;
    if (!in.u64(parts)) return false;                            // v2: the stage-parts tail
    if (parts != expected_parts)
        return expected_parts == 0 ? in.fail("nested stage parts")
                                   : in.fail("stage parts count does not match the stage table");
    for (uint64_t k = 0; k < parts; ++k) {
        const auto& smax = k < in.limits.max_stage_state_bytes.size()
                               ? in.limits.max_stage_state_bytes[(size_t) k]
                               : kOpen;
        ConversationCheckpoint part;
        if (!get_checkpoint(in, part, 0, smax, framing)) return false;  // one level deep, by format
        if (!framing) c.stage_parts.push_back(std::move(part));
    }
    return true;
}

bool get_payload(In& in, SavedConversation& s) {
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
    // format v2 stage table: bounded by count and bytes left, then compared against the live layout
    // before any state array is read
    uint64_t stages = 0;
    if (!in.count(stages, 40, in.limits.max_stages)) return false;
    s.stage_table.resize((size_t) stages);
    for (auto& g : s.stage_table) {
        uint64_t draft = 0;
        if (!in.i64(g.layer_lo) || !in.i64(g.layer_hi) || !in.i64(g.qsa_alloc) || !in.i64(g.gdn_alloc) ||
            !in.u64(draft))
            return false;
        if (draft > 1) return in.fail("invalid stage draft flag");
        g.draft = draft == 1;
    }
    if (in.limits.stages && *in.limits.stages != s.stage_table)
        return in.fail("saved with a different layer split than this runtime's");
    if (!get_checkpoint(in, s.live, s.stage_table.size(), in.limits.max_state_bytes)) return false;
    uint64_t n = 0;
    // a checkpoint is at least 8 counts + used = 72 bytes; a K/V layer at least 7 + 5 = 96
    if (!in.count(n, 72, in.limits.max_checkpoints)) return false;
    s.checkpoints.resize((size_t) n);
    for (auto& c : s.checkpoints)
        if (!get_checkpoint(in, c, s.stage_table.size(), in.limits.max_state_bytes)) return false;
    if (!in.count(n, 96, in.limits.max_kv_layers)) return false;
    s.kv.resize((size_t) n);
    for (auto& k : s.kv) {
        in.kv_layer = (size_t) (&k - s.kv.data());
        int64_t format = 0;
        if (!in.i64(format) || !in.i64(k.cells) || !in.i64(k.heads) || !in.i64(k.head_dim) ||
            !in.i64(k.page_size) || !in.i64(k.pooled_rows) || !in.i64(k.idx_dim)) return false;
        if (format < INT32_MIN || format > INT32_MAX) return in.fail("invalid K/V format");
        k.format = (int) format;
        if (!in.buffer(k.k, 0) || !in.buffer(k.v, 1) || !in.buffer(k.k_scale, 2) || !in.buffer(k.v_scale, 3) ||
            !in.buffer(k.pooled, 4)) return false;
    }
    return true;
}

// v2 stage consistency (checked before any disk access): every checkpoint carries exactly one stage part
// per stage-table entry, and a part has no parts of its own. `why` names the first violation.
bool stage_parts_consistent(const SavedConversation& s, std::string& why) {
    auto one = [&](const ConversationCheckpoint& c, const char* what) -> bool {
        if (c.stage_parts.size() != s.stage_table.size()) {
            why = std::string("a ") + what + " checkpoint whose stage parts count does not match the stage table";
            return false;
        }
        for (const auto& p : c.stage_parts)
            if (!p.stage_parts.empty()) { why = "nested stage parts"; return false; }
        return true;
    };
    if (!one(s.live, "live")) return false;
    for (const auto& c : s.checkpoints)
        if (!one(c, "saved")) return false;
    return true;
}

// header v2, little-endian: magic[8] version:u32 header_size:u32 model:u64 config:u64 payload:u64 reserved:u64[2]
// header_hash:u64 (session_hash64 of bytes 0..55, seed 0)
void header_bytes(uint8_t* h, const SessionFileIdentity& id, uint64_t payload) {
    std::memset(h, 0, kHeader);
    std::memcpy(h, kMagic, 8);
    const uint32_t version = kVersion, size = kHeader;
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

SessionHasherState SessionHasher::state() const {
    SessionHasherState s{};  // value-init: the padding past npending is zeroed too, so two
                             // states for equal inputs compare byte-exactly whole
    std::memcpy(s.lanes, lane_, sizeof lane_);
    std::memcpy(s.pending, pending_, npending_);
    std::memset(s.pending + npending_, 0, sizeof s.pending - npending_);
    s.npending = (uint8_t) npending_;
    s.total = total_;
    return s;
}
void SessionHasher::restore(const SessionHasherState& s) {
    std::memcpy(lane_, s.lanes, sizeof lane_);
    std::memcpy(pending_, s.pending, sizeof pending_);
    npending_ = s.npending;  // callers guarantee 0..31 (trailer_parse enforces it on disk reads)
    total_ = s.total;
}

// v3 trailer record helpers (layout in the header comment).
constexpr uint32_t kTrailerMagic = 0x54525453u;  // little-endian bytes 'S','T','R','T'
constexpr size_t kRecordV3 = 128;  // the v3 checkpoint RECORD size (the kTrailer above is v2's 16 B end marker)

bool trailer_parse(const uint8_t* b, SessionTrailer& out) {
    SessionTrailer t;
    uint32_t magic;
    std::memcpy(&magic, b, 4);
    if (magic != kTrailerMagic) return false;
    if (b[4] != 3) return false;
    t.version = 3;
    t.npending = b[5];
    if (t.npending > 31) return false;
    std::memcpy(&t.reserved, b + 6, 2);
    if (t.reserved != 0) return false;
    uint64_t v[5];
    for (int i = 0; i < 5; ++i) std::memcpy(&v[i], b + 8 + 8 * i, 8);
    t.gen = v[0];
    t.committed_bytes = v[1];
    t.saved_tokens = v[2];
    t.digest = v[3];
    t.chain = v[4];
    std::memcpy(t.lanes, b + 48, 32);
    std::memcpy(t.pending, b + 80, 32);
    std::memcpy(&t.hashed_bytes, b + 112, 8);
    uint64_t self;
    std::memcpy(&self, b + 120, 8);
    if (self != session_hash64(b, 120, 0)) return false;  // self_hash re-derivation: covers [0,120),
    t.self_hash = self;   // and the tail [124,128) holds the value's own high half - corruption of ANY
    out = t;              // byte in the record breaks this equality (the drill's F3-class pad case
    return true;          // cannot arise: there is no byte in use that the hash does not pin)
}

void trailer_bytes(const SessionTrailer& t, uint8_t (&b)[128]) {
    std::memset(b, 0, 128);
    std::memcpy(b, &kTrailerMagic, 4);
    b[4] = t.version;
    b[5] = t.npending;
    std::memcpy(b + 6, &t.reserved, 2);
    const uint64_t v[5] = {t.gen, t.committed_bytes, t.saved_tokens, t.digest, t.chain};
    for (int i = 0; i < 5; ++i) std::memcpy(b + 8 + 8 * i, &v[i], 8);
    std::memcpy(b + 48, t.lanes, 32);
    std::memcpy(b + 80, t.pending, 32);
    std::memcpy(b + 112, &t.hashed_bytes, 8);
    const uint64_t self = session_hash64(b, 120, 0);  // self_hash covers [0,120): every field, hashed
    std::memcpy(b + 120, &self, 8);                   // directly off the record.  The full 8-byte value
    // lands at [120,128): its high half at [124,128) is the record's tail - pinned by construction and
    // by the parse-side re-hash equality (the drill's F3-class torn-pad case demotes at parse, never
    // trusts: corrupting the tail breaks the equality).
}

uint64_t trailer_genesis_chain(const SessionFileIdentity& id) {
    uint8_t buf[16];
    std::memcpy(buf, &id.model, 8);
    std::memcpy(buf + 8, &id.config, 8);
    return session_hash64(buf, 16, 0);
}

constexpr uint32_t kVersion3 = 3;

void header_bytes_v3(uint8_t (&h)[64], const SessionFileIdentity& id, const SessionHeaderV3& body) {
    std::memset(h, 0, 64);
    std::memcpy(h, kMagic, 8);
    const uint32_t version = kVersion3, size = (uint32_t) kHeader;
    std::memcpy(h + 8, &version, 4);
    std::memcpy(h + 12, &size, 4);
    std::memcpy(h + 16, &id.model, 8);
    std::memcpy(h + 24, &id.config, 8);
    const uint64_t root = session_hash64(h, 32, 0);  // covers the immutable region only
    std::memcpy(h + 32, &root, 8);
    std::memcpy(h + 40, &body.payload, 8);
    std::memcpy(h + 48, &body.reserved0, 8);
    std::memcpy(h + 56, &body.reserved1, 8);
}

bool header_parse_v3(const uint8_t* h, const SessionFileIdentity& expect, SessionHeaderV3& body, std::string& error) {
    if (std::memcmp(h, kMagic, 8) != 0) { error = "session file: not a session file"; return false; }
    uint32_t version, size;
    std::memcpy(&version, h + 8, 4);
    std::memcpy(&size, h + 12, 4);
    if (version != kVersion3 || size != (uint32_t) kHeader) { error = "session file: not a v3 header"; return false; }
    uint64_t model, config, root;
    std::memcpy(&model, h + 16, 8);
    std::memcpy(&config, h + 24, 8);
    std::memcpy(&root, h + 32, 8);
    if (root != session_hash64(h, 32, 0)) { error = "session file: header checksum mismatch"; return false; }
    if (model != expect.model || config != expect.config) { error = "session file: another model or configuration"; return false; }
    std::memcpy(&body.payload, h + 40, 8);
    std::memcpy(&body.reserved0, h + 48, 8);
    std::memcpy(&body.reserved1, h + 56, 8);
    return true;
}

bool session_file_checkpoints(const std::string& path, const SessionFileIdentity& id,
                              std::vector<SessionCheckpoint>& newest_first, std::string& error) {
    newest_first.clear();
    RawFile f;
    uint64_t size = 0;
    if (!f.open_session(path, size)) { error = f.error(); return false; }
    if (size < kHeader + 128) { error = "session file: v3 file has no room for a trailer"; return false; }
    f.force_buffered();  // small random reads: the direct-IO mode the writer path uses would
                         // only add alignment constraints the probes do not need
    uint8_t h[kHeader];
    if (!f.read_at(0, h, kHeader)) { error = f.error(); return false; }
    SessionHeaderV3 hv;
    if (!header_parse_v3(h, id, hv, error)) return false;
    const uint64_t genesis = trailer_genesis_chain(id);

    struct Cand {
        SessionTrailer t;
        uint64_t off;
        bool trusted;
    };
    std::vector<Cand> cands;
    auto add = [&](uint64_t off, const SessionTrailer& t) {
        for (const auto& c : cands)
            if (c.off == off) return;  // the same record found by fast path, anchor and scan
        cands.push_back({t, off, false});
    };
    // A record is structurally sane when its committed_bytes equals its own offset.
    auto sane = [&](const SessionTrailer& t, uint64_t off) { return t.committed_bytes == off; };
    auto probe = [&](uint64_t off) {
        if (off < kHeader || off > size - 128) return;
        uint8_t b[128];
        if (!f.read_at(off, b, 128)) return;
        SessionTrailer t;
        if (!trailer_parse(b, t) || !sane(t, off)) return;
        add(off, t);
    };
    probe(hv.reserved0);                                             // the fast path
    if (hv.reserved1 >= kHeader + 128) probe(hv.reserved1 - 128);   // the append anchor

    // Byte-wise scan-back through the bounded torn window, in sequential chunks (each read
    // runs 127 B past its block so a record spanning a chunk border is seen).
    const uint64_t window = std::min<uint64_t>(size - kHeader, 256ull << 20);
    const uint64_t low = size - 128 > window ? size - 128 - window : kHeader;
    constexpr uint64_t kChunk = 8ull << 20;
    std::vector<uint8_t> buf;
    for (uint64_t start = low; start <= size - 128; start += kChunk) {
        const uint64_t len = std::min(kChunk + 127, size - start);
        buf.resize((size_t) len);
        if (!f.read_at(start, buf.data(), (size_t) len)) break;
        for (uint64_t i = 0; i + 128 <= len && start + i <= size - 128; ++i) {
            uint32_t magic;
            std::memcpy(&magic, buf.data() + i, 4);
            if (magic != kTrailerMagic) continue;  // the 4-byte filter runs at every offset
            SessionTrailer t;
            if (!trailer_parse(buf.data() + i, t) || !sane(t, start + i)) continue;
            add(start + i, t);
        }
    }

    // Trust, oldest first (off == committed_bytes, so offset order is generation order).
    std::sort(cands.begin(), cands.end(), [](const Cand& a, const Cand& b) { return a.off < b.off; });
    bool any = false;
    for (auto& c : cands) {
        if (c.t.gen == 1 && c.t.chain == genesis) { c.trusted = any = true; continue; }
        for (const auto& p : cands) {
            if (p.off >= c.off || !p.trusted || p.t.gen != c.t.gen - 1) continue;
            if (c.off < p.off + kRecordV3 + 1) continue;  // a trusted pairing needs a generation GROUP between
            // the records: two trailer records back-to-back (a planted phantom - drill F2a geometry) is not
            // a chain of generations, whatever the chain hash says
            if (p.t.self_hash == c.t.chain) { c.trusted = any = true; break; }
        }
        if (!c.trusted && c.off == hv.reserved0 && hv.reserved1 >= kHeader + 128 && hv.reserved1 - 128 == c.off) {
            c.trusted = any = true;  // a fresh header pair wrote both slots in the engine's own commit
        }
    }
    if (!any) { error = "session file: no valid checkpoint"; return false; }
    for (auto it = cands.rbegin(); it != cands.rend(); ++it)
        if (it->trusted) newest_first.push_back({it->t, it->off, size});
    return true;
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

namespace {
uint64_t sat_add(uint64_t a, uint64_t b) { return a > UINT64_MAX - b ? UINT64_MAX : a + b; }
uint64_t sat_mul(uint64_t a, uint64_t b) { return a && b > UINT64_MAX / a ? UINT64_MAX : a * b; }
} // namespace

uint64_t session_read_max_file_bytes(const SessionReadLimits& l) {
    const uint64_t open = UINT64_MAX;
    if (l.max_tokens == open || l.max_checkpoints == open || l.max_kv_layers == open || l.max_stages == open)
        return l.max_file_bytes;
    for (uint64_t b : l.max_state_bytes) if (b == open) return l.max_file_bytes;
    if (l.max_kv_bytes.size() < l.max_kv_layers) return l.max_file_bytes;
    // One checkpoint record over a carve's state sizes `sb`: ids (8 + 4 per token), images (8 + 16 per image),
    // five state arrays (8 + bytes), and the two u64 fields v2 adds to every record (used, stage-parts count).
    const auto cp_over = [&](const std::array<uint64_t, 5>& sb) {
        uint64_t cp = sat_add(16, sat_mul(l.max_tokens, 4 + 16));
        for (uint64_t b : sb) cp = sat_add(cp, sat_add(8, b));
        return sat_add(cp, 16);
    };
    if (l.max_stage_state_bytes.size() < l.max_stages) return l.max_file_bytes;
    // v2: every checkpoint carries the main record plus one part per admitted stage, each part budgeted with
    // the stage's OWN state bounds (later carves differ from the main one).
    uint64_t cp = cp_over(l.max_state_bytes);
    for (size_t k = 0; k < l.max_stages && k < l.max_stage_state_bytes.size(); ++k) cp = sat_add(cp, cp_over(l.max_stage_state_bytes[k]));
    uint64_t n = kHeader + kTrailer + 18 * 8 + 3 * 8;             // geometry, layer range, cvec
    n = sat_add(n, sat_add(8, sat_mul(l.max_stages, 40)));        // stage table (v2): count + 40 bytes per entry
    n = sat_add(n, cp);                                            // live
    n = sat_add(n, sat_add(8, sat_mul(l.max_checkpoints, cp)));    // checkpoints
    n = sat_add(n, 8);
    for (uint64_t i = 0; i < l.max_kv_layers; ++i) {
        n = sat_add(n, 7 * 8);
        for (uint64_t b : l.max_kv_bytes[(size_t) i]) n = sat_add(n, sat_add(8, b));
    }
    return std::min(n, l.max_file_bytes);
}

uint64_t session_read_peak_bytes(uint64_t file_bytes) {
    const uint64_t segments = file_bytes / kBlock + 1;
    // the parsed image (about the payload), the read buffer, a page per buffer segment, the small vectors
    return sat_add(sat_add(sat_add(file_bytes, kBlock), sat_mul(segments, 4096)), uint64_t(1) << 20);
}

namespace {
bool write_impl(const std::string& path, const SavedConversation& image, const std::vector<SessionKvSource>* sources,
                const SessionFileIdentity& id, size_t& bytes, std::string& error, const SessionWriteOptions& opt,
                SessionStatus& st) {
    st = {};
    st.error = SessionError::invalid;   // refusals before any disk access
    std::string bad;
    if (!stage_parts_consistent(image, bad)) { error = "session file: " + bad; return false; }
    if (sources && !image.kv.empty()) { error = "session file: K/V given both in the image and as sources"; return false; }
    Out sizing;
    if (opt.use_v3) put_group_v3(sizing, image, sources);
    else put_payload(sizing, image, sources);
    const uint64_t group = sizing.n;
    const uint64_t pad = opt.use_v3 ? (512 - (group + 128) % 512) % 512 : 0;
    const uint64_t committed = kHeader + group + pad;          // v3: the trailer record's offset
    const uint64_t payload = group, total = opt.use_v3 ? committed + 128 : kHeader + group + kTrailer;
    if (sizing.overflow || group > UINT64_MAX - kHeader - 512 - 128 ||
        total > (uint64_t) std::numeric_limits<size_t>::max()) {
        error = "session file: the state is too large for this build";
        return false;
    }
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
    FileSink f(raw);
    if (!f.ok()) { st.error = SessionError::memory; error = "session file: " + f.error(); return false; }
    if (opt.progress) f.progress.emplace(opt.progress);
    if (opt.block_span) f.block_span.emplace(opt.block_span);
    f.phase = opt.phase;
    f.expected = total;
    if (f.progress) (*f.progress)(0, total);   // the temporary file exists: the transfer starts
    const std::function<int(const char*)>* fault = opt.fault ? &opt.fault : nullptr;
    f.write_fault = injected(fault, "write");
    f.flush_fault = injected(fault, "file_flush");
    f.writer_thread_fault = injected(fault, "writer_thread");
    uint8_t h[kHeader];
    if (opt.use_v3) header_bytes_v3(h, id, {committed, committed, committed + 128});
    else header_bytes(h, id, payload);
    bool ok = f.write(h, kHeader);
    SessionHasher hash(0);
    Out out;
    out.f = &f;
    out.hash = &hash;
    if (ok) {
        if (opt.use_v3) put_group_v3(out, image, sources);
        else put_payload(out, image, sources);
    }
    ok = ok && out.ok && out.n == payload;
    if (ok && opt.use_v3) {
        static const uint8_t kZeros[512] = {};
        out.raw(kZeros, pad);  // the pad is part of the hashed zone (the digest covers each contiguous group region)
        SessionTrailer tr;
        tr.gen = 1;
        tr.committed_bytes = committed;
        tr.saved_tokens = (uint64_t) image.live.ids.size();  // the tokens this save carries
        tr.digest = hash.digest();
        tr.chain = trailer_genesis_chain(id);
        const SessionHasherState hs = hash.state();  // resume state: the next append continues this digest
        for (int i = 0; i < 4; ++i) tr.lanes[i] = hs.lanes[i];
        std::memcpy(tr.pending, hs.pending, 32);
        tr.npending = hs.npending;
        tr.hashed_bytes = hs.total;
        uint8_t tb[128];
        trailer_bytes(tr, tb);
        ok = out.ok && f.write(tb, 128);  // the trailer itself is not in the payload zone
    }
    if (ok && !opt.use_v3) {
        const uint64_t ph = hash.digest();
        ok = f.write(&ph, 8) && f.write(kEnd, 8);
    }
    ok = f.finish(opt.durable) && ok;
    std::string why;
    if (!ok) {
        why = out.read_failed ? "K/V source read" + (out.read_error.empty() ? std::string() : ": " + out.read_error)
                              : f.error();
        st.error = out.read_failed ? SessionError::io : f.kind();
    } else {
        if (opt.phase) opt.phase("publish", total);
        if (pub.publish(why, opt.durable, st, fault)) {
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
    uint64_t n = sat_add(sat_mul(2, kBlock), uint64_t(1) << 20);              // the two pipelined write buffers, small vectors
    if (deepest) n = sat_add(n, sat_mul(2, deepest->bytes()));                // the list and the image's copy
    n = sat_add(n, live.state_bytes);                                         // the live running state
    n = sat_add(n, sat_mul(live.tokens, sizeof(int32_t)));                    // its token list
    n = sat_add(n, sat_mul(live.images, sizeof(ConversationImageKey)));       // its images
    n = sat_add(n, sat_mul(live.kv_layers, sizeof(SessionKvSource) + 256));   // the K/V source directory
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
// ---- format v3 reading: restore is the authoritative verifier ------------------------------
//
// The validator (session_file_checkpoints) locates chain-and-structure-valid trailer candidates;
// the reader then walks the generation chain forward, re-deriving every group's boundary from the
// group's OWN framing (all region lengths are recorded), re-hashing every generation's zone bytes
// in physical order, and at the newest candidate compares the digest, the stored resume state, the
// generation number, the structural offset and saved_tokens against what it actually read. A
// candidate that fails verification demotes to the next older one and the walk restarts at its
// limit (the cascade).  Intermediate generations' trailer records are cross-checked but never
// disqualifying on their own: the candidate's digest covers every zone byte below its own record,
// so a damaged middle record costs nothing here (the save-side gate is the strict one, where a
// declined append is just a full rewrite).  The walk is sequential: one pass over the chain, and
// the FileSource ticker reports as bytes really move.
struct V3Walk {
    In& in;
    SavedConversation& s;
    uint64_t covered;   // this candidate's file coverage: trailer offset + 128
    uint64_t genesis;

    // Append-only streams accumulate across generations; every re-emitted (mutable) field takes
    // the newest group's copy.
    struct Acc {
        std::vector<int32_t> ids;
        std::vector<ConversationImageKey> imgs;
        uint64_t want_ids = 0, want_imgs = 0;   // record counts declared by the current group
        uint64_t have_ids = 0, have_imgs = 0;   // framing mode: records the chain has already accumulated
    };
    Acc live;
    std::vector<Acc> parts;   // one per stage-table entry (a part never has parts of its own)
    std::vector<std::array<ConversationBuffer, 5>> kv;   // per-layer K/V parts, grown by recorded suffixes
    struct Geom {
        int64_t format = 0, cells = 0, heads = 0, head_dim = 0, page_size = 0, pooled_rows = 0, idx_dim = 0,
                idx_block = 0;
    };
    std::vector<Geom> geom;
    std::vector<bool> seen;
    size_t layers = 0;
    bool framing_only = false;  // the append-plan walk: frame and cross-check, seek over segment bytes, never hash
                                // (the gate verifies the trailer's own resume state instead)
    std::vector<std::array<uint64_t, 5>> have_bytes;  // bytes accumulated per layer part (framing mode needs them;
                                                      // read mode tracks the same count through its kv buffers)
    std::vector<std::array<uint64_t, 5>> last_to;     // newest group: each part's recorded total size
    bool have_prev_saved = false;  // the previous generation's trailer parsed intact (its saved_tokens usable)
    uint64_t prev_saved = 0;

    // The engine's append rule (spike S1, spec Anchor), recomputed from the previous trailer's
    // saved_tokens and the previous generation's geometry: the committed byte boundary of one
    // part - whole_cells with the draft layer's -1, whole pooled rows when idx_block is known.
    // UINT64_MAX = not recomputable (idx_block unknown, or non-uniform bytes-per-cell): the
    // recorded window governs and the digest stays the authority.
    uint64_t committed_boundary(size_t j, size_t p, uint64_t have) const {
        const Geom& prev = geom[j];
        // The draft layer, identified STRUCTURALLY: the last layer, sourced un-indexed.  A real save
        // records idx_dim from the geometry (never zero - conversation_kv_source), so the discriminator
        // is the sourced shape: no pooled part AND no indexer block (conversation_snapshot.cpp:532).
        const bool draft = j + 1 == layers && prev.pooled_rows == 0 && prev.idx_block == 0;
        const uint64_t t = draft && prev_saved > 0 ? prev_saved - 1 : prev_saved;
        if (p < 4) {
            if (prev.cells == 0) return 0;
            if (have % (uint64_t) prev.cells) return UINT64_MAX;  // not cell-proportional: cannot recompute
            const uint64_t w = prev.page_size > 0 ? (t / (uint64_t) prev.page_size) * (uint64_t) prev.page_size : t;
            return (have / (uint64_t) prev.cells) * w;
        }
        if (prev.pooled_rows > 0) {
            if (prev.idx_block == 0) return UINT64_MAX;  // pooled rows recorded without an idx_block: not append-trusted (Ruling B)
            if (have % (uint64_t) prev.pooled_rows) return UINT64_MAX;
            const uint64_t rows = t / (uint64_t) prev.idx_block;  // whole completed pooled rows below the boundary
            return (have / (uint64_t) prev.pooled_rows) * rows;
        }
        return have == 0 ? 0 : UINT64_MAX;  // a generation with no pooled part: nothing committed there; bytes with no row geometry cannot bound an append
    }

    bool preamble() {
        for (auto& g : s.geometry)
            if (!in.i64(g)) return false;
        if (in.limits.geometry && *in.limits.geometry != s.geometry)
            return in.fail("saved with another model geometry than this runtime's");
        uint64_t cvec = 0;
        if (!in.i64(s.layer_lo) || !in.i64(s.layer_hi) || !in.u64(cvec)) return false;
        if (in.limits.layer_range &&
            (in.limits.layer_range->first != s.layer_lo || in.limits.layer_range->second != s.layer_hi))
            return in.fail("saved with another layer range than this runtime's");
        if (cvec > 1) return in.fail("invalid cvec flag");
        s.cvec = cvec == 1;
        uint64_t stages = 0;
        if (!in.count(stages, 40, in.limits.max_stages)) return false;
        s.stage_table.resize((size_t) stages);
        for (auto& g : s.stage_table) {
            uint64_t draft = 0;
            if (!in.i64(g.layer_lo) || !in.i64(g.layer_hi) || !in.i64(g.qsa_alloc) || !in.i64(g.gdn_alloc) ||
                !in.u64(draft))
                return false;
            if (draft > 1) return in.fail("invalid stage draft flag");
            g.draft = draft == 1;
        }
        if (in.limits.stages && *in.limits.stages != s.stage_table)
            return in.fail("saved with a different layer split than this runtime's");
        if (!in.count(layers, 144, in.limits.max_kv_layers)) return false;  // one generation-layer is >= 8 i64 + 5 (from,to) u64 pairs
        if (!framing_only) kv.assign(layers, {});
        have_bytes.assign(layers, {});
        last_to.assign(layers, {});
        geom.assign(layers, {});
        seen.assign(layers, false);
        parts.resize(s.stage_table.size());
        return true;
    }

    // The live record's mutable half (mirrors put_live_v3): declared counts - the append-only
    // streams may only grow - and the freshly re-emitted running-state arrays; the ids/imgs DATA
    // is read in the segment phase.
    bool live_mutable(ConversationCheckpoint& c, Acc& a, uint64_t expected_parts,
                      const std::array<uint64_t, 5>& state_max, bool is_part) {
        if (!in.count(a.want_ids, 4, in.limits.max_tokens)) return false;
        if (!in.count(a.want_imgs, 16, in.limits.max_tokens)) return false;
        if (a.want_ids < a.ids.size() || a.want_imgs < a.imgs.size())
            return in.fail("a generation that shrank the live token list (an append chain only grows it)");
        if (framing_only) {  // the framing walk needs the arrays' SIZES only (review finding M3): count, skip
            uint64_t an = 0;
            for (size_t arr = 0; arr < 5; ++arr)
                if (!in.count(an, 1, state_max[arr]) || !in.skip((size_t) an)) return false;
            uint64_t used = 0;
            if (!in.u64(used)) return false;
        } else if (!in.vec(c.gdn, state_max[0]) || !in.vec(c.ple, state_max[1]) || !in.vec(c.tails, state_max[2]) ||
                   !in.vec(c.dead, state_max[3]) || !in.vec(c.block_pos, state_max[4]) || !in.u64(c.used))
            return false;
        uint64_t nparts = 0;
        if (!in.u64(nparts) || nparts != expected_parts)
            return in.fail("a live checkpoint whose stage parts count does not match the stage table");
        for (uint64_t k = 0; k < nparts; ++k) {
            static constexpr std::array<uint64_t, 5> kOpen{UINT64_MAX, UINT64_MAX, UINT64_MAX, UINT64_MAX,
                                                           UINT64_MAX};
            const auto& kmax = k < in.limits.max_stage_state_bytes.size()
                                   ? in.limits.max_stage_state_bytes[(size_t) k]
                                   : kOpen;
            ConversationCheckpoint part;
            if (!live_mutable(part, parts[(size_t) k], 0, kmax, true)) return false;  // parts hold no parts
            c.stage_parts.push_back(std::move(part));
        }
        return true;
    }

    bool live_segments(Acc& a) {  // the records appended since the previous generation
        if (framing_only) {  // declared counts are totals; the DATA in this group is only the suffix since
                             // the accumulated have - and the same growth rule the reader enforces
            if (a.want_ids < a.have_ids || a.want_imgs < a.have_imgs)
                return in.fail("a generation that shrank the live token list (an append chain only grows it)");
            if (!in.skip((size_t) (a.want_ids - a.have_ids) * sizeof(int32_t) +
                          (size_t) (a.want_imgs - a.have_imgs) * sizeof(ConversationImageKey)))
                return false;
            a.have_ids = a.want_ids;
            a.have_imgs = a.want_imgs;
            return true;
        }
        const size_t have = a.ids.size();
        a.ids.resize((size_t) a.want_ids);
        if (!in.raw(a.ids.data() + have, (size_t)(a.want_ids - have) * sizeof(int32_t))) return false;
        const size_t have_i = a.imgs.size();
        a.imgs.resize((size_t) a.want_imgs);
        if (!in.raw(a.imgs.data() + have_i, (size_t)(a.want_imgs - have_i) * sizeof(ConversationImageKey)))
            return false;
        return true;
    }

    bool group_body() {  // mutable block then segment block, in the canonical group order
        // Each generation re-emits the live record's stage parts: the list must describe THIS group,
        // never a mixture of the chain's generations (the same fresh-per-group discipline as
        // s.checkpoints below; ids/imgs arrive separately through the segment accumulators).
        s.live.stage_parts.clear();
        if (!live_mutable(s.live, live, (uint64_t) s.stage_table.size(), in.limits.max_state_bytes, false))
            return false;
        uint64_t n = 0;
        if (!in.count(n, 72, in.limits.max_checkpoints)) return false;  // a checkpoint is >= 72 bytes
        if (!framing_only) {  // the framing walk frames every record and discards it - no vector of records
                              // materializes on the gate (review finding M3's allocation path)
            s.checkpoints.clear();
            s.checkpoints.resize((size_t) n);
        }
        for (uint64_t ci = 0; ci < n; ++ci) {
            ConversationCheckpoint scratch;
            if (!get_checkpoint(in, framing_only ? scratch : s.checkpoints[(size_t) ci], s.stage_table.size(),
                                in.limits.max_state_bytes, framing_only))
                return false;
        }
        std::vector<std::array<uint64_t, 5>> wfrom(layers), wto(layers);
        for (size_t j = 0; j < layers; ++j) {
            Geom g;
            int64_t format = 0;
            if (!in.i64(format) || !in.i64(g.cells) || !in.i64(g.heads) || !in.i64(g.head_dim) ||
                !in.i64(g.page_size) || !in.i64(g.pooled_rows) || !in.i64(g.idx_dim) || !in.i64(g.idx_block))
                return false;
            if (format < INT32_MIN || format > INT32_MAX) return in.fail("invalid K/V format");
            if (g.cells < 0 || g.heads < 0 || g.head_dim < 0 || g.page_size < 0 || g.pooled_rows < 0 ||
                g.idx_dim < 0 || g.idx_block < 0)
                return in.fail("negative K/V geometry");
            g.format = format;
            if (seen[j]) {  // a chain append may only grow the carve; anything else needs a full rewrite
                if (g.format != geom[j].format || g.heads != geom[j].heads || g.head_dim != geom[j].head_dim ||
                    g.page_size != geom[j].page_size || g.idx_dim != geom[j].idx_dim ||
                    g.idx_block != geom[j].idx_block)
                    return in.fail("a generation whose K/V geometry differs from the chain's");
                if (g.cells < geom[j].cells || g.pooled_rows < geom[j].pooled_rows)
                    return in.fail("a generation that shrank the K/V carve (an append chain only grows it)");
            }
            for (size_t p = 0; p < 5; ++p) {
                const uint64_t got = have_bytes[j][p];  // bytes the chain has accumulated before this group
                if (!in.u64(wfrom[j][p]) || !in.u64(wto[j][p])) return false;
                if (wto[j][p] < wfrom[j][p]) return in.fail("an invalid K/V append window (to < from)");
                if (wto[j][p] < got) return in.fail("a generation that shrank a K/V part (an append chain only grows it)");
                if (wfrom[j][p] > got)
                    return in.fail("a generation appends past the bytes its predecessor committed");
                if (!seen[j]) {  // generation 1 is a full write
                    if (wfrom[j][p] != 0) return in.fail("the first generation does not append from 0");
                } else if (have_prev_saved) {  // Ruling C: the recorded window must start exactly at the committed boundary
                    const uint64_t b = committed_boundary(j, p, got);
                    if (b != UINT64_MAX && b != wfrom[j][p])
                        return in.fail("a generation does not append at its committed boundary");
                }
                if (j < in.limits.max_kv_bytes.size() && wto[j][p] > in.limits.max_kv_bytes[j][p])
                    return in.fail("a K/V part exceeds this engine's limit");
            }
            geom[j] = g;
            seen[j] = true;
        }
        if (!live_segments(live)) return false;
        for (auto& pa : parts)
            if (!live_segments(pa)) return false;
        for (size_t j = 0; j < layers; ++j)
            for (size_t p = 0; p < 5; ++p) {
                const uint64_t from = wfrom[j][p], to = wto[j][p];
                last_to[j][p] = to;
                have_bytes[j][p] = to;
                if (framing_only) {  // the gate walk frames the file; the bytes themselves are never needed
                    if (to > from && !in.skip((size_t) (to - from))) return false;
                    continue;
                }
                if (from < kv[j][p].size()) kv[j][p].resize_for_overwrite((size_t) from);  // shadow the stale copy
                const size_t cur = kv[j][p].size();  // == from now
                if (to == from) continue;
                kv[j][p].resize_for_overwrite(cur + (size_t) (to - from));  // grows without touching the old bytes
                if (!kv[j][p].visit(cur, (size_t) (to - from),
                                    [&](uint8_t* ptr, size_t c, size_t) { return in.raw(ptr, c); }))
                    return false;  // in.raw recorded the reason
            }
        return true;
    }

    // The group's zero pad (part of the hashed zone, pinned to zero) and its trailer record
    // (read but never hashed).  For the candidate: full verification - the reader's own digest,
    // resume state, generation, structural offset and saved_tokens must all agree.  For an
    // intermediate generation: cross-check when the record is intact, tolerate it when damaged.
    bool close_group(uint64_t gen, uint64_t frame, bool is_candidate, SessionTrailer& out, uint64_t& prev_self,
                     bool& prev_ok) {
        const uint64_t pad = (512 - (frame + 128) % 512) % 512;
        if (pad && framing_only) {  // the gate walk only frames: the pad's bytes are the digest's problem, not its own
            if (!in.skip(pad)) return false;
        } else if (pad) {
            static const uint8_t kZero[512] = {};
            uint8_t pb[512];
            if (pad > in.left || !in.raw(pb, pad)) return in.left < pad ? in.fail("the group pad is truncated") : false;
            if (std::memcmp(pb, kZero, pad) != 0) return in.fail("the group pad is not zero");
        }
        if (in.left < 128) return in.fail("the chain ends before a complete checkpoint record");
        const uint64_t trailer_pos = covered - in.left;  // the framing walk put the record exactly here
        uint8_t tb[128];
        if (!in.f->read(tb, 128)) {
            in.kind = in.f->kind();
            return in.fail("read error: " + in.f->error());
        }
        in.left -= 128;  // trailer records are not part of the hashed zones
        SessionTrailer t;
        const bool parsed = trailer_parse(tb, t);
        if (!is_candidate) {
            if (parsed && t.committed_bytes == trailer_pos && t.gen == gen) {
                prev_self = t.self_hash;
                prev_ok = true;
                out = t;  // its saved_tokens bound the next group's append boundaries
            } else {
                prev_ok = false;  // damaged or misplaced: the candidate's digest is the authority below
            }
            return true;
        }
        if (!parsed) return in.fail("the checkpoint record is corrupt");
        if (in.left != 0 || t.committed_bytes != trailer_pos)
            return in.fail("the checkpoint record does not sit at the end of the generation chain");
        if (t.gen != gen) return in.fail("generation numbering does not match the chain");
        if (t.saved_tokens != live.want_ids)
            return in.fail("saved_tokens does not match the live token list");
        if (framing_only) {  // no hashing here: the gate re-verifies the trailer's own stored resume state,
                             // and this cross-check catches a framing walk that lied about the group sizes
            if (t.hashed_bytes != (covered - kHeader) - 128 * gen)
                return in.fail("the checkpoint's hashed_bytes does not match the chain framing");
        } else {
            if (in.hash.digest() != t.digest) return in.fail("payload checksum mismatch");
            const SessionHasherState hs = in.hash.state();  // byte-exact resume-state honesty (drill weakness 6)
            if (hs.npending != t.npending || hs.total != t.hashed_bytes || std::memcmp(hs.lanes, t.lanes, 32) != 0 ||
                std::memcmp(hs.pending, t.pending, 32) != 0)
                return in.fail("the stored resume state does not match this file");
        }
        if (prev_ok) {
            if (t.chain != prev_self) return in.fail("the checkpoint chain is broken");
        } else if (gen == 1 && t.chain != genesis) {
            return in.fail("the first generation does not chain to the identity");
        }
        out = t;
        return true;
    }
};

bool read_v3(const std::string& path, const SessionFileIdentity& id, SavedConversation& image, size_t& bytes,
             std::string& error, const SessionReadLimits& limits, SessionStatus& st) {
    std::vector<SessionCheckpoint> cands;
    if (!session_file_checkpoints(path, id, cands, error)) {
        st.error = SessionError::invalid;
        return false;
    }
    std::string last;
    for (const auto& cand : cands) {  // newest first; a failing candidate demotes to the next older one
        const uint64_t covered = cand.trailer_offset + 128;
        error.clear();
        RawFile rf;  // fresh sequential descriptor per attempt (the validator and prior attempts used their own)
        uint64_t sz = 0;
        if (!rf.open_session(path, sz)) {
            st.error = rf.refused() ? SessionError::invalid : rf.kind();
            error = "session file: " + path + ": " + rf.error();
            return false;
        }
        FileSource f(rf);
        if (!f.ok()) { st.error = SessionError::memory; error = "session file: " + f.error(); return false; }
        if (limits.progress) f.progress.emplace(limits.progress);
        f.expected = covered;
        uint8_t h[kHeader];
        if (!f.read(h, kHeader)) { st.error = f.kind(); error = "session file: header read error: " + f.error(); return false; }
        SessionHeaderV3 hv;
        if (!header_parse_v3(h, id, hv, error)) { st.error = SessionError::invalid; return false; }
        if (limits.admit) {
            std::string why;
            if (!limits.admit(session_read_peak_bytes(covered), why)) {
                st.error = SessionError::memory;
                error = "session file: " + why;
                return false;
            }
        }
        SavedConversation parsed;
        In in{&f, SessionHasher(0), covered - kHeader, error, limits};
        V3Walk w{in, parsed, covered, trailer_genesis_chain(id)};
        SessionTrailer tr;
        uint64_t prev_self = 0;
        bool prev_ok = false, ok = false;
        for (uint64_t g = 1; g <= cand.trailer.gen; ++g) {
            const uint64_t left0 = in.left;  // the whole group: preamble (gen 1 only), mutable block, segments
            if (g == 1 && !w.preamble()) break;
            if (!w.group_body()) break;
            if (!w.close_group(g, left0 - in.left, g == cand.trailer.gen, tr, prev_self, prev_ok)) break;
            w.have_prev_saved = prev_ok;
            w.prev_saved = tr.saved_tokens;
            if (g == cand.trailer.gen) ok = true;
        }
        if (!ok) {
            last = error.empty() ? "the checkpoint chain is unreadable" : error;
            if (in.kind != SessionError::invalid) st.error = in.kind;
            continue;
        }
        parsed.live.ids = std::move(w.live.ids);
        parsed.live.imgs = std::move(w.live.imgs);
        for (size_t k = 0; k < parsed.live.stage_parts.size() && k < w.parts.size(); ++k) {
            parsed.live.stage_parts[k].ids = std::move(w.parts[k].ids);
            parsed.live.stage_parts[k].imgs = std::move(w.parts[k].imgs);
        }
        parsed.kv.resize(w.layers);
        for (size_t j = 0; j < w.layers; ++j) {
            ConversationKv& k = parsed.kv[j];
            k.format = (int) w.geom[j].format;
            k.cells = w.geom[j].cells; k.heads = w.geom[j].heads; k.head_dim = w.geom[j].head_dim;
            k.page_size = w.geom[j].page_size; k.pooled_rows = w.geom[j].pooled_rows; k.idx_dim = w.geom[j].idx_dim;
            k.k = std::move(w.kv[j][0]); k.v = std::move(w.kv[j][1]); k.k_scale = std::move(w.kv[j][2]);
            k.v_scale = std::move(w.kv[j][3]); k.pooled = std::move(w.kv[j][4]);
        }
        image = std::move(parsed);
        bytes = (size_t) covered;  // the trusted chain's coverage (garbage past it is never read)
        st.error = SessionError::none;
        return true;
    }
    error = "session file: " + path + ": no generation verified against its checkpoint digest" +
            (last.empty() ? std::string() : " (" + last + ")");
    if (st.error == SessionError::none) st.error = SessionError::invalid;
    return false;
}

bool read_impl(const std::string& path, const SessionFileIdentity& id, SavedConversation& image,
               size_t& bytes, std::string& error, const SessionReadLimits& limits, SessionStatus& st) {
    error.clear();
    st = {};
    st.error = SessionError::invalid;   // every refusal below, unless the step says otherwise
    RawFile raw;
    uint64_t size = 0;
    if (!raw.open_session(path, size)) {
        st.error = raw.refused() ? SessionError::invalid : raw.kind();
        error = "session file: " + path + ": " + raw.error();
        return false;
    }
    if (size < kHeader + kTrailer) { error = "session file: size " + std::to_string(size) + " is below the minimum"; return false; }
    const uint64_t max_size = session_read_max_file_bytes(limits);
    if (size > max_size || size > (uint64_t) std::numeric_limits<size_t>::max()) {
        error = "session file: size " + std::to_string(size) + " exceeds this engine's limit" +
                (max_size == UINT64_MAX ? std::string() : " of " + std::to_string(max_size));
        return false;
    }
    FileSource f(raw);
    if (!f.ok()) { st.error = SessionError::memory; error = "session file: " + f.error(); return false; }
    // installed before the first block (the header's) is read: every block reports, the first one included
    if (limits.progress) f.progress.emplace(limits.progress);
    f.expected = size;
    uint8_t h[kHeader];
    if (!f.read(h, kHeader)) { st.error = f.kind(); error = "session file: header read error: " + f.error(); return false; }
    uint32_t version = 0, hsize = 0;
    uint64_t model = 0, config = 0, payload = 0, r0 = 0, r1 = 0, hh = 0;
    std::memcpy(&version, h + 8, 4); std::memcpy(&hsize, h + 12, 4);
    std::memcpy(&model, h + 16, 8); std::memcpy(&config, h + 24, 8); std::memcpy(&payload, h + 32, 8);
    std::memcpy(&r0, h + 40, 8); std::memcpy(&r1, h + 48, 8); std::memcpy(&hh, h + 56, 8);
    if (std::memcmp(h, kMagic, 8) != 0) { error = "session file: not a Strata session file (magic)"; return false; }
    if (version == kVersion3) {
        uint64_t root32 = 0;
        std::memcpy(&root32, h + 32, 8);
        if (root32 == session_hash64(h, 32, 0))  // a genuine v3 header never reaches the v2 parse
            return read_v3(path, id, image, bytes, error, limits, st);
        // version 3 in a v2-shaped header: falls through to the "unsupported version" refusal below
    }
    if (hh != session_hash64(h, 56, 0)) { error = "session file: header checksum mismatch"; return false; }
    if (version != kVersion) { error = "session file: unsupported version " + std::to_string(version); return false; }
    if (hsize != kHeader || r0 || r1) { error = "session file: invalid header fields"; return false; }
    if (model != id.model) { error = "session file: saved with another model (model fingerprint differs)"; return false; }
    if (config != id.config) { error = "session file: saved with another engine configuration (config fingerprint differs)"; return false; }
    if (payload > size || size - kHeader - kTrailer != payload) {
        error = "session file: size " + std::to_string(size) + " does not match the header (" +
                std::to_string(kHeader + payload + kTrailer) + ")";
        return false;
    }
    if (limits.admit) {
        std::string why;
        if (!limits.admit(session_read_peak_bytes(size), why)) {
            st.error = SessionError::memory;
            error = "session file: " + why;
            return false;
        }
    }
    SavedConversation parsed;
    In in{&f, SessionHasher(0), payload, error, limits};
    if (!get_payload(in, parsed)) { st.error = in.kind; return false; }
    if (in.left) { error = "session file: payload has " + std::to_string(in.left) + " unparsed bytes"; return false; }
    uint8_t t[kTrailer];
    if (!f.read(t, kTrailer)) { st.error = f.kind(); error = "session file: trailer read error: " + f.error(); return false; }
    uint64_t ph = 0;
    std::memcpy(&ph, t, 8);
    if (ph != in.hash.digest()) { error = "session file: payload checksum mismatch"; return false; }
    if (std::memcmp(t + 8, kEnd, 8) != 0) { error = "session file: bad end marker"; return false; }
    image = std::move(parsed);
    bytes = (size_t) size;
    st.error = SessionError::none;
    return true;
}

// ---- append path: the save-side gate plan and the commit-disciplined group writer ----------------
//
// Commit discipline (spec §Commit discipline, drill S3's fault-hook map): gate re-verify -> header
// anchor at the new group's start (degrading) -> streamed group (mutable block fresh, ids/imgs and
// K/V parts suffix-only) -> hashed pad -> 128 B trailer -> fsync (the commit point) -> header
// advance (non-fatal) -> folder flush (dir_flush honesty).  No temp, no rename, and never a
// truncation or rollback of a torn tail: the validator and reader cascade over it, and the gate
// declines appends until a full rewrite re-lays the file.

bool append_plan_impl(const std::string& path, const SessionFileIdentity& id, SessionAppendPlan& out,
                      std::string& error, const SessionReadLimits& limits) {
    std::vector<SessionCheckpoint> cands;
    if (!session_file_checkpoints(path, id, cands, error)) {
        if (error.empty()) error = "session file: " + path + ": append refused (no validated checkpoint)";
        return false;
    }
    // The gate walk reads bounded framing only - but it obeys the RUNTIME's own limits (review finding
    // M3): counts are capped before any allocation, and the framing mode skips array bytes it never
    // needs.  A hostile file cannot make the gate allocate past what this engine could ever restore.
    std::string last;
    for (const auto& cand : cands) {  // newest first; a failing candidate demotes exactly like the reader's
        error.clear();
        RawFile rf;
        uint64_t sz = 0;
        if (!rf.open_session(path, sz)) {
            error = "session file: " + path + ": " + rf.error();
            return false;  // the file vanished or turned un-openable under the validator: refuse, never guess
        }
        rf.force_buffered();  // small reads alternating with seeks: O_DIRECT alignment rules refuse that mix
        FileSource f(rf);
        if (!f.ok()) { error = "session file: " + f.error(); return false; }
        uint8_t h[kHeader];
        if (!f.read(h, kHeader)) { error = "session file: header read error: " + f.error(); return false; }
        SessionHeaderV3 hv;
        if (!header_parse_v3(h, id, hv, error)) return false;
        const uint64_t covered = cand.trailer_offset + 128;
        SavedConversation probe;  // only the preamble fields matter here; segment zones are seeked past
        In in{&f, SessionHasher(0), covered - kHeader, error, limits};
        V3Walk w{in, probe, covered, trailer_genesis_chain(id)};
        w.framing_only = true;
        SessionTrailer tr;
        uint64_t prev_self = 0;
        bool prev_ok = false, ok = false;
        for (uint64_t g = 1; g <= cand.trailer.gen; ++g) {
            const uint64_t left0 = in.left;
            if (g == 1 && !w.preamble()) break;
            if (!w.group_body()) break;
            if (!w.close_group(g, left0 - in.left, g == cand.trailer.gen, tr, prev_self, prev_ok)) break;
            w.have_prev_saved = prev_ok;
            w.prev_saved = tr.saved_tokens;
            if (g == cand.trailer.gen) ok = true;
        }
        if (!ok) { last = error.empty() ? "the checkpoint chain is unreadable" : error; continue; }
        // resume-state self-verification: the stored state must reproduce the stored digest, no file bytes
        SessionHasherState hs{};
        for (int i = 0; i < 4; ++i) hs.lanes[i] = tr.lanes[i];
        std::memcpy(hs.pending, tr.pending, 32);
        hs.npending = tr.npending;
        hs.total = tr.hashed_bytes;
        SessionHasher hh(0);
        hh.restore(hs);
        if (hh.digest() != tr.digest) { last = "the stored resume state does not reproduce this checkpoint's digest"; continue; }
        // the EOF rule: a torn tail is rewritten in full before appending again
        if (sz != covered) {
            last = "the file has " + std::to_string(sz - covered) +
                   " bytes past its newest checkpoint record (a torn tail is rewritten in full before appending)";
            continue;
        }
        // the engine's append boundaries for the next group, recomputed from this generation's own framing
        w.have_prev_saved = true;
        w.prev_saved = tr.saved_tokens;  // geom[] now holds this generation's geometry, last_to[] its part sizes
        std::vector<std::array<uint64_t, 5>> bounds(w.layers);
        bool bad = false;
        for (size_t j = 0; j < w.layers && !bad; ++j)
            for (size_t p = 0; p < 5; ++p) {
                const uint64_t b = w.committed_boundary(j, p, w.last_to[j][p]);
                if (b == UINT64_MAX) {
                    last = "layer " + std::to_string(j + 1) + " part " + std::to_string(p + 1) +
                           " has no recomputable committed boundary (a recorded idx_block of 0 or a non-uniform "
                           "part): its prefix is not append-trusted";
                    bad = true;
                    break;
                }
                if (b > w.last_to[j][p]) { last = "the file's recorded sizes fall below the engine's committed boundary"; bad = true; break; }
                bounds[j][p] = b;
            }
        if (bad) continue;
        out.checkpoint = cand;
        out.group_start = covered;
        out.prev_ids = w.live.want_ids;
        out.prev_imgs = w.live.want_imgs;
        out.prev_parts.clear();
        for (const auto& pa : w.parts) out.prev_parts.push_back({pa.want_ids, pa.want_imgs});
        out.stage_table = probe.stage_table;
        out.geometry = probe.geometry;
        out.layer_lo = probe.layer_lo;
        out.layer_hi = probe.layer_hi;
        out.prev_geom.clear();
        for (size_t j = 0; j < w.layers; ++j) {
            const auto& g = w.geom[j];
            out.prev_geom.push_back({g.format, g.cells, g.heads, g.head_dim, g.page_size, g.pooled_rows, g.idx_dim,
                                     g.idx_block});
        }
        out.prev_bounds = std::move(bounds);
        out.hasher = SessionHasher(0);
        out.hasher.restore(hs);  // append continues this digest without re-hashing any committed byte
        error.clear();
        return true;
    }
    error = "session file: " + path + ": append refused: " +
            (last.empty() ? std::string("no checkpoint chain validated") : last);
    return false;
}

bool append_impl(const std::string& path, const SavedConversation& meta, const std::vector<SessionKvSource>* sources,
                 const SessionFileIdentity& id, const SessionAppendPlan& plan, size_t& bytes, std::string& error,
                 const SessionWriteOptions& opt, SessionStatus& st) {
    st = {};
    st.error = SessionError::invalid;   // every refusal below happens before any disk access
    std::string bad;
    if (!stage_parts_consistent(meta, bad)) { error = "session file: " + bad; return false; }
    if (!meta.kv.empty()) { error = "session file: append takes the K/V as sources only"; return false; }
    if (!meta.stage_images.empty()) { error = "session file: append cannot write a staged multi-image session"; return false; }

    // The gate: never trust a stale plan.  The validator scan runs unconditionally (its job: candidate
    // trust, max-gen resolution and stale-header self-heal); the re-verification of the ANCHORED
    // checkpoint is the cheap one (review finding M2): the 128 B record at the plan's own offset must
    // parse, be validator-trusted, and be the newest trusted checkpoint.  The record's self_hash pins
    // every byte of it, so a matching parsed trailer IS the exact checkpoint the plan was derived
    // from - the plan's framing (geometry, windows, counts) stands for these bytes.  The file-size and
    // group_start cross-checks below (against the open fd) enforce the EOF rule: an append never starts
    // below committed bytes and never past a torn tail.  The full append_plan_impl (with its framing
    // walk) remains what session_file_append_plan produces and what session_save_gate consumes.
    std::vector<SessionCheckpoint> cands;
    if (!session_file_checkpoints(path, id, cands, error)) {
        if (error.empty()) error = "session file: " + path + ": append refused (no validated checkpoint)";
        return false;
    }
    const SessionCheckpoint* anchored = nullptr;
    for (const auto& cc : cands)
        if (cc.trailer_offset == plan.checkpoint.trailer_offset &&
            cc.trailer.self_hash == plan.checkpoint.trailer.self_hash && cc.file_size == plan.checkpoint.file_size) {
            anchored = &cc;
            break;
        }
    if (!anchored || anchored->trailer_offset != cands.front().trailer_offset ||
        anchored->trailer.gen != cands.front().trailer.gen) {
        error = "session file: the append plan no longer matches " + path +
                " (it changed since the plan was made: the newest validated checkpoint differs)";
        return false;
    }
    // resume-state self-verification, re-derived from the anchored record (never taken from the plan):
    // the stored state must reproduce the stored digest, no file bytes needed
    SessionHasherState hs{};
    for (int i = 0; i < 4; ++i) hs.lanes[i] = anchored->trailer.lanes[i];
    std::memcpy(hs.pending, anchored->trailer.pending, 32);
    hs.npending = anchored->trailer.npending;
    hs.total = anchored->trailer.hashed_bytes;
    {
        SessionHasher hh(0);
        hh.restore(hs);
        if (hh.digest() != anchored->trailer.digest) {
            error = "session file: " + path +
                    ": append refused: the stored resume state does not reproduce this checkpoint's digest";
            return false;
        }
    }
    const SessionAppendPlan& fresh = plan;  // re-verified against the file by the anchored trailer above
    if (meta.geometry != fresh.geometry) { error = "session file: append with a different model geometry than this file's"; return false; }
    if (meta.layer_lo != fresh.layer_lo || meta.layer_hi != fresh.layer_hi) { error = "session file: append with a different layer range than this file's"; return false; }
    if (meta.stage_table != fresh.stage_table) { error = "session file: append with a different stage table (only a full write records that)"; return false; }
    if (meta.live.ids.size() < fresh.prev_ids || meta.live.imgs.size() < fresh.prev_imgs) {
        error = "session file: the live state shrank below what this file committed (an append only extends it)";
        return false;
    }
    if (meta.live.stage_parts.size() != fresh.prev_parts.size()) { error = "session file: the stage parts changed (only a full write records that)"; return false; }
    for (size_t k = 0; k < meta.live.stage_parts.size(); ++k) {
        const auto& part = meta.live.stage_parts[k];
        if (part.ids.size() < fresh.prev_parts[k][0] || part.imgs.size() < fresh.prev_parts[k][1]) {
            error = "session file: a stage part shrank below what this file committed"; return false;
        }
    }
    if (sources->size() != fresh.prev_geom.size()) { error = "session file: the K/V source count does not match the file's layer count"; return false; }
    for (size_t j = 0; j < sources->size(); ++j) {
        const auto& k = (*sources)[j];
        const auto& g = fresh.prev_geom[j];
        if (k.format != g.format || k.heads != g.heads || k.head_dim != g.head_dim || k.page_size != g.page_size ||
            k.idx_dim != g.idx_dim || k.idx_block != g.idx_block) {
            error = "session file: layer " + std::to_string(j + 1) + " K/V geometry changed (only a full write records that)";
            return false;
        }
        if (k.cells < g.cells || k.pooled_rows < g.pooled_rows) { error = "session file: a K/V carve shrank (only a full write records that)"; return false; }
        for (size_t p = 0; p < 5; ++p)
            if (k.sizes[p] < fresh.prev_bounds[j][p]) { error = "session file: a K/V part shrank below its committed boundary"; return false; }
    }

    std::vector<std::array<uint64_t, 5>> froms = fresh.prev_bounds;
    V3LiveFrom live_from;
    live_from.ids = fresh.prev_ids;
    live_from.imgs = fresh.prev_imgs;
    for (const auto& pp : fresh.prev_parts) { V3LiveFrom q; q.ids = pp[0]; q.imgs = pp[1]; live_from.parts.push_back(std::move(q)); }

    Out sizing;
    put_group_v3(sizing, meta, sources, &froms, &live_from, false);
    const uint64_t group = sizing.n;
    const uint64_t group_start = fresh.group_start;
    const uint64_t pad = (512 - (group + 128) % 512) % 512;
    const uint64_t delta = group + pad + 128;
    if (sizing.overflow || group > UINT64_MAX - group_start - 512 ||
        group_start + delta > (uint64_t) std::numeric_limits<size_t>::max()) {
        error = "session file: the appended state is too large for this build";
        return false;
    }
    st.error = SessionError::io;   // every failure from here on is disk-side
    if (opt.min_free_bytes) {
        uint64_t avail = 0;
        if (!free_space(path, avail)) { error = "session file: cannot read the free disk space for " + path; st.error = SessionError::storage; return false; }
        if (avail < delta || avail - delta < opt.min_free_bytes) {
            st.error = SessionError::storage;
            error = "session file: not enough disk space (" + std::to_string(delta >> 20) + " MiB plus a reserve of " +
                    std::to_string(opt.min_free_bytes >> 20) + " MiB needed, " + std::to_string(avail >> 20) +
                    " MiB free)";
            return false;
        }
    }
    const std::function<int(const char*)>* fault = opt.fault ? &opt.fault : nullptr;

    RawFile raw;
    uint64_t size = 0;
    if (!raw.open_append(path, size)) {
        st.error = raw.refused() ? SessionError::invalid : raw.kind();
        error = "session file: " + path + ": " + raw.error();
        return false;
    }
    if (size != fresh.checkpoint.file_size || size != group_start) {
        error = "session file: " + path + " changed underneath the append gate";
        return false;
    }
    // seam "append_verify": the last refusal point.  Nothing has been written yet: refusal leaves the
    // file byte-identical, and the append path has no temp to clean up.
    if (const int e = injected(fault, "append_verify")) {
        error = "session file: " + path + ": append refused at append_verify: " + std::strerror(e);
        st.error = errno_kind(e);
        return false;
    }

    // seam "append_anchor": point the header fast pair at the new group's start BEFORE writing it, so a
    // crash in the torn window is found by the fast path.  Degrading, never fatal: scan-back covers a stale pair.
    const uint64_t committed_prev = fresh.checkpoint.trailer.committed_bytes;
    uint8_t h[kHeader];
    header_bytes_v3(h, id, {committed_prev, committed_prev, group_start});
    if (!injected(fault, "append_anchor")) (void) (raw.write_at(0, h, kHeader) && (!opt.durable || raw.sync()));

    FileSink f(raw);
    if (!f.ok()) { st.error = SessionError::memory; error = "session file: " + f.error(); return false; }
    f.keep_open = true;   // the commit fsync happens in finish(); the post-commit header update needs this fd
    if (opt.progress) f.progress.emplace(opt.progress);
    if (opt.block_span) f.block_span.emplace(opt.block_span);
    f.phase = opt.phase;
    f.expected = delta;
    if (f.progress) (*f.progress)(0, delta);   // the descriptor is open and gated: the transfer starts
    f.write_after_fault = injected(fault, "append_write");    // torn-tail seam: it fires only after a block landed
    f.flush_fault = injected(fault, "trailer_flush");         // the post-trailer commit fsync
    f.writer_thread_fault = injected(fault, "writer_thread"); // starvation fallback (an optimization loss)

    SessionHasher hash(0);
    hash.restore(hs);  // byte-exact continuation of the VALIDATED trailer state (re-derived, never plan-supplied)
    Out out;
    out.f = &f;
    out.hash = &hash;
    put_group_v3(out, meta, sources, &froms, &live_from, false);
    bool ok = out.ok && out.n == group;
    if (ok) { static const uint8_t kZeros[512] = {}; out.raw(kZeros, pad); }   // the pad is hashed; the trailer is not
    const int e_trailer = injected(fault, "trailer_write");
    if (ok && !e_trailer) {
        SessionTrailer tr;
        tr.gen = fresh.checkpoint.trailer.gen + 1;
        tr.committed_bytes = group_start + group + pad;
        tr.saved_tokens = (uint64_t) meta.live.ids.size();
        tr.digest = hash.digest();
        tr.chain = fresh.checkpoint.trailer.self_hash;
        const SessionHasherState hs = hash.state();
        for (int i = 0; i < 4; ++i) tr.lanes[i] = hs.lanes[i];
        std::memcpy(tr.pending, hs.pending, 32);
        tr.npending = hs.npending;
        tr.hashed_bytes = hs.total;
        uint8_t tb[128];
        trailer_bytes(tr, tb);
        ok = f.write(tb, 128);
    }
    // finish(): flushes the remaining blocks (seam "trailer_flush" fires at its fsync) - the commit point
    const bool torn_trailer = e_trailer != 0;
    const bool fin = f.finish(opt.durable);
    if (!ok || torn_trailer || !fin) {
        // the seam refused the trailer before handing it to the sink: the errno is the seam's own
        std::string why = torn_trailer ? std::string(std::strerror(e_trailer))
                                       : out.read_failed ? "K/V source read" + (out.read_error.empty() ? std::string() : ": " + out.read_error)
                                                         : f.error();
        st.error = torn_trailer ? errno_kind(e_trailer)
                                : out.read_failed ? SessionError::io : f.kind();
        error = "session file: appending to " + path + " failed" + (why.empty() ? std::string() : ": " + why) +
                " (whatever landed stands as a torn tail: the previous checkpoint is intact, and the gate declines appends until a full rewrite)";
        return false;   // no truncation, no rollback, no temp: the torn tail is the design
    }

    // seam "header_update": the commit is already durable, so advancing the header is non-fatal by design -
    // a stale fast pair self-heals at the next save's anchor (drill scenario I).
    const uint64_t committed_new = group_start + group + pad;
    header_bytes_v3(h, id, {committed_new, committed_new, committed_new + 128});
    if (!injected(fault, "header_update")) (void) (raw.write_at(0, h, kHeader) && (!opt.durable || raw.sync()));

    // seam "append_dir_flush": the folder flush, same honesty semantics as a full save's "dir_flush"
    Publisher pub;
    std::string perr;
    if (!pub.open(path, perr)) {
        st.published = true;   // the append itself is committed and flushed
        st.error = pub.kind();
        error = "session file: appending to " + path + " committed, but " + perr;
        return false;
    }
    if (!pub.dir_flush(perr, opt.durable, st, fault, "append_dir_flush")) {
        st.published = true;   // the file was flushed; only the folder flush is lost
        error = "session file: appending to " + path + " committed, but " + perr + " (the appended bytes are flushed; the folder entry may not survive a power loss)";
        return false;
    }
    if (!raw.close()) {
        st.published = true;   // the commit was already fsynced: a close failure cannot undo it
        st.error = raw.kind();
        error = "session file: closing " + path + " after the append failed: " + raw.error();
        return false;
    }
    st.error = SessionError::none;
    bytes = (size_t) delta;
    return true;
}
} // namespace

bool session_file_append_plan(const std::string& path, const SessionFileIdentity& id, SessionAppendPlan& plan,
                              std::string& error, const SessionReadLimits& limits) {
    try {
        return append_plan_impl(path, id, plan, error, limits);
    } catch (const std::bad_alloc&) {
        error = "session file: out of memory while checking " + path + " for an append";
        return false;
    }
}

bool session_file_append(const std::string& path, const SavedConversation& meta, const std::vector<SessionKvSource>& kv,
                         const SessionFileIdentity& id, const SessionAppendPlan& plan, size_t& bytes, std::string& error,
                         const SessionWriteOptions& options, SessionStatus* status) {
    SessionStatus st;
    bool ok = false;
    try {
        ok = append_impl(path, meta, &kv, id, plan, bytes, error, options, st);
    } catch (const std::bad_alloc&) {
        error = "session file: out of memory while appending to " + path;
        st.error = SessionError::memory;
        ok = false;
    }
    if (status) *status = st;
    return ok;
}

SessionSaveChoice session_save_gate(const SessionAppendPlan* plan, const std::vector<SessionKvSource>& live,
                                    int64_t kv_first_dirty, int64_t page, std::string& why) {
    why.clear();
    const auto full = [&](std::string r) { why = std::move(r); return SessionSaveChoice::full_rewrite; };
    if (!plan) return full("no verified checkpoint chain for this file (v2 file, torn tail, or foreign identity)");
    if (live.size() != plan->prev_geom.size())
        return full("the K/V layer count differs from the file's");
    for (size_t j = 0; j < live.size(); ++j) {
        const auto& s = live[j];
        const auto& g = plan->prev_geom[j];
        if (s.format != g.format || s.heads != g.heads || s.head_dim != g.head_dim || s.page_size != g.page_size ||
            s.idx_dim != g.idx_dim || s.idx_block != g.idx_block)
            return full("layer " + std::to_string(j + 1) + " K/V geometry differs from the file's (only a full "
                        "write records that)");
        if (s.cells < g.cells || s.pooled_rows < g.pooled_rows)
            return full("layer " + std::to_string(j + 1) + " carve shrank below the file's (an append only grows it)");
    }
    // the immutability watermark (SessionState::kv_append_ok's comparison, kept header-free here): the
    // whole-page prefix of the lowest rewritten position must reach the boundary the file committed.
    const int64_t saved = (int64_t) plan->checkpoint.trailer.saved_tokens;
    const int64_t whole = page > 0 ? (kv_first_dirty / page) * page : kv_first_dirty;
    if (kv_first_dirty != std::numeric_limits<int64_t>::max() && whole < saved)
        return full("pool rows at or below the file's saved boundary may differ from what it committed");
    return SessionSaveChoice::append;
}

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

} // namespace strata::core
